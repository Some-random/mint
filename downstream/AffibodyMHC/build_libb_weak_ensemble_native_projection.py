#!/usr/bin/env python
"""Fit LibB weak-label ensembles using the native-projection structure readouts.

The additive and frozen-MINT out-of-fold (OOF) scores are reused from the
previous weak-only artifact because those two models did not change.  The StaB
and RDE columns are replaced with scores from the clean readouts that learn a
direct native-dimension -> 64 -> 32 projection.  All four columns are aligned
on the same 10,181 identity-cold weak-validation rows.

The nonnegative logistic stacker, its L2 grid, cross-fitting, and selection
criterion are exactly those in :mod:`build_libb_weak_ensemble`.  This command
does not accept or read retention outcomes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import build_libb_weak_ensemble as base
from downstream.AffibodyMHC.code_only_baseline import sha256_file


SCHEMA_VERSION = "libb-weak-oof-ensemble-native-projection-all4-v1"
STAB_MEMBER = "stab_designed_ordered_native_projection"
RDE_MEMBER = "rde_network_designed_3fold_native_projection"
ALL_MEMBERS = ("additive_7site", "mint_layer5", STAB_MEMBER, RDE_MEMBER)
RDE_HEADS = (
    "rde_network_fold0_designed_ordered",
    "rde_network_fold1_designed_ordered",
    "rde_network_fold2_designed_ordered",
)

DEFAULT_BASE_OOF = (
    REPO_ROOT
    / "private_data/experiments/libb_weak_ensemble_selection_v1/"
    "matched_oof_predictions.csv.gz"
)
DEFAULT_BASE_MANIFEST = DEFAULT_BASE_OOF.with_name("manifest.json")
DEFAULT_STAB_OOF = (
    REPO_ROOT
    / "private_data/experiments/stab_libb_native_projection_120_full_v1/"
    "weak_validation_predictions.csv.gz"
)
DEFAULT_STAB_AUDIT = DEFAULT_STAB_OOF.with_name("audit.json")
DEFAULT_RDE_OOF = (
    REPO_ROOT
    / "private_data/experiments/rde_libb_native_projection_120_full_v1/"
    "weak_validation_predictions.csv.gz"
)
DEFAULT_RDE_AUDIT = DEFAULT_RDE_OOF.with_name("audit.json")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "private_data/experiments/libb_weak_ensemble_native_projection_all4_v1"
)


def _load_native_audit(path: Path, expected_model: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    base._require(
        audit.get("readout_mode") == "native_learned_projection",
        f"{expected_model} is not a native learned projection",
    )
    base._require(audit.get("retention_labels_read") is False, f"{expected_model} read retention")
    base._require(audit.get("weak_rows") == base.EXPECTED_ROWS, f"{expected_model} row count changed")
    return audit


def _load_one_oof(path: Path, source_model: str, output_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": str})
    required = {"model", "fold", "row_id", "weak_label", "probability"}
    base._require(required.issubset(frame.columns), f"OOF schema changed: {path}")
    key = ["row_id", "fold", "weak_label"]
    chosen = frame.loc[frame["model"].eq(source_model), key + ["probability"]].copy()
    base._require(len(chosen) == base.EXPECTED_OOF_ROWS, f"{source_model} OOF rows changed")
    base._require(not bool(chosen["row_id"].duplicated().any()), f"duplicate {source_model} rows")
    chosen = chosen.rename(columns={"probability": output_name})
    values = chosen[output_name].to_numpy(dtype=float)
    base._require(bool(np.isfinite(values).all()), f"{source_model} has non-finite values")
    base._require(bool(((values >= 0.0) & (values <= 1.0)).all()), f"{source_model} outside [0,1]")
    return chosen


def _load_rde_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": str})
    required = {"model", "fold", "row_id", "weak_label", "probability"}
    base._require(required.issubset(frame.columns), f"RDE OOF schema changed: {path}")
    key = ["row_id", "fold", "weak_label"]
    blocks = []
    for head in RDE_HEADS:
        chosen = frame.loc[frame["model"].eq(head), key + ["probability"]].copy()
        base._require(len(chosen) == base.EXPECTED_OOF_ROWS, f"{head} OOF rows changed")
        base._require(not bool(chosen["row_id"].duplicated().any()), f"duplicate {head} rows")
        blocks.append(chosen.rename(columns={"probability": head}))
    aligned = blocks[0]
    for block in blocks[1:]:
        aligned = aligned.merge(block, on=key, validate="one_to_one")
    aligned[RDE_MEMBER] = base.mean_logit_score([aligned[head] for head in RDE_HEADS])
    return aligned[key + [RDE_MEMBER]]


def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    first_rank = pd.Series(np.asarray(first, dtype=float)).rank(method="average").to_numpy()
    second_rank = pd.Series(np.asarray(second, dtype=float)).rank(method="average").to_numpy()
    if np.ptp(first_rank) == 0.0 or np.ptp(second_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def macro_weak_label_spearman(frame: pd.DataFrame, score_name: str) -> tuple[float, int]:
    values = []
    for _, group in frame.groupby("peptide_id", sort=True):
        value = _spearman(group[score_name], group["weak_label"])
        if math.isfinite(value):
            values.append(value)
    base._require(values, "no weak-label peptide supports Spearman")
    return float(np.mean(values)), int(len(values))


def render_report(
    selection: Mapping[str, object], metrics: pd.DataFrame, final_stack: Mapping[str, object]
) -> str:
    lines = [
        "# LibB native-projection weak-label ensemble",
        "",
        "The additive and frozen-MINT scores are unchanged. The StaB and RDE scores come ",
        "from the cleaner readouts that learn directly from their native feature dimensions, ",
        "without the former fixed expansion to 1,280 values. All results below use the same ",
        "10,181 identity-cold weak-validation pairs. Retention outcomes were not read.",
        "",
        "| Candidate | Members | Within-peptide AP | Within-peptide AUROC | Weak-label Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.sort_values("within_peptide_ap", ascending=False).itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.members} | {row.within_peptide_ap:.6f} | "
            f"{row.within_peptide_auroc:.6f} | {row.within_peptide_weak_label_spearman:.6f} |"
        )
    lines.extend(
        [
            "",
            "The Spearman column ranks a binary weak label, not numerical retention. It is ",
            "included only as a secondary description and was not used to choose a model.",
            "",
            f"Best individual model by weak-label within-peptide AP: `{selection['best_candidate']['candidate']}`.",
            "",
            f"Best multi-model mean-logit candidate: `{selection['best_ensemble_candidate']['candidate']}`.",
            "",
            f"Final four-model stacker L2: `{final_stack['l2']}`.",
            "",
            "Final nonnegative standardized-logit weights: `{} `.".format(
                ", ".join(
                    f"{member}={coefficient:.8g}"
                    for member, coefficient in zip(ALL_MEMBERS, final_stack["coefficient"])
                )
            ),
            "",
            "The stacker weights and comparisons in this file were calculated from weak labels ",
            "only. The retention panel had already been examined elsewhere in the project, so a ",
            "later retention comparison is retrospective rather than a fresh prospective test.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    paths = {
        "base_oof": Path(args.base_oof).resolve(),
        "base_manifest": Path(args.base_manifest).resolve(),
        "stab_oof": Path(args.stab_oof).resolve(),
        "stab_audit": Path(args.stab_audit).resolve(),
        "rde_oof": Path(args.rde_oof).resolve(),
        "rde_audit": Path(args.rde_audit).resolve(),
    }
    for path in paths.values():
        base._require(path.is_file(), f"missing input {path}")
    output = Path(args.output_dir).resolve()
    base._require(not output.exists(), f"output exists: {output}")
    base._require("private_data" in output.parts, "output must remain under private_data")

    with paths["base_manifest"].open("r", encoding="utf-8") as handle:
        base_manifest = json.load(handle)
    base._require(base_manifest.get("retention_labels_read") is False, "base scores read retention")
    stab_audit = _load_native_audit(paths["stab_audit"], STAB_MEMBER)
    rde_audit = _load_native_audit(paths["rde_audit"], RDE_MEMBER)

    old_frame = pd.read_csv(paths["base_oof"], dtype={"row_id": str})
    base_columns = [
        "row_id", "fold", "weak_label", "peptide_id", "affibody_id",
        "additive_7site", "mint_layer5",
    ]
    base._require(set(base_columns).issubset(old_frame.columns), "base OOF columns changed")
    old_frame = old_frame[base_columns].copy()
    stab = _load_one_oof(paths["stab_oof"], "stab_designed_ordered", STAB_MEMBER)
    rde = _load_rde_oof(paths["rde_oof"])
    key = ["row_id", "fold", "weak_label"]
    frame = old_frame.merge(stab, on=key, validate="one_to_one")
    frame = frame.merge(rde, on=key, validate="one_to_one")
    frame = frame.sort_values("row_id", kind="stable").reset_index(drop=True)
    base._require(len(frame) == base.EXPECTED_OOF_ROWS, "common OOF row count changed")
    base._require(int(frame["weak_label"].sum()) == base.EXPECTED_OOF_POSITIVE, "positive count changed")

    original_members = base.MEMBERS
    try:
        base.MEMBERS = ALL_MEMBERS
        metrics, outer_audit, final_stack, final_grid = base.evaluate_candidates(frame)
    finally:
        base.MEMBERS = original_members

    # The reused helper's historical column name says all3 even when it is
    # parameterized with four members; rename it explicitly in this artifact.
    old_stack = "cross_fitted_nonnegative_stack__all3"
    new_stack = "cross_fitted_nonnegative_stack__all4_native_projection"
    base._require(old_stack in frame.columns, "cross-fitted stacker output changed")
    frame = frame.rename(columns={old_stack: new_stack})
    metrics["candidate"] = metrics["candidate"].replace({old_stack: new_stack})

    spearman = []
    evaluable = []
    for candidate in metrics["candidate"]:
        value, count = macro_weak_label_spearman(frame, str(candidate))
        spearman.append(value)
        evaluable.append(count)
    metrics["within_peptide_weak_label_spearman"] = spearman
    metrics["within_peptide_weak_label_spearman_evaluable"] = evaluable
    selection = base.select_candidates(metrics)

    output.mkdir(parents=True, mode=0o700)
    prediction_columns = [
        "row_id", "fold", "weak_label", "peptide_id", "affibody_id", *ALL_MEMBERS,
        *metrics["candidate"].tolist(),
    ]
    prediction_path = output / "matched_oof_predictions.csv.gz"
    base._write_csv(prediction_path, frame[prediction_columns], compression="gzip")
    metric_path = output / "candidate_metrics.csv"
    base._write_csv(metric_path, metrics.sort_values("candidate").reset_index(drop=True))
    outer_path = output / "stacker_crossfit_audit.json"
    base._write_json(outer_path, outer_audit)
    grid_path = output / "stacker_final_l2_grid.json"
    base._write_json(grid_path, final_grid)
    stack_path = output / "stacker_locked_parameters.json"
    base._write_json(
        stack_path,
        {
            "schema_version": SCHEMA_VERSION,
            "members_in_coefficient_order": list(ALL_MEMBERS),
            "retention_labels_read": False,
            **final_stack,
        },
    )
    selection_path = output / "locked_weak_selection.json"
    base._write_json(
        selection_path,
        {"schema_version": SCHEMA_VERSION, "retention_labels_read": False, **selection},
    )
    report_path = output / "report.md"
    report_path.write_text(render_report(selection, metrics, final_stack), encoding="utf-8")
    os.chmod(report_path, 0o600)

    outputs = [
        prediction_path, metric_path, outer_path, grid_path, stack_path, selection_path,
        report_path,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "retention_labels_read": False,
        "common_oof_rows": base.EXPECTED_OOF_ROWS,
        "common_oof_positive": base.EXPECTED_OOF_POSITIVE,
        "members": list(ALL_MEMBERS),
        "readout_modes": {
            STAB_MEMBER: stab_audit["readout_mode"],
            RDE_MEMBER: rde_audit["readout_mode"],
        },
        "rde_definition": {"method": "mean_logit", "heads": list(RDE_HEADS)},
        "selection": selection,
        "stacker": {
            "input": "clipped component logits standardized on stacker-training rows only",
            "coefficient_bounds": "non-negative",
            "training_loss": "peptide-equal-weighted logistic loss plus L2",
            "l2_grid": list(base.STACK_L2_GRID),
            "selection_metric": "macro within-peptide AP on weak labels",
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        },
    }
    base._write_json(output / "manifest.json", manifest)
    print(metrics.sort_values("within_peptide_ap", ascending=False).to_string(index=False))
    print(json.dumps({"selection": selection, "final_stack": final_stack}, indent=2, default=float))
    print(f"wrote {output}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", default=str(DEFAULT_BASE_OOF))
    parser.add_argument("--base-manifest", default=str(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--stab-oof", default=str(DEFAULT_STAB_OOF))
    parser.add_argument("--stab-audit", default=str(DEFAULT_STAB_AUDIT))
    parser.add_argument("--rde-oof", default=str(DEFAULT_RDE_OOF))
    parser.add_argument("--rde-audit", default=str(DEFAULT_RDE_AUDIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
