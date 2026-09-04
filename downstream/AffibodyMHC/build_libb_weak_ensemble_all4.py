#!/usr/bin/env python
"""Extend the locked weak-only LibB ensemble analysis with the selected RDE arm.

This consumes the completed three-member OOF artifact rather than refitting its
components.  The fourth score is the prespecified mean-logit ensemble of the
three RDE-Network designed-site heads selected in the structural weak-label
audit.  No retention input is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import build_libb_weak_ensemble as base
from downstream.AffibodyMHC.code_only_baseline import sha256_file


SCHEMA_VERSION = "libb-weak-oof-ensemble-all4-v1"
RDE_MEMBER = "rde_network_designed_3fold"
ALL_MEMBERS = (*base.MEMBERS, RDE_MEMBER)
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
DEFAULT_RDE_OOF = (
    REPO_ROOT
    / "private_data/experiments/rde_libb_provider_revision_120_pilot_readouts_v2_common_env/"
    "rde_network_designed/weak_validation_predictions.csv.gz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "private_data/experiments/libb_weak_ensemble_selection_all4_v2"
)


def load_rde_mean_logit_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": str})
    base._require(
        {"model", "fold", "row_id", "weak_label", "probability"}.issubset(frame.columns),
        "RDE OOF schema changed",
    )
    blocks = []
    key = ["row_id", "fold", "weak_label"]
    for head in RDE_HEADS:
        current = frame.loc[frame["model"].eq(head), key + ["probability"]].copy()
        base._require(len(current) == base.EXPECTED_OOF_ROWS, f"RDE head {head} row count changed")
        base._require(not bool(current["row_id"].duplicated().any()), f"duplicate rows for {head}")
        current = current.rename(columns={"probability": head})
        blocks.append(current)
    aligned = blocks[0]
    for block in blocks[1:]:
        aligned = aligned.merge(block, on=key, validate="one_to_one")
    base._require(len(aligned) == base.EXPECTED_OOF_ROWS, "common RDE OOF rows changed")
    aligned[RDE_MEMBER] = base.mean_logit_score([aligned[head] for head in RDE_HEADS])
    return aligned[key + [RDE_MEMBER]]


def render_report(
    selection: Mapping[str, object], metrics: pd.DataFrame, final_stack: Mapping[str, object]
) -> str:
    lines = [
        "# LibB four-member retention-blind weak-label ensemble selection",
        "",
        "This extends the three-member analysis with the locked RDE-Network designed-site ",
        "representation. Its three independently pretrained RDE heads are first combined by ",
        "mean logit, exactly as in the structural weak-label selection. Every component is ",
        "aligned to the same 10,181 double-cold OOF rows. No retention labels were read.",
        "",
        "The primary metric is average precision calculated within each weak-data peptide ",
        "containing both label classes and then averaged equally across those peptides.",
        "",
        "| Candidate | Members | Within-peptide AP | Within-peptide AUROC | Pooled log loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.sort_values("within_peptide_ap", ascending=False).itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.members} | {row.within_peptide_ap:.6f} | "
            f"{row.within_peptide_auroc:.6f} | {row.pooled_log_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Locked weak-only decision",
            "",
            f"Best candidate: `{selection['best_candidate']['candidate']}`.",
            "",
            f"Best multi-member candidate: `{selection['best_ensemble_candidate']['candidate']}`.",
            "",
            f"Final all-four stacker L2: `{final_stack['l2']}`.",
            "",
            "Final standardized-logit coefficients: `{} `.".format(
                ", ".join(
                    f"{member}={coefficient:.8g}"
                    for member, coefficient in zip(ALL_MEMBERS, final_stack["coefficient"])
                )
            ),
            "",
            "These are weak-selection scores, not calibrated retention probabilities. The ",
            "corrected 120-pair panel may be opened only after this decision is written, and ",
            "its result remains retrospective because component results were already known.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    base_oof_path = Path(args.base_oof).resolve()
    base_manifest_path = Path(args.base_manifest).resolve()
    rde_path = Path(args.rde_oof).resolve()
    output = Path(args.output_dir).resolve()
    for path in (base_oof_path, base_manifest_path, rde_path):
        base._require(path.is_file(), f"missing input {path}")
    base._require(not output.exists(), "output directory exists; refusing overwrite")
    base._require("private_data" in output.parts, "output must remain under private_data")
    with base_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    base._require(source_manifest.get("retention_labels_read") is False, "base OOF artifact read retention")
    base._require(source_manifest.get("common_oof_rows") == base.EXPECTED_OOF_ROWS, "base OOF count changed")

    frame = pd.read_csv(base_oof_path, dtype={"row_id": str})
    required = {"row_id", "fold", "weak_label", "peptide_id", "affibody_id", *base.MEMBERS}
    base._require(required.issubset(frame.columns), "base OOF columns changed")
    rde = load_rde_mean_logit_oof(rde_path)
    frame = frame[list(required)].merge(
        rde, on=["row_id", "fold", "weak_label"], validate="one_to_one"
    )
    base._require(len(frame) == base.EXPECTED_OOF_ROWS, "four-member OOF join changed")
    frame = frame.sort_values("row_id").reset_index(drop=True)

    original_members = base.MEMBERS
    try:
        base.MEMBERS = ALL_MEMBERS
        metrics, outer_audit, final_stack, final_grid = base.evaluate_candidates(frame)
        old_stack_name = "cross_fitted_nonnegative_stack__all3"
        new_stack_name = "cross_fitted_nonnegative_stack__all4"
        base._require(old_stack_name in frame.columns, "stacker output name changed")
        frame = frame.rename(columns={old_stack_name: new_stack_name})
        metrics["candidate"] = metrics["candidate"].replace(
            {old_stack_name: new_stack_name}
        )
        selection = base.select_candidates(metrics)
    finally:
        base.MEMBERS = original_members

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
    base._write_json(selection_path, {"schema_version": SCHEMA_VERSION, **selection})
    report_path = output / "report.md"
    report_path.write_text(render_report(selection, metrics, final_stack), encoding="utf-8")
    os.chmod(report_path, 0o600)

    outputs = [prediction_path, metric_path, outer_path, grid_path, stack_path, selection_path, report_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "retention_labels_read": False,
        "common_oof_rows": base.EXPECTED_OOF_ROWS,
        "common_oof_positive": base.EXPECTED_OOF_POSITIVE,
        "members": list(ALL_MEMBERS),
        "rde_definition": {
            "method": "mean_logit",
            "heads": list(RDE_HEADS),
        },
        "selection": selection,
        "inputs": {
            "base_oof": {"path": str(base_oof_path), "sha256": sha256_file(base_oof_path)},
            "base_manifest": {"path": str(base_manifest_path), "sha256": sha256_file(base_manifest_path)},
            "rde_oof": {"path": str(rde_path), "sha256": sha256_file(rde_path)},
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "base_module_path": str(Path(base.__file__).resolve()),
            "base_module_sha256": sha256_file(Path(base.__file__).resolve()),
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        },
    }
    base._write_json(output / "manifest.json", manifest)
    print(json.dumps(selection, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-oof", default=str(DEFAULT_BASE_OOF))
    parser.add_argument("--base-manifest", default=str(DEFAULT_BASE_MANIFEST))
    parser.add_argument("--rde-oof", default=str(DEFAULT_RDE_OOF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
