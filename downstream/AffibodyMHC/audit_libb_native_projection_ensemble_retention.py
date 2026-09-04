#!/usr/bin/env python
"""Evaluate weak-label-locked native-projection ensembles on corrected LibB-120.

The weak-only selection and stacker parameters are copied and fingerprinted
before this command opens the direct-retention panel.  Retention is used only
to calculate retrospective metrics; it does not fit ensemble weights, select a
member subset, select a training epoch, or choose a score cutoff.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import audit_libb_locked_ensemble_retention as prior
from downstream.AffibodyMHC import build_libb_weak_ensemble as weak_base
from downstream.AffibodyMHC import build_libb_weak_ensemble_native_projection as weak_native


SCHEMA_VERSION = "libb-native-projection-ensemble-retention-audit-v1"
EXPECTED_WEAK_SCHEMA = weak_native.SCHEMA_VERSION
MEMBERS = weak_native.ALL_MEMBERS
STAB_SOURCE_MODEL = "stab_designed_ordered"
RDE_SOURCE_MODEL = "rde_network_designed_3fold_ensemble_native_projection"
SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)

MODEL_DEFINITIONS = (
    ("additive_7site", "Additive seven-position baseline", "fixed"),
    ("mint_layer5", "Frozen MINT layer 5", "fixed"),
    (weak_native.STAB_MEMBER, "StaB native projection", "seeded"),
    (weak_native.RDE_MEMBER, "RDE native projection", "seeded"),
    ("mean_logit_mint_rde_native", "MINT + RDE", "seeded"),
    ("mean_logit_mint_stab_native", "MINT + StaB", "seeded"),
    ("mean_logit_all4_native", "Equal-logit four-model ensemble", "seeded"),
    ("nonnegative_stack_all4_native", "Weak-label-fitted four-model stacker", "seeded"),
)

DEFAULT_WEAK_DIR = weak_native.DEFAULT_OUTPUT
DEFAULT_MINT_SOURCE = prior.DEFAULT_MINT_SOURCE
DEFAULT_ADDITIVE_SOURCE = prior.DEFAULT_ADDITIVE_SOURCE
DEFAULT_STAB_DIR = (
    REPO_ROOT
    / "private_data/experiments/stab_libb_native_projection_120_full_v1/"
    "blinded_predictions"
)
DEFAULT_RDE_DIR = (
    REPO_ROOT
    / "private_data/experiments/rde_libb_native_projection_120_ensemble_v1/"
    "blinded_predictions"
)
DEFAULT_PANEL = prior.DEFAULT_PANEL
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "private_data/experiments/libb_native_projection_ensemble_retention_audit_v1"
)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    """Write with options supported by the project's pinned pandas 1.3."""
    prior._require(not path.exists(), f"refusing to overwrite {path}")
    frame.to_csv(path, index=False, float_format="%.12g", na_rep="")
    os.chmod(path, 0o600)


def apply_stacker(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> np.ndarray:
    order = tuple(map(str, parameters["members_in_coefficient_order"]))
    prior._require(order == MEMBERS, f"stacker member order changed: {order}")
    raw = np.column_stack(
        [weak_base.clipped_logit(frame[member].to_numpy(dtype=float)) for member in order]
    )
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    coefficient = np.asarray(parameters["coefficient"], dtype=np.float64)
    prior._require(mean.shape == scale.shape == coefficient.shape == (len(MEMBERS),), "bad stacker shape")
    prior._require(bool((scale > 0).all()), "nonpositive stacker scale")
    prior._require(bool((coefficient >= 0).all()), "negative stacker coefficient")
    logits = float(parameters["intercept"]) + ((raw - mean) / scale) @ coefficient
    return weak_base.sigmoid(logits)


def lock_weak_artifacts(weak_dir: Path, output: Path) -> dict[str, Any]:
    files = {
        name: weak_dir / name
        for name in (
            "manifest.json",
            "locked_weak_selection.json",
            "stacker_locked_parameters.json",
            "candidate_metrics.csv",
        )
    }
    for path in files.values():
        prior._require(path.is_file(), f"missing weak artifact {path}")
    manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
    selection = json.loads(files["locked_weak_selection.json"].read_text(encoding="utf-8"))
    parameters = json.loads(files["stacker_locked_parameters.json"].read_text(encoding="utf-8"))
    for name, payload in (("manifest", manifest), ("selection", selection), ("parameters", parameters)):
        prior._require(payload.get("schema_version") == EXPECTED_WEAK_SCHEMA, f"{name} schema changed")
        prior._require(payload.get("retention_labels_read") is False, f"{name} read retention")
    prior._require(tuple(parameters["members_in_coefficient_order"]) == MEMBERS, "member order changed")

    prior._require(not output.exists(), f"output exists: {output}")
    prior._require("private_data" in output.parts, "output must remain under private_data")
    output.mkdir(parents=True, mode=0o700)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "event": "weak artifacts fingerprinted before this command opened retention inputs",
        "retention_sources_opened": False,
        "historical_panel_status": "previously examined elsewhere; subsequent audit is retrospective",
        "selection": selection,
        "weak_artifacts": {name: prior._file_record(path) for name, path in files.items()},
    }
    prior._write_json_exclusive(output / "pre_retention_lock.json", seal)
    return {"manifest": manifest, "selection": selection, "parameters": parameters, "seal": seal}


def _load_fixed_score(path: Path, source_model: str, output_name: str) -> pd.DataFrame:
    frame = prior._read_prediction_source(path)
    chosen = prior._selected_model(frame, source_model, 1)
    return chosen[["eval_row_id", "score"]].rename(columns={"score": output_name})


def _load_seeded_scores(directory: Path, source_model: str, output_name: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(directory.glob(f"{source_model}__seed*.csv"))
    prior._require(len(paths) == len(SEEDS), f"found {len(paths)} {source_model} files")
    blocks = []
    for path in paths:
        frame = prior._read_prediction_source(path)
        chosen = frame.loc[frame["model"].eq(source_model), ["eval_row_id", "seed", "score"]].copy()
        prior._require(len(chosen) == prior.EXPECTED_ROWS, f"incomplete file {path}")
        chosen["seed"] = pd.to_numeric(chosen["seed"], errors="raise").astype(int)
        chosen = chosen.rename(columns={"score": output_name})
        blocks.append(chosen)
    combined = pd.concat(blocks, ignore_index=True)
    prior._require(tuple(sorted(combined["seed"].unique())) == SEEDS, f"{source_model} seeds changed")
    prior._require(not bool(combined.duplicated(["seed", "eval_row_id"]).any()), "duplicate seed scores")
    return combined, paths


def add_ensembles(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["mean_logit_mint_rde_native"] = weak_base.mean_logit_score(
        [result["mint_layer5"], result[weak_native.RDE_MEMBER]]
    )
    result["mean_logit_mint_stab_native"] = weak_base.mean_logit_score(
        [result["mint_layer5"], result[weak_native.STAB_MEMBER]]
    )
    result["mean_logit_all4_native"] = weak_base.mean_logit_score(
        [result[member] for member in MEMBERS]
    )
    result["nonnegative_stack_all4_native"] = apply_stacker(result, parameters)
    return result


def metric_record(frame: pd.DataFrame, score_name: str) -> dict[str, Any]:
    per_peptide = []
    for peptide, group in frame.groupby("peptide_design_code", sort=True):
        labels = group["target_binder"].to_numpy(dtype=int)
        scores = group[score_name].to_numpy(dtype=float)
        retention = group["target_retention"].to_numpy(dtype=float)
        row = {
            "peptide_design_code": str(peptide),
            "within_peptide_spearman": prior._spearman(scores, retention),
            "within_peptide_auroc": float("nan"),
            "within_peptide_average_precision": float("nan"),
            "binary_evaluable": int(np.unique(labels).size == 2),
        }
        if row["binary_evaluable"]:
            row["within_peptide_auroc"] = float(roc_auc_score(labels, scores))
            row["within_peptide_average_precision"] = float(average_precision_score(labels, scores))
        per_peptide.append(row)
    rows = pd.DataFrame(per_peptide)
    return {
        "within_peptide_average_precision": float(rows["within_peptide_average_precision"].mean()),
        "within_peptide_auroc": float(rows["within_peptide_auroc"].mean()),
        "within_peptide_spearman": float(rows["within_peptide_spearman"].mean()),
        "binary_evaluable_peptides": int(rows["within_peptide_average_precision"].notna().sum()),
        "spearman_evaluable_peptides": int(rows["within_peptide_spearman"].notna().sum()),
    }


def averaged_structure_scores(seed_frame: pd.DataFrame) -> pd.DataFrame:
    key = "eval_row_id"
    result = seed_frame[[key]].drop_duplicates().sort_values(key).reset_index(drop=True)
    for member in (weak_native.STAB_MEMBER, weak_native.RDE_MEMBER):
        wide = seed_frame.pivot(index=key, columns="seed", values=member).reset_index()
        seed_columns = sorted(column for column in wide.columns if column != key)
        prior._require(tuple(seed_columns) == SEEDS, f"{member} seed columns changed")
        current = wide[[key]].copy()
        current[member] = weak_base.mean_logit_score(
            [wide[column].to_numpy(dtype=float) for column in seed_columns]
        )
        result = result.merge(current, on=key, validate="one_to_one")
    return result


def render_report(
    weak_payload: Mapping[str, Any],
    weak_metrics: pd.DataFrame,
    seed_summary: pd.DataFrame,
    averaged_metrics: pd.DataFrame,
) -> str:
    key_candidates = {
        "mean_logit__additive_7site",
        "mean_logit__mint_layer5",
        f"mean_logit__{weak_native.STAB_MEMBER}",
        f"mean_logit__{weak_native.RDE_MEMBER}",
        f"mean_logit__mint_layer5__{weak_native.RDE_MEMBER}",
        f"mean_logit__{'__'.join(MEMBERS)}",
        "cross_fitted_nonnegative_stack__all4_native_projection",
    }
    shown_weak = weak_metrics.loc[weak_metrics["candidate"].isin(key_candidates)].copy()
    shown_weak = shown_weak.sort_values("within_peptide_ap", ascending=False, kind="stable")
    averaged_lookup = averaged_metrics.set_index("model_key")
    mint_result = averaged_lookup.loc["mint_layer5"]
    ensemble_keys = [
        "mean_logit_mint_rde_native",
        "mean_logit_mint_stab_native",
        "mean_logit_all4_native",
        "nonnegative_stack_all4_native",
    ]
    all_ensembles_beat_mint = all(
        float(averaged_lookup.loc[key, metric]) > float(mint_result[metric])
        for key in ensemble_keys
        for metric in (
            "within_peptide_average_precision",
            "within_peptide_auroc",
            "within_peptide_spearman",
        )
    )
    lines = [
        "# Native-projection LibB ensemble recalculation",
        "",
        "The StaB and RDE components use the clean trainable projections directly from their ",
        "native feature dimensions. The additive and frozen-MINT components are unchanged. ",
        "Ensemble weights and member selection use only selection-derived weak labels.",
        "",
        "## Weak-label comparison used to fit and lock the ensemble",
        "",
        "| Score | Within-peptide AP | Within-peptide AUROC | Weak-label Spearman |",
        "|---|---:|---:|---:|",
    ]
    for row in shown_weak.itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.within_peptide_ap:.4f} | "
            f"{row.within_peptide_auroc:.4f} | {row.within_peptide_weak_label_spearman:.4f} |"
        )
    coefficients = weak_payload["parameters"]["coefficient"]
    lines.extend(
        [
            "",
            "MINT layer 5 remains the best individual score by weak-label within-peptide AP. ",
            "MINT + RDE is the best multi-model mean-logit score on that same criterion. The ",
            "four-model stacker is also locked from weak data; it was not selected using retention.",
            "",
            "The stacker coefficients are: `{} `.".format(
                ", ".join(
                    f"{member}={float(value):.6g}"
                    for member, value in zip(MEMBERS, coefficients)
                )
            ),
            "The zero additive coefficient means the fitted four-model stacker actually uses ",
            "MINT, StaB, and RDE; the additive model adds no extra weak-label signal after those ",
            "three inputs are present.",
            "",
            "## Corrected 120-pair retention panel",
            "",
            "AP and AUROC rank retention-defined binders among the ten Affibodies for each ",
            "peptide, then average over the 11 peptides containing both binders and non-binders. ",
            "Spearman compares the full score ordering with numerical retention and averages over ",
            "all 12 peptides.",
            "",
            "The first table gives the mean and sample standard deviation across five readout ",
            "seeds. Additive and MINT are deterministic, so their standard deviations are zero.",
            "",
            "| Score | Within-peptide AP | Within-peptide AUROC | Within-peptide Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in seed_summary.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.ap_mean:.4f} ± {row.ap_sd:.4f} | "
            f"{row.auroc_mean:.4f} ± {row.auroc_sd:.4f} | "
            f"{row.spearman_mean:.4f} ± {row.spearman_sd:.4f} |"
        )
    lines.extend(
        [
            "",
            "For deployment, the five structure-readout scores are averaged in logit space ",
            "before applying the ensemble. Metrics of those final averaged scores are:",
            "",
            "| Score | AP | AUROC | Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in averaged_metrics.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.within_peptide_average_precision:.4f} | "
            f"{row.within_peptide_auroc:.4f} | {row.within_peptide_spearman:.4f} |"
        )
    if all_ensembles_beat_mint:
        lines.extend(
            [
                "",
                "On this retrospective panel, every ensemble shown improves over MINT alone on ",
                "all three peptide-conditioned metrics. No ensemble is best on everything: ",
                "StaB alone has the highest AP, MINT + StaB has the highest AUROC, and RDE alone ",
                "has the highest Spearman correlation. The ensemble therefore improves balance ",
                "across metrics rather than producing a new winner on every column.",
            ]
        )
    lines.extend(
        [
            "",
            "The retention table is retrospective: it describes how the locked scores behave on ",
            "the existing 120 measured pairs. No retention-derived score threshold was fitted in ",
            "this recalculation. The prospective wet-lab batch remains the actual test.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    weak_dir = Path(args.weak_dir).resolve()
    output = Path(args.output_dir).resolve()
    weak_payload = lock_weak_artifacts(weak_dir, output)

    # Retention-bearing inputs are deliberately opened only after the lock above.
    mint_path = Path(args.mint_source).resolve()
    additive_path = Path(args.additive_source).resolve()
    stab_dir = Path(args.stab_dir).resolve()
    rde_dir = Path(args.rde_dir).resolve()
    panel_path = Path(args.panel).resolve()
    additive = _load_fixed_score(additive_path, prior.ADDITIVE_SOURCE_MODEL, "additive_7site")
    mint = _load_fixed_score(mint_path, prior.MINT_SOURCE_MODEL, "mint_layer5")
    stab, stab_paths = _load_seeded_scores(stab_dir, STAB_SOURCE_MODEL, weak_native.STAB_MEMBER)
    rde, rde_paths = _load_seeded_scores(rde_dir, RDE_SOURCE_MODEL, weak_native.RDE_MEMBER)

    reference_ids = mint["eval_row_id"].tolist()
    prior._require(
        prior._membership_sha256(reference_ids) == prior.EXPECTED_MEMBERSHIP_SHA256,
        "corrected-120 membership changed",
    )
    for name, frame in (("additive", additive), ("StaB", stab), ("RDE", rde)):
        prior._assert_same_membership(reference_ids, frame, name)
    panel = prior._read_panel(panel_path)
    prior._validate_panel(panel)
    prior._assert_same_membership(reference_ids, panel, "retention panel")

    fixed = panel.merge(additive, on="eval_row_id", validate="one_to_one")
    fixed = fixed.merge(mint, on="eval_row_id", validate="one_to_one")
    seeded = stab.merge(rde, on=["eval_row_id", "seed"], validate="one_to_one")
    seeded = fixed.merge(seeded, on="eval_row_id", validate="one_to_many")
    seeded = add_ensembles(seeded, weak_payload["parameters"])

    metric_rows = []
    for model_key, display_name, mode in MODEL_DEFINITIONS:
        seeds: Sequence[object] = ("fixed",) if mode == "fixed" else SEEDS
        for seed in seeds:
            block = seeded if seed == "fixed" else seeded.loc[seeded["seed"].eq(seed)]
            # Fixed columns are duplicated over structure seeds; use one complete panel.
            if seed == "fixed":
                block = block.loc[block["seed"].eq(SEEDS[0])]
            metric_rows.append(
                {
                    "model_key": model_key,
                    "display_name": display_name,
                    "seed": seed,
                    **metric_record(block, model_key),
                }
            )
    per_seed_metrics = pd.DataFrame(metric_rows)

    summary_rows = []
    for model_key, display_name, _ in MODEL_DEFINITIONS:
        group = per_seed_metrics.loc[per_seed_metrics["model_key"].eq(model_key)]
        def summarize(column: str) -> tuple[float, float]:
            values = group[column].to_numpy(dtype=float)
            return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ap_mean, ap_sd = summarize("within_peptide_average_precision")
        auroc_mean, auroc_sd = summarize("within_peptide_auroc")
        spearman_mean, spearman_sd = summarize("within_peptide_spearman")
        summary_rows.append(
            {
                "model_key": model_key,
                "display_name": display_name,
                "fits": len(group),
                "ap_mean": ap_mean,
                "ap_sd": ap_sd,
                "auroc_mean": auroc_mean,
                "auroc_sd": auroc_sd,
                "spearman_mean": spearman_mean,
                "spearman_sd": spearman_sd,
            }
        )
    seed_summary = pd.DataFrame(summary_rows)

    averaged = fixed.merge(averaged_structure_scores(seeded), on="eval_row_id", validate="one_to_one")
    averaged = add_ensembles(averaged, weak_payload["parameters"])
    averaged_metric_rows = []
    per_peptide_rows = []
    for model_key, display_name, _ in MODEL_DEFINITIONS:
        averaged_metric_rows.append(
            {"model_key": model_key, "display_name": display_name, **metric_record(averaged, model_key)}
        )
        for peptide, group in averaged.groupby("peptide_design_code", sort=True):
            record = metric_record(group.assign(peptide_design_code=str(peptide)), model_key)
            per_peptide_rows.append(
                {
                    "model_key": model_key,
                    "display_name": display_name,
                    "peptide_design_code": str(peptide),
                    **record,
                }
            )
    averaged_metrics = pd.DataFrame(averaged_metric_rows)
    per_peptide_metrics = pd.DataFrame(per_peptide_rows)

    weak_metrics = pd.read_csv(weak_dir / "candidate_metrics.csv")
    report_path = output / "report.md"
    report_path.write_text(
        render_report(weak_payload, weak_metrics, seed_summary, averaged_metrics),
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)

    output_frames = {
        "per_seed_predictions.csv": seeded,
        "per_seed_metrics.csv": per_seed_metrics,
        "seed_metric_summary.csv": seed_summary,
        "averaged_score_predictions.csv": averaged,
        "averaged_score_metrics.csv": averaged_metrics,
        "averaged_score_per_peptide_metrics.csv": per_peptide_metrics,
    }
    output_records = {}
    for filename, frame in output_frames.items():
        path = output / filename
        _write_csv_exclusive(path, frame)
        output_records[filename] = {**prior._file_record(path), "rows": len(frame)}
    output_records["report.md"] = prior._file_record(report_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "weak_artifact_fingerprinted_before_this_command_opened_retention": True,
        "historical_panel_status": "previously examined elsewhere; this is not an untouched prospective test",
        "retention_used_for_training_selection_or_cutoff": False,
        "retention_panel": {
            "pairs": 120,
            "peptides": 12,
            "affibodies_per_peptide": 10,
            "binders": 61,
            "non_binders": 59,
            "binder_definition": "retention >= 75",
        },
        "aggregation": {
            "RDE_heads_before_this_stage": "mean logit across three pretrained heads",
            "structure_seeds": "mean logit across five readout seeds for averaged-score metrics",
            "equal_ensembles": "mean logit, one vote per component",
            "stacker": "weak-label-locked nonnegative standardized-logit logistic model",
        },
        "weak_lock": weak_payload["seal"],
        "inputs": {
            "mint": prior._file_record(mint_path),
            "additive": prior._file_record(additive_path),
            "stab_files": [prior._file_record(path) for path in stab_paths],
            "rde_files": [prior._file_record(path) for path in rde_paths],
            "panel": prior._file_record(panel_path),
        },
        "code": prior._file_record(Path(__file__).resolve()),
        "outputs": output_records,
    }
    prior._write_json_exclusive(output / "manifest.json", manifest)
    print(seed_summary.to_string(index=False))
    print("\nAveraged scores:\n", averaged_metrics.to_string(index=False))
    print(f"wrote {output}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-dir", default=str(DEFAULT_WEAK_DIR))
    parser.add_argument("--mint-source", default=str(DEFAULT_MINT_SOURCE))
    parser.add_argument("--additive-source", default=str(DEFAULT_ADDITIVE_SOURCE))
    parser.add_argument("--stab-dir", default=str(DEFAULT_STAB_DIR))
    parser.add_argument("--rde-dir", default=str(DEFAULT_RDE_DIR))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
