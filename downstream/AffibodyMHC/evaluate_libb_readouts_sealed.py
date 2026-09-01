#!/usr/bin/env python
"""Sealed final evaluation for LibB structure/readout predictions.

Candidate-model prediction files are deliberately blind: their complete CSV
schema is ``eval_row_id,model,seed,score``.  They may contain neither retention
targets nor peptide/Affibody identities.  This runner validates those files,
optionally extracts the current sequence controls without reading the target
columns embedded in their legacy artifacts, and only then opens the separate
retention audit for the final merge and metric calculation.

The score direction is always "larger means more likely to bind."  Retention is
used only by this final evaluator and never for fitting, early stopping,
architecture choice, or hyperparameter choice.

Example (run only after all model scores have been frozen)::

    venv/bin/python downstream/AffibodyMHC/evaluate_libb_readouts_sealed.py \
      --predictions private_data/experiments/esmfold2_readout/scores_seed1.csv \
      --predictions private_data/experiments/esmfold2_readout/scores_seed2.csv \
      --output-dir private_data/experiments/esmfold2_readout_final_evaluation
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.wetlab_metrics import (
    TIE_POLICY,
    evaluate_wetlab_predictions,
)


SCHEMA_VERSION = "libb-sealed-readout-evaluation-v1"
BLINDED_PREDICTION_COLUMNS = (
    "eval_row_id",
    "model",
    "seed",
    "score",
)
EXPECTED_EVALUATION_ROWS = 119
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_MISSING_MATRIX_CELLS = 1
RETENTION_THRESHOLD = 75.0

DEFAULT_RETENTION_AUDIT = REPO_ROOT / "private_data/derived/retention_sequences_v2.csv"
DEFAULT_BASELINE_CONTROLS = (
    REPO_ROOT
    / "private_data/experiments/mint_cached_primary_libb_double_cold_v1/predictions.csv"
)
DEFAULT_LORA_CONTROL = (
    REPO_ROOT
    / "private_data/experiments/mint_selection_one_epoch_libb_v1/retention_predictions.csv"
)

REPORT_METRICS = (
    "global_auroc",
    "global_average_precision",
    "global_spearman",
    "within_peptide_spearman_mean",
    "peptide_macro_precision_at_1",
    "peptide_macro_precision_at_3",
    "peptide_macro_hit_at_1",
    "peptide_macro_hit_at_3",
    "peptide_macro_best_retention_at_1",
    "peptide_macro_best_retention_at_3",
    "peptide_macro_regret_at_1",
    "peptide_macro_regret_at_3",
    "peptide_macro_ndcg_at_3",
    "distinct_top1_affibodies",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _membership_sha256(values):
    ordered = sorted(str(value) for value in values)
    return hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()


def _read_csv_columns(path):
    return tuple(pd.read_csv(str(path), nrows=0).columns)


def load_blinded_predictions(paths):
    """Load target-free prediction files under the exact sealed schema."""
    paths = [Path(path) for path in paths]
    _require(paths, "at least one blinded prediction file is required")
    frames = []
    expected = set(BLINDED_PREDICTION_COLUMNS)
    for path in paths:
        _require(path.is_file(), "blinded prediction file does not exist: {}".format(path))
        observed = set(_read_csv_columns(path))
        _require(
            observed == expected,
            (
                "blinded prediction schema must be exactly {}; observed {}. "
                "Targets and sequence identities are forbidden in this artifact"
            ).format(sorted(expected), sorted(observed)),
        )
        frame = pd.read_csv(
            str(path),
            dtype={"eval_row_id": str, "model": str, "seed": str},
            keep_default_na=False,
            na_filter=False,
        )
        frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
        frame["source_role"] = "candidate"
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    validate_blinded_prediction_groups(output)
    return output


def validate_blinded_prediction_groups(frame):
    """Validate opaque IDs and matched panels without opening retention data."""
    required = set(BLINDED_PREDICTION_COLUMNS).union({"source_role"})
    missing = required.difference(frame.columns)
    _require(not missing, "prediction table is missing {}".format(sorted(missing)))
    _require(len(frame) > 0, "prediction table is empty")
    for column in ("eval_row_id", "model", "seed", "source_role"):
        _require(
            not bool(frame[column].astype(str).eq("").any()),
            "{} contains empty values".format(column),
        )
    score = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(score).all()), "prediction score contains non-finite values")
    duplicate = frame.duplicated(["model", "seed", "eval_row_id"], keep=False)
    _require(
        not bool(duplicate.any()),
        "a model/seed has duplicate opaque evaluation row IDs",
    )
    roles_per_group = frame.groupby(["model", "seed"])["source_role"].nunique()
    _require(
        bool(roles_per_group.eq(1).all()),
        "a model/seed occurs under multiple source roles",
    )
    memberships = []
    for _, group in frame.groupby(["model", "seed"], sort=True):
        memberships.append(_membership_sha256(group["eval_row_id"]))
    _require(
        len(set(memberships)) == 1,
        "model/seed prediction panels do not contain identical opaque ID sets",
    )
    return memberships[0]


def _load_csv_without_targets(path, required_columns):
    """Read only named non-target columns from a legacy control artifact."""
    path = Path(path)
    _require(path.is_file(), "control prediction file does not exist: {}".format(path))
    observed = set(_read_csv_columns(path))
    missing = set(required_columns).difference(observed)
    _require(not missing, "control artifact is missing {}".format(sorted(missing)))
    return pd.read_csv(
        str(path),
        usecols=list(required_columns),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def load_site_and_frozen_controls(path):
    """Extract the matched site and frozen-MINT controls without target columns."""
    columns = (
        "pair_uid",
        "library",
        "regime",
        "cleaning",
        "balance",
        "balance_seed",
        "representation",
        "binder_probability",
    )
    frame = _load_csv_without_targets(path, columns)
    selected = frame.loc[
        frame["library"].eq("LibB")
        & frame["regime"].eq("double_cold")
        & frame["cleaning"].eq("c0")
        & frame["balance"].eq("all_class_weighted")
        & frame["balance_seed"].eq("-1")
        & frame["representation"].isin(("site", "frozen_mint_chain_mean"))
    ].copy()
    _require(not selected.empty, "no current LibB site/frozen controls were found")
    expected_representations = {"site", "frozen_mint_chain_mean"}
    _require(
        set(selected["representation"]) == expected_representations,
        "current site/frozen control representations are incomplete",
    )
    selected["model"] = selected["representation"].map(
        {
            "site": "site_additive_control",
            "frozen_mint_chain_mean": "frozen_mint_control",
        }
    )
    selected["eval_row_id"] = selected["pair_uid"].astype(str)
    selected["seed"] = "-1"
    selected["score"] = pd.to_numeric(
        selected["binder_probability"], errors="raise"
    ).astype(float)
    selected["source_role"] = "control"
    return selected[
        list(BLINDED_PREDICTION_COLUMNS) + ["source_role"]
    ].reset_index(drop=True)


def load_lora_control(path):
    """Extract the frozen one-epoch LoRA control without loading its targets."""
    columns = (
        "pair_uid",
        "library",
        "arm",
        "training_seed",
        "epoch",
        "probability",
    )
    frame = _load_csv_without_targets(path, columns)
    selected = frame.loc[
        frame["library"].eq("LibB")
        & frame["arm"].eq("lora_cross")
        & frame["epoch"].eq("1")
    ].copy()
    _require(not selected.empty, "no current one-epoch LibB LoRA control was found")
    selected["eval_row_id"] = selected["pair_uid"].astype(str)
    selected["model"] = "lora_mint_one_epoch_control"
    selected["seed"] = selected["training_seed"].astype(str)
    selected["score"] = pd.to_numeric(selected["probability"], errors="raise").astype(float)
    selected["source_role"] = "control"
    return selected[
        list(BLINDED_PREDICTION_COLUMNS) + ["source_role"]
    ].reset_index(drop=True)


def discover_current_controls(
    baseline_path=DEFAULT_BASELINE_CONTROLS,
    lora_path=DEFAULT_LORA_CONTROL,
):
    """Load each current control whose default artifact is present."""
    frames = []
    discovered = []
    baseline_path = Path(baseline_path)
    lora_path = Path(lora_path)
    if baseline_path.is_file():
        frames.append(load_site_and_frozen_controls(baseline_path))
        discovered.append(baseline_path)
    if lora_path.is_file():
        frames.append(load_lora_control(lora_path))
        discovered.append(lora_path)
    if not frames:
        return pd.DataFrame(columns=list(BLINDED_PREDICTION_COLUMNS) + ["source_role"]), discovered
    output = pd.concat(frames, ignore_index=True)
    validate_blinded_prediction_groups(output)
    return output, discovered


def load_libb_retention_audit(
    path,
    expected_rows=None,
    expected_peptides=None,
    expected_affibodies=None,
    expected_missing_cells=None,
):
    """Open and validate the direct-retention audit at the final-evaluation gate."""
    path = Path(path)
    _require(path.is_file(), "retention audit does not exist: {}".format(path))
    frame = pd.read_csv(
        str(path),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    if "library" in frame:
        frame = frame.loc[frame["library"].eq("LibB")].copy()
    _require("target_retention" in frame, "retention audit lacks target_retention")
    frame = frame.loc[frame["target_retention"].astype(str).ne("")].copy()
    id_candidates = [
        column
        for column in ("eval_row_id", "row_id", "pair_uid")
        if column in frame.columns
    ]
    _require(
        len(id_candidates) == 1,
        (
            "retention audit must contain exactly one opaque ID column from "
            "eval_row_id, row_id, or pair_uid"
        ),
    )
    id_column = id_candidates[0]
    required = {
        id_column,
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    }
    missing = required.difference(frame.columns)
    _require(not missing, "retention audit is missing {}".format(sorted(missing)))
    output = frame[
        [
            id_column,
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
            "target_binder",
        ]
    ].copy()
    output = output.rename(columns={id_column: "eval_row_id"})
    for column in ("eval_row_id", "peptide_design_code", "affibody_design_code"):
        _require(
            not bool(output[column].astype(str).eq("").any()),
            "retention audit {} contains empty values".format(column),
        )
    _require(
        not bool(output["eval_row_id"].duplicated().any()),
        "retention audit contains duplicate opaque evaluation row IDs",
    )
    _require(
        not bool(
            output.duplicated(
                ["peptide_design_code", "affibody_design_code"], keep=False
            ).any()
        ),
        "retention audit contains duplicate peptide/Affibody pairs",
    )
    output["target_retention"] = pd.to_numeric(
        output["target_retention"], errors="raise"
    ).astype(float)
    output["target_binder"] = pd.to_numeric(
        output["target_binder"], errors="raise"
    ).astype(int)
    _require(
        bool(np.isfinite(output["target_retention"].to_numpy(dtype=float)).all()),
        "retention audit has non-finite measurements",
    )
    _require(
        set(output["target_binder"].tolist()).issubset({0, 1}),
        "retention audit binder labels are not binary",
    )
    expected_binder = output["target_retention"].ge(RETENTION_THRESHOLD).astype(int)
    _require(
        bool(output["target_binder"].eq(expected_binder).all()),
        "retention audit binder label does not equal retention >=75%",
    )
    n_peptides = int(output["peptide_design_code"].nunique())
    n_affibodies = int(output["affibody_design_code"].nunique())
    missing_cells = int(n_peptides * n_affibodies - len(output))
    if expected_rows is not None:
        _require(len(output) == int(expected_rows), "retention audit row count changed")
    if expected_peptides is not None:
        _require(n_peptides == int(expected_peptides), "retention peptide count changed")
    if expected_affibodies is not None:
        _require(n_affibodies == int(expected_affibodies), "retention Affibody count changed")
    if expected_missing_cells is not None:
        _require(
            missing_cells == int(expected_missing_cells),
            "retention matrix missing-cell count changed",
        )
    return output.sort_values("eval_row_id").reset_index(drop=True)


def evaluate_matched_predictions(predictions, retention_audit):
    """Apply the final target merge and compute matched model/seed metrics."""
    prediction_membership = validate_blinded_prediction_groups(predictions)
    audit_membership = _membership_sha256(retention_audit["eval_row_id"])
    _require(
        prediction_membership == audit_membership,
        "prediction and retention audit opaque ID sets differ",
    )
    audit_ids = set(retention_audit["eval_row_id"].astype(str))
    metric_rows = []
    peptide_frames = []
    ranked_frames = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        _require(
            set(group["eval_row_id"].astype(str)) == audit_ids,
            "{} seed {} is not evaluated on the exact audit panel".format(model, seed),
        )
        role_values = set(group["source_role"].astype(str))
        _require(len(role_values) == 1, "model/seed source role is not unique")
        role = next(iter(role_values))
        blinded = group[list(BLINDED_PREDICTION_COLUMNS)].copy()
        merged = blinded.merge(
            retention_audit,
            on="eval_row_id",
            how="inner",
            validate="one_to_one",
        )
        _require(len(merged) == len(retention_audit), "final target merge lost rows")
        metrics, peptide, ranked = evaluate_wetlab_predictions(
            merged,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
            score_column="score",
            binder_column="target_binder",
            retention_column="target_retention",
            k_values=(1, 3),
        )
        metric_row = {
            "model": str(model),
            "seed": str(seed),
            "source_role": role,
        }
        metric_row.update(metrics)
        metric_rows.append(metric_row)
        peptide.insert(0, "source_role", role)
        peptide.insert(0, "seed", str(seed))
        peptide.insert(0, "model", str(model))
        peptide_frames.append(peptide)
        ranked.insert(0, "source_role", role)
        ranked_frames.append(ranked)
    metrics = pd.DataFrame(metric_rows)
    per_peptide = pd.concat(peptide_frames, ignore_index=True)
    ranked = pd.concat(ranked_frames, ignore_index=True)
    _require(
        bool(metrics["tie_policy"].eq(TIE_POLICY).all()),
        "metric tie policy changed",
    )
    return metrics, per_peptide, ranked


def summarize_across_seeds(metrics):
    """Create one matched row per model with mean, SD, and individual values."""
    missing = set(REPORT_METRICS).difference(metrics.columns)
    _require(not missing, "metrics table is missing {}".format(sorted(missing)))
    rows = []
    for (model, role), group in metrics.groupby(["model", "source_role"], sort=True):
        group = group.sort_values("seed", kind="mergesort")
        row = {
            "model": str(model),
            "source_role": str(role),
            "n_seeds": int(len(group)),
            "seeds": json.dumps(group["seed"].astype(str).tolist()),
        }
        for metric in REPORT_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row["{}_seed_mean".format(metric)] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row["{}_seed_sd".format(metric)] = (
                float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
            )
            row["{}_individual".format(metric)] = json.dumps(
                [float(value) if math.isfinite(value) else None for value in values]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def publish_evaluation(output_dir, tables, manifest):
    """Publish private final-evaluation artifacts; manifest is the commit marker."""
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(str(output_dir), 0o700)
    output_records = {}
    for filename, frame in tables.items():
        _require(Path(filename).name == filename, "output filename must be local")
        path = output_dir / filename
        _write_csv(frame, path)
        output_records[filename] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        }
    final_manifest = dict(manifest)
    final_manifest["outputs"] = output_records
    # Writing the manifest last makes its presence the completion marker.
    _write_json(final_manifest, output_dir / "manifest.json")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Target-free CSV; repeat for multiple readout/seed files.",
    )
    parser.add_argument(
        "--retention-audit",
        type=Path,
        default=DEFAULT_RETENTION_AUDIT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-controls",
        type=Path,
        default=DEFAULT_BASELINE_CONTROLS,
    )
    parser.add_argument(
        "--lora-control",
        type=Path,
        default=DEFAULT_LORA_CONTROL,
    )
    parser.add_argument(
        "--no-current-controls",
        action="store_true",
        help="Do not add discoverable site/frozen-MINT/LoRA controls.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    validate_private_output_path(args.output_dir, REPO_ROOT)
    candidates = load_blinded_predictions(args.predictions)
    discovered_controls = []
    if not args.no_current_controls:
        controls, discovered_controls = discover_current_controls(
            args.baseline_controls,
            args.lora_control,
        )
        predictions = pd.concat([candidates, controls], ignore_index=True)
    else:
        predictions = candidates
    prediction_membership = validate_blinded_prediction_groups(predictions)

    # This is intentionally the first point at which direct-retention values
    # are opened.  Every candidate score and panel membership is already fixed.
    audit = load_libb_retention_audit(
        args.retention_audit,
        expected_rows=EXPECTED_EVALUATION_ROWS,
        expected_peptides=EXPECTED_PEPTIDES,
        expected_affibodies=EXPECTED_AFFIBODIES,
        expected_missing_cells=EXPECTED_MISSING_MATRIX_CELLS,
    )
    metrics, per_peptide, ranked = evaluate_matched_predictions(predictions, audit)
    seed_summary = summarize_across_seeds(metrics)
    sources = {
        "blinded_predictions": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.predictions
        ],
        "retention_audit": {
            "path": str(args.retention_audit.resolve()),
            "sha256": sha256_file(args.retention_audit),
        },
        "discovered_controls": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in discovered_controls
        ],
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "sources": sources,
        "prediction_input_contract": list(BLINDED_PREDICTION_COLUMNS),
        "score_direction": "higher_is_more_likely_binder",
        "retention_threshold_for_binary_metrics": RETENTION_THRESHOLD,
        "retention_usage": (
            "final retrospective evaluation only; never training, early stopping, "
            "architecture selection, or hyperparameter selection"
        ),
        "tie_policy": TIE_POLICY,
        "prediction_membership_sha256": prediction_membership,
        "audit_membership_sha256": _membership_sha256(audit["eval_row_id"]),
        "evaluation_rows": int(len(audit)),
        "peptides": int(audit["peptide_design_code"].nunique()),
        "affibodies": int(audit["affibody_design_code"].nunique()),
        "unmeasured_matrix_cells": int(
            audit["peptide_design_code"].nunique()
            * audit["affibody_design_code"].nunique()
            - len(audit)
        ),
        "models": sorted(set(metrics["model"].astype(str))),
        "analysis_status": "sealed_final_retrospective_evaluation",
    }
    publish_evaluation(
        args.output_dir,
        {
            "matched_metrics_by_seed.csv": metrics,
            "matched_metrics_seed_summary.csv": seed_summary,
            "per_peptide_metrics.csv": per_peptide,
            "ranked_per_pair.csv": ranked,
        },
        manifest,
    )


if __name__ == "__main__":
    main()
