#!/usr/bin/env python3
"""Sealed final evaluation for LibA ESMFold2 readout predictions.

Prediction files must already be frozen and contain only
``eval_row_id,model,seed,score``.  This command validates their common opaque-ID
panel before it opens the physically separate 108-row direct-retention sidecar.
It never selects a model, epoch, threshold, or hyperparameter from retention.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.evaluate_libb_readouts_sealed import (
    BLINDED_PREDICTION_COLUMNS,
    RETENTION_THRESHOLD,
    _membership_sha256,
    evaluate_matched_predictions,
    load_blinded_predictions,
    load_libb_retention_audit as load_retention_audit,
    publish_evaluation,
    summarize_across_seeds,
    validate_blinded_prediction_groups,
)
from downstream.AffibodyMHC.wetlab_metrics import TIE_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "liba-sealed-readout-evaluation-v1"
EVALUATION_LABEL_SCHEMA = "esmfold2-liba-evaluation-labels-v1"
EXPECTED_ROWS_SCHEMA = "esmfold2-liba-canonical-rows-v1"
EXPECTED_ROWS_SHA256 = "74b8716719b07e0040db57e58644d48eb5f1c36c813fdc95df431029319ae086"
EXPECTED_EVALUATION_SHA256 = "bbf0c9b3ee30ec8f43418a73014b5cb6bb138b4defafd2bd463916db7c171f1a"
EXPECTED_EVALUATION_COLUMNS = (
    "row_id",
    "peptide_design_code",
    "affibody_design_code",
    "target_retention",
    "target_binder",
)
EXPECTED_EVALUATION_ROWS = 108
EXPECTED_PEPTIDES = 9
EXPECTED_AFFIBODIES = 12
EXPECTED_MISSING_MATRIX_CELLS = 0
EXPECTED_BINDERS = 38
EXPECTED_NONBINDERS = 70
EXPECTED_MEMBERSHIP_SHA256 = "35622dfeeb42740e522c95fdbaadb9f04c2f2a5e26fcb09b1a1c1e6243ae68ea"
DEFAULT_EVALUATION_ROOT = (
    REPO_ROOT / "private_data/derived/esmfold2_liba_evaluation_labels_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _validate_evaluation_sidecar(
    labels_path: Path, manifest_path: Path
) -> Mapping[str, Any]:
    _require(labels_path.is_file(), f"evaluation labels do not exist: {labels_path}")
    _require(manifest_path.is_file(), f"evaluation manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == EVALUATION_LABEL_SCHEMA,
        "LibA evaluation-label schema changed",
    )
    _require(
        manifest.get("canonical_rows_sha256") == EXPECTED_ROWS_SHA256,
        "LibA canonical-row membership changed",
    )
    output = manifest.get("output", {})
    _require(output.get("filename") == labels_path.name, "evaluation filename changed")
    _require(
        tuple(output.get("columns", ())) == EXPECTED_EVALUATION_COLUMNS,
        "evaluation columns changed",
    )
    observed_sha256 = sha256_file(labels_path)
    _require(output.get("sha256") == observed_sha256, "evaluation checksum disagrees with manifest")
    _require(observed_sha256 == EXPECTED_EVALUATION_SHA256, "locked evaluation checksum changed")
    sealed = manifest.get("sealed_evaluation", {})
    expected = {
        "rows": EXPECTED_EVALUATION_ROWS,
        "positive": EXPECTED_BINDERS,
        "negative": EXPECTED_NONBINDERS,
        "binder_threshold": RETENTION_THRESHOLD,
    }
    for key, value in expected.items():
        _require(sealed.get(key) == value, f"sealed evaluation contract changed for {key}")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Frozen target-free CSV; repeat for model/seed files.",
    )
    parser.add_argument(
        "--evaluation-labels",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT / "evaluation_labels.csv",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=DEFAULT_EVALUATION_ROOT / "manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    validate_private_output_path(args.output_dir, REPO_ROOT)
    predictions = load_blinded_predictions(args.predictions)
    prediction_membership = validate_blinded_prediction_groups(predictions)

    # Hash/contract checks occur only after all candidate scores are frozen.
    # This is the first stage that opens direct outcomes.
    evaluation_manifest = _validate_evaluation_sidecar(
        args.evaluation_labels, args.evaluation_manifest
    )
    audit = load_retention_audit(
        args.evaluation_labels,
        expected_rows=EXPECTED_EVALUATION_ROWS,
        expected_peptides=EXPECTED_PEPTIDES,
        expected_affibodies=EXPECTED_AFFIBODIES,
        expected_missing_cells=EXPECTED_MISSING_MATRIX_CELLS,
        expected_binders=EXPECTED_BINDERS,
        expected_nonbinders=EXPECTED_NONBINDERS,
    )
    audit_membership = _membership_sha256(audit["eval_row_id"])
    _require(audit_membership == EXPECTED_MEMBERSHIP_SHA256, "LibA evaluation ID set changed")
    metrics, per_peptide, ranked = evaluate_matched_predictions(predictions, audit)
    summary = summarize_across_seeds(metrics)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "sources": {
            "blinded_predictions": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.predictions
            ],
            "evaluation_labels": {
                "path": str(args.evaluation_labels.resolve()),
                "sha256": sha256_file(args.evaluation_labels),
            },
            "evaluation_manifest": {
                "path": str(args.evaluation_manifest.resolve()),
                "sha256": sha256_file(args.evaluation_manifest),
                "schema_version": evaluation_manifest["schema_version"],
            },
        },
        "prediction_input_contract": list(BLINDED_PREDICTION_COLUMNS),
        "score_direction": "higher_is_more_likely_binder",
        "retention_threshold_for_binary_metrics": RETENTION_THRESHOLD,
        "retention_usage": (
            "sealed retrospective evaluation only; never training, early stopping, "
            "architecture selection, cutoff selection, or hyperparameter selection"
        ),
        "tie_policy": TIE_POLICY,
        "prediction_membership_sha256": prediction_membership,
        "audit_membership_sha256": audit_membership,
        "evaluation_rows": EXPECTED_EVALUATION_ROWS,
        "peptides": EXPECTED_PEPTIDES,
        "affibodies": EXPECTED_AFFIBODIES,
        "unmeasured_matrix_cells": EXPECTED_MISSING_MATRIX_CELLS,
        "models": sorted(set(metrics["model"].astype(str))),
        "analysis_status": "sealed_final_retrospective_evaluation",
        "threshold_was_optimized_on_retention": False,
    }
    publish_evaluation(
        args.output_dir,
        {
            "matched_metrics_by_seed.csv": metrics,
            "matched_metrics_seed_summary.csv": summary,
            "per_peptide_metrics.csv": per_peptide,
            "ranked_per_pair.csv": ranked,
        },
        manifest,
    )


if __name__ == "__main__":
    main()
