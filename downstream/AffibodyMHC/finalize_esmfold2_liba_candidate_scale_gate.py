#!/usr/bin/env python3
"""Apply the locked LibA ESMFold2 candidate-scale decision gate.

The feature family is fixed by weak-label cross-validation before this command
reads the retrospective 108-pair evaluation.  Retention is used only for the
prespecified development decision about whether expensive candidate-scale
feature extraction is justified; it is never used to switch feature families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SELECTION_SCHEMA = "esmfold2-liba-replicated-cv-aggregate-v1"
AGGREGATION_COMPLETION_SCHEMA = (
    "esmfold2-liba-replicated-cv-aggregate-completion-v1"
)
EVALUATION_MANIFEST_SCHEMA = "liba-final-model-comparison-v1"
OUTPUT_SCHEMA = "esmfold2-liba-candidate-scale-decision-gate-v1"

FAMILY_TO_EVALUATION_MODEL = {
    "distogram_only": "esmfold2_distogram_only",
    "pair_state_only": "esmfold2_pair_state_only",
    "distogram_pair": "esmfold2_distogram_pair",
    "single_inputs_only": "esmfold2_single_inputs_control",
    "full": "esmfold2_full",
}
STRUCTURE_DERIVED_FAMILIES = {
    "distogram_only",
    "pair_state_only",
    "distogram_pair",
    "full",
}
SEQUENCE_CONTROL_IDS = {
    "additive_6site",
    "mint_l9",
    "mint_l33_control",
    "nonlinear_6site_mean_logit",
    "equal_logit_additive_mint_l9",
}
FINAL_SEEDS = {"20260811", "20260812", "20260813", "20260814", "20260815"}
EXPECTED_ROWS = 108
EXPECTED_BINARY_GROUPS = 5
EXPECTED_SPEARMAN_GROUPS = 9
EXPECTED_BEST_SEQUENCE_AP = 0.600250426341
EXPECTED_BEST_SEQUENCE_SPEARMAN = -0.303807303807
TOLERANCE = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"JSON input is absent: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"JSON input must be an object: {path}")
    return value


def _validate_receipt(record: Any, path: Path, label: str) -> None:
    _require(isinstance(record, Mapping), f"{label} receipt is absent")
    _require(Path(str(record.get("path", ""))).resolve() == path.resolve(), f"{label} path changed")
    _require(record.get("sha256") == _sha256_file(path), f"{label} checksum changed")


def _metric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise")
    _require(bool(values.map(math.isfinite).all()), f"{column} contains non-finite values")
    return values


def run(
    selection_path: Path,
    aggregation_completion_path: Path,
    evaluation_dir: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    selection_path = selection_path.resolve()
    aggregation_completion_path = aggregation_completion_path.resolve()
    evaluation_dir = evaluation_dir.resolve()
    output_dir = output_dir.resolve()

    selection = _read_json(selection_path)
    completion = _read_json(aggregation_completion_path)
    _require(selection.get("schema_version") == SELECTION_SCHEMA, "weak selection schema changed")
    _require(selection.get("retention_labels_read") is False, "weak selection read retention")
    _require(
        completion.get("schema_version") == AGGREGATION_COMPLETION_SCHEMA,
        "weak aggregation completion schema changed",
    )
    _require(completion.get("retention_labels_read") is False, "weak aggregation read retention")
    _validate_receipt(
        completion.get("outputs", {}).get("all_model_selection"),
        selection_path,
        "weak selection",
    )

    selected_family = str(selection.get("selected_feature_family", ""))
    _require(selected_family in FAMILY_TO_EVALUATION_MODEL, "selected weak feature family changed")
    selected_model = FAMILY_TO_EVALUATION_MODEL[selected_family]
    role = selection.get("family_roles", {}).get(selected_family)
    model_record = selection.get("models", {}).get(selected_family)
    _require(isinstance(model_record, Mapping), "selected family weak record is absent")
    weak_reproducible_vs_mint = bool(model_record.get("reproducibly_beats_mint_layer9"))
    weak_passes_pretrunk = bool(model_record.get("passes_pretrunk_control"))
    weak_supports_folding_claim = bool(model_record.get("supports_structure_claim"))
    weak_gate_passed = bool(
        selected_family in STRUCTURE_DERIVED_FAMILIES
        and role == "folding_derived"
        and weak_reproducible_vs_mint
        and weak_passes_pretrunk
        and weak_supports_folding_claim
    )

    manifest_path = evaluation_dir / "manifest.json"
    deployment_path = evaluation_dir / "deployment_metrics.csv"
    raw_path = evaluation_dir / "raw_metrics_by_seed.csv"
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == EVALUATION_MANIFEST_SCHEMA, "evaluation manifest schema changed")
    _require(
        manifest.get("retention_usage", {}).get("final_retrospective_evaluation") is True,
        "evaluation manifest does not declare retrospective retention use",
    )
    _require(
        manifest.get("retention_usage", {}).get("model_selection") is False,
        "evaluation manifest declares retention-based model selection",
    )
    _validate_receipt(manifest.get("outputs", {}).get("deployment_metrics.csv"), deployment_path, "deployment metrics")
    _validate_receipt(manifest.get("outputs", {}).get("raw_metrics_by_seed.csv"), raw_path, "raw seed metrics")

    deployment = pd.read_csv(deployment_path, dtype={"model": str, "seed": str})
    raw = pd.read_csv(raw_path, dtype={"model": str, "seed": str})
    required = {
        "model",
        "seed",
        "n_examples",
        "within_peptide_binary_evaluable_groups",
        "within_peptide_spearman_evaluable_groups",
        "within_peptide_average_precision_mean",
        "within_peptide_spearman_mean",
    }
    _require(required.issubset(deployment.columns), "deployment metrics lack required columns")
    _require(required.issubset(raw.columns), "raw seed metrics lack required columns")

    controls = deployment.loc[deployment["model"].isin(SEQUENCE_CONTROL_IDS)].copy()
    _require(set(controls["model"]) == SEQUENCE_CONTROL_IDS, "sequence-control set changed")
    control_ap = _metric(controls, "within_peptide_average_precision_mean")
    control_spearman = _metric(controls, "within_peptide_spearman_mean")
    best_ap = float(control_ap.max())
    best_spearman = float(control_spearman.max())
    _require(
        math.isclose(best_ap, EXPECTED_BEST_SEQUENCE_AP, rel_tol=0.0, abs_tol=TOLERANCE),
        "best sequence-control within-peptide AP changed",
    )
    _require(
        math.isclose(
            best_spearman,
            EXPECTED_BEST_SEQUENCE_SPEARMAN,
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ),
        "best sequence-control within-peptide Spearman changed",
    )

    deployment_selected = deployment.loc[deployment["model"].eq(selected_model)].copy()
    _require(len(deployment_selected) == 1, "selected family lacks exactly one deployment row")
    deployment_row = deployment_selected.iloc[0]
    _require(str(deployment_row["seed"]) == "deployment", "selected family deployment rule changed")
    _require(int(deployment_row["n_examples"]) == EXPECTED_ROWS, "evaluation row count changed")
    _require(
        int(deployment_row["within_peptide_binary_evaluable_groups"]) == EXPECTED_BINARY_GROUPS,
        "binary evaluable-peptide count changed",
    )
    _require(
        int(deployment_row["within_peptide_spearman_evaluable_groups"]) == EXPECTED_SPEARMAN_GROUPS,
        "Spearman evaluable-peptide count changed",
    )
    deployment_ap = float(deployment_row["within_peptide_average_precision_mean"])
    deployment_spearman = float(deployment_row["within_peptide_spearman_mean"])
    _require(math.isfinite(deployment_ap), "selected deployment AP is non-finite")
    _require(math.isfinite(deployment_spearman), "selected deployment Spearman is non-finite")

    # The finalizer preserves source-family names for individual seeds and
    # applies the public evaluation-model IDs only to deployment aggregates.
    selected_raw = raw.loc[raw["model"].eq(selected_family)].copy()
    _require(len(selected_raw) == 5, "selected family does not have five final seeds")
    _require(set(selected_raw["seed"]) == FINAL_SEEDS, "selected family final seed set changed")
    _require(bool(selected_raw["n_examples"].astype(int).eq(EXPECTED_ROWS).all()), "raw evaluation row count changed")
    _require(
        bool(selected_raw["within_peptide_binary_evaluable_groups"].astype(int).eq(EXPECTED_BINARY_GROUPS).all()),
        "raw binary evaluable-peptide count changed",
    )
    _require(
        bool(selected_raw["within_peptide_spearman_evaluable_groups"].astype(int).eq(EXPECTED_SPEARMAN_GROUPS).all()),
        "raw Spearman evaluable-peptide count changed",
    )
    raw_ap = _metric(selected_raw, "within_peptide_average_precision_mean")
    raw_spearman = _metric(selected_raw, "within_peptide_spearman_mean")

    deployment_beats_ap = deployment_ap > best_ap + TOLERANCE
    deployment_beats_spearman = deployment_spearman > best_spearman + TOLERANCE
    seeds_beating_ap = int((raw_ap > best_ap + TOLERANCE).sum())
    seeds_beating_spearman = int((raw_spearman > best_spearman + TOLERANCE).sum())
    seeds_beating_both = int(
        ((raw_ap > best_ap + TOLERANCE) & (raw_spearman > best_spearman + TOLERANCE)).sum()
    )
    retention_gate_passed = bool(
        deployment_beats_ap
        and deployment_beats_spearman
        and seeds_beating_ap >= 4
        and seeds_beating_spearman >= 4
    )
    candidate_scale_go = bool(weak_gate_passed and retention_gate_passed)

    record = {
        "schema_version": OUTPUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd().resolve()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "decision": "GO" if candidate_scale_go else "NO_GO",
        "candidate_scale_esmfold2_extraction_authorized": candidate_scale_go,
        "selected_feature_family_locked_before_retention": selected_family,
        "selected_evaluation_model": selected_model,
        "selected_family_role": role,
        "retention_driven_family_switching_allowed": False,
        "weak_label_gate": {
            "passed": weak_gate_passed,
            "reproducibly_beats_mint_layer9": weak_reproducible_vs_mint,
            "passes_pretrunk_control": weak_passes_pretrunk,
            "supports_folding_derived_claim": weak_supports_folding_claim,
        },
        "retrospective_retention_gate": {
            "passed": retention_gate_passed,
            "purpose": "prespecified development decision about candidate-scale extraction",
            "not_a_retention_blind_model_selection_result": True,
            "deployment_within_peptide_ap": deployment_ap,
            "best_sequence_control_within_peptide_ap": best_ap,
            "deployment_beats_ap": deployment_beats_ap,
            "individual_seeds_beating_ap": seeds_beating_ap,
            "deployment_within_peptide_spearman": deployment_spearman,
            "best_sequence_control_within_peptide_spearman": best_spearman,
            "deployment_beats_spearman": deployment_beats_spearman,
            "individual_seeds_beating_spearman": seeds_beating_spearman,
            "individual_seeds_beating_both": seeds_beating_both,
            "required_individual_seeds_per_metric": 4,
            "four_of_five_rule_is_a_prespecified_stability_heuristic_not_a_significance_test": True,
        },
        "inputs": {
            "weak_selection": {"path": str(selection_path), "sha256": _sha256_file(selection_path)},
            "weak_aggregation_completion": {
                "path": str(aggregation_completion_path),
                "sha256": _sha256_file(aggregation_completion_path),
            },
            "evaluation_manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "deployment_metrics": {"path": str(deployment_path), "sha256": _sha256_file(deployment_path)},
            "raw_metrics_by_seed": {"path": str(raw_path), "sha256": _sha256_file(raw_path)},
        },
    }
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True)
    output_path = output_dir / "decision_gate.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(output_path, 0o600)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-selection", type=Path, required=True)
    parser.add_argument("--weak-aggregation-completion", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record = run(
        args.weak_selection,
        args.weak_aggregation_completion,
        args.evaluation_dir,
        args.output_dir,
    )
    print(json.dumps({"decision": record["decision"], "selected_feature_family": record["selected_feature_family_locked_before_retention"]}, sort_keys=True))


if __name__ == "__main__":
    main()
