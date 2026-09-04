#!/usr/bin/env python3
"""Compile retention-blind LibA handoff manifests from weak-selector receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


LOCK_SCHEMA = "liba-weak-oof-deployment-locks-v1"
PROJECT_SCHEMA = "liba-wetlab-project-spec-v1"
SELECTOR_SCHEMA = "liba-generic-weak-selector-bundle-v1"
EXPECTED_UNIVERSE = 447_731
EXPECTED_DOUBLE_COLD = 330_880
EXPECTED_PEPTIDE_COLD_ONLY = 116_851
EXPECTED_EVIDENCE = 7_786
FORBIDDEN = re.compile(r"retention|binder_ge_75|experimental_outcome", re.I)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def resolve_source_path(raw: str, selector_path: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (selector_path.parent / path).resolve()


def validate_no_outcomes(value, location="root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            allowed_negative_assertion = (
                key == "threshold_uses_retention" and child is False
            )
            require(allowed_negative_assertion or not FORBIDDEN.search(str(key)),
                    f"outcome-derived field forbidden at {location}.{key}")
            validate_no_outcomes(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_no_outcomes(child, f"{location}[{index}]")


def compile_manifests(selector_path: Path, output_dir: Path) -> None:
    require(not output_dir.exists(), "output exists; refusing overwrite")
    bundle = load_json(selector_path)
    validate_no_outcomes(bundle)
    require(bundle.get("schema_version") == SELECTOR_SCHEMA, "unexpected selector bundle schema")
    require(bundle.get("library") == "LibA", "selector bundle is not LibA")
    require(bundle.get("labels") == "weak_selection_only", "selector bundle is not weak-only")
    models = bundle.get("models")
    require(isinstance(models, list) and models, "selector bundle has no models")
    require(len({m.get("model_id") for m in models}) == len(models), "duplicate model IDs")
    eligible = [m for m in models if m.get("scientifically_eligible") is True]
    require(eligible, "selector bundle has no eligible models")
    primary = bundle.get("primary_model_id")
    require(primary in {m["model_id"] for m in eligible}, "primary model is not eligible")
    targets = bundle.get("target_sequences")
    require(isinstance(targets, dict) and len(targets) == 9, "exactly nine targets required")
    require(all(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}", str(v)) for v in targets.values()),
            "target sequences must be valid 9-mers")

    lock_models = {}
    training_rows = {}
    for model in models:
        model_id = str(model.get("model_id", ""))
        require(model_id, "model_id missing")
        if model.get("scientifically_eligible") is not True:
            lock_models[model_id] = {
                "display_name": model.get("display_name", model_id),
                "scientifically_eligible": False,
            }
            continue
        required = {
            "display_name", "lock_id", "score_threshold", "threshold_metric",
            "threshold_provenance", "max_candidates_per_peptide",
            "minimum_code_hamming_distance", "weak_oof_sha256", "head_sha256",
            "config_sha256", "training_rows", "training_data_sha256",
            "weak_oof_within_peptide_average_precision",
            "weak_oof_evaluable_peptides_for_within_peptide_ap",
            "weak_metric_provenance",
        }
        require(required.issubset(model), f"{model_id} selector receipt is incomplete")
        require(model.get("threshold_uses_retention") is False,
                f"{model_id} threshold must explicitly be retention-blind")
        require(math.isfinite(float(model["score_threshold"])), f"{model_id} threshold invalid")
        for key in ("weak_oof_sha256", "head_sha256", "config_sha256", "training_data_sha256"):
            require(re.fullmatch(r"[0-9a-f]{64}", str(model[key])) is not None,
                    f"{model_id} {key} invalid")
        require(int(model["training_rows"]) > 0, f"{model_id} training count invalid")
        weak_ap = float(model["weak_oof_within_peptide_average_precision"])
        weak_groups = int(model["weak_oof_evaluable_peptides_for_within_peptide_ap"])
        require(math.isfinite(weak_ap) and 0 <= weak_ap <= 1,
                f"{model_id} weak within-peptide AP invalid")
        require(weak_groups > 0, f"{model_id} weak AP evaluable-group count invalid")
        provenance = model["weak_metric_provenance"]
        provenance_fields = {
            "generic_lock_path", "generic_lock_sha256", "candidate_metrics_path",
            "candidate_metrics_sha256", "candidate",
        }
        require(isinstance(provenance, dict) and provenance_fields.issubset(provenance),
                f"{model_id} weak metric provenance incomplete")
        source_candidate = str(provenance["candidate"])
        require(source_candidate,
                f"{model_id} weak metric source candidate is empty")
        for path_key, hash_key in (("generic_lock_path", "generic_lock_sha256"),
                                   ("candidate_metrics_path", "candidate_metrics_sha256")):
            source = resolve_source_path(str(provenance[path_key]), selector_path)
            require(source.is_file(), f"{model_id} provenance source missing: {source}")
            require(re.fullmatch(r"[0-9a-f]{64}", str(provenance[hash_key])) is not None,
                    f"{model_id} {hash_key} invalid")
            require(sha256_file(source) == provenance[hash_key],
                    f"{model_id} provenance hash mismatch for {path_key}")
        metrics_path = resolve_source_path(str(provenance["candidate_metrics_path"]), selector_path)
        generic_lock_path = resolve_source_path(str(provenance["generic_lock_path"]), selector_path)
        generic_lock = load_json(generic_lock_path)
        require(str(generic_lock.get("candidate", "")) == source_candidate,
                f"{model_id} generic lock candidate does not match source candidate")
        metrics = pd.read_csv(metrics_path, sep="\t" if metrics_path.suffix == ".tsv" else ",")
        metric_columns = {"candidate", "within_peptide_ap", "within_peptide_evaluable"}
        require(metric_columns.issubset(metrics.columns),
                f"{model_id} candidate metrics schema incomplete")
        match = metrics.loc[metrics["candidate"].astype(str).eq(source_candidate)]
        require(len(match) == 1,
                f"{model_id} source-candidate metrics row missing or duplicated")
        require(abs(float(match.iloc[0]["within_peptide_ap"]) - weak_ap) <= 1e-12,
                f"{model_id} weak AP differs from candidate metrics")
        require(int(match.iloc[0]["within_peptide_evaluable"]) == weak_groups,
                f"{model_id} weak AP group count differs from candidate metrics")
        training_rows[model_id] = int(model["training_rows"])
        lock_models[model_id] = {
            "display_name": model["display_name"],
            "scientifically_eligible": True,
            "lock_id": model["lock_id"],
            "score_threshold": float(model["score_threshold"]),
            "threshold_metric": model["threshold_metric"],
            "threshold_uses_retention": False,
            "threshold_provenance": model["threshold_provenance"],
            "higher_is_better": True,
            "max_candidates_per_peptide": int(model["max_candidates_per_peptide"]),
            "minimum_code_hamming_distance": int(model["minimum_code_hamming_distance"]),
            "pad_below_threshold": False,
            "weak_oof_sha256": model["weak_oof_sha256"],
            "head_sha256": model["head_sha256"],
            "config_sha256": model["config_sha256"],
            "training_data_sha256": model["training_data_sha256"],
            "weak_oof_within_peptide_average_precision": weak_ap,
            "weak_oof_evaluable_peptides_for_within_peptide_ap": weak_groups,
            "weak_metric_provenance": provenance,
            "source_candidate": source_candidate,
        }

    unavailable = bundle.get("unavailable_structure_families")
    require(isinstance(unavailable, list), "unavailable structure audit missing")
    by_family = {str(x.get("family")): x for x in unavailable if isinstance(x, dict)}
    for family in ("RDE-PPI", "StaB-ddG"):
        require(family in by_family and str(by_family[family].get("reason", "")).strip(),
                f"{family} unavailable reason missing")

    lock = {
        "schema_version": LOCK_SCHEMA, "library": "LibA",
        "retention_labels_read": False, "primary_model_id": primary,
        "models": lock_models,
        "compiler_input_sha256": sha256_file(selector_path),
    }
    project = {
        "schema_version": PROJECT_SCHEMA, "library": "LibA",
        "target_sequences": targets,
        "expected_selection_missed_candidate_pairs": EXPECTED_UNIVERSE,
        "expected_double_cold_candidate_pairs": EXPECTED_DOUBLE_COLD,
        "expected_peptide_cold_only_candidate_pairs": EXPECTED_PEPTIDE_COLD_ONLY,
        "training_data": {
            "description": bundle["training_description"],
            "weak_label_definition": bundle["weak_label_definition"],
            "split_rule": bundle["split_rule"],
            "training_rows_by_model": training_rows,
            "training_data_sha256_by_model": {
                m["model_id"]: m["training_data_sha256"] for m in eligible
            },
        },
        "evidence_positive_control": {
            "expected_pairs": EXPECTED_EVIDENCE,
            "max_controls_per_peptide": int(bundle["evidence_max_controls_per_peptide"]),
        },
        "unavailable_structure_families": unavailable,
        "compiler_input_sha256": sha256_file(selector_path),
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        for name, value in (("liba_weak_oof_deployment_locks.json", lock),
                            ("liba_wetlab_project_spec.json", project)):
            path = stage / name
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            os.chmod(path, 0o600)
        os.replace(stage, output_dir)
    except Exception:
        for child in stage.glob("*"):
            child.unlink()
        stage.rmdir()
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    compile_manifests(args.selector_bundle.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
