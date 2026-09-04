#!/usr/bin/env python
"""Retention-blind completeness check for LibB structural score files.

This command deliberately reads only the label-free canonical row inventory,
the two structural readout configs, and the emitted blinded score CSVs.  It
does not accept or discover a retention-label path.  A successful exit means
that every configured model has exactly the prescribed five opaque 120-row
score files and that the duplicated sequence control is identical across the
RDE and StaB runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BLINDED_COLUMNS = ("eval_row_id", "model", "seed", "score")
CONTROL_MODEL = "nonlinear_7site_control"

DEFAULT_CANONICAL_ROWS = (
    REPO_ROOT
    / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json"
)
DEFAULT_RDE_CONFIG = (
    REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_frozen_readouts_v1.json"
)
DEFAULT_STAB_CONFIG = (
    REPO_ROOT / "downstream/AffibodyMHC/configs/stab_libb_frozen_readouts_v1.json"
)
DEFAULT_RDE_ROOT = (
    REPO_ROOT
    / "private_data/experiments/rde_libb_provider_revision_120_full_readouts_v2_common_env"
)
DEFAULT_STAB_ROOT = (
    REPO_ROOT
    / "private_data/experiments/stab_libb_provider_revision_120_readout_full_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _configured_models(config: dict[str, Any]) -> list[str]:
    models = [str(spec["name"]) for spec in config.get("models", [])]
    models.extend(str(spec["name"]) for spec in config.get("prediction_ensembles", []))
    if len(models) != len(set(models)):
        raise ValueError("configured model names are not unique")
    return models


def _read_prediction(path: Path, expected_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = list(reader)

    if columns != BLINDED_COLUMNS:
        errors.append(f"schema={columns!r}; expected={BLINDED_COLUMNS!r}")

    ids = [str(row.get("eval_row_id", "")) for row in rows]
    models = {str(row.get("model", "")) for row in rows}
    seeds = {str(row.get("seed", "")) for row in rows}
    if len(rows) != len(expected_ids):
        errors.append(f"rows={len(rows)}; expected={len(expected_ids)}")
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate_eval_row_ids={duplicate_ids}")
    observed_ids = set(ids)
    if observed_ids != expected_ids:
        errors.append(
            "opaque_id_set_mismatch: missing={}, extra={}".format(
                len(expected_ids - observed_ids), len(observed_ids - expected_ids)
            )
        )
    if len(models) != 1:
        errors.append(f"model_values={sorted(models)!r}")
    if len(seeds) != 1:
        errors.append(f"seed_values={sorted(seeds)!r}")

    scores: list[float] = []
    try:
        scores = [float(row.get("score", "nan")) for row in rows]
    except (TypeError, ValueError):
        errors.append("score_parse_failure")
    if scores and not all(math.isfinite(value) for value in scores):
        errors.append("nonfinite_score")

    model = next(iter(models)) if len(models) == 1 else None
    seed_text = next(iter(seeds)) if len(seeds) == 1 else None
    try:
        seed = int(seed_text) if seed_text is not None else None
    except ValueError:
        seed = None
        errors.append(f"invalid_seed={seed_text!r}")

    match = re.fullmatch(r"(.+)__seed(\d+)\.csv", path.name)
    if (
        match is None
        or model is None
        or seed is None
        or match.group(1) != model
        or int(match.group(2)) != seed
    ):
        errors.append("filename_model_seed_mismatch")

    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "columns": list(columns),
        "rows": len(rows),
        "unique_eval_row_ids": len(observed_ids),
        "model": model,
        "seed": seed,
        "errors": errors,
        "_scores_by_id": dict(zip(ids, scores)) if len(ids) == len(scores) else {},
    }


def _audit_family(
    name: str,
    root: Path,
    config_path: Path,
    expected_ids: set[str],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    config = _load_json(config_path)
    models = _configured_models(config)
    prescribed_seeds = [int(value) for value in config["final_training"]["seeds"]]
    if len(prescribed_seeds) != len(set(prescribed_seeds)):
        raise ValueError(f"{name} prescribed seeds are not unique")

    records: dict[tuple[str, int], dict[str, Any]] = {}
    invalid_files: list[dict[str, Any]] = []
    unexpected_files: list[dict[str, Any]] = []
    duplicate_groups: list[dict[str, Any]] = []
    paths = sorted(root.glob("*/blinded_predictions/*.csv"))
    output_directories = sorted({path.parent.parent for path in paths})
    unfinished_output_directories = [
        str(path.resolve())
        for path in output_directories
        if not (path / "final_training_summary.json").is_file()
    ]
    for path in paths:
        record = _read_prediction(path, expected_ids)
        public_record = {
            field: value for field, value in record.items() if not field.startswith("_")
        }
        model = record["model"]
        seed = record["seed"]
        if model not in models or seed not in prescribed_seeds:
            unexpected_files.append(public_record)
            continue
        key = (model, seed)
        if key in records:
            duplicate_groups.append({"first": records[key]["path"], "second": record["path"]})
        records[key] = record
        if record["errors"]:
            invalid_files.append(public_record)

    by_model: dict[str, Any] = {}
    for model in models:
        present = sorted(seed for seed in prescribed_seeds if (model, seed) in records)
        valid = sorted(
            seed
            for seed in prescribed_seeds
            if (model, seed) in records and not records[(model, seed)]["errors"]
        )
        by_model[model] = {
            "present_seeds": present,
            "valid_seeds": valid,
            "missing_seeds": sorted(set(prescribed_seeds) - set(present)),
            "invalid_seeds": sorted(set(present) - set(valid)),
            "complete": valid == sorted(prescribed_seeds),
        }

    complete = (
        all(value["complete"] for value in by_model.values())
        and not invalid_files
        and not unexpected_files
        and not duplicate_groups
        and not unfinished_output_directories
    )
    clean_records = {
        key: {field: value for field, value in record.items() if not field.startswith("_")}
        for key, record in records.items()
    }
    return (
        {
            "family": name,
            "root": str(root.resolve()),
            "config": str(config_path.resolve()),
            "configured_models": models,
            "prescribed_seeds": prescribed_seeds,
            "models": by_model,
            "prediction_files_seen": len(paths),
            "invalid_files": invalid_files,
            "unexpected_files": unexpected_files,
            "duplicate_groups": duplicate_groups,
            "unfinished_output_directories": unfinished_output_directories,
            "files": [clean_records[key] for key in sorted(clean_records)],
            "complete": complete,
        },
        records,
    )


def _control_identity(
    rde_records: dict[tuple[str, int], dict[str, Any]],
    stab_records: dict[tuple[str, int], dict[str, Any]],
    seeds: list[int],
) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    all_complete = True
    all_identical = True
    for seed in seeds:
        rde = rde_records.get((CONTROL_MODEL, seed))
        stab = stab_records.get((CONTROL_MODEL, seed))
        complete = rde is not None and stab is not None and not rde["errors"] and not stab["errors"]
        identical = False
        max_abs_difference = None
        if complete:
            rde_scores = rde["_scores_by_id"]
            stab_scores = stab["_scores_by_id"]
            identical = rde_scores == stab_scores and rde["sha256"] == stab["sha256"]
            max_abs_difference = max(
                abs(rde_scores[row_id] - stab_scores[row_id]) for row_id in rde_scores
            )
        all_complete = all_complete and complete
        all_identical = all_identical and identical
        by_seed[str(seed)] = {
            "complete": complete,
            "byte_and_score_identical": identical,
            "max_abs_score_difference": max_abs_difference,
            "rde_sha256": None if rde is None else rde["sha256"],
            "stab_sha256": None if stab is None else stab["sha256"],
        }
    return {
        "model": CONTROL_MODEL,
        "by_seed": by_seed,
        "complete": all_complete,
        "identical": all_complete and all_identical,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-rows", type=Path, default=DEFAULT_CANONICAL_ROWS)
    parser.add_argument("--rde-config", type=Path, default=DEFAULT_RDE_CONFIG)
    parser.add_argument("--rde-root", type=Path, default=DEFAULT_RDE_ROOT)
    parser.add_argument("--stab-config", type=Path, default=DEFAULT_STAB_CONFIG)
    parser.add_argument("--stab-root", type=Path, default=DEFAULT_STAB_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    canonical = _load_json(args.canonical_rows)
    expected_ids = {
        str(row["row_id"]) for row in canonical["rows"] if str(row["split"]) == "eval"
    }
    if len(expected_ids) != 120:
        raise ValueError(f"canonical corrected evaluation membership is {len(expected_ids)}, not 120")

    rde, rde_records = _audit_family("RDE", args.rde_root, args.rde_config, expected_ids)
    stab, stab_records = _audit_family("StaB", args.stab_root, args.stab_config, expected_ids)
    rde_seeds = rde["prescribed_seeds"]
    if rde_seeds != stab["prescribed_seeds"]:
        raise ValueError("RDE and StaB prescribed seed lists differ")
    control = _control_identity(rde_records, stab_records, rde_seeds)
    complete = bool(rde["complete"] and stab["complete"] and control["identical"])
    report = {
        "schema_version": "libb-structural-blinded-preflight-v1",
        "retention_blind": True,
        "inputs_read": {
            "canonical_rows": str(args.canonical_rows.resolve()),
            "canonical_rows_sha256": _sha256(args.canonical_rows),
            "rde_config": str(args.rde_config.resolve()),
            "rde_config_sha256": _sha256(args.rde_config),
            "stab_config": str(args.stab_config.resolve()),
            "stab_config_sha256": _sha256(args.stab_config),
            "blinded_prediction_roots": [
                str(args.rde_root.resolve()),
                str(args.stab_root.resolve()),
            ],
        },
        "forbidden_inputs_read": [],
        "expected_evaluation_rows": len(expected_ids),
        "exact_prediction_schema": list(BLINDED_COLUMNS),
        "families": {"RDE": rde, "StaB": stab},
        "common_environment_control_identity": control,
        "complete_and_safe_to_unseal": complete,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
