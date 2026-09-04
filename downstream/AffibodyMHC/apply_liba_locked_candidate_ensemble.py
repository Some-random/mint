#!/usr/bin/env python3
"""Apply a weak-label-only LibA single/equal-logit lock to all candidates.

The deployment score is calculated from true component logits named by an
``affibody-weak-oof-score-lock-v1`` file.  Before scoring 447,731 candidates,
the formula must exactly reproduce its matched weak OOF score column and must
give identical values from the two independently shaped, label-free 108-pair
evaluation tables.

Nonnegative stackers are deliberately rejected for deployment: their stored
OOF scores are outer-cross-fitted, while their inference formula is a separate
fit on all OOF rows.  A cutoff selected on the former cannot be exact-parity
validated on the latter without an additional calibration contract.

No direct-retention values or retention-derived binder labels are accepted or
read.  The output is a selection-label score, not a retention percentage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
LIBRARY = "LibA"
LOCK_SCHEMA_VERSION = "affibody-weak-oof-score-lock-v1"
OUTPUT_SCHEMA_VERSION = "liba-locked-candidate-ensemble-scores-v1"
EXPECTED_CANDIDATE_ROWS = 447_731
EXPECTED_EVALUATION_ROWS = 108
EXPECTED_PEPTIDE_COUNT = 9
# OOF probabilities are serialized as decimal CSV values.  Five trillionths
# is tight enough to detect a changed formula while accommodating that single
# text round trip (the production equal-logit replay differs by 2.12e-12).
PARITY_TOLERANCE = 5e-12
SAFE_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

STANDARD_COLUMNS = (
    "model_id",
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "model_score",
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)
ALIGNMENT_COLUMNS = tuple(column for column in STANDARD_COLUMNS if column not in {"model_id", "model_score"})
FORBIDDEN_OUTCOME_PARTS = (
    "retention",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
    "target_binder",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("ascii")).hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    display = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {"path": display, "bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved)}


def _read_json(path: Path) -> dict[str, Any]:
    _require(Path(path).is_file(), f"missing JSON file: {path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _private_new_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must remain below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not resolved.exists(), f"output exists; refusing overwrite: {resolved}")
    return resolved


def _forbidden_columns(columns: Iterable[object]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if any(part in str(column).strip().lower() for part in FORBIDDEN_OUTCOME_PARTS)
    )


def sigmoid(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def clipped_logit(values: Sequence[float], clip: float = 1e-7) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    _require(bool(np.isfinite(probability).all()), "non-finite component probability")
    _require(bool(((probability >= 0.0) & (probability <= 1.0)).all()),
             "component probability outside [0,1]")
    probability = np.clip(probability, clip, 1.0 - clip)
    return np.log(probability) - np.log1p(-probability)


def parse_named_paths(values: Sequence[str], purpose: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for raw in values:
        _require("=" in raw, f"{purpose} must be NAME=PATH: {raw}")
        name, path_text = raw.split("=", 1)
        name = name.strip()
        path = Path(path_text).expanduser().resolve()
        _require(bool(name) and name not in output, f"duplicate/empty {purpose} name: {name}")
        _require(path.is_file(), f"missing {purpose} file: {path}")
        output[name] = path
    return output


def load_lock(path: Path) -> dict[str, Any]:
    lock = _read_json(path)
    _require(lock.get("schema_version") == LOCK_SCHEMA_VERSION, "ensemble-lock schema changed")
    _require(lock.get("library") == LIBRARY, "ensemble lock is not LibA")
    _require(lock.get("retention_labels_read") is False, "ensemble lock does not forbid retention")
    family = str(lock.get("candidate_family", ""))
    _require(
        family in {"single", "equal_mean_logit"},
        (
            "only single/equal-logit locks are deployable; a cross-fitted stacker "
            "uses a different final inference fit and its OOF cutoff lacks exact score parity"
        ),
    )
    formula = lock.get("formula")
    _require(isinstance(formula, dict), "ensemble lock lacks a formula")
    _require(formula.get("input_semantics") == "true component logits", "lock input semantics changed")
    _require(formula.get("output_transform") == "sigmoid", "lock output transform changed")
    components = formula.get("components")
    _require(isinstance(components, list) and components, "lock formula has no components")
    names = [str(component.get("name", "")) for component in components]
    _require(names == list(map(str, lock.get("members", []))), "lock component order differs from members")
    _require(len(names) == len(set(names)), "duplicate lock component")
    expected_weight = 1.0 / len(components)
    for component in components:
        _require(bool(component.get("inference_logit_column")), "component logit column is missing")
        _require(float(component.get("center", math.nan)) == 0.0, "equal-logit center must be zero")
        _require(float(component.get("scale", math.nan)) == 1.0, "equal-logit scale must be one")
        _require(abs(float(component.get("weight", math.nan)) - expected_weight) <= 1e-15,
                 "equal-logit component weight changed")
    _require(float(formula.get("intercept", math.nan)) == 0.0, "equal-logit intercept must be zero")
    cutoff = lock.get("cutoff", {})
    _require(cutoff.get("decision_rule") == "score >= threshold", "lock cutoff rule changed")
    threshold = cutoff.get("score_threshold")
    _require(isinstance(threshold, (int, float)) and math.isfinite(float(threshold)),
             "lock score threshold is invalid")
    return lock


def apply_formula(lock: Mapping[str, Any], logits: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    formula = lock["formula"]
    components = formula["components"]
    missing = [str(component["name"]) for component in components if str(component["name"]) not in logits]
    _require(not missing, f"missing component logits: {missing}")
    lengths: set[int] = set()
    linear = None
    for component in components:
        name = str(component["name"])
        value = np.asarray(logits[name], dtype=np.float64)
        _require(value.ndim == 1 and len(value) > 0, f"bad component logit vector for {name}")
        _require(bool(np.isfinite(value).all()), f"non-finite component logits for {name}")
        lengths.add(len(value))
        contribution = float(component["weight"]) * (
            value - float(component["center"])
        ) / float(component["scale"])
        linear = contribution if linear is None else linear + contribution
    _require(len(lengths) == 1 and linear is not None, "component logit lengths differ")
    linear = np.asarray(linear, dtype=np.float64) + float(formula["intercept"])
    score = sigmoid(linear)
    _require(bool(np.isfinite(score).all()), "non-finite ensemble score")
    return linear, score


def _manifest_score_receipt(score_path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        score_path.parent / "manifest.json",
        score_path.with_suffix(score_path.suffix + ".manifest.json"),
    )
    existing = [path for path in candidates if path.is_file()]
    _require(len(existing) == 1, f"expected exactly one sibling manifest for {score_path}")
    path = existing[0]
    manifest = _read_json(path)
    expected_hash = sha256_file(score_path)
    receipts: list[Mapping[str, Any]] = []
    for key in ("output", "combined_candidate_scores"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            receipts.append(value)
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        for value in outputs.values():
            if isinstance(value, Mapping):
                receipts.append(value)
    _require(any(value.get("sha256") == expected_hash for value in receipts),
             f"component manifest does not attest score hash: {score_path}")
    return path, manifest


def load_component_scores(
    name: str,
    path: Path,
    component: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    _require(not _forbidden_columns(columns), f"{name} score file contains outcome columns")
    logit_column = str(component["inference_logit_column"])
    required = {*ALIGNMENT_COLUMNS, "model_id", "model_score", logit_column}
    _require(required.issubset(columns), f"{name} score file lacks {sorted(required.difference(columns))}")
    _require(parquet.metadata.num_rows == EXPECTED_CANDIDATE_ROWS,
             f"{name} candidate row count changed")
    frame = pd.read_parquet(path, columns=list(required))
    _require(len(frame) == EXPECTED_CANDIDATE_ROWS, f"{name} candidate row count changed")
    _require(frame["pair_uid"].astype(str).is_unique, f"duplicate {name} candidate pair UID")
    _require(frame["peptide_design_code"].astype(str).nunique() == EXPECTED_PEPTIDE_COUNT,
             f"{name} candidate peptide count changed")
    for flag in (
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
    ):
        _require(pd.api.types.is_bool_dtype(frame[flag]) and frame[flag].notna().all(),
                 f"{name} {flag} is not complete Boolean data")
    score = pd.to_numeric(frame["model_score"], errors="raise").to_numpy(dtype=np.float64)
    logit = pd.to_numeric(frame[logit_column], errors="raise").to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(score).all() and np.isfinite(logit).all()), f"non-finite {name} score")
    _require(bool(((score >= 0.0) & (score <= 1.0)).all()), f"{name} score outside [0,1]")
    maximum = float(np.max(np.abs(sigmoid(logit) - score)))
    _require(maximum <= PARITY_TOLERANCE, f"{name} score/logit parity failed: {maximum}")
    manifest_path, manifest = _manifest_score_receipt(path)
    return frame, {
        "name": name,
        "model_ids": sorted(frame["model_id"].astype(str).unique().tolist()),
        "score": _record(path),
        "manifest": _record(manifest_path),
        "manifest_schema_version": manifest.get("schema_version"),
        "logit_column": logit_column,
        "score_logit_maximum_absolute_difference": maximum,
        "pair_uid_membership_sha256": membership_sha256(frame["pair_uid"]),
    }


def verify_oof_formula(lock: Mapping[str, Any], path: Path) -> dict[str, Any]:
    candidate = str(lock["candidate"])
    members = list(map(str, lock["members"]))
    header = pd.read_csv(path, nrows=0)
    _require(not _forbidden_columns(header.columns), "matched OOF contains direct-outcome columns")
    required = {"row_id", candidate, *members}
    _require(required.issubset(header.columns), "matched OOF lacks formula columns")
    frame = pd.read_csv(path, usecols=list(required), dtype={"row_id": str})
    _require(frame["row_id"].is_unique and len(frame) > 0, "matched OOF row IDs changed")
    logits = {member: clipped_logit(frame[member]) for member in members}
    _, replay = apply_formula(lock, logits)
    expected = pd.to_numeric(frame[candidate], errors="raise").to_numpy(dtype=np.float64)
    maximum = float(np.max(np.abs(replay - expected)))
    _require(maximum <= PARITY_TOLERANCE, f"weak-OOF formula parity failed: {maximum}")
    return {
        "status": "passed",
        "rows": int(len(frame)),
        "candidate_score_column": candidate,
        "maximum_absolute_score_difference": maximum,
        "tolerance": PARITY_TOLERANCE,
        "source": _record(path),
        "columns_read": ["row_id", *members, candidate],
        "weak_label_column_read": False,
        "retention_columns_read": [],
    }


def _long_evaluation_components(path: Path, members: Sequence[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    _require(not _forbidden_columns(header.columns), "long evaluation table contains outcome columns")
    required = {"eval_row_id", "model", "seed", "score"}
    _require(required.issubset(header.columns), "long evaluation table lacks required columns")
    frame = pd.read_csv(path, usecols=list(required), dtype={"eval_row_id": str, "model": str, "seed": str})
    blocks = []
    for member in members:
        block = frame.loc[frame["model"].eq(member)].copy()
        if len(block) == EXPECTED_EVALUATION_ROWS and block["eval_row_id"].is_unique:
            _require(block["seed"].eq("fixed").all(), f"{member} is not a deterministic fixed component")
            value = block[["eval_row_id", "score"]].rename(columns={"score": member})
        else:
            # The five-seed nonlinear component is stored long under
            # ``nonlinear_6site`` and aligned under the explicit aggregate
            # name ``nonlinear_6site_mean_logit``.  Rebuild that aggregate
            # independently as sigmoid(mean(seed logits)).
            _require(member.endswith("_mean_logit"),
                     f"long evaluation coverage changed for {member}")
            base_model = member[: -len("_mean_logit")]
            block = frame.loc[frame["model"].eq(base_model)].copy()
            _require(len(block) > EXPECTED_EVALUATION_ROWS,
                     f"long evaluation lacks seed rows for {member}")
            _require(not block.duplicated(["eval_row_id", "seed"]).any(),
                     f"duplicate long evaluation seed row for {member}")
            pivot = block.pivot(index="eval_row_id", columns="seed", values="score")
            _require(len(pivot) == EXPECTED_EVALUATION_ROWS and pivot.notna().all().all(),
                     f"incomplete long evaluation seed matrix for {member}")
            _require(pivot.shape[1] >= 2, f"too few long evaluation seeds for {member}")
            aggregate = sigmoid(
                np.column_stack(
                    [clipped_logit(pivot[column]) for column in sorted(pivot.columns)]
                ).mean(axis=1)
            )
            value = pd.DataFrame(
                {"eval_row_id": pivot.index.astype(str), member: aggregate}
            )
        blocks.append(value)
    output = blocks[0]
    for block in blocks[1:]:
        output = output.merge(block, on="eval_row_id", validate="one_to_one")
    return output.sort_values("eval_row_id", kind="stable").reset_index(drop=True)


def verify_evaluation_formula(
    lock: Mapping[str, Any], aligned_path: Path, long_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    members = list(map(str, lock["members"]))
    header = pd.read_csv(aligned_path, nrows=0)
    _require(not _forbidden_columns(header.columns), "aligned evaluation table contains outcome columns")
    required = {"eval_row_id", *members}
    _require(required.issubset(header.columns), "aligned evaluation table lacks components")
    aligned = pd.read_csv(aligned_path, usecols=list(required), dtype={"eval_row_id": str})
    _require(len(aligned) == EXPECTED_EVALUATION_ROWS and aligned["eval_row_id"].is_unique,
             "aligned evaluation roster changed")
    aligned = aligned.sort_values("eval_row_id", kind="stable").reset_index(drop=True)
    long = _long_evaluation_components(long_path, members)
    _require(aligned["eval_row_id"].equals(long["eval_row_id"]), "evaluation row IDs differ across layouts")
    component_delta: dict[str, float] = {}
    for member in members:
        left = pd.to_numeric(aligned[member], errors="raise").to_numpy(dtype=np.float64)
        right = pd.to_numeric(long[member], errors="raise").to_numpy(dtype=np.float64)
        maximum = float(np.max(np.abs(left - right)))
        _require(maximum <= PARITY_TOLERANCE, f"evaluation component parity failed for {member}: {maximum}")
        component_delta[member] = maximum
    _, score_aligned = apply_formula(
        lock, {member: clipped_logit(aligned[member]) for member in members}
    )
    logit_long, score_long = apply_formula(
        lock, {member: clipped_logit(long[member]) for member in members}
    )
    formula_maximum = float(np.max(np.abs(score_aligned - score_long)))
    _require(formula_maximum <= PARITY_TOLERANCE,
             f"evaluation ensemble formula parity failed: {formula_maximum}")
    output = pd.DataFrame(
        {
            "eval_row_id": long["eval_row_id"].astype(str),
            "model_score": score_long,
            "ensemble_logit": logit_long,
        }
    )
    return output, {
        "status": "passed",
        "rows": EXPECTED_EVALUATION_ROWS,
        "component_maximum_absolute_differences": component_delta,
        "ensemble_maximum_absolute_score_difference": formula_maximum,
        "tolerance": PARITY_TOLERANCE,
        "aligned_source": _record(aligned_path),
        "long_source": _record(long_path),
        "outcome_columns_read": [],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    model_id = str(args.model_id)
    _require(bool(SAFE_MODEL_ID.fullmatch(model_id)), "invalid output model ID")
    output_dir = _private_new_directory(args.output_dir)
    lock_path = Path(args.lock_json).resolve()
    lock = load_lock(lock_path)
    head_binding_path = Path(args.head_binding).resolve()
    config_binding_path = Path(args.config_binding).resolve()
    weak_oof_binding_path = Path(args.weak_oof_binding).resolve()
    for purpose, path in (
        ("deployment head/formula binding", head_binding_path),
        ("deployment config binding", config_binding_path),
        ("weak OOF binding", weak_oof_binding_path),
    ):
        _require(path.is_file(), f"missing {purpose}: {path}")
    components_by_name = {str(value["name"]): value for value in lock["formula"]["components"]}
    score_paths = parse_named_paths(args.component_score, "component score")
    _require(set(score_paths) == set(components_by_name),
             "component score names must exactly match the lock formula")

    # OOF and target-free evaluation formula parity must pass before candidate
    # files are opened or an output directory is created.
    oof_parity = verify_oof_formula(lock, Path(args.matched_oof).resolve())
    evaluation_predictions, evaluation_parity = verify_evaluation_formula(
        lock,
        Path(args.evaluation_aligned).resolve(),
        Path(args.evaluation_long).resolve(),
    )

    frames: dict[str, pd.DataFrame] = {}
    component_records: dict[str, Any] = {}
    for name in lock["members"]:
        frame, record = load_component_scores(
            str(name), score_paths[str(name)], components_by_name[str(name)]
        )
        frames[str(name)] = frame
        component_records[str(name)] = record
    first_name = str(lock["members"][0])
    base = frames[first_name].sort_values("pair_uid", kind="stable").reset_index(drop=True)
    for name in map(str, lock["members"][1:]):
        current = frames[name].sort_values("pair_uid", kind="stable").reset_index(drop=True)
        for column in ALIGNMENT_COLUMNS:
            _require(base[column].equals(current[column]),
                     f"candidate component {name} differs at {column}")
    logits = {
        name: frames[name]
        .set_index("pair_uid")
        .loc[base["pair_uid"], str(components_by_name[name]["inference_logit_column"])]
        .to_numpy(dtype=np.float64)
        for name in map(str, lock["members"])
    }
    ensemble_logit, ensemble_score = apply_formula(lock, logits)
    output = base[list(ALIGNMENT_COLUMNS)].copy()
    output.insert(0, "model_id", model_id)
    output.insert(8, "model_score", ensemble_score)
    # Preserve exact component logits for audit/replay without copying component probabilities.
    for name in map(str, lock["members"]):
        logit_column = str(components_by_name[name]["inference_logit_column"])
        _require(logit_column not in output.columns, f"duplicate component logit column {logit_column}")
        output[logit_column] = logits[name]
    output["ensemble_logit"] = ensemble_logit
    extra_columns = [
        str(components_by_name[name]["inference_logit_column"])
        for name in map(str, lock["members"])
    ] + ["ensemble_logit"]
    output = output[list(STANDARD_COLUMNS) + extra_columns]
    _require(set(STANDARD_COLUMNS).issubset(output.columns), "standard ensemble schema changed")
    _require(len(output) == EXPECTED_CANDIDATE_ROWS and output["pair_uid"].is_unique,
             "ensemble candidate membership changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    _require(not staging.exists(), f"staging directory exists: {staging}")
    staging.mkdir(mode=0o700)
    score_path = staging / "candidate_scores.parquet"
    table = pa.Table.from_pandas(output, preserve_index=False)
    pq.write_table(table, score_path, compression="zstd", row_group_size=65_536)
    os.chmod(score_path, 0o600)
    evaluation_predictions.insert(0, "model_id", model_id)
    evaluation_path = staging / "target_free_108_predictions.csv"
    evaluation_predictions.to_csv(evaluation_path, index=False, float_format="%.17g")
    os.chmod(evaluation_path, 0o600)
    selected_count = int((ensemble_score >= float(lock["cutoff"]["score_threshold"])).sum())
    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": float(time.time() - started),
        "library": LIBRARY,
        "model_id": model_id,
        "rows": EXPECTED_CANDIDATE_ROWS,
        "pair_uid_membership_sha256": membership_sha256(output["pair_uid"]),
        "lock": {
            **_record(lock_path),
            "lock_id": lock.get("lock_id"),
            "lock_role": lock.get("lock_role"),
            "candidate": lock.get("candidate"),
            "candidate_family": lock.get("candidate_family"),
            "formula": lock["formula"],
            "weak_oof_score_threshold": float(lock["cutoff"]["score_threshold"]),
        },
        "score_definition": {
            "formula": lock["formula"]["linear_predictor"],
            "output_transform": "sigmoid",
            "semantics": "selection-label score; not a retention percentage",
            "direction": "higher_is_more_likely_selection_derived_binder",
        },
        "component_scores": component_records,
        "deployment_binding": {
            "head_sha256": sha256_file(head_binding_path),
            "config_sha256": sha256_file(config_binding_path),
            "generic_lock_sha256": sha256_file(lock_path),
            "weak_oof_sha256": sha256_file(weak_oof_binding_path),
            "sources": {
                "head_or_formula": _record(head_binding_path),
                "config": _record(config_binding_path),
                "generic_lock": _record(lock_path),
                "weak_oof": _record(weak_oof_binding_path),
            },
        },
        "parity": {
            "matched_weak_oof": oof_parity,
            "target_free_108_two_layouts": evaluation_parity,
        },
        "weak_oof_cutoff_application": {
            "decision_rule": "model_score >= threshold",
            "threshold": float(lock["cutoff"]["score_threshold"]),
            "candidate_rows_above_threshold_before_per_peptide_cap": selected_count,
            "used_retention": False,
        },
        "outputs": {
            "candidate_scores": {
                **_record(score_path, relative_to=staging),
                "rows": EXPECTED_CANDIDATE_ROWS,
            },
            "target_free_108_predictions": _record(evaluation_path, relative_to=staging),
        },
        "outcome_access": {
            "weak_label_column_read_for_formula_parity": False,
            "retention_measurements_read": False,
            "retention_derived_binder_labels_read": False,
        },
        "validation": {
            "exact_component_candidate_alignment": True,
            "exact_weak_oof_formula_parity": True,
            "target_free_108_formula_parity_across_two_layouts": True,
            "cross_fitted_stacker_deployment_forbidden": True,
            "manifest_written_last": True,
        },
        "code": _record(Path(__file__).resolve()),
    }
    _write_json(staging / "manifest.json", manifest)
    os.rename(staging, output_dir)
    print(json.dumps({"output": str(output_dir), "model_id": model_id,
                      "rows": EXPECTED_CANDIDATE_ROWS, "above_cutoff": selected_count}))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-json", type=Path, required=True)
    parser.add_argument("--component-score", action="append", default=[], required=True,
                        help="Repeat NAME=PATH exactly for every lock component.")
    parser.add_argument("--matched-oof", type=Path, required=True)
    parser.add_argument("--evaluation-aligned", type=Path, required=True)
    parser.add_argument("--evaluation-long", type=Path, required=True)
    parser.add_argument("--head-binding", type=Path, required=True,
                        help="Deployment head, or the generic formula lock for an ensemble.")
    parser.add_argument("--config-binding", type=Path, required=True)
    parser.add_argument("--weak-oof-binding", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
