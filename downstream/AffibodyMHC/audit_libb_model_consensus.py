#!/usr/bin/env python
"""Audit whether independently trained LibB models confirm one another.

The primary analysis applies the already locked, weak-label-only score cutoff
for frozen MINT layer 5, the native StaB projection, and the native RDE
projection.  A pair is called a binder by a model when ``score >= cutoff``.
Intersections then ask whether requiring two or three positive calls improves
the fraction of called pairs that are direct-retention binders.

No model is trained here.  No cutoff in the primary analysis is selected from
retention.  A deliberately circular same-panel F1 threshold exercise is
written to separately named files and must not be interpreted prospectively.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-model-consensus-audit-v1"
EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_BINDERS = 61
EXPECTED_NONBINDERS = 59
EXPECTED_MEMBERSHIP_SHA256 = (
    "083cebcb2c0e83c4f61196211059ab16c4774ee99c3dafc4c573a71f42ece705"
)
EXPECTED_PREDICTIONS_SHA256 = "b7292ce2211722018f7b4786a772e7d2bf672d610594d3e54ac171d64830e01e"
EXPECTED_SOURCE_MANIFEST_SHA256 = "649860d5b8b8b8fa2ec841476bec3e16e2bd7dc5426142d09f72edf11c16159e"
EXPECTED_WEAK_OOF_SHA256 = "9a5e1181309c73e1f36d504f91f0bb45b4521fa8fcb6b1e115d917e7f0029529"
RETENTION_BINDER_THRESHOLD = 75.0
FALTA_CODE = "FALTA"

DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "private_data/experiments/libb_native_projection_ensemble_retention_audit_v1/"
    "averaged_score_predictions.csv"
)
DEFAULT_SOURCE_MANIFEST = DEFAULT_PREDICTIONS.parent / "manifest.json"
DEFAULT_LOCK_DIR = (
    REPO_ROOT
    / "private_data/prospective/libb_native_projection_weak_oof_deployment_locks_v1"
)
DEFAULT_WEAK_OOF = (
    REPO_ROOT
    / "private_data/experiments/libb_weak_ensemble_native_projection_all4_v1/"
    "matched_oof_predictions.csv.gz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "private_data/experiments/libb_model_consensus_audit_v1"
)

PRIMARY_MODELS: Mapping[str, Mapping[str, str]] = {
    "mint": {
        "display_name": "Frozen MINT layer 5",
        "score_column": "mint_layer5",
        "lock_filename": "mint_layer5_control.lock.json",
        "lock_id": "libb-native-mint-layer5-control-weak-oof-v1",
        "component_name": "mint_layer5",
        "lock_sha256": "7bafbb49a8875f255e838c37c11ad7e8926f200752d3195822ec26f4f3db528a",
    },
    "stab": {
        "display_name": "StaB-derived native projection",
        "score_column": "stab_designed_ordered_native_projection",
        "lock_filename": "stab_standalone.lock.json",
        "lock_id": "libb-native-stab-standalone-weak-oof-v1",
        "component_name": "stab_designed_ordered",
        "lock_sha256": "b7f28c3ecc899c2c6814a95dc4445953cfc26d6b7c31de00abd58cc82f954857",
    },
    "rde": {
        "display_name": "RDE-derived native projection",
        "score_column": "rde_network_designed_3fold_native_projection",
        "lock_filename": "rde_standalone.lock.json",
        "lock_id": "libb-native-rde-standalone-weak-oof-v1",
        "component_name": "rde_network_designed_3fold",
        "lock_sha256": "8c61de01e67f7981dc3bc02c00f9d15da8748f15ee161929b6563afc2d531d44",
    },
}

ADDITIVE_KEY = "additive"
ADDITIVE_SCORE_COLUMN = "additive_7site"
ADDITIVE_DISPLAY_NAME = "Additive seven-position baseline"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": _relative(resolved),
        "bytes": resolved.stat().st_size,
        "mtime_utc": pd.Timestamp(
            resolved.stat().st_mtime, unit="s", tz="UTC"
        ).isoformat(),
        "sha256": _sha256_file(resolved),
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _write_text_exclusive(path: Path, text: str) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")
    os.chmod(path, 0o600)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    _require(not Path(path).exists(), f"refusing to overwrite {path}")
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        na_rep="",
    )
    os.chmod(path, 0o600)


def _membership_sha256(row_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, row_ids)))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def select_f1_threshold(
    scores: Sequence[float], labels: Sequence[int]
) -> dict[str, Any]:
    """Choose score >= cutoff by F1, precision, fewer calls, then cutoff."""

    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    _require(score_array.ndim == label_array.ndim == 1, "scores and labels must be vectors")
    _require(len(score_array) == len(label_array) and len(score_array), "bad threshold arrays")
    _require(bool(np.isfinite(score_array).all()), "non-finite threshold score")
    _require(set(np.unique(label_array).tolist()) == {0, 1}, "threshold labels need both classes")
    positives = int(label_array.sum())
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for threshold in np.unique(score_array):
        called = score_array >= threshold
        call_count = int(called.sum())
        true_positive = int(np.logical_and(called, label_array == 1).sum())
        false_positive = call_count - true_positive
        false_negative = positives - true_positive
        true_negative = len(label_array) - true_positive - false_positive - false_negative
        precision = Fraction(true_positive, call_count)
        recall = Fraction(true_positive, positives)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = Fraction(2 * true_positive, denominator) if denominator else Fraction(0, 1)
        result = {
            "threshold": float(threshold),
            "calls": call_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        candidates.append(((f1, precision, -call_count, float(threshold)), result))
    return max(candidates, key=lambda item: item[0])[1]


def load_weak_cutoffs_before_retention(
    lock_dir: Path, weak_oof_path: Path, output_dir: Path
) -> tuple[dict[str, float], dict[str, Any]]:
    """Validate all primary locks and derive one secondary additive cutoff."""

    lock_dir = Path(lock_dir).resolve()
    weak_oof_path = Path(weak_oof_path).resolve()
    output_dir = Path(output_dir).resolve()
    _require(not output_dir.exists(), f"output already exists: {output_dir}")
    _require("private_data" in output_dir.parts, "output must remain under private_data")
    _require(lock_dir.is_dir(), f"missing deployment-lock directory: {lock_dir}")
    _require(weak_oof_path.is_file(), f"missing canonical weak OOF predictions: {weak_oof_path}")

    cutoffs: dict[str, float] = {}
    lock_records: dict[str, Any] = {}
    expected_oof_hashes: set[str] = set()
    for model_key, spec in PRIMARY_MODELS.items():
        path = lock_dir / spec["lock_filename"]
        _require(path.is_file(), f"missing standalone deployment lock: {path}")
        _require(_sha256_file(path) == spec["lock_sha256"], f"{model_key} canonical lock hash changed")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(payload.get("retention_labels_read") is False, f"{model_key} lock read retention")
        _require(payload.get("lock_id") == spec["lock_id"], f"{model_key} lock ID changed")
        components = payload.get("components", [])
        _require(len(components) == 1, f"{model_key} standalone lock has {len(components)} components")
        _require(
            components[0].get("name") == spec["component_name"],
            f"{model_key} lock component changed",
        )
        cutoff = float(payload["selection"]["score_threshold"])
        _require(math.isfinite(cutoff) and 0.0 <= cutoff <= 1.0, f"invalid {model_key} cutoff")
        provenance = str(payload.get("threshold_selection_provenance", "")).lower()
        _require("weak-label" in provenance and "no retention" in provenance, f"bad {model_key} provenance")
        oof_hash = str(payload.get("weak_source_sha256", {}).get("oof", ""))
        _require(re.fullmatch(r"[0-9a-f]{64}", oof_hash) is not None, f"bad {model_key} OOF hash")
        expected_oof_hashes.add(oof_hash)
        cutoffs[model_key] = cutoff
        lock_records[model_key] = {
            "file": _file_record(path),
            "lock_id": payload["lock_id"],
            "retention_labels_read": False,
            "score_cutoff": cutoff,
            "threshold_selection_provenance": payload["threshold_selection_provenance"],
            "weak_oof_training_summary": payload.get("threshold_training_summary"),
        }

    actual_oof_hash = _sha256_file(weak_oof_path)
    _require(actual_oof_hash == EXPECTED_WEAK_OOF_SHA256, "canonical weak OOF hash changed")
    _require(len(expected_oof_hashes) == 1, "standalone locks disagree on weak OOF source")
    _require(actual_oof_hash in expected_oof_hashes, "weak OOF file does not match lock provenance")
    weak = pd.read_csv(
        weak_oof_path,
        usecols=[
            "row_id",
            "fold",
            "weak_label",
            ADDITIVE_SCORE_COLUMN,
            *(str(spec["score_column"]) for spec in PRIMARY_MODELS.values()),
        ],
        dtype={"row_id": str},
        float_precision="round_trip",
    )
    _require(len(weak) == 10181, f"weak OOF row count changed: {len(weak)}")
    _require(weak["row_id"].nunique() == len(weak), "weak OOF row IDs are not unique")
    weak["weak_label"] = pd.to_numeric(weak["weak_label"], errors="raise").astype(int)
    weak[ADDITIVE_SCORE_COLUMN] = pd.to_numeric(
        weak[ADDITIVE_SCORE_COLUMN], errors="raise"
    ).astype(float)
    _require(int(weak["weak_label"].sum()) == 7939, "weak OOF positive count changed")
    recomputed_primary: dict[str, Any] = {}
    for model_key, spec in PRIMARY_MODELS.items():
        column = str(spec["score_column"])
        weak[column] = pd.to_numeric(weak[column], errors="raise").astype(float)
        chosen = select_f1_threshold(weak[column], weak["weak_label"])
        _require(
            math.isclose(float(chosen["threshold"]), cutoffs[model_key], rel_tol=0.0, abs_tol=1e-15),
            f"{model_key} locked cutoff does not reproduce from canonical weak OOF predictions",
        )
        recomputed_primary[model_key] = chosen
    additive_cutoff = select_f1_threshold(
        weak[ADDITIVE_SCORE_COLUMN], weak["weak_label"]
    )
    cutoffs[ADDITIVE_KEY] = float(additive_cutoff["threshold"])

    output_dir.mkdir(parents=True, mode=0o700)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "event": "all primary weak-label cutoffs validated and the additive cutoff derived before retention-bearing predictions were opened",
        "retention_sources_opened": False,
        "primary_analysis_retention_labels_read_for_cutoffs": False,
        "primary_locks": lock_records,
        "primary_cutoffs_recomputed_from_canonical_weak_oof": recomputed_primary,
        "secondary_additive_cutoff": {
            **additive_cutoff,
            "score_column": ADDITIVE_SCORE_COLUMN,
            "retention_labels_read": False,
            "provenance": "F1-max cutoff derived from the same canonical 10,181 weak-label OOF rows; ties prefer precision, fewer calls, then higher score",
            "weak_oof_file": _file_record(weak_oof_path),
        },
    }
    _write_json_exclusive(output_dir / "pre_retention_lock.json", seal)
    return cutoffs, seal


def load_and_validate_predictions(path: Path, source_manifest_path: Path) -> pd.DataFrame:
    path = Path(path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    _require(path.is_file(), f"missing canonical averaged predictions: {path}")
    _require(source_manifest_path.is_file(), f"missing source manifest: {source_manifest_path}")
    _require(_sha256_file(path) == EXPECTED_PREDICTIONS_SHA256, "canonical averaged-prediction hash changed")
    _require(
        _sha256_file(source_manifest_path) == EXPECTED_SOURCE_MANIFEST_SHA256,
        "canonical averaged-prediction manifest hash changed",
    )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    recorded = source_manifest.get("outputs", {}).get(path.name, {})
    _require(recorded.get("sha256") == _sha256_file(path), "averaged prediction hash differs from source manifest")

    string_columns = {
        "eval_row_id": str,
        "peptide_design_code": str,
        "affibody_design_code": str,
    }
    frame = pd.read_csv(
        path,
        dtype=string_columns,
        keep_default_na=False,
        float_precision="round_trip",
    )
    required = {
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
        ADDITIVE_SCORE_COLUMN,
        *(spec["score_column"] for spec in PRIMARY_MODELS.values()),
    }
    _require(required.issubset(frame.columns), f"prediction columns changed: {sorted(required - set(frame.columns))}")
    numeric = [
        "target_retention",
        "target_binder",
        ADDITIVE_SCORE_COLUMN,
        *(spec["score_column"] for spec in PRIMARY_MODELS.values()),
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        _require(bool(np.isfinite(frame[column]).all()), f"non-finite values in {column}")

    _require(len(frame) == EXPECTED_ROWS, f"panel has {len(frame)} rows, expected {EXPECTED_ROWS}")
    _require(frame["eval_row_id"].nunique() == EXPECTED_ROWS, "eval_row_id values are not unique")
    _require(
        bool(frame["eval_row_id"].map(lambda value: re.fullmatch(r"[0-9a-f]{20}", value) is not None).all()),
        "eval_row_id format changed",
    )
    _require(
        _membership_sha256(frame["eval_row_id"]) == EXPECTED_MEMBERSHIP_SHA256,
        "corrected LibB-120 row-ID membership changed",
    )
    _require(frame["peptide_design_code"].nunique() == EXPECTED_PEPTIDES, "peptide count changed")
    _require(frame["affibody_design_code"].nunique() == EXPECTED_AFFIBODIES, "Affibody count changed")
    _require(
        not bool(frame.duplicated(["peptide_design_code", "affibody_design_code"]).any()),
        "duplicate peptide-Affibody pairs",
    )
    peptide_sizes = frame.groupby("peptide_design_code", sort=True).size()
    affibody_sizes = frame.groupby("affibody_design_code", sort=True).size()
    _require(bool(peptide_sizes.eq(EXPECTED_AFFIBODIES).all()), "panel is not 12 x 10 by peptide")
    _require(bool(affibody_sizes.eq(EXPECTED_PEPTIDES).all()), "panel is not 12 x 10 by Affibody")
    frame["target_binder"] = frame["target_binder"].astype(int)
    _require(set(frame["target_binder"].unique()) == {0, 1}, "target_binder is not binary")
    expected_labels = frame["target_retention"].ge(RETENTION_BINDER_THRESHOLD).astype(int)
    _require(bool(expected_labels.eq(frame["target_binder"]).all()), "binder labels are not retention >= 75")
    _require(int(frame["target_binder"].sum()) == EXPECTED_BINDERS, "binder count changed")
    _require(len(frame) - int(frame["target_binder"].sum()) == EXPECTED_NONBINDERS, "non-binder count changed")
    for column in [ADDITIVE_SCORE_COLUMN, *(spec["score_column"] for spec in PRIMARY_MODELS.values())]:
        _require(bool(frame[column].between(0.0, 1.0).all()), f"scores outside [0,1] in {column}")
    return frame.sort_values(
        ["peptide_design_code", "affibody_design_code"], kind="stable"
    ).reset_index(drop=True)


def add_calls(frame: pd.DataFrame, cutoffs: Mapping[str, float], include_additive: bool) -> pd.DataFrame:
    output = frame.copy()
    model_items = list(PRIMARY_MODELS.items())
    if include_additive:
        model_items.append(
            (
                ADDITIVE_KEY,
                {"score_column": ADDITIVE_SCORE_COLUMN, "display_name": ADDITIVE_DISPLAY_NAME},
            )
        )
    for model_key, spec in model_items:
        output[f"cutoff_{model_key}"] = float(cutoffs[model_key])
        output[f"call_{model_key}"] = (
            output[str(spec["score_column"])].to_numpy(dtype=float) >= float(cutoffs[model_key])
        ).astype(int)
    primary_call_columns = [f"call_{key}" for key in PRIMARY_MODELS]
    output["primary_positive_call_count"] = output[primary_call_columns].sum(axis=1).astype(int)
    output["primary_agreement_pattern"] = output.apply(
        lambda row: "".join(str(int(row[column])) for column in primary_call_columns), axis=1
    )
    return output


def primary_rules() -> list[dict[str, Any]]:
    keys = tuple(PRIMARY_MODELS)
    rules: list[dict[str, Any]] = []
    for size in range(1, len(keys) + 1):
        for members in itertools.combinations(keys, size):
            rules.append(
                {
                    "call_rule": "_and_".join(members),
                    "rule_type": "single" if size == 1 else "intersection",
                    "members": members,
                }
            )
    rules.append(
        {
            "call_rule": "at_least_two_of_three",
            "rule_type": "majority",
            "members": keys,
        }
    )
    return rules


def secondary_additive_rules() -> list[dict[str, Any]]:
    keys = tuple(PRIMARY_MODELS)
    rules = [
        {"call_rule": "additive", "rule_type": "single", "members": (ADDITIVE_KEY,)}
    ]
    for size in range(1, len(keys) + 1):
        for others in itertools.combinations(keys, size):
            members = (ADDITIVE_KEY, *others)
            rules.append(
                {
                    "call_rule": "_and_".join(members),
                    "rule_type": "intersection",
                    "members": members,
                }
            )
    return rules


def rule_mask(frame: pd.DataFrame, rule: Mapping[str, Any]) -> np.ndarray:
    if rule["rule_type"] == "majority":
        return frame[[f"call_{member}" for member in rule["members"]]].sum(axis=1).ge(2).to_numpy()
    return np.logical_and.reduce(
        [frame[f"call_{member}"].to_numpy(dtype=bool) for member in rule["members"]]
    )


def _metric_counts(frame: pd.DataFrame, mask: Sequence[bool]) -> dict[str, Any]:
    called = np.asarray(mask, dtype=bool)
    labels = frame["target_binder"].to_numpy(dtype=int)
    calls = int(called.sum())
    true_positive = int(np.logical_and(called, labels == 1).sum())
    false_positive = calls - true_positive
    positives = int(labels.sum())
    false_negative = positives - true_positive
    true_negative = len(frame) - true_positive - false_positive - false_negative
    precision = true_positive / calls if calls else float("nan")
    recall = true_positive / positives if positives else float("nan")
    f1 = (
        2 * true_positive / (2 * true_positive + false_positive + false_negative)
        if 2 * true_positive + false_positive + false_negative
        else float("nan")
    )
    return {
        "pairs": len(frame),
        "available_binders": positives,
        "calls": calls,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def consensus_summary(
    frame: pd.DataFrame, rules: Sequence[Mapping[str, Any]], analysis_status: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = (
        ("complete_120", frame),
        ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)].copy()),
    )
    for panel_scope, current in scopes:
        for order, rule in enumerate(rules):
            rows.append(
                {
                    "panel_scope": panel_scope,
                    "rule_order": order,
                    "call_rule": rule["call_rule"],
                    "rule_type": rule["rule_type"],
                    "members": "+".join(rule["members"]),
                    **_metric_counts(current, rule_mask(current, rule)),
                    "analysis_status": analysis_status,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["panel_scope", "rule_order"], kind="stable"
    ).reset_index(drop=True)


def pairwise_call_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for panel_scope, current in (
        ("complete_120", frame),
        ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)]),
    ):
        for model_a, model_b in itertools.combinations(PRIMARY_MODELS, 2):
            a = current[f"call_{model_a}"].to_numpy(dtype=bool)
            b = current[f"call_{model_b}"].to_numpy(dtype=bool)
            intersection = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum())
            a_calls = int(a.sum())
            b_calls = int(b.sum())
            rows.append(
                {
                    "panel_scope": panel_scope,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_calls": a_calls,
                    "model_b_calls": b_calls,
                    "both_call": intersection,
                    "either_calls": union,
                    "jaccard_positive_calls": intersection / union if union else float("nan"),
                    "fraction_of_model_a_calls_confirmed_by_b": intersection / a_calls if a_calls else float("nan"),
                    "fraction_of_model_b_calls_confirmed_by_a": intersection / b_calls if b_calls else float("nan"),
                    "all_call_agreement_fraction": float(np.equal(a, b).mean()),
                }
            )
    return pd.DataFrame(rows)


def agreement_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_keys = tuple(PRIMARY_MODELS)
    for panel_scope, current in (
        ("complete_120", frame),
        ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)]),
    ):
        for bits in itertools.product((0, 1), repeat=len(model_keys)):
            mask = np.ones(len(current), dtype=bool)
            for key, bit in zip(model_keys, bits):
                mask &= current[f"call_{key}"].to_numpy(dtype=int) == bit
            counts = _metric_counts(current, mask)
            rows.append(
                {
                    "panel_scope": panel_scope,
                    "agreement_pattern_mint_stab_rde": "".join(map(str, bits)),
                    "mint_call": bits[0],
                    "stab_call": bits[1],
                    "rde_call": bits[2],
                    "positive_model_count": sum(bits),
                    "pairs_in_pattern": counts["calls"],
                    "binders_in_pattern": counts["true_positive"],
                    "non_binders_in_pattern": counts["false_positive"],
                    "binder_fraction_in_pattern": counts["precision"],
                    "fraction_of_panel": counts["calls"] / len(current),
                }
            )
    return pd.DataFrame(rows)


def stratified_rule_table(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    group_column: str,
    include_falta_excluded_scope: bool,
) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [("complete_120", frame)]
    if include_falta_excluded_scope:
        scopes.append(
            ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)])
        )
    rows: list[dict[str, Any]] = []
    for panel_scope, current in scopes:
        for group_value, group in current.groupby(group_column, sort=True):
            for order, rule in enumerate(rules):
                rows.append(
                    {
                        "panel_scope": panel_scope,
                        group_column: str(group_value),
                        "rule_order": order,
                        "call_rule": rule["call_rule"],
                        "rule_type": rule["rule_type"],
                        "members": "+".join(rule["members"]),
                        **_metric_counts(group, rule_mask(group, rule)),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["panel_scope", group_column, "rule_order"], kind="stable"
    ).reset_index(drop=True)


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if np.ptp(x_array) == 0.0 or np.ptp(y_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    return _pearson(x_rank, y_rank)


def score_correlations(frame: pd.DataFrame, include_additive: bool = True) -> pd.DataFrame:
    score_columns = {
        key: str(spec["score_column"]) for key, spec in PRIMARY_MODELS.items()
    }
    if include_additive:
        score_columns[ADDITIVE_KEY] = ADDITIVE_SCORE_COLUMN
    rows: list[dict[str, Any]] = []
    for panel_scope, current in (
        ("complete_120", frame),
        ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)]),
    ):
        for model_a, model_b in itertools.combinations(score_columns, 2):
            column_a = score_columns[model_a]
            column_b = score_columns[model_b]
            within_pearson: list[float] = []
            within_spearman: list[float] = []
            for _, group in current.groupby("peptide_design_code", sort=True):
                current_pearson = _pearson(group[column_a], group[column_b])
                current_spearman = _spearman(group[column_a], group[column_b])
                if math.isfinite(current_pearson):
                    within_pearson.append(current_pearson)
                if math.isfinite(current_spearman):
                    within_spearman.append(current_spearman)
            rows.append(
                {
                    "panel_scope": panel_scope,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_scope": (
                        "primary" if model_a in PRIMARY_MODELS and model_b in PRIMARY_MODELS else "includes_secondary_additive"
                    ),
                    "global_pearson": _pearson(current[column_a], current[column_b]),
                    "global_spearman": _spearman(current[column_a], current[column_b]),
                    "mean_within_peptide_pearson": float(np.mean(within_pearson)),
                    "mean_within_peptide_spearman": float(np.mean(within_spearman)),
                    "within_peptide_pearson_groups": len(within_pearson),
                    "within_peptide_spearman_groups": len(within_spearman),
                }
            )
    return pd.DataFrame(rows)


def peptide_cluster_bootstrap(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap 12 peptide rows and return metric intervals and paired deltas."""

    _require(int(draws) >= 1000, "bootstrap draws must be at least 1000")
    interval_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for scope_index, (panel_scope, current) in enumerate(
        (
            ("complete_120", frame),
            ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)]),
        )
    ):
        peptides = sorted(current["peptide_design_code"].unique())
        _require(len(peptides) == EXPECTED_PEPTIDES, "bootstrap peptide count changed")
        positive_by_peptide = np.asarray(
            [
                int(current.loc[current["peptide_design_code"].eq(peptide), "target_binder"].sum())
                for peptide in peptides
            ],
            dtype=int,
        )
        rule_call_counts: dict[str, np.ndarray] = {}
        rule_true_counts: dict[str, np.ndarray] = {}
        for rule in rules:
            calls = []
            true_calls = []
            for peptide in peptides:
                group = current.loc[current["peptide_design_code"].eq(peptide)]
                mask = rule_mask(group, rule)
                calls.append(int(mask.sum()))
                true_calls.append(
                    int(np.logical_and(mask, group["target_binder"].to_numpy(dtype=int) == 1).sum())
                )
            rule_call_counts[str(rule["call_rule"])] = np.asarray(calls, dtype=int)
            rule_true_counts[str(rule["call_rule"])] = np.asarray(true_calls, dtype=int)

        rng = np.random.default_rng(int(seed) + scope_index * 100003)
        sampled = rng.integers(0, len(peptides), size=(int(draws), len(peptides)))
        bootstrap_positives = positive_by_peptide[sampled].sum(axis=1)
        precision_draws: dict[str, np.ndarray] = {}
        recall_draws: dict[str, np.ndarray] = {}
        for rule in rules:
            name = str(rule["call_rule"])
            calls = rule_call_counts[name][sampled].sum(axis=1)
            true_positive = rule_true_counts[name][sampled].sum(axis=1)
            precision = np.divide(
                true_positive,
                calls,
                out=np.full(int(draws), np.nan, dtype=float),
                where=calls > 0,
            )
            recall = np.divide(
                true_positive,
                bootstrap_positives,
                out=np.full(int(draws), np.nan, dtype=float),
                where=bootstrap_positives > 0,
            )
            precision_draws[name] = precision
            recall_draws[name] = recall
            point = _metric_counts(current, rule_mask(current, rule))
            for metric_name, point_value, estimates in (
                ("precision", point["precision"], precision),
                ("recall", point["recall"], recall),
            ):
                finite = estimates[np.isfinite(estimates)]
                _require(len(finite) >= int(0.99 * draws), f"too few finite {metric_name} draws")
                interval_rows.append(
                    {
                        "panel_scope": panel_scope,
                        "call_rule": name,
                        "metric": metric_name,
                        "point_estimate": point_value,
                        "bootstrap_mean": float(np.mean(finite)),
                        "ci_lower_2_5": float(np.quantile(finite, 0.025)),
                        "ci_upper_97_5": float(np.quantile(finite, 0.975)),
                        "draws_requested": int(draws),
                        "draws_finite": len(finite),
                        "bootstrap_unit": "peptide_design_code",
                        "bootstrap_seed": int(seed) + scope_index * 100003,
                    }
                )

        for rule in rules:
            if rule["rule_type"] != "intersection":
                continue
            name = str(rule["call_rule"])
            for reference in rule["members"]:
                delta = precision_draws[name] - precision_draws[str(reference)]
                finite = delta[np.isfinite(delta)]
                point_rule = _metric_counts(current, rule_mask(current, rule))["precision"]
                reference_rule = next(item for item in rules if item["call_rule"] == reference)
                point_reference = _metric_counts(
                    current, rule_mask(current, reference_rule)
                )["precision"]
                delta_rows.append(
                    {
                        "panel_scope": panel_scope,
                        "intersection_rule": name,
                        "reference_single_model": reference,
                        "precision_delta_point": point_rule - point_reference,
                        "precision_delta_bootstrap_mean": float(np.mean(finite)),
                        "ci_lower_2_5": float(np.quantile(finite, 0.025)),
                        "ci_upper_97_5": float(np.quantile(finite, 0.975)),
                        "fraction_bootstrap_delta_above_zero": float(np.mean(finite > 0.0)),
                        "draws_finite": len(finite),
                        "bootstrap_unit": "peptide_design_code",
                        "bootstrap_seed": int(seed) + scope_index * 100003,
                    }
                )
    return pd.DataFrame(interval_rows), pd.DataFrame(delta_rows)


def matched_budget_component_controls(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare consensus with each member's top scores at the same peptide budget.

    If a consensus rule calls ``n`` pairs for one peptide, its matched control
    takes that component's top ``n`` scores for that same peptide.  Thus the
    number of proposed wet-lab assays is identical peptide by peptide and any
    improvement cannot be attributed merely to making fewer calls.
    """

    _require(int(draws) >= 1000, "matched-budget bootstrap draws must be at least 1000")
    score_columns = {
        key: str(spec["score_column"]) for key, spec in PRIMARY_MODELS.items()
    }
    summary_rows: list[dict[str, Any]] = []
    exact_rows: list[pd.DataFrame] = []
    intersection_rules = [rule for rule in rules if rule["rule_type"] == "intersection"]
    for scope_index, (panel_scope, current) in enumerate(
        (
            ("complete_120", frame),
            ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)].copy()),
        )
    ):
        peptides = sorted(current["peptide_design_code"].unique())
        rng = np.random.default_rng(int(seed) + 500009 + scope_index * 100003)
        sampled = rng.integers(0, len(peptides), size=(int(draws), len(peptides)))
        for rule in intersection_rules:
            consensus = rule_mask(current, rule)
            consensus_series = pd.Series(consensus, index=current.index)
            consensus_counts = _metric_counts(current, consensus)
            consensus_calls_by_peptide = np.asarray(
                [
                    int(consensus_series.loc[current["peptide_design_code"].eq(peptide)].sum())
                    for peptide in peptides
                ],
                dtype=int,
            )
            consensus_true_by_peptide = np.asarray(
                [
                    int(
                        np.logical_and(
                            consensus_series.loc[
                                current["peptide_design_code"].eq(peptide)
                            ].to_numpy(dtype=bool),
                            current.loc[
                                current["peptide_design_code"].eq(peptide), "target_binder"
                            ].to_numpy(dtype=int)
                            == 1,
                        ).sum()
                    )
                    for peptide in peptides
                ],
                dtype=int,
            )
            for control_model in rule["members"]:
                control_series = pd.Series(False, index=current.index)
                for peptide in peptides:
                    group = current.loc[current["peptide_design_code"].eq(peptide)].copy()
                    budget = int(
                        consensus_series.loc[
                            current["peptide_design_code"].eq(peptide)
                        ].sum()
                    )
                    ranked = group.sort_values(
                        [score_columns[control_model], "affibody_design_code", "eval_row_id"],
                        ascending=[False, True, True],
                        kind="stable",
                    )
                    control_series.loc[ranked.head(budget).index] = True
                control = control_series.to_numpy(dtype=bool)
                control_counts = _metric_counts(current, control)
                _require(
                    control_counts["calls"] == consensus_counts["calls"],
                    "matched-budget global call count differs",
                )
                for peptide in peptides:
                    peptide_mask = current["peptide_design_code"].eq(peptide)
                    _require(
                        int(control_series.loc[peptide_mask].sum())
                        == int(consensus_series.loc[peptide_mask].sum()),
                        f"matched budget differs for {rule['call_rule']}/{control_model}/{peptide}",
                    )

                control_true_by_peptide = np.asarray(
                    [
                        int(
                            np.logical_and(
                                control_series.loc[
                                    current["peptide_design_code"].eq(peptide)
                                ].to_numpy(dtype=bool),
                                current.loc[
                                    current["peptide_design_code"].eq(peptide),
                                    "target_binder",
                                ].to_numpy(dtype=int)
                                == 1,
                            ).sum()
                        )
                        for peptide in peptides
                    ],
                    dtype=int,
                )
                sampled_calls = consensus_calls_by_peptide[sampled].sum(axis=1)
                consensus_true = consensus_true_by_peptide[sampled].sum(axis=1)
                control_true = control_true_by_peptide[sampled].sum(axis=1)
                delta = np.divide(
                    consensus_true - control_true,
                    sampled_calls,
                    out=np.full(int(draws), np.nan, dtype=float),
                    where=sampled_calls > 0,
                )
                finite = delta[np.isfinite(delta)]
                _require(len(finite) >= int(0.99 * draws), "too few matched-budget draws")
                summary_rows.append(
                    {
                        "panel_scope": panel_scope,
                        "consensus_rule": rule["call_rule"],
                        "control_model": control_model,
                        "budget_definition": "same number selected separately within every peptide",
                        "calls_each": consensus_counts["calls"],
                        "consensus_true_positive": consensus_counts["true_positive"],
                        "control_true_positive": control_counts["true_positive"],
                        "consensus_precision": consensus_counts["precision"],
                        "control_precision": control_counts["precision"],
                        "precision_delta_consensus_minus_control": (
                            consensus_counts["precision"] - control_counts["precision"]
                        ),
                        "bootstrap_delta_ci_lower_2_5": float(np.quantile(finite, 0.025)),
                        "bootstrap_delta_ci_upper_97_5": float(np.quantile(finite, 0.975)),
                        "bootstrap_fraction_delta_above_zero": float(np.mean(finite > 0.0)),
                        "bootstrap_draws_finite": len(finite),
                        "bootstrap_unit": "peptide_design_code",
                        "bootstrap_seed": int(seed) + 500009 + scope_index * 100003,
                    }
                )
                exact = current[
                    [
                        "eval_row_id",
                        "peptide_design_code",
                        "affibody_design_code",
                        "target_retention",
                        "target_binder",
                    ]
                ].copy()
                exact.insert(0, "panel_scope", panel_scope)
                exact.insert(1, "consensus_rule", rule["call_rule"])
                exact.insert(2, "control_model", control_model)
                exact["consensus_call"] = consensus.astype(int)
                exact["matched_budget_control_call"] = control.astype(int)
                exact["control_score"] = current[score_columns[control_model]].to_numpy(dtype=float)
                exact_rows.append(exact)
    return pd.DataFrame(summary_rows), pd.concat(exact_rows, ignore_index=True)


def matched_budget_negative_controls(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare unanimous rejection with each model's bottom scores at equal budgets."""

    score_columns = {
        key: str(spec["score_column"]) for key, spec in PRIMARY_MODELS.items()
    }
    summary_rows: list[dict[str, Any]] = []
    exact_rows: list[pd.DataFrame] = []
    for panel_scope, current in (
        ("complete_120", frame),
        ("FALTA_excluded", frame.loc[frame["affibody_design_code"].ne(FALTA_CODE)].copy()),
    ):
        peptides = sorted(current["peptide_design_code"].unique())
        unanimous_negative = current[
            [f"call_{key}" for key in PRIMARY_MODELS]
        ].sum(axis=1).eq(0)
        consensus_count = int(unanimous_negative.sum())
        consensus_binders = int(
            current.loc[unanimous_negative, "target_binder"].sum()
        )
        consensus_non_binders = consensus_count - consensus_binders
        rejection_budget = {
            peptide: int(
                unanimous_negative.loc[
                    current["peptide_design_code"].eq(peptide)
                ].sum()
            )
            for peptide in peptides
        }
        for control_model, score_column in score_columns.items():
            control = pd.Series(False, index=current.index)
            for peptide in peptides:
                group = current.loc[current["peptide_design_code"].eq(peptide)]
                ranked = group.sort_values(
                    [score_column, "affibody_design_code", "eval_row_id"],
                    ascending=[True, True, True],
                    kind="stable",
                )
                control.loc[ranked.head(rejection_budget[peptide]).index] = True
            _require(int(control.sum()) == consensus_count, "negative control budget differs")
            for peptide in peptides:
                peptide_mask = current["peptide_design_code"].eq(peptide)
                _require(
                    int(control.loc[peptide_mask].sum()) == rejection_budget[peptide],
                    f"negative budget differs for {control_model}/{peptide}",
                )
            control_binders = int(current.loc[control, "target_binder"].sum())
            control_non_binders = int(control.sum()) - control_binders
            overlap = int(np.logical_and(unanimous_negative.to_numpy(), control.to_numpy()).sum())
            summary_rows.append(
                {
                    "panel_scope": panel_scope,
                    "control_model": control_model,
                    "budget_definition": "same number rejected separately within every peptide",
                    "rejected_pairs_each": consensus_count,
                    "unanimous_negative_binders_rejected": consensus_binders,
                    "control_bottom_score_binders_rejected": control_binders,
                    "unanimous_negative_non_binders_rejected": consensus_non_binders,
                    "control_bottom_score_non_binders_rejected": control_non_binders,
                    "unanimous_negative_non_binder_fraction": (
                        consensus_non_binders / consensus_count if consensus_count else float("nan")
                    ),
                    "control_non_binder_fraction": (
                        control_non_binders / consensus_count if consensus_count else float("nan")
                    ),
                    "unanimous_negative_max_retention": (
                        float(current.loc[unanimous_negative, "target_retention"].max())
                        if consensus_count
                        else float("nan")
                    ),
                    "control_max_retention": (
                        float(current.loc[control, "target_retention"].max())
                        if consensus_count
                        else float("nan")
                    ),
                    "pair_overlap": overlap,
                    "pair_sets_identical": int(overlap == consensus_count),
                    "per_peptide_rejection_budget": json.dumps(
                        rejection_budget, sort_keys=True, separators=(",", ":")
                    ),
                }
            )
            exact = current[
                [
                    "eval_row_id",
                    "peptide_design_code",
                    "affibody_design_code",
                    "target_retention",
                    "target_binder",
                ]
            ].copy()
            exact.insert(0, "panel_scope", panel_scope)
            exact.insert(1, "control_model", control_model)
            exact["unanimous_negative_reject"] = unanimous_negative.astype(int)
            exact["matched_budget_bottom_score_reject"] = control.astype(int)
            exact["control_score"] = current[score_column].to_numpy(dtype=float)
            exact_rows.append(exact)
    return pd.DataFrame(summary_rows), pd.concat(exact_rows, ignore_index=True)


def build_retrospective_threshold_analysis(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit cutoffs on these same labels; output is intentionally circular."""

    thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, Any]] = []
    for model_key, spec in PRIMARY_MODELS.items():
        chosen = select_f1_threshold(frame[str(spec["score_column"])], frame["target_binder"])
        thresholds[model_key] = float(chosen["threshold"])
        threshold_rows.append(
            {
                "model_key": model_key,
                "display_name": spec["display_name"],
                **chosen,
                "analysis_status": "circular_cutoff_fit_and_evaluated_on_same_120_retention_labels",
            }
        )
    calls = add_calls(frame, thresholds, include_additive=False)
    rules = [rule for rule in primary_rules() if rule["rule_type"] != "majority"]
    summary = consensus_summary(
        calls,
        rules,
        "circular_cutoff_fit_and_evaluated_on_same_120_retention_labels",
    )
    exact = calls[
        [
            "eval_row_id",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
            "target_binder",
            *(str(spec["score_column"]) for spec in PRIMARY_MODELS.values()),
            *(f"cutoff_{key}" for key in PRIMARY_MODELS),
            *(f"call_{key}" for key in PRIMARY_MODELS),
            "primary_positive_call_count",
            "primary_agreement_pattern",
        ]
    ].copy()
    exact["analysis_status"] = "circular_cutoff_fit_and_evaluated_on_same_120_retention_labels"
    return thresholds, pd.DataFrame(threshold_rows), exact, summary


def _pct(value: float) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{100.0 * float(value):.1f}%"


def _interval_lookup(
    bootstrap: pd.DataFrame, panel_scope: str, call_rule: str, metric: str
) -> tuple[float, float]:
    row = bootstrap.loc[
        bootstrap["panel_scope"].eq(panel_scope)
        & bootstrap["call_rule"].eq(call_rule)
        & bootstrap["metric"].eq(metric)
    ]
    _require(len(row) == 1, f"missing bootstrap row {panel_scope}/{call_rule}/{metric}")
    return float(row.iloc[0]["ci_lower_2_5"]), float(row.iloc[0]["ci_upper_97_5"])


def render_report(
    cutoffs: Mapping[str, float],
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
    patterns: pd.DataFrame,
    correlations: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_deltas: pd.DataFrame,
    matched_budget: pd.DataFrame,
    negative_budget: pd.DataFrame,
    additive_summary: pd.DataFrame,
    retrospective_thresholds: pd.DataFrame,
    retrospective_summary: pd.DataFrame,
) -> str:
    primary = summary.loc[
        summary["panel_scope"].eq("complete_120")
        & summary["rule_type"].isin(["single", "intersection"])
    ].copy()
    no_falta = summary.loc[
        summary["panel_scope"].eq("FALTA_excluded")
        & summary["rule_type"].isin(["single", "intersection"])
    ].copy()
    lookup = primary.set_index("call_rule")
    rde = lookup.loc["rde"]
    triple = lookup.loc["mint_and_stab_and_rde"]
    none_pattern = patterns.loc[
        patterns["panel_scope"].eq("complete_120")
        & patterns["agreement_pattern_mint_stab_rde"].eq("000")
    ].iloc[0]
    primary_corr = correlations.loc[
        correlations["panel_scope"].eq("complete_120")
        & correlations["model_scope"].eq("primary")
    ]
    min_global_spearman = float(primary_corr["global_spearman"].min())
    max_global_spearman = float(primary_corr["global_spearman"].max())
    min_within_spearman = float(primary_corr["mean_within_peptide_spearman"].min())
    max_within_spearman = float(primary_corr["mean_within_peptide_spearman"].max())
    rde_stab_overlap = overlap.loc[
        overlap["panel_scope"].eq("complete_120")
        & overlap["model_a"].eq("stab")
        & overlap["model_b"].eq("rde")
    ].iloc[0]
    delta_row = bootstrap_deltas.loc[
        bootstrap_deltas["panel_scope"].eq("complete_120")
        & bootstrap_deltas["intersection_rule"].eq("mint_and_stab_and_rde")
        & bootstrap_deltas["reference_single_model"].eq("rde")
    ].iloc[0]
    budget_show = matched_budget.loc[
        matched_budget["panel_scope"].eq("complete_120")
        & matched_budget["consensus_rule"].isin(
            ["mint_and_stab", "mint_and_stab_and_rde"]
        )
    ].copy()
    negative_budget_show = negative_budget.loc[
        negative_budget["panel_scope"].eq("complete_120")
    ].copy()

    lines = [
        "# LibB cross-model agreement audit",
        "",
        "## Question",
        "",
        "If different models independently call the same peptide–Affibody pair a binder, is that pair more likely to be a binder in the direct-retention experiment? Here a direct-retention binder means retention at least 75%. This analysis uses all 120 measured LibB pairs: 12 peptides, 10 Affibodies per peptide, 61 binders, and 59 non-binders.",
        "",
        "The phrase *cross-model confirmation* is more precise here than *cross-validation*. Cross-validation normally means repeatedly splitting training data. Here we keep each trained model fixed and compare whether their positive calls overlap.",
        "",
        "## Binder calls were fixed without retention",
        "",
        "Each model has a cutoff chosen earlier from selection-derived weak-label out-of-fold predictions. A score equal to the cutoff is called positive. Retention values were opened only after these cutoffs were validated and recorded in `pre_retention_lock.json`.",
        "",
        "| Model | Weak-label cutoff |",
        "|---|---:|",
    ]
    for key, spec in PRIMARY_MODELS.items():
        lines.append(f"| {spec['display_name']} | {float(cutoffs[key]):.4f} |")
    lines.extend(
        [
            "",
            "The scores are selection-label scores. They are not retention percentages and are not calibrated probabilities that a pair will bind.",
            "",
            "## Main result: success among model-positive pairs",
            "",
            "Precision is the requested success rate: experimentally confirmed binders divided by pairs called binder. Recall is the fraction of all 61 measured binders recovered by that rule. Intersections require every named model to call the pair positive.",
            "",
            "| Call rule | Called pairs | Confirmed binders | Success rate (precision) | Recall of 61 binders | Peptide-bootstrap 95% CI for precision |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    display = {
        "mint": "MINT",
        "stab": "StaB",
        "rde": "RDE",
        "mint_and_stab": "MINT and StaB",
        "mint_and_rde": "MINT and RDE",
        "stab_and_rde": "StaB and RDE",
        "mint_and_stab_and_rde": "All three",
    }
    for row in primary.itertuples(index=False):
        low, high = _interval_lookup(bootstrap, "complete_120", row.call_rule, "precision")
        lines.append(
            f"| {display[row.call_rule]} | {int(row.calls)} | {int(row.true_positive)} | "
            f"{_pct(row.precision)} | {_pct(row.recall)} | {_pct(low)}–{_pct(high)} |"
        )
    lines.extend(
        [
            "",
            f"Requiring MINT and RDE raises the observed success rate from 66.7% for MINT alone to 75.3% ({int(lookup.loc['mint_and_rde', 'true_positive'])}/{int(lookup.loc['mint_and_rde', 'calls'])}). Requiring all three gives exactly the same 77 pairs on this panel. RDE alone is already 74.7% ({int(rde.true_positive)}/{int(rde.calls)}), so all-three agreement improves precision by only {_pct(float(triple.precision) - float(rde.precision))} and misses one additional binder.",
            "",
            f"The peptide-cluster bootstrap interval for the all-three minus RDE precision difference is {_pct(float(delta_row.ci_lower_2_5))} to {_pct(float(delta_row.ci_upper_97_5))}. These intervals resample the 12 peptide rows while keeping the same fixed set of 10 tested Affibodies; they do not measure uncertainty over new Affibody designs. This does not establish that the small observed increase will generalize.",
            "",
            f"The all-negative pattern contains {int(none_pattern.pairs_in_pattern)} pairs, and 0/{int(none_pattern.pairs_in_pattern)} were measured binders. That result alone is not evidence that agreement improves rejection, because it also rejects fewer pairs than most individual cutoffs.",
            "",
            "## Essential same-budget control",
            "",
            "Consensus normally makes fewer positive calls. To check whether its higher precision comes from model agreement or simply from testing fewer pairs, this control matches the workload exactly: if consensus selects *n* Affibodies for one peptide, take the top *n* scores from one component for that same peptide.",
            "",
            "| Consensus rule | Same-budget component | Pairs tested by each | Consensus binders | Component binders | Consensus precision | Component precision | Difference |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in budget_show.itertuples(index=False):
        lines.append(
            f"| {display[row.consensus_rule]} | {display[row.control_model]} | "
            f"{int(row.calls_each)} | {int(row.consensus_true_positive)} | "
            f"{int(row.control_true_positive)} | {_pct(row.consensus_precision)} | "
            f"{_pct(row.control_precision)} | "
            f"{100.0 * float(row.precision_delta_consensus_minus_control):+.1f} points |"
        )
    lines.extend(
        [
            "",
            "MINT alone, restricted to the same number of highest-scoring pairs per peptide, finds the same **number** of binders—59/84 for the MINT-and-StaB budget and 58/77 for the all-three budget. The pair identities need not be identical. Therefore the current panel does **not** show that cross-model confirmation beats score ranking at an equal workload. This is a diagnostic control, not a standalone MINT deployment rule: its per-peptide budgets were defined by the consensus calls and would themselves require those other models.",
            "",
            "The same control was applied in the opposite direction. For every peptide, each model rejects its bottom-scoring *n* pairs, where *n* is exactly the number rejected by unanimous-negative agreement for that peptide.",
            "",
            "| Rejection rule compared with unanimous negative | Pairs rejected by each | Binders rejected by agreement | Binders rejected by model bottom scores | Maximum retention rejected by agreement | Maximum retention rejected by control |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in negative_budget_show.itertuples(index=False):
        lines.append(
            f"| Bottom {display[row.control_model]} scores at same peptide budgets | "
            f"{int(row.rejected_pairs_each)} | "
            f"{int(row.unanimous_negative_binders_rejected)} | "
            f"{int(row.control_bottom_score_binders_rejected)} | "
            f"{float(row.unanimous_negative_max_retention):.2f} | "
            f"{float(row.control_max_retention):.2f} |"
        )
    lines.extend(
        [
            "",
            "Every single-model bottom-score control also rejects 24/24 non-binders and no binders, with the same maximum retention of 63.66. Thus unanimous-negative agreement has **no demonstrated advantage for rejection either**. It may still be operationally convenient, but a new wet-lab panel is required to show that it is safer than simply rejecting the lowest scores from one model at the same workload.",
            "",
            "## The models agree strongly, but are not independent votes",
            "",
            f"Pairwise model-score Spearman correlations are {min_global_spearman:.2f}–{max_global_spearman:.2f} across all 120 pairs and {min_within_spearman:.2f}–{max_within_spearman:.2f} after correlations are calculated separately within each peptide and averaged. Although the model architectures differ, their predictions are therefore highly similar.",
            "",
            f"In particular, all {int(rde_stab_overlap.model_b_calls)} RDE-positive pairs are also StaB-positive. Adding StaB to an RDE intersection cannot remove any pair on this panel. Exact call overlap and score correlations are saved in `weak_locked_pairwise_call_overlap.csv` and `score_correlations.csv`.",
            "",
            "## Check after removing FALTA",
            "",
            "FALTA is a binder for 11 of 12 peptides, so a rule that selects FALTA can look strong without learning much peptide-specific behavior. FALTA remains in the primary result above; this is a sensitivity check only.",
            "",
            "| Call rule | Called pairs | Confirmed binders | Success rate | Recall of 50 non-FALTA binders |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in no_falta.itertuples(index=False):
        lines.append(
            f"| {display[row.call_rule]} | {int(row.calls)} | {int(row.true_positive)} | "
            f"{_pct(row.precision)} | {_pct(row.recall)} |"
        )
    lines.extend(
        [
            "",
            "All-three precision falls from 75.3% to 71.2% when FALTA is removed. Thus FALTA contributes to the headline success rate, but does not fully account for it. The all-three rule is still only 0.6 percentage points above RDE alone after removing FALTA.",
            "",
            "## Secondary additive-model check",
            "",
            "No standalone additive deployment lock existed, so its cutoff was reconstructed before retention was read by applying the same F1 rule to the canonical weak-label OOF predictions. This is valid as a weak-label-only secondary analysis, but it is not part of the primary three-model question.",
            "",
            "| Rule involving additive baseline | Called pairs | Confirmed binders | Success rate | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    additive_show = additive_summary.loc[
        additive_summary["panel_scope"].eq("complete_120")
        & additive_summary["call_rule"].isin(
            ["additive", "additive_and_mint", "additive_and_stab", "additive_and_rde"]
        )
    ]
    additive_display = {
        "additive": "Additive alone",
        "additive_and_mint": "Additive and MINT",
        "additive_and_stab": "Additive and StaB",
        "additive_and_rde": "Additive and RDE",
    }
    for row in additive_show.itertuples(index=False):
        lines.append(
            f"| {additive_display[row.call_rule]} | {int(row.calls)} | {int(row.true_positive)} | "
            f"{_pct(row.precision)} | {_pct(row.recall)} |"
        )
    lines.extend(
        [
            "",
            "Additive + RDE is 59/78 = 75.6%, only one fewer false positive than RDE alone. This is another small descriptive gain rather than strong evidence for independent confirmation.",
            "",
            "## What should be validated next",
            "",
            "On a new wet-lab panel, lock the existing weak-label cutoffs before results arrive and label candidates by agreement tier: all three positive, exactly two positive, one positive, and all three negative. Sample some candidates from each tier instead of testing only consensus hits. That design can directly estimate whether agreement raises success and whether unanimous negative calls are a safe filter.",
            "",
            "The present 120-pair panel has already been examined repeatedly. It supports the hypothesis that agreement can filter MINT-positive calls, but it does not prove a prospective gain over RDE alone.",
            "",
            "## Separate circular threshold exercise",
            "",
            "For illustration only, each model was also given the cutoff that maximizes F1 on these same 120 known outcomes. This is circular: the outcomes select the rule and then score that rule. These values must not be reported as expected future performance.",
            "",
            "| Model | Same-panel cutoff | Calls | Precision | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in retrospective_thresholds.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.threshold:.4f} | {int(row.calls)} | "
            f"{_pct(row.precision)} | {_pct(row.recall)} |"
        )
    circular_triple = retrospective_summary.loc[
        retrospective_summary["panel_scope"].eq("complete_120")
        & retrospective_summary["call_rule"].eq("mint_and_stab_and_rde")
    ].iloc[0]
    lines.extend(
        [
            "",
            f"With those circular cutoffs, all-three agreement is {int(circular_triple.true_positive)}/{int(circular_triple.calls)} = {_pct(float(circular_triple.precision))} precision and {_pct(float(circular_triple.recall))} recall. It is kept in `retrospective_same_panel_*` files precisely so it cannot be confused with the weak-label-locked primary result.",
            "",
            "## Files",
            "",
            "- `weak_locked_pair_calls.csv`: every score, cutoff, and primary model call for every measured pair.",
            "- `weak_locked_consensus_summary.csv`: single-model and intersection success/recall, including the FALTA-excluded check.",
            "- `weak_locked_agreement_patterns.csv`: all eight possible MINT/StaB/RDE call patterns.",
            "- `weak_locked_per_peptide.csv` and `weak_locked_by_affibody.csv`: where agreement succeeds or fails.",
            "- `peptide_cluster_bootstrap_intervals.csv` and `peptide_cluster_bootstrap_precision_deltas.csv`: uncertainty across peptide rows.",
            "- `same_budget_component_controls.csv`: consensus versus each component at identical per-peptide assay counts.",
            "- `same_budget_component_pair_calls.csv`: exact pair membership for that control.",
            "- `same_budget_negative_controls.csv` and `same_budget_negative_pair_calls.csv`: unanimous-negative rejection versus each model's bottom scores at identical per-peptide budgets.",
            "- `secondary_additive_consensus_summary.csv`: retention-blind additive cutoff analysis.",
            "- `retrospective_same_panel_*`: explicitly circular threshold results.",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    parser.add_argument("--weak-oof", type=Path, default=DEFAULT_WEAK_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    _require(int(args.bootstrap_draws) >= 1000, "bootstrap draws must be at least 1000")
    _require(0 <= int(args.bootstrap_seed) < 2**32, "bootstrap seed is outside uint32")

    cutoffs, pre_retention_seal = load_weak_cutoffs_before_retention(
        args.lock_dir, args.weak_oof, args.output_dir
    )
    frame = load_and_validate_predictions(args.predictions, args.source_manifest)
    calls = add_calls(frame, cutoffs, include_additive=True)
    rules = primary_rules()
    main_summary = consensus_summary(
        calls,
        rules,
        "weak_label_locked_cutoffs_retention_used_only_for_this_audit",
    )
    overlap = pairwise_call_overlap(calls)
    patterns = agreement_patterns(calls)
    per_peptide = stratified_rule_table(
        calls, rules, "peptide_design_code", include_falta_excluded_scope=True
    )
    by_affibody = stratified_rule_table(
        calls, rules, "affibody_design_code", include_falta_excluded_scope=False
    )
    correlations = score_correlations(calls, include_additive=True)
    bootstrap, bootstrap_deltas = peptide_cluster_bootstrap(
        calls, rules, int(args.bootstrap_draws), int(args.bootstrap_seed)
    )
    matched_budget, matched_budget_exact = matched_budget_component_controls(
        calls, rules, int(args.bootstrap_draws), int(args.bootstrap_seed)
    )
    negative_budget, negative_budget_exact = matched_budget_negative_controls(calls)
    additive_summary = consensus_summary(
        calls,
        secondary_additive_rules(),
        "secondary_additive_cutoff_derived_from_weak_oof_without_retention",
    )
    _, retrospective_thresholds, retrospective_calls, retrospective_summary = (
        build_retrospective_threshold_analysis(frame)
    )

    exact_columns = [
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
        ADDITIVE_SCORE_COLUMN,
        *(str(spec["score_column"]) for spec in PRIMARY_MODELS.values()),
        *(f"cutoff_{key}" for key in (*PRIMARY_MODELS, ADDITIVE_KEY)),
        *(f"call_{key}" for key in (*PRIMARY_MODELS, ADDITIVE_KEY)),
        "primary_positive_call_count",
        "primary_agreement_pattern",
    ]
    exact_calls = calls[exact_columns].copy()
    exact_calls["call_mint_and_stab"] = (
        exact_calls["call_mint"] & exact_calls["call_stab"]
    ).astype(int)
    exact_calls["call_mint_and_rde"] = (
        exact_calls["call_mint"] & exact_calls["call_rde"]
    ).astype(int)
    exact_calls["call_stab_and_rde"] = (
        exact_calls["call_stab"] & exact_calls["call_rde"]
    ).astype(int)
    exact_calls["call_all_three"] = (
        exact_calls["call_mint"] & exact_calls["call_stab"] & exact_calls["call_rde"]
    ).astype(int)
    exact_calls["is_FALTA"] = exact_calls["affibody_design_code"].eq(FALTA_CODE).astype(int)
    exact_calls["analysis_status"] = "weak_label_locked_cutoffs_retention_used_only_for_this_audit"

    outputs: dict[str, pd.DataFrame] = {
        "weak_locked_pair_calls.csv": exact_calls,
        "weak_locked_consensus_summary.csv": main_summary,
        "weak_locked_pairwise_call_overlap.csv": overlap,
        "weak_locked_agreement_patterns.csv": patterns,
        "weak_locked_per_peptide.csv": per_peptide,
        "weak_locked_by_affibody.csv": by_affibody,
        "score_correlations.csv": correlations,
        "peptide_cluster_bootstrap_intervals.csv": bootstrap,
        "peptide_cluster_bootstrap_precision_deltas.csv": bootstrap_deltas,
        "same_budget_component_controls.csv": matched_budget,
        "same_budget_component_pair_calls.csv": matched_budget_exact,
        "same_budget_negative_controls.csv": negative_budget,
        "same_budget_negative_pair_calls.csv": negative_budget_exact,
        "secondary_additive_consensus_summary.csv": additive_summary,
        "retrospective_same_panel_thresholds.csv": retrospective_thresholds,
        "retrospective_same_panel_pair_calls.csv": retrospective_calls,
        "retrospective_same_panel_consensus_summary.csv": retrospective_summary,
    }
    output_dir = Path(args.output_dir).resolve()
    for filename, table in outputs.items():
        _write_csv_exclusive(output_dir / filename, table)

    report = render_report(
        cutoffs,
        main_summary,
        overlap,
        patterns,
        correlations,
        bootstrap,
        bootstrap_deltas,
        matched_budget,
        negative_budget,
        additive_summary,
        retrospective_thresholds,
        retrospective_summary,
    )
    _write_text_exclusive(output_dir / "report.md", report)

    file_records = {
        filename: {**_file_record(output_dir / filename), "rows": len(table)}
        for filename, table in outputs.items()
    }
    file_records["report.md"] = _file_record(output_dir / "report.md")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": time.time() - started,
        "code": _file_record(Path(__file__)),
        "inputs": {
            "averaged_score_predictions": _file_record(args.predictions),
            "averaged_score_source_manifest": _file_record(args.source_manifest),
            "weak_oof_predictions": _file_record(args.weak_oof),
            "deployment_lock_directory": _relative(args.lock_dir),
        },
        "pre_retention_seal": pre_retention_seal,
        "retention_panel": {
            "pairs": len(frame),
            "peptides": frame["peptide_design_code"].nunique(),
            "affibodies": frame["affibody_design_code"].nunique(),
            "binders": int(frame["target_binder"].sum()),
            "non_binders": len(frame) - int(frame["target_binder"].sum()),
            "binder_definition": "target_retention >= 75",
            "row_id_membership_sha256": _membership_sha256(frame["eval_row_id"]),
        },
        "primary_analysis": {
            "models": list(PRIMARY_MODELS),
            "call_definition": "model score >= its weak-label-only locked cutoff",
            "cutoffs": {key: float(cutoffs[key]) for key in PRIMARY_MODELS},
            "retention_labels_read_for_cutoff": False,
            "retention_use": "outcome evaluation only on an already examined panel",
        },
        "secondary_additive_analysis": {
            "cutoff": float(cutoffs[ADDITIVE_KEY]),
            "cutoff_source": "canonical weak-label OOF predictions",
            "retention_labels_read_for_cutoff": False,
        },
        "bootstrap": {
            "unit": "peptide_design_code",
            "draws": int(args.bootstrap_draws),
            "seed_complete_120": int(args.bootstrap_seed),
            "seed_FALTA_excluded": int(args.bootstrap_seed) + 100003,
            "interval": "percentile 2.5% to 97.5%",
        },
        "retrospective_same_panel_analysis": {
            "status": "circular; cutoffs fitted and evaluated on the same 120 retention labels",
            "allowed_interpretation": "descriptive illustration only, never prospective performance",
        },
        "outputs": file_records,
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
