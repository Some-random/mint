#!/usr/bin/env python3
"""Build a candidate assembly with explicitly separated retrospective metrics.

This program packages seven already-assembled, weak-label-locked candidate
menus.  MINT+RDE is the sole primary candidate rule.  It does not fit a model,
change a prospective cutoff, combine menus by voting, or rerank any candidate.
The seven input shortlists are immutable, hashed artifacts before corrected-
panel outcomes are loaded for two retrospective displays: ranking metrics and
an F1-optimized threshold exercise on the already-measured 120 pairs.  Those
outcomes and thresholds are never passed to prospective ranking code.  The
proposed ten-target scientific-review subset is an unchanged subset of the
primary 12-target menu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()

EXPECTED_TARGET_SEQUENCES = {
    "AF": "SLLAFITQV",
    "AH": "SLLAHITQV",
    "DP": "SLLDPITQV",
    "EA": "SLLEAITQV",
    "EL": "SLLELITQV",
    "LL": "SLLLLITQV",
    "LV": "SLLLVITQV",
    "MW": "SLLMWITQV",
    "NF": "SLLNFITQV",
    "PH": "SLLPHITQV",
    "TL": "SLLTLITQV",
    "VV": "SLLVVITQV",
}
READY_TARGETS = ("AF", "AH", "DP", "EA", "LV", "MW", "NF", "PH", "TL", "VV")
LIBB_CODE_ALPHABET = frozenset("ADEFIKLMNSTVY")
DISPLAYED_AFFIBODY_CODE_POSITIONS = (6, 10, 13, 14, 17)
CRYSTAL_ALIGNED_AFFIBODY_CODE_POSITIONS = (8, 12, 15, 16, 19)

FORBIDDEN_OUTCOME_COLUMN_FRAGMENTS = (
    "retention",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
)


@dataclass(frozen=True)
class ModelRole:
    key: str
    display_name: str
    lock_id: str
    cli_flag: str
    components: tuple[str, ...]


MODEL_ROLES = (
    ModelRole(
        "mint_rde_primary",
        "MINT + RDE ensemble (primary)",
        "libb-native-mint-rde-primary-ensemble-weak-oof-v1",
        "primary_dir",
        ("mint_layer5", "rde_network_designed_3fold"),
    ),
    ModelRole(
        "mint_control",
        "Frozen MINT layer 5 (control)",
        "libb-native-mint-layer5-control-weak-oof-v1",
        "mint_dir",
        ("mint_layer5",),
    ),
    ModelRole(
        "mint_stab",
        "MINT + StaB ensemble",
        "libb-native-mint-stab-comparator-weak-oof-v1",
        "mint_stab_dir",
        ("mint_layer5", "stab_designed_ordered"),
    ),
    ModelRole(
        "equal_all4",
        "Equal-logit four-model ensemble",
        "libb-native-equal-all4-comparator-weak-oof-v1",
        "equal_all4_dir",
        (
            "additive_7site",
            "mint_layer5",
            "stab_designed_ordered",
            "rde_network_designed_3fold",
        ),
    ),
    ModelRole(
        "stacker_all4",
        "Weak-label-fitted four-model stacker",
        "libb-native-all4-stacker-comparator-weak-oof-v1",
        "stacker_dir",
        (
            "additive_7site",
            "mint_layer5",
            "stab_designed_ordered",
            "rde_network_designed_3fold",
        ),
    ),
    ModelRole(
        "rde_standalone",
        "RDE native-projection readout",
        "libb-native-rde-standalone-weak-oof-v1",
        "rde_dir",
        ("rde_network_designed_3fold",),
    ),
    ModelRole(
        "stab_standalone",
        "StaB native-projection readout",
        "libb-native-stab-standalone-weak-oof-v1",
        "stab_dir",
        ("stab_designed_ordered",),
    ),
)

COMPONENT_SCORE_SPECS = {
    "additive_7site": {
        "cli_flag": "additive_scores_dir",
        "prefix": "additive_7site_",
        "required": ("additive_7site_logit", "additive_7site_score"),
    },
    "mint_layer5": {
        "cli_flag": "mint_scores_dir",
        "prefix": "mint_layer5_",
        "required": ("mint_layer5_logit", "mint_layer5_score"),
    },
    "rde_network_designed_3fold": {
        "cli_flag": "rde_scores_dir",
        "prefix": "rde_",
        "required": (
            "rde_logit",
            "rde_probability",
            "rde_seed_mean_probability",
            "rde_seed_probability_sd",
        ),
    },
    "stab_designed_ordered": {
        "cli_flag": "stab_scores_dir",
        "prefix": "stab_",
        "required": (
            "stab_logit",
            "stab_probability",
            "stab_seed_mean_probability",
            "stab_seed_probability_sd",
        ),
    },
}

METRIC_MODEL_SPECS = (
    (
        "additive_7site",
        "Additive seven-position baseline",
        "mean_logit__additive_7site",
        "additive_7site",
    ),
    (
        "mint_layer5",
        "Frozen MINT layer 5",
        "mean_logit__mint_layer5",
        "mint_layer5",
    ),
    (
        "stab_native",
        "StaB native-projection readout",
        "mean_logit__stab_designed_ordered_native_projection",
        "stab_designed_ordered_native_projection",
    ),
    (
        "rde_native",
        "RDE native-projection readout",
        "mean_logit__rde_network_designed_3fold_native_projection",
        "rde_network_designed_3fold_native_projection",
    ),
    (
        "mint_rde",
        "MINT + RDE ensemble",
        "mean_logit__mint_layer5__rde_network_designed_3fold_native_projection",
        "mean_logit_mint_rde_native",
    ),
    (
        "mint_stab",
        "MINT + StaB ensemble",
        "mean_logit__mint_layer5__stab_designed_ordered_native_projection",
        "mean_logit_mint_stab_native",
    ),
    (
        "equal_all4",
        "Equal-logit four-model ensemble",
        "mean_logit__additive_7site__mint_layer5__stab_designed_ordered_native_projection__rde_network_designed_3fold_native_projection",
        "mean_logit_all4_native",
    ),
    (
        "stacker_all4",
        "Weak-label-fitted four-model stacker",
        "cross_fitted_nonnegative_stack__all4_native_projection",
        "nonnegative_stack_all4_native",
    ),
)

RETROSPECTIVE_PANEL_ROWS = 120
RETROSPECTIVE_PANEL_BINDERS = 61
RETROSPECTIVE_PANEL_NONBINDERS = 59
RETROSPECTIVE_BINDER_THRESHOLD = 75.0
RETROSPECTIVE_PREDICTION_ID_COLUMNS = (
    "eval_row_id",
    "peptide_design_code",
    "affibody_design_code",
    "target_retention",
    "target_binder",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbidden_outcome_columns(columns: Iterable[str]) -> list[str]:
    leaked = []
    for column in columns:
        normalized = str(column).strip().lower()
        if any(fragment in normalized for fragment in FORBIDDEN_OUTCOME_COLUMN_FRAGMENTS):
            leaked.append(str(column))
    return sorted(set(leaked))


def _validate_private_output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    _require(
        output == PRIVATE_ROOT or PRIVATE_ROOT in output.parents,
        "candidate handoff must remain under private_data",
    )
    _require(output != PRIVATE_ROOT, "refusing to use private_data itself as output")
    return output


def _load_json(path: Path, label: str) -> dict:
    _require(path.is_file(), f"{label} is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _as_boolean(series: pd.Series, label: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    _require(not series.isna().any(), f"{label} contains a missing value")
    converted = series.map(mapping)
    _require(not converted.isna().any(), f"{label} contains a non-boolean value")
    return converted.astype(bool)


def _hamming_distance(left: str, right: str) -> int:
    _require(len(left) == len(right) == 5, "LibB codes must contain five characters")
    return sum(a != b for a, b in zip(left, right))


def _load_candidate_universe_manifest(
    directory: Path, shortlist_manifests: dict[str, dict],
) -> tuple[dict, Path]:
    directory = directory.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path, "candidate-universe manifest")
    _require(
        manifest.get("schema_version")
        == "libb-existing-target-selection-missed-candidates-v1",
        "unexpected candidate-universe schema",
    )
    expected_hashes = {
        item.get("candidate_universe", {}).get("manifest_sha256")
        for item in shortlist_manifests.values()
    }
    _require(
        expected_hashes == {sha256_file(manifest_path)},
        "candidate-universe manifest does not match the assembled shortlists",
    )
    scope = manifest.get("scope", {})
    _require(scope.get("library") == "LibB", "candidate universe is not LibB")
    _require(
        scope.get("target_peptide_codes") == list(EXPECTED_TARGET_SEQUENCES),
        "candidate-universe target order or identities changed",
    )
    _require(
        scope.get("target_peptide_full_sequences") == EXPECTED_TARGET_SEQUENCES,
        "candidate-universe target sequences changed",
    )
    _require(
        scope.get("affibody_design_alphabet_each_position")
        == "".join(sorted(LIBB_CODE_ALPHABET)),
        "candidate-universe LibB alphabet changed",
    )
    _require(
        scope.get("affibody_mutable_positions", {}).get(
            "provider_displayed_58aa_positions_1_based"
        ) == list(DISPLAYED_AFFIBODY_CODE_POSITIONS),
        "candidate-universe displayed Affibody positions changed",
    )
    _require(
        scope.get("affibody_mutable_positions", {}).get(
            "provider_crystal_aligned_labels_1_based"
        ) == list(CRYSTAL_ALIGNED_AFFIBODY_CODE_POSITIONS),
        "candidate-universe crystal-aligned Affibody positions changed",
    )
    outputs = manifest.get("outputs", {})
    exclusions = manifest.get("exclusions", {})
    _require(int(outputs.get("candidate_rows", -1)) == 4_447_848,
             "candidate-universe row count changed")
    _require(int(exclusions.get("directly_measured_exact_pairs", -1)) == 120,
             "measured-pair exclusion count changed")
    _require(int(exclusions.get("pooled_positive_nonmeasured_pairs", -1)) == 7_548,
             "nonmeasured pooled-positive exclusion count changed")
    _require(int(exclusions.get("measured_and_pooled_positive_overlap", -1)) == 51,
             "measured/pooled-positive overlap changed")
    _require(int(exclusions.get("excluded_union_rows", -1)) == 7_668,
             "candidate-universe unique exclusion count changed")
    selection = manifest.get("selection_rule", {})
    _require(
        selection.get("operation")
        == "outer-join R009/R010 by exact pair, zero-fill, sum counts, then tie-inclusive top 2%",
        "candidate-universe positive rule changed",
    )
    weak = manifest.get("weak_label_audit", {})
    _require(int(weak.get("strict_training_rows", -1)) == 30_648,
             "strict LibB training count changed")
    _require(int(weak.get("strict_training_unique_affibodies", -1)) == 23_081,
             "strict-training Affibody count changed")
    return manifest, manifest_path


def _load_metric_comparison(
    weak_path: Path, corrected_path: Path,
) -> pd.DataFrame:
    weak_path = weak_path.expanduser().resolve()
    corrected_path = corrected_path.expanduser().resolve()
    weak = pd.read_csv(weak_path)
    corrected = pd.read_csv(corrected_path)
    weak_required = {
        "candidate",
        "rows",
        "positive",
        "within_peptide_evaluable",
        "within_peptide_ap",
    }
    corrected_required = {
        "model_key",
        "within_peptide_average_precision",
        "within_peptide_auroc",
        "within_peptide_spearman",
        "binary_evaluable_peptides",
        "spearman_evaluable_peptides",
    }
    _require(weak_required.issubset(weak.columns), "weak metric table schema changed")
    _require(corrected_required.issubset(corrected.columns),
             "corrected-120 metric table schema changed")
    _require(not weak["candidate"].astype(str).duplicated().any(),
             "weak metric table repeats a candidate")
    _require(not corrected["model_key"].astype(str).duplicated().any(),
             "corrected-120 metric table repeats a model")
    weak_by_key = weak.set_index("candidate", drop=False)
    corrected_by_key = corrected.set_index("model_key", drop=False)
    rows = []
    for key, display, weak_key, corrected_key in METRIC_MODEL_SPECS:
        _require(weak_key in weak_by_key.index, f"weak metric missing {weak_key}")
        _require(corrected_key in corrected_by_key.index,
                 f"corrected-120 metric missing {corrected_key}")
        weak_row = weak_by_key.loc[weak_key]
        corrected_row = corrected_by_key.loc[corrected_key]
        _require(int(weak_row["rows"]) == 10_181, "weak metric row count changed")
        _require(int(weak_row["positive"]) == 7_939, "weak positive count changed")
        _require(int(weak_row["within_peptide_evaluable"]) == 169,
                 "weak evaluable-peptide count changed")
        _require(int(corrected_row["binary_evaluable_peptides"]) == 11,
                 "corrected binary-evaluable peptide count changed")
        _require(int(corrected_row["spearman_evaluable_peptides"]) == 12,
                 "corrected Spearman-evaluable peptide count changed")
        rows.append({
            "model_key": key,
            "model": display,
            "weak_label_within_peptide_average_precision": float(
                weak_row["within_peptide_ap"]
            ),
            "corrected_120_retention_within_peptide_average_precision": float(
                corrected_row["within_peptide_average_precision"]
            ),
            "corrected_120_retention_within_peptide_auroc": float(
                corrected_row["within_peptide_auroc"]
            ),
            "corrected_120_retention_within_peptide_spearman": float(
                corrected_row["within_peptide_spearman"]
            ),
            "weak_label_evaluable_peptide_groups": 169,
            "corrected_120_binary_evaluable_peptides": 11,
            "corrected_120_spearman_evaluable_peptides": 12,
        })
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(float)
    _require(np.isfinite(numeric).all(), "metric comparison contains a non-finite value")
    return result


def _select_retrospective_f1_threshold(
    scores: Sequence[float], labels: Sequence[int],
) -> dict[str, Any]:
    """Choose ``score >= threshold`` using the established exact tie rule.

    Every distinct observed score is considered.  The objective and tie order
    match ``summarize_libb_operational_metrics.select_f1_threshold``: maximize
    F1, then precision, then prefer fewer recommendations, then prefer the
    higher numerical threshold.
    """

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    _require(score_array.ndim == 1 and label_array.ndim == 1,
             "retrospective threshold inputs must be vectors")
    _require(len(score_array) == len(label_array) and len(score_array) > 0,
             "retrospective threshold inputs have invalid lengths")
    _require(bool(np.isfinite(score_array).all()),
             "retrospective threshold scores contain non-finite values")
    _require(set(np.unique(label_array)) == {0, 1},
             "retrospective threshold labels must contain both binary classes")

    positives = int(label_array.sum())
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for threshold in np.unique(score_array):
        recommended = score_array >= float(threshold)
        count = int(recommended.sum())
        true_positive = int(np.logical_and(recommended, label_array == 1).sum())
        false_positive = count - true_positive
        false_negative = positives - true_positive
        true_negative = len(label_array) - true_positive - false_positive - false_negative
        precision = Fraction(true_positive, count) if count else Fraction(0, 1)
        recall = Fraction(true_positive, positives) if positives else Fraction(0, 1)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (
            Fraction(2 * true_positive, denominator)
            if denominator else Fraction(0, 1)
        )
        result = {
            "retrospective_threshold": float(threshold),
            "recommended_pairs": count,
            "true_positive_binders": true_positive,
            "false_positive_nonbinders": false_positive,
            "missed_binders": false_negative,
            "true_negative_nonbinders": true_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        candidates.append(((f1, precision, -count, float(threshold)), result))
    return max(candidates, key=lambda item: item[0])[1]


def _load_retrospective_threshold_analysis(
    predictions_path: Path,
    corrected_metrics_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the display-only threshold summary and known-panel pair lists."""

    predictions_path = predictions_path.expanduser().resolve()
    corrected_metrics_path = corrected_metrics_path.expanduser().resolve()
    _require(predictions_path.is_file(),
             f"corrected-120 averaged predictions are missing: {predictions_path}")
    manifest_path = predictions_path.parent / "manifest.json"
    manifest = _load_json(manifest_path, "corrected-120 ensemble audit manifest")
    _require(
        manifest.get("schema_version")
        == "libb-native-projection-ensemble-retention-audit-v1",
        "unexpected corrected-120 ensemble audit schema",
    )
    _require(corrected_metrics_path.parent == predictions_path.parent,
             "corrected metrics and predictions must come from the same audit")
    outputs = manifest.get("outputs", {})
    prediction_record = outputs.get(predictions_path.name, {})
    metric_record = outputs.get(corrected_metrics_path.name, {})
    _require(int(prediction_record.get("rows", -1)) == RETROSPECTIVE_PANEL_ROWS,
             "corrected-120 prediction row count changed in its manifest")
    _require(prediction_record.get("sha256") == sha256_file(predictions_path),
             "corrected-120 predictions differ from their manifest")
    _require(metric_record.get("sha256") == sha256_file(corrected_metrics_path),
             "corrected-120 metrics differ from their manifest")
    panel_record = manifest.get("retention_panel", {})
    _require(int(panel_record.get("pairs", -1)) == RETROSPECTIVE_PANEL_ROWS,
             "corrected retention panel is not 120 pairs")
    _require(int(panel_record.get("binders", -1)) == RETROSPECTIVE_PANEL_BINDERS,
             "corrected retention panel binder count changed")
    _require(int(panel_record.get("non_binders", -1)) == RETROSPECTIVE_PANEL_NONBINDERS,
             "corrected retention panel nonbinder count changed")
    _require(panel_record.get("binder_definition") == "retention >= 75",
             "corrected retention panel binder definition changed")
    _require(manifest.get("retention_used_for_training_selection_or_cutoff") is False,
             "source audit reports retention-dependent model fitting or selection")

    frame = pd.read_csv(predictions_path)
    score_columns = [spec[3] for spec in METRIC_MODEL_SPECS]
    expected_columns = set(RETROSPECTIVE_PREDICTION_ID_COLUMNS) | set(score_columns)
    _require(set(frame.columns) == expected_columns,
             "corrected-120 averaged prediction schema changed")
    _require(len(frame) == RETROSPECTIVE_PANEL_ROWS,
             "corrected-120 averaged predictions are not 120 rows")
    _require(not frame["eval_row_id"].astype(str).duplicated().any(),
             "corrected-120 averaged predictions repeat a pair ID")
    _require(
        not frame[["peptide_design_code", "affibody_design_code"]].astype(str)
        .duplicated().any(),
        "corrected-120 averaged predictions repeat a peptide-Affibody pair",
    )
    _require(set(frame["peptide_design_code"].astype(str)) == set(EXPECTED_TARGET_SEQUENCES),
             "corrected-120 averaged predictions have unexpected peptide codes")
    peptide_sizes = frame.groupby("peptide_design_code", sort=False).size()
    _require(len(peptide_sizes) == 12 and peptide_sizes.eq(10).all(),
             "corrected-120 panel is not a complete 12-by-10 matrix")
    affibody_sets = frame.groupby("peptide_design_code", sort=False)[
        "affibody_design_code"
    ].agg(lambda values: frozenset(map(str, values)))
    _require(len(set(affibody_sets.tolist())) == 1,
             "corrected-120 peptide rows do not share the same ten Affibodies")
    _require(len(affibody_sets.iloc[0]) == 10,
             "corrected-120 panel does not contain ten Affibodies")

    retention = pd.to_numeric(frame["target_retention"], errors="raise").to_numpy(float)
    binder = pd.to_numeric(frame["target_binder"], errors="raise").to_numpy(int)
    _require(bool(np.isfinite(retention).all()),
             "corrected-120 retention contains a non-finite value")
    _require(set(np.unique(binder)) == {0, 1},
             "corrected-120 binder labels are not binary")
    _require(bool(np.array_equal(binder, (retention >= RETROSPECTIVE_BINDER_THRESHOLD).astype(int))),
             "corrected-120 binder labels do not equal retention >= 75")
    _require(int(binder.sum()) == RETROSPECTIVE_PANEL_BINDERS,
             "corrected-120 binder count changed")
    for column in score_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        _require(bool(np.isfinite(values).all()), f"{column} contains a non-finite score")
        _require(bool(((values >= 0.0) & (values <= 1.0)).all()),
                 f"{column} contains a score outside [0,1]")

    target_order = {code: index for index, code in enumerate(EXPECTED_TARGET_SEQUENCES)}
    summary_rows: list[dict[str, Any]] = []
    recommendation_frames: list[pd.DataFrame] = []
    code_rows: list[dict[str, Any]] = []
    base_columns = list(RETROSPECTIVE_PREDICTION_ID_COLUMNS)
    for order, (model_key, display_name, _weak_key, score_column) in enumerate(
        METRIC_MODEL_SPECS, start=1
    ):
        chosen = _select_retrospective_f1_threshold(frame[score_column], binder)
        decisions = frame[base_columns].copy()
        decisions["model_score"] = pd.to_numeric(frame[score_column], errors="raise")
        decisions["retrospective_threshold"] = float(chosen["retrospective_threshold"])
        decisions["retrospectively_recommended"] = decisions["model_score"].ge(
            float(chosen["retrospective_threshold"])
        )
        decisions["model_order"] = order
        decisions["model_key"] = model_key
        decisions["model"] = display_name
        decisions["score_column"] = score_column
        decisions["peptide_9mer_sequence"] = decisions["peptide_design_code"].map(
            EXPECTED_TARGET_SEQUENCES
        )
        decisions["peptide_order"] = decisions["peptide_design_code"].map(target_order)
        decisions = decisions.sort_values(
            ["peptide_order", "model_score", "affibody_design_code", "eval_row_id"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        decisions["rank_within_peptide_on_measured_panel"] = (
            decisions.groupby("peptide_design_code", sort=False).cumcount() + 1
        )
        recommendations = decisions.loc[decisions["retrospectively_recommended"]].copy()
        _require(len(recommendations) == int(chosen["recommended_pairs"]),
                 f"retrospective recommendation count mismatch for {model_key}")
        recommendations["measured_recommendation_outcome"] = np.where(
            recommendations["target_binder"].eq(1), "binder", "nonbinder"
        )
        recommendations["score_minus_threshold"] = (
            recommendations["model_score"] - recommendations["retrospective_threshold"]
        )
        recommendation_frames.append(recommendations)
        summary_rows.append({
            "model_order": order,
            "model_key": model_key,
            "model": display_name,
            "score_column": score_column,
            "decision_rule": "model_score >= retrospective_threshold",
            "retrospective_threshold": chosen["retrospective_threshold"],
            "recommended_pairs": chosen["recommended_pairs"],
            "true_positive_binders": chosen["true_positive_binders"],
            "false_positive_nonbinders": chosen["false_positive_nonbinders"],
            "missed_binders": chosen["missed_binders"],
            "true_negative_nonbinders": chosen["true_negative_nonbinders"],
            "precision": chosen["precision"],
            "recall": chosen["recall"],
            "f1": chosen["f1"],
            "panel_pairs": RETROSPECTIVE_PANEL_ROWS,
            "panel_binders": RETROSPECTIVE_PANEL_BINDERS,
            "panel_nonbinders": RETROSPECTIVE_PANEL_NONBINDERS,
            "binder_definition": "target_retention >= 75",
            "threshold_selection_status": (
                "retrospective_F1_optimum_on_same_known_120_pairs_not_prospective"
            ),
        })
        for peptide_code in EXPECTED_TARGET_SEQUENCES:
            subset = recommendations.loc[
                recommendations["peptide_design_code"].eq(peptide_code)
            ].sort_values(
                ["rank_within_peptide_on_measured_panel", "affibody_design_code"],
                kind="mergesort",
            )
            codes = subset["affibody_design_code"].astype(str).tolist()
            code_rows.append({
                "model_order": order,
                "model_key": model_key,
                "model": display_name,
                "peptide_design_code": peptide_code,
                "peptide_9mer_sequence": EXPECTED_TARGET_SEQUENCES[peptide_code],
                "recommended_pair_count": len(codes),
                "recommended_affibody_codes_in_score_order": ";".join(codes),
            })

    summary = pd.DataFrame(summary_rows).sort_values("model_order", kind="mergesort")
    recommendations = pd.concat(recommendation_frames, ignore_index=True).sort_values(
        ["model_order", "peptide_order", "rank_within_peptide_on_measured_panel"],
        kind="mergesort",
    )
    codes_by_target = pd.DataFrame(code_rows).sort_values(
        ["model_order", "peptide_design_code"],
        key=lambda series: (
            series.map(target_order)
            if series.name == "peptide_design_code" else series
        ),
        kind="mergesort",
    )
    _require(int(summary["recommended_pairs"].sum()) == len(recommendations),
             "retrospective recommendation totals do not reconcile")
    _require(len(codes_by_target) == len(METRIC_MODEL_SPECS) * len(EXPECTED_TARGET_SEQUENCES),
             "retrospective code appendix is incomplete")
    provenance = {
        "path": str(predictions_path),
        "sha256": sha256_file(predictions_path),
        "rows": len(frame),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_schema": manifest["schema_version"],
        "used_for": (
            "display-only retrospective F1 thresholds and measured-pair recommendation list; "
            "never passed to prospective candidate ranking"
        ),
    }
    recommendation_columns = [
        "model_order", "model_key", "model", "score_column",
        "eval_row_id", "peptide_design_code", "peptide_9mer_sequence",
        "affibody_design_code", "rank_within_peptide_on_measured_panel",
        "model_score", "retrospective_threshold", "score_minus_threshold",
        "target_retention", "target_binder", "measured_recommendation_outcome",
        "retrospectively_recommended",
    ]
    return (
        summary.reset_index(drop=True),
        recommendations[recommendation_columns].reset_index(drop=True),
        codes_by_target.reset_index(drop=True),
        provenance,
    )


def _read_primary_cutoff_audit(directory: Path, manifest: dict) -> pd.DataFrame:
    path = directory / "wetlab_shortlist_summary.csv"
    _require(path.is_file(), "primary shortlist summary is missing")
    _require(
        manifest.get("outputs", {}).get("summary_sha256") == sha256_file(path),
        "primary shortlist summary differs from its manifest",
    )
    summary = pd.read_csv(path)
    required = {
        "peptide_design_code",
        "universe_rows_scored",
        "rows_above_locked_cutoff",
        "wetlab_candidates_selected",
    }
    _require(required.issubset(summary.columns), "primary shortlist summary schema changed")
    _require(len(summary) == 12, "primary shortlist summary is not 12 rows")
    _require(
        set(summary["peptide_design_code"].astype(str)) == set(EXPECTED_TARGET_SEQUENCES),
        "primary shortlist summary does not cover the exact targets",
    )
    summary["raw_top_ten_all_pass_locked_cutoff"] = (
        pd.to_numeric(summary["rows_above_locked_cutoff"], errors="raise").ge(10)
        & pd.to_numeric(summary["wetlab_candidates_selected"], errors="raise").eq(10)
    )
    _require(summary["raw_top_ten_all_pass_locked_cutoff"].all(),
             "the locked cutoff removes at least one raw top-ten primary candidate")
    return summary.sort_values("peptide_design_code", kind="mergesort").reset_index(drop=True)


def _load_component_scores_for_primary(
    primary: pd.DataFrame,
    score_roots: dict[str, Path],
    all4_manifest: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict]]]:
    """Join the four exact deployed component outputs onto the 120 primary rows."""
    enriched = primary.copy()
    provenance: dict[str, dict[str, dict]] = {}
    source_records = all4_manifest.get("component_score_files", {})
    _require(set(source_records) == set(EXPECTED_TARGET_SEQUENCES),
             "all-four component provenance does not cover all targets")
    for component, spec in COMPONENT_SCORE_SPECS.items():
        root = score_roots[component].expanduser().resolve()
        _require(root.is_dir(), f"component score directory is missing: {root}")
        blocks = []
        provenance[component] = {}
        for peptide in EXPECTED_TARGET_SEQUENCES:
            record = source_records[peptide].get(component)
            _require(isinstance(record, dict),
                     f"all-four manifest lacks {component} for {peptide}")
            source_path = Path(str(record.get("score_path", ""))).resolve()
            expected_path = (root / f"peptide_{peptide}.parquet").resolve()
            _require(source_path == expected_path,
                     f"provided {component} root differs from deployed {peptide} source")
            _require(source_path.is_file(), f"missing deployed score file: {source_path}")
            observed_sha = sha256_file(source_path)
            _require(observed_sha == record.get("score_sha256"),
                     f"deployed {component}/{peptide} score hash changed")
            frame = pd.read_parquet(source_path)
            key_columns = {"pair_uid", "peptide_design_code", "affibody_design_code"}
            _require(key_columns.issubset(frame.columns),
                     f"{component}/{peptide} score schema lacks candidate keys")
            score_columns = [
                column for column in frame.columns
                if str(column).startswith(str(spec["prefix"]))
            ]
            _require(set(spec["required"]).issubset(score_columns),
                     f"{component}/{peptide} score schema lacks deployed outputs")
            if component in {"rde_network_designed_3fold", "stab_designed_ordered"}:
                seed_columns = [
                    column for column in score_columns
                    if re.fullmatch(rf"{spec['prefix']}probability_seed_\d+", str(column))
                ]
                _require(len(seed_columns) == 5,
                         f"{component}/{peptide} does not contain five seed scores")
            wanted = set(
                primary.loc[primary["peptide_design_code"].eq(peptide), "pair_uid"].astype(str)
            )
            frame["pair_uid"] = frame["pair_uid"].astype(str)
            selected = frame.loc[frame["pair_uid"].isin(wanted), [
                "pair_uid", "peptide_design_code", "affibody_design_code", *score_columns
            ]].copy()
            _require(len(selected) == 10 and set(selected["pair_uid"]) == wanted,
                     f"{component}/{peptide} does not cover the primary ten")
            _require(not selected["pair_uid"].duplicated().any(),
                     f"{component}/{peptide} repeats a primary pair")
            numeric_scores = selected[score_columns].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=np.float64)
            _require(np.isfinite(numeric_scores).all(),
                     f"{component}/{peptide} contains a non-finite score")
            probability_columns = [
                column for column in score_columns
                if "probability" in column or column.endswith("_score")
            ]
            if probability_columns:
                probabilities = selected[probability_columns].apply(
                    pd.to_numeric, errors="raise"
                ).to_numpy(dtype=np.float64)
                _require(((probabilities >= 0.0) & (probabilities <= 1.0)).all(),
                         f"{component}/{peptide} probability leaves [0,1]")
            blocks.append(selected)
            provenance[component][peptide] = {
                "path": str(source_path),
                "sha256": observed_sha,
                "producer_receipt_path": str(record.get("producer_receipt_path")),
                "producer_receipt_sha256": str(record.get("producer_receipt_sha256")),
            }
        component_frame = pd.concat(blocks, ignore_index=True)
        before = len(enriched)
        enriched = enriched.merge(
            component_frame,
            on=["pair_uid", "peptide_design_code", "affibody_design_code"],
            how="left",
            validate="one_to_one",
        )
        _require(len(enriched) == before, f"{component} join changed primary row count")
        _require(not enriched[score_columns].isna().any().any(),
                 f"{component} join left a missing primary score")

    expected_primary = 1.0 / (
        1.0
        + np.exp(-np.clip(
            (
                pd.to_numeric(enriched["mint_layer5_logit"], errors="raise")
                + pd.to_numeric(enriched["rde_logit"], errors="raise")
            ) / 2.0,
            -700,
            700,
        ))
    )
    _require(
        np.allclose(
            expected_primary,
            pd.to_numeric(enriched["model_score"], errors="raise"),
            rtol=1e-10,
            atol=1e-12,
        ),
        "primary displayed score is not sigmoid(mean MINT/RDE logit)",
    )
    return enriched, provenance


def _annotate_primary_candidate_qc(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "high_confidence_weak_negative",
        "affibody_identity_seen_in_strict_training",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
    }
    _require(required.issubset(frame.columns),
             "primary shortlist lacks selection-history fields")
    output = frame.copy()
    for column in (
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "high_confidence_weak_negative",
        "affibody_identity_seen_in_strict_training",
    ):
        output[column] = _as_boolean(output[column], column)
    output["selection_observation_summary"] = np.select(
        [
            ~output["observed_in_any_raw_round"],
            output["observed_in_r009_or_r010"],
        ],
        [
            "not_observed_in_any_R000_R014_file",
            "observed_in_R009_or_R010_but_below_pooled_top2pct",
        ],
        default="observed_in_other_rounds_only",
    )
    output["candidate_is_in_pooled_R009_R010_top2pct"] = False
    output["candidate_is_in_corrected_120_measured_panel"] = False
    output["target_complete_peptide_sequence_seen_in_strict_training"] = False
    output["sequence_readiness_warning"] = (
        "MODEL INPUT ONLY; provider must confirm hidden N-terminal MA, vector, tags, "
        "signal peptide, and linker context before construct ordering"
    )
    codes = output["affibody_design_code"].astype(str)
    sequences = output["provider_displayed_58aa_affibody_sequence"].astype(str)
    output["qc_code_contains_cysteine"] = codes.str.contains("C", regex=False)
    output["qc_full_affibody_contains_cysteine"] = sequences.str.contains("C", regex=False)
    output["qc_full_affibody_contains_N_X_S_or_T_motif"] = sequences.map(
        lambda value: bool(re.search(r"N[^P][ST]", value))
    )
    output["qc_hamming_distance_to_measured_FALTA"] = codes.map(
        lambda value: _hamming_distance(value, "FALTA")
    ).astype(np.int64)
    output["qc_review_flags"] = [
        ";".join([
            name
            for name, present in (
                ("code_contains_cysteine", bool(code_cys)),
                ("full_affibody_contains_cysteine", bool(full_cys)),
                ("full_affibody_contains_N_X_S_or_T_motif", bool(motif)),
                ("high_confidence_weak_negative", bool(weak_negative)),
            )
            if present
        ])
        for code_cys, full_cys, motif, weak_negative in zip(
            output["qc_code_contains_cysteine"],
            output["qc_full_affibody_contains_cysteine"],
            output["qc_full_affibody_contains_N_X_S_or_T_motif"],
            output["high_confidence_weak_negative"],
        )
    ]
    return output


def _primary_output_columns(frame: pd.DataFrame) -> list[str]:
    base = _menu_columns()
    audit = [
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "selection_observation_summary",
        "high_confidence_weak_negative",
        "affibody_identity_seen_in_strict_training",
        "candidate_is_in_pooled_R009_R010_top2pct",
        "candidate_is_in_corrected_120_measured_panel",
        "target_complete_peptide_sequence_seen_in_strict_training",
        "sequence_readiness_warning",
        "qc_code_contains_cysteine",
        "qc_full_affibody_contains_cysteine",
        "qc_full_affibody_contains_N_X_S_or_T_motif",
        "qc_hamming_distance_to_measured_FALTA",
        "qc_review_flags",
    ]
    scores = ["additive_7site_logit", "additive_7site_score",
              "mint_layer5_logit", "mint_layer5_score"]
    for family in ("rde", "stab"):
        scores.extend(sorted(
            column for column in frame.columns
            if re.fullmatch(rf"{family}_probability_seed_\d+", column)
        ))
        scores.extend([
            f"{family}_seed_mean_probability",
            f"{family}_probability",
            f"{family}_logit",
            f"{family}_seed_probability_sd",
        ])
    _require(set(scores).issubset(frame.columns), "primary component score columns missing")
    return base + audit + [column for column in scores if column not in base + audit]


def _validate_affibody_sequence_mapping(frame: pd.DataFrame, label: str) -> None:
    codes = frame["affibody_design_code"].astype(str)
    sequences = frame["chain2_affibody_sequence"].astype(str)
    _require(codes.str.len().eq(5).all(), f"{label} contains a non-five-letter code")
    _require(
        codes.map(lambda value: set(value).issubset(LIBB_CODE_ALPHABET)).all(),
        f"{label} contains an amino acid outside the LibB design alphabet",
    )
    _require(sequences.str.len().eq(58).all(), f"{label} Affibody sequence is not 58 aa")
    for code_index, displayed_position in enumerate(DISPLAYED_AFFIBODY_CODE_POSITIONS):
        observed = sequences.str[displayed_position - 1]
        expected = codes.str[code_index]
        _require(
            observed.eq(expected).all(),
            f"{label} code character {code_index + 1} does not map to displayed "
            f"Affibody position {displayed_position}",
        )


def _validate_shortlist(frame: pd.DataFrame, role: ModelRole) -> pd.DataFrame:
    required = {
        "pair_uid",
        "library",
        "peptide_design_code",
        "peptide_full_sequence",
        "affibody_design_code",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "ensemble_score",
        "wetlab_rank",
    }
    _require(required.issubset(frame.columns), f"{role.key} shortlist lacks required columns")
    leaked = _forbidden_outcome_columns(frame.columns)
    _require(not leaked, f"{role.key} shortlist contains outcome columns: {leaked}")
    if role.key == "mint_rde_primary":
        _require(len(frame) == 120, "primary shortlist is not exactly 120 rows")
    else:
        _require(0 <= len(frame) <= 120, f"{role.key} shortlist exceeds 120 rows")
    _require(frame["library"].astype(str).eq("LibB").all(), f"{role.key} is not LibB")
    _require(not frame["pair_uid"].astype(str).duplicated().any(), f"{role.key} repeats a pair")
    observed_targets = set(frame["peptide_design_code"].astype(str))
    _require(observed_targets.issubset(set(EXPECTED_TARGET_SEQUENCES)),
             f"{role.key} contains an unexpected target")
    if role.key == "mint_rde_primary":
        _require(observed_targets == set(EXPECTED_TARGET_SEQUENCES),
                 "primary shortlist does not contain the exact corrected 12 targets")
    scores = pd.to_numeric(frame["ensemble_score"], errors="raise").to_numpy(float)
    _require(np.isfinite(scores).all(), f"{role.key} contains a non-finite score")
    _require(((scores >= 0.0) & (scores <= 1.0)).all(), f"{role.key} scores leave [0, 1]")
    for peptide, block in frame.groupby("peptide_design_code", sort=True):
        peptide = str(peptide)
        _require(1 <= len(block) <= 10,
                 f"{role.key} has an invalid candidate count for {peptide}")
        _require(
            set(block["peptide_full_sequence"].astype(str))
            == {EXPECTED_TARGET_SEQUENCES[peptide]},
            f"{role.key} peptide sequence mismatch for {peptide}",
        )
        _require(
            block["chain1_smart_hla_linker_peptide_sequence"].astype(str)
            .str.endswith(EXPECTED_TARGET_SEQUENCES[peptide])
            .all(),
            f"{role.key} full assay-side sequence does not end in {peptide}'s 9-mer",
        )
        _require(
            not block["affibody_design_code"].astype(str).duplicated().any(),
            f"{role.key} repeats an Affibody within {peptide}",
        )
        ranks = pd.to_numeric(block["wetlab_rank"], errors="raise").astype(int)
        _require(sorted(ranks.tolist()) == list(range(1, len(block) + 1)),
                 f"{role.key} ranks are not contiguous for {peptide}")
    _validate_affibody_sequence_mapping(frame, role.key)
    return frame.sort_values(
        ["peptide_design_code", "wetlab_rank"], kind="mergesort"
    ).reset_index(drop=True)


def _read_shortlist(directory: Path, role: ModelRole) -> tuple[pd.DataFrame, dict]:
    directory = directory.expanduser().resolve()
    shortlist_path = directory / "wetlab_shortlist.csv"
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path, f"{role.key} manifest")
    _require(
        manifest.get("schema_version") == "libb-ensemble-wetlab-shortlist-v1",
        f"{role.key} has an unexpected shortlist schema",
    )
    lock = manifest.get("ensemble_lock", {})
    _require(lock.get("lock_id") == role.lock_id, f"{role.key} has the wrong lock")
    _require(lock.get("expected_lock_id") == role.lock_id,
             f"{role.key} expected-lock role changed")
    _require(
        lock.get("bundle_schema_version")
        == "libb-native-projection-weak-oof-deployment-locks-v1",
        f"{role.key} is not from the native-projection weak-label lock bundle",
    )
    _require(
        isinstance(lock.get("bundle_manifest_sha256"), str)
        and len(lock["bundle_manifest_sha256"]) == 64,
        f"{role.key} lock-bundle hash is missing",
    )
    access = manifest.get("outcome_access", {})
    _require(
        access.get("retention_table_read") is False
        and access.get("retention_used_for_lock_or_selection") is False
        and access.get("retention_columns_allowed_in_component_scores") is False,
        f"{role.key} does not explicitly declare outcome-free assembly",
    )
    _require(shortlist_path.is_file(), f"{role.key} shortlist is missing")
    _require(
        manifest.get("outputs", {}).get("wetlab_shortlist_sha256")
        == sha256_file(shortlist_path),
        f"{role.key} shortlist differs from its manifest",
    )
    sources = manifest.get("component_score_files", {})
    _require(set(sources) == set(EXPECTED_TARGET_SEQUENCES),
             f"{role.key} component provenance does not cover all targets")
    for peptide, mapping in sources.items():
        _require(set(mapping) == set(role.components),
                 f"{role.key} component roles changed for {peptide}")
        for component, record in mapping.items():
            _require(
                isinstance(record.get("score_sha256"), str)
                and len(record["score_sha256"]) == 64,
                f"{role.key}/{peptide}/{component} score hash is missing",
            )
            _require(
                isinstance(record.get("producer_receipt_sha256"), str)
                and len(record["producer_receipt_sha256"]) == 64,
                f"{role.key}/{peptide}/{component} producer hash is missing",
            )
    header = pd.read_csv(shortlist_path, nrows=0)
    leaked = _forbidden_outcome_columns(header.columns)
    _require(not leaked, f"{role.key} shortlist header contains outcome columns: {leaked}")
    frame = pd.read_csv(shortlist_path, dtype={"pair_uid": str})
    frame = _validate_shortlist(frame, role)
    _require(
        int(manifest.get("outputs", {}).get("wetlab_shortlist_rows", -1)) == len(frame),
        f"{role.key} manifest row count differs from its shortlist",
    )
    expected_missing = sorted(set(EXPECTED_TARGET_SEQUENCES) - set(
        frame["peptide_design_code"].astype(str)
    ))
    _require(
        manifest.get("outputs", {}).get("targets_without_above_cutoff_candidates", [])
        == expected_missing,
        f"{role.key} manifest missing-target list differs from its shortlist",
    )
    return frame, manifest


def _validate_shared_provenance(manifests: dict[str, dict]) -> None:
    bundle_hashes = {
        manifest["ensemble_lock"]["bundle_manifest_sha256"]
        for manifest in manifests.values()
    }
    _require(len(bundle_hashes) == 1, "shortlists do not share one weak-label lock bundle")
    universe_hashes = {
        manifest.get("candidate_universe", {}).get("manifest_sha256")
        for manifest in manifests.values()
    }
    _require(len(universe_hashes) == 1 and None not in universe_hashes,
             "shortlists do not share one candidate universe")
    universe_rows = {
        int(manifest.get("candidate_universe", {}).get("rows_scored", -1))
        for manifest in manifests.values()
    }
    _require(len(universe_rows) == 1 and next(iter(universe_rows)) > 0,
             "shortlists disagree on candidate-universe size")

    # A repeated component name must refer to the same immutable score file in
    # every model that uses it.  Otherwise menu overlap is not a model-only
    # comparison.
    component_records: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for manifest in manifests.values():
        for peptide, mapping in manifest["component_score_files"].items():
            for component, record in mapping.items():
                component_records.setdefault((str(peptide), str(component)), set()).add(
                    (str(record["score_sha256"]), str(record["producer_receipt_sha256"]))
                )
    bad = [key for key, records in component_records.items() if len(records) != 1]
    _require(not bad, f"shared component score provenance differs for: {bad[:5]}")


def _add_common_fields(frame: pd.DataFrame, model_key: str, display_name: str) -> pd.DataFrame:
    output = frame.copy()
    output["candidate_model_key"] = model_key
    output["candidate_model"] = display_name
    output["peptide_9mer_sequence"] = output["peptide_full_sequence"].astype(str)
    output["provider_displayed_58aa_affibody_sequence"] = output[
        "chain2_affibody_sequence"
    ].astype(str)
    output["mint_input_affibody_sequence"] = output["chain2_affibody_sequence"].astype(str)
    output["mint_input_smart_hla_linker_peptide_sequence"] = output[
        "chain1_smart_hla_linker_peptide_sequence"
    ].astype(str)
    output["model_score"] = pd.to_numeric(output["ensemble_score"], errors="raise")
    output["model_rank_within_peptide"] = pd.to_numeric(
        output["wetlab_rank"], errors="raise"
    ).astype(np.int64)
    reuse = output.groupby("affibody_design_code")["peptide_design_code"]
    output["selected_for_n_peptides"] = reuse.transform("nunique").astype(np.int64)
    output["selected_for_peptides"] = reuse.transform(
        lambda values: ";".join(sorted(set(map(str, values))))
    )
    return output


def _menu_columns() -> list[str]:
    return [
        "candidate_model_key",
        "candidate_model",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "model_rank_within_peptide",
        "affibody_design_code",
        "model_score",
        "provider_displayed_58aa_affibody_sequence",
        "mint_input_affibody_sequence",
        "mint_input_smart_hla_linker_peptide_sequence",
        "pair_uid",
        "selected_for_n_peptides",
        "selected_for_peptides",
    ]


def _pair_set(frame: pd.DataFrame, targets: Iterable[str] | None = None) -> set[tuple[str, str]]:
    source = frame
    if targets is not None:
        source = source.loc[source["peptide_design_code"].isin(set(targets))]
    return set(zip(
        source["peptide_design_code"].astype(str),
        source["affibody_design_code"].astype(str),
    ))


def _universal_top1(frame: pd.DataFrame) -> str:
    top = frame.loc[frame["model_rank_within_peptide"].eq(1), "affibody_design_code"]
    if top.shape[0] != len(EXPECTED_TARGET_SEQUENCES):
        return ""
    values = sorted(set(top.astype(str)))
    return values[0] if len(values) == 1 else ""


def _build_model_summaries(menus: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = menus["mint_rde_primary"]
    primary_pairs = _pair_set(primary)
    primary_ready_pairs = _pair_set(primary, READY_TARGETS)
    primary_top = dict(zip(
        primary.loc[primary["model_rank_within_peptide"].eq(1), "peptide_design_code"],
        primary.loc[primary["model_rank_within_peptide"].eq(1), "affibody_design_code"],
    ))
    rows = []
    per_target_rows = []
    for role in MODEL_ROLES:
        frame = menus[role.key]
        pairs = _pair_set(frame)
        ready_pairs = _pair_set(frame, READY_TARGETS)
        overlap = len(pairs & primary_pairs)
        ready_overlap = len(ready_pairs & primary_ready_pairs)
        top = frame.loc[frame["model_rank_within_peptide"].eq(1)]
        top_map = dict(zip(top["peptide_design_code"], top["affibody_design_code"]))
        reuse = frame.groupby("affibody_design_code")["peptide_design_code"].nunique()
        targets_with_candidates = set(frame["peptide_design_code"].astype(str))
        missing_targets = sorted(set(EXPECTED_TARGET_SEQUENCES) - targets_with_candidates)
        rows.append({
            "model_key": role.key,
            "model": role.display_name,
            "role": "primary" if role.key == "mint_rde_primary" else "comparator",
            "candidate_pairs": len(frame),
            "targets_with_candidates": len(targets_with_candidates),
            "missing_targets": ";".join(missing_targets),
            "distinct_affibody_designs": int(
                frame["affibody_design_code"].nunique()
            ),
            "largest_number_of_targets_sharing_one_affibody": (
                int(reuse.max()) if len(reuse) else 0
            ),
            "one_affibody_is_ranked_first_for_all_12_targets": bool(_universal_top1(frame)),
            "universal_top_ranked_affibody": _universal_top1(frame),
            "exact_pair_overlap_with_primary": overlap,
            "exact_pair_overlap_fraction_of_this_menu": (
                overlap / float(len(frame)) if len(frame) else np.nan
            ),
            "pair_set_jaccard_with_primary": overlap / float(len(pairs | primary_pairs)),
            "same_top_ranked_affibody_as_primary": sum(
                top_map.get(peptide) == primary_top.get(peptide)
                for peptide in targets_with_candidates
            ),
            "top_rank_comparable_targets": len(targets_with_candidates),
            "ready_batch_pair_overlap_with_primary_out_of_100": ready_overlap,
        })
        for peptide in EXPECTED_TARGET_SEQUENCES:
            role_block = frame.loc[frame["peptide_design_code"].eq(peptide)]
            primary_block = primary.loc[primary["peptide_design_code"].eq(peptide)]
            role_codes = set(role_block["affibody_design_code"].astype(str))
            primary_codes = set(primary_block["affibody_design_code"].astype(str))
            has_candidate = len(role_block) > 0
            per_target_rows.append({
                "model_key": role.key,
                "model": role.display_name,
                "peptide_design_code": peptide,
                "model_candidates_for_peptide": len(role_block),
                "candidate_overlap_with_primary_out_of_10": len(role_codes & primary_codes),
                "same_top_ranked_affibody_as_primary": (
                    has_candidate
                    and str(role_block.loc[
                        role_block["model_rank_within_peptide"].eq(1),
                        "affibody_design_code",
                    ].iloc[0])
                    == str(primary_block.loc[
                        primary_block["model_rank_within_peptide"].eq(1),
                        "affibody_design_code",
                    ].iloc[0])
                ),
            })
    return pd.DataFrame(rows), pd.DataFrame(per_target_rows)


def _build_prospective_cutoff_summary(
    menus: dict[str, pd.DataFrame], manifests: dict[str, dict],
) -> pd.DataFrame:
    """Summarize the seven weak-label cutoffs that produced the new menus."""

    rows: list[dict[str, Any]] = []
    for order, role in enumerate(MODEL_ROLES, start=1):
        lock = manifests[role.key].get("ensemble_lock", {})
        _require(lock.get("lock_id") == role.lock_id,
                 f"prospective lock ID changed for {role.key}")
        selection = lock.get("selection", {})
        cutoff = float(selection.get("score_threshold", np.nan))
        _require(np.isfinite(cutoff) and 0.0 <= cutoff <= 1.0,
                 f"prospective weak-label cutoff is invalid for {role.key}")
        _require(selection.get("pad_below_threshold") is False,
                 f"prospective menu pads below cutoff for {role.key}")
        provenance = str(lock.get("threshold_selection_provenance", ""))
        _require("10,181 weak-label OOF rows" in provenance and "No retention used" in provenance,
                 f"prospective threshold provenance changed for {role.key}")
        frame = menus[role.key]
        _require(bool(pd.to_numeric(frame["model_score"], errors="raise").ge(cutoff).all()),
                 f"prospective menu contains a below-cutoff row for {role.key}")
        represented = set(frame["peptide_design_code"].astype(str))
        rows.append({
            "model_order": order,
            "model_key": role.key,
            "model": role.display_name,
            "weak_label_score_cutoff": cutoff,
            "decision_rule": "model_score >= weak_label_score_cutoff",
            "maximum_candidates_per_peptide": int(selection.get("max_candidates_per_peptide", -1)),
            "candidate_pairs_in_new_menu": len(frame),
            "targets_represented": len(represented),
            "missing_targets": ";".join(
                sorted(set(EXPECTED_TARGET_SEQUENCES) - represented)
            ),
            "cutoff_selection_data": "10,181 weak-label out-of-fold rows",
            "cutoff_used_retention": False,
            "below_cutoff_rows_padded": False,
        })
    result = pd.DataFrame(rows)
    _require(result["maximum_candidates_per_peptide"].eq(10).all(),
             "prospective menu cap changed")
    return result


def _build_overlap_matrix(menus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sets = {key: _pair_set(frame) for key, frame in menus.items()}
    rows = []
    for left in MODEL_ROLES:
        row = {"model": left.display_name}
        for right in MODEL_ROLES:
            row[right.key] = len(sets[left.key] & sets[right.key])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_construct_roster(menus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    blocks = []
    for role in MODEL_ROLES:
        frame = menus[role.key]
        roster = (
            frame.groupby(
                ["affibody_design_code", "provider_displayed_58aa_affibody_sequence"],
                as_index=False,
            )
            .agg(
                selected_for_n_peptides=("peptide_design_code", "nunique"),
                selected_for_peptides=(
                    "peptide_design_code",
                    lambda values: ";".join(sorted(set(map(str, values)))),
                ),
                best_rank_within_any_peptide=("model_rank_within_peptide", "min"),
            )
        )
        roster.insert(0, "model", role.display_name)
        roster.insert(0, "model_key", role.key)
        blocks.append(roster)
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["model_key", "selected_for_n_peptides", "affibody_design_code"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def _build_codes_by_target(menus: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for role in MODEL_ROLES:
        for peptide in EXPECTED_TARGET_SEQUENCES:
            block = menus[role.key].loc[
                menus[role.key]["peptide_design_code"].eq(peptide)
            ]
            block = block.sort_values("model_rank_within_peptide", kind="mergesort")
            rows.append({
                "model_key": role.key,
                "model": role.display_name,
                "peptide_design_code": peptide,
                "candidate_count": len(block),
                "ranked_affibody_codes_1_to_10": ";".join(
                    block["affibody_design_code"].astype(str)
                ),
            })
    return pd.DataFrame(rows)


def _load_parity_audit(
    directory: Path,
    production_score_dirs: dict[str, Path],
) -> dict:
    resolved = directory.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    manifest = _load_json(manifest_path, "native-projection release-gate manifest")
    _require(
        manifest.get("schema_version")
        == "libb-native-projection-release-gates-v1",
        "unexpected native-projection release-gate schema",
    )
    _require(manifest.get("status") == "pass", "native-projection release gate did not pass")
    access = manifest.get("outcome_access", {})
    _require(access.get("retention_values_loaded") is False,
             "native-projection release gate loaded retention values")
    _require(access.get("binder_values_loaded") is False,
             "native-projection release gate loaded binder values")
    _require(access.get("selection_labels_loaded_for_release_decision") is False,
             "native-projection release decision loaded weak-label outcomes")
    _require(access.get("weak_metadata_loaded_for_release_decision") is False,
             "native-projection release decision loaded weak-label metadata")

    thresholds = manifest.get("thresholds", {})
    production_tolerance = float(thresholds.get("production_probability_tolerance", np.nan))
    historical_seed_tolerance = float(
        thresholds.get("historical_seed_probability_tolerance", np.nan)
    )
    historical_aggregate_tolerance = float(
        thresholds.get("historical_aggregate_probability_tolerance", np.nan)
    )
    cross_batch_tolerance = float(
        thresholds.get("cross_batch_probability_tolerance", np.nan)
    )
    _require(np.isfinite(production_tolerance) and 0 < production_tolerance <= 1e-6,
             "production replay tolerance is missing or looser than 1e-6")
    _require(np.isfinite(historical_seed_tolerance)
             and 0 < historical_seed_tolerance <= 2e-6,
             "historical seed-compatibility tolerance is invalid")
    _require(np.isfinite(historical_aggregate_tolerance)
             and 0 < historical_aggregate_tolerance <= 1e-6,
             "historical aggregate-compatibility tolerance is invalid")
    _require(np.isfinite(cross_batch_tolerance) and 0 < cross_batch_tolerance <= 1e-4,
             "cross-batch compatibility tolerance is invalid")

    outputs = manifest.get("outputs", {})
    expected_outputs = {
        "production_batch_replay.csv": 16,
        "historical_tail_compatibility.csv": 12,
        "corrected120_cross_batch_seed.csv": 10,
        "corrected120_cross_batch_aggregate.csv": 2,
        "primary_top10_boundary_by_peptide.csv": 12,
    }
    for name, expected_rows in expected_outputs.items():
        record = outputs.get(name, {})
        path = resolved / str(record.get("path", ""))
        _require(path.is_file() and record.get("sha256") == sha256_file(path),
                 f"release-gate output differs from manifest: {name}")
        frame = pd.read_csv(path)
        _require(len(frame) == expected_rows and int(record.get("rows", -1)) == expected_rows,
                 f"release-gate output row count changed: {name}")
    report_record = outputs.get("report.md", {})
    report_path = resolved / str(report_record.get("path", ""))
    _require(report_path.is_file() and report_record.get("sha256") == sha256_file(report_path),
             "native-projection release-gate report differs from manifest")

    gates = manifest.get("gates", {})
    _require(set(gates) == {
        "production_batch_replay", "historical_tail_compatibility",
        "corrected120_cross_batch_compatibility", "primary_top10_boundary",
    }, "native-projection release-gate set changed")
    _require(all(gates[name].get("status") == "pass" for name in gates),
             "one or more native-projection release gates failed")
    production = gates["production_batch_replay"]
    historical = gates["historical_tail_compatibility"]
    cross_batch = gates["corrected120_cross_batch_compatibility"]
    boundary = gates["primary_top10_boundary"]
    production_maxima = []
    historical_seed_maxima = []
    historical_aggregate_maxima = []
    cross_batch_maxima = []
    for family in ("rde", "stab"):
        item = production.get("families", {}).get(family, {})
        maximum = float(item.get("maximum_absolute_probability_difference", np.nan))
        _require(np.isfinite(maximum) and 0 <= maximum <= production_tolerance,
                 f"{family} production replay exceeds its strict tolerance")
        _require(item.get("rows") == 640
                 and item.get("verified_batch_size") == {"rde": 128, "stab": 20}[family],
                 f"{family} production replay size/batch contract changed")
        _require(item.get("all_columns_within_1e-6") is True
                 and item.get("all_rankings_identical") is True,
                 f"{family} production replay did not preserve scores/ranks")
        production_maxima.append(maximum)
        for record_name in (
            "invocation_manifest", "invocation_wrapper", "replay_receipt",
            "replay_chunk", "merged_partition",
        ):
            record = item.get(record_name, {})
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            _require(path.is_file() and record.get("sha256") == sha256_file(path),
                     f"{family} production replay asset changed: {record_name}")

        item = historical.get("families", {}).get(family, {})
        seed_maximum = float(item.get("maximum_seed_probability_difference", np.nan))
        aggregate_maximum = float(
            item.get("maximum_aggregate_probability_difference", np.nan)
        )
        _require(np.isfinite(seed_maximum) and 0 <= seed_maximum <= historical_seed_tolerance,
                 f"{family} historical seed compatibility exceeds tolerance")
        _require(np.isfinite(aggregate_maximum)
                 and 0 <= aggregate_maximum <= historical_aggregate_tolerance,
                 f"{family} historical aggregate compatibility exceeds tolerance")
        _require(item.get("rows") == 120
                 and item.get("all_seed_within_peptide_rankings_identical") is True
                 and item.get("aggregate_within_peptide_and_global_rankings_identical") is True,
                 f"{family} historical compatibility ranking contract failed")
        historical_seed_maxima.append(seed_maximum)
        historical_aggregate_maxima.append(aggregate_maximum)
        for record_name in ("replay_manifest", "replay_predictions"):
            record = item.get(record_name, {})
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            _require(path.is_file() and record.get("sha256") == sha256_file(path),
                     f"{family} historical replay asset changed: {record_name}")

        item = cross_batch.get("families", {}).get(family, {})
        seed_maximum = float(item.get("maximum_seed_probability_difference", np.nan))
        aggregate_maximum = float(item.get("maximum_aggregate_probability_difference", np.nan))
        _require(np.isfinite(seed_maximum) and 0 <= seed_maximum <= cross_batch_tolerance,
                 f"{family} cross-batch seed compatibility exceeds tolerance")
        _require(np.isfinite(aggregate_maximum) and 0 <= aggregate_maximum <= cross_batch_tolerance,
                 f"{family} cross-batch aggregate compatibility exceeds tolerance")
        _require(item.get("all_seed_and_aggregate_within_peptide_rankings_identical") is True
                 and item.get("aggregate_global_ranking_identical") is True
                 and item.get("cutoff_decision_flips") == 0,
                 f"{family} cross-batch ranking/cutoff contract failed")
        cross_batch_maxima.extend([seed_maximum, aggregate_maximum])

    boundary_numeric = [
        float(boundary.get("minimum_batch16_rank10_vs_rank11_logit_gap", np.nan)),
        float(boundary.get("maximum_rde_logit_difference_top50", np.nan)),
        float(boundary.get("maximum_weighted_ensemble_logit_difference_top50", np.nan)),
        float(boundary.get("minimum_selected_probability_margin_above_cutoff", np.nan)),
    ]
    _require(np.isfinite(boundary_numeric).all(), "primary boundary gate contains non-finite values")
    _require(boundary.get("peptides") == 12 and boundary.get("candidate_rows_rescored") == 600
             and boundary.get("verified_alternative_batch_size") == 16,
             "primary boundary coverage/batch contract changed")
    _require(boundary.get("all_top10_membership_identical") is True
             and boundary.get("all_top10_order_identical") is True,
             "primary top-ten decision changed under batch repacking")
    _require(boundary_numeric[0] > 0 and boundary_numeric[3] > 0,
             "primary top-ten boundary/cutoff margin is not positive")
    for record_name in (
        "selection_manifest", "production_review_pool", "boundary_input_manifest",
        "boundary_replay_invocation", "boundary_wrapper", "boundary_worker_receipt",
    ):
        record = boundary.get(record_name, {})
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        _require(path.is_file() and record.get("sha256") == sha256_file(path),
                 f"primary boundary release asset changed: {record_name}")

    producer_start = manifest.get("producer_start", {})
    producer_end = manifest.get("producer_end", {})
    imported_start = manifest.get("imported_panel_auditor_start", {})
    imported_end = manifest.get("imported_panel_auditor_end", {})
    _require(producer_start == producer_end and imported_start == imported_end,
             "release-gate source changed during execution")
    for label, record in (("release auditor", producer_start),
                          ("imported panel auditor", imported_start)):
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        _require(path.is_file() and record.get("sha256") == sha256_file(path),
                 f"{label} changed after release-gate execution")

    production_merged = {}
    for family in ("rde", "stab"):
        score_dir = production_score_dirs[family].expanduser().resolve()
        score_manifest_path = score_dir / "manifest.json"
        score_manifest = _load_json(
            score_manifest_path, f"production {family} merged-score manifest"
        )
        _require(
            score_manifest.get("schema_version")
            == "libb-fixed-structure-candidate-scores-merged-v1",
            f"unexpected production {family} merged-score schema",
        )
        _require(score_manifest.get("family") == family,
                 f"production merged-score manifest has wrong {family} family")
        _require(int(score_manifest.get("candidate_universe", {}).get("candidate_rows", -1))
                 == 4_447_848,
                 f"production {family} merged scores do not cover the full universe")
        receipt = manifest.get("exhaustive_manifests", {}).get(family, {})
        _require(receipt.get("sha256") == sha256_file(score_manifest_path),
                 f"release gate is not bound to production {family} merged scores")
        production_merged[family] = {
            "directory": str(score_dir),
            "manifest_path": str(score_manifest_path),
            "manifest_sha256": sha256_file(score_manifest_path),
            "release_gate_cross_bound": True,
        }
    return {
        "directory": str(resolved),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status": "pass",
        "families": ["rde", "stab"],
        "production_probability_tolerance": production_tolerance,
        "production_maximum_absolute_difference": max(production_maxima),
        "historical_seed_probability_tolerance": historical_seed_tolerance,
        "historical_maximum_seed_difference": max(historical_seed_maxima),
        "historical_aggregate_probability_tolerance": historical_aggregate_tolerance,
        "historical_maximum_aggregate_difference": max(historical_aggregate_maxima),
        "cross_batch_probability_tolerance": cross_batch_tolerance,
        "cross_batch_maximum_difference": max(cross_batch_maxima),
        "primary_top10_membership_and_order_stable": True,
        "primary_minimum_rank10_vs_rank11_logit_gap": boundary_numeric[0],
        "primary_minimum_selected_probability_margin_above_cutoff": boundary_numeric[3],
        "production_merged_score_artifacts": production_merged,
    }


def _write_workbook(
    path: Path,
    review_batch: pd.DataFrame,
    result_entry: pd.DataFrame,
    primary: pd.DataFrame,
    model_summary: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    prospective_cutoffs: pd.DataFrame,
    retrospective_thresholds: pd.DataFrame,
    retrospective_recommendations: pd.DataFrame,
    retrospective_codes_by_target: pd.DataFrame,
    codes_by_target: pd.DataFrame,
    all_menus: pd.DataFrame,
    mint_control_ready: pd.DataFrame,
    comparator_ready_menus: pd.DataFrame,
) -> None:
    readme = pd.DataFrame([
        ("Primary rule", "MINT + RDE ensemble; ten ranked Affibodies per peptide."),
        ("PRIMARY_ALL12", "120 specified candidate pairs: 12 peptides x 10 assigned Affibodies."),
        ("REVIEW_100_PAIRS", "Proposed scientific-review subset: 100 candidate pairs from the fixed ten-target subset of PRIMARY_ALL12."),
        ("Important", "The batch is not every cross-combination of its distinct Affibody sequences and peptides."),
        ("Model score", "Ranks similarity to a selection-derived positive; it is not retention percent or a calibrated binding probability."),
        ("WEAK_CUTOFFS", "Seven prospective-menu cutoffs fitted only on weak-label out-of-fold rows."),
        ("RETRO_F1_THRESHOLDS", "Display-only cutoffs optimized on the same already-measured 120 retention pairs; never used for the new menus."),
        ("RETRO_RECOMMENDED", "Every already-measured pair above its model-specific retrospective cutoff, including the known outcome."),
        ("RESULT_ENTRY", "Outcome columns are deliberately blank at release and are filled only after new wet-lab measurements return."),
        ("Sequence warning", "Displayed/model-input sequences require provider confirmation of cloning/vector context before ordering constructs."),
    ], columns=["Item", "Meaning"])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        review_batch.to_excel(writer, sheet_name="REVIEW_100_PAIRS", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)
        result_entry.to_excel(writer, sheet_name="RESULT_ENTRY", index=False)
        primary.to_excel(writer, sheet_name="PRIMARY_ALL12", index=False)
        mint_control_ready.to_excel(writer, sheet_name="MINT_CONTROL_REVIEW", index=False)
        comparator_ready_menus.to_excel(writer, sheet_name="COMPARATOR_REVIEW", index=False)
        model_summary.to_excel(writer, sheet_name="MODEL_SUMMARY", index=False)
        overall_metrics.to_excel(writer, sheet_name="MODEL_METRICS", index=False)
        prospective_cutoffs.to_excel(writer, sheet_name="WEAK_CUTOFFS", index=False)
        retrospective_thresholds.to_excel(writer, sheet_name="RETRO_F1_THRESHOLDS", index=False)
        retrospective_recommendations.to_excel(writer, sheet_name="RETRO_RECOMMENDED", index=False)
        retrospective_codes_by_target.to_excel(writer, sheet_name="RETRO_CODES_BY_TARGET", index=False)
        codes_by_target.to_excel(writer, sheet_name="CODES_BY_TARGET", index=False)
        all_menus.to_excel(writer, sheet_name="ALL_MODEL_MENUS", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:100]]
                width = min(60, max(10, max(map(len, values), default=10) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    os.chmod(path, 0o600)


def _report_text(
    menus: dict[str, pd.DataFrame],
    manifests: dict[str, dict],
    candidate_manifest: dict,
    model_summary: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    prospective_cutoffs: pd.DataFrame,
    retrospective_thresholds: pd.DataFrame,
    retrospective_codes_by_target: pd.DataFrame,
    cutoff_audit: pd.DataFrame,
    codes_by_target: pd.DataFrame,
    parity_audit: dict,
) -> str:
    primary = menus["mint_rde_primary"]
    primary_summary = model_summary.loc[
        model_summary["model_key"].eq("mint_rde_primary")
    ].iloc[0]
    universe_rows = int(candidate_manifest["outputs"]["candidate_rows"])
    raw_rows = len(EXPECTED_TARGET_SEQUENCES) * len(LIBB_CODE_ALPHABET) ** 5
    first_batch = primary.loc[primary["peptide_design_code"].isin(READY_TARGETS)]
    first_unique = int(first_batch["affibody_design_code"].nunique())
    primary_unique = int(primary["affibody_design_code"].nunique())
    cutoff = float(
        manifests["mint_rde_primary"]["ensemble_lock"]["selection"]["score_threshold"]
    )
    cutoff_total = int(cutoff_audit["rows_above_locked_cutoff"].sum())
    cutoff_min = int(cutoff_audit["rows_above_locked_cutoff"].min())
    cutoff_max = int(cutoff_audit["rows_above_locked_cutoff"].max())

    summary_lines = []
    for row in model_summary.itertuples(index=False):
        summary_lines.append(
            f"| {row.model} | {row.candidate_pairs} | {row.targets_with_candidates}/12 | "
            f"{row.distinct_affibody_designs} | "
            f"{row.exact_pair_overlap_with_primary}/{row.candidate_pairs} | "
            f"{row.same_top_ranked_affibody_as_primary}/{row.top_rank_comparable_targets} | "
            f"{row.largest_number_of_targets_sharing_one_affibody} |"
        )
    incomplete_rows = model_summary.loc[
        model_summary["missing_targets"].astype(str).ne("")
    ]
    if len(incomplete_rows):
        incomplete_descriptions = "; ".join(
            f"{row.model}: {str(row.missing_targets).replace(';', ', ')}"
            for row in incomplete_rows.itertuples(index=False)
        )
        incomplete_menu_note = (
            "Incomplete comparator menus and their missing targets are: "
            f"{incomplete_descriptions}. This means the model's one global weak-label "
            "cutoff leaves no eligible row for those targets. It does not mean the model "
            "cannot rank their Affibodies; below-cutoff rows are deliberately not padded "
            "into a cutoff-respecting shortlist."
        )
    else:
        incomplete_menu_note = (
            "Every comparator has at least ten above-cutoff candidates for every target, "
            "so all comparator menus contain 120 rows."
        )
    metric_lines = []
    for row in overall_metrics.itertuples(index=False):
        metric_lines.append(
            f"| {row.model} | {row.weak_label_within_peptide_average_precision:.4f} | "
            f"{row.corrected_120_retention_within_peptide_average_precision:.4f} | "
            f"{row.corrected_120_retention_within_peptide_auroc:.4f} | "
            f"{row.corrected_120_retention_within_peptide_spearman:.4f} |"
        )

    prospective_cutoff_lines = []
    for row in prospective_cutoffs.itertuples(index=False):
        missing = str(row.missing_targets).replace(";", ", ")
        prospective_cutoff_lines.append(
            f"| {row.model} | {row.weak_label_score_cutoff:.7g} | "
            f"{row.candidate_pairs_in_new_menu} | {row.targets_represented}/12 | "
            f"{missing if missing else 'None'} |"
        )

    retrospective_threshold_lines = []
    for row in retrospective_thresholds.itertuples(index=False):
        retrospective_threshold_lines.append(
            f"| {row.model} | {row.retrospective_threshold:.7g} | "
            f"{row.recommended_pairs}/120 | {row.true_positive_binders}/{row.recommended_pairs} | "
            f"{row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} |"
        )

    retrospective_appendix_lines: list[str] = []
    for model_key, model_block in retrospective_codes_by_target.groupby(
        "model_key", sort=False
    ):
        model_name = str(model_block["model"].iloc[0])
        retrospective_appendix_lines.extend([f"### {model_name}", ""])
        lookup = model_block.set_index("peptide_design_code", drop=False)
        for peptide in EXPECTED_TARGET_SEQUENCES:
            row = lookup.loc[peptide]
            codes = str(row["recommended_affibody_codes_in_score_order"])
            if not codes:
                rendered_codes = "None"
            else:
                rendered_codes = ", ".join(
                    f"`{code}`" for code in codes.split(";") if code
                )
            retrospective_appendix_lines.append(f"- `{peptide}`: {rendered_codes}")
        retrospective_appendix_lines.append("")

    primary_codes = codes_by_target.loc[
        codes_by_target["model_key"].eq("mint_rde_primary")
    ]
    code_lines = [
        f"| {row.peptide_design_code} | {row.ranked_affibody_codes_1_to_10.replace(';', ', ')} |"
        for row in primary_codes.itertuples(index=False)
    ]

    universal = str(primary_summary["universal_top_ranked_affibody"])
    if universal:
        reuse_paragraph = (
            f"The primary ensemble ranks `{universal}` first for all 12 peptides. "
            "That is an explicit, testable broadly useful-Affibody hypothesis; it is not "
            "evidence already obtained from the new experiment."
        )
    else:
        reuse_paragraph = (
            "The primary ensemble does not give every peptide the same first-ranked "
            "Affibody. Repeated designs are still shown in the sequence roster so the "
            "laboratory can distinguish reuse from peptide-specific choices."
        )

    def qc_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "unobserved": int((~frame["observed_in_any_raw_round"]).sum()),
            "r9r10": int(frame["observed_in_r009_or_r010"].sum()),
            "aff_seen_pairs": int(frame["affibody_identity_seen_in_strict_training"].sum()),
            "aff_seen_designs": int(frame.loc[
                frame["affibody_identity_seen_in_strict_training"],
                "affibody_design_code",
            ].nunique()),
            "weak_negative": int(frame["high_confidence_weak_negative"].sum()),
            "code_cys": int(frame["qc_code_contains_cysteine"].sum()),
            "full_cys": int(frame["qc_full_affibody_contains_cysteine"].sum()),
            "motif": int(frame["qc_full_affibody_contains_N_X_S_or_T_motif"].sum()),
        }

    all_qc = qc_counts(primary)
    ready_qc = qc_counts(first_batch)
    falta_distance_one_pairs = int(
        primary["qc_hamming_distance_to_measured_FALTA"].eq(1).sum()
    )
    falta_distance_one_designs = int(primary.loc[
        primary["qc_hamming_distance_to_measured_FALTA"].eq(1),
        "affibody_design_code",
    ].nunique())
    falta_distance_le_two_pairs = int(
        primary["qc_hamming_distance_to_measured_FALTA"].le(2).sum()
    )
    falta_distance_le_two_designs = int(primary.loc[
        primary["qc_hamming_distance_to_measured_FALTA"].le(2),
        "affibody_design_code",
    ].nunique())
    ready_falta_distance_one_pairs = int(
        first_batch["qc_hamming_distance_to_measured_FALTA"].eq(1).sum()
    )
    ready_falta_distance_le_two_pairs = int(
        first_batch["qc_hamming_distance_to_measured_FALTA"].le(2).sum()
    )
    minimum_falta_distance = int(primary["qc_hamming_distance_to_measured_FALTA"].min())
    reuse_counts = (
        primary.groupby("affibody_design_code")["peptide_design_code"]
        .nunique()
        .sort_values(ascending=False, kind="mergesort")
    )
    broadly_reused = ", ".join(
        f"`{code}` ({int(count)} targets)"
        for code, count in reuse_counts.head(5).items()
    )

    return "\n".join([
        "# Selecting LibB Affibody candidates for prospective wet-lab testing",
        "",
        "Affibodies are small engineered proteins designed here to bind peptide--HLA "
        "targets. This package turns the locked LibB models into a concrete list of new "
        "peptide--Affibody pairs for laboratory testing. The primary rule is the "
        "MINT+RDE ensemble: MINT supplies a protein-pair sequence representation, RDE "
        "supplies a structure-derived representation, and the two logits are averaged "
        "before Affibodies are ranked separately for each peptide.",
        "",
        "The complete primary menu contains 120 specified candidate pairs: 12 peptide "
        "targets, each assigned its ten highest-ranked Affibodies. The scientific-review "
        f"subset proposed as a first batch contains 100 unchanged candidate pairs for ten "
        "prespecified peptide targets and "
        f"uses {first_unique} distinct Affibody designs. It is a subset of the 120-pair "
        "menu, not a different model, not the globally highest 100 scores, and not every "
        "cross-combination between the listed peptides and Affibodies.",
        "",
        "No raw retention table or pair-level retention value was opened to assemble "
        "these menus. All seven input shortlists were already immutable, hashed artifacts "
        "before corrected-panel outcomes were loaded for the explicitly retrospective "
        "ranking and threshold sections. Those outcomes are never passed to prospective "
        "ranking code and cannot add, remove, or reorder a new candidate.",
        "",
        "Before release, the production scorer was replayed at the actual exhaustive batch "
        "sizes on 640 candidates per structure family. Its saved seed and aggregate scores "
        f"matched the merged exhaustive archive within {parity_audit['production_maximum_absolute_difference']:.3g} "
        "probability units, below the strict 1e-6 release tolerance. Reconstructing the "
        "historical corrected-panel batch context preserved every within-peptide ranking; "
        f"the largest seed-level numerical difference was {parity_audit['historical_maximum_seed_difference']:.3g}. "
        "A separate batch-16 stress test of the primary model's top 50 candidates preserved "
        "the exact top-ten set and order for all 12 peptides. These checks establish scorer, "
        "mapping, checkpoint, and candidate-decision consistency; they do not imply that "
        "floating-point scores are invariant to GPU batch shape. The release audit loaded "
        "identity and model-score columns, but no retention or binder values.",
        "",
        "## Data used to train the project-specific predictors",
        "",
        "The project-specific readouts use the same 30,648 LibB training pairs: 23,725 "
        "selection-derived positives and 6,923 selection-derived negatives. A positive "
        "is in the tie-inclusive highest 2% after R009 and R010 counts are pooled by "
        "exact pair and summed. A negative has at least three R001 reads and no observed "
        "read in R002--R014. These labels are evidence from selection sequencing, not "
        "direct retention measurements.",
        "",
        "All complete peptide sequences in the 12-target retention panel were removed "
        "from weak-label training, as were the ten complete Affibody sequences in that "
        "panel. The new candidate menu therefore tests new exact pairs for the same 12 "
        "known peptide targets. It is not automatically a test in which both partners "
        "are new: some proposed Affibody identities occurred in strict weak-label "
        "training, and their exact counts are reported below.",
        "",
        "## The candidate space that was scored",
        "",
        f"LibB permits 13 amino acids at each of five Affibody positions. That gives "
        f"`13^5 = 371,293` Affibody codes per peptide and {raw_rows:,} pair assignments "
        "for the 12 current targets. The candidate universe excludes exactly:",
        "",
        "- 120 pairs already present in the corrected measured panel;",
        "- 7,599 current-target pairs in the pooled R009+R010 top-2% positive set;",
        "- with the 51 pairs shared by those sets counted only once.",
        "",
        f"Thus, `4,455,516 - (120 + 7,599 - 51) = {universe_rows:,}` new pair "
        "assignments were scored. Equivalently, the pooled-positive set contributes "
        "7,548 exclusions beyond the 120 measured pairs. This is exhaustive for the "
        "current 13-amino-acid LibB design space and these 12 targets; it is not an "
        "all-20-amino-acid search and does not introduce new peptide targets.",
        "",
        "Every candidate record contains the complete SMART--HLA--linker--peptide "
        "sequence and the complete provider-displayed Affibody sequence; these are the "
        "two MINT inputs. RDE and StaB do not receive the unresolved SMART/linker "
        "sequence. RDE uses the fixed resolved crystal patch with the current seven "
        "designed amino-acid identities. StaB uses the fixed resolved complex and "
        "separated structural fragments. The five-character Affibody code records "
        "the amino acids at crystal-aligned positions 8, 12, 15, 16, and 19, which are "
        "displayed-sequence positions 6, 10, 13, 14, and 17. The hidden preceding `MA` "
        "changes the structure-aligned labels by two; it is not silently inserted into "
        "the displayed model-input sequence.",
        "",
        "## The models and why MINT+RDE is primary",
        "",
        "| Candidate score | Information used |",
        "|---|---|",
        "| Seven-position additive model | The two varied peptide amino acids and five varied Affibody amino acids; learned effects are added. |",
        "| Frozen MINT layer 5 | Complete reconstructed peptide-side and Affibody sequences; a project-specific classifier reads frozen MINT features. |",
        "| StaB native-projection readout | Frozen ProteinMPNN/StaB-derived features at the designed structural sites, followed by a trainable project classifier. |",
        "| RDE native-projection readout | Frozen mutation-trained RDE features around the fixed LibB crystal neighborhood, followed by the directly learned 896-to-64-to-32-to-1 project readout. |",
        "| Equal-logit ensembles | Arithmetic mean of selected component logits, followed by a sigmoid. |",
        "| Weak-label-fitted stacker | A nonnegative logistic combination fitted only to held-out weak-label predictions. |",
        "",
        "The upstream MINT, RDE, and StaB encoders are frozen: their pretrained weights "
        "do not change. The downstream project readouts are the small classifiers trained "
        "on this project's 30,648 weak-label pairs. Their readouts and the combination "
        "weights were fitted or selected using selection-derived labels. MINT alone was "
        "the overall weak-label AP winner. MINT+RDE was only the best rule containing at "
        "least two models; testing it as the primary ensemble is an experimental-design "
        "choice, not a weak-metric win over MINT. MINT therefore remains the prespecified "
        "single-model control.",
        "",
        "The corrected retention panel had already been examined while the model families "
        "and direct readouts were being developed. Retention was not used in the model "
        "formulas, weights, prospective weak-label cutoffs, or new-candidate ranks. The "
        "separate thresholds fitted to retention below are display-only retrospective "
        "summaries. The model-family comparison remains retrospective rather than an "
        "untouched prospective comparison; only the new candidate outcomes are prospective.",
        "",
        "The primary calculation takes the arithmetic mean of the MINT and RDE logits, "
        "then applies a sigmoid. Equal coefficients do not imply equal biological "
        "contribution, and the two raw logit distributions may have different spreads. "
        "This combination was chosen empirically on weak-label out-of-fold predictions. "
        "Its displayed 0--1 score indicates similarity to the selection-derived positive "
        "class; it is neither a predicted retention percentage nor a calibrated binding "
        "probability.",
        "",
        "RDE reuses the same validated LibB reference-crystal geometry for every "
        "candidate, changes the seven designed amino-acid identities, and applies the "
        "latest directly learned 896-to-64-to-32-to-1 readout. The upstream RDE encoder "
        "remains frozen. It does not fold 4.45 million pairs separately.",
        "",
        "## Matched model comparison",
        "",
        "The corrected panel contains 120 measured pairs: 61 are binders and 59 are "
        "nonbinders under the project's retention-at-least-75% definition. It was used "
        "only for the retrospective columns below, not for model training or candidate "
        "selection.",
        "",
        "Both average-precision columns rank Affibodies within peptide, but they use "
        "different labels and have different positive fractions. They are not a simple "
        "training-versus-test pair and should not be compared as though they were on the "
        "same scale. `Weak-label AP` asks whether selection-derived positives are "
        "near the top within peptide on 10,181 matched out-of-fold rows across 169 "
        "evaluable peptide groups. `Corrected-panel AP` asks whether measured pairs with "
        "retention of at least 75% are near the top within peptide. `Corrected-panel "
        "AUROC` is the probability that a measured binder outranks a measured nonbinder "
        "from the same peptide. `Corrected-panel Spearman` compares the complete model "
        "order with the complete numerical-retention order within peptide. AP and AUROC "
        "use 11 of 12 measured peptides because DP has no binder at the 75% threshold; "
        "Spearman uses all 12. AP is a ranking summary across all ten measured Affibodies; "
        "it is not Precision@k and is not an estimate of prospective wet-lab yield. No "
        "value below pools scores globally across peptides.",
        "",
        "| Model | Weak-selection validation AP (used to lock rules) | Direct-retention panel AP (retrospective) | Direct-retention panel AUROC (retrospective binder/nonbinder ranking) | Direct-retention panel Spearman (retrospective) |",
        "|---|---:|---:|---:|---:|",
        *metric_lines,
        "",
        "No model wins every column. MINT has the highest weak-selection validation AP "
        "(0.9219); StaB has the highest direct-retention AP (0.9368); MINT+StaB has the "
        "highest direct-retention AUROC (0.8966); and RDE has the highest direct-retention "
        "Spearman correlation (0.6593). The primary MINT+RDE rule scores "
        "0.9268/0.8783/0.6290 on those three direct-retention summaries, respectively, "
        "which is higher than MINT alone (0.9111/0.8442/0.5370) but is not the winner "
        "of any one retrospective column.",
        "",
        "The main table uses each deployed score. For structure-derived models, it first "
        "averages the five final readout logits and then calculates one metric from the "
        "resulting score, matching exhaustive candidate scoring. The structure report's "
        "mean-plus-or-minus-standard-deviation table instead calculates a metric for "
        "each seed and summarizes those five metric values. Those two aggregation "
        "conventions answer different questions and should not be mixed.",
        "",
        "The corrected-panel columns are retrospective context. Retention did not fit the "
        "displayed component weights, weak-label cutoff, or candidate ordering. The new "
        "candidate outcomes will be the actual prospective test.",
        "",
        "### Retrospective score thresholds on the same 120 measured pairs",
        "",
        "**This second table is fitted to the same 120 already-known outcomes on which "
        "it is scored. It is not an independent test and is not an estimate of future "
        "wet-lab yield.** For each model, every distinct observed score is considered as "
        "a cutoff and a pair is called recommended when its score is greater than or "
        "equal to that cutoff. The selected cutoff maximizes F1 across all 120 pairs. "
        "Exact F1 ties prefer higher precision, then fewer recommendations, then the "
        "higher threshold.",
        "",
        "Precision is the fraction of retrospectively recommended measured pairs that "
        "were binders. Recall is the fraction of all 61 measured binders that were "
        "recommended. F1 balances those two numbers. The score scales differ between "
        "models, so their numerical thresholds should not be compared directly.",
        "",
        "| Model | Retrospective score threshold (`score >=`) | Recommended measured pairs | Measured binders among recommendations | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *retrospective_threshold_lines,
        "",
        "These retention-optimized cutoffs apply only to the already-measured 120-pair "
        "exercise. They do not replace the weak-label cutoffs used to create the new "
        "candidate menus, and they did not change any prospective score, rank, or pair. "
        "Every retrospectively recommended measured pair is listed in the appendix and "
        "in `retrospective_f1_recommendations_on_measured120.csv`.",
        "",
        "One broadly effective measured Affibody contributes substantially to these "
        "aggregate threshold results: `FALTA` is a binder for 11 of the 12 peptides, and "
        "all eight models recommend all 11 of those known binder pairs. MINT also "
        "recommends the nonbinding `DP`--`FALTA` pair (retention 73.57%). The table "
        "therefore reflects both peptide-dependent discrimination and the ease of "
        "recovering a generally strong Affibody; it should not be read as pure evidence "
        "of peptide-specific recognition.",
        "",
        "## Weak-label cutoffs used for the prospective menus",
        "",
        "The seven cutoffs below were optimized for F1 on the same 10,181 weak-label "
        "out-of-fold rows used for deployment locking, never on retention. Each new "
        "candidate menu keeps score-at-least-cutoff pairs and then at most ten ranked "
        "pairs per peptide; it never pads with a below-cutoff pair. These are the cutoffs "
        "that actually govern the prospective files.",
        "",
        "| Prospective candidate rule | Weak-label score cutoff | New candidate pairs | Targets represented | Missing targets |",
        "|---|---:|---:|---:|---|",
        *prospective_cutoff_lines,
        "",
        "`all_model_candidate_menus.csv` contains every new pair assignment from all "
        "seven rules, including pair IDs and ranks. `candidate_codes_by_target.csv` is "
        "the compact model-by-peptide list of those assignments.",
        "",
        "### How the primary MINT+RDE cutoff affects its menu",
        "",
        f"The optional eligibility guard is score `{cutoff:.6f}`, the cutoff maximizing "
        "F1 on 10,181 weak-label out-of-fold rows, not on retention. Across the complete "
        f"candidate universe, {cutoff_total:,} pairs pass it; the number per peptide "
        f"ranges from {cutoff_min:,} to {cutoff_max:,}. Every peptide has at least ten "
        "passing candidates, so the guard removes none of the raw top-ten slots. It only "
        "filters lower-ranked universe rows before the ten-per-peptide cap. The cutoff is "
        "not a performance metric, and passing it does not mean a corresponding percent "
        "chance of binding.",
        "",
        "## The primary 120-pair menu",
        "",
        "The Affibody codes below are in model rank order from 1 to 10 within each "
        "peptide. The machine-readable CSV contains the exact pair ID, both complete "
        "candidate sequences (which are the MINT inputs), all four component scores, "
        "and RDE/StaB scores from "
        "each of the five readout seeds.",
        "",
        "| Peptide code | Affibody codes, ranks 1 through 10 |",
        "|---|---|",
        *code_lines,
        "",
        f"Across all 120 assignments, the primary menu uses {primary_unique} distinct "
        "Affibody designs. Reusing one Affibody for several peptide targets can reduce "
        "the number of unique constructs needed, but it can also mean that the model "
        "favors a broadly useful Affibody rather than a different optimum for each "
        "peptide.",
        "",
        reuse_paragraph,
        "",
        "`FALTA` itself is excluded from the prospective universe because all 12 of its "
        "current-target pairs are already measured. Related designs are not excluded or "
        "penalized: their similarity is a legitimate model hypothesis to test. The "
        f"closest primary candidates are Hamming distance {minimum_falta_distance} from "
        f"`FALTA`; {falta_distance_one_pairs} pair assignments representing "
        f"{falta_distance_one_designs} distinct one-change designs are at distance 1, "
        f"and {falta_distance_le_two_pairs} assignments representing "
        f"{falta_distance_le_two_designs} designs are within distance 2. In the "
        f"100-pair review subset, the corresponding pair counts are "
        f"{ready_falta_distance_one_pairs} at distance 1 and "
        f"{ready_falta_distance_le_two_pairs} within distance 2. "
        f"The five most broadly reused primary designs are {broadly_reused}. Hamming "
        "distance simply counts how many of the five code characters differ. This menu "
        "therefore tests FALTA-like, potentially broadly useful designs without assuming "
        "that such reuse represents peptide-specific recognition.",
        "",
        "## How the seven model menus differ",
        "",
        "Each rule keeps up to ten candidates per peptide, but a locked cutoff can leave "
        "a comparator with fewer than ten or no eligible candidate for a target; rows "
        "are never padded below cutoff. Exact-pair overlap means the same Affibody "
        "assigned to the same peptide; reuse "
        "for a different peptide does not count. `Same rank 1` asks whether two models "
        "make the identical first choice for a peptide. The last column reveals how "
        "strongly each rule reuses one design across targets.",
        "",
        "| Candidate rule | Candidate pairs | Targets represented | Distinct Affibody designs | Exact pairs also in primary / this menu | Same rank 1 as primary / comparable targets | Most targets sharing one design |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *summary_lines,
        "",
        incomplete_menu_note,
        "",
        "Menu agreement is not evidence that a candidate binds, and disagreement is not "
        "evidence that it fails. These are different model-generated hypotheses. Only "
        "new measurements can compare their prospective usefulness.",
        "",
        "## Selection-history and sequence checks",
        "",
        f"In the full primary 120-pair menu, {all_qc['unobserved']} pairs were never "
        f"observed in any R000--R014 file; {all_qc['r9r10']} appeared in R009 or R010 "
        "but were below the pooled top-2% boundary; "
        f"{all_qc['aff_seen_pairs']} pair assignments use "
        f"{all_qc['aff_seen_designs']} distinct Affibody identities that appeared in "
        f"strict weak-label training; and {all_qc['weak_negative']} are recorded as "
        "high-confidence weak negatives. Every exact candidate pair remains new under "
        "the universe exclusions, and every target peptide sequence was absent from "
        "strict training.",
        "",
        f"Sequence QC flags {all_qc['code_cys']} codes with an introduced cysteine, "
        f"{all_qc['full_cys']} full displayed Affibody sequences containing cysteine, and "
        f"{all_qc['motif']} sequences containing an N-X-S/T motif. An N-X-S/T motif is a "
        "potential glycosylation site in some expression systems; it is a review flag, "
        "not proof that the construct will fail. These flags do not silently remove or "
        "rerank rows.",
        "",
        "## The proposed 100-pair scientific-review subset",
        "",
        "`READY_FOR_SCIENTIFIC_REVIEW_100_PAIRS.csv` contains the primary model's ten "
        "candidates for `AF`, `AH`, `DP`, `EA`, `LV`, `MW`, `NF`, `PH`, `TL`, and `VV`. "
        "`EL` and `LL` remain in the complete 120-pair menu. The ten-target set was fixed "
        "for coverage and operational scope, not by comparing scores between peptides "
        "or by using retention outcomes. It is therefore not a global best-100 list. "
        "Ranks are copied unchanged from the complete primary menu.",
        "",
        f"Within these 100 pairs, {ready_qc['unobserved']} were not observed in any raw "
        f"round, {ready_qc['r9r10']} appeared in R009/R010 below the pooled top-2% "
        f"boundary, and {ready_qc['aff_seen_pairs']} assignments use "
        f"{ready_qc['aff_seen_designs']} Affibody identities seen in strict training. "
        f"The batch contains {ready_qc['weak_negative']} high-confidence weak negatives, "
        f"{ready_qc['code_cys']} introduced-cysteine codes, and {ready_qc['motif']} "
        "N-X-S/T motif flags.",
        "",
        "These rows are ready for scientific review, not for vendor ordering or cloning. "
        "The 58-amino-acid Affibody and SMART--HLA--linker--peptide strings are model "
        "inputs. The provider must confirm the hidden N-terminal `MA`, vector, tag, "
        "signal-peptide, and linker context before constructs are ordered.",
        "",
        "## How the new experiment should be recorded",
        "",
        "`blank_result_entry_template.csv` contains the same 100 stable sample and pair "
        "identifiers followed by deliberately empty result fields. Those fields should "
        "be filled only after new assays return. Retention of at least 75% can then be "
        "used as the project's experimental binder definition. Report the fraction of "
        "valid tested pairs that meet this definition, the numerical retention values, "
        "and the within-peptide ordering. Technical failures should be recorded rather "
        "than silently treated as nonbinders.",
        "",
        "Because the candidate list is fixed before those outcome fields are completed, "
        "the returned wet-lab measurements are the actual prospective test. Comparing "
        "MINT+RDE directly with another rule would require testing candidates unique to "
        "both prespecified menus; overlap alone cannot establish which rule is better.",
        "",
        "## Appendix: pairs recommended by retention-optimized thresholds",
        "",
        "This appendix lists every already-measured peptide--Affibody pair above each "
        "model's retrospective F1-optimal threshold. Codes are ordered by model score "
        "within peptide. These are diagnostic reclassifications of known data, not new "
        "prospective candidates. The CSV also gives every score, threshold, measured "
        "retention, and binder outcome.",
        "",
        *retrospective_appendix_lines,
        "## Files for use",
        "",
        "- `READY_FOR_SCIENTIFIC_REVIEW_100_PAIRS.csv`: 100 candidate pairs proposed for scientific review.",
        "- `blank_result_entry_template.csv`: the same 100 rows with empty result fields.",
        "- `primary_mint_rde_candidate_menu_all_12_targets.csv`: the complete primary 120-pair menu with component and seed scores.",
        "- `comparator_candidate_menus.csv`: all six cutoff-respecting comparator menus in one table; missing targets are not padded.",
        "- `all_model_candidate_menus.csv` and `candidate_codes_by_target.csv`: every new assignment from all seven prospective rules, in detailed and compact forms.",
        "- `mint_control_candidate_menu_ready_10_targets.csv`: the frozen-MINT control choices for the same ten review targets.",
        "- `overall_model_comparison.csv`: matched weak-label and corrected-panel model results.",
        "- `prospective_weak_label_cutoffs.csv`: all seven weak-label deployment cutoffs and menu sizes.",
        "- `retrospective_f1_thresholds_on_measured120.csv`: the display-only threshold, precision, recall, F1, and counts for all eight models.",
        "- `retrospective_f1_recommendations_on_measured120.csv`: every already-measured pair called recommended by each retrospective threshold, with its known outcome.",
        "- `retrospective_f1_codes_by_peptide.csv`: a compact model-by-peptide list of the same retrospective recommendations.",
        "- `retrospective_seed_metric_summary.csv`: the separate mean and standard deviation of metrics calculated once per readout seed.",
        "- `model_menu_summary.csv` and `model_overlap_by_target.csv`: agreement and reuse summaries.",
        "- `unique_affibody_designs_by_model.csv`: the actual construct roster and reuse for every model.",
        "- `primary_cutoff_audit.csv`: passing-candidate counts for the weak-label cutoff.",
        "- `libb_native_projection_candidate_handoff.xlsx`: the principal tables in one workbook.",
        "",
    ])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in MODEL_ROLES:
        parser.add_argument(f"--{role.cli_flag.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--candidate-universe-dir", required=True, type=Path)
    parser.add_argument("--additive-scores-dir", required=True, type=Path)
    parser.add_argument("--mint-scores-dir", required=True, type=Path)
    parser.add_argument("--rde-scores-dir", required=True, type=Path)
    parser.add_argument("--stab-scores-dir", required=True, type=Path)
    parser.add_argument("--weak-metrics-csv", required=True, type=Path)
    parser.add_argument("--corrected120-metrics-csv", required=True, type=Path)
    parser.add_argument("--corrected120-predictions-csv", required=True, type=Path)
    parser.add_argument("--seed-metric-summary-csv", required=True, type=Path)
    parser.add_argument("--parity-audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    final_output = _validate_private_output_path(args.output_dir)
    _require(not final_output.exists(), "output directory exists; refusing overwrite")

    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict] = {}
    input_dirs: dict[str, Path] = {}
    for role in MODEL_ROLES:
        directory = getattr(args, role.cli_flag).expanduser().resolve()
        frame, manifest = _read_shortlist(directory, role)
        frames[role.key] = frame
        manifests[role.key] = manifest
        input_dirs[role.key] = directory
    _validate_shared_provenance(manifests)
    candidate_manifest, candidate_manifest_path = _load_candidate_universe_manifest(
        args.candidate_universe_dir, manifests
    )
    parity_audit = _load_parity_audit(
        args.parity_audit_dir,
        {
            "rde": args.rde_scores_dir,
            "stab": args.stab_scores_dir,
        },
    )
    cutoff_audit = _read_primary_cutoff_audit(
        input_dirs["mint_rde_primary"], manifests["mint_rde_primary"]
    )

    menus = {
        role.key: _add_common_fields(frames[role.key], role.key, role.display_name)
        for role in MODEL_ROLES
    }
    score_roots = {
        component: getattr(args, str(spec["cli_flag"]))
        for component, spec in COMPONENT_SCORE_SPECS.items()
    }
    enriched_primary, component_score_provenance = _load_component_scores_for_primary(
        menus["mint_rde_primary"], score_roots, manifests["equal_all4"]
    )
    enriched_primary = _annotate_primary_candidate_qc(enriched_primary)
    menus["mint_rde_primary"] = enriched_primary
    primary = enriched_primary[_primary_output_columns(enriched_primary)].copy()
    _require(len(primary) == 120, "primary menu changed during rendering")
    ready = primary.loc[primary["peptide_design_code"].isin(READY_TARGETS)].copy()
    ready_reuse = ready.groupby("affibody_design_code")["peptide_design_code"]
    ready["selected_for_n_peptides"] = ready_reuse.transform("nunique").astype(np.int64)
    ready["selected_for_peptides"] = ready_reuse.transform(
        lambda values: ";".join(sorted(set(map(str, values))))
    )
    ready["assay_slot"] = ready["model_rank_within_peptide"].astype(int)
    ready["sample_id"] = [
        f"LIBB-{peptide}-{int(slot):02d}"
        for peptide, slot in zip(ready["peptide_design_code"], ready["assay_slot"])
    ]
    _require(len(ready) == 100 and ready["sample_id"].nunique() == 100,
             "ready batch is not exactly ten targets by ten candidates")
    _require(
        _pair_set(ready) == _pair_set(primary, READY_TARGETS),
        "ready batch is not an unchanged primary-menu subset",
    )
    ready["batch_design"] = (
        f"100 candidate pairs proposed for scientific review; {ready['affibody_design_code'].nunique()} distinct "
        "Affibody designs; not a Cartesian cross"
    )
    ready["sequence_readiness_warning"] = (
        "MODEL INPUT ONLY; provider must confirm hidden N-terminal MA, vector, tags, "
        "signal peptide, and linker context before construct ordering"
    )
    review_leading_columns = [
        "sample_id",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "model_rank_within_peptide",
        "affibody_design_code",
        "model_score",
        "provider_displayed_58aa_affibody_sequence",
        "mint_input_affibody_sequence",
        "mint_input_smart_hla_linker_peptide_sequence",
        "pair_uid",
        "selected_for_n_peptides",
        "selected_for_peptides",
        "batch_design",
        "sequence_readiness_warning",
    ]
    review_columns = review_leading_columns + [
        column for column in _primary_output_columns(ready)
        if column not in review_leading_columns
    ]
    review_batch = ready[review_columns].copy()
    result_entry = review_batch[[
        "sample_id",
        "pair_uid",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "model_rank_within_peptide",
        "affibody_design_code",
    ]].copy()
    outcome_fields = [
        "retention_at_30min_percent",
        "binder_ge_75",
        "technical_failure_status",
        "replicate_id",
        "assay_date",
        "experimental_batch_id",
        "notes",
    ]
    for column in outcome_fields:
        result_entry[column] = ""
    _require(
        result_entry[outcome_fields].fillna("").astype(str).eq("").all().all(),
        "result template contains an outcome value at release",
    )

    model_summary, overlap_by_target = _build_model_summaries(menus)
    overlap_matrix = _build_overlap_matrix(menus)
    constructs = _build_construct_roster(menus)
    codes_by_target = _build_codes_by_target(menus)
    all_menus = pd.concat(
        [menus[role.key][_menu_columns()] for role in MODEL_ROLES], ignore_index=True
    )
    comparator_menus = all_menus.loc[
        ~all_menus["candidate_model_key"].eq("mint_rde_primary")
    ].copy()
    _require(
        len(comparator_menus) == sum(
            len(menus[role.key]) for role in MODEL_ROLES if role.key != "mint_rde_primary"
        ),
        "combined comparator menu row count changed",
    )
    comparator_ready_menus = comparator_menus.loc[
        comparator_menus["peptide_design_code"].isin(READY_TARGETS)
    ].copy()
    mint_control_ready = comparator_ready_menus.loc[
        comparator_ready_menus["candidate_model_key"].eq("mint_control")
    ].copy()
    _require(len(mint_control_ready) == 100,
             "MINT control is not exactly ten candidates for each review target")
    primary_constructs = constructs.loc[
        constructs["model_key"].eq("mint_rde_primary")
    ].copy()

    # The prospective menus, ranks, and fixed ten-target subset above are fully
    # constructed before any retention-derived table is opened.  Everything
    # loaded below is display-only and is never passed back into menu assembly.
    overall_metrics = _load_metric_comparison(
        args.weak_metrics_csv, args.corrected120_metrics_csv
    )
    seed_metric_summary_path = args.seed_metric_summary_csv.expanduser().resolve()
    seed_metric_summary = pd.read_csv(seed_metric_summary_path)
    seed_required = {
        "model_key", "fits", "ap_mean", "ap_sd", "auroc_mean", "auroc_sd",
        "spearman_mean", "spearman_sd",
    }
    _require(seed_required.issubset(seed_metric_summary.columns),
             "seed metric summary schema changed")
    _require(len(seed_metric_summary) == 8, "seed metric summary is not eight models")
    _require(not seed_metric_summary["model_key"].astype(str).duplicated().any(),
             "seed metric summary repeats a model")
    expected_seed_models = {spec[3] for spec in METRIC_MODEL_SPECS}
    _require(set(seed_metric_summary["model_key"].astype(str)) == expected_seed_models,
             "seed metric summary model set changed")
    fixed_models = {"additive_7site", "mint_layer5"}
    for row in seed_metric_summary.itertuples(index=False):
        expected_fits = 1 if str(row.model_key) in fixed_models else 5
        _require(int(row.fits) == expected_fits,
                 f"unexpected fit count for seed summary model {row.model_key}")
    prospective_cutoffs = _build_prospective_cutoff_summary(menus, manifests)
    (
        retrospective_thresholds,
        retrospective_recommendations,
        retrospective_codes_by_target,
        retrospective_prediction_provenance,
    ) = _load_retrospective_threshold_analysis(
        args.corrected120_predictions_csv,
        args.corrected120_metrics_csv,
    )

    final_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{final_output.name}.staging-", dir=final_output.parent
    ))
    os.chmod(staging, 0o700)
    paths = {
        "review_100_pairs": staging / "READY_FOR_SCIENTIFIC_REVIEW_100_PAIRS.csv",
        "blank_result_entry_template": staging / "blank_result_entry_template.csv",
        "primary_all12_menu": staging / "primary_mint_rde_candidate_menu_all_12_targets.csv",
        "primary_ready10_menu": staging / "primary_mint_rde_candidate_menu_ready_10_targets.csv",
        "comparator_menus": staging / "comparator_candidate_menus.csv",
        "comparator_ready_menus": staging / "comparator_candidate_menus_ready_10_targets.csv",
        "mint_control_ready10_menu": staging / "mint_control_candidate_menu_ready_10_targets.csv",
        "all_model_menus": staging / "all_model_candidate_menus.csv",
        "overall_model_comparison": staging / "overall_model_comparison.csv",
        "prospective_weak_label_cutoffs": staging / "prospective_weak_label_cutoffs.csv",
        "retrospective_f1_thresholds": staging / "retrospective_f1_thresholds_on_measured120.csv",
        "retrospective_f1_recommendations": staging / "retrospective_f1_recommendations_on_measured120.csv",
        "retrospective_f1_codes_by_peptide": staging / "retrospective_f1_codes_by_peptide.csv",
        "retrospective_seed_metric_summary": staging / "retrospective_seed_metric_summary.csv",
        "model_menu_summary": staging / "model_menu_summary.csv",
        "model_overlap_by_target": staging / "model_overlap_by_target.csv",
        "model_pair_overlap_matrix": staging / "model_pair_overlap_matrix.csv",
        "unique_affibody_designs_by_model": staging / "unique_affibody_designs_by_model.csv",
        "primary_unique_affibody_designs": staging / "primary_unique_affibody_designs.csv",
        "candidate_codes_by_target": staging / "candidate_codes_by_target.csv",
        "primary_cutoff_audit": staging / "primary_cutoff_audit.csv",
    }
    outputs = {
        "review_100_pairs": review_batch,
        "blank_result_entry_template": result_entry,
        "primary_all12_menu": primary,
        "primary_ready10_menu": ready[_primary_output_columns(ready)],
        "comparator_menus": comparator_menus,
        "comparator_ready_menus": comparator_ready_menus,
        "mint_control_ready10_menu": mint_control_ready,
        "all_model_menus": all_menus,
        "overall_model_comparison": overall_metrics,
        "prospective_weak_label_cutoffs": prospective_cutoffs,
        "retrospective_f1_thresholds": retrospective_thresholds,
        "retrospective_f1_recommendations": retrospective_recommendations,
        "retrospective_f1_codes_by_peptide": retrospective_codes_by_target,
        "retrospective_seed_metric_summary": seed_metric_summary,
        "model_menu_summary": model_summary,
        "model_overlap_by_target": overlap_by_target,
        "model_pair_overlap_matrix": overlap_matrix,
        "unique_affibody_designs_by_model": constructs,
        "primary_unique_affibody_designs": primary_constructs,
        "candidate_codes_by_target": codes_by_target,
        "primary_cutoff_audit": cutoff_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)
        os.chmod(paths[name], 0o600)

    workbook_path = staging / "libb_native_projection_candidate_handoff.xlsx"
    _write_workbook(
        workbook_path,
        review_batch,
        result_entry,
        primary,
        model_summary,
        overall_metrics,
        prospective_cutoffs,
        retrospective_thresholds,
        retrospective_recommendations,
        retrospective_codes_by_target,
        codes_by_target,
        all_menus,
        mint_control_ready,
        comparator_ready_menus,
    )
    report_path = staging / "ensemble_candidate_selection_report.md"
    report_path.write_text(
        _report_text(
            menus,
            manifests,
            candidate_manifest,
            model_summary,
            overall_metrics,
            prospective_cutoffs,
            retrospective_thresholds,
            retrospective_codes_by_target,
            cutoff_audit,
            codes_by_target,
            parity_audit,
        ),
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)

    manifest = {
        "schema_version": "libb-native-projection-candidate-handoff-v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "primary_candidate_rule": "locked MINT+RDE ensemble shortlist exactly",
        "primary_lock_id": MODEL_ROLES[0].lock_id,
        "primary_menu": {
            "target_count": 12,
            "candidate_pairs": 120,
            "candidates_per_target": 10,
            "distinct_affibody_designs": int(primary["affibody_design_code"].nunique()),
        },
        "scientific_review_batch": {
            "target_codes_in_fixed_order": list(READY_TARGETS),
            "target_count": 10,
            "candidate_pairs": 100,
            "candidates_per_target": 10,
            "distinct_affibody_designs": int(ready["affibody_design_code"].nunique()),
            "unchanged_subset_of_primary_menu": True,
            "not_a_cartesian_cross": True,
            "target_choice_used_model_scores": False,
            "target_choice_used_retention_outcomes": False,
        },
        "outcome_access": {
            "raw_pair_level_retention_table_read_by_handoff_builder": True,
            "pair_level_retention_opened_only_after_prospective_menus_were_constructed": True,
            "pair_level_retention_used_only_for_retrospective_threshold_display": True,
            "precomputed_aggregate_corrected120_metric_csv_read_for_report": True,
            "input_shortlists_hashed_before_aggregate_metric_csv_loaded": True,
            "aggregate_metrics_never_passed_to_ranking_code": True,
            "retention_values_present_in_retrospective_outputs": True,
            "retention_values_present_in_prospective_candidate_menu_outputs": False,
            "retention_used_to_add_remove_or_rerank_candidates": False,
            "retention_optimized_thresholds_used_for_prospective_candidates": False,
            "blank_result_headers_are_not_observed_outcomes": True,
        },
        "validation": {
            "primary_menu_exactly_12_by_10": True,
            "comparator_menus_have_zero_to_ten_candidates_per_target_without_padding": True,
            "all_seven_menus_share_candidate_universe": True,
            "all_seven_menus_share_weak_label_lock_bundle": True,
            "reused_component_score_hashes_identical": True,
            "affibody_code_to_displayed_sequence_mapping_verified": True,
            "all_four_deployed_component_scores_joined_to_primary_rows": True,
            "five_rde_and_five_stab_seed_scores_present_per_primary_row": True,
            "primary_score_recomputed_as_sigmoid_mean_mint_rde_logit": True,
            "candidate_universe_exclusions_validated": True,
            "ready_batch_exactly_primary_subset": True,
            "result_entry_fields_blank_at_release": True,
            "retrospective_thresholds_use_score_greater_equal_rule": True,
            "retrospective_threshold_ties_prefer_precision_then_fewer_then_higher": True,
            "retrospective_recommendation_rows_equal_summary_counts": True,
            "all_seven_prospective_weak_label_cutoffs_reported": True,
            "published_by_atomic_directory_rename": True,
        },
        "model_menu_counts": {
            role.key: {
                "candidate_pairs": int(len(menus[role.key])),
                "targets_with_candidates": int(
                    menus[role.key]["peptide_design_code"].nunique()
                ),
                "missing_targets": sorted(
                    set(EXPECTED_TARGET_SEQUENCES)
                    - set(menus[role.key]["peptide_design_code"].astype(str))
                ),
            }
            for role in MODEL_ROLES
        },
        "retrospective_threshold_analysis": {
            "panel_pairs": RETROSPECTIVE_PANEL_ROWS,
            "binders": RETROSPECTIVE_PANEL_BINDERS,
            "nonbinders": RETROSPECTIVE_PANEL_NONBINDERS,
            "binder_definition": "target_retention >= 75",
            "decision_rule": "deployed_model_score >= model_specific_threshold",
            "objective": "maximize micro F1 on these same 120 already-measured pairs",
            "tie_break_order": [
                "higher precision",
                "fewer recommendations",
                "higher threshold",
            ],
            "independent_evaluation": False,
            "prospective_candidate_rule": False,
            "models": len(retrospective_thresholds),
            "recommended_model_pair_rows": len(retrospective_recommendations),
        },
        "shortlist_inputs": {
            role.key: {
                "directory": str(input_dirs[role.key]),
                "manifest_sha256": sha256_file(input_dirs[role.key] / "manifest.json"),
                "shortlist_sha256": sha256_file(input_dirs[role.key] / "wetlab_shortlist.csv"),
                "lock_id": role.lock_id,
            }
            for role in MODEL_ROLES
        },
        "candidate_universe_input": {
            "directory": str(args.candidate_universe_dir.expanduser().resolve()),
            "manifest_path": str(candidate_manifest_path),
            "manifest_sha256": sha256_file(candidate_manifest_path),
            "candidate_rows": int(candidate_manifest["outputs"]["candidate_rows"]),
        },
        "metric_inputs": {
            "weak_metrics": {
                "path": str(args.weak_metrics_csv.expanduser().resolve()),
                "sha256": sha256_file(args.weak_metrics_csv.expanduser().resolve()),
                "used_for": "display and documentation of weak-label model comparison",
            },
            "corrected120_aggregate_metrics": {
                "path": str(args.corrected120_metrics_csv.expanduser().resolve()),
                "sha256": sha256_file(args.corrected120_metrics_csv.expanduser().resolve()),
                "used_for": "retrospective aggregate comparison displayed in report only",
            },
            "corrected120_seed_metric_summary": {
                "path": str(seed_metric_summary_path),
                "sha256": sha256_file(seed_metric_summary_path),
                "used_for": "separate per-seed metric uncertainty reference",
            },
            "corrected120_pair_predictions_and_outcomes": retrospective_prediction_provenance,
        },
        "component_score_inputs": component_score_provenance,
        "candidate_scorer_parity_audit": parity_audit,
        "outputs": {},
        "interpretation": (
            "Scores rank similarity to the selection-derived positive class. They are "
            "not retention percentages or calibrated probabilities of wet-lab binding. "
            "The retrospective F1 thresholds are fitted on the already-measured panel "
            "and do not govern prospective candidate selection."
        ),
    }
    for name, path in paths.items():
        manifest["outputs"][name] = {
            "path": path.name,
            "rows": int(len(outputs[name])),
            "sha256": sha256_file(path),
        }
    manifest["outputs"]["report"] = {
        "path": report_path.name,
        "bytes": int(report_path.stat().st_size),
        "sha256": sha256_file(report_path),
    }
    manifest["outputs"]["workbook"] = {
        "path": workbook_path.name,
        "bytes": int(workbook_path.stat().st_size),
        "sha256": sha256_file(workbook_path),
    }
    manifest_path = staging / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    _require(not final_output.exists(), "final output appeared while handoff was staged")
    os.replace(staging, final_output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
