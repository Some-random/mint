#!/usr/bin/env python
"""Evaluate arbitrary saved model scores on the canonical 108-cell LibA panel.

This utility does not train a model.  It joins target-free, already-saved model
scores to direct-retention outcomes only at evaluation time.  Every model/seed
group must score exactly the canonical 9 peptide x 12 Affibody matrix.  A group
with missing, extra, duplicate, or non-finite predictions is recorded as
incomparable and is not silently evaluated on an intersection.

Prediction sources are described by a JSON configuration so heterogeneous
historical artifacts can be evaluated under one metric implementation.  Each
source specifies a stable model name, file, pair-ID column, score column,
filters, and either a fixed seed label or a seed column.  Paths in the config
are resolved relative to the repository root.

The F1-optimal score threshold is explicitly retrospective: it is selected on
the same 108 direct-retention outcomes on which it is reported.  It is saved as
a descriptive operating point, not as a calibrated or prospectively validated
wet-lab decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.wetlab_metrics import (  # noqa: E402
    TIE_POLICY,
    evaluate_wetlab_predictions,
    rank_within_peptide,
)


SCHEMA_VERSION = "liba-model-comparison-108-v1"
CONFIG_SCHEMA_VERSION = "liba-model-comparison-config-v1"
RETENTION_THRESHOLD = 75.0
EXPECTED_ROWS = 108
EXPECTED_PEPTIDES = 9
EXPECTED_AFFIBODIES = 12
EXPECTED_BINDERS = 38
EXPECTED_NONBINDERS = 70
K_VALUES = (1, 3)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _read_string_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _membership_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def file_provenance(path: Path) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), "source file does not exist: {}".format(resolved))
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(stat.st_size),
        "mtime_utc": _utc_timestamp(stat.st_mtime),
    }


def load_liba_panel(path: Path) -> pd.DataFrame:
    """Load and fail-closed validate the canonical complete LibA panel."""

    path = Path(path)
    _require(path.is_file(), "retention panel does not exist: {}".format(path))
    frame = _read_string_csv(path)
    required = {
        "pair_uid",
        "library",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
    }
    missing = required.difference(frame.columns)
    _require(not missing, "retention panel is missing columns {}".format(sorted(missing)))
    frame = frame.loc[frame["library"].eq("LibA")].copy()
    output = frame[
        [
            "pair_uid",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
        ]
    ].copy()

    for column in ("pair_uid", "peptide_design_code", "affibody_design_code"):
        values = output[column].astype(str)
        _require(not bool(values.eq("").any()), "{} contains empty values".format(column))
        _require(
            bool(values.eq(values.str.strip()).all()),
            "{} contains surrounding whitespace".format(column),
        )
    _require(output["pair_uid"].is_unique, "LibA panel contains duplicate pair_uid values")
    _require(
        not bool(
            output.duplicated(
                ["peptide_design_code", "affibody_design_code"], keep=False
            ).any()
        ),
        "LibA panel contains duplicate peptide/Affibody pairs",
    )
    _require(
        not bool(output["target_retention"].astype(str).eq("").any()),
        "all 108 LibA cells must have a direct retention value",
    )
    output["target_retention"] = pd.to_numeric(
        output["target_retention"], errors="raise"
    ).astype(float)
    retention = output["target_retention"].to_numpy(dtype=float)
    _require(bool(np.isfinite(retention).all()), "retention contains non-finite values")
    _require(
        bool(((retention >= 0.0) & (retention <= 100.0)).all()),
        "retention must be in [0, 100]",
    )

    peptides = sorted(set(output["peptide_design_code"].astype(str)))
    affibodies = sorted(set(output["affibody_design_code"].astype(str)))
    _require(len(output) == EXPECTED_ROWS, "canonical LibA panel must contain 108 rows")
    _require(len(peptides) == EXPECTED_PEPTIDES, "LibA panel must contain 9 peptides")
    _require(
        len(affibodies) == EXPECTED_AFFIBODIES,
        "LibA panel must contain 12 Affibodies",
    )
    observed_pairs = set(
        zip(output["peptide_design_code"], output["affibody_design_code"])
    )
    expected_pairs = {
        (peptide, affibody) for peptide in peptides for affibody in affibodies
    }
    _require(
        observed_pairs == expected_pairs,
        "LibA panel is not the complete 9-by-12 peptide/Affibody matrix",
    )

    # Derive labels from the measurement instead of trusting a potentially
    # stale label column carried by an input or prediction artifact.
    output["target_binder"] = output["target_retention"].ge(
        RETENTION_THRESHOLD
    ).astype(int)
    binders = int(output["target_binder"].sum())
    _require(
        binders == EXPECTED_BINDERS and len(output) - binders == EXPECTED_NONBINDERS,
        "LibA retention >=75 must yield 38 binders and 70 non-binders",
    )
    return output.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    _require(path.is_file(), "prediction config does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "prediction config must be a JSON object")
    _require(
        payload.get("schema_version") == CONFIG_SCHEMA_VERSION,
        "prediction config schema_version must be {}".format(CONFIG_SCHEMA_VERSION),
    )
    sources = payload.get("sources")
    _require(isinstance(sources, list) and sources, "prediction config sources must be non-empty")
    return payload


def _filter_source(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    output = frame
    for column, expected in filters.items():
        _require(column in output.columns, "prediction filter column is missing: {}".format(column))
        if isinstance(expected, list):
            expected_values = {str(value) for value in expected}
            output = output.loc[output[column].astype(str).isin(expected_values)]
        else:
            output = output.loc[output[column].astype(str).eq(str(expected))]
    return output.copy()


def _source_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_prediction_sources(
    config: Mapping[str, Any], panel: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Normalize valid source groups and audit every configured artifact.

    Data-quality membership failures are retained in ``comparability`` and the
    affected group is skipped.  Configuration errors such as missing columns
    fail immediately because they indicate an invalid evaluator invocation.
    """

    panel_ids = set(panel["pair_uid"].astype(str))
    expected_hash = _membership_sha256(panel_ids)
    normalized_frames: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    seen_models = set()

    for source_index, spec_value in enumerate(config["sources"]):
        _require(isinstance(spec_value, dict), "each source spec must be an object")
        spec = dict(spec_value)
        required = {"model", "path", "id_column", "score_column"}
        missing = required.difference(spec)
        _require(not missing, "source spec is missing keys {}".format(sorted(missing)))
        model = str(spec["model"])
        _require(model and model not in seen_models, "source model names must be non-empty and unique")
        seen_models.add(model)
        display_name = str(spec.get("display_name", model))
        path = _source_path(str(spec["path"]))
        _require(path.is_file(), "prediction source does not exist: {}".format(path))
        frame = _read_string_csv(path)
        id_column = str(spec["id_column"])
        score_column = str(spec["score_column"])
        seed_column = spec.get("seed_column")
        _require(id_column in frame.columns, "prediction ID column is missing: {}".format(id_column))
        _require(score_column in frame.columns, "prediction score column is missing: {}".format(score_column))
        if seed_column is not None:
            seed_column = str(seed_column)
            _require(seed_column in frame.columns, "prediction seed column is missing: {}".format(seed_column))
            _require("seed" not in spec, "source cannot specify both seed and seed_column")
        filters = spec.get("filters", {})
        _require(isinstance(filters, dict), "source filters must be a JSON object")
        selected = _filter_source(frame, filters)
        source_record = {
            "source_index": source_index,
            "model": model,
            "display_name": display_name,
            "id_column": id_column,
            "score_column": score_column,
            "seed_column": seed_column,
            "fixed_seed": None if seed_column is not None else str(spec.get("seed", "fixed")),
            "filters": filters,
            "file": file_provenance(path),
            "rows_in_file": int(len(frame)),
            "rows_after_filters": int(len(selected)),
        }
        source_records.append(source_record)

        if len(selected) == 0:
            audit_rows.append(
                {
                    "source_index": source_index,
                    "model": model,
                    "display_name": display_name,
                    "seed": str(spec.get("seed", "fixed")),
                    "source_path": str(path),
                    "rows_after_filters": 0,
                    "unique_pair_ids": 0,
                    "duplicate_pair_id_rows": 0,
                    "missing_pair_ids": EXPECTED_ROWS,
                    "extra_pair_ids": 0,
                    "nonfinite_scores": 0,
                    "expected_membership_sha256": expected_hash,
                    "observed_membership_sha256": _membership_sha256([]),
                    "missing_pair_ids_json": _json_compact(sorted(panel_ids)),
                    "extra_pair_ids_json": "[]",
                    "status": "incomparable_no_rows_after_filters",
                }
            )
            continue

        work = selected[[id_column, score_column] + ([seed_column] if seed_column else [])].copy()
        work = work.rename(columns={id_column: "pair_uid", score_column: "score"})
        if seed_column:
            work = work.rename(columns={seed_column: "seed"})
        else:
            work["seed"] = str(spec.get("seed", "fixed"))
        work["pair_uid"] = work["pair_uid"].astype(str)
        work["seed"] = work["seed"].astype(str)
        numeric_score = pd.to_numeric(work["score"], errors="coerce")
        work["score"] = numeric_score.astype(float)

        for seed, group in work.groupby("seed", sort=True, dropna=False):
            seed = str(seed)
            observed_ids = set(group["pair_uid"].astype(str))
            missing_ids = sorted(panel_ids.difference(observed_ids))
            extra_ids = sorted(observed_ids.difference(panel_ids))
            duplicate_rows = int(group["pair_uid"].duplicated(keep=False).sum())
            scores = group["score"].to_numpy(dtype=float)
            nonfinite = int((~np.isfinite(scores)).sum())
            issues = []
            if bool(group["pair_uid"].eq("").any()):
                issues.append("empty_pair_id")
            if duplicate_rows:
                issues.append("duplicate_pair_ids")
            if missing_ids:
                issues.append("missing_pair_ids")
            if extra_ids:
                issues.append("extra_pair_ids")
            if nonfinite:
                issues.append("nonfinite_scores")
            if len(group) != EXPECTED_ROWS:
                issues.append("wrong_row_count")
            status = "comparable_exact_108" if not issues else "incomparable_" + "+".join(issues)
            audit_rows.append(
                {
                    "source_index": source_index,
                    "model": model,
                    "display_name": display_name,
                    "seed": seed,
                    "source_path": str(path),
                    "rows_after_filters": int(len(group)),
                    "unique_pair_ids": int(len(observed_ids)),
                    "duplicate_pair_id_rows": duplicate_rows,
                    "missing_pair_ids": int(len(missing_ids)),
                    "extra_pair_ids": int(len(extra_ids)),
                    "nonfinite_scores": nonfinite,
                    "expected_membership_sha256": expected_hash,
                    "observed_membership_sha256": _membership_sha256(observed_ids),
                    "missing_pair_ids_json": _json_compact(missing_ids),
                    "extra_pair_ids_json": _json_compact(extra_ids),
                    "status": status,
                }
            )
            if issues:
                continue
            valid = group[["pair_uid", "seed", "score"]].copy()
            valid.insert(0, "display_name", display_name)
            valid.insert(0, "model", model)
            normalized_frames.append(valid)

    comparability = pd.DataFrame(audit_rows).sort_values(
        ["source_index", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    _require(normalized_frames, "none of the configured prediction groups matches the exact LibA panel")
    predictions = pd.concat(normalized_frames, ignore_index=True)
    duplicates = predictions.duplicated(["model", "seed", "pair_uid"], keep=False)
    _require(not bool(duplicates.any()), "normalized predictions contain duplicate model/seed/pair rows")
    return predictions, comparability, source_records


def _within_peptide_binary_metrics(
    merged: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    excluded = []
    for peptide, group in merged.groupby("peptide_design_code", sort=True):
        labels = group["target_binder"].to_numpy(dtype=int)
        row = {
            "peptide_design_code": str(peptide),
            "within_peptide_average_precision": float("nan"),
            "within_peptide_auroc": float("nan"),
            "within_peptide_binary_evaluable": int(np.unique(labels).size == 2),
        }
        if row["within_peptide_binary_evaluable"]:
            scores = group["score"].to_numpy(dtype=float)
            row["within_peptide_average_precision"] = float(
                average_precision_score(labels, scores)
            )
            row["within_peptide_auroc"] = float(roc_auc_score(labels, scores))
        else:
            excluded.append(str(peptide))
        rows.append(row)
    per_peptide = pd.DataFrame(rows)
    evaluable = per_peptide.loc[per_peptide["within_peptide_binary_evaluable"].eq(1)]
    _require(len(evaluable) > 0, "within-peptide binary metrics are undefined for all peptides")
    summary = {
        "within_peptide_average_precision_mean": float(
            evaluable["within_peptide_average_precision"].mean()
        ),
        "within_peptide_auroc_mean": float(evaluable["within_peptide_auroc"].mean()),
        "within_peptide_binary_evaluable_groups": int(len(evaluable)),
        "within_peptide_binary_excluded_groups": _json_compact(sorted(excluded)),
    }
    return per_peptide, summary


def evaluate_models(
    predictions: pd.DataFrame, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute requested aggregate, per-peptide, and ranked outputs."""

    aggregate_rows = []
    peptide_frames = []
    ranked_frames = []
    group_columns = ["model", "display_name", "seed"]
    for identifiers, group in predictions.groupby(group_columns, sort=True):
        model, display_name, seed = identifiers
        merged = group.merge(panel, on="pair_uid", how="inner", validate="one_to_one")
        _require(len(merged) == EXPECTED_ROWS, "validated prediction/panel join lost rows")
        summary, per_peptide, ranked = evaluate_wetlab_predictions(
            merged,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
            score_column="score",
            binder_column="target_binder",
            retention_column="target_retention",
            k_values=K_VALUES,
        )
        binary_by_peptide, binary_summary = _within_peptide_binary_metrics(merged)
        per_peptide = per_peptide.merge(
            binary_by_peptide,
            on="peptide_design_code",
            how="left",
            validate="one_to_one",
        )
        top_three = (
            ranked.loc[ranked["within_peptide_rank"].le(3)]
            .sort_values(["peptide_design_code", "within_peptide_rank"], kind="mergesort")
            .groupby("peptide_design_code", sort=True)["affibody_design_code"]
            .apply(lambda values: _json_compact(list(map(str, values))))
        )
        per_peptide = per_peptide.merge(
            top_three.rename("top3_affibodies"),
            left_on="peptide_design_code",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        peptide_columns = [
            "peptide_design_code",
            "n_candidates",
            "n_binders",
            "within_peptide_binary_evaluable",
            "within_peptide_average_precision",
            "within_peptide_auroc",
            "within_peptide_spearman",
            "precision_at_1",
            "precision_at_3",
            "experimental_best_retention",
            "best_retention_at_1",
            "best_retention_at_3",
            "regret_at_1",
            "regret_at_3",
            "top1_affibody",
            "top3_affibodies",
        ]
        per_peptide = per_peptide[peptide_columns]
        per_peptide.insert(0, "seed", str(seed))
        per_peptide.insert(0, "display_name", str(display_name))
        per_peptide.insert(0, "model", str(model))
        peptide_frames.append(per_peptide)

        aggregate_rows.append(
            {
                "model": str(model),
                "display_name": str(display_name),
                "seed": str(seed),
                "n_examples": int(summary["n_examples"]),
                "n_peptides": int(summary["n_peptides"]),
                "n_binders": int(summary["n_binders"]),
                "binder_fraction": float(summary["binder_fraction"]),
                **binary_summary,
                "within_peptide_spearman_mean": float(summary["within_peptide_spearman_mean"]),
                "within_peptide_spearman_evaluable_groups": int(
                    summary["within_peptide_spearman_evaluable_groups"]
                ),
                "precision_at_1": float(summary["peptide_macro_precision_at_1"]),
                "precision_at_3": float(summary["peptide_macro_precision_at_3"]),
                "best_retention_at_1": float(summary["peptide_macro_best_retention_at_1"]),
                "best_retention_at_3": float(summary["peptide_macro_best_retention_at_3"]),
                "regret_at_1": float(summary["peptide_macro_regret_at_1"]),
                "regret_at_3": float(summary["peptide_macro_regret_at_3"]),
                "distinct_top1_affibodies": int(summary["distinct_top1_affibodies"]),
                "global_auroc_secondary": float(summary["global_auroc"]),
                "global_average_precision_secondary": float(
                    summary["global_average_precision"]
                ),
            }
        )

        ranked = ranked[
            [
                "model",
                "display_name",
                "seed",
                "pair_uid",
                "peptide_design_code",
                "affibody_design_code",
                "score",
                "target_retention",
                "target_binder",
                "within_peptide_rank",
                "selected_at_1",
                "selected_at_3",
            ]
        ]
        ranked_frames.append(ranked)

    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["model", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    per_peptide_output = pd.concat(peptide_frames, ignore_index=True).sort_values(
        ["model", "seed", "peptide_design_code"], kind="mergesort"
    ).reset_index(drop=True)
    ranked_output = pd.concat(ranked_frames, ignore_index=True).sort_values(
        ["model", "seed", "peptide_design_code", "within_peptide_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    return aggregate, per_peptide_output, ranked_output


def select_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    """Select score >= threshold by F1, precision, count, then threshold."""

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    _require(score_array.ndim == 1 and label_array.ndim == 1, "scores and labels must be vectors")
    _require(len(score_array) == len(label_array) and len(score_array) > 0, "bad threshold arrays")
    _require(bool(np.isfinite(score_array).all()), "threshold scores contain non-finite values")
    _require(set(np.unique(label_array)).issubset({0, 1}), "threshold labels must be binary")
    positives = int(label_array.sum())
    candidates = []
    for threshold in np.unique(score_array):
        selected = score_array >= threshold
        selected_count = int(selected.sum())
        tp = int(np.logical_and(selected, label_array == 1).sum())
        fp = selected_count - tp
        fn = positives - tp
        tn = len(label_array) - tp - fp - fn
        precision = Fraction(tp, selected_count) if selected_count else Fraction(0, 1)
        recall = Fraction(tp, positives) if positives else Fraction(0, 1)
        denominator = 2 * tp + fp + fn
        f1 = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
        result = {
            "threshold": float(threshold),
            "selected_candidates": selected_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        candidates.append(((f1, precision, -selected_count, float(threshold)), result))
    return max(candidates, key=lambda item: item[0])[1]


def build_retrospective_recommendations(
    predictions: pd.DataFrame, panel: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return explicit same-panel thresholds, all decisions, and selected rows."""

    threshold_rows = []
    decision_frames = []
    for identifiers, group in predictions.groupby(
        ["model", "display_name", "seed"], sort=True
    ):
        model, display_name, seed = identifiers
        merged = group.merge(panel, on="pair_uid", how="inner", validate="one_to_one")
        chosen = select_f1_threshold(merged["score"], merged["target_binder"])
        threshold_rows.append(
            {
                "model": str(model),
                "display_name": str(display_name),
                "seed": str(seed),
                **chosen,
                "decision_rule": "score >= threshold",
                "threshold_selection": "maximum F1 on this same 108-pair direct-retention panel",
                "tie_break": "higher precision, then fewer selected candidates, then higher threshold",
                "threshold_is_evaluation_selected": 1,
                "threshold_is_calibrated_probability": 0,
                "threshold_is_prospectively_validated": 0,
            }
        )
        decisions = rank_within_peptide(
            merged,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
            score_column="score",
        ).drop(columns=["_original_row"])
        decisions["retrospective_f1_threshold"] = float(chosen["threshold"])
        decisions["recommended_by_retrospective_f1_threshold"] = decisions["score"].ge(
            float(chosen["threshold"])
        ).astype(int)
        decisions = decisions.rename(
            columns={"within_peptide_rank": "within_peptide_score_rank"}
        )
        decisions["model"] = str(model)
        decisions["display_name"] = str(display_name)
        decisions["seed"] = str(seed)
        decision_frames.append(decisions)

    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["model", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    decisions = pd.concat(decision_frames, ignore_index=True)
    decisions = decisions.sort_values(
        ["model", "seed", "peptide_design_code", "score", "affibody_design_code"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    decision_columns = [
        "model",
        "display_name",
        "seed",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "score",
        "retrospective_f1_threshold",
        "recommended_by_retrospective_f1_threshold",
        "within_peptide_score_rank",
        "target_retention",
        "target_binder",
    ]
    decisions = decisions[decision_columns]
    recommendations = decisions.loc[
        decisions["recommended_by_retrospective_f1_threshold"].eq(1)
    ].copy()
    return thresholds, decisions, recommendations.reset_index(drop=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists() and not temporary.exists(), "output exists: {}".format(path))
    try:
        frame.to_csv(temporary, index=False, float_format="%.12g")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists() and not temporary.exists(), "output exists: {}".format(path))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir = validate_private_output_path(output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    output_records = {}
    for filename, frame in tables.items():
        _require(Path(filename).name == filename, "output filename must be local")
        path = output_dir / filename
        _atomic_write_csv(frame, path)
        output_records[filename] = {**file_provenance(path), "rows": int(len(frame))}
    payload = dict(manifest)
    payload["outputs"] = output_records
    _atomic_write_json(payload, output_dir / "manifest.json")
    return payload


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-panel", type=Path, required=True)
    parser.add_argument("--prediction-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    panel = load_liba_panel(args.retention_panel)
    config = load_config(args.prediction_config)
    predictions, comparability, source_records = load_prediction_sources(config, panel)
    aggregate, per_peptide, ranked = evaluate_models(predictions, panel)
    thresholds, decisions, recommendations = build_retrospective_recommendations(
        predictions, panel
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "evaluation_only_no_training",
        "script": file_provenance(Path(__file__)),
        "sources": {
            "retention_panel": file_provenance(args.retention_panel),
            "prediction_config": file_provenance(args.prediction_config),
            "prediction_artifacts": source_records,
        },
        "panel_contract": {
            "library": "LibA",
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_AFFIBODIES,
            "complete_cartesian_matrix": True,
            "retention_binder_threshold": RETENTION_THRESHOLD,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "membership_sha256": _membership_sha256(panel["pair_uid"]),
        },
        "comparability": {
            "groups_configured": int(len(comparability)),
            "groups_evaluated": int(comparability["status"].eq("comparable_exact_108").sum()),
            "groups_skipped": int(comparability["status"].ne("comparable_exact_108").sum()),
            "membership_failures_are_not_intersected": True,
        },
        "metric_contract": {
            "primary": [
                "within_peptide_average_precision_mean",
                "within_peptide_auroc_mean",
                "within_peptide_spearman_mean",
                "precision_at_1",
                "precision_at_3",
                "best_retention_at_1",
                "best_retention_at_3",
                "regret_at_1",
                "regret_at_3",
            ],
            "secondary": ["global_auroc_secondary", "global_average_precision_secondary"],
            "within_peptide_binary_rule": "AP and AUROC require both binder classes; all-one-class peptide rows are excluded from both macro means and listed explicitly",
            "within_peptide_aggregation": "unweighted arithmetic mean over evaluable peptide rows",
            "precision_at_k": "within each peptide, binder fraction among the k highest-scored Affibodies; then unweighted mean across all nine peptides",
            "best_retention_at_k": "within each peptide, highest direct retention among the k highest-scored Affibodies; then unweighted mean",
            "regret_at_k": "within each peptide, experimentally best retention minus best retention among the model's top k; then unweighted mean",
            "ranking_tie_policy": TIE_POLICY,
            "k_values": list(K_VALUES),
        },
        "threshold_contract": {
            "selection": "maximum F1 on this same 108-pair direct-retention evaluation panel",
            "decision_rule": "score >= threshold",
            "tie_break": "higher precision, then fewer selected candidates, then higher threshold",
            "retrospective_only": True,
            "calibrated_probability": False,
            "prospectively_validated": False,
        },
    }
    published = publish_outputs(
        args.output_dir,
        {
            "aggregate_metrics.csv": aggregate,
            "per_peptide_metrics.csv": per_peptide,
            "ranked_predictions.csv": ranked,
            "retrospective_thresholds.csv": thresholds,
            "retrospective_threshold_decisions.csv": decisions,
            "retrospective_recommendations.csv": recommendations,
            "artifact_comparability.csv": comparability,
        },
        manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "models": sorted(set(aggregate["model"].astype(str))),
                "groups_evaluated": published["comparability"]["groups_evaluated"],
                "groups_skipped": published["comparability"]["groups_skipped"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
