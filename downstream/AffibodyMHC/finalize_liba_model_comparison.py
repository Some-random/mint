#!/usr/bin/env python3
"""Finalize a retention-isolated comparison of frozen LibA model scores.

This command has a deliberately one-way evaluation boundary:

1. load exact four-column, target-free score sidecars;
2. construct every prespecified deployment score (including seed aggregation);
3. validate the retention-blind weak-OOF locks and their metric provenance; and
4. only then open the sealed 108-pair direct-retention panel.

The command evaluates all configured models.  It never selects a model, seed,
aggregation rule, or prospective cutoff from retention.  The F1-maximizing
threshold is emitted only as a clearly labelled retrospective description of
the already-measured panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.code_only_baseline import sha256_file
from downstream.AffibodyMHC.evaluate_liba_esmfold2_readouts_sealed import (
    DEFAULT_EVALUATION_ROOT,
    EXPECTED_AFFIBODIES,
    EXPECTED_BINDERS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_MEMBERSHIP_SHA256,
    EXPECTED_MISSING_MATRIX_CELLS,
    EXPECTED_NONBINDERS,
    EXPECTED_PEPTIDES,
    _validate_evaluation_sidecar,
)
from downstream.AffibodyMHC.evaluate_liba_model_comparison import (
    build_retrospective_recommendations,
    evaluate_models,
    publish_outputs,
)
from downstream.AffibodyMHC.evaluate_libb_readouts_sealed import (
    BLINDED_PREDICTION_COLUMNS,
    _membership_sha256,
    load_blinded_predictions,
    load_libb_retention_audit,
    validate_blinded_prediction_groups,
)
from downstream.AffibodyMHC.wetlab_metrics import TIE_POLICY


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "liba-final-model-comparison-v1"
CONFIG_SCHEMA_VERSION = "liba-final-model-evaluation-config-v1"
HANDOFF_LOCK_SCHEMA_VERSION = "liba-weak-oof-deployment-locks-v1"
GENERIC_SCORE_LOCK_SCHEMA_VERSION = "affibody-weak-oof-score-lock-v1"
GENERIC_ENSEMBLE_SCHEMA_VERSION = "affibody-weak-oof-ensemble-v1"
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
PROBABILITY_CLIP = 1e-7

HANDOFF_COMPARISON_COLUMNS = (
    "model_id",
    "display_name",
    "evaluation_pairs",
    "within_peptide_auroc",
    "evaluable_peptides_for_within_peptide_auroc",
    "within_peptide_average_precision",
    "evaluable_peptides_for_within_peptide_ap",
    "within_peptide_spearman",
    "evaluable_peptides_for_within_peptide_spearman",
    "weak_oof_within_peptide_average_precision",
    "weak_oof_evaluable_peptides_for_within_peptide_ap",
    "retention_optimized_threshold",
    "pairs_above_retention_optimized_threshold",
    "binders_above_retention_optimized_threshold",
    "retrospective_precision",
    "retrospective_recall",
    "retrospective_f1",
    "global_auroc",
    "global_average_precision",
)

SUMMARY_METRICS = (
    "within_peptide_average_precision_mean",
    "within_peptide_auroc_mean",
    "within_peptide_spearman_mean",
    "precision_at_1",
    "precision_at_3",
    "best_retention_at_1",
    "best_retention_at_3",
    "regret_at_1",
    "regret_at_3",
    "global_auroc_secondary",
    "global_average_precision_secondary",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve(path_value: str | Path, base: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT if base is None else base) / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"JSON input does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"JSON input must contain an object: {path}")
    return payload


def _mean_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] >= 2, "mean-logit needs at least two seeds")
    _require(bool(np.isfinite(values).all()), "mean-logit input contains non-finite values")
    clipped = np.clip(values, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    logits = np.log(clipped) - np.log1p(-clipped)
    mean = logits.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-mean))


def load_evaluation_config(path: Path) -> dict[str, Any]:
    """Load the prespecified target-free deployment mapping."""

    payload = _read_json(path)
    _require(
        payload.get("schema_version") == CONFIG_SCHEMA_VERSION,
        f"evaluation config schema must be {CONFIG_SCHEMA_VERSION}",
    )
    _require(payload.get("library") == "LibA", "evaluation config is not LibA")
    _require(
        payload.get("retention_labels_allowed") is False,
        "evaluation config must explicitly forbid retention labels",
    )
    models = payload.get("models")
    _require(isinstance(models, list) and models, "evaluation config models must be non-empty")
    allowed_model_fields = {
        "model_id",
        "display_name",
        "source_model",
        "deployment",
        "handoff_eligible",
    }
    seen: set[str] = set()
    for spec in models:
        _require(isinstance(spec, dict), "each evaluation model must be an object")
        _require(
            not set(spec).difference(allowed_model_fields),
            f"unexpected evaluation model fields: {sorted(set(spec).difference(allowed_model_fields))}",
        )
        for field in ("model_id", "display_name", "source_model", "deployment"):
            _require(field in spec, f"evaluation model is missing {field}")
        model_id = str(spec["model_id"])
        _require(SAFE_NAME.fullmatch(model_id) is not None, f"unsafe model_id: {model_id}")
        _require(model_id not in seen, f"duplicate model_id: {model_id}")
        seen.add(model_id)
        _require(str(spec["display_name"]).strip(), f"{model_id} display_name is empty")
        _require(str(spec["source_model"]).strip(), f"{model_id} source_model is empty")
        _require(isinstance(spec.get("handoff_eligible", False), bool), f"{model_id} bad eligibility flag")
        deployment = spec["deployment"]
        _require(isinstance(deployment, dict), f"{model_id} deployment must be an object")
        method = deployment.get("method")
        if method == "fixed_seed":
            _require(set(deployment) == {"method", "seed"}, f"{model_id} fixed-seed contract changed")
            _require(str(deployment["seed"]).strip(), f"{model_id} deployment seed is empty")
        elif method == "mean_logit":
            _require(set(deployment) == {"method", "seeds"}, f"{model_id} mean-logit contract changed")
            seeds = list(map(str, deployment["seeds"]))
            _require(len(seeds) >= 2 and len(seeds) == len(set(seeds)), f"{model_id} bad seed list")
        else:
            raise ValueError(f"{model_id} unsupported deployment method: {method}")
    return payload


def validate_target_free_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on score panels before any direct outcomes are opened."""

    membership = validate_blinded_prediction_groups(predictions)
    group_sizes = predictions.groupby(["model", "seed"], sort=True).size()
    _require(
        bool(group_sizes.eq(EXPECTED_EVALUATION_ROWS).all()),
        "every target-free model/seed must contain exactly 108 opaque IDs",
    )
    _require(
        membership == EXPECTED_MEMBERSHIP_SHA256,
        "target-free prediction membership is not the locked LibA 108-ID panel",
    )
    return predictions.copy()


def build_deployment_predictions(
    predictions: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Apply only the deployment aggregation rules fixed in the config."""

    rows = []
    for spec in config["models"]:
        model_id = str(spec["model_id"])
        source_model = str(spec["source_model"])
        source = predictions.loc[predictions["model"].astype(str).eq(source_model)].copy()
        _require(not source.empty, f"no target-free scores for source_model {source_model}")
        available_seeds = set(source["seed"].astype(str))
        deployment = spec["deployment"]
        if deployment["method"] == "fixed_seed":
            seeds = [str(deployment["seed"])]
            _require(set(seeds) == available_seeds, f"{model_id} fixed seed does not exactly match source seeds")
        else:
            seeds = list(map(str, deployment["seeds"]))
            _require(set(seeds) == available_seeds, f"{model_id} mean-logit seeds do not exactly match source seeds")

        blocks = []
        for seed in seeds:
            block = source.loc[source["seed"].astype(str).eq(seed), ["eval_row_id", "score"]].copy()
            block = block.rename(columns={"score": seed})
            blocks.append(block)
        aligned = blocks[0]
        for block in blocks[1:]:
            aligned = aligned.merge(block, on="eval_row_id", how="inner", validate="one_to_one")
        _require(len(aligned) == EXPECTED_EVALUATION_ROWS, f"{model_id} aggregation lost IDs")
        if deployment["method"] == "fixed_seed":
            score = aligned[seeds[0]].to_numpy(dtype=float)
        else:
            score = _mean_logit(aligned[seeds].to_numpy(dtype=float))
        rows.append(
            pd.DataFrame(
                {
                    "eval_row_id": aligned["eval_row_id"].astype(str),
                    "model": model_id,
                    "seed": "deployment",
                    "score": score,
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    output["source_role"] = "deployment"
    validate_blinded_prediction_groups(output)
    _require(
        not bool(output.duplicated(["model", "eval_row_id"]).any()),
        "deployment output contains duplicate model/ID rows",
    )
    return output


def _check_hash(path: Path, expected: str, purpose: str) -> None:
    _require(re.fullmatch(r"[0-9a-f]{64}", str(expected)) is not None, f"bad {purpose} SHA")
    _require(path.is_file(), f"missing {purpose}: {path}")
    _require(sha256_file(path) == expected, f"{purpose} SHA mismatch: {path}")


def _extract_weak_metric(
    model_id: str, lock: Mapping[str, Any]
) -> tuple[float, int, dict[str, Any]]:
    # The handoff compiler keeps these two values flat because they are also
    # the exact public comparison-table column names.  Accept the nested form
    # only for backwards-compatible unit fixtures.
    metrics = lock.get("weak_oof_metrics", {})
    _require(isinstance(metrics, dict), f"{model_id} weak_oof_metrics are malformed")
    ap = float(
        lock.get(
            "weak_oof_within_peptide_average_precision",
            metrics.get("within_peptide_average_precision", math.nan),
        )
    )
    groups = int(
        lock.get(
            "weak_oof_evaluable_peptides_for_within_peptide_ap",
            metrics.get("evaluable_peptides_for_average_precision", -1),
        )
    )
    _require(math.isfinite(ap) and 0.0 <= ap <= 1.0, f"{model_id} weak OOF AP is invalid")
    _require(groups > 0, f"{model_id} weak OOF evaluable-peptide count is invalid")

    provenance = lock.get("weak_metric_provenance")
    _require(isinstance(provenance, dict), f"{model_id} weak metric provenance is missing")
    required = {
        "generic_lock_path",
        "generic_lock_sha256",
        "candidate_metrics_path",
        "candidate_metrics_sha256",
        "candidate",
    }
    _require(required.issubset(provenance), f"{model_id} weak metric provenance is incomplete")
    generic_path = _resolve(str(provenance["generic_lock_path"]))
    metric_path = _resolve(str(provenance["candidate_metrics_path"]))
    _check_hash(generic_path, str(provenance["generic_lock_sha256"]), f"{model_id} generic lock")
    _check_hash(metric_path, str(provenance["candidate_metrics_sha256"]), f"{model_id} candidate metrics")

    generic = _read_json(generic_path)
    _require(
        generic.get("schema_version") == GENERIC_SCORE_LOCK_SCHEMA_VERSION,
        f"{model_id} generic weak-lock schema changed",
    )
    _require(generic.get("library") == "LibA", f"{model_id} generic weak lock is not LibA")
    _require(generic.get("retention_labels_read") is False, f"{model_id} weak lock read retention")
    candidate = str(provenance["candidate"])
    _require(str(generic.get("candidate")) == candidate, f"{model_id} weak-lock candidate differs")
    generic_metric = generic.get("selection_provenance", {}).get("candidate_metrics", {})
    generic_ap = generic_metric.get(
        "within_peptide_ap",
        generic_metric.get("weak_oof_within_peptide_average_precision", math.nan),
    )
    generic_groups = generic_metric.get(
        "within_peptide_evaluable",
        generic_metric.get("weak_oof_evaluable_peptides_for_within_peptide_ap", -1),
    )
    _require(
        math.isclose(float(generic_ap), ap, abs_tol=1e-12),
        f"{model_id} weak AP differs from generic lock",
    )
    _require(
        int(generic_groups) == groups,
        f"{model_id} weak evaluable count differs from generic lock",
    )

    table = pd.read_csv(metric_path, dtype={"candidate": str})
    ap_column = (
        "weak_oof_within_peptide_average_precision"
        if "weak_oof_within_peptide_average_precision" in table.columns
        else "within_peptide_ap"
    )
    group_column = (
        "weak_oof_evaluable_peptides_for_within_peptide_ap"
        if "weak_oof_evaluable_peptides_for_within_peptide_ap" in table.columns
        else "within_peptide_evaluable"
    )
    _require(
        {"candidate", ap_column, group_column}.issubset(table.columns),
        f"{model_id} candidate metric table schema changed",
    )
    selected = table.loc[table["candidate"].astype(str).eq(candidate)]
    _require(len(selected) == 1, f"{model_id} weak candidate must occur exactly once")
    row = selected.iloc[0]
    _require(math.isclose(float(row[ap_column]), ap, abs_tol=1e-12), f"{model_id} metric-table AP differs")
    _require(int(row[group_column]) == groups, f"{model_id} metric-table group count differs")
    return ap, groups, {
        "generic_lock": {"path": str(generic_path), "sha256": sha256_file(generic_path)},
        "candidate_metrics": {"path": str(metric_path), "sha256": sha256_file(metric_path)},
        "candidate": candidate,
    }


def load_weak_lock_metrics(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load weak-only AP values from the handoff locks and verify their sources."""

    manifest = _read_json(path)
    _require(
        manifest.get("schema_version") == HANDOFF_LOCK_SCHEMA_VERSION,
        f"weak lock schema must be {HANDOFF_LOCK_SCHEMA_VERSION}",
    )
    _require(manifest.get("library") == "LibA", "weak lock manifest is not LibA")
    _require(manifest.get("retention_labels_read") is False, "weak locks read retention labels")
    locks = manifest.get("models")
    _require(isinstance(locks, dict), "weak lock manifest has no models")
    expected_eligible = {
        str(spec["model_id"])
        for spec in config["models"]
        if bool(spec.get("handoff_eligible", False))
    }
    observed_eligible = {
        str(model_id)
        for model_id, lock in locks.items()
        if isinstance(lock, dict) and lock.get("scientifically_eligible") is True
    }
    _require(
        observed_eligible == expected_eligible,
        "config handoff eligibility differs from scientifically eligible weak locks",
    )
    display_by_id = {str(spec["model_id"]): str(spec["display_name"]) for spec in config["models"]}
    rows = []
    provenance = {}
    for model_id in sorted(expected_eligible):
        lock = locks[model_id]
        _require(lock.get("threshold_uses_retention") is False, f"{model_id} threshold used retention")
        _require(str(lock.get("display_name")) == display_by_id[model_id], f"{model_id} display name differs")
        ap, groups, record = _extract_weak_metric(model_id, lock)
        rows.append(
            {
                "model_id": model_id,
                "weak_oof_within_peptide_average_precision": ap,
                "weak_oof_evaluable_peptides_for_within_peptide_ap": groups,
            }
        )
        provenance[model_id] = record
    _require(rows, "no scientifically eligible weak-lock metrics were found")
    return pd.DataFrame(rows), {"manifest": manifest, "verified_metric_sources": provenance}


def _as_metric_predictions(frame: pd.DataFrame, display_by_model: Mapping[str, str]) -> pd.DataFrame:
    output = frame[list(BLINDED_PREDICTION_COLUMNS)].rename(columns={"eval_row_id": "pair_uid"})
    output["display_name"] = output["model"].astype(str).map(display_by_model)
    output["display_name"] = output["display_name"].fillna(output["model"].astype(str))
    return output[["model", "display_name", "seed", "pair_uid", "score"]]


def summarize_raw_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Keep seed means, sample SDs, and individual values without choosing a seed."""

    rows = []
    for (model, display_name), group in metrics.groupby(["model", "display_name"], sort=True):
        group = group.sort_values("seed", kind="mergesort")
        row: dict[str, Any] = {
            "model": str(model),
            "display_name": str(display_name),
            "n_seeds": int(len(group)),
            "seeds_json": json.dumps(group["seed"].astype(str).tolist(), separators=(",", ":")),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_seed_mean"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{metric}_seed_sd"] = (
                float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")
            )
            row[f"{metric}_individual_json"] = json.dumps(
                [float(value) if math.isfinite(value) else None for value in values],
                separators=(",", ":"),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_handoff_comparison(
    deployment_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    weak_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Emit the exact explicit schema consumed by the LibA handoff builder."""

    eligible_specs = [spec for spec in config["models"] if spec.get("handoff_eligible", False)]
    eligible_ids = {str(spec["model_id"]) for spec in eligible_specs}
    metrics = deployment_metrics.loc[deployment_metrics["model"].isin(eligible_ids)].copy()
    _require(len(metrics) == len(eligible_ids), "expected one deployment metric row per eligible model")
    _require(metrics["model"].is_unique, "eligible deployment metrics contain duplicate models")
    chosen = thresholds.loc[thresholds["model"].isin(eligible_ids)].copy()
    _require(len(chosen) == len(eligible_ids) and chosen["model"].is_unique, "eligible threshold rows changed")
    joined = metrics.merge(
        chosen,
        on=["model", "display_name", "seed"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_threshold"),
    ).merge(
        weak_metrics,
        left_on="model",
        right_on="model_id",
        how="inner",
        validate="one_to_one",
    )
    _require(len(joined) == len(eligible_ids), "handoff comparison join lost a model")
    output = pd.DataFrame(
        {
            "model_id": joined["model"].astype(str),
            "display_name": joined["display_name"].astype(str),
            "evaluation_pairs": joined["n_examples"].astype(int),
            "within_peptide_auroc": joined["within_peptide_auroc_mean"].astype(float),
            "evaluable_peptides_for_within_peptide_auroc": joined[
                "within_peptide_binary_evaluable_groups"
            ].astype(int),
            "within_peptide_average_precision": joined[
                "within_peptide_average_precision_mean"
            ].astype(float),
            "evaluable_peptides_for_within_peptide_ap": joined[
                "within_peptide_binary_evaluable_groups"
            ].astype(int),
            "within_peptide_spearman": joined["within_peptide_spearman_mean"].astype(float),
            "evaluable_peptides_for_within_peptide_spearman": joined[
                "within_peptide_spearman_evaluable_groups"
            ].astype(int),
            "weak_oof_within_peptide_average_precision": joined[
                "weak_oof_within_peptide_average_precision"
            ].astype(float),
            "weak_oof_evaluable_peptides_for_within_peptide_ap": joined[
                "weak_oof_evaluable_peptides_for_within_peptide_ap"
            ].astype(int),
            "retention_optimized_threshold": joined["threshold"].astype(float),
            "pairs_above_retention_optimized_threshold": joined[
                "selected_candidates"
            ].astype(int),
            "binders_above_retention_optimized_threshold": joined["true_positives"].astype(int),
            "retrospective_precision": joined["precision"].astype(float),
            "retrospective_recall": joined["recall"].astype(float),
            "retrospective_f1": joined["f1"].astype(float),
            "global_auroc": joined["global_auroc_secondary"].astype(float),
            "global_average_precision": joined[
                "global_average_precision_secondary"
            ].astype(float),
        }
    )
    _require(tuple(output.columns) == HANDOFF_COMPARISON_COLUMNS, "handoff comparison schema changed")
    return output.sort_values("model_id", kind="mergesort").reset_index(drop=True)


def _prediction_group_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        rows.append(
            {
                "model": str(model),
                "seed": str(seed),
                "rows": int(len(group)),
                "unique_opaque_ids": int(group["eval_row_id"].nunique()),
                "membership_sha256": _membership_sha256(group["eval_row_id"]),
                "finite_scores": int(np.isfinite(group["score"].to_numpy(dtype=float)).sum()),
                "status": "frozen_exact_108",
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the frozen-score -> weak-lock -> sealed-outcome sequence."""

    config_path = args.evaluation_config.resolve()
    config = load_evaluation_config(config_path)

    # Phase 1: these exact-schema files contain opaque IDs and scores only.
    score_paths = [path.resolve() for path in args.predictions]
    raw_predictions = validate_target_free_scores(load_blinded_predictions(score_paths))
    deployment = build_deployment_predictions(raw_predictions, config)
    score_audit = _prediction_group_audit(raw_predictions)
    deployment_audit = _prediction_group_audit(deployment)

    # Phase 2: weak-label locks and metrics are checked before outcomes exist in memory.
    weak_lock_path = args.weak_oof_locks.resolve()
    weak_metrics, weak_lock_audit = load_weak_lock_metrics(weak_lock_path, config)

    # Phase 3: this is the first point at which direct retention is opened.
    evaluation_labels = args.evaluation_labels.resolve()
    evaluation_manifest = args.evaluation_manifest.resolve()
    sealed_manifest = _validate_evaluation_sidecar(evaluation_labels, evaluation_manifest)
    audit = load_libb_retention_audit(
        evaluation_labels,
        expected_rows=EXPECTED_EVALUATION_ROWS,
        expected_peptides=EXPECTED_PEPTIDES,
        expected_affibodies=EXPECTED_AFFIBODIES,
        expected_missing_cells=EXPECTED_MISSING_MATRIX_CELLS,
        expected_binders=EXPECTED_BINDERS,
        expected_nonbinders=EXPECTED_NONBINDERS,
    )
    _require(
        _membership_sha256(audit["eval_row_id"]) == EXPECTED_MEMBERSHIP_SHA256,
        "sealed LibA outcome membership changed",
    )
    panel = audit.rename(columns={"eval_row_id": "pair_uid"})
    display_by_source = {
        str(spec["source_model"]): str(spec["display_name"]) for spec in config["models"]
    }
    display_by_deployment = {
        str(spec["model_id"]): str(spec["display_name"]) for spec in config["models"]
    }
    raw_metric_input = _as_metric_predictions(raw_predictions, display_by_source)
    deployment_metric_input = _as_metric_predictions(deployment, display_by_deployment)
    raw_metrics, raw_per_peptide, raw_ranked = evaluate_models(raw_metric_input, panel)
    deployment_metrics, deployment_per_peptide, deployment_ranked = evaluate_models(
        deployment_metric_input, panel
    )
    thresholds, threshold_decisions, threshold_recommendations = (
        build_retrospective_recommendations(deployment_metric_input, panel)
    )
    handoff_comparison = build_handoff_comparison(
        deployment_metrics, thresholds, weak_metrics, config
    )
    seed_summary = summarize_raw_seeds(raw_metrics)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase_order": [
            "target_free_score_sidecars_frozen_and_validated",
            "prespecified_deployment_aggregates_built",
            "retention_blind_weak_oof_locks_validated",
            "sealed_direct_retention_opened_for_final_evaluation",
        ],
        "retention_usage": {
            "training": False,
            "early_stopping": False,
            "model_selection": False,
            "seed_selection": False,
            "aggregation_selection": False,
            "prospective_cutoff_selection": False,
            "final_retrospective_evaluation": True,
        },
        "retrospective_threshold": {
            "purpose": "descriptive operating point on the already-measured 108-pair panel",
            "selection": "maximum F1 on the same sealed panel",
            "prospectively_validated": False,
            "calibrated_probability": False,
        },
        "panel": {
            "rows": EXPECTED_EVALUATION_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_AFFIBODIES,
            "binders_retention_ge_75": EXPECTED_BINDERS,
            "nonbinders_retention_lt_75": EXPECTED_NONBINDERS,
            "membership_sha256": EXPECTED_MEMBERSHIP_SHA256,
        },
        "metric_contract": {
            "primary_context": "rank Affibodies separately within each peptide",
            "within_peptide_binary_evaluable_groups": 5,
            "within_peptide_spearman_evaluable_groups": 9,
            "precision_k": [1, 3],
            "ranking_tie_policy": TIE_POLICY,
            "global_auroc_and_average_precision_are_secondary": True,
        },
        "sources": {
            "evaluation_config": {
                "path": str(config_path),
                "sha256": sha256_file(config_path),
            },
            "target_free_predictions": [
                {"path": str(path), "sha256": sha256_file(path)} for path in score_paths
            ],
            "weak_oof_locks": {
                "path": str(weak_lock_path),
                "sha256": sha256_file(weak_lock_path),
                "verified_metric_sources": weak_lock_audit["verified_metric_sources"],
            },
            "sealed_evaluation_labels": {
                "path": str(evaluation_labels),
                "sha256": sha256_file(evaluation_labels),
            },
            "sealed_evaluation_manifest": {
                "path": str(evaluation_manifest),
                "sha256": sha256_file(evaluation_manifest),
                "schema_version": sealed_manifest["schema_version"],
            },
        },
        "models_evaluated": [str(spec["model_id"]) for spec in config["models"]],
        "handoff_models": handoff_comparison["model_id"].astype(str).tolist(),
        "handoff_comparison_schema": list(HANDOFF_COMPARISON_COLUMNS),
    }
    deployment_sidecar = deployment[list(BLINDED_PREDICTION_COLUMNS)].copy()
    return publish_outputs(
        args.output_dir,
        {
            "target_free_prediction_group_audit.csv": score_audit,
            "frozen_deployment_prediction_group_audit.csv": deployment_audit,
            "frozen_deployment_predictions.csv": deployment_sidecar,
            "raw_metrics_by_seed.csv": raw_metrics,
            "raw_metrics_seed_summary.csv": seed_summary,
            "raw_per_peptide_metrics.csv": raw_per_peptide,
            "raw_ranked_predictions.csv": raw_ranked,
            "deployment_metrics.csv": deployment_metrics,
            "deployment_per_peptide_metrics.csv": deployment_per_peptide,
            "deployment_ranked_predictions.csv": deployment_ranked,
            "retrospective_f1_thresholds.csv": thresholds,
            "retrospective_f1_threshold_decisions.csv": threshold_decisions,
            "retrospective_f1_recommendations.csv": threshold_recommendations,
            "weak_oof_metrics_from_locks.csv": weak_metrics,
            "model_comparison_for_handoff.csv": handoff_comparison,
        },
        manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Exact target-free eval_row_id,model,seed,score CSV; repeat as needed.",
    )
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--weak-oof-locks", type=Path, required=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    published = run(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "models_evaluated": published["models_evaluated"],
                "handoff_models": published["handoff_models"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
