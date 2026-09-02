#!/usr/bin/env python
"""Select a frozen MINT representation layer using weak labels only.

This evaluator is deliberately matched to the primary Affibody analysis:

* LibA and LibB are fitted separately;
* the primary C0, library-local double-identity-cold PN pools are reused;
* three diagonal identity-blocked folds use split seed 17;
* every fold is standardized from its training rows only;
* a balanced L2 logistic readout is fitted for each layer/C pair; and
* the layer and C are selected solely by pooled weak-validation log loss.

Direct retention measurements are touched only after the weak-label selection
has been frozen.  The reported F1-optimal operating threshold is intentionally
labelled retrospective: it is selected and evaluated on the same retention
panel and is not a calibrated or prospectively validated decision threshold.

Before screening intermediate layers, the program checks that the newly
extracted layer-33 features and resulting frozen-head predictions reproduce the
canonical merged cache.  It separately reports the expected small difference
from the historical public-table probabilities, whose direct-retention rows
were scored from an older feature archive.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "affibody-mint-multilayer-evaluation-v1"
LIBRARIES = ("LibA", "LibB")
CANONICAL_LAYERS = (1, 5, 9, 13, 17, 21, 25, 29, 33)
CANONICAL_C_GRID = matched.CANONICAL_C_GRID
CANONICAL_FOLDS = matched.CANONICAL_FOLDS
CANONICAL_SPLIT_SEED = matched.CANONICAL_SPLIT_SEED
RETENTION_BINDER_THRESHOLD = 75.0
FEATURE_DIMENSION = 2560
DEFAULT_LAYER33_FEATURE_TOLERANCE = 1e-5
DEFAULT_LAYER33_PREDICTION_TOLERANCE = 5e-3
PREDICTION_TOLERANCE_REASON = (
    "A fresh 32-GPU extraction differs from the canonical float32 layer-33 cache "
    "by at most 2.146e-6 per feature. At the canonical practical liblinear solver "
    "tolerance (1e-4), the high-dimensional LibB C=0.1 refit amplifies that tiny "
    "input perturbation to a measured maximum probability difference of 0.003460 "
    "(LibA: 0.0000383). The 0.005 bound covers this measured numerical "
    "re-extraction/refit variation while the independent feature-parity bound "
    "remains 1e-5."
)
METADATA_COLUMNS = (
    "row_index",
    "cache_uid",
    "source_kind",
    "library",
    "pair_uid",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)
DEFAULT_ROWS = REPO_ROOT / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv"
DEFAULT_ROWS_MANIFEST = REPO_ROOT / "private_data/derived/mint_weak_cache_v1/rows/manifest.json"
DEFAULT_CANONICAL_CACHE = (
    REPO_ROOT
    / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz"
)
DEFAULT_REFERENCE_RUNS = {
    "LibA": REPO_ROOT
    / "private_data/experiments/mint_cached_primary_practical_liba_double_cold_v1",
    "LibB": REPO_ROOT
    / "private_data/experiments/mint_cached_primary_practical_libb_double_cold_v1",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def layer_feature_name(layer):
    """Return the stable merged-archive key for one MINT layer."""
    layer = int(layer)
    _require(0 <= layer <= 99, "layer is outside the supported key range")
    return "mint_layer_{:02d}_chain_mean".format(layer)


def sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _load_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _shell_join(arguments):
    """Python-3.7-compatible equivalent of :func:`shlex.join`."""
    return " ".join(shlex.quote(str(value)) for value in arguments)


def load_archive_header(feature_archive):
    """Read and validate only the small metadata arrays from an NPZ archive."""
    feature_archive = Path(feature_archive)
    _require(feature_archive.is_file(), "missing multilayer archive {}".format(feature_archive))
    with np.load(str(feature_archive), allow_pickle=False) as archive:
        _require("representation_layers" in archive.files, "archive lacks representation_layers")
        layers = tuple(int(value) for value in np.asarray(archive["representation_layers"]).tolist())
        _require(layers and len(layers) == len(set(layers)), "duplicate/empty representation layers")
        for layer in layers:
            name = layer_feature_name(layer)
            _require(name in archive.files, "archive lacks {}".format(name))
        missing = set(METADATA_COLUMNS).difference(archive.files)
        _require(not missing, "archive lacks metadata {}".format(sorted(missing)))
        header = {column: np.asarray(archive[column]).copy() for column in METADATA_COLUMNS}
        n_rows = len(header["row_index"])
        for column, values in header.items():
            _require(len(values) == n_rows, "metadata {} length mismatch".format(column))
    return layers, header


def load_layer_matrix(feature_archive, layer):
    """Load one [rows, 2560] layer matrix without retaining other layers."""
    name = layer_feature_name(layer)
    with np.load(str(feature_archive), allow_pickle=False) as archive:
        _require(name in archive.files, "archive lacks {}".format(name))
        values = np.asarray(archive[name], dtype=np.float32).copy()
    _require(values.ndim == 2 and values.shape[1] == FEATURE_DIMENSION, "bad layer matrix")
    _require(bool(np.isfinite(values).all()), "layer {} contains non-finite values".format(layer))
    return values


def load_canonical_cache(canonical_cache):
    canonical_cache = Path(canonical_cache)
    _require(canonical_cache.is_file(), "missing canonical cache {}".format(canonical_cache))
    with np.load(str(canonical_cache), allow_pickle=False) as archive:
        required = set(METADATA_COLUMNS).union({"mint_chain_mean"})
        missing = required.difference(archive.files)
        _require(not missing, "canonical cache lacks {}".format(sorted(missing)))
        header = {column: np.asarray(archive[column]).copy() for column in METADATA_COLUMNS}
        features = np.asarray(archive["mint_chain_mean"], dtype=np.float32).copy()
    _require(
        features.shape == (len(header["row_index"]), FEATURE_DIMENSION),
        "canonical feature shape mismatch",
    )
    _require(bool(np.isfinite(features).all()), "canonical features contain non-finite values")
    return header, features


def validate_row_alignment(rows, archive_header, canonical_header=None):
    """Prove that row CSV, multilayer NPZ, and optional canonical NPZ align."""
    _require(len(rows) == len(archive_header["row_index"]), "row/archive count mismatch")
    for column in METADATA_COLUMNS:
        _require(column in rows.columns, "row table lacks {}".format(column))
        left = rows[column].astype(str).to_numpy()
        right = np.asarray(archive_header[column]).astype(str)
        _require(np.array_equal(left, right), "row/archive {} mismatch".format(column))
        if canonical_header is not None:
            canonical = np.asarray(canonical_header[column]).astype(str)
            _require(np.array_equal(left, canonical), "row/canonical {} mismatch".format(column))


def feature_difference(new_features, canonical_features, chunk_rows=1024):
    """Return stable layer-33 parity statistics without one giant float64 copy."""
    left = np.asarray(new_features)
    right = np.asarray(canonical_features)
    _require(left.shape == right.shape, "feature parity shape mismatch")
    maximum = 0.0
    absolute_sum = 0.0
    square_sum = 0.0
    total = int(left.size)
    for start in range(0, len(left), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(left))
        difference = left[start:stop].astype(np.float64) - right[start:stop].astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(difference))))
        absolute_sum += float(np.sum(np.abs(difference), dtype=np.float64))
        square_sum += float(np.sum(difference * difference, dtype=np.float64))
    return {
        "rows": int(left.shape[0]),
        "features": int(left.shape[1]),
        "max_abs_difference": maximum,
        "mean_abs_difference": absolute_sum / float(total),
        "root_mean_square_difference": math.sqrt(square_sum / float(total)),
        "new_sha256": sha256_array(left),
        "canonical_sha256": sha256_array(right),
        "bitwise_identical": bool(np.array_equal(left, right)),
    }


def prepare_library(rows, library):
    """Reconstruct the primary PN pool/folds without reading retention outcomes."""
    _require(library in LIBRARIES, "unknown library {}".format(library))
    frame = rows.copy()
    frame["_cache_index"] = np.arange(len(frame), dtype=int)
    primary, retention = matched.make_primary_pool(frame, library)
    plans, fold_membership = matched.canonical_fold_plans(primary, library)
    retention = retention.sort_values("pair_uid").reset_index(drop=True)
    contract = matched.PRIMARY_CONTRACT[library]
    _require(len(retention) == contract["retention_rows"], "retention row count changed")
    _require(
        cached_eval.membership_sha256(retention) == contract["retention_membership_sha256"],
        "retention membership changed",
    )
    return {
        "primary": primary,
        "retention": retention,
        "plans": plans,
        "fold_membership": fold_membership,
    }


def validate_retention_outcomes(retention, library):
    """Parse direct outcomes only after weak-label model selection is frozen."""
    retention = retention.copy()
    retention["target_retention"] = pd.to_numeric(
        retention["target_retention"], errors="raise"
    ).astype(float)
    retention["target_binder"] = pd.to_numeric(
        retention["target_binder"], errors="raise"
    ).astype(int)
    expected = retention["target_retention"].ge(RETENTION_BINDER_THRESHOLD).astype(int)
    _require(bool(retention["target_binder"].eq(expected).all()), "retention binder threshold changed")
    contract = matched.PRIMARY_CONTRACT[library]
    _require(
        int(retention["target_binder"].sum()) == contract["retention_positive"],
        "retention positive count changed",
    )
    return retention


def _fit_fold_matrices(features, plans):
    prepared = []
    for plan in plans:
        train = np.asarray(plan["train_keys"], dtype=int)
        validation = np.asarray(plan["validation_keys"], dtype=int)
        mean, scale = matched.fit_standardizer(features[train])
        prepared.append(
            {
                "plan": plan,
                "x_train": matched.standardize(features[train], mean, scale),
                "x_validation": matched.standardize(features[validation], mean, scale),
            }
        )
    return prepared


def _fit_one_fold_candidate(item, labels, c_value):
    plan = item["plan"]
    train = np.asarray(plan["train_keys"], dtype=int)
    validation = np.asarray(plan["validation_keys"], dtype=int)
    classifier = matched.fit_logistic(item["x_train"], labels[train], c_value)
    probability = classifier.predict_proba(item["x_validation"])[:, 1]
    metrics = matched.binary_metrics(labels[validation], probability)
    return {
        "C": float(c_value),
        "fold": int(plan["fold"]),
        "plan": plan,
        "train": train,
        "validation": validation,
        "probability": probability,
        "metrics": metrics,
    }


def cross_validate_layer(features, primary, plans, layer, c_grid, jobs=1):
    """Evaluate one layer/C grid using only canonical weak validation rows."""
    values = np.asarray(features, dtype=np.float32)
    labels = primary["weak_label"].to_numpy(dtype=int)
    _require(values.shape == (len(primary), FEATURE_DIMENSION), "weak feature shape mismatch")
    _require(int(jobs) >= 1, "jobs must be positive")
    prepared = _fit_fold_matrices(values, plans)
    rows = []
    aggregate_rows = []
    candidates = [
        (float(c_value), item)
        for c_value in tuple(float(value) for value in c_grid)
        for item in prepared
    ]
    if int(jobs) == 1:
        fitted = [
            _fit_one_fold_candidate(item, labels, c_value)
            for c_value, item in candidates
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(int(jobs), len(candidates))
        ) as executor:
            futures = [
                executor.submit(_fit_one_fold_candidate, item, labels, c_value)
                for c_value, item in candidates
            ]
            fitted = [future.result() for future in futures]
    fitted = sorted(fitted, key=lambda item: (item["C"], item["fold"]))
    for c_value in tuple(float(value) for value in c_grid):
        pooled_label = []
        pooled_probability = []
        for result in fitted:
            if float(result["C"]) != float(c_value):
                continue
            plan = result["plan"]
            train = result["train"]
            validation = result["validation"]
            probability = result["probability"]
            metrics = result["metrics"]
            rows.append(
                dict(
                    {
                        "layer": int(layer),
                        "representation": layer_feature_name(layer),
                        "record_type": "fold",
                        "C": float(c_value),
                        "fold": int(plan["fold"]),
                        "n_train": int(len(train)),
                        "n_guard": int(plan["n_guard"]),
                        "train_membership_sha256": plan["train_membership_sha256"],
                        "validation_membership_sha256": plan[
                            "validation_membership_sha256"
                        ],
                    },
                    **metrics
                )
            )
            pooled_label.extend(labels[validation].tolist())
            pooled_probability.extend(probability.tolist())
        metrics = matched.binary_metrics(pooled_label, pooled_probability)
        aggregate = dict(
            {
                "layer": int(layer),
                "representation": layer_feature_name(layer),
                "record_type": "aggregate",
                "C": float(c_value),
                "fold": -1,
                "n_train": int(sum(len(plan["train_keys"]) for plan in plans)),
                "n_guard": int(sum(plan["n_guard"] for plan in plans)),
                "train_membership_sha256": "multiple_folds",
                "validation_membership_sha256": "multiple_folds",
            },
            **metrics
        )
        rows.append(aggregate)
        aggregate_rows.append(aggregate)
    best_c = min(aggregate_rows, key=lambda row: (row["log_loss"], row["C"]))
    for row in rows:
        row["selected_within_layer"] = int(float(row["C"]) == float(best_c["C"]))
        row["selected_joint"] = 0
    return rows


def select_joint_configuration(validation_rows):
    """Select exactly one aggregate (layer, C) by weak log loss alone."""
    aggregates = [row for row in validation_rows if row["record_type"] == "aggregate"]
    _require(aggregates, "no aggregate validation records")
    best = min(
        aggregates,
        key=lambda row: (float(row["log_loss"]), int(row["layer"]), float(row["C"])),
    )
    for row in validation_rows:
        row["selected_joint"] = int(
            row["record_type"] == "aggregate"
            and int(row["layer"]) == int(best["layer"])
            and float(row["C"]) == float(best["C"])
        )
    return {
        "layer": int(best["layer"]),
        "C": float(best["C"]),
        "weak_validation_log_loss": float(best["log_loss"]),
        "weak_validation_auroc_descriptive_only": float(best["auroc"]),
        "weak_validation_ap_descriptive_only": float(best["ap"]),
        "selection_metric": "pooled weak-validation log loss only",
    }


def selected_c_for_layer(validation_rows, layer):
    candidates = [
        row
        for row in validation_rows
        if row["record_type"] == "aggregate" and int(row["layer"]) == int(layer)
    ]
    _require(candidates, "no validation rows for layer {}".format(layer))
    best = min(candidates, key=lambda row: (float(row["log_loss"]), float(row["C"])))
    return float(best["C"])


def refit_and_predict(weak_features, retention_features, primary, c_value):
    """Refit one frozen readout on all weak rows and score retention rows."""
    weak = np.asarray(weak_features, dtype=np.float32)
    retention = np.asarray(retention_features, dtype=np.float32)
    labels = primary["weak_label"].to_numpy(dtype=int)
    mean, scale = matched.fit_standardizer(weak)
    x_weak = matched.standardize(weak, mean, scale)
    x_retention = matched.standardize(retention, mean, scale)
    classifier = matched.fit_logistic(x_weak, labels, float(c_value))
    probability = classifier.predict_proba(x_retention)[:, 1]
    return probability, {
        "C": float(c_value),
        "trainable_parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
        "nonzero_coefficients": int(np.count_nonzero(classifier.coef_)),
        "iterations": int(classifier.n_iter_[0]),
        "training_membership_sha256": cached_eval.membership_sha256(primary),
        "feature_mean_sha256": sha256_array(np.asarray(mean, dtype=np.float64)),
        "feature_scale_sha256": sha256_array(np.asarray(scale, dtype=np.float64)),
        "coefficient_sha256": sha256_array(np.asarray(classifier.coef_, dtype=np.float64)),
        "intercept_sha256": sha256_array(np.asarray(classifier.intercept_, dtype=np.float64)),
    }


def best_retrospective_f1(labels, scores):
    """Choose the best ``score >= threshold`` rule on these same labels.

    Ties prefer higher precision, then higher recall, then a higher threshold.
    This function is intentionally retrospective and its output must never be
    described as a validated or calibrated threshold.
    """
    observed = np.asarray(labels, dtype=int)
    probability = np.asarray(scores, dtype=float)
    _require(len(observed) == len(probability) and len(observed) > 0, "threshold length mismatch")
    _require(set(observed.tolist()) == {0, 1}, "threshold selection needs both classes")
    _require(bool(np.isfinite(probability).all()), "non-finite threshold score")
    candidates = []
    total_positive = int(observed.sum())
    total_negative = int(len(observed) - total_positive)
    for threshold in np.unique(probability):
        predicted = probability >= float(threshold)
        tp = int(np.sum(predicted & (observed == 1)))
        fp = int(np.sum(predicted & (observed == 0)))
        fn = total_positive - tp
        tn = total_negative - fp
        precision = float(tp / float(tp + fp)) if tp + fp else 0.0
        recall = float(tp / float(tp + fn)) if tp + fn else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "selected_pairs": int(tp + fp),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    best = max(
        candidates,
        key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"]),
    )
    best.update(
        {
            "threshold_selection_source": "same direct-retention evaluation rows",
            "threshold_is_evaluation_selected": True,
            "threshold_is_calibrated_probability": False,
            "threshold_is_prospectively_validated": False,
        }
    )
    return best


def retention_metric_record(retention, probability, library, model_name, layer, c_value):
    ranking = cached_eval.ranking_metrics(retention.reset_index(drop=True), probability)
    threshold = best_retrospective_f1(retention["target_binder"], probability)
    return dict(
        {
            "library": library,
            "model": model_name,
            "layer": int(layer),
            "C": float(c_value),
            "auroc": float(ranking["global_auroc"]),
            "average_precision": float(ranking["global_auprc"]),
            "within_peptide_spearman": float(ranking["within_peptide_macro_spearman"]),
            "within_peptide_spearman_groups": int(
                ranking["within_peptide_evaluable_spearman_groups"]
            ),
            "retrospective_f1_threshold": float(threshold["threshold"]),
            "pairs_scoring_above_threshold": int(threshold["selected_pairs"]),
            "threshold_precision": float(threshold["precision"]),
            "threshold_recall": float(threshold["recall"]),
            "threshold_f1": float(threshold["f1"]),
            "threshold_true_positive": int(threshold["true_positive"]),
            "threshold_false_positive": int(threshold["false_positive"]),
            "threshold_false_negative": int(threshold["false_negative"]),
            "threshold_true_negative": int(threshold["true_negative"]),
            "threshold_selection_source": threshold["threshold_selection_source"],
            "threshold_is_evaluation_selected": int(
                threshold["threshold_is_evaluation_selected"]
            ),
            "threshold_is_calibrated_probability": int(
                threshold["threshold_is_calibrated_probability"]
            ),
            "threshold_is_prospectively_validated": int(
                threshold["threshold_is_prospectively_validated"]
            ),
        },
        n=int(ranking["n"]),
        positive=int(ranking["positive"]),
    )


def prediction_frame(retention, probability, metrics, model_name, layer, c_value):
    columns = [
        "pair_uid",
        "library",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
    ]
    frame = retention[columns].copy()
    frame["model"] = model_name
    frame["layer"] = int(layer)
    frame["C"] = float(c_value)
    frame["binder_score"] = np.asarray(probability, dtype=float)
    frame["retrospective_f1_threshold"] = float(metrics["retrospective_f1_threshold"])
    frame["scores_above_retrospective_threshold"] = frame["binder_score"].ge(
        float(metrics["retrospective_f1_threshold"])
    ).astype(int)
    frame["threshold_is_evaluation_selected"] = 1
    return frame


def compare_reference_predictions(reference_run, retention, probability, expected_c):
    """Compare against historical scores made with the legacy retention cache.

    The historical primary evaluator trained from ``mint_weak_cache_v1`` but
    scored the direct-retention rows from the older ``mint_features_v1.npz``.
    The two feature archives have an accepted small numerical difference.
    Consequently this comparison is descriptive; exact parity is checked
    separately by refitting/scoring both sides from the canonical merged cache.
    """
    reference_run = Path(reference_run)
    prediction_path = reference_run / "predictions.csv"
    _require(prediction_path.is_file(), "missing reference predictions {}".format(prediction_path))
    reference = pd.read_csv(
        prediction_path,
        usecols=["pair_uid", "representation", "selected_C", "binder_probability"],
    )
    reference = reference.loc[reference["representation"].eq(matched.REPRESENTATION)].copy()
    _require(len(reference) == len(retention), "reference prediction row count mismatch")
    _require(reference["selected_C"].nunique() == 1, "reference has multiple selected C values")
    _require(float(reference["selected_C"].iloc[0]) == float(expected_c), "layer-33 selected C changed")
    current = pd.DataFrame(
        {"pair_uid": retention["pair_uid"].astype(str), "new_probability": probability}
    )
    joined = reference[["pair_uid", "binder_probability"]].merge(
        current, on="pair_uid", how="inner", validate="one_to_one"
    )
    _require(len(joined) == len(retention), "reference prediction identity mismatch")
    difference = joined["new_probability"].to_numpy(float) - joined[
        "binder_probability"
    ].to_numpy(float)
    return {
        "rows": int(len(joined)),
        "selected_C": float(expected_c),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference * difference))),
        "reference_predictions_sha256": sha256_file(prediction_path),
        "comparison_semantics": (
            "descriptive comparison to historical probabilities scored from "
            "the separate legacy retention-feature archive"
        ),
    }


def probability_difference(new_probability, canonical_probability):
    """Compare predictions generated from matched new/canonical cache rows."""
    new = np.asarray(new_probability, dtype=float)
    canonical = np.asarray(canonical_probability, dtype=float)
    _require(new.shape == canonical.shape, "prediction parity shape mismatch")
    difference = new - canonical
    return {
        "rows": int(len(new)),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference * difference))),
        "new_probability_sha256": sha256_array(new),
        "canonical_probability_sha256": sha256_array(canonical),
        "bitwise_identical": bool(np.array_equal(new, canonical)),
        "comparison_semantics": (
            "new layer-33 features versus canonical merged-cache layer-33 features, "
            "with the same weak rows, standardization, C, and retention rows"
        ),
    }


def _source_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def run(args):
    started = time.time()
    libraries = tuple(args.libraries)
    _require(libraries and set(libraries).issubset(LIBRARIES), "bad libraries")
    _require(len(libraries) == len(set(libraries)), "duplicate libraries")
    _require(int(args.folds) == CANONICAL_FOLDS, "this evaluator requires three folds")
    _require(int(args.split_seed) == CANONICAL_SPLIT_SEED, "this evaluator requires split seed 17")
    c_grid = tuple(float(value) for value in args.c_grid)
    _require(c_grid == CANONICAL_C_GRID, "this evaluator requires the canonical C grid")
    _require(int(args.jobs) >= 1, "jobs must be positive")
    _require(args.max_layer33_feature_difference >= 0.0, "negative feature tolerance")
    _require(args.max_layer33_prediction_difference >= 0.0, "negative prediction tolerance")

    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    feature_archive = Path(args.features)
    feature_manifest = Path(args.feature_manifest) if args.feature_manifest else feature_archive.with_name("manifest.json")
    rows_path = Path(args.rows)
    rows_manifest_path = Path(args.rows_manifest)
    canonical_cache_path = Path(args.canonical_cache)
    for path in (feature_archive, feature_manifest, rows_path, rows_manifest_path, canonical_cache_path):
        _require(path.is_file(), "missing input {}".format(path))

    rows, rows_manifest = matched.read_cache_rows(rows_path, rows_manifest_path)
    available_layers, archive_header = load_archive_header(feature_archive)
    requested_layers = tuple(int(value) for value in args.layers) if args.layers else available_layers
    _require(requested_layers and len(requested_layers) == len(set(requested_layers)), "bad layer list")
    _require(set(requested_layers).issubset(available_layers), "requested layer absent from archive")
    _require(33 in requested_layers, "layer 33 is required as the canonical control")

    canonical_header, canonical_features = load_canonical_cache(canonical_cache_path)
    validate_row_alignment(rows, archive_header, canonical_header)
    new_layer33 = load_layer_matrix(feature_archive, 33)
    _require(len(new_layer33) == len(rows), "layer-33 row count differs from row table")
    parity = feature_difference(new_layer33, canonical_features)
    parity["tolerance"] = float(args.max_layer33_feature_difference)
    parity["passed"] = bool(
        parity["max_abs_difference"] <= float(args.max_layer33_feature_difference)
    )
    _require(parity["passed"], "new layer-33 features do not reproduce canonical cache")
    prepared = {library: prepare_library(rows, library) for library in libraries}
    validation_by_library = {library: [] for library in libraries}
    layer33_final = {}
    prediction_parity = {}
    historical_prediction_difference = {}

    # Layer 33 is deliberately processed first and parity-checked before any
    # intermediate layer is screened.
    for library in libraries:
        data = prepared[library]
        weak_features = new_layer33[data["primary"]["_cache_index"].to_numpy(dtype=int)]
        retention_features = new_layer33[
            data["retention"]["_cache_index"].to_numpy(dtype=int)
        ]
        rows33 = cross_validate_layer(
            weak_features, data["primary"], data["plans"], 33, c_grid, jobs=args.jobs
        )
        validation_by_library[library].extend(rows33)
        c33 = selected_c_for_layer(rows33, 33)
        probability, model_audit = refit_and_predict(
            weak_features, retention_features, data["primary"], c33
        )
        canonical_weak_features = canonical_features[
            data["primary"]["_cache_index"].to_numpy(dtype=int)
        ]
        canonical_retention_features = canonical_features[
            data["retention"]["_cache_index"].to_numpy(dtype=int)
        ]
        canonical_probability, _ = refit_and_predict(
            canonical_weak_features,
            canonical_retention_features,
            data["primary"],
            c33,
        )
        comparison = probability_difference(probability, canonical_probability)
        comparison["selected_C"] = float(c33)
        comparison["tolerance"] = float(args.max_layer33_prediction_difference)
        comparison["tolerance_reason"] = PREDICTION_TOLERANCE_REASON
        comparison["passed"] = bool(
            comparison["max_abs_difference"]
            <= float(args.max_layer33_prediction_difference)
        )
        _require(
            comparison["passed"],
            "new layer-33 {} predictions do not reproduce canonical merged cache".format(
                library
            ),
        )
        prediction_parity[library] = comparison
        reference_run = Path(
            args.reference_liba if library == "LibA" else args.reference_libb
        )
        historical = compare_reference_predictions(
            reference_run, data["retention"], probability, c33
        )
        historical["expected_nonzero_reason"] = (
            "historical direct-retention probabilities used mint_features_v1.npz; "
            "this run uses the aligned merged multilayer cache"
        )
        historical_prediction_difference[library] = historical
        layer33_final[library] = {
            "probability": probability,
            "C": c33,
            "model_audit": model_audit,
        }
    del canonical_features
    del new_layer33

    for layer in requested_layers:
        if int(layer) == 33:
            continue
        layer_matrix = load_layer_matrix(feature_archive, layer)
        _require(
            len(layer_matrix) == len(rows),
            "layer {} row count differs from row table".format(layer),
        )
        for library in libraries:
            data = prepared[library]
            weak_features = layer_matrix[
                data["primary"]["_cache_index"].to_numpy(dtype=int)
            ]
            validation_by_library[library].extend(
                cross_validate_layer(
                    weak_features,
                    data["primary"],
                    data["plans"],
                    layer,
                    c_grid,
                    jobs=args.jobs,
                )
            )
        del layer_matrix

    selections = {}
    for library in libraries:
        selection = select_joint_configuration(validation_by_library[library])
        selection.update(
            {
                "library": library,
                "folds": CANONICAL_FOLDS,
                "split_seed": CANONICAL_SPLIT_SEED,
                "class_weight": "balanced",
                "per_fold_standardization": True,
                "selection_data": "weak PN labels only",
                "direct_retention_used_for_selection": False,
                "selection_frozen_before_retention_evaluation": True,
            }
        )
        selections[library] = selection

    output_dir.mkdir(parents=True, mode=0o700)
    selected_config_path = output_dir / "selected_config.json"
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "selection_rule": "minimum pooled weak-validation log loss; deterministic layer/C tie-break only",
            "retention_labels_used": False,
            "selections": selections,
        },
        selected_config_path,
    )

    # This is the first point at which numerical retention outcomes or binder
    # labels are parsed.  Layer/C selection above is therefore outcome-blind.
    for library in libraries:
        prepared[library]["retention"] = validate_retention_outcomes(
            prepared[library]["retention"], library
        )

    metric_rows = []
    prediction_blocks = []
    model_audits = {}
    for library in libraries:
        data = prepared[library]
        selection = selections[library]
        selected_layer = int(selection["layer"])
        selected_c = float(selection["C"])
        if selected_layer == 33 and selected_c == float(layer33_final[library]["C"]):
            selected_probability = layer33_final[library]["probability"].copy()
            selected_audit = dict(layer33_final[library]["model_audit"])
        else:
            selected_matrix = load_layer_matrix(feature_archive, selected_layer)
            _require(
                len(selected_matrix) == len(rows),
                "selected layer row count differs from row table",
            )
            weak_features = selected_matrix[
                data["primary"]["_cache_index"].to_numpy(dtype=int)
            ]
            retention_features = selected_matrix[
                data["retention"]["_cache_index"].to_numpy(dtype=int)
            ]
            selected_probability, selected_audit = refit_and_predict(
                weak_features, retention_features, data["primary"], selected_c
            )
            del selected_matrix

        model_specs = (
            (
                "weak_selected_frozen_mint_layer",
                selected_layer,
                selected_c,
                selected_probability,
                selected_audit,
            ),
            (
                "canonical_frozen_mint_layer33_control",
                33,
                float(layer33_final[library]["C"]),
                layer33_final[library]["probability"],
                layer33_final[library]["model_audit"],
            ),
        )
        model_audits[library] = {}
        for model_name, layer, c_value, probability, audit in model_specs:
            metric = retention_metric_record(
                data["retention"], probability, library, model_name, layer, c_value
            )
            metric_rows.append(metric)
            prediction_blocks.append(
                prediction_frame(
                    data["retention"], probability, metric, model_name, layer, c_value
                )
            )
            model_audits[library][model_name] = audit

    weak_rows = []
    fold_blocks = []
    for library in libraries:
        for row in validation_by_library[library]:
            weak_rows.append(dict({"library": library}, **row))
        block = prepared[library]["fold_membership"].copy()
        block.insert(0, "library", library)
        fold_blocks.append(block)

    weak_validation_path = output_dir / "weak_validation_by_layer.csv"
    fold_membership_path = output_dir / "fold_membership.csv"
    predictions_path = output_dir / "retention_predictions.csv"
    metrics_path = output_dir / "retention_metrics.csv"
    parity_path = output_dir / "canonical_parity.json"
    model_audit_path = output_dir / "model_audit.json"
    _write_csv(pd.DataFrame(weak_rows), weak_validation_path)
    _write_csv(pd.concat(fold_blocks, ignore_index=True), fold_membership_path)
    _write_csv(pd.concat(prediction_blocks, ignore_index=True), predictions_path)
    _write_csv(pd.DataFrame(metric_rows), metrics_path)
    _write_json(
        {
            "layer33_features": parity,
            "layer33_predictions_against_canonical_merged_cache": prediction_parity,
            "historical_predictions_using_legacy_retention_cache": historical_prediction_difference,
            "all_checks_passed": True,
        },
        parity_path,
    )
    _write_json(model_audits, model_audit_path)

    source_paths = {
        "script": Path(__file__).resolve(),
        "multilayer_features": feature_archive,
        "multilayer_features_manifest": feature_manifest,
        "cache_rows": rows_path,
        "cache_rows_manifest": rows_manifest_path,
        "canonical_layer33_cache": canonical_cache_path,
        "canonical_liba_predictions": Path(args.reference_liba) / "predictions.csv",
        "canonical_libb_predictions": Path(args.reference_libb) / "predictions.csv",
        "matched_pipeline": Path(matched.__file__).resolve(),
        "cached_evaluator": Path(cached_eval.__file__).resolve(),
    }
    source_records = {}
    for name, path in source_paths.items():
        if not path.is_file():
            continue
        if name == "canonical_liba_predictions" and "LibA" not in libraries:
            continue
        if name == "canonical_libb_predictions" and "LibB" not in libraries:
            continue
        source_records[name] = _source_record(path)
    output_paths = (
        selected_config_path,
        weak_validation_path,
        fold_membership_path,
        predictions_path,
        metrics_path,
        parity_path,
        model_audit_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "command": _shell_join(sys.argv),
        "configuration": {
            "libraries": list(libraries),
            "layers": list(requested_layers),
            "c_grid": list(c_grid),
            "folds": CANONICAL_FOLDS,
            "split_seed": CANONICAL_SPLIT_SEED,
            "parallel_logistic_jobs": int(args.jobs),
            "class_weight": "balanced",
            "standardization": "per-fold training-only; full weak pool for final refit",
            "retention_binder_threshold": RETENTION_BINDER_THRESHOLD,
            "layer_C_selection": "pooled weak-validation log loss only",
            "operating_threshold_selection": "retrospective maximum F1 on the same retention evaluation rows",
            "operating_threshold_is_prospectively_validated": False,
            "layer33_prediction_parity_tolerance_reason": PREDICTION_TOLERANCE_REASON,
        },
        "primary_contract": {library: matched.PRIMARY_CONTRACT[library] for library in libraries},
        "selection": selections,
        "canonical_parity": {
            "features_passed": parity["passed"],
            "predictions_passed": {
                library: prediction_parity[library]["passed"] for library in libraries
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": source_records,
        "input_manifests": {
            "row_manifest_schema": rows_manifest.get("schema_version"),
            "feature_manifest_schema": _load_json(feature_manifest).get("schema_version"),
        },
        "outputs": {
            path.name: _source_record(path) for path in output_paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    print("wrote {}".format(output_dir))
    print(pd.DataFrame(metric_rows).to_string(index=False))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="merged multilayer NPZ")
    parser.add_argument("--feature-manifest", default=None)
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--rows-manifest", default=str(DEFAULT_ROWS_MANIFEST))
    parser.add_argument("--canonical-cache", default=str(DEFAULT_CANONICAL_CACHE))
    parser.add_argument("--reference-liba", default=str(DEFAULT_REFERENCE_RUNS["LibA"]))
    parser.add_argument("--reference-libb", default=str(DEFAULT_REFERENCE_RUNS["LibB"]))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES))
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(CANONICAL_C_GRID))
    parser.add_argument("--folds", type=int, default=CANONICAL_FOLDS)
    parser.add_argument("--split-seed", type=int, default=CANONICAL_SPLIT_SEED)
    parser.add_argument("--jobs", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument(
        "--max-layer33-feature-difference",
        type=float,
        default=DEFAULT_LAYER33_FEATURE_TOLERANCE,
    )
    parser.add_argument(
        "--max-layer33-prediction-difference",
        type=float,
        default=DEFAULT_LAYER33_PREDICTION_TOLERANCE,
        help=(
            "maximum new-vs-canonical merged-cache probability delta; default "
            "0.005 is justified and recorded from the layer-33 numerical parity audit"
        ),
    )
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
