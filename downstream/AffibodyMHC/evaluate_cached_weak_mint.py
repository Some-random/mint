#!/usr/bin/env python
"""Evaluate frozen MINT and site-only heads on identical weak-label splits.

This evaluator consumes the immutable frozen-MINT weak/retention cache built by
``build_mint_weak_feature_cache.py``.  It fits only low-capacity L2-logistic
heads and a matched position-additive control.  Every data-cleaning, balancing,
and generalization condition is constructed once and shared by the two
representations.

The measured retention matrix is never used to choose a cleaning rule, class
balance, regularization strength, or model.  It is used only (a) to define the
partner identities held out by a requested transfer regime and (b) for the
final retrospective evaluation.  Regularization is chosen on weak-label
validation folds having the same identity-coldness contract as the final
retention transfer.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "cached-weak-mint-evaluation-v1"
FEATURE_NAME = "mint_chain_mean"
LIBRARIES = ("LibA", "LibB")
REGIMES = ("pair_only", "peptide_cold", "affibody_cold", "double_cold")
CLEANINGS = ("c0", "count_both_ge3", "promiscuity_ge10")
BALANCES = (
    "all_unweighted",
    "all_class_weighted",
    "balanced_downsample",
)
REPRESENTATIONS = ("site", "frozen_mint_chain_mean")
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_DOWNSAMPLE_SEEDS = (0, 1, 2)

REGIME_ALIASES = {
    "pair": "pair_only",
    "random_pair": "pair_only",
    "pair_only": "pair_only",
    "peptide": "peptide_cold",
    "peptide_cold": "peptide_cold",
    "affibody": "affibody_cold",
    "affibody_cold": "affibody_cold",
    "double": "double_cold",
    "double_cold": "double_cold",
}
CLEANING_ALIASES = {
    "none": "c0",
    "c0": "c0",
    "count3_both": "count_both_ge3",
    "count_both_ge3": "count_both_ge3",
    "promiscuous_affibody": "promiscuity_ge10",
    "promiscuity_ge10": "promiscuity_ge10",
}
BALANCE_ALIASES = {
    "natural": "all_unweighted",
    "all_unweighted": "all_unweighted",
    "class_weighted": "all_class_weighted",
    "all_class_weighted": "all_class_weighted",
    "downsample": "balanced_downsample",
    "balanced_downsample": "balanced_downsample",
}

CACHE_REQUIRED_KEYS = {
    "row_index",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "weak_label",
    "r009_count",
    "r010_count",
    "measurement_missing",
    "target_retention",
    "target_binder",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
    FEATURE_NAME,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical_regime(value):
    _require(value in REGIME_ALIASES, "unknown generalization regime {}".format(value))
    return REGIME_ALIASES[value]


def _canonical_cleaning(value):
    _require(value in CLEANING_ALIASES, "unknown cleaning {}".format(value))
    return CLEANING_ALIASES[value]


def _canonical_balance(value):
    _require(value in BALANCE_ALIASES, "unknown balance {}".format(value))
    return BALANCE_ALIASES[value]


def _identity_columns(frame):
    """Return global sequence identity columns, with test-fixture aliases."""
    peptide = "peptide_identity" if "peptide_identity" in frame else "chain1_sha256"
    affibody = "affibody_identity" if "affibody_identity" in frame else "chain2_sha256"
    if "pair_identity" in frame:
        pair = "pair_identity"
    elif "sequence_pair_sha256" in frame:
        pair = "sequence_pair_sha256"
    else:
        pair = "pair_uid"
    missing = {peptide, affibody, pair}.difference(frame.columns)
    _require(not missing, "missing global identity columns {}".format(sorted(missing)))
    return peptide, affibody, pair


def _pair_identity_column(frame):
    if "pair_identity" in frame:
        return "pair_identity"
    if "sequence_pair_sha256" in frame:
        return "sequence_pair_sha256"
    _require("pair_uid" in frame, "missing pair identity column")
    return "pair_uid"


def _partner_identity_columns(frame):
    peptide = "peptide_identity" if "peptide_identity" in frame else "chain1_sha256"
    affibody = "affibody_identity" if "affibody_identity" in frame else "chain2_sha256"
    missing = {peptide, affibody}.difference(frame.columns)
    _require(not missing, "missing global partner identity columns {}".format(sorted(missing)))
    return peptide, affibody


def _stable_bin(value, axis, seed, n_folds):
    payload = "{}|{}|{}".format(axis, int(seed), value).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(n_folds)


def _stable_rank(value, salt):
    payload = "{}|{}".format(salt, value).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def membership_sha256(frame):
    """Order-invariant digest of the exact pair membership."""
    pair_column = _pair_identity_column(frame)
    values = sorted(frame[pair_column].astype(str).tolist())
    return hashlib.sha256("\n".join(values).encode("ascii")).hexdigest()


def regime_split_indices(frame, regime, fold, n_folds, seed=17):
    """Construct a weak-label fold with the requested identity-cold contract.

    ``guard`` rows share exactly one held identity for double-cold validation;
    for one-partner-cold validation they are rows that do not satisfy either the
    train or validation role.  Pair-only validation is a deterministic pair
    split and later compatibility filtering ensures both partners are seen.
    """
    regime = _canonical_regime(regime)
    _require(int(n_folds) >= 2, "n_folds must be at least two")
    _require(0 <= int(fold) < int(n_folds), "fold is out of range")
    peptide_column, affibody_column, pair_column = _identity_columns(frame)
    peptide_block = np.asarray(
        [_stable_bin(value, "peptide", seed, n_folds) for value in frame[peptide_column]],
        dtype=int,
    )
    affibody_block = np.asarray(
        [_stable_bin(value, "affibody", seed, n_folds) for value in frame[affibody_column]],
        dtype=int,
    )
    pair_block = np.asarray(
        [_stable_bin(value, "pair", seed, n_folds) for value in frame[pair_column]],
        dtype=int,
    )
    if regime == "pair_only":
        validation = pair_block == int(fold)
        train = ~validation
    elif regime == "peptide_cold":
        validation = peptide_block == int(fold)
        train = ~validation
    elif regime == "affibody_cold":
        validation = affibody_block == int(fold)
        train = ~validation
    else:
        peptide_held = peptide_block == int(fold)
        affibody_held = affibody_block == int(fold)
        validation = peptide_held & affibody_held
        train = ~peptide_held & ~affibody_held
    guard = ~(train | validation)
    output = {
        "train": np.flatnonzero(train),
        "guard": np.flatnonzero(guard),
        "validation": np.flatnonzero(validation),
    }
    _require(
        sum(len(indices) for indices in output.values()) == len(frame),
        "fold roles do not partition rows",
    )
    return output


def apply_static_cleaning(frame, cleaning):
    """Apply cleaning that can be decided independently for every row.

    Count cleaning applies to positive labels only.  Selection negatives are
    defined by absence from later positive rounds and would otherwise all be
    removed by an R009/R010 count requirement.
    """
    cleaning = _canonical_cleaning(cleaning)
    output = frame.copy()
    if cleaning == "count_both_ge3":
        label = pd.to_numeric(output["weak_label"], errors="raise").astype(int)
        r009 = pd.to_numeric(output["r009_count"], errors="raise")
        r010 = pd.to_numeric(output["r010_count"], errors="raise")
        keep = label.eq(0) | (r009.ge(3) & r010.ge(3))
        output = output.loc[keep].copy()
    return output.reset_index(drop=True)


def apply_train_only_cleaning(frame, cleaning, threshold=10):
    """Remove train-fold promiscuous Affibodies without inspecting evaluation.

    Breadth is the number of distinct *positive* peptide identities paired with
    an Affibody in the supplied training frame.  All rows for an Affibody at or
    above the threshold are removed.  Callers must pass training rows only.
    """
    cleaning = _canonical_cleaning(cleaning)
    output = frame.copy()
    if cleaning != "promiscuity_ge10":
        return output.reset_index(drop=True)
    _require(int(threshold) >= 1, "promiscuity threshold must be positive")
    peptide_column, affibody_column, _ = _identity_columns(output)
    label = pd.to_numeric(output["weak_label"], errors="raise").astype(int)
    positive = output.loc[label.eq(1)]
    breadth = positive.groupby(affibody_column)[peptide_column].nunique()
    flagged = set(breadth.loc[breadth.ge(int(threshold))].index.astype(str))
    keep = ~output[affibody_column].astype(str).isin(flagged)
    return output.loc[keep].reset_index(drop=True)


def balance_training_indices(frame, balance, seed=0):
    """Return exact estimator rows for a balancing condition."""
    balance = _canonical_balance(balance)
    label = pd.to_numeric(frame["weak_label"], errors="raise").astype(int).to_numpy()
    _require(set(label) == {0, 1}, "training data must contain both classes")
    if balance != "balanced_downsample":
        return np.arange(len(frame), dtype=int)
    pair_column = _pair_identity_column(frame)
    per_class = min(int((label == 0).sum()), int((label == 1).sum()))
    selected = []
    for class_value in (0, 1):
        candidates = np.flatnonzero(label == class_value)
        ranked = sorted(
            candidates.tolist(),
            key=lambda index: _stable_rank(
                str(frame.iloc[index][pair_column]),
                "downsample|{}|{}".format(int(seed), class_value),
            ),
        )
        selected.extend(ranked[:per_class])
    return np.asarray(sorted(selected), dtype=int)


def retention_compatible_mask(retention, actual_train, regime):
    """Identify retention rows satisfying a regime against actual fit rows."""
    regime = _canonical_regime(regime)
    ret_peptide, ret_affibody, ret_pair = _identity_columns(retention)
    train_peptide, train_affibody, train_pair = _identity_columns(actual_train)
    train_peptides = set(actual_train[train_peptide].astype(str))
    train_affibodies = set(actual_train[train_affibody].astype(str))
    train_pairs = set(actual_train[train_pair].astype(str))
    pair_absent = ~retention[ret_pair].astype(str).isin(train_pairs).to_numpy()
    peptide_seen = retention[ret_peptide].astype(str).isin(train_peptides).to_numpy()
    affibody_seen = retention[ret_affibody].astype(str).isin(train_affibodies).to_numpy()
    if regime == "pair_only":
        compatible = peptide_seen & affibody_seen
    elif regime == "peptide_cold":
        compatible = ~peptide_seen & affibody_seen
    elif regime == "affibody_cold":
        compatible = peptide_seen & ~affibody_seen
    else:
        compatible = ~peptide_seen & ~affibody_seen
    return np.asarray(pair_absent & compatible, dtype=bool)


def _safe_spearman(observed, score):
    observed = np.asarray(observed, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(observed) < 2 or len(np.unique(observed)) < 2 or len(np.unique(score)) < 2:
        return float("nan")
    return float(spearmanr(observed, score)[0])


def _group_ranking_metrics(frame, score, group_column, prefix):
    labels = pd.to_numeric(frame["target_binder"], errors="raise").astype(int).to_numpy()
    retention = pd.to_numeric(frame["target_retention"], errors="raise").to_numpy(float)
    values = np.asarray(score, dtype=float)
    ap_values = []
    prevalence_values = []
    auc_values = []
    spearman_values = []
    p_at = {1: [], 3: []}
    for _, indices in frame.groupby(group_column, sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        group_y = labels[index]
        group_score = values[index]
        group_retention = retention[index]
        if set(group_y) == {0, 1}:
            ap_values.append(float(average_precision_score(group_y, group_score)))
            prevalence_values.append(float(group_y.mean()))
            auc_values.append(float(roc_auc_score(group_y, group_score)))
        rho = _safe_spearman(group_retention, group_score)
        if math.isfinite(rho):
            spearman_values.append(rho)
        order = np.argsort(-group_score, kind="mergesort")
        for k in p_at:
            selected = order[: min(k, len(order))]
            p_at[k].append(float(group_y[selected].mean()))
    mean = lambda values: float(np.mean(values)) if values else float("nan")
    macro_ap = mean(ap_values)
    macro_prevalence = mean(prevalence_values)
    return {
        "{}_evaluable_binary_groups".format(prefix): int(len(ap_values)),
        "{}_evaluable_spearman_groups".format(prefix): int(len(spearman_values)),
        "{}_macro_auroc".format(prefix): mean(auc_values),
        "{}_macro_auprc".format(prefix): macro_ap,
        "{}_macro_prevalence".format(prefix): macro_prevalence,
        "{}_macro_auprc_lift".format(prefix): (
            macro_ap - macro_prevalence
            if math.isfinite(macro_ap) and math.isfinite(macro_prevalence)
            else float("nan")
        ),
        "{}_macro_spearman".format(prefix): mean(spearman_values),
        "{}_p_at_1".format(prefix): mean(p_at[1]),
        "{}_p_at_3".format(prefix): mean(p_at[3]),
    }


def _two_way_residual(values, peptide, affibody):
    """Residualize additive partner effects by least squares.

    Direct double-centering is correct only for a complete balanced matrix.
    The LibB matrix has a missing cell and transfer-compatible panels can be
    incomplete, so fit the intercept and both categorical main effects.
    """
    observed = np.asarray(values, dtype=float)
    peptide_codes, peptide_levels = pd.factorize(np.asarray(peptide, dtype=str), sort=True)
    affibody_codes, affibody_levels = pd.factorize(np.asarray(affibody, dtype=str), sort=True)
    columns = [np.ones(len(observed), dtype=float)]
    columns.extend(
        (peptide_codes == level).astype(float)
        for level in range(1, len(peptide_levels))
    )
    columns.extend(
        (affibody_codes == level).astype(float)
        for level in range(1, len(affibody_levels))
    )
    design = np.column_stack(columns)
    coefficient = np.linalg.lstsq(design, observed, rcond=None)[0]
    return observed - design.dot(coefficient)


def ranking_metrics(retention, score):
    """Global, conditional-ranking, and interaction metrics for retention."""
    values = np.asarray(score, dtype=float)
    _require(len(values) == len(retention), "score length mismatch")
    _require(bool(np.isfinite(values).all()), "score contains non-finite values")
    peptide_column, affibody_column = _partner_identity_columns(retention)
    labels = pd.to_numeric(retention["target_binder"], errors="raise").astype(int).to_numpy()
    observed = pd.to_numeric(retention["target_retention"], errors="raise").to_numpy(float)
    _require(set(labels) == {0, 1}, "retention metrics require both classes")
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, values))
    output = {
        "n": int(len(retention)),
        "positive": int(labels.sum()),
        "global_prevalence": prevalence,
        "global_auroc": float(roc_auc_score(labels, values)),
        "global_auprc": auprc,
        "global_auprc_lift": auprc - prevalence,
        "global_spearman": _safe_spearman(observed, values),
    }
    output.update(_group_ranking_metrics(retention, values, peptide_column, "within_peptide"))
    output.update(_group_ranking_metrics(retention, values, affibody_column, "within_affibody"))
    observed_residual = _two_way_residual(
        observed,
        retention[peptide_column].astype(str),
        retention[affibody_column].astype(str),
    )
    score_residual = _two_way_residual(
        values,
        retention[peptide_column].astype(str),
        retention[affibody_column].astype(str),
    )
    output["two_way_interaction_spearman"] = _safe_spearman(
        observed_residual, score_residual
    )
    # Backward-compatible concise aliases used by unit tests/report notebooks.
    output["within_peptide_evaluable_groups"] = output[
        "within_peptide_evaluable_binary_groups"
    ]
    output["within_affibody_evaluable_groups"] = output[
        "within_affibody_evaluable_binary_groups"
    ]
    output["peptide_p_at_1"] = output["within_peptide_p_at_1"]
    output["peptide_p_at_3"] = output["within_peptide_p_at_3"]
    output["peptide_precision_at_1"] = output["within_peptide_p_at_1"]
    output["peptide_precision_at_3"] = output["within_peptide_p_at_3"]
    return output


def _binary_validation_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(labels) == {0, 1}, "validation needs both classes")
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
    }


def _read_manifest(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _load_npz_arrays(path):
    with np.load(str(path), allow_pickle=False) as archive:
        _require(CACHE_REQUIRED_KEYS.issubset(archive.files), "cache NPZ schema mismatch")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    return arrays


def _manifest_for_npz(path):
    if path.name.startswith("features-shard-"):
        return path.with_name(path.name.replace(".npz", ".manifest.json"))
    return path.parent / "manifest.json"


def _validate_npz_manifest(path, manifest_path):
    _require(manifest_path.is_file(), "missing cache manifest {}".format(manifest_path))
    manifest = _read_manifest(manifest_path)
    _require(
        manifest.get("schema_version") == "mint-weak-feature-cache-v1",
        "cache manifest schema mismatch",
    )
    expected = manifest.get("output", {}).get("sha256")
    _require(expected == sha256_file(path), "cache output hash disagrees with manifest")
    return manifest


def load_feature_cache(cache_path, cache_manifest=None):
    """Load either one merged cache NPZ or a complete set of shard NPZs."""
    cache_path = Path(cache_path)
    manifests = []
    if cache_path.is_file():
        paths = [cache_path]
    else:
        _require(cache_path.is_dir(), "cache path does not exist")
        merged = cache_path / "mint_chain_mean_features.npz"
        if merged.is_file():
            paths = [merged]
        else:
            paths = sorted(cache_path.glob("features-shard-*.npz"))
            _require(paths, "cache directory contains neither merged cache nor shards")
    blocks = []
    for index, path in enumerate(paths):
        manifest_path = (
            Path(cache_manifest)
            if cache_manifest is not None and len(paths) == 1
            else _manifest_for_npz(path)
        )
        manifests.append(_validate_npz_manifest(path, manifest_path))
        blocks.append(_load_npz_arrays(path))
    if len(manifests) > 1:
        _require(
            all(manifest.get("stage") == "extract_shard" for manifest in manifests),
            "cache shard manifest has the wrong stage",
        )
        contracts = {
            manifest.get("contract", {}).get("sha256") for manifest in manifests
        }
        _require(None not in contracts and len(contracts) == 1, "cache shards have different feature contracts")
        shard_counts = {
            int(manifest.get("sharding", {}).get("shard_count", -1))
            for manifest in manifests
        }
        _require(shard_counts == {len(manifests)}, "cache shard set is incomplete")
        shard_indices = {
            int(manifest.get("sharding", {}).get("shard_index", -1))
            for manifest in manifests
        }
        _require(
            shard_indices == set(range(len(manifests))),
            "cache shard indices are incomplete or duplicated",
        )
        rows_hashes = {
            manifest.get("input", {}).get("rows_csv", {}).get("sha256")
            for manifest in manifests
        }
        prepare_hashes = {
            manifest.get("input", {}).get("prepare_manifest", {}).get("sha256")
            for manifest in manifests
        }
        _require(
            None not in rows_hashes and len(rows_hashes) == 1,
            "cache shards use different prepared row tables",
        )
        _require(
            None not in prepare_hashes and len(prepare_hashes) == 1,
            "cache shards use different prepared manifests",
        )
    else:
        _require(
            manifests[0].get("stage") in ("merge", "extract_shard"),
            "cache manifest has the wrong stage",
        )
    keys = set(blocks[0])
    _require(all(set(block) == keys for block in blocks), "cache shards have different schemas")
    if len(blocks) == 1:
        arrays = blocks[0]
    else:
        arrays = {
            key: np.concatenate([block[key] for block in blocks], axis=0)
            for key in sorted(keys)
        }
    order = np.argsort(np.asarray(arrays["row_index"], dtype=np.int64))
    if not np.array_equal(order, np.arange(len(order), dtype=int)):
        arrays = {key: value[order] for key, value in arrays.items()}
    row_index = np.asarray(arrays["row_index"], dtype=np.int64)
    _require(len(np.unique(row_index)) == len(row_index), "duplicate cache row index")
    _require(
        np.array_equal(row_index, np.arange(len(row_index), dtype=np.int64)),
        "cache shards are incomplete or noncontiguous",
    )
    features = np.asarray(arrays.pop(FEATURE_NAME), dtype=np.float32)
    _require(features.shape == (len(row_index), 2560), "unexpected frozen MINT shape")
    _require(bool(np.isfinite(features).all()), "non-finite frozen MINT features")
    frame = pd.DataFrame(
        {
            key: np.asarray(value).astype(str)
            if key != "row_index"
            else np.asarray(value, dtype=np.int64)
            for key, value in arrays.items()
        }
    )
    _require(set(frame["source_kind"]) == {"weak", "retention"}, "cache lacks a source kind")
    _require(not bool(frame["sequence_pair_sha256"].duplicated().any()), "duplicate cache pair")
    return frame, features, manifests, paths


def load_legacy_retention_features(path):
    path = Path(path)
    _require(path.is_file(), "legacy retention feature NPZ is missing")
    manifest_path = path.with_suffix(".manifest.json")
    _require(manifest_path.is_file(), "legacy retention feature manifest is missing")
    manifest = _read_manifest(manifest_path)
    expected = manifest.get("output", {}).get("sha256")
    if isinstance(manifest.get("output"), dict) and "path" in manifest.get("output", {}):
        _require(expected == sha256_file(path), "legacy retention feature hash mismatch")
    with np.load(str(path), allow_pickle=False) as archive:
        required = {"pair_uid", "library", "measurement_missing", FEATURE_NAME}
        _require(required.issubset(archive.files), "legacy retention feature schema mismatch")
        pair_uid = np.asarray(archive["pair_uid"]).astype(str)
        library = np.asarray(archive["library"]).astype(str)
        missing = np.asarray(archive["measurement_missing"]).astype(str)
        features = np.asarray(archive[FEATURE_NAME], dtype=np.float32).copy()
    _require(features.shape == (len(pair_uid), 2560), "legacy MINT feature shape mismatch")
    table = pd.DataFrame(
        {"pair_uid": pair_uid, "legacy_library": library, "legacy_missing": missing}
    )
    table["legacy_index"] = np.arange(len(table), dtype=int)
    _require(not bool(table["pair_uid"].duplicated().any()), "duplicate legacy pair UID")
    return table, features, manifest


def _site_code_matrix(frame, library):
    spec = LIBRARY_SPECS[library]
    peptide = frame["peptide_design_code"].astype(str)
    affibody = frame["affibody_design_code"].astype(str)
    _require(bool(peptide.map(len).eq(spec["pep_length"]).all()), "bad peptide code")
    _require(bool(affibody.map(len).eq(spec["aff_length"]).all()), "bad Affibody code")
    columns = [peptide.str[index].to_numpy(str) for index in range(spec["pep_length"])]
    columns += [affibody.str[index].to_numpy(str) for index in range(spec["aff_length"])]
    return np.column_stack(columns)


def _make_site_encoder(n_positions):
    arguments = {
        "categories": [list(AA_ALPHABET) for _ in range(int(n_positions))],
        "handle_unknown": "error",
        "dtype": np.float64,
    }
    try:
        return OneHotEncoder(sparse_output=True, **arguments)
    except TypeError:  # scikit-learn <1.2, including the project environment.
        return OneHotEncoder(sparse=True, **arguments)


def _fit_standardization(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _make_logistic(c_value, balance):
    balance = _canonical_balance(balance)
    return LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced" if balance == "all_class_weighted" else None,
        random_state=0,
        max_iter=2000,
        tol=1e-6,
    )


def _fit_logistic_matrices(c_value, balance, x_train, y_train, x_prediction):
    model = _make_logistic(c_value, balance)
    model.fit(x_train, y_train)
    _require(int(model.n_iter_[0]) < model.max_iter, "logistic head did not converge")
    return model.predict_proba(x_prediction)[:, 1], model


def _fit_predict_head(
    representation,
    c_value,
    balance,
    train_indices,
    prediction_indices,
    labels,
    site_features,
    mint_features,
    direct_prediction_features=None,
):
    if representation == "site":
        x_train = site_features[train_indices]
        x_prediction = (
            direct_prediction_features
            if direct_prediction_features is not None
            else site_features[prediction_indices]
        )
    else:
        mean, scale = _fit_standardization(mint_features[train_indices])
        x_train = ((mint_features[train_indices] - mean) / scale).astype(np.float32)
        prediction_values = (
            direct_prediction_features
            if direct_prediction_features is not None
            else mint_features[prediction_indices]
        )
        x_prediction = ((prediction_values - mean) / scale).astype(np.float32)
    return _fit_logistic_matrices(
        c_value, balance, x_train, labels[train_indices], x_prediction
    )


def _filter_validation_compatibility(validation, actual_train, regime):
    mask = retention_compatible_mask(validation, actual_train, regime)
    return validation.loc[mask].copy()


def build_fold_plans(frame, regime, cleaning, balance, seed, folds, split_seed):
    plans = []
    for fold in range(int(folds)):
        split = regime_split_indices(frame, regime, fold, folds, split_seed)
        train = frame.iloc[split["train"]].copy()
        validation = frame.iloc[split["validation"]].copy()
        train = apply_train_only_cleaning(train, cleaning)
        if train.empty or set(train["weak_label"].astype(int)) != {0, 1}:
            continue
        selected = balance_training_indices(train, balance, seed)
        train = train.iloc[selected].copy()
        validation = _filter_validation_compatibility(validation, train, regime)
        if validation.empty or set(validation["weak_label"].astype(int)) != {0, 1}:
            continue
        plans.append(
            {
                "fold": int(fold),
                "train_keys": train["_condition_index"].to_numpy(dtype=int),
                "validation_keys": validation["_condition_index"].to_numpy(dtype=int),
                "n_guard": int(len(split["guard"])),
                "train_membership_sha256": membership_sha256(train),
                "validation_membership_sha256": membership_sha256(validation),
            }
        )
    _require(len(plans) >= 2, "fewer than two evaluable weak validation folds")
    return plans


def tune_c(
    representation,
    c_grid,
    balance,
    plans,
    labels,
    site_features,
    mint_features,
    metadata,
):
    rows = []
    aggregate = []
    prepared_plans = []
    for plan in plans:
        train_indices = plan["train_keys"]
        validation_indices = plan["validation_keys"]
        if representation == "site":
            x_train = site_features[train_indices]
            x_validation = site_features[validation_indices]
        else:
            mean, scale = _fit_standardization(mint_features[train_indices])
            x_train = (
                (mint_features[train_indices] - mean) / scale
            ).astype(np.float32)
            x_validation = (
                (mint_features[validation_indices] - mean) / scale
            ).astype(np.float32)
        prepared_plans.append((plan, x_train, x_validation))
    for c_value in c_grid:
        pooled_label = []
        pooled_probability = []
        for plan, x_train, x_validation in prepared_plans:
            probability, _ = _fit_logistic_matrices(
                c_value,
                balance,
                x_train,
                labels[plan["train_keys"]],
                x_validation,
            )
            observed = labels[plan["validation_keys"]]
            metrics = _binary_validation_metrics(observed, probability)
            rows.append(
                dict(
                    metadata,
                    representation=representation,
                    record_type="fold",
                    C=float(c_value),
                    fold=int(plan["fold"]),
                    n_train=int(len(plan["train_keys"])),
                    n_guard=int(plan["n_guard"]),
                    train_membership_sha256=plan["train_membership_sha256"],
                    validation_membership_sha256=plan["validation_membership_sha256"],
                    **metrics
                )
            )
            pooled_label.extend(observed.tolist())
            pooled_probability.extend(probability.tolist())
        metrics = _binary_validation_metrics(pooled_label, pooled_probability)
        aggregate.append(dict(C=float(c_value), **metrics))
        rows.append(
            dict(
                metadata,
                representation=representation,
                record_type="aggregate",
                C=float(c_value),
                fold=-1,
                n_train=int(sum(len(plan["train_keys"]) for plan in plans)),
                n_guard=int(sum(plan["n_guard"] for plan in plans)),
                train_membership_sha256="multiple_folds",
                validation_membership_sha256="multiple_folds",
                **metrics
            )
        )
    selected = min(
        aggregate,
        key=lambda row: (row["log_loss"], -row["auprc"], -row["auroc"], row["C"]),
    )
    for row in rows:
        row["selected"] = int(float(row["C"]) == float(selected["C"]))
    return float(selected["C"]), rows


def _regime_pool(weak, retention, regime):
    regime = _canonical_regime(regime)
    weak_peptide, weak_affibody, weak_pair = _identity_columns(weak)
    ret_peptide, ret_affibody, ret_pair = _identity_columns(retention)
    retention_pairs = set(retention[ret_pair].astype(str))
    keep = ~weak[weak_pair].astype(str).isin(retention_pairs)
    if regime in ("peptide_cold", "double_cold"):
        keep &= ~weak[weak_peptide].astype(str).isin(set(retention[ret_peptide].astype(str)))
    if regime in ("affibody_cold", "double_cold"):
        keep &= ~weak[weak_affibody].astype(str).isin(set(retention[ret_affibody].astype(str)))
    output = weak.loc[keep].copy().reset_index(drop=True)
    _require(not output.empty, "generalization regime produced no weak rows")
    return output


def _write_private_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _condition_seeds(balance, seeds):
    return tuple(int(value) for value in seeds) if balance == "balanced_downsample" else (-1,)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(int(args.folds) >= 2, "folds must be at least two")
    c_grid = tuple(sorted(set(float(value) for value in args.c_grid)))
    _require(c_grid and all(value > 0 and math.isfinite(value) for value in c_grid), "bad C grid")
    libraries = tuple(args.libraries)
    regimes = tuple(_canonical_regime(value) for value in args.regimes)
    cleanings = tuple(_canonical_cleaning(value) for value in args.cleanings)
    balances = tuple(_canonical_balance(value) for value in args.balances)
    representations = tuple(args.representations)
    _require(set(libraries).issubset(LIBRARIES), "unknown library")
    _require(set(representations).issubset(REPRESENTATIONS), "unknown representation")

    cache_frame, cache_features, cache_manifests, cache_paths = load_feature_cache(
        args.cache, args.cache_manifest
    )
    legacy_table, legacy_features, legacy_manifest = load_legacy_retention_features(
        args.retention_features
    )
    retention_cache = cache_frame.loc[cache_frame["source_kind"].eq("retention")].copy()
    retention_cache["_cache_index"] = retention_cache.index.to_numpy(dtype=int)
    retention = retention_cache.merge(legacy_table, on="pair_uid", validate="one_to_one")
    _require(len(retention) == len(legacy_table), "cache/legacy retention pair mismatch")
    _require(bool(retention["library"].eq(retention["legacy_library"]).all()), "retention library mismatch")
    _require(
        bool(retention["measurement_missing"].astype(str).eq(retention["legacy_missing"]).all()),
        "retention missingness mismatch",
    )
    cached_retention_features = cache_features[retention["_cache_index"].to_numpy(dtype=int)]
    ordered_legacy_features = legacy_features[retention["legacy_index"].to_numpy(dtype=int)]
    max_feature_difference = float(
        np.max(np.abs(cached_retention_features - ordered_legacy_features))
    )
    _require(
        max_feature_difference <= float(args.max_retention_feature_difference),
        "cache/legacy retention MINT features differ (max abs {})".format(max_feature_difference),
    )
    retention["measurement_missing"] = pd.to_numeric(
        retention["measurement_missing"], errors="raise"
    ).astype(int)
    measured = retention.loc[retention["measurement_missing"].eq(0)].copy()
    measured["target_retention"] = pd.to_numeric(measured["target_retention"], errors="raise")
    measured["target_binder"] = pd.to_numeric(measured["target_binder"], errors="raise").astype(int)
    measured["_legacy_feature_index"] = measured["legacy_index"].astype(int)

    weak_cache = cache_frame.loc[cache_frame["source_kind"].eq("weak")].copy()
    weak_cache["_cache_index"] = weak_cache.index.to_numpy(dtype=int)
    weak_cache["weak_label"] = pd.to_numeric(weak_cache["weak_label"], errors="raise").astype(int)
    for column in ("r009_count", "r010_count"):
        weak_cache[column] = pd.to_numeric(weak_cache[column], errors="raise").astype(np.int64)
    _require(set(weak_cache["weak_label"]) == {0, 1}, "weak cache lacks a class")

    tuning_rows = []
    metric_rows = []
    prediction_blocks = []
    condition_rows = []
    for library in libraries:
        weak_library = weak_cache.loc[weak_cache["library"].eq(library)].copy()
        retention_library = measured.loc[measured["library"].eq(library)].copy()
        _require(not weak_library.empty and not retention_library.empty, "empty library data")
        all_reference_pools = {
            regime: _regime_pool(weak_library, retention_library, regime)
            for regime in REGIMES
        }
        all_reference_masks = {
            regime: retention_compatible_mask(
                retention_library, all_reference_pools[regime], regime
            )
            for regime in REGIMES
        }
        common_mask = np.ones(len(retention_library), dtype=bool)
        for regime in REGIMES:
            common_mask &= all_reference_masks[regime]
        common_pair_identities = set(
            retention_library.loc[common_mask, "sequence_pair_sha256"].astype(str)
        )
        for regime in regimes:
            regime_pool = all_reference_pools[regime]
            for cleaning in cleanings:
                static_pool = apply_static_cleaning(regime_pool, cleaning)
                _require(set(static_pool["weak_label"]) == {0, 1}, "cleaning removed a class")
                static_pool = static_pool.reset_index(drop=True)
                static_pool["_condition_index"] = np.arange(len(static_pool), dtype=int)
                weak_mint = cache_features[static_pool["_cache_index"].to_numpy(dtype=int)]
                retention_mint = legacy_features[
                    retention_library["_legacy_feature_index"].to_numpy(dtype=int)
                ]
                combined_site = pd.concat(
                    [static_pool, retention_library], ignore_index=True, sort=False
                )
                encoder = _make_site_encoder(
                    LIBRARY_SPECS[library]["pep_length"]
                    + LIBRARY_SPECS[library]["aff_length"]
                )
                site_all = encoder.fit_transform(_site_code_matrix(combined_site, library))
                site_weak = site_all[: len(static_pool)]
                site_retention = site_all[len(static_pool) :]
                labels = static_pool["weak_label"].to_numpy(dtype=int)
                for balance in balances:
                    for balance_seed in _condition_seeds(balance, args.downsample_seeds):
                        metadata = {
                            "library": library,
                            "regime": regime,
                            "cleaning": cleaning,
                            "balance": balance,
                            "balance_seed": int(balance_seed),
                        }
                        plans = build_fold_plans(
                            static_pool,
                            regime,
                            cleaning,
                            balance,
                            balance_seed,
                            args.folds,
                            args.split_seed,
                        )
                        final_train = apply_train_only_cleaning(static_pool, cleaning)
                        final_selected = balance_training_indices(final_train, balance, balance_seed)
                        final_train = final_train.iloc[final_selected].copy()
                        final_condition_indices = final_train["_condition_index"].to_numpy(dtype=int)
                        # Keep the retention panel fixed across cleaning and
                        # balancing ablations.  The exposure contract is defined
                        # from the full C0 regime pool; changes in actual exposure
                        # after row removal are reported separately below.
                        compatible = all_reference_masks[regime]
                        actual_compatible = retention_compatible_mask(
                            retention_library, final_train, regime
                        )
                        evaluation_positions = np.flatnonzero(compatible)
                        evaluation = retention_library.iloc[evaluation_positions].copy().reset_index(drop=True)
                        _require(
                            not evaluation.empty and set(evaluation["target_binder"].astype(int)) == {0, 1},
                            "retention-compatible subset lacks a class",
                        )
                        condition_rows.append(
                            dict(
                                metadata,
                                weak_regime_rows=int(len(regime_pool)),
                                weak_static_clean_rows=int(len(static_pool)),
                                weak_fit_rows=int(len(final_train)),
                                weak_fit_positive=int(final_train["weak_label"].astype(int).sum()),
                                weak_fit_negative=int((final_train["weak_label"].astype(int) == 0).sum()),
                                weak_fit_unique_peptide=int(final_train["chain1_sha256"].nunique()),
                                weak_fit_unique_affibody=int(final_train["chain2_sha256"].nunique()),
                                train_membership_sha256=membership_sha256(final_train),
                                retention_evaluation_rows=int(len(evaluation)),
                                retention_evaluation_positive=int(evaluation["target_binder"].astype(int).sum()),
                                retention_membership_sha256=membership_sha256(evaluation),
                                retention_rows_still_matching_actual_fit_exposure=int(
                                    np.sum(compatible & actual_compatible)
                                ),
                                retention_rows_shifted_by_cleaning_or_downsampling=int(
                                    np.sum(compatible & ~actual_compatible)
                                ),
                                evaluable_weak_folds=int(len(plans)),
                            )
                        )
                        for representation in representations:
                            selected_c, records = tune_c(
                                representation,
                                c_grid,
                                balance,
                                plans,
                                labels,
                                site_weak,
                                weak_mint,
                                metadata,
                            )
                            tuning_rows.extend(records)
                            if representation == "site":
                                probability, model = _fit_predict_head(
                                    representation,
                                    selected_c,
                                    balance,
                                    final_condition_indices,
                                    np.asarray([], dtype=int),
                                    labels,
                                    site_weak,
                                    weak_mint,
                                    direct_prediction_features=site_retention[
                                        evaluation_positions
                                    ],
                                )
                            else:
                                probability, model = _fit_predict_head(
                                    representation,
                                    selected_c,
                                    balance,
                                    final_condition_indices,
                                    np.asarray([], dtype=int),
                                    labels,
                                    site_weak,
                                    weak_mint,
                                    direct_prediction_features=retention_mint[
                                        evaluation_positions
                                    ],
                                )
                            core_mask = evaluation["sequence_pair_sha256"].astype(str).isin(
                                common_pair_identities
                            ).to_numpy()
                            actual_scope_mask = actual_compatible[evaluation_positions]
                            evaluation_scopes = (
                                ("fixed_c0_regime_panel", evaluation, probability),
                                (
                                    "common_regime_core",
                                    evaluation.loc[core_mask].reset_index(drop=True),
                                    probability[core_mask],
                                ),
                                (
                                    "actual_fit_compatible",
                                    evaluation.loc[actual_scope_mask].reset_index(drop=True),
                                    probability[actual_scope_mask],
                                ),
                            )
                            for evaluation_scope, scoped_evaluation, scoped_probability in evaluation_scopes:
                                if scoped_evaluation.empty or set(
                                    scoped_evaluation["target_binder"].astype(int)
                                ) != {0, 1}:
                                    continue
                                metrics = ranking_metrics(scoped_evaluation, scoped_probability)
                                metric_rows.append(
                                    dict(
                                        metadata,
                                        evaluation_scope=evaluation_scope,
                                        representation=representation,
                                        selected_C=float(selected_c),
                                        model_nonzero_coefficients=int(np.count_nonzero(model.coef_)),
                                        train_membership_sha256=membership_sha256(final_train),
                                        retention_membership_sha256=membership_sha256(
                                            scoped_evaluation
                                        ),
                                        **metrics
                                    )
                                )
                            predictions = evaluation[
                                [
                                    "pair_uid",
                                    "library",
                                    "target_retention",
                                    "target_binder",
                                    "chain1_sha256",
                                    "chain2_sha256",
                                    "sequence_pair_sha256",
                                ]
                            ].copy()
                            predictions["regime"] = regime
                            predictions["cleaning"] = cleaning
                            predictions["balance"] = balance
                            predictions["balance_seed"] = int(balance_seed)
                            predictions["representation"] = representation
                            predictions["selected_C"] = float(selected_c)
                            predictions["binder_probability"] = probability
                            predictions["in_common_regime_core"] = core_mask.astype(int)
                            predictions["matches_actual_fit_exposure"] = actual_scope_mask.astype(int)
                            predictions["train_membership_sha256"] = membership_sha256(final_train)
                            prediction_blocks.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    tuning = pd.DataFrame(tuning_rows)
    conditions = pd.DataFrame(condition_rows)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    _require(
        not bool(
            metrics.groupby(
                [
                    "library",
                    "regime",
                    "cleaning",
                    "balance",
                    "balance_seed",
                    "evaluation_scope",
                ]
            )["retention_membership_sha256"].nunique().gt(1).any()
        ),
        "site and MINT were evaluated on different retention memberships",
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    outputs = {
        "metrics.csv": output_dir / "metrics.csv",
        "weak_validation.csv": output_dir / "weak_validation.csv",
        "conditions.csv": output_dir / "conditions.csv",
        "predictions.csv": output_dir / "predictions.csv",
    }
    _write_private_csv(metrics, outputs["metrics.csv"])
    _write_private_csv(tuning, outputs["weak_validation.csv"])
    _write_private_csv(conditions, outputs["conditions.csv"])
    _write_private_csv(predictions, outputs["predictions.csv"])

    summary = [
        "# Frozen MINT weak-label factorial evaluation",
        "",
        "Site-only and frozen-MINT logistic heads used identical weak-label and retention memberships. "
        "Regularization was tuned on leakage-matched weak-label validation. Each prespecified factorial "
        "condition was then scored retrospectively on one fixed retention panel; comparisons across the many "
        "conditions are exploratory, not independent confirmatory tests.",
        "",
        "| Library | Regime | Scope | Cleaning | Balance | Seed | Representation | N | Prev. | AUROC | AUPRC | AP lift | Spearman | Within-peptide AP lift | P@1 | Interaction rho |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.sort_values(
        ["library", "regime", "cleaning", "balance", "balance_seed", "representation"]
    ).iterrows():
        summary.append(
            "| {library} | {regime} | {evaluation_scope} | {cleaning} | {balance} | {balance_seed} | "
            "{representation} | {n} | {global_prevalence:.3f} | {global_auroc:.3f} | "
            "{global_auprc:.3f} | {global_auprc_lift:.3f} | {global_spearman:.3f} | "
            "{within_peptide_macro_auprc_lift:.3f} | {within_peptide_p_at_1:.3f} | "
            "{two_way_interaction_spearman:.3f} |".format(**row.to_dict())
        )
    summary.extend(
        [
            "",
            "Definitions: pair-only requires both partners seen but the exact pair absent; peptide-cold "
            "holds out peptide identity while requiring the Affibody seen; Affibody-cold is the converse; "
            "double-cold holds out both. Identities are full-chain SHA256 values, including across libraries.",
            "",
            "Promiscuity breadth was computed inside each weak training fold (or the final weak fit pool) only. "
            "No measured retention outcome was used for cleaning or hyperparameter selection.",
            "Retention membership is fixed from the uncleaned regime pool so cleaning/balance ablations remain "
            "comparable. conditions.csv reports when row removal changes a test row's actual partner-exposure status.",
        ]
    )
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary) + "\n")
    os.chmod(str(summary_path), 0o600)
    outputs["run_summary.md"] = summary_path

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory",
        "configuration": {
            "libraries": list(libraries),
            "regimes": list(regimes),
            "cleanings": list(cleanings),
            "balances": list(balances),
            "representations": list(representations),
            "downsample_seeds": list(args.downsample_seeds),
            "folds": int(args.folds),
            "split_seed": int(args.split_seed),
            "c_grid": list(c_grid),
            "retention_usage": "identity holdout definition and final retrospective evaluation only",
            "regularization_selection": "minimum pooled weak-validation log loss; AP/AUROC/smaller-C tie-break",
            "global_identity_columns": ["chain1_sha256", "chain2_sha256", "sequence_pair_sha256"],
            "promiscuity_rule": "remove all train-fold rows for Affibody positive peptide breadth >= 10",
        },
        "sources": {
            "cache_npz": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in cache_paths
            ],
            "cache_manifests": cache_manifests,
            "retention_features": {
                "path": str(Path(args.retention_features).resolve()),
                "sha256": sha256_file(args.retention_features),
                "manifest": legacy_manifest,
            },
            "cache_legacy_retention_feature_max_abs_difference": max_feature_difference,
        },
        "rows": {
            "cache_total": int(len(cache_frame)),
            "weak": int(len(weak_cache)),
            "retention_total": int(len(retention)),
            "retention_measured": int(len(measured)),
            "conditions": int(len(conditions)),
            "metric_records": int(len(metrics)),
            "prediction_records": int(len(predictions)),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(outputs.items())
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "permissions": {"directory": "0700", "files": "0600"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "conditions": int(len(conditions)),
                "metric_records": int(len(metrics)),
                "retention_feature_max_abs_difference": max_feature_difference,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="merged NPZ, merged directory, or directory containing all shard NPZs",
    )
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument(
        "--retention-features",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_features_v1.npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libraries", nargs="+", default=list(LIBRARIES))
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES))
    parser.add_argument("--cleanings", nargs="+", default=list(CLEANINGS))
    parser.add_argument("--balances", nargs="+", default=list(BALANCES))
    parser.add_argument(
        "--representations", nargs="+", default=list(REPRESENTATIONS)
    )
    parser.add_argument(
        "--downsample-seeds", type=int, nargs="+", default=list(DEFAULT_DOWNSAMPLE_SEEDS)
    )
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--max-retention-feature-difference", type=float, default=1e-4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
