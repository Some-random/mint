#!/usr/bin/env python
"""Matched frozen-head, head-only, and MINT-LoRA selection-label experiment.

This driver deliberately follows the frozen-MINT primary analysis contract:
``c0`` labels, library-local peptide-and-Affibody-cold exclusion, the three
identity-blocked weak-validation folds made with seed 17, class-weighted
training, and a final refit on every eligible weak-label row.  Retention labels
are unavailable to model selection and are read only for the final metrics.

The historical ``finetune_mint_selection.py`` pilot is intentionally left
unchanged.  That pilot balanced the classes by discarding rows and used a
different validation split, so its values are not comparable to the primary
frozen-MINT result.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import evaluate_cached_weak_mint_practical as practical_eval
from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path
from downstream.AffibodyMHC.finetune_mint_retention import (
    set_deterministic_seed,
    update_learning_rates,
)
from downstream.AffibodyMHC.finetune_mint_selection import (
    MINTSelectionClassifier,
    load_head,
    make_loader,
    predict_live,
    trainable_audit,
)


SCHEMA_VERSION = "mint-selection-matched-lora-v1"
LIBRARIES = ("LibA", "LibB")
ARMS = ("head_only", "lora_cross")
REPRESENTATION = "frozen_mint_chain_mean"
REGIME = "double_cold"
CLEANING = "c0"
BALANCE = "all_class_weighted"
CANONICAL_SPLIT_SEED = 17
CANONICAL_FOLDS = 3
CANONICAL_C_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_TRAINING_SEEDS = (20260811, 20260812, 20260813)

PRIMARY_CONTRACT = {
    "LibA": {
        "rows": 22542,
        "positive": 11320,
        "negative": 11222,
        "membership_sha256": "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e",
        "retention_rows": 108,
        "retention_positive": 38,
        "retention_membership_sha256": "7f8f017b82b796f2613bc294cae385a19093aadd065552fd5288edbc1f7a93dc",
        "selected_c": 0.01,
        "folds": {
            0: {
                "train": 9076,
                "guard": 10513,
                "validation": 2953,
                "validation_positive": 1530,
                "train_sha256": "fd90f5dac3ee87c350940842211d1c3c82194c74ad6c5ab6b4429f011a3fd856",
                "validation_sha256": "57622c6653b18879b5f237b050dc0322d36b8d9a29d70d380c8b9df9c375b641",
            },
            1: {
                "train": 10488,
                "guard": 9791,
                "validation": 2263,
                "validation_positive": 1267,
                "train_sha256": "faf658d1b9f6e34a89e0deb6738f27f479f0c8bfeefbd8dda4d977262cb7ff8a",
                "validation_sha256": "6f6ae2b7f9d8bbb75c4e5563152a76f9620ba5169471278b4ba2eeba699a3224",
            },
            2: {
                "train": 10493,
                "guard": 9750,
                "validation": 2299,
                "validation_positive": 962,
                "train_sha256": "d2ffa04c39e502416a0ce6565cb7f2f8919af49892cd68a98927b73aa7c3b11b",
                "validation_sha256": "a0b2f9cd5e33cf5f182ca9294e3c703b2b99fc32009760943639a8f443f4a38f",
            },
        },
    },
    "LibB": {
        "rows": 30648,
        "positive": 23725,
        "negative": 6923,
        "membership_sha256": "1a9a0527c5f9f0e2bbeff6a0ea3d9afaaeadace894d46e73311b049a19f8a393",
        "retention_rows": 119,
        "retention_positive": 60,
        "retention_membership_sha256": "af173650631665199847f5421701f12b6b21eabb41b0b2154916aac805651620",
        "selected_c": 0.1,
        "folds": {
            0: {
                "train": 13174,
                "guard": 13895,
                "validation": 3579,
                "validation_positive": 2790,
                "train_sha256": "129aa59306d4816dc65e4dd5621e06160a51a46fb0d9fc59abcca0d1c3bf19ad",
                "validation_sha256": "d7652d12b3b042fb30dd86217d1d2e8394390d355650e7f0485240f946e5bef1",
            },
            1: {
                "train": 14825,
                "guard": 13060,
                "validation": 2763,
                "validation_positive": 2232,
                "train_sha256": "51c8948ab4bbf46dd343990d70e56f0a594842679f88caf837c136b7dd3a03f8",
                "validation_sha256": "de82dff8b2bf16fc6d7e9529b3cf55f27bdf92b9fb33bb3bcea8bf08bdede8d0",
            },
            2: {
                "train": 12830,
                "guard": 13979,
                "validation": 3839,
                "validation_positive": 2917,
                "train_sha256": "c8fbae12761c3793667ba7dcfe06bff861e0577fb687cc5731dc97072ccb4acb",
                "validation_sha256": "294619a95304b9731db2f5d36ed52f8c0e220371b528cc2b5fc63806adc68b98",
            },
        },
    },
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def derived_seed(training_seed, stage, fold=-1):
    payload = "{}|{}|{}".format(int(training_seed), stage, int(fold)).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2 ** 31 - 1)


def balanced_class_weights(labels):
    """Return sklearn ``class_weight='balanced'`` weights for labels 0 and 1."""
    values = np.asarray(labels, dtype=int)
    _require(set(values.tolist()) == {0, 1}, "balanced weights require both classes")
    total = float(len(values))
    negative = int(np.sum(values == 0))
    positive = int(np.sum(values == 1))
    return {0: total / (2.0 * negative), 1: total / (2.0 * positive)}


def binary_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "binary metrics require both classes")
    _require(len(labels) == len(probability), "metric length mismatch")
    _require(bool(np.isfinite(probability).all()), "non-finite probability")
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "ap": float(average_precision_score(labels, probability)),
    }


def select_epoch(validation_predictions, arm, training_seed, max_epochs):
    """Select 0..max_epochs from pooled fold predictions without retention data."""
    frame = validation_predictions.loc[
        validation_predictions["arm"].eq(arm)
        & validation_predictions["training_seed"].eq(int(training_seed))
    ].copy()
    _require(not frame.empty, "no validation predictions for epoch selection")
    rows = []
    for epoch in range(int(max_epochs) + 1):
        current = frame.loc[frame["epoch"].eq(epoch)]
        _require(not current.empty, "missing validation epoch {}".format(epoch))
        metrics = binary_metrics(current["weak_label"], current["probability"])
        rows.append(dict({"arm": arm, "training_seed": int(training_seed), "epoch": epoch}, **metrics))
    selected = min(
        rows,
        key=lambda row: (row["log_loss"], -row["ap"], -row["auroc"], row["epoch"]),
    )
    for row in rows:
        row["selected"] = int(row["epoch"] == selected["epoch"])
    return int(selected["epoch"]), rows


def read_cache_rows(path, manifest_path):
    """Read the prepared full-sequence table and prove it matches its manifest."""
    path = Path(path)
    manifest_path = Path(manifest_path)
    _require(path.is_file(), "cache row table is missing")
    _require(manifest_path.is_file(), "cache row manifest is missing")
    with open(str(manifest_path), "r") as handle:
        manifest = json.load(handle)
    expected = manifest.get("output", {}).get("rows_csv", {}).get("sha256")
    _require(expected == sha256_file(path), "cache row hash disagrees with manifest")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "row_index",
        "cache_uid",
        "source_kind",
        "library",
        "pair_uid",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    }
    _require(required.issubset(frame.columns), "cache row schema mismatch")
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(np.int64)
    frame = frame.sort_values("row_index").reset_index(drop=True)
    expected_index = np.arange(len(frame), dtype=np.int64)
    _require(np.array_equal(frame["row_index"].to_numpy(), expected_index), "cache rows are noncontiguous")
    _require(not bool(frame["cache_uid"].duplicated().any()), "duplicate cache UID")
    return frame, manifest


def attach_full_sequences(cache_frame, row_frame):
    """Attach raw chains after checking every cache identity field."""
    _require(len(cache_frame) == len(row_frame), "cache feature/row count mismatch")
    _require(
        np.array_equal(cache_frame["row_index"].to_numpy(dtype=np.int64), row_frame["row_index"].to_numpy(dtype=np.int64)),
        "cache feature/row index mismatch",
    )
    identity_columns = (
        "cache_uid",
        "source_kind",
        "library",
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    )
    for column in identity_columns:
        _require(
            bool(cache_frame[column].astype(str).eq(row_frame[column].astype(str)).all()),
            "cache feature/row {} mismatch".format(column),
        )
    output = cache_frame.copy()
    output["chain1_smart_hla_linker_peptide_sequence"] = row_frame[
        "chain1_smart_hla_linker_peptide_sequence"
    ].astype(str)
    output["chain2_affibody_sequence"] = row_frame["chain2_affibody_sequence"].astype(str)
    observed_chain1 = output["chain1_smart_hla_linker_peptide_sequence"].map(
        lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    )
    observed_chain2 = output["chain2_affibody_sequence"].map(
        lambda value: hashlib.sha256(value.encode("ascii")).hexdigest()
    )
    _require(bool(observed_chain1.eq(output["chain1_sha256"]).all()), "chain-1 sequence hash mismatch")
    _require(bool(observed_chain2.eq(output["chain2_sha256"]).all()), "chain-2 sequence hash mismatch")
    return output


def validate_reference_run(reference_run_dir, library):
    """Bind the rerun to the exact public-table frozen-MINT artifact."""
    reference_run_dir = Path(reference_run_dir)
    required = (
        reference_run_dir / "manifest.json",
        reference_run_dir / "conditions.csv",
        reference_run_dir / "weak_validation.csv",
        reference_run_dir / "metrics.csv",
    )
    for path in required:
        _require(path.is_file(), "missing primary reference artifact {}".format(path))
    with open(str(required[0]), "r") as handle:
        manifest = json.load(handle)
    manifest_outputs = manifest.get("outputs", {})
    for path in required[1:]:
        entry = manifest_outputs.get(path.name)
        _require(isinstance(entry, dict), "reference manifest lacks {}".format(path.name))
        _require(entry.get("sha256") == sha256_file(path), "reference {} hash mismatch".format(path.name))
    configuration = manifest.get("configuration", {})
    _require(configuration.get("regimes") == [REGIME], "reference regime changed")
    _require(configuration.get("cleanings") == [CLEANING], "reference cleaning changed")
    _require(configuration.get("balances") == [BALANCE], "reference balance changed")
    _require(int(configuration.get("folds", -1)) == CANONICAL_FOLDS, "reference fold count changed")
    _require(int(configuration.get("split_seed", -1)) == CANONICAL_SPLIT_SEED, "reference split seed changed")
    _require(
        tuple(float(value) for value in configuration.get("c_grid", ())) == CANONICAL_C_GRID,
        "reference C grid changed",
    )
    conditions = pd.read_csv(required[1])
    _require(len(conditions) == 1, "reference run must contain one condition")
    condition = conditions.iloc[0]
    contract = PRIMARY_CONTRACT[library]
    _require(str(condition["library"]) == library, "reference library mismatch")
    _require(int(condition["weak_fit_rows"]) == contract["rows"], "reference row count changed")
    _require(int(condition["weak_fit_positive"]) == contract["positive"], "reference positive count changed")
    _require(int(condition["weak_fit_negative"]) == contract["negative"], "reference negative count changed")
    _require(str(condition["train_membership_sha256"]) == contract["membership_sha256"], "reference membership changed")
    tuning = pd.read_csv(required[2])
    selected = tuning.loc[
        tuning["representation"].eq(REPRESENTATION)
        & tuning["record_type"].eq("aggregate")
        & tuning["selected"].eq(1)
    ]
    _require(len(selected) == 1, "reference selected-C record mismatch")
    selected_c = float(selected.iloc[0]["C"])
    _require(selected_c == float(contract["selected_c"]), "reference selected C changed")
    metrics = pd.read_csv(required[3])
    frozen = metrics.loc[
        metrics["library"].eq(library)
        & metrics["regime"].eq(REGIME)
        & metrics["cleaning"].eq(CLEANING)
        & metrics["balance"].eq(BALANCE)
        & metrics["evaluation_scope"].eq("fixed_c0_regime_panel")
        & metrics["representation"].eq(REPRESENTATION)
    ]
    _require(len(frozen) == 1, "reference frozen metric record mismatch")
    reference_metrics = {
        key: float(frozen.iloc[0][key])
        for key in (
            "global_auroc",
            "global_auprc",
            "global_spearman",
            "within_peptide_macro_spearman",
        )
    }
    return {
        "directory": str(reference_run_dir.resolve()),
        "manifest_sha256": sha256_file(required[0]),
        "conditions_sha256": sha256_file(required[1]),
        "weak_validation_sha256": sha256_file(required[2]),
        "metrics_sha256": sha256_file(required[3]),
        "selected_c": selected_c,
        "metrics": reference_metrics,
    }


def make_primary_pool(cache_frame, library):
    """Reconstruct the library-local C0/double-cold pool used in the main table."""
    weak = cache_frame.loc[
        cache_frame["source_kind"].eq("weak") & cache_frame["library"].eq(library)
    ].copy()
    retention = cache_frame.loc[
        cache_frame["source_kind"].eq("retention")
        & cache_frame["library"].eq(library)
        & cache_frame["measurement_missing"].astype(str).eq("0")
    ].copy()
    _require(not weak.empty and not retention.empty, "empty library data")
    weak["weak_label"] = pd.to_numeric(weak["weak_label"], errors="raise").astype(int)
    # This intentionally does not use strict_retention_identity_cold_eligible:
    # that prepared flag excludes identities across both libraries, whereas the
    # public primary result excludes retention identities within this library.
    primary = cached_eval._regime_pool(weak, retention, REGIME)
    primary = cached_eval.apply_static_cleaning(primary, CLEANING)
    primary = primary.reset_index(drop=True)
    primary["_condition_index"] = np.arange(len(primary), dtype=int)
    contract = PRIMARY_CONTRACT[library]
    observed = {
        "rows": int(len(primary)),
        "positive": int(primary["weak_label"].sum()),
        "negative": int(primary["weak_label"].eq(0).sum()),
        "membership_sha256": cached_eval.membership_sha256(primary),
    }
    for key in ("rows", "positive", "negative", "membership_sha256"):
        _require(observed[key] == contract[key], "canonical primary {} changed".format(key))
    return primary, retention


def canonical_fold_plans(primary, library):
    plans = cached_eval.build_fold_plans(
        primary,
        REGIME,
        CLEANING,
        BALANCE,
        seed=-1,
        folds=CANONICAL_FOLDS,
        split_seed=CANONICAL_SPLIT_SEED,
    )
    _require(len(plans) == CANONICAL_FOLDS, "canonical fold count changed")
    membership_rows = []
    for plan in plans:
        fold = int(plan["fold"])
        expected = PRIMARY_CONTRACT[library]["folds"][fold]
        train_keys = np.asarray(plan["train_keys"], dtype=int)
        validation_keys = np.asarray(plan["validation_keys"], dtype=int)
        train = primary.iloc[train_keys]
        validation = primary.iloc[validation_keys]
        guard_mask = np.ones(len(primary), dtype=bool)
        guard_mask[train_keys] = False
        guard_mask[validation_keys] = False
        guard_keys = np.flatnonzero(guard_mask)
        observed = {
            "train": int(len(train_keys)),
            "guard": int(len(guard_keys)),
            "validation": int(len(validation_keys)),
            "validation_positive": int(validation["weak_label"].sum()),
            "train_sha256": cached_eval.membership_sha256(train),
            "validation_sha256": cached_eval.membership_sha256(validation),
        }
        for key, value in observed.items():
            _require(value == expected[key], "canonical fold {} {} changed".format(fold, key))
        role = np.full(len(primary), "guard", dtype=object)
        role[train_keys] = "train"
        role[validation_keys] = "validation"
        block = primary[["pair_uid", "chain1_sha256", "chain2_sha256", "weak_label"]].copy()
        block["fold"] = fold
        block["role"] = role
        membership_rows.append(block)
    return plans, pd.concat(membership_rows, ignore_index=True)


def load_matched_inputs(args):
    """Load exact frozen features, raw chains, pools, folds, and retention rows."""
    cache_frame, cache_features, cache_manifests, cache_paths = cached_eval.load_feature_cache(
        args.cache, args.cache_manifest
    )
    row_frame, row_manifest = read_cache_rows(args.cache_rows, args.cache_rows_manifest)
    cache_frame = attach_full_sequences(cache_frame, row_frame)
    cache_frame["_cache_index"] = cache_frame.index.to_numpy(dtype=int)
    primary, retention_cache = make_primary_pool(cache_frame, args.library)
    plans, fold_membership = canonical_fold_plans(primary, args.library)

    legacy_table, legacy_features, legacy_manifest = cached_eval.load_legacy_retention_features(
        args.retention_features
    )
    retention = retention_cache.merge(legacy_table, on="pair_uid", validate="one_to_one")
    _require(len(retention) == len(retention_cache), "retention legacy join mismatch")
    _require(bool(retention["library"].eq(retention["legacy_library"]).all()), "retention library mismatch")
    _require(bool(retention["measurement_missing"].astype(str).eq(retention["legacy_missing"]).all()), "retention missingness mismatch")
    retention = retention.sort_values("pair_uid").reset_index(drop=True)
    retention["target_retention"] = pd.to_numeric(retention["target_retention"], errors="raise")
    retention["target_binder"] = pd.to_numeric(retention["target_binder"], errors="raise").astype(int)
    expected_binder = retention["target_retention"].ge(75.0).astype(int)
    _require(bool(retention["target_binder"].eq(expected_binder).all()), "75-percent binder labels changed")
    contract = PRIMARY_CONTRACT[args.library]
    _require(len(retention) == contract["retention_rows"], "retention row count changed")
    _require(int(retention["target_binder"].sum()) == contract["retention_positive"], "retention positive count changed")
    _require(
        cached_eval.membership_sha256(retention) == contract["retention_membership_sha256"],
        "retention membership changed",
    )

    primary_indices = primary["_cache_index"].to_numpy(dtype=int)
    weak_features = cache_features[primary_indices]
    retention_indices = retention["legacy_index"].to_numpy(dtype=int)
    retention_features = legacy_features[retention_indices]
    cached_retention_features = cache_features[retention["_cache_index"].to_numpy(dtype=int)]
    max_difference = float(np.max(np.abs(cached_retention_features - retention_features)))
    _require(max_difference <= float(args.max_retention_feature_difference), "retention feature archives disagree")
    return {
        "cache_frame": cache_frame,
        "cache_features": cache_features,
        "cache_manifests": cache_manifests,
        "cache_paths": cache_paths,
        "row_manifest": row_manifest,
        "legacy_manifest": legacy_manifest,
        "primary": primary,
        "weak_features": weak_features,
        "retention": retention,
        "retention_features": retention_features,
        "plans": plans,
        "fold_membership": fold_membership,
        "retention_feature_max_abs_difference": max_difference,
    }


def fit_standardizer(features):
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    _require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()), "bad standardizer")
    return mean, scale


def standardize(features, mean, scale):
    output = (np.asarray(features, dtype=np.float64) - mean) / scale
    _require(bool(np.isfinite(output).all()), "non-finite standardized features")
    return output.astype(np.float32)


def fit_logistic(features, labels, c_value):
    classifier = practical_eval.make_logistic(float(c_value), BALANCE)
    classifier.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=int))
    _require(int(classifier.n_iter_[0]) < int(classifier.max_iter), "logistic head did not converge")
    return classifier


def select_primary_c(weak_features, labels, plans, c_grid):
    """Reproduce primary pooled weak-validation C selection."""
    prepared = []
    for plan in plans:
        train = np.asarray(plan["train_keys"], dtype=int)
        validation = np.asarray(plan["validation_keys"], dtype=int)
        mean, scale = fit_standardizer(weak_features[train])
        prepared.append(
            (
                plan,
                standardize(weak_features[train], mean, scale),
                standardize(weak_features[validation], mean, scale),
            )
        )
    rows = []
    aggregates = []
    for c_value in c_grid:
        pooled_label = []
        pooled_probability = []
        for plan, x_train, x_validation in prepared:
            train = np.asarray(plan["train_keys"], dtype=int)
            validation = np.asarray(plan["validation_keys"], dtype=int)
            classifier = fit_logistic(x_train, labels[train], c_value)
            probability = classifier.predict_proba(x_validation)[:, 1]
            metrics = binary_metrics(labels[validation], probability)
            rows.append(
                dict(
                    {"record_type": "fold", "C": float(c_value), "fold": int(plan["fold"]), "selected": 0},
                    **metrics
                )
            )
            pooled_label.extend(labels[validation].tolist())
            pooled_probability.extend(probability.tolist())
        metrics = binary_metrics(pooled_label, pooled_probability)
        aggregate = dict({"record_type": "aggregate", "C": float(c_value), "fold": -1, "selected": 0}, **metrics)
        aggregates.append(aggregate)
        rows.append(aggregate)
    selected = min(
        aggregates,
        key=lambda row: (row["log_loss"], -row["ap"], -row["auroc"], row["C"]),
    )
    for row in rows:
        row["selected"] = int(float(row["C"]) == float(selected["C"]))
    return float(selected["C"]), rows


def weighted_bce_mean(logits, targets, class_weights):
    """Sklearn-style balanced per-example BCE, normalized by row count."""
    _require(logits.ndim == 1 and targets.ndim == 1, "weighted BCE needs vectors")
    _require(len(logits) == len(targets), "weighted BCE length mismatch")
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    )
    weights = torch.where(
        targets.eq(1),
        torch.as_tensor(float(class_weights[1]), dtype=losses.dtype, device=losses.device),
        torch.as_tensor(float(class_weights[0]), dtype=losses.dtype, device=losses.device),
    )
    return torch.sum(losses * weights) / float(len(targets))


def live_trainable_audit(model, arm, rank):
    if arm == "head_only":
        for parameter in model.adapter_parameters():
            parameter.requires_grad_(False)
    elif arm != "lora_cross":
        raise ValueError("unknown training arm {}".format(arm))
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    expected = 2561 if arm == "head_only" else 2561 + 10240 * int(rank)
    _require(count == expected, "unexpected {} trainable count {}".format(arm, count))
    if arm == "head_only":
        _require(all(name.startswith("head.") for name in names), "head-only arm trains an adapter")
    else:
        trainable_audit(model, rank)
    return names, count


def optimizer_for_live(model, arm, args):
    groups = [
        {
            "params": [model.head.weight],
            "lr": float(args.head_lr),
            "base_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
        },
        {
            "params": [model.head.bias],
            "lr": float(args.head_lr),
            "base_lr": float(args.head_lr),
            "weight_decay": 0.0,
        },
    ]
    if arm == "lora_cross":
        adapter_parameters = model.adapter_parameters()
        _require(adapter_parameters, "LoRA arm has no adapters")
        groups.append(
            {
                "params": adapter_parameters,
                "lr": float(args.adapter_lr),
                "base_lr": float(args.adapter_lr),
                "weight_decay": float(args.weight_decay),
            }
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)


def compact_model_state(model, arm):
    payload = {
        "head_state_dict": {
            name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()
        }
    }
    if arm == "lora_cross":
        payload["adapter_state"] = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        }
    else:
        payload["adapter_state"] = {}
    return payload


def parameter_delta(initial, current):
    """Return L2 and max-absolute changes for identically keyed tensor maps."""
    _require(set(initial) == set(current), "parameter delta key mismatch")
    squared = 0.0
    maximum = 0.0
    for name in sorted(initial):
        before = initial[name].detach().cpu().double()
        after = current[name].detach().cpu().double()
        _require(before.shape == after.shape, "parameter delta shape mismatch for {}".format(name))
        difference = after - before
        squared += float(torch.sum(difference * difference))
        if difference.numel():
            maximum = max(maximum, float(torch.max(torch.abs(difference))))
    return {"l2": float(math.sqrt(squared)), "max_abs": maximum}


def train_live_arm(
    arm,
    train_frame,
    evaluation_frame,
    mean,
    scale,
    classifier,
    training_seed,
    stage,
    fold,
    epochs_to_run,
    args,
    device,
    expected_epoch0_probability=None,
    track_each_epoch=True,
):
    """Train one live-MINT arm and return epoch-wise evaluation predictions."""
    _require(arm in ARMS, "unknown live arm")
    _require(0 <= int(epochs_to_run) <= int(args.max_epochs), "epoch count outside candidates")
    train_frame = train_frame.reset_index(drop=True).copy()
    evaluation_frame = evaluation_frame.reset_index(drop=True).copy()
    for frame, label in ((train_frame, "train"), (evaluation_frame, "evaluation")):
        required = {
            "pair_uid",
            "weak_label",
            "chain1_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence",
        }
        _require(required.issubset(frame.columns), "{} live frame schema mismatch".format(label))
    labels = train_frame["weak_label"].to_numpy(dtype=int)
    class_weights = balanced_class_weights(labels)
    # Both arms deliberately receive the same model-initialization seed and
    # DataLoader generator seed.  The adapter trainability is the only arm
    # difference.
    run_seed = derived_seed(training_seed, stage, fold)
    set_deterministic_seed(run_seed)
    model = MINTSelectionClassifier(
        args.config,
        args.checkpoint,
        device,
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    ).to(device)
    model.set_feature_standardization(mean, scale)
    load_head(model, classifier)
    trainable_names, trainable_count = live_trainable_audit(model, arm, args.lora_rank)
    initial_head = {
        name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()
    }
    initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    _require(len(initial_adapter) == 8, "unexpected adapter tensor count")

    evaluation_loader = make_loader(evaluation_frame, args.eval_batch_size, False, run_seed)
    parity_error = None
    prediction_blocks = []
    history = []
    if track_each_epoch or expected_epoch0_probability is not None:
        _, probability, observed, pair_uids = predict_live(model, evaluation_loader, device)
        _require(pair_uids == evaluation_frame["pair_uid"].tolist(), "live evaluation UID order mismatch")
        _require(
            bool(np.array_equal(observed, evaluation_frame["weak_label"].to_numpy(dtype=int))),
            "live evaluation labels changed",
        )
    if expected_epoch0_probability is not None:
        expected = np.asarray(expected_epoch0_probability, dtype=float)
        _require(len(expected) == len(probability), "epoch-0 parity length mismatch")
        parity_error = float(np.max(np.abs(expected - probability)))
        _require(parity_error <= float(args.max_live_feature_probability_difference), "cached/live epoch-0 mismatch")
    if track_each_epoch:
        prediction_blocks.append(
            pd.DataFrame(
                {
                    "pair_uid": pair_uids,
                    "weak_label": observed,
                    "epoch": 0,
                    "probability": probability,
                }
            )
        )
        history.append(dict({"epoch": 0, "train_loss": None}, **binary_metrics(observed, probability)))

    optimizer = optimizer_for_live(model, arm, args)
    train_loader = make_loader(train_frame, args.batch_size, True, run_seed)
    updates_per_epoch = max(1, int(math.ceil(float(len(train_loader)) / float(args.accumulation_steps))))
    schedule_total_steps = max(1, updates_per_epoch * int(args.max_epochs))
    global_step = 0
    for epoch in range(1, int(epochs_to_run) + 1):
        model.set_train_mode()
        optimizer.zero_grad()
        total_weighted_loss = 0.0
        examples = 0
        window_examples = 0
        for batch_number, (chains, chain_ids, target, _) in enumerate(train_loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            target = target.to(device)
            logits = model(chains, chain_ids)
            # Multiply back by the batch size because gradients are normalized
            # once across an accumulation window below.
            loss = weighted_bce_mean(logits, target, class_weights) * float(len(target))
            loss.backward()
            total_weighted_loss += float(loss.detach().cpu())
            examples += int(len(target))
            window_examples += int(len(target))
            tail = batch_number + 1 == len(train_loader)
            if (batch_number + 1) % int(args.accumulation_steps) == 0 or tail:
                update_learning_rates(
                    optimizer, global_step, schedule_total_steps, args.warmup_fraction
                )
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.div_(float(window_examples))
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(args.clip_norm),
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                window_examples = 0
        if track_each_epoch:
            _, probability, observed, pair_uids = predict_live(model, evaluation_loader, device)
            metrics = binary_metrics(observed, probability)
            history.append(dict({"epoch": epoch, "train_loss": total_weighted_loss / float(examples)}, **metrics))
            prediction_blocks.append(
                pd.DataFrame(
                    {
                        "pair_uid": pair_uids,
                        "weak_label": observed,
                        "epoch": epoch,
                        "probability": probability,
                    }
                )
            )
            message_tail = " validation_log_loss {:.6f}".format(metrics["log_loss"])
        else:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_weighted_loss / float(examples),
                }
            )
            message_tail = ""
        print(
            "{} {} seed {} fold {} epoch {} train_loss {:.6f}{}".format(
                stage,
                arm,
                training_seed,
                fold,
                epoch,
                total_weighted_loss / float(examples),
                message_tail,
            ),
            flush=True,
        )
    if not track_each_epoch:
        # The final retention panel is scored once, after the all-row refit.
        _, probability, observed, pair_uids = predict_live(model, evaluation_loader, device)
        _require(pair_uids == evaluation_frame["pair_uid"].tolist(), "final evaluation UID order mismatch")
        prediction_blocks.append(
            pd.DataFrame(
                {
                    "pair_uid": pair_uids,
                    "weak_label": observed,
                    "epoch": int(epochs_to_run),
                    "probability": probability,
                }
            )
        )
    final_head = {
        name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()
    }
    final_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    head_delta = parameter_delta(initial_head, final_head)
    adapter_delta = parameter_delta(initial_adapter, final_adapter)
    if arm == "head_only":
        _require(adapter_delta["max_abs"] == 0.0, "head-only arm changed an adapter")
    elif int(epochs_to_run) == 0:
        _require(adapter_delta["max_abs"] == 0.0, "epoch-zero LoRA changed an adapter")
    else:
        _require(adapter_delta["l2"] > 0.0, "trained LoRA arm did not change an adapter")
    return {
        "predictions": pd.concat(prediction_blocks, ignore_index=True),
        "history": history,
        "model_state": compact_model_state(model, arm),
        "trainable_names": trainable_names,
        "trainable_count": trainable_count,
        "class_weights": class_weights,
        "run_seed": run_seed,
        "epoch0_probability_max_abs_error": parity_error,
        "schedule_total_steps": schedule_total_steps,
        "head_parameter_delta": head_delta,
        "adapter_parameter_delta": adapter_delta,
    }


def retention_metric_record(retention, probability, library, arm, training_seed, selected_epoch):
    probability = np.asarray(probability, dtype=float)
    _require(len(probability) == len(retention), "retention probability length mismatch")
    metrics = cached_eval.ranking_metrics(retention.reset_index(drop=True), probability)
    return dict(
        {
            "library": library,
            "arm": arm,
            "training_seed": int(training_seed),
            "selected_epoch": int(selected_epoch),
        },
        **metrics
    )


def per_peptide_records(retention, probability, library, arm, training_seed, selected_epoch):
    frame = retention.reset_index(drop=True)
    probability = np.asarray(probability, dtype=float)
    rows = []
    for peptide_identity, indices in frame.groupby("chain1_sha256", sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        labels = frame.iloc[index]["target_binder"].to_numpy(dtype=int)
        target = frame.iloc[index]["target_retention"].to_numpy(dtype=float)
        score = probability[index]
        row = {
            "library": library,
            "arm": arm,
            "training_seed": int(training_seed),
            "selected_epoch": int(selected_epoch),
            "peptide_identity": str(peptide_identity),
            "n": int(len(index)),
            "positive": int(labels.sum()),
            "spearman": cached_eval._safe_spearman(target, score),
            "auroc": float("nan"),
            "ap": float("nan"),
        }
        if set(labels.tolist()) == {0, 1}:
            row["auroc"] = float(roc_auc_score(labels, score))
            row["ap"] = float(average_precision_score(labels, score))
        rows.append(row)
    return rows


def training_audit_record(
    stage,
    arm,
    training_seed,
    fold,
    train_frame,
    mean,
    scale,
    result,
    selected_epoch,
):
    labels = train_frame["weak_label"].to_numpy(dtype=int)
    return {
        "stage": stage,
        "arm": arm,
        "training_seed": int(training_seed),
        "fold": int(fold),
        "rows": int(len(train_frame)),
        "positive": int(labels.sum()),
        "negative": int(np.sum(labels == 0)),
        "class_weight_negative": float(result["class_weights"][0]),
        "class_weight_positive": float(result["class_weights"][1]),
        "membership_sha256": cached_eval.membership_sha256(train_frame),
        "feature_mean_sha256": sha256_array(np.asarray(mean, dtype=np.float64)),
        "feature_scale_sha256": sha256_array(np.asarray(scale, dtype=np.float64)),
        "trainable_parameters": int(result["trainable_count"]),
        "trainable_names": json.dumps(result["trainable_names"], sort_keys=True),
        "run_seed": int(result["run_seed"]),
        "selected_epoch": int(selected_epoch),
        "schedule_total_steps": int(result["schedule_total_steps"]),
        "epoch0_probability_max_abs_error": result["epoch0_probability_max_abs_error"],
        "head_parameter_delta_l2": float(result["head_parameter_delta"]["l2"]),
        "head_parameter_delta_max_abs": float(result["head_parameter_delta"]["max_abs"]),
        "adapter_parameter_delta_l2": float(result["adapter_parameter_delta"]["l2"]),
        "adapter_parameter_delta_max_abs": float(result["adapter_parameter_delta"]["max_abs"]),
    }


def aggregate_seed_metrics(metrics):
    requested = (
        "global_auroc",
        "global_auprc",
        "global_spearman",
        "within_peptide_macro_spearman",
    )
    rows = []
    for (library, arm), group in metrics.groupby(["library", "arm"], sort=True):
        row = {
            "library": library,
            "arm": arm,
            "n_seed_runs": int(len(group)),
            "training_seeds": ",".join(str(value) for value in group["training_seed"].tolist()),
        }
        for metric in requested:
            values = group[metric].to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_metric_differences(metrics):
    requested = (
        "global_auroc",
        "global_auprc",
        "global_spearman",
        "within_peptide_macro_spearman",
    )
    frozen = metrics.loc[metrics["arm"].eq("frozen_logistic")].iloc[0]
    rows = []
    for _, row in metrics.loc[~metrics["arm"].eq("frozen_logistic")].iterrows():
        output = {
            "library": row["library"],
            "arm": row["arm"],
            "training_seed": int(row["training_seed"]),
        }
        for metric in requested:
            output[metric + "_minus_frozen"] = float(row[metric] - frozen[metric])
        rows.append(output)
    for training_seed, group in metrics.loc[
        metrics["arm"].isin(["head_only", "lora_cross"])
    ].groupby("training_seed", sort=True):
        by_arm = group.set_index("arm")
        _require(set(by_arm.index) == {"head_only", "lora_cross"}, "paired arm result missing")
        output = {
            "library": str(group.iloc[0]["library"]),
            "arm": "lora_cross_minus_head_only",
            "training_seed": int(training_seed),
        }
        for metric in requested:
            output[metric + "_minus_frozen"] = float(
                by_arm.loc["lora_cross", metric] - by_arm.loc["head_only", metric]
            )
        rows.append(output)
    return pd.DataFrame(rows)


def _write_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def run(args):
    started = time.time()
    _require(args.library in LIBRARIES, "unknown library")
    _require(args.split_seed == CANONICAL_SPLIT_SEED, "matched run requires split seed 17")
    _require(args.folds == CANONICAL_FOLDS, "matched run requires three folds")
    _require(tuple(float(value) for value in args.c_grid) == CANONICAL_C_GRID, "matched run requires canonical C grid")
    _require(args.max_epochs == 3, "matched run requires candidate epochs 0--3")
    _require(args.lora_rank == 2, "matched run requires rank-2 LoRA")
    _require(args.batch_size >= 1 and args.eval_batch_size >= 1, "batch sizes must be positive")
    _require(args.accumulation_steps >= 1, "accumulation steps must be positive")
    _require(0.0 <= args.warmup_fraction < 1.0, "bad warmup fraction")
    _require(args.weight_decay >= 0.0, "negative weight decay")
    _require(args.clip_norm > 0.0, "clip norm must be positive")
    training_seeds = tuple(int(value) for value in args.training_seeds)
    _require(training_seeds and len(set(training_seeds)) == len(training_seeds), "training seeds must be unique")
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required for live MINT training")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    source_paths = {
        "script": Path(__file__).resolve(),
        "cache_rows": Path(args.cache_rows),
        "cache_rows_manifest": Path(args.cache_rows_manifest),
        "retention_features": Path(args.retention_features),
        "retention_features_manifest": Path(args.retention_features).with_suffix(".manifest.json"),
        "checkpoint": Path(args.checkpoint),
        "config": Path(args.config),
        "cached_evaluator": Path(cached_eval.__file__).resolve(),
        "practical_evaluator": Path(practical_eval.__file__).resolve(),
        "historical_lora_dependency": REPO_ROOT / "downstream/AffibodyMHC/finetune_mint_selection.py",
        "retention_lora_dependency": REPO_ROOT / "downstream/AffibodyMHC/finetune_mint_retention.py",
        "pair_collator_dependency": REPO_ROOT / "downstream/AffibodyMHC/extract_mint_features.py",
        "mint_wrapper_dependency": REPO_ROOT / "mint/helpers/extract.py",
        "mint_esm2_dependency": REPO_ROOT / "mint/model/esm2.py",
        "mint_modules_dependency": REPO_ROOT / "mint/modules.py",
        "mint_attention_dependency": REPO_ROOT / "mint/multihead_attention.py",
        "mint_data_dependency": REPO_ROOT / "mint/data.py",
        "mint_rotary_dependency": REPO_ROOT / "mint/rotary_embedding.py",
    }
    for name, path in source_paths.items():
        _require(path.is_file(), "missing {} input {}".format(name, path))
    initial_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    reference = validate_reference_run(args.reference_run_dir, args.library)
    inputs = load_matched_inputs(args)
    for index, cache_path in enumerate(inputs["cache_paths"]):
        source_paths["cache_npz_{:03d}".format(index)] = Path(cache_path)
        if args.cache_manifest is not None and len(inputs["cache_paths"]) == 1:
            cache_manifest_path = Path(args.cache_manifest)
        else:
            cache_manifest_path = cached_eval._manifest_for_npz(Path(cache_path))
        source_paths["cache_manifest_{:03d}".format(index)] = cache_manifest_path
    for name in ("manifest.json", "conditions.csv", "weak_validation.csv", "metrics.csv"):
        source_paths["reference_{}".format(name.replace(".", "_"))] = Path(
            args.reference_run_dir
        ) / name
    for name, path in source_paths.items():
        _require(path.is_file(), "missing immutable source {}".format(path))
        if name not in initial_hashes:
            initial_hashes[name] = sha256_file(path)
    primary = inputs["primary"]
    weak_features = inputs["weak_features"]
    retention = inputs["retention"]
    retention_features = inputs["retention_features"]
    labels = primary["weak_label"].to_numpy(dtype=int)

    selected_c, c_history = select_primary_c(
        weak_features, labels, inputs["plans"], tuple(args.c_grid)
    )
    _require(selected_c == float(reference["selected_c"]), "recomputed selected C differs from primary")
    full_mean, full_scale = fit_standardizer(weak_features)
    full_x = standardize(weak_features, full_mean, full_scale)
    full_classifier = fit_logistic(full_x, labels, selected_c)

    validation_predictions = []
    validation_history = []
    epoch_selection_rows = []
    training_audit_rows = []
    final_prediction_blocks = []
    metric_rows = []
    per_peptide_rows = []
    saved_states = {}
    selected_epochs_by_seed = {}

    for training_seed in training_seeds:
        seed_validation_blocks = []
        for plan in inputs["plans"]:
            fold = int(plan["fold"])
            train_keys = np.asarray(plan["train_keys"], dtype=int)
            validation_keys = np.asarray(plan["validation_keys"], dtype=int)
            train_frame = primary.iloc[train_keys].copy().reset_index(drop=True)
            validation_frame = primary.iloc[validation_keys].copy().reset_index(drop=True)
            fold_mean, fold_scale = fit_standardizer(weak_features[train_keys])
            fold_x_train = standardize(weak_features[train_keys], fold_mean, fold_scale)
            fold_x_validation = standardize(weak_features[validation_keys], fold_mean, fold_scale)
            fold_classifier = fit_logistic(fold_x_train, labels[train_keys], selected_c)
            cached_probability = fold_classifier.predict_proba(fold_x_validation)[:, 1]
            for arm in ARMS:
                result = train_live_arm(
                    arm,
                    train_frame,
                    validation_frame,
                    fold_mean,
                    fold_scale,
                    fold_classifier,
                    training_seed,
                    "cross_validation",
                    fold,
                    args.max_epochs,
                    args,
                    device,
                    expected_epoch0_probability=cached_probability,
                    track_each_epoch=True,
                )
                predictions = result["predictions"].copy()
                predictions["library"] = args.library
                predictions["arm"] = arm
                predictions["training_seed"] = int(training_seed)
                predictions["fold"] = fold
                seed_validation_blocks.append(predictions)
                validation_predictions.append(predictions)
                for row in result["history"]:
                    validation_history.append(
                        dict(
                            {
                                "library": args.library,
                                "arm": arm,
                                "training_seed": int(training_seed),
                                "fold": fold,
                            },
                            **row
                        )
                    )
                training_audit_rows.append(
                    training_audit_record(
                        "cross_validation",
                        arm,
                        training_seed,
                        fold,
                        train_frame,
                        fold_mean,
                        fold_scale,
                        result,
                        selected_epoch=-1,
                    )
                )
                del result
                torch.cuda.empty_cache()

        seed_validation = pd.concat(seed_validation_blocks, ignore_index=True)
        selected_epochs = {}
        for arm in ARMS:
            selected_epoch, selection_rows = select_epoch(
                seed_validation, arm, training_seed, args.max_epochs
            )
            selected_epochs[arm] = selected_epoch
            for row in selection_rows:
                row["library"] = args.library
            epoch_selection_rows.extend(selection_rows)
        selected_epochs_by_seed[int(training_seed)] = selected_epochs

    # The leakage boundary is mechanical: every C and epoch has now been fixed
    # from weak labels.  Only below this line are retention values or 75%-binder
    # labels scored.  Partner sequence identities were already used, without
    # outcomes, to construct the cold pool.
    frozen_retention_x = standardize(retention_features, full_mean, full_scale)
    frozen_probability = full_classifier.predict_proba(frozen_retention_x)[:, 1]
    frozen_metric = retention_metric_record(
        retention, frozen_probability, args.library, "frozen_logistic", -1, 0
    )
    for metric_name, expected in reference["metrics"].items():
        _require(
            abs(float(frozen_metric[metric_name]) - float(expected)) <= 1e-10,
            "frozen primary {} was not reproduced".format(metric_name),
        )
    metric_rows.append(frozen_metric)
    per_peptide_rows.extend(
        per_peptide_records(
            retention, frozen_probability, args.library, "frozen_logistic", -1, 0
        )
    )
    frozen_predictions = retention[
        [
            "pair_uid",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "target_retention",
            "target_binder",
        ]
    ].copy()
    frozen_predictions["library"] = args.library
    frozen_predictions["arm"] = "frozen_logistic"
    frozen_predictions["training_seed"] = -1
    frozen_predictions["selected_epoch"] = 0
    frozen_predictions["probability"] = frozen_probability
    final_prediction_blocks.append(frozen_predictions)

    final_evaluation = retention.copy()
    # The collator API requires a label field, but inference does not use it.
    # A dummy value prevents true retention labels from entering the live model
    # training/inference path; metrics are joined from ``retention`` afterward.
    final_evaluation["weak_label"] = 0
    for training_seed in training_seeds:
        selected_epochs = selected_epochs_by_seed[int(training_seed)]
        for arm in ARMS:
            selected_epoch = selected_epochs[arm]
            result = train_live_arm(
                arm,
                primary,
                final_evaluation,
                full_mean,
                full_scale,
                full_classifier,
                training_seed,
                "final_refit",
                -1,
                selected_epoch,
                args,
                device,
                expected_epoch0_probability=frozen_probability,
                track_each_epoch=False,
            )
            predictions = result["predictions"].copy()
            _require(len(predictions) == len(retention), "final prediction count changed")
            probability = predictions["probability"].to_numpy(dtype=float)
            output = retention[
                [
                    "pair_uid",
                    "chain1_sha256",
                    "chain2_sha256",
                    "sequence_pair_sha256",
                    "target_retention",
                    "target_binder",
                ]
            ].copy()
            output["library"] = args.library
            output["arm"] = arm
            output["training_seed"] = int(training_seed)
            output["selected_epoch"] = int(selected_epoch)
            output["probability"] = probability
            final_prediction_blocks.append(output)
            metric_rows.append(
                retention_metric_record(
                    retention,
                    probability,
                    args.library,
                    arm,
                    training_seed,
                    selected_epoch,
                )
            )
            per_peptide_rows.extend(
                per_peptide_records(
                    retention,
                    probability,
                    args.library,
                    arm,
                    training_seed,
                    selected_epoch,
                )
            )
            training_audit_rows.append(
                training_audit_record(
                    "final_refit",
                    arm,
                    training_seed,
                    -1,
                    primary,
                    full_mean,
                    full_scale,
                    result,
                    selected_epoch,
                )
            )
            saved_states[(training_seed, arm)] = {
                "library": args.library,
                "arm": arm,
                "training_seed": int(training_seed),
                "selected_epoch": int(selected_epoch),
                "selected_C": float(selected_c),
                "base_checkpoint_sha256": initial_hashes["checkpoint"],
                "primary_membership_sha256": PRIMARY_CONTRACT[args.library]["membership_sha256"],
                "feature_mean": torch.from_numpy(np.asarray(full_mean, dtype=np.float32)),
                "feature_scale": torch.from_numpy(np.asarray(full_scale, dtype=np.float32)),
                **result["model_state"]
            }
            del result
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    aggregate_metrics = aggregate_seed_metrics(metrics)
    differences = paired_metric_differences(metrics)
    final_predictions = pd.concat(final_prediction_blocks, ignore_index=True)
    validation_predictions_frame = pd.concat(validation_predictions, ignore_index=True)
    validation_history_frame = pd.DataFrame(validation_history)
    epoch_selection = pd.DataFrame(epoch_selection_rows)
    training_audit = pd.DataFrame(training_audit_rows)
    per_peptide = pd.DataFrame(per_peptide_rows)
    c_history_frame = pd.DataFrame(c_history)

    _require(not output_dir.exists(), "output directory appeared during run")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    artifact_paths = {}
    csv_outputs = {
        "c_validation.csv": c_history_frame,
        "fold_membership.csv": inputs["fold_membership"],
        "weak_validation_predictions.csv": validation_predictions_frame,
        "weak_validation_history.csv": validation_history_frame,
        "epoch_selection.csv": epoch_selection,
        "training_audit.csv": training_audit,
        "retention_predictions.csv": final_predictions,
        "retention_metrics.csv": metrics,
        "aggregate_metrics.csv": aggregate_metrics,
        "paired_differences.csv": differences,
        "per_peptide_metrics.csv": per_peptide,
    }
    for name, frame in csv_outputs.items():
        path = output_dir / name
        _write_csv(frame, path)
        artifact_paths[name] = path
    for (training_seed, arm), state in sorted(saved_states.items()):
        name = "model_delta_seed{}_{}.pt".format(training_seed, arm)
        path = output_dir / name
        torch.save(state, str(path))
        os.chmod(str(path), 0o600)
        artifact_paths[name] = path

    summary_path = output_dir / "run_summary.md"
    lines = [
        "# Matched MINT LoRA rerun: {}".format(args.library),
        "",
        "- Primary weak-label rows: {:,} (all used for the final refit)".format(len(primary)),
        "- Validation: {} identity-double-cold folds, split seed {}".format(CANONICAL_FOLDS, CANONICAL_SPLIT_SEED),
        "- Frozen-head C: {}".format(selected_c),
        "- Retention panel: {} rows; 75% defines AUROC/AP positives only".format(len(retention)),
        "- Retention labels were not used for C or epoch selection.",
        "",
        "| Arm | Seeds | AUROC mean | AP mean | Global Spearman mean | Within-peptide Spearman mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in aggregate_metrics.iterrows():
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["arm"],
                int(row["n_seed_runs"]),
                row["global_auroc_mean"],
                row["global_auprc_mean"],
                row["global_spearman_mean"],
                row["within_peptide_macro_spearman_mean"],
            )
        )
    lines.extend(["", "This remains a retrospective exploratory analysis.", ""])
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(lines))
    os.chmod(str(summary_path), 0o600)
    artifact_paths[summary_path.name] = summary_path

    for name, path in source_paths.items():
        _require(sha256_file(path) == initial_hashes[name], "{} changed during run".format(name))
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "configuration": {
            "library": args.library,
            "regime": REGIME,
            "cleaning": CLEANING,
            "balance": BALANCE,
            "class_weight_formula": "N / (2 * N_class); weighted BCE sum divided by row count",
            "folds": CANONICAL_FOLDS,
            "split_seed": CANONICAL_SPLIT_SEED,
            "c_grid": list(CANONICAL_C_GRID),
            "selected_C": float(selected_c),
            "epoch_candidates": list(range(args.max_epochs + 1)),
            "epoch_selection": "minimum pooled weak-validation log loss; AP/AUROC/earlier-epoch tie-break",
            "training_seeds": list(training_seeds),
            "arms": ["frozen_logistic", "head_only", "lora_cross"],
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "accumulation_steps": int(args.accumulation_steps),
            "head_lr": float(args.head_lr),
            "adapter_lr": float(args.adapter_lr),
            "weight_decay": float(args.weight_decay),
            "warmup_fraction": float(args.warmup_fraction),
            "clip_norm": float(args.clip_norm),
            "lora": {
                "rank": int(args.lora_rank),
                "alpha": float(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "layers_zero_based": [31, 32],
                "projections": ["q_proj", "v_proj"],
            },
            "retention_threshold_for_auroc_ap": 75.0,
            "retention_usage": {
                "partner_identities": "used before fitting to construct the library-local peptide-and-Affibody-cold pool",
                "numeric_retention_and_75_percent_labels": "final metrics only; never used for C or epoch selection",
            },
        },
        "canonical_contract": PRIMARY_CONTRACT[args.library],
        "reference_primary_run": reference,
        "sources": {
            name: {"path": str(path.resolve()), "sha256": initial_hashes[name]}
            for name, path in source_paths.items()
        },
        "cache_sources": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for path in inputs["cache_paths"]
        ],
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "rows": {
            "primary": int(len(primary)),
            "primary_positive": int(labels.sum()),
            "primary_negative": int(np.sum(labels == 0)),
            "retention": int(len(retention)),
            "retention_positive": int(retention["target_binder"].sum()),
            "validation_prediction_records": int(len(validation_predictions_frame)),
            "final_prediction_records": int(len(final_predictions)),
        },
        "retention_feature_max_abs_difference": inputs["retention_feature_max_abs_difference"],
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "permissions": {"directory": "0700", "files": "0600"},
    }
    _write_json(manifest, manifest_path)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--cache-rows", type=Path, required=True)
    parser.add_argument("--cache-rows-manifest", type=Path, required=True)
    parser.add_argument("--retention-features", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--library", choices=LIBRARIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=list(DEFAULT_TRAINING_SEEDS))
    parser.add_argument("--split-seed", type=int, default=CANONICAL_SPLIT_SEED)
    parser.add_argument("--folds", type=int, default=CANONICAL_FOLDS)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(CANONICAL_C_GRID))
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-live-feature-probability-difference", type=float, default=0.01)
    parser.add_argument("--max-retention-feature-difference", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
