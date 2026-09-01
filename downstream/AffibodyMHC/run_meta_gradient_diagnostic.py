#!/usr/bin/env python
"""Leakage-safe gradient-alignment diagnostic for Affibody weak labels.

This is the deliberately small first stage of the proposed DataRater work.  It
does not train MINT.  It fits a 2,561-parameter linear classifier on cached,
fixed 2,560-dimensional MINT pair representations and asks whether weak-label
examples whose gradients agree with a clean retention-ranking gradient are
more useful than deterministic, size-matched random examples.

For every library and outer fold:

* peptide and Affibody identities assigned to the outer test block are absent
  from the weak training pool and the clean development-retention panel;
* the clean objective consists only of within-peptide retention comparisons in
  the development block;
* alignment selection preserves the positive and negative counts separately;
* five SHA-256-defined random controls are retrained at each retained fraction;
* the both-partners-held retention block is touched only for final scoring.

All outputs are private, immutable, and tied to exact input and source hashes.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "affibody-meta-gradient-diagnostic-v1"
FEATURE_NAME = "mint_chain_mean"
LIBRARIES = ("LibA", "LibB")
FRACTIONS = (0.25, 0.50, 0.75)
CHECKPOINTS = ("initial", "full_fit")
EXPECTED_CACHE_ROWS = 69945
EXPECTED_WEAK_ROWS = 69717
EXPECTED_MEASURED = {"LibA": 108, "LibB": 119}
EXPECTED_FEATURE_DIMENSION = 2560


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _membership_digest(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def _read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _validate_private_output(path):
    private_root = (REPO_ROOT / "private_data").resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError:
        raise ValueError("output must be under {}".format(private_root))
    return resolved


def _write_csv(frame, path):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_text(value, path):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    with open(str(path), "w") as handle:
        handle.write(value)
    os.chmod(str(path), 0o600)


def _hash_order(values, seed, library, axis):
    def key(value):
        payload = "{}|blocked3|{}|{}|{}".format(seed, library, axis, value)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), str(value)

    return sorted(set(map(str, values)), key=key)


def make_outer_roles(retention, library, fold, seed, fold_mode="diagonal3"):
    """Assign target-free identity blocks to development, guard, and test."""
    _require(fold_mode in ("diagonal3", "crossed9"), "unknown fold mode")
    maximum = 3 if fold_mode == "diagonal3" else 9
    _require(0 <= int(fold) < maximum, "outer fold is out of range")
    peptide_block = int(fold) if fold_mode == "diagonal3" else int(fold) // 3
    affibody_block = int(fold) if fold_mode == "diagonal3" else int(fold) % 3
    peptide_order = _hash_order(retention["peptide_uid"], seed, library, "peptide_uid")
    affibody_order = _hash_order(retention["affibody_uid"], seed, library, "affibody_uid")
    held_peptides = {
        value for index, value in enumerate(peptide_order) if index % 3 == peptide_block
    }
    held_affibodies = {
        value for index, value in enumerate(affibody_order) if index % 3 == affibody_block
    }
    peptide_held = retention["peptide_uid"].astype(str).isin(held_peptides).to_numpy()
    affibody_held = retention["affibody_uid"].astype(str).isin(held_affibodies).to_numpy()
    roles = np.full(len(retention), "guard", dtype="U11")
    roles[~peptide_held & ~affibody_held] = "development"
    roles[peptide_held & affibody_held] = "test"
    _require(bool(np.any(roles == "development")), "empty development block")
    _require(bool(np.any(roles == "test")), "empty test block")
    development = retention.loc[roles == "development"]
    test = retention.loc[roles == "test"]
    _require(
        set(development["peptide_uid"]).isdisjoint(set(test["peptide_uid"])),
        "peptide identity leaks into outer test",
    )
    _require(
        set(development["affibody_uid"]).isdisjoint(set(test["affibody_uid"])),
        "Affibody identity leaks into outer test",
    )
    return roles, held_peptides, held_affibodies, peptide_block, affibody_block


def weak_outer_roles(weak, held_peptides, held_affibodies):
    peptide_held = weak["peptide_uid"].astype(str).isin(held_peptides).to_numpy()
    affibody_held = weak["affibody_uid"].astype(str).isin(held_affibodies).to_numpy()
    roles = np.full(len(weak), "eligible", dtype="U27")
    roles[peptide_held & ~affibody_held] = "excluded_held_peptide"
    roles[~peptide_held & affibody_held] = "excluded_held_affibody"
    roles[peptide_held & affibody_held] = "excluded_both_held"
    return roles


def load_cache(cache_path, manifest_path):
    manifest = _read_json(manifest_path)
    _require(manifest.get("stage") == "merge", "cache manifest is not a merged artifact")
    _require(
        manifest.get("features", {}).get("name") == FEATURE_NAME,
        "unexpected cached feature name",
    )
    expected_hash = manifest.get("output", {}).get("sha256")
    _require(expected_hash == sha256_file(cache_path), "cache hash disagrees with manifest")
    required = {
        "row_index",
        "source_kind",
        "library",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_design_code",
        "affibody_design_code",
        "weak_label",
        "measurement_missing",
        "target_retention",
        "target_binder",
        "sequence_pair_sha256",
        FEATURE_NAME,
    }
    with np.load(str(cache_path), allow_pickle=False) as archive:
        _require(required.issubset(archive.files), "cache NPZ schema mismatch")
        arrays = {key: np.asarray(archive[key]).copy() for key in required if key != FEATURE_NAME}
        features = np.asarray(archive[FEATURE_NAME], dtype=np.float32).copy()
    _require(len(features) == EXPECTED_CACHE_ROWS, "unexpected cache row count")
    _require(
        features.shape == (EXPECTED_CACHE_ROWS, EXPECTED_FEATURE_DIMENSION),
        "unexpected MINT feature shape",
    )
    _require(bool(np.isfinite(features).all()), "non-finite MINT feature")
    frame = pd.DataFrame(
        {
            key: np.asarray(value, dtype=np.int64)
            if key == "row_index"
            else np.asarray(value).astype(str)
            for key, value in arrays.items()
        }
    )
    order = np.argsort(frame["row_index"].to_numpy(dtype=np.int64))
    _require(
        np.array_equal(frame.iloc[order]["row_index"].to_numpy(dtype=np.int64), np.arange(len(frame))),
        "cache row indices are not contiguous",
    )
    frame = frame.iloc[order].reset_index(drop=True)
    features = features[order]
    _require(not bool(frame["sequence_pair_sha256"].duplicated().any()), "duplicate sequence pair")
    return frame, features, manifest


def validate_trajectory(trajectory_path, trajectory_manifest_path, cache_frame):
    manifest = _read_json(trajectory_manifest_path)
    _require(
        manifest.get("schema_version") in ("selection-trajectories-v1", "selection-trajectories-v2"),
        "trajectory manifest schema mismatch",
    )
    expected = manifest.get("output", {}).get("npz", {}).get("sha256")
    _require(expected == sha256_file(trajectory_path), "trajectory hash disagrees with manifest")
    with np.load(str(trajectory_path), allow_pickle=False) as archive:
        required = {
            "cache_row_index",
            "pair_uid",
            "library",
            "weak_label",
            "trajectory_features",
            "trajectory_feature_names",
        }
        _require(required.issubset(archive.files), "trajectory NPZ schema mismatch")
        cache_index = np.asarray(archive["cache_row_index"], dtype=np.int64)
        pair_uid = np.asarray(archive["pair_uid"]).astype(str)
        trajectory_features = np.asarray(archive["trajectory_features"], dtype=np.float32)
        feature_names = np.asarray(archive["trajectory_feature_names"]).astype(str)
    weak = cache_frame.loc[cache_frame["source_kind"].eq("weak")]
    _require(len(cache_index) == EXPECTED_WEAK_ROWS, "trajectory weak-row count mismatch")
    _require(
        trajectory_features.shape == (EXPECTED_WEAK_ROWS, 63),
        "trajectory feature shape mismatch",
    )
    _require(bool(np.isfinite(trajectory_features).all()), "non-finite trajectory features")
    _require(len(feature_names) == 63, "trajectory feature-name count mismatch")
    _require(
        np.array_equal(cache_frame.iloc[cache_index]["pair_uid"].astype(str).to_numpy(), pair_uid),
        "trajectory/cache pair order mismatch",
    )
    _require(set(cache_index) == set(weak.index.to_numpy(dtype=int)), "trajectory misses weak cache rows")
    return manifest


def standardize_from_training(training_features, prediction_features):
    training_features = np.asarray(training_features, dtype=np.float32)
    mean = training_features.mean(axis=0, dtype=np.float64)
    scale = training_features.std(axis=0, dtype=np.float64, ddof=0)
    zero_variance = scale <= 0.0
    scale[zero_variance] = 1.0
    transformed_training = ((training_features - mean) / scale).astype(np.float32)
    transformed_prediction = ((np.asarray(prediction_features) - mean) / scale).astype(np.float32)
    _require(bool(np.isfinite(transformed_training).all()), "non-finite standardized training feature")
    _require(bool(np.isfinite(transformed_prediction).all()), "non-finite standardized prediction feature")
    return transformed_training, transformed_prediction, mean, scale, int(zero_variance.sum())


def make_pair_differences(frame, features):
    """Orient within-peptide feature differences from higher to lower retention."""
    _require(len(frame) == len(features), "pair-difference row mismatch")
    differences = []
    metadata = []
    local = frame.reset_index(drop=True)
    target = local["target_retention"].to_numpy(dtype=float)
    for peptide_uid, group in local.groupby("peptide_uid", sort=True):
        positions = group.index.to_numpy(dtype=int)
        for first, second in combinations(positions.tolist(), 2):
            if target[first] == target[second]:
                continue
            high, low = (first, second) if target[first] > target[second] else (second, first)
            differences.append(features[high] - features[low])
            metadata.append((str(peptide_uid), int(high), int(low)))
    _require(differences, "clean panel has no within-peptide ordered pairs")
    output = np.asarray(differences, dtype=np.float32)
    _require(bool(np.isfinite(output).all()), "non-finite pair differences")
    return output, metadata


def _import_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required; use the project GPU environment") from error
    return torch


def fit_linear_head(x, y, indices, l2, max_iter, class_balanced=True):
    """Fit a deterministic class-balanced logistic head with full-batch LBFGS."""
    torch = _import_torch()
    indices = np.asarray(indices, dtype=np.int64)
    _require(len(indices) > 0, "cannot fit an empty subset")
    selected = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    x_train = torch.index_select(x, 0, selected)
    y_train = torch.index_select(y, 0, selected)
    negative = int(torch.sum(y_train < 0.5).item())
    positive = int(torch.sum(y_train > 0.5).item())
    _require(negative > 0 and positive > 0, "head subset lacks a weak-label class")
    n = len(indices)
    positive_weight = float(n) / (2.0 * positive)
    negative_weight = float(n) / (2.0 * negative)
    if class_balanced:
        sample_weight = torch.where(
            y_train > 0.5,
            torch.full_like(y_train, positive_weight),
            torch.full_like(y_train, negative_weight),
        )
    else:
        sample_weight = torch.ones_like(y_train)
    weight = torch.zeros(x.shape[1], dtype=torch.float32, device=x.device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        lr=1.0,
        max_iter=int(max_iter),
        max_eval=int(max_iter) * 2,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    closure_calls = [0]

    def objective(backward):
        logits = torch.mv(x_train, weight) + bias
        data_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_train, reduction="none"
        )
        loss = torch.mean(data_loss * sample_weight) + 0.5 * float(l2) * torch.sum(weight.square())
        if backward:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        return loss

    initial_loss = float(objective(False).detach().cpu().item())

    def closure():
        closure_calls[0] += 1
        return objective(True)

    optimizer.step(closure)
    final_loss_tensor = objective(True)
    gradient_norm = math.sqrt(
        float(torch.sum(weight.grad.square()).detach().cpu().item())
        + float(bias.grad.square().detach().cpu().item())
    )
    result = {
        "weight": weight.detach().clone(),
        "bias": bias.detach().clone(),
        "initial_objective": initial_loss,
        "final_objective": float(final_loss_tensor.detach().cpu().item()),
        "gradient_norm": float(gradient_norm),
        "closure_calls": int(closure_calls[0]),
        "n_negative": negative,
        "n_positive": positive,
        "class_balanced": bool(class_balanced),
    }
    del x_train, y_train, sample_weight, optimizer, weight, bias
    return result


def clean_ranking_gradient(pair_differences, weight):
    """Gradient of mean softplus(-higher_minus_lower_score) with respect to weight."""
    torch = _import_torch()
    margins = torch.mv(pair_differences, weight)
    coefficients = -torch.sigmoid(-margins)
    gradient = torch.mean(coefficients[:, None] * pair_differences, dim=0)
    _require(bool(torch.isfinite(gradient).all().item()), "non-finite clean gradient")
    return gradient


def weak_alignment_scores(x, y, weight, bias, clean_gradient):
    """Return g_clean dot g_i for each class-balanced weak-example gradient."""
    torch = _import_torch()
    n = len(y)
    positive = int(torch.sum(y > 0.5).item())
    negative = n - positive
    positive_weight = float(n) / (2.0 * positive)
    negative_weight = float(n) / (2.0 * negative)
    class_weight = torch.where(
        y > 0.5,
        torch.full_like(y, positive_weight),
        torch.full_like(y, negative_weight),
    )
    residual = (torch.sigmoid(torch.mv(x, weight) + bias) - y) * class_weight
    directional_feature = torch.mv(x, clean_gradient)
    alignment = residual * directional_feature
    _require(bool(torch.isfinite(alignment).all().item()), "non-finite alignment score")
    return alignment.detach().cpu().numpy().astype(np.float64)


def within_class_percentile(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    output = np.zeros(len(scores), dtype=np.float64)
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        _require(len(positions) > 0, "missing class in percentile ranking")
        output[positions] = rankdata(scores[positions], method="average") / float(len(positions))
    return output


def deterministic_top_indices(scores, labels, pair_uids, fraction):
    selected = []
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pair_uids = np.asarray(pair_uids).astype(str)
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        count = max(1, int(math.floor(float(fraction) * len(positions))))
        order = sorted(positions.tolist(), key=lambda index: (-scores[index], pair_uids[index]))
        selected.extend(order[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def deterministic_random_indices(labels, pair_uids, fraction, seed, library, fold, control):
    selected = []
    labels = np.asarray(labels, dtype=int)
    pair_uids = np.asarray(pair_uids).astype(str)
    for label in (0, 1):
        positions = np.flatnonzero(labels == label)
        count = max(1, int(math.floor(float(fraction) * len(positions))))

        def key(index):
            payload = "{}|random-control|{}|{}|{}|{}|{}|{}".format(
                seed, library, fold, control, fraction, label, pair_uids[index]
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest(), pair_uids[index]

        selected.extend(sorted(positions.tolist(), key=key)[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def _safe_spearman(observed, score):
    observed = np.asarray(observed, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(observed) < 2 or np.unique(observed).size < 2 or np.unique(score).size < 2:
        return np.nan
    return float(spearmanr(observed, score)[0])


def ranking_metrics(frame, logits):
    frame = frame.reset_index(drop=True).copy()
    logits = np.asarray(logits, dtype=float)
    _require(len(frame) == len(logits), "metric row mismatch")
    frame["score"] = logits
    target = frame["target_retention"].to_numpy(dtype=float)
    binder = frame["target_binder"].to_numpy(dtype=int)
    probability = expit(logits)
    metrics = {
        "n": int(len(frame)),
        "positive": int(binder.sum()),
        "prevalence": float(binder.mean()),
        "overall_spearman": _safe_spearman(target, logits),
        "auroc": float(roc_auc_score(binder, probability)) if np.unique(binder).size == 2 else np.nan,
        "average_precision": float(average_precision_score(binder, probability))
        if int(binder.sum()) > 0
        else np.nan,
    }
    micro_correct = 0.0
    micro_pairs = 0
    peptide_accuracies = []
    peptide_spearman = []
    top_all = []
    top_binder_available = []
    for _, group in frame.groupby("peptide_uid", sort=True):
        local_correct = 0.0
        local_pairs = 0
        indices = group.index.to_numpy(dtype=int)
        for first, second in combinations(indices.tolist(), 2):
            target_difference = target[first] - target[second]
            if target_difference == 0.0:
                continue
            prediction_difference = logits[first] - logits[second]
            local_correct += 1.0 if prediction_difference * target_difference > 0.0 else (
                0.5 if prediction_difference == 0.0 else 0.0
            )
            local_pairs += 1
        if local_pairs:
            peptide_accuracies.append(local_correct / local_pairs)
            micro_correct += local_correct
            micro_pairs += local_pairs
        rho = _safe_spearman(group["target_retention"], group["score"])
        if np.isfinite(rho):
            peptide_spearman.append(rho)
        chosen = group.sort_values(["score", "affibody_uid"], ascending=[False, True]).iloc[0]
        success = float(int(chosen["target_binder"]))
        top_all.append(success)
        if bool(group["target_binder"].astype(int).eq(1).any()):
            top_binder_available.append(success)
    metrics.update(
        {
            "within_peptide_pairwise_accuracy_micro": float(micro_correct / micro_pairs)
            if micro_pairs
            else np.nan,
            "within_peptide_pair_count": int(micro_pairs),
            "within_peptide_pairwise_accuracy_macro": float(np.mean(peptide_accuracies))
            if peptide_accuracies
            else np.nan,
            "within_peptide_macro_spearman": float(np.mean(peptide_spearman))
            if peptide_spearman
            else np.nan,
            "within_peptide_spearman_groups": int(len(peptide_spearman)),
            "p_at_1_all_peptides": float(np.mean(top_all)) if top_all else np.nan,
            "p_at_1_binder_available": float(np.mean(top_binder_available))
            if top_binder_available
            else np.nan,
            "p_at_1_binder_available_groups": int(len(top_binder_available)),
        }
    )
    return metrics


def checkpoint_diagnostic_rows(library, fold, labels, initial, fitted):
    rows = []
    for label in (0, 1):
        mask = np.asarray(labels, dtype=int) == label
        first = np.asarray(initial)[mask]
        second = np.asarray(fitted)[mask]
        rho = _safe_spearman(first, second)
        n = int(mask.sum())
        k = max(1, int(math.floor(0.25 * n)))
        first_top = set(np.argsort(-first, kind="mergesort")[:k].tolist())
        second_top = set(np.argsort(-second, kind="mergesort")[:k].tolist())
        overlap = len(first_top & second_top)
        union = len(first_top | second_top)
        rows.append(
            {
                "library": library,
                "fold": int(fold),
                "weak_label": int(label),
                "n": n,
                "checkpoint_spearman": rho,
                "top25_overlap_count": int(overlap),
                "top25_overlap_fraction": float(overlap / k),
                "top25_jaccard": float(overlap / union),
                "independent_random_expected_overlap_fraction": 0.25,
            }
        )
    return rows


def summarize_comparisons(metrics, random_controls, selected_method="alignment"):
    primary = "within_peptide_pairwise_accuracy_micro"
    rows = []
    for (library, fold, fraction), top in metrics.loc[metrics["method"].eq(selected_method)].groupby(
        ["library", "fold", "fraction"], sort=True
    ):
        _require(len(top) == 1, "duplicate alignment condition")
        controls = metrics.loc[
            metrics["library"].eq(library)
            & metrics["fold"].eq(fold)
            & metrics["fraction"].eq(fraction)
            & metrics["method"].eq("random")
        ]
        _require(len(controls) == int(random_controls), "missing random control")
        top_value = float(top.iloc[0][primary])
        random_values = controls[primary].to_numpy(dtype=float)
        rows.append(
            {
                "library": library,
                "fold": int(fold),
                "fraction": float(fraction),
                "metric": primary,
                "alignment": top_value,
                "random_mean": float(np.mean(random_values)),
                "random_sd": float(np.std(random_values, ddof=1)),
                "alignment_minus_random_mean": float(top_value - np.mean(random_values)),
                "alignment_beats_random_mean": int(top_value > np.mean(random_values)),
                "random_controls": int(len(random_values)),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_gate(comparisons):
    rows = []
    library_pass = {}
    for library, library_frame in comparisons.groupby("library", sort=True):
        passing_fractions = 0
        for fraction, group in library_frame.groupby("fraction", sort=True):
            delta = group["alignment_minus_random_mean"].to_numpy(dtype=float)
            positive_folds = int(np.sum(delta > 0.0))
            required_positive_folds = int(math.ceil(2.0 * len(delta) / 3.0))
            fraction_pass = bool(
                float(np.mean(delta)) > 0.0
                and positive_folds >= required_positive_folds
            )
            passing_fractions += int(fraction_pass)
            rows.append(
                {
                    "library": library,
                    "fraction": float(fraction),
                    "mean_alignment_minus_random": float(np.mean(delta)),
                    "positive_folds": positive_folds,
                    "required_positive_folds": required_positive_folds,
                    "total_folds": int(len(delta)),
                    "same_positive_sign_all_folds": int(positive_folds == len(delta)),
                    "fraction_pass": int(fraction_pass),
                }
            )
        library_pass[library] = bool(passing_fractions >= 2)
    return pd.DataFrame(rows), library_pass


def _format_number(value):
    return "NA" if not np.isfinite(float(value)) else "{:.3f}".format(float(value))


def build_summary(args, metrics, comparison_summary, library_pass, checkpoint_rows, elapsed):
    lines = [
        "# Leakage-safe meta-gradient diagnostic",
        "",
        "This pilot tested whether weak selection examples chosen by gradient alignment "
        "beat five deterministic, class-matched random subsets of the same size. MINT "
        "was frozen; every fitted model was only a 2,561-parameter linear head.",
        "",
        "The clean ranking signal came from retention measurements in the development "
        "identity block. The reported results came from a different block in which both "
        "the peptide and Affibody identities were held out. Rows sharing either held test "
        "identity were excluded from weak-label training.",
        "",
        "## Primary control comparison",
        "",
        "The number is the fraction of within-peptide retention orderings ranked correctly. "
        "`Delta` is alignment selection minus the mean of five random controls.",
        "",
        "| Library | Weak rows kept | Alignment | Random mean | Delta | Positive folds | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in comparison_summary.sort_values(["library", "fraction"]).iterrows():
        library = row["library"]
        fraction = float(row["fraction"])
        selected = metrics.loc[
            metrics["library"].eq(library)
            & metrics["method"].eq("alignment")
            & metrics["fraction"].eq(fraction),
            "within_peptide_pairwise_accuracy_micro",
        ]
        controls = metrics.loc[
            metrics["library"].eq(library)
            & metrics["method"].eq("random")
            & metrics["fraction"].eq(fraction),
            "within_peptide_pairwise_accuracy_micro",
        ]
        lines.append(
            "| {} | {:.0f}% | {} | {} | {} | {}/{} | {} |".format(
                library,
                100.0 * fraction,
                _format_number(selected.mean()),
                _format_number(controls.mean()),
                _format_number(row["mean_alignment_minus_random"]),
                int(row["positive_folds"]),
                int(row["total_folds"]),
                "pass" if int(row["fraction_pass"]) else "fail",
            )
        )
    lines.extend(
        [
            "",
        "A library passes the fixed pilot gate only if at least two of the three kept "
            "fractions have a positive mean delta and beat the random mean in at least "
            "two thirds of the outer folds.",
            "",
            "## Full-data reference",
            "",
            "| Library | Pairwise accuracy | Overall Spearman | Within-peptide Spearman | P@1 (binder available) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    gate_lines = [
        "- {} gate: **{}**".format(
            library, "pass" if library_pass.get(library) else "fail"
        )
        for library in sorted(library_pass)
    ]
    # Insert selected-library gate lines immediately before the full-data table.
    insertion = lines.index("## Full-data reference")
    lines[insertion:insertion] = gate_lines + [""]
    full = metrics.loc[metrics["method"].eq("full")]
    for library, group in full.groupby("library", sort=True):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                library,
                _format_number(group["within_peptide_pairwise_accuracy_micro"].mean()),
                _format_number(group["overall_spearman"].mean()),
                _format_number(group["within_peptide_macro_spearman"].mean()),
                _format_number(group["p_at_1_binder_available"].mean()),
            )
        )
    lines.extend(
        [
            "",
            "## Checkpoint stability",
            "",
            "Alignment was calculated at zero initialization and after fitting on all eligible "
            "weak rows. The selection score averages the within-class percentile ranks from "
            "those two checkpoints.",
            "",
            "| Library | Median rank correlation | Median top-25% overlap |",
            "|---|---:|---:|",
        ]
    )
    for library, group in checkpoint_rows.groupby("library", sort=True):
        lines.append(
            "| {} | {} | {} |".format(
                library,
                _format_number(group["checkpoint_spearman"].median()),
                _format_number(group["top25_overlap_fraction"].median()),
            )
        )
    lines.extend(
        [
            "",
            (
                "This is an exploratory {}-model diagnostic on a small retention matrix, not a "
                "prospective validation. The crossed blocks reuse overlapping development and "
                "weak-label pools, so their results are correlated rather than nine independent "
                "replicates. A second-order trajectory rater should be run only for libraries "
                "that pass the control gate above."
            ).format(len(args.libraries) * (3 if args.fold_mode == "diagonal3" else 9)),
            "",
            "Elapsed time: {:.1f} seconds.".format(elapsed),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/merged/manifest.json",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/selection_trajectories_v2/selection_trajectories.npz",
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/selection_trajectories_v2/manifest.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES))
    parser.add_argument(
        "--fold-mode", choices=("diagonal3", "crossed9"), default="diagonal3"
    )
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--l2", type=float, default=0.005)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    _require(int(args.random_controls) >= 5, "at least five random controls are required")
    _require(float(args.l2) > 0.0 and np.isfinite(args.l2), "L2 must be positive")
    _require(int(args.max_iter) >= 2, "max_iter must be at least two")
    for path, label in (
        (args.cache, "cache"),
        (args.cache_manifest, "cache manifest"),
        (args.trajectory, "trajectory"),
        (args.trajectory_manifest, "trajectory manifest"),
    ):
        _require(path.is_file(), "{} does not exist: {}".format(label, path))
    output_dir = _validate_private_output(args.output_dir)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")

    # CuBLAS requires this workspace contract for deterministic GEMV/GEMM on
    # CUDA >= 10.2.  Set it before the first CUDA operation in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _import_torch()
    _require(torch.cuda.is_available() or not str(args.device).startswith("cuda"), "CUDA unavailable")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    cache_frame, cache_features, cache_manifest = load_cache(args.cache, args.cache_manifest)
    trajectory_manifest = validate_trajectory(
        args.trajectory, args.trajectory_manifest, cache_frame
    )
    weak_all = cache_frame.loc[cache_frame["source_kind"].eq("weak")].copy()
    retention_all = cache_frame.loc[cache_frame["source_kind"].eq("retention")].copy()
    weak_all["weak_label"] = pd.to_numeric(weak_all["weak_label"], errors="raise").astype(int)
    retention_all["measurement_missing"] = pd.to_numeric(
        retention_all["measurement_missing"], errors="raise"
    ).astype(int)
    retention_all = retention_all.loc[retention_all["measurement_missing"].eq(0)].copy()
    retention_all["target_retention"] = pd.to_numeric(
        retention_all["target_retention"], errors="raise"
    )
    retention_all["target_binder"] = pd.to_numeric(
        retention_all["target_binder"], errors="raise"
    ).astype(int)
    _require(len(weak_all) == EXPECTED_WEAK_ROWS, "unexpected weak row count")
    _require(set(weak_all["weak_label"]) == {0, 1}, "weak cache lacks a class")
    _require(
        set(weak_all["sequence_pair_sha256"]).isdisjoint(
            set(retention_all["sequence_pair_sha256"])
        ),
        "exact retention pairs occur in weak labels",
    )

    retention_membership_rows = []
    weak_membership_rows = []
    alignment_rows = []
    checkpoint_rows = []
    condition_rows = []
    metric_rows = []
    prediction_rows = []

    selected_libraries = tuple(args.libraries)
    _require(len(set(selected_libraries)) == len(selected_libraries), "duplicate library argument")
    outer_folds = 3 if args.fold_mode == "diagonal3" else 9
    for library in selected_libraries:
        weak_library = weak_all.loc[weak_all["library"].eq(library)].copy()
        retention_library = retention_all.loc[retention_all["library"].eq(library)].copy()
        _require(
            len(retention_library) == EXPECTED_MEASURED[library],
            "{} measured-row mismatch".format(library),
        )
        for fold in range(outer_folds):
            print("{} fold {}: preparing identity-held data".format(library, fold), flush=True)
            retention_roles, held_peptides, held_affibodies, peptide_block, affibody_block = make_outer_roles(
                retention_library, library, fold, args.seed, args.fold_mode
            )
            for position, (_, row) in enumerate(retention_library.iterrows()):
                retention_membership_rows.append(
                    {
                        "library": library,
                        "fold": int(fold),
                        "pair_uid": row["pair_uid"],
                        "peptide_uid": row["peptide_uid"],
                        "affibody_uid": row["affibody_uid"],
                        "role": retention_roles[position],
                        "peptide_block": int(peptide_block),
                        "affibody_block": int(affibody_block),
                        "target_hidden_during_selection": int(retention_roles[position] != "development"),
                    }
                )
            weak_roles = weak_outer_roles(weak_library, held_peptides, held_affibodies)
            for position, (_, row) in enumerate(weak_library.iterrows()):
                weak_membership_rows.append(
                    {
                        "library": library,
                        "fold": int(fold),
                        "cache_row_index": int(row["row_index"]),
                        "pair_uid": row["pair_uid"],
                        "peptide_uid": row["peptide_uid"],
                        "affibody_uid": row["affibody_uid"],
                        "weak_label": int(row["weak_label"]),
                        "role": weak_roles[position],
                        "peptide_block": int(peptide_block),
                        "affibody_block": int(affibody_block),
                    }
                )
            weak = weak_library.loc[weak_roles == "eligible"].copy().reset_index(drop=True)
            development = retention_library.loc[
                retention_roles == "development"
            ].copy().reset_index(drop=True)
            test = retention_library.loc[retention_roles == "test"].copy().reset_index(drop=True)
            _require(set(weak["weak_label"]) == {0, 1}, "eligible weak pool lacks a class")
            _require(
                set(weak["peptide_uid"]).isdisjoint(set(test["peptide_uid"])),
                "eligible weak pool leaks held peptide",
            )
            _require(
                set(weak["affibody_uid"]).isdisjoint(set(test["affibody_uid"])),
                "eligible weak pool leaks held Affibody",
            )
            cache_indices = weak["row_index"].to_numpy(dtype=int)
            retention_indices = np.concatenate(
                [
                    development["row_index"].to_numpy(dtype=int),
                    test["row_index"].to_numpy(dtype=int),
                ]
            )
            x_weak, x_retention, _, _, zero_variance = standardize_from_training(
                cache_features[cache_indices], cache_features[retention_indices]
            )
            x_development = x_retention[: len(development)]
            x_test = x_retention[len(development) :]
            clean_differences, clean_pairs = make_pair_differences(development, x_development)
            x_tensor = torch.as_tensor(x_weak, dtype=torch.float32, device=device)
            y_numpy = weak["weak_label"].to_numpy(dtype=int)
            y_tensor = torch.as_tensor(y_numpy, dtype=torch.float32, device=device)
            clean_tensor = torch.as_tensor(clean_differences, dtype=torch.float32, device=device)
            test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
            all_indices = np.arange(len(weak), dtype=np.int64)

            print(
                "{} fold {}: fitting full head on {:,} weak pairs".format(
                    library, fold, len(weak)
                ),
                flush=True,
            )
            full_model = fit_linear_head(
                x_tensor, y_tensor, all_indices, args.l2, args.max_iter
            )
            zero_weight = torch.zeros_like(full_model["weight"])
            zero_bias = torch.zeros_like(full_model["bias"])
            initial_clean_gradient = clean_ranking_gradient(clean_tensor, zero_weight)
            fitted_clean_gradient = clean_ranking_gradient(
                clean_tensor, full_model["weight"]
            )
            initial_alignment = weak_alignment_scores(
                x_tensor, y_tensor, zero_weight, zero_bias, initial_clean_gradient
            )
            fitted_alignment = weak_alignment_scores(
                x_tensor,
                y_tensor,
                full_model["weight"],
                full_model["bias"],
                fitted_clean_gradient,
            )
            combined_alignment = 0.5 * (
                within_class_percentile(initial_alignment, y_numpy)
                + within_class_percentile(fitted_alignment, y_numpy)
            )
            checkpoint_rows.extend(
                checkpoint_diagnostic_rows(
                    library, fold, y_numpy, initial_alignment, fitted_alignment
                )
            )
            for index, (_, row) in enumerate(weak.iterrows()):
                alignment_rows.append(
                    {
                        "library": library,
                        "fold": int(fold),
                        "cache_row_index": int(row["row_index"]),
                        "pair_uid": row["pair_uid"],
                        "weak_label": int(y_numpy[index]),
                        "initial_alignment": float(initial_alignment[index]),
                        "full_fit_alignment": float(fitted_alignment[index]),
                        "combined_within_class_percentile": float(combined_alignment[index]),
                    }
                )
            pair_uids = weak["pair_uid"].astype(str).to_numpy()
            conditions = [("full", 1.0, -1, all_indices, full_model)]
            for fraction in FRACTIONS:
                selected = deterministic_top_indices(
                    combined_alignment, y_numpy, pair_uids, fraction
                )
                conditions.append(("alignment", fraction, -1, selected, None))
                for control in range(int(args.random_controls)):
                    random_selected = deterministic_random_indices(
                        y_numpy,
                        pair_uids,
                        fraction,
                        args.seed,
                        library,
                        fold,
                        control,
                    )
                    conditions.append(("random", fraction, control, random_selected, None))
            seen_memberships = {}
            for method, fraction, control, selected, prefit in conditions:
                membership = _membership_digest(pair_uids[selected])
                condition_key = (method, float(fraction), int(control))
                _require(condition_key not in seen_memberships, "duplicate condition key")
                seen_memberships[condition_key] = membership
                model = prefit
                if model is None:
                    print(
                        "{} fold {}: fitting {} {:.0f}% control {} ({:,} rows)".format(
                            library,
                            fold,
                            method,
                            100.0 * float(fraction),
                            control,
                            len(selected),
                        ),
                        flush=True,
                    )
                    model = fit_linear_head(
                        x_tensor, y_tensor, selected, args.l2, args.max_iter
                    )
                logits = (
                    torch.mv(test_tensor, model["weight"]) + model["bias"]
                ).detach().cpu().numpy().astype(np.float64)
                metadata = {
                    "library": library,
                    "fold": int(fold),
                    "method": method,
                    "fraction": float(fraction),
                    "control": int(control),
                }
                condition_rows.append(
                    dict(
                        metadata,
                        weak_eligible=int(len(weak)),
                        weak_selected=int(len(selected)),
                        weak_selected_negative=int(np.sum(y_numpy[selected] == 0)),
                        weak_selected_positive=int(np.sum(y_numpy[selected] == 1)),
                        selected_membership_sha256=membership,
                        weak_eligible_membership_sha256=_membership_digest(pair_uids),
                        development_retention=int(len(development)),
                        development_pairwise_comparisons=int(len(clean_pairs)),
                        test_retention=int(len(test)),
                        standardized_zero_variance_features=int(zero_variance),
                        initial_objective=float(model["initial_objective"]),
                        final_objective=float(model["final_objective"]),
                        final_gradient_norm=float(model["gradient_norm"]),
                        lbfgs_closure_calls=int(model["closure_calls"]),
                    )
                )
                metric_rows.append(dict(metadata, **ranking_metrics(test, logits)))
                for row_position, (_, row) in enumerate(test.iterrows()):
                    prediction_rows.append(
                        dict(
                            metadata,
                            pair_uid=row["pair_uid"],
                            peptide_uid=row["peptide_uid"],
                            affibody_uid=row["affibody_uid"],
                            peptide_design_code=row["peptide_design_code"],
                            affibody_design_code=row["affibody_design_code"],
                            target_retention=float(row["target_retention"]),
                            target_binder=int(row["target_binder"]),
                            score=float(logits[row_position]),
                            binder_probability=float(expit(logits[row_position])),
                        )
                    )
            for fraction in FRACTIONS:
                random_hashes = {
                    seen_memberships[("random", float(fraction), control)]
                    for control in range(int(args.random_controls))
                }
                _require(
                    len(random_hashes) == int(args.random_controls),
                    "random controls duplicate exact memberships",
                )
            del x_tensor, y_tensor, clean_tensor, test_tensor, full_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    conditions = pd.DataFrame(condition_rows)
    predictions = pd.DataFrame(prediction_rows)
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    comparisons = summarize_comparisons(metrics, args.random_controls)
    comparison_summary, library_pass = diagnostic_gate(comparisons)
    elapsed = float(time.time() - started)
    summary = build_summary(
        args, metrics, comparison_summary, library_pass, checkpoint_frame, elapsed
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    frames = {
        "metrics.csv": metrics,
        "conditions.csv": conditions,
        "predictions.csv": predictions,
        "retention_split_membership.csv": pd.DataFrame(retention_membership_rows),
        "weak_split_membership.csv": pd.DataFrame(weak_membership_rows),
        "alignment_scores.csv": pd.DataFrame(alignment_rows),
        "checkpoint_stability.csv": checkpoint_frame,
        "matched_random_comparisons.csv": comparisons,
        "diagnostic_gate.csv": comparison_summary,
    }
    output_paths = {}
    for name, frame in frames.items():
        path = output_dir / name
        _write_csv(frame, path)
        output_paths[name] = path
    summary_path = output_dir / "run_summary.md"
    _write_text(summary, summary_path)
    output_paths[summary_path.name] = summary_path

    script_path = Path(__file__).resolve()
    source_snapshot_path = output_dir / "producing_source_snapshot.py"
    with open(str(script_path), "r") as handle:
        _write_text(handle.read(), source_snapshot_path)
    output_paths[source_snapshot_path.name] = source_snapshot_path
    _require(
        sha256_file(source_snapshot_path) == sha256_file(script_path),
        "source snapshot hash mismatch",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": int(time.time()),
        "elapsed_seconds": elapsed,
        "host": socket.gethostname(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "protocol": {
            "libraries": list(selected_libraries),
            "outer_folds": int(outer_folds),
            "fold_mode": args.fold_mode,
            "split": "hash-block peptide and Affibody identities; development=neither held, test=both held, guard=exactly one held; crossed9 uses every peptide-block x Affibody-block combination",
            "clean_objective": "mean within-peptide pairwise softplus ranking loss on development retention only",
            "weak_pool": "cached weak rows excluding every row that shares either held test identity",
            "head": "class-balanced L2 logistic linear head on fixed MINT chain-mean features",
            "head_parameters": EXPECTED_FEATURE_DIMENSION + 1,
            "alignment": "g_clean dot g_weak at zero and full-fit checkpoints; mean within-class percentile",
            "fractions": list(FRACTIONS),
            "random_controls": int(args.random_controls),
            "primary_metric": "within_peptide_pairwise_accuracy_micro",
            "gate": "fixed pilot gate: at least two fractions have positive mean alignment-minus-random and positive delta in at least two thirds of folds; crossed folds are correlated, not independent replicates",
            "seed": int(args.seed),
            "l2": float(args.l2),
            "lbfgs_max_iter": int(args.max_iter),
            "test_retention_used_for_selection": False,
        },
        "gate_result": {library: bool(value) for library, value in library_pass.items()},
        "input": {
            "cache": {"path": str(args.cache.resolve()), "sha256": sha256_file(args.cache)},
            "cache_manifest": {
                "path": str(args.cache_manifest.resolve()),
                "sha256": sha256_file(args.cache_manifest),
                "schema_version": cache_manifest.get("contract", {}).get("payload", {}).get(
                    "schema_version"
                ),
            },
            "trajectory": {
                "path": str(args.trajectory.resolve()),
                "sha256": sha256_file(args.trajectory),
            },
            "trajectory_manifest": {
                "path": str(args.trajectory_manifest.resolve()),
                "sha256": sha256_file(args.trajectory_manifest),
                "schema_version": trajectory_manifest.get("schema_version"),
            },
        },
        "source": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "source_snapshot": {
            "path": str(source_snapshot_path.resolve()),
            "sha256": sha256_file(source_snapshot_path),
        },
        "optimizer_audit": {
            "maximum_final_gradient_norm": float(conditions["final_gradient_norm"].max()),
            "median_final_gradient_norm": float(conditions["final_gradient_norm"].median()),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "mode": _private_mode(path),
                "rows": int(len(frames[name])) if name in frames else None,
            }
            for name, path in output_paths.items()
        },
    }
    _write_json(manifest, output_dir / "manifest.json")
    _require(_private_mode(output_dir) == "0700", "output directory is not private")
    for path in output_dir.iterdir():
        _require(_private_mode(path) == "0600", "output file is not private: {}".format(path))
    print(json.dumps({"gate_result": manifest["gate_result"], "elapsed_seconds": elapsed}, indent=2))
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
