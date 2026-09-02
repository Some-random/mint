#!/usr/bin/env python
"""Train frozen-MINT PN, PU, and PNU linear readouts.

This program is deliberately limited to the 2,560-dimensional, layer-33 MINT
features produced by the Affibody project.  It does not fine-tune MINT.  The
three label groups are kept separate throughout optimization:

``P``
    the established pooled-R009/R010 top-count positives;
``N``
    the established conservative R001-derived negatives; and
``U``
    pooled-R009/R010 pairs below the positive cutoff, whose binder status is
    unknown.

For an assumed positive prior ``pi`` and mixing value ``eta``, the implemented
non-negative PNU risk is

    pi Rp+ + max(0, (1-eta)(1-pi) Rn- + eta(Ru- - pi Rp-)).

Thus eta=0 is prior-weighted PN and eta=1 is non-negative PU.  In addition, an
unchanged class-balanced sklearn logistic regression on P versus N is retained
as the exact frozen-MINT control used by the existing project pipeline.

Hyperparameters are chosen only from double-identity-cold P/N weak-label
validation folds.  Numeric retention values and the 75-percent binder labels
are quarantined until every weak-label choice has been made.  The final
retention panel is used only for retrospective evaluation.
"""

from __future__ import print_function

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "affibody-frozen-mint-pnu-readout-v1"
FEATURE_KEY = "mint_chain_mean"
FEATURE_DIMENSION = 2560
ROLES = ("P", "N", "U")
DEFAULT_PI_GRID = (0.02, 0.05, 0.10, 0.15)
DEFAULT_ETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0)
DEFAULT_EPOCH_GRID = (1, 2, 4, 8, 16)
DEFAULT_LIBRARIES = ("LibA", "LibB")

CANONICAL_COLUMNS = (
    "library",
    "pnu_role",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_float(value, name):
    output = float(value)
    _require(math.isfinite(output), "{} must be finite".format(name))
    return output


def _read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _atomic_csv(frame, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        frame.to_csv(str(temporary), index=False)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        with open(str(temporary), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(arrays, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        with open(str(temporary), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_alias(frame, canonical, aliases, required=True):
    if canonical in frame.columns:
        return canonical
    matches = [name for name in aliases if name in frame.columns]
    if not matches:
        _require(not required, "missing column {}".format(canonical))
        return None
    _require(len(matches) == 1, "ambiguous aliases for {}".format(canonical))
    frame.rename(columns={matches[0]: canonical}, inplace=True)
    return canonical


def canonicalize_pnu_rows(frame):
    """Return the common P/N/U metadata schema without outcome columns."""
    output = frame.copy()
    _canonical_alias(output, "pnu_role", ("class_source", "label_role", "role"))
    _canonical_alias(output, "peptide_design_code", ("pep", "peptide_code"))
    _canonical_alias(output, "affibody_design_code", ("aff", "affibody_code"))
    _canonical_alias(output, "chain1_sha256", ("peptide_identity", "chain_1_sha256"))
    _canonical_alias(output, "chain2_sha256", ("affibody_identity", "chain_2_sha256"))
    _canonical_alias(output, "sequence_pair_sha256", ("pair_identity",), required=False)
    _canonical_alias(output, "pair_uid", ("row_id", "cache_uid"))
    _canonical_alias(output, "peptide_uid", (), required=False)
    _canonical_alias(output, "affibody_uid", (), required=False)
    if "sequence_pair_sha256" not in output:
        output["sequence_pair_sha256"] = output["pair_uid"].astype(str)
    if "peptide_uid" not in output:
        output["peptide_uid"] = output["chain1_sha256"].astype(str)
    if "affibody_uid" not in output:
        output["affibody_uid"] = output["chain2_sha256"].astype(str)
    missing = set(CANONICAL_COLUMNS).difference(output.columns)
    _require(not missing, "PNU rows missing columns {}".format(sorted(missing)))

    # Any non-empty outcome value in a training row is an immediate failure.
    # Empty compatibility columns in the old immutable cache are harmless.
    for column in ("target_retention", "target_binder", "retention"):
        if column in output:
            values = output[column].astype(str).str.strip()
            bad = ~values.isin(("", "nan", "NaN", "NA", "None"))
            _require(not bool(bad.any()), "outcome values found in P/N/U training rows")

    output = output.loc[:, list(CANONICAL_COLUMNS)].copy()
    for column in CANONICAL_COLUMNS:
        output[column] = output[column].astype(str)
    output["pnu_role"] = output["pnu_role"].str.upper()
    _require(set(output["pnu_role"]).issubset(set(ROLES)), "unknown PNU role")
    _require(not bool(output["pair_uid"].duplicated().any()), "duplicate PNU pair_uid")
    _require(
        not bool(output["sequence_pair_sha256"].duplicated().any()),
        "duplicate PNU sequence pair",
    )
    return output.reset_index(drop=True)


def _resolve_npz(path):
    path = Path(path)
    if path.is_file():
        return path
    _require(path.is_dir(), "PNU cache path does not exist")
    candidates = []
    for pattern in (
        "mint_chain_mean_features.npz",
        "pnu_mint_chain_mean_features.npz",
        "merged/*.npz",
        "*.npz",
    ):
        candidates.extend(path.glob(pattern))
    candidates = sorted(set(item.resolve() for item in candidates if item.is_file()))
    _require(len(candidates) == 1, "could not uniquely resolve PNU cache NPZ")
    return candidates[0]


def _resolve_rows(npz_path, rows_path=None):
    if rows_path is not None:
        path = Path(rows_path)
        _require(path.is_file(), "PNU rows CSV is missing")
        return path
    candidates = (
        npz_path.parent / "cache_rows.csv",
        npz_path.parent / "cache_rows.csv.gz",
        npz_path.parent / "pnu_rows.csv",
        npz_path.parent / "pnu_rows.csv.gz",
        npz_path.parent.parent / "rows" / "cache_rows.csv",
        npz_path.parent.parent / "rows" / "cache_rows.csv.gz",
        npz_path.parent.parent / "rows" / "pnu_rows.csv",
        npz_path.parent.parent / "rows" / "pnu_rows.csv.gz",
    )
    matches = [path for path in candidates if path.is_file()]
    return matches[0] if len(matches) == 1 else None


def _resolve_manifest(npz_path, manifest_path=None):
    if manifest_path is not None:
        path = Path(manifest_path)
        _require(path.is_file(), "PNU manifest is missing")
        return path
    candidates = (
        npz_path.parent / "manifest.json",
        npz_path.with_suffix(".manifest.json"),
        npz_path.parent.parent / "manifest.json",
    )
    matches = [path for path in candidates if path.is_file()]
    return matches[0] if matches else None


def load_pnu_feature_cache(cache_path, rows_path=None, manifest_path=None):
    """Load the new P/N/U cache while tolerating CSV- or NPZ-held metadata."""
    npz_path = _resolve_npz(cache_path)
    with np.load(str(npz_path), allow_pickle=False) as archive:
        _require(FEATURE_KEY in archive.files, "PNU cache lacks {}".format(FEATURE_KEY))
        features = np.asarray(archive[FEATURE_KEY]).copy()
        metadata = {
            key: np.asarray(archive[key]).copy()
            for key in archive.files
            if key != FEATURE_KEY and np.asarray(archive[key]).ndim == 1
        }
    _require(features.ndim == 2, "PNU feature array must be a matrix")
    _require(features.shape[1] == FEATURE_DIMENSION, "PNU feature dimension changed")
    _require(bool(np.isfinite(features).all()), "PNU features contain non-finite values")

    resolved_rows = _resolve_rows(npz_path, rows_path)
    if resolved_rows is not None:
        raw = pd.read_csv(
            str(resolved_rows), dtype=str, keep_default_na=False, na_filter=False
        )
        _require(len(raw) == len(features), "PNU rows/features length mismatch")
        for identifier in ("pair_uid", "row_id", "cache_uid"):
            if identifier in raw and identifier in metadata:
                _require(
                    bool(raw[identifier].astype(str).eq(np.asarray(metadata[identifier]).astype(str)).all()),
                    "PNU CSV/NPZ row order mismatch",
                )
    else:
        _require(metadata, "PNU cache has neither rows CSV nor NPZ metadata")
        raw = pd.DataFrame({key: value.astype(str) for key, value in metadata.items()})
        _require(len(raw) == len(features), "PNU NPZ metadata length mismatch")
    rows = canonicalize_pnu_rows(raw)

    resolved_manifest = _resolve_manifest(npz_path, manifest_path)
    manifest = {}
    if resolved_manifest is not None:
        manifest = _read_json(resolved_manifest)
        expected = manifest.get("output", {}).get("sha256")
        if expected is not None:
            _require(expected == sha256_file(npz_path), "PNU cache hash disagrees with manifest")
    return rows, features, manifest, npz_path, resolved_rows, resolved_manifest


@dataclass
class RolePool(object):
    frame: pd.DataFrame
    features: np.ndarray

    def __post_init__(self):
        _require(len(self.frame) == len(self.features), "role-pool length mismatch")
        _require(self.features.ndim == 2, "role-pool features must be a matrix")
        _require(self.features.shape[1] == FEATURE_DIMENSION, "role-pool feature width changed")


@dataclass
class LibraryPools(object):
    positive: RolePool
    negative: RolePool
    unlabeled: RolePool


def _old_weak_to_canonical(frame):
    output = frame.copy()
    output["pnu_role"] = np.where(
        pd.to_numeric(output["weak_label"], errors="raise").astype(int).eq(1),
        "P",
        "N",
    )
    return canonicalize_pnu_rows(output)


def load_existing_pn_and_retention(cache_path, cache_manifest=None):
    """Load immutable P/N features and keep outcomes in a separate object."""
    frame, features, manifests, paths = cached_eval.load_feature_cache(
        cache_path, cache_manifest
    )
    frame = frame.copy()
    frame["_feature_index"] = np.arange(len(frame), dtype=int)
    weak = frame.loc[frame["source_kind"].eq("weak")].copy()
    retention = frame.loc[frame["source_kind"].eq("retention")].copy()
    retention["measurement_missing"] = pd.to_numeric(
        retention["measurement_missing"], errors="raise"
    ).astype(int)
    retention = retention.loc[retention["measurement_missing"].eq(0)].copy()
    _require(not weak.empty and not retention.empty, "existing cache lacks weak or retention rows")
    return weak, retention, features, manifests, paths


def _strict_library_pn(weak, retention, features, library):
    weak_library = weak.loc[weak["library"].eq(library)].copy()
    evaluation = retention.loc[retention["library"].eq(library)].copy()
    _require(not weak_library.empty and not evaluation.empty, "empty library in existing cache")
    strict = cached_eval._regime_pool(weak_library, evaluation, "double_cold")
    indices = strict["_feature_index"].to_numpy(dtype=int)
    canonical = _old_weak_to_canonical(strict)
    selected_features = np.asarray(features[indices])
    p_mask = canonical["pnu_role"].eq("P").to_numpy()
    n_mask = canonical["pnu_role"].eq("N").to_numpy()
    _require(bool(p_mask.any()) and bool(n_mask.any()), "strict pool lacks P or N")
    return (
        RolePool(canonical.loc[p_mask].reset_index(drop=True), selected_features[p_mask]),
        RolePool(canonical.loc[n_mask].reset_index(drop=True), selected_features[n_mask]),
        evaluation.reset_index(drop=True),
        np.asarray(features[evaluation["_feature_index"].to_numpy(dtype=int)]),
    )


def _audit_unlabeled_contract(
    positive,
    negative,
    unlabeled,
    evaluation,
    contract,
    u_per_positive,
):
    p_pairs = set(positive.frame["sequence_pair_sha256"])
    n_pairs = set(negative.frame["sequence_pair_sha256"])
    u_pairs = set(unlabeled.frame["sequence_pair_sha256"])
    _require(p_pairs.isdisjoint(n_pairs), "P/N overlap")
    _require(p_pairs.isdisjoint(u_pairs), "P/U overlap")
    _require(n_pairs.isdisjoint(u_pairs), "N/U overlap")
    eval_peptides = set(evaluation["chain1_sha256"].astype(str))
    eval_affibodies = set(evaluation["chain2_sha256"].astype(str))
    for name, pool in (("P", positive), ("N", negative), ("U", unlabeled)):
        _require(
            set(pool.frame["chain1_sha256"]).isdisjoint(eval_peptides),
            "{} shares an evaluation peptide identity".format(name),
        )
        _require(
            set(pool.frame["chain2_sha256"]).isdisjoint(eval_affibodies),
            "{} shares an evaluation Affibody identity".format(name),
        )
    if contract == "sampled_u_per_positive":
        expected = int(round(float(u_per_positive) * len(positive.frame)))
        _require(
            len(unlabeled.frame) == expected,
            "sampled-U count {} != {} x P ({})".format(
                len(unlabeled.frame), u_per_positive, expected
            ),
        )
    elif contract == "all_observed_u":
        _require(len(unlabeled.frame) > 0, "full-U contract has no U rows")
    else:
        raise ValueError("unknown U contract {}".format(contract))


def build_library_pools(
    weak,
    retention,
    existing_features,
    pnu_rows,
    pnu_features,
    library,
    u_contract,
    u_per_positive,
):
    positive, negative, evaluation, evaluation_features = _strict_library_pn(
        weak, retention, existing_features, library
    )
    external = pnu_rows.loc[pnu_rows["library"].eq(library)].copy()
    _require(not external.empty, "PNU cache has no {} rows".format(library))
    external_indices = external.index.to_numpy(dtype=int)
    external_features = np.asarray(pnu_features[external_indices])
    u_mask = external["pnu_role"].eq("U").to_numpy()
    _require(bool(u_mask.any()), "PNU cache has no {} unlabeled rows".format(library))
    unlabeled = RolePool(
        external.loc[u_mask].reset_index(drop=True), external_features[u_mask]
    )

    # If the new cache repeats P or N, require exact membership agreement.  The
    # existing immutable cache remains the feature source for these two groups.
    for role, pool in (("P", positive), ("N", negative)):
        mask = external["pnu_role"].eq(role).to_numpy()
        if bool(mask.any()):
            observed = set(external.loc[mask, "sequence_pair_sha256"].astype(str))
            expected = set(pool.frame["sequence_pair_sha256"].astype(str))
            _require(observed == expected, "external {} membership differs".format(role))

    _audit_unlabeled_contract(
        positive,
        negative,
        unlabeled,
        evaluation,
        u_contract,
        u_per_positive,
    )
    return LibraryPools(positive, negative, unlabeled), evaluation, evaluation_features


def _stable_bin(value, axis, seed, n_folds):
    payload = "{}|{}|{}".format(axis, int(seed), value).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(n_folds)


def identity_cold_roles(frame, fold, n_folds, seed=17):
    """Return train/guard/validation indices for double-partner-cold CV."""
    _require(int(n_folds) >= 2, "n_folds must be at least two")
    _require(0 <= int(fold) < int(n_folds), "fold out of range")
    peptide = np.asarray(
        [_stable_bin(value, "peptide", seed, n_folds) for value in frame["chain1_sha256"]],
        dtype=int,
    )
    affibody = np.asarray(
        [_stable_bin(value, "affibody", seed, n_folds) for value in frame["chain2_sha256"]],
        dtype=int,
    )
    peptide_held = peptide == int(fold)
    affibody_held = affibody == int(fold)
    train = ~peptide_held & ~affibody_held
    validation = peptide_held & affibody_held
    guard = ~(train | validation)
    output = {
        "train": np.flatnonzero(train),
        "guard": np.flatnonzero(guard),
        "validation": np.flatnonzero(validation),
    }
    _require(
        sum(len(indices) for indices in output.values()) == len(frame),
        "identity roles do not partition rows",
    )
    return output


def make_fold_plan(pools, fold, n_folds, split_seed):
    plans = {
        "P": identity_cold_roles(pools.positive.frame, fold, n_folds, split_seed),
        "N": identity_cold_roles(pools.negative.frame, fold, n_folds, split_seed),
        "U": identity_cold_roles(pools.unlabeled.frame, fold, n_folds, split_seed),
    }
    _require(len(plans["P"]["train"]) > 0, "fold has no P training rows")
    _require(len(plans["N"]["train"]) > 0, "fold has no N training rows")
    _require(len(plans["U"]["train"]) > 0, "fold has no U training rows")
    _require(len(plans["P"]["validation"]) > 0, "fold has no P validation rows")
    _require(len(plans["N"]["validation"]) > 0, "fold has no N validation rows")
    return plans


def _running_mean_scale(groups, chunk_size=8192):
    """Compute train-only feature moments without materializing a giant copy."""
    total = 0
    sums = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    squares = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    for values, indices in groups:
        indices = np.asarray(indices, dtype=int)
        for start in range(0, len(indices), int(chunk_size)):
            block = np.asarray(values[indices[start : start + int(chunk_size)]], dtype=np.float32)
            sums += block.sum(axis=0, dtype=np.float64)
            squares += np.square(block, dtype=np.float32).sum(axis=0, dtype=np.float64)
            total += len(block)
    _require(total > 0, "cannot standardize an empty training set")
    mean = sums / float(total)
    variance = np.maximum(squares / float(total) - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def pnu_risk_from_logits(
    positive_logits,
    negative_logits,
    unlabeled_logits,
    pi,
    eta,
    weight=None,
    l2_coefficient=0.0,
):
    """Return the exact requested non-negative PNU risk and components."""
    pi = _finite_float(pi, "pi")
    eta = _finite_float(eta, "eta")
    _require(0.0 < pi < 1.0, "pi must lie strictly between zero and one")
    _require(0.0 <= eta <= 1.0, "eta must lie in [0, 1]")
    _require(positive_logits.numel() > 0, "positive batch is empty")
    _require(negative_logits.numel() > 0, "negative batch is empty")
    if eta > 0.0:
        _require(unlabeled_logits is not None, "eta>0 requires an unlabeled batch")
        _require(unlabeled_logits.numel() > 0, "unlabeled batch is empty")
    rp_positive = F.softplus(-positive_logits).mean()
    rp_negative = F.softplus(positive_logits).mean()
    rn_negative = F.softplus(negative_logits).mean()
    if unlabeled_logits is None:
        ru_negative = torch.zeros_like(rp_positive)
    else:
        ru_negative = F.softplus(unlabeled_logits).mean()
    raw_negative = (
        (1.0 - eta) * (1.0 - pi) * rn_negative
        + eta * (ru_negative - pi * rp_negative)
    )
    nonnegative = torch.clamp(raw_negative, min=0.0)
    penalty = torch.zeros_like(rp_positive)
    if weight is not None and float(l2_coefficient) > 0.0:
        penalty = 0.5 * float(l2_coefficient) * torch.square(weight).sum()
    total = pi * rp_positive + nonnegative + penalty
    return total, {
        "rp_positive": rp_positive,
        "rp_negative": rp_negative,
        "rn_negative": rn_negative,
        "ru_negative": ru_negative,
        "raw_negative": raw_negative,
        "nonnegative_negative": nonnegative,
        "l2_penalty": penalty,
    }


def _cycled_batches(indices, batch_size, steps, rng):
    indices = np.asarray(indices, dtype=int)
    _require(len(indices) > 0, "cannot batch an empty group")
    permutation = rng.permutation(indices)
    cursor = 0
    output = []
    for _ in range(int(steps)):
        remaining = int(batch_size)
        pieces = []
        while remaining > 0:
            available = len(permutation) - cursor
            take = min(remaining, available)
            pieces.append(permutation[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == len(permutation):
                permutation = rng.permutation(indices)
                cursor = 0
        output.append(np.concatenate(pieces))
    return output


def _next_cyclic_batch(indices, state, size, rng):
    """Draw a shuffled group batch, cycling only P/N when an arm exhausts."""
    indices = np.asarray(indices, dtype=int)
    _require(len(indices) > 0, "cannot batch an empty group")
    needed = min(int(size), len(indices))
    pieces = []
    while needed > 0:
        if state.get("order") is None or state["cursor"] >= len(state["order"]):
            state["order"] = rng.permutation(indices)
            state["cursor"] = 0
        available = len(state["order"]) - state["cursor"]
        take = min(needed, available)
        pieces.append(state["order"][state["cursor"] : state["cursor"] + take])
        state["cursor"] += take
        needed -= take
    return np.concatenate(pieces)


def _feature_tensor(values, indices, mean, scale, device):
    block = np.asarray(values[np.asarray(indices, dtype=int)], dtype=np.float32)
    tensor = torch.from_numpy(block).to(device=device, non_blocking=True)
    return (tensor - mean) / scale


def _predict_linear(model, values, mean, scale, device, batch_size=8192):
    predictions = []
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            block = np.asarray(values[start : start + int(batch_size)], dtype=np.float32)
            tensor = torch.from_numpy(block).to(device=device, non_blocking=True)
            logits = model((tensor - mean_tensor) / scale_tensor).reshape(-1)
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def train_pnu_head(
    pools,
    train_indices,
    prediction_features,
    pi,
    eta,
    c_value,
    epochs,
    checkpoints,
    batch_size,
    prediction_batch_size,
    learning_rate,
    seed,
    device,
):
    """Fit one linear readout with separate P/N/U mini-batch means."""
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.RandomState(int(seed))
    p_indices = np.asarray(train_indices["P"], dtype=int)
    n_indices = np.asarray(train_indices["N"], dtype=int)
    u_indices = np.asarray(train_indices["U"], dtype=int)
    mean, scale = _running_mean_scale(
        ((pools.positive.features, p_indices), (pools.negative.features, n_indices))
    )
    model = torch.nn.Linear(FEATURE_DIMENSION, 1, bias=True).to(device)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    n_reference = len(p_indices) + len(n_indices)
    l2_coefficient = 1.0 / (float(c_value) * float(n_reference))
    max_epochs = int(epochs)
    checkpoints = set(int(value) for value in checkpoints)
    history = {}
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)

    for epoch in range(1, max_epochs + 1):
        if float(eta) > 0.0:
            # Visit each sampled U row exactly once per epoch.  The final U
            # batch may be smaller; P and N use equally sized separate means.
            u_order = rng.permutation(u_indices)
            u_batches = [
                u_order[start : start + int(batch_size)]
                for start in range(0, len(u_order), int(batch_size))
            ]
            current_sizes = [len(batch) for batch in u_batches]
        else:
            largest = max(len(p_indices), len(n_indices))
            current_sizes = [
                min(int(batch_size), largest - start)
                for start in range(0, largest, int(batch_size))
            ]
            u_batches = [None] * len(current_sizes)
        steps = len(current_sizes)
        p_state = {"order": None, "cursor": 0}
        n_state = {"order": None, "cursor": 0}
        model.train()
        epoch_raw_negative = []
        u_rows_seen = 0
        for current_size, u_batch in zip(current_sizes, u_batches):
            p_batch = _next_cyclic_batch(
                p_indices, p_state, current_size, rng
            )
            n_batch = _next_cyclic_batch(
                n_indices, n_state, current_size, rng
            )
            optimizer.zero_grad(set_to_none=True)
            p_x = _feature_tensor(
                pools.positive.features, p_batch, mean_tensor, scale_tensor, device
            )
            n_x = _feature_tensor(
                pools.negative.features, n_batch, mean_tensor, scale_tensor, device
            )
            p_logits = model(p_x).reshape(-1)
            n_logits = model(n_x).reshape(-1)
            if u_batch is None:
                u_logits = None
            else:
                u_x = _feature_tensor(
                    pools.unlabeled.features, u_batch, mean_tensor, scale_tensor, device
                )
                u_logits = model(u_x).reshape(-1)
                u_rows_seen += len(u_batch)
            loss, components = pnu_risk_from_logits(
                p_logits,
                n_logits,
                u_logits,
                pi,
                eta,
                weight=model.weight,
                l2_coefficient=l2_coefficient,
            )
            _require(bool(torch.isfinite(loss).item()), "non-finite PNU loss")
            loss.backward()
            optimizer.step()
            epoch_raw_negative.append(float(components["raw_negative"].detach().cpu()))
        if float(eta) > 0.0:
            _require(u_rows_seen == len(u_indices), "an epoch did not visit each sampled U row exactly once")
        if epoch in checkpoints:
            history[epoch] = {
                "probability": _predict_linear(
                    model,
                    prediction_features,
                    mean,
                    scale,
                    device,
                    prediction_batch_size,
                ),
                "mean_raw_negative_risk": float(np.mean(epoch_raw_negative)),
                "fraction_steps_clamped": float(
                    np.mean(np.asarray(epoch_raw_negative) < 0.0)
                ),
                "u_rows_seen": int(u_rows_seen),
            }
    _require(checkpoints.issubset(set(history)), "not every epoch checkpoint was recorded")
    return model, mean, scale, history


def _binary_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "binary metrics need both classes")
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probability)),
        "ap": float(average_precision_score(labels, probability)),
    }


def _validation_features(pools, plans):
    p = np.asarray(pools.positive.features[plans["P"]["validation"]])
    n = np.asarray(pools.negative.features[plans["N"]["validation"]])
    labels = np.concatenate(
        (np.ones(len(p), dtype=int), np.zeros(len(n), dtype=int))
    )
    return np.concatenate((p, n), axis=0), labels


def run_pnu_weak_validation(
    pools,
    pi,
    eta_grid,
    c_grid,
    epoch_grid,
    folds,
    split_seed,
    training_seed,
    batch_size,
    prediction_batch_size,
    learning_rate,
    device,
):
    """Evaluate all PNU choices using weak labels only."""
    records = []
    pooled = {}
    max_epoch = max(int(value) for value in epoch_grid)
    for fold in range(int(folds)):
        plans = make_fold_plan(pools, fold, folds, split_seed)
        prediction_features, labels = _validation_features(pools, plans)
        train_indices = {role: plans[role]["train"] for role in ROLES}
        for eta in eta_grid:
            for c_value in c_grid:
                condition_seed = int(training_seed) + 100000 * fold + int(
                    round(float(eta) * 1000)
                ) + int(round(math.log10(float(c_value)) + 10)) * 1000
                _, _, _, history = train_pnu_head(
                    pools,
                    train_indices,
                    prediction_features,
                    pi,
                    eta,
                    c_value,
                    max_epoch,
                    epoch_grid,
                    batch_size,
                    prediction_batch_size,
                    learning_rate,
                    condition_seed,
                    device,
                )
                for epoch in epoch_grid:
                    probability = history[int(epoch)]["probability"]
                    metrics = _binary_metrics(labels, probability)
                    key = (float(eta), float(c_value), int(epoch))
                    pooled.setdefault(key, {"labels": [], "probability": []})
                    pooled[key]["labels"].append(labels)
                    pooled[key]["probability"].append(probability)
                    records.append(
                        dict(
                            record_type="fold",
                            pi=float(pi),
                            eta=float(eta),
                            C=float(c_value),
                            epoch=int(epoch),
                            fold=int(fold),
                            training_seed=int(training_seed),
                            n_train_p=int(len(plans["P"]["train"])),
                            n_train_n=int(len(plans["N"]["train"])),
                            n_train_u=int(len(plans["U"]["train"])),
                            n_guard_p=int(len(plans["P"]["guard"])),
                            n_guard_n=int(len(plans["N"]["guard"])),
                            n_guard_u=int(len(plans["U"]["guard"])),
                            mean_raw_negative_risk=float(
                                history[int(epoch)]["mean_raw_negative_risk"]
                            ),
                            fraction_steps_clamped=float(
                                history[int(epoch)]["fraction_steps_clamped"]
                            ),
                            **metrics
                        )
                    )
    for key, values in sorted(pooled.items()):
        eta, c_value, epoch = key
        labels = np.concatenate(values["labels"])
        probability = np.concatenate(values["probability"])
        records.append(
            dict(
                record_type="aggregate",
                pi=float(pi),
                eta=float(eta),
                C=float(c_value),
                epoch=int(epoch),
                fold=-1,
                training_seed=int(training_seed),
                n_train_p=-1,
                n_train_n=-1,
                n_train_u=-1,
                n_guard_p=-1,
                n_guard_n=-1,
                n_guard_u=-1,
                mean_raw_negative_risk=float("nan"),
                fraction_steps_clamped=float("nan"),
                **_binary_metrics(labels, probability)
            )
        )
    return pd.DataFrame(records)


def select_weak_hyperparameters(
    validation_records, fixed_eta=None, selection_metric="auroc"
):
    """Select eta/C/epoch from aggregate weak validation; no outcome input."""
    frame = validation_records.loc[
        validation_records["record_type"].eq("aggregate")
    ].copy()
    if fixed_eta is not None:
        frame = frame.loc[np.isclose(frame["eta"].astype(float), float(fixed_eta))]
    _require(not frame.empty, "no aggregate weak-validation records")
    selection_metric = str(selection_metric)
    _require(
        selection_metric in ("auroc", "ap", "log_loss"),
        "unknown weak selection metric",
    )
    if selection_metric == "auroc":
        columns = ["auroc", "ap", "log_loss", "eta", "C", "epoch"]
        ascending = [False, False, True, True, True, True]
    elif selection_metric == "ap":
        columns = ["ap", "auroc", "log_loss", "eta", "C", "epoch"]
        ascending = [False, False, True, True, True, True]
    else:
        columns = ["log_loss", "ap", "auroc", "eta", "C", "epoch"]
        ascending = [True, False, False, True, True, True]
    frame = frame.sort_values(columns, ascending=ascending, kind="mergesort")
    row = frame.iloc[0]
    return {
        "pi": float(row["pi"]),
        "eta": float(row["eta"]),
        "C": float(row["C"]),
        "epoch": int(row["epoch"]),
        "weak_validation_log_loss": float(row["log_loss"]),
        "weak_validation_ap": float(row["ap"]),
        "weak_validation_auroc": float(row["auroc"]),
        "weak_selection_metric": selection_metric,
    }


def _fit_sklearn_balanced(values, labels, c_value):
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        max_iter=2000,
        tol=1e-4,
    )
    model.fit(values, labels)
    _require(int(model.n_iter_[0]) < int(model.max_iter), "sklearn PN control did not converge")
    return model


def select_sklearn_pn_control(
    pools, c_grid, folds, split_seed, selection_metric="auroc"
):
    records = []
    pooled = {}
    for fold in range(int(folds)):
        plans = make_fold_plan(pools, fold, folds, split_seed)
        p_train = plans["P"]["train"]
        n_train = plans["N"]["train"]
        mean, scale = _running_mean_scale(
            ((pools.positive.features, p_train), (pools.negative.features, n_train))
        )
        train_x = np.concatenate(
            (
                np.asarray(pools.positive.features[p_train], dtype=np.float32),
                np.asarray(pools.negative.features[n_train], dtype=np.float32),
            ),
            axis=0,
        )
        train_x = (train_x - mean) / scale
        train_y = np.concatenate(
            (np.ones(len(p_train), dtype=int), np.zeros(len(n_train), dtype=int))
        )
        validation_x, validation_y = _validation_features(pools, plans)
        validation_x = (np.asarray(validation_x, dtype=np.float32) - mean) / scale
        for c_value in c_grid:
            model = _fit_sklearn_balanced(train_x, train_y, c_value)
            probability = model.predict_proba(validation_x)[:, 1]
            metrics = _binary_metrics(validation_y, probability)
            pooled.setdefault(float(c_value), {"labels": [], "probability": []})
            pooled[float(c_value)]["labels"].append(validation_y)
            pooled[float(c_value)]["probability"].append(probability)
            records.append(
                dict(
                    record_type="fold",
                    C=float(c_value),
                    fold=int(fold),
                    **metrics
                )
            )
    for c_value, values in sorted(pooled.items()):
        metrics = _binary_metrics(
            np.concatenate(values["labels"]), np.concatenate(values["probability"])
        )
        records.append(
            dict(record_type="aggregate", C=float(c_value), fold=-1, **metrics)
        )
    frame = pd.DataFrame(records)
    selection_metric = str(selection_metric)
    _require(
        selection_metric in ("auroc", "ap", "log_loss"),
        "unknown weak selection metric",
    )
    aggregate = frame.loc[frame["record_type"].eq("aggregate")]
    if selection_metric == "auroc":
        columns, ascending = ["auroc", "ap", "log_loss", "C"], [False, False, True, True]
    elif selection_metric == "ap":
        columns, ascending = ["ap", "auroc", "log_loss", "C"], [False, False, True, True]
    else:
        columns, ascending = ["log_loss", "ap", "auroc", "C"], [True, False, False, True]
    selected = aggregate.sort_values(
        columns, ascending=ascending, kind="mergesort"
    ).iloc[0]
    return float(selected["C"]), frame


def fit_final_sklearn_control(pools, retention_features, c_value):
    p = np.arange(len(pools.positive.frame), dtype=int)
    n = np.arange(len(pools.negative.frame), dtype=int)
    mean, scale = _running_mean_scale(
        ((pools.positive.features, p), (pools.negative.features, n))
    )
    train_x = np.concatenate(
        (
            np.asarray(pools.positive.features, dtype=np.float32),
            np.asarray(pools.negative.features, dtype=np.float32),
        ),
        axis=0,
    )
    train_x = (train_x - mean) / scale
    labels = np.concatenate(
        (np.ones(len(p), dtype=int), np.zeros(len(n), dtype=int))
    )
    model = _fit_sklearn_balanced(train_x, labels, c_value)
    prediction_x = (np.asarray(retention_features, dtype=np.float32) - mean) / scale
    probability = model.predict_proba(prediction_x)[:, 1]
    return model, mean, scale, probability


def best_retrospective_f1(labels, score):
    """Return the deterministic score>=threshold F1-maximizing diagnostic."""
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "F1 threshold needs both classes")
    candidates = []
    for threshold in np.unique(score):
        predicted = score >= float(threshold)
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum((~predicted) & (labels == 1)))
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        candidates.append((f1, precision, recall, float(threshold), tp, fp, fn))
    selected = max(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
    return {
        "retrospective_f1_threshold": selected[3],
        "retrospective_f1_precision": selected[1],
        "retrospective_f1_recall": selected[2],
        "retrospective_f1": selected[0],
        "retrospective_f1_tp": selected[4],
        "retrospective_f1_fp": selected[5],
        "retrospective_f1_fn": selected[6],
        "retrospective_f1_selected": selected[4] + selected[5],
    }


def retention_metrics(evaluation, score):
    labels = pd.to_numeric(evaluation["target_binder"], errors="raise").astype(int).to_numpy()
    observed = pd.to_numeric(evaluation["target_retention"], errors="raise").to_numpy(float)
    values = np.asarray(score, dtype=float)
    _require(len(values) == len(evaluation), "retention score length mismatch")
    _require(bool(np.isfinite(values).all()), "non-finite retention score")
    rhos = []
    for _, indices in evaluation.groupby("chain1_sha256", sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        if (
            len(index) >= 2
            and len(np.unique(observed[index])) >= 2
            and len(np.unique(values[index])) >= 2
        ):
            rhos.append(float(spearmanr(observed[index], values[index])[0]))
    output = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, values)),
        "ap": float(average_precision_score(labels, values)),
        "global_spearman": float(spearmanr(observed, values)[0]),
        "average_within_peptide_spearman": (
            float(np.mean(rhos)) if rhos else float("nan")
        ),
        "within_peptide_groups": int(len(rhos)),
    }
    output.update(best_retrospective_f1(labels, values))
    return output


def _retention_prediction_frame(
    evaluation,
    score,
    arm,
    training_seed,
    pi=None,
    eta=None,
    c_value=None,
    epoch=None,
    selected_eta=False,
):
    columns = [
        "library",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_sha256",
        "chain2_sha256",
        "target_retention",
        "target_binder",
    ]
    output = evaluation.loc[:, columns].copy()
    output["arm"] = arm
    output["training_seed"] = int(training_seed)
    output["pi"] = float(pi) if pi is not None else np.nan
    output["eta"] = float(eta) if eta is not None else np.nan
    output["C"] = float(c_value) if c_value is not None else np.nan
    output["epoch"] = int(epoch) if epoch is not None else 0
    output["selected_eta_by_weak_validation"] = int(bool(selected_eta))
    output["score"] = np.asarray(score, dtype=float)
    return output


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    _require(_private_mode(output_dir) == "0700", "output directory is not private")

    libraries = tuple(args.libraries)
    _require(set(libraries).issubset(set(DEFAULT_LIBRARIES)), "unknown library")
    pi_grid = tuple(sorted(set(_finite_float(value, "pi") for value in args.pi_grid)))
    eta_grid = tuple(sorted(set(_finite_float(value, "eta") for value in args.eta_grid)))
    c_grid = tuple(sorted(set(_finite_float(value, "C") for value in args.c_grid)))
    epoch_grid = tuple(sorted(set(int(value) for value in args.epoch_grid)))
    _require(all(0.0 < value < 1.0 for value in pi_grid), "invalid pi grid")
    _require(all(0.0 <= value <= 1.0 for value in eta_grid), "invalid eta grid")
    if not args.allow_partial_eta_grid:
        _require(
            0.0 in eta_grid and 1.0 in eta_grid,
            "eta grid must include PN and PU endpoints unless this is an explicit distributed shard",
        )
    _require(all(value > 0.0 for value in c_grid), "C values must be positive")
    _require(epoch_grid and min(epoch_grid) >= 1, "epoch grid must be positive")
    _require(int(args.folds) >= 2, "at least two folds are required")
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    if str(args.device).startswith("cuda"):
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")

    weak, retention, existing_features, pn_manifests, pn_paths = (
        load_existing_pn_and_retention(args.pn_cache, args.pn_cache_manifest)
    )
    pnu_rows, pnu_features, pnu_manifest, pnu_npz, pnu_rows_path, pnu_manifest_path = (
        load_pnu_feature_cache(args.pnu_cache, args.pnu_rows, args.pnu_manifest)
    )

    validation_blocks = []
    sklearn_validation_blocks = []
    selection_rows = []
    prediction_blocks = []
    metric_rows = []
    model_arrays = {}
    count_rows = []

    for library in libraries:
        pools, evaluation, evaluation_features = build_library_pools(
            weak,
            retention,
            existing_features,
            pnu_rows,
            pnu_features,
            library,
            args.u_contract,
            args.u_per_positive,
        )
        count_rows.append(
            {
                "library": library,
                "positive": len(pools.positive.frame),
                "negative": len(pools.negative.frame),
                "unlabeled_used": len(pools.unlabeled.frame),
                "retention_evaluation": len(evaluation),
                "u_contract": args.u_contract,
                "u_per_positive": float(args.u_per_positive),
            }
        )

        # Exact current P/N control.  It is expensive and independent of pi,
        # so distributed pi jobs may explicitly run it in only one shard.
        if not args.skip_sklearn_control:
            pn_c, pn_validation = select_sklearn_pn_control(
                pools, c_grid, args.folds, args.split_seed, args.selection_metric
            )
            pn_validation.insert(0, "library", library)
            sklearn_validation_blocks.append(pn_validation)
            pn_model, pn_mean, pn_scale, pn_probability = fit_final_sklearn_control(
                pools, evaluation_features, pn_c
            )
            pn_predictions = _retention_prediction_frame(
                evaluation,
                pn_probability,
                "balanced_sklearn_pn_control",
                -1,
                c_value=pn_c,
            )
            prediction_blocks.append(pn_predictions)
            metric_rows.append(
                dict(
                    library=library,
                    arm="balanced_sklearn_pn_control",
                    training_seed=-1,
                    pi=np.nan,
                    eta=np.nan,
                    C=float(pn_c),
                    epoch=0,
                    selected_eta_by_weak_validation=0,
                    **retention_metrics(evaluation, pn_probability)
                )
            )
            model_arrays["{}_sklearn_weight".format(library)] = pn_model.coef_.astype(np.float32)
            model_arrays["{}_sklearn_bias".format(library)] = pn_model.intercept_.astype(np.float32)
            model_arrays["{}_sklearn_mean".format(library)] = pn_mean.astype(np.float32)
            model_arrays["{}_sklearn_scale".format(library)] = pn_scale.astype(np.float32)

        for training_seed in args.training_seeds:
            for pi in pi_grid:
                weak_records = run_pnu_weak_validation(
                    pools,
                    pi,
                    eta_grid,
                    c_grid,
                    epoch_grid,
                    args.folds,
                    args.split_seed,
                    training_seed,
                    args.batch_size,
                    args.prediction_batch_size,
                    args.learning_rate,
                    args.device,
                )
                weak_records.insert(0, "library", library)
                validation_blocks.append(weak_records)
                selected_global = select_weak_hyperparameters(
                    weak_records, selection_metric=args.selection_metric
                )
                selected_global["epoch_at_upper_boundary"] = int(
                    int(selected_global["epoch"]) == max(epoch_grid)
                )
                selection_rows.append(
                    dict(
                        library=library,
                        training_seed=int(training_seed),
                        selection_scope="eta_C_epoch",
                        **selected_global
                    )
                )
                full_indices = {
                    "P": np.arange(len(pools.positive.frame), dtype=int),
                    "N": np.arange(len(pools.negative.frame), dtype=int),
                    "U": np.arange(len(pools.unlabeled.frame), dtype=int),
                }
                for eta in eta_grid:
                    selected_fixed = select_weak_hyperparameters(
                        weak_records,
                        fixed_eta=eta,
                        selection_metric=args.selection_metric,
                    )
                    selected_fixed["epoch_at_upper_boundary"] = int(
                        int(selected_fixed["epoch"]) == max(epoch_grid)
                    )
                    selection_rows.append(
                        dict(
                            library=library,
                            training_seed=int(training_seed),
                            selection_scope="C_epoch_at_fixed_eta",
                            **selected_fixed
                        )
                    )
                    model, mean, scale, history = train_pnu_head(
                        pools,
                        full_indices,
                        evaluation_features,
                        pi,
                        eta,
                        selected_fixed["C"],
                        selected_fixed["epoch"],
                        (selected_fixed["epoch"],),
                        args.batch_size,
                        args.prediction_batch_size,
                        args.learning_rate,
                        int(training_seed) + 900000,
                        args.device,
                    )
                    probability = history[selected_fixed["epoch"]]["probability"]
                    is_selected = (
                        float(eta) == float(selected_global["eta"])
                        and float(selected_fixed["C"]) == float(selected_global["C"])
                        and int(selected_fixed["epoch"]) == int(selected_global["epoch"])
                    )
                    arm = (
                        "prior_weighted_pn_eta0"
                        if float(eta) == 0.0
                        else "nnpu_eta1"
                        if float(eta) == 1.0
                        else "nnpnu_eta{}".format(str(float(eta)).replace(".", "p"))
                    )
                    prediction_blocks.append(
                        _retention_prediction_frame(
                            evaluation,
                            probability,
                            arm,
                            training_seed,
                            pi,
                            eta,
                            selected_fixed["C"],
                            selected_fixed["epoch"],
                            is_selected,
                        )
                    )
                    metric_rows.append(
                        dict(
                            library=library,
                            arm=arm,
                            training_seed=int(training_seed),
                            pi=float(pi),
                            eta=float(eta),
                            C=float(selected_fixed["C"]),
                            epoch=int(selected_fixed["epoch"]),
                            selected_eta_by_weak_validation=int(is_selected),
                            weak_validation_log_loss=float(
                                selected_fixed["weak_validation_log_loss"]
                            ),
                            **retention_metrics(evaluation, probability)
                        )
                    )
                    prefix = "{}_seed{}_pi{}_eta{}".format(
                        library,
                        int(training_seed),
                        str(float(pi)).replace(".", "p"),
                        str(float(eta)).replace(".", "p"),
                    )
                    model_arrays[prefix + "_weight"] = model.weight.detach().cpu().numpy().astype(np.float32)
                    model_arrays[prefix + "_bias"] = model.bias.detach().cpu().numpy().astype(np.float32)
                    model_arrays[prefix + "_mean"] = mean.astype(np.float32)
                    model_arrays[prefix + "_scale"] = scale.astype(np.float32)

    weak_validation = pd.concat(validation_blocks, ignore_index=True, sort=False)
    sklearn_validation = (
        pd.concat(sklearn_validation_blocks, ignore_index=True, sort=False)
        if sklearn_validation_blocks
        else pd.DataFrame(
            columns=("library", "record_type", "C", "fold", "n", "positive", "prevalence", "log_loss", "auroc", "ap")
        )
    )
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_blocks, ignore_index=True, sort=False)
    metrics = pd.DataFrame(metric_rows)
    counts = pd.DataFrame(count_rows)

    _atomic_csv(weak_validation, output_dir / "pnu_weak_validation.csv")
    _atomic_csv(sklearn_validation, output_dir / "sklearn_pn_weak_validation.csv")
    _atomic_csv(selections, output_dir / "selected_hyperparameters.csv")
    _atomic_csv(predictions, output_dir / "retention_predictions.csv")
    _atomic_csv(metrics, output_dir / "retention_metrics.csv")
    _atomic_csv(counts, output_dir / "data_counts.csv")
    _atomic_npz(model_arrays, output_dir / "fitted_heads.npz")

    output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "runtime_seconds": float(time.time() - started),
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(args.device),
        },
        "inputs": {
            "pn_cache_paths": [
                {"path": str(path), "sha256": sha256_file(path)} for path in pn_paths
            ],
            "pnu_cache": {"path": str(pnu_npz), "sha256": sha256_file(pnu_npz)},
            "pnu_rows": (
                {"path": str(pnu_rows_path), "sha256": sha256_file(pnu_rows_path)}
                if pnu_rows_path is not None
                else None
            ),
            "pnu_manifest": (
                {
                    "path": str(pnu_manifest_path),
                    "sha256": sha256_file(pnu_manifest_path),
                    "payload": pnu_manifest,
                }
                if pnu_manifest_path is not None
                else None
            ),
        },
        "configuration": {
            "libraries": list(libraries),
            "pi_grid": list(pi_grid),
            "eta_grid": list(eta_grid),
            "C_grid": list(c_grid),
            "epoch_grid": list(epoch_grid),
            "folds": int(args.folds),
            "split_seed": int(args.split_seed),
            "training_seeds": [int(value) for value in args.training_seeds],
            "weak_selection_metric": str(args.selection_metric),
            "sklearn_pn_control_included": not bool(args.skip_sklearn_control),
            "partial_eta_grid_distributed_shard": bool(args.allow_partial_eta_grid),
            "batch_size_per_role": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "u_contract": args.u_contract,
            "u_per_positive": float(args.u_per_positive),
            "risk": "pi Rp+ + max(0, (1-eta)(1-pi) Rn- + eta(Ru- - pi Rp-))",
            "l2": "0.5 * ||w||^2 / (C * number_of_labeled_training_rows)",
            "standardization": "P/N training-fold mean and population SD only",
            "readout": "one linear logit; 2561 trainable parameters",
        },
        "retention_usage": {
            "partner_identities": "strict training exclusion only",
            "numeric_retention_and_75_percent_binder": (
                "retrospective metrics only, after all weak-label hyperparameter selection"
            ),
            "used_for_training_early_stopping_or_selection": False,
            "f1_threshold": "retrospective diagnostic selected on the same retention panel",
        },
        "u_scope_warning": (
            "This is a deterministic sampled-U frozen-MINT pilot, not full-U PNU."
            if args.u_contract == "sampled_u_per_positive"
            else "All observed eligible U rows were used."
        ),
        "epoch_boundary_audit": {
            "maximum_epoch_evaluated": int(max(epoch_grid)),
            "selection_rows_at_upper_boundary": int(
                selections["epoch_at_upper_boundary"].astype(int).sum()
            ),
            "selection_rows_total": int(len(selections)),
            "requires_extension_if_highlighted_model_is_at_boundary": True,
        },
        "counts": counts.to_dict(orient="records"),
        "outputs": output_hashes,
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "runtime_seconds": manifest["runtime_seconds"],
                "metric_rows": len(metrics),
                "u_contract": args.u_contract,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pn-cache",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz",
    )
    parser.add_argument("--pn-cache-manifest", type=Path)
    parser.add_argument("--pnu-cache", type=Path, required=True)
    parser.add_argument("--pnu-rows", type=Path)
    parser.add_argument("--pnu-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--libraries", nargs="+", default=list(DEFAULT_LIBRARIES))
    parser.add_argument("--pi-grid", nargs="+", type=float, default=list(DEFAULT_PI_GRID))
    parser.add_argument("--eta-grid", nargs="+", type=float, default=list(DEFAULT_ETA_GRID))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument(
        "--epoch-grid", nargs="+", type=int, default=list(DEFAULT_EPOCH_GRID)
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--skip-sklearn-control",
        action="store_true",
        help="distributed pi shard only; an exact control must run elsewhere",
    )
    parser.add_argument(
        "--allow-partial-eta-grid",
        action="store_true",
        help="distributed fixed-eta shard; aggregate every prespecified eta afterward",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("auroc", "ap", "log_loss"),
        default="auroc",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--u-contract",
        choices=("sampled_u_per_positive", "all_observed_u"),
        default="sampled_u_per_positive",
    )
    parser.add_argument("--u-per-positive", type=float, default=5.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
