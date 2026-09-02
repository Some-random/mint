#!/usr/bin/env python
"""Train additive Affibody site models with PN, nnPU, or nnPNU risk.

This is a deliberately small control experiment.  Each peptide/Affibody pair
is represented only by the amino acids at the designed positions, and the
model logit is the sum of one learned weight per position/residue plus an
intercept.  LibA and LibB are always fitted separately.

The labeled examples use the existing strict selection-derived definition:

* P: pooled-R009/R010 top-2% positives;
* N: provider negatives with R001 count >= 3;
* both P and N must be on-design and share neither retention peptide nor
  retention Affibody identity.

The unlabeled pool U is supplied separately.  Its canonical construction is
the strict, below-positive-cutoff part of the outer union of R009 and R010.
Every eligible U row is visited once per training epoch whenever the risk has
an unlabeled component; U is never randomly downsampled.

For logits z, l+(z)=softplus(-z) and l-(z)=softplus(z).  The formal nnPNU risk
implemented here is

  pi * Rp+ + max(0,
      (1-eta) * (1-pi) * Rn- + eta * (Ru- - pi * Rp-)).

Thus eta=0 is prior-weighted PN and eta=1 is nnPU.  ``risk=pn`` is a separate
ordinary balanced PN control, 0.5*Rp+ + 0.5*Rn-, matching the intent of the
current class-balanced additive baseline.

All hyperparameter and epoch choices use deterministic double-identity-cold
weak-label validation.  Retention is loaded only after those choices have
been made and is used solely for final retrospective evaluation.
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_selection_weak_labels import (
    EXPECTED_POOLED_CUTOFF,
    within_declared_alphabet,
)
from downstream.AffibodyMHC.build_pu_unlabeled_pool import load_retention_identities
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    load_retention_table,
    opaque_id,
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.evaluate_selection_weak_baseline import (
    LIBRARIES,
    add_identity_blocks,
    audit_identity_cold_fold,
    identity_blocked_indices,
    load_and_validate_weak_inputs,
    primary_training_frame,
    stable_identity_bin,
)


DEFAULT_PI_GRID = (0.02, 0.05, 0.10, 0.15)
DEFAULT_ETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
DEFAULT_MAX_EPOCHS = 40
DEFAULT_FOLDS = 5
DEFAULT_BATCH_SIZE = 65536
DEFAULT_SEED = 20260902
RETENTION_BINDER_THRESHOLD = 75.0

U_REQUIRED_COLUMNS = (
    "library",
    "pep",
    "aff",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "positive_cutoff",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
)
CANONICAL_U_ROWS = {"LibA": 726532, "LibB": 1157625}
CANONICAL_U_MEMBERSHIP_SHA256 = {
    "LibA": "53664441c56be277fa4695cddd1d2e57855fb258e6f003b4a635c20d35e451fd",
    "LibB": "e533d4b9b57aad866ae97788caf9cf0e27df6547281d3c3b37d93db819c7b5d7",
}
CANONICAL_U_ALL_MEMBERSHIP_SHA256 = (
    "eb5f2a8b19608656c030e3339901694272c44d43340d61192e22eea83618ffab"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_probability(value, name, closed=False):
    value = float(value)
    interval = 0.0 <= value <= 1.0 if closed else 0.0 < value < 1.0
    _require(math.isfinite(value) and interval, "{} must lie in {}".format(
        name, "[0,1]" if closed else "(0,1)"
    ))
    return value


def write_private_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _manifest_digest(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("sha256")
    return None


def membership_sha256(frame):
    digest = hashlib.sha256()
    for value in sorted(frame["pair_uid"].astype(str)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_unlabeled_uids(frame):
    expected_pair = np.asarray(
        [
            opaque_id(library, peptide, affibody)
            for library, peptide, affibody in zip(
                frame["library"], frame["pep"], frame["aff"]
            )
        ],
        dtype=object,
    )
    _require(
        bool(np.equal(expected_pair, frame["pair_uid"].to_numpy(dtype=object)).all()),
        "unlabeled pair UID does not match its biological key",
    )
    peptide_keys = frame[["library", "pep", "peptide_uid"]].drop_duplicates()
    expected_peptide = [
        opaque_id(library, "pep", peptide)
        for library, peptide in zip(peptide_keys["library"], peptide_keys["pep"])
    ]
    _require(
        peptide_keys["peptide_uid"].tolist() == expected_peptide,
        "unlabeled peptide UID does not match its biological identity",
    )
    affibody_keys = frame[["library", "aff", "affibody_uid"]].drop_duplicates()
    expected_affibody = [
        opaque_id(library, "aff", affibody)
        for library, affibody in zip(affibody_keys["library"], affibody_keys["aff"])
    ]
    _require(
        affibody_keys["affibody_uid"].tolist() == expected_affibody,
        "unlabeled Affibody UID does not match its biological identity",
    )


def load_unlabeled_pool(path, manifest_path=None):
    """Load and fail-closed validate the canonical strict R009/R010 U pool."""
    path = Path(path)
    _require(path.is_file(), "unlabeled pool does not exist: {}".format(path))
    if manifest_path is None:
        manifest_path = path.parent / "manifest.json"
    manifest_path = Path(manifest_path)
    _require(manifest_path.is_file(), "missing unlabeled-pool manifest")
    with open(str(manifest_path), "r") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("artifact") == {"name": "selection_pu_pool", "version": 1},
        "unsupported unlabeled-pool artifact",
    )
    _require(manifest.get("schema_version") == 1, "unsupported U schema version")
    _require(
        tuple(manifest.get("schema", {}).get("columns", ())) == U_REQUIRED_COLUMNS,
        "unlabeled-pool manifest schema mismatch",
    )
    output_entry = manifest.get("outputs", {}).get(path.name)
    _require(output_entry is not None, "manifest does not record {}".format(path.name))
    _require(
        _manifest_digest(output_entry) == sha256_file(path),
        "unlabeled-pool hash mismatch",
    )

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing = set(U_REQUIRED_COLUMNS).difference(frame.columns)
    _require(not missing, "unlabeled pool missing columns {}".format(sorted(missing)))
    _require(tuple(frame.columns) == U_REQUIRED_COLUMNS, "unexpected unlabeled-pool schema")
    _require(len(frame) > 0, "unlabeled pool is empty")
    for column in (
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "positive_cutoff",
    ):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        _require(bool(numeric.notna().all()), "nonnumeric unlabeled {}".format(column))
        _require(bool(np.equal(numeric, np.floor(numeric)).all()), "noninteger unlabeled {}".format(column))
        _require(bool(numeric.ge(0).all()), "negative unlabeled {}".format(column))
        frame[column] = numeric.astype(np.int64)

    _require(
        bool(
            frame["pooled_r009_r010_count"].eq(
                frame["r009_count"] + frame["r010_count"]
            ).all()
        ),
        "unlabeled pooled counts do not equal R009+R010",
    )
    _require(
        bool(
            frame["pooled_r009_r010_count"].lt(frame["positive_cutoff"]).all()
        ),
        "positive-cutoff row leaked into unlabeled pool",
    )
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate U pair UID")
    _require(set(frame["library"]) == set(LIBRARIES), "U pool must contain LibA and LibB")
    _validate_unlabeled_uids(frame)

    alphabet = set(AA_ALPHABET)
    for library in LIBRARIES:
        subset = frame.loc[frame["library"].eq(library)]
        spec = LIBRARY_SPECS[library]
        _require(
            len(subset) == CANONICAL_U_ROWS[library],
            "{} U row count differs from the canonical full pool".format(library),
        )
        _require(
            bool(subset["positive_cutoff"].eq(EXPECTED_POOLED_CUTOFF[library]).all()),
            "{} U cutoff differs from canonical pooled cutoff".format(library),
        )
        _require(bool(subset["pep"].str.len().eq(spec["pep_length"]).all()), "bad U peptide length")
        _require(bool(subset["aff"].str.len().eq(spec["aff_length"]).all()), "bad U Affibody length")
        _require(bool(subset["pep"].map(lambda value: set(value).issubset(alphabet)).all()), "noncanonical U peptide")
        _require(bool(subset["aff"].map(lambda value: set(value).issubset(alphabet)).all()), "noncanonical U Affibody")
        _require(
            bool(
                pd.Series(
                    [
                        within_declared_alphabet(library, peptide, affibody)
                        for peptide, affibody in zip(subset["pep"], subset["aff"])
                    ],
                    index=subset.index,
                ).all()
            ),
            "off-design row leaked into U pool",
        )
        observed_membership = membership_sha256(subset)
        _require(
            observed_membership == CANONICAL_U_MEMBERSHIP_SHA256[library],
            "{} U membership differs from the canonical full pool".format(library),
        )
        _require(
            manifest.get("membership_sha256", {}).get("by_library", {}).get(library)
            == observed_membership,
            "{} U manifest membership mismatch".format(library),
        )
    all_membership = membership_sha256(frame)
    _require(
        all_membership == CANONICAL_U_ALL_MEMBERSHIP_SHA256,
        "combined U membership differs from the canonical full pool",
    )
    _require(
        manifest.get("membership_sha256", {}).get("all") == all_membership,
        "combined U manifest membership mismatch",
    )
    return frame.sort_values(["library", "pair_uid"]).reset_index(drop=True), manifest


def validate_training_pools(primary, unlabeled, retention, library):
    """Audit pair/partner isolation among strict P/N, U, and retention."""
    p = primary.loc[primary["weak_label"].eq(1)]
    n = primary.loc[primary["weak_label"].eq(0)]
    u = unlabeled.loc[unlabeled["library"].eq(library)]
    _require(len(p) > 0 and len(n) > 0 and len(u) > 0, "P/N/U must all be nonempty")
    _require(set(p["pair_uid"]).isdisjoint(set(n["pair_uid"])), "P/N pair overlap")
    _require(set(primary["pair_uid"]).isdisjoint(set(u["pair_uid"])), "labeled/U pair overlap")
    held = retention.loc[retention["library"].eq(library)]
    held_peptides = set(held["peptide_uid"])
    held_affibodies = set(held["affibody_uid"])
    for name, frame in (("P/N", primary), ("U", u)):
        _require(set(frame["peptide_uid"]).isdisjoint(held_peptides), "{} shares retention peptide".format(name))
        _require(set(frame["affibody_uid"]).isdisjoint(held_affibodies), "{} shares retention Affibody".format(name))
    return p.reset_index(drop=True), n.reset_index(drop=True), u.reset_index(drop=True)


def site_index_matrix(frame, library, peptide_column="pep", affibody_column="aff"):
    """Encode designed residues as integer amino-acid indices by position."""
    _require(library in LIBRARIES, "unknown library {}".format(library))
    spec = LIBRARY_SPECS[library]
    peptide = frame[peptide_column].astype(str)
    affibody = frame[affibody_column].astype(str)
    _require(bool(peptide.str.len().eq(spec["pep_length"]).all()), "bad peptide code length")
    _require(bool(affibody.str.len().eq(spec["aff_length"]).all()), "bad Affibody code length")
    aa_index = {aa: index for index, aa in enumerate(AA_ALPHABET)}
    columns = []
    for index in range(spec["pep_length"]):
        values = peptide.str[index].map(aa_index)
        _require(bool(values.notna().all()), "unknown peptide amino acid")
        columns.append(values.to_numpy(dtype=np.int64))
    for index in range(spec["aff_length"]):
        values = affibody.str[index].map(aa_index)
        _require(bool(values.notna().all()), "unknown Affibody amino acid")
        columns.append(values.to_numpy(dtype=np.int64))
    return np.column_stack(columns).astype(np.int64, copy=False)


class AdditiveSiteLogit(nn.Module):
    """A linear one-hot model without materializing a dense one-hot matrix."""

    def __init__(self, n_positions, n_amino_acids=len(AA_ALPHABET)):
        super(AdditiveSiteLogit, self).__init__()
        self.n_positions = int(n_positions)
        self.n_amino_acids = int(n_amino_acids)
        self.weight = nn.Parameter(torch.zeros(self.n_positions, self.n_amino_acids))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, site_indices):
        _require(site_indices.ndim == 2, "site tensor must be two-dimensional")
        _require(site_indices.shape[1] == self.n_positions, "site tensor width mismatch")
        positions = torch.arange(self.n_positions, device=site_indices.device)
        return self.weight[positions[None, :], site_indices].sum(dim=1) + self.bias


def empirical_risk_components(positive_logits, negative_logits, unlabeled_logits=None):
    """Return the four empirical logistic-loss terms used by PNU learning."""
    _require(positive_logits.numel() > 0, "positive batch is empty")
    _require(negative_logits.numel() > 0, "negative batch is empty")
    rp_positive = F.softplus(-positive_logits).mean()
    rp_negative = F.softplus(positive_logits).mean()
    rn_negative = F.softplus(negative_logits).mean()
    ru_negative = None
    if unlabeled_logits is not None:
        _require(unlabeled_logits.numel() > 0, "unlabeled batch is empty")
        ru_negative = F.softplus(unlabeled_logits).mean()
    return {
        "rp_positive": rp_positive,
        "rp_negative": rp_negative,
        "rn_negative": rn_negative,
        "ru_negative": ru_negative,
    }


def pnu_objective(components, risk, class_prior=None, eta=None):
    """Compute balanced PN, nnPU, or the formal nonnegative PNU risk."""
    risk = str(risk).lower()
    _require(risk in ("pn", "nnpu", "nnpnu"), "unknown risk {}".format(risk))
    rp_positive = components["rp_positive"]
    rp_negative = components["rp_negative"]
    rn_negative = components["rn_negative"]
    if risk == "pn":
        value = 0.5 * rp_positive + 0.5 * rn_negative
        zero = torch.zeros_like(value)
        return value, {
            "positive_component": 0.5 * rp_positive,
            "corrected_component_raw": 0.5 * rn_negative,
            "corrected_component": 0.5 * rn_negative,
            "nonnegative_correction_active": zero,
        }

    pi = _finite_probability(class_prior, "class_prior")
    if risk == "nnpu":
        eta_value = 1.0
    else:
        eta_value = _finite_probability(eta, "eta", closed=True)
    ru_negative = components["ru_negative"]
    if eta_value > 0.0:
        _require(ru_negative is not None, "{} risk requires U logits".format(risk))
    elif ru_negative is None:
        # eta=0 is exactly prior-weighted PN; no U term is present.
        ru_negative = torch.zeros_like(rp_positive)
    positive_component = pi * rp_positive
    correction_raw = (
        (1.0 - eta_value) * (1.0 - pi) * rn_negative
        + eta_value * (ru_negative - pi * rp_negative)
    )
    corrected = torch.clamp(correction_raw, min=0.0)
    return positive_component + corrected, {
        "positive_component": positive_component,
        "corrected_component_raw": correction_raw,
        "corrected_component": corrected,
        "nonnegative_correction_active": (correction_raw < 0.0).to(rp_positive.dtype),
    }


def add_identity_blocks_fast(frame, library, n_folds):
    """Vectorized-over-identities equivalent of ``add_identity_blocks``."""
    output = frame.copy()
    peptide_uids = pd.unique(output["peptide_uid"])
    affibody_uids = pd.unique(output["affibody_uid"])
    peptide_map = {
        uid: stable_identity_bin(uid, n_folds, "{}|peptide".format(library))
        for uid in peptide_uids
    }
    affibody_map = {
        uid: stable_identity_bin(uid, n_folds, "{}|affibody".format(library))
        for uid in affibody_uids
    }
    output["peptide_block"] = output["peptide_uid"].map(peptide_map).astype(np.int16)
    output["affibody_block"] = output["affibody_uid"].map(affibody_map).astype(np.int16)
    return output


def identity_cold_u_train_indices(blocked_u, fold):
    """Keep only U rows sharing neither identity block with the held fold."""
    keep = ~blocked_u["peptide_block"].eq(int(fold)) & ~blocked_u["affibody_block"].eq(int(fold))
    return np.flatnonzero(keep.to_numpy())


def _cyclic_batch(indices, state, size, rng):
    """Draw a fixed-size shuffled batch, reshuffling when an arm is exhausted."""
    _require(len(indices) > 0, "cannot batch an empty arm")
    chunks = []
    needed = int(size)
    while needed > 0:
        if state["order"] is None or state["cursor"] >= len(state["order"]):
            state["order"] = rng.permutation(indices)
            state["cursor"] = 0
        take = min(needed, len(state["order"]) - state["cursor"])
        chunks.append(state["order"][state["cursor"] : state["cursor"] + take])
        state["cursor"] += take
        needed -= take
    return np.concatenate(chunks)


def _tensor_rows(encoded, rows, device):
    return torch.as_tensor(encoded[rows], dtype=torch.long, device=device)


def train_epoch(
    model,
    optimizer,
    p_encoded,
    n_encoded,
    u_encoded,
    risk,
    class_prior,
    eta,
    c_value,
    labeled_train_size,
    batch_size,
    device,
    rng,
):
    """Train one epoch and return exact row-coverage/correction diagnostics."""
    model.train()
    p_indices = np.arange(len(p_encoded), dtype=np.int64)
    n_indices = np.arange(len(n_encoded), dtype=np.int64)
    needs_u = risk == "nnpu" or (risk == "nnpnu" and float(eta) > 0.0)
    if needs_u:
        _require(u_encoded is not None and len(u_encoded) > 0, "U is required")
        u_order = rng.permutation(len(u_encoded))
        steps = int(math.ceil(len(u_order) / float(batch_size)))
    else:
        u_order = None
        steps = max(
            int(math.ceil(len(p_indices) / float(batch_size))),
            int(math.ceil(len(n_indices) / float(batch_size))),
        )
    p_state = {"order": None, "cursor": 0}
    n_state = {"order": None, "cursor": 0}
    objective_sum = 0.0
    correction_steps = 0
    u_rows_seen = 0
    for step in range(steps):
        if needs_u:
            start = step * int(batch_size)
            u_rows = u_order[start : start + int(batch_size)]
            current_size = len(u_rows)
        else:
            u_rows = None
            current_size = min(int(batch_size), max(len(p_indices), len(n_indices)))
        # P, N, and U contribute separate empirical means, so their minibatch
        # sizes need not match.  Avoid repeating a small arm merely to make it
        # as large as the much larger U minibatch.
        p_rows = _cyclic_batch(
            p_indices, p_state, min(current_size, len(p_indices)), rng
        )
        n_rows = _cyclic_batch(
            n_indices, n_state, min(current_size, len(n_indices)), rng
        )
        optimizer.zero_grad()
        p_logits = model(_tensor_rows(p_encoded, p_rows, device))
        n_logits = model(_tensor_rows(n_encoded, n_rows, device))
        u_logits = (
            model(_tensor_rows(u_encoded, u_rows, device)) if needs_u else None
        )
        components = empirical_risk_components(p_logits, n_logits, u_logits)
        objective, detail = pnu_objective(
            components,
            risk=risk,
            class_prior=class_prior,
            eta=eta,
        )
        # Same C interpretation as a sum-loss L2 logistic model, expressed on
        # the empirical mean scale.  The intercept is deliberately unpenalized.
        penalty = model.weight.square().sum() / (
            2.0 * float(c_value) * float(labeled_train_size)
        )
        total = objective + penalty
        total.backward()
        optimizer.step()
        objective_sum += float(objective.detach().cpu())
        correction_steps += int(float(detail["nonnegative_correction_active"].detach().cpu()) > 0.5)
        if needs_u:
            u_rows_seen += int(len(u_rows))
    if needs_u:
        _require(u_rows_seen == len(u_encoded), "an epoch did not visit every U row")
    return {
        "steps": int(steps),
        "mean_empirical_objective": objective_sum / float(steps),
        "nonnegative_correction_steps": int(correction_steps),
        "u_rows_seen": int(u_rows_seen),
        "u_rows_available": int(len(u_encoded)) if needs_u else 0,
    }


def predict_probability(model, encoded, device, batch_size=65536):
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(encoded), int(batch_size)):
            tensor = torch.as_tensor(
                encoded[start : start + int(batch_size)],
                dtype=torch.long,
                device=device,
            )
            blocks.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(blocks).astype(float, copy=False)


def validation_metrics(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(y_true) == {0, 1}, "weak validation needs both labels")
    _require(bool(np.isfinite(probability).all()), "nonfinite validation probability")
    return {
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
    }


def make_config(risk, c_value, class_prior=None, eta=None):
    risk = str(risk).lower()
    _require(risk in ("pn", "nnpu", "nnpnu"), "invalid risk")
    c_value = float(c_value)
    _require(math.isfinite(c_value) and c_value > 0.0, "C must be positive")
    if risk == "pn":
        class_prior = None
        eta = None
    else:
        class_prior = _finite_probability(class_prior, "class_prior")
        eta = 1.0 if risk == "nnpu" else _finite_probability(eta, "eta", closed=True)
    fields = [risk]
    if class_prior is not None:
        fields.append("pi{:g}".format(class_prior))
    if risk == "nnpnu":
        fields.append("eta{:g}".format(eta))
    fields.append("c{:g}".format(c_value))
    config_id = "_".join(fields).replace(".", "p")
    return {
        "config_id": config_id,
        "risk": risk,
        "class_prior": class_prior,
        "eta": eta,
        "C": c_value,
    }


def grid_configs(risks, pi_grid, eta_grid, c_grid):
    """Enumerate the prespecified PN/nnPU/nnPNU grid deterministically."""
    output = []
    for risk in risks:
        if risk == "pn":
            output.extend(make_config("pn", c_value) for c_value in c_grid)
        elif risk == "nnpu":
            for pi in pi_grid:
                for c_value in c_grid:
                    output.append(make_config("nnpu", c_value, class_prior=pi))
        elif risk == "nnpnu":
            for pi in pi_grid:
                for eta in eta_grid:
                    for c_value in c_grid:
                        output.append(
                            make_config(
                                "nnpnu", c_value, class_prior=pi, eta=eta
                            )
                        )
        else:
            raise ValueError("unknown grid risk {}".format(risk))
    ids = [item["config_id"] for item in output]
    _require(len(ids) == len(set(ids)), "duplicate grid config ID")
    return output


def _config_seed(base_seed, library, config_id, fold):
    payload = "{}|{}|{}|{}".format(base_seed, library, config_id, fold).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def cross_validate_config(
    config,
    library,
    blocked_labeled,
    blocked_u,
    labeled_encoded,
    u_encoded,
    folds,
    max_epochs,
    batch_size,
    learning_rate,
    device,
    seed,
    selection_metric,
):
    """Select an epoch using only pooled double-identity-cold weak labels."""
    epoch_observed = {epoch: [] for epoch in range(1, int(max_epochs) + 1)}
    epoch_probability = {epoch: [] for epoch in range(1, int(max_epochs) + 1)}
    fold_rows = []
    coverage_rows = []
    labels = blocked_labeled["weak_label"].astype(int).to_numpy()
    for fold in range(int(folds)):
        split = identity_blocked_indices(blocked_labeled, fold)
        audit_identity_cold_fold(blocked_labeled, split, fold)
        train_rows = split["train"]
        validation_rows = split["validation"]
        p_rows = train_rows[labels[train_rows] == 1]
        n_rows = train_rows[labels[train_rows] == 0]
        _require(len(p_rows) > 0 and len(n_rows) > 0, "empty fold P or N")
        uses_u = config["risk"] == "nnpu" or (
            config["risk"] == "nnpnu" and float(config["eta"]) > 0.0
        )
        if uses_u:
            u_rows = identity_cold_u_train_indices(blocked_u, fold)
            _require(len(u_rows) > 0, "empty fold U")
            fold_u_encoded = u_encoded[u_rows]
        else:
            u_rows = np.empty(0, dtype=np.int64)
            fold_u_encoded = None
        model = AdditiveSiteLogit(labeled_encoded.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        rng = np.random.RandomState(
            _config_seed(seed, library, config["config_id"], fold)
        )
        for epoch in range(1, int(max_epochs) + 1):
            coverage = train_epoch(
                model=model,
                optimizer=optimizer,
                p_encoded=labeled_encoded[p_rows],
                n_encoded=labeled_encoded[n_rows],
                u_encoded=fold_u_encoded,
                risk=config["risk"],
                class_prior=config["class_prior"],
                eta=config["eta"],
                c_value=config["C"],
                labeled_train_size=len(p_rows) + len(n_rows),
                batch_size=batch_size,
                device=device,
                rng=rng,
            )
            probability = predict_probability(
                model, labeled_encoded[validation_rows], device
            )
            observed = labels[validation_rows]
            metrics = validation_metrics(observed, probability)
            epoch_observed[epoch].extend(observed.tolist())
            epoch_probability[epoch].extend(probability.tolist())
            fold_rows.append(
                {
                    "library": library,
                    "config_id": config["config_id"],
                    "risk": config["risk"],
                    "class_prior": config["class_prior"],
                    "eta": config["eta"],
                    "C": config["C"],
                    "record_type": "fold",
                    "fold": int(fold),
                    "epoch": int(epoch),
                    "n_train_p": int(len(p_rows)),
                    "n_train_n": int(len(n_rows)),
                    "n_train_u": int(len(u_rows)),
                    "n_validation": int(len(validation_rows)),
                    "validation_positive": int(observed.sum()),
                    **metrics
                }
            )
            coverage_rows.append(
                {
                    "library": library,
                    "config_id": config["config_id"],
                    "fold": int(fold),
                    "epoch": int(epoch),
                    **coverage
                }
            )

    aggregate_rows = []
    for epoch in range(1, int(max_epochs) + 1):
        metrics = validation_metrics(epoch_observed[epoch], epoch_probability[epoch])
        aggregate_rows.append(
            {
                "library": library,
                "config_id": config["config_id"],
                "risk": config["risk"],
                "class_prior": config["class_prior"],
                "eta": config["eta"],
                "C": config["C"],
                "record_type": "aggregate",
                "fold": -1,
                "epoch": int(epoch),
                "n_train_p": -1,
                "n_train_n": -1,
                "n_train_u": -1,
                "n_validation": int(len(epoch_observed[epoch])),
                "validation_positive": int(sum(epoch_observed[epoch])),
                **metrics
            }
        )
    _require(selection_metric in ("log_loss", "auroc"), "invalid selection metric")
    if selection_metric == "log_loss":
        selection_key = lambda row: (
            row["log_loss"],
            -row["auroc"],
            -row["average_precision"],
            row["epoch"],
        )
    else:
        selection_key = lambda row: (
            -row["auroc"],
            -row["average_precision"],
            row["log_loss"],
            row["epoch"],
        )
    selected = min(aggregate_rows, key=selection_key)
    all_rows = fold_rows + aggregate_rows
    for row in all_rows:
        row["epoch_selected_within_config"] = int(
            row["record_type"] == "aggregate"
            and row["epoch"] == selected["epoch"]
        )
    result = dict(config)
    result.update(
        {
            "selected_epoch": int(selected["epoch"]),
            "validation_log_loss": float(selected["log_loss"]),
            "validation_auroc": float(selected["auroc"]),
            "validation_average_precision": float(selected["average_precision"]),
            "selection_metric": selection_metric,
        }
    )
    return result, pd.DataFrame(all_rows), pd.DataFrame(coverage_rows)


def selection_arm(config):
    if config["risk"] == "pn":
        return "pn_balanced"
    return "{}_pi{:g}".format(config["risk"], config["class_prior"])


def select_configs_with_weak_validation(config_results, selection_metric="log_loss"):
    """Select C/eta/epoch within PN or each fixed-pi arm; never across pi."""
    _require(selection_metric in ("log_loss", "auroc"), "invalid selection metric")
    frame = pd.DataFrame(config_results).copy()
    frame["selection_arm"] = [selection_arm(row) for row in config_results]
    frame["selected_within_fixed_pi"] = 0
    for _, indices in frame.groupby("selection_arm", sort=True).groups.items():
        block = frame.loc[list(indices)]
        def common_tail(index):
            return (
                float(frame.loc[index, "eta"])
                if pd.notna(frame.loc[index, "eta"])
                else -1.0,
                frame.loc[index, "C"],
                frame.loc[index, "selected_epoch"],
            )

        if selection_metric == "log_loss":
            key = lambda index: (
                frame.loc[index, "validation_log_loss"],
                -frame.loc[index, "validation_auroc"],
                -frame.loc[index, "validation_average_precision"],
            ) + common_tail(index)
        else:
            key = lambda index: (
                -frame.loc[index, "validation_auroc"],
                -frame.loc[index, "validation_average_precision"],
                frame.loc[index, "validation_log_loss"],
            ) + common_tail(index)
        best_index = min(block.index, key=key)
        frame.loc[best_index, "selected_within_fixed_pi"] = 1
    return frame


def fit_final_model(
    config,
    selected_epoch,
    p_encoded,
    n_encoded,
    u_encoded,
    batch_size,
    learning_rate,
    device,
    seed,
    library,
):
    model = AdditiveSiteLogit(p_encoded.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.RandomState(
        _config_seed(seed, library, config["config_id"], "refit")
    )
    audit = []
    for epoch in range(1, int(selected_epoch) + 1):
        coverage = train_epoch(
            model=model,
            optimizer=optimizer,
            p_encoded=p_encoded,
            n_encoded=n_encoded,
            u_encoded=u_encoded,
            risk=config["risk"],
            class_prior=config["class_prior"],
            eta=config["eta"],
            c_value=config["C"],
            labeled_train_size=len(p_encoded) + len(n_encoded),
            batch_size=batch_size,
            device=device,
            rng=rng,
        )
        audit.append({"epoch": int(epoch), **coverage})
    return model, pd.DataFrame(audit)


def retention_metrics(retention, probability):
    observed = retention["target_retention"].to_numpy(dtype=float)
    binder = retention["target_binder"].astype(int).to_numpy()
    probability = np.asarray(probability, dtype=float)
    _require(len(observed) == len(probability), "retention score length mismatch")
    global_rho = spearmanr(observed, probability)[0]
    within = []
    for _, group_indices in retention.groupby("peptide_uid", sort=True).groups.items():
        indices = np.asarray(list(group_indices), dtype=int)
        y_group = observed[indices]
        p_group = probability[indices]
        if np.unique(y_group).size < 2 or np.unique(p_group).size < 2:
            continue
        within.append(float(spearmanr(y_group, p_group)[0]))
    return {
        "n": int(len(retention)),
        "binders_ge_75": int(binder.sum()),
        "auroc": float(roc_auc_score(binder, probability)),
        "average_precision": float(average_precision_score(binder, probability)),
        "global_spearman": float(global_rho),
        "within_peptide_spearman": float(np.mean(within)) if within else float("nan"),
        "within_peptide_groups": int(len(within)),
    }


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
    return device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-label-dir", required=True, type=Path)
    parser.add_argument("--unlabeled-csv", required=True, type=Path)
    parser.add_argument("--unlabeled-manifest", type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library", choices=LIBRARIES, required=True)
    parser.add_argument("--mode", choices=("single", "grid"), default="single")
    parser.add_argument("--risk", choices=("pn", "nnpu", "nnpnu"), default="pn")
    parser.add_argument("--class-prior", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--grid-risks", nargs="+", choices=("pn", "nnpu", "nnpnu"), default=["pn", "nnpu", "nnpnu"])
    parser.add_argument("--pi-grid", nargs="+", type=float, default=list(DEFAULT_PI_GRID))
    parser.add_argument("--eta-grid", nargs="+", type=float, default=list(DEFAULT_ETA_GRID))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--selection-metric",
        choices=("log_loss", "auroc"),
        default="auroc",
        help="weak-label-only criterion for epoch and C/eta selection",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-negative-r001-count", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(int(args.folds) >= 2, "folds must be at least two")
    _require(int(args.max_epochs) >= 1, "max epochs must be positive")
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    _require(math.isfinite(float(args.learning_rate)) and float(args.learning_rate) > 0, "learning rate must be positive")
    _require(int(args.min_negative_r001_count) == 3, "this experiment preserves the current three-read negative rule")
    device = _device(args.device)

    if args.mode == "single":
        configs = [
            make_config(
                args.risk,
                args.c_value,
                class_prior=args.class_prior,
                eta=args.eta,
            )
        ]
    else:
        pi_grid = tuple(sorted(set(_finite_probability(value, "pi_grid") for value in args.pi_grid)))
        eta_grid = tuple(sorted(set(_finite_probability(value, "eta_grid", closed=True) for value in args.eta_grid)))
        c_grid = tuple(sorted(set(float(value) for value in args.c_grid)))
        _require(all(math.isfinite(value) and value > 0 for value in c_grid), "C grid must be positive")
        configs = grid_configs(args.grid_risks, pi_grid, eta_grid, c_grid)

    # Inputs and their hashes are fixed before any model fitting.
    weak_labels, _, weak_manifest, weak_paths = load_and_validate_weak_inputs(
        args.weak_label_dir
    )
    unlabeled, unlabeled_manifest = load_unlabeled_pool(
        args.unlabeled_csv, args.unlabeled_manifest
    )
    retention_identities = load_retention_identities(args.retention_csv)
    retention_identities["peptide_uid"] = [
        opaque_id(library, "pep", peptide)
        for library, peptide in zip(
            retention_identities["library"],
            retention_identities["peptide_design_code"],
        )
    ]
    retention_identities["affibody_uid"] = [
        opaque_id(library, "aff", affibody)
        for library, affibody in zip(
            retention_identities["library"],
            retention_identities["affibody_design_code"],
        )
    ]
    source_paths = {
        "weak_labels": weak_paths["weak_labels"],
        "weak_manifest": weak_paths["manifest"],
        "unlabeled_pool": args.unlabeled_csv,
        "unlabeled_manifest": args.unlabeled_manifest
        if args.unlabeled_manifest is not None
        else args.unlabeled_csv.parent / "manifest.json",
        "retention_csv": args.retention_csv,
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    _require(
        unlabeled_manifest["sources"]["established_positive_negative_labels"]["sha256"]
        == source_hashes["weak_labels"],
        "U pool and trainer use different weak-label sources",
    )
    _require(
        unlabeled_manifest["sources"]["retention_identities"]["sha256"]
        == source_hashes["retention_csv"],
        "U pool and trainer use different retention identity sources",
    )

    primary = primary_training_frame(
        weak_labels,
        args.library,
        min_negative_r001_count=args.min_negative_r001_count,
    )
    p_frame, n_frame, u_frame = validate_training_pools(
        primary, unlabeled, retention_identities, args.library
    )
    labeled = pd.concat([p_frame, n_frame], ignore_index=True)
    labeled = labeled.sort_values("pair_uid").reset_index(drop=True)
    blocked_labeled = add_identity_blocks(labeled, args.library, args.folds)
    blocked_u = add_identity_blocks_fast(u_frame, args.library, args.folds)
    labeled_encoded = site_index_matrix(labeled, args.library)
    u_encoded = site_index_matrix(u_frame, args.library)

    result_rows = []
    validation_blocks = []
    coverage_blocks = []
    for config in configs:
        result, validation, coverage = cross_validate_config(
            config=config,
            library=args.library,
            blocked_labeled=blocked_labeled,
            blocked_u=blocked_u,
            labeled_encoded=labeled_encoded,
            u_encoded=u_encoded,
            folds=args.folds,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            seed=args.seed,
            selection_metric=args.selection_metric,
        )
        result_rows.append(result)
        validation_blocks.append(validation)
        coverage_blocks.append(coverage)

    selected = select_configs_with_weak_validation(
        result_rows, selection_metric=args.selection_metric
    )
    # This is the last point at which any model/config choice is made.  Only
    # now is the measured retention subset materialized for evaluation.
    retention_all = load_retention_table(args.retention_csv)
    held = retention_all.loc[
        retention_all["library"].eq(args.library)
        & retention_all["target_retention"].notna()
    ].copy()
    held = held.sort_values("pair_uid").reset_index(drop=True)
    held_encoded = site_index_matrix(
        held,
        args.library,
        peptide_column="peptide_design_code",
        affibody_column="affibody_design_code",
    )
    p_encoded = site_index_matrix(p_frame, args.library)
    n_encoded = site_index_matrix(n_frame, args.library)

    metric_rows = []
    prediction_blocks = []
    refit_blocks = []
    for _, selected_row in selected.loc[selected["selected_within_fixed_pi"].eq(1)].iterrows():
        config = next(item for item in configs if item["config_id"] == selected_row["config_id"])
        model, refit_audit = fit_final_model(
            config=config,
            selected_epoch=int(selected_row["selected_epoch"]),
            p_encoded=p_encoded,
            n_encoded=n_encoded,
            u_encoded=u_encoded,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            seed=args.seed,
            library=args.library,
        )
        probability = predict_probability(model, held_encoded, device)
        metrics = retention_metrics(held, probability)
        metrics.update(
            {
                "library": args.library,
                "selection_arm": selected_row["selection_arm"],
                "config_id": config["config_id"],
                "risk": config["risk"],
                "class_prior": config["class_prior"],
                "eta": config["eta"],
                "C": config["C"],
                "selected_epoch": int(selected_row["selected_epoch"]),
                "selection_basis": "weak_double_identity_cold_{}_only".format(
                    args.selection_metric
                ),
            }
        )
        metric_rows.append(metrics)
        prediction = held[
            [
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
                "library",
                "target_retention",
                "target_binder",
            ]
        ].copy()
        prediction.insert(0, "config_id", config["config_id"])
        prediction.insert(1, "selection_arm", selected_row["selection_arm"])
        prediction["predicted_probability"] = probability
        prediction_blocks.append(prediction)
        refit_audit.insert(0, "config_id", config["config_id"])
        refit_audit.insert(1, "selection_arm", selected_row["selection_arm"])
        refit_blocks.append(refit_audit)

    validation = pd.concat(validation_blocks, ignore_index=True)
    coverage = pd.concat(coverage_blocks, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    refit_audit = pd.concat(refit_blocks, ignore_index=True)
    split_audit = blocked_labeled[
        [
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "weak_label",
            "peptide_block",
            "affibody_block",
        ]
    ].copy()
    split_audit.insert(0, "library", args.library)

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    output_paths = {
        "config_selection.csv": output_dir / "config_selection.csv",
        "validation_history.csv": output_dir / "validation_history.csv",
        "epoch_coverage.csv": output_dir / "epoch_coverage.csv",
        "split_audit.csv": output_dir / "split_audit.csv",
        "retention_metrics.csv": output_dir / "retention_metrics.csv",
        "retention_predictions.csv": output_dir / "retention_predictions.csv",
        "refit_history.csv": output_dir / "refit_history.csv",
    }
    write_private_csv(selected, output_paths["config_selection.csv"])
    write_private_csv(validation, output_paths["validation_history.csv"])
    write_private_csv(coverage, output_paths["epoch_coverage.csv"])
    write_private_csv(split_audit, output_paths["split_audit.csv"])
    write_private_csv(metrics, output_paths["retention_metrics.csv"])
    write_private_csv(predictions, output_paths["retention_predictions.csv"])
    write_private_csv(refit_audit, output_paths["refit_history.csv"])

    summary_lines = [
        "# PN / nnPU / nnPNU designed-position experiment",
        "",
        "Library: {}".format(args.library),
        "",
        "Training and model selection used only strict selection-derived P/N labels and the "
        "strict below-cutoff R009/R010 U pool. Retention was used only after every "
        "configuration and epoch choice was frozen.",
        "",
        "| Arm | P | N | U | Selected configuration | Weak-validation log loss | Retention AUROC | Retention AP | Within-peptide Spearman |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for _, metric in metrics.sort_values("selection_arm").iterrows():
        config_row = selected.loc[selected["config_id"].eq(metric["config_id"])].iloc[0]
        summary_lines.append(
            "| {selection_arm} | {positive_rows} | {negative_rows} | {unlabeled_rows} | {config_id} | {validation_log_loss:.4f} | "
            "{auroc:.4f} | {average_precision:.4f} | {within_peptide_spearman:.4f} |".format(
                positive_rows=len(p_frame),
                negative_rows=len(n_frame),
                unlabeled_rows=len(u_frame),
                validation_log_loss=config_row["validation_log_loss"],
                **metric.to_dict()
            )
        )
    summary_lines.extend(
        [
            "",
        "Class-prior arms are reported separately. No class prior was selected using retention.",
        "The numerical outputs are retrospective and are not calibrated retention probabilities.",
        "Because P is the high-count tail and U is the below-cutoff remainder, the standard "
        "random-labeled-positive assumption of PU learning is not satisfied. These are "
        "PNU-inspired empirical comparisons, not unbiased prevalence estimates.",
            "",
        ]
    )
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary_lines))
    os.chmod(str(summary_path), 0o600)
    output_paths["run_summary.md"] = summary_path

    _require(
        source_hashes == {name: sha256_file(path) for name, path in source_paths.items()},
        "an input changed during training",
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory",
        "library": args.library,
        "configuration": {
            "mode": args.mode,
            "model": "designed-position one-hot additive logit",
            "risks": sorted(set(item["risk"] for item in configs)),
            "risk_formula": "pi*Rp+ + max(0,(1-eta)*(1-pi)*Rn- + eta*(Ru- - pi*Rp-))",
            "balanced_pn_control": "0.5*Rp+ + 0.5*Rn-",
            "class_prior_grid": sorted(set(item["class_prior"] for item in configs if item["class_prior"] is not None)),
            "eta_grid": sorted(set(item["eta"] for item in configs if item["risk"] == "nnpnu")),
            "c_grid": sorted(set(item["C"] for item in configs)),
            "folds": int(args.folds),
            "max_epochs": int(args.max_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "selection": "C/eta/epoch by deterministic double-identity-cold weak-label {} within each fixed pi".format(
                args.selection_metric
            ),
            "selection_metric": args.selection_metric,
            "retention_usage": "final retrospective evaluation only",
            "retention_binder_threshold": RETENTION_BINDER_THRESHOLD,
            "pu_assumption_status": (
                "SCAR/unbiased-PU assumption not satisfied: P is the pooled-count top tail "
                "and U is the deterministic below-cutoff remainder"
            ),
        },
        "rows": {
            "positive": int(len(p_frame)),
            "negative": int(len(n_frame)),
            "unlabeled": int(len(u_frame)),
            "retention": int(len(held)),
        },
        "sources": {
            name: {"path": str(Path(path).resolve()), "sha256": digest}
            for name, (path, digest) in zip(
                source_paths.keys(),
                [(source_paths[name], source_hashes[name]) for name in source_paths],
            )
        },
        "source_manifests": {
            "weak_label_top_fraction": weak_manifest["configuration"]["top_fraction"],
            "unlabeled_manifest_sha256": source_hashes["unlabeled_manifest"],
        },
        "code": {str(script_path): sha256_file(script_path)},
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(output_paths.items())
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "privacy": {
            "directory": "0700",
            "files": "0600",
            "row_level_outputs_use_opaque_ids": True,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "library": args.library,
                "positive": len(p_frame),
                "negative": len(n_frame),
                "unlabeled": len(u_frame),
                "evaluated_arms": metrics["selection_arm"].tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run(parse_args())
