#!/usr/bin/env python
"""Compare LibA round-label definitions with one frozen-MINT head.

This is a retrospective, label-definition ablation.  It fits an independent
L2-logistic head for every arm and sampling replicate while keeping the MINT
backbone frozen.  Partner identities in the measured LibA retention panel are
removed before weak-label cross-validation or fitting.  Numerical retention
outcomes are loaded only after every retention prediction has been written to
an immutable, pre-target table.

The feature inputs deliberately consist of two archives: the immutable
R009/R010 cache and a delta archive for sequence pairs not present there.  A
deterministic overlap-sentinel set must agree between the archives before any
model is fitted.
"""

from __future__ import print_function

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from scipy.stats import rankdata, spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "mint-liba-late-round-label-ablation-v1"
FEATURE_NAME = "mint_chain_mean"
FEATURE_DIMENSION = 2560
LIBRARY = "LibA"
ARMS = (
    "r9_r10_raw",
    "r9_r14_raw",
    "r9_r14_equal_frequency",
    "r11_r14_raw",
)
ARM_ALIASES = {
    "A": "r9_r10_raw",
    "B": "r9_r14_raw",
    "C": "r9_r14_equal_frequency",
    "D": "r11_r14_raw",
    "r9_r10_raw": "r9_r10_raw",
    "r9_r14_raw": "r9_r14_raw",
    "r9_r14_equal_frequency": "r9_r14_equal_frequency",
    "r11_r14_raw": "r11_r14_raw",
}
ARM_LETTERS = dict(zip(ARMS, ("A", "B", "C", "D")))
ARM_ROLES = {
    "r9_r10_raw": "reference",
    "r9_r14_raw": "primary_later_round_comparison",
    "r9_r14_equal_frequency": "round_depth_sensitivity",
    "r11_r14_raw": "late_only_diagnostic",
}
SAMPLINGS = ("natural", "size_matched")
SIZE_MATCH_SEEDS = (20260811, 20260812, 20260813)
NATURAL_SEED = -1
CANONICAL_FOLDS = 3
CANONICAL_SPLIT_SEED = 17
CANONICAL_C_GRID = (0.001, 0.01, 0.1, 1.0)
EXPECTED_ARM_A_POSITIVE = 11320
EXPECTED_ARM_A_NEGATIVE = 11222
EXPECTED_RETENTION_ROWS = 108
EXPECTED_RETENTION_POSITIVE = 38
EXPECTED_PEPTIDE_GROUPS = 9
SOLVER_TOLERANCE = 1e-4
REFERENCE_MANIFEST_SHA256 = (
    "948c053bb2e26334697f7348c289bc18e1a8c46dc7befc3db4642350b569f2ce"
)
REFERENCE_PREDICTIONS_SHA256 = (
    "e4482e841e1f6ce0c0c23d46b20b5b726f9dac471ba570be5c224fee9773375c"
)
REFERENCE_OLD_CACHE_SHA256 = (
    "309ec02050820cd1a8c0dc39805dee858b70e46f451e331ab4b6c132086b6e00"
)
FEATURE_CONTRACT_PATHS = (
    ("model", "checkpoint", "sha256"),
    ("model", "config", "sha256"),
    ("model", "feature_dimension"),
    ("model", "feature_dtype"),
    ("model", "feature_name"),
    ("model", "layer"),
    ("model", "mint_source_tree_sha256"),
    ("model", "pair_collator_source", "sha256"),
    ("model", "pooling"),
    ("model", "sep_chains"),
    ("model", "use_multimer"),
    ("model", "chain_order"),
    ("model", "experimental_construct_linker_omitted"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _canonical_arm(value):
    value = str(value)
    _require(value in ARM_ALIASES, "unknown LibA label arm {}".format(value))
    return ARM_ALIASES[value]


def _stable_bin(value, axis, seed, folds):
    payload = "{}|{}|{}".format(axis, int(seed), value).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(folds)


def _size_match_rank(seed, pair_uid):
    payload = "size-match|{}|{}".format(int(seed), pair_uid).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _labeled_membership_sha256(frame):
    rows = sorted(
        "{}|{}".format(identity, int(label))
        for identity, label in zip(frame["sequence_pair_sha256"], frame["weak_label"])
    )
    return hashlib.sha256("\n".join(rows).encode("ascii")).hexdigest()


def _prediction_sha256(frame):
    columns = [
        "arm",
        "sampling",
        "subset_seed",
        "sequence_pair_sha256",
        "binder_probability",
    ]
    ordered = frame.loc[:, columns].sort_values(columns[:-1], kind="mergesort")
    rows = []
    for row in ordered.itertuples(index=False, name=None):
        rows.append("{}|{}|{}|{}|{:.17g}".format(*row))
    return hashlib.sha256("\n".join(rows).encode("ascii")).hexdigest()


def _nested_value(payload, path):
    value = payload
    for key in path:
        _require(isinstance(value, dict) and key in value, "manifest lacks {}".format(".".join(path)))
        value = value[key]
    return value


def _claimed_hashes(payload):
    """Yield resolved-path/hash claims from an arbitrarily nested manifest."""
    claims = {}

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                claims[str(Path(value["path"]).resolve())] = value["sha256"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return claims


def _verify_claimed_file(path, manifests, description):
    path = Path(path).resolve()
    _require(path.is_file(), "missing {} {}".format(description, path))
    expected = None
    for manifest in manifests:
        expected = _claimed_hashes(manifest).get(str(path), expected)
        outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
        named = outputs.get(path.name) if isinstance(outputs, dict) else None
        if isinstance(named, str):
            expected = named
        elif isinstance(named, dict) and isinstance(named.get("sha256"), str):
            expected = named["sha256"]
    _require(expected is not None, "{} is not hash-bound by its manifest".format(description))
    observed = sha256_file(path)
    _require(observed == expected, "{} hash disagrees with manifest".format(description))
    return observed


def validate_feature_contract(old_manifest, delta_manifest):
    """Require every model-defining MINT feature setting to be identical."""
    values = {}
    for path in FEATURE_CONTRACT_PATHS:
        old_value = _nested_value(old_manifest, path)
        delta_value = _nested_value(delta_manifest, path)
        _require(old_value == delta_value, "old/delta feature contract differs at {}".format(".".join(path)))
        values[".".join(path)] = old_value
    _require(values["model.feature_name"] == FEATURE_NAME, "unexpected MINT feature name")
    _require(int(values["model.feature_dimension"]) == FEATURE_DIMENSION, "unexpected MINT feature dimension")
    _require(values["model.feature_dtype"] == "float32", "unexpected MINT feature dtype")
    _require(int(values["model.layer"]) == 33, "unexpected MINT layer")
    _require(values["model.sep_chains"] is True, "MINT sep_chains contract changed")
    _require(values["model.use_multimer"] is True, "MINT use_multimer contract changed")
    _require(
        values["model.pooling"] == "separate residue mean for chain 1 and chain 2, concatenated",
        "MINT pooling contract changed",
    )
    _require(
        values["model.chain_order"] == ["smart-HLA-linker-peptide", "Affibody"],
        "MINT chain order changed",
    )
    return values


def _load_feature_npz(path, old):
    required = {
        "row_index",
        "library",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        FEATURE_NAME,
    }
    if old:
        required.update(("source_kind", "measurement_missing"))
    else:
        required.add("cache_role")
    with np.load(str(path), allow_pickle=False) as archive:
        _require(required.issubset(archive.files), "feature NPZ schema mismatch: {}".format(sorted(required.difference(archive.files))))
        arrays = {key: np.asarray(archive[key]).copy() for key in required}
    row_index = np.asarray(arrays.pop("row_index"), dtype=np.int64)
    order = np.argsort(row_index, kind="mergesort")
    row_index = row_index[order]
    _require(
        np.array_equal(row_index, np.arange(len(row_index), dtype=np.int64)),
        "feature NPZ row_index is incomplete or noncontiguous",
    )
    features = np.asarray(arrays.pop(FEATURE_NAME), dtype=np.float32)[order]
    _require(features.shape == (len(row_index), FEATURE_DIMENSION), "unexpected MINT feature matrix shape")
    _require(bool(np.isfinite(features).all()), "MINT feature matrix contains non-finite values")
    frame = pd.DataFrame({key: np.asarray(value)[order].astype(str) for key, value in arrays.items()})
    frame.insert(0, "row_index", row_index)
    if not old:
        _require(
            set(frame["cache_role"]) == {"missing", "overlap_sentinel"},
            "delta cache lacks a cache role",
        )
        frame["is_overlap_sentinel"] = frame["cache_role"].eq(
            "overlap_sentinel"
        ).astype(int)
    _require(not bool(frame["sequence_pair_sha256"].duplicated().any()), "duplicate pair in one feature NPZ")
    return frame, features


def validate_overlap_sentinels(old_frame, old_features, delta_frame, delta_features, tolerance):
    sentinel_flag = pd.to_numeric(delta_frame["is_overlap_sentinel"], errors="raise").astype(int)
    _require(bool(sentinel_flag.isin([0, 1]).all()), "invalid overlap-sentinel flag")
    sentinel = delta_frame.loc[sentinel_flag.eq(1)].copy()
    _require(not sentinel.empty, "delta cache contains no overlap sentinels")
    old_lookup = pd.Series(old_frame.index.to_numpy(dtype=int), index=old_frame["sequence_pair_sha256"])
    _require(bool(sentinel["sequence_pair_sha256"].isin(old_lookup.index).all()), "overlap sentinel is absent from old cache")
    records = []
    for index, row in sentinel.iterrows():
        old_index = int(old_lookup.loc[row["sequence_pair_sha256"]])
        difference = float(np.max(np.abs(delta_features[int(index)] - old_features[old_index])))
        records.append(
            {
                "sequence_pair_sha256": row["sequence_pair_sha256"],
                "old_row_index": int(old_frame.iloc[old_index]["row_index"]),
                "delta_row_index": int(row["row_index"]),
                "max_abs_difference": difference,
            }
        )
    audit = pd.DataFrame(records).sort_values("sequence_pair_sha256").reset_index(drop=True)
    maximum = float(audit["max_abs_difference"].max())
    _require(maximum <= float(tolerance), "old/delta overlap sentinel differs (max abs {})".format(maximum))
    return audit, maximum


def _normalize_membership(frame):
    aliases = {
        "sample": "sampling",
        "sample_type": "sampling",
        "sampling_scheme": "sampling",
        "seed": "subset_seed",
        "sampling_seed": "subset_seed",
        "label": "weak_label",
    }
    for source, destination in aliases.items():
        if destination not in frame and source in frame:
            frame = frame.rename(columns={source: destination})
    required = {
        "arm",
        "sampling",
        "subset_seed",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "sequence_pair_sha256",
        "weak_label",
    }
    _require(required.issubset(frame.columns), "membership schema mismatch: {}".format(sorted(required.difference(frame.columns))))
    output = frame.loc[:, sorted(required)].copy()
    output["arm"] = output["arm"].map(_canonical_arm)
    output["sampling"] = output["sampling"].astype(str)
    _require(set(output["sampling"]).issubset(SAMPLINGS), "unknown sampling scheme")
    output["subset_seed"] = pd.to_numeric(output["subset_seed"], errors="raise").astype(int)
    output["weak_label"] = pd.to_numeric(output["weak_label"], errors="raise").astype(int)
    _require(bool(output["weak_label"].isin([0, 1]).all()), "weak label is not binary")
    for column in required.difference(("subset_seed", "weak_label")):
        output[column] = output[column].astype(str)
        _require(bool(output[column].str.len().gt(0).all()), "empty membership {}".format(column))
    output.loc[output["sampling"].eq("natural"), "subset_seed"] = NATURAL_SEED
    keys = ["arm", "sampling", "subset_seed", "sequence_pair_sha256"]
    _require(not bool(output.duplicated(keys).any()), "duplicate condition/pair membership")
    return output.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _expected_conditions():
    return [(arm, "natural", NATURAL_SEED) for arm in ARMS] + [
        (arm, "size_matched", seed)
        for arm in ARMS[1:]
        for seed in SIZE_MATCH_SEEDS
    ]


def validate_memberships(membership, retention_identity):
    """Recompute LibA-local identity exclusion and size-match membership."""
    retention_peptides = set(retention_identity["chain1_sha256"].astype(str))
    retention_affibodies = set(retention_identity["chain2_sha256"].astype(str))
    locally_cold = ~membership["chain1_sha256"].isin(retention_peptides) & ~membership[
        "chain2_sha256"
    ].isin(retention_affibodies)
    removed = int((~locally_cold).sum())
    membership = membership.loc[locally_cold].copy().reset_index(drop=True)
    _require(not membership.empty, "LibA-local retention exclusion removed every weak row")
    observed = set(
        membership[["arm", "sampling", "subset_seed"]].itertuples(index=False, name=None)
    )
    expected = set(_expected_conditions())
    _require(observed == expected, "membership condition grid is incomplete")
    natural = {
        arm: membership.loc[
            membership["arm"].eq(arm)
            & membership["sampling"].eq("natural")
            & membership["subset_seed"].eq(NATURAL_SEED)
        ].copy()
        for arm in ARMS
    }
    for arm, frame in natural.items():
        _require(set(frame["weak_label"]) == {0, 1}, "{} natural membership lacks a class".format(arm))
        _require(not bool(frame["sequence_pair_sha256"].duplicated().any()), "{} natural pair duplicate".format(arm))
    baseline_counts = natural[ARMS[0]]["weak_label"].value_counts().to_dict()
    _require(
        baseline_counts == {1: EXPECTED_ARM_A_POSITIVE, 0: EXPECTED_ARM_A_NEGATIVE},
        "arm-A LibA-local count contract changed: {}".format(baseline_counts),
    )
    baseline_negatives = set(natural[ARMS[0]].loc[natural[ARMS[0]]["weak_label"].eq(0), "sequence_pair_sha256"])
    target_positive = int(baseline_counts[1])
    for arm in ARMS:
        arm_natural = natural[arm]
        _require(
            set(arm_natural.loc[arm_natural["weak_label"].eq(0), "sequence_pair_sha256"])
            == baseline_negatives,
            "negative membership changes across label arms",
        )
        candidates = arm_natural.loc[arm_natural["weak_label"].eq(1)].copy()
        _require(len(candidates) >= target_positive, "{} has too few positives for size matching".format(arm))
        for seed in (() if arm == ARMS[0] else SIZE_MATCH_SEEDS):
            selected = candidates.assign(
                _rank=[_size_match_rank(seed, uid) for uid in candidates["pair_uid"]]
            ).sort_values(["_rank", "pair_uid"], kind="mergesort").head(target_positive)
            expected_positive = set(selected["sequence_pair_sha256"])
            sampled = membership.loc[
                membership["arm"].eq(arm)
                & membership["sampling"].eq("size_matched")
                & membership["subset_seed"].eq(seed)
            ]
            _require(set(sampled["weak_label"]) == {0, 1}, "size-matched membership lacks a class")
            _require(
                set(sampled.loc[sampled["weak_label"].eq(0), "sequence_pair_sha256"])
                == baseline_negatives,
                "size matching changed negatives",
            )
            _require(
                set(sampled.loc[sampled["weak_label"].eq(1), "sequence_pair_sha256"])
                == expected_positive,
                "size-match membership disagrees with deterministic SHA256 ranking",
            )
    return membership, removed


def _normalize_reuse_index(frame):
    aliases = {
        "source": "feature_source",
        "cache_source": "feature_source",
        "row_index": "feature_row_index",
        "source_row_index": "feature_row_index",
    }
    for source, destination in aliases.items():
        if destination not in frame and source in frame:
            frame = frame.rename(columns={source: destination})
    if "feature_row_index" not in frame and {
        "feature_source",
        "old_cache_row_index",
        "delta_row_index",
    }.issubset(frame.columns):
        old_source = frame["feature_source"].astype(str).isin(
            ["old", "existing", "old_cache"]
        )
        frame["feature_row_index"] = np.where(
            old_source,
            frame["old_cache_row_index"],
            frame["delta_row_index"],
        )
    required = {
        "sequence_pair_sha256",
        "feature_source",
        "feature_row_index",
        "chain1_sha256",
        "chain2_sha256",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
    }
    _require(required.issubset(frame.columns), "reuse-index schema mismatch")
    output = frame.loc[:, sorted(required)].copy()
    output["sequence_pair_sha256"] = output["sequence_pair_sha256"].astype(str)
    output["feature_source"] = output["feature_source"].astype(str).str.lower().replace(
        {"existing": "old", "old_cache": "old", "delta_cache": "delta"}
    )
    _require(set(output["feature_source"]).issubset({"old", "delta"}), "unknown reuse feature source")
    output["feature_row_index"] = pd.to_numeric(output["feature_row_index"], errors="raise").astype(int)
    _require(not bool(output["sequence_pair_sha256"].duplicated().any()), "duplicate reuse-index pair")
    return output


def attach_membership_identities(membership, reuse):
    """Attach full-chain identities and independently validate design metadata."""
    columns = [
        "sequence_pair_sha256",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_sha256",
        "chain2_sha256",
    ]
    right = reuse.loc[:, columns].rename(
        columns={
            "pair_uid": "resolver_pair_uid",
            "peptide_design_code": "resolver_peptide_design_code",
            "affibody_design_code": "resolver_affibody_design_code",
        }
    )
    output = membership.merge(
        right, on="sequence_pair_sha256", how="left", validate="many_to_one"
    )
    _require(bool(output["chain1_sha256"].notna().all()), "membership identity is absent from reuse index")
    _require(
        bool(output["pair_uid"].eq(output["resolver_pair_uid"]).all()),
        "membership/reuse pair UID mismatch",
    )
    _require(
        bool(
            output["peptide_design_code"]
            .eq(output["resolver_peptide_design_code"])
            .all()
        ),
        "membership/reuse peptide code mismatch",
    )
    _require(
        bool(
            output["affibody_design_code"]
            .eq(output["resolver_affibody_design_code"])
            .all()
        ),
        "membership/reuse Affibody code mismatch",
    )
    return output.drop(
        columns=[
            "resolver_pair_uid",
            "resolver_peptide_design_code",
            "resolver_affibody_design_code",
        ]
    )


def validate_canonical_label_inputs(arm_labels_path, size_membership_path, membership):
    """Cross-check the cache-oriented membership against the CPU label artifact."""
    labels = pd.read_csv(
        arm_labels_path, dtype=str, keep_default_na=False, na_filter=False
    )
    required_labels = {
        "arm",
        "library",
        "pair_uid",
        "weak_label",
        "strict_liba_retention_identity_cold_eligible",
    }
    _require(required_labels.issubset(labels.columns), "canonical arm-label schema mismatch")
    labels["arm"] = labels["arm"].map(_canonical_arm)
    labels["weak_label"] = pd.to_numeric(labels["weak_label"], errors="raise").astype(int)
    labels["strict_liba_retention_identity_cold_eligible"] = pd.to_numeric(
        labels["strict_liba_retention_identity_cold_eligible"], errors="raise"
    ).astype(int)
    _require(bool(labels["library"].eq(LIBRARY).all()), "canonical labels include another library")
    _require(not bool(labels.duplicated(["arm", "pair_uid"]).any()), "duplicate canonical arm/pair")
    _require(
        bool(labels["strict_liba_retention_identity_cold_eligible"].eq(1).all()),
        "canonical arm-label artifact contains non-cold rows",
    )
    natural = membership.loc[membership["sampling"].eq("natural")].copy()
    left = natural[["arm", "pair_uid", "weak_label"]].sort_values(
        ["arm", "pair_uid"], kind="mergesort"
    ).reset_index(drop=True)
    right = labels[["arm", "pair_uid", "weak_label"]].sort_values(
        ["arm", "pair_uid"], kind="mergesort"
    ).reset_index(drop=True)
    _require(left.equals(right), "cache membership differs from canonical natural arm labels")

    sized = pd.read_csv(
        size_membership_path, dtype=str, keep_default_na=False, na_filter=False
    )
    required_sized = {"arm", "sampling_seed", "pair_uid", "rank_sha256"}
    _require(required_sized.issubset(sized.columns), "canonical size-membership schema mismatch")
    sized["arm"] = sized["arm"].map(_canonical_arm)
    sized["sampling_seed"] = pd.to_numeric(sized["sampling_seed"], errors="raise").astype(int)
    _require(set(sized["arm"]) == set(ARMS[1:]), "size-membership arms must be B/C/D")
    _require(set(sized["sampling_seed"]) == set(SIZE_MATCH_SEEDS), "size-membership seeds changed")
    _require(not bool(sized.duplicated(["arm", "sampling_seed", "pair_uid"]).any()), "duplicate canonical size membership")
    expected_rank = [
        _size_match_rank(seed, pair_uid)
        for seed, pair_uid in zip(sized["sampling_seed"], sized["pair_uid"])
    ]
    _require(sized["rank_sha256"].astype(str).tolist() == expected_rank, "canonical size-match SHA256 rank changed")
    for arm in ARMS[1:]:
        for seed in SIZE_MATCH_SEEDS:
            expected = set(
                sized.loc[
                    sized["arm"].eq(arm) & sized["sampling_seed"].eq(seed),
                    "pair_uid",
                ]
            )
            observed = set(
                membership.loc[
                    membership["arm"].eq(arm)
                    & membership["sampling"].eq("size_matched")
                    & membership["subset_seed"].eq(seed)
                    & membership["weak_label"].eq(1),
                    "pair_uid",
                ]
            )
            _require(observed == expected, "cache membership differs from canonical size-matched positives")
    return labels, sized


def load_legacy_retention_features(path, retention_identity, tolerance, cached_features):
    """Load target-free legacy retention features and validate cache parity."""
    required = {"pair_uid", "library", "measurement_missing", FEATURE_NAME}
    with np.load(str(path), allow_pickle=False) as archive:
        _require(required.issubset(archive.files), "legacy retention feature schema mismatch")
        table = pd.DataFrame(
            {
                "pair_uid": np.asarray(archive["pair_uid"]).astype(str),
                "legacy_library": np.asarray(archive["library"]).astype(str),
                "legacy_missing": np.asarray(archive["measurement_missing"]).astype(str),
            }
        )
        features = np.asarray(archive[FEATURE_NAME], dtype=np.float32).copy()
    table["legacy_index"] = np.arange(len(table), dtype=int)
    _require(features.shape == (len(table), FEATURE_DIMENSION), "legacy retention feature shape mismatch")
    _require(not bool(table["pair_uid"].duplicated().any()), "duplicate legacy retention pair")
    joined = retention_identity.merge(table, on="pair_uid", validate="one_to_one")
    _require(len(joined) == len(retention_identity), "legacy retention pair membership changed")
    _require(bool(joined["library"].eq(joined["legacy_library"]).all()), "legacy retention library mismatch")
    _require(
        bool(joined["measurement_missing"].astype(str).eq(joined["legacy_missing"]).all()),
        "legacy retention missingness mismatch",
    )
    ordered = features[joined["legacy_index"].to_numpy(dtype=int)]
    difference = float(np.max(np.abs(ordered - cached_features)))
    _require(difference <= float(tolerance), "old/legacy retention features differ (max abs {})".format(difference))
    return ordered, difference


def load_reference_invariant(run_dir, old_cache_path):
    """Load only target-free fields from the hash-locked arm-A reference."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "retention_predictions.csv"
    _require(manifest_path.is_file() and predictions_path.is_file(), "arm-A reference run is incomplete")
    _require(
        sha256_file(manifest_path) == REFERENCE_MANIFEST_SHA256,
        "arm-A reference manifest changed",
    )
    _require(
        sha256_file(predictions_path) == REFERENCE_PREDICTIONS_SHA256,
        "arm-A reference predictions changed",
    )
    _require(
        sha256_file(old_cache_path) == REFERENCE_OLD_CACHE_SHA256,
        "arm-A old feature cache changed",
    )
    predictions = pd.read_csv(
        predictions_path,
        usecols=["pair_uid", "sequence_pair_sha256", "arm", "probability"],
        dtype={"pair_uid": str, "sequence_pair_sha256": str, "arm": str},
    )
    predictions = predictions.loc[predictions["arm"].eq("frozen_logistic")].copy()
    predictions["probability"] = pd.to_numeric(predictions["probability"], errors="raise")
    _require(len(predictions) == EXPECTED_RETENTION_ROWS, "arm-A reference prediction count changed")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "predictions_path": predictions_path,
        "predictions_sha256": sha256_file(predictions_path),
        "predictions": predictions,
        "selected_c": 0.01,
    }


def load_reference_metrics_after_freeze(reference, old_cache_path):
    """Read outcome-derived reference metrics only after predictions are frozen."""
    manifest = _read_json(reference["manifest_path"])
    configuration = manifest.get("configuration", {})
    _require(int(configuration.get("folds", -1)) == CANONICAL_FOLDS, "arm-A reference fold count changed")
    _require(int(configuration.get("split_seed", -1)) == CANONICAL_SPLIT_SEED, "arm-A reference split seed changed")
    _require(tuple(float(value) for value in configuration.get("c_grid", [])) == CANONICAL_C_GRID, "arm-A reference C grid changed")
    cache_claim = manifest.get("sources", {}).get("cache_npz_000", {})
    _require(cache_claim.get("sha256") == sha256_file(old_cache_path), "arm-A reference used a different old feature cache")
    reference_record = manifest.get("reference_primary_run", {})
    _require(float(reference_record.get("selected_c", float("nan"))) == 0.01, "arm-A reference selected C changed")
    expected_metrics = reference_record.get("metrics", {})
    required_metrics = {
        "within_peptide_macro_spearman",
        "global_spearman",
        "global_auroc",
        "global_auprc",
    }
    _require(required_metrics.issubset(expected_metrics), "arm-A reference metrics are incomplete")
    output = dict(reference)
    output["metrics"] = expected_metrics
    return output


def validate_arm_a_frozen_predictions(frozen_predictions, selected_c, reference, tolerance):
    _require(float(selected_c) == float(reference["selected_c"]), "arm-A selected C does not reproduce reference")
    observed = frozen_predictions.loc[
        frozen_predictions["arm"].eq(ARMS[0])
        & frozen_predictions["sampling"].eq("natural")
        & frozen_predictions["subset_seed"].eq(NATURAL_SEED),
        [
            "pair_uid",
            "sequence_pair_sha256",
            "chain1_sha256",
            "binder_probability",
        ],
    ].copy()
    expected = reference["predictions"].rename(columns={"probability": "reference_probability"})
    joined = observed.merge(
        expected[["pair_uid", "sequence_pair_sha256", "reference_probability"]],
        on=["pair_uid", "sequence_pair_sha256"],
        validate="one_to_one",
    )
    _require(len(joined) == EXPECTED_RETENTION_ROWS, "arm-A/reference prediction identities differ")
    maximum = float(
        np.max(np.abs(joined["binder_probability"] - joined["reference_probability"]))
    )
    _require(maximum <= float(tolerance), "arm-A predictions do not reproduce reference (max abs {})".format(maximum))
    for peptide_identity, group in joined.groupby("chain1_sha256", sort=True):
        _require(
            np.array_equal(
                rankdata(
                    group["binder_probability"].to_numpy(dtype=float),
                    method="average",
                ),
                rankdata(
                    group["reference_probability"].to_numpy(dtype=float),
                    method="average",
                ),
            ),
            "arm-A within-peptide ranking does not exactly reproduce reference for {}".format(
                peptide_identity
            ),
        )
    global_rank_rho = _safe_spearman(
        joined["binder_probability"], joined["reference_probability"]
    )
    return {
        "probability_max_abs_difference": maximum,
        "within_peptide_rankings_exact": True,
        "global_probability_rank_spearman": global_rank_rho,
    }


def validate_arm_a_metrics(metrics, reference, primary_tolerance, global_tolerance):
    row = metrics.loc[
        metrics["arm"].eq(ARMS[0])
        & metrics["sampling"].eq("natural")
        & metrics["subset_seed"].eq(NATURAL_SEED)
    ]
    _require(len(row) == 1, "missing natural arm-A retention metrics")
    row = row.iloc[0]
    mapping = {
        "within_peptide_macro_spearman": "within_peptide_macro_spearman",
        "global_spearman": "global_spearman",
        "global_auroc": "global_auroc",
        "global_average_precision": "global_auprc",
    }
    differences = {}
    for observed_name, reference_name in mapping.items():
        difference = abs(float(row[observed_name]) - float(reference["metrics"][reference_name]))
        tolerance = (
            float(primary_tolerance)
            if observed_name == "within_peptide_macro_spearman"
            else float(global_tolerance)
        )
        _require(difference <= tolerance, "arm-A {} does not reproduce reference".format(observed_name))
        differences[observed_name] = difference
    return differences


def resolve_reuse_features(reuse, old_frame, old_features, delta_frame, delta_features):
    """Resolve each unique pair once; condition memberships index this matrix."""
    old_by_row = pd.Series(old_frame.index.to_numpy(dtype=int), index=old_frame["row_index"].astype(int))
    delta_by_row = pd.Series(delta_frame.index.to_numpy(dtype=int), index=delta_frame["row_index"].astype(int))
    feature_rows = []
    for record in reuse.itertuples(index=False):
        identity = str(record.sequence_pair_sha256)
        source = str(record.feature_source)
        row = int(record.feature_row_index)
        if source == "old":
            _require(row in old_by_row.index, "reuse index points outside old cache")
            position = int(old_by_row.loc[row])
            _require(old_frame.iloc[position]["sequence_pair_sha256"] == identity, "reuse old identity mismatch")
            feature_rows.append(old_features[position])
        else:
            _require(row in delta_by_row.index, "reuse index points outside delta cache")
            position = int(delta_by_row.loc[row])
            _require(delta_frame.iloc[position]["sequence_pair_sha256"] == identity, "reuse delta identity mismatch")
            sentinel = int(pd.to_numeric(delta_frame.iloc[position]["is_overlap_sentinel"], errors="raise"))
            _require(sentinel == 0, "reuse index uses a delta overlap sentinel as the primary feature")
            feature_rows.append(delta_features[position])
    features = np.asarray(feature_rows, dtype=np.float32)
    _require(features.shape == (len(reuse), FEATURE_DIMENSION), "resolved MINT feature shape mismatch")
    return features


def make_diagonal_folds(frame, folds=CANONICAL_FOLDS, split_seed=CANONICAL_SPLIT_SEED):
    peptide_fold = np.asarray(
        [_stable_bin(value, "peptide", split_seed, folds) for value in frame["chain1_sha256"]],
        dtype=int,
    )
    affibody_fold = np.asarray(
        [_stable_bin(value, "affibody", split_seed, folds) for value in frame["chain2_sha256"]],
        dtype=int,
    )
    plans = []
    for fold in range(int(folds)):
        validation = (peptide_fold == fold) & (affibody_fold == fold)
        train = (peptide_fold != fold) & (affibody_fold != fold)
        guard = ~(train | validation)
        _require(bool(train.any()) and bool(validation.any()), "empty diagonal fold {}".format(fold))
        train_frame = frame.loc[train]
        validation_frame = frame.loc[validation]
        _require(set(train_frame["weak_label"]) == {0, 1}, "fold train lacks a class")
        _require(set(validation_frame["weak_label"]) == {0, 1}, "fold validation lacks a class")
        _require(
            set(train_frame["chain1_sha256"]).isdisjoint(validation_frame["chain1_sha256"]),
            "fold shares peptide identities",
        )
        _require(
            set(train_frame["chain2_sha256"]).isdisjoint(validation_frame["chain2_sha256"]),
            "fold shares Affibody identities",
        )
        plans.append(
            {
                "fold": int(fold),
                "train": np.flatnonzero(train),
                "guard": np.flatnonzero(guard),
                "validation": np.flatnonzero(validation),
                "peptide_fold": peptide_fold,
                "affibody_fold": affibody_fold,
            }
        )
    return plans


def _standardize_fit(train):
    values = np.asarray(train, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


def _fit_probability(x_train, labels, x_prediction, c_value):
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        max_iter=2000,
        tol=SOLVER_TOLERANCE,
    )
    model.fit(x_train, labels)
    _require(int(model.n_iter_[0]) < int(model.max_iter), "frozen-MINT logistic head did not converge")
    probability = model.predict_proba(x_prediction)[:, 1]
    _require(bool(np.isfinite(probability).all()), "non-finite logistic probability")
    return np.asarray(probability, dtype=float), model


def _binary_metrics(labels, probability):
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(set(labels) == {0, 1}, "binary metrics require both classes")
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
    }


def fit_condition(frame, features, retention_features, c_grid, folds, split_seed):
    labels = frame["weak_label"].to_numpy(dtype=int)
    plans = make_diagonal_folds(frame, folds=folds, split_seed=split_seed)
    tuning_rows = []
    validation_predictions = []
    fold_memberships = []
    pooled = []
    prepared = []
    for plan in plans:
        train = plan["train"]
        validation = plan["validation"]
        mean, scale = _standardize_fit(features[train])
        x_train = ((features[train] - mean) / scale).astype(np.float32)
        x_validation = ((features[validation] - mean) / scale).astype(np.float32)
        prepared.append((plan, x_train, x_validation))
        roles = np.full(len(frame), "guard", dtype="U10")
        roles[train] = "train"
        roles[validation] = "validation"
        block = frame[
            ["pair_uid", "sequence_pair_sha256", "chain1_sha256", "chain2_sha256", "weak_label"]
        ].copy()
        block["fold"] = int(plan["fold"])
        block["role"] = roles
        fold_memberships.append(block)
    for c_value in c_grid:
        pooled_labels = []
        pooled_probability = []
        for plan, x_train, x_validation in prepared:
            probability, _ = _fit_probability(
                x_train,
                labels[plan["train"]],
                x_validation,
                c_value,
            )
            observed = labels[plan["validation"]]
            metrics = _binary_metrics(observed, probability)
            tuning_rows.append(
                {
                    "record_type": "fold",
                    "C": float(c_value),
                    "fold": int(plan["fold"]),
                    "n_train": int(len(plan["train"])),
                    "n_guard": int(len(plan["guard"])),
                    "train_membership_sha256": _labeled_membership_sha256(frame.iloc[plan["train"]]),
                    "validation_membership_sha256": _labeled_membership_sha256(frame.iloc[plan["validation"]]),
                    **metrics
                }
            )
            prediction = frame.iloc[plan["validation"]][
                ["pair_uid", "sequence_pair_sha256", "weak_label"]
            ].copy()
            prediction["fold"] = int(plan["fold"])
            prediction["C"] = float(c_value)
            prediction["binder_probability"] = probability
            validation_predictions.append(prediction)
            pooled_labels.extend(observed.tolist())
            pooled_probability.extend(probability.tolist())
        metrics = _binary_metrics(pooled_labels, pooled_probability)
        row = {
            "record_type": "aggregate",
            "C": float(c_value),
            "fold": -1,
            "n_train": int(sum(len(plan["train"]) for plan in plans)),
            "n_guard": int(sum(len(plan["guard"]) for plan in plans)),
            "train_membership_sha256": "multiple_folds",
            "validation_membership_sha256": "multiple_folds",
            **metrics
        }
        tuning_rows.append(row)
        pooled.append(row)
    selected = min(
        pooled,
        key=lambda row: (
            row["log_loss"],
            -row["average_precision"],
            -row["auroc"],
            row["C"],
        ),
    )
    for row in tuning_rows:
        row["selected"] = int(float(row["C"]) == float(selected["C"]))
    mean, scale = _standardize_fit(features)
    probability, model = _fit_probability(
        ((features - mean) / scale).astype(np.float32),
        labels,
        ((retention_features - mean) / scale).astype(np.float32),
        selected["C"],
    )
    return {
        "selected_c": float(selected["C"]),
        "probability": probability,
        "model_nonzero_coefficients": int(np.count_nonzero(model.coef_)),
        "tuning": pd.DataFrame(tuning_rows),
        "validation_predictions": pd.concat(validation_predictions, ignore_index=True),
        "fold_membership": pd.concat(fold_memberships, ignore_index=True),
    }


def _safe_spearman(observed, score):
    observed = np.asarray(observed, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(observed) < 2 or len(np.unique(observed)) < 2 or len(np.unique(score)) < 2:
        return float("nan")
    return float(spearmanr(observed, score)[0])


def _tie_aware_top_choice(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    maximum = np.max(scores)
    tied = scores == maximum
    _require(bool(tied.any()), "top-choice tie set is empty")
    return float(labels[tied].mean()), int(tied.sum())


def retention_metrics(frame):
    _require(len(frame) == EXPECTED_RETENTION_ROWS, "retention panel must contain 108 LibA pairs")
    labels = frame["target_binder"].to_numpy(dtype=int)
    observed = frame["target_retention"].to_numpy(dtype=float)
    score = frame["binder_probability"].to_numpy(dtype=float)
    _require(set(labels) == {0, 1}, "retention panel lacks a binder class")
    per_peptide = []
    for identity, group in frame.groupby("chain1_sha256", sort=True):
        rho = _safe_spearman(group["target_retention"], group["binder_probability"])
        top, tied = _tie_aware_top_choice(group["target_binder"], group["binder_probability"])
        group_labels = group["target_binder"].to_numpy(dtype=int)
        group_score = group["binder_probability"].to_numpy(dtype=float)
        binary_evaluable = set(group_labels) == {0, 1}
        per_peptide.append(
            {
                "peptide_identity": identity,
                "n": int(len(group)),
                "positive": int(group_labels.sum()),
                "spearman": rho,
                "top_choice_binder": top,
                "top_choice_tied_candidates": tied,
                "auroc": float(roc_auc_score(group_labels, group_score)) if binary_evaluable else float("nan"),
                "average_precision": float(average_precision_score(group_labels, group_score)) if binary_evaluable else float("nan"),
            }
        )
    per_peptide = pd.DataFrame(per_peptide)
    _require(len(per_peptide) == EXPECTED_PEPTIDE_GROUPS, "retention panel must contain exactly nine peptide identities")
    _require(bool(np.isfinite(per_peptide["spearman"]).all()), "a peptide group has undefined Spearman")
    return {
        "n": int(len(frame)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "global_spearman": _safe_spearman(observed, score),
        "global_auroc": float(roc_auc_score(labels, score)),
        "global_average_precision": float(average_precision_score(labels, score)),
        "within_peptide_macro_spearman": float(per_peptide["spearman"].mean()),
        "within_peptide_top_choice_binder": float(per_peptide["top_choice_binder"].mean()),
        "within_peptide_groups": int(len(per_peptide)),
    }, per_peptide


def load_retention_targets_after_freeze(rows_csv, retention_identity):
    """Read numerical retention values only after predictions are frozen."""
    columns = [
        "source_kind",
        "library",
        "pair_uid",
        "measurement_missing",
        "target_retention",
        "target_binder",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    target = pd.read_csv(
        rows_csv,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    target = target.loc[
        target["source_kind"].eq("retention")
        & target["library"].eq(LIBRARY)
        & target["measurement_missing"].eq("0")
    ].copy()
    target["target_retention"] = pd.to_numeric(target["target_retention"], errors="raise")
    target["target_binder"] = pd.to_numeric(target["target_binder"], errors="raise").astype(int)
    _require(
        np.array_equal(
            target["target_binder"].to_numpy(dtype=int),
            target["target_retention"].ge(75.0).astype(int).to_numpy(),
        ),
        "retention binder label is not retention >= 75%",
    )
    identity_columns = ["pair_uid", "chain1_sha256", "chain2_sha256", "sequence_pair_sha256"]
    left = retention_identity.loc[:, identity_columns].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    right = target.loc[:, identity_columns].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(left.astype(str).equals(right.astype(str)), "retention targets differ from frozen prediction identities")
    output = retention_identity.merge(
        target[["pair_uid", "target_retention", "target_binder"]],
        on="pair_uid",
        validate="one_to_one",
    ).sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(len(output) == EXPECTED_RETENTION_ROWS, "retention row count changed")
    _require(int(output["target_binder"].sum()) == EXPECTED_RETENTION_POSITIVE, "retention positive count changed")
    return output


def paired_deltas(metric_frame, per_peptide):
    metric_names = [
        "within_peptide_macro_spearman",
        "global_spearman",
        "global_auroc",
        "global_average_precision",
        "within_peptide_top_choice_binder",
    ]
    delta_rows = []
    peptide_rows = []
    comparisons = [
        (arm, "natural", NATURAL_SEED) for arm in ARMS[1:]
    ] + [
        (arm, "size_matched", seed)
        for arm in ARMS[1:]
        for seed in SIZE_MATCH_SEEDS
    ]
    for arm, sampling, seed in comparisons:
        reference = metric_frame.loc[
            metric_frame["arm"].eq(ARMS[0])
            & metric_frame["sampling"].eq("natural")
            & metric_frame["subset_seed"].eq(NATURAL_SEED)
        ]
        _require(len(reference) == 1, "missing arm-A metric reference")
        reference = reference.iloc[0]
        reference_peptide = per_peptide.loc[
            per_peptide["arm"].eq(ARMS[0])
            & per_peptide["sampling"].eq("natural")
            & per_peptide["subset_seed"].eq(NATURAL_SEED)
        ].set_index("peptide_identity")
        comparison = metric_frame.loc[
            metric_frame["arm"].eq(arm)
            & metric_frame["sampling"].eq(sampling)
            & metric_frame["subset_seed"].eq(seed)
        ]
        _require(len(comparison) == 1, "missing comparison metric")
        comparison = comparison.iloc[0]
        row = {
            "arm": arm,
            "arm_letter": ARM_LETTERS[arm],
            "arm_role": ARM_ROLES[arm],
            "reference_arm": ARMS[0],
            "reference_sampling": "natural",
            "reference_subset_seed": NATURAL_SEED,
            "sampling": sampling,
            "subset_seed": int(seed),
        }
        for metric in metric_names:
            row[metric + "_delta"] = float(comparison[metric] - reference[metric])
        comparison_peptide = per_peptide.loc[
            per_peptide["arm"].eq(arm)
            & per_peptide["sampling"].eq(sampling)
            & per_peptide["subset_seed"].eq(seed)
        ].set_index("peptide_identity")
        _require(set(comparison_peptide.index) == set(reference_peptide.index), "peptide metric membership changes across arms")
        differences = comparison_peptide["spearman"] - reference_peptide["spearman"]
        row["peptide_groups_spearman_improved"] = int((differences > 0.0).sum())
        row["peptide_groups_spearman_tied"] = int((differences == 0.0).sum())
        row["peptide_groups_total"] = int(len(differences))
        delta_rows.append(row)
        for peptide_identity in sorted(differences.index):
            peptide_rows.append(
                {
                    "arm": arm,
                    "arm_letter": ARM_LETTERS[arm],
                    "arm_role": ARM_ROLES[arm],
                    "reference_arm": ARMS[0],
                    "reference_sampling": "natural",
                    "reference_subset_seed": NATURAL_SEED,
                    "sampling": sampling,
                    "subset_seed": int(seed),
                    "peptide_identity": peptide_identity,
                    "spearman_delta": float(differences.loc[peptide_identity]),
                    "top_choice_binder_delta": float(
                        comparison_peptide.loc[peptide_identity, "top_choice_binder"]
                        - reference_peptide.loc[peptide_identity, "top_choice_binder"]
                    ),
                }
            )
    return pd.DataFrame(delta_rows), pd.DataFrame(peptide_rows)


def peptide_cluster_bootstrap(per_peptide_deltas, draws, seed):
    """Exploratory paired bootstrap over the nine peptide identities."""
    _require(int(draws) >= 1, "bootstrap draws must be positive")
    records = []
    endpoints = ("spearman_delta", "top_choice_binder_delta")
    groups = list(
        per_peptide_deltas.groupby(["arm", "sampling", "subset_seed"], sort=True)
    )
    for group_index, (key, frame) in enumerate(groups):
        arm, sampling, subset_seed = key
        frame = frame.sort_values("peptide_identity", kind="mergesort").reset_index(drop=True)
        _require(len(frame) == EXPECTED_PEPTIDE_GROUPS, "bootstrap comparison lacks nine peptide clusters")
        derived_seed = int(hashlib.sha256(
            "bootstrap|{}|{}|{}|{}".format(seed, arm, sampling, subset_seed).encode("ascii")
        ).hexdigest()[:8], 16)
        rng = np.random.RandomState(derived_seed)
        sampled = rng.randint(0, len(frame), size=(int(draws), len(frame)))
        for endpoint in endpoints:
            values = frame[endpoint].to_numpy(dtype=float)
            estimates = values[sampled].mean(axis=1)
            records.append(
                {
                    "arm": arm,
                    "arm_letter": ARM_LETTERS[arm],
                    "arm_role": ARM_ROLES[arm],
                    "reference_arm": ARMS[0],
                    "sampling": sampling,
                    "subset_seed": int(subset_seed),
                    "endpoint": endpoint,
                    "observed_delta": float(values.mean()),
                    "bootstrap_mean": float(estimates.mean()),
                    "ci_2_5": float(np.percentile(estimates, 2.5)),
                    "ci_97_5": float(np.percentile(estimates, 97.5)),
                    "fraction_above_zero": float((estimates > 0.0).mean()),
                    "draws": int(draws),
                    "bootstrap_seed": int(derived_seed),
                    "interpretation": "exploratory paired peptide-cluster bootstrap",
                }
            )
    return pd.DataFrame(records)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(int(args.folds) == CANONICAL_FOLDS, "this comparison requires exactly three folds")
    _require(int(args.split_seed) == CANONICAL_SPLIT_SEED, "split seed must be 17")
    c_grid = tuple(float(value) for value in args.c_grid)
    _require(c_grid == CANONICAL_C_GRID, "C grid differs from the prespecified canonical grid")
    _require(tuple(int(value) for value in args.size_match_seeds) == SIZE_MATCH_SEEDS, "size-match seeds changed")
    _require(1 <= int(args.jobs) <= 4, "jobs must be between one and four")

    old_manifest = _read_json(args.old_cache_manifest)
    delta_manifest = _read_json(args.delta_cache_manifest)
    prepare_manifest = _read_json(args.delta_prepare_manifest)
    rows_manifest = _read_json(args.old_cache_rows_manifest)
    label_manifest = _read_json(args.label_manifest)
    retention_feature_manifest = _read_json(args.retention_features_manifest)
    contract = validate_feature_contract(old_manifest, delta_manifest)
    _verify_claimed_file(args.old_cache, [old_manifest], "old cache")
    _verify_claimed_file(args.delta_cache, [delta_manifest], "delta cache")
    _verify_claimed_file(args.membership, [prepare_manifest, delta_manifest], "label membership")
    _verify_claimed_file(args.reuse_index, [prepare_manifest, delta_manifest], "reuse index")
    _verify_claimed_file(args.old_cache_rows, [rows_manifest, old_manifest], "old cache rows")
    _verify_claimed_file(args.arm_labels, [label_manifest, prepare_manifest], "canonical arm labels")
    _verify_claimed_file(
        args.size_matched_membership,
        [label_manifest, prepare_manifest],
        "canonical size-matched membership",
    )
    _verify_claimed_file(
        args.retention_features,
        [retention_feature_manifest],
        "legacy retention features",
    )
    reference = load_reference_invariant(args.reference_run_dir, args.old_cache)

    old_frame, old_features = _load_feature_npz(args.old_cache, old=True)
    delta_frame, delta_features = _load_feature_npz(args.delta_cache, old=False)
    overlap_audit, overlap_maximum = validate_overlap_sentinels(
        old_frame,
        old_features,
        delta_frame,
        delta_features,
        args.max_overlap_feature_difference,
    )
    retention_flag = (
        old_frame["source_kind"].eq("retention")
        & old_frame["library"].eq(LIBRARY)
        & old_frame["measurement_missing"].eq("0")
    )
    retention_identity = old_frame.loc[retention_flag].copy()
    retention_identity["_old_position"] = retention_identity.index.to_numpy(dtype=int)
    retention_identity = retention_identity.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(len(retention_identity) == EXPECTED_RETENTION_ROWS, "old cache does not contain 108 measured LibA pairs")
    _require(retention_identity["chain1_sha256"].nunique() == EXPECTED_PEPTIDE_GROUPS, "retention panel does not contain nine peptide identities")
    cached_retention_features = old_features[
        retention_identity["_old_position"].to_numpy(dtype=int)
    ]
    retention_features, retention_feature_difference = load_legacy_retention_features(
        args.retention_features,
        retention_identity,
        args.max_retention_feature_difference,
        cached_retention_features,
    )

    raw_membership = pd.read_csv(args.membership, dtype=str, keep_default_na=False, na_filter=False)
    membership = _normalize_membership(raw_membership)
    reuse = _normalize_reuse_index(
        pd.read_csv(args.reuse_index, dtype=str, keep_default_na=False, na_filter=False)
    )
    membership = attach_membership_identities(membership, reuse)
    membership, local_identity_rows_removed = validate_memberships(
        membership, retention_identity
    )
    canonical_labels, canonical_size_membership = validate_canonical_label_inputs(
        args.arm_labels, args.size_matched_membership, membership
    )
    resolved_features = resolve_reuse_features(
        reuse,
        old_frame,
        old_features,
        delta_frame,
        delta_features,
    )
    resolver_positions = pd.Series(
        np.arange(len(reuse), dtype=int), index=reuse["sequence_pair_sha256"]
    )
    _require(
        bool(membership["sequence_pair_sha256"].isin(resolver_positions.index).all()),
        "membership pair absent from reuse index",
    )
    membership["_resolver_position"] = membership["sequence_pair_sha256"].map(
        resolver_positions
    ).astype(int)

    # Output creation is intentionally delayed until every input contract has
    # passed, but occurs before the expensive CPU fits.  Existing directories
    # are never reused or overwritten.
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    overlap_path = output_dir / "overlap_sentinel_audit.csv"
    _write_csv(overlap_audit, overlap_path)

    reuse_by_identity = reuse.set_index("sequence_pair_sha256")

    def fit_one_condition(condition):
        arm, sampling, subset_seed = condition
        condition_started = time.time()
        print(
            "fitting arm={} sampling={} subset_seed={}".format(
                ARM_LETTERS[arm], sampling, subset_seed
            ),
            flush=True,
        )
        condition_mask = (
            membership["arm"].eq(arm)
            & membership["sampling"].eq(sampling)
            & membership["subset_seed"].eq(subset_seed)
        )
        condition_positions = np.flatnonzero(condition_mask.to_numpy())
        frame = membership.iloc[condition_positions].copy()
        resolver_order = reuse_by_identity.loc[
            frame["sequence_pair_sha256"]
        ].reset_index(drop=True)
        frame["_source_order"] = resolver_order["feature_source"].map(
            {"old": 0, "delta": 1}
        ).to_numpy(dtype=int)
        frame["_feature_row_order"] = resolver_order[
            "feature_row_index"
        ].to_numpy(dtype=int)
        frame["_global_membership_position"] = condition_positions
        frame = frame.sort_values(
            ["_source_order", "_feature_row_order"], kind="mergesort"
        ).reset_index(drop=True)
        features = resolved_features[
            frame["_resolver_position"].to_numpy(dtype=int)
        ]
        result = fit_condition(
            frame,
            features,
            retention_features,
            c_grid,
            args.folds,
            args.split_seed,
        )
        metadata = {
            "arm": arm,
            "arm_letter": ARM_LETTERS[arm],
            "arm_role": ARM_ROLES[arm],
            "sampling": sampling,
            "subset_seed": int(subset_seed),
        }
        blocks = {}
        for key in ("tuning", "validation_predictions", "fold_membership"):
            block = result[key].copy()
            for column, value in metadata.items():
                block[column] = value
            blocks[key] = block
        condition_row = {
            **metadata,
            "weak_rows": int(len(frame)),
            "weak_positive": int(frame["weak_label"].sum()),
            "weak_negative": int((frame["weak_label"] == 0).sum()),
            "weak_unique_peptide": int(frame["chain1_sha256"].nunique()),
            "weak_unique_affibody": int(frame["chain2_sha256"].nunique()),
            "weak_labeled_membership_sha256": _labeled_membership_sha256(frame),
            "selected_C": float(result["selected_c"]),
            "model_nonzero_coefficients": int(result["model_nonzero_coefficients"]),
        }
        frozen = retention_identity[
            [
                "pair_uid",
                "peptide_design_code",
                "affibody_design_code",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
            ]
        ].copy()
        for column, value in metadata.items():
            frozen[column] = value
        frozen["selected_C"] = float(result["selected_c"])
        frozen["binder_probability"] = result["probability"]
        print(
            "finished arm={} sampling={} subset_seed={} selected_C={} elapsed_seconds={:.1f}".format(
                ARM_LETTERS[arm],
                sampling,
                subset_seed,
                result["selected_c"],
                time.time() - condition_started,
            ),
            flush=True,
        )

        return {
            "tuning": blocks["tuning"],
            "validation_predictions": blocks["validation_predictions"],
            "fold_membership": blocks["fold_membership"],
            "condition": condition_row,
            "frozen": frozen,
        }

    condition_grid = _expected_conditions()
    # Each condition is already a coarse parallel task.  Limit BLAS/OpenMP to
    # one thread per worker so --jobs does not multiply hidden thread pools.
    with threadpool_limits(limits=1):
        if int(args.jobs) == 1:
            fitted = [fit_one_condition(condition) for condition in condition_grid]
        else:
            with ThreadPoolExecutor(max_workers=int(args.jobs)) as executor:
                fitted = list(executor.map(fit_one_condition, condition_grid))
    tuning_blocks = [item["tuning"] for item in fitted]
    weak_prediction_blocks = [item["validation_predictions"] for item in fitted]
    fold_blocks = [item["fold_membership"] for item in fitted]
    condition_rows = [item["condition"] for item in fitted]
    frozen_blocks = [item["frozen"] for item in fitted]

    tuning = pd.concat(tuning_blocks, ignore_index=True)
    weak_predictions = pd.concat(weak_prediction_blocks, ignore_index=True)
    fold_membership = pd.concat(fold_blocks, ignore_index=True)
    conditions = pd.DataFrame(condition_rows)
    frozen_predictions = pd.concat(frozen_blocks, ignore_index=True)
    freeze_hash = _prediction_sha256(frozen_predictions)
    arm_a_selected_c = float(
        conditions.loc[
            conditions["arm"].eq(ARMS[0])
            & conditions["sampling"].eq("natural")
            & conditions["subset_seed"].eq(NATURAL_SEED),
            "selected_C",
        ].iloc[0]
    )
    arm_a_reference_parity = validate_arm_a_frozen_predictions(
        frozen_predictions,
        arm_a_selected_c,
        reference,
        args.max_reference_probability_difference,
    )

    output_frames = {
        "conditions.csv": conditions,
        "weak_validation.csv": tuning,
        "weak_validation_predictions.csv": weak_predictions,
        "fold_membership.csv": fold_membership,
        "retention_predictions_frozen.csv": frozen_predictions,
    }
    output_paths = {}
    for name, frame in output_frames.items():
        path = output_dir / name
        _write_csv(frame, path)
        output_paths[name] = path

    # This is the first point at which any outcome-derived reference metric is
    # read.  The model predictions are already on disk and hash-frozen.
    reference = load_reference_metrics_after_freeze(reference, args.old_cache)
    # This is the first point at which row-level numerical retention outcomes
    # are read.
    retention_target = load_retention_targets_after_freeze(
        args.old_cache_rows, retention_identity
    )
    scored = frozen_predictions.merge(
        retention_target[["pair_uid", "target_retention", "target_binder"]],
        on="pair_uid",
        validate="many_to_one",
    )
    _require(_prediction_sha256(scored) == freeze_hash, "prediction vector changed after target load")
    metric_rows = []
    peptide_blocks = []
    for key, frame in scored.groupby(["arm", "sampling", "subset_seed"], sort=True):
        arm, sampling, subset_seed = key
        frame = frame.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
        metrics, peptide = retention_metrics(frame)
        metadata = {
            "arm": arm,
            "arm_letter": ARM_LETTERS[arm],
            "arm_role": ARM_ROLES[arm],
            "sampling": sampling,
            "subset_seed": int(subset_seed),
        }
        metric_rows.append({**metadata, **metrics})
        for column, value in metadata.items():
            peptide[column] = value
        peptide_blocks.append(peptide)
    metrics = pd.DataFrame(metric_rows)
    arm_a_metric_differences = validate_arm_a_metrics(
        metrics,
        reference,
        args.max_reference_primary_metric_difference,
        args.max_reference_global_metric_difference,
    )
    per_peptide = pd.concat(peptide_blocks, ignore_index=True)
    deltas, peptide_deltas = paired_deltas(metrics, per_peptide)
    bootstrap = peptide_cluster_bootstrap(
        peptide_deltas, args.bootstrap_draws, args.bootstrap_seed
    )
    scored_outputs = {
        "retention_predictions.csv": scored,
        "retention_metrics.csv": metrics,
        "per_peptide_metrics.csv": per_peptide,
        "paired_deltas_vs_arm_a.csv": deltas,
        "per_peptide_deltas_vs_arm_a.csv": peptide_deltas,
        "peptide_cluster_bootstrap.csv": bootstrap,
    }
    for name, frame in scored_outputs.items():
        path = output_dir / name
        _write_csv(frame, path)
        output_paths[name] = path
    output_paths["overlap_sentinel_audit.csv"] = overlap_path

    summary = [
        "# Frozen-MINT LibA later-round label ablation",
        "",
        "This retrospective comparison changes the selection-round label definition while keeping the frozen MINT representation, logistic head, three identity-cold weak-validation folds, class-weighted loss, C grid, and 108-pair retention panel fixed.",
        "",
        "Arm B (R9–R14 raw pooled counts) is the primary later-round comparison with arm A (R9–R10). Arm C is the equal-round-weight sensitivity analysis. Arm D is a late-only diagnostic. Natural-size results are primary; three deterministic size-matched positive subsets isolate label breadth from sample count.",
        "",
        "Retention outcomes were loaded only after retention_predictions_frozen.csv was written. Bootstrap intervals are exploratory paired resampling of the nine peptide identities, not confirmatory confidence intervals.",
        "",
        "| Arm | Role | Sampling | Seed | Weak +/− | C | Within-peptide Spearman | Global Spearman | AUROC | AP | Top-choice binder |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    combined = metrics.merge(
        conditions[
            ["arm", "sampling", "subset_seed", "weak_positive", "weak_negative", "selected_C"]
        ],
        on=["arm", "sampling", "subset_seed"],
        validate="one_to_one",
    )
    for _, row in combined.sort_values(["sampling", "subset_seed", "arm_letter"]).iterrows():
        summary.append(
            "| {arm_letter} | {arm_role} | {sampling} | {subset_seed} | {weak_positive}/{weak_negative} | {selected_C:.3g} | {within_peptide_macro_spearman:.3f} | {global_spearman:.3f} | {global_auroc:.3f} | {global_average_precision:.3f} | {within_peptide_top_choice_binder:.3f} |".format(**row.to_dict())
        )
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary) + "\n")
    os.chmod(str(summary_path), 0o600)
    output_paths["run_summary.md"] = summary_path

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory_label_ablation",
        "predeclaration": {
            "reference": "A: R9-R10 raw pooled top 2%",
            "primary": "B: R9-R14 raw pooled top 2%",
            "sensitivity": "C: R9-R14 equal round-frequency top 2%",
            "diagnostic": "D: R11-R14 raw pooled top 2%",
            "primary_endpoint": "macro average of Spearman correlations within exactly nine peptide identities",
            "secondary_endpoints": [
                "global Spearman",
                "AUROC at retention >= 75%",
                "average precision at retention >= 75%",
                "tie-aware probability that a top-scored Affibody is a binder",
            ],
        },
        "configuration": {
            "library": LIBRARY,
            "arms": list(ARMS),
            "arm_roles": ARM_ROLES,
            "samplings": list(SAMPLINGS),
            "size_match_seeds": list(SIZE_MATCH_SEEDS),
            "size_match_rank": "SHA256('size-match|<seed>|<pair_uid>'), lexical ascending",
            "folds": int(args.folds),
            "split_seed": int(args.split_seed),
            "fold_rule": "three diagonal double-identity-cold folds; validation has both partner hashes in fold, train has neither, remaining rows guard",
            "c_grid": list(c_grid),
            "c_selection": "minimum pooled weak-validation log loss; AP, AUROC, then smaller C tie-break",
            "head": "L2 logistic regression, liblinear, class_weight=balanced, tol=1e-4; MINT frozen",
            "parallel_condition_jobs": int(args.jobs),
            "blas_openmp_threads_per_condition_worker": 1,
            "retention_threshold": "binder iff measured retention >= 75%",
            "retention_outcome_firewall": "all probabilities frozen to retention_predictions_frozen.csv before numerical targets loaded",
            "identity_exclusion": "recomputed library-local LibA chain1 and chain2 SHA256 exclusion; old global cross-library strict flag deliberately unused",
            "bootstrap": "exploratory paired resampling of nine peptide identities",
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
            "top_choice_ties": "exactly equal maximum probabilities share one top choice; report mean binder label among tied choices",
        },
        "feature_contract": contract,
        "feature_validation": {
            "overlap_sentinels": int(len(overlap_audit)),
            "max_abs_difference": float(overlap_maximum),
            "allowed_max_abs_difference": float(args.max_overlap_feature_difference),
            "cached_vs_legacy_retention_max_abs_difference": float(
                retention_feature_difference
            ),
            "allowed_retention_max_abs_difference": float(
                args.max_retention_feature_difference
            ),
            "arm_a_reference_probability_max_abs_difference": float(
                arm_a_reference_parity["probability_max_abs_difference"]
            ),
            "arm_a_reference_within_peptide_rankings_exact": bool(
                arm_a_reference_parity["within_peptide_rankings_exact"]
            ),
            "arm_a_reference_global_probability_rank_spearman": float(
                arm_a_reference_parity["global_probability_rank_spearman"]
            ),
            "allowed_arm_a_reference_probability_max_abs_difference": float(
                args.max_reference_probability_difference
            ),
            "arm_a_reference_metric_abs_differences": arm_a_metric_differences,
            "allowed_arm_a_reference_primary_metric_abs_difference": float(
                args.max_reference_primary_metric_difference
            ),
            "allowed_arm_a_reference_global_metric_abs_difference": float(
                args.max_reference_global_metric_difference
            ),
        },
        "sources": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "old_cache": args.old_cache,
                "old_cache_manifest": args.old_cache_manifest,
                "old_cache_rows": args.old_cache_rows,
                "old_cache_rows_manifest": args.old_cache_rows_manifest,
                "delta_cache": args.delta_cache,
                "delta_cache_manifest": args.delta_cache_manifest,
                "delta_prepare_manifest": args.delta_prepare_manifest,
                "membership": args.membership,
                "reuse_index": args.reuse_index,
                "label_manifest": args.label_manifest,
                "arm_labels": args.arm_labels,
                "size_matched_membership": args.size_matched_membership,
                "retention_features": args.retention_features,
                "retention_features_manifest": args.retention_features_manifest,
                "arm_a_reference_manifest": reference["manifest_path"],
                "arm_a_reference_predictions": reference["predictions_path"],
            }.items()
        },
        "rows": {
            "old_cache": int(len(old_frame)),
            "delta_cache": int(len(delta_frame)),
            "normalized_membership": int(len(membership)),
            "canonical_arm_labels": int(len(canonical_labels)),
            "canonical_size_matched_positive_membership": int(
                len(canonical_size_membership)
            ),
            "membership_rows_removed_by_recomputed_liba_local_identity_cold": int(local_identity_rows_removed),
            "conditions": int(len(conditions)),
            "retention_predictions": int(len(scored)),
            "retention_pairs_per_condition": EXPECTED_RETENTION_ROWS,
            "peptide_groups_per_condition": EXPECTED_PEPTIDE_GROUPS,
        },
        "prediction_freeze_sha256": freeze_hash,
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(output_paths.items())
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
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
                "overlap_max_abs_difference": overlap_maximum,
                "prediction_freeze_sha256": freeze_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-cache",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz",
    )
    parser.add_argument(
        "--old-cache-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/merged/manifest.json",
    )
    parser.add_argument(
        "--old-cache-rows",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv",
    )
    parser.add_argument(
        "--old-cache-rows-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/rows/manifest.json",
    )
    parser.add_argument("--delta-cache", type=Path, required=True)
    parser.add_argument("--delta-cache-manifest", type=Path, required=True)
    parser.add_argument("--delta-prepare-manifest", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--reuse-index", type=Path, required=True)
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/liba_later_round_labels_v1/manifest.json",
    )
    parser.add_argument(
        "--arm-labels",
        type=Path,
        default=REPO_ROOT / "private_data/derived/liba_later_round_labels_v1/arm_labels.csv",
    )
    parser.add_argument(
        "--size-matched-membership",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/liba_later_round_labels_v1/size_matched_positive_membership.csv",
    )
    parser.add_argument(
        "--retention-features",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_features_v1.npz",
    )
    parser.add_argument(
        "--retention-features-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_features_v1.manifest.json",
    )
    parser.add_argument(
        "--reference-run-dir",
        type=Path,
        default=REPO_ROOT
        / "private_data/experiments/mint_selection_matched_confirmatory_liba_seed20260811_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=CANONICAL_FOLDS)
    parser.add_argument("--split-seed", type=int, default=CANONICAL_SPLIT_SEED)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(CANONICAL_C_GRID))
    parser.add_argument("--size-match-seeds", type=int, nargs="+", default=list(SIZE_MATCH_SEEDS))
    parser.add_argument("--max-overlap-feature-difference", type=float, default=1e-4)
    parser.add_argument("--max-retention-feature-difference", type=float, default=1e-4)
    parser.add_argument(
        "--max-reference-probability-difference",
        type=float,
        default=1e-3,
        help="tight rerun parity tolerance for the historical arm-A probability vector",
    )
    parser.add_argument(
        "--max-reference-primary-metric-difference", type=float, default=1e-12
    )
    parser.add_argument(
        "--max-reference-global-metric-difference", type=float, default=5e-4
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel independent condition fits (1-4; use 2-4 only on a high-memory CPU node)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
