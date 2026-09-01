#!/usr/bin/env python
"""Evaluate LibA later-round weak-label definitions with one site model.

This is the CPU, position-additive companion to the matched MINT/LoRA
experiment.  It compares four prespecified positive-label definitions while
holding the provider-negative set, the retention panel, the model family, and
the weak-validation procedure fixed.  Natural-size conditions retain every
positive in an arm.  Size-matched conditions retain exactly 11,320 positives
per arm for each of the three prespecified membership seeds.

The retention CSV is first read using identity/design columns only.  Every
weak-validation C is selected and every 108-row prediction vector is then
computed and made read-only before the numerical retention targets are loaded.

Canonical command (the output directory must not already exist)::

    venv/bin/python downstream/AffibodyMHC/evaluate_liba_later_round_site.py \
      --label-dir private_data/derived/liba_later_round_labels_v1 \
      --retention-csv private_data/derived/retention_matrix.csv \
      --output-dir private_data/experiments/liba_later_round_site_v1
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    EXPECTED_COLUMNS,
    LIBRARY_SPECS,
    load_retention_table,
    opaque_id,
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "liba-later-round-site-evaluation-v1"
LIBRARY = "LibA"
ARM_ORDER = ("A", "B", "C", "D")
ARM_IDS = {arm: arm for arm in ARM_ORDER}
ARM_NAMES = {
    "A": "R009-R010 raw-count sum",
    "B": "R009-R014 raw-count sum",
    "C": "R009-R014 equal-round mean frequency",
    "D": "R011-R014 raw-count sum",
}
ARM_ROLES = {
    "A": "reference",
    "B": "primary_later_round_test",
    "C": "depth_normalized_sensitivity",
    "D": "late_only_diagnostic",
}
EXPECTED_POSITIVES = {
    "A": 11320,
    "B": 24925,
    "C": 23189,
    "D": 19596,
}
EXPECTED_NEGATIVES = 11222
MATCHED_POSITIVES = 11320
MATCHED_SEEDS = (20260811, 20260812, 20260813)
C_GRID = (0.001, 0.01, 0.1, 1.0)
FOLDS = 3
SPLIT_SEED = 17
BALANCE = "all_class_weighted"
REGIME = "double_cold"
CLEANING = "c0"
RETENTION_THRESHOLD = 75.0
RETENTION_ROWS = 108
RETENTION_POSITIVES = 38
RETENTION_PEPTIDES = 9
RETENTION_AFFIBODIES_PER_PEPTIDE = 12
DEFAULT_BOOTSTRAP_DRAWS = 10000
DEFAULT_BOOTSTRAP_SEED = 20260820

LABEL_FILES = (
    "arm_labels.csv",
    "size_matched_positive_membership.csv",
    "arm_summary.csv",
)
LABEL_REQUIRED_COLUMNS = {
    "arm",
    "pep",
    "aff",
    "r001_count",
    "weak_label",
    "label_score",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
}
MEMBERSHIP_REQUIRED_COLUMNS = {
    "arm",
    "sampling_seed",
    "target_positive_rows",
    "pair_uid",
    "rank_sha256",
    "size_match_rank",
}
SUMMARY_REQUIRED_COLUMNS = {
    "arm",
    "arm_name",
    "role",
    "eligible_positive",
    "eligible_negative",
    "natural_training_rows",
    "positive_membership_sha256",
    "negative_membership_sha256",
}
METRIC_NAMES = (
    "within_peptide_macro_spearman",
    "global_spearman",
    "global_auroc",
    "global_average_precision",
    "within_peptide_top_choice_success",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _is_sha256(value):
    value = str(value)
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _manifest_output_sha256(manifest, filename):
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "label manifest lacks outputs")
    _require(filename in outputs, "label manifest lacks {}".format(filename))
    entry = outputs[filename]
    digest = entry.get("sha256") if isinstance(entry, dict) else entry
    _require(_is_sha256(digest), "invalid manifest hash for {}".format(filename))
    return str(digest)


def _numeric_integer(series, name):
    values = pd.to_numeric(series, errors="coerce")
    _require(bool(values.notna().all()), "nonnumeric {}".format(name))
    _require(bool(np.equal(values, np.floor(values)).all()), "noninteger {}".format(name))
    return values.astype(np.int64)


def _read_private_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _write_private_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _write_private_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _prediction_sha256(pair_uid, score):
    pairs = np.asarray(pair_uid, dtype=str)
    values = np.asarray(score, dtype=np.float64)
    _require(len(pairs) == len(values), "prediction hash length mismatch")
    _require(bool(np.isfinite(values).all()), "prediction vector is non-finite")
    lines = [
        "{}|{}".format(pair, float(value).hex())
        for pair, value in zip(pairs.tolist(), values.tolist())
    ]
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _derived_seed(seed, *parts):
    payload = "|".join([str(int(seed))] + [str(value) for value in parts])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _size_match_rank(pair_uid, seed):
    payload = "size-match|{}|{}".format(int(seed), str(pair_uid)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_label_bundle(label_dir):
    """Load and hash-check the canonical four-arm label bundle."""
    label_dir = Path(label_dir)
    _require(label_dir.is_dir(), "label directory does not exist")
    paths = {name: label_dir / name for name in LABEL_FILES}
    paths["manifest.json"] = label_dir / "manifest.json"
    for name, path in paths.items():
        _require(path.is_file(), "missing label input {}".format(name))
    with open(str(paths["manifest.json"]), "r") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("schema_version") == "liba-later-round-labels-v1",
        "label-bundle schema version changed",
    )
    for filename in LABEL_FILES:
        _require(
            sha256_file(paths[filename]) == _manifest_output_sha256(manifest, filename),
            "label-bundle hash mismatch for {}".format(filename),
        )

    labels = _read_private_csv(paths["arm_labels.csv"])
    membership = _read_private_csv(paths["size_matched_positive_membership.csv"])
    summary = _read_private_csv(paths["arm_summary.csv"])
    missing = LABEL_REQUIRED_COLUMNS.difference(labels.columns)
    _require(not missing, "arm labels missing columns {}".format(sorted(missing)))
    missing = MEMBERSHIP_REQUIRED_COLUMNS.difference(membership.columns)
    _require(not missing, "matched membership missing columns {}".format(sorted(missing)))
    missing = SUMMARY_REQUIRED_COLUMNS.difference(summary.columns)
    _require(not missing, "arm summary missing columns {}".format(sorted(missing)))
    _require(set(labels["arm"].astype(str)) == set(ARM_ORDER), "arm-label names changed")
    _require(set(summary["arm"].astype(str)) == set(ARM_ORDER), "arm-summary names changed")
    _require(len(summary) == len(ARM_ORDER) and not bool(summary["arm"].duplicated().any()), "arm summary is not one row per arm")
    _require(set(membership["arm"].astype(str)).issubset(set(ARM_ORDER)), "unknown matched arm")

    labels = labels.copy()
    labels["weak_label"] = _numeric_integer(labels["weak_label"], "weak_label")
    labels["r001_count"] = _numeric_integer(labels["r001_count"], "r001_count")
    labels["label_score"] = pd.to_numeric(labels["label_score"], errors="coerce")
    _require(bool(np.isfinite(labels["label_score"].to_numpy(float)).all()), "non-finite label score")
    _require(bool(labels["label_score"].ge(0).all()), "negative label score")
    _require(set(labels["weak_label"].astype(int)) == {0, 1}, "labels are not binary")
    _require(not bool(labels.duplicated(["arm", "pair_uid"]).any()), "duplicate arm/pair label")
    if "library" in labels.columns:
        _require(bool(labels["library"].eq(LIBRARY).all()), "non-LibA label row")
    for flag in (
        "within_declared_library_alphabet",
        "strict_liba_retention_identity_cold_eligible",
    ):
        _require(flag in labels.columns, "arm labels lack {}".format(flag))
        converted = _numeric_integer(labels[flag], flag)
        _require(bool(converted.eq(1).all()), "ineligible row in {}".format(flag))

    spec = LIBRARY_SPECS[LIBRARY]
    _require(bool(labels["pep"].map(len).eq(spec["pep_length"]).all()), "bad peptide code length")
    _require(bool(labels["aff"].map(len).eq(spec["aff_length"]).all()), "bad Affibody code length")
    canonical = set(AA_ALPHABET)
    _require(bool(labels["pep"].map(lambda value: set(value).issubset(canonical)).all()), "bad peptide residue")
    _require(bool(labels["aff"].map(lambda value: set(value).issubset(canonical)).all()), "bad Affibody residue")
    expected_pair = [opaque_id(LIBRARY, pep, aff) for pep, aff in zip(labels["pep"], labels["aff"])]
    expected_peptide = [opaque_id(LIBRARY, "pep", pep) for pep in labels["pep"]]
    expected_affibody = [opaque_id(LIBRARY, "aff", aff) for aff in labels["aff"]]
    _require(bool(labels["pair_uid"].eq(expected_pair).all()), "pair UID/code mismatch")
    _require(bool(labels["peptide_uid"].eq(expected_peptide).all()), "peptide UID/code mismatch")
    _require(bool(labels["affibody_uid"].eq(expected_affibody).all()), "Affibody UID/code mismatch")
    negative = labels["weak_label"].eq(0)
    _require(bool(labels.loc[negative, "r001_count"].ge(3).all()), "negative below R001 count 3")

    membership = membership.copy()
    membership["sampling_seed"] = _numeric_integer(membership["sampling_seed"], "sampling_seed")
    membership["target_positive_rows"] = _numeric_integer(
        membership["target_positive_rows"], "target_positive_rows"
    )
    membership["size_match_rank"] = _numeric_integer(
        membership["size_match_rank"], "size_match_rank"
    )
    _require(bool(membership["target_positive_rows"].eq(MATCHED_POSITIVES).all()), "matched target size changed")
    _require(
        set(membership["sampling_seed"].astype(int)) == set(MATCHED_SEEDS),
        "matched sampling seeds changed",
    )
    _require(
        not bool(membership.duplicated(["arm", "sampling_seed", "pair_uid"]).any()),
        "duplicate matched membership row",
    )
    positive_keys = set(
        labels.loc[labels["weak_label"].eq(1), ["arm", "pair_uid"]].itertuples(index=False, name=None)
    )
    observed_membership = set(
        membership[["arm", "pair_uid"]].itertuples(index=False, name=None)
    )
    _require(observed_membership.issubset(positive_keys), "matched membership contains a nonpositive pair")

    contracts = manifest.get("contracts", {})
    _require(contracts.get("eligible_positive_by_arm") == EXPECTED_POSITIVES, "manifest positive contract changed")
    _require(int(contracts.get("eligible_negative", -1)) == EXPECTED_NEGATIVES, "manifest negative contract changed")
    configuration = manifest.get("configuration", {})
    _require(configuration.get("library") == LIBRARY, "manifest library changed")
    _require(float(configuration.get("top_fraction", -1.0)) == 0.02, "manifest top fraction changed")
    _require(int(configuration.get("size_match_target_positive", -1)) == MATCHED_POSITIVES, "manifest size-match target changed")
    _require(tuple(configuration.get("size_match_arms", ())) == ARM_ORDER[1:], "manifest size-match arms changed")
    _require(tuple(configuration.get("size_match_seeds", ())) == MATCHED_SEEDS, "manifest size-match seeds changed")
    downstream = configuration.get("downstream_validation_contract", {})
    _require(int(downstream.get("folds", -1)) == FOLDS, "manifest fold contract changed")
    _require(int(downstream.get("split_seed", -1)) == SPLIT_SEED, "manifest split seed changed")
    _require(tuple(float(value) for value in downstream.get("c_grid", ())) == C_GRID, "manifest C grid changed")
    return labels, membership, summary, manifest, paths


def validate_arm_contract(labels, expected_positives=None, expected_negatives=EXPECTED_NEGATIVES):
    """Validate natural arm sizes and the identical negative set."""
    expected_positives = EXPECTED_POSITIVES if expected_positives is None else expected_positives
    negative_sets = {}
    rows = []
    for arm in ARM_ORDER:
        block = labels.loc[labels["arm"].eq(arm)]
        positive = block.loc[block["weak_label"].eq(1)]
        negative = block.loc[block["weak_label"].eq(0)]
        _require(len(positive) == int(expected_positives[arm]), "{} positive count changed".format(arm))
        _require(len(negative) == int(expected_negatives), "{} negative count changed".format(arm))
        _require(set(positive["pair_uid"]).isdisjoint(set(negative["pair_uid"])), "positive/negative overlap")
        negative_sets[arm] = set(negative["pair_uid"].astype(str))
        rows.append(
            {
                "arm": arm,
                "arm_id": ARM_IDS[arm],
                "natural_rows": int(len(block)),
                "natural_positive": int(len(positive)),
                "natural_negative": int(len(negative)),
                "natural_membership_sha256": cached_eval.membership_sha256(
                    block.assign(pair_identity=block["pair_uid"])
                ),
            }
        )
    reference = negative_sets[ARM_ORDER[0]]
    _require(all(values == reference for values in negative_sets.values()), "negative set differs across arms")
    return pd.DataFrame(rows)


def validate_summary_contract(summary, arm_contract, manifest, labels):
    """Bind summary counts and memberships back to the row-level labels."""
    merged = summary.merge(
        arm_contract,
        on="arm",
        how="left",
        validate="one_to_one",
    )
    for column in ("eligible_positive", "eligible_negative", "natural_training_rows"):
        merged[column] = _numeric_integer(merged[column], column)
    _require(bool(merged["eligible_positive"].eq(merged["natural_positive"]).all()), "summary positive count mismatch")
    _require(bool(merged["eligible_negative"].eq(merged["natural_negative"]).all()), "summary negative count mismatch")
    _require(bool(merged["natural_training_rows"].eq(merged["natural_rows"]).all()), "summary row count mismatch")
    for arm in ARM_ORDER:
        row = merged.loc[merged["arm"].eq(arm)].iloc[0]
        _require(str(row["role"]) == ("baseline" if arm == "A" else ARM_ROLES[arm]), "summary arm role changed")
        positive = labels.loc[labels["arm"].eq(arm) & labels["weak_label"].eq(1)].copy()
        positive["pair_identity"] = positive["pair_uid"].astype(str)
        _require(
            str(row["positive_membership_sha256"]) == cached_eval.membership_sha256(positive),
            "summary positive membership mismatch for {}".format(arm),
        )
    negative_hashes = set(merged["negative_membership_sha256"].astype(str))
    _require(len(negative_hashes) == 1, "summary negative hashes differ")
    recorded = manifest["contracts"].get("eligible_negative_membership_sha256")
    _require(negative_hashes == {str(recorded)}, "summary/manifest negative hash mismatch")


def validate_matched_contract(membership, labels, matched_positive=MATCHED_POSITIVES):
    """Recompute the builder's exact deterministic later-arm samples."""
    arms_in_file = set(membership["arm"].astype(str))
    later_arms = set(ARM_ORDER[1:])
    _require(arms_in_file == later_arms, "matched membership must contain B/C/D only")
    positive_by_arm = {
        arm: set(
            labels.loc[
                labels["arm"].eq(arm) & labels["weak_label"].eq(1), "pair_uid"
            ].astype(str)
        )
        for arm in ARM_ORDER
    }
    for arm in sorted(arms_in_file):
        for seed in MATCHED_SEEDS:
            selected = membership.loc[
                membership["arm"].eq(arm)
                & membership["sampling_seed"].eq(int(seed)),
            ].copy()
            _require(len(selected) == int(matched_positive), "{} seed {} matched size changed".format(arm, seed))
            _require(set(selected["pair_uid"].astype(str)).issubset(positive_by_arm[arm]), "matched pair is outside its arm")
            expected_hash = selected["pair_uid"].map(lambda uid: _size_match_rank(uid, seed))
            _require(bool(selected["rank_sha256"].eq(expected_hash).all()), "size-match rank hash changed")
            selected = selected.sort_values(["size_match_rank", "pair_uid"], kind="mergesort")
            _require(
                np.array_equal(
                    selected["size_match_rank"].to_numpy(dtype=np.int64),
                    np.arange(1, int(matched_positive) + 1, dtype=np.int64),
                ),
                "size-match integer ranks changed",
            )
            all_positive = labels.loc[
                labels["arm"].eq(arm) & labels["weak_label"].eq(1), ["pair_uid"]
            ].copy()
            all_positive["rank_sha256"] = all_positive["pair_uid"].map(
                lambda uid: _size_match_rank(uid, seed)
            )
            expected = set(
                all_positive.sort_values(["rank_sha256", "pair_uid"], kind="mergesort")
                .head(int(matched_positive))["pair_uid"]
                .astype(str)
            )
            _require(set(selected["pair_uid"].astype(str)) == expected, "size-match selected set changed")


def load_retention_design_panel(path):
    """Load LibA identities/codes without reading either numerical target column."""
    path = Path(path)
    _require(path.is_file(), "retention CSV does not exist")
    header = pd.read_csv(path, nrows=0)
    _require(list(header.columns) == EXPECTED_COLUMNS, "unexpected retention table schema")
    usecols = [
        "provisional_matrix_key",
        "library",
        "affibody_design_code",
        "peptide_design_code",
    ]
    panel = pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(len(panel) == 228, "retention design must contain 228 cells")
    panel = panel.loc[panel["library"].eq(LIBRARY)].copy()
    spec = LIBRARY_SPECS[LIBRARY]
    _require(len(panel) == RETENTION_ROWS, "LibA retention design row count changed")
    _require(panel["peptide_design_code"].nunique() == RETENTION_PEPTIDES, "LibA peptide count changed")
    _require(panel["affibody_design_code"].nunique() == spec["affibody_codes"], "LibA Affibody count changed")
    _require(
        not bool(panel.duplicated(["peptide_design_code", "affibody_design_code"]).any()),
        "duplicate LibA retention pair",
    )
    _require(bool(panel["peptide_design_code"].map(len).eq(spec["pep_length"]).all()), "bad retention peptide code")
    _require(bool(panel["affibody_design_code"].map(len).eq(spec["aff_length"]).all()), "bad retention Affibody code")
    panel["pair_uid"] = [
        opaque_id(LIBRARY, peptide, affibody)
        for peptide, affibody in zip(panel["peptide_design_code"], panel["affibody_design_code"])
    ]
    panel["peptide_uid"] = [
        opaque_id(LIBRARY, "pep", peptide) for peptide in panel["peptide_design_code"]
    ]
    panel["affibody_uid"] = [
        opaque_id(LIBRARY, "aff", affibody) for affibody in panel["affibody_design_code"]
    ]
    panel = panel.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(not bool(panel["pair_uid"].duplicated().any()), "duplicate retention pair UID")
    return panel


def construct_conditions(labels, membership):
    """Return all four natural and all twelve size-matched train frames."""
    conditions = []
    for arm in ARM_ORDER:
        natural = labels.loc[labels["arm"].eq(arm)].copy()
        natural = natural.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
        conditions.append((arm, "natural", -1, natural))

    for seed in MATCHED_SEEDS:
        for arm in ARM_ORDER:
            block = labels.loc[labels["arm"].eq(arm)].copy()
            negative = block.loc[block["weak_label"].eq(0)]
            positive = block.loc[block["weak_label"].eq(1)]
            selected_rows = membership.loc[
                membership["arm"].eq(arm)
                & membership["sampling_seed"].eq(int(seed)),
                "pair_uid",
            ]
            if arm == ARM_ORDER[0] and selected_rows.empty:
                selected = positive
            else:
                selected_set = set(selected_rows.astype(str))
                selected = positive.loc[positive["pair_uid"].astype(str).isin(selected_set)]
                _require(len(selected) == len(selected_set), "matched membership join lost a pair")
            _require(len(selected) == MATCHED_POSITIVES, "matched positive count changed")
            matched = pd.concat([selected, negative], ignore_index=True)
            matched = matched.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
            _require(int(matched["weak_label"].sum()) == MATCHED_POSITIVES, "matched label sum changed")
            _require(len(matched) == MATCHED_POSITIVES + EXPECTED_NEGATIVES, "matched row count changed")
            conditions.append((arm, "size_matched", int(seed), matched))
    return conditions


def assert_local_retention_identity_cold(training, retention_design):
    """Require both LibA partner identities to be unseen in the retention grid."""
    _require(
        set(training["pair_uid"].astype(str)).isdisjoint(set(retention_design["pair_uid"].astype(str))),
        "exact retention pair leaked into training",
    )
    _require(
        set(training["peptide_uid"].astype(str)).isdisjoint(set(retention_design["peptide_uid"].astype(str))),
        "retention peptide identity leaked into training",
    )
    _require(
        set(training["affibody_uid"].astype(str)).isdisjoint(set(retention_design["affibody_uid"].astype(str))),
        "retention Affibody identity leaked into training",
    )


def site_code_matrix(frame, peptide_column="pep", affibody_column="aff"):
    spec = LIBRARY_SPECS[LIBRARY]
    peptide = frame[peptide_column].astype(str)
    affibody = frame[affibody_column].astype(str)
    _require(bool(peptide.map(len).eq(spec["pep_length"]).all()), "bad site peptide code")
    _require(bool(affibody.map(len).eq(spec["aff_length"]).all()), "bad site Affibody code")
    columns = [peptide.str[index].to_numpy(str) for index in range(spec["pep_length"])]
    columns += [affibody.str[index].to_numpy(str) for index in range(spec["aff_length"])]
    return np.column_stack(columns)


def make_site_encoder(n_positions):
    arguments = {
        "categories": [list(AA_ALPHABET) for _ in range(int(n_positions))],
        "handle_unknown": "error",
        "dtype": np.float64,
    }
    try:
        return OneHotEncoder(sparse_output=True, **arguments)
    except TypeError:
        return OneHotEncoder(sparse=True, **arguments)


def _condition_metadata(arm, sampling_mode, sampling_seed):
    return {
        "library": LIBRARY,
        "arm_id": ARM_IDS[arm],
        "arm": arm,
        "arm_role": ARM_ROLES[arm],
        "sampling_mode": sampling_mode,
        "sampling_seed": int(sampling_seed),
    }


def build_and_audit_fold_plans(training, metadata):
    """Build the exact three matched-LoRA folds and summarize memberships."""
    primary = training.copy().reset_index(drop=True)
    primary["peptide_identity"] = primary["peptide_uid"].astype(str)
    primary["affibody_identity"] = primary["affibody_uid"].astype(str)
    primary["pair_identity"] = primary["pair_uid"].astype(str)
    primary["_condition_index"] = np.arange(len(primary), dtype=int)
    plans = cached_eval.build_fold_plans(
        primary,
        REGIME,
        CLEANING,
        BALANCE,
        seed=-1,
        folds=FOLDS,
        split_seed=SPLIT_SEED,
    )
    _require(len(plans) == FOLDS, "condition does not have all three evaluable folds")
    split_rows = []
    all_indices = set(range(len(primary)))
    for plan in plans:
        train_keys = np.asarray(plan["train_keys"], dtype=int)
        validation_keys = np.asarray(plan["validation_keys"], dtype=int)
        guard_keys = np.asarray(sorted(all_indices.difference(train_keys).difference(validation_keys)), dtype=int)
        _require(len(guard_keys) == int(plan["n_guard"]), "fold guard count changed")
        train = primary.iloc[train_keys]
        validation = primary.iloc[validation_keys]
        _require(set(train["weak_label"].astype(int)) == {0, 1}, "fold train lacks a class")
        _require(set(validation["weak_label"].astype(int)) == {0, 1}, "fold validation lacks a class")
        _require(set(train["peptide_uid"]).isdisjoint(set(validation["peptide_uid"])), "fold peptide leakage")
        _require(set(train["affibody_uid"]).isdisjoint(set(validation["affibody_uid"])), "fold Affibody leakage")
        _require(
            cached_eval.membership_sha256(train) == plan["train_membership_sha256"],
            "train membership hash changed",
        )
        _require(
            cached_eval.membership_sha256(validation) == plan["validation_membership_sha256"],
            "validation membership hash changed",
        )
        for role, indices in (
            ("train", train_keys),
            ("guard", guard_keys),
            ("validation", validation_keys),
        ):
            block = primary.iloc[indices]
            row = dict(metadata)
            row.update(
                {
                    "fold": int(plan["fold"]),
                    "role": role,
                    "rows": int(len(block)),
                    "positive": int(block["weak_label"].astype(int).sum()),
                    "negative": int(block["weak_label"].eq(0).sum()),
                    "unique_peptides": int(block["peptide_uid"].nunique()),
                    "unique_affibodies": int(block["affibody_uid"].nunique()),
                    "membership_sha256": cached_eval.membership_sha256(block),
                    "train_validation_peptide_overlap": int(
                        len(set(train["peptide_uid"]).intersection(set(validation["peptide_uid"])))
                    ),
                    "train_validation_affibody_overlap": int(
                        len(set(train["affibody_uid"]).intersection(set(validation["affibody_uid"])))
                    ),
                }
            )
            split_rows.append(row)
    return primary, plans, split_rows


def _coefficient_rows(model, encoder, metadata, selected_c):
    spec = LIBRARY_SPECS[LIBRARY]
    features = ["pep_p{}".format(value) for value in spec["pep_positions"]]
    features += ["aff_p{}".format(value) for value in spec["aff_positions"]]
    _require(len(features) == len(encoder.categories_), "site feature/category mismatch")
    rows = []
    offset = 0
    for feature, categories in zip(features, encoder.categories_):
        for residue in categories:
            row = dict(metadata)
            row.update(
                {
                    "feature": feature,
                    "residue": str(residue),
                    "coefficient": float(model.coef_[0, offset]),
                    "intercept": float(model.intercept_[0]),
                    "selected_C": float(selected_c),
                }
            )
            rows.append(row)
            offset += 1
    _require(offset == int(model.coef_.shape[1]), "coefficient width changed")
    return rows


def fit_prediction_vectors(conditions, retention_design):
    """Tune/final-fit all conditions and freeze scores before target loading."""
    prediction_blocks = []
    tuning_rows = []
    split_rows = []
    coefficient_rows = []
    vector_rows = []
    held_sites = site_code_matrix(
        retention_design,
        peptide_column="peptide_design_code",
        affibody_column="affibody_design_code",
    )
    for arm, sampling_mode, sampling_seed, training in conditions:
        metadata = _condition_metadata(arm, sampling_mode, sampling_seed)
        assert_local_retention_identity_cold(training, retention_design)
        primary, plans, condition_splits = build_and_audit_fold_plans(training, metadata)
        split_rows.extend(condition_splits)
        raw_sites = site_code_matrix(primary)
        encoder = make_site_encoder(raw_sites.shape[1])
        encoded = encoder.fit_transform(raw_sites)
        labels = primary["weak_label"].astype(int).to_numpy()
        selected_c, condition_tuning = cached_eval.tune_c(
            "site",
            C_GRID,
            BALANCE,
            plans,
            labels,
            encoded,
            None,
            metadata,
        )
        tuning_rows.extend(condition_tuning)
        probability, model = cached_eval._fit_logistic_matrices(
            selected_c,
            BALANCE,
            encoded,
            labels,
            encoder.transform(held_sites),
        )
        probability = np.asarray(probability, dtype=np.float64).copy()
        _require(len(probability) == RETENTION_ROWS, "retention prediction length changed")
        _require(bool(np.isfinite(probability).all()), "non-finite retention prediction")
        _require(bool(((probability >= 0.0) & (probability <= 1.0)).all()), "probability outside [0,1]")
        probability.setflags(write=False)
        digest = _prediction_sha256(retention_design["pair_uid"], probability)
        block = retention_design[["pair_uid", "peptide_uid", "affibody_uid"]].copy()
        for key, value in reversed(list(metadata.items())):
            block.insert(0, key, value)
        block["selected_C"] = float(selected_c)
        block["score"] = probability
        block["prediction_vector_sha256"] = digest
        prediction_blocks.append(block)
        vector_row = dict(metadata)
        vector_row.update(
            {
                "rows": int(len(probability)),
                "training_rows": int(len(primary)),
                "training_positive": int(labels.sum()),
                "training_negative": int((labels == 0).sum()),
                "training_membership_sha256": cached_eval.membership_sha256(primary),
                "selected_C": float(selected_c),
                "prediction_vector_sha256": digest,
            }
        )
        vector_rows.append(vector_row)
        coefficient_rows.extend(_coefficient_rows(model, encoder, metadata, selected_c))
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    expected_conditions = len(conditions)
    _require(len(vector_rows) == expected_conditions, "prediction condition count changed")
    _require(len(predictions) == expected_conditions * RETENTION_ROWS, "prediction table size changed")
    return (
        predictions,
        pd.DataFrame(tuning_rows),
        pd.DataFrame(split_rows),
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(vector_rows),
    )


def attach_retention_targets(predictions, retention_csv):
    """Load targets only after scores are frozen, then verify the score hashes."""
    retention = load_retention_table(retention_csv)
    retention = retention.loc[
        retention["library"].eq(LIBRARY) & retention["target_retention"].notna()
    ].copy()
    retention = retention.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(len(retention) == RETENTION_ROWS, "LibA measured retention count changed")
    _require(int(retention["target_binder"].astype(int).sum()) == RETENTION_POSITIVES, "LibA binder count changed")
    _require(retention["peptide_uid"].nunique() == RETENTION_PEPTIDES, "LibA peptide count changed")
    per_peptide = retention.groupby("peptide_uid").size()
    _require(bool(per_peptide.eq(RETENTION_AFFIBODIES_PER_PEPTIDE).all()), "LibA is not a 9x12 panel")
    target = retention[
        ["pair_uid", "target_retention", "target_binder"]
    ].copy()
    output = predictions.merge(target, on="pair_uid", how="left", validate="many_to_one")
    _require(bool(output["target_retention"].notna().all()), "prediction lacks retention target")
    _require(
        np.array_equal(
            output["target_binder"].astype(int).to_numpy(),
            output["target_retention"].ge(RETENTION_THRESHOLD).astype(int).to_numpy(),
        ),
        "retention binder does not equal retention >=75",
    )
    keys = ["arm", "sampling_mode", "sampling_seed"]
    for _, block in output.groupby(keys, sort=False):
        _require(block["prediction_vector_sha256"].nunique() == 1, "multiple frozen score hashes")
        observed = _prediction_sha256(block["pair_uid"], block["score"])
        _require(observed == block["prediction_vector_sha256"].iloc[0], "prediction changed after target load")
    return output


def _safe_spearman(observed, score):
    observed = np.asarray(observed, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(observed) < 2 or len(np.unique(observed)) < 2 or len(np.unique(score)) < 2:
        return float("nan")
    return float(spearmanr(observed, score)[0])


def condition_metrics(panel, score, group_column="peptide_uid"):
    """Return the five report metrics and one row per peptide."""
    values = np.asarray(score, dtype=float)
    observed = pd.to_numeric(panel["target_retention"], errors="raise").to_numpy(float)
    binder = pd.to_numeric(panel["target_binder"], errors="raise").astype(int).to_numpy()
    _require(len(values) == len(panel), "metric length mismatch")
    _require(bool(np.isfinite(values).all()), "metric score is non-finite")
    _require(set(binder.tolist()) == {0, 1}, "retention metrics require both classes")
    peptide_rows = []
    for peptide_uid, indices in panel.groupby(group_column, sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        group_score = values[index]
        group_retention = observed[index]
        group_binder = binder[index]
        maximum = float(np.max(group_score))
        tied = np.flatnonzero(group_score == maximum)
        _require(len(tied) >= 1, "empty top-score tie set")
        peptide_rows.append(
            {
                "peptide_uid": str(peptide_uid),
                "n": int(len(index)),
                "positive": int(group_binder.sum()),
                "spearman": _safe_spearman(group_retention, group_score),
                "top_tie_count": int(len(tied)),
                "top_choice_binder_fraction": float(group_binder[tied].mean()),
            }
        )
    peptide = pd.DataFrame(peptide_rows)
    finite_rho = peptide["spearman"].to_numpy(float)
    _require(bool(np.isfinite(finite_rho).all()), "a peptide has undefined Spearman")
    metrics = {
        "n": int(len(panel)),
        "positive": int(binder.sum()),
        "within_peptide_groups": int(len(peptide)),
        "within_peptide_macro_spearman": float(np.mean(finite_rho)),
        "global_spearman": _safe_spearman(observed, values),
        "global_auroc": float(roc_auc_score(binder, values)),
        "global_average_precision": float(average_precision_score(binder, values)),
        "within_peptide_top_choice_success": float(peptide["top_choice_binder_fraction"].mean()),
    }
    return metrics, peptide


def compute_metrics(predictions):
    metric_rows = []
    peptide_blocks = []
    keys = ["arm", "sampling_mode", "sampling_seed"]
    for values, block in predictions.groupby(keys, sort=False):
        arm, sampling_mode, sampling_seed = values
        block = block.reset_index(drop=True)
        metrics, peptide = condition_metrics(block, block["score"])
        metadata = _condition_metadata(str(arm), str(sampling_mode), int(sampling_seed))
        row = dict(metadata)
        row.update(metrics)
        metric_rows.append(row)
        for key, value in metadata.items():
            peptide[key] = value
        peptide_blocks.append(peptide)
    metrics = pd.DataFrame(metric_rows)
    _require(bool(metrics["n"].eq(RETENTION_ROWS).all()), "metric panel size changed")
    _require(bool(metrics["positive"].eq(RETENTION_POSITIVES).all()), "metric binder count changed")
    _require(bool(metrics["within_peptide_groups"].eq(RETENTION_PEPTIDES).all()), "metric peptide count changed")
    return metrics, pd.concat(peptide_blocks, ignore_index=True)


def paired_metric_deltas(metrics):
    rows = []
    for (sampling_mode, sampling_seed), block in metrics.groupby(
        ["sampling_mode", "sampling_seed"], sort=True
    ):
        reference = block.loc[block["arm"].eq(ARM_ORDER[0])]
        _require(len(reference) == 1, "paired comparison lacks one Arm A")
        reference = reference.iloc[0]
        for arm in ARM_ORDER[1:]:
            candidate = block.loc[block["arm"].eq(arm)]
            _require(len(candidate) == 1, "paired comparison lacks {}".format(arm))
            candidate = candidate.iloc[0]
            for metric_name in METRIC_NAMES:
                rows.append(
                    {
                        "library": LIBRARY,
                        "arm_id": ARM_IDS[arm],
                        "arm": arm,
                        "arm_role": ARM_ROLES[arm],
                        "reference_arm_id": "A",
                        "reference_arm": ARM_ORDER[0],
                        "sampling_mode": str(sampling_mode),
                        "sampling_seed": int(sampling_seed),
                        "metric": metric_name,
                        "arm_value": float(candidate[metric_name]),
                        "reference_value": float(reference[metric_name]),
                        "delta_arm_minus_A": float(candidate[metric_name] - reference[metric_name]),
                    }
                )
    return pd.DataFrame(rows)


def paired_peptide_cluster_bootstrap(panel, arm_score, reference_score, draws, seed):
    """Paired percentile intervals from resampling the nine peptide clusters."""
    _require(int(draws) >= 1, "bootstrap draws must be positive")
    frame = panel.reset_index(drop=True)
    arm_score = np.asarray(arm_score, dtype=float)
    reference_score = np.asarray(reference_score, dtype=float)
    _require(len(frame) == len(arm_score) == len(reference_score), "bootstrap length mismatch")
    observed = pd.to_numeric(frame["target_retention"], errors="raise").to_numpy(float)
    binder = pd.to_numeric(frame["target_binder"], errors="raise").astype(int).to_numpy()
    groups = [np.asarray(indices, dtype=int) for _, indices in frame.groupby("peptide_uid", sort=True).indices.items()]
    _require(len(groups) >= 2, "bootstrap requires at least two peptide clusters")
    arm_group_rho = np.asarray([_safe_spearman(observed[index], arm_score[index]) for index in groups])
    ref_group_rho = np.asarray([_safe_spearman(observed[index], reference_score[index]) for index in groups])

    def top_success(score, index):
        values = score[index]
        tied = index[values == np.max(values)]
        return float(binder[tied].mean())

    arm_group_top = np.asarray([top_success(arm_score, index) for index in groups])
    ref_group_top = np.asarray([top_success(reference_score, index) for index in groups])
    _require(bool(np.isfinite(arm_group_rho).all() & np.isfinite(ref_group_rho).all()), "undefined peptide rho")
    rng = np.random.RandomState(int(seed))
    counts = rng.multinomial(
        len(groups),
        np.repeat(1.0 / len(groups), len(groups)),
        size=int(draws),
    )
    estimates = {metric: [] for metric in METRIC_NAMES}
    estimates["within_peptide_macro_spearman"] = (
        counts.dot(arm_group_rho - ref_group_rho) / float(len(groups))
    ).tolist()
    estimates["within_peptide_top_choice_success"] = (
        counts.dot(arm_group_top - ref_group_top) / float(len(groups))
    ).tolist()
    for draw_counts in counts:
        indices = np.concatenate(
            [np.tile(index, int(count)) for index, count in zip(groups, draw_counts) if int(count) > 0]
        )
        y = binder[indices]
        retention = observed[indices]
        arm = arm_score[indices]
        reference = reference_score[indices]
        estimates["global_spearman"].append(
            _safe_spearman(retention, arm) - _safe_spearman(retention, reference)
        )
        if set(y.tolist()) == {0, 1}:
            estimates["global_auroc"].append(
                float(roc_auc_score(y, arm) - roc_auc_score(y, reference))
            )
            estimates["global_average_precision"].append(
                float(average_precision_score(y, arm) - average_precision_score(y, reference))
            )
        else:
            estimates["global_auroc"].append(float("nan"))
            estimates["global_average_precision"].append(float("nan"))
    rows = []
    for metric_name in METRIC_NAMES:
        values = np.asarray(estimates[metric_name], dtype=float)
        finite = values[np.isfinite(values)]
        _require(len(finite) >= max(1, int(0.9 * int(draws))), "too few finite bootstrap draws")
        rows.append(
            {
                "metric": metric_name,
                "bootstrap_draws_requested": int(draws),
                "bootstrap_draws_finite": int(len(finite)),
                "ci_lower_2_5": float(np.percentile(finite, 2.5)),
                "ci_upper_97_5": float(np.percentile(finite, 97.5)),
                "probability_delta_gt_0": float(np.mean(finite > 0.0)),
                "probability_delta_ge_0": float(np.mean(finite >= 0.0)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_all_comparisons(predictions, draws, base_seed):
    rows = []
    keys = ["sampling_mode", "sampling_seed"]
    for (sampling_mode, sampling_seed), block in predictions.groupby(keys, sort=True):
        reference = block.loc[block["arm"].eq(ARM_ORDER[0])].sort_values("pair_uid", kind="mergesort")
        _require(len(reference) == RETENTION_ROWS, "bootstrap reference panel changed")
        for arm in ARM_ORDER[1:]:
            candidate = block.loc[block["arm"].eq(arm)].sort_values("pair_uid", kind="mergesort")
            _require(len(candidate) == RETENTION_ROWS, "bootstrap candidate panel changed")
            _require(
                np.array_equal(
                    candidate["pair_uid"].astype(str).to_numpy(),
                    reference["pair_uid"].astype(str).to_numpy(),
                ),
                "bootstrap panels do not align",
            )
            seed = _derived_seed(base_seed, sampling_mode, sampling_seed, arm)
            result = paired_peptide_cluster_bootstrap(
                reference,
                candidate["score"].to_numpy(float),
                reference["score"].to_numpy(float),
                draws,
                seed,
            )
            result.insert(0, "bootstrap_seed", int(seed))
            result.insert(0, "sampling_seed", int(sampling_seed))
            result.insert(0, "sampling_mode", str(sampling_mode))
            result.insert(0, "reference_arm", ARM_ORDER[0])
            result.insert(0, "reference_arm_id", "A")
            result.insert(0, "arm", arm)
            result.insert(0, "arm_id", ARM_IDS[arm])
            result.insert(0, "library", LIBRARY)
            rows.append(result)
    return pd.concat(rows, ignore_index=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(int(args.bootstrap_draws) >= 1000, "bootstrap draws must be at least 1000")
    _require(0 <= int(args.bootstrap_seed) < 2 ** 32, "bootstrap seed is outside uint32")

    script_path = Path(__file__).resolve()
    dependency_paths = (
        script_path,
        REPO_ROOT / "downstream/AffibodyMHC/build_liba_later_round_labels.py",
        REPO_ROOT / "downstream/AffibodyMHC/evaluate_cached_weak_mint.py",
        REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py",
    )
    code_hashes = {str(path.resolve()): sha256_file(path) for path in dependency_paths}
    labels, membership, arm_summary, label_manifest, label_paths = load_label_bundle(args.label_dir)
    initial_label_source_hashes = {
        filename: sha256_file(label_paths[filename])
        for filename in list(LABEL_FILES) + ["manifest.json"]
    }
    arm_contract = validate_arm_contract(labels)
    validate_summary_contract(arm_summary, arm_contract, label_manifest, labels)
    validate_matched_contract(membership, labels)
    retention_source_sha256 = sha256_file(args.retention_csv)
    _require(
        label_manifest.get("sources", {}).get("retention_csv", {}).get("sha256")
        == retention_source_sha256,
        "label bundle and evaluator use different retention identity sources",
    )
    retention_design = load_retention_design_panel(args.retention_csv)
    conditions = construct_conditions(labels, membership)
    _require(
        len(conditions) == len(ARM_ORDER) * (1 + len(MATCHED_SEEDS)),
        "canonical condition grid changed",
    )

    # This is the deliberate information barrier: numerical retention values
    # have not been loaded when all model-selection decisions and score vectors
    # below are completed and hashed.
    predictions, tuning, splits, coefficients, frozen_vectors = fit_prediction_vectors(
        conditions, retention_design
    )
    predictions = attach_retention_targets(predictions, args.retention_csv)
    metrics, per_peptide = compute_metrics(predictions)
    paired_deltas = paired_metric_deltas(metrics)
    bootstrap = bootstrap_all_comparisons(
        predictions,
        int(args.bootstrap_draws),
        int(args.bootstrap_seed),
    )
    paired_deltas = paired_deltas.merge(
        bootstrap[
            [
                "arm",
                "sampling_mode",
                "sampling_seed",
                "metric",
                "bootstrap_seed",
                "bootstrap_draws_requested",
                "bootstrap_draws_finite",
                "ci_lower_2_5",
                "ci_upper_97_5",
                "probability_delta_gt_0",
                "probability_delta_ge_0",
            ]
        ],
        on=["arm", "sampling_mode", "sampling_seed", "metric"],
        how="left",
        validate="one_to_one",
    )
    _require(bool(paired_deltas["ci_lower_2_5"].notna().all()), "paired bootstrap join failed")

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    outputs = {
        "arm_contract.csv": arm_contract,
        "prediction_vectors.csv": frozen_vectors,
        "predictions.csv": predictions,
        "tuning.csv": tuning,
        "splits.csv": splits,
        "metrics.csv": metrics,
        "per_peptide_metrics.csv": per_peptide,
        "paired_deltas.csv": paired_deltas,
        "model_coefficients.csv": coefficients,
    }
    output_paths = {}
    for filename, frame in outputs.items():
        path = output_dir / filename
        _write_private_csv(frame, path)
        output_paths[filename] = path

    natural = metrics.loc[metrics["sampling_mode"].eq("natural")].sort_values("arm_id")
    summary_lines = [
        "# LibA later-round label comparison: site-additive model",
        "",
        "The model, retention panel, negative examples, three identity-cold weak-validation folds, "
        "class-weighted loss, and C grid are fixed across arms. B is the primary later-round test; "
        "C checks sequencing-depth normalization; D is a late-only diagnostic.",
        "",
        "| Arm | Positive-label definition | Train positives | Macro within-peptide Spearman | Global Spearman | AUROC | AP | Top-choice success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    contract_by_arm = arm_contract.set_index("arm")
    for _, row in natural.iterrows():
        summary_lines.append(
            "| {arm_id} | {arm_name} | {positive} | {within:.4f} | {global_rho:.4f} | {auroc:.4f} | {ap:.4f} | {top:.4f} |".format(
                arm_id=row["arm_id"],
                arm_name=ARM_NAMES[row["arm"]],
                positive=int(contract_by_arm.loc[row["arm"], "natural_positive"]),
                within=row["within_peptide_macro_spearman"],
                global_rho=row["global_spearman"],
                auroc=row["global_auroc"],
                ap=row["global_average_precision"],
                top=row["within_peptide_top_choice_success"],
            )
        )
    summary_lines.extend(
        [
            "",
            "Size-matched results and paired peptide-cluster bootstrap intervals are in "
            "`paired_deltas.csv`. Positive deltas favor the later-round arm over A.",
            "",
            "Top-choice success averages the binder fraction among all Affibodies tied for the "
            "highest predicted score within each peptide; it never breaks a tie by row order.",
            "",
            "All results are retrospective. Retention values were loaded only after every score "
            "vector had been computed, made read-only, and hashed.",
            "",
        ]
    )
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary_lines))
    os.chmod(str(summary_path), 0o600)
    output_paths["run_summary.md"] = summary_path

    final_label_source_hashes = {
        filename: sha256_file(label_paths[filename]) for filename in list(LABEL_FILES) + ["manifest.json"]
    }
    _require(initial_label_source_hashes == final_label_source_hashes, "label source changed during evaluation")
    _require(retention_source_sha256 == sha256_file(args.retention_csv), "retention source changed during evaluation")
    _require(
        code_hashes == {str(path.resolve()): sha256_file(path) for path in dependency_paths},
        "analysis code changed during evaluation",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_status": "retrospective_exploratory",
        "configuration": {
            "library": LIBRARY,
            "arms": list(ARM_ORDER),
            "arm_roles": ARM_ROLES,
            "natural_positive_counts": EXPECTED_POSITIVES,
            "common_negative_count": EXPECTED_NEGATIVES,
            "size_matched_positive_count": MATCHED_POSITIVES,
            "size_matched_seeds": list(MATCHED_SEEDS),
            "model": "position-wise additive L2 logistic regression",
            "class_weight": "balanced",
            "regime": REGIME,
            "cleaning": CLEANING,
            "folds": FOLDS,
            "split_seed": SPLIT_SEED,
            "c_grid": list(C_GRID),
            "c_selection": "minimum pooled weak-validation log loss; AP, AUROC, then smaller C tie-break",
            "retention_usage": "identities/design codes before fitting; numerical targets only after frozen score-vector hashes",
            "retention_threshold": RETENTION_THRESHOLD,
            "primary_metric": "mean of nine within-peptide Spearman correlations, each across twelve Affibodies",
            "top_choice": "mean binder fraction among exact maximum-score ties within each peptide",
            "bootstrap": {
                "unit": "peptide cluster",
                "paired": True,
                "draws": int(args.bootstrap_draws),
                "base_seed": int(args.bootstrap_seed),
                "interval": "2.5th and 97.5th percentile",
            },
        },
        "sources": {
            "label_dir": str(Path(args.label_dir).resolve()),
            "label_files": initial_label_source_hashes,
            "label_manifest_recorded_outputs": label_manifest.get("outputs", {}),
            "retention_csv": {
                "path": str(Path(args.retention_csv).resolve()),
                "sha256": retention_source_sha256,
            },
        },
        "code": code_hashes,
        "rows": {name: int(len(frame)) for name, frame in outputs.items()},
        "outputs": {
            filename: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for filename, path in sorted(output_paths.items())
        },
        "permissions": {"directory": "0700", "files": "0600"},
        "elapsed_seconds": float(time.time() - started),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    _write_private_json(manifest, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "conditions": int(len(metrics)),
                "natural_metrics": natural[
                    ["arm_id", "within_peptide_macro_spearman", "global_spearman", "global_auroc", "global_average_precision"]
                ].to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
