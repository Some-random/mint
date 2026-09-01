#!/usr/bin/env python
"""Evaluate a selection-trained position-additive binder baseline.

The model is trained only from provider-defined R001/R009/R010 weak labels.
The 227 measured retention values are never used for training, splitting, or
regularization selection; they are read only for the final retrospective
evaluation.  LibA and LibB are fitted separately.

Primary weak-label filter:

* within the declared library alphabet;
* shares neither peptide nor Affibody identity with the retention matrix;
* pooled-R009/R010 top-2% positives; or
* R001>=3 negatives absent from every positive-selection round R002--R014.

Regularization is chosen with deterministic diagonal double-identity-cold
validation folds.  For each fold, validation rows have both partners in the
held blocks, training rows have neither, and one-partner overlaps are guarded
out.  After tuning, the selected model is refit on every primary weak-label
row.  Row-level outputs contain opaque identifiers, never short codes or full
sequences, and are restricted to the ignored private_data tree.
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_selection_weak_labels import (
    EXPECTED_POOLED_CUTOFF,
    discover_raw_round_files,
    load_count_round,
    pooled_top_fraction,
)
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    load_retention_table,
    sha256_file,
    validate_private_output_path,
)


LIBRARIES = ("LibA", "LibB")
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_FOLDS = 5
DEFAULT_MIN_NEGATIVE_R001_COUNT = 3
RETENTION_BINDER_THRESHOLD = 75.0
WEAK_REQUIRED_COLUMNS = (
    "library",
    "pep",
    "aff",
    "r001_count",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "weak_label",
    "within_declared_library_alphabet",
    "strict_retention_identity_cold_eligible",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
)
SEQUENCE_REQUIRED_COLUMNS = (
    "library",
    "affibody_design_code",
    "peptide_design_code",
    "target_retention",
    "target_binder",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_length",
    "chain2_length",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def write_private_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def stable_identity_bin(uid, n_folds, salt):
    """Map an opaque partner identifier to a reproducible block."""
    _require(int(n_folds) >= 2, "n_folds must be at least two")
    payload = "{}|{}".format(salt, uid).encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(n_folds)


def primary_training_frame(labels, library, min_negative_r001_count=3):
    """Apply the prespecified high-confidence, identity-cold filter."""
    _require(library in LIBRARIES, "unknown library {}".format(library))
    missing = set(WEAK_REQUIRED_COLUMNS).difference(labels.columns)
    _require(not missing, "weak labels missing columns {}".format(sorted(missing)))
    subset = labels.loc[labels["library"].eq(library)].copy()
    label_ok = subset["weak_label"].eq(1) | (
        subset["weak_label"].eq(0)
        & subset["r001_count"].ge(int(min_negative_r001_count))
    )
    mask = (
        subset["within_declared_library_alphabet"].eq(1)
        & subset["strict_retention_identity_cold_eligible"].eq(1)
        & label_ok
    )
    result = subset.loc[mask].copy()
    result = result.sort_values("pair_uid").reset_index(drop=True)
    _require(len(result) > 0, "primary filter produced no {} rows".format(library))
    _require(set(result["weak_label"].astype(int)) == {0, 1}, "{} primary data need both labels".format(library))
    _require(not bool(result["pair_uid"].duplicated().any()), "duplicate primary pair UID")
    _require(bool(result["strict_retention_identity_cold_eligible"].eq(1).all()), "non-cold primary row")
    negatives = result["weak_label"].eq(0)
    _require(
        bool(result.loc[negatives, "r001_count"].ge(int(min_negative_r001_count)).all()),
        "low-count negative survived primary filter",
    )
    return result


def add_identity_blocks(frame, library, n_folds):
    """Attach deterministic peptide and Affibody identity block numbers."""
    output = frame.copy()
    output["peptide_block"] = [
        stable_identity_bin(uid, n_folds, "{}|peptide".format(library))
        for uid in output["peptide_uid"]
    ]
    output["affibody_block"] = [
        stable_identity_bin(uid, n_folds, "{}|affibody".format(library))
        for uid in output["affibody_uid"]
    ]
    return output


def identity_blocked_indices(blocked, fold):
    """Return train/guard/validation indices for one double-cold fold."""
    peptide_held = blocked["peptide_block"].eq(int(fold)).to_numpy()
    affibody_held = blocked["affibody_block"].eq(int(fold)).to_numpy()
    validation_mask = peptide_held & affibody_held
    guard_mask = peptide_held ^ affibody_held
    train_mask = ~peptide_held & ~affibody_held
    indices = {
        "train": np.flatnonzero(train_mask),
        "guard": np.flatnonzero(guard_mask),
        "validation": np.flatnonzero(validation_mask),
    }
    _require(
        sum(len(value) for value in indices.values()) == len(blocked),
        "fold roles do not partition rows",
    )
    return indices


def audit_identity_cold_fold(blocked, indices, fold):
    train = blocked.iloc[indices["train"]]
    validation = blocked.iloc[indices["validation"]]
    _require(len(train) > 0 and len(validation) > 0, "empty train/validation in fold {}".format(fold))
    _require(set(train["weak_label"].astype(int)) == {0, 1}, "training fold lacks a label")
    _require(set(validation["weak_label"].astype(int)) == {0, 1}, "validation fold lacks a label")
    _require(
        set(train["peptide_uid"]).isdisjoint(set(validation["peptide_uid"])),
        "peptide identity leakage in fold {}".format(fold),
    )
    _require(
        set(train["affibody_uid"]).isdisjoint(set(validation["affibody_uid"])),
        "Affibody identity leakage in fold {}".format(fold),
    )


def site_feature_names(library):
    spec = LIBRARY_SPECS[library]
    return tuple(
        ["pep_p{}".format(position) for position in spec["pep_positions"]]
        + ["aff_p{}".format(position) for position in spec["aff_positions"]]
    )


def site_code_matrix(frame, library, peptide_column="pep", affibody_column="aff"):
    """Return one categorical residue column for every designed position."""
    spec = LIBRARY_SPECS[library]
    peptide = frame[peptide_column].astype(str)
    affibody = frame[affibody_column].astype(str)
    _require(bool(peptide.map(len).eq(spec["pep_length"]).all()), "bad peptide code length")
    _require(bool(affibody.map(len).eq(spec["aff_length"]).all()), "bad Affibody code length")
    columns = []
    for index in range(spec["pep_length"]):
        columns.append(peptide.str[index].to_numpy(dtype=str))
    for index in range(spec["aff_length"]):
        columns.append(affibody.str[index].to_numpy(dtype=str))
    return np.column_stack(columns)


def make_encoder(n_positions):
    return OneHotEncoder(
        categories=[list(AA_ALPHABET) for _ in range(int(n_positions))],
        handle_unknown="error",
        sparse=True,
        dtype=np.float64,
    )


def make_logistic(c_value):
    return LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        max_iter=2000,
        tol=1e-8,
    )


def _binary_metrics(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    _require(len(y_true) == len(score), "metric length mismatch")
    _require(np.isfinite(score).all(), "non-finite score")
    _require(set(y_true) == {0, 1}, "binary metrics require both classes")
    return {
        "log_loss": float(log_loss(y_true, score, labels=[0, 1])),
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
    }


def tune_regularization(encoded, labels, blocked, c_grid, n_folds):
    """Tune C exclusively on deterministic double-identity-cold weak folds."""
    fold_rows = []
    aggregate_rows = []
    validation_predictions = {}
    y = np.asarray(labels, dtype=int)
    for c_value in c_grid:
        pooled_y = []
        pooled_probability = []
        for fold in range(int(n_folds)):
            indices = identity_blocked_indices(blocked, fold)
            audit_identity_cold_fold(blocked, indices, fold)
            model = make_logistic(c_value)
            model.fit(encoded[indices["train"]], y[indices["train"]])
            probability = model.predict_proba(encoded[indices["validation"]])[:, 1]
            observed = y[indices["validation"]]
            metrics = _binary_metrics(observed, probability)
            fold_rows.append(
                {
                    "record_type": "fold",
                    "C": float(c_value),
                    "fold": int(fold),
                    "n_train": int(len(indices["train"])),
                    "n_guard": int(len(indices["guard"])),
                    "n_validation": int(len(indices["validation"])),
                    "validation_positive": int(observed.sum()),
                    "log_loss": metrics["log_loss"],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                }
            )
            pooled_y.extend(observed.tolist())
            pooled_probability.extend(probability.tolist())
        aggregate = _binary_metrics(pooled_y, pooled_probability)
        aggregate_rows.append(
            {
                "record_type": "aggregate",
                "C": float(c_value),
                "fold": -1,
                "n_train": int(sum(row["n_train"] for row in fold_rows if row["C"] == float(c_value))),
                "n_guard": int(sum(row["n_guard"] for row in fold_rows if row["C"] == float(c_value))),
                "n_validation": int(len(pooled_y)),
                "validation_positive": int(sum(pooled_y)),
                "log_loss": aggregate["log_loss"],
                "auroc": aggregate["auroc"],
                "auprc": aggregate["auprc"],
            }
        )
        validation_predictions[float(c_value)] = (pooled_y, pooled_probability)

    ranked = sorted(
        aggregate_rows,
        key=lambda row: (row["log_loss"], -row["auroc"], row["C"]),
    )
    selected_c = float(ranked[0]["C"])
    all_rows = fold_rows + aggregate_rows
    for row in all_rows:
        row["selected"] = int(row["C"] == selected_c)
    return selected_c, pd.DataFrame(all_rows)


def pool_counts_for_pairs(round9, round10, pairs, cutoff):
    """Join raw R009/R010 counts onto assay pairs and apply a global cutoff."""
    keys = ["pep", "aff"]
    _require(not bool(pairs[keys].duplicated().any()), "duplicate evaluation pair")
    left = round9[keys + ["count"]].rename(columns={"count": "r009_count"})
    right = round10[keys + ["count"]].rename(columns={"count": "r010_count"})
    output = pairs.merge(left, on=keys, how="left", validate="one_to_one")
    output = output.merge(right, on=keys, how="left", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        output[column] = output[column].fillna(0).astype(np.int64)
    output["pooled_r009_r010_count"] = output["r009_count"] + output["r010_count"]
    output["pooled_r009_r010_log1p_count"] = np.log1p(
        output["pooled_r009_r010_count"].to_numpy(dtype=float)
    )
    output["pooled_r009_r010_top2_binary"] = output[
        "pooled_r009_r010_count"
    ].ge(int(cutoff)).astype(int)
    return output


def evaluation_metrics(retention, score, predictor, score_definition, library, scope):
    observed = retention["target_retention"].to_numpy(dtype=float)
    binder = retention["target_binder"].astype(int).to_numpy()
    values = np.asarray(score, dtype=float)
    _require(len(observed) == len(values), "evaluation score length mismatch")
    _require(np.isfinite(values).all(), "non-finite evaluation score")
    rho, pvalue = spearmanr(observed, values)
    return {
        "library": library,
        "evaluation_scope": scope,
        "predictor": predictor,
        "score_definition": score_definition,
        "n": int(len(observed)),
        "binders_ge_75": int(binder.sum()),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
        "auroc": float(roc_auc_score(binder, values)),
        "auprc": float(average_precision_score(binder, values)),
    }


def load_and_validate_weak_inputs(weak_label_dir):
    paths = {
        "weak_labels": weak_label_dir / "weak_labels.csv",
        "holdout_overlap": weak_label_dir / "holdout_overlap.csv",
        "label_summary": weak_label_dir / "label_summary.csv",
        "manifest": weak_label_dir / "manifest.json",
    }
    for name, path in paths.items():
        _require(path.is_file(), "missing weak-label {}: {}".format(name, path))
    with open(str(paths["manifest"]), "r") as handle:
        manifest = json.load(handle)
    for filename in ("weak_labels.csv", "holdout_overlap.csv", "label_summary.csv"):
        _require(
            manifest["outputs"].get(filename) == sha256_file(weak_label_dir / filename),
            "weak-label {} hash mismatch".format(filename),
        )
    _require(float(manifest["configuration"]["top_fraction"]) == 0.02, "weak labels are not top 2%")
    # Codes such as the literal peptide ``NA`` are biological strings, not
    # missing values.  Fail closed by loading strings first and converting the
    # explicitly numeric fields below.
    labels = pd.read_csv(
        paths["weak_labels"], dtype=str, keep_default_na=False, na_filter=False
    )
    holdout = pd.read_csv(
        paths["holdout_overlap"], dtype=str, keep_default_na=False, na_filter=False
    )
    missing = set(WEAK_REQUIRED_COLUMNS).difference(labels.columns)
    _require(not missing, "weak-label schema missing {}".format(sorted(missing)))
    numeric_columns = (
        "r001_count",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "weak_label",
        "within_declared_library_alphabet",
        "strict_retention_identity_cold_eligible",
    )
    for column in numeric_columns:
        converted = pd.to_numeric(labels[column], errors="coerce")
        _require(bool(converted.notna().all()), "nonnumeric weak-label {}".format(column))
        _require(bool(np.equal(converted, np.floor(converted)).all()), "noninteger weak-label {}".format(column))
        labels[column] = converted.astype(np.int64)
    for column in (
        "measurement_missing",
        "is_pooled_positive_before_exclusion",
        "is_provider_negative_before_exclusion",
    ):
        converted = pd.to_numeric(holdout[column], errors="coerce")
        _require(bool(converted.notna().all()), "nonnumeric holdout {}".format(column))
        _require(bool(np.equal(converted, np.floor(converted)).all()), "noninteger holdout {}".format(column))
        holdout[column] = converted.astype(np.int64)
    _require(len(holdout) == 228, "weak-label holdout audit must have 228 rows")
    _require(set(labels["pair_uid"]).isdisjoint(set(holdout["pair_uid"])), "exact holdout leaked into weak labels")
    return labels, holdout, manifest, paths


def load_and_validate_sequences(sequence_csv, retention):
    # Preserve literal codes such as ``NA`` and materialize numeric columns
    # explicitly, matching the private retention loader's NA-safe policy.
    sequences = pd.read_csv(
        sequence_csv, dtype=str, keep_default_na=False, na_filter=False
    )
    missing = set(SEQUENCE_REQUIRED_COLUMNS).difference(sequences.columns)
    _require(not missing, "retention sequence table missing {}".format(sorted(missing)))
    _require(len(sequences) == 228, "sequence table must have 228 rows")
    _require(not bool(sequences["pair_uid"].duplicated().any()), "duplicate sequence pair UID")
    retained = retention[
        [
            "pair_uid",
            "library",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
            "target_binder",
            "peptide_uid",
            "affibody_uid",
        ]
    ].copy()
    sequence_keys = sequences[
        [
            "pair_uid",
            "library",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
            "target_binder",
            "peptide_uid",
            "affibody_uid",
        ]
    ].copy()
    merged = retained.merge(sequence_keys, on="pair_uid", suffixes=("_matrix", "_sequence"), validate="one_to_one")
    _require(len(merged) == 228, "matrix/sequence pair mismatch")
    for column in ("library", "peptide_design_code", "affibody_design_code", "peptide_uid", "affibody_uid"):
        _require(
            bool(merged[column + "_matrix"].eq(merged[column + "_sequence"]).all()),
            "matrix/sequence {} mismatch".format(column),
        )
    matrix_retention = pd.to_numeric(
        merged["target_retention_matrix"], errors="coerce"
    ).astype("float64").to_numpy()
    sequence_retention = pd.to_numeric(
        merged["target_retention_sequence"], errors="coerce"
    ).astype("float64").to_numpy()
    _require(bool(np.allclose(matrix_retention, sequence_retention, equal_nan=True)), "matrix/sequence retention mismatch")
    matrix_binder = pd.to_numeric(
        merged["target_binder_matrix"], errors="coerce"
    ).astype("float64").to_numpy()
    sequence_binder = pd.to_numeric(
        merged["target_binder_sequence"], errors="coerce"
    ).astype("float64").to_numpy()
    _require(bool(np.allclose(matrix_binder, sequence_binder, equal_nan=True)), "matrix/sequence binder mismatch")
    chain1_length = pd.to_numeric(sequences["chain1_length"], errors="coerce")
    chain2_length = pd.to_numeric(sequences["chain2_length"], errors="coerce")
    _require(bool(chain1_length.eq(270).all()), "chain 1 length mismatch")
    _require(bool(chain2_length.eq(58).all()), "chain 2 length mismatch")
    for _, row in sequences.iterrows():
        chain1 = str(row["chain1_smart_hla_linker_peptide_sequence"])
        chain2 = str(row["chain2_affibody_sequence"])
        _require(_sha256_text(chain1) == row["chain1_sha256"], "chain 1 hash mismatch")
        _require(_sha256_text(chain2) == row["chain2_sha256"], "chain 2 hash mismatch")
        _require(_sha256_text(chain1 + "|" + chain2) == row["sequence_pair_sha256"], "sequence pair hash mismatch")
    return sequences


def adjacent_sequence_manifest(sequence_csv):
    name = sequence_csv.name
    _require(name.endswith(".csv"), "sequence input must be CSV")
    return sequence_csv.with_name(name[:-4] + ".manifest.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-label-dir", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--retention-sequences-csv", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--c-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_C_GRID),
    )
    parser.add_argument(
        "--min-negative-r001-count",
        type=int,
        default=DEFAULT_MIN_NEGATIVE_R001_COUNT,
    )
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.weak_label_dir.is_dir(), "weak-label directory does not exist")
    _require(args.retention_csv.is_file(), "retention matrix does not exist")
    _require(args.retention_sequences_csv.is_file(), "retention sequence table does not exist")
    _require(args.raw_root.is_dir(), "raw root does not exist")
    _require(int(args.folds) >= 2, "folds must be at least two")
    c_grid = tuple(sorted(set(float(value) for value in args.c_grid)))
    _require(c_grid and all(math.isfinite(value) and value > 0 for value in c_grid), "C values must be positive and finite")
    _require(int(args.min_negative_r001_count) >= 1, "negative R001 threshold must be positive")

    script_path = Path(__file__).resolve()
    dependencies = (
        script_path,
        REPO_ROOT / "downstream/AffibodyMHC/build_selection_weak_labels.py",
        REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py",
    )
    code_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in dependencies
    }
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    weak_labels, holdout_overlap, weak_manifest, weak_paths = load_and_validate_weak_inputs(
        args.weak_label_dir
    )
    retention = load_retention_table(args.retention_csv)
    sequences = load_and_validate_sequences(args.retention_sequences_csv, retention)
    del sequences
    sequence_manifest_path = adjacent_sequence_manifest(args.retention_sequences_csv)
    _require(sequence_manifest_path.is_file(), "missing adjacent sequence manifest")
    with open(str(sequence_manifest_path), "r") as handle:
        sequence_manifest = json.load(handle)
    _require(
        sequence_manifest["output"]["sha256"] == sha256_file(args.retention_sequences_csv),
        "sequence manifest output hash mismatch",
    )
    _require(
        sequence_manifest["sources"]["retention_csv"]["sha256"] == sha256_file(args.retention_csv),
        "sequence manifest retention hash mismatch",
    )
    _require(
        weak_manifest["sources"]["retention_csv"]["sha256"] == sha256_file(args.retention_csv),
        "weak-label manifest retention hash mismatch",
    )

    raw_files = discover_raw_round_files(args.raw_root)
    direct_raw_paths = {
        "{}:R{:03d}".format(library, round_index): raw_files[library][round_index]
        for library in LIBRARIES
        for round_index in (9, 10)
    }
    source_hashes = {
        "weak_labels.csv": sha256_file(weak_paths["weak_labels"]),
        "weak_manifest.json": sha256_file(weak_paths["manifest"]),
        "weak_holdout_overlap.csv": sha256_file(weak_paths["holdout_overlap"]),
        "retention_csv": sha256_file(args.retention_csv),
        "retention_sequences_csv": sha256_file(args.retention_sequences_csv),
        "retention_sequences_manifest": sha256_file(sequence_manifest_path),
    }
    direct_raw_hashes = {name: sha256_file(path) for name, path in direct_raw_paths.items()}
    for name, digest in direct_raw_hashes.items():
        _require(
            weak_manifest["sources"]["raw_rounds"][name]["sha256"] == digest,
            "{} differs from the raw round used to build weak labels".format(name),
        )

    prediction_blocks = []
    metric_rows = []
    tuning_blocks = []
    split_blocks = []
    coefficient_blocks = []
    library_manifest = {}
    for library in LIBRARIES:
        primary = primary_training_frame(
            weak_labels,
            library,
            min_negative_r001_count=args.min_negative_r001_count,
        )
        blocked = add_identity_blocks(primary, library, args.folds)
        raw_sites = site_code_matrix(primary, library)
        encoder = make_encoder(raw_sites.shape[1])
        encoded = encoder.fit_transform(raw_sites)
        y_weak = primary["weak_label"].astype(int).to_numpy()
        selected_c, tuning = tune_regularization(
            encoded,
            y_weak,
            blocked,
            c_grid,
            args.folds,
        )
        tuning.insert(0, "library", library)
        tuning_blocks.append(tuning)

        split_audit = blocked[
            [
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
                "weak_label",
                "r001_count",
                "peptide_block",
                "affibody_block",
            ]
        ].copy()
        split_audit.insert(0, "library", library)
        split_audit["is_diagonal_validation_row"] = split_audit["peptide_block"].eq(
            split_audit["affibody_block"]
        ).astype(int)
        split_blocks.append(split_audit)

        model = make_logistic(selected_c)
        model.fit(encoded, y_weak)
        _require(bool(model.n_iter_[0] < model.max_iter), "{} refit did not converge".format(library))
        names = site_feature_names(library)
        _require(len(names) == len(encoder.categories_), "feature/category mismatch")
        offset = 0
        coefficient_rows = []
        for feature_name, categories in zip(names, encoder.categories_):
            for category in categories:
                coefficient_rows.append(
                    {
                        "library": library,
                        "feature": feature_name,
                        "residue": str(category),
                        "coefficient": float(model.coef_[0, offset]),
                        "selected_C": float(selected_c),
                        "intercept": float(model.intercept_[0]),
                    }
                )
                offset += 1
        _require(offset == model.coef_.shape[1], "coefficient width mismatch")
        coefficient_blocks.append(pd.DataFrame(coefficient_rows))

        held = retention.loc[
            retention["library"].eq(library) & retention["target_retention"].notna()
        ].copy()
        held = held.sort_values("pair_uid").reset_index(drop=True)
        held_sites = site_code_matrix(
            held,
            library,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
        )
        weak_probability = model.predict_proba(encoder.transform(held_sites))[:, 1]

        round9 = load_count_round(raw_files[library][9], library)
        round10 = load_count_round(raw_files[library][10], library)
        _, cutoff, requested_rank, pooled_rows = pooled_top_fraction(round9, round10, 0.02)
        _require(cutoff == EXPECTED_POOLED_CUTOFF[library], "{} top-2% cutoff changed".format(library))
        pair_keys = held[
            ["pair_uid", "peptide_design_code", "affibody_design_code"]
        ].rename(columns={"peptide_design_code": "pep", "affibody_design_code": "aff"})
        direct = pool_counts_for_pairs(round9, round10, pair_keys, cutoff)
        _require(bool(direct["pair_uid"].eq(held["pair_uid"]).all()), "direct proxy order changed")
        overlap_library = holdout_overlap.loc[
            holdout_overlap["library"].eq(library),
            ["pair_uid", "is_pooled_positive_before_exclusion"],
        ]
        proxy_audit = direct[["pair_uid", "pooled_r009_r010_top2_binary"]].merge(
            overlap_library, on="pair_uid", validate="one_to_one"
        )
        _require(
            bool(
                proxy_audit["pooled_r009_r010_top2_binary"].eq(
                    proxy_audit["is_pooled_positive_before_exclusion"]
                ).all()
            ),
            "{} direct top-2% proxy disagrees with weak-label holdout audit".format(library),
        )

        predictions = held[
            [
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
                "library",
                "target_retention",
                "target_binder",
            ]
        ].copy()
        predictions["weak_site_logistic_probability"] = weak_probability
        predictions["direct_r009_count"] = direct["r009_count"].to_numpy(dtype=np.int64)
        predictions["direct_r010_count"] = direct["r010_count"].to_numpy(dtype=np.int64)
        predictions["direct_pooled_r009_r010_count"] = direct[
            "pooled_r009_r010_count"
        ].to_numpy(dtype=np.int64)
        predictions["direct_pooled_r009_r010_log1p_count"] = direct[
            "pooled_r009_r010_log1p_count"
        ].to_numpy(dtype=float)
        predictions["direct_pooled_r009_r010_top2_binary"] = direct[
            "pooled_r009_r010_top2_binary"
        ].to_numpy(dtype=int)
        prediction_blocks.append(predictions)

        metric_rows.extend(
            [
                evaluation_metrics(
                    held,
                    predictions["weak_site_logistic_probability"],
                    "weak_site_logistic",
                    "probability from selection-only position-additive logistic model",
                    library,
                    "within_library_primary",
                ),
                evaluation_metrics(
                    held,
                    predictions["direct_pooled_r009_r010_log1p_count"],
                    "direct_pooled_r009_r010_log1p_count",
                    "log1p(raw R009 count + raw R010 count) for the same assayed pair",
                    library,
                    "within_library_retrospective_proxy",
                ),
                evaluation_metrics(
                    held,
                    predictions["direct_pooled_r009_r010_top2_binary"],
                    "direct_pooled_r009_r010_top2_binary",
                    "inclusive global top-2% indicator from raw pooled R009+R010 count",
                    library,
                    "within_library_retrospective_proxy",
                ),
            ]
        )
        library_manifest[library] = {
            "primary_weak_rows": int(len(primary)),
            "primary_weak_positive": int(y_weak.sum()),
            "primary_weak_negative": int((y_weak == 0).sum()),
            "unique_weak_peptides": int(primary["peptide_uid"].nunique()),
            "unique_weak_affibodies": int(primary["affibody_uid"].nunique()),
            "selected_C": float(selected_c),
            "refit_rows": int(len(primary)),
            "refit_iterations": int(model.n_iter_[0]),
            "retention_evaluation_rows": int(len(held)),
            "retention_evaluation_binders_ge_75": int(held["target_binder"].astype(int).sum()),
            "pooled_union_rows": int(pooled_rows),
            "top2_requested_rank": int(requested_rank),
            "top2_inclusive_count_cutoff": int(cutoff),
        }

    predictions = pd.concat(prediction_blocks, ignore_index=True)
    _require(len(predictions) == 227, "final predictions must contain 227 measured pairs")
    _require(not bool(predictions["pair_uid"].duplicated().any()), "duplicate final prediction")
    _require(set(predictions["pair_uid"]).isdisjoint(set(weak_labels["pair_uid"])), "exact weak-label pair leaked into evaluation")
    for predictor, column, definition in (
        (
            "weak_site_logistic",
            "weak_site_logistic_probability",
            "pooled probabilities from separately fitted LibA/LibB models",
        ),
        (
            "direct_pooled_r009_r010_log1p_count",
            "direct_pooled_r009_r010_log1p_count",
            "pooled log-count across libraries; sequencing scales may differ",
        ),
        (
            "direct_pooled_r009_r010_top2_binary",
            "direct_pooled_r009_r010_top2_binary",
            "library-specific inclusive top-2% indicators pooled descriptively",
        ),
    ):
        metric_rows.append(
            evaluation_metrics(
                predictions,
                predictions[column],
                predictor,
                definition,
                "Combined",
                "combined_descriptive",
            )
        )

    metrics = pd.DataFrame(metric_rows)
    tuning = pd.concat(tuning_blocks, ignore_index=True)
    splits = pd.concat(split_blocks, ignore_index=True)
    coefficients = pd.concat(coefficient_blocks, ignore_index=True)
    output_paths = {
        "predictions.csv": output_dir / "predictions.csv",
        "metrics.csv": output_dir / "metrics.csv",
        "tuning.csv": output_dir / "tuning.csv",
        "split_audit.csv": output_dir / "split_audit.csv",
        "model_coefficients.csv": output_dir / "model_coefficients.csv",
    }
    write_private_csv(predictions, output_paths["predictions.csv"])
    write_private_csv(metrics, output_paths["metrics.csv"])
    write_private_csv(tuning, output_paths["tuning.csv"])
    write_private_csv(splits, output_paths["split_audit.csv"])
    write_private_csv(coefficients, output_paths["model_coefficients.csv"])

    primary_metrics = metrics.loc[metrics["library"].isin(LIBRARIES)].copy()
    summary_lines = [
        "# Selection weak-label additive baseline",
        "",
        "LibA and LibB were trained separately using only selection-derived binary labels. "
        "Retention was not used for training, split construction, or C selection.",
        "",
        "## Data used",
        "",
        "| Library | Weak train rows | Positive | Negative | Retention evaluation rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for library in LIBRARIES:
        item = library_manifest[library]
        summary_lines.append(
            "| {library} | {primary_weak_rows} | {primary_weak_positive} | "
            "{primary_weak_negative} | {retention_evaluation_rows} |".format(
                library=library, **item
            )
        )
    summary_lines.extend(
        [
            "",
            "Primary filter: declared-library alphabet, strict retention identity-cold eligibility, "
            "all top-2% positives, and provider negatives with R001 count >= {}.".format(
                int(args.min_negative_r001_count)
            ),
            "",
            "## Retrospective retention evaluation",
            "",
            "| Library | Predictor | Spearman rho | AUROC (retention >=75) | AUPRC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in primary_metrics.iterrows():
        summary_lines.append(
            "| {library} | {predictor} | {spearman_rho:.4f} | {auroc:.4f} | {auprc:.4f} |".format(
                **row.to_dict()
            )
        )
    summary_lines.extend(
        [
            "",
            "The direct R009/R010 rows are retrospective proxies on the same assayed pairs, not "
            "generalization estimates. R009/R010 were themselves chosen after comparison with the "
            "retention experiment. The weak model is identity-cold with respect to the matrix, but "
            "this entire analysis remains retrospective.",
            "",
            "Regularization was selected by pooled log loss across deterministic diagonal "
            "double-identity-cold weak-label folds. Rows sharing exactly one held partner were "
            "guarded out for that fold. The selected model was then refit on all primary weak data.",
            "",
        ]
    )
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary_lines))
    os.chmod(str(summary_path), 0o600)
    output_paths["run_summary.md"] = summary_path

    # Re-hash every directly consumed mutable source before accepting results.
    final_source_hashes = {
        "weak_labels.csv": sha256_file(weak_paths["weak_labels"]),
        "weak_manifest.json": sha256_file(weak_paths["manifest"]),
        "weak_holdout_overlap.csv": sha256_file(weak_paths["holdout_overlap"]),
        "retention_csv": sha256_file(args.retention_csv),
        "retention_sequences_csv": sha256_file(args.retention_sequences_csv),
        "retention_sequences_manifest": sha256_file(sequence_manifest_path),
    }
    _require(source_hashes == final_source_hashes, "a direct input changed during the run")
    final_raw_hashes = {name: sha256_file(path) for name, path in direct_raw_paths.items()}
    _require(direct_raw_hashes == final_raw_hashes, "a raw R009/R010 input changed during the run")

    final_code_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in dependencies
    }
    _require(
        code_hashes == final_code_hashes,
        "analysis code changed during the run",
    )
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory",
        "code": code_hashes,
        "configuration": {
            "libraries_fitted_separately": True,
            "model": "position-wise additive L2 logistic regression",
            "class_weight": "balanced",
            "primary_filter": {
                "within_declared_library_alphabet": 1,
                "strict_retention_identity_cold_eligible": 1,
                "positive": "all provider pooled-R009/R010 top-2% positives",
                "negative": "provider negative and R001 count >= {}".format(
                    int(args.min_negative_r001_count)
                ),
            },
            "c_grid": list(c_grid),
            "selected_by": "minimum pooled validation log loss; AUROC then smaller C tie-break",
            "validation": "{} deterministic diagonal double-identity-cold folds; one-partner rows guarded".format(
                int(args.folds)
            ),
            "retention_usage": "final retrospective evaluation only",
            "retention_binder_threshold": RETENTION_BINDER_THRESHOLD,
            "direct_proxy": "log1p raw R009+R010 count and inclusive library-specific top-2% indicator",
        },
        "sources": {
            "weak_label_dir": str(args.weak_label_dir.resolve()),
            "weak_label_hashes": source_hashes,
            "weak_label_source_manifest_sha256": source_hashes["weak_manifest.json"],
            "retention_csv": str(args.retention_csv.resolve()),
            "retention_sequences_csv": str(args.retention_sequences_csv.resolve()),
            "retention_sequences_manifest": str(sequence_manifest_path.resolve()),
            "raw_root": str(args.raw_root.resolve()),
            "direct_raw_rounds": {
                name: {"path": str(direct_raw_paths[name].resolve()), "sha256": digest}
                for name, digest in sorted(direct_raw_hashes.items())
            },
            "weak_builder_recorded_raw_round_hashes": weak_manifest["sources"]["raw_rounds"],
        },
        "libraries": library_manifest,
        "rows": {
            "weak_primary": int(len(splits)),
            "retention_predictions": int(len(predictions)),
            "tuning_records": int(len(tuning)),
            "coefficient_records": int(len(coefficients)),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(output_paths.items())
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
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "weak_primary_rows": int(len(splits)),
                "retention_predictions": int(len(predictions)),
                "libraries": library_manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run(parse_args())
