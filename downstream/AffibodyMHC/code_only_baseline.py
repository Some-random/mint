#!/usr/bin/env python
"""Run position-aware code-only baselines on private Affibody retention data.

The short peptide and Affibody codes are treated as categorical residues at
known designed positions. They are never treated as complete protein chains.
All row-level outputs use opaque hashes rather than emitting the source codes.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_COLUMNS = [
    "provisional_matrix_key",
    "library",
    "affibody_design_code",
    "peptide_design_code",
    "axis_assignment_basis",
    "retention_percent",
    "retention_time_min_as_labeled",
    "timepoint_status",
    "binder_label_ge_75",
    "measurement_missing",
    "source_slide",
]
LIBRARY_SPECS = {
    "LibA": {
        "aff_positions": (13, 17, 27, 31),
        "pep_positions": (4, 5),
        "aff_length": 4,
        "pep_length": 2,
        "grid_rows": 108,
        "measured_rows": 108,
        "peptide_codes": 9,
        "affibody_codes": 12,
    },
    "LibB": {
        "aff_positions": (6, 10, 13, 14, 17),
        "pep_positions": (4, 5),
        "aff_length": 5,
        "pep_length": 2,
        "grid_rows": 120,
        "measured_rows": 119,
        "peptide_codes": 12,
        "affibody_codes": 10,
    },
}
MODEL_NAMES = ("mean_prior", "whole_code_additive", "site_residue_additive")
SPLIT_SCHEMES = ("random_pair", "peptide_cold", "affibody_cold", "double_cold")
TASKS = ("regression", "classification")
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_id(*parts):
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:20]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_private_output_path(output_dir, repo_root):
    """Fail closed unless output is below this repo's ignored private_data tree."""
    repo_root = Path(repo_root).resolve()
    private_root = (repo_root / "private_data").resolve()
    output_dir = Path(output_dir).resolve()
    _require(private_root.is_dir(), "repository private_data directory does not exist")
    _require(output_dir != private_root, "output must be a child of private_data, not private_data itself")
    try:
        output_dir.relative_to(private_root)
    except ValueError:
        raise ValueError("output directory must be inside the repository private_data tree")

    relative_output = output_dir.relative_to(repo_root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative_output)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(
        ignored.returncode == 0,
        "output directory is not Git-ignored; refusing to write private results",
    )
    return output_dir


def run_fingerprint(source_sha256, script_sha256, configuration):
    payload = {
        "source_sha256": source_sha256,
        "script_sha256": script_sha256,
        "configuration": configuration,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _parse_retention(value, row_number):
    if value == "":
        return np.nan
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError("retention is not numeric at source row {}".format(row_number))
    _require(math.isfinite(parsed), "non-finite retention at source row {}".format(row_number))
    _require(0.0 <= parsed <= 100.0, "retention outside [0,100] at row {}".format(row_number))
    return parsed


def load_retention_table(path):
    """Load and fail-closed validate the private deck-derived retention table."""
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(list(frame.columns) == EXPECTED_COLUMNS, "unexpected retention table schema")
    _require(len(frame) == 228, "expected 228 designed cells, found {}".format(len(frame)))

    critical = [
        "provisional_matrix_key",
        "library",
        "affibody_design_code",
        "peptide_design_code",
        "retention_percent",
        "binder_label_ge_75",
        "measurement_missing",
    ]
    for column in critical:
        bad = frame[column].map(lambda value: value != value.strip())
        _require(not bool(bad.any()), "whitespace found in column {}".format(column))

    _require(set(frame["library"]) == set(LIBRARY_SPECS), "unexpected library values")
    key_columns = ["library", "peptide_design_code", "affibody_design_code"]
    _require(not bool(frame.duplicated(key_columns).any()), "duplicate biological pair key")
    _require(frame["provisional_matrix_key"].nunique() == len(frame), "duplicate provisional key")

    parsed_retention = []
    for offset, value in enumerate(frame["retention_percent"], start=2):
        parsed_retention.append(_parse_retention(value, offset))
    frame = frame.copy()
    frame["target_retention"] = np.asarray(parsed_retention, dtype=float)
    # Keep an unmeasured retention cell unlabeled.  Treating NaN >= 75 as False
    # would silently turn that cell into a negative example if a future caller
    # forgot to apply the supervised-row filter before classification.
    frame["target_binder"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    measured_target = frame["target_retention"].notna()
    frame.loc[measured_target, "target_binder"] = (
        frame.loc[measured_target, "target_retention"] >= 75.0
    ).astype(int)

    for library, spec in LIBRARY_SPECS.items():
        subset = frame.loc[frame["library"] == library].copy()
        _require(len(subset) == spec["grid_rows"], "{} grid-row mismatch".format(library))
        _require(
            subset["peptide_design_code"].nunique() == spec["peptide_codes"],
            "{} peptide-code count mismatch".format(library),
        )
        _require(
            subset["affibody_design_code"].nunique() == spec["affibody_codes"],
            "{} Affibody-code count mismatch".format(library),
        )
        expected_grid = spec["peptide_codes"] * spec["affibody_codes"]
        _require(expected_grid == len(subset), "{} is not a complete designed grid".format(library))

        for code_column, expected_length in (
            ("peptide_design_code", spec["pep_length"]),
            ("affibody_design_code", spec["aff_length"]),
        ):
            lengths_ok = subset[code_column].map(len).eq(expected_length)
            _require(bool(lengths_ok.all()), "{} has invalid {} length".format(library, code_column))
            alphabet_ok = subset[code_column].map(lambda code: set(code).issubset(AA_ALPHABET))
            _require(bool(alphabet_ok.all()), "{} has noncanonical code characters".format(library))

        measured = subset["target_retention"].notna()
        _require(int(measured.sum()) == spec["measured_rows"], "{} measured-row mismatch".format(library))
        missing_flag = subset["measurement_missing"].eq("1")
        _require(bool((missing_flag == ~measured).all()), "{} missing flag mismatch".format(library))
        _require(
            bool(subset.loc[~measured, "binder_label_ge_75"].eq("").all()),
            "{} missing target has a binder label".format(library),
        )
        stored = subset.loc[measured, "binder_label_ge_75"]
        expected = subset.loc[measured, "target_binder"].astype(str)
        _require(bool(stored.eq(expected).all()), "{} binder threshold mismatch".format(library))

        aff_columns = []
        for index, position in enumerate(spec["aff_positions"]):
            name = "aff_p{}".format(position)
            frame.loc[subset.index, name] = subset["affibody_design_code"].str[index]
            aff_columns.append(name)
        pep_columns = []
        for index, position in enumerate(spec["pep_positions"]):
            name = "pep_p{}".format(position)
            frame.loc[subset.index, name] = subset["peptide_design_code"].str[index]
            pep_columns.append(name)
        spec["site_columns"] = tuple(aff_columns + pep_columns)

    frame["pair_uid"] = [
        opaque_id(library, peptide, affibody)
        for library, peptide, affibody in zip(
            frame["library"], frame["peptide_design_code"], frame["affibody_design_code"]
        )
    ]
    frame["peptide_uid"] = [
        opaque_id(library, "pep", peptide)
        for library, peptide in zip(frame["library"], frame["peptide_design_code"])
    ]
    frame["affibody_uid"] = [
        opaque_id(library, "aff", affibody)
        for library, affibody in zip(frame["library"], frame["affibody_design_code"])
    ]
    _require(frame["pair_uid"].nunique() == len(frame), "opaque pair ID collision")
    return frame


def supervised_library_frame(frame, library):
    subset = frame.loc[
        frame["library"].eq(library) & frame["target_retention"].notna()
    ].copy()
    subset = subset.sort_values(["peptide_design_code", "affibody_design_code"])
    return subset.reset_index(drop=True)


def feature_columns(model_name, library):
    if model_name == "mean_prior":
        return ()
    if model_name == "whole_code_additive":
        return ("peptide_design_code", "affibody_design_code")
    if model_name == "site_residue_additive":
        return LIBRARY_SPECS[library]["site_columns"]
    raise ValueError("unknown model {}".format(model_name))


def feature_matrix(frame, model_name, library):
    columns = feature_columns(model_name, library)
    if not columns:
        return np.zeros((len(frame), 1), dtype=float)
    return frame.loc[:, list(columns)].to_numpy(dtype=str)


def make_estimator(model_name, task, library, parameter, seed):
    if model_name == "mean_prior":
        if task == "regression":
            return DummyRegressor(strategy="mean")
        return DummyClassifier(strategy="prior")

    columns = feature_columns(model_name, library)
    if model_name == "site_residue_additive":
        categories = [list(AA_ALPHABET) for _ in columns]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="error",
            sparse=False,
            dtype=np.float64,
        )
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float64,
        )

    if task == "regression":
        head = Ridge(alpha=float(parameter), fit_intercept=True, solver="lsqr")
    else:
        head = LogisticRegression(
            C=float(parameter),
            penalty="l2",
            solver="liblinear",
            fit_intercept=True,
            max_iter=2000,
            random_state=int(seed),
        )
    return Pipeline([("onehot", encoder), ("head", head)])


def make_outer_splits(frame, scheme, seed, random_repeats):
    indices = np.arange(len(frame), dtype=int)
    binder = frame["target_binder"].to_numpy(dtype=int)
    records = []
    if scheme == "random_pair":
        splitter = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=int(random_repeats),
            random_state=int(seed),
        )
        for number, (train, test) in enumerate(splitter.split(indices, binder)):
            repeat = number // 5
            fold = number % 5
            records.append(
                {
                    "train": train,
                    "test": test,
                    "guarded": np.asarray([], dtype=int),
                    "outer_fold": "repeat{:02d}_fold{:02d}".format(repeat, fold),
                    "repeat": repeat,
                }
            )
        return records

    if scheme in ("peptide_cold", "affibody_cold"):
        group_column = "peptide_design_code" if scheme == "peptide_cold" else "affibody_design_code"
        groups = frame[group_column].to_numpy(dtype=str)
        splitter = LeaveOneGroupOut()
        for number, (train, test) in enumerate(splitter.split(indices, binder, groups=groups)):
            records.append(
                {
                    "train": train,
                    "test": test,
                    "guarded": np.asarray([], dtype=int),
                    "outer_fold": "group{:02d}".format(number),
                    "repeat": 0,
                }
            )
        return records

    if scheme == "double_cold":
        peptide = frame["peptide_design_code"].to_numpy(dtype=str)
        affibody = frame["affibody_design_code"].to_numpy(dtype=str)
        for test_index in indices:
            test = np.asarray([test_index], dtype=int)
            train_mask = (peptide != peptide[test_index]) & (affibody != affibody[test_index])
            train = indices[train_mask]
            guarded = indices[(~train_mask) & (indices != test_index)]
            records.append(
                {
                    "train": train,
                    "test": test,
                    "guarded": guarded,
                    "outer_fold": "cell{:03d}".format(test_index),
                    "repeat": 0,
                }
            )
        return records
    raise ValueError("unknown split scheme {}".format(scheme))


def calculate_site_coverage_events(frame, library, scheme, seed, random_repeats):
    """Describe designed-position categories absent from each outer train fold."""
    site_columns = feature_columns("site_residue_additive", library)
    peptide_columns = [column for column in site_columns if column.startswith("pep_")]
    affibody_columns = [column for column in site_columns if column.startswith("aff_")]
    events = []
    for split in make_outer_splits(frame, scheme, seed, random_repeats):
        train_frame = frame.iloc[split["train"]]
        for row_index in split["test"]:
            row = frame.iloc[row_index]
            peptide_unseen = sum(
                row[column] not in set(train_frame[column]) for column in peptide_columns
            )
            affibody_unseen = sum(
                row[column] not in set(train_frame[column]) for column in affibody_columns
            )
            unseen = int(peptide_unseen + affibody_unseen)
            events.append(
                {
                    "library": library,
                    "split_scheme": scheme,
                    "seed": int(seed),
                    "repeat": int(split["repeat"]),
                    "unseen_site_count": unseen,
                    "unseen_peptide_site_count": int(peptide_unseen),
                    "unseen_affibody_site_count": int(affibody_unseen),
                    "any_unseen_site": int(unseen > 0),
                }
            )
    return events


def summarize_site_coverage(events):
    event_frame = pd.DataFrame(events)
    by_repeat_rows = []
    group_columns = ["library", "split_scheme", "seed", "repeat"]
    for group_values, subset in event_frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "n": int(len(subset)),
                "any_unseen_count": int(subset["any_unseen_site"].sum()),
                "any_unseen_fraction": float(subset["any_unseen_site"].mean()),
                "mean_unseen_site_count": float(subset["unseen_site_count"].mean()),
                "max_unseen_site_count": int(subset["unseen_site_count"].max()),
                "any_unseen_peptide_count": int(
                    subset["unseen_peptide_site_count"].gt(0).sum()
                ),
                "any_unseen_affibody_count": int(
                    subset["unseen_affibody_site_count"].gt(0).sum()
                ),
            }
        )
        by_repeat_rows.append(row)
    by_repeat = pd.DataFrame(by_repeat_rows).sort_values(group_columns).reset_index(drop=True)

    summary_rows = []
    summary_groups = ["library", "split_scheme", "seed"]
    value_columns = [
        "any_unseen_count",
        "any_unseen_fraction",
        "mean_unseen_site_count",
        "max_unseen_site_count",
        "any_unseen_peptide_count",
        "any_unseen_affibody_count",
    ]
    for group_values, subset in by_repeat.groupby(summary_groups, sort=True):
        row = dict(zip(summary_groups, group_values))
        _require(subset["n"].nunique() == 1, "coverage repeats have different row counts")
        row["n"] = int(subset["n"].iloc[0])
        row["repeats"] = int(len(subset))
        for column in value_columns:
            values = subset[column].to_numpy(dtype=float)
            row[column] = float(np.mean(values))
            row[column + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(summary_groups).reset_index(drop=True)
    return by_repeat, summary


def _balanced_assignment(values, folds, seed):
    unique = np.asarray(sorted(set(values)), dtype=str)
    rng = np.random.RandomState(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    mapping = {value: index % folds for index, value in enumerate(shuffled)}
    return np.asarray([mapping[value] for value in values], dtype=int)


def _audit_inner_splits(frame, splits, task, scheme, assignment_attempt=0):
    indices = np.arange(len(frame), dtype=int)
    binder = frame["target_binder"].to_numpy(dtype=int)
    validation_coverage = np.zeros(len(frame), dtype=int)
    for train, test in splits:
        _require(len(train) > 0 and len(test) > 0, "empty inner train or validation fold")
        _require(not set(train).intersection(test), "inner train/validation overlap")
        validation_coverage[test] += 1
        if task == "classification":
            _require(
                len(np.unique(binder[train])) == 2,
                "one-class inner classification training fold",
            )
        if scheme == "double_cold":
            train_frame = frame.iloc[train]
            test_frame = frame.iloc[test]
            _require(
                set(train_frame["peptide_design_code"]).isdisjoint(
                    test_frame["peptide_design_code"]
                ),
                "double-cold inner peptide leakage",
            )
            _require(
                set(train_frame["affibody_design_code"]).isdisjoint(
                    test_frame["affibody_design_code"]
                ),
                "double-cold inner Affibody leakage",
            )
    _require(
        bool(np.array_equal(validation_coverage, np.ones(len(indices), dtype=int))),
        "inner validation folds must cover every outer-training pair exactly once",
    )
    return {
        "scheme": scheme,
        "task": task,
        "folds": int(len(splits)),
        "validation_events": int(validation_coverage.sum()),
        "validation_unique_pairs": int(np.sum(validation_coverage > 0)),
        "validation_min_count": int(validation_coverage.min()),
        "validation_max_count": int(validation_coverage.max()),
        "assignment_attempt": int(assignment_attempt),
    }


def make_inner_splits(frame, scheme, seed, task):
    indices = np.arange(len(frame), dtype=int)
    binder = frame["target_binder"].to_numpy(dtype=int)
    if scheme == "random_pair":
        smallest_class = int(np.bincount(binder).min())
        folds = max(2, min(5, smallest_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(indices, binder))
        return splits, _audit_inner_splits(frame, splits, task, scheme)

    if scheme in ("peptide_cold", "affibody_cold"):
        group_column = "peptide_design_code" if scheme == "peptide_cold" else "affibody_design_code"
        groups = frame[group_column].to_numpy(dtype=str)
        splits = list(LeaveOneGroupOut().split(indices, binder, groups=groups))
        return splits, _audit_inner_splits(frame, splits, task, scheme)

    if scheme == "double_cold":
        peptide = frame["peptide_design_code"].to_numpy(dtype=str)
        affibody = frame["affibody_design_code"].to_numpy(dtype=str)
        # Select the complete block assignment atomically. Requiring two-class
        # training folds makes this same deterministic assignment usable by
        # both regression and classification; no individual block is dropped.
        max_attempts = 1000
        for attempt in range(max_attempts):
            peptide_seed = int(seed + attempt * 1000003)
            affibody_seed = int(seed + 104729 + attempt * 1000033)
            peptide_fold = _balanced_assignment(peptide, 3, peptide_seed)
            affibody_fold = _balanced_assignment(affibody, 3, affibody_seed)
            splits = []
            # The nine blocks partition every outer-training pair exactly once
            # while excluding its peptide and Affibody groups from that fold's
            # inner training set.
            for peptide_block in range(3):
                for affibody_block in range(3):
                    test_mask = (peptide_fold == peptide_block) & (
                        affibody_fold == affibody_block
                    )
                    train_mask = (peptide_fold != peptide_block) & (
                        affibody_fold != affibody_block
                    )
                    splits.append((indices[train_mask], indices[test_mask]))
            if any(len(np.unique(binder[train])) != 2 for train, _ in splits):
                continue
            audit = _audit_inner_splits(frame, splits, task, scheme, attempt)
            _require(len(splits) == 9, "double-cold inner CV must contain nine blocks")
            return splits, audit
        raise ValueError(
            "could not construct complete two-class double-cold classification inner CV"
        )
    raise ValueError("unknown split scheme {}".format(scheme))


def _predict_score(estimator, task, matrix):
    if task == "regression":
        return np.asarray(estimator.predict(matrix), dtype=float)
    probabilities = estimator.predict_proba(matrix)
    class_to_column = {int(label): index for index, label in enumerate(estimator.classes_)}
    _require(1 in class_to_column, "classifier did not learn positive class")
    return np.asarray(probabilities[:, class_to_column[1]], dtype=float)


def select_regularization(
    outer_train,
    model_name,
    task,
    library,
    scheme,
    grid,
    seed,
):
    if model_name == "mean_prior":
        return None, {"rule": "none", "candidates": []}
    matrix = feature_matrix(outer_train, model_name, library)
    target_column = "target_retention" if task == "regression" else "target_binder"
    target = outer_train[target_column].to_numpy()
    splits, split_audit = make_inner_splits(outer_train, scheme, seed, task)
    candidate_records = []
    for parameter in grid:
        losses = []
        for inner_train, inner_test in splits:
            estimator = make_estimator(model_name, task, library, parameter, seed)
            estimator.fit(matrix[inner_train], target[inner_train])
            prediction = _predict_score(estimator, task, matrix[inner_test])
            if task == "regression":
                loss = mean_absolute_error(target[inner_test], prediction)
            else:
                loss = log_loss(target[inner_test], prediction, labels=[0, 1])
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        if len(losses) > 1:
            standard_error = float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
        else:
            standard_error = 0.0
        candidate_records.append(
            {
                "parameter": float(parameter),
                "mean_loss": mean_loss,
                "standard_error": standard_error,
                "inner_folds": len(losses),
            }
        )

    best = min(candidate_records, key=lambda record: record["mean_loss"])
    acceptable_limit = best["mean_loss"] + best["standard_error"]
    acceptable = [
        record for record in candidate_records if record["mean_loss"] <= acceptable_limit
    ]
    if task == "regression":
        chosen = max(acceptable, key=lambda record: record["parameter"])
        rule = "one_standard_error_largest_alpha"
    else:
        chosen = min(acceptable, key=lambda record: record["parameter"])
        rule = "one_standard_error_smallest_C"
    audit = {
        "rule": rule,
        "best_parameter": best["parameter"],
        "best_mean_loss": best["mean_loss"],
        "best_standard_error": best["standard_error"],
        "acceptable_limit": acceptable_limit,
        "chosen_parameter": chosen["parameter"],
        "inner_split": split_audit,
        "candidates": candidate_records,
    }
    return chosen["parameter"], audit


def run_one_configuration(
    frame,
    library,
    scheme,
    task,
    model_name,
    seed,
    random_repeats,
    parameter_grid,
):
    matrix = feature_matrix(frame, model_name, library)
    target_column = "target_retention" if task == "regression" else "target_binder"
    target = frame[target_column].to_numpy()
    outer_splits = make_outer_splits(frame, scheme, seed, random_repeats)
    events = []
    tuning_rows = []
    for split_number, split in enumerate(outer_splits):
        train = split["train"]
        test = split["test"]
        _require(len(np.unique(frame.iloc[train]["target_binder"])) == 2, "one-class outer train")
        # Share inner folds across tasks and non-reference models for a given
        # outer split. The estimators themselves remain fit independently.
        fold_seed = int(seed + split_number * 1009)
        parameter, tuning = select_regularization(
            frame.iloc[train].reset_index(drop=True),
            model_name,
            task,
            library,
            scheme,
            parameter_grid,
            fold_seed,
        )
        estimator = make_estimator(model_name, task, library, parameter, fold_seed)
        estimator.fit(matrix[train], target[train])
        prediction = _predict_score(estimator, task, matrix[test])
        _require(bool(np.isfinite(prediction).all()), "non-finite held-out prediction")
        for local_index, row_index in enumerate(test):
            row = frame.iloc[row_index]
            events.append(
                {
                    "pair_uid": row["pair_uid"],
                    "peptide_uid": row["peptide_uid"],
                    "affibody_uid": row["affibody_uid"],
                    "library": library,
                    "task": task,
                    "model": model_name,
                    "feature_set": model_name,
                    "split_scheme": scheme,
                    "outer_fold": split["outer_fold"],
                    "repeat": split["repeat"],
                    "seed": int(seed),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_guarded": int(len(split["guarded"])),
                    "y_true": float(target[row_index]),
                    "y_score": float(prediction[local_index]),
                    "chosen_parameter": "" if parameter is None else float(parameter),
                }
            )
        tuning_rows.append(
            {
                "library": library,
                "task": task,
                "model": model_name,
                "split_scheme": scheme,
                "outer_fold": split["outer_fold"],
                "repeat": split["repeat"],
                "seed": int(seed),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_guarded": int(len(split["guarded"])),
                "chosen_parameter": "" if parameter is None else float(parameter),
                "tuning_audit_json": json.dumps(tuning, sort_keys=True, separators=(",", ":")),
            }
        )
    return events, tuning_rows


def aggregate_prediction_events(events):
    event_frame = pd.DataFrame(events)
    keys = [
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "library",
        "task",
        "model",
        "feature_set",
        "split_scheme",
        "seed",
    ]
    aggregate = (
        event_frame.groupby(keys, as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_score=("y_score", "mean"),
            prediction_count=("y_score", "size"),
            n_train_min=("n_train", "min"),
            n_train_max=("n_train", "max"),
            n_test_min=("n_test", "min"),
            n_test_max=("n_test", "max"),
            n_guarded_min=("n_guarded", "min"),
            n_guarded_max=("n_guarded", "max"),
        )
        .sort_values(["library", "task", "split_scheme", "model", "pair_uid"])
        .reset_index(drop=True)
    )
    return event_frame, aggregate


def _safe_correlation(function, observed, predicted):
    if len(np.unique(observed)) < 2 or len(np.unique(predicted)) < 2:
        return np.nan
    return float(function(observed, predicted)[0])


def regression_metrics(observed, predicted):
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "pearson": _safe_correlation(pearsonr, observed, predicted),
        "spearman": _safe_correlation(spearmanr, observed, predicted),
        "predictions_below_0": int(np.sum(predicted < 0.0)),
        "predictions_above_100": int(np.sum(predicted > 100.0)),
    }


def classification_metrics(observed, probability):
    observed = np.asarray(observed, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= 0.5).astype(int)
    negative = observed == 0
    specificity = float(np.mean(predicted[negative] == 0)) if bool(negative.any()) else np.nan
    return {
        "prevalence": float(np.mean(observed)),
        "average_precision": float(average_precision_score(observed, probability)),
        "roc_auc": float(roc_auc_score(observed, probability)),
        "mcc": float(matthews_corrcoef(observed, predicted)),
        "accuracy": float(accuracy_score(observed, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(observed, predicted)),
        "precision": float(precision_score(observed, predicted, zero_division=0)),
        "recall": float(recall_score(observed, predicted, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(observed, predicted, zero_division=0)),
        "brier": float(brier_score_loss(observed, probability)),
        "log_loss": float(log_loss(observed, probability, labels=[0, 1])),
    }


def calculate_metrics_by_repeat(event_frame):
    rows = []
    group_columns = [
        "library",
        "task",
        "model",
        "feature_set",
        "split_scheme",
        "seed",
        "repeat",
    ]
    for group_values, subset in event_frame.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, group_values))
        _require(
            subset["pair_uid"].nunique() == len(subset),
            "duplicate held-out pair inside one evaluation repeat",
        )
        observed = subset["y_true"].to_numpy()
        predicted = subset["y_score"].to_numpy()
        row["n"] = int(len(subset))
        if row["task"] == "regression":
            row.update(regression_metrics(observed.astype(float), predicted.astype(float)))
        else:
            row.update(classification_metrics(observed.astype(int), predicted.astype(float)))
        rows.append(row)
    metrics = pd.DataFrame(rows)

    metrics["delta_mae_vs_mean_prior"] = np.nan
    metrics["delta_average_precision_vs_prevalence"] = np.nan
    metrics["delta_average_precision_vs_cv_prior"] = np.nan
    metrics["delta_mcc_vs_cv_prior"] = np.nan
    for index, row in metrics.iterrows():
        same = (
            metrics["library"].eq(row["library"])
            & metrics["task"].eq(row["task"])
            & metrics["split_scheme"].eq(row["split_scheme"])
            & metrics["repeat"].eq(row["repeat"])
            & metrics["model"].eq("mean_prior")
        )
        reference_rows = metrics.loc[same]
        _require(len(reference_rows) == 1, "missing or duplicate mean/prior metric row")
        reference = reference_rows.iloc[0]
        if row["task"] == "regression":
            metrics.loc[index, "delta_mae_vs_mean_prior"] = float(
                reference["mae"] - row["mae"]
            )
        else:
            # Evaluation-set prevalence is the correct no-skill AUPRC reference.
            # Fold-varying fitted priors can acquire artificial pooled ranking in
            # grouped CV, so their AP is retained only as a secondary diagnostic.
            metrics.loc[index, "delta_average_precision_vs_prevalence"] = float(
                row["average_precision"] - row["prevalence"]
            )
            metrics.loc[index, "delta_average_precision_vs_cv_prior"] = float(
                row["average_precision"] - reference["average_precision"]
            )
            metrics.loc[index, "delta_mcc_vs_cv_prior"] = float(
                row["mcc"] - reference["mcc"]
            )
    return metrics.sort_values(
        ["library", "task", "split_scheme", "model", "repeat"]
    ).reset_index(drop=True)


def summarize_metrics(metrics_by_repeat):
    group_columns = ["library", "task", "model", "feature_set", "split_scheme", "seed"]
    excluded = set(group_columns + ["repeat", "n"])
    value_columns = [column for column in metrics_by_repeat.columns if column not in excluded]
    rows = []
    for group_values, subset in metrics_by_repeat.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, group_values))
        _require(subset["n"].nunique() == 1, "evaluation repeats have different row counts")
        row["n"] = int(subset["n"].iloc[0])
        row["repeats"] = int(len(subset))
        for column in value_columns:
            values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[column] = np.nan
                row[column + "_sd"] = np.nan
            else:
                row[column] = float(np.mean(values))
                row[column + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["library", "task", "split_scheme", "model"]
    ).reset_index(drop=True)


def git_metadata(workdir):
    def command(args):
        result = subprocess.run(
            args,
            cwd=str(workdir),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return result.stdout.strip()

    return {
        "head": command(["git", "rev-parse", "HEAD"]),
        "status_short": command(["git", "status", "--short"]),
    }


def write_summary(path, metrics, coverage, manifest):
    def formatted(row, column):
        value = float(row[column])
        if int(row["repeats"]) > 1:
            return "{:.4f} +/- {:.4f}".format(value, float(row[column + "_sd"]))
        return "{:.4f}".format(value)

    lines = [
        "# Code-only strong-label baseline summary",
        "",
        "Run ID: `{}`".format(manifest["run_id"]),
        "",
        "These are held-out predictions from the measured retention table. The short codes were",
        "treated as designed-position categories, never as complete protein chains.",
        "",
        "## Regression",
        "",
        "| Library | Split | Model | MAE | Spearman | Pearson | R2 | Delta MAE vs mean/prior |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    regression = metrics.loc[metrics["task"].eq("regression")]
    for _, row in regression.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["library"],
                row["split_scheme"],
                row["model"],
                formatted(row, "mae"),
                formatted(row, "spearman"),
                formatted(row, "pearson"),
                formatted(row, "r2"),
                formatted(row, "delta_mae_vs_mean_prior"),
            )
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "| Library | Split | Model | Brier | Log loss | AUPRC | AUROC | MCC | Delta AUPRC vs prevalence |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    classification = metrics.loc[metrics["task"].eq("classification")]
    for _, row in classification.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["library"],
                row["split_scheme"],
                row["model"],
                formatted(row, "brier"),
                formatted(row, "log_loss"),
                formatted(row, "average_precision"),
                formatted(row, "roc_auc"),
                formatted(row, "mcc"),
                formatted(row, "delta_average_precision_vs_prevalence"),
            )
        )
    lines.extend(
        [
            "",
            "## Designed-residue coverage",
            "",
            "A held-out prediction is marked unseen when at least one residue-at-position",
            "category in that test pair never occurs in the corresponding outer training fold.",
            "For the fixed-alphabet site model, such coefficients remain at zero.",
            "",
            "| Library | Split | Test pairs/events | Any unseen site | Fraction | Mean unseen sites |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in coverage.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["library"],
                row["split_scheme"],
                int(row["n"]),
                formatted(row, "any_unseen_count"),
                formatted(row, "any_unseen_fraction"),
                formatted(row, "mean_unseen_site_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Random-pair results are interpolation diagnostics, not the main generalization claim.",
            "- Cold-split metrics are pooled over held-out predictions; group folds can contain one class.",
            "- Lead cold conclusions with pointwise MAE and Brier/log loss. Pooled cold rank metrics",
            "  compare predictions from different fitted models and are exploratory.",
            "- A zero cold-split SD means one complete OOF traversal, not zero uncertainty.",
            "- Cold extrapolation can involve residue-at-position categories absent from training.",
            "- Classification reuses the same retention measurements thresholded at 75.",
            "- Ridge predictions are not clipped to [0,100]; out-of-range counts are in metrics.csv.",
            "- These models apply only to the current designed positions and assay context.",
            "- Full-chain MINT comparison remains pending full construct mappings.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--c-grid", type=float, nargs="+", default=DEFAULT_C_GRID)
    return parser.parse_args(argv)


def run(args):
    start = time.time()
    _require(args.random_repeats >= 1, "random repeats must be positive")
    _require(all(value > 0 for value in args.alpha_grid), "ridge alpha values must be positive")
    _require(all(value > 0 for value in args.c_grid), "logistic C values must be positive")
    _require(args.input.is_file(), "input file does not exist")
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    args.output_dir = validate_private_output_path(args.output_dir, repo_root)
    _require(not args.output_dir.exists(), "output directory already exists; refusing overwrite")

    source_sha256 = sha256_file(args.input)
    script_sha256 = sha256_file(script_path)
    configuration = {
        "seed": int(args.seed),
        "random_repeats": int(args.random_repeats),
        "alpha_grid": [float(value) for value in args.alpha_grid],
        "c_grid": [float(value) for value in args.c_grid],
        "models": list(MODEL_NAMES),
        "split_schemes": list(SPLIT_SCHEMES),
        "tasks": list(TASKS),
        "binder_threshold": 75.0,
        "automatic_na_parsing": False,
    }
    fingerprint = run_fingerprint(source_sha256, script_sha256, configuration)
    run_id = "code_only_strong_v1_{}".format(fingerprint)

    source = load_retention_table(args.input)

    all_events = []
    all_tuning = []
    all_coverage = []
    for library in sorted(LIBRARY_SPECS):
        library_frame = supervised_library_frame(source, library)
        for scheme in SPLIT_SCHEMES:
            all_coverage.extend(
                calculate_site_coverage_events(
                    library_frame, library, scheme, args.seed, args.random_repeats
                )
            )
            for task in TASKS:
                grid = args.alpha_grid if task == "regression" else args.c_grid
                for model_name in MODEL_NAMES:
                    events, tuning = run_one_configuration(
                        library_frame,
                        library,
                        scheme,
                        task,
                        model_name,
                        args.seed,
                        args.random_repeats,
                        grid,
                    )
                    all_events.extend(events)
                    all_tuning.extend(tuning)
                    print(
                        "completed {} {} {} {}".format(library, scheme, task, model_name),
                        flush=True,
                    )

    event_frame, aggregate = aggregate_prediction_events(all_events)
    metrics_by_repeat = calculate_metrics_by_repeat(event_frame)
    metrics = summarize_metrics(metrics_by_repeat)
    coverage_by_repeat, coverage = summarize_site_coverage(all_coverage)
    for table in (event_frame, aggregate, metrics_by_repeat, metrics):
        table.insert(0, "run_id", run_id)
    for table in (coverage_by_repeat, coverage):
        table.insert(0, "run_id", run_id)
    tuning_frame = pd.DataFrame(all_tuning)
    tuning_frame.insert(0, "run_id", run_id)

    _require(sha256_file(args.input) == source_sha256, "source data changed during the run")
    _require(sha256_file(script_path) == script_sha256, "runner code changed during the run")
    manifest = {
        "run_id": run_id,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": None,
        "source": {
            "path": str(args.input.resolve()),
            "sha256": source_sha256,
            "designed_rows": int(len(source)),
            "measured_rows": int(source["target_retention"].notna().sum()),
            "missing_rows": int(source["target_retention"].isna().sum()),
            "library_measured_rows": {
                library: int(
                    (
                        source["library"].eq(library)
                        & source["target_retention"].notna()
                    ).sum()
                )
                for library in sorted(LIBRARY_SPECS)
            },
        },
        "code": {
            "script_path": str(script_path),
            "script_sha256": script_sha256,
            "git": git_metadata(repo_root),
        },
        "configuration": configuration,
        "privacy": {
            "output_path_policy": "Git-ignored child of repository private_data",
            "directory_mode": "0700",
            "file_mode": "0600",
            "row_identifiers": "pseudonymous deterministic hashes; not anonymous",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            "prediction_events": "prediction_events.csv",
            "predictions": "predictions.csv",
            "metrics": "metrics.csv",
            "metrics_by_repeat": "metrics_by_repeat.csv",
            "hyperparameters": "hyperparameters.csv",
            "feature_coverage": "feature_coverage.csv",
            "feature_coverage_by_repeat": "feature_coverage_by_repeat.csv",
            "summary": "run_summary.md",
        },
    }

    _require(not args.output_dir.exists(), "output directory appeared during run; refusing overwrite")
    args.output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    args.output_dir.chmod(0o700)
    event_frame.to_csv(args.output_dir / "prediction_events.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    aggregate.to_csv(args.output_dir / "predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metrics_by_repeat.to_csv(
        args.output_dir / "metrics_by_repeat.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    tuning_frame.to_csv(args.output_dir / "hyperparameters.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(args.output_dir / "feature_coverage.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage_by_repeat.to_csv(
        args.output_dir / "feature_coverage_by_repeat.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    write_summary(args.output_dir / "run_summary.md", metrics, coverage, manifest)
    manifest["artifact_sha256"] = {
        filename: sha256_file(args.output_dir / filename)
        for filename in sorted(manifest["outputs"].values())
    }
    manifest["elapsed_seconds"] = round(time.time() - start, 6)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for artifact in args.output_dir.iterdir():
        _require(artifact.is_file(), "unexpected non-file output artifact")
        artifact.chmod(0o600)
    args.output_dir.chmod(0o700)
    print("wrote private artifacts to {}".format(args.output_dir), flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    previous_umask = os.umask(0o077)
    try:
        return run(args)
    finally:
        os.umask(previous_umask)


if __name__ == "__main__":
    sys.exit(main())
