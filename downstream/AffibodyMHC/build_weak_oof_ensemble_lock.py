#!/usr/bin/env python3
"""Select and lock Affibody score ensembles from weak-label OOF predictions.

This command is deliberately library-agnostic.  It accepts a JSON contract
describing eligible component models, aligns their double-cold out-of-fold
(OOF) predictions, compares every non-empty equal-logit combination, and can
also cross-fit a non-negative logistic stacker.  Model and cutoff selection use
only the binary weak labels carried by the OOF and fold-membership artifacts.

Direct-retention inputs are not accepted.  Paths or table columns that appear
to contain retention outcomes are rejected before their contents are loaded.
The command is intended for LibA once its component OOF artifacts have been
created; it is equally usable for another library whose contract satisfies the
same validation rules.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA_VERSION = "affibody-weak-oof-ensemble-config-v1"
ARTIFACT_SCHEMA_VERSION = "affibody-weak-oof-ensemble-v1"
LOCK_SCHEMA_VERSION = "affibody-weak-oof-score-lock-v1"
PROBABILITY_CLIP = 1e-7
DEFAULT_STACK_L2_GRID = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
STACK_MAX_ITER = 2_000
STACK_FTOL = 1e-12
SELECTION_TOLERANCE = 1e-12
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
RETENTION_TOKEN = re.compile(
    r"(?:retention|target[_-]?binder|binder[_-]?label[_-]?ge[_-]?75|"
    r"measurement[_-]?missing)",
    re.IGNORECASE,
)
ALLOWED_NEGATIVE_RETENTION_AUDIT_FIELDS = {
    "retention_labels_allowed",
    "retention_labels_read",
    "retention_labels_used",
    "retention_outcomes_read",
    "direct_retention_used_for_selection",
}
FORBIDDEN_AUDIT_FIELDS = {
    "evidence_positive_control",
    "retrospective_metrics",
    "direct_retention_audit",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        display = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        display = str(path)
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _write_json_exclusive(path: Path, payload: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _write_csv_exclusive(
    path: Path, frame: pd.DataFrame, *, compression: str | None = None
) -> None:
    _require(not Path(path).exists(), f"refusing to overwrite {path}")
    frame.to_csv(
        path,
        index=False,
        compression=compression,
        lineterminator="\n",
        float_format="%.12g",
    )
    os.chmod(path, 0o600)


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _reject_retention_path(path: Path, purpose: str) -> None:
    _require(
        RETENTION_TOKEN.search(str(path)) is None,
        f"{purpose} path appears outcome-bearing and is forbidden: {path}",
    )


def _reject_retention_columns(columns: Iterable[object], purpose: str) -> None:
    forbidden = [str(value) for value in columns if RETENTION_TOKEN.search(str(value))]
    _require(not forbidden, f"{purpose} contains forbidden outcome columns: {forbidden}")


def _read_csv_outcome_safe(path: Path, purpose: str, **kwargs: Any) -> pd.DataFrame:
    """Reject outcome-looking paths/headers before loading a complete table."""

    path = Path(path).resolve()
    _reject_retention_path(path, purpose)
    _require(path.is_file(), f"missing {purpose}: {path}")
    header = pd.read_csv(path, nrows=0)
    _reject_retention_columns(header.columns, purpose)
    return pd.read_csv(path, **kwargs)


def _validate_exact_keys(value: Mapping[str, Any], allowed: set[str], purpose: str) -> None:
    extra = sorted(set(value).difference(allowed))
    _require(not extra, f"unexpected {purpose} fields: {extra}")


def _walk_mapping(
    value: object, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], object]]:
    """Yield nested mapping fields, including mappings contained in lists."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key_path = path + (str(raw_key),)
            yield key_path, child
            yield from _walk_mapping(child, key_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_mapping(child, path + (f"[{index}]",))


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _reject_retention_path(path, "config")
    _require(path.is_file(), f"missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _require(isinstance(config, dict), "config must be a JSON object")
    _validate_exact_keys(
        config,
        {
            "schema_version",
            "library",
            "fold_membership",
            "members",
            "common_row_policy",
            "expected_folds",
            "expected_common_oof_rows",
            "expected_common_oof_positive",
            "fit_nonnegative_stacker",
            "stack_l2_grid",
            "max_candidates_per_peptide",
        },
        "config",
    )
    _require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "config schema changed")
    library = str(config.get("library", "")).strip()
    _require(bool(library), "library is required")
    _require(RETENTION_TOKEN.search(library) is None, "invalid library name")
    _require(bool(config.get("fold_membership")), "fold_membership is required")
    policy = str(config.get("common_row_policy", ""))
    _require(policy in {"require_exact", "intersection"}, "invalid common_row_policy")
    expected_folds = int(config.get("expected_folds", 0))
    _require(expected_folds >= 3, "at least three OOF folds are required")
    raw_members = config.get("members")
    _require(isinstance(raw_members, list) and len(raw_members) >= 2, "at least two members are required")
    allowed_member_fields = {
        "name",
        "oof_path",
        "audit_path",
        "probability_column",
        "row_id_column",
        "fold_column",
        "weak_label_column",
        "model_column",
        "model_value",
        "inference_logit_column",
    }
    names: list[str] = []
    for index, member in enumerate(raw_members):
        _require(isinstance(member, dict), f"member {index} must be an object")
        _validate_exact_keys(member, allowed_member_fields, f"member {index}")
        name = str(member.get("name", ""))
        _require(bool(SAFE_NAME.fullmatch(name)), f"unsafe member name: {name!r}")
        _require("__" not in name, f"member name may not contain '__': {name}")
        names.append(name)
        for required in ("oof_path", "audit_path", "inference_logit_column"):
            _require(bool(member.get(required)), f"{name} lacks {required}")
        model_column = member.get("model_column")
        model_value = member.get("model_value")
        _require(
            (model_column is None) == (model_value is None),
            f"{name} must provide model_column and model_value together",
        )
    _require(len(set(names)) == len(names), "duplicate member name")
    l2_grid = config.get("stack_l2_grid", list(DEFAULT_STACK_L2_GRID))
    _require(isinstance(l2_grid, list) and l2_grid, "stack_l2_grid must be nonempty")
    l2_values = [float(value) for value in l2_grid]
    _require(all(math.isfinite(value) and value >= 0.0 for value in l2_values), "bad stack L2")
    _require(len(set(l2_values)) == len(l2_values), "duplicate stack L2")
    config["stack_l2_grid"] = l2_values
    config["fit_nonnegative_stacker"] = bool(config.get("fit_nonnegative_stacker", False))
    if config.get("expected_common_oof_rows") is not None:
        _require(int(config["expected_common_oof_rows"]) > 0, "expected rows must be positive")
    if config.get("expected_common_oof_positive") is not None:
        _require(int(config["expected_common_oof_positive"]) > 0, "expected positives must be positive")
    if config.get("max_candidates_per_peptide") is not None:
        _require(int(config["max_candidates_per_peptide"]) > 0, "candidate cap must be positive")
    return config


def clipped_logit(probability: Sequence[float]) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    _require(value.ndim == 1 and len(value) > 0, "probability must be a nonempty vector")
    _require(bool(np.isfinite(value).all()), "probability contains non-finite values")
    _require(bool(((value >= 0.0) & (value <= 1.0)).all()), "probability is outside [0,1]")
    value = np.clip(value, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.log(value) - np.log1p(-value)


def sigmoid(logit: Sequence[float]) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def mean_logit_score(score_columns: Sequence[Sequence[float]]) -> np.ndarray:
    _require(score_columns, "mean-logit ensemble has no members")
    logits = [clipped_logit(values) for values in score_columns]
    _require(len({len(values) for values in logits}) == 1, "member lengths differ")
    return sigmoid(np.mean(np.stack(logits, axis=1), axis=1))


def _membership_id_column(frame: pd.DataFrame) -> str:
    choices = [name for name in ("row_id", "pair_uid") if name in frame.columns]
    _require(len(choices) == 1, "fold membership needs exactly one of row_id or pair_uid")
    return choices[0]


def load_and_validate_membership(
    path: Path, library: str, expected_folds: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv_outcome_safe(path, "fold membership", dtype=str)
    id_column = _membership_id_column(frame)
    required = {id_column, "fold", "role", "weak_label", "chain1_sha256", "chain2_sha256"}
    _require(required.issubset(frame.columns), f"fold membership lacks {sorted(required.difference(frame.columns))}")
    if "library" in frame.columns:
        frame = frame.loc[frame["library"].astype(str).eq(library)].copy()
        _require(not frame.empty, f"fold membership has no rows for {library}")
    frame = frame.rename(columns={id_column: "row_id"})
    frame["row_id"] = frame["row_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="raise").astype(int)
    _require(set(frame["weak_label"].unique()).issubset({0, 1}), "membership labels are not binary")
    _require(set(frame["role"].astype(str)) == {"train", "guard", "validation"}, "membership roles changed")
    folds = tuple(sorted(frame["fold"].unique().tolist()))
    _require(folds == tuple(range(expected_folds)), f"expected folds 0..{expected_folds - 1}, found {folds}")
    _require(not bool(frame.duplicated(["row_id", "fold"]).any()), "duplicate membership row/fold")

    validation_blocks = []
    fold_audit = []
    for fold in folds:
        current = frame.loc[frame["fold"].eq(fold)]
        train = current.loc[current["role"].eq("train")]
        guard = current.loc[current["role"].eq("guard")]
        validation = current.loc[current["role"].eq("validation")]
        _require(not train.empty and not validation.empty, f"fold {fold} has empty train/validation")
        peptide_overlap = set(train["chain1_sha256"]).intersection(validation["chain1_sha256"])
        affibody_overlap = set(train["chain2_sha256"]).intersection(validation["chain2_sha256"])
        _require(not peptide_overlap, f"fold {fold} is not peptide-cold")
        _require(not affibody_overlap, f"fold {fold} is not Affibody-cold")
        block = validation[
            ["row_id", "fold", "weak_label", "chain1_sha256", "chain2_sha256"]
        ].copy()
        block = block.rename(
            columns={"chain1_sha256": "peptide_id", "chain2_sha256": "affibody_id"}
        )
        validation_blocks.append(block)
        fold_audit.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "guard_rows": int(len(guard)),
                "validation_rows": int(len(validation)),
                "validation_positive": int(validation["weak_label"].sum()),
                "train_validation_peptide_overlap": 0,
                "train_validation_affibody_overlap": 0,
            }
        )
    validation = pd.concat(validation_blocks, ignore_index=True)
    _require(not bool(validation["row_id"].duplicated().any()), "a row is validation in multiple folds")
    return validation, {"folds": fold_audit, "validation_rows": int(len(validation))}


def _load_audit(path: Path, member_name: str, library: str) -> dict[str, Any]:
    path = Path(path).resolve()
    _reject_retention_path(path, f"{member_name} audit")
    _require(path.is_file(), f"missing {member_name} audit: {path}")
    with path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    _require(isinstance(audit, dict), f"{member_name} audit must be an object")
    for key_path, value in _walk_mapping(audit):
        key = key_path[-1]
        _require(
            key not in FORBIDDEN_AUDIT_FIELDS,
            f"{member_name} audit contains outcome-bearing field {'.'.join(key_path)}",
        )
        if RETENTION_TOKEN.search(key):
            _require(
                key in ALLOWED_NEGATIVE_RETENTION_AUDIT_FIELDS and value is False,
                f"{member_name} audit contains unsafe retention field {'.'.join(key_path)}",
            )
    _require(audit.get("retention_labels_read") is False, f"{member_name} audit does not forbid retention")
    if audit.get("library") is not None:
        _require(str(audit["library"]) == library, f"{member_name} audit library changed")
    return audit


def load_member_oof(
    config_path: Path,
    member: Mapping[str, Any],
    library: str,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    name = str(member["name"])
    oof_path = _resolve_from_config(config_path, str(member["oof_path"]))
    audit_path = _resolve_from_config(config_path, str(member["audit_path"]))
    audit = _load_audit(audit_path, name, library)
    frame = _read_csv_outcome_safe(oof_path, f"{name} OOF", dtype=str)
    row_column = str(member.get("row_id_column", "row_id"))
    fold_column = str(member.get("fold_column", "fold"))
    label_column = str(member.get("weak_label_column", "weak_label"))
    probability_column = str(member.get("probability_column", "probability"))
    required = {row_column, fold_column, label_column, probability_column}
    model_column = member.get("model_column")
    if model_column is not None:
        required.add(str(model_column))
    _require(required.issubset(frame.columns), f"{name} OOF lacks {sorted(required.difference(frame.columns))}")
    if model_column is not None:
        frame = frame.loc[
            frame[str(model_column)].astype(str).eq(str(member["model_value"]))
        ].copy()
        _require(not frame.empty, f"{name} model filter matched no rows")
    frame = frame[[row_column, fold_column, label_column, probability_column]].rename(
        columns={
            row_column: "row_id",
            fold_column: "fold",
            label_column: "weak_label",
            probability_column: name,
        }
    )
    frame["row_id"] = frame["row_id"].astype(str)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="raise").astype(int)
    frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
    _require(not bool(frame["row_id"].duplicated().any()), f"duplicate {name} OOF row")
    _require(set(frame["weak_label"].unique()).issubset({0, 1}), f"{name} labels are not binary")
    values = frame[name].to_numpy(dtype=float)
    _require(bool(np.isfinite(values).all()), f"{name} has non-finite probabilities")
    _require(bool(((values >= 0.0) & (values <= 1.0)).all()), f"{name} probability outside [0,1]")

    checked = frame.merge(
        validation[["row_id", "fold", "weak_label"]],
        on="row_id",
        how="left",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    _require(not bool(checked["fold_expected"].isna().any()), f"{name} contains non-validation rows")
    _require(bool(checked["fold"].eq(checked["fold_expected"]).all()), f"{name} fold differs from membership")
    _require(
        bool(checked["weak_label"].eq(checked["weak_label_expected"]).all()),
        f"{name} weak label differs from membership",
    )
    return frame, {
        "name": name,
        "rows": int(len(frame)),
        "positive": int(frame["weak_label"].sum()),
        "inference_logit_column": str(member["inference_logit_column"]),
        "oof": _record(oof_path),
        "audit": _record(audit_path),
        "audit_schema_version": audit.get("schema_version"),
        "retention_labels_read": False,
    }


def align_common_oof(
    member_frames: Sequence[pd.DataFrame],
    member_names: Sequence[str],
    validation: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require(len(member_frames) == len(member_names) >= 2, "member frames differ from names")
    row_sets = [set(frame["row_id"]) for frame in member_frames]
    if policy == "require_exact":
        first = row_sets[0]
        _require(all(values == first for values in row_sets[1:]), "member OOF row sets differ")
        common = first
    else:
        common = set.intersection(*row_sets)
    _require(common, "members have no common OOF rows")
    combined: pd.DataFrame | None = None
    key = ["row_id", "fold", "weak_label"]
    for frame, name in zip(member_frames, member_names):
        block = frame.loc[frame["row_id"].isin(common), key + [name]].copy()
        combined = block if combined is None else combined.merge(block, on=key, validate="one_to_one")
    assert combined is not None
    combined = combined.merge(
        validation[["row_id", "peptide_id", "affibody_id"]],
        on="row_id",
        validate="one_to_one",
    )
    combined = combined.sort_values("row_id", kind="stable").reset_index(drop=True)
    _require(len(combined) == len(common), "common OOF merge lost rows")
    _require(set(combined["weak_label"].unique()) == {0, 1}, "common OOF needs both classes")
    _require(combined["fold"].nunique() >= 3, "common OOF must retain at least three folds")
    return combined, {
        "policy": policy,
        "common_rows": int(len(combined)),
        "common_positive": int(combined["weak_label"].sum()),
        "member_rows": {
            name: int(len(frame)) for name, frame in zip(member_names, member_frames)
        },
        "member_rows_dropped": {
            name: int(len(frame) - len(combined))
            for name, frame in zip(member_names, member_frames)
        },
    }


def macro_within_peptide_metrics(
    peptide_id: Sequence[str], labels: Sequence[int], scores: Sequence[float]
) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {
            "peptide_id": np.asarray(peptide_id, dtype=str),
            "weak_label": np.asarray(labels, dtype=int),
            "score": np.asarray(scores, dtype=float),
        }
    )
    rows: list[dict[str, float]] = []
    for _, group in frame.groupby("peptide_id", sort=True):
        observed = group["weak_label"].to_numpy(dtype=int)
        predicted = group["score"].to_numpy(dtype=float)
        if np.unique(observed).size != 2:
            continue
        rows.append(
            {
                "ap": float(average_precision_score(observed, predicted)),
                "auroc": float(roc_auc_score(observed, predicted)),
            }
        )
    _require(rows, "no peptide has both weak-label classes")
    return {
        "within_peptide_evaluable": int(len(rows)),
        "within_peptide_ap": float(np.mean([row["ap"] for row in rows])),
        "within_peptide_auroc": float(np.mean([row["auroc"] for row in rows])),
    }


def score_metrics(frame: pd.DataFrame, scores: Sequence[float]) -> dict[str, float | int]:
    labels = frame["weak_label"].to_numpy(dtype=int)
    probability = np.asarray(scores, dtype=float)
    _require(len(labels) == len(probability), "score length differs from OOF rows")
    _require(set(labels.tolist()) == {0, 1}, "OOF rows need both classes")
    output: dict[str, float | int] = {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "pooled_log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "pooled_ap": float(average_precision_score(labels, probability)),
        "pooled_auroc": float(roc_auc_score(labels, probability)),
    }
    output.update(macro_within_peptide_metrics(frame["peptide_id"], labels, probability))
    return output


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    return mean, np.where(scale > 0.0, scale, 1.0)


def peptide_equal_weights(peptide_ids: Sequence[str]) -> np.ndarray:
    values = pd.Series(np.asarray(peptide_ids, dtype=str))
    count = values.map(values.value_counts(sort=False)).to_numpy(dtype=float)
    weight = 1.0 / count
    return weight * (len(weight) / float(weight.sum()))


def fit_nonnegative_stacker(
    features: np.ndarray,
    labels: Sequence[int],
    l2: float,
    sample_weight: Sequence[float] | None = None,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _require(x.ndim == 2 and x.shape[1] >= 1 and len(x) == len(y), "bad stacker matrix")
    _require(set(y.tolist()) == {0.0, 1.0}, "stacker needs both classes")
    weight = (
        np.ones(len(y), dtype=np.float64)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    _require(weight.shape == y.shape and bool((weight > 0.0).all()), "bad sample weights")
    weight = weight / weight.mean()
    l2 = float(l2)
    _require(l2 >= 0.0, "negative stacker L2")
    prevalence = float(np.average(y, weights=weight))
    prevalence = min(max(prevalence, PROBABILITY_CLIP), 1.0 - PROBABILITY_CLIP)
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[0] = math.log(prevalence / (1.0 - prevalence))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = theta[0]
        coefficient = theta[1:]
        linear = intercept + x.dot(coefficient)
        loss = np.logaddexp(0.0, linear) - y * linear
        probability = sigmoid(linear)
        residual = (probability - y) * weight
        value = float(np.mean(weight * loss) + 0.5 * l2 * np.dot(coefficient, coefficient))
        gradient = np.concatenate(
            (
                np.asarray([np.mean(residual)]),
                x.T.dot(residual) / float(len(y)) + l2 * coefficient,
            )
        )
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None)] + [(0.0, None)] * x.shape[1],
        options={"maxiter": STACK_MAX_ITER, "ftol": STACK_FTOL},
    )
    _require(bool(result.success), f"stacker optimization failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "coefficient": np.asarray(result.x[1:], dtype=np.float64),
        "l2": l2,
        "iterations": int(result.nit),
        "objective": float(result.fun),
    }


def predict_stacker(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    linear = float(model["intercept"]) + np.asarray(features, dtype=np.float64).dot(coefficient)
    return sigmoid(linear)


def _stack_features(frame: pd.DataFrame, members: Sequence[str]) -> np.ndarray:
    return np.column_stack([clipped_logit(frame[name]) for name in members])


def _fit_stacker_partition(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
    members: Sequence[str],
    l2: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_raw = _stack_features(train, members)
    prediction_raw = _stack_features(prediction, members)
    mean, scale = standardizer(train_raw)
    model = fit_nonnegative_stacker(
        (train_raw - mean) / scale,
        train["weak_label"],
        l2,
        sample_weight=peptide_equal_weights(train["peptide_id"]),
    )
    score = predict_stacker(model, (prediction_raw - mean) / scale)
    return score, {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": model["intercept"],
        "coefficient": np.asarray(model["coefficient"]).tolist(),
        "l2": model["l2"],
        "iterations": model["iterations"],
        "objective": model["objective"],
    }


def _select_grid_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["within_peptide_ap"]),
            float(row["pooled_log_loss"]),
            -float(row["l2"]),
        ),
    )[0]


def _choose_l2_inner(
    train: pd.DataFrame, members: Sequence[str], l2_grid: Sequence[float]
) -> tuple[float, list[dict[str, Any]]]:
    folds = sorted(train["fold"].unique().tolist())
    _require(len(folds) >= 2, "nested L2 selection needs at least two training folds")
    records = []
    for l2 in l2_grid:
        blocks = []
        for held in folds:
            fit = train.loc[train["fold"].ne(held)]
            validation = train.loc[train["fold"].eq(held)]
            score, _ = _fit_stacker_partition(fit, validation, members, l2)
            block = validation[["row_id", "peptide_id", "weak_label"]].copy()
            block["score"] = score
            blocks.append(block)
        prediction = pd.concat(blocks, ignore_index=True)
        records.append({"l2": float(l2), **score_metrics(prediction, prediction["score"])})
    return float(_select_grid_row(records)["l2"]), records


def cross_fitted_stacker(
    frame: pd.DataFrame, members: Sequence[str], l2_grid: Sequence[float]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = pd.Series(index=frame.index, dtype=float)
    audits = []
    for held in sorted(frame["fold"].unique().tolist()):
        train = frame.loc[frame["fold"].ne(held)].copy()
        validation = frame.loc[frame["fold"].eq(held)].copy()
        selected_l2, grid = _choose_l2_inner(train, members, l2_grid)
        score, fit = _fit_stacker_partition(train, validation, members, selected_l2)
        output.loc[validation.index] = score
        audits.append(
            {
                "held_fold": int(held),
                "train_rows": int(len(train)),
                "validation_rows": int(len(validation)),
                "selected_l2": selected_l2,
                "inner_grid": grid,
                "fit": fit,
            }
        )
    _require(not bool(output.isna().any()), "cross-fitted stacker missed rows")
    return output.to_numpy(dtype=float), audits


def fit_final_stacker(
    frame: pd.DataFrame, members: Sequence[str], l2_grid: Sequence[float]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    for l2 in l2_grid:
        output = pd.Series(index=frame.index, dtype=float)
        for held in sorted(frame["fold"].unique().tolist()):
            train = frame.loc[frame["fold"].ne(held)]
            validation = frame.loc[frame["fold"].eq(held)]
            score, _ = _fit_stacker_partition(train, validation, members, l2)
            output.loc[validation.index] = score
        grid.append({"l2": float(l2), **score_metrics(frame, output.to_numpy(dtype=float))})
    selected = _select_grid_row(grid)
    _, fit = _fit_stacker_partition(frame, frame.iloc[:1], members, float(selected["l2"]))
    fit["members_in_coefficient_order"] = list(members)
    fit["selection_rule"] = (
        "maximum cross-fitted macro within-peptide AP; then lower pooled log loss; "
        "then stronger L2"
    )
    return fit, grid


def evaluate_candidates(
    frame: pd.DataFrame,
    members: Sequence[str],
    *,
    fit_stacker: bool,
    l2_grid: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    rows = []
    specifications: dict[str, dict[str, Any]] = {}
    for size in range(1, len(members) + 1):
        for chosen in itertools.combinations(members, size):
            name = "mean_logit__" + "__".join(chosen)
            score = mean_logit_score([frame[member] for member in chosen])
            frame[name] = score
            specification = {
                "candidate": name,
                "family": "single" if size == 1 else "equal_mean_logit",
                "members": list(chosen),
                "member_count": size,
            }
            specifications[name] = specification
            rows.append(
                {
                    **specification,
                    "members": "+".join(chosen),
                    **score_metrics(frame, score),
                }
            )
    outer_audit: list[dict[str, Any]] = []
    final_stack: dict[str, Any] | None = None
    final_grid: list[dict[str, Any]] = []
    if fit_stacker:
        score, outer_audit = cross_fitted_stacker(frame, members, l2_grid)
        name = "cross_fitted_nonnegative_stack__all_members"
        frame[name] = score
        specification = {
            "candidate": name,
            "family": "cross_fitted_nonnegative_logistic",
            "members": list(members),
            "member_count": len(members),
        }
        specifications[name] = specification
        rows.append(
            {
                **specification,
                "members": "+".join(members),
                **score_metrics(frame, score),
            }
        )
        final_stack, final_grid = fit_final_stacker(frame, members, l2_grid)
    return pd.DataFrame(rows), specifications, outer_audit, final_stack, final_grid


def _within_tolerance(value: float, target: float) -> bool:
    return abs(float(value) - float(target)) <= SELECTION_TOLERANCE


def select_one_candidate(metrics: pd.DataFrame) -> dict[str, Any]:
    _require(not metrics.empty, "no eligible candidate")
    primary = float(metrics["within_peptide_ap"].max())
    pool = metrics.loc[
        metrics["within_peptide_ap"].map(lambda value: _within_tolerance(value, primary))
    ]
    secondary = float(pool["pooled_log_loss"].min())
    pool = pool.loc[
        pool["pooled_log_loss"].map(lambda value: _within_tolerance(value, secondary))
    ]
    member_count = int(pool["member_count"].min())
    pool = pool.loc[pool["member_count"].eq(member_count)]
    return pool.sort_values("candidate", kind="mergesort").iloc[0].to_dict()


def select_candidates(metrics: pd.DataFrame) -> dict[str, Any]:
    multi = metrics.loc[
        metrics["family"].eq("equal_mean_logit") & metrics["member_count"].gt(1)
    ]
    _require(not multi.empty, "no multi-model equal-logit candidate")
    return {
        "selection_data": "selection-derived weak binder/non-binder OOF labels only",
        "retention_labels_read": False,
        "primary_metric": "macro within-peptide average precision",
        "evaluable_peptide_rule": "include peptides with both weak-label classes",
        "candidate_tie_tolerance": SELECTION_TOLERANCE,
        "candidate_tie_break": (
            "within tolerance: lower pooled log loss, then fewer members, then candidate name"
        ),
        "best_candidate": select_one_candidate(metrics),
        "best_equal_logit_ensemble": select_one_candidate(multi),
    }


def select_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Choose score >= cutoff by F1, precision, fewer rows, then cutoff."""

    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int64)
    _require(score.ndim == label.ndim == 1, "scores/labels must be vectors")
    _require(len(score) == len(label) and len(score) > 0, "bad score length")
    _require(bool(np.isfinite(score).all()), "non-finite weak OOF score")
    _require(set(np.unique(label).tolist()) == {0, 1}, "weak OOF labels need both classes")
    positives = int(label.sum())
    candidates = []
    for threshold in np.unique(score):
        selected = score >= threshold
        selected_count = int(selected.sum())
        true_positive = int(np.logical_and(selected, label == 1).sum())
        false_positive = selected_count - true_positive
        false_negative = positives - true_positive
        true_negative = len(label) - true_positive - false_positive - false_negative
        precision = Fraction(true_positive, selected_count)
        recall = Fraction(true_positive, positives)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = Fraction(2 * true_positive, denominator) if denominator else Fraction(0, 1)
        payload = {
            "threshold": float(threshold),
            "recommended": selected_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        candidates.append(((f1, precision, -selected_count, float(threshold)), payload))
    return max(candidates, key=lambda item: item[0])[1]


def _score_formula(
    specification: Mapping[str, Any],
    member_contracts: Mapping[str, Mapping[str, Any]],
    final_stack: Mapping[str, Any] | None,
) -> dict[str, Any]:
    members = list(map(str, specification["members"]))
    if specification["family"] == "cross_fitted_nonnegative_logistic":
        _require(final_stack is not None, "stacker formula lacks final fit")
        _require(
            list(final_stack["members_in_coefficient_order"]) == members,
            "stacker member order changed",
        )
        components = []
        for member, coefficient, center, scale in zip(
            members,
            final_stack["coefficient"],
            final_stack["mean"],
            final_stack["scale"],
        ):
            components.append(
                {
                    "name": member,
                    "inference_logit_column": member_contracts[member]["inference_logit_column"],
                    "weight": float(coefficient),
                    "center": float(center),
                    "scale": float(scale),
                }
            )
        return {
            "input_semantics": "true component logits",
            "linear_predictor": "intercept + sum(weight_j * (logit_j - center_j) / scale_j)",
            "intercept": float(final_stack["intercept"]),
            "components": components,
            "output_transform": "sigmoid",
            "coefficient_constraint": "nonnegative",
            "l2": float(final_stack["l2"]),
        }
    weight = 1.0 / len(members)
    components = [
        {
            "name": member,
            "inference_logit_column": member_contracts[member]["inference_logit_column"],
            "weight": weight,
            "center": 0.0,
            "scale": 1.0,
        }
        for member in members
    ]
    return {
        "input_semantics": "true component logits",
        "linear_predictor": "sum(weight_j * logit_j)",
        "intercept": 0.0,
        "components": components,
        "output_transform": "sigmoid",
        "equal_logit_weights": True,
    }


def build_lock_payload(
    *,
    library: str,
    role: str,
    candidate_row: Mapping[str, Any],
    specification: Mapping[str, Any],
    member_contracts: Mapping[str, Mapping[str, Any]],
    final_stack: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    sources: Mapping[str, Any],
    max_candidates_per_peptide: int | None,
) -> dict[str, Any]:
    candidate = str(candidate_row["candidate"])
    threshold = select_f1_threshold(frame[candidate], frame["weak_label"])
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "library": library,
        "lock_role": role,
        "lock_id": f"{library.lower()}-{role}-{hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:12]}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidate": candidate,
        "candidate_family": specification["family"],
        "members": list(specification["members"]),
        "formula": _score_formula(specification, member_contracts, final_stack),
        "cutoff": {
            "decision_rule": "score >= threshold",
            "score_threshold": float(threshold["threshold"]),
            "selection_objective": "maximum F1 on matched weak-label OOF rows",
            "tie_break": "higher precision, then fewer recommendations, then higher threshold",
            "training_summary": threshold,
            "max_candidates_per_peptide": max_candidates_per_peptide,
            "pad_below_threshold": False,
        },
        "score_semantics": (
            "Selection-label score; not a retention percentage or calibrated direct-retention probability."
        ),
        "selection_provenance": {
            "primary_metric": "macro within-peptide average precision on weak labels",
            "candidate_metrics": {key: value for key, value in candidate_row.items()},
            "weak_source_sha256": {
                key: value["sha256"]
                for key, value in sources.items()
                if isinstance(value, dict) and "sha256" in value
            },
        },
        "retention_labels_read": False,
    }


def render_report(
    library: str,
    metrics: pd.DataFrame,
    selection: Mapping[str, Any],
    alignment: Mapping[str, Any],
    stacker_enabled: bool,
) -> str:
    lines = [
        f"# {library} retention-blind weak-label ensemble selection",
        "",
        f"All eligible scores were aligned on {alignment['common_rows']:,} common ",
        "double-cold weak-label OOF rows. Direct-retention values and labels were not read. ",
        "The primary comparison is average precision calculated separately within each ",
        "evaluable peptide and then averaged across peptides.",
        "",
        "The resulting score predicts the constructed selection label. It is not a retention ",
        "percentage or a calibrated probability of passing the direct-retention assay.",
        "",
        "| Candidate | Members | Within-peptide AP | Within-peptide AUROC | Pooled log loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.sort_values(
        ["within_peptide_ap", "pooled_log_loss"], ascending=[False, True], kind="mergesort"
    ).itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.members} | {row.within_peptide_ap:.6f} | "
            f"{row.within_peptide_auroc:.6f} | {row.pooled_log_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Locked weak-only choices",
            "",
            f"Best candidate: `{selection['best_candidate']['candidate']}`.",
            "",
            "Best equal-logit multi-model candidate: "
            f"`{selection['best_equal_logit_ensemble']['candidate']}`.",
            "",
            "Ties within 1e-12 on within-peptide AP are resolved by lower pooled log loss, ",
            "then fewer component models, then candidate name. Cutoff ties are resolved by ",
            "higher precision, fewer recommendations, then higher score.",
            "",
        ]
    )
    if stacker_enabled:
        lines.extend(
            [
                "The optional nonnegative stacker was evaluated with second-level cross-fitting. ",
                "Its regularization, standardization, and coefficients use only weak-label OOF ",
                "rows; the held outer fold is excluded from each stacker fit.",
                "",
            ]
        )
    return "\n".join(lines)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    library = str(config["library"])
    output_dir = Path(output_dir).resolve()
    _reject_retention_path(output_dir, "output")
    _require(not output_dir.exists(), f"output exists; refusing overwrite: {output_dir}")
    _require("private_data" in output_dir.parts, "output must remain below private_data")

    membership_path = _resolve_from_config(config_path, str(config["fold_membership"]))
    validation, membership_audit = load_and_validate_membership(
        membership_path, library, int(config["expected_folds"])
    )
    member_frames = []
    member_records = []
    for member in config["members"]:
        frame, record = load_member_oof(config_path, member, library, validation)
        member_frames.append(frame)
        member_records.append(record)
    member_names = [str(member["name"]) for member in config["members"]]
    frame, alignment = align_common_oof(
        member_frames,
        member_names,
        validation,
        str(config["common_row_policy"]),
    )
    if config.get("expected_common_oof_rows") is not None:
        _require(
            len(frame) == int(config["expected_common_oof_rows"]),
            "common OOF row count changed",
        )
    if config.get("expected_common_oof_positive") is not None:
        _require(
            int(frame["weak_label"].sum()) == int(config["expected_common_oof_positive"]),
            "common OOF positive count changed",
        )

    metrics, specifications, outer_audit, final_stack, final_grid = evaluate_candidates(
        frame,
        member_names,
        fit_stacker=bool(config["fit_nonnegative_stacker"]),
        l2_grid=config["stack_l2_grid"],
    )
    threshold_rows = [
        {"candidate": row.candidate, **select_f1_threshold(frame[row.candidate], frame["weak_label"])}
        for row in metrics.itertuples(index=False)
    ]
    threshold_frame = pd.DataFrame(threshold_rows)
    metrics = metrics.merge(threshold_frame, on="candidate", validate="one_to_one")
    selection = select_candidates(metrics)

    output_dir.mkdir(parents=True, mode=0o700)
    prediction_columns = [
        "row_id",
        "fold",
        "weak_label",
        "peptide_id",
        "affibody_id",
        *member_names,
        *metrics["candidate"].tolist(),
    ]
    prediction_path = output_dir / "matched_oof_predictions.csv.gz"
    _write_csv_exclusive(prediction_path, frame[prediction_columns], compression="gzip")
    metric_path = output_dir / "candidate_metrics.csv"
    _write_csv_exclusive(metric_path, metrics.sort_values("candidate").reset_index(drop=True))
    membership_audit_path = output_dir / "double_cold_membership_audit.json"
    _write_json_exclusive(membership_audit_path, membership_audit)
    outer_path = output_dir / "stacker_crossfit_audit.json"
    _write_json_exclusive(outer_path, outer_audit)
    grid_path = output_dir / "stacker_final_l2_grid.json"
    _write_json_exclusive(grid_path, final_grid)
    stack_path = output_dir / "stacker_locked_parameters.json"
    _write_json_exclusive(
        stack_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "library": library,
            "enabled": final_stack is not None,
            "retention_labels_read": False,
            "parameters": final_stack,
        },
    )
    selection_path = output_dir / "locked_weak_selection.json"
    _write_json_exclusive(
        selection_path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "library": library,
            "retention_labels_read": False,
            **selection,
        },
    )

    sources: dict[str, Any] = {
        "config": _record(config_path),
        "fold_membership": _record(membership_path),
    }
    for record in member_records:
        sources[f"member:{record['name']}:oof"] = record["oof"]
        sources[f"member:{record['name']}:audit"] = record["audit"]
    member_contracts = {record["name"]: record for record in member_records}
    lock_definitions = {
        "selected_score": selection["best_candidate"],
        "best_equal_logit_ensemble": selection["best_equal_logit_ensemble"],
    }
    if final_stack is not None:
        stack_candidate = metrics.loc[
            metrics["family"].eq("cross_fitted_nonnegative_logistic")
        ].iloc[0].to_dict()
        lock_definitions["nonnegative_stacker"] = stack_candidate
    lock_records: dict[str, Any] = {}
    for role, candidate_row in lock_definitions.items():
        candidate = str(candidate_row["candidate"])
        payload = build_lock_payload(
            library=library,
            role=role,
            candidate_row=candidate_row,
            specification=specifications[candidate],
            member_contracts=member_contracts,
            final_stack=final_stack,
            frame=frame,
            sources=sources,
            max_candidates_per_peptide=(
                None
                if config.get("max_candidates_per_peptide") is None
                else int(config["max_candidates_per_peptide"])
            ),
        )
        lock_path = output_dir / f"{role}.lock.json"
        _write_json_exclusive(lock_path, payload)
        lock_records[lock_path.name] = _record(lock_path)

    report_path = output_dir / "report.md"
    report_path.write_text(
        render_report(
            library,
            metrics,
            selection,
            alignment,
            bool(config["fit_nonnegative_stacker"]),
        ),
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    output_paths = [
        prediction_path,
        metric_path,
        membership_audit_path,
        outer_path,
        grid_path,
        stack_path,
        selection_path,
        report_path,
    ]
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": float(time.time() - started),
        "library": library,
        "retention_labels_read": False,
        "validation_regime": "double_cold",
        "folds": int(config["expected_folds"]),
        "alignment": alignment,
        "members": member_records,
        "selection": selection,
        "stacker": {
            "enabled": final_stack is not None,
            "input": "clipped component logits standardized on stacker-training OOF rows only",
            "coefficient_bounds": "nonnegative",
            "training_loss": "peptide-equal-weighted logistic loss plus L2",
            "l2_grid": config["stack_l2_grid"],
            "selection_metric": "macro within-peptide AP on weak labels",
        },
        "cutoff": {
            "objective": "F1 on matched weak-label OOF rows",
            "tie_break": "higher precision, fewer recommendations, higher threshold",
        },
        "sources": sources,
        "locks": lock_records,
        "outputs": {path.name: _record(path) for path in output_paths},
        "code": _record(Path(__file__).resolve()),
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    print(metrics.sort_values("within_peptide_ap", ascending=False).to_string(index=False))
    print(json.dumps(selection, indent=2, sort_keys=True, default=float))
    print(f"wrote {output_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
