#!/usr/bin/env python3
"""Build retention-blind LibA sequence-model OOF and deployment predictions.

This program reads only explicitly named, label-safe members of the existing
MINT multilayer archive.  It reconstructs the strict 22,542-row LibA weak-label
pool, uses the established three diagonal double-cold folds, and emits scores
for the label-free 108-row LibA evaluation roster.  Numerical retention values
and retention-derived binder labels are neither accepted as arguments nor read.

Four sequence-only model families are included:

* an additive balanced logistic model over the six designed residues;
* the weak-label-selected frozen MINT layer-9 representation plus a balanced
  logistic head;
* the canonical frozen MINT layer-33 control plus the same kind of head; and
* a small nonlinear six-residue MLP, trained with five fixed final seeds.

Regularization strengths and the MLP epoch count are selected exclusively by
weak-validation log loss.  The output contains aligned OOF scores, target-free
evaluation scores, serialized logistic heads, and deployable MLP checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()

SCHEMA_VERSION = "liba-common-oof-sequence-predictions-v1"
CONFIG_SCHEMA_VERSION = "liba-common-oof-sequence-config-v1"
HEAD_SCHEMA_VERSION = "liba-additive-mint-deployment-heads-v1"
ADDITIVE_HEAD_SCHEMA_VERSION = "liba-additive-deployment-head-v1"
MINT9_HEAD_SCHEMA_VERSION = "liba-mint-layer9-deployment-head-v1"
MLP_CHECKPOINT_SCHEMA_VERSION = "liba-nonlinear-6site-checkpoint-v1"
ARCHIVE_SCHEMA_VERSION = "mint-multilayer-chain-mean-cache-v1"

LIBRARY = "LibA"
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {value: index for index, value in enumerate(AA_ALPHABET)}
POSITION_NAMES = (
    "peptide_position_4",
    "peptide_position_5",
    "affibody_displayed_13_crystal_15",
    "affibody_displayed_17_crystal_19",
    "affibody_displayed_27_crystal_29",
    "affibody_displayed_31_crystal_33",
)
POOLING_CONTRACT = (
    "exclude cls/eos/padding; mean residues separately within chain_id 0 and "
    "chain_id 1; concatenate chain 0 then chain 1"
)

FOLDS = 3
SPLIT_SEED = 17
EXPECTED_ARCHIVE_ROWS = 69_945
EXPECTED_TRAIN_ROWS = 22_542
EXPECTED_POSITIVE = 11_320
EXPECTED_NEGATIVE = 11_222
EXPECTED_EVAL_ROWS = 108
EXPECTED_OOF_ROWS = 7_515
EXPECTED_OOF_POSITIVE = 3_759
EXPECTED_TRAIN_MEMBERSHIP_SHA256 = (
    "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e"
)
EXPECTED_FOLDS = {
    0: {
        "train": 9_076,
        "guard": 10_513,
        "validation": 2_953,
        "validation_positive": 1_530,
        "train_sha256": "fd90f5dac3ee87c350940842211d1c3c82194c74ad6c5ab6b4429f011a3fd856",
        "validation_sha256": "57622c6653b18879b5f237b050dc0322d36b8d9a29d70d380c8b9df9c375b641",
    },
    1: {
        "train": 10_488,
        "guard": 9_791,
        "validation": 2_263,
        "validation_positive": 1_267,
        "train_sha256": "faf658d1b9f6e34a89e0deb6738f27f479f0c8bfeefbd8dda4d977262cb7ff8a",
        "validation_sha256": "6f6ae2b7f9d8bbb75c4e5563152a76f9620ba5169471278b4ba2eeba699a3224",
    },
    2: {
        "train": 10_493,
        "guard": 9_750,
        "validation": 2_299,
        "validation_positive": 962,
        "train_sha256": "d2ffa04c39e502416a0ce6565cb7f2f8919af49892cd68a98927b73aa7c3b11b",
        "validation_sha256": "a0b2f9cd5e33cf5f182ca9294e3c703b2b99fc32009760943639a8f443f4a38f",
    },
}

SAFE_ARCHIVE_MEMBERS = (
    "row_index",
    "cache_uid",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "weak_label",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
    "representation_layers",
)
MINT_FEATURE_KEYS = {
    "frozen_mint_layer9": "mint_layer_09_chain_mean",
    "frozen_mint_layer33_control": "mint_layer_33_chain_mean",
}
FORBIDDEN_INPUT_FIELD_PARTS = ("retention", "target", "outcome", "binder")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("ascii")).hexdigest()


def row_fold_sha256(frame: pd.DataFrame) -> str:
    selected = frame[["row_id", "fold"]].drop_duplicates().sort_values("row_id")
    payload = "\n".join(
        selected["row_id"].astype(str) + "\t" + selected["fold"].astype(str)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _validate_output_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    return output


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _require(config.get("schema_version") == CONFIG_SCHEMA_VERSION, "config schema changed")
    dataset = config.get("dataset", {})
    expected_dataset = {
        "library": LIBRARY,
        "weak_rows": EXPECTED_TRAIN_ROWS,
        "weak_positive": EXPECTED_POSITIVE,
        "weak_negative": EXPECTED_NEGATIVE,
        "evaluation_rows": EXPECTED_EVAL_ROWS,
        "weak_membership_sha256": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
    }
    for key, value in expected_dataset.items():
        _require(dataset.get(key) == value, f"config dataset {key} changed")
    validation = config.get("validation", {})
    _require(validation.get("regime") == "double_cold", "validation regime changed")
    _require(int(validation.get("folds", -1)) == FOLDS, "fold count changed")
    _require(int(validation.get("split_seed", -1)) == SPLIT_SEED, "split seed changed")
    _require(
        validation.get("selection_metric") == "pooled_weak_validation_log_loss",
        "selection metric changed",
    )
    logistic = config.get("logistic", {})
    c_grid = tuple(float(value) for value in logistic.get("c_grid", ()))
    _require(c_grid == (0.001, 0.01, 0.1, 1.0), "logistic C grid changed")
    _require(logistic.get("class_weight") == "balanced", "logistic balance changed")
    _require(logistic.get("solver") == "liblinear", "logistic solver changed")
    mlp = config.get("models", {}).get("nonlinear_6site", {})
    _require(mlp.get("input_dimension") == 120, "MLP input dimension changed")
    _require(tuple(mlp.get("hidden_dimensions", ())) == (64, 32), "MLP widths changed")
    seeds = tuple(map(int, mlp.get("final_seeds", ())))
    _require(len(seeds) == 5 and len(set(seeds)) == 5, "five unique MLP seeds required")
    _require(int(mlp.get("epoch_selection_seed", -1)) in seeds, "selection seed is not final")
    _require(config.get("retention_labels_allowed") is False, "retention labels must be forbidden")
    return config


def validate_layer9_selection_audit(path: str | Path) -> dict[str, Any]:
    """Bind layer 9 to the completed weak-label-only multilayer screen."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("retention_labels_used") is False, "layer screen used retention labels")
    selection = payload.get("selections", {}).get(LIBRARY, {})
    _require(selection.get("direct_retention_used_for_selection") is False, "layer selection used retention")
    _require(selection.get("selection_data") == "weak PN labels only", "layer selection data changed")
    _require(int(selection.get("layer", -1)) == 9, "weak-selected LibA layer is not 9")
    _require(int(selection.get("folds", -1)) == FOLDS, "layer-selection folds changed")
    _require(int(selection.get("split_seed", -1)) == SPLIT_SEED, "layer-selection split seed changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "selected_layer": 9,
        "selected_C": float(selection["C"]),
        "selection_metric": str(selection["selection_metric"]),
        "retention_labels_used": False,
    }


def load_label_safe_archive_header(path: str | Path) -> dict[str, np.ndarray]:
    """Load only explicitly enumerated metadata members needed by this task."""

    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        header = {name: np.asarray(archive[name]).copy() for name in SAFE_ARCHIVE_MEMBERS}
    length = len(header["row_index"])
    _require(length == EXPECTED_ARCHIVE_ROWS, "multilayer archive row count changed")
    for name, values in header.items():
        if name == "representation_layers":
            continue
        _require(len(values) == length, f"archive member {name} length changed")
    _require(
        np.array_equal(header["row_index"], np.arange(length, dtype=header["row_index"].dtype)),
        "archive row indices changed",
    )
    _require(len(set(header["pair_uid"].astype(str))) == length, "duplicate pair UID")
    layers = tuple(map(int, header["representation_layers"].tolist()))
    _require(9 in layers and 33 in layers, "archive lacks required MINT layers")
    return header


def load_layer_matrix(path: str | Path, feature_key: str) -> np.ndarray:
    _require(feature_key in set(MINT_FEATURE_KEYS.values()), "unapproved MINT feature key")
    with np.load(Path(path), allow_pickle=False) as archive:
        values = np.asarray(archive[feature_key], dtype=np.float32).copy()
    _require(values.shape == (EXPECTED_ARCHIVE_ROWS, 2560), "MINT layer shape changed")
    _require(bool(np.isfinite(values).all()), "MINT layer contains non-finite values")
    return values


def validate_feature_manifest(path: str | Path, archive_path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    archive_path = Path(archive_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema_version") == ARCHIVE_SCHEMA_VERSION, "feature manifest schema changed")
    output = manifest.get("output", {})
    _require(output.get("compressed") is False, "multilayer archive compression changed")
    _require(Path(str(output.get("path", ""))).resolve() == archive_path, "manifest archive path changed")
    features = manifest.get("features", {})
    _require(features.get("pooling") == POOLING_CONTRACT, "MINT pooling contract changed")
    by_name = {str(item.get("name")): item for item in features.get("by_layer", [])}
    for key in MINT_FEATURE_KEYS.values():
        _require(by_name.get(key, {}).get("shape") == [EXPECTED_ARCHIVE_ROWS, 2560], f"manifest {key} shape changed")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "declared_archive_sha256": str(output.get("sha256", "")),
        "pooling": POOLING_CONTRACT,
    }


def _header_frame(header: Mapping[str, np.ndarray]) -> pd.DataFrame:
    names = [name for name in SAFE_ARCHIVE_MEMBERS if name != "representation_layers"]
    frame = pd.DataFrame({name: np.asarray(header[name]) for name in names})
    for column in frame.columns:
        if column != "row_index":
            frame[column] = frame[column].astype(str)
    frame["_cache_index"] = frame["row_index"].astype(int)
    return frame


def prepare_liba_rows(header: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the primary strict pool without opening direct outcomes."""

    frame = _header_frame(header)
    evaluation = frame.loc[
        frame["source_kind"].eq("retention") & frame["library"].eq(LIBRARY)
    ].copy()
    weak = frame.loc[
        frame["source_kind"].eq("weak") & frame["library"].eq(LIBRARY)
    ].copy()
    _require(len(evaluation) == EXPECTED_EVAL_ROWS, "LibA evaluation roster changed")
    _require(bool(evaluation["weak_label"].eq("").all()), "evaluation roster contains weak labels")
    eval_peptides = set(evaluation["chain1_sha256"])
    eval_affibodies = set(evaluation["chain2_sha256"])
    primary = weak.loc[
        ~weak["chain1_sha256"].isin(eval_peptides)
        & ~weak["chain2_sha256"].isin(eval_affibodies)
    ].copy()
    primary["weak_label"] = pd.to_numeric(primary["weak_label"], errors="raise").astype(int)
    primary = primary.reset_index(drop=True)
    primary["_condition_index"] = np.arange(len(primary), dtype=int)
    evaluation = evaluation.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    observed = {
        "rows": len(primary),
        "positive": int(primary["weak_label"].sum()),
        "negative": int(primary["weak_label"].eq(0).sum()),
        "membership": membership_sha256(primary["sequence_pair_sha256"]),
    }
    expected = {
        "rows": EXPECTED_TRAIN_ROWS,
        "positive": EXPECTED_POSITIVE,
        "negative": EXPECTED_NEGATIVE,
        "membership": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
    }
    _require(observed == expected, f"strict LibA pool changed: {observed}")
    _require(set(primary["chain1_sha256"]).isdisjoint(eval_peptides), "evaluation peptide entered training")
    _require(set(primary["chain2_sha256"]).isdisjoint(eval_affibodies), "evaluation Affibody entered training")
    return primary, evaluation


def _stable_bin(value: str, axis: str) -> int:
    payload = f"{axis}|{SPLIT_SEED}|{value}".encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % FOLDS


@dataclass(frozen=True)
class FoldPlan:
    fold: int
    train: np.ndarray
    guard: np.ndarray
    validation: np.ndarray


def build_fold_plans(primary: pd.DataFrame) -> tuple[tuple[FoldPlan, ...], pd.DataFrame]:
    plans = []
    blocks = []
    for fold in range(FOLDS):
        peptide_held = np.asarray(
            [_stable_bin(value, "peptide") == fold for value in primary["chain1_sha256"]]
        )
        affibody_held = np.asarray(
            [_stable_bin(value, "affibody") == fold for value in primary["chain2_sha256"]]
        )
        validation = np.flatnonzero(peptide_held & affibody_held)
        train = np.flatnonzero(~peptide_held & ~affibody_held)
        guard = np.flatnonzero(np.logical_xor(peptide_held, affibody_held))
        plan = FoldPlan(fold, train, guard, validation)
        expected = EXPECTED_FOLDS[fold]
        observed = {
            "train": len(train),
            "guard": len(guard),
            "validation": len(validation),
            "validation_positive": int(primary.iloc[validation]["weak_label"].sum()),
            "train_sha256": membership_sha256(primary.iloc[train]["sequence_pair_sha256"]),
            "validation_sha256": membership_sha256(primary.iloc[validation]["sequence_pair_sha256"]),
        }
        _require(observed == expected, f"LibA fold {fold} changed: {observed}")
        role = np.full(len(primary), "guard", dtype=object)
        role[train] = "train"
        role[validation] = "validation"
        block = pd.DataFrame(
            {
                "fold": fold,
                "row_id": primary["pair_uid"].astype(str),
                "weak_label": primary["weak_label"].astype(int),
                "chain1_sha256": primary["chain1_sha256"].astype(str),
                "chain2_sha256": primary["chain2_sha256"].astype(str),
                "sequence_pair_sha256": primary["sequence_pair_sha256"].astype(str),
                "peptide_design_code": primary["peptide_design_code"].astype(str),
                "affibody_design_code": primary["affibody_design_code"].astype(str),
                "role": role,
            }
        )
        blocks.append(block)
        plans.append(plan)
    validation_ids = pd.concat(
        [block.loc[block["role"].eq("validation"), ["row_id", "weak_label"]] for block in blocks],
        ignore_index=True,
    )
    _require(len(validation_ids) == EXPECTED_OOF_ROWS, "LibA OOF row count changed")
    _require(validation_ids["row_id"].is_unique, "LibA OOF row occurs in multiple folds")
    _require(int(validation_ids["weak_label"].sum()) == EXPECTED_OOF_POSITIVE, "LibA OOF positives changed")
    return tuple(plans), pd.concat(blocks, ignore_index=True)


def designed_codes(frame: pd.DataFrame) -> np.ndarray:
    peptide = frame["peptide_design_code"].astype(str)
    affibody = frame["affibody_design_code"].astype(str)
    _require(bool(peptide.str.len().eq(2).all()), "LibA peptide code length changed")
    _require(bool(affibody.str.len().eq(4).all()), "LibA Affibody code length changed")
    values = np.asarray([list(left + right) for left, right in zip(peptide, affibody)])
    _require(values.shape == (len(frame), 6), "six-position code shape changed")
    _require(set(values.reshape(-1)).issubset(AA_TO_INDEX), "noncanonical designed residue")
    return values


def encode_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    _require(codes.ndim == 2 and codes.shape[1] == 6, "code matrix must be [N,6]")
    output = np.zeros((len(codes), 6, len(AA_ALPHABET)), dtype=np.float32)
    for position in range(6):
        indices = [AA_TO_INDEX[str(value)] for value in codes[:, position]]
        output[np.arange(len(codes)), position, indices] = 1.0
    return output.reshape(len(codes), -1)


def binary_metrics(labels: Sequence[int], probability: Sequence[float]) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    _require(len(labels) == len(probability) and len(labels) > 0, "metric length changed")
    _require(set(labels.tolist()) == {0, 1}, "metrics require both classes")
    _require(bool(np.isfinite(probability).all()), "non-finite probability")
    return {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
    }


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    _require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()), "bad standardizer")
    return mean, scale


def standardize(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = (np.asarray(features, dtype=np.float64) - mean) / scale
    _require(bool(np.isfinite(output).all()), "non-finite standardized feature")
    return output.astype(np.float32)


def make_logistic(c_value: float, tol: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        max_iter=int(max_iter),
        tol=float(tol),
    )


@dataclass
class LogisticResult:
    selected_c: float
    oof: pd.DataFrame
    validation_rows: list[dict[str, Any]]


def cross_validate_logistic(
    model_name: str,
    features: np.ndarray,
    primary: pd.DataFrame,
    plans: Sequence[FoldPlan],
    c_grid: Sequence[float],
    *,
    standardize_features: bool,
    tol: float,
    max_iter: int,
) -> LogisticResult:
    labels = primary["weak_label"].to_numpy(dtype=int)
    _require(features.shape[0] == len(primary), f"{model_name} feature rows changed")
    by_c: dict[float, list[pd.DataFrame]] = {}
    validation_rows: list[dict[str, Any]] = []
    aggregate_rows = []
    for c_value in map(float, c_grid):
        blocks = []
        for plan in plans:
            x_train = features[plan.train]
            x_validation = features[plan.validation]
            if standardize_features:
                mean, scale = fit_standardizer(x_train)
                x_train = standardize(x_train, mean, scale)
                x_validation = standardize(x_validation, mean, scale)
            classifier = make_logistic(c_value, tol, max_iter)
            classifier.fit(x_train, labels[plan.train])
            _require(int(classifier.n_iter_[0]) < max_iter, f"{model_name} logistic did not converge")
            probability = classifier.predict_proba(x_validation)[:, 1]
            metrics = binary_metrics(labels[plan.validation], probability)
            validation_rows.append(
                {
                    "model": model_name,
                    "record_type": "fold",
                    "C": c_value,
                    "fold": plan.fold,
                    "selected": 0,
                    "n_train": len(plan.train),
                    "n_guard": len(plan.guard),
                    **metrics,
                }
            )
            blocks.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "seed": "fixed",
                        "fold": plan.fold,
                        "row_id": primary.iloc[plan.validation]["pair_uid"].astype(str).to_numpy(),
                        "weak_label": labels[plan.validation],
                        "probability": probability,
                    }
                )
            )
        combined = pd.concat(blocks, ignore_index=True)
        metrics = binary_metrics(combined["weak_label"], combined["probability"])
        record = {
            "model": model_name,
            "record_type": "aggregate",
            "C": c_value,
            "fold": -1,
            "selected": 0,
            "n_train": sum(len(plan.train) for plan in plans),
            "n_guard": sum(len(plan.guard) for plan in plans),
            **metrics,
        }
        validation_rows.append(record)
        aggregate_rows.append(record)
        by_c[c_value] = blocks
    selected = min(aggregate_rows, key=lambda row: (row["log_loss"], row["C"]))
    selected_c = float(selected["C"])
    for row in validation_rows:
        row["selected"] = int(float(row["C"]) == selected_c)
    oof = pd.concat(by_c[selected_c], ignore_index=True)
    _require(len(oof) == EXPECTED_OOF_ROWS and oof["row_id"].is_unique, f"{model_name} OOF coverage changed")
    return LogisticResult(selected_c, oof, validation_rows)


def fit_final_logistic(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    *,
    standardize_features: bool,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if standardize_features:
        mean, scale = fit_standardizer(train_features)
        x_train = standardize(train_features, mean, scale)
        x_eval = standardize(eval_features, mean, scale)
    else:
        mean = np.zeros(train_features.shape[1], dtype=np.float64)
        scale = np.ones(train_features.shape[1], dtype=np.float64)
        x_train = train_features
        x_eval = eval_features
    classifier = make_logistic(c_value, tol, max_iter)
    classifier.fit(x_train, labels)
    _require(int(classifier.n_iter_[0]) < max_iter, "final logistic did not converge")
    probability = classifier.predict_proba(x_eval)[:, 1]
    head = {
        "mean": np.asarray(mean, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "coef": np.asarray(classifier.coef_, dtype=np.float64).reshape(-1),
        "intercept": np.asarray(classifier.intercept_, dtype=np.float64).reshape(-1),
    }
    return probability, head, {
        "C": float(c_value),
        "iterations": int(classifier.n_iter_[0]),
        "trainable_parameters": int(classifier.coef_.size + classifier.intercept_.size),
    }


class NonlinearSixSiteMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        hidden = tuple(map(int, hidden_dims))
        _require(input_dim == 120 and len(hidden) == 2, "MLP architecture changed")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(hidden[0]),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        _require(values.ndim == 2 and values.shape[1] == self.input_dim, "MLP input shape changed")
        return self.network(values.float()).squeeze(-1)


def trainable_parameters(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def derived_seed(seed: int, stage: str, fold: int) -> int:
    payload = f"{seed}|{stage}|{fold}".encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def class_weights(labels: np.ndarray) -> Tensor:
    values = np.asarray(labels, dtype=int)
    counts = np.bincount(values, minlength=2)
    _require(bool((counts > 0).all()), "both classes required")
    return torch.as_tensor(len(values) / (2.0 * counts), dtype=torch.float32)


def _loader(
    features: np.ndarray,
    labels: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensors: tuple[Tensor, ...]
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    if labels is None:
        tensors = (x,)
    else:
        tensors = (x, torch.from_numpy(np.asarray(labels, dtype=np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _predict_mlp(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    blocks = []
    with torch.inference_mode():
        for (values,) in _loader(features, None, batch_size, False, 0):
            blocks.append(torch.sigmoid(model(values.to(device))).cpu().numpy())
    output = np.concatenate(blocks)
    _require(len(output) == len(features), "MLP prediction length changed")
    return output


def _train_mlp_epochs(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    prediction_features: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    epochs: int,
    *,
    validation_labels: np.ndarray | None = None,
    early_stop: bool = False,
) -> tuple[nn.Module, np.ndarray, list[dict[str, Any]], int]:
    _set_seed(seed)
    model = NonlinearSixSiteMLP(
        int(config["input_dimension"]), config["hidden_dimensions"], float(config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    weights = class_weights(train_labels).to(device)
    loader = _loader(train_features, train_labels, int(config["batch_size"]), True, seed)
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    best_probability = None
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for values, labels in loader:
            values = values.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss_by_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = torch.mean(loss_by_row * weights[labels.long()])
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(labels)
            count += len(labels)
        probability = _predict_mlp(
            model, prediction_features, int(config["evaluation_batch_size"]), device
        )
        record: dict[str, Any] = {"epoch": epoch, "train_weighted_bce": total / count}
        if validation_labels is not None:
            metrics = binary_metrics(validation_labels, probability)
            record.update(metrics)
            if metrics["log_loss"] < best_loss - float(config["early_stopping_min_delta"]):
                best_loss = float(metrics["log_loss"])
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_probability = probability.copy()
                stale = 0
            else:
                stale += 1
        history.append(record)
        if early_stop and stale >= int(config["early_stopping_patience"]):
            break
    if validation_labels is not None:
        _require(best_state is not None and best_probability is not None, "MLP selection produced no checkpoint")
        model.load_state_dict(best_state)
        return model, best_probability, history, best_epoch
    return model, probability, history, int(epochs)


def select_mlp_epoch(
    features: np.ndarray,
    primary: pd.DataFrame,
    plans: Sequence[FoldPlan],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    labels = primary["weak_label"].to_numpy(dtype=int)
    best_epochs = []
    rows = []
    selection_seed = int(config["epoch_selection_seed"])
    for plan in plans:
        seed = derived_seed(selection_seed, "epoch_selection", plan.fold)
        _, _, history, best_epoch = _train_mlp_epochs(
            features[plan.train],
            labels[plan.train],
            features[plan.validation],
            config,
            seed,
            device,
            int(config["max_epochs"]),
            validation_labels=labels[plan.validation],
            early_stop=True,
        )
        best_epochs.append(best_epoch)
        for record in history:
            rows.append(
                {
                    "model": "nonlinear_6site",
                    "record_type": "epoch_selection",
                    "fold": plan.fold,
                    "seed": selection_seed,
                    "selected": 0,
                    **record,
                }
            )
    selected_epoch = max(1, int(np.median(best_epochs)))
    for row in rows:
        row["selected"] = int(int(row["epoch"]) == selected_epoch)
    rows.append(
        {
            "model": "nonlinear_6site",
            "record_type": "epoch_selection_summary",
            "fold": -1,
            "seed": selection_seed,
            "selected": 1,
            "epoch": selected_epoch,
            "fold_best_epochs": json.dumps(best_epochs),
        }
    )
    return selected_epoch, rows


def fit_mlp_oof_and_final(
    train_features: np.ndarray,
    eval_features: np.ndarray,
    primary: pd.DataFrame,
    evaluation: pd.DataFrame,
    plans: Sequence[FoldPlan],
    config: Mapping[str, Any],
    selected_epoch: int,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    labels = primary["weak_label"].to_numpy(dtype=int)
    oof_blocks = []
    eval_blocks = []
    summaries = []
    parity = {}
    expected_parameters = trainable_parameters(
        NonlinearSixSiteMLP(120, config["hidden_dimensions"], config["dropout"])
    )
    _require(expected_parameters == 10_225, "MLP parameter count changed")
    for seed in map(int, config["final_seeds"]):
        seed_oof = []
        for plan in plans:
            run_seed = derived_seed(seed, "oof", plan.fold)
            _, probability, history, _ = _train_mlp_epochs(
                train_features[plan.train],
                labels[plan.train],
                train_features[plan.validation],
                config,
                run_seed,
                device,
                selected_epoch,
            )
            block = pd.DataFrame(
                {
                    "model": "nonlinear_6site",
                    "seed": str(seed),
                    "fold": plan.fold,
                    "row_id": primary.iloc[plan.validation]["pair_uid"].astype(str).to_numpy(),
                    "weak_label": labels[plan.validation],
                    "probability": probability,
                }
            )
            oof_blocks.append(block)
            seed_oof.append(block)
            summaries.append(
                {
                    "stage": "oof",
                    "model": "nonlinear_6site",
                    "seed": seed,
                    "fold": plan.fold,
                    "epochs": selected_epoch,
                    "last_train_weighted_bce": float(history[-1]["train_weighted_bce"]),
                    **binary_metrics(block["weak_label"], block["probability"]),
                }
            )
        combined_oof = pd.concat(seed_oof, ignore_index=True)
        _require(len(combined_oof) == EXPECTED_OOF_ROWS and combined_oof["row_id"].is_unique, "MLP OOF coverage changed")
        summaries.append(
            {
                "stage": "oof_aggregate",
                "model": "nonlinear_6site",
                "seed": seed,
                "fold": -1,
                "epochs": selected_epoch,
                **binary_metrics(combined_oof["weak_label"], combined_oof["probability"]),
            }
        )

        run_seed = derived_seed(seed, "final", -1)
        model, probability, history, _ = _train_mlp_epochs(
            train_features,
            labels,
            eval_features,
            config,
            run_seed,
            device,
            selected_epoch,
        )
        checkpoint_path = checkpoint_dir / f"nonlinear_6site__seed{seed}.pt"
        torch.save(
            {
                "schema_version": MLP_CHECKPOINT_SCHEMA_VERSION,
                "model": "nonlinear_6site",
                "seed": seed,
                "derived_training_seed": run_seed,
                "epochs": selected_epoch,
                "input_dimension": 120,
                "hidden_dimensions": list(map(int, config["hidden_dimensions"])),
                "dropout": float(config["dropout"]),
                "position_names": list(POSITION_NAMES),
                "amino_acid_alphabet": list(AA_ALPHABET),
                "weak_training_membership_sha256": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            },
            checkpoint_path,
        )
        os.chmod(checkpoint_path, 0o600)
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
        replay = NonlinearSixSiteMLP(
            int(loaded["input_dimension"]), loaded["hidden_dimensions"], float(loaded["dropout"])
        ).to(device)
        replay.load_state_dict(loaded["state_dict"], strict=True)
        replay_probability = _predict_mlp(
            replay, eval_features, int(config["evaluation_batch_size"]), device
        )
        maximum_difference = float(np.max(np.abs(replay_probability - probability)))
        _require(maximum_difference <= 1e-7, "serialized MLP checkpoint failed self-parity")
        parity[str(seed)] = {
            "checkpoint": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "maximum_absolute_probability_difference": maximum_difference,
        }
        eval_blocks.append(
            pd.DataFrame(
                {
                    "eval_row_id": evaluation["pair_uid"].astype(str),
                    "model": "nonlinear_6site",
                    "seed": str(seed),
                    "score": probability,
                }
            )
        )
        summaries.append(
            {
                "stage": "final",
                "model": "nonlinear_6site",
                "seed": seed,
                "fold": -1,
                "epochs": selected_epoch,
                "rows": EXPECTED_TRAIN_ROWS,
                "trainable_parameters": expected_parameters,
                "last_train_weighted_bce": float(history[-1]["train_weighted_bce"]),
            }
        )
        del model, replay
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        pd.concat(oof_blocks, ignore_index=True),
        pd.concat(eval_blocks, ignore_index=True),
        summaries,
        parity,
    )


def mean_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    mean = logits.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-mean))


def build_aligned_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    key = predictions[["row_id", "fold", "weak_label"]].drop_duplicates()
    _require(len(key) == EXPECTED_OOF_ROWS and key["row_id"].is_unique, "OOF alignment key changed")
    output = key.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    for (model, seed), block in predictions.groupby(["model", "seed"], sort=True):
        _require(len(block) == EXPECTED_OOF_ROWS and block["row_id"].is_unique, f"{model}/{seed} OOF coverage changed")
        column = model if seed == "fixed" else f"{model}__seed{seed}"
        output = output.merge(
            block[["row_id", "probability"]].rename(columns={"probability": column}),
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
    seed_columns = [value for value in output if value.startswith("nonlinear_6site__seed")]
    _require(len(seed_columns) == 5, "aligned MLP seed columns changed")
    output["nonlinear_6site_mean_logit"] = mean_logit(output[seed_columns].to_numpy(float))
    return output


def build_aligned_eval(predictions: pd.DataFrame) -> pd.DataFrame:
    ids = predictions[["eval_row_id"]].drop_duplicates().sort_values("eval_row_id").reset_index(drop=True)
    _require(len(ids) == EXPECTED_EVAL_ROWS, "evaluation alignment key changed")
    output = ids
    for (model, seed), block in predictions.groupby(["model", "seed"], sort=True):
        _require(len(block) == EXPECTED_EVAL_ROWS and block["eval_row_id"].is_unique, f"{model}/{seed} eval coverage changed")
        column = model if seed == "fixed" else f"{model}__seed{seed}"
        output = output.merge(
            block[["eval_row_id", "score"]].rename(columns={"score": column}),
            on="eval_row_id",
            how="inner",
            validate="one_to_one",
        )
    seed_columns = [value for value in output if value.startswith("nonlinear_6site__seed")]
    output["nonlinear_6site_mean_logit"] = mean_logit(output[seed_columns].to_numpy(float))
    return output


def save_deployment_heads(
    path: Path,
    additive_head: Mapping[str, np.ndarray],
    additive_c: float,
    mint9_head: Mapping[str, np.ndarray],
    mint9_c: float,
    mint33_head: Mapping[str, np.ndarray],
    mint33_c: float,
) -> dict[str, Any]:
    categories = np.asarray([AA_ALPHABET for _ in range(6)])
    np.savez(
        path,
        schema_version=np.asarray([HEAD_SCHEMA_VERSION]),
        position_names=np.asarray(POSITION_NAMES),
        amino_acid_alphabet=np.asarray(AA_ALPHABET),
        additive_categories=categories,
        additive_coef=additive_head["coef"],
        additive_intercept=additive_head["intercept"],
        additive_C=np.asarray([additive_c], dtype=np.float64),
        mint_feature_key=np.asarray([MINT_FEATURE_KEYS["frozen_mint_layer9"]]),
        mint_layer=np.asarray([9], dtype=np.int64),
        mint_pooling=np.asarray([POOLING_CONTRACT]),
        mint_mean=mint9_head["mean"],
        mint_scale=mint9_head["scale"],
        mint_coef=mint9_head["coef"],
        mint_intercept=mint9_head["intercept"],
        mint_C=np.asarray([mint9_c], dtype=np.float64),
        mint_layer33_feature_key=np.asarray([MINT_FEATURE_KEYS["frozen_mint_layer33_control"]]),
        mint_layer33_layer=np.asarray([33], dtype=np.int64),
        mint_layer33_pooling=np.asarray([POOLING_CONTRACT]),
        mint_layer33_mean=mint33_head["mean"],
        mint_layer33_scale=mint33_head["scale"],
        mint_layer33_coef=mint33_head["coef"],
        mint_layer33_intercept=mint33_head["intercept"],
        mint_layer33_C=np.asarray([mint33_c], dtype=np.float64),
    )
    os.chmod(path, 0o600)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }
    return {
        "schema_version": HEAD_SCHEMA_VERSION,
        "retention_labels_read": False,
        "archive": {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)},
        "arrays": arrays,
        "primary_mint_head": {
            "feature_key": MINT_FEATURE_KEYS["frozen_mint_layer9"],
            "layer": 9,
            "pooling": POOLING_CONTRACT,
        },
        "formulas": {
            "additive_6site": "sigmoid(one_hot_6x20 @ additive_coef + additive_intercept)",
            "frozen_mint_layer9": "sigmoid(((feature - mint_mean) / mint_scale) @ mint_coef + mint_intercept)",
            "frozen_mint_layer33_control": "sigmoid(((feature - mint_layer33_mean) / mint_layer33_scale) @ mint_layer33_coef + mint_layer33_intercept)",
        },
    }


def save_additive_deployment_head(
    path: Path,
    head: Mapping[str, np.ndarray],
    selected_c: float,
) -> dict[str, Any]:
    """Write the minimal scorer-facing six-position additive head."""

    weight = np.asarray(head["coef"], dtype=np.float64).reshape(6, len(AA_ALPHABET))
    bias = np.asarray(head["intercept"], dtype=np.float64).reshape(1)
    np.savez(
        path,
        schema_version=np.asarray([ADDITIVE_HEAD_SCHEMA_VERSION]),
        feature_names=np.asarray(POSITION_NAMES),
        residue_alphabet=np.asarray(AA_ALPHABET),
        weight=weight,
        bias=bias,
        selected_C=np.asarray([selected_c], dtype=np.float64),
        training_membership_sha256=np.asarray([EXPECTED_TRAIN_MEMBERSHIP_SHA256]),
    )
    os.chmod(path, 0o600)
    return {
        "schema_version": ADDITIVE_HEAD_SCHEMA_VERSION,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "feature_names": list(POSITION_NAMES),
        "residue_alphabet": "".join(AA_ALPHABET),
        "weight_shape": list(weight.shape),
        "selected_C": float(selected_c),
        "training_membership_sha256": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
    }


def save_mint9_deployment_head(
    path: Path,
    head: Mapping[str, np.ndarray],
    selected_c: float,
) -> dict[str, Any]:
    """Write the canonical generic-key contract used by MINT scorers."""

    np.savez(
        path,
        schema_version=np.asarray([MINT9_HEAD_SCHEMA_VERSION]),
        mint_mean=np.asarray(head["mean"], dtype=np.float64),
        mint_scale=np.asarray(head["scale"], dtype=np.float64),
        mint_coef=np.asarray(head["coef"], dtype=np.float64),
        mint_intercept=np.asarray(head["intercept"], dtype=np.float64).reshape(1),
        mint_layer=np.asarray([9], dtype=np.int64),
        mint_feature_key=np.asarray([MINT_FEATURE_KEYS["frozen_mint_layer9"]]),
        mint_pooling=np.asarray([POOLING_CONTRACT]),
        mint_C=np.asarray([selected_c], dtype=np.float64),
        training_membership_sha256=np.asarray([EXPECTED_TRAIN_MEMBERSHIP_SHA256]),
    )
    os.chmod(path, 0o600)
    return {
        "schema_version": MINT9_HEAD_SCHEMA_VERSION,
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "feature_key": MINT_FEATURE_KEYS["frozen_mint_layer9"],
        "layer": 9,
        "pooling": POOLING_CONTRACT,
        "selected_C": float(selected_c),
        "training_membership_sha256": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
    }


def verify_historical_additive_parity(
    path: str | Path,
    evaluation: pd.DataFrame,
    probability: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compare against the locked score column without opening outcomes."""

    path = Path(path).resolve()
    reference = pd.read_csv(
        path,
        usecols=["pair_uid", "library", "weak_site_logistic_probability"],
        dtype={"pair_uid": str, "library": str},
    )
    reference = reference.loc[reference["library"].eq(LIBRARY)].copy()
    reference["weak_site_logistic_probability"] = pd.to_numeric(
        reference["weak_site_logistic_probability"], errors="raise"
    ).astype(np.float64)
    observed = pd.DataFrame(
        {
            "pair_uid": evaluation["pair_uid"].astype(str),
            "replayed_probability": np.asarray(probability, dtype=np.float64),
        }
    )
    joined = observed.merge(
        reference[["pair_uid", "weak_site_logistic_probability"]],
        on="pair_uid",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    _require(len(joined) == EXPECTED_EVAL_ROWS, "historical LibA score membership changed")
    _require(bool(joined["_merge"].eq("both").all()), "historical LibA score IDs changed")
    delta = np.abs(
        joined["replayed_probability"].to_numpy(np.float64)
        - joined["weak_site_logistic_probability"].to_numpy(np.float64)
    )
    maximum = float(delta.max())
    _require(maximum <= tolerance, f"historical additive replay failed: {maximum} > {tolerance}")
    return {
        "status": "passed",
        "rows": int(len(joined)),
        "maximum_absolute_probability_difference": maximum,
        "mean_absolute_probability_difference": float(delta.mean()),
        "tolerance": float(tolerance),
        "reference_path": str(path),
        "reference_sha256": sha256_file(path),
        "reference_columns_read": [
            "pair_uid",
            "library",
            "weak_site_logistic_probability",
        ],
        "outcome_columns_read": [],
    }


def logistic_replay_probability(
    features: np.ndarray, head: Mapping[str, np.ndarray]
) -> np.ndarray:
    standardized = standardize(features, head["mean"], head["scale"])
    logit_value = standardized @ np.asarray(head["coef"], dtype=np.float64)
    logit_value += float(np.asarray(head["intercept"])[0])
    return 1.0 / (1.0 + np.exp(-logit_value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = _validate_output_dir(args.output_dir)
    config = read_config(args.config)
    selection_audit = validate_layer9_selection_audit(args.layer_selection_audit)
    feature_manifest = validate_feature_manifest(args.feature_manifest, args.features)
    if args.verify_archive_checksum:
        observed = sha256_file(args.features)
        _require(observed == feature_manifest["declared_archive_sha256"], "feature archive checksum changed")
        feature_manifest["verified_archive_sha256"] = observed

    header = load_label_safe_archive_header(args.features)
    primary, evaluation = prepare_liba_rows(header)
    plans, membership = build_fold_plans(primary)
    train_indices = primary["_cache_index"].to_numpy(dtype=int)
    eval_indices = evaluation["_cache_index"].to_numpy(dtype=int)
    train_code = encode_codes(designed_codes(primary))
    eval_code = encode_codes(designed_codes(evaluation))
    labels = primary["weak_label"].to_numpy(dtype=int)
    logistic_config = config["logistic"]
    c_grid = tuple(map(float, logistic_config["c_grid"]))

    validation_rows: list[dict[str, Any]] = []
    oof_blocks = []
    eval_blocks = []
    final_audits: dict[str, Any] = {}

    additive = cross_validate_logistic(
        "additive_6site",
        train_code,
        primary,
        plans,
        c_grid,
        standardize_features=False,
        tol=float(logistic_config["additive_tol"]),
        max_iter=int(logistic_config["max_iter"]),
    )
    validation_rows.extend(additive.validation_rows)
    oof_blocks.append(additive.oof)
    additive_probability, additive_head, additive_audit = fit_final_logistic(
        train_code,
        eval_code,
        labels,
        additive.selected_c,
        standardize_features=False,
        tol=float(logistic_config["additive_tol"]),
        max_iter=int(logistic_config["max_iter"]),
    )
    eval_blocks.append(
        pd.DataFrame(
            {
                "eval_row_id": evaluation["pair_uid"].astype(str),
                "model": "additive_6site",
                "seed": "fixed",
                "score": additive_probability,
            }
        )
    )
    final_audits["additive_6site"] = additive_audit

    mint_heads = {}
    mint_eval_features = {}
    for model_name, feature_key in MINT_FEATURE_KEYS.items():
        layer = load_layer_matrix(args.features, feature_key)
        train_features = layer[train_indices]
        eval_features = layer[eval_indices]
        del layer
        result = cross_validate_logistic(
            model_name,
            train_features,
            primary,
            plans,
            c_grid,
            standardize_features=True,
            tol=float(logistic_config["mint_tol"]),
            max_iter=int(logistic_config["max_iter"]),
        )
        if model_name == "frozen_mint_layer9":
            _require(
                result.selected_c == float(selection_audit["selected_C"]),
                "layer-9 C no longer reproduces the weak-only selection audit",
            )
        validation_rows.extend(result.validation_rows)
        oof_blocks.append(result.oof)
        probability, head, audit = fit_final_logistic(
            train_features,
            eval_features,
            labels,
            result.selected_c,
            standardize_features=True,
            tol=float(logistic_config["mint_tol"]),
            max_iter=int(logistic_config["max_iter"]),
        )
        eval_blocks.append(
            pd.DataFrame(
                {
                    "eval_row_id": evaluation["pair_uid"].astype(str),
                    "model": model_name,
                    "seed": "fixed",
                    "score": probability,
                }
            )
        )
        mint_heads[model_name] = (result.selected_c, head)
        mint_eval_features[model_name] = eval_features
        final_audits[model_name] = audit
        del train_features

    device = torch.device(args.device)
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    mlp_config = config["models"]["nonlinear_6site"]
    selected_epoch, mlp_selection_rows = select_mlp_epoch(
        train_code, primary, plans, mlp_config, device
    )
    validation_rows.extend(mlp_selection_rows)

    output.mkdir(parents=True, mode=0o700)
    checkpoint_dir = output / "mlp_checkpoints"
    checkpoint_dir.mkdir(mode=0o700)
    mlp_oof, mlp_eval, mlp_summaries, mlp_parity = fit_mlp_oof_and_final(
        train_code,
        eval_code,
        primary,
        evaluation,
        plans,
        mlp_config,
        selected_epoch,
        device,
        checkpoint_dir,
    )
    oof_blocks.append(mlp_oof)
    eval_blocks.append(mlp_eval)

    oof = pd.concat(oof_blocks, ignore_index=True)
    final = pd.concat(eval_blocks, ignore_index=True)
    aligned_oof = build_aligned_oof(oof)
    aligned_eval = build_aligned_eval(final)
    _require(tuple(final.columns) == ("eval_row_id", "model", "seed", "score"), "target-free prediction schema changed")

    heads_path = output / "deployment_heads.npz"
    heads_manifest = save_deployment_heads(
        heads_path,
        additive_head,
        additive.selected_c,
        mint_heads["frozen_mint_layer9"][1],
        mint_heads["frozen_mint_layer9"][0],
        mint_heads["frozen_mint_layer33_control"][1],
        mint_heads["frozen_mint_layer33_control"][0],
    )
    additive_replay = logistic_replay_probability(eval_code, additive_head)
    mint9_replay = logistic_replay_probability(
        mint_eval_features["frozen_mint_layer9"], mint_heads["frozen_mint_layer9"][1]
    )
    mint33_replay = logistic_replay_probability(
        mint_eval_features["frozen_mint_layer33_control"], mint_heads["frozen_mint_layer33_control"][1]
    )
    self_parity = {}
    for name, replay in (
        ("additive_6site", additive_replay),
        ("frozen_mint_layer9", mint9_replay),
        ("frozen_mint_layer33_control", mint33_replay),
    ):
        expected = final.loc[final["model"].eq(name), "score"].to_numpy(float)
        delta = np.abs(replay - expected)
        _require(float(delta.max()) <= 2e-12, f"serialized {name} head failed self-parity")
        self_parity[name] = {
            "rows": len(delta),
            "maximum_absolute_probability_difference": float(delta.max()),
        }
    heads_manifest["serialized_head_self_parity"] = self_parity
    additive_head_path = output / "additive_deployment_head.npz"
    additive_head_manifest = save_additive_deployment_head(
        additive_head_path, additive_head, additive.selected_c
    )
    mint9_head_path = output / "mint_layer9_deployment_head.npz"
    mint9_head_manifest = save_mint9_deployment_head(
        mint9_head_path,
        mint_heads["frozen_mint_layer9"][1],
        mint_heads["frozen_mint_layer9"][0],
    )
    historical_additive_parity = verify_historical_additive_parity(
        args.historical_additive_predictions,
        evaluation,
        additive_replay,
    )
    heads_manifest["additive_scorer_head"] = additive_head_manifest
    heads_manifest["primary_mint_scorer_head"] = mint9_head_manifest
    heads_manifest["historical_additive_108_replay"] = historical_additive_parity
    heads_manifest_path = output / "deployment_heads.json"
    _write_json(heads_manifest_path, heads_manifest)

    weak_row_metadata = primary[
        [
            "pair_uid",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "peptide_design_code",
            "affibody_design_code",
        ]
    ].rename(columns={"pair_uid": "row_id"})
    evaluation_roster_metadata = evaluation[
        [
            "pair_uid",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "peptide_design_code",
            "affibody_design_code",
        ]
    ].rename(columns={"pair_uid": "eval_row_id"})
    paths = {
        "weak_oof_predictions.csv.gz": oof,
        "weak_oof_aligned.csv.gz": aligned_oof,
        "evaluation_predictions.csv": final,
        "evaluation_predictions_aligned.csv": aligned_eval,
        "fold_membership.csv.gz": membership,
        "weak_row_metadata.csv.gz": weak_row_metadata,
        "evaluation_roster_metadata.csv": evaluation_roster_metadata,
        "weak_validation_records.csv": pd.DataFrame(validation_rows),
        "mlp_training_summary.csv": pd.DataFrame(mlp_summaries),
    }
    for filename, frame in paths.items():
        path = output / filename
        frame.to_csv(path, index=False, compression="gzip" if filename.endswith(".gz") else None)
        os.chmod(path, 0o600)

    oof_metrics = []
    for (model, seed), block in oof.groupby(["model", "seed"], sort=True):
        oof_metrics.append(
            {
                "model": model,
                "seed": seed,
                **binary_metrics(block["weak_label"], block["probability"]),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "retention_labels_read": False,
        "retention_outcomes_read": False,
        "weak_training": {
            "rows": EXPECTED_TRAIN_ROWS,
            "positive": EXPECTED_POSITIVE,
            "negative": EXPECTED_NEGATIVE,
            "membership_sha256": EXPECTED_TRAIN_MEMBERSHIP_SHA256,
        },
        "evaluation": {"rows": EXPECTED_EVAL_ROWS, "labels_present": False},
        "validation": {
            "folds": FOLDS,
            "split_seed": SPLIT_SEED,
            "oof_unique_rows": EXPECTED_OOF_ROWS,
            "oof_positive": EXPECTED_OOF_POSITIVE,
            "row_fold_sha256": row_fold_sha256(aligned_oof),
            "selection_metric": "weak-validation log loss only",
        },
        "selected_configuration": {
            "additive_6site_C": additive.selected_c,
            "frozen_mint_layer9_C": mint_heads["frozen_mint_layer9"][0],
            "frozen_mint_layer33_control_C": mint_heads["frozen_mint_layer33_control"][0],
            "nonlinear_6site_epoch": selected_epoch,
            "nonlinear_6site_final_seeds": list(map(int, mlp_config["final_seeds"])),
        },
        "oof_metrics": oof_metrics,
        "final_model_audit": final_audits,
        "mlp_checkpoint_self_parity": mlp_parity,
        "historical_additive_108_replay": historical_additive_parity,
        "runtime_seconds": float(time.time() - started),
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)

    output_records = {}
    for path in sorted(value for value in output.rglob("*") if value.is_file()):
        relative = str(path.relative_to(output))
        output_records[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retention_labels_read": False,
        "retention_outcomes_read": False,
        "source_npz_members_read": list(SAFE_ARCHIVE_MEMBERS)
        + list(MINT_FEATURE_KEYS.values()),
        "forbidden_supervision_members_read": [],
        "inputs": {
            "config": {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)},
            "features": {
                "path": str(Path(args.features).resolve()),
                "declared_sha256": feature_manifest["declared_archive_sha256"],
                "full_checksum_verified": bool(args.verify_archive_checksum),
            },
            "feature_manifest": feature_manifest,
            "weak_layer_selection_audit": selection_audit,
            "historical_additive_predictions": {
                "path": str(Path(args.historical_additive_predictions).resolve()),
                "sha256": historical_additive_parity["reference_sha256"],
                "columns_read": historical_additive_parity["reference_columns_read"],
                "outcome_columns_read": [],
                "used_for_fitting_or_selection": False,
            },
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "outputs": output_records,
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"output": str(output), **summary["selected_configuration"]}, sort_keys=True))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "downstream/AffibodyMHC/configs/liba_common_oof_sequence_v1.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_multilayer_v1/merged/mint_multilayer_chain_mean_features.npz",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_multilayer_v1/merged/manifest.json",
    )
    parser.add_argument(
        "--layer-selection-audit",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/mint_multilayer_eval_parallel_by_library_v1/LibA_layer09/selected_config.json",
    )
    parser.add_argument(
        "--historical-additive-predictions",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/selection_weak_site_v2/predictions.csv",
        help="Locked score column used only for post-fit 108-row additive replay.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verify-archive-checksum", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
