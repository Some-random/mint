#!/usr/bin/env python
"""Build a retention-blind LibB score ensemble from matched weak-label OOF rows.

The three locked inputs are the seven-site additive classifier, the frozen
MINT layer-5 logistic readout, and the selected StaB designed-residue readout.
The first two out-of-fold (OOF) score vectors are reconstructed on the same
three diagonal double-cold folds used by the StaB experiment.  No direct
retention file is accepted or read.

Two deliberately small ensemble families are compared:

* equal-weight mean-logit ensembles over every non-empty member subset; and
* a non-negative L2-regularized logistic stacker over all three members.

The stacker's OOF score is second-level cross-fitted: for each held canonical
fold, its regularization is selected using only the other two folds and the
stacker is then fitted without the held fold.  The final deployment stacker is
fitted to all common OOF rows only after its L2 value is selected by three-fold
cross-fitting.  The primary metric is macro average precision across peptides
that have both weak-label classes.

These scores concern the constructed weak PN labels.  They are not calibrated
probabilities of retention >= 75 and must not be described that way.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_selection_weak_baseline as additive
from downstream.AffibodyMHC import finetune_mint_selection_matched as mint_matched
from downstream.AffibodyMHC.code_only_baseline import sha256_file


SCHEMA_VERSION = "libb-weak-oof-ensemble-v1"
EXPECTED_ROWS = 30_648
EXPECTED_POSITIVE = 23_725
EXPECTED_OOF_ROWS = 10_181
EXPECTED_OOF_POSITIVE = 7_939
EXPECTED_FOLD_ROWS = {0: 3_579, 1: 2_763, 2: 3_839}
EXPECTED_MEMBERSHIP_HASHES = {
    0: "3fe08579a40ebcce63a0b1500e8eaaae5832e49efe3e574ff519fc33e54a966c",
    1: "7b128d05c3e0c8664853c3ad47dbc842c7d45b5d49f38d67765a864718dc0b2a",
    2: "93e501084970f120ff01d54c4a9311d193f7116cdd1cb8f7b2aeea64bd6b316d",
}
MEMBERS = ("additive_7site", "mint_layer5", "stab_designed_ordered")
MINT_C = 0.1
ADDITIVE_C = 10.0
PROBABILITY_CLIP = 1e-7
STACK_L2_GRID = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
STACK_MAX_ITER = 2_000
STACK_FTOL = 1e-12

DEFAULT_LABELS = (
    REPO_ROOT
    / "private_data/experiments/esmfold2_libb_readout_cv_v1/training_labels.csv"
)
DEFAULT_FOLDS = (
    REPO_ROOT
    / "private_data/experiments/mint_multilayer_eval_aggregate_v1/fold_membership.csv"
)
DEFAULT_SEQUENCES = (
    REPO_ROOT
    / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1/"
    "current_sequence_records.json"
)
DEFAULT_MINT_DIR = REPO_ROOT / "private_data/derived/libb_mint_layer05_global_opaque_v1"
DEFAULT_STAB_OOF = (
    REPO_ROOT
    / "private_data/experiments/stab_libb_provider_revision_120_readout_pilot_v1/"
    "stab_designed_ordered/weak_validation_predictions.csv.gz"
)
DEFAULT_OUTPUT = REPO_ROOT / "private_data/experiments/libb_weak_ensemble_selection_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _write_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    frame.to_csv(path, index=False, compression=compression)
    os.chmod(path, 0o600)


def _membership_sha256(row_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, row_ids)))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


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
    _require(len(score_columns) >= 1, "mean-logit ensemble has no members")
    logits = [clipped_logit(values) for values in score_columns]
    _require(len({len(values) for values in logits}) == 1, "member lengths differ")
    return sigmoid(np.mean(np.stack(logits, axis=1), axis=1))


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
    _require(len(frame) > 0, "metric frame is empty")
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
    output.update(
        macro_within_peptide_metrics(frame["peptide_id"], labels, probability)
    )
    return output


def _load_base_tables(
    labels_path: Path, folds_path: Path, sequences_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(labels_path, dtype={"row_id": str})
    _require(
        tuple(labels.columns)
        == ("row_id", "weak_label", "peptide_id", "affibody_id", "chain1_sha256", "chain2_sha256"),
        "training-label schema changed",
    )
    _require(len(labels) == EXPECTED_ROWS, "weak row count changed")
    _require(int(labels["weak_label"].sum()) == EXPECTED_POSITIVE, "weak positive count changed")
    _require(not bool(labels["row_id"].duplicated().any()), "duplicate weak row ID")

    with sequences_path.open("r", encoding="utf-8") as handle:
        sequence_payload = json.load(handle)
    sequence_rows = pd.DataFrame(sequence_payload["rows"])
    sequence_rows = sequence_rows.loc[sequence_rows["split"].eq("train")].copy()
    _require(len(sequence_rows) == EXPECTED_ROWS, "sequence weak row count changed")
    sequence_rows = sequence_rows[["row_id", "peptide_code", "affibody_code"]]
    labels = labels.merge(sequence_rows, on="row_id", how="inner", validate="one_to_one")
    _require(len(labels) == EXPECTED_ROWS, "weak labels and sequence rows differ")
    _require(bool(labels["peptide_code"].str.len().eq(2).all()), "bad peptide code")
    _require(bool(labels["affibody_code"].str.len().eq(5).all()), "bad Affibody code")

    membership = pd.read_csv(folds_path, dtype={"pair_uid": str})
    membership = membership.loc[membership["library"].eq("LibB")].copy()
    membership = membership.rename(columns={"pair_uid": "row_id"})
    _require(len(membership) == EXPECTED_ROWS * 3, "fold-membership row count changed")
    _require(set(membership["fold"].astype(int)) == {0, 1, 2}, "fold IDs changed")
    _require(set(membership["role"].astype(str)) == {"train", "guard", "validation"}, "fold roles changed")
    for fold, expected_rows in EXPECTED_FOLD_ROWS.items():
        validation = membership.loc[
            membership["fold"].eq(fold) & membership["role"].eq("validation")
        ]
        _require(len(validation) == expected_rows, f"fold {fold} validation count changed")
        _require(
            _membership_sha256(validation["row_id"]) == EXPECTED_MEMBERSHIP_HASHES[fold],
            f"fold {fold} validation membership changed",
        )
    return labels.sort_values("row_id").reset_index(drop=True), membership


def _fold_indices(
    labels: pd.DataFrame, membership: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    records = membership.loc[membership["fold"].eq(int(fold)), ["row_id", "role"]]
    role = labels[["row_id"]].merge(records, on="row_id", validate="one_to_one")["role"]
    return (
        np.flatnonzero(role.eq("train").to_numpy()),
        np.flatnonzero(role.eq("validation").to_numpy()),
    )


def build_additive_oof(labels: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    codes = pd.DataFrame({"pep": labels["peptide_code"], "aff": labels["affibody_code"]})
    raw = additive.site_code_matrix(codes, "LibB")
    alphabet = tuple(additive.AA_ALPHABET)
    lookup = {residue: index for index, residue in enumerate(alphabet)}
    features = np.zeros((len(raw), raw.shape[1] * len(alphabet)), dtype=np.float64)
    for row_index in range(len(raw)):
        for position in range(raw.shape[1]):
            residue = str(raw[row_index, position])
            _require(residue in lookup, f"noncanonical designed residue {residue!r}")
            features[row_index, position * len(alphabet) + lookup[residue]] = 1.0
    observed = labels["weak_label"].to_numpy(dtype=int)
    blocks = []
    for fold in range(3):
        train, validation = _fold_indices(labels, membership, fold)
        model = additive.make_logistic(ADDITIVE_C)
        model.fit(features[train], observed[train])
        _require(int(model.n_iter_[0]) < int(model.max_iter), "additive fit did not converge")
        blocks.append(
            pd.DataFrame(
                {
                    "row_id": labels.iloc[validation]["row_id"].to_numpy(),
                    "fold": fold,
                    "weak_label": observed[validation],
                    "additive_7site": model.predict_proba(features[validation])[:, 1],
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def build_mint_oof(
    labels: pd.DataFrame, membership: pd.DataFrame, mint_dir: Path
) -> pd.DataFrame:
    metadata = pd.read_csv(mint_dir / "metadata.csv", dtype={"row_id": str})
    _require(tuple(metadata.columns) == ("row_index", "row_id", "split"), "MINT metadata changed")
    _require(not bool(metadata["row_id"].duplicated().any()), "duplicate MINT row ID")
    train_metadata = metadata.loc[metadata["split"].eq("train")].copy()
    aligned = labels[["row_id"]].merge(
        train_metadata[["row_id", "row_index"]], on="row_id", how="left", validate="one_to_one"
    )
    _require(not bool(aligned["row_index"].isna().any()), "MINT archive misses weak rows")
    cache_indices = aligned["row_index"].to_numpy(dtype=np.int64)
    features = np.load(mint_dir / "mint_layer05_global.npy", mmap_mode="r")
    _require(features.ndim == 2 and features.shape[1] == 2560, "MINT feature shape changed")
    observed = labels["weak_label"].to_numpy(dtype=int)
    blocks = []
    for fold in range(3):
        train, validation = _fold_indices(labels, membership, fold)
        train_values = np.asarray(features[cache_indices[train]], dtype=np.float32)
        mean, scale = mint_matched.fit_standardizer(train_values)
        x_train = mint_matched.standardize(train_values, mean, scale)
        del train_values
        x_validation = mint_matched.standardize(
            np.asarray(features[cache_indices[validation]], dtype=np.float32), mean, scale
        )
        model = mint_matched.fit_logistic(x_train, observed[train], MINT_C)
        probability = model.predict_proba(x_validation)[:, 1]
        blocks.append(
            pd.DataFrame(
                {
                    "row_id": labels.iloc[validation]["row_id"].to_numpy(),
                    "fold": fold,
                    "weak_label": observed[validation],
                    "mint_layer5": probability,
                }
            )
        )
        del x_train, x_validation, model
    return pd.concat(blocks, ignore_index=True)


def load_stab_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": str})
    _require(
        {"model", "fold", "row_id", "weak_label", "probability"}.issubset(frame.columns),
        "StaB OOF schema changed",
    )
    frame = frame.loc[frame["model"].eq("stab_designed_ordered")].copy()
    _require(len(frame) == EXPECTED_OOF_ROWS, "StaB OOF row count changed")
    _require(not bool(frame["row_id"].duplicated().any()), "duplicate StaB OOF row")
    return frame[["row_id", "fold", "weak_label", "probability"]].rename(
        columns={"probability": "stab_designed_ordered"}
    )


def align_oof(
    labels: pd.DataFrame,
    additive_oof: pd.DataFrame,
    mint_oof: pd.DataFrame,
    stab_oof: pd.DataFrame,
) -> pd.DataFrame:
    key = ["row_id", "fold", "weak_label"]
    combined = additive_oof.merge(mint_oof, on=key, validate="one_to_one")
    combined = combined.merge(stab_oof, on=key, validate="one_to_one")
    combined = combined.merge(
        labels[["row_id", "peptide_id", "affibody_id"]],
        on="row_id",
        validate="one_to_one",
    )
    _require(len(combined) == EXPECTED_OOF_ROWS, "common OOF row count changed")
    _require(int(combined["weak_label"].sum()) == EXPECTED_OOF_POSITIVE, "common OOF positive count changed")
    for name in MEMBERS:
        value = combined[name].to_numpy(dtype=float)
        _require(bool(np.isfinite(value).all()), f"{name} contains non-finite scores")
        _require(bool(((value >= 0.0) & (value <= 1.0)).all()), f"{name} score is outside [0,1]")
    return combined.sort_values("row_id").reset_index(drop=True)


def standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return mean, scale


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
) -> dict[str, object]:
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
    prevalence = min(max(prevalence, 1e-7), 1.0 - 1e-7)
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


def predict_stacker(model: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    linear = float(model["intercept"]) + np.asarray(features, dtype=np.float64).dot(coefficient)
    return sigmoid(linear)


def _stack_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([clipped_logit(frame[name]) for name in MEMBERS])


def _fit_stacker_partition(
    train: pd.DataFrame, prediction: pd.DataFrame, l2: float
) -> tuple[np.ndarray, dict[str, object]]:
    train_raw = _stack_features(train)
    prediction_raw = _stack_features(prediction)
    mean, scale = standardizer(train_raw)
    train_x = (train_raw - mean) / scale
    prediction_x = (prediction_raw - mean) / scale
    model = fit_nonnegative_stacker(
        train_x,
        train["weak_label"],
        l2,
        sample_weight=peptide_equal_weights(train["peptide_id"]),
    )
    score = predict_stacker(model, prediction_x)
    audit = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": model["intercept"],
        "coefficient": np.asarray(model["coefficient"]).tolist(),
        "l2": model["l2"],
        "iterations": model["iterations"],
        "objective": model["objective"],
    }
    return score, audit


def _choose_l2_two_fold(train: pd.DataFrame) -> tuple[float, list[dict[str, object]]]:
    folds = sorted(train["fold"].unique().tolist())
    _require(len(folds) == 2, "nested L2 selection requires two training folds")
    records = []
    for l2 in STACK_L2_GRID:
        prediction_blocks = []
        for held in folds:
            fit = train.loc[train["fold"].ne(held)]
            validation = train.loc[train["fold"].eq(held)]
            score, _ = _fit_stacker_partition(fit, validation, l2)
            block = validation[["row_id", "peptide_id", "weak_label"]].copy()
            block["score"] = score
            prediction_blocks.append(block)
        prediction = pd.concat(prediction_blocks, ignore_index=True)
        metrics = score_metrics(prediction, prediction["score"])
        records.append({"l2": float(l2), **metrics})
    selected = max(
        records,
        key=lambda row: (
            float(row["within_peptide_ap"]),
            -float(row["pooled_log_loss"]),
            float(row["l2"]),
        ),
    )
    return float(selected["l2"]), records


def cross_fitted_stacker(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = pd.Series(index=frame.index, dtype=float)
    audits = []
    for held in sorted(frame["fold"].unique().tolist()):
        train = frame.loc[frame["fold"].ne(held)].copy()
        validation = frame.loc[frame["fold"].eq(held)].copy()
        selected_l2, grid = _choose_l2_two_fold(train)
        score, fit = _fit_stacker_partition(train, validation, selected_l2)
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


def fit_final_stacker(frame: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    grid = []
    for l2 in STACK_L2_GRID:
        output = pd.Series(index=frame.index, dtype=float)
        for held in sorted(frame["fold"].unique().tolist()):
            train = frame.loc[frame["fold"].ne(held)]
            validation = frame.loc[frame["fold"].eq(held)]
            score, _ = _fit_stacker_partition(train, validation, l2)
            output.loc[validation.index] = score
        metrics = score_metrics(frame, output.to_numpy(dtype=float))
        grid.append({"l2": float(l2), **metrics})
    selected = max(
        grid,
        key=lambda row: (
            float(row["within_peptide_ap"]),
            -float(row["pooled_log_loss"]),
            float(row["l2"]),
        ),
    )
    _, fit = _fit_stacker_partition(frame, frame.iloc[:1], float(selected["l2"]))
    fit["selection_rule"] = (
        "maximum cross-fitted macro within-peptide AP; then lower pooled log loss; "
        "then stronger L2"
    )
    return fit, grid


def evaluate_candidates(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    rows = []
    for size in range(1, len(MEMBERS) + 1):
        for members in itertools.combinations(MEMBERS, size):
            name = "mean_logit__" + "__".join(members)
            score = mean_logit_score([frame[member] for member in members])
            frame[name] = score
            rows.append(
                {
                    "candidate": name,
                    "family": "single" if len(members) == 1 else "equal_mean_logit",
                    "members": "+".join(members),
                    "member_count": len(members),
                    **score_metrics(frame, score),
                }
            )

    stack_score, outer_audit = cross_fitted_stacker(frame)
    stack_name = "cross_fitted_nonnegative_stack__all3"
    frame[stack_name] = stack_score
    rows.append(
        {
            "candidate": stack_name,
            "family": "cross_fitted_nonnegative_logistic",
            "members": "+".join(MEMBERS),
            "member_count": len(MEMBERS),
            **score_metrics(frame, stack_score),
        }
    )
    final_stack, final_grid = fit_final_stacker(frame)
    return pd.DataFrame(rows), outer_audit, final_stack, final_grid


def select_candidates(metrics: pd.DataFrame) -> dict[str, object]:
    order = metrics.sort_values(
        ["within_peptide_ap", "pooled_log_loss", "member_count", "candidate"],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ensembles = order.loc[order["member_count"].gt(1)].reset_index(drop=True)
    _require(not ensembles.empty, "no ensemble candidate")
    best = order.iloc[0]
    best_ensemble = ensembles.iloc[0]
    return {
        "selection_data": "selection-derived weak binder/non-binder labels only",
        "retention_labels_read": False,
        "primary_metric": "macro within-peptide average precision",
        "evaluable_peptide_rule": "include peptides with both weak-label classes",
        "candidate_tie_break": "lower pooled log loss, then fewer members, then candidate name",
        "best_candidate": best.to_dict(),
        "best_ensemble_candidate": best_ensemble.to_dict(),
    }


def render_report(selection: Mapping[str, object], metrics: pd.DataFrame, final_stack: Mapping[str, object]) -> str:
    lines = [
        "# LibB retention-blind weak-label ensemble selection",
        "",
        "This experiment aligned all component scores to the same 10,181 three-fold ",
        "double-cold validation pairs. Neither the 120 retention measurements nor their ",
        "binder labels were read. The primary comparison is average precision calculated ",
        "separately within each weak-data peptide containing both classes and then averaged ",
        "across those peptides.",
        "",
        "The score is a weak-selection score, not a calibrated probability that retention ",
        "will be at least 75%.",
        "",
        "| Candidate | Members | Within-peptide AP | Within-peptide AUROC | Pooled log loss |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics.sort_values("within_peptide_ap", ascending=False).itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.members} | {row.within_peptide_ap:.6f} | "
            f"{row.within_peptide_auroc:.6f} | {row.pooled_log_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Locked weak-only decision",
            "",
            f"Best candidate: `{selection['best_candidate']['candidate']}`.",
            "",
            f"Best multi-member candidate: `{selection['best_ensemble_candidate']['candidate']}`.",
            "",
            "The regularized stacker is restricted to non-negative component coefficients. ",
            "Its outer score is cross-fitted at the second level, and each outer fold chooses ",
            "regularization using only the other two folds. The deployment stacker's final ",
            "L2 and coefficients are locked below from weak OOF data only.",
            "",
            f"Final stacker L2: `{final_stack['l2']}`.",
            "",
            "Final standardized-logit coefficients: `{} `.".format(
                ", ".join(
                    f"{member}={coefficient:.8g}"
                    for member, coefficient in zip(MEMBERS, final_stack["coefficient"])
                )
            ),
            "",
            "Only after this selection file is locked may the corrected retention panel be ",
            "opened for a retrospective audit. Because earlier component results on that ",
            "panel are already known, that audit is not a fresh confirmatory test.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    paths = {
        "training_labels": Path(args.training_labels).resolve(),
        "fold_membership": Path(args.fold_membership).resolve(),
        "sequence_records": Path(args.sequence_records).resolve(),
        "mint_metadata": Path(args.mint_dir).resolve() / "metadata.csv",
        "mint_features": Path(args.mint_dir).resolve() / "mint_layer05_global.npy",
        "stab_oof": Path(args.stab_oof).resolve(),
    }
    for path in paths.values():
        _require(path.is_file(), f"missing input {path}")
    output = Path(args.output_dir).resolve()
    _require(output.parent != output, "invalid output directory")
    _require(not output.exists(), "output directory exists; refusing overwrite")
    _require("private_data" in output.parts, "output must remain under private_data")

    labels, membership = _load_base_tables(
        paths["training_labels"], paths["fold_membership"], paths["sequence_records"]
    )
    additive_oof = build_additive_oof(labels, membership)
    mint_oof = build_mint_oof(labels, membership, Path(args.mint_dir).resolve())
    stab_oof = load_stab_oof(paths["stab_oof"])
    frame = align_oof(labels, additive_oof, mint_oof, stab_oof)
    metrics, stack_outer, final_stack, final_grid = evaluate_candidates(frame)
    selection = select_candidates(metrics)

    output.mkdir(parents=True, mode=0o700)
    prediction_columns = [
        "row_id", "fold", "weak_label", "peptide_id", "affibody_id", *MEMBERS,
        *metrics["candidate"].tolist(),
    ]
    prediction_path = output / "matched_oof_predictions.csv.gz"
    _write_csv(prediction_path, frame[prediction_columns], compression="gzip")
    metric_path = output / "candidate_metrics.csv"
    _write_csv(metric_path, metrics.sort_values("candidate").reset_index(drop=True))
    outer_path = output / "stacker_crossfit_audit.json"
    _write_json(outer_path, stack_outer)
    final_grid_path = output / "stacker_final_l2_grid.json"
    _write_json(final_grid_path, final_grid)
    stack_path = output / "stacker_locked_parameters.json"
    _write_json(
        stack_path,
        {
            "schema_version": SCHEMA_VERSION,
            "members_in_coefficient_order": list(MEMBERS),
            "retention_labels_read": False,
            **final_stack,
        },
    )
    selection_path = output / "locked_weak_selection.json"
    _write_json(selection_path, {"schema_version": SCHEMA_VERSION, **selection})
    report_path = output / "report.md"
    report_path.write_text(render_report(selection, metrics, final_stack), encoding="utf-8")
    os.chmod(report_path, 0o600)

    outputs = [prediction_path, metric_path, outer_path, final_grid_path, stack_path, selection_path, report_path]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "retention_labels_read": False,
        "weak_rows": EXPECTED_ROWS,
        "common_oof_rows": EXPECTED_OOF_ROWS,
        "common_oof_positive": EXPECTED_OOF_POSITIVE,
        "fold_rows": EXPECTED_FOLD_ROWS,
        "members": {
            "additive_7site": {"C": ADDITIVE_C, "folds": 3, "class_weight": "balanced"},
            "mint_layer5": {"layer": 5, "C": MINT_C, "folds": 3, "class_weight": "balanced"},
            "stab_designed_ordered": {"source": "locked pilot OOF", "folds": 3},
        },
        "stacker": {
            "members": list(MEMBERS),
            "input": "clipped component logits standardized on stacker-training rows only",
            "coefficient_bounds": "non-negative",
            "training_loss": "peptide-equal-weighted logistic loss plus L2 on coefficients",
            "l2_grid": list(STACK_L2_GRID),
            "primary_selection_metric": "macro within-peptide average precision",
            "crossfit_note": (
                "second-level cross-fitting over already OOF component scores; not a full "
                "nested refit of every base representation"
            ),
        },
        "selection": selection,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "outputs": {
            path.name: {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        },
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    print(json.dumps(selection, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--fold-membership", default=str(DEFAULT_FOLDS))
    parser.add_argument("--sequence-records", default=str(DEFAULT_SEQUENCES))
    parser.add_argument("--mint-dir", default=str(DEFAULT_MINT_DIR))
    parser.add_argument("--stab-oof", default=str(DEFAULT_STAB_OOF))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
