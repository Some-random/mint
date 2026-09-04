#!/usr/bin/env python
"""Build retention-blind, row-aligned OOF scores for LibB base predictors.

The historical seven-site additive model used five identity-separated folds and
did not save per-row weak-validation predictions.  The selected frozen MINT
layer-5 experiment saved fold summaries but likewise omitted per-row scores.
This utility refits both already-selected configurations on the shared three-
fold LibB contract used by ESMFold2, RDE, StaB, and the MINT LoRA experiment.

Only selection-derived training labels are opened.  The final 120-row table is
label-free.  Both models are refitted on all eligible weak-label rows; the
additive model is scored from canonical sequences and the MINT model is scored
from the 120 layer-5 feature rows now present in the multilayer cache.  The
legacy one-row import remains a guarded fallback for an older 119-row cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.evaluate_selection_weak_baseline import (
    make_encoder,
    make_logistic,
)
from downstream.AffibodyMHC.finetune_mint_selection_matched import (
    fit_logistic,
    fit_standardizer,
    standardize,
)


SCHEMA_VERSION = "libb-common-oof-base-predictions-v1"
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
FOLDS = 3
SPLIT_SEED = 17
ADDITIVE_C = 10.0
MINT_C = 0.1
MINT_KEY = "mint_layer_05_chain_mean"
MISSING_ROW_ID = "6e8454b1ba8587952e0d"
EXPECTED_TRAIN_ROWS = 30_648
EXPECTED_POSITIVE = 23_725
EXPECTED_NEGATIVE = 6_923
EXPECTED_EVAL_ROWS = 120
EXPECTED_OOF_ROWS = 10_181
EXPECTED_OOF_POSITIVE = 7_939


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _membership_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _row_fold_sha256(frame: pd.DataFrame) -> str:
    selected = frame[["row_id", "fold"]].drop_duplicates().sort_values("row_id")
    payload = "\n".join(
        selected["row_id"].astype(str) + "\t" + selected["fold"].astype(str)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _stable_bin(value: str, axis: str) -> int:
    payload = f"{axis}|{SPLIT_SEED}|{value}".encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % FOLDS


def _fold_role(chain1_sha256: str, chain2_sha256: str, fold: int) -> str:
    peptide_held = _stable_bin(chain1_sha256, "peptide") == int(fold)
    affibody_held = _stable_bin(chain2_sha256, "affibody") == int(fold)
    if peptide_held and affibody_held:
        return "validation"
    if not peptide_held and not affibody_held:
        return "train"
    return "guard"


def _load_rows(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict) and isinstance(payload.get("rows"), list), "bad rows JSON")
    rows = pd.DataFrame(payload["rows"])
    expected = {
        "row_index",
        "row_id",
        "split",
        "chain1_sequence",
        "chain2_sequence",
        "sequence_pair_sha256",
    }
    _require(expected.issubset(rows.columns), "canonical rows schema changed")
    _require(len(rows) == EXPECTED_TRAIN_ROWS + EXPECTED_EVAL_ROWS, "canonical row count changed")
    _require(not bool(rows["row_id"].duplicated().any()), "duplicate canonical row ID")
    _require(int(rows["split"].eq("train").sum()) == EXPECTED_TRAIN_ROWS, "training split changed")
    _require(int(rows["split"].eq("eval").sum()) == EXPECTED_EVAL_ROWS, "evaluation split changed")
    return rows


def _load_labels(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    expected = [
        "row_id",
        "weak_label",
        "peptide_id",
        "affibody_id",
        "chain1_sha256",
        "chain2_sha256",
    ]
    _require(header == expected, "training-label schema changed")
    _require(not any("retention" in value.lower() for value in header), "retention field in training labels")
    labels = pd.read_csv(path, dtype={column: str for column in expected if column != "weak_label"})
    labels["weak_label"] = pd.to_numeric(labels["weak_label"], errors="raise").astype(int)
    labels = labels.sort_values("row_id").reset_index(drop=True)
    _require(len(labels) == EXPECTED_TRAIN_ROWS, "training-label count changed")
    _require(int(labels["weak_label"].sum()) == EXPECTED_POSITIVE, "positive count changed")
    _require(int(labels["weak_label"].eq(0).sum()) == EXPECTED_NEGATIVE, "negative count changed")
    _require(not bool(labels["row_id"].duplicated().any()), "duplicate training-label row ID")
    return labels


def _additive_codes(rows: pd.DataFrame) -> np.ndarray:
    values = []
    for row in rows.itertuples(index=False):
        chain1 = str(row.chain1_sequence)
        chain2 = str(row.chain2_sequence)
        _require(len(chain1) == 270 and len(chain2) == 58, "model sequence length changed")
        peptide = chain1[-9:]
        _require(peptide[:3] == "SLL" and peptide[5:] == "ITQV", "LibB peptide scaffold changed")
        code = [peptide[3], peptide[4]] + [chain2[index] for index in (5, 9, 12, 13, 16)]
        _require(set(code).issubset(AA_ALPHABET), "noncanonical designed residue")
        values.append(code)
    return np.asarray(values, dtype=str)


def _fold_membership(labels: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for fold in range(FOLDS):
        block = labels[["row_id", "weak_label", "chain1_sha256", "chain2_sha256"]].copy()
        block.insert(0, "fold", fold)
        block["role"] = [
            _fold_role(chain1, chain2, fold)
            for chain1, chain2 in zip(block["chain1_sha256"], block["chain2_sha256"])
        ]
        blocks.append(block)
    membership = pd.concat(blocks, ignore_index=True)
    _require(set(membership["role"]) == {"train", "guard", "validation"}, "fold roles changed")
    return membership


def _load_mint_features(
    archive_path: Path, requested_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    # Deliberately touch exactly two NPZ members.  The legacy archive contains
    # historical sidecars that are outside this task's data contract.
    with np.load(archive_path, allow_pickle=False) as archive:
        _require("pair_uid" in archive.files and MINT_KEY in archive.files, "MINT archive keys changed")
        available_ids = np.asarray(archive["pair_uid"]).astype(str)
        _require(len(set(available_ids.tolist())) == len(available_ids), "duplicate MINT pair UID")
        requested = set(requested_ids)
        # Preserve the historical cache order.  liblinear at the selected
        # tolerance can differ by ~1e-4 in probability if rows are permuted.
        found = [value for value in available_ids.tolist() if value in requested]
        indices = np.asarray(
            [index for index, value in enumerate(available_ids.tolist()) if value in requested],
            dtype=np.int64,
        )
        source = archive[MINT_KEY]
        _require(source.ndim == 2 and source.shape[1] == 2560, "MINT layer-5 shape changed")
        features = np.asarray(source[indices], dtype=np.float32)
    _require(bool(np.isfinite(features).all()), "non-finite MINT feature")
    return features, found


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(labels)),
        "positive": int(np.sum(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def _fit_oof(
    labels: pd.DataFrame,
    mint_labels: pd.DataFrame,
    membership: pd.DataFrame,
    additive: np.ndarray,
    mint: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = []
    summary: dict[str, Any] = {"folds": {}}
    y = labels["weak_label"].to_numpy(dtype=int)
    mint_y = mint_labels["weak_label"].to_numpy(dtype=int)
    for fold in range(FOLDS):
        roles = membership.loc[membership["fold"].eq(fold)].set_index("row_id")["role"]
        role = labels["row_id"].map(roles)
        train = np.flatnonzero(role.eq("train").to_numpy())
        validation = np.flatnonzero(role.eq("validation").to_numpy())
        guard = np.flatnonzero(role.eq("guard").to_numpy())
        _require(len(train) + len(validation) + len(guard) == len(labels), "bad fold partition")
        mint_role = mint_labels["row_id"].map(roles)
        mint_train = np.flatnonzero(mint_role.eq("train").to_numpy())
        mint_validation = np.flatnonzero(mint_role.eq("validation").to_numpy())
        _require(
            set(labels.iloc[train]["row_id"]) == set(mint_labels.iloc[mint_train]["row_id"])
            and set(labels.iloc[validation]["row_id"])
            == set(mint_labels.iloc[mint_validation]["row_id"]),
            "additive/MINT fold membership differs",
        )

        additive_model = make_logistic(ADDITIVE_C)
        additive_model.fit(additive[train], y[train])
        additive_probability = additive_model.predict_proba(additive[validation])[:, 1]

        mean, scale = fit_standardizer(mint[mint_train])
        mint_model = fit_logistic(
            standardize(mint[mint_train], mean, scale), mint_y[mint_train], MINT_C
        )
        mint_probability = mint_model.predict_proba(
            standardize(mint[mint_validation], mean, scale)
        )[:, 1]

        validation_ids = labels.iloc[validation]["row_id"].astype(str).to_numpy()
        validation_labels = y[validation]
        output.append(
            pd.DataFrame(
                {
                    "model": "additive_7site",
                    "fold": fold,
                    "row_id": validation_ids,
                    "weak_label": validation_labels,
                    "probability": additive_probability,
                }
            )
        )
        output.append(
            pd.DataFrame(
                {
                    "model": "frozen_mint_layer5",
                    "fold": fold,
                    "row_id": mint_labels.iloc[mint_validation]["row_id"].astype(str).to_numpy(),
                    "weak_label": mint_y[mint_validation],
                    "probability": mint_probability,
                }
            )
        )
        summary["folds"][str(fold)] = {
            "train": int(len(train)),
            "guard": int(len(guard)),
            "validation": int(len(validation)),
            "validation_positive": int(validation_labels.sum()),
            "additive": _metrics(validation_labels, additive_probability),
            "mint_layer5": _metrics(mint_y[mint_validation], mint_probability),
        }
    predictions = pd.concat(output, ignore_index=True)
    for model in ("additive_7site", "frozen_mint_layer5"):
        subset = predictions.loc[predictions["model"].eq(model)]
        _require(len(subset) == EXPECTED_OOF_ROWS, f"{model} OOF row count changed")
        _require(not bool(subset["row_id"].duplicated().any()), f"duplicate {model} OOF row")
        _require(int(subset["weak_label"].sum()) == EXPECTED_OOF_POSITIVE, f"{model} OOF labels changed")
        summary[model] = _metrics(
            subset["weak_label"].to_numpy(dtype=int),
            subset["probability"].to_numpy(dtype=float),
        )
    summary["row_fold_sha256"] = _row_fold_sha256(
        predictions.loc[predictions["model"].eq("additive_7site")]
    )
    return predictions, summary


def _load_missing_mint_score(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("retention_outcomes_read") is False, "missing-pair inference was not outcome-blind")
    _require(payload.get("training_performed") is False, "missing-pair artifact unexpectedly trained")
    target = payload.get("target", {})
    _require(target.get("peptide_design_code") == "AH", "missing-pair peptide changed")
    _require(target.get("affibody_design_code") == "LIFTK", "missing-pair Affibody changed")
    score = float(payload["scores"]["weak_selected_frozen_mint_layer5_deterministic_replay"])
    _require(np.isfinite(score) and 0.0 <= score <= 1.0, "bad missing-pair MINT score")
    return score


def _fit_final(
    labels: pd.DataFrame,
    mint_labels: pd.DataFrame,
    train_additive: np.ndarray,
    eval_rows: pd.DataFrame,
    eval_additive: np.ndarray,
    train_mint: np.ndarray,
    eval_mint_ids: list[str],
    eval_mint: np.ndarray,
    missing_mint_score: Optional[float],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    y = labels["weak_label"].to_numpy(dtype=int)
    mint_y = mint_labels["weak_label"].to_numpy(dtype=int)
    additive_model = make_logistic(ADDITIVE_C)
    additive_model.fit(train_additive, y)
    additive_probability = additive_model.predict_proba(eval_additive)[:, 1]

    mean, scale = fit_standardizer(train_mint)
    mint_model = fit_logistic(standardize(train_mint, mean, scale), mint_y, MINT_C)
    mint_probability = mint_model.predict_proba(standardize(eval_mint, mean, scale))[:, 1]
    mint_by_id = dict(zip(eval_mint_ids, mint_probability.tolist()))
    imported_rows = 0
    if MISSING_ROW_ID not in mint_by_id:
        _require(missing_mint_score is not None, "missing MINT cell needs outcome-blind replay")
        mint_by_id[MISSING_ROW_ID] = float(missing_mint_score)
        imported_rows = 1
    _require(set(mint_by_id) == set(eval_rows["row_id"].astype(str)), "final MINT coverage is incomplete")
    ordered_mint = np.asarray([mint_by_id[value] for value in eval_rows["row_id"].astype(str)], dtype=float)

    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "eval_row_id": eval_rows["row_id"].astype(str),
                    "model": "additive_7site",
                    "seed": "fixed",
                    "score": additive_probability,
                }
            ),
            pd.DataFrame(
                {
                    "eval_row_id": eval_rows["row_id"].astype(str),
                    "model": "frozen_mint_layer5",
                    "seed": "fixed",
                    "score": ordered_mint,
                }
            ),
        ],
        ignore_index=True,
    )
    _require(len(output) == 2 * EXPECTED_EVAL_ROWS, "final prediction row count changed")
    summary = {
        "additive_solver_iterations": int(additive_model.n_iter_[0]),
        "mint_solver_iterations": int(mint_model.n_iter_[0]),
        "mint_cached_evaluation_rows": int(len(eval_mint_ids)),
        "mint_imported_outcome_blind_rows": imported_rows,
    }
    heads = {
        "additive_coef": np.asarray(additive_model.coef_, dtype=np.float64).reshape(-1),
        "additive_intercept": np.asarray(additive_model.intercept_, dtype=np.float64).reshape(-1),
        "mint_mean": np.asarray(mean, dtype=np.float64).reshape(-1),
        "mint_scale": np.asarray(scale, dtype=np.float64).reshape(-1),
        "mint_coef": np.asarray(mint_model.coef_, dtype=np.float64).reshape(-1),
        "mint_intercept": np.asarray(mint_model.intercept_, dtype=np.float64).reshape(-1),
    }
    return output, summary, heads


def _reference_parity(
    final: pd.DataFrame,
    additive_reference_path: Path,
    mint_reference_path: Path,
) -> dict[str, Any]:
    additive_reference = pd.read_csv(additive_reference_path)
    additive_reference = additive_reference.loc[
        additive_reference["model"].eq("site_primary_pn_control"),
        ["eval_row_id", "score"],
    ].copy()
    mint_reference = pd.read_csv(mint_reference_path)
    mint_reference = mint_reference.loc[
        mint_reference["model"].eq("frozen_mint_layer5_deterministic_replay120"),
        ["eval_row_id", "score"],
    ].copy()
    _require(len(additive_reference) == EXPECTED_EVAL_ROWS, "additive reference coverage changed")
    _require(len(mint_reference) == EXPECTED_EVAL_ROWS, "MINT reference coverage changed")
    records = {}
    for model, reference in (
        ("additive_7site", additive_reference),
        ("frozen_mint_layer5", mint_reference),
    ):
        observed = final.loc[final["model"].eq(model), ["eval_row_id", "score"]]
        joined = observed.merge(
            reference,
            on="eval_row_id",
            how="outer",
            suffixes=("_regenerated", "_reference"),
            validate="one_to_one",
        )
        _require(len(joined) == EXPECTED_EVAL_ROWS and not bool(joined.isna().any().any()), f"{model} parity join failed")
        delta = np.abs(
            joined["score_regenerated"].to_numpy(dtype=float)
            - joined["score_reference"].to_numpy(dtype=float)
        )
        records[model] = {
            "rows": int(len(joined)),
            "maximum_absolute_probability_difference": float(delta.max()),
            "mean_absolute_probability_difference": float(delta.mean()),
            "reference_path": str(
                additive_reference_path.resolve()
                if model == "additive_7site"
                else mint_reference_path.resolve()
            ),
        }
    return records


def _save_deployment_heads(
    path: Path,
    manifest_path: Path,
    encoder: Any,
    heads: dict[str, np.ndarray],
    historical_parity: dict[str, Any],
    self_parity: dict[str, Any],
) -> None:
    position_names = np.asarray(
        [
            "peptide_position_4",
            "peptide_position_5",
            "affibody_displayed_6_crystal_8",
            "affibody_displayed_10_crystal_12",
            "affibody_displayed_13_crystal_15",
            "affibody_displayed_14_crystal_16",
            "affibody_displayed_17_crystal_19",
        ]
    )
    categories = np.stack([np.asarray(value).astype(str) for value in encoder.categories_])
    np.savez(
        path,
        schema_version=np.asarray(["libb-additive-mint-deployment-heads-v1"]),
        position_names=position_names,
        additive_categories=categories,
        additive_coef=heads["additive_coef"],
        additive_intercept=heads["additive_intercept"],
        additive_C=np.asarray([ADDITIVE_C], dtype=np.float64),
        mint_layer=np.asarray([5], dtype=np.int64),
        mint_feature_key=np.asarray([MINT_KEY]),
        mint_mean=heads["mint_mean"],
        mint_scale=heads["mint_scale"],
        mint_coef=heads["mint_coef"],
        mint_intercept=heads["mint_intercept"],
        mint_C=np.asarray([MINT_C], dtype=np.float64),
    )
    os.chmod(path, 0o600)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: {"shape": list(archive[name].shape), "dtype": str(archive[name].dtype)}
            for name in archive.files
        }
    _write_json(
        manifest_path,
        {
            "schema_version": "libb-additive-mint-deployment-heads-v1",
            "retention_labels_read": False,
            "archive": {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)},
            "arrays": arrays,
            "formulas": {
                "additive_7site": "sigmoid(one_hot_7x20 @ additive_coef + additive_intercept)",
                "frozen_mint_layer5": "sigmoid(float32((float64(mint_layer_05_chain_mean) - mint_mean) / mint_scale) @ mint_coef + mint_intercept)",
            },
            "serialized_head_self_parity": self_parity,
            "historical_120_reference_parity": historical_parity,
        },
    )


def _deployment_self_parity(
    final: pd.DataFrame,
    eval_rows: pd.DataFrame,
    eval_additive: Any,
    eval_mint_ids: list[str],
    eval_mint: np.ndarray,
    missing_mint_score: Optional[float],
    heads: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Verify the serialized numerical recipe against every emitted score."""

    additive_logit = np.asarray(eval_additive @ heads["additive_coef"]).reshape(-1)
    additive_logit += float(heads["additive_intercept"][0])
    additive_probability = 1.0 / (1.0 + np.exp(-additive_logit))

    mint_standardized = standardize(
        eval_mint, heads["mint_mean"], heads["mint_scale"]
    )
    mint_logit = np.asarray(mint_standardized @ heads["mint_coef"]).reshape(-1)
    mint_logit += float(heads["mint_intercept"][0])
    mint_probability = 1.0 / (1.0 + np.exp(-mint_logit))
    mint_by_id = dict(zip(eval_mint_ids, mint_probability.tolist()))
    if MISSING_ROW_ID not in mint_by_id:
        _require(missing_mint_score is not None, "missing deployment fallback score")
        mint_by_id[MISSING_ROW_ID] = float(missing_mint_score)
    ordered_mint = np.asarray(
        [mint_by_id[value] for value in eval_rows["row_id"].astype(str)], dtype=float
    )

    records = {}
    for model, regenerated in (
        ("additive_7site", additive_probability),
        ("frozen_mint_layer5", ordered_mint),
    ):
        expected = final.loc[final["model"].eq(model), "score"].to_numpy(dtype=float)
        _require(len(expected) == EXPECTED_EVAL_ROWS, f"{model} final coverage changed")
        delta = np.abs(regenerated - expected)
        _require(float(delta.max()) <= 2e-15, f"serialized {model} head failed self-parity")
        records[model] = {
            "rows": int(len(delta)),
            "maximum_absolute_probability_difference": float(delta.max()),
            "mean_absolute_probability_difference": float(delta.mean()),
        }
    return records


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    private_root = (REPO_ROOT / "private_data").resolve()
    try:
        relative = output.relative_to(private_root)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    output.mkdir(parents=True, mode=0o700)
    started = time.time()

    rows = _load_rows(args.canonical_rows)
    labels = _load_labels(args.training_labels)
    train_rows = rows.loc[rows["split"].eq("train")].copy()
    eval_rows = rows.loc[rows["split"].eq("eval")].copy()
    train_rows = labels[["row_id"]].merge(train_rows, on="row_id", how="left", validate="one_to_one")
    _require(not bool(train_rows.isna().any().any()), "training labels do not align to canonical rows")

    encoder = make_encoder(7)
    train_codes = _additive_codes(train_rows)
    eval_codes = _additive_codes(eval_rows)
    train_additive = encoder.fit_transform(train_codes)
    eval_additive = encoder.transform(eval_codes)

    requested_ids = labels["row_id"].astype(str).tolist() + eval_rows["row_id"].astype(str).tolist()
    all_mint, found_ids = _load_mint_features(args.mint_archive, requested_ids)
    train_ids = labels["row_id"].astype(str).tolist()
    eval_ids = eval_rows["row_id"].astype(str).tolist()
    train_id_set = set(train_ids)
    eval_id_set = set(eval_ids)
    found = set(found_ids)
    _require(train_id_set.issubset(found), "MINT cache lacks a training row")
    found_train_ids = [value for value in found_ids if value in train_id_set]
    found_eval_ids = [value for value in found_ids if value in eval_id_set]
    _require(len(found_train_ids) == EXPECTED_TRAIN_ROWS, "MINT training coverage changed")
    absent_eval_ids = set(eval_ids) - set(found_eval_ids)
    _require(
        (len(found_eval_ids) == 120 and not absent_eval_ids)
        or (len(found_eval_ids) == 119 and absent_eval_ids == {MISSING_ROW_ID}),
        "legacy MINT evaluation coverage changed",
    )

    train_mask = np.asarray([value in train_id_set for value in found_ids], dtype=bool)
    eval_mask = np.asarray([value in eval_id_set for value in found_ids], dtype=bool)
    train_mint = all_mint[train_mask]
    eval_mint = all_mint[eval_mask]
    _require(eval_mint.shape[0] == len(found_eval_ids), "MINT evaluation alignment changed")
    mint_labels = (
        labels.set_index("row_id", drop=False).loc[found_train_ids].reset_index(drop=True)
    )
    _require(
        set(mint_labels["row_id"]) == set(labels["row_id"]),
        "MINT training order lost a canonical row",
    )

    membership = _fold_membership(labels)
    oof, oof_summary = _fit_oof(
        labels, mint_labels, membership, train_additive, train_mint
    )
    missing_score = (
        None
        if not absent_eval_ids
        else _load_missing_mint_score(args.missing_mint_scores)
    )
    final, final_summary, heads = _fit_final(
        labels,
        mint_labels,
        train_additive,
        eval_rows,
        eval_additive,
        train_mint,
        found_eval_ids,
        eval_mint,
        missing_score,
    )
    parity = _reference_parity(
        final,
        args.additive_reference_predictions,
        args.mint_reference_predictions,
    )
    self_parity = _deployment_self_parity(
        final,
        eval_rows,
        eval_additive,
        found_eval_ids,
        eval_mint,
        missing_score,
        heads,
    )

    oof_path = output / "weak_oof_predictions.csv.gz"
    final_path = output / "final_120_predictions.csv"
    membership_path = output / "fold_membership.csv.gz"
    heads_path = output / "deployment_heads.npz"
    heads_manifest_path = output / "deployment_heads.json"
    oof.to_csv(oof_path, index=False, compression="gzip")
    final.to_csv(final_path, index=False)
    membership[["fold", "row_id", "weak_label", "role"]].to_csv(
        membership_path, index=False, compression="gzip"
    )
    for path in (oof_path, final_path, membership_path):
        os.chmod(path, 0o600)
    _save_deployment_heads(
        heads_path,
        heads_manifest_path,
        encoder,
        heads,
        parity,
        self_parity,
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "retention_labels_read": False,
        "weak_training": {
            "rows": EXPECTED_TRAIN_ROWS,
            "positive": EXPECTED_POSITIVE,
            "negative": EXPECTED_NEGATIVE,
            "membership_sha256": _membership_sha256(labels["row_id"]),
        },
        "validation": {
            "regime": "three diagonal double-cold folds; XOR partner rows guarded",
            "folds": FOLDS,
            "split_seed": SPLIT_SEED,
            "oof_unique_rows": EXPECTED_OOF_ROWS,
            "oof_positive": EXPECTED_OOF_POSITIVE,
        },
        "models": {
            "additive_7site": {
                "C": ADDITIVE_C,
                "features": "7 x 20 fixed-alphabet one-hot; no standardization",
                "estimator": "balanced L2 liblinear logistic regression",
            },
            "frozen_mint_layer5": {
                "C": MINT_C,
                "features": MINT_KEY,
                "dimension": 2560,
                "standardization": "training-fold mean and population standard deviation",
                "estimator": "balanced L2 liblinear logistic regression; tol=1e-4",
            },
        },
        "oof": oof_summary,
        "final": final_summary,
        "known_120_reference_parity": parity,
        "serialized_head_self_parity": self_parity,
        "runtime_seconds": float(time.time() - started),
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    outputs = {}
    for path in (
        oof_path,
        final_path,
        membership_path,
        heads_path,
        heads_manifest_path,
        summary_path,
    ):
        outputs[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retention_labels_read": False,
        "source_fields_read_from_mint_npz": ["pair_uid", MINT_KEY],
        "inputs": {
            "canonical_rows": {"path": str(args.canonical_rows.resolve()), "sha256": _sha256(args.canonical_rows)},
            "training_labels": {"path": str(args.training_labels.resolve()), "sha256": _sha256(args.training_labels)},
            "mint_archive": {"path": str(args.mint_archive.resolve()), "sha256": _sha256(args.mint_archive)},
            "missing_mint_scores": {"path": str(args.missing_mint_scores.resolve()), "sha256": _sha256(args.missing_mint_scores)},
            "additive_reference_predictions": {"path": str(args.additive_reference_predictions.resolve()), "sha256": _sha256(args.additive_reference_predictions)},
            "mint_reference_predictions": {"path": str(args.mint_reference_predictions.resolve()), "sha256": _sha256(args.mint_reference_predictions)},
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "outputs": outputs,
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "runtime_seconds": summary["runtime_seconds"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-rows",
        type=Path,
        default=REPO_ROOT / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json",
    )
    parser.add_argument(
        "--training-labels",
        type=Path,
        default=REPO_ROOT / "private_data/derived/esmfold2_libb_training_labels_provider_revision_120_v1/training_labels.csv",
    )
    parser.add_argument(
        "--mint-archive",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_multilayer_v1/merged/mint_multilayer_chain_mean_features.npz",
    )
    parser.add_argument(
        "--missing-mint-scores",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/libb_revision_AH_LIFTK_mint_scores_v2/scores.json",
    )
    parser.add_argument(
        "--additive-reference-predictions",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/pnu_libb_retention_120_revision_v1/normalized_predictions.csv",
    )
    parser.add_argument(
        "--mint-reference-predictions",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/libb_revision_mint_predictions_120_v1/normalized_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
