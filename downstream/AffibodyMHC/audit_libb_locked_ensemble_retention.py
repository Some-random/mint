#!/usr/bin/env python
"""Audit weak-label-locked LibB ensembles on the corrected 120-pair panel.

The weak-label selection and stacker parameters are verified and hashed into
the output directory *before* any retention-bearing prediction table is read.
The retention panel is then used only for a retrospective audit; it cannot
change the already locked weak-label selection or stacker parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-weak-locked-retention-audit-v1"
WEAK_SCHEMA_VERSION = "libb-weak-oof-ensemble-all4-v1"
CALIBRATION_STATUS = "retrospective_threshold_fit_on_same_corrected_120_panel"

EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_BINDERS = 61
EXPECTED_NONBINDERS = 59
EXPECTED_STRUCTURE_SEEDS = 5
EXPECTED_MEMBERSHIP_SHA256 = "083cebcb2c0e83c4f61196211059ab16c4774ee99c3dafc4c573a71f42ece705"
RETENTION_BINDER_THRESHOLD = 75.0
FALTA_CODE = "FALTA"

MEMBERS = (
    "additive_7site",
    "mint_layer5",
    "stab_designed_ordered",
    "rde_network_designed_3fold",
)
AUDIT_MODELS = (
    ("mint_layer5", "Frozen MINT layer 5"),
    ("mean_logit_mint_stab", "Equal-logit MINT + StaB"),
    ("mean_logit_all4", "Equal-logit additive + MINT + StaB + RDE"),
    ("locked_nonnegative_stack_all4", "Weak-label-locked all-four stacker"),
)

DEFAULT_WEAK_DIR = (
    REPO_ROOT / "private_data/experiments/libb_weak_ensemble_selection_all4_v2"
)
DEFAULT_MINT_SOURCE = (
    REPO_ROOT
    / "private_data/experiments/libb_revision_mint_predictions_120_v1/normalized_predictions.csv"
)
DEFAULT_ADDITIVE_SOURCE = (
    REPO_ROOT
    / "private_data/experiments/pnu_libb_retention_120_revision_v1/normalized_predictions.csv"
)
DEFAULT_STAB_DIR = (
    REPO_ROOT
    / "private_data/experiments/stab_libb_provider_revision_120_readout_full_v1/"
    "stab_designed_ordered/blinded_predictions"
)
DEFAULT_RDE_DIR = (
    REPO_ROOT
    / "private_data/experiments/rde_libb_provider_revision_120_full_readouts_v2_common_env/"
    "rde_network_designed/blinded_predictions"
)
DEFAULT_PANEL = (
    REPO_ROOT
    / "private_data/derived/retention_panel_provider_revision_2026-09-03_v2/"
    "libb_evaluation_panel.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "private_data/experiments/libb_weak_ensemble_retention_audit_v2"
)

MINT_SOURCE_MODEL = "frozen_mint_layer5_archived119_plus_replay_target"
ADDITIVE_SOURCE_MODEL = "site_primary_pn_control"
STAB_SOURCE_MODEL = "stab_designed_ordered"
RDE_SOURCE_MODEL = "rde_network_designed_3fold_ensemble"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "mtime_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        "sha256": _sha256_file(path),
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    _require(not Path(path).exists(), f"refusing to overwrite {path}")
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )
    os.chmod(path, 0o600)


def clipped_logit(probability: Sequence[float], clip: float = 1e-7) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    _require(values.ndim == 1 and len(values) > 0, "probabilities must be a nonempty vector")
    _require(bool(np.isfinite(values).all()), "probabilities contain non-finite values")
    _require(bool(((values >= 0.0) & (values <= 1.0)).all()), "probabilities are outside [0,1]")
    values = np.clip(values, clip, 1.0 - clip)
    return np.log(values) - np.log1p(-values)


def sigmoid(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def mean_logit_score(score_columns: Sequence[Sequence[float]]) -> np.ndarray:
    _require(len(score_columns) > 0, "mean-logit ensemble has no members")
    logits = [clipped_logit(column) for column in score_columns]
    _require(len({len(column) for column in logits}) == 1, "mean-logit member lengths differ")
    return sigmoid(np.mean(np.column_stack(logits), axis=1))


def apply_locked_stacker(
    member_scores: pd.DataFrame, parameters: Mapping[str, Any]
) -> np.ndarray:
    """Apply the weak-only standardized-logit stacker without refitting it."""

    member_order = tuple(map(str, parameters["members_in_coefficient_order"]))
    _require(member_order == MEMBERS, f"locked stacker member order changed: {member_order}")
    raw_logits = np.column_stack(
        [clipped_logit(member_scores[member].to_numpy(dtype=float)) for member in member_order]
    )
    mean = np.asarray(parameters["mean"], dtype=np.float64)
    scale = np.asarray(parameters["scale"], dtype=np.float64)
    coefficient = np.asarray(parameters["coefficient"], dtype=np.float64)
    _require(mean.shape == (len(MEMBERS),), "locked stacker mean has the wrong shape")
    _require(scale.shape == (len(MEMBERS),), "locked stacker scale has the wrong shape")
    _require(coefficient.shape == (len(MEMBERS),), "locked stacker coefficient has the wrong shape")
    _require(bool(np.isfinite(mean).all()), "locked stacker mean contains non-finite values")
    _require(bool(np.isfinite(scale).all()) and bool((scale > 0.0).all()), "invalid stacker scale")
    _require(bool(np.isfinite(coefficient).all()) and bool((coefficient >= 0.0).all()), "invalid stacker coefficient")
    intercept = float(parameters["intercept"])
    _require(math.isfinite(intercept), "invalid stacker intercept")
    standardized = (raw_logits - mean) / scale
    return sigmoid(intercept + standardized @ coefficient)


def validate_and_preserve_weak_lock(weak_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate/hash weak artifacts and write a seal before retention is opened."""

    weak_dir = Path(weak_dir).resolve()
    output_dir = Path(output_dir).resolve()
    lock_path = weak_dir / "locked_weak_selection.json"
    stack_path = weak_dir / "stacker_locked_parameters.json"
    manifest_path = weak_dir / "manifest.json"
    candidate_path = weak_dir / "candidate_metrics.csv"
    for path in (lock_path, stack_path, manifest_path, candidate_path):
        _require(path.is_file(), f"missing weak-label lock artifact: {path}")

    with lock_path.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    with stack_path.open("r", encoding="utf-8") as handle:
        parameters = json.load(handle)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for name, payload in (
        ("selection", selection),
        ("parameters", parameters),
        ("manifest", manifest),
    ):
        _require(payload.get("schema_version") == WEAK_SCHEMA_VERSION, f"{name} schema changed")
        _require(payload.get("retention_labels_read") is False, f"{name} is not retention-blind")
    _require(
        selection["best_candidate"]["candidate"] == "mean_logit__mint_layer5",
        "locked weak-label best candidate changed",
    )
    _require(
        selection["best_ensemble_candidate"]["candidate"]
        == "mean_logit__mint_layer5__stab_designed_ordered",
        "locked weak-label best ensemble changed",
    )
    _require(tuple(parameters["members_in_coefficient_order"]) == MEMBERS, "stacker members changed")

    _require(not output_dir.exists(), f"output directory exists; refusing overwrite: {output_dir}")
    _require("private_data" in output_dir.parts, "output must remain under private_data")
    output_dir.mkdir(parents=True, mode=0o700)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "event": "weak_label_lock_hashed_before_retention_sources_were_opened",
        "retention_sources_opened": False,
        "selection_is_immutable_during_audit": True,
        "locked_best_candidate": selection["best_candidate"],
        "locked_best_ensemble_candidate": selection["best_ensemble_candidate"],
        "weak_artifacts": {
            path.name: _file_record(path)
            for path in (lock_path, stack_path, manifest_path, candidate_path)
        },
    }
    _write_json_exclusive(output_dir / "pre_retention_lock.json", seal)
    return {"selection": selection, "parameters": parameters, "manifest": manifest, "seal": seal}


PANEL_COLUMNS = [
    "eval_row_id",
    "peptide_design_code",
    "affibody_design_code",
    "target_retention",
    "target_binder",
]


def _read_prediction_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "eval_row_id": str,
            "model": str,
            "seed": str,
            "peptide_design_code": str,
            "affibody_design_code": str,
        },
        keep_default_na=False,
    )
    required = {"eval_row_id", "model", "seed", "score"}
    _require(required.issubset(frame.columns), f"prediction source schema changed: {path}")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    _require(bool(np.isfinite(frame["score"]).all()), f"non-finite scores in {path}")
    return frame


def _read_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "pair_uid": str,
            "peptide_design_code": str,
            "affibody_design_code": str,
        },
        keep_default_na=False,
    )
    required = {
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    }
    _require(required.issubset(frame.columns), f"retention panel schema changed: {path}")
    frame = frame.rename(columns={"pair_uid": "eval_row_id"})[PANEL_COLUMNS].copy()
    frame["target_retention"] = pd.to_numeric(frame["target_retention"], errors="raise").astype(float)
    frame["target_binder"] = pd.to_numeric(frame["target_binder"], errors="raise").astype(int)
    _require(bool(np.isfinite(frame["target_retention"]).all()), f"non-finite retention in {path}")
    return frame.sort_values("eval_row_id", kind="stable").reset_index(drop=True)


def _validate_panel(panel: pd.DataFrame) -> None:
    _require(len(panel) == EXPECTED_ROWS, f"panel has {len(panel)} rows, expected {EXPECTED_ROWS}")
    _require(panel["eval_row_id"].nunique() == EXPECTED_ROWS, "panel row IDs are not unique")
    _require(panel["peptide_design_code"].nunique() == EXPECTED_PEPTIDES, "peptide count changed")
    _require(panel["affibody_design_code"].nunique() == EXPECTED_AFFIBODIES, "Affibody count changed")
    sizes = panel.groupby("peptide_design_code", sort=True).size()
    _require(bool(sizes.eq(EXPECTED_AFFIBODIES).all()), "panel is not a complete 12 x 10 matrix")
    expected_labels = panel["target_retention"].ge(RETENTION_BINDER_THRESHOLD).astype(int)
    _require(bool(expected_labels.eq(panel["target_binder"]).all()), "binder labels are not retention >= 75")
    binders = int(panel["target_binder"].sum())
    _require(binders == EXPECTED_BINDERS, f"binder count is {binders}, expected {EXPECTED_BINDERS}")
    _require(len(panel) - binders == EXPECTED_NONBINDERS, "non-binder count changed")


def _selected_model(frame: pd.DataFrame, source_model: str, expected_fits: int) -> pd.DataFrame:
    chosen = frame.loc[
        frame["model"].eq(source_model), ["eval_row_id", "model", "seed", "score"]
    ].copy()
    _require(not chosen.empty, f"source model is absent: {source_model}")
    _require(not bool(chosen.duplicated(["seed", "eval_row_id"]).any()), f"duplicates in {source_model}")
    fit_sizes = chosen.groupby("seed", sort=True).size()
    _require(len(fit_sizes) == expected_fits, f"{source_model} fit count changed: {len(fit_sizes)}")
    _require(bool(fit_sizes.eq(EXPECTED_ROWS).all()), f"an incomplete fit exists for {source_model}")
    return chosen


def _membership_sha256(row_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, row_ids)))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _assert_same_membership(reference_ids: Sequence[str], candidate_rows: pd.DataFrame, name: str) -> None:
    candidate_ids = sorted(candidate_rows["eval_row_id"].drop_duplicates().astype(str).tolist())
    reference = sorted(map(str, reference_ids))
    _require(candidate_ids == reference, f"prediction membership differs in {name}")


def aggregate_structure_seed_logits(chosen: pd.DataFrame, output_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one mean-logit score per pair and the aligned wide seed table."""

    index_columns = ["eval_row_id"]
    wide = chosen.pivot(index=index_columns, columns="seed", values="score").reset_index()
    seed_columns = sorted([column for column in wide.columns if column not in index_columns])
    _require(len(seed_columns) == EXPECTED_STRUCTURE_SEEDS, "structure seed count changed after pivot")
    wide.columns.name = None
    wide = wide.sort_values("eval_row_id", kind="stable").reset_index(drop=True)
    aggregate = wide[["eval_row_id"]].copy()
    aggregate[output_name] = mean_logit_score(
        [wide[column].to_numpy(dtype=float) for column in seed_columns]
    )
    renamed = wide.rename(columns={column: f"{output_name}__seed_{column}" for column in seed_columns})
    return aggregate, renamed


def _load_structure_seed_files(directory: Path, source_model: str) -> tuple[pd.DataFrame, list[Path]]:
    directory = Path(directory).resolve()
    paths = sorted(directory.glob(f"{source_model}__seed*.csv"))
    _require(
        len(paths) == EXPECTED_STRUCTURE_SEEDS,
        f"found {len(paths)} {source_model} seed files in {directory}, expected {EXPECTED_STRUCTURE_SEEDS}",
    )
    frames = [_read_prediction_source(path) for path in paths]
    return pd.concat(frames, ignore_index=True), paths


def load_fixed_component_scores(
    mint_source: Path,
    additive_source: Path,
    stab_dir: Path,
    rde_dir: Path,
    panel_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    # All scores are loaded from outcome-blind prediction files before the
    # direct-retention panel is opened.
    mint_all = _read_prediction_source(mint_source)
    additive_all = _read_prediction_source(additive_source)
    stab_all, stab_paths = _load_structure_seed_files(stab_dir, STAB_SOURCE_MODEL)
    rde_all, rde_paths = _load_structure_seed_files(rde_dir, RDE_SOURCE_MODEL)
    mint = _selected_model(mint_all, MINT_SOURCE_MODEL, 1)
    additive = _selected_model(additive_all, ADDITIVE_SOURCE_MODEL, 1)
    stab = _selected_model(stab_all, STAB_SOURCE_MODEL, EXPECTED_STRUCTURE_SEEDS)
    rde = _selected_model(rde_all, RDE_SOURCE_MODEL, EXPECTED_STRUCTURE_SEEDS)

    reference_ids = mint["eval_row_id"].drop_duplicates().tolist()
    _require(len(reference_ids) == EXPECTED_ROWS, "MINT prediction row count changed")
    _require(
        _membership_sha256(reference_ids) == EXPECTED_MEMBERSHIP_SHA256,
        "corrected-120 prediction membership hash changed",
    )
    for name, frame in (("additive", additive), ("StaB", stab), ("RDE", rde)):
        _assert_same_membership(reference_ids, frame, name)

    # Only now do we open the retention outcomes and attach them by opaque ID.
    panel = _read_panel(panel_path)
    _validate_panel(panel)
    _assert_same_membership(reference_ids, panel, "corrected retention panel")

    mint_score = mint[["eval_row_id", "score"]].rename(columns={"score": "mint_layer5"})
    additive_score = additive[["eval_row_id", "score"]].rename(columns={"score": "additive_7site"})
    stab_score, stab_seed_wide = aggregate_structure_seed_logits(stab, "stab_designed_ordered")
    rde_score, rde_seed_wide = aggregate_structure_seed_logits(rde, "rde_network_designed_3fold")

    scores = panel.copy()
    for current in (additive_score, mint_score, stab_score, rde_score):
        scores = scores.merge(
            current[["eval_row_id", current.columns[-1]]],
            on="eval_row_id",
            how="left",
            validate="one_to_one",
        )
    _require(not bool(scores[list(MEMBERS)].isna().any().any()), "missing component scores")

    seed_scores = stab_seed_wide.merge(
        rde_seed_wide,
        on="eval_row_id",
        how="inner",
        validate="one_to_one",
    )
    source_audit = {
        "mint": {
            "file": _file_record(mint_source),
            "source_model": MINT_SOURCE_MODEL,
            "fits": 1,
        },
        "additive": {
            "file": _file_record(additive_source),
            "source_model": ADDITIVE_SOURCE_MODEL,
            "fits": 1,
        },
        "StaB": {
            "source_model": STAB_SOURCE_MODEL,
            "fits": EXPECTED_STRUCTURE_SEEDS,
            "files": [_file_record(path) for path in stab_paths],
        },
        "RDE": {
            "source_model": RDE_SOURCE_MODEL,
            "fits": EXPECTED_STRUCTURE_SEEDS,
            "files": [_file_record(path) for path in rde_paths],
        },
        "corrected_retention_panel": _file_record(panel_path),
    }
    return scores, seed_scores, source_audit


def build_audit_predictions(
    component_scores: pd.DataFrame, stacker_parameters: Mapping[str, Any]
) -> pd.DataFrame:
    output = component_scores.copy()
    output["mean_logit_mint_stab"] = mean_logit_score(
        [output["mint_layer5"], output["stab_designed_ordered"]]
    )
    output["mean_logit_all4"] = mean_logit_score([output[member] for member in MEMBERS])
    output["locked_nonnegative_stack_all4"] = apply_locked_stacker(output, stacker_parameters)
    return output


def _spearman(scores: Sequence[float], retention: Sequence[float]) -> float:
    score_rank = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    retention_rank = pd.Series(np.asarray(retention, dtype=float)).rank(method="average").to_numpy()
    if np.ptp(score_rank) == 0.0 or np.ptp(retention_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(score_rank, retention_rank)[0, 1])


def calculate_ranking_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_peptide_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_order, (model_key, display_name) in enumerate(AUDIT_MODELS):
        for peptide, group in predictions.groupby("peptide_design_code", sort=True):
            labels = group["target_binder"].to_numpy(dtype=int)
            scores = group[model_key].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "model_order": model_order,
                "model_key": model_key,
                "display_name": display_name,
                "peptide_design_code": str(peptide),
                "pairs": len(group),
                "binders": int(labels.sum()),
                "within_peptide_spearman": _spearman(scores, group["target_retention"]),
                "within_peptide_auroc": float("nan"),
                "within_peptide_average_precision": float("nan"),
                "binary_metric_evaluable": int(np.unique(labels).size == 2),
            }
            if row["binary_metric_evaluable"]:
                row["within_peptide_auroc"] = float(roc_auc_score(labels, scores))
                row["within_peptide_average_precision"] = float(
                    average_precision_score(labels, scores)
                )
            per_peptide_rows.append(row)

        model_rows = [row for row in per_peptide_rows if row["model_key"] == model_key]
        spearman = [row["within_peptide_spearman"] for row in model_rows if math.isfinite(row["within_peptide_spearman"])]
        auroc = [row["within_peptide_auroc"] for row in model_rows if math.isfinite(row["within_peptide_auroc"])]
        average_precision = [row["within_peptide_average_precision"] for row in model_rows if math.isfinite(row["within_peptide_average_precision"])]
        excluded = sorted(
            row["peptide_design_code"]
            for row in model_rows
            if not row["binary_metric_evaluable"]
        )
        summary_rows.append(
            {
                "model_order": model_order,
                "model_key": model_key,
                "display_name": display_name,
                "within_peptide_average_precision": float(np.mean(average_precision)),
                "within_peptide_auroc": float(np.mean(auroc)),
                "within_peptide_spearman": float(np.mean(spearman)),
                "binary_evaluable_peptides": len(average_precision),
                "binary_excluded_peptides": json.dumps(excluded, separators=(",", ":")),
                "spearman_evaluable_peptides": len(spearman),
            }
        )
    per_peptide = pd.DataFrame(per_peptide_rows).sort_values(
        ["model_order", "peptide_design_code"], kind="stable"
    )
    summary = pd.DataFrame(summary_rows).sort_values("model_order", kind="stable")
    return summary.reset_index(drop=True), per_peptide.reset_index(drop=True)


def select_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Fit score >= threshold by F1, then precision, count, and threshold."""

    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    _require(len(scores_array) == len(labels_array) and len(scores_array) > 0, "bad threshold arrays")
    positives = int(labels_array.sum())
    candidates = []
    for threshold in np.unique(scores_array):
        selected = scores_array >= threshold
        selected_count = int(selected.sum())
        tp = int(np.logical_and(selected, labels_array == 1).sum())
        fp = selected_count - tp
        fn = positives - tp
        tn = len(labels_array) - tp - fp - fn
        precision_fraction = Fraction(tp, selected_count) if selected_count else Fraction(0, 1)
        recall_fraction = Fraction(tp, positives) if positives else Fraction(0, 1)
        denominator = 2 * tp + fp + fn
        f1_fraction = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
        precision = float(precision_fraction)
        recall = float(recall_fraction)
        f1 = float(f1_fraction)
        result = {
            "threshold": float(threshold),
            "recommended": selected_count,
            "true_binders_recommended": tp,
            "non_binders_recommended": fp,
            "binders_missed": fn,
            "non_binders_not_recommended": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        candidates.append(
            ((f1_fraction, precision_fraction, -selected_count, float(threshold)), result)
        )
    return max(candidates, key=lambda item: item[0])[1]


def build_threshold_audit(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    peptide_count_rows: list[dict[str, Any]] = []
    labels = predictions["target_binder"].to_numpy(dtype=int)
    for model_order, (model_key, display_name) in enumerate(AUDIT_MODELS):
        chosen = select_f1_threshold(predictions[model_key], labels)
        decisions = predictions[PANEL_COLUMNS].copy()
        decisions.insert(0, "model_order", model_order)
        decisions.insert(1, "model_key", model_key)
        decisions.insert(2, "display_name", display_name)
        decisions["score"] = predictions[model_key].to_numpy(dtype=float)
        decisions["threshold"] = chosen["threshold"]
        decisions["recommended"] = decisions["score"].ge(chosen["threshold"]).astype(int)
        decisions["correct_binder_recommendation"] = (
            decisions["recommended"].eq(1) & decisions["target_binder"].eq(1)
        ).astype(int)
        decisions["calibration_status"] = CALIBRATION_STATUS
        decision_frames.append(decisions)

        count_lookup: dict[str, int] = {}
        for peptide, group in decisions.groupby("peptide_design_code", sort=True):
            recommended = int(group["recommended"].sum())
            correct = int(group["correct_binder_recommendation"].sum())
            available = int(group["target_binder"].sum())
            count_lookup[str(peptide)] = recommended
            peptide_count_rows.append(
                {
                    "model_order": model_order,
                    "model_key": model_key,
                    "display_name": display_name,
                    "peptide_design_code": str(peptide),
                    "recommended": recommended,
                    "true_binders_recommended": correct,
                    "non_binders_recommended": recommended - correct,
                    "available_binders": available,
                    "binders_missed": available - correct,
                    "calibration_status": CALIBRATION_STATUS,
                }
            )
        zero_candidate_peptides = sorted(
            peptide for peptide, count in count_lookup.items() if count == 0
        )
        non_falta = decisions.loc[
            decisions["target_binder"].eq(1)
            & decisions["affibody_design_code"].ne(FALTA_CODE)
        ]
        non_falta_selected = int(non_falta["recommended"].sum())
        summary_rows.append(
            {
                "model_order": model_order,
                "model_key": model_key,
                "display_name": display_name,
                **chosen,
                "non_falta_binders_recommended": non_falta_selected,
                "non_falta_binders_total": len(non_falta),
                "non_falta_binder_recall": non_falta_selected / len(non_falta),
                "recommended_per_peptide": json.dumps(count_lookup, sort_keys=True, separators=(",", ":")),
                "zero_candidate_peptides": json.dumps(zero_candidate_peptides, separators=(",", ":")),
                "calibration_status": CALIBRATION_STATUS,
            }
        )
    return (
        pd.DataFrame(summary_rows).sort_values("model_order", kind="stable").reset_index(drop=True),
        pd.concat(decision_frames, ignore_index=True).sort_values(
            ["model_order", "peptide_design_code", "affibody_design_code"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(peptide_count_rows).sort_values(
            ["model_order", "peptide_design_code"], kind="stable"
        ).reset_index(drop=True),
    )


def _format_metric(value: float) -> str:
    return f"{float(value):.3f}"


def render_report(
    weak_payload: Mapping[str, Any],
    ranking: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> str:
    mint_ap = float(
        ranking.loc[ranking["model_key"].eq("mint_layer5"), "within_peptide_average_precision"].item()
    )
    comparisons = ranking.loc[~ranking["model_key"].eq("mint_layer5")].copy()
    best_comparison = comparisons.sort_values(
        "within_peptide_average_precision", ascending=False, kind="stable"
    ).iloc[0]
    delta = float(best_comparison["within_peptide_average_precision"]) - mint_ap
    direction = "higher" if delta > 0 else "lower"

    lines = [
        "# Post-lock LibB ensemble audit on the corrected 120-pair panel",
        "",
        "The model definitions and ensemble weights were fitted using only the selection-derived ",
        "weak labels. Before this audit command opened its 120-pair retention input, it recorded ",
        "their SHA-256 fingerprints in `pre_retention_lock.json`. That records the execution order ",
        "for this audit; it does not make the previously examined panel an independent blind test. ",
        "Nothing in this audit changes ",
        "the locked choice: MINT layer 5 was the best weak-label model, and MINT + StaB was the ",
        "best weak-label multi-model ensemble.",
        "",
        "For StaB and RDE, the five saved seed scores are averaged in logit space first. The ",
        "equal-weight ensembles then give each model one vote. The locked stacker uses exactly ",
        "the four coefficients, scaling values, and intercept fitted on weak-label out-of-fold data.",
        "",
        "## Ranking Affibodies separately for each peptide",
        "",
        "Average precision (AP) and AUROC ask whether binders (retention at least 75%) are ranked ",
        "above non-binders among the ten Affibodies for the same peptide. Spearman compares the ",
        "full ordering with the numerical retention values. AP and AUROC are averaged across the ",
        "11 peptides containing both classes; DP has no binders and is excluded from those two ",
        "metrics only. Spearman is averaged across all 12 peptides.",
        "",
        "| Fixed score | Within-peptide AP | Within-peptide AUROC | Within-peptide Spearman |",
        "|---|---:|---:|---:|",
    ]
    for row in ranking.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {_format_metric(row.within_peptide_average_precision)} | "
            f"{_format_metric(row.within_peptide_auroc)} | "
            f"{_format_metric(row.within_peptide_spearman)} |"
        )
    lines.extend(
        [
            "",
            f"The strongest ensemble in this fixed comparison is **{best_comparison['display_name']}**. "
            f"Its within-peptide AP is {abs(delta):.3f} {direction} than MINT layer 5 alone. "
            "This is an audit result, not a reason to revise the weak-label lock.",
            "",
            "## Turning each score into a recommend/do-not-recommend rule",
            "",
            "For this retrospective operating exercise, each model receives its own score cutoff ",
            "that maximizes F1 on these same 120 known labels. It is useful for seeing how many ",
            "samples the rule would send to the lab, but its precision, recall, and F1 are optimistic: ",
            "the cutoff was chosen on the panel being summarized. It must be frozen before new ",
            "wet-lab candidates are measured. Score cutoffs cannot be compared across models because ",
            "their numerical score scales differ.",
            "",
            "| Fixed score | Cutoff | Recommended | Correct binders | Incorrect recommendations | Missed binders | Precision | Recall | F1 | Peptides with none |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in thresholds.itertuples(index=False):
        zero = json.loads(row.zero_candidate_peptides)
        lines.append(
            f"| {row.display_name} | {row.threshold:.6g} | {row.recommended}/120 | "
            f"{row.true_binders_recommended} | {row.non_binders_recommended} | "
            f"{row.binders_missed} | {row.precision:.3f} | {row.recall:.3f} | "
            f"{row.f1:.3f} | {', '.join(zero) if zero else 'None'} |"
        )
    lines.extend(
        [
            "",
            "No Precision@3 result is used as a headline or as a selection rule here. Per-pair ",
            "scores, decisions, and per-peptide breakdowns are saved beside this report.",
            "",
            "## Interpretation boundary",
            "",
            "This is a retrospective audit of models whose component behavior on this panel had ",
            "already been examined elsewhere. It is not a new independent validation set. The ",
            "actual test of the frozen model and cutoff is the next prospective wet-lab batch.",
            "",
            "Weak-label lock retained: `{}`. Locked all-four stacker L2: `{}`.".format(
                weak_payload["selection"]["best_candidate"]["candidate"],
                weak_payload["parameters"]["l2"],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    weak_dir = Path(args.weak_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # This is deliberately the first operation that creates audit output. It
    # reads only retention-blind weak-label artifacts and seals their hashes.
    weak_payload = validate_and_preserve_weak_lock(weak_dir, output_dir)

    # Retention-bearing files may be opened only after pre_retention_lock.json exists.
    component_scores, seed_scores, source_audit = load_fixed_component_scores(
        Path(args.mint_source).resolve(),
        Path(args.additive_source).resolve(),
        Path(args.stab_dir).resolve(),
        Path(args.rde_dir).resolve(),
        Path(args.panel).resolve(),
    )
    predictions = build_audit_predictions(component_scores, weak_payload["parameters"])
    ranking, per_peptide = calculate_ranking_metrics(predictions)
    thresholds, decisions, counts = build_threshold_audit(predictions)

    files: list[tuple[str, pd.DataFrame]] = [
        ("per_pair_predictions.csv", predictions),
        ("structure_component_scores_by_seed.csv", seed_scores),
        ("ranking_metrics.csv", ranking.drop(columns=["model_order"])),
        ("per_peptide_ranking_metrics.csv", per_peptide.drop(columns=["model_order"])),
        ("retrospective_threshold_metrics.csv", thresholds.drop(columns=["model_order"])),
        ("retrospective_candidate_decisions.csv", decisions.drop(columns=["model_order"])),
        ("retrospective_candidate_counts_by_peptide.csv", counts.drop(columns=["model_order"])),
    ]
    output_records: dict[str, Any] = {}
    for filename, frame in files:
        path = output_dir / filename
        _write_csv_exclusive(path, frame)
        output_records[filename] = {**_file_record(path), "rows": len(frame)}

    report_path = output_dir / "post_lock_retention_audit.md"
    report_path.write_text(render_report(weak_payload, ranking, thresholds), encoding="utf-8")
    os.chmod(report_path, 0o600)
    output_records[report_path.name] = _file_record(report_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "weak_lock_preceded_retention_read": True,
        "selection_changed_after_retention_audit": False,
        "retention_panel": {
            "pairs": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies_per_peptide": EXPECTED_AFFIBODIES,
            "binder_definition": "retention >= 75",
            "binders": EXPECTED_BINDERS,
            "non_binders": EXPECTED_NONBINDERS,
        },
        "aggregation": {
            "structure_seed_rule": "mean logit across five saved seeds, then sigmoid",
            "equal_ensemble_rule": "one mean-logit vote per component model",
            "stacker_rule": "apply weak-label-locked scaling/intercept/coefficients without refitting",
        },
        "threshold_status": CALIBRATION_STATUS,
        "weak_lock": weak_payload["seal"],
        "retention_sources": source_audit,
        "code": _file_record(Path(__file__).resolve()),
        "outputs": output_records,
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    print(ranking.drop(columns=["model_order"]).to_string(index=False))
    print()
    print(thresholds.drop(columns=["model_order"]).to_string(index=False))
    print(f"\nwrote {output_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-dir", default=str(DEFAULT_WEAK_DIR))
    parser.add_argument("--mint-source", default=str(DEFAULT_MINT_SOURCE))
    parser.add_argument("--additive-source", default=str(DEFAULT_ADDITIVE_SOURCE))
    parser.add_argument("--stab-dir", default=str(DEFAULT_STAB_DIR))
    parser.add_argument("--rde-dir", default=str(DEFAULT_RDE_DIR))
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
