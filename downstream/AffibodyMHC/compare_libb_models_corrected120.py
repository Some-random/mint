#!/usr/bin/env python3
"""Build a peptide-conditioned LibB model comparison without global metrics.

The two inputs are deliberately separate: direct-retention results are used only
for retrospective evaluation, while matched out-of-fold weak-label predictions
provide the training-domain comparison.  Nothing is trained or selected here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


MODEL_COLUMNS = (
    ("additive", "Additive", "additive_7site", "additive_7site"),
    ("mint_layer5", "MINT layer 5", "mint_layer5", "mint_layer5"),
    (
        "stab_selected",
        "StaB selected representation",
        "stab_designed_ordered",
        "stab_designed_ordered",
    ),
    (
        "rde_selected",
        "RDE selected representation",
        "rde_network_designed_3fold",
        "rde_network_designed_3fold",
    ),
    (
        "mint_stab",
        "MINT + StaB equal-logit",
        "mean_logit_mint_stab",
        "mean_logit__mint_layer5__stab_designed_ordered",
    ),
    (
        "equal_all4",
        "Equal-logit all four",
        "mean_logit_all4",
        "mean_logit__additive_7site__mint_layer5__stab_designed_ordered__rde_network_designed_3fold",
    ),
    (
        "locked_all4",
        "Locked nonnegative stacker all four",
        "locked_nonnegative_stack_all4",
        "cross_fitted_nonnegative_stack__all4",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_retention(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    } | {row[2] for row in MODEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    _require(not missing, f"retention predictions missing columns: {missing}")
    _require(len(frame) == 120, f"expected 120 retention rows, found {len(frame)}")
    _require(frame["eval_row_id"].is_unique, "eval_row_id is not unique")
    _require(
        not frame.duplicated(["peptide_design_code", "affibody_design_code"]).any(),
        "retention peptide/Affibody pairs are not unique",
    )
    sizes = frame.groupby("peptide_design_code", sort=True).size()
    _require(len(sizes) == 12 and bool((sizes == 10).all()), "panel is not 12 peptides x 10 Affibodies")
    binders = (frame["target_retention"].to_numpy(dtype=float) >= 75.0).astype(int)
    _require(int(binders.sum()) == 61, f"retention >=75 should yield 61 binders, found {binders.sum()}")
    stated = frame["target_binder"].to_numpy(dtype=int)
    _require(np.array_equal(stated, binders), "target_binder differs from retention >=75")
    for _, _, column, _ in MODEL_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        _require(bool(np.isfinite(values).all()), f"non-finite retention scores in {column}")
    return frame.copy()


def _validate_oof(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "weak_label", "peptide_id"} | {row[3] for row in MODEL_COLUMNS}
    missing = sorted(required - set(frame.columns))
    _require(not missing, f"weak OOF predictions missing columns: {missing}")
    _require(len(frame) > 0, "weak OOF predictions are empty")
    _require(frame["row_id"].is_unique, "weak OOF row_id is not unique")
    labels = frame["weak_label"].to_numpy(dtype=int)
    _require(set(labels.tolist()) == {0, 1}, "weak_label must contain exactly 0 and 1")
    for _, _, _, column in MODEL_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        _require(bool(np.isfinite(values).all()), f"non-finite weak OOF scores in {column}")
    return frame.copy()


def _macro_binary(
    frame: pd.DataFrame, group_column: str, label_column: str, score_column: str
) -> tuple[float, float, int]:
    aps: list[float] = []
    aurocs: list[float] = []
    for _, group in frame.groupby(group_column, sort=True):
        labels = group[label_column].to_numpy(dtype=int)
        if np.unique(labels).size != 2:
            continue
        scores = group[score_column].to_numpy(dtype=float)
        aps.append(float(average_precision_score(labels, scores)))
        aurocs.append(float(roc_auc_score(labels, scores)))
    _require(bool(aps), f"no evaluable groups for {score_column}")
    return float(np.mean(aps)), float(np.mean(aurocs)), len(aps)


def _macro_spearman(frame: pd.DataFrame, score_column: str) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("peptide_design_code", sort=True):
        retention = group["target_retention"].to_numpy(dtype=float)
        score = group[score_column].to_numpy(dtype=float)
        if np.unique(retention).size < 2 or np.unique(score).size < 2:
            continue
        rho = float(spearmanr(retention, score)[0])
        if np.isfinite(rho):
            values.append(rho)
    _require(bool(values), f"no evaluable Spearman groups for {score_column}")
    return float(np.mean(values)), len(values)


def build_comparison(retention: pd.DataFrame, weak_oof: pd.DataFrame) -> pd.DataFrame:
    retention = _validate_retention(retention)
    weak_oof = _validate_oof(weak_oof)
    rows: list[dict[str, object]] = []
    for key, display, retention_column, oof_column in MODEL_COLUMNS:
        retention_ap, retention_auroc, binary_n = _macro_binary(
            retention,
            "peptide_design_code",
            "target_binder",
            retention_column,
        )
        retention_spearman, spearman_n = _macro_spearman(retention, retention_column)
        weak_ap, _, weak_n = _macro_binary(
            weak_oof, "peptide_id", "weak_label", oof_column
        )
        rows.append(
            {
                "model": key,
                "display_name": display,
                "corrected120_score_column": retention_column,
                "corrected120_rows": len(retention),
                "corrected120_binders": int(retention["target_binder"].sum()),
                "within_peptide_ap": retention_ap,
                "within_peptide_auroc": retention_auroc,
                "within_peptide_binary_evaluable_peptides": binary_n,
                "within_peptide_spearman": retention_spearman,
                "within_peptide_spearman_evaluable_peptides": spearman_n,
                "weak_oof_score_column": oof_column,
                "weak_oof_rows": len(weak_oof),
                "weak_oof_positives": int(weak_oof["weak_label"].sum()),
                "weak_oof_macro_ap": weak_ap,
                "weak_oof_evaluable_peptides": weak_n,
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected120-predictions", type=Path, required=True)
    parser.add_argument("--weak-oof-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retention = pd.read_csv(args.corrected120_predictions)
    weak_oof = pd.read_csv(args.weak_oof_predictions)
    comparison = build_comparison(retention, weak_oof)
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False, float_format="%.12g")
    print(f"wrote {len(comparison)} models to {args.output}")


if __name__ == "__main__":
    main()
