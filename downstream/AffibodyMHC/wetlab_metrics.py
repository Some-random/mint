"""Canonical evaluation metrics for Affibody wet-lab candidate selection.

The retention matrix is evaluated as a collection of peptide-specific
candidate lists.  This module deliberately does not depend on a rectangular
matrix representation: the corrected LibB panel currently contains all
12 x 10 = 120 measured pairs, while the same metric implementation also works
for future candidate panels with unequal numbers of Affibodies per peptide.

Ranking ties are resolved deterministically by ascending Affibody identifier
after sorting by descending model score.  Callers should pass the stable,
human-readable Affibody design code when it is available.  This explicit rule
replaces legacy evaluators whose ties depended either on input row order or on
an opaque identity hash.

Retention is used here only for final evaluation.  It must not be passed to a
training, early-stopping, architecture-selection, or hyperparameter-selection
routine.
"""

from __future__ import print_function

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


TIE_POLICY = "score_desc_then_affibody_id_asc"
DEFAULT_K_VALUES = (1, 3)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _safe_spearman(observed, predicted):
    """Return Spearman rho, or NaN when either vector has no rank variation."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (
        len(observed) < 2
        or np.unique(observed).size < 2
        or np.unique(predicted).size < 2
    ):
        return float("nan")
    return float(spearmanr(observed, predicted)[0])


def _linear_ndcg(retention_in_predicted_order, all_retention, k):
    """Compute NDCG@k using non-negative retention as a linear relevance gain.

    The common exponential gain ``2**relevance - 1`` is inappropriate for a
    0--100 retention percentage.  Linear gain preserves the assay scale and is
    invariant to dividing all retention values by 100.
    """
    predicted = np.asarray(retention_in_predicted_order, dtype=float)[: int(k)]
    ideal = np.sort(np.asarray(all_retention, dtype=float))[::-1][: int(k)]
    discount = 1.0 / np.log2(np.arange(2, 2 + len(predicted), dtype=float))
    ideal_discount = 1.0 / np.log2(np.arange(2, 2 + len(ideal), dtype=float))
    dcg = float(np.sum(predicted * discount))
    ideal_dcg = float(np.sum(ideal * ideal_discount))
    if ideal_dcg <= 0.0:
        return float("nan")
    return dcg / ideal_dcg


def _validate_frame(
    frame,
    peptide_column,
    affibody_column,
    score_column,
    binder_column,
    retention_column,
):
    required = {
        peptide_column,
        affibody_column,
        score_column,
        binder_column,
        retention_column,
    }
    missing = required.difference(frame.columns)
    _require(not missing, "evaluation frame is missing columns {}".format(sorted(missing)))
    _require(len(frame) > 0, "evaluation frame is empty")

    for column in (peptide_column, affibody_column):
        values = frame[column]
        _require(not bool(values.isna().any()), "{} contains missing values".format(column))
        _require(
            not bool(values.astype(str).eq("").any()),
            "{} contains empty identifiers".format(column),
        )

    duplicate = frame.duplicated([peptide_column, affibody_column], keep=False)
    _require(
        not bool(duplicate.any()),
        "each peptide/Affibody pair must occur exactly once in the evaluation panel",
    )

    score = pd.to_numeric(frame[score_column], errors="raise").to_numpy(dtype=float)
    retention = pd.to_numeric(frame[retention_column], errors="raise").to_numpy(dtype=float)
    binder = pd.to_numeric(frame[binder_column], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(score).all()), "model score contains non-finite values")
    _require(bool(np.isfinite(retention).all()), "retention contains non-finite values")
    _require(bool((retention >= 0.0).all()), "retention must be non-negative for NDCG")
    _require(bool(np.isfinite(binder).all()), "binder label contains non-finite values")
    _require(set(binder.tolist()).issubset({0.0, 1.0}), "binder labels must be 0 or 1")


def rank_within_peptide(
    frame,
    peptide_column="peptide_design_code",
    affibody_column="affibody_design_code",
    score_column="score",
):
    """Return a copy ordered and ranked under the canonical deterministic rule."""
    required = {peptide_column, affibody_column, score_column}
    missing = required.difference(frame.columns)
    _require(not missing, "ranking frame is missing columns {}".format(sorted(missing)))
    values = pd.to_numeric(frame[score_column], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(values).all()), "model score contains non-finite values")
    output = frame.copy()
    output["_original_row"] = np.arange(len(output), dtype=int)
    # Affibody identifiers are normalized to strings solely for the tie key;
    # the original column is preserved in the returned frame.
    output["_affibody_tie_key"] = output[affibody_column].astype(str)
    output = output.sort_values(
        [peptide_column, score_column, "_affibody_tie_key"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["within_peptide_rank"] = (
        output.groupby(peptide_column, sort=False).cumcount() + 1
    ).astype(int)
    return output.drop(columns=["_affibody_tie_key"])


def evaluate_wetlab_predictions(
    frame,
    peptide_column="peptide_design_code",
    affibody_column="affibody_design_code",
    score_column="score",
    binder_column="target_binder",
    retention_column="target_retention",
    k_values=DEFAULT_K_VALUES,
):
    """Evaluate one prediction vector globally and as wet-lab candidate lists.

    Returns ``(summary, per_peptide, ranked_examples)``.  Precision, hit rate,
    best retention, and regret are computed independently inside each peptide
    and macro-averaged, giving each peptide equal weight even when a matrix
    cell is missing.

    ``regret@k`` is the best measured retention available for a peptide minus
    the best measured retention among the model's top-k choices.  ``NDCG@k``
    uses linear retention gain and logarithmic rank discount.
    """
    _validate_frame(
        frame,
        peptide_column,
        affibody_column,
        score_column,
        binder_column,
        retention_column,
    )
    k_values = tuple(sorted(set(int(value) for value in k_values)))
    _require(k_values, "at least one k value is required")
    _require(all(value >= 1 for value in k_values), "all k values must be positive")

    work = frame.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="raise").astype(float)
    work[binder_column] = pd.to_numeric(work[binder_column], errors="raise").astype(int)
    work[retention_column] = pd.to_numeric(
        work[retention_column], errors="raise"
    ).astype(float)
    ranked = rank_within_peptide(
        work,
        peptide_column=peptide_column,
        affibody_column=affibody_column,
        score_column=score_column,
    )

    peptide_rows = []
    for peptide, group in ranked.groupby(peptide_column, sort=True):
        group = group.sort_values("within_peptide_rank", kind="mergesort")
        observed = group[retention_column].to_numpy(dtype=float)
        predicted = group[score_column].to_numpy(dtype=float)
        binder = group[binder_column].to_numpy(dtype=int)
        row = {
            peptide_column: peptide,
            "n_candidates": int(len(group)),
            "n_binders": int(binder.sum()),
            "experimental_best_retention": float(np.max(observed)),
            "within_peptide_spearman": _safe_spearman(observed, predicted),
            "top1_affibody": group.iloc[0][affibody_column],
        }
        for k in k_values:
            selected_n = min(int(k), len(group))
            selected_binder = binder[:selected_n]
            selected_retention = observed[:selected_n]
            best_retention = float(np.max(selected_retention))
            row["selected_n_at_{}".format(k)] = int(selected_n)
            row["precision_at_{}".format(k)] = float(np.mean(selected_binder))
            row["hit_at_{}".format(k)] = float(np.any(selected_binder == 1))
            row["best_retention_at_{}".format(k)] = best_retention
            row["regret_at_{}".format(k)] = float(np.max(observed) - best_retention)
            row["ndcg_at_{}".format(k)] = _linear_ndcg(observed, observed, k)
        peptide_rows.append(row)

    per_peptide = pd.DataFrame(peptide_rows)
    labels = work[binder_column].to_numpy(dtype=int)
    score = work[score_column].to_numpy(dtype=float)
    retention = work[retention_column].to_numpy(dtype=float)
    summary = {
        "n_examples": int(len(work)),
        "n_peptides": int(per_peptide.shape[0]),
        "n_binders": int(labels.sum()),
        "binder_fraction": float(labels.mean()),
        "tie_policy": TIE_POLICY,
        "global_auroc": (
            float(roc_auc_score(labels, score))
            if np.unique(labels).size == 2
            else float("nan")
        ),
        "global_average_precision": (
            float(average_precision_score(labels, score))
            if np.unique(labels).size == 2
            else float("nan")
        ),
        "global_spearman": _safe_spearman(retention, score),
        "distinct_top1_affibodies": int(per_peptide["top1_affibody"].nunique()),
    }
    within_rho = per_peptide["within_peptide_spearman"].to_numpy(dtype=float)
    finite_rho = within_rho[np.isfinite(within_rho)]
    summary["within_peptide_spearman_evaluable_groups"] = int(len(finite_rho))
    summary["within_peptide_spearman_mean"] = (
        float(np.mean(finite_rho)) if len(finite_rho) else float("nan")
    )
    for k in k_values:
        for metric in (
            "precision",
            "hit",
            "best_retention",
            "regret",
            "ndcg",
        ):
            column = "{}_at_{}".format(metric, k)
            values = per_peptide[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary["peptide_macro_{}".format(column)] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            summary["peptide_{}_evaluable_groups".format(column)] = int(len(finite))

    # Selection flags make the ranked table directly useful for saved
    # per-example prediction artifacts and wet-lab candidate sheets.
    for k in k_values:
        ranked["selected_at_{}".format(k)] = (
            ranked["within_peptide_rank"] <= int(k)
        ).astype(int)
    ranked = ranked.drop(columns=["_original_row"])
    _require(
        len(ranked) == len(frame),
        "ranking changed the number of evaluation examples",
    )
    return summary, per_peptide, ranked
