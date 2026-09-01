import math

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.wetlab_metrics import (
    TIE_POLICY,
    evaluate_wetlab_predictions,
)


def _panel():
    # The second peptide intentionally lacks Affibody d.  This is the same
    # kind of irregularity as LibB's 119/120 measured matrix and proves the
    # evaluator does not reshape or impute the missing cell.
    return pd.DataFrame(
        {
            "peptide_design_code": ["p1"] * 4 + ["p2"] * 3,
            "affibody_design_code": ["a", "b", "c", "d", "a", "b", "c"],
            "target_retention": [100.0, 80.0, 20.0, 0.0, 10.0, 90.0, 50.0],
            "target_binder": [1, 1, 0, 0, 0, 1, 0],
            # p1 order: a, c, b, d.  p2 has a score tie resolved as a, b, c.
            "score": [0.9, 0.7, 0.8, 0.1, 0.5, 0.5, 0.4],
        }
    )


def test_wetlab_metrics_on_irregular_panel_have_literal_definitions():
    summary, peptide, ranked = evaluate_wetlab_predictions(_panel())

    assert summary["n_examples"] == 7
    assert summary["n_peptides"] == 2
    assert summary["tie_policy"] == TIE_POLICY
    assert summary["global_auroc"] == pytest.approx(19.0 / 24.0)
    assert summary["global_average_precision"] == pytest.approx(34.0 / 45.0)

    rows = peptide.set_index("peptide_design_code")
    assert rows.loc["p1", "precision_at_1"] == 1.0
    assert rows.loc["p1", "precision_at_3"] == pytest.approx(2.0 / 3.0)
    assert rows.loc["p1", "hit_at_3"] == 1.0
    assert rows.loc["p1", "best_retention_at_1"] == 100.0
    assert rows.loc["p1", "best_retention_at_3"] == 100.0
    assert rows.loc["p1", "regret_at_1"] == 0.0
    assert rows.loc["p1", "regret_at_3"] == 0.0

    # The p2 score tie is resolved a before b, so top-1 misses its binder.
    assert rows.loc["p2", "top1_affibody"] == "a"
    assert rows.loc["p2", "precision_at_1"] == 0.0
    assert rows.loc["p2", "hit_at_1"] == 0.0
    assert rows.loc["p2", "precision_at_3"] == pytest.approx(1.0 / 3.0)
    assert rows.loc["p2", "hit_at_3"] == 1.0
    assert rows.loc["p2", "best_retention_at_1"] == 10.0
    assert rows.loc["p2", "best_retention_at_3"] == 90.0
    assert rows.loc["p2", "regret_at_1"] == 80.0
    assert rows.loc["p2", "regret_at_3"] == 0.0

    assert summary["peptide_macro_precision_at_1"] == 0.5
    assert summary["peptide_macro_precision_at_3"] == 0.5
    assert summary["peptide_macro_hit_at_1"] == 0.5
    assert summary["peptide_macro_hit_at_3"] == 1.0
    assert summary["peptide_macro_best_retention_at_1"] == 55.0
    assert summary["peptide_macro_best_retention_at_3"] == 95.0
    assert summary["peptide_macro_regret_at_1"] == 40.0
    assert summary["peptide_macro_regret_at_3"] == 0.0
    assert summary["distinct_top1_affibodies"] == 1

    assert ranked.groupby("peptide_design_code")["selected_at_3"].sum().to_dict() == {
        "p1": 3,
        "p2": 3,
    }


def test_ndcg_at_3_uses_linear_retention_gain_and_rank_discount():
    _, peptide, _ = evaluate_wetlab_predictions(_panel())
    rows = peptide.set_index("peptide_design_code")
    p1_dcg = 100.0 + 20.0 / math.log2(3.0) + 80.0 / math.log2(4.0)
    p1_ideal = 100.0 + 80.0 / math.log2(3.0) + 20.0 / math.log2(4.0)
    p2_dcg = 10.0 + 90.0 / math.log2(3.0) + 50.0 / math.log2(4.0)
    p2_ideal = 90.0 + 50.0 / math.log2(3.0) + 10.0 / math.log2(4.0)
    assert rows.loc["p1", "ndcg_at_3"] == pytest.approx(p1_dcg / p1_ideal)
    assert rows.loc["p2", "ndcg_at_3"] == pytest.approx(p2_dcg / p2_ideal)


def test_tie_break_is_deterministic_and_independent_of_input_order():
    panel = _panel()
    first = evaluate_wetlab_predictions(panel)[2]
    shuffled = panel.sample(frac=1.0, random_state=19).reset_index(drop=True)
    second = evaluate_wetlab_predictions(shuffled)[2]
    columns = [
        "peptide_design_code",
        "affibody_design_code",
        "within_peptide_rank",
    ]
    assert first[columns].equals(second[columns])


def test_constant_score_group_is_excluded_only_from_spearman_mean():
    panel = _panel()
    panel.loc[panel["peptide_design_code"].eq("p2"), "score"] = 0.5
    summary, peptide, _ = evaluate_wetlab_predictions(panel)
    p2 = peptide.set_index("peptide_design_code").loc["p2"]
    assert np.isnan(p2["within_peptide_spearman"])
    assert summary["within_peptide_spearman_evaluable_groups"] == 1
    assert np.isfinite(summary["within_peptide_spearman_mean"])
    assert np.isfinite(summary["peptide_macro_precision_at_3"])


def test_all_zero_retention_has_undefined_ndcg_but_other_metrics_survive():
    panel = _panel()
    panel.loc[panel["peptide_design_code"].eq("p2"), "target_retention"] = 0.0
    summary, peptide, _ = evaluate_wetlab_predictions(panel)
    p2 = peptide.set_index("peptide_design_code").loc["p2"]
    assert np.isnan(p2["ndcg_at_3"])
    assert summary["peptide_ndcg_at_3_evaluable_groups"] == 1


def test_duplicate_pair_and_nonfinite_score_are_rejected():
    duplicate = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_wetlab_predictions(duplicate)

    nonfinite = _panel()
    nonfinite.loc[0, "score"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_wetlab_predictions(nonfinite)
