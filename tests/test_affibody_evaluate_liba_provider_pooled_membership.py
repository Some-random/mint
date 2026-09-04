from __future__ import annotations

import pandas as pd

from downstream.AffibodyMHC.evaluate_liba_provider_pooled_membership import calculate


def test_calculate_membership_metrics_and_menu_extrapolation() -> None:
    panel = pd.DataFrame(
        {
            "pair_uid": ["a", "b", "c", "d"],
            "peptide_design_code": ["AA"] * 4,
            "target_binder": [True, True, False, False],
        }
    )
    scores = pd.DataFrame(
        {
            "eval_row_id": ["a", "b", "c", "d"],
            "peptide_design_code": ["AA"] * 4,
            "pooled_r009_r010_count": [20, 2, 15, 0],
        }
    )
    menu = pd.DataFrame(
        {
            "pair_uid": ["x", "y"],
            "peptide_design_code": ["AA", "AA"],
            "within_peptide_rank": [1, 2],
        }
    )

    metrics, composition = calculate(panel, scores, menu)

    row = metrics.iloc[0]
    assert int(row["true_positives"]) == 1
    assert int(row["false_positives"]) == 1
    assert int(row["false_negatives"]) == 1
    assert int(row["true_negatives"]) == 1
    assert float(row["precision"]) == 0.5
    assert float(row["recall"]) == 0.5
    assert float(row["extrapolated_expected_binders"]) == 1.0
    assert int(composition.iloc[0]["fixed_unmeasured_menu_rows"]) == 2
