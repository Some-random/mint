import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.summarize_libb_operational_metrics import (
    CALIBRATION_STATUS,
    build_ranking_metrics,
    build_retrospective_threshold_metrics,
    calculate_ranking_metrics,
    select_f1_threshold,
)


def test_peptide_conditioned_binary_metrics_exclude_single_class_peptide():
    frame = pd.DataFrame(
        [
            {
                "peptide_design_code": "mixed",
                "score": 0.1,
                "target_retention": 10.0,
                "target_binder": 0,
            },
            {
                "peptide_design_code": "mixed",
                "score": 0.9,
                "target_retention": 90.0,
                "target_binder": 1,
            },
            {
                "peptide_design_code": "all-negative",
                "score": 0.2,
                "target_retention": 10.0,
                "target_binder": 0,
            },
            {
                "peptide_design_code": "all-negative",
                "score": 0.8,
                "target_retention": 20.0,
                "target_binder": 0,
            },
        ]
    )

    metrics = calculate_ranking_metrics(frame)

    assert metrics["within_peptide_spearman"] == pytest.approx(1.0)
    assert metrics["within_peptide_spearman_evaluable_peptides"] == 2
    assert metrics["within_peptide_auroc"] == pytest.approx(1.0)
    assert metrics["within_peptide_average_precision"] == pytest.approx(1.0)
    assert metrics["within_peptide_binary_evaluable_peptides"] == 1
    assert json.loads(metrics["within_peptide_binary_excluded_peptides"]) == [
        "all-negative"
    ]


def test_f1_threshold_tie_prefers_higher_precision_and_fewer_candidates():
    # At thresholds 4 and 1, F1 is exactly 2/3.  Threshold 4 recommends only
    # the first positive (precision 1); threshold 1 recommends all four rows
    # (precision 1/2).  The documented tie-break must select threshold 4.
    result = select_f1_threshold(
        scores=[4.0, 3.0, 2.0, 1.0],
        labels=[1, 0, 0, 1],
    )

    assert result["threshold"] == 4.0
    assert result["selected_candidates"] == 1
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2.0 / 3.0)


def _mini_panel_and_predictions():
    panel = pd.DataFrame(
        [
            {
                "eval_row_id": "p1-falta",
                "peptide_design_code": "P1",
                "affibody_design_code": "FALTA",
                "target_retention": 90.0,
                "target_binder": 1,
            },
            {
                "eval_row_id": "p1-a",
                "peptide_design_code": "P1",
                "affibody_design_code": "AAAAA",
                "target_retention": 80.0,
                "target_binder": 1,
            },
            {
                "eval_row_id": "p2-falta",
                "peptide_design_code": "P2",
                "affibody_design_code": "FALTA",
                "target_retention": 10.0,
                "target_binder": 0,
            },
            {
                "eval_row_id": "p2-a",
                "peptide_design_code": "P2",
                "affibody_design_code": "AAAAA",
                "target_retention": 20.0,
                "target_binder": 0,
            },
        ]
    )
    scores_by_seed = {
        "1": [0.9, 0.7, 0.1, 0.2],
        "2": [0.7, 0.5, 0.3, 0.2],
    }
    frames = []
    for seed, scores in scores_by_seed.items():
        frame = panel.copy()
        frame["score"] = scores
        frame["model_order"] = 0
        frame["model_key"] = "model"
        frame["display_name"] = "Model"
        frame["source_model"] = "source"
        frame["model"] = "source"
        frame["seed"] = seed
        frames.append(frame)
    return panel, pd.concat(frames, ignore_index=True)


def test_threshold_exercise_averages_fits_and_reports_variable_candidate_counts():
    panel, selected = _mini_panel_and_predictions()
    summary, counts, decisions = build_retrospective_threshold_metrics(selected, panel)

    row = summary.iloc[0]
    assert row["n_fits_averaged"] == 2
    assert row["threshold"] == pytest.approx(0.6)
    assert row["selected_candidates"] == 2
    assert row["true_positives"] == 2
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["non_falta_selected_binders"] == 1
    assert row["non_falta_total_binders"] == 1
    assert row["non_falta_binder_recall"] == 1.0
    assert json.loads(row["zero_candidate_peptides"]) == ["P2"]
    assert json.loads(row["candidate_counts_per_peptide"]) == {"P1": 2, "P2": 0}
    assert set(counts["candidate_count"]) == {0, 2}
    assert decisions.loc[decisions["eval_row_id"].eq("p1-a"), "mean_score"].item() == pytest.approx(0.6)
    assert decisions["calibration_status"].eq(CALIBRATION_STATUS).all()


def test_seed_summary_uses_sample_sd_and_keeps_one_fit_sd_empty():
    panel, selected = _mini_panel_and_predictions()
    # Make P1 a mixed-class row so peptide-conditioned AUROC/AP are defined;
    # this test is about seed aggregation rather than panel-label validation.
    selected.loc[selected["eval_row_id"].eq("p1-a"), "target_binder"] = 0
    by_seed, summary = build_ranking_metrics(selected)

    assert len(by_seed) == 2
    row = summary.iloc[0]
    expected = by_seed["within_peptide_spearman"].to_numpy()
    assert row["within_peptide_spearman_mean"] == pytest.approx(expected.mean())
    assert row["within_peptide_spearman_sd"] == pytest.approx(np.std(expected, ddof=1))

    one_fit = selected[selected["seed"].eq("1")]
    _, one_summary = build_ranking_metrics(one_fit)
    assert one_summary.iloc[0]["n_fits"] == 1
    assert pd.isna(one_summary.iloc[0]["within_peptide_spearman_sd"])
