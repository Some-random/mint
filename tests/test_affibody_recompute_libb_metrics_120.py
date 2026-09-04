import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.recompute_libb_metrics_120 import (
    EXPECTED_BINDERS,
    EXPECTED_ROWS,
    evaluate_predictions,
    load_corrected_retention_panel,
    load_normalized_predictions,
    summarize_across_seeds,
    validate_prediction_membership,
)


def _complete_panel():
    peptides = ["AH"] + [f"P{index:02d}" for index in range(1, 12)]
    affibodies = ["LIFTK"] + [f"A{index:02d}" for index in range(1, 10)]
    rows = []
    for peptide_index, peptide in enumerate(peptides):
        for affibody_index, affibody in enumerate(affibodies):
            flat_index = peptide_index * len(affibodies) + affibody_index
            if flat_index == 0:
                retention = 87.94
            elif flat_index <= 60:
                retention = 80.0 + flat_index / 100.0
            else:
                retention = 10.0 + flat_index / 100.0
            rows.append(
                {
                    "pair_uid": f"row-{flat_index:03d}",
                    "library": "LibB",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": retention,
                    # Deliberately wrong: the loader must derive labels from
                    # retention rather than trusting this stale source field.
                    "target_binder": 0,
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == EXPECTED_ROWS
    assert int(frame["target_retention"].ge(75.0).sum()) == EXPECTED_BINDERS
    return frame


def _predictions(panel, model="model", seed="1"):
    return pd.DataFrame(
        {
            "eval_row_id": panel["eval_row_id"].astype(str),
            "model": model,
            "seed": seed,
            "score": panel["target_retention"].astype(float),
        }
    )


def test_complete_panel_is_loaded_and_binary_target_is_recomputed(tmp_path):
    path = tmp_path / "panel.csv"
    _complete_panel().to_csv(path, index=False)
    panel = load_corrected_retention_panel(path)
    assert len(panel) == 120
    assert panel["peptide_design_code"].nunique() == 12
    assert panel["affibody_design_code"].nunique() == 10
    assert panel["target_binder"].sum() == 61
    corrected = panel.loc[
        panel["peptide_design_code"].eq("AH")
        & panel["affibody_design_code"].eq("LIFTK")
    ].iloc[0]
    assert corrected["target_retention"] == pytest.approx(87.94)
    assert corrected["target_binder"] == 1


def test_exact_120_join_computes_requested_metrics_and_ah_row(tmp_path):
    panel_path = tmp_path / "panel.csv"
    _complete_panel().to_csv(panel_path, index=False)
    panel = load_corrected_retention_panel(panel_path)
    predictions = pd.concat(
        [
            _predictions(panel, "candidate", "1"),
            _predictions(panel, "candidate", "2"),
        ],
        ignore_index=True,
    )
    metrics, per_peptide, ranked = evaluate_predictions(predictions, panel)
    assert len(metrics) == 2
    assert metrics["global_auroc"].eq(1.0).all()
    assert metrics["global_average_precision"].eq(1.0).all()
    assert metrics["global_spearman"].eq(1.0).all()
    assert metrics["within_peptide_spearman_mean"].to_numpy() == pytest.approx(
        np.ones(2)
    )
    for k in (1, 3):
        assert metrics[f"peptide_macro_precision_at_{k}"].notna().all()
        assert metrics[f"peptide_macro_hit_at_{k}"].notna().all()
        assert metrics[f"peptide_macro_best_retention_at_{k}"].notna().all()
        assert metrics[f"peptide_macro_regret_at_{k}"].notna().all()
    assert len(per_peptide) == 24
    assert per_peptide["peptide_design_code"].eq("AH").sum() == 2
    assert per_peptide["n_candidates"].eq(10).all()
    assert len(ranked) == 240
    assert ranked.groupby(["model", "seed"]).size().eq(120).all()

    summary = summarize_across_seeds(metrics)
    assert summary.loc[0, "n_seeds"] == 2
    assert json.loads(summary.loc[0, "seeds"]) == ["1", "2"]
    assert summary.loc[0, "global_auroc_mean"] == 1.0
    assert summary.loc[0, "global_auroc_sd"] == 0.0


def test_seed_summary_preserves_undefined_correlation_without_failing():
    metrics = pd.DataFrame(
        {
            "model": ["constant", "constant"],
            "seed": ["1", "2"],
            **{
                metric: [0.5, 0.5]
                for metric in (
                    "global_auroc",
                    "global_average_precision",
                    "within_peptide_spearman_mean",
                    "peptide_macro_precision_at_1",
                    "peptide_macro_precision_at_3",
                    "peptide_macro_hit_at_1",
                    "peptide_macro_hit_at_3",
                    "peptide_macro_best_retention_at_1",
                    "peptide_macro_best_retention_at_3",
                    "peptide_macro_regret_at_1",
                    "peptide_macro_regret_at_3",
                )
            },
            "global_spearman": [np.nan, 0.25],
        }
    )
    summary = summarize_across_seeds(metrics)
    assert summary.loc[0, "global_spearman_mean"] == pytest.approx(0.25)
    assert np.isnan(summary.loc[0, "global_spearman_sd"])
    assert json.loads(summary.loc[0, "global_spearman_individual"]) == [None, 0.25]


def test_score_ties_use_affibody_identifier_and_are_order_independent(tmp_path):
    path = tmp_path / "panel.csv"
    _complete_panel().sample(frac=1.0, random_state=7).to_csv(path, index=False)
    panel = load_corrected_retention_panel(path)
    predictions = _predictions(panel)
    predictions["score"] = 0.5
    _, per_peptide, ranked = evaluate_predictions(predictions, panel)
    ah = per_peptide.loc[per_peptide["peptide_design_code"].eq("AH")].iloc[0]
    assert ah["top1_affibody"] == "A01"
    assert json.loads(ah["top3_affibodies"]) == ["A01", "A02", "A03"]
    ah_ranked = ranked.loc[ranked["peptide_design_code"].eq("AH")]
    assert ah_ranked["affibody_design_code"].tolist()[:3] == ["A01", "A02", "A03"]


def test_incomplete_panel_and_incomplete_prediction_group_are_rejected(tmp_path):
    incomplete_path = tmp_path / "incomplete.csv"
    _complete_panel().iloc[:-1].to_csv(incomplete_path, index=False)
    with pytest.raises(ValueError, match="120 rows"):
        load_corrected_retention_panel(incomplete_path)

    panel_path = tmp_path / "panel.csv"
    _complete_panel().to_csv(panel_path, index=False)
    panel = load_corrected_retention_panel(panel_path)
    with pytest.raises(ValueError, match="exact 120-row panel"):
        validate_prediction_membership(_predictions(panel).iloc[:-1], panel)


def test_prediction_loader_rejects_targets_and_nonfinite_scores(tmp_path):
    panel_path = tmp_path / "panel.csv"
    _complete_panel().to_csv(panel_path, index=False)
    panel = load_corrected_retention_panel(panel_path)
    leaked = _predictions(panel).assign(target_retention=1.0)
    leaked_path = tmp_path / "leaked.csv"
    leaked.to_csv(leaked_path, index=False)
    with pytest.raises(ValueError, match="schema must be exactly"):
        load_normalized_predictions([leaked_path])

    nonfinite = _predictions(panel)
    nonfinite.loc[0, "score"] = np.nan
    nonfinite_path = tmp_path / "nonfinite.csv"
    nonfinite.to_csv(nonfinite_path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_normalized_predictions([nonfinite_path])
