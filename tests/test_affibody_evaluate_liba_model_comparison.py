import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import evaluate_liba_model_comparison as comparison


PEPTIDES = ["AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL"]
AFFIBODIES = ["A{:02d}".format(index) for index in range(12)]
BINDER_COUNTS = {"AF": 11, "DL": 0, "DP": 0, "EA": 0, "KF": 10, "LA": 0, "LL": 10, "NF": 4, "TL": 3}


def _complete_panel():
    rows = []
    for peptide_index, peptide in enumerate(PEPTIDES):
        binder_count = BINDER_COUNTS[peptide]
        for affibody_index, affibody in enumerate(AFFIBODIES):
            binder = affibody_index < binder_count
            # All peptide rows have continuous retention variation, including
            # the four rows with no >=75 binders.
            retention = (
                99.0 - affibody_index - peptide_index / 100.0
                if binder
                else 50.0 - affibody_index - peptide_index / 100.0
            )
            rows.append(
                {
                    "pair_uid": "{}-{}".format(peptide, affibody),
                    "library": "LibA",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": retention,
                    # Deliberately wrong: the loader must derive this field.
                    "target_binder": 0,
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 108
    assert sum(BINDER_COUNTS.values()) == 38
    return frame


def _loaded_panel(tmp_path):
    path = tmp_path / "panel.csv"
    _complete_panel().to_csv(path, index=False)
    return comparison.load_liba_panel(path)


def test_liba_panel_is_exact_and_rederives_75_percent_labels(tmp_path):
    panel = _loaded_panel(tmp_path)
    assert len(panel) == 108
    assert panel["peptide_design_code"].nunique() == 9
    assert panel["affibody_design_code"].nunique() == 12
    assert int(panel["target_binder"].sum()) == 38
    assert panel.groupby("peptide_design_code")["target_binder"].sum().to_dict() == BINDER_COUNTS


def test_perfect_scores_compute_requested_metrics_and_exclude_one_class_rows(tmp_path):
    panel = _loaded_panel(tmp_path)
    predictions = panel[["pair_uid"]].copy()
    predictions["model"] = "perfect"
    predictions["display_name"] = "Perfect"
    predictions["seed"] = "1"
    predictions["score"] = panel["target_retention"].to_numpy(dtype=float)

    aggregate, per_peptide, ranked = comparison.evaluate_models(predictions, panel)
    row = aggregate.iloc[0]
    assert row["within_peptide_average_precision_mean"] == pytest.approx(1.0)
    assert row["within_peptide_auroc_mean"] == pytest.approx(1.0)
    assert row["within_peptide_spearman_mean"] == pytest.approx(1.0)
    assert row["within_peptide_binary_evaluable_groups"] == 5
    assert json.loads(row["within_peptide_binary_excluded_groups"]) == ["DL", "DP", "EA", "LA"]
    assert row["precision_at_1"] == pytest.approx(5.0 / 9.0)
    assert row["precision_at_3"] == pytest.approx(5.0 / 9.0)
    assert row["regret_at_1"] == pytest.approx(0.0)
    assert row["regret_at_3"] == pytest.approx(0.0)
    assert row["global_auroc_secondary"] == pytest.approx(1.0)
    assert row["global_average_precision_secondary"] == pytest.approx(1.0)
    assert len(per_peptide) == 9
    assert len(ranked) == 108
    assert set(["selected_at_1", "selected_at_3"]).issubset(ranked.columns)


def test_configured_source_membership_is_audited_without_intersection(tmp_path):
    panel = _loaded_panel(tmp_path)
    source = tmp_path / "scores.csv"
    scores = panel[["pair_uid", "target_retention"]].rename(
        columns={"target_retention": "prediction"}
    )
    valid = scores.assign(arm="valid")
    incomplete = scores.iloc[:-1].assign(arm="incomplete")
    pd.concat([valid, incomplete], ignore_index=True).to_csv(source, index=False)
    config = {
        "schema_version": comparison.CONFIG_SCHEMA_VERSION,
        "sources": [
            {
                "model": "valid",
                "path": str(source),
                "id_column": "pair_uid",
                "score_column": "prediction",
                "seed": "0",
                "filters": {"arm": "valid"},
            },
            {
                "model": "incomplete",
                "path": str(source),
                "id_column": "pair_uid",
                "score_column": "prediction",
                "seed": "0",
                "filters": {"arm": "incomplete"},
            },
        ],
    }
    predictions, audit, _ = comparison.load_prediction_sources(config, panel)
    assert set(predictions["model"]) == {"valid"}
    valid_audit = audit.loc[audit["model"].eq("valid")].iloc[0]
    bad_audit = audit.loc[audit["model"].eq("incomplete")].iloc[0]
    assert valid_audit["status"] == "comparable_exact_108"
    assert bad_audit["missing_pair_ids"] == 1
    assert "incomparable" in bad_audit["status"]


def test_retrospective_threshold_is_explicit_and_tie_break_is_deterministic(tmp_path):
    tied = comparison.select_f1_threshold([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1])
    assert tied["threshold"] == 4.0
    assert tied["selected_candidates"] == 1
    assert tied["precision"] == 1.0
    assert tied["recall"] == 0.5
    assert tied["f1"] == pytest.approx(2.0 / 3.0)

    panel = _loaded_panel(tmp_path)
    predictions = panel[["pair_uid"]].copy()
    predictions["model"] = "perfect"
    predictions["display_name"] = "Perfect"
    predictions["seed"] = "1"
    predictions["score"] = panel["target_retention"].to_numpy(dtype=float)
    thresholds, decisions, recommendations = comparison.build_retrospective_recommendations(
        predictions, panel
    )
    threshold = thresholds.iloc[0]
    assert threshold["threshold_is_evaluation_selected"] == 1
    assert threshold["threshold_is_calibrated_probability"] == 0
    assert threshold["threshold_is_prospectively_validated"] == 0
    assert threshold["f1"] == pytest.approx(1.0)
    assert len(decisions) == 108
    assert len(recommendations) == 38


def test_recommendation_ranking_uses_affibody_id_for_score_ties(tmp_path):
    panel = _loaded_panel(tmp_path)
    predictions = panel[["pair_uid"]].copy().sample(frac=1.0, random_state=19)
    predictions["model"] = "constant"
    predictions["display_name"] = "Constant"
    predictions["seed"] = "1"
    predictions["score"] = 0.5
    _, decisions, _ = comparison.build_retrospective_recommendations(predictions, panel)
    af = decisions.loc[decisions["peptide_design_code"].eq("AF")]
    assert af.sort_values("within_peptide_score_rank")["affibody_design_code"].tolist() == AFFIBODIES
