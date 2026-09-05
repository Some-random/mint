import math

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import audit_libb_model_consensus as consensus


def test_score_equal_to_cutoff_is_called_positive():
    frame = pd.DataFrame(
        {
            "mint_layer5": [0.25, 0.249999],
            "stab_designed_ordered_native_projection": [0.5, 0.6],
            "rde_network_designed_3fold_native_projection": [0.75, 0.8],
            "additive_7site": [0.1, 0.2],
        }
    )
    observed = consensus.add_calls(
        frame,
        {"mint": 0.25, "stab": 0.5, "rde": 0.75, "additive": 0.1},
        include_additive=True,
    )

    assert observed["call_mint"].tolist() == [1, 0]
    assert observed["call_stab"].tolist() == [1, 1]
    assert observed["call_rde"].tolist() == [1, 1]
    assert observed["call_additive"].tolist() == [1, 1]


def test_empty_call_set_has_nan_precision_not_zero():
    frame = pd.DataFrame({"target_binder": [0, 1, 1]})
    result = consensus._metric_counts(frame, [False, False, False])

    assert result["calls"] == 0
    assert result["true_positive"] == 0
    assert math.isnan(result["precision"])
    assert result["recall"] == 0.0


def test_f1_threshold_tie_prefers_precision_then_fewer_calls():
    result = consensus.select_f1_threshold([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1])

    assert result["threshold"] == 4.0
    assert result["calls"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2.0 / 3.0)


def test_canonical_locked_consensus_counts_and_same_budget_control(tmp_path):
    output = tmp_path / "private_data" / "lock_seal"
    cutoffs, seal = consensus.load_weak_cutoffs_before_retention(
        consensus.DEFAULT_LOCK_DIR,
        consensus.DEFAULT_WEAK_OOF,
        output,
    )
    assert seal["retention_sources_opened"] is False
    assert cutoffs["mint"] == pytest.approx(0.1451619880974242)
    assert cutoffs["stab"] == pytest.approx(0.30089325)
    assert cutoffs["rde"] == pytest.approx(0.427056985722644)

    frame = consensus.load_and_validate_predictions(
        consensus.DEFAULT_PREDICTIONS, consensus.DEFAULT_SOURCE_MANIFEST
    )
    assert len(frame) == 120
    assert int(frame["target_binder"].sum()) == 61
    assert consensus._membership_sha256(frame["eval_row_id"]) == consensus.EXPECTED_MEMBERSHIP_SHA256

    calls = consensus.add_calls(frame, cutoffs, include_additive=True)
    rules = consensus.primary_rules()
    summary = consensus.consensus_summary(calls, rules, "test")
    full = summary.loc[summary["panel_scope"].eq("complete_120")].set_index("call_rule")
    assert (int(full.loc["mint_and_stab", "calls"]), int(full.loc["mint_and_stab", "true_positive"])) == (84, 59)
    assert (int(full.loc["mint_and_stab_and_rde", "calls"]), int(full.loc["mint_and_stab_and_rde", "true_positive"])) == (77, 58)
    assert full.loc["mint_and_stab_and_rde", "precision"] == pytest.approx(58 / 77)
    assert full.loc["mint_and_stab_and_rde", "recall"] == pytest.approx(58 / 61)

    controls, exact = consensus.matched_budget_component_controls(
        calls, rules, draws=1000, seed=31
    )
    chosen = controls.loc[
        controls["panel_scope"].eq("complete_120")
        & controls["consensus_rule"].eq("mint_and_stab_and_rde")
        & controls["control_model"].eq("mint")
    ].iloc[0]
    assert int(chosen["calls_each"]) == 77
    assert int(chosen["consensus_true_positive"]) == 58
    assert int(chosen["control_true_positive"]) == 58
    grouped = exact.loc[
        exact["panel_scope"].eq("complete_120")
        & exact["consensus_rule"].eq("mint_and_stab_and_rde")
        & exact["control_model"].eq("mint")
    ].groupby("peptide_design_code")
    assert all(
        int(group["consensus_call"].sum())
        == int(group["matched_budget_control_call"].sum())
        for _, group in grouped
    )

    negative, negative_exact = consensus.matched_budget_negative_controls(calls)
    full_negative = negative.loc[negative["panel_scope"].eq("complete_120")]
    assert set(full_negative["rejected_pairs_each"]) == {24}
    assert set(full_negative["unanimous_negative_binders_rejected"]) == {0}
    assert set(full_negative["control_bottom_score_binders_rejected"]) == {0}
    assert set(full_negative["unanimous_negative_max_retention"]) == {63.66}
    assert set(full_negative["control_max_retention"]) == {63.66}
    for (_, control_model), group in negative_exact.loc[
        negative_exact["panel_scope"].eq("complete_120")
    ].groupby(["peptide_design_code", "control_model"]):
        assert int(group["unanimous_negative_reject"].sum()) == int(
            group["matched_budget_bottom_score_reject"].sum()
        ), control_model


def test_peptide_cluster_bootstrap_is_deterministic():
    rows = []
    for peptide_index in range(3):
        for affibody_index in range(2):
            binder = int(affibody_index == 0)
            rows.append(
                {
                    "peptide_design_code": f"P{peptide_index}",
                    "affibody_design_code": f"A{affibody_index}",
                    "target_binder": binder,
                    "call_mint": binder,
                    "call_stab": binder,
                    "call_rde": binder,
                }
            )
    frame = pd.DataFrame(rows)
    # The production routine deliberately enforces the 12-peptide panel.
    frame = pd.concat(
        [frame.assign(peptide_design_code=f"P{index}") for index in range(12)],
        ignore_index=True,
    )
    rules = consensus.primary_rules()
    first, first_delta = consensus.peptide_cluster_bootstrap(frame, rules, 1000, 17)
    second, second_delta = consensus.peptide_cluster_bootstrap(frame, rules, 1000, 17)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_delta, second_delta)
