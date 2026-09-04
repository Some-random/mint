import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.summarize_libb_structural_evaluation import (
    ALL_SUMMARY_METRICS,
    EXPECTED_FALTA_NONOPTIMAL_PEPTIDES,
    build_falta_panel_context,
    build_main_model_specs,
    summarize_metrics,
    validate_selection_lock,
)


def _selection_lock(selected_rde="rde_b", selected_stab="stab_a"):
    def family(selected, losses):
        return {
            "selected_model": selected,
            "selected_pooled_weak_validation_log_loss": losses[selected],
            "candidates": [
                {
                    "model": model,
                    "eligible": True,
                    "pooled_weak_validation_log_loss": loss,
                }
                for model, loss in losses.items()
            ],
        }

    return {
        "schema_version": "libb-structural-weak-selection-v1",
        "retention_labels_read": False,
        "selection_metric": "pooled_weak_validation_log_loss",
        "selection_direction": "minimize",
        "selected_models_by_family": {
            "rde": selected_rde,
            "stab": selected_stab,
        },
        "families": {
            "rde": family(selected_rde, {"rde_a": 0.4, "rde_b": 0.3}),
            "stab": family(selected_stab, {"stab_a": 0.2, "stab_b": 0.5}),
        },
    }


def _panel_with_expected_falta_pattern():
    peptides = ["EL", "MW"] + [f"P{index}" for index in range(10)]
    affibodies = ["FALTA"] + [f"A{index}" for index in range(9)]
    rows = []
    for peptide_index, peptide in enumerate(peptides):
        for affibody_index, affibody in enumerate(affibodies):
            retention = 20.0 + affibody_index
            if affibody == "FALTA":
                retention = 80.0
                if peptide == "EL":
                    retention = 70.0
                elif peptide == "MW":
                    retention = 85.0
            if peptide in {"EL", "MW"} and affibody == "A0":
                retention = 90.0
            rows.append(
                {
                    "eval_row_id": f"row-{peptide_index}-{affibody_index}",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": retention,
                    "target_binder": int(retention >= 75.0),
                }
            )
    return pd.DataFrame(rows)


def test_selection_lock_is_authoritative_and_weak_only():
    lock = _selection_lock()
    assert validate_selection_lock(lock) == {"rde": "rde_b", "stab": "stab_a"}

    leaked = _selection_lock()
    leaked["retention_labels_read"] = True
    with pytest.raises(ValueError, match="retention was read"):
        validate_selection_lock(leaked)

    wrong = _selection_lock(selected_rde="rde_a")
    with pytest.raises(ValueError, match="weak-loss minimum"):
        validate_selection_lock(wrong)


def test_main_structural_names_come_from_lock_not_metrics():
    specs = build_main_model_specs({"rde": "locked_rde", "stab": "locked_stab"})
    keys = {item["model_key"] for item in specs}
    assert "structural::locked_rde" in keys
    assert "structural::locked_stab" in keys
    assert "structural::nonlinear_7site_control" in keys
    assert all("retention" not in item["selection_basis"].lower() or "not read" in item["selection_basis"].lower() for item in specs)


def test_falta_context_fails_closed_on_expected_el_mw_pattern():
    context = build_falta_panel_context(_panel_with_expected_falta_pattern())
    assert context["falta_is_binder"].sum() == 11
    assert context["falta_is_experimentally_optimal"].sum() == 10
    observed = tuple(
        sorted(
            context.loc[
                context["falta_is_experimentally_optimal"].eq(0),
                "peptide_design_code",
            ]
        )
    )
    assert observed == EXPECTED_FALTA_NONOPTIMAL_PEPTIDES
    mw = context.loc[context["peptide_design_code"].eq("MW")].iloc[0]
    assert json.loads(mw["experimental_best_affibodies"]) == ["A0"]


def test_common_seed_summary_uses_sample_sd_and_preserves_individuals():
    rows = []
    for seed, base in (("2", 0.2), ("1", 0.1)):
        row = {
            "bundle": "structural",
            "model": "candidate",
            "model_key": "structural::candidate",
            "seed": seed,
            "constant_prediction_score": 0,
        }
        for metric in ALL_SUMMARY_METRICS:
            row[metric] = base
        # P@3 must be an integer count out of 36.
        row["peptide_macro_precision_at_3"] = 18.0 / 36.0 if seed == "1" else 27.0 / 36.0
        rows.append(row)

    summary = summarize_metrics(pd.DataFrame(rows)).iloc[0]
    assert summary["within_peptide_spearman_mean_mean"] == pytest.approx(0.15)
    assert summary["within_peptide_spearman_mean_sd"] == pytest.approx(np.std([0.1, 0.2], ddof=1))
    individual = json.loads(summary["within_peptide_spearman_mean_individual"])
    assert [item["seed"] for item in individual] == ["1", "2"]
    counts = json.loads(summary["precision_at_3_binders_of_36_individual"])
    assert [item["binders"] for item in counts] == [18, 27]
