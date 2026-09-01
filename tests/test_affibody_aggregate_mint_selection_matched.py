import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_mint_selection_matched as aggregate
from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC.code_only_baseline import sha256_file


def _retention_panel(prefix="x"):
    return pd.DataFrame(
        {
            "pair_uid": ["{}-pair-{}".format(prefix, i) for i in range(6)],
            "chain1_sha256": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "chain2_sha256": ["a1", "a2", "a3", "a1", "a2", "a3"],
            "sequence_pair_sha256": ["{}-sequence-{}".format(prefix, i) for i in range(6)],
            "target_retention": [10.0, 80.0, 90.0, 20.0, 70.0, 100.0],
            "target_binder": [0, 1, 1, 0, 0, 1],
        }
    )


def _retention_predictions(library="LibA", seed=20260811, prefix="x"):
    panel = _retention_panel(prefix)
    probabilities = {
        "frozen_logistic": np.array([0.1, 0.8, 0.7, 0.2, 0.5, 0.9]),
        "head_only": np.array([0.2, 0.75, 0.6, 0.25, 0.55, 0.85]),
        "lora_cross": np.array([0.15, 0.85, 0.65, 0.3, 0.45, 0.95]),
    }
    blocks = []
    for arm, probability in probabilities.items():
        block = panel.copy()
        block["library"] = library
        block["arm"] = arm
        block["training_seed"] = -1 if arm == "frozen_logistic" else seed
        block["selected_epoch"] = 0 if arm == "frozen_logistic" else 2
        block["probability"] = probability
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _stored_metrics(predictions, library="LibA", seed=20260811):
    rows = []
    for arm, arm_seed in (
        ("frozen_logistic", -1),
        ("head_only", seed),
        ("lora_cross", seed),
    ):
        block = predictions.loc[
            predictions["arm"].eq(arm)
            & pd.to_numeric(predictions["training_seed"]).eq(arm_seed)
        ].reset_index(drop=True)
        rows.append(
            aggregate._canonical_retention_metrics(
                block,
                library,
                arm,
                arm_seed,
                int(block["selected_epoch"].iloc[0]),
            )
        )
    return pd.DataFrame(rows)


def _patch_small_retention_contract(monkeypatch, library, panel):
    contract = copy.deepcopy(matched.PRIMARY_CONTRACT)
    contract[library]["retention_rows"] = len(panel)
    contract[library]["retention_positive"] = int(panel["target_binder"].sum())
    contract[library]["retention_membership_sha256"] = cached_eval.membership_sha256(panel)
    monkeypatch.setattr(matched, "PRIMARY_CONTRACT", contract)


def test_audit_recomputes_every_stored_retention_metric(monkeypatch):
    predictions = _retention_predictions()
    metrics = _stored_metrics(predictions)
    _patch_small_retention_contract(
        monkeypatch, "LibA", predictions.loc[predictions["arm"].eq("frozen_logistic")]
    )
    observed, panel = aggregate.audit_retention_tables(
        predictions, metrics, "LibA", 20260811, "synthetic"
    )
    assert len(observed) == 3
    assert len(panel) == 6
    assert set(observed.columns) == set(metrics.columns)
    assert observed.loc[observed["arm"].eq("lora_cross"), "global_auprc"].iloc[0] == pytest.approx(
        metrics.loc[metrics["arm"].eq("lora_cross"), "global_auprc"].iloc[0]
    )


def test_audit_rejects_one_tampered_stored_metric(monkeypatch):
    predictions = _retention_predictions()
    metrics = _stored_metrics(predictions)
    metrics.loc[metrics["arm"].eq("head_only"), "within_affibody_p_at_3"] += 0.01
    _patch_small_retention_contract(
        monkeypatch, "LibA", predictions.loc[predictions["arm"].eq("frozen_logistic")]
    )
    with pytest.raises(ValueError, match="stored within_affibody_p_at_3 differs"):
        aggregate.audit_retention_tables(
            predictions, metrics, "LibA", 20260811, "tampered"
        )


def test_audit_rejects_target_panel_that_changes_between_arms(monkeypatch):
    predictions = _retention_predictions()
    metrics = _stored_metrics(predictions)
    changed = predictions.copy()
    changed.loc[
        changed["arm"].eq("lora_cross") & changed["pair_uid"].eq("x-pair-0"),
        "target_retention",
    ] = 11.0
    _patch_small_retention_contract(
        monkeypatch, "LibA", predictions.loc[predictions["arm"].eq("frozen_logistic")]
    )
    with pytest.raises(ValueError, match="target panel changes across arms"):
        aggregate.audit_retention_tables(
            changed, metrics, "LibA", 20260811, "changed-panel"
        )


def test_audit_rejects_binder_label_inconsistent_with_75_percent(monkeypatch):
    predictions = _retention_predictions()
    predictions.loc[predictions["pair_uid"].eq("x-pair-1"), "target_binder"] = 0
    metrics = _stored_metrics(predictions)
    _patch_small_retention_contract(
        monkeypatch,
        "LibA",
        predictions.loc[predictions["arm"].eq("frozen_logistic")],
    )
    with pytest.raises(ValueError, match="binder labels do not equal retention >= 75"):
        aggregate.audit_retention_tables(
            predictions, metrics, "LibA", 20260811, "wrong-threshold"
        )


def test_recorded_file_verification_detects_post_manifest_edit(tmp_path):
    path = tmp_path / "retention_metrics.csv"
    path.write_text("a\n1\n")
    record = {"path": str(path), "sha256": sha256_file(path)}
    aggregate._verified_recorded_file(record, path, "fixture")
    path.write_text("a\n2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate._verified_recorded_file(record, path, "fixture")


def test_audit_scalar_comparison_handles_epoch_metadata_strings_and_nan():
    assert aggregate._same_value("head_only", "head_only")
    assert not aggregate._same_value("lora_cross", "head_only")
    assert aggregate._same_value(float("nan"), float("nan"))


def _fake_metric_rows(library, seed, frozen_value=0.5):
    offset = {20260811: 0.0, 20260812: 0.1, 20260813: 0.2}[seed]
    rows = []
    for arm, arm_seed, value in (
        ("frozen_logistic", -1, frozen_value),
        ("head_only", seed, frozen_value + 0.1 + offset),
        ("lora_cross", seed, frozen_value + 0.15 + offset),
    ):
        row = {
            "library": library,
            "arm": arm,
            "training_seed": arm_seed,
            "selected_epoch": 0 if arm == "frozen_logistic" else 1,
        }
        for metric in aggregate.SUMMARY_METRICS:
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _fake_verified_run(library, seed):
    panel = _retention_panel(prefix=library.lower())
    predictions = _retention_predictions(library, seed, prefix=library.lower())
    # Frozen must be exactly the same across the three runs in a library.
    predictions.loc[predictions["arm"].eq("frozen_logistic"), "probability"] = np.array(
        [0.1, 0.8, 0.7, 0.2, 0.5, 0.9]
    )
    configuration = {
        "library": library,
        "training_seeds": [seed],
        "selected_C": 0.01 if library == "LibA" else 0.1,
        "head_lr": 1e-4,
        "adapter_lr": 2e-4,
    }
    return {
        "library": library,
        "training_seed": seed,
        "run_name": "seed{}".format(seed),
        "configuration": configuration,
        "runtime_contract": {"gpu_name": "synthetic-A100", "torch": "test"},
        "sources": {
            "script": {"path": "/unused", "sha256": "common-script"},
            "checkpoint": {"path": "/unused", "sha256": "common-checkpoint"},
            "reference_manifest_json": {
                "path": "/unused",
                "sha256": "{}-reference".format(library),
            },
        },
        "reference": {
            "directory": "/unused/{}".format(library),
            "selected_c": configuration["selected_C"],
        },
        "panel": panel,
        "predictions": predictions,
        "metrics": _fake_metric_rows(library, seed),
    }


def _six_fake_runs():
    return [
        _fake_verified_run(library, seed)
        for library in aggregate.EXPECTED_LIBRARIES
        for seed in aggregate.EXPECTED_TRAINING_SEEDS
    ]


def test_six_run_combine_deduplicates_deterministic_frozen_and_keeps_seed_pairs():
    combined = aggregate.combine_verified_runs(_six_fake_runs())
    metrics = combined["metrics"]
    for library in aggregate.EXPECTED_LIBRARIES:
        library_rows = metrics.loc[metrics["library"].eq(library)]
        assert len(library_rows) == 7
        assert len(library_rows.loc[library_rows["arm"].eq("frozen_logistic")]) == 1
        assert len(library_rows.loc[library_rows["arm"].eq("head_only")]) == 3
        assert len(library_rows.loc[library_rows["arm"].eq("lora_cross")]) == 3


def test_six_run_combine_rejects_missing_or_duplicate_expected_seed():
    runs = _six_fake_runs()
    runs[-1] = copy.deepcopy(runs[-2])
    with pytest.raises(ValueError, match="duplicate library/training-seed"):
        aggregate.combine_verified_runs(runs)


def test_six_run_combine_rejects_runtime_mismatch():
    runs = _six_fake_runs()
    runs[-1]["runtime_contract"] = {
        "gpu_name": "different-GPU",
        "torch": "test",
    }
    with pytest.raises(ValueError, match="different GPU/software runtime"):
        aggregate.combine_verified_runs(runs)


def test_paired_change_is_computed_per_seed_and_summary_uses_sample_sd():
    metrics = aggregate.combine_verified_runs(_six_fake_runs())["metrics"]
    paired = aggregate.build_paired_changes(metrics)
    lora_minus_head = paired.loc[
        paired["comparison"].eq("lora_cross_minus_head_only")
    ]
    assert len(lora_minus_head) == 6
    assert np.allclose(lora_minus_head["global_auroc_change"], 0.05)

    summary = aggregate.build_combined_summary(metrics, paired)
    head = summary.loc[
        summary["library"].eq("LibA")
        & summary["result_type"].eq("model_score")
        & summary["result"].eq("head_only")
    ].iloc[0]
    assert head["global_auroc_mean"] == pytest.approx(0.7)
    assert head["global_auroc_sample_sd"] == pytest.approx(0.1)
    frozen = summary.loc[
        summary["library"].eq("LibA")
        & summary["result_type"].eq("model_score")
        & summary["result"].eq("frozen_logistic")
    ].iloc[0]
    assert frozen["n_seed_runs"] == 1
    assert math.isnan(frozen["global_auroc_sample_sd"])
    paired_summary = summary.loc[
        summary["library"].eq("LibA")
        & summary["result_type"].eq("paired_change")
        & summary["result"].eq("lora_cross")
        & summary["reference"].eq("head_only")
    ].iloc[0]
    assert paired_summary["global_auroc_mean"] == pytest.approx(0.05)
    assert paired_summary["global_auroc_sample_sd"] == pytest.approx(0.0, abs=1e-15)


def test_markdown_calls_auprc_ap_and_states_frozen_has_no_seed_sd():
    metrics = aggregate.combine_verified_runs(_six_fake_runs())["metrics"]
    paired = aggregate.build_paired_changes(metrics)
    summary = aggregate.build_combined_summary(metrics, paired)
    markdown = aggregate.summary_markdown(summary, 6)
    assert "| AP |" in markdown
    assert "no seed SD is shown" in markdown
    assert "LoRA + head arm − Head-only arm" in markdown
    assert "Epochs (model/reference)" in markdown
