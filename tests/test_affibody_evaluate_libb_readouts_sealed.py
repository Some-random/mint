import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.evaluate_libb_readouts_sealed import (
    BLINDED_PREDICTION_COLUMNS,
    DEFAULT_RETENTION_AUDIT,
    EXPECTED_AFFIBODIES,
    EXPECTED_BINDERS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_MISSING_MATRIX_CELLS,
    EXPECTED_NONBINDERS,
    EXPECTED_PEPTIDES,
    discover_current_controls,
    evaluate_matched_predictions,
    load_blinded_predictions,
    load_libb_retention_audit,
    load_lora_control,
    load_site_and_frozen_controls,
    publish_evaluation,
    summarize_across_seeds,
    validate_blinded_prediction_groups,
)


def _audit_frame():
    # p2/d is deliberately absent: final evaluation must preserve an
    # irregular panel rather than fill a rectangular matrix.
    return pd.DataFrame(
        {
            "eval_row_id": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
            "peptide_design_code": ["p1"] * 4 + ["p2"] * 3,
            "affibody_design_code": ["a", "b", "c", "d", "a", "b", "c"],
            "target_retention": [100.0, 80.0, 20.0, 0.0, 10.0, 90.0, 50.0],
            "target_binder": [1, 1, 0, 0, 0, 1, 0],
        }
    )


def _prediction_frame(model="candidate", seed="1", role="candidate"):
    return pd.DataFrame(
        {
            "eval_row_id": ["x1", "x2", "x3", "x4", "x5", "x6", "x7"],
            "model": model,
            "seed": seed,
            "score": [0.9, 0.7, 0.8, 0.1, 0.5, 0.5, 0.4],
            "source_role": role,
        }
    )


def test_blinded_loader_accepts_only_the_four_column_contract(tmp_path):
    valid = _prediction_frame().drop(columns="source_role")
    path = tmp_path / "valid.csv"
    valid.to_csv(path, index=False)
    loaded = load_blinded_predictions([path])
    assert tuple(loaded.columns) == BLINDED_PREDICTION_COLUMNS + ("source_role",)
    assert loaded["source_role"].eq("candidate").all()

    leaked = valid.assign(target_retention=100.0)
    leaked_path = tmp_path / "leaked.csv"
    leaked.to_csv(leaked_path, index=False)
    with pytest.raises(ValueError, match="Targets and sequence identities are forbidden"):
        load_blinded_predictions([leaked_path])


def test_prediction_panels_require_no_duplicate_and_identical_opaque_ids():
    first = _prediction_frame()
    duplicate = pd.concat([first, first.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate opaque"):
        validate_blinded_prediction_groups(duplicate)

    second = _prediction_frame(model="second")
    second.loc[0, "eval_row_id"] = "different"
    with pytest.raises(ValueError, match="identical opaque ID sets"):
        validate_blinded_prediction_groups(pd.concat([first, second], ignore_index=True))


def test_retention_audit_filters_libb_measured_rows_and_checks_threshold(tmp_path):
    audit = _audit_frame().rename(columns={"eval_row_id": "pair_uid"})
    audit.insert(0, "library", "LibB")
    unmeasured = audit.iloc[[0]].copy()
    unmeasured["pair_uid"] = "missing"
    unmeasured["target_retention"] = ""
    unmeasured["target_binder"] = ""
    liba = audit.iloc[[0]].copy()
    liba["library"] = "LibA"
    liba["pair_uid"] = "liba"
    combined = pd.concat([audit, unmeasured, liba], ignore_index=True)
    path = tmp_path / "audit.csv"
    combined.to_csv(path, index=False)

    loaded = load_libb_retention_audit(
        path,
        expected_rows=7,
        expected_peptides=2,
        expected_affibodies=4,
        expected_missing_cells=1,
    )
    assert set(loaded["eval_row_id"]) == set(_audit_frame()["eval_row_id"])
    assert len(loaded) == 7

    bad = combined.copy()
    bad.loc[bad["pair_uid"].eq("x1"), "target_binder"] = "0"
    bad_path = tmp_path / "bad_audit.csv"
    bad.to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="does not equal retention >=75"):
        load_libb_retention_audit(bad_path)


def test_retention_audit_accepts_row_id_sidecar_key(tmp_path):
    audit = _audit_frame().rename(columns={"eval_row_id": "row_id"})
    path = tmp_path / "audit_sidecar.csv"
    audit.to_csv(path, index=False)
    loaded = load_libb_retention_audit(
        path,
        expected_rows=7,
        expected_peptides=2,
        expected_affibodies=4,
        expected_missing_cells=1,
    )
    assert "eval_row_id" in loaded
    assert "row_id" not in loaded
    assert set(loaded["eval_row_id"]) == set(audit["row_id"])


def test_default_corrected_private_audit_is_complete_if_present():
    if not DEFAULT_RETENTION_AUDIT.is_file():
        pytest.skip("corrected private LibB panel is unavailable")
    loaded = load_libb_retention_audit(
        DEFAULT_RETENTION_AUDIT,
        expected_rows=EXPECTED_EVALUATION_ROWS,
        expected_peptides=EXPECTED_PEPTIDES,
        expected_affibodies=EXPECTED_AFFIBODIES,
        expected_missing_cells=EXPECTED_MISSING_MATRIX_CELLS,
        expected_binders=EXPECTED_BINDERS,
        expected_nonbinders=EXPECTED_NONBINDERS,
    )
    assert len(loaded) == 120
    assert loaded["target_binder"].value_counts().to_dict() == {1: 61, 0: 59}
    target = loaded.loc[
        loaded["peptide_design_code"].eq("AH")
        & loaded["affibody_design_code"].eq("LIFTK")
    ]
    assert len(target) == 1
    assert target.iloc[0]["target_retention"] == pytest.approx(87.94)


def test_historical_119_row_controls_are_not_auto_discovered():
    controls, paths = discover_current_controls()
    assert controls.empty
    assert paths == []


def test_final_merge_computes_all_required_metrics_on_matched_panel():
    candidate = _prediction_frame()
    control = _prediction_frame("control", "-1", "control")
    predictions = pd.concat([candidate, control], ignore_index=True)
    metrics, peptide, ranked = evaluate_matched_predictions(predictions, _audit_frame())

    assert set(metrics["model"]) == {"candidate", "control"}
    assert metrics["global_auroc"].notna().all()
    assert metrics["global_average_precision"].notna().all()
    assert metrics["global_spearman"].notna().all()
    assert metrics["within_peptide_spearman_mean"].notna().all()
    for k in (1, 3):
        assert metrics["peptide_macro_precision_at_{}".format(k)].notna().all()
        assert metrics["peptide_macro_hit_at_{}".format(k)].notna().all()
        assert metrics["peptide_macro_best_retention_at_{}".format(k)].notna().all()
        assert metrics["peptide_macro_regret_at_{}".format(k)].notna().all()
    assert metrics["peptide_macro_ndcg_at_3"].notna().all()
    assert metrics["distinct_top1_affibodies"].eq(1).all()
    assert len(peptide) == 4
    assert len(ranked) == 14
    assert ranked.groupby(["model", "seed"]).size().eq(7).all()
    assert not bool(
        ranked.duplicated(["model", "seed", "eval_row_id"], keep=False).any()
    )


def test_final_merge_rejects_prediction_audit_membership_mismatch():
    audit = _audit_frame()
    audit.loc[0, "eval_row_id"] = "different"
    with pytest.raises(ValueError, match="prediction and retention audit"):
        evaluate_matched_predictions(_prediction_frame(), audit)


def test_legacy_control_adapters_drop_embedded_targets(tmp_path):
    ids = ["x{}".format(index) for index in range(1, 4)]
    baseline_rows = []
    for representation in ("site", "frozen_mint_chain_mean"):
        for index, row_id in enumerate(ids):
            baseline_rows.append(
                {
                    "pair_uid": row_id,
                    "library": "LibB",
                    "regime": "double_cold",
                    "cleaning": "c0",
                    "balance": "all_class_weighted",
                    "balance_seed": "-1",
                    "representation": representation,
                    "binder_probability": 0.1 + index / 10.0,
                    # These legacy columns exist but must not survive adapter.
                    "target_retention": 100.0,
                    "target_binder": 1,
                }
            )
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    baseline = load_site_and_frozen_controls(baseline_path)
    assert set(baseline["model"]) == {
        "site_additive_control",
        "frozen_mint_control",
    }
    assert "target_retention" not in baseline
    assert "target_binder" not in baseline

    lora_path = tmp_path / "lora.csv"
    pd.DataFrame(
        {
            "pair_uid": ids,
            "library": "LibB",
            "arm": "lora_cross",
            "training_seed": "7",
            "epoch": "1",
            "probability": [0.1, 0.2, 0.3],
            "target_retention": [0.0, 50.0, 100.0],
            "target_binder": [0, 0, 1],
        }
    ).to_csv(lora_path, index=False)
    lora = load_lora_control(lora_path)
    assert lora["model"].eq("lora_mint_one_epoch_control").all()
    assert "target_retention" not in lora
    assert "target_binder" not in lora


def test_seed_summary_and_private_outputs_are_machine_readable(tmp_path):
    first = _prediction_frame(seed="1")
    second = _prediction_frame(seed="2")
    second["score"] = second["score"] + np.linspace(0.0, 0.01, len(second))
    metrics, peptide, ranked = evaluate_matched_predictions(
        pd.concat([first, second], ignore_index=True), _audit_frame()
    )
    summary = summarize_across_seeds(metrics)
    assert summary.loc[0, "n_seeds"] == 2
    assert json.loads(summary.loc[0, "seeds"]) == ["1", "2"]
    assert np.isfinite(summary.loc[0, "global_auroc_seed_sd"])

    output = tmp_path / "final"
    publish_evaluation(
        output,
        {
            "matched_metrics_by_seed.csv": metrics,
            "matched_metrics_seed_summary.csv": summary,
            "per_peptide_metrics.csv": peptide,
            "ranked_per_pair.csv": ranked,
        },
        {"schema_version": "test"},
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["outputs"]) == {
        "matched_metrics_by_seed.csv",
        "matched_metrics_seed_summary.csv",
        "per_peptide_metrics.csv",
        "ranked_per_pair.csv",
    }
    with pytest.raises(ValueError, match="refusing overwrite"):
        publish_evaluation(output, {}, {})
