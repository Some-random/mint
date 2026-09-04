from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import hashlib
import json

from downstream.AffibodyMHC import (
    audit_libb_native_projection_candidate_scorer_parity as parity,
)
from downstream.AffibodyMHC import build_libb_native_projection_candidate_handoff as handoff


def _reference_files(root, family: str, corrupt: str | None = None):
    spec = parity.FAMILY_SPECS[family]
    if family == "rde":
        template = "rde_network_designed_3fold_ensemble_native_projection__seed{seed}.csv"
    else:
        template = "stab_designed_ordered__seed{seed}.csv"
    for seed in parity.SEEDS:
        score = np.linspace(0.1, 0.9, 120)
        if corrupt == "nan" and seed == parity.SEEDS[0]:
            score[0] = np.nan
        frame = pd.DataFrame({
            "eval_row_id": [f"pair-{index:03d}" for index in range(120)],
            "model": spec["reference_model"],
            "seed": seed,
            "score": score,
        })
        frame.to_csv(root / template.format(seed=seed), index=False)


def _parity_frames(stored_offset: float = 0.0):
    pair_uid = [f"pair-{index:03d}" for index in range(120)]
    peptides = np.repeat(parity.EXPECTED_TARGETS, 10)
    affibodies = [f"A{index % 10:04d}" for index in range(120)]
    base = np.tile(np.linspace(0.1, 0.9, 10), 12)
    inputs = pd.DataFrame({
        "candidate_row_index": np.tile(np.arange(10), 12),
        "pair_uid": pair_uid,
        "peptide_design_code": peptides,
        "affibody_design_code": affibodies,
    })
    candidate = inputs.copy()
    reference = pd.DataFrame({"pair_uid": pair_uid})
    for seed in parity.SEEDS:
        candidate[f"score_seed_{seed}"] = base
        reference[f"reference_seed_{seed}"] = base
    candidate["score_mean_probability"] = base
    candidate["score_mean_logit"] = base
    candidate.loc[0, "score_mean_logit"] += stored_offset
    candidate["score_seed_sd"] = 0.0
    aggregate_column = parity.FAMILY_SPECS["rde"]["aggregate_column"]
    aggregate = inputs.drop(columns="candidate_row_index").copy()
    aggregate[aggregate_column] = base
    return inputs, candidate, reference, aggregate


def test_reference_seed_nan_fails_closed(tmp_path):
    _reference_files(tmp_path, "rde", corrupt="nan")
    with pytest.raises(ValueError, match="non-finite"):
        parity._load_reference_scores(tmp_path, "rde")


def test_saved_aggregate_nan_fails_closed(tmp_path):
    frame = pd.DataFrame({
        "eval_row_id": [f"pair-{index:03d}" for index in range(120)],
        "peptide_design_code": np.repeat(parity.EXPECTED_TARGETS, 10),
        "affibody_design_code": [f"A{index % 10:04d}" for index in range(120)],
        parity.FAMILY_SPECS["rde"]["aggregate_column"]: np.linspace(0.1, 0.9, 120),
        parity.FAMILY_SPECS["stab"]["aggregate_column"]: np.linspace(0.2, 0.8, 120),
    })
    frame.loc[0, parity.FAMILY_SPECS["rde"]["aggregate_column"]] = np.nan
    path = tmp_path / "aggregate.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        parity._load_aggregate_reference(path)


def test_stored_deployment_aggregate_uses_declared_tolerance():
    inputs, candidate, reference, aggregate = _parity_frames(stored_offset=1.5e-6)
    with pytest.raises(ValueError, match="aggregate parity exceeds tolerance"):
        parity._audit_family(
            "rde", inputs, candidate, reference, aggregate, tolerance=1e-6
        )


def test_exact_candidate_reference_parity_passes():
    inputs, candidate, reference, aggregate = _parity_frames()
    seed, summary, pairs = parity._audit_family(
        "rde", inputs, candidate, reference, aggregate, tolerance=1e-6
    )
    assert seed["within_tolerance"].all()
    assert summary.filter(like="within_tolerance").iloc[0].all()
    assert summary.filter(like="rankings_identical").iloc[0].all()
    assert len(pairs) == 120


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _handoff_parity_fixture(tmp_path, maximum=5e-13, producer_sha="a" * 64):
    audit_dir = tmp_path / "parity"
    audit_dir.mkdir()
    output_specs = {
        "production_batch_replay.csv": 16,
        "historical_tail_compatibility.csv": 12,
        "corrected120_cross_batch_seed.csv": 10,
        "corrected120_cross_batch_aggregate.csv": 2,
        "primary_top10_boundary_by_peptide.csv": 12,
    }
    output_records = {}
    for name, rows in output_specs.items():
        path = audit_dir / name
        pd.DataFrame({"row": range(rows)}).to_csv(path, index=False)
        output_records[name] = {"path": name, "rows": rows, "sha256": _sha256(path)}
    report_path = audit_dir / "report.md"
    report_path.write_text("parity report\n", encoding="utf-8")
    output_records["report.md"] = {"path": report_path.name, "sha256": _sha256(report_path)}
    asset_path = tmp_path / "asset.bin"
    asset_path.write_bytes(b"asset")
    asset = {"path": str(asset_path.resolve()), "sha256": _sha256(asset_path), "bytes": 5}
    production_dirs = {}
    exhaustive_receipts = {}
    for family in ("rde", "stab"):
        directory = tmp_path / family
        directory.mkdir()
        production_dirs[family] = directory
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": "libb-fixed-structure-candidate-scores-merged-v1",
            "family": family,
            "candidate_universe": {"candidate_rows": 4_447_848},
        }), encoding="utf-8")
        exhaustive_receipts[family] = {
            "path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)
        }
    production_families = {
        family: {
            "status": "pass", "rows": 640,
            "verified_batch_size": {"rde": 128, "stab": 20}[family],
            "maximum_absolute_probability_difference": maximum,
            "all_columns_within_1e-6": True, "all_rankings_identical": True,
            **{name: asset for name in (
                "invocation_manifest", "invocation_wrapper", "replay_receipt",
                "replay_chunk", "merged_partition",
            )},
        }
        for family in ("rde", "stab")
    }
    historical_families = {
        family: {
            "status": "pass", "rows": 120,
            "maximum_seed_probability_difference": maximum,
            "maximum_aggregate_probability_difference": maximum,
            "all_seed_within_peptide_rankings_identical": True,
            "aggregate_within_peptide_and_global_rankings_identical": True,
            "replay_manifest": asset, "replay_predictions": asset,
        }
        for family in ("rde", "stab")
    }
    cross_families = {
        family: {
            "status": "pass",
            "maximum_seed_probability_difference": maximum,
            "maximum_aggregate_probability_difference": maximum,
            "all_seed_and_aggregate_within_peptide_rankings_identical": True,
            "aggregate_global_ranking_identical": True,
            "cutoff_decision_flips": 0,
        }
        for family in ("rde", "stab")
    }
    manifest = {
        "schema_version": "libb-native-projection-release-gates-v1",
        "status": "pass",
        "thresholds": {
            "production_probability_tolerance": 1e-6,
            "historical_seed_probability_tolerance": 2e-6,
            "historical_aggregate_probability_tolerance": 1e-6,
            "cross_batch_probability_tolerance": 1e-4,
        },
        "outcome_access": {
            "retention_values_loaded": False, "binder_values_loaded": False,
            "selection_labels_loaded_for_release_decision": False,
            "weak_metadata_loaded_for_release_decision": False,
        },
        "gates": {
            "production_batch_replay": {"status": "pass", "families": production_families},
            "historical_tail_compatibility": {"status": "pass", "families": historical_families},
            "corrected120_cross_batch_compatibility": {"status": "pass", "families": cross_families},
            "primary_top10_boundary": {
                "status": "pass", "peptides": 12, "candidate_rows_rescored": 600,
                "verified_alternative_batch_size": 16,
                "all_top10_membership_identical": True, "all_top10_order_identical": True,
                "minimum_batch16_rank10_vs_rank11_logit_gap": 0.002,
                "maximum_rde_logit_difference_top50": 0.0003,
                "maximum_weighted_ensemble_logit_difference_top50": 0.00015,
                "minimum_selected_probability_margin_above_cutoff": 0.2,
                **{name: asset for name in (
                    "selection_manifest", "production_review_pool", "boundary_input_manifest",
                    "boundary_replay_invocation", "boundary_wrapper", "boundary_worker_receipt",
                )},
            },
        },
        "exhaustive_manifests": exhaustive_receipts,
        "producer_start": asset, "producer_end": asset,
        "imported_panel_auditor_start": asset, "imported_panel_auditor_end": asset,
        "outputs": output_records,
    }
    (audit_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return audit_dir, production_dirs


def test_handoff_parity_loader_cross_binds_and_checks_numeric_maxima(tmp_path):
    audit_dir, production_dirs = _handoff_parity_fixture(tmp_path)
    result = handoff._load_parity_audit(audit_dir, production_dirs)
    assert result["status"] == "pass"
    assert set(result["production_merged_score_artifacts"]) == {"rde", "stab"}

    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad_audit, bad_production = _handoff_parity_fixture(
        bad_root, maximum=1.1e-6
    )
    with pytest.raises(ValueError, match="exceeds"):
        handoff._load_parity_audit(bad_audit, bad_production)


def test_handoff_parity_loader_rejects_production_scorer_mismatch(tmp_path):
    audit_dir, production_dirs = _handoff_parity_fixture(tmp_path)
    path = production_dirs["rde"] / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_universe"]["candidate_rows"] = 4_447_847
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="full universe|not bound"):
        handoff._load_parity_audit(audit_dir, production_dirs)


def test_retrospective_f1_threshold_uses_inclusive_score_rule():
    result = handoff._select_retrospective_f1_threshold(
        [0.9, 0.8, 0.8, 0.1], [1, 1, 0, 0]
    )
    assert result["retrospective_threshold"] == pytest.approx(0.8)
    assert result["recommended_pairs"] == 3
    assert result["true_positive_binders"] == 2
    assert result["false_positive_nonbinders"] == 1
    assert result["missed_binders"] == 0
    assert result["precision"] == pytest.approx(2.0 / 3.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)


def test_retrospective_f1_threshold_tie_prefers_precision_then_fewer():
    # Thresholds 4 and 1 both have F1=2/3.  Threshold 4 has higher precision
    # and recommends fewer pairs, so the documented deterministic tie rule
    # must choose it.
    result = handoff._select_retrospective_f1_threshold(
        [4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1]
    )
    assert result["retrospective_threshold"] == pytest.approx(4.0)
    assert result["recommended_pairs"] == 1
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2.0 / 3.0)
