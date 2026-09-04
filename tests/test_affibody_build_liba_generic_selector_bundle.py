import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_liba_generic_selector_bundle as bundle


def _generic_lock(candidate: str, family: str = "single") -> dict:
    return {
        "schema_version": bundle.GENERIC_LOCK_SCHEMA,
        "library": "LibA",
        "retention_labels_read": False,
        "candidate": candidate,
        "candidate_family": family,
        "lock_id": "weak-only-lock",
        "cutoff": {
            "decision_rule": "score >= threshold",
            "selection_objective": "maximum F1 on matched weak-label OOF rows",
            "pad_below_threshold": False,
            "score_threshold": 0.4,
        },
        "selection_provenance": {
            "candidate_metrics": {
                "within_peptide_ap": 0.71,
                "within_peptide_evaluable": 189,
            }
        },
    }


def test_generic_lock_preserves_source_candidate_name(tmp_path):
    candidate = "mean_logit__frozen_mint_layer9"
    path = tmp_path / "selected.lock.json"
    path.write_text(json.dumps(_generic_lock(candidate)), encoding="utf-8")
    metrics = pd.DataFrame(
        {
            "candidate": [candidate],
            "family": ["single"],
            "within_peptide_ap": [0.71],
            "within_peptide_evaluable": [189],
            "threshold": [0.4],
        }
    )
    lock, row = bundle.validate_generic_lock(path, candidate, "single", metrics)
    assert lock["candidate"] == candidate
    assert row["within_peptide_ap"] == pytest.approx(0.71)


def test_generic_lock_rejects_metric_drift(tmp_path):
    candidate = "mean_logit__frozen_mint_layer9"
    path = tmp_path / "selected.lock.json"
    path.write_text(json.dumps(_generic_lock(candidate)), encoding="utf-8")
    metrics = pd.DataFrame(
        {
            "candidate": [candidate],
            "family": ["single"],
            "within_peptide_ap": [0.72],
            "within_peptide_evaluable": [189],
            "threshold": [0.4],
        }
    )
    with pytest.raises(ValueError, match="weak AP differs"):
        bundle.validate_generic_lock(path, candidate, "single", metrics)


def _candidate_scores(path: Path, binding: dict) -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["mint_l9"] * 3,
            "pair_uid": ["p1", "p2", "p3"],
            "peptide_design_code": ["AF", "DL", "EA"],
            "peptide_9mer_sequence": ["SLLAFITQV", "SLLDLITQV", "SLLEAITQV"],
            "affibody_design_code": ["AAAA", "AAAD", "AAAE"],
            "model_score": [0.7, 0.8, 0.9],
            "affibody_identity_seen_in_strict_training": [False, False, True],
        }
    )
    frame.to_parquet(path, index=False)
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "mint_l9",
                "rows": 3,
                "pair_uid_membership_sha256": bundle.membership_sha256(frame["pair_uid"]),
                "deployment_binding": binding,
                "output": {"sha256": bundle.sha256_file(path)},
            }
        ),
        encoding="utf-8",
    )


def test_candidate_score_audit_checks_scope_cutoff_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(bundle, "EXPECTED_DOUBLE_COLD", 2)
    monkeypatch.setattr(bundle, "EXPECTED_PEPTIDE_COLD_ONLY", 1)
    monkeypatch.setattr(bundle, "EXPECTED_PEPTIDES", ("AF", "DL", "EA"))
    monkeypatch.setattr(bundle, "EXPECTED_ZERO_CANDIDATE_TARGETS", ())
    path = tmp_path / "candidate_scores.parquet"
    binding = {key: key[0] * 64 for key in (
        "head_sha256", "config_sha256", "generic_lock_sha256", "weak_oof_sha256"
    )}
    _candidate_scores(path, binding)
    frame, audit = bundle.validate_candidate_score(path, "mint_l9", 0.6, binding)
    assert len(frame) == 3
    assert audit["pairs_above_cutoff_by_peptide"] == {"AF": 1, "DL": 1, "EA": 1}
    assert audit["outcome_columns_read"] == []


def test_candidate_score_audit_rejects_empty_target_after_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(bundle, "EXPECTED_DOUBLE_COLD", 2)
    monkeypatch.setattr(bundle, "EXPECTED_PEPTIDE_COLD_ONLY", 1)
    monkeypatch.setattr(bundle, "EXPECTED_PEPTIDES", ("AF", "DL", "EA"))
    monkeypatch.setattr(bundle, "EXPECTED_ZERO_CANDIDATE_TARGETS", ())
    path = tmp_path / "candidate_scores.parquet"
    binding = {key: key[0] * 64 for key in (
        "head_sha256", "config_sha256", "generic_lock_sha256", "weak_oof_sha256"
    )}
    _candidate_scores(path, binding)
    with pytest.raises(ValueError, match="unexpected zero-candidate targets"):
        bundle.validate_candidate_score(path, "mint_l9", 0.85, binding)


def test_prospective_menu_agreement_is_thresholded_and_outcome_blind(monkeypatch):
    monkeypatch.setattr(bundle, "EXPECTED_PEPTIDES", ("AF", "DL"))
    primary = pd.DataFrame(
        {
            "pair_uid": ["af-a", "af-b", "af-c", "dl-a", "dl-b", "dl-c"],
            "peptide_design_code": ["AF"] * 3 + ["DL"] * 3,
            "affibody_design_code": ["AAAA", "AAAD", "AAAE"] * 2,
            "model_score": [0.9, 0.8, 0.2, 0.9, 0.8, 0.2],
        }
    )
    comparator = primary.copy()
    comparator["model_score"] = [0.8, 0.9, 0.2, 0.9, 0.7, 0.2]
    by_target, recurrence, summary = bundle.prospective_menu_agreement(
        primary, comparator, 0.5, 0.5, maximum=2
    )
    assert by_target["overlap_count"].tolist() == [2, 2]
    assert by_target["same_top1"].tolist() == [False, True]
    assert summary["targets_with_same_top1"] == 1
    assert summary["mean_menu_overlap_count_across_all_nine_targets"] == pytest.approx(2.0)
    assert summary["mean_menu_overlap_count_across_targets_with_both_models"] == pytest.approx(2.0)
    assert summary["outcome_columns_read"] == []
    mint_aaaa = recurrence.loc[
        recurrence["model_id"].eq("mint_l9")
        & recurrence["affibody_design_code"].eq("AAAA")
    ].iloc[0]
    assert mint_aaaa["selected_for_n_targets"] == 2
    assert mint_aaaa["top1_for_n_targets"] == 2
