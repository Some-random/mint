import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import audit_libb_wetlab_candidate_qc as qc


TARGETS = tuple(sorted(qc.EXPECTED_TARGETS))
CODES = (
    "AAAAA", "AAAAA", "AAAAD", "AAAAE", "AAAAF", "AAAAI",
    "AAAAK", "AAAAL", "AAAAM", "AAAAN", "AAAAS", "AAAAT",
)


def _inputs():
    all4_rows = []
    handoff_rows = []
    for index, (peptide, code) in enumerate(zip(TARGETS, CODES)):
        sequence = "A" * 58
        if index == 2:
            sequence = "NAT" + "A" * 55
        if index == 3:
            sequence = "C" + "A" * 57
        common = {
            "pair_uid": "pair-%s" % peptide,
            "peptide_design_code": peptide,
            "affibody_design_code": code,
            "wetlab_rank": 1,
            "ensemble_score": 0.9 - index / 100.0,
        }
        all4_rows.append({**common, "chain2_affibody_sequence": sequence})
        handoff_rows.append({
            **common,
            "peptide_9mer_sequence": "SLL%sITQV" % peptide,
            "affibody_full_sequence": sequence,
            "candidate_list_membership_count": 3 if index == 0 else 1,
            "component_top10_count": 4 if index == 0 else 1,
            "selected_by_mint": index == 0,
            "selected_by_mint_stab": index == 0,
            "stab_seed_probability_sd": 0.01 + index / 1000.0,
            "rde_seed_probability_sd": 0.02 + index / 1000.0,
            "observed_in_any_raw_round": index == 2,
            "observed_in_r009_or_r010": index == 2,
            "high_confidence_weak_negative": index == 3,
            "affibody_identity_seen_in_strict_training": code in {"AAAAA", "AAAAD"},
        })
    return pd.DataFrame(all4_rows), pd.DataFrame(handoff_rows)


def _weak_labels():
    return pd.DataFrame(
        {
            "library": ["LibB", "LibB", "LibB", "LibB", "LibA"],
            "pep": ["ZZ", "ZY", "ZX", "ZW", "ZZ"],
            "aff": ["AAAAA", "AAAAD", "AAAAD", "DDDDD", "AAAAA"],
            "weak_label": [1, 0, 1, 0, 1],
            "within_declared_library_alphabet": [1, 1, 1, 1, 1],
            "negative_r001_count_ge_3": [0, 1, 0, 0, 0],
            "strict_retention_identity_cold_eligible": [1, 1, 1, 1, 1],
            # These two metadata columns may exist in the real table but are
            # deliberately not read by the QC utility.
            "shares_retention_peptide": [0, 0, 0, 0, 0],
            "shares_retention_affibody": [0, 0, 0, 0, 0],
        }
    )


def test_alignment_annotation_and_summaries_preserve_ranks():
    all4, handoff = _inputs()
    aligned = qc.align_candidate_inputs(all4, handoff)
    training_codes = {"AAAAA", "AAAAD"}
    training_pairs = {("ZZ", "AAAAA"), ("ZY", "AAAAD"), ("ZX", "AAAAD")}
    annotated = qc.annotate_candidates(
        aligned, training_codes, training_pairs, {"AAAAA": 1, "AAAAD": 2}
    )
    reuse = qc.build_affibody_reuse_table(annotated)
    per_target = qc.build_per_target_table(annotated)
    summary = qc.build_summary(annotated, reuse, 3, 2)

    assert annotated["wetlab_rank"].tolist() == handoff["wetlab_rank"].tolist()
    assert annotated.loc[0, "qc_selected_for_n_peptide_targets"] == 2
    assert annotated.loc[1, "qc_selected_for_n_peptide_targets"] == 2
    assert annotated.loc[0, "qc_exact_affibody_code_seen_in_strict_training"]
    assert annotated.loc[2, "qc_nearest_strict_training_affibody_hamming_distance"] == 0
    assert annotated.loc[2, "qc_full_affibody_contains_N_X_S_or_T_motif"]
    assert annotated.loc[3, "qc_full_affibody_contains_cysteine"]
    assert len(per_target) == 12
    assert summary["unique_affibody_codes"] == 11
    assert summary["affibody_codes_selected_for_multiple_targets"] == 1
    assert summary["weak_selection_flags"]["high_confidence_weak_negative"] == 1


def test_alignment_rejects_changed_rank():
    all4, handoff = _inputs()
    handoff.loc[0, "wetlab_rank"] = 2
    with pytest.raises(ValueError, match="ranks are not exactly"):
        qc.align_candidate_inputs(all4, handoff)


def test_alignment_rejects_fractional_rank_instead_of_truncating():
    all4, handoff = _inputs()
    all4.loc[0, "wetlab_rank"] = 1.5
    with pytest.raises(ValueError, match="non-integral rank"):
        qc.align_candidate_inputs(all4, handoff)


def test_outcome_guard_rejects_any_retention_column(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"pair_uid": ["x"], "measured_retention_value": [88.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="forbidden outcome"):
        qc._read_outcome_free_csv(path, "bad input")


def test_weak_membership_reads_exact_training_rule(tmp_path, monkeypatch):
    path = tmp_path / "weak_labels.csv"
    _weak_labels().to_csv(path, index=False)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_ROWS", 3)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_AFFIBODIES", 2)
    strict, codes, pairs, row_counts = qc.load_strict_training_membership(path)
    assert len(strict) == 3
    assert codes == {"AAAAA", "AAAAD"}
    assert pairs == {("ZZ", "AAAAA"), ("ZY", "AAAAD"), ("ZX", "AAAAD")}
    assert row_counts == {"AAAAA": 1, "AAAAD": 2}


def test_weak_membership_rejects_embedded_outcome_column(tmp_path, monkeypatch):
    path = tmp_path / "weak_labels.csv"
    frame = _weak_labels()
    frame["target_retention"] = 90.0
    frame.to_csv(path, index=False)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_ROWS", 3)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_AFFIBODIES", 2)
    with pytest.raises(ValueError, match="forbidden outcomes"):
        qc.load_strict_training_membership(path)


def test_cli_writes_machine_readable_outputs_without_reselection(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(qc, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_ROWS", 3)
    monkeypatch.setattr(qc, "EXPECTED_STRICT_TRAINING_AFFIBODIES", 2)
    private = tmp_path / "private_data"
    private.mkdir()
    all4, handoff = _inputs()
    all4_path = private / "all4.csv"
    handoff_path = private / "menu.csv"
    weak_path = private / "weak.csv"
    all4.to_csv(all4_path, index=False)
    handoff.to_csv(handoff_path, index=False)
    _weak_labels().to_csv(weak_path, index=False)
    output = private / "qc"
    qc.run(SimpleNamespace(
        all4_shortlist=all4_path,
        handoff_menu=handoff_path,
        weak_labels=weak_path,
        output_dir=output,
    ))

    assert (output / "candidate_qc.csv").is_file()
    assert (output / "affibody_reuse_qc.csv").is_file()
    assert (output / "per_target_qc.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["invariants"] == {
        "candidate_set_changed": False,
        "models_trained": False,
        "retention_outcomes_read": False,
        "scores_changed": False,
        "within_peptide_ranks_changed": False,
    }
    written = pd.read_csv(output / "candidate_qc.csv")
    assert written[["pair_uid", "wetlab_rank"]].to_dict("records") == handoff[
        ["pair_uid", "wetlab_rank"]
    ].to_dict("records")
