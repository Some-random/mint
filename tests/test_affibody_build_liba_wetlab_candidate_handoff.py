import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_liba_wetlab_candidate_handoff as handoff


TARGETS = {
    "AA": "SLLAAITQV",
    "AB": "SLLAFITQV",
    "AC": "SLLAHITQV",
    "AD": "SLLDPITQV",
    "AE": "SLLEAITQV",
    "AF": "SLLLLITQV",
    "AG": "SLLMWITQV",
    "AH": "SLLNFITQV",
    "AI": "SLLVVITQV",
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _affibody_sequence(code):
    sequence = list(
        "VDNKFNKEQQNAFYEILHLPNLNEEQRNAFIQSLKDDPSQSANLLAEAKKLNDAQAPK"
    )
    assert len(sequence) == 58
    for amino_acid, position in zip(code, handoff.LIBA_DISPLAYED_POSITIONS):
        sequence[position - 1] = amino_acid
    return "".join(sequence)


@pytest.mark.parametrize("column", ["binder", "label", "target", "outcome"])
def test_bare_outcome_columns_are_forbidden(column):
    assert handoff.forbidden_score_columns([column]) == [column]


def test_evidence_control_ties_use_affibody_code_before_pair_uid():
    # This reproduces the production EA tie: LANA and NALN both have pooled
    # count 23, while their opaque pair IDs sort in the opposite order.
    frame = pd.DataFrame({
        "pair_uid": ["z-pair", "a-pair"],
        "peptide_design_code": ["EA", "EA"],
        "affibody_design_code": ["LANA", "NALN"],
        "pooled_r009_r010_count": [23, 23],
    })
    selected = handoff.evidence_review_menu(frame, maximum=1)
    assert selected.iloc[0]["affibody_design_code"] == "LANA"
    assert selected.iloc[0]["pair_uid"] == "z-pair"

    evidence = frame.copy()
    evidence["peptide_9mer_sequence"] = "SLLEAITQV"
    evidence["provider_displayed_58aa_affibody_sequence"] = evidence[
        "affibody_design_code"
    ].map(_affibody_sequence)
    evidence["model_input_affibody_sequence"] = evidence[
        "provider_displayed_58aa_affibody_sequence"
    ]
    discoveries = pd.DataFrame({
        "pair_uid": [f"discovery-{index}" for index in range(8)],
        "peptide_design_code": ["EA"] * 8,
        "peptide_9mer_sequence": ["SLLEAITQV"] * 8,
        "affibody_design_code": [
            "AAAA", "DDDD", "EEEE", "FFFF", "HHHH", "IIII", "KKKK", "PPPP"
        ],
        "model_score": [0.9 - index / 100 for index in range(8)],
        "affibody_identity_seen_in_strict_training": [False] * 8,
        "high_confidence_weak_negative": [False] * 8,
    })
    discoveries["provider_displayed_58aa_affibody_sequence"] = discoveries[
        "affibody_design_code"
    ].map(_affibody_sequence)
    discoveries["model_input_affibody_sequence"] = discoveries[
        "provider_displayed_58aa_affibody_sequence"
    ]
    batch, _ = handoff.build_tiered_first_batch(
        discoveries,
        {"score_threshold": 0.5},
        evidence,
        {"EA": "SLLEAITQV"},
        "mint_l9",
    )
    assert batch.loc[batch["tier_role"].eq("control")].iloc[0][
        "affibody_design_code"
    ] == "LANA"


def test_evidence_menu_must_match_benchmark_membership():
    menu = pd.DataFrame({"pair_uid": ["pair-a", "pair-b"]})
    audit = {
        "retrospective_metrics": {"extrapolative_menu_rows": 2},
        "source_manifest": {
            "memberships": {
                "fixed_unmeasured_menu_sha256": handoff.membership_sha256(
                    menu["pair_uid"]
                )
            }
        },
    }
    handoff.validate_evidence_menu_against_benchmark(menu, audit)
    audit["source_manifest"]["memberships"][
        "fixed_unmeasured_menu_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="membership differs"):
        handoff.validate_evidence_menu_against_benchmark(menu, audit)


def test_esmfold_weak_selection_loader_requires_failed_retention_blind_gate(tmp_path):
    path = tmp_path / "esmfold_selection.json"
    payload = {
        "retention_labels_read": False,
        "selection_data": "selection-derived weak labels only",
        "selected_feature_family": "single_inputs_only",
        "best_structure_derived_family_by_gate_metric": "full",
        "passing_structure_families": [],
        "mint_layer9_reference": {"within_peptide_ap": 0.71},
        "models": {
            "single_inputs_only": {
                "aggregate_mean_logit_metrics": {"within_peptide_ap": 0.66}
            },
            "full": {"aggregate_mean_logit_metrics": {"within_peptide_ap": 0.60}},
        },
    }
    path.write_text(json.dumps(payload))
    loaded = handoff.load_esmfold_weak_selection(path)
    assert loaded["passing_structure_families"] == []
    assert loaded["best_folding_derived_within_peptide_ap"] == pytest.approx(0.60)
    payload["passing_structure_families"] = ["full"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="failed structure gate"):
        handoff.load_esmfold_weak_selection(path)


def _score_rows(model_id):
    rows = []
    amino_acids = "ADEFIK"
    for target, peptide in TARGETS.items():
        for index, amino_acid in enumerate(amino_acids, start=1):
            code = f"AAA{amino_acid}"
            sequence = _affibody_sequence(code)
            rows.append({
                "model_id": model_id,
                "pair_uid": f"{target}|{code}",
                "peptide_design_code": target,
                "peptide_9mer_sequence": peptide,
                "affibody_design_code": code,
                "provider_displayed_58aa_affibody_sequence": sequence,
                "model_input_affibody_sequence": sequence,
                "model_input_smart_hla_linker_peptide_sequence": f"ASSAYSIDE{peptide}",
            "model_score": (1.0 - index * 0.1) + (0.01 if model_id == "mint_l9" else 0),
                "observed_in_any_raw_round": index % 2 == 0,
                "observed_in_r009_or_r010": False,
                "affibody_identity_seen_in_strict_training": index == 1,
                "high_confidence_weak_negative": False,
            })
    return pd.DataFrame(rows)


def _build_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "REPO_ROOT", tmp_path)
    private = tmp_path / "private_data"
    private.mkdir()
    locks = {
        "schema_version": "liba-weak-oof-deployment-locks-v1",
        "library": "LibA",
        "retention_labels_read": False,
        "primary_model_id": "mint_l9",
        "models": {
            "additive": {
                "display_name": "Designed-position additive model",
                "scientifically_eligible": True,
                "lock_id": "liba-additive-weak-oof-v1",
                "score_threshold": 0.59,
                "threshold_uses_retention": False,
                "threshold_provenance": "F1 selection on weak OOF labels only",
                "higher_is_better": True,
                "max_candidates_per_peptide": 3,
                "minimum_code_hamming_distance": 0,
                "pad_below_threshold": False,
                "weak_oof_within_peptide_average_precision": 0.65,
                "weak_oof_sha256": "1" * 64,
                "head_sha256": "2" * 64,
                "config_sha256": "3" * 64,
                "weak_metric_provenance": {"generic_lock_sha256": "7" * 64},
            },
            "mint_l9": {
                "display_name": "Frozen MINT layer 9",
                "scientifically_eligible": True,
                "lock_id": "liba-mint-l9-weak-oof-v1",
                "score_threshold": 0.49,
                "threshold_uses_retention": False,
                "threshold_provenance": "F1 selection on weak OOF labels only",
                "higher_is_better": True,
                "max_candidates_per_peptide": 4,
                "minimum_code_hamming_distance": 0,
                "pad_below_threshold": False,
                "weak_oof_within_peptide_average_precision": 0.67,
                "weak_oof_sha256": "4" * 64,
                "head_sha256": "5" * 64,
                "config_sha256": "6" * 64,
                "weak_metric_provenance": {"generic_lock_sha256": "8" * 64},
            },
            "rde": {
                "display_name": "RDE-PPI",
                "scientifically_eligible": False,
            },
        },
    }
    project_spec = {
        "schema_version": "liba-wetlab-project-spec-v1",
        "library": "LibA",
        "target_sequences": TARGETS,
        "expected_selection_missed_candidate_pairs": 54,
        "expected_double_cold_candidate_pairs": 45,
        "expected_peptide_cold_only_candidate_pairs": 9,
        "training_data": {
            "description": "Both models use the provider-defined LibA weak-label training set.",
            "weak_label_definition": "Pooled R009+R010 top 2% positives and locked negatives.",
            "split_rule": "No evaluation peptide or Affibody sequence occurs in training.",
            "training_rows_by_model": {"additive": 22542, "mint_l9": 22542},
        },
        "evidence_positive_control": {
            "expected_pairs": 18,
            "max_controls_per_peptide": 1,
        },
        "unavailable_structure_families": [
            {"family": "StaB-ddG", "reason": "No validated LibA candidate score artifact."},
            {"family": "RDE-PPI", "reason": "No validated LibA candidate score artifact."},
        ],
    }
    lock_path = private / "weak_locks.json"
    lock_path.write_text(json.dumps(locks))
    score_paths = {}
    project_path = private / "project_spec.json"
    project_path.write_text(json.dumps(project_spec))
    for model_id in ("additive", "mint_l9"):
        model_dir = private / f"{model_id}_scores"
        model_dir.mkdir()
        path = model_dir / "candidate_scores.parquet"
        score_rows = _score_rows(model_id)
        score_rows.to_parquet(path, index=False)
        model_lock = locks["models"][model_id]
        membership_sha = hashlib.sha256(
            "\n".join(sorted(score_rows["pair_uid"].astype(str))).encode("ascii")
        ).hexdigest()
        (model_dir / "manifest.json").write_text(json.dumps({
            "model_id": model_id,
            "rows": 54,
            "pair_uid_membership_sha256": membership_sha,
            "outputs": {"candidate_scores": {
                "path": path.name, "rows": 54, "sha256": _sha(path),
            }},
            "deployment_binding": {
                "head_sha256": model_lock["head_sha256"],
                "config_sha256": model_lock["config_sha256"],
                "weak_oof_sha256": model_lock["weak_oof_sha256"],
                "generic_lock_sha256": model_lock["weak_metric_provenance"]["generic_lock_sha256"],
            },
        }))
        score_paths[model_id] = path
    comparison = pd.DataFrame([
        {
            "model_id": "additive",
            "display_name": "Designed-position additive model",
            "evaluation_pairs": 108,
            "within_peptide_auroc": 0.61,
            "evaluable_peptides_for_within_peptide_auroc": 8,
            "within_peptide_average_precision": 0.72,
            "evaluable_peptides_for_within_peptide_ap": 8,
            "within_peptide_spearman": -0.336,
            "evaluable_peptides_for_within_peptide_spearman": 9,
            "weak_oof_within_peptide_average_precision": 0.65,
            "weak_oof_evaluable_peptides_for_within_peptide_ap": 9,
            "retention_optimized_threshold": 0.987,
            "pairs_above_retention_optimized_threshold": 71,
            "binders_above_retention_optimized_threshold": 38,
            "retrospective_precision": 0.535,
            "retrospective_recall": 1.0,
            "retrospective_f1": 0.697,
        },
        {
            "model_id": "mint_l9",
            "display_name": "Frozen MINT layer 9",
            "evaluation_pairs": 108,
            "within_peptide_auroc": 0.63,
            "evaluable_peptides_for_within_peptide_auroc": 8,
            "within_peptide_average_precision": 0.74,
            "evaluable_peptides_for_within_peptide_ap": 8,
            "within_peptide_spearman": -0.304,
            "evaluable_peptides_for_within_peptide_spearman": 9,
            "weak_oof_within_peptide_average_precision": 0.67,
            "weak_oof_evaluable_peptides_for_within_peptide_ap": 9,
            "retention_optimized_threshold": 0.979,
            "pairs_above_retention_optimized_threshold": 70,
            "binders_above_retention_optimized_threshold": 37,
            "retrospective_precision": 0.529,
            "retrospective_recall": 0.974,
            "retrospective_f1": 0.685,
        },
    ])
    comparison_path = private / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    evidence_rows = []
    for target, peptide in TARGETS.items():
        for rank, code in enumerate(("DDDA", "DDDE"), start=1):
            sequence = _affibody_sequence(code)
            evidence_rows.append({
                "pair_uid": f"evidence-{target}-{rank}",
                "peptide_design_code": target,
                "peptide_9mer_sequence": peptide,
                "affibody_design_code": code,
                "provider_displayed_58aa_affibody_sequence": sequence,
                "model_input_affibody_sequence": sequence,
                "model_input_smart_hla_linker_peptide_sequence": f"ASSAYSIDE{peptide}",
                "r009_count": 100 - rank,
                "r010_count": 50,
                "pooled_r009_r010_count": 150 - rank,
                "directly_measured": False,
                "pooled_r009_r010_top2pct": True,
            })
    evidence_path = private / "evidence_controls.parquet"
    pd.DataFrame(evidence_rows).to_parquet(evidence_path, index=False)
    evidence_audit = {
        "schema_version": "liba-evidence-control-retrospective-audit-v1",
        "retention_panel_read": True,
        "retrospective_metrics": {
            "evaluation_pairs": 108,
            "global_auroc": 0.847,
            "global_average_precision": 0.692,
            "within_peptide_spearman": 0.204,
            "within_peptide_average_precision": 0.766,
            "evaluable_peptides_for_within_peptide_ap": 5,
            "pooled_top2pct_count_cutoff": 13,
            "pooled_top2pct_true_positive_count": 32,
            "pooled_top2pct_predicted_positive_count": 55,
            "pooled_top2pct_precision": 32 / 55,
            "pooled_top2pct_recall": 32 / 55,
            "extrapolative_expected_binders": 44.22,
            "extrapolative_menu_rows": 76,
        },
    }
    evidence_audit_path = private / "evidence_audit.json"
    evidence_audit_path.write_text(json.dumps(evidence_audit))
    args = SimpleNamespace(
        weak_oof_locks=lock_path,
        project_spec=project_path,
        candidate_scores=[
            f"additive={score_paths['additive']}", f"mint_l9={score_paths['mint_l9']}"
        ],
        evidence_positive_controls=evidence_path,
        evidence_control_comparison=evidence_audit_path,
        model_comparison=comparison_path,
        output_dir=private / "handoff",
    )
    return args, score_paths


def test_liba_handoff_is_input_driven_and_keeps_threshold_meanings_separate(
    tmp_path, monkeypatch
):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    handoff.run(args)
    output = args.output_dir
    order = pd.read_csv(output / "ORDER_THIS_BATCH.csv")
    assert len(order) == 54
    assert order["sample_id"].is_unique
    assert order.iloc[0]["sample_id"] == "LIBA-AA-01"
    assert order.groupby("peptide_design_code").size().eq(6).all()
    assert order.groupby(["peptide_design_code", "tier_role"]).size().unstack().eq(
        {"control": 2, "discovery": 4}
    ).all().all()
    assert order.loc[
        order.tier_role.eq("discovery"), "high_confidence_weak_negative"
    ].eq(False).all()
    # The retrospective 0.979 threshold was not applied: weak threshold 0.49
    # produces four prospective candidates per target.
    assert order["model_score"].min() < 0.979

    mint_menu = pd.read_csv(
        output / "model_mint_l9" / "mint_l9_candidate_menu_all_9_targets.csv"
    )
    additive_menu = pd.read_csv(
        output / "model_additive" / "additive_candidate_menu_all_9_targets.csv"
    )
    assert len(mint_menu) == 36
    assert len(additive_menu) == 27
    mint_qc = pd.read_csv(
        output / "model_mint_l9" / "mint_l9_candidate_qc_summary_all_9_targets.csv"
    )
    assert len(mint_qc) == 9
    assert "prior_high_confidence_weak_negative" in mint_qc.columns
    assert set(mint_menu["affibody_code_position_mapping"]) == {handoff.LIBA_MAPPING_TEXT}
    compact = pd.read_csv(
        output / "model_mint_l9" / "mint_l9_compact_codes_by_target.csv"
    )
    assert len(compact) == 9
    assert "rank_04_affibody_code" in compact.columns
    roster = pd.read_csv(
        output / "model_mint_l9" / "mint_l9_unique_affibody_sequence_review_roster.csv"
    )
    assert len(roster) == 4
    double_cold = pd.read_csv(
        output / "model_mint_l9" / "mint_l9_strict_double_cold_menu_all_9_targets.csv"
    )
    assert len(double_cold) == 36
    assert double_cold["affibody_identity_seen_in_strict_training"].eq(False).all()
    assert set(double_cold["generalization_scope"]) == {"strict_double_cold"}
    evidence_full = pd.read_csv(
        output / "evidence_positive_control_tier_all_unmeasured_pairs.csv"
    )
    evidence_menu = pd.read_csv(
        output / "evidence_positive_control_review_menu_by_target.csv"
    )
    assert len(evidence_full) == 18
    assert len(evidence_menu) == 9
    assert evidence_menu.groupby("peptide_design_code").size().eq(1).all()
    assert set(evidence_menu["tier"]) == {
        "separate_evidence_positive_control_not_model_candidate"
    }
    assert set(evidence_full["pair_uid"]).isdisjoint(set(mint_menu["pair_uid"]))

    results = pd.read_csv(output / "RESULT_ENTRY.csv")
    outcome_fields = [
        "retention_at_30min_percent", "binder_ge_75", "technical_failure_status",
        "replicate_id", "assay_date", "experimental_batch_id", "notes",
    ]
    assert results[outcome_fields].isna().all().all()
    assert results[["sample_id", "pair_uid"]].equals(order[["sample_id", "pair_uid"]])
    model_only = pd.read_csv(output / "PRIMARY_MODEL_ONLY_BATCH.csv")
    assert len(model_only) == 36
    assert model_only["sample_id"].str.startswith("LIBA-MODELONLY-").all()
    assert set(model_only["sample_id"]).isdisjoint(set(order["sample_id"]))
    combined_ids = pd.concat(
        [order[["sample_id", "pair_uid"]], model_only[["sample_id", "pair_uid"]]],
        ignore_index=True,
    )
    assert combined_ids.groupby("sample_id")["pair_uid"].nunique().max() == 1
    tier_summary = pd.read_csv(output / "FIRST_BATCH_TIER_SUMMARY.csv")
    assert tier_summary["evidence_controls_selected"].eq(2).all()
    assert tier_summary["strict_double_cold_discoveries_selected"].eq(4).all()

    thresholds = pd.read_csv(output / "threshold_context.csv")
    mint_threshold = thresholds.loc[thresholds.model_id.eq("mint_l9")].iloc[0]
    assert mint_threshold["weak_label_deployment_threshold"] == pytest.approx(0.49)
    assert mint_threshold["retention_optimized_descriptive_threshold"] == pytest.approx(0.979)
    assert "descriptive" in mint_threshold["threshold_use"]

    report = (output / "liba_candidate_selection_report.md").read_text()
    assert "0.70" in report  # 0.695 rounded to two decimals for display.
    assert "-0.30" in report
    assert "weak-label deployment threshold" in report
    assert "retention-optimized threshold" in report
    assert "**StaB-ddG: unavailable.**" in report
    assert "**RDE-PPI: unavailable.**" in report
    assert "54\nselection-missed pairs" in report
    assert "18 unmeasured current-target pairs" in report
    assert "0.58" in report
    assert "44.22" not in report
    assert "binder-yield estimate" in report
    assert "the new assay is the test" in report
    assert "have not been confirmed as binders" in report
    assert "positive control" not in report.lower()
    assert "not evidence of generalization" in report
    assert "more defensible than learned-model" in report
    assert "not a Cartesian cross" in report
    assert "not a final vendor or cloning construct" in report
    assert "actual mixed batch" in report
    assert "Sequencing-evidence comparators: code (pooled count)" in report
    assert "Secondary model-only comparison" in report
    assert "allocation is fixed for this prospective handoff" in report
    assert "allocation is prespecified" not in report
    assert "determine threshold eligibility and rank discoveries" in report
    assert "LoRA showed no" in report
    assert "PNU and later-round labels did not improve" in report
    assert "meta-gradient and trajectory-rater models" in report
    assert "ineligible for retention-blind" in report
    assert "pragmatic decision to mix evidence comparators" in report
    assert "at least\nthree R001 reads" in report
    assert "rounded and need not be exact ties" in report
    assert "unrounded locked comparison selected MINT layer 9" in report
    assert "distinct displayed\nAffibody sequences" in report
    assert "layer-9 residue vectors are averaged within each chain" in report
    assert "MINT head's logit" in report
    assert "330,880" not in report  # fixture counts, never hard-coded production counts
    assert "strict-double-cold menu" in report

    workbook = openpyxl.load_workbook(output / "liba_wetlab_candidate_handoff.xlsx")
    assert workbook.sheetnames[0] == "ORDER_THIS_BATCH"
    assert {"RESULT_ENTRY", "MODEL_RECOMMENDATIONS", "RETROSPECTIVE_RESULTS",
            "THRESHOLD_CONTEXT", "mint_l9_MENU", "EVIDENCE_CONTROL_MENU",
            "EVIDENCE_CONTROL_CODES", "EVIDENCE_CONTROL_ROSTER"}.issubset(
                workbook.sheetnames
            )
    readme_values = [cell.value for row in workbook["README"].iter_rows() for cell in row]
    assert any("evidence comparators" in str(value) for value in readme_values)
    assert any("compatibility names" in str(value) for value in readme_values)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["producer"]["script"]["path"] == str(Path(handoff.__file__).resolve())
    assert manifest["producer"]["script"]["sha256"] == handoff.sha256_file(
        Path(handoff.__file__).resolve()
    )
    assert manifest["producer"]["python_executable"]
    assert manifest["producer"]["python_version"]
    assert manifest["primary_model_id"] == "mint_l9"
    assert manifest["outcome_access"]["retention_used_for_model_training"] is False
    assert manifest["outcome_access"]["retention_used_for_deployment_cutoff"] is False
    assert manifest["outcome_access"][
        "retention_used_for_exact_within_tier_pair_identity_selection"
    ] is False
    assert manifest["outcome_access"][
        "retrospective_performance_informed_tier_allocation"
    ] is True
    assert manifest["primary_pair_assays"] == 54
    assert manifest["primary_model_only_pair_assays"] == 36
    assert manifest["first_batch"]["evidence_control_assays"] == 18
    assert manifest["first_batch"]["strict_double_cold_model_discovery_assays"] == 36
    assert manifest["selection_missed_candidate_pairs_per_model"] == 54
    assert manifest["strict_double_cold_candidate_pairs_per_model"] == 45
    assert manifest["peptide_cold_only_candidate_pairs_per_model"] == 9
    assert manifest["evidence_positive_control_pairs"] == 18
    assert manifest["validation"]["result_entry_fields_blank"] is True
    assert manifest["inputs"]["weak_oof_locks"]["sha256"] == _sha(args.weak_oof_locks)
    weak_lock = json.loads(args.weak_oof_locks.read_text())
    assert "retrospective_metrics" not in json.dumps(weak_lock)
    assert manifest["inputs"]["evidence_control_comparison"][
        "loaded_after_candidate_menus_were_fixed"
    ] is True
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_liba_handoff_rejects_retention_column_in_candidate_scores(tmp_path, monkeypatch):
    args, score_paths = _build_inputs(tmp_path, monkeypatch)
    scores = pd.read_parquet(score_paths["mint_l9"])
    scores["retention_percent"] = 99.0
    scores.to_parquet(score_paths["mint_l9"], index=False)
    manifest_path = score_paths["mint_l9"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["candidate_scores"]["sha256"] = _sha(score_paths["mint_l9"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leak outcomes"):
        handoff.run(args)
    assert not args.output_dir.exists()


def test_liba_handoff_rejects_stale_candidate_score_manifest(tmp_path, monkeypatch):
    args, score_paths = _build_inputs(tmp_path, monkeypatch)
    manifest_path = score_paths["mint_l9"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["candidate_scores"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash differs"):
        handoff.run(args)


def test_liba_handoff_rejects_wrong_score_deployment_binding(tmp_path, monkeypatch):
    args, score_paths = _build_inputs(tmp_path, monkeypatch)
    manifest_path = score_paths["mint_l9"].parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["deployment_binding"]["generic_lock_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generic_lock_sha256 differs"):
        handoff.run(args)


def test_liba_handoff_requires_all_and_only_eligible_score_inputs(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    args.candidate_scores = args.candidate_scores[:1]
    with pytest.raises(ValueError, match="exactly match scientifically eligible"):
        handoff.run(args)
    assert not args.output_dir.exists()


def test_zero_control_target_is_not_padded_or_fabricated(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    evidence = pd.read_parquet(args.evidence_positive_controls)
    evidence = evidence.loc[evidence["peptide_design_code"].ne("AD")].copy()
    evidence.to_parquet(args.evidence_positive_controls, index=False)
    project = json.loads(args.project_spec.read_text())
    project["evidence_positive_control"]["expected_pairs"] = len(evidence)
    args.project_spec.write_text(json.dumps(project))

    handoff.run(args)
    batch = pd.read_csv(args.output_dir / "ORDER_THIS_BATCH.csv")
    omitted = batch.loc[batch["peptide_design_code"].eq("AD")]
    assert omitted["tier_role"].eq("discovery").all()
    assert not omitted["pair_uid"].astype(str).str.startswith("evidence-").any()
    summary = pd.read_csv(args.output_dir / "FIRST_BATCH_TIER_SUMMARY.csv")
    row = summary.loc[summary["peptide_design_code"].eq("AD")].iloc[0]
    assert row["evidence_controls_available"] == 0
    assert row["evidence_controls_selected"] == 0
    compact = pd.read_csv(
        args.output_dir / "evidence_positive_control_compact_codes_by_target.csv"
    )
    compact_row = compact.loc[compact["peptide_design_code"].eq("AD")].iloc[0]
    assert compact_row["selected_evidence_positive_controls"] == 0
    assert pd.isna(compact_row["rank_01_affibody_code"])


def test_target_below_all_locked_thresholds_is_audited_but_not_ordered(
    tmp_path, monkeypatch
):
    args, score_paths = _build_inputs(tmp_path, monkeypatch)
    evidence = pd.read_parquet(args.evidence_positive_controls)
    evidence = evidence.loc[evidence["peptide_design_code"].ne("AD")].copy()
    evidence.to_parquet(args.evidence_positive_controls, index=False)
    project = json.loads(args.project_spec.read_text())
    project["evidence_positive_control"]["expected_pairs"] = len(evidence)
    args.project_spec.write_text(json.dumps(project))
    for path in score_paths.values():
        scores = pd.read_parquet(path)
        scores.loc[scores["peptide_design_code"].eq("AD"), "model_score"] = 0.01
        scores.to_parquet(path, index=False)
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["candidate_scores"]["sha256"] = _sha(path)
        manifest_path.write_text(json.dumps(manifest))

    handoff.run(args)
    order = pd.read_csv(args.output_dir / "ORDER_THIS_BATCH.csv")
    assert "AD" not in set(order["peptide_design_code"])
    audit = pd.read_csv(
        args.output_dir / "NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD.csv"
    )
    target = audit.loc[audit["peptide_design_code"].eq("AD")]
    assert set(target["model_id"]) == {"additive", "mint_l9"}
    assert target["status"].eq("NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD").all()
    assert target["orderable"].eq(False).all()
    assert target["maximum_model_score"].eq(0.01).all()
    summary = pd.read_csv(args.output_dir / "FIRST_BATCH_TIER_SUMMARY.csv")
    row = summary.loc[summary["peptide_design_code"].eq("AD")].iloc[0]
    assert row["batch_status"] == "NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD"
    assert row["total_pair_assays"] == 0


def test_liba_lock_rejects_retention_derived_deployment_threshold(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    lock = json.loads(args.weak_oof_locks.read_text())
    lock["models"]["mint_l9"]["threshold_uses_retention"] = True
    args.weak_oof_locks.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="not weak-label-only"):
        handoff.run(args)
    assert not args.output_dir.exists()


def test_liba_lock_rejects_mint_layer5_as_scientifically_eligible(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    lock = json.loads(args.weak_oof_locks.read_text())
    model = lock["models"].pop("mint_l9")
    model["display_name"] = "Frozen MINT layer 5"
    lock["models"]["mint_l5"] = model
    lock["primary_model_id"] = "mint_l5"
    args.weak_oof_locks.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="layer 9 or 33"):
        handoff.run(args)
    assert not args.output_dir.exists()


def test_liba_lock_accepts_layer33_as_control_but_not_primary(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    lock = json.loads(args.weak_oof_locks.read_text())
    layer33 = dict(lock["models"]["mint_l9"])
    layer33["display_name"] = "Frozen MINT layer 33 control"
    layer33["lock_id"] = "liba-mint-l33-control-weak-oof-v1"
    lock["models"]["mint_l33"] = layer33
    args.weak_oof_locks.write_text(json.dumps(lock))
    loaded = handoff.load_lock_manifest(args.weak_oof_locks)
    assert loaded["primary_model_id"] == "mint_l9"
    lock["primary_model_id"] = "mint_l33"
    args.weak_oof_locks.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="primary model must be layer 9"):
        handoff.load_lock_manifest(args.weak_oof_locks)


def test_native_provider_evidence_artifact_is_standardized_and_hash_checked(tmp_path):
    rows = []
    for target, peptide in TARGETS.items():
        sequence = _affibody_sequence("DDDA")
        rows.append({
            "candidate_tier": "existing_target_pooled_positive_unmeasured_liba_design",
            "selected_top10": True,
            "pair_uid": f"native-{target}",
            "peptide_design_code": target,
            "peptide_full_sequence": peptide,
            "affibody_design_code": "DDDA",
            "r009_count": 9,
            "r010_count": 8,
            "pooled_r009_r010_count": 17,
            "chain1_smart_hla_linker_peptide_sequence": f"ASSAYSIDE{peptide}",
            "chain2_affibody_sequence": sequence,
        })
    path = tmp_path / "pooled_positive_unmeasured_candidates.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "outputs": {path.name: {"sha256": _sha(path), "rows": len(rows)}}
    }))
    frame = handoff.load_evidence_positive_controls(path, TARGETS, len(rows))
    assert set(frame["peptide_9mer_sequence"]) == set(TARGETS.values())
    assert frame["directly_measured"].eq(False).all()
    assert frame["pooled_r009_r010_top2pct"].eq(True).all()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["outputs"][path.name]["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash does not match"):
        handoff.load_evidence_positive_controls(path, TARGETS, len(rows))


def test_provider_membership_benchmark_directory_adapter(tmp_path):
    metrics = pd.DataFrame([{
        "measured_panel_rows": 108,
        "pooled_positive_membership_cutoff": 13,
        "true_positives": 32,
        "measured_pooled_positive_members": 55,
        "precision": 32 / 55,
        "recall": 32 / 38,
        "fixed_unmeasured_menu_rows": 76,
        "extrapolated_expected_binders": 44.2181818182,
    }])
    path = tmp_path / "measured_membership_metrics.csv"
    metrics.to_csv(path, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": "liba-provider-pooled-membership-retrospective-benchmark-v1",
        "outputs": {path.name: {"sha256": _sha(path), "rows": 1}},
    }))
    audit = handoff.load_evidence_retrospective_audit(tmp_path)
    assert audit["retrospective_metrics"]["pooled_top2pct_count_cutoff"] == 13
    assert audit["retrospective_metrics"]["extrapolative_expected_binders"] == pytest.approx(44.2181818182)


def test_handoff_records_directory_evidence_benchmark_receipt(tmp_path, monkeypatch):
    args, _ = _build_inputs(tmp_path, monkeypatch)
    evidence = pd.read_parquet(args.evidence_positive_controls)
    menu = handoff.evidence_review_menu(evidence, maximum=1)
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    metrics = pd.DataFrame([{
        "measured_panel_rows": 108,
        "pooled_positive_membership_cutoff": 13,
        "true_positives": 32,
        "measured_pooled_positive_members": 55,
        "precision": 32 / 55,
        "recall": 32 / 38,
        "fixed_unmeasured_menu_rows": len(menu),
        "extrapolated_expected_binders": len(menu) * 32 / 55,
    }])
    metrics_path = benchmark / "measured_membership_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    receipt = benchmark / "manifest.json"
    receipt.write_text(json.dumps({
        "schema_version": "liba-provider-pooled-membership-retrospective-benchmark-v1",
        "memberships": {
            "fixed_unmeasured_menu_sha256": handoff.membership_sha256(menu["pair_uid"])
        },
        "outputs": {metrics_path.name: {"sha256": _sha(metrics_path), "rows": 1}},
    }))
    args.evidence_control_comparison = benchmark
    handoff.run(args)
    release = json.loads((args.output_dir / "manifest.json").read_text())
    recorded = release["inputs"]["evidence_control_comparison"]
    assert recorded["path"] == str(benchmark.resolve())
    assert recorded["receipt_path"] == str(receipt.resolve())
    assert recorded["receipt_sha256"] == _sha(receipt)
