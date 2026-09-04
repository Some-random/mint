import hashlib
import json

import pytest

from downstream.AffibodyMHC import compile_liba_handoff_manifests as compiler


TARGETS = {f"P{i}": "SLLMWITQV" for i in range(9)}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path, model_id="mint_l9", source_candidate="mint_l9"):
    generic_lock = tmp_path / "generic_lock.json"
    generic_lock.write_text(json.dumps({"candidate": source_candidate, "labels": "weak_only"}))
    candidate_metrics = tmp_path / "candidate_metrics.csv"
    candidate_metrics.write_text(
        "candidate,within_peptide_ap,within_peptide_evaluable\n"
        f"{source_candidate},0.71,9\n"
    )
    model = {
        "model_id": model_id, "display_name": "Frozen MINT layer 9",
        "scientifically_eligible": True, "lock_id": "mint-l9-weak-v1",
        "score_threshold": 0.42, "threshold_metric": "weak_oof_macro_within_peptide_ap",
        "threshold_uses_retention": False,
        "threshold_provenance": "weak-label OOF selection only",
        "max_candidates_per_peptide": 10, "minimum_code_hamming_distance": 0,
        "weak_oof_sha256": "1" * 64, "head_sha256": "2" * 64,
        "config_sha256": "3" * 64, "training_data_sha256": "4" * 64,
        "training_rows": 12345,
        "weak_oof_within_peptide_average_precision": 0.71,
        "weak_oof_evaluable_peptides_for_within_peptide_ap": 9,
        "weak_metric_provenance": {
            "generic_lock_path": generic_lock.name,
            "generic_lock_sha256": _sha(generic_lock),
            "candidate_metrics_path": candidate_metrics.name,
            "candidate_metrics_sha256": _sha(candidate_metrics),
            "candidate": source_candidate,
        },
    }
    return {
        "schema_version": compiler.SELECTOR_SCHEMA, "library": "LibA",
        "labels": "weak_selection_only", "primary_model_id": model_id,
        "target_sequences": TARGETS, "models": [model],
        "training_description": "Provider-defined weak selection data.",
        "weak_label_definition": "Pooled R009+R010 top 2% positives.",
        "split_rule": "Both partner identities held out.",
        "evidence_max_controls_per_peptide": 10,
        "unavailable_structure_families": [
            {"family": "RDE-PPI", "reason": "No eligible LibA score artifact."},
            {"family": "StaB-ddG", "reason": "No eligible LibA score artifact."},
        ],
    }


def test_compiler_emits_deterministic_retention_blind_manifests(tmp_path):
    source = tmp_path / "selector.json"
    source.write_text(json.dumps(_bundle(tmp_path)))
    output = tmp_path / "compiled"
    compiler.compile_manifests(source, output)
    lock = json.loads((output / "liba_weak_oof_deployment_locks.json").read_text())
    project = json.loads((output / "liba_wetlab_project_spec.json").read_text())
    assert lock["retention_labels_read"] is False
    assert lock["models"]["mint_l9"]["training_data_sha256"] == "4" * 64
    assert lock["models"]["mint_l9"]["weak_oof_within_peptide_average_precision"] == 0.71
    assert lock["models"]["mint_l9"]["weak_metric_provenance"]["candidate"] == "mint_l9"
    assert project["expected_selection_missed_candidate_pairs"] == 447731
    assert project["expected_double_cold_candidate_pairs"] == 330880
    assert project["expected_peptide_cold_only_candidate_pairs"] == 116851
    assert project["evidence_positive_control"]["expected_pairs"] == 7786
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert lock["compiler_input_sha256"] == expected == project["compiler_input_sha256"]


def test_compiler_rejects_any_retention_field(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["retention_labels_read"] = False
    source = tmp_path / "selector.json"
    source.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="outcome-derived field forbidden"):
        compiler.compile_manifests(source, tmp_path / "compiled")


@pytest.mark.parametrize(("deployed_id", "source_candidate"), [
    ("mint_l9", "mean_logit__frozen_mint_layer9"),
    ("equal_logit_additive_mint_l9",
     "mean_logit__additive_6site__frozen_mint_layer9"),
])
def test_compiler_preserves_deployment_to_source_candidate_alias(
    tmp_path, deployed_id, source_candidate
):
    bundle = _bundle(tmp_path, deployed_id, source_candidate)
    source = tmp_path / "selector.json"
    source.write_text(json.dumps(bundle))
    output = tmp_path / "compiled"
    compiler.compile_manifests(source, output)
    lock = json.loads((output / "liba_weak_oof_deployment_locks.json").read_text())
    record = lock["models"][deployed_id]
    assert record["source_candidate"] == source_candidate
    assert record["weak_metric_provenance"]["candidate"] == source_candidate
