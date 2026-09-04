import hashlib
import json
from pathlib import Path

import pandas as pd

from downstream.AffibodyMHC import finalize_esmfold2_liba_candidate_scale_gate as gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _metric_row(model: str, seed: str, ap: float, spearman: float) -> dict:
    return {
        "model": model,
        "seed": seed,
        "n_examples": 108,
        "within_peptide_binary_evaluable_groups": 5,
        "within_peptide_spearman_evaluable_groups": 9,
        "within_peptide_average_precision_mean": ap,
        "within_peptide_spearman_mean": spearman,
    }


def _fixture(tmp_path: Path, selected_family: str = "distogram_only") -> tuple[Path, Path, Path]:
    selection = tmp_path / "all_model_selection.json"
    model_records = {}
    roles = {}
    for family in gate.FAMILY_TO_EVALUATION_MODEL:
        folding = family in gate.STRUCTURE_DERIVED_FAMILIES
        roles[family] = "folding_derived" if folding else "pre_trunk_sequence_control"
        model_records[family] = {
            "reproducibly_beats_mint_layer9": folding,
            "passes_pretrunk_control": folding,
            "supports_structure_claim": folding,
        }
    _write_json(
        selection,
        {
            "schema_version": gate.SELECTION_SCHEMA,
            "retention_labels_read": False,
            "selected_feature_family": selected_family,
            "family_roles": roles,
            "models": model_records,
        },
    )
    completion = tmp_path / "aggregation_completion.json"
    _write_json(
        completion,
        {
            "schema_version": gate.AGGREGATION_COMPLETION_SCHEMA,
            "retention_labels_read": False,
            "outputs": {
                "all_model_selection": {
                    "path": str(selection.resolve()),
                    "sha256": _sha(selection),
                }
            },
        },
    )

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    controls = [
        _metric_row("additive_6site", "deployment", 0.59, -0.34),
        _metric_row("mint_l9", "deployment", 0.587, -0.41),
        _metric_row(
            "mint_l33_control",
            "deployment",
            gate.EXPECTED_BEST_SEQUENCE_AP,
            gate.EXPECTED_BEST_SEQUENCE_SPEARMAN,
        ),
        _metric_row("nonlinear_6site_mean_logit", "deployment", 0.599, -0.35),
        _metric_row("equal_logit_additive_mint_l9", "deployment", 0.588, -0.37),
    ]
    esm_models = [
        _metric_row(model, "deployment", 0.61, -0.29)
        for model in gate.FAMILY_TO_EVALUATION_MODEL.values()
    ]
    deployment = evaluation / "deployment_metrics.csv"
    pd.DataFrame(controls + esm_models).to_csv(deployment, index=False)
    raw = evaluation / "raw_metrics_by_seed.csv"
    pd.DataFrame(
        [
            _metric_row(family, seed, 0.61, -0.29)
            for family in gate.FAMILY_TO_EVALUATION_MODEL
            for seed in sorted(gate.FINAL_SEEDS)
        ]
    ).to_csv(raw, index=False)
    _write_json(
        evaluation / "manifest.json",
        {
            "schema_version": gate.EVALUATION_MANIFEST_SCHEMA,
            "retention_usage": {
                "final_retrospective_evaluation": True,
                "model_selection": False,
            },
            "outputs": {
                "deployment_metrics.csv": {
                    "path": str(deployment.resolve()),
                    "sha256": _sha(deployment),
                },
                "raw_metrics_by_seed.csv": {
                    "path": str(raw.resolve()),
                    "sha256": _sha(raw),
                },
            },
        },
    )
    return selection, completion, evaluation


def test_gate_passes_only_for_weak_locked_folding_family(tmp_path: Path) -> None:
    selection, completion, evaluation = _fixture(tmp_path)
    result = gate.run(selection, completion, evaluation, tmp_path / "out")
    assert result["decision"] == "GO"
    assert result["retrospective_retention_gate"]["individual_seeds_beating_ap"] == 5
    assert (tmp_path / "out/decision_gate.json").is_file()


def test_pretrunk_winner_cannot_authorize_structure_scale_up(tmp_path: Path) -> None:
    selection, completion, evaluation = _fixture(tmp_path, "single_inputs_only")
    result = gate.run(selection, completion, evaluation, tmp_path / "out")
    assert result["decision"] == "NO_GO"
    assert result["weak_label_gate"]["passed"] is False
