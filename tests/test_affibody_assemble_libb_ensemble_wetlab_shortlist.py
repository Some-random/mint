import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.assemble_libb_ensemble_wetlab_shortlist import (
    diversity_select,
    load_component_scores,
    load_lock,
    score_partition,
    sha256_file,
    validate_component_score_provenance,
)


def _lock():
    return {
        "schema_version": "libb-score-ensemble-lock-v1",
        "lock_id": "libb-mint-stab-equal-logit-exploratory-weak-oof-v1",
        "display_name": "synthetic MINT plus StaB test lock",
        "components": [
            {
                "name": "mint_layer5",
                "score_column": "score",
                "weight": 0.5,
                "center": 0.5,
                "scale": 0.25,
                "higher_is_better": True,
            },
            {
                "name": "stab_designed_ordered",
                "score_column": "score",
                "weight": 1.0,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
        ],
        "intercept": -0.25,
        "output_transform": "sigmoid",
        "selection": {
            "score_threshold": 0.6,
            "max_candidates_per_peptide": 2,
            "minimum_code_hamming_distance": 2,
            "review_pool_per_peptide": 3,
            "pad_below_threshold": False,
        },
        "weight_selection_provenance": "synthetic weak-validation test",
        "threshold_selection_provenance": "synthetic frozen test cutoff",
        "score_semantics": "synthetic selection score, not retention",
        "component_score_contract": "synthetic true logits",
        "weak_source_sha256": {"oof": "a" * 64},
        "retention_labels_read": False,
    }


def test_lock_requires_explicit_nonpadding_policy(tmp_path):
    lock = _lock()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    assert load_lock(path)["selection"]["pad_below_threshold"] is False
    lock["selection"]["pad_below_threshold"] = True
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="padding"):
        load_lock(path)


def test_component_loader_rejects_retention_columns(tmp_path):
    path = tmp_path / "scores.parquet"
    pd.DataFrame(
        {"pair_uid": ["p1"], "score": [0.5], "target_retention": [99.0]}
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="forbidden outcomes"):
        load_component_scores(path, "AF", "mint_layer5", "score")


def test_component_loader_rejects_unlisted_retention_spelling(tmp_path):
    path = tmp_path / "scores.parquet"
    pd.DataFrame(
        {"pair_uid": ["p1"], "score": [0.5], "measured_retention_value": [99.0]}
    ).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="forbidden outcomes"):
        load_component_scores(path, "AF", "mint_layer5", "score")


def test_score_join_and_diversity_selection(tmp_path):
    universe = pd.DataFrame(
        {
            "pair_uid": ["p1", "p2", "p3", "p4"],
            "peptide_design_code": ["AF"] * 4,
            "affibody_design_code": ["AAAAA", "AAAAD", "AADDA", "DDDDD"],
            "observed_in_any_raw_round": [False, True, False, False],
            "high_confidence_weak_negative": [False, False, True, False],
            "affibody_identity_seen_in_strict_training": [False, True, False, False],
        }
    )
    roots = {}
    for name, values in {
        "mint_layer5": [0.90, 0.85, 0.80, 0.70],
        "stab_designed_ordered": [1.00, 0.95, 0.90, 0.85],
    }.items():
        root = tmp_path / name
        root.mkdir()
        pd.DataFrame(
            {
                "pair_uid": universe["pair_uid"],
                "peptide_design_code": "AF",
                "affibody_design_code": universe["affibody_design_code"],
                "score": values,
            }
        ).to_parquet(root / "peptide_AF.parquet", index=False)
        roots[name] = root

    ranked, files = score_partition(universe, "AF", _lock(), roots)
    assert set(files) == {"mint_layer5", "stab_designed_ordered"}
    assert ranked["pair_uid"].tolist() == ["p1", "p2", "p3", "p4"]
    expected_first_logit = -0.25 + 0.5 * ((0.90 - 0.5) / 0.25) + 1.0
    assert ranked.iloc[0]["ensemble_linear_score"] == pytest.approx(expected_first_logit)
    assert ranked.iloc[0]["ensemble_score"] == pytest.approx(
        1.0 / (1.0 + np.exp(-expected_first_logit))
    )
    assert ranked["within_peptide_mint_layer5_rank"].tolist() == [1, 2, 3, 4]
    assert ranked["within_peptide_stab_designed_ordered_rank"].tolist() == [1, 2, 3, 4]
    assert ranked["component_top10_votes"].tolist() == [2, 2, 2, 2]

    selected = diversity_select(ranked, threshold=0.6, maximum=2, minimum_hamming=2)
    # p2 differs from p1 at only one code position, so the greedy selector skips
    # it and takes p3, whose code differs at two positions.
    assert selected["pair_uid"].tolist() == ["p1", "p3"]
    assert selected["wetlab_rank"].tolist() == [1, 2]


def test_native_lock_rejects_fixed_adapter_structure_scores(tmp_path):
    score_path = tmp_path / "peptide_AF.parquet"
    pd.DataFrame(
        {
            "pair_uid": ["p1"],
            "peptide_design_code": ["AF"],
            "affibody_design_code": ["AAAAA"],
            "rde_logit": [0.5],
        }
    ).to_parquet(score_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": "libb-fixed-structure-candidate-scores-merged-v1",
        "family": "rde",
        "candidate_universe": {"manifest_sha256": "u" * 64},
        "source_chunks": {
            "readout_provenance": {"readout_mode": "legacy_fixed_adapter"},
        },
        "outputs": {
            "partitions": [
                {
                    "peptide_design_code": "AF",
                    "universe_partition_sha256": "p" * 64,
                    "output_sha256": sha256_file(score_path),
                }
            ]
        },
        "outcome_access": {
            "selection_labels_read": False,
            "retention_measurements_read": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'native_learned_projection'"):
        validate_component_score_provenance(
            tmp_path,
            score_path,
            "rde_network_designed_3fold",
            "AF",
            "u" * 64,
            "p" * 64,
            expected_readout_mode="native_learned_projection",
        )

    manifest["source_chunks"]["readout_provenance"]["readout_mode"] = (
        "native_learned_projection"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = validate_component_score_provenance(
        tmp_path,
        score_path,
        "rde_network_designed_3fold",
        "AF",
        "u" * 64,
        "p" * 64,
        expected_readout_mode="native_learned_projection",
    )
    assert record["score_sha256"] == sha256_file(score_path)
