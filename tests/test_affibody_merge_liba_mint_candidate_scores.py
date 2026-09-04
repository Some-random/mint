import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.merge_liba_mint_candidate_scores import (
    FLAG_COLUMNS,
    MODEL_BY_LAYER,
    SOURCE_SCHEMA_BY_LAYER,
    sha256_file,
    validate_and_standardize_partition,
)


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_uid": ["pair-a", "pair-b", "pair-c"],
            "library": ["LibA"] * 3,
            "peptide_design_code": ["AF"] * 3,
            "peptide_9mer_sequence": ["SLLAFITQV"] * 3,
            "affibody_design_code": ["AAAA", "AAAD", "AAAE"],
            "peptide_uid": ["pep-a"] * 3,
            "affibody_uid": ["aff-a", "aff-b", "aff-c"],
            "chain1_sha256": ["1" * 64] * 3,
            "chain2_sha256": ["2" * 64, "3" * 64, "4" * 64],
            "provider_displayed_58aa_affibody_sequence": ["A" * 58, "D" * 58, "E" * 58],
            "model_input_affibody_sequence": ["A" * 58, "D" * 58, "E" * 58],
            "model_input_smart_hla_linker_peptide_sequence": ["M" * 261 + "SLLAFITQV"] * 3,
            "observed_in_any_raw_round": [False, True, True],
            "observed_in_r009_or_r010": [False, False, True],
            "affibody_identity_seen_in_strict_training": [False, True, True],
            "high_confidence_weak_negative": [False, True, False],
        }
    )


def _write_fixture(tmp_path: Path, layer: int = 9):
    universe_path = tmp_path / "universe.parquet"
    score_path = tmp_path / "score.parquet"
    manifest_path = tmp_path / "score.parquet.manifest.json"
    universe = _universe_frame()
    universe.to_parquet(universe_path, index=False)
    logits = np.asarray([-1.0, 0.0, 1.0])
    scores = 1.0 / (1.0 + np.exp(-logits))
    scored = universe.copy()
    scored.insert(0, "input_row_index", np.arange(3, dtype=np.uint64))
    scored.insert(1, "model_id", MODEL_BY_LAYER[layer])
    scored["model_score"] = scores
    scored[f"mint_layer{layer}_logit"] = logits
    scored.to_parquet(score_path, index=False)
    manifest = {
        "schema_version": SOURCE_SCHEMA_BY_LAYER[layer],
        "rows": 3,
        "peptide_design_code": "AF",
        "model": {"library": "LibA", "layer": layer, "model_id": MODEL_BY_LAYER[layer]},
        "inputs": {
            "candidate_parquet": {
                "path": str(universe_path.resolve()),
                "sha256": sha256_file(universe_path),
            },
            "parity_receipt": {"path": "parity.json", "sha256": "a" * 64},
        },
        "parity": {
            "ordinary_vs_early_features_bitwise_equal": True,
            "head_score_vs_reference_max_abs": 0.0,
            "score_atol": 1e-12,
        },
        "output": {
            "path": str(score_path.resolve()),
            "sha256": sha256_file(score_path),
            "bytes": score_path.stat().st_size,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return universe_path, score_path, manifest_path


def test_partition_merge_validates_identity_and_sigmoid(tmp_path):
    universe_path, score_path, manifest_path = _write_fixture(tmp_path)
    output, audit = validate_and_standardize_partition(
        universe_path, score_path, manifest_path, "AF", 9, 3
    )
    assert output["model_id"].eq("mint_l9").all()
    assert output["pair_uid"].tolist() == ["pair-a", "pair-b", "pair-c"]
    assert set(FLAG_COLUMNS).issubset(output.columns)
    assert audit["maximum_absolute_sigmoid_parity_difference"] <= 2e-12
    assert audit["source"]["score"]["sha256"] == sha256_file(score_path)


def test_partition_merge_rejects_sequence_mapping_mismatch(tmp_path):
    universe_path, score_path, manifest_path = _write_fixture(tmp_path)
    frame = pd.read_parquet(score_path)
    frame.loc[1, "model_input_affibody_sequence"] = "W" * 58
    frame.to_parquet(score_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = sha256_file(score_path)
    manifest["output"]["bytes"] = score_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="model_input_affibody_sequence mismatch"):
        validate_and_standardize_partition(
            universe_path, score_path, manifest_path, "AF", 9, 3
        )


def test_partition_merge_rejects_probability_not_equal_to_logit(tmp_path):
    universe_path, score_path, manifest_path = _write_fixture(tmp_path)
    frame = pd.read_parquet(score_path)
    frame.loc[2, "model_score"] = 0.25
    frame.to_parquet(score_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = sha256_file(score_path)
    manifest["output"]["bytes"] = score_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="logit/probability parity failed"):
        validate_and_standardize_partition(
            universe_path, score_path, manifest_path, "AF", 9, 3
        )


def test_partition_merge_rejects_stale_source_receipt(tmp_path):
    universe_path, score_path, manifest_path = _write_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = hashlib.sha256(b"stale").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source score hash changed"):
        validate_and_standardize_partition(
            universe_path, score_path, manifest_path, "AF", 9, 3
        )
