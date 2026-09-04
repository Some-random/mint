import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.apply_liba_locked_candidate_ensemble import (
    PARITY_TOLERANCE,
    apply_formula,
    load_lock,
    sha256_file,
    verify_evaluation_formula,
    verify_oof_formula,
)


def _score(probability):
    probability = np.asarray(probability, dtype=float)
    return np.log(probability) - np.log1p(-probability)


def _lock(family="equal_mean_logit") -> dict:
    components = [
        {
            "name": "additive_6site",
            "inference_logit_column": "additive_6site_logit",
            "weight": 0.5,
            "center": 0.0,
            "scale": 1.0,
        },
        {
            "name": "frozen_mint_layer9",
            "inference_logit_column": "mint_layer9_logit",
            "weight": 0.5,
            "center": 0.0,
            "scale": 1.0,
        },
    ]
    return {
        "schema_version": "affibody-weak-oof-score-lock-v1",
        "library": "LibA",
        "lock_role": "best_equal_logit_ensemble",
        "lock_id": "liba-equal-test",
        "candidate": "mean_logit__additive_6site__frozen_mint_layer9",
        "candidate_family": family,
        "members": [value["name"] for value in components],
        "formula": {
            "input_semantics": "true component logits",
            "linear_predictor": "sum(weight_j * logit_j)",
            "intercept": 0.0,
            "components": components,
            "output_transform": "sigmoid",
            "equal_logit_weights": True,
        },
        "cutoff": {"decision_rule": "score >= threshold", "score_threshold": 0.6},
        "retention_labels_read": False,
    }


def _write_lock(path: Path, family="equal_mean_logit") -> None:
    path.write_text(json.dumps(_lock(family)), encoding="utf-8")


def test_equal_logit_formula_uses_logits_not_probability_mean():
    left = np.asarray([0.01, 0.8])
    right = np.asarray([0.5, 0.9])
    linear, output = apply_formula(
        _lock(),
        {
            "additive_6site": _score(left),
            "frozen_mint_layer9": _score(right),
        },
    )
    expected = 1.0 / (1.0 + np.exp(-(_score(left) + _score(right)) / 2.0))
    assert np.allclose(output, expected, atol=1e-15, rtol=0)
    assert np.allclose(linear, (_score(left) + _score(right)) / 2.0)
    assert output[0] != pytest.approx((left[0] + right[0]) / 2.0)


def test_load_lock_rejects_cross_fitted_stacker_for_deployment(tmp_path):
    path = tmp_path / "stacker.lock.json"
    _write_lock(path, family="cross_fitted_nonnegative_logistic")
    with pytest.raises(ValueError, match="only single/equal-logit locks are deployable"):
        load_lock(path)


def test_weak_oof_formula_parity_reads_no_label(tmp_path):
    left = np.asarray([0.1, 0.3, 0.9])
    right = np.asarray([0.2, 0.8, 0.7])
    _, expected = apply_formula(
        _lock(),
        {"additive_6site": _score(left), "frozen_mint_layer9": _score(right)},
    )
    path = tmp_path / "matched_oof.csv.gz"
    pd.DataFrame(
        {
            "row_id": ["a", "b", "c"],
            "weak_label": [0, 1, 1],  # deliberately present but must not be read
            "additive_6site": left,
            "frozen_mint_layer9": right,
            "mean_logit__additive_6site__frozen_mint_layer9": expected,
        }
    ).to_csv(path, index=False, compression="gzip")
    audit = verify_oof_formula(_lock(), path)
    assert audit["maximum_absolute_score_difference"] <= PARITY_TOLERANCE
    assert audit["weak_label_column_read"] is False
    assert "weak_label" not in audit["columns_read"]


def test_target_free_evaluation_formula_parity_across_long_and_aligned(tmp_path, monkeypatch):
    import downstream.AffibodyMHC.apply_liba_locked_candidate_ensemble as module

    monkeypatch.setattr(module, "EXPECTED_EVALUATION_ROWS", 3)
    ids = ["e1", "e2", "e3"]
    left = np.asarray([0.1, 0.2, 0.3])
    right = np.asarray([0.6, 0.7, 0.8])
    aligned_path = tmp_path / "evaluation_aligned.csv"
    long_path = tmp_path / "evaluation_long.csv"
    pd.DataFrame(
        {
            "eval_row_id": ids,
            "additive_6site": left,
            "frozen_mint_layer9": right,
        }
    ).to_csv(aligned_path, index=False)
    blocks = []
    for model, values in (
        ("additive_6site", left),
        ("frozen_mint_layer9", right),
    ):
        blocks.append(
            pd.DataFrame(
                {"eval_row_id": ids, "model": model, "seed": "fixed", "score": values}
            )
        )
    pd.concat(blocks, ignore_index=True).to_csv(long_path, index=False)
    output, audit = verify_evaluation_formula(_lock(), aligned_path, long_path)
    assert len(output) == 3
    assert audit["ensemble_maximum_absolute_score_difference"] <= PARITY_TOLERANCE
    assert audit["outcome_columns_read"] == []


def test_target_free_evaluation_reconstructs_multiseed_mean_logit(tmp_path, monkeypatch):
    import downstream.AffibodyMHC.apply_liba_locked_candidate_ensemble as module

    monkeypatch.setattr(module, "EXPECTED_EVALUATION_ROWS", 3)
    ids = ["e1", "e2", "e3"]
    seed1 = np.asarray([0.1, 0.2, 0.3])
    seed2 = np.asarray([0.3, 0.4, 0.5])
    aggregate = 1.0 / (1.0 + np.exp(-(_score(seed1) + _score(seed2)) / 2.0))
    nonlinear_lock = _lock()
    nonlinear_lock["members"] = ["nonlinear_6site_mean_logit"]
    nonlinear_lock["candidate"] = "mean_logit__nonlinear_6site_mean_logit"
    nonlinear_lock["candidate_family"] = "single"
    nonlinear_lock["formula"]["components"] = [
        {
            "name": "nonlinear_6site_mean_logit",
            "inference_logit_column": "nonlinear_6site_ensemble_logit",
            "weight": 1.0,
            "center": 0.0,
            "scale": 1.0,
        }
    ]
    aligned_path = tmp_path / "evaluation_aligned.csv"
    long_path = tmp_path / "evaluation_long.csv"
    pd.DataFrame(
        {"eval_row_id": ids, "nonlinear_6site_mean_logit": aggregate}
    ).to_csv(aligned_path, index=False)
    pd.concat(
        [
            pd.DataFrame(
                {"eval_row_id": ids, "model": "nonlinear_6site", "seed": "1", "score": seed1}
            ),
            pd.DataFrame(
                {"eval_row_id": ids, "model": "nonlinear_6site", "seed": "2", "score": seed2}
            ),
        ],
        ignore_index=True,
    ).to_csv(long_path, index=False)
    output, audit = verify_evaluation_formula(nonlinear_lock, aligned_path, long_path)
    assert np.allclose(output["model_score"], aggregate, atol=PARITY_TOLERANCE, rtol=0)
    assert audit["component_maximum_absolute_differences"]["nonlinear_6site_mean_logit"] <= PARITY_TOLERANCE


def test_component_manifest_hash_can_be_verified(tmp_path):
    # Guard the hash helper used by the completion-manifest chain against text
    # rather than binary hashing regressions.
    path = tmp_path / "score.parquet"
    path.write_bytes(b"candidate-scores")
    assert sha256_file(path) == __import__("hashlib").sha256(b"candidate-scores").hexdigest()


def _candidate_component(path: Path, model_id: str, logit_column: str, probability) -> None:
    path.parent.mkdir()
    probability = np.asarray(probability, dtype=float)
    frame = pd.DataFrame(
        {
            "model_id": model_id,
            "pair_uid": ["p1", "p2", "p3"],
            "peptide_design_code": ["AF", "DL", "EA"],
            "peptide_9mer_sequence": ["SLLAFITQV", "SLLDLITQV", "SLLEAITQV"],
            "affibody_design_code": ["AAAA", "AAAD", "AAAE"],
            "provider_displayed_58aa_affibody_sequence": ["A" * 58, "D" * 58, "E" * 58],
            "model_input_affibody_sequence": ["A" * 58, "D" * 58, "E" * 58],
            "model_input_smart_hla_linker_peptide_sequence": ["M" * 261 + x for x in ("SLLAFITQV", "SLLDLITQV", "SLLEAITQV")],
            "model_score": probability,
            "observed_in_any_raw_round": [False, True, True],
            "observed_in_r009_or_r010": [False, False, True],
            "affibody_identity_seen_in_strict_training": [False, True, True],
            "high_confidence_weak_negative": [False, True, False],
            logit_column: _score(probability),
        }
    )
    frame.to_parquet(path, index=False)
    (path.parent / "manifest.json").write_text(
        json.dumps({"schema_version": "test", "output": {"sha256": sha256_file(path)}}),
        encoding="utf-8",
    )


def test_full_equal_logit_application_is_atomic_and_standardized(tmp_path, monkeypatch):
    import downstream.AffibodyMHC.apply_liba_locked_candidate_ensemble as module

    monkeypatch.setattr(module, "EXPECTED_CANDIDATE_ROWS", 3)
    monkeypatch.setattr(module, "EXPECTED_EVALUATION_ROWS", 3)
    monkeypatch.setattr(module, "EXPECTED_PEPTIDE_COUNT", 3)
    monkeypatch.setattr(module, "PRIVATE_ROOT", tmp_path.resolve())
    lock_path = tmp_path / "equal.lock.json"
    _write_lock(lock_path)
    additive_path = tmp_path / "additive" / "candidate_scores.parquet"
    mint_path = tmp_path / "mint" / "candidate_scores.parquet"
    left = np.asarray([0.1, 0.3, 0.9])
    right = np.asarray([0.2, 0.8, 0.7])
    _candidate_component(additive_path, "additive_6site", "additive_6site_logit", left)
    _candidate_component(mint_path, "mint_l9", "mint_layer9_logit", right)
    _, expected = apply_formula(
        _lock(),
        {"additive_6site": _score(left), "frozen_mint_layer9": _score(right)},
    )
    oof_path = tmp_path / "matched_oof.csv.gz"
    pd.DataFrame(
        {
            "row_id": ["o1", "o2", "o3"],
            "additive_6site": left,
            "frozen_mint_layer9": right,
            "mean_logit__additive_6site__frozen_mint_layer9": expected,
        }
    ).to_csv(oof_path, index=False, compression="gzip")
    aligned_path = tmp_path / "evaluation_aligned.csv"
    long_path = tmp_path / "evaluation_long.csv"
    pd.DataFrame(
        {"eval_row_id": ["e1", "e2", "e3"], "additive_6site": left, "frozen_mint_layer9": right}
    ).to_csv(aligned_path, index=False)
    pd.concat(
        [
            pd.DataFrame({"eval_row_id": ["e1", "e2", "e3"], "model": "additive_6site", "seed": "fixed", "score": left}),
            pd.DataFrame({"eval_row_id": ["e1", "e2", "e3"], "model": "frozen_mint_layer9", "seed": "fixed", "score": right}),
        ],
        ignore_index=True,
    ).to_csv(long_path, index=False)
    output_dir = tmp_path / "output"
    module.run(
        Namespace(
            model_id="equal_logit_additive_mint_l9",
            output_dir=output_dir,
            lock_json=lock_path,
            component_score=[
                f"additive_6site={additive_path}",
                f"frozen_mint_layer9={mint_path}",
            ],
            matched_oof=oof_path,
            evaluation_aligned=aligned_path,
            evaluation_long=long_path,
            head_binding=lock_path,
            config_binding=lock_path,
            weak_oof_binding=oof_path,
        )
    )
    scored = pd.read_parquet(output_dir / "candidate_scores.parquet")
    assert scored.columns[: len(module.STANDARD_COLUMNS)].tolist() == list(module.STANDARD_COLUMNS)
    assert np.allclose(scored["model_score"], expected, atol=PARITY_TOLERANCE, rtol=0)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"]["exact_weak_oof_formula_parity"] is True
    assert manifest["outcome_access"]["retention_measurements_read"] is False
    assert manifest["deployment_binding"]["generic_lock_sha256"] == sha256_file(lock_path)
