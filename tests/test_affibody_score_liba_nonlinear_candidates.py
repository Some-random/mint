import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC import score_liba_nonlinear_candidates as scorer
from downstream.AffibodyMHC.build_liba_wetlab_candidate_handoff import (
    load_candidate_scores,
)
from downstream.AffibodyMHC.code_only_baseline import opaque_id


SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)


def _write_checkpoints(root):
    root.mkdir()
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = scorer.NonlinearSixSiteMLP(120, (64, 32), 0.1)
        torch.save(
            {
                "schema_version": scorer.CHECKPOINT_SCHEMA_VERSION,
                "model": scorer.PRODUCER_MODEL,
                "seed": seed,
                "derived_training_seed": seed + 100,
                "epochs": 3,
                "input_dimension": 120,
                "hidden_dimensions": [64, 32],
                "dropout": 0.1,
                "position_names": list(scorer.POSITION_NAMES),
                "amino_acid_alphabet": list(scorer.AA_ALPHABET),
                "weak_training_membership_sha256": scorer.EXPECTED_TRAINING_MEMBERSHIP_SHA256,
                "state_dict": model.state_dict(),
            },
            root / f"nonlinear_6site__seed{seed}.pt",
        )


def _known_panel():
    peptides = ["AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL"]
    affibodies = [
        "".join(code)
        for code in itertools.islice(itertools.product(scorer.AA_ALPHABET, repeat=4), 12)
    ]
    rows = []
    for peptide in peptides:
        for affibody in affibodies:
            rows.append(
                {
                    "eval_row_id": opaque_id("LibA", peptide, affibody),
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": "SECRET_RETENTION",
                    "target_binder": "SECRET_BINDER",
                }
            )
    return pd.DataFrame(rows)


def _reference_scores(panel, checkpoint_dir):
    models, records = scorer.load_checkpoints(checkpoint_dir, torch.device("cpu"))
    _, probability, _, _ = scorer.score_features(
        models, scorer.encode_codes(panel), 64, torch.device("cpu")
    )
    rows = []
    for index, record in enumerate(records):
        for row_id, score in zip(panel["eval_row_id"], probability[:, index]):
            rows.append(
                {
                    "eval_row_id": row_id,
                    "model": scorer.PRODUCER_MODEL,
                    "seed": str(record["seed"]),
                    "score": score,
                    "target_retention": "SECRET_PARITY_RETENTION",
                }
            )
    return pd.DataFrame(rows)


def _displayed_affibody(code):
    sequence = list("A" * 58)
    for index, residue in zip((12, 16, 26, 30), code):
        sequence[index] = residue
    return "".join(sequence)


def _write_universe(root, peptides):
    partitions_dir = root / "candidates_by_peptide"
    partitions_dir.mkdir(parents=True)
    records = []
    for peptide in peptides:
        codes = ("YYYY", "VVVV")
        frame = pd.DataFrame(
            {
                "pair_uid": [opaque_id("LibA", peptide, code) for code in codes],
                "peptide_design_code": [peptide] * 2,
                "affibody_design_code": list(codes),
                "peptide_9mer_sequence": [f"SLL{peptide}ITQV"] * 2,
                "provider_displayed_58aa_affibody_sequence": [
                    _displayed_affibody(code) for code in codes
                ],
                "model_input_smart_hla_linker_peptide_sequence": [
                    "A" * 261 + f"SLL{peptide}ITQV"
                ] * 2,
                "observed_in_any_raw_round": [False, True],
                "observed_in_r009_or_r010": [False, True],
                "affibody_identity_seen_in_strict_training": [True, False],
                "high_confidence_weak_negative": [False, True],
            }
        )
        frame["model_input_affibody_sequence"] = frame[
            "provider_displayed_58aa_affibody_sequence"
        ]
        relative = f"candidates_by_peptide/peptide_{peptide}.parquet"
        path = root / relative
        frame.to_parquet(path, index=False)
        records.append(
            {
                "peptide_design_code": peptide,
                "candidate_rows": len(frame),
                "partition": relative,
                "partition_sha256": scorer.sha256_file(path),
            }
        )
    partitions = pd.DataFrame(records)
    partitions_path = root / "candidate_partitions.csv"
    partitions.to_csv(partitions_path, index=False)
    manifest = {
        "schema_version": scorer.UNIVERSE_SCHEMA_VERSION,
        "input_data_contract": {"retention_outcome_columns_read": []},
        "outputs": {
            "candidate_rows": int(partitions["candidate_rows"].sum()),
            "candidate_partitions_csv": {"sha256": scorer.sha256_file(partitions_path)},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")


def test_mean_logit_uses_logit_average_not_probability_average():
    probability = np.asarray([[0.01, 0.10, 0.70, 0.80, 0.90]], dtype=float)
    mean, score = scorer.mean_logit_score(probability)
    expected_logit = np.mean(np.log(probability) - np.log1p(-probability), axis=1)
    np.testing.assert_allclose(mean, expected_logit, rtol=0, atol=1e-15)
    np.testing.assert_allclose(score, 1.0 / (1.0 + np.exp(-expected_logit)))
    assert not np.isclose(score[0], probability.mean())


def test_five_seed_parity_gate_and_standardized_candidate_output(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    _write_checkpoints(checkpoints)
    panel = _known_panel()
    panel_path = tmp_path / "evaluation_roster.csv"
    panel.to_csv(panel_path, index=False)
    reference = _reference_scores(panel, checkpoints)
    reference_path = tmp_path / "evaluation_predictions.csv"
    reference.to_csv(reference_path, index=False)
    universe = tmp_path / "universe"
    universe.mkdir()
    peptides = sorted(panel["peptide_design_code"].unique())
    _write_universe(universe, peptides)

    output = tmp_path / "scores"
    monkeypatch.setattr(scorer, "PRIVATE_ROOT", tmp_path.resolve())
    args = SimpleNamespace(
        universe_dir=universe,
        checkpoint_dir=checkpoints,
        known_panel=panel_path,
        parity_predictions=reference_path,
        output_dir=output,
        device="cpu",
        batch_size=64,
    )
    manifest = scorer.run(args)
    assert manifest["rows"] == 18
    assert manifest["known_108_pair_parity"]["status"] == "passed"
    assert len(manifest["known_108_pair_parity"]["seeds"]) == 5
    assert manifest["retention_or_binder_outcomes_read"] is False
    assert "SECRET" not in (output / "manifest.json").read_text()

    combined = pd.read_parquet(output / "candidate_scores.parquet")
    assert len(combined) == 18
    assert combined["model_id"].eq(scorer.MODEL_ID).all()
    np.testing.assert_array_equal(
        combined["model_score"], combined["nonlinear_6site_mean_logit"]
    )
    seed_scores = combined[
        [f"nonlinear_6site__seed{seed}_score" for seed in SEEDS]
    ].to_numpy(float)
    _, expected = scorer.mean_logit_score(seed_scores)
    np.testing.assert_allclose(combined["model_score"], expected, rtol=0, atol=0)
    validated = load_candidate_scores(
        scorer.MODEL_ID,
        output / "candidate_scores.parquet",
        {peptide: f"SLL{peptide}ITQV" for peptide in peptides},
    )
    assert len(validated) == 18

    with pytest.raises(ValueError, match="output exists"):
        scorer.run(args)


def test_parity_failure_occurs_before_output_creation(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    _write_checkpoints(checkpoints)
    panel = _known_panel()
    panel_path = tmp_path / "evaluation_roster.csv"
    panel.to_csv(panel_path, index=False)
    reference = _reference_scores(panel, checkpoints)
    reference.loc[0, "score"] += 2e-6
    reference_path = tmp_path / "evaluation_predictions.csv"
    reference.to_csv(reference_path, index=False)
    universe = tmp_path / "universe"
    universe.mkdir()
    _write_universe(universe, sorted(panel["peptide_design_code"].unique()))
    output = tmp_path / "failed"
    monkeypatch.setattr(scorer, "PRIVATE_ROOT", tmp_path.resolve())
    args = SimpleNamespace(
        universe_dir=universe,
        checkpoint_dir=checkpoints,
        known_panel=panel_path,
        parity_predictions=reference_path,
        output_dir=output,
        device="cpu",
        batch_size=64,
    )
    with pytest.raises(ValueError, match="108-row parity failed"):
        scorer.run(args)
    assert not output.exists()


def test_checkpoint_contract_rejects_wrong_position_order(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    _write_checkpoints(checkpoints)
    path = checkpoints / f"nonlinear_6site__seed{SEEDS[0]}.pt"
    payload = torch.load(path, weights_only=True)
    payload["position_names"] = list(reversed(payload["position_names"]))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="position order"):
        scorer.load_checkpoints(checkpoints, torch.device("cpu"))


def test_membership_hash_is_order_independent():
    values = pd.Series(["b", "a", "c"])
    expected = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert scorer._membership_sha256(values) == expected
    assert scorer._membership_sha256(values.iloc[::-1]) == expected
