import hashlib
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import score_liba_additive_candidates as scorer
from downstream.AffibodyMHC.build_liba_wetlab_candidate_handoff import (
    load_candidate_scores,
)
from downstream.AffibodyMHC.code_only_baseline import opaque_id


def _write_head(path, feature_names=None):
    weight = np.arange(6 * 20, dtype=np.float64).reshape(6, 20) / 100.0
    np.savez(
        path,
        schema_version=np.asarray([scorer.HEAD_SCHEMA_VERSION]),
        feature_names=np.asarray(feature_names or scorer.POSITION_NAMES),
        residue_alphabet=np.asarray(scorer.AA_ALPHABET),
        weight=weight,
        bias=np.asarray([-0.7], dtype=np.float64),
        selected_C=np.asarray([1.0], dtype=np.float64),
        training_membership_sha256=np.asarray(["a" * 64]),
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
                    "pair_uid": opaque_id("LibA", peptide, affibody),
                    "library": "LibA",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": "SECRET_RETENTION",
                    "target_binder": "SECRET_BINDER",
                }
            )
    return pd.DataFrame(rows)


def _write_universe(root, peptides):
    partitions_dir = root / "candidates_by_peptide"
    partitions_dir.mkdir(parents=True)
    records = []
    for peptide in peptides:
        frame = pd.DataFrame(
            {
                "pair_uid": [
                    opaque_id("LibA", peptide, "YYYY"),
                    opaque_id("LibA", peptide, "VVVV"),
                ],
                "peptide_design_code": [peptide, peptide],
                "affibody_design_code": ["YYYY", "VVVV"],
                "peptide_9mer_sequence": [f"SLL{peptide}ITQV"] * 2,
                "provider_displayed_58aa_affibody_sequence": [
                    "A" * 12 + "Y" + "A" * 3 + "Y" + "A" * 9 + "Y" + "A" * 3 + "Y" + "A" * 27,
                    "A" * 12 + "V" + "A" * 3 + "V" + "A" * 9 + "V" + "A" * 3 + "V" + "A" * 27,
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
            "candidate_partitions_csv": {
                "sha256": scorer.sha256_file(partitions_path)
            },
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")


def test_head_contract_and_vectorized_scoring(tmp_path):
    path = tmp_path / "head.npz"
    _write_head(path)
    head = scorer.load_head(path)
    frame = pd.DataFrame(
        {
            "peptide_design_code": ["AF", "TL"],
            "affibody_design_code": ["AAAA", "YYYY"],
        }
    )
    logit, score = scorer.score_codes(frame, head)
    alphabet = {value: index for index, value in enumerate(scorer.AA_ALPHABET)}
    expected = []
    for code in ("AFAAAA", "TLYYYY"):
        value = head["bias"]
        for position, residue in enumerate(code):
            value += head["weight"][position, alphabet[residue]]
        expected.append(value)
    np.testing.assert_allclose(logit, expected, rtol=0, atol=0)
    np.testing.assert_allclose(score, 1.0 / (1.0 + np.exp(-np.asarray(expected))))

    bad = tmp_path / "bad.npz"
    _write_head(bad, feature_names=list(reversed(scorer.POSITION_NAMES)))
    with pytest.raises(ValueError, match="feature order"):
        scorer.load_head(bad)


def test_scoring_requires_108_row_exact_parity_before_writing(tmp_path, monkeypatch):
    head_path = tmp_path / "head.npz"
    _write_head(head_path)
    head = scorer.load_head(head_path)
    panel = _known_panel()
    panel_path = tmp_path / "known_panel.csv"
    panel.to_csv(panel_path, index=False)
    _, reference_score = scorer.score_codes(panel, head)
    reference = panel[["pair_uid", "library"]].copy()
    reference[scorer.PARITY_SCORE_COLUMN] = reference_score
    reference["target_retention"] = "SECRET_PARITY_RETENTION"
    reference_path = tmp_path / "reference.csv"
    reference.to_csv(reference_path, index=False)

    universe_dir = tmp_path / "universe"
    universe_dir.mkdir()
    peptides = sorted(panel["peptide_design_code"].unique())
    _write_universe(universe_dir, peptides)
    output_dir = tmp_path / "scores"
    monkeypatch.setattr(scorer, "PRIVATE_ROOT", tmp_path.resolve())
    args = SimpleNamespace(
        universe_dir=universe_dir,
        head_npz=head_path,
        known_panel=panel_path,
        parity_predictions=reference_path,
        output_dir=output_dir,
    )

    scorer.run(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["model_id"] == scorer.MODEL_ID
    receipt = manifest["known_108_pair_parity"]
    assert receipt["status"] == "passed"
    assert receipt["rows"] == 108
    assert receipt["maximum_absolute_probability_difference"] <= scorer.PARITY_TOLERANCE
    assert receipt["tolerance"] == scorer.PARITY_TOLERANCE
    assert receipt["retention_outcome_columns_read"] == []
    assert manifest["rows"] == 18
    assert manifest["retention_or_binder_outcomes_read"] is False
    assert "SECRET_RETENTION" not in (output_dir / "manifest.json").read_text()
    result = pd.read_parquet(output_dir / "peptide_AF.parquet")
    assert list(result.columns) == [
        *scorer.STANDARD_SCORE_COLUMNS,
        "additive_6site_logit",
        "additive_6site_score",
    ]
    assert result["model_id"].eq(scorer.MODEL_ID).all()
    assert np.array_equal(result["model_score"], result["additive_6site_score"])
    assert len(result) == 2
    combined = pd.read_parquet(output_dir / "candidate_scores.parquet")
    assert list(combined.columns) == [
        *scorer.STANDARD_SCORE_COLUMNS,
        "additive_6site_logit",
        "additive_6site_score",
    ]
    assert len(combined) == 18
    validated = load_candidate_scores(
        scorer.MODEL_ID,
        output_dir / "candidate_scores.parquet",
        {peptide: f"SLL{peptide}ITQV" for peptide in peptides},
    )
    assert len(validated) == 18

    with pytest.raises(ValueError, match="output exists"):
        scorer.run(args)

    corrupt = reference.copy()
    corrupt.loc[0, scorer.PARITY_SCORE_COLUMN] += 1e-6
    corrupt_path = tmp_path / "corrupt_reference.csv"
    corrupt.to_csv(corrupt_path, index=False)
    failed_output = tmp_path / "failed_scores"
    bad_args = SimpleNamespace(
        universe_dir=universe_dir,
        head_npz=head_path,
        known_panel=panel_path,
        parity_predictions=corrupt_path,
        output_dir=failed_output,
    )
    with pytest.raises(ValueError, match="108-row parity failed"):
        scorer.run(bad_args)
    assert not failed_output.exists()


def test_membership_hash_is_order_independent():
    values = pd.Series(["b", "a", "c"])
    expected = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert scorer._membership_sha256(values) == expected
    assert scorer._membership_sha256(values.iloc[::-1]) == expected
