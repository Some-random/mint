import hashlib

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import evaluate_mint_liba_late_rounds as late


def _fold_frame(n_peptides=12, n_affibodies=12):
    rows = []
    for peptide in range(n_peptides):
        for affibody in range(n_affibodies):
            rows.append(
                {
                    "pair_uid": "pair-{}-{}".format(peptide, affibody),
                    "sequence_pair_sha256": "seq-{}-{}".format(peptide, affibody),
                    "chain1_sha256": "pep-{}".format(peptide),
                    "chain2_sha256": "aff-{}".format(affibody),
                    "weak_label": (peptide + affibody) % 2,
                }
            )
    return pd.DataFrame(rows)


def test_canonical_protocol_constants_are_locked():
    assert late.CANONICAL_FOLDS == 3
    assert late.CANONICAL_SPLIT_SEED == 17
    assert late.CANONICAL_C_GRID == (0.001, 0.01, 0.1, 1.0)
    assert late.SIZE_MATCH_SEEDS == (20260811, 20260812, 20260813)
    args = late.parse_args(
        [
            "--delta-cache",
            "delta.npz",
            "--delta-cache-manifest",
            "delta.json",
            "--delta-prepare-manifest",
            "prepare.json",
            "--membership",
            "membership.csv",
            "--reuse-index",
            "reuse.csv",
            "--output-dir",
            "private_data/test",
        ]
    )
    assert args.folds == 3
    assert args.split_seed == 17
    assert args.c_grid == [0.001, 0.01, 0.1, 1.0]


def test_size_match_hash_is_exact_prespecified_payload():
    pair_uid = "pair-abc"
    seed = 20260811
    expected = hashlib.sha256(
        b"size-match|20260811|pair-abc"
    ).hexdigest()
    assert late._size_match_rank(seed, pair_uid) == expected


def test_condition_grid_uses_one_natural_a_reference_not_three_fake_a_subsets():
    conditions = late._expected_conditions()
    assert len(conditions) == 13
    assert (late.ARMS[0], "natural", -1) in conditions
    assert not any(
        arm == late.ARMS[0] and sampling == "size_matched"
        for arm, sampling, _ in conditions
    )


def test_finalized_delta_reuse_schema_normalizes_and_attaches_chain_hashes():
    reuse = pd.DataFrame(
        {
            "sequence_pair_sha256": ["seq-old", "seq-new"],
            "feature_source": ["old_cache", "delta_cache"],
            "old_cache_row_index": [7, -1],
            "delta_row_index": [-1, 3],
            "pair_uid": ["pair-old", "pair-new"],
            "peptide_design_code": ["AA", "CC"],
            "affibody_design_code": ["AAAA", "CCCC"],
            "chain1_sha256": ["pep-old", "pep-new"],
            "chain2_sha256": ["aff-old", "aff-new"],
        }
    )
    normalized = late._normalize_reuse_index(reuse)
    assert normalized["feature_source"].tolist() == ["old", "delta"]
    assert normalized["feature_row_index"].tolist() == [7, 3]
    membership = pd.DataFrame(
        {
            "arm": ["A", "B"],
            "sampling": ["natural", "natural"],
            "subset_seed": [-1, -1],
            "pair_uid": ["pair-old", "pair-new"],
            "peptide_design_code": ["AA", "CC"],
            "affibody_design_code": ["AAAA", "CCCC"],
            "sequence_pair_sha256": ["seq-old", "seq-new"],
            "weak_label": [0, 1],
        }
    )
    attached = late.attach_membership_identities(
        late._normalize_membership(membership), normalized
    )
    assert attached["chain1_sha256"].tolist() == ["pep-old", "pep-new"]
    assert attached["chain2_sha256"].tolist() == ["aff-old", "aff-new"]


def test_three_diagonal_folds_are_deterministic_and_double_identity_cold():
    frame = _fold_frame()
    first = late.make_diagonal_folds(frame, folds=3, split_seed=17)
    second = late.make_diagonal_folds(frame, folds=3, split_seed=17)
    assert len(first) == 3
    for left, right in zip(first, second):
        assert np.array_equal(left["train"], right["train"])
        assert np.array_equal(left["guard"], right["guard"])
        assert np.array_equal(left["validation"], right["validation"])
        roles = np.concatenate(
            [left["train"], left["guard"], left["validation"]]
        )
        assert sorted(roles.tolist()) == list(range(len(frame)))
        assert len(np.unique(roles)) == len(frame)
        train = frame.iloc[left["train"]]
        validation = frame.iloc[left["validation"]]
        assert set(train["chain1_sha256"]).isdisjoint(
            validation["chain1_sha256"]
        )
        assert set(train["chain2_sha256"]).isdisjoint(
            validation["chain2_sha256"]
        )


def test_overlap_sentinel_requires_old_delta_feature_equality():
    old = pd.DataFrame(
        {
            "row_index": [0, 1],
            "sequence_pair_sha256": ["shared", "old-only"],
        }
    )
    delta = pd.DataFrame(
        {
            "row_index": [0, 1],
            "sequence_pair_sha256": ["shared", "new"],
            "is_overlap_sentinel": [1, 0],
        }
    )
    old_features = np.zeros((2, late.FEATURE_DIMENSION), dtype=np.float32)
    delta_features = np.ones((2, late.FEATURE_DIMENSION), dtype=np.float32)
    delta_features[0] = 5e-5
    audit, maximum = late.validate_overlap_sentinels(
        old, old_features, delta, delta_features, tolerance=1e-4
    )
    assert len(audit) == 1
    assert maximum == pytest.approx(5e-5)
    delta_features[0, 0] = 2e-4
    with pytest.raises(ValueError, match="overlap sentinel differs"):
        late.validate_overlap_sentinels(
            old, old_features, delta, delta_features, tolerance=1e-4
        )


def test_tie_aware_top_choice_averages_all_exact_maxima():
    value, tied = late._tie_aware_top_choice(
        np.asarray([1, 0, 0, 1]), np.asarray([0.8, 0.8, 0.2, 0.1])
    )
    assert tied == 2
    assert value == pytest.approx(0.5)


def test_prediction_digest_is_order_invariant_but_probability_sensitive():
    frame = pd.DataFrame(
        {
            "arm": [late.ARMS[0], late.ARMS[1]],
            "sampling": ["natural", "natural"],
            "subset_seed": [-1, -1],
            "sequence_pair_sha256": ["a", "b"],
            "binder_probability": [0.2, 0.8],
        }
    )
    assert late._prediction_sha256(frame) == late._prediction_sha256(
        frame.iloc[::-1].reset_index(drop=True)
    )
    changed = frame.copy()
    changed.loc[0, "binder_probability"] = 0.21
    assert late._prediction_sha256(frame) != late._prediction_sha256(changed)


def test_arm_a_reference_probability_gate_is_identity_aligned():
    observed = pd.DataFrame(
        {
            "arm": [late.ARMS[0], late.ARMS[0]],
            "sampling": ["natural", "natural"],
            "subset_seed": [-1, -1],
            "pair_uid": ["p1", "p2"],
            "sequence_pair_sha256": ["s1", "s2"],
            "chain1_sha256": ["peptide", "peptide"],
            "binder_probability": [0.2, 0.8],
        }
    )
    reference = {
        "selected_c": 0.01,
        "predictions": pd.DataFrame(
            {
                "pair_uid": ["p2", "p1"],
                "sequence_pair_sha256": ["s2", "s1"],
                "probability": [0.8 + 1e-7, 0.2],
            }
        ),
    }
    # Patch the production panel-size invariant only for this focused fixture.
    original = late.EXPECTED_RETENTION_ROWS
    late.EXPECTED_RETENTION_ROWS = 2
    try:
        parity = late.validate_arm_a_frozen_predictions(
            observed, 0.01, reference, tolerance=1e-6
        )
        assert parity["probability_max_abs_difference"] == pytest.approx(1e-7)
        assert parity["within_peptide_rankings_exact"] is True
        with pytest.raises(ValueError, match="do not reproduce"):
            late.validate_arm_a_frozen_predictions(
                observed, 0.01, reference, tolerance=1e-8
            )
    finally:
        late.EXPECTED_RETENTION_ROWS = original
