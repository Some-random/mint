import numpy as np
import pandas as pd

from downstream.AffibodyMHC.evaluate_selection_weak_baseline import (
    add_identity_blocks,
    identity_blocked_indices,
    pool_counts_for_pairs,
    primary_training_frame,
    site_code_matrix,
    stable_identity_bin,
)


def _weak_row(
    pair,
    peptide,
    affibody,
    label,
    r001,
    on_design=1,
    cold=1,
    library="LibA",
):
    return {
        "library": library,
        "pep": peptide,
        "aff": affibody,
        "r001_count": r001,
        "r009_count": 1 if label else 0,
        "r010_count": 1 if label else 0,
        "pooled_r009_r010_count": 2 if label else 0,
        "weak_label": label,
        "within_declared_library_alphabet": on_design,
        "strict_retention_identity_cold_eligible": cold,
        "pair_uid": pair,
        "peptide_uid": "pep-{}".format(peptide),
        "affibody_uid": "aff-{}".format(affibody),
    }


def test_primary_filter_keeps_all_positives_and_only_count_ge_3_negatives():
    labels = pd.DataFrame(
        [
            _weak_row("positive", "AA", "AAAA", 1, 0),
            _weak_row("negative-kept", "AD", "AAAD", 0, 3),
            _weak_row("negative-low", "AE", "AAAE", 0, 2),
            _weak_row("off-design", "AF", "AAAF", 1, 0, on_design=0),
            _weak_row("not-cold", "AH", "AAAH", 0, 9, cold=0),
            _weak_row("other-library", "AI", "AAAA", 1, 0, library="LibB"),
        ]
    )

    selected = primary_training_frame(labels, "LibA", min_negative_r001_count=3)

    assert set(selected["pair_uid"]) == {"positive", "negative-kept"}
    assert set(selected["weak_label"]) == {0, 1}


def test_identity_bins_are_deterministic_and_double_cold_roles_are_disjoint():
    rows = []
    for peptide_index in range(18):
        for affibody_index in range(18):
            rows.append(
                _weak_row(
                    "pair-{}-{}".format(peptide_index, affibody_index),
                    "{:02d}".format(peptide_index),
                    "{:04d}".format(affibody_index),
                    (peptide_index + affibody_index) % 2,
                    5,
                )
            )
    frame = pd.DataFrame(rows)
    blocked = add_identity_blocks(frame, "LibA", 3)

    assert stable_identity_bin("opaque", 3, "salt") == stable_identity_bin(
        "opaque", 3, "salt"
    )
    for fold in range(3):
        indices = identity_blocked_indices(blocked, fold)
        train = blocked.iloc[indices["train"]]
        guard = blocked.iloc[indices["guard"]]
        validation = blocked.iloc[indices["validation"]]
        assert len(train) + len(guard) + len(validation) == len(blocked)
        assert set(train["pair_uid"]).isdisjoint(set(validation["pair_uid"]))
        assert set(train["peptide_uid"]).isdisjoint(set(validation["peptide_uid"]))
        assert set(train["affibody_uid"]).isdisjoint(set(validation["affibody_uid"]))
        assert validation["peptide_block"].eq(fold).all()
        assert validation["affibody_block"].eq(fold).all()


def test_site_code_matrix_uses_peptide_then_affibody_position_order():
    frame = pd.DataFrame({"pep": ["AD", "EF"], "aff": ["HIKL", "MNPS"]})

    matrix = site_code_matrix(frame, "LibA")

    assert matrix.shape == (2, 6)
    assert matrix.tolist() == [
        ["A", "D", "H", "I", "K", "L"],
        ["E", "F", "M", "N", "P", "S"],
    ]


def test_direct_pool_uses_zero_fill_sum_log1p_and_inclusive_cutoff():
    round9 = pd.DataFrame(
        {"pep": ["AA", "AD"], "aff": ["AAAA", "AAAD"], "count": [8, 2]}
    )
    round10 = pd.DataFrame(
        {"pep": ["AA", "AE"], "aff": ["AAAA", "AAAE"], "count": [4, 5]}
    )
    pairs = pd.DataFrame(
        {
            "pair_uid": ["both", "r9", "r10", "absent"],
            "pep": ["AA", "AD", "AE", "AF"],
            "aff": ["AAAA", "AAAD", "AAAE", "AAAF"],
        }
    )

    pooled = pool_counts_for_pairs(round9, round10, pairs, cutoff=5)

    assert pooled["pooled_r009_r010_count"].tolist() == [12, 2, 5, 0]
    assert pooled["pooled_r009_r010_top2_binary"].tolist() == [1, 0, 1, 0]
    assert np.allclose(
        pooled["pooled_r009_r010_log1p_count"], np.log1p([12, 2, 5, 0])
    )
