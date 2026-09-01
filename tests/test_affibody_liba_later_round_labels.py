import hashlib

import pandas as pd

from downstream.AffibodyMHC.build_liba_later_round_labels import (
    build_arm_score_series,
    deterministic_size_match,
    fixed_negative_candidates,
    inclusive_top_fraction,
    local_identity_cold_filter,
    membership_sha256,
    size_match_rank,
)


def _round(rows):
    frame = pd.DataFrame(rows, columns=["pep", "aff", "count"])
    frame["frequency"] = frame["count"] / frame["count"].sum()
    return frame


def test_arm_scores_zero_fill_absence_and_equalize_round_depth():
    frames = {}
    for round_index in range(9, 15):
        rows = [
            ("AA", "AAAA", round_index),
            ("AD", "AAAD", 1),
        ]
        if round_index >= 11:
            rows.append(("AE", "AAAE", 2))
        frames[round_index] = _round(rows)

    scores = build_arm_score_series(frames)

    assert scores["A"].loc[("AA", "AAAA")] == 19
    assert scores["B"].loc[("AA", "AAAA")] == sum(range(9, 15))
    assert scores["D"].loc[("AA", "AAAA")] == sum(range(11, 15))
    assert ("AE", "AAAE") not in scores["A"].index
    assert scores["B"].loc[("AE", "AAAE")] == 8
    expected_ae = sum(
        0.0 if value < 11 else 2.0 / (value + 3.0) for value in range(9, 15)
    ) / 6.0
    expected_aa = sum(
        value / (value + (3 if value >= 11 else 1)) for value in range(9, 15)
    ) / 6.0
    assert scores["C"].loc[("AE", "AAAE")] == expected_ae
    assert scores["C"].loc[("AA", "AAAA")] == expected_aa


def test_inclusive_top_fraction_keeps_every_cutoff_tie():
    index = pd.MultiIndex.from_tuples(
        [("AA", "AAAA"), ("AD", "AAAA"), ("AE", "AAAA"), ("AF", "AAAA")],
        names=["pep", "aff"],
    )
    score = pd.Series([10, 5, 5, 1], index=index)

    selected, cutoff, requested, union_rows = inclusive_top_fraction(score, 0.5)

    assert union_rows == 4
    assert requested == 2
    assert cutoff == 5
    assert selected[["pep", "aff"]].values.tolist() == [
        ["AA", "AAAA"],
        ["AD", "AAAA"],
        ["AE", "AAAA"],
    ]


def test_fixed_negatives_require_r001_support_and_absence_from_all_later_rounds():
    reference = pd.DataFrame(
        {
            "pep": ["AA", "AD", "AE", "AF"],
            "aff": ["AAAA"] * 4,
            "count": [8, 3, 2, 7],
        }
    )
    later = [
        pd.DataFrame({"pep": ["AA"], "aff": ["AAAA"]}),
        pd.DataFrame({"pep": ["AE"], "aff": ["AAAA"]}),
    ]

    selected = fixed_negative_candidates(reference, later, min_r001_count=3)

    assert selected[["pep", "count"]].values.tolist() == [["AF", 7], ["AD", 3]]


def test_local_cold_filter_uses_liba_partner_identities_and_design_alphabet():
    retention = pd.DataFrame(
        {
            "library": ["LibA"] * 108,
            "peptide_design_code": ["AA"] * 108,
            "affibody_design_code": ["AAAA"] * 108,
        }
    )
    candidates = pd.DataFrame(
        {
            "pep": ["AA", "AD", "AD", "CM", "AD"],
            "aff": ["DDDD", "AAAA", "DDDD", "DDDD", "CDDD"],
        }
    )

    selected = local_identity_cold_filter(candidates, retention)

    assert selected[["pep", "aff"]].values.tolist() == [["AD", "DDDD"]]


def test_size_match_uses_exact_seed_uid_hash_and_is_input_order_invariant():
    positives = pd.DataFrame({"pair_uid": ["u4", "u2", "u1", "u3"]})
    seed = 20260811
    expected = sorted(
        positives["pair_uid"],
        key=lambda uid: hashlib.sha256(
            "size-match|{}|{}".format(seed, uid).encode("ascii")
        ).hexdigest(),
    )[:2]

    first = deterministic_size_match(positives, 2, seed)
    second = deterministic_size_match(positives.iloc[::-1], 2, seed)

    assert first["pair_uid"].tolist() == expected
    assert second["pair_uid"].tolist() == expected
    assert first["rank_sha256"].tolist() == [size_match_rank(uid, seed) for uid in expected]


def test_membership_hash_is_order_invariant_newline_delimited_sha256():
    values = ["uid-b", "uid-a"]
    expected = hashlib.sha256(b"uid-a\nuid-b").hexdigest()
    assert membership_sha256(values) == expected
    assert membership_sha256(pd.DataFrame({"pair_uid": values[::-1]})) == expected
