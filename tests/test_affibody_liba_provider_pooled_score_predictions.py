from __future__ import annotations

import pandas as pd

from downstream.AffibodyMHC.build_liba_provider_pooled_score_predictions import (
    MODEL_NAME,
    PREDICTION_COLUMNS,
    SEED,
    build_candidate_membership,
    build_panel_scores,
    pool_rounds,
    tie_inclusive_top_fraction,
)
from downstream.AffibodyMHC.build_sequence_table import OMITTED_LINKER
from downstream.AffibodyMHC.code_only_baseline import opaque_id


def _panel() -> pd.DataFrame:
    peptides = [f"P{index}" for index in range(9)]
    affibodies = [f"A{index:02d}" for index in range(12)]
    return pd.DataFrame(
        [
            {
                "library": "LibA",
                "pep": peptide,
                "aff": affibody,
                "pair_uid": f"id-{peptide}-{affibody}",
            }
            for peptide in peptides
            for affibody in affibodies
        ]
    )


def test_panel_scores_pool_rounds_and_zero_fill_absence() -> None:
    identities = _panel()
    round9 = pd.DataFrame(
        [
            {"pep": "P0", "aff": "A00", "r009_count": 8},
            {"pep": "P0", "aff": "A01", "r009_count": 7},
        ]
    )
    round10 = pd.DataFrame(
        [
            {"pep": "P0", "aff": "A00", "r010_count": 13},
            {"pep": "P0", "aff": "A02", "r010_count": 11},
        ]
    )
    pooled = pool_rounds(round9, round10)
    predictions, components = build_panel_scores(identities, pooled)

    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 108
    assert predictions["model"].eq(MODEL_NAME).all()
    assert predictions["seed"].eq(SEED).all()
    lookup = components.set_index(["peptide_design_code", "affibody_design_code"])
    assert int(lookup.loc[("P0", "A00"), "pooled_r009_r010_count"]) == 21
    assert int(lookup.loc[("P0", "A01"), "pooled_r009_r010_count"]) == 7
    assert int(lookup.loc[("P0", "A02"), "pooled_r009_r010_count"]) == 11
    assert int(lookup.loc[("P8", "A11"), "pooled_r009_r010_count"]) == 0


def test_tie_inclusive_top_fraction_keeps_boundary_ties() -> None:
    pooled = pd.DataFrame(
        {
            "pep": ["AA", "AA", "AA", "AA", "AA"],
            "aff": ["AAAA", "AAAD", "AAAE", "AAAF", "AAAH"],
            "r009_count": [10, 9, 9, 2, 1],
            "r010_count": [0, 0, 0, 0, 0],
            "pooled_r009_r010_count": [10, 9, 9, 2, 1],
        }
    )
    selected, cutoff, rank = tie_inclusive_top_fraction(pooled, top_fraction=0.4)

    assert rank == 2
    assert cutoff == 9
    assert len(selected) == 3
    assert selected["aff"].tolist() == ["AAAA", "AAAD", "AAAE"]


def test_candidates_exclude_measured_and_break_score_ties_by_affibody() -> None:
    chain1_template = "A" * 264 + "XX" + "A" * 4
    chain2_characters = list("A" * 58)
    for index in (12, 16, 26, 30):
        chain2_characters[index] = "X"
    template = chain1_template + OMITTED_LINKER + "".join(chain2_characters)
    identities = pd.DataFrame(
        [
            {
                "library": "LibA",
                "pep": "DE",
                "aff": "AAAA",
                "pair_uid": opaque_id("LibA", "DE", "AAAA"),
            }
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "pep": "DE",
                "aff": "AAAF",
                "r009_count": 10,
                "r010_count": 10,
                "pooled_r009_r010_count": 20,
            },
            {
                "pep": "DE",
                "aff": "AAAA",
                "r009_count": 20,
                "r010_count": 10,
                "pooled_r009_r010_count": 30,
            },
            {
                "pep": "DE",
                "aff": "AAAE",
                "r009_count": 15,
                "r010_count": 5,
                "pooled_r009_r010_count": 20,
            },
        ]
    )

    candidates, summary = build_candidate_membership(selected, identities, template)

    assert candidates["affibody_design_code"].tolist() == ["AAAE", "AAAF"]
    assert candidates["within_peptide_rank"].tolist() == [1, 2]
    assert candidates["pooled_count_rank_min"].tolist() == [1, 1]
    assert candidates["pooled_count_tie_size"].tolist() == [2, 2]
    assert candidates["peptide_full_sequence"].tolist() == ["AAADEAAAA", "AAADEAAAA"]
    assert candidates["chain2_affibody_sequence"].str.len().eq(58).all()
    assert int(summary.iloc[0]["measured_pooled_positive_pairs_removed"]) == 1
