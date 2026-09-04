from __future__ import annotations

import pandas as pd

from downstream.AffibodyMHC.build_libb_provider_pooled_score_predictions import (
    MODEL_NAME,
    PREDICTION_COLUMNS,
    SEED,
    build_scores,
)


def test_build_scores_pools_rounds_and_zero_fills_absence() -> None:
    peptides = [f"P{index:02d}" for index in range(12)]
    affibodies = [f"A{index:02d}" for index in range(10)]
    identities = pd.DataFrame(
        [
            {
                "library": "LibB",
                "pep": peptide,
                "aff": affibody,
                "pair_uid": f"id-{peptide}-{affibody}",
            }
            for peptide in peptides
            for affibody in affibodies
        ]
    )
    round9 = pd.DataFrame(
        [
            {"pep": "P00", "aff": "A00", "r009_count": 352},
            {"pep": "P00", "aff": "A01", "r009_count": 7},
        ]
    )
    round10 = pd.DataFrame(
        [
            {"pep": "P00", "aff": "A00", "r010_count": 903},
            {"pep": "P00", "aff": "A02", "r010_count": 11},
        ]
    )

    predictions, components = build_scores(identities, round9, round10)

    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 120
    assert predictions["model"].eq(MODEL_NAME).all()
    assert predictions["seed"].eq(SEED).all()
    lookup = components.set_index(["peptide_design_code", "affibody_design_code"])
    assert int(lookup.loc[("P00", "A00"), "pooled_r009_r010_count"]) == 1255
    assert int(lookup.loc[("P00", "A01"), "pooled_r009_r010_count"]) == 7
    assert int(lookup.loc[("P00", "A02"), "pooled_r009_r010_count"]) == 11
    assert int(lookup.loc[("P11", "A09"), "pooled_r009_r010_count"]) == 0
