import json

import numpy as np
import pandas as pd

from downstream.AffibodyMHC import build_liba_sequence_evaluation_sidecar as sidecar


def _lock(path, candidate, members, components):
    path.write_text(
        json.dumps(
            {
                "schema_version": "affibody-weak-oof-score-lock-v1",
                "library": "LibA",
                "candidate": candidate,
                "members": members,
                "retention_labels_read": False,
                "formula": {
                    "equal_logit_weights": True,
                    "components": components,
                },
            }
        )
    )


def test_sidecar_adds_exact_locked_deployment_scores(tmp_path):
    ids = [f"opaque-{index:03d}" for index in range(108)]
    rows = []
    for model, seeds in (
        ("additive_6site", ["fixed"]),
        ("frozen_mint_layer9", ["fixed"]),
        ("frozen_mint_layer33_control", ["fixed"]),
        ("nonlinear_6site", sorted(sidecar.EXPECTED_NONLINEAR_SEEDS)),
    ):
        for seed_index, seed in enumerate(seeds):
            for row_index, row_id in enumerate(ids):
                rows.append(
                    {
                        "eval_row_id": row_id,
                        "model": model,
                        "seed": seed,
                        "score": 0.1 + 0.7 * row_index / 107 + seed_index / 100,
                    }
                )
    source = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    selected = tmp_path / "selected.lock.json"
    _lock(
        selected,
        "mean_logit__frozen_mint_layer9",
        ["frozen_mint_layer9"],
        [{"name": "frozen_mint_layer9", "weight": 1.0}],
    )
    equal = tmp_path / "equal.lock.json"
    _lock(
        equal,
        "mean_logit__additive_6site__frozen_mint_layer9",
        ["additive_6site", "frozen_mint_layer9"],
        [
            {"name": "additive_6site", "weight": 0.5},
            {"name": "frozen_mint_layer9", "weight": 0.5},
        ],
    )
    output, receipt = sidecar.build_sidecar(source, selected, equal)
    assert tuple(output.columns) == sidecar.BLINDED_PREDICTION_COLUMNS
    assert len(output) == 11 * 108
    assert output.groupby(["model", "seed"]).size().eq(108).all()
    assert receipt["retention_labels_read"] is False
    expected = {
        "nonlinear_6site_mean_logit",
        "mint_l9",
        "equal_logit_additive_mint_l9",
    }
    assert expected.issubset(set(output["model"]))
    mint = output.loc[output["model"].eq("mint_l9")].sort_values("eval_row_id")
    raw = output.loc[
        output["model"].eq("frozen_mint_layer9")
    ].sort_values("eval_row_id")
    assert np.array_equal(mint["score"].to_numpy(), raw["score"].to_numpy())
