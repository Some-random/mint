import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_mint_multilayer_parallel as aggregate


def _weak_row(library, layer, c_value, record_type, fold, log_loss):
    return {
        "library": library,
        "layer": layer,
        "representation": "mint_layer_{:02d}_chain_mean".format(layer),
        "record_type": record_type,
        "C": c_value,
        "fold": fold,
        "n": 10,
        "positive": 5,
        "log_loss": log_loss,
        "auroc": 0.5,
        "ap": 0.5,
        "n_train": 20,
        "n_guard": 2,
        "train_membership_sha256": "train",
        "validation_membership_sha256": "validation",
    }


def test_select_from_weak_uses_log_loss_then_layer_then_c_only():
    rows = []
    for library in aggregate.LIBRARIES:
        for layer in aggregate.ALL_LAYERS:
            for c_value in aggregate.C_GRID:
                loss = 1.0
                if library == "LibA" and layer == 5 and c_value == 0.1:
                    loss = 0.2
                if library == "LibB" and layer in (1, 5) and c_value == 1.0:
                    loss = 0.3
                rows.append(_weak_row(library, layer, c_value, "aggregate", -1, loss))
    marked, selected = aggregate.select_from_weak(pd.DataFrame(rows))
    assert selected["LibA"]["layer"] == 5
    assert selected["LibA"]["C"] == pytest.approx(0.1)
    assert selected["LibB"]["layer"] == 1
    assert selected["LibB"]["C"] == pytest.approx(1.0)
    assert marked["selected_joint_by_aggregate"].sum() == 2


def test_combine_weak_runs_deduplicates_identical_layer33_by_library():
    runs = {}
    input_contract = {"features": "a" * 64}
    for library in aggregate.LIBRARIES:
        for layer in aggregate.INTERMEDIATE_LAYERS:
            rows = []
            for candidate_layer in (layer, 33):
                for c_value in aggregate.C_GRID:
                    rows.append(
                        _weak_row(
                            library,
                            candidate_layer,
                            c_value,
                            "aggregate",
                            -1,
                            0.4 + candidate_layer / 1000.0 + c_value / 10000.0,
                        )
                    )
                    for fold in range(aggregate.FOLDS):
                        rows.append(
                            _weak_row(
                                library,
                                candidate_layer,
                                c_value,
                                "fold",
                                fold,
                                0.5 + fold / 100.0,
                            )
                        )
            runs[(library, layer)] = {
                "input_contract": input_contract,
                "fold_hashes": {library: "fold-{}".format(library)},
                "weak": pd.DataFrame(rows),
            }
    combined = aggregate.combine_weak_runs(runs)
    expected_rows = (
        len(aggregate.LIBRARIES)
        * len(aggregate.ALL_LAYERS)
        * len(aggregate.C_GRID)
        * (aggregate.FOLDS + 1)
    )
    assert len(combined) == expected_rows
    assert len(combined.loc[combined["layer"].eq(33)]) == (
        len(aggregate.LIBRARIES) * len(aggregate.C_GRID) * (aggregate.FOLDS + 1)
    )
