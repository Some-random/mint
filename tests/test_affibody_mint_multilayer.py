import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import evaluate_mint_multilayer as multilayer


def _metadata(n):
    return {
        "row_index": np.arange(n, dtype=np.int64),
        "cache_uid": np.asarray(["cache{}".format(i) for i in range(n)]),
        "source_kind": np.asarray(["weak"] * n),
        "library": np.asarray(["LibA"] * n),
        "pair_uid": np.asarray(["pair{}".format(i) for i in range(n)]),
        "chain1_sha256": np.asarray(["pep{}".format(i) for i in range(n)]),
        "chain2_sha256": np.asarray(["aff{}".format(i) for i in range(n)]),
        "sequence_pair_sha256": np.asarray(["seq{}".format(i) for i in range(n)]),
    }


def test_layer_key_and_archive_loading(tmp_path):
    assert multilayer.layer_feature_name(1) == "mint_layer_01_chain_mean"
    assert multilayer.layer_feature_name(33) == "mint_layer_33_chain_mean"
    metadata = _metadata(3)
    layer1 = np.zeros((3, multilayer.FEATURE_DIMENSION), dtype=np.float32)
    layer33 = np.ones((3, multilayer.FEATURE_DIMENSION), dtype=np.float32)
    path = tmp_path / "features.npz"
    np.savez(
        path,
        representation_layers=np.asarray([1, 33], dtype=np.int64),
        mint_layer_01_chain_mean=layer1,
        mint_layer_33_chain_mean=layer33,
        **metadata
    )

    layers, header = multilayer.load_archive_header(path)
    assert layers == (1, 33)
    assert np.array_equal(header["pair_uid"], metadata["pair_uid"])
    assert np.array_equal(multilayer.load_layer_matrix(path, 33), layer33)

    rows = pd.DataFrame(metadata)
    multilayer.validate_row_alignment(rows, header)
    rows.loc[0, "pair_uid"] = "wrong"
    with pytest.raises(ValueError, match="pair_uid mismatch"):
        multilayer.validate_row_alignment(rows, header)


def test_default_prediction_parity_bound_records_measured_numerical_reason():
    args = multilayer.build_parser().parse_args(
        ["--features", "features.npz", "--output-dir", "private_data/test-output"]
    )
    assert args.max_layer33_feature_difference == pytest.approx(1e-5)
    assert args.max_layer33_prediction_difference == pytest.approx(0.005)
    assert "0.003460" in multilayer.PREDICTION_TOLERANCE_REASON
    assert "LibB" in multilayer.PREDICTION_TOLERANCE_REASON


def test_feature_difference_reports_exact_and_nonexact_values():
    left = np.zeros((2, 3), dtype=np.float32)
    same = multilayer.feature_difference(left, left.copy(), chunk_rows=1)
    assert same["bitwise_identical"] is True
    assert same["max_abs_difference"] == 0.0

    right = left.copy()
    right[1, 2] = 3.0
    changed = multilayer.feature_difference(left, right, chunk_rows=1)
    assert changed["bitwise_identical"] is False
    assert changed["max_abs_difference"] == pytest.approx(3.0)
    assert changed["mean_abs_difference"] == pytest.approx(0.5)

    probabilities = multilayer.probability_difference(
        np.asarray([0.1, 0.2]), np.asarray([0.1, 0.25])
    )
    assert probabilities["max_abs_difference"] == pytest.approx(0.05)
    assert probabilities["bitwise_identical"] is False


def test_joint_selector_uses_log_loss_not_retention_or_descriptive_metrics():
    rows = [
        {
            "record_type": "aggregate",
            "layer": 1,
            "C": 0.1,
            "log_loss": 0.40,
            "auroc": 0.51,
            "ap": 0.52,
            "selected_joint": 0,
        },
        {
            "record_type": "aggregate",
            "layer": 33,
            "C": 0.01,
            "log_loss": 0.50,
            "auroc": 0.99,
            "ap": 0.99,
            "selected_joint": 0,
        },
    ]
    selected = multilayer.select_joint_configuration(rows)
    assert selected["layer"] == 1
    assert selected["C"] == pytest.approx(0.1)
    assert sum(row["selected_joint"] for row in rows) == 1


def test_joint_selector_has_only_deterministic_tie_break_after_log_loss():
    rows = [
        {
            "record_type": "aggregate",
            "layer": 5,
            "C": 0.1,
            "log_loss": 0.4,
            "auroc": 0.9,
            "ap": 0.9,
        },
        {
            "record_type": "aggregate",
            "layer": 1,
            "C": 1.0,
            "log_loss": 0.4,
            "auroc": 0.1,
            "ap": 0.1,
        },
    ]
    selected = multilayer.select_joint_configuration(rows)
    assert selected["layer"] == 1
    assert selected["C"] == pytest.approx(1.0)


def test_best_f1_threshold_is_explicitly_retrospective():
    result = multilayer.best_retrospective_f1(
        np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.7, 0.1])
    )
    assert result["threshold"] == pytest.approx(0.7)
    assert result["selected_pairs"] == 3
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["precision"] == pytest.approx(2.0 / 3.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(0.8)
    assert result["threshold_is_evaluation_selected"] is True
    assert result["threshold_is_calibrated_probability"] is False
    assert result["threshold_is_prospectively_validated"] is False


def test_retention_outcomes_are_parsed_and_threshold_checked_after_selection(monkeypatch):
    frame = pd.DataFrame(
        {
            "target_retention": [74.9, 75.0],
            "target_binder": [0, 1],
        }
    )
    # Use a temporary contract size/positive count so this unit test exercises
    # parsing without constructing the full private panel.
    contract = dict(
        multilayer.matched.PRIMARY_CONTRACT["LibA"],
        retention_rows=2,
        retention_positive=1,
    )
    monkeypatch.setitem(multilayer.matched.PRIMARY_CONTRACT, "LibA", contract)
    observed = multilayer.validate_retention_outcomes(frame, "LibA")
    assert observed["target_binder"].tolist() == [0, 1]
    wrong = frame.copy()
    wrong.loc[0, "target_binder"] = 1
    with pytest.raises(ValueError, match="threshold changed"):
        multilayer.validate_retention_outcomes(wrong, "LibA")


def test_cross_validation_standardizes_each_fold_and_returns_pooled_record():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(18, multilayer.FEATURE_DIMENSION)).astype(np.float32)
    labels = np.asarray([0, 1] * 9, dtype=int)
    primary = pd.DataFrame(
        {
            "weak_label": labels,
            "pair_uid": ["pair{}".format(i) for i in range(18)],
            "chain1_sha256": ["pep{}".format(i) for i in range(18)],
            "chain2_sha256": ["aff{}".format(i) for i in range(18)],
            "sequence_pair_sha256": ["seq{}".format(i) for i in range(18)],
        }
    )
    plans = []
    for fold, (train, validation) in enumerate(
        [
            (np.arange(0, 12), np.arange(12, 18)),
            (np.arange(6, 18), np.arange(0, 6)),
        ]
    ):
        plans.append(
            {
                "fold": fold,
                "train_keys": train,
                "validation_keys": validation,
                "n_guard": 0,
                "train_membership_sha256": "train{}".format(fold),
                "validation_membership_sha256": "validation{}".format(fold),
            }
        )

    rows = multilayer.cross_validate_layer(
        features, primary, plans, layer=5, c_grid=(0.001, 0.01), jobs=2
    )
    aggregate = [row for row in rows if row["record_type"] == "aggregate"]
    folds = [row for row in rows if row["record_type"] == "fold"]
    assert len(aggregate) == 2
    assert len(folds) == 4
    assert all(row["n"] == 12 for row in aggregate)
    assert sum(row["selected_within_layer"] for row in aggregate) == 1
    assert all(np.isfinite(row["log_loss"]) for row in rows)
