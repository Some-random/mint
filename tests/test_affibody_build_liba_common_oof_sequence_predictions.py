from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "downstream/AffibodyMHC/build_liba_common_oof_sequence_predictions.py"
CONFIG = REPO_ROOT / "downstream/AffibodyMHC/configs/liba_common_oof_sequence_v1.json"
FEATURES = (
    REPO_ROOT
    / "private_data/derived/mint_multilayer_v1/merged/mint_multilayer_chain_mean_features.npz"
)
HISTORICAL = REPO_ROOT / "private_data/experiments/selection_weak_site_v2/predictions.csv"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("liba_common_oof_builder_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def production_rows(builder):
    header = builder.load_label_safe_archive_header(FEATURES)
    return builder.prepare_liba_rows(header)


def test_config_forbids_retention_and_locks_expected_counts(builder):
    config = builder.read_config(CONFIG)
    assert config["retention_labels_allowed"] is False
    assert config["dataset"] == {
        "library": "LibA",
        "weak_rows": 22_542,
        "weak_positive": 11_320,
        "weak_negative": 11_222,
        "evaluation_rows": 108,
        "weak_membership_sha256": builder.EXPECTED_TRAIN_MEMBERSHIP_SHA256,
    }


def test_six_position_encoder_and_mlp_contract(builder):
    frame = pd.DataFrame(
        {
            "peptide_design_code": ["AC", "DE"],
            "affibody_design_code": ["FGHI", "KLMN"],
        }
    )
    codes = builder.designed_codes(frame)
    encoded = builder.encode_codes(codes)
    assert codes.tolist() == [list("ACFGHI"), list("DEKLMN")]
    assert encoded.shape == (2, 120)
    np.testing.assert_array_equal(encoded.reshape(2, 6, 20).sum(axis=2), 1.0)
    model = builder.NonlinearSixSiteMLP(120, [64, 32], 0.1)
    assert builder.trainable_parameters(model) == 10_225


def test_production_pool_and_double_cold_folds(builder, production_rows):
    primary, evaluation = production_rows
    plans, membership = builder.build_fold_plans(primary)
    assert (len(primary), int(primary["weak_label"].sum()), len(evaluation)) == (
        22_542,
        11_320,
        108,
    )
    assert [tuple(map(len, (plan.train, plan.guard, plan.validation))) for plan in plans] == [
        (9_076, 10_513, 2_953),
        (10_488, 9_791, 2_263),
        (10_493, 9_750, 2_299),
    ]
    required = {
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "peptide_design_code",
        "affibody_design_code",
    }
    assert required.issubset(membership.columns)
    for plan in plans:
        train = primary.iloc[plan.train]
        validation = primary.iloc[plan.validation]
        assert set(train["chain1_sha256"]).isdisjoint(validation["chain1_sha256"])
        assert set(train["chain2_sha256"]).isdisjoint(validation["chain2_sha256"])


def test_additive_cv_and_historical_108_replay(builder, production_rows):
    primary, evaluation = production_rows
    plans, _ = builder.build_fold_plans(primary)
    train = builder.encode_codes(builder.designed_codes(primary))
    held = builder.encode_codes(builder.designed_codes(evaluation))
    result = builder.cross_validate_logistic(
        "additive_6site",
        train,
        primary,
        plans,
        (0.001, 0.01, 0.1, 1.0),
        standardize_features=False,
        tol=1e-8,
        max_iter=2000,
    )
    assert result.selected_c == 1.0
    assert len(result.oof) == 7_515
    probability, _, _ = builder.fit_final_logistic(
        train,
        held,
        primary["weak_label"].to_numpy(dtype=int),
        result.selected_c,
        standardize_features=False,
        tol=1e-8,
        max_iter=2000,
    )
    parity = builder.verify_historical_additive_parity(
        HISTORICAL, evaluation, probability, tolerance=1e-12
    )
    assert parity["rows"] == 108
    assert parity["maximum_absolute_probability_difference"] <= 1e-12
    assert parity["outcome_columns_read"] == []


def test_scorer_facing_deployment_head_schemas(builder, tmp_path):
    additive = {
        "coef": np.linspace(-1.0, 1.0, 120, dtype=np.float64),
        "intercept": np.asarray([0.25], dtype=np.float64),
    }
    additive_path = tmp_path / "additive.npz"
    builder.save_additive_deployment_head(additive_path, additive, 1.0)
    with np.load(additive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "feature_names",
            "residue_alphabet",
            "weight",
            "bias",
            "selected_C",
            "training_membership_sha256",
        }
        assert str(archive["schema_version"][0]) == "liba-additive-deployment-head-v1"
        assert tuple(archive["feature_names"].astype(str)) == builder.POSITION_NAMES
        assert archive["weight"].shape == (6, 20)

    mint = {
        "mean": np.zeros(2560, dtype=np.float64),
        "scale": np.ones(2560, dtype=np.float64),
        "coef": np.zeros(2560, dtype=np.float64),
        "intercept": np.asarray([-0.1], dtype=np.float64),
    }
    mint_path = tmp_path / "mint9.npz"
    builder.save_mint9_deployment_head(mint_path, mint, 0.01)
    with np.load(mint_path, allow_pickle=False) as archive:
        assert archive["mint_mean"].shape == (2560,)
        assert archive["mint_scale"].shape == (2560,)
        assert archive["mint_coef"].shape == (2560,)
        assert int(archive["mint_layer"][0]) == 9
        assert str(archive["mint_feature_key"][0]) == "mint_layer_09_chain_mean"
        assert str(archive["mint_pooling"][0]) == builder.POOLING_CONTRACT


def test_source_never_indexes_retention_outcomes():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        'archive["target_retention"]',
        "archive['target_retention']",
        'archive["target_binder"]',
        "archive['target_binder']",
    ):
        assert forbidden not in source


def test_layer33_export_uses_unambiguous_generic_keys(tmp_path):
    exporter_path = (
        REPO_ROOT / "downstream/AffibodyMHC/export_liba_mint_layer33_deployment_head.py"
    )
    spec = importlib.util.spec_from_file_location("liba_layer33_exporter_for_test", exporter_path)
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exporter
    spec.loader.exec_module(exporter)
    source = tmp_path / "combined.npz"
    np.savez(
        source,
        schema_version=np.asarray([exporter.SOURCE_SCHEMA]),
        mint_layer33_mean=np.zeros(2560),
        mint_layer33_scale=np.ones(2560),
        mint_layer33_coef=np.zeros(2560),
        mint_layer33_intercept=np.asarray([0.0]),
        mint_layer33_layer=np.asarray([33]),
        mint_layer33_feature_key=np.asarray([exporter.FEATURE_KEY]),
        mint_layer33_pooling=np.asarray([exporter.POOLING]),
        mint_layer33_C=np.asarray([0.01]),
    )
    old_private_root = exporter.PRIVATE_ROOT
    exporter.PRIVATE_ROOT = tmp_path
    try:
        output = tmp_path / "layer33.npz"
        receipt = exporter.export(source, output)
    finally:
        exporter.PRIVATE_ROOT = old_private_root
    assert receipt["retention_labels_read"] is False
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["schema_version"][0]) == "liba-mint-layer33-deployment-head-v1"
        assert archive["mint_mean"].shape == (2560,)
        assert archive["mint_coef"].shape == (2560,)
        assert int(archive["mint_layer"][0]) == 33
        assert float(archive["mint_C"][0]) == 0.01
