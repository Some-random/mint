import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from downstream.AffibodyMHC.libb_structural_readout import (
    FEATURE_ARCHIVE_SCHEMA_VERSION,
    FEATURE_SOURCE,
    NATIVE_LEARNED_READOUT,
    SEQUENCE_CONTROL_DIM,
    SEQUENCE_SOURCE,
    CapacityMatchedNonlinearReadout,
    FeatureView,
    MatchedInputDataset,
    NativeProjectionNonlinearReadout,
    OpaqueFeatureStore,
    canonical_json_sha256,
    build_matched_readout,
    encode_designed_codes,
    load_sequence_control_vectors,
    materialize_input_vectors,
    pool_feature_row,
    sha256_file,
    trainable_parameter_count,
)
from downstream.AffibodyMHC.train_libb_structural_readouts import (
    BLINDED_PREDICTION_COLUMNS,
    EXPECTED_EVALUATION_ROWS,
    ModelSpec,
    SELECTED_EPOCH_SCHEMA_VERSION,
    _fold_seed,
    final_seeds,
    load_selected_epochs,
    mean_logit_ensemble,
    predict,
    read_config,
    validate_blinded_output,
    validate_matched_capacity,
    validate_sequence_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_archive(root: Path) -> None:
    root.mkdir()
    metadata = root / "metadata.csv"
    metadata.write_text(
        "row_index,row_id,split\n"
        "0,train-a,train\n"
        "1,train-b,train\n"
        "2,eval-a,eval\n"
    )
    arrays = {
        "global_features": (np.arange(15, dtype=np.float32).reshape(3, 5), "feature"),
        "residue_features": (np.arange(36, dtype=np.float32).reshape(3, 3, 4), "feature"),
        "designed_mask": (
            np.asarray([[True, False, True], [False, True, True], [True, True, True]]),
            "mask",
        ),
    }
    contracts = {}
    for name, (values, kind) in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        contracts[name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "kind": kind,
        }
    rows = ["train-a", "train-b", "eval-a"]
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": FEATURE_ARCHIVE_SCHEMA_VERSION,
                "supervision_fields_read": [],
                "row_count": 3,
                "row_ids_sha256": canonical_json_sha256(rows),
                "metadata": {
                    "file": "metadata.csv",
                    "bytes": metadata.stat().st_size,
                    "sha256": sha256_file(metadata),
                    "rows": 3,
                },
                "arrays": contracts,
            }
        )
    )


def _write_sequence_records(path: Path, include_forbidden=False) -> None:
    rows = []
    for index, (row_id, split, peptide, affibody) in enumerate(
        (
            ("train-a", "train", "AAAMWAAAA", "AAAAACAAANAAAYYAAF"),
            ("train-b", "train", "AAAFRAAAA", "AAAAADAAAEAAAGGAAH"),
            ("eval-a", "eval", "AAAIWAAAA", "AAAAAFAAAGAAAIIAAK"),
        )
    ):
        record = {
            "row_index": index,
            "row_id": row_id,
            "split": split,
            "peptide_sequence": peptide,
            "affibody_sequence": affibody,
        }
        if include_forbidden:
            record["target_retention"] = 100
        rows.append(record)
    path.write_text(json.dumps({"schema_version": "test", "rows": rows}))


def _architecture():
    return {
        "adapter_dim": 32,
        "hidden_dims": [8, 4],
        "dropout": 0.1,
        "projection_seed": 17,
    }


def test_seven_position_control_is_position_specific_one_hot():
    encoded = encode_designed_codes("MW", "NNYYF")
    assert encoded.shape == (SEQUENCE_CONTROL_DIM,)
    assert encoded.sum() == 7
    matrix = encoded.reshape(7, 20)
    assert np.all(matrix.sum(axis=1) == 1)
    assert not np.array_equal(matrix[0], matrix[1])


def test_sequence_records_are_label_free_and_sequence_mapping_is_checked(tmp_path):
    path = tmp_path / "rows.json"
    _write_sequence_records(path)
    vectors = load_sequence_control_vectors(
        path,
        ("train-a", "train-b", "eval-a"),
        ("train", "train", "eval"),
    )
    assert vectors.shape == (3, 140)

    leaked = json.loads(path.read_text())
    leaked["rows"][2]["peptide_sequence"] = leaked["rows"][0]["peptide_sequence"]
    path.write_text(json.dumps(leaked))
    with pytest.raises(ValueError, match="evaluation peptide identity occurs in training"):
        load_sequence_control_vectors(
            path,
            ("train-a", "train-b", "eval-a"),
            ("train", "train", "eval"),
        )

    _write_sequence_records(path, include_forbidden=True)
    with pytest.raises(ValueError, match="forbidden supervision"):
        load_sequence_control_vectors(
            path,
            ("train-a", "train-b", "eval-a"),
            ("train", "train", "eval"),
        )


def test_sequence_contract_binds_records_and_zero_partner_overlap(tmp_path):
    del tmp_path
    root = (
        REPO_ROOT
        / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1"
    )
    records = root / "current_sequence_records.json"
    manifest_path = root / "manifest.json"
    if not records.is_file() or not manifest_path.is_file():
        pytest.skip("the private v2 sequence contract is not present")
    manifest = validate_sequence_contract(records, manifest_path)
    assert manifest["schema_version"] == "libb-fixed-crystal-contract-v2"
    assert manifest["split_audit"]["partner_overlap"] == {
        "affibody_sequence_overlap": 0,
        "peptide_sequence_overlap": 0,
        "row_id_overlap": 0,
        "sequence_pair_overlap": 0,
    }
    assert [
        value["substitution"]
        for value in manifest["structure_audit"]["assay_hla_alignment"][
            "identity_overrides"
        ]
    ] == ["Y84A", "W167A"]

def test_opaque_store_is_mmap_safe_and_rejects_supervision_metadata(tmp_path):
    root = tmp_path / "archive"
    _write_archive(root)
    store = OpaqueFeatureStore.open(
        root,
        ("global_features", "residue_features", "designed_mask"),
        verify_all_checksums=True,
    )
    assert store.row_ids == ("train-a", "train-b", "eval-a")
    assert isinstance(store.arrays["residue_features"], np.memmap)

    metadata_only = OpaqueFeatureStore.open(root, ())
    assert metadata_only.row_ids == store.row_ids
    assert metadata_only.arrays == {}

    with pytest.raises(ValueError, match="manifest.producer"):
        OpaqueFeatureStore.open(
            root,
            (),
            required_manifest={"producer": "locked-extractor"},
        )

    metadata = root / "metadata.csv"
    metadata.write_text("row_index,row_id,split,target_retention\n0,a,train,100\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["metadata"]["bytes"] = metadata.stat().st_size
    manifest["metadata"]["sha256"] = sha256_file(metadata)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly row_index,row_id,split"):
        OpaqueFeatureStore.open(root, ("global_features",))


def test_masked_pooling_uses_only_selected_residues():
    values = np.asarray([[1.0, 10.0], [1000.0, 1000.0], [3.0, 30.0]])
    mask = np.asarray([True, False, True])
    pooled = pool_feature_row(values, "mean_max", mask)
    np.testing.assert_allclose(pooled, [2.0, 20.0, 3.0, 30.0])

    with pytest.raises(ValueError, match="selects no positions"):
        pool_feature_row(values, "mean", np.zeros(3, dtype=bool))

    ordered = pool_feature_row(values, "ordered_flatten", mask)
    np.testing.assert_array_equal(ordered, [1.0, 10.0, 3.0, 30.0])


def test_feature_and_sequence_datasets_expose_only_one_input_vector(tmp_path):
    root = tmp_path / "archive"
    _write_archive(root)
    rows = tmp_path / "rows.json"
    _write_sequence_records(rows)
    store = OpaqueFeatureStore.open(
        root, ("global_features", "residue_features", "designed_mask")
    )
    sequence = load_sequence_control_vectors(rows, store.row_ids, store.splits)
    feature_dataset = MatchedInputDataset(
        store=store,
        cache_indices=[0, 2],
        row_ids=["train-a", "eval-a"],
        source=FEATURE_SOURCE,
        views=(
            FeatureView("global_features", "identity"),
            FeatureView("residue_features", "mean_max", "designed_mask"),
        ),
        labels=[1, 0],
    )
    sequence_dataset = MatchedInputDataset(
        store=store,
        cache_indices=[0, 2],
        row_ids=["train-a", "eval-a"],
        source=SEQUENCE_SOURCE,
        sequence_vectors=sequence,
        labels=[1, 0],
    )
    assert feature_dataset.input_dim == 13
    assert feature_dataset[0]["input_vector"].shape == (13,)
    assert sequence_dataset.input_dim == 140
    assert set(sequence_dataset[0]) == {"input_vector", "row_id", "label"}


def test_one_time_materialization_exactly_matches_rowwise_pooling(tmp_path):
    root = tmp_path / "archive"
    _write_archive(root)
    store = OpaqueFeatureStore.open(
        root, ("global_features", "residue_features", "designed_mask")
    )
    views = (
        FeatureView("global_features", "identity"),
        FeatureView("residue_features", "mean_max", "designed_mask"),
    )
    materialized = materialize_input_vectors(
        store, FEATURE_SOURCE, views, sequence_vectors=None
    )
    rowwise = MatchedInputDataset(
        store=store,
        cache_indices=[0, 1, 2],
        row_ids=store.row_ids,
        source=FEATURE_SOURCE,
        views=views,
    )
    expected = np.stack([rowwise[index]["input_vector"] for index in range(3)])
    np.testing.assert_array_equal(materialized, expected)


def test_every_arm_has_the_exact_same_trainable_nonlinear_head_capacity():
    architecture = {
        "adapter_dim": 256,
        "hidden_dims": [32, 16],
        "dropout": 0.1,
        "projection_seed": 20260903,
    }
    expected_model = CapacityMatchedNonlinearReadout(140, **architecture)
    expected = trainable_parameter_count(expected_model)
    config = {
        "architecture": architecture,
        "expected_trainable_parameters": expected,
        "expected_input_dimensions": {
            "nonlinear_7site_control": 140,
            "rde_designed": 256,
            "stab_global": 128,
        },
    }
    observed = validate_matched_capacity(
        {"nonlinear_7site_control": 140, "rde_designed": 256, "stab_global": 128},
        config,
    )
    assert set(observed.values()) == {expected}
    assert sum(isinstance(layer, nn.GELU) for layer in expected_model.classifier) == 2
    assert not any(parameter.requires_grad for parameter in expected_model.adapter.parameters())


def test_native_projection_learns_directly_from_each_input_dimension():
    architecture = {
        "readout_mode": NATIVE_LEARNED_READOUT,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
    }
    small = build_matched_readout(140, architecture)
    large = build_matched_readout(1050, architecture)
    assert isinstance(small, NativeProjectionNonlinearReadout)
    assert isinstance(large, NativeProjectionNonlinearReadout)
    assert small.projection.in_features == 140
    assert small.projection.out_features == 64
    assert large.projection.in_features == 1050
    assert large.projection.out_features == 64
    assert not hasattr(small, "adapter")

    loss = small(torch.randn(3, 140)).sum()
    loss.backward()
    assert small.projection.weight.grad is not None
    assert torch.count_nonzero(small.projection.weight.grad).item() > 0


def test_native_projection_uses_explicit_per_model_parameter_contracts():
    architecture = {
        "readout_mode": NATIVE_LEARNED_READOUT,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
    }
    dimensions = {"sequence": 140, "structure": 1050}
    expected = {
        name: trainable_parameter_count(build_matched_readout(size, architecture))
        for name, size in dimensions.items()
    }
    assert expected["sequence"] != expected["structure"]
    config = {
        "architecture": architecture,
        "expected_trainable_parameters": expected,
        "expected_input_dimensions": dimensions,
    }
    assert validate_matched_capacity(dimensions, config) == expected

    config["expected_trainable_parameters"] = {
        **expected,
        "structure": expected["structure"] + 1,
    }
    with pytest.raises(ValueError, match="trainable parameter count changed"):
        validate_matched_capacity(dimensions, config)


def test_structural_config_accepts_native_mode_and_rejects_legacy_adapter_keys(
    tmp_path,
):
    source = read_config(
        REPO_ROOT
        / "downstream/AffibodyMHC/configs/stab_libb_frozen_readouts_v1.json"
    )
    source.pop("protocol_config", None)
    architecture = {
        "readout_mode": NATIVE_LEARNED_READOUT,
        "input_adapter": "trainable native projection",
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "classifier": (
            "LayerNormNative-Linear64-GELU-Dropout-LayerNorm64-"
            "Linear32-GELU-Dropout-Linear1"
        ),
        "output": "one current-pair binary logit",
    }
    source["architecture"] = architecture
    source["expected_trainable_parameters"] = {
        name: trainable_parameter_count(
            build_matched_readout(int(size), architecture)
        )
        for name, size in source["expected_input_dimensions"].items()
    }
    config_path = tmp_path / "native.json"
    config_path.write_text(json.dumps(source))
    assert read_config(config_path)["architecture"]["readout_mode"] == (
        NATIVE_LEARNED_READOUT
    )

    source["architecture"]["adapter_dim"] = 1280
    config_path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="must not contain legacy adapter settings"):
        read_config(config_path)


def test_pilot_is_one_of_the_fixed_five_seeds():
    config = {
        "final_training": {
            "pilot_seed": 11,
            "seeds": [11, 12, 13, 14, 15],
        }
    }
    assert final_seeds(config, "pilot") == (11,)
    assert final_seeds(config, "full") == (11, 12, 13, 14, 15)


def test_weak_cv_seed_is_paired_across_model_arms():
    assert _fold_seed(20260829, 0) == _fold_seed(20260829, 0)
    assert _fold_seed(20260829, 0) != _fold_seed(20260829, 1)


def test_selected_epochs_are_bound_to_weak_cv_medians(tmp_path):
    base = pd.DataFrame({"row_id": ["weak-a", "weak-b"]})
    config = {"locked": "configuration"}
    summary = [
        {
            "model": "candidate",
            "folds": [{"best_epoch": 2}, {"best_epoch": 5}, {"best_epoch": 3}],
            "selected_final_epochs": 3,
        }
    ]
    summary_path = tmp_path / "weak_validation_summary.json"
    summary_path.write_text(json.dumps(summary))
    from downstream.AffibodyMHC.train_esmfold2_libb_readouts import _membership_sha256

    payload = {
        "schema_version": SELECTED_EPOCH_SCHEMA_VERSION,
        "config_sha256": canonical_json_sha256(config),
        "feature_archive_manifest_sha256": "feature-manifest-a",
        "sequence_records_sha256": "sequence-records-a",
        "sequence_contract_manifest_sha256": "sequence-contract-a",
        "weak_membership_sha256": _membership_sha256(base["row_id"]),
        "weak_validation_summary_file": summary_path.name,
        "weak_validation_summary_sha256": sha256_file(summary_path),
        "models": {
            "candidate": {
                "fold_best_epochs": [2, 5, 3],
                "selected_final_epochs": 3,
            }
        },
    }
    selected_path = tmp_path / "selected_epochs.json"
    selected_path.write_text(json.dumps(payload))
    assert load_selected_epochs(
        selected_path,
        config,
        base,
        ["candidate"],
        feature_archive_manifest_sha256="feature-manifest-a",
        sequence_records_sha256="sequence-records-a",
        sequence_contract_manifest_sha256="sequence-contract-a",
    ) == {
        "candidate": 3
    }

    with pytest.raises(ValueError, match="different feature archive"):
        load_selected_epochs(
            selected_path,
            config,
            base,
            ["candidate"],
            feature_archive_manifest_sha256="feature-manifest-b",
            sequence_records_sha256="sequence-records-a",
            sequence_contract_manifest_sha256="sequence-contract-a",
        )

    payload["models"]["candidate"]["selected_final_epochs"] = 5
    selected_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not the fold median"):
        load_selected_epochs(
            selected_path,
            config,
            base,
            ["candidate"],
            feature_archive_manifest_sha256="feature-manifest-a",
            sequence_records_sha256="sequence-records-a",
            sequence_contract_manifest_sha256="sequence-contract-a",
        )


@pytest.mark.parametrize(
    "filename,models",
    [
        (
            "stab_libb_frozen_readouts_v1.json",
            {
                "nonlinear_7site_control",
                "stab_global",
                "stab_designed_ordered",
                "stab_designed_mean_max",
                "stab_global_plus_designed_ordered",
            },
        ),
        (
            "rde_libb_frozen_readouts_v1.json",
            {
                "nonlinear_7site_control",
                "rde_context_designed_ordered",
                "rde_context_designed_mean_max",
                "rde_context_all_resolved",
                "rde_network_fold0_designed_ordered",
                "rde_network_fold1_designed_ordered",
                "rde_network_fold2_designed_ordered",
                "rde_network_fold0_all_resolved",
                "rde_network_fold1_all_resolved",
                "rde_network_fold2_all_resolved",
            },
        ),
    ],
)
def test_locked_configs_inherit_the_exact_existing_libb_protocol(filename, models):
    config = read_config(
        REPO_ROOT / "downstream/AffibodyMHC/configs" / filename
    )
    assert config["dataset"]["rows"] == 30_648
    assert config["dataset"]["positive"] == 23_725
    assert config["dataset"]["negative"] == 6_923
    assert config["validation"]["folds"] == 3
    assert config["validation"]["split_seed"] == 17
    assert config["optimization"]["loss"] == "class_weighted_bce"
    assert {value["name"] for value in config["models"]} == models


@pytest.mark.parametrize(
    "filename",
    (
        "rde_libb_native_projection_readouts_v1.json",
        "stab_libb_native_projection_readouts_v1.json",
    ),
)
def test_native_projection_configs_pin_the_direct_readout_parameter_counts(filename):
    config = read_config(
        REPO_ROOT / "downstream/AffibodyMHC/configs" / filename
    )
    assert config["architecture"]["readout_mode"] == NATIVE_LEARNED_READOUT
    assert "adapter_dim" not in config["architecture"]
    assert "projection_seed" not in config["architecture"]
    assert validate_matched_capacity(
        config["expected_input_dimensions"], config
    ) == config["expected_trainable_parameters"]


def test_locked_structural_configs_require_the_corrected_hla_contract():
    config_root = REPO_ROOT / "downstream/AffibodyMHC/configs"
    rde = read_config(config_root / "rde_libb_frozen_readouts_v1.json")
    rde_required = rde["feature_archive_contract"]["required_manifest"]
    assert rde_required["producer_schema_version"] == "rde-libb-opaque-merge-v2"
    assert rde_required["fixed_input"]["structure_contract"] == {
        "schema_version": "libb-fixed-crystal-contract-v2",
        "manifest_sha256": "d53fc4da6ec45273a07d6fa9c4a33bcde2849e9d688295f7673250738e52f51a",
        "current_sequence_records_sha256": "f69f162d07a8cbcd778ef3e7db108e03163ebbc04f7eb6b1945089c9d49153e4",
        "residue_mapping_sha256": "a3d08cfd4a4800d3cb2048dab420e50896f36534548988f282dfd184779c2258",
    }
    assert [
        value["assay_amino_acid"]
        for value in rde_required["fixed_input"]["assay_hla_identity_graft"]
    ] == ["A", "A"]

    stab = read_config(config_root / "stab_libb_frozen_readouts_v1.json")
    stab_required = stab["feature_archive_contract"]["required_manifest"]
    assert stab_required["extractor_schema_version"] == "stab-libb-current-pair-features-v4"
    assert (
        stab_required["feature_semantics"]["assay_scaffold_identity"][
            "crystal_to_assay_substitutions"
        ]
        == ["A:Y84A", "A:W167A"]
    )
    assert stab_required["provenance"]["context_mode"] == "assay_resolved_fragment"
    assert (
        stab_required["provenance"]["structure_mapping_schema"]
        == "libb-fixed-crystal-contract-v2"
    )


def test_blinded_output_is_exactly_the_sealed_120_row_schema():
    frame = pd.DataFrame(
        {
            "eval_row_id": [f"opaque-{index}" for index in range(EXPECTED_EVALUATION_ROWS)],
            "model": "rde_designed",
            "seed": "11",
            "score": np.linspace(0.0, 1.0, EXPECTED_EVALUATION_ROWS),
        },
        columns=BLINDED_PREDICTION_COLUMNS,
    )
    validate_blinded_output(frame)
    with pytest.raises(ValueError, match="columns changed"):
        validate_blinded_output(frame.assign(target_retention=100.0))


def test_rde_fold_ensemble_averages_scalar_logits_not_latent_features():
    row_ids = [f"opaque-{index}" for index in range(EXPECTED_EVALUATION_ROWS)]
    frames = []
    for model, probability in (("fold0", 0.2), ("fold1", 0.5), ("fold2", 0.8)):
        frames.append(
            pd.DataFrame(
                {
                    "eval_row_id": row_ids,
                    "model": model,
                    "seed": "11",
                    "score": probability,
                },
                columns=BLINDED_PREDICTION_COLUMNS,
            )
        )
    ensemble = mean_logit_ensemble(frames, "rde_network_ensemble", 11)
    # logit(.2), logit(.5), and logit(.8) average to zero.
    np.testing.assert_allclose(ensemble["score"], 0.5, atol=1e-7)
    assert tuple(ensemble.columns) == BLINDED_PREDICTION_COLUMNS


def test_prediction_returns_one_opaque_id_per_score(tmp_path):
    root = tmp_path / "archive"
    _write_archive(root)
    rows = tmp_path / "rows.json"
    _write_sequence_records(rows)
    store = OpaqueFeatureStore.open(root, ("global_features",))
    sequence = load_sequence_control_vectors(rows, store.row_ids, store.splits)
    dataset = MatchedInputDataset(
        store=store,
        cache_indices=[0, 1, 2],
        row_ids=store.row_ids,
        source=SEQUENCE_SOURCE,
        sequence_vectors=sequence,
    )
    model = CapacityMatchedNonlinearReadout(
        input_dim=140,
        adapter_dim=140,
        hidden_dims=(8, 4),
        dropout=0.0,
        projection_seed=3,
    )
    row_ids, scores, labels = predict(
        model, DataLoader(dataset, batch_size=2), torch.device("cpu")
    )
    assert row_ids == list(store.row_ids)
    assert len(row_ids) == len(scores) == 3
    assert labels is None


def test_training_cli_has_no_direct_measurement_argument(tmp_path):
    script = REPO_ROOT / "downstream/AffibodyMHC/train_libb_structural_readouts.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--feature-archive" in result.stdout
    assert "--sequence-records-manifest" in result.stdout
    assert "--retention" not in result.stdout
    assert "--target" not in result.stdout
