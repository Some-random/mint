from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from downstream.AffibodyMHC.export_libb_mint_layer5_fusion_archive import (
    OUTPUT_MINT_NAME,
    OUTPUT_STRUCTURE_NAME,
    export_archive,
)
from downstream.AffibodyMHC.libb_structural_readout import (
    FEATURE_ARCHIVE_SCHEMA_VERSION,
    OpaqueFeatureStore,
    canonical_json_sha256,
    sha256_file,
    trainable_parameter_count,
    build_matched_readout,
)
from downstream.AffibodyMHC.mint_structure_fusion import (
    MINT_CHAIN_MEAN_DIM,
    MINT_LAYER5_FEATURE_NAME,
    MINT_ROW_ID_NAME,
)
from downstream.AffibodyMHC.train_libb_mint_structure_late_fusion import (
    CONFIG_PATH,
    EXPECTED_MODEL_VIEWS,
    EXPECTED_TRAINABLE_PARAMETERS,
    read_template_config,
)
from downstream.AffibodyMHC import train_libb_structural_readouts as shared_trainer
from downstream.AffibodyMHC.lock_libb_mint_structure_fusion_config import (
    lock_config,
)


def _write_metadata(root: Path, row_ids: list[str], splits: list[str]) -> tuple[Path, dict]:
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("row_index", "row_id", "split"))
        for index, (row_id, split) in enumerate(zip(row_ids, splits)):
            writer.writerow((index, row_id, split))
    contract = {
        "file": metadata.name,
        "bytes": metadata.stat().st_size,
        "sha256": sha256_file(metadata),
        "columns": ["row_index", "row_id", "split"],
        "rows": len(row_ids),
    }
    return metadata, contract


def _write_canonical_source(root: Path, row_ids: list[str], splits: list[str]):
    root.mkdir(parents=True)
    metadata, contract = _write_metadata(root, row_ids, splits)
    manifest = root / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "test-label-free-rows-v1",
                "supervision_fields_read": [],
                "row_count": len(row_ids),
                "row_ids_sha256": canonical_json_sha256(row_ids),
                "metadata": contract,
            }
        )
    )
    return metadata, manifest


def _write_mint_source(root: Path, row_ids: list[str]):
    root.mkdir(parents=True)
    archive = root / "mint_multilayer_chain_mean_features.npz"
    features = np.empty((len(row_ids), MINT_CHAIN_MEAN_DIM), dtype=np.float32)
    for index in range(len(row_ids)):
        features[index] = index + np.arange(MINT_CHAIN_MEAN_DIM, dtype=np.float32) / 10_000
    # The object array raises if code using allow_pickle=False tries to open it.
    # Successful export proves that historical outcome-like members are ignored.
    np.savez(
        archive,
        **{
            MINT_ROW_ID_NAME: np.asarray(row_ids),
            MINT_LAYER5_FEATURE_NAME: features,
            "target_retention_do_not_read": np.asarray([object()], dtype=object),
        },
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "mint-multilayer-chain-mean-cache-v1",
                "features": {
                    "by_layer": [
                        {
                            "layer": 5,
                            "name": MINT_LAYER5_FEATURE_NAME,
                            "shape": [len(row_ids), MINT_CHAIN_MEAN_DIM],
                            "dtype": "float32",
                        }
                    ]
                },
                "output": {
                    "path": str(archive),
                    "sha256": sha256_file(archive),
                    "keys": [
                        MINT_ROW_ID_NAME,
                        MINT_LAYER5_FEATURE_NAME,
                        "target_retention_do_not_read",
                    ],
                },
            }
        )
    )
    return archive, manifest, features


def _write_structural_source(root: Path, row_ids: list[str], splits: list[str]):
    root.mkdir(parents=True)
    metadata, metadata_contract = _write_metadata(root, row_ids, splits)
    values = np.arange(len(row_ids) * 4, dtype=np.float32).reshape(len(row_ids), 4)
    path = root / "native_structure.npy"
    np.save(path, values, allow_pickle=False)
    manifest = {
        "schema_version": FEATURE_ARCHIVE_SCHEMA_VERSION,
        "supervision_fields_read": [],
        "row_count": len(row_ids),
        "row_ids_sha256": canonical_json_sha256(row_ids),
        "metadata": metadata_contract,
        "arrays": {
            "native_structure": {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": "float32",
                "kind": "feature",
            }
        },
        "provenance": {
            "checkpoint_sha256": "1" * 64,
            "pdb_sha256": "2" * 64,
            "residue_mapping_sha256": "3" * 64,
            "canonical_rows_sha256": "4" * 64,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return values


def _export_kwargs(tmp_path: Path):
    canonical_ids = ["pair-c", "pair-a", "pair-b"]
    splits = ["train", "train", "eval"]
    metadata, canonical_manifest = _write_canonical_source(
        tmp_path / "canonical", canonical_ids, splits
    )
    mint_ids = ["pair-a", "extra-pair", "pair-b", "pair-c"]
    mint_archive, mint_manifest, mint_values = _write_mint_source(
        tmp_path / "mint", mint_ids
    )
    private_root = tmp_path / "private_data"
    private_root.mkdir()
    return {
        "canonical_ids": canonical_ids,
        "splits": splits,
        "mint_ids": mint_ids,
        "mint_values": mint_values,
        "private_root": private_root,
        "arguments": {
            "canonical_metadata": metadata,
            "canonical_manifest": canonical_manifest,
            "mint_archive": mint_archive,
            "mint_manifest": mint_manifest,
            "output_dir": private_root / "output",
            "verify_mint_checksum": True,
            "private_root": private_root,
            "expected_canonical_rows": 3,
            "expected_train_rows": 2,
            "expected_evaluation_rows": 1,
            "expected_mint_rows": 4,
            "expected_row_ids_sha256": canonical_json_sha256(canonical_ids),
            "chunk_rows": 2,
        },
    }


def test_export_aligns_only_pair_ids_and_layer5_features(tmp_path):
    fixture = _export_kwargs(tmp_path)
    output = export_archive(**fixture["arguments"])
    store = OpaqueFeatureStore.open(
        output, [OUTPUT_MINT_NAME], verify_all_checksums=True
    )

    assert store.row_ids == tuple(fixture["canonical_ids"])
    assert store.splits == tuple(fixture["splits"])
    expected_indices = [fixture["mint_ids"].index(value) for value in fixture["canonical_ids"]]
    np.testing.assert_array_equal(
        store.arrays[OUTPUT_MINT_NAME], fixture["mint_values"][expected_indices]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["supervision_fields_read"] == []
    assert manifest["source_fields_read"] == [MINT_ROW_ID_NAME, MINT_LAYER5_FEATURE_NAME]
    assert set(manifest["arrays"]) == {OUTPUT_MINT_NAME}
    assert not (output / "EXTRACTION_INCOMPLETE").exists()
    assert oct(os.stat(output).st_mode & 0o777) == "0o700"


def test_optional_structure_vector_creates_complete_late_fusion_archive(tmp_path):
    fixture = _export_kwargs(tmp_path)
    structure_root = tmp_path / "structure"
    expected_structure = _write_structural_source(
        structure_root, fixture["canonical_ids"], fixture["splits"]
    )
    arguments = dict(fixture["arguments"])
    arguments["output_dir"] = fixture["private_root"] / "combined"
    arguments["structural_archive"] = structure_root
    arguments["structural_array"] = "native_structure"
    arguments["verify_structural_checksums"] = True

    output = export_archive(**arguments)
    store = OpaqueFeatureStore.open(
        output,
        [OUTPUT_MINT_NAME, OUTPUT_STRUCTURE_NAME],
        verify_all_checksums=True,
    )

    np.testing.assert_array_equal(store.arrays[OUTPUT_STRUCTURE_NAME], expected_structure)
    assert store.arrays[OUTPUT_STRUCTURE_NAME].shape == (3, 4)
    assert store.arrays[OUTPUT_MINT_NAME].shape == (3, MINT_CHAIN_MEAN_DIM)


def test_locked_config_has_exactly_three_capacity_matched_frozen_arms():
    config = read_template_config(CONFIG_PATH)
    observed = tuple(
        (spec.name, spec.source, spec.views)
        for spec in shared_trainer.model_specs(config)
    )

    assert observed == EXPECTED_MODEL_VIEWS
    assert config["architecture"]["adapter_dim"] == 4096
    assert MINT_CHAIN_MEAN_DIM + 128 <= config["architecture"]["adapter_dim"]
    model = build_matched_readout(8, config["architecture"])
    assert trainable_parameter_count(model) == EXPECTED_TRAINABLE_PARAMETERS


def test_post_selection_lock_pins_vector_width_and_all_provenance(tmp_path):
    fixture = _export_kwargs(tmp_path)
    structure_root = tmp_path / "structure"
    _write_structural_source(
        structure_root, fixture["canonical_ids"], fixture["splits"]
    )
    arguments = dict(fixture["arguments"])
    arguments["output_dir"] = fixture["private_root"] / "combined"
    arguments["structural_archive"] = structure_root
    arguments["structural_array"] = "native_structure"
    fusion_archive = export_archive(**arguments)
    fusion_manifest = json.loads((fusion_archive / "manifest.json").read_text())
    weak_cv = tmp_path / "weak_cv_predictions.csv.gz"
    weak_cv.write_bytes(b"weak-validation-only")
    selection = fixture["private_root"] / "weak_selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "libb-structural-weak-selection-v1",
                "selection_metric": "pooled_weak_validation_log_loss",
                "retention_labels_read": False,
                "selected_model": "toy_structure_arm",
                "source_weak_cv_artifact_sha256": sha256_file(weak_cv),
                "source_feature_manifest_sha256": fusion_manifest["inputs"][
                    "structural_archive"
                ]["manifest_sha256"],
                "selected_vector_sha256": fusion_manifest["arrays"][
                    OUTPUT_STRUCTURE_NAME
                ]["sha256"],
            }
        )
    )
    output_config = fixture["private_root"] / "locked" / "config.json"

    lock_config(
        template_config=CONFIG_PATH,
        fusion_archive=fusion_archive,
        weak_selection_record=selection,
        output_config=output_config,
        private_root=fixture["private_root"],
        expected_canonical_rows=3,
        expected_row_ids_sha256=canonical_json_sha256(fixture["canonical_ids"]),
    )

    locked = json.loads(output_config.read_text())
    assert locked["expected_input_dimensions"] == {
        "mint_layer5_global": 2560,
        "structure_selected": 4,
        "mint_layer5_plus_structure": 2564,
    }
    assert locked["comparison"]["selected_structural_model"] == "toy_structure_arm"
    assert locked["comparison"]["template_requires_post_selection_lock"] is False
    assert locked["feature_archive_contract"]["required_manifest"] == fusion_manifest


def test_locked_trainer_cli_exposes_no_retention_input():
    script = Path(
        "downstream/AffibodyMHC/train_libb_mint_structure_late_fusion.py"
    ).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--feature-archive" in result.stdout
    assert "--training-labels" in result.stdout
    assert "retention" not in result.stdout.lower()
