from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from downstream.AffibodyMHC import run_esmfold2_liba_final_one as final_one
from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import MODEL_NAMES


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _aggregate_fixture(tmp_path: Path):
    config = tmp_path / "config.json"
    labels = tmp_path / "labels.csv"
    labels_manifest = tmp_path / "labels_manifest.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    merge = cache / "merge_complete.json"
    config.write_text("config", encoding="utf-8")
    labels.write_text("labels", encoding="utf-8")
    labels_manifest.write_text("labels manifest", encoding="utf-8")
    merge.write_text("merge", encoding="utf-8")
    selection = tmp_path / "all_model_selection.json"
    _dump(
        selection,
        {
            "schema_version": final_one.AGGREGATE_SCHEMA_VERSION,
            "retention_labels_read": False,
            "selected_feature_family": "distogram_pair",
            "model_selected_epochs": {name: index + 1 for index, name in enumerate(MODEL_NAMES)},
            "family_roles": {
                name: ("pre_trunk_sequence_control" if name == "single_inputs_only" else "folding_derived")
                for name in MODEL_NAMES
            },
        },
    )
    completion = tmp_path / "aggregation_completion.json"
    trainer_sha256 = training._sha256_file(Path(training.__file__).resolve())
    readout_path = Path(
        sys.modules[final_one.FrozenFeatureStore.__module__].__file__
    ).resolve()
    readout_sha256 = training._sha256_file(readout_path)
    run_receipts = []
    for index in range(25):
        contract_path = tmp_path / f"cv_contract_{index}.json"
        _dump(
            contract_path,
            {
                "trainer_sha256": trainer_sha256,
                "readout_sha256": readout_sha256,
                "runtime_environment": {
                    **final_one.THREAD_ENV_CONTRACT,
                    "torch_intraop_threads": 4,
                    "torch_interop_threads": final_one.EXPECTED_TORCH_INTEROP_THREADS,
                    "thread_values_inherited_from_environment": True,
                },
                "feature_transfer": {
                    "host_copy_dtype": "source float16",
                    "model_input_dtype": "float32",
                    "float32_conversion_stage": "destination device in _model_inputs",
                },
            },
        )
        run_receipts.append(
            {
                "execution_contract": {
                    "path": str(contract_path),
                    "sha256": training._sha256_file(contract_path),
                }
            }
        )
    _dump(
        completion,
        {
            "schema_version": final_one.AGGREGATE_COMPLETION_SCHEMA_VERSION,
            "retention_labels_read": False,
            "completed_family_seed_runs": 25,
            "completed_fold_fits": 75,
            "inputs": {
                "config": {"path": str(config), "sha256": training._sha256_file(config)},
                "training_labels": {"path": str(labels), "sha256": training._sha256_file(labels)},
                "training_labels_manifest": {
                    "path": str(labels_manifest),
                    "sha256": training._sha256_file(labels_manifest),
                },
                "feature_cache_merge_manifest": {
                    "path": str(merge),
                    "sha256": training._sha256_file(merge),
                },
                "run_receipts": run_receipts,
            },
            "outputs": {
                "all_model_selection": {
                    "path": str(selection),
                    "sha256": training._sha256_file(selection),
                }
            },
        },
    )
    return config, labels, labels_manifest, cache, selection, completion


def test_loads_only_hash_bound_replicated_cv_selection(tmp_path: Path) -> None:
    config, labels, labels_manifest, cache, selection, completion = _aggregate_fixture(tmp_path)
    loaded, _ = final_one._load_aggregate_selection(
        selection,
        completion,
        config_path=config,
        labels_path=labels,
        labels_manifest_path=labels_manifest,
        feature_cache=cache,
    )
    assert loaded["selected_feature_family"] == "distogram_pair"
    assert set(loaded["model_selected_epochs"]) == set(MODEL_NAMES)


def test_rejects_tampered_selected_epochs(tmp_path: Path) -> None:
    config, labels, labels_manifest, cache, selection, completion = _aggregate_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["model_selected_epochs"]["full"] = 50
    _dump(selection, payload)
    with pytest.raises(ValueError, match="selection checksum"):
        final_one._load_aggregate_selection(
            selection,
            completion,
            config_path=config,
            labels_path=labels,
            labels_manifest_path=labels_manifest,
            feature_cache=cache,
        )
