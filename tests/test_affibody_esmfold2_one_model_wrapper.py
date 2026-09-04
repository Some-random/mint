from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from downstream.AffibodyMHC import run_esmfold2_readout_one_model as wrapper
from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import MODEL_NAMES


CONFIG = Path(
    "downstream/AffibodyMHC/configs/esmfold2_liba_frozen_readouts_v1.json"
)


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("model_index,model", tuple(enumerate(MODEL_NAMES)))
def test_parallel_wrapper_preserves_canonical_cv_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_index: int,
    model: str,
) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("placeholder\n", encoding="utf-8")
    labels_manifest = tmp_path / "labels_manifest.json"
    _dump(labels_manifest, {})
    cache = tmp_path / "cache"
    cache.mkdir()
    _dump(cache / "merge_complete.json", {})
    output = tmp_path / "output"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        training,
        "load_canonical_training_labels",
        lambda *_args, **_kwargs: (pd.DataFrame({"row_id": ["x"]}), pd.DataFrame()),
    )
    monkeypatch.setattr(wrapper.FrozenFeatureStore, "open", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(training, "_cache_indices_for", lambda *_args: None)
    monkeypatch.setattr(training, "_validate_cache_split", lambda *_args: None)
    monkeypatch.setattr(training, "_validate_output_dir", lambda path: path.resolve())

    def fake_cv(_base, _folds, _store, config, output_dir, _device):
        captured["config"] = config
        _dump(output_dir / "selected_epochs.json", {model: 3})
        _dump(
            output_dir / "weak_validation_summary.json",
            [{"model": model, "folds": [{"best_epoch": 2}, {"best_epoch": 3}, {"best_epoch": 4}]}],
        )
        (output_dir / "weak_validation_predictions.csv.gz").write_bytes(b"predictions")
        return {model: 3}

    monkeypatch.setattr(training, "run_cross_validation", fake_cv)
    wrapper.main(
        [
            "--config",
            str(CONFIG),
            "--training-labels",
            str(labels),
            "--training-labels-manifest",
            str(labels_manifest),
            "--feature-cache",
            str(cache),
            "--model",
            model,
            "--mode",
            "cross_validate",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )

    config = captured["config"]
    expected_base_seed = 20260829 + 1000 * model_index
    assert config["models"] == [model]
    assert config["validation"]["training_seed"] == expected_base_seed
    contract = json.loads((output / "execution_contract.json").read_text())
    assert contract["original_model_index"] == model_index
    assert contract["effective_cv_fold_seeds"] == [
        expected_base_seed + fold for fold in range(3)
    ]
    completion = json.loads((output / "cv_completion.json").read_text())
    assert completion["selected_epochs_sha256"] == training._sha256_file(
        output / "selected_epochs.json"
    )


def test_selected_epochs_are_hash_bound_and_median_checked(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected_epochs.json"
    summary_path = tmp_path / "weak_validation_summary.json"
    contract_path = tmp_path / "execution_contract.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    merge = cache / "merge_complete.json"
    merge.write_text("merge", encoding="utf-8")
    _dump(selected_path, {"full": 3})
    _dump(
        summary_path,
        [{"model": "full", "folds": [{"best_epoch": 2}, {"best_epoch": 5}, {"best_epoch": 3}]}],
    )
    contract = {
        "schema_version": wrapper.SCHEMA_VERSION,
        "model": "full",
        "mode": "cross_validate",
        "master_config_sha256": "config",
        "training_labels_sha256": "labels",
        "training_labels_manifest_sha256": "manifest",
        "feature_cache": str(cache.resolve()),
        "feature_cache_merge_manifest_sha256": training._sha256_file(merge),
    }
    _dump(contract_path, contract)
    _dump(
        tmp_path / "cv_completion.json",
        {
            "schema_version": wrapper.CV_COMPLETION_SCHEMA_VERSION,
            "execution_contract_sha256": training._sha256_file(contract_path),
            "selected_epochs_sha256": training._sha256_file(selected_path),
            "weak_validation_summary_sha256": training._sha256_file(summary_path),
        },
    )
    kwargs = {
        "model": "full",
        "master_config_sha256": "config",
        "training_labels_sha256": "labels",
        "training_labels_manifest_sha256": "manifest",
        "feature_cache": cache,
        "feature_cache_merge_manifest_sha256": training._sha256_file(merge),
    }
    assert wrapper._load_selected_epochs(selected_path, **kwargs) == {"full": 3}

    _dump(selected_path, {"full": 5})
    with pytest.raises(ValueError, match="file checksum"):
        wrapper._load_selected_epochs(selected_path, **kwargs)
