#!/usr/bin/env python3
"""Run one locked ESMFold2 readout model for parallel cross-validation/refit.

The canonical trainer evaluates five prespecified readouts sequentially.  This
wrapper preserves its data, folds, loss, architecture, and seed contracts but
runs one named model per process so independent models can use separate GPUs.
It has no retention-data argument.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.esmfold2_libb_readout import (
    FrozenFeatureStore,
    model_feature_names,
    parameter_report,
)
from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training


SCHEMA_VERSION = "esmfold2-one-model-run-v1"
CV_COMPLETION_SCHEMA_VERSION = "esmfold2-one-model-cv-completion-v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-labels-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("cross_validate", "fit_final"), required=True)
    parser.add_argument("--selected-epochs-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-cache-checksums", action="store_true")
    return parser.parse_args(argv)


def _load_selected_epochs(
    path: Path,
    *,
    model: str,
    master_config_sha256: str,
    training_labels_sha256: str,
    training_labels_manifest_sha256: str,
    feature_cache: Path,
    feature_cache_merge_manifest_sha256: str,
) -> dict[str, int]:
    """Load an epoch choice only from this wrapper's matching CV receipt."""

    selected_path = path.resolve()
    contract_path = selected_path.parent / "execution_contract.json"
    completion_path = selected_path.parent / "cv_completion.json"
    training._require(
        contract_path.is_file(), "selected epochs lack a CV execution contract"
    )
    training._require(
        completion_path.is_file(), "selected epochs lack a CV completion receipt"
    )
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    with completion_path.open("r", encoding="utf-8") as handle:
        completion = json.load(handle)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "mode": "cross_validate",
        "master_config_sha256": master_config_sha256,
        "training_labels_sha256": training_labels_sha256,
        "training_labels_manifest_sha256": training_labels_manifest_sha256,
        "feature_cache": str(feature_cache.resolve()),
        "feature_cache_merge_manifest_sha256": feature_cache_merge_manifest_sha256,
    }
    for key, value in expected.items():
        training._require(
            contract.get(key) == value,
            f"selected-epoch CV contract mismatch for {key}",
        )
    training._require(
        completion.get("schema_version") == CV_COMPLETION_SCHEMA_VERSION,
        "selected-epoch CV completion schema mismatch",
    )
    training._require(
        completion.get("execution_contract_sha256")
        == training._sha256_file(contract_path),
        "selected-epoch execution-contract checksum mismatch",
    )
    training._require(
        completion.get("selected_epochs_sha256")
        == training._sha256_file(selected_path),
        "selected-epoch file checksum mismatch",
    )
    summary_path = selected_path.parent / "weak_validation_summary.json"
    training._require(summary_path.is_file(), "weak-validation summary is absent")
    training._require(
        completion.get("weak_validation_summary_sha256")
        == training._sha256_file(summary_path),
        "weak-validation summary checksum mismatch",
    )
    with selected_path.open("r", encoding="utf-8") as handle:
        selected = {str(key): int(value) for key, value in json.load(handle).items()}
    training._require(
        set(selected) == {model},
        "selected-epoch file must contain exactly this model",
    )
    with summary_path.open("r", encoding="utf-8") as handle:
        summaries = json.load(handle)
    matching = [record for record in summaries if record.get("model") == model]
    training._require(
        len(matching) == 1, "weak-validation summary lacks exactly one model record"
    )
    fold_epochs = [int(record["best_epoch"]) for record in matching[0]["folds"]]
    training._require(len(fold_epochs) == 3, "weak-validation fold count changed")
    expected_epoch = sorted(fold_epochs)[1]
    training._require(
        selected[model] == expected_epoch,
        "selected final epoch is not the median of the three CV folds",
    )
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = args.config.resolve()
    training_labels_path = args.training_labels.resolve()
    training_labels_manifest_path = args.training_labels_manifest.resolve()
    feature_cache = args.feature_cache.resolve()
    master_config = training._read_config(config_path)
    if args.model not in master_config["models"]:
        raise ValueError(f"model is not in the locked comparison: {args.model}")
    original_model_index = master_config["models"].index(args.model)
    config = copy.deepcopy(master_config)
    config["models"] = [args.model]
    # The canonical sequential trainer offsets each model's CV seed by its
    # position in the locked comparison.  After reducing the model list to a
    # singleton, carry that offset into the base seed explicitly so parallel
    # and sequential runs remain exactly matched.
    config["validation"]["training_seed"] = (
        int(master_config["validation"]["training_seed"])
        + 1000 * original_model_index
    )

    master_config_sha256 = training._sha256_file(config_path)
    training_labels_sha256 = training._sha256_file(training_labels_path)
    training_labels_manifest_sha256 = training._sha256_file(
        training_labels_manifest_path
    )
    feature_cache_merge_manifest_sha256 = training._sha256_file(
        feature_cache / "merge_complete.json"
    )

    base, fold_membership = training.load_canonical_training_labels(
        training_labels_path,
        training_labels_manifest_path,
        config,
    )
    store = FrozenFeatureStore.open(
        feature_cache,
        required_features=model_feature_names(args.model),
        verify_all_checksums=args.verify_cache_checksums,
    )
    training._cache_indices_for(base, store)
    training._validate_cache_split(base, store, config)

    device = torch.device(args.device)
    training._require(
        device.type != "cuda" or torch.cuda.is_available(),
        "CUDA device requested but unavailable",
    )

    selected: dict[str, int] | None = None
    selected_epochs_sha256: str | None = None
    cv_completion_sha256: str | None = None
    if args.mode == "fit_final":
        training._require(
            args.selected_epochs_json is not None,
            "fit_final requires --selected-epochs-json from this model's CV run",
        )
        selected = _load_selected_epochs(
            args.selected_epochs_json,
            model=args.model,
            master_config_sha256=master_config_sha256,
            training_labels_sha256=training_labels_sha256,
            training_labels_manifest_sha256=training_labels_manifest_sha256,
            feature_cache=feature_cache,
            feature_cache_merge_manifest_sha256=feature_cache_merge_manifest_sha256,
        )
        selected_epochs_sha256 = training._sha256_file(
            args.selected_epochs_json.resolve()
        )
        cv_completion_sha256 = training._sha256_file(
            args.selected_epochs_json.resolve().parent / "cv_completion.json"
        )
        training._require(
            1 <= selected[args.model] <= int(config["optimization"]["max_epochs"]),
            "selected epoch is out of range",
        )

    output_dir = training._validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "master_config.json")
    shutil.copy2(training_labels_path, output_dir / "training_labels.csv")
    shutil.copy2(
        training_labels_manifest_path, output_dir / "training_labels_manifest.json"
    )
    for copied in (
        output_dir / "master_config.json",
        output_dir / "training_labels.csv",
        output_dir / "training_labels_manifest.json",
    ):
        os.chmod(copied, 0o600)
    training._json_dump(
        output_dir / "execution_contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "library": config["dataset"]["library"],
            "model": args.model,
            "original_model_index": original_model_index,
            "mode": args.mode,
            "master_config_sha256": master_config_sha256,
            "training_labels_sha256": training_labels_sha256,
            "training_labels_manifest_sha256": training_labels_manifest_sha256,
            "feature_cache": str(feature_cache),
            "feature_cache_merge_manifest_sha256": feature_cache_merge_manifest_sha256,
            "effective_cv_training_seed": int(config["validation"]["training_seed"]),
            "effective_cv_fold_seeds": [
                int(config["validation"]["training_seed"]) + fold
                for fold in range(int(config["validation"]["folds"]))
            ],
            "selected_epochs_sha256": selected_epochs_sha256,
            "cv_completion_sha256": cv_completion_sha256,
            "weak_rows": int(len(base)),
            "evaluation_rows": training._evaluation_rows(config),
            "trainable_parameters": parameter_report(
                config["architecture"], [args.model]
            )[args.model],
            "retention_labels_read": False,
        },
    )

    if args.mode == "cross_validate":
        training.run_cross_validation(
            base,
            fold_membership,
            store,
            config,
            output_dir,
            device,
        )
        training._json_dump(
            output_dir / "cv_completion.json",
            {
                "schema_version": CV_COMPLETION_SCHEMA_VERSION,
                "model": args.model,
                "execution_contract_sha256": training._sha256_file(
                    output_dir / "execution_contract.json"
                ),
                "selected_epochs_sha256": training._sha256_file(
                    output_dir / "selected_epochs.json"
                ),
                "weak_validation_summary_sha256": training._sha256_file(
                    output_dir / "weak_validation_summary.json"
                ),
                "weak_validation_predictions_sha256": training._sha256_file(
                    output_dir / "weak_validation_predictions.csv.gz"
                ),
            },
        )
        return

    if selected is None:
        raise RuntimeError("validated selected epochs are unexpectedly absent")
    prediction_frame = training._read_prediction_row_ids(
        None,
        store,
        expected_evaluation_rows=training._evaluation_rows(config),
    )
    training.run_final_training(
        base,
        prediction_frame,
        store,
        config,
        selected,
        output_dir,
        device,
    )


if __name__ == "__main__":
    main()
