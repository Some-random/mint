#!/usr/bin/env python3
"""Fit one frozen LibA ESMFold2 readout seed after replicated weak CV.

One process handles one ``(feature family, final seed)`` so the 25 final fits
can run on independent GPUs.  The selected epoch for each family is read from
the hash-bound replicated-CV aggregate.  This command has no retention input;
its only prediction output is the canonical 108-row opaque-ID score panel.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import (
    FrozenFeatureStore,
    MODEL_NAMES,
    model_feature_names,
    parameter_report,
)


SCHEMA_VERSION = "esmfold2-liba-final-one-v1"
COMPLETION_SCHEMA_VERSION = "esmfold2-liba-final-one-completion-v1"
AGGREGATE_SCHEMA_VERSION = "esmfold2-liba-replicated-cv-aggregate-v1"
AGGREGATE_COMPLETION_SCHEMA_VERSION = (
    "esmfold2-liba-replicated-cv-aggregate-completion-v1"
)
EXPECTED_EVALUATION_ROWS = 108
EXPECTED_EVALUATION_MEMBERSHIP_SHA256 = (
    "35622dfeeb42740e522c95fdbaadb9f04c2f2a5e26fcb09b1a1c1e6243ae68ea"
)
PREDICTION_COLUMNS = ("eval_row_id", "model", "seed", "score")
THREAD_ENV_CONTRACT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
EXPECTED_TORCH_INTEROP_THREADS = 48


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file(), f"JSON input is absent: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"JSON input must be an object: {path}")
    return payload


def _validate_record(
    record: Mapping[str, Any], path: Path, label: str
) -> None:
    _require(Path(str(record.get("path", ""))).resolve() == path.resolve(), f"{label} path changed")
    _require(record.get("sha256") == training._sha256_file(path), f"{label} checksum changed")


def _load_aggregate_selection(
    selection_path: Path,
    completion_path: Path,
    *,
    config_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    feature_cache: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    selection = _read_json(selection_path)
    completion = _read_json(completion_path)
    _require(
        selection.get("schema_version") == AGGREGATE_SCHEMA_VERSION,
        "replicated-CV selection schema changed",
    )
    _require(selection.get("retention_labels_read") is False, "selection used retention")
    _require(
        completion.get("schema_version") == AGGREGATE_COMPLETION_SCHEMA_VERSION,
        "replicated-CV completion schema changed",
    )
    _require(completion.get("retention_labels_read") is False, "CV aggregate used retention")
    _require(
        int(completion.get("completed_family_seed_runs", -1)) == 25,
        "CV aggregate does not contain 25 family/seed runs",
    )
    _require(
        int(completion.get("completed_fold_fits", -1)) == 75,
        "CV aggregate does not contain 75 fold fits",
    )
    outputs = completion.get("outputs")
    _require(isinstance(outputs, Mapping), "CV aggregate lacks output receipts")
    selection_record = outputs.get("all_model_selection")
    _require(isinstance(selection_record, Mapping), "CV aggregate lacks selection receipt")
    _validate_record(selection_record, selection_path, "all-model selection")
    inputs = completion.get("inputs")
    _require(isinstance(inputs, Mapping), "CV aggregate lacks input receipts")
    expected_inputs = {
        "config": config_path,
        "training_labels": labels_path,
        "training_labels_manifest": labels_manifest_path,
        "feature_cache_merge_manifest": feature_cache / "merge_complete.json",
    }
    for key, path in expected_inputs.items():
        record = inputs.get(key)
        _require(isinstance(record, Mapping), f"CV aggregate lacks {key} receipt")
        _validate_record(record, path, key)
    run_receipts = inputs.get("run_receipts")
    _require(isinstance(run_receipts, list), "CV aggregate lacks run receipts")
    _require(len(run_receipts) == 25, "CV aggregate does not bind 25 run receipts")
    trainer_path = Path(training.__file__).resolve()
    readout_path = Path(sys.modules[FrozenFeatureStore.__module__].__file__).resolve()
    current_trainer_sha256 = training._sha256_file(trainer_path)
    current_readout_sha256 = training._sha256_file(readout_path)
    for index, receipt in enumerate(run_receipts):
        _require(isinstance(receipt, Mapping), f"CV run receipt {index} is invalid")
        contract_record = receipt.get("execution_contract")
        _require(
            isinstance(contract_record, Mapping),
            f"CV run receipt {index} lacks its execution contract",
        )
        contract_path = Path(str(contract_record.get("path", ""))).resolve()
        _validate_record(contract_record, contract_path, f"CV run {index} contract")
        contract = _read_json(contract_path)
        _require(
            contract.get("trainer_sha256") == current_trainer_sha256,
            "trainer source changed after replicated CV",
        )
        _require(
            contract.get("readout_sha256") == current_readout_sha256,
            "readout source changed after replicated CV",
        )
        runtime = contract.get("runtime_environment")
        _require(isinstance(runtime, Mapping), "CV runtime environment is absent")
        for key, expected in THREAD_ENV_CONTRACT.items():
            _require(runtime.get(key) == expected, f"CV thread setting changed: {key}")
        _require(
            int(runtime.get("torch_intraop_threads", -1)) == 4,
            "CV PyTorch intra-op thread count changed",
        )
        _require(
            int(runtime.get("torch_interop_threads", -1))
            == EXPECTED_TORCH_INTEROP_THREADS,
            "CV PyTorch inter-op thread count changed",
        )
        _require(
            contract.get("feature_transfer")
            == {
                "host_copy_dtype": "source float16",
                "model_input_dtype": "float32",
                "float32_conversion_stage": "destination device in _model_inputs",
            },
            "CV feature-transfer contract changed",
        )
    epochs = selection.get("model_selected_epochs")
    _require(isinstance(epochs, Mapping), "selection lacks model epochs")
    _require(set(epochs) == set(MODEL_NAMES), "selection model epoch set changed")
    roles = selection.get("family_roles")
    _require(isinstance(roles, Mapping), "selection lacks family roles")
    _require(set(roles) == set(MODEL_NAMES), "selection family-role set changed")
    return selection, completion


def _validate_prediction(path: Path, model: str, seed: int) -> pd.DataFrame:
    _require(path.is_file(), "final blind prediction file is absent")
    frame = pd.read_csv(path, dtype={"eval_row_id": str, "model": str, "seed": str})
    _require(tuple(frame.columns) == PREDICTION_COLUMNS, "prediction columns changed")
    _require(len(frame) == EXPECTED_EVALUATION_ROWS, "prediction row count changed")
    _require(frame["eval_row_id"].is_unique, "prediction IDs are duplicated")
    _require(set(frame["model"]) == {model}, "prediction model changed")
    _require(set(frame["seed"]) == {str(seed)}, "prediction seed changed")
    scores = pd.to_numeric(frame["score"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(scores).all()), "prediction score is non-finite")
    _require(bool(((scores >= 0.0) & (scores <= 1.0)).all()), "prediction score is out of range")
    _require(
        training._membership_sha256(frame["eval_row_id"].astype(str).tolist())
        == EXPECTED_EVALUATION_MEMBERSHIP_SHA256,
        "prediction evaluation membership changed",
    )
    return frame


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-labels-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--aggregation-completion", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-cache-checksums", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.time()
    observed_thread_env = {
        key: os.environ.get(key) for key in THREAD_ENV_CONTRACT
    }
    _require(
        observed_thread_env == THREAD_ENV_CONTRACT,
        f"thread environment differs from locked contract: {observed_thread_env}",
    )
    _require(
        torch.get_num_threads() == 4,
        f"PyTorch intra-op thread count must be 4, found {torch.get_num_threads()}",
    )
    _require(
        torch.get_num_interop_threads() == EXPECTED_TORCH_INTEROP_THREADS,
        "PyTorch inter-op thread count must be "
        f"{EXPECTED_TORCH_INTEROP_THREADS}, found {torch.get_num_interop_threads()}",
    )
    config_path = args.config.resolve()
    labels_path = args.training_labels.resolve()
    labels_manifest_path = args.training_labels_manifest.resolve()
    feature_cache = args.feature_cache.resolve()
    selection_path = args.selection_json.resolve()
    aggregation_completion_path = args.aggregation_completion.resolve()
    config = training._read_config(config_path)
    _require(config["dataset"]["library"] == "LibA", "config is not LibA")
    _require(args.model in MODEL_NAMES, "model is not in the locked comparison")
    fixed_seeds = tuple(map(int, config["final_training"]["seeds"]))
    _require(args.seed in fixed_seeds, "seed is not in the locked final seed set")
    selection, _ = _load_aggregate_selection(
        selection_path,
        aggregation_completion_path,
        config_path=config_path,
        labels_path=labels_path,
        labels_manifest_path=labels_manifest_path,
        feature_cache=feature_cache,
    )
    selected_epoch = int(selection["model_selected_epochs"][args.model])
    _require(
        1 <= selected_epoch <= int(config["optimization"]["max_epochs"]),
        "selected epoch is out of range",
    )

    effective_config = json.loads(json.dumps(config))
    effective_config["models"] = [args.model]
    effective_config["final_training"]["seeds"] = [args.seed]
    base, fold_membership = training.load_canonical_training_labels(
        labels_path, labels_manifest_path, effective_config
    )
    _require(len(fold_membership) > 0, "canonical fold membership is empty")
    store = FrozenFeatureStore.open(
        feature_cache,
        required_features=model_feature_names(args.model),
        verify_all_checksums=bool(args.verify_cache_checksums),
    )
    training._cache_indices_for(base, store)
    training._validate_cache_split(base, store, effective_config)
    prediction_frame = training._read_prediction_row_ids(
        None,
        store,
        expected_evaluation_rows=EXPECTED_EVALUATION_ROWS,
    )
    _require(
        training._membership_sha256(prediction_frame["row_id"].astype(str).tolist())
        == EXPECTED_EVALUATION_MEMBERSHIP_SHA256,
        "label-free evaluation roster membership changed",
    )
    device = torch.device(args.device)
    training._require(
        device.type != "cuda" or torch.cuda.is_available(),
        "CUDA device requested but unavailable",
    )

    output_dir = training._validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "master_config.json")
    shutil.copy2(labels_path, output_dir / "training_labels.csv")
    shutil.copy2(labels_manifest_path, output_dir / "training_labels_manifest.json")
    for copied in (
        output_dir / "master_config.json",
        output_dir / "training_labels.csv",
        output_dir / "training_labels_manifest.json",
    ):
        os.chmod(copied, 0o600)
    contract_path = output_dir / "execution_contract.json"
    training._json_dump(
        contract_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "library": "LibA",
            "model": args.model,
            "family_role": selection["family_roles"][args.model],
            "weak_selected_feature_family": selection["selected_feature_family"],
            "seed": args.seed,
            "fixed_final_seeds": list(fixed_seeds),
            "selected_epoch_median_of_15_cv_folds": selected_epoch,
            "config_sha256": training._sha256_file(config_path),
            "training_labels_sha256": training._sha256_file(labels_path),
            "training_labels_manifest_sha256": training._sha256_file(
                labels_manifest_path
            ),
            "feature_cache_merge_manifest_sha256": training._sha256_file(
                feature_cache / "merge_complete.json"
            ),
            "all_model_selection_sha256": training._sha256_file(selection_path),
            "aggregation_completion_sha256": training._sha256_file(
                aggregation_completion_path
            ),
            "runner_sha256": training._sha256_file(Path(__file__).resolve()),
            "trainer_sha256": training._sha256_file(Path(training.__file__).resolve()),
            "readout_sha256": training._sha256_file(
                Path(sys.modules[FrozenFeatureStore.__module__].__file__).resolve()
            ),
            "runtime_environment": {
                **observed_thread_env,
                "torch_intraop_threads": int(torch.get_num_threads()),
                "torch_interop_threads": int(torch.get_num_interop_threads()),
                "thread_values_inherited_from_environment": True,
            },
            "feature_transfer": {
                "host_copy_dtype": "source float16",
                "model_input_dtype": "float32",
                "float32_conversion_stage": "destination device in _model_inputs",
            },
            "trainable_parameters": parameter_report(
                config["architecture"], [args.model]
            )[args.model],
            "weak_rows": len(base),
            "evaluation_rows": len(prediction_frame),
            "evaluation_membership_sha256": EXPECTED_EVALUATION_MEMBERSHIP_SHA256,
            "retention_labels_read": False,
        },
    )
    training.run_final_training(
        base,
        prediction_frame,
        store,
        effective_config,
        {args.model: selected_epoch},
        output_dir,
        device,
    )
    prediction_path = (
        output_dir / "blinded_predictions" / f"{args.model}__seed{args.seed}.csv"
    )
    _validate_prediction(prediction_path, args.model, args.seed)
    checkpoint_path = (
        output_dir / "final_checkpoints" / f"{args.model}__seed{args.seed}.pt"
    )
    summary_path = output_dir / "final_training_summary.json"
    _require(checkpoint_path.is_file(), "final checkpoint is absent")
    _require(summary_path.is_file(), "final training summary is absent")
    completion_path = output_dir / "final_completion.json"
    training._json_dump(
        completion_path,
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "seed": args.seed,
            "execution_contract_sha256": training._sha256_file(contract_path),
            "prediction_sha256": training._sha256_file(prediction_path),
            "checkpoint_sha256": training._sha256_file(checkpoint_path),
            "final_training_summary_sha256": training._sha256_file(summary_path),
            "runtime_seconds": float(time.time() - started),
            "retention_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
