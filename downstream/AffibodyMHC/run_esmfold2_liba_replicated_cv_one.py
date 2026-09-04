#!/usr/bin/env python3
"""Run one seed of one frozen ESMFold2 LibA readout under the locked CV split.

This command is intentionally limited to the five prespecified optimization
seeds and the canonical three double-cold folds.  It reads only the weak-label
sidecar; there is no argument through which direct-retention outcomes can be
provided.  One process handles one ``(readout family, replicate seed)`` pair so
the 25 matched runs can be distributed across independent GPUs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import (
    FrozenFeatureStore,
    model_feature_names,
    parameter_report,
)


SCHEMA_VERSION = "esmfold2-liba-replicated-cv-one-v1"
COMPLETION_SCHEMA_VERSION = "esmfold2-liba-replicated-cv-completion-v1"
FIXED_REPLICATE_SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
EXPECTED_OOF_ROWS = 7515
THREAD_ENV_CONTRACT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-labels-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--replicate-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-cache-checksums", action="store_true")
    return parser.parse_args(argv)


def _sha256_membership(values: Sequence[str]) -> str:
    return training._membership_sha256(list(map(str, values)))


def _validate_predictions(
    predictions_path: Path,
    fold_membership: pd.DataFrame,
    model: str,
) -> list[dict[str, Any]]:
    predictions = pd.read_csv(predictions_path)
    expected_columns = {"model", "fold", "row_id", "weak_label", "probability"}
    training._require(
        set(predictions.columns) == expected_columns,
        "weak-validation prediction columns changed",
    )
    training._require(len(predictions) == EXPECTED_OOF_ROWS, "OOF row count changed")
    training._require(
        not bool(predictions["row_id"].duplicated().any()),
        "OOF row IDs are duplicated",
    )
    training._require(
        set(predictions["model"].astype(str)) == {model},
        "OOF prediction model changed",
    )
    records: list[dict[str, Any]] = []
    for fold in range(3):
        observed = predictions.loc[predictions["fold"].eq(fold)].copy()
        expected = fold_membership.loc[
            fold_membership["fold"].eq(fold)
            & fold_membership["role"].eq("validation"),
            ["row_id", "weak_label"],
        ].copy()
        joined = observed.merge(
            expected,
            on="row_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("", "_expected"),
        )
        training._require(
            bool(joined["_merge"].eq("both").all()),
            f"fold {fold} OOF membership changed",
        )
        training._require(
            bool(joined["weak_label"].eq(joined["weak_label_expected"]).all()),
            f"fold {fold} weak labels changed",
        )
        records.append(
            {
                "fold": fold,
                "rows": int(len(observed)),
                "positive": int(observed["weak_label"].sum()),
                "row_id_membership_sha256": _sha256_membership(
                    observed["row_id"].astype(str).tolist()
                ),
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    started = time.time()
    observed_thread_env = {
        key: os.environ.get(key) for key in THREAD_ENV_CONTRACT
    }
    training._require(
        observed_thread_env == THREAD_ENV_CONTRACT,
        f"thread environment differs from locked contract: {observed_thread_env}",
    )
    training._require(
        torch.get_num_threads() == 4,
        f"PyTorch intra-op thread count must be 4, found {torch.get_num_threads()}",
    )
    config_path = args.config.resolve()
    labels_path = args.training_labels.resolve()
    labels_manifest_path = args.training_labels_manifest.resolve()
    feature_cache = args.feature_cache.resolve()
    master_config = training._read_config(config_path)
    training._require(master_config["dataset"]["library"] == "LibA", "library is not LibA")
    training._require(
        int(args.replicate_seed) in FIXED_REPLICATE_SEEDS,
        f"replicate seed must be one of {FIXED_REPLICATE_SEEDS}",
    )
    training._require(
        args.model in master_config["models"],
        "model is not in the locked five-family comparison",
    )
    original_model_index = master_config["models"].index(args.model)

    config = copy.deepcopy(master_config)
    config["models"] = [args.model]
    # Preserve the canonical sequential model offset while replicating the
    # experiment over the five fixed base optimization seeds.
    effective_base_seed = int(args.replicate_seed) + 1000 * original_model_index
    config["validation"]["training_seed"] = effective_base_seed

    base, fold_membership = training.load_canonical_training_labels(
        labels_path, labels_manifest_path, config
    )
    store = FrozenFeatureStore.open(
        feature_cache,
        required_features=model_feature_names(args.model),
        verify_all_checksums=bool(args.verify_cache_checksums),
    )
    training._cache_indices_for(base, store)
    training._validate_cache_split(base, store, config)
    merge_manifest = feature_cache / "merge_complete.json"
    training._require(merge_manifest.is_file(), "merged feature-cache receipt is absent")

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

    runner_path = Path(__file__).resolve()
    trainer_path = Path(training.__file__).resolve()
    readout_path = Path(sys.modules[FrozenFeatureStore.__module__].__file__).resolve()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library": "LibA",
        "model": args.model,
        "original_model_index": original_model_index,
        "replicate_seed": int(args.replicate_seed),
        "fixed_replicate_seeds": list(FIXED_REPLICATE_SEEDS),
        "effective_cv_training_seed": effective_base_seed,
        "effective_cv_fold_seeds": [effective_base_seed + fold for fold in range(3)],
        "split_seed": int(config["validation"]["split_seed"]),
        "folds": int(config["validation"]["folds"]),
        "master_config_sha256": training._sha256_file(config_path),
        "training_labels_sha256": training._sha256_file(labels_path),
        "training_labels_manifest_sha256": training._sha256_file(labels_manifest_path),
        "feature_cache": str(feature_cache),
        "feature_cache_merge_manifest_sha256": training._sha256_file(merge_manifest),
        "runner_sha256": training._sha256_file(runner_path),
        "trainer_sha256": training._sha256_file(trainer_path),
        "readout_sha256": training._sha256_file(readout_path),
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
        "weak_rows": int(len(base)),
        "weak_positive": int(base["weak_label"].sum()),
        "weak_negative": int(base["weak_label"].eq(0).sum()),
        "trainable_parameters": parameter_report(
            config["architecture"], [args.model]
        )[args.model],
        "selection_labels_read": True,
        "retention_labels_read": False,
        "single_inputs_only_semantics": (
            "pre-trunk sequence-derived control, not structural evidence"
            if args.model == "single_inputs_only"
            else None
        ),
    }
    training._json_dump(output_dir / "execution_contract.json", contract)

    device = torch.device(args.device)
    training._require(
        device.type != "cuda" or torch.cuda.is_available(),
        "CUDA device requested but unavailable",
    )
    training.run_cross_validation(
        base, fold_membership, store, config, output_dir, device
    )

    predictions_path = output_dir / "weak_validation_predictions.csv.gz"
    fold_records = _validate_predictions(
        predictions_path, fold_membership, args.model
    )
    summary_path = output_dir / "weak_validation_summary.json"
    selected_path = output_dir / "selected_epochs.json"
    checkpoints = sorted((output_dir / "cv_checkpoints").glob("*.pt"))
    training._require(len(checkpoints) == 3, "expected exactly three CV checkpoints")
    training._json_dump(
        output_dir / "cv_completion.json",
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "replicate_seed": int(args.replicate_seed),
            "execution_contract_sha256": training._sha256_file(
                output_dir / "execution_contract.json"
            ),
            "selected_epochs_sha256": training._sha256_file(selected_path),
            "weak_validation_summary_sha256": training._sha256_file(summary_path),
            "weak_validation_predictions_sha256": training._sha256_file(
                predictions_path
            ),
            "checkpoint_sha256": {
                path.name: training._sha256_file(path) for path in checkpoints
            },
            "fold_prediction_memberships": fold_records,
            "runtime_seconds": float(time.time() - started),
            "retention_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
