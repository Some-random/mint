#!/usr/bin/env python
"""Urgent, table-compatible one-epoch MINT-LoRA refit.

This is deliberately narrower than ``finetune_mint_selection_matched.py``.
It uses the exact primary weak-label pool, class weighting, full-data feature
standardizer, frozen logistic head, and fixed retention panel from the public
table, but performs no new cross-validation or epoch selection.  A single
prespecified rank-2 LoRA model is trained for exactly one epoch with seed
20260811 on every eligible weak-label row.  Retention outcomes are used only
after both frozen and LoRA predictions have been fixed.
"""

from __future__ import print_function

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path


SCHEMA_VERSION = "mint-selection-one-epoch-lora-v1"
TRAINING_SEED = 20260811
EPOCHS = 1
ARM = "lora_cross"


def validate_one_epoch_contract(args):
    """Reject command-line changes that would make the result incomparable."""
    matched._require(args.library in matched.LIBRARIES, "unknown library")
    matched._require(int(args.training_seed) == TRAINING_SEED, "training seed must be 20260811")
    matched._require(int(args.epochs) == EPOCHS, "this runner trains exactly one epoch")
    matched._require(int(args.lora_rank) == 2, "this runner requires rank-2 LoRA")
    matched._require(args.batch_size >= 1 and args.eval_batch_size >= 1, "batch sizes must be positive")
    matched._require(args.accumulation_steps >= 1, "accumulation steps must be positive")
    matched._require(0.0 <= args.warmup_fraction < 1.0, "bad warmup fraction")
    matched._require(args.weight_decay >= 0.0, "negative weight decay")
    matched._require(args.clip_norm > 0.0, "clip norm must be positive")


def _write_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def run(args):
    started = time.time()
    validate_one_epoch_contract(args)
    # ``train_live_arm`` uses this value to define and audit its one-epoch
    # learning-rate schedule.
    args.max_epochs = EPOCHS
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    matched._require(not output_dir.exists(), "output directory exists; refusing overwrite")
    matched._require(torch.cuda.is_available(), "CUDA is required for live MINT training")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    source_paths = {
        "script": Path(__file__).resolve(),
        "matched_dependency": Path(matched.__file__).resolve(),
        "cache_rows": Path(args.cache_rows),
        "cache_rows_manifest": Path(args.cache_rows_manifest),
        "retention_features": Path(args.retention_features),
        "retention_features_manifest": Path(args.retention_features).with_suffix(".manifest.json"),
        "checkpoint": Path(args.checkpoint),
        "config": Path(args.config),
        "historical_lora_dependency": REPO_ROOT / "downstream/AffibodyMHC/finetune_mint_selection.py",
        "retention_lora_dependency": REPO_ROOT / "downstream/AffibodyMHC/finetune_mint_retention.py",
        "pair_collator_dependency": REPO_ROOT / "downstream/AffibodyMHC/extract_mint_features.py",
        "mint_wrapper_dependency": REPO_ROOT / "mint/helpers/extract.py",
        "mint_esm2_dependency": REPO_ROOT / "mint/model/esm2.py",
        "mint_modules_dependency": REPO_ROOT / "mint/modules.py",
        "mint_attention_dependency": REPO_ROOT / "mint/multihead_attention.py",
        "mint_data_dependency": REPO_ROOT / "mint/data.py",
        "mint_rotary_dependency": REPO_ROOT / "mint/rotary_embedding.py",
    }
    for name, path in source_paths.items():
        matched._require(path.is_file(), "missing {} input {}".format(name, path))
    initial_hashes = {name: sha256_file(path) for name, path in source_paths.items()}

    reference = matched.validate_reference_run(args.reference_run_dir, args.library)
    inputs = matched.load_matched_inputs(args)
    for index, cache_path in enumerate(inputs["cache_paths"]):
        source_paths["cache_npz_{:03d}".format(index)] = Path(cache_path)
        if args.cache_manifest is not None and len(inputs["cache_paths"]) == 1:
            manifest_path = Path(args.cache_manifest)
        else:
            manifest_path = matched.cached_eval._manifest_for_npz(Path(cache_path))
        source_paths["cache_manifest_{:03d}".format(index)] = manifest_path
    for name in ("manifest.json", "conditions.csv", "weak_validation.csv", "metrics.csv"):
        source_paths["reference_{}".format(name.replace(".", "_"))] = Path(args.reference_run_dir) / name
    for name, path in source_paths.items():
        matched._require(path.is_file(), "missing immutable source {}".format(path))
        if name not in initial_hashes:
            initial_hashes[name] = sha256_file(path)

    primary = inputs["primary"]
    weak_features = inputs["weak_features"]
    labels = primary["weak_label"].to_numpy(dtype=int)
    retention = inputs["retention"].reset_index(drop=True)
    retention_features = inputs["retention_features"]
    selected_c = float(reference["selected_c"])
    matched._require(
        selected_c == float(matched.PRIMARY_CONTRACT[args.library]["selected_c"]),
        "reference selected C differs from canonical contract",
    )

    # This is the same all-row standardizer and logistic head used by the
    # frozen-MINT row in the public table.  No rows are downsampled.
    full_mean, full_scale = matched.fit_standardizer(weak_features)
    full_x = matched.standardize(weak_features, full_mean, full_scale)
    full_classifier = matched.fit_logistic(full_x, labels, selected_c)
    frozen_retention_x = matched.standardize(retention_features, full_mean, full_scale)
    frozen_probability = full_classifier.predict_proba(frozen_retention_x)[:, 1]

    # The live collator requires a label column.  It receives a constant dummy;
    # neither measured retention nor the 75% binder label enters inference or
    # training.  Those outcomes are consulted only after this call returns.
    final_evaluation = retention.copy()
    final_evaluation["weak_label"] = 0
    result = matched.train_live_arm(
        ARM,
        primary,
        final_evaluation,
        full_mean,
        full_scale,
        full_classifier,
        TRAINING_SEED,
        "one_epoch_full_refit",
        -1,
        EPOCHS,
        args,
        device,
        expected_epoch0_probability=frozen_probability,
        track_each_epoch=False,
    )
    lora_probability = result["predictions"]["probability"].to_numpy(dtype=float)
    matched._require(len(lora_probability) == len(retention), "LoRA prediction count changed")

    # Retention outcomes are scored only here, after both probability vectors
    # are final.  They cannot affect the fixed C, seed, epoch, or hyperparameters.
    frozen_metric = matched.retention_metric_record(
        retention, frozen_probability, args.library, "frozen_logistic", -1, 0
    )
    for metric_name, expected in reference["metrics"].items():
        matched._require(
            abs(float(frozen_metric[metric_name]) - float(expected)) <= 1e-10,
            "frozen primary {} was not reproduced".format(metric_name),
        )
    lora_metric = matched.retention_metric_record(
        retention, lora_probability, args.library, ARM, TRAINING_SEED, EPOCHS
    )
    metrics = pd.DataFrame([frozen_metric, lora_metric])

    identity_columns = [
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
    ]
    prediction_blocks = []
    for arm, seed, epoch, probability in (
        ("frozen_logistic", -1, 0, frozen_probability),
        (ARM, TRAINING_SEED, EPOCHS, lora_probability),
    ):
        block = retention[identity_columns].copy()
        block["library"] = args.library
        block["arm"] = arm
        block["training_seed"] = seed
        block["epoch"] = epoch
        block["probability"] = probability
        prediction_blocks.append(block)
    predictions = pd.concat(prediction_blocks, ignore_index=True)

    audit = pd.DataFrame(
        [
            matched.training_audit_record(
                "one_epoch_full_refit",
                ARM,
                TRAINING_SEED,
                -1,
                primary,
                full_mean,
                full_scale,
                result,
                selected_epoch=EPOCHS,
            )
        ]
    )
    per_peptide = pd.DataFrame(
        matched.per_peptide_records(retention, frozen_probability, args.library, "frozen_logistic", -1, 0)
        + matched.per_peptide_records(retention, lora_probability, args.library, ARM, TRAINING_SEED, EPOCHS)
    )
    differences = {
        "library": args.library,
        "arm": ARM,
        "training_seed": TRAINING_SEED,
    }
    for metric_name in (
        "global_auroc",
        "global_auprc",
        "global_spearman",
        "within_peptide_macro_spearman",
    ):
        differences[metric_name + "_minus_frozen"] = float(
            lora_metric[metric_name] - frozen_metric[metric_name]
        )

    matched._require(not output_dir.exists(), "output directory appeared during run")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    artifact_paths = {}
    for name, frame in (
        ("retention_metrics.csv", metrics),
        ("retention_predictions.csv", predictions),
        ("training_audit.csv", audit),
        ("per_peptide_metrics.csv", per_peptide),
        ("difference_from_frozen.csv", pd.DataFrame([differences])),
    ):
        path = output_dir / name
        _write_csv(frame, path)
        artifact_paths[name] = path

    state_path = output_dir / "model_delta_seed20260811_lora_cross.pt"
    torch.save(
        {
            "library": args.library,
            "arm": ARM,
            "training_seed": TRAINING_SEED,
            "epochs": EPOCHS,
            "selected_C": selected_c,
            "base_checkpoint_sha256": initial_hashes["checkpoint"],
            "primary_membership_sha256": matched.PRIMARY_CONTRACT[args.library]["membership_sha256"],
            "feature_mean": torch.from_numpy(np.asarray(full_mean, dtype=np.float32)),
            "feature_scale": torch.from_numpy(np.asarray(full_scale, dtype=np.float32)),
            **result["model_state"],
        },
        str(state_path),
    )
    os.chmod(str(state_path), 0o600)
    artifact_paths[state_path.name] = state_path

    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write(
            "# One-epoch MINT LoRA: {}\n\n"
            "Exact table-compatible pool: {:,} weak-label pairs; no downsampling.\n\n"
            "| Model | AUROC | AP | Global Spearman | Within-peptide Spearman |\n"
            "|---|---:|---:|---:|---:|\n"
            "| Frozen MINT | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n"
            "| MINT LoRA, one epoch | {:.4f} | {:.4f} | {:.4f} | {:.4f} |\n".format(
                args.library,
                len(primary),
                frozen_metric["global_auroc"],
                frozen_metric["global_auprc"],
                frozen_metric["global_spearman"],
                frozen_metric["within_peptide_macro_spearman"],
                lora_metric["global_auroc"],
                lora_metric["global_auprc"],
                lora_metric["global_spearman"],
                lora_metric["within_peptide_macro_spearman"],
            )
        )
    os.chmod(str(summary_path), 0o600)
    artifact_paths[summary_path.name] = summary_path

    for name, path in source_paths.items():
        matched._require(sha256_file(path) == initial_hashes[name], "{} changed during run".format(name))
    elapsed = float(time.time() - started)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "configuration": {
            "library": args.library,
            "regime": matched.REGIME,
            "cleaning": matched.CLEANING,
            "balance": matched.BALANCE,
            "primary_rows": int(len(primary)),
            "primary_positive": int(labels.sum()),
            "primary_negative": int(np.sum(labels == 0)),
            "primary_membership_sha256": matched.cached_eval.membership_sha256(primary),
            "selected_C_inherited_from_primary_weak_label_cv": selected_c,
            "training_seed": TRAINING_SEED,
            "epochs": EPOCHS,
            "epoch_selection": "none; exactly one epoch prespecified",
            "arms": ["frozen_logistic", ARM],
            "class_weights": result["class_weights"],
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "accumulation_steps": int(args.accumulation_steps),
            "head_lr": float(args.head_lr),
            "adapter_lr": float(args.adapter_lr),
            "weight_decay": float(args.weight_decay),
            "warmup_fraction": float(args.warmup_fraction),
            "clip_norm": float(args.clip_norm),
            "lora": {
                "rank": int(args.lora_rank),
                "alpha": float(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "layers_zero_based": [31, 32],
                "projections": ["q_proj", "v_proj"],
            },
            "retention_threshold_for_auroc_ap": 75.0,
            "retention_usage": {
                "partner_identities": "used to construct the library-local peptide-and-Affibody-cold pool",
                "numeric_retention_and_75_percent_labels": "final metrics only; never used for fitting or selection",
            },
        },
        "canonical_contract": matched.PRIMARY_CONTRACT[args.library],
        "reference_primary_run": reference,
        "parameter_deltas": {
            "head": result["head_parameter_delta"],
            "adapter": result["adapter_parameter_delta"],
            "trainable_parameters": int(result["trainable_count"]),
            "trainable_names": result["trainable_names"],
        },
        "epoch0_probability_max_abs_error": result["epoch0_probability_max_abs_error"],
        "retention_feature_max_abs_difference": inputs["retention_feature_max_abs_difference"],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": initial_hashes[name]}
            for name, path in source_paths.items()
        },
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "permissions": {"directory": "0700", "files": "0600"},
    }
    _write_json(manifest, output_dir / "manifest.json")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--cache-rows", type=Path, required=True)
    parser.add_argument("--cache-rows-manifest", type=Path, required=True)
    parser.add_argument("--retention-features", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--library", choices=matched.LIBRARIES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-live-feature-probability-difference", type=float, default=0.01)
    parser.add_argument("--max-retention-feature-difference", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
