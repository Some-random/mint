#!/usr/bin/env python
"""Audit and aggregate the six shared-epoch head-versus-LoRA artifacts.

The inputs must be exactly LibA and LibB crossed with the three prespecified
training seeds.  Each input artifact is produced by the shared-epoch trainer:
the epoch-0--3 validation trajectory is inherited from one already-audited
matched run, one common positive epoch is selected from weak labels only, and
the head-only and cross-chain-LoRA arms are refit for that same epoch.

This program fails closed before publishing output.  It checks every recorded
file hash, re-audits the matched source run, reconstructs shared-epoch
selection, validates the paired optimizer/refit audit, and recomputes all
retention metrics from row-level predictions with the canonical evaluator.
PyTorch model-delta files are deliberately verified only as opaque, nonempty,
hash-bound files; they are never deserialized.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import aggregate_mint_selection_matched as matched_aggregate
from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC import finetune_mint_selection_shared_epoch as shared_trainer
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "mint-selection-shared-epoch-aggregate-v1"
TRAINER_SCHEMA_VERSION = shared_trainer.SCHEMA_VERSION
EXPECTED_LIBRARIES = tuple(matched.LIBRARIES)
EXPECTED_TRAINING_SEEDS = tuple(matched.DEFAULT_TRAINING_SEEDS)
ARMS = tuple(matched.ARMS)
METRIC_METADATA = ("library", "arm", "training_seed", "selected_epoch")
SUMMARY_METRICS = (
    "global_auroc",
    "global_auprc",
    "global_spearman",
    "within_peptide_macro_spearman",
)
SELECTION_COLUMNS = (
    "library",
    "training_seed",
    "epoch",
    "n",
    "positive",
    "head_only_log_loss",
    "lora_cross_log_loss",
    "equal_arm_mean_log_loss",
    "selected",
    "chosen_positive_epoch",
    "epoch0_better",
    "epoch0_no_worse",
    "epoch0_minus_selected_log_loss",
)
TRAINING_AUDIT_COLUMNS = (
    "stage",
    "arm",
    "training_seed",
    "fold",
    "rows",
    "positive",
    "negative",
    "class_weight_negative",
    "class_weight_positive",
    "membership_sha256",
    "feature_mean_sha256",
    "feature_scale_sha256",
    "trainable_parameters",
    "trainable_names",
    "run_seed",
    "selected_epoch",
    "schedule_total_steps",
    "epoch0_probability_max_abs_error",
    "head_parameter_delta_l2",
    "head_parameter_delta_max_abs",
    "adapter_parameter_delta_l2",
    "adapter_parameter_delta_max_abs",
    "head_initialization_sha256",
    "lr_schedule_horizon_epochs",
    "source_run_manifest_sha256",
    "source_validation_epoch0_arm_max_abs_difference",
    "final_refit_epoch0_arm_max_abs_difference",
    "source_frozen_reproduction_max_abs_difference",
)
FIXED_OUTPUTS = {
    "c_validation.csv",
    "fold_membership.csv",
    "weak_validation_predictions.csv",
    "shared_epoch_selection.csv",
    "training_audit.csv",
    "retention_predictions.csv",
    "retention_metrics.csv",
    "paired_differences.csv",
    "per_peptide_metrics.csv",
    "run_summary.md",
}
SELECTION_DESCRIPTION = shared_trainer.EPOCH_SELECTION_DESCRIPTION
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "JSON root is not an object: {}".format(path))
    return value


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _read_output_table(run_dir, filename, run_name):
    path = Path(run_dir) / filename
    frame = pd.read_csv(path, float_precision="round_trip")
    _require(not frame.empty, "{} {} is empty".format(run_name, filename))
    return frame


def _assert_columns(frame, required, label):
    missing = set(required).difference(frame.columns)
    _require(not missing, "{} lacks columns {}".format(label, sorted(missing)))


def _assert_unique(frame, columns, label):
    _assert_columns(frame, columns, label)
    duplicated = frame.duplicated(list(columns), keep=False)
    if bool(duplicated.any()):
        example = frame.loc[duplicated, list(columns)].head(3).to_dict(orient="records")
        raise ValueError("{} has duplicate keys: {}".format(label, example))


def _normalized_json(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _same_number(observed, expected, atol=1e-12, rtol=1e-12):
    observed = float(observed)
    expected = float(expected)
    if math.isnan(observed) or math.isnan(expected):
        return math.isnan(observed) and math.isnan(expected)
    return math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol)


def _same_value(observed, expected):
    if isinstance(expected, str):
        return str(observed) == expected
    if isinstance(expected, (bool, np.bool_)):
        return bool(observed) is bool(expected)
    return _same_number(observed, expected)


def _as_bool(value, label):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ValueError("{} is not Boolean: {!r}".format(label, value))


def _is_sha256(value):
    return bool(SHA256_RE.match(str(value)))


def _expected_trainable_names(arm):
    names = {"head.weight", "head.bias"}
    if arm == "lora_cross":
        for layer in (31, 32):
            for projection in ("q_proj", "v_proj"):
                for parameter in ("lora_a", "lora_b"):
                    names.add(
                        "wrapper.model.layers.{}.multimer_attn.{}.{}".format(
                            layer, projection, parameter
                        )
                    )
    return names


def _verified_recorded_file(record, expected_path, label):
    _require(isinstance(record, dict), "{} record is malformed".format(label))
    _require(
        set(("path", "sha256")).issubset(record),
        "{} record is incomplete".format(label),
    )
    expected_lexical = Path(os.path.abspath(str(expected_path)))
    recorded_lexical = Path(os.path.abspath(str(record["path"])))
    _require(
        recorded_lexical == expected_lexical,
        "{} path differs from its run directory".format(label),
    )
    _require(
        not expected_lexical.is_symlink(),
        "{} may not be a symlink".format(label),
    )
    _require(
        expected_lexical.is_file(),
        "{} is missing: {}".format(label, expected_lexical),
    )
    expected_path = expected_lexical.resolve()
    observed_hash = sha256_file(expected_path)
    _require(observed_hash == record["sha256"], "{} hash mismatch".format(label))
    return {"path": str(expected_path), "sha256": observed_hash}


def _expected_outputs(training_seed):
    return FIXED_OUTPUTS | {
        "model_delta_seed{}_head_only.pt".format(int(training_seed)),
        "model_delta_seed{}_lora_cross.pt".format(int(training_seed)),
    }


def _validate_configuration(manifest, run_name):
    configuration = manifest.get("configuration")
    _require(isinstance(configuration, dict), "{} has no configuration".format(run_name))
    library = configuration.get("library")
    _require(library in EXPECTED_LIBRARIES, "{} has an unknown library".format(run_name))
    seeds = configuration.get("training_seeds")
    _require(
        isinstance(seeds, list) and len(seeds) == 1,
        "{} is not a one-seed artifact".format(run_name),
    )
    training_seed = int(seeds[0])
    _require(
        training_seed in EXPECTED_TRAINING_SEEDS,
        "{} has an unexpected training seed".format(run_name),
    )
    expected = {
        "regime": matched.REGIME,
        "cleaning": matched.CLEANING,
        "balance": matched.BALANCE,
        "folds": matched.CANONICAL_FOLDS,
        "split_seed": matched.CANONICAL_SPLIT_SEED,
        "c_grid": list(matched.CANONICAL_C_GRID),
        "epoch_candidates": [0, 1, 2, 3],
        "shared_positive_epoch_candidates": [1, 2, 3],
        "max_epochs": 3,
        "lr_schedule_horizon_epochs": 3,
        "arms": list(ARMS),
        "retention_threshold_for_auroc_ap": 75.0,
        "head_lr": 1e-4,
        "adapter_lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "clip_norm": 1.0,
        "primary_endpoint": "within_peptide_macro_spearman",
        "secondary_endpoints": [
            "global_spearman",
            "global_auprc",
            "global_auroc",
        ],
    }
    for key, value in expected.items():
        _require(
            _normalized_json(configuration.get(key)) == _normalized_json(value),
            "{} configuration {} violates the shared-epoch contract".format(run_name, key),
        )
    _require(
        str(configuration.get("epoch_selection")) == SELECTION_DESCRIPTION,
        "{} epoch-selection description changed".format(run_name),
    )
    for key in ("batch_size", "eval_batch_size", "accumulation_steps"):
        _require(
            int(configuration.get(key, 0)) >= 1,
            "{} configuration {} must be positive".format(run_name, key),
        )
    contract = matched.PRIMARY_CONTRACT[library]
    _require(
        float(configuration.get("selected_C")) == float(contract["selected_c"]),
        "{} selected C differs from the primary contract".format(run_name),
    )
    lora = configuration.get("lora")
    _require(isinstance(lora, dict), "{} lacks its LoRA contract".format(run_name))
    for key, value in {
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.05,
        "layers_zero_based": [31, 32],
        "projections": ["q_proj", "v_proj"],
        "placement": "multimer_attn",
    }.items():
        _require(
            _normalized_json(lora.get(key)) == _normalized_json(value),
            "{} LoRA {} differs from the matched contract".format(run_name, key),
        )
    usage = configuration.get("retention_usage")
    _require(isinstance(usage, dict), "{} lacks the retention-use boundary".format(run_name))
    _require(
        "final metrics only" in str(usage.get("numeric_retention_and_75_percent_labels", "")),
        "{} does not keep retention outcomes outside model selection".format(run_name),
    )
    _require(
        _normalized_json(manifest.get("canonical_contract")) == _normalized_json(contract),
        "{} canonical primary contract changed".format(run_name),
    )
    return library, training_seed, configuration


def _validate_shared_code_and_source_records(manifest, source, run_name):
    code = manifest.get("code")
    expected_paths = {
        "script": Path(shared_trainer.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "matched_source_verifier": Path(matched_aggregate.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    _require(isinstance(code, dict) and set(code) == set(expected_paths), "{} code provenance changed".format(run_name))
    verified = {}
    for name, path in sorted(expected_paths.items()):
        record = code[name]
        _require(isinstance(record, dict) and set(("path", "sha256")).issubset(record), "{} {} code record is malformed".format(run_name, name))
        _require(Path(record["path"]).resolve() == path, "{} {} code path changed".format(run_name, name))
        observed_hash = sha256_file(path)
        _require(observed_hash == record["sha256"], "{} {} code hash mismatch".format(run_name, name))
        verified[name] = {"path": str(path), "sha256": observed_hash}
    recorded_sources = manifest.get("source_data_model_and_code")
    _require(isinstance(recorded_sources, dict), "{} lacks inherited source records".format(run_name))
    _require(
        _normalized_json(recorded_sources) == _normalized_json(source["manifest"]["sources"]),
        "{} inherited data/model/code provenance differs from its matched source".format(run_name),
    )
    return verified


def _validate_runtime(manifest, run_name):
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "{} has no runtime record".format(run_name))
    required = {
        "gpu_name",
        "python",
        "platform",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "torch_cuda",
    }
    _require(required.issubset(runtime), "{} runtime record is incomplete".format(run_name))
    return {key: runtime[key] for key in sorted(required)}


def _validate_source_matched_run(manifest, run_dir, outputs, library, training_seed, run_name):
    """Bind one shared artifact to, and fully re-audit, its matched source run."""
    provenance = manifest.get("source_matched_run")
    _require(isinstance(provenance, dict), "{} lacks source-matched-run provenance".format(run_name))
    _require(
        set(provenance) == {
            "directory",
            "manifest_path",
            "manifest_sha256",
            "schema_version",
            "library",
            "training_seed",
            "weak_validation_path",
            "weak_validation_sha256",
        },
        "{} source-matched-run provenance schema changed".format(run_name),
    )
    source_dir = Path(provenance["directory"]).resolve()
    source_manifest_lexical = Path(
        os.path.abspath(str(provenance["manifest_path"]))
    )
    source_weak_lexical = Path(
        os.path.abspath(str(provenance["weak_validation_path"]))
    )
    _require(
        not source_manifest_lexical.is_symlink()
        and not source_weak_lexical.is_symlink(),
        "{} source manifest/validation may not be symlinks".format(run_name),
    )
    source_manifest_path = source_manifest_lexical.resolve()
    source_weak_path = source_weak_lexical.resolve()
    _require(source_dir.is_dir(), "{} source matched-run directory is missing".format(run_name))
    _require(
        provenance["schema_version"] == matched.SCHEMA_VERSION,
        "{} source matched-run schema changed".format(run_name),
    )
    _require(
        provenance["library"] == library
        and int(provenance["training_seed"]) == int(training_seed),
        "{} source matched-run identity changed".format(run_name),
    )
    _require(
        source_manifest_path == source_dir / "manifest.json",
        "{} source manifest is outside its matched-run directory".format(run_name),
    )
    _require(
        source_weak_path == source_dir / "weak_validation_predictions.csv",
        "{} source validation path is outside its matched-run directory".format(run_name),
    )
    _require(source_manifest_path.is_file(), "{} source manifest is missing".format(run_name))
    _require(source_weak_path.is_file(), "{} source validation output is missing".format(run_name))
    _require(
        sha256_file(source_manifest_path) == str(provenance["manifest_sha256"]),
        "{} source matched-run manifest hash mismatch".format(run_name),
    )
    _require(
        sha256_file(source_weak_path) == str(provenance["weak_validation_sha256"]),
        "{} source weak-validation hash mismatch".format(run_name),
    )

    source = matched_aggregate.load_verified_run(source_dir)
    _require(source["library"] == library, "{} source matched-run library differs".format(run_name))
    _require(
        int(source["training_seed"]) == int(training_seed),
        "{} source matched-run seed differs".format(run_name),
    )
    _require(
        source["manifest_sha256"] == str(provenance["manifest_sha256"]),
        "{} re-audited source manifest hash differs".format(run_name),
    )
    source_outputs = source["manifest"]["outputs"]
    _require(
        source_outputs["weak_validation_predictions.csv"]["sha256"]
        == str(provenance["weak_validation_sha256"]),
        "{} source manifest does not bind the recorded validation output".format(run_name),
    )
    for filename in (
        "c_validation.csv",
        "fold_membership.csv",
        "weak_validation_predictions.csv",
    ):
        _require(
            outputs[filename]["sha256"] == source_outputs[filename]["sha256"],
            "{} copied {} differs from the audited matched source".format(run_name, filename),
        )
    return source, _normalized_json(provenance)


def recompute_shared_epoch_selection(predictions, library, training_seed):
    """Recompute the forced-positive, equal-arm epoch choice from pooled rows."""
    label = "{} seed {} weak validation".format(library, int(training_seed))
    required = (
        "library",
        "arm",
        "training_seed",
        "fold",
        "epoch",
        "pair_uid",
        "weak_label",
        "probability",
    )
    _assert_columns(predictions, required, label)
    predictions = predictions.loc[:, required].copy()
    for column in ("training_seed", "fold", "epoch", "weak_label"):
        numeric = pd.to_numeric(predictions[column], errors="raise").to_numpy(
            dtype=float
        )
        _require(
            bool(np.isfinite(numeric).all())
            and bool(np.equal(numeric, np.floor(numeric)).all()),
            "{} {} must be integer-valued".format(label, column),
        )
        predictions[column] = numeric.astype(int)
    _require(
        set(predictions["weak_label"].tolist()) == {0, 1},
        "{} weak labels are not binary".format(label),
    )
    _assert_unique(
        predictions,
        ("arm", "training_seed", "fold", "epoch", "pair_uid"),
        label,
    )
    _require(bool(predictions["library"].eq(library).all()), "{} library changed".format(label))
    _require(
        set(predictions["arm"].astype(str)) == set(ARMS),
        "{} arms changed".format(label),
    )
    seeds = set(predictions["training_seed"].astype(int))
    _require(seeds == {int(training_seed)}, "{} training seed changed".format(label))
    probability = pd.to_numeric(predictions["probability"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(probability).all()), "{} has non-finite probabilities".format(label))
    _require(bool(((probability >= 0.0) & (probability <= 1.0)).all()), "{} probabilities leave [0,1]".format(label))

    rows = []
    reference_membership = {}
    for epoch in range(4):
        arm_metrics = {}
        n = None
        positive = None
        for arm in ARMS:
            block = predictions.loc[
                predictions["arm"].eq(arm)
                & predictions["epoch"].eq(epoch)
            ].copy()
            _require(not block.empty, "{} lacks {} epoch {}".format(label, arm, epoch))
            membership = (
                block[["fold", "pair_uid", "weak_label"]]
                .assign(
                    fold=lambda frame: pd.to_numeric(frame["fold"], errors="raise").astype(int),
                    weak_label=lambda frame: pd.to_numeric(frame["weak_label"], errors="raise").astype(int),
                )
                .sort_values(["fold", "pair_uid"], kind="mergesort")
                .reset_index(drop=True)
            )
            if epoch not in reference_membership:
                reference_membership[epoch] = membership
            else:
                _require(
                    membership.equals(reference_membership[epoch]),
                    "{} arm membership or labels differ at epoch {}".format(label, epoch),
                )
            metrics = matched.binary_metrics(
                membership["weak_label"].to_numpy(dtype=int),
                pd.to_numeric(
                    block.sort_values(["fold", "pair_uid"], kind="mergesort")["probability"],
                    errors="raise",
                ).to_numpy(dtype=float),
            )
            arm_metrics[arm] = metrics
            if n is None:
                n = int(metrics["n"])
                positive = int(metrics["positive"])
            else:
                _require(
                    n == int(metrics["n"]) and positive == int(metrics["positive"]),
                    "{} arm row counts differ at epoch {}".format(label, epoch),
                )
        head_loss = float(arm_metrics["head_only"]["log_loss"])
        cross_loss = float(arm_metrics["lora_cross"]["log_loss"])
        rows.append(
            {
                "library": library,
                "training_seed": int(training_seed),
                "epoch": int(epoch),
                "n": int(n),
                "positive": int(positive),
                "head_only_log_loss": head_loss,
                "lora_cross_log_loss": cross_loss,
                "equal_arm_mean_log_loss": 0.5 * (head_loss + cross_loss),
            }
        )
    # Validation membership must remain identical across epochs as well as arms.
    epoch0_membership = reference_membership[0]
    for epoch in range(1, 4):
        _require(
            reference_membership[epoch].equals(epoch0_membership),
            "{} validation membership changes across epochs".format(label),
        )
    chosen = min(rows[1:], key=lambda row: (row["equal_arm_mean_log_loss"], row["epoch"]))
    chosen_epoch = int(chosen["epoch"])
    epoch0_loss = float(rows[0]["equal_arm_mean_log_loss"])
    chosen_loss = float(chosen["equal_arm_mean_log_loss"])
    gap = epoch0_loss - chosen_loss
    for row in rows:
        row["selected"] = int(row["epoch"] == chosen_epoch)
        row["chosen_positive_epoch"] = chosen_epoch
        row["epoch0_better"] = bool(epoch0_loss < chosen_loss)
        row["epoch0_no_worse"] = bool(epoch0_loss <= chosen_loss)
        row["epoch0_minus_selected_log_loss"] = gap
    return chosen_epoch, pd.DataFrame(rows, columns=SELECTION_COLUMNS)


def audit_shared_epoch_selection(stored, expected, run_name):
    _require(
        set(stored.columns) == set(SELECTION_COLUMNS),
        "{} shared-epoch selection schema changed".format(run_name),
    )
    _assert_unique(stored, ("library", "training_seed", "epoch"), "{} shared epoch".format(run_name))
    _require(len(stored) == 4, "{} shared-epoch table must contain epochs 0--3".format(run_name))
    stored = stored.set_index("epoch").sort_index()
    expected = expected.set_index("epoch").sort_index()
    _require(list(stored.index.astype(int)) == [0, 1, 2, 3], "{} epoch coverage changed".format(run_name))
    for epoch in range(4):
        for column in SELECTION_COLUMNS:
            if column == "epoch":
                continue
            observed = stored.loc[epoch, column]
            wanted = expected.loc[epoch, column]
            if column in ("epoch0_better", "epoch0_no_worse"):
                agrees = _as_bool(observed, "{} {}".format(run_name, column)) == bool(wanted)
            else:
                agrees = _same_value(observed, wanted)
            _require(
                agrees,
                "{} epoch {} {} differs from recomputation".format(run_name, epoch, column),
            )
    return expected.reset_index()[list(SELECTION_COLUMNS)]


def _epoch0_arm_parity(predictions):
    numeric = predictions.copy()
    for column in ("epoch", "fold", "weak_label"):
        values = pd.to_numeric(numeric[column], errors="raise").to_numpy(dtype=float)
        _require(
            bool(np.isfinite(values).all())
            and bool(np.equal(values, np.floor(values)).all()),
            "epoch-zero validation {} must be integer-valued".format(column),
        )
        numeric[column] = values.astype(int)
    _require(
        set(numeric["weak_label"].tolist()) == {0, 1},
        "epoch-zero validation labels are not binary",
    )
    epoch0 = numeric.loc[numeric["epoch"].eq(0)].pivot(
        index=["fold", "pair_uid", "weak_label"],
        columns="arm",
        values="probability",
    )
    _require(set(epoch0.columns.astype(str)) == set(ARMS), "epoch-zero validation arms are not paired")
    parity = float(
        np.max(
            np.abs(
                pd.to_numeric(epoch0["head_only"], errors="raise").to_numpy(dtype=float)
                - pd.to_numeric(epoch0["lora_cross"], errors="raise").to_numpy(dtype=float)
            )
        )
    )
    _require(
        parity <= float(shared_trainer.EPOCH0_ARM_PARITY_ATOL),
        "head/cross epoch-zero validation parity failed",
    )
    return parity


def _validate_training_audit(audit, library, training_seed, chosen_epoch, configuration, run_name):
    _require(
        set(audit.columns) == set(TRAINING_AUDIT_COLUMNS),
        "{} training-audit schema changed".format(run_name),
    )
    _assert_unique(audit, ("stage", "arm", "training_seed", "fold"), "{} training audit".format(run_name))
    _require(len(audit) == 2, "{} training audit must contain two final refits".format(run_name))
    _require(set(audit["stage"].astype(str)) == {"final_refit"}, "{} has non-final training audit rows".format(run_name))
    _require(set(audit["arm"].astype(str)) == set(ARMS), "{} training-audit arms changed".format(run_name))
    _require(
        set(pd.to_numeric(audit["training_seed"], errors="raise").astype(int)) == {int(training_seed)},
        "{} training-audit seed changed".format(run_name),
    )
    _require(
        set(pd.to_numeric(audit["fold"], errors="raise").astype(int)) == {-1},
        "{} final-refit fold sentinel changed".format(run_name),
    )
    _require(
        set(pd.to_numeric(audit["selected_epoch"], errors="raise").astype(int)) == {int(chosen_epoch)},
        "{} final arms did not use the same chosen epoch".format(run_name),
    )
    expected_run_seed = matched.derived_seed(training_seed, "final_refit", -1)
    _require(
        set(pd.to_numeric(audit["run_seed"], errors="raise").astype(int)) == {expected_run_seed},
        "{} final arms did not use the paired run seed".format(run_name),
    )
    paired_columns = (
        "rows",
        "positive",
        "negative",
        "class_weight_negative",
        "class_weight_positive",
        "membership_sha256",
        "feature_mean_sha256",
        "feature_scale_sha256",
        "run_seed",
        "selected_epoch",
        "schedule_total_steps",
        "lr_schedule_horizon_epochs",
        "head_initialization_sha256",
        "source_run_manifest_sha256",
        "source_validation_epoch0_arm_max_abs_difference",
        "final_refit_epoch0_arm_max_abs_difference",
        "source_frozen_reproduction_max_abs_difference",
    )
    for column in paired_columns:
        _require(
            audit[column].nunique(dropna=False) == 1,
            "{} paired final arms differ in {}".format(run_name, column),
        )
    contract = matched.PRIMARY_CONTRACT[library]
    first = audit.iloc[0]
    _require(
        int(first["rows"]) == int(contract["rows"])
        and int(first["positive"]) == int(contract["positive"])
        and int(first["negative"]) == int(contract["negative"]),
        "{} final-refit class counts changed".format(run_name),
    )
    _require(
        str(first["membership_sha256"]) == str(contract["membership_sha256"]),
        "{} final-refit membership changed".format(run_name),
    )
    weights = matched.balanced_class_weights(
        np.concatenate(
            [
                np.ones(int(contract["positive"]), dtype=int),
                np.zeros(int(contract["negative"]), dtype=int),
            ]
        )
    )
    _require(
        _same_number(first["class_weight_negative"], weights[0])
        and _same_number(first["class_weight_positive"], weights[1]),
        "{} final-refit class weights changed".format(run_name),
    )
    _require(
        _is_sha256(first["head_initialization_sha256"]),
        "{} head-initialization hash is malformed".format(run_name),
    )
    for column in (
        "membership_sha256",
        "feature_mean_sha256",
        "feature_scale_sha256",
        "source_run_manifest_sha256",
    ):
        _require(
            _is_sha256(first[column]),
            "{} {} is not a SHA-256 digest".format(run_name, column),
        )
    batches_per_epoch = int(
        math.ceil(float(contract["rows"]) / float(configuration["batch_size"]))
    )
    updates_per_epoch = int(
        math.ceil(
            float(batches_per_epoch)
            / float(configuration["accumulation_steps"])
        )
    )
    expected_schedule_steps = max(
        1, updates_per_epoch * int(configuration["lr_schedule_horizon_epochs"])
    )
    _require(
        int(first["schedule_total_steps"]) == expected_schedule_steps,
        "{} final refits did not retain the three-epoch LR schedule".format(run_name),
    )
    _require(
        int(first["lr_schedule_horizon_epochs"]) == 3
        and int(configuration["max_epochs"]) == 3
        and int(configuration["lr_schedule_horizon_epochs"]) == 3,
        "{} compressed its LR schedule".format(run_name),
    )
    expected_trainable = {"head_only": 2561, "lora_cross": 23041}
    for arm, count in expected_trainable.items():
        row = audit.loc[audit["arm"].eq(arm)]
        _require(len(row) == 1, "{} lacks one {} final audit".format(run_name, arm))
        row = row.iloc[0]
        _require(int(row["trainable_parameters"]) == count, "{} {} trainable count changed".format(run_name, arm))
        parity = float(row["epoch0_probability_max_abs_error"])
        _require(math.isfinite(parity) and 0.0 <= parity <= 0.01, "{} {} epoch-zero parity exceeds 0.01".format(run_name, arm))
        head_l2 = float(row["head_parameter_delta_l2"])
        head_max = float(row["head_parameter_delta_max_abs"])
        adapter_l2 = float(row["adapter_parameter_delta_l2"])
        adapter_max = float(row["adapter_parameter_delta_max_abs"])
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in (head_l2, head_max, adapter_l2, adapter_max)),
            "{} {} contains an invalid parameter delta".format(run_name, arm),
        )
        _require(head_l2 > 0.0 and head_max > 0.0, "{} {} trained head did not move".format(run_name, arm))
        if arm == "head_only":
            _require(adapter_l2 == 0.0 and adapter_max == 0.0, "{} head-only arm changed adapters".format(run_name))
        else:
            _require(adapter_l2 > 0.0 and adapter_max > 0.0, "{} trained cross-LoRA adapters did not move".format(run_name))
        try:
            names = json.loads(str(row["trainable_names"]))
        except (TypeError, ValueError) as error:
            raise ValueError("{} {} trainable_names is not JSON: {}".format(run_name, arm, error))
        _require(
            isinstance(names, list)
            and len(names) == len(set(str(name) for name in names))
            and set(str(name) for name in names) == _expected_trainable_names(arm),
            "{} {} trainable_names do not match the prescribed placement".format(
                run_name, arm
            ),
        )
    return audit.copy()


def _canonical_retention_metrics(predictions, library, arm, training_seed, selected_epoch):
    frame = predictions.reset_index(drop=True)
    score = pd.to_numeric(frame["probability"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(score).all()), "retention probabilities are non-finite")
    _require(bool(((score >= 0.0) & (score <= 1.0)).all()), "retention probabilities are outside [0,1]")
    return {
        "library": library,
        "arm": arm,
        "training_seed": int(training_seed),
        "selected_epoch": int(selected_epoch),
        **cached_eval.ranking_metrics(frame, score),
    }


def audit_retention_tables(predictions, stored_metrics, library, training_seed, chosen_epoch, run_name):
    required = {
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
        "library",
        "arm",
        "training_seed",
        "selected_epoch",
        "probability",
    }
    _assert_columns(predictions, required, "{} retention predictions".format(run_name))
    _require(
        set(predictions.columns) == required,
        "{} retention-prediction schema changed".format(run_name),
    )
    _assert_unique(
        predictions,
        ("arm", "training_seed", "sequence_pair_sha256"),
        "{} retention predictions".format(run_name),
    )
    binder = pd.to_numeric(predictions["target_binder"], errors="raise").to_numpy(dtype=float)
    retention = pd.to_numeric(predictions["target_retention"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(binder).all()) and set(binder.tolist()) == {0.0, 1.0}, "{} retention labels are not binary".format(run_name))
    _require(bool(np.isfinite(retention).all()) and bool(((retention >= 0.0) & (retention <= 100.0)).all()), "{} retention values leave [0,100]".format(run_name))
    _require(np.array_equal(binder.astype(int), (retention >= 75.0).astype(int)), "{} binder labels do not equal retention >= 75".format(run_name))

    groups = set(
        predictions[["arm", "training_seed"]]
        .assign(training_seed=lambda frame: pd.to_numeric(frame["training_seed"], errors="raise").astype(int))
        .itertuples(index=False, name=None)
    )
    expected_groups = {(arm, int(training_seed)) for arm in ARMS}
    _require(groups == expected_groups, "{} retention arms/seeds changed".format(run_name))
    _assert_columns(stored_metrics, METRIC_METADATA, "{} retention metrics".format(run_name))
    _assert_unique(stored_metrics, ("arm", "training_seed"), "{} retention metrics".format(run_name))
    stored_groups = set(
        stored_metrics[["arm", "training_seed"]]
        .assign(training_seed=lambda frame: pd.to_numeric(frame["training_seed"], errors="raise").astype(int))
        .itertuples(index=False, name=None)
    )
    _require(stored_groups == expected_groups, "{} stored retention arms/seeds changed".format(run_name))

    panel_columns = (
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
    )
    reference_panel = None
    rows = []
    canonical_names = None
    for arm in ARMS:
        block = predictions.loc[predictions["arm"].eq(arm)].reset_index(drop=True)
        _require(bool(block["library"].eq(library).all()), "{} {} library changed".format(run_name, arm))
        _require(set(pd.to_numeric(block["training_seed"], errors="raise").astype(int)) == {int(training_seed)}, "{} {} seed changed".format(run_name, arm))
        _require(set(pd.to_numeric(block["selected_epoch"], errors="raise").astype(int)) == {int(chosen_epoch)}, "{} {} final epoch changed".format(run_name, arm))
        panel = block.loc[:, panel_columns].sort_values("sequence_pair_sha256", kind="mergesort").reset_index(drop=True)
        if reference_panel is None:
            reference_panel = panel
        else:
            _require(panel.equals(reference_panel), "{} retention target panel changes across arms".format(run_name))
        row = _canonical_retention_metrics(block, library, arm, training_seed, chosen_epoch)
        names = set(row).difference(METRIC_METADATA)
        if canonical_names is None:
            canonical_names = names
        else:
            _require(names == canonical_names, "canonical retention metric schema changed")
        stored = stored_metrics.loc[stored_metrics["arm"].eq(arm)]
        _require(len(stored) == 1, "{} lacks one stored {} metric row".format(run_name, arm))
        stored = stored.iloc[0]
        _require(set(stored_metrics.columns) == set(METRIC_METADATA) | names, "{} stored retention metric schema changed".format(run_name))
        _require(str(stored["library"]) == library, "{} stored metric library changed".format(run_name))
        _require(int(stored["training_seed"]) == int(training_seed), "{} stored metric seed changed".format(run_name))
        _require(int(stored["selected_epoch"]) == int(chosen_epoch), "{} stored metric epoch changed".format(run_name))
        for name in names:
            _require(_same_number(stored[name], row[name]), "{} {} stored {} differs from recomputation".format(run_name, arm, name))
        rows.append(row)
    contract = matched.PRIMARY_CONTRACT[library]
    _require(len(reference_panel) == int(contract["retention_rows"]), "{} retention row count changed".format(run_name))
    _require(int(pd.to_numeric(reference_panel["target_binder"], errors="raise").sum()) == int(contract["retention_positive"]), "{} retention positive count changed".format(run_name))
    _require(cached_eval.membership_sha256(reference_panel) == contract["retention_membership_sha256"], "{} retention membership changed".format(run_name))
    return pd.DataFrame(rows), reference_panel


def _audit_per_peptide(stored, predictions, library, training_seed, chosen_epoch, run_name):
    expected_rows = []
    for arm in ARMS:
        block = predictions.loc[predictions["arm"].eq(arm)].reset_index(drop=True)
        expected_rows.extend(
            matched.per_peptide_records(
                block,
                pd.to_numeric(block["probability"], errors="raise").to_numpy(dtype=float),
                library,
                arm,
                training_seed,
                chosen_epoch,
            )
        )
    expected = pd.DataFrame(expected_rows)
    keys = ("arm", "training_seed", "peptide_identity")
    _assert_unique(stored, keys, "{} per-peptide metrics".format(run_name))
    _require(set(stored.columns) == set(expected.columns), "{} per-peptide metric schema changed".format(run_name))
    expected = expected.set_index(list(keys)).sort_index()
    stored = stored.set_index(list(keys)).sort_index()
    _require(list(expected.index) == list(stored.index), "{} per-peptide membership changed".format(run_name))
    for key in expected.index:
        for column in expected.columns:
            _require(
                _same_value(stored.loc[key, column], expected.loc[key, column]),
                "{} per-peptide {} {} differs from recomputation".format(run_name, key, column),
            )
    return expected.reset_index()


def _expected_paired_difference(metrics, library, training_seed, chosen_epoch):
    by_arm = metrics.set_index("arm")
    _require(set(by_arm.index) == set(ARMS), "paired retention metrics lack an arm")
    row = {
        "library": library,
        "training_seed": int(training_seed),
        "selected_epoch": int(chosen_epoch),
        "comparison": "lora_cross_minus_head_only",
    }
    for metric in SUMMARY_METRICS:
        row[metric + "_change"] = float(by_arm.loc["lora_cross", metric] - by_arm.loc["head_only", metric])
    return row


def _audit_paired_difference(stored, expected, run_name):
    _require(len(stored) == 1, "{} paired differences must contain one row".format(run_name))
    required = set(expected)
    _require(
        set(stored.columns) == required,
        "{} paired-difference schema changed".format(run_name),
    )
    row = stored.iloc[0]
    for key, value in expected.items():
        _require(
            _same_value(row[key], value),
            "{} paired difference {} differs from recomputation".format(run_name, key),
        )
    return expected


def _audit_retention_against_cache(panel, source, library, run_name):
    cache_rows = matched_aggregate._read_cache_audit_rows(source["sources"]["cache_rows"]["path"])
    expected = cache_rows.loc[
        cache_rows["source_kind"].eq("retention")
        & cache_rows["library"].eq(library)
        & cache_rows["measurement_missing"].eq("0"),
        [
            "pair_uid",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "target_retention",
            "target_binder",
        ],
    ].copy()
    expected = expected.sort_values("sequence_pair_sha256", kind="mergesort").reset_index(drop=True)
    observed = panel.sort_values("sequence_pair_sha256", kind="mergesort").reset_index(drop=True)
    _require(len(expected) == len(observed), "{} retention panel differs from cache rows".format(run_name))
    for column in ("pair_uid", "chain1_sha256", "chain2_sha256", "sequence_pair_sha256"):
        _require(expected[column].astype(str).equals(observed[column].astype(str)), "{} retention {} differs from cache rows".format(run_name, column))
    for column, dtype in (("target_retention", float), ("target_binder", int)):
        _require(
            np.array_equal(
                pd.to_numeric(expected[column], errors="raise").to_numpy(dtype=dtype),
                pd.to_numeric(observed[column], errors="raise").to_numpy(dtype=dtype),
            ),
            "{} retention {} differs from cache rows".format(run_name, column),
        )


def _validate_model_delta_metadata(
    manifest,
    verified_outputs,
    audit,
    source,
    library,
    training_seed,
    chosen_epoch,
    run_name,
):
    """Validate opaque model-delta files through external metadata and hashes."""
    contract = matched.PRIMARY_CONTRACT[library]
    delta_metadata = manifest.get("model_deltas")
    _require(
        isinstance(delta_metadata, dict) and set(delta_metadata) == set(ARMS),
        "{} model-delta metadata changed".format(run_name),
    )
    final_by_arm = audit.set_index("arm")
    for arm in ARMS:
        filename = "model_delta_seed{}_{}.pt".format(training_seed, arm)
        record = delta_metadata[arm]
        _require(
            isinstance(record, dict),
            "{} {} model-delta metadata is malformed".format(run_name, arm),
        )
        expected_metadata = {
            "filename": filename,
            "sha256": verified_outputs[filename]["sha256"],
            "library": library,
            "arm": arm,
            "training_seed": int(training_seed),
            "selected_epoch": int(chosen_epoch),
            "base_checkpoint_sha256": source["sources"]["checkpoint"]["sha256"],
            "primary_membership_sha256": contract["membership_sha256"],
            "head_initialization_sha256": str(
                final_by_arm.loc[arm, "head_initialization_sha256"]
            ),
        }
        _require(
            set(record) == set(expected_metadata),
            "{} {} model-delta metadata schema changed".format(run_name, arm),
        )
        for key, value in expected_metadata.items():
            _require(
                _same_value(record.get(key), value),
                "{} {} model-delta {} changed".format(run_name, arm, key),
            )
    return _normalized_json(delta_metadata)


def load_verified_run(run_dir):
    """Verify one complete shared-epoch artifact and recompute its results."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "shared-epoch run directory is missing: {}".format(run_dir))
    run_name = run_dir.name
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "{} has no manifest.json".format(run_name))
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == TRAINER_SCHEMA_VERSION, "{} trainer schema changed".format(run_name))
    _require(manifest.get("analysis_status") == "retrospective_exploratory", "{} analysis status changed".format(run_name))
    library, training_seed, configuration = _validate_configuration(manifest, run_name)
    runtime = _validate_runtime(manifest, run_name)

    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "{} has no output records".format(run_name))
    expected_outputs = _expected_outputs(training_seed)
    _require(set(outputs) == expected_outputs, "{} output set is incomplete or unexpected".format(run_name))
    verified_outputs = {}
    for filename, record in sorted(outputs.items()):
        _require(Path(filename).name == filename, "{} contains a non-local output name".format(run_name))
        verified_outputs[filename] = _verified_recorded_file(record, run_dir / filename, "{} {}".format(run_name, filename))
        if filename.endswith(".pt"):
            _require((run_dir / filename).stat().st_size > 0, "{} {} is empty".format(run_name, filename))

    source, source_provenance = _validate_source_matched_run(
        manifest,
        run_dir,
        verified_outputs,
        library,
        training_seed,
        run_name,
    )
    verified_code = _validate_shared_code_and_source_records(
        manifest, source, run_name
    )

    weak = _read_output_table(run_dir, "weak_validation_predictions.csv", run_name)
    epoch0_arm_parity = _epoch0_arm_parity(weak)
    chosen_epoch, expected_selection = recompute_shared_epoch_selection(weak, library, training_seed)
    _require(
        int(configuration.get("chosen_positive_epoch", -1)) == int(chosen_epoch),
        "{} configured chosen epoch differs from recomputation".format(run_name),
    )
    stored_selection = _read_output_table(run_dir, "shared_epoch_selection.csv", run_name)
    selection = audit_shared_epoch_selection(stored_selection, expected_selection, run_name)
    audit = _read_output_table(run_dir, "training_audit.csv", run_name)
    audit = _validate_training_audit(audit, library, training_seed, chosen_epoch, configuration, run_name)
    _require(
        set(audit["source_run_manifest_sha256"].astype(str))
        == {source["manifest_sha256"]},
        "{} final refits do not bind the source manifest".format(run_name),
    )
    _require(
        _same_number(
            audit["source_validation_epoch0_arm_max_abs_difference"].iloc[0],
            epoch0_arm_parity,
        ),
        "{} training audit epoch-zero arm parity changed".format(run_name),
    )
    final_epoch0_parity = float(
        audit["final_refit_epoch0_arm_max_abs_difference"].iloc[0]
    )
    _require(
        math.isfinite(final_epoch0_parity)
        and 0.0
        <= final_epoch0_parity
        <= shared_trainer.EPOCH0_ARM_PARITY_ATOL,
        "{} direct final-refit epoch-zero arm parity failed".format(run_name),
    )
    selection_diagnostics = manifest.get("selection_diagnostics")
    _require(
        isinstance(selection_diagnostics, dict),
        "{} lacks selection diagnostics".format(run_name),
    )
    expected_selection_diagnostics = {
        "epoch0_validation_arm_max_abs_difference": epoch0_arm_parity,
        "epoch0_better": bool(selection["epoch0_better"].iloc[0]),
        "epoch0_no_worse": bool(selection["epoch0_no_worse"].iloc[0]),
        "epoch0_minus_selected_log_loss": float(
            selection["epoch0_minus_selected_log_loss"].iloc[0]
        ),
    }
    for key, expected_value in expected_selection_diagnostics.items():
        observed_value = selection_diagnostics.get(key)
        if isinstance(expected_value, bool):
            agrees = _as_bool(observed_value, "{} {}".format(run_name, key)) == expected_value
        else:
            agrees = _same_number(observed_value, expected_value)
        _require(
            agrees,
            "{} selection diagnostic {} changed".format(run_name, key),
        )
    refit_diagnostics = manifest.get("refit_diagnostics")
    _require(
        isinstance(refit_diagnostics, dict),
        "{} lacks refit diagnostics".format(run_name),
    )
    _require(
        set(refit_diagnostics)
        == {
            "head_initialization_sha256",
            "live_head_initialization_sha256",
            "live_adapter_initialization_sha256",
            "final_refit_epoch0_arm_max_abs_difference",
            "frozen_source_reproduction_max_abs_difference",
        },
        "{} refit-diagnostic schema changed".format(run_name),
    )
    head_initialization = str(audit["head_initialization_sha256"].iloc[0])
    _require(
        str(refit_diagnostics.get("head_initialization_sha256"))
        == head_initialization
        and str(refit_diagnostics.get("live_head_initialization_sha256"))
        == head_initialization,
        "{} refit head-initialization hash changed".format(run_name),
    )
    _require(
        _is_sha256(refit_diagnostics.get("live_adapter_initialization_sha256")),
        "{} live adapter-initialization hash is malformed".format(run_name),
    )
    _require(
        _same_number(
            refit_diagnostics.get("final_refit_epoch0_arm_max_abs_difference"),
            final_epoch0_parity,
        ),
        "{} refit direct epoch-zero diagnostic changed".format(run_name),
    )
    frozen_reproduction = float(
        refit_diagnostics.get("frozen_source_reproduction_max_abs_difference")
    )
    _require(
        math.isfinite(frozen_reproduction)
        and 0.0
        <= frozen_reproduction
        <= shared_trainer.SOURCE_REPRODUCTION_ATOL,
        "{} frozen-source reproduction exceeds tolerance".format(run_name),
    )
    _require(
        _same_number(
            audit["source_frozen_reproduction_max_abs_difference"].iloc[0],
            frozen_reproduction,
        ),
        "{} training audit frozen-source diagnostic changed".format(run_name),
    )
    predictions = _read_output_table(run_dir, "retention_predictions.csv", run_name)
    stored_metrics = _read_output_table(run_dir, "retention_metrics.csv", run_name)
    metrics, panel = audit_retention_tables(
        predictions,
        stored_metrics,
        library,
        training_seed,
        chosen_epoch,
        run_name,
    )
    _audit_retention_against_cache(panel, source, library, run_name)
    per_peptide = _read_output_table(run_dir, "per_peptide_metrics.csv", run_name)
    _audit_per_peptide(per_peptide, predictions, library, training_seed, chosen_epoch, run_name)
    paired_expected = _expected_paired_difference(metrics, library, training_seed, chosen_epoch)
    paired_stored = _read_output_table(run_dir, "paired_differences.csv", run_name)
    _audit_paired_difference(paired_stored, paired_expected, run_name)

    rows = manifest.get("rows")
    contract = matched.PRIMARY_CONTRACT[library]
    _require(isinstance(rows, dict), "{} has no row-count audit".format(run_name))
    expected_rows = {
        "primary": contract["rows"],
        "primary_positive": contract["positive"],
        "primary_negative": contract["negative"],
        "retention": contract["retention_rows"],
        "retention_positive": contract["retention_positive"],
        "source_validation_prediction_records": len(weak),
        "retention_prediction_records": 2 * contract["retention_rows"],
    }
    for key, value in expected_rows.items():
        _require(int(rows.get(key, -1)) == int(value), "{} manifest row count {} changed".format(run_name, key))
    retention_feature_error = float(
        manifest.get("retention_feature_max_abs_difference", float("inf"))
    )
    _require(
        math.isfinite(retention_feature_error)
        and 0.0 <= retention_feature_error <= 1e-4,
        "{} live/cached retention feature parity exceeds 1e-4".format(run_name),
    )

    _validate_model_delta_metadata(
        manifest,
        verified_outputs,
        audit,
        source,
        library,
        training_seed,
        chosen_epoch,
        run_name,
    )

    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "library": library,
        "training_seed": int(training_seed),
        "configuration": configuration,
        "runtime_contract": runtime,
        "verified_outputs": verified_outputs,
        "verified_code": verified_code,
        "source": source,
        "source_provenance": source_provenance,
        "selection": selection,
        "training_audit": audit,
        "metrics": metrics,
        "panel": panel,
        "predictions": predictions,
        "paired": pd.DataFrame([paired_expected]),
    }


def _configuration_contract(configuration):
    output = copy.deepcopy(configuration)
    for key in ("library", "training_seeds", "selected_C", "chosen_positive_epoch"):
        output.pop(key, None)
    return _normalized_json(output)


def combine_verified_runs(verified_runs):
    expected_keys = {
        (library, seed)
        for library in EXPECTED_LIBRARIES
        for seed in EXPECTED_TRAINING_SEEDS
    }
    _require(len(verified_runs) == len(expected_keys), "exactly six shared-epoch runs are required")
    keys = [(run["library"], int(run["training_seed"])) for run in verified_runs]
    _require(len(keys) == len(set(keys)), "duplicate library/training-seed run")
    _require(set(keys) == expected_keys, "runs do not cover LibA/LibB x the three expected seeds")

    configuration = _configuration_contract(verified_runs[0]["configuration"])
    runtime = verified_runs[0]["runtime_contract"]
    for run in verified_runs[1:]:
        _require(_configuration_contract(run["configuration"]) == configuration, "{} used different shared-training hyperparameters".format(run["run_name"]))
        _require(run["runtime_contract"] == runtime, "{} used a different GPU/software runtime".format(run["run_name"]))

    # Reuse the existing full six-run source audit to enforce common model,
    # data, code, reference, configuration, runtime, and panel contracts.
    source_combined = matched_aggregate.combine_verified_runs([run["source"] for run in verified_runs])
    metric_blocks = []
    selection_blocks = []
    prediction_blocks = []
    audit_blocks = []
    for run in sorted(verified_runs, key=lambda item: (item["library"], item["training_seed"])):
        metrics = run["metrics"].copy()
        metrics.insert(0, "source_run", run["run_name"])
        metric_blocks.append(metrics)
        selection = run["selection"].copy()
        selection.insert(0, "source_run", run["run_name"])
        selection_blocks.append(selection)
        predictions = run["predictions"].copy()
        predictions.insert(0, "source_run", run["run_name"])
        prediction_blocks.append(predictions)
        audit = run["training_audit"].copy()
        audit.insert(0, "source_run", run["run_name"])
        if "library" not in audit.columns:
            audit.insert(1, "library", run["library"])
        audit_blocks.append(audit)
    metrics = pd.concat(metric_blocks, ignore_index=True, sort=False)
    selection = pd.concat(selection_blocks, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_blocks, ignore_index=True, sort=False)
    audit = pd.concat(audit_blocks, ignore_index=True, sort=False)
    _assert_unique(metrics, ("library", "arm", "training_seed"), "combined retention metrics")
    _assert_unique(selection, ("library", "training_seed", "epoch"), "combined epoch selection")
    _assert_unique(predictions, ("library", "arm", "training_seed", "sequence_pair_sha256"), "combined retention predictions")
    return {
        "metrics": metrics.sort_values(["library", "arm", "training_seed"], kind="mergesort").reset_index(drop=True),
        "selection": selection.sort_values(["library", "training_seed", "epoch"], kind="mergesort").reset_index(drop=True),
        "predictions": predictions.sort_values(["library", "training_seed", "arm", "sequence_pair_sha256"], kind="mergesort").reset_index(drop=True),
        "training_audit": audit.sort_values(["library" if "library" in audit.columns else "source_run", "training_seed", "arm"], kind="mergesort").reset_index(drop=True),
        "configuration_contract": configuration,
        "runtime_contract": runtime,
        "common_source_hashes": source_combined["common_source_hashes"],
        "library_reference_source_hashes": source_combined["library_reference_source_hashes"],
    }


def build_paired_changes(metrics):
    rows = []
    for library in EXPECTED_LIBRARIES:
        for seed in EXPECTED_TRAINING_SEEDS:
            block = metrics.loc[
                metrics["library"].eq(library)
                & pd.to_numeric(metrics["training_seed"], errors="raise").eq(int(seed))
            ].set_index("arm")
            _require(set(block.index) == set(ARMS), "{} seed {} lacks a paired arm".format(library, seed))
            head_epoch = int(block.loc["head_only", "selected_epoch"])
            cross_epoch = int(block.loc["lora_cross", "selected_epoch"])
            _require(head_epoch == cross_epoch and head_epoch in (1, 2, 3), "{} seed {} arms do not share a positive epoch".format(library, seed))
            row = {
                "library": library,
                "training_seed": int(seed),
                "selected_epoch": cross_epoch,
                "comparison": "lora_cross_minus_head_only",
            }
            for metric in SUMMARY_METRICS:
                row[metric + "_change"] = float(block.loc["lora_cross", metric] - block.loc["head_only", metric])
            rows.append(row)
    output = pd.DataFrame(rows)
    _assert_unique(output, ("library", "comparison", "training_seed"), "paired cross-head changes")
    return output.sort_values(["library", "training_seed"], kind="mergesort").reset_index(drop=True)


def build_summary(metrics, paired):
    rows = []
    for (library, arm), group in metrics.groupby(["library", "arm"], sort=True):
        group = group.sort_values("training_seed", kind="mergesort")
        _require(len(group) == 3, "{} {} must contain three seeds".format(library, arm))
        row = {
            "library": library,
            "result_type": "model_score",
            "result": arm,
            "reference": "",
            "n_seed_runs": 3,
            "training_seeds": ",".join(str(int(value)) for value in group["training_seed"]),
            "selected_epochs": ",".join(str(int(value)) for value in group["selected_epoch"]),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    for library, group in paired.groupby("library", sort=True):
        group = group.sort_values("training_seed", kind="mergesort")
        _require(len(group) == 3, "{} paired changes must contain three seeds".format(library))
        row = {
            "library": library,
            "result_type": "paired_change",
            "result": "lora_cross",
            "reference": "head_only",
            "n_seed_runs": 3,
            "training_seeds": ",".join(str(int(value)) for value in group["training_seed"]),
            "selected_epochs": ",".join(str(int(value)) for value in group["selected_epoch"]),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric + "_change"], errors="raise").to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["library", "result_type", "result"], kind="mergesort").reset_index(drop=True)


def build_within_chain_gate(selection, paired):
    """Evaluate the predeclared, retention-informed exploratory placement gate."""
    rows = []
    for library in EXPECTED_LIBRARIES:
        library_selection = selection.loc[selection["library"].eq(library)].copy()
        selected = library_selection.loc[
            pd.to_numeric(library_selection["selected"], errors="raise").eq(1)
        ].copy()
        epoch0 = library_selection.loc[
            pd.to_numeric(library_selection["epoch"], errors="raise").eq(0)
        ].copy()
        changes = paired.loc[paired["library"].eq(library)].copy()
        _require(
            len(selected) == 3 and len(epoch0) == 3 and len(changes) == 3,
            "{} gate lacks three paired seeds".format(library),
        )
        selected = selected.sort_values("training_seed", kind="mergesort")
        epoch0 = epoch0.sort_values("training_seed", kind="mergesort")
        changes = changes.sort_values("training_seed", kind="mergesort")
        selected_seeds = pd.to_numeric(
            selected["training_seed"], errors="raise"
        ).to_numpy(dtype=int)
        _require(
            np.array_equal(
                selected_seeds,
                pd.to_numeric(epoch0["training_seed"], errors="raise").to_numpy(dtype=int),
            )
            and np.array_equal(
                selected_seeds,
                pd.to_numeric(changes["training_seed"], errors="raise").to_numpy(dtype=int),
            ),
            "{} gate seed pairing changed".format(library),
        )
        cross_better_than_epoch0 = bool(
            (
                pd.to_numeric(selected["lora_cross_log_loss"], errors="raise").to_numpy(dtype=float)
                < pd.to_numeric(epoch0["lora_cross_log_loss"], errors="raise").to_numpy(dtype=float)
            ).all()
        )
        cross_validation_better = bool(
            (
                pd.to_numeric(selected["lora_cross_log_loss"], errors="raise").to_numpy(dtype=float)
                < pd.to_numeric(selected["head_only_log_loss"], errors="raise").to_numpy(dtype=float)
            ).all()
        )
        retention_primary_better = bool(
            pd.to_numeric(
                changes["within_peptide_macro_spearman_change"], errors="raise"
            ).gt(0.0).all()
        )
        rows.append(
            {
                "library": library,
                "n_seeds": 3,
                "cross_log_loss_better_than_epoch0_all_seeds": cross_better_than_epoch0,
                "cross_log_loss_better_than_head_all_seeds": cross_validation_better,
                "within_peptide_spearman_better_all_seeds": retention_primary_better,
                "placement_gate_pass": bool(
                    cross_better_than_epoch0
                    and cross_validation_better
                    and retention_primary_better
                ),
                "interpretation": "exploratory: measured retention participates in this gate",
            }
        )
    return pd.DataFrame(rows)


def _metric_text(row, metric):
    return "{:.3f}+-{:.3f}".format(
        float(row[metric + "_mean"]),
        float(row[metric + "_sample_sd"]),
    )


def summary_markdown(summary, gate):
    labels = {"head_only": "Head-only", "lora_cross": "Cross-chain LoRA + head"}
    lines = [
        "# Shared-epoch MINT/LoRA aggregation",
        "",
        "All six library-by-seed artifacts passed hash, source-lineage, shared-epoch, paired-training, panel-membership, and direct metric-recomputation checks. Values are mean +- sample SD across the three prespecified optimization seeds.",
    ]
    for library in EXPECTED_LIBRARIES:
        model = summary.loc[
            summary["library"].eq(library) & summary["result_type"].eq("model_score")
        ].set_index("result")
        change = summary.loc[
            summary["library"].eq(library) & summary["result_type"].eq("paired_change")
        ].iloc[0]
        gate_row = gate.loc[gate["library"].eq(library)].iloc[0]
        lines.extend(
            [
                "",
                "## {}".format(library),
                "",
                "| Result | Epochs by seed | AUROC | AP | Global Spearman | Within-peptide Spearman |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            row = model.loc[arm]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    labels[arm],
                    row["selected_epochs"],
                    _metric_text(row, "global_auroc"),
                    _metric_text(row, "global_auprc"),
                    _metric_text(row, "global_spearman"),
                    _metric_text(row, "within_peptide_macro_spearman"),
                )
            )
        lines.append(
            "| Cross-chain LoRA + head - Head-only | {} | {} | {} | {} | {} |".format(
                change["selected_epochs"],
                _metric_text(change, "global_auroc"),
                _metric_text(change, "global_auprc"),
                _metric_text(change, "global_spearman"),
                _metric_text(change, "within_peptide_macro_spearman"),
            )
        )
        lines.extend(
            [
                "",
                "Within-chain placement gate: **{}**. This gate is exploratory because measured retention participates in it.".format(
                    "pass" if bool(gate_row["placement_gate_pass"]) else "fail"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Both arms use the same selected positive epoch within each library/seed pair. Epoch selection uses only pooled weak-validation log loss, weighting the two arms equally; epoch 0 is reported as a sensitivity but cannot replace the positive epoch. AUROC and AP use the 75% retention threshold only on the measured evaluation panel. Seed SD is optimization-seed variability, not biological or evaluation-set uncertainty. This analysis remains retrospective and exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def _recheck_verified_inputs(verified):
    matched_aggregate._recheck_verified_inputs([run["source"] for run in verified])
    for run in verified:
        _require(sha256_file(run["manifest_path"]) == run["manifest_sha256"], "{} manifest changed during aggregation".format(run["run_name"]))
        for filename, record in run["verified_outputs"].items():
            _require(sha256_file(run["run_dir"] / filename) == record["sha256"], "{} {} changed during aggregation".format(run["run_name"], filename))
        provenance = run["source_provenance"]
        _require(sha256_file(provenance["manifest_path"]) == provenance["manifest_sha256"], "{} source manifest changed during aggregation".format(run["run_name"]))
        _require(sha256_file(provenance["weak_validation_path"]) == provenance["weak_validation_sha256"], "{} source validation changed during aggregation".format(run["run_name"]))
        for name, record in run["verified_code"].items():
            _require(
                sha256_file(record["path"]) == record["sha256"],
                "{} {} code changed during aggregation".format(run["run_name"], name),
            )


def _paths_overlap(first, second):
    first = Path(first).resolve()
    second = Path(second).resolve()
    return first == second or first in second.parents or second in first.parents


def _assert_output_disjoint(output_dir, protected_paths):
    output_dir = Path(output_dir).resolve()
    for protected in protected_paths:
        protected = Path(protected).resolve()
        _require(
            not _paths_overlap(output_dir, protected),
            "aggregate output must be disjoint from input artifact {}".format(
                protected
            ),
        )


def _file_identity(value):
    return int(value.st_dev), int(value.st_ino)


def _path_matches_open_directory(path, directory_fd):
    """Return whether ``path`` still names the directory held by ``directory_fd``."""
    try:
        observed = os.stat(str(path), follow_symlinks=False)
    except OSError:
        return False
    expected = os.fstat(directory_fd)
    return stat.S_ISDIR(observed.st_mode) and _file_identity(observed) == _file_identity(
        expected
    )


def _link_staged_file_noreplace(source, filename, output_fd):
    """Atomically link one staged regular file into a claimed output directory."""
    _require(Path(filename).name == filename, "non-local aggregate output name")
    source_stat = os.stat(str(source), follow_symlinks=False)
    _require(stat.S_ISREG(source_stat.st_mode), "staged aggregate output is not regular")
    try:
        os.link(
            str(source),
            filename,
            dst_dir_fd=output_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        raise ValueError(
            "aggregate output entry exists; refusing overwrite: {}".format(filename)
        )
    output_stat = os.stat(filename, dir_fd=output_fd, follow_symlinks=False)
    _require(
        stat.S_ISREG(output_stat.st_mode)
        and _file_identity(output_stat) == _file_identity(source_stat),
        "published aggregate output identity changed: {}".format(filename),
    )
    return _file_identity(output_stat)


def _fsync_regular_file(path):
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require(
            stat.S_ISREG(os.fstat(descriptor).st_mode),
            "staged aggregate output is not regular",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_claimed_output(output_dir, output_fd, linked_entries):
    """Remove only entries whose inodes this process linked into its claimed dir."""
    for filename, identity in reversed(linked_entries):
        try:
            observed = os.stat(filename, dir_fd=output_fd, follow_symlinks=False)
        except OSError:
            continue
        if _file_identity(observed) != identity:
            continue
        try:
            os.unlink(filename, dir_fd=output_fd)
        except OSError:
            pass
    if _path_matches_open_directory(output_dir, output_fd):
        try:
            os.rmdir(str(output_dir))
        except OSError:
            pass


def _atomic_publish(output_dir, tables, markdown, manifest_builder):
    """Publish without replacing output, committing ``manifest.json`` last.

    Lustre does not implement ``renameat2(RENAME_NOREPLACE)``.  Instead, an
    atomic ``mkdir`` claims the final path, and hard links publish complete
    staged files without replacement.  ``manifest.json`` is linked last and
    is therefore the commit marker: a failed publication is never a valid
    artifact even if hostile interference prevents cleanup of the claimed
    directory.
    """
    requested_output = Path(os.path.abspath(str(output_dir)))
    _require(
        not os.path.lexists(str(requested_output)),
        "output directory exists; refusing overwrite",
    )
    output_dir = requested_output.parent.resolve() / requested_output.name
    _require(
        not os.path.lexists(str(output_dir)),
        "output directory exists; refusing overwrite",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".{}-staging-".format(output_dir.name),
            dir=str(output_dir.parent),
        )
    ).resolve()
    os.chmod(str(staging), 0o700)
    _require(staging.parent == output_dir.parent, "staging directory escaped output parent")
    output_fd = None
    linked_entries = []
    claimed_output = False
    published = False
    staged_manifest_identity = None
    try:
        staged_outputs = {}
        for name, frame in tables.items():
            _require(Path(name).name == name, "non-local aggregate output name")
            _require(
                name not in ("summary.md", "manifest.json"),
                "reserved aggregate output name: {}".format(name),
            )
            path = staging / name
            _write_csv(frame, path)
            staged_outputs[name] = path
        summary_path = staging / "summary.md"
        with open(str(summary_path), "w") as handle:
            handle.write(markdown)
        os.chmod(str(summary_path), 0o600)
        staged_outputs["summary.md"] = summary_path
        output_records = {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(staged_outputs.items())
        }
        manifest = manifest_builder(output_records)
        staged_manifest = staging / "manifest.json"
        _write_json(manifest, staged_manifest)
        staged_manifest_identity = _file_identity(
            os.stat(str(staged_manifest), follow_symlinks=False)
        )
        for path in list(staged_outputs.values()) + [staged_manifest]:
            _fsync_regular_file(path)
        staging_fd = os.open(
            str(staging), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        _require(
            not os.path.lexists(str(output_dir)),
            "output directory appeared before publication",
        )
        try:
            os.mkdir(str(output_dir), 0o700)
        except FileExistsError:
            raise ValueError("output directory exists; refusing overwrite")
        claimed_output = True
        open_flags = os.O_RDONLY
        open_flags |= getattr(os, "O_DIRECTORY", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(str(output_dir), open_flags)
        os.fchmod(output_fd, 0o700)
        _require(
            _path_matches_open_directory(output_dir, output_fd),
            "claimed aggregate output directory identity changed",
        )
        parent_fd = os.open(
            str(output_dir.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        for name, path in sorted(staged_outputs.items()):
            _require(
                _path_matches_open_directory(output_dir, output_fd),
                "claimed aggregate output directory identity changed",
            )
            intended_identity = _file_identity(
                os.stat(str(path), follow_symlinks=False)
            )
            linked_entries.append((name, intended_identity))
            identity = _link_staged_file_noreplace(path, name, output_fd)
            _require(
                identity == intended_identity,
                "published aggregate output identity changed: {}".format(name),
            )
        _require(
            _path_matches_open_directory(output_dir, output_fd),
            "claimed aggregate output directory identity changed before commit",
        )
        _require(
            set(os.listdir(output_fd)) == set(staged_outputs),
            "aggregate output set changed before commit",
        )
        os.fsync(output_fd)
        linked_entries.append(("manifest.json", staged_manifest_identity))
        manifest_identity = _link_staged_file_noreplace(
            staged_manifest, "manifest.json", output_fd
        )
        _require(
            manifest_identity == staged_manifest_identity,
            "published aggregate output identity changed: manifest.json",
        )
        # Once the manifest hard link exists, the artifact has been committed.
        # Never roll it back: another process may already have opened it.
        published = True
        os.fsync(output_fd)
        _require(
            _path_matches_open_directory(output_dir, output_fd),
            "published aggregate output directory identity changed",
        )
        _require(
            set(os.listdir(output_fd)) == set(staged_outputs) | {"manifest.json"},
            "published aggregate output set changed",
        )
        return manifest
    finally:
        if (
            not published
            and claimed_output
            and output_fd is not None
            and staged_manifest_identity is not None
        ):
            try:
                observed_manifest = os.stat(
                    "manifest.json", dir_fd=output_fd, follow_symlinks=False
                )
            except OSError:
                observed_manifest = None
            if (
                observed_manifest is not None
                and _file_identity(observed_manifest) == staged_manifest_identity
            ):
                published = True
        if not published and claimed_output and output_fd is not None:
            _cleanup_claimed_output(output_dir, output_fd, linked_entries)
        if output_fd is not None:
            os.close(output_fd)
        if staging.is_dir() and staging.parent == output_dir.parent:
            shutil.rmtree(str(staging))


def run(args):
    started = time.time()
    aggregate_code_hash = sha256_file(Path(__file__).resolve())
    dependency_hashes = {
        "shared_epoch_trainer": sha256_file(Path(shared_trainer.__file__).resolve()),
        "matched_aggregator": sha256_file(Path(matched_aggregate.__file__).resolve()),
        "matched_trainer": sha256_file(Path(matched.__file__).resolve()),
        "canonical_evaluator": sha256_file(Path(cached_eval.__file__).resolve()),
    }
    requested_output = Path(os.path.abspath(str(args.output_dir)))
    _require(
        not os.path.lexists(str(requested_output)),
        "output directory exists; refusing overwrite",
    )
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(
        not os.path.lexists(str(output_dir)),
        "output directory exists; refusing overwrite",
    )
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    _require(len(run_dirs) == 6, "exactly six --run-dirs values are required")
    _require(len(run_dirs) == len(set(run_dirs)), "duplicate input directory")

    verified = [load_verified_run(path) for path in run_dirs]
    _assert_output_disjoint(
        output_dir,
        run_dirs + [run["source"]["run_dir"] for run in verified],
    )
    combined = combine_verified_runs(verified)
    metrics = combined["metrics"]
    paired = build_paired_changes(metrics)
    # Ensure the recomputed aggregate pairwise values reproduce every input's
    # stored and independently audited one-row pairwise result.
    source_paired = pd.concat([run["paired"] for run in verified], ignore_index=True)
    paired_compare = paired.sort_values(["library", "training_seed"]).reset_index(drop=True)
    source_compare = source_paired.sort_values(["library", "training_seed"]).reset_index(drop=True)
    _require(set(paired_compare.columns) == set(source_compare.columns), "source paired-difference schema changed")
    for column in paired_compare.columns:
        for observed, expected in zip(paired_compare[column], source_compare[column]):
            _require(_same_value(observed, expected), "source paired difference {} changed".format(column))
    summary = build_summary(metrics, paired)
    gate = build_within_chain_gate(combined["selection"], paired)
    markdown = summary_markdown(summary, gate)

    _recheck_verified_inputs(verified)
    _require(sha256_file(Path(__file__).resolve()) == aggregate_code_hash, "aggregate code changed during execution")
    for name, expected_hash in dependency_hashes.items():
        path = {
            "shared_epoch_trainer": Path(shared_trainer.__file__).resolve(),
            "matched_aggregator": Path(matched_aggregate.__file__).resolve(),
            "matched_trainer": Path(matched.__file__).resolve(),
            "canonical_evaluator": Path(cached_eval.__file__).resolve(),
        }[name]
        _require(sha256_file(path) == expected_hash, "{} changed during execution".format(name))

    tables = {
        "retention_predictions_by_seed.csv": combined["predictions"],
        "recomputed_retention_metrics_by_seed.csv": metrics,
        "shared_epoch_selection_by_seed.csv": combined["selection"],
        "training_audit_by_seed.csv": combined["training_audit"],
        "paired_cross_minus_head_by_seed.csv": paired,
        "mean_sample_sd_summary.csv": summary,
        "within_chain_placement_gate.csv": gate,
    }

    def build_manifest(output_records):
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis_status": "retrospective_exploratory",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "configuration": {
                "expected_libraries": list(EXPECTED_LIBRARIES),
                "expected_training_seeds": list(EXPECTED_TRAINING_SEEDS),
                "expected_arms": list(ARMS),
                "shared_positive_epoch_candidates": [1, 2, 3],
                "epoch0_policy": "report only; never selectable",
                "epoch_selection": SELECTION_DESCRIPTION,
                "metric_recomputation": "canonical evaluate_cached_weak_mint.ranking_metrics on every prediction block",
                "metric_comparison_tolerance": {"absolute": 1e-12, "relative": 1e-12},
                "seed_aggregation": "arithmetic mean and sample SD (ddof=1)",
                "paired_comparison": "lora_cross_minus_head_only within library and training seed",
                "model_delta_audit": "external metadata/path/SHA256 only; PyTorch pickle payloads are not deserialized",
                "placement_gate": "separately by library, all three seeds require cross-LoRA selected-epoch log loss below cross-LoRA epoch 0, cross-LoRA selected-epoch log loss below head-only at the same epoch, and positive within-peptide Spearman change; ties/NaNs fail and the result is exploratory because retention participates",
            },
            "shared_training_configuration": combined["configuration_contract"],
            "shared_runtime": combined["runtime_contract"],
            "common_source_hashes": combined["common_source_hashes"],
            "library_reference_source_hashes": combined["library_reference_source_hashes"],
            "source_runs": [
                {
                    "name": run["run_name"],
                    "path": str(run["run_dir"]),
                    "library": run["library"],
                    "training_seed": int(run["training_seed"]),
                    "manifest_path": str(run["manifest_path"]),
                    "manifest_sha256": run["manifest_sha256"],
                    "source_matched_run": run["source_provenance"],
                }
                for run in sorted(verified, key=lambda item: (item["library"], item["training_seed"]))
            ],
            "rows": {
                "source_runs": 6,
                "retention_prediction_records": int(len(combined["predictions"])),
                "recomputed_metric_records": int(len(metrics)),
                "shared_epoch_records": int(len(combined["selection"])),
                "training_audit_records": int(len(combined["training_audit"])),
                "paired_change_records": int(len(paired)),
                "summary_records": int(len(summary)),
                "placement_gate_records": int(len(gate)),
            },
            "outputs": output_records,
            "code": {"path": str(Path(__file__).resolve()), "sha256": aggregate_code_hash},
            "dependencies": {
                "shared_epoch_trainer": {"path": str(Path(shared_trainer.__file__).resolve()), "sha256": dependency_hashes["shared_epoch_trainer"]},
                "matched_aggregator": {"path": str(Path(matched_aggregate.__file__).resolve()), "sha256": dependency_hashes["matched_aggregator"]},
                "matched_trainer": {"path": str(Path(matched.__file__).resolve()), "sha256": dependency_hashes["matched_trainer"]},
                "canonical_evaluator": {"path": str(Path(cached_eval.__file__).resolve()), "sha256": dependency_hashes["canonical_evaluator"]},
            },
            "permissions": {"directory": "0700", "files": "0600"},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

    manifest = _atomic_publish(output_dir, tables, markdown, build_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_runs": 6,
                "metric_records": len(metrics),
                "paired_change_records": len(paired),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="exactly six one-library/one-seed shared-epoch artifact directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
