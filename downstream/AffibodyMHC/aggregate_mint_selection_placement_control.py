#!/usr/bin/env python
"""Audit and aggregate gated cross- versus within-chain LoRA controls.

The input set must contain all three prespecified optimization seeds for every
library that passed the shared-epoch placement gate, and no runs for a library
that failed it.  All row-level retention metrics are recomputed before paired
within-minus-cross changes and mean/sample-SD summaries are written.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC import finetune_mint_selection_placement_control as trainer
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "mint-selection-lora-placement-control-aggregate-v1"
ARMS = tuple(trainer.ARMS)
METRICS = tuple(trainer.CORE_METRICS)
FIXED_OUTPUTS = {
    "placement_contract.csv",
    "training_audit.csv",
    "retention_predictions.csv",
    "retention_metrics.csv",
    "paired_differences.csv",
    "per_peptide_metrics.csv",
    "run_summary.md",
}
PAIRED_PANEL_COLUMNS = (
    "pair_uid",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
    "target_retention",
    "target_binder",
)
OPTIMIZER_CONTRACT = {
    "name": "AdamW",
    "betas": [0.9, 0.98],
    "eps": 1e-8,
}
TRAINING_CONTRACT = {
    "lr_schedule_horizon_epochs": 3,
    "batch_size": 64,
    "eval_batch_size": 64,
    "accumulation_steps": 1,
    "head_lr": 1e-4,
    "adapter_lr": 2e-4,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "clip_norm": 1.0,
    "optimizer": OPTIMIZER_CONTRACT,
    "primary_endpoint": "within_peptide_macro_spearman",
    "secondary_endpoints": ["global_spearman", "global_auprc", "global_auroc"],
    "retention_threshold_for_auroc_ap": 75.0,
    "gate_policy": "library must pass all three seeds on all three predeclared conditions; ties, NaNs, or one failed seed block the control",
    "retention_usage": "measured retention participates in the predeclared gate and final metrics, but not the within-chain refit",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "JSON root is not an object: {}".format(path))
    return value


def _normalized_json(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _strict_integer_scalar(value, label):
    return trainer._strict_integer_values(pd.Series([value]), label)[0]


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _same(observed, expected, atol=1e-12, rtol=1e-12):
    if isinstance(expected, str):
        return str(observed) == expected
    if isinstance(expected, (bool, np.bool_)):
        return trainer._as_bool(observed, "comparison value") == bool(expected)
    observed = float(observed)
    expected = float(expected)
    if math.isnan(observed) or math.isnan(expected):
        return math.isnan(observed) and math.isnan(expected)
    return math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol)


def _compare_record(observed, expected, label):
    _require(set(observed) == set(expected), "{} schema changed".format(label))
    for key, value in expected.items():
        _require(_same(observed[key], value), "{} {} differs".format(label, key))


def _verify_output_set(run_dir, manifest, training_seed):
    outputs = manifest.get("outputs")
    expected = FIXED_OUTPUTS | {
        "model_delta_seed{}_lora_within.pt".format(int(training_seed))
    }
    _require(isinstance(outputs, dict) and set(outputs) == expected, "placement output set changed")
    verified = {}
    for name, record in sorted(outputs.items()):
        verified[name] = trainer._verify_recorded_file(
            record, run_dir / name, "placement {}".format(name)
        )
        if name.endswith(".pt"):
            _require((run_dir / name).stat().st_size > 0, "placement model delta is empty")
    return verified


def _validate_configuration(manifest):
    configuration = manifest.get("configuration")
    _require(isinstance(configuration, dict), "placement manifest has no configuration")
    library = str(configuration.get("library"))
    seed = _strict_integer_scalar(
        configuration.get("training_seed", -1), "placement training seed"
    )
    _require(library in matched.LIBRARIES, "unknown placement library")
    _require(seed in matched.DEFAULT_TRAINING_SEEDS, "unknown placement seed")
    _require(configuration.get("arms") == list(ARMS), "placement arms changed")
    _require(
        configuration.get("arm_reuse")
        == {
            "lora_cross": "verified shared-epoch source",
            "lora_within": "new refit",
        },
        "placement reuse contract changed",
    )
    selected_epoch = _strict_integer_scalar(
        configuration.get("shared_positive_epoch", -1),
        "placement shared positive epoch",
    )
    _require(1 <= selected_epoch <= 3, "bad shared epoch")
    for key, expected in TRAINING_CONTRACT.items():
        _require(configuration.get(key) == expected, "placement {} changed".format(key))
    _require(
        float(configuration.get("selected_C", float("nan")))
        == float(matched.PRIMARY_CONTRACT[library]["selected_c"]),
        "placement selected C changed",
    )
    lora = configuration.get("lora", {})
    expected_lora = {
        "rank": trainer.RANK,
        "alpha": trainer.ALPHA,
        "dropout": trainer.DROPOUT,
        "layers_zero_based": list(trainer.LAYERS),
        "projections": list(trainer.PROJECTIONS),
        "placements": dict(trainer.PLACEMENTS),
        "adapter_parameters_per_arm": trainer.ADAPTER_PARAMETERS,
        "total_trainable_parameters_per_arm": trainer.TOTAL_TRAINABLE_PARAMETERS,
    }
    _require(lora == expected_lora, "placement LoRA contract changed")
    _require(
        _normalized_json(manifest.get("canonical_contract"))
        == _normalized_json(matched.PRIMARY_CONTRACT[library]),
        "placement canonical contract changed",
    )
    return library, seed, configuration


def _audit_configuration_against_source(configuration, source_shared):
    """Bind every refit-affecting placement setting to the verified source."""
    source = source_shared["configuration"]
    source_mapping = {
        "selected_C": "selected_C",
        "lr_schedule_horizon_epochs": "lr_schedule_horizon_epochs",
        "batch_size": "batch_size",
        "eval_batch_size": "eval_batch_size",
        "accumulation_steps": "accumulation_steps",
        "head_lr": "head_lr",
        "adapter_lr": "adapter_lr",
        "weight_decay": "weight_decay",
        "warmup_fraction": "warmup_fraction",
        "clip_norm": "clip_norm",
    }
    for placement_key, source_key in source_mapping.items():
        _require(
            _same(configuration[placement_key], source[source_key]),
            "placement {} differs from verified source".format(placement_key),
        )
    _require(
        _strict_integer_scalar(
            configuration["shared_positive_epoch"],
            "placement shared positive epoch",
        )
        == _strict_integer_scalar(
            source["chosen_positive_epoch"], "verified source chosen epoch"
        ),
        "placement epoch differs from verified source",
    )
    source_lora = source["lora"]
    placement_lora = configuration["lora"]
    for key in ("rank", "alpha", "dropout", "layers_zero_based", "projections"):
        _require(
            placement_lora[key] == source_lora[key],
            "placement LoRA {} differs from verified source".format(key),
        )
    _require(
        placement_lora["placements"]["lora_cross"] == source_lora["placement"],
        "cross placement differs from verified source",
    )


def _audit_runtime_against_source(manifest, source_shared):
    runtime = manifest.get("runtime")
    source_runtime = source_shared.get("runtime_contract")
    _require(isinstance(runtime, dict), "placement runtime record is missing")
    _require(isinstance(source_runtime, dict), "verified source runtime record is missing")
    _require(
        set(source_runtime).issubset(runtime),
        "placement runtime record is incomplete",
    )
    for key, expected in source_runtime.items():
        _require(
            runtime[key] == expected,
            "placement runtime {} differs from verified source".format(key),
        )


def _canonical_metrics(block, library, arm, seed, epoch):
    block = block.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    contract = matched.PRIMARY_CONTRACT[library]
    _require(len(block) == int(contract["retention_rows"]), "retention row count changed")
    binder = pd.to_numeric(block["target_binder"], errors="raise").to_numpy(dtype=float)
    retention = pd.to_numeric(block["target_retention"], errors="raise").to_numpy(dtype=float)
    _require(
        bool(np.isfinite(binder).all()) and set(binder.tolist()) == {0.0, 1.0},
        "retention labels are not binary",
    )
    _require(
        bool(np.isfinite(retention).all())
        and bool(((retention >= 0.0) & (retention <= 100.0)).all()),
        "retention values leave [0,100]",
    )
    _require(int(binder.sum()) == int(contract["retention_positive"]), "binder count changed")
    expected_binder = pd.to_numeric(
        block["target_retention"], errors="raise"
    ).ge(75.0).astype(int)
    _require(
        expected_binder.equals(pd.to_numeric(block["target_binder"], errors="raise").astype(int)),
        "75-percent binder definition changed",
    )
    _require(
        cached_eval.membership_sha256(block) == contract["retention_membership_sha256"],
        "retention panel membership changed",
    )
    return matched.retention_metric_record(
        block,
        pd.to_numeric(block["probability"], errors="raise").to_numpy(dtype=float),
        library,
        arm,
        seed,
        epoch,
    )


def _audit_paired_prediction_panel(predictions):
    """Require both arms to score the identical, identically ordered panel."""
    blocks = {}
    for arm in ARMS:
        block = predictions.loc[predictions["arm"].eq(arm)].reset_index(drop=True)
        _require(not block.empty, "missing {} prediction panel".format(arm))
        blocks[arm] = block
    cross = blocks["lora_cross"]
    within = blocks["lora_within"]
    _require(len(cross) == len(within), "cross/within prediction panel lengths differ")
    for column in PAIRED_PANEL_COLUMNS:
        if column == "target_retention":
            left = pd.to_numeric(cross[column], errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(within[column], errors="raise").to_numpy(dtype=float)
            _require(
                bool(np.isfinite(left).all())
                and bool(np.isfinite(right).all())
                and np.array_equal(left, right),
                "cross/within prediction panels differ in {}".format(column),
            )
        elif column == "target_binder":
            left = pd.to_numeric(cross[column], errors="raise").to_numpy(dtype=float)
            right = pd.to_numeric(within[column], errors="raise").to_numpy(dtype=float)
            _require(
                bool(np.isfinite(left).all())
                and bool(np.isfinite(right).all())
                and np.array_equal(left, right),
                "cross/within prediction panels differ in {}".format(column),
            )
        else:
            _require(
                cross[column].astype(str).equals(within[column].astype(str)),
                "cross/within prediction panels differ in {}".format(column),
            )
    return cross


def _audit_contract(contract, library, seed, epoch, source_shared, gate):
    _require(len(contract) == 1, "placement contract must have one row")
    row = contract.iloc[0]
    _require(str(row["library"]) == library, "placement-contract library changed")
    _require(
        _strict_integer_scalar(row["training_seed"], "placement-contract seed")
        == seed,
        "placement-contract seed changed",
    )
    _require(
        _strict_integer_scalar(row["selected_epoch"], "placement-contract epoch")
        == epoch,
        "placement-contract epoch changed",
    )
    canonical = matched.PRIMARY_CONTRACT[library]
    _require(int(row["primary_rows"]) == canonical["rows"], "primary rows changed")
    _require(int(row["primary_positive"]) == canonical["positive"], "primary positives changed")
    _require(int(row["primary_negative"]) == canonical["negative"], "primary negatives changed")
    _require(str(row["primary_membership_sha256"]) == canonical["membership_sha256"], "primary membership changed")
    _require(int(row["cross_trainable_parameters"]) == trainer.TOTAL_TRAINABLE_PARAMETERS, "cross trainable count changed")
    _require(int(row["within_expected_trainable_parameters"]) == trainer.TOTAL_TRAINABLE_PARAMETERS, "within trainable count changed")
    _require(int(row["adapter_parameters_per_arm"]) == trainer.ADAPTER_PARAMETERS, "adapter count changed")
    _require(str(row["source_shared_manifest_sha256"]) == source_shared["manifest_sha256"], "contract source changed")
    _require(str(row["gate_aggregate_manifest_sha256"]) == gate["manifest_sha256"], "contract gate changed")
    for column in (
        "placement_gate_pass",
        "identical_rows",
        "identical_head_initialization",
        "identical_adapter_initialization",
        "identical_batch_order",
        "identical_shared_epoch",
        "identical_optimizer_schedule",
        "identical_trainable_parameter_count",
    ):
        _require(trainer._as_bool(row[column], column), "placement contract failed {}".format(column))
    parity = float(row["cross_within_epoch0_max_abs_probability_difference"])
    _require(
        math.isfinite(parity) and 0.0 <= parity <= trainer.EPOCH0_PLACEMENT_PARITY_ATOL,
        "placement epoch-0 parity failed",
    )
    return contract


def _audit_training(training, library, seed, epoch, source_shared, gate):
    _require(len(training) == 2 and set(training["arm"].astype(str)) == set(ARMS), "training arms changed")
    _require(
        set(trainer._strict_integer_values(training["training_seed"], "training seed"))
        == {seed},
        "training seed changed",
    )
    _require(
        set(
            trainer._strict_integer_values(
                training["selected_epoch"], "training selected epoch"
            )
        )
        == {epoch},
        "training epoch changed",
    )
    for column in (
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
        "head_initialization_sha256",
        "normalized_adapter_initialization_sha256",
        "training_order_sha256",
        "source_shared_manifest_sha256",
        "gate_aggregate_manifest_sha256",
    ):
        _require(training[column].nunique(dropna=False) == 1, "paired training differs in {}".format(column))
    contract = matched.PRIMARY_CONTRACT[library]
    first = training.iloc[0]
    _require(
        int(first["rows"]) == contract["rows"]
        and int(first["positive"]) == contract["positive"]
        and int(first["negative"]) == contract["negative"],
        "training class counts changed",
    )
    _require(str(first["membership_sha256"]) == contract["membership_sha256"], "training membership changed")
    _require(int(first["run_seed"]) == matched.derived_seed(seed, "final_refit", -1), "training run seed changed")
    _require(set(pd.to_numeric(training["trainable_parameters"], errors="raise").astype(int)) == {trainer.TOTAL_TRAINABLE_PARAMETERS}, "total trainable counts differ")
    _require(set(pd.to_numeric(training["adapter_parameters"], errors="raise").astype(int)) == {trainer.ADAPTER_PARAMETERS}, "adapter counts differ")
    _require(str(first["source_shared_manifest_sha256"]) == source_shared["manifest_sha256"], "training source changed")
    _require(str(first["gate_aggregate_manifest_sha256"]) == gate["manifest_sha256"], "training gate changed")
    source_cross = source_shared["training_audit"].loc[
        source_shared["training_audit"]["arm"].eq("lora_cross")
    ]
    _require(len(source_cross) == 1, "verified source lacks one cross training audit")
    source_cross = source_cross.iloc[0]
    observed_cross = training.loc[training["arm"].eq("lora_cross")]
    _require(len(observed_cross) == 1, "placement audit lacks one reused cross row")
    observed_cross = observed_cross.iloc[0]
    _require(
        set(source_cross.index).issubset(observed_cross.index),
        "reused cross training audit schema changed",
    )
    for column in source_cross.index:
        _require(
            _same(observed_cross[column], source_cross[column]),
            "reused cross training audit {} changed".format(column),
        )
    expected_sources = {
        "lora_cross": "reused_verified_shared_epoch",
        "lora_within": "new_parameter_matched_refit",
    }
    expected_placements = dict(trainer.PLACEMENTS)
    for arm in ARMS:
        row = training.loc[training["arm"].eq(arm)]
        _require(len(row) == 1, "missing {} training row".format(arm))
        row = row.iloc[0]
        _require(str(row["arm_source"]) == expected_sources[arm], "{} source changed".format(arm))
        _require(str(row["attention_placement"]) == expected_placements[arm], "{} placement changed".format(arm))
        names = set(json.loads(str(row["trainable_names"])))
        _require(names == trainer.expected_trainable_names(arm), "{} trainable names changed".format(arm))
        _require(float(row["head_parameter_delta_l2"]) > 0.0, "{} head did not move".format(arm))
        _require(float(row["adapter_parameter_delta_l2"]) > 0.0, "{} adapter did not move".format(arm))
    return training


def _audit_pairing_diagnostics(manifest, contract, training, source_shared):
    diagnostics = manifest.get("pairing_diagnostics")
    expected_keys = {
        "training_order_sha256",
        "run_seed",
        "max_abs_probability_difference",
        "head_initialization_sha256",
        "cross_raw_adapter_initialization_sha256",
        "within_raw_adapter_initialization_sha256",
        "normalized_adapter_initialization_sha256",
        "trainable",
    }
    _require(
        isinstance(diagnostics, dict) and set(diagnostics) == expected_keys,
        "placement pairing diagnostics changed",
    )
    contract_row = contract.iloc[0]
    first_training = training.iloc[0]
    _require(
        _strict_integer_scalar(diagnostics["run_seed"], "pairing run seed")
        == _strict_integer_scalar(first_training["run_seed"], "training run seed"),
        "placement pairing run seed changed",
    )
    _require(
        _same(
            diagnostics["max_abs_probability_difference"],
            contract_row["cross_within_epoch0_max_abs_probability_difference"],
        ),
        "placement pairing probability parity changed",
    )
    _require(
        str(diagnostics["head_initialization_sha256"])
        == str(first_training["head_initialization_sha256"]),
        "placement pairing head initialization changed",
    )
    _require(
        str(diagnostics["training_order_sha256"])
        == str(first_training["training_order_sha256"]),
        "placement pairing training order changed",
    )
    _require(
        str(diagnostics["normalized_adapter_initialization_sha256"])
        == str(first_training["normalized_adapter_initialization_sha256"])
        == str(contract_row["normalized_adapter_initialization_sha256"]),
        "placement normalized adapter initialization changed",
    )
    source_raw = source_shared["manifest"].get("refit_diagnostics", {}).get(
        "live_adapter_initialization_sha256"
    )
    _require(
        str(diagnostics["cross_raw_adapter_initialization_sha256"])
        == str(source_raw),
        "placement cross adapter initialization differs from verified source",
    )
    for key in (
        "training_order_sha256",
        "head_initialization_sha256",
        "cross_raw_adapter_initialization_sha256",
        "within_raw_adapter_initialization_sha256",
        "normalized_adapter_initialization_sha256",
    ):
        _require(
            trainer.shared_aggregate._is_sha256(str(diagnostics[key])),
            "placement pairing {} is malformed".format(key),
        )
    trainable = diagnostics["trainable"]
    _require(isinstance(trainable, dict) and set(trainable) == set(ARMS), "placement pairing trainable arms changed")
    for arm in ARMS:
        record = trainable[arm]
        _require(
            isinstance(record, dict)
            and set(record) == {"names", "count", "adapter_count"}
            and set(record["names"]) == trainer.expected_trainable_names(arm)
            and int(record["count"]) == trainer.TOTAL_TRAINABLE_PARAMETERS
            and int(record["adapter_count"]) == trainer.ADAPTER_PARAMETERS,
            "{} pairing trainable contract changed".format(arm),
        )


def load_verified_run(run_dir):
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "placement run directory is missing")
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "placement run lacks manifest.json")
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == trainer.SCHEMA_VERSION, "placement schema changed")
    _require(
        manifest.get("analysis_status") == "retrospective_exploratory_retention_gated",
        "placement status changed",
    )
    library, seed, configuration = _validate_configuration(manifest)
    outputs = _verify_output_set(run_dir, manifest, seed)

    code = manifest.get("code", {})
    expected_code = {
        "script": Path(trainer.__file__).resolve(),
        "shared_aggregator": Path(trainer.shared_aggregate.__file__).resolve(),
        "shared_trainer": Path(trainer.shared_trainer.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    _require(set(code) == set(expected_code), "placement code provenance changed")
    for name, path in expected_code.items():
        record = code[name]
        _require(Path(record["path"]).resolve() == path, "{} code path changed".format(name))
        _require(sha256_file(path) == str(record["sha256"]), "{} code hash changed".format(name))

    source_record = manifest.get("source_shared_run", {})
    source_dir = Path(source_record.get("path", "")).resolve()
    source_shared = trainer.shared_aggregate.load_verified_run(source_dir)
    _require(source_shared["library"] == library and int(source_shared["training_seed"]) == seed, "source shared identity changed")
    _require(source_shared["manifest_sha256"] == str(source_record.get("manifest_sha256")), "source shared hash changed")
    _audit_configuration_against_source(configuration, source_shared)
    _audit_runtime_against_source(manifest, source_shared)
    epoch = _strict_integer_scalar(
        configuration["shared_positive_epoch"], "placement shared positive epoch"
    )
    _require(
        _strict_integer_scalar(
            source_shared["configuration"]["chosen_positive_epoch"],
            "verified source chosen epoch",
        )
        == epoch,
        "source shared epoch changed",
    )

    gate_record = manifest.get("gate_aggregate", {})
    gate = trainer.load_verified_gate_aggregate(
        Path(gate_record.get("path", "")).resolve(), source_shared
    )
    _require(gate["manifest_sha256"] == str(gate_record.get("manifest_sha256")), "gate aggregate hash changed")

    contract = pd.read_csv(run_dir / "placement_contract.csv", float_precision="round_trip")
    contract = _audit_contract(contract, library, seed, epoch, source_shared, gate)
    training = pd.read_csv(run_dir / "training_audit.csv", float_precision="round_trip")
    training = _audit_training(training, library, seed, epoch, source_shared, gate)
    _audit_pairing_diagnostics(manifest, contract, training, source_shared)
    predictions = pd.read_csv(run_dir / "retention_predictions.csv", float_precision="round_trip")
    _require(len(predictions) == 2 * matched.PRIMARY_CONTRACT[library]["retention_rows"], "prediction count changed")
    _require(set(predictions["arm"].astype(str)) == set(ARMS), "prediction arms changed")
    required_prediction_columns = {
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
    _require(
        required_prediction_columns.issubset(predictions.columns),
        "prediction schema changed",
    )
    _require(
        not bool(predictions.duplicated(["arm", "pair_uid"]).any()),
        "duplicate placement prediction",
    )
    _audit_paired_prediction_panel(predictions)
    probability = pd.to_numeric(predictions["probability"], errors="raise").to_numpy(dtype=float)
    _require(
        bool(np.isfinite(probability).all())
        and bool(((probability >= 0.0) & (probability <= 1.0)).all()),
        "placement probability is invalid",
    )
    reused = predictions.loc[predictions["arm"].eq("lora_cross")].sort_values(
        "pair_uid", kind="mergesort"
    ).reset_index(drop=True)
    source_reused = source_shared["predictions"].loc[
        source_shared["predictions"]["arm"].eq("lora_cross")
    ].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    for column in (
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
        "probability",
    ):
        for observed, expected in zip(reused[column], source_reused[column]):
            _require(_same(observed, expected), "reused cross prediction {} changed".format(column))
    metric_rows = []
    for arm in ARMS:
        block = predictions.loc[predictions["arm"].eq(arm)].copy()
        _require(set(block["library"].astype(str)) == {library}, "prediction library changed")
        _require(
            set(
                trainer._strict_integer_values(
                    block["training_seed"], "prediction training seed"
                )
            )
            == {seed},
            "prediction seed changed",
        )
        _require(
            set(
                trainer._strict_integer_values(
                    block["selected_epoch"], "prediction selected epoch"
                )
            )
            == {epoch},
            "prediction epoch changed",
        )
        metric_rows.append(_canonical_metrics(block, library, arm, seed, epoch))
    metrics = pd.DataFrame(metric_rows)
    stored_metrics = pd.read_csv(run_dir / "retention_metrics.csv", float_precision="round_trip")
    stored_metrics = stored_metrics.sort_values("arm").reset_index(drop=True)
    expected_metrics = metrics.sort_values("arm").reset_index(drop=True)
    _require(set(stored_metrics.columns) == set(expected_metrics.columns), "metric schema changed")
    for column in expected_metrics.columns:
        for observed, expected in zip(stored_metrics[column], expected_metrics[column]):
            _require(_same(observed, expected), "stored metric {} changed".format(column))
    paired = trainer.paired_difference_record(metrics, library, seed, epoch)
    stored_paired = pd.read_csv(run_dir / "paired_differences.csv", float_precision="round_trip")
    _require(len(stored_paired) == 1, "paired-difference count changed")
    _compare_record(stored_paired.iloc[0].to_dict(), paired, "paired difference")

    delta = manifest.get("model_delta", {})
    filename = "model_delta_seed{}_lora_within.pt".format(seed)
    expected_delta = {
        "filename": filename,
        "sha256": outputs[filename]["sha256"],
        "arm": "lora_within",
        "attention_placement": "self_attn",
        "training_seed": seed,
        "selected_epoch": epoch,
    }
    _require(delta == expected_delta, "within model-delta metadata changed")
    return {
        "run_dir": run_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "library": library,
        "training_seed": seed,
        "selected_epoch": epoch,
        "configuration": configuration,
        "outputs": outputs,
        "source_shared": source_shared,
        "gate": gate,
        "contract": contract,
        "training": training,
        "predictions": predictions,
        "metrics": metrics,
        "paired": pd.DataFrame([paired]),
    }


def expected_passed_libraries(gate):
    table = gate.get("gate_table")
    _require(isinstance(table, pd.DataFrame), "verified gate table is missing")
    _require(
        len(table) == len(matched.LIBRARIES)
        and not bool(table["library"].astype(str).duplicated().any())
        and set(table["library"].astype(str)) == set(matched.LIBRARIES),
        "verified gate library grid changed",
    )
    if "n_seeds" in table.columns:
        _require(
            set(pd.to_numeric(table["n_seeds"], errors="raise").astype(int))
            == {len(matched.DEFAULT_TRAINING_SEEDS)},
            "verified gate seed count changed",
        )
    return {
        str(row["library"])
        for _, row in table.iterrows()
        if trainer._as_bool(row["placement_gate_pass"], "placement_gate_pass")
    }


def validate_run_grid(runs):
    _require(runs, "no placement-control runs supplied")
    expected_seeds = tuple(int(seed) for seed in matched.DEFAULT_TRAINING_SEEDS)
    _require(
        len(expected_seeds) == 3 and len(set(expected_seeds)) == 3,
        "prespecified placement seed contract changed",
    )
    gate_hashes = {run["gate"]["manifest_sha256"] for run in runs}
    gate_paths = {run["gate"]["run_dir"] for run in runs}
    _require(len(gate_hashes) == 1 and len(gate_paths) == 1, "runs use different gate aggregates")
    passed = expected_passed_libraries(runs[0]["gate"])
    observed = {run["library"] for run in runs}
    _require(observed == passed, "run libraries differ from the libraries that passed the gate")
    identities = {(run["library"], run["training_seed"]) for run in runs}
    expected = {
        (library, int(seed))
        for library in passed
        for seed in expected_seeds
    }
    _require(identities == expected and len(runs) == len(expected), "placement run library/seed grid is incomplete")
    for library in passed:
        library_seeds = [
            int(run["training_seed"])
            for run in runs
            if run["library"] == library
        ]
        _require(
            len(library_seeds) == len(expected_seeds)
            and len(set(library_seeds)) == len(expected_seeds)
            and set(library_seeds) == set(expected_seeds),
            "{} placement runs do not contain exactly the three prespecified seeds".format(
                library
            ),
        )
    return passed


def build_summary(metrics, paired):
    rows = []
    for library in sorted(set(metrics["library"].astype(str))):
        for arm in ARMS:
            block = metrics.loc[
                metrics["library"].eq(library) & metrics["arm"].eq(arm)
            ].sort_values("training_seed")
            _require(len(block) == 3, "{} {} lacks three seeds".format(library, arm))
            row = {
                "library": library,
                "result_type": "model_score",
                "result": arm,
                "reference": "",
                "n_seed_runs": 3,
                "training_seeds": ",".join(str(int(value)) for value in block["training_seed"]),
                "selected_epochs": ",".join(str(int(value)) for value in block["selected_epoch"]),
            }
            for metric in METRICS:
                values = pd.to_numeric(block[metric], errors="raise").to_numpy(dtype=float)
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_sample_sd"] = float(np.std(values, ddof=1))
            rows.append(row)
        changes = paired.loc[paired["library"].eq(library)].sort_values("training_seed")
        _require(len(changes) == 3, "{} paired placement changes lack three seeds".format(library))
        row = {
            "library": library,
            "result_type": "paired_change",
            "result": "lora_within",
            "reference": "lora_cross",
            "n_seed_runs": 3,
            "training_seeds": ",".join(str(int(value)) for value in changes["training_seed"]),
            "selected_epochs": ",".join(str(int(value)) for value in changes["selected_epoch"]),
        }
        for metric in METRICS:
            values = pd.to_numeric(
                changes[metric + "_change"], errors="raise"
            ).to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["library", "result_type", "result"], kind="mergesort"
    ).reset_index(drop=True)


def summary_markdown(summary):
    lines = [
        "# MINT LoRA placement control",
        "",
        "The table compares the verified cross-chain LoRA arm with a parameter-matched within-chain LoRA arm. Values are mean +- sample SD across the three prespecified optimization seeds.",
    ]
    labels = {"lora_cross": "Cross-chain LoRA", "lora_within": "Within-chain LoRA"}
    for library in sorted(set(summary["library"].astype(str))):
        lines.extend(
            [
                "",
                "## {}".format(library),
                "",
                "| Result | AUROC | AP | Global Spearman | Within-peptide Spearman |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        model = summary.loc[
            summary["library"].eq(library)
            & summary["result_type"].eq("model_score")
        ].set_index("result")
        for arm in ARMS:
            row = model.loc[arm]
            values = [
                "{:.3f}+-{:.3f}".format(
                    row[metric + "_mean"], row[metric + "_sample_sd"]
                )
                for metric in METRICS
            ]
            lines.append("| {} | {} | {} | {} | {} |".format(labels[arm], *values))
        change = summary.loc[
            summary["library"].eq(library)
            & summary["result_type"].eq("paired_change")
        ].iloc[0]
        values = [
            "{:.3f}+-{:.3f}".format(
                change[metric + "_mean"], change[metric + "_sample_sd"]
            )
            for metric in METRICS
        ]
        lines.append("| Within - cross | {} | {} | {} | {} |".format(*values))
    lines.extend(
        [
            "",
            "A negative within-minus-cross value favors cross-chain placement. Because measured retention determined whether this control was run, the placement result is exploratory. Seed SD measures optimization variability, not biological uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def _recheck(runs, code_hashes):
    code_paths = {
        "aggregate": Path(__file__).resolve(),
        "placement_trainer": Path(trainer.__file__).resolve(),
        "shared_aggregator": Path(trainer.shared_aggregate.__file__).resolve(),
        "shared_trainer": Path(trainer.shared_trainer.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    for name, path in code_paths.items():
        _require(sha256_file(path) == code_hashes[name], "{} code changed".format(name))
    source_runs = []
    seen_sources = set()
    for run in runs:
        _require(sha256_file(run["manifest_path"]) == run["manifest_sha256"], "placement manifest changed")
        for record in run["outputs"].values():
            _require(sha256_file(record["path"]) == record["sha256"], "placement output changed")
        source = run["source_shared"]
        source_identity = (source["run_dir"], source["manifest_sha256"])
        if source_identity not in seen_sources:
            seen_sources.add(source_identity)
            source_runs.append(source)
        gate = run["gate"]
        _require(
            sha256_file(gate["manifest_path"]) == gate["manifest_sha256"],
            "gate aggregate manifest changed",
        )
        for record in gate["verified_outputs"].values():
            _require(
                sha256_file(record["path"]) == record["sha256"],
                "gate aggregate output changed",
            )
        for gate_source in gate["verified_source_runs"]:
            source_identity = (
                gate_source["run_dir"],
                gate_source["manifest_sha256"],
            )
            if source_identity not in seen_sources:
                seen_sources.add(source_identity)
                source_runs.append(gate_source)
    trainer.shared_aggregate._recheck_verified_inputs(source_runs)


def _matched_source_protected_paths(source):
    """Protect a matched run, its files, and each sharded-cache inventory."""
    paths = [Path(source["run_dir"]).resolve()]
    reference = source.get("reference", {})
    if isinstance(reference, dict) and "directory" in reference:
        paths.append(Path(reference["directory"]).resolve())
    for name, record in source.get("sources", {}).items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = Path(record["path"]).resolve()
        paths.append(path)
        if str(name).startswith("cache_"):
            paths.append(path.parent)
    return paths


def run(args):
    started = time.time()
    code_paths = {
        "aggregate": Path(__file__).resolve(),
        "placement_trainer": Path(trainer.__file__).resolve(),
        "shared_aggregator": Path(trainer.shared_aggregate.__file__).resolve(),
        "shared_trainer": Path(trainer.shared_trainer.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    requested_output = Path(os.path.abspath(str(args.output_dir)))
    _require(
        not os.path.lexists(str(requested_output)),
        "output directory exists; refusing overwrite",
    )
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not os.path.lexists(str(output_dir)), "output directory exists; refusing overwrite")
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    _require(len(run_dirs) == len(set(run_dirs)), "duplicate placement input directory")
    runs = [load_verified_run(path) for path in run_dirs]
    protected = list(run_dirs)
    protected.extend(run["source_shared"]["run_dir"] for run in runs)
    protected.extend(run["gate"]["run_dir"] for run in runs)
    for run in runs:
        protected.extend(
            _matched_source_protected_paths(run["source_shared"]["source"])
        )
    for gate_source in runs[0]["gate"]["verified_source_runs"]:
        protected.append(gate_source["run_dir"])
        protected.extend(_matched_source_protected_paths(gate_source["source"]))
    trainer.shared_aggregate._assert_output_disjoint(output_dir, protected)
    passed = validate_run_grid(runs)
    predictions = pd.concat([run["predictions"] for run in runs], ignore_index=True)
    metrics = pd.concat([run["metrics"] for run in runs], ignore_index=True)
    paired = pd.concat([run["paired"] for run in runs], ignore_index=True)
    contracts = pd.concat([run["contract"] for run in runs], ignore_index=True)
    training = pd.concat([run["training"] for run in runs], ignore_index=True)
    summary = build_summary(metrics, paired)
    markdown = summary_markdown(summary)
    _recheck(runs, code_hashes)

    tables = {
        "retention_predictions_by_seed.csv": predictions,
        "recomputed_retention_metrics_by_seed.csv": metrics,
        "paired_within_minus_cross_by_seed.csv": paired,
        "placement_contract_by_seed.csv": contracts,
        "training_audit_by_seed.csv": training,
        "mean_sample_sd_summary.csv": summary,
    }

    def build_manifest(output_records):
        # Recheck after all output payloads have been staged and immediately
        # before constructing the commit manifest.  A mutable authenticated
        # input must therefore fail the run instead of being silently paired
        # with results computed from its earlier contents.
        _recheck(runs, code_hashes)
        trainer.shared_aggregate._assert_output_disjoint(output_dir, protected)
        _require(
            not os.path.lexists(str(output_dir)),
            "output directory appeared before manifest construction",
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "analysis_status": "retrospective_exploratory_retention_gated",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "configuration": {
                "passed_libraries": sorted(passed),
                "training_seeds": list(matched.DEFAULT_TRAINING_SEEDS),
                "arms": list(ARMS),
                "paired_comparison": "lora_within_minus_lora_cross",
                "seed_aggregation": "arithmetic mean and sample SD (ddof=1)",
                "primary_endpoint": "within_peptide_macro_spearman",
                "retention_in_gate": True,
            },
            "source_runs": [
                {
                    "path": str(run["run_dir"]),
                    "manifest_path": str(run["manifest_path"]),
                    "manifest_sha256": run["manifest_sha256"],
                    "library": run["library"],
                    "training_seed": run["training_seed"],
                    "selected_epoch": run["selected_epoch"],
                }
                for run in sorted(runs, key=lambda item: (item["library"], item["training_seed"]))
            ],
            "gate_aggregate": {
                "path": str(runs[0]["gate"]["run_dir"]),
                "manifest_path": str(runs[0]["gate"]["manifest_path"]),
                "manifest_sha256": runs[0]["gate"]["manifest_sha256"],
            },
            "rows": {
                "source_runs": len(runs),
                "prediction_records": len(predictions),
                "metric_records": len(metrics),
                "paired_records": len(paired),
                "summary_records": len(summary),
            },
            "outputs": output_records,
            "code": {
                "path": str(code_paths["aggregate"]),
                "sha256": code_hashes["aggregate"],
            },
            "dependencies": {
                name: {"path": str(path), "sha256": code_hashes[name]}
                for name, path in sorted(code_paths.items())
                if name != "aggregate"
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "permissions": {"directory": "0700", "files": "0600"},
        }

    def precommit_recheck():
        _recheck(runs, code_hashes)
        trainer.shared_aggregate._assert_output_disjoint(output_dir, protected)

    manifest = trainer._atomic_publish_run(
        output_dir,
        tables,
        "summary.md",
        markdown,
        build_manifest,
        precommit=precommit_recheck,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "passed_libraries": sorted(passed),
                "source_runs": len(runs),
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
        help="all three seed artifacts for every library that passed the gate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
