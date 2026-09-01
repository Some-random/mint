#!/usr/bin/env python
"""Audit and aggregate the six one-seed matched MINT/LoRA runs.

The expected inputs are exactly LibA and LibB crossed with the three
prespecified training seeds.  Each input directory was produced by
``finetune_mint_selection_matched.py`` with one training seed and contains a
deterministic frozen-logistic reference plus paired head-only and LoRA fits.

This program fails closed before writing anything.  It verifies every output
and source hash recorded by each run, enforces the matched experiment
contract, and recomputes every stored retention metric from
``retention_predictions.csv`` with the canonical evaluator.  The compact
summary reports seed means and sample standard deviations; the frozen model is
represented once per library because it has no training-seed randomness.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
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
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "mint-selection-matched-aggregate-v1"
TRAINER_SCHEMA_VERSION = matched.SCHEMA_VERSION
EXPECTED_LIBRARIES = tuple(matched.LIBRARIES)
EXPECTED_TRAINING_SEEDS = tuple(matched.DEFAULT_TRAINING_SEEDS)
TRAINED_ARMS = tuple(matched.ARMS)
ALL_ARMS = ("frozen_logistic",) + TRAINED_ARMS
METRIC_METADATA = ("library", "arm", "training_seed", "selected_epoch")
SUMMARY_METRICS = (
    "global_auroc",
    "global_auprc",
    "global_spearman",
    "within_peptide_macro_spearman",
)
FIXED_OUTPUTS = {
    "c_validation.csv",
    "fold_membership.csv",
    "weak_validation_predictions.csv",
    "weak_validation_history.csv",
    "epoch_selection.csv",
    "training_audit.csv",
    "retention_predictions.csv",
    "retention_metrics.csv",
    "aggregate_metrics.csv",
    "paired_differences.csv",
    "per_peptide_metrics.csv",
    "run_summary.md",
}
BASE_SOURCE_NAMES = {
    "script",
    "cache_rows",
    "cache_rows_manifest",
    "retention_features",
    "retention_features_manifest",
    "checkpoint",
    "config",
    "cached_evaluator",
    "practical_evaluator",
    "historical_lora_dependency",
    "retention_lora_dependency",
    "pair_collator_dependency",
    "mint_wrapper_dependency",
    "mint_esm2_dependency",
    "mint_modules_dependency",
    "mint_attention_dependency",
    "mint_data_dependency",
    "mint_rotary_dependency",
    "reference_manifest_json",
    "reference_conditions_csv",
    "reference_weak_validation_csv",
    "reference_metrics_csv",
}
REFERENCE_SOURCE_NAMES = {
    "reference_manifest_json",
    "reference_conditions_csv",
    "reference_weak_validation_csv",
    "reference_metrics_csv",
}
CODE_SOURCE_PATHS = {
    "script": Path(matched.__file__).resolve(),
    "cached_evaluator": Path(cached_eval.__file__).resolve(),
    "practical_evaluator": REPO_ROOT
    / "downstream/AffibodyMHC/evaluate_cached_weak_mint_practical.py",
    "historical_lora_dependency": REPO_ROOT
    / "downstream/AffibodyMHC/finetune_mint_selection.py",
    "retention_lora_dependency": REPO_ROOT
    / "downstream/AffibodyMHC/finetune_mint_retention.py",
    "pair_collator_dependency": REPO_ROOT
    / "downstream/AffibodyMHC/extract_mint_features.py",
    "mint_wrapper_dependency": REPO_ROOT / "mint/helpers/extract.py",
    "mint_esm2_dependency": REPO_ROOT / "mint/model/esm2.py",
    "mint_modules_dependency": REPO_ROOT / "mint/modules.py",
    "mint_attention_dependency": REPO_ROOT / "mint/multihead_attention.py",
    "mint_data_dependency": REPO_ROOT / "mint/data.py",
    "mint_rotary_dependency": REPO_ROOT / "mint/rotary_embedding.py",
}
_VERIFICATION_HASH_CACHE = {}
_CACHE_AUDIT_ROWS = {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "JSON root is not an object: {}".format(path))
    return payload


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _live_sha256(path):
    """Avoid hashing shared multi-GB sources six times during one audit."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _VERIFICATION_HASH_CACHE:
        _VERIFICATION_HASH_CACHE[key] = sha256_file(path)
    return _VERIFICATION_HASH_CACHE[key]


def _read_cache_audit_rows(path):
    path = Path(path).resolve()
    key = (str(path), _live_sha256(path))
    if key not in _CACHE_AUDIT_ROWS:
        columns = [
            "source_kind",
            "library",
            "pair_uid",
            "weak_label",
            "measurement_missing",
            "target_retention",
            "target_binder",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
        ]
        frame = pd.read_csv(
            path,
            usecols=columns,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
        )
        _require(
            not bool(frame["pair_uid"].duplicated().any()),
            "cache rows have duplicate pair UIDs",
        )
        _CACHE_AUDIT_ROWS[key] = frame
    return _CACHE_AUDIT_ROWS[key]


def _assert_columns(frame, required, label):
    missing = set(required).difference(frame.columns)
    _require(not missing, "{} lacks columns {}".format(label, sorted(missing)))


def _assert_unique(frame, columns, label):
    _assert_columns(frame, columns, label)
    duplicate = frame.duplicated(list(columns), keep=False)
    if bool(duplicate.any()):
        example = frame.loc[duplicate, list(columns)].head(3).to_dict(orient="records")
        raise ValueError("{} has duplicate keys: {}".format(label, example))


def _normalized_json(value):
    """Normalize tuples and integer dictionary keys through JSON encoding."""
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
    return _same_number(observed, expected)


def _verified_recorded_file(record, expected_path, label):
    _require(isinstance(record, dict), "{} record is malformed".format(label))
    _require(set(("path", "sha256")).issubset(record), "{} record is incomplete".format(label))
    expected_path = Path(expected_path).resolve()
    recorded_path = Path(record["path"]).resolve()
    _require(recorded_path == expected_path, "{} path differs from its run directory".format(label))
    _require(expected_path.is_file(), "{} is missing: {}".format(label, expected_path))
    observed_hash = sha256_file(expected_path)
    _require(observed_hash == record["sha256"], "{} hash mismatch".format(label))
    return {"path": str(expected_path), "sha256": observed_hash}


def _verified_source(record, label, expected_path=None):
    _require(isinstance(record, dict), "{} source record is malformed".format(label))
    _require(set(("path", "sha256")).issubset(record), "{} source record is incomplete".format(label))
    path = Path(record["path"]).resolve()
    _require(path.is_file(), "{} source is missing: {}".format(label, path))
    if expected_path is not None:
        _require(path == Path(expected_path).resolve(), "{} points to the wrong dependency".format(label))
    observed_hash = _live_sha256(path)
    _require(observed_hash == record["sha256"], "{} source hash mismatch".format(label))
    return {"path": str(path), "sha256": observed_hash}


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
    _require(isinstance(seeds, list) and len(seeds) == 1, "{} is not a one-seed run".format(run_name))
    training_seed = int(seeds[0])
    _require(training_seed in EXPECTED_TRAINING_SEEDS, "{} has an unexpected training seed".format(run_name))
    expected = {
        "regime": matched.REGIME,
        "cleaning": matched.CLEANING,
        "balance": matched.BALANCE,
        "folds": matched.CANONICAL_FOLDS,
        "split_seed": matched.CANONICAL_SPLIT_SEED,
        "c_grid": list(matched.CANONICAL_C_GRID),
        "epoch_candidates": [0, 1, 2, 3],
        "arms": list(ALL_ARMS),
        "retention_threshold_for_auroc_ap": 75.0,
        "head_lr": 1e-4,
        "adapter_lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "clip_norm": 1.0,
    }
    for key, value in expected.items():
        _require(
            _normalized_json(configuration.get(key)) == _normalized_json(value),
            "{} configuration {} violates the matched contract".format(run_name, key),
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
    }.items():
        _require(
            _normalized_json(lora.get(key)) == _normalized_json(value),
            "{} LoRA {} differs from the matched contract".format(run_name, key),
        )
    usage = configuration.get("retention_usage")
    _require(isinstance(usage, dict), "{} lacks the retention-use boundary".format(run_name))
    _require(
        "final metrics only" in str(usage.get("numeric_retention_and_75_percent_labels", "")),
        "{} does not state that retention outcomes are final-metric-only".format(run_name),
    )
    _require(
        _normalized_json(manifest.get("canonical_contract"))
        == _normalized_json(contract),
        "{} canonical primary contract changed".format(run_name),
    )
    return library, training_seed, configuration


def _validate_sources(manifest, library, run_name):
    sources = manifest.get("sources")
    _require(isinstance(sources, dict), "{} has no source records".format(run_name))
    missing = BASE_SOURCE_NAMES.difference(sources)
    _require(not missing, "{} lacks source records {}".format(run_name, sorted(missing)))
    cache_npz = sorted(name for name in sources if name.startswith("cache_npz_"))
    cache_manifest = sorted(name for name in sources if name.startswith("cache_manifest_"))
    _require(cache_npz, "{} has no feature-cache source".format(run_name))
    expected_npz = ["cache_npz_{:03d}".format(i) for i in range(len(cache_npz))]
    expected_manifest = ["cache_manifest_{:03d}".format(i) for i in range(len(cache_npz))]
    _require(cache_npz == expected_npz, "{} cache NPZ numbering is not contiguous".format(run_name))
    _require(cache_manifest == expected_manifest, "{} cache manifest coverage differs".format(run_name))
    allowed = BASE_SOURCE_NAMES | set(cache_npz) | set(cache_manifest)
    _require(set(sources) == allowed, "{} has unexpected or missing source names".format(run_name))

    verified = {}
    for name, record in sorted(sources.items()):
        verified[name] = _verified_source(
            record,
            "{} {}".format(run_name, name),
            expected_path=CODE_SOURCE_PATHS.get(name),
        )

    cache_sources = manifest.get("cache_sources")
    _require(
        isinstance(cache_sources, list) and len(cache_sources) == len(cache_npz),
        "{} cache_sources coverage differs from sources".format(run_name),
    )
    for index, record in enumerate(cache_sources):
        checked = _verified_source(record, "{} cache_sources {}".format(run_name, index))
        expected = verified["cache_npz_{:03d}".format(index)]
        _require(checked == expected, "{} duplicate cache source record disagrees".format(run_name))
        feature_manifest = _read_json(
            verified["cache_manifest_{:03d}".format(index)]["path"]
        )
        _require(
            feature_manifest.get("schema_version")
            == "mint-weak-feature-cache-v1"
            and feature_manifest.get("output", {}).get("sha256")
            == expected["sha256"],
            "{} feature-cache manifest {} does not bind its NPZ".format(
                run_name, index
            ),
        )
        feature_contract = feature_manifest.get("contract", {})
        contract_payload = feature_contract.get("payload")
        _require(
            isinstance(contract_payload, dict)
            and hashlib.sha256(
                json.dumps(
                    contract_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            == feature_contract.get("sha256"),
            "{} feature-cache contract {} is malformed".format(
                run_name, index
            ),
        )
        contract_bindings = {
            "checkpoint_sha256": verified["checkpoint"]["sha256"],
            "config_sha256": verified["config"]["sha256"],
            "rows_sha256": verified["cache_rows"]["sha256"],
            "prepare_manifest_sha256": verified["cache_rows_manifest"][
                "sha256"
            ],
            "pair_collator_source_sha256": verified[
                "pair_collator_dependency"
            ]["sha256"],
        }
        for key, expected_hash in contract_bindings.items():
            _require(
                contract_payload.get(key) == expected_hash,
                "{} feature-cache contract {} does not bind {}".format(
                    run_name, index, key
                ),
            )

    cache_row_manifest = _read_json(verified["cache_rows_manifest"]["path"])
    _require(
        cache_row_manifest.get("output", {})
        .get("rows_csv", {})
        .get("sha256")
        == verified["cache_rows"]["sha256"],
        "{} cache-row manifest does not bind cache_rows".format(run_name),
    )
    retention_manifest = _read_json(
        verified["retention_features_manifest"]["path"]
    )
    _require(
        retention_manifest.get("output", {}).get("sha256")
        == verified["retention_features"]["sha256"],
        "{} retention-feature manifest does not bind its NPZ".format(run_name),
    )

    reference = manifest.get("reference_primary_run")
    _require(isinstance(reference, dict), "{} has no primary reference record".format(run_name))
    _require(
        float(reference.get("selected_c")) == float(matched.PRIMARY_CONTRACT[library]["selected_c"]),
        "{} primary reference selected C changed".format(run_name),
    )
    reference_bindings = {
        "reference_manifest_json": "manifest_sha256",
        "reference_conditions_csv": "conditions_sha256",
        "reference_weak_validation_csv": "weak_validation_sha256",
        "reference_metrics_csv": "metrics_sha256",
    }
    reference_directory = Path(reference.get("directory", "")).resolve()
    _require(reference_directory.is_dir(), "{} primary reference directory is missing".format(run_name))
    for source_name, reference_key in reference_bindings.items():
        _require(
            verified[source_name]["sha256"] == reference.get(reference_key),
            "{} {} disagrees with reference_primary_run".format(run_name, source_name),
        )
        _require(
            Path(verified[source_name]["path"]).parent == reference_directory,
            "{} {} is outside the recorded reference directory".format(run_name, source_name),
        )
    revalidated_reference = matched.validate_reference_run(
        reference_directory, library
    )
    _require(
        _normalized_json(revalidated_reference) == _normalized_json(reference),
        "{} primary reference record does not reproduce".format(run_name),
    )
    return verified, reference


def _read_output_table(run_dir, name, run_name):
    path = run_dir / name
    frame = pd.read_csv(path, float_precision="round_trip")
    _require(not frame.empty, "{} {} is empty".format(run_name, name))
    return frame


def _canonical_retention_metrics(predictions, library, arm, training_seed, selected_epoch):
    frame = predictions.reset_index(drop=True)
    score = pd.to_numeric(frame["probability"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(score).all()), "retention probabilities are non-finite")
    _require(bool(((score >= 0.0) & (score <= 1.0)).all()), "retention probabilities are outside [0,1]")
    canonical = cached_eval.ranking_metrics(frame, score)
    return {
        "library": library,
        "arm": arm,
        "training_seed": int(training_seed),
        "selected_epoch": int(selected_epoch),
        **canonical,
    }


def audit_retention_tables(predictions, stored_metrics, library, training_seed, run_name):
    """Recompute and compare every canonical metric in one run."""
    required_prediction = {
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
    _assert_columns(predictions, required_prediction, "{} retention predictions".format(run_name))
    _assert_unique(
        predictions,
        ("arm", "training_seed", "sequence_pair_sha256"),
        "{} retention predictions".format(run_name),
    )
    binder = pd.to_numeric(predictions["target_binder"], errors="raise").to_numpy(
        dtype=float
    )
    retention_value = pd.to_numeric(
        predictions["target_retention"], errors="raise"
    ).to_numpy(dtype=float)
    _require(
        bool(np.isfinite(binder).all()) and set(binder.tolist()) == {0.0, 1.0},
        "{} retention binder labels are not binary".format(run_name),
    )
    _require(
        bool(np.isfinite(retention_value).all())
        and bool(((retention_value >= 0.0) & (retention_value <= 100.0)).all()),
        "{} retention values are outside [0,100]".format(run_name),
    )
    _require(
        np.array_equal(binder.astype(int), (retention_value >= 75.0).astype(int)),
        "{} binder labels do not equal retention >= 75".format(run_name),
    )
    _assert_columns(stored_metrics, METRIC_METADATA, "{} retention metrics".format(run_name))
    _assert_unique(stored_metrics, ("arm", "training_seed"), "{} retention metrics".format(run_name))

    observed_groups = set(
        predictions[["arm", "training_seed"]]
        .assign(training_seed=lambda frame: pd.to_numeric(frame["training_seed"], errors="raise").astype(int))
        .itertuples(index=False, name=None)
    )
    expected_groups = {
        ("frozen_logistic", -1),
        ("head_only", int(training_seed)),
        ("lora_cross", int(training_seed)),
    }
    _require(observed_groups == expected_groups, "{} retention prediction arms/seeds differ".format(run_name))
    stored_groups = set(
        stored_metrics[["arm", "training_seed"]]
        .assign(training_seed=lambda frame: pd.to_numeric(frame["training_seed"], errors="raise").astype(int))
        .itertuples(index=False, name=None)
    )
    _require(stored_groups == expected_groups, "{} stored metric arms/seeds differ".format(run_name))

    panel_columns = (
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        "target_retention",
        "target_binder",
    )
    reference_panel = None
    recomputed = []
    canonical_metric_names = None
    for arm, seed in sorted(expected_groups):
        block = predictions.loc[
            predictions["arm"].eq(arm)
            & pd.to_numeric(predictions["training_seed"], errors="raise").eq(seed)
        ].copy()
        _require(bool(block["library"].eq(library).all()), "{} prediction library changed".format(run_name))
        _require(block["selected_epoch"].nunique() == 1, "{} {} has multiple selected epochs".format(run_name, arm))
        selected_epoch = int(block["selected_epoch"].iloc[0])
        _require(0 <= selected_epoch <= 3, "{} {} selected epoch is outside 0--3".format(run_name, arm))
        if arm == "frozen_logistic":
            _require(selected_epoch == 0, "{} frozen model has a nonzero epoch".format(run_name))
        # Preserve trainer row order for P@k tie-breaking.  Only the panel copy
        # is sorted for order-independent equality checks across arms.
        block = block.reset_index(drop=True)
        panel = (
            block.loc[:, panel_columns]
            .sort_values("sequence_pair_sha256", kind="mergesort")
            .reset_index(drop=True)
        )
        if reference_panel is None:
            reference_panel = panel
        else:
            _require(panel.equals(reference_panel), "{} retention target panel changes across arms".format(run_name))
        row = _canonical_retention_metrics(block, library, arm, seed, selected_epoch)
        metric_names = set(row).difference(METRIC_METADATA)
        if canonical_metric_names is None:
            canonical_metric_names = metric_names
        else:
            _require(metric_names == canonical_metric_names, "canonical metric schema changed within run")
        stored = stored_metrics.loc[
            stored_metrics["arm"].eq(arm)
            & pd.to_numeric(stored_metrics["training_seed"], errors="raise").eq(seed)
        ]
        _require(len(stored) == 1, "{} lacks one stored metric for {}".format(run_name, arm))
        stored = stored.iloc[0]
        _require(str(stored["library"]) == library, "{} stored metric library changed".format(run_name))
        _require(int(stored["selected_epoch"]) == selected_epoch, "{} stored selected epoch changed".format(run_name))
        _require(
            set(stored_metrics.columns) == set(METRIC_METADATA) | metric_names,
            "{} stored retention metric schema differs from the canonical evaluator".format(run_name),
        )
        for metric_name in sorted(metric_names):
            _require(
                _same_number(stored[metric_name], row[metric_name]),
                "{} {} stored {} differs from direct recomputation".format(run_name, arm, metric_name),
            )
        recomputed.append(row)

    contract = matched.PRIMARY_CONTRACT[library]
    _require(len(reference_panel) == int(contract["retention_rows"]), "{} retention row count changed".format(run_name))
    positives = int(pd.to_numeric(reference_panel["target_binder"], errors="raise").sum())
    _require(positives == int(contract["retention_positive"]), "{} retention positive count changed".format(run_name))
    _require(
        cached_eval.membership_sha256(reference_panel) == contract["retention_membership_sha256"],
        "{} retention membership changed".format(run_name),
    )
    return pd.DataFrame(recomputed), reference_panel


def _validate_training_tables(run_dir, manifest, library, training_seed, run_name):
    epoch = _read_output_table(run_dir, "epoch_selection.csv", run_name)
    _assert_columns(epoch, ("library", "arm", "training_seed", "epoch", "selected"), "{} epoch selection".format(run_name))
    _require(len(epoch) == 8, "{} epoch selection must have 2 arms x 4 epochs".format(run_name))
    _require(set(epoch["arm"].astype(str)) == set(TRAINED_ARMS), "{} epoch-selection arms changed".format(run_name))
    _require(bool(epoch["library"].eq(library).all()), "{} epoch-selection library changed".format(run_name))
    _require(set(pd.to_numeric(epoch["training_seed"], errors="raise").astype(int)) == {training_seed}, "{} epoch-selection seed changed".format(run_name))
    for arm, group in epoch.groupby("arm"):
        _require(set(pd.to_numeric(group["epoch"], errors="raise").astype(int)) == {0, 1, 2, 3}, "{} {} epoch candidates changed".format(run_name, arm))
        _require(int(pd.to_numeric(group["selected"], errors="raise").sum()) == 1, "{} {} does not select exactly one epoch".format(run_name, arm))

    audit = _read_output_table(run_dir, "training_audit.csv", run_name)
    required = (
        "stage",
        "arm",
        "training_seed",
        "fold",
        "selected_epoch",
        "run_seed",
        "head_parameter_delta_l2",
        "head_parameter_delta_max_abs",
        "adapter_parameter_delta_l2",
        "adapter_parameter_delta_max_abs",
    )
    _assert_columns(audit, required, "{} training audit".format(run_name))
    _require(len(audit) == 8, "{} training audit must have 6 CV and 2 final records".format(run_name))
    _require(set(audit["arm"].astype(str)) == set(TRAINED_ARMS), "{} training-audit arms changed".format(run_name))
    _require(set(pd.to_numeric(audit["training_seed"], errors="raise").astype(int)) == {training_seed}, "{} training-audit seed changed".format(run_name))
    cross = audit.loc[audit["stage"].eq("cross_validation")]
    final = audit.loc[audit["stage"].eq("final_refit")]
    _require(len(cross) == 6 and len(final) == 2, "{} training stages are incomplete".format(run_name))
    for _, audit_row in audit.iterrows():
        expected_run_seed = matched.derived_seed(
            training_seed,
            str(audit_row["stage"]),
            int(audit_row["fold"]),
        )
        _require(
            int(audit_row["run_seed"]) == expected_run_seed,
            "{} has an incorrect derived run seed for {} fold {}".format(
                run_name, audit_row["stage"], audit_row["fold"]
            ),
        )
    for fold, group in cross.groupby("fold"):
        _require(int(fold) in {0, 1, 2} and set(group["arm"]) == set(TRAINED_ARMS), "{} CV fold coverage changed".format(run_name))
        _require(group["run_seed"].nunique() == 1, "{} arm-specific randomness was used in fold {}".format(run_name, fold))
    _require(set(pd.to_numeric(final["fold"], errors="raise").astype(int)) == {-1}, "{} final-refit fold sentinel changed".format(run_name))
    _require(final["run_seed"].nunique() == 1, "{} final arms used different random seeds".format(run_name))
    matched_audit_columns = (
        "rows",
        "positive",
        "negative",
        "class_weight_negative",
        "class_weight_positive",
        "membership_sha256",
        "feature_mean_sha256",
        "feature_scale_sha256",
        "run_seed",
        "schedule_total_steps",
    )
    _assert_columns(
        audit,
        matched_audit_columns + ("trainable_parameters",),
        "{} training audit".format(run_name),
    )
    for (stage, fold), group in audit.groupby(["stage", "fold"], sort=True):
        _require(
            set(group["arm"].astype(str)) == set(TRAINED_ARMS),
            "{} {} fold {} lacks a paired optimizer arm".format(run_name, stage, fold),
        )
        for column in matched_audit_columns:
            _require(
                group[column].nunique(dropna=False) == 1,
                "{} paired arms differ in {} for {} fold {}".format(
                    run_name, column, stage, fold
                ),
            )
    expected_trainable = {"head_only": 2561, "lora_cross": 23041}
    for arm, count in expected_trainable.items():
        observed = set(
            pd.to_numeric(
                audit.loc[audit["arm"].eq(arm), "trainable_parameters"],
                errors="raise",
            ).astype(int)
        )
        _require(
            observed == {count},
            "{} {} trainable-parameter count changed".format(run_name, arm),
        )
    parity_error = pd.to_numeric(
        audit["epoch0_probability_max_abs_error"], errors="raise"
    ).to_numpy(dtype=float)
    _require(
        bool(np.isfinite(parity_error).all())
        and bool(((parity_error >= 0.0) & (parity_error <= 0.01)).all()),
        "{} live/cached epoch-zero parity exceeds 0.01".format(run_name),
    )
    head_adapter = pd.to_numeric(audit.loc[audit["arm"].eq("head_only"), "adapter_parameter_delta_max_abs"], errors="raise")
    _require(bool(head_adapter.eq(0.0).all()), "{} head-only arm changed LoRA adapters".format(run_name))
    for column in (
        "head_parameter_delta_l2",
        "head_parameter_delta_max_abs",
        "adapter_parameter_delta_l2",
        "adapter_parameter_delta_max_abs",
    ):
        delta = pd.to_numeric(audit[column], errors="raise").to_numpy(dtype=float)
        _require(
            bool(np.isfinite(delta).all()) and bool((delta >= 0.0).all()),
            "{} {} contains an invalid parameter delta".format(run_name, column),
        )

    predictions = _read_output_table(run_dir, "retention_predictions.csv", run_name)
    metrics = _read_output_table(run_dir, "retention_metrics.csv", run_name)
    recomputed, panel = audit_retention_tables(predictions, metrics, library, training_seed, run_name)
    cache_audit_rows = _read_cache_audit_rows(
        manifest["sources"]["cache_rows"]["path"]
    )
    expected_retention = cache_audit_rows.loc[
        cache_audit_rows["source_kind"].eq("retention")
        & cache_audit_rows["library"].eq(library)
        & cache_audit_rows["measurement_missing"].eq("0"),
        [
            "pair_uid",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "target_retention",
            "target_binder",
        ],
    ].copy()
    expected_retention = expected_retention.sort_values(
        "sequence_pair_sha256", kind="mergesort"
    ).reset_index(drop=True)
    observed_retention = panel.sort_values(
        "sequence_pair_sha256", kind="mergesort"
    ).reset_index(drop=True)
    _require(
        len(expected_retention) == len(observed_retention),
        "{} measured retention panel differs from the cache source".format(run_name),
    )
    for column in (
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ):
        _require(
            expected_retention[column]
            .astype(str)
            .equals(observed_retention[column].astype(str)),
            "{} retention {} differs from the cache source".format(
                run_name, column
            ),
        )
    _require(
        np.array_equal(
            pd.to_numeric(
                expected_retention["target_retention"], errors="raise"
            ).to_numpy(dtype=float),
            pd.to_numeric(
                observed_retention["target_retention"], errors="raise"
            ).to_numpy(dtype=float),
        )
        and np.array_equal(
            pd.to_numeric(
                expected_retention["target_binder"], errors="raise"
            ).to_numpy(dtype=int),
            pd.to_numeric(
                observed_retention["target_binder"], errors="raise"
            ).to_numpy(dtype=int),
        ),
        "{} retention outcomes differ from the cache source".format(run_name),
    )
    selected_by_arm = {
        str(row["arm"]): int(row["epoch"])
        for _, row in epoch.loc[pd.to_numeric(epoch["selected"], errors="raise").eq(1)].iterrows()
    }
    for arm in TRAINED_ARMS:
        metric_epoch = int(recomputed.loc[recomputed["arm"].eq(arm), "selected_epoch"].iloc[0])
        _require(metric_epoch == selected_by_arm[arm], "{} {} final epoch differs from weak validation".format(run_name, arm))
        final_epoch = int(final.loc[final["arm"].eq(arm), "selected_epoch"].iloc[0])
        _require(final_epoch == selected_by_arm[arm], "{} {} audit final epoch differs".format(run_name, arm))
    lora_final = final.loc[final["arm"].eq("lora_cross")].iloc[0]
    if int(lora_final["selected_epoch"]) > 0:
        _require(float(lora_final["adapter_parameter_delta_l2"]) > 0.0, "{} trained LoRA adapter did not move".format(run_name))
    else:
        _require(
            float(lora_final["adapter_parameter_delta_max_abs"]) == 0.0,
            "{} epoch-zero LoRA final state changed an adapter".format(run_name),
        )
    _require(
        bool(
            pd.to_numeric(
                cross.loc[
                    cross["arm"].eq("lora_cross"),
                    "adapter_parameter_delta_l2",
                ],
                errors="raise",
            ).gt(0.0).all()
        ),
        "{} cross-validation LoRA adapters did not move".format(run_name),
    )

    expected_validation_rows = sum(
        int(fold["validation"])
        for fold in matched.PRIMARY_CONTRACT[library]["folds"].values()
    )
    cardinalities = {
        "c_validation.csv": 4 * (matched.CANONICAL_FOLDS + 1),
        "fold_membership.csv": matched.CANONICAL_FOLDS
        * matched.PRIMARY_CONTRACT[library]["rows"],
        "weak_validation_predictions.csv": len(TRAINED_ARMS)
        * 4
        * expected_validation_rows,
        "weak_validation_history.csv": len(TRAINED_ARMS)
        * matched.CANONICAL_FOLDS
        * 4,
        "epoch_selection.csv": 8,
        "training_audit.csv": 8,
        "retention_predictions.csv": 3
        * matched.PRIMARY_CONTRACT[library]["retention_rows"],
        "retention_metrics.csv": 3,
        "aggregate_metrics.csv": 3,
        "paired_differences.csv": 3,
        "per_peptide_metrics.csv": 3 * int(panel["chain1_sha256"].nunique()),
    }
    already_loaded = {
        "epoch_selection.csv": epoch,
        "training_audit.csv": audit,
        "retention_predictions.csv": predictions,
        "retention_metrics.csv": metrics,
    }
    for filename, expected_count in cardinalities.items():
        frame = already_loaded.get(filename)
        if frame is None:
            frame = _read_output_table(run_dir, filename, run_name)
        _require(
            len(frame) == int(expected_count),
            "{} {} row count changed ({} != {})".format(
                run_name, filename, len(frame), int(expected_count)
            ),
        )

    c_validation = _read_output_table(run_dir, "c_validation.csv", run_name)
    c_aggregate = c_validation.loc[c_validation["record_type"].eq("aggregate")]
    selected_aggregate = c_aggregate.loc[
        pd.to_numeric(c_aggregate["selected"], errors="raise").eq(1)
    ]
    _require(len(selected_aggregate) == 1, "{} must select one aggregate C".format(run_name))
    _require(
        float(selected_aggregate.iloc[0]["C"])
        == float(matched.PRIMARY_CONTRACT[library]["selected_c"]),
        "{} C validation selected a different C".format(run_name),
    )

    fold_membership = _read_output_table(
        run_dir, "fold_membership.csv", run_name
    )
    _assert_columns(
        fold_membership,
        ("pair_uid", "chain1_sha256", "chain2_sha256", "weak_label", "fold", "role"),
        "{} fold membership".format(run_name),
    )
    _assert_unique(
        fold_membership,
        ("fold", "pair_uid"),
        "{} fold membership".format(run_name),
    )
    # The trainer's compact fold table omits sequence_pair_sha256, while the
    # frozen-primary golden hashes use that global identity.  Recover it from
    # the already hash-verified cache-row source and also bind every retained
    # label/partner identity to that source.
    cache_identity = cache_audit_rows.loc[
        cache_audit_rows["source_kind"].eq("weak")
        & cache_audit_rows["library"].eq(library),
        [
            "pair_uid",
            "sequence_pair_sha256",
            "chain1_sha256",
            "chain2_sha256",
            "weak_label",
        ],
    ].rename(
        columns={
            "chain1_sha256": "source_chain1_sha256",
            "chain2_sha256": "source_chain2_sha256",
            "weak_label": "source_weak_label",
        }
    )
    fold_membership = fold_membership.merge(
        cache_identity,
        on="pair_uid",
        how="left",
        validate="many_to_one",
    )
    _require(
        not bool(fold_membership["sequence_pair_sha256"].isna().any())
        and bool(fold_membership["sequence_pair_sha256"].ne("").all()),
        "{} fold membership cannot be mapped to global pair identities".format(
            run_name
        ),
    )
    _require(
        fold_membership["chain1_sha256"]
        .astype(str)
        .eq(fold_membership["source_chain1_sha256"].astype(str))
        .all()
        and fold_membership["chain2_sha256"]
        .astype(str)
        .eq(fold_membership["source_chain2_sha256"].astype(str))
        .all()
        and np.array_equal(
            pd.to_numeric(
                fold_membership["weak_label"], errors="raise"
            ).to_numpy(dtype=int),
            pd.to_numeric(
                fold_membership["source_weak_label"], errors="raise"
            ).to_numpy(dtype=int),
        ),
        "{} fold labels or partner identities differ from cache rows".format(
            run_name
        ),
    )
    contract = matched.PRIMARY_CONTRACT[library]
    reference_fold_pairs = None
    for fold, fold_contract in contract["folds"].items():
        fold_rows = fold_membership.loc[
            pd.to_numeric(fold_membership["fold"], errors="raise").eq(int(fold))
        ].copy()
        _require(len(fold_rows) == int(contract["rows"]), "{} fold {} does not cover the primary pool".format(run_name, fold))
        fold_pairs = set(fold_rows["pair_uid"].astype(str))
        if reference_fold_pairs is None:
            reference_fold_pairs = fold_pairs
        else:
            _require(fold_pairs == reference_fold_pairs, "{} primary pool changes across folds".format(run_name))
        role_counts = fold_rows["role"].value_counts().to_dict()
        for role in ("train", "guard", "validation"):
            _require(
                int(role_counts.get(role, 0)) == int(fold_contract[role]),
                "{} fold {} {} count changed".format(run_name, fold, role),
            )
        train = fold_rows.loc[fold_rows["role"].eq("train")]
        validation = fold_rows.loc[fold_rows["role"].eq("validation")]
        _require(
            int(pd.to_numeric(validation["weak_label"], errors="raise").sum())
            == int(fold_contract["validation_positive"]),
            "{} fold {} validation-positive count changed".format(run_name, fold),
        )
        _require(
            cached_eval.membership_sha256(train) == fold_contract["train_sha256"],
            "{} fold {} train membership changed".format(run_name, fold),
        )
        _require(
            cached_eval.membership_sha256(validation)
            == fold_contract["validation_sha256"],
            "{} fold {} validation membership changed".format(run_name, fold),
        )
        _require(
            set(train["chain1_sha256"].astype(str)).isdisjoint(
                set(validation["chain1_sha256"].astype(str))
            ),
            "{} fold {} shares peptide identities".format(run_name, fold),
        )
        _require(
            set(train["chain2_sha256"].astype(str)).isdisjoint(
                set(validation["chain2_sha256"].astype(str))
            ),
            "{} fold {} shares Affibody identities".format(run_name, fold),
        )
        audit_fold = cross.loc[
            pd.to_numeric(cross["fold"], errors="raise").eq(int(fold))
        ]
        _require(
            set(audit_fold["membership_sha256"].astype(str))
            == {str(fold_contract["train_sha256"])},
            "{} fold {} training audit membership changed".format(run_name, fold),
        )
        train_labels = pd.to_numeric(train["weak_label"], errors="raise").astype(int)
        expected_weights = matched.balanced_class_weights(train_labels.to_numpy())
        for _, audit_row in audit_fold.iterrows():
            _require(
                int(audit_row["rows"]) == len(train)
                and int(audit_row["positive"]) == int(train_labels.sum())
                and int(audit_row["negative"])
                == int((train_labels == 0).sum()),
                "{} fold {} training-audit class counts changed".format(
                    run_name, fold
                ),
            )
            _require(
                _same_number(
                    audit_row["class_weight_negative"], expected_weights[0]
                )
                and _same_number(
                    audit_row["class_weight_positive"], expected_weights[1]
                ),
                "{} fold {} training-audit class weights changed".format(
                    run_name, fold
                ),
            )
    primary_once = fold_membership.loc[
        pd.to_numeric(fold_membership["fold"], errors="raise").eq(0)
    ]
    _require(
        cached_eval.membership_sha256(primary_once) == contract["membership_sha256"],
        "{} full primary membership changed".format(run_name),
    )
    primary_labels = pd.to_numeric(
        primary_once["weak_label"], errors="raise"
    ).astype(int)
    primary_weights = matched.balanced_class_weights(primary_labels.to_numpy())
    for _, audit_row in final.iterrows():
        _require(
            str(audit_row["membership_sha256"]) == contract["membership_sha256"],
            "{} final training-audit membership changed".format(run_name),
        )
        _require(
            int(audit_row["rows"]) == int(contract["rows"])
            and int(audit_row["positive"]) == int(contract["positive"])
            and int(audit_row["negative"]) == int(contract["negative"]),
            "{} final training-audit class counts changed".format(run_name),
        )
        _require(
            _same_number(
                audit_row["class_weight_negative"], primary_weights[0]
            )
            and _same_number(
                audit_row["class_weight_positive"], primary_weights[1]
            ),
            "{} final training-audit class weights changed".format(run_name),
        )
    _require(
        set(primary_once["chain1_sha256"].astype(str)).isdisjoint(
            set(panel["chain1_sha256"].astype(str))
        ),
        "{} primary pool shares a peptide identity with retention".format(run_name),
    )
    _require(
        set(primary_once["chain2_sha256"].astype(str)).isdisjoint(
            set(panel["chain2_sha256"].astype(str))
        ),
        "{} primary pool shares an Affibody identity with retention".format(run_name),
    )

    weak_predictions = _read_output_table(
        run_dir, "weak_validation_predictions.csv", run_name
    )
    _assert_columns(
        weak_predictions,
        (
            "library",
            "arm",
            "training_seed",
            "fold",
            "epoch",
            "pair_uid",
            "weak_label",
            "probability",
        ),
        "{} weak validation predictions".format(run_name),
    )
    _require(
        bool(weak_predictions["library"].eq(library).all()),
        "{} weak-validation library changed".format(run_name),
    )
    _require(
        set(weak_predictions["arm"].astype(str)) == set(TRAINED_ARMS),
        "{} weak-validation arms changed".format(run_name),
    )
    _require(
        set(
            pd.to_numeric(
                weak_predictions["training_seed"], errors="raise"
            ).astype(int)
        )
        == {training_seed},
        "{} weak-validation seed changed".format(run_name),
    )
    _assert_unique(
        weak_predictions,
        ("arm", "training_seed", "fold", "epoch", "pair_uid"),
        "{} weak validation predictions".format(run_name),
    )
    for fold, fold_contract in matched.PRIMARY_CONTRACT[library]["folds"].items():
        expected_validation = fold_membership.loc[
            pd.to_numeric(fold_membership["fold"], errors="raise").eq(int(fold))
            & fold_membership["role"].eq("validation"),
            ["pair_uid", "weak_label"],
        ].copy()
        expected_label_by_pair = dict(
            zip(
                expected_validation["pair_uid"].astype(str),
                pd.to_numeric(
                    expected_validation["weak_label"], errors="raise"
                ).astype(int),
            )
        )
        for arm in TRAINED_ARMS:
            for candidate_epoch in range(4):
                observed_validation = weak_predictions.loc[
                        weak_predictions["arm"].eq(arm)
                        & pd.to_numeric(weak_predictions["fold"], errors="raise").eq(
                            int(fold)
                        )
                        & pd.to_numeric(
                            weak_predictions["epoch"], errors="raise"
                        ).eq(candidate_epoch)
                    ].copy()
                count = len(observed_validation)
                _require(
                    count == int(fold_contract["validation"]),
                    "{} weak-validation fold/arm/epoch coverage changed".format(
                        run_name
                    ),
                )
                observed_pairs = set(
                    observed_validation["pair_uid"].astype(str)
                )
                _require(
                    observed_pairs == set(expected_label_by_pair),
                    "{} weak-validation membership changed for fold/arm/epoch".format(
                        run_name
                    ),
                )
                observed_labels = dict(
                    zip(
                        observed_validation["pair_uid"].astype(str),
                        pd.to_numeric(
                            observed_validation["weak_label"], errors="raise"
                        ).astype(int),
                    )
                )
                _require(
                    observed_labels == expected_label_by_pair,
                    "{} weak-validation labels changed for fold/arm/epoch".format(
                        run_name
                    ),
                )

    # Reapply the pooled-row epoch selection and all documented tie-breaks.
    for arm in TRAINED_ARMS:
        selected_epoch, selection_rows = matched.select_epoch(
            weak_predictions, arm, training_seed, 3
        )
        stored_arm = epoch.loc[epoch["arm"].eq(arm)].copy()
        for expected_row in selection_rows:
            stored_row = stored_arm.loc[
                pd.to_numeric(stored_arm["epoch"], errors="raise").eq(
                    int(expected_row["epoch"])
                )
            ]
            _require(
                len(stored_row) == 1,
                "{} {} lacks epoch {} selection metrics".format(
                    run_name, arm, expected_row["epoch"]
                ),
            )
            stored_row = stored_row.iloc[0]
            for key, expected_value in expected_row.items():
                agrees = key in stored_row.index and _same_value(
                    stored_row[key], expected_value
                )
                _require(
                    agrees,
                    "{} {} epoch {} {} differs from recomputation".format(
                        run_name, arm, expected_row["epoch"], key
                    ),
                )
        _require(
            selected_epoch == selected_by_arm[arm],
            "{} {} selected epoch differs from pooled predictions".format(
                run_name, arm
            ),
        )

    # Per-peptide records are secondary outputs, but verify them rather than
    # merely trusting their manifest hash.
    per_peptide_stored = _read_output_table(
        run_dir, "per_peptide_metrics.csv", run_name
    )
    per_peptide_expected = []
    for arm, seed in (
        ("frozen_logistic", -1),
        ("head_only", training_seed),
        ("lora_cross", training_seed),
    ):
        block = predictions.loc[
            predictions["arm"].eq(arm)
            & pd.to_numeric(predictions["training_seed"], errors="raise").eq(seed)
        ].reset_index(drop=True)
        selected = int(block["selected_epoch"].iloc[0])
        per_peptide_expected.extend(
            matched.per_peptide_records(
                block,
                pd.to_numeric(block["probability"], errors="raise").to_numpy(
                    dtype=float
                ),
                library,
                arm,
                seed,
                selected,
            )
        )
    per_peptide_expected = pd.DataFrame(per_peptide_expected)
    key_columns = ("arm", "training_seed", "peptide_identity")
    _assert_unique(
        per_peptide_stored,
        key_columns,
        "{} stored per-peptide metrics".format(run_name),
    )
    _require(
        set(per_peptide_stored.columns) == set(per_peptide_expected.columns),
        "{} per-peptide metric schema changed".format(run_name),
    )
    expected_indexed = per_peptide_expected.set_index(list(key_columns)).sort_index()
    stored_indexed = per_peptide_stored.set_index(list(key_columns)).sort_index()
    _require(
        list(expected_indexed.index) == list(stored_indexed.index),
        "{} per-peptide metric membership changed".format(run_name),
    )
    for key in expected_indexed.index:
        for column in expected_indexed.columns:
            expected_value = expected_indexed.loc[key, column]
            stored_value = stored_indexed.loc[key, column]
            equal = _same_value(stored_value, expected_value)
            _require(
                equal,
                "{} per-peptide {} {} differs from recomputation".format(
                    run_name, key, column
                ),
            )

    rows = manifest.get("rows")
    contract = matched.PRIMARY_CONTRACT[library]
    _require(isinstance(rows, dict), "{} has no row-count audit".format(run_name))
    expected_rows = {
        "primary": contract["rows"],
        "primary_positive": contract["positive"],
        "primary_negative": contract["negative"],
        "retention": contract["retention_rows"],
        "retention_positive": contract["retention_positive"],
        "final_prediction_records": 3 * contract["retention_rows"],
    }
    for key, value in expected_rows.items():
        _require(int(rows.get(key, -1)) == int(value), "{} manifest row count {} changed".format(run_name, key))
    _require(
        int(rows.get("validation_prediction_records", -1))
        == cardinalities["weak_validation_predictions.csv"],
        "{} manifest weak-validation prediction count changed".format(run_name),
    )
    retention_feature_error = float(
        manifest.get("retention_feature_max_abs_difference", float("inf"))
    )
    _require(
        math.isfinite(retention_feature_error)
        and 0.0 <= retention_feature_error <= 1e-4,
        "{} live/cached retention feature parity exceeds 1e-4".format(run_name),
    )
    return recomputed, panel, predictions


def load_verified_run(run_dir):
    """Verify one complete one-seed trainer directory and recompute its metrics."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "matched run directory is missing: {}".format(run_dir))
    run_name = run_dir.name
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "{} has no manifest.json".format(run_name))
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == TRAINER_SCHEMA_VERSION, "{} trainer schema changed".format(run_name))
    _require(manifest.get("analysis_status") == "retrospective_exploratory", "{} analysis status changed".format(run_name))
    library, training_seed, configuration = _validate_configuration(manifest, run_name)
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "{} has no runtime record".format(run_name))
    runtime_fields = {
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
    _require(
        runtime_fields.issubset(runtime),
        "{} runtime record is incomplete".format(run_name),
    )

    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "{} has no output records".format(run_name))
    expected_outputs = _expected_outputs(training_seed)
    _require(set(outputs) == expected_outputs, "{} output set is incomplete or unexpected".format(run_name))
    for filename, record in sorted(outputs.items()):
        _require(Path(filename).name == filename, "{} contains a non-local output name".format(run_name))
        _verified_recorded_file(record, run_dir / filename, "{} {}".format(run_name, filename))

    sources, reference = _validate_sources(manifest, library, run_name)
    recomputed, panel, predictions = _validate_training_tables(
        run_dir, manifest, library, training_seed, run_name
    )
    frozen = recomputed.loc[recomputed["arm"].eq("frozen_logistic")].iloc[0]
    for metric_name in SUMMARY_METRICS:
        _require(
            _same_number(
                frozen[metric_name],
                reference["metrics"][metric_name],
                atol=1e-10,
                rtol=1e-10,
            ),
            "{} frozen {} differs from its primary reference".format(run_name, metric_name),
        )
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "library": library,
        "training_seed": training_seed,
        "configuration": configuration,
        "runtime_contract": {
            key: runtime[key] for key in sorted(runtime_fields)
        },
        "sources": sources,
        "reference": reference,
        "metrics": recomputed,
        "panel": panel,
        "predictions": predictions,
    }


def _source_hash_contract(run, reference_sources):
    return {
        name: record["sha256"]
        for name, record in sorted(run["sources"].items())
        if (name in REFERENCE_SOURCE_NAMES) == bool(reference_sources)
    }


def _configuration_contract(configuration):
    output = copy.deepcopy(configuration)
    for key in ("library", "training_seeds", "selected_C"):
        output.pop(key, None)
    return _normalized_json(output)


def combine_verified_runs(verified_runs):
    """Enforce the exact 2-library x 3-seed experiment and join audited metrics."""
    expected_keys = {
        (library, seed)
        for library in EXPECTED_LIBRARIES
        for seed in EXPECTED_TRAINING_SEEDS
    }
    _require(len(verified_runs) == len(expected_keys), "exactly six one-seed runs are required")
    observed_keys = [(run["library"], int(run["training_seed"])) for run in verified_runs]
    _require(len(observed_keys) == len(set(observed_keys)), "duplicate library/training-seed run")
    _require(set(observed_keys) == expected_keys, "run directories do not cover LibA/LibB x the three expected seeds")
    reference_configuration = _configuration_contract(verified_runs[0]["configuration"])
    reference_common_sources = _source_hash_contract(verified_runs[0], False)
    for run in verified_runs[1:]:
        _require(
            _configuration_contract(run["configuration"]) == reference_configuration,
            "{} used different matched-training hyperparameters".format(run["run_name"]),
        )
        _require(
            _source_hash_contract(run, False) == reference_common_sources,
            "{} used different data/model/code sources".format(run["run_name"]),
        )
        _require(
            run["runtime_contract"] == verified_runs[0]["runtime_contract"],
            "{} used a different GPU/software runtime".format(run["run_name"]),
        )
    library_reference_sources = {}
    for library in EXPECTED_LIBRARIES:
        group = [run for run in verified_runs if run["library"] == library]
        source_contract = _source_hash_contract(group[0], True)
        reference_record = _normalized_json(group[0]["reference"])
        for run in group[1:]:
            _require(
                _source_hash_contract(run, True) == source_contract,
                "{} used a different {} frozen reference".format(run["run_name"], library),
            )
            _require(
                _normalized_json(run["reference"]) == reference_record,
                "{} primary-reference record differs within {}".format(run["run_name"], library),
            )
        library_reference_sources[library] = source_contract

    metric_blocks = []
    for library in EXPECTED_LIBRARIES:
        group = sorted(
            (run for run in verified_runs if run["library"] == library),
            key=lambda run: run["training_seed"],
        )
        reference_panel = group[0]["panel"].sort_values("sequence_pair_sha256").reset_index(drop=True)
        reference_frozen = group[0]["predictions"].loc[
            group[0]["predictions"]["arm"].eq("frozen_logistic")
        ].sort_values("sequence_pair_sha256").reset_index(drop=True)
        for run in group[1:]:
            panel = run["panel"].sort_values("sequence_pair_sha256").reset_index(drop=True)
            _require(panel.equals(reference_panel), "{} retention panel differs within {}".format(run["run_name"], library))
            frozen = run["predictions"].loc[
                run["predictions"]["arm"].eq("frozen_logistic")
            ].sort_values("sequence_pair_sha256").reset_index(drop=True)
            _require(
                np.allclose(
                    pd.to_numeric(frozen["probability"], errors="raise"),
                    pd.to_numeric(reference_frozen["probability"], errors="raise"),
                    rtol=1e-10,
                    atol=1e-10,
                ),
                "{} frozen predictions differ within {}".format(run["run_name"], library),
            )
        frozen_metric = group[0]["metrics"].loc[
            group[0]["metrics"]["arm"].eq("frozen_logistic")
        ].copy()
        frozen_metric.insert(0, "source_run", "{}_frozen".format(library.lower()))
        metric_blocks.append(frozen_metric)
        for run in group:
            trained = run["metrics"].loc[run["metrics"]["arm"].isin(TRAINED_ARMS)].copy()
            trained.insert(
                0,
                "source_run",
                "{}_seed{}".format(library.lower(), int(run["training_seed"])),
            )
            metric_blocks.append(trained)
    metrics = pd.concat(metric_blocks, ignore_index=True, sort=False)
    _assert_unique(metrics, ("library", "arm", "training_seed"), "combined recomputed metrics")
    return {
        "metrics": metrics.sort_values(["library", "arm", "training_seed"], kind="mergesort").reset_index(drop=True),
        "common_source_hashes": reference_common_sources,
        "library_reference_source_hashes": library_reference_sources,
        "configuration_contract": reference_configuration,
        "runtime_contract": verified_runs[0]["runtime_contract"],
    }


def build_paired_changes(metrics):
    """Calculate comparisons within each seed before any aggregation."""
    rows = []
    for library in EXPECTED_LIBRARIES:
        library_rows = metrics.loc[metrics["library"].eq(library)]
        frozen = library_rows.loc[library_rows["arm"].eq("frozen_logistic")]
        _require(len(frozen) == 1, "{} must have one deterministic frozen row".format(library))
        frozen = frozen.iloc[0]
        for training_seed in EXPECTED_TRAINING_SEEDS:
            seeded = library_rows.loc[pd.to_numeric(library_rows["training_seed"], errors="raise").eq(training_seed)]
            by_arm = seeded.set_index("arm")
            _require(set(by_arm.index) == set(TRAINED_ARMS), "{} seed {} lacks a paired arm".format(library, training_seed))
            comparisons = (
                ("head_only", "frozen_logistic", frozen),
                ("lora_cross", "frozen_logistic", frozen),
                ("lora_cross", "head_only", by_arm.loc["head_only"]),
            )
            for arm, reference_arm, reference in comparisons:
                row = {
                    "library": library,
                    "arm": arm,
                    "reference_arm": reference_arm,
                    "comparison": "{}_minus_{}".format(arm, reference_arm),
                    "training_seed": int(training_seed),
                    "selected_epoch": int(by_arm.loc[arm, "selected_epoch"]),
                    "reference_selected_epoch": int(reference["selected_epoch"]),
                }
                for metric in SUMMARY_METRICS:
                    row[metric + "_change"] = float(by_arm.loc[arm, metric] - reference[metric])
                rows.append(row)
    output = pd.DataFrame(rows)
    _assert_unique(output, ("library", "comparison", "training_seed"), "paired changes")
    return output.sort_values(["library", "comparison", "training_seed"], kind="mergesort").reset_index(drop=True)


def build_combined_summary(metrics, paired_changes):
    """Return one compact score/change table with means and sample SDs."""
    rows = []
    for (library, arm), group in metrics.groupby(["library", "arm"], sort=True):
        seeds = sorted(pd.to_numeric(group["training_seed"], errors="raise").astype(int))
        row = {
            "library": library,
            "result_type": "model_score",
            "result": arm,
            "reference": "",
            "n_seed_runs": int(len(group)),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "selected_epochs": ",".join(
                str(value)
                for value in pd.to_numeric(
                    group.sort_values("training_seed")["selected_epoch"],
                    errors="raise",
                ).astype(int)
            ),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
        rows.append(row)
    for (library, comparison), group in paired_changes.groupby(["library", "comparison"], sort=True):
        first = group.iloc[0]
        seeds = sorted(pd.to_numeric(group["training_seed"], errors="raise").astype(int))
        row = {
            "library": library,
            "result_type": "paired_change",
            "result": str(first["arm"]),
            "reference": str(first["reference_arm"]),
            "n_seed_runs": int(len(group)),
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "selected_epochs": ",".join(
                "{}/{}".format(int(row["selected_epoch"]), int(row["reference_selected_epoch"]))
                for _, row in group.sort_values("training_seed").iterrows()
            ),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric + "_change"], errors="raise").to_numpy(dtype=float)
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sample_sd"] = float(np.std(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["library", "result_type", "result", "reference"], kind="mergesort"
    ).reset_index(drop=True)


def _metric_text(row, metric):
    mean = float(row[metric + "_mean"])
    sample_sd = float(row[metric + "_sample_sd"])
    if math.isfinite(sample_sd):
        return "{:.3f}±{:.3f}".format(mean, sample_sd)
    return "{:.3f}".format(mean)


def summary_markdown(summary, n_source_runs):
    labels = {
        "frozen_logistic": "Frozen MINT + logistic head",
        "head_only": "Head-only arm",
        "lora_cross": "LoRA + head arm",
    }
    lines = [
        "# Matched MINT/LoRA aggregation",
        "",
        "All six one-seed artifacts passed hash, experiment-contract, panel-membership, and direct metric-recomputation checks. Head-only and LoRA entries are mean±sample SD across the three prespecified paired seeds. The frozen reference is deterministic and is counted once per library, so no seed SD is shown.",
        "",
        "Source run directories: {}.".format(int(n_source_runs)),
    ]
    for library in EXPECTED_LIBRARIES:
        model_rows = summary.loc[
            summary["library"].eq(library) & summary["result_type"].eq("model_score")
        ].set_index("result")
        lines.extend(
            [
                "",
                "## {}".format(library),
                "",
                "| Model | Seed runs | Selected epochs | AUROC | AP | Global Spearman | Within-peptide Spearman |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for arm in ALL_ARMS:
            row = model_rows.loc[arm]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    labels[arm],
                    int(row["n_seed_runs"]),
                    row["selected_epochs"],
                    _metric_text(row, "global_auroc"),
                    _metric_text(row, "global_auprc"),
                    _metric_text(row, "global_spearman"),
                    _metric_text(row, "within_peptide_macro_spearman"),
                )
            )
        change_rows = summary.loc[
            summary["library"].eq(library) & summary["result_type"].eq("paired_change")
        ].copy()
        change_rows["comparison"] = change_rows["result"] + "_minus_" + change_rows["reference"]
        change_rows = change_rows.set_index("comparison")
        lines.extend(
            [
                "",
                "| Paired change | Seeds | Epochs (model/reference) | ΔAUROC | ΔAP | ΔGlobal Spearman | ΔWithin-peptide Spearman |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for comparison in (
            "head_only_minus_frozen_logistic",
            "lora_cross_minus_frozen_logistic",
            "lora_cross_minus_head_only",
        ):
            row = change_rows.loc[comparison]
            display = labels[str(row["result"])] + " − " + labels[str(row["reference"])]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    display,
                    int(row["n_seed_runs"]),
                    row["selected_epochs"],
                    _metric_text(row, "global_auroc"),
                    _metric_text(row, "global_auprc"),
                    _metric_text(row, "global_spearman"),
                    _metric_text(row, "within_peptide_macro_spearman"),
                )
            )
    lines.extend(
        [
            "",
            "Epochs are listed in prespecified seed order; epoch 0 means that arm received no optimizer update. AUROC and AP use the 75% retention threshold only on the measured evaluation panel. Spearman uses the measured retention values directly. Seed SD describes optimization-seed variation, not evaluation-set uncertainty. The primary adapter comparison is LoRA + head minus head-only; LoRA minus frozen also includes head retraining. All comparisons remain retrospective and exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def _recheck_verified_inputs(verified):
    """Second stability check, performed before creating aggregate outputs."""
    for item in verified:
        _require(
            sha256_file(item["manifest_path"]) == item["manifest_sha256"],
            "{} manifest changed during aggregation".format(item["run_name"]),
        )
        for filename, record in item["manifest"]["outputs"].items():
            _require(
                sha256_file(item["run_dir"] / filename) == record["sha256"],
                "{} {} changed during aggregation".format(
                    item["run_name"], filename
                ),
            )
    stable_sources = {}
    for item in verified:
        for record in item["sources"].values():
            path = str(Path(record["path"]).resolve())
            previous = stable_sources.setdefault(path, record["sha256"])
            _require(
                previous == record["sha256"],
                "one source path was recorded with two hashes",
            )
    for path, expected_hash in stable_sources.items():
        _require(
            sha256_file(path) == expected_hash,
            "source changed during aggregation: {}".format(path),
        )


def run(args):
    started = time.time()
    aggregate_code_hash = sha256_file(Path(__file__).resolve())
    trainer_code_hash = sha256_file(Path(matched.__file__).resolve())
    evaluator_code_hash = sha256_file(Path(cached_eval.__file__).resolve())
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    _require(len(run_dirs) == 6, "exactly six --run-dirs values are required")
    _require(len(run_dirs) == len(set(run_dirs)), "duplicate input directory")

    verified = [load_verified_run(path) for path in run_dirs]
    combined = combine_verified_runs(verified)
    metrics = combined["metrics"]
    paired = build_paired_changes(metrics)
    summary = build_combined_summary(metrics, paired)
    markdown = summary_markdown(summary, len(verified))

    _recheck_verified_inputs(verified)
    _require(
        sha256_file(Path(__file__).resolve()) == aggregate_code_hash,
        "aggregate code changed during execution",
    )
    _require(
        sha256_file(Path(matched.__file__).resolve()) == trainer_code_hash
        and sha256_file(Path(cached_eval.__file__).resolve())
        == evaluator_code_hash,
        "an imported dependency changed during execution",
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    outputs = {}
    tables = {
        "recomputed_retention_metrics.csv": metrics,
        "paired_changes_by_seed.csv": paired,
        "combined_summary.csv": summary,
    }
    for name, frame in tables.items():
        path = output_dir / name
        _write_csv(frame, path)
        outputs[name] = path
    summary_path = output_dir / "summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write(markdown)
    os.chmod(str(summary_path), 0o600)
    outputs[summary_path.name] = summary_path

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "configuration": {
            "expected_libraries": list(EXPECTED_LIBRARIES),
            "expected_training_seeds": list(EXPECTED_TRAINING_SEEDS),
            "expected_arms": list(ALL_ARMS),
            "metric_recomputation": "canonical evaluate_cached_weak_mint.ranking_metrics on each prediction block",
            "metric_comparison_tolerance": {"absolute": 1e-12, "relative": 1e-12},
            "seed_aggregation": "arithmetic mean and sample SD (ddof=1)",
            "frozen_aggregation": "one deterministic row per library; repeated copies verified equal",
            "paired_comparisons": [
                "head_only_minus_frozen_logistic",
                "lora_cross_minus_frozen_logistic",
                "lora_cross_minus_head_only",
            ],
            "model_delta_audit": "filename/path/SHA256 only; pickle payloads are not deserialized",
        },
        "matched_training_configuration": combined["configuration_contract"],
        "matched_runtime": combined["runtime_contract"],
        "common_source_hashes": combined["common_source_hashes"],
        "library_reference_source_hashes": combined["library_reference_source_hashes"],
        "source_runs": [
            {
                "name": item["run_name"],
                "path": str(item["run_dir"]),
                "library": item["library"],
                "training_seed": int(item["training_seed"]),
                "manifest_path": str(item["manifest_path"]),
                "manifest_sha256": item["manifest_sha256"],
            }
            for item in sorted(verified, key=lambda row: (row["library"], row["training_seed"]))
        ],
        "rows": {
            "source_runs": int(len(verified)),
            "recomputed_metric_records": int(len(metrics)),
            "paired_change_records": int(len(paired)),
            "summary_records": int(len(summary)),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(outputs.items())
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": aggregate_code_hash,
        },
        "dependencies": {
            "trainer": {"path": str(Path(matched.__file__).resolve()), "sha256": trainer_code_hash},
            "canonical_evaluator": {"path": str(Path(cached_eval.__file__).resolve()), "sha256": evaluator_code_hash},
        },
        "permissions": {"directory": "0700", "files": "0600"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_runs": len(verified),
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
        help="exactly six one-seed finetune_mint_selection_matched.py output directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
