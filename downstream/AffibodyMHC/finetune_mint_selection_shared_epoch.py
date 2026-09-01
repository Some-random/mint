#!/usr/bin/env python
"""Refit the matched head-only and cross-LoRA arms at one shared epoch.

This program is deliberately a small follow-up to
``finetune_mint_selection_matched.py``.  It accepts one *completed and fully
verified* one-library/one-seed matched run, reuses that run's paired weak-label
validation predictions for epochs 0--3, and chooses one common positive epoch
for both training arms.  It then reruns only the two all-row final fits.

Retention outcomes are unavailable to epoch selection and model fitting.  They
are scored only after both final prediction vectors have been fixed.
"""

from __future__ import print_function

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import aggregate_mint_selection_matched as source_audit
from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "mint-selection-shared-epoch-v1"
SOURCE_SCHEMA_VERSION = matched.SCHEMA_VERSION
ARMS = tuple(matched.ARMS)
EPOCH_CANDIDATES = (0, 1, 2, 3)
SHARED_POSITIVE_EPOCH_CANDIDATES = (1, 2, 3)
EPOCH_SELECTION_DESCRIPTION = (
    "minimum equal-arm mean pooled weak-validation unweighted log loss over "
    "epochs 1-3; earlier epoch tie-break only; epoch 0 diagnostic only"
)
CORE_METRICS = (
    "global_auroc",
    "global_auprc",
    "global_spearman",
    "within_peptide_macro_spearman",
)
COPIED_SOURCE_OUTPUTS = (
    "c_validation.csv",
    "fold_membership.csv",
    "weak_validation_predictions.csv",
)
EPOCH0_ARM_PARITY_ATOL = 1e-8
SOURCE_REPRODUCTION_ATOL = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_csv(path):
    return pd.read_csv(str(path), float_precision="round_trip")


def _integer_series(values, label):
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(numeric).all()), "{} contains non-finite values".format(label))
    _require(
        bool(np.equal(numeric, np.floor(numeric)).all()),
        "{} contains non-integer values".format(label),
    )
    return pd.Series(numeric.astype(np.int64), index=values.index)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _hash_named_arrays(named_arrays):
    """Hash names, shapes, dtypes, and bytes for a small tensor/array map."""
    digest = hashlib.sha256()
    for name, value in sorted(named_arrays.items()):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def head_initialization_sha256(classifier):
    """Hash the exact float32 head tensors loaded into each live model."""
    return _hash_named_arrays(
        {
            "weight": np.asarray(classifier.coef_, dtype=np.float32),
            "bias": np.asarray(classifier.intercept_, dtype=np.float32),
        }
    )


def validate_paired_validation_predictions(
    frame,
    library,
    training_seed,
    fold_contract=None,
    expected_membership=None,
    epoch0_atol=EPOCH0_ARM_PARITY_ATOL,
):
    """Validate arm pairing and return a canonical, numeric prediction table."""
    required = {
        "pair_uid",
        "weak_label",
        "epoch",
        "probability",
        "library",
        "arm",
        "training_seed",
        "fold",
    }
    _require(required.issubset(frame.columns), "weak-validation prediction schema mismatch")
    output = frame.loc[:, sorted(required)].copy()
    _require(not output.empty, "weak-validation predictions are empty")
    _require(bool(output["library"].astype(str).eq(str(library)).all()), "validation library changed")
    output["training_seed"] = _integer_series(
        output["training_seed"], "validation training_seed"
    )
    _require(
        set(output["training_seed"].tolist()) == {int(training_seed)},
        "validation training seed changed",
    )
    output["epoch"] = _integer_series(output["epoch"], "validation epoch")
    output["fold"] = _integer_series(output["fold"], "validation fold")
    output["weak_label"] = _integer_series(
        output["weak_label"], "validation weak_label"
    )
    output["probability"] = pd.to_numeric(
        output["probability"], errors="raise"
    ).astype(float)
    _require(set(output["arm"].astype(str)) == set(ARMS), "validation arms changed")
    _require(set(output["epoch"].tolist()) == set(EPOCH_CANDIDATES), "validation epochs changed")
    _require(set(output["weak_label"].tolist()) == {0, 1}, "validation labels need both classes")
    probability = output["probability"].to_numpy(dtype=float)
    _require(bool(np.isfinite(probability).all()), "non-finite validation probability")
    _require(
        bool(((probability >= 0.0) & (probability <= 1.0)).all()),
        "validation probability outside [0,1]",
    )
    duplicate = output.duplicated(["arm", "epoch", "fold", "pair_uid"])
    _require(not bool(duplicate.any()), "duplicate weak-validation prediction")

    if fold_contract is not None:
        expected_folds = set(int(value) for value in fold_contract)
        _require(set(output["fold"].tolist()) == expected_folds, "validation folds changed")
    else:
        expected_folds = set(output["fold"].tolist())

    reference_membership = None
    for arm in ARMS:
        for epoch in EPOCH_CANDIDATES:
            block = output.loc[
                output["arm"].eq(arm) & output["epoch"].eq(epoch),
                ["fold", "pair_uid", "weak_label"],
            ].sort_values(["fold", "pair_uid"], kind="mergesort").reset_index(drop=True)
            _require(not block.empty, "missing {} epoch {} validation rows".format(arm, epoch))
            _require(set(block["fold"].tolist()) == expected_folds, "fold coverage differs by arm/epoch")
            _require(
                not bool(block["pair_uid"].duplicated().any()),
                "a pair appears in more than one validation fold",
            )
            if fold_contract is not None:
                for fold, contract in sorted(fold_contract.items()):
                    fold_block = block.loc[block["fold"].eq(int(fold))]
                    _require(
                        len(fold_block) == int(contract["validation"]),
                        "fold {} validation row count changed".format(fold),
                    )
                    _require(
                        int(fold_block["weak_label"].sum())
                        == int(contract["validation_positive"]),
                        "fold {} validation positive count changed".format(fold),
                    )
            if reference_membership is None:
                reference_membership = block
            else:
                _require(
                    block.equals(reference_membership),
                    "validation row membership or labels differ by arm/epoch",
                )

    if expected_membership is not None:
        expected = expected_membership[
            ["fold", "pair_uid", "weak_label"]
        ].copy()
        expected["fold"] = _integer_series(expected["fold"], "expected fold")
        expected["weak_label"] = _integer_series(
            expected["weak_label"], "expected weak_label"
        )
        expected = expected.sort_values(
            ["fold", "pair_uid"], kind="mergesort"
        ).reset_index(drop=True)
        _require(
            expected.equals(reference_membership),
            "validation predictions differ from reconstructed canonical folds",
        )

    epoch0 = output.loc[output["epoch"].eq(0)].pivot(
        index=["fold", "pair_uid", "weak_label"], columns="arm", values="probability"
    )
    _require(set(epoch0.columns.astype(str)) == set(ARMS), "epoch-0 arm pairing failed")
    parity = float(
        np.max(
            np.abs(
                epoch0["head_only"].to_numpy(dtype=float)
                - epoch0["lora_cross"].to_numpy(dtype=float)
            )
        )
    )
    _require(parity <= float(epoch0_atol), "head/cross epoch-0 validation parity failed")
    output = output.sort_values(
        ["arm", "epoch", "fold", "pair_uid"], kind="mergesort"
    ).reset_index(drop=True)
    return output, parity


def select_shared_positive_epoch(frame, library, training_seed):
    """Choose a common epoch from 1--3 using only equal-arm mean log loss."""
    rows = []
    for epoch in EPOCH_CANDIDATES:
        by_arm = {}
        n = None
        positive = None
        for arm in ARMS:
            current = frame.loc[
                frame["arm"].eq(arm) & frame["epoch"].eq(int(epoch))
            ]
            _require(not current.empty, "missing {} epoch {}".format(arm, epoch))
            metrics = matched.binary_metrics(
                current["weak_label"].to_numpy(dtype=int),
                current["probability"].to_numpy(dtype=float),
            )
            if n is None:
                n = int(metrics["n"])
                positive = int(metrics["positive"])
            else:
                _require(
                    n == int(metrics["n"]) and positive == int(metrics["positive"]),
                    "arm validation counts differ",
                )
            by_arm[arm] = float(metrics["log_loss"])
        rows.append(
            {
                "library": str(library),
                "training_seed": int(training_seed),
                "epoch": int(epoch),
                "n": int(n),
                "positive": int(positive),
                "head_only_log_loss": by_arm["head_only"],
                "lora_cross_log_loss": by_arm["lora_cross"],
                "equal_arm_mean_log_loss": 0.5
                * (by_arm["head_only"] + by_arm["lora_cross"]),
            }
        )
    selected = min(
        (row for row in rows if row["epoch"] in SHARED_POSITIVE_EPOCH_CANDIDATES),
        key=lambda row: (row["equal_arm_mean_log_loss"], row["epoch"]),
    )
    epoch0 = next(row for row in rows if row["epoch"] == 0)
    selected_epoch = int(selected["epoch"])
    gap = float(
        epoch0["equal_arm_mean_log_loss"] - selected["equal_arm_mean_log_loss"]
    )
    epoch0_better = bool(gap < 0.0)
    epoch0_no_worse = bool(gap <= 0.0)
    for row in rows:
        row["selected"] = int(row["epoch"] == selected_epoch)
        row["chosen_positive_epoch"] = selected_epoch
        row["epoch0_better"] = int(epoch0_better)
        row["epoch0_no_worse"] = int(epoch0_no_worse)
        row["epoch0_minus_selected_log_loss"] = gap
    selection = pd.DataFrame(rows)
    _require(
        int(selection["selected"].sum()) == 1
        and int(selection.loc[selection["selected"].eq(1), "epoch"].iloc[0]) > 0,
        "shared positive epoch selection failed",
    )
    return selected_epoch, selection


def _source_cache_arguments(source):
    sources = source["sources"]
    cache_names = sorted(name for name in sources if name.startswith("cache_npz_"))
    manifest_names = sorted(
        name for name in sources if name.startswith("cache_manifest_")
    )
    _require(cache_names, "verified source has no feature cache")
    _require(len(cache_names) == len(manifest_names), "cache/manifest coverage differs")
    cache_paths = [Path(sources[name]["path"]).resolve() for name in cache_names]
    if len(cache_paths) == 1:
        return cache_paths[0], Path(sources[manifest_names[0]]["path"]).resolve()
    parents = {path.parent for path in cache_paths}
    _require(len(parents) == 1, "cache shards are not in one directory")
    parent = next(iter(parents))
    _require(not (parent / "mint_chain_mean_features.npz").is_file(), "cache directory gained a merged archive")
    observed = sorted(path.resolve() for path in parent.glob("features-shard-*.npz"))
    _require(observed == sorted(cache_paths), "cache shard directory contents changed")
    return parent, None


def prefit_verify_source_run(run_dir):
    """Verify a complete matched artifact without parsing retention outcomes.

    All output files are hashed here, including the old retention tables, but
    those tables are not opened.  The existing comprehensive verifier is run
    only after both new prediction vectors have been fixed.
    """
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "source matched-run directory is missing")
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "source matched run has no manifest.json")
    with open(str(manifest_path), "r") as handle:
        manifest = json.load(handle)
    _require(isinstance(manifest, dict), "source manifest root is not an object")
    _require(manifest.get("schema_version") == SOURCE_SCHEMA_VERSION, "source schema changed")
    _require(
        manifest.get("analysis_status") == "retrospective_exploratory",
        "source analysis status changed",
    )
    library, training_seed, configuration = source_audit._validate_configuration(
        manifest, run_dir.name
    )
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "source manifest has no outputs")
    _require(
        set(outputs) == source_audit._expected_outputs(training_seed),
        "source output set is incomplete or unexpected",
    )
    for filename, record in sorted(outputs.items()):
        source_audit._verified_recorded_file(
            record, run_dir / filename, "source {}".format(filename)
        )

    records = manifest.get("sources")
    _require(isinstance(records, dict), "source manifest has no source records")
    missing = source_audit.BASE_SOURCE_NAMES.difference(records)
    _require(not missing, "source manifest lacks source records {}".format(sorted(missing)))
    cache_npz = sorted(name for name in records if name.startswith("cache_npz_"))
    cache_manifest = sorted(
        name for name in records if name.startswith("cache_manifest_")
    )
    _require(cache_npz, "source manifest has no feature-cache NPZ")
    _require(
        cache_npz == ["cache_npz_{:03d}".format(i) for i in range(len(cache_npz))]
        and cache_manifest
        == ["cache_manifest_{:03d}".format(i) for i in range(len(cache_npz))],
        "source cache records are incomplete or noncontiguous",
    )
    allowed = source_audit.BASE_SOURCE_NAMES | set(cache_npz) | set(cache_manifest)
    _require(set(records) == allowed, "source manifest has unexpected source records")
    verified_sources = {}
    for name, record in sorted(records.items()):
        verified_sources[name] = source_audit._verified_source(
            record,
            "source {}".format(name),
            expected_path=source_audit.CODE_SOURCE_PATHS.get(name),
        )
    # Bind the feature and row archives structurally without loading any target
    # column.  The comprehensive post-prediction audit rechecks their contents.
    for index, npz_name in enumerate(cache_npz):
        manifest_name = "cache_manifest_{:03d}".format(index)
        with open(verified_sources[manifest_name]["path"], "r") as handle:
            feature_manifest = json.load(handle)
        _require(
            feature_manifest.get("schema_version") == "mint-weak-feature-cache-v1"
            and feature_manifest.get("output", {}).get("sha256")
            == verified_sources[npz_name]["sha256"],
            "feature-cache manifest does not bind NPZ {}".format(index),
        )
    with open(verified_sources["cache_rows_manifest"]["path"], "r") as handle:
        row_manifest = json.load(handle)
    _require(
        row_manifest.get("output", {}).get("rows_csv", {}).get("sha256")
        == verified_sources["cache_rows"]["sha256"],
        "cache-row manifest does not bind its CSV",
    )
    with open(verified_sources["retention_features_manifest"]["path"], "r") as handle:
        retention_manifest = json.load(handle)
    _require(
        retention_manifest.get("output", {}).get("sha256")
        == verified_sources["retention_features"]["sha256"],
        "retention-feature manifest does not bind its NPZ",
    )
    reference = manifest.get("reference_primary_run")
    _require(isinstance(reference, dict), "source lacks primary reference record")
    _require(
        float(reference.get("selected_c"))
        == float(matched.PRIMARY_CONTRACT[library]["selected_c"]),
        "source primary reference C changed",
    )
    reference_bindings = {
        "reference_manifest_json": "manifest_sha256",
        "reference_conditions_csv": "conditions_sha256",
        "reference_weak_validation_csv": "weak_validation_sha256",
        "reference_metrics_csv": "metrics_sha256",
    }
    reference_directory = Path(reference.get("directory", "")).resolve()
    _require(reference_directory.is_dir(), "source reference directory is missing")
    for source_name, reference_key in reference_bindings.items():
        _require(
            verified_sources[source_name]["sha256"] == reference.get(reference_key)
            and Path(verified_sources[source_name]["path"]).parent
            == reference_directory,
            "source primary reference binding changed for {}".format(source_name),
        )
    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "source runtime record missing")
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "library": library,
        "training_seed": int(training_seed),
        "configuration": configuration,
        "sources": verified_sources,
        "reference": reference,
        "runtime": runtime,
    }


def _load_prefit_cache(source):
    """Load only identities, weak labels, sequences, and frozen features."""
    sources = source["sources"]
    cache_names = sorted(name for name in sources if name.startswith("cache_npz_"))
    required = {
        "row_index",
        "cache_uid",
        "source_kind",
        "library",
        "pair_uid",
        "weak_label",
        "measurement_missing",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        cached_eval.FEATURE_NAME,
    }
    blocks = []
    for name in cache_names:
        path = Path(sources[name]["path"])
        with np.load(str(path), allow_pickle=False) as archive:
            _require(
                cached_eval.CACHE_REQUIRED_KEYS.issubset(archive.files)
                and required.issubset(archive.files),
                "source feature-cache NPZ schema changed",
            )
            blocks.append({key: np.asarray(archive[key]).copy() for key in required})
    arrays = (
        blocks[0]
        if len(blocks) == 1
        else {
            key: np.concatenate([block[key] for block in blocks], axis=0)
            for key in sorted(required)
        }
    )
    order = np.argsort(np.asarray(arrays["row_index"], dtype=np.int64))
    arrays = {key: value[order] for key, value in arrays.items()}
    row_index = np.asarray(arrays.pop("row_index"), dtype=np.int64)
    _require(
        np.array_equal(row_index, np.arange(len(row_index), dtype=np.int64)),
        "prefit cache rows are incomplete or noncontiguous",
    )
    features = np.asarray(arrays.pop(cached_eval.FEATURE_NAME), dtype=np.float32)
    _require(features.shape == (len(row_index), 2560), "prefit feature shape changed")
    _require(bool(np.isfinite(features).all()), "prefit features are non-finite")
    cache_frame = pd.DataFrame(
        {key: np.asarray(value).astype(str) for key, value in arrays.items()}
    )
    cache_frame.insert(0, "row_index", row_index)

    row_columns = [
        "row_index",
        "cache_uid",
        "source_kind",
        "library",
        "pair_uid",
        "measurement_missing",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    row_frame = pd.read_csv(
        sources["cache_rows"]["path"],
        usecols=row_columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    row_frame["row_index"] = pd.to_numeric(
        row_frame["row_index"], errors="raise"
    ).astype(np.int64)
    row_frame = row_frame.sort_values("row_index").reset_index(drop=True)
    _require(
        np.array_equal(
            row_frame["row_index"].to_numpy(dtype=np.int64), row_index
        ),
        "prefit row-table index differs from features",
    )
    cache_frame = matched.attach_full_sequences(cache_frame, row_frame)
    cache_frame["_cache_index"] = np.arange(len(cache_frame), dtype=int)
    primary, retention_identity = matched.make_primary_pool(
        cache_frame, source["library"]
    )
    _require(
        "target_retention" not in primary.columns
        and "target_binder" not in primary.columns
        and "target_retention" not in retention_identity.columns
        and "target_binder" not in retention_identity.columns,
        "retention outcomes entered prefit inputs",
    )
    legacy_table, legacy_features, _ = cached_eval.load_legacy_retention_features(
        sources["retention_features"]["path"]
    )
    retention_identity = retention_identity.merge(
        legacy_table, on="pair_uid", validate="one_to_one"
    )
    _require(
        len(retention_identity)
        == int(matched.PRIMARY_CONTRACT[source["library"]]["retention_rows"]),
        "prefit retention identity count changed",
    )
    _require(
        bool(
            retention_identity["library"]
            .eq(retention_identity["legacy_library"])
            .all()
        )
        and bool(
            retention_identity["measurement_missing"]
            .astype(str)
            .eq(retention_identity["legacy_missing"])
            .all()
        ),
        "legacy retention identity join changed",
    )
    retention_identity = retention_identity.sort_values("pair_uid").reset_index(
        drop=True
    )
    _require(
        cached_eval.membership_sha256(retention_identity)
        == matched.PRIMARY_CONTRACT[source["library"]][
            "retention_membership_sha256"
        ],
        "prefit retention membership changed",
    )
    primary_indices = primary["_cache_index"].to_numpy(dtype=int)
    weak_features = features[primary_indices]
    retention_features = legacy_features[
        retention_identity["legacy_index"].to_numpy(dtype=int)
    ]
    cached_retention_features = features[
        retention_identity["_cache_index"].to_numpy(dtype=int)
    ]
    feature_difference = float(
        np.max(np.abs(cached_retention_features - retention_features))
    )
    _require(feature_difference <= 1e-4, "retention feature archives disagree")
    return {
        "primary": primary,
        "retention_identity": retention_identity,
        "weak_features": weak_features,
        "retention_features": retention_features,
        "retention_feature_max_abs_difference": feature_difference,
    }


def load_retention_targets_after_predictions(source, retention_identity):
    """Load retention values only after both model prediction vectors exist."""
    columns = [
        "source_kind",
        "library",
        "pair_uid",
        "measurement_missing",
        "target_retention",
        "target_binder",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    target = pd.read_csv(
        source["sources"]["cache_rows"]["path"],
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    target = target.loc[
        target["source_kind"].eq("retention")
        & target["library"].eq(source["library"])
        & target["measurement_missing"].eq("0")
    ].copy()
    _require(not bool(target["pair_uid"].duplicated().any()), "duplicate retention target")
    target["target_retention"] = pd.to_numeric(
        target["target_retention"], errors="raise"
    )
    target["target_binder"] = pd.to_numeric(
        target["target_binder"], errors="raise"
    ).astype(int)
    _require(
        bool(
            target["target_binder"]
            .eq(target["target_retention"].ge(75.0).astype(int))
            .all()
        ),
        "75-percent binder labels changed",
    )
    identity_columns = [
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    left = retention_identity.sort_values("pair_uid", kind="mergesort").reset_index(
        drop=True
    )
    right = target.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(
        left[identity_columns].astype(str).equals(right[identity_columns].astype(str)),
        "retention target identities differ from prediction identities",
    )
    output = retention_identity.merge(
        target[["pair_uid", "target_retention", "target_binder"]],
        on="pair_uid",
        validate="one_to_one",
    ).sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    contract = matched.PRIMARY_CONTRACT[source["library"]]
    _require(
        len(output) == int(contract["retention_rows"])
        and int(output["target_binder"].sum())
        == int(contract["retention_positive"]),
        "retention target counts changed",
    )
    _require(
        cached_eval.membership_sha256(output)
        == contract["retention_membership_sha256"],
        "retention target membership changed",
    )
    return output


def training_arguments_from_verified_source(source, device):
    """Reconstruct the immutable matched training arguments from its manifest."""
    configuration = source["configuration"]
    library = source["library"]
    training_seed = int(source["training_seed"])
    _require(configuration.get("training_seeds") == [training_seed], "source is not one seed")
    _require(configuration.get("epoch_candidates") == [0, 1, 2, 3], "source epoch horizon changed")
    _require(tuple(configuration.get("arms", [])) == ("frozen_logistic",) + ARMS, "source arms changed")
    lora = configuration.get("lora", {})
    cache, cache_manifest = _source_cache_arguments(source)
    sources = source["sources"]
    return SimpleNamespace(
        cache=cache,
        cache_manifest=cache_manifest,
        cache_rows=Path(sources["cache_rows"]["path"]).resolve(),
        cache_rows_manifest=Path(sources["cache_rows_manifest"]["path"]).resolve(),
        retention_features=Path(sources["retention_features"]["path"]).resolve(),
        reference_run_dir=Path(source["reference"]["directory"]).resolve(),
        checkpoint=Path(sources["checkpoint"]["path"]).resolve(),
        config=Path(sources["config"]["path"]).resolve(),
        library=library,
        device=str(device),
        training_seeds=[training_seed],
        split_seed=int(configuration["split_seed"]),
        folds=int(configuration["folds"]),
        c_grid=[float(value) for value in configuration["c_grid"]],
        max_epochs=3,
        batch_size=int(configuration["batch_size"]),
        eval_batch_size=int(configuration["eval_batch_size"]),
        accumulation_steps=int(configuration["accumulation_steps"]),
        head_lr=float(configuration["head_lr"]),
        adapter_lr=float(configuration["adapter_lr"]),
        weight_decay=float(configuration["weight_decay"]),
        warmup_fraction=float(configuration["warmup_fraction"]),
        clip_norm=float(configuration["clip_norm"]),
        lora_rank=int(lora["rank"]),
        lora_alpha=float(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        max_live_feature_probability_difference=0.01,
        max_retention_feature_difference=1e-4,
    )


def paired_difference_record(metrics, library, training_seed, selected_epoch):
    by_arm = metrics.set_index("arm")
    _require(set(by_arm.index) == set(ARMS), "paired retention metric arms missing")
    row = {
        "library": str(library),
        "training_seed": int(training_seed),
        "selected_epoch": int(selected_epoch),
        "comparison": "lora_cross_minus_head_only",
    }
    for metric in CORE_METRICS:
        row[metric + "_change"] = float(
            by_arm.loc["lora_cross", metric] - by_arm.loc["head_only", metric]
        )
    return row


def _copy_verified_source_output(source, filename, output_dir):
    record = source["manifest"]["outputs"].get(filename)
    _require(isinstance(record, dict), "source lacks {}".format(filename))
    source_path = source["run_dir"] / filename
    _require(sha256_file(source_path) == record["sha256"], "source {} changed".format(filename))
    destination = output_dir / filename
    shutil.copyfile(str(source_path), str(destination))
    os.chmod(str(destination), 0o600)
    _require(sha256_file(destination) == record["sha256"], "copied {} changed bytes".format(filename))
    return destination


def _assert_paired_training_audits(audits, selected_epoch, head_hash):
    _require(len(audits) == 2 and set(audits["arm"]) == set(ARMS), "final audit arms missing")
    common = (
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
        "final_refit_epoch0_arm_max_abs_difference",
    )
    for column in common:
        _require(audits[column].nunique(dropna=False) == 1, "paired arms differ in {}".format(column))
    _require(int(audits["selected_epoch"].iloc[0]) == int(selected_epoch), "refit epoch differs")
    _require(audits["head_initialization_sha256"].iloc[0] == head_hash, "head initialization hash differs")
    _require(
        float(audits["final_refit_epoch0_arm_max_abs_difference"].iloc[0])
        <= EPOCH0_ARM_PARITY_ATOL,
        "final-refit epoch-0 arm parity differs",
    )
    _require(bool(audits["head_parameter_delta_l2"].gt(0.0).all()), "a trained head did not move")
    head = audits.loc[audits["arm"].eq("head_only")].iloc[0]
    cross = audits.loc[audits["arm"].eq("lora_cross")].iloc[0]
    _require(float(head["adapter_parameter_delta_max_abs"]) == 0.0, "head-only adapter moved")
    _require(float(cross["adapter_parameter_delta_l2"]) > 0.0, "cross-LoRA adapter did not move")


def audit_final_epoch0_pairing(
    evaluation_frame,
    mean,
    scale,
    classifier,
    training_seed,
    args,
    device,
    expected_head_hash,
):
    """Directly compare both outcome-free live arms before any optimizer step."""
    probabilities = {}
    head_hashes = {}
    adapter_hashes = {}
    run_seeds = {}
    for arm in ARMS:
        run_seed = matched.derived_seed(training_seed, "final_refit", -1)
        matched.set_deterministic_seed(run_seed)
        model = matched.MINTSelectionClassifier(
            args.config,
            args.checkpoint,
            device,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
        ).to(device)
        model.set_feature_standardization(mean, scale)
        matched.load_head(model, classifier)
        matched.live_trainable_audit(model, arm, args.lora_rank)
        head_hashes[arm] = _hash_named_arrays(
            {
                name: value.detach().cpu().numpy()
                for name, value in model.head.state_dict().items()
            }
        )
        adapter_hashes[arm] = _hash_named_arrays(
            {
                name: parameter.detach().cpu().numpy()
                for name, parameter in model.named_parameters()
                if "lora_" in name
            }
        )
        loader = matched.make_loader(
            evaluation_frame, args.eval_batch_size, False, run_seed
        )
        _, probability, observed, pair_uids = matched.predict_live(
            model, loader, device
        )
        _require(
            pair_uids == evaluation_frame["pair_uid"].astype(str).tolist()
            and np.array_equal(
                observed,
                evaluation_frame["weak_label"].to_numpy(dtype=int),
            ),
            "direct final epoch-0 audit changed evaluation rows",
        )
        probabilities[arm] = probability
        run_seeds[arm] = run_seed
        del loader
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _require(len(set(run_seeds.values())) == 1, "final epoch-0 arms used different seeds")
    _require(
        set(head_hashes.values()) == {str(expected_head_hash)},
        "actual live head initialization differs across arms or from classifier",
    )
    _require(
        len(set(adapter_hashes.values())) == 1,
        "actual live adapter initialization differs across arms",
    )
    parity = float(
        np.max(
            np.abs(
                probabilities["head_only"] - probabilities["lora_cross"]
            )
        )
    )
    _require(
        parity <= EPOCH0_ARM_PARITY_ATOL,
        "final-refit head/cross epoch-0 prediction parity failed",
    )
    return {
        "max_abs_probability_difference": parity,
        "live_head_sha256": head_hashes["head_only"],
        "live_adapter_sha256": adapter_hashes["head_only"],
        "run_seed": int(run_seeds["head_only"]),
    }


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required for live MINT refitting")

    code_paths = {
        "script": Path(__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "matched_source_verifier": Path(source_audit.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    source = prefit_verify_source_run(Path(args.source_run_dir).resolve())
    _require(
        source["run_dir"] not in output_dir.parents,
        "output directory may not be nested inside its source matched run",
    )
    _require(
        source["manifest"].get("schema_version") == SOURCE_SCHEMA_VERSION,
        "source matched-run schema changed",
    )
    library = source["library"]
    training_seed = int(source["training_seed"])
    train_args = training_arguments_from_verified_source(source, args.device)

    source_validation_path = source["run_dir"] / "weak_validation_predictions.csv"
    validation_raw = _read_csv(source_validation_path)
    inputs = _load_prefit_cache(source)
    primary = inputs["primary"]
    retention_identity = inputs["retention_identity"]
    weak_features = inputs["weak_features"]
    retention_features = inputs["retention_features"]
    labels = primary["weak_label"].to_numpy(dtype=int)
    plans, _ = matched.canonical_fold_plans(primary, library)
    expected_validation_membership = pd.concat(
        [
            primary.iloc[np.asarray(plan["validation_keys"], dtype=int)][
                ["pair_uid", "weak_label"]
            ].assign(fold=int(plan["fold"]))
            for plan in plans
        ],
        ignore_index=True,
    )
    validation, epoch0_arm_parity = validate_paired_validation_predictions(
        validation_raw,
        library,
        training_seed,
        fold_contract=matched.PRIMARY_CONTRACT[library]["folds"],
        expected_membership=expected_validation_membership,
    )
    selected_epoch, epoch_selection = select_shared_positive_epoch(
        validation, library, training_seed
    )
    selected_c = float(source["configuration"]["selected_C"])
    _require(
        selected_c == float(matched.PRIMARY_CONTRACT[library]["selected_c"]),
        "selected C differs from canonical contract",
    )
    full_mean, full_scale = matched.fit_standardizer(weak_features)
    full_x = matched.standardize(weak_features, full_mean, full_scale)
    full_classifier = matched.fit_logistic(full_x, labels, selected_c)
    head_hash = head_initialization_sha256(full_classifier)

    frozen_x = matched.standardize(retention_features, full_mean, full_scale)
    frozen_probability = full_classifier.predict_proba(frozen_x)[:, 1]

    device = torch.device(str(args.device))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    final_evaluation = retention_identity.copy()
    _require(
        "target_retention" not in final_evaluation.columns
        and "target_binder" not in final_evaluation.columns,
        "retention outcomes entered live inference",
    )
    final_evaluation["weak_label"] = 0
    final_epoch0_audit = audit_final_epoch0_pairing(
        final_evaluation,
        full_mean,
        full_scale,
        full_classifier,
        training_seed,
        train_args,
        device,
        head_hash,
    )
    outcome_free_prediction_blocks = []
    audit_rows = []
    saved_states = {}
    for arm in ARMS:
        result = matched.train_live_arm(
            arm,
            primary,
            final_evaluation,
            full_mean,
            full_scale,
            full_classifier,
            training_seed,
            "final_refit",
            -1,
            selected_epoch,
            train_args,
            device,
            expected_epoch0_probability=frozen_probability,
            track_each_epoch=False,
        )
        predictions = result["predictions"]
        _require(len(predictions) == len(retention_identity), "final prediction count changed")
        _require(
            predictions["pair_uid"].astype(str).tolist()
            == final_evaluation["pair_uid"].astype(str).tolist(),
            "final prediction row order changed",
        )
        probability = predictions["probability"].to_numpy(dtype=float)
        output = retention_identity[
            [
                "pair_uid",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
            ]
        ].copy()
        output["library"] = library
        output["arm"] = arm
        output["training_seed"] = training_seed
        output["selected_epoch"] = selected_epoch
        output["probability"] = probability
        outcome_free_prediction_blocks.append(output)
        audit = matched.training_audit_record(
            "final_refit",
            arm,
            training_seed,
            -1,
            primary,
            full_mean,
            full_scale,
            result,
            selected_epoch,
        )
        audit.update(
            {
                "head_initialization_sha256": head_hash,
                "lr_schedule_horizon_epochs": 3,
                "source_run_manifest_sha256": source["manifest_sha256"],
                "source_validation_epoch0_arm_max_abs_difference": epoch0_arm_parity,
                "final_refit_epoch0_arm_max_abs_difference": final_epoch0_audit[
                    "max_abs_probability_difference"
                ],
            }
        )
        audit_rows.append(audit)
        saved_states[arm] = {
            "library": library,
            "arm": arm,
            "training_seed": training_seed,
            "selected_epoch": selected_epoch,
            "selected_C": selected_c,
            "head_initialization_sha256": head_hash,
            "base_checkpoint_sha256": source["sources"]["checkpoint"]["sha256"],
            "primary_membership_sha256": matched.PRIMARY_CONTRACT[library]["membership_sha256"],
            "source_run_manifest_sha256": source["manifest_sha256"],
            "feature_mean": torch.from_numpy(np.asarray(full_mean, dtype=np.float32)),
            "feature_scale": torch.from_numpy(np.asarray(full_scale, dtype=np.float32)),
            **result["model_state"]
        }
        del result
        torch.cuda.empty_cache()

    # This is the mechanical outcome boundary.  Both arm predictions must be
    # complete before either the comprehensive source audit or the target
    # loader is allowed to open retention values/binder labels.
    _require(
        len(outcome_free_prediction_blocks) == 2
        and set(block["arm"].iloc[0] for block in outcome_free_prediction_blocks)
        == set(ARMS),
        "both outcome-free prediction vectors were not fixed",
    )
    fully_verified_source = source_audit.load_verified_run(source["run_dir"])
    _require(
        fully_verified_source["manifest_sha256"] == source["manifest_sha256"]
        and fully_verified_source["library"] == library
        and int(fully_verified_source["training_seed"]) == training_seed,
        "source matched run changed across the outcome boundary",
    )
    retention = load_retention_targets_after_predictions(source, retention_identity)
    source_frozen = fully_verified_source["predictions"].loc[
        fully_verified_source["predictions"]["arm"].eq("frozen_logistic")
    ].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    ordered_retention = retention_identity.sort_values(
        "pair_uid", kind="mergesort"
    ).reset_index()
    _require(
        ordered_retention["pair_uid"].astype(str).equals(
            source_frozen["pair_uid"].astype(str)
        ),
        "reconstructed retention row order differs from source frozen predictions",
    )
    ordered_probability = frozen_probability[
        ordered_retention["index"].to_numpy(dtype=int)
    ]
    frozen_reproduction_error = float(
        np.max(
            np.abs(
                ordered_probability
                - source_frozen["probability"].to_numpy(dtype=float)
            )
        )
    )
    _require(
        frozen_reproduction_error <= SOURCE_REPRODUCTION_ATOL,
        "reconstructed frozen head does not reproduce source predictions",
    )

    training_audit = pd.DataFrame(audit_rows)
    training_audit["source_frozen_reproduction_max_abs_difference"] = (
        frozen_reproduction_error
    )
    _assert_paired_training_audits(training_audit, selected_epoch, head_hash)
    prediction_blocks = []
    metric_rows = []
    per_peptide_rows = []
    for outcome_free in outcome_free_prediction_blocks:
        _require(
            outcome_free["pair_uid"].astype(str).tolist()
            == retention["pair_uid"].astype(str).tolist(),
            "prediction and retention target order changed",
        )
        output = retention[
            [
                "pair_uid",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
                "target_retention",
                "target_binder",
            ]
        ].copy()
        for column in ("library", "arm", "training_seed", "selected_epoch", "probability"):
            output[column] = outcome_free[column].to_numpy()
        prediction_blocks.append(output)
        arm = str(outcome_free["arm"].iloc[0])
        probability = outcome_free["probability"].to_numpy(dtype=float)
        metric_rows.append(
            matched.retention_metric_record(
                retention,
                probability,
                library,
                arm,
                training_seed,
                selected_epoch,
            )
        )
        per_peptide_rows.extend(
            matched.per_peptide_records(
                retention,
                probability,
                library,
                arm,
                training_seed,
                selected_epoch,
            )
        )
    retention_predictions = pd.concat(prediction_blocks, ignore_index=True)
    retention_metrics = pd.DataFrame(metric_rows)
    paired_differences = pd.DataFrame(
        [
            paired_difference_record(
                retention_metrics, library, training_seed, selected_epoch
            )
        ]
    )
    per_peptide_metrics = pd.DataFrame(per_peptide_rows)

    # Recheck every input before publishing anything.  In particular, this
    # rejects a source run or dependency changed while the GPU refits ran.
    source_audit._recheck_verified_inputs([fully_verified_source])
    for name, path in code_paths.items():
        _require(sha256_file(path) == code_hashes[name], "{} changed during run".format(name))

    _require(not output_dir.exists(), "output directory appeared during run")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    artifacts = {}
    for filename in COPIED_SOURCE_OUTPUTS:
        artifacts[filename] = _copy_verified_source_output(source, filename, output_dir)
    tables = {
        "shared_epoch_selection.csv": epoch_selection,
        "training_audit.csv": training_audit,
        "retention_predictions.csv": retention_predictions,
        "retention_metrics.csv": retention_metrics,
        "paired_differences.csv": paired_differences,
        "per_peptide_metrics.csv": per_peptide_metrics,
    }
    for filename, frame in tables.items():
        path = output_dir / filename
        _write_csv(frame, path)
        artifacts[filename] = path
    model_deltas = {}
    for arm in ARMS:
        filename = "model_delta_seed{}_{}.pt".format(training_seed, arm)
        path = output_dir / filename
        torch.save(saved_states[arm], str(path))
        os.chmod(str(path), 0o600)
        artifacts[filename] = path
        model_deltas[arm] = {
            "filename": filename,
            "sha256": sha256_file(path),
            "library": library,
            "arm": arm,
            "training_seed": training_seed,
            "selected_epoch": selected_epoch,
            "base_checkpoint_sha256": source["sources"]["checkpoint"]["sha256"],
            "primary_membership_sha256": matched.PRIMARY_CONTRACT[library][
                "membership_sha256"
            ],
            "head_initialization_sha256": head_hash,
        }

    paired = paired_differences.iloc[0]
    summary_lines = [
        "# Shared-epoch MINT comparison: {} seed {}".format(library, training_seed),
        "",
        "- Shared positive epoch: {}".format(selected_epoch),
        "- Epoch 0 strictly better on equal-arm weak-validation log loss: {}".format(
            bool(epoch_selection["epoch0_better"].iloc[0])
        ),
        "- Both arms used the same all-row training data, initialization, batch order, and three-epoch learning-rate horizon.",
        "- Retention outcomes were used only after both prediction vectors were fixed.",
        "",
        "| Paired change (cross-LoRA minus head-only) | Value |",
        "|---|---:|",
        "| Within-peptide Spearman | {:.4f} |".format(
            paired["within_peptide_macro_spearman_change"]
        ),
        "| Global Spearman | {:.4f} |".format(paired["global_spearman_change"]),
        "| AP | {:.4f} |".format(paired["global_auprc_change"]),
        "| AUROC | {:.4f} |".format(paired["global_auroc_change"]),
        "",
        "This remains a retrospective exploratory analysis.",
        "",
    ]
    summary_path = output_dir / "run_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(summary_lines))
    os.chmod(str(summary_path), 0o600)
    artifacts[summary_path.name] = summary_path

    source_weak_record = source["manifest"]["outputs"][
        "weak_validation_predictions.csv"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "configuration": {
            "library": library,
            "regime": matched.REGIME,
            "cleaning": matched.CLEANING,
            "balance": matched.BALANCE,
            "class_weight_formula": "N / (2 * N_class); weighted BCE sum divided by row count",
            "folds": matched.CANONICAL_FOLDS,
            "split_seed": matched.CANONICAL_SPLIT_SEED,
            "c_grid": list(matched.CANONICAL_C_GRID),
            "selected_C": selected_c,
            "epoch_candidates": list(EPOCH_CANDIDATES),
            "shared_positive_epoch_candidates": list(SHARED_POSITIVE_EPOCH_CANDIDATES),
            "epoch_selection": EPOCH_SELECTION_DESCRIPTION,
            "chosen_positive_epoch": selected_epoch,
            "training_seeds": [training_seed],
            "arms": list(ARMS),
            "max_epochs": 3,
            "lr_schedule_horizon_epochs": 3,
            "batch_size": int(train_args.batch_size),
            "eval_batch_size": int(train_args.eval_batch_size),
            "accumulation_steps": int(train_args.accumulation_steps),
            "head_lr": float(train_args.head_lr),
            "adapter_lr": float(train_args.adapter_lr),
            "weight_decay": float(train_args.weight_decay),
            "warmup_fraction": float(train_args.warmup_fraction),
            "clip_norm": float(train_args.clip_norm),
            "lora": {
                "rank": int(train_args.lora_rank),
                "alpha": float(train_args.lora_alpha),
                "dropout": float(train_args.lora_dropout),
                "layers_zero_based": [31, 32],
                "projections": ["q_proj", "v_proj"],
                "placement": "multimer_attn",
            },
            "primary_endpoint": "within_peptide_macro_spearman",
            "secondary_endpoints": [
                "global_spearman",
                "global_auprc",
                "global_auroc",
            ],
            "retention_threshold_for_auroc_ap": 75.0,
            "retention_usage": {
                "partner_identities": "used before fitting to construct the library-local peptide-and-Affibody-cold pool",
                "numeric_retention_and_75_percent_labels": "final metrics only; never used for C or epoch selection",
            },
        },
        "canonical_contract": matched.PRIMARY_CONTRACT[library],
        "source_matched_run": {
            "directory": str(source["run_dir"]),
            "manifest_path": str(source["manifest_path"]),
            "manifest_sha256": source["manifest_sha256"],
            "schema_version": source["manifest"]["schema_version"],
            "library": library,
            "training_seed": training_seed,
            "weak_validation_path": str(source_validation_path.resolve()),
            "weak_validation_sha256": source_weak_record["sha256"],
        },
        "source_data_model_and_code": source["manifest"]["sources"],
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
        "selection_diagnostics": {
            "epoch0_validation_arm_max_abs_difference": epoch0_arm_parity,
            "epoch0_better": bool(epoch_selection["epoch0_better"].iloc[0]),
            "epoch0_no_worse": bool(epoch_selection["epoch0_no_worse"].iloc[0]),
            "epoch0_minus_selected_log_loss": float(
                epoch_selection["epoch0_minus_selected_log_loss"].iloc[0]
            ),
        },
        "refit_diagnostics": {
            "head_initialization_sha256": head_hash,
            "live_head_initialization_sha256": final_epoch0_audit[
                "live_head_sha256"
            ],
            "live_adapter_initialization_sha256": final_epoch0_audit[
                "live_adapter_sha256"
            ],
            "final_refit_epoch0_arm_max_abs_difference": final_epoch0_audit[
                "max_abs_probability_difference"
            ],
            "frozen_source_reproduction_max_abs_difference": frozen_reproduction_error,
        },
        "model_deltas": model_deltas,
        "rows": {
            "primary": int(len(primary)),
            "primary_positive": int(labels.sum()),
            "primary_negative": int(np.sum(labels == 0)),
            "retention": int(len(retention)),
            "retention_positive": int(retention["target_binder"].sum()),
            "source_validation_prediction_records": int(len(validation)),
            "retention_prediction_records": int(len(retention_predictions)),
        },
        "retention_feature_max_abs_difference": inputs[
            "retention_feature_max_abs_difference"
        ],
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(artifacts.items())
        },
        "code": {
            name: {"path": str(path), "sha256": code_hashes[name]}
            for name, path in sorted(code_paths.items())
        },
        "permissions": {"directory": "0700", "files": "0600"},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "library": library,
                "training_seed": training_seed,
                "chosen_positive_epoch": selected_epoch,
                "epoch0_better": bool(epoch_selection["epoch0_better"].iloc[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="one completed one-library/one-seed finetune_mint_selection_matched.py output",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
