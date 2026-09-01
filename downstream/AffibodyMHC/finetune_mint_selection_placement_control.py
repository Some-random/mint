#!/usr/bin/env python
"""Gated, parameter-matched cross- versus within-chain MINT LoRA control.

This is an exploratory follow-up to the shared-epoch experiment.  It may run
for a library only when that library passes the prespecified placement gate in
the audited six-run shared-epoch aggregate.  The already-trained and verified
cross-chain arm is reused.  A within-chain arm is then trained with the exact
same weak-label rows, head initialization, LoRA initialization, batch order,
positive epoch, optimizer settings, and three-epoch learning-rate horizon.

The sole intended model difference is where the rank-2 q/v LoRA modules are
placed in blocks 31 and 32: ``multimer_attn`` for the source cross-chain arm
versus ``self_attn`` for the new within-chain arm.  Both arms jointly train the
same 2,561-parameter head and 20,480 adapter parameters.
"""

from __future__ import print_function

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import aggregate_mint_selection_shared_epoch as shared_aggregate
from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC import finetune_mint_selection_shared_epoch as shared_trainer
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.finetune_mint_retention import LoRALinear
from mint.helpers.extract import MINTWrapper, load_config


SCHEMA_VERSION = "mint-selection-lora-placement-control-v1"
ARMS = ("lora_cross", "lora_within")
PLACEMENTS = {"lora_cross": "multimer_attn", "lora_within": "self_attn"}
LAYERS = (31, 32)
PROJECTIONS = ("q_proj", "v_proj")
RANK = 2
ALPHA = 4.0
DROPOUT = 0.05
HEAD_PARAMETERS = 2561
ADAPTER_PARAMETERS = 20480
TOTAL_TRAINABLE_PARAMETERS = HEAD_PARAMETERS + ADAPTER_PARAMETERS
EPOCH0_PLACEMENT_PARITY_ATOL = 1e-8
OPTIMIZER_CONTRACT = {"name": "AdamW", "betas": [0.9, 0.98], "eps": 1e-8}
TRAINING_CONTRACT = {
    "max_epochs": 3,
    "batch_size": 64,
    "eval_batch_size": 64,
    "accumulation_steps": 1,
    "head_lr": 1e-4,
    "adapter_lr": 2e-4,
    "weight_decay": 0.01,
    "warmup_fraction": 0.1,
    "clip_norm": 1.0,
}
CORE_METRICS = (
    "global_auroc",
    "global_auprc",
    "global_spearman",
    "within_peptide_macro_spearman",
)
GATE_OUTPUTS = {
    "retention_predictions_by_seed.csv",
    "recomputed_retention_metrics_by_seed.csv",
    "shared_epoch_selection_by_seed.csv",
    "training_audit_by_seed.csv",
    "paired_cross_minus_head_by_seed.csv",
    "mean_sample_sd_summary.csv",
    "within_chain_placement_gate.csv",
    "summary.md",
}
RETENTION_OUTCOME_COLUMNS = frozenset(("target_retention", "target_binder"))


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


class _TorchSaveArtifact(object):
    """Adapt a torch payload to the shared atomic publisher's staging hook."""

    def __init__(self, payload):
        self.payload = payload

    def to_csv(self, path, index=False):
        _require(index is False, "torch artifact staging changed")
        torch.save(self.payload, str(path))


def _cleanup_unopened_claim(output_dir, claimed_identity):
    """Best-effort cleanup only while the final path still names our empty claim."""
    try:
        observed = os.stat(str(output_dir), follow_symlinks=False)
    except OSError:
        return
    if (
        not stat.S_ISDIR(observed.st_mode)
        or shared_aggregate._file_identity(observed) != claimed_identity
    ):
        return
    try:
        if os.listdir(str(output_dir)):
            return
        observed = os.stat(str(output_dir), follow_symlinks=False)
        if shared_aggregate._file_identity(observed) != claimed_identity:
            return
        os.rmdir(str(output_dir))
    except OSError:
        pass


def _atomic_publish_run(
    output_dir,
    artifacts,
    markdown_name,
    markdown,
    manifest_builder,
    precommit=None,
):
    """Lustre-safe no-replace publication with ``manifest.json`` linked last."""
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
    _require(
        Path(markdown_name).name == markdown_name
        and markdown_name not in artifacts
        and markdown_name != "manifest.json",
        "invalid placement summary name",
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
    claimed_output_identity = None
    output_fd_owns_claim = False
    published = False
    staged_manifest_identity = None
    try:
        staged_outputs = {}
        for name, artifact in artifacts.items():
            _require(
                Path(name).name == name and name != "manifest.json",
                "invalid placement output name: {}".format(name),
            )
            path = staging / name
            _write_csv(artifact, path)
            staged_outputs[name] = path
        summary_path = staging / markdown_name
        with open(str(summary_path), "w") as handle:
            handle.write(markdown)
        os.chmod(str(summary_path), 0o600)
        staged_outputs[markdown_name] = summary_path
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
        staged_manifest_identity = shared_aggregate._file_identity(
            os.stat(str(staged_manifest), follow_symlinks=False)
        )
        for path in list(staged_outputs.values()) + [staged_manifest]:
            shared_aggregate._fsync_regular_file(path)
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
        claimed_stat = os.stat(str(output_dir), follow_symlinks=False)
        _require(
            stat.S_ISDIR(claimed_stat.st_mode),
            "claimed placement output is not a directory",
        )
        claimed_output_identity = shared_aggregate._file_identity(claimed_stat)
        open_flags = os.O_RDONLY
        open_flags |= getattr(os, "O_DIRECTORY", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(str(output_dir), open_flags)
        _require(
            shared_aggregate._file_identity(os.fstat(output_fd))
            == claimed_output_identity,
            "claimed placement output directory was replaced before open",
        )
        output_fd_owns_claim = True
        os.fchmod(output_fd, 0o700)
        _require(
            shared_aggregate._path_matches_open_directory(output_dir, output_fd),
            "claimed placement output directory identity changed",
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
                shared_aggregate._path_matches_open_directory(output_dir, output_fd),
                "claimed placement output directory identity changed",
            )
            intended_identity = shared_aggregate._file_identity(
                os.stat(str(path), follow_symlinks=False)
            )
            linked_entries.append((name, intended_identity))
            identity = shared_aggregate._link_staged_file_noreplace(
                path, name, output_fd
            )
            _require(
                identity == intended_identity,
                "published placement output identity changed: {}".format(name),
            )
        _require(
            shared_aggregate._path_matches_open_directory(output_dir, output_fd),
            "claimed placement output directory identity changed before commit",
        )
        _require(
            set(os.listdir(output_fd)) == set(staged_outputs),
            "placement output set changed before commit",
        )
        os.fsync(output_fd)
        if precommit is not None:
            precommit()
        _require(
            shared_aggregate._path_matches_open_directory(output_dir, output_fd),
            "claimed placement output directory identity changed before commit",
        )
        _require(
            set(os.listdir(output_fd)) == set(staged_outputs),
            "placement output set changed before commit",
        )
        for name, path in staged_outputs.items():
            _require(
                sha256_file(path) == output_records[name]["sha256"],
                "staged placement output changed before commit: {}".format(name),
            )
        linked_entries.append(("manifest.json", staged_manifest_identity))
        manifest_identity = shared_aggregate._link_staged_file_noreplace(
            staged_manifest, "manifest.json", output_fd
        )
        _require(
            manifest_identity == staged_manifest_identity,
            "published placement manifest identity changed",
        )
        published = True
        os.fsync(output_fd)
        _require(
            shared_aggregate._path_matches_open_directory(output_dir, output_fd),
            "published placement output directory identity changed",
        )
        _require(
            set(os.listdir(output_fd)) == set(staged_outputs) | {"manifest.json"},
            "published placement output set changed",
        )
        return manifest
    finally:
        if (
            not published
            and claimed_output
            and output_fd is not None
            and staged_manifest_identity is not None
            and output_fd_owns_claim
        ):
            try:
                observed_manifest = os.stat(
                    "manifest.json", dir_fd=output_fd, follow_symlinks=False
                )
            except OSError:
                observed_manifest = None
            if (
                observed_manifest is not None
                and shared_aggregate._file_identity(observed_manifest)
                == staged_manifest_identity
            ):
                published = True
        if not published and claimed_output and output_fd is not None and output_fd_owns_claim:
            shared_aggregate._cleanup_claimed_output(
                output_dir, output_fd, linked_entries
            )
        if (
            not published
            and claimed_output
            and output_fd is None
            and claimed_output_identity is not None
        ):
            _cleanup_unopened_claim(output_dir, claimed_output_identity)
        if output_fd is not None:
            os.close(output_fd)
        if staging.is_dir() and staging.parent == output_dir.parent:
            shutil.rmtree(str(staging))


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


def _same_value(observed, expected, atol=1e-12, rtol=1e-12):
    if isinstance(expected, (bool, np.bool_)):
        return _as_bool(observed, "comparison value") == bool(expected)
    if isinstance(expected, str):
        return str(observed) == expected
    observed = float(observed)
    expected = float(expected)
    if math.isnan(observed) or math.isnan(expected):
        return math.isnan(observed) and math.isnan(expected)
    return math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_outcome_blind_frame(frame, label):
    leaked = sorted(RETENTION_OUTCOME_COLUMNS.intersection(frame.columns))
    _require(
        not leaked,
        "{} contains retention outcome columns: {}".format(label, ", ".join(leaked)),
    )


def _strict_integer_values(series, label):
    values = []
    for value in series.tolist():
        _require(
            not isinstance(value, (bool, np.bool_)),
            "{} contains a Boolean integer alias".format(label),
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError("{} contains a malformed integer".format(label))
        _require(
            math.isfinite(numeric) and numeric.is_integer(),
            "{} contains a non-integral value".format(label),
        )
        values.append(int(numeric))
    return values


def _strict_single_integer(value, label):
    return _strict_integer_values(pd.Series([value]), label)[0]


def _audit_verified_shared_source_contract(run, label):
    """Close permissive numeric aliases left by the upstream source verifier."""
    raw_configuration = run.get("manifest", {}).get("configuration")
    _require(isinstance(raw_configuration, dict), "{} configuration is missing".format(label))
    raw_seeds = raw_configuration.get("training_seeds")
    _require(
        isinstance(raw_seeds, list) and len(raw_seeds) == 1,
        "{} training_seeds must contain exactly one seed".format(label),
    )
    training_seed = int(run["training_seed"])
    _require(
        _strict_single_integer(raw_seeds[0], "{} training_seeds".format(label))
        == training_seed,
        "{} raw training seed changed".format(label),
    )
    source_provenance = run.get("source_provenance")
    _require(
        isinstance(source_provenance, dict)
        and str(source_provenance.get("library")) == str(run["library"])
        and _strict_single_integer(
            source_provenance.get("training_seed"),
            "{} source provenance training_seed".format(label),
        )
        == training_seed,
        "{} matched-source provenance identity changed".format(label),
    )
    matched_configuration = run.get("source", {}).get("configuration")
    _require(
        isinstance(matched_configuration, dict),
        "{} nested matched-source configuration is missing".format(label),
    )
    matched_seeds = matched_configuration.get("training_seeds")
    _require(
        isinstance(matched_seeds, list)
        and len(matched_seeds) == 1
        and _strict_single_integer(
            matched_seeds[0], "{} nested matched training_seeds".format(label)
        )
        == training_seed,
        "{} nested matched-source training seed changed".format(label),
    )
    chosen_epoch = _strict_single_integer(
        raw_configuration.get("chosen_positive_epoch"),
        "{} chosen_positive_epoch".format(label),
    )
    _require(1 <= chosen_epoch <= 3, "{} chosen epoch is not positive".format(label))
    integer_configuration = dict(
        TRAINING_CONTRACT,
        lr_schedule_horizon_epochs=3,
    )
    for name in (
        "max_epochs",
        "batch_size",
        "eval_batch_size",
        "accumulation_steps",
        "lr_schedule_horizon_epochs",
    ):
        _require(
            _strict_single_integer(
                raw_configuration.get(name), "{} {}".format(label, name)
            )
            == int(integer_configuration[name]),
            "{} {} differs from the fixed training contract".format(label, name),
        )
    for name in (
        "head_lr",
        "adapter_lr",
        "weight_decay",
        "warmup_fraction",
        "clip_norm",
    ):
        _require(
            raw_configuration.get(name) == TRAINING_CONTRACT[name],
            "{} {} differs from the fixed training contract".format(label, name),
        )
    library = str(run["library"])
    _require(
        raw_configuration.get("selected_C")
        == matched.PRIMARY_CONTRACT[library]["selected_c"],
        "{} selected_C differs from the canonical contract".format(label),
    )

    selection = run.get("selection")
    _require(isinstance(selection, pd.DataFrame), "{} selection is missing".format(label))
    selection_epochs = _strict_integer_values(
        selection["epoch"], "{} selection epoch".format(label)
    )
    selected_flags = _strict_integer_values(
        selection["selected"], "{} selection selected".format(label)
    )
    selected_epochs = [
        epoch for epoch, selected in zip(selection_epochs, selected_flags) if selected == 1
    ]
    _require(
        selected_epochs == [chosen_epoch],
        "{} selected epoch differs from raw configuration".format(label),
    )

    audit = run.get("training_audit")
    _require(isinstance(audit, pd.DataFrame), "{} training audit is missing".format(label))
    expected_run_seed = matched.derived_seed(training_seed, "final_refit", -1)
    contract = matched.PRIMARY_CONTRACT[library]
    expected_schedule_steps = max(
        1,
        int(math.ceil(float(contract["rows"]) / TRAINING_CONTRACT["batch_size"]))
        * 3,
    )
    expected_sets = {
        "training_seed": {training_seed},
        "fold": {-1},
        "rows": {int(contract["rows"])},
        "positive": {int(contract["positive"])},
        "negative": {int(contract["negative"])},
        "run_seed": {expected_run_seed},
        "selected_epoch": {chosen_epoch},
        "schedule_total_steps": {expected_schedule_steps},
        "lr_schedule_horizon_epochs": {3},
    }
    for name, expected in expected_sets.items():
        _require(
            set(_strict_integer_values(audit[name], "{} audit {}".format(label, name)))
            == expected,
            "{} audit {} changed".format(label, name),
        )
    trainable = {
        str(arm): value
        for arm, value in zip(
            audit["arm"],
            _strict_integer_values(
                audit["trainable_parameters"],
                "{} audit trainable_parameters".format(label),
            ),
        )
    }
    _require(
        trainable == {"head_only": HEAD_PARAMETERS, "lora_cross": TOTAL_TRAINABLE_PARAMETERS},
        "{} audit trainable counts changed".format(label),
    )


def _verify_recorded_file(record, expected_path, label):
    _require(isinstance(record, dict), "{} record is malformed".format(label))
    _require(set(("path", "sha256")).issubset(record), "{} record is incomplete".format(label))
    expected_path = Path(os.path.abspath(str(expected_path)))
    recorded_path = Path(os.path.abspath(str(record["path"])))
    _require(recorded_path == expected_path, "{} path changed".format(label))
    _require(not expected_path.is_symlink(), "{} may not be a symlink".format(label))
    _require(expected_path.is_file(), "{} is missing".format(label))
    observed = sha256_file(expected_path)
    _require(observed == str(record["sha256"]), "{} hash mismatch".format(label))
    return {"path": str(expected_path.resolve()), "sha256": observed}


def _compare_tables(observed, expected, keys, label):
    _require(set(observed.columns) == set(expected.columns), "{} schema changed".format(label))
    observed = observed.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    _require(len(observed) == len(expected), "{} row count changed".format(label))
    for column in expected.columns:
        for left, right in zip(observed[column], expected[column]):
            _require(_same_value(left, right), "{} {} differs".format(label, column))


def _audit_gate_seed_grid(selection, paired):
    """Require the complete prespecified library/seed grid behind the gate."""
    expected_libraries = tuple(matched.LIBRARIES)
    expected_seeds = tuple(int(seed) for seed in matched.DEFAULT_TRAINING_SEEDS)
    _require(len(expected_seeds) == 3 and len(set(expected_seeds)) == 3, "prespecified seed contract changed")

    selection_keys = {"library", "training_seed", "epoch"}
    _require(selection_keys.issubset(selection.columns), "gate selection schema changed")
    selection_identity = list(
        zip(
            selection["library"].astype(str),
            _strict_integer_values(selection["training_seed"], "gate selection training_seed"),
            _strict_integer_values(selection["epoch"], "gate selection epoch"),
        )
    )
    expected_selection = {
        (library, seed, epoch)
        for library in expected_libraries
        for seed in expected_seeds
        for epoch in range(4)
    }
    _require(
        len(selection_identity) == len(expected_selection)
        and len(selection_identity) == len(set(selection_identity))
        and set(selection_identity) == expected_selection,
        "gate selection does not contain exactly LibA/LibB x the three prespecified seeds x epochs 0..3",
    )

    paired_keys = {"library", "training_seed"}
    _require(paired_keys.issubset(paired.columns), "gate paired schema changed")
    paired_identity = list(
        zip(
            paired["library"].astype(str),
            _strict_integer_values(paired["training_seed"], "gate paired training_seed"),
        )
    )
    expected_paired = {
        (library, seed)
        for library in expected_libraries
        for seed in expected_seeds
    }
    _require(
        len(paired_identity) == len(expected_paired)
        and len(paired_identity) == len(set(paired_identity))
        and set(paired_identity) == expected_paired,
        "gate paired changes do not contain exactly the three prespecified seeds per library",
    )


def _verify_gate_source_records(source_records, source_shared):
    """Fully re-audit every run named by the six-run gate manifest."""
    _require(
        isinstance(source_records, list) and len(source_records) == 6,
        "gate lacks exactly six source runs",
    )
    expected_fields = {
        "name",
        "path",
        "library",
        "training_seed",
        "manifest_path",
        "manifest_sha256",
        "source_matched_run",
    }
    parsed = []
    identities = []
    for index, record in enumerate(source_records):
        label = "gate source record {}".format(index)
        _require(
            isinstance(record, dict) and set(record) == expected_fields,
            "{} schema changed".format(label),
        )
        library = str(record["library"])
        try:
            numeric_seed = float(record["training_seed"])
        except (TypeError, ValueError):
            raise ValueError("{} training seed is malformed".format(label))
        _require(
            math.isfinite(numeric_seed) and numeric_seed.is_integer(),
            "{} training seed is malformed".format(label),
        )
        training_seed = int(numeric_seed)
        identities.append((library, training_seed))
        parsed.append((label, record, library, training_seed))

    expected_identities = {
        (library, int(seed))
        for library in matched.LIBRARIES
        for seed in matched.DEFAULT_TRAINING_SEEDS
    }
    _require(
        len(identities) == len(set(identities))
        and set(identities) == expected_identities,
        "gate source records do not contain exactly LibA/LibB x the three prespecified seeds",
    )

    selected_path = Path(source_shared["run_dir"]).resolve()
    selected_identity = (
        str(source_shared["library"]),
        int(source_shared["training_seed"]),
    )
    selected_hash = str(source_shared["manifest_sha256"])
    verified_runs = []
    source_paths = []
    source_found = False
    for label, record, library, training_seed in parsed:
        recorded_lexical = Path(os.path.abspath(str(record["path"])))
        _require(
            Path(str(record["path"])).is_absolute(),
            "{} path is not absolute".format(label),
        )
        _require(
            not recorded_lexical.is_symlink()
            and recorded_lexical == recorded_lexical.resolve(),
            "{} path may not traverse a symlink".format(label),
        )
        _require(recorded_lexical.is_dir(), "{} directory is missing".format(label))
        record_path = recorded_lexical.resolve()
        source_paths.append(record_path)
        _require(
            str(record["name"]) == record_path.name,
            "{} name differs from its path".format(label),
        )

        manifest_lexical = Path(os.path.abspath(str(record["manifest_path"])))
        _require(
            Path(str(record["manifest_path"])).is_absolute(),
            "{} manifest path is not absolute".format(label),
        )
        _require(
            manifest_lexical == recorded_lexical / "manifest.json",
            "{} manifest path changed".format(label),
        )
        _require(
            not manifest_lexical.is_symlink()
            and manifest_lexical == manifest_lexical.resolve(),
            "{} manifest may not traverse a symlink".format(label),
        )
        _require(manifest_lexical.is_file(), "{} manifest is missing".format(label))
        manifest_hash = str(record["manifest_sha256"])
        _require(
            shared_aggregate._is_sha256(manifest_hash),
            "{} manifest hash is malformed".format(label),
        )
        _require(
            sha256_file(manifest_lexical) == manifest_hash,
            "{} manifest hash mismatch".format(label),
        )

        is_selected_record = (
            (library, training_seed) == selected_identity
            and record_path == selected_path
            and manifest_hash == selected_hash
        )
        verified = (
            source_shared
            if is_selected_record
            else shared_aggregate.load_verified_run(record_path)
        )
        _audit_verified_shared_source_contract(verified, label)
        _require(
            Path(verified["run_dir"]).resolve() == record_path,
            "{} verified path changed".format(label),
        )
        _require(
            str(verified["library"]) == library
            and int(verified["training_seed"]) == training_seed,
            "{} verified library/seed identity changed".format(label),
        )
        _require(
            Path(verified["manifest_path"]).resolve() == manifest_lexical
            and str(verified["manifest_sha256"]) == manifest_hash,
            "{} verified manifest binding changed".format(label),
        )
        _require(
            shared_aggregate._normalized_json(record["source_matched_run"])
            == shared_aggregate._normalized_json(verified["source_provenance"]),
            "{} matched-source provenance changed".format(label),
        )
        verified_runs.append(verified)
        source_found = source_found or is_selected_record

    _require(
        len(source_paths) == len(set(source_paths)),
        "gate source paths are not unique",
    )
    _require(source_found, "selected shared run is not bound by the gate aggregate")
    return sorted(
        verified_runs,
        key=lambda run: (str(run["library"]), int(run["training_seed"])),
    )


def load_verified_gate_aggregate(gate_dir, source_shared):
    """Verify the aggregate, independently reconstruct its gate, and bind a source."""
    gate_dir = Path(gate_dir).resolve()
    _require(gate_dir.is_dir(), "shared-epoch aggregate directory is missing")
    manifest_path = gate_dir / "manifest.json"
    _require(manifest_path.is_file(), "shared-epoch aggregate lacks manifest.json")
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == shared_aggregate.SCHEMA_VERSION,
        "shared-epoch aggregate schema changed",
    )
    _require(
        manifest.get("analysis_status") == "retrospective_exploratory",
        "shared-epoch aggregate status changed",
    )
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict) and set(outputs) == GATE_OUTPUTS, "gate aggregate outputs changed")
    verified_outputs = {
        name: _verify_recorded_file(record, gate_dir / name, "gate {}".format(name))
        for name, record in sorted(outputs.items())
    }

    code = manifest.get("code", {})
    _require(
        Path(code.get("path", "")).resolve() == Path(shared_aggregate.__file__).resolve(),
        "gate aggregate code path changed",
    )
    _require(
        sha256_file(Path(shared_aggregate.__file__).resolve()) == str(code.get("sha256")),
        "gate aggregate code hash changed",
    )
    dependencies = manifest.get("dependencies", {})
    expected_dependencies = {
        "shared_epoch_trainer": Path(shared_trainer.__file__).resolve(),
        "matched_aggregator": Path(shared_aggregate.matched_aggregate.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    _require(set(dependencies) == set(expected_dependencies), "gate dependency set changed")
    for name, path in expected_dependencies.items():
        record = dependencies[name]
        _require(Path(record.get("path", "")).resolve() == path, "gate {} path changed".format(name))
        _require(sha256_file(path) == str(record.get("sha256")), "gate {} hash changed".format(name))

    verified_source_runs = _verify_gate_source_records(
        manifest.get("source_runs"), source_shared
    )

    selection = pd.read_csv(gate_dir / "shared_epoch_selection_by_seed.csv", float_precision="round_trip")
    paired = pd.read_csv(gate_dir / "paired_cross_minus_head_by_seed.csv", float_precision="round_trip")
    stored_gate = pd.read_csv(gate_dir / "within_chain_placement_gate.csv", float_precision="round_trip")
    _audit_gate_seed_grid(selection, paired)
    source_combined = shared_aggregate.combine_verified_runs(verified_source_runs)
    source_selection = source_combined["selection"]
    source_paired = shared_aggregate.build_paired_changes(source_combined["metrics"])
    aggregate_contract_fields = {
        "shared_training_configuration": "configuration_contract",
        "shared_runtime": "runtime_contract",
        "common_source_hashes": "common_source_hashes",
        "library_reference_source_hashes": "library_reference_source_hashes",
    }
    for recorded_name, combined_name in aggregate_contract_fields.items():
        _require(
            shared_aggregate._normalized_json(manifest.get(recorded_name))
            == shared_aggregate._normalized_json(source_combined[combined_name]),
            "gate {} differs from verified source runs".format(recorded_name),
        )
    _compare_tables(
        selection,
        source_selection,
        ("library", "training_seed", "epoch"),
        "gate selection versus verified source runs",
    )
    _compare_tables(
        paired,
        source_paired,
        ("library", "training_seed"),
        "gate paired changes versus verified source runs",
    )
    expected_gate = shared_aggregate.build_within_chain_gate(selection, paired)
    _compare_tables(stored_gate, expected_gate, ("library",), "within-chain placement gate")
    library = source_shared["library"]
    row = expected_gate.loc[expected_gate["library"].eq(library)]
    _require(len(row) == 1, "gate has no unique row for {}".format(library))
    row = row.iloc[0]
    required_flags = (
        "cross_log_loss_better_than_epoch0_all_seeds",
        "cross_log_loss_better_than_head_all_seeds",
        "within_peptide_spearman_better_all_seeds",
        "placement_gate_pass",
    )
    for flag in required_flags:
        _require(_as_bool(row[flag], flag), "{} failed {}; control must not run".format(library, flag))
    _require(int(row["n_seeds"]) == 3, "placement gate was not based on three seeds")
    return {
        "run_dir": gate_dir,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "verified_outputs": verified_outputs,
        "verified_source_runs": verified_source_runs,
        "gate_table": expected_gate.copy(),
        "gate_row": {column: row[column] for column in row.index},
    }


def normalized_adapter_state(model):
    """Return adapter tensors with placement-neutral names for exact comparison."""
    output = {}
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            continue
        normalized = name.replace(".multimer_attn.", ".placement_attn.")
        normalized = normalized.replace(".self_attn.", ".placement_attn.")
        _require(normalized not in output, "duplicate normalized adapter name")
        output[normalized] = parameter.detach().cpu().numpy()
    return output


def raw_adapter_state(model):
    return {
        name: parameter.detach().cpu().numpy()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


class MINTPlacementClassifier(nn.Module):
    """MINT pair classifier with q/v LoRA at one explicitly named placement."""

    def __init__(self, config, checkpoint, device, placement, rank, alpha, dropout):
        super().__init__()
        _require(placement in set(PLACEMENTS.values()), "unknown LoRA placement")
        self.placement = str(placement)
        self.wrapper = MINTWrapper(
            load_config(str(config)),
            str(checkpoint),
            freeze_percent=1.0,
            use_multimer=True,
            sep_chains=True,
            device=str(device),
        )
        self.wrapper.model.requires_grad_(False)
        self.adapters = []
        for layer_index in LAYERS:
            attention = getattr(self.wrapper.model.layers[layer_index], self.placement)
            for projection in PROJECTIONS:
                adapter = LoRALinear(
                    getattr(attention, projection),
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                )
                setattr(attention, projection, adapter)
                self.adapters.append(adapter)
        self.head = nn.Linear(2560, 1)
        self.register_buffer("feature_mean", torch.zeros(2560, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.ones(2560, dtype=torch.float32))

    def forward(self, chains, chain_ids):
        features = self.wrapper(chains, chain_ids)
        features = (features - self.feature_mean) / self.feature_scale
        return self.head(features).flatten()

    def set_feature_standardization(self, mean, scale):
        _require(np.asarray(mean).shape == (2560,), "feature mean shape mismatch")
        _require(np.asarray(scale).shape == (2560,), "feature scale shape mismatch")
        _require(bool(np.isfinite(mean).all()), "non-finite feature mean")
        _require(bool(np.isfinite(scale).all()), "non-finite feature scale")
        _require(bool((np.asarray(scale) > 0.0).all()), "nonpositive feature scale")
        with torch.no_grad():
            self.feature_mean.copy_(torch.from_numpy(np.asarray(mean, dtype=np.float32)))
            self.feature_scale.copy_(torch.from_numpy(np.asarray(scale, dtype=np.float32)))

    def set_train_mode(self):
        self.wrapper.model.eval()
        self.head.train()
        for adapter in self.adapters:
            adapter.train()

    def adapter_parameters(self):
        head_ids = {id(parameter) for parameter in self.head.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]


def expected_trainable_names(arm):
    _require(arm in ARMS, "unknown placement arm")
    placement = PLACEMENTS[arm]
    names = {"head.weight", "head.bias"}
    for layer in LAYERS:
        for projection in PROJECTIONS:
            for parameter in ("lora_a", "lora_b"):
                names.add(
                    "wrapper.model.layers.{}.{}.{}.{}".format(
                        layer, placement, projection, parameter
                    )
                )
    return names


def live_trainable_audit(model, arm, rank):
    _require(int(rank) == RANK, "placement control requires rank-2 LoRA")
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    adapter_count = int(sum(parameter.numel() for parameter in model.adapter_parameters()))
    _require(names == expected_trainable_names(arm), "{} trainable names changed".format(arm))
    _require(count == TOTAL_TRAINABLE_PARAMETERS, "{} trainable count changed".format(arm))
    _require(adapter_count == ADAPTER_PARAMETERS, "{} adapter count changed".format(arm))
    return sorted(names), count, adapter_count


def optimizer_for_placement(model, args):
    # The within-placement arm deliberately uses the canonical cross-LoRA
    # optimizer constructor.  ``adapter_parameters`` is placement-neutral, so
    # this preserves the exact group order, decay policy, betas, and epsilon.
    return matched.optimizer_for_live(model, "lora_cross", args)


def audit_optimizer_for_placement(optimizer, model, args):
    """Bind the live optimizer to the canonical three-group LoRA contract."""
    _require(
        type(optimizer) is torch.optim.AdamW,
        "placement optimizer is not AdamW",
    )
    adapters = model.adapter_parameters()
    expected_parameters = ([model.head.weight], [model.head.bias], adapters)
    expected_settings = (
        (float(args.head_lr), float(args.weight_decay)),
        (float(args.head_lr), 0.0),
        (float(args.adapter_lr), float(args.weight_decay)),
    )
    _require(len(optimizer.param_groups) == 3, "placement optimizer group count changed")
    _require(not optimizer.state, "placement optimizer has preexisting state")
    expected_default_keys = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
    }
    _require(
        set(optimizer.defaults) == expected_default_keys
        and float(optimizer.defaults["lr"]) == 1e-3
        and float(optimizer.defaults["weight_decay"]) == 0.01
        and optimizer.defaults["amsgrad"] is False
        and optimizer.defaults["maximize"] is False
        and optimizer.defaults["capturable"] is False
        and optimizer.defaults["foreach"] is None,
        "placement optimizer defaults changed",
    )
    expected_group_keys = {
        "params",
        "lr",
        "base_lr",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
    }
    observed_ids = []
    for index, (group, expected, settings) in enumerate(
        zip(optimizer.param_groups, expected_parameters, expected_settings)
    ):
        _require(
            set(group) == expected_group_keys,
            "placement optimizer group {} schema changed".format(index),
        )
        _require(
            [id(parameter) for parameter in group["params"]]
            == [id(parameter) for parameter in expected],
            "placement optimizer group {} parameters changed".format(index),
        )
        observed_ids.extend(id(parameter) for parameter in group["params"])
        expected_lr, expected_decay = settings
        _require(
            float(group["lr"]) == expected_lr
            and float(group["base_lr"]) == expected_lr
            and float(group["weight_decay"]) == expected_decay,
            "placement optimizer group {} settings changed".format(index),
        )
        _require(
            tuple(float(value) for value in group.get("betas", ()))
            == (0.9, 0.98)
            and float(group.get("eps", float("nan"))) == 1e-8,
            "placement optimizer group {} AdamW settings changed".format(index),
        )
        _require(
            group["amsgrad"] is False
            and group["maximize"] is False
            and group["capturable"] is False
            and group["foreach"] is None,
            "placement optimizer group {} execution flags changed".format(index),
        )
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    _require(
        len(observed_ids) == len(set(observed_ids))
        and set(observed_ids) == set(trainable_ids),
        "placement optimizer does not contain exactly the trainable parameters",
    )
    betas = tuple(float(value) for value in optimizer.defaults.get("betas", ()))
    epsilon = float(optimizer.defaults.get("eps", float("nan")))
    _require(betas == (0.9, 0.98), "placement optimizer betas changed")
    _require(epsilon == 1e-8, "placement optimizer epsilon changed")
    return dict(OPTIMIZER_CONTRACT)


def _require_source_runtime_compatibility(source_shared, device=None):
    runtime = source_shared.get("runtime_contract", {})
    current = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    for name, observed in current.items():
        _require(
            runtime.get(name) == observed,
            "current {} differs from the verified cross-arm runtime".format(name),
        )
    if device is not None:
        _require(
            str(runtime.get("gpu_name")) == str(torch.cuda.get_device_name(device)),
            "current GPU differs from the verified cross-arm runtime",
        )


def _require_deterministic_backend():
    if hasattr(torch.backends, "cudnn"):
        _require(
            not bool(torch.backends.cudnn.benchmark)
            and bool(torch.backends.cudnn.deterministic),
            "deterministic cuDNN policy is not active",
        )


def training_order_sha256(frame, batch_size, run_seed, epochs):
    """Hash the exact RandomSampler order implied by the live DataLoader."""
    pair_uids = frame["pair_uid"].astype(str).tolist()
    generator = torch.Generator()
    generator.manual_seed(int(run_seed))
    loader = DataLoader(
        TensorDataset(torch.arange(len(pair_uids), dtype=torch.int64)),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    digest = hashlib.sha256()
    for epoch in range(1, int(epochs) + 1):
        digest.update("epoch:{}\n".format(epoch).encode("ascii"))
        for batch_number, (indices,) in enumerate(loader):
            digest.update(
                "batch:{}:{}\n".format(epoch, batch_number).encode("ascii")
            )
            for index in indices.tolist():
                digest.update(pair_uids[int(index)].encode("utf-8"))
                digest.update(b"\0")
    return digest.hexdigest()


def _head_hash(model):
    return shared_trainer._hash_named_arrays(
        {
            name: value.detach().cpu().numpy()
            for name, value in model.head.state_dict().items()
        }
    )


def audit_epoch0_placement_pair(
    evaluation_frame,
    mean,
    scale,
    classifier,
    training_seed,
    args,
    device,
    source_adapter_hash,
    expected_head_hash,
):
    """Reconstruct both placements before training and prove exact pairing."""
    probabilities = {}
    head_hashes = {}
    raw_hashes = {}
    normalized_hashes = {}
    trainable = {}
    run_seed = matched.derived_seed(training_seed, "final_refit", -1)
    for arm in ARMS:
        matched.set_deterministic_seed(run_seed)
        _require_deterministic_backend()
        model = MINTPlacementClassifier(
            args.config,
            args.checkpoint,
            device,
            PLACEMENTS[arm],
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
        ).to(device)
        model.set_feature_standardization(mean, scale)
        matched.load_head(model, classifier)
        names, count, adapter_count = live_trainable_audit(model, arm, args.lora_rank)
        trainable[arm] = {
            "names": names,
            "count": count,
            "adapter_count": adapter_count,
        }
        head_hashes[arm] = _head_hash(model)
        raw_hashes[arm] = shared_trainer._hash_named_arrays(raw_adapter_state(model))
        normalized_hashes[arm] = shared_trainer._hash_named_arrays(normalized_adapter_state(model))
        loader = matched.make_loader(evaluation_frame, args.eval_batch_size, False, run_seed)
        _, probability, observed, pair_uids = matched.predict_live(model, loader, device)
        _require(pair_uids == evaluation_frame["pair_uid"].astype(str).tolist(), "epoch-0 UID order changed")
        _require(
            np.array_equal(observed, evaluation_frame["weak_label"].to_numpy(dtype=int)),
            "epoch-0 labels changed",
        )
        probabilities[arm] = probability
        del loader
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _require(set(head_hashes.values()) == {str(expected_head_hash)}, "placement head initialization differs")
    _require(raw_hashes["lora_cross"] == str(source_adapter_hash), "reconstructed cross adapter initialization differs from source")
    _require(len(set(normalized_hashes.values())) == 1, "placement-neutral adapter initialization differs")
    parity = float(np.max(np.abs(probabilities["lora_cross"] - probabilities["lora_within"])))
    _require(parity <= EPOCH0_PLACEMENT_PARITY_ATOL, "cross/within epoch-0 prediction parity failed")
    return {
        "run_seed": int(run_seed),
        "max_abs_probability_difference": parity,
        "head_initialization_sha256": head_hashes["lora_cross"],
        "cross_raw_adapter_initialization_sha256": raw_hashes["lora_cross"],
        "within_raw_adapter_initialization_sha256": raw_hashes["lora_within"],
        "normalized_adapter_initialization_sha256": normalized_hashes["lora_cross"],
        "trainable": trainable,
    }


def train_within_arm(
    train_frame,
    evaluation_frame,
    mean,
    scale,
    classifier,
    training_seed,
    selected_epoch,
    args,
    device,
    expected_epoch0_probability,
):
    """Train only the within-chain placement using the source final-refit seed."""
    train_frame = train_frame.reset_index(drop=True).copy()
    evaluation_frame = evaluation_frame.reset_index(drop=True).copy()
    _require_outcome_blind_frame(train_frame, "within-chain training frame")
    _require_outcome_blind_frame(evaluation_frame, "within-chain evaluation frame")
    _require(1 <= int(selected_epoch) <= int(args.max_epochs), "selected epoch is not positive")
    labels = train_frame["weak_label"].to_numpy(dtype=int)
    class_weights = matched.balanced_class_weights(labels)
    run_seed = matched.derived_seed(training_seed, "final_refit", -1)
    matched.set_deterministic_seed(run_seed)
    _require_deterministic_backend()
    model = MINTPlacementClassifier(
        args.config,
        args.checkpoint,
        device,
        PLACEMENTS["lora_within"],
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    ).to(device)
    model.set_feature_standardization(mean, scale)
    matched.load_head(model, classifier)
    names, trainable_count, adapter_count = live_trainable_audit(
        model, "lora_within", args.lora_rank
    )
    initial_head = {
        name: value.detach().cpu().clone()
        for name, value in model.head.state_dict().items()
    }
    initial_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    normalized_initial_hash = shared_trainer._hash_named_arrays(normalized_adapter_state(model))
    initial_head_hash = _head_hash(model)

    evaluation_loader = matched.make_loader(
        evaluation_frame, args.eval_batch_size, False, run_seed
    )
    _, epoch0_probability, observed, pair_uids = matched.predict_live(
        model, evaluation_loader, device
    )
    _require(pair_uids == evaluation_frame["pair_uid"].astype(str).tolist(), "within epoch-0 UID order changed")
    _require(
        np.array_equal(observed, evaluation_frame["weak_label"].to_numpy(dtype=int)),
        "within epoch-0 labels changed",
    )
    epoch0_error = float(
        np.max(
            np.abs(
                epoch0_probability
                - np.asarray(expected_epoch0_probability, dtype=float)
            )
        )
    )
    _require(
        epoch0_error <= float(args.max_live_feature_probability_difference),
        "within live/cached epoch-0 probability mismatch",
    )

    optimizer = optimizer_for_placement(model, args)
    optimizer_contract = audit_optimizer_for_placement(optimizer, model, args)
    train_loader = matched.make_loader(train_frame, args.batch_size, True, run_seed)
    updates_per_epoch = max(
        1,
        int(math.ceil(float(len(train_loader)) / float(args.accumulation_steps))),
    )
    schedule_total_steps = max(1, updates_per_epoch * int(args.max_epochs))
    global_step = 0
    history = []
    order_digest = hashlib.sha256()
    for epoch in range(1, int(selected_epoch) + 1):
        order_digest.update("epoch:{}\n".format(epoch).encode("ascii"))
        model.set_train_mode()
        optimizer.zero_grad()
        total_weighted_loss = 0.0
        examples = 0
        window_examples = 0
        for batch_number, (chains, chain_ids, weak_target, pair_uids) in enumerate(
            train_loader
        ):
            order_digest.update(
                "batch:{}:{}\n".format(epoch, batch_number).encode("ascii")
            )
            for pair_uid in pair_uids:
                order_digest.update(str(pair_uid).encode("utf-8"))
                order_digest.update(b"\0")
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            weak_target = weak_target.to(device)
            logits = model(chains, chain_ids)
            loss = matched.weighted_bce_mean(
                logits, weak_target, class_weights
            ) * float(len(weak_target))
            loss.backward()
            total_weighted_loss += float(loss.detach().cpu())
            examples += int(len(weak_target))
            window_examples += int(len(weak_target))
            tail = batch_number + 1 == len(train_loader)
            if (batch_number + 1) % int(args.accumulation_steps) == 0 or tail:
                matched.update_learning_rates(
                    optimizer, global_step, schedule_total_steps, args.warmup_fraction
                )
                for parameter in model.parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        parameter.grad.div_(float(window_examples))
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(args.clip_norm),
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                window_examples = 0
        train_loss = total_weighted_loss / float(examples)
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(
            "placement_control lora_within seed {} epoch {} train_loss {:.6f}".format(
                training_seed, epoch, train_loss
            ),
            flush=True,
        )

    _, probability, observed, pair_uids = matched.predict_live(
        model, evaluation_loader, device
    )
    _require(pair_uids == evaluation_frame["pair_uid"].astype(str).tolist(), "within final UID order changed")
    final_head = {
        name: value.detach().cpu().clone()
        for name, value in model.head.state_dict().items()
    }
    final_adapter = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    head_delta = matched.parameter_delta(initial_head, final_head)
    adapter_delta = matched.parameter_delta(initial_adapter, final_adapter)
    _require(head_delta["l2"] > 0.0, "within-chain head did not move")
    _require(adapter_delta["l2"] > 0.0, "within-chain adapter did not move")
    model_state = {
        "head_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.head.state_dict().items()
        },
        "adapter_state": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        },
    }
    return {
        "probability": probability,
        "observed": observed,
        "pair_uids": pair_uids,
        "history": history,
        "model_state": model_state,
        "trainable_names": names,
        "trainable_count": trainable_count,
        "adapter_count": adapter_count,
        "class_weights": class_weights,
        "run_seed": int(run_seed),
        "epoch0_probability_max_abs_error": epoch0_error,
        "schedule_total_steps": int(schedule_total_steps),
        "head_parameter_delta": head_delta,
        "adapter_parameter_delta": adapter_delta,
        "normalized_adapter_initialization_sha256": normalized_initial_hash,
        "head_initialization_sha256": initial_head_hash,
        "optimizer": optimizer_contract,
        "training_order_sha256": order_digest.hexdigest(),
    }


def paired_difference_record(metrics, library, training_seed, selected_epoch):
    by_arm = metrics.set_index("arm")
    _require(set(by_arm.index) == set(ARMS), "placement metric arms missing")
    row = {
        "library": library,
        "training_seed": int(training_seed),
        "selected_epoch": int(selected_epoch),
        "comparison": "lora_within_minus_lora_cross",
    }
    for metric in CORE_METRICS:
        row[metric + "_change"] = float(
            by_arm.loc["lora_within", metric]
            - by_arm.loc["lora_cross", metric]
        )
    return row


def preflight(args):
    """Run the complete CPU-side gate, data, hyperparameter, and lineage audit."""
    source_shared = shared_aggregate.load_verified_run(
        Path(args.source_shared_run_dir).resolve()
    )
    _audit_verified_shared_source_contract(source_shared, "selected shared source")
    gate = load_verified_gate_aggregate(
        Path(args.gate_aggregate_dir).resolve(), source_shared
    )
    configuration = source_shared["configuration"]
    lora = configuration.get("lora", {})
    _require(
        int(lora.get("rank", -1)) == RANK
        and float(lora.get("alpha", float("nan"))) == ALPHA
        and float(lora.get("dropout", float("nan"))) == DROPOUT
        and lora.get("layers_zero_based") == list(LAYERS)
        and lora.get("projections") == list(PROJECTIONS)
        and lora.get("placement") == "multimer_attn",
        "source cross-LoRA contract changed",
    )
    matched_source = source_shared["source"]
    inputs = shared_trainer._load_prefit_cache(matched_source)
    primary = inputs["primary"]
    _require_outcome_blind_frame(primary, "placement-control primary training data")
    _require_source_runtime_compatibility(source_shared)
    labels = primary["weak_label"].to_numpy(dtype=int)
    full_mean, full_scale = matched.fit_standardizer(inputs["weak_features"])
    full_x = matched.standardize(inputs["weak_features"], full_mean, full_scale)
    selected_c = float(configuration["selected_C"])
    classifier = matched.fit_logistic(full_x, labels, selected_c)
    head_hash = shared_trainer.head_initialization_sha256(classifier)
    source_audit = source_shared["training_audit"]
    source_cross_audit = source_audit.loc[source_audit["arm"].eq("lora_cross")]
    _require(len(source_cross_audit) == 1, "source lacks one cross-LoRA final audit")
    source_cross_audit = source_cross_audit.iloc[0]
    _require(
        str(source_cross_audit["head_initialization_sha256"]) == head_hash,
        "reconstructed head initialization differs from source",
    )
    selected_epoch = int(configuration["chosen_positive_epoch"])
    _require(1 <= selected_epoch <= 3, "source shared epoch is not positive")
    train_args = shared_trainer.training_arguments_from_verified_source(
        matched_source, getattr(args, "device", "cuda:0")
    )
    _require(
        train_args.lora_rank == RANK
        and train_args.lora_alpha == ALPHA
        and train_args.lora_dropout == DROPOUT,
        "reconstructed LoRA hyperparameters changed",
    )
    for name, expected in TRAINING_CONTRACT.items():
        _require(
            getattr(train_args, name) == expected,
            "placement-control {} differs from the fixed training contract".format(
                name
            ),
        )
    run_seed = matched.derived_seed(source_shared["training_seed"], "final_refit", -1)
    _require(int(source_cross_audit["run_seed"]) == run_seed, "source cross run seed changed")
    order_hash = training_order_sha256(
        primary, train_args.batch_size, run_seed, selected_epoch
    )
    contract = matched.PRIMARY_CONTRACT[source_shared["library"]]
    _require(
        len(primary) == int(contract["rows"])
        and int(labels.sum()) == int(contract["positive"])
        and cached_eval.membership_sha256(primary) == contract["membership_sha256"],
        "placement-control primary data changed",
    )
    return {
        "source_shared": source_shared,
        "gate": gate,
        "inputs": inputs,
        "classifier": classifier,
        "full_mean": full_mean,
        "full_scale": full_scale,
        "head_hash": head_hash,
        "selected_epoch": selected_epoch,
        "train_args": train_args,
        "run_seed": run_seed,
        "training_order_sha256": order_hash,
        "source_cross_audit": source_cross_audit,
        "public_audit": {
            "library": source_shared["library"],
            "training_seed": int(source_shared["training_seed"]),
            "selected_epoch": selected_epoch,
            "primary_rows": int(len(primary)),
            "primary_positive": int(labels.sum()),
            "primary_negative": int(np.sum(labels == 0)),
            "primary_membership_sha256": contract["membership_sha256"],
            "head_initialization_sha256": head_hash,
            "training_run_seed": run_seed,
            "training_order_sha256": order_hash,
            "cross_trainable_parameters": int(source_cross_audit["trainable_parameters"]),
            "within_expected_trainable_parameters": TOTAL_TRAINABLE_PARAMETERS,
            "adapter_parameters_per_arm": ADAPTER_PARAMETERS,
            "placement_gate_pass": True,
            "source_shared_manifest_sha256": source_shared["manifest_sha256"],
            "gate_aggregate_manifest_sha256": gate["manifest_sha256"],
        },
    }


def _placement_input_artifact_paths(prefit):
    """Return every verified artifact tree that publication must not overlap."""
    paths = {
        Path(prefit["gate"]["run_dir"]).resolve(),
        Path(prefit["source_shared"]["run_dir"]).resolve(),
        Path(prefit["source_shared"]["source"]["run_dir"]).resolve(),
    }
    for run in prefit["gate"]["verified_source_runs"]:
        paths.add(Path(run["run_dir"]).resolve())
        paths.add(Path(run["source"]["run_dir"]).resolve())
        reference = run["source"].get("reference", {})
        if isinstance(reference, dict) and "directory" in reference:
            paths.add(Path(reference["directory"]).resolve())
        for name, record in run["source"].get("sources", {}).items():
            if isinstance(record, dict) and "path" in record:
                source_path = Path(record["path"]).resolve()
                paths.add(source_path)
                if str(name).startswith("cache_"):
                    paths.add(source_path.parent)
    reconstructed_cache = Path(prefit["train_args"].cache).resolve()
    if reconstructed_cache.is_dir():
        paths.add(reconstructed_cache)
    return sorted(paths, key=str)


def _recheck_inputs(prefit, code_hashes):
    source = prefit["source_shared"]
    shared_aggregate._recheck_verified_inputs(
        prefit["gate"]["verified_source_runs"]
    )
    _require(
        sha256_file(source["manifest_path"]) == source["manifest_sha256"],
        "source shared manifest changed during run",
    )
    gate = prefit["gate"]
    _require(
        sha256_file(gate["manifest_path"]) == gate["manifest_sha256"],
        "gate aggregate manifest changed during run",
    )
    for name, record in gate["verified_outputs"].items():
        _require(
            sha256_file(Path(record["path"])) == record["sha256"],
            "gate {} changed during run".format(name),
        )
    for name, expected in code_hashes.items():
        path = {
            "script": Path(__file__).resolve(),
            "shared_aggregator": Path(shared_aggregate.__file__).resolve(),
            "shared_trainer": Path(shared_trainer.__file__).resolve(),
            "matched_trainer": Path(matched.__file__).resolve(),
            "canonical_evaluator": Path(cached_eval.__file__).resolve(),
        }[name]
        _require(sha256_file(path) == expected, "{} changed during run".format(name))


def run(args):
    started = time.time()
    code_paths = {
        "script": Path(__file__).resolve(),
        "shared_aggregator": Path(shared_aggregate.__file__).resolve(),
        "shared_trainer": Path(shared_trainer.__file__).resolve(),
        "matched_trainer": Path(matched.__file__).resolve(),
        "canonical_evaluator": Path(cached_eval.__file__).resolve(),
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    audit_only = bool(getattr(args, "audit_only", False))
    requested_output = None
    if not audit_only:
        _require(
            args.output_dir is not None,
            "--output-dir is required unless --audit-only is used",
        )
        requested_output = Path(os.path.abspath(str(args.output_dir)))
        _require(
            not os.path.lexists(str(requested_output)),
            "output directory exists; refusing overwrite",
        )
    prefit = preflight(args)
    if audit_only:
        _recheck_inputs(prefit, code_hashes)
        print(json.dumps(prefit["public_audit"], indent=2, sort_keys=True))
        return prefit["public_audit"]

    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not os.path.lexists(str(output_dir)), "output directory exists; refusing overwrite")
    protected_inputs = _placement_input_artifact_paths(prefit)
    shared_aggregate._assert_output_disjoint(output_dir, protected_inputs)
    _recheck_inputs(prefit, code_hashes)
    _require(torch.cuda.is_available(), "CUDA is required for the placement-control refit")

    source_shared = prefit["source_shared"]
    matched_source = source_shared["source"]
    library = source_shared["library"]
    training_seed = int(source_shared["training_seed"])
    selected_epoch = int(prefit["selected_epoch"])
    primary = prefit["inputs"]["primary"]
    retention_identity = prefit["inputs"]["retention_identity"]
    train_args = prefit["train_args"]
    full_mean = prefit["full_mean"]
    full_scale = prefit["full_scale"]
    classifier = prefit["classifier"]
    frozen_x = matched.standardize(
        prefit["inputs"]["retention_features"], full_mean, full_scale
    )
    frozen_probability = classifier.predict_proba(frozen_x)[:, 1]

    device = torch.device(str(args.device))
    torch.cuda.set_device(device)
    _require_source_runtime_compatibility(source_shared, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    final_evaluation = retention_identity.copy()
    _require_outcome_blind_frame(final_evaluation, "placement-control evaluation identity")
    final_evaluation["weak_label"] = 0
    source_adapter_hash = source_shared["manifest"]["refit_diagnostics"][
        "live_adapter_initialization_sha256"
    ]
    epoch0_audit = audit_epoch0_placement_pair(
        final_evaluation,
        full_mean,
        full_scale,
        classifier,
        training_seed,
        train_args,
        device,
        source_adapter_hash,
        prefit["head_hash"],
    )
    _require(
        int(epoch0_audit["run_seed"]) == int(prefit["run_seed"]),
        "placement pairing run seed changed",
    )
    within = train_within_arm(
        primary,
        final_evaluation,
        full_mean,
        full_scale,
        classifier,
        training_seed,
        selected_epoch,
        train_args,
        device,
        frozen_probability,
    )
    _require(
        within["normalized_adapter_initialization_sha256"]
        == epoch0_audit["normalized_adapter_initialization_sha256"],
        "trained within arm did not use the audited adapter initialization",
    )
    _require(
        within["head_initialization_sha256"] == prefit["head_hash"],
        "trained within arm did not use the audited head initialization",
    )
    _require(
        within["optimizer"] == OPTIMIZER_CONTRACT,
        "trained within arm optimizer contract changed",
    )
    _require(
        int(within["run_seed"]) == int(prefit["source_cross_audit"]["run_seed"]),
        "cross and within arms used different run seeds",
    )
    _require(
        int(within["schedule_total_steps"])
        == int(prefit["source_cross_audit"]["schedule_total_steps"]),
        "cross and within arms used different LR horizons",
    )
    _require(
        within["training_order_sha256"] == prefit["training_order_sha256"],
        "within arm did not follow the preflight-audited training order",
    )

    # Only now use the gate-audited source predictions/retention panel to score
    # the new arm.  The source verifier necessarily inspected those values to
    # authenticate the retention-informed gate, but they never enter the
    # within-chain optimizer or its epoch choice.
    retention = shared_trainer.load_retention_targets_after_predictions(
        matched_source, retention_identity
    )
    source_cross_predictions = source_shared["predictions"].loc[
        source_shared["predictions"]["arm"].eq("lora_cross")
    ].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    _require(
        source_cross_predictions["pair_uid"].astype(str).equals(
            retention["pair_uid"].astype(str)
        ),
        "source cross predictions differ from retention panel",
    )
    _require(
        within["pair_uids"] == retention["pair_uid"].astype(str).tolist(),
        "within predictions differ from retention panel",
    )
    probabilities = {
        "lora_cross": source_cross_predictions["probability"].to_numpy(dtype=float),
        "lora_within": np.asarray(within["probability"], dtype=float),
    }
    prediction_blocks = []
    metric_rows = []
    per_peptide_rows = []
    for arm in ARMS:
        block = retention[
            [
                "pair_uid",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
                "target_retention",
                "target_binder",
            ]
        ].copy()
        block["library"] = library
        block["arm"] = arm
        block["training_seed"] = training_seed
        block["selected_epoch"] = selected_epoch
        block["probability"] = probabilities[arm]
        prediction_blocks.append(block)
        metric_rows.append(
            matched.retention_metric_record(
                retention,
                probabilities[arm],
                library,
                arm,
                training_seed,
                selected_epoch,
            )
        )
        per_peptide_rows.extend(
            matched.per_peptide_records(
                retention,
                probabilities[arm],
                library,
                arm,
                training_seed,
                selected_epoch,
            )
        )
    retention_predictions = pd.concat(prediction_blocks, ignore_index=True)
    retention_metrics = pd.DataFrame(metric_rows)
    source_cross_metrics = source_shared["metrics"].loc[
        source_shared["metrics"]["arm"].eq("lora_cross")
    ].iloc[0]
    observed_cross = retention_metrics.loc[
        retention_metrics["arm"].eq("lora_cross")
    ].iloc[0]
    for metric in CORE_METRICS:
        _require(
            _same_value(observed_cross[metric], source_cross_metrics[metric]),
            "reused source cross {} changed".format(metric),
        )
    paired = pd.DataFrame(
        [
            paired_difference_record(
                retention_metrics, library, training_seed, selected_epoch
            )
        ]
    )
    per_peptide = pd.DataFrame(per_peptide_rows)

    source_cross_audit = prefit["source_cross_audit"]
    common_training_fields = (
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
    )
    within_audit = matched.training_audit_record(
        "final_refit",
        "lora_cross",  # same record schema/count formula; relabeled below
        training_seed,
        -1,
        primary,
        full_mean,
        full_scale,
        {
            "trainable_names": within["trainable_names"],
            "trainable_count": within["trainable_count"],
            "class_weights": within["class_weights"],
            "run_seed": within["run_seed"],
            "epoch0_probability_max_abs_error": within[
                "epoch0_probability_max_abs_error"
            ],
            "schedule_total_steps": within["schedule_total_steps"],
            "head_parameter_delta": within["head_parameter_delta"],
            "adapter_parameter_delta": within["adapter_parameter_delta"],
        },
        selected_epoch,
    )
    within_audit["arm"] = "lora_within"
    within_audit.update(
        {
            "head_initialization_sha256": prefit["head_hash"],
            "lr_schedule_horizon_epochs": 3,
            "source_run_manifest_sha256": source_shared["source"]["manifest_sha256"],
            "source_validation_epoch0_arm_max_abs_difference": source_cross_audit[
                "source_validation_epoch0_arm_max_abs_difference"
            ],
            "final_refit_epoch0_arm_max_abs_difference": epoch0_audit[
                "max_abs_probability_difference"
            ],
            "source_frozen_reproduction_max_abs_difference": source_cross_audit[
                "source_frozen_reproduction_max_abs_difference"
            ],
        }
    )
    cross_audit = source_cross_audit.to_dict()
    training_rows = []
    for arm, record, arm_source in (
        ("lora_cross", cross_audit, "reused_verified_shared_epoch"),
        ("lora_within", within_audit, "new_parameter_matched_refit"),
    ):
        row = dict(record)
        row.update(
            {
                "arm": arm,
                "arm_source": arm_source,
                "attention_placement": PLACEMENTS[arm],
                "adapter_parameters": ADAPTER_PARAMETERS,
                "normalized_adapter_initialization_sha256": epoch0_audit[
                    "normalized_adapter_initialization_sha256"
                ],
                "training_order_sha256": prefit["training_order_sha256"],
                "source_shared_manifest_sha256": source_shared["manifest_sha256"],
                "gate_aggregate_manifest_sha256": prefit["gate"]["manifest_sha256"],
            }
        )
        training_rows.append(row)
    training_audit = pd.DataFrame(training_rows)
    for field in common_training_fields:
        _require(
            training_audit[field].nunique(dropna=False) == 1,
            "cross/within arms differ in {}".format(field),
        )
    _require(
        set(training_audit["trainable_parameters"].astype(int))
        == {TOTAL_TRAINABLE_PARAMETERS},
        "cross/within total trainable counts differ",
    )
    _require(
        set(training_audit["adapter_parameters"].astype(int))
        == {ADAPTER_PARAMETERS},
        "cross/within adapter counts differ",
    )

    placement_contract = pd.DataFrame(
        [
            {
                **prefit["public_audit"],
                "cross_attention_placement": "multimer_attn",
                "within_attention_placement": "self_attn",
                "layers_zero_based": "31,32",
                "projections": "q_proj,v_proj",
                "lora_rank": RANK,
                "lora_alpha": ALPHA,
                "lora_dropout": DROPOUT,
                "cross_within_epoch0_max_abs_probability_difference": epoch0_audit[
                    "max_abs_probability_difference"
                ],
                "normalized_adapter_initialization_sha256": epoch0_audit[
                    "normalized_adapter_initialization_sha256"
                ],
                "identical_rows": True,
                "identical_head_initialization": True,
                "identical_adapter_initialization": True,
                "identical_batch_order": True,
                "identical_shared_epoch": True,
                "identical_optimizer_schedule": True,
                "identical_trainable_parameter_count": True,
            }
        ]
    )

    _recheck_inputs(prefit, code_hashes)
    _require(not os.path.lexists(str(output_dir)), "output directory appeared during run")
    shared_aggregate._assert_output_disjoint(output_dir, protected_inputs)
    artifacts = {
        "placement_contract.csv": placement_contract,
        "training_audit.csv": training_audit,
        "retention_predictions.csv": retention_predictions,
        "retention_metrics.csv": retention_metrics,
        "paired_differences.csv": paired,
        "per_peptide_metrics.csv": per_peptide,
    }
    model_filename = "model_delta_seed{}_lora_within.pt".format(training_seed)
    artifacts[model_filename] = _TorchSaveArtifact(
        {
            "library": library,
            "arm": "lora_within",
            "attention_placement": "self_attn",
            "training_seed": training_seed,
            "selected_epoch": selected_epoch,
            "selected_C": float(source_shared["configuration"]["selected_C"]),
            "head_initialization_sha256": prefit["head_hash"],
            "normalized_adapter_initialization_sha256": epoch0_audit[
                "normalized_adapter_initialization_sha256"
            ],
            "base_checkpoint_sha256": matched_source["sources"]["checkpoint"]["sha256"],
            "primary_membership_sha256": matched.PRIMARY_CONTRACT[library][
                "membership_sha256"
            ],
            "source_shared_manifest_sha256": source_shared["manifest_sha256"],
            "gate_aggregate_manifest_sha256": prefit["gate"]["manifest_sha256"],
            "feature_mean": torch.from_numpy(np.asarray(full_mean, dtype=np.float32)),
            "feature_scale": torch.from_numpy(np.asarray(full_scale, dtype=np.float32)),
            **within["model_state"]
        }
    )
    delta = paired.iloc[0]
    summary_lines = [
        "# MINT LoRA placement control: {} seed {}".format(library, training_seed),
        "",
        "The prespecified library-level placement gate passed. The verified cross-chain arm was compared with a parameter-matched within-chain LoRA refit at the same shared epoch.",
        "",
        "| Within-chain minus cross-chain | Value |",
        "|---|---:|",
        "| Within-peptide Spearman | {:.4f} |".format(
            delta["within_peptide_macro_spearman_change"]
        ),
        "| Global Spearman | {:.4f} |".format(delta["global_spearman_change"]),
        "| AP | {:.4f} |".format(delta["global_auprc_change"]),
        "| AUROC | {:.4f} |".format(delta["global_auroc_change"]),
        "",
        "Because measured retention participated in the gate, this placement comparison is exploratory.",
        "",
    ]
    summary_markdown = "\n".join(summary_lines)

    def build_manifest(output_records):
        _recheck_inputs(prefit, code_hashes)
        shared_aggregate._assert_output_disjoint(output_dir, protected_inputs)
        _require(
            not os.path.lexists(str(output_dir)),
            "output directory appeared before manifest construction",
        )
        return {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory_retention_gated",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "hostname": socket.gethostname(),
        "argv": list(sys.argv),
        "configuration": {
            "library": library,
            "training_seed": training_seed,
            "arms": list(ARMS),
            "arm_reuse": {"lora_cross": "verified shared-epoch source", "lora_within": "new refit"},
            "selected_C": float(source_shared["configuration"]["selected_C"]),
            "shared_positive_epoch": selected_epoch,
            "lr_schedule_horizon_epochs": 3,
            "batch_size": int(train_args.batch_size),
            "eval_batch_size": int(train_args.eval_batch_size),
            "accumulation_steps": int(train_args.accumulation_steps),
            "head_lr": float(train_args.head_lr),
            "adapter_lr": float(train_args.adapter_lr),
            "weight_decay": float(train_args.weight_decay),
            "warmup_fraction": float(train_args.warmup_fraction),
            "clip_norm": float(train_args.clip_norm),
            "optimizer": within["optimizer"],
            "lora": {
                "rank": RANK,
                "alpha": ALPHA,
                "dropout": DROPOUT,
                "layers_zero_based": list(LAYERS),
                "projections": list(PROJECTIONS),
                "placements": {
                    "lora_cross": "multimer_attn",
                    "lora_within": "self_attn",
                },
                "adapter_parameters_per_arm": ADAPTER_PARAMETERS,
                "total_trainable_parameters_per_arm": TOTAL_TRAINABLE_PARAMETERS,
            },
            "primary_endpoint": "within_peptide_macro_spearman",
            "secondary_endpoints": ["global_spearman", "global_auprc", "global_auroc"],
            "retention_threshold_for_auroc_ap": 75.0,
            "gate_policy": "library must pass all three seeds on all three predeclared conditions; ties, NaNs, or one failed seed block the control",
            "retention_usage": "measured retention participates in the predeclared gate and final metrics, but not the within-chain refit",
        },
        "canonical_contract": matched.PRIMARY_CONTRACT[library],
        "source_shared_run": {
            "path": str(source_shared["run_dir"]),
            "manifest_path": str(source_shared["manifest_path"]),
            "manifest_sha256": source_shared["manifest_sha256"],
            "schema_version": source_shared["manifest"]["schema_version"],
        },
        "gate_aggregate": {
            "path": str(prefit["gate"]["run_dir"]),
            "manifest_path": str(prefit["gate"]["manifest_path"]),
            "manifest_sha256": prefit["gate"]["manifest_sha256"],
            "schema_version": prefit["gate"]["manifest"]["schema_version"],
            "library_gate": {
                key: (
                    bool(_as_bool(value, key))
                    if key.endswith("all_seeds") or key == "placement_gate_pass"
                    else _json_scalar(value)
                )
                for key, value in prefit["gate"]["gate_row"].items()
            },
        },
        "pairing_diagnostics": {
            "training_order_sha256": prefit["training_order_sha256"],
            **epoch0_audit
        },
        "rows": {
            "primary": int(len(primary)),
            "primary_positive": int(primary["weak_label"].sum()),
            "primary_negative": int(primary["weak_label"].eq(0).sum()),
            "retention": int(len(retention)),
            "retention_positive": int(retention["target_binder"].sum()),
            "retention_prediction_records": int(len(retention_predictions)),
        },
        "model_delta": {
            "filename": model_filename,
            "sha256": output_records[model_filename]["sha256"],
            "arm": "lora_within",
            "attention_placement": "self_attn",
            "training_seed": training_seed,
            "selected_epoch": selected_epoch,
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
        "outputs": output_records,
        "code": {
            name: {"path": str(path), "sha256": code_hashes[name]}
            for name, path in sorted(code_paths.items())
        },
        "permissions": {"directory": "0700", "files": "0600"},
    }

    def precommit_recheck():
        _recheck_inputs(prefit, code_hashes)
        shared_aggregate._assert_output_disjoint(output_dir, protected_inputs)

    manifest = _atomic_publish_run(
        output_dir,
        artifacts,
        "run_summary.md",
        summary_markdown,
        build_manifest,
        precommit=precommit_recheck,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "library": library,
                "training_seed": training_seed,
                "shared_positive_epoch": selected_epoch,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-shared-run-dir",
        type=Path,
        required=True,
        help="one completed, audited shared-epoch library/seed artifact",
    )
    parser.add_argument(
        "--gate-aggregate-dir",
        type=Path,
        required=True,
        help="the completed six-run shared-epoch aggregate containing the gate",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="verify gate/data/provenance and print the run contract without CUDA or writes",
    )
    args = parser.parse_args(argv)
    if not args.audit_only and args.output_dir is None:
        parser.error("--output-dir is required unless --audit-only is used")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
