#!/usr/bin/env python3
"""Aggregate the locked five-seed LibA ESMFold2 weak-label CV experiment.

This command is deliberately retention-blind.  It accepts only the canonical
weak-label sidecar, the label-safe sequence-model OOF reference, the frozen
feature-cache receipt, and completed runs made by
``run_esmfold2_liba_replicated_cv_one.py``.  Before computing any result it
verifies the hashes binding every run receipt to its predictions and verifies
that every run covers the exact same 7,515 row/fold assignments as the frozen
MINT layer-9 reference.

The five optimization replicates are combined by averaging logits for each
row and applying a sigmoid.  Family selection and the structure-evidence gate
use only weak labels; direct-retention outcomes cannot be passed to this
program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import run_esmfold2_liba_replicated_cv_one as runner
from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import MODEL_NAMES


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()

SCHEMA_VERSION = "esmfold2-liba-replicated-cv-aggregate-v1"
COMPLETION_SCHEMA_VERSION = "esmfold2-liba-replicated-cv-aggregate-completion-v1"
EXPECTED_OOF_ROWS = 7_515
EXPECTED_OOF_POSITIVE = 3_759
EXPECTED_OOF_MEMBERSHIP_SHA256 = (
    "b20d5a9858ff553cbff89fa99e2e44d8d19bad1342e0ce17d2d24308aa844961"
)
EXPECTED_FOLD_ROWS = {0: 2_953, 1: 2_263, 2: 2_299}
EXPECTED_FOLD_POSITIVE = {0: 1_530, 1: 1_267, 2: 962}
EXPECTED_FOLD_MEMBERSHIP_SHA256 = {
    0: "a87340dabc5e9c39e6e1ba90867e9e9ea6e2062cd9484bf32573485d82932e82",
    1: "dbdaf4d6521fdcafa9e57310c7722e11bc7bbb3081685f2ef5bd4c083b1f8307",
    2: "cd586ecb211725cb76f5de482a0826ed8f79803e44e1c37fc19d3ab9fa5156d6",
}
EXPECTED_WITHIN_PEPTIDE_GROUPS = 189
MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP = 0.7097951490678114
AP_BEAT_TOLERANCE = 1e-12
METRIC_RECEIPT_TOLERANCE = 1e-8

FIXED_REPLICATE_SEEDS = tuple(runner.FIXED_REPLICATE_SEEDS)
PRETRUNK_CONTROL = "single_inputs_only"
STRUCTURE_DERIVED_FAMILIES = (
    "distogram_only",
    "pair_state_only",
    "distogram_pair",
    "full",
)
FAMILY_ROLE = {
    "distogram_only": "folding_derived",
    "pair_state_only": "folding_derived",
    "distogram_pair": "folding_derived",
    PRETRUNK_CONTROL: "pre_trunk_sequence_control",
    "full": "folding_derived",
}
FORBIDDEN_DATA_FIELD_PARTS = ("retention", "outcome", "target", "binder")
ALLOWED_SAFETY_FIELDS = ("retention_labels_read", "retention_labels_allowed")

DEFAULT_CONFIG = (
    REPO_ROOT / "downstream/AffibodyMHC/configs/esmfold2_liba_frozen_readouts_v1.json"
)
DEFAULT_TRAINING_LABELS = (
    REPO_ROOT / "private_data/derived/esmfold2_liba_training_labels_v1/training_labels.csv"
)
DEFAULT_TRAINING_LABELS_MANIFEST = (
    REPO_ROOT / "private_data/derived/esmfold2_liba_training_labels_v1/manifest.json"
)
DEFAULT_REFERENCE_OOF = (
    REPO_ROOT
    / "private_data/experiments/liba_common_oof_sequence_predictions_v1/weak_oof_aligned.csv.gz"
)


@dataclass(frozen=True)
class InputContract:
    config_path: Path
    config: Mapping[str, Any]
    config_sha256: str
    labels_path: Path
    labels_sha256: str
    labels_manifest_path: Path
    labels_manifest_sha256: str
    feature_cache: Path
    cache_manifest_path: Path
    cache_manifest_sha256: str
    cache_payload_verification: Mapping[str, Any]
    reference_path: Path
    reference_sha256: str
    reference: pd.DataFrame
    reference_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedRun:
    family: str
    replicate_seed: int
    run_dir: Path
    contract: Mapping[str, Any]
    summary: Mapping[str, Any]
    selected_epoch_in_run: int
    predictions: pd.DataFrame
    source_hashes: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _membership_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, values))).encode("ascii")
    ).hexdigest()


def _row_fold_sha256(frame: pd.DataFrame) -> str:
    selected = frame[["row_id", "fold"]].sort_values("row_id")
    payload = "\n".join(
        selected["row_id"].astype(str) + "\t" + selected["fold"].astype(str)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _validate_output_dir(path: Path) -> Path:
    output = path.resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    return output


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _forbidden_field_names(columns: Iterable[str]) -> list[str]:
    return sorted(
        column
        for column in map(str, columns)
        if any(part in column.lower() for part in FORBIDDEN_DATA_FIELD_PARTS)
    )


def _assert_no_outcome_keys(value: Any, source: str, prefix: str = "") -> None:
    """Reject outcome-like JSON fields, except explicit false safety flags."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            lowered = key.lower()
            forbidden = any(part in lowered for part in FORBIDDEN_DATA_FIELD_PARTS)
            if forbidden:
                _require(
                    key in ALLOWED_SAFETY_FIELDS and child is False,
                    f"{source} contains forbidden outcome field {path}",
                )
                continue
            _assert_no_outcome_keys(child, source, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_outcome_keys(child, source, f"{prefix}[{index}]")


def _metric_bundle(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    labels = frame["weak_label"].to_numpy(dtype=int)
    scores = frame[score_column].to_numpy(dtype=float)
    _require(set(np.unique(labels)) == {0, 1}, "metric labels are not binary")
    _require(bool(np.isfinite(scores).all()), "metric scores are non-finite")
    _require(
        bool(((scores >= 0.0) & (scores <= 1.0)).all()),
        "metric probabilities must be between zero and one",
    )
    within: list[tuple[float, float, float]] = []
    for _, group in frame.groupby("peptide_id", sort=True):
        group_labels = group["weak_label"].to_numpy(dtype=int)
        if np.unique(group_labels).size != 2:
            continue
        group_scores = group[score_column].to_numpy(dtype=float)
        within.append(
            (
                float(average_precision_score(group_labels, group_scores)),
                float(roc_auc_score(group_labels, group_scores)),
                float(log_loss(group_labels, group_scores, labels=[0, 1])),
            )
        )
    _require(bool(within), "no peptide has both weak-label classes")
    return {
        "rows": int(len(frame)),
        "positive": int(labels.sum()),
        "within_peptide_evaluable": int(len(within)),
        "within_peptide_ap": float(np.mean([row[0] for row in within])),
        "within_peptide_auroc": float(np.mean([row[1] for row in within])),
        "within_peptide_log_loss": float(np.mean([row[2] for row in within])),
        "pooled_ap": float(average_precision_score(labels, scores)),
        "pooled_auroc": float(roc_auc_score(labels, scores)),
        "pooled_log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }


def _load_reference(path: Path, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    with path.open("rb") as handle:
        # pandas handles compression from the path; opening here only asserts readability.
        _require(bool(handle.read(1)), "reference OOF file is empty")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    forbidden = _forbidden_field_names(header)
    _require(not forbidden, f"reference OOF contains outcome fields: {forbidden}")
    required = {"row_id", "fold", "weak_label", "frozen_mint_layer9"}
    _require(required.issubset(header), "reference OOF lacks required columns")
    frame = pd.read_csv(path, usecols=sorted(required), dtype={"row_id": str})
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["weak_label"] = pd.to_numeric(
        frame["weak_label"], errors="raise"
    ).astype(int)
    frame["frozen_mint_layer9"] = pd.to_numeric(
        frame["frozen_mint_layer9"], errors="raise"
    ).astype(float)
    _require(len(frame) == EXPECTED_OOF_ROWS, "reference OOF row count changed")
    _require(frame["row_id"].is_unique, "reference OOF row IDs are duplicated")
    _require(
        _membership_sha256(frame["row_id"])
        == EXPECTED_OOF_MEMBERSHIP_SHA256,
        "reference OOF membership changed",
    )
    joined = frame.merge(
        labels[["row_id", "peptide_id", "weak_label"]],
        on="row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_labels"),
    )
    _require(not bool(joined["peptide_id"].isna().any()), "reference row absent from labels")
    _require(
        bool(joined["weak_label"].eq(joined["weak_label_labels"]).all()),
        "reference OOF weak labels differ from canonical labels",
    )
    joined = joined.drop(columns="weak_label_labels")
    metrics = _metric_bundle(joined, "frozen_mint_layer9")
    _require(
        math.isclose(
            float(metrics["within_peptide_ap"]),
            MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "frozen MINT layer-9 reference AP changed",
    )
    _require(
        int(metrics["within_peptide_evaluable"]) == EXPECTED_WITHIN_PEPTIDE_GROUPS,
        "reference evaluable peptide count changed",
    )
    return joined.sort_values("row_id").reset_index(drop=True), metrics


def _verify_cache_payloads(
    feature_cache: Path, merge_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Stream every merged cache payload and verify the receipt at aggregation time."""

    records: dict[str, Any] = {}
    declared: list[tuple[str, Mapping[str, Any]]] = []
    metadata = merge_receipt.get("metadata")
    _require(isinstance(metadata, Mapping), "cache metadata receipt is absent")
    declared.append(("metadata", metadata))
    arrays = merge_receipt.get("arrays")
    _require(isinstance(arrays, Mapping) and bool(arrays), "cache array receipts are absent")
    _require(
        set(map(str, arrays))
        == {"distogram_probabilities", "pair_states_symmetric", "single_inputs"},
        "merged cache payload set changed",
    )
    for name, record in sorted(arrays.items()):
        _require(isinstance(record, Mapping), f"cache array receipt is invalid: {name}")
        declared.append((str(name), record))
    for name, record in declared:
        filename = str(record.get("file", ""))
        _require(filename and Path(filename).name == filename, f"unsafe cache filename: {filename}")
        path = (feature_cache / filename).resolve()
        _require(path.parent == feature_cache, f"cache payload escapes cache root: {filename}")
        _require(path.is_file(), f"cache payload is absent: {path}")
        expected_bytes = int(record.get("bytes", -1))
        observed_bytes = int(path.stat().st_size)
        _require(observed_bytes == expected_bytes, f"cache payload size changed: {filename}")
        observed_sha256 = _sha256_file(path)
        _require(
            observed_sha256 == record.get("sha256"),
            f"cache payload checksum changed: {filename}",
        )
        records[name] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha256,
        }
    return {
        "stage": "aggregation_time_not_training_time",
        "all_declared_payloads_streamed": True,
        "payloads": records,
        "total_bytes": int(sum(record["bytes"] for record in records.values())),
    }


def _load_input_contract(
    config_path: Path,
    labels_path: Path,
    labels_manifest_path: Path,
    feature_cache: Path,
    reference_path: Path,
) -> InputContract:
    config_path = config_path.resolve()
    labels_path = labels_path.resolve()
    labels_manifest_path = labels_manifest_path.resolve()
    feature_cache = feature_cache.resolve()
    reference_path = reference_path.resolve()
    config = training._read_config(config_path)
    _require(config["dataset"]["library"] == "LibA", "config is not LibA")
    _require(tuple(config["models"]) == tuple(MODEL_NAMES), "five-family order changed")
    _require(
        tuple(map(int, config["final_training"]["seeds"]))
        == FIXED_REPLICATE_SEEDS,
        "fixed replicate seeds changed",
    )
    _require(
        config["final_training"].get("retention_labels_allowed") is False,
        "config allows retention labels",
    )
    with labels_path.open("r", encoding="utf-8") as handle:
        label_header = handle.readline().rstrip("\n").split(",")
    forbidden = _forbidden_field_names(label_header)
    _require(not forbidden, f"training sidecar contains outcome fields: {forbidden}")
    labels = pd.read_csv(
        labels_path,
        usecols=["row_id", "weak_label", "peptide_id"],
        dtype={"row_id": str, "peptide_id": str},
    )
    labels["weak_label"] = pd.to_numeric(labels["weak_label"], errors="raise").astype(int)
    _require(labels["row_id"].is_unique, "canonical training row IDs are duplicated")
    _require(set(labels["weak_label"]) == {0, 1}, "canonical weak labels are not binary")
    labels_manifest = _read_json(labels_manifest_path)
    _assert_no_outcome_keys(labels_manifest, str(labels_manifest_path))
    merge_manifest = feature_cache / "merge_complete.json"
    _require(merge_manifest.is_file(), "feature-cache merge receipt is absent")
    cache_receipt = _read_json(merge_manifest)
    _assert_no_outcome_keys(cache_receipt, str(merge_manifest))
    cache_payload_verification = _verify_cache_payloads(feature_cache, cache_receipt)
    reference, reference_metrics = _load_reference(reference_path, labels)
    return InputContract(
        config_path=config_path,
        config=config,
        config_sha256=_sha256_file(config_path),
        labels_path=labels_path,
        labels_sha256=_sha256_file(labels_path),
        labels_manifest_path=labels_manifest_path,
        labels_manifest_sha256=_sha256_file(labels_manifest_path),
        feature_cache=feature_cache,
        cache_manifest_path=merge_manifest,
        cache_manifest_sha256=_sha256_file(merge_manifest),
        cache_payload_verification=cache_payload_verification,
        reference_path=reference_path,
        reference_sha256=_sha256_file(reference_path),
        reference=reference,
        reference_metrics=reference_metrics,
    )


def _validate_completion_hashes(run_dir: Path, completion: Mapping[str, Any]) -> dict[str, Any]:
    mapping = {
        "execution_contract": ("execution_contract.json", "execution_contract_sha256"),
        "selected_epochs": ("selected_epochs.json", "selected_epochs_sha256"),
        "weak_validation_summary": (
            "weak_validation_summary.json",
            "weak_validation_summary_sha256",
        ),
        "weak_validation_predictions": (
            "weak_validation_predictions.csv.gz",
            "weak_validation_predictions_sha256",
        ),
    }
    observed: dict[str, Any] = {}
    for key, (filename, receipt_key) in mapping.items():
        path = run_dir / filename
        _require(path.is_file(), f"run artifact is absent: {path}")
        sha256 = _sha256_file(path)
        _require(completion.get(receipt_key) == sha256, f"{filename} checksum mismatch")
        observed[key] = {"path": str(path), "sha256": sha256}
    checkpoint_receipts = completion.get("checkpoint_sha256")
    _require(isinstance(checkpoint_receipts, Mapping), "checkpoint receipt is absent")
    _require(len(checkpoint_receipts) == 3, "expected three checkpoint receipts")
    contract = _read_json(run_dir / "execution_contract.json")
    family = str(contract.get("model"))
    _require(
        set(map(str, checkpoint_receipts))
        == {f"{family}__fold{fold}.pt" for fold in range(3)},
        "checkpoint receipt filenames changed",
    )
    observed_checkpoints = {}
    for filename, expected_sha256 in sorted(checkpoint_receipts.items()):
        _require(Path(str(filename)).name == str(filename), "unsafe checkpoint filename")
        path = run_dir / "cv_checkpoints" / str(filename)
        _require(path.is_file(), f"checkpoint is absent: {path}")
        sha256 = _sha256_file(path)
        _require(sha256 == expected_sha256, f"checkpoint checksum mismatch: {filename}")
        observed_checkpoints[str(filename)] = {"path": str(path), "sha256": sha256}
    _require(
        {path.name for path in (run_dir / "cv_checkpoints").glob("*.pt")}
        == set(map(str, checkpoint_receipts)),
        "checkpoint directory and receipt membership differ",
    )
    observed["checkpoints"] = observed_checkpoints
    return observed


def _validate_summary_against_predictions(
    summary: Mapping[str, Any],
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Verify every reported fold metric against the hash-bound OOF rows."""

    _require(
        summary.get("selection_rule")
        == "integer median of three weak-validation best epochs",
        "summary selection rule changed",
    )
    metric_names = ("log_loss", "brier", "auroc", "average_precision")
    observed_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for record in summary["folds"]:
        fold = int(record["fold"])
        block = predictions.loc[predictions["fold"].eq(fold)]
        _require(
            int(record.get("n_train", -1))
            == int(config["validation"]["fold_contracts"][str(fold)]["train"]),
            f"fold {fold} summary training count changed",
        )
        recomputed = training._binary_metrics(
            block["weak_label"].to_numpy(dtype=int),
            block["probability"].to_numpy(dtype=float),
        )
        reported = record["best_metrics"]
        _require(int(reported.get("rows", -1)) == len(block), f"fold {fold} metric rows changed")
        _require(
            int(reported.get("positive", -1)) == int(block["weak_label"].sum()),
            f"fold {fold} metric positives changed",
        )
        _require(
            math.isclose(
                float(reported.get("prevalence", math.nan)),
                float(block["weak_label"].mean()),
                rel_tol=0.0,
                abs_tol=METRIC_RECEIPT_TOLERANCE,
            ),
            f"fold {fold} metric prevalence changed",
        )
        for name in metric_names:
            value = float(reported.get(name, math.nan))
            _require(math.isfinite(value), f"fold {fold} {name} is non-finite")
            _require(
                math.isclose(
                    value,
                    float(recomputed[name]),
                    rel_tol=METRIC_RECEIPT_TOLERANCE,
                    abs_tol=METRIC_RECEIPT_TOLERANCE,
                ),
                f"fold {fold} reported {name} differs from predictions",
            )
            observed_by_metric[name].append(value)
    reported_means = summary.get("mean_best_metrics")
    _require(isinstance(reported_means, Mapping), "mean fold metrics are absent")
    for name in metric_names:
        value = float(reported_means.get(name, math.nan))
        _require(math.isfinite(value), f"mean {name} is non-finite")
        _require(
            math.isclose(
                value,
                float(np.mean(observed_by_metric[name])),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"mean {name} differs from the three fold records",
        )


def _validate_run(
    run_dir: Path,
    inputs: InputContract,
    current_source_hashes: Mapping[str, str],
) -> ValidatedRun:
    run_dir = run_dir.resolve()
    contract_path = run_dir / "execution_contract.json"
    completion_path = run_dir / "cv_completion.json"
    _require(contract_path.is_file(), f"execution contract is absent: {run_dir}")
    _require(completion_path.is_file(), f"completion receipt is absent: {run_dir}")
    contract = _read_json(contract_path)
    completion = _read_json(completion_path)
    _assert_no_outcome_keys(contract, str(contract_path))
    _assert_no_outcome_keys(completion, str(completion_path))
    _require(contract.get("schema_version") == runner.SCHEMA_VERSION, "run schema changed")
    _require(
        completion.get("schema_version") == runner.COMPLETION_SCHEMA_VERSION,
        "completion schema changed",
    )
    family = str(contract.get("model"))
    _require(family in MODEL_NAMES, f"unexpected family: {family}")
    family_index = tuple(MODEL_NAMES).index(family)
    seed = int(contract.get("replicate_seed", -1))
    _require(seed in FIXED_REPLICATE_SEEDS, f"unexpected replicate seed: {seed}")
    _require(contract.get("library") == "LibA", "run is not LibA")
    _require(int(contract.get("original_model_index", -1)) == family_index, "model index changed")
    _require(
        tuple(map(int, contract.get("fixed_replicate_seeds", ())))
        == FIXED_REPLICATE_SEEDS,
        "run fixed-seed contract changed",
    )
    effective = seed + 1000 * family_index
    _require(
        int(contract.get("effective_cv_training_seed", -1)) == effective,
        "effective CV seed does not equal replicate seed plus model offset",
    )
    _require(
        tuple(map(int, contract.get("effective_cv_fold_seeds", ())))
        == tuple(effective + fold for fold in range(3)),
        "fold training seeds changed",
    )
    _require(int(contract.get("split_seed", -1)) == 17, "split seed changed")
    _require(int(contract.get("folds", -1)) == 3, "fold count changed")
    _require(
        int(contract.get("weak_rows", -1)) == int(inputs.config["dataset"]["rows"]),
        "weak row count changed",
    )
    _require(
        int(contract.get("weak_positive", -1))
        == int(inputs.config["dataset"]["positive"]),
        "weak positive count changed",
    )
    _require(
        int(contract.get("weak_negative", -1))
        == int(inputs.config["dataset"]["negative"]),
        "weak negative count changed",
    )
    _require(
        int(contract.get("trainable_parameters", -1))
        == int(inputs.config["expected_trainable_parameters"][family]),
        "trainable parameter count changed",
    )
    _require(contract.get("master_config_sha256") == inputs.config_sha256, "config hash changed")
    _require(contract.get("training_labels_sha256") == inputs.labels_sha256, "label hash changed")
    _require(
        contract.get("training_labels_manifest_sha256")
        == inputs.labels_manifest_sha256,
        "label-manifest hash changed",
    )
    _require(
        Path(str(contract.get("feature_cache"))).resolve() == inputs.feature_cache,
        "feature-cache path changed",
    )
    _require(
        contract.get("feature_cache_merge_manifest_sha256")
        == inputs.cache_manifest_sha256,
        "feature-cache receipt hash changed",
    )
    for key, expected in current_source_hashes.items():
        _require(contract.get(key) == expected, f"run source hash changed: {key}")
    _require(contract.get("selection_labels_read") is True, "weak labels were not declared")
    _require(contract.get("retention_labels_read") is False, "run read retention labels")
    _require(completion.get("retention_labels_read") is False, "completion read retention labels")
    runtime_environment = contract.get("runtime_environment")
    _require(isinstance(runtime_environment, Mapping), "runtime environment is absent")
    for key, expected_value in runner.THREAD_ENV_CONTRACT.items():
        _require(
            runtime_environment.get(key) == expected_value,
            f"runtime thread setting changed: {key}",
        )
    _require(
        int(runtime_environment.get("torch_intraop_threads", -1)) == 4,
        "PyTorch intra-op thread count changed",
    )
    _require(
        int(runtime_environment.get("torch_interop_threads", -1)) >= 1,
        "PyTorch inter-op thread count is invalid",
    )
    _require(
        runtime_environment.get("thread_values_inherited_from_environment") is True,
        "thread settings were not inherited from the recorded environment",
    )
    _require(
        contract.get("feature_transfer")
        == {
            "host_copy_dtype": "source float16",
            "model_input_dtype": "float32",
            "float32_conversion_stage": "destination device in _model_inputs",
        },
        "feature-transfer contract changed",
    )
    _require(completion.get("model") == family, "completion family differs")
    _require(int(completion.get("replicate_seed", -1)) == seed, "completion seed differs")
    _require(
        math.isfinite(float(completion.get("runtime_seconds", math.nan)))
        and float(completion["runtime_seconds"]) >= 0.0,
        "completion runtime is invalid",
    )
    if family == PRETRUNK_CONTROL:
        _require(
            contract.get("single_inputs_only_semantics")
            == "pre-trunk sequence-derived control, not structural evidence",
            "single_inputs_only control semantics changed",
        )
    else:
        _require(
            contract.get("single_inputs_only_semantics") is None,
            "non-control run has unexpected single-input semantics",
        )

    for copied_name, expected_sha256 in (
        ("master_config.json", inputs.config_sha256),
        ("training_labels.csv", inputs.labels_sha256),
        ("training_labels_manifest.json", inputs.labels_manifest_sha256),
    ):
        copied_path = run_dir / copied_name
        _require(copied_path.is_file(), f"copied source is absent: {copied_path}")
        _require(_sha256_file(copied_path) == expected_sha256, f"copied {copied_name} changed")

    source_hashes = _validate_completion_hashes(run_dir, completion)
    summary_payload = _read_json(run_dir / "weak_validation_summary.json")
    selected_payload = _read_json(run_dir / "selected_epochs.json")
    _assert_no_outcome_keys(summary_payload, str(run_dir / "weak_validation_summary.json"))
    _assert_no_outcome_keys(selected_payload, str(run_dir / "selected_epochs.json"))
    _require(isinstance(summary_payload, list) and len(summary_payload) == 1, "summary shape changed")
    summary = summary_payload[0]
    _require(summary.get("model") == family, "summary family changed")
    _require(set(selected_payload) == {family}, "selected-epoch keys changed")
    selected_epoch = int(selected_payload[family])
    folds = summary.get("folds")
    _require(isinstance(folds, list) and len(folds) == 3, "summary must have three folds")
    _require({int(record.get("fold", -1)) for record in folds} == {0, 1, 2}, "summary folds changed")
    for record in folds:
        fold = int(record["fold"])
        _require(record.get("model") == family, "fold summary family changed")
        _require(int(record.get("seed", -1)) == effective + fold, "fold summary seed changed")
        best_epoch = int(record.get("best_epoch", -1))
        _require(1 <= best_epoch <= int(inputs.config["optimization"]["max_epochs"]), "invalid best epoch")
        _require(isinstance(record.get("best_metrics"), Mapping), "fold metrics absent")
    expected_run_epoch = int(np.median([int(record["best_epoch"]) for record in folds]))
    _require(selected_epoch == expected_run_epoch, "run selected epoch is not median of its folds")
    _require(
        int(summary.get("selected_final_epochs", -1)) == selected_epoch,
        "summary and selected-epoch file differ",
    )

    predictions_path = run_dir / "weak_validation_predictions.csv.gz"
    prediction_header = pd.read_csv(predictions_path, nrows=0).columns.tolist()
    expected_columns = {"model", "fold", "row_id", "weak_label", "probability"}
    _require(set(prediction_header) == expected_columns, "prediction columns changed")
    forbidden = _forbidden_field_names(prediction_header)
    _require(not forbidden, f"run predictions contain outcome fields: {forbidden}")
    predictions = pd.read_csv(predictions_path, dtype={"row_id": str, "model": str})
    predictions["fold"] = pd.to_numeric(predictions["fold"], errors="raise").astype(int)
    predictions["weak_label"] = pd.to_numeric(
        predictions["weak_label"], errors="raise"
    ).astype(int)
    predictions["probability"] = pd.to_numeric(
        predictions["probability"], errors="raise"
    ).astype(float)
    _require(len(predictions) == EXPECTED_OOF_ROWS, "run OOF row count changed")
    _require(predictions["row_id"].is_unique, "run OOF row IDs are duplicated")
    _require(set(predictions["model"]) == {family}, "prediction family changed")
    _require(set(predictions["fold"]) == {0, 1, 2}, "prediction fold set changed")
    _require(set(predictions["weak_label"]) == {0, 1}, "prediction labels changed")
    _require(
        bool(np.isfinite(predictions["probability"]).all()),
        "run has non-finite predictions",
    )
    _require(
        bool(((predictions["probability"] >= 0) & (predictions["probability"] <= 1)).all()),
        "run probabilities must be between zero and one",
    )

    aligned = predictions.merge(
        inputs.reference[["row_id", "fold", "weak_label", "peptide_id"]],
        on="row_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_reference"),
    )
    _require(len(aligned) == EXPECTED_OOF_ROWS, "run membership differs from reference")
    _require(bool(aligned["_merge"].eq("both").all()), "run membership differs from reference")
    _require(bool(aligned["fold"].eq(aligned["fold_reference"]).all()), "run fold assignments differ")
    _require(
        bool(aligned["weak_label"].eq(aligned["weak_label_reference"]).all()),
        "run weak labels differ from reference",
    )
    _require(
        _membership_sha256(aligned["row_id"]) == EXPECTED_OOF_MEMBERSHIP_SHA256,
        "run OOF membership hash changed",
    )
    reference_row_fold = _row_fold_sha256(inputs.reference)
    _require(_row_fold_sha256(predictions) == reference_row_fold, "run row/fold hash changed")
    _validate_summary_against_predictions(summary, predictions, inputs.config)

    receipt_folds = completion.get("fold_prediction_memberships")
    _require(isinstance(receipt_folds, list) and len(receipt_folds) == 3, "fold receipts changed")
    by_fold_receipt = {int(record["fold"]): record for record in receipt_folds}
    _require(set(by_fold_receipt) == {0, 1, 2}, "fold receipt IDs changed")
    for fold in range(3):
        block = predictions.loc[predictions["fold"].eq(fold)]
        receipt = by_fold_receipt[fold]
        _require(len(block) == EXPECTED_FOLD_ROWS[fold], f"fold {fold} row count changed")
        _require(int(block["weak_label"].sum()) == EXPECTED_FOLD_POSITIVE[fold], f"fold {fold} positives changed")
        _require(int(receipt.get("rows", -1)) == len(block), f"fold {fold} receipt rows changed")
        _require(
            int(receipt.get("positive", -1)) == int(block["weak_label"].sum()),
            f"fold {fold} receipt positives changed",
        )
        _require(
            receipt.get("row_id_membership_sha256")
            == _membership_sha256(block["row_id"]),
            f"fold {fold} receipt membership hash changed",
        )
        _require(
            _membership_sha256(block["row_id"])
            == EXPECTED_FOLD_MEMBERSHIP_SHA256[fold],
            f"fold {fold} canonical membership hash changed",
        )
    retained = aligned[
        ["row_id", "fold_reference", "weak_label_reference", "peptide_id", "probability"]
    ].rename(
        columns={"fold_reference": "fold", "weak_label_reference": "weak_label"}
    )
    return ValidatedRun(
        family=family,
        replicate_seed=seed,
        run_dir=run_dir,
        contract=contract,
        summary=summary,
        selected_epoch_in_run=selected_epoch,
        predictions=retained.sort_values("row_id").reset_index(drop=True),
        source_hashes={
            "run_dir": str(run_dir),
            "execution_contract_sha256": _sha256_file(contract_path),
            "cv_completion_sha256": _sha256_file(completion_path),
            **source_hashes,
        },
    )


def _mean_logit_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    _require(values.ndim == 2, "replicate probability matrix must be two-dimensional")
    _require(bool(np.isfinite(values).all()), "replicate probabilities are non-finite")
    _require(
        bool(((values > 0.0) & (values < 1.0)).all()),
        "stored probabilities include zero or one; exact logits cannot be reconstructed",
    )
    logits = np.log(values) - np.log1p(-values)
    mean_logits = logits.mean(axis=1)
    probabilities_out = 1.0 / (1.0 + np.exp(-mean_logits))
    return mean_logits, probabilities_out


def _best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    probability = np.asarray(scores, dtype=float)
    _require(set(np.unique(y)) == {0, 1}, "threshold labels are not binary")
    _require(bool(np.isfinite(probability).all()), "threshold scores are non-finite")
    candidates = np.unique(probability)
    best: list[dict[str, Any]] = []
    best_f1 = -1.0
    for threshold in candidates:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & y.astype(bool)))
        fp = int(np.sum(predicted & ~y.astype(bool)))
        fn = int(np.sum(~predicted & y.astype(bool)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        record = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "predicted_positive": int(predicted.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        }
        if f1 > best_f1 and not math.isclose(f1, best_f1, rel_tol=0.0, abs_tol=1e-15):
            best_f1 = f1
            best = [record]
        elif math.isclose(f1, best_f1, rel_tol=0.0, abs_tol=1e-15):
            best.append(record)
    _require(bool(best), "no threshold candidate was evaluated")
    # Locked deterministic tie: keep the highest equally optimal threshold.
    return max(best, key=lambda record: float(record["threshold"]))


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    _require(array.size == len(FIXED_REPLICATE_SEEDS), "expected five replicate values")
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "individual_values": [float(value) for value in array],
    }


def _best_family_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Apply the prespecified AP, log-loss, then config-order tie rule."""

    _require(bool(records), "no family records to rank")
    best_ap = max(float(record["aggregate_within_peptide_ap"]) for record in records)
    ap_tied = [
        record
        for record in records
        if best_ap - float(record["aggregate_within_peptide_ap"]) <= AP_BEAT_TOLERANCE
    ]
    best_loss = min(
        float(record["aggregate_within_peptide_log_loss"]) for record in ap_tied
    )
    loss_tied = [
        record
        for record in ap_tied
        if abs(float(record["aggregate_within_peptide_log_loss"]) - best_loss)
        <= AP_BEAT_TOLERANCE
    ]
    return min(
        loss_tied,
        key=lambda record: tuple(MODEL_NAMES).index(str(record["family"])),
    )


def _aggregate_runs(
    runs: Sequence[ValidatedRun],
    inputs: InputContract,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    expected_pairs = {
        (family, seed) for family in MODEL_NAMES for seed in FIXED_REPLICATE_SEEDS
    }
    observed_pairs = [(run.family, run.replicate_seed) for run in runs]
    _require(len(observed_pairs) == len(set(observed_pairs)), "duplicate family/seed run")
    _require(set(observed_pairs) == expected_pairs, "run set is not the locked 5 x 5 Cartesian product")

    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    aggregate_metric_rows: list[dict[str, Any]] = []
    aggregate_prediction_blocks: list[pd.DataFrame] = []
    threshold_records: dict[str, Any] = {}
    run_lookup = {(run.family, run.replicate_seed): run for run in runs}

    for family in MODEL_NAMES:
        family_runs = [run_lookup[(family, seed)] for seed in FIXED_REPLICATE_SEEDS]
        for run in family_runs:
            metrics = _metric_bundle(run.predictions, "probability")
            metric_rows.append(
                {
                    "family": family,
                    "family_role": FAMILY_ROLE[family],
                    "replicate_seed": run.replicate_seed,
                    "run_median_selected_epoch": run.selected_epoch_in_run,
                    **metrics,
                    "delta_within_peptide_ap_vs_mint_layer9": (
                        float(metrics["within_peptide_ap"])
                        - MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP
                    ),
                    "beats_mint_layer9_within_peptide_ap": int(
                        float(metrics["within_peptide_ap"])
                        > MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP
                        + AP_BEAT_TOLERANCE
                    ),
                }
            )
            for record in sorted(run.summary["folds"], key=lambda value: int(value["fold"])):
                best = record["best_metrics"]
                fold_rows.append(
                    {
                        "family": family,
                        "replicate_seed": run.replicate_seed,
                        "fold": int(record["fold"]),
                        "fold_training_seed": int(record["seed"]),
                        "best_epoch": int(record["best_epoch"]),
                        "run_median_selected_epoch": run.selected_epoch_in_run,
                        "best_log_loss": float(best["log_loss"]),
                        "best_auroc": float(best["auroc"]),
                        "best_average_precision": float(best["average_precision"]),
                    }
                )

        base = family_runs[0].predictions[
            ["row_id", "fold", "weak_label", "peptide_id"]
        ].sort_values("row_id").reset_index(drop=True)
        aligned_probabilities: list[np.ndarray] = []
        for run in family_runs:
            candidate = run.predictions.sort_values("row_id").reset_index(drop=True)
            _require(
                candidate[["row_id", "fold", "weak_label", "peptide_id"]].equals(base),
                f"{family} replicate rows are not exactly aligned",
            )
            aligned_probabilities.append(candidate["probability"].to_numpy(dtype=float))
        probability_matrix = np.column_stack(
            aligned_probabilities
        )
        probability_diagnostics = {
            "minimum_stored_probability": float(probability_matrix.min()),
            "maximum_stored_probability": float(probability_matrix.max()),
            "stored_probability_equal_zero": int(np.sum(probability_matrix == 0.0)),
            "stored_probability_equal_one": int(np.sum(probability_matrix == 1.0)),
            "stored_values_clipped_before_inverse_logit": 0,
        }
        mean_logits, mean_logit_probability = _mean_logit_probabilities(probability_matrix)
        base["mean_logit"] = mean_logits
        base["mean_logit_probability"] = mean_logit_probability
        aggregate_metrics = _metric_bundle(base, "mean_logit_probability")
        aggregate_metric_rows.append(
            {
                "family": family,
                "family_role": FAMILY_ROLE[family],
                **aggregate_metrics,
                **probability_diagnostics,
                "delta_within_peptide_ap_vs_mint_layer9": (
                    float(aggregate_metrics["within_peptide_ap"])
                    - MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP
                ),
            }
        )
        output_block = base.drop(columns="peptide_id").copy()
        output_block.insert(0, "family", family)
        aggregate_prediction_blocks.append(output_block)
        threshold_records[family] = {
            **_best_f1_threshold(base["weak_label"], base["mean_logit_probability"]),
            "score": "sigmoid(mean of five logits reconstructed from stored probabilities)",
            "label_source": "selection-derived weak labels only",
            "prediction_rule": "score >= threshold",
            "optimization": "maximum weak-OOF F1",
            "tie_break": "highest threshold among equal maximum-F1 thresholds",
        }

    metrics_by_seed = pd.DataFrame(metric_rows).sort_values(
        ["family", "replicate_seed"]
    ).reset_index(drop=True)
    fold_epochs = pd.DataFrame(fold_rows).sort_values(
        ["family", "replicate_seed", "fold"]
    ).reset_index(drop=True)
    aggregate_metrics_frame = pd.DataFrame(aggregate_metric_rows).sort_values(
        "family"
    ).reset_index(drop=True)
    aggregate_predictions = pd.concat(aggregate_prediction_blocks, ignore_index=True)

    single_aggregate = float(
        aggregate_metrics_frame.loc[
            aggregate_metrics_frame["family"].eq(PRETRUNK_CONTROL),
            "within_peptide_ap",
        ].iloc[0]
    )
    winner_counts = {family: 0 for family in MODEL_NAMES}
    for seed in FIXED_REPLICATE_SEEDS:
        contenders = metrics_by_seed.loc[metrics_by_seed["replicate_seed"].eq(seed)]
        comparable = [
            {
                "family": row["family"],
                "aggregate_within_peptide_ap": row["within_peptide_ap"],
                "aggregate_within_peptide_log_loss": row["within_peptide_log_loss"],
            }
            for row in contenders.to_dict("records")
        ]
        winner = _best_family_record(comparable)["family"]
        winner_counts[str(winner)] += 1

    family_rows: list[dict[str, Any]] = []
    model_records: dict[str, Any] = {}
    for family in MODEL_NAMES:
        per_seed = metrics_by_seed.loc[
            metrics_by_seed["family"].eq(family)
        ].sort_values("replicate_seed")
        aggregate = aggregate_metrics_frame.loc[
            aggregate_metrics_frame["family"].eq(family)
        ].iloc[0].to_dict()
        seed_ap = per_seed["within_peptide_ap"].astype(float).tolist()
        beats_reference = int(
            np.sum(
                np.asarray(seed_ap)
                > MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP + AP_BEAT_TOLERANCE
            )
        )
        if family == PRETRUNK_CONTROL:
            beats_pretrunk = None
        else:
            pretrunk_by_seed = metrics_by_seed.loc[
                metrics_by_seed["family"].eq(PRETRUNK_CONTROL)
            ].set_index("replicate_seed")["within_peptide_ap"]
            beats_pretrunk = int(
                sum(
                    float(row.within_peptide_ap)
                    > float(pretrunk_by_seed.loc[int(row.replicate_seed)])
                    + AP_BEAT_TOLERANCE
                    for row in per_seed.itertuples()
                )
            )
        reproducible_vs_reference = bool(
            float(aggregate["within_peptide_ap"])
            > MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP + AP_BEAT_TOLERANCE
            and beats_reference >= 4
        )
        passes_pretrunk = (
            None
            if family == PRETRUNK_CONTROL
            else bool(
                float(aggregate["within_peptide_ap"])
                > single_aggregate + AP_BEAT_TOLERANCE
                and int(beats_pretrunk) >= 4
            )
        )
        supports_structure = bool(
            family in STRUCTURE_DERIVED_FAMILIES
            and reproducible_vs_reference
            and passes_pretrunk
        )
        family_fold_epochs = fold_epochs.loc[
            fold_epochs["family"].eq(family), "best_epoch"
        ].astype(int).tolist()
        _require(len(family_fold_epochs) == 15, f"{family} lacks 15 fold epochs")
        common_epoch = int(np.median(family_fold_epochs))
        metric_summaries = {
            metric: _sample_summary(per_seed[metric].astype(float).tolist())
            for metric in (
                "within_peptide_ap",
                "within_peptide_auroc",
                "within_peptide_log_loss",
                "pooled_ap",
                "pooled_auroc",
                "pooled_log_loss",
            )
        }
        family_row: dict[str, Any] = {
            "family": family,
            "family_role": FAMILY_ROLE[family],
            "replicates": len(per_seed),
            "aggregate_within_peptide_ap": aggregate["within_peptide_ap"],
            "aggregate_within_peptide_auroc": aggregate["within_peptide_auroc"],
            "aggregate_within_peptide_log_loss": aggregate["within_peptide_log_loss"],
            "aggregate_pooled_ap": aggregate["pooled_ap"],
            "aggregate_pooled_auroc": aggregate["pooled_auroc"],
            "aggregate_pooled_log_loss": aggregate["pooled_log_loss"],
            "seeds_beating_mint_layer9": beats_reference,
            "paired_seeds_beating_single_inputs": beats_pretrunk,
            "replicate_winner_count": winner_counts[family],
            "selected_epoch_median_of_15": common_epoch,
            "reproducibly_beats_mint_layer9": int(reproducible_vs_reference),
            "passes_pretrunk_control": (
                "not_applicable" if passes_pretrunk is None else int(passes_pretrunk)
            ),
            "supports_structure_claim": int(supports_structure),
        }
        for metric, summary_record in metric_summaries.items():
            family_row[f"{metric}_mean"] = summary_record["mean"]
            family_row[f"{metric}_sample_sd"] = summary_record["sample_sd"]
            family_row[f"individual_{metric}"] = json.dumps(
                summary_record["individual_values"], separators=(",", ":")
            )
        family_rows.append(family_row)
        model_records[family] = {
            "family_role": FAMILY_ROLE[family],
            "aggregate_mean_logit_metrics": {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in aggregate.items()
                if key not in {"family", "family_role"}
            },
            "per_seed_metrics": per_seed.to_dict("records"),
            "replicate_metric_summaries": metric_summaries,
            "fold_best_epochs": family_fold_epochs,
            "selected_final_epoch": common_epoch,
            "seeds_beating_mint_layer9": beats_reference,
            "paired_seeds_beating_single_inputs": beats_pretrunk,
            "replicate_winner_count": winner_counts[family],
            "reproducibly_beats_mint_layer9": reproducible_vs_reference,
            "passes_pretrunk_control": passes_pretrunk,
            "supports_structure_claim": supports_structure,
            "weak_oof_threshold": threshold_records[family],
        }

    family_summary = pd.DataFrame(family_rows).sort_values(
        "family"
    ).reset_index(drop=True)
    selected_record = _best_family_record(family_rows)
    selected_family = str(selected_record["family"])
    structure_records = [
        row for row in family_rows if row["family"] in STRUCTURE_DERIVED_FAMILIES
    ]
    best_structure = str(_best_family_record(structure_records)["family"])
    passing_structure = sorted(
        row["family"] for row in structure_records if int(row["supports_structure_claim"]) == 1
    )
    selection = {
        "schema_version": SCHEMA_VERSION,
        "selection_data": "selection-derived weak labels only",
        "retention_labels_read": False,
        "replicate_seeds": list(FIXED_REPLICATE_SEEDS),
        "family_order": list(MODEL_NAMES),
        "family_roles": FAMILY_ROLE,
        "aggregation": {
            "primary_score": (
                "sigmoid(mean of five logits reconstructed exactly from stored "
                "strictly interior probabilities per row)"
            ),
            "probability_clipping": "none; fail if any stored probability equals 0 or 1",
        },
        "family_selection_rule": {
            "primary": "highest aggregate mean-logit macro within-peptide weak-label AP",
            "first_tie_break": "lowest aggregate macro within-peptide weak-label log loss",
            "second_tie_break": "earliest family in locked config order",
        },
        "selected_feature_family": selected_family,
        "best_structure_derived_family_by_gate_metric": best_structure,
        "best_structure_family_is_deployable_only_if_gate_passes": True,
        "model_selected_epochs": {
            family: int(model_records[family]["selected_final_epoch"])
            for family in MODEL_NAMES
        },
        "aggregate_epoch_rule": (
            "integer median of all 15 fold-best epochs per family (five replicate "
            "labels times three folds); this is not the median of five run medians"
        ),
        "legacy_final_fit_wrapper_compatible": False,
        "legacy_final_fit_wrapper_note": (
            "run_esmfold2_readout_one_model.py validates a single-run epoch receipt; "
            "a replicated-CV-aware final-fit runner must consume this aggregate lock"
        ),
        "mint_layer9_reference": {
            **dict(inputs.reference_metrics),
            "fixed_within_peptide_ap": MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP,
        },
        "reproducibility_rule_vs_mint": (
            "prespecified heuristic: aggregate mean-logit within-peptide AP exceeds "
            "the fixed MINT layer-9 reference by more than 1e-12 and at least four "
            "of five individual replicate AP values exceed it by more than 1e-12"
        ),
        "structure_evidence_rule": (
            "a folding-derived family must satisfy the MINT rule, exceed the "
            "single_inputs_only aggregate AP by more than 1e-12, and beat the "
            "schedule-matched single_inputs_only replicate in at least four of five "
            "replicate labels; this heuristic is not a statistical significance test"
        ),
        "passing_structure_families": passing_structure,
        "structure_claim_supported": bool(passing_structure),
        "replicate_family_winner_counts": winner_counts,
        "models": model_records,
    }
    thresholds = {
        "schema_version": "esmfold2-liba-replicated-cv-weak-oof-thresholds-v1",
        "selection_data": "selection-derived weak labels only",
        "retention_labels_read": False,
        "score_definition": (
            "sigmoid(mean of five logits reconstructed exactly from stored strictly "
            "interior probabilities per row)"
        ),
        "prediction_rule": "score >= threshold",
        "optimization": "maximum weak-OOF F1",
        "tie_break": "highest threshold among equal maximum-F1 thresholds",
        "selected_feature_family": selected_family,
        "selected_feature_family_threshold": threshold_records[selected_family],
        "families": threshold_records,
    }
    return (
        metrics_by_seed,
        family_summary,
        fold_epochs,
        aggregate_metrics_frame,
        aggregate_predictions,
        selection,
        thresholds,
    )


def _discover_run_dirs(run_root: Path) -> list[Path]:
    run_root = run_root.resolve()
    _require(run_root.is_dir(), f"run root is absent: {run_root}")
    run_dirs = sorted({path.parent.resolve() for path in run_root.rglob("cv_completion.json")})
    _require(len(run_dirs) == 25, f"expected exactly 25 completed runs, found {len(run_dirs)}")
    return run_dirs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-labels", type=Path, default=DEFAULT_TRAINING_LABELS)
    parser.add_argument(
        "--training-labels-manifest",
        type=Path,
        default=DEFAULT_TRAINING_LABELS_MANIFEST,
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, default=DEFAULT_REFERENCE_OOF)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    inputs = _load_input_contract(
        args.config,
        args.training_labels,
        args.training_labels_manifest,
        args.feature_cache,
        args.reference_oof,
    )
    runner_path = Path(runner.__file__).resolve()
    trainer_path = Path(training.__file__).resolve()
    readout_path = Path(sys.modules[training.FrozenFeatureStore.__module__].__file__).resolve()
    current_source_hashes = {
        "runner_sha256": _sha256_file(runner_path),
        "trainer_sha256": _sha256_file(trainer_path),
        "readout_sha256": _sha256_file(readout_path),
    }
    run_dirs = _discover_run_dirs(args.run_root)
    runs = [_validate_run(path, inputs, current_source_hashes) for path in run_dirs]
    (
        metrics_by_seed,
        family_summary,
        fold_epochs,
        aggregate_metrics,
        aggregate_predictions,
        selection,
        thresholds,
    ) = _aggregate_runs(runs, inputs)

    output_dir = _validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True)
    output_paths = {
        "replicated_metrics_by_seed": output_dir / "replicated_metrics_by_seed.csv",
        "family_summary": output_dir / "family_summary.csv",
        "fold_best_epochs": output_dir / "fold_best_epochs.csv",
        "aggregate_metrics": output_dir / "aggregate_mean_logit_metrics.csv",
        "aggregate_predictions": output_dir / "aggregate_mean_logit_oof_predictions.csv.gz",
        "all_model_selection": output_dir / "all_model_selection.json",
        "weak_oof_thresholds": output_dir / "weak_oof_thresholds.json",
    }
    metrics_by_seed.to_csv(output_paths["replicated_metrics_by_seed"], index=False)
    family_summary.to_csv(output_paths["family_summary"], index=False)
    fold_epochs.to_csv(output_paths["fold_best_epochs"], index=False)
    aggregate_metrics.to_csv(output_paths["aggregate_metrics"], index=False)
    aggregate_predictions.to_csv(output_paths["aggregate_predictions"], index=False)
    _json_dump(output_paths["all_model_selection"], selection)
    _json_dump(output_paths["weak_oof_thresholds"], thresholds)
    for path in output_paths.values():
        os.chmod(path, 0o600)

    run_receipts = []
    for run in sorted(runs, key=lambda value: (tuple(MODEL_NAMES).index(value.family), value.replicate_seed)):
        run_receipts.append(
            {
                "family": run.family,
                "replicate_seed": run.replicate_seed,
                **dict(run.source_hashes),
            }
        )
    _json_dump(
        output_dir / "aggregation_completion.json",
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "inputs": {
                "config": {"path": str(inputs.config_path), "sha256": inputs.config_sha256},
                "training_labels": {"path": str(inputs.labels_path), "sha256": inputs.labels_sha256},
                "training_labels_manifest": {
                    "path": str(inputs.labels_manifest_path),
                    "sha256": inputs.labels_manifest_sha256,
                },
                "feature_cache_merge_manifest": {
                    "path": str(inputs.cache_manifest_path),
                    "sha256": inputs.cache_manifest_sha256,
                },
                "feature_cache_payload_verification": dict(
                    inputs.cache_payload_verification
                ),
                "reference_oof": {
                    "path": str(inputs.reference_path),
                    "sha256": inputs.reference_sha256,
                    "membership_sha256": EXPECTED_OOF_MEMBERSHIP_SHA256,
                    "row_fold_sha256": _row_fold_sha256(inputs.reference),
                },
                "run_receipts": run_receipts,
            },
            "outputs": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in output_paths.items()
            },
            "completed_family_seed_runs": len(runs),
            "completed_fold_fits": len(fold_epochs),
            "families": len(MODEL_NAMES),
            "oof_rows_per_family": EXPECTED_OOF_ROWS,
            "aggregate_prediction_rows": int(len(aggregate_predictions)),
            "within_peptide_evaluable_groups": EXPECTED_WITHIN_PEPTIDE_GROUPS,
            "aggregate_epoch_rule": (
                "integer median of all 15 fold-best epochs per family"
            ),
            "probability_to_logit_rule": (
                "exact inverse sigmoid with no clipping; zero or one causes failure"
            ),
            "retention_labels_read": False,
        },
    )


if __name__ == "__main__":
    main()
