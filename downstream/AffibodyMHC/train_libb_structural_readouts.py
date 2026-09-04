#!/usr/bin/env python
"""Train matched LibB classifiers over frozen structural representations.

The command accepts selection-derived labels, a label-free structural feature
archive, and a label-free sequence-record file.  It has no direct-measurement
argument.  Model selection uses only the existing three diagonal double-cold
weak-label folds; rows sharing exactly one held partner remain guard rows.

``--seed-mode pilot`` trains one prespecified final seed.  Once that run is
audited, ``--seed-mode full`` trains the five fixed seeds.  Every final score
file has exactly the sealed evaluator schema ``eval_row_id,model,seed,score``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.libb_structural_readout import (
    FEATURE_SOURCE,
    FROZEN_INJECTIVE_READOUT,
    SEQUENCE_SOURCE,
    FeatureView,
    MatchedInputDataset,
    OpaqueFeatureStore,
    build_matched_readout,
    canonical_json_sha256,
    class_balanced_weights,
    class_weighted_binary_cross_entropy,
    configured_readout_mode,
    load_sequence_control_vectors,
    materialize_input_vectors,
    sha256_file,
    trainable_parameter_count,
)
from downstream.AffibodyMHC.train_esmfold2_libb_readouts import (
    _membership_sha256,
    load_canonical_training_labels,
)


SCHEMA_VERSION = "libb-structural-frozen-readouts-v1"
SELECTED_EPOCH_SCHEMA_VERSION = "libb-structural-selected-epochs-v1"
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
EXPECTED_WEAK_ROWS = 30_648
EXPECTED_WEAK_POSITIVE = 23_725
EXPECTED_WEAK_NEGATIVE = 6_923
EXPECTED_EVALUATION_ROWS = 120
EXPECTED_FINAL_SEEDS = 5
EXPECTED_CANONICAL_ROWS_SHA256 = "c179996caafc91fb17ed6e56c56953ab01c6dfdf921f138b254c8aea1951e98e"
EXPECTED_REFERENCE_PDB_SHA256 = "59d026542cc42006302117cc79ad720497e69c4298f95598413f1d172d407b0e"
EXPECTED_SEQUENCE_RECORDS_SHA256 = "f69f162d07a8cbcd778ef3e7db108e03163ebbc04f7eb6b1945089c9d49153e4"
EXPECTED_SEQUENCE_CONTRACT_MANIFEST_SHA256 = "d53fc4da6ec45273a07d6fa9c4a33bcde2849e9d688295f7673250738e52f51a"
BLINDED_PREDICTION_COLUMNS = ("eval_row_id", "model", "seed", "score")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_sequence_contract(records_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Bind label-free records to the audited fixed-crystal/split contract."""

    records_path = Path(records_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        sha256_file(manifest_path) == EXPECTED_SEQUENCE_CONTRACT_MANIFEST_SHA256,
        "sequence-contract manifest checksum changed",
    )
    _require(manifest.get("schema_version") == "libb-fixed-crystal-contract-v2", "sequence-contract schema must be v2")
    access = manifest.get("data_access", {})
    _require(access.get("evaluation_outcome_table_read") is False, "sequence contract read direct outcomes")
    _require(access.get("training_label_table_read") is False, "sequence contract read weak labels")
    inputs = manifest.get("input_hashes", {})
    _require(inputs.get("canonical_rows_sha256") == EXPECTED_CANONICAL_ROWS_SHA256, "sequence contract canonical rows changed")
    _require(inputs.get("pdb_sha256") == EXPECTED_REFERENCE_PDB_SHA256, "sequence contract reference PDB changed")
    output = manifest.get("outputs", {}).get("current_sequence_records.json", {})
    _require(output.get("rows") == EXPECTED_WEAK_ROWS + EXPECTED_EVALUATION_ROWS, "sequence-contract row count changed")
    _require(output.get("sha256") == EXPECTED_SEQUENCE_RECORDS_SHA256, "sequence-record contract hash changed")
    _require(output.get("sha256") == sha256_file(records_path), "sequence-record checksum changed")
    split = manifest.get("split_audit", {})
    _require(split.get("train") == EXPECTED_WEAK_ROWS and split.get("eval") == EXPECTED_EVALUATION_ROWS, "sequence-contract split counts changed")
    overlap = split.get("partner_overlap", {})
    for key in ("affibody_sequence_overlap", "peptide_sequence_overlap", "row_id_overlap", "sequence_pair_overlap"):
        _require(overlap.get(key) == 0, f"sequence-contract {key} is not zero")
    hla = manifest.get("structure_audit", {}).get("assay_hla_alignment", {})
    _require(hla.get("aligned_hla_residue_count") == 181, "v2 HLA aligned length changed")
    _require(hla.get("exact_identity_match_count") == 179, "v2 HLA identity count changed")
    _require(hla.get("identity_override_count") == 2, "v2 HLA override count changed")
    _require(hla.get("all_other_aligned_residues_match") is True, "v2 HLA alignment audit failed")
    overrides = hla.get("identity_overrides", [])
    _require(
        [value.get("substitution") for value in overrides] == ["Y84A", "W167A"],
        "v2 HLA identity overrides changed",
    )
    return manifest


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    views: tuple[FeatureView, ...]

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "ModelSpec":
        name = str(value.get("name", ""))
        source = str(value.get("source", ""))
        _require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)), f"unsafe model name {name!r}")
        _require(source in {SEQUENCE_SOURCE, FEATURE_SOURCE}, f"unknown source for {name}")
        raw_views = value.get("views", ())
        _require(isinstance(raw_views, list), f"{name} views must be a list")
        views = tuple(FeatureView.from_config(item) for item in raw_views)
        if source == SEQUENCE_SOURCE:
            _require(not views, "the sequence control cannot inspect structural features")
        else:
            _require(views, f"{name} has no structural feature views")
        return cls(name=name, source=source, views=views)


@dataclass(frozen=True)
class EnsembleSpec:
    """Prespecified scalar ensemble for independently pretrained RDE folds."""

    name: str
    members: tuple[str, ...]
    method: str

    @classmethod
    def from_config(
        cls, value: Mapping[str, object], available_models: set[str]
    ) -> "EnsembleSpec":
        name = str(value.get("name", ""))
        members = tuple(map(str, value.get("members", ())))
        method = str(value.get("method", ""))
        _require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)), "unsafe ensemble name")
        _require(len(members) >= 2 and len(set(members)) == len(members), f"{name} members changed")
        _require(set(members).issubset(available_models), f"{name} has unknown members")
        _require(method == "mean_logit", f"{name} must use prespecified mean_logit")
        _require(name not in available_models, f"ensemble name collides with model {name}")
        return cls(name=name, members=members, method=method)


def read_config(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on deviations from the established LibB protocol."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    _require(config.get("schema_version") == SCHEMA_VERSION, "config schema changed")
    protocol_name = config.get("protocol_config")
    if protocol_name is not None:
        protocol_path = (path.parent / str(protocol_name)).resolve()
        _require(protocol_path.parent == path.parent, "protocol config must be a sibling file")
        with protocol_path.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        _require(
            protocol.get("schema_version") == "esmfold2-libb-frozen-readouts-v1",
            "base LibB protocol schema changed",
        )
        for section in ("dataset", "validation", "optimization"):
            merged = dict(protocol.get(section, {}))
            merged.update(config.get(section, {}))
            config[section] = merged
    dataset = config.get("dataset", {})
    _require(dataset.get("library") == "LibB", "only LibB is allowed")
    _require(int(dataset.get("rows", -1)) == EXPECTED_WEAK_ROWS, "weak-label row count changed")
    _require(int(dataset.get("positive", -1)) == EXPECTED_WEAK_POSITIVE, "weak positive count changed")
    _require(int(dataset.get("negative", -1)) == EXPECTED_WEAK_NEGATIVE, "weak negative count changed")
    _require(int(dataset.get("evaluation_rows", -1)) == EXPECTED_EVALUATION_ROWS, "evaluation row count changed")

    validation = config.get("validation", {})
    _require(validation.get("regime") == "double_cold", "validation must be double_cold")
    _require(int(validation.get("folds", -1)) == 3, "validation must use three folds")
    _require(int(validation.get("split_seed", -1)) == 17, "split seed must remain 17")
    _require(validation.get("model_selection_target") == "weak_validation_log_loss", "model selection target changed")
    _require(set(validation.get("fold_contracts", {})) == {"0", "1", "2"}, "fold contracts are incomplete")

    optimization = config.get("optimization", {})
    _require(optimization.get("loss") == "class_weighted_bce", "loss must be class-weighted BCE")
    _require(optimization.get("class_weight_formula") == "N/(2*N_class)", "class weights changed")
    _require(optimization.get("optimizer") == "AdamW", "optimizer changed")
    _require(int(optimization.get("max_epochs", 0)) >= 1, "max epochs must be positive")
    _require(int(optimization.get("early_stopping_patience", 0)) >= 1, "patience must be positive")

    raw_models = config.get("models")
    _require(isinstance(raw_models, list) and raw_models, "config has no models")
    specs = [ModelSpec.from_config(value) for value in raw_models]
    _require(len({spec.name for spec in specs}) == len(specs), "duplicate model name")
    sequence_specs = [spec for spec in specs if spec.source == SEQUENCE_SOURCE]
    require_sequence_control = bool(
        config.get("comparison", {}).get("require_sequence_control", True)
    )
    if require_sequence_control:
        _require(len(sequence_specs) == 1, "exactly one seven-position sequence control is required")
    else:
        _require(len(sequence_specs) <= 1, "at most one seven-position sequence control is allowed")
    raw_ensembles = config.get("prediction_ensembles", [])
    _require(isinstance(raw_ensembles, list), "prediction ensembles must be a list")
    ensembles = [
        EnsembleSpec.from_config(value, {spec.name for spec in specs})
        for value in raw_ensembles
    ]
    _require(len({value.name for value in ensembles}) == len(ensembles), "duplicate ensemble name")

    architecture = config.get("architecture", {})
    readout_mode = configured_readout_mode(architecture)
    hidden_dims = architecture.get("hidden_dims")
    _require(isinstance(hidden_dims, list) and len(hidden_dims) == 2, "two hidden widths are required")
    _require(min(map(int, hidden_dims)) >= 1, "hidden widths must be positive")
    _require(0.0 <= float(architecture.get("dropout", -1.0)) < 1.0, "dropout must be in [0,1)")
    expected_parameters = config.get("expected_trainable_parameters")
    if readout_mode == FROZEN_INJECTIVE_READOUT:
        _require(int(architecture.get("adapter_dim", 0)) >= 140, "shared adapter is narrower than sequence input")
        _require(int(architecture.get("projection_seed", -1)) >= 0, "projection seed is missing")
        _require(isinstance(expected_parameters, int) and expected_parameters >= 1, "expected parameter count is missing")
    else:
        _require(
            "adapter_dim" not in architecture and "projection_seed" not in architecture,
            "native learned projection must not contain legacy adapter settings",
        )
        _require(
            architecture.get("input_adapter") in (None, "trainable native projection"),
            "native learned projection has stale input-adapter metadata",
        )
        _require(
            isinstance(expected_parameters, dict) and bool(expected_parameters),
            "native learned projection requires per-model expected parameter counts",
        )
        _require(
            set(map(str, expected_parameters)) == {spec.name for spec in specs}
            and all(
                isinstance(value, int) and value >= 1
                for value in expected_parameters.values()
            ),
            "native expected parameter counts must cover every configured model",
        )
    archive_contract = config.get("feature_archive_contract", {})
    _require(
        isinstance(archive_contract.get("required_manifest"), dict)
        and bool(archive_contract["required_manifest"]),
        "model-specific feature archive provenance is not pinned",
    )
    expected_dimensions = config.get("expected_input_dimensions")
    _require(isinstance(expected_dimensions, dict), "expected input dimensions are missing")
    _require(
        set(expected_dimensions) == {spec.name for spec in specs},
        "expected input dimensions do not cover every configured model",
    )

    final = config.get("final_training", {})
    seeds = tuple(map(int, final.get("seeds", ())))
    _require(len(seeds) == EXPECTED_FINAL_SEEDS and len(set(seeds)) == EXPECTED_FINAL_SEEDS, "exactly five unique final seeds are required")
    _require(int(final.get("pilot_seed", -1)) in seeds, "pilot seed must be one fixed final seed")
    _require(final.get("backbone_frozen") is True, "structural backbone must remain frozen")
    _require(final.get("retention_labels_allowed") is False, "direct labels must be forbidden")
    return config


def model_specs(config: Mapping[str, Any]) -> tuple[ModelSpec, ...]:
    return tuple(ModelSpec.from_config(value) for value in config["models"])


def ensemble_specs(config: Mapping[str, Any]) -> tuple[EnsembleSpec, ...]:
    available = {spec.name for spec in model_specs(config)}
    return tuple(
        EnsembleSpec.from_config(value, available)
        for value in config.get("prediction_ensembles", ())
    )


def select_model_specs(
    specs: Sequence[ModelSpec], requested: str | None
) -> tuple[ModelSpec, ...]:
    """Select prespecified arms so independent GPU jobs can run in parallel."""

    if requested is None:
        return tuple(specs)
    names = tuple(value.strip() for value in requested.split(",") if value.strip())
    _require(names and len(set(names)) == len(names), "--models must list unique names")
    lookup = {spec.name: spec for spec in specs}
    unknown = sorted(set(names).difference(lookup))
    _require(not unknown, f"--models contains unknown configured arms: {unknown}")
    return tuple(lookup[name] for name in names)


def required_archive_arrays(specs: Sequence[ModelSpec]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            name
            for spec in specs
            for view in spec.views
            for name in ((view.array,) if view.mask is None else (view.array, view.mask))
        )
    )


def validate_store_membership(
    base: pd.DataFrame, store: OpaqueFeatureStore, config: Mapping[str, Any]
) -> None:
    expected_total = int(config["dataset"]["rows"]) + int(config["dataset"]["evaluation_rows"])
    _require(len(store.row_ids) == expected_total, "feature archive total row count changed")
    train_ids = {row_id for row_id, split in zip(store.row_ids, store.splits) if split == "train"}
    eval_ids = {row_id for row_id, split in zip(store.row_ids, store.splits) if split == "eval"}
    _require(len(train_ids) == EXPECTED_WEAK_ROWS, "feature archive weak split changed")
    _require(len(eval_ids) == EXPECTED_EVALUATION_ROWS, "feature archive evaluation split changed")
    _require(train_ids == set(base["row_id"]), "feature and weak-label training membership differ")
    _require(train_ids.isdisjoint(eval_ids), "feature train/evaluation membership overlaps")


def cache_indices(frame: pd.DataFrame, store: OpaqueFeatureStore) -> np.ndarray:
    lookup = store.row_id_to_index
    missing = sorted(set(frame["row_id"]).difference(lookup))
    _require(not missing, f"feature archive misses {len(missing)} requested rows")
    return np.asarray([lookup[str(row_id)] for row_id in frame["row_id"]], dtype=np.int64)


def make_dataset(
    frame: pd.DataFrame,
    store: OpaqueFeatureStore,
    spec: ModelSpec,
    input_vectors: np.ndarray,
    *,
    labels: bool,
) -> MatchedInputDataset:
    return MatchedInputDataset(
        store=store,
        cache_indices=cache_indices(frame, store),
        row_ids=frame["row_id"].astype(str).tolist(),
        source=spec.source,
        views=spec.views,
        materialized_vectors=input_vectors,
        labels=frame["weak_label"].to_numpy(dtype=int) if labels else None,
    )


def input_dimensions(
    specs: Sequence[ModelSpec],
    store: OpaqueFeatureStore,
    vectors_by_model: Mapping[str, np.ndarray],
) -> dict[str, int]:
    probe = pd.DataFrame({"row_id": [store.row_ids[0]], "weak_label": [0]})
    return {
        spec.name: int(
            make_dataset(
                probe, store, spec, vectors_by_model[spec.name], labels=False
            ).input_dim
        )
        for spec in specs
    }


def validate_matched_capacity(
    dimensions: Mapping[str, int], config: Mapping[str, Any]
) -> dict[str, int]:
    """Validate legacy equal capacity or native projection parameter counts."""

    architecture = config["architecture"]
    expected_dimensions = {
        str(name): int(value)
        for name, value in config.get("expected_input_dimensions", {}).items()
        if name in dimensions
    }
    _require(
        dict(dimensions) == expected_dimensions,
        f"materialized input dimensions changed: {dict(dimensions)}",
    )
    readout_mode = configured_readout_mode(architecture)
    if readout_mode == FROZEN_INJECTIVE_READOUT:
        adapter_dim = int(architecture["adapter_dim"])
        _require(max(dimensions.values()) <= adapter_dim, "adapter would compress at least one input")
    observed = {
        name: trainable_parameter_count(build_matched_readout(size, architecture))
        for name, size in dimensions.items()
    }
    expected_raw = config["expected_trainable_parameters"]
    if readout_mode == FROZEN_INJECTIVE_READOUT:
        _require(len(set(observed.values())) == 1, "readouts are not capacity matched")
        expected = int(expected_raw)
        _require(set(observed.values()) == {expected}, f"trainable parameter count changed: {observed}")
    else:
        _require(isinstance(expected_raw, dict), "native expected parameter counts must be a mapping")
        expected = {
            str(name): int(value)
            for name, value in expected_raw.items()
            if name in dimensions
        }
        _require(set(expected) == set(dimensions), "native expected parameter counts do not cover every model")
        _require(observed == expected, f"trainable parameter count changed: {observed}")
    return observed


def make_loader(
    dataset: MatchedInputDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        persistent_workers=bool(workers),
        generator=torch.Generator().manual_seed(int(seed)),
    )


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    model.eval()
    row_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            values = batch["input_vector"].to(device=device, dtype=torch.float32, non_blocking=True)
            probabilities.append(torch.sigmoid(model(values)).cpu().numpy())
            row_ids.extend(map(str, batch["row_id"]))
            if "label" in batch:
                labels.append(batch["label"].cpu().numpy())
    scores = np.concatenate(probabilities)
    observed = np.concatenate(labels).astype(int) if labels else None
    _require(len(row_ids) == len(scores), "prediction row-ID and score lengths differ")
    if observed is not None:
        _require(len(observed) == len(scores), "prediction label and score lengths differ")
    return row_ids, scores, observed


def weak_validation_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "weak validation fold needs both classes")
    return {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    examples = 0
    for batch in loader:
        values = batch["input_vector"].to(device=device, dtype=torch.float32, non_blocking=True)
        labels = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = class_weighted_binary_cross_entropy(model(values), labels, class_weights)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * int(labels.numel())
        examples += int(labels.numel())
    return total / examples


def fit_one_fold(
    spec: ModelSpec,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    store: OpaqueFeatureStore,
    input_vectors: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame]:
    _set_seed(seed)
    optimization = config["optimization"]
    model = build_matched_readout(input_dim, config["architecture"]).to(device)
    train_loader = make_loader(
        make_dataset(train_frame, store, spec, input_vectors, labels=True),
        optimization["batch_size"], True, seed, optimization["num_workers"], device.type == "cuda",
    )
    validation_loader = make_loader(
        make_dataset(validation_frame, store, spec, input_vectors, labels=True),
        optimization["evaluation_batch_size"], False, seed, optimization["num_workers"], device.type == "cuda",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    weights = class_balanced_weights(train_frame["weak_label"].to_numpy(dtype=int)).to(device)
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: tuple[list[str], np.ndarray, np.ndarray | None] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, int(optimization["max_epochs"]) + 1):
        training_loss = train_epoch(model, train_loader, optimizer, weights, device)
        prediction = predict(model, validation_loader, device)
        if prediction[2] is None:
            raise RuntimeError("weak validation labels unexpectedly absent")
        metrics = weak_validation_metrics(prediction[2], prediction[1])
        history.append({"epoch": epoch, "train_weighted_bce": training_loss, **metrics})
        if float(metrics["log_loss"]) < best_loss - float(optimization["early_stopping_min_delta"]):
            best_loss = float(metrics["log_loss"])
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_prediction = prediction
            stale = 0
        else:
            stale += 1
            if stale >= int(optimization["early_stopping_patience"]):
                break
    if best_state is None or best_prediction is None:
        raise RuntimeError("training never produced a weak-validation checkpoint")
    model.load_state_dict(best_state)
    row_ids, probabilities, labels = best_prediction
    if labels is None:
        raise RuntimeError("best weak-validation labels unexpectedly absent")
    record = {
        "model": spec.name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_metrics": weak_validation_metrics(labels, probabilities),
        "history": history,
    }
    return model, record, pd.DataFrame(
        {"row_id": row_ids, "weak_label": labels, "probability": probabilities}
    )


def _fold_seed(base_seed: int, fold: int) -> int:
    """Use paired initialization/shuffle seeds across all matched model arms."""

    payload = f"{int(base_seed)}|{int(fold)}".encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def run_cross_validation(
    base: pd.DataFrame,
    fold_membership: pd.DataFrame,
    specs: Sequence[ModelSpec],
    dimensions: Mapping[str, int],
    store: OpaqueFeatureStore,
    vectors_by_model: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    *,
    sequence_records_sha256: str,
    sequence_contract_manifest_sha256: str,
) -> dict[str, int]:
    summaries: list[dict[str, Any]] = []
    prediction_blocks: list[pd.DataFrame] = []
    selected_epochs: dict[str, int] = {}
    checkpoints = output_dir / "cv_checkpoints"
    checkpoints.mkdir()
    for spec in specs:
        records: list[dict[str, Any]] = []
        for fold in range(3):
            membership = fold_membership.loc[fold_membership["fold"].eq(fold)]
            train_ids = set(membership.loc[membership["role"].eq("train"), "row_id"])
            validation_ids = set(membership.loc[membership["role"].eq("validation"), "row_id"])
            train_frame = base.loc[base["row_id"].isin(train_ids)].reset_index(drop=True)
            validation_frame = base.loc[base["row_id"].isin(validation_ids)].reset_index(drop=True)
            seed = _fold_seed(int(config["validation"]["training_seed"]), fold)
            model, record, predictions = fit_one_fold(
                spec, train_frame, validation_frame, store, vectors_by_model[spec.name],
                config, seed, dimensions[spec.name], device,
            )
            record.update({"fold": fold, "n_train": int(len(train_frame)), "n_guard": int(membership["role"].eq("guard").sum())})
            records.append(record)
            predictions.insert(0, "fold", fold)
            predictions.insert(0, "model", spec.name)
            prediction_blocks.append(predictions)
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": spec.name,
                    "source": spec.source,
                    "fold": fold,
                    "seed": seed,
                    "best_epoch": record["best_epoch"],
                    "input_dim": dimensions[spec.name],
                    "config_sha256": canonical_json_sha256(config),
                    "feature_archive_manifest_sha256": store.manifest_sha256,
                    "state_dict": model.state_dict(),
                },
                checkpoints / f"{spec.name}__fold{fold}.pt",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        epochs = [int(record["best_epoch"]) for record in records]
        selected_epochs[spec.name] = max(1, int(np.median(epochs)))
        summaries.append(
            {
                "model": spec.name,
                "source": spec.source,
                "folds": records,
                "selected_final_epochs": selected_epochs[spec.name],
                "selection_rule": "integer median of three weak-validation best epochs",
            }
        )
    pd.concat(prediction_blocks, ignore_index=True).to_csv(
        output_dir / "weak_validation_predictions.csv.gz", index=False
    )
    os.chmod(output_dir / "weak_validation_predictions.csv.gz", 0o600)
    summary_path = output_dir / "weak_validation_summary.json"
    _json_dump(summary_path, summaries)
    _json_dump(
        output_dir / "selected_epochs.json",
        {
            "schema_version": SELECTED_EPOCH_SCHEMA_VERSION,
            "config_sha256": canonical_json_sha256(config),
            "feature_archive_manifest_sha256": store.manifest_sha256,
            "sequence_records_sha256": str(sequence_records_sha256),
            "sequence_contract_manifest_sha256": str(
                sequence_contract_manifest_sha256
            ),
            "weak_membership_sha256": _membership_sha256(base["row_id"]),
            "weak_validation_summary_file": summary_path.name,
            "weak_validation_summary_sha256": sha256_file(summary_path),
            "models": {
                summary["model"]: {
                    "fold_best_epochs": [
                        int(record["best_epoch"]) for record in summary["folds"]
                    ],
                    "selected_final_epochs": int(summary["selected_final_epochs"]),
                }
                for summary in summaries
            },
        },
    )
    return selected_epochs


def load_selected_epochs(
    path: str | Path,
    config: Mapping[str, Any],
    base: pd.DataFrame,
    requested_models: Sequence[str],
    *,
    feature_archive_manifest_sha256: str,
    sequence_records_sha256: str,
    sequence_contract_manifest_sha256: str,
) -> dict[str, int]:
    """Verify that final epochs are exactly the medians from bound weak CV."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("schema_version") == SELECTED_EPOCH_SCHEMA_VERSION, "selected-epoch schema changed")
    _require(payload.get("config_sha256") == canonical_json_sha256(config), "selected epochs use a different config")
    _require(
        payload.get("feature_archive_manifest_sha256")
        == str(feature_archive_manifest_sha256),
        "selected epochs use a different feature archive",
    )
    _require(
        payload.get("sequence_records_sha256") == str(sequence_records_sha256),
        "selected epochs use different sequence records",
    )
    _require(
        payload.get("sequence_contract_manifest_sha256")
        == str(sequence_contract_manifest_sha256),
        "selected epochs use a different sequence contract",
    )
    _require(
        payload.get("weak_membership_sha256") == _membership_sha256(base["row_id"]),
        "selected epochs use a different weak-label membership",
    )
    summary_path = path.parent / str(payload.get("weak_validation_summary_file", ""))
    _require(summary_path.parent == path.parent and summary_path.is_file(), "bound weak-validation summary is missing")
    _require(
        payload.get("weak_validation_summary_sha256") == sha256_file(summary_path),
        "bound weak-validation summary checksum changed",
    )
    with summary_path.open("r", encoding="utf-8") as handle:
        summaries = json.load(handle)
    summary_by_model = {str(value["model"]): value for value in summaries}
    model_contracts = payload.get("models", {})
    output: dict[str, int] = {}
    for name in requested_models:
        _require(name in model_contracts and name in summary_by_model, f"weak CV lacks {name}")
        record = model_contracts[name]
        epochs = tuple(map(int, record.get("fold_best_epochs", ())))
        _require(len(epochs) == 3 and min(epochs) >= 1, f"{name} fold epochs changed")
        summary_epochs = tuple(
            int(value["best_epoch"]) for value in summary_by_model[name]["folds"]
        )
        _require(epochs == summary_epochs, f"{name} fold epochs disagree with weak CV")
        selected = max(1, int(np.median(epochs)))
        _require(selected == int(record.get("selected_final_epochs", -1)), f"{name} selected epoch is not the fold median")
        _require(
            selected == int(summary_by_model[name].get("selected_final_epochs", -1)),
            f"{name} selected epoch disagrees with weak CV summary",
        )
        output[name] = selected
    return output


def fit_fixed_epochs(
    spec: ModelSpec,
    frame: pd.DataFrame,
    store: OpaqueFeatureStore,
    input_vectors: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    epochs: int,
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    _set_seed(seed)
    optimization = config["optimization"]
    model = build_matched_readout(input_dim, config["architecture"]).to(device)
    loader = make_loader(
        make_dataset(frame, store, spec, input_vectors, labels=True),
        optimization["batch_size"], True, seed, optimization["num_workers"], device.type == "cuda",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    weights = class_balanced_weights(frame["weak_label"].to_numpy(dtype=int)).to(device)
    losses = [train_epoch(model, loader, optimizer, weights, device) for _ in range(int(epochs))]
    return model, losses


def _evaluation_frame(store: OpaqueFeatureStore) -> pd.DataFrame:
    row_ids = [row_id for row_id, split in zip(store.row_ids, store.splits) if split == "eval"]
    _require(len(row_ids) == EXPECTED_EVALUATION_ROWS, "sealed panel must contain 120 rows")
    return pd.DataFrame({"row_id": row_ids})


def validate_blinded_output(frame: pd.DataFrame) -> None:
    _require(tuple(frame.columns) == BLINDED_PREDICTION_COLUMNS, "blinded prediction columns changed")
    _require(len(frame) == EXPECTED_EVALUATION_ROWS, "blinded prediction must contain 120 rows")
    _require(not bool(frame["eval_row_id"].duplicated().any()), "duplicate blinded row ID")
    _require(bool(np.isfinite(frame["score"].to_numpy(dtype=float)).all()), "nonfinite blinded score")


def mean_logit_ensemble(
    frames: Sequence[pd.DataFrame], ensemble_name: str, seed: int
) -> pd.DataFrame:
    """Average scalar logits, never latent vectors from unrelated RDE folds."""

    _require(len(frames) >= 2, "an ensemble needs at least two prediction frames")
    expected_ids = frames[0]["eval_row_id"].astype(str).tolist()
    logits = []
    for frame in frames:
        validate_blinded_output(frame)
        _require(
            frame["eval_row_id"].astype(str).tolist() == expected_ids,
            "ensemble prediction row order differs",
        )
        probability = np.clip(frame["score"].to_numpy(dtype=float), 1e-7, 1.0 - 1e-7)
        logits.append(np.log(probability) - np.log1p(-probability))
    mean_logit = np.mean(np.stack(logits), axis=0)
    probability = np.empty_like(mean_logit)
    positive = mean_logit >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-mean_logit[positive]))
    exp_value = np.exp(mean_logit[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    output = pd.DataFrame(
        {
            "eval_row_id": expected_ids,
            "model": str(ensemble_name),
            "seed": str(seed),
            "score": probability,
        },
        columns=BLINDED_PREDICTION_COLUMNS,
    )
    validate_blinded_output(output)
    return output


def final_seeds(config: Mapping[str, Any], seed_mode: str) -> tuple[int, ...]:
    _require(seed_mode in {"pilot", "full"}, "seed mode must be pilot or full")
    final = config["final_training"]
    return (
        (int(final["pilot_seed"]),)
        if seed_mode == "pilot"
        else tuple(map(int, final["seeds"]))
    )


def run_final_training(
    base: pd.DataFrame,
    specs: Sequence[ModelSpec],
    dimensions: Mapping[str, int],
    store: OpaqueFeatureStore,
    vectors_by_model: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    selected_epochs: Mapping[str, int],
    seed_mode: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    prediction_frame = _evaluation_frame(store)
    checkpoints = output_dir / "final_checkpoints"
    predictions_dir = output_dir / "blinded_predictions"
    checkpoints.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    frozen_outputs: dict[tuple[str, int], pd.DataFrame] = {}
    for spec in specs:
        _require(spec.name in selected_epochs, f"selected epoch missing for {spec.name}")
        epochs = int(selected_epochs[spec.name])
        _require(1 <= epochs <= int(config["optimization"]["max_epochs"]), "selected epoch is out of range")
        for seed in final_seeds(config, seed_mode):
            started = time.time()
            model, losses = fit_fixed_epochs(
                spec, base, store, vectors_by_model[spec.name], config, seed, epochs,
                dimensions[spec.name], device,
            )
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model": spec.name,
                    "source": spec.source,
                    "seed": seed,
                    "epochs": epochs,
                    "input_dim": dimensions[spec.name],
                    "config_sha256": canonical_json_sha256(config),
                    "feature_archive_manifest_sha256": store.manifest_sha256,
                    "weak_membership_sha256": _membership_sha256(base["row_id"]),
                    "state_dict": model.state_dict(),
                },
                checkpoints / f"{spec.name}__seed{seed}.pt",
            )
            dataset = make_dataset(
                prediction_frame.assign(weak_label=0),
                store,
                spec,
                vectors_by_model[spec.name],
                labels=False,
            )
            loader = make_loader(
                dataset, config["optimization"]["evaluation_batch_size"], False,
                seed, config["optimization"]["num_workers"], device.type == "cuda",
            )
            row_ids, probabilities, labels = predict(model, loader, device)
            if labels is not None:
                raise RuntimeError("sealed prediction panel unexpectedly contained labels")
            output = pd.DataFrame(
                {
                    "eval_row_id": row_ids,
                    "model": spec.name,
                    "seed": str(seed),
                    "score": probabilities,
                },
                columns=BLINDED_PREDICTION_COLUMNS,
            )
            validate_blinded_output(output)
            frozen_outputs[(spec.name, seed)] = output.copy()
            path = predictions_dir / f"{spec.name}__seed{seed}.csv"
            output.to_csv(path, index=False)
            os.chmod(path, 0o600)
            records.append(
                {
                    "model": spec.name,
                    "source": spec.source,
                    "seed": seed,
                    "epochs": epochs,
                    "epoch_weighted_bce": losses,
                    "prediction_rows": len(output),
                    "elapsed_seconds": time.time() - started,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    selected_names = {spec.name for spec in specs}
    for ensemble in ensemble_specs(config):
        if not set(ensemble.members).issubset(selected_names):
            if set(ensemble.members).intersection(selected_names):
                warnings.warn(
                    f"skipping {ensemble.name}: this job did not include all members",
                    RuntimeWarning,
                )
                records.append(
                    {
                        "model": ensemble.name,
                        "source": "prespecified_scalar_ensemble",
                        "status": "skipped_missing_members",
                        "members": list(ensemble.members),
                        "selected_members": sorted(
                            set(ensemble.members).intersection(selected_names)
                        ),
                    }
                )
            continue
        for seed in final_seeds(config, seed_mode):
            output = mean_logit_ensemble(
                [frozen_outputs[(member, seed)] for member in ensemble.members],
                ensemble.name,
                seed,
            )
            path = predictions_dir / f"{ensemble.name}__seed{seed}.csv"
            output.to_csv(path, index=False)
            os.chmod(path, 0o600)
            records.append(
                {
                    "model": ensemble.name,
                    "source": "prespecified_scalar_ensemble",
                    "seed": seed,
                    "members": list(ensemble.members),
                    "method": ensemble.method,
                    "prediction_rows": len(output),
                }
            )
    _json_dump(output_dir / "final_training_summary.json", records)


def validate_output_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("outputs must stay under private_data") from error
    _require(relative.parts, "refusing to write directly into private_data")
    _require(not output.exists(), "output directory already exists; refusing overwrite")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-labels-manifest", type=Path, required=True)
    parser.add_argument("--feature-archive", type=Path, required=True)
    parser.add_argument(
        "--sequence-records",
        type=Path,
        required=True,
        help="label-free records used to recheck strict train/evaluation partner disjointness",
    )
    parser.add_argument("--sequence-records-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("audit", "cross_validate", "fit_final", "all"), default="audit")
    parser.add_argument("--seed-mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument(
        "--models",
        help="optional comma-separated configured arms; use distinct output dirs for parallel GPU jobs",
    )
    parser.add_argument("--selected-epochs-json", type=Path)
    parser.add_argument("--verify-cache-checksums", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    specs = select_model_specs(model_specs(config), args.models)
    base, fold_membership = load_canonical_training_labels(
        args.training_labels.resolve(), args.training_labels_manifest.resolve(), config
    )
    checksums_verified = bool(args.verify_cache_checksums or args.mode == "audit")
    store = OpaqueFeatureStore.open(
        args.feature_archive,
        required_archive_arrays(specs),
        verify_all_checksums=checksums_verified,
        required_manifest=config["feature_archive_contract"]["required_manifest"],
    )
    validate_store_membership(base, store, config)
    sequence_contract = validate_sequence_contract(
        args.sequence_records, args.sequence_records_manifest
    )
    sequence_records_sha256 = sha256_file(args.sequence_records.resolve())
    sequence_contract_manifest_sha256 = sha256_file(
        args.sequence_records_manifest.resolve()
    )
    sequence_vectors = load_sequence_control_vectors(
        args.sequence_records, store.row_ids, store.splits
    )
    vectors_by_model = {
        spec.name: materialize_input_vectors(
            store,
            spec.source,
            spec.views,
            sequence_vectors if spec.source == SEQUENCE_SOURCE else None,
        )
        for spec in specs
    }
    dimensions = input_dimensions(specs, store, vectors_by_model)
    parameters = validate_matched_capacity(dimensions, config)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "retention_labels_read": False,
        "library": "LibB",
        "weak_rows": len(base),
        "weak_positive": int(base["weak_label"].sum()),
        "weak_negative": int(base["weak_label"].eq(0).sum()),
        "evaluation_rows": int(sum(split == "eval" for split in store.splits)),
        "validation": "three diagonal double-cold weak-label folds with XOR guard rows",
        "input_dimensions": dimensions,
        "trainable_parameters": parameters,
        "readout_mode": configured_readout_mode(config["architecture"]),
        "capacity_matched": len(set(parameters.values())) == 1,
        "feature_archive_manifest_sha256": store.manifest_sha256,
        "feature_array_checksums_verified": checksums_verified,
        "sequence_records_sha256": sequence_records_sha256,
        "sequence_contract_manifest_sha256": sequence_contract_manifest_sha256,
        "seed_mode": args.seed_mode,
        "final_seeds": list(final_seeds(config, args.seed_mode)),
    }
    if args.mode == "audit":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    _require(args.output_dir is not None, "training modes require --output-dir")
    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, mode=0o700)
    shutil.copy2(args.config, output_dir / "config.json")
    os.chmod(output_dir / "config.json", 0o600)
    _json_dump(output_dir / "audit.json", audit)
    device = torch.device(args.device)
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")

    if args.mode in {"cross_validate", "all"}:
        selected_epochs = run_cross_validation(
            base, fold_membership, specs, dimensions, store, vectors_by_model,
            config, output_dir, device,
            sequence_records_sha256=sequence_records_sha256,
            sequence_contract_manifest_sha256=sequence_contract_manifest_sha256,
        )
    else:
        _require(args.selected_epochs_json is not None, "fit_final requires --selected-epochs-json")
        selected_epochs = load_selected_epochs(
            args.selected_epochs_json,
            config,
            base,
            [spec.name for spec in specs],
            feature_archive_manifest_sha256=store.manifest_sha256,
            sequence_records_sha256=sequence_records_sha256,
            sequence_contract_manifest_sha256=sequence_contract_manifest_sha256,
        )

    if args.mode in {"fit_final", "all"}:
        run_final_training(
            base, specs, dimensions, store, vectors_by_model, config,
            selected_epochs, args.seed_mode, output_dir, device,
        )


if __name__ == "__main__":
    main()
