"""Train leakage-safe LibA or LibB readouts over frozen ESMFold2 features.

This command deliberately has no retention-data argument.  Its only labels are
the canonical selection-derived weak labels in a physically separate
training sidecar.  Partner sequence hashes in that sidecar reproduce the exact
three diagonal double-identity-cold partitions (split seed 17), including the
XOR guard rows.

The normal workflow is:

1. ``cross_validate`` to choose only the number of training epochs from weak
   validation loss for each prespecified readout;
2. ``fit_final`` to train five head seeds on all weak pairs and emit
   label-free predictions for requested cache row IDs.

The separate sealed evaluator may later join those predictions to direct-
retention measurements.  This module never performs that join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Permit both ``python -m downstream...`` and direct script execution.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.esmfold2_libb_readout import (
    MODEL_NAMES,
    FrozenFeatureDataset,
    FrozenFeatureStore,
    build_readout,
    class_balanced_weights,
    class_weighted_binary_cross_entropy,
    model_feature_names,
    parameter_report,
)


PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SCHEMA_VERSION = "esmfold2-libb-frozen-readouts-v1"
LIBA_SCHEMA_VERSION = "esmfold2-liba-frozen-readouts-v1"
READOUT_SCHEMA_BY_LIBRARY = {
    "LibA": LIBA_SCHEMA_VERSION,
    "LibB": SCHEMA_VERSION,
}
TRAINING_LABEL_COLUMNS = (
    "row_id",
    "weak_label",
    "peptide_id",
    "affibody_id",
    "chain1_sha256",
    "chain2_sha256",
)
TRAINING_LABEL_SCHEMA_VERSION = "esmfold2-libb-training-labels-v1"
LIBA_TRAINING_LABEL_SCHEMA_VERSION = "esmfold2-liba-training-labels-v1"
TRAINING_LABEL_SCHEMA_BY_LIBRARY = {
    "LibA": LIBA_TRAINING_LABEL_SCHEMA_VERSION,
    "LibB": TRAINING_LABEL_SCHEMA_VERSION,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _membership_sha256(row_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, row_ids))).encode("ascii")).hexdigest()


def _stable_bin(value: str, axis: str, seed: int, folds: int) -> int:
    payload = f"{axis}|{int(seed)}|{value}".encode("ascii")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % int(folds)


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    library = str(config.get("dataset", {}).get("library"))
    _require(library in READOUT_SCHEMA_BY_LIBRARY, "dataset library must be LibA or LibB")
    _require(
        config.get("schema_version") == READOUT_SCHEMA_BY_LIBRARY[library],
        "config schema mismatch",
    )
    _require(
        config.get("validation", {}).get("regime") == "double_cold",
        "validation must remain double_cold",
    )
    _require(
        int(config.get("validation", {}).get("folds", -1)) == 3,
        "canonical validation uses exactly three folds",
    )
    _require(
        int(config.get("validation", {}).get("split_seed", -1)) == 17,
        "canonical split seed must remain 17",
    )
    _require(
        config.get("optimization", {}).get("loss") == "class_weighted_bce",
        "loss must remain class-weighted BCE",
    )
    model_names = tuple(config.get("models", ()))
    _require(model_names == MODEL_NAMES, "model list or order differs from the locked comparison")
    final_seeds = tuple(int(value) for value in config.get("final_training", {}).get("seeds", ()))
    _require(len(final_seeds) == 5 and len(set(final_seeds)) == 5, "exactly five final seeds are required")
    observed_parameters = parameter_report(config["architecture"], model_names)
    expected_parameters = {
        str(key): int(value)
        for key, value in config.get("expected_trainable_parameters", {}).items()
    }
    _require(
        observed_parameters == expected_parameters,
        f"architecture parameter counts changed: {observed_parameters}",
    )
    return config


def _validate_output_dir(path: Path) -> Path:
    output = path.resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("outputs must stay under private_data") from error
    _require(relative.parts, "refusing to use private_data itself as output")
    _require(not output.exists(), "output directory already exists; refusing overwrite")
    return output


def _expected_role(
    chain1_identity: str,
    chain2_identity: str,
    fold: int,
    folds: int,
    split_seed: int,
) -> str:
    peptide_held = _stable_bin(chain1_identity, "peptide", split_seed, folds) == fold
    affibody_held = _stable_bin(chain2_identity, "affibody", split_seed, folds) == fold
    if peptide_held and affibody_held:
        return "validation"
    if not peptide_held and not affibody_held:
        return "train"
    return "guard"


def load_canonical_training_labels(
    path: Path,
    manifest_path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load weak labels and independently reconstruct the frozen fold contract."""

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    _require(tuple(header) == TRAINING_LABEL_COLUMNS, "unexpected training-label columns")
    _require(
        not any("retention" in column.lower() for column in header),
        "training sidecar must not contain retention data",
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("schema_version")
        == TRAINING_LABEL_SCHEMA_BY_LIBRARY[str(config["dataset"]["library"])],
        "training-label manifest schema changed",
    )
    output_contract = manifest.get("output", {})
    _require(output_contract.get("sha256") == _sha256_file(path), "training-label checksum changed")
    _require(tuple(output_contract.get("columns", ())) == TRAINING_LABEL_COLUMNS, "training-label manifest columns changed")

    base = pd.read_csv(
        path,
        dtype={column: str for column in TRAINING_LABEL_COLUMNS if column != "weak_label"},
        keep_default_na=False,
        na_filter=False,
    )
    base["weak_label"] = pd.to_numeric(base["weak_label"], errors="raise").astype(int)
    _require(set(base["weak_label"]) == {0, 1}, "weak labels are not binary")
    _require(not bool(base["row_id"].duplicated().any()), "duplicate training row ID")
    _require(bool(base[list(TRAINING_LABEL_COLUMNS)].astype(str).ne("").all().all()), "empty training-sidecar value")
    validation = config["validation"]
    folds = int(validation["folds"])
    split_seed = int(validation["split_seed"])
    expected_rows = int(config["dataset"]["rows"])
    _require(len(base) == expected_rows, "unique weak-pair count changed")
    base = base.sort_values("row_id").reset_index(drop=True)
    dataset_contract = config["dataset"]
    _require(int(base["weak_label"].sum()) == int(dataset_contract["positive"]), "positive count changed")
    _require(int(base["weak_label"].eq(0).sum()) == int(dataset_contract["negative"]), "negative count changed")
    _require(
        _membership_sha256(base["row_id"].tolist())
        == dataset_contract["row_id_membership_sha256"],
        "canonical training membership changed",
    )

    blocks = []
    for fold in range(folds):
        block = base.copy()
        block["fold"] = fold
        block["role"] = [
            _expected_role(chain1, chain2, fold, folds, split_seed)
            for chain1, chain2 in zip(block["chain1_sha256"], block["chain2_sha256"])
        ]
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    _require(set(frame["role"]) == {"train", "guard", "validation"}, "fold roles changed")

    fold_contracts = validation["fold_contracts"]
    for fold in range(folds):
        block = frame.loc[frame["fold"].eq(fold)]
        contract = fold_contracts[str(fold)]
        for role in ("train", "guard", "validation"):
            _require(
                int(block["role"].eq(role).sum()) == int(contract[role]),
                f"fold {fold} {role} count changed",
            )
        selected = block.loc[block["role"].eq("validation")]
        _require(
            int(selected["weak_label"].sum()) == int(contract["validation_positive"]),
            f"fold {fold} validation-positive count changed",
        )
        for role in ("train", "validation"):
            subset = block.loc[block["role"].eq(role)]
            _require(
                _membership_sha256(subset["row_id"].tolist())
                == contract[f"{role}_row_id_sha256"],
                f"fold {fold} {role} membership changed",
            )
        # Explicit leakage check, independent of the hashed role assignment.
        train = block.loc[block["role"].eq("train")]
        held = block.loc[block["role"].eq("validation")]
        _require(
            set(train["chain1_sha256"]).isdisjoint(held["chain1_sha256"]),
            f"fold {fold} peptide identity leaked",
        )
        _require(
            set(train["chain2_sha256"]).isdisjoint(held["chain2_sha256"]),
            f"fold {fold} Affibody identity leaked",
        )
    return base, frame


def _cache_indices_for(frame: pd.DataFrame, store: FrozenFeatureStore) -> np.ndarray:
    lookup = store.row_id_to_index
    missing = sorted(set(frame["row_id"]).difference(lookup))
    _require(not missing, f"feature cache misses {len(missing)} canonical pairs")
    return np.asarray([lookup[row_id] for row_id in frame["row_id"]], dtype=np.int64)


def _evaluation_rows(config: Mapping[str, Any]) -> int:
    dataset = config["dataset"]
    if "evaluation_rows" in dataset:
        return int(dataset["evaluation_rows"])
    _require(dataset.get("library") == "LibB", "LibA config must declare evaluation_rows")
    return 119


def _validate_cache_split(
    base: pd.DataFrame,
    store: FrozenFeatureStore,
    config: Mapping[str, Any],
) -> None:
    expected_train = int(config["dataset"]["rows"])
    expected_eval = _evaluation_rows(config)
    _require(
        len(store.row_ids) == expected_train + expected_eval,
        "merged cache total row count changed",
    )
    train_ids = {
        row_id for row_id, split in zip(store.row_ids, store.splits) if split == "train"
    }
    eval_ids = {
        row_id for row_id, split in zip(store.row_ids, store.splits) if split == "eval"
    }
    _require(
        len(train_ids) == expected_train and len(eval_ids) == expected_eval,
        "cache split sizes changed",
    )
    _require(train_ids == set(base["row_id"]), "training-label and cache train memberships differ")
    _require(train_ids.isdisjoint(eval_ids), "cache train/eval row IDs overlap")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: FrozenFeatureDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
        persistent_workers=bool(workers),
    )


def _model_inputs(batch: Mapping[str, object], feature_names: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device=device, dtype=torch.float32, non_blocking=True)
        for name in feature_names
    }


def _predict(
    model: nn.Module,
    loader: DataLoader,
    feature_names: Sequence[str],
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    model.eval()
    row_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(**_model_inputs(batch, feature_names, device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            row_ids.extend(map(str, batch["row_id"]))
            if "label" in batch:
                labels.append(batch["label"].cpu().numpy())
    observed = np.concatenate(labels).astype(int) if labels else None
    return row_ids, np.concatenate(probabilities), observed


def _binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "validation metrics need both classes")
    return {
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    feature_names: Sequence[str],
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    examples = 0
    for batch in loader:
        labels = batch["label"].to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**_model_inputs(batch, feature_names, device))
        loss = class_weighted_binary_cross_entropy(logits, labels, class_weights)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.numel())
        total += float(loss.detach()) * batch_size
        examples += batch_size
    return total / examples


def _model_dataset(
    store: FrozenFeatureStore,
    frame: pd.DataFrame,
    model_name: str,
    labels: bool,
) -> FrozenFeatureDataset:
    return FrozenFeatureDataset(
        store=store,
        cache_indices=_cache_indices_for(frame, store),
        row_ids=frame["row_id"].tolist(),
        required_features=model_feature_names(model_name),
        labels=frame["weak_label"].to_numpy(dtype=int) if labels else None,
    )


def _fit_one_fold(
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    store: FrozenFeatureStore,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame]:
    _set_seed(seed)
    architecture = config["architecture"]
    optimization = config["optimization"]
    model = build_readout(model_name, architecture).to(device)
    feature_names = model_feature_names(model_name)
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        _model_dataset(store, train_frame, model_name, labels=True),
        optimization["batch_size"],
        True,
        seed,
        optimization["num_workers"],
        pin_memory,
    )
    validation_loader = _loader(
        _model_dataset(store, validation_frame, model_name, labels=True),
        optimization["evaluation_batch_size"],
        False,
        seed,
        optimization["num_workers"],
        pin_memory,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    class_weights = class_balanced_weights(
        train_frame["weak_label"].to_numpy(dtype=int)
    ).to(device)
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: tuple[list[str], np.ndarray, np.ndarray | None] | None = None
    history: list[dict[str, Any]] = []
    patience = int(optimization["early_stopping_patience"])
    min_delta = float(optimization["early_stopping_min_delta"])
    stale_epochs = 0
    for epoch in range(1, int(optimization["max_epochs"]) + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, feature_names, class_weights, device
        )
        prediction = _predict(model, validation_loader, feature_names, device)
        if prediction[2] is None:
            raise RuntimeError("validation labels unexpectedly absent")
        metrics = _binary_metrics(prediction[2], prediction[1])
        history.append({"epoch": epoch, "train_weighted_bce": train_loss, **metrics})
        if float(metrics["log_loss"]) < best_loss - min_delta:
            best_loss = float(metrics["log_loss"])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_prediction = prediction
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None or best_prediction is None:
        raise RuntimeError("training never produced a validation checkpoint")
    model.load_state_dict(best_state)
    uids, probabilities, labels = best_prediction
    if labels is None:
        raise RuntimeError("best validation labels unexpectedly absent")
    record = {
        "model": model_name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_metrics": _binary_metrics(labels, probabilities),
        "history": history,
    }
    predictions = pd.DataFrame(
        {
            "row_id": uids,
            "weak_label": labels,
            "probability": probabilities,
        }
    )
    return model, record, predictions


def run_cross_validation(
    base: pd.DataFrame,
    fold_membership: pd.DataFrame,
    store: FrozenFeatureStore,
    config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, int]:
    validation_seed = int(config["validation"]["training_seed"])
    summaries: list[dict[str, Any]] = []
    prediction_blocks: list[pd.DataFrame] = []
    selected_epochs: dict[str, int] = {}
    checkpoints = output_dir / "cv_checkpoints"
    checkpoints.mkdir()
    for model_index, model_name in enumerate(config["models"]):
        model_records: list[dict[str, Any]] = []
        for fold in range(int(config["validation"]["folds"])):
            membership = fold_membership.loc[fold_membership["fold"].eq(fold)]
            train_uids = set(membership.loc[membership["role"].eq("train"), "row_id"])
            validation_uids = set(
                membership.loc[membership["role"].eq("validation"), "row_id"]
            )
            train_frame = base.loc[base["row_id"].isin(train_uids)].reset_index(drop=True)
            validation_frame = base.loc[
                base["row_id"].isin(validation_uids)
            ].reset_index(drop=True)
            fold_seed = validation_seed + 1000 * model_index + fold
            model, record, predictions = _fit_one_fold(
                model_name,
                train_frame,
                validation_frame,
                store,
                config,
                fold_seed,
                device,
            )
            record["fold"] = fold
            record["n_train"] = int(len(train_frame))
            model_records.append(record)
            predictions.insert(0, "fold", fold)
            predictions.insert(0, "model", model_name)
            prediction_blocks.append(predictions)
            torch.save(
                {
                    "schema_version": config["schema_version"],
                    "model": model_name,
                    "fold": fold,
                    "seed": fold_seed,
                    "best_epoch": record["best_epoch"],
                    "state_dict": model.state_dict(),
                },
                checkpoints / f"{model_name}__fold{fold}.pt",
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        epoch_values = [int(record["best_epoch"]) for record in model_records]
        selected_epochs[model_name] = max(1, int(np.median(epoch_values)))
        summaries.append(
            {
                "model": model_name,
                "folds": model_records,
                "selected_final_epochs": selected_epochs[model_name],
                "selection_rule": "integer median of three weak-validation best epochs",
                "mean_best_metrics": {
                    metric: float(
                        np.mean(
                            [record["best_metrics"][metric] for record in model_records]
                        )
                    )
                    for metric in (
                        "log_loss",
                        "brier",
                        "auroc",
                        "average_precision",
                    )
                },
            }
        )
    pd.concat(prediction_blocks, ignore_index=True).to_csv(
        output_dir / "weak_validation_predictions.csv.gz", index=False
    )
    _json_dump(output_dir / "weak_validation_summary.json", summaries)
    _json_dump(output_dir / "selected_epochs.json", selected_epochs)
    return selected_epochs


def _fit_fixed_epochs(
    model_name: str,
    frame: pd.DataFrame,
    store: FrozenFeatureStore,
    config: Mapping[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, list[float]]:
    _set_seed(seed)
    optimization = config["optimization"]
    model = build_readout(model_name, config["architecture"]).to(device)
    loader = _loader(
        _model_dataset(store, frame, model_name, labels=True),
        optimization["batch_size"],
        True,
        seed,
        optimization["num_workers"],
        device.type == "cuda",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    weights = class_balanced_weights(frame["weak_label"].to_numpy(dtype=int)).to(device)
    losses = [
        _train_epoch(
            model, loader, optimizer, model_feature_names(model_name), weights, device
        )
        for _ in range(int(epochs))
    ]
    return model, losses


def _read_prediction_row_ids(
    path: Path | None,
    store: FrozenFeatureStore,
    expected_evaluation_rows: int | None = None,
) -> pd.DataFrame:
    """Load a label-free roster; by default select cache rows marked eval."""

    if path is None:
        row_ids = [
            row_id
            for row_id, split in zip(store.row_ids, store.splits)
            if split == "eval"
        ]
        if expected_evaluation_rows is not None:
            _require(
                len(row_ids) == int(expected_evaluation_rows),
                "default evaluation cache panel size changed",
            )
        return pd.DataFrame({"row_id": row_ids})
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    _require(header == ["row_id"], "prediction roster must contain only row_id")
    frame = pd.read_csv(path, dtype={"row_id": str})
    _require(not frame.empty, "prediction row-ID file is empty")
    _require(not bool(frame["row_id"].duplicated().any()), "duplicate prediction row ID")
    missing = sorted(set(frame["row_id"]).difference(store.row_id_to_index))
    _require(not missing, f"prediction roster has {len(missing)} row IDs absent from cache")
    return frame


def run_final_training(
    base: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    store: FrozenFeatureStore,
    config: Mapping[str, Any],
    selected_epochs: Mapping[str, int],
    output_dir: Path,
    device: torch.device,
) -> None:
    checkpoints = output_dir / "final_checkpoints"
    predictions_dir = output_dir / "blinded_predictions"
    checkpoints.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for model_name in config["models"]:
        _require(model_name in selected_epochs, f"no selected epoch for {model_name}")
        epochs = int(selected_epochs[model_name])
        _require(1 <= epochs <= int(config["optimization"]["max_epochs"]), "selected epoch is out of range")
        for seed in map(int, config["final_training"]["seeds"]):
            started = time.time()
            model, losses = _fit_fixed_epochs(
                model_name, base, store, config, seed, epochs, device
            )
            torch.save(
                {
                    "schema_version": config["schema_version"],
                    "model": model_name,
                    "seed": seed,
                    "epochs": epochs,
                    "weak_membership_sha256": _membership_sha256(base["row_id"]),
                    "state_dict": model.state_dict(),
                },
                checkpoints / f"{model_name}__seed{seed}.pt",
            )
            prediction_dataset = _model_dataset(
                store,
                prediction_frame.assign(weak_label=0),
                model_name,
                labels=False,
            )
            prediction_loader = _loader(
                prediction_dataset,
                config["optimization"]["evaluation_batch_size"],
                False,
                seed,
                config["optimization"]["num_workers"],
                device.type == "cuda",
            )
            row_ids, probabilities, labels = _predict(
                model, prediction_loader, model_feature_names(model_name), device
            )
            if labels is not None:
                raise RuntimeError("prediction roster unexpectedly contained labels")
            output = pd.DataFrame(
                {
                    "eval_row_id": row_ids,
                    "model": model_name,
                    "seed": str(seed),
                    "score": probabilities,
                }
            )
            output.to_csv(
                predictions_dir / f"{model_name}__seed{seed}.csv", index=False
            )
            records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "epochs": epochs,
                    "epoch_weighted_bce": losses,
                    "prediction_rows": int(len(output)),
                    "elapsed_seconds": float(time.time() - started),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _json_dump(output_dir / "final_training_summary.json", records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--training-labels-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="required for training modes; audit is read-only and needs no output",
    )
    parser.add_argument(
        "--mode",
        choices=("audit", "cross_validate", "fit_final", "all"),
        default="audit",
    )
    parser.add_argument(
        "--selected-epochs-json",
        type=Path,
        help="required by fit_final; generated by cross_validate",
    )
    parser.add_argument(
        "--prediction-row-ids",
        type=Path,
        help="optional label-free one-column roster; default predicts cache rows marked eval",
    )
    parser.add_argument(
        "--verify-cache-checksums",
        action="store_true",
        help="stream and verify every large .npy hash (audit mode always does this)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    training_labels_path = args.training_labels.resolve()
    training_manifest_path = args.training_labels_manifest.resolve()
    config = _read_config(config_path)
    base, fold_membership = load_canonical_training_labels(
        training_labels_path, training_manifest_path, config
    )
    required_features = tuple(
        dict.fromkeys(
            feature
            for model_name in config["models"]
            for feature in model_feature_names(model_name)
        )
    )
    store = FrozenFeatureStore.open(
        args.feature_cache,
        required_features,
        verify_all_checksums=bool(args.verify_cache_checksums or args.mode == "audit"),
    )
    _cache_indices_for(base, store)
    _validate_cache_split(base, store, config)
    audit = {
        "schema_version": config["schema_version"],
        "retention_labels_read": False,
        "library": config["dataset"]["library"],
        "weak_rows": int(len(base)),
        "weak_positive": int(base["weak_label"].sum()),
        "weak_negative": int(base["weak_label"].eq(0).sum()),
        "folds": int(config["validation"]["folds"]),
        "split_seed": int(config["validation"]["split_seed"]),
        "cache_rows": int(len(store.row_ids)),
        "cache_train_rows": int(sum(split == "train" for split in store.splits)),
        "cache_eval_rows": int(sum(split == "eval" for split in store.splits)),
        "cache_arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in store.arrays.items()
        },
        "trainable_parameters": parameter_report(config["architecture"], config["models"]),
    }
    if args.mode == "audit":
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    _require(args.output_dir is not None, "training modes require --output-dir")
    output_dir = _validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "config.json")
    shutil.copy2(training_labels_path, output_dir / "training_labels.csv")
    shutil.copy2(training_manifest_path, output_dir / "training_labels_manifest.json")
    for copied in (
        output_dir / "config.json",
        output_dir / "training_labels.csv",
        output_dir / "training_labels_manifest.json",
    ):
        os.chmod(copied, 0o600)
    _json_dump(output_dir / "audit.json", audit)
    device = torch.device(args.device)
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA device requested but unavailable")

    selected_epochs: dict[str, int]
    if args.mode in ("cross_validate", "all"):
        selected_epochs = run_cross_validation(
            base, fold_membership, store, config, output_dir, device
        )
    else:
        _require(args.selected_epochs_json is not None, "fit_final requires --selected-epochs-json")
        with args.selected_epochs_json.open("r", encoding="utf-8") as handle:
            selected_epochs = {str(key): int(value) for key, value in json.load(handle).items()}

    if args.mode in ("fit_final", "all"):
        prediction_frame = _read_prediction_row_ids(
            args.prediction_row_ids,
            store,
            expected_evaluation_rows=_evaluation_rows(config),
        )
        run_final_training(
            base,
            prediction_frame,
            store,
            config,
            selected_epochs,
            output_dir,
            device,
        )


if __name__ == "__main__":
    main()
