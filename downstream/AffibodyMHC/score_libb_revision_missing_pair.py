#!/usr/bin/env python
"""Outcome-blind scoring of the revised LibB AH x LIFTK pair.

This repair command deliberately has no retention-table argument.  It derives
the AH pMHC-side sequence and LIFTK Affibody sequence independently from the
label-free canonical sequence records, extracts frozen MINT layers 5 and 33,
and scores the new pair with three already-fixed models:

* the public primary frozen-MINT logistic readout;
* the weak-selected frozen layer-5 logistic readout; and
* the saved one-epoch LoRA checkpoint.

The two logistic heads were not serialized by their historical runners, so
they are deterministically refit on the unchanged 30,648 weak-label rows.
Before their new scores are accepted, their probabilities must reproduce all
119 archived evaluation probabilities.  The LoRA checkpoint is not refit; it
is loaded for inference and is subjected to the same 119-row parity check.

Only ``pair_uid`` and archived model-score columns are read from historical
prediction CSVs.  Numeric retention and binder-label columns are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import evaluate_mint_multilayer as multilayer_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path
from downstream.AffibodyMHC.extract_mint_multilayer_cache import (
    extract_multilayer_chain_means,
    feature_name,
)


SCHEMA_VERSION = "libb-revision-missing-pair-mint-scores-v1"
LIBRARY = "LibB"
PEPTIDE_CODE = "AH"
AFFIBODY_CODE = "LIFTK"
PRIMARY_LAYER = 33
SELECTED_LAYER = 5
PRIMARY_C = 0.1
SELECTED_LAYER_C = 0.1
EXPECTED_WEAK_ROWS = 30_648
EXPECTED_EVALUATION_ROWS = 119
EXPECTED_PRIMARY_MEMBERSHIP_SHA256 = (
    "1a9a0527c5f9f0e2bbeff6a0ea3d9afaaeadace894d46e73311b049a19f8a393"
)
EXPECTED_LORA_SHA256 = (
    "8a1bcea5b862bde48c1c0d603ddcef8062c188def35edbc64b0f2a1b76848b5d"
)
EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "84a4016365997cd9f0bccb07d746fa8f076ffd8e45aa0cbcf4e50a037161a342"
)
PRIMARY_PARITY_TOLERANCE = 1e-12
LAYER5_PARITY_TOLERANCE = 1e-12
LORA_PARITY_TOLERANCE = 2e-6

DEFAULT_CANONICAL_ROWS = (
    REPO_ROOT / "private_data/derived/esmfold2_libb_canonical_rows_v1/rows.json"
)
DEFAULT_CACHE_ROWS = (
    REPO_ROOT / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv"
)
DEFAULT_PRIMARY_FEATURES = (
    REPO_ROOT
    / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz"
)
DEFAULT_LEGACY_EVALUATION_FEATURES = (
    REPO_ROOT / "private_data/derived/mint_features_v1.npz"
)
DEFAULT_MULTILAYER_FEATURES = (
    REPO_ROOT
    / "private_data/derived/mint_multilayer_v1/merged/"
    "mint_multilayer_chain_mean_features.npz"
)
DEFAULT_PRIMARY_ARCHIVED_PREDICTIONS = (
    REPO_ROOT
    / "private_data/experiments/mint_selection_one_epoch_libb_v1/"
    "retention_predictions.csv"
)
DEFAULT_LAYER5_ARCHIVED_PREDICTIONS = (
    REPO_ROOT
    / "private_data/experiments/mint_multilayer_eval_aggregate_v1/"
    "retention_predictions.csv"
)
DEFAULT_LORA_CHECKPOINT = (
    REPO_ROOT
    / "private_data/experiments/mint_selection_one_epoch_libb_v1/"
    "model_delta_seed20260811_lora_cross.pt"
)
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "checkpoints/mint.ckpt"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "data/esm2_t33_650M_UR50D.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _torch_load(path: Path, map_location: str):
    """Use the safe modern argument when available, while preserving the recorded PyTorch 1.12 environment."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_json(path: Path, payload: Any) -> None:
    _require(not path.exists(), f"output exists; refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a private, label-free score table without overwriting artifacts."""

    _require(not path.exists(), f"output exists; refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_record(path: Path, *, safe_to_hash: bool, columns_read: tuple[str, ...] = ()) -> dict[str, Any]:
    """Record provenance without byte-reading files that also contain outcomes."""

    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
    }
    if safe_to_hash:
        record["sha256"] = sha256_file(path)
    else:
        record["sha256_not_recomputed_reason"] = (
            "file also contains outcome columns; only the explicitly projected columns were read"
        )
        record["columns_read"] = list(columns_read)
    return record


def _designed_codes(chain1: str, chain2: str) -> tuple[str, str]:
    _require(len(chain1) == 270, "pMHC-side chain length changed")
    _require(len(chain2) == 58, "Affibody chain length changed")
    peptide = chain1[-9:]
    return peptide[3:5], "".join(chain2[position - 1] for position in (6, 10, 13, 14, 17))


def derive_missing_pair(path: Path) -> dict[str, str]:
    """Derive each partner independently from label-free canonical records."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(
        payload.get("schema_version") == "esmfold2-libb-canonical-rows-v1",
        "canonical sequence schema changed",
    )
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 30_767, "canonical row count changed")
    chain1_candidates: set[str] = set()
    chain2_candidates: set[str] = set()
    existing_pairs: set[tuple[str, str]] = set()
    for row in rows:
        _require(isinstance(row, dict), "canonical sequence row is malformed")
        chain1 = str(row["chain1_sequence"])
        chain2 = str(row["chain2_sequence"])
        peptide_code, affibody_code = _designed_codes(chain1, chain2)
        if peptide_code == PEPTIDE_CODE:
            chain1_candidates.add(chain1)
        if affibody_code == AFFIBODY_CODE:
            chain2_candidates.add(chain2)
        existing_pairs.add((chain1, chain2))
    _require(len(chain1_candidates) == 1, "AH does not map to exactly one pMHC-side sequence")
    _require(len(chain2_candidates) == 1, "LIFTK does not map to exactly one Affibody sequence")
    chain1 = next(iter(chain1_candidates))
    chain2 = next(iter(chain2_candidates))
    _require((chain1, chain2) not in existing_pairs, "AH x LIFTK was already in the old panel")
    _require(_designed_codes(chain1, chain2) == (PEPTIDE_CODE, AFFIBODY_CODE), "target code mismatch")
    return {
        "chain1_sequence": chain1,
        "chain2_sequence": chain2,
        "chain1_sha256": _sha256_text(chain1),
        "chain2_sha256": _sha256_text(chain2),
        "sequence_pair_sha256": _sha256_text(chain1 + "|" + chain2),
        "opaque_inference_id": "libb-revision-" + _sha256_text(chain1 + "|" + chain2)[:20],
    }


ROW_USECOLS = (
    "row_index",
    "source_kind",
    "library",
    "pair_uid",
    "weak_label",
    "measurement_missing",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)


def load_outcome_blind_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load weak labels and sequences while excluding every retention outcome field."""

    frame = pd.read_csv(
        path,
        usecols=list(ROW_USECOLS),
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(tuple(frame.columns) == ROW_USECOLS, "cache-row projection changed")
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="raise").astype(int)
    weak = frame.loc[frame["source_kind"].eq("weak") & frame["library"].eq(LIBRARY)].copy()
    evaluation = frame.loc[
        frame["source_kind"].eq("retention")
        & frame["library"].eq(LIBRARY)
        & frame["measurement_missing"].eq("0")
    ].copy()
    primary = matched.cached_eval._regime_pool(weak, evaluation, matched.REGIME)
    primary = matched.cached_eval.apply_static_cleaning(primary, matched.CLEANING).reset_index(drop=True)
    primary["weak_label"] = pd.to_numeric(primary["weak_label"], errors="raise").astype(int)
    _require(len(primary) == EXPECTED_WEAK_ROWS, "primary weak-label row count changed")
    _require(
        matched.cached_eval.membership_sha256(primary) == EXPECTED_PRIMARY_MEMBERSHIP_SHA256,
        "primary weak-label membership changed",
    )
    _require(len(evaluation) == EXPECTED_EVALUATION_ROWS, "old evaluation row count changed")
    _require(set(primary["weak_label"]) == {0, 1}, "weak labels are not binary")
    return frame, primary, evaluation.sort_values("pair_uid").reset_index(drop=True)


def _load_named_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(names).difference(archive.files)
        _require(not missing, f"{path} lacks arrays {sorted(missing)}")
        return {name: np.asarray(archive[name]).copy() for name in names}


def _aligned_feature_rows(
    pair_uids: np.ndarray,
    features: np.ndarray,
    requested: pd.Series,
) -> np.ndarray:
    observed = np.asarray(pair_uids).astype(str)
    _require(len(set(observed.tolist())) == len(observed), "feature archive has duplicate pair IDs")
    lookup = {pair_uid: index for index, pair_uid in enumerate(observed)}
    missing = sorted(set(requested.astype(str)).difference(lookup))
    _require(not missing, f"feature archive misses {len(missing)} requested pairs")
    indices = np.asarray([lookup[value] for value in requested.astype(str)], dtype=np.int64)
    output = np.asarray(features[indices], dtype=np.float32)
    _require(output.shape == (len(requested), 2560), "aligned feature shape changed")
    _require(bool(np.isfinite(output).all()), "aligned features are non-finite")
    return output


def fit_fixed_logistic(
    weak_features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    mean, scale = matched.fit_standardizer(weak_features)
    x = matched.standardize(weak_features, mean, scale)
    classifier = matched.fit_logistic(x, labels, float(c_value))
    return mean, scale, classifier


def predict_fixed(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    classifier: Any,
) -> np.ndarray:
    values = matched.standardize(features, mean, scale)
    probability = classifier.predict_proba(values)[:, 1]
    _require(bool(np.isfinite(probability).all()), "logistic probability is non-finite")
    return np.asarray(probability, dtype=float)


def archived_scores(
    path: Path,
    *,
    score_column: str,
    filters: dict[str, object],
) -> pd.DataFrame:
    """Read only identity/model-score columns; outcome columns remain unopened."""

    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"pair_uid", score_column, *filters}
    _require(required.issubset(header), f"archived prediction schema changed: {path}")
    frame = pd.read_csv(
        path,
        usecols=sorted(required),
        dtype={"pair_uid": str},
        float_precision="round_trip",
    )
    for column, expected in filters.items():
        if isinstance(expected, float):
            frame = frame.loc[np.isclose(pd.to_numeric(frame[column]), expected)]
        else:
            frame = frame.loc[frame[column].astype(str).eq(str(expected))]
    output = frame[["pair_uid", score_column]].copy()
    output[score_column] = pd.to_numeric(output[score_column], errors="raise").astype(float)
    _require(len(output) == EXPECTED_EVALUATION_ROWS, "archived model does not have 119 scores")
    _require(not bool(output["pair_uid"].duplicated().any()), "archived score IDs are duplicated")
    return output.sort_values("pair_uid").reset_index(drop=True)


def parity_record(
    pair_uids: pd.Series,
    recomputed: np.ndarray,
    archived: pd.DataFrame,
    score_column: str,
    tolerance: float,
) -> dict[str, Any]:
    current = pd.DataFrame(
        {"pair_uid": pair_uids.astype(str), "recomputed_probability": np.asarray(recomputed, dtype=float)}
    )
    joined = archived.merge(current, on="pair_uid", validate="one_to_one")
    _require(len(joined) == EXPECTED_EVALUATION_ROWS, "parity pair membership differs")
    difference = joined["recomputed_probability"].to_numpy(float) - joined[score_column].to_numpy(float)
    maximum = float(np.max(np.abs(difference)))
    record = {
        "rows": int(len(joined)),
        "max_abs_difference": maximum,
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference * difference))),
        "bitwise_equal_float64": bool(
            np.array_equal(
                joined["recomputed_probability"].to_numpy(float),
                joined[score_column].to_numpy(float),
            )
        ),
        "tolerance": float(tolerance),
        "passed": bool(maximum <= float(tolerance)),
    }
    return record


def _target_frame(target: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_uid": target["opaque_inference_id"],
                "weak_label": 0,
                "chain1_smart_hla_linker_peptide_sequence": target["chain1_sequence"],
                "chain2_affibody_sequence": target["chain2_sequence"],
            }
        ]
    )


def load_lora_model(args: argparse.Namespace, state: dict[str, Any], device: torch.device):
    _require(state.get("library") == LIBRARY, "LoRA checkpoint library changed")
    _require(state.get("arm") == "lora_cross", "LoRA checkpoint arm changed")
    _require(int(state.get("training_seed", -1)) == 20260811, "LoRA seed changed")
    _require(int(state.get("epochs", -1)) == 1, "LoRA epoch count changed")
    _require(float(state.get("selected_C", -1)) == PRIMARY_C, "LoRA selected C changed")
    _require(
        state.get("primary_membership_sha256") == EXPECTED_PRIMARY_MEMBERSHIP_SHA256,
        "LoRA training membership changed",
    )
    _require(
        state.get("base_checkpoint_sha256") == EXPECTED_BASE_CHECKPOINT_SHA256,
        "LoRA base checkpoint identity changed",
    )
    model = matched.MINTSelectionClassifier(
        args.model_config,
        args.base_checkpoint,
        device,
        rank=2,
        alpha=4.0,
        dropout=0.05,
    ).to(device)
    model.set_feature_standardization(
        state["feature_mean"].cpu().numpy(), state["feature_scale"].cpu().numpy()
    )
    model.head.load_state_dict(state["head_state_dict"])
    named = dict(model.named_parameters())
    _require(set(state["adapter_state"]) == {name for name in named if "lora_" in name}, "LoRA tensor names changed")
    with torch.no_grad():
        for name, value in state["adapter_state"].items():
            _require(named[name].shape == value.shape, f"LoRA tensor shape changed: {name}")
            named[name].copy_(value.to(device=device, dtype=named[name].dtype))
    model.eval()
    return model


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required for MINT inference")
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    input_paths = {
        "canonical_rows": args.canonical_rows,
        "cache_rows": args.cache_rows,
        "primary_features": args.primary_features,
        "legacy_evaluation_features": args.legacy_evaluation_features,
        "multilayer_features": args.multilayer_features,
        "primary_archived_predictions": args.primary_archived_predictions,
        "layer5_archived_predictions": args.layer5_archived_predictions,
        "lora_checkpoint": args.lora_checkpoint,
        "base_checkpoint": args.base_checkpoint,
        "model_config": args.model_config,
    }
    for name, path in input_paths.items():
        _require(Path(path).is_file(), f"missing {name}: {path}")
    _require(sha256_file(args.lora_checkpoint) == EXPECTED_LORA_SHA256, "LoRA checkpoint hash changed")
    _require(sha256_file(args.base_checkpoint) == EXPECTED_BASE_CHECKPOINT_SHA256, "base checkpoint hash changed")

    target = derive_missing_pair(args.canonical_rows)
    _, primary, evaluation = load_outcome_blind_rows(args.cache_rows)
    labels = primary["weak_label"].to_numpy(dtype=int)

    # Refit the historical primary frozen layer-33 head on the exact old weak pool.
    primary_cache = _load_named_npz(
        args.primary_features, ("pair_uid", "mint_chain_mean")
    )
    primary_weak_features = _aligned_feature_rows(
        primary_cache["pair_uid"], primary_cache["mint_chain_mean"], primary["pair_uid"]
    )
    legacy = _load_named_npz(
        args.legacy_evaluation_features,
        ("pair_uid", "library", "measurement_missing", "mint_chain_mean"),
    )
    legacy_table = pd.DataFrame(
        {
            "pair_uid": legacy["pair_uid"].astype(str),
            "library": legacy["library"].astype(str),
            "measurement_missing": legacy["measurement_missing"].astype(str),
            "feature_index": np.arange(len(legacy["pair_uid"]), dtype=int),
        }
    )
    legacy_libb = legacy_table.loc[
        legacy_table["library"].eq(LIBRARY) & legacy_table["measurement_missing"].eq("0")
    ].copy()
    legacy_libb = evaluation[["pair_uid"]].merge(legacy_libb, on="pair_uid", validate="one_to_one")
    primary_eval_features = np.asarray(
        legacy["mint_chain_mean"][legacy_libb["feature_index"].to_numpy(int)], dtype=np.float32
    )
    primary_mean, primary_scale, primary_head = fit_fixed_logistic(
        primary_weak_features, labels, PRIMARY_C
    )
    primary_eval_probability = predict_fixed(
        primary_eval_features, primary_mean, primary_scale, primary_head
    )
    primary_archived = archived_scores(
        args.primary_archived_predictions,
        score_column="probability",
        filters={"arm": "frozen_logistic"},
    )
    primary_parity = parity_record(
        legacy_libb["pair_uid"],
        primary_eval_probability,
        primary_archived,
        "probability",
        PRIMARY_PARITY_TOLERANCE,
    )
    primary_parity["exact_archived_model_recovery"] = bool(
        primary_parity["bitwise_equal_float64"]
    )

    # Refit the already weak-selected layer-5 head; no layer/C reselection occurs.
    multilayer = _load_named_npz(
        args.multilayer_features,
        ("pair_uid", feature_name(SELECTED_LAYER)),
    )
    layer5_weak_features = _aligned_feature_rows(
        multilayer["pair_uid"], multilayer[feature_name(SELECTED_LAYER)], primary["pair_uid"]
    )
    layer5_eval_features = _aligned_feature_rows(
        multilayer["pair_uid"], multilayer[feature_name(SELECTED_LAYER)], evaluation["pair_uid"]
    )
    layer5_mean, layer5_scale, layer5_head = fit_fixed_logistic(
        layer5_weak_features, labels, SELECTED_LAYER_C
    )
    layer5_eval_probability = predict_fixed(
        layer5_eval_features, layer5_mean, layer5_scale, layer5_head
    )
    layer5_archived = archived_scores(
        args.layer5_archived_predictions,
        score_column="binder_score",
        filters={
            "library": LIBRARY,
            "model": "weak_selected_frozen_mint_layer",
            "layer": str(SELECTED_LAYER),
            "C": SELECTED_LAYER_C,
        },
    )
    layer5_parity = parity_record(
        evaluation["pair_uid"],
        layer5_eval_probability,
        layer5_archived,
        "binder_score",
        LAYER5_PARITY_TOLERANCE,
    )
    layer5_parity["exact_archived_model_recovery"] = bool(
        layer5_parity["bitwise_equal_float64"]
    )

    # Build one MINT model. Before loading saved LoRA tensors, its zero-delta
    # adapters provide the unchanged base model used for target feature extraction.
    state = _torch_load(args.lora_checkpoint, map_location="cpu")
    model = matched.MINTSelectionClassifier(
        args.model_config,
        args.base_checkpoint,
        device,
        rank=2,
        alpha=4.0,
        dropout=0.05,
    ).to(device)
    model.eval()
    target_loader = matched.make_loader(_target_frame(target), 1, False, 0)
    target_batch = next(iter(target_loader))
    target_chains, target_chain_ids = target_batch[0].to(device), target_batch[1].to(device)
    with torch.inference_mode():
        target_layers = extract_multilayer_chain_means(
            model.wrapper.model,
            target_chains,
            target_chain_ids,
            layers=(SELECTED_LAYER, PRIMARY_LAYER),
        )
    target_layer5 = target_layers[feature_name(SELECTED_LAYER)].float().cpu().numpy()
    target_layer33 = target_layers[feature_name(PRIMARY_LAYER)].float().cpu().numpy()
    primary_target_probability = float(
        predict_fixed(target_layer33, primary_mean, primary_scale, primary_head)[0]
    )
    layer5_target_probability = float(
        predict_fixed(target_layer5, layer5_mean, layer5_scale, layer5_head)[0]
    )

    # Load the saved LoRA/head tensors and run true sequence inference over the
    # old 119 rows plus the new target. No optimizer or training path is called.
    model.set_feature_standardization(
        state["feature_mean"].cpu().numpy(), state["feature_scale"].cpu().numpy()
    )
    model.head.load_state_dict(state["head_state_dict"])
    named = dict(model.named_parameters())
    lora_names = {name for name in named if "lora_" in name}
    _require(set(state["adapter_state"]) == lora_names, "LoRA tensor names changed")
    with torch.no_grad():
        for name, value in state["adapter_state"].items():
            named[name].copy_(value.to(device=device, dtype=named[name].dtype))
    model.eval()
    inference = evaluation[
        [
            "pair_uid",
            "chain1_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence",
        ]
    ].copy()
    inference["weak_label"] = 0
    inference = pd.concat([inference, _target_frame(target)], ignore_index=True)
    loader = matched.make_loader(inference, args.batch_size, False, 0)
    _, lora_probability, _, observed_uids = matched.predict_live(model, loader, device)
    _require(observed_uids == inference["pair_uid"].tolist(), "LoRA inference row order changed")
    lora_old = np.asarray(lora_probability[:-1], dtype=float)
    lora_target_probability = float(lora_probability[-1])
    lora_archived = archived_scores(
        args.primary_archived_predictions,
        score_column="probability",
        filters={"arm": "lora_cross"},
    )
    lora_parity = parity_record(
        evaluation["pair_uid"],
        lora_old,
        lora_archived,
        "probability",
        LORA_PARITY_TOLERANCE,
    )
    _require(lora_parity["passed"], "saved-checkpoint LoRA inference failed 119-row parity")
    lora_parity["exact_archived_model_recovery"] = bool(
        lora_parity["bitwise_equal_float64"]
    )
    torch.cuda.synchronize(device)

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    # Preserve both the historical 119 scores and deterministic replay scores.
    # This permits an evaluation-only downstream check of whether the small
    # replay discrepancy changes any 120-row metric or within-peptide ranking.
    replay = evaluation[["pair_uid"]].copy()
    replay["primary_frozen_replay"] = primary_eval_probability
    replay["layer5_replay"] = layer5_eval_probability
    replay["lora_checkpoint_inference"] = lora_old
    replay = replay.merge(
        primary_archived.rename(columns={"probability": "primary_frozen_archived"}),
        on="pair_uid",
        validate="one_to_one",
    ).merge(
        layer5_archived.rename(columns={"binder_score": "layer5_archived"}),
        on="pair_uid",
        validate="one_to_one",
    ).merge(
        lora_archived.rename(columns={"probability": "lora_archived"}),
        on="pair_uid",
        validate="one_to_one",
    )
    replay = pd.concat(
        [
            replay,
            pd.DataFrame(
                [{
                    "pair_uid": target["opaque_inference_id"],
                    "primary_frozen_replay": primary_target_probability,
                    "layer5_replay": layer5_target_probability,
                    "lora_checkpoint_inference": lora_target_probability,
                    "primary_frozen_archived": np.nan,
                    "layer5_archived": np.nan,
                    "lora_archived": np.nan,
                }]
            ),
        ],
        ignore_index=True,
    )
    replay_path = output_dir / "label_free_predictions.csv"
    _atomic_csv(replay_path, replay)
    result_path = output_dir / "scores.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": "outcome-blind inference for revised LibB AH x LIFTK",
        "retention_outcomes_read": False,
        "training_performed": False,
        "target": {
            "library": LIBRARY,
            "peptide_design_code": PEPTIDE_CODE,
            "affibody_design_code": AFFIBODY_CODE,
            **{key: value for key, value in target.items() if not key.endswith("sequence")},
        },
        "scores": {
            "public_primary_frozen_mint_layer33_deterministic_replay": primary_target_probability,
            "saved_one_epoch_lora_cross": lora_target_probability,
            "weak_selected_frozen_mint_layer5_deterministic_replay": layer5_target_probability,
        },
        "parity_against_all_119_archived_scores": {
            "public_primary_frozen_mint_layer33": primary_parity,
            "saved_one_epoch_lora_cross": lora_parity,
            "weak_selected_frozen_mint_layer5": layer5_parity,
        },
        "fixed_contract": {
            "weak_rows": EXPECTED_WEAK_ROWS,
            "weak_membership_sha256": EXPECTED_PRIMARY_MEMBERSHIP_SHA256,
            "primary_layer": PRIMARY_LAYER,
            "primary_C": PRIMARY_C,
            "selected_intermediate_layer": SELECTED_LAYER,
            "selected_intermediate_C": SELECTED_LAYER_C,
            "lora_training_seed": 20260811,
            "lora_epochs": 1,
            "model_or_hyperparameter_selection_changed": False,
            "replay_interpretation": (
                "Frozen logistic scores are exact archived-model scores only when their "
                "119-row bitwise parity flag is true; otherwise they are deterministic "
                "same-data/same-settings replays with the reported discrepancy."
            ),
        },
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "sources": {
            name: _source_record(
                Path(path),
                safe_to_hash=name
                not in {
                    "cache_rows",
                    "primary_archived_predictions",
                    "layer5_archived_predictions",
                },
                columns_read=(
                    ROW_USECOLS
                    if name == "cache_rows"
                    else (
                        ("pair_uid", "arm", "probability")
                        if name == "primary_archived_predictions"
                        else (
                            "pair_uid",
                            "library",
                            "model",
                            "layer",
                            "C",
                            "binder_score",
                        )
                    )
                ),
            )
            for name, path in input_paths.items()
        },
        "command": " ".join(map(str, sys.argv)),
    }
    _atomic_json(result_path, result)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "retention_outcomes_read": False,
        "training_performed": False,
        "scores_file": result_path.name,
        "scores_sha256": sha256_file(result_path),
        "label_free_predictions_file": replay_path.name,
        "label_free_predictions_sha256": sha256_file(replay_path),
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-rows", type=Path, default=DEFAULT_CANONICAL_ROWS)
    parser.add_argument("--cache-rows", type=Path, default=DEFAULT_CACHE_ROWS)
    parser.add_argument("--primary-features", type=Path, default=DEFAULT_PRIMARY_FEATURES)
    parser.add_argument(
        "--legacy-evaluation-features", type=Path, default=DEFAULT_LEGACY_EVALUATION_FEATURES
    )
    parser.add_argument("--multilayer-features", type=Path, default=DEFAULT_MULTILAYER_FEATURES)
    parser.add_argument(
        "--primary-archived-predictions", type=Path, default=DEFAULT_PRIMARY_ARCHIVED_PREDICTIONS
    )
    parser.add_argument(
        "--layer5-archived-predictions", type=Path, default=DEFAULT_LAYER5_ARCHIVED_PREDICTIONS
    )
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
