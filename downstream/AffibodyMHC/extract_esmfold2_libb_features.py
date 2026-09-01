#!/usr/bin/env python3
"""Extract label-free ESMFold2 trunk features for the canonical LibB rows.

The input is a separately prepared, label-free JSON manifest.  This program
refuses manifests containing retention, binder, weak-label, enrichment, or
selection-count fields; it never opens either training-label or evaluation-
label files.  The fixed canonical contract is 30,648 training rows plus 119
evaluation rows under the strict identity-cold split.

Each process owns one persistent ESMFold2 model and extracts rows whose
``row_index % num_shards == shard_index``.  Full pair tensors are never written.
Only the following interface tensors are serialized:

* categorical distogram probabilities, ``[9, 58, 64]`` float16;
* direction-symmetrized pair states, ``[9, 58, 256]`` float16;
* peptide-followed-by-Affibody single inputs, ``[67, 451]`` float16.

The 64 distogram categories are deliberately not converted to Angstroms: the
public ``biohub/ESMFold2-hf`` checkpoint does not publish calibrated bin edges.
Chunks are written through temporary files and atomically renamed.  Existing
chunks are checksummed and fully validated before being accepted for resume.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import EsmFold2Model

# Permit both ``python -m downstream...`` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.esmfold2_multichain_features import (
    prepare_multichain_trunk_inputs,
)
from downstream.AffibodyMHC.extract_esmfold2_libb_pilot import (
    DISTOGRAM_BIN_SEMANTICS,
    EXPECTED_CHAIN_LENGTHS,
    PEPTIDE_LENGTH,
    _derive_interface_indices,
    _set_common_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPO_ROOT / "private_data"
OUTPUT_PARENT = PRIVATE_ROOT / "derived"
OUTPUT_PREFIX = "esmfold2_libb_features_"
SCHEMA_VERSION = "esmfold2-libb-feature-shard-v1"
ROW_MANIFEST_SCHEMA = "esmfold2-libb-canonical-rows-v1"
EXPECTED_CHECKPOINT_REVISION = "bce015efb23b5dc604842d0ab5c2bbb02c7bd3ee"
EXPECTED_TRANSFORMERS_VERSION = "5.16.1"
EXPECTED_TRAIN_ROWS = 30_648
EXPECTED_EVAL_ROWS = 119
EXPECTED_TOTAL_ROWS = EXPECTED_TRAIN_ROWS + EXPECTED_EVAL_ROWS
EXPECTED_UNIQUE_COUNTS = {
    "train": {"chain1": 214, "chain2": 23_081},
    "eval": {"chain1": 12, "chain2": 10},
}
EXPECTED_OUTPUT_SHAPES = {
    "distogram_logits": (1, 328, 328, 64),
    "pair_states": (1, 328, 328, 256),
    "single_inputs": (1, 328, 451),
}
SAVED_FEATURE_SPECS = {
    "distogram_probabilities": ((PEPTIDE_LENGTH, 58, 64), np.dtype(np.float16)),
    "pair_states_symmetric": ((PEPTIDE_LENGTH, 58, 256), np.dtype(np.float16)),
    "single_inputs": ((PEPTIDE_LENGTH + 58, 451), np.dtype(np.float16)),
}
CHUNK_SIZE = 64
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
ROUND_COUNT_PATTERN = re.compile(r"^r0*(?:1|9|10)(?:_.*)?count$")
ALLOWED_ROW_KEYS = {
    "row_index",
    "row_id",
    "split",
    "library",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
    "peptide_design_code",
    "affibody_design_code",
}
NPZ_KEYS = {
    "row_index",
    "row_id",
    "split",
    *SAVED_FEATURE_SPECS,
}


@dataclass(frozen=True)
class CanonicalRow:
    row_index: int
    row_id: str
    split: str
    chain1_sequence: str
    chain2_sequence: str
    sequence_pair_sha256: str


@dataclass(frozen=True)
class DatasetExpectations:
    train_rows: int = EXPECTED_TRAIN_ROWS
    eval_rows: int = EXPECTED_EVAL_ROWS
    train_chain1: int = EXPECTED_UNIQUE_COUNTS["train"]["chain1"]
    train_chain2: int = EXPECTED_UNIQUE_COUNTS["train"]["chain2"]
    eval_chain1: int = EXPECTED_UNIQUE_COUNTS["eval"]["chain1"]
    eval_chain2: int = EXPECTED_UNIQUE_COUNTS["eval"]["chain2"]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--row-manifest",
        type=Path,
        required=True,
        help="Extractor-facing label-free JSON, normally <builder-output>/rows.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz_dump(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            # Intentionally uncompressed: the production bottleneck is GPU
            # inference, and compression would add avoidable CPU latency.
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _is_label_like_field(name: str) -> bool:
    normalized = _normalize_field_name(name)
    tokens = set(normalized.split("_"))
    if any(fragment in normalized for fragment in ("retention", "binder", "weak_label")):
        return True
    if tokens.intersection({"label", "labels", "target", "targets", "positive", "positives", "negative", "negatives"}):
        return True
    if normalized in {
        "label",
        "labels",
        "target",
        "targets",
        "positive",
        "positives",
        "negative",
        "negatives",
        "class_label",
        "enrichment",
        "selection_count",
        "pooled_count",
        "pooled_r009_r010_count",
    }:
        return True
    if normalized.startswith("target_") or normalized.endswith("_label"):
        return True
    return bool(ROUND_COUNT_PATTERN.fullmatch(normalized))


def _reject_label_like_keys(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _is_label_like_field(key):
                raise ValueError(
                    f"label-like field {key!r} is forbidden in extraction manifest at {location}"
                )
            _reject_label_like_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_label_like_keys(child, f"{location}[{index}]")


def _validate_private_output_root(path: Path) -> Path:
    expected_parent = OUTPUT_PARENT.resolve()
    # ``absolute`` normalizes ``..`` without following the final component's
    # symlink.  Both lexical and resolved parents are checked so a symlink
    # cannot redirect a seemingly valid output elsewhere.
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.parent != expected_parent:
        raise ValueError(
            f"output must be one dedicated direct child of {expected_parent}"
        )
    if not absolute.name.startswith(OUTPUT_PREFIX) or absolute.name == OUTPUT_PREFIX:
        raise ValueError(f"output directory name must begin with {OUTPUT_PREFIX!r}")
    if absolute.is_symlink():
        raise ValueError("output directory must not be a symlink")
    resolved = absolute.resolve()
    if resolved.parent != expected_parent or resolved == expected_parent:
        raise ValueError("resolved output path escapes the dedicated derived directory")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("output path exists but is not a directory")
    return resolved


def _ensure_private_directory(path: Path) -> None:
    """Create one run-owned directory, or validate it without chmod side effects."""

    if path.is_symlink():
        raise ValueError(f"directory must not be a symlink: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"path exists but is not a directory: {path}")
        mode = path.stat().st_mode & 0o777
        if mode != 0o700:
            raise PermissionError(
                f"existing private output directory must already be mode 0700: {path} ({mode:04o})"
            )
        return
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        # Concurrent shards may race only while creating their shared run
        # root.  Validate the winner exactly as an existing path; never chmod
        # it on behalf of this process.
        _ensure_private_directory(path)
        return
    # Protect against a permissive or unusual process umask.
    os.chmod(path, 0o700)


def _open_private_lock(path: Path):
    """Open a private lock file without following a pre-existing symlink."""

    if path.is_symlink():
        raise ValueError(f"lock path must not be a symlink: {path}")
    descriptor = os.open(
        str(path),
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    mode = os.fstat(descriptor).st_mode & 0o777
    if mode != 0o600:
        os.close(descriptor)
        raise PermissionError(f"lock must be mode 0600, found {mode:04o}: {path}")
    return os.fdopen(descriptor, "a+")


def _validate_row_mapping(raw: Mapping[str, Any]) -> CanonicalRow:
    unknown = set(raw).difference(ALLOWED_ROW_KEYS)
    required = {
        "row_index",
        "row_id",
        "split",
        "chain1_sequence",
        "chain2_sequence",
        "sequence_pair_sha256",
    }
    missing = required.difference(raw)
    if missing or unknown:
        raise ValueError(
            f"row schema mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    try:
        row_index = int(raw["row_index"])
    except (TypeError, ValueError) as exc:
        raise ValueError("row_index must be an integer") from exc
    if isinstance(raw["row_index"], bool) or row_index < 0:
        raise ValueError("row_index must be a non-negative integer")
    row_id = str(raw["row_id"])
    if not row_id or len(row_id) > 256:
        raise ValueError(f"row {row_index}: row_id must contain 1..256 characters")
    split = str(raw["split"])
    if split not in {"train", "eval"}:
        raise ValueError(f"row {row_index}: split must be 'train' or 'eval'")
    if "library" in raw and raw["library"] != "LibB":
        raise ValueError(f"row {row_index}: expected library LibB")

    chain1 = str(raw["chain1_sequence"])
    chain2 = str(raw["chain2_sequence"])
    if (len(chain1), len(chain2)) != EXPECTED_CHAIN_LENGTHS:
        raise ValueError(
            f"row {row_index}: expected chain lengths {EXPECTED_CHAIN_LENGTHS}, "
            f"found {(len(chain1), len(chain2))}"
        )
    if not AA_PATTERN.fullmatch(chain1) or not AA_PATTERN.fullmatch(chain2):
        raise ValueError(f"row {row_index}: sequences must use the 20 standard amino acids")
    pair_sha256 = str(raw["sequence_pair_sha256"])
    observed_sha256 = _sha256_text(chain1 + "|" + chain2)
    if pair_sha256 != observed_sha256:
        raise ValueError(f"row {row_index}: sequence_pair_sha256 mismatch")

    peptide_code = raw.get("peptide_design_code")
    if peptide_code is not None:
        peptide_code = str(peptide_code)
        if len(peptide_code) != 2 or not chain1.endswith(
            "SLL" + peptide_code + "ITQV"
        ):
            raise ValueError(f"row {row_index}: peptide design code mapping mismatch")
    affibody_code = raw.get("affibody_design_code")
    if affibody_code is not None:
        affibody_code = str(affibody_code)
        observed = "".join(chain2[position - 1] for position in (6, 10, 13, 14, 17))
        if len(affibody_code) != 5 or affibody_code != observed:
            raise ValueError(f"row {row_index}: Affibody design code mapping mismatch")

    return CanonicalRow(
        row_index=row_index,
        row_id=row_id,
        split=split,
        chain1_sequence=chain1,
        chain2_sequence=chain2,
        sequence_pair_sha256=pair_sha256,
    )


def _validate_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    expectations: DatasetExpectations = DatasetExpectations(),
) -> list[CanonicalRow]:
    expected_total = expectations.train_rows + expectations.eval_rows
    if len(raw_rows) != expected_total:
        raise ValueError(f"expected {expected_total} canonical rows, found {len(raw_rows)}")
    rows = [_validate_row_mapping(raw) for raw in raw_rows]
    rows.sort(key=lambda row: row.row_index)
    observed_indices = [row.row_index for row in rows]
    if observed_indices != list(range(expected_total)):
        raise ValueError("row_index must be a unique contiguous range starting at zero")
    row_ids = [row.row_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    pair_hashes = [row.sequence_pair_sha256 for row in rows]
    if len(set(pair_hashes)) != len(pair_hashes):
        raise ValueError("sequence-identical pairs must not be duplicated")

    by_split = {
        split: [row for row in rows if row.split == split] for split in ("train", "eval")
    }
    expected_split_rows = {
        "train": expectations.train_rows,
        "eval": expectations.eval_rows,
    }
    expected_unique = {
        "train": {
            "chain1": expectations.train_chain1,
            "chain2": expectations.train_chain2,
        },
        "eval": {
            "chain1": expectations.eval_chain1,
            "chain2": expectations.eval_chain2,
        },
    }
    for split, split_rows in by_split.items():
        if len(split_rows) != expected_split_rows[split]:
            raise ValueError(
                f"expected {expected_split_rows[split]} {split} rows, found {len(split_rows)}"
            )
        observed_unique = {
            "chain1": len({row.chain1_sequence for row in split_rows}),
            "chain2": len({row.chain2_sequence for row in split_rows}),
        }
        if observed_unique != expected_unique[split]:
            raise ValueError(
                f"{split} unique-chain counts changed: expected {expected_unique[split]}, "
                f"found {observed_unique}"
            )

    train_chain1 = {row.chain1_sequence for row in by_split["train"]}
    eval_chain1 = {row.chain1_sequence for row in by_split["eval"]}
    train_chain2 = {row.chain2_sequence for row in by_split["train"]}
    eval_chain2 = {row.chain2_sequence for row in by_split["eval"]}
    if train_chain1.intersection(eval_chain1):
        raise ValueError("strict split violated: an evaluation chain1 occurs in training")
    if train_chain2.intersection(eval_chain2):
        raise ValueError("strict split violated: an evaluation Affibody occurs in training")
    return rows


def _load_row_manifest(
    path: Path,
    expectations: DatasetExpectations = DatasetExpectations(),
) -> tuple[dict[str, Any], list[CanonicalRow]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("row manifest must be one JSON object")
    _reject_label_like_keys(payload)
    if payload.get("schema_version") != ROW_MANIFEST_SCHEMA:
        raise ValueError(
            f"expected row-manifest schema {ROW_MANIFEST_SCHEMA!r}, "
            f"found {payload.get('schema_version')!r}"
        )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("row manifest must contain a rows list")
    if not all(isinstance(row, dict) for row in raw_rows):
        raise ValueError("every row-manifest entry must be an object")
    return payload, _validate_rows(raw_rows, expectations)


def _partition_rows(
    rows: Sequence[CanonicalRow], shard_index: int, num_shards: int
) -> list[CanonicalRow]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    selected = [row for row in rows if row.row_index % num_shards == shard_index]
    expected = sum(
        1 for row_index in range(len(rows)) if row_index % num_shards == shard_index
    )
    if len(selected) != expected:
        raise AssertionError("modulo partition count mismatch")
    return selected


def _chunks(rows: Sequence[CanonicalRow], size: int = CHUNK_SIZE) -> Iterable[list[CanonicalRow]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _validate_output_shapes(output: Any) -> None:
    observed = {
        name: tuple(getattr(output, name).shape) for name in EXPECTED_OUTPUT_SHAPES
    }
    if observed != EXPECTED_OUTPUT_SHAPES:
        raise ValueError(f"unexpected ESMFold2 output shapes: {observed}")


def _to_float16_numpy(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.cpu().numpy().astype(np.float16, copy=False)


def _compact_feature_bundle(
    output: Any, peptide_indices: torch.Tensor, affibody_indices: torch.Tensor
) -> dict[str, np.ndarray]:
    logits = output.distogram_logits[0][peptide_indices][:, affibody_indices, :]
    probabilities = torch.softmax(logits.float(), dim=-1)
    pair_ab = output.pair_states[0][peptide_indices][:, affibody_indices, :]
    pair_ba = (
        output.pair_states[0][affibody_indices][:, peptide_indices, :]
        .transpose(0, 1)
        .contiguous()
    )
    pair_symmetric = 0.5 * (pair_ab + pair_ba)
    single_inputs = torch.cat(
        (
            output.single_inputs[0, peptide_indices, :],
            output.single_inputs[0, affibody_indices, :],
        ),
        dim=0,
    )
    features = {
        "distogram_probabilities": _to_float16_numpy(probabilities),
        "pair_states_symmetric": _to_float16_numpy(pair_symmetric),
        "single_inputs": _to_float16_numpy(single_inputs),
    }
    _validate_one_feature_bundle(features)
    return features


def _validate_one_feature_bundle(features: Mapping[str, np.ndarray]) -> None:
    if set(features) != set(SAVED_FEATURE_SPECS):
        raise ValueError(f"unexpected saved features: {sorted(features)}")
    for name, (shape, dtype) in SAVED_FEATURE_SPECS.items():
        value = np.asarray(features[name])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} must have shape {shape} and dtype {dtype}, "
                f"found {value.shape}/{value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    probabilities = np.asarray(features["distogram_probabilities"], dtype=np.float32)
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("distogram probabilities fall outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=2e-3, rtol=0.0):
        raise ValueError("distogram categories do not sum to one")


def _extract_one(
    model: Any,
    row: CanonicalRow,
    device: torch.device,
    seed: int,
    prepare_fn: Callable[..., Any] = prepare_multichain_trunk_inputs,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    preparation_start = time.perf_counter()
    prepared = prepare_fn(
        (row.chain1_sequence, row.chain2_sequence), device=device
    )
    preparation_seconds = time.perf_counter() - preparation_start
    peptide_indices, affibody_indices = _derive_interface_indices(
        prepared.forward_kwargs, row.chain1_sequence, row.chain2_sequence
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    _set_common_seed(seed)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        output = model(**prepared.forward_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    _validate_output_shapes(output)
    features = _compact_feature_bundle(output, peptide_indices, affibody_indices)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    timing = {
        "row_index": row.row_index,
        "row_id": row.row_id,
        "sequence_pair_sha256": row.sequence_pair_sha256,
        "preparation_seconds": preparation_seconds,
        "forward_seconds": forward_seconds,
        "peak_gpu_memory_bytes": peak_bytes,
    }
    del output, prepared
    return features, timing


def _stack_chunk_features(
    rows: Sequence[CanonicalRow], bundles: Sequence[Mapping[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    if not rows or len(rows) != len(bundles):
        raise ValueError("chunk rows/features must have one non-empty aligned entry")
    for bundle in bundles:
        _validate_one_feature_bundle(bundle)
    arrays: dict[str, np.ndarray] = {
        "row_index": np.asarray([row.row_index for row in rows], dtype=np.int64),
        "row_id": np.asarray([row.row_id for row in rows], dtype=np.str_),
        "split": np.asarray([row.split for row in rows], dtype=np.str_),
    }
    for name in SAVED_FEATURE_SPECS:
        arrays[name] = np.stack([bundle[name] for bundle in bundles], axis=0)
    return arrays


def _validate_chunk_arrays(
    arrays: Mapping[str, np.ndarray], expected_rows: Sequence[CanonicalRow]
) -> None:
    if set(arrays) != NPZ_KEYS:
        raise ValueError(f"chunk NPZ keys changed: {sorted(arrays)}")
    count = len(expected_rows)
    row_index = np.asarray(arrays["row_index"])
    row_id = np.asarray(arrays["row_id"])
    split = np.asarray(arrays["split"])
    if row_index.dtype != np.int64 or row_index.shape != (count,):
        raise ValueError("chunk row_index schema mismatch")
    if row_id.dtype.kind != "U" or row_id.shape != (count,):
        raise ValueError("chunk row_id schema mismatch")
    if split.dtype.kind != "U" or split.shape != (count,):
        raise ValueError("chunk split schema mismatch")
    if row_index.tolist() != [row.row_index for row in expected_rows]:
        raise ValueError("chunk row_index values do not match the canonical shard")
    if row_id.tolist() != [row.row_id for row in expected_rows]:
        raise ValueError("chunk row_id values do not match the canonical shard")
    if split.tolist() != [row.split for row in expected_rows]:
        raise ValueError("chunk split values do not match the canonical shard")
    for name, (shape, dtype) in SAVED_FEATURE_SPECS.items():
        value = np.asarray(arrays[name])
        expected_shape = (count, *shape)
        if value.shape != expected_shape or value.dtype != dtype:
            raise ValueError(
                f"chunk {name} expected {expected_shape}/{dtype}, "
                f"found {value.shape}/{value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"chunk {name} contains non-finite values")
    probabilities = np.asarray(arrays["distogram_probabilities"], dtype=np.float32)
    if not np.allclose(probabilities.sum(axis=-1), 1.0, atol=2e-3, rtol=0.0):
        raise ValueError("chunk distogram probabilities do not sum to one")


def _write_chunk(
    artifact_path: Path,
    metadata_path: Path,
    expected_rows: Sequence[CanonicalRow],
    bundles: Sequence[Mapping[str, np.ndarray]],
    timings: Sequence[Mapping[str, Any]],
    run_contract_sha256: str,
    shard_index: int,
    chunk_index: int,
) -> dict[str, Any]:
    if artifact_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite an existing chunk artifact")
    arrays = _stack_chunk_features(expected_rows, bundles)
    _validate_chunk_arrays(arrays, expected_rows)
    _atomic_npz_dump(artifact_path, arrays)
    artifact_sha256 = _sha256_file(artifact_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "run_contract_sha256": run_contract_sha256,
        "shard_index": shard_index,
        "chunk_index": chunk_index,
        "row_count": len(expected_rows),
        "row_indices": [row.row_index for row in expected_rows],
        "row_ids": [row.row_id for row in expected_rows],
        "sequence_pair_sha256": [row.sequence_pair_sha256 for row in expected_rows],
        "timings": list(timings),
        "artifact": artifact_path.name,
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "feature_shapes_per_row": {
            name: list(spec[0]) for name, spec in SAVED_FEATURE_SPECS.items()
        },
        "feature_dtypes": {
            name: str(spec[1]) for name, spec in SAVED_FEATURE_SPECS.items()
        },
    }
    _atomic_json_dump(metadata_path, payload)
    return payload


def _load_validated_chunk(
    artifact_path: Path,
    metadata_path: Path,
    expected_rows: Sequence[CanonicalRow],
    run_contract_sha256: str,
    shard_index: int,
    chunk_index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if artifact_path.exists() != metadata_path.exists():
        raise RuntimeError(
            f"incomplete chunk {chunk_index}: artifact/metadata presence differs"
        )
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"chunk {chunk_index} does not exist")
    metadata = _read_json(metadata_path)
    required_pairs = {
        "schema_version": SCHEMA_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "shard_index": shard_index,
        "chunk_index": chunk_index,
        "row_count": len(expected_rows),
        "row_indices": [row.row_index for row in expected_rows],
        "row_ids": [row.row_id for row in expected_rows],
        "sequence_pair_sha256": [row.sequence_pair_sha256 for row in expected_rows],
        "artifact": artifact_path.name,
        "artifact_bytes": artifact_path.stat().st_size,
    }
    for name, expected in required_pairs.items():
        if metadata.get(name) != expected:
            raise ValueError(f"chunk {chunk_index} metadata mismatch for {name}")
    observed_sha256 = _sha256_file(artifact_path)
    if metadata.get("artifact_sha256") != observed_sha256:
        raise ValueError(f"chunk {chunk_index} checksum mismatch")
    timings = metadata.get("timings")
    if not isinstance(timings, list) or len(timings) != len(expected_rows):
        raise ValueError(f"chunk {chunk_index} timing schema mismatch")
    for expected_row, timing in zip(expected_rows, timings):
        if timing.get("row_index") != expected_row.row_index or timing.get(
            "row_id"
        ) != expected_row.row_id:
            raise ValueError(f"chunk {chunk_index} timing row mismatch")
        for field in ("preparation_seconds", "forward_seconds"):
            value = timing.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"chunk {chunk_index} invalid {field}")
    with np.load(artifact_path, allow_pickle=False) as handle:
        arrays = {name: handle[name] for name in handle.files}
    _validate_chunk_arrays(arrays, expected_rows)
    return metadata, arrays


def _validate_chunk(
    artifact_path: Path,
    metadata_path: Path,
    expected_rows: Sequence[CanonicalRow],
    run_contract_sha256: str,
    shard_index: int,
    chunk_index: int,
) -> dict[str, Any]:
    metadata, _ = _load_validated_chunk(
        artifact_path,
        metadata_path,
        expected_rows,
        run_contract_sha256,
        shard_index,
        chunk_index,
    )
    return metadata


def _checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("checkpoint lacks config.json or safetensors index")
    tree_dir = checkpoint / ".cache" / "huggingface" / "trees"
    tree_paths = sorted(tree_dir.glob("*.json")) if tree_dir.is_dir() else []
    if len(tree_paths) != 1 or tree_paths[0].stem != EXPECTED_CHECKPOINT_REVISION:
        raise ValueError(
            "checkpoint must retain exactly the pinned Hugging Face tree manifest "
            f"for revision {EXPECTED_CHECKPOINT_REVISION}"
        )
    tree = _read_json(tree_paths[0])
    files = tree.get("files", {})
    weight_records = []
    for index in range(1, 7):
        name = f"model-{index:05d}-of-00006.safetensors"
        path = checkpoint / name
        record = files.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise FileNotFoundError(f"checkpoint shard missing from disk/tree: {name}")
        expected_size = int(record.get("lfs_size", -1))
        if path.stat().st_size != expected_size:
            raise ValueError(f"checkpoint shard size mismatch: {name}")
        weight_records.append(
            {
                "name": name,
                "bytes": expected_size,
                "lfs_sha256": record.get("lfs_sha256"),
            }
        )
    config = _read_json(config_path)
    observed_config = {
        "num_loops": config.get("num_loops"),
        "folding_trunk_num_hidden_layers": config.get(
            "folding_trunk_num_hidden_layers"
        ),
        "pairwise_hidden_size": config.get("pairwise_hidden_size"),
        "single_inputs_size": config.get("single_inputs_size"),
        "distogram_bins": config.get("structure_head", {}).get(
            "num_distogram_bins"
        ),
        "esmc_hidden_size": config.get("esmc_config", {}).get("hidden_size"),
        "esmc_num_hidden_layers": config.get("esmc_config", {}).get(
            "num_hidden_layers"
        ),
    }
    expected_config = {
        "num_loops": 3,
        "folding_trunk_num_hidden_layers": 48,
        "pairwise_hidden_size": 256,
        "single_inputs_size": 451,
        "distogram_bins": 64,
        "esmc_hidden_size": 2560,
        "esmc_num_hidden_layers": 80,
    }
    if observed_config != expected_config:
        raise ValueError(
            f"ESMFold2 checkpoint architecture changed: {observed_config}"
        )
    return {
        "path": str(checkpoint),
        "revision": EXPECTED_CHECKPOINT_REVISION,
        "tree_manifest_sha256": _sha256_file(tree_paths[0]),
        "config_sha256": _sha256_file(config_path),
        "safetensors_index_sha256": _sha256_file(index_path),
        "weight_shards": weight_records,
        "architecture": expected_config,
    }


def _configure_determinism() -> str:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "set CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before launching Python"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return workspace


def _build_run_contract(
    checkpoint_contract: Mapping[str, Any],
    row_manifest: Path,
    num_shards: int,
    seed: int,
    cublas_workspace_config: str,
) -> dict[str, Any]:
    source_files = {
        "extractor": Path(__file__).resolve(),
        "pilot_helpers": REPO_ROOT
        / "downstream/AffibodyMHC/extract_esmfold2_libb_pilot.py",
        "multichain_utility": REPO_ROOT
        / "downstream/AffibodyMHC/esmfold2_multichain_features.py",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "row_manifest": {
            "path": str(row_manifest),
            "sha256": _sha256_file(row_manifest),
            "schema_version": ROW_MANIFEST_SCHEMA,
            "row_count": EXPECTED_TOTAL_ROWS,
            "train_rows": EXPECTED_TRAIN_ROWS,
            "eval_rows": EXPECTED_EVAL_ROWS,
        },
        "checkpoint": dict(checkpoint_contract),
        "source_sha256": {
            name: _sha256_file(path) for name, path in source_files.items()
        },
        "feature_contract": {
            "distogram_probabilities": {
                "shape": [9, 58, 64],
                "dtype": "float16",
                "semantics": DISTOGRAM_BIN_SEMANTICS,
                "physical_distance_conversion": None,
            },
            "pair_states_symmetric": {
                "shape": [9, 58, 256],
                "dtype": "float16",
                "definition": "0.5 * (peptide_to_affibody + aligned_affibody_to_peptide)",
            },
            "single_inputs": {
                "shape": [67, 451],
                "dtype": "float16",
                "residue_order": "nine peptide residues followed by 58 Affibody residues",
            },
        },
        "settings": {
            "dtype": "torch.bfloat16",
            "attention_implementation": "sdpa",
            "use_kernels": False,
            "coordinate_diffusion": False,
            "checkpoint_default_num_loops": 3,
            "actual_trunk_steps": 4,
            "batch_size": 1,
            "chunk_size": CHUNK_SIZE,
            "num_shards": num_shards,
            "partition": "row_index modulo num_shards",
            "common_seed_reset_before_every_forward": seed,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": cublas_workspace_config,
            "allow_tf32": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "supervision_fields_read": [],
    }


def _ensure_contract(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        observed = _read_json(path)
        if observed != payload:
            raise ValueError(f"existing run contract does not match: {path}")
    else:
        _atomic_json_dump(path, payload)


def _assert_no_temporary_files(directory: Path) -> None:
    leftovers = sorted(path.name for path in directory.glob(".*.tmp-*"))
    if leftovers:
        raise RuntimeError(
            "temporary files from an interrupted write require review: "
            + ", ".join(leftovers)
        )


def _load_model(checkpoint: Path, device: torch.device) -> tuple[EsmFold2Model, float]:
    started = time.perf_counter()
    model = EsmFold2Model.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_kernels=False,
    )
    model.eval().to(device)
    torch.cuda.synchronize(device)
    return model, time.perf_counter() - started


def _run(args: argparse.Namespace) -> None:
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"production extractor requires transformers=={EXPECTED_TRANSFORMERS_VERSION}, "
            f"found {transformers.__version__}"
        )
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    workspace = _configure_determinism()
    checkpoint = args.checkpoint.resolve()
    row_manifest = args.row_manifest.resolve()
    output_root = _validate_private_output_root(args.output_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not row_manifest.is_file():
        raise FileNotFoundError(row_manifest)

    # The JSON is the only biological-data input opened by the extractor.
    _, rows = _load_row_manifest(row_manifest)
    shard_rows = _partition_rows(rows, args.shard_index, args.num_shards)
    checkpoint_info = _checkpoint_contract(checkpoint)
    run_contract = _build_run_contract(
        checkpoint_contract=checkpoint_info,
        row_manifest=row_manifest,
        num_shards=args.num_shards,
        seed=args.seed,
        cublas_workspace_config=workspace,
    )
    run_contract_sha256 = _canonical_json_sha256(run_contract)

    if not OUTPUT_PARENT.is_dir() or OUTPUT_PARENT.is_symlink():
        raise RuntimeError(
            f"expected a real pre-existing private derived directory: {OUTPUT_PARENT}"
        )
    _ensure_private_directory(output_root)
    shard_dir = output_root / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    _ensure_private_directory(shard_dir)
    chunks_dir = shard_dir / "chunks"
    _ensure_private_directory(chunks_dir)

    lock_handle = _open_private_lock(shard_dir / ".lock")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns shard {args.shard_index}") from exc
        _assert_no_temporary_files(shard_dir)
        _assert_no_temporary_files(chunks_dir)

        shard_contract = {
            "run_contract": run_contract,
            "run_contract_sha256": run_contract_sha256,
            "shard_index": args.shard_index,
            "shard_row_count": len(shard_rows),
            "shard_row_indices_sha256": _canonical_json_sha256(
                [row.row_index for row in shard_rows]
            ),
            "shard_row_ids_sha256": _canonical_json_sha256(
                [row.row_id for row in shard_rows]
            ),
        }
        _ensure_contract(shard_dir / "shard_contract.json", shard_contract)

        chunk_rows = list(_chunks(shard_rows))
        completed: dict[int, dict[str, Any]] = {}
        missing: list[int] = []
        for chunk_index, expected_rows in enumerate(chunk_rows):
            artifact = chunks_dir / f"chunk-{chunk_index:05d}.npz"
            metadata = chunks_dir / f"chunk-{chunk_index:05d}.json"
            if artifact.exists() or metadata.exists():
                completed[chunk_index] = _validate_chunk(
                    artifact,
                    metadata,
                    expected_rows,
                    run_contract_sha256,
                    args.shard_index,
                    chunk_index,
                )
            else:
                missing.append(chunk_index)

        completion_path = shard_dir / "shard_complete.json"
        if completion_path.exists() and missing:
            raise RuntimeError("shard_complete.json exists while chunks are missing")
        if completion_path.exists() and not missing:
            completion = _read_json(completion_path)
            if completion.get("run_contract_sha256") != run_contract_sha256:
                raise ValueError("completed shard uses a different run contract")
            if completion.get("row_count") != len(shard_rows):
                raise ValueError("completed shard row count mismatch")
            if completion.get("chunk_count") != len(chunk_rows):
                raise ValueError("completed shard chunk count mismatch")
            completion_chunks = completion.get("chunks")
            if not isinstance(completion_chunks, list) or len(completion_chunks) != len(
                chunk_rows
            ):
                raise ValueError("completed shard chunk inventory mismatch")
            for chunk_index, entry in enumerate(completion_chunks):
                if entry.get("chunk_index") != chunk_index or entry.get(
                    "artifact_sha256"
                ) != completed[chunk_index]["artifact_sha256"]:
                    raise ValueError("completed shard checksum inventory mismatch")
            print(
                f"shard {args.shard_index} already complete; all chunks validated",
                flush=True,
            )
            return

        invocation_started = time.perf_counter()
        load_seconds: float | None = None
        device: torch.device | None = None
        if missing:
            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("production ESMFold2 extraction requires CUDA")
            torch.cuda.set_device(device)
            model, load_seconds = _load_model(checkpoint, device)
            print(
                f"loaded persistent model for shard {args.shard_index} in {load_seconds:.1f}s; "
                f"{len(missing)}/{len(chunk_rows)} chunks remain",
                flush=True,
            )
            for chunk_index in missing:
                expected_rows = chunk_rows[chunk_index]
                bundles: list[dict[str, np.ndarray]] = []
                timings: list[dict[str, Any]] = []
                for offset, row in enumerate(expected_rows, start=1):
                    print(
                        f"shard={args.shard_index} chunk={chunk_index} "
                        f"row={row.row_index} ({offset}/{len(expected_rows)})",
                        flush=True,
                    )
                    bundle, timing = _extract_one(
                        model=model,
                        row=row,
                        device=device,
                        seed=args.seed,
                    )
                    bundles.append(bundle)
                    timings.append(timing)
                artifact = chunks_dir / f"chunk-{chunk_index:05d}.npz"
                metadata = chunks_dir / f"chunk-{chunk_index:05d}.json"
                completed[chunk_index] = _write_chunk(
                    artifact,
                    metadata,
                    expected_rows,
                    bundles,
                    timings,
                    run_contract_sha256,
                    args.shard_index,
                    chunk_index,
                )
                print(
                    f"committed shard={args.shard_index} chunk={chunk_index} "
                    f"rows={len(expected_rows)} sha256={completed[chunk_index]['artifact_sha256']}",
                    flush=True,
                )

        # Revalidate every artifact after all writes before declaring completion.
        validated_chunks = []
        for chunk_index, expected_rows in enumerate(chunk_rows):
            artifact = chunks_dir / f"chunk-{chunk_index:05d}.npz"
            metadata = chunks_dir / f"chunk-{chunk_index:05d}.json"
            chunk_metadata = _validate_chunk(
                artifact,
                metadata,
                expected_rows,
                run_contract_sha256,
                args.shard_index,
                chunk_index,
            )
            validated_chunks.append(
                {
                    "chunk_index": chunk_index,
                    "row_count": len(expected_rows),
                    "artifact": str(artifact.relative_to(shard_dir)),
                    "artifact_sha256": chunk_metadata["artifact_sha256"],
                    "metadata": str(metadata.relative_to(shard_dir)),
                    "metadata_sha256": _sha256_file(metadata),
                }
            )
        completion = {
            "schema_version": SCHEMA_VERSION,
            "completed_utc": _utc_now(),
            "run_contract_sha256": run_contract_sha256,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "row_count": len(shard_rows),
            "chunk_count": len(chunk_rows),
            "model_load_seconds_this_invocation": load_seconds,
            "extraction_seconds_this_invocation": time.perf_counter()
            - invocation_started,
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(device) if device is not None else None,
            "chunks": validated_chunks,
            "supervision_fields_read": [],
        }
        _atomic_json_dump(completion_path, completion)
        print(f"completed shard {args.shard_index}: {len(shard_rows)} rows", flush=True)
    finally:
        lock_handle.close()


def main(argv: Sequence[str] | None = None) -> None:
    _run(_parse_args(argv))


if __name__ == "__main__":
    main()
