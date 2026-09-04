#!/usr/bin/env python3
"""Validate and merge all canonical LibA or LibB ESMFold2 feature shards.

The merge is label-free.  It requires every completed modulo shard, verifies
that their run contracts are identical, verifies that all canonical row
indices and opaque row IDs occur exactly once, and validates every chunk
checksum and tensor contract while copying it.  It then atomically publishes
three uncompressed ``.npy`` arrays that can be memory-mapped by downstream
readouts:

* ``distogram_probabilities.npy``: ``[N, 9, 58, 64]`` float16;
* ``pair_states_symmetric.npy``: ``[N, 9, 58, 256]`` float16;
* ``single_inputs.npy``: ``[N, 67, 451]`` float16.

``metadata.csv`` contains only ``row_index,row_id,split``.  No weak-selection
or direct-retention label file is opened or copied by this program.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# Permit both ``python -m downstream...`` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import extract_esmfold2_libb_features as extraction


MERGE_SCHEMA_VERSION = "esmfold2-libb-feature-merge-v1"
LIBA_MERGE_SCHEMA_VERSION = "esmfold2-liba-feature-merge-v1"
MERGE_SCHEMA_BY_LIBRARY = {
    "LibA": LIBA_MERGE_SCHEMA_VERSION,
    "LibB": MERGE_SCHEMA_VERSION,
}
EXPECTED_NUM_SHARDS = 8
MERGED_DIRECTORY_NAME = "merged"
METADATA_COLUMNS = ("row_index", "row_id", "split")


@dataclass(frozen=True)
class ChunkSource:
    shard_index: int
    chunk_index: int
    artifact_path: Path
    metadata_path: Path
    expected_rows: tuple[extraction.CanonicalRow, ...]


@dataclass(frozen=True)
class MergePlan:
    run_contract: Mapping[str, Any]
    run_contract_sha256: str
    chunk_sources: tuple[ChunkSource, ...]
    shard_completion_paths: tuple[Path, ...]
    profile: extraction.DatasetProfile


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Dedicated esmfold2_liba_features_* or esmfold2_libb_features_* directory.",
    )
    parser.add_argument(
        "--row-manifest",
        type=Path,
        required=True,
        help="The same label-free <builder-output>/rows.json used for extraction.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Require and fully validate an already published merged directory.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=EXPECTED_NUM_SHARDS,
        help="Exact expected shard count; defaults to the historical eight-shard run.",
    )
    return parser.parse_args(argv)


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"expected a real regular file: {path}")


def _read_json(path: Path) -> Any:
    _require_regular_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _open_private_lock(path: Path):
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
        raise PermissionError(f"merge lock must be mode 0600, found {mode:04o}: {path}")
    return os.fdopen(descriptor, "a+")


def _validate_run_contract(
    run_contract: Mapping[str, Any],
    run_contract_sha256: str,
    row_manifest: Path,
    rows: Sequence[extraction.CanonicalRow],
    expected_num_shards: int,
    profile: extraction.DatasetProfile = extraction.LIBB_PROFILE,
) -> None:
    observed_hash = extraction._canonical_json_sha256(run_contract)
    if run_contract_sha256 != observed_hash:
        raise ValueError("shard run-contract hash does not match its payload")
    if run_contract.get("schema_version") != profile.feature_schema:
        raise ValueError("unexpected extraction run-contract schema")
    manifest_contract = run_contract.get("row_manifest")
    if not isinstance(manifest_contract, Mapping):
        raise ValueError("run contract lacks row_manifest metadata")
    train_count = sum(row.split == "train" for row in rows)
    eval_count = sum(row.split == "eval" for row in rows)
    expected_manifest_fields = {
        "sha256": extraction._sha256_file(row_manifest),
        "schema_version": profile.row_manifest_schema,
        "row_count": len(rows),
        "train_rows": train_count,
        "eval_rows": eval_count,
    }
    for key, expected in expected_manifest_fields.items():
        if manifest_contract.get(key) != expected:
            raise ValueError(f"run-contract row_manifest mismatch for {key}")
    settings = run_contract.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("run contract lacks extraction settings")
    if settings.get("num_shards") != expected_num_shards:
        raise ValueError("run-contract shard count changed")
    if settings.get("partition") != "row_index modulo num_shards":
        raise ValueError("run-contract partition rule changed")
    if settings.get("chunk_size") != extraction.CHUNK_SIZE:
        raise ValueError("run-contract chunk size changed")
    if run_contract.get("supervision_fields_read") != []:
        raise ValueError("run contract is not label-free")

    feature_contract = run_contract.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ValueError("run contract lacks a feature contract")
    for name, (shape, dtype) in extraction.SAVED_FEATURE_SPECS.items():
        record = feature_contract.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"run contract lacks feature {name}")
        if record.get("shape") != list(shape) or record.get("dtype") != str(dtype):
            raise ValueError(f"run-contract feature schema changed for {name}")


def _validate_chunk_metadata_membership(
    metadata: Mapping[str, Any],
    rows: Sequence[extraction.CanonicalRow],
    run_contract_sha256: str,
    shard_index: int,
    chunk_index: int,
    artifact_name: str,
    feature_schema: str = extraction.SCHEMA_VERSION,
) -> None:
    expected = {
        "schema_version": feature_schema,
        "run_contract_sha256": run_contract_sha256,
        "shard_index": shard_index,
        "chunk_index": chunk_index,
        "row_count": len(rows),
        "row_indices": [row.row_index for row in rows],
        "row_ids": [row.row_id for row in rows],
        "sequence_pair_sha256": [row.sequence_pair_sha256 for row in rows],
        "artifact": artifact_name,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"shard {shard_index} chunk {chunk_index} metadata mismatch for {key}"
            )


def _preflight_merge(
    feature_root: Path,
    row_manifest: Path,
    rows: Sequence[extraction.CanonicalRow],
    expected_num_shards: int = EXPECTED_NUM_SHARDS,
    profile: extraction.DatasetProfile = extraction.LIBB_PROFILE,
) -> MergePlan:
    expected_names = {
        f"shard-{index:05d}-of-{expected_num_shards:05d}"
        for index in range(expected_num_shards)
    }
    discovered_names = {
        path.name
        for path in feature_root.iterdir()
        if path.name.startswith("shard-")
    }
    if discovered_names != expected_names:
        missing = sorted(expected_names - discovered_names)
        unexpected = sorted(discovered_names - expected_names)
        raise RuntimeError(
            f"exactly {expected_num_shards} canonical shard directories are required; "
            f"missing={missing}, unexpected={unexpected}"
        )

    reference_contract: Mapping[str, Any] | None = None
    reference_hash: str | None = None
    sources: list[ChunkSource] = []
    completion_paths: list[Path] = []
    observed_indices: list[int] = []
    observed_ids: list[str] = []

    for shard_index in range(expected_num_shards):
        shard_dir = feature_root / f"shard-{shard_index:05d}-of-{expected_num_shards:05d}"
        if shard_dir.is_symlink() or not shard_dir.is_dir():
            raise ValueError(f"shard directory must be a real directory: {shard_dir}")
        shard_contract_path = shard_dir / "shard_contract.json"
        completion_path = shard_dir / "shard_complete.json"
        shard_contract = _read_json(shard_contract_path)
        completion = _read_json(completion_path)
        if not isinstance(shard_contract, Mapping) or not isinstance(completion, Mapping):
            raise ValueError(f"invalid JSON object in shard {shard_index}")

        run_contract = shard_contract.get("run_contract")
        run_hash = shard_contract.get("run_contract_sha256")
        if not isinstance(run_contract, Mapping) or not isinstance(run_hash, str):
            raise ValueError(f"shard {shard_index} lacks a complete run contract")
        _validate_run_contract(
            run_contract,
            run_hash,
            row_manifest,
            rows,
            expected_num_shards,
            profile,
        )
        if reference_contract is None:
            reference_contract = run_contract
            reference_hash = run_hash
        elif run_contract != reference_contract or run_hash != reference_hash:
            raise ValueError("the completed shards do not have identical run contracts")

        shard_rows = extraction._partition_rows(rows, shard_index, expected_num_shards)
        expected_shard_contract = {
            "shard_index": shard_index,
            "shard_row_count": len(shard_rows),
            "shard_row_indices_sha256": extraction._canonical_json_sha256(
                [row.row_index for row in shard_rows]
            ),
            "shard_row_ids_sha256": extraction._canonical_json_sha256(
                [row.row_id for row in shard_rows]
            ),
        }
        for key, expected in expected_shard_contract.items():
            if shard_contract.get(key) != expected:
                raise ValueError(f"shard {shard_index} contract mismatch for {key}")

        chunk_rows = list(extraction._chunks(shard_rows))
        expected_completion = {
            "schema_version": profile.feature_schema,
            "run_contract_sha256": run_hash,
            "shard_index": shard_index,
            "num_shards": expected_num_shards,
            "row_count": len(shard_rows),
            "chunk_count": len(chunk_rows),
            "supervision_fields_read": [],
        }
        for key, expected in expected_completion.items():
            if completion.get(key) != expected:
                raise ValueError(f"shard {shard_index} completion mismatch for {key}")
        inventory = completion.get("chunks")
        if not isinstance(inventory, list) or len(inventory) != len(chunk_rows):
            raise ValueError(f"shard {shard_index} has an invalid chunk inventory")

        for chunk_index, (expected_rows, entry) in enumerate(zip(chunk_rows, inventory)):
            if not isinstance(entry, Mapping):
                raise ValueError("chunk inventory entries must be objects")
            artifact_path = shard_dir / "chunks" / f"chunk-{chunk_index:05d}.npz"
            metadata_path = shard_dir / "chunks" / f"chunk-{chunk_index:05d}.json"
            _require_regular_file(artifact_path)
            metadata = _read_json(metadata_path)
            if not isinstance(metadata, Mapping):
                raise ValueError("chunk metadata must be an object")
            _validate_chunk_metadata_membership(
                metadata,
                expected_rows,
                run_hash,
                shard_index,
                chunk_index,
                artifact_path.name,
                profile.feature_schema,
            )
            expected_inventory = {
                "chunk_index": chunk_index,
                "row_count": len(expected_rows),
                "artifact": str(artifact_path.relative_to(shard_dir)),
                "artifact_sha256": metadata.get("artifact_sha256"),
                "metadata": str(metadata_path.relative_to(shard_dir)),
                "metadata_sha256": extraction._sha256_file(metadata_path),
            }
            for key, expected in expected_inventory.items():
                if entry.get(key) != expected:
                    raise ValueError(
                        f"shard {shard_index} completion inventory mismatch for {key}"
                    )
            sources.append(
                ChunkSource(
                    shard_index=shard_index,
                    chunk_index=chunk_index,
                    artifact_path=artifact_path,
                    metadata_path=metadata_path,
                    expected_rows=tuple(expected_rows),
                )
            )
            observed_indices.extend(row.row_index for row in expected_rows)
            observed_ids.extend(row.row_id for row in expected_rows)
        completion_paths.append(completion_path)

    canonical_indices = [row.row_index for row in rows]
    canonical_ids = [row.row_id for row in rows]
    if len(observed_indices) != len(set(observed_indices)):
        raise ValueError("a canonical row index occurs in more than one shard chunk")
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("a canonical row ID occurs in more than one shard chunk")
    if sorted(observed_indices) != canonical_indices:
        raise ValueError("shard chunks do not cover every canonical row index exactly once")
    id_by_index = dict(zip(observed_indices, observed_ids))
    if [id_by_index[index] for index in canonical_indices] != canonical_ids:
        raise ValueError("shard row IDs are not aligned to canonical row indices")
    assert reference_contract is not None and reference_hash is not None
    return MergePlan(
        run_contract=reference_contract,
        run_contract_sha256=reference_hash,
        chunk_sources=tuple(sources),
        shard_completion_paths=tuple(completion_paths),
        profile=profile,
    )


def _write_label_free_metadata(path: Path, rows: Sequence[extraction.CanonicalRow]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(METADATA_COLUMNS)
        for row in rows:
            writer.writerow((row.row_index, row.row_id, row.split))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_chunks_to_memmaps(
    temporary_directory: Path,
    rows: Sequence[extraction.CanonicalRow],
    plan: MergePlan,
) -> dict[str, dict[str, Any]]:
    memmaps: dict[str, np.memmap] = {}
    output_paths: dict[str, Path] = {}
    for name, (per_row_shape, dtype) in extraction.SAVED_FEATURE_SPECS.items():
        output_path = temporary_directory / f"{name}.npy"
        output_paths[name] = output_path
        memmaps[name] = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=dtype,
            shape=(len(rows), *per_row_shape),
        )
        os.chmod(output_path, 0o600)

    written = np.zeros(len(rows), dtype=np.bool_)
    try:
        for source in plan.chunk_sources:
            _, arrays = extraction._load_validated_chunk(
                source.artifact_path,
                source.metadata_path,
                source.expected_rows,
                plan.run_contract_sha256,
                source.shard_index,
                source.chunk_index,
                plan.profile.feature_schema,
            )
            indices = np.asarray(
                [row.row_index for row in source.expected_rows], dtype=np.int64
            )
            if written[indices].any():
                raise ValueError("attempted to write a canonical row more than once")
            for name in extraction.SAVED_FEATURE_SPECS:
                memmaps[name][indices] = arrays[name]
            written[indices] = True
        if not written.all():
            missing = np.flatnonzero(~written).tolist()
            raise ValueError(f"merged tensors are missing canonical rows: {missing[:20]}")
        for value in memmaps.values():
            value.flush()
    finally:
        # Closing the mmap before hashing prevents buffered writes from being
        # mistaken for a completed global artifact.
        for value in memmaps.values():
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        memmaps.clear()

    records: dict[str, dict[str, Any]] = {}
    for name, path in output_paths.items():
        _fsync_file(path)
        per_row_shape, dtype = extraction.SAVED_FEATURE_SPECS[name]
        records[name] = {
            "file": path.name,
            "shape": [len(rows), *per_row_shape],
            "dtype": str(dtype),
            "bytes": path.stat().st_size,
            "sha256": extraction._sha256_file(path),
        }
    return records


def _read_metadata_rows(path: Path) -> list[tuple[int, str, str]]:
    _require_regular_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != METADATA_COLUMNS:
            raise ValueError("merged metadata.csv schema changed")
        records: list[tuple[int, str, str]] = []
        for record in reader:
            if set(record) != set(METADATA_COLUMNS):
                raise ValueError("merged metadata.csv row schema changed")
            records.append(
                (int(record["row_index"]), str(record["row_id"]), str(record["split"]))
            )
    return records


def _validate_merged_output(
    merged_directory: Path,
    rows: Sequence[extraction.CanonicalRow],
    plan: MergePlan,
) -> Mapping[str, Any]:
    if merged_directory.is_symlink() or not merged_directory.is_dir():
        raise ValueError(f"merged output must be a real directory: {merged_directory}")
    if merged_directory.stat().st_mode & 0o777 != 0o700:
        raise PermissionError("merged directory must be mode 0700")
    completion_path = merged_directory / "merge_complete.json"
    completion = _read_json(completion_path)
    if not isinstance(completion, Mapping):
        raise ValueError("merge_complete.json must contain an object")
    expected_completion = {
        "schema_version": MERGE_SCHEMA_BY_LIBRARY[plan.profile.library],
        "run_contract_sha256": plan.run_contract_sha256,
        "num_shards": len(plan.shard_completion_paths),
        "row_count": len(rows),
        "row_indices_sha256": extraction._canonical_json_sha256(
            [row.row_index for row in rows]
        ),
        "row_ids_sha256": extraction._canonical_json_sha256(
            [row.row_id for row in rows]
        ),
        "supervision_fields_read": [],
    }
    for key, expected in expected_completion.items():
        if completion.get(key) != expected:
            raise ValueError(f"merged completion mismatch for {key}")

    metadata_path = merged_directory / "metadata.csv"
    metadata_record = completion.get("metadata")
    if not isinstance(metadata_record, Mapping):
        raise ValueError("merged completion lacks metadata.csv record")
    if metadata_record.get("file") != metadata_path.name:
        raise ValueError("merged metadata filename changed")
    if metadata_record.get("bytes") != metadata_path.stat().st_size:
        raise ValueError("merged metadata byte count changed")
    if metadata_record.get("sha256") != extraction._sha256_file(metadata_path):
        raise ValueError("merged metadata checksum mismatch")
    observed_metadata = _read_metadata_rows(metadata_path)
    expected_metadata = [(row.row_index, row.row_id, row.split) for row in rows]
    if observed_metadata != expected_metadata:
        raise ValueError("merged metadata rows changed or are out of canonical order")

    array_records = completion.get("arrays")
    if not isinstance(array_records, Mapping):
        raise ValueError("merged completion lacks array records")
    if set(array_records) != set(extraction.SAVED_FEATURE_SPECS):
        raise ValueError("merged array inventory changed")
    for name, (per_row_shape, dtype) in extraction.SAVED_FEATURE_SPECS.items():
        record = array_records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"merged array record is invalid for {name}")
        path = merged_directory / f"{name}.npy"
        _require_regular_file(path)
        expected_record = {
            "file": path.name,
            "shape": [len(rows), *per_row_shape],
            "dtype": str(dtype),
            "bytes": path.stat().st_size,
            "sha256": extraction._sha256_file(path),
        }
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise ValueError(f"merged {name} record mismatch for {key}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if value.shape != (len(rows), *per_row_shape) or value.dtype != dtype:
                raise ValueError(f"merged {name} array schema changed")
        finally:
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
    return completion


def _publish_merge(
    feature_root: Path,
    rows: Sequence[extraction.CanonicalRow],
    plan: MergePlan,
) -> Path:
    final_directory = feature_root / MERGED_DIRECTORY_NAME
    if final_directory.exists() or final_directory.is_symlink():
        _validate_merged_output(final_directory, rows, plan)
        return final_directory

    leftovers = sorted(path.name for path in feature_root.glob(".merged.tmp-*"))
    if leftovers:
        raise RuntimeError(
            "an interrupted merge requires review before retrying: " + ", ".join(leftovers)
        )
    temporary_directory = feature_root / f".merged.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    extraction._ensure_private_directory(temporary_directory)
    started = time.perf_counter()

    arrays = _copy_chunks_to_memmaps(temporary_directory, rows, plan)
    metadata_path = temporary_directory / "metadata.csv"
    _write_label_free_metadata(metadata_path, rows)
    metadata_record = {
        "file": metadata_path.name,
        "columns": list(METADATA_COLUMNS),
        "rows": len(rows),
        "bytes": metadata_path.stat().st_size,
        "sha256": extraction._sha256_file(metadata_path),
    }
    completion = {
        "schema_version": MERGE_SCHEMA_BY_LIBRARY[plan.profile.library],
        "completed_utc": extraction._utc_now(),
        "run_contract_sha256": plan.run_contract_sha256,
        "num_shards": len(plan.shard_completion_paths),
        "row_count": len(rows),
        "row_indices_sha256": extraction._canonical_json_sha256(
            [row.row_index for row in rows]
        ),
        "row_ids_sha256": extraction._canonical_json_sha256(
            [row.row_id for row in rows]
        ),
        "merge_seconds": time.perf_counter() - started,
        "arrays": arrays,
        "metadata": metadata_record,
        "source_shard_completions": [
            {
                "path": str(path.relative_to(feature_root)),
                "sha256": extraction._sha256_file(path),
            }
            for path in plan.shard_completion_paths
        ],
        "supervision_fields_read": [],
    }
    extraction._atomic_json_dump(
        temporary_directory / "merge_complete.json", completion
    )
    directory_descriptor = os.open(str(temporary_directory), os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.replace(temporary_directory, final_directory)
    extraction._fsync_parent(final_directory)
    _validate_merged_output(final_directory, rows, plan)
    return final_directory


def _run(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    row_manifest = args.row_manifest.resolve()
    if row_manifest.is_symlink() or not row_manifest.is_file():
        raise FileNotFoundError(f"row manifest must be a real file: {row_manifest}")
    payload, rows = extraction._load_row_manifest(row_manifest)
    profile = extraction._profile_for_manifest(payload)
    feature_root = extraction._validate_private_output_root(
        args.feature_root, output_prefix=profile.output_prefix
    )
    extraction._ensure_private_directory(feature_root)

    lock_handle = _open_private_lock(feature_root / ".merge.lock")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another merge process owns this feature root") from exc
        plan = _preflight_merge(
            feature_root,
            row_manifest,
            rows,
            expected_num_shards=args.num_shards,
            profile=profile,
        )
        final_directory = feature_root / MERGED_DIRECTORY_NAME
        if args.validate_only:
            _validate_merged_output(final_directory, rows, plan)
            print(f"validated merged ESMFold2 features: {final_directory}", flush=True)
            return
        published = _publish_merge(feature_root, rows, plan)
        print(f"published and validated merged ESMFold2 features: {published}", flush=True)
    finally:
        lock_handle.close()


def main(argv: Sequence[str] | None = None) -> None:
    _run(_parse_args(argv))


if __name__ == "__main__":
    main()
