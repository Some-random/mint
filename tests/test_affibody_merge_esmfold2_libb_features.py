"""CPU-only tests for validating and merging LibB ESMFold2 feature shards."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from downstream.AffibodyMHC import extract_esmfold2_libb_features as extraction
from downstream.AffibodyMHC import merge_esmfold2_libb_features as merge


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _chain1(first: str) -> str:
    return first + "A" * 260 + "SLLMWITQV"


def _chain2(first: str) -> str:
    return first + "G" * 57


def _raw_rows() -> list[dict]:
    rows = []
    for index, (split, first) in enumerate(
        (("train", "A"), ("train", "C"), ("eval", "D"))
    ):
        chain1 = _chain1(first)
        chain2 = _chain2(first)
        rows.append(
            {
                "row_index": index,
                "row_id": f"opaque-{index}",
                "split": split,
                "chain1_sequence": chain1,
                "chain2_sequence": chain2,
                "sequence_pair_sha256": _hash(chain1 + "|" + chain2),
            }
        )
    return rows


def _expectations() -> extraction.DatasetExpectations:
    return extraction.DatasetExpectations(
        train_rows=2,
        eval_rows=1,
        train_chain1=2,
        train_chain2=2,
        eval_chain1=1,
        eval_chain2=1,
    )


def _bundle(value: float) -> dict[str, np.ndarray]:
    result = {}
    for name, (shape, dtype) in extraction.SAVED_FEATURE_SPECS.items():
        if name == "distogram_probabilities":
            array = np.full(shape, 1.0 / shape[-1], dtype=dtype)
        else:
            array = np.full(shape, value, dtype=dtype)
        result[name] = array
    return result


def _write_source_tree(
    tmp_path,
    num_shards: int = 2,
    profile: extraction.DatasetProfile = extraction.LIBB_PROFILE,
):
    feature_root = tmp_path / f"{profile.output_prefix}test"
    feature_root.mkdir(mode=0o700)
    payload = {
        "schema_version": profile.row_manifest_schema,
        "rows": _raw_rows(),
    }
    row_manifest = tmp_path / "rows.json"
    row_manifest.write_text(json.dumps(payload), encoding="utf-8")
    canonical_rows = extraction._validate_rows(
        payload["rows"], _expectations(), profile=profile
    )
    run_contract = {
        "schema_version": profile.feature_schema,
        "row_manifest": {
            "path": str(row_manifest),
            "sha256": extraction._sha256_file(row_manifest),
            "schema_version": profile.row_manifest_schema,
            "row_count": len(canonical_rows),
            "train_rows": 2,
            "eval_rows": 1,
        },
        "feature_contract": {
            name: {"shape": list(shape), "dtype": str(dtype)}
            for name, (shape, dtype) in extraction.SAVED_FEATURE_SPECS.items()
        },
        "settings": {
            "num_shards": num_shards,
            "partition": "row_index modulo num_shards",
            "chunk_size": extraction.CHUNK_SIZE,
        },
        "supervision_fields_read": [],
    }
    run_hash = extraction._canonical_json_sha256(run_contract)

    for shard_index in range(num_shards):
        shard_rows = extraction._partition_rows(
            canonical_rows, shard_index, num_shards
        )
        shard_dir = feature_root / f"shard-{shard_index:05d}-of-{num_shards:05d}"
        chunks_dir = shard_dir / "chunks"
        chunks_dir.mkdir(parents=True, mode=0o700)
        shard_contract = {
            "run_contract": run_contract,
            "run_contract_sha256": run_hash,
            "shard_index": shard_index,
            "shard_row_count": len(shard_rows),
            "shard_row_indices_sha256": extraction._canonical_json_sha256(
                [row.row_index for row in shard_rows]
            ),
            "shard_row_ids_sha256": extraction._canonical_json_sha256(
                [row.row_id for row in shard_rows]
            ),
        }
        extraction._atomic_json_dump(shard_dir / "shard_contract.json", shard_contract)

        inventory = []
        for chunk_index, expected_rows in enumerate(extraction._chunks(shard_rows)):
            artifact = chunks_dir / f"chunk-{chunk_index:05d}.npz"
            metadata = chunks_dir / f"chunk-{chunk_index:05d}.json"
            bundles = [_bundle(float(row.row_index)) for row in expected_rows]
            timings = [
                {
                    "row_index": row.row_index,
                    "row_id": row.row_id,
                    "sequence_pair_sha256": row.sequence_pair_sha256,
                    "preparation_seconds": 0.0,
                    "forward_seconds": 0.0,
                    "peak_gpu_memory_bytes": None,
                }
                for row in expected_rows
            ]
            chunk_metadata = extraction._write_chunk(
                artifact,
                metadata,
                expected_rows,
                bundles,
                timings,
                run_hash,
                shard_index,
                chunk_index,
                profile.feature_schema,
            )
            inventory.append(
                {
                    "chunk_index": chunk_index,
                    "row_count": len(expected_rows),
                    "artifact": str(artifact.relative_to(shard_dir)),
                    "artifact_sha256": chunk_metadata["artifact_sha256"],
                    "metadata": str(metadata.relative_to(shard_dir)),
                    "metadata_sha256": extraction._sha256_file(metadata),
                }
            )
        completion = {
            "schema_version": profile.feature_schema,
            "run_contract_sha256": run_hash,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "row_count": len(shard_rows),
            "chunk_count": len(inventory),
            "chunks": inventory,
            "supervision_fields_read": [],
        }
        extraction._atomic_json_dump(shard_dir / "shard_complete.json", completion)
    return feature_root, row_manifest, canonical_rows


def test_merge_validates_all_shards_and_publishes_canonical_memmaps(tmp_path):
    feature_root, row_manifest, rows = _write_source_tree(tmp_path)
    plan = merge._preflight_merge(
        feature_root, row_manifest, rows, expected_num_shards=2
    )
    merged = merge._publish_merge(feature_root, rows, plan)

    assert merged.name == merge.MERGED_DIRECTORY_NAME
    assert merged.stat().st_mode & 0o777 == 0o700
    for name, (shape, dtype) in extraction.SAVED_FEATURE_SPECS.items():
        values = np.load(merged / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        assert values.shape == (3, *shape)
        assert values.dtype == dtype
        if name != "distogram_probabilities":
            assert values.reshape(3, -1)[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert (merged / "metadata.csv").read_text(encoding="utf-8").splitlines() == [
        "row_index,row_id,split",
        "0,opaque-0,train",
        "1,opaque-1,train",
        "2,opaque-2,eval",
    ]
    merge._validate_merged_output(merged, rows, plan)
    # A second publish is a checksum-validation no-op, never an overwrite.
    assert merge._publish_merge(feature_root, rows, plan) == merged


def test_merge_requires_every_expected_shard_complete_manifest(tmp_path):
    feature_root, row_manifest, rows = _write_source_tree(tmp_path)
    (feature_root / "shard-00001-of-00002" / "shard_complete.json").unlink()
    with pytest.raises(FileNotFoundError, match="shard_complete"):
        merge._preflight_merge(
            feature_root, row_manifest, rows, expected_num_shards=2
        )


def test_liba_merge_accepts_explicit_64_shards_and_publishes_liba_schema(tmp_path):
    feature_root, row_manifest, rows = _write_source_tree(
        tmp_path,
        num_shards=64,
        profile=extraction.LIBA_PROFILE,
    )
    plan = merge._preflight_merge(
        feature_root,
        row_manifest,
        rows,
        expected_num_shards=64,
        profile=extraction.LIBA_PROFILE,
    )
    merged = merge._publish_merge(feature_root, rows, plan)
    completion = json.loads((merged / "merge_complete.json").read_text())
    assert completion["schema_version"] == merge.LIBA_MERGE_SCHEMA_VERSION
    assert completion["num_shards"] == 64
