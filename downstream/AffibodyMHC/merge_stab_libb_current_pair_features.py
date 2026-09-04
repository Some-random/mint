#!/usr/bin/env python3
"""Merge deterministic StaB LibB feature shards into canonical row order."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import stab_libb_current_pair_features as features


DEFAULT_ROWS = (
    features.PRIVATE_ROOT
    / "derived"
    / "esmfold2_libb_canonical_rows_provider_revision_120_v1"
    / "rows.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-manifest", type=Path, default=DEFAULT_ROWS)
    parser.add_argument(
        "--context-mode", choices=features.CONTEXT_MODES, required=True
    )
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def _load_metadata(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header != ["row_index", "row_id", "split"]:
            raise ValueError(f"invalid shard metadata header: {path}")
        row_ids = []
        splits = []
        for output_index, row in enumerate(reader):
            if len(row) != 3 or row[0] != str(output_index):
                raise ValueError(f"invalid shard metadata row {output_index}: {path}")
            row_ids.append(row[1])
            splits.append(row[2])
    return row_ids, splits


def _open_shard(path: Path, context_mode: str):
    path = path.resolve()
    if (path / "EXTRACTION_INCOMPLETE").exists():
        raise ValueError(f"feature shard is incomplete: {path}")
    with (path / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != features.FEATURE_ARCHIVE_SCHEMA:
        raise ValueError(f"feature shard schema changed: {path}")
    if manifest.get("extractor_schema_version") != features.EXTRACTOR_SCHEMA:
        raise ValueError(f"extractor schema changed: {path}")
    if (
        manifest.get("template_context_schema_version")
        != features.CONTEXT_SCHEMA_VERSIONS[context_mode]
    ):
        raise ValueError(f"template context schema changed: {path}")
    if manifest.get("supervision_fields_read") != []:
        raise ValueError(f"feature shard was not label-free: {path}")
    metadata = path / "metadata.csv"
    contract = manifest.get("metadata", {})
    if contract.get("sha256") != features.sha256_file(metadata):
        raise ValueError(f"metadata checksum changed: {path}")
    row_ids, splits = _load_metadata(metadata)
    if manifest.get("row_count") != len(row_ids):
        raise ValueError(f"shard row count changed: {path}")
    if manifest.get("row_ids_sha256") != features.canonical_json_sha256(row_ids):
        raise ValueError(f"shard row-ID digest changed: {path}")

    expected_specs = features.feature_specs()
    contracts = manifest.get("arrays", {})
    if set(contracts) != set(expected_specs):
        raise ValueError(f"shard feature-array set changed: {path}")
    arrays = {}
    for name, (tail_shape, dtype) in expected_specs.items():
        array_path = path / f"{name}.npy"
        observed_contract = contracts[name]
        if observed_contract.get("file") != array_path.name:
            raise ValueError(f"shard feature filename changed: {path}/{name}")
        if observed_contract.get("sha256") != features.sha256_file(array_path):
            raise ValueError(f"shard feature checksum changed: {path}/{name}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(row_ids),) + tuple(tail_shape):
            raise ValueError(f"shard feature shape changed: {path}/{name}")
        if array.dtype != np.dtype(dtype):
            raise ValueError(f"shard feature dtype changed: {path}/{name}")
        arrays[name] = array
    return manifest, row_ids, splits, arrays


def merge_shards(
    row_manifest: Path,
    shard_dirs: list[Path],
    output_dir: Path,
    context_mode: str,
    merge_batch_size: int = 512,
):
    if context_mode not in features.CONTEXT_MODES:
        raise ValueError(f"unknown context mode: {context_mode}")
    _, canonical_rows = features.load_canonical_rows(row_manifest)
    row_manifest_sha256 = features.sha256_file(row_manifest)
    expected = {row.row_id: row for row in canonical_rows}
    locations = {}
    stores = []
    shard_manifest_hashes = []
    observed_shard_indices = []
    for shard_number, shard_dir in enumerate(shard_dirs):
        manifest, row_ids, splits, arrays = _open_shard(shard_dir, context_mode)
        provenance = manifest.get("provenance", {})
        expected_provenance = {
            "context_mode": context_mode,
            "template_context_schema": features.CONTEXT_SCHEMA_VERSIONS[
                context_mode
            ],
            "upstream_repository": features.UPSTREAM_REPOSITORY,
            "upstream_commit": features.UPSTREAM_COMMIT,
            "checkpoint_sha256": features.FINAL_CHECKPOINT_SHA256,
            "pdb_sha256": features.REFERENCE_PDB_SHA256,
            "structure_mapping_schema": features.STRUCTURE_MAPPING_SCHEMA,
            "structure_mapping_sha256": features.STRUCTURE_MAPPING_SHA256,
            "row_manifest_sha256": row_manifest_sha256,
            "source_row_schema": features.ROW_MANIFEST_SCHEMA,
        }
        for field, expected_value in expected_provenance.items():
            if provenance.get(field) != expected_value:
                raise ValueError(
                    f"shard provenance mismatch for {field}: {shard_dir}"
                )
        if provenance.get("explicit_row_selection") is not False:
            raise ValueError(f"pilot shard cannot enter production merge: {shard_dir}")
        if int(provenance.get("num_shards", -1)) != len(shard_dirs):
            raise ValueError(f"num_shards provenance changed: {shard_dir}")
        observed_shard_indices.append(int(provenance.get("shard_index", -1)))
        stores.append(arrays)
        shard_manifest_hashes.append(
            features.sha256_file(Path(shard_dir) / "manifest.json")
        )
        for local_index, (row_id, split) in enumerate(zip(row_ids, splits)):
            if row_id not in expected:
                raise ValueError(f"unknown row ID in feature shards: {row_id}")
            if split != expected[row_id].split:
                raise ValueError(f"split changed for row {row_id}")
            if row_id in locations:
                raise ValueError(f"duplicate row ID across feature shards: {row_id}")
            locations[row_id] = (shard_number, local_index)
    missing = sorted(set(expected).difference(locations))
    if missing:
        raise ValueError(f"feature shards are missing {len(missing)} canonical rows")
    if sorted(observed_shard_indices) != list(range(len(shard_dirs))):
        raise ValueError("shard indices are incomplete or duplicated")

    merge_contract = features.archive_run_contract(
        context_mode,
        additional={
            "operation": "deterministic_shard_merge",
            "template_context_schema": features.CONTEXT_SCHEMA_VERSIONS[
                context_mode
            ],
            "upstream_repository": features.UPSTREAM_REPOSITORY,
            "upstream_commit": features.UPSTREAM_COMMIT,
            "checkpoint_sha256": features.FINAL_CHECKPOINT_SHA256,
            "pdb_sha256": features.REFERENCE_PDB_SHA256,
            "row_manifest_sha256": row_manifest_sha256,
            "source_row_schema": features.ROW_MANIFEST_SCHEMA,
            "source_shard_count": len(shard_dirs),
            "source_shard_manifest_sha256": shard_manifest_hashes,
            "source_shard_paths": [
                str(Path(path).resolve()) for path in shard_dirs
            ],
        },
    )
    writer = features.FeatureArchiveWriter(
        output_dir, canonical_rows, features.feature_specs(), merge_contract
    )
    for batch in features.iter_batches(canonical_rows, merge_batch_size):
        assembled = {name: [] for name in features.feature_specs()}
        for row in batch:
            shard_number, local_index = locations[row.row_id]
            for name in assembled:
                assembled[name].append(np.asarray(stores[shard_number][name][local_index]))
        writer.append(
            {
                name: np.stack(values).astype(features.feature_specs()[name][1], copy=False)
                for name, values in assembled.items()
            }
        )
    return writer.finish(merge_contract)


def main(argv=None):
    args = parse_args(argv)
    manifest = merge_shards(
        args.row_manifest,
        args.shard_dir,
        args.output_dir,
        args.context_mode,
        args.merge_batch_size,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "rows": manifest["row_count"],
                "shards": len(args.shard_dir),
                "context_mode": args.context_mode,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
