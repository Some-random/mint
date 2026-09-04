#!/usr/bin/env python
"""Export frozen MINT layer-5 vectors into a label-free LibB archive.

The source MINT NPZ contains many historical arrays, including supervision
fields that must never enter a structural trainer.  This exporter accesses
exactly two NPZ members: opaque ``pair_uid`` values and frozen
``mint_layer_05_chain_mean`` features.  It aligns those values to the 30,768
canonical LibB ``row_id`` values, then writes the generic opaque-feature
schema used by the matched structural readout.

An optional already-label-free structural archive can be supplied.  Its one
selected two-dimensional representation is copied under the fixed name
``structural_selected_vector``.  This produces the three-arm late-fusion archive:
MINT alone, structure alone, and their concatenation.  Without that optional
input, the command produces a usable MINT-only archive while structure
extraction is still running.

No labels, retention measurements, model scores, or evaluation subsets are
arguments to this program.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.libb_structural_readout import (
    FEATURE_ARCHIVE_SCHEMA_VERSION,
    OpaqueFeatureStore,
    canonical_json_sha256,
    sha256_file,
)
from downstream.AffibodyMHC.mint_structure_fusion import (
    DEFAULT_MINT_MULTILAYER_ARCHIVE,
    DEFAULT_MINT_MULTILAYER_MANIFEST,
    MINT_CHAIN_MEAN_DIM,
    MINT_LAYER,
    MINT_LAYER5_FEATURE_NAME,
    MINT_ROW_ID_NAME,
    aligned_row_indices,
)


PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
OUTPUT_MINT_NAME = "mint_layer05_global"
OUTPUT_STRUCTURE_NAME = "structural_selected_vector"
EXPECTED_CANONICAL_ROWS = 30_768
EXPECTED_TRAIN_ROWS = 30_648
EXPECTED_EVALUATION_ROWS = 120
EXPECTED_MINT_ROWS = 69_945
EXPECTED_CANONICAL_ROW_IDS_SHA256 = (
    "4e47d29c1d91b6890ba4c0de465eb22f9ad01aae1c500653d6160bc799f82674"
)
MINT_SOURCE_SCHEMA_VERSION = "mint-multilayer-chain-mean-cache-v1"
SOURCE_FIELDS_READ = (MINT_ROW_ID_NAME, MINT_LAYER5_FEATURE_NAME)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _validate_private_new_output(path: str | Path, private_root: Path) -> Path:
    output = Path(path).resolve()
    private_root = Path(private_root).resolve()
    try:
        relative = output.relative_to(private_root)
    except ValueError as error:
        raise ValueError("output must stay under private_data") from error
    _require(relative.parts, "refusing to write directly into private_data")
    _require(not output.exists(), "output directory exists; refusing overwrite")
    return output


def load_canonical_metadata(
    metadata_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_rows: int = EXPECTED_CANONICAL_ROWS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_evaluation_rows: int = EXPECTED_EVALUATION_ROWS,
    expected_row_ids_sha256: str = EXPECTED_CANONICAL_ROW_IDS_SHA256,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Validate a label-free canonical metadata table without opening features."""

    metadata_path = Path(metadata_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    _require(metadata_path.is_file(), f"missing canonical metadata {metadata_path}")
    _require(manifest_path.is_file(), f"missing canonical manifest {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("supervision_fields_read") == [],
        "canonical metadata source was not label-free",
    )
    contract = manifest.get("metadata")
    _require(isinstance(contract, dict), "canonical manifest lacks metadata contract")
    _require(contract.get("file") == metadata_path.name, "metadata filename changed")
    _require(contract.get("bytes") == metadata_path.stat().st_size, "metadata byte size changed")
    _require(contract.get("sha256") == sha256_file(metadata_path), "metadata checksum changed")
    _require(
        contract.get("columns") == ["row_index", "row_id", "split"],
        "metadata columns changed",
    )

    row_ids: list[str] = []
    splits: list[str] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        _require(next(reader, []) == ["row_index", "row_id", "split"], "metadata header changed")
        for row_number, row in enumerate(reader, start=2):
            _require(len(row) == 3, f"metadata row {row_number} is malformed")
            _require(row[0] == str(row_number - 2), "metadata row order changed")
            row_ids.append(row[1])
            splits.append(row[2])
    _require(len(row_ids) == int(expected_rows), "canonical row count changed")
    _require(len(set(row_ids)) == len(row_ids) and all(row_ids), "canonical row IDs are invalid")
    _require(set(splits) == {"train", "eval"}, "canonical split values changed")
    _require(splits.count("train") == int(expected_train_rows), "canonical training count changed")
    _require(
        splits.count("eval") == int(expected_evaluation_rows),
        "canonical evaluation count changed",
    )
    digest = canonical_json_sha256(row_ids)
    _require(digest == str(expected_row_ids_sha256), "canonical row-ID membership/order changed")
    _require(manifest.get("row_count") == len(row_ids), "canonical manifest row count changed")
    _require(manifest.get("row_ids_sha256") == digest, "canonical manifest row-ID digest changed")
    return tuple(row_ids), tuple(splits), contract


def validate_mint_manifest(
    manifest_path: str | Path,
    archive_path: str | Path,
    *,
    expected_rows: int = EXPECTED_MINT_ROWS,
) -> dict[str, Any]:
    """Validate the layer-5 source contract without reading any NPZ member."""

    manifest_path = Path(manifest_path).resolve()
    archive_path = Path(archive_path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("schema_version") == MINT_SOURCE_SCHEMA_VERSION,
        "MINT cache schema changed",
    )
    records = manifest.get("features", {}).get("by_layer", [])
    selected = [
        record for record in records if int(record.get("layer", -1)) == MINT_LAYER
    ]
    _require(len(selected) == 1, "MINT manifest lacks exactly one layer-5 record")
    contract = selected[0]
    _require(
        contract.get("name") == MINT_LAYER5_FEATURE_NAME,
        "MINT layer-5 key changed",
    )
    _require(
        contract.get("shape") == [int(expected_rows), MINT_CHAIN_MEAN_DIM],
        "MINT layer-5 shape changed",
    )
    _require(contract.get("dtype") == "float32", "MINT layer-5 dtype changed")
    output = manifest.get("output", {})
    _require(
        Path(str(output.get("path", ""))).name == archive_path.name,
        "MINT archive filename changed",
    )
    _require(
        MINT_LAYER5_FEATURE_NAME in output.get("keys", ()),
        "MINT output lacks layer-5 features",
    )
    _require(MINT_ROW_ID_NAME in output.get("keys", ()), "MINT output lacks pair IDs")
    return {
        "manifest_schema_version": manifest["schema_version"],
        "declared_archive_sha256": str(output.get("sha256", "")),
        "feature_contract": contract,
    }


def _write_float_array(
    path: Path,
    shape: tuple[int, ...],
    source_chunks: Any,
) -> dict[str, Any]:
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for start, stop, values in source_chunks:
        values = np.asarray(values, dtype=np.float32)
        _require(values.shape == (stop - start, *shape[1:]), "feature chunk shape changed")
        _require(bool(np.isfinite(values).all()), "feature chunk contains non-finite values")
        output[start:stop] = values
    output.flush()
    del output
    os.chmod(path, 0o600)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(shape),
        "dtype": "float32",
        "kind": "feature",
    }


def _chunks(length: int, chunk_rows: int):
    for start in range(0, int(length), int(chunk_rows)):
        yield start, min(start + int(chunk_rows), int(length))


def export_archive(
    *,
    canonical_metadata: str | Path,
    canonical_manifest: str | Path,
    mint_archive: str | Path,
    mint_manifest: str | Path,
    output_dir: str | Path,
    structural_archive: str | Path | None = None,
    structural_array: str | None = None,
    verify_mint_checksum: bool = False,
    verify_structural_checksums: bool = False,
    chunk_rows: int = 256,
    private_root: Path = PRIVATE_ROOT,
    expected_canonical_rows: int = EXPECTED_CANONICAL_ROWS,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_evaluation_rows: int = EXPECTED_EVALUATION_ROWS,
    expected_mint_rows: int = EXPECTED_MINT_ROWS,
    expected_row_ids_sha256: str = EXPECTED_CANONICAL_ROW_IDS_SHA256,
) -> Path:
    """Create one new opaque archive; existing source artifacts are untouched."""

    _require(int(chunk_rows) >= 1, "chunk_rows must be positive")
    _require(
        (structural_archive is None) == (structural_array is None),
        "structural archive and array must be supplied together",
    )
    output_dir = _validate_private_new_output(output_dir, private_root)
    row_ids, splits, metadata_contract = load_canonical_metadata(
        canonical_metadata,
        canonical_manifest,
        expected_rows=expected_canonical_rows,
        expected_train_rows=expected_train_rows,
        expected_evaluation_rows=expected_evaluation_rows,
        expected_row_ids_sha256=expected_row_ids_sha256,
    )
    mint_archive = Path(mint_archive).resolve()
    mint_manifest = Path(mint_manifest).resolve()
    _require(mint_archive.is_file(), f"missing MINT archive {mint_archive}")
    mint_contract = validate_mint_manifest(
        mint_manifest, mint_archive, expected_rows=expected_mint_rows
    )
    if verify_mint_checksum:
        _require(
            mint_contract["declared_archive_sha256"] == sha256_file(mint_archive),
            "MINT archive checksum changed",
        )

    structural_store = None
    if structural_archive is not None:
        _require(
            str(structural_array) not in SOURCE_FIELDS_READ,
            "structural array name collides with MINT",
        )
        structural_store = OpaqueFeatureStore.open(
            structural_archive,
            [str(structural_array)],
            verify_all_checksums=bool(verify_structural_checksums),
        )
        _require(structural_store.row_ids == row_ids, "structural and canonical row order differ")
        _require(structural_store.splits == splits, "structural and canonical splits differ")
        structural_values = structural_store.arrays[str(structural_array)]
        _require(structural_values.ndim == 2, "selected structural representation must be [N,D]")

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    marker = output_dir / "EXTRACTION_INCOMPLETE"
    marker.write_text("incomplete\n", encoding="ascii")
    os.chmod(marker, 0o600)
    started = time.time()
    metadata_output = output_dir / "metadata.csv"
    shutil.copyfile(canonical_metadata, metadata_output)
    os.chmod(metadata_output, 0o600)

    arrays: dict[str, Any] = {}
    with np.load(mint_archive, allow_pickle=False) as source:
        _require(MINT_ROW_ID_NAME in source.files, "MINT NPZ lacks pair_uid")
        _require(MINT_LAYER5_FEATURE_NAME in source.files, "MINT NPZ lacks layer-5 features")
        available_ids = np.asarray(source[MINT_ROW_ID_NAME]).astype(str)
        _require(available_ids.shape == (int(expected_mint_rows),), "MINT pair-ID shape changed")
        indices = aligned_row_indices(row_ids, available_ids.tolist())
        features = source[MINT_LAYER5_FEATURE_NAME]
        _require(
            features.shape == (int(expected_mint_rows), MINT_CHAIN_MEAN_DIM),
            "MINT layer-5 array shape changed",
        )
        _require(features.dtype == np.float32, "MINT layer-5 array dtype changed")
        mint_chunks = (
            (start, stop, features[indices[start:stop]])
            for start, stop in _chunks(len(row_ids), chunk_rows)
        )
        arrays[OUTPUT_MINT_NAME] = _write_float_array(
            output_dir / f"{OUTPUT_MINT_NAME}.npy",
            (len(row_ids), MINT_CHAIN_MEAN_DIM),
            mint_chunks,
        )

    if structural_store is not None:
        structural_values = structural_store.arrays[str(structural_array)]
        structure_chunks = (
            (start, stop, structural_values[start:stop])
            for start, stop in _chunks(len(row_ids), chunk_rows)
        )
        arrays[OUTPUT_STRUCTURE_NAME] = _write_float_array(
            output_dir / f"{OUTPUT_STRUCTURE_NAME}.npy",
            tuple(map(int, structural_values.shape)),
            structure_chunks,
        )

    metadata_output_contract = {
        "file": metadata_output.name,
        "bytes": metadata_output.stat().st_size,
        "sha256": sha256_file(metadata_output),
        "columns": ["row_index", "row_id", "split"],
        "rows": len(row_ids),
    }
    _require(
        metadata_output_contract["sha256"] == metadata_contract["sha256"],
        "copied metadata checksum changed",
    )
    manifest = {
        "schema_version": FEATURE_ARCHIVE_SCHEMA_VERSION,
        "supervision_fields_read": [],
        "source_fields_read": list(SOURCE_FIELDS_READ),
        "row_count": len(row_ids),
        "row_ids_sha256": canonical_json_sha256(row_ids),
        "metadata": metadata_output_contract,
        "arrays": arrays,
        "fusion_contract": {
            "mint_layer": MINT_LAYER,
            "mint_source_key": MINT_LAYER5_FEATURE_NAME,
            "mint_output_array": OUTPUT_MINT_NAME,
            "mint_dimension": MINT_CHAIN_MEAN_DIM,
            "structural_output_array": (
                OUTPUT_STRUCTURE_NAME if structural_store is not None else None
            ),
            "operation": "raw vector concatenation immediately before matched readout",
            "upstream_encoders_frozen": True,
        },
        "inputs": {
            "canonical_metadata": {
                "path": str(Path(canonical_metadata).resolve()),
                "sha256": metadata_contract["sha256"],
            },
            "canonical_manifest": {
                "path": str(Path(canonical_manifest).resolve()),
                "sha256": sha256_file(Path(canonical_manifest).resolve()),
            },
            "mint_archive": {
                "path": str(mint_archive),
                "declared_sha256": mint_contract["declared_archive_sha256"],
                "checksum_verified": bool(verify_mint_checksum),
            },
            "mint_manifest": {
                "path": str(mint_manifest),
                "sha256": sha256_file(mint_manifest),
            },
            "structural_archive": (
                None
                if structural_store is None
                else {
                    "path": str(structural_store.root),
                    "source_array": str(structural_array),
                    "checksums_verified": bool(verify_structural_checksums),
                    "manifest_sha256": structural_store.manifest_sha256,
                    "source_manifest": structural_store.manifest,
                }
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    _json_dump(output_dir / "manifest.json", manifest)
    marker.unlink()
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-metadata", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--mint-archive", type=Path, default=DEFAULT_MINT_MULTILAYER_ARCHIVE)
    parser.add_argument("--mint-manifest", type=Path, default=DEFAULT_MINT_MULTILAYER_MANIFEST)
    parser.add_argument("--structural-archive", type=Path)
    parser.add_argument("--structural-array")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--verify-mint-checksum", action="store_true")
    parser.add_argument("--verify-structural-checksums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = export_archive(
        canonical_metadata=args.canonical_metadata,
        canonical_manifest=args.canonical_manifest,
        mint_archive=args.mint_archive,
        mint_manifest=args.mint_manifest,
        structural_archive=args.structural_archive,
        structural_array=args.structural_array,
        output_dir=args.output_dir,
        verify_mint_checksum=args.verify_mint_checksum,
        verify_structural_checksums=args.verify_structural_checksums,
        chunk_rows=args.chunk_rows,
    )
    print(output)


if __name__ == "__main__":
    main()
