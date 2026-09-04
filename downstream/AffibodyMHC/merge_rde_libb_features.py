#!/usr/bin/env python3
"""Merge verified RDE LibB chunks into the shared opaque-feature archive.

The resulting directory implements ``libb-opaque-frozen-features-v1`` for
``libb_structural_readout.py``:

* ``metadata.csv`` contains only contiguous row index, opaque row ID, split;
* each frozen feature or boolean mask is a memory-mappable ``.npy`` array;
* ``manifest.json`` binds dimensions, dtypes, byte sizes, and checksums; and
* ``supervision_fields_read`` is explicitly empty.

No weak label or direct-retention file is accepted by this command.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.extract_rde_libb_features import (
    EXPECTED_CONTEXT_DIM,
    EXPECTED_RDE_NETWORK_SHA256,
    EXPECTED_RDE_REPOSITORY_COMMIT,
    EXPECTED_RDE_SHA256,
    EXPECTED_ROWS,
    PATCH_SIZE,
    SCHEMA_VERSION as SHARD_SCHEMA_VERSION,
    expected_feature_keys,
    sha256_file,
    validate_chunk,
    validate_private_output_path,
)
from downstream.AffibodyMHC.rde_libb_fixed_crystal import (
    AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED,
    FIXED_CRYSTAL_APPROXIMATION_OMISSIONS,
    HLA_ASSAY_IDENTITY_CORRECTIONS,
    validate_canonical_payload,
)


ARCHIVE_SCHEMA_VERSION = "libb-opaque-frozen-features-v1"
MERGE_SCHEMA_VERSION = "rde-libb-opaque-merge-v2"
MASK_NAMES = ("residue_mask", "peptide_mask", "affibody_mask", "designed_mask")
CANONICAL_DESIGNED_SITE_ORDER = (
    ("P", 4),
    ("P", 5),
    ("H", 6),
    ("H", 10),
    ("H", 13),
    ("H", 14),
    ("H", 17),
)
IGNORED_CHUNK_ARRAYS = frozenset(
    {"row_index", "row_id", "split", "sequence_pair_sha256", "chain_nb", "aa"}
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _atomic_json(path, payload):
    temporary = path.with_name(".{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with open(str(temporary), "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_metadata(path, rows):
    temporary = path.with_name(".{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with open(str(temporary), "x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_index", "row_id", "split"])
            for output_index, row in enumerate(rows):
                writer.writerow([output_index, row.row_id, row.split])
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def discover_chunks(shard_root):
    shard_root = Path(shard_root).resolve()
    _require(shard_root.is_dir(), "shard root does not exist")
    shard_directories = sorted(shard_root.glob("shard-*-of-*"))
    _require(shard_directories, "no RDE shard directories found")
    manifests = []
    chunks = []
    feature_source = None
    shard_fraction = None
    designed_patch_indices = None
    locked_provenance = None
    for directory in shard_directories:
        manifest_path = directory / "manifest.json"
        _require(manifest_path.is_file(), "shard lacks manifest: {}".format(directory.name))
        manifest = _read_json(manifest_path)
        _require(manifest.get("schema_version") == SHARD_SCHEMA_VERSION, "shard schema changed")
        _require(manifest.get("absolute_current_pair") is True, "shard is not current-pair")
        _require(manifest.get("wt_mutant_subtraction_used") is False, "shard used WT subtraction")
        this_source = manifest.get("feature_source")
        _require(this_source in {"rde", "rde_network", "both"}, "unknown feature source")
        if feature_source is None:
            feature_source = this_source
        _require(feature_source == this_source, "feature source differs across shards")
        upstream = manifest.get("upstream")
        inputs = manifest.get("input")
        crystal = manifest.get("crystal")
        structure_contract = manifest.get("structure_contract")
        _require(isinstance(upstream, dict), "shard lacks upstream provenance")
        _require(isinstance(inputs, dict), "shard lacks input provenance")
        _require(isinstance(crystal, dict), "shard lacks crystal provenance")
        _require(
            isinstance(structure_contract, dict),
            "shard lacks shared structure-contract provenance",
        )
        provenance = {
            "upstream_repository": upstream.get("repository"),
            "upstream_commit": upstream.get("commit"),
            "upstream_license": upstream.get("license"),
            "rde_checkpoint_sha256": upstream.get("rde_checkpoint_sha256"),
            "rde_network_checkpoint_sha256": upstream.get(
                "rde_network_checkpoint_sha256"
            ),
            "rde_network_folds_emitted_separately": upstream.get(
                "rde_network_folds_emitted_separately"
            ),
            "row_manifest_sha256": inputs.get("row_manifest_sha256"),
            "pdb_sha256": inputs.get("pdb_sha256"),
            "canonical_schema": inputs.get("canonical_schema"),
            "crystal_chains": crystal.get("chains"),
            "patch_size": crystal.get("patch_size"),
            "designed_residue_count": crystal.get("designed_residue_count"),
            "designed_sidechain_cb_hidden": crystal.get(
                "designed_sidechain_cb_hidden"
            ),
            "designed_chi_hidden": crystal.get("designed_chi_hidden"),
            "assay_hla_identity_graft": crystal.get("assay_hla_identity_graft"),
            "patch_chain_counts": crystal.get("patch_chain_counts"),
            "patch_hla_residue_numbers": crystal.get("patch_hla_residue_numbers"),
            "patch_hla_residue_number_min": crystal.get(
                "patch_hla_residue_number_min"
            ),
            "patch_hla_residue_number_max": crystal.get(
                "patch_hla_residue_number_max"
            ),
            "assay_hla_identity_graft_patch_status": crystal.get(
                "assay_hla_identity_graft_patch_status"
            ),
            "patch_excludes_beta2m": crystal.get("patch_excludes_beta2m"),
            "patch_excludes_hla_after_assay_local_position_181": crystal.get(
                "patch_excludes_hla_after_assay_local_position_181"
            ),
            "fixed_crystal_approximation_omissions": crystal.get(
                "fixed_crystal_approximation_omissions"
            ),
            "structure_contract": structure_contract,
        }
        _require(
            provenance["upstream_commit"] == EXPECTED_RDE_REPOSITORY_COMMIT,
            "shard upstream commit changed",
        )
        _require(
            provenance["rde_checkpoint_sha256"] == EXPECTED_RDE_SHA256,
            "shard RDE checkpoint changed",
        )
        if feature_source in {"rde_network", "both"}:
            _require(
                provenance["rde_network_checkpoint_sha256"]
                == EXPECTED_RDE_NETWORK_SHA256,
                "shard RDE-Network checkpoint changed",
            )
            _require(
                provenance["rde_network_folds_emitted_separately"] == 3,
                "shard RDE-Network fold contract changed",
            )
        _require(provenance["patch_size"] == PATCH_SIZE, "shard patch size changed")
        _require(
            provenance["designed_residue_count"] == 7,
            "shard designed-residue count changed",
        )
        _require(
            provenance["designed_sidechain_cb_hidden"] is True
            and provenance["designed_chi_hidden"] is True,
            "shard retained reference designed-sidechain information",
        )
        observed_graft = provenance["assay_hla_identity_graft"]
        _require(isinstance(observed_graft, list), "shard lacks assay HLA graft")
        _require(
            [
                (
                    int(record.get("pdb_residue_number")),
                    record.get("crystal_amino_acid"),
                    record.get("assay_amino_acid"),
                )
                for record in observed_graft
            ]
            == list(HLA_ASSAY_IDENTITY_CORRECTIONS),
            "shard assay HLA identity graft changed",
        )
        _require(
            all(
                record.get("backbone_retained") is True
                and record.get("cb_retained_as_alanine_approximation") is True
                and record.get("chi_hidden") is True
                for record in observed_graft
            ),
            "shard assay HLA graft geometry/chi policy changed",
        )
        _require(
            provenance["patch_chain_counts"].get("B") == 0
            and sum(provenance["patch_chain_counts"].values()) == PATCH_SIZE
            and provenance["patch_excludes_beta2m"] is True
            and provenance[
                "patch_excludes_hla_after_assay_local_position_181"
            ]
            is True,
            "shard patch includes residues absent from the assay approximation",
        )
        _require(
            len(provenance["patch_hla_residue_numbers"])
            == provenance["patch_chain_counts"].get("A")
            and provenance["patch_hla_residue_number_min"]
            == min(provenance["patch_hla_residue_numbers"])
            and provenance["patch_hla_residue_number_max"]
            == max(provenance["patch_hla_residue_numbers"])
            and provenance["patch_hla_residue_number_max"] <= 181,
            "shard HLA patch composition changed",
        )
        graft_status = provenance["assay_hla_identity_graft_patch_status"]
        _require(
            isinstance(graft_status, list) and len(graft_status) == 2,
            "shard lacks patch-level HLA graft status",
        )
        _require(
            graft_status[0]
            == {
                "pdb_residue_number": 84,
                "retained_in_patch": False,
                "model_amino_acid": None,
                "chi_hidden": None,
            }
            and graft_status[1]
            == {
                "pdb_residue_number": 167,
                "retained_in_patch": True,
                "model_amino_acid": "A",
                "chi_hidden": True,
            },
            "patch-level assay HLA graft status changed",
        )
        _require(
            provenance["fixed_crystal_approximation_omissions"]
            == list(FIXED_CRYSTAL_APPROXIMATION_OMISSIONS),
            "shard fixed-crystal approximation statement changed",
        )
        if locked_provenance is None:
            locked_provenance = provenance
        _require(provenance == locked_provenance, "provenance differs across shards")
        shard = manifest.get("shard", {})
        fraction = int(shard.get("num_shards", -1))
        if shard_fraction is None:
            shard_fraction = fraction
        _require(fraction == shard_fraction, "num_shards differs across manifests")
        mapping = manifest.get("crystal", {}).get("residue_mapping")
        _require(isinstance(mapping, list), "shard lacks crystal residue mapping")
        site_to_patch = {
            (str(record.get("chain_id")), record.get("partner_sequence_position_1_based")):
            int(record.get("patch_index_0_based"))
            for record in mapping
            if record.get("designed") is True
        }
        _require(
            set(CANONICAL_DESIGNED_SITE_ORDER).issubset(site_to_patch),
            "shard omits a canonical designed site",
        )
        this_indices = tuple(
            site_to_patch[site] for site in CANONICAL_DESIGNED_SITE_ORDER
        )
        _require(len(set(this_indices)) == 7, "designed patch indices are not unique")
        if designed_patch_indices is None:
            designed_patch_indices = this_indices
        _require(
            this_indices == designed_patch_indices,
            "designed patch ordering differs across shards",
        )
        records = shard.get("chunks")
        _require(isinstance(records, list) and records, "shard has no chunk records")
        for record in records:
            path = directory / str(record.get("filename", ""))
            _require(path.is_file(), "recorded chunk is missing")
            _require(path.name == record.get("filename"), "unsafe chunk filename")
            _require(path.stat().st_size > 0, "empty chunk")
            _require(sha256_file(path) == record.get("sha256"), "chunk checksum changed")
            chunks.append(path)
        manifests.append(
            {
                "file": str(manifest_path.relative_to(shard_root)),
                "sha256": sha256_file(manifest_path),
                "rows": int(shard.get("rows", -1)),
            }
        )
    return (
        feature_source,
        chunks,
        manifests,
        designed_patch_indices,
        locked_provenance,
    )


def inspect_chunk_rows(chunks, canonical_by_index, feature_source):
    records = []
    for path in chunks:
        with np.load(str(path), allow_pickle=False) as archive:
            _require(set(archive.files) == expected_feature_keys(feature_source), "chunk fields changed")
            indices = archive["row_index"].astype(np.int64).tolist()
            _require(indices, "empty chunk")
            rows = []
            for index in indices:
                _require(index in canonical_by_index, "chunk row index is not canonical")
                rows.append(canonical_by_index[index])
            validate_chunk(path, rows, feature_source, PATCH_SIZE)
            for local_index, row in enumerate(rows):
                records.append((row.row_index, path, local_index, row))
    records.sort(key=lambda value: value[0])
    indices = [record[0] for record in records]
    _require(len(indices) == len(set(indices)), "duplicate row across chunks")
    return records


def _array_specs(first_chunk, feature_source):
    specs = {}
    with np.load(str(first_chunk), allow_pickle=False) as archive:
        for name in sorted(set(archive.files).difference(IGNORED_CHUNK_ARRAYS)):
            values = archive[name]
            kind = "mask" if name in MASK_NAMES else "feature"
            if kind == "mask":
                _require(values.dtype == np.bool_, "{} must be boolean".format(name))
                _require(values.shape[1:] == (PATCH_SIZE,), "{} shape changed".format(name))
            else:
                _require(np.issubdtype(values.dtype, np.floating), "{} must be floating".format(name))
                _require(
                    values.shape[1:] == (PATCH_SIZE, EXPECTED_CONTEXT_DIM),
                    "{} shape changed".format(name),
                )
            specs[name] = {
                "dtype": values.dtype,
                "tail_shape": tuple(values.shape[1:]),
                "kind": kind,
            }
    expected_names = expected_feature_keys(feature_source).difference(IGNORED_CHUNK_ARRAYS)
    _require(set(specs) == set(expected_names), "merged feature fields changed")
    return specs


def _with_designed_ordered_specs(specs, designed_patch_indices):
    output = {name: dict(spec) for name, spec in specs.items()}
    for name, spec in list(specs.items()):
        if spec["kind"] != "feature":
            continue
        derived_name = name + "_designed_ordered"
        output[derived_name] = {
            "dtype": spec["dtype"],
            "tail_shape": (len(designed_patch_indices), EXPECTED_CONTEXT_DIM),
            "kind": "feature",
            "source": name,
            "designed_patch_indices": tuple(designed_patch_indices),
        }
    return output


def run(args):
    started = time.time()
    shard_root = Path(args.shard_root).resolve()
    row_manifest = Path(args.row_manifest).resolve()
    _require(row_manifest.is_file(), "canonical rows are missing")
    canonical_rows = validate_canonical_payload(
        _read_json(row_manifest),
        expected_total=None if args.allow_partial else EXPECTED_ROWS,
    )
    canonical_by_index = {row.row_index: row for row in canonical_rows}
    (
        feature_source,
        chunks,
        source_manifests,
        designed_patch_indices,
        provenance,
    ) = discover_chunks(shard_root)
    _require(
        provenance["row_manifest_sha256"] == sha256_file(row_manifest),
        "shards were extracted from a different canonical row manifest",
    )
    records = inspect_chunk_rows(chunks, canonical_by_index, feature_source)
    if not args.allow_partial:
        _require(len(records) == EXPECTED_ROWS, "merged row count changed")
        _require([record[0] for record in records] == list(range(EXPECTED_ROWS)), "full rows are incomplete")
    else:
        _require(records, "partial pilot has no rows")
        _require(
            {record[3].split for record in records} == {"train", "eval"},
            "partial readout pilot must include train and eval rows",
        )

    output_dir = validate_private_output_path(args.output_dir)
    _require(not output_dir.exists(), "output archive already exists")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    incomplete = output_dir / "EXTRACTION_INCOMPLETE"
    incomplete.write_text("merge in progress\n", encoding="ascii")
    os.chmod(str(incomplete), 0o600)
    try:
        source_specs = _array_specs(chunks[0], feature_source)
        specs = _with_designed_ordered_specs(source_specs, designed_patch_indices)
        arrays = {}
        for name, spec in specs.items():
            path = output_dir / "{}.npy".format(name)
            arrays[name] = np.lib.format.open_memmap(
                str(path),
                mode="w+",
                dtype=spec["dtype"],
                shape=(len(records),) + spec["tail_shape"],
            )

        # A full eight-way extraction has roughly two thousand chunks.  Open
        # exactly one archive at a time rather than retaining file descriptors.
        writes_by_path = {}
        for output_index, (_, path, local_index, _) in enumerate(records):
            writes_by_path.setdefault(path, []).append((output_index, local_index))
        for path in sorted(writes_by_path, key=str):
            with np.load(str(path), allow_pickle=False) as archive:
                for output_index, local_index in writes_by_path[path]:
                    for name, spec in specs.items():
                        source_name = spec.get("source", name)
                        values = archive[source_name][local_index]
                        if "designed_patch_indices" in spec:
                            values = values[list(spec["designed_patch_indices"])]
                        arrays[name][output_index] = values
        for array in arrays.values():
            array.flush()
        del arrays

        for name in specs:
            os.chmod(str(output_dir / "{}.npy".format(name)), 0o600)
        selected_rows = [record[3] for record in records]
        metadata_path = output_dir / "metadata.csv"
        _write_metadata(metadata_path, selected_rows)

        contracts = {}
        for name, spec in specs.items():
            path = output_dir / "{}.npy".format(name)
            values = np.load(str(path), mmap_mode="r", allow_pickle=False)
            _require(values.shape == (len(records),) + spec["tail_shape"], "merged shape changed")
            if spec["kind"] == "feature":
                # Scan in bounded blocks rather than materializing a multi-GB array.
                for start in range(0, len(values), 256):
                    _require(bool(np.isfinite(values[start : start + 256]).all()), "non-finite merged feature")
            contracts[name] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "kind": spec["kind"],
            }

        row_ids = [row.row_id for row in selected_rows]
        manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "producer_schema_version": MERGE_SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supervision_fields_read": [],
            "row_count": len(records),
            "row_ids_sha256": _canonical_json_sha256(row_ids),
            "canonical_row_indices_sha256": _canonical_json_sha256(
                [record[0] for record in records]
            ),
            "feature_source": feature_source,
            "absolute_current_pair": True,
            "wt_mutant_subtraction_used": False,
            "upstream": {
                "repository": provenance["upstream_repository"],
                "commit": provenance["upstream_commit"],
                "license": provenance["upstream_license"],
                "rde_checkpoint_sha256": provenance["rde_checkpoint_sha256"],
                "rde_network_checkpoint_sha256": provenance[
                    "rde_network_checkpoint_sha256"
                ],
                "rde_network_folds_emitted_separately": provenance[
                    "rde_network_folds_emitted_separately"
                ],
            },
            "fixed_input": {
                "canonical_rows_sha256": provenance["row_manifest_sha256"],
                "canonical_schema": provenance["canonical_schema"],
                "pdb_sha256": provenance["pdb_sha256"],
                "crystal_chains": provenance["crystal_chains"],
                "patch_size": provenance["patch_size"],
                "designed_residue_count": provenance["designed_residue_count"],
                "designed_sidechain_cb_hidden": provenance[
                    "designed_sidechain_cb_hidden"
                ],
                "designed_chi_hidden": provenance["designed_chi_hidden"],
                "assay_hla_identity_graft": provenance[
                    "assay_hla_identity_graft"
                ],
                "patch_chain_counts": provenance["patch_chain_counts"],
                "patch_hla_residue_numbers": provenance[
                    "patch_hla_residue_numbers"
                ],
                "patch_hla_residue_number_min": provenance[
                    "patch_hla_residue_number_min"
                ],
                "patch_hla_residue_number_max": provenance[
                    "patch_hla_residue_number_max"
                ],
                "assay_hla_identity_graft_patch_status": provenance[
                    "assay_hla_identity_graft_patch_status"
                ],
                "patch_excludes_beta2m": provenance["patch_excludes_beta2m"],
                "patch_excludes_hla_after_assay_local_position_181": provenance[
                    "patch_excludes_hla_after_assay_local_position_181"
                ],
                "fixed_crystal_approximation_omissions": provenance[
                    "fixed_crystal_approximation_omissions"
                ],
                "structure_contract": provenance["structure_contract"],
            },
            "designed_feature_order": [
                {
                    "chain_id": chain_id,
                    "partner_sequence_position_1_based": position,
                    "displayed_sequence_position_1_based": (
                        position if chain_id == "H" else None
                    ),
                    "provider_crystal_aligned_label_1_based": (
                        AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED.get(position)
                        if chain_id == "H"
                        else None
                    ),
                    "source_patch_index_0_based": patch_index,
                }
                for (chain_id, position), patch_index in zip(
                    CANONICAL_DESIGNED_SITE_ORDER, designed_patch_indices
                )
            ],
            "metadata": {
                "file": metadata_path.name,
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
                "rows": len(records),
            },
            "arrays": contracts,
            "sources": {
                "canonical_rows_sha256": sha256_file(row_manifest),
                "shard_root_name": shard_root.name,
                "shard_manifests": source_manifests,
            },
            "runtime": {"elapsed_seconds": time.time() - started},
        }
        _atomic_json(output_dir / "manifest.json", manifest)
        incomplete.unlink()
        _require(_private_mode(output_dir) == "0700", "archive directory mode changed")
        print(
            json.dumps(
                {
                    "archive": str(output_dir),
                    "rows": len(records),
                    "arrays": sorted(contracts),
                    "feature_source": feature_source,
                },
                sort_keys=True,
            )
        )
        return manifest
    except Exception:
        # Preserve the sentinel and partial files for diagnosis/resume by a new
        # output path.  Never publish a partially written archive as complete.
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a small train+eval pilot archive rather than all 30,768 rows.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
