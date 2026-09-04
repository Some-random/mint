#!/usr/bin/env python3
"""Validate and merge exhaustive LibB RDE/StaB score chunks.

The GPU scorer writes resumable gzip-CSV chunks from several disjoint workers.
This program is the fail-closed handoff from those chunks to the Parquet files
consumed by the prospective ensemble.  It verifies the worker receipts, every
chunk boundary, and the exact row-index/pair-ID mapping against the immutable
candidate universe before publishing any output.

The scorer's historical ``score_mean_logit`` column is named misleadingly: it
contains a probability obtained by applying sigmoid to the mean seed logit.
The merged schema calls that value ``{family}_probability`` and also records its
actual inverse-sigmoid as ``{family}_logit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
EXPECTED_MODELS = {
    "rde": "rde_network_designed_3fold_ensemble",
    "stab": "stab_designed_ordered",
}
CHUNK_NAME = re.compile(r"rows_(\d{6})_(\d{6})\.csv\.gz")
CHUNK_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    *("score_seed_%d" % seed for seed in SEEDS),
    "score_mean_probability",
    "score_mean_logit",
    "score_seed_sd",
    "model",
)
SCORING_INPUT_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_sha256(records: Sequence[Dict[str, Any]]) -> str:
    """Hash an ordered list of path/hash/size triples without file timestamps."""
    canonical = [
        {
            "relative_path": str(record["relative_path"]),
            "sha256": str(record["sha256"]),
            "bytes": int(record["bytes"]),
        }
        for record in records
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _private_output_path(path: Path) -> Path:
    private_root = (REPO_ROOT / "private_data").resolve()
    resolved = path.resolve()
    _require(private_root.is_dir(), "repository private_data directory is missing")
    _require(resolved != private_root and private_root in resolved.parents,
             "output must be a new directory below repository private_data")
    return resolved


def _read_universe_identity(path: Path, peptide: str) -> pd.DataFrame:
    universe = pd.read_parquet(
        path,
        columns=["pair_uid", "peptide_design_code", "affibody_design_code"],
    )
    _require(len(universe) > 0, "%s universe partition is empty" % peptide)
    universe["pair_uid"] = universe["pair_uid"].astype(str)
    universe["peptide_design_code"] = universe["peptide_design_code"].astype(str)
    universe["affibody_design_code"] = universe["affibody_design_code"].astype(str)
    _require(universe["peptide_design_code"].eq(peptide).all(),
             "%s universe partition contains another peptide" % peptide)
    _require(not universe["pair_uid"].duplicated().any(),
             "%s universe contains duplicate pair_uid" % peptide)
    _require(not universe["affibody_design_code"].duplicated().any(),
             "%s universe contains duplicate Affibody code" % peptide)
    universe.insert(0, "candidate_row_index", np.arange(len(universe), dtype=np.int64))
    return universe


def _validate_scoring_partition_against_universe(
    scoring_path: Path,
    universe_path: Path,
    peptide: str,
) -> int:
    """Verify the exact sequences that the GPU scorer consumed.

    Manifest hashes alone establish file identity, but not that the file's
    sequences were derived correctly from the candidate universe.  This check
    closes that semantic link before any score is published.
    """
    _require(scoring_path.is_file(), "candidate scoring input is missing: %s" % scoring_path)
    _require(universe_path.is_file(), "candidate universe input is missing: %s" % universe_path)
    scoring = pd.read_csv(scoring_path, dtype=str)
    _require(tuple(scoring.columns) == SCORING_INPUT_COLUMNS,
             "candidate scoring-input columns changed for %s" % peptide)
    universe = pd.read_parquet(
        universe_path,
        columns=[
            "pair_uid",
            "peptide_design_code",
            "affibody_design_code",
            "chain1_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence",
        ],
    ).rename(columns={
        "chain1_smart_hla_linker_peptide_sequence": "chain1_sequence",
        "chain2_affibody_sequence": "chain2_sequence",
    })
    _require(len(scoring) == len(universe),
             "scoring-input/universe row count differs for %s" % peptide)
    expected_indices = np.arange(len(scoring), dtype=np.int64)
    observed_indices = pd.to_numeric(
        scoring["candidate_row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    _require(np.array_equal(observed_indices, expected_indices),
             "candidate scoring indices are not contiguous for %s" % peptide)
    for column in (
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_sequence",
        "chain2_sequence",
    ):
        _require(scoring[column].astype(str).equals(universe[column].astype(str)),
                 "candidate scoring-input %s differs from universe for %s"
                 % (column, peptide))
    recomputed_hashes = [
        hashlib.sha256((chain1 + "|" + chain2).encode("ascii")).hexdigest()
        for chain1, chain2 in zip(
            scoring["chain1_sequence"].astype(str),
            scoring["chain2_sequence"].astype(str),
        )
    ]
    _require(scoring["sequence_pair_sha256"].astype(str).tolist() == recomputed_hashes,
             "candidate scoring-input sequence hash differs for %s" % peptide)
    return len(scoring)


def _read_and_validate_chunk(
    path: Path,
    peptide: str,
    expected_model: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    match = CHUNK_NAME.fullmatch(path.name)
    _require(match is not None, "unrecognized chunk filename: %s" % path)
    file_start = int(match.group(1))
    file_last = int(match.group(2))
    _require(file_last >= file_start, "invalid chunk range: %s" % path)
    frame = pd.read_csv(
        path,
        dtype={
            "pair_uid": str,
            "peptide_design_code": str,
            "affibody_design_code": str,
            "model": str,
        },
    )
    _require(tuple(frame.columns) == CHUNK_COLUMNS,
             "chunk schema changed: %s" % path)
    _require(len(frame) == file_last - file_start + 1,
             "chunk filename/row-count mismatch: %s" % path)
    expected_indices = np.arange(file_start, file_last + 1, dtype=np.int64)
    observed_indices = pd.to_numeric(
        frame["candidate_row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    _require(np.array_equal(observed_indices, expected_indices),
             "chunk rows are not exactly the filename interval: %s" % path)
    _require(frame["peptide_design_code"].eq(peptide).all(),
             "wrong peptide in chunk: %s" % path)
    _require(frame["model"].eq(expected_model).all(),
             "wrong model in chunk: %s" % path)
    _require(not frame["pair_uid"].duplicated().any(),
             "duplicate pair_uid inside chunk: %s" % path)

    seed_columns = ["score_seed_%d" % seed for seed in SEEDS]
    numeric_columns = seed_columns + [
        "score_mean_probability",
        "score_mean_logit",
        "score_seed_sd",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(values).all()), "non-finite score in chunk: %s" % path)
    probabilities = numeric[seed_columns].to_numpy(dtype=np.float64)
    _require(bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()),
             "seed probability outside [0,1] in chunk: %s" % path)
    for column in ("score_mean_probability", "score_mean_logit"):
        aggregate = numeric[column].to_numpy(dtype=np.float64)
        _require(bool(((aggregate >= 0.0) & (aggregate <= 1.0)).all()),
                 "%s outside [0,1] in chunk: %s" % (column, path))
    _require(bool((numeric["score_seed_sd"].to_numpy(dtype=np.float64) >= 0.0).all()),
             "negative score_seed_sd in chunk: %s" % path)

    # Recompute all three aggregate columns.  The 2e-6 absolute tolerance is
    # wider than float32-to-CSV rounding but far too narrow to hide a changed
    # aggregation rule.
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    seed_logits = np.log(clipped) - np.log1p(-clipped)
    expected_logit_probability = 1.0 / (1.0 + np.exp(-seed_logits.mean(axis=1)))
    checks = (
        (numeric["score_mean_probability"].to_numpy(dtype=np.float64),
         probabilities.mean(axis=1), "score_mean_probability"),
        (numeric["score_mean_logit"].to_numpy(dtype=np.float64),
         expected_logit_probability, "score_mean_logit"),
        (numeric["score_seed_sd"].to_numpy(dtype=np.float64),
         probabilities.std(axis=1, ddof=1), "score_seed_sd"),
    )
    for observed, expected, name in checks:
        _require(np.allclose(observed, expected, rtol=0.0, atol=2e-6),
                 "%s does not match seed scores in chunk: %s" % (name, path))

    return frame, {
        "relative_path": "",  # filled relative to the caller's chunk root
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "first_candidate_row_index": file_start,
        "last_candidate_row_index": file_last,
        "rows": int(len(frame)),
    }


def merge_peptide(
    universe_path: Path,
    peptide_chunk_dir: Path,
    chunk_root: Path,
    peptide: str,
    family: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Return a validated, index-ordered compact score table for one peptide."""
    _require(family in EXPECTED_MODELS, "family must be rde or stab")
    _require(universe_path.is_file(), "universe partition is missing: %s" % universe_path)
    _require(peptide_chunk_dir.is_dir(), "chunk directory is missing: %s" % peptide_chunk_dir)
    paths = sorted(peptide_chunk_dir.glob("*.csv.gz"))
    _require(paths, "no score chunks for peptide %s" % peptide)
    _require(
        len(paths) == len(list(peptide_chunk_dir.iterdir())),
        "unexpected non-chunk file below %s" % peptide_chunk_dir,
    )

    frames = []
    records = []
    for path in paths:
        frame, record = _read_and_validate_chunk(path, peptide, EXPECTED_MODELS[family])
        record["relative_path"] = str(path.resolve().relative_to(chunk_root.resolve()))
        frames.append(frame)
        records.append(record)
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.sort_values("candidate_row_index", kind="mergesort").reset_index(drop=True)
    observed_indices = scores["candidate_row_index"].to_numpy(dtype=np.int64)
    _require(not scores["candidate_row_index"].duplicated().any(),
             "duplicate candidate_row_index across %s chunks" % peptide)
    _require(not scores["pair_uid"].duplicated().any(),
             "duplicate pair_uid across %s chunks" % peptide)

    universe = _read_universe_identity(universe_path, peptide)
    expected_indices = universe["candidate_row_index"].to_numpy(dtype=np.int64)
    _require(np.array_equal(observed_indices, expected_indices),
             "%s chunks have a gap, extra row, or incomplete coverage" % peptide)
    for column in ("pair_uid", "peptide_design_code", "affibody_design_code"):
        _require(scores[column].astype(str).equals(universe[column].astype(str)),
                 "%s score/universe %s mapping mismatch" % (peptide, column))

    prefix = family
    output = scores[[
        "candidate_row_index",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
    ]].copy()
    output["candidate_row_index"] = output["candidate_row_index"].astype(np.int32)
    for seed in SEEDS:
        output["%s_probability_seed_%d" % (prefix, seed)] = scores[
            "score_seed_%d" % seed
        ].to_numpy(dtype=np.float32)
    output["%s_seed_mean_probability" % prefix] = scores[
        "score_mean_probability"
    ].to_numpy(dtype=np.float32)
    # Despite its old name, this source field is already a probability: it is
    # sigmoid(mean(seed logits)).  Publish an unambiguous probability plus its
    # true logit for downstream locked linear combinations.
    probability = scores["score_mean_logit"].to_numpy(dtype=np.float64)
    clipped_probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    output["%s_probability" % prefix] = probability.astype(np.float32)
    output["%s_logit" % prefix] = (
        np.log(clipped_probability) - np.log1p(-clipped_probability)
    ).astype(np.float32)
    output["%s_seed_probability_sd" % prefix] = scores[
        "score_seed_sd"
    ].to_numpy(dtype=np.float32)
    _require(len(output) == len(universe), "internal merged row-count error")
    _require(bool(np.isfinite(output.select_dtypes(include=[np.number])).to_numpy().all()),
             "non-finite value introduced while merging %s" % peptide)
    return output, records


def _validate_worker_manifests(
    chunk_dir: Path,
    family: str,
    expected_num_slices: int,
    partitions: pd.DataFrame,
    expected_readout_mode: str | None = None,
    expected_input_manifest_sha256: str | None = None,
) -> Dict[str, Any]:
    paths = sorted(chunk_dir.glob("worker_%s_slice*.json" % family))
    _require(paths, "no completed worker manifests found for %s" % family)
    expected_peptides = set(partitions["peptide_design_code"].astype(str))
    expected_rows = dict(zip(
        partitions["peptide_design_code"].astype(str),
        partitions["candidate_rows"].astype(int),
    ))
    expected_cells = {
        (peptide, slice_index)
        for peptide in expected_peptides
        for slice_index in range(expected_num_slices)
    }
    observed_cells = set()
    cell_contracts = {}
    cell_attestation_paths = {}
    input_hashes = set()
    provenance_contracts: Dict[str, Dict[str, Any]] = {}
    missing_provenance = 0
    producer_contracts: Dict[str, Dict[str, str]] = {}
    missing_producer = 0
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _require(payload.get("schema_version") == "libb-fixed-structure-candidate-scores-v1",
                 "wrong worker-manifest schema: %s" % path)
        _require(payload.get("family") == family, "wrong family in %s" % path)
        _require(payload.get("model") == EXPECTED_MODELS[family], "wrong model in %s" % path)
        _require(payload.get("readout_seeds") == list(SEEDS), "wrong seeds in %s" % path)
        _require(int(payload.get("num_slices", -1)) == expected_num_slices,
                 "wrong num_slices in %s" % path)
        slice_index = int(payload.get("slice_index", -1))
        _require(0 <= slice_index < expected_num_slices, "bad slice index in %s" % path)
        input_manifest_sha256 = str(payload.get("input_manifest_sha256", ""))
        input_hashes.add(input_manifest_sha256)
        provenance = payload.get("readout_provenance")
        if provenance is None:
            missing_provenance += 1
        else:
            _require(isinstance(provenance, dict), "malformed readout provenance in %s" % path)
            checkpoints = provenance.get("checkpoints")
            _require(isinstance(checkpoints, list) and checkpoints,
                     "readout checkpoint provenance is missing in %s" % path)
            normalized_provenance = {
                "readout_mode": str(provenance.get("readout_mode", "")),
                "config_path": str(provenance.get("config_path", "")),
                "config_sha256": str(provenance.get("config_sha256", "")),
                "checkpoint_directory": str(provenance.get("checkpoint_directory", "")),
                "checkpoints": sorted(
                    [
                        {
                            "filename": str(record.get("filename", "")),
                            "sha256": str(record.get("sha256", "")),
                        }
                        for record in checkpoints
                    ],
                    key=lambda record: record["filename"],
                ),
            }
            _require(len(normalized_provenance["config_sha256"]) == 64,
                     "invalid readout-config hash in %s" % path)
            _require(
                all(record["filename"] and len(record["sha256"]) == 64
                    for record in normalized_provenance["checkpoints"]),
                "invalid readout-checkpoint record in %s" % path,
            )
            if expected_readout_mode is not None:
                _require(
                    normalized_provenance["readout_mode"] == expected_readout_mode,
                    "readout mode in %s is %s, expected %s"
                    % (path, normalized_provenance["readout_mode"], expected_readout_mode),
                )
            canonical_provenance = json.dumps(
                normalized_provenance, sort_keys=True, separators=(",", ":")
            )
            provenance_contracts[canonical_provenance] = normalized_provenance
        producer = payload.get("producer")
        if producer is None:
            missing_producer += 1
        else:
            _require(isinstance(producer, dict), "malformed scorer producer in %s" % path)
            normalized_producer = {
                "path": str(producer.get("path", "")),
                "sha256": str(producer.get("sha256", "")),
            }
            _require(normalized_producer["path"] and len(normalized_producer["sha256"]) == 64,
                     "invalid scorer producer record in %s" % path)
            canonical_producer = json.dumps(
                normalized_producer, sort_keys=True, separators=(",", ":")
            )
            producer_contracts[canonical_producer] = normalized_producer
        peptide_records = payload.get("peptides")
        _require(isinstance(peptide_records, list) and peptide_records,
                 "nonempty peptide records missing in %s" % path)
        manifest_peptides = [str(row.get("peptide")) for row in peptide_records]
        _require(len(manifest_peptides) == len(set(manifest_peptides)),
                 "duplicate peptide receipt inside %s" % path)
        _require(set(manifest_peptides).issubset(expected_peptides),
                 "worker contains an unexpected peptide: %s" % path)
        relative_path = str(path.resolve().relative_to(chunk_dir.resolve()))
        for row in peptide_records:
            peptide = str(row["peptide"])
            cell = (peptide, slice_index)
            full_rows = expected_rows[peptide]
            expected_start = full_rows * slice_index // expected_num_slices
            expected_stop = full_rows * (slice_index + 1) // expected_num_slices
            _require(int(row.get("full_rows", -1)) == full_rows,
                     "worker full_rows changed for %s in %s" % (peptide, path))
            _require(int(row.get("slice_start", -1)) == expected_start and
                     int(row.get("slice_stop", -1)) == expected_stop and
                     int(row.get("rows", -1)) == expected_stop - expected_start,
                     "worker slice is incomplete for %s in %s" % (peptide, path))
            contract = (
                family,
                EXPECTED_MODELS[family],
                tuple(SEEDS),
                input_manifest_sha256,
                full_rows,
                expected_start,
                expected_stop,
                expected_stop - expected_start,
            )
            if cell in cell_contracts:
                _require(
                    cell_contracts[cell] == contract,
                    "conflicting duplicate worker receipt for peptide %s slice %d"
                    % cell,
                )
            else:
                cell_contracts[cell] = contract
            observed_cells.add(cell)
            cell_attestation_paths.setdefault(cell, []).append(relative_path)
        records.append({
            "relative_path": relative_path,
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "slice_index": slice_index,
            "peptides": sorted(manifest_peptides),
        })
    missing_cells = expected_cells - observed_cells
    extra_cells = observed_cells - expected_cells
    _require(not extra_cells,
             "worker receipt matrix contains unexpected peptide/slice cells: %s" % (
                 sorted(extra_cells),
             ))
    _require(not missing_cells,
             "worker receipt matrix is incomplete; missing peptide/slice cells: %s" % (
                 sorted(missing_cells),
             ))
    _require(len(input_hashes) == 1 and next(iter(input_hashes)),
             "workers do not share one non-empty scoring-input manifest hash")
    if expected_input_manifest_sha256 is not None:
        _require(
            next(iter(input_hashes)) == expected_input_manifest_sha256,
            "worker scoring-input manifest hash does not match the supplied "
            "scoring-input directory",
        )
    if provenance_contracts:
        _require(missing_provenance == 0,
                 "only some worker manifests contain readout provenance")
        _require(len(provenance_contracts) == 1,
                 "worker manifests do not share one readout configuration/checkpoint set")
    if expected_readout_mode is not None:
        _require(missing_provenance == 0 and len(provenance_contracts) == 1,
                 "native merge requires readout provenance in every worker manifest")
        _require(missing_producer == 0 and len(producer_contracts) == 1,
                 "native merge requires one scorer source in every worker manifest")
    receipt_attestations = [
        {
            "peptide": peptide,
            "slice_index": slice_index,
            "attestation_count": len(cell_attestation_paths[(peptide, slice_index)]),
            "manifest_paths": sorted(cell_attestation_paths[(peptide, slice_index)]),
        }
        for peptide, slice_index in sorted(expected_cells)
    ]
    return {
        "expected_num_slices": expected_num_slices,
        "expected_peptides": sorted(expected_peptides),
        "expected_receipt_cells": len(expected_cells),
        "validated_receipt_cells": len(observed_cells),
        "receipt_attestation_count": sum(
            row["attestation_count"] for row in receipt_attestations
        ),
        "duplicate_attestation_cells": sum(
            row["attestation_count"] > 1 for row in receipt_attestations
        ),
        "receipt_attestations": receipt_attestations,
        "scoring_input_manifest_sha256": next(iter(input_hashes)),
        "readout_provenance": (
            next(iter(provenance_contracts.values())) if provenance_contracts else None
        ),
        "scorer_producer": (
            next(iter(producer_contracts.values())) if producer_contracts else None
        ),
        "worker_manifests": sorted(
            records,
            key=lambda row: (row["slice_index"], row["relative_path"]),
        ),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("rde", "stab"), required=True)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument(
        "--scoring-input-dir",
        type=Path,
        required=True,
        help=(
            "Exact label-free scorer input whose manifest hash is attested by "
            "every worker receipt."
        ),
    )
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-num-slices", type=int, default=8)
    parser.add_argument(
        "--expected-readout-mode",
        help="Fail unless every worker attests to this identical readout mode/config/checkpoint set.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    universe_dir = args.universe_dir.resolve()
    scoring_input_dir = args.scoring_input_dir.resolve()
    chunk_dir = args.chunk_dir.resolve()
    output_dir = _private_output_path(args.output_dir)
    _require(universe_dir.is_dir(), "candidate universe directory is missing")
    _require(scoring_input_dir.is_dir(), "candidate scoring-input directory is missing")
    _require(chunk_dir.is_dir(), "score chunk directory is missing")
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.expected_num_slices > 0, "expected-num-slices must be positive")

    universe_manifest_path = universe_dir / "manifest.json"
    partitions_path = universe_dir / "candidate_partitions.csv"
    _require(universe_manifest_path.is_file(), "candidate-universe manifest is missing")
    _require(partitions_path.is_file(), "candidate partition index is missing")
    with universe_manifest_path.open("r", encoding="utf-8") as handle:
        universe_manifest = json.load(handle)
    _require(
        universe_manifest.get("schema_version")
        == "libb-existing-target-selection-missed-candidates-v1",
        "unexpected candidate-universe schema",
    )
    partitions = pd.read_csv(partitions_path)
    _require(len(partitions) == 12, "expected exactly 12 candidate-universe partitions")
    _require(not partitions["peptide_design_code"].astype(str).duplicated().any(),
             "duplicate peptide in candidate partition index")
    expected_total = int(partitions["candidate_rows"].sum())
    _require(expected_total == int(universe_manifest["outputs"]["candidate_rows"]),
             "candidate total differs between partition index and manifest")

    scoring_manifest_path = scoring_input_dir / "manifest.json"
    _require(scoring_manifest_path.is_file(), "candidate scoring-input manifest is missing")
    with scoring_manifest_path.open("r", encoding="utf-8") as handle:
        scoring_manifest = json.load(handle)
    _require(
        scoring_manifest.get("schema_version") == "libb-candidate-scoring-inputs-v1",
        "unexpected candidate scoring-input schema",
    )
    _require(
        scoring_manifest.get("contains_training_or_retention_labels") is False,
        "candidate scoring inputs do not explicitly exclude supervision columns",
    )
    _require(int(scoring_manifest.get("rows", -1)) == expected_total,
             "candidate scoring-input total differs from the universe")
    _require(
        scoring_manifest.get("candidate_manifest_sha256")
        == sha256_file(universe_manifest_path),
        "candidate scoring inputs do not point to the supplied universe manifest",
    )
    scoring_partitions = scoring_manifest.get("partitions")
    _require(isinstance(scoring_partitions, list) and len(scoring_partitions) == 12,
             "candidate scoring-input partition inventory changed")
    scoring_by_peptide = {
        str(record.get("peptide_design_code")): record
        for record in scoring_partitions
    }
    _require(set(scoring_by_peptide) == set(partitions["peptide_design_code"].astype(str)),
             "candidate scoring inputs do not cover the same peptides as the universe")
    validated_scoring_input_rows = 0
    for row in partitions.itertuples(index=False):
        peptide = str(row.peptide_design_code)
        record = scoring_by_peptide[peptide]
        input_path = scoring_input_dir / str(record.get("path", ""))
        universe_path = universe_dir / str(row.partition)
        _require(input_path.is_file(), "missing candidate scoring input for %s" % peptide)
        _require(universe_path.is_file(), "missing candidate universe input for %s" % peptide)
        _require(int(record.get("rows", -1)) == int(row.candidate_rows),
                 "candidate scoring-input row count changed for %s" % peptide)
        _require(sha256_file(input_path) == str(record.get("sha256", "")),
                 "candidate scoring-input hash changed for %s" % peptide)
        _require(sha256_file(universe_path) == str(row.partition_sha256),
                 "candidate-universe hash changed for %s" % peptide)
        validated_scoring_input_rows += _validate_scoring_partition_against_universe(
            input_path,
            universe_path,
            peptide,
        )
    _require(validated_scoring_input_rows == expected_total,
             "candidate scoring-input semantic validation is incomplete")
    worker_validation = _validate_worker_manifests(
        chunk_dir,
        args.family,
        args.expected_num_slices,
        partitions,
        expected_readout_mode=args.expected_readout_mode,
        expected_input_manifest_sha256=sha256_file(scoring_manifest_path),
    )
    if args.expected_readout_mode is not None:
        scorer_source = (
            REPO_ROOT / "downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py"
        ).resolve()
        attested_producer = worker_validation["scorer_producer"]
        _require(Path(attested_producer["path"]).resolve() == scorer_source,
                 "workers attest to an unexpected candidate scorer source path")
        _require(sha256_file(scorer_source) == attested_producer["sha256"],
                 "candidate scorer source changed after worker attestation")

    # Write into a private sibling staging directory.  The requested output
    # appears atomically only after all 12 partitions have passed validation.
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = output_dir.with_name(".%s.staging-%d" % (output_dir.name, os.getpid()))
    _require(not staging.exists(), "staging directory already exists: %s" % staging)
    staging.mkdir(mode=0o700)
    peptide_records = []
    all_chunk_records = []
    merged_total = 0
    for row in partitions.sort_values("peptide_design_code").itertuples(index=False):
        peptide = str(row.peptide_design_code)
        universe_path = universe_dir / str(row.partition)
        _require(sha256_file(universe_path) == str(row.partition_sha256),
                 "candidate-universe hash mismatch for %s" % peptide)
        merged, chunk_records = merge_peptide(
            universe_path,
            chunk_dir / ("peptide_%s" % peptide),
            chunk_dir,
            peptide,
            args.family,
        )
        _require(len(merged) == int(row.candidate_rows),
                 "candidate row count changed for %s" % peptide)
        output_name = "peptide_%s.parquet" % peptide
        output_path = staging / output_name
        merged.to_parquet(
            output_path,
            index=False,
            compression="zstd",
            row_group_size=65_536,
        )
        os.chmod(output_path, 0o600)
        output_hash = sha256_file(output_path)
        all_chunk_records.extend(chunk_records)
        peptide_records.append({
            "peptide_design_code": peptide,
            "candidate_rows": int(len(merged)),
            "universe_partition": str(row.partition),
            "universe_partition_sha256": str(row.partition_sha256),
            "source_chunk_count": int(len(chunk_records)),
            "source_chunk_collection_sha256": _collection_sha256(chunk_records),
            "output_file": output_name,
            "output_sha256": output_hash,
            "output_bytes": int(output_path.stat().st_size),
        })
        merged_total += len(merged)
        print(json.dumps({
            "family": args.family,
            "peptide": peptide,
            "rows_validated_and_merged": int(len(merged)),
            "chunks": int(len(chunk_records)),
        }), flush=True)

    _require(merged_total == expected_total, "merged candidate total is incomplete")
    manifest = {
        "schema_version": "libb-fixed-structure-candidate-scores-merged-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "family": args.family,
        "model": EXPECTED_MODELS[args.family],
        "readout_seeds": list(SEEDS),
        "candidate_universe": {
            "path": str(universe_dir),
            "manifest_sha256": sha256_file(universe_manifest_path),
            "candidate_partitions_csv_sha256": sha256_file(partitions_path),
            "candidate_rows": expected_total,
        },
        "candidate_scoring_inputs": {
            "path": str(scoring_input_dir),
            "manifest_sha256": sha256_file(scoring_manifest_path),
            "candidate_universe_manifest_sha256": str(
                scoring_manifest["candidate_manifest_sha256"]
            ),
            "candidate_rows": expected_total,
            "rows_semantically_validated_against_universe": int(
                validated_scoring_input_rows
            ),
            "contains_training_or_retention_labels": False,
        },
        "source_chunks": {
            "path": str(chunk_dir),
            "count": len(all_chunk_records),
            "collection_sha256": _collection_sha256(all_chunk_records),
            **worker_validation,
        },
        "outputs": {
            "candidate_rows": int(merged_total),
            "partitions": peptide_records,
        },
        "score_semantics": {
            "%s_probability" % args.family: (
                "sigmoid of the arithmetic mean of the five readout-seed logits; "
                "this was named score_mean_logit in the chunk files"
            ),
            "%s_logit" % args.family: (
                "inverse-sigmoid of %s_probability, for locked downstream ensembling"
                % args.family
            ),
            "%s_seed_mean_probability" % args.family: (
                "arithmetic mean of the five readout-seed probabilities"
            ),
            "%s_seed_probability_sd" % args.family: (
                "sample standard deviation across the five readout-seed probabilities"
            ),
        },
        "outcome_access": {
            "selection_labels_read": False,
            "retention_measurements_read": False,
        },
        "validation": {
            "candidate_row_index_exact_once": True,
            "pair_uid_exact_once": True,
            "universe_identity_and_order_match": True,
            "worker_scoring_input_manifest_matches_supplied_input": True,
            "scoring_input_manifest_links_to_supplied_universe": True,
            "scoring_input_identities_sequences_and_hashes_match_universe": True,
            "all_values_finite": True,
            "seed_aggregates_recomputed": True,
        },
    }
    manifest_path = staging / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    os.rename(str(staging), str(output_dir))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
