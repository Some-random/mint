#!/usr/bin/env python3
"""Create compact, model-facing CSV shards from the LibB candidate universe.

The prospective universe is stored as Parquet because it carries all raw-round
annotations.  The pinned RDE-PPI and StaB-ddG environments intentionally do
not contain PyArrow, so this utility writes one small gzip CSV per peptide with
only the outcome-free fields needed by frozen feature extractors.  The richer
Parquet rows remain the source used when the final wet-lab sheet is assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
)
EXPECTED_PEPTIDES = 12
EXPECTED_ROWS = 4_447_848


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_pair_sha256(chain1: str, chain2: str) -> str:
    return hashlib.sha256((chain1 + "|" + chain2).encode("ascii")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    candidate_dir = args.candidate_dir.resolve()
    output_dir = args.output_dir.resolve()
    private_root = (REPO_ROOT / "private_data").resolve()
    _require(candidate_dir.is_dir(), "candidate universe does not exist")
    _require(output_dir != private_root and private_root in output_dir.parents,
             "output must be below private_data")
    _require(not output_dir.exists(), "output exists; refusing overwrite")
    partitions = pd.read_csv(candidate_dir / "candidate_partitions.csv")
    _require(len(partitions) == EXPECTED_PEPTIDES, "candidate peptide count changed")
    _require(int(partitions["candidate_rows"].sum()) == EXPECTED_ROWS,
             "candidate row count changed")

    started = time.time()
    output_dir.mkdir(parents=True, mode=0o700)
    records = []
    seen_ids: set[str] = set()
    for record in partitions.sort_values("peptide_design_code").itertuples(index=False):
        source = candidate_dir / str(record.partition)
        frame = pd.read_parquet(source, columns=list(REQUIRED))
        _require(len(frame) == int(record.candidate_rows), "partition row count changed")
        _require(tuple(frame.columns) == REQUIRED, "candidate columns changed")
        _require(not frame["pair_uid"].duplicated().any(), "duplicate pair within peptide")
        overlap = seen_ids.intersection(frame["pair_uid"].astype(str))
        _require(not overlap, "duplicate pair across peptides")
        seen_ids.update(frame["pair_uid"].astype(str))
        _require(frame["peptide_design_code"].eq(record.peptide_design_code).all(),
                 "mixed peptide partition")

        frame = frame.rename(
            columns={
                "chain1_smart_hla_linker_peptide_sequence": "chain1_sequence",
                "chain2_affibody_sequence": "chain2_sequence",
            }
        )
        frame.insert(0, "candidate_row_index", range(len(frame)))
        frame["sequence_pair_sha256"] = [
            _sequence_pair_sha256(chain1, chain2)
            for chain1, chain2 in zip(frame["chain1_sequence"], frame["chain2_sequence"])
        ]
        _require(frame["chain1_sequence"].str.len().eq(270).all(), "chain1 length changed")
        _require(frame["chain2_sequence"].str.len().eq(58).all(), "chain2 length changed")
        destination = output_dir / f"peptide_{record.peptide_design_code}.csv.gz"
        frame.to_csv(destination, index=False, compression={"method": "gzip", "compresslevel": 6})
        os.chmod(destination, 0o600)
        records.append(
            {
                "peptide_design_code": str(record.peptide_design_code),
                "rows": int(len(frame)),
                "path": destination.name,
                "bytes": int(destination.stat().st_size),
                "sha256": _sha256_file(destination),
            }
        )

    _require(len(seen_ids) == EXPECTED_ROWS, "global candidate membership changed")
    manifest = {
        "schema_version": "libb-candidate-scoring-inputs-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "candidate_universe": str(candidate_dir),
        "candidate_manifest_sha256": _sha256_file(candidate_dir / "manifest.json"),
        "rows": EXPECTED_ROWS,
        "peptides": EXPECTED_PEPTIDES,
        "columns": list(frame.columns),
        "partitions": records,
        "contains_training_or_retention_labels": False,
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(output_dir / "manifest.json", 0o600)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
