#!/usr/bin/env python3
"""Write the corrected 120-cell LibB panel as label-free scoring inputs.

This is a parity fixture for prospective scorers.  It deliberately copies
only identities and sequences; retention and binder columns never enter the
output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--canonical-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError("output exists; refusing overwrite")
    # Read only the identity columns.  The source panel also contains retention
    # outcomes, but a scorer-parity fixture must not load those values at all.
    identity_columns = [
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
    ]
    panel_path = args.panel.expanduser().resolve()
    canonical_path = args.canonical_rows.expanduser().resolve()
    panel = pd.read_csv(panel_path, usecols=identity_columns)
    with canonical_path.open(encoding="utf-8") as handle:
        canonical = pd.DataFrame(json.load(handle)["rows"])
    canonical = canonical.loc[canonical["split"].eq("eval")].copy()
    if len(canonical) != 120:
        raise ValueError("canonical evaluation roster must have 120 rows")
    identity = panel.merge(
        canonical[
            ["row_id", "chain1_sequence", "chain2_sequence", "sequence_pair_sha256"]
        ],
        left_on="pair_uid",
        right_on="row_id",
        validate="one_to_one",
    ).drop(columns="row_id")
    if len(identity) != 120:
        raise ValueError("panel/canonical identity join is incomplete")
    args.output_dir.mkdir(parents=True, mode=0o700)
    records = []
    for peptide, frame in identity.groupby("peptide_design_code", sort=True):
        frame = frame.sort_values("affibody_design_code").reset_index(drop=True)
        frame.insert(0, "candidate_row_index", range(len(frame)))
        path = args.output_dir / f"peptide_{peptide}.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        os.chmod(path, 0o600)
        records.append({
            "peptide": peptide,
            "rows": len(frame),
            "path": path.name,
            "sha256": sha256_file(path),
        })
    partition_digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        partition_digest.update(str(record["path"]).encode("utf-8"))
        partition_digest.update(b"\0")
        partition_digest.update(bytes.fromhex(str(record["sha256"])))
        partition_digest.update(b"\0")
    payload = {
        "schema_version": "libb-panel-label-free-scoring-inputs-v1",
        "rows": 120,
        "partitions": records,
        "partition_bundle_sha256": partition_digest.hexdigest(),
        "panel_columns_read": identity_columns,
        "retention_columns_read": False,
        "binder_columns_read": False,
        "retention_columns_written": False,
        "input_sources": {
            "corrected_panel": {
                "path": str(panel_path),
                "sha256": sha256_file(panel_path),
            },
            "canonical_rows": {
                "path": str(canonical_path),
                "sha256": sha256_file(canonical_path),
            },
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    with (args.output_dir / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(args.output_dir / "manifest.json", 0o600)


if __name__ == "__main__":
    main()
