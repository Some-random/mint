#!/usr/bin/env python3
"""Score the complete LibB prospective universe with the locked additive head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = 4_447_848


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--head-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    universe = args.universe_dir.resolve()
    output = args.output_dir.resolve()
    _require(universe.is_dir(), "candidate universe does not exist")
    _require(args.head_npz.is_file(), "deployment head does not exist")
    _require(not output.exists(), "output exists; refusing overwrite")
    with np.load(args.head_npz, allow_pickle=False) as archive:
        _require(str(archive["schema_version"][0]) == "libb-additive-mint-deployment-heads-v1",
                 "deployment-head schema changed")
        categories = np.asarray(archive["additive_categories"]).astype(str)
        coefficient = np.asarray(archive["additive_coef"], dtype=np.float64).reshape(7, 20)
        intercept = float(np.asarray(archive["additive_intercept"], dtype=float).reshape(-1)[0])
    _require(categories.shape == (7, 20) and coefficient.shape == (7, 20),
             "additive head shape changed")
    lookup = [dict(zip(categories[position], coefficient[position])) for position in range(7)]
    partitions = pd.read_csv(universe / "candidate_partitions.csv")
    _require(int(partitions["candidate_rows"].sum()) == EXPECTED_ROWS,
             "candidate universe size changed")
    output.mkdir(parents=True, mode=0o700)
    started = time.time()
    records = []
    for row in partitions.sort_values("peptide_design_code").itertuples(index=False):
        source = universe / str(row.partition)
        frame = pd.read_parquet(
            source,
            columns=["pair_uid", "peptide_design_code", "affibody_design_code"],
        )
        peptide = str(row.peptide_design_code)
        _require(len(frame) == int(row.candidate_rows), "candidate partition changed")
        _require(frame["peptide_design_code"].eq(peptide).all(), "mixed peptide partition")
        code = frame["peptide_design_code"].astype(str) + frame["affibody_design_code"].astype(str)
        logit = np.full(len(frame), intercept, dtype=np.float64)
        for position in range(7):
            value = code.str[position].map(lookup[position])
            _require(value.notna().all(), f"unknown amino acid at designed position {position}")
            logit += value.to_numpy(dtype=np.float64)
        score = np.empty_like(logit)
        positive = logit >= 0
        score[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
        exp_value = np.exp(logit[~positive])
        score[~positive] = exp_value / (1.0 + exp_value)
        result = frame.copy()
        result["additive_7site_logit"] = logit
        result["additive_7site_score"] = score
        destination = output / f"peptide_{peptide}.parquet"
        result.to_parquet(destination, index=False, compression="zstd")
        os.chmod(destination, 0o600)
        records.append(
            {
                "peptide": peptide,
                "rows": int(len(result)),
                "path": destination.name,
                "sha256": sha256_file(destination),
            }
        )
    manifest = {
        "schema_version": "libb-additive-candidate-scores-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "rows": int(sum(value["rows"] for value in records)),
        "head_npz": str(args.head_npz.resolve()),
        "head_sha256": sha256_file(args.head_npz),
        "candidate_manifest_sha256": sha256_file(universe / "manifest.json"),
        "partitions": records,
        "retention_or_training_labels_read": False,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(output / "manifest.json", 0o600)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
