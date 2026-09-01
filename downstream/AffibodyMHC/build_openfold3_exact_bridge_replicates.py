#!/usr/bin/env python3
"""Build controlled paired-RNG repeats of the exact SMART NNYYF/MW bridge."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
from pathlib import Path

from downstream.AffibodyMHC.build_openfold3_native_panel import (
    RUNNER_TEMPLATE,
    harden_private_tree,
    sha256,
)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-query-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--replicate-seeds", type=int, nargs="+", default=[42, 43, 2746317213]
    )
    parser.add_argument("--global-feature-seed", type=int, default=20260820)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    harden_private_tree(args.output_root)
    inputs = args.output_root / "inputs"
    runners = args.output_root / "runners"
    outputs = args.output_root / "outputs"
    for directory in (inputs, runners, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    document = json.loads(args.base_query_json.read_text())
    if len(document["queries"]) != 1:
        raise ValueError("Base exact-construct JSON must contain one query")
    base_query = next(iter(document["queries"].values()))
    chain_ids = tuple(chain["chain_ids"][0] for chain in base_query["chains"])
    if chain_ids != ("S", "H"):
        raise ValueError(f"Expected exact S/H chains, found {chain_ids}")
    if base_query.get("use_paired_msas", True):
        raise ValueError("Paired MSA must be disabled")
    if len(base_query["chains"][0]["sequence"]) != 270:
        raise ValueError("Expected exact 270-aa SMART chain")
    if len(base_query["chains"][1]["sequence"]) != 58:
        raise ValueError("Expected exact 58-aa Affibody chain")
    if base_query["chains"][0]["sequence"][264:266] != "MW":
        raise ValueError("Expected designed peptide positions S265/S266 to be MW")

    rows = []
    for replicate, paired_seed in enumerate(args.replicate_seeds, start=1):
        runner = runners / f"model_seed_{paired_seed}.yml"
        runner.write_text(RUNNER_TEMPLATE.format(seed=paired_seed))
        job_id = (
            f"exact_smart_NNYYF_MW_rep{replicate}_"
            f"ref{paired_seed}_model{paired_seed}"
        )
        query = copy.deepcopy(base_query)
        query_path = inputs / f"{job_id}.json"
        query_path.write_text(
            json.dumps({"seeds": [paired_seed], "queries": {job_id: query}}, indent=2)
            + "\n"
        )
        rows.append(
            {
                "job_id": job_id,
                "affibody_design_code": "NNYYF",
                "replicate": replicate,
                "model_seed": paired_seed,
                "ref_conformer_seed": paired_seed,
                "global_feature_seed": args.global_feature_seed,
                "duplicate_control": 0,
                "retention_percent": "83.77",
                "pair_uid": "6f535097d1927fc71b91",
                "sequence_pair_sha256": "2bf98970576cfe1e45fe5f0bf2e93bd35d82cd3a86f7808e6cf41c40820ee690",
                "query_json": str(query_path.resolve()),
                "query_json_sha256": sha256(query_path),
                "runner_yaml": str(runner.resolve()),
                "runner_yaml_sha256": sha256(runner),
                "output_dir": str((outputs / job_id).resolve()),
            }
        )

    manifest = args.output_root / "job_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "input_provenance.json").write_text(
        json.dumps(
            {
                "condition": "exact supplied two-chain SMART assay input",
                "base_query_json": str(args.base_query_json.resolve()),
                "base_query_json_sha256": sha256(args.base_query_json),
                "global_feature_seed": args.global_feature_seed,
                "paired_replicate_seeds": args.replicate_seeds,
                "manifest_sha256": sha256(manifest),
                "paired_msa_used": False,
                "templates_used": False,
            },
            indent=2,
        )
        + "\n"
    )
    harden_private_tree(args.output_root)


if __name__ == "__main__":
    main()
