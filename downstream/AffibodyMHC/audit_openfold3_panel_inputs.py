#!/usr/bin/env python3
"""Audit controlled OpenFold3 panel inputs before biological analysis."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from downstream.AffibodyMHC.analyze_openfold3_native_panel import (
    EXPECTED_REPLICATES,
    compare_batch_tensors,
    find_job_artifacts,
)
from downstream.AffibodyMHC.openfold3_distogram_utils import (
    harden_private_tree,
    sha256_file,
    validate_experiment_config,
    validate_query_document,
    write_json_private,
)


def canonical_query(query: dict) -> dict:
    return {
        "use_main_msas": query.get("use_main_msas"),
        "use_paired_msas": query.get("use_paired_msas"),
        "chains": [
            {
                "chain_ids": chain["chain_ids"],
                "sequence": chain["sequence"],
                "main_msa_file_paths": chain.get("main_msa_file_paths") or [],
                "paired_msa_file_paths": chain.get("paired_msa_file_paths") or [],
                "template_alignment_file_path": chain.get(
                    "template_alignment_file_path"
                ) or None,
                "template_cif_paths": chain.get("template_cif_paths") or [],
            }
            for chain in query["chains"]
        ],
    }


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if len(jobs) != 33:
        raise ValueError(f"Expected 33 jobs, found {len(jobs)}")
    provenance_path = args.manifest.parent / "input_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("Input-provenance manifest hash mismatch")

    expected_design = {
        (str(rep), str(ref), str(model)) for rep, ref, model in EXPECTED_REPLICATES
    }
    observed_design = {
        (job["replicate"], job["ref_conformer_seed"], job["model_seed"])
        for job in jobs
    }
    if observed_design != expected_design:
        raise ValueError(f"Unexpected paired-seed design: {observed_design}")

    artifacts_by_job = {}
    grouped_primary = defaultdict(dict)
    duplicates = {}
    query_output_matches = []
    for job in jobs:
        model_seed = int(job["model_seed"])
        query_path = Path(job["query_json"])
        runner_path = Path(job["runner_yaml"])
        if sha256_file(query_path) != job["query_json_sha256"]:
            raise ValueError(f"Query hash mismatch for {job['job_id']}")
        if sha256_file(runner_path) != job["runner_yaml_sha256"]:
            raise ValueError(f"Runner hash mismatch for {job['job_id']}")
        input_document = json.loads(query_path.read_text())
        input_query = validate_query_document(input_document, model_seed)
        artifacts = find_job_artifacts(job)
        artifacts_by_job[job["job_id"]] = artifacts
        output_root = Path(job["output_dir"])
        output_document = json.loads(
            (output_root / "inference_query_set.json").read_text()
        )
        output_query = validate_query_document(output_document, model_seed)
        query_output_matches.append(
            canonical_query(input_query) == canonical_query(output_query)
        )
        experiment = json.loads((output_root / "experiment_config.json").read_text())
        validate_experiment_config(experiment, model_seed)
        key = (job["affibody_design_code"], int(job["replicate"]))
        if int(job["duplicate_control"]):
            duplicates[key] = job
        else:
            grouped_primary[job["affibody_design_code"]][int(job["replicate"])] = job

    if len(grouped_primary) != 10 or any(len(value) != 3 for value in grouped_primary.values()):
        raise ValueError("Expected 10 variants with three primary jobs each")
    if set(duplicates) != {("NNYYF", 1), ("NNYYF", 2), ("NNYYF", 3)}:
        raise ValueError(f"Unexpected duplicate controls: {sorted(duplicates)}")

    duplicate_checks = []
    for key, duplicate_job in sorted(duplicates.items()):
        primary_job = grouped_primary[key[0]][key[1]]
        primary_batch = torch.load(
            artifacts_by_job[primary_job["job_id"]]["batch"],
            map_location="cpu",
            weights_only=False,
        )
        duplicate_batch = torch.load(
            artifacts_by_job[duplicate_job["job_id"]]["batch"],
            map_location="cpu",
            weights_only=False,
        )
        comparison = compare_batch_tensors(primary_batch, duplicate_batch)
        duplicate_checks.append(
            {
                "affibody_design_code": key[0],
                "replicate": key[1],
                "primary_job_id": primary_job["job_id"],
                "duplicate_job_id": duplicate_job["job_id"],
                **comparison,
                "ref_pos_exact": bool(
                    torch.equal(primary_batch["ref_pos"], duplicate_batch["ref_pos"])
                ),
            }
        )
        del primary_batch, duplicate_batch

    cross_replicate_checks = []
    for code, by_replicate in sorted(grouped_primary.items()):
        for left, right in ((1, 2), (1, 3), (2, 3)):
            left_job = by_replicate[left]
            right_job = by_replicate[right]
            left_batch = torch.load(
                artifacts_by_job[left_job["job_id"]]["batch"],
                map_location="cpu",
                weights_only=False,
            )
            right_batch = torch.load(
                artifacts_by_job[right_job["job_id"]]["batch"],
                map_location="cpu",
                weights_only=False,
            )
            comparison = compare_batch_tensors(left_batch, right_batch)
            cross_replicate_checks.append(
                {
                    "affibody_design_code": code,
                    "replicate_pair": f"{left}-{right}",
                    **comparison,
                    "non_ref_pos_unequal_paths": [
                        path
                        for path in comparison["unequal_paths"]
                        if path != "ref_pos"
                    ],
                }
            )
            del left_batch, right_batch

    result = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "input_provenance": str(provenance_path.resolve()),
        "input_provenance_manifest_hash_verified": True,
        "n_jobs": len(jobs),
        "all_jobs_successful": True,
        "all_output_queries_match_input_queries": all(query_output_matches),
        "duplicate_checks": duplicate_checks,
        "same_variant_cross_replicate_checks": cross_replicate_checks,
        "analysis_allowed": all(query_output_matches),
        "note": (
            "This is an input/batch audit only. Feature correlations and retention "
            "labels were not loaded or calculated."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json_private(args.output_json, result)
    harden_private_tree(args.output_json.parent)
    print(json.dumps({
        "n_jobs": result["n_jobs"],
        "analysis_allowed": result["analysis_allowed"],
        "duplicate_unequal_paths": [
            row["unequal_paths"] for row in duplicate_checks
        ],
        "cross_replicate_non_ref_paths": sorted({
            path
            for row in cross_replicate_checks
            for path in row["non_ref_pos_unequal_paths"]
        }),
    }, indent=2))


if __name__ == "__main__":
    main()
