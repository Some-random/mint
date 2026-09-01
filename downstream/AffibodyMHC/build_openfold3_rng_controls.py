#!/usr/bin/env python3
"""Build a 2x2 NNYYF control separating ref-conformer and model/MSA RNG."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from downstream.AffibodyMHC.build_openfold3_native_panel import (
    RUNNER_TEMPLATE,
    assert_no_pairing_or_templates,
    harden_private_tree,
    sha256,
)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-query-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--global-feature-seed", type=int, default=20260820)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    harden_private_tree(args.output_root)
    inputs = args.output_root / "inputs"
    runners = args.output_root / "runners"
    outputs = args.output_root / "outputs"
    for directory in (inputs, runners, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    base_document = json.loads(args.base_query_json.read_text())
    if len(base_document["queries"]) != 1:
        raise ValueError("Base input must contain one query")
    base_query = next(iter(base_document["queries"].values()))
    assert_no_pairing_or_templates(base_query)

    runner_paths = {}
    for model_seed in (42, 43):
        path = runners / f"model_seed_{model_seed}.yml"
        path.write_text(RUNNER_TEMPLATE.format(seed=model_seed))
        runner_paths[model_seed] = path.resolve()

    rows = []
    for ref_conformer_seed in (42, 43):
        for model_seed in (42, 43):
            job_id = (
                f"nnyyf_rng_control_ref{ref_conformer_seed}_model{model_seed}"
            )
            query_path = inputs / f"{job_id}.json"
            query_path.write_text(
                json.dumps(
                    {"seeds": [model_seed], "queries": {job_id: base_query}},
                    indent=2,
                )
                + "\n"
            )
            rows.append(
                {
                    "job_id": job_id,
                    "affibody_design_code": "NNYYF",
                    "replicate": "factorial_control",
                    "model_seed": model_seed,
                    "ref_conformer_seed": ref_conformer_seed,
                    "global_feature_seed": args.global_feature_seed,
                    "duplicate_control": 0,
                    "retention_percent": "83.77",
                    "pair_uid": "6f535097d1927fc71b91",
                    "sequence_pair_sha256": "2bf98970576cfe1e45fe5f0bf2e93bd35d82cd3a86f7808e6cf41c40820ee690",
                    "query_json": str(query_path.resolve()),
                    "query_json_sha256": sha256(query_path),
                    "runner_yaml": str(runner_paths[model_seed]),
                    "runner_yaml_sha256": sha256(runner_paths[model_seed]),
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
                "purpose": "2x2 NNYYF ref-conformer by model/MSA RNG control",
                "global_feature_seed": args.global_feature_seed,
                "ref_conformer_seeds": [42, 43],
                "model_seeds": [42, 43],
                "base_query_json": str(args.base_query_json.resolve()),
                "base_query_json_sha256": sha256(args.base_query_json),
                "manifest_sha256": sha256(manifest),
            },
            indent=2,
        )
        + "\n"
    )
    harden_private_tree(args.output_root)


if __name__ == "__main__":
    main()
