#!/usr/bin/env python3
"""Run one deterministic shard of single-query OpenFold3 jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def build_command(job: dict, python_executable: str, launcher: Path) -> list[str]:
    return [
        python_executable,
        str(launcher),
        "--global-feature-seed",
        job["global_feature_seed"],
        "--ref-conformer-seed",
        job["ref_conformer_seed"],
        "predict",
        "--query-json",
        job["query_json"],
        "--runner-yaml",
        job["runner_yaml"],
        "--num-diffusion-samples",
        "1",
        "--use-msa-server",
        "false",
        "--use-templates",
        "false",
        "--output-dir",
        job["output_dir"],
    ]


def main() -> None:
    # Preserve private-data permissions for model outputs produced by subprocesses.
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")

    with args.manifest.open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    selected = jobs[args.shard_index :: args.num_shards]
    logs_dir = args.manifest.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.chmod(0o700)
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(selected)} jobs",
        flush=True,
    )
    for index, job in enumerate(selected, start=1):
        output_dir = Path(job["output_dir"])
        print(
            f"[{index}/{len(selected)}] {job['job_id']} -> {output_dir}",
            flush=True,
        )
        launcher = Path(__file__).with_name("run_openfold3_seeded.py")
        command = build_command(job, sys.executable, launcher)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = job["global_feature_seed"]
        environment["OPENFOLD_REF_CONFORMER_SEED"] = job["ref_conformer_seed"]
        log_path = logs_dir / f"{job['job_id']}.log"
        status_path = logs_dir / f"{job['job_id']}.status.json"
        with log_path.open("w") as log_handle:
            log_handle.write("command: " + json.dumps(command) + "\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        log_path.chmod(0o600)
        status_path.write_text(
            json.dumps(
                {
                    "job_id": job["job_id"],
                    "command": command,
                    "exit_code": result.returncode,
                    "log": str(log_path.resolve()),
                },
                indent=2,
            )
            + "\n"
        )
        status_path.chmod(0o600)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)


if __name__ == "__main__":
    main()
