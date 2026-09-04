#!/usr/bin/env python3
"""Attest an RDE batch-16 replay of the primary menu's top-50 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SCORER = REPO_ROOT / "downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py"
PEPTIDES = "AF,AH,DP,EA,EL,LL,LV,MW,NF,PH,TL,VV"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict:
    path = path.expanduser().resolve()
    _require(path.is_file(), f"missing asset: {path}")
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--exhaustive-manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    _require(output != PRIVATE_ROOT and PRIVATE_ROOT in output.parents,
             "output must be a new directory below private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required")
    input_dir = args.input_dir.expanduser().resolve()
    input_manifest = input_dir / "manifest.json"
    _require(input_manifest.is_file(), "boundary input manifest missing")
    producer_start = _asset(Path(__file__))
    scorer_start = _asset(SCORER)
    exhaustive_path = args.exhaustive_manifest.expanduser().resolve()
    exhaustive = json.loads(exhaustive_path.read_text(encoding="utf-8"))
    _require(exhaustive.get("family") == "rde", "expected RDE exhaustive manifest")
    _require(exhaustive.get("source_chunks", {}).get("scorer_producer")
             == {key: scorer_start[key] for key in ("path", "sha256")},
             "current scorer differs from exhaustive scorer")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    command = [
        sys.executable, str(SCORER), "--family", "rde",
        "--input-dir", str(input_dir), "--output-dir", str(staging),
        "--peptides", PEPTIDES, "--device", args.device,
        "--batch-size", "16", "--chunk-rows", "50",
        "--canonical-rows", str(REPO_ROOT / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json"),
        "--pdb", str(REPO_ROOT / "nyeso_xx133_complex.pdb"),
        "--residue-mapping", str(REPO_ROOT / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1/residue_mapping.json"),
        "--rde-root", str(REPO_ROOT / "private_data/vendor/rde-ppi"),
        "--rde-checkpoint", str(REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/RDE.pt"),
        "--rde-network-checkpoint", str(REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/DDG_RDE_Network_30k.pt"),
        "--rde-checkpoint-dir", str(REPO_ROOT / "private_data/experiments/rde_libb_native_projection_deployment_v1/final_checkpoints"),
        "--rde-readout-config", str(REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_native_projection_readouts_v1.json"),
    ]
    started = time.time()
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)
    workers = sorted(staging.glob("worker_rde_slice00of01_*.json"))
    _require(len(workers) == 1, "boundary scorer receipt missing")
    worker = json.loads(workers[0].read_text(encoding="utf-8"))
    _require(worker.get("producer") == {key: scorer_start[key] for key in ("path", "sha256")},
             "boundary scorer source differs from invocation")
    _require(worker.get("readout_provenance")
             == exhaustive.get("source_chunks", {}).get("readout_provenance"),
             "boundary readouts differ from exhaustive scoring")
    _require(worker.get("labels_read") is False and worker.get("retention_read") is False,
             "boundary scorer did not declare label-free inference")
    chunks = sorted(staging.glob("peptide_*/*.csv.gz"))
    _require(len(chunks) == 12, "boundary replay does not have 12 chunks")
    _require(sum(len(pd.read_csv(path)) for path in chunks) == 600,
             "boundary replay does not have 600 rows")
    producer_end = _asset(Path(__file__))
    scorer_end = _asset(SCORER)
    _require(producer_start == producer_end, "boundary wrapper changed during execution")
    _require(scorer_start == scorer_end, "candidate scorer changed during execution")
    manifest = {
        "schema_version": "libb-native-projection-primary-boundary-replay-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "family": "rde", "peptides": PEPTIDES.split(","), "rows": 600,
        "rows_per_peptide": 50, "batch_size": 16, "chunk_rows": 50,
        "command_argv": command,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": args.device},
        "outcome_access": {"selection_labels_loaded": False,
                           "retention_values_loaded": False,
                           "binder_values_loaded": False},
        "source_integrity": {
            "producer_at_start": producer_start,
            "producer_before_publish": producer_end,
            "producer_unchanged_during_run": True,
            "candidate_scorer_at_start": scorer_start,
            "candidate_scorer_before_publish": scorer_end,
            "candidate_scorer_unchanged_during_run": True,
        },
        "input_manifest": _asset(input_manifest),
        "exhaustive_manifest": _asset(exhaustive_path),
        "worker_receipt": {"path": workers[0].name, "sha256": _sha256(workers[0]),
                           "bytes": workers[0].stat().st_size},
        "score_chunks": [
            {"path": str(path.relative_to(staging)), "sha256": _sha256(path),
             "bytes": path.stat().st_size, "rows": 50}
            for path in chunks
        ],
    }
    manifest_path = staging / "invocation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    os.replace(staging, output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
