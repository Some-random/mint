#!/usr/bin/env python3
"""Run and attest a 640-row native scorer replay at production batch size."""

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
INPUT_DIR = REPO_ROOT / "private_data/prospective/libb_existing_targets_selection_missed_scoring_inputs_v1"
CANONICAL_ROWS = REPO_ROOT / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json"
PDB = REPO_ROOT / "nyeso_xx133_complex.pdb"
RESIDUE_MAPPING = REPO_ROOT / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1/residue_mapping.json"
BATCH_SIZE = {"rde": 128, "stab": 20}
SEEDS = [20260811, 20260812, 20260813, 20260814, 20260815]


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
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"missing asset: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved),
            "bytes": int(resolved.stat().st_size)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("rde", "stab"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exhaustive-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    _require(output != PRIVATE_ROOT and PRIVATE_ROOT in output.parents,
             "output must be a new directory below private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is required")
    producer_start = _asset(Path(__file__))
    scorer_start = _asset(SCORER)
    exhaustive_path = args.exhaustive_manifest.expanduser().resolve()
    exhaustive = json.loads(exhaustive_path.read_text(encoding="utf-8"))
    _require(exhaustive.get("schema_version")
             == "libb-fixed-structure-candidate-scores-merged-v1",
             "unexpected exhaustive manifest schema")
    _require(exhaustive.get("family") == args.family,
             "exhaustive manifest family mismatch")
    _require(exhaustive.get("source_chunks", {}).get("scorer_producer")
             == {key: scorer_start[key] for key in ("path", "sha256")},
             "current scorer differs from exhaustive scorer")

    batch_size = BATCH_SIZE[args.family]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    command = [
        sys.executable, str(SCORER),
        "--family", args.family,
        "--input-dir", str(INPUT_DIR),
        "--output-dir", str(staging),
        "--peptides", "AF",
        "--device", args.device,
        "--batch-size", str(batch_size),
        "--chunk-rows", "640",
        "--limit", "640",
        "--canonical-rows", str(CANONICAL_ROWS),
        "--pdb", str(PDB),
        "--residue-mapping", str(RESIDUE_MAPPING),
    ]
    if args.family == "rde":
        command.extend([
            "--rde-root", str(REPO_ROOT / "private_data/vendor/rde-ppi"),
            "--rde-checkpoint", str(REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/RDE.pt"),
            "--rde-network-checkpoint", str(REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/DDG_RDE_Network_30k.pt"),
            "--rde-checkpoint-dir", str(REPO_ROOT / "private_data/experiments/rde_libb_native_projection_deployment_v1/final_checkpoints"),
            "--rde-readout-config", str(REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_native_projection_readouts_v1.json"),
        ])
    else:
        command.extend([
            "--stab-root", str(REPO_ROOT / "private_data/vendor/StaB-ddG"),
            "--stab-checkpoint", str(REPO_ROOT / "private_data/vendor/StaB-ddG/model_ckpts/stabddg.pt"),
            "--stab-checkpoint-dir", str(REPO_ROOT / "private_data/experiments/stab_libb_native_projection_120_full_v1/final_checkpoints"),
            "--stab-readout-config", str(REPO_ROOT / "downstream/AffibodyMHC/configs/stab_libb_native_projection_readouts_v1.json"),
        ])
    started = time.time()
    subprocess.run(command, cwd=str(REPO_ROOT), check=True)

    worker_paths = sorted(staging.glob(f"worker_{args.family}_slice00of01_AF.json"))
    _require(len(worker_paths) == 1, "scorer worker receipt missing")
    worker = json.loads(worker_paths[0].read_text(encoding="utf-8"))
    _require(worker.get("producer") == {key: scorer_start[key] for key in ("path", "sha256")},
             "worker scorer provenance differs from invocation")
    _require(worker.get("readout_provenance")
             == exhaustive.get("source_chunks", {}).get("readout_provenance"),
             "worker readouts differ from exhaustive scoring")
    _require(worker.get("labels_read") is False and worker.get("retention_read") is False,
             "worker did not declare label-free inference")
    _require(worker.get("readout_seeds") == SEEDS, "worker seed set changed")
    chunks = sorted((staging / "peptide_AF").glob("*.csv.gz"))
    _require(len(chunks) == 1, "production replay score chunk missing")
    frame = pd.read_csv(chunks[0])
    _require(len(frame) == 640, "production replay did not score 640 rows")

    producer_end = _asset(Path(__file__))
    scorer_end = _asset(SCORER)
    _require(producer_start == producer_end, "replay wrapper changed during execution")
    _require(scorer_start == scorer_end, "candidate scorer changed during execution")
    manifest = {
        "schema_version": "libb-native-projection-production-batch-replay-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "family": args.family,
        "peptide": "AF",
        "rows": 640,
        "batch_size": batch_size,
        "chunk_rows": 640,
        "limit": 640,
        "command_argv": command,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
        "outcome_access": {
            "selection_labels_loaded": False,
            "retention_values_loaded": False,
            "binder_values_loaded": False,
        },
        "source_integrity": {
            "producer_at_start": producer_start,
            "producer_before_publish": producer_end,
            "producer_unchanged_during_run": True,
            "candidate_scorer_at_start": scorer_start,
            "candidate_scorer_before_publish": scorer_end,
            "candidate_scorer_unchanged_during_run": True,
        },
        "exhaustive_manifest": _asset(exhaustive_path),
        "worker_receipt": {
            "path": worker_paths[0].name,
            "sha256": _sha256(worker_paths[0]),
            "bytes": int(worker_paths[0].stat().st_size),
        },
        "score_chunk": {
            "path": str(chunks[0].relative_to(staging)),
            "sha256": _sha256(chunks[0]),
            "bytes": int(chunks[0].stat().st_size),
        },
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
