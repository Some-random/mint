#!/usr/bin/env python3
"""Replay the exact batch-16 tails that produced the 120 LibB archive rows.

The original frozen-feature extraction used eight modulo shards, each with
3,846 canonical rows and batches of 16.  In every shard, the last 22 rows are
seven training identities followed by 15 label-free evaluation identities.
They were evaluated as one batch of 16 (7 train + 9 evaluation rows) and one
batch of 6 (6 evaluation rows).  Replaying only those aligned tails reproduces
the numerical context of all 120 evaluation representations without reading
selection labels or retention outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.score_libb_fixed_structure_candidates import (
    RDEScorer,
    SEEDS,
    StaBScorer,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
EXPECTED_TOTAL = 30_768
EXPECTED_TRAIN = 30_648
SHARDS = 8
ROWS_PER_SHARD = 3_846
TAIL_ROWS_PER_SHARD = 22
HISTORICAL_BATCH_SIZE = 16
SCORER_SOURCE = REPO_ROOT / "downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"missing runtime asset: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _readout_provenance(config: Path, checkpoint_dir: Path) -> dict:
    config = config.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    _require(checkpoints, f"no native-readout checkpoints found in {checkpoint_dir}")
    return {
        "checkpoint_directory": str(checkpoint_dir),
        "checkpoints": [
            {"filename": path.name, "sha256": sha256_file(path)}
            for path in checkpoints
        ],
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "readout_mode": "native_learned_projection",
    }


def _validate_exhaustive_provenance(
    manifest_path: Path,
    family: str,
    readout_provenance: dict,
) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("schema_version") == "libb-fixed-structure-candidate-scores-merged-v1",
        "unexpected exhaustive-score manifest schema",
    )
    _require(manifest.get("family") == family, "exhaustive-score family mismatch")
    source = manifest.get("source_chunks", {})
    scorer_source = _asset(SCORER_SOURCE)
    _require(
        source.get("scorer_producer")
        == {key: scorer_source[key] for key in ("path", "sha256")},
        "historical replay scorer source differs from exhaustive scoring",
    )
    _require(
        source.get("readout_provenance") == readout_provenance,
        "historical replay readout config/checkpoints differ from exhaustive scoring",
    )
    return {
        "manifest": _asset(manifest_path),
        "family": family,
        "scorer_producer_cross_bound": True,
        "readout_provenance_cross_bound": True,
    }


def _output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved != PRIVATE_ROOT and PRIVATE_ROOT in resolved.parents,
             "output must be a new directory below private_data")
    _require(not resolved.exists(), "output exists; refusing overwrite")
    return resolved


def _codes(chain1: str, chain2: str) -> tuple[str, str]:
    return chain1[-9:][3:5], "".join(chain2[index] for index in (5, 9, 12, 13, 16))


def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_row_index": [int(record["row_index"]) for record in records],
        "pair_uid": [str(record["row_id"]) for record in records],
        "peptide_design_code": [
            _codes(str(record["chain1_sequence"]), str(record["chain2_sequence"]))[0]
            for record in records
        ],
        "affibody_design_code": [
            _codes(str(record["chain1_sequence"]), str(record["chain2_sequence"]))[1]
            for record in records
        ],
        "chain1_sequence": [str(record["chain1_sequence"]) for record in records],
        "chain2_sequence": [str(record["chain2_sequence"]) for record in records],
        "sequence_pair_sha256": [
            str(record["sequence_pair_sha256"]) for record in records
        ],
    })


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("rde", "stab"), required=True)
    parser.add_argument("--canonical-rows", required=True, type=Path)
    parser.add_argument("--pdb", required=True, type=Path)
    parser.add_argument("--residue-mapping", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exhaustive-manifest", required=True, type=Path)
    parser.add_argument("--rde-root", type=Path)
    parser.add_argument("--rde-checkpoint", type=Path)
    parser.add_argument("--rde-network-checkpoint", type=Path)
    parser.add_argument("--rde-checkpoint-dir", type=Path)
    parser.add_argument("--rde-readout-config", type=Path)
    parser.add_argument("--stab-root", type=Path)
    parser.add_argument("--stab-checkpoint", type=Path)
    parser.add_argument("--stab-checkpoint-dir", type=Path)
    parser.add_argument("--stab-readout-config", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    _require(torch.cuda.is_available(), "CUDA is required")
    output = _output_path(args.output_dir)
    # Capture the two Python sources before any GPU work.  Re-hash them again
    # immediately before publication so an edit during a replay cannot be
    # misrepresented by a receipt produced only at the end of the run.
    producer_at_start = _asset(Path(__file__))
    scorer_at_start = _asset(SCORER_SOURCE)
    canonical_path = args.canonical_rows.expanduser().resolve()
    with canonical_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("schema_version") == "esmfold2-libb-canonical-rows-v1",
             "unexpected canonical-row schema")
    records = sorted(payload.get("rows", []), key=lambda record: int(record["row_index"]))
    _require(len(records) == EXPECTED_TOTAL,
             "canonical row count differs from the historical extraction")
    _require([int(record["row_index"]) for record in records] == list(range(EXPECTED_TOTAL)),
             "canonical row indices are not contiguous")
    _require(sum(str(record["split"]) == "eval" for record in records) == 120,
             "canonical evaluation count changed")
    split_counts = pd.Series([str(record["split"]) for record in records]).value_counts()
    _require(
        split_counts.to_dict() == {"train": EXPECTED_TRAIN, "eval": 120},
        "canonical rows must contain exactly 30,648 train and 120 eval rows",
    )

    device = torch.device(args.device)
    scorer = RDEScorer(args, device) if args.family == "rde" else StaBScorer(args, device)
    started = time.time()
    result_rows = []
    batch_receipts = []
    for shard in range(SHARDS):
        shard_records = [
            record for record in records if int(record["row_index"]) % SHARDS == shard
        ]
        _require(len(shard_records) == ROWS_PER_SHARD,
                 f"historical shard {shard} row count changed")
        tail = shard_records[-TAIL_ROWS_PER_SHARD:]
        _require(
            [str(record["split"]) for record in tail]
            == ["train"] * 7 + ["eval"] * 15,
            f"historical shard {shard} tail composition changed",
        )
        for offset in range(0, len(tail), HISTORICAL_BATCH_SIZE):
            block = tail[offset : offset + HISTORICAL_BATCH_SIZE]
            block_frame = _frame(block)
            probabilities = scorer.score(block_frame)
            _require(probabilities.shape == (len(block), len(SEEDS)),
                     "historical replay score shape changed")
            batch_receipts.append({
                "shard": shard,
                "tail_offset": offset,
                "batch_rows": len(block),
                "canonical_row_indices": [int(record["row_index"]) for record in block],
                "training_context_rows": sum(str(record["split"]) == "train" for record in block),
                "evaluation_rows": sum(str(record["split"]) == "eval" for record in block),
            })
            for index, record in enumerate(block):
                if str(record["split"]) != "eval":
                    continue
                peptide, affibody = _codes(
                    str(record["chain1_sequence"]), str(record["chain2_sequence"])
                )
                result = {
                    "canonical_row_index": int(record["row_index"]),
                    "pair_uid": str(record["row_id"]),
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "historical_shard": shard,
                    "historical_batch_size": len(block),
                    "historical_batch_position": index,
                    "sequence_pair_sha256": str(record["sequence_pair_sha256"]),
                }
                for column, seed in enumerate(SEEDS):
                    result[f"score_seed_{seed}"] = float(probabilities[index, column])
                result_rows.append(result)

    results = pd.DataFrame(result_rows).sort_values(
        "canonical_row_index", kind="mergesort"
    ).reset_index(drop=True)
    _require(len(results) == 120 and not results["pair_uid"].duplicated().any(),
             "historical replay did not produce 120 unique evaluation rows")
    probability_columns = [f"score_seed_{seed}" for seed in SEEDS]
    probabilities = results[probability_columns].to_numpy(float)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    results["score_mean_probability"] = probabilities.mean(axis=1)
    results["score_mean_logit"] = 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))
    results["score_seed_sd"] = probabilities.std(axis=1, ddof=1)
    results["model"] = scorer.name

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    predictions_path = staging / "historical_tail_batch_predictions.csv"
    results.to_csv(predictions_path, index=False)
    os.chmod(predictions_path, 0o600)
    family_assets = {}
    if args.family == "rde":
        readout_provenance = _readout_provenance(
            args.rde_readout_config, args.rde_checkpoint_dir
        )
        family_assets = {
            "vendor_checkpoint": _asset(args.rde_checkpoint),
            "vendor_network_checkpoint": _asset(args.rde_network_checkpoint),
            "readout_config": _asset(args.rde_readout_config),
            "readout_checkpoints": [
                _asset(path) for path in sorted(args.rde_checkpoint_dir.glob("*.pt"))
            ],
        }
    else:
        readout_provenance = _readout_provenance(
            args.stab_readout_config, args.stab_checkpoint_dir
        )
        family_assets = {
            "vendor_checkpoint": _asset(args.stab_checkpoint),
            "readout_config": _asset(args.stab_readout_config),
            "readout_checkpoints": [
                _asset(path) for path in sorted(args.stab_checkpoint_dir.glob("*.pt"))
            ],
        }
    exhaustive_provenance = _validate_exhaustive_provenance(
        args.exhaustive_manifest,
        args.family,
        readout_provenance,
    )
    producer_before_publish = _asset(Path(__file__))
    scorer_before_publish = _asset(SCORER_SOURCE)
    _require(
        producer_before_publish == producer_at_start,
        "historical replay source changed while the GPU replay was running",
    )
    _require(
        scorer_before_publish == scorer_at_start,
        "imported candidate scorer changed while the GPU replay was running",
    )
    manifest = {
        "schema_version": "libb-native-projection-historical-tail-batch-replay-v1",
        "family": args.family,
        "model": scorer.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "rows": 120,
        "seeds": list(SEEDS),
        "historical_batch_contract": {
            "shards": SHARDS,
            "rows_per_shard": ROWS_PER_SHARD,
            "tail_rows_per_shard": TAIL_ROWS_PER_SHARD,
            "batch_size": HISTORICAL_BATCH_SIZE,
            "first_tail_batch_composition": "7 training identities + 9 evaluation identities",
            "last_tail_batch_composition": "6 evaluation identities",
            "batch_receipts": batch_receipts,
        },
        "canonical_split_counts": {"train": EXPECTED_TRAIN, "eval": 120},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "inputs": {
            "canonical_rows": _asset(canonical_path),
            "pdb": _asset(args.pdb),
            "residue_mapping": _asset(args.residue_mapping),
            **family_assets,
        },
        "outcome_access": {
            "selection_labels_loaded": False,
            "retention_values_loaded": False,
            "binder_values_loaded": False,
            "canonical_split_role_loaded_for_batch_reconstruction": True,
        },
        "source_integrity": {
            "producer_at_start": producer_at_start,
            "producer_before_publish": producer_before_publish,
            "producer_unchanged_during_run": True,
            "imported_candidate_scorer_at_start": scorer_at_start,
            "imported_candidate_scorer_before_publish": scorer_before_publish,
            "imported_candidate_scorer_unchanged_during_run": True,
        },
        "producer": producer_at_start,
        "imported_candidate_scorer": scorer_at_start,
        "readout_provenance": readout_provenance,
        "exhaustive_score_cross_bind": exhaustive_provenance,
        "output": {
            "path": predictions_path.name,
            "rows": 120,
            "sha256": sha256_file(predictions_path),
        },
    }
    manifest_path = staging / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    os.replace(staging, output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
