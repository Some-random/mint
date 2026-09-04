#!/usr/bin/env python3
"""Extract a small, label-free LibA ESMFold2 feature pilot.

The production extractor deliberately requires the complete canonical row
contract.  This companion pilot loads that same validated contract but runs a
caller-specified list of row indices through one persistent checkpoint.  It
never opens the weak-label or retention sidecars.  Repeating a row index tests
determinism; choosing matched rows tests whether peptide-only and Affibody-only
sequence changes alter the intended interface features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import extract_esmfold2_libb_features as extraction


SCHEMA_VERSION = "esmfold2-liba-feature-pilot-v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-index", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args(argv)


def _sequence_differences(left: str, right: str) -> list[dict[str, Any]]:
    if len(left) != len(right):
        raise ValueError("pilot sequences have different lengths")
    return [
        {
            "position_1_based": index,
            "left": left_amino_acid,
            "right": right_amino_acid,
        }
        for index, (left_amino_acid, right_amino_acid) in enumerate(
            zip(left, right), start=1
        )
        if left_amino_acid != right_amino_acid
    ]


def _feature_delta(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in extraction.SAVED_FEATURE_SPECS:
        difference = np.asarray(left[name], dtype=np.float32) - np.asarray(
            right[name], dtype=np.float32
        )
        result[name] = {
            "exactly_equal": bool(np.array_equal(left[name], right[name])),
            "mean_absolute_difference": float(np.mean(np.abs(difference))),
            "root_mean_square_difference": float(
                math.sqrt(float(np.mean(np.square(difference))))
            ),
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
        }
    return result


def _run(args: argparse.Namespace) -> None:
    if transformers.__version__ != extraction.EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "pilot requires transformers=={}; found {}".format(
                extraction.EXPECTED_TRANSFORMERS_VERSION, transformers.__version__
            )
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if len(args.row_index) < 2:
        raise ValueError("provide at least two --row-index values")

    workspace = extraction._configure_determinism()
    checkpoint = args.checkpoint.resolve()
    row_manifest = args.row_manifest.resolve()
    payload, canonical_rows = extraction._load_row_manifest(row_manifest)
    profile = extraction._profile_for_manifest(payload)
    if profile != extraction.LIBA_PROFILE:
        raise ValueError("this pilot accepts only the canonical LibA row schema")
    output_dir = extraction._validate_private_output_root(
        args.output_dir, output_prefix="esmfold2_liba_features_pilot_"
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite pilot output: {output_dir}")
    extraction._ensure_private_directory(output_dir)

    row_lookup = {row.row_index: row for row in canonical_rows}
    missing = sorted(set(args.row_index).difference(row_lookup))
    if missing:
        raise ValueError(f"pilot row indices are absent from the contract: {missing}")
    rows = [row_lookup[index] for index in args.row_index]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LibA ESMFold2 pilot requires CUDA")
    torch.cuda.set_device(device)
    checkpoint_contract = extraction._checkpoint_contract(checkpoint)
    model, load_seconds = extraction._load_model(checkpoint, device)

    started = time.perf_counter()
    bundles: list[dict[str, np.ndarray]] = []
    timings: list[dict[str, Any]] = []
    for invocation_index, row in enumerate(rows):
        print(
            f"pilot invocation={invocation_index} row={row.row_index} id={row.row_id}",
            flush=True,
        )
        bundle, timing = extraction._extract_one(
            model=model,
            row=row,
            device=device,
            seed=args.seed,
        )
        timing["invocation_index"] = invocation_index
        bundles.append(bundle)
        timings.append(timing)

    arrays: dict[str, np.ndarray] = {
        "invocation_index": np.arange(len(rows), dtype=np.int64),
        "row_index": np.asarray([row.row_index for row in rows], dtype=np.int64),
        "row_id": np.asarray([row.row_id for row in rows], dtype=np.str_),
    }
    for name in extraction.SAVED_FEATURE_SPECS:
        arrays[name] = np.stack([bundle[name] for bundle in bundles], axis=0)
    artifact_path = output_dir / "features.npz"
    extraction._atomic_npz_dump(artifact_path, arrays)

    comparisons = []
    for left_index in range(len(rows)):
        for right_index in range(left_index + 1, len(rows)):
            left = rows[left_index]
            right = rows[right_index]
            comparisons.append(
                {
                    "left_invocation": left_index,
                    "right_invocation": right_index,
                    "same_canonical_row": left.row_index == right.row_index,
                    "chain1_differences": _sequence_differences(
                        left.chain1_sequence, right.chain1_sequence
                    ),
                    "chain2_differences": _sequence_differences(
                        left.chain2_sequence, right.chain2_sequence
                    ),
                    "feature_delta": _feature_delta(
                        bundles[left_index], bundles[right_index]
                    ),
                }
            )

    duplicate_comparisons = [
        comparison
        for comparison in comparisons
        if comparison["same_canonical_row"]
    ]
    if not duplicate_comparisons:
        raise ValueError("repeat at least one row index to test deterministic extraction")
    for comparison in duplicate_comparisons:
        if not all(
            record["exactly_equal"]
            for record in comparison["feature_delta"].values()
        ):
            raise RuntimeError("repeated identical input produced different saved features")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": extraction._utc_now(),
        "library": profile.library,
        "row_manifest": {
            "path": str(row_manifest),
            "sha256": extraction._sha256_file(row_manifest),
            "schema_version": profile.row_manifest_schema,
        },
        "checkpoint": checkpoint_contract,
        "settings": {
            "seed_reset_before_every_forward": args.seed,
            "cublas_workspace_config": workspace,
            "device": str(device),
            "coordinate_diffusion": False,
        },
        "environment": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "rows": [
            {
                "invocation_index": invocation_index,
                "row_index": row.row_index,
                "row_id": row.row_id,
                "split": row.split,
                "sequence_pair_sha256": row.sequence_pair_sha256,
                "peptide_sequence": row.chain1_sequence[-9:],
                "affibody_design_code": "".join(
                    row.chain2_sequence[position - 1]
                    for position in profile.affibody_code_positions_1_based
                ),
            }
            for invocation_index, row in enumerate(rows)
        ],
        "timings": {
            "checkpoint_load_seconds": load_seconds,
            "feature_loop_seconds": time.perf_counter() - started,
            "per_invocation": timings,
        },
        "comparisons": comparisons,
        "artifact": {
            "filename": artifact_path.name,
            "sha256": extraction._sha256_file(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
        },
        "supervision_fields_read": [],
    }
    extraction._atomic_json_dump(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["timings"], indent=2, sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    _run(_parse_args(argv))


if __name__ == "__main__":
    main()
