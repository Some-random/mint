#!/usr/bin/env python3
"""Extract frozen StaB/ProteinMPNN current-pair features for canonical LibB rows.

This program reads only the label-free canonical row JSON.  It never opens the
selection-label or retention files.  Multiple GPUs can run independent modulo
shards; use ``merge_stab_libb_current_pair_features.py`` afterward.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import stab_libb_current_pair_features as features


DEFAULT_VENDOR = features.PRIVATE_ROOT / "vendor" / "StaB-ddG"
DEFAULT_CHECKPOINT = DEFAULT_VENDOR / "model_ckpts" / "stabddg.pt"
DEFAULT_ROWS = (
    features.PRIVATE_ROOT
    / "derived"
    / "esmfold2_libb_canonical_rows_provider_revision_120_v1"
    / "rows.json"
)
DEFAULT_PDB = features.REPO_ROOT / "nyeso_xx133_complex.pdb"
DEFAULT_RESIDUE_MAPPING = features.DEFAULT_STRUCTURE_MAPPING


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-manifest", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--pdb-path", type=Path, default=DEFAULT_PDB)
    parser.add_argument(
        "--residue-mapping", type=Path, default=DEFAULT_RESIDUE_MAPPING
    )
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--context-mode",
        choices=features.CONTEXT_MODES,
        required=True,
        help="Prespecified fixed-crystal context; contexts must be archived separately.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an incomplete shard at its last atomically flushed batch.",
    )
    parser.add_argument(
        "--canonical-row-indices",
        default="",
        help="Comma-separated row indices for a small mapping/sensitivity pilot.",
    )
    return parser.parse_args(argv)


def _select_rows(rows, args):
    if args.canonical_row_indices:
        requested = [
            int(value.strip())
            for value in args.canonical_row_indices.split(",")
            if value.strip()
        ]
        if len(set(requested)) != len(requested):
            raise ValueError("duplicate pilot row index")
        by_index = {row.row_index: row for row in rows}
        missing = sorted(set(requested).difference(by_index))
        if missing:
            raise ValueError(f"canonical row indices do not exist: {missing}")
        return [by_index[index] for index in requested]
    return features.partition_rows(rows, args.shard_index, args.num_shards)


def main(argv=None):
    args = parse_args(argv)
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    payload, canonical_rows = features.load_canonical_rows(args.row_manifest)
    selected_rows = _select_rows(canonical_rows, args)
    if not selected_rows:
        raise ValueError("this extraction shard has no rows")

    started = time.perf_counter()
    template = features.build_template_bundle(
        args.pdb_path,
        args.vendor_root,
        args.context_mode,
        args.residue_mapping,
    )
    for row in selected_rows:
        features.validate_row_against_template(row, template)
    model = features.load_pinned_vendor(args.vendor_root, args.checkpoint)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_contract = features.archive_run_contract(
        template.context_mode,
        structure_mapping_sha256=template.structure_mapping_sha256,
        additional={
            "template_context_schema": features.CONTEXT_SCHEMA_VERSIONS[
                template.context_mode
            ],
            "upstream_repository": features.UPSTREAM_REPOSITORY,
            "upstream_commit": features.UPSTREAM_COMMIT,
            "checkpoint_sha256": features.sha256_file(args.checkpoint),
            "pdb_sha256": features.sha256_file(args.pdb_path),
            "row_manifest_sha256": features.sha256_file(args.row_manifest),
            "source_row_schema": payload["schema_version"],
            "selected_canonical_row_indices_sha256": features.canonical_json_sha256(
                [row.row_index for row in selected_rows]
            ),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "explicit_row_selection": bool(args.canonical_row_indices),
            "batch_size": args.batch_size,
            "device": str(device),
        },
    )

    if args.resume:
        writer = features.FeatureArchiveWriter.resume(
            args.output_dir, selected_rows, features.feature_specs(), run_contract
        )
    else:
        writer = features.FeatureArchiveWriter(
            args.output_dir, selected_rows, features.feature_specs(), run_contract
        )
    resumed_rows = writer.offset
    batch_times = []
    for batch_number, batch in enumerate(
        features.iter_batches(selected_rows[writer.offset :], args.batch_size), start=1
    ):
        batch_started = time.perf_counter()
        arrays = features.extract_feature_batch(model, batch, template, device)
        writer.append(arrays)
        writer.checkpoint()
        elapsed = time.perf_counter() - batch_started
        batch_times.append(elapsed)
        print(
            json.dumps(
                {
                    "batch": batch_number,
                    "rows_complete": writer.offset,
                    "rows_total": len(selected_rows),
                    "batch_seconds": round(elapsed, 4),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    total_seconds = time.perf_counter() - started
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    provenance = {
        **run_contract,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "proteinmpnn_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "runtime_seconds": total_seconds,
        "mean_batch_seconds": float(np.mean(batch_times)) if batch_times else None,
        "peak_gpu_memory_bytes": peak_gpu,
        "resumed_completed_rows": resumed_rows,
        "domain_residue_counts": {
            "complex": len(template.complex_reference_sequence),
            "partner1_fragment": len(template.pmhc_reference_sequence),
            "affibody_fragment": len(template.affibody_reference_sequence),
        },
    }
    manifest = writer.finish(provenance)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(Path(args.output_dir).resolve()),
                "rows": manifest["row_count"],
                "context_mode": template.context_mode,
                "runtime_seconds": round(total_seconds, 3),
                "peak_gpu_memory_bytes": peak_gpu,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
