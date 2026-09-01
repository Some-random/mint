#!/usr/bin/env python3
"""Build reproducible single-query OpenFold3 jobs for the LibB/MW panel.

This is the approximate native-pMHC condition: crystal-calibrated HLA, beta2m,
and peptide chains plus the measured Affibody sequence. It is deliberately not
the exact two-chain SMART assay construct.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


AFFIBODY_DESIGN_POSITIONS = (6, 10, 13, 14, 17)
PANEL_SELECTION = "library=LibB, peptide_design_code=MW, measurement_missing=0"


RUNNER_TEMPLATE = """experiment_settings:
  seeds:
    - {seed}
  use_msa_server: false
  use_templates: false

pl_trainer_args:
  accelerator: gpu
  devices: 1
  num_nodes: 1

model_update:
  presets:
    - predict
    - low_mem
  custom:
    settings:
      memory:
        eval:
          use_cueq_triangle_kernels: true
          use_deepspeed_evo_attention: false
          use_triton_triangle_kernels: false

data_module_args:
  batch_size: 1
  num_workers: 0
  num_workers_validation: 0

output_writer_settings:
  structure_format: cif
  full_confidence_output_format: npz
  write_features: true
  write_latent_outputs: true
  write_full_confidence_scores: false
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harden_private_tree(root: Path) -> None:
    """Force private-data convention even when ``root`` already existed."""
    root.chmod(0o700)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def assert_no_pairing_or_templates(query: dict) -> None:
    if query.get("use_paired_msas", True):
        raise ValueError("use_paired_msas must be false")
    for chain in query["chains"]:
        forbidden = (
            "paired_msa_file_paths",
            "template_alignment_file_path",
            "template_entry_chain_ids",
            "template_cif_paths",
            "template_cif_chain_ids",
        )
        populated = [key for key in forbidden if chain.get(key)]
        if populated:
            raise ValueError(
                f"Chain {chain['chain_ids']} has forbidden inputs: {populated}"
            )


def main() -> None:
    # These inputs contain unpublished sequences and retention measurements.
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--replicate-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--duplicate-code", default="NNYYF")
    parser.add_argument("--global-feature-seed", type=int, default=20260820)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    harden_private_tree(args.output_root)
    snapshot_dir = args.output_root / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = snapshot_dir / args.source_csv.name
    reference_snapshot = snapshot_dir / args.reference_json.name
    # copyfile intentionally does not preserve a potentially permissive source mode.
    shutil.copyfile(args.source_csv, source_snapshot)
    shutil.copyfile(args.reference_json, reference_snapshot)
    source_snapshot.chmod(0o600)
    reference_snapshot.chmod(0o600)

    reference_doc = json.loads(reference_snapshot.read_text())
    if len(reference_doc["queries"]) != 1:
        raise ValueError("Reference JSON must contain one query")
    reference_query = next(iter(reference_doc["queries"].values()))
    chain_map = {
        chain["chain_ids"][0]: chain for chain in reference_query["chains"]
    }
    if set(chain_map) != {"A", "B", "P", "H"}:
        raise ValueError(f"Expected A/B/P/H chains, found {sorted(chain_map)}")
    if chain_map["P"]["sequence"] != "SLLMWITQV":
        raise ValueError("Reference query is not the MW peptide")
    assert_no_pairing_or_templates(reference_query)

    with source_snapshot.open(newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row["library"] == "LibB"
            and row["peptide_design_code"] == "MW"
            and row["measurement_missing"] == "0"
        ]
    if len(selected) != 10:
        raise ValueError(f"Expected 10 measured LibB/MW rows, found {len(selected)}")
    if sum(row["affibody_design_code"] == args.duplicate_code for row in selected) != 1:
        raise ValueError("Duplicate-control code must identify exactly one row")

    inputs_dir = args.output_root / "inputs"
    msa_dir = args.output_root / "query_only_affibody_msas"
    runners_dir = args.output_root / "runners"
    outputs_dir = args.output_root / "outputs"
    for directory in (inputs_dir, msa_dir, runners_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runner_paths = {}
    for model_seed in args.replicate_seeds:
        runner_path = runners_dir / f"model_seed_{model_seed}.yml"
        runner_path.write_text(RUNNER_TEMPLATE.format(seed=model_seed))
        runner_paths[model_seed] = runner_path.resolve()

    manifest = []
    for row in selected:
        code = row["affibody_design_code"]
        sequence = row["chain2_affibody_sequence"]
        observed_code = "".join(
            sequence[position - 1] for position in AFFIBODY_DESIGN_POSITIONS
        )
        if observed_code != code:
            raise ValueError(f"{code}: Affibody sequence encodes {observed_code}")

        code_msa_dir = msa_dir / code
        code_msa_dir.mkdir(parents=True, exist_ok=True)
        affibody_msa_path = code_msa_dir / "colabfold_main.a3m"
        affibody_msa_path.write_text(f">101\n{sequence}\n")

        for replicate_index, paired_seed in enumerate(args.replicate_seeds, start=1):
            model_seed = paired_seed
            ref_conformer_seed = paired_seed
            repeats = (False, True) if code == args.duplicate_code else (False,)
            for duplicate in repeats:
                suffix = "_duplicate" if duplicate else ""
                job_id = (
                    f"native_libb_mw_{code}_rep{replicate_index}_"
                    f"ref{ref_conformer_seed}_model{model_seed}{suffix}"
                )
                query = copy.deepcopy(reference_query)
                variant_chain_map = {
                    chain["chain_ids"][0]: chain for chain in query["chains"]
                }
                variant_chain_map["H"]["sequence"] = sequence
                variant_chain_map["H"]["main_msa_file_paths"] = [
                    str(affibody_msa_path.resolve())
                ]
                assert_no_pairing_or_templates(query)
                query_path = inputs_dir / f"{job_id}.json"
                query_path.write_text(
                    json.dumps(
                        {"seeds": [model_seed], "queries": {job_id: query}}, indent=2
                    )
                    + "\n"
                )
                manifest.append(
                    {
                        "job_id": job_id,
                        "affibody_design_code": code,
                        "replicate": replicate_index,
                        "model_seed": model_seed,
                        "ref_conformer_seed": ref_conformer_seed,
                        "global_feature_seed": args.global_feature_seed,
                        "duplicate_control": int(duplicate),
                        "retention_percent": row["retention_percent"],
                        "pair_uid": row["pair_uid"],
                        "sequence_pair_sha256": row["sequence_pair_sha256"],
                        "query_json": str(query_path.resolve()),
                        "query_json_sha256": sha256(query_path),
                        "runner_yaml": str(runner_paths[model_seed]),
                        "runner_yaml_sha256": sha256(runner_paths[model_seed]),
                        "output_dir": str((outputs_dir / job_id).resolve()),
                    }
                )

    manifest_path = args.output_root / "job_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    provenance = {
        "condition": "approximate native-pMHC crystal-calibrated four-chain context",
        "not_the_exact_assay_construct": True,
        "selection": PANEL_SELECTION,
        "paired_replicates": [
            {
                "replicate": index,
                "ref_conformer_seed": seed,
                "model_seed": seed,
            }
            for index, seed in enumerate(args.replicate_seeds, start=1)
        ],
        "global_feature_seed": args.global_feature_seed,
        "ref_conformer_rng_controlled_by_openfold_env_patch": True,
        "design_limitation": "paired-seed sensitivity, not a full factorial",
        "one_diffusion_sample_per_job": True,
        "single_query_process_per_job": True,
        "paired_msa_used": False,
        "templates_used": False,
        "source_csv_snapshot": str(source_snapshot.resolve()),
        "source_csv_snapshot_sha256": sha256(source_snapshot),
        "reference_json_snapshot": str(reference_snapshot.resolve()),
        "reference_json_snapshot_sha256": sha256(reference_snapshot),
        "n_primary_jobs": len(selected) * len(args.replicate_seeds),
        "n_duplicate_control_jobs": len(args.replicate_seeds),
        "n_total_jobs": len(manifest),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
    }
    (args.output_root / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    harden_private_tree(args.output_root)


if __name__ == "__main__":
    main()
