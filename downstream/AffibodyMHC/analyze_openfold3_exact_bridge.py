#!/usr/bin/env python3
"""Summarize controlled distograms for the exact two-chain SMART bridge."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from downstream.AffibodyMHC.openfold3_distogram_utils import (
    DISTOGRAM_BIN_EDGES_A,
    distogram_logits_from_latent,
    distogram_matrices,
    harden_private_tree,
    pair_tensor,
    sha256_file,
    token_lookup_from_batch,
    validate_experiment_config,
    validate_query_document,
    write_json_private,
)


SMART_PEPTIDE_POSITIONS = tuple(range(262, 271))
SMART_DESIGNED_PEPTIDE_POSITIONS = (265, 266)
AFFIBODY_DESIGN_POSITIONS = (6, 10, 13, 14, 17)
PRIMARY_EXACT_PAIRS = ((265, 14), (265, 17), (266, 10))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    compact_dir = args.output_dir / "compact_designed_pair_distograms"
    compact_dir.mkdir(mode=0o700)

    with args.manifest.open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if len(jobs) != 3:
        raise ValueError(f"Expected three exact-bridge jobs, found {len(jobs)}")
    input_provenance_path = args.manifest.parent / "input_provenance.json"
    input_provenance = json.loads(input_provenance_path.read_text())
    if input_provenance.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("Input provenance manifest hash does not match")

    rows = []
    msa_audit = None
    for job in jobs:
        model_seed = int(job["model_seed"])
        if sha256_file(Path(job["query_json"])) != job["query_json_sha256"]:
            raise ValueError(f"Query hash mismatch for {job['job_id']}")
        if sha256_file(Path(job["runner_yaml"])) != job["runner_yaml_sha256"]:
            raise ValueError(f"Runner hash mismatch for {job['job_id']}")
        document = json.loads(Path(job["query_json"]).read_text())
        query = validate_query_document(document, model_seed, ("S", "H"))
        if query.get("use_paired_msas", True):
            raise ValueError("Paired MSA unexpectedly enabled")
        chain_map = {chain["chain_ids"][0]: chain for chain in query["chains"]}
        if tuple(chain_map) != ("S", "H"):
            raise ValueError(f"Expected S/H chains, found {tuple(chain_map)}")
        if chain_map["S"]["sequence"][261:270] != "SLLMWITQV":
            raise ValueError("S262-S270 is not the MW peptide")
        forbidden = (
            "paired_msa_file_paths",
            "template_alignment_file_path",
            "template_cif_paths",
        )
        if any(chain.get(key) for chain in query["chains"] for key in forbidden):
            raise ValueError("Paired/template path populated")

        output = Path(job["output_dir"])
        experiment = json.loads((output / "experiment_config.json").read_text())
        validate_experiment_config(experiment, model_seed)
        output_query = json.loads((output / "inference_query_set.json").read_text())
        validate_query_document(output_query, model_seed, ("S", "H"))
        summary_text = (output / "summary.txt").read_text()
        if "Successful Queries:  1" not in summary_text:
            raise ValueError(f"Failed output {job['job_id']}")
        seed_dir = output / job["job_id"] / f"seed_{model_seed}"
        latent_path = next(seed_dir.glob("*_latent_output.pt"))
        batch_path = next(seed_dir.glob("*_batch.pt"))
        confidence_path = next(seed_dir.glob("*_confidences_aggregated.json"))
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        batch = torch.load(batch_path, map_location="cpu", weights_only=False)
        logits = distogram_logits_from_latent(latent)
        probabilities, contacts, expected = distogram_matrices(logits)
        lookup, _, _ = token_lookup_from_batch(batch)
        designed_logits = pair_tensor(
            logits,
            lookup,
            "S",
            SMART_DESIGNED_PEPTIDE_POSITIONS,
            "H",
            AFFIBODY_DESIGN_POSITIONS,
        )
        designed_probabilities = pair_tensor(
            probabilities,
            lookup,
            "S",
            SMART_DESIGNED_PEPTIDE_POSITIONS,
            "H",
            AFFIBODY_DESIGN_POSITIONS,
        )
        designed_contacts = pair_tensor(
            contacts,
            lookup,
            "S",
            SMART_DESIGNED_PEPTIDE_POSITIONS,
            "H",
            AFFIBODY_DESIGN_POSITIONS,
        )
        designed_expected = pair_tensor(
            expected,
            lookup,
            "S",
            SMART_DESIGNED_PEPTIDE_POSITIONS,
            "H",
            AFFIBODY_DESIGN_POSITIONS,
        )
        primary_contacts = np.asarray(
            [
                contacts[lookup[("S", smart_position)], lookup[("H", affibody_position)]]
                for smart_position, affibody_position in PRIMARY_EXACT_PAIRS
            ],
            dtype=np.float64,
        )
        primary_expected = np.asarray(
            [
                expected[lookup[("S", smart_position)], lookup[("H", affibody_position)]]
                for smart_position, affibody_position in PRIMARY_EXACT_PAIRS
            ],
            dtype=np.float64,
        )
        all_peptide_contacts = pair_tensor(
            contacts,
            lookup,
            "S",
            SMART_PEPTIDE_POSITIONS,
            "H",
            tuple(range(1, 59)),
        )
        compact_path = compact_dir / f"{job['job_id']}.npz"
        np.savez_compressed(
            compact_path,
            designed_pair_logits=designed_logits.astype(np.float32),
            designed_pair_probabilities=designed_probabilities.astype(np.float32),
            designed_pair_contact_probabilities=designed_contacts.astype(np.float32),
            designed_pair_expected_distances_A=designed_expected.astype(np.float32),
            primary_pair_contact_probabilities=primary_contacts.astype(np.float32),
            primary_pair_expected_distances_A=primary_expected.astype(np.float32),
            smart_chain_positions=np.asarray(SMART_DESIGNED_PEPTIDE_POSITIONS),
            affibody_positions=np.asarray(AFFIBODY_DESIGN_POSITIONS),
            primary_pairs=np.asarray(PRIMARY_EXACT_PAIRS),
            distogram_bin_edges_A=DISTOGRAM_BIN_EDGES_A,
            orientation=np.asarray("S_to_H"),
        )
        compact_path.chmod(0o600)
        confidence = json.loads(confidence_path.read_text())
        rows.append(
            {
                "job_id": job["job_id"],
                "replicate": int(job["replicate"]),
                "ref_conformer_seed": int(job["ref_conformer_seed"]),
                "model_seed": model_seed,
                "crystal_3_contact_mean_probability": float(primary_contacts.mean()),
                "crystal_3_contact_mean_expected_distance_A": float(primary_expected.mean()),
                "designed_10_mean_contact_probability": float(designed_contacts.mean()),
                "designed_10_max_contact_probability": float(designed_contacts.max()),
                "designed_10_min_expected_distance_A": float(designed_expected.min()),
                "all_peptide_affibody_max_contact_probability": float(all_peptide_contacts.max()),
                "iptm": float(confidence["iptm"]),
                "max_abs_contact_probability_asymmetry": float(
                    np.max(np.abs(contacts - contacts.T))
                ),
                "raw_latent_path": str(latent_path.resolve()),
                "compact_distogram_path": str(compact_path.resolve()),
            }
        )
        if msa_audit is None:
            msa_path = Path(chain_map["S"]["main_msa_file_paths"][0])
            with np.load(msa_path, allow_pickle=True) as archive:
                msa = archive["colabfold_main"].item()["msa"]
            msa_audit = {
                "main_msa_path": str(msa_path.resolve()),
                "n_rows": int(msa.shape[0]),
                "s265_non_gap_rows": int(np.sum(msa[:, 264] != "-")),
                "s265_rows_matching_query_M": int(np.sum(msa[:, 264] == "M")),
                "s266_non_gap_rows": int(np.sum(msa[:, 265] != "-")),
                "s266_rows_matching_query_W": int(np.sum(msa[:, 265] == "W")),
                "caveat": (
                    "The server aligned many non-query rows through the artificial "
                    "GS-linker/peptide tail, so this is not a biological paired "
                    "peptide-evolution MSA."
                ),
            }
        del latent, batch, logits

    write_csv(args.output_dir / "exact_bridge_replicates.csv", rows)
    gate_pass = any(row["all_peptide_affibody_max_contact_probability"] >= 0.10 for row in rows)
    summary = {
        "input_provenance": str(input_provenance_path.resolve()),
        "input_provenance_manifest_hash_verified": True,
        "condition": "exact supplied 270-aa SMART chain plus 58-aa Affibody",
        "orientation": "S_to_H without symmetrization",
        "n_controlled_paired_replicates": 3,
        "absolute_interface_gate": "any replicate all-peptide/Affibody max P<8A >=0.10",
        "gate_pass": gate_pass,
        "decision": (
            "continue exact-condition variants" if gate_pass else "stop exact-condition variants"
        ),
        "crystal_3_contact_feature": {
            "smart_to_affibody_pairs_one_based": [list(pair) for pair in PRIMARY_EXACT_PAIRS],
            "mean_probability_across_replicates": float(
                np.mean([row["crystal_3_contact_mean_probability"] for row in rows])
            ),
            "range_probability_across_replicates": [
                float(min(row["crystal_3_contact_mean_probability"] for row in rows)),
                float(max(row["crystal_3_contact_mean_probability"] for row in rows)),
            ],
            "mean_expected_distance_A_across_replicates": float(
                np.mean(
                    [
                        row["crystal_3_contact_mean_expected_distance_A"]
                        for row in rows
                    ]
                )
            ),
            "range_expected_distance_A_across_replicates": [
                float(
                    min(
                        row["crystal_3_contact_mean_expected_distance_A"]
                        for row in rows
                    )
                ),
                float(
                    max(
                        row["crystal_3_contact_mean_expected_distance_A"]
                        for row in rows
                    )
                ),
            ],
        },
        "replicates": rows,
        "msa_audit": msa_audit,
    }
    write_json_private(args.output_dir / "exact_bridge_summary.json", summary)
    harden_private_tree(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
