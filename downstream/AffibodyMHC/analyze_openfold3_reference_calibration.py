#!/usr/bin/env python3
"""Compare controlled NNYYF native-pMHC distograms with the LibB crystal."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from downstream.AffibodyMHC.analyze_nyeso_xx133_structure import (
    euclidean,
    map_contiguous_subsequence,
    parse_pdb,
    representative_atom,
)
from downstream.AffibodyMHC.analyze_openfold3_native_panel import (
    find_job_artifacts,
    write_csv_private,
)
from downstream.AffibodyMHC.openfold3_distogram_utils import (
    distogram_logits_from_latent,
    distogram_matrices,
    harden_private_tree,
    sha256_file,
    spearman,
    token_lookup_from_batch,
    validate_experiment_config,
    validate_query_document,
    write_json_private,
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    ranks = average_ranks(scores)
    u = ranks[positive].sum() - n_positive * (n_positive + 1) / 2
    return float(u / (n_positive * n_negative))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order].astype(bool)
    precision = np.cumsum(ranked) / (np.arange(ranked.size) + 1)
    return float(precision[ranked].mean())


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crystal-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    jobs = [
        job
        for job in manifest
        if job["affibody_design_code"] == "NNYYF"
        and job["duplicate_control"] == "0"
    ]
    if len(jobs) != 3:
        raise ValueError(f"Expected three primary NNYYF jobs, found {len(jobs)}")

    crystal = parse_pdb(args.crystal_pdb)
    if "P" not in crystal or "H" not in crystal:
        raise ValueError("Crystal lacks P/H chains")
    summary_rows = []
    pair_rows = []
    for job in sorted(jobs, key=lambda row: int(row["replicate"])):
        model_seed = int(job["model_seed"])
        if sha256_file(Path(job["query_json"])) != job["query_json_sha256"]:
            raise ValueError(f"Query hash mismatch for {job['job_id']}")
        document = json.loads(Path(job["query_json"]).read_text())
        query = validate_query_document(document, model_seed)
        sequences = {
            chain["chain_ids"][0]: chain["sequence"] for chain in query["chains"]
        }
        if sequences["P"][3:5] != "MW" or any(
            sequences["H"][position - 1] != amino_acid
            for position, amino_acid in zip((6, 10, 13, 14, 17), "NNYYF")
        ):
            raise ValueError("Selected job is not the NNYYF/MW crystal reference")
        mappings = {
            chain_id: map_contiguous_subsequence(crystal[chain_id], sequences[chain_id])
            for chain_id in ("P", "H")
        }

        artifacts = find_job_artifacts(job)
        output_root = Path(job["output_dir"])
        validate_query_document(
            json.loads((output_root / "inference_query_set.json").read_text()),
            model_seed,
        )
        validate_experiment_config(
            json.loads((output_root / "experiment_config.json").read_text()),
            model_seed,
        )
        latent = torch.load(artifacts["latent"], map_location="cpu", weights_only=True)
        batch = torch.load(artifacts["batch"], map_location="cpu", weights_only=False)
        logits = distogram_logits_from_latent(latent)
        _, contacts, expected = distogram_matrices(logits, 8.0)
        lookup, _, _ = token_lookup_from_batch(batch)

        labels = []
        scores = []
        observed_distances = []
        expected_distances = []
        for peptide_residue in crystal["P"].residues:
            peptide_position = mappings["P"][id(peptide_residue)]
            peptide_token = lookup[("P", peptide_position)]
            for affibody_residue in crystal["H"].residues:
                affibody_position = mappings["H"][id(affibody_residue)]
                affibody_token = lookup[("H", affibody_position)]
                observed = euclidean(
                    representative_atom(peptide_residue).xyz,
                    representative_atom(affibody_residue).xyz,
                )
                label = int(observed < 8.0)
                score = float(contacts[peptide_token, affibody_token])
                expected_distance = float(expected[peptide_token, affibody_token])
                labels.append(label)
                scores.append(score)
                observed_distances.append(observed)
                expected_distances.append(expected_distance)
                pair_rows.append(
                    {
                        "replicate": int(job["replicate"]),
                        "model_seed": model_seed,
                        "peptide_position": peptide_position,
                        "peptide_aa": peptide_residue.name1,
                        "affibody_position": affibody_position,
                        "affibody_aa": affibody_residue.name1,
                        "crystal_distance_A": observed,
                        "crystal_contact_lt8A": label,
                        "predicted_contact_probability": score,
                        "predicted_expected_distance_A": expected_distance,
                    }
                )
        labels_array = np.asarray(labels, dtype=np.int8)
        scores_array = np.asarray(scores, dtype=np.float64)
        observed_array = np.asarray(observed_distances, dtype=np.float64)
        expected_array = np.asarray(expected_distances, dtype=np.float64)
        n_contacts = int(labels_array.sum())
        top = np.argsort(-scores_array, kind="mergesort")[:n_contacts]
        summary_rows.append(
            {
                "replicate": int(job["replicate"]),
                "ref_conformer_seed": int(job["ref_conformer_seed"]),
                "model_seed": model_seed,
                "n_resolved_pairs": int(labels_array.size),
                "n_crystal_contacts_lt8A": n_contacts,
                "contact_auroc": roc_auc(labels_array, scores_array),
                "contact_average_precision": average_precision(labels_array, scores_array),
                "precision_at_n_crystal_contacts": float(labels_array[top].mean()),
                "mean_probability_true_contacts": float(scores_array[labels_array == 1].mean()),
                "mean_probability_noncontacts": float(scores_array[labels_array == 0].mean()),
                "max_probability_true_contacts": float(scores_array[labels_array == 1].max()),
                "spearman_contact_probability_vs_negative_crystal_distance": spearman(
                    scores_array, -observed_array
                ),
                "spearman_expected_distance_vs_crystal_distance": spearman(
                    expected_array, observed_array
                ),
                "raw_latent_path": str(artifacts["latent"].resolve()),
            }
        )
        del latent, batch, logits

    write_csv_private(args.output_dir / "reference_calibration_replicates.csv", summary_rows)
    write_csv_private(args.output_dir / "reference_calibration_pairs.csv", pair_rows)
    metrics = [
        "contact_auroc",
        "contact_average_precision",
        "precision_at_n_crystal_contacts",
        "mean_probability_true_contacts",
        "mean_probability_noncontacts",
        "max_probability_true_contacts",
        "spearman_contact_probability_vs_negative_crystal_distance",
        "spearman_expected_distance_vs_crystal_distance",
    ]
    summary = {
        "condition": "native four-chain NNYYF/MW crystal reference",
        "model": "OpenFold3-preview2 0.4.5 / of3-p2-155k",
        "orientation": "P_to_H without symmetrization",
        "contact_definition": "representative-atom crystal distance <8 A and sum of learned distogram bins whose upper edge is <=8 A",
        "n_controlled_paired_replicates": 3,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "crystal_pdb": str(args.crystal_pdb.resolve()),
        "crystal_pdb_sha256": sha256_file(args.crystal_pdb),
        "metrics_across_replicates": {
            metric: {
                "mean": float(np.mean([row[metric] for row in summary_rows])),
                "range": [
                    float(min(row[metric] for row in summary_rows)),
                    float(max(row[metric] for row in summary_rows)),
                ],
            }
            for metric in metrics
        },
        "replicates": summary_rows,
    }
    write_json_private(args.output_dir / "reference_calibration_summary.json", summary)
    harden_private_tree(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
