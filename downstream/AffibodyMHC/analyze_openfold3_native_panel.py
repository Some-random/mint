#!/usr/bin/env python3
"""Analyze the preregistered controlled LibB/MW native-pMHC distogram panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from downstream.AffibodyMHC.analyze_nyeso_xx133_structure import (
    euclidean,
    map_contiguous_subsequence,
    parse_pdb,
    representative_atom,
)
from downstream.AffibodyMHC.openfold3_distogram_utils import (
    DISTOGRAM_BIN_EDGES_A,
    distogram_logits_from_latent,
    distogram_matrices,
    harden_private_tree,
    pair_tensor,
    sha256_file,
    spearman,
    token_lookup_from_batch,
    validate_experiment_config,
    validate_query_document,
    write_json_private,
)


PRIMARY_CRYSTAL_PAIRS = ((4, 14), (4, 17), (5, 10))
DESIGNED_PEPTIDE_POSITIONS = (4, 5)
DESIGNED_AFFIBODY_POSITIONS = (6, 10, 13, 14, 17)
EXPECTED_REPLICATES = (
    (1, 42, 42),
    (2, 43, 43),
    (3, 2746317213, 2746317213),
)


def select_crystal_reference_job(jobs: list[dict]) -> dict:
    """Select the one primary NNYYF job that matches the crystal Affibody."""
    matches = [
        job
        for job in jobs
        if job["affibody_design_code"] == "NNYYF"
        and job["duplicate_control"] == "0"
        and job["replicate"] == "1"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one primary NNYYF crystal-reference job, found {len(matches)}")
    return matches[0]


def write_csv_private(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def recursive_tensors(value, prefix="") -> dict[str, torch.Tensor]:
    """Flatten every tensor, including tensors nested in dicts/lists."""
    if torch.is_tensor(value):
        return {prefix or "<root>": value}
    tensors = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            tensors.update(recursive_tensors(value[key], child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            tensors.update(recursive_tensors(item, child))
    return tensors


def compare_batch_tensors(first_batch: dict, second_batch: dict) -> dict:
    first = recursive_tensors(first_batch)
    second = recursive_tensors(second_batch)
    if set(first) != set(second):
        raise ValueError("Compared batches have different recursive tensor paths")
    unequal = [path for path in sorted(first) if not torch.equal(first[path], second[path])]
    entry_counts = {}
    max_abs = {}
    for path in unequal:
        if first[path].shape == second[path].shape:
            entry_counts[path] = int(torch.count_nonzero(first[path] != second[path]))
            if first[path].is_floating_point():
                max_abs[path] = float(torch.max(torch.abs(first[path] - second[path])))
    active_deletion = None
    masked_deletion = None
    if "deletion_value" in unequal and "msa_mask" in first:
        difference = first["deletion_value"] != second["deletion_value"]
        active = first["msa_mask"].bool()
        while active.ndim < difference.ndim:
            active = active.unsqueeze(-1)
        if active.shape != difference.shape:
            active = torch.broadcast_to(active, difference.shape)
        active_deletion = int(torch.count_nonzero(difference & active))
        masked_deletion = int(torch.count_nonzero(difference & ~active))
    return {
        "n_recursive_tensor_paths": len(first),
        "unequal_paths": unequal,
        "entry_counts": entry_counts,
        "max_abs": max_abs,
        "active_deletion_value_differences": active_deletion,
        "masked_deletion_value_differences": masked_deletion,
    }


def crystal_primary_pair_rows(
    pdb_path: Path, peptide_sequence: str, affibody_sequence: str
) -> list[dict]:
    chains = parse_pdb(pdb_path)
    if "P" not in chains or "H" not in chains:
        raise ValueError("Crystal PDB lacks the selected P/H chains")
    mappings = {
        "P": map_contiguous_subsequence(chains["P"], peptide_sequence),
        "H": map_contiguous_subsequence(chains["H"], affibody_sequence),
    }
    residues = {
        chain_id: {mappings[chain_id][id(residue)]: residue for residue in chains[chain_id].residues}
        for chain_id in ("P", "H")
    }
    rows = []
    for peptide_position, affibody_position in PRIMARY_CRYSTAL_PAIRS:
        peptide_residue = residues["P"][peptide_position]
        affibody_residue = residues["H"][affibody_position]
        distance = euclidean(
            representative_atom(peptide_residue).xyz,
            representative_atom(affibody_residue).xyz,
        )
        if not distance < 8.0:
            raise ValueError(
                f"Preregistered pair P{peptide_position}/H{affibody_position} "
                f"is not a crystal contact: {distance:.3f} A"
            )
        rows.append(
            {
                "peptide_position": peptide_position,
                "peptide_reference_aa": peptide_residue.name1,
                "affibody_position": affibody_position,
                "affibody_reference_aa": affibody_residue.name1,
                "crystal_representative_atom_distance_A": distance,
            }
        )
    return rows


def summarize_distogram(
    logits: np.ndarray, batch: dict
) -> tuple[dict, dict[str, np.ndarray]]:
    probabilities, contacts, expected = distogram_matrices(logits, 8.0)
    lookup, token_chain_ids, token_residue_indices = token_lookup_from_batch(batch)
    designed_logits = pair_tensor(
        logits,
        lookup,
        "P",
        DESIGNED_PEPTIDE_POSITIONS,
        "H",
        DESIGNED_AFFIBODY_POSITIONS,
    )
    designed_probabilities = pair_tensor(
        probabilities,
        lookup,
        "P",
        DESIGNED_PEPTIDE_POSITIONS,
        "H",
        DESIGNED_AFFIBODY_POSITIONS,
    )
    designed_contacts = pair_tensor(
        contacts,
        lookup,
        "P",
        DESIGNED_PEPTIDE_POSITIONS,
        "H",
        DESIGNED_AFFIBODY_POSITIONS,
    )
    designed_expected = pair_tensor(
        expected,
        lookup,
        "P",
        DESIGNED_PEPTIDE_POSITIONS,
        "H",
        DESIGNED_AFFIBODY_POSITIONS,
    )
    primary_contacts = np.asarray(
        [
            contacts[lookup[("P", peptide)], lookup[("H", affibody)]]
            for peptide, affibody in PRIMARY_CRYSTAL_PAIRS
        ],
        dtype=np.float64,
    )
    primary_expected = np.asarray(
        [
            expected[lookup[("P", peptide)], lookup[("H", affibody)]]
            for peptide, affibody in PRIMARY_CRYSTAL_PAIRS
        ],
        dtype=np.float64,
    )
    summary = {
        "crystal_3_contact_mean_probability": float(primary_contacts.mean()),
        "all_10_designed_pair_mean_probability": float(designed_contacts.mean()),
        "crystal_3_contact_mean_expected_distance_A": float(
            primary_expected.mean()
        ),
        "all_10_designed_pair_max_probability": float(designed_contacts.max()),
        "max_abs_distogram_logit_asymmetry": float(
            np.max(np.abs(logits - np.swapaxes(logits, 0, 1)))
        ),
        "max_abs_contact_probability_asymmetry": float(
            np.max(np.abs(contacts - contacts.T))
        ),
    }
    compact = {
        "designed_pair_logits": designed_logits.astype(np.float32),
        "designed_pair_probabilities": designed_probabilities.astype(np.float32),
        "designed_pair_contact_probabilities": designed_contacts.astype(np.float32),
        "designed_pair_expected_distances_A": designed_expected.astype(np.float32),
        "primary_pair_contact_probabilities": primary_contacts.astype(np.float32),
        "primary_pair_expected_distances_A": primary_expected.astype(np.float32),
        "peptide_positions": np.asarray(DESIGNED_PEPTIDE_POSITIONS, dtype=np.int16),
        "affibody_positions": np.asarray(DESIGNED_AFFIBODY_POSITIONS, dtype=np.int16),
        "primary_pairs": np.asarray(PRIMARY_CRYSTAL_PAIRS, dtype=np.int16),
        "distogram_bin_edges_A": DISTOGRAM_BIN_EDGES_A,
        "orientation": np.asarray("P_to_H"),
        "token_chain_ids": token_chain_ids,
        "token_residue_indices": token_residue_indices,
    }
    return summary, compact


def find_job_artifacts(job: dict) -> dict[str, Path]:
    output = Path(job["output_dir"])
    summary_path = output / "summary.txt"
    if not summary_path.exists():
        raise ValueError(f"Missing summary for {job['job_id']}")
    summary_text = summary_path.read_text()
    if "Successful Queries:  1" not in summary_text or "Failed Queries:      0" not in summary_text:
        raise ValueError(f"Unsuccessful output for {job['job_id']}")
    seed_dir = output / job["job_id"] / f"seed_{job['model_seed']}"
    patterns = {
        "batch": "*_batch.pt",
        "latent": "*_latent_output.pt",
        "confidence": "*_confidences_aggregated.json",
    }
    artifacts = {"summary": summary_path}
    for kind, pattern in patterns.items():
        matches = list(seed_dir.glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {kind} for {job['job_id']}, found {matches}"
            )
        artifacts[kind] = matches[0]
    return artifacts


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predeclared-analysis", type=Path, required=True)
    parser.add_argument("--orientation-addendum", type=Path, required=True)
    parser.add_argument("--crystal-pdb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir.chmod(0o700)
    compact_dir = args.output_dir / "compact_designed_pair_distograms"
    compact_dir.mkdir(mode=0o700)
    plan = json.loads(args.predeclared_analysis.read_text())
    orientation_plan = json.loads(args.orientation_addendum.read_text())
    if orientation_plan["distogram_orientation_for_all_reported_features"] != (
        "peptide token to Affibody token (P-to-H)"
    ) or orientation_plan["symmetrization"] != "none":
        raise ValueError("Distogram orientation addendum changed")
    if plan["primary_feature"]["pairs_one_based"] != [
        {"peptide_position": 4, "affibody_position": 14, "reference_pair": "M4-Y14"},
        {"peptide_position": 4, "affibody_position": 17, "reference_pair": "M4-F17"},
        {"peptide_position": 5, "affibody_position": 10, "reference_pair": "W5-N10"},
    ]:
        raise ValueError("Predeclared primary contact mask changed")

    with args.manifest.open(newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if len(jobs) != 33:
        raise ValueError(f"Expected 33 jobs, found {len(jobs)}")
    input_provenance_path = args.manifest.parent / "input_provenance.json"
    input_provenance = json.loads(input_provenance_path.read_text())
    if input_provenance.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("Input provenance manifest hash does not match")
    expected_design = {(str(rep), str(ref), str(model)) for rep, ref, model in EXPECTED_REPLICATES}
    observed_design = {
        (job["replicate"], job["ref_conformer_seed"], job["model_seed"])
        for job in jobs
    }
    if observed_design != expected_design:
        raise ValueError(f"Unexpected replicate design {observed_design}")

    reference_job = select_crystal_reference_job(jobs)
    reference_query_document = json.loads(Path(reference_job["query_json"]).read_text())
    reference_query = validate_query_document(
        reference_query_document, int(reference_job["model_seed"])
    )
    chain_sequences = {
        chain["chain_ids"][0]: chain["sequence"] for chain in reference_query["chains"]
    }
    crystal_rows = crystal_primary_pair_rows(
        args.crystal_pdb, chain_sequences["P"], chain_sequences["H"]
    )
    observed_reference_pairs = {
        (row["peptide_position"], row["affibody_position"]): (
            row["peptide_reference_aa"],
            row["affibody_reference_aa"],
        )
        for row in crystal_rows
    }
    expected_reference_pairs = {
        (4, 14): ("M", "Y"),
        (4, 17): ("M", "F"),
        (5, 10): ("W", "N"),
    }
    if observed_reference_pairs != expected_reference_pairs:
        raise ValueError(
            "Crystal-reference residue identities changed: "
            f"{observed_reference_pairs}"
        )
    write_csv_private(args.output_dir / "crystal_primary_pairs.csv", crystal_rows)

    per_job_rows = []
    artifacts_by_job = {}
    for job in jobs:
        model_seed = int(job["model_seed"])
        if sha256_file(Path(job["query_json"])) != job["query_json_sha256"]:
            raise ValueError(f"Query hash mismatch for {job['job_id']}")
        if sha256_file(Path(job["runner_yaml"])) != job["runner_yaml_sha256"]:
            raise ValueError(f"Runner hash mismatch for {job['job_id']}")
        query_document = json.loads(Path(job["query_json"]).read_text())
        validate_query_document(query_document, model_seed)
        artifacts = find_job_artifacts(job)
        artifacts_by_job[job["job_id"]] = artifacts
        output_query_set = json.loads(
            (Path(job["output_dir"]) / "inference_query_set.json").read_text()
        )
        validate_query_document(output_query_set, model_seed)
        experiment_config = json.loads(
            (Path(job["output_dir"]) / "experiment_config.json").read_text()
        )
        validate_experiment_config(experiment_config, model_seed)

        latent = torch.load(artifacts["latent"], map_location="cpu", weights_only=True)
        batch = torch.load(artifacts["batch"], map_location="cpu", weights_only=False)
        logits = distogram_logits_from_latent(latent)
        summary, compact = summarize_distogram(logits, batch)
        compact_path = compact_dir / f"{job['job_id']}.npz"
        np.savez_compressed(compact_path, **compact)
        compact_path.chmod(0o600)
        confidence = json.loads(artifacts["confidence"].read_text())
        per_job_rows.append(
            {
                "job_id": job["job_id"],
                "affibody_design_code": job["affibody_design_code"],
                "replicate": int(job["replicate"]),
                "ref_conformer_seed": int(job["ref_conformer_seed"]),
                "model_seed": model_seed,
                "global_feature_seed": int(job["global_feature_seed"]),
                "duplicate_control": int(job["duplicate_control"]),
                "retention_percent": float(job["retention_percent"]),
                **summary,
                "iptm": float(confidence["iptm"]),
                "raw_latent_path": str(artifacts["latent"].resolve()),
                "batch_path": str(artifacts["batch"].resolve()),
                "compact_distogram_path": str(compact_path.resolve()),
            }
        )
        del latent, batch, logits
    write_csv_private(args.output_dir / "per_job_features.csv", per_job_rows)

    primary_rows = [row for row in per_job_rows if row["duplicate_control"] == 0]
    grouped = defaultdict(list)
    for row in primary_rows:
        grouped[row["affibody_design_code"]].append(row)
    if len(grouped) != 10 or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("Expected 10 variants with three primary replicates each")

    features = [
        "crystal_3_contact_mean_probability",
        "all_10_designed_pair_mean_probability",
        "crystal_3_contact_mean_expected_distance_A",
        "all_10_designed_pair_max_probability",
    ]
    aggregate_rows = []
    for code, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["replicate"])
        aggregate = {
            "affibody_design_code": code,
            "retention_percent": rows[0]["retention_percent"],
        }
        for feature in features:
            values = [row[feature] for row in rows]
            aggregate[f"{feature}_mean"] = float(np.mean(values))
            aggregate[f"{feature}_sd"] = sample_sd(values)
        aggregate_rows.append(aggregate)
    write_csv_private(args.output_dir / "variant_replicate_means.csv", aggregate_rows)

    retention = np.asarray([row["retention_percent"] for row in aggregate_rows])
    correlations = {}
    directions = {
        "crystal_3_contact_mean_probability": "positive",
        "all_10_designed_pair_mean_probability": "positive",
        "crystal_3_contact_mean_expected_distance_A": "negative",
        "all_10_designed_pair_max_probability": "positive",
    }
    for feature in features:
        values = np.asarray([row[f"{feature}_mean"] for row in aggregate_rows])
        correlations[feature] = {
            "spearman_vs_retention": spearman(values, retention),
            "prespecified_expected_direction": directions[feature],
        }

    per_replicate_spearman = {}
    replicate_vectors = {}
    code_order = [row["affibody_design_code"] for row in aggregate_rows]
    for replicate, _, _ in EXPECTED_REPLICATES:
        by_code = {
            row["affibody_design_code"]: row
            for row in primary_rows
            if row["replicate"] == replicate
        }
        vector = np.asarray(
            [by_code[code]["crystal_3_contact_mean_probability"] for code in code_order]
        )
        replicate_vectors[replicate] = vector
        per_replicate_spearman[str(replicate)] = spearman(vector, retention)

    rank_pairs = []
    for left, right in ((1, 2), (1, 3), (2, 3)):
        rank_pairs.append(
            {
                "replicate_pair": f"{left}-{right}",
                "spearman": spearman(replicate_vectors[left], replicate_vectors[right]),
            }
        )
    primary_means = np.asarray(
        [row["crystal_3_contact_mean_probability_mean"] for row in aggregate_rows]
    )
    primary_sds = np.asarray(
        [row["crystal_3_contact_mean_probability_sd"] for row in aggregate_rows]
    )
    between_variant_sd = float(np.std(primary_means, ddof=1))
    within_variant_rms_sd = float(np.sqrt(np.mean(primary_sds**2)))

    duplicate_rows = []
    for replicate, _, model_seed in EXPECTED_REPLICATES:
        candidates = [
            row
            for row in per_job_rows
            if row["affibody_design_code"] == "NNYYF"
            and row["replicate"] == replicate
        ]
        primary = next(row for row in candidates if row["duplicate_control"] == 0)
        duplicate = next(row for row in candidates if row["duplicate_control"] == 1)
        primary_artifacts = artifacts_by_job[primary["job_id"]]
        duplicate_artifacts = artifacts_by_job[duplicate["job_id"]]
        first_batch = torch.load(
            primary_artifacts["batch"], map_location="cpu", weights_only=False
        )
        second_batch = torch.load(
            duplicate_artifacts["batch"], map_location="cpu", weights_only=False
        )
        batch_comparison = compare_batch_tensors(first_batch, second_batch)
        first_latent = torch.load(
            primary_artifacts["latent"], map_location="cpu", weights_only=True
        )
        second_latent = torch.load(
            duplicate_artifacts["latent"], map_location="cpu", weights_only=True
        )
        first_logits = distogram_logits_from_latent(first_latent)
        second_logits = distogram_logits_from_latent(second_latent)
        logit_difference = np.abs(first_logits - second_logits)
        duplicate_rows.append(
            {
                "replicate": replicate,
                "model_seed": model_seed,
                "primary_job_id": primary["job_id"],
                "duplicate_job_id": duplicate["job_id"],
                "n_recursive_batch_tensor_paths": batch_comparison["n_recursive_tensor_paths"],
                "unequal_batch_tensor_paths": ";".join(batch_comparison["unequal_paths"]),
                "unequal_batch_tensor_entry_counts": json.dumps(batch_comparison["entry_counts"], sort_keys=True),
                "unequal_batch_tensor_max_abs": json.dumps(batch_comparison["max_abs"], sort_keys=True),
                "active_deletion_value_differences": batch_comparison["active_deletion_value_differences"],
                "masked_deletion_value_differences": batch_comparison["masked_deletion_value_differences"],
                "ref_pos_exact": int(
                    torch.equal(first_batch["ref_pos"], second_batch["ref_pos"])
                ),
                "distogram_logits_exact": int(np.array_equal(first_logits, second_logits)),
                "distogram_logits_max_abs_difference": float(np.max(logit_difference)),
                "distogram_logits_mean_abs_difference": float(np.mean(logit_difference)),
                "primary_feature_abs_difference": abs(
                    primary["crystal_3_contact_mean_probability"]
                    - duplicate["crystal_3_contact_mean_probability"]
                ),
                "all_10_mean_feature_abs_difference": abs(
                    primary["all_10_designed_pair_mean_probability"]
                    - duplicate["all_10_designed_pair_mean_probability"]
                ),
                "expected_distance_3_abs_difference_A": abs(
                    primary["crystal_3_contact_mean_expected_distance_A"]
                    - duplicate["crystal_3_contact_mean_expected_distance_A"]
                ),
            }
        )
        del first_batch, second_batch, first_latent, second_latent
    write_csv_private(args.output_dir / "duplicate_determinism_checks.csv", duplicate_rows)

    cross_replicate_rows = []
    for code, code_rows in sorted(grouped.items()):
        by_replicate = {row["replicate"]: row for row in code_rows}
        for left, right in ((1, 2), (1, 3), (2, 3)):
            left_row = by_replicate[left]
            right_row = by_replicate[right]
            left_batch = torch.load(
                artifacts_by_job[left_row["job_id"]]["batch"],
                map_location="cpu",
                weights_only=False,
            )
            right_batch = torch.load(
                artifacts_by_job[right_row["job_id"]]["batch"],
                map_location="cpu",
                weights_only=False,
            )
            comparison = compare_batch_tensors(left_batch, right_batch)
            non_ref_paths = [
                path for path in comparison["unequal_paths"] if path != "ref_pos"
            ]
            cross_replicate_rows.append(
                {
                    "affibody_design_code": code,
                    "replicate_pair": f"{left}-{right}",
                    "unequal_batch_tensor_paths": ";".join(comparison["unequal_paths"]),
                    "non_ref_pos_unequal_paths": ";".join(non_ref_paths),
                    "unequal_batch_tensor_entry_counts": json.dumps(comparison["entry_counts"], sort_keys=True),
                    "unequal_batch_tensor_max_abs": json.dumps(comparison["max_abs"], sort_keys=True),
                    "active_deletion_value_differences": comparison["active_deletion_value_differences"],
                    "masked_deletion_value_differences": comparison["masked_deletion_value_differences"],
                    "primary_feature_abs_difference": abs(
                        left_row["crystal_3_contact_mean_probability"]
                        - right_row["crystal_3_contact_mean_probability"]
                    ),
                }
            )
            del left_batch, right_batch
    write_csv_private(
        args.output_dir / "same_variant_cross_replicate_batch_audit.csv",
        cross_replicate_rows,
    )

    summary = {
        "analysis_plan": str(args.predeclared_analysis.resolve()),
        "analysis_plan_sha256": sha256_file(args.predeclared_analysis),
        "orientation_addendum": str(args.orientation_addendum.resolve()),
        "orientation_addendum_sha256": sha256_file(args.orientation_addendum),
        "input_provenance": str(input_provenance_path.resolve()),
        "input_provenance_manifest_hash_verified": True,
        "n_variants": 10,
        "n_primary_jobs": 30,
        "n_duplicate_jobs": 3,
        "no_model_fitting": True,
        "primary_result": correlations["crystal_3_contact_mean_probability"],
        "secondary_correlations": {
            key: value
            for key, value in correlations.items()
            if key != "crystal_3_contact_mean_probability"
        },
        "primary_per_replicate_spearman": per_replicate_spearman,
        "primary_rank_stability": {
            "pairwise": rank_pairs,
            "mean_pairwise_spearman": float(
                np.mean([row["spearman"] for row in rank_pairs])
            ),
        },
        "primary_variant_vs_replicate_spread": {
            "between_variant_sd_of_replicate_means": between_variant_sd,
            "within_variant_root_mean_square_replicate_sd": within_variant_rms_sd,
            "ratio": (
                between_variant_sd / within_variant_rms_sd
                if within_variant_rms_sd > 0
                else None
            ),
        },
        "duplicate_batch_and_distogram_exact_all_pass": all(
            row["ref_pos_exact"] == 1
            and row["distogram_logits_exact"] == 1
            and row["unequal_batch_tensor_paths"] == ""
            for row in duplicate_rows
        ),
        "duplicate_feature_drift": {
            "max_primary_abs_difference": max(
                row["primary_feature_abs_difference"] for row in duplicate_rows
            ),
            "max_primary_difference_over_between_variant_sd": (
                max(row["primary_feature_abs_difference"] for row in duplicate_rows)
                / between_variant_sd
                if between_variant_sd > 0
                else None
            ),
            "max_primary_difference_over_within_variant_rms_replicate_sd": (
                max(row["primary_feature_abs_difference"] for row in duplicate_rows)
                / within_variant_rms_sd
                if within_variant_rms_sd > 0
                else None
            ),
            "note": (
                "Exact equality is reported separately. Proportional acceptance "
                "must compare this drift with biological-variant and paired-seed spread."
            ),
        },
        "reported_distogram_orientation": "P_to_H without symmetrization",
        "panel_max_abs_distogram_logit_asymmetry": max(
            row["max_abs_distogram_logit_asymmetry"] for row in per_job_rows
        ),
        "panel_max_abs_contact_probability_asymmetry": max(
            row["max_abs_contact_probability_asymmetry"] for row in per_job_rows
        ),
        "same_variant_cross_replicate_non_ref_batch_drift": {
            "comparisons": len(cross_replicate_rows),
            "comparisons_with_non_ref_pos_unequal_paths": sum(
                bool(row["non_ref_pos_unequal_paths"])
                for row in cross_replicate_rows
            ),
            "paths_observed": sorted(
                {
                    path
                    for row in cross_replicate_rows
                    for path in row["non_ref_pos_unequal_paths"].split(";")
                    if path
                }
            ),
            "total_active_deletion_value_differences": sum(
                row["active_deletion_value_differences"] or 0
                for row in cross_replicate_rows
            ),
            "note": "Paired replicates intentionally combine ref-conformer and model/MSA seed changes; all non-ref tensor drift is reported rather than silently ignored."
        },
        "interpretation": (
            "Descriptive n=10 same-peptide comparison under an approximate "
            "native-pMHC format; no structure-to-retention model was fitted."
        ),
    }
    write_json_private(args.output_dir / "analysis_summary.json", summary)
    harden_private_tree(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
