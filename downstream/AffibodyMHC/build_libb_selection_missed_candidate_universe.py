#!/usr/bin/env python3
"""Build the immediate prospective LibB candidate universe.

This program enumerates every LibB-designed five-residue Affibody code for the
12 peptide targets in the corrected 120-pair direct-retention panel.  It then
removes:

1. every directly measured peptide--Affibody pair; and
2. every pair in the tie-inclusive top 2% of pooled R009+R010 raw counts.

The remaining rows are the current, directly actionable
"existing-target/selection-missed" prospective tier.  No model is loaded and no
candidate is scored.  Each output row carries the complete current model-input
sequences and raw R000--R014 count provenance.

The output is partitioned by peptide because the complete table contains more
than four million rows.  Existing paths are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (  # noqa: E402
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    EXPECTED_AFFIBODY_X_POSITIONS,
    fill_template,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    opaque_id,
    sha256_file,
    validate_private_output_path,
)


LIBRARY = "LibB"
RAW_ROUNDS = tuple(range(15))
POSITIVE_ROUNDS = (9, 10)
TOP_FRACTION = 0.02
EXPECTED_POOLED_ROWS = 1_451_567
EXPECTED_POOLED_RANK = 29_032
EXPECTED_POOLED_CUTOFF = 12
EXPECTED_POSITIVES_GLOBAL = 31_656
EXPECTED_PANEL_ROWS = 120
EXPECTED_TARGET_PEPTIDES = 12
EXPECTED_MEASURED_AFFIBODIES = 10
EXPECTED_AFFIBODY_CODES_PER_PEPTIDE = 13**5
EXPECTED_CANDIDATES = 4_447_848
EXPECTED_SELECTION_POSITIVE_EXCLUSIONS = 7_548
EXPECTED_MEASURED_EXCLUSIONS = 120

# Protein-level alphabets encoded by the provider's LibB DHS and peptide NHM
# library designs.  The Affibody alphabet, rather than the full 20-aa alphabet,
# defines this immediate prospective tier.
PEPTIDE_DESIGN_ALPHABET = frozenset("ADEFHIKLNPQSTVY")
AFFIBODY_DESIGN_ALPHABET = tuple("ADEFIKLMNSTVY")
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

RAW_FILE_PATTERN = re.compile(r"(\d{3})n?_count_+freq_pvalue\.tsv$")
RAW_COLUMNS = ("pep", "aff", "count", "frequency", "pvalue")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def discover_round_files(raw_root: Path) -> dict[int, Path]:
    directory = raw_root / "LibB Raw data"
    _require(directory.is_dir(), f"missing LibB raw directory: {directory}")
    result: dict[int, Path] = {}
    for path in sorted(directory.glob("*.tsv")):
        match = RAW_FILE_PATTERN.search(path.name)
        _require(match is not None, f"cannot parse round from {path.name}")
        round_index = int(match.group(1))
        _require(round_index not in result, f"duplicate raw round R{round_index:03d}")
        result[round_index] = path
    _require(tuple(sorted(result)) == RAW_ROUNDS, "LibB raw rounds R000--R014 are incomplete")
    return result


def load_count_round(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"pep": str, "aff": str, "count": np.int64},
        keep_default_na=False,
        na_filter=False,
    )
    _require(tuple(frame.columns) == RAW_COLUMNS, f"unexpected schema in {path}")
    _require(not frame[["pep", "aff"]].duplicated().any(), f"duplicate pair in {path}")
    _require(frame["pep"].str.len().eq(2).all(), f"bad peptide code in {path}")
    _require(frame["aff"].str.len().eq(5).all(), f"bad Affibody code in {path}")
    _require(frame["count"].gt(0).all(), f"non-positive count in {path}")
    return frame[["pep", "aff", "count"]].copy()


def pooled_positive_membership(
    round9: pd.DataFrame, round10: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    left = round9.rename(columns={"count": "r009_count"})
    right = round10.rename(columns={"count": "r010_count"})
    pooled = left.merge(right, on=["pep", "aff"], how="outer", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        pooled[column] = pooled[column].fillna(0).astype(np.int64)
    pooled["pooled_r009_r010_count"] = pooled["r009_count"] + pooled["r010_count"]
    rank = max(1, int(math.ceil(TOP_FRACTION * len(pooled))))
    counts = pooled["pooled_r009_r010_count"].to_numpy(dtype=np.int64)
    cutoff = int(np.partition(counts, len(counts) - rank)[len(counts) - rank])
    selected = pooled.loc[pooled["pooled_r009_r010_count"].ge(cutoff)].copy()
    selected = selected.sort_values(
        ["pooled_r009_r010_count", "pep", "aff"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    _require(len(pooled) == EXPECTED_POOLED_ROWS, "R009+R010 pooled union size changed")
    _require(rank == EXPECTED_POOLED_RANK, "nominal top-2% pooled rank changed")
    _require(cutoff == EXPECTED_POOLED_CUTOFF, "tie-inclusive pooled cutoff changed")
    _require(len(selected) == EXPECTED_POSITIVES_GLOBAL, "global pooled-positive count changed")
    return selected, {
        "pooled_union_rows": int(len(pooled)),
        "nominal_top_fraction_rank": int(rank),
        "inclusive_count_cutoff": int(cutoff),
        "selected_rows_including_boundary_ties": int(len(selected)),
    }


def load_panel(panel_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    panel = pd.read_csv(panel_path, keep_default_na=False, na_filter=False)
    required = {
        "library",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
    }
    _require(required.issubset(panel.columns), "corrected panel lacks required columns")
    panel = panel.loc[panel["library"].eq(LIBRARY)].copy()
    _require(len(panel) == EXPECTED_PANEL_ROWS, "corrected LibB panel must contain 120 rows")
    _require(panel["target_retention"].ne("").all(), "corrected panel contains missing retention")
    _require(not panel[["peptide_design_code", "affibody_design_code"]].duplicated().any(),
             "corrected panel contains duplicate pairs")
    peptides = sorted(panel["peptide_design_code"].unique())
    affibodies = sorted(panel["affibody_design_code"].unique())
    _require(len(peptides) == EXPECTED_TARGET_PEPTIDES, "expected 12 current peptide targets")
    _require(len(affibodies) == EXPECTED_MEASURED_AFFIBODIES, "expected ten measured Affibodies")
    _require(
        panel.groupby("peptide_design_code")["affibody_design_code"].nunique().eq(10).all(),
        "corrected panel is not a complete 12x10 matrix",
    )
    full_peptides = {}
    for peptide, group in panel.groupby("peptide_design_code", sort=True):
        chains = group["chain1_smart_hla_linker_peptide_sequence"].unique()
        _require(len(chains) == 1, f"peptide {peptide} has inconsistent chain-1 sequences")
        full = chains[0][-9:]
        _require(full[3:5] == peptide, f"peptide {peptide} does not map to positions 4/5")
        full_peptides[peptide] = full
    return panel, peptides, full_peptides


def load_weak_label_annotations(
    path: Path, target_peptides: set[str]
) -> tuple[dict[str, pd.DataFrame], set[str], dict]:
    columns = [
        "library",
        "pep",
        "aff",
        "weak_label",
        "weak_label_source",
        "within_declared_library_alphabet",
        "negative_r001_count_ge_3",
        "strict_retention_identity_cold_eligible",
    ]
    frame = pd.read_csv(path, usecols=columns, keep_default_na=False, na_filter=False)
    frame = frame.loc[frame["library"].eq(LIBRARY)].copy()
    for column in (
        "weak_label",
        "within_declared_library_alphabet",
        "negative_r001_count_ge_3",
        "strict_retention_identity_cold_eligible",
    ):
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    training = frame.loc[
        frame["within_declared_library_alphabet"].eq(1)
        & frame["strict_retention_identity_cold_eligible"].eq(1)
        & (frame["weak_label"].eq(1) | frame["negative_r001_count_ge_3"].eq(1))
    ]
    _require(len(training) == 30_648, "current strict LibB training membership changed")
    _require(not training["pep"].isin(target_peptides).any(),
             "a current evaluation peptide leaked into strict training")
    seen_training_affibodies = set(training["aff"])

    target = frame.loc[
        frame["pep"].isin(target_peptides)
        & frame["aff"].map(lambda value: set(value).issubset(set(AFFIBODY_DESIGN_ALPHABET)))
    ].copy()
    _require(not target[["pep", "aff"]].duplicated().any(), "duplicate target weak label")
    annotations = {
        peptide: group.set_index("aff", verify_integrity=True)
        for peptide, group in target.groupby("pep", sort=True)
    }
    return annotations, seen_training_affibodies, {
        "strict_training_rows": int(len(training)),
        "strict_training_unique_affibodies": int(len(seen_training_affibodies)),
        "target_pair_weak_labels": int(len(target)),
        "target_pair_positive_labels": int(target["weak_label"].eq(1).sum()),
        "target_pair_negative_labels": int(target["weak_label"].eq(0).sum()),
        "target_pair_high_confidence_negatives": int(target["negative_r001_count_ge_3"].sum()),
    }


def load_target_round_counts(
    path: Path,
    target_peptides: set[str],
    chunksize: int,
) -> dict[str, pd.Series]:
    blocks = []
    allowed_aff = set(AFFIBODY_DESIGN_ALPHABET)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["pep", "aff", "count"],
        dtype={"pep": str, "aff": str, "count": np.int64},
        keep_default_na=False,
        na_filter=False,
        chunksize=chunksize,
    ):
        mask = chunk["pep"].isin(target_peptides) & chunk["aff"].map(
            lambda value: len(value) == 5 and set(value).issubset(allowed_aff)
        )
        if mask.any():
            blocks.append(chunk.loc[mask].copy())
    if not blocks:
        return {}
    selected = pd.concat(blocks, ignore_index=True)
    _require(not selected[["pep", "aff"]].duplicated().any(), f"duplicate target pair in {path}")
    _require(selected["count"].gt(0).all(), f"non-positive count in {path}")
    return {
        peptide: group.set_index("aff", verify_integrity=True)["count"].astype(np.uint32)
        for peptide, group in selected.groupby("pep", sort=True)
    }


def _map_uint32(index: pd.Index, series: pd.Series | None) -> np.ndarray:
    if series is None:
        return np.zeros(len(index), dtype=np.uint32)
    return index.to_series(index=index).map(series).fillna(0).to_numpy(dtype=np.uint32)


def _candidate_frame(
    peptide: str,
    peptide_full_sequence: str,
    affibody_codes: np.ndarray,
    affibody_sequences: np.ndarray,
    affibody_uids: np.ndarray,
    affibody_hashes: np.ndarray,
    chain1: str,
    measured_affibodies: set[str],
    selected_positive_affibodies: set[str],
    round_counts: dict[int, dict[str, pd.Series]],
    weak_annotations: dict[str, pd.DataFrame],
    seen_training_affibodies: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.Index(affibody_codes, name="affibody_design_code")
    frame = pd.DataFrame(
        {
            "library": LIBRARY,
            "peptide_design_code": peptide,
            "peptide_full_sequence": peptide_full_sequence,
            "affibody_design_code": affibody_codes,
            "chain1_smart_hla_linker_peptide_sequence": chain1,
            "chain2_affibody_sequence": affibody_sequences,
            "peptide_uid": opaque_id(LIBRARY, "pep", peptide),
            "affibody_uid": affibody_uids,
            "chain1_sha256": _sha256_text(chain1),
            "chain2_sha256": affibody_hashes,
        }
    )
    frame["pair_uid"] = np.asarray(
        [opaque_id(LIBRARY, peptide, code) for code in affibody_codes], dtype=object
    )
    for round_index in RAW_ROUNDS:
        series = round_counts[round_index].get(peptide)
        frame[f"r{round_index:03d}_count"] = _map_uint32(index, series)

    later_columns = [f"r{round_index:03d}_count" for round_index in range(2, 15)]
    raw_columns = [f"r{round_index:03d}_count" for round_index in RAW_ROUNDS]
    frame["pooled_r009_r010_count"] = (
        frame["r009_count"].astype(np.uint64) + frame["r010_count"].astype(np.uint64)
    ).astype(np.uint32)
    frame["observed_in_any_raw_round"] = frame[raw_columns].gt(0).any(axis=1)
    frame["observed_in_r009_or_r010"] = frame["pooled_r009_r010_count"].gt(0)
    frame["observed_in_any_r002_r014"] = frame[later_columns].gt(0).any(axis=1)
    frame["high_confidence_weak_negative"] = (
        frame["r001_count"].ge(3) & ~frame["observed_in_any_r002_r014"]
    )
    frame["affibody_identity_seen_in_strict_training"] = frame[
        "affibody_design_code"
    ].isin(seen_training_affibodies)

    annotation = weak_annotations.get(peptide)
    if annotation is None:
        frame["prior_weak_label"] = pd.Series(pd.array([pd.NA] * len(frame), dtype="Int8"))
        frame["prior_weak_label_source"] = ""
    else:
        frame["prior_weak_label"] = pd.array(
            index.to_series(index=index).map(annotation["weak_label"]), dtype="Int8"
        )
        frame["prior_weak_label_source"] = (
            index.to_series(index=index)
            .map(annotation["weak_label_source"])
            .fillna("")
            .to_numpy(dtype=object)
        )

    measured_mask = frame["affibody_design_code"].isin(measured_affibodies)
    selected_mask = frame["affibody_design_code"].isin(selected_positive_affibodies)
    keep = ~(measured_mask | selected_mask)

    excluded = frame.loc[~keep, [
        "pair_uid",
        "peptide_design_code",
        "peptide_full_sequence",
        "affibody_design_code",
        "chain2_affibody_sequence",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
    ]].copy()
    excluded["excluded_directly_measured"] = measured_mask[~keep].to_numpy()
    excluded["excluded_pooled_r009_r010_top2pct"] = selected_mask[~keep].to_numpy()

    candidates = frame.loc[keep].copy().reset_index(drop=True)
    _require(not candidates["prior_weak_label"].eq(1).any(),
             f"pooled-positive pair survived for peptide {peptide}")
    _require(not candidates["pair_uid"].isin(set(excluded.loc[
        excluded["excluded_directly_measured"], "pair_uid"
    ])).any(), f"measured pair survived for peptide {peptide}")
    candidates.insert(0, "candidate_tier", "existing_target_selection_missed_libb_design")
    candidates.insert(1, "selection_missed", True)
    return candidates, excluded


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--corrected-panel", required=True, type=Path)
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--weak-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args(argv)


def run(args) -> None:
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.corrected_panel.is_file(), "corrected panel does not exist")
    _require(args.sequence_zip.is_file(), "provider sequence ZIP does not exist")
    _require(args.weak_labels.is_file(), "weak-label table does not exist")
    _require(args.chunksize > 0, "chunksize must be positive")
    round_files = discover_round_files(args.raw_root)

    source_hashes = {
        "corrected_panel": sha256_file(args.corrected_panel),
        "sequence_zip": sha256_file(args.sequence_zip),
        "weak_labels": sha256_file(args.weak_labels),
    }
    for round_index, path in sorted(round_files.items()):
        source_hashes[f"R{round_index:03d}"] = sha256_file(path)

    panel, peptides, peptide_full_sequences = load_panel(args.corrected_panel)
    templates, sequence_member, sequence_deck_sha256 = load_provider_templates(args.sequence_zip)
    template = templates[LIBRARY]
    affibody_template = template[-CHAIN2_LENGTH:]
    _require(
        tuple(index + 1 for index, value in enumerate(affibody_template) if value == "X")
        == EXPECTED_AFFIBODY_X_POSITIONS[LIBRARY],
        "LibB Affibody placeholder positions changed",
    )

    round9 = load_count_round(round_files[9])
    round10 = load_count_round(round_files[10])
    selected, pooled_summary = pooled_positive_membership(round9, round10)
    del round9, round10

    allowed_aff = set(AFFIBODY_DESIGN_ALPHABET)
    selected_target = selected.loc[
        selected["pep"].isin(peptides)
        & selected["aff"].map(lambda value: set(value).issubset(allowed_aff))
    ].copy()
    selected_by_peptide = {
        peptide: set(group["aff"])
        for peptide, group in selected_target.groupby("pep", sort=True)
    }

    measured_by_peptide = {
        peptide: set(group["affibody_design_code"])
        for peptide, group in panel.groupby("peptide_design_code", sort=True)
    }
    measured_keys = set(zip(panel["peptide_design_code"], panel["affibody_design_code"]))
    positive_keys = set(zip(selected_target["pep"], selected_target["aff"]))
    selected_measured_overlap = measured_keys & positive_keys
    selection_positive_nonmeasured = positive_keys - measured_keys
    _require(
        len(selection_positive_nonmeasured) == EXPECTED_SELECTION_POSITIVE_EXCLUSIONS,
        "selection-positive non-measured exclusion count changed",
    )

    weak_annotations, seen_training_affibodies, weak_summary = load_weak_label_annotations(
        args.weak_labels, set(peptides)
    )
    weak_positive_keys = {
        (peptide, affibody)
        for peptide, annotation in weak_annotations.items()
        for affibody, row in annotation.iterrows()
        if int(row["weak_label"]) == 1
    }
    _require(
        weak_positive_keys == selection_positive_nonmeasured,
        "recomputed target pooled-positive membership differs from weak-label artifact",
    )

    round_counts = {}
    raw_target_rows = {}
    for round_index, path in sorted(round_files.items()):
        per_peptide = load_target_round_counts(path, set(peptides), args.chunksize)
        round_counts[round_index] = per_peptide
        raw_target_rows[f"R{round_index:03d}"] = int(sum(len(value) for value in per_peptide.values()))

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    partitions_dir = output_dir / "candidates_by_peptide"
    partitions_dir.mkdir(mode=0o700)

    affibody_codes = np.asarray(
        ["".join(code) for code in itertools.product(AFFIBODY_DESIGN_ALPHABET, repeat=5)],
        dtype=object,
    )
    _require(len(affibody_codes) == EXPECTED_AFFIBODY_CODES_PER_PEPTIDE,
             "Affibody enumeration size changed")
    _require(len(set(affibody_codes)) == len(affibody_codes), "duplicate enumerated Affibody code")
    affibody_sequences = np.asarray(
        [fill_template(affibody_template, code) for code in affibody_codes], dtype=object
    )
    affibody_uids = np.asarray(
        [opaque_id(LIBRARY, "aff", code) for code in affibody_codes], dtype=object
    )
    affibody_hashes = np.asarray([_sha256_text(sequence) for sequence in affibody_sequences], dtype=object)

    partition_rows = []
    excluded_blocks = []
    seen_pair_uids = set()
    for peptide in peptides:
        full_construct = fill_template(template, peptide + AFFIBODY_DESIGN_ALPHABET[0] * 5)
        chain1 = full_construct[:CHAIN1_LENGTH]
        _require(chain1[-9:] == peptide_full_sequences[peptide],
                 f"provider sequence reconstruction changed for {peptide}")
        candidates, excluded = _candidate_frame(
            peptide=peptide,
            peptide_full_sequence=peptide_full_sequences[peptide],
            affibody_codes=affibody_codes,
            affibody_sequences=affibody_sequences,
            affibody_uids=affibody_uids,
            affibody_hashes=affibody_hashes,
            chain1=chain1,
            measured_affibodies=measured_by_peptide[peptide],
            selected_positive_affibodies=selected_by_peptide.get(peptide, set()),
            round_counts=round_counts,
            weak_annotations=weak_annotations,
            seen_training_affibodies=seen_training_affibodies,
        )
        _require(not candidates["pair_uid"].isin(seen_pair_uids).any(), "duplicate pair UID across partitions")
        seen_pair_uids.update(candidates["pair_uid"])
        path = partitions_dir / f"peptide_{peptide}.parquet"
        candidates.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        os.chmod(path, 0o600)
        partition_rows.append(
            {
                "peptide_design_code": peptide,
                "peptide_full_sequence": peptide_full_sequences[peptide],
                "candidate_rows": int(len(candidates)),
                "measured_pairs_excluded": int(excluded["excluded_directly_measured"].sum()),
                "selection_positive_pairs_excluded": int(
                    ((~excluded["excluded_directly_measured"])
                     & excluded["excluded_pooled_r009_r010_top2pct"]).sum()
                ),
                "weak_negative_candidates": int(candidates["prior_weak_label"].eq(0).sum()),
                "high_confidence_weak_negative_candidates": int(
                    candidates["high_confidence_weak_negative"].sum()
                ),
                "unobserved_in_all_raw_rounds": int(
                    (~candidates["observed_in_any_raw_round"]).sum()
                ),
                "affibody_identity_unseen_in_strict_training": int(
                    (~candidates["affibody_identity_seen_in_strict_training"]).sum()
                ),
                "partition": str(path.relative_to(output_dir)),
                "partition_sha256": sha256_file(path),
                "partition_bytes": int(path.stat().st_size),
            }
        )
        excluded_blocks.append(excluded)

    partition_table = pd.DataFrame(partition_rows)
    _require(partition_table["candidate_rows"].sum() == EXPECTED_CANDIDATES,
             "total immediate candidate count changed")
    _require(partition_table["measured_pairs_excluded"].sum() == EXPECTED_MEASURED_EXCLUSIONS,
             "measured exclusion count changed")
    _require(
        partition_table["selection_positive_pairs_excluded"].sum()
        == EXPECTED_SELECTION_POSITIVE_EXCLUSIONS,
        "selection-positive exclusion count changed",
    )

    excluded = pd.concat(excluded_blocks, ignore_index=True)
    excluded = excluded.sort_values(
        ["peptide_design_code", "affibody_design_code"], kind="mergesort"
    ).reset_index(drop=True)
    exclusions_path = output_dir / "excluded_pairs.parquet"
    excluded.to_parquet(exclusions_path, index=False, engine="pyarrow", compression="zstd")
    os.chmod(exclusions_path, 0o600)

    partitions_path = output_dir / "candidate_partitions.csv"
    partition_table.to_csv(partitions_path, index=False)
    os.chmod(partitions_path, 0o600)

    manifest = {
        "schema_version": "libb-existing-target-selection-missed-candidates-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "scope": {
            "library": LIBRARY,
            "tier": "existing targets, selection-missed Affibodies",
            "target_peptide_codes": peptides,
            "target_peptide_full_sequences": peptide_full_sequences,
            "affibody_mutable_positions": {
                "provider_displayed_58aa_positions_1_based": [6, 10, 13, 14, 17],
                "provider_crystal_aligned_labels_1_based": [8, 12, 15, 16, 19],
                "python_indices_0_based": [5, 9, 12, 13, 16],
            },
            "affibody_design_alphabet_each_position": "".join(AFFIBODY_DESIGN_ALPHABET),
            "codes_enumerated_per_peptide": int(len(affibody_codes)),
            "full_20aa_codes_per_peptide_not_in_this_tier": 20**5,
            "hidden_MA_policy": (
                "The locally available provider template is the displayed 58-aa sequence. "
                "The two hidden preceding residues MA affect residue labels only and are not "
                "silently inserted into model-input or candidate sequences."
            ),
        },
        "selection_rule": {
            "operation": "outer-join R009/R010 by exact pair, zero-fill, sum counts, then tie-inclusive top 2%",
            "top_fraction": TOP_FRACTION,
            **pooled_summary,
        },
        "exclusions": {
            "directly_measured_exact_pairs": int(EXPECTED_MEASURED_EXCLUSIONS),
            "pooled_positive_nonmeasured_pairs": int(EXPECTED_SELECTION_POSITIVE_EXCLUSIONS),
            "measured_and_pooled_positive_overlap": int(len(selected_measured_overlap)),
            "excluded_union_rows": int(len(excluded)),
        },
        "outputs": {
            "candidate_rows": int(partition_table["candidate_rows"].sum()),
            "candidate_partitions": int(len(partition_table)),
            "candidate_partitions_csv_sha256": sha256_file(partitions_path),
            "excluded_pairs_rows": int(len(excluded)),
            "excluded_pairs_sha256": sha256_file(exclusions_path),
        },
        "weak_label_audit": weak_summary,
        "raw_target_rows": raw_target_rows,
        "source_paths": {
            "corrected_panel": str(args.corrected_panel.resolve()),
            "sequence_zip": str(args.sequence_zip.resolve()),
            "sequence_deck_member": sequence_member,
            "sequence_deck_sha256": sequence_deck_sha256,
            "weak_labels": str(args.weak_labels.resolve()),
            "raw_rounds": {
                f"R{index:03d}": str(path.resolve()) for index, path in sorted(round_files.items())
            },
        },
        "source_sha256": source_hashes,
        "no_model_scores": True,
        "prospective_warning": (
            "These rows are a searchable candidate universe, not wet-lab recommendations. "
            "Model scoring, ensemble selection, sequence-quality review, and provider approval "
            "must occur before synthesis."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    # Recheck source immutability after all output writes.
    _require(sha256_file(args.corrected_panel) == source_hashes["corrected_panel"],
             "corrected panel changed during build")
    _require(sha256_file(args.sequence_zip) == source_hashes["sequence_zip"],
             "sequence ZIP changed during build")
    _require(sha256_file(args.weak_labels) == source_hashes["weak_labels"],
             "weak labels changed during build")
    for round_index, path in sorted(round_files.items()):
        _require(sha256_file(path) == source_hashes[f"R{round_index:03d}"],
                 f"raw R{round_index:03d} changed during build")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
