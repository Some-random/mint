#!/usr/bin/env python3
"""Build the retention-blind prospective LibA candidate universe.

For each of the nine measured LibA peptide targets, this program enumerates
all four-position Affibody codes over the provider's 15-amino-acid design
alphabet.  It removes exact pairs that are already in the 108-pair measured
panel and exact pairs in the tie-inclusive top 2% after R009 and R010 counts
are pooled.  Every surviving row retains both complete model-input sequences
and the raw R000--R014 counts.

The measured-panel CSV is opened with an explicit membership/sequence column
allow-list.  Retention values and binder labels are never loaded.  Outputs are
partitioned by peptide, existing paths are never overwritten, and the private
manifest is written last as the completion marker.
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


LIBRARY = "LibA"
RAW_ROUNDS = tuple(range(15))
POSITIVE_ROUNDS = (9, 10)
TOP_FRACTION = 0.02
AFFIBODY_CODE_LENGTH = 4
EXPECTED_PANEL_ROWS = 108
EXPECTED_TARGET_PEPTIDES = 9
EXPECTED_MEASURED_AFFIBODIES_PER_PEPTIDE = 12

PEPTIDE_DESIGN_ALPHABET = frozenset("ADEFHIKLNPQSTVY")
AFFIBODY_DESIGN_ALPHABET = tuple("ADEFHIKLNPQSTVY")
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

RAW_FILE_PATTERN = re.compile(r"(\d{3})n?_count_+freq_pvalue\.tsv$")
RAW_COLUMNS = ("pep", "aff", "count", "frequency", "pvalue")
PANEL_COLUMNS = (
    "pair_uid",
    "library",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
)
FORBIDDEN_PANEL_OUTCOME_COLUMNS = frozenset(
    {
        "retention_percent",
        "retention_time_min_as_labeled",
        "target_retention",
        "target_binder",
        "binder_label_ge_75",
        "measurement_missing",
    }
)
WEAK_LABEL_COLUMNS = (
    "library",
    "pep",
    "aff",
    "weak_label",
    "weak_label_source",
    "within_declared_library_alphabet",
    "negative_r001_count_ge_3",
    "strict_retention_identity_cold_eligible",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _file_receipt(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def discover_round_files(raw_root: Path) -> dict[int, Path]:
    directory = raw_root / "LibA Raw data"
    _require(directory.is_dir(), f"missing LibA raw directory: {directory}")
    result: dict[int, Path] = {}
    for path in sorted(directory.glob("*.tsv")):
        match = RAW_FILE_PATTERN.search(path.name)
        _require(match is not None, f"cannot parse round from {path.name}")
        round_index = int(match.group(1))
        _require(round_index not in result, f"duplicate raw round R{round_index:03d}")
        result[round_index] = path
    _require(tuple(sorted(result)) == RAW_ROUNDS, "LibA raw rounds R000--R014 are incomplete")
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
    _require(frame["aff"].str.len().eq(AFFIBODY_CODE_LENGTH).all(), f"bad Affibody code in {path}")
    _require(frame["count"].gt(0).all(), f"non-positive count in {path}")
    _require(
        frame["pep"].map(lambda value: set(value).issubset(AA_ALPHABET)).all(),
        f"noncanonical peptide code in {path}",
    )
    _require(
        frame["aff"].map(lambda value: set(value).issubset(AA_ALPHABET)).all(),
        f"noncanonical Affibody code in {path}",
    )
    return frame[["pep", "aff", "count"]].copy()


def pooled_positive_membership(
    round9: pd.DataFrame,
    round10: pd.DataFrame,
    top_fraction: float = TOP_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Pool first, then apply a tie-inclusive top-fraction cutoff."""

    _require(0.0 < top_fraction <= 1.0, "top fraction must be in (0, 1]")
    left = round9.rename(columns={"count": "r009_count"})
    right = round10.rename(columns={"count": "r010_count"})
    pooled = left.merge(right, on=["pep", "aff"], how="outer", validate="one_to_one")
    _require(len(pooled) > 0, "pooled R009+R010 membership is empty")
    for column in ("r009_count", "r010_count"):
        pooled[column] = pooled[column].fillna(0).astype(np.int64)
    pooled["pooled_r009_r010_count"] = (
        pooled["r009_count"].astype(np.uint64)
        + pooled["r010_count"].astype(np.uint64)
    )
    pooled["pooled_global_count_rank_1_based"] = (
        pooled["pooled_r009_r010_count"]
        .rank(method="min", ascending=False)
        .astype(np.int64)
    )
    rank = max(1, int(math.ceil(top_fraction * len(pooled))))
    counts = pooled["pooled_r009_r010_count"].to_numpy(dtype=np.uint64)
    cutoff = int(np.partition(counts, len(counts) - rank)[len(counts) - rank])
    selected = pooled.loc[pooled["pooled_r009_r010_count"].ge(cutoff)].copy()
    selected = selected.sort_values(
        ["pooled_r009_r010_count", "pep", "aff"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["pooled_selected_order_1_based"] = np.arange(
        1, len(selected) + 1, dtype=np.int64
    )
    _require(len(selected) >= rank, "tie-inclusive selection returned fewer than nominal rank")
    return selected, {
        "pooled_union_rows": int(len(pooled)),
        "nominal_top_fraction_rank": int(rank),
        "inclusive_count_cutoff": int(cutoff),
        "selected_rows_including_boundary_ties": int(len(selected)),
        "boundary_tie_expansion_rows": int(len(selected) - rank),
    }


def load_panel(panel_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Load only measured membership, target identity, and model-input sequences."""

    header = tuple(pd.read_csv(panel_path, nrows=0).columns)
    missing = set(PANEL_COLUMNS).difference(header)
    _require(not missing, f"measured panel lacks required columns: {sorted(missing)}")
    _require(
        FORBIDDEN_PANEL_OUTCOME_COLUMNS.isdisjoint(PANEL_COLUMNS),
        "internal error: panel allow-list contains a retention outcome",
    )
    panel = pd.read_csv(
        panel_path,
        usecols=list(PANEL_COLUMNS),
        dtype={column: str for column in PANEL_COLUMNS},
        keep_default_na=False,
        na_filter=False,
    )
    panel = panel.loc[panel["library"].eq(LIBRARY)].copy()
    _require(len(panel) == EXPECTED_PANEL_ROWS, "LibA measured panel must contain 108 pairs")
    _require(
        not panel[["peptide_design_code", "affibody_design_code"]].duplicated().any(),
        "LibA panel contains duplicate exact pairs",
    )
    _require(not panel["pair_uid"].duplicated().any(), "LibA panel contains duplicate pair UIDs")

    peptides = sorted(panel["peptide_design_code"].unique())
    _require(len(peptides) == EXPECTED_TARGET_PEPTIDES, "expected nine measured LibA peptide targets")
    per_peptide = panel.groupby("peptide_design_code")["affibody_design_code"].nunique()
    _require(
        per_peptide.eq(EXPECTED_MEASURED_AFFIBODIES_PER_PEPTIDE).all(),
        "LibA panel is not a complete 9x12 measured-pair matrix",
    )
    _require(
        panel["affibody_design_code"].str.len().eq(AFFIBODY_CODE_LENGTH).all(),
        "LibA panel contains a bad Affibody code length",
    )

    expected_pair_uids = [
        opaque_id(LIBRARY, peptide, affibody)
        for peptide, affibody in zip(
            panel["peptide_design_code"], panel["affibody_design_code"]
        )
    ]
    _require(
        panel["pair_uid"].astype(str).tolist() == expected_pair_uids,
        "LibA panel pair UID does not match exact design-code membership",
    )

    full_peptides: dict[str, str] = {}
    affibody_positions = EXPECTED_AFFIBODY_X_POSITIONS[LIBRARY]
    for peptide, group in panel.groupby("peptide_design_code", sort=True):
        chains = group["chain1_smart_hla_linker_peptide_sequence"].unique()
        _require(len(chains) == 1, f"peptide {peptide} has inconsistent chain-1 sequences")
        chain1 = str(chains[0])
        _require(len(chain1) == CHAIN1_LENGTH, f"peptide {peptide} chain-1 length changed")
        full = chain1[-9:]
        _require(full[3:5] == peptide, f"peptide {peptide} does not map to positions 4/5")
        full_peptides[peptide] = full
        for row in group.itertuples(index=False):
            chain2 = str(row.chain2_affibody_sequence)
            _require(len(chain2) == CHAIN2_LENGTH, "LibA panel chain-2 length changed")
            observed = "".join(chain2[position - 1] for position in affibody_positions)
            _require(observed == row.affibody_design_code, "LibA panel Affibody mapping changed")
    return panel, peptides, full_peptides


def load_weak_label_annotations(
    path: Path, target_peptides: set[str]
) -> tuple[dict[str, pd.DataFrame], set[str], dict]:
    frame = pd.read_csv(
        path,
        usecols=list(WEAK_LABEL_COLUMNS),
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[frame["library"].eq(LIBRARY)].copy()
    for column in (
        "weak_label",
        "within_declared_library_alphabet",
        "negative_r001_count_ge_3",
        "strict_retention_identity_cold_eligible",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise", downcast="integer")

    training = frame.loc[
        frame["within_declared_library_alphabet"].eq(1)
        & frame["strict_retention_identity_cold_eligible"].eq(1)
        & (frame["weak_label"].eq(1) | frame["negative_r001_count_ge_3"].eq(1))
    ].copy()
    _require(len(training) > 0, "strict LibA weak training membership is empty")
    _require(
        not training["pep"].isin(target_peptides).any(),
        "a measured LibA peptide leaked into strict training",
    )
    seen_training_affibodies = set(training["aff"].astype(str))

    allowed_aff = set(AFFIBODY_DESIGN_ALPHABET)
    target = frame.loc[
        frame["pep"].isin(target_peptides)
        & frame["aff"].map(
            lambda value: len(value) == AFFIBODY_CODE_LENGTH
            and set(value).issubset(allowed_aff)
        )
    ].copy()
    _require(not target[["pep", "aff"]].duplicated().any(), "duplicate target weak label")
    annotations = {
        peptide: group.set_index("aff", verify_integrity=True)
        for peptide, group in target.groupby("pep", sort=True)
    }
    return annotations, seen_training_affibodies, {
        "strict_training_rows": int(len(training)),
        "strict_training_positive_rows": int(training["weak_label"].eq(1).sum()),
        "strict_training_high_confidence_negative_rows": int(
            training["negative_r001_count_ge_3"].eq(1).sum()
        ),
        "strict_training_unique_affibodies": int(len(seen_training_affibodies)),
        "strict_training_unique_peptides": int(training["pep"].nunique()),
        "target_pair_weak_labels": int(len(target)),
        "target_pair_positive_labels": int(target["weak_label"].eq(1).sum()),
        "target_pair_negative_labels": int(target["weak_label"].eq(0).sum()),
        "target_pair_high_confidence_negatives": int(
            target["negative_r001_count_ge_3"].eq(1).sum()
        ),
    }


def load_target_round_counts(
    path: Path,
    target_peptides: set[str],
    chunksize: int,
) -> dict[str, pd.Series]:
    blocks: list[pd.DataFrame] = []
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
            lambda value: len(value) == AFFIBODY_CODE_LENGTH
            and set(value).issubset(allowed_aff)
        )
        if mask.any():
            blocks.append(chunk.loc[mask].copy())
    if not blocks:
        return {}
    selected = pd.concat(blocks, ignore_index=True)
    _require(not selected[["pep", "aff"]].duplicated().any(), f"duplicate target pair in {path}")
    _require(selected["count"].gt(0).all(), f"non-positive count in {path}")
    return {
        peptide: group.set_index("aff", verify_integrity=True)["count"].astype(np.uint64)
        for peptide, group in selected.groupby("pep", sort=True)
    }


def _map_uint64(index: pd.Index, series: pd.Series | None) -> np.ndarray:
    if series is None:
        return np.zeros(len(index), dtype=np.uint64)
    return index.to_series(index=index).map(series).fillna(0).to_numpy(dtype=np.uint64)


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
    chain1_hash = _sha256_text(chain1)
    frame = pd.DataFrame(
        {
            "library": LIBRARY,
            "peptide_design_code": peptide,
            "peptide_full_sequence": peptide_full_sequence,
            # Stable handoff names are retained alongside the historical
            # chain-1/chain-2 names.  For this dataset the provider-displayed
            # 58-aa Affibody is also the exact model-input Affibody sequence.
            "peptide_9mer_sequence": peptide_full_sequence,
            "affibody_design_code": affibody_codes,
            "chain1_smart_hla_linker_peptide_sequence": chain1,
            "chain2_affibody_sequence": affibody_sequences,
            "provider_displayed_58aa_affibody_sequence": affibody_sequences,
            "model_input_affibody_sequence": affibody_sequences,
            "model_input_smart_hla_linker_peptide_sequence": chain1,
            "peptide_uid": opaque_id(LIBRARY, "pep", peptide),
            "affibody_uid": affibody_uids,
            "chain1_sha256": chain1_hash,
            "chain2_sha256": affibody_hashes,
        }
    )
    frame["pair_uid"] = np.asarray(
        [opaque_id(LIBRARY, peptide, code) for code in affibody_codes], dtype=object
    )
    frame["sequence_pair_sha256"] = np.asarray(
        [_sha256_text(chain1 + "|" + sequence) for sequence in affibody_sequences],
        dtype=object,
    )
    for round_index in RAW_ROUNDS:
        series = round_counts[round_index].get(peptide)
        frame[f"r{round_index:03d}_count"] = _map_uint64(index, series)

    later_columns = [f"r{round_index:03d}_count" for round_index in range(2, 15)]
    raw_columns = [f"r{round_index:03d}_count" for round_index in RAW_ROUNDS]
    frame["pooled_r009_r010_count"] = (
        frame["r009_count"].astype(np.uint64) + frame["r010_count"].astype(np.uint64)
    )
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

    excluded = frame.loc[
        ~keep,
        [
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "peptide_design_code",
            "peptide_full_sequence",
            "peptide_9mer_sequence",
            "affibody_design_code",
            "chain1_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence",
            "provider_displayed_58aa_affibody_sequence",
            "model_input_affibody_sequence",
            "model_input_smart_hla_linker_peptide_sequence",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "r009_count",
            "r010_count",
            "pooled_r009_r010_count",
        ],
    ].copy()
    excluded["excluded_directly_measured"] = measured_mask[~keep].to_numpy()
    excluded["excluded_pooled_r009_r010_top2pct"] = selected_mask[~keep].to_numpy()

    candidates = frame.loc[keep].copy().reset_index(drop=True)
    _require(
        not candidates["prior_weak_label"].eq(1).any(),
        f"pooled-positive pair survived for peptide {peptide}",
    )
    _require(
        not candidates["affibody_design_code"].isin(measured_affibodies).any(),
        f"measured pair survived for peptide {peptide}",
    )
    candidates.insert(0, "candidate_tier", "existing_target_selection_missed_liba_design")
    candidates.insert(1, "selection_missed", True)
    return candidates, excluded


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--measured-panel", required=True, type=Path)
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--weak-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.measured_panel.is_file(), "measured panel does not exist")
    _require(args.sequence_zip.is_file(), "provider sequence ZIP does not exist")
    _require(args.weak_labels.is_file(), "weak-label table does not exist")
    _require(args.chunksize > 0, "chunksize must be positive")
    round_files = discover_round_files(args.raw_root)

    input_paths = {
        "measured_panel": args.measured_panel,
        "sequence_zip": args.sequence_zip,
        "weak_labels": args.weak_labels,
        **{f"R{index:03d}": path for index, path in sorted(round_files.items())},
    }
    source_receipts = {name: _file_receipt(path) for name, path in input_paths.items()}

    panel, peptides, peptide_full_sequences = load_panel(args.measured_panel)
    templates, sequence_member, sequence_deck_sha256 = load_provider_templates(args.sequence_zip)
    template = templates[LIBRARY]
    affibody_template = template[-CHAIN2_LENGTH:]
    _require(
        tuple(index + 1 for index, value in enumerate(affibody_template) if value == "X")
        == EXPECTED_AFFIBODY_X_POSITIONS[LIBRARY],
        "LibA Affibody placeholder positions changed",
    )

    round9 = load_count_round(round_files[9])
    round10 = load_count_round(round_files[10])
    selected, pooled_summary = pooled_positive_membership(round9, round10)
    del round9, round10

    allowed_aff = set(AFFIBODY_DESIGN_ALPHABET)
    selected_target = selected.loc[
        selected["pep"].isin(peptides)
        & selected["aff"].map(
            lambda value: len(value) == AFFIBODY_CODE_LENGTH
            and set(value).issubset(allowed_aff)
        )
    ].copy()
    _require(
        not selected_target[["pep", "aff"]].duplicated().any(),
        "duplicate target pair in pooled-positive membership",
    )
    selected_by_peptide = {
        peptide: set(group["aff"].astype(str))
        for peptide, group in selected_target.groupby("pep", sort=True)
    }
    measured_by_peptide = {
        peptide: set(group["affibody_design_code"].astype(str))
        for peptide, group in panel.groupby("peptide_design_code", sort=True)
    }
    measured_keys = set(zip(panel["peptide_design_code"], panel["affibody_design_code"]))
    positive_keys = set(zip(selected_target["pep"], selected_target["aff"]))
    selected_measured_overlap = measured_keys & positive_keys
    selection_positive_nonmeasured = positive_keys - measured_keys
    excluded_keys = measured_keys | positive_keys

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

    round_counts: dict[int, dict[str, pd.Series]] = {}
    raw_target_rows: dict[str, int] = {}
    for round_index, path in sorted(round_files.items()):
        per_peptide = load_target_round_counts(path, set(peptides), args.chunksize)
        round_counts[round_index] = per_peptide
        raw_target_rows[f"R{round_index:03d}"] = int(
            sum(len(value) for value in per_peptide.values())
        )

    affibody_codes = np.asarray(
        [
            "".join(code)
            for code in itertools.product(
                AFFIBODY_DESIGN_ALPHABET, repeat=AFFIBODY_CODE_LENGTH
            )
        ],
        dtype=object,
    )
    codes_per_peptide = len(AFFIBODY_DESIGN_ALPHABET) ** AFFIBODY_CODE_LENGTH
    _require(len(affibody_codes) == codes_per_peptide, "Affibody enumeration size changed")
    _require(len(set(affibody_codes)) == len(affibody_codes), "duplicate enumerated Affibody code")
    universe_rows = len(peptides) * codes_per_peptide
    expected_candidate_rows = universe_rows - len(excluded_keys)

    affibody_sequences = np.asarray(
        [fill_template(affibody_template, code) for code in affibody_codes], dtype=object
    )
    affibody_uids = np.asarray(
        [opaque_id(LIBRARY, "aff", code) for code in affibody_codes], dtype=object
    )
    affibody_hashes = np.asarray(
        [_sha256_text(sequence) for sequence in affibody_sequences], dtype=object
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    partitions_dir = output_dir / "candidates_by_peptide"
    partitions_dir.mkdir(mode=0o700)

    partition_rows: list[dict] = []
    excluded_blocks: list[pd.DataFrame] = []
    seen_pair_uids: set[str] = set()
    for peptide in peptides:
        full_construct = fill_template(
            template, peptide + AFFIBODY_DESIGN_ALPHABET[0] * AFFIBODY_CODE_LENGTH
        )
        chain1 = full_construct[:CHAIN1_LENGTH]
        _require(
            chain1[-9:] == peptide_full_sequences[peptide],
            f"provider sequence reconstruction changed for {peptide}",
        )
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
        expected_partition_rows = codes_per_peptide - len(
            measured_by_peptide[peptide] | selected_by_peptide.get(peptide, set())
        )
        _require(
            len(candidates) == expected_partition_rows,
            f"candidate count does not match input-derived exclusion set for {peptide}",
        )
        _require(
            not candidates["pair_uid"].isin(seen_pair_uids).any(),
            "duplicate pair UID across partitions",
        )
        seen_pair_uids.update(candidates["pair_uid"].astype(str))
        path = partitions_dir / f"peptide_{peptide}.parquet"
        candidates.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        os.chmod(path, 0o600)
        partition_rows.append(
            {
                "peptide_design_code": peptide,
                "peptide_full_sequence": peptide_full_sequences[peptide],
                "universe_rows_before_exclusion": int(codes_per_peptide),
                "candidate_rows": int(len(candidates)),
                "measured_pairs_excluded": int(excluded["excluded_directly_measured"].sum()),
                "selection_positive_pairs_excluded": int(
                    (
                        (~excluded["excluded_directly_measured"])
                        & excluded["excluded_pooled_r009_r010_top2pct"]
                    ).sum()
                ),
                "measured_and_selection_positive_overlap": int(
                    (
                        excluded["excluded_directly_measured"]
                        & excluded["excluded_pooled_r009_r010_top2pct"]
                    ).sum()
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
    _require(
        int(partition_table["candidate_rows"].sum()) == expected_candidate_rows,
        "total candidate count differs from input-derived expectation",
    )
    _require(
        int(partition_table["measured_pairs_excluded"].sum()) == len(measured_keys),
        "measured exclusion count differs from panel membership",
    )
    _require(
        int(partition_table["selection_positive_pairs_excluded"].sum())
        == len(selection_positive_nonmeasured),
        "selection-positive exclusion count differs from pooled membership",
    )

    excluded_union = pd.concat(excluded_blocks, ignore_index=True)
    excluded_union = excluded_union.sort_values(
        ["peptide_design_code", "affibody_design_code"], kind="mergesort"
    ).reset_index(drop=True)
    _require(len(excluded_union) == len(excluded_keys), "excluded union row count changed")
    _require(
        not excluded_union["pair_uid"].duplicated().any(),
        "duplicate excluded pair UID",
    )

    # Keep measured exclusions and the unmeasured pooled-positive controls in
    # distinct machine-readable files.  A pair that is both measured and a
    # pooled positive belongs only to the measured file.
    measured_exclusions = excluded_union.loc[
        excluded_union["excluded_directly_measured"]
    ].copy()
    _require(len(measured_exclusions) == len(measured_keys), "measured exclusion tier changed")
    measured_exclusions.insert(0, "exclusion_tier", "directly_measured_liba_pair")
    measured_exclusions["directly_measured"] = True
    measured_exclusions["pooled_r009_r010_top2pct"] = measured_exclusions[
        "excluded_pooled_r009_r010_top2pct"
    ].astype(bool)

    pooled_positive_unmeasured = excluded_union.loc[
        (~excluded_union["excluded_directly_measured"])
        & excluded_union["excluded_pooled_r009_r010_top2pct"]
    ].copy()
    rank_columns = selected_target[
        [
            "pep",
            "aff",
            "pooled_global_count_rank_1_based",
            "pooled_selected_order_1_based",
        ]
    ].rename(columns={"pep": "peptide_design_code", "aff": "affibody_design_code"})
    pooled_positive_unmeasured = pooled_positive_unmeasured.merge(
        rank_columns,
        on=["peptide_design_code", "affibody_design_code"],
        how="left",
        validate="one_to_one",
    )
    _require(
        len(pooled_positive_unmeasured) == len(selection_positive_nonmeasured),
        "unmeasured pooled-positive control tier changed",
    )
    _require(
        not pooled_positive_unmeasured[
            ["pooled_global_count_rank_1_based", "pooled_selected_order_1_based"]
        ].isna().any().any(),
        "unmeasured pooled-positive control is missing rank provenance",
    )
    pooled_positive_unmeasured.insert(
        0,
        "candidate_tier",
        "existing_target_pooled_positive_unmeasured_liba_design",
    )
    pooled_positive_unmeasured["directly_measured"] = False
    pooled_positive_unmeasured["pooled_r009_r010_top2pct"] = True

    measured_path = output_dir / "measured_exclusions.parquet"
    pooled_control_path = output_dir / "pooled_positive_unmeasured_controls.parquet"
    measured_exclusions.to_parquet(
        measured_path, index=False, engine="pyarrow", compression="zstd"
    )
    pooled_positive_unmeasured.to_parquet(
        pooled_control_path, index=False, engine="pyarrow", compression="zstd"
    )
    for path in (measured_path, pooled_control_path):
        os.chmod(path, 0o600)

    partitions_path = output_dir / "candidate_partitions.csv"
    partition_table.to_csv(partitions_path, index=False)
    os.chmod(partitions_path, 0o600)

    # Recheck every input before publishing the completion manifest.
    for name, path in input_paths.items():
        _require(
            sha256_file(path) == source_receipts[name]["sha256"],
            f"source {name} changed during build",
        )

    manifest = {
        "schema_version": "liba-existing-target-selection-missed-candidates-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "scope": {
            "library": LIBRARY,
            "tier": "existing targets, selection-missed Affibodies",
            "target_peptide_codes": peptides,
            "target_peptide_full_sequences": peptide_full_sequences,
            "affibody_mutable_positions": {
                "provider_displayed_58aa_positions_1_based": [13, 17, 27, 31],
                "provider_crystal_aligned_labels_1_based": [15, 19, 29, 33],
                "python_indices_0_based": [12, 16, 26, 30],
            },
            "affibody_design_alphabet_each_position": "".join(AFFIBODY_DESIGN_ALPHABET),
            "affibody_code_length": AFFIBODY_CODE_LENGTH,
            "codes_enumerated_per_peptide": int(codes_per_peptide),
            "universe_rows_before_exclusion": int(universe_rows),
            "full_20aa_codes_per_peptide_not_in_this_tier": 20**AFFIBODY_CODE_LENGTH,
            "hidden_MA_policy": (
                "The locally available provider template is the displayed 58-aa sequence. "
                "The two hidden preceding residues MA affect residue labels only and are not "
                "silently inserted into model-input or candidate sequences."
            ),
        },
        "selection_rule": {
            "operation": (
                "outer-join R009/R010 by exact pair, zero-fill, sum counts, "
                "then apply the tie-inclusive global top 2% cutoff"
            ),
            "top_fraction": TOP_FRACTION,
            **pooled_summary,
            "selected_target_pairs_in_design_alphabet": int(len(positive_keys)),
        },
        "exclusions": {
            "directly_measured_exact_pairs": int(len(measured_keys)),
            "pooled_positive_target_pairs": int(len(positive_keys)),
            "pooled_positive_nonmeasured_pairs": int(len(selection_positive_nonmeasured)),
            "measured_and_pooled_positive_overlap": int(len(selected_measured_overlap)),
            "excluded_union_rows": int(len(excluded_keys)),
            "partition_policy": (
                "measured pairs and unmeasured pooled-positive pairs are emitted in "
                "separate files; measured-and-positive overlaps remain in the measured tier"
            ),
        },
        "outputs": {
            "candidate_rows": int(partition_table["candidate_rows"].sum()),
            "candidate_partitions": int(len(partition_table)),
            "candidate_partitions_csv": _file_receipt(partitions_path),
            "measured_exclusions_rows": int(len(measured_exclusions)),
            "measured_exclusions": _file_receipt(measured_path),
            "pooled_positive_unmeasured_control_rows": int(
                len(pooled_positive_unmeasured)
            ),
            "pooled_positive_unmeasured_controls": _file_receipt(pooled_control_path),
            "partition_files": [
                {
                    "peptide_design_code": str(row.peptide_design_code),
                    "path": str(row.partition),
                    "rows": int(row.candidate_rows),
                    "sha256": str(row.partition_sha256),
                    "bytes": int(row.partition_bytes),
                }
                for row in partition_table.itertuples(index=False)
            ],
        },
        "weak_label_audit": weak_summary,
        "raw_target_rows": raw_target_rows,
        "input_data_contract": {
            "measured_panel_columns_read": list(PANEL_COLUMNS),
            "retention_outcome_columns_read": [],
            "measured_panel_usage": (
                "exact pair exclusion, peptide-target identity, and model-input sequence "
                "validation only"
            ),
            "weak_label_columns_read": list(WEAK_LABEL_COLUMNS),
        },
        "source_files": source_receipts,
        "sequence_deck": {
            "archive_member": sequence_member,
            "embedded_deck_sha256": sequence_deck_sha256,
        },
        "immutability": {
            "existing_output_directory_refused": True,
            "manifest_written_last": True,
            "directory_mode": "0700",
            "file_mode": "0600",
        },
        "no_model_scores": True,
        "prospective_warning": (
            "These rows are a searchable candidate universe, not wet-lab recommendations. "
            "Model scoring, sequence-quality review, and provider approval must occur before "
            "synthesis."
        ),
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
