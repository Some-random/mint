#!/usr/bin/env python3
"""Annotate a frozen LibB candidate menu with rank-preserving QC checks.

This is a post-assembly audit, not a model or selector.  It never changes the
candidate set, target assignment, score, or within-peptide rank.  The weak-label
table is read only to reconstruct the exact Affibody-code membership of the
strict weak-training set; no direct-retention outcome column is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = "LibB"
FALTA_CODE = "FALTA"
LIBB_DESIGN_ALPHABET = frozenset("ADEFIKLMNSTVY")
EXPECTED_STRICT_TRAINING_ROWS = 30_648
EXPECTED_STRICT_TRAINING_AFFIBODIES = 23_081
EXPECTED_TARGETS = frozenset((
    "AF", "AH", "DP", "EA", "EL", "LL", "LV", "MW", "NF", "PH", "TL", "VV",
))
WEAK_COLUMNS_READ = (
    "library",
    "pep",
    "aff",
    "weak_label",
    "within_declared_library_alphabet",
    "negative_r001_count_ge_3",
    "strict_retention_identity_cold_eligible",
)
# This field contains the word "retention" but is identity-overlap metadata,
# not a measurement or outcome.  It is necessary to reproduce the training set.
SAFE_RETENTION_METADATA = frozenset(("strict_retention_identity_cold_eligible",))
OUTCOME_FRAGMENTS = (
    "retention",
    "target_binder",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
    "measurement_missing",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden_outcome_columns(
    columns: Iterable[str], allowed_metadata: Iterable[str] = (),
) -> List[str]:
    allowed = {str(value).strip().lower() for value in allowed_metadata}
    leaked = []
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in allowed:
            continue
        if any(fragment in normalized for fragment in OUTCOME_FRAGMENTS):
            leaked.append(str(column))
    return sorted(set(leaked))


def _read_outcome_free_csv(path: Path, label: str) -> pd.DataFrame:
    _require(path.is_file(), "%s is missing: %s" % (label, path))
    header = pd.read_csv(path, nrows=0)
    leaked = forbidden_outcome_columns(header.columns)
    _require(not leaked, "%s contains forbidden outcome columns: %s" % (label, leaked))
    return pd.read_csv(path, dtype={"pair_uid": str})


def _numeric_binary(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(numeric).all()), "%s contains a non-finite value" % name)
    _require(bool(np.equal(numeric, np.rint(numeric)).all()),
             "%s contains a non-integral value" % name)
    values = pd.Series(numeric.astype(np.int64), index=series.index)
    _require(set(values.unique()).issubset({0, 1}), "%s is not binary" % name)
    return values


def _boolean(series: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    _require(not series.isna().any(), "%s contains missing values" % name)
    converted = series.map(mapping)
    _require(not converted.isna().any(), "%s contains a non-boolean value" % name)
    return converted.astype(bool)


def hamming(left: str, right: str) -> int:
    _require(len(left) == len(right) == 5, "LibB Affibody codes must have length five")
    return sum(a != b for a, b in zip(left, right))


def load_strict_training_membership(
    path: Path,
) -> Tuple[pd.DataFrame, Set[str], Set[Tuple[str, str]], Dict[str, int]]:
    """Read only the columns needed to recover exact strict-training membership."""
    _require(path.is_file(), "weak-label table is missing: %s" % path)
    header = pd.read_csv(path, nrows=0)
    _require(set(WEAK_COLUMNS_READ).issubset(header.columns),
             "weak-label table lacks strict-training membership columns")
    leaked = forbidden_outcome_columns(header.columns, allowed_metadata=(
        "strict_retention_identity_cold_eligible",
        "shares_retention_peptide",
        "shares_retention_affibody",
    ))
    _require(not leaked, "weak-label table contains forbidden outcomes: %s" % leaked)
    frame = pd.read_csv(
        path,
        usecols=list(WEAK_COLUMNS_READ),
        dtype={"library": str, "pep": str, "aff": str},
    )
    for column in (
        "weak_label",
        "within_declared_library_alphabet",
        "negative_r001_count_ge_3",
        "strict_retention_identity_cold_eligible",
    ):
        frame[column] = _numeric_binary(frame[column], column)
    libb = frame.loc[frame["library"].eq(LIBRARY)].copy()
    strict = libb.loc[
        libb["within_declared_library_alphabet"].eq(1)
        & libb["strict_retention_identity_cold_eligible"].eq(1)
        & (
            libb["weak_label"].eq(1)
            | libb["negative_r001_count_ge_3"].eq(1)
        )
    ].copy()
    _require(len(strict) == EXPECTED_STRICT_TRAINING_ROWS,
             "strict LibB training row count changed: %d" % len(strict))
    _require(not strict[["pep", "aff"]].duplicated().any(),
             "strict-training weak table contains duplicate pairs")
    _require(strict["aff"].str.len().eq(5).all(), "bad strict-training Affibody code")
    _require(strict["aff"].map(lambda value: set(value).issubset(LIBB_DESIGN_ALPHABET)).all(),
             "strict-training Affibody uses an amino acid outside the LibB design")
    codes = set(strict["aff"].astype(str))
    _require(len(codes) == EXPECTED_STRICT_TRAINING_AFFIBODIES,
             "strict LibB Affibody identity count changed: %d" % len(codes))
    pairs = set(zip(strict["pep"].astype(str), strict["aff"].astype(str)))
    row_count = strict.groupby("aff", sort=False).size().astype(int).to_dict()
    return strict, codes, pairs, row_count


def _validate_candidate_keys(frame: pd.DataFrame, label: str) -> None:
    required = {
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "wetlab_rank",
        "ensemble_score",
    }
    _require(required.issubset(frame.columns), "%s schema is missing %s" % (
        label, sorted(required - set(frame.columns))
    ))
    _require(not frame["pair_uid"].astype(str).duplicated().any(),
             "%s contains duplicate pair_uid" % label)
    _require(set(frame["peptide_design_code"].astype(str)) == EXPECTED_TARGETS,
             "%s does not contain the exact 12 LibB targets" % label)
    _require(frame["affibody_design_code"].astype(str).str.len().eq(5).all(),
             "%s contains a non-five-character Affibody code" % label)
    _require(
        frame["affibody_design_code"].astype(str).map(
            lambda value: set(value).issubset(LIBB_DESIGN_ALPHABET)
        ).all(),
        "%s contains an Affibody outside the LibB design alphabet" % label,
    )
    scores = pd.to_numeric(frame["ensemble_score"], errors="raise").to_numpy(float)
    _require(bool(np.isfinite(scores).all()), "%s contains non-finite ensemble scores" % label)
    _require(bool(((scores >= 0.0) & (scores <= 1.0)).all()),
             "%s all4 ensemble score is outside [0,1]" % label)
    for peptide, block in frame.groupby("peptide_design_code", sort=True):
        _require(1 <= len(block) <= 10, "%s has an invalid candidate count for %s" % (
            label, peptide
        ))
        numeric_ranks = pd.to_numeric(
            block["wetlab_rank"], errors="raise"
        ).to_numpy(dtype=np.float64)
        _require(bool(np.isfinite(numeric_ranks).all()) and bool(
            np.equal(numeric_ranks, np.rint(numeric_ranks)).all()
        ), "%s has a non-integral rank for %s" % (label, peptide))
        ranks = numeric_ranks.astype(np.int64).tolist()
        _require(sorted(ranks) == list(range(1, len(block) + 1)),
                 "%s ranks are not exactly 1..N for %s" % (label, peptide))
        _require(not block["affibody_design_code"].astype(str).duplicated().any(),
                 "%s repeats an Affibody within %s" % (label, peptide))


def align_candidate_inputs(
    all4: pd.DataFrame, handoff: pd.DataFrame,
) -> pd.DataFrame:
    """Validate that the handoff is an unchanged rendering of the all4 shortlist."""
    _validate_candidate_keys(all4, "all4 shortlist")
    _validate_candidate_keys(handoff, "handoff candidate menu")
    _require(len(all4) == len(handoff), "all4 shortlist and handoff row counts differ")
    key_columns = [
        "pair_uid", "peptide_design_code", "affibody_design_code", "wetlab_rank"
    ]
    left = all4[key_columns + ["ensemble_score"]].copy()
    right = handoff[key_columns + ["ensemble_score"]].copy()
    for frame in (left, right):
        frame["pair_uid"] = frame["pair_uid"].astype(str)
        frame.sort_values("pair_uid", inplace=True, kind="mergesort")
        frame.reset_index(drop=True, inplace=True)
    _require(left[key_columns].astype(str).equals(right[key_columns].astype(str)),
             "handoff changed candidate identity, target, code, or rank")
    _require(np.allclose(
        pd.to_numeric(left["ensemble_score"]).to_numpy(float),
        pd.to_numeric(right["ensemble_score"]).to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ), "handoff changed an all4 ensemble score")
    return handoff.copy()


def _nearest_training_codes(
    selected_codes: Sequence[str], training_codes: Set[str],
) -> Dict[str, Tuple[int, str, int]]:
    ordered = sorted(training_codes)
    _require(ordered, "strict-training Affibody set is empty")
    training_matrix = np.asarray([list(code) for code in ordered], dtype="U1")
    output = {}
    for code in sorted(set(map(str, selected_codes))):
        distances = (training_matrix != np.asarray(list(code), dtype="U1")).sum(axis=1)
        minimum = int(distances.min())
        nearest_indices = np.flatnonzero(distances == minimum)
        output[code] = (minimum, ordered[int(nearest_indices[0])], len(nearest_indices))
    return output


def annotate_candidates(
    frame: pd.DataFrame,
    training_codes: Set[str],
    training_pairs: Set[Tuple[str, str]],
    training_code_row_count: Dict[str, int],
) -> pd.DataFrame:
    required = {
        "candidate_list_membership_count",
        "component_top10_count",
        "selected_by_mint",
        "selected_by_mint_stab",
        "stab_seed_probability_sd",
        "rde_seed_probability_sd",
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "high_confidence_weak_negative",
    }
    _require(required.issubset(frame.columns), "handoff lacks QC fields: %s" % sorted(
        required - set(frame.columns)
    ))
    sequence_columns = (
        "provider_displayed_58aa_affibody_sequence",
        "affibody_full_sequence",
    )
    sequence_column = next(
        (column for column in sequence_columns if column in frame.columns), None
    )
    _require(
        sequence_column is not None,
        "handoff lacks a provider-displayed/model-input Affibody sequence column",
    )
    output = frame.copy()
    codes = output["affibody_design_code"].astype(str)
    peptides = output["peptide_design_code"].astype(str)
    sequences = output[sequence_column].astype(str)

    boolean_columns = (
        "selected_by_mint",
        "selected_by_mint_stab",
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "high_confidence_weak_negative",
    )
    for column in boolean_columns:
        output[column] = _boolean(output[column], column)
    membership_numeric = pd.to_numeric(
        output["candidate_list_membership_count"], errors="raise"
    ).to_numpy(dtype=np.float64)
    top10_numeric = pd.to_numeric(
        output["component_top10_count"], errors="raise"
    ).to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(membership_numeric).all()) and bool(
        np.equal(membership_numeric, np.rint(membership_numeric)).all()
    ), "candidate-list membership count is non-integral")
    _require(bool(np.isfinite(top10_numeric).all()) and bool(
        np.equal(top10_numeric, np.rint(top10_numeric)).all()
    ), "component top-10 count is non-integral")
    membership = pd.Series(
        membership_numeric.astype(np.int64), index=output.index
    )
    top10 = pd.Series(top10_numeric.astype(np.int64), index=output.index)
    _require(membership.between(1, 3).all(), "candidate-list membership count is outside 1..3")
    _require(top10.between(0, 4).all(), "component top-10 count is outside 0..4")
    expected_membership = (
        1 + output["selected_by_mint"].astype(int)
        + output["selected_by_mint_stab"].astype(int)
    )
    _require(membership.equals(expected_membership.astype(np.int64)),
             "candidate-list membership count disagrees with selector flags")
    output["candidate_list_membership_count"] = membership
    output["component_top10_count"] = top10

    for column in ("stab_seed_probability_sd", "rde_seed_probability_sd"):
        values = pd.to_numeric(output[column], errors="raise").to_numpy(float)
        _require(bool(np.isfinite(values).all()) and bool((values >= 0.0).all()),
                 "%s is missing, negative, or non-finite" % column)
        output[column] = values

    nearest = _nearest_training_codes(codes, training_codes)
    output["qc_hamming_distance_to_FALTA"] = codes.map(
        lambda code: hamming(code, FALTA_CODE)
    ).astype(np.int64)
    output["qc_code_is_FALTA"] = codes.eq(FALTA_CODE)
    output["qc_nearest_strict_training_affibody_hamming_distance"] = codes.map(
        lambda code: nearest[code][0]
    ).astype(np.int64)
    output["qc_nearest_strict_training_affibody_code"] = codes.map(
        lambda code: nearest[code][1]
    )
    output["qc_nearest_strict_training_affibody_tie_count"] = codes.map(
        lambda code: nearest[code][2]
    ).astype(np.int64)
    output["qc_exact_affibody_code_seen_in_strict_training"] = codes.isin(training_codes)
    output["qc_exact_affibody_code_training_row_count"] = codes.map(
        lambda code: int(training_code_row_count.get(code, 0))
    ).astype(np.int64)
    output["qc_exact_peptide_affibody_pair_seen_in_strict_training"] = [
        (peptide, code) in training_pairs for peptide, code in zip(peptides, codes)
    ]
    output["qc_strict_training_affibody_status"] = np.where(
        output["qc_exact_affibody_code_seen_in_strict_training"],
        "exact_affibody_code_seen_in_strict_training",
        "exact_affibody_code_unseen_in_strict_training",
    )

    upstream_seen_column = "affibody_identity_seen_in_strict_training"
    if upstream_seen_column in output.columns:
        upstream = _boolean(output[upstream_seen_column], upstream_seen_column)
        _require(upstream.equals(output["qc_exact_affibody_code_seen_in_strict_training"]),
                 "upstream strict-training Affibody annotation disagrees with weak_labels.csv")

    target_reuse = output.groupby("affibody_design_code")[
        "peptide_design_code"
    ].transform("nunique").astype(np.int64)
    pair_reuse = output.groupby("affibody_design_code")["pair_uid"].transform("size").astype(
        np.int64
    )
    target_names = output.groupby("affibody_design_code")[
        "peptide_design_code"
    ].transform(lambda values: ";".join(sorted(set(map(str, values)))))
    output["qc_selected_for_n_peptide_targets"] = target_reuse
    output["qc_selected_pair_count_for_affibody"] = pair_reuse
    output["qc_selected_for_peptide_targets"] = target_names
    output["qc_affibody_repeated_across_targets"] = target_reuse.gt(1)
    output["qc_affibody_code_contains_cysteine"] = codes.str.contains("C", regex=False)
    output["qc_full_affibody_contains_cysteine"] = sequences.str.contains("C", regex=False)
    output["qc_full_affibody_contains_N_X_S_or_T_motif"] = sequences.map(
        lambda sequence: bool(re.search(r"N[^P][ST]", sequence))
    )

    flags = []
    for row in output.itertuples(index=False):
        current = []
        if getattr(row, "qc_code_is_FALTA"):
            current.append("code_is_FALTA")
        if getattr(row, "qc_affibody_repeated_across_targets"):
            current.append("Affibody_repeated_across_targets")
        if getattr(row, "qc_exact_affibody_code_seen_in_strict_training"):
            current.append("exact_Affibody_seen_in_strict_training")
        if getattr(row, "observed_in_any_raw_round"):
            current.append("observed_in_a_raw_selection_round")
        if getattr(row, "high_confidence_weak_negative"):
            current.append("flagged_as_high_confidence_weak_negative")
        if getattr(row, "qc_affibody_code_contains_cysteine"):
            current.append("designed_code_contains_cysteine")
        if getattr(row, "qc_full_affibody_contains_N_X_S_or_T_motif"):
            current.append("full_Affibody_contains_N_X_S_or_T_motif")
        flags.append(";".join(current) if current else "none")
    output["qc_attention_flags"] = flags
    return output


def build_affibody_reuse_table(candidates: pd.DataFrame) -> pd.DataFrame:
    sequence_column = (
        "provider_displayed_58aa_affibody_sequence"
        if "provider_displayed_58aa_affibody_sequence" in candidates.columns
        else "affibody_full_sequence"
    )
    _require(sequence_column in candidates.columns,
             "candidate QC lacks an Affibody model-sequence column")
    rows = []
    for code, block in candidates.groupby("affibody_design_code", sort=True):
        sequence_values = sorted(set(block[sequence_column].astype(str)))
        _require(len(sequence_values) == 1, "one Affibody code maps to several full sequences")
        target_rank = ";".join(
            "%s:%d" % (peptide, int(rank))
            for peptide, rank in sorted(zip(
                block["peptide_design_code"].astype(str),
                pd.to_numeric(block["wetlab_rank"]).astype(int),
            ))
        )
        rows.append({
            "affibody_design_code": str(code),
            "provider_displayed_58aa_affibody_sequence": sequence_values[0],
            "selected_pair_count": int(len(block)),
            "selected_for_n_peptide_targets": int(block["peptide_design_code"].nunique()),
            "selected_for_peptide_targets": ";".join(sorted(set(
                block["peptide_design_code"].astype(str)
            ))),
            "within_peptide_ranks": target_rank,
            "hamming_distance_to_FALTA": int(block[
                "qc_hamming_distance_to_FALTA"
            ].iloc[0]),
            "exact_code_seen_in_strict_training": bool(block[
                "qc_exact_affibody_code_seen_in_strict_training"
            ].iloc[0]),
            "nearest_strict_training_hamming_distance": int(block[
                "qc_nearest_strict_training_affibody_hamming_distance"
            ].iloc[0]),
            "nearest_strict_training_affibody_code": str(block[
                "qc_nearest_strict_training_affibody_code"
            ].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(
        ["selected_for_n_peptide_targets", "affibody_design_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_per_target_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for peptide, block in candidates.groupby("peptide_design_code", sort=True):
        rows.append({
            "peptide_design_code": str(peptide),
            "candidate_count": int(len(block)),
            "unique_affibody_count": int(block["affibody_design_code"].nunique()),
            "candidate_rows_using_cross_target_repeated_affibody": int(
                block["qc_affibody_repeated_across_targets"].sum()
            ),
            "minimum_hamming_distance_to_FALTA": int(
                block["qc_hamming_distance_to_FALTA"].min()
            ),
            "mean_hamming_distance_to_FALTA": float(
                block["qc_hamming_distance_to_FALTA"].mean()
            ),
            "exact_FALTA_count": int(block["qc_code_is_FALTA"].sum()),
            "exact_strict_training_affibody_seen_count": int(
                block["qc_exact_affibody_code_seen_in_strict_training"].sum()
            ),
            "exact_strict_training_affibody_unseen_count": int(
                (~block["qc_exact_affibody_code_seen_in_strict_training"]).sum()
            ),
            "minimum_nearest_training_hamming_distance": int(
                block["qc_nearest_strict_training_affibody_hamming_distance"].min()
            ),
            "mean_nearest_training_hamming_distance": float(
                block["qc_nearest_strict_training_affibody_hamming_distance"].mean()
            ),
            "observed_in_any_raw_round_count": int(block["observed_in_any_raw_round"].sum()),
            "observed_in_r009_or_r010_count": int(block["observed_in_r009_or_r010"].sum()),
            "high_confidence_weak_negative_count": int(
                block["high_confidence_weak_negative"].sum()
            ),
            "designed_code_cysteine_count": int(
                block["qc_affibody_code_contains_cysteine"].sum()
            ),
            "full_affibody_cysteine_count": int(
                block["qc_full_affibody_contains_cysteine"].sum()
            ),
            "N_X_S_or_T_motif_count": int(
                block["qc_full_affibody_contains_N_X_S_or_T_motif"].sum()
            ),
            "mean_candidate_list_membership_count": float(
                block["candidate_list_membership_count"].mean()
            ),
            "in_all_three_candidate_lists_count": int(
                block["candidate_list_membership_count"].eq(3).sum()
            ),
            "mean_component_top10_count": float(block["component_top10_count"].mean()),
            "top10_for_at_least_three_components_count": int(
                block["component_top10_count"].ge(3).sum()
            ),
            "mean_stab_seed_probability_sd": float(block["stab_seed_probability_sd"].mean()),
            "maximum_stab_seed_probability_sd": float(block["stab_seed_probability_sd"].max()),
            "mean_rde_seed_probability_sd": float(block["rde_seed_probability_sd"].mean()),
            "maximum_rde_seed_probability_sd": float(block["rde_seed_probability_sd"].max()),
        })
    return pd.DataFrame(rows)


def _value_counts_json(series: pd.Series) -> Dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def build_summary(
    candidates: pd.DataFrame,
    reuse: pd.DataFrame,
    strict_rows: int,
    strict_codes: int,
) -> Dict[str, object]:
    structure = {}
    for family in ("stab", "rde"):
        values = candidates["%s_seed_probability_sd" % family].to_numpy(float)
        structure[family] = {
            "mean_seed_probability_sd": float(np.mean(values)),
            "median_seed_probability_sd": float(np.median(values)),
            "p90_seed_probability_sd": float(np.quantile(values, 0.9)),
            "maximum_seed_probability_sd": float(np.max(values)),
        }
    repeated = reuse["selected_for_n_peptide_targets"].gt(1)
    return {
        "candidate_rows": int(len(candidates)),
        "peptide_targets": int(candidates["peptide_design_code"].nunique()),
        "unique_affibody_codes": int(len(reuse)),
        "affibody_codes_selected_for_multiple_targets": int(repeated.sum()),
        "candidate_rows_using_repeated_affibodies": int(
            candidates["qc_affibody_repeated_across_targets"].sum()
        ),
        "maximum_targets_using_one_affibody": int(
            reuse["selected_for_n_peptide_targets"].max()
        ),
        "FALTA": {
            "code": FALTA_CODE,
            "exact_selected_rows": int(candidates["qc_code_is_FALTA"].sum()),
            "distance_histogram": _value_counts_json(
                candidates["qc_hamming_distance_to_FALTA"]
            ),
        },
        "strict_training_membership": {
            "training_rows": int(strict_rows),
            "unique_affibody_codes": int(strict_codes),
            "exact_seen_selected_rows": int(
                candidates["qc_exact_affibody_code_seen_in_strict_training"].sum()
            ),
            "exact_unseen_selected_rows": int(
                (~candidates["qc_exact_affibody_code_seen_in_strict_training"]).sum()
            ),
            "nearest_hamming_distance_histogram": _value_counts_json(
                candidates["qc_nearest_strict_training_affibody_hamming_distance"]
            ),
        },
        "weak_selection_flags": {
            "observed_in_any_raw_round": int(candidates["observed_in_any_raw_round"].sum()),
            "observed_in_r009_or_r010": int(candidates["observed_in_r009_or_r010"].sum()),
            "high_confidence_weak_negative": int(
                candidates["high_confidence_weak_negative"].sum()
            ),
        },
        "sequence_flags": {
            "designed_code_contains_cysteine": int(
                candidates["qc_affibody_code_contains_cysteine"].sum()
            ),
            "full_affibody_contains_cysteine": int(
                candidates["qc_full_affibody_contains_cysteine"].sum()
            ),
            "full_affibody_contains_N_X_S_or_T_motif": int(
                candidates["qc_full_affibody_contains_N_X_S_or_T_motif"].sum()
            ),
        },
        "selector_overlap": {
            "candidate_list_membership_count_histogram": _value_counts_json(
                candidates["candidate_list_membership_count"]
            ),
            "component_top10_count_histogram": _value_counts_json(
                candidates["component_top10_count"]
            ),
        },
        "structure_seed_uncertainty": structure,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all4-shortlist", type=Path, required=True)
    parser.add_argument("--handoff-menu", type=Path, required=True)
    parser.add_argument("--weak-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output = args.output_dir.resolve()
    private = (REPO_ROOT / "private_data").resolve()
    _require(output != private and private in output.parents,
             "output must be a new directory below private_data")
    _require(not output.exists(), "output exists; refusing overwrite")

    all4_path = args.all4_shortlist.resolve()
    handoff_path = args.handoff_menu.resolve()
    weak_path = args.weak_labels.resolve()
    all4 = _read_outcome_free_csv(all4_path, "all4 shortlist")
    handoff = _read_outcome_free_csv(handoff_path, "handoff candidate menu")
    candidates = align_candidate_inputs(all4, handoff)
    strict, training_codes, training_pairs, training_code_rows = (
        load_strict_training_membership(weak_path)
    )
    candidates = annotate_candidates(
        candidates, training_codes, training_pairs, training_code_rows
    )
    reuse = build_affibody_reuse_table(candidates)
    per_target = build_per_target_table(candidates)
    summary = build_summary(candidates, reuse, len(strict), len(training_codes))

    staging = output.with_name(".%s.staging-%d" % (output.name, os.getpid()))
    _require(not staging.exists(), "staging output exists: %s" % staging)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging.mkdir(mode=0o700)
    paths = {
        "candidate_qc": staging / "candidate_qc.csv",
        "affibody_reuse_qc": staging / "affibody_reuse_qc.csv",
        "per_target_qc": staging / "per_target_qc.csv",
        "summary": staging / "candidate_qc_summary.json",
    }
    candidates.to_csv(paths["candidate_qc"], index=False)
    reuse.to_csv(paths["affibody_reuse_qc"], index=False)
    per_target.to_csv(paths["per_target_qc"], index=False)
    for name in ("candidate_qc", "affibody_reuse_qc", "per_target_qc"):
        os.chmod(paths[name], 0o600)
    with paths["summary"].open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(paths["summary"], 0o600)

    manifest = {
        "schema_version": "libb-wetlab-candidate-qc-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "inputs": {
            "all4_shortlist": {"path": str(all4_path), "sha256": sha256_file(all4_path)},
            "handoff_candidate_menu": {
                "path": str(handoff_path), "sha256": sha256_file(handoff_path)
            },
            "weak_labels": {"path": str(weak_path), "sha256": sha256_file(weak_path)},
        },
        "weak_labels_use": {
            "purpose": "strict-training Affibody and exact-pair membership only",
            "columns_read": list(WEAK_COLUMNS_READ),
            "strict_training_rule": (
                "LibB AND within declared alphabet AND strict identity-cold eligible AND "
                "(positive OR R001 negative count >= 3)"
            ),
            "safe_nonoutcome_metadata_note": (
                "strict_retention_identity_cold_eligible records identity overlap only; "
                "it is not a retention value or binder outcome"
            ),
        },
        "invariants": {
            "models_trained": False,
            "candidate_set_changed": False,
            "scores_changed": False,
            "within_peptide_ranks_changed": False,
            "retention_outcomes_read": False,
        },
        "outputs": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
        "summary": summary,
    }
    manifest_path = staging / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    os.rename(str(staging), str(output))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
