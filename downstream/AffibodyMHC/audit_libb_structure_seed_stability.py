#!/usr/bin/env python3
"""Audit exact top-candidate stability across LibB structure-readout seeds.

This program is deliberately post-selection and outcome-blind.  It reads the
validated exhaustive StaB and RDE score partitions and their already assembled
aggregate shortlists.  It does not rerank, filter, or otherwise change those
shortlists.  For each peptide, it independently asks which ten candidates each
of the five trained readout seeds would rank highest and measures agreement.

Direct-retention outcomes are neither an input nor allowed in any input table.
The output is therefore an uncertainty annotation of the frozen prospective
candidate lists, not another model-selection step.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TOP_K = 10
SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
EXPECTED_PEPTIDES = (
    "AF", "AH", "DP", "EA", "EL", "LL", "LV", "MW", "NF", "PH", "TL", "VV",
)
FAMILY_SPECS = {
    "stab": {
        "model": "stab_designed_ordered",
        "probability": "stab_probability",
        "seed_sd": "stab_seed_probability_sd",
    },
    "rde": {
        "model": "rde_network_designed_3fold_ensemble",
        "probability": "rde_probability",
        "seed_sd": "rde_seed_probability_sd",
    },
}
MERGED_SCHEMA = "libb-fixed-structure-candidate-scores-merged-v1"
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
REQUIRED_MERGE_VALIDATIONS = (
    "all_values_finite",
    "candidate_row_index_exact_once",
    "pair_uid_exact_once",
    "seed_aggregates_recomputed",
    "universe_identity_and_order_match",
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


def _canonical_rows_sha256(frame: pd.DataFrame, family: str) -> str:
    """Hash shortlist identity/order/rank without depending on CSV formatting."""
    digest = hashlib.sha256()
    for row_index, row in enumerate(frame.itertuples(index=False)):
        fields = (
            family,
            str(row_index),
            str(getattr(row, "pair_uid")),
            str(getattr(row, "peptide_design_code")),
            str(getattr(row, "affibody_design_code")),
            str(int(getattr(row, "wetlab_rank"))),
        )
        digest.update(("\t".join(fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def forbidden_outcome_columns(columns: Iterable[str]) -> List[str]:
    leaked = []
    for column in columns:
        normalized = str(column).strip().lower()
        if any(fragment in normalized for fragment in OUTCOME_FRAGMENTS):
            leaked.append(str(column))
    return sorted(set(leaked))


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    _require(path.is_file(), "%s is missing: %s" % (label, path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "%s is not a JSON object" % label)
    return payload


def _read_shortlist(path: Path, family: str) -> pd.DataFrame:
    _require(path.is_file(), "%s shortlist is missing: %s" % (family, path))
    header = pd.read_csv(path, nrows=0)
    leaked = forbidden_outcome_columns(header.columns)
    _require(not leaked, "%s shortlist contains forbidden outcomes: %s" % (family, leaked))
    required = {
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        "wetlab_rank",
        "ensemble_score",
    }
    _require(required.issubset(header.columns), "%s shortlist lacks columns: %s" % (
        family, sorted(required - set(header.columns))
    ))
    frame = pd.read_csv(
        path,
        usecols=list(required),
        dtype={
            "pair_uid": str,
            "peptide_design_code": str,
            "affibody_design_code": str,
        },
    )
    _require(len(frame) > 0, "%s shortlist is empty" % family)
    _require(not frame["pair_uid"].duplicated().any(),
             "%s shortlist contains duplicate pair_uid" % family)
    unknown = set(frame["peptide_design_code"]) - set(EXPECTED_PEPTIDES)
    _require(not unknown, "%s shortlist contains unknown peptides: %s" % (
        family, sorted(unknown)
    ))
    rank_numeric = pd.to_numeric(frame["wetlab_rank"], errors="raise").to_numpy(float)
    _require(bool(np.isfinite(rank_numeric).all()), "%s shortlist rank is non-finite" % family)
    _require(bool(np.equal(rank_numeric, np.rint(rank_numeric)).all()),
             "%s shortlist rank is non-integral" % family)
    frame["wetlab_rank"] = rank_numeric.astype(np.int64)
    scores = pd.to_numeric(frame["ensemble_score"], errors="raise").to_numpy(float)
    _require(bool(np.isfinite(scores).all()), "%s shortlist score is non-finite" % family)
    _require(bool(((scores >= 0.0) & (scores <= 1.0)).all()),
             "%s shortlist score is outside [0,1]" % family)
    frame["ensemble_score"] = scores
    for peptide, block in frame.groupby("peptide_design_code", sort=False):
        _require(len(block) <= TOP_K, "%s shortlist has more than %d rows for %s" % (
            family, TOP_K, peptide
        ))
        _require(block["wetlab_rank"].tolist() == list(range(1, len(block) + 1)),
                 "%s shortlist order/ranks changed for %s" % (family, peptide))
        _require(bool(np.all(np.diff(block["ensemble_score"].to_numpy(float)) <= 0.0)),
                 "%s shortlist score order disagrees with ranks for %s" % (family, peptide))
    return frame


def _stable_top_k_indices(
    probabilities: np.ndarray, candidate_row_index: np.ndarray, k: int,
) -> np.ndarray:
    """Top-k with an explicit ascending-row-index tie rule.

    An argpartition keeps this linear in the approximately 371,000 candidates
    per peptide.  Rows tied at the boundary are resolved by the immutable
    candidate-row index, and the final result is sorted by score then index.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    row_index = np.asarray(candidate_row_index, dtype=np.int64)
    _require(values.ndim == row_index.ndim == 1 and len(values) == len(row_index),
             "top-k inputs have incompatible shapes")
    _require(len(values) >= k > 0, "top-k requires at least k candidates")
    _require(bool(np.isfinite(values).all()), "top-k probabilities contain non-finite values")
    threshold = float(np.partition(values, len(values) - k)[len(values) - k])
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)
    needed = k - len(above)
    _require(0 <= needed <= len(tied), "top-k boundary calculation failed")
    tied_order = tied[np.argsort(row_index[tied], kind="mergesort")[:needed]]
    selected = np.concatenate((above, tied_order))
    order = np.lexsort((row_index[selected], -values[selected]))
    answer = selected[order]
    _require(len(answer) == k, "top-k did not return exactly k rows")
    return answer


def _score_columns(family: str) -> Tuple[List[str], str, str]:
    _require(family in FAMILY_SPECS, "family must be stab or rde")
    seed_columns = [
        "%s_probability_seed_%d" % (family, seed) for seed in SEEDS
    ]
    return (
        seed_columns,
        str(FAMILY_SPECS[family]["probability"]),
        str(FAMILY_SPECS[family]["seed_sd"]),
    )


def analyze_peptide(
    scores: pd.DataFrame,
    shortlist: pd.DataFrame,
    family: str,
    peptide: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Analyze one peptide without modifying the aggregate shortlist."""
    seed_columns, aggregate_column, seed_sd_column = _score_columns(family)
    required = {
        "candidate_row_index",
        "pair_uid",
        "peptide_design_code",
        "affibody_design_code",
        aggregate_column,
        seed_sd_column,
        *seed_columns,
    }
    leaked = forbidden_outcome_columns(scores.columns)
    _require(not leaked, "%s score partition contains forbidden outcomes: %s" % (
        family, leaked
    ))
    _require(required.issubset(scores.columns), "%s score partition lacks columns: %s" % (
        family, sorted(required - set(scores.columns))
    ))
    _require(len(scores) >= TOP_K, "%s/%s has fewer than %d candidates" % (
        family, peptide, TOP_K
    ))
    _require(scores["peptide_design_code"].astype(str).eq(peptide).all(),
             "%s partition contains another peptide" % peptide)
    _require(not scores["pair_uid"].astype(str).duplicated().any(),
             "%s/%s score partition repeats pair_uid" % (family, peptide))
    _require(not scores["affibody_design_code"].astype(str).duplicated().any(),
             "%s/%s score partition repeats an Affibody code" % (family, peptide))

    row_index_numeric = pd.to_numeric(
        scores["candidate_row_index"], errors="raise"
    ).to_numpy(float)
    _require(bool(np.isfinite(row_index_numeric).all()), "candidate_row_index is non-finite")
    _require(bool(np.equal(row_index_numeric, np.rint(row_index_numeric)).all()),
             "candidate_row_index is non-integral")
    row_index = row_index_numeric.astype(np.int64)
    _require(np.array_equal(row_index, np.arange(len(scores), dtype=np.int64)),
             "%s/%s candidate_row_index is not exactly 0..N-1" % (family, peptide))

    numeric_columns = seed_columns + [aggregate_column, seed_sd_column]
    numeric = scores[numeric_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(values).all()), "%s/%s contains a non-finite score" % (
        family, peptide
    ))
    seed_values = numeric[seed_columns].to_numpy(dtype=np.float64)
    aggregate_values = numeric[aggregate_column].to_numpy(dtype=np.float64)
    seed_sd_values = numeric[seed_sd_column].to_numpy(dtype=np.float64)
    _require(bool(((seed_values >= 0.0) & (seed_values <= 1.0)).all()),
             "%s/%s seed probability is outside [0,1]" % (family, peptide))
    _require(bool(((aggregate_values >= 0.0) & (aggregate_values <= 1.0)).all()),
             "%s/%s aggregate probability is outside [0,1]" % (family, peptide))
    _require(bool((seed_sd_values >= 0.0).all()),
             "%s/%s seed SD is negative" % (family, peptide))

    pair_uids = scores["pair_uid"].astype(str).to_numpy()
    affibody_codes = scores["affibody_design_code"].astype(str).to_numpy()
    uid_to_position = {uid: index for index, uid in enumerate(pair_uids)}
    seed_sets: Dict[int, set] = {}
    seed_top1_codes: Dict[int, str] = {}
    for column_index, seed in enumerate(SEEDS):
        top_indices = _stable_top_k_indices(
            seed_values[:, column_index], row_index, TOP_K
        )
        top_uids = set(pair_uids[top_indices].tolist())
        _require(len(top_uids) == TOP_K, "%s/%s seed top-10 repeats a pair" % (
            family, peptide
        ))
        seed_sets[seed] = top_uids
        seed_top1_codes[seed] = str(affibody_codes[top_indices[0]])

    pairwise_rows: List[Dict[str, Any]] = []
    for seed_a, seed_b in itertools.combinations(SEEDS, 2):
        left = seed_sets[seed_a]
        right = seed_sets[seed_b]
        intersection = len(left & right)
        union = len(left | right)
        pairwise_rows.append({
            "family": family,
            "peptide_design_code": peptide,
            "seed_a": int(seed_a),
            "seed_b": int(seed_b),
            "top_k": TOP_K,
            "intersection_count": int(intersection),
            "union_count": int(union),
            "jaccard": float(intersection / union),
        })

    current = shortlist.loc[
        shortlist["peptide_design_code"].astype(str).eq(peptide)
    ].copy()
    membership_rows = []
    for shortlist_row_index, row in current.iterrows():
        uid = str(row["pair_uid"])
        _require(uid in uid_to_position, "%s/%s shortlist pair is absent from scores: %s" % (
            family, peptide, uid
        ))
        position = uid_to_position[uid]
        _require(str(row["affibody_design_code"]) == str(affibody_codes[position]),
                 "%s/%s shortlist Affibody mapping changed for %s" % (
                     family, peptide, uid
                 ))
        aggregate_probability = float(aggregate_values[position])
        _require(np.isclose(
            float(row["ensemble_score"]), aggregate_probability, rtol=0.0, atol=5e-7,
        ), "%s/%s shortlist score no longer matches merged scores for %s" % (
            family, peptide, uid
        ))
        in_seed = {seed: uid in seed_sets[seed] for seed in SEEDS}
        record: Dict[str, Any] = {
            "family": family,
            "source_shortlist_row_index": int(shortlist_row_index),
            "pair_uid": uid,
            "peptide_design_code": peptide,
            "affibody_design_code": str(row["affibody_design_code"]),
            "wetlab_rank": int(row["wetlab_rank"]),
            "aggregate_score": float(row["ensemble_score"]),
            "seed_probability_sd": float(seed_sd_values[position]),
            "seed_top10_count": int(sum(in_seed.values())),
            "in_any_seed_top10": bool(any(in_seed.values())),
            "in_all_five_seed_top10": bool(all(in_seed.values())),
        }
        for seed in SEEDS:
            record["in_seed_%d_top10" % seed] = bool(in_seed[seed])
        membership_rows.append(record)
    membership = pd.DataFrame(membership_rows)
    if len(current) == 0:
        membership = pd.DataFrame(columns=[
            "family", "source_shortlist_row_index", "pair_uid",
            "peptide_design_code", "affibody_design_code", "wetlab_rank",
            "aggregate_score", "seed_probability_sd", "seed_top10_count",
            "in_any_seed_top10", "in_all_five_seed_top10",
            *("in_seed_%d_top10" % seed for seed in SEEDS),
        ])
    else:
        _require(membership["pair_uid"].tolist() == current["pair_uid"].astype(str).tolist(),
                 "%s/%s membership annotation reordered shortlist pairs" % (family, peptide))
        _require(membership["wetlab_rank"].tolist() == current["wetlab_rank"].tolist(),
                 "%s/%s membership annotation changed ranks" % (family, peptide))

    jaccards = np.asarray([row["jaccard"] for row in pairwise_rows], dtype=float)
    top1_values = [seed_top1_codes[seed] for seed in SEEDS]
    per_peptide = {
        "family": family,
        "peptide_design_code": peptide,
        "universe_rows": int(len(scores)),
        "seed_count": int(len(SEEDS)),
        "top_k": TOP_K,
        "seed_top1_codes": ";".join(top1_values),
        "distinct_seed_top1_codes": int(len(set(top1_values))),
        "all_five_top1_agree": bool(len(set(top1_values)) == 1),
        "mean_pairwise_top10_jaccard": float(np.mean(jaccards)),
        "median_pairwise_top10_jaccard": float(np.median(jaccards)),
        "minimum_pairwise_top10_jaccard": float(np.min(jaccards)),
        "maximum_pairwise_top10_jaccard": float(np.max(jaccards)),
        "aggregate_shortlist_rows": int(len(membership)),
        "aggregate_shortlist_in_any_seed_top10": int(
            membership["in_any_seed_top10"].sum() if len(membership) else 0
        ),
        "aggregate_shortlist_in_all_five_seed_top10": int(
            membership["in_all_five_seed_top10"].sum() if len(membership) else 0
        ),
    }
    per_peptide["fraction_aggregate_shortlist_in_any_seed_top10"] = (
        float(per_peptide["aggregate_shortlist_in_any_seed_top10"] / len(membership))
        if len(membership) else None
    )
    per_peptide["fraction_aggregate_shortlist_in_all_five_seed_top10"] = (
        float(per_peptide["aggregate_shortlist_in_all_five_seed_top10"] / len(membership))
        if len(membership) else None
    )
    return membership, per_peptide, pairwise_rows


def _validate_merged_manifest(
    score_dir: Path, family: str, verify_hashes: bool,
) -> Tuple[Mapping[str, Any], Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    manifest_path = score_dir / "manifest.json"
    manifest = _load_json(manifest_path, "%s merged-score manifest" % family)
    _require(manifest.get("schema_version") == MERGED_SCHEMA,
             "%s merged-score schema changed" % family)
    _require(manifest.get("family") == family, "%s merged manifest has wrong family" % family)
    _require(manifest.get("model") == FAMILY_SPECS[family]["model"],
             "%s merged manifest has wrong model" % family)
    _require(tuple(manifest.get("readout_seeds", ())) == SEEDS,
             "%s readout seed set or order changed" % family)
    validation = manifest.get("validation", {})
    for key in REQUIRED_MERGE_VALIDATIONS:
        _require(validation.get(key) is True,
                 "%s merged manifest did not pass validation %s" % (family, key))
    outputs = manifest.get("outputs", {})
    records = outputs.get("partitions", []) if isinstance(outputs, dict) else []
    _require(isinstance(records, list), "%s manifest partition records are invalid" % family)
    by_peptide = {}
    input_partitions = []
    total_rows = 0
    for record in records:
        _require(isinstance(record, dict), "%s has a non-object partition record" % family)
        peptide = str(record.get("peptide_design_code", ""))
        _require(peptide and peptide not in by_peptide,
                 "%s manifest repeats peptide %s" % (family, peptide))
        filename = str(record.get("output_file", ""))
        _require(Path(filename).name == filename and filename,
                 "%s/%s output filename is unsafe" % (family, peptide))
        path = score_dir / filename
        _require(path.is_file(), "%s/%s score partition is missing" % (family, peptide))
        expected_bytes = int(record.get("output_bytes", -1))
        expected_hash = str(record.get("output_sha256", ""))
        _require(path.stat().st_size == expected_bytes,
                 "%s/%s partition byte count changed" % (family, peptide))
        observed_hash = sha256_file(path) if verify_hashes else expected_hash
        _require(not verify_hashes or observed_hash == expected_hash,
                 "%s/%s partition hash changed" % (family, peptide))
        candidate_rows = int(record.get("candidate_rows", -1))
        _require(candidate_rows >= TOP_K, "%s/%s manifest row count is invalid" % (
            family, peptide
        ))
        total_rows += candidate_rows
        enriched = dict(record)
        enriched["resolved_path"] = str(path.resolve())
        by_peptide[peptide] = enriched
        input_partitions.append({
            "peptide_design_code": peptide,
            "path": str(path.resolve()),
            "sha256": observed_hash,
            "bytes": expected_bytes,
            "rows": candidate_rows,
        })
    _require(tuple(sorted(by_peptide)) == tuple(sorted(EXPECTED_PEPTIDES)),
             "%s merged manifest peptide set changed: %s" % (
                 family, sorted(by_peptide)
             ))
    _require(int(outputs.get("candidate_rows", -1)) == total_rows,
             "%s merged manifest total row count is inconsistent" % family)
    source = {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "partitions": input_partitions,
        "candidate_rows": int(total_rows),
    }
    return manifest, by_peptide, source


def analyze_family(
    score_dir: Path,
    shortlist_path: Path,
    family: str,
    verify_hashes: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Validate one family and return outcome-blind stability annotations."""
    score_dir = score_dir.resolve()
    shortlist_path = shortlist_path.resolve()
    _require(score_dir.is_dir(), "%s merged-score directory is missing" % family)
    _, partition_records, score_source = _validate_merged_manifest(
        score_dir, family, verify_hashes
    )
    shortlist = _read_shortlist(shortlist_path, family)
    shortlist_source = {
        "path": str(shortlist_path),
        "sha256": sha256_file(shortlist_path),
        "rows": int(len(shortlist)),
        "identity_order_rank_sha256": _canonical_rows_sha256(shortlist, family),
    }

    memberships = []
    per_peptide_rows = []
    pairwise_rows = []
    for peptide in EXPECTED_PEPTIDES:
        record = partition_records[peptide]
        path = Path(str(record["resolved_path"]))
        scores = pd.read_parquet(path)
        _require(len(scores) == int(record["candidate_rows"]),
                 "%s/%s partition row count changed" % (family, peptide))
        membership, per_peptide, current_pairwise = analyze_peptide(
            scores, shortlist, family, peptide
        )
        if len(membership):
            memberships.append(membership)
        per_peptide_rows.append(per_peptide)
        pairwise_rows.extend(current_pairwise)

    membership_table = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame()
    per_peptide_table = pd.DataFrame(per_peptide_rows)
    pairwise_table = pd.DataFrame(pairwise_rows)
    _require(len(pairwise_table) == len(EXPECTED_PEPTIDES) * 10,
             "%s pairwise comparison count changed" % family)
    _require(membership_table["pair_uid"].tolist() == shortlist["pair_uid"].astype(str).tolist(),
             "%s membership output changed aggregate shortlist order" % family)
    _require(membership_table["wetlab_rank"].astype(int).tolist()
             == shortlist["wetlab_rank"].astype(int).tolist(),
             "%s membership output changed aggregate shortlist ranks" % family)

    jaccards = pairwise_table["jaccard"].to_numpy(float)
    seed_sd = membership_table["seed_probability_sd"].to_numpy(float)
    summary = {
        "candidate_universe_rows": int(per_peptide_table["universe_rows"].sum()),
        "peptide_targets": int(len(per_peptide_table)),
        "top_k_per_seed": TOP_K,
        "readout_seeds": list(SEEDS),
        "pairwise_seed_comparisons": int(len(pairwise_table)),
        "mean_pairwise_top10_jaccard": float(np.mean(jaccards)),
        "median_pairwise_top10_jaccard": float(np.median(jaccards)),
        "minimum_pairwise_top10_jaccard": float(np.min(jaccards)),
        "maximum_pairwise_top10_jaccard": float(np.max(jaccards)),
        "peptides_with_all_five_top1_agreement": int(
            per_peptide_table["all_five_top1_agree"].sum()
        ),
        "aggregate_shortlist_rows": int(len(membership_table)),
        "aggregate_shortlist_rows_in_any_seed_top10": int(
            membership_table["in_any_seed_top10"].sum()
        ),
        "aggregate_shortlist_rows_in_all_five_seed_top10": int(
            membership_table["in_all_five_seed_top10"].sum()
        ),
        "targets_without_aggregate_shortlist_candidates": sorted(
            per_peptide_table.loc[
                per_peptide_table["aggregate_shortlist_rows"].eq(0),
                "peptide_design_code",
            ].astype(str).tolist()
        ),
        "aggregate_shortlist_seed_probability_sd": {
            "mean": float(np.mean(seed_sd)),
            "median": float(np.median(seed_sd)),
            "maximum": float(np.max(seed_sd)),
        },
    }
    source = {"merged_scores": score_source, "aggregate_shortlist": shortlist_source}
    return membership_table, per_peptide_table, pairwise_table, summary, source


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stab-score-dir", type=Path, required=True)
    parser.add_argument("--rde-score-dir", type=Path, required=True)
    parser.add_argument("--stab-shortlist", type=Path, required=True)
    parser.add_argument("--rde-shortlist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-partition-hashes",
        action="store_true",
        help="For tests only; production runs should verify every Parquet hash.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    final_output = args.output_dir.resolve()
    private_root = (REPO_ROOT / "private_data").resolve()
    _require(private_root.is_dir(), "repository private_data directory is missing")
    _require(final_output != private_root and private_root in final_output.parents,
             "output must be a new directory below repository private_data")
    _require(not final_output.exists(), "output exists; refusing overwrite")

    family_inputs = {
        "stab": (args.stab_score_dir, args.stab_shortlist),
        "rde": (args.rde_score_dir, args.rde_shortlist),
    }
    all_membership = []
    all_per_peptide = []
    all_pairwise = []
    summaries = {}
    sources = {}
    for family in ("stab", "rde"):
        score_dir, shortlist_path = family_inputs[family]
        membership, per_peptide, pairwise, summary, source = analyze_family(
            score_dir,
            shortlist_path,
            family,
            verify_hashes=not args.skip_partition_hashes,
        )
        all_membership.append(membership)
        all_per_peptide.append(per_peptide)
        all_pairwise.append(pairwise)
        summaries[family] = summary
        sources[family] = source

    membership_table = pd.concat(all_membership, ignore_index=True)
    per_peptide_table = pd.concat(all_per_peptide, ignore_index=True)
    pairwise_table = pd.concat(all_pairwise, ignore_index=True)
    overall_summary = {
        "schema_version": "libb-structure-seed-stability-summary-v1",
        "definition": {
            "top10": (
                "For each frozen readout seed and peptide, take the ten highest seed "
                "probabilities over the unchanged exhaustive candidate universe. "
                "Exact score ties prefer smaller immutable candidate_row_index."
            ),
            "pairwise_top10_jaccard": (
                "size of intersection divided by size of union for two seed-specific "
                "top-10 pair sets"
            ),
            "all_five_top1_agree": (
                "true only when all five trained readout seeds choose the same exact "
                "peptide-Affibody pair at rank one"
            ),
            "aggregate_shortlist_membership": (
                "whether each already selected aggregate-shortlist pair appears in "
                "at least one or all five seed-specific top-10 sets"
            ),
        },
        "families": summaries,
        "combined": {
            "families": ["stab", "rde"],
            "peptide_family_rows": int(len(per_peptide_table)),
            "pairwise_seed_comparisons": int(len(pairwise_table)),
            "mean_pairwise_top10_jaccard": float(pairwise_table["jaccard"].mean()),
            "peptides_with_all_five_top1_agreement": int(
                per_peptide_table["all_five_top1_agree"].sum()
            ),
            "aggregate_shortlist_rows": int(len(membership_table)),
            "aggregate_shortlist_rows_in_any_seed_top10": int(
                membership_table["in_any_seed_top10"].sum()
            ),
            "aggregate_shortlist_rows_in_all_five_seed_top10": int(
                membership_table["in_all_five_seed_top10"].sum()
            ),
        },
        "interpretation_guard": (
            "This audit measures seed-level ranking stability. It did not read direct "
            "retention outcomes, change aggregate scores, or select new candidates."
        ),
    }

    final_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(
        prefix=".%s.staging-" % final_output.name, dir=str(final_output.parent)
    ))
    try:
        output_paths = {
            "aggregate_shortlist_seed_membership": staging / "aggregate_shortlist_seed_membership.csv",
            "per_peptide_seed_stability": staging / "per_peptide_seed_stability.csv",
            "pairwise_seed_top10_jaccard": staging / "pairwise_seed_top10_jaccard.csv",
            "summary": staging / "structure_seed_stability_summary.json",
        }
        membership_table.to_csv(
            output_paths["aggregate_shortlist_seed_membership"], index=False
        )
        per_peptide_table.to_csv(
            output_paths["per_peptide_seed_stability"], index=False
        )
        pairwise_table.to_csv(
            output_paths["pairwise_seed_top10_jaccard"], index=False
        )
        with output_paths["summary"].open("x", encoding="utf-8") as handle:
            json.dump(overall_summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        for path in output_paths.values():
            os.chmod(str(path), 0o600)

        script_path = Path(__file__).resolve()
        manifest = {
            "schema_version": "libb-structure-seed-stability-audit-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - started, 6),
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "parameters": {
                "families": ["stab", "rde"],
                "readout_seeds": list(SEEDS),
                "top_k": TOP_K,
                "tie_break": "ascending immutable candidate_row_index",
                "partition_hashes_verified": bool(not args.skip_partition_hashes),
            },
            "inputs": sources,
            "outputs": {
                name: {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                    "rows": (
                        int(len(membership_table)) if name == "aggregate_shortlist_seed_membership"
                        else int(len(per_peptide_table)) if name == "per_peptide_seed_stability"
                        else int(len(pairwise_table)) if name == "pairwise_seed_top10_jaccard"
                        else None
                    ),
                }
                for name, path in output_paths.items()
            },
            "invariants": {
                "retention_outcomes_read": False,
                "models_trained": False,
                "aggregate_scores_changed": False,
                "aggregate_shortlist_candidates_changed": False,
                "aggregate_shortlist_order_changed": False,
                "aggregate_shortlist_ranks_changed": False,
            },
            "summary": summaries,
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(str(manifest_path), 0o600)
        os.rename(str(staging), str(final_output))
    except BaseException:
        if staging.is_dir():
            shutil.rmtree(str(staging))
        raise
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "summary": manifest["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
