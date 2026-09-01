#!/usr/bin/env python
"""Build the frozen-MINT feature delta for the LibA later-round ablation.

The existing ``mint_weak_cache_v1`` artifact is immutable.  This program
reconstructs the four prespecified LibA label arms from raw R001--R014 data,
validates the independently materialized CPU label artifact, and resolves each
required sequence pair by SHA-256 against the old cache.  Only pairs absent
from the old cache are embedded.  A small deterministic set of old-cache rows
is embedded again as sentinels; merge refuses publication unless those new
features reproduce the old representation within a tight numerical tolerance.

The three restartable stages are:

``prepare``
    Recompute and validate labels, write the natural and three size-matched
    memberships, write a feature-resolution index, and materialize sequences
    only for cache misses plus overlap sentinels.
``extract-shard``
    Run the same frozen MINT chain-mean representation as the old cache for one
    modulo shard.  This stage requires CUDA and validates the old model contract.
``merge``
    Validate all shards, compare overlap sentinels to the immutable old cache,
    and publish the small delta NPZ plus the sentinel comparison.

All row-level artifacts must live below the Git-ignored ``private_data`` tree.
Biological codes are always read with ``keep_default_na=False`` so ``NA`` stays
an amino-acid code rather than becoming a missing value.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import build_mint_weak_feature_cache as old_builder
from downstream.AffibodyMHC.build_selection_weak_labels import (
    AFFIBODY_ALPHABET,
    PEPTIDE_ALPHABET,
    discover_raw_round_files,
    key_series,
    load_count_round,
    load_presence,
)
from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    OMITTED_LINKER,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (
    opaque_id,
    sha256_file,
)
from downstream.AffibodyMHC.extract_mint_features import PairCollator
from mint.helpers.extract import MINTWrapper, load_config


SCHEMA_VERSION = "mint-liba-late-feature-delta-v1"
LIBRARY = "LibA"
FEATURE_NAME = old_builder.FEATURE_NAME
FEATURE_DIMENSION = old_builder.FEATURE_DIMENSION
ARM_ORDER = (
    "A",
    "B",
    "C",
    "D",
)
SIZE_MATCH_SEEDS = (20260811, 20260812, 20260813)
TOP_FRACTION = 0.02
MIN_NEGATIVE_R001_COUNT = 3
POSITIVE_ROUNDS = tuple(range(9, 15))
NEGATIVE_EXCLUSION_ROUNDS = tuple(range(2, 15))
DEFAULT_SENTINEL_COUNT = 16
EXPECTED_BASELINE_POSITIVES = 11320
EXPECTED_BASELINE_NEGATIVES = 11222
EXPECTED_POSITIVES_BY_ARM = {
    "A": 11320,
    "B": 24925,
    "C": 23189,
    "D": 19596,
}

ROWS_FILENAME = "cache_rows.csv"
REUSE_FILENAME = "reuse_index.csv"
MEMBERSHIP_FILENAME = "label_membership_long.csv"
SUMMARY_FILENAME = "membership_summary.csv"
PREPARE_MANIFEST_FILENAME = "manifest.json"
FINAL_CACHE_FILENAME = "mint_chain_mean_delta.npz"
FINAL_MANIFEST_FILENAME = "manifest.json"
SENTINEL_COMPARISON_FILENAME = "overlap_sentinel_comparison.csv"

ARM_FORMULAS = {
    "A": {
        "name": "r009_r010_raw_count",
        "rounds": [9, 10],
        "universe": "outer union of pairs present in the named rounds",
        "missing_round_value": 0,
        "score": "sum of raw read counts across the named rounds",
    },
    "B": {
        "name": "r009_r014_raw_count",
        "rounds": [9, 10, 11, 12, 13, 14],
        "universe": "outer union of pairs present in the named rounds",
        "missing_round_value": 0,
        "score": "sum of raw read counts across the named rounds",
    },
    "C": {
        "name": "r009_r014_equal_round_frequency",
        "rounds": [9, 10, 11, 12, 13, 14],
        "universe": "outer union of pairs present in the named rounds",
        "missing_round_value": 0,
        "score": "arithmetic mean of count divided by total round depth across the named rounds, with absent pairs zero-filled; this equals the mean provider frequency",
    },
    "D": {
        "name": "r011_r014_raw_count",
        "rounds": [11, 12, 13, 14],
        "universe": "outer union of pairs present in the named rounds",
        "missing_round_value": 0,
        "score": "sum of raw read counts across the named rounds",
    },
}
LABEL_CONTRACT = {
    "library": LIBRARY,
    "positive_rule": {
        "top_fraction": TOP_FRACTION,
        "rank": "ceil(top_fraction * arm-universe row count), at least one",
        "ties": "inclusive: score >= the score at the requested rank",
        "cutoff_timing": "before retention-pair and identity-cold filtering",
        "arms": ARM_FORMULAS,
    },
    "negative_rule": {
        "definition": "present in raw R001 with count >= 3 and absent from every raw R002--R014 file",
        "shared_across_arms": True,
    },
    "training_filters": {
        "declared_library_alphabet": True,
        "exact_retention_pairs_removed": True,
        "identity_cold": "within-LibA full reconstructed chain identity for both partners; LibB-only identities are not excluded",
    },
    "size_match": {
        "positive_target": "natural arm A positive count after all filters",
        "negative_rows": "all fixed eligible negatives in every scenario",
        "seeds": list(SIZE_MATCH_SEEDS),
        "ordering": "ascending SHA256('size-match|<seed>|<pair_uid>')",
    },
}

LABEL_REQUIRED_COLUMNS = {
    "arm",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "pep",
    "aff",
    "r001_count",
    "weak_label",
    "label_score",
}
SIZE_REQUIRED_COLUMNS = {"arm", "sampling_seed", "pair_uid"}
RETENTION_REQUIRED_COLUMNS = {
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "measurement_missing",
    "target_retention",
    "target_binder",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
}

ROW_COLUMNS = (
    "row_index",
    "delta_cache_uid",
    "cache_role",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "old_cache_row_index",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)
SEQUENCE_COLUMNS = (
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
)
NPZ_METADATA_COLUMNS = tuple(
    column for column in ROW_COLUMNS if column not in SEQUENCE_COLUMNS
)
REUSE_COLUMNS = (
    "resolver_index",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
    "feature_source",
    "old_cache_row_index",
    "delta_row_index",
    "is_overlap_sentinel",
    "sentinel_delta_row_index",
)
MEMBERSHIP_COLUMNS = (
    "membership_index",
    "membership_uid",
    "arm",
    "sampling",
    "subset_seed",
    "sampling_rank",
    "sampling_sha256",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "sequence_pair_sha256",
    "weak_label",
    "label_score",
    "r001_count",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_string_csv(path):
    return pd.read_csv(
        str(path), dtype=str, keep_default_na=False, na_filter=False
    )


def read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def canonical_json_sha256(payload):
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _ensure_private_directory(path, must_be_new):
    return old_builder._ensure_private_directory(path, must_be_new)


def _atomic_write_csv(path, frame):
    return old_builder._atomic_write_csv(path, frame)


def _atomic_write_json(path, payload):
    return old_builder._atomic_write_json(path, payload)


def _atomic_write_npz(path, arrays, compressed=True):
    return old_builder._atomic_write_npz(path, arrays, compressed=compressed)


def _manifest_output_hash(manifest, filename):
    outputs = manifest.get("outputs", {})
    value = outputs.get(filename)
    if isinstance(value, dict):
        value = value.get("sha256")
    _require(isinstance(value, str) and len(value) == 64,
             "label manifest lacks output hash for {}".format(filename))
    return value


def validate_retention_lineage(retention_path, manifest_path, sequence_zip):
    manifest = read_json(manifest_path)
    retention_hash = sha256_file(retention_path)
    zip_hash = sha256_file(sequence_zip)
    _require(
        manifest.get("output", {}).get("sha256") == retention_hash,
        "retention manifest/output hash mismatch",
    )
    _require(
        manifest.get("sources", {}).get("sequence_zip", {}).get("sha256")
        == zip_hash,
        "retention manifest uses a different sequence ZIP",
    )
    return manifest, retention_hash, zip_hash


def validate_old_model_manifest(old_cache_path, old_manifest_path):
    """Validate the immutable cache and return lightweight metadata arrays."""
    manifest = read_json(old_manifest_path)
    _require(
        manifest.get("schema_version") == old_builder.SCHEMA_VERSION,
        "old cache schema mismatch",
    )
    _require(manifest.get("stage") == "merge", "old cache is not a merged cache")
    old_cache_hash = sha256_file(old_cache_path)
    _require(
        manifest.get("output", {}).get("sha256") == old_cache_hash,
        "old cache artifact hash mismatch",
    )
    _require(
        manifest.get("features", {}).get("name") == FEATURE_NAME
        and int(manifest.get("features", {}).get("shape", [0, -1])[1])
        == FEATURE_DIMENSION,
        "old cache feature contract mismatch",
    )
    model = manifest.get("model", {})
    required_model = {
        "use_multimer": True,
        "sep_chains": True,
        "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
        "experimental_construct_linker_omitted": OMITTED_LINKER,
        "layer": 33,
        "pooling": "separate residue mean for chain 1 and chain 2, concatenated",
        "feature_name": FEATURE_NAME,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
    }
    for field, expected in required_model.items():
        _require(model.get(field) == expected,
                 "old cache model field mismatch: {}".format(field))
    contract = manifest.get("contract", {})
    payload = contract.get("payload")
    _require(isinstance(payload, dict), "old cache lacks contract payload")
    _require(
        canonical_json_sha256(payload) == contract.get("sha256"),
        "old cache contract payload/hash mismatch",
    )
    for field in (
        "checkpoint_sha256",
        "config_sha256",
        "mint_source_tree_sha256",
        "pair_collator_source_sha256",
    ):
        if field == "checkpoint_sha256":
            observed = model.get("checkpoint", {}).get("sha256")
        elif field == "config_sha256":
            observed = model.get("config", {}).get("sha256")
        elif field == "pair_collator_source_sha256":
            observed = model.get("pair_collator_source", {}).get("sha256")
        else:
            observed = model.get(field)
        _require(payload.get(field) == observed,
                 "old model/contract mismatch for {}".format(field))
    required_keys = {
        "row_index",
        "library",
        "source_kind",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_design_code",
        "affibody_design_code",
        "weak_label",
        "upstream_library_local_strict_retention_identity_cold_eligible",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
        FEATURE_NAME,
    }
    with np.load(str(old_cache_path), allow_pickle=False) as archive:
        _require(required_keys.issubset(set(archive.files)),
                 "old cache NPZ schema mismatch")
        metadata = {
            key: np.asarray(archive[key]).copy()
            for key in required_keys
            if key != FEATURE_NAME
        }
    n_rows = len(metadata["row_index"])
    _require(
        np.array_equal(
            metadata["row_index"].astype(np.int64),
            np.arange(n_rows, dtype=np.int64),
        ),
        "old cache row indices are not contiguous",
    )
    for key, values in metadata.items():
        _require(len(values) == n_rows,
                 "old cache metadata length mismatch in {}".format(key))
    sequence_hashes = metadata["sequence_pair_sha256"].astype(str)
    _require(len(set(sequence_hashes.tolist())) == n_rows,
             "old cache sequence hashes are not unique")
    return manifest, old_cache_hash, metadata


def validate_model_files_against_old(old_manifest, checkpoint, config):
    model = old_manifest["model"]
    _require(sha256_file(checkpoint) == model["checkpoint"]["sha256"],
             "checkpoint differs from old cache")
    _require(sha256_file(config) == model["config"]["sha256"],
             "config differs from old cache")
    current_mint_hash = old_builder.sha256_source_tree(REPO_ROOT / "mint")
    _require(current_mint_hash == model["mint_source_tree_sha256"],
             "MINT source tree differs from old cache")
    collator_path = REPO_ROOT / "downstream" / "AffibodyMHC" / "extract_mint_features.py"
    _require(
        sha256_file(collator_path)
        == model["pair_collator_source"]["sha256"],
        "pair collator differs from old cache",
    )
    return current_mint_hash, collator_path


def _round_column(round_index, quantity):
    return "r{:03d}_{}".format(int(round_index), quantity)


def build_positive_universe(round_frames):
    """Outer-join LibA R009--R014 once, zero-filling missing observations."""
    _require(set(round_frames) == set(POSITIVE_ROUNDS),
             "positive round mapping must contain R009--R014")
    merged = None
    for round_index in POSITIVE_ROUNDS:
        frame = round_frames[round_index]
        required = {"pep", "aff", "count", "frequency"}
        _require(required.issubset(frame.columns),
                 "round {} schema mismatch".format(round_index))
        block = frame[["pep", "aff", "count", "frequency"]].copy()
        block = block.rename(
            columns={
                "count": _round_column(round_index, "count"),
                "frequency": _round_column(round_index, "frequency"),
            }
        )
        if merged is None:
            merged = block
        else:
            merged = merged.merge(
                block,
                on=["pep", "aff"],
                how="outer",
                sort=False,
                validate="one_to_one",
            )
    for round_index in POSITIVE_ROUNDS:
        count_column = _round_column(round_index, "count")
        frequency_column = _round_column(round_index, "frequency")
        merged[count_column] = (
            pd.to_numeric(merged[count_column], errors="raise")
            .fillna(0)
            .astype(np.int64)
        )
        merged[frequency_column] = (
            pd.to_numeric(merged[frequency_column], errors="raise")
            .fillna(0.0)
            .astype(np.float64)
        )
    _require(not bool(merged[["pep", "aff"]].duplicated().any()),
             "positive universe contains duplicate pairs")
    return merged


def inclusive_top_fraction(frame, score, top_fraction=TOP_FRACTION):
    """Return tie-inclusive top-fraction rows and the prespecified cutoff."""
    _require(len(frame) > 0, "cannot rank an empty arm universe")
    _require(0.0 < float(top_fraction) < 1.0,
             "top fraction must lie in (0,1)")
    values = np.asarray(score)
    _require(values.shape == (len(frame),), "score length mismatch")
    _require(bool(np.isfinite(values).all()), "arm score is non-finite")
    _require(bool((values > 0).all()), "arm-universe scores must be positive")
    requested_rank = max(1, int(math.ceil(float(top_fraction) * len(frame))))
    cutoff = np.partition(values, len(values) - requested_rank)[
        len(values) - requested_rank
    ]
    selected = frame.loc[values >= cutoff].copy()
    return selected, cutoff.item(), requested_rank


def derive_raw_positive_arms(round_frames, top_fraction=TOP_FRACTION):
    """Reproduce the four positive arms before biological holdout filters."""
    universe = build_positive_universe(round_frames)
    round_depths = {
        index: int(universe[_round_column(index, "count")].sum())
        for index in POSITIVE_ROUNDS
    }
    _require(all(value > 0 for value in round_depths.values()),
             "a positive-selection round has zero total depth")
    outputs = {}
    summaries = {}
    for arm in ARM_ORDER:
        definition = ARM_FORMULAS[arm]
        rounds = tuple(definition["rounds"])
        present = np.zeros(len(universe), dtype=bool)
        for round_index in rounds:
            present |= universe[_round_column(round_index, "count")].to_numpy() > 0
        arm_universe = universe.loc[present].copy().reset_index(drop=True)
        if arm == "C":
            score = np.mean(
                np.column_stack(
                    [
                        arm_universe[_round_column(index, "count")].to_numpy(
                            dtype=np.float64
                        )
                        / float(round_depths[index])
                        for index in rounds
                    ]
                ),
                axis=1,
            )
        else:
            score = np.sum(
                np.column_stack(
                    [
                        arm_universe[_round_column(index, "count")].to_numpy(
                            dtype=np.int64
                        )
                        for index in rounds
                    ]
                ),
                axis=1,
                dtype=np.int64,
            )
        selected, cutoff, requested_rank = inclusive_top_fraction(
            arm_universe, score, top_fraction=top_fraction
        )
        if arm == "C":
            selected["label_score"] = np.mean(
                np.column_stack(
                    [
                        selected[_round_column(index, "count")].to_numpy(
                            dtype=np.float64
                        )
                        / float(round_depths[index])
                        for index in rounds
                    ]
                ),
                axis=1,
            )
        else:
            selected["label_score"] = np.sum(
                np.column_stack(
                    [
                        selected[_round_column(index, "count")].to_numpy(
                            dtype=np.int64
                        )
                        for index in rounds
                    ]
                ),
                axis=1,
                dtype=np.int64,
            )
        selected.insert(0, "arm", arm)
        selected = selected.sort_values(
            ["label_score", "pep", "aff"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        outputs[arm] = selected
        summaries[arm] = {
            "arm": arm,
            "universe_rows": int(len(arm_universe)),
            "requested_rank": int(requested_rank),
            "inclusive_cutoff": (
                int(cutoff)
                if arm != "C"
                else format(float(cutoff), ".17g")
            ),
            "positive_rows_before_filters": int(len(selected)),
        }
    return outputs, summaries


def deterministic_size_hash(seed, pair_uid):
    return sha256_text("size-match|{}|{}".format(int(seed), str(pair_uid)))


def _within_liba_alphabet(peptide, affibody):
    return set(peptide).issubset(PEPTIDE_ALPHABET) and set(affibody).issubset(
        AFFIBODY_ALPHABET[LIBRARY]
    )


def _sequence_record(template, peptide, affibody):
    chain1, chain2 = old_builder.reconstruct_chains(
        template, LIBRARY, str(peptide), str(affibody)
    )
    return {
        "chain1_smart_hla_linker_peptide_sequence": chain1,
        "chain2_affibody_sequence": chain2,
        "chain1_sha256": sha256_text(chain1),
        "chain2_sha256": sha256_text(chain2),
        "sequence_pair_sha256": sha256_text(chain1 + "|" + chain2),
    }


def build_pair_sequence_table(pairs, template):
    """Reconstruct one sequence record per unique LibA mutation-code pair."""
    _require({"pep", "aff"}.issubset(pairs.columns), "pair table lacks codes")
    unique = pairs[["pep", "aff"]].drop_duplicates().copy()
    unique = unique.sort_values(["pep", "aff"], kind="mergesort").reset_index(
        drop=True
    )
    records = []
    for row in unique.itertuples(index=False):
        _require(_within_liba_alphabet(str(row.pep), str(row.aff)),
                 "off-design pair reached sequence reconstruction")
        record = {
            "library": LIBRARY,
            "pep": str(row.pep),
            "aff": str(row.aff),
            "pair_uid": opaque_id(LIBRARY, str(row.pep), str(row.aff)),
            "peptide_uid": opaque_id(LIBRARY, "pep", str(row.pep)),
            "affibody_uid": opaque_id(LIBRARY, "aff", str(row.aff)),
        }
        record.update(_sequence_record(template, row.pep, row.aff))
        records.append(record)
    output = pd.DataFrame(records)
    _require(not output.empty, "no pair sequences were reconstructed")
    _require(output["pair_uid"].nunique() == len(output),
             "pair UID collision")
    _require(output["sequence_pair_sha256"].nunique() == len(output),
             "sequence-pair hash collision")
    return output


def validate_retention_rows(retention, template):
    _require(RETENTION_REQUIRED_COLUMNS.issubset(retention.columns),
             "retention sequence schema mismatch")
    selected = retention.loc[retention["library"].eq(LIBRARY)].copy()
    _require(not selected.empty, "retention table lacks LibA")
    _require(not bool(selected["pair_uid"].duplicated().any()),
             "duplicate LibA retention pair UID")
    _require(not bool(selected["sequence_pair_sha256"].duplicated().any()),
             "duplicate LibA retention sequence pair")
    for row in selected.itertuples(index=False):
        peptide = str(row.peptide_design_code)
        affibody = str(row.affibody_design_code)
        reconstructed = _sequence_record(template, peptide, affibody)
        _require(
            reconstructed["chain1_smart_hla_linker_peptide_sequence"]
            == str(row.chain1_smart_hla_linker_peptide_sequence),
            "retention chain 1 differs from provider template",
        )
        _require(
            reconstructed["chain2_affibody_sequence"]
            == str(row.chain2_affibody_sequence),
            "retention chain 2 differs from provider template",
        )
        for field in ("chain1_sha256", "chain2_sha256", "sequence_pair_sha256"):
            _require(reconstructed[field] == str(getattr(row, field)),
                     "retention {} mismatch".format(field))
    return selected.reset_index(drop=True)


def derive_fixed_negative_rows(raw_files, min_r001_count=MIN_NEGATIVE_R001_COUNT):
    """Recompute R001>=3 minus every observed R002--R014 key."""
    _require(int(min_r001_count) >= 1, "negative count floor must be positive")
    reference = load_count_round(raw_files[1], LIBRARY)
    reference = reference.loc[
        reference["count"].ge(int(min_r001_count)), ["pep", "aff", "count"]
    ].copy()
    remaining = set(key_series(reference).tolist())
    for round_index in NEGATIVE_EXCLUSION_ROUNDS:
        observed = load_presence(raw_files[round_index], LIBRARY)
        remaining.difference_update(key_series(observed).tolist())
    mask = key_series(reference).isin(remaining)
    result = reference.loc[mask].rename(columns={"count": "r001_count"})
    result = result.sort_values(["pep", "aff"], kind="mergesort").reset_index(
        drop=True
    )
    _require(not result.empty, "fixed negative rule returned no rows")
    return result


def derive_eligible_arm_labels(
    raw_files,
    retention,
    template,
    top_fraction=TOP_FRACTION,
    min_negative_r001_count=MIN_NEGATIVE_R001_COUNT,
):
    """Independently reconstruct all natural arm labels after fixed filters."""
    round_frames = {
        round_index: load_count_round(raw_files[round_index], LIBRARY)
        for round_index in POSITIVE_ROUNDS
    }
    positive_arms, summaries = derive_raw_positive_arms(
        round_frames, top_fraction=top_fraction
    )
    reference = load_count_round(raw_files[1], LIBRARY)[
        ["pep", "aff", "count"]
    ].rename(columns={"count": "r001_count"})
    negatives = derive_fixed_negative_rows(
        raw_files, min_r001_count=min_negative_r001_count
    )

    retention_keys = set(
        zip(
            retention["peptide_design_code"].astype(str),
            retention["affibody_design_code"].astype(str),
        )
    )
    retention_chain1 = set(retention["chain1_sha256"].astype(str))
    retention_chain2 = set(retention["chain2_sha256"].astype(str))
    pair_blocks = [negatives[["pep", "aff"]]]
    pair_blocks.extend(frame[["pep", "aff"]] for frame in positive_arms.values())
    all_pairs = pd.concat(pair_blocks, ignore_index=True).drop_duplicates()
    all_pairs = all_pairs.loc[
        [
            _within_liba_alphabet(str(peptide), str(affibody))
            for peptide, affibody in zip(all_pairs["pep"], all_pairs["aff"])
        ]
    ].copy()
    sequences = build_pair_sequence_table(all_pairs, template)
    sequences["eligible"] = (
        ~sequences[["pep", "aff"]].apply(tuple, axis=1).isin(retention_keys)
        & ~sequences["chain1_sha256"].isin(retention_chain1)
        & ~sequences["chain2_sha256"].isin(retention_chain2)
    )
    eligible_sequences = sequences.loc[sequences["eligible"]].drop(
        columns=["eligible"]
    )
    sequence_join_columns = [
        "pep",
        "aff",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    negative_eligible = negatives.merge(
        eligible_sequences[sequence_join_columns],
        on=["pep", "aff"],
        how="inner",
        validate="one_to_one",
    )
    negative_eligible["weak_label"] = 0
    negative_eligible["label_score"] = "0"

    blocks = []
    for arm in ARM_ORDER:
        positive = positive_arms[arm].merge(
            reference,
            on=["pep", "aff"],
            how="left",
            validate="one_to_one",
        )
        positive["r001_count"] = positive["r001_count"].fillna(0).astype(np.int64)
        positive = positive.merge(
            eligible_sequences[sequence_join_columns],
            on=["pep", "aff"],
            how="inner",
            validate="one_to_one",
        )
        positive["weak_label"] = 1
        positive_block = positive[
            [
                "pep",
                "aff",
                "r001_count",
                "weak_label",
                "label_score",
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
            ]
        ].copy()
        negative_block = negative_eligible[
            [
                "pep",
                "aff",
                "r001_count",
                "weak_label",
                "label_score",
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
                "chain1_sha256",
                "chain2_sha256",
                "sequence_pair_sha256",
            ]
        ].copy()
        arm_block = pd.concat(
            [positive_block, negative_block], ignore_index=True
        )
        arm_block.insert(0, "arm", arm)
        arm_block = arm_block.sort_values(
            ["weak_label", "pair_uid"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        _require(not bool(arm_block["pair_uid"].duplicated().any()),
                 "positive/negative overlap in {}".format(arm))
        _require(set(arm_block["weak_label"]) == {0, 1},
                 "{} lacks a class".format(arm))
        blocks.append(arm_block)
        summaries[arm]["eligible_positive_rows"] = int(
            arm_block["weak_label"].eq(1).sum()
        )
        summaries[arm]["eligible_negative_rows"] = int(
            arm_block["weak_label"].eq(0).sum()
        )
    labels = pd.concat(blocks, ignore_index=True)
    labels["r001_count"] = labels["r001_count"].astype(np.int64)
    _require(
        labels.loc[labels["weak_label"].eq(0)].groupby("arm")["pair_uid"]
        .apply(lambda values: sha256_text("\n".join(sorted(values))))
        .nunique()
        == 1,
        "fixed negative membership differs across arms",
    )
    return labels, summaries, eligible_sequences


def build_long_membership(labels, seeds=SIZE_MATCH_SEEDS):
    """Expand natural labels into natural and deterministic size-matched sets."""
    _require(set(labels["arm"]) == set(ARM_ORDER), "label arms mismatch")
    baseline = labels.loc[
        labels["arm"].eq("A") & labels["weak_label"].eq(1)
    ]
    target = int(len(baseline))
    _require(target > 0, "baseline positive target is empty")
    records = []
    for arm in ARM_ORDER:
        block = labels.loc[labels["arm"].eq(arm)].copy()
        positives = block.loc[block["weak_label"].eq(1)].copy()
        negatives = block.loc[block["weak_label"].eq(0)].copy()
        _require(len(positives) >= target,
                 "{} has fewer positives than baseline".format(arm))
        scenarios = [("natural", -1, block, {})]
        for seed in (() if arm == "A" else seeds):
            hashes = {
                pair_uid: deterministic_size_hash(seed, pair_uid)
                for pair_uid in positives["pair_uid"]
            }
            ranked_uids = sorted(hashes, key=lambda value: (hashes[value], value))
            selected_uids = set(ranked_uids[:target])
            selected_positive = positives.loc[
                positives["pair_uid"].isin(selected_uids)
            ].copy()
            selected = pd.concat([selected_positive, negatives], ignore_index=True)
            rank_map = {
                pair_uid: rank + 1 for rank, pair_uid in enumerate(ranked_uids)
            }
            scenarios.append(("size_matched", int(seed), selected, (hashes, rank_map)))
        for sampling, seed, selected, sampling_data in scenarios:
            if sampling == "size_matched":
                hashes, rank_map = sampling_data
            else:
                hashes, rank_map = {}, {}
            for row in selected.itertuples(index=False):
                is_positive = int(row.weak_label) == 1
                records.append(
                    {
                        "membership_uid": opaque_id(
                            SCHEMA_VERSION,
                            row.arm,
                            sampling,
                            str(seed),
                            row.pair_uid,
                        ),
                        "arm": row.arm,
                        "sampling": sampling,
                        "subset_seed": int(seed),
                        "sampling_rank": (
                            int(rank_map[row.pair_uid])
                            if sampling == "size_matched" and is_positive
                            else -1
                        ),
                        "sampling_sha256": (
                            hashes[row.pair_uid]
                            if sampling == "size_matched" and is_positive
                            else ""
                        ),
                        "pair_uid": row.pair_uid,
                        "peptide_uid": row.peptide_uid,
                        "affibody_uid": row.affibody_uid,
                        "peptide_design_code": row.pep,
                        "affibody_design_code": row.aff,
                        "sequence_pair_sha256": row.sequence_pair_sha256,
                        "weak_label": int(row.weak_label),
                        "label_score": row.label_score,
                        "r001_count": int(row.r001_count),
                    }
                )
    output = pd.DataFrame(records)
    output["_arm_order"] = output["arm"].map(
        {arm: index for index, arm in enumerate(ARM_ORDER)}
    )
    output["_sampling_order"] = output["sampling"].map(
        {"natural": 0, "size_matched": 1}
    )
    output = output.sort_values(
        ["_arm_order", "_sampling_order", "subset_seed", "weak_label", "pair_uid"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).drop(columns=["_arm_order", "_sampling_order"])
    output.insert(0, "membership_index", np.arange(len(output), dtype=np.int64))
    output = output.loc[:, list(MEMBERSHIP_COLUMNS)].reset_index(drop=True)
    _require(output["membership_uid"].nunique() == len(output),
             "duplicate membership UID")
    _require(
        not bool(
            output[["arm", "sampling", "subset_seed", "pair_uid"]]
            .duplicated()
            .any()
        ),
        "duplicate arm/scenario membership",
    )
    for arm in ARM_ORDER:
        natural = output.loc[
            output["arm"].eq(arm) & output["sampling"].eq("natural")
        ]
        for seed in (() if arm == "A" else seeds):
            matched = output.loc[
                output["arm"].eq(arm)
                & output["sampling"].eq("size_matched")
                & output["subset_seed"].eq(int(seed))
            ]
            _require(int(matched["weak_label"].eq(1).sum()) == target,
                     "size-matched positive count mismatch")
            _require(
                set(matched.loc[matched["weak_label"].eq(0), "pair_uid"])
                == set(natural.loc[natural["weak_label"].eq(0), "pair_uid"]),
                "size matching changed the fixed negatives",
            )
    return output


def membership_summary(membership):
    rows = []
    for keys, block in membership.groupby(
        ["arm", "sampling", "subset_seed"], sort=False
    ):
        arm, sampling, subset_seed = keys
        rows.append(
            {
                "arm": arm,
                "sampling": sampling,
                "subset_seed": int(subset_seed),
                "rows": int(len(block)),
                "positive_rows": int(block["weak_label"].eq(1).sum()),
                "negative_rows": int(block["weak_label"].eq(0).sum()),
                "unique_peptides": int(block["peptide_uid"].nunique()),
                "unique_affibodies": int(block["affibody_uid"].nunique()),
                "pair_uid_sha256": sha256_text(
                    "\n".join(sorted(block["pair_uid"].astype(str)))
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_label_artifact(
    label_dir,
    derived_labels,
    derived_summaries,
    raw_files,
    retention_source_sha256,
):
    """Validate the independently generated CPU labels against recomputation."""
    label_path = label_dir / "arm_labels.csv"
    size_path = label_dir / "size_matched_positive_membership.csv"
    summary_path = label_dir / "arm_summary.csv"
    manifest_path = label_dir / "manifest.json"
    for path in (label_path, size_path, summary_path, manifest_path):
        _require(path.is_file(), "missing label artifact {}".format(path))
    manifest = read_json(manifest_path)
    _require(
        manifest.get("schema_version") == "liba-later-round-labels-v1",
        "later-round label schema mismatch",
    )
    for path in (label_path, size_path, summary_path):
        _require(
            _manifest_output_hash(manifest, path.name) == sha256_file(path),
            "label output hash mismatch for {}".format(path.name),
        )
    code = manifest.get("code", {})
    code_path = Path(str(code.get("path", "")))
    _require(code_path.is_file(), "label builder source is missing")
    _require(sha256_file(code_path) == code.get("sha256"),
             "label builder source hash mismatch")
    configuration = manifest.get("configuration", {})
    _require(configuration.get("library") == LIBRARY,
             "label artifact library mismatch")
    _require(float(configuration.get("top_fraction", -1)) == TOP_FRACTION,
             "label top fraction mismatch")
    _require(configuration.get("top_fraction_ties") == "inclusive",
             "label tie policy mismatch")
    _require(int(configuration.get("size_match_target_positive", -1))
             == EXPECTED_BASELINE_POSITIVES,
             "label size-match target mismatch")
    _require(tuple(configuration.get("size_match_seeds", [])) == SIZE_MATCH_SEEDS,
             "label size-match seeds mismatch")
    for arm in ARM_ORDER:
        observed = configuration.get("arms", {}).get(arm, {})
        expected = ARM_FORMULAS[arm]
        _require(tuple(observed.get("rounds", [])) == tuple(expected["rounds"]),
                 "label round definition mismatch for arm {}".format(arm))
        expected_kind = (
            "equal_round_mean_count_over_round_depth"
            if arm == "C"
            else "raw_count_sum"
        )
        _require(observed.get("score_kind") == expected_kind,
                 "label score definition mismatch for arm {}".format(arm))
    _require(
        manifest.get("sources", {}).get("retention_csv", {}).get("sha256")
        == retention_source_sha256,
        "label and retention sequence artifacts use different retention matrices",
    )
    raw_manifest = manifest.get("sources", {}).get("raw_rounds", {})
    for round_index in range(1, 15):
        key = "R{:03d}".format(round_index)
        _require(
            raw_manifest.get(key, {}).get("sha256")
            == sha256_file(raw_files[round_index]),
            "label raw hash mismatch for {}".format(key),
        )

    labels = read_string_csv(label_path)
    _require(LABEL_REQUIRED_COLUMNS.issubset(labels.columns),
             "label table schema mismatch")
    _require(set(labels["arm"]) == set(ARM_ORDER), "label table arms mismatch")
    _require(not bool(labels[["arm", "pair_uid"]].duplicated().any()),
             "duplicate canonical arm/pair key")
    for column in (
        "r001_count",
        "weak_label",
        "within_declared_library_alphabet",
        "strict_liba_retention_identity_cold_eligible",
    ):
        labels[column] = pd.to_numeric(labels[column], errors="raise").astype(
            np.int64
        )
    _require(bool(labels["library"].eq(LIBRARY).all()),
             "canonical label row has wrong library")
    _require(bool(labels["within_declared_library_alphabet"].eq(1).all()),
             "canonical labels contain off-design rows")
    _require(
        bool(labels["strict_liba_retention_identity_cold_eligible"].eq(1).all()),
        "canonical labels contain identity-overlap rows",
    )
    derived = derived_labels.copy()
    key_columns = ["arm", "pair_uid"]
    comparison = labels.merge(
        derived,
        on=key_columns,
        how="outer",
        suffixes=("_artifact", "_derived"),
        indicator=True,
        validate="one_to_one",
    )
    _require(bool(comparison["_merge"].eq("both").all()),
             "canonical and recomputed label membership differ")
    for column in (
        "pep",
        "aff",
        "peptide_uid",
        "affibody_uid",
        "r001_count",
        "weak_label",
    ):
        left = comparison["{}_artifact".format(column)].astype(str)
        right = comparison["{}_derived".format(column)].astype(str)
        _require(bool(left.eq(right).all()),
                 "canonical and recomputed {} differ".format(column))
    artifact_score = pd.to_numeric(
        comparison["label_score_artifact"], errors="raise"
    ).to_numpy(dtype=np.float64)
    derived_score = pd.to_numeric(
        comparison["label_score_derived"], errors="raise"
    ).to_numpy(dtype=np.float64)
    _require(
        bool(np.allclose(artifact_score, derived_score, rtol=1e-12, atol=1e-15)),
        "canonical and recomputed label scores differ",
    )

    size_membership = read_string_csv(size_path)
    _require(SIZE_REQUIRED_COLUMNS.issubset(size_membership.columns),
             "size-matched membership schema mismatch")
    _require(
        not bool(
            size_membership[["arm", "sampling_seed", "pair_uid"]]
            .duplicated()
            .any()
        ),
        "duplicate canonical size-matched key",
    )
    _require(set(size_membership["arm"]) == {"B", "C", "D"},
             "canonical size matching must contain arms B/C/D")
    expected_size = []
    for arm in ("B", "C", "D"):
        positives = derived.loc[
            derived["arm"].eq(arm) & derived["weak_label"].eq(1)
        ]
        for seed in SIZE_MATCH_SEEDS:
            ranked = sorted(
                positives["pair_uid"].astype(str),
                key=lambda uid: (deterministic_size_hash(seed, uid), uid),
            )[:EXPECTED_BASELINE_POSITIVES]
            expected_size.extend((arm, str(seed), uid) for uid in ranked)
    observed_size = set(
        zip(
            size_membership["arm"].astype(str),
            size_membership["sampling_seed"].astype(str),
            size_membership["pair_uid"].astype(str),
        )
    )
    _require(observed_size == set(expected_size),
             "canonical and recomputed size-matched memberships differ")

    summary = read_string_csv(summary_path)
    _require(set(summary["arm"]) == set(ARM_ORDER),
             "canonical arm summary mismatch")
    for arm in ARM_ORDER:
        row = summary.loc[summary["arm"].eq(arm)]
        _require(len(row) == 1, "canonical summary has duplicate arm")
        expected = derived_summaries[arm]
        mappings = {
            "union_rows": "universe_rows",
            "requested_top2_rank": "requested_rank",
            "positive_before_design_and_identity_filter": "positive_rows_before_filters",
            "eligible_positive": "eligible_positive_rows",
            "eligible_negative": "eligible_negative_rows",
        }
        for observed_field, expected_field in mappings.items():
            _require(
                int(row.iloc[0][observed_field]) == int(expected[expected_field]),
                "canonical summary {} mismatch for arm {}".format(
                    observed_field, arm
                ),
            )
        observed_cutoff = float(row.iloc[0]["inclusive_cutoff"])
        expected_cutoff = float(expected["inclusive_cutoff"])
        _require(
            bool(np.isclose(observed_cutoff, expected_cutoff, rtol=1e-12, atol=1e-15)),
            "canonical cutoff mismatch for arm {}".format(arm),
        )
    return labels, size_membership, summary, manifest


def validate_baseline_and_retention_reuse(labels, retention, old_metadata):
    """Bind arm A and all LibA retention rows to the immutable old cache."""
    old = pd.DataFrame(
        {
            key: np.asarray(values).astype(str)
            for key, values in old_metadata.items()
        }
    )
    baseline_old = old.loc[
        old["library"].eq(LIBRARY)
        & old["source_kind"].eq("weak")
        & old[
            "upstream_library_local_strict_retention_identity_cold_eligible"
        ].eq("1")
    ].copy()
    baseline = labels.loc[labels["arm"].eq("A")].copy()
    _require(
        int(baseline["weak_label"].eq(1).sum()) == EXPECTED_BASELINE_POSITIVES,
        "arm A positive membership count changed",
    )
    _require(
        int(baseline["weak_label"].eq(0).sum()) == EXPECTED_BASELINE_NEGATIVES,
        "fixed negative membership count changed",
    )
    _require(len(baseline_old) == len(baseline),
             "old-cache LibA baseline row count differs")
    joined = baseline.merge(
        baseline_old[["pair_uid", "sequence_pair_sha256", "weak_label"]],
        on="pair_uid",
        how="outer",
        suffixes=("_derived", "_old"),
        indicator=True,
        validate="one_to_one",
    )
    _require(bool(joined["_merge"].eq("both").all()),
             "arm A membership differs from immutable old cache")
    _require(
        bool(
            joined["sequence_pair_sha256_derived"]
            .astype(str)
            .eq(joined["sequence_pair_sha256_old"].astype(str))
            .all()
        ),
        "arm A sequence hashes differ from immutable old cache",
    )
    _require(
        bool(
            joined["weak_label_derived"]
            .astype(str)
            .eq(joined["weak_label_old"].astype(str))
            .all()
        ),
        "arm A labels differ from immutable old cache",
    )
    retention_old = old.loc[
        old["library"].eq(LIBRARY) & old["source_kind"].eq("retention")
    ]
    retention_join = retention.merge(
        retention_old[
            ["pair_uid", "sequence_pair_sha256", "chain1_sha256", "chain2_sha256"]
        ],
        on="pair_uid",
        how="outer",
        suffixes=("_retention", "_old"),
        indicator=True,
        validate="one_to_one",
    )
    _require(bool(retention_join["_merge"].eq("both").all()),
             "LibA retention rows are not fully reusable from old cache")
    for field in ("sequence_pair_sha256", "chain1_sha256", "chain2_sha256"):
        _require(
            bool(
                retention_join["{}_retention".format(field)]
                .astype(str)
                .eq(retention_join["{}_old".format(field)].astype(str))
                .all()
            ),
            "LibA retention {} differs from old cache".format(field),
        )
    return old


def assemble_delta_rows_and_reuse(
    labels,
    eligible_sequences,
    retention,
    old_metadata,
    sentinel_count=DEFAULT_SENTINEL_COUNT,
):
    """Resolve union pairs by sequence hash and materialize misses + sentinels."""
    _require(int(sentinel_count) >= 1, "sentinel count must be positive")
    training_uids = set(labels["pair_uid"].astype(str))
    training = eligible_sequences.loc[
        eligible_sequences["pair_uid"].astype(str).isin(training_uids)
    ].copy()
    _require(len(training) == len(training_uids),
             "training union lacks reconstructed sequences")
    training = training.rename(
        columns={"pep": "peptide_design_code", "aff": "affibody_design_code"}
    )
    training.insert(0, "source_kind", "weak")
    retention_block = retention[
        [
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "peptide_design_code",
            "affibody_design_code",
            "chain1_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
        ]
    ].copy()
    retention_block.insert(0, "library", LIBRARY)
    retention_block.insert(0, "source_kind", "retention")
    candidate_columns = [
        "source_kind",
        "library",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_design_code",
        "affibody_design_code",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ]
    candidates = pd.concat(
        [training[candidate_columns], retention_block[candidate_columns]],
        ignore_index=True,
    )
    _require(not bool(candidates["pair_uid"].duplicated().any()),
             "training/retention pair UID overlap")
    _require(not bool(candidates["sequence_pair_sha256"].duplicated().any()),
             "training/retention sequence overlap")
    candidates["_source_order"] = candidates["source_kind"].map(
        {"weak": 0, "retention": 1}
    )
    candidates = candidates.sort_values(
        ["_source_order", "pair_uid"], kind="mergesort"
    ).drop(columns=["_source_order"]).reset_index(drop=True)

    old_hashes = np.asarray(old_metadata["sequence_pair_sha256"]).astype(str)
    old_lookup = {value: index for index, value in enumerate(old_hashes)}
    candidates["old_cache_row_index"] = [
        int(old_lookup.get(value, -1))
        for value in candidates["sequence_pair_sha256"].astype(str)
    ]
    retention_missing = candidates["source_kind"].eq("retention") & candidates[
        "old_cache_row_index"
    ].eq(-1)
    _require(not bool(retention_missing.any()),
             "a LibA retention feature is missing from the old cache")
    missing = candidates.loc[candidates["old_cache_row_index"].eq(-1)].copy()
    _require(not missing.empty, "later-round union produced no cache misses")
    reusable_training = candidates.loc[
        candidates["source_kind"].eq("weak")
        & candidates["old_cache_row_index"].ge(0)
    ].copy()
    _require(len(reusable_training) >= int(sentinel_count),
             "not enough reusable training rows for overlap sentinels")
    reusable_training["_sentinel_order"] = [
        sha256_text("overlap-sentinel|{}".format(value))
        for value in reusable_training["sequence_pair_sha256"].astype(str)
    ]
    sentinels = reusable_training.sort_values(
        ["_sentinel_order", "sequence_pair_sha256"], kind="mergesort"
    ).head(int(sentinel_count)).drop(columns=["_sentinel_order"])
    missing["cache_role"] = "missing"
    sentinels["cache_role"] = "overlap_sentinel"
    rows = pd.concat([missing, sentinels], ignore_index=True)
    rows["_role_order"] = rows["cache_role"].map(
        {"missing": 0, "overlap_sentinel": 1}
    )
    rows = rows.sort_values(
        ["_role_order", "sequence_pair_sha256"], kind="mergesort"
    ).drop(columns=["_role_order"]).reset_index(drop=True)
    rows.insert(0, "row_index", np.arange(len(rows), dtype=np.int64))
    rows.insert(
        1,
        "delta_cache_uid",
        [
            opaque_id(SCHEMA_VERSION, role, value)
            for role, value in zip(rows["cache_role"], rows["sequence_pair_sha256"])
        ],
    )
    rows = rows.loc[:, list(ROW_COLUMNS)]
    delta_lookup = {
        value: int(index)
        for index, value in zip(rows["row_index"], rows["sequence_pair_sha256"])
    }
    sentinel_hashes = set(sentinels["sequence_pair_sha256"].astype(str))
    reuse = candidates[
        [
            "source_kind",
            "library",
            "pair_uid",
            "peptide_uid",
            "affibody_uid",
            "peptide_design_code",
            "affibody_design_code",
            "chain1_sha256",
            "chain2_sha256",
            "sequence_pair_sha256",
            "old_cache_row_index",
        ]
    ].copy()
    reuse["feature_source"] = np.where(
        reuse["old_cache_row_index"].ge(0), "old_cache", "delta_cache"
    )
    reuse["delta_row_index"] = [
        int(delta_lookup.get(value, -1))
        if source == "delta_cache"
        else -1
        for value, source in zip(
            reuse["sequence_pair_sha256"], reuse["feature_source"]
        )
    ]
    reuse["is_overlap_sentinel"] = reuse["sequence_pair_sha256"].isin(
        sentinel_hashes
    ).astype(int)
    reuse["sentinel_delta_row_index"] = [
        int(delta_lookup[value]) if value in sentinel_hashes else -1
        for value in reuse["sequence_pair_sha256"]
    ]
    reuse.insert(0, "resolver_index", np.arange(len(reuse), dtype=np.int64))
    reuse = reuse.loc[:, list(REUSE_COLUMNS)]
    validate_prepared_rows(rows.astype(str))
    validate_reuse_index(reuse.astype(str), rows.astype(str))
    return rows, reuse


def validate_prepared_rows(frame):
    _require(tuple(frame.columns) == ROW_COLUMNS,
             "prepared delta row schema mismatch")
    indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    _require(np.array_equal(indices, np.arange(len(frame), dtype=np.int64)),
             "prepared delta indices are not contiguous")
    _require(frame["delta_cache_uid"].nunique() == len(frame),
             "duplicate delta cache UID")
    _require(frame["sequence_pair_sha256"].nunique() == len(frame),
             "duplicate prepared sequence pair")
    _require(set(frame["library"]) == {LIBRARY},
             "prepared delta includes another library")
    _require(set(frame["cache_role"]) == {"missing", "overlap_sentinel"},
             "prepared delta lacks a cache role")
    _require(
        bool(
            frame.loc[frame["cache_role"].eq("missing"), "old_cache_row_index"]
            .eq("-1")
            .all()
        ),
        "a cache miss has an old-cache row index",
    )
    _require(
        bool(
            pd.to_numeric(
                frame.loc[
                    frame["cache_role"].eq("overlap_sentinel"),
                    "old_cache_row_index",
                ],
                errors="raise",
            ).ge(0).all()
        ),
        "an overlap sentinel lacks an old-cache row index",
    )
    for sequence_column, length in zip(SEQUENCE_COLUMNS, (CHAIN1_LENGTH, CHAIN2_LENGTH)):
        _require(bool(frame[sequence_column].map(len).eq(length).all()),
                 "prepared delta sequence length mismatch")
        _require(
            not bool(
                frame[sequence_column].str.contains(OMITTED_LINKER, regex=False).any()
            ),
            "experimental linker leaked into a MINT chain",
        )
    expected_pair_hashes = [
        sha256_text(chain1 + "|" + chain2)
        for chain1, chain2 in zip(frame[SEQUENCE_COLUMNS[0]], frame[SEQUENCE_COLUMNS[1]])
    ]
    _require(expected_pair_hashes == frame["sequence_pair_sha256"].tolist(),
             "prepared sequence-pair hash mismatch")
    _require(
        [sha256_text(value) for value in frame[SEQUENCE_COLUMNS[0]]]
        == frame["chain1_sha256"].tolist(),
        "prepared chain-1 hash mismatch",
    )
    _require(
        [sha256_text(value) for value in frame[SEQUENCE_COLUMNS[1]]]
        == frame["chain2_sha256"].tolist(),
        "prepared chain-2 hash mismatch",
    )
    return indices


def validate_reuse_index(reuse, rows):
    _require(tuple(reuse.columns) == REUSE_COLUMNS,
             "reuse-index schema mismatch")
    indices = pd.to_numeric(reuse["resolver_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    _require(np.array_equal(indices, np.arange(len(reuse), dtype=np.int64)),
             "reuse indices are not contiguous")
    _require(reuse["pair_uid"].nunique() == len(reuse),
             "duplicate reuse pair UID")
    _require(reuse["sequence_pair_sha256"].nunique() == len(reuse),
             "duplicate reuse sequence pair")
    _require(set(reuse["feature_source"]) == {"old_cache", "delta_cache"},
             "reuse index lacks a feature source")
    missing = reuse["feature_source"].eq("delta_cache")
    old = reuse["feature_source"].eq("old_cache")
    _require(
        bool(pd.to_numeric(reuse.loc[missing, "old_cache_row_index"]).eq(-1).all()),
        "delta-resolved row unexpectedly maps to old cache",
    )
    _require(
        bool(pd.to_numeric(reuse.loc[missing, "delta_row_index"]).ge(0).all()),
        "delta-resolved row lacks delta index",
    )
    _require(
        bool(pd.to_numeric(reuse.loc[old, "old_cache_row_index"]).ge(0).all()),
        "old-resolved row lacks old index",
    )
    missing_rows = set(
        rows.loc[rows["cache_role"].eq("missing"), "sequence_pair_sha256"]
    )
    _require(
        missing_rows
        == set(reuse.loc[missing, "sequence_pair_sha256"]),
        "prepared misses and reuse index disagree",
    )
    sentinel_rows = set(
        rows.loc[
            rows["cache_role"].eq("overlap_sentinel"), "sequence_pair_sha256"
        ]
    )
    _require(
        sentinel_rows
        == set(
            reuse.loc[reuse["is_overlap_sentinel"].eq("1"), "sequence_pair_sha256"]
        ),
        "prepared sentinels and reuse index disagree",
    )


def run_prepare(args):
    started = time.time()
    _require(float(args.top_fraction) == TOP_FRACTION,
             "this artifact is locked to top_fraction=0.02")
    _require(int(args.min_negative_r001_count) == MIN_NEGATIVE_R001_COUNT,
             "this artifact is locked to R001>=3 negatives")
    _require(int(args.sentinel_count) >= 1, "sentinel count must be positive")
    retention_manifest_path = args.retention_manifest
    if retention_manifest_path is None:
        retention_manifest_path = args.retention_sequences.with_suffix(
            ".manifest.json"
        )
    for path in (
        args.raw_root,
        args.label_dir,
        args.retention_sequences,
        retention_manifest_path,
        args.sequence_zip,
        args.old_cache,
        args.old_cache_manifest,
    ):
        _require(path.exists(), "missing prepare input {}".format(path))
    retention_manifest, retention_hash, zip_hash = validate_retention_lineage(
        args.retention_sequences, retention_manifest_path, args.sequence_zip
    )
    templates, template_member, template_deck_hash = load_provider_templates(
        args.sequence_zip
    )
    _require(
        retention_manifest.get("sources", {})
        .get("sequence_zip", {})
        .get("deck_sha256")
        == template_deck_hash,
        "loaded provider template differs from retention manifest",
    )
    retention_all = read_string_csv(args.retention_sequences)
    retention = validate_retention_rows(retention_all, templates[LIBRARY])
    raw_files_all = discover_raw_round_files(args.raw_root)
    raw_files = raw_files_all[LIBRARY]
    labels, arm_summaries, eligible_sequences = derive_eligible_arm_labels(
        raw_files,
        retention,
        templates[LIBRARY],
        top_fraction=args.top_fraction,
        min_negative_r001_count=args.min_negative_r001_count,
    )
    for arm in ARM_ORDER:
        subset = labels.loc[labels["arm"].eq(arm)]
        _require(
            int(subset["weak_label"].eq(1).sum())
            == EXPECTED_POSITIVES_BY_ARM[arm],
            "eligible positive count changed for arm {}".format(arm),
        )
        _require(
            int(subset["weak_label"].eq(0).sum())
            == EXPECTED_BASELINE_NEGATIVES,
            "eligible negative count changed for arm {}".format(arm),
        )
    retention_source_hash = (
        retention_manifest.get("sources", {})
        .get("retention_csv", {})
        .get("sha256")
    )
    _, canonical_size, canonical_summary, label_manifest = validate_label_artifact(
        args.label_dir,
        labels,
        arm_summaries,
        raw_files,
        retention_source_hash,
    )
    del canonical_size, canonical_summary
    old_manifest, old_cache_hash, old_metadata = validate_old_model_manifest(
        args.old_cache, args.old_cache_manifest
    )
    old_prepare_record = old_manifest.get("input", {}).get("prepare_manifest", {})
    old_prepare_path = Path(str(old_prepare_record.get("path", "")))
    _require(old_prepare_path.is_file(), "old prepare manifest is missing")
    _require(
        sha256_file(old_prepare_path) == old_prepare_record.get("sha256"),
        "old prepare-manifest hash mismatch",
    )
    old_prepare = read_json(old_prepare_path)
    _require(
        old_prepare.get("sources", {})
        .get("retention_sequences", {})
        .get("sha256")
        == retention_hash,
        "old cache uses different retention sequences",
    )
    _require(
        old_prepare.get("sources", {}).get("sequence_zip", {}).get("sha256")
        == zip_hash,
        "old cache uses a different provider sequence ZIP",
    )
    old_frame = validate_baseline_and_retention_reuse(
        labels, retention, old_metadata
    )
    del old_frame
    membership = build_long_membership(labels, seeds=SIZE_MATCH_SEEDS)
    rows, reuse = assemble_delta_rows_and_reuse(
        labels,
        eligible_sequences,
        retention,
        old_metadata,
        sentinel_count=args.sentinel_count,
    )
    summary = membership_summary(membership)

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=True)
    paths = {
        ROWS_FILENAME: output_dir / ROWS_FILENAME,
        REUSE_FILENAME: output_dir / REUSE_FILENAME,
        MEMBERSHIP_FILENAME: output_dir / MEMBERSHIP_FILENAME,
        SUMMARY_FILENAME: output_dir / SUMMARY_FILENAME,
    }
    _atomic_write_csv(paths[ROWS_FILENAME], rows)
    _atomic_write_csv(paths[REUSE_FILENAME], reuse)
    _atomic_write_csv(paths[MEMBERSHIP_FILENAME], membership)
    _atomic_write_csv(paths[SUMMARY_FILENAME], summary)
    raw_hashes = {
        "R{:03d}".format(round_index): {
            "path": str(raw_files[round_index].resolve()),
            "sha256": sha256_file(raw_files[round_index]),
        }
        for round_index in range(1, 15)
    }
    formula_hash = canonical_json_sha256(LABEL_CONTRACT)
    cutoff_records = [arm_summaries[arm] for arm in ARM_ORDER]
    manifest_path = output_dir / PREPARE_MANIFEST_FILENAME
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare",
        "analysis_status": "retrospective_exploratory",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "label_contract": {
            "payload": LABEL_CONTRACT,
            "sha256": formula_hash,
            "cutoffs": cutoff_records,
            "cutoffs_sha256": canonical_json_sha256(cutoff_records),
        },
        "sources": {
            "raw_root": str(args.raw_root.resolve()),
            "raw_rounds": raw_hashes,
            "canonical_labels": {
                "directory": str(args.label_dir.resolve()),
                "manifest": str((args.label_dir / "manifest.json").resolve()),
                "manifest_sha256": sha256_file(args.label_dir / "manifest.json"),
                "outputs": {
                    name: sha256_file(args.label_dir / name)
                    for name in (
                        "arm_labels.csv",
                        "size_matched_positive_membership.csv",
                        "arm_summary.csv",
                    )
                },
                "schema_version": label_manifest.get("schema_version"),
            },
            "retention_sequences": {
                "path": str(args.retention_sequences.resolve()),
                "sha256": retention_hash,
            },
            "retention_manifest": {
                "path": str(retention_manifest_path.resolve()),
                "sha256": sha256_file(retention_manifest_path),
            },
            "retention_matrix_sha256": retention_source_hash,
            "sequence_zip": {
                "path": str(args.sequence_zip.resolve()),
                "sha256": zip_hash,
                "template_member": template_member,
                "template_deck_sha256": template_deck_hash,
            },
            "old_cache": {
                "path": str(args.old_cache.resolve()),
                "sha256": old_cache_hash,
            },
            "old_cache_manifest": {
                "path": str(args.old_cache_manifest.resolve()),
                "sha256": sha256_file(args.old_cache_manifest),
            },
            "old_prepare_manifest": {
                "path": str(old_prepare_path.resolve()),
                "sha256": sha256_file(old_prepare_path),
            },
        },
        "identity_policy": {
            "scope": "LibA only",
            "comparison": "full reconstructed chain1_sha256 and chain2_sha256",
            "equivalence": "under the fixed LibA templates this is equivalent to exact peptide and Affibody design-code identity",
            "libb_only_identities_excluded": False,
        },
        "provider_model_input": {
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
            "tcr_present": False,
            "separate_b2m_supplied": False,
        },
        "memberships": {
            "natural_and_size_matched_rows": int(len(membership)),
            "natural_positive_by_arm": {
                arm: int(
                    labels.loc[
                        labels["arm"].eq(arm) & labels["weak_label"].eq(1)
                    ].shape[0]
                )
                for arm in ARM_ORDER
            },
            "fixed_negative_rows": EXPECTED_BASELINE_NEGATIVES,
            "size_match_seeds": list(SIZE_MATCH_SEEDS),
            "membership_csv_sha256": sha256_file(paths[MEMBERSHIP_FILENAME]),
        },
        "resolution": {
            "candidate_union_rows_including_retention": int(len(reuse)),
            "old_cache_rows_reused": int(reuse["feature_source"].eq("old_cache").sum()),
            "true_cache_misses": int(reuse["feature_source"].eq("delta_cache").sum()),
            "overlap_sentinels": int(reuse["is_overlap_sentinel"].eq(1).sum()),
            "prepared_embedding_rows": int(len(rows)),
            "old_cache_is_immutable": True,
            "resolver_key": "sequence_pair_sha256",
        },
        "old_model_contract": old_manifest.get("model"),
        "code": {
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "mint_source_tree_sha256": old_builder.sha256_source_tree(
                REPO_ROOT / "mint"
            ),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "mode": private_mode(path),
                "columns": list(read_string_csv(path).columns),
            }
            for name, path in paths.items()
        },
        "permissions": {
            "directory": private_mode(output_dir),
            "files": "0600",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _require(old_cache_hash == sha256_file(args.old_cache),
             "old cache changed during prepare")
    _require(
        manifest["sources"]["old_cache_manifest"]["sha256"]
        == sha256_file(args.old_cache_manifest),
        "old cache manifest changed during prepare",
    )
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "output_dir": str(output_dir),
                "prepared_embedding_rows": int(len(rows)),
                "true_cache_misses": int(
                    reuse["feature_source"].eq("delta_cache").sum()
                ),
                "old_cache_rows_reused": int(
                    reuse["feature_source"].eq("old_cache").sum()
                ),
                "overlap_sentinels": int(
                    reuse["is_overlap_sentinel"].eq(1).sum()
                ),
            },
            sort_keys=True,
        )
    )


def load_prepared_input(input_dir):
    paths = {
        ROWS_FILENAME: input_dir / ROWS_FILENAME,
        REUSE_FILENAME: input_dir / REUSE_FILENAME,
        MEMBERSHIP_FILENAME: input_dir / MEMBERSHIP_FILENAME,
        SUMMARY_FILENAME: input_dir / SUMMARY_FILENAME,
    }
    manifest_path = input_dir / PREPARE_MANIFEST_FILENAME
    for path in list(paths.values()) + [manifest_path]:
        _require(path.is_file(), "prepared input is missing {}".format(path))
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION,
             "prepared schema mismatch")
    _require(manifest.get("stage") == "prepare",
             "input manifest is not prepare stage")
    _require(
        canonical_json_sha256(manifest.get("label_contract", {}).get("payload"))
        == manifest.get("label_contract", {}).get("sha256"),
        "prepared label contract hash mismatch",
    )
    _require(manifest["label_contract"]["payload"] == LABEL_CONTRACT,
             "prepared label contract differs from current contract")
    for name, path in paths.items():
        _require(
            manifest.get("outputs", {}).get(name, {}).get("sha256")
            == sha256_file(path),
            "prepared output hash mismatch for {}".format(name),
        )
    rows = read_string_csv(paths[ROWS_FILENAME])
    reuse = read_string_csv(paths[REUSE_FILENAME])
    membership = read_string_csv(paths[MEMBERSHIP_FILENAME])
    validate_prepared_rows(rows)
    validate_reuse_index(reuse, rows)
    _require(tuple(membership.columns) == MEMBERSHIP_COLUMNS,
             "prepared membership schema mismatch")
    _require(
        manifest.get("memberships", {}).get("membership_csv_sha256")
        == sha256_file(paths[MEMBERSHIP_FILENAME]),
        "prepared membership hash mismatch",
    )
    old_cache = Path(manifest["sources"]["old_cache"]["path"])
    old_manifest = Path(manifest["sources"]["old_cache_manifest"]["path"])
    _require(
        old_cache.is_file()
        and sha256_file(old_cache) == manifest["sources"]["old_cache"]["sha256"],
        "immutable old cache changed after prepare",
    )
    _require(
        old_manifest.is_file()
        and sha256_file(old_manifest)
        == manifest["sources"]["old_cache_manifest"]["sha256"],
        "immutable old cache manifest changed after prepare",
    )
    return rows, reuse, membership, manifest, paths, manifest_path


def shard_filename(shard_index, shard_count):
    return "delta-features-shard-{:03d}-of-{:03d}.npz".format(
        int(shard_index), int(shard_count)
    )


def shard_manifest_filename(shard_index, shard_count):
    return "delta-features-shard-{:03d}-of-{:03d}.manifest.json".format(
        int(shard_index), int(shard_count)
    )


def select_shard_rows(frame, shard_index, shard_count):
    _require(int(shard_count) >= 1, "shard count must be positive")
    _require(0 <= int(shard_index) < int(shard_count),
             "shard index is out of range")
    indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    selected = np.mod(indices, int(shard_count)) == int(shard_index)
    output = frame.loc[selected].copy().reset_index(drop=True)
    _require(not output.empty, "shard has no rows")
    return output


class DeltaPairDataset(Dataset):
    def __init__(self, frame):
        self.chain1 = frame[SEQUENCE_COLUMNS[0]].tolist()
        self.chain2 = frame[SEQUENCE_COLUMNS[1]].tolist()
        self.library = frame["library"].tolist()
        self.cache_uid = frame["delta_cache_uid"].tolist()

    def __len__(self):
        return len(self.cache_uid)

    def __getitem__(self, index):
        return (
            self.chain1[index],
            self.chain2[index],
            self.library[index],
            self.cache_uid[index],
        )


def metadata_arrays(frame):
    arrays = {}
    for column in NPZ_METADATA_COLUMNS:
        if column in ("row_index", "old_cache_row_index"):
            arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
                dtype=np.int64
            )
        else:
            arrays[column] = frame[column].to_numpy(dtype=str)
    return arrays


def feature_contract(
    rows_sha256,
    prepare_manifest_sha256,
    old_cache_sha256,
    old_cache_manifest_sha256,
    old_contract_sha256,
    checkpoint_sha256,
    config_sha256,
    mint_source_tree_sha256,
    script_sha256,
    pair_collator_source_sha256,
    shard_count,
    batch_size,
    gpu_name,
    gpu_capability,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows_sha256": rows_sha256,
        "prepare_manifest_sha256": prepare_manifest_sha256,
        "old_cache_sha256": old_cache_sha256,
        "old_cache_manifest_sha256": old_cache_manifest_sha256,
        "old_feature_contract_sha256": old_contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "mint_source_tree_sha256": mint_source_tree_sha256,
        "script_sha256": script_sha256,
        "pair_collator_source_sha256": pair_collator_source_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "batch_size": int(batch_size),
            "gpu_name": str(gpu_name),
            "gpu_capability": [int(value) for value in gpu_capability],
        },
        "model": "MINT use_multimer=True sep_chains=True",
        "layer": 33,
        "pooling": "separate residue mean for chain 1 and chain 2, concatenated",
        "representation": FEATURE_NAME,
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float32",
        "shard_rule": "row_index modulo shard_count",
        "shard_count": int(shard_count),
    }
    return canonical_json_sha256(payload), payload


def _validate_runtime_compatibility(old_manifest, batch_size, gpu_name, gpu_capability):
    old_runtime = old_manifest.get("contract", {}).get("payload", {}).get(
        "runtime", {}
    )
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "batch_size": int(batch_size),
        "gpu_name": str(gpu_name),
        "gpu_capability": [int(value) for value in gpu_capability],
    }
    for field in (
        "python",
        "torch",
        "torch_cuda",
        "numpy",
        "batch_size",
        "gpu_name",
        "gpu_capability",
    ):
        _require(observed[field] == old_runtime.get(field),
                 "runtime differs from old cache for {}".format(field))
    return observed


def run_extract_shard(args):
    started = time.time()
    _require(args.batch_size >= 1, "batch size must be positive")
    _require(args.shard_count >= 1, "shard count must be positive")
    _require(0 <= args.shard_index < args.shard_count,
             "shard index is out of range")
    _require(args.checkpoint.is_file(), "MINT checkpoint does not exist")
    _require(args.config.is_file(), "MINT config does not exist")
    rows, _, _, prepare_manifest, paths, prepare_manifest_path = load_prepared_input(
        args.input_dir
    )
    shard = select_shard_rows(rows, args.shard_index, args.shard_count)
    old_cache_path = Path(prepare_manifest["sources"]["old_cache"]["path"])
    old_manifest_path = Path(
        prepare_manifest["sources"]["old_cache_manifest"]["path"]
    )
    old_manifest, old_cache_hash, _ = validate_old_model_manifest(
        old_cache_path, old_manifest_path
    )
    mint_tree_hash, collator_path = validate_model_files_against_old(
        old_manifest, args.checkpoint, args.config
    )
    device = torch.device(args.device)
    _require(device.type == "cuda", "extract-shard requires a CUDA device")
    _require(torch.cuda.is_available(), "CUDA is not available")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_capability = torch.cuda.get_device_capability(device)
    runtime = _validate_runtime_compatibility(
        old_manifest, args.batch_size, gpu_name, gpu_capability
    )

    checkpoint_hash = sha256_file(args.checkpoint)
    config_hash = sha256_file(args.config)
    rows_hash = sha256_file(paths[ROWS_FILENAME])
    prepare_manifest_hash = sha256_file(prepare_manifest_path)
    old_manifest_hash = sha256_file(old_manifest_path)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    collator_hash = sha256_file(collator_path)
    old_contract_hash = old_manifest["contract"]["sha256"]
    contract_hash, contract_payload = feature_contract(
        rows_hash,
        prepare_manifest_hash,
        old_cache_hash,
        old_manifest_hash,
        old_contract_hash,
        checkpoint_hash,
        config_hash,
        mint_tree_hash,
        script_hash,
        collator_hash,
        args.shard_count,
        args.batch_size,
        gpu_name,
        gpu_capability,
    )
    output_dir = _ensure_private_directory(args.output_dir, must_be_new=False)
    output_path = output_dir / shard_filename(args.shard_index, args.shard_count)
    manifest_path = output_dir / shard_manifest_filename(
        args.shard_index, args.shard_count
    )
    _require(not output_path.exists(), "shard output exists; refusing overwrite")
    _require(not manifest_path.exists(), "shard manifest exists; refusing overwrite")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(
        DeltaPairDataset(shard),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    wrapper = MINTWrapper(
        load_config(str(args.config)),
        str(args.checkpoint),
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()
    observed_uids = []
    blocks = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, _, cache_uids) in enumerate(loader):
            values = wrapper(chains.to(device), chain_ids.to(device))
            _require(tuple(values.shape[1:]) == (FEATURE_DIMENSION,),
                     "unexpected MINT feature dimension")
            blocks.append(values.float().cpu().numpy())
            observed_uids.extend(cache_uids)
            if (step + 1) % 50 == 0 or step + 1 == len(loader):
                print(
                    "delta shard {}/{}: embedded {}/{} batches".format(
                        args.shard_index, args.shard_count, step + 1, len(loader)
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    _require(observed_uids == shard["delta_cache_uid"].tolist(),
             "embedding row order changed")
    features = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    _require(features.shape == (len(shard), FEATURE_DIMENSION),
             "shard feature shape mismatch")
    _require(bool(np.isfinite(features).all()),
             "shard contains non-finite features")
    arrays = metadata_arrays(shard)
    arrays[FEATURE_NAME] = features
    stable_inputs = {
        "rows": (paths[ROWS_FILENAME], rows_hash),
        "prepare_manifest": (prepare_manifest_path, prepare_manifest_hash),
        "old_cache": (old_cache_path, old_cache_hash),
        "old_cache_manifest": (old_manifest_path, old_manifest_hash),
        "checkpoint": (args.checkpoint, checkpoint_hash),
        "config": (args.config, config_hash),
        "script": (script_path, script_hash),
        "pair_collator": (collator_path, collator_hash),
    }
    for name, values in stable_inputs.items():
        path, expected_hash = values
        _require(sha256_file(path) == expected_hash,
                 "{} changed during extraction".format(name))
    _require(old_builder.sha256_source_tree(REPO_ROOT / "mint") == mint_tree_hash,
             "MINT source tree changed during extraction")
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    index_array = arrays["row_index"].astype("<i8", copy=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "extract_shard",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "hostname": socket.gethostname(),
        "input": {
            "rows_csv": {"path": str(paths[ROWS_FILENAME].resolve()), "sha256": rows_hash},
            "prepare_manifest": {"path": str(prepare_manifest_path.resolve()), "sha256": prepare_manifest_hash},
            "old_cache": {"path": str(old_cache_path.resolve()), "sha256": old_cache_hash},
            "old_cache_manifest": {"path": str(old_manifest_path.resolve()), "sha256": old_manifest_hash},
        },
        "model": {
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": checkpoint_hash},
            "config": {"path": str(args.config.resolve()), "sha256": config_hash},
            "mint_source_tree_sha256": mint_tree_hash,
            "script_sha256": script_hash,
            "pair_collator_source": {"path": str(collator_path.resolve()), "sha256": collator_hash},
            "use_multimer": True,
            "sep_chains": True,
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
            "layer": 33,
            "pooling": "separate residue mean for chain 1 and chain 2, concatenated",
            "feature_name": FEATURE_NAME,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_dtype": "float32",
        },
        "contract": {"sha256": contract_hash, "payload": contract_payload},
        "sharding": {
            "rule": "row_index modulo shard_count equals shard_index",
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "rows": int(len(shard)),
            "row_index_sha256": hashlib.sha256(index_array.tobytes()).hexdigest(),
            "delta_cache_uid_sha256": sha256_text("\n".join(observed_uids)),
        },
        "runtime": dict(runtime, peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)), device_argument=str(device), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
            "feature_shape": list(features.shape),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps({"stage": "extract_shard", "shard_index": int(args.shard_index), "shard_count": int(args.shard_count), "rows": int(len(shard)), "output": str(output_path)}, sort_keys=True))


def _load_and_validate_shard(
    shard_path,
    shard_manifest_path,
    rows,
    shard_index,
    shard_count,
    expected_contract=None,
):
    manifest = read_json(shard_manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION,
             "shard schema mismatch")
    _require(manifest.get("stage") == "extract_shard",
             "not an extraction manifest")
    sharding = manifest.get("sharding", {})
    _require(int(sharding.get("shard_index", -1)) == int(shard_index),
             "shard-index mismatch")
    _require(int(sharding.get("shard_count", -1)) == int(shard_count),
             "shard-count mismatch")
    contract = manifest.get("contract", {}).get("sha256")
    payload = manifest.get("contract", {}).get("payload")
    _require(isinstance(payload, dict) and bool(contract),
             "shard lacks a feature contract")
    _require(canonical_json_sha256(payload) == contract,
             "shard contract payload/hash mismatch")
    if expected_contract is not None:
        _require(contract == expected_contract,
                 "shards use different feature contracts")
    _require(int(payload.get("shard_count", -1)) == int(shard_count),
             "shard contract count mismatch")
    _require(
        payload.get("representation") == FEATURE_NAME
        and int(payload.get("feature_dimension", -1)) == FEATURE_DIMENSION
        and payload.get("feature_dtype") == "float32",
        "shard feature contract mismatch",
    )
    _require(
        payload.get("model") == "MINT use_multimer=True sep_chains=True"
        and int(payload.get("layer", -1)) == 33,
        "shard model contract mismatch",
    )
    model = manifest.get("model", {})
    model_contract_fields = {
        "checkpoint_sha256": model.get("checkpoint", {}).get("sha256"),
        "config_sha256": model.get("config", {}).get("sha256"),
        "mint_source_tree_sha256": model.get("mint_source_tree_sha256"),
        "script_sha256": model.get("script_sha256"),
        "pair_collator_source_sha256": model.get("pair_collator_source", {}).get(
            "sha256"
        ),
    }
    for field, observed in model_contract_fields.items():
        _require(payload.get(field) == observed,
                 "shard model/contract mismatch for {}".format(field))
    runtime_contract = payload.get("runtime", {})
    runtime_manifest = manifest.get("runtime", {})
    for field in (
        "python",
        "torch",
        "torch_cuda",
        "numpy",
        "batch_size",
        "gpu_name",
        "gpu_capability",
    ):
        _require(runtime_contract.get(field) == runtime_manifest.get(field),
                 "shard runtime/contract mismatch for {}".format(field))
    _require(
        manifest.get("output", {}).get("sha256") == sha256_file(shard_path),
        "shard output hash mismatch",
    )
    expected = select_shard_rows(rows, shard_index, shard_count)
    with np.load(str(shard_path), allow_pickle=False) as archive:
        expected_keys = set(NPZ_METADATA_COLUMNS) | {FEATURE_NAME}
        _require(set(archive.files) == expected_keys,
                 "shard NPZ schema mismatch")
        indices = np.asarray(archive["row_index"], dtype=np.int64)
        expected_indices = pd.to_numeric(
            expected["row_index"], errors="raise"
        ).to_numpy(dtype=np.int64)
        _require(np.array_equal(indices, expected_indices),
                 "shard row assignment mismatch")
        _require(
            hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()
            == sharding.get("row_index_sha256"),
            "shard row-index hash mismatch",
        )
        for column in NPZ_METADATA_COLUMNS:
            if column == "row_index":
                continue
            if column == "old_cache_row_index":
                observed = np.asarray(archive[column], dtype=np.int64)
                wanted = pd.to_numeric(
                    expected[column], errors="raise"
                ).to_numpy(dtype=np.int64)
            else:
                observed = np.asarray(archive[column]).astype(str)
                wanted = expected[column].to_numpy(dtype=str)
            _require(np.array_equal(observed, wanted),
                     "shard metadata mismatch in {}".format(column))
        features = np.asarray(archive[FEATURE_NAME], dtype=np.float32)
        _require(features.shape == (len(expected), FEATURE_DIMENSION),
                 "shard feature shape mismatch")
        _require(bool(np.isfinite(features).all()),
                 "shard contains non-finite features")
        features = features.copy()
    return manifest, expected_indices, features, contract


def build_sentinel_comparison(rows, delta_features, old_cache_path):
    sentinel = rows.loc[rows["cache_role"].eq("overlap_sentinel")].copy()
    _require(not sentinel.empty, "no overlap sentinels were prepared")
    delta_indices = pd.to_numeric(
        sentinel["row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    old_indices = pd.to_numeric(
        sentinel["old_cache_row_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    with np.load(str(old_cache_path), allow_pickle=False) as archive:
        old_hashes = np.asarray(archive["sequence_pair_sha256"]).astype(str)
        _require(
            np.array_equal(
                old_hashes[old_indices],
                sentinel["sequence_pair_sha256"].to_numpy(dtype=str),
            ),
            "sentinel old-cache indices resolve to different sequences",
        )
        old_features = np.asarray(archive[FEATURE_NAME], dtype=np.float32)[
            old_indices
        ].copy()
    new_features = delta_features[delta_indices]
    records = []
    for position, row in enumerate(sentinel.itertuples(index=False)):
        difference = np.abs(new_features[position] - old_features[position])
        records.append(
            {
                "sequence_pair_sha256": row.sequence_pair_sha256,
                "pair_uid": row.pair_uid,
                "old_cache_row_index": int(row.old_cache_row_index),
                "delta_row_index": int(row.row_index),
                "max_abs_difference": float(np.max(difference)),
                "mean_abs_difference": float(np.mean(difference)),
                "exact_equal": int(
                    np.array_equal(new_features[position], old_features[position])
                ),
            }
        )
    comparison = pd.DataFrame(records).sort_values(
        "sequence_pair_sha256", kind="mergesort"
    ).reset_index(drop=True)
    _require(bool(comparison["exact_equal"].eq(1).all()),
             "overlap sentinels do not exactly reproduce old features")
    return comparison


def run_merge(args):
    started = time.time()
    _require(args.shard_count >= 1, "shard count must be positive")
    rows, reuse, membership, prepare_manifest, paths, prepare_manifest_path = (
        load_prepared_input(args.input_dir)
    )
    _require(args.shards_dir.is_dir(), "shard directory does not exist")
    rows_hash = sha256_file(paths[ROWS_FILENAME])
    prepare_manifest_hash = sha256_file(prepare_manifest_path)
    old_cache_path = Path(prepare_manifest["sources"]["old_cache"]["path"])
    old_manifest_path = Path(
        prepare_manifest["sources"]["old_cache_manifest"]["path"]
    )
    old_manifest, old_cache_hash, _ = validate_old_model_manifest(
        old_cache_path, old_manifest_path
    )
    old_manifest_hash = sha256_file(old_manifest_path)
    _require(old_cache_hash == prepare_manifest["sources"]["old_cache"]["sha256"],
             "old cache changed since prepare")
    _require(
        old_manifest_hash
        == prepare_manifest["sources"]["old_cache_manifest"]["sha256"],
        "old cache manifest changed since prepare",
    )
    features = np.empty((len(rows), FEATURE_DIMENSION), dtype=np.float32)
    covered = np.zeros(len(rows), dtype=bool)
    shard_records = []
    expected_contract = None
    first_manifest = None
    for shard_index in range(args.shard_count):
        shard_path = args.shards_dir / shard_filename(shard_index, args.shard_count)
        shard_manifest_path = args.shards_dir / shard_manifest_filename(
            shard_index, args.shard_count
        )
        _require(shard_path.is_file(), "missing shard {}".format(shard_index))
        _require(shard_manifest_path.is_file(),
                 "missing shard manifest {}".format(shard_index))
        manifest, indices, values, contract = _load_and_validate_shard(
            shard_path,
            shard_manifest_path,
            rows,
            shard_index,
            args.shard_count,
            expected_contract=expected_contract,
        )
        if expected_contract is None:
            expected_contract = contract
            first_manifest = manifest
        _require(
            manifest.get("input", {}).get("rows_csv", {}).get("sha256")
            == rows_hash,
            "shard was extracted from different rows",
        )
        _require(
            manifest.get("input", {}).get("prepare_manifest", {}).get("sha256")
            == prepare_manifest_hash,
            "shard was extracted from a different prepare manifest",
        )
        _require(
            manifest.get("input", {}).get("old_cache", {}).get("sha256")
            == old_cache_hash,
            "shard references a different old cache",
        )
        _require(not bool(covered[indices].any()),
                 "duplicate row coverage across shards")
        features[indices] = values
        covered[indices] = True
        shard_records.append(
            {
                "shard_index": int(shard_index),
                "rows": int(len(indices)),
                "npz_path": str(shard_path.resolve()),
                "npz_sha256": sha256_file(shard_path),
                "manifest_path": str(shard_manifest_path.resolve()),
                "manifest_sha256": sha256_file(shard_manifest_path),
                "hostname": manifest.get("hostname"),
                "gpu_name": manifest.get("runtime", {}).get("gpu_name"),
            }
        )
    _require(bool(covered.all()), "shards do not cover every prepared row")
    _require(bool(np.isfinite(features).all()),
             "merged delta feature array is non-finite")
    comparison = build_sentinel_comparison(rows, features, old_cache_path)
    _require(old_cache_hash == sha256_file(old_cache_path),
             "old cache changed during sentinel comparison")

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=True)
    output_path = output_dir / FINAL_CACHE_FILENAME
    comparison_path = output_dir / SENTINEL_COMPARISON_FILENAME
    manifest_path = output_dir / FINAL_MANIFEST_FILENAME
    arrays = metadata_arrays(rows)
    arrays[FEATURE_NAME] = features
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    _atomic_write_csv(comparison_path, comparison)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "merge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "input": {
            "rows_csv": {"path": str(paths[ROWS_FILENAME].resolve()), "sha256": rows_hash},
            "reuse_index": {"path": str(paths[REUSE_FILENAME].resolve()), "sha256": sha256_file(paths[REUSE_FILENAME])},
            "label_membership_long": {"path": str(paths[MEMBERSHIP_FILENAME].resolve()), "sha256": sha256_file(paths[MEMBERSHIP_FILENAME])},
            "prepare_manifest": {"path": str(prepare_manifest_path.resolve()), "sha256": prepare_manifest_hash},
            "old_cache": {"path": str(old_cache_path.resolve()), "sha256": old_cache_hash},
            "old_cache_manifest": {"path": str(old_manifest_path.resolve()), "sha256": old_manifest_hash},
            "shards": shard_records,
        },
        "contract": {
            "sha256": expected_contract,
            "payload": first_manifest.get("contract", {}).get("payload"),
        },
        "model": old_manifest.get("model"),
        "delta_extraction_model": first_manifest.get("model"),
        "resolution": prepare_manifest.get("resolution"),
        "features": {
            "name": FEATURE_NAME,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "row_order": "ascending row_index from prepared cache_rows.csv",
            "cache_roles": {
                role: int(rows["cache_role"].eq(role).sum())
                for role in ("missing", "overlap_sentinel")
            },
        },
        "sentinel_validation": {
            "rule": "new and old float32 feature vectors must be exactly equal",
            "rows": int(len(comparison)),
            "all_exact_equal": bool(comparison["exact_equal"].eq(1).all()),
            "maximum_absolute_difference": float(comparison["max_abs_difference"].max()),
            "comparison": {
                "path": str(comparison_path.resolve()),
                "sha256": sha256_file(comparison_path),
                "mode": private_mode(comparison_path),
            },
        },
        "merge_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "directory_mode": private_mode(output_dir),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
        },
    }
    _require(old_cache_hash == sha256_file(old_cache_path),
             "old cache changed during merge")
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps({"stage": "merge", "rows": int(len(rows)), "feature_shape": list(features.shape), "sentinels_exact": True, "output": str(output_path)}, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser(
        "prepare", help="validate labels and materialize cache misses"
    )
    prepare.add_argument("--raw-root", required=True, type=Path)
    prepare.add_argument("--label-dir", required=True, type=Path)
    prepare.add_argument("--retention-sequences", required=True, type=Path)
    prepare.add_argument("--retention-manifest", type=Path)
    prepare.add_argument("--sequence-zip", required=True, type=Path)
    prepare.add_argument("--old-cache", required=True, type=Path)
    prepare.add_argument("--old-cache-manifest", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    prepare.add_argument(
        "--min-negative-r001-count", type=int, default=MIN_NEGATIVE_R001_COUNT
    )
    prepare.add_argument(
        "--sentinel-count", type=int, default=DEFAULT_SENTINEL_COUNT
    )

    extract = subparsers.add_parser(
        "extract-shard", help="extract one deterministic GPU delta shard"
    )
    extract.add_argument("--input-dir", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--checkpoint", required=True, type=Path)
    extract.add_argument("--config", required=True, type=Path)
    extract.add_argument("--shard-index", required=True, type=int)
    extract.add_argument("--shard-count", required=True, type=int)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--uncompressed", action="store_true")

    merge = subparsers.add_parser(
        "merge", help="validate shards and publish the frozen-MINT delta"
    )
    merge.add_argument("--input-dir", required=True, type=Path)
    merge.add_argument("--shards-dir", required=True, type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.add_argument("--shard-count", required=True, type=int)
    merge.add_argument("--uncompressed", action="store_true")

    args = parser.parse_args(argv)
    _require(args.command is not None, "a subcommand is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "extract-shard":
        run_extract_shard(args)
    elif args.command == "merge":
        run_merge(args)
    else:
        raise ValueError("unknown command {}".format(args.command))


if __name__ == "__main__":
    main()
