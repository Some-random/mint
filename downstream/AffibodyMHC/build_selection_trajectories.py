#!/usr/bin/env python
"""Build one R000--R014 sequencing trajectory per cached weak-label pair.

The raw provider tables contain one row per observed peptide--Affibody pair in
one sequencing round.  This script joins the fifteen rounds onto the distinct
weak-label pairs already present in the frozen-MINT cache.  A missing row is
stored as zero *observed reads*; it must not be interpreted as proof that the
biological pair was absent from the selected population.

Outputs are private, immutable, and bound to the exact cache row table and raw
round files by SHA-256 hashes.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_selection_weak_labels import (
    EXPECTED_RAW_ROUNDS,
    LIBRARIES,
    RAW_COLUMNS,
    discover_raw_round_files,
)
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "selection-trajectories-v2"
OUTPUT_FILENAME = "selection_trajectories.npz"
MANIFEST_FILENAME = "manifest.json"
SUMMARY_FILENAME = "run_summary.md"
CHUNK_SIZE = 250000
PAIR_COLUMNS = ("library", "pair_uid", "peptide_design_code", "affibody_design_code")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def _atomic_write_json(path, payload):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with open(str(temporary), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path, value):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with open(str(temporary), "w") as handle:
            handle.write(value)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path, arrays):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with open(str(temporary), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_target_rows(cache_rows_csv, cache_prepare_manifest):
    manifest = _read_json(cache_prepare_manifest)
    _require(
        manifest.get("schema_version") == "mint-weak-feature-cache-v1",
        "cache prepare manifest schema mismatch",
    )
    expected = manifest.get("output", {}).get("rows_csv", {}).get("sha256")
    _require(expected == sha256_file(cache_rows_csv), "cache row hash disagrees with manifest")
    frame = pd.read_csv(
        cache_rows_csv,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=[
            "row_index",
            "source_kind",
            "library",
            "pair_uid",
            "peptide_design_code",
            "affibody_design_code",
            "weak_label",
        ],
    )
    frame = frame.loc[frame["source_kind"].eq("weak")].copy()
    frame["cache_row_index"] = pd.to_numeric(frame.pop("row_index"), errors="raise").astype(np.int64)
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="raise").astype(np.int8)
    _require(set(frame["library"]) == set(LIBRARIES), "weak cache lacks a library")
    _require(set(frame["weak_label"]) == {0, 1}, "weak labels are not binary")
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate weak pair UID")
    _require(
        not bool(frame[list(PAIR_COLUMNS[:1] + PAIR_COLUMNS[2:])].duplicated().any()),
        "duplicate weak peptide--Affibody code",
    )
    return frame.reset_index(drop=True), manifest


def _scan_round(path, target_index, count_column, frequency_column, presence_column, chunk_size):
    """Scan one raw table and fill target-aligned arrays in place."""
    header = pd.read_csv(
        path,
        sep="\t",
        nrows=0,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(tuple(header.columns) == RAW_COLUMNS, "unexpected raw schema in {}".format(path))
    n_source_rows = 0
    total_count = 0
    total_frequency = 0.0
    matched = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=list(RAW_COLUMNS[:4]),
        dtype={"pep": str, "aff": str, "count": np.int64, "frequency": np.float64},
        keep_default_na=False,
        na_filter=False,
        chunksize=int(chunk_size),
    ):
        _require(tuple(chunk.columns) == RAW_COLUMNS[:4], "unexpected raw schema in {}".format(path))
        _require(bool(chunk["count"].gt(0).all()), "nonpositive raw count in {}".format(path))
        _require(
            bool(np.isfinite(chunk["frequency"].to_numpy(dtype=np.float64)).all()),
            "non-finite raw frequency in {}".format(path),
        )
        _require(bool(chunk["frequency"].gt(0).all()), "nonpositive raw frequency in {}".format(path))
        n_source_rows += int(len(chunk))
        total_count += int(chunk["count"].sum())
        total_frequency += float(chunk["frequency"].sum())
        keys = chunk["pep"].str.cat(chunk["aff"], sep="|")
        destination = keys.map(target_index)
        keep = destination.notna().to_numpy()
        if not bool(keep.any()):
            continue
        indices = destination.loc[keep].astype(np.int64).to_numpy()
        _require(
            len(np.unique(indices)) == len(indices),
            "duplicate target key within one raw chunk in {}".format(path),
        )
        _require(not bool(presence_column[indices].any()), "duplicate target key in {}".format(path))
        presence_column[indices] = 1
        count_column[indices] = chunk.loc[keep, "count"].to_numpy(dtype=np.int64)
        frequency_column[indices] = chunk.loc[keep, "frequency"].to_numpy(dtype=np.float64)
        matched += int(len(indices))
    _require(n_source_rows > 0 and total_count > 0, "empty raw round {}".format(path))
    _require(abs(total_frequency - 1.0) < 5e-3, "raw frequencies do not sum near one in {}".format(path))
    return {
        "source_rows": int(n_source_rows),
        "total_count": int(total_count),
        "frequency_sum": float(total_frequency),
        "matched_target_pairs": int(matched),
    }


def scan_trajectories(targets, raw_files, chunk_size=CHUNK_SIZE):
    n_rows = len(targets)
    n_rounds = len(EXPECTED_RAW_ROUNDS)
    count = np.zeros((n_rows, n_rounds), dtype=np.int64)
    frequency = np.zeros((n_rows, n_rounds), dtype=np.float64)
    presence = np.zeros((n_rows, n_rounds), dtype=np.uint8)
    round_stats = {}
    for library in LIBRARIES:
        positions = np.flatnonzero(targets["library"].eq(library).to_numpy())
        key_to_local = {
            "{}|{}".format(targets.iloc[index]["peptide_design_code"], targets.iloc[index]["affibody_design_code"]): int(index)
            for index in positions
        }
        _require(len(key_to_local) == len(positions), "duplicate target code in {}".format(library))
        for round_index in EXPECTED_RAW_ROUNDS:
            stats = _scan_round(
                raw_files[library][round_index],
                key_to_local,
                count[:, round_index],
                frequency[:, round_index],
                presence[:, round_index],
                chunk_size,
            )
            round_stats["{}:R{:03d}".format(library, round_index)] = stats
    _require(np.array_equal(presence, (count > 0).astype(np.uint8)), "count/presence mismatch")
    _require(bool(np.isfinite(frequency).all()), "non-finite frequency")
    _require(bool((frequency >= 0.0).all()), "negative frequency")
    return count, frequency, presence, round_stats


def make_trajectory_features(count, frequency, presence, libraries, round_stats):
    """Return transparent rater inputs derived only from sequencing rounds."""
    count = np.asarray(count, dtype=np.int64)
    frequency = np.asarray(frequency, dtype=np.float64)
    presence = np.asarray(presence, dtype=np.uint8)
    libraries = np.asarray(libraries).astype(str)
    names = []
    blocks = []

    blocks.append(presence.astype(np.float32))
    names.extend("present_r{:03d}".format(index) for index in EXPECTED_RAW_ROUNDS)

    blocks.append(np.log1p(count).astype(np.float32))
    names.extend("log1p_count_r{:03d}".format(index) for index in EXPECTED_RAW_ROUNDS)

    stabilized_log2_frequency = np.zeros_like(frequency, dtype=np.float64)
    for library in LIBRARIES:
        mask = libraries == library
        for round_index in EXPECTED_RAW_ROUNDS:
            depth = float(round_stats["{}:R{:03d}".format(library, round_index)]["total_count"])
            pseudocount = 0.5 / depth
            stabilized_log2_frequency[mask, round_index] = np.log2(
                frequency[mask, round_index] + pseudocount
            )
    blocks.append(stabilized_log2_frequency.astype(np.float32))
    names.extend("log2_frequency_half_read_r{:03d}".format(index) for index in EXPECTED_RAW_ROUNDS)

    enrichment = np.diff(stabilized_log2_frequency, axis=1)
    # When neither round observed a read, the half-read detection limits differ
    # with sequencing depth.  Their subtraction is not biological enrichment;
    # treat that transition as censored zero and retain the two presence flags.
    both_missing = (presence[:, :-1] == 0) & (presence[:, 1:] == 0)
    enrichment[both_missing] = 0.0
    blocks.append(enrichment.astype(np.float32))
    names.extend(
        "censored_log2_frequency_change_r{:03d}_to_r{:03d}".format(index - 1, index)
        for index in EXPECTED_RAW_ROUNDS[1:]
    )

    summaries = np.column_stack(
        [
            presence.sum(axis=1),
            presence[:, 9] * presence[:, 10],
            presence[:, 11:15].sum(axis=1),
            np.abs(stabilized_log2_frequency[:, 10] - stabilized_log2_frequency[:, 9]),
        ]
    ).astype(np.float32)
    blocks.append(summaries)
    names.extend(
        [
            "rounds_observed",
            "observed_in_both_r009_r010",
            "later_rounds_observed_r011_r014",
            "absolute_log2_frequency_difference_r009_r010",
        ]
    )
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    _require(features.shape == (len(count), len(names)), "trajectory feature shape mismatch")
    _require(bool(np.isfinite(features).all()), "non-finite trajectory feature")
    return features, np.asarray(names, dtype="U80")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--cache-rows-csv", required=True, type=Path)
    parser.add_argument("--cache-prepare-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    _require(int(args.chunk_size) > 0, "chunk size must be positive")
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.raw_root.is_dir(), "raw root does not exist")
    _require(args.cache_rows_csv.is_file(), "cache row table does not exist")
    _require(args.cache_prepare_manifest.is_file(), "cache prepare manifest does not exist")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    try:
        targets, cache_manifest = _load_target_rows(
            args.cache_rows_csv, args.cache_prepare_manifest
        )
        raw_files = discover_raw_round_files(args.raw_root)
        raw_hashes = {
            "{}:R{:03d}".format(library, round_index): {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for library, files in raw_files.items()
            for round_index, path in sorted(files.items())
        }
        weak_manifest_declared = cache_manifest.get("sources", {}).get("weak_manifest", {})
        weak_manifest_path = Path(weak_manifest_declared.get("path", ""))
        _require(weak_manifest_path.is_file(), "cache manifest does not identify its weak-label manifest")
        _require(
            sha256_file(weak_manifest_path) == weak_manifest_declared.get("sha256"),
            "weak-label manifest hash disagrees with cache manifest",
        )
        weak_manifest = _read_json(weak_manifest_path)
        declared_raw = weak_manifest.get("sources", {}).get("raw_rounds", {})
        _require(set(declared_raw) == set(raw_hashes), "weak-label raw-round lineage is incomplete")
        for key in sorted(raw_hashes):
            _require(
                declared_raw[key].get("sha256") == raw_hashes[key]["sha256"],
                "trajectory raw round does not match weak-label lineage: {}".format(key),
            )
        count, frequency, presence, round_stats = scan_trajectories(
            targets, raw_files, chunk_size=args.chunk_size
        )
        features, feature_names = make_trajectory_features(
            count,
            frequency,
            presence,
            targets["library"].to_numpy(str),
            round_stats,
        )
        arrays = {
            "cache_row_index": targets["cache_row_index"].to_numpy(dtype=np.int64),
            "pair_uid": targets["pair_uid"].to_numpy(dtype="U64"),
            "library": targets["library"].to_numpy(dtype="U4"),
            "weak_label": targets["weak_label"].to_numpy(dtype=np.int8),
            "round_index": np.asarray(EXPECTED_RAW_ROUNDS, dtype=np.int8),
            "count": count,
            "frequency": frequency,
            "presence": presence,
            "trajectory_features": features,
            "trajectory_feature_names": feature_names,
        }
        output_path = output_dir / OUTPUT_FILENAME
        _atomic_write_npz(output_path, arrays)
        total_raw_rows = int(sum(item["source_rows"] for item in round_stats.values()))
        summary = (
            "# Selection trajectory extraction\n\n"
            "- Distinct weak-label pairs: {:,}\n"
            "- Raw round-level rows scanned: {:,}\n"
            "- Libraries: LibA and LibB, each with R000--R014\n"
            "- Stored evidence: observed read count, reported frequency, and presence in each round\n"
            "- Important: zero means no read was observed in that round, not proven biological absence.\n"
        ).format(len(targets), total_raw_rows)
        summary_path = output_dir / SUMMARY_FILENAME
        _atomic_write_text(summary_path, summary)
        script_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_unix": int(time.time()),
            "elapsed_seconds": float(time.time() - started),
            "host": platform.node(),
            "python": platform.python_version(),
            "contract": {
                "unit": "one distinct cached weak-label peptide--Affibody pair",
                "rounds": list(EXPECTED_RAW_ROUNDS),
                "missing_round_row_semantics": "zero observed reads; biological absence is not asserted",
                "trajectory_feature_names": feature_names.tolist(),
            },
            "counts": {
                "weak_pairs": int(len(targets)),
                "LibA_pairs": int(targets["library"].eq("LibA").sum()),
                "LibB_pairs": int(targets["library"].eq("LibB").sum()),
                "raw_round_level_rows": total_raw_rows,
            },
            "input": {
                "cache_rows_csv": {
                    "path": str(args.cache_rows_csv.resolve()),
                    "sha256": sha256_file(args.cache_rows_csv),
                },
                "cache_prepare_manifest": {
                    "path": str(args.cache_prepare_manifest.resolve()),
                    "sha256": sha256_file(args.cache_prepare_manifest),
                    "declared_cache_schema": cache_manifest.get("schema_version"),
                },
                "weak_label_manifest": {
                    "path": str(weak_manifest_path.resolve()),
                    "sha256": sha256_file(weak_manifest_path),
                },
                "raw_rounds": raw_hashes,
            },
            "round_statistics": round_stats,
            "source": {
                "path": str(script_path),
                "sha256": sha256_file(script_path),
            },
            "output": {
                "npz": {
                    "path": str(output_path.resolve()),
                    "sha256": sha256_file(output_path),
                    "mode": _private_mode(output_path),
                },
                "summary": {
                    "path": str(summary_path.resolve()),
                    "sha256": sha256_file(summary_path),
                    "mode": _private_mode(summary_path),
                },
                "directory_mode": _private_mode(output_dir),
            },
        }
        _atomic_write_json(output_dir / MANIFEST_FILENAME, manifest)
        _require(_private_mode(output_dir) == "0700", "output directory is not private")
        for path in output_dir.iterdir():
            _require(_private_mode(path) == "0600", "output file is not private: {}".format(path))
        return manifest
    except Exception:
        # Keep a failed directory for forensic inspection only if it contains data.
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        raise


def main(argv=None):
    manifest = run(parse_args(argv))
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
