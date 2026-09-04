#!/usr/bin/env python3
"""Validate and merge the nine exhaustive LibA MINT score partitions.

The GPU scorer writes one Parquet file and one completion receipt per peptide.
This command verifies those receipts against the immutable 447,731-row LibA
candidate universe, checks every identity/sequence/score row, and atomically
publishes one standardized Parquet file for a single MINT layer.

No weak labels, binder labels, or retention outcomes are accepted or read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
LIBRARY = "LibA"
EXPECTED_ROWS = 447_731
EXPECTED_PEPTIDES = ("AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL")
MODEL_BY_LAYER = {9: "mint_l9", 33: "mint_l33"}
SOURCE_SCHEMA_BY_LAYER = {
    layer: f"mint-layer{layer}-streaming-candidate-scores-v1" for layer in MODEL_BY_LAYER
}
OUTPUT_SCHEMA_VERSION = "liba-mint-candidate-scores-merged-v1"
UNIVERSE_SCHEMA_VERSION = "liba-existing-target-selection-missed-candidates-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

IDENTITY_COLUMNS = (
    "pair_uid",
    "library",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "peptide_uid",
    "affibody_uid",
    "chain1_sha256",
    "chain2_sha256",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
)
FLAG_COLUMNS = (
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)
STANDARD_COLUMNS = (
    "model_id",
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "model_score",
    *FLAG_COLUMNS,
)
FORBIDDEN_OUTCOME_PARTS = (
    "retention",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def ordered_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(Path(path).is_file(), f"missing JSON file: {path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    display = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": display,
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _private_new_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must remain below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not resolved.exists(), f"output exists; refusing overwrite: {resolved}")
    return resolved


def _outcome_columns(columns: Iterable[object]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_OUTCOME_PARTS)
    )


def _source_paths(score_dir: Path, peptide: str) -> tuple[Path, Path]:
    candidates = (
        score_dir / "candidates_by_peptide" / f"peptide_{peptide}.parquet",
        score_dir / f"peptide_{peptide}.parquet",
    )
    existing = [path for path in candidates if path.is_file()]
    _require(len(existing) == 1, f"expected exactly one MINT score partition for {peptide}")
    score_path = existing[0]
    manifest_path = score_path.with_suffix(score_path.suffix + ".manifest.json")
    _require(manifest_path.is_file(), f"missing MINT completion receipt for {peptide}")
    return score_path, manifest_path


def validate_source_manifest(
    manifest_path: Path,
    score_path: Path,
    universe_path: Path,
    peptide: str,
    layer: int,
    expected_rows: int,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    model_id = MODEL_BY_LAYER[layer]
    _require(
        manifest.get("schema_version") == SOURCE_SCHEMA_BY_LAYER[layer],
        f"wrong MINT source schema for {peptide}",
    )
    _require(int(manifest.get("rows", -1)) == expected_rows, f"source row count changed for {peptide}")
    _require(str(manifest.get("peptide_design_code")) == peptide, f"source peptide changed for {peptide}")
    model = manifest.get("model", {})
    _require(model.get("library") == LIBRARY, f"source library changed for {peptide}")
    _require(int(model.get("layer", -1)) == layer, f"source MINT layer changed for {peptide}")
    _require(model.get("model_id") == model_id, f"source model ID changed for {peptide}")
    output = manifest.get("output", {})
    _require(output.get("sha256") == sha256_file(score_path), f"source score hash changed for {peptide}")
    _require(int(output.get("bytes", -1)) == score_path.stat().st_size, f"source score size changed for {peptide}")
    _require(Path(str(output.get("path", ""))).resolve() == score_path.resolve(), f"source score path changed for {peptide}")
    candidate = manifest.get("inputs", {}).get("candidate_parquet", {})
    _require(candidate.get("sha256") == sha256_file(universe_path), f"source universe hash changed for {peptide}")
    _require(Path(str(candidate.get("path", ""))).resolve() == universe_path.resolve(), f"source universe path changed for {peptide}")
    parity = manifest.get("parity", {})
    _require(parity.get("ordinary_vs_early_features_bitwise_equal") is True,
             f"source lacks exact early-stop parity for {peptide}")
    difference = parity.get("head_score_vs_reference_max_abs")
    tolerance = parity.get("score_atol")
    _require(
        isinstance(difference, (int, float))
        and isinstance(tolerance, (int, float))
        and math.isfinite(float(difference))
        and math.isfinite(float(tolerance))
        and float(difference) <= float(tolerance),
        f"source lacks passing saved-head parity for {peptide}",
    )
    receipt_hash = sha256_file(manifest_path)
    _require(bool(HEX64.fullmatch(receipt_hash)), "internal manifest hash error")
    return {
        "manifest": _record(manifest_path),
        "score": _record(score_path),
        "parity_receipt": manifest.get("inputs", {}).get("parity_receipt", {}),
    }


def validate_and_standardize_partition(
    universe_path: Path,
    score_path: Path,
    manifest_path: Path,
    peptide: str,
    layer: int,
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate one peptide partition and return the public handoff schema."""

    model_id = MODEL_BY_LAYER[layer]
    logit_column = f"mint_layer{layer}_logit"
    source = validate_source_manifest(
        manifest_path, score_path, universe_path, peptide, layer, expected_rows
    )
    universe_columns = {
        "pair_uid": "pair_uid",
        "library": "library",
        "peptide_design_code": "peptide_design_code",
        "peptide_9mer_sequence": "peptide_9mer_sequence",
        "affibody_design_code": "affibody_design_code",
        "peptide_uid": "peptide_uid",
        "affibody_uid": "affibody_uid",
        "chain1_sha256": "chain1_sha256",
        "chain2_sha256": "chain2_sha256",
        "provider_displayed_58aa_affibody_sequence": "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence": "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence": "model_input_smart_hla_linker_peptide_sequence",
        **{name: name for name in FLAG_COLUMNS},
    }
    universe = pd.read_parquet(universe_path, columns=list(universe_columns))
    score_schema = pq.ParquetFile(score_path).schema_arrow.names
    _require(not _outcome_columns(score_schema), f"MINT score partition has outcome columns for {peptide}")
    required_score = {
        "input_row_index",
        "model_id",
        *IDENTITY_COLUMNS,
        "model_score",
        logit_column,
        *FLAG_COLUMNS,
    }
    _require(required_score.issubset(score_schema), f"MINT score columns changed for {peptide}")
    score = pd.read_parquet(score_path, columns=list(required_score))
    _require(len(universe) == expected_rows == len(score), f"partition row count changed for {peptide}")
    expected_index = np.arange(expected_rows, dtype=np.uint64)
    observed_index = pd.to_numeric(score["input_row_index"], errors="raise").to_numpy(dtype=np.uint64)
    _require(np.array_equal(observed_index, expected_index), f"source row order/index changed for {peptide}")
    _require(score["model_id"].astype(str).eq(model_id).all(), f"mixed MINT model IDs for {peptide}")
    _require(score["library"].astype(str).eq(LIBRARY).all(), f"mixed library for {peptide}")
    _require(score["peptide_design_code"].astype(str).eq(peptide).all(), f"mixed peptide for {peptide}")
    _require(not score["pair_uid"].astype(str).duplicated().any(), f"duplicate pair UID for {peptide}")

    for column in (*IDENTITY_COLUMNS, *FLAG_COLUMNS):
        if column == "library":
            expected = universe["library"]
        else:
            expected = universe[column]
        observed = score[column]
        _require(
            observed.reset_index(drop=True).equals(expected.reset_index(drop=True)),
            f"MINT score/universe {column} mismatch for {peptide}",
        )
    for column in FLAG_COLUMNS:
        _require(pd.api.types.is_bool_dtype(score[column]) and score[column].notna().all(),
                 f"incomplete Boolean {column} for {peptide}")

    probability = pd.to_numeric(score["model_score"], errors="raise").to_numpy(dtype=np.float64)
    logit = pd.to_numeric(score[logit_column], errors="raise").to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(probability).all() and np.isfinite(logit).all()),
             f"non-finite MINT score for {peptide}")
    _require(bool(((probability >= 0.0) & (probability <= 1.0)).all()),
             f"MINT probability outside [0,1] for {peptide}")
    replay = np.empty_like(logit)
    positive = logit >= 0.0
    replay[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponent = np.exp(logit[~positive])
    replay[~positive] = exponent / (1.0 + exponent)
    maximum = float(np.max(np.abs(replay - probability)))
    _require(maximum <= 2e-12, f"MINT logit/probability parity failed for {peptide}: {maximum}")

    output = score[list(STANDARD_COLUMNS)].copy()
    output[logit_column] = logit
    output["library"] = LIBRARY
    output["peptide_uid"] = score["peptide_uid"].astype(str)
    output["affibody_uid"] = score["affibody_uid"].astype(str)
    output["chain1_sha256"] = score["chain1_sha256"].astype(str)
    output["chain2_sha256"] = score["chain2_sha256"].astype(str)
    output = output[
        list(STANDARD_COLUMNS)
        + [logit_column, "library", "peptide_uid", "affibody_uid", "chain1_sha256", "chain2_sha256"]
    ]
    return output, {
        "peptide_design_code": peptide,
        "rows": int(len(output)),
        "pair_uid_membership_sha256": membership_sha256(output["pair_uid"]),
        "pair_uid_order_sha256": ordered_sha256(output["pair_uid"]),
        "maximum_absolute_sigmoid_parity_difference": maximum,
        "source": source,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    layer = int(args.layer)
    _require(layer in MODEL_BY_LAYER, "layer must be 9 or 33")
    model_id = MODEL_BY_LAYER[layer]
    universe_dir = Path(args.universe_dir).resolve()
    score_dir = Path(args.score_dir).resolve()
    output_dir = _private_new_directory(args.output_dir)
    _require(universe_dir.is_dir(), "candidate-universe directory is missing")
    _require(score_dir.is_dir(), "MINT score directory is missing")
    universe_manifest_path = universe_dir / "manifest.json"
    partitions_path = universe_dir / "candidate_partitions.csv"
    universe_manifest = _read_json(universe_manifest_path)
    _require(universe_manifest.get("schema_version") == UNIVERSE_SCHEMA_VERSION,
             "candidate-universe schema changed")
    _require(
        universe_manifest.get("input_data_contract", {}).get("retention_outcome_columns_read") == [],
        "candidate universe does not attest outcome-blind construction",
    )
    _require(
        universe_manifest.get("outputs", {}).get("candidate_partitions_csv", {}).get("sha256")
        == sha256_file(partitions_path),
        "candidate partition table differs from universe manifest",
    )
    partitions = pd.read_csv(partitions_path, keep_default_na=False, na_filter=False)
    required = {"peptide_design_code", "candidate_rows", "partition", "partition_sha256"}
    _require(required.issubset(partitions.columns), "candidate partition table lacks required columns")
    _require(tuple(sorted(partitions["peptide_design_code"].astype(str))) == EXPECTED_PEPTIDES,
             "LibA candidate peptide set changed")
    _require(int(partitions["candidate_rows"].sum()) == EXPECTED_ROWS,
             "LibA candidate row count changed")

    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    _require(not staging.exists(), f"staging directory exists: {staging}")
    staging.mkdir(mode=0o700)
    combined_path = staging / "candidate_scores.parquet"
    partition_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    writer: pq.ParquetWriter | None = None
    try:
        for row in partitions.sort_values("peptide_design_code", kind="stable").itertuples(index=False):
            peptide = str(row.peptide_design_code)
            universe_path = (universe_dir / str(row.partition)).resolve()
            _require(universe_dir in universe_path.parents, "universe partition escapes its directory")
            _require(universe_path.is_file(), f"missing universe partition for {peptide}")
            _require(sha256_file(universe_path) == str(row.partition_sha256),
                     f"universe partition hash changed for {peptide}")
            score_path, receipt_path = _source_paths(score_dir, peptide)
            output, record = validate_and_standardize_partition(
                universe_path,
                score_path,
                receipt_path,
                peptide,
                layer,
                int(row.candidate_rows),
            )
            ids = output["pair_uid"].astype(str)
            _require(not ids.isin(seen).any(), f"duplicate pair UID across peptide partitions at {peptide}")
            seen.update(ids)
            table = pa.Table.from_pandas(output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(combined_path, table.schema, compression="zstd")
            else:
                _require(table.schema.equals(writer.schema), f"output schema changed at {peptide}")
            writer.write_table(table, row_group_size=65_536)
            partition_records.append(record)
            print(json.dumps({"model_id": model_id, "peptide": peptide, "rows": len(output)}), flush=True)
        _require(writer is not None, "no MINT score partition was merged")
        writer.close()
        writer = None
        os.chmod(combined_path, 0o600)
        _require(len(seen) == EXPECTED_ROWS, "merged MINT candidate membership is incomplete")
        combined = pq.ParquetFile(combined_path)
        _require(combined.metadata.num_rows == EXPECTED_ROWS, "combined MINT row count changed")
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": float(time.time() - started),
            "library": LIBRARY,
            "model_id": model_id,
            "mint_layer": layer,
            "rows": EXPECTED_ROWS,
            "peptides": list(EXPECTED_PEPTIDES),
            "pair_uid_membership_sha256": membership_sha256(seen),
            "score_definition": {
                "model_score": f"sigmoid(mint_layer{layer}_logit)",
                "direction": "higher_is_more_likely_selection_derived_binder",
                "semantics": "selection-label score; not a retention percentage",
            },
            "candidate_universe": {
                "path": str(universe_dir),
                "manifest_sha256": sha256_file(universe_manifest_path),
                "candidate_partitions_sha256": sha256_file(partitions_path),
            },
            "source_score_directory": str(score_dir),
            "partitions": partition_records,
            "output": _record(combined_path, relative_to=staging),
            "outcome_access": {
                "selection_labels_read": False,
                "retention_measurements_read": False,
                "binder_outcomes_read": False,
            },
            "validation": {
                "exact_universe_identity_and_order": True,
                "source_manifests_and_hashes_verified": True,
                "saved_head_and_early_stop_parity_required": True,
                "sigmoid_logit_parity": True,
                "manifest_written_last": True,
            },
            "code": _record(Path(__file__).resolve()),
        }
        _write_json(staging / "manifest.json", manifest)
        os.rename(staging, output_dir)
        print(json.dumps({"output": str(output_dir), "model_id": model_id, "rows": EXPECTED_ROWS}))
        return manifest
    except Exception:
        if writer is not None:
            writer.close()
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=sorted(MODEL_BY_LAYER), required=True)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
