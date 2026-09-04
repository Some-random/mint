#!/usr/bin/env python3
"""Score LibA candidates with a serialized, parity-checked additive head.

Before opening any candidate partition, the command reconstructs scores for
all 108 known LibA pairs and compares them with the canonical saved
``weak_site_logistic_probability`` values.  The known-pair and reference CSVs
are read with explicit outcome-free column allow-lists.  Candidate output is
published only if that replay passes the fixed tolerance.
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
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
LIBRARY = "LibA"
HEAD_SCHEMA_VERSION = "liba-additive-deployment-head-v1"
UNIVERSE_SCHEMA_VERSION = "liba-existing-target-selection-missed-candidates-v1"
OUTPUT_SCHEMA_VERSION = "liba-additive-candidate-scores-v1"
MODEL_ID = "additive_6site"
EXPECTED_KNOWN_ROWS = 108
EXPECTED_KNOWN_PEPTIDES = 9
EXPECTED_KNOWN_AFFIBODIES_PER_PEPTIDE = 12
PARITY_SCORE_COLUMN = "weak_site_logistic_probability"
PARITY_TOLERANCE = 1e-12
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
POSITION_NAMES = (
    "peptide_position_4",
    "peptide_position_5",
    "affibody_displayed_13_crystal_15",
    "affibody_displayed_17_crystal_19",
    "affibody_displayed_27_crystal_29",
    "affibody_displayed_31_crystal_33",
)
KNOWN_PANEL_COLUMNS = (
    "pair_uid",
    "library",
    "peptide_design_code",
    "affibody_design_code",
)
PARITY_COLUMNS = ("pair_uid", "library", PARITY_SCORE_COLUMN)
FORBIDDEN_OUTCOME_TOKENS = ("retention", "binder", "target")
UNIVERSE_SCORE_INPUT_COLUMNS = (
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)
STANDARD_SCORE_COLUMNS = (
    "model_id",
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "model_score",
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
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


def _membership_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"expected JSON object in {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _validate_private_output(path: Path) -> Path:
    output = path.resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    return output


def _validate_allow_list(columns: tuple[str, ...], role: str) -> None:
    for column in columns:
        lowered = column.lower()
        _require(
            not any(token in lowered for token in FORBIDDEN_OUTCOME_TOKENS),
            f"{role} allow-list contains forbidden outcome-like column {column}",
        )


def load_head(path: Path) -> dict[str, Any]:
    _require(path.is_file(), "serialized LibA additive head does not exist")
    required = {
        "schema_version",
        "feature_names",
        "residue_alphabet",
        "weight",
        "bias",
        "selected_C",
        "training_membership_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        _require(not missing, f"LibA additive head lacks arrays: {sorted(missing)}")
        schema = str(np.asarray(archive["schema_version"]).reshape(-1)[0])
        feature_names = tuple(np.asarray(archive["feature_names"]).astype(str).tolist())
        residue_alphabet = tuple(
            np.asarray(archive["residue_alphabet"]).astype(str).tolist()
        )
        weight = np.asarray(archive["weight"], dtype=np.float64)
        bias_values = np.asarray(archive["bias"], dtype=np.float64).reshape(-1)
        c_values = np.asarray(archive["selected_C"], dtype=np.float64).reshape(-1)
        membership_values = np.asarray(
            archive["training_membership_sha256"]
        ).astype(str).reshape(-1)

    _require(schema == HEAD_SCHEMA_VERSION, "LibA additive-head schema changed")
    _require(feature_names == POSITION_NAMES, "LibA additive feature order changed")
    _require(residue_alphabet == AA_ALPHABET, "LibA additive residue alphabet changed")
    _require(weight.shape == (len(POSITION_NAMES), len(AA_ALPHABET)), "bad additive weight shape")
    _require(len(bias_values) == 1 and math.isfinite(float(bias_values[0])), "bad additive bias")
    _require(len(c_values) == 1 and math.isfinite(float(c_values[0])) and c_values[0] > 0,
             "bad selected C")
    _require(len(membership_values) == 1, "bad training-membership receipt")
    membership = str(membership_values[0])
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", membership)), "bad training-membership SHA-256")
    _require(bool(np.isfinite(weight).all()), "non-finite additive weight")
    return {
        "feature_names": feature_names,
        "residue_alphabet": residue_alphabet,
        "weight": weight,
        "bias": float(bias_values[0]),
        "selected_C": float(c_values[0]),
        "training_membership_sha256": membership,
    }


def _codes(frame: pd.DataFrame) -> np.ndarray:
    peptide = frame["peptide_design_code"].astype(str)
    affibody = frame["affibody_design_code"].astype(str)
    _require(peptide.str.len().eq(2).all(), "bad LibA peptide code length")
    _require(affibody.str.len().eq(4).all(), "bad LibA Affibody code length")
    code = peptide + affibody
    _require(code.str.len().eq(len(POSITION_NAMES)).all(), "bad LibA designed code length")
    _require(
        code.map(lambda value: set(value).issubset(set(AA_ALPHABET))).all(),
        "noncanonical residue in LibA designed code",
    )
    return np.asarray([list(value) for value in code], dtype=str)


def score_codes(frame: pd.DataFrame, head: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    code = _codes(frame)
    alphabet_index = {value: index for index, value in enumerate(head["residue_alphabet"])}
    logit = np.full(len(frame), head["bias"], dtype=np.float64)
    for position in range(len(POSITION_NAMES)):
        indices = np.asarray([alphabet_index.get(value, -1) for value in code[:, position]])
        _require(bool((indices >= 0).all()), f"unknown amino acid at designed position {position}")
        logit += head["weight"][position, indices]
    score = np.empty_like(logit)
    positive = logit >= 0
    score[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_value = np.exp(logit[~positive])
    score[~positive] = exp_value / (1.0 + exp_value)
    _require(bool(np.isfinite(logit).all() and np.isfinite(score).all()), "non-finite additive score")
    return logit, score


def verify_known_pair_parity(
    known_panel_path: Path,
    parity_predictions_path: Path,
    head: dict[str, Any],
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Replay all known pairs while deliberately omitting outcome columns."""

    _require(known_panel_path.is_file(), "known-pair panel does not exist")
    _require(parity_predictions_path.is_file(), "parity prediction table does not exist")
    _require(tolerance >= 0 and math.isfinite(tolerance), "invalid parity tolerance")
    _validate_allow_list(KNOWN_PANEL_COLUMNS, "known-panel")
    _validate_allow_list(PARITY_COLUMNS, "parity")
    panel = pd.read_csv(
        known_panel_path,
        usecols=list(KNOWN_PANEL_COLUMNS),
        dtype={column: str for column in KNOWN_PANEL_COLUMNS},
        keep_default_na=False,
        na_filter=False,
    )
    panel = panel.loc[panel["library"].eq(LIBRARY)].copy()
    _require(len(panel) == EXPECTED_KNOWN_ROWS, "known LibA panel must contain 108 pairs")
    _require(not panel["pair_uid"].duplicated().any(), "duplicate known LibA pair UID")
    _require(panel["peptide_design_code"].nunique() == EXPECTED_KNOWN_PEPTIDES,
             "known LibA peptide count changed")
    _require(
        panel.groupby("peptide_design_code")["affibody_design_code"]
        .nunique()
        .eq(EXPECTED_KNOWN_AFFIBODIES_PER_PEPTIDE)
        .all(),
        "known LibA panel is not complete 9x12 membership",
    )

    reference = pd.read_csv(
        parity_predictions_path,
        usecols=list(PARITY_COLUMNS),
        dtype={"pair_uid": str, "library": str},
        keep_default_na=False,
        na_filter=False,
    )
    reference = reference.loc[reference["library"].eq(LIBRARY)].copy()
    reference[PARITY_SCORE_COLUMN] = pd.to_numeric(
        reference[PARITY_SCORE_COLUMN], errors="raise"
    ).astype(np.float64)
    _require(len(reference) == EXPECTED_KNOWN_ROWS, "parity table must contain 108 LibA scores")
    _require(not reference["pair_uid"].duplicated().any(), "duplicate parity pair UID")
    _require(
        bool(np.isfinite(reference[PARITY_SCORE_COLUMN]).all()),
        "non-finite reference probability",
    )

    joined = panel.merge(
        reference[["pair_uid", PARITY_SCORE_COLUMN]],
        on="pair_uid",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    _require(len(joined) == EXPECTED_KNOWN_ROWS, "known-pair parity membership changed")
    _require(joined["_merge"].eq("both").all(), "known panel and parity IDs differ")
    _, replay = score_codes(joined, head)
    expected = joined[PARITY_SCORE_COLUMN].to_numpy(dtype=np.float64)
    delta = np.abs(replay - expected)
    maximum = float(delta.max())
    _require(maximum <= tolerance, f"LibA additive 108-row parity failed: {maximum} > {tolerance}")
    return {
        "status": "passed",
        "rows": int(len(joined)),
        "maximum_absolute_probability_difference": maximum,
        "mean_absolute_probability_difference": float(delta.mean()),
        "tolerance": float(tolerance),
        "known_pair_membership_sha256": _membership_sha256(joined["pair_uid"]),
        "known_panel": {
            "path": str(known_panel_path.resolve()),
            "sha256": sha256_file(known_panel_path),
            "columns_read": list(KNOWN_PANEL_COLUMNS),
        },
        "reference_predictions": {
            "path": str(parity_predictions_path.resolve()),
            "sha256": sha256_file(parity_predictions_path),
            "columns_read": list(PARITY_COLUMNS),
            "score_column": PARITY_SCORE_COLUMN,
        },
        "retention_outcome_columns_read": [],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--head-npz", type=Path, required=True)
    parser.add_argument("--known-panel", type=Path, required=True)
    parser.add_argument("--parity-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    universe = args.universe_dir.resolve()
    output = _validate_private_output(args.output_dir)
    _require(universe.is_dir(), "candidate universe does not exist")
    universe_manifest_path = universe / "manifest.json"
    partitions_path = universe / "candidate_partitions.csv"
    _require(universe_manifest_path.is_file(), "candidate universe has no completion manifest")
    _require(partitions_path.is_file(), "candidate universe has no partition table")

    head = load_head(args.head_npz)
    parity = verify_known_pair_parity(
        args.known_panel,
        args.parity_predictions,
        head,
        tolerance=PARITY_TOLERANCE,
    )
    # Nothing is written before the complete 108-row parity gate above passes.

    universe_manifest = _read_json(universe_manifest_path)
    _require(
        universe_manifest.get("schema_version") == UNIVERSE_SCHEMA_VERSION,
        "candidate-universe schema changed",
    )
    _require(
        universe_manifest.get("input_data_contract", {}).get(
            "retention_outcome_columns_read"
        )
        == [],
        "candidate universe does not attest outcome-blind construction",
    )
    expected_partition_receipt = universe_manifest.get("outputs", {}).get(
        "candidate_partitions_csv", {}
    )
    _require(
        expected_partition_receipt.get("sha256") == sha256_file(partitions_path),
        "candidate partition table differs from immutable manifest",
    )

    partitions = pd.read_csv(partitions_path, keep_default_na=False, na_filter=False)
    required_partition_columns = {
        "peptide_design_code",
        "candidate_rows",
        "partition",
        "partition_sha256",
    }
    _require(
        required_partition_columns.issubset(partitions.columns),
        "candidate partition table lacks required columns",
    )
    _require(len(partitions) == EXPECTED_KNOWN_PEPTIDES, "expected nine candidate partitions")
    _require(not partitions["peptide_design_code"].duplicated().any(), "duplicate peptide partition")
    expected_rows = int(universe_manifest["outputs"]["candidate_rows"])
    _require(
        int(pd.to_numeric(partitions["candidate_rows"], errors="raise").sum())
        == expected_rows,
        "candidate rows differ from immutable manifest",
    )

    source_hashes = {
        "head_npz": sha256_file(args.head_npz),
        "known_panel": sha256_file(args.known_panel),
        "parity_predictions": sha256_file(args.parity_predictions),
        "universe_manifest": sha256_file(universe_manifest_path),
        "candidate_partitions": sha256_file(partitions_path),
    }
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    started = time.time()
    records: list[dict[str, Any]] = []
    seen_pair_uids: set[str] = set()
    combined_path = output / "candidate_scores.parquet"
    combined_writer: pq.ParquetWriter | None = None
    try:
        for row in partitions.sort_values("peptide_design_code").itertuples(index=False):
            source = (universe / str(row.partition)).resolve()
            try:
                source.relative_to(universe)
            except ValueError as error:
                raise ValueError("candidate partition escapes universe directory") from error
            _require(source.is_file(), f"missing candidate partition {source}")
            source_sha256 = sha256_file(source)
            _require(source_sha256 == str(row.partition_sha256), "candidate partition hash changed")
            frame = pd.read_parquet(source, columns=list(UNIVERSE_SCORE_INPUT_COLUMNS))
            peptide = str(row.peptide_design_code)
            _require(len(frame) == int(row.candidate_rows), "candidate partition row count changed")
            _require(frame["peptide_design_code"].astype(str).eq(peptide).all(), "mixed peptide partition")
            _require(not frame["pair_uid"].duplicated().any(), "duplicate pair UID within partition")
            _require(
                not frame["pair_uid"].astype(str).isin(seen_pair_uids).any(),
                "duplicate pair UID across candidate partitions",
            )
            seen_pair_uids.update(frame["pair_uid"].astype(str))
            _require(
                frame["peptide_9mer_sequence"].astype(str).str.len().eq(9).all(),
                "candidate peptide 9-mer length changed",
            )
            _require(
                frame["provider_displayed_58aa_affibody_sequence"]
                .astype(str)
                .str.len()
                .eq(58)
                .all(),
                "candidate displayed Affibody length changed",
            )
            _require(
                frame["provider_displayed_58aa_affibody_sequence"].astype(str).eq(
                    frame["model_input_affibody_sequence"].astype(str)
                ).all(),
                "displayed and model-input LibA Affibody sequences differ",
            )
            _require(
                all(
                    str(assay).endswith(str(peptide_9mer))
                    for assay, peptide_9mer in zip(
                        frame["model_input_smart_hla_linker_peptide_sequence"],
                        frame["peptide_9mer_sequence"],
                    )
                ),
                "assay-side model input does not end in its peptide 9-mer",
            )
            for column in (
                "observed_in_any_raw_round",
                "observed_in_r009_or_r010",
                "affibody_identity_seen_in_strict_training",
                "high_confidence_weak_negative",
            ):
                _require(
                    pd.api.types.is_bool_dtype(frame[column]) and frame[column].notna().all(),
                    f"candidate {column} must be complete Boolean data",
                )

            logit, score = score_codes(frame, head)
            result = frame.copy()
            result.insert(0, "model_id", MODEL_ID)
            result["model_score"] = score
            result["additive_6site_logit"] = logit
            result["additive_6site_score"] = score
            result = result[list(STANDARD_SCORE_COLUMNS) + [
                "additive_6site_logit",
                "additive_6site_score",
            ]]
            _require(list(result.columns[: len(STANDARD_SCORE_COLUMNS)]) == list(STANDARD_SCORE_COLUMNS),
                     "standard candidate-score schema changed")
            destination = output / f"peptide_{peptide}.parquet"
            result.to_parquet(destination, index=False, engine="pyarrow", compression="zstd")
            os.chmod(destination, 0o600)

            arrow_table = pa.Table.from_pandas(result, preserve_index=False)
            if combined_writer is None:
                combined_writer = pq.ParquetWriter(
                    combined_path,
                    arrow_table.schema,
                    compression="zstd",
                )
            else:
                _require(
                    arrow_table.schema.equals(combined_writer.schema),
                    "candidate output schema differs across peptide partitions",
                )
            combined_writer.write_table(arrow_table)

            _require(sha256_file(source) == source_sha256, "candidate partition changed during scoring")
            records.append(
                {
                    "peptide_design_code": peptide,
                    "rows": int(len(result)),
                    "source_path": str(source),
                    "source_sha256": source_sha256,
                    "output_path": destination.name,
                    "output_sha256": sha256_file(destination),
                    "output_bytes": int(destination.stat().st_size),
                }
            )
    finally:
        if combined_writer is not None:
            combined_writer.close()

    _require(combined_path.is_file(), "combined candidate-score table was not written")
    os.chmod(combined_path, 0o600)

    _require(len(seen_pair_uids) == expected_rows, "scored candidate membership changed")
    for name, path in (
        ("head_npz", args.head_npz),
        ("known_panel", args.known_panel),
        ("parity_predictions", args.parity_predictions),
        ("universe_manifest", universe_manifest_path),
        ("candidate_partitions", partitions_path),
    ):
        _require(sha256_file(path) == source_hashes[name], f"source {name} changed during scoring")

    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "library": LIBRARY,
        "model_id": MODEL_ID,
        "rows": int(sum(record["rows"] for record in records)),
        "score_definition": {
            "logit_column": "additive_6site_logit",
            "score_column": "additive_6site_score",
            "formula": "sigmoid(bias + sum of six position-specific amino-acid weights)",
            "score_direction": "higher_is_more_likely_selection_derived_binder",
        },
        "standard_score_columns": list(STANDARD_SCORE_COLUMNS),
        "combined_candidate_scores": {
            "path": combined_path.name,
            "rows": int(len(seen_pair_uids)),
            "sha256": sha256_file(combined_path),
            "bytes": int(combined_path.stat().st_size),
        },
        "head": {
            "path": str(args.head_npz.resolve()),
            "sha256": source_hashes["head_npz"],
            "schema_version": HEAD_SCHEMA_VERSION,
            "feature_names": list(head["feature_names"]),
            "residue_alphabet": list(head["residue_alphabet"]),
            "selected_C": head["selected_C"],
            "training_membership_sha256": head["training_membership_sha256"],
        },
        "known_108_pair_parity": parity,
        "candidate_universe": {
            "path": str(universe),
            "manifest_sha256": source_hashes["universe_manifest"],
            "candidate_partitions_sha256": source_hashes["candidate_partitions"],
            "rows": expected_rows,
        },
        "partitions": records,
        "retention_or_binder_outcomes_read": False,
        "manifest_written_last": True,
    }
    _write_json_exclusive(output / "manifest.json", manifest)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
