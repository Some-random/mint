#!/usr/bin/env python
"""Validate completed LibB MINT layer-5 candidate-score Parquets."""

from __future__ import print_function

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import opaque_id  # noqa: E402
from downstream.AffibodyMHC import score_mint_layer5_candidates as scorer  # noqa: E402


SCHEMA_VERSION = "mint-layer5-candidate-score-validation-v1"
AFFIBODY_CODE_INDICES = (5, 9, 12, 13, 16)
DEFAULT_UNIVERSE = (
    REPO_ROOT
    / "private_data"
    / "prospective"
    / "libb_existing_targets_selection_missed_universe_v1"
)
DEFAULT_SCORES = (
    REPO_ROOT
    / "private_data"
    / "prospective"
    / "libb_all_design_mint_layer5_scores_v1"
)
DEFAULT_HISTORICAL_PARITY = (
    REPO_ROOT
    / "private_data"
    / "prospective"
    / "libb_mint_layer5_candidate_scores_v1"
    / "parity_120.v2_head_path.json"
)
DEFAULT_CURRENT_PARITY = (
    REPO_ROOT
    / "private_data"
    / "prospective"
    / "libb_mint_layer5_candidate_scores_v1"
    / "parity_120.json"
)
OLD_ARTIFACT_DIRECTORY = "libb_common_oof_base_predictions_v2_exact"
CURRENT_ARTIFACT_DIRECTORY = "libb_common_oof_base_predictions_v3_deployable"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _validate_code_mapping(input_values, peptide):
    count = len(input_values["pair_uid"])
    for index in range(count):
        peptide_code = str(input_values["peptide_design_code"][index])
        full_peptide = str(input_values["peptide_full_sequence"][index])
        affibody_code = str(input_values["affibody_design_code"][index])
        chain1 = str(input_values["chain1_smart_hla_linker_peptide_sequence"][index])
        chain2 = str(input_values["chain2_affibody_sequence"][index])
        _require(peptide_code == peptide, "input partition peptide changed")
        _require(len(chain1) == 270 and chain1[-9:] == full_peptide, "peptide chain mapping changed")
        _require(len(full_peptide) == 9 and full_peptide[3:5] == peptide_code, "peptide code mapping changed")
        _require(len(chain2) == 58, "Affibody chain length changed")
        observed_code = "".join(chain2[position] for position in AFFIBODY_CODE_INDICES)
        _require(observed_code == affibody_code, "Affibody code mapping changed")
        _require(
            str(input_values["pair_uid"][index])
            == opaque_id("LibB", peptide_code, affibody_code),
            "pair UID/code mapping changed",
        )


def _validate_partition(
    input_path,
    output_path,
    manifest_path,
    expected_rows,
    expected_input_sha256,
    peptide,
    receipt,
):
    _require(input_path.is_file(), "missing universe partition {}".format(input_path))
    _require(output_path.is_file(), "missing score partition {}".format(output_path))
    _require(manifest_path.is_file(), "missing score manifest {}".format(manifest_path))
    input_file = pq.ParquetFile(str(input_path))
    output_file = pq.ParquetFile(str(output_path))
    _require(input_file.metadata.num_rows == expected_rows, "universe row count changed")
    _require(output_file.metadata.num_rows == expected_rows, "score row count changed")
    _require(output_file.schema_arrow.names == scorer._output_schema().names, "score columns changed")
    _require(
        not any("retention" in name.lower() for name in output_file.schema_arrow.names),
        "score output contains retention data",
    )
    input_columns = list(scorer.INPUT_COLUMNS)
    output_columns = [
        "input_row_index",
        "pair_uid",
        "library",
        "peptide_design_code",
        "peptide_full_sequence",
        "affibody_design_code",
        "peptide_uid",
        "affibody_uid",
        "chain1_sha256",
        "chain2_sha256",
        "mint_layer5_logit",
        "mint_layer5_score",
    ]
    offset = 0
    pair_ids = set()
    pair_digest = hashlib.sha256()
    maximum_sigmoid_error = 0.0
    input_batches = input_file.iter_batches(batch_size=65536, columns=input_columns)
    output_batches = output_file.iter_batches(batch_size=65536, columns=output_columns)
    for input_batch, output_batch in itertools.zip_longest(input_batches, output_batches):
        _require(input_batch is not None and output_batch is not None, "input/output batches differ")
        input_values = input_batch.to_pydict()
        output_values = output_batch.to_pydict()
        count = len(input_values["pair_uid"])
        _require(len(output_values["pair_uid"]) == count, "input/output batch lengths differ")
        _validate_code_mapping(input_values, peptide)
        for name in scorer.OUTPUT_ID_COLUMNS:
            _require(input_values[name] == output_values[name], "{} order/content changed".format(name))
        expected_indices = list(range(offset, offset + count))
        _require(output_values["input_row_index"] == expected_indices, "row index changed")
        batch_ids = [str(value) for value in output_values["pair_uid"]]
        _require(len(set(batch_ids)) == len(batch_ids), "duplicate pair UID within batch")
        _require(not pair_ids.intersection(batch_ids), "duplicate pair UID across batches")
        pair_ids.update(batch_ids)
        pair_digest.update(("\n".join(batch_ids) + "\n").encode("ascii"))
        logits = np.asarray(output_values["mint_layer5_logit"], dtype=np.float64)
        scores = np.asarray(output_values["mint_layer5_score"], dtype=np.float64)
        _require(bool(np.isfinite(logits).all()), "non-finite MINT logit")
        _require(bool(np.isfinite(scores).all()), "non-finite MINT score")
        _require(bool(((scores >= 0.0) & (scores <= 1.0)).all()), "MINT score outside [0,1]")
        error = float(np.max(np.abs(scores - _stable_sigmoid(logits))))
        maximum_sigmoid_error = max(maximum_sigmoid_error, error)
        offset += count
    _require(offset == expected_rows and len(pair_ids) == expected_rows, "partition coverage changed")
    _require(maximum_sigmoid_error <= 2e-15, "score does not equal sigmoid(logit)")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema_version") == scorer.SCHEMA_VERSION, "manifest schema changed")
    _require(manifest.get("rows") == expected_rows, "manifest row count changed")
    _require(manifest.get("peptide_design_code") == peptide, "manifest peptide changed")
    _require(
        manifest.get("row_order_pair_uid_sha256") == pair_digest.hexdigest(),
        "manifest pair-order digest changed",
    )
    input_sha256 = _sha256(input_path)
    output_sha256 = _sha256(output_path)
    _require(input_sha256 == expected_input_sha256, "partition-table input checksum changed")
    _require(
        manifest.get("inputs", {}).get("candidate_parquet", {}).get("sha256") == input_sha256,
        "manifest input checksum changed",
    )
    _require(
        manifest.get("output", {}).get("sha256") == output_sha256,
        "manifest output checksum changed",
    )
    _require(
        manifest.get("inputs", {}).get("head_npz", {}).get("sha256")
        == receipt["inputs"]["head_npz"]["sha256"],
        "manifest head differs from parity receipt",
    )
    _require(
        manifest.get("inputs", {}).get("parity_receipt", {}).get("sha256")
        == receipt["receipt_sha256"],
        "manifest parity receipt checksum changed",
    )
    _require(manifest.get("parity") == receipt["checks"], "manifest parity checks changed")
    _require((os.stat(output_path).st_mode & 0o7777) == 0o600, "score Parquet is not mode 0600")
    _require((os.stat(manifest_path).st_mode & 0o7777) == 0o600, "score manifest is not mode 0600")
    return {
        "peptide_design_code": peptide,
        "rows": expected_rows,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "output_bytes": output_path.stat().st_size,
        "output_row_groups": output_file.metadata.num_row_groups,
        "maximum_sigmoid_error": maximum_sigmoid_error,
        "rows_per_second": float(manifest["runtime"]["rows_per_second_including_load"]),
        "peak_allocated_bytes": int(manifest["runtime"]["peak_allocated_bytes"]),
    }


def run(args):
    started = time.time()
    universe = Path(args.universe_dir).resolve()
    scores = Path(args.scores_dir).resolve()
    receipt_path = Path(args.parity_receipt).resolve()
    current_receipt_path = Path(args.current_parity_receipt).resolve()
    _require(universe.is_dir(), "universe directory is missing")
    _require(scores.is_dir(), "score directory is missing")
    _require(receipt_path.is_file(), "parity receipt is missing")
    _require(current_receipt_path.is_file(), "current parity receipt is missing")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    with current_receipt_path.open("r", encoding="utf-8") as handle:
        current_receipt = json.load(handle)
    _require(receipt.get("schema_version") == scorer.PARITY_SCHEMA_VERSION, "parity schema changed")
    _require(receipt.get("passed") is True, "parity receipt did not pass")
    _require(
        current_receipt.get("schema_version") == scorer.PARITY_SCHEMA_VERSION,
        "current parity schema changed",
    )
    _require(current_receipt.get("passed") is True, "current parity receipt did not pass")
    _require(
        receipt.get("checks", {}).get("ordinary_vs_early_features_bitwise_equal") is True,
        "parity receipt lacks exact early-stop equality",
    )
    _require(
        current_receipt.get("checks", {}).get("ordinary_vs_early_features_bitwise_equal")
        is True,
        "current parity receipt lacks exact early-stop equality",
    )
    receipt["receipt_sha256"] = _sha256(receipt_path)
    current_receipt["receipt_sha256"] = _sha256(current_receipt_path)

    # The production jobs started immediately before two default artifact paths
    # were renamed from v2_exact to v3_deployable.  Prove the old source bytes
    # from the current source rather than relaxing the source-hash check.
    current_source_path = Path(scorer.__file__).resolve()
    current_source = current_source_path.read_text(encoding="utf-8")
    _require(
        current_source.count(CURRENT_ARTIFACT_DIRECTORY) == 2,
        "current scorer does not contain exactly two v3 artifact defaults",
    )
    reconstructed_historical_source = current_source.replace(
        CURRENT_ARTIFACT_DIRECTORY, OLD_ARTIFACT_DIRECTORY
    )
    reconstructed_historical_sha256 = hashlib.sha256(
        reconstructed_historical_source.encode("utf-8")
    ).hexdigest()
    _require(
        reconstructed_historical_sha256
        == receipt.get("inputs", {}).get("script", {}).get("sha256"),
        "historical scorer cannot be reconstructed by the documented path-only change",
    )
    _require(
        current_receipt["inputs"]["script"]["sha256"] == _sha256(current_source_path),
        "current parity receipt does not match scorer source",
    )
    shared_inputs = ("checkpoint", "config", "head_npz", "canonical_rows", "reference_scores")
    for name in shared_inputs:
        record = receipt.get("inputs", {}).get(name, {})
        path = Path(str(record.get("path", "")))
        _require(path.is_file(), "parity input is missing: {}".format(name))
        _require(_sha256(path) == record.get("sha256"), "parity input checksum changed: {}".format(name))
        current_record = current_receipt.get("inputs", {}).get(name, {})
        current_path = Path(str(current_record.get("path", "")))
        _require(current_path.is_file(), "current parity input is missing: {}".format(name))
        _require(
            _sha256(current_path) == current_record.get("sha256"),
            "current parity input checksum changed: {}".format(name),
        )
        _require(
            record.get("sha256") == current_record.get("sha256"),
            "historical/current parity input bytes differ: {}".format(name),
        )
    _require(
        receipt.get("checks") == current_receipt.get("checks"),
        "historical/current parity numerical checks differ",
    )
    provenance_bridge = {
        "validated": True,
        "change_kind": "two default artifact-directory strings only; scoring logic unchanged",
        "replacement_count": 2,
        "historical_directory": OLD_ARTIFACT_DIRECTORY,
        "current_directory": CURRENT_ARTIFACT_DIRECTORY,
        "historical_script_sha256": reconstructed_historical_sha256,
        "current_script_sha256": _sha256(current_source_path),
        "historical_receipt": {"path": str(receipt_path), "sha256": receipt["receipt_sha256"]},
        "current_receipt": {
            "path": str(current_receipt_path),
            "sha256": current_receipt["receipt_sha256"],
        },
        "byte_identical_logical_inputs": {
            name: receipt["inputs"][name]["sha256"] for name in shared_inputs
        },
    }

    partition_table = pd.read_csv(universe / "candidate_partitions.csv")
    _require(len(partition_table) == 12, "expected 12 universe partitions")
    _require(partition_table["peptide_design_code"].nunique() == 12, "duplicate peptide partition")
    records = []
    for row in partition_table.itertuples(index=False):
        input_path = universe / str(row.partition)
        output_path = scores / Path(str(row.partition)).name
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
        records.append(
            _validate_partition(
                input_path,
                output_path,
                manifest_path,
                int(row.candidate_rows),
                str(row.partition_sha256),
                str(row.peptide_design_code),
                receipt,
            )
        )
        print("validated {}: {} rows".format(row.peptide_design_code, row.candidate_rows), flush=True)
    total_rows = sum(record["rows"] for record in records)
    _require(total_rows == 4_447_848, "total candidate score count changed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": True,
        "partitions": records,
        "summary": {
            "partition_count": len(records),
            "rows": total_rows,
            "output_bytes": sum(record["output_bytes"] for record in records),
            "maximum_sigmoid_error": max(record["maximum_sigmoid_error"] for record in records),
            "minimum_rows_per_second": min(record["rows_per_second"] for record in records),
            "maximum_peak_allocated_bytes": max(record["peak_allocated_bytes"] for record in records),
            "elapsed_validation_seconds": float(time.time() - started),
        },
        "inputs": {
            "universe_dir": str(universe),
            "scores_dir": str(scores),
            "production_parity_receipt": {
                "path": str(receipt_path),
                "sha256": receipt["receipt_sha256"],
            },
            "current_parity_receipt": {
                "path": str(current_receipt_path),
                "sha256": current_receipt["receipt_sha256"],
            },
        },
        "provenance_bridge": provenance_bridge,
    }
    if args.output_json is not None:
        output_json = Path(args.output_json).resolve()
        _require(not output_json.exists(), "validation report exists; refusing overwrite")
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(output_json, 0o600)
    print(json.dumps(report["summary"], sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES)
    parser.add_argument(
        "--parity-receipt", type=Path, default=DEFAULT_HISTORICAL_PARITY
    )
    parser.add_argument(
        "--current-parity-receipt", type=Path, default=DEFAULT_CURRENT_PARITY
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
