#!/usr/bin/env python3
"""Fail-closed release audit for native LibB exhaustive candidate scores.

This combines four deliberately different checks:

1. the deployed scorer, using the exhaustive batch sizes, reproduces rows from
   the merged exhaustive score archive to a strict numerical tolerance;
2. replaying the historical batch-16 panel context remains compatible with the
   sealed 120-pair predictions;
3. the deliberately different ten-row panel packing preserves every relevant
   ranking and cutoff decision despite small GPU batch-shape roundoff; and
4. repacking the primary menu's top-50 rows at batch 16 preserves the exact
   top-ten decision for every peptide.

The script reads model scores and label-free identities/sequences only.  It
does not load retention values or binder labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import (  # noqa: E402
    audit_libb_native_projection_candidate_scorer_parity as panel_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SEEDS = tuple(panel_audit.SEEDS)
FAMILIES = ("rde", "stab")
MODEL_NAMES = {
    "rde": "rde_network_designed_3fold_ensemble",
    "stab": "stab_designed_ordered",
}
PRODUCTION_BATCH_SIZES = {"rde": 128, "stab": 20}
PRODUCTION_ROWS = 640
PRODUCTION_TOLERANCE = 1e-6
HISTORICAL_TOLERANCE = 2e-6
HISTORICAL_AGGREGATE_TOLERANCE = 1e-6
CROSS_BATCH_TOLERANCE = 1e-4
EXPECTED_TARGETS = tuple(panel_audit.EXPECTED_TARGETS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"missing asset: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _new_private_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    _require(resolved != PRIVATE_ROOT and PRIVATE_ROOT in resolved.parents,
             "output must be a new directory below private_data")
    _require(not resolved.exists(), "output exists; refusing overwrite")
    return resolved


def _stable_rank(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    ranked = frame.sort_values(
        [column, "pair_uid"], ascending=[False, True], kind="mergesort"
    )
    return tuple(ranked["pair_uid"].astype(str))


def _sigmoid_mean_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    return 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))


def _probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    _require(np.isfinite(values).all(), "probability-to-logit input is non-finite")
    _require(((values >= 0.0) & (values <= 1.0)).all(),
             "probability-to-logit input leaves [0,1]")
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    _require(np.isfinite(values).all(), "sigmoid input is non-finite")
    return 1.0 / (1.0 + np.exp(-values))


def _validate_score_frame(frame: pd.DataFrame, family: str, context: str) -> np.ndarray:
    """Validate every stored score and independently recompute aggregates."""

    _require(tuple(frame.columns) == tuple(panel_audit.SCORE_COLUMNS),
             f"{context} score columns changed")
    _require(frame["model"].astype(str).eq(MODEL_NAMES[family]).all(),
             f"{context} model name changed")
    seed_columns = [f"score_seed_{seed}" for seed in SEEDS]
    seeds = frame[seed_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    _require(np.isfinite(seeds).all(), f"{context} seed scores are non-finite")
    _require(((seeds >= 0.0) & (seeds <= 1.0)).all(),
             f"{context} seed scores leave [0,1]")
    expected_mean = seeds.mean(axis=1)
    expected_logit_mean = _sigmoid_mean_logit(seeds)
    expected_sd = seeds.std(axis=1, ddof=1)
    stored_mean = pd.to_numeric(frame["score_mean_probability"], errors="raise").to_numpy(float)
    stored_logit_mean = pd.to_numeric(frame["score_mean_logit"], errors="raise").to_numpy(float)
    stored_sd = pd.to_numeric(frame["score_seed_sd"], errors="raise").to_numpy(float)
    for name, values in (
        ("score_mean_probability", stored_mean),
        ("score_mean_logit", stored_logit_mean),
        ("score_seed_sd", stored_sd),
    ):
        _require(np.isfinite(values).all(), f"{context}/{name} is non-finite")
    _require(((stored_mean >= 0.0) & (stored_mean <= 1.0)).all(),
             f"{context}/score_mean_probability leaves [0,1]")
    _require(((stored_logit_mean >= 0.0) & (stored_logit_mean <= 1.0)).all(),
             f"{context}/score_mean_logit leaves [0,1]")
    _require((stored_sd >= 0.0).all(), f"{context}/score_seed_sd is negative")
    # CSV serialization is float32-like for seeds, so 1e-6 is a strict but
    # realistic identity check for the three derived columns.
    _require(np.max(np.abs(stored_mean - expected_mean)) <= PRODUCTION_TOLERANCE,
             f"{context}/score_mean_probability is not the seed mean")
    _require(np.max(np.abs(stored_logit_mean - expected_logit_mean)) <= PRODUCTION_TOLERANCE,
             f"{context}/score_mean_logit is not sigmoid(mean seed logits)")
    _require(np.max(np.abs(stored_sd - expected_sd)) <= PRODUCTION_TOLERANCE,
             f"{context}/score_seed_sd is not the sample seed SD")
    return seeds


def _command_value(argv: list, option: str) -> str:
    _require(argv.count(option) == 1, f"command does not contain exactly one {option}")
    index = argv.index(option)
    _require(index + 1 < len(argv), f"command lacks value for {option}")
    return str(argv[index + 1])


def _validate_source_integrity(
    integrity: dict,
    producer_prefix: str,
    scorer_prefix: str,
    context: str,
) -> tuple[dict, dict]:
    producer_start = integrity.get(f"{producer_prefix}_at_start", {})
    producer_end = integrity.get(f"{producer_prefix}_before_publish", {})
    scorer_start = integrity.get(f"{scorer_prefix}_at_start", {})
    scorer_end = integrity.get(f"{scorer_prefix}_before_publish", {})
    _require(integrity.get(f"{producer_prefix}_unchanged_during_run") is True,
             f"{context} producer stability flag is not true")
    _require(integrity.get(f"{scorer_prefix}_unchanged_during_run") is True,
             f"{context} scorer stability flag is not true")
    _require(producer_start == producer_end and scorer_start == scorer_end,
             f"{context} source changed during execution")
    _validate_current_asset(producer_start, f"{context} producer")
    _validate_current_asset(scorer_start, f"{context} candidate scorer")
    return producer_start, scorer_start


def _validate_lock(path: Path, expected_id: str, expected_components: tuple[str, ...]) -> tuple[float, dict]:
    path = path.expanduser().resolve()
    lock = _load_json(path)
    _require(lock.get("schema_version") == "libb-score-ensemble-lock-v1",
             f"unexpected lock schema: {path.name}")
    _require(lock.get("lock_id") == expected_id, f"wrong lock ID: {path.name}")
    _require(lock.get("retention_labels_read") is False,
             f"lock was not fitted independently of retention: {path.name}")
    _require(lock.get("output_transform") == "sigmoid" and float(lock.get("intercept")) == 0.0,
             f"lock transform/intercept changed: {path.name}")
    component_contracts = {
        "mint_layer5": {
            "name": "mint_layer5", "weight": 0.5 if len(expected_components) == 2 else 1.0,
            "center": 0.0, "scale": 1.0, "score_column": "mint_layer5_logit",
            "higher_is_better": True,
        },
        "rde_network_designed_3fold": {
            "name": "rde_network_designed_3fold",
            "weight": 0.5 if len(expected_components) == 2 else 1.0,
            "center": 0.0, "scale": 1.0, "score_column": "rde_logit",
            "higher_is_better": True,
        },
        "stab_designed_ordered": {
            "name": "stab_designed_ordered", "weight": 1.0,
            "center": 0.0, "scale": 1.0, "score_column": "stab_logit",
            "higher_is_better": True,
        },
    }
    components = lock.get("components", [])
    _require(tuple(str(item.get("name")) for item in components) == expected_components,
             f"lock components changed: {path.name}")
    for observed, name in zip(components, expected_components):
        expected = component_contracts[name]
        for field, value in expected.items():
            _require(observed.get(field) == value,
                     f"lock component field changed: {path.name}/{name}/{field}")
    selection = lock.get("selection", {})
    _require(selection.get("max_candidates_per_peptide") == 10,
             f"lock max candidates changed: {path.name}")
    _require(selection.get("review_pool_per_peptide") == 50,
             f"lock review pool changed: {path.name}")
    _require(selection.get("minimum_code_hamming_distance") == 0,
             f"lock diversity filter changed: {path.name}")
    _require(selection.get("pad_below_threshold") is False,
             f"lock padding rule changed: {path.name}")
    threshold = float(selection.get("score_threshold", math.nan))
    _require(math.isfinite(threshold) and 0.0 <= threshold <= 1.0,
             f"lock threshold is invalid: {path.name}")

    bundle_path = path.parent / "manifest.json"
    bundle = _load_json(bundle_path)
    _require(bundle.get("schema_version")
             == "libb-native-projection-weak-oof-deployment-locks-v1",
             "unexpected deployment-lock bundle schema")
    _require(bundle.get("retention_labels_read") is False,
             "deployment-lock bundle does not declare retention unread")
    record = bundle.get("locks", {}).get(path.name, {})
    _require(record.get("sha256") == _sha256(path),
             f"lock hash differs from bundle manifest: {path.name}")
    weak_sources = lock.get("weak_source_sha256", {})
    bundle_sources = bundle.get("sources", {})
    _require(set(bundle_sources) == {"manifest", "oof", "selection", "stacker"}
             and set(weak_sources) == set(bundle_sources),
             "deployment-lock weak-source set changed")
    for name, source_record in bundle_sources.items():
        source_path = (REPO_ROOT / str(source_record.get("path", ""))).resolve()
        _require(source_path.is_file(), f"weak source missing: {name}")
        observed = _sha256(source_path)
        _require(source_record.get("sha256") == observed,
                 f"weak source changed after lock creation: {name}")
        _require(weak_sources.get(name) == observed,
                 f"lock weak-source hash differs from bundle: {name}")
    return threshold, {
        "lock": _asset(path),
        "bundle_manifest": _asset(bundle_path),
        "lock_id": expected_id,
        "score_threshold": threshold,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_current_asset(record: dict, message: str) -> None:
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    _require(path.is_file(), f"{message}: file missing")
    _require(record.get("sha256") == _sha256(path), f"{message}: hash changed")


def _load_exhaustive(directory: Path, family: str) -> tuple[dict, dict]:
    directory = directory.expanduser().resolve()
    manifest_path = directory / "manifest.json"
    payload, receipt = panel_audit._load_exhaustive_manifest(manifest_path, family)
    outcome_access = payload.get("outcome_access", {})
    _require(outcome_access.get("retention_measurements_read") is False,
             f"{family} exhaustive manifest does not declare retention unread")
    _require(outcome_access.get("selection_labels_read") is False,
             f"{family} exhaustive manifest does not declare selection labels unread")
    _require(payload.get("candidate_scoring_inputs", {}).get(
        "contains_training_or_retention_labels") is False,
        f"{family} exhaustive scoring input does not declare labels absent")
    source = payload.get("source_chunks", {})
    _validate_current_asset(source.get("scorer_producer", {}),
                            f"{family} exhaustive scorer")
    readout = source.get("readout_provenance", {})
    _require(readout.get("readout_mode") == "native_learned_projection",
             f"{family} exhaustive readout is not native learned projection")
    config = Path(str(readout.get("config_path", ""))).expanduser().resolve()
    _require(config.is_file() and readout.get("config_sha256") == _sha256(config),
             f"{family} exhaustive readout config changed")
    checkpoints = readout.get("checkpoints", [])
    _require(len(checkpoints) == (15 if family == "rde" else 5),
             f"{family} exhaustive checkpoint count changed")
    _require(len({str(record.get('filename')) for record in checkpoints}) == len(checkpoints),
             f"{family} exhaustive checkpoint names repeat")
    for checkpoint in checkpoints:
        checkpoint_dir = Path(readout["checkpoint_directory"])
        path = checkpoint_dir / str(checkpoint["filename"])
        _require(path.is_file() and checkpoint.get("sha256") == _sha256(path),
                 f"{family} exhaustive readout checkpoint changed: {path.name}")
    partition_records = payload.get("outputs", {}).get("partitions", [])
    _require(len(partition_records) == 12
             and {str(item.get("peptide_design_code")) for item in partition_records}
             == set(EXPECTED_TARGETS),
             f"{family} exhaustive output partition set changed")
    total_rows = 0
    for record in partition_records:
        peptide = str(record["peptide_design_code"])
        path = directory / str(record.get("output_file", ""))
        _require(path == directory / f"peptide_{peptide}.parquet",
                 f"{family}/{peptide} exhaustive output filename changed")
        _require(path.is_file() and record.get("output_sha256") == _sha256(path)
                 and int(record.get("output_bytes", -1)) == path.stat().st_size,
                 f"{family}/{peptide} exhaustive output differs from manifest")
        rows = int(record.get("candidate_rows", -1))
        _require(rows > 0, f"{family}/{peptide} exhaustive row count is invalid")
        total_rows += rows
    _require(total_rows == 4_447_848
             and int(payload.get("outputs", {}).get("candidate_rows", -1)) == total_rows,
             f"{family} exhaustive output row total changed")
    return payload, receipt


def _production_replay(
    family: str,
    replay_dir: Path,
    exhaustive_dir: Path,
    exhaustive: dict,
) -> tuple[list[dict], dict]:
    replay_dir = replay_dir.expanduser().resolve()
    invocation_path = replay_dir / "invocation_manifest.json"
    invocation = _load_json(invocation_path)
    _require(
        invocation.get("schema_version")
        == "libb-native-projection-production-batch-replay-v1",
        f"{family} production invocation schema changed",
    )
    expected_batch = PRODUCTION_BATCH_SIZES[family]
    _require(invocation.get("family") == family and invocation.get("peptide") == "AF",
             f"{family} production invocation family/peptide changed")
    _require(int(invocation.get("rows", -1)) == PRODUCTION_ROWS,
             f"{family} production invocation row count changed")
    _require(int(invocation.get("batch_size", -1)) == expected_batch,
             f"{family} production replay did not use the exhaustive batch size")
    _require(int(invocation.get("chunk_rows", -1)) == PRODUCTION_ROWS
             and int(invocation.get("limit", -1)) == PRODUCTION_ROWS,
             f"{family} production invocation chunk/limit changed")
    access = invocation.get("outcome_access", {})
    _require(access == {
        "binder_values_loaded": False,
        "retention_values_loaded": False,
        "selection_labels_loaded": False,
    }, f"{family} production replay outcome-access contract changed")
    wrapper_source, invocation_scorer = _validate_source_integrity(
        invocation.get("source_integrity", {}), "producer", "candidate_scorer",
        f"{family} production replay",
    )
    expected_wrapper = (
        REPO_ROOT / "downstream/AffibodyMHC/run_libb_native_projection_production_replay.py"
    ).resolve()
    _require(Path(wrapper_source["path"]).resolve() == expected_wrapper,
             f"{family} production replay used an unexpected wrapper")
    argv = [str(value) for value in invocation.get("command_argv", [])]
    _require(_command_value(argv, "--family") == family,
             f"{family} production invocation family argument changed")
    _require(_command_value(argv, "--peptides") == "AF",
             f"{family} production invocation peptide argument changed")
    _require(int(_command_value(argv, "--batch-size")) == expected_batch,
             f"{family} production invocation batch argument changed")
    _require(int(_command_value(argv, "--chunk-rows")) == PRODUCTION_ROWS
             and int(_command_value(argv, "--limit")) == PRODUCTION_ROWS,
             f"{family} production invocation row arguments changed")
    _require(Path(_command_value(argv, "--input-dir")).resolve()
             == Path(exhaustive.get("candidate_scoring_inputs", {}).get("path", "")).resolve(),
             f"{family} production invocation used different scoring inputs")

    exhaustive_manifest_path = exhaustive_dir.expanduser().resolve() / "manifest.json"
    _require(invocation.get("exhaustive_manifest") == _asset(exhaustive_manifest_path),
             f"{family} production invocation points to a different exhaustive manifest")
    receipts = sorted(replay_dir.glob(f"worker_{family}_slice00of01_AF.json"))
    _require(len(receipts) == 1, f"{family} production replay receipt missing")
    receipt_path = receipts[0]
    recorded_worker = invocation.get("worker_receipt", {})
    _require(recorded_worker.get("path") == receipt_path.name
             and recorded_worker.get("sha256") == _sha256(receipt_path)
             and int(recorded_worker.get("bytes", -1)) == receipt_path.stat().st_size,
             f"{family} production worker receipt differs from invocation manifest")
    receipt = _load_json(receipt_path)
    _require(receipt.get("schema_version") == "libb-fixed-structure-candidate-scores-v1",
             f"{family} production replay schema changed")
    _require(receipt.get("family") == family and receipt.get("model") == MODEL_NAMES[family],
             f"{family} production replay family/model mismatch")
    _require(receipt.get("labels_read") is False and receipt.get("retention_read") is False,
             f"{family} production replay was not label-free")
    _require(receipt.get("readout_seeds") == list(SEEDS),
             f"{family} production replay seed set changed")
    _require(receipt.get("slice_index") == 0 and receipt.get("num_slices") == 1,
             f"{family} production replay is not one complete slice")
    peptide_records = receipt.get("peptides", [])
    _require(len(peptide_records) == 1 and peptide_records[0].get("peptide") == "AF",
             f"{family} production replay target changed")
    _require(int(peptide_records[0].get("rows", -1)) == PRODUCTION_ROWS,
             f"{family} production replay row count changed")
    source = exhaustive.get("source_chunks", {})
    _require(receipt.get("producer") == source.get("scorer_producer"),
             f"{family} replay scorer differs from exhaustive scorer")
    _require(invocation_scorer == _asset(Path(source["scorer_producer"]["path"])),
             f"{family} invocation scorer differs from exhaustive scorer")
    _require(receipt.get("readout_provenance") == source.get("readout_provenance"),
             f"{family} replay readouts differ from exhaustive readouts")
    _require(
        receipt.get("input_manifest_sha256")
        == exhaustive.get("candidate_scoring_inputs", {}).get("manifest_sha256"),
        f"{family} replay input manifest differs from exhaustive input",
    )

    chunks = sorted((replay_dir / "peptide_AF").glob("*.csv.gz"))
    _require(len(chunks) == 1, f"{family} production replay must have one score chunk")
    recorded_chunk = invocation.get("score_chunk", {})
    _require(recorded_chunk.get("path") == str(chunks[0].relative_to(replay_dir))
             and recorded_chunk.get("sha256") == _sha256(chunks[0])
             and int(recorded_chunk.get("bytes", -1)) == chunks[0].stat().st_size,
             f"{family} production score chunk differs from invocation manifest")
    replay = pd.read_csv(chunks[0], dtype={"pair_uid": str})
    _validate_score_frame(replay, family, f"{family} production replay")
    _require(len(replay) == PRODUCTION_ROWS,
             f"{family} production replay does not contain 640 rows")
    merged_path = exhaustive_dir.expanduser().resolve() / "peptide_AF.parquet"
    _require(merged_path.is_file(), f"{family} AF merged partition missing")
    af_records = [
        record for record in exhaustive.get("outputs", {}).get("partitions", [])
        if record.get("peptide_design_code") == "AF"
    ]
    _require(len(af_records) == 1, f"{family} exhaustive manifest lacks one AF partition")
    _require(af_records[0].get("output_file") == merged_path.name
             and af_records[0].get("output_sha256") == _sha256(merged_path)
             and int(af_records[0].get("candidate_rows", -1)) > PRODUCTION_ROWS,
             f"{family} merged AF partition differs from exhaustive manifest")
    merged = pd.read_parquet(merged_path).sort_values(
        "candidate_row_index", kind="mergesort"
    ).head(PRODUCTION_ROWS).reset_index(drop=True)
    replay = replay.sort_values("candidate_row_index", kind="mergesort").reset_index(drop=True)
    for field in ("candidate_row_index", "pair_uid", "peptide_design_code", "affibody_design_code"):
        _require(replay[field].astype(str).equals(merged[field].astype(str)),
                 f"{family} production replay changed ordered {field}")

    merged_seed_columns = [f"{family}_probability_seed_{seed}" for seed in SEEDS]
    merged_seeds = merged[merged_seed_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    _require(np.isfinite(merged_seeds).all()
             and ((merged_seeds >= 0.0) & (merged_seeds <= 1.0)).all(),
             f"{family} merged AF seed scores are invalid")
    merged_mean = pd.to_numeric(
        merged[f"{family}_seed_mean_probability"], errors="raise"
    ).to_numpy(float)
    merged_aggregate = pd.to_numeric(
        merged[f"{family}_probability"], errors="raise"
    ).to_numpy(float)
    merged_sd = pd.to_numeric(
        merged[f"{family}_seed_probability_sd"], errors="raise"
    ).to_numpy(float)
    _require(np.isfinite(merged_mean).all() and np.isfinite(merged_aggregate).all()
             and np.isfinite(merged_sd).all(), f"{family} merged aggregates are non-finite")
    _require(((merged_mean >= 0.0) & (merged_mean <= 1.0)).all()
             and ((merged_aggregate >= 0.0) & (merged_aggregate <= 1.0)).all()
             and (merged_sd >= 0.0).all(), f"{family} merged aggregates are outside range")
    _require(np.max(np.abs(merged_mean - merged_seeds.mean(axis=1))) <= PRODUCTION_TOLERANCE,
             f"{family} merged seed mean is inconsistent")
    _require(np.max(np.abs(merged_aggregate - _sigmoid_mean_logit(merged_seeds)))
             <= PRODUCTION_TOLERANCE, f"{family} merged logit aggregate is inconsistent")
    _require(np.max(np.abs(merged_sd - merged_seeds.std(axis=1, ddof=1)))
             <= PRODUCTION_TOLERANCE, f"{family} merged seed SD is inconsistent")

    column_pairs = [
        (f"score_seed_{seed}", f"{family}_probability_seed_{seed}") for seed in SEEDS
    ] + [
        ("score_mean_probability", f"{family}_seed_mean_probability"),
        ("score_mean_logit", f"{family}_probability"),
        ("score_seed_sd", f"{family}_seed_probability_sd"),
    ]
    rows = []
    for observed_column, reference_column in column_pairs:
        observed = pd.to_numeric(replay[observed_column], errors="raise").to_numpy(float)
        reference = pd.to_numeric(merged[reference_column], errors="raise").to_numpy(float)
        _require(np.isfinite(observed).all() and np.isfinite(reference).all(),
                 f"{family}/{observed_column} has non-finite values")
        difference = np.abs(observed - reference)
        maximum = float(difference.max())
        _require(maximum <= PRODUCTION_TOLERANCE,
                 f"{family}/{observed_column} exceeds production tolerance")
        _require(_stable_rank(replay, observed_column) == _stable_rank(merged, reference_column),
                 f"{family}/{observed_column} production ranking changed")
        rows.append({
            "family": family,
            "observed_column": observed_column,
            "exhaustive_column": reference_column,
            "rows": PRODUCTION_ROWS,
            "maximum_absolute_difference": maximum,
            "mean_absolute_difference": float(difference.mean()),
            "tolerance": PRODUCTION_TOLERANCE,
            "within_tolerance": True,
            "full_640_ranking_identical": True,
        })
    return rows, {
        "status": "pass",
        "family": family,
        "rows": PRODUCTION_ROWS,
        "peptide": "AF",
        "verified_batch_size": expected_batch,
        "maximum_absolute_probability_difference": max(
            row["maximum_absolute_difference"] for row in rows
        ),
        "all_columns_within_1e-6": True,
        "all_rankings_identical": True,
        "invocation_manifest": _asset(invocation_path),
        "invocation_wrapper": wrapper_source,
        "replay_receipt": _asset(receipt_path),
        "replay_chunk": _asset(chunks[0]),
        "merged_partition": _asset(merged_path),
    }


def _historical_replay(
    family: str,
    replay_dir: Path,
    reference_dir: Path,
    canonical_rows: Path,
    exhaustive_manifest: dict,
    exhaustive_manifest_path: Path,
    sealed_reference_hashes: dict[str, str],
) -> tuple[list[dict], dict]:
    replay_dir = replay_dir.expanduser().resolve()
    manifest_path = replay_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version")
        == "libb-native-projection-historical-tail-batch-replay-v1",
        f"{family} historical replay schema changed",
    )
    _require(manifest.get("family") == family and manifest.get("model") == MODEL_NAMES[family],
             f"{family} historical replay family/model mismatch")
    _require(manifest.get("canonical_split_counts") == {"train": 30648, "eval": 120},
             f"{family} historical replay split counts changed")
    _require(manifest.get("rows") == 120 and manifest.get("seeds") == list(SEEDS),
             f"{family} historical replay row/seed contract changed")
    access = manifest.get("outcome_access", {})
    _require(access.get("selection_labels_loaded") is False,
             f"{family} historical replay loaded selection labels")
    _require(access.get("retention_values_loaded") is False,
             f"{family} historical replay loaded retention")
    _require(access.get("binder_values_loaded") is False,
             f"{family} historical replay loaded binder labels")
    replay_producer, replay_scorer = _validate_source_integrity(
        manifest.get("source_integrity", {}), "producer", "imported_candidate_scorer",
        f"{family} historical replay",
    )
    expected_replay_producer = (
        REPO_ROOT
        / "downstream/AffibodyMHC/replay_libb_native_projection_historical_tail_batches.py"
    ).resolve()
    _require(Path(replay_producer["path"]).resolve() == expected_replay_producer,
             f"{family} historical replay used an unexpected producer")
    _require(manifest.get("producer") == replay_producer,
             f"{family} historical producer record changed")
    _require(manifest.get("imported_candidate_scorer") == replay_scorer,
             f"{family} historical scorer record changed")
    cross = manifest.get("exhaustive_score_cross_bind", {})
    _require(cross.get("family") == family,
             f"{family} replay exhaustive cross-bind family changed")
    _require(cross.get("scorer_producer_cross_bound") is True
             and cross.get("readout_provenance_cross_bound") is True,
             f"{family} replay cross-bind flags are not true")
    _require(cross.get("manifest", {}).get("sha256") == _sha256(exhaustive_manifest_path),
             f"{family} replay points to a different exhaustive manifest")
    _require(cross.get("manifest") == _asset(exhaustive_manifest_path),
             f"{family} replay exhaustive-manifest asset changed")
    exhaustive_source = exhaustive_manifest.get("source_chunks", {})
    _require(
        {key: replay_scorer[key] for key in ("path", "sha256")}
        == exhaustive_source.get("scorer_producer"),
        f"{family} historical replay scorer differs from exhaustive scorer",
    )
    _require(manifest.get("readout_provenance")
             == exhaustive_source.get("readout_provenance"),
             f"{family} historical replay readouts differ from exhaustive readouts")

    inputs = manifest.get("inputs", {})
    _require(set(inputs) == {
        "canonical_rows", "pdb", "readout_checkpoints", "readout_config",
        "residue_mapping", "vendor_checkpoint",
        *({"vendor_network_checkpoint"} if family == "rde" else set()),
    }, f"{family} historical replay input asset set changed")
    for name, record in inputs.items():
        if isinstance(record, list):
            _require(record, f"{family} historical replay has no {name}")
            for index, item in enumerate(record):
                _validate_current_asset(item, f"{family} historical {name}[{index}]")
        else:
            _validate_current_asset(record, f"{family} historical {name}")
    _require(inputs["canonical_rows"] == _asset(canonical_rows),
             f"{family} historical replay used different canonical rows")
    readout = exhaustive_source["readout_provenance"]
    _require(inputs["readout_config"]["sha256"] == readout["config_sha256"],
             f"{family} historical readout config differs from exhaustive scoring")
    checkpoint_hashes = {
        str(item["sha256"]) for item in inputs["readout_checkpoints"]
    }
    _require(checkpoint_hashes == {
        str(item["sha256"]) for item in readout["checkpoints"]
    }, f"{family} historical checkpoint set differs from exhaustive scoring")

    contract = manifest.get("historical_batch_contract", {})
    _require(contract.get("batch_size") == 16 and contract.get("shards") == 8
             and contract.get("rows_per_shard") == 3846
             and contract.get("tail_rows_per_shard") == 22,
             f"{family} historical batch contract changed")
    receipts = contract.get("batch_receipts", [])
    _require(len(receipts) == 16, f"{family} historical replay lacks 16 batch receipts")
    observed_receipts = {}
    for receipt in receipts:
        key = (int(receipt.get("shard", -1)), int(receipt.get("tail_offset", -1)))
        _require(key not in observed_receipts, f"{family} historical receipt repeats {key}")
        observed_receipts[key] = receipt
    for shard in range(8):
        expected = {
            0: {
                "batch_rows": 16,
                "training_context_rows": 7,
                "evaluation_rows": 9,
                "canonical_row_indices": list(range(30592 + shard, 30713 + shard, 8)),
            },
            16: {
                "batch_rows": 6,
                "training_context_rows": 0,
                "evaluation_rows": 6,
                "canonical_row_indices": list(range(30720 + shard, 30761 + shard, 8)),
            },
        }
        for offset, values in expected.items():
            receipt = observed_receipts.get((shard, offset), {})
            for field, value in values.items():
                _require(receipt.get(field) == value,
                         f"{family} historical batch receipt changed: shard {shard}/{offset}/{field}")

    prediction_path = replay_dir / str(manifest.get("output", {}).get("path", ""))
    _require(prediction_path.is_file(), f"{family} historical replay output missing")
    _require(manifest["output"].get("sha256") == _sha256(prediction_path),
             f"{family} historical replay prediction hash changed")
    replay = pd.read_csv(prediction_path, dtype={"pair_uid": str})
    _require(len(replay) == 120 and not replay["pair_uid"].duplicated().any(),
             f"{family} historical replay is not 120 unique rows")
    historical_columns = (
        "canonical_row_index", "pair_uid", "peptide_design_code", "affibody_design_code",
        "historical_shard", "historical_batch_size", "historical_batch_position",
        "sequence_pair_sha256", *(f"score_seed_{seed}" for seed in SEEDS),
        "score_mean_probability", "score_mean_logit", "score_seed_sd", "model",
    )
    _require(tuple(replay.columns) == historical_columns,
             f"{family} historical replay output columns changed")
    score_view = replay[[
        "canonical_row_index", "pair_uid", "peptide_design_code", "affibody_design_code",
        *(f"score_seed_{seed}" for seed in SEEDS), "score_mean_probability",
        "score_mean_logit", "score_seed_sd", "model",
    ]].rename(columns={"canonical_row_index": "candidate_row_index"})
    _validate_score_frame(score_view, family, f"{family} historical replay")

    canonical_payload = _load_json(canonical_rows)
    canonical = pd.DataFrame(canonical_payload.get("rows", []))
    canonical = canonical.loc[canonical["split"].astype(str).eq("eval")].copy()
    canonical["pair_uid"] = canonical["row_id"].astype(str)
    canonical["peptide_design_code"] = canonical["chain1_sequence"].str[-9:].str[3:5]
    canonical["affibody_design_code"] = canonical["chain2_sequence"].map(
        lambda sequence: "".join(sequence[index] for index in (5, 9, 12, 13, 16))
    )
    expected = canonical[[
        "row_index", "pair_uid", "peptide_design_code", "affibody_design_code",
        "sequence_pair_sha256",
    ]].rename(columns={"row_index": "canonical_row_index"})
    observed = replay.sort_values("canonical_row_index", kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values("canonical_row_index", kind="mergesort").reset_index(drop=True)
    for field in expected.columns:
        _require(observed[field].astype(str).equals(expected[field].astype(str)),
                 f"{family} historical replay changed {field} mapping")
    _require(observed.groupby("historical_shard").size().to_dict()
             == {index: 15 for index in range(8)},
             f"{family} historical shard coverage changed")
    _require(observed["historical_batch_size"].value_counts().to_dict() == {16: 72, 6: 48},
             f"{family} historical batch composition changed")
    for row in observed.itertuples(index=False):
        row_index = int(row.canonical_row_index)
        shard = row_index % 8
        _require(int(row.historical_shard) == shard,
                 f"{family} historical shard assignment changed")
        local_eval_index = (row_index - (30648 + shard)) // 8
        expected_size = 16 if local_eval_index < 9 else 6
        expected_position = local_eval_index + 7 if local_eval_index < 9 else local_eval_index - 9
        _require(int(row.historical_batch_size) == expected_size
                 and int(row.historical_batch_position) == expected_position,
                 f"{family} historical batch position changed")

    reference, reference_records = panel_audit._load_reference_scores(
        reference_dir.expanduser().resolve(), family
    )
    _require({Path(record["path"]).name: record["sha256"] for record in reference_records}
             == sealed_reference_hashes,
             f"{family} historical references are not sealed by retrospective audit")
    joined = observed.merge(reference, on="pair_uid", validate="one_to_one")
    rows = []
    for seed in SEEDS:
        observed_column = f"score_seed_{seed}"
        reference_column = f"reference_seed_{seed}"
        a = pd.to_numeric(joined[observed_column], errors="raise").to_numpy(float)
        b = pd.to_numeric(joined[reference_column], errors="raise").to_numpy(float)
        _require(np.isfinite(a).all() and np.isfinite(b).all(),
                 f"{family}/{seed} historical values are non-finite")
        difference = np.abs(a - b)
        maximum = float(difference.max())
        _require(maximum <= HISTORICAL_TOLERANCE,
                 f"{family}/{seed} historical compatibility exceeds 2e-6")
        peptide_ranks = all(
            _stable_rank(block, observed_column) == _stable_rank(block, reference_column)
            for _, block in joined.groupby("peptide_design_code", sort=True)
        )
        _require(peptide_ranks, f"{family}/{seed} historical within-peptide ranking changed")
        rows.append({
            "family": family,
            "comparison": "historical_seed",
            "seed": seed,
            "rows": 120,
            "maximum_absolute_probability_difference": maximum,
            "mean_absolute_probability_difference": float(difference.mean()),
            "tolerance": HISTORICAL_TOLERANCE,
            "within_tolerance": True,
            "all_12_within_peptide_rankings_identical": True,
        })

    candidate_matrix = joined[[f"score_seed_{seed}" for seed in SEEDS]].to_numpy(float)
    reference_matrix = joined[[f"reference_seed_{seed}" for seed in SEEDS]].to_numpy(float)
    joined["candidate_aggregate"] = _sigmoid_mean_logit(candidate_matrix)
    joined["reference_aggregate"] = _sigmoid_mean_logit(reference_matrix)
    stored = pd.to_numeric(joined["score_mean_logit"], errors="raise").to_numpy(float)
    _require(np.max(np.abs(stored - joined["candidate_aggregate"].to_numpy(float))) <= 1e-12,
             f"{family} historical stored aggregate was not recomputed from seeds")
    aggregate_difference = np.abs(
        joined["candidate_aggregate"].to_numpy(float)
        - joined["reference_aggregate"].to_numpy(float)
    )
    aggregate_max = float(aggregate_difference.max())
    _require(aggregate_max <= HISTORICAL_AGGREGATE_TOLERANCE,
             f"{family} historical aggregate exceeds 1e-6")
    within_ranks = all(
        _stable_rank(block, "candidate_aggregate") == _stable_rank(block, "reference_aggregate")
        for _, block in joined.groupby("peptide_design_code", sort=True)
    )
    global_rank = _stable_rank(joined, "candidate_aggregate") == _stable_rank(
        joined, "reference_aggregate"
    )
    _require(within_ranks and global_rank,
             f"{family} historical aggregate ranking changed")
    rows.append({
        "family": family,
        "comparison": "historical_aggregate",
        "seed": "mean_logit",
        "rows": 120,
        "maximum_absolute_probability_difference": aggregate_max,
        "mean_absolute_probability_difference": float(aggregate_difference.mean()),
        "tolerance": HISTORICAL_AGGREGATE_TOLERANCE,
        "within_tolerance": True,
        "all_12_within_peptide_rankings_identical": True,
    })
    return rows, {
        "status": "pass",
        "family": family,
        "rows": 120,
        "maximum_seed_probability_difference": max(
            float(row["maximum_absolute_probability_difference"])
            for row in rows if row["comparison"] == "historical_seed"
        ),
        "maximum_aggregate_probability_difference": aggregate_max,
        "all_seed_within_peptide_rankings_identical": True,
        "aggregate_within_peptide_and_global_rankings_identical": True,
        "replay_manifest": _asset(manifest_path),
        "replay_predictions": _asset(prediction_path),
        "sealed_references": reference_records,
        "environment": manifest.get("environment", {}),
    }


def _cross_batch_panel(
    canonical_rows: Path,
    input_dir: Path,
    rde_output: Path,
    stab_output: Path,
    rde_reference: Path,
    stab_reference: Path,
    aggregate_reference_path: Path,
    retrospective_manifest_path: Path,
    exhaustive_payloads: dict,
    rde_cutoff: float,
    stab_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inputs, input_provenance = panel_audit._load_label_free_inputs(
        input_dir.expanduser().resolve(), canonical_rows.expanduser().resolve()
    )
    aggregate_reference, aggregate_provenance = panel_audit._load_aggregate_reference(
        aggregate_reference_path.expanduser().resolve()
    )
    sealed_hashes, retrospective_provenance = panel_audit._load_retrospective_audit_manifest(
        retrospective_manifest_path.expanduser().resolve(),
        aggregate_reference_path.expanduser().resolve(),
    )
    seed_frames = []
    aggregate_frames = []
    summaries = {}
    for family, score_dir, reference_dir, cutoff in (
        ("rde", rde_output, rde_reference, rde_cutoff),
        ("stab", stab_output, stab_reference, stab_cutoff),
    ):
        candidate, scorer_provenance = panel_audit._load_candidate_scores(
            score_dir.expanduser().resolve(), family,
            input_provenance["manifest_sha256"], CROSS_BATCH_TOLERANCE,
        )
        source = exhaustive_payloads[family]["source_chunks"]
        _require(scorer_provenance["producer"] == source["scorer_producer"],
                 f"{family} panel scorer differs from exhaustive scorer")
        _require(scorer_provenance["readout_provenance"] == source["readout_provenance"],
                 f"{family} panel readouts differ from exhaustive readouts")
        reference, reference_records = panel_audit._load_reference_scores(
            reference_dir.expanduser().resolve(), family
        )
        observed_hashes = {
            Path(record["path"]).name: record["sha256"] for record in reference_records
        }
        _require(observed_hashes == sealed_hashes[family],
                 f"{family} panel reference files are not audit-sealed")
        seed, aggregate, _ = panel_audit._audit_family(
            family, inputs, candidate, reference, aggregate_reference,
            CROSS_BATCH_TOLERANCE,
        )
        candidate_join = candidate.merge(reference, on="pair_uid", validate="one_to_one")
        reference_matrix = candidate_join[[f"reference_seed_{seed}" for seed in SEEDS]].to_numpy(float)
        reference_aggregate = _sigmoid_mean_logit(reference_matrix)
        candidate_aggregate = pd.to_numeric(candidate_join["score_mean_logit"], errors="raise").to_numpy(float)
        cutoff_flips = int(np.sum((candidate_aggregate >= cutoff) != (reference_aggregate >= cutoff)))
        _require(cutoff_flips == 0, f"{family} panel batch shape changes cutoff decisions")
        seed_rank_columns = [
            column for column in seed.columns if column.endswith("rankings_identical")
        ]
        aggregate_rank_columns = [
            column for column in aggregate.columns if "rankings_identical" in column
        ]
        aggregate_global_columns = [
            column for column in aggregate.columns if "global_ranking_identical" in column
        ]
        _require(seed_rank_columns and bool(seed[seed_rank_columns].all(axis=None)),
                 f"{family} cross-batch seed ranks changed")
        _require(aggregate_rank_columns and bool(aggregate[aggregate_rank_columns].all(axis=None)),
                 f"{family} cross-batch aggregate ranks changed")
        _require(aggregate_global_columns
                 and bool(aggregate[aggregate_global_columns].all(axis=None)),
                 f"{family} cross-batch aggregate global rank changed")
        aggregate_difference_columns = [
            column for column in aggregate.columns if column.startswith("maximum_")
        ]
        aggregate_maximum = float(
            aggregate[aggregate_difference_columns].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(float).max()
        )
        _require(math.isfinite(aggregate_maximum) and aggregate_maximum <= CROSS_BATCH_TOLERANCE,
                 f"{family} cross-batch aggregate difference exceeds tolerance")
        seed_frames.append(seed)
        aggregate_frames.append(aggregate)
        summaries[family] = {
            "status": "pass",
            "rows": 120,
            "effective_batch_size": 10,
            "maximum_seed_probability_difference": float(
                seed["maximum_absolute_probability_difference"].max()
            ),
            "maximum_aggregate_probability_difference": aggregate_maximum,
            "all_seed_and_aggregate_within_peptide_rankings_identical": True,
            "aggregate_global_ranking_identical": True,
            "locked_standalone_cutoff": cutoff,
            "cutoff_decision_flips": cutoff_flips,
            "candidate_scorer": scorer_provenance,
            "sealed_references": reference_records,
        }
    return pd.concat(seed_frames, ignore_index=True), pd.concat(aggregate_frames, ignore_index=True), {
        "status": "pass",
        "tolerance": CROSS_BATCH_TOLERANCE,
        "interpretation": (
            "Compatibility across a deliberately different GPU batch shape; exact rankings "
            "and cutoff decisions are required, while scalar scores may differ by <=1e-4."
        ),
        "families": summaries,
        "input_provenance": input_provenance,
        "aggregate_reference": aggregate_provenance,
        "retrospective_score_manifest": retrospective_provenance,
    }


def _primary_boundary(
    replay_dir: Path,
    input_dir: Path,
    review_pool_path: Path,
    primary_lock: Path,
    exhaustive_rde: dict,
    exhaustive_manifest_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Recompute the top-ten stability decision from hash-bound raw scores."""

    replay_dir = replay_dir.expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    review_pool_path = review_pool_path.expanduser().resolve()
    exhaustive_manifest_path = exhaustive_manifest_path.expanduser().resolve()
    threshold, lock_receipt = _validate_lock(
        primary_lock,
        "libb-native-mint-rde-primary-ensemble-weak-oof-v1",
        ("mint_layer5", "rde_network_designed_3fold"),
    )

    selection_manifest_path = review_pool_path.parent / "manifest.json"
    selection_manifest = _load_json(selection_manifest_path)
    _require(selection_manifest.get("schema_version") == "libb-ensemble-wetlab-shortlist-v1",
             "primary selection manifest schema changed")
    selection_access = selection_manifest.get("outcome_access", {})
    _require(selection_access.get("retention_table_read") is False
             and selection_access.get("retention_columns_allowed_in_component_scores") is False
             and selection_access.get("retention_used_for_lock_or_selection") is False,
             "primary selection outcome-access contract changed")
    _require(selection_manifest.get("outputs", {}).get("ranked_review_pool_rows") == 600
             and selection_manifest.get("outputs", {}).get("ranked_review_pool_sha256")
             == _sha256(review_pool_path), "primary review pool differs from selection manifest")
    _require(selection_manifest.get("ensemble_lock", {}).get("sha256")
             == _sha256(primary_lock), "primary selection used a different ensemble lock")
    _require(selection_manifest.get("ensemble_lock", {}).get("lock_id")
             == lock_receipt["lock_id"], "primary selection lock ID changed")
    exhaustive_hash = _sha256(exhaustive_manifest_path)
    for peptide in EXPECTED_TARGETS:
        record = selection_manifest.get("component_score_files", {}).get(peptide, {}).get(
            "rde_network_designed_3fold", {}
        )
        _require(record.get("producer_receipt_sha256") == exhaustive_hash,
                 f"primary {peptide} RDE score source differs from exhaustive manifest")
        partition = next(
            (item for item in exhaustive_rde.get("outputs", {}).get("partitions", [])
             if item.get("peptide_design_code") == peptide), None
        )
        _require(partition is not None and record.get("score_sha256")
                 == partition.get("output_sha256"),
                 f"primary {peptide} RDE partition hash differs from exhaustive manifest")

    review_columns = [
        "pair_uid", "peptide_design_code", "affibody_design_code",
        "chain1_smart_hla_linker_peptide_sequence", "chain2_affibody_sequence",
        "mint_layer5",
        "rde_network_designed_3fold", "ensemble_linear_score", "ensemble_score",
        "within_peptide_ensemble_rank",
    ]
    header = pd.read_csv(review_pool_path, nrows=0)
    _require(set(review_columns).issubset(header.columns), "primary review-pool schema changed")
    review = pd.read_csv(review_pool_path, usecols=review_columns, dtype={"pair_uid": str})
    _require(len(review) == 600 and not review["pair_uid"].duplicated().any(),
             "primary review pool is not 600 unique candidates")
    numeric_columns = [
        "mint_layer5", "rde_network_designed_3fold", "ensemble_linear_score",
        "ensemble_score", "within_peptide_ensemble_rank",
    ]
    numeric = review[numeric_columns].apply(pd.to_numeric, errors="raise")
    _require(np.isfinite(numeric.to_numpy(float)).all(), "primary review scores are non-finite")
    review[numeric_columns] = numeric
    expected_linear = 0.5 * (
        review["mint_layer5"].to_numpy(float)
        + review["rde_network_designed_3fold"].to_numpy(float)
    )
    _require(np.max(np.abs(review["ensemble_linear_score"] - expected_linear)) <= 1e-12,
             "primary production ensemble is not the locked 0.5/0.5 logit mean")
    _require(np.max(np.abs(review["ensemble_score"] - _sigmoid(expected_linear))) <= 1e-12,
             "primary production ensemble sigmoid is inconsistent")

    input_manifest_path = input_dir / "manifest.json"
    input_manifest = _load_json(input_manifest_path)
    _require(input_manifest.get("schema_version")
             == "libb-native-primary-boundary-label-free-input-v1",
             "primary boundary input schema changed")
    _require(input_manifest.get("labels_read") is False
             and input_manifest.get("retention_read") is False,
             "primary boundary fixture was not built label-free")
    source_path = (REPO_ROOT / str(input_manifest.get("selection_source", ""))).resolve()
    _require(source_path == review_pool_path
             and input_manifest.get("selection_source_sha256") == _sha256(review_pool_path),
             "primary boundary fixture points to a different review pool")
    _require(input_manifest.get("production_scoring_input_manifest_sha256")
             == exhaustive_rde.get("candidate_scoring_inputs", {}).get("manifest_sha256"),
             "primary boundary fixture points to different production scoring inputs")
    partition_records = {
        str(item.get("peptide_design_code")): item
        for item in input_manifest.get("partitions", [])
    }
    _require(set(partition_records) == set(EXPECTED_TARGETS),
             "primary boundary fixture target set changed")
    input_frames = []
    for peptide in EXPECTED_TARGETS:
        path = input_dir / f"peptide_{peptide}.csv.gz"
        record = partition_records[peptide]
        _require(path.is_file() and record.get("sha256") == _sha256(path)
                 and int(record.get("rows", -1)) == 50,
                 f"primary boundary input changed for {peptide}")
        frame = pd.read_csv(path, dtype={"pair_uid": str})
        _require(tuple(frame.columns) == tuple(panel_audit.INPUT_COLUMNS),
                 f"primary boundary input columns changed for {peptide}")
        _require(frame["candidate_row_index"].astype(int).tolist() == list(range(50))
                 and frame["peptide_design_code"].astype(str).eq(peptide).all(),
                 f"primary boundary input ordering changed for {peptide}")
        expected = review.loc[review["peptide_design_code"].astype(str).eq(peptide)].sort_values(
            ["within_peptide_ensemble_rank", "pair_uid"], kind="mergesort"
        )
        _require(expected["within_peptide_ensemble_rank"].astype(int).tolist() == list(range(1, 51)),
                 f"primary review ranks changed for {peptide}")
        for field in ("pair_uid", "peptide_design_code", "affibody_design_code"):
            _require(frame[field].astype(str).reset_index(drop=True).equals(
                expected[field].astype(str).reset_index(drop=True)),
                f"primary boundary input is not the production top50 for {peptide}")
        _require(frame["chain1_sequence"].astype(str).reset_index(drop=True).equals(
            expected["chain1_smart_hla_linker_peptide_sequence"].astype(str).reset_index(drop=True)),
            f"primary boundary peptide-side sequences differ from review pool for {peptide}")
        _require(frame["chain2_sequence"].astype(str).reset_index(drop=True).equals(
            expected["chain2_affibody_sequence"].astype(str).reset_index(drop=True)),
            f"primary boundary Affibody sequences differ from review pool for {peptide}")
        for row in frame.itertuples(index=False):
            chain1 = str(row.chain1_sequence)
            chain2 = str(row.chain2_sequence)
            peptide_code = str(row.peptide_design_code)
            affibody_code = str(row.affibody_design_code)
            _require(chain1[-9:][3:5] == peptide_code,
                     f"primary boundary peptide code/sequence mismatch: {row.pair_uid}")
            _require("".join(chain2[index] for index in (5, 9, 12, 13, 16)) == affibody_code,
                     f"primary boundary Affibody code/sequence mismatch: {row.pair_uid}")
            expected_uid = hashlib.sha256(
                f"LibB|{peptide_code}|{affibody_code}".encode("utf-8")
            ).hexdigest()[:20]
            expected_sequence_hash = hashlib.sha256(
                f"{chain1}|{chain2}".encode("utf-8")
            ).hexdigest()
            _require(str(row.pair_uid) == expected_uid,
                     f"primary boundary pair UID is inconsistent: {row.pair_uid}")
            _require(str(row.sequence_pair_sha256) == expected_sequence_hash,
                     f"primary boundary sequence hash is inconsistent: {row.pair_uid}")
        input_frames.append(frame)
    boundary_inputs = pd.concat(input_frames, ignore_index=True)
    _require(len(boundary_inputs) == 600 and not boundary_inputs["pair_uid"].duplicated().any(),
             "primary boundary inputs are not 600 unique rows")

    invocation_path = replay_dir / "invocation_manifest.json"
    invocation = _load_json(invocation_path)
    _require(invocation.get("schema_version")
             == "libb-native-projection-primary-boundary-replay-v1",
             "primary boundary invocation schema changed")
    _require(invocation.get("family") == "rde" and invocation.get("peptides") == list(EXPECTED_TARGETS),
             "primary boundary invocation target set changed")
    _require(invocation.get("rows") == 600 and invocation.get("rows_per_peptide") == 50
             and invocation.get("batch_size") == 16 and invocation.get("chunk_rows") == 50,
             "primary boundary invocation batching changed")
    _require(invocation.get("outcome_access") == {
        "binder_values_loaded": False,
        "retention_values_loaded": False,
        "selection_labels_loaded": False,
    }, "primary boundary replay outcome-access contract changed")
    wrapper_source, scorer_source = _validate_source_integrity(
        invocation.get("source_integrity", {}), "producer", "candidate_scorer",
        "primary boundary replay",
    )
    expected_wrapper = (
        REPO_ROOT
        / "downstream/AffibodyMHC/run_libb_native_projection_primary_boundary_replay.py"
    ).resolve()
    _require(Path(wrapper_source["path"]).resolve() == expected_wrapper,
             "primary boundary replay used an unexpected wrapper")
    _require(invocation.get("input_manifest") == _asset(input_manifest_path),
             "primary boundary invocation used a different input manifest")
    _require(invocation.get("exhaustive_manifest") == _asset(exhaustive_manifest_path),
             "primary boundary invocation used a different exhaustive manifest")
    argv = [str(value) for value in invocation.get("command_argv", [])]
    _require(_command_value(argv, "--family") == "rde"
             and _command_value(argv, "--peptides") == ",".join(EXPECTED_TARGETS)
             and int(_command_value(argv, "--batch-size")) == 16
             and int(_command_value(argv, "--chunk-rows")) == 50,
             "primary boundary invocation command changed")

    worker_path = replay_dir / str(invocation.get("worker_receipt", {}).get("path", ""))
    _require(worker_path.is_file()
             and invocation["worker_receipt"].get("sha256") == _sha256(worker_path),
             "primary boundary worker receipt changed")
    worker = _load_json(worker_path)
    source = exhaustive_rde["source_chunks"]
    _require(worker.get("producer") == source.get("scorer_producer")
             and {key: scorer_source[key] for key in ("path", "sha256")}
             == source.get("scorer_producer"),
             "primary boundary scorer differs from exhaustive scorer")
    _require(worker.get("readout_provenance") == source.get("readout_provenance"),
             "primary boundary readouts differ from exhaustive scoring")
    _require(worker.get("labels_read") is False and worker.get("retention_read") is False,
             "primary boundary worker did not declare label-free inference")
    _require(worker.get("readout_seeds") == list(SEEDS),
             "primary boundary worker seed set changed")

    chunk_records = invocation.get("score_chunks", [])
    _require(len(chunk_records) == 12, "primary boundary invocation lacks 12 score chunks")
    score_frames = []
    for record in chunk_records:
        path = replay_dir / str(record.get("path", ""))
        _require(path.is_file() and record.get("sha256") == _sha256(path)
                 and int(record.get("rows", -1)) == 50,
                 "primary boundary score chunk changed")
        frame = pd.read_csv(path, dtype={"pair_uid": str})
        _require(len(frame) == 50, "primary boundary score chunk row count changed")
        _validate_score_frame(frame, "rde", f"primary boundary/{path.parent.name}")
        score_frames.append(frame)
    scores = pd.concat(score_frames, ignore_index=True)
    _require(len(scores) == 600 and not scores["pair_uid"].duplicated().any(),
             "primary boundary replay does not contain 600 unique scores")
    ordered_inputs = boundary_inputs.sort_values(
        ["peptide_design_code", "candidate_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    ordered_scores = scores.sort_values(
        ["peptide_design_code", "candidate_row_index"], kind="mergesort"
    ).reset_index(drop=True)
    for field in ("candidate_row_index", "pair_uid", "peptide_design_code", "affibody_design_code"):
        _require(ordered_inputs[field].astype(str).equals(ordered_scores[field].astype(str)),
                 f"primary boundary scorer changed ordered {field}")

    alternate = scores[["pair_uid", "score_mean_logit"]].copy()
    alternate["batch16_rde_logit"] = _probability_to_logit(
        pd.to_numeric(alternate["score_mean_logit"], errors="raise").to_numpy(float)
    )
    comparison = review.merge(
        alternate[["pair_uid", "batch16_rde_logit"]], on="pair_uid", validate="one_to_one"
    )
    comparison["batch16_ensemble_linear_score"] = 0.5 * (
        comparison["mint_layer5"] + comparison["batch16_rde_logit"]
    )
    comparison["batch16_ensemble_score"] = _sigmoid(
        comparison["batch16_ensemble_linear_score"].to_numpy(float)
    )
    comparison["rde_logit_absolute_difference"] = (
        comparison["batch16_rde_logit"] - comparison["rde_network_designed_3fold"]
    ).abs()
    comparison["ensemble_logit_absolute_difference"] = (
        comparison["batch16_ensemble_linear_score"] - comparison["ensemble_linear_score"]
    ).abs()

    rows = []
    for peptide, block in comparison.groupby("peptide_design_code", sort=True):
        production = block.sort_values(
            ["ensemble_linear_score", "pair_uid"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        batch16 = block.sort_values(
            ["batch16_ensemble_linear_score", "pair_uid"], ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        production_top10 = tuple(production.loc[:9, "pair_uid"].astype(str))
        batch16_top10 = tuple(batch16.loc[:9, "pair_uid"].astype(str))
        _require(set(production_top10) == set(batch16_top10),
                 f"primary top10 membership changed for {peptide}")
        _require(production_top10 == batch16_top10,
                 f"primary top10 order changed for {peptide}")
        production_gap = float(
            production.loc[9, "ensemble_linear_score"]
            - production.loc[10, "ensemble_linear_score"]
        )
        batch16_gap = float(
            batch16.loc[9, "batch16_ensemble_linear_score"]
            - batch16.loc[10, "batch16_ensemble_linear_score"]
        )
        cutoff_margin = float(batch16.loc[:9, "batch16_ensemble_score"].min() - threshold)
        _require(production_gap > 0 and batch16_gap > 0,
                 f"primary rank10/rank11 gap is not positive for {peptide}")
        _require(cutoff_margin > 0, f"primary top10 crosses the cutoff for {peptide}")
        rows.append({
            "peptide_design_code": peptide,
            "rows_replayed": len(block),
            "top10_membership_identical": True,
            "top10_order_identical": True,
            "production_rank10_vs_rank11_logit_gap": production_gap,
            "batch16_rank10_vs_rank11_logit_gap": batch16_gap,
            "maximum_rde_logit_absolute_difference": float(
                block["rde_logit_absolute_difference"].max()
            ),
            "maximum_weighted_ensemble_logit_absolute_difference": float(
                block["ensemble_logit_absolute_difference"].max()
            ),
            "minimum_selected_probability_margin_above_cutoff": cutoff_margin,
        })
    per_peptide = pd.DataFrame(rows)
    _require(len(per_peptide) == 12 and set(per_peptide["peptide_design_code"])
             == set(EXPECTED_TARGETS), "primary boundary result target set changed")
    return per_peptide, {
        "status": "pass",
        "peptides": 12,
        "candidate_rows_rescored": 600,
        "verified_alternative_batch_size": 16,
        "all_top10_membership_identical": True,
        "all_top10_order_identical": True,
        "minimum_batch16_rank10_vs_rank11_logit_gap": float(
            per_peptide["batch16_rank10_vs_rank11_logit_gap"].min()
        ),
        "maximum_rde_logit_difference_top50": float(
            per_peptide["maximum_rde_logit_absolute_difference"].max()
        ),
        "maximum_weighted_ensemble_logit_difference_top50": float(
            per_peptide["maximum_weighted_ensemble_logit_absolute_difference"].max()
        ),
        "minimum_selected_probability_margin_above_cutoff": float(
            per_peptide["minimum_selected_probability_margin_above_cutoff"].min()
        ),
        "weak_oof_cutoff": threshold,
        "scope_limitation": (
            "This directly repacks the production-ranked top 50 per peptide, not all "
            "4,447,848 candidate pairs. Full-universe merge validation and production replay "
            "cover the upstream exhaustive-score path."
        ),
        "selection_manifest": _asset(selection_manifest_path),
        "production_review_pool": _asset(review_pool_path),
        "boundary_input_manifest": _asset(input_manifest_path),
        "boundary_replay_invocation": _asset(invocation_path),
        "boundary_wrapper": wrapper_source,
        "boundary_worker_receipt": _asset(worker_path),
        "primary_lock": lock_receipt,
    }


def _report(manifest: dict) -> str:
    production = manifest["gates"]["production_batch_replay"]
    historical = manifest["gates"]["historical_tail_compatibility"]
    cross = manifest["gates"]["corrected120_cross_batch_compatibility"]
    boundary = manifest["gates"]["primary_top10_boundary"]
    return "\n".join([
        "# Native-projection candidate-score release gates",
        "",
        "Status: **PASS**.",
        "",
        "This is a numerical/provenance check, not a biological-performance result. No "
        "retention value or binder label was loaded.",
        "",
        "## 1. Exact production-path replay",
        "",
        f"The same scorer and checkpoints were rerun on 640 AF candidates at the "
        f"exhaustive batch sizes (RDE {PRODUCTION_BATCH_SIZES['rde']}, StaB "
        f"{PRODUCTION_BATCH_SIZES['stab']}). The maximum difference from the merged "
        f"exhaustive archive was {max(row['maximum_absolute_probability_difference'] for row in production['families'].values()):.3g}; "
        "all five seed rankings and deployed aggregate rankings were identical.",
        "",
        "## 2. Historical corrected-120 compatibility",
        "",
        "The original eight-shard batch-16/tail-6 context was reconstructed from canonical "
        "label-free rows. Every seed stayed within 2e-6, each deployed aggregate stayed "
        "within 1e-6, and every within-peptide and aggregate-global ranking was identical. "
        f"The largest seed difference was {max(row['maximum_seed_probability_difference'] for row in historical['families'].values()):.3g}.",
        "",
        "## 3. Deliberately different panel batch shape",
        "",
        "Scoring ten rows per peptide produces small GPU batch-shape roundoff, but all seed "
        "within-peptide rankings, deployed aggregate within-peptide/global rankings, and "
        "standalone cutoff decisions remained identical. This is recorded as compatibility, "
        "not bitwise equality.",
        "",
        "## 4. Primary top-ten decision",
        "",
        f"The production top 50 for each peptide were repacked at batch 16. All 12 top-ten "
        f"sets and exact orders were unchanged. The smallest batch-16 rank-10/rank-11 logit "
        f"gap was {boundary['minimum_batch16_rank10_vs_rank11_logit_gap']:.6g}; the smallest "
        f"selected-score margin above the weak-OOF cutoff was "
        f"{boundary['minimum_selected_probability_margin_above_cutoff']:.6g}.",
        "",
        "The current exhaustive score archives and staged candidate menus therefore pass the "
        "release gate. This does not validate biological performance; only new wet-lab "
        "measurements can do that.",
        "",
    ])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-rows", required=True, type=Path)
    parser.add_argument("--label-free-input-dir", required=True, type=Path)
    parser.add_argument("--rde-panel-output-dir", required=True, type=Path)
    parser.add_argument("--stab-panel-output-dir", required=True, type=Path)
    parser.add_argument("--rde-reference-dir", required=True, type=Path)
    parser.add_argument("--stab-reference-dir", required=True, type=Path)
    parser.add_argument("--aggregate-reference", required=True, type=Path)
    parser.add_argument("--retrospective-audit-manifest", required=True, type=Path)
    parser.add_argument("--rde-exhaustive-dir", required=True, type=Path)
    parser.add_argument("--stab-exhaustive-dir", required=True, type=Path)
    parser.add_argument("--rde-production-replay-dir", required=True, type=Path)
    parser.add_argument("--stab-production-replay-dir", required=True, type=Path)
    parser.add_argument("--rde-historical-replay-dir", required=True, type=Path)
    parser.add_argument("--stab-historical-replay-dir", required=True, type=Path)
    parser.add_argument("--rde-standalone-lock", required=True, type=Path)
    parser.add_argument("--stab-standalone-lock", required=True, type=Path)
    parser.add_argument("--primary-boundary-replay-dir", required=True, type=Path)
    parser.add_argument("--primary-boundary-input-dir", required=True, type=Path)
    parser.add_argument("--primary-review-pool", required=True, type=Path)
    parser.add_argument("--primary-lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output = _new_private_dir(args.output_dir)
    producer_start = _asset(Path(__file__))
    imported_auditor_path = Path(panel_audit.__file__).resolve()
    imported_auditor_start = _asset(imported_auditor_path)
    canonical_rows = args.canonical_rows.expanduser().resolve()
    _require(canonical_rows.is_file(), "canonical rows missing")
    sealed_reference_hashes, retrospective_receipt = (
        panel_audit._load_retrospective_audit_manifest(
            args.retrospective_audit_manifest.expanduser().resolve(),
            args.aggregate_reference.expanduser().resolve(),
        )
    )

    exhaustive_payloads = {}
    exhaustive_receipts = {}
    for family in FAMILIES:
        directory = getattr(args, f"{family}_exhaustive_dir")
        payload, receipt = _load_exhaustive(directory, family)
        exhaustive_payloads[family] = payload
        exhaustive_receipts[family] = receipt

    production_rows = []
    production_summaries = {}
    for family in FAMILIES:
        rows, summary = _production_replay(
            family,
            getattr(args, f"{family}_production_replay_dir"),
            getattr(args, f"{family}_exhaustive_dir"),
            exhaustive_payloads[family],
        )
        production_rows.extend(rows)
        production_summaries[family] = summary

    historical_rows = []
    historical_summaries = {}
    for family in FAMILIES:
        rows, summary = _historical_replay(
            family,
            getattr(args, f"{family}_historical_replay_dir"),
            getattr(args, f"{family}_reference_dir"),
            canonical_rows,
            exhaustive_payloads[family],
            getattr(args, f"{family}_exhaustive_dir").expanduser().resolve() / "manifest.json",
            sealed_reference_hashes[family],
        )
        historical_rows.extend(rows)
        historical_summaries[family] = summary

    rde_cutoff, rde_lock_receipt = _validate_lock(
        args.rde_standalone_lock,
        "libb-native-rde-standalone-weak-oof-v1",
        ("rde_network_designed_3fold",),
    )
    stab_cutoff, stab_lock_receipt = _validate_lock(
        args.stab_standalone_lock,
        "libb-native-stab-standalone-weak-oof-v1",
        ("stab_designed_ordered",),
    )
    cross_seed, cross_aggregate, cross_summary = _cross_batch_panel(
        canonical_rows,
        args.label_free_input_dir,
        args.rde_panel_output_dir,
        args.stab_panel_output_dir,
        args.rde_reference_dir,
        args.stab_reference_dir,
        args.aggregate_reference,
        args.retrospective_audit_manifest,
        exhaustive_payloads,
        rde_cutoff,
        stab_cutoff,
    )
    boundary_per_peptide, boundary_summary = _primary_boundary(
        args.primary_boundary_replay_dir,
        args.primary_boundary_input_dir,
        args.primary_review_pool,
        args.primary_lock,
        exhaustive_payloads["rde"],
        args.rde_exhaustive_dir.expanduser().resolve() / "manifest.json",
    )

    manifest = {
        "schema_version": "libb-native-projection-release-gates-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "status": "pass",
        "decision": "native exhaustive scores and primary candidate menu cleared for handoff",
        "thresholds": {
            "production_probability_tolerance": PRODUCTION_TOLERANCE,
            "historical_seed_probability_tolerance": HISTORICAL_TOLERANCE,
            "historical_aggregate_probability_tolerance": HISTORICAL_AGGREGATE_TOLERANCE,
            "cross_batch_probability_tolerance": CROSS_BATCH_TOLERANCE,
        },
        "gates": {
            "production_batch_replay": {
                "status": "pass",
                "interpretation": "strict replay of the actual exhaustive scoring path",
                "families": production_summaries,
            },
            "historical_tail_compatibility": {
                "status": "pass",
                "interpretation": (
                    "historical batch-context compatibility; the 2e-6 seed bound is separate "
                    "from the strict production replay bound"
                ),
                "families": historical_summaries,
            },
            "corrected120_cross_batch_compatibility": cross_summary,
            "primary_top10_boundary": boundary_summary,
        },
        "exhaustive_manifests": exhaustive_receipts,
        "sealed_retrospective_score_provenance": retrospective_receipt,
        "standalone_cutoff_locks": {
            "rde": rde_lock_receipt,
            "stab": stab_lock_receipt,
        },
        "outcome_access": {
            "retention_values_loaded": False,
            "binder_values_loaded": False,
            "selection_labels_loaded_for_release_decision": False,
            "weak_metadata_loaded_for_release_decision": False,
            "aggregate_model_score_columns_loaded": True,
        },
        "producer_start": producer_start,
        "imported_panel_auditor_start": imported_auditor_start,
    }

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    frames = {
        "production_batch_replay.csv": pd.DataFrame(production_rows),
        "historical_tail_compatibility.csv": pd.DataFrame(historical_rows),
        "corrected120_cross_batch_seed.csv": cross_seed,
        "corrected120_cross_batch_aggregate.csv": cross_aggregate,
        "primary_top10_boundary_by_peptide.csv": boundary_per_peptide,
    }
    file_records = {}
    for name, frame in frames.items():
        path = staging / name
        frame.to_csv(path, index=False)
        os.chmod(path, 0o600)
        file_records[name] = {
            "path": name,
            "sha256": _sha256(path),
            "bytes": int(path.stat().st_size),
            "rows": int(len(frame)),
        }
    report_path = staging / "report.md"
    report_path.write_text(_report(manifest), encoding="utf-8")
    os.chmod(report_path, 0o600)
    file_records["report.md"] = {
        "path": report_path.name,
        "sha256": _sha256(report_path),
        "bytes": int(report_path.stat().st_size),
    }
    manifest["outputs"] = file_records

    producer_end = _asset(Path(__file__))
    imported_auditor_end = _asset(imported_auditor_path)
    _require(producer_start == producer_end, "release auditor changed during execution")
    _require(imported_auditor_start == imported_auditor_end,
             "imported panel auditor changed during execution")
    manifest["producer_end"] = producer_end
    manifest["imported_panel_auditor_end"] = imported_auditor_end
    manifest_path = staging / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    os.replace(staging, output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
