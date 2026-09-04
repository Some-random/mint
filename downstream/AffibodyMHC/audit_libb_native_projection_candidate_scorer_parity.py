#!/usr/bin/env python3
"""Audit native RDE/StaB candidate-scorer parity on label-free LibB rows.

The GPU scorer is run separately.  This audit reads only candidate identities,
sequences, and saved model scores.  It never loads retention or binder values.
It compares all five seed probabilities, the sigmoid-of-mean-logit aggregate,
and the complete within-peptide ordering against the sealed native-readout
predictions used in the corrected-120 analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
EXPECTED_TARGETS = ("AF", "AH", "DP", "EA", "EL", "LL", "LV", "MW", "NF", "PH", "TL", "VV")
INPUT_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
)
SCORE_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    *(f"score_seed_{seed}" for seed in SEEDS),
    "score_mean_probability",
    "score_mean_logit",
    "score_seed_sd",
    "model",
)
FAMILY_SPECS = {
    "rde": {
        "candidate_model": "rde_network_designed_3fold_ensemble",
        "reference_model": "rde_network_designed_3fold_ensemble_native_projection",
        "aggregate_column": "rde_network_designed_3fold_native_projection",
        "checkpoint_count": 15,
        "readout_config": REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_native_projection_readouts_v1.json",
        "checkpoint_dir": REPO_ROOT / "private_data/experiments/rde_libb_native_projection_deployment_v1/final_checkpoints",
    },
    "stab": {
        "candidate_model": "stab_designed_ordered",
        "reference_model": "stab_designed_ordered",
        "aggregate_column": "stab_designed_ordered_native_projection",
        "checkpoint_count": 5,
        "readout_config": REPO_ROOT / "downstream/AffibodyMHC/configs/stab_libb_native_projection_readouts_v1.json",
        "checkpoint_dir": REPO_ROOT / "private_data/experiments/stab_libb_native_projection_120_full_v1/final_checkpoints",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_record(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"required provenance asset is missing: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(stat.st_size),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    }


def _git_output(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _git_repo_provenance(repo: Path) -> dict:
    """Fingerprint tracked Python code while reporting, not hiding, dirty state."""

    resolved = repo.expanduser().resolve()
    _require((resolved / ".git").exists(), f"vendor root is not a Git checkout: {resolved}")
    head = _git_output(resolved, "rev-parse", "HEAD").strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", head) is not None,
             f"invalid Git HEAD for {resolved}")
    tracked_status = _git_output(
        resolved, "status", "--porcelain=v1", "--untracked-files=no"
    ).splitlines()
    python_paths = [
        value for value in _git_output(resolved, "ls-files", "-z", "--", "*.py").split("\0")
        if value
    ]
    _require(python_paths, f"vendor checkout has no tracked Python sources: {resolved}")
    digest = hashlib.sha256()
    for relative in sorted(python_paths):
        path = resolved / relative
        _require(path.is_file(), f"tracked vendor Python source is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    return {
        "path": str(resolved),
        "git_head": head,
        "tracked_worktree_clean": not tracked_status,
        "tracked_status_lines": tracked_status,
        "tracked_python_file_count": len(python_paths),
        "tracked_python_tree_sha256": digest.hexdigest(),
    }


def _private_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    _require(output != PRIVATE_ROOT and PRIVATE_ROOT in output.parents,
             "output must be a new directory below private_data")
    _require(not output.exists(), "output directory exists; refusing overwrite")
    return output


def _load_label_free_inputs(input_dir: Path, canonical_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest_path = input_dir / "manifest.json"
    _require(manifest_path.is_file(), "label-free input manifest is missing")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(
        manifest.get("schema_version") == "libb-panel-label-free-scoring-inputs-v1",
        "unexpected label-free input schema",
    )
    _require(int(manifest.get("rows", -1)) == 120, "label-free input count changed")
    _require(manifest.get("retention_columns_written") is False,
             "input manifest does not explicitly exclude retention columns")
    _require(manifest.get("retention_columns_read") is False,
             "input builder does not explicitly declare retention values unread")
    _require(manifest.get("binder_columns_read") is False,
             "input builder does not explicitly declare binder values unread")
    _require(
        manifest.get("panel_columns_read")
        == ["pair_uid", "peptide_design_code", "affibody_design_code"],
        "input builder read columns beyond the three permitted identity columns",
    )
    records = manifest.get("partitions", [])
    _require(len(records) == 12, "label-free input must contain 12 partitions")
    _require({str(row.get("peptide")) for row in records} == set(EXPECTED_TARGETS),
             "label-free input target set changed")
    frames = []
    partition_records = []
    receipt_by_peptide = {str(row.get("peptide")): row for row in records}
    for peptide in EXPECTED_TARGETS:
        path = input_dir / f"peptide_{peptide}.csv.gz"
        _require(path.is_file(), f"missing label-free input for {peptide}")
        frame = pd.read_csv(path, dtype={"pair_uid": str})
        _require(tuple(frame.columns) == INPUT_COLUMNS,
                 f"label-free input columns changed for {peptide}")
        _require(len(frame) == 10, f"label-free input is not ten rows for {peptide}")
        _require(frame["peptide_design_code"].astype(str).eq(peptide).all(),
                 f"mixed peptide input for {peptide}")
        _require(frame["candidate_row_index"].astype(int).tolist() == list(range(10)),
                 f"candidate row indices changed for {peptide}")
        receipt = receipt_by_peptide[peptide]
        _require(receipt.get("path") == path.name,
                 f"input-builder receipt has wrong partition path for {peptide}")
        _require(int(receipt.get("rows", -1)) == len(frame),
                 f"input-builder receipt has wrong row count for {peptide}")
        _require(receipt.get("sha256") == sha256_file(path),
                 f"label-free partition changed after manifest creation for {peptide}")
        frames.append(frame)
        partition_records.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(frame),
        })
    inputs = pd.concat(frames, ignore_index=True)
    _require(len(inputs) == 120 and not inputs["pair_uid"].duplicated().any(),
             "label-free input pair identities are not exactly 120 unique rows")
    bundle_digest = hashlib.sha256()
    for record in sorted(partition_records, key=lambda item: Path(item["path"]).name):
        bundle_digest.update(Path(record["path"]).name.encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(bytes.fromhex(str(record["sha256"])))
        bundle_digest.update(b"\0")
    _require(
        manifest.get("partition_bundle_sha256") == bundle_digest.hexdigest(),
        "label-free partition bundle differs from its manifest",
    )
    amino_alphabet = set("ACDEFGHIKLMNPQRSTVWY")
    for row in inputs.itertuples(index=False):
        chain1 = str(row.chain1_sequence)
        chain2 = str(row.chain2_sequence)
        peptide_code = str(row.peptide_design_code)
        affibody_code = str(row.affibody_design_code)
        _require(len(chain1) >= 9 and len(chain2) > 16,
                 f"sequence is too short for code reconstruction: {row.pair_uid}")
        _require(set(chain1) <= amino_alphabet and set(chain2) <= amino_alphabet,
                 f"sequence contains a noncanonical amino acid: {row.pair_uid}")
        _require(chain1[-9:][3:5] == peptide_code,
                 f"peptide code does not match the full pMHC input: {row.pair_uid}")
        _require("".join(chain2[index] for index in (5, 9, 12, 13, 16)) == affibody_code,
                 f"Affibody code does not match the full Affibody input: {row.pair_uid}")
        expected_uid = hashlib.sha256(
            f"LibB|{peptide_code}|{affibody_code}".encode("utf-8")
        ).hexdigest()[:20]
        _require(str(row.pair_uid) == expected_uid,
                 f"pair_uid does not match biological codes: {row.pair_uid}")
        expected_sequence_hash = hashlib.sha256(
            f"{chain1}|{chain2}".encode("utf-8")
        ).hexdigest()
        _require(str(row.sequence_pair_sha256) == expected_sequence_hash,
                 f"sequence-pair hash does not match sequences: {row.pair_uid}")

    with canonical_path.open(encoding="utf-8") as handle:
        canonical_payload = json.load(handle)
    _require(canonical_payload.get("schema_version") == "esmfold2-libb-canonical-rows-v1",
             "unexpected canonical-row schema")
    canonical = pd.DataFrame(canonical_payload.get("rows", []))
    canonical = canonical.loc[canonical["split"].astype(str).eq("eval")].copy()
    _require(len(canonical) == 120 and not canonical["row_id"].astype(str).duplicated().any(),
             "canonical evaluation roster is not 120 unique rows")
    expected = canonical[[
        "row_id", "chain1_sequence", "chain2_sequence", "sequence_pair_sha256"
    ]].rename(columns={"row_id": "pair_uid"})
    expected["pair_uid"] = expected["pair_uid"].astype(str)
    joined = inputs.merge(expected, on="pair_uid", how="left", suffixes=("", "_canonical"),
                          validate="one_to_one")
    for column in ("chain1_sequence", "chain2_sequence", "sequence_pair_sha256"):
        _require(
            joined[column].astype(str).eq(joined[f"{column}_canonical"].astype(str)).all(),
            f"label-free input {column} differs from canonical rows",
        )
    source_inputs = manifest.get("input_sources", {})
    canonical_source = source_inputs.get("canonical_rows", {})
    _require(
        canonical_source.get("sha256") == sha256_file(canonical_path),
        "input-builder receipt does not match the audited canonical-row file",
    )
    producer = manifest.get("producer", {})
    expected_producer = (
        REPO_ROOT / "downstream/AffibodyMHC/prepare_libb_panel_scoring_inputs.py"
    ).resolve()
    _require(Path(str(producer.get("path", ""))).resolve() == expected_producer,
             "label-free input-builder receipt has the wrong producer")
    _require(producer.get("sha256") == sha256_file(expected_producer),
             "label-free input-builder source changed after fixture generation")
    return inputs, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "canonical_path": str(canonical_path.resolve()),
        "canonical_sha256": sha256_file(canonical_path),
        "partitions": partition_records,
        "input_builder": producer,
        "source_inputs": source_inputs,
        "outcome_columns_read_by_input_builder": False,
    }


def _runtime_asset_provenance(args: argparse.Namespace) -> dict:
    scorer = REPO_ROOT / "downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py"
    runner = (
        REPO_ROOT
        / "downstream/AffibodyMHC/run_libb_native_projection_candidate_scorer_parity.sh"
    )
    return {
        "fixed_crystal_pdb": _asset_record(args.pdb),
        "residue_mapping": _asset_record(args.residue_mapping),
        "candidate_scorer_source": _asset_record(scorer),
        "parity_runner_source": _asset_record(runner),
        "parity_auditor_source": _asset_record(Path(__file__)),
        "rde": {
            "vendor_repository": _git_repo_provenance(args.rde_root),
            "unsupervised_checkpoint": _asset_record(args.rde_vendor_checkpoint),
            "mutation_trained_network_checkpoint": _asset_record(
                args.rde_network_vendor_checkpoint
            ),
            "native_readout_config": _asset_record(FAMILY_SPECS["rde"]["readout_config"]),
            "native_readout_checkpoints": [
                _asset_record(path)
                for path in sorted(Path(FAMILY_SPECS["rde"]["checkpoint_dir"]).glob("*.pt"))
            ],
        },
        "stab": {
            "vendor_repository": _git_repo_provenance(args.stab_root),
            "vendor_checkpoint": _asset_record(args.stab_vendor_checkpoint),
            "native_readout_config": _asset_record(FAMILY_SPECS["stab"]["readout_config"]),
            "native_readout_checkpoints": [
                _asset_record(path)
                for path in sorted(Path(FAMILY_SPECS["stab"]["checkpoint_dir"]).glob("*.pt"))
            ],
        },
        "receipt_scope_limitation": (
            "The scorer worker receipt cryptographically binds its label-free input manifest, "
            "producer source, native-readout config, and native-readout checkpoints. The current "
            "production scorer receipt does not serialize the PDB, residue mapping, or upstream "
            "vendor paths/hashes. This harness fingerprints those runtime assets and invokes them "
            "explicitly; exact parity to sealed scores is the behavioral check on that remaining "
            "dependency surface."
        ),
    }


def _load_candidate_scores(
    output_dir: Path,
    family: str,
    input_manifest_sha256: str,
    tolerance: float,
) -> tuple[pd.DataFrame, dict]:
    spec = FAMILY_SPECS[family]
    manifest_paths = sorted(output_dir.glob(f"worker_{family}_slice00of01_*.json"))
    _require(len(manifest_paths) == 1, f"expected one {family} candidate-scorer manifest")
    manifest_path = manifest_paths[0]
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema_version") == "libb-fixed-structure-candidate-scores-v1",
             f"unexpected {family} candidate-score schema")
    _require(manifest.get("family") == family, f"wrong family in {family} manifest")
    _require(manifest.get("model") == spec["candidate_model"],
             f"wrong model in {family} manifest")
    _require(manifest.get("labels_read") is False and manifest.get("retention_read") is False,
             f"{family} scorer does not explicitly declare label-free inference")
    _require(manifest.get("readout_seeds") == list(SEEDS),
             f"{family} readout seed set changed")
    _require(manifest.get("slice_index") == 0 and manifest.get("num_slices") == 1,
             f"{family} parity output is not one complete slice")
    _require(manifest.get("input_manifest_sha256") == input_manifest_sha256,
             f"{family} scorer used a different label-free input manifest")
    readout = manifest.get("readout_provenance", {})
    _require(readout.get("readout_mode") == "native_learned_projection",
             f"{family} parity output is not native learned projection")
    _require(len(readout.get("checkpoints", [])) == int(spec["checkpoint_count"]),
             f"{family} checkpoint count changed")
    expected_config = Path(spec["readout_config"]).resolve()
    expected_checkpoints = Path(spec["checkpoint_dir"]).resolve()
    _require(Path(str(readout.get("config_path", ""))).resolve() == expected_config,
             f"{family} scorer used the wrong readout config")
    _require(readout.get("config_sha256") == sha256_file(expected_config),
             f"{family} readout config hash changed")
    _require(Path(str(readout.get("checkpoint_directory", ""))).resolve() == expected_checkpoints,
             f"{family} scorer used the wrong checkpoint directory")
    expected_checkpoint_records = {
        path.name: sha256_file(path) for path in sorted(expected_checkpoints.glob("*.pt"))
    }
    observed_checkpoint_records = {
        str(row.get("filename")): str(row.get("sha256"))
        for row in readout.get("checkpoints", [])
    }
    _require(observed_checkpoint_records == expected_checkpoint_records,
             f"{family} scorer checkpoint hashes differ from deployment checkpoints")
    producer = manifest.get("producer", {})
    expected_producer = (REPO_ROOT / "downstream/AffibodyMHC/score_libb_fixed_structure_candidates.py").resolve()
    _require(Path(str(producer.get("path", ""))).resolve() == expected_producer,
             f"{family} scorer manifest has wrong producer")
    _require(producer.get("sha256") == sha256_file(expected_producer),
             f"{family} candidate-scorer code hash changed after inference")
    peptide_records = manifest.get("peptides", [])
    _require(len(peptide_records) == 12, f"{family} manifest lacks peptide records")
    _require({str(row.get("peptide")) for row in peptide_records} == set(EXPECTED_TARGETS),
             f"{family} manifest target set changed")

    frames = []
    chunk_records = []
    for peptide in EXPECTED_TARGETS:
        paths = sorted((output_dir / f"peptide_{peptide}").glob("*.csv.gz"))
        _require(len(paths) == 1, f"expected one {family} score chunk for {peptide}")
        path = paths[0]
        frame = pd.read_csv(path, dtype={"pair_uid": str})
        _require(tuple(frame.columns) == SCORE_COLUMNS,
                 f"{family}/{peptide} candidate-score columns changed")
        _require(len(frame) == 10, f"{family}/{peptide} is not ten scored rows")
        _require(frame["peptide_design_code"].astype(str).eq(peptide).all(),
                 f"{family}/{peptide} contains a mixed peptide")
        _require(frame["candidate_row_index"].astype(int).tolist() == list(range(10)),
                 f"{family}/{peptide} candidate row indices changed")
        _require(frame["model"].astype(str).eq(spec["candidate_model"]).all(),
                 f"{family}/{peptide} has wrong model name")
        frames.append(frame)
        chunk_records.append({
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": len(frame),
        })
    scores = pd.concat(frames, ignore_index=True)
    _require(len(scores) == 120 and not scores["pair_uid"].duplicated().any(),
             f"{family} scorer output is not 120 unique pairs")
    probability_columns = [f"score_seed_{seed}" for seed in SEEDS]
    probabilities = scores[probability_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    _require(np.isfinite(probabilities).all(), f"{family} has non-finite seed scores")
    _require(((probabilities >= 0.0) & (probabilities <= 1.0)).all(),
             f"{family} seed score leaves [0,1]")
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    expected = 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))
    mean_probability = pd.to_numeric(scores["score_mean_probability"], errors="raise").to_numpy(float)
    stored_mean_logit = pd.to_numeric(scores["score_mean_logit"], errors="raise").to_numpy(float)
    stored_sd = pd.to_numeric(scores["score_seed_sd"], errors="raise").to_numpy(float)
    _require(np.isfinite(mean_probability).all(),
             f"{family} has non-finite score_mean_probability")
    _require(np.isfinite(stored_mean_logit).all(),
             f"{family} has non-finite score_mean_logit")
    _require(np.isfinite(stored_sd).all(), f"{family} has non-finite score_seed_sd")
    _require(((mean_probability >= 0.0) & (mean_probability <= 1.0)).all(),
             f"{family} score_mean_probability leaves [0,1]")
    _require(((stored_mean_logit >= 0.0) & (stored_mean_logit <= 1.0)).all(),
             f"{family} score_mean_logit leaves [0,1]")
    _require((stored_sd >= 0.0).all(), f"{family} score_seed_sd is negative")
    _require(np.allclose(
        probabilities.mean(axis=1),
        mean_probability,
        rtol=0.0,
        atol=tolerance,
    ), f"{family} score_mean_probability is not the seed probability mean")
    _require(np.allclose(
        expected,
        stored_mean_logit,
        rtol=0.0,
        atol=tolerance,
    ), f"{family} score_mean_logit is not sigmoid(mean seed logits)")
    _require(np.allclose(
        probabilities.std(axis=1, ddof=1),
        stored_sd,
        rtol=0.0,
        atol=tolerance,
    ), f"{family} score_seed_sd is not the sample SD across seed probabilities")
    return scores, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "input_manifest_sha256": str(manifest["input_manifest_sha256"]),
        "readout_provenance": readout,
        "producer": producer,
        "chunks": chunk_records,
    }


def _load_reference_scores(reference_dir: Path, family: str) -> tuple[pd.DataFrame, list[dict]]:
    spec = FAMILY_SPECS[family]
    blocks = []
    records = []
    if family == "rde":
        pattern = "rde_network_designed_3fold_ensemble_native_projection__seed{seed}.csv"
    else:
        pattern = "stab_designed_ordered__seed{seed}.csv"
    for seed in SEEDS:
        path = reference_dir / pattern.format(seed=seed)
        _require(path.is_file(), f"missing sealed {family} reference for seed {seed}")
        frame = pd.read_csv(path, dtype={"eval_row_id": str})
        _require(tuple(frame.columns) == ("eval_row_id", "model", "seed", "score"),
                 f"sealed {family}/{seed} schema changed")
        _require(len(frame) == 120 and not frame["eval_row_id"].duplicated().any(),
                 f"sealed {family}/{seed} is not 120 unique rows")
        _require(frame["model"].astype(str).eq(spec["reference_model"]).all(),
                 f"sealed {family}/{seed} has wrong model name")
        _require(pd.to_numeric(frame["seed"], errors="raise").eq(seed).all(),
                 f"sealed {family}/{seed} has wrong seed")
        reference_score = pd.to_numeric(frame["score"], errors="raise").to_numpy(float)
        _require(np.isfinite(reference_score).all(),
                 f"sealed {family}/{seed} has non-finite scores")
        _require(((reference_score >= 0.0) & (reference_score <= 1.0)).all(),
                 f"sealed {family}/{seed} score leaves [0,1]")
        frame["score"] = reference_score
        blocks.append(frame[["eval_row_id", "score"]].rename(
            columns={"eval_row_id": "pair_uid", "score": f"reference_seed_{seed}"}
        ))
        records.append({"path": str(path.resolve()), "sha256": sha256_file(path), "rows": 120})
    merged = blocks[0]
    for block in blocks[1:]:
        merged = merged.merge(block, on="pair_uid", how="inner", validate="one_to_one")
    _require(len(merged) == 120, f"sealed {family} seed files have different identities")
    return merged, records


def _load_aggregate_reference(path: Path) -> tuple[pd.DataFrame, dict]:
    # Deliberately load only identity and model-score columns.  The source CSV
    # also stores retrospective outcome fields, but their values never enter
    # this process.
    usecols = [
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        FAMILY_SPECS["rde"]["aggregate_column"],
        FAMILY_SPECS["stab"]["aggregate_column"],
    ]
    header = pd.read_csv(path, nrows=0)
    _require(set(usecols).issubset(header.columns), "aggregate score reference schema changed")
    frame = pd.read_csv(path, usecols=usecols, dtype={"eval_row_id": str})
    _require(len(frame) == 120 and not frame["eval_row_id"].duplicated().any(),
             "aggregate score reference is not 120 unique rows")
    for family in ("rde", "stab"):
        column = str(FAMILY_SPECS[family]["aggregate_column"])
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        _require(np.isfinite(values).all(),
                 f"saved corrected-120 {family} aggregate has non-finite scores")
        _require(((values >= 0.0) & (values <= 1.0)).all(),
                 f"saved corrected-120 {family} aggregate leaves [0,1]")
        frame[column] = values
    return frame.rename(columns={"eval_row_id": "pair_uid"}), {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "columns_loaded": usecols,
        "pair_level_retention_or_binder_columns_loaded": False,
    }


def _load_retrospective_audit_manifest(
    path: Path,
    aggregate_reference_path: Path,
) -> tuple[dict[str, dict[str, str]], dict]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(
        payload.get("schema_version")
        == "libb-native-projection-ensemble-retention-audit-v1",
        "unexpected retrospective audit-manifest schema",
    )
    aggregate_record = payload.get("outputs", {}).get("averaged_score_predictions.csv", {})
    aggregate_resolved = aggregate_reference_path.expanduser().resolve()
    _require(
        aggregate_record.get("sha256") == sha256_file(aggregate_resolved),
        "aggregate score reference is not the artifact sealed by the audit manifest",
    )
    expected: dict[str, dict[str, str]] = {}
    for family in ("rde", "stab"):
        records = payload.get("inputs", {}).get(f"{family}_files", [])
        _require(len(records) == 5, f"audit manifest does not seal five {family} seed files")
        mapping = {Path(str(row.get("path"))).name: str(row.get("sha256")) for row in records}
        _require(len(mapping) == 5 and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in mapping.values()),
                 f"audit manifest has invalid {family} seed-file records")
        expected[family] = mapping
    return expected, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema_version": payload.get("schema_version"),
        "aggregate_reference_sha256": aggregate_record.get("sha256"),
    }


def _load_exhaustive_manifest(path: Path, family: str) -> tuple[dict, dict]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(
        payload.get("schema_version") == "libb-fixed-structure-candidate-scores-merged-v1",
        f"unexpected exhaustive {family} merged-score schema",
    )
    _require(payload.get("family") == family,
             f"exhaustive merged-score manifest has wrong {family} family")
    _require(payload.get("model") == FAMILY_SPECS[family]["candidate_model"],
             f"exhaustive merged-score manifest has wrong {family} model")
    _require(payload.get("readout_seeds") == list(SEEDS),
             f"exhaustive merged-score manifest has wrong {family} seed set")
    _require(int(payload.get("candidate_universe", {}).get("candidate_rows", -1)) == 4_447_848,
             f"exhaustive {family} merged scores do not cover the full candidate universe")
    source = payload.get("source_chunks", {})
    _require(source.get("readout_provenance") is not None,
             f"exhaustive {family} manifest lacks readout provenance")
    _require(source.get("scorer_producer") is not None,
             f"exhaustive {family} manifest lacks scorer-source provenance")
    return payload, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "family": family,
        "model": payload.get("model"),
        "candidate_rows": int(payload["candidate_universe"]["candidate_rows"]),
    }


def _ranked_ids(frame: pd.DataFrame, score_column: str) -> tuple[str, ...]:
    ranked = frame.sort_values(
        [score_column, "pair_uid"], ascending=[False, True], kind="mergesort"
    )
    return tuple(ranked["pair_uid"].astype(str))


def _audit_family(
    family: str,
    inputs: pd.DataFrame,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    aggregate_reference: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = inputs[["pair_uid", "peptide_design_code", "affibody_design_code"]].copy()
    candidate_keys = candidate[[
        "candidate_row_index", "pair_uid", "peptide_design_code", "affibody_design_code"
    ]].copy()
    input_keys = inputs[[
        "candidate_row_index", "pair_uid", "peptide_design_code", "affibody_design_code"
    ]].copy()
    for column in (
        "candidate_row_index", "pair_uid", "peptide_design_code", "affibody_design_code"
    ):
        _require(candidate_keys[column].astype(str).equals(input_keys[column].astype(str)),
                 f"{family} candidate scorer changed ordered {column} mapping")
    joined = keys.merge(candidate, on=["pair_uid", "peptide_design_code", "affibody_design_code"],
                        validate="one_to_one")
    joined = joined.merge(reference, on="pair_uid", validate="one_to_one")
    _require(len(joined) == 120, f"{family} parity join is incomplete")
    seed_rows = []
    difference_columns = ["pair_uid", "peptide_design_code", "affibody_design_code"]
    for seed in SEEDS:
        candidate_column = f"score_seed_{seed}"
        reference_column = f"reference_seed_{seed}"
        difference_column = f"absolute_difference_seed_{seed}"
        joined[difference_column] = (
            pd.to_numeric(joined[candidate_column], errors="raise")
            - pd.to_numeric(joined[reference_column], errors="raise")
        ).abs()
        _require(np.isfinite(joined[difference_column].to_numpy(float)).all(),
                 f"{family}/{seed} seed comparison contains a non-finite difference")
        rank_matches = []
        for _, block in joined.groupby("peptide_design_code", sort=True):
            rank_matches.append(
                _ranked_ids(block, candidate_column) == _ranked_ids(block, reference_column)
            )
        max_abs = float(joined[difference_column].max())
        seed_rows.append({
            "family": family,
            "seed": seed,
            "rows": len(joined),
            "maximum_absolute_probability_difference": max_abs,
            "mean_absolute_probability_difference": float(joined[difference_column].mean()),
            "within_tolerance": max_abs <= tolerance,
            "all_12_within_peptide_rankings_identical": all(rank_matches),
        })
        difference_columns.extend([candidate_column, reference_column, difference_column])

    candidate_probabilities = joined[[f"score_seed_{seed}" for seed in SEEDS]].to_numpy(float)
    reference_probabilities = joined[[f"reference_seed_{seed}" for seed in SEEDS]].to_numpy(float)
    candidate_clipped = np.clip(candidate_probabilities, 1e-7, 1.0 - 1e-7)
    reference_clipped = np.clip(reference_probabilities, 1e-7, 1.0 - 1e-7)
    joined["recomputed_candidate_sigmoid_mean_seed_logit"] = 1.0 / (
        1.0 + np.exp(-(np.log(candidate_clipped) - np.log1p(-candidate_clipped)).mean(axis=1))
    )
    joined["recomputed_reference_sigmoid_mean_seed_logit"] = 1.0 / (
        1.0 + np.exp(-(np.log(reference_clipped) - np.log1p(-reference_clipped)).mean(axis=1))
    )
    aggregate_column = str(FAMILY_SPECS[family]["aggregate_column"])
    aggregate = aggregate_reference[["pair_uid", aggregate_column]].copy()
    joined = joined.merge(aggregate, on="pair_uid", validate="one_to_one")
    joined["stored_candidate_vs_seed_reference_aggregate_abs_difference"] = (
        pd.to_numeric(joined["score_mean_logit"], errors="raise")
        - joined["recomputed_reference_sigmoid_mean_seed_logit"]
    ).abs()
    joined["stored_candidate_vs_saved_audit_aggregate_abs_difference"] = (
        pd.to_numeric(joined["score_mean_logit"], errors="raise")
        - pd.to_numeric(joined[aggregate_column], errors="raise")
    ).abs()
    joined["sealed_seed_aggregate_vs_saved_audit_aggregate_abs_difference"] = (
        joined["recomputed_reference_sigmoid_mean_seed_logit"]
        - pd.to_numeric(joined[aggregate_column], errors="raise")
    ).abs()
    joined["recomputed_candidate_vs_stored_candidate_aggregate_abs_difference"] = (
        joined["recomputed_candidate_sigmoid_mean_seed_logit"]
        - pd.to_numeric(joined["score_mean_logit"], errors="raise")
    ).abs()
    difference_names = [
        "stored_candidate_vs_seed_reference_aggregate_abs_difference",
        "stored_candidate_vs_saved_audit_aggregate_abs_difference",
        "sealed_seed_aggregate_vs_saved_audit_aggregate_abs_difference",
        "recomputed_candidate_vs_stored_candidate_aggregate_abs_difference",
    ]
    _require(np.isfinite(joined[difference_names].to_numpy(float)).all(),
             f"{family} aggregate comparison contains a non-finite difference")
    rank_stored_vs_seed = []
    rank_stored_vs_audit = []
    rank_seed_vs_audit = []
    for _, block in joined.groupby("peptide_design_code", sort=True):
        rank_stored_vs_seed.append(_ranked_ids(
            block, "score_mean_logit"
        ) == _ranked_ids(block, "recomputed_reference_sigmoid_mean_seed_logit"))
        rank_stored_vs_audit.append(_ranked_ids(
            block, "score_mean_logit"
        ) == _ranked_ids(block, aggregate_column))
        rank_seed_vs_audit.append(_ranked_ids(
            block, "recomputed_reference_sigmoid_mean_seed_logit"
        ) == _ranked_ids(block, aggregate_column))
    global_rank_stored_vs_seed = (
        _ranked_ids(joined, "score_mean_logit")
        == _ranked_ids(joined, "recomputed_reference_sigmoid_mean_seed_logit")
    )
    global_rank_stored_vs_audit = (
        _ranked_ids(joined, "score_mean_logit") == _ranked_ids(joined, aggregate_column)
    )
    global_rank_seed_vs_audit = (
        _ranked_ids(joined, "recomputed_reference_sigmoid_mean_seed_logit")
        == _ranked_ids(joined, aggregate_column)
    )
    aggregate_row = pd.DataFrame([{
        "family": family,
        "rows": len(joined),
        "maximum_stored_candidate_difference_vs_recomputed_sealed_seed_aggregate": float(
            joined["stored_candidate_vs_seed_reference_aggregate_abs_difference"].max()
        ),
        "maximum_stored_candidate_difference_vs_saved_corrected120_score_column": float(
            joined["stored_candidate_vs_saved_audit_aggregate_abs_difference"].max()
        ),
        "maximum_recomputed_sealed_difference_vs_saved_corrected120_score_column": float(
            joined["sealed_seed_aggregate_vs_saved_audit_aggregate_abs_difference"].max()
        ),
        "maximum_recomputed_candidate_difference_vs_stored_candidate_aggregate": float(
            joined["recomputed_candidate_vs_stored_candidate_aggregate_abs_difference"].max()
        ),
        "stored_candidate_within_tolerance_vs_recomputed_sealed_seed_aggregate": bool(
            joined["stored_candidate_vs_seed_reference_aggregate_abs_difference"].max()
            <= tolerance
        ),
        "stored_candidate_within_tolerance_vs_saved_corrected120_score_column": bool(
            joined["stored_candidate_vs_saved_audit_aggregate_abs_difference"].max()
            <= tolerance
        ),
        "recomputed_sealed_within_tolerance_vs_saved_corrected120_score_column": bool(
            joined["sealed_seed_aggregate_vs_saved_audit_aggregate_abs_difference"].max()
            <= tolerance
        ),
        "all_12_stored_candidate_rankings_identical_vs_recomputed_sealed_seed_aggregate": all(
            rank_stored_vs_seed
        ),
        "all_12_stored_candidate_rankings_identical_vs_saved_corrected120_score_column": all(
            rank_stored_vs_audit
        ),
        "all_12_recomputed_sealed_rankings_identical_vs_saved_corrected120_score_column": all(
            rank_seed_vs_audit
        ),
        "all_120_stored_candidate_global_ranking_identical_vs_recomputed_sealed_seed_aggregate": bool(
            global_rank_stored_vs_seed
        ),
        "all_120_stored_candidate_global_ranking_identical_vs_saved_corrected120_score_column": bool(
            global_rank_stored_vs_audit
        ),
        "all_120_recomputed_sealed_global_ranking_identical_vs_saved_corrected120_score_column": bool(
            global_rank_seed_vs_audit
        ),
    }])
    pair_columns = difference_columns + [
        "score_mean_probability",
        "score_mean_logit",
        "score_seed_sd",
        "recomputed_candidate_sigmoid_mean_seed_logit",
        "recomputed_reference_sigmoid_mean_seed_logit",
        aggregate_column,
        *difference_names,
    ]
    pair = joined[pair_columns].copy()
    pair.insert(0, "family", family)
    seed = pd.DataFrame(seed_rows)
    _require(seed["within_tolerance"].all(), f"{family} seed parity exceeds tolerance")
    _require(seed["all_12_within_peptide_rankings_identical"].all(),
             f"{family} seed parity changes a within-peptide ranking")
    _require(bool(aggregate_row.filter(like="within_tolerance").iloc[0].all()),
             f"{family} aggregate parity exceeds tolerance")
    _require(bool(aggregate_row.filter(like="rankings_identical").iloc[0].all()),
             f"{family} aggregate parity changes a within-peptide ranking")
    _require(bool(aggregate_row.filter(like="global_ranking_identical").iloc[0].all()),
             f"{family} aggregate parity changes the global 120-pair ranking")
    return seed, aggregate_row, pair


def _report(
    seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    tolerance: float,
    runtime_provenance: dict,
) -> str:
    lines = [
        "# Native candidate-scorer parity check",
        "",
        "This smoke test sends the corrected 120 LibB identities and sequences through "
        "the exact RDE and StaB scorer used for exhaustive candidate scoring. The input "
        "fixture contains no retention or binder columns. Candidate scores are compared "
        "with the sealed native-readout predictions from the earlier 120-pair run.",
        "",
        f"The required absolute probability tolerance is `{tolerance:g}`. Passing also "
        "requires the complete ten-Affibody order to be identical for every peptide, "
        "for every seed and for the five-seed logit aggregate.",
        "",
        "| Family | Seed | Maximum absolute difference | Mean absolute difference | All 12 peptide rankings identical |",
        "|---|---:|---:|---:|---|",
    ]
    for row in seed.itertuples(index=False):
        lines.append(
            f"| {row.family.upper()} | {row.seed} | "
            f"{row.maximum_absolute_probability_difference:.3g} | "
            f"{row.mean_absolute_probability_difference:.3g} | "
            f"{'yes' if row.all_12_within_peptide_rankings_identical else 'no'} |"
        )
    lines.extend([
        "",
        "`score_mean_logit` in the production scorer is historically named: it is a "
        "probability calculated as `sigmoid(mean(logit(seed probability)))`, not a raw "
        "logit. The audit recomputes this value from the five seed columns.",
        "",
        "| Family | Stored production score vs sealed seed aggregate | Stored production score vs saved corrected-120 score | Sealed seed aggregate vs saved corrected-120 score | All aggregate rankings identical |",
        "|---|---:|---:|---:|---|",
    ])
    for row in aggregate.itertuples(index=False):
        identical = (
            row.all_12_stored_candidate_rankings_identical_vs_recomputed_sealed_seed_aggregate
            and row.all_12_stored_candidate_rankings_identical_vs_saved_corrected120_score_column
            and row.all_12_recomputed_sealed_rankings_identical_vs_saved_corrected120_score_column
        )
        lines.append(
            f"| {row.family.upper()} | "
            f"{row.maximum_stored_candidate_difference_vs_recomputed_sealed_seed_aggregate:.3g} | "
            f"{row.maximum_stored_candidate_difference_vs_saved_corrected120_score_column:.3g} | "
            f"{row.maximum_recomputed_sealed_difference_vs_saved_corrected120_score_column:.3g} | "
            f"{'yes' if identical else 'no'} |"
        )
    lines.extend([
        "",
        "The saved corrected-120 CSV contains outcome columns, but this program requests "
        "only the pair identifier and the two precomputed model-score columns. No "
        "retention or binder value is loaded or used. A passing result therefore checks "
        "deployment-path score parity; it does not recalculate biological performance.",
        "",
        "## Runtime provenance",
        "",
        "The artifact fingerprints the canonical rows, every label-free input partition, "
        "reference crystal, residue mapping, RDE and StaB vendor checkpoints and tracked "
        "Python source trees, native readout configurations/checkpoints, and scorer source. "
        "Vendor repositories are recorded with both Git commit and tracked dirty state.",
        "",
        "The parity receipt is also cross-checked against the completed exhaustive RDE and "
        "StaB merged-score manifests, including the scorer source, native-readout mode, "
        "configuration, and ordered checkpoint hashes.",
        "",
        str(runtime_provenance["receipt_scope_limitation"]),
        "",
        "Each corrected-panel peptide has ten rows, so this check exercises effective GPU "
        "batches of ten. It validates candidate-format feature and score parity, but by itself "
        "does not test numerical invariance across the larger batches used by exhaustive scoring.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--canonical-rows", required=True, type=Path)
    parser.add_argument("--rde-output-dir", required=True, type=Path)
    parser.add_argument("--stab-output-dir", required=True, type=Path)
    parser.add_argument("--rde-reference-dir", required=True, type=Path)
    parser.add_argument("--stab-reference-dir", required=True, type=Path)
    parser.add_argument("--aggregate-reference", required=True, type=Path)
    parser.add_argument("--retrospective-audit-manifest", required=True, type=Path)
    parser.add_argument("--rde-exhaustive-manifest", required=True, type=Path)
    parser.add_argument("--stab-exhaustive-manifest", required=True, type=Path)
    parser.add_argument("--pdb", required=True, type=Path)
    parser.add_argument("--residue-mapping", required=True, type=Path)
    parser.add_argument("--rde-root", required=True, type=Path)
    parser.add_argument("--rde-vendor-checkpoint", required=True, type=Path)
    parser.add_argument("--rde-network-vendor-checkpoint", required=True, type=Path)
    parser.add_argument("--stab-root", required=True, type=Path)
    parser.add_argument("--stab-vendor-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    _require(
        np.isfinite(args.absolute_tolerance) and args.absolute_tolerance > 0,
        "absolute tolerance must be finite and positive",
    )
    output = _private_output(args.output_dir)
    input_dir = args.input_dir.expanduser().resolve()
    canonical = args.canonical_rows.expanduser().resolve()
    inputs, input_provenance = _load_label_free_inputs(input_dir, canonical)
    runtime_provenance = _runtime_asset_provenance(args)
    aggregate_path = args.aggregate_reference.expanduser().resolve()
    sealed_reference_hashes, retrospective_manifest_provenance = (
        _load_retrospective_audit_manifest(
            args.retrospective_audit_manifest,
            aggregate_path,
        )
    )
    aggregate_reference, aggregate_provenance = _load_aggregate_reference(
        aggregate_path
    )
    exhaustive_payloads = {}
    exhaustive_provenance = {}
    for family in ("rde", "stab"):
        payload, record = _load_exhaustive_manifest(
            getattr(args, f"{family}_exhaustive_manifest"), family
        )
        exhaustive_payloads[family] = payload
        exhaustive_provenance[family] = record
    _require(set(aggregate_reference["pair_uid"].astype(str)) == set(inputs["pair_uid"].astype(str)),
             "aggregate reference identities differ from label-free input")
    identity_check = inputs[[
        "pair_uid", "peptide_design_code", "affibody_design_code"
    ]].merge(
        aggregate_reference[[
            "pair_uid", "peptide_design_code", "affibody_design_code"
        ]],
        on="pair_uid",
        suffixes=("_input", "_reference"),
        validate="one_to_one",
    )
    for field in ("peptide_design_code", "affibody_design_code"):
        _require(
            identity_check[f"{field}_input"].astype(str).eq(
                identity_check[f"{field}_reference"].astype(str)
            ).all(),
            f"aggregate reference {field} mapping differs from label-free input",
        )

    seeds = []
    aggregates = []
    pairs = []
    family_provenance = {}
    for family in ("rde", "stab"):
        score_dir = getattr(args, f"{family}_output_dir").expanduser().resolve()
        reference_dir = getattr(args, f"{family}_reference_dir").expanduser().resolve()
        candidate, scorer_provenance = _load_candidate_scores(
            score_dir,
            family,
            input_provenance["manifest_sha256"],
            args.absolute_tolerance,
        )
        exhaustive_source = exhaustive_payloads[family]["source_chunks"]
        _require(
            scorer_provenance["readout_provenance"]
            == exhaustive_source["readout_provenance"],
            f"{family} parity and exhaustive runs use different readout assets",
        )
        _require(
            scorer_provenance["producer"] == exhaustive_source["scorer_producer"],
            f"{family} parity and exhaustive runs use different scorer source",
        )
        _require(set(candidate["pair_uid"].astype(str)) == set(inputs["pair_uid"].astype(str)),
                 f"{family} candidate-score identities differ from input")
        reference, reference_provenance = _load_reference_scores(reference_dir, family)
        observed_reference_hashes = {
            Path(str(record["path"])).name: str(record["sha256"])
            for record in reference_provenance
        }
        _require(
            observed_reference_hashes == sealed_reference_hashes[family],
            f"{family} seed references are not the files sealed by the audit manifest",
        )
        _require(set(reference["pair_uid"].astype(str)) == set(inputs["pair_uid"].astype(str)),
                 f"{family} sealed-reference identities differ from input")
        seed, aggregate, pair = _audit_family(
            family,
            inputs,
            candidate,
            reference,
            aggregate_reference,
            args.absolute_tolerance,
        )
        seeds.append(seed)
        aggregates.append(aggregate)
        pairs.append(pair)
        family_provenance[family] = {
            "candidate_scorer": scorer_provenance,
            "sealed_seed_references": reference_provenance,
        }
    seed_results = pd.concat(seeds, ignore_index=True)
    aggregate_results = pd.concat(aggregates, ignore_index=True)
    pair_results = pd.concat(pairs, ignore_index=True)

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    os.chmod(staging, 0o700)
    output_paths = {
        "per_seed_parity": staging / "per_seed_parity.csv",
        "aggregate_parity": staging / "aggregate_parity.csv",
        "per_pair_differences": staging / "per_pair_differences.csv",
    }
    output_frames = {
        "per_seed_parity": seed_results,
        "aggregate_parity": aggregate_results,
        "per_pair_differences": pair_results,
    }
    for name, frame in output_frames.items():
        frame.to_csv(output_paths[name], index=False)
        os.chmod(output_paths[name], 0o600)
    report_path = staging / "report.md"
    report_path.write_text(
        _report(
            seed_results,
            aggregate_results,
            args.absolute_tolerance,
            runtime_provenance,
        ),
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    manifest = {
        "schema_version": "libb-native-projection-candidate-scorer-parity-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "absolute_probability_tolerance": args.absolute_tolerance,
        "rows": 120,
        "seeds": list(SEEDS),
        "status": "pass",
        "input_provenance": input_provenance,
        "runtime_asset_provenance": runtime_provenance,
        "family_provenance": family_provenance,
        "aggregate_reference": aggregate_provenance,
        "retrospective_audit_manifest": retrospective_manifest_provenance,
        "exhaustive_production_manifests": exhaustive_provenance,
        "outcome_access": {
            "retention_values_loaded": False,
            "binder_values_loaded": False,
            "aggregate_model_score_columns_loaded": True,
            "biological_metrics_calculated": False,
        },
        "checks": {
            "exact_120_pair_identity_and_sequence_mapping": True,
            "all_five_seed_scores_within_tolerance": True,
            "all_seed_within_peptide_rankings_identical": True,
            "sigmoid_mean_seed_logit_recomputed": True,
            "saved_aggregate_score_within_tolerance": True,
            "all_aggregate_within_peptide_rankings_identical": True,
        },
        "outputs": {
            name: {
                "path": path.name,
                "rows": len(output_frames[name]),
                "sha256": sha256_file(path),
            }
            for name, path in output_paths.items()
        },
    }
    manifest["outputs"]["report"] = {
        "path": report_path.name,
        "bytes": report_path.stat().st_size,
        "sha256": sha256_file(report_path),
    }
    manifest_path = staging / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    _require(not output.exists(), "output appeared during staged parity audit")
    os.replace(staging, output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
