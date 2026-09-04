#!/usr/bin/env python3
"""Build an input-driven LibA prospective candidate handoff and public report.

Candidate selection uses only immutable weak-label deployment locks and model
score files. The already-measured retention comparison is loaded afterward and
is used only for clearly labelled retrospective tables in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBA_CRYSTAL_POSITIONS = (15, 19, 29, 33)
LIBA_DISPLAYED_POSITIONS = (13, 17, 27, 31)
LIBA_MAPPING_TEXT = (
    "code_char_1=crystal_15/displayed_13;"
    "code_char_2=crystal_19/displayed_17;"
    "code_char_3=crystal_29/displayed_27;"
    "code_char_4=crystal_33/displayed_31"
)
FORBIDDEN_SCORE_COLUMN_FRAGMENTS = (
    "retention",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
)
FORBIDDEN_SCORE_COLUMNS_EXACT = {
    "binder", "label", "target", "outcome", "measured_value",
}
LIBA_DESIGN_ALPHABET = frozenset("ADEFHIKLNPQSTVY")
FIRST_BATCH_TOTAL_PER_PEPTIDE = 10
FIRST_BATCH_CONTROL_TARGET = 3
FIRST_BATCH_PREFERRED_CODE_HAMMING = 2
SCORE_REQUIRED_COLUMNS = {
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
}
COMPARISON_REQUIRED_COLUMNS = {
    "model_id",
    "display_name",
    "evaluation_pairs",
    "within_peptide_auroc",
    "evaluable_peptides_for_within_peptide_auroc",
    "within_peptide_average_precision",
    "evaluable_peptides_for_within_peptide_ap",
    "within_peptide_spearman",
    "evaluable_peptides_for_within_peptide_spearman",
    "weak_oof_within_peptide_average_precision",
    "weak_oof_evaluable_peptides_for_within_peptide_ap",
    "retention_optimized_threshold",
    "pairs_above_retention_optimized_threshold",
    "binders_above_retention_optimized_threshold",
    "retrospective_precision",
    "retrospective_recall",
    "retrospective_f1",
}
COMPARISON_OPTIONAL_COLUMNS = {"global_auroc", "global_average_precision"}
EVIDENCE_REQUIRED_COLUMNS = {
    "pair_uid", "peptide_design_code", "peptide_9mer_sequence",
    "affibody_design_code", "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence", "model_input_smart_hla_linker_peptide_sequence",
    "r009_count", "r010_count", "pooled_r009_r010_count",
    "directly_measured", "pooled_r009_r010_top2pct",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden_score_columns(columns) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if str(column).strip().lower() in FORBIDDEN_SCORE_COLUMNS_EXACT
        or any(fragment in str(column).strip().lower()
               for fragment in FORBIDDEN_SCORE_COLUMN_FRAGMENTS)
    )


def parse_model_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        _require("=" in value, f"model score must be MODEL_ID=PATH: {value}")
        model_id, raw_path = value.split("=", 1)
        model_id = model_id.strip()
        path = Path(raw_path).expanduser().resolve()
        _require(model_id and model_id not in paths, f"duplicate/empty model ID: {model_id}")
        _require(path.is_file(), f"candidate score file does not exist: {path}")
        paths[model_id] = path
    return paths


def load_json(path: Path) -> dict:
    _require(path.is_file(), f"input JSON does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_lock_manifest(path: Path) -> dict:
    manifest = load_json(path)
    _require(manifest.get("schema_version") == "liba-weak-oof-deployment-locks-v1",
             "unexpected LibA weak-OOF lock schema")
    _require(manifest.get("library") == "LibA", "weak-OOF lock is not LibA")
    _require(manifest.get("retention_labels_read") is False,
             "weak-OOF lock does not explicitly forbid retention use")
    models = manifest.get("models")
    _require(isinstance(models, dict) and models, "weak-OOF lock contains no models")
    eligible = []
    for model_id, lock in models.items():
        _require(isinstance(lock, dict), f"lock is missing for {model_id}")
        if lock.get("scientifically_eligible") is not True:
            continue
        eligible.append(model_id)
        _require(lock.get("threshold_uses_retention") is False,
                 f"{model_id} deployment threshold is not weak-label-only")
        _require(isinstance(lock.get("threshold_provenance"), str)
                 and "weak" in lock["threshold_provenance"].lower(),
                 f"{model_id} weak-label threshold provenance is missing")
        _require(isinstance(lock.get("score_threshold"), (int, float))
                 and math.isfinite(float(lock["score_threshold"])),
                 f"{model_id} score threshold is invalid")
        _require(lock.get("higher_is_better") is True,
                 f"{model_id} scores must be explicitly higher-is-better")
        _require(0 < int(lock.get("max_candidates_per_peptide", 0)) <= 10,
                 f"{model_id} candidate cap must be 1--10")
        _require(0 <= int(lock.get("minimum_code_hamming_distance", -1)) <= 4,
                 f"{model_id} Hamming-distance rule is invalid")
        _require(lock.get("pad_below_threshold") is False,
                 f"{model_id} must not pad below its weak-label threshold")
        _require(isinstance(lock.get("lock_id"), str) and lock["lock_id"],
                 f"{model_id} lock_id is missing")
        _require(isinstance(lock.get("display_name"), str) and lock["display_name"],
                 f"{model_id} display name is missing")
        model_text = f"{model_id} {lock['display_name']}".lower()
        if "mint" in model_text:
            _require(re.search(r"(?:layer[ _-]*(?:0?9|33)|l(?:0?9|33))\b", model_text)
                     is not None,
                     "scientifically eligible LibA MINT input must identify layer 9 or 33")
        for hash_name in ("weak_oof_sha256", "head_sha256", "config_sha256"):
            _require(isinstance(lock.get(hash_name), str) and len(lock[hash_name]) == 64,
                     f"{model_id} {hash_name} is missing")
    _require(eligible, "no scientifically eligible LibA model is locked")
    primary = manifest.get("primary_model_id")
    _require(primary in eligible, "primary LibA model is not scientifically eligible")
    if "mint" in str(primary).lower():
        primary_text = f"{primary} {models[primary].get('display_name', '')}".lower()
        _require(re.search(r"(?:layer[ _-]*0?9|l0?9)\b", primary_text) is not None,
                 "a MINT primary model must be layer 9; layer 33 is a control")
    return manifest


def load_project_spec(path: Path, eligible: list[str]) -> dict:
    manifest = load_json(path)
    _require(manifest.get("schema_version") == "liba-wetlab-project-spec-v1",
             "unexpected LibA project-spec schema")
    _require(manifest.get("library") == "LibA", "project spec is not LibA")
    targets = manifest.get("target_sequences")
    _require(isinstance(targets, dict) and len(targets) == 9,
             "LibA handoff requires exactly nine target peptide sequences")
    _require(all(re.fullmatch(r"[A-Z]{2,20}", str(code)) for code in targets),
             "target codes must be stable uppercase identifiers")
    _require(all(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{9}", str(seq))
                 for seq in targets.values()),
             "each target must map to one valid 9-aa peptide sequence")
    training = manifest.get("training_data")
    _require(isinstance(training, dict), "LibA training-data description is missing")
    for key in ("description", "weak_label_definition", "split_rule"):
        _require(isinstance(training.get(key), str) and training[key].strip(),
                 f"training_data.{key} is missing")
    training_rows = training.get("training_rows_by_model")
    _require(isinstance(training_rows, dict) and set(training_rows) == set(eligible),
             "training row counts must match the eligible model set")
    _require(all(isinstance(value, int) and value > 0 for value in training_rows.values()),
             "training row counts must be positive integers")
    _require(isinstance(manifest.get("expected_selection_missed_candidate_pairs"), int)
             and manifest["expected_selection_missed_candidate_pairs"] > 0,
             "expected selection-missed candidate-universe size is missing")
    for key in ("expected_double_cold_candidate_pairs", "expected_peptide_cold_only_candidate_pairs"):
        _require(isinstance(manifest.get(key), int) and manifest[key] >= 0,
                 f"{key} is missing")
    _require(manifest["expected_double_cold_candidate_pairs"] +
             manifest["expected_peptide_cold_only_candidate_pairs"] ==
             manifest["expected_selection_missed_candidate_pairs"],
             "double-cold and peptide-cold-only counts do not partition the universe")
    evidence = manifest.get("evidence_positive_control")
    _require(isinstance(evidence, dict), "evidence-positive control specification is missing")
    _require(isinstance(evidence.get("expected_pairs"), int) and evidence["expected_pairs"] > 0,
             "evidence-positive expected pair count is missing")
    _require(0 < int(evidence.get("max_controls_per_peptide", 0)) <= 10,
             "evidence-positive per-target review cap must be 1--10")
    unavailable = manifest.get("unavailable_structure_families")
    _require(isinstance(unavailable, list) and unavailable,
             "unavailable structure families must be stated explicitly")
    for record in unavailable:
        _require(isinstance(record, dict) and record.get("family") and record.get("reason"),
                 "each unavailable structure family needs a family and reason")
    return manifest


def load_evidence_retrospective_audit(path: Path) -> dict:
    if path.is_dir():
        manifest_path = path / "manifest.json"
        metrics_path = path / "measured_membership_metrics.csv"
        _require(manifest_path.exists() and metrics_path.exists(),
                 "evidence benchmark directory is incomplete")
        manifest = load_json(manifest_path)
        _require(manifest.get("schema_version") ==
                 "liba-provider-pooled-membership-retrospective-benchmark-v1",
                 "unexpected provider evidence benchmark schema")
        output = manifest.get("outputs", {}).get(metrics_path.name, {})
        _require(output.get("sha256") == sha256_file(metrics_path),
                 "evidence benchmark metrics hash does not match manifest")
        frame = read_table(metrics_path)
        _require(len(frame) == 1 and int(output.get("rows", -1)) == 1,
                 "evidence benchmark must contain exactly one metrics row")
        row = frame.iloc[0]
        return {
            "schema_version": manifest["schema_version"],
            "retention_panel_read": True,
            "retrospective_metrics": {
                "evaluation_pairs": int(row["measured_panel_rows"]),
                "pooled_top2pct_count_cutoff": int(row["pooled_positive_membership_cutoff"]),
                "pooled_top2pct_true_positive_count": int(row["true_positives"]),
                "pooled_top2pct_predicted_positive_count": int(row["measured_pooled_positive_members"]),
                "pooled_top2pct_precision": float(row["precision"]),
                "pooled_top2pct_recall": float(row["recall"]),
                "extrapolative_expected_binders": float(row["extrapolated_expected_binders"]),
                "extrapolative_menu_rows": int(row["fixed_unmeasured_menu_rows"]),
            },
            "source_manifest": manifest,
        }
    audit = load_json(path)
    _require(audit.get("schema_version")
             == "liba-evidence-control-retrospective-audit-v1",
             "unexpected evidence-control retrospective-audit schema")
    _require(audit.get("retention_panel_read") is True,
             "evidence-control audit must identify retrospective retention access")
    metrics = audit.get("retrospective_metrics")
    required_metrics = {
        "evaluation_pairs",
        "pooled_top2pct_count_cutoff", "pooled_top2pct_true_positive_count",
        "pooled_top2pct_predicted_positive_count", "pooled_top2pct_precision",
        "pooled_top2pct_recall", "extrapolative_expected_binders",
        "extrapolative_menu_rows",
    }
    _require(isinstance(metrics, dict) and required_metrics.issubset(metrics),
             "evidence-positive retrospective metrics are incomplete")
    _require(all(isinstance(metrics[key], (int, float))
                 and math.isfinite(float(metrics[key])) for key in required_metrics),
             "evidence-positive retrospective metrics must be finite")
    return audit


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    _require(path.suffix.lower() in {".csv", ".tsv"},
             f"unsupported table extension: {path}")
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def validate_candidate_score_manifest(
    model_id: str, score_path: Path, expected_rows: int, lock: dict
) -> dict:
    manifest_path = score_path.parent / "manifest.json"
    _require(manifest_path.is_file(), f"{model_id} candidate scores have no sibling manifest.json")
    manifest = load_json(manifest_path)
    _require(str(manifest.get("model_id", "")) == model_id,
             f"{model_id} score manifest identifies the wrong model")
    entry = manifest.get("outputs", {}).get("candidate_scores")
    _require(isinstance(entry, dict),
             f"{model_id} score manifest lacks outputs.candidate_scores")
    _require(entry.get("sha256") == sha256_file(score_path),
             f"{model_id} candidate-score hash differs from its manifest")
    _require(int(entry.get("rows", -1)) == expected_rows,
             f"{model_id} candidate-score manifest has the wrong universe row count")
    named_path = Path(str(entry.get("path", "")))
    _require(named_path.name == score_path.name,
             f"{model_id} candidate-score manifest names a different file")
    _require(int(manifest.get("rows", -1)) == expected_rows,
             f"{model_id} manifest does not attest the exact candidate-universe row count")
    _require(re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("pair_uid_membership_sha256", "")))
             is not None, f"{model_id} candidate membership hash is missing")
    binding = manifest.get("deployment_binding")
    _require(isinstance(binding, dict), f"{model_id} deployment binding is missing")
    expected_bindings = {
        "head_sha256": lock["head_sha256"],
        "config_sha256": lock["config_sha256"],
        "weak_oof_sha256": lock["weak_oof_sha256"],
        "generic_lock_sha256": lock["weak_metric_provenance"]["generic_lock_sha256"],
    }
    for key, expected in expected_bindings.items():
        _require(binding.get(key) == expected,
                 f"{model_id} score manifest {key} differs from deployment lock")
    return manifest


def validate_code_sequence(code: str, sequence: str, label: str) -> None:
    _require(len(code) == 4, f"{label} LibA Affibody code must have four characters")
    _require(set(code).issubset(LIBA_DESIGN_ALPHABET),
             f"{label} Affibody code contains an amino acid outside the LibA design alphabet")
    _require(len(sequence) == 58, f"{label} displayed Affibody sequence must have 58 aa")
    reconstructed = "".join(sequence[position - 1] for position in LIBA_DISPLAYED_POSITIONS)
    _require(reconstructed == code,
             f"{label} Affibody code does not match displayed positions 13/17/27/31")


def affibody_scaffold_signature(sequence: str) -> str:
    values = list(str(sequence))
    _require(len(values) == 58, "LibA Affibody scaffold sequence must have 58 aa")
    for position in LIBA_DISPLAYED_POSITIONS:
        values[position - 1] = "X"
    return "".join(values)


def validate_sequence_identity_maps(frame: pd.DataFrame, label: str) -> None:
    affibody_map = frame.groupby("affibody_design_code")[
        "provider_displayed_58aa_affibody_sequence"
    ].nunique()
    _require(affibody_map.le(1).all(),
             f"{label} maps one Affibody code to multiple full sequences")
    assay_map = frame.groupby("peptide_design_code")[
        "model_input_smart_hla_linker_peptide_sequence"
    ].nunique()
    _require(assay_map.le(1).all(),
             f"{label} maps one peptide target to multiple assay-side sequences")
    scaffolds = frame["provider_displayed_58aa_affibody_sequence"].astype(str).map(
        affibody_scaffold_signature
    )
    _require(scaffolds.nunique() == 1,
             f"{label} contains more than one fixed LibA Affibody scaffold")


def load_candidate_scores(
    model_id: str, path: Path, target_sequences: dict[str, str]
) -> pd.DataFrame:
    frame = read_table(path)
    leaked = forbidden_score_columns(frame.columns)
    _require(not leaked, f"{model_id} candidate scores leak outcomes: {leaked}")
    _require(SCORE_REQUIRED_COLUMNS.issubset(frame.columns),
             f"{model_id} candidate-score schema is missing required columns")
    _require(set(frame["model_id"].astype(str)) == {model_id},
             f"candidate-score file identifies the wrong model for {model_id}")
    _require(frame["pair_uid"].notna().all()
             and not frame["pair_uid"].astype(str).duplicated().any(),
             f"{model_id} candidate pair IDs are missing or duplicated")
    _require(set(frame["peptide_design_code"].astype(str)) == set(target_sequences),
             f"{model_id} candidate scores are not the exact nine-target set")
    for target, block in frame.groupby("peptide_design_code", sort=True):
        _require(set(block["peptide_9mer_sequence"].astype(str))
                 == {target_sequences[str(target)]},
                 f"{model_id} peptide sequence mismatch for {target}")
    score = pd.to_numeric(frame["model_score"], errors="raise").astype(float)
    _require(np.isfinite(score).all(), f"{model_id} contains non-finite model scores")
    frame["model_score"] = score
    for row in frame[["affibody_design_code", "provider_displayed_58aa_affibody_sequence",
                      "model_input_affibody_sequence"]].itertuples(index=False):
        validate_code_sequence(str(row[0]), str(row[1]), model_id)
        _require(str(row[1]) == str(row[2]),
                 f"{model_id} displayed and model-input Affibody sequences differ unexpectedly")
    _require(all(
        str(assay).endswith(str(peptide))
        for assay, peptide in zip(
            frame["model_input_smart_hla_linker_peptide_sequence"],
            frame["peptide_9mer_sequence"],
        )
    ), f"{model_id} assay-side input does not end in its peptide 9-mer")
    validate_sequence_identity_maps(frame, f"{model_id} candidate scores")
    return frame


def load_evidence_positive_controls(
    path: Path, target_sequences: dict[str, str], expected_rows: int
) -> pd.DataFrame:
    frame = read_table(path)
    # Accept the provider-audited native artifact without manual renaming.
    native = {
        "peptide_full_sequence", "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence", "candidate_tier", "selected_top10",
    }
    if native.issubset(frame.columns):
        manifest_path = path.parent / "manifest.json"
        _require(manifest_path.exists(), "native evidence artifact has no sibling manifest.json")
        manifest = load_json(manifest_path)
        output = manifest.get("outputs", {}).get(path.name, {})
        _require(output.get("sha256") == sha256_file(path),
                 "native evidence artifact hash does not match its manifest")
        _require(int(output.get("rows", -1)) == len(frame),
                 "native evidence artifact row count does not match its manifest")
        frame = frame.rename(columns={
            "peptide_full_sequence": "peptide_9mer_sequence",
            "chain1_smart_hla_linker_peptide_sequence":
                "model_input_smart_hla_linker_peptide_sequence",
            "chain2_affibody_sequence": "provider_displayed_58aa_affibody_sequence",
        })
        frame["model_input_affibody_sequence"] = frame[
            "provider_displayed_58aa_affibody_sequence"
        ]
        frame["directly_measured"] = False
        frame["pooled_r009_r010_top2pct"] = True
    leaked = forbidden_score_columns(frame.columns)
    _require(not leaked, f"evidence-positive controls leak outcomes: {leaked}")
    _require(EVIDENCE_REQUIRED_COLUMNS.issubset(frame.columns),
             "evidence-positive control schema is incomplete")
    _require(len(frame) == expected_rows,
             "evidence-positive control count differs from its lock")
    _require(frame["pair_uid"].notna().all()
             and not frame["pair_uid"].astype(str).duplicated().any(),
             "evidence-positive controls have missing or duplicate pair IDs")
    for flag in ("directly_measured", "pooled_r009_r010_top2pct"):
        _require(pd.api.types.is_bool_dtype(frame[flag]) and frame[flag].notna().all(),
                 f"evidence-positive {flag} must be nonmissing Boolean values")
    _require((~frame["directly_measured"]).all(),
             "evidence-positive tier contains an already measured pair")
    _require(frame["pooled_r009_r010_top2pct"].all(),
             "evidence-positive tier contains a pair outside pooled R009+R010 top 2%")
    observed_targets = set(frame["peptide_design_code"].astype(str))
    _require(observed_targets.issubset(target_sequences),
             "evidence-positive controls contain a non-project target")
    _require(observed_targets,
             "evidence-positive controls contain no project targets")
    for target, block in frame.groupby("peptide_design_code", sort=True):
        _require(set(block["peptide_9mer_sequence"].astype(str))
                 == {target_sequences[str(target)]},
                 f"evidence-positive peptide sequence mismatch for {target}")
    for row in frame[["affibody_design_code", "provider_displayed_58aa_affibody_sequence",
                      "model_input_affibody_sequence"]].itertuples(index=False):
        validate_code_sequence(str(row[0]), str(row[1]), "evidence-positive controls")
        _require(str(row[1]) == str(row[2]),
                 "evidence-positive displayed and model-input Affibody sequences differ")
    for column in ("r009_count", "r010_count", "pooled_r009_r010_count"):
        values = pd.to_numeric(frame[column], errors="raise")
        _require(values.notna().all() and values.ge(0).all()
                 and np.equal(values, np.floor(values)).all(),
                 f"evidence-positive {column} must be nonnegative integer counts")
        frame[column] = values.astype(np.uint64)
    _require(frame["pooled_r009_r010_count"].eq(
        frame["r009_count"] + frame["r010_count"]
    ).all(), "evidence-positive pooled count is not R009 + R010")
    _require(all(
        str(assay).endswith(str(peptide))
        for assay, peptide in zip(
            frame["model_input_smart_hla_linker_peptide_sequence"],
            frame["peptide_9mer_sequence"],
        )
    ), "evidence-positive assay-side input does not end in its peptide")
    validate_sequence_identity_maps(frame, "evidence-positive controls")
    return add_liba_mapping(frame)


def evidence_review_menu(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    selected = (
        frame.sort_values(
            ["peptide_design_code", "pooled_r009_r010_count",
             "affibody_design_code", "pair_uid"],
            ascending=[True, False, True, True], kind="mergesort",
        )
        .groupby("peptide_design_code", sort=True, as_index=False)
        .head(maximum)
        .copy()
    )
    selected["evidence_rank"] = selected.groupby(
        "peptide_design_code", sort=True
    ).cumcount() + 1
    selected["tier"] = "separate_evidence_positive_control_not_model_candidate"
    return selected.reset_index(drop=True)


def membership_sha256(values: pd.Series) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values.astype(str))).encode("utf-8")
    ).hexdigest()


def validate_evidence_menu_against_benchmark(
    menu: pd.DataFrame, audit: dict
) -> None:
    """Bind retrospective extrapolation to the exact menu it evaluated."""
    source = audit.get("source_manifest")
    if source is None:
        return
    expected_rows = int(audit["retrospective_metrics"]["extrapolative_menu_rows"])
    expected_hash = source.get("memberships", {}).get(
        "fixed_unmeasured_menu_sha256"
    )
    _require(isinstance(expected_hash, str) and len(expected_hash) == 64,
             "evidence benchmark lacks fixed-menu membership provenance")
    _require(len(menu) == expected_rows,
             "generated evidence menu row count differs from retrospective benchmark")
    _require(membership_sha256(menu["pair_uid"]) == expected_hash,
             "generated evidence menu membership differs from retrospective benchmark")


def compact_evidence_codes_by_target(
    menu: pd.DataFrame, maximum: int, target_sequences: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for target, peptide in target_sequences.items():
        block = menu.loc[menu["peptide_design_code"].astype(str).eq(str(target))]
        record = {
            "peptide_design_code": target,
            "peptide_9mer_sequence": peptide,
            "selected_evidence_positive_controls": int(len(block)),
        }
        for rank in range(1, maximum + 1):
            hit = block.loc[block["evidence_rank"].eq(rank)]
            record[f"rank_{rank:02d}_affibody_code"] = (
                str(hit["affibody_design_code"].iloc[0]) if len(hit) else ""
            )
            record[f"rank_{rank:02d}_pooled_count"] = (
                int(hit["pooled_r009_r010_count"].iloc[0]) if len(hit) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def hamming(left: str, right: str) -> int:
    _require(len(left) == len(right) == 4, "LibA code length changed")
    return sum(a != b for a, b in zip(left, right))


def _diverse_pick(
    ranked: pd.DataFrame,
    maximum: int,
    existing_codes: list[str] | None = None,
) -> pd.DataFrame:
    """Prefer Hamming >=2, then deterministically relax only to fill capacity."""
    if maximum <= 0:
        output = ranked.iloc[0:0].copy()
        output["diversity_selection_phase"] = pd.Series(dtype=str)
        return output
    existing = list(existing_codes or [])
    chosen_indices: list[int] = []
    phases: dict[int, str] = {}
    for minimum, phase in (
        (FIRST_BATCH_PREFERRED_CODE_HAMMING, "preferred_hamming_at_least_2"),
        (1, "relaxed_to_unique_code_to_fill_available_slot"),
    ):
        for index, row in ranked.iterrows():
            if index in chosen_indices:
                continue
            code = str(row["affibody_design_code"])
            if all(hamming(code, prior) >= minimum for prior in existing):
                chosen_indices.append(index)
                existing.append(code)
                phases[index] = phase
                if len(chosen_indices) == maximum:
                    break
        if len(chosen_indices) == maximum:
            break
    selected = ranked.loc[chosen_indices].copy()
    selected["diversity_selection_phase"] = [phases[index] for index in chosen_indices]
    return selected


def build_tiered_first_batch(
    primary_scores: pd.DataFrame,
    primary_lock: dict,
    evidence_full: pd.DataFrame,
    target_sequences: dict[str, str],
    primary_model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a retention-blind 3-control + 7-discovery first assay batch."""
    threshold = float(primary_lock["score_threshold"])
    discovery_pool = primary_scores.loc[
        (~primary_scores["affibody_identity_seen_in_strict_training"].astype(bool))
        & (~primary_scores["high_confidence_weak_negative"].astype(bool))
        & primary_scores["model_score"].ge(threshold)
    ].copy()
    discovery_pool["candidate_tier"] = "strict_double_cold_model_discovery"
    discovery_pool["selection_basis"] = (
        f"{primary_model_id}_score_at_or_above_weak_oof_threshold"
    )

    blocks = []
    summaries = []
    for target_order, (target, peptide) in enumerate(target_sequences.items(), start=1):
        controls_ranked = evidence_full.loc[
            evidence_full["peptide_design_code"].astype(str).eq(target)
        ].sort_values(
            ["pooled_r009_r010_count", "affibody_design_code", "pair_uid"],
            ascending=[False, True, True], kind="mergesort",
        )
        controls = _diverse_pick(
            controls_ranked, min(FIRST_BATCH_CONTROL_TARGET, len(controls_ranked))
        )
        controls["candidate_tier"] = "pooled_R009_R010_top2pct_evidence_control"
        controls["selection_basis"] = "pooled_R009_plus_R010_count"
        controls["within_tier_rank"] = np.arange(1, len(controls) + 1, dtype=np.int64)

        discovery_slots = FIRST_BATCH_TOTAL_PER_PEPTIDE - len(controls)
        discoveries_ranked = discovery_pool.loc[
            discovery_pool["peptide_design_code"].astype(str).eq(target)
        ].sort_values(["model_score", "pair_uid"], ascending=[False, True], kind="mergesort")
        discoveries = _diverse_pick(
            discoveries_ranked,
            min(discovery_slots, len(discoveries_ranked)),
            controls["affibody_design_code"].astype(str).tolist(),
        )
        discoveries["within_tier_rank"] = np.arange(
            1, len(discoveries) + 1, dtype=np.int64
        )
        target_rows = []
        for source, selected in (("control", controls), ("discovery", discoveries)):
            for row in selected.to_dict("records"):
                record = dict(row)
                record["target_order"] = target_order
                record["tier_role"] = source
                record["primary_model_id"] = (
                    "not_applicable" if source == "control" else primary_model_id
                )
                record["weak_label_deployment_threshold"] = (
                    np.nan if source == "control" else threshold
                )
                target_rows.append(record)
        if not target_rows:
            summaries.append({
                "peptide_design_code": target,
                "peptide_9mer_sequence": peptide,
                "batch_status": "NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD",
                "evidence_controls_available": 0,
                "evidence_controls_selected": 0,
                "strict_double_cold_discoveries_available_above_threshold_after_qc": 0,
                "strict_double_cold_discoveries_selected": 0,
                "total_pair_assays": 0,
                "control_shortfall_filled_by_discoveries": 0,
                "maximum_primary_model_score": float(primary_scores.loc[
                    primary_scores["peptide_design_code"].astype(str).eq(target),
                    "model_score",
                ].max()),
                "locked_primary_model_threshold": threshold,
                "reason": (
                    "No evidence control exists and no strict-double-cold candidate passes "
                    "the locked weak-label threshold; threshold was not lowered and no row "
                    "was padded."
                ),
            })
            continue
        target_frame = pd.DataFrame(target_rows)
        target_frame["assay_slot"] = np.arange(1, len(target_frame) + 1, dtype=np.int64)
        codes = target_frame["affibody_design_code"].astype(str).tolist()
        target_frame["minimum_hamming_to_another_batch_code"] = [
            min((hamming(code, other) for other in codes if other != code), default=4)
            for code in codes
        ]
        _require(len(target_frame) <= FIRST_BATCH_TOTAL_PER_PEPTIDE,
                 f"first batch exceeds ten Affibodies for {target}")
        _require(target_frame["pair_uid"].astype(str).is_unique,
                 f"duplicate pair IDs in first batch for {target}")
        _require(not target_frame.loc[
            target_frame["tier_role"].eq("discovery"),
            "high_confidence_weak_negative",
        ].astype(bool).any(), f"high-confidence weak negative entered discovery tier for {target}")
        blocks.append(target_frame)
        summaries.append({
            "peptide_design_code": target,
            "peptide_9mer_sequence": peptide,
            "batch_status": "ORDERABLE_CANDIDATES_AVAILABLE",
            "evidence_controls_available": int(len(controls_ranked)),
            "evidence_controls_selected": int(len(controls)),
            "strict_double_cold_discoveries_available_above_threshold_after_qc": int(
                len(discoveries_ranked)
            ),
            "strict_double_cold_discoveries_selected": int(len(discoveries)),
            "total_pair_assays": int(len(target_frame)),
            "control_shortfall_filled_by_discoveries": int(
                FIRST_BATCH_CONTROL_TARGET - len(controls)
            ),
            "maximum_primary_model_score": float(primary_scores.loc[
                primary_scores["peptide_design_code"].astype(str).eq(target),
                "model_score",
            ].max()),
            "locked_primary_model_threshold": threshold,
            "reason": "",
        })
    batch = pd.concat(blocks, ignore_index=True).sort_values(
        ["target_order", "assay_slot"], kind="mergesort"
    ).reset_index(drop=True)
    _require(batch["pair_uid"].astype(str).is_unique,
             "first batch contains duplicate pair IDs across targets")
    batch["sample_id"] = [
        f"LIBA-{target}-{int(slot):02d}"
        for target, slot in zip(batch["peptide_design_code"], batch["assay_slot"])
    ]
    batch = add_liba_mapping(batch)
    return batch, pd.DataFrame(summaries)


def select_candidates(frame: pd.DataFrame, lock: dict) -> pd.DataFrame:
    threshold = float(lock["score_threshold"])
    maximum = int(lock["max_candidates_per_peptide"])
    minimum_hamming = int(lock["minimum_code_hamming_distance"])
    blocks = []
    for target, block in frame.groupby("peptide_design_code", sort=True):
        ranked = block.sort_values(
            ["model_score", "pair_uid"], ascending=[False, True], kind="mergesort"
        )
        eligible = ranked.loc[ranked["model_score"].ge(threshold)]
        chosen_indices = []
        chosen_codes: list[str] = []
        for index, row in eligible.iterrows():
            code = str(row["affibody_design_code"])
            if all(hamming(code, prior) >= minimum_hamming for prior in chosen_codes):
                chosen_indices.append(index)
                chosen_codes.append(code)
                if len(chosen_indices) == maximum:
                    break
        selected = eligible.loc[chosen_indices].copy()
        if len(selected) == 0:
            continue
        selected["model_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
        selected["assay_slot"] = selected["model_rank"]
        selected["weak_label_deployment_threshold"] = threshold
        selected["threshold_provenance"] = lock["threshold_provenance"]
        blocks.append(selected)
    _require(blocks, "weak-label threshold selects no candidates for any target")
    return pd.concat(blocks, ignore_index=True).sort_values(
        ["peptide_design_code", "model_rank"], kind="mergesort"
    ).reset_index(drop=True)


def no_candidate_above_threshold_table(
    score_frames: dict[str, pd.DataFrame],
    menus: dict[str, pd.DataFrame],
    lock_manifest: dict,
    target_sequences: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for model_id, scores in score_frames.items():
        selected_targets = set(menus[model_id]["peptide_design_code"].astype(str))
        threshold = float(lock_manifest["models"][model_id]["score_threshold"])
        for target, peptide in target_sequences.items():
            if target in selected_targets:
                continue
            block = scores.loc[scores["peptide_design_code"].astype(str).eq(target)]
            _require(len(block) > 0, f"candidate universe has no rows for {target}")
            rows.append({
                "model_id": model_id,
                "display_name": lock_manifest["models"][model_id]["display_name"],
                "peptide_design_code": target,
                "peptide_9mer_sequence": peptide,
                "status": "NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD",
                "maximum_model_score": float(block["model_score"].max()),
                "locked_weak_label_threshold": threshold,
                "score_gap_below_threshold": threshold - float(block["model_score"].max()),
                "orderable": False,
                "reason": (
                    "No candidate passes the locked weak-label threshold. The threshold was "
                    "not lowered and no candidate was padded into the wet-lab order."
                ),
            })
    columns = [
        "model_id", "display_name", "peptide_design_code", "peptide_9mer_sequence",
        "status", "maximum_model_score", "locked_weak_label_threshold",
        "score_gap_below_threshold", "orderable", "reason",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["peptide_design_code", "model_id"], kind="mergesort"
    ).reset_index(drop=True)


def rank_only_below_threshold_diagnostic(
    score_frames: dict[str, pd.DataFrame],
    no_candidate_targets: pd.DataFrame,
    lock_manifest: dict,
    maximum: int = 10,
) -> pd.DataFrame:
    """Show, but never order, leading scores for target/model failures."""
    blocks = []
    for row in no_candidate_targets.itertuples(index=False):
        threshold = float(lock_manifest["models"][row.model_id]["score_threshold"])
        ranked = score_frames[row.model_id].loc[
            score_frames[row.model_id]["peptide_design_code"].astype(str).eq(
                str(row.peptide_design_code)
            )
        ].sort_values(
            ["model_score", "affibody_design_code", "pair_uid"],
            ascending=[False, True, True], kind="mergesort",
        ).head(maximum).copy()
        ranked.insert(0, "diagnostic_model_id", row.model_id)
        ranked.insert(1, "diagnostic_rank", np.arange(1, len(ranked) + 1))
        ranked["locked_weak_label_threshold"] = threshold
        ranked["score_gap_below_threshold"] = threshold - ranked["model_score"]
        ranked["status"] = "BELOW_LOCKED_THRESHOLD_RANK_ONLY"
        ranked["orderable"] = False
        ranked["recommended_for_wetlab"] = False
        ranked["warning"] = (
            "Diagnostic only: no candidate passed the locked threshold; do not copy this row "
            "into ORDER_THIS_BATCH."
        )
        blocks.append(ranked)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True)


def add_liba_mapping(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["affibody_code_position_mapping"] = LIBA_MAPPING_TEXT
    output["affibody_code_character_order_crystal_aligned_positions"] = "15,19,29,33"
    output["affibody_code_character_order_displayed_58aa_positions"] = "13,17,27,31"
    for index, position in enumerate(LIBA_CRYSTAL_POSITIONS):
        output[f"affibody_crystal_position_{position}_amino_acid"] = output[
            "affibody_design_code"
        ].astype(str).str[index]
    return output


def compact_codes_by_target(menu: pd.DataFrame, maximum: int) -> pd.DataFrame:
    rows = []
    for target, block in menu.groupby("peptide_design_code", sort=True):
        record = {
            "peptide_design_code": target,
            "peptide_9mer_sequence": block["peptide_9mer_sequence"].iloc[0],
        }
        for rank in range(1, maximum + 1):
            hit = block.loc[block["model_rank"].eq(rank)]
            record[f"rank_{rank:02d}_affibody_code"] = (
                str(hit["affibody_design_code"].iloc[0]) if len(hit) else ""
            )
            record[f"rank_{rank:02d}_score"] = (
                float(hit["model_score"].iloc[0]) if len(hit) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def unique_sequence_roster(menu: pd.DataFrame) -> pd.DataFrame:
    return (
        menu.groupby(
            ["affibody_design_code", "provider_displayed_58aa_affibody_sequence",
             "model_input_affibody_sequence"], as_index=False
        )
        .agg(
            selected_for_n_targets=("peptide_design_code", "nunique"),
            selected_for_targets=(
                "peptide_design_code", lambda values: ";".join(sorted(set(map(str, values))))
            ),
            best_model_rank=("model_rank", "min"),
        )
        .sort_values(["selected_for_n_targets", "affibody_design_code"],
                     ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def candidate_qc_summary(menu: pd.DataFrame) -> pd.DataFrame:
    return (
        menu.groupby("peptide_design_code", as_index=False)
        .agg(
            peptide_9mer_sequence=("peptide_9mer_sequence", "first"),
            selected_pair_assays=("pair_uid", "size"),
            unique_affibody_sequences=("affibody_design_code", "nunique"),
            not_observed_in_any_raw_round=(
                "observed_in_any_raw_round", lambda values: int((~values.astype(bool)).sum())
            ),
            observed_in_r009_or_r010=(
                "observed_in_r009_or_r010", lambda values: int(values.astype(bool).sum())
            ),
            affibody_identity_seen_in_strict_training=(
                "affibody_identity_seen_in_strict_training",
                lambda values: int(values.astype(bool).sum()),
            ),
            prior_high_confidence_weak_negative=(
                "high_confidence_weak_negative", lambda values: int(values.astype(bool).sum())
            ),
        )
        .sort_values("peptide_design_code", kind="mergesort")
        .reset_index(drop=True)
    )


def evidence_sequence_roster(menu: pd.DataFrame) -> pd.DataFrame:
    return (
        menu.groupby(
            ["affibody_design_code", "provider_displayed_58aa_affibody_sequence",
             "model_input_affibody_sequence"], as_index=False
        )
        .agg(
            selected_for_n_targets=("peptide_design_code", "nunique"),
            selected_for_targets=(
                "peptide_design_code", lambda values: ";".join(sorted(set(map(str, values))))
            ),
            best_evidence_rank=("evidence_rank", "min"),
        )
        .sort_values(["selected_for_n_targets", "affibody_design_code"],
                     ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def load_retrospective_comparison(path: Path, eligible_ids: list[str]) -> pd.DataFrame:
    frame = read_table(path)
    ambiguous = {"auroc", "average_precision", "average_within_peptide_spearman"} & set(frame.columns)
    _require(not ambiguous,
             f"ambiguous retrospective metric names are forbidden: {sorted(ambiguous)}")
    _require(COMPARISON_REQUIRED_COLUMNS.issubset(frame.columns),
             "retrospective model-comparison schema is incomplete")
    _require(not frame["model_id"].astype(str).duplicated().any(),
             "retrospective comparison contains duplicate model IDs")
    _require(set(frame["model_id"].astype(str)) == set(eligible_ids),
             "retrospective comparison does not match the eligible model set")
    numeric = ((COMPARISON_REQUIRED_COLUMNS | COMPARISON_OPTIONAL_COLUMNS)
               & set(frame.columns)) - {"model_id", "display_name"}
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
        _require(np.isfinite(values).all(), f"retrospective {column} is not finite")
        frame[column] = values
    return frame.sort_values("model_id", kind="mergesort").reset_index(drop=True)


def load_retrospective_per_peptide_context(
    path: Path, eligible_ids: list[str], target_sequences: dict[str, str]
) -> pd.DataFrame:
    """Load report-only historical top choices after prospective menus are fixed."""

    frame = read_table(path)
    required = {
        "model", "seed", "peptide_design_code", "n_candidates", "top1_affibody"
    }
    _require(required.issubset(frame.columns),
             "retrospective per-peptide context schema is incomplete")
    frame = frame.loc[frame["model"].astype(str).isin(eligible_ids)].copy()
    _require(
        not frame.duplicated(["model", "peptide_design_code"]).any(),
        "retrospective per-peptide context duplicates a model--peptide row",
    )
    expected = {(model, peptide) for model in eligible_ids for peptide in target_sequences}
    observed = set(zip(
        frame["model"].astype(str), frame["peptide_design_code"].astype(str)
    ))
    _require(observed == expected,
             "retrospective per-peptide context does not cover both models and all targets")
    _require(frame["seed"].astype(str).eq("deployment").all(),
             "retrospective per-peptide context is not the frozen deployment view")
    _require(pd.to_numeric(frame["n_candidates"], errors="raise").eq(12).all(),
             "retrospective LibA panel no longer contains 12 Affibodies per peptide")
    _require(frame["top1_affibody"].astype(str).str.fullmatch(r"[A-Z]{4}").all(),
             "retrospective top Affibody code is invalid")
    return frame.sort_values(["model", "peptide_design_code"], kind="mergesort").reset_index(drop=True)


def load_esmfold_weak_selection(path: Path) -> dict:
    """Load the retention-blind ESMFold2 family gate for report-only exclusion context."""

    payload = load_json(path)
    _require(payload.get("retention_labels_read") is False,
             "ESMFold2 selection artifact does not attest retention blindness")
    _require(payload.get("selection_data") == "selection-derived weak labels only",
             "ESMFold2 selection artifact has an unexpected label source")
    _require(payload.get("selected_feature_family") == "single_inputs_only",
             "unexpected selected ESMFold2 family")
    _require(payload.get("best_structure_derived_family_by_gate_metric") == "full",
             "unexpected best folding-derived ESMFold2 family")
    _require(payload.get("passing_structure_families") == [],
             "ESMFold2 artifact no longer records a failed structure gate")
    models = payload.get("models", {})
    _require({"single_inputs_only", "full"}.issubset(models),
             "ESMFold2 selection artifact lacks required family results")
    pre_fold_ap = float(
        models["single_inputs_only"]["aggregate_mean_logit_metrics"]["within_peptide_ap"]
    )
    full_ap = float(models["full"]["aggregate_mean_logit_metrics"]["within_peptide_ap"])
    mint_ap = float(payload["mint_layer9_reference"]["within_peptide_ap"])
    _require(all(np.isfinite([pre_fold_ap, full_ap, mint_ap])),
             "ESMFold2 gate metrics are not finite")
    _require(full_ap < mint_ap and pre_fold_ap < mint_ap,
             "ESMFold2 gate artifact no longer supports exclusion from candidate scoring")
    return {
        "selected_feature_family": "single_inputs_only",
        "selected_pre_fold_within_peptide_ap": pre_fold_ap,
        "best_folding_derived_family": "full",
        "best_folding_derived_within_peptide_ap": full_ap,
        "mint_layer9_within_peptide_ap": mint_ap,
        "passing_structure_families": [],
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str],
                   decimals: set[str] | None = None,
                   integers: set[str] | None = None) -> str:
    decimals = decimals or set()
    integers = integers or set()
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if column in decimals:
                values.append(str(Decimal(str(value)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )))
            elif column in integers:
                values.append(str(int(value)))
            else:
                values.append(str(int(value)) if isinstance(value, (np.integer,)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    lock_manifest: dict,
    project_spec: dict,
    evidence_audit: dict,
    score_frames: dict[str, pd.DataFrame],
    menus: dict[str, pd.DataFrame],
    order: pd.DataFrame,
    no_candidate_targets: pd.DataFrame,
    comparison: pd.DataFrame,
    retrospective_per_peptide: pd.DataFrame | None,
    esmfold_weak_selection: dict | None,
    recommendations: pd.DataFrame,
    threshold_table: pd.DataFrame,
    first_batch_summary: pd.DataFrame,
) -> str:
    primary = lock_manifest["primary_model_id"]
    primary_name = lock_manifest["models"][primary]["display_name"]
    primary_weak_ap = float(
        lock_manifest["models"][primary]["weak_oof_within_peptide_average_precision"]
    )
    comparison_weak_ap_text = ", ".join(
        f"{lock_manifest['models'][model_id]['display_name']} "
        f"{float(lock_manifest['models'][model_id]['weak_oof_within_peptide_average_precision']):.6f}"
        for model_id in lock_manifest["models"]
        if model_id != primary
        and bool(lock_manifest["models"][model_id].get("scientifically_eligible", False))
    )
    performance = comparison.copy()
    esmfold_exclusion = ""
    if esmfold_weak_selection is not None:
        esmfold_exclusion = (
            "- **ESMFold2: excluded by the locked weak-label gate.** No folding-derived "
            "family passed. The pre-folding sequence control scored within-peptide AP "
            f"{esmfold_weak_selection['selected_pre_fold_within_peptide_ap']:.6f}; the "
            "best actually folding-derived family (`full`) scored "
            f"{esmfold_weak_selection['best_folding_derived_within_peptide_ap']:.6f}, "
            "versus "
            f"{esmfold_weak_selection['mint_layer9_within_peptide_ap']:.6f} for MINT "
            "layer 9. It therefore contributes no candidate scores.\n"
        )
    unavailable_lines = "\n".join(
        f"- **{record['family']}: unavailable.** {record['reason']}"
        for record in project_spec["unavailable_structure_families"]
    )
    performance_table = markdown_table(
        performance,
        ["display_name", "evaluation_pairs", "within_peptide_auroc",
         "evaluable_peptides_for_within_peptide_auroc",
         "within_peptide_average_precision",
         "evaluable_peptides_for_within_peptide_ap", "within_peptide_spearman",
         "evaluable_peptides_for_within_peptide_spearman",
         "weak_oof_within_peptide_average_precision",
         "weak_oof_evaluable_peptides_for_within_peptide_ap"],
        ["Model", "Measured pairs", "Within-peptide AUROC", "AUROC groups",
         "Within-peptide AP", "AP groups", "Within-peptide Spearman",
         "Spearman groups", "Weak-OOF within-peptide AP", "Weak-OOF groups"],
        {"within_peptide_auroc", "within_peptide_average_precision",
         "within_peptide_spearman", "weak_oof_within_peptide_average_precision"},
        {"evaluation_pairs", "evaluable_peptides_for_within_peptide_auroc",
         "evaluable_peptides_for_within_peptide_ap",
         "evaluable_peptides_for_within_peptide_spearman",
         "weak_oof_evaluable_peptides_for_within_peptide_ap"},
    )
    threshold_md = markdown_table(
        threshold_table,
        ["display_name", "weak_label_deployment_threshold",
         "retention_optimized_descriptive_threshold", "threshold_use"],
        ["Model", "Weak-label deployment threshold",
         "Retention-optimized descriptive threshold", "How it may be used"],
        {"weak_label_deployment_threshold", "retention_optimized_descriptive_threshold"},
    )
    retrospective_recommendation_md = markdown_table(
        performance,
        ["display_name", "retention_optimized_threshold",
         "pairs_above_retention_optimized_threshold",
         "binders_above_retention_optimized_threshold", "retrospective_precision",
         "retrospective_recall", "retrospective_f1"],
        ["Model", "Descriptive cutoff", "Pairs above cutoff", "Measured binders above cutoff",
         "Precision", "Recall", "F1"],
        {"retention_optimized_threshold", "retrospective_precision",
         "retrospective_recall", "retrospective_f1"},
        {"pairs_above_retention_optimized_threshold",
         "binders_above_retention_optimized_threshold"},
    )
    evidence_metrics = evidence_audit["retrospective_metrics"]
    evidence_metrics_frame = pd.DataFrame([{
        "evaluation_pairs": evidence_metrics["evaluation_pairs"],
        "count_cutoff": evidence_metrics["pooled_top2pct_count_cutoff"],
        "cutoff_tp": evidence_metrics["pooled_top2pct_true_positive_count"],
        "cutoff_called": evidence_metrics["pooled_top2pct_predicted_positive_count"],
        "cutoff_precision": evidence_metrics["pooled_top2pct_precision"],
        "cutoff_recall": evidence_metrics["pooled_top2pct_recall"],
    }])
    evidence_md = markdown_table(
        evidence_metrics_frame,
        ["evaluation_pairs", "count_cutoff", "cutoff_tp", "cutoff_called",
         "cutoff_precision", "cutoff_recall"],
        ["Measured pairs", "Pooled-count cutoff",
         "Measured binders above cutoff", "Measured pairs above cutoff",
         "Cutoff precision", "Cutoff recall"],
        {"cutoff_precision", "cutoff_recall"},
        {"evaluation_pairs", "count_cutoff", "cutoff_tp",
         "cutoff_called"},
    )
    model_diagnostic_lines = []
    for model_id, menu in menus.items():
        top_codes = sorted(set(
            menu.loc[menu["model_rank"].eq(1), "affibody_design_code"].astype(str)
        ))
        recurrence = menu.groupby("affibody_design_code")["peptide_design_code"].nunique()
        maximum_recurrence = int(recurrence.max())
        recurrent_codes = sorted(recurrence.loc[recurrence.eq(maximum_recurrence)].index.astype(str))
        orderable_targets = int(menu["peptide_design_code"].astype(str).nunique())
        spear = float(performance.loc[
            performance["model_id"].astype(str).eq(model_id),
            "within_peptide_spearman",
        ].iloc[0])
        top_description = (
            f"the same top Affibody `{top_codes[0]}` for all targets"
            if len(top_codes) == 1
            else f"{len(top_codes)} different top Affibody codes across targets"
        )
        model_diagnostic_lines.append(
            f"- **{lock_manifest['models'][model_id]['display_name']}:** {top_description}; "
            f"the most recurrent top-ten code(s) {', '.join(f'`{code}`' for code in recurrent_codes)} "
            f"appear for {maximum_recurrence} of {orderable_targets} orderable targets; "
            f"average within-peptide Spearman {Decimal(str(spear)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}."
        )
    model_ids = list(menus)
    if len(model_ids) == 2:
        left_id, right_id = model_ids
        common_targets = sorted(
            set(menus[left_id]["peptide_design_code"].astype(str))
            & set(menus[right_id]["peptide_design_code"].astype(str))
        )
        overlaps = []
        same_top1 = 0
        identical_ranked = 0
        for target in common_targets:
            left = menus[left_id].loc[
                menus[left_id]["peptide_design_code"].astype(str).eq(target)
            ].sort_values("model_rank", kind="mergesort")["affibody_design_code"].astype(str).tolist()
            right = menus[right_id].loc[
                menus[right_id]["peptide_design_code"].astype(str).eq(target)
            ].sort_values("model_rank", kind="mergesort")["affibody_design_code"].astype(str).tolist()
            overlaps.append(len(set(left) & set(right)))
            same_top1 += int(bool(left and right and left[0] == right[0]))
            identical_ranked += int(bool(left and right and left == right))
        _require(common_targets, "eligible models share no orderable targets")
        mean_overlap = Decimal(str(float(np.mean(overlaps)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        model_diagnostic_lines.append(
            f"- **Agreement between the two prospective menus:** they share an average of "
            f"{mean_overlap} of ten codes across {len(common_targets)} orderable targets, "
            f"choose the same top code for {same_top1}/{len(common_targets)} targets, and "
            f"have the identical ordered top ten for {identical_ranked}/{len(common_targets)} targets."
        )
    model_diagnostics = "\n".join(model_diagnostic_lines)
    rec_md = markdown_table(
        recommendations,
        ["display_name", "prospective_role", "candidate_pairs", "unique_affibody_sequences"],
        ["Model", "Prospective role", "Candidate pair assays", "Unique Affibody sequences"],
    )
    first_batch_md = markdown_table(
        first_batch_summary,
        ["peptide_design_code", "peptide_9mer_sequence", "batch_status",
         "evidence_controls_available", "evidence_controls_selected",
         "strict_double_cold_discoveries_selected", "total_pair_assays"],
        ["Peptide", "9-aa sequence", "Batch status", "Evidence-supported pairs available",
         "Evidence comparators selected", "Model discoveries selected", "Total assays"],
        integers={"evidence_controls_available", "evidence_controls_selected",
                  "strict_double_cold_discoveries_selected", "total_pair_assays"},
    )
    primary_threshold = float(lock_manifest["models"][primary]["score_threshold"])
    primary_scores = score_frames[primary]
    primary_passing = primary_scores["model_score"].ge(primary_threshold)
    passing_by_target = primary_scores.assign(_passes=primary_passing).groupby(
        "peptide_design_code", sort=True
    )["_passes"].sum().astype(int)
    nonzero_passing = passing_by_target.loc[passing_by_target.gt(0)]
    _require(len(nonzero_passing) > 0, "primary threshold passes no prospective candidates")
    candidate_rows = []
    for target, peptide in project_spec["target_sequences"].items():
        block = menus[primary].loc[
            menus[primary]["peptide_design_code"].astype(str).eq(str(target))
        ].sort_values("model_rank", kind="mergesort")
        if len(block):
            ranked = ", ".join(
                f"{row.affibody_design_code} ({Decimal(str(float(row.model_score))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)})"
                for row in block.itertuples(index=False)
            )
        else:
            missing = no_candidate_targets.loc[
                no_candidate_targets["model_id"].astype(str).eq(primary)
                & no_candidate_targets["peptide_design_code"].astype(str).eq(str(target))
            ]
            _require(len(missing) == 1, f"missing no-candidate receipt for {target}")
            maximum = Decimal(str(float(missing.iloc[0]["maximum_model_score"]))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            cutoff = Decimal(str(primary_threshold)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            ranked = f"None passes the lock (maximum {maximum}; threshold {cutoff})"
        candidate_rows.append({
            "peptide_design_code": target,
            "peptide_9mer_sequence": peptide,
            "ranked_candidates": ranked,
        })
    primary_candidate_md = markdown_table(
        pd.DataFrame(candidate_rows),
        ["peptide_design_code", "peptide_9mer_sequence", "ranked_candidates"],
        ["Peptide", "9-aa sequence", "Primary model candidates in rank order: code (score)"],
    )
    order_rows = []
    for target, peptide in project_spec["target_sequences"].items():
        block = order.loc[
            order["peptide_design_code"].astype(str).eq(str(target))
        ].sort_values("assay_slot", kind="mergesort")
        if len(block):
            evidence = block.loc[block["tier_role"].astype(str).eq("control")]
            discovery = block.loc[block["tier_role"].astype(str).eq("discovery")]
            evidence_text = ", ".join(
                f"{row.affibody_design_code} (count {int(row.pooled_r009_r010_count)})"
                for row in evidence.itertuples(index=False)
            ) or "None"
            discovery_text = ", ".join(
                f"{row.affibody_design_code} "
                f"({Decimal(str(float(row.model_score))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)})"
                for row in discovery.itertuples(index=False)
            ) or "None"
        else:
            _require(
                str(target) in set(no_candidate_targets["peptide_design_code"].astype(str)),
                f"target {target} absent from both assay order and no-candidate receipt",
            )
            evidence_text = "None"
            discovery_text = "None passes the locked threshold"
        order_rows.append({
            "peptide_design_code": target,
            "peptide_9mer_sequence": peptide,
            "evidence_candidates": evidence_text,
            "discovery_candidates": discovery_text,
        })
    order_candidate_md = markdown_table(
        pd.DataFrame(order_rows),
        ["peptide_design_code", "peptide_9mer_sequence", "evidence_candidates",
         "discovery_candidates"],
        ["Peptide", "9-aa sequence", "Sequencing-evidence comparators: code (pooled count)",
         "Model discoveries: code (score)"],
    )
    historical_top1_context = ""
    if retrospective_per_peptide is not None:
        historical_codes = (
            retrospective_per_peptide.groupby("model")["top1_affibody"]
            .agg(lambda values: sorted(set(map(str, values))))
            .to_dict()
        )
        if all(len(codes) == 1 for codes in historical_codes.values()):
            union = sorted({code for codes in historical_codes.values() for code in codes})
            if len(union) == 1:
                historical_top1_context = (
                    f"On the already-measured 108-pair panel, both deployed rules ranked "
                    f"`{union[0]}` first for all nine peptides. That is historical behavior "
                    f"on the old 12-Affibody panel, not the prospective result below.\n\n"
                )
    return f"""# LibA candidate selection for prospective wet-lab testing

This handoff applies already-locked models to nine LibA peptide targets. The
primary discovery model is **{primary_name}**. Because the sequence models have
weak peptide-specific ranking on the measured LibA panel, we made a pragmatic,
retrospectively informed choice not to rely on model rank alone for the batch
composition. The batch targets three unmeasured pairs supported by
direct pooled-R009/R010 sequencing evidence and seven strict-double-cold model discoveries
per peptide when eligible candidates exist. If fewer than three evidence comparators exist,
the unused slots are filled only by model discoveries that pass the locked
threshold. A target with neither kind of candidate is explicitly reported and
omitted from the order; no comparator or discovery is fabricated. Candidate scores were
used only to determine threshold eligibility and rank discoveries within the same peptide.
Retention measurements were not used to train the models, choose the deployment
threshold, or choose exact candidates within either tier; they informed only the
pragmatic decision to mix evidence comparators and model discoveries in this first batch.
Retention is the percentage of assay signal remaining after the construct is
cleaved. It is the project's laboratory readout, not a direct measurement of
binding affinity.

## Training and evaluation data

{project_spec['training_data']['description']}

Weak labels: {project_spec['training_data']['weak_label_definition']}

Split rule: {project_spec['training_data']['split_rule']}

The table below uses the supplied retrospective comparison of directly measured
retention pairs. Those measurements describe prior performance; they do not
enter the prospective candidate-selection rule.

## What the wet lab should test

`ORDER_THIS_BATCH.csv` gives the exact tiered peptide--Affibody pairs. It is a
list of specified pair assays, not a Cartesian cross between every peptide and
every listed Affibody. Each row has a stable sample ID and a blank matching row
in `RESULT_ENTRY.csv`. `PRIMARY_MODEL_ONLY_BATCH.csv` preserves the original
model-only proposal for comparison, but it is not the recommended first batch.
`NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD.csv` records every model--target case with
no passing candidate, including its maximum score and locked threshold. Such
rows are diagnostic and are not orderable.

The {FIRST_BATCH_CONTROL_TARGET}+{FIRST_BATCH_TOTAL_PER_PEPTIDE - FIRST_BATCH_CONTROL_TARGET}
allocation is fixed for this prospective handoff: {FIRST_BATCH_CONTROL_TARGET}
sequencing-evidence comparators provide an immediate check that the assay and
pooled-selection evidence transfer, while
{FIRST_BATCH_TOTAL_PER_PEPTIDE - FIRST_BATCH_CONTROL_TARGET} discoveries
leave most capacity for testing strict generalization. Discoveries exclude pairs
flagged as high-confidence weak negatives. Here, that means pairs with at least
three R001 reads that never appeared in any positive-selection round R002--R014.
Selection first prefers Affibody codes
that differ at two or more of the four designed positions from comparators and
discoveries already chosen for that peptide, then relaxes only to fill otherwise
available assay slots.

{first_batch_md}

The actual mixed batch contains {len(order):,} specified pair assays and
{order['provider_displayed_58aa_affibody_sequence'].nunique():,} distinct displayed
Affibody sequences; an Affibody can be paired with more than one peptide.

The compact table below shows the actual mixed batch. Evidence comparators show
their pooled R009+R010 counts; model discoveries show scores rounded to two
decimals. The scores are not binding probabilities or predicted retention
percentages. Values displayed as `1.00` are rounded and need not be exact ties;
the full-precision values in `ORDER_THIS_BATCH.csv` determine the order.

{order_candidate_md}

### Secondary model-only comparison

The table below lists every threshold-passing entry in the separate primary
model-only menu. It is retained to show what the model alone would have selected;
it is not the recommended mixed batch.

{primary_candidate_md}

Each model directory also contains a separately named strict-double-cold menu:
the peptide and the Affibody identity were both absent from that model's strict
training set. It ranks only within each peptide and uses the same outcome-blind
weak-label threshold. The full selection-missed universe contains
{project_spec['expected_double_cold_candidate_pairs']:,} such pairs and
{project_spec['expected_peptide_cold_only_candidate_pairs']:,} peptide-cold-only
pairs. The main menu is retained as the yield-oriented view; the double-cold menu
is the stronger generalization view.

This is a computational shortlist, not yet an order-ready construct list. The
displayed 58-aa Affibody sequence is the provider-displayed/model-input sequence,
not a final vendor or cloning construct. The hidden N-terminal `MA`,
vector, and tag context must be confirmed. Likewise, the SMART--HLA--linker--
peptide sequence is a model/assay-side input; signal peptide, tag, linker, and
vector details must be verified before cloning.

## Models that can currently be used

Only models marked scientifically eligible in the supplied weak-OOF lock are
included. No result is silently hard-coded into this report.

MINT layer 9 is primary because its retention-blind weak-label out-of-fold
within-peptide AP was {primary_weak_ap:.6f}, compared with
{comparison_weak_ap_text}. The rounded table shows both values as `0.71`, but the
unrounded locked comparison selected MINT layer 9.

For the MINT model, the two full chain sequences enter a frozen MINT backbone;
layer-9 residue vectors are averaged within each chain and a trained logistic
head scores the concatenated chain averages. The equal-logit comparator averages
the MINT head's logit and the six-position additive model's logit before applying
the sigmoid;
the additive model uses peptide positions 4 and 5 plus the four designed
Affibody positions.

{rec_md}

Other sequence experiments were excluded for specific reasons: LoRA showed no
reproducible gain; PNU and later-round labels did not improve the locked
weak-label ranking objective; and the meta-gradient and trajectory-rater models
used retention information, so they are ineligible for retention-blind
prospective selection.

## Retrospective comparison on the existing retention panel

These values describe performance on already-measured pairs. Scores are not
retention percentages or calibrated binding probabilities. All displayed
metrics are rounded to two decimal places; machine-readable CSVs retain the
input precision.

{performance_table}

At each retention-optimized descriptive cutoff, the old panel would have
produced the following recommendations. This table is explanatory, not a rule
for the new batch.

{retrospective_recommendation_md}

## Two different kinds of threshold

The weak-label deployment threshold was fixed from selection-derived weak
labels and is the only threshold used to create the prospective menus. The
retention-optimized threshold maximizes a stated retrospective criterion on an
already-measured panel; it is descriptive and must not be used to select this
batch or claimed as prospectively validated.

For the primary MINT model, {int(primary_passing.sum()):,} of
{len(primary_scores):,} scored pairs pass the global weak-label threshold. Every
non-DP target has at least {int(nonzero_passing.min()):,} passing pairs, so the
within-peptide ranking and ten-candidate cap--not the cutoff--determine those
eight menus. DP is the exception: no pair passes. This sharp peptide-to-peptide
shift is another reason not to interpret the score as a calibrated probability.

{threshold_md}

## Separate pooled-selection evidence tier

The model-scored universe contains {project_spec['expected_selection_missed_candidate_pairs']:,}
selection-missed pairs. It is kept separate from
{project_spec['evidence_positive_control']['expected_pairs']:,} unmeasured current-target pairs
that already belong to pooled R009+R010's top 2%. The latter have direct sequencing
evidence but have not been confirmed as binders in the direct retention assay.
They are not model rescues. Their full menu remains separate
from every model menu. The first assay batch intentionally samples both tiers,
with an explicit tier label on every row.

On the previously measured LibA panel, ranking by raw pooled count gave:

{evidence_md}

This raw-count result is not evidence of generalization to new sequence space: it
uses direct evidence from the same peptide--Affibody pair. If these pairs have not
already been tested, this separate tier may be more defensible than learned-model
rescues when the immediate goal is wet-lab yield. The candidate menus and supplied
retrospective results are interpreted separately below. No binder-yield estimate
is assigned to the unmeasured menu; the new assay is the test.

The learned-model diagnostics are:

{historical_top1_context}{model_diagnostics}

The learned-model recommendations therefore remain a distinct prospective test.

## Structure-model availability

{esmfold_exclusion}
{unavailable_lines}

Unavailable structure families are omitted rather than represented by a
different model under the same name.

## How the prospective result will be read

After laboratory results return, prospective yield is the number of valid
pair assays with retention at 30 minutes of at least 75%, divided by the number
of valid returned pair assays. Report this separately for pooled-selection-evidence
comparators and strict-double-cold model discoveries, then for the combined batch.
Technical failures are excluded. The released result template contains no
retention outcomes.
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-oof-locks", type=Path, required=True)
    parser.add_argument("--project-spec", type=Path, required=True)
    parser.add_argument("--candidate-scores", action="append", default=[],
                        metavar="MODEL_ID=PATH", required=True)
    parser.add_argument("--evidence-positive-controls", type=Path, required=True)
    parser.add_argument("--evidence-control-comparison", type=Path, required=True)
    parser.add_argument("--model-comparison", type=Path, required=True)
    parser.add_argument("--retrospective-per-peptide", type=Path,
                        help="Optional report-only frozen per-peptide evaluation table.")
    parser.add_argument("--esmfold-weak-selection", type=Path,
                        help="Optional report-only retention-blind ESMFold2 family gate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    final_output = args.output_dir.resolve()
    private_root = (REPO_ROOT / "private_data").resolve()
    _require(final_output != private_root and private_root in final_output.parents,
             "output must be below private_data")
    _require(not final_output.exists(), "output exists; refusing overwrite")

    lock_path = args.weak_oof_locks.resolve()
    lock_manifest = load_lock_manifest(lock_path)
    eligible_ids = [
        model_id for model_id, lock in lock_manifest["models"].items()
        if lock.get("scientifically_eligible") is True
    ]
    project_path = args.project_spec.resolve()
    project_spec = load_project_spec(project_path, eligible_ids)
    target_sequences = {
        str(key): str(value) for key, value in project_spec["target_sequences"].items()
    }
    score_paths = parse_model_paths(args.candidate_scores)
    _require(set(score_paths) == set(eligible_ids),
             "candidate-score inputs must exactly match scientifically eligible models")

    # Prospective menus are fully fixed before the retention-derived comparison
    # table is opened below.
    score_frames: dict[str, pd.DataFrame] = {}
    score_manifests: dict[str, dict] = {}
    menus: dict[str, pd.DataFrame] = {}
    compact: dict[str, pd.DataFrame] = {}
    rosters: dict[str, pd.DataFrame] = {}
    qc_summaries: dict[str, pd.DataFrame] = {}
    double_cold_menus: dict[str, pd.DataFrame] = {}
    for model_id in eligible_ids:
        score_manifests[model_id] = validate_candidate_score_manifest(
            model_id, score_paths[model_id],
            int(project_spec["expected_selection_missed_candidate_pairs"]),
            lock_manifest["models"][model_id],
        )
        score_frames[model_id] = load_candidate_scores(
            model_id, score_paths[model_id], target_sequences
        )
        actual_membership_sha256 = hashlib.sha256(
            "\n".join(sorted(score_frames[model_id]["pair_uid"].astype(str))).encode("ascii")
        ).hexdigest()
        _require(
            score_manifests[model_id]["pair_uid_membership_sha256"]
            == actual_membership_sha256,
            f"{model_id} candidate membership differs from its producer manifest",
        )
        _require(
            len(score_frames[model_id])
            == int(project_spec["expected_selection_missed_candidate_pairs"]),
            f"{model_id} does not cover the exact selection-missed candidate universe",
        )
        menu = select_candidates(score_frames[model_id], lock_manifest["models"][model_id])
        menu = add_liba_mapping(menu)
        menu.insert(0, "candidate_menu_source_model", model_id)
        menus[model_id] = menu
        compact[model_id] = compact_codes_by_target(
            menu, int(lock_manifest["models"][model_id]["max_candidates_per_peptide"])
        )
        rosters[model_id] = unique_sequence_roster(menu)
        qc_summaries[model_id] = candidate_qc_summary(menu)
        double_frame = score_frames[model_id].loc[
            ~score_frames[model_id]["affibody_identity_seen_in_strict_training"].astype(bool)
        ]
        _require(len(double_frame) == int(project_spec["expected_double_cold_candidate_pairs"]),
                 f"{model_id} double-cold universe count differs from project specification")
        double_menu = add_liba_mapping(select_candidates(
            double_frame, lock_manifest["models"][model_id]
        ))
        double_menu.insert(0, "candidate_menu_source_model", model_id)
        double_menu.insert(1, "generalization_scope", "strict_double_cold")
        double_cold_menus[model_id] = double_menu

    membership_hashes = {
        manifest["pair_uid_membership_sha256"] for manifest in score_manifests.values()
    }
    _require(len(membership_hashes) == 1,
             "eligible candidate-score manifests attest different pair memberships")
    no_candidate_targets = no_candidate_above_threshold_table(
        score_frames, menus, lock_manifest, target_sequences
    )
    rank_only_diagnostic = rank_only_below_threshold_diagnostic(
        score_frames, no_candidate_targets, lock_manifest
    )

    universe_identity_columns = [
        "pair_uid", "peptide_design_code", "peptide_9mer_sequence",
        "affibody_design_code", "provider_displayed_58aa_affibody_sequence",
    ]
    reference_model = eligible_ids[0]
    reference_universe = score_frames[reference_model][universe_identity_columns].sort_values(
        "pair_uid", kind="mergesort"
    ).reset_index(drop=True)
    for model_id in eligible_ids[1:]:
        observed_universe = score_frames[model_id][universe_identity_columns].sort_values(
            "pair_uid", kind="mergesort"
        ).reset_index(drop=True)
        _require(observed_universe.equals(reference_universe),
                 f"{model_id} candidate universe differs from {reference_model}")

    evidence_path = args.evidence_positive_controls.resolve()
    evidence_full = load_evidence_positive_controls(
        evidence_path,
        target_sequences,
        int(project_spec["evidence_positive_control"]["expected_pairs"]),
    )
    model_scaffold = affibody_scaffold_signature(
        reference_universe["provider_displayed_58aa_affibody_sequence"].iloc[0]
    )
    evidence_scaffold = affibody_scaffold_signature(
        evidence_full["provider_displayed_58aa_affibody_sequence"].iloc[0]
    )
    _require(model_scaffold == evidence_scaffold,
             "model candidates and evidence controls use different Affibody scaffolds")
    evidence_ids = set(evidence_full["pair_uid"].astype(str))
    for model_id, frame in score_frames.items():
        _require(evidence_ids.isdisjoint(frame["pair_uid"].astype(str)),
                 f"{model_id} selection-missed universe overlaps evidence-positive controls")
    evidence_menu = evidence_review_menu(
        evidence_full,
        int(project_spec["evidence_positive_control"]["max_controls_per_peptide"]),
    )
    evidence_roster = evidence_sequence_roster(evidence_menu)
    evidence_compact = compact_evidence_codes_by_target(
        evidence_menu,
        int(project_spec["evidence_positive_control"]["max_controls_per_peptide"]),
        target_sequences,
    )

    primary_id = lock_manifest["primary_model_id"]
    first_batch, first_batch_summary = build_tiered_first_batch(
        score_frames[primary_id],
        lock_manifest["models"][primary_id],
        evidence_full,
        target_sequences,
        primary_id,
    )
    # The mixed batch is fixed here, before either retention-derived audit is
    # opened. The old model-only batch remains a separate comparison artifact.

    evidence_audit_path = args.evidence_control_comparison.resolve()
    evidence_audit = load_evidence_retrospective_audit(evidence_audit_path)
    evidence_audit_receipt_path = (
        evidence_audit_path / "manifest.json"
        if evidence_audit_path.is_dir()
        else evidence_audit_path
    )
    _require(evidence_audit_receipt_path.is_file(),
             "evidence benchmark completion receipt is missing")
    validate_evidence_menu_against_benchmark(evidence_menu, evidence_audit)
    comparison_path = args.model_comparison.resolve()
    comparison = load_retrospective_comparison(comparison_path, eligible_ids)
    comparison = comparison.merge(
        pd.DataFrame([
            {"model_id": model_id,
             "locked_display_name": lock_manifest["models"][model_id]["display_name"]}
            for model_id in eligible_ids
        ]), on="model_id", validate="one_to_one"
    )
    _require(comparison["display_name"].eq(comparison["locked_display_name"]).all(),
             "model display names differ between lock and comparison")
    comparison = comparison.drop(columns="locked_display_name")
    retrospective_per_peptide_path = getattr(args, "retrospective_per_peptide", None)
    retrospective_per_peptide = None
    if retrospective_per_peptide_path is not None:
        retrospective_per_peptide_path = retrospective_per_peptide_path.resolve()
        retrospective_per_peptide = load_retrospective_per_peptide_context(
            retrospective_per_peptide_path, eligible_ids, target_sequences
        )
    esmfold_selection_path = getattr(args, "esmfold_weak_selection", None)
    esmfold_weak_selection = None
    if esmfold_selection_path is not None:
        esmfold_selection_path = esmfold_selection_path.resolve()
        esmfold_weak_selection = load_esmfold_weak_selection(esmfold_selection_path)

    primary = menus[primary_id].copy()
    target_order = list(target_sequences)
    primary["target_order"] = primary["peptide_design_code"].map(
        {target: index + 1 for index, target in enumerate(target_order)}
    ).astype(np.int64)
    primary = primary.sort_values(["target_order", "model_rank"], kind="mergesort")
    primary["sample_id"] = [
        f"LIBA-MODELONLY-{target}-{int(slot):02d}"
        for target, slot in zip(primary["peptide_design_code"], primary["assay_slot"])
    ]
    order_columns = [
        "sample_id", "peptide_design_code", "peptide_9mer_sequence", "assay_slot",
        "model_rank", "affibody_design_code",
        "provider_displayed_58aa_affibody_sequence", "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence", "pair_uid", "model_score",
    ]
    model_only_order = primary[order_columns].copy()
    _require(model_only_order["sample_id"].is_unique,
             "primary model-only sample IDs are not unique")

    order_columns = [
        "sample_id", "peptide_design_code", "peptide_9mer_sequence", "assay_slot",
        "candidate_tier", "tier_role", "within_tier_rank", "selection_basis",
        "affibody_design_code", "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence", "model_input_smart_hla_linker_peptide_sequence",
        "pair_uid", "primary_model_id", "model_score", "pooled_r009_r010_count",
        "r009_count", "r010_count", "weak_label_deployment_threshold",
        "diversity_selection_phase", "minimum_hamming_to_another_batch_code",
        "observed_in_any_raw_round", "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training", "high_confidence_weak_negative",
        "affibody_code_position_mapping",
    ]
    for column in order_columns:
        if column not in first_batch.columns:
            first_batch[column] = np.nan
    order = first_batch[order_columns].copy()
    _require(order["sample_id"].is_unique, "tiered first-batch sample IDs are not unique")
    _require(
        set(order["sample_id"].astype(str)).isdisjoint(
            set(model_only_order["sample_id"].astype(str))
        ),
        "tiered and model-only batch sample-ID namespaces overlap",
    )
    _require(order.groupby("peptide_design_code").size().le(10).all(),
             "tiered first batch exceeds ten Affibodies per peptide")
    result_entry = order[["sample_id", "pair_uid", "peptide_design_code",
                          "peptide_9mer_sequence", "assay_slot", "candidate_tier",
                          "tier_role", "within_tier_rank",
                          "affibody_design_code"]].copy()
    outcome_fields = [
        "retention_at_30min_percent", "binder_ge_75", "technical_failure_status",
        "replicate_id", "assay_date", "experimental_batch_id", "notes",
    ]
    for column in outcome_fields:
        result_entry[column] = ""
    _require(result_entry[outcome_fields].eq("").all().all(),
             "result-entry fields must be blank at release")

    recommendations = pd.DataFrame([
        {
            "model_id": model_id,
            "display_name": lock_manifest["models"][model_id]["display_name"],
            "prospective_role": (
                "Primary discovery component of tiered batch"
                if model_id == primary_id else "Separate comparison menu"
            ),
            "candidate_pairs": len(menus[model_id]),
            "unique_affibody_sequences": len(rosters[model_id]),
        }
        for model_id in eligible_ids
    ])
    threshold_table = comparison[["model_id", "display_name",
                                  "retention_optimized_threshold"]].copy()
    threshold_table["weak_label_deployment_threshold"] = threshold_table["model_id"].map(
        {model_id: float(lock_manifest["models"][model_id]["score_threshold"])
         for model_id in eligible_ids}
    )
    threshold_table = threshold_table.rename(columns={
        "retention_optimized_threshold": "retention_optimized_descriptive_threshold"
    })
    threshold_table["threshold_use"] = (
        "Deploy weak-label threshold only; retention-optimized threshold is descriptive"
    )
    retrospective_threshold_recommendations = comparison[[
        "model_id", "display_name", "retention_optimized_threshold",
        "pairs_above_retention_optimized_threshold",
        "binders_above_retention_optimized_threshold", "retrospective_precision",
        "retrospective_recall", "retrospective_f1",
    ]].copy()
    retrospective_threshold_recommendations["use"] = (
        "Retrospective description only; do not use for prospective selection"
    )
    report = build_report(
        lock_manifest, project_spec, evidence_audit, score_frames, menus, order,
        no_candidate_targets, comparison, retrospective_per_peptide,
        esmfold_weak_selection,
        recommendations, threshold_table, first_batch_summary
    )

    final_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = Path(tempfile.mkdtemp(prefix=f".{final_output.name}.staging-",
                                  dir=final_output.parent))
    os.chmod(stage, 0o700)
    paths: dict[str, Path] = {}
    paths["order_this_batch"] = stage / "ORDER_THIS_BATCH.csv"
    paths["primary_model_only_batch"] = stage / "PRIMARY_MODEL_ONLY_BATCH.csv"
    paths["first_batch_tier_summary"] = stage / "FIRST_BATCH_TIER_SUMMARY.csv"
    paths["no_candidate_above_threshold"] = (
        stage / "NO_CANDIDATE_ABOVE_LOCKED_THRESHOLD.csv"
    )
    paths["rank_only_below_threshold_diagnostic"] = (
        stage / "RANK_ONLY_BELOW_THRESHOLD_DIAGNOSTIC_DO_NOT_ORDER.csv"
    )
    paths["result_entry"] = stage / "RESULT_ENTRY.csv"
    paths["model_recommendations"] = stage / "model_recommendations.csv"
    paths["retrospective_model_comparison"] = stage / "retrospective_model_comparison.csv"
    paths["threshold_context"] = stage / "threshold_context.csv"
    paths["retrospective_threshold_recommendations"] = (
        stage / "retrospective_threshold_recommendations.csv"
    )
    paths["evidence_positive_all_pairs"] = (
        stage / "evidence_positive_control_tier_all_unmeasured_pairs.csv"
    )
    paths["evidence_positive_review_menu"] = (
        stage / "evidence_positive_control_review_menu_by_target.csv"
    )
    paths["evidence_positive_compact_codes"] = (
        stage / "evidence_positive_control_compact_codes_by_target.csv"
    )
    paths["evidence_positive_sequence_roster"] = (
        stage / "evidence_positive_control_unique_affibody_sequence_review_roster.csv"
    )
    paths["report"] = stage / "liba_candidate_selection_report.md"
    order.to_csv(paths["order_this_batch"], index=False)
    model_only_order.to_csv(paths["primary_model_only_batch"], index=False)
    first_batch_summary.to_csv(paths["first_batch_tier_summary"], index=False)
    no_candidate_targets.to_csv(paths["no_candidate_above_threshold"], index=False)
    rank_only_diagnostic.to_csv(
        paths["rank_only_below_threshold_diagnostic"], index=False
    )
    result_entry.to_csv(paths["result_entry"], index=False)
    recommendations.to_csv(paths["model_recommendations"], index=False)
    comparison.to_csv(paths["retrospective_model_comparison"], index=False)
    threshold_table.to_csv(paths["threshold_context"], index=False)
    retrospective_threshold_recommendations.to_csv(
        paths["retrospective_threshold_recommendations"], index=False
    )
    evidence_full.to_csv(paths["evidence_positive_all_pairs"], index=False)
    evidence_menu.to_csv(paths["evidence_positive_review_menu"], index=False)
    evidence_compact.to_csv(paths["evidence_positive_compact_codes"], index=False)
    evidence_roster.to_csv(paths["evidence_positive_sequence_roster"], index=False)
    paths["report"].write_text(report, encoding="utf-8")

    model_outputs = {}
    for model_id in eligible_ids:
        model_dir = stage / f"model_{model_id}"
        model_dir.mkdir(mode=0o700)
        model_paths = {
            "candidate_menu": model_dir / f"{model_id}_candidate_menu_all_9_targets.csv",
            "compact_codes": model_dir / f"{model_id}_compact_codes_by_target.csv",
            "sequence_roster": model_dir / f"{model_id}_unique_affibody_sequence_review_roster.csv",
            "qc_summary": model_dir / f"{model_id}_candidate_qc_summary_all_9_targets.csv",
            "double_cold_menu": model_dir / f"{model_id}_strict_double_cold_menu_all_9_targets.csv",
            "double_cold_compact_codes": model_dir / f"{model_id}_strict_double_cold_compact_codes_by_target.csv",
        }
        menus[model_id].to_csv(model_paths["candidate_menu"], index=False)
        compact[model_id].to_csv(model_paths["compact_codes"], index=False)
        rosters[model_id].to_csv(model_paths["sequence_roster"], index=False)
        qc_summaries[model_id].to_csv(model_paths["qc_summary"], index=False)
        double_cold_menus[model_id].to_csv(model_paths["double_cold_menu"], index=False)
        compact_codes_by_target(
            double_cold_menus[model_id],
            int(lock_manifest["models"][model_id]["max_candidates_per_peptide"]),
        ).to_csv(model_paths["double_cold_compact_codes"], index=False)
        for path in model_paths.values():
            os.chmod(path, 0o600)
        model_outputs[model_id] = model_paths

    workbook_path = stage / "liba_wetlab_candidate_handoff.xlsx"
    readme = pd.DataFrame([
        ("First batch", "Mixed batch: target 3 pooled-R009/R010 evidence comparators plus 7 strict-double-cold primary-model discoveries per peptide; the allocation was informed by retrospective model weakness, while exact within-tier choices remained retention-blind"),
        ("Primary discovery model", lock_manifest["models"][primary_id]["display_name"]),
        ("Missing evidence comparators", "If fewer than 3 evidence comparators exist, discovery slots increase only for candidates above the locked threshold; no comparator is fabricated"),
        ("Empty target", "DP has no evidence comparator and no model candidate above the locked threshold, so it is audited in NO_CANDIDATE but absent from ORDER_THIS_BATCH; do not lower or pad the threshold"),
        ("Rank-only diagnostic", "RANK_ONLY_BELOW_THRESHOLD_DIAGNOSTIC_DO_NOT_ORDER.csv is non-orderable and nonrecommended; it exists only to show why an empty target failed the locked cutoff"),
        ("Batch geometry", f"{len(order)} specified pair assays using {order['provider_displayed_58aa_affibody_sequence'].nunique()} unique displayed/model-input Affibody sequences; NOT a Cartesian cross"),
        ("Affibody sequence warning", "Displayed/model-input 58-aa sequence is NOT vendor/cloning-ready until hidden N-terminal MA/vector/tag context is confirmed"),
        ("Assay-side warning", "SMART-HLA-linker-peptide is a model/assay input, NOT cloning-ready until signal peptide/tag/linker/vector context is verified"),
        ("Threshold rule", "Prospective menus use weak-label deployment thresholds only; retention-optimized thresholds are descriptive"),
        ("Prospective yield", "Report valid retention>=75% results separately for evidence comparators and model discoveries, plus the combined batch; exclude technical failures"),
        ("Legacy machine terms", "Fields, sheet names, or filenames containing 'control' are compatibility names for unmeasured sequencing-evidence comparators, not confirmed positive controls"),
    ], columns=["item", "instruction"])
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sheets = {
            "ORDER_THIS_BATCH": order,
            "README": readme,
            "MODEL_ONLY_BATCH": model_only_order,
            "BATCH_TIER_SUMMARY": first_batch_summary,
            "NO_CANDIDATE": no_candidate_targets,
            "RANK_ONLY_DO_NOT_ORDER": rank_only_diagnostic,
            "RESULT_ENTRY": result_entry,
            "MODEL_RECOMMENDATIONS": recommendations,
            "RETROSPECTIVE_RESULTS": comparison,
            "RETRO_RECOMMENDATIONS": retrospective_threshold_recommendations,
            "THRESHOLD_CONTEXT": threshold_table,
            "EVIDENCE_CONTROL_MENU": evidence_menu,
            "EVIDENCE_CONTROL_CODES": evidence_compact,
            "EVIDENCE_CONTROL_ROSTER": evidence_roster,
        }
        for model_id in eligible_ids:
            safe = re.sub(r"[^A-Za-z0-9_]", "_", model_id)[:15]
            sheets[f"{safe}_MENU"] = menus[model_id]
            sheets[f"{safe}_CODES"] = compact[model_id]
            sheets[f"{safe}_ROSTER"] = rosters[model_id]
            sheets[f"{safe}_QC"] = qc_summaries[model_id]
            sheets[f"{safe}_DC_MENU"] = double_cold_menus[model_id]
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    os.chmod(workbook_path, 0o600)
    for path in paths.values():
        os.chmod(path, 0o600)

    output_records = {
        name: {"path": str(path.relative_to(stage)), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    output_records["workbook"] = {
        "path": workbook_path.name,
        "sha256": sha256_file(workbook_path),
        "first_sheet": "ORDER_THIS_BATCH",
    }
    for model_id, mapping in model_outputs.items():
        output_records[f"model_{model_id}"] = {
            name: {"path": str(path.relative_to(stage)), "sha256": sha256_file(path)}
            for name, path in mapping.items()
        }
    release_manifest = {
        "schema_version": "liba-wetlab-candidate-handoff-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "producer": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "argv": [str(value) for value in sys.argv],
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "library": "LibA",
        "target_count": 9,
        "primary_model_id": primary_id,
        "scientifically_eligible_model_ids": eligible_ids,
        "selection_policy": (
            "The three-evidence-comparator plus seven-discovery tier allocation was informed by "
            "retrospective model weakness. Given that operational allocation, retention did not "
            "score or rank candidates, choose exact pair identities within either tier, tune model "
            "weights, or set the deployment cutoff; the candidate menus are fixed before this run "
            "loads its retention-derived comparison tables."
        ),
        "threshold_semantics": {
            "deployment": "weak-label OOF threshold; used for prospective selection",
            "retrospective": "retention-optimized descriptive threshold; never used for selection",
        },
        "legacy_schema_note": (
            "Machine-readable field names and filenames containing 'control' are retained "
            "for compatibility. They denote unmeasured pooled-selection evidence comparators, "
            "not experimentally confirmed positive controls."
        ),
        "structure_families_unavailable": project_spec["unavailable_structure_families"],
        "primary_pair_assays": int(len(order)),
        "primary_unique_affibody_sequences": int(
            order["provider_displayed_58aa_affibody_sequence"].nunique()
        ),
        "first_batch": {
            "pair_assays": int(len(order)),
            "evidence_control_assays": int(order["tier_role"].eq("control").sum()),
            "strict_double_cold_model_discovery_assays": int(
                order["tier_role"].eq("discovery").sum()
            ),
            "maximum_affibodies_per_peptide": FIRST_BATCH_TOTAL_PER_PEPTIDE,
            "target_controls_per_peptide_when_available": FIRST_BATCH_CONTROL_TARGET,
            "preferred_minimum_code_hamming_distance": (
                FIRST_BATCH_PREFERRED_CODE_HAMMING
            ),
            "high_confidence_weak_negatives_allowed_in_discovery_tier": False,
            "targets_with_orderable_candidates": int(
                order["peptide_design_code"].nunique()
            ),
            "targets_searched_and_audited": len(target_sequences),
        },
        "no_candidate_above_locked_threshold_rows": int(len(no_candidate_targets)),
        "rank_only_nonorderable_diagnostic_rows": int(len(rank_only_diagnostic)),
        "primary_model_only_pair_assays": int(len(model_only_order)),
        "selection_missed_candidate_pairs_per_model": int(
            project_spec["expected_selection_missed_candidate_pairs"]
        ),
        "strict_double_cold_candidate_pairs_per_model": int(
            project_spec["expected_double_cold_candidate_pairs"]
        ),
        "peptide_cold_only_candidate_pairs_per_model": int(
            project_spec["expected_peptide_cold_only_candidate_pairs"]
        ),
        "evidence_positive_control_pairs": int(len(evidence_full)),
        "evidence_positive_review_pairs": int(len(evidence_menu)),
        "outcome_access": {
            "retention_used_for_model_training": False,
            "retention_used_for_deployment_cutoff": False,
            "retention_used_for_candidate_scoring_or_ranking": False,
            "retention_used_for_exact_within_tier_pair_identity_selection": False,
            "retrospective_performance_informed_tier_allocation": True,
            "result_template_outcome_values_present_at_release": False,
        },
        "inputs": {
            "weak_oof_locks": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
            "project_spec": {"path": str(project_path), "sha256": sha256_file(project_path)},
            "model_comparison": {"path": str(comparison_path),
                                 "sha256": sha256_file(comparison_path)},
            **({
                "retrospective_per_peptide_context": {
                    "path": str(retrospective_per_peptide_path),
                    "sha256": sha256_file(retrospective_per_peptide_path),
                    "loaded_after_candidate_menus_were_fixed": True,
                }
            } if retrospective_per_peptide_path is not None else {}),
            **({
                "esmfold_weak_selection": {
                    "path": str(esmfold_selection_path),
                    "sha256": sha256_file(esmfold_selection_path),
                    "loaded_after_candidate_menus_were_fixed": True,
                    "retention_labels_read": False,
                }
            } if esmfold_selection_path is not None else {}),
            "evidence_positive_controls": {
                "path": str(evidence_path), "sha256": sha256_file(evidence_path)
            },
            "evidence_control_comparison": {
                "path": str(evidence_audit_path),
                "receipt_path": str(evidence_audit_receipt_path),
                "receipt_sha256": sha256_file(evidence_audit_receipt_path),
                "loaded_after_candidate_menus_were_fixed": True,
            },
            "candidate_scores": {
                model_id: {
                    "path": str(path), "sha256": sha256_file(path),
                    "manifest_path": str(path.parent / "manifest.json"),
                    "manifest_sha256": sha256_file(path.parent / "manifest.json"),
                    "pair_uid_membership_sha256":
                        score_manifests[model_id]["pair_uid_membership_sha256"],
                    "deployment_binding": score_manifests[model_id]["deployment_binding"],
                }
                for model_id, path in score_paths.items()
            },
        },
        "outputs": output_records,
        "validation": {
            "exact_nine_target_scope": True,
            "eligible_models_match_score_inputs": True,
            "candidate_score_manifests_and_deployment_bindings_match": True,
            "candidate_score_pair_membership_hash_shared": True,
            "no_retention_columns_in_candidate_scores": True,
            "model_and_evidence_positive_tiers_disjoint": True,
            "tiered_first_batch_maximum_ten_per_peptide": True,
            "zero_control_targets_not_padded_or_fabricated": True,
            "targets_without_threshold_passing_candidates_absent_from_order": True,
            "discovery_tier_strict_double_cold": True,
            "discovery_tier_excludes_high_confidence_weak_negatives": True,
            "result_entry_fields_blank": True,
            "atomic_directory_publish": True,
        },
    }
    manifest_path = stage / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(release_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    _require(not final_output.exists(), "final output appeared during staging")
    os.replace(stage, final_output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
