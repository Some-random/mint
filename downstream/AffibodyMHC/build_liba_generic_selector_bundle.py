#!/usr/bin/env python3
"""Build the hash-bound LibA selector bundle used by the wet-lab compiler.

This bridges generic weak-label selector candidate names to concise deployment
IDs without losing provenance.  The current release contains the weak-selected
MINT layer-9 score as primary and the best prespecified equal-logit ensemble as
a comparison menu.  It verifies both complete 447,731-row score files and
their weak-only locks before writing the bundle.

No direct-retention value or retention-derived binder label is accepted or
read.  The universe is exhaustive only within the provider's 15-residue LibA
design alphabet, not all 20 amino acids at each mutable position.
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
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SCHEMA_VERSION = "liba-generic-weak-selector-bundle-v1"
GENERIC_LOCK_SCHEMA = "affibody-weak-oof-score-lock-v1"
EXPECTED_ROWS = 447_731
EXPECTED_DOUBLE_COLD = 330_880
EXPECTED_PEPTIDE_COLD_ONLY = 116_851
EXPECTED_EVIDENCE = 7_786
EXPECTED_TRAINING_ROWS = 22_542
EXPECTED_TRAINING_POSITIVE = 11_320
EXPECTED_TRAINING_NEGATIVE = 11_222
EXPECTED_OOF_ROWS = 7_515
EXPECTED_OOF_POSITIVE = 3_759
EXPECTED_EVALUABLE_PEPTIDES = 189
EXPECTED_PEPTIDES = ("AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL")
EXPECTED_ZERO_CANDIDATE_TARGETS = ("DP",)
EXPECTED_TRAINING_MEMBERSHIP = "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e"
FORBIDDEN_SCORE_PARTS = (
    "retention",
    "binder_label",
    "binding_label",
    "target_binder",
    "ground_truth",
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
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("ascii")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(Path(path).is_file(), f"missing JSON file: {path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(Path(path).resolve(), start=Path(base).resolve())


def _record(path: Path, base: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": _relative(resolved, base) if base is not None else str(resolved),
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


def _forbidden_score_columns(columns: Iterable[object]) -> list[str]:
    return sorted(
        str(column)
        for column in columns
        if any(part in str(column).lower() for part in FORBIDDEN_SCORE_PARTS)
    )


def _find_score_receipt(path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (path.parent / "manifest.json", path.with_suffix(path.suffix + ".manifest.json"))
    existing = [candidate for candidate in candidates if candidate.is_file()]
    _require(len(existing) == 1, f"expected exactly one completion manifest for {path}")
    manifest_path = existing[0]
    manifest = _read_json(manifest_path)
    score_hash = sha256_file(path)
    possible: list[Mapping[str, Any]] = []
    for key in ("output", "combined_candidate_scores"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            possible.append(value)
    outputs = manifest.get("outputs")
    if isinstance(outputs, Mapping):
        possible.extend(value for value in outputs.values() if isinstance(value, Mapping))
    _require(any(value.get("sha256") == score_hash for value in possible),
             f"score manifest does not attest {path}")
    return manifest_path, manifest


def validate_candidate_score(
    path: Path,
    model_id: str,
    threshold: float,
    expected_binding: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    _require(not _forbidden_score_columns(columns), f"{model_id} score table contains outcomes")
    required = {
        "model_id",
        "pair_uid",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "affibody_design_code",
        "model_score",
        "affibody_identity_seen_in_strict_training",
    }
    _require(required.issubset(columns), f"{model_id} score table lacks required columns")
    _require(parquet.metadata.num_rows == EXPECTED_ROWS, f"{model_id} candidate row count changed")
    frame = pd.read_parquet(path, columns=list(required))
    _require(len(frame) == EXPECTED_ROWS and frame["pair_uid"].astype(str).is_unique,
             f"{model_id} candidate membership changed")
    _require(frame["model_id"].astype(str).eq(model_id).all(), f"{model_id} file identifies another model")
    _require(tuple(sorted(frame["peptide_design_code"].astype(str).unique())) == EXPECTED_PEPTIDES,
             f"{model_id} candidate peptides changed")
    _require(
        frame.groupby("peptide_design_code")["peptide_9mer_sequence"].nunique().eq(1).all(),
        f"{model_id} target sequence mapping changed",
    )
    score = pd.to_numeric(frame["model_score"], errors="raise").to_numpy(dtype=np.float64)
    _require(bool(np.isfinite(score).all() and ((score >= 0.0) & (score <= 1.0)).all()),
             f"{model_id} contains invalid scores")
    seen = frame["affibody_identity_seen_in_strict_training"]
    _require(pd.api.types.is_bool_dtype(seen) and seen.notna().all(),
             f"{model_id} training-identity flag is invalid")
    double_cold = int((~seen).sum())
    peptide_cold_only = int(seen.sum())
    _require(double_cold == EXPECTED_DOUBLE_COLD, f"{model_id} double-cold count changed")
    _require(peptide_cold_only == EXPECTED_PEPTIDE_COLD_ONLY,
             f"{model_id} peptide-cold-only count changed")
    frame["passes_cutoff"] = score >= float(threshold)
    by_peptide = (
        frame.groupby("peptide_design_code", sort=True)["passes_cutoff"]
        .sum()
        .astype(int)
        .to_dict()
    )
    zero_candidate_targets = tuple(sorted(
        peptide for peptide, count in by_peptide.items() if int(count) == 0
    ))
    _require(
        zero_candidate_targets == EXPECTED_ZERO_CANDIDATE_TARGETS,
        (
            f"{model_id} unexpected zero-candidate targets: "
            f"{list(zero_candidate_targets)}; expected {list(EXPECTED_ZERO_CANDIDATE_TARGETS)}"
        ),
    )
    manifest_path, manifest = _find_score_receipt(path)
    _require(manifest.get("model_id") == model_id, f"{model_id} score manifest identifies another model")
    _require(int(manifest.get("rows", -1)) == EXPECTED_ROWS, f"{model_id} score manifest row count changed")
    _require(manifest.get("pair_uid_membership_sha256") == membership_sha256(frame["pair_uid"]),
             f"{model_id} score manifest membership hash changed")
    binding = manifest.get("deployment_binding")
    _require(isinstance(binding, Mapping), f"{model_id} score manifest lacks deployment binding")
    binding_keys = {"head_sha256", "config_sha256", "generic_lock_sha256", "weak_oof_sha256"}
    _require(binding_keys.issubset(binding), f"{model_id} deployment binding is incomplete")
    for key in binding_keys:
        _require(str(binding[key]) == str(expected_binding[key]),
                 f"{model_id} deployment {key} differs from the selector bundle")
    return frame, {
        "model_id": model_id,
        "score_file": _record(path),
        "score_manifest": _record(manifest_path),
        "score_manifest_schema_version": manifest.get("schema_version"),
        "deployment_binding": {key: str(binding[key]) for key in sorted(binding_keys)},
        "rows": EXPECTED_ROWS,
        "pair_uid_membership_sha256": membership_sha256(frame["pair_uid"]),
        "pairs_above_cutoff_by_peptide": by_peptide,
        "targets_without_candidates_above_locked_threshold": list(zero_candidate_targets),
        "strict_double_cold_pairs": double_cold,
        "peptide_cold_only_pairs": peptide_cold_only,
        "outcome_columns_read": [],
    }


def validate_generic_lock(
    path: Path,
    expected_candidate: str,
    expected_family: str,
    metrics: pd.DataFrame,
) -> tuple[dict[str, Any], pd.Series]:
    lock = _read_json(path)
    _require(lock.get("schema_version") == GENERIC_LOCK_SCHEMA, "generic lock schema changed")
    _require(lock.get("library") == "LibA", "generic lock is not LibA")
    _require(lock.get("retention_labels_read") is False, "generic lock does not forbid retention")
    _require(lock.get("candidate") == expected_candidate, "generic source candidate changed")
    _require(lock.get("candidate_family") == expected_family, "generic candidate family changed")
    cutoff = lock.get("cutoff", {})
    _require(cutoff.get("decision_rule") == "score >= threshold", "generic cutoff rule changed")
    _require(cutoff.get("selection_objective") == "maximum F1 on matched weak-label OOF rows",
             "generic threshold objective changed")
    _require(cutoff.get("pad_below_threshold") is False, "generic cutoff pads candidates")
    threshold = float(cutoff.get("score_threshold", math.nan))
    _require(math.isfinite(threshold), "generic threshold is invalid")
    row = metrics.loc[metrics["candidate"].astype(str).eq(expected_candidate)]
    _require(len(row) == 1, "generic candidate metric is missing or duplicated")
    row = row.iloc[0]
    _require(abs(float(row["threshold"]) - threshold) <= 1e-12,
             "generic threshold differs from candidate metrics")
    provenance = lock.get("selection_provenance", {}).get("candidate_metrics", {})
    _require(abs(float(provenance.get("within_peptide_ap", math.nan)) - float(row["within_peptide_ap"])) <= 1e-12,
             "generic lock weak AP differs from candidate metrics")
    _require(int(provenance.get("within_peptide_evaluable", -1)) == int(row["within_peptide_evaluable"]),
             "generic lock weak AP group count differs from candidate metrics")
    _require(int(row["within_peptide_evaluable"]) == EXPECTED_EVALUABLE_PEPTIDES,
             "weak evaluable peptide count changed")
    return lock, row


def prospective_menu_agreement(
    primary: pd.DataFrame,
    comparator: pd.DataFrame,
    primary_threshold: float,
    comparator_threshold: float,
    maximum: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare two outcome-blind locked menus on their shared universe.

    Each menu applies its own weak-OOF threshold, then takes up to ``maximum``
    scores per peptide.  No retention value or binder outcome is involved.
    """

    def locked_menu(frame: pd.DataFrame, threshold: float, model_id: str) -> pd.DataFrame:
        blocks = []
        for peptide, block in frame.groupby("peptide_design_code", sort=True):
            eligible = block.loc[
                pd.to_numeric(block["model_score"], errors="raise").ge(float(threshold))
            ].sort_values(
                ["model_score", "pair_uid"], ascending=[False, True], kind="mergesort"
            ).head(int(maximum)).copy()
            if len(eligible):
                _require(
                    eligible["affibody_design_code"].astype(str).is_unique,
                    f"{model_id} repeats an Affibody code within {peptide}",
                )
            eligible["model_id"] = model_id
            eligible["locked_menu_rank"] = np.arange(1, len(eligible) + 1, dtype=np.int64)
            blocks.append(eligible)
        return pd.concat(blocks, ignore_index=True)

    primary_id = "mint_l9"
    comparator_id = "equal_logit_additive_mint_l9"
    menus = {
        primary_id: locked_menu(primary, primary_threshold, primary_id),
        comparator_id: locked_menu(comparator, comparator_threshold, comparator_id),
    }
    rows = []
    for peptide in EXPECTED_PEPTIDES:
        left = menus[primary_id].loc[
            menus[primary_id]["peptide_design_code"].astype(str).eq(peptide)
        ].sort_values("locked_menu_rank", kind="stable")
        right = menus[comparator_id].loc[
            menus[comparator_id]["peptide_design_code"].astype(str).eq(peptide)
        ].sort_values("locked_menu_rank", kind="stable")
        left_codes = left["affibody_design_code"].astype(str).tolist()
        right_codes = right["affibody_design_code"].astype(str).tolist()
        intersection = set(left_codes) & set(right_codes)
        union = set(left_codes) | set(right_codes)
        both_available = bool(left_codes and right_codes)
        rows.append({
            "peptide_design_code": peptide,
            "mint_l9_selected": len(left_codes),
            "equal_logit_selected": len(right_codes),
            "overlap_count": len(intersection),
            "union_count": len(union),
            "jaccard_overlap": len(intersection) / len(union) if union else np.nan,
            "both_models_have_candidates": both_available,
            "same_top1": bool(both_available and left_codes[0] == right_codes[0]),
            "mint_l9_top1": left_codes[0] if left_codes else "",
            "equal_logit_top1": right_codes[0] if right_codes else "",
            "identical_ranked_menu": bool(both_available and left_codes == right_codes),
        })
    per_target = pd.DataFrame(rows)

    recurrence_blocks = []
    for model_id, menu in menus.items():
        recurrence = (
            menu.groupby("affibody_design_code", as_index=False)
            .agg(
                selected_for_n_targets=("peptide_design_code", "nunique"),
                best_locked_menu_rank=("locked_menu_rank", "min"),
            )
        )
        top1 = (
            menu.loc[menu["locked_menu_rank"].eq(1)]
            .groupby("affibody_design_code")["peptide_design_code"]
            .nunique()
        )
        recurrence["top1_for_n_targets"] = (
            recurrence["affibody_design_code"].map(top1).fillna(0).astype(int)
        )
        recurrence.insert(0, "model_id", model_id)
        recurrence_blocks.append(recurrence)
    recurrence = pd.concat(recurrence_blocks, ignore_index=True).sort_values(
        ["model_id", "selected_for_n_targets", "top1_for_n_targets", "affibody_design_code"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    model_summary: dict[str, Any] = {}
    for model_id, menu in menus.items():
        model_recurrence = recurrence.loc[recurrence["model_id"].eq(model_id)]
        maximum_recurrence = int(model_recurrence["selected_for_n_targets"].max())
        model_summary[model_id] = {
            "locked_threshold": float(
                primary_threshold if model_id == primary_id else comparator_threshold
            ),
            "menu_pairs": int(len(menu)),
            "targets_with_candidates": int(
                menu["peptide_design_code"].astype(str).nunique()
            ),
            "targets_without_candidates_above_locked_threshold": sorted(
                set(EXPECTED_PEPTIDES)
                - set(menu["peptide_design_code"].astype(str).unique())
            ),
            "distinct_affibody_codes_across_orderable_target_menus": int(
                menu["affibody_design_code"].astype(str).nunique()
            ),
            "distinct_top1_affibody_codes": int(
                menu.loc[menu["locked_menu_rank"].eq(1), "affibody_design_code"]
                .astype(str).nunique()
            ),
            "top1_codes": sorted(
                menu.loc[menu["locked_menu_rank"].eq(1), "affibody_design_code"]
                .astype(str).unique().tolist()
            ),
            "maximum_targets_sharing_one_menu_code": maximum_recurrence,
            "most_recurrent_menu_codes": sorted(
                model_recurrence.loc[
                    model_recurrence["selected_for_n_targets"].eq(maximum_recurrence),
                    "affibody_design_code",
                ].astype(str).tolist()
            ),
        }
    summary = {
        "definition": (
            "For each model and peptide, apply that model's locked weak-OOF threshold and "
            "take up to ten highest scores; Hamming distance is not constrained."
        ),
        "outcome_columns_read": [],
        "maximum_candidates_per_peptide": int(maximum),
        "targets": len(EXPECTED_PEPTIDES),
        "targets_with_candidates_for_both_models": int(
            per_target["both_models_have_candidates"].sum()
        ),
        "targets_with_same_top1": int(per_target["same_top1"].sum()),
        "targets_with_identical_ranked_menu": int(per_target["identical_ranked_menu"].sum()),
        "mean_menu_overlap_count_across_all_nine_targets": float(
            per_target["overlap_count"].mean()
        ),
        "mean_menu_overlap_count_across_targets_with_both_models": float(
            per_target.loc[
                per_target["both_models_have_candidates"], "overlap_count"
            ].mean()
        ),
        "mean_jaccard_overlap_across_targets_with_both_models": float(
            per_target.loc[
                per_target["both_models_have_candidates"], "jaccard_overlap"
            ].mean()
        ),
        "models": model_summary,
    }
    return per_target, recurrence, summary


def _model_record(
    *,
    deployed_id: str,
    display_name: str,
    lock_path: Path,
    lock: Mapping[str, Any],
    metric: pd.Series,
    metrics_path: Path,
    weak_oof_path: Path,
    head_path: Path,
    config_path: Path,
    training_data_path: Path,
    bundle_parent: Path,
) -> dict[str, Any]:
    source_candidate = str(lock["candidate"])
    return {
        "model_id": deployed_id,
        "display_name": display_name,
        "scientifically_eligible": True,
        "lock_id": str(lock["lock_id"]),
        "score_threshold": float(lock["cutoff"]["score_threshold"]),
        "threshold_metric": "maximum_F1_on_matched_weak_label_OOF",
        "threshold_uses_retention": False,
        "threshold_provenance": (
            "Cutoff maximizes F1 on 7,515 matched double-cold weak-label OOF rows; "
            "ties prefer higher precision, fewer calls, then a higher cutoff."
        ),
        "max_candidates_per_peptide": 10,
        "minimum_code_hamming_distance": 0,
        "weak_oof_sha256": sha256_file(weak_oof_path),
        "head_sha256": sha256_file(head_path),
        "config_sha256": sha256_file(config_path),
        "training_rows": EXPECTED_TRAINING_ROWS,
        "training_data_sha256": sha256_file(training_data_path),
        "weak_oof_within_peptide_average_precision": float(metric["within_peptide_ap"]),
        "weak_oof_evaluable_peptides_for_within_peptide_ap": int(metric["within_peptide_evaluable"]),
        "weak_metric_provenance": {
            "generic_lock_path": _relative(lock_path, bundle_parent),
            "generic_lock_sha256": sha256_file(lock_path),
            "candidate_metrics_path": _relative(metrics_path, bundle_parent),
            "candidate_metrics_sha256": sha256_file(metrics_path),
            "candidate": source_candidate,
        },
        "model_source_provenance": {
            "head_path": _relative(head_path, bundle_parent),
            "config_path": _relative(config_path, bundle_parent),
            "weak_oof_path": _relative(weak_oof_path, bundle_parent),
            "training_data_path": _relative(training_data_path, bundle_parent),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = _private_new_directory(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    selector_dir = Path(args.selector_dir).resolve()
    common_dir = Path(args.common_sequence_dir).resolve()
    universe_dir = Path(args.universe_dir).resolve()
    _require(selector_dir.is_dir() and common_dir.is_dir() and universe_dir.is_dir(),
             "selector/common/universe input directory is missing")
    selector_manifest_path = selector_dir / "manifest.json"
    selector_manifest = _read_json(selector_manifest_path)
    _require(selector_manifest.get("schema_version") == "affibody-weak-oof-ensemble-v1",
             "generic selector manifest schema changed")
    _require(selector_manifest.get("library") == "LibA", "generic selector manifest is not LibA")
    _require(selector_manifest.get("retention_labels_read") is False,
             "generic selector manifest does not forbid retention")
    metrics_path = selector_dir / "candidate_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    required_metric = {"candidate", "family", "within_peptide_ap", "within_peptide_evaluable", "threshold"}
    _require(required_metric.issubset(metrics.columns), "generic candidate-metric schema changed")

    primary_lock_path = selector_dir / "selected_score.lock.json"
    ensemble_lock_path = selector_dir / "best_equal_logit_ensemble.lock.json"
    primary_candidate = "mean_logit__frozen_mint_layer9"
    ensemble_candidate = "mean_logit__additive_6site__frozen_mint_layer9"
    primary_lock, primary_metric = validate_generic_lock(
        primary_lock_path, primary_candidate, "single", metrics
    )
    ensemble_lock, ensemble_metric = validate_generic_lock(
        ensemble_lock_path, ensemble_candidate, "equal_mean_logit", metrics
    )
    _require(float(primary_metric["within_peptide_ap"]) > float(ensemble_metric["within_peptide_ap"]),
             "equal-logit ensemble unexpectedly became the weak-selected primary")
    selection = _read_json(selector_dir / "locked_weak_selection.json")
    _require(selection.get("best_candidate", {}).get("candidate") == primary_candidate,
             "generic selected primary changed")
    _require(selection.get("best_equal_logit_ensemble", {}).get("candidate") == ensemble_candidate,
             "generic selected ensemble changed")

    weak_oof_path = common_dir / "weak_oof_aligned.csv.gz"
    training_data_path = common_dir / "weak_row_metadata.csv.gz"
    common_summary = _read_json(common_dir / "summary.json")
    _require(common_summary.get("retention_labels_read") is False, "common model build used retention")
    weak = common_summary.get("weak_training", {})
    _require(
        (int(weak.get("rows", -1)), int(weak.get("positive", -1)), int(weak.get("negative", -1)))
        == (EXPECTED_TRAINING_ROWS, EXPECTED_TRAINING_POSITIVE, EXPECTED_TRAINING_NEGATIVE),
        "common weak-training counts changed",
    )
    _require(weak.get("membership_sha256") == EXPECTED_TRAINING_MEMBERSHIP,
             "common weak-training membership changed")
    validation = common_summary.get("validation", {})
    _require((int(validation.get("oof_unique_rows", -1)), int(validation.get("oof_positive", -1)))
             == (EXPECTED_OOF_ROWS, EXPECTED_OOF_POSITIVE), "common OOF counts changed")

    universe_manifest = _read_json(universe_dir / "manifest.json")
    _require(int(universe_manifest.get("outputs", {}).get("candidate_rows", -1)) == EXPECTED_ROWS,
             "candidate universe row count changed")
    _require(int(universe_manifest.get("outputs", {}).get("pooled_positive_unmeasured_control_rows", -1))
             == EXPECTED_EVIDENCE, "evidence-positive control count changed")
    target_sequences = universe_manifest.get("scope", {}).get("target_peptide_full_sequences", {})
    _require(tuple(sorted(target_sequences)) == EXPECTED_PEPTIDES,
             "candidate-universe target mapping changed")
    _require(universe_manifest.get("scope", {}).get("affibody_design_alphabet_each_position")
             == "ADEFHIKLNPQSTVY", "LibA 15-residue design alphabet changed")

    common_config = REPO_ROOT / "downstream/AffibodyMHC/configs/liba_common_oof_sequence_v1.json"
    ensemble_config = REPO_ROOT / "downstream/AffibodyMHC/configs/liba_weak_oof_ensemble_v1.json"
    primary_head = common_dir / "mint_layer9_deployment_head.npz"
    primary_binding = {
        "head_sha256": sha256_file(primary_head),
        "config_sha256": sha256_file(common_config),
        "generic_lock_sha256": sha256_file(primary_lock_path),
        "weak_oof_sha256": sha256_file(weak_oof_path),
    }
    ensemble_binding = {
        "head_sha256": sha256_file(ensemble_lock_path),
        "config_sha256": sha256_file(ensemble_config),
        "generic_lock_sha256": sha256_file(ensemble_lock_path),
        "weak_oof_sha256": sha256_file(weak_oof_path),
    }
    primary_scores, primary_score_audit = validate_candidate_score(
        Path(args.primary_scores).resolve(),
        "mint_l9",
        float(primary_lock["cutoff"]["score_threshold"]),
        primary_binding,
    )
    ensemble_scores, ensemble_score_audit = validate_candidate_score(
        Path(args.ensemble_scores).resolve(),
        "equal_logit_additive_mint_l9",
        float(ensemble_lock["cutoff"]["score_threshold"]),
        ensemble_binding,
    )
    primary_ids = primary_scores[["pair_uid"]].sort_values("pair_uid", kind="stable").reset_index(drop=True)
    ensemble_ids = ensemble_scores[["pair_uid"]].sort_values("pair_uid", kind="stable").reset_index(drop=True)
    _require(primary_ids.equals(ensemble_ids), "primary and ensemble candidate memberships differ")
    agreement_by_target, recurrence, agreement_summary = prospective_menu_agreement(
        primary_scores,
        ensemble_scores,
        float(primary_lock["cutoff"]["score_threshold"]),
        float(ensemble_lock["cutoff"]["score_threshold"]),
        maximum=10,
    )

    # The selector bundle is written inside a staging directory.  Relative
    # provenance paths are computed from the final directory, which has the
    # same parent/leaf layout after atomic rename.
    staging = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    _require(not staging.exists(), f"staging directory exists: {staging}")
    staging.mkdir(mode=0o700)
    agreement_path = staging / "prospective_menu_agreement_by_target.csv"
    recurrence_path = staging / "prospective_affibody_recurrence.csv"
    agreement_by_target.to_csv(agreement_path, index=False, float_format="%.17g")
    recurrence.to_csv(recurrence_path, index=False)
    os.chmod(agreement_path, 0o600)
    os.chmod(recurrence_path, 0o600)
    bundle_parent = output_dir
    primary_model = _model_record(
        deployed_id="mint_l9",
        display_name="Weak-label-selected frozen MINT layer 9",
        lock_path=primary_lock_path,
        lock=primary_lock,
        metric=primary_metric,
        metrics_path=metrics_path,
        weak_oof_path=weak_oof_path,
        head_path=primary_head,
        config_path=common_config,
        training_data_path=training_data_path,
        bundle_parent=bundle_parent,
    )
    ensemble_model = _model_record(
        deployed_id="equal_logit_additive_mint_l9",
        display_name="Equal-logit additive plus MINT layer 9",
        lock_path=ensemble_lock_path,
        lock=ensemble_lock,
        metric=ensemble_metric,
        metrics_path=metrics_path,
        weak_oof_path=weak_oof_path,
        # The immutable generic formula lock is the ensemble's deployable head.
        head_path=ensemble_lock_path,
        config_path=ensemble_config,
        training_data_path=training_data_path,
        bundle_parent=bundle_parent,
    )
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library": "LibA",
        "labels": "weak_selection_only",
        "primary_model_id": "mint_l9",
        "target_sequences": target_sequences,
        "models": [primary_model, ensemble_model],
        "training_description": (
            "22,542 strict weak-label pairs: 11,320 positives and 11,222 negatives; "
            "all direct evaluation peptide and Affibody identities are excluded."
        ),
        "weak_label_definition": (
            "Positives are selected after summing exact-pair R009 and R010 counts and taking "
            "the tie-inclusive top 2%. Negatives have at least three reads in R001 and never "
            "appear in any positive-selection round R002-R014. The strict split then excludes "
            "evaluation peptide and Affibody identities from training."
        ),
        "split_rule": (
            "Three diagonal double-cold folds; neither validation peptide identities nor "
            "validation Affibody identities occur in the corresponding training fold."
        ),
        "evidence_max_controls_per_peptide": 10,
        "unavailable_structure_families": [
            {
                "family": "RDE-PPI",
                "reason": (
                    "The only local experimental complex is the LibB MW+NNYYF crystal; "
                    "using that geometry as LibA ground truth would be an invalid transfer."
                ),
            },
            {
                "family": "StaB-ddG",
                "reason": (
                    "The only local experimental complex is the LibB MW+NNYYF crystal; "
                    "no validated LibA geometry exists for this structure-derived encoder."
                ),
            },
        ],
        "candidate_scope": {
            "selection_missed_pairs": EXPECTED_ROWS,
            "directly_measured_pairs_excluded": 108,
            "pooled_positive_unmeasured_controls_excluded_and_kept_separate": EXPECTED_EVIDENCE,
            "designed_residue_alphabet_each_position": "ADEFHIKLNPQSTVY",
            "designed_residue_alphabet_size": 15,
            "affibody_code_positions": 4,
            "scope_warning": (
                "Exhaustive within the provider's 15-amino-acid-per-position LibA design "
                "alphabet only; this is not the full 20^4 Affibody space."
            ),
        },
        "candidate_score_artifacts": {
            "mint_l9": primary_score_audit,
            "equal_logit_additive_mint_l9": ensemble_score_audit,
        },
        "targets_without_candidates_above_locked_threshold": {
            "mint_l9": primary_score_audit[
                "targets_without_candidates_above_locked_threshold"
            ],
            "equal_logit_additive_mint_l9": ensemble_score_audit[
                "targets_without_candidates_above_locked_threshold"
            ],
        },
        "prospective_model_agreement": agreement_summary,
        "prospective_diagnostic_outputs": {
            "menu_agreement_by_target": _record(agreement_path, staging),
            "affibody_recurrence": _record(recurrence_path, staging),
        },
        "source_artifacts": {
            "generic_selector_manifest": _record(selector_manifest_path, bundle_parent),
            "candidate_universe_manifest": _record(universe_dir / "manifest.json", bundle_parent),
            "common_sequence_manifest": _record(common_dir / "manifest.json", bundle_parent),
        },
    }
    bundle_path = staging / "selector_bundle.json"
    _write_json(bundle_path, bundle)
    manifest = {
        "schema_version": "liba-generic-weak-selector-bundle-manifest-v1",
        "created_utc": bundle["created_utc"],
        "selector_bundle": _record(bundle_path, staging),
        "primary_model_id": "mint_l9",
        "eligible_model_ids": ["mint_l9", "equal_logit_additive_mint_l9"],
        "candidate_rows_per_model": EXPECTED_ROWS,
        "targets_without_candidates_above_locked_threshold": {
            "mint_l9": primary_score_audit[
                "targets_without_candidates_above_locked_threshold"
            ],
            "equal_logit_additive_mint_l9": ensemble_score_audit[
                "targets_without_candidates_above_locked_threshold"
            ],
        },
        "pair_uid_membership_sha256": primary_score_audit["pair_uid_membership_sha256"],
        "prospective_model_agreement": agreement_summary,
        "prospective_diagnostic_outputs": {
            "menu_agreement_by_target": _record(agreement_path, staging),
            "affibody_recurrence": _record(recurrence_path, staging),
        },
        "retention_labels_read": False,
        "code": _record(Path(__file__).resolve()),
    }
    _write_json(staging / "manifest.json", manifest)
    os.rename(staging, output_dir)
    print(json.dumps({"output": str(output_dir), "primary": "mint_l9", "rows": EXPECTED_ROWS}))
    return bundle


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--common-sequence-dir", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--primary-scores", type=Path, required=True)
    parser.add_argument("--ensemble-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
