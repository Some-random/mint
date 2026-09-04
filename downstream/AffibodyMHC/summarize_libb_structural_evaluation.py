#!/usr/bin/env python
"""Build deterministic report tables for the final LibB structure comparison.

Run this command only after every structural score has been frozen and the
sealed evaluator has joined those scores to the corrected 120-cell retention
panel.  This script does not fit, tune, or select a model from retention.  The
RDE-PPI-derived and StaB-ddG-derived representatives come exclusively from the
supplied weak-label selection lock.  The sequence/context rows shown in the
main table are fixed below by model name, before their metric values are read.

The command verifies the selection lock, source manifests, normalized
prediction vectors, corrected panel membership, and saved evaluation tables.
It then writes machine-readable main/secondary/appendix tables, per-peptide
diagnostics, a focused description of the unusually broad FALTA Affibody, and
a compact Markdown rendering for the public structure report.

Example (only after unsealing is authorized)::

    python downstream/AffibodyMHC/summarize_libb_structural_evaluation.py \
      --selection-lock private_data/experiments/libb_structure_provider_revision_120_weak_selection_v1/selection.json \
      --structural-evaluation private_data/experiments/libb_structure_provider_revision_120_sealed_evaluation_v1 \
      --esmfold2-evaluation private_data/experiments/esmfold2_libb_provider_revision_120_metrics_v1 \
      --mint-evaluation private_data/experiments/libb_revision_mint_metrics_120_v1 \
      --retention-panel private_data/derived/retention_panel_provider_revision_2026-09-03_v2/libb_evaluation_panel.csv \
      --output-dir private_data/experiments/libb_structure_provider_revision_120_summary_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.recompute_libb_metrics_120 import (  # noqa: E402
    load_corrected_retention_panel,
)
from downstream.AffibodyMHC.wetlab_metrics import (  # noqa: E402
    TIE_POLICY,
    evaluate_wetlab_predictions,
)


SCHEMA_VERSION = "libb-structural-evaluation-summary-v1"
LOCK_SCHEMA_VERSION = "libb-structural-weak-selection-v1"
LOCK_SELECTION_METRIC = "pooled_weak_validation_log_loss"
SEALED_EVALUATION_SCHEMA = "libb-sealed-readout-evaluation-v1"
CORRECTED_EVALUATION_SCHEMA = "libb-corrected-120-metric-recompute-v1"

EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_BINDERS = 61
EXPECTED_NONBINDERS = 59
EXPECTED_STRUCTURAL_SEEDS = 5
EXPECTED_FALTA_BINDER_PEPTIDES = 11
EXPECTED_FALTA_OPTIMAL_PEPTIDES = 10
EXPECTED_FALTA_NONOPTIMAL_PEPTIDES = ("EL", "MW")
FALTA_CODE = "FALTA"
MATCHED_CONTROL = "nonlinear_7site_control"
PRECISION_AT_3_DENOMINATOR = EXPECTED_PEPTIDES * 3
NUMERIC_TOLERANCE = 2e-10

PRIMARY_METRICS = (
    "within_peptide_spearman_mean",
    "peptide_macro_precision_at_3",
    "peptide_macro_best_retention_at_3",
    "peptide_macro_regret_at_3",
)
SECONDARY_METRICS = (
    "global_auroc",
    "global_average_precision",
)
ALL_SUMMARY_METRICS = PRIMARY_METRICS + SECONDARY_METRICS

# These context rows are declared in code, not selected from retention.  Other
# MINT and ESMFold2 arms remain visible in the all-arm appendix.
FIXED_CONTEXT_MAIN_MODELS = (
    {
        "bundle": "mint",
        "model": "frozen_mint_layer5_archived119_plus_replay_target",
        "display_name": "Frozen MINT layer 5 + classifier",
        "selection_basis": "fixed context row; weak-selected layer from the earlier sequence study",
    },
    {
        "bundle": "mint",
        "model": "lora_one_epoch_saved_checkpoint_exact120",
        "display_name": "MINT after one LoRA epoch",
        "selection_basis": "fixed context row from the earlier sequence study",
    },
    {
        "bundle": "esmfold2",
        "model": "single_inputs_only",
        "display_name": "ESMFold2 pre-folding residue features--not structure",
        "selection_basis": "fixed ESMFold2 weak-validation representative",
    },
    {
        "bundle": "esmfold2",
        "model": "pair_state_only",
        "display_name": "ESMFold2 folding-derived pair state",
        "selection_basis": "prespecified folding-derived context row",
    },
)


@dataclass(frozen=True)
class EvaluationBundle:
    name: str
    root: Path
    manifest: Mapping[str, Any]
    metrics: pd.DataFrame
    per_peptide: pd.DataFrame
    ranked: pd.DataFrame
    predictions: pd.DataFrame
    panel_source: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_record_path(record: Mapping[str, Any]) -> Path:
    raw = Path(str(record.get("path", "")))
    _require(str(raw) not in ("", "."), "provenance record lacks a path")
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _validate_provenance_record(
    record: Mapping[str, Any], *, expected_path: Path | None = None
) -> Path:
    _require(isinstance(record, Mapping), "invalid provenance record")
    path = _resolve_record_path(record).resolve()
    _require(path.is_file(), f"provenance source does not exist: {path}")
    if expected_path is not None:
        _require(path == Path(expected_path).resolve(), f"provenance path changed: {path}")
    checksum = str(record.get("sha256", ""))
    _require(len(checksum) == 64, f"provenance source lacks SHA-256: {path}")
    _require(sha256_file(path) == checksum, f"provenance SHA-256 mismatch: {path}")
    if "bytes" in record:
        _require(int(record["bytes"]) == path.stat().st_size, f"provenance size mismatch: {path}")
    return path


def _bundle_layout(schema: str) -> dict[str, str]:
    if schema == SEALED_EVALUATION_SCHEMA:
        return {
            "metrics": "matched_metrics_by_seed.csv",
            "seed_summary": "matched_metrics_seed_summary.csv",
            "per_peptide": "per_peptide_metrics.csv",
            "ranked": "ranked_per_pair.csv",
        }
    if schema == CORRECTED_EVALUATION_SCHEMA:
        return {
            "metrics": "metrics_by_model_seed.csv",
            "seed_summary": "metrics_seed_summary.csv",
            "per_peptide": "per_peptide_metrics.csv",
            "ranked": "ranked_per_pair.csv",
        }
    raise ValueError(f"unsupported evaluation schema {schema!r}")


def _load_manifest_output(root: Path, manifest: Mapping[str, Any], filename: str) -> pd.DataFrame:
    outputs = manifest.get("outputs")
    _require(isinstance(outputs, Mapping), f"evaluation manifest lacks outputs: {root}")
    _require(filename in outputs, f"evaluation manifest lacks {filename}: {root}")
    record = outputs[filename]
    _require(isinstance(record, Mapping), f"invalid output record for {filename}")
    path = (root / filename).resolve()
    recorded_path = _validate_provenance_record(record)
    _require(recorded_path == path, f"manifest output path does not match bundle root: {filename}")
    frame = pd.read_csv(path, dtype={"model": str, "seed": str}, keep_default_na=False)
    _require(int(record.get("rows", -1)) == len(frame), f"manifest row count changed: {path}")
    return frame


def _prediction_source_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schema = str(manifest.get("schema_version", ""))
    sources = manifest.get("sources")
    _require(isinstance(sources, Mapping), "evaluation manifest lacks sources")
    key = "blinded_predictions" if schema == SEALED_EVALUATION_SCHEMA else "normalized_predictions"
    records = sources.get(key)
    _require(isinstance(records, list) and records, f"evaluation manifest lacks {key}")
    _require(all(isinstance(item, Mapping) for item in records), f"invalid {key} provenance")
    return list(records)


def _panel_source_record(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = str(manifest.get("schema_version", ""))
    sources = manifest.get("sources")
    _require(isinstance(sources, Mapping), "evaluation manifest lacks sources")
    key = "retention_audit" if schema == SEALED_EVALUATION_SCHEMA else "corrected_retention_panel"
    record = sources.get(key)
    _require(isinstance(record, Mapping), f"evaluation manifest lacks {key}")
    return record


def _load_manifest_predictions(manifest: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    exact_columns = ["eval_row_id", "model", "seed", "score"]
    for record in _prediction_source_records(manifest):
        path = _validate_provenance_record(record)
        frame = pd.read_csv(
            path,
            dtype={"eval_row_id": str, "model": str, "seed": str},
            keep_default_na=False,
        )
        _require(list(frame.columns) == exact_columns, f"prediction schema changed: {path}")
        frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
        _require(bool(np.isfinite(frame["score"]).all()), f"non-finite prediction score: {path}")
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    duplicate = output.duplicated(["model", "seed", "eval_row_id"], keep=False)
    _require(not bool(duplicate.any()), "prediction sources contain duplicate model/seed/ID rows")
    return output


def load_evaluation_bundle(name: str, root: Path, expected_schema: str) -> EvaluationBundle:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"evaluation bundle is incomplete: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(isinstance(manifest, Mapping), f"invalid evaluation manifest: {manifest_path}")
    schema = str(manifest.get("schema_version", ""))
    _require(schema == expected_schema, f"{name} evaluation schema is {schema!r}, not {expected_schema!r}")
    layout = _bundle_layout(schema)
    metrics = _load_manifest_output(root, manifest, layout["metrics"])
    # The saved summary is provenance-validated even though a common summary is
    # recomputed below to normalize the two evaluator naming conventions.
    _load_manifest_output(root, manifest, layout["seed_summary"])
    per_peptide = _load_manifest_output(root, manifest, layout["per_peptide"])
    ranked = _load_manifest_output(root, manifest, layout["ranked"])
    predictions = _load_manifest_predictions(manifest)

    required_metrics = {"model", "seed", *ALL_SUMMARY_METRICS}
    _require(not required_metrics.difference(metrics.columns), f"{name} metrics are incomplete")
    required_peptide = {
        "model",
        "seed",
        "peptide_design_code",
        "within_peptide_spearman",
        "precision_at_3",
        "best_retention_at_3",
        "regret_at_3",
        "top1_affibody",
    }
    _require(not required_peptide.difference(per_peptide.columns), f"{name} per-peptide table is incomplete")
    required_ranked = {
        "eval_row_id",
        "model",
        "seed",
        "score",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
        "within_peptide_rank",
    }
    _require(not required_ranked.difference(ranked.columns), f"{name} ranked table is incomplete")

    for frame in (metrics, per_peptide, ranked, predictions):
        frame["model"] = frame["model"].astype(str)
        frame["seed"] = frame["seed"].astype(str)
    prediction_groups = set(map(tuple, predictions[["model", "seed"]].drop_duplicates().to_numpy()))
    metric_groups = set(map(tuple, metrics[["model", "seed"]].drop_duplicates().to_numpy()))
    ranked_groups = set(map(tuple, ranked[["model", "seed"]].drop_duplicates().to_numpy()))
    peptide_groups = set(map(tuple, per_peptide[["model", "seed"]].drop_duplicates().to_numpy()))
    _require(
        prediction_groups == metric_groups == ranked_groups == peptide_groups,
        f"{name} prediction and evaluation model/seed groups disagree",
    )
    return EvaluationBundle(
        name=name,
        root=root,
        manifest=manifest,
        metrics=metrics,
        per_peptide=per_peptide,
        ranked=ranked,
        predictions=predictions,
        panel_source=_panel_source_record(manifest),
    )


def validate_selection_lock(payload: Mapping[str, Any]) -> dict[str, str]:
    """Validate and return the already-locked family representatives."""

    _require(payload.get("schema_version") == LOCK_SCHEMA_VERSION, "weak selection lock schema changed")
    _require(payload.get("retention_labels_read") is False, "weak selection lock says retention was read")
    _require(payload.get("selection_metric") == LOCK_SELECTION_METRIC, "weak selection metric changed")
    _require(payload.get("selection_direction") == "minimize", "weak selection direction changed")
    selected = payload.get("selected_models_by_family")
    families = payload.get("families")
    _require(isinstance(selected, Mapping) and set(selected) == {"rde", "stab"}, "lock lacks RDE/StaB selections")
    _require(isinstance(families, Mapping) and set(families) == {"rde", "stab"}, "lock lacks RDE/StaB families")

    output: dict[str, str] = {}
    for key in ("rde", "stab"):
        family = families[key]
        _require(isinstance(family, Mapping), f"invalid {key} lock family")
        selected_name = str(selected[key])
        _require(selected_name == str(family.get("selected_model", "")), f"{key} selected names disagree")
        candidates = family.get("candidates")
        _require(isinstance(candidates, list) and candidates, f"{key} lock has no candidates")
        by_name = {
            str(item.get("model", "")): item
            for item in candidates
            if isinstance(item, Mapping) and bool(item.get("eligible"))
        }
        _require(selected_name in by_name, f"{key} selected model is not an eligible candidate")
        losses = {
            name: float(item[LOCK_SELECTION_METRIC]) for name, item in by_name.items()
        }
        _require(all(math.isfinite(value) for value in losses.values()), f"{key} weak losses are non-finite")
        expected_name = min(losses, key=lambda name: (losses[name], name))
        _require(selected_name == expected_name, f"{key} selected model is not the locked weak-loss minimum")
        reported = float(family.get("selected_pooled_weak_validation_log_loss", float("nan")))
        _require(
            math.isclose(reported, losses[selected_name], rel_tol=0.0, abs_tol=1e-14),
            f"{key} selected weak loss is internally inconsistent",
        )
        output[key] = selected_name
    return output


def _numeric_close(left: object, right: object, label: str) -> None:
    left_number = float("nan") if str(left).strip() == "" else float(left)
    right_number = float("nan") if str(right).strip() == "" else float(right)
    if math.isnan(left_number) and math.isnan(right_number):
        return
    _require(
        math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=NUMERIC_TOLERANCE),
        f"saved and recomputed {label} disagree: {left_number} versus {right_number}",
    )


def validate_bundle_against_panel(
    bundle: EvaluationBundle, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute every metric from frozen predictions and the supplied panel."""

    panel_hash = sha256_file(_resolve_record_path(bundle.panel_source))
    _require(panel_hash == str(bundle.panel_source.get("sha256", "")), f"{bundle.name} panel source hash changed")
    panel_ids = set(panel["eval_row_id"].astype(str))
    prediction_ids_by_group = bundle.predictions.groupby(["model", "seed"])["eval_row_id"].agg(set)
    _require(
        all(ids == panel_ids for ids in prediction_ids_by_group),
        f"{bundle.name} does not score the exact corrected 120 IDs",
    )

    prediction_scores = bundle.predictions.rename(columns={"score": "source_prediction_score"})
    ranked_scores = bundle.ranked[["model", "seed", "eval_row_id", "score"]].copy()
    joined_scores = ranked_scores.merge(
        prediction_scores,
        on=["model", "seed", "eval_row_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    _require(joined_scores["_merge"].eq("both").all(), f"{bundle.name} ranked rows do not match predictions")
    _require(
        bool(
            np.allclose(
                pd.to_numeric(joined_scores["score"]).to_numpy(float),
                pd.to_numeric(joined_scores["source_prediction_score"]).to_numpy(float),
                rtol=0.0,
                atol=1e-15,
            )
        ),
        f"{bundle.name} saved ranked scores differ from frozen predictions",
    )

    saved_metrics = bundle.metrics.set_index(["model", "seed"], verify_integrity=True)
    saved_peptide = bundle.per_peptide.set_index(
        ["model", "seed", "peptide_design_code"], verify_integrity=True
    )
    peptide_frames: list[pd.DataFrame] = []
    ranked_frames: list[pd.DataFrame] = []
    for (model, seed), predictions in bundle.predictions.groupby(["model", "seed"], sort=True):
        merged = predictions.merge(panel, on="eval_row_id", how="inner", validate="one_to_one")
        _require(len(merged) == EXPECTED_ROWS, f"{bundle.name} final join lost rows")
        summary, peptide, ranked = evaluate_wetlab_predictions(
            merged,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
            score_column="score",
            binder_column="target_binder",
            retention_column="target_retention",
            k_values=(1, 3),
        )
        saved = saved_metrics.loc[(str(model), str(seed))]
        for metric in ALL_SUMMARY_METRICS:
            _numeric_close(saved[metric], summary[metric], f"{bundle.name}/{model}/{seed}/{metric}")
        for _, row in peptide.iterrows():
            key = (str(model), str(seed), str(row["peptide_design_code"]))
            _require(key in saved_peptide.index, f"{bundle.name} lacks per-peptide row {key}")
            old = saved_peptide.loc[key]
            for metric in (
                "within_peptide_spearman",
                "precision_at_3",
                "best_retention_at_3",
                "regret_at_3",
            ):
                _numeric_close(old[metric], row[metric], f"{bundle.name}/{key}/{metric}")
            _require(str(old["top1_affibody"]) == str(row["top1_affibody"]), f"{bundle.name}/{key} top1 changed")
        peptide.insert(0, "seed", str(seed))
        peptide.insert(0, "model", str(model))
        peptide.insert(0, "bundle", bundle.name)
        peptide_frames.append(peptide)
        ranked.insert(0, "bundle", bundle.name)
        ranked_frames.append(ranked)

    recomputed_peptide = pd.concat(peptide_frames, ignore_index=True)
    recomputed_ranked = pd.concat(ranked_frames, ignore_index=True)
    return recomputed_peptide, recomputed_ranked


def validate_structural_models(
    structural: EvaluationBundle,
    lock: Mapping[str, Any],
    selected: Mapping[str, str],
) -> None:
    families = lock["families"]
    expected_models = {MATCHED_CONTROL}
    for key in ("rde", "stab"):
        family = families[key]
        expected_models.update(
            str(item["model"])
            for item in family["candidates"]
            if bool(item.get("eligible"))
        )
        expected_models.update(map(str, family.get("ineligible_sequence_controls", [])))
    observed_models = set(structural.metrics["model"].astype(str))
    _require(observed_models == expected_models, "sealed structural evaluation does not contain every locked arm")
    _require(set(selected.values()).issubset(observed_models), "locked representatives are absent from evaluation")
    for model, group in structural.metrics.groupby("model", sort=True):
        seeds = sorted(set(group["seed"].astype(str)), key=_seed_sort_key)
        _require(len(seeds) == EXPECTED_STRUCTURAL_SEEDS, f"{model} does not have five final seeds")


def _seed_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _model_key(bundle: str, model: str) -> str:
    return f"{bundle}::{model}"


def _attach_bundle(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "bundle", name)
    output.insert(1, "model_key", [_model_key(name, value) for value in output["model"].astype(str)])
    return output


def combine_metrics(bundles: Sequence[EvaluationBundle]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = _attach_bundle(bundle.metrics, bundle.name)
        score_stats = (
            bundle.predictions.groupby(["model", "seed"], sort=True)["score"]
            .agg(
                prediction_score_unique_values="nunique",
                prediction_score_min="min",
                prediction_score_max="max",
            )
            .reset_index()
        )
        score_stats["model"] = score_stats["model"].astype(str)
        score_stats["seed"] = score_stats["seed"].astype(str)
        score_stats["constant_prediction_score"] = score_stats[
            "prediction_score_unique_values"
        ].eq(1).astype(int)
        frame = frame.merge(
            score_stats,
            on=["model", "seed"],
            how="left",
            validate="one_to_one",
        )
        _require(frame["prediction_score_unique_values"].notna().all(), f"{bundle.name} lacks score statistics")
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    duplicate = output.duplicated(["model_key", "seed"], keep=False)
    _require(not bool(duplicate.any()), "combined evaluation has duplicate model/seed groups")
    for metric in ALL_SUMMARY_METRICS:
        output[metric] = pd.to_numeric(output[metric], errors="raise").astype(float)
        _require(not bool(np.isinf(output[metric]).any()), f"combined metric {metric} contains infinity")
    return output.sort_values(["bundle", "model", "seed"], key=lambda column: column.map(_seed_sort_key) if column.name == "seed" else column, kind="mergesort").reset_index(drop=True)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (bundle, model, model_key), group in metrics.groupby(
        ["bundle", "model", "model_key"], sort=True
    ):
        group = group.assign(_seed_order=group["seed"].map(_seed_sort_key)).sort_values("_seed_order")
        row: dict[str, Any] = {
            "bundle": str(bundle),
            "model": str(model),
            "model_key": str(model_key),
            "n_seeds": int(len(group)),
            "seeds": json.dumps(group["seed"].astype(str).tolist(), separators=(",", ":")),
        }
        constant = group.loc[group["constant_prediction_score"].eq(1), "seed"].astype(str).tolist()
        row["constant_prediction_score_fits"] = int(len(constant))
        row["constant_prediction_score_seeds"] = json.dumps(constant, separators=(",", ":"))
        row["stability_warning"] = (
            f"{len(constant)}/{len(group)} fits collapsed to a constant score"
            if constant
            else ""
        )
        for metric in ALL_SUMMARY_METRICS:
            values = group[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            _require(len(finite) > 0, f"{model} has no evaluable values for {metric}")
            row[f"{metric}_n_evaluable"] = int(len(finite))
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
            row[f"{metric}_individual"] = json.dumps(
                [
                    {
                        "seed": str(seed),
                        "value": float(value) if math.isfinite(float(value)) else None,
                    }
                    for seed, value in zip(group["seed"], values)
                ],
                separators=(",", ":"),
            )
        p3 = group["peptide_macro_precision_at_3"].to_numpy(float)
        counts = p3 * PRECISION_AT_3_DENOMINATOR
        rounded = np.rint(counts).astype(int)
        _require(bool(np.allclose(counts, rounded, rtol=0.0, atol=1e-8)), f"{model} P@3 is not a count out of 36")
        row["precision_at_3_binders_of_36_individual"] = json.dumps(
            [
                {"seed": str(seed), "binders": int(value), "slots": PRECISION_AT_3_DENOMINATOR}
                for seed, value in zip(group["seed"], rounded)
            ],
            separators=(",", ":"),
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["bundle", "model"], kind="mergesort").reset_index(drop=True)


def build_main_model_specs(selected: Mapping[str, str]) -> list[dict[str, str]]:
    specs = [
        {
            "bundle": "structural",
            "model": MATCHED_CONTROL,
            "display_name": "Matched nonlinear seven-position sequence control",
            "selection_basis": "prespecified capacity-matched sequence control",
        },
        *[dict(item) for item in FIXED_CONTEXT_MAIN_MODELS],
        {
            "bundle": "structural",
            "model": str(selected["rde"]),
            "display_name": f"RDE-PPI-derived weak-label representative ({selected['rde']})",
            "selection_basis": "weak-label selection lock; retention not read",
        },
        {
            "bundle": "structural",
            "model": str(selected["stab"]),
            "display_name": f"StaB-ddG-derived weak-label representative ({selected['stab']})",
            "selection_basis": "weak-label selection lock; retention not read",
        },
    ]
    for index, item in enumerate(specs):
        item["model_key"] = _model_key(item["bundle"], item["model"])
        item["main_order"] = index
    return specs


def select_main_summary(
    all_summary: pd.DataFrame, specs: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    by_key = all_summary.set_index("model_key", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        key = str(spec["model_key"])
        _require(key in by_key.index, f"fixed main-table model is absent: {key}")
        row = by_key.loc[key].to_dict()
        for metric in ALL_SUMMARY_METRICS:
            _require(
                int(row[f"{metric}_n_evaluable"]) == int(row["n_seeds"]),
                f"main-table model {key} has an unevaluable {metric} fit",
            )
        _require(
            int(row["constant_prediction_score_fits"]) == 0,
            f"main-table model {key} contains a collapsed constant-score fit",
        )
        row.update(spec)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("main_order", kind="mergesort").reset_index(drop=True)


def build_falta_panel_context(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for peptide, group in panel.groupby("peptide_design_code", sort=True):
        falta = group.loc[group["affibody_design_code"].eq(FALTA_CODE)]
        _require(len(falta) == 1, f"peptide {peptide} lacks exactly one FALTA result")
        best = float(group["target_retention"].max())
        best_affibodies = sorted(
            group.loc[np.isclose(group["target_retention"], best, rtol=0.0, atol=1e-12), "affibody_design_code"].astype(str)
        )
        falta_row = falta.iloc[0]
        falta_retention = float(falta_row["target_retention"])
        rows.append(
            {
                "peptide_design_code": str(peptide),
                "n_binders": int(group["target_binder"].sum()),
                "falta_retention": falta_retention,
                "falta_is_binder": int(falta_row["target_binder"]),
                "experimental_best_retention": best,
                "experimental_best_affibodies": json.dumps(best_affibodies, separators=(",", ":")),
                "falta_is_experimentally_optimal": int(math.isclose(falta_retention, best, rel_tol=0.0, abs_tol=1e-12)),
                "falta_retention_gap_to_best": best - falta_retention,
            }
        )
    output = pd.DataFrame(rows)
    _require(len(output) == EXPECTED_PEPTIDES, "FALTA context does not contain 12 peptides")
    _require(int(output["falta_is_binder"].sum()) == EXPECTED_FALTA_BINDER_PEPTIDES, "FALTA binder count changed")
    _require(
        int(output["falta_is_experimentally_optimal"].sum()) == EXPECTED_FALTA_OPTIMAL_PEPTIDES,
        "FALTA optimal-peptide count changed",
    )
    nonoptimal = tuple(sorted(output.loc[output["falta_is_experimentally_optimal"].eq(0), "peptide_design_code"].astype(str)))
    _require(nonoptimal == EXPECTED_FALTA_NONOPTIMAL_PEPTIDES, f"FALTA nonoptimal peptides changed: {nonoptimal}")
    return output.sort_values("peptide_design_code", kind="mergesort").reset_index(drop=True)


def build_model_diagnostics(
    ranked: pd.DataFrame,
    per_peptide: pd.DataFrame,
    panel_context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = panel_context.set_index("peptide_design_code", verify_integrity=True)
    peptide_index = per_peptide.set_index(
        ["bundle", "model", "seed", "peptide_design_code"], verify_integrity=True
    )
    detail_rows: list[dict[str, Any]] = []
    model_seed_rows: list[dict[str, Any]] = []
    for (bundle, model, seed), model_rows in ranked.groupby(["bundle", "model", "seed"], sort=True):
        per_model_detail: list[dict[str, Any]] = []
        for peptide, group in model_rows.groupby("peptide_design_code", sort=True):
            ordered = group.sort_values("within_peptide_rank", kind="mergesort")
            _require(len(ordered) == EXPECTED_AFFIBODIES, f"{bundle}/{model}/{seed}/{peptide} is not a 10-Affibody row")
            falta = ordered.loc[ordered["affibody_design_code"].eq(FALTA_CODE)]
            _require(len(falta) == 1, f"{bundle}/{model}/{seed}/{peptide} lacks FALTA")
            top3 = ordered.iloc[:3]
            metric_key = (str(bundle), str(model), str(seed), str(peptide))
            _require(metric_key in peptide_index.index, f"missing recomputed peptide metrics {metric_key}")
            metric = peptide_index.loc[metric_key]
            panel_row = context.loc[str(peptide)]
            row = {
                "bundle": str(bundle),
                "model": str(model),
                "model_key": _model_key(str(bundle), str(model)),
                "seed": str(seed),
                "peptide_design_code": str(peptide),
                "falta_is_experimentally_optimal": int(panel_row["falta_is_experimentally_optimal"]),
                "falta_retention": float(panel_row["falta_retention"]),
                "experimental_best_retention": float(panel_row["experimental_best_retention"]),
                "experimental_best_affibodies": str(panel_row["experimental_best_affibodies"]),
                "predicted_top1_affibody": str(ordered.iloc[0]["affibody_design_code"]),
                "predicted_top1_retention": float(ordered.iloc[0]["target_retention"]),
                "predicted_top1_is_falta": int(str(ordered.iloc[0]["affibody_design_code"]) == FALTA_CODE),
                "falta_predicted_rank": int(falta.iloc[0]["within_peptide_rank"]),
                "falta_selected_at_3": int(int(falta.iloc[0]["within_peptide_rank"]) <= 3),
                "top3_affibodies": json.dumps(top3["affibody_design_code"].astype(str).tolist(), separators=(",", ":")),
                "top3_contains_experimental_best": int(float(metric["regret_at_3"]) <= 1e-12),
                "within_peptide_spearman": float(metric["within_peptide_spearman"]),
                "precision_at_3": float(metric["precision_at_3"]),
                "best_retention_at_3": float(metric["best_retention_at_3"]),
                "regret_at_3": float(metric["regret_at_3"]),
            }
            detail_rows.append(row)
            per_model_detail.append(row)
        detail = pd.DataFrame(per_model_detail)
        nonoptimal = detail.loc[detail["falta_is_experimentally_optimal"].eq(0)]
        _require(tuple(sorted(nonoptimal["peptide_design_code"])) == EXPECTED_FALTA_NONOPTIMAL_PEPTIDES, "nonoptimal FALTA subset changed")
        model_seed_rows.append(
            {
                "bundle": str(bundle),
                "model": str(model),
                "model_key": _model_key(str(bundle), str(model)),
                "seed": str(seed),
                "top1_falta_peptides_of_12": int(detail["predicted_top1_is_falta"].sum()),
                "top3_contains_falta_peptides_of_12": int(detail["falta_selected_at_3"].sum()),
                "distinct_predicted_top1_affibodies": int(detail["predicted_top1_affibody"].nunique()),
                "top1_falta_on_el_mw_of_2": int(nonoptimal["predicted_top1_is_falta"].sum()),
                "top3_contains_falta_on_el_mw_of_2": int(nonoptimal["falta_selected_at_3"].sum()),
                "top3_contains_experimental_best_on_el_mw_of_2": int(nonoptimal["top3_contains_experimental_best"].sum()),
                "mean_within_peptide_spearman_on_el_mw": float(nonoptimal["within_peptide_spearman"].mean()),
                "mean_regret_at_3_on_el_mw": float(nonoptimal["regret_at_3"].mean()),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values(
        ["bundle", "model", "seed", "peptide_design_code"], kind="mergesort"
    ).reset_index(drop=True)
    by_seed = pd.DataFrame(model_seed_rows).sort_values(
        ["bundle", "model", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    return detail, by_seed


def summarize_falta_by_model(by_seed: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "top1_falta_peptides_of_12",
        "top3_contains_falta_peptides_of_12",
        "distinct_predicted_top1_affibodies",
        "top1_falta_on_el_mw_of_2",
        "top3_contains_falta_on_el_mw_of_2",
        "top3_contains_experimental_best_on_el_mw_of_2",
        "mean_within_peptide_spearman_on_el_mw",
        "mean_regret_at_3_on_el_mw",
    ]
    rows: list[dict[str, Any]] = []
    for (bundle, model, key), group in by_seed.groupby(["bundle", "model", "model_key"], sort=True):
        group = group.assign(_seed_order=group["seed"].map(_seed_sort_key)).sort_values("_seed_order")
        row: dict[str, Any] = {
            "bundle": bundle,
            "model": model,
            "model_key": key,
            "n_seeds": len(group),
        }
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="raise").to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{column}_n_evaluable"] = int(len(finite))
            row[f"{column}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{column}_sd"] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
            row[f"{column}_individual"] = json.dumps(
                [
                    {
                        "seed": str(seed),
                        "value": float(value) if math.isfinite(float(value)) else None,
                    }
                    for seed, value in zip(group["seed"], values)
                ],
                separators=(",", ":"),
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["bundle", "model"], kind="mergesort").reset_index(drop=True)


def summarize_per_peptide(detail: pd.DataFrame, main_keys: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = detail.loc[detail["model_key"].isin(main_keys)].copy()
    rows: list[dict[str, Any]] = []
    numeric = (
        "within_peptide_spearman",
        "precision_at_3",
        "best_retention_at_3",
        "regret_at_3",
        "predicted_top1_is_falta",
        "falta_predicted_rank",
        "falta_selected_at_3",
        "top3_contains_experimental_best",
    )
    for (bundle, model, key, peptide), group in main.groupby(
        ["bundle", "model", "model_key", "peptide_design_code"], sort=True
    ):
        group = group.assign(_seed_order=group["seed"].map(_seed_sort_key)).sort_values("_seed_order")
        row: dict[str, Any] = {
            "bundle": bundle,
            "model": model,
            "model_key": key,
            "peptide_design_code": peptide,
            "n_seeds": len(group),
            "experimental_best_retention": float(group["experimental_best_retention"].iloc[0]),
            "experimental_best_affibodies": str(group["experimental_best_affibodies"].iloc[0]),
            "falta_retention": float(group["falta_retention"].iloc[0]),
            "falta_is_experimentally_optimal": int(group["falta_is_experimentally_optimal"].iloc[0]),
            "predicted_top1_affibody_counts": json.dumps(
                dict(sorted(Counter(group["predicted_top1_affibody"].astype(str)).items())),
                separators=(",", ":"),
            ),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="raise").to_numpy(float)
            finite = values[np.isfinite(values)]
            row[f"{column}_n_evaluable"] = int(len(finite))
            row[f"{column}_mean"] = float(np.mean(finite)) if len(finite) else np.nan
            row[f"{column}_sd"] = float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan
            row[f"{column}_individual"] = json.dumps(
                [
                    {
                        "seed": str(seed),
                        "value": float(value) if math.isfinite(float(value)) else None,
                    }
                    for seed, value in zip(group["seed"], values)
                ],
                separators=(",", ":"),
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    focus = main.loc[main["peptide_design_code"].isin(EXPECTED_FALTA_NONOPTIMAL_PEPTIDES)].copy()
    return (
        main.sort_values(["model_key", "seed", "peptide_design_code"], kind="mergesort").reset_index(drop=True),
        summary.sort_values(["model_key", "peptide_design_code"], kind="mergesort").reset_index(drop=True),
    ), focus.sort_values(["model_key", "seed", "peptide_design_code"], kind="mergesort").reset_index(drop=True)


def _format_mean_sd(row: Mapping[str, Any], metric: str, digits: int = 4) -> str:
    mean = float(row[f"{metric}_mean"])
    sd = float(row[f"{metric}_sd"])
    evaluable = int(row.get(f"{metric}_n_evaluable", row["n_seeds"]))
    if not math.isfinite(mean) or evaluable == 0:
        return "not evaluable"
    if evaluable < 2 or not math.isfinite(sd):
        return f"{mean:.{digits}f} (one fit)"
    suffix = "" if evaluable == int(row["n_seeds"]) else f" ({evaluable}/{int(row['n_seeds'])} fits evaluable)"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}{suffix}"


def _decorate_summary(summary: pd.DataFrame, specs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    metadata = pd.DataFrame(specs)[
        ["model_key", "display_name", "selection_basis", "main_order"]
    ]
    return metadata.merge(summary, on="model_key", how="left", validate="one_to_one").sort_values("main_order")


def render_markdown(
    main: pd.DataFrame,
    all_summary: pd.DataFrame,
    focus: pd.DataFrame,
    panel_context: pd.DataFrame,
    selected: Mapping[str, str],
) -> str:
    lines = [
        "# LibB structural evaluation tables",
        "",
        "These tables summarize frozen scores on the corrected 120-pair panel. "
        "RDE-PPI-derived and StaB-ddG-derived representatives were read from the "
        "weak-label selection lock before retention was opened; no row below was "
        "selected for having a favorable retention result.",
        "",
        f"- Locked RDE-PPI-derived representative: `{selected['rde']}`",
        f"- Locked StaB-ddG-derived representative: `{selected['stab']}`",
        "- FALTA is a binder for 11/12 peptides and experimentally optimal for 10/12; it is not optimal for EL or MW.",
        "",
        "## Main peptide-conditioned and wet-lab metrics",
        "",
        "| Model | Average within-peptide Spearman ↑ | Precision at 3 ↑ | Best retention at 3 (%) ↑ | Regret at 3 (points) ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in main.iterrows():
        p3 = _format_mean_sd(row, "peptide_macro_precision_at_3")
        counts = json.loads(str(row["precision_at_3_binders_of_36_individual"]))
        unique_counts = sorted({int(item["binders"]) for item in counts})
        if len(unique_counts) == 1:
            p3 += f" ({unique_counts[0]}/36 in every fit)"
        else:
            p3 += " (individual binder counts: " + ", ".join(map(str, [item["binders"] for item in counts])) + "/36)"
        lines.append(
            "| {name} | {rho} | {p3} | {best} | {regret} |".format(
                name=row["display_name"],
                rho=_format_mean_sd(row, "within_peptide_spearman_mean"),
                p3=p3,
                best=_format_mean_sd(row, "peptide_macro_best_retention_at_3"),
                regret=_format_mean_sd(row, "peptide_macro_regret_at_3"),
            )
        )

    lines.extend(
        [
            "",
            "Individual seed values are preserved in `main_metrics_by_seed.csv` and as JSON in `main_metrics_seed_summary.csv`.",
            "",
            "## Secondary whole-panel metrics",
            "",
            "| Model | AUROC | Average precision |",
            "|---|---:|---:|",
        ]
    )
    for _, row in main.iterrows():
        lines.append(
            f"| {row['display_name']} | {_format_mean_sd(row, 'global_auroc')} | {_format_mean_sd(row, 'global_average_precision')} |"
        )

    lines.extend(
        [
            "",
            "## FALTA diagnostic for EL and MW",
            "",
            "FALTA remains in the primary 120-pair evaluation. The following is a post-hoc explanation of the two peptide rows where another Affibody has higher measured retention; it is not a model-selection rule.",
            "",
            "| Model | Peptide | FALTA retention | Experimental best retention (Affibody) | FALTA ranked first | FALTA included in top 3 | Regret at 3 |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    display_by_key = dict(zip(main["model_key"], main["display_name"]))
    for (key, peptide), group in focus.groupby(["model_key", "peptide_design_code"], sort=False):
        context = panel_context.loc[panel_context["peptide_design_code"].eq(peptide)].iloc[0]
        best_names = ", ".join(json.loads(context["experimental_best_affibodies"]))
        top1_count = int(group["predicted_top1_is_falta"].sum())
        top3_count = int(group["falta_selected_at_3"].sum())
        regret_mean = float(group["regret_at_3"].mean())
        regret_sd = float(group["regret_at_3"].std(ddof=1)) if len(group) >= 2 else float("nan")
        regret_text = f"{regret_mean:.4f}" if not math.isfinite(regret_sd) else f"{regret_mean:.4f} ± {regret_sd:.4f}"
        lines.append(
            f"| {display_by_key[key]} | {peptide} | {float(context['falta_retention']):.2f} | "
            f"{float(context['experimental_best_retention']):.2f} ({best_names}) | "
            f"{top1_count}/{len(group)} fits | {top3_count}/{len(group)} fits | {regret_text} |"
        )

    lines.extend(
        [
            "",
            "## All prespecified arms",
            "",
            "| Bundle | Model | Seeds | Within-peptide Spearman | Precision at 3 | Best retention at 3 | Regret at 3 | AUROC | Average precision | Stability warning |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in all_summary.iterrows():
        lines.append(
            "| {bundle} | `{model}` | {seeds} | {rho} | {p3} | {best} | {regret} | {auc} | {ap} | {warning} |".format(
                bundle=row["bundle"],
                model=row["model"],
                seeds=int(row["n_seeds"]),
                rho=_format_mean_sd(row, "within_peptide_spearman_mean"),
                p3=_format_mean_sd(row, "peptide_macro_precision_at_3"),
                best=_format_mean_sd(row, "peptide_macro_best_retention_at_3"),
                regret=_format_mean_sd(row, "peptide_macro_regret_at_3"),
                auc=_format_mean_sd(row, "global_auroc"),
                ap=_format_mean_sd(row, "global_average_precision"),
                warning=row["stability_warning"] or "--",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", path)


def publish_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    markdown: str,
    manifest: Mapping[str, Any],
) -> None:
    output_dir = validate_private_output_path(output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    records: dict[str, Any] = {}
    for filename, frame in tables.items():
        path = output_dir / filename
        _atomic_write_csv(frame, path)
        records[filename] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(frame)}
    markdown_path = output_dir / "summary_tables.md"
    _atomic_write_text(markdown, markdown_path)
    records[markdown_path.name] = {
        "path": str(markdown_path.resolve()),
        "sha256": sha256_file(markdown_path),
        "bytes": markdown_path.stat().st_size,
    }
    payload = dict(manifest)
    payload["outputs"] = records
    _atomic_write_json(payload, output_dir / "manifest.json")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--structural-evaluation", type=Path, required=True)
    parser.add_argument("--esmfold2-evaluation", type=Path, required=True)
    parser.add_argument("--mint-evaluation", type=Path, required=True)
    parser.add_argument("--retention-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    lock_path = args.selection_lock.resolve()
    lock = _read_json(lock_path)
    _require(isinstance(lock, Mapping), "invalid weak selection lock")
    selected = validate_selection_lock(lock)

    panel_path = args.retention_panel.resolve()
    panel = load_corrected_retention_panel(panel_path)
    _require(len(panel) == EXPECTED_ROWS, "corrected panel row count changed")
    _require(panel["peptide_design_code"].nunique() == EXPECTED_PEPTIDES, "corrected peptide count changed")
    _require(panel["affibody_design_code"].nunique() == EXPECTED_AFFIBODIES, "corrected Affibody count changed")
    _require(int(panel["target_binder"].sum()) == EXPECTED_BINDERS, "corrected binder count changed")

    structural = load_evaluation_bundle("structural", args.structural_evaluation, SEALED_EVALUATION_SCHEMA)
    esmfold2 = load_evaluation_bundle("esmfold2", args.esmfold2_evaluation, CORRECTED_EVALUATION_SCHEMA)
    mint = load_evaluation_bundle("mint", args.mint_evaluation, CORRECTED_EVALUATION_SCHEMA)
    bundles = (structural, esmfold2, mint)
    panel_sha256 = sha256_file(panel_path)
    for bundle in bundles:
        _require(str(bundle.panel_source.get("sha256", "")) == panel_sha256, f"{bundle.name} used another panel")

    validate_structural_models(structural, lock, selected)
    recomputed_peptide_frames: list[pd.DataFrame] = []
    recomputed_ranked_frames: list[pd.DataFrame] = []
    for bundle in bundles:
        peptide, ranked = validate_bundle_against_panel(bundle, panel)
        recomputed_peptide_frames.append(peptide)
        recomputed_ranked_frames.append(ranked)
    per_peptide = pd.concat(recomputed_peptide_frames, ignore_index=True)
    ranked = pd.concat(recomputed_ranked_frames, ignore_index=True)

    metrics = combine_metrics(bundles)
    all_summary = summarize_metrics(metrics)
    main_specs = build_main_model_specs(selected)
    main_summary = select_main_summary(all_summary, main_specs)
    main_keys = set(main_summary["model_key"].astype(str))
    main_by_seed = metrics.loc[metrics["model_key"].isin(main_keys)].merge(
        pd.DataFrame(main_specs)[["model_key", "display_name", "selection_basis", "main_order"]],
        on="model_key",
        how="left",
        validate="many_to_one",
    ).sort_values(["main_order", "seed"], kind="mergesort")

    panel_context = build_falta_panel_context(panel)
    detail, falta_by_seed = build_model_diagnostics(ranked, per_peptide, panel_context)
    falta_summary = summarize_falta_by_model(falta_by_seed)
    (main_per_peptide, main_per_peptide_summary), focus = summarize_per_peptide(detail, main_keys)
    markdown = render_markdown(main_summary, all_summary, focus, panel_context, selected)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "deterministic_summary_of_already_frozen_scores",
        "model_selection": {
            "structural_representatives_source": "weak-label selection lock only",
            "selection_metric": LOCK_SELECTION_METRIC,
            "retention_used_for_model_selection": False,
            "selected_models_by_family": dict(selected),
            "fixed_context_main_models": [dict(item) for item in FIXED_CONTEXT_MAIN_MODELS],
        },
        "panel": {
            "path": str(panel_path),
            "sha256": panel_sha256,
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_AFFIBODIES,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "binder_threshold": 75.0,
            "tie_policy": TIE_POLICY,
        },
        "selection_lock": {"path": str(lock_path), "sha256": sha256_file(lock_path)},
        "evaluation_bundles": {
            bundle.name: {
                "path": str(bundle.root),
                "manifest_sha256": sha256_file(bundle.root / "manifest.json"),
                "schema_version": bundle.manifest["schema_version"],
                "models": sorted(set(bundle.metrics["model"].astype(str))),
                "model_seed_groups": len(bundle.metrics),
            }
            for bundle in bundles
        },
        "metrics": {
            "primary": list(PRIMARY_METRICS),
            "secondary": list(SECONDARY_METRICS),
            "seed_sd": "sample standard deviation with ddof=1; undefined for a single archived fit",
            "precision_at_3_denominator": PRECISION_AT_3_DENOMINATOR,
        },
        "falta_diagnostic": {
            "post_hoc_description_not_model_selection": True,
            "binder_peptides": EXPECTED_FALTA_BINDER_PEPTIDES,
            "experimentally_optimal_peptides": EXPECTED_FALTA_OPTIMAL_PEPTIDES,
            "nonoptimal_peptides": list(EXPECTED_FALTA_NONOPTIMAL_PEPTIDES),
        },
        "producer": {
            "path": _relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    publish_outputs(
        args.output_dir,
        {
            "main_metrics_seed_summary.csv": main_summary,
            "main_metrics_by_seed.csv": main_by_seed,
            "secondary_metrics_seed_summary.csv": main_summary[
                [
                    "main_order",
                    "display_name",
                    "selection_basis",
                    "bundle",
                    "model",
                    "model_key",
                    "n_seeds",
                    "seeds",
                    *[column for metric in SECONDARY_METRICS for column in (f"{metric}_mean", f"{metric}_sd", f"{metric}_individual")],
                ]
            ],
            "all_arm_metrics_seed_summary.csv": all_summary,
            "all_arm_metrics_by_seed.csv": metrics,
            "falta_panel_context.csv": panel_context,
            "falta_diagnostics_by_model_seed.csv": falta_by_seed,
            "falta_diagnostics_seed_summary.csv": falta_summary,
            "per_peptide_main_models_by_seed.csv": main_per_peptide,
            "per_peptide_main_models_seed_summary.csv": main_per_peptide_summary,
            "falta_el_mw_main_models_by_seed.csv": focus,
        },
        markdown,
        manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_models_by_family": selected,
                "main_models": main_summary["model_key"].tolist(),
                "all_models": len(all_summary),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
