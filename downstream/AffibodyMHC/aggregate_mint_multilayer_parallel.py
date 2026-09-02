#!/usr/bin/env python
"""Audit and aggregate the distributed frozen-MINT layer sweep.

Each distributed evaluator compares one intermediate layer with the canonical
layer-33 control.  This program first reads only manifests, fold memberships,
and weak-label validation tables.  It fails closed unless all eight
prespecified intermediate layers and their layer-33 controls reproduce one
common experiment contract.  Layer and logistic-readout C are then selected
separately for LibA and LibB by minimum pooled weak-validation log loss.

Only after that choice has been frozen are direct-retention predictions and
metrics opened.  Retention outcomes therefore cannot influence layer or C
selection.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
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

from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC import evaluate_mint_multilayer as evaluator


SCHEMA_VERSION = "affibody-mint-multilayer-parallel-aggregate-v1"
EVALUATOR_SCHEMA = evaluator.SCHEMA_VERSION
LIBRARIES = ("LibA", "LibB")
INTERMEDIATE_LAYERS = (1, 5, 9, 13, 17, 21, 25, 29)
ALL_LAYERS = INTERMEDIATE_LAYERS + (33,)
C_GRID = tuple(float(value) for value in evaluator.CANONICAL_C_GRID)
FOLDS = int(evaluator.CANONICAL_FOLDS)
SPLIT_SEED = int(evaluator.CANONICAL_SPLIT_SEED)
RUN_REQUIRED_FILES = (
    "canonical_parity.json",
    "fold_membership.csv",
    "manifest.json",
    "model_audit.json",
    "retention_metrics.csv",
    "retention_predictions.csv",
    "selected_config.json",
    "weak_validation_by_layer.csv",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "JSON root is not an object: {}".format(path))
    return value


def _atomic_text(value, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        with open(str(temporary), "w") as handle:
            handle.write(value)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(value, path):
    _atomic_text(json.dumps(value, indent=2, sort_keys=True) + "\n", path)


def _write_csv(frame, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        frame.to_csv(str(temporary), index=False)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_record(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _close(left, right, rel_tol=1e-12, abs_tol=1e-14):
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _hash_frame(frame, columns):
    values = frame.loc[:, list(columns)].copy()
    text = values.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verify_manifest_outputs(run_dir, manifest):
    outputs = manifest.get("outputs", {})
    _require(isinstance(outputs, dict), "invalid output table in {}".format(run_dir))
    for filename in RUN_REQUIRED_FILES:
        path = run_dir / filename
        _require(path.is_file(), "missing completed output {}".format(path))
        if filename == "manifest.json":
            continue
        record = outputs.get(filename)
        _require(isinstance(record, dict), "manifest omits {}".format(path))
        _require(
            str(record.get("sha256", "")) == sha256_file(path),
            "output hash mismatch: {}".format(path),
        )


def _input_contract(manifest):
    inputs = manifest.get("inputs", {})
    names = (
        "multilayer_features",
        "multilayer_features_manifest",
        "cache_rows",
        "cache_rows_manifest",
        "canonical_layer33_cache",
        "matched_pipeline",
        "cached_evaluator",
        "script",
    )
    contract = {}
    for name in names:
        record = inputs.get(name)
        _require(isinstance(record, dict), "manifest lacks input {}".format(name))
        digest = str(record.get("sha256", ""))
        _require(len(digest) == 64, "invalid {} hash".format(name))
        contract[name] = digest
    return contract


def _verify_input_files(manifest):
    verified = []
    for name, record in sorted(manifest["inputs"].items()):
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        _require(path.is_file(), "recorded input is missing: {}".format(path))
        observed = sha256_file(path)
        _require(observed == str(record["sha256"]), "input hash changed: {}".format(path))
        verified.append({"name": name, "path": str(path), "sha256": observed})
    return verified


def _expected_weak_keys(layers, libraries=LIBRARIES):
    keys = set()
    for library in libraries:
        for layer in layers:
            for c_value in C_GRID:
                keys.add((library, int(layer), float(c_value), "aggregate", -1))
                for fold in range(FOLDS):
                    keys.add((library, int(layer), float(c_value), "fold", int(fold)))
    return keys


def _weak_keys(frame):
    return [
        (
            str(row.library),
            int(row.layer),
            float(row.C),
            str(row.record_type),
            int(row.fold),
        )
        for row in frame.itertuples(index=False)
    ]


def _validate_weak_table(frame, target_layer, run_dir, libraries):
    required = {
        "library",
        "layer",
        "representation",
        "record_type",
        "C",
        "fold",
        "n",
        "positive",
        "log_loss",
        "auroc",
        "ap",
        "n_train",
        "n_guard",
        "train_membership_sha256",
        "validation_membership_sha256",
    }
    _require(required.issubset(frame.columns), "weak table lacks columns in {}".format(run_dir))
    keys = _weak_keys(frame)
    expected = _expected_weak_keys((target_layer, 33), libraries=libraries)
    _require(len(keys) == len(set(keys)), "duplicate weak cells in {}".format(run_dir))
    _require(set(keys) == expected, "partial or mixed weak grid in {}".format(run_dir))
    for row in frame.itertuples(index=False):
        _require(
            str(row.representation) == evaluator.layer_feature_name(int(row.layer)),
            "representation/layer mismatch in {}".format(run_dir),
        )
        for name in ("log_loss", "auroc", "ap"):
            _require(math.isfinite(float(getattr(row, name))), "non-finite {} in {}".format(name, run_dir))
    return frame


def _validate_fold_membership(frame, run_dir, libraries):
    required = {"library", "fold"}
    _require(required.issubset(frame.columns), "fold table lacks keys in {}".format(run_dir))
    _require(set(frame["library"].astype(str)) == set(libraries), "fold libraries changed")
    _require(set(frame["fold"].astype(int)) == set(range(FOLDS)), "fold IDs changed")
    return {
        library: _hash_frame(
            frame.loc[frame["library"].astype(str).eq(library)]
            .sort_values(list(frame.columns))
            .reset_index(drop=True),
            frame.columns,
        )
        for library in libraries
    }


def load_weak_run(run_dir, target_layer, libraries):
    """Audit one run without opening any retention outcome artifact."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    _require(manifest.get("schema_version") == EVALUATOR_SCHEMA, "schema mismatch in {}".format(run_dir))
    _verify_manifest_outputs(run_dir, manifest)
    configuration = manifest.get("configuration", {})
    libraries = tuple(str(value) for value in libraries)
    _require(configuration.get("libraries") == list(libraries), "library contract changed")
    _require(
        tuple(int(value) for value in configuration.get("layers", [])) == (int(target_layer), 33),
        "layer contract changed in {}".format(run_dir),
    )
    _require(tuple(float(value) for value in configuration.get("c_grid", [])) == C_GRID, "C grid changed")
    _require(int(configuration.get("folds", -1)) == FOLDS, "fold count changed")
    _require(int(configuration.get("split_seed", -1)) == SPLIT_SEED, "split seed changed")
    _require(configuration.get("class_weight") == "balanced", "class weighting changed")
    _require(
        configuration.get("layer_C_selection") == "pooled weak-validation log loss only",
        "selection rule changed",
    )
    selection = manifest.get("selection", {})
    for library in libraries:
        value = selection.get(library, {})
        _require(value.get("direct_retention_used_for_selection") is False, "retention leaked into selection")
        _require(value.get("selection_frozen_before_retention_evaluation") is True, "selection was not frozen")
        _require(value.get("selection_data") == "weak PN labels only", "selection data changed")
    parity = _read_json(run_dir / "canonical_parity.json")
    _require(parity.get("all_checks_passed") is True, "canonical parity did not pass")
    _require(manifest.get("canonical_parity", {}).get("features_passed") is True, "feature parity failed")
    prediction_parity = manifest.get("canonical_parity", {}).get("predictions_passed", {})
    _require(all(prediction_parity.get(library) is True for library in libraries), "prediction parity failed")
    weak = pd.read_csv(run_dir / "weak_validation_by_layer.csv", float_precision="round_trip")
    weak = _validate_weak_table(weak, target_layer, run_dir, libraries)
    folds = pd.read_csv(run_dir / "fold_membership.csv", dtype=str)
    fold_hashes = _validate_fold_membership(folds, run_dir, libraries)
    return {
        "target_layer": int(target_layer),
        "run_dir": run_dir.resolve(),
        "manifest": manifest,
        "manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "input_contract": _input_contract(manifest),
        "fold_hashes": fold_hashes,
        "libraries": libraries,
        "weak": weak,
        "folds": folds,
    }


def combine_weak_runs(runs):
    """Combine weak tables, proving repeated layer-33 rows are identical."""
    expected_run_keys = {
        (library, layer) for library in LIBRARIES for layer in INTERMEDIATE_LAYERS
    }
    _require(set(runs) == expected_run_keys, "library/layer run grid is incomplete")
    input_contracts = {json.dumps(run["input_contract"], sort_keys=True) for run in runs.values()}
    _require(len(input_contracts) == 1, "input/code hashes differ across layer runs")

    blocks = []
    comparison_columns = [
        "library", "layer", "representation", "record_type", "C", "fold",
        "n", "positive", "log_loss", "auroc", "ap", "n_train", "n_guard",
        "train_membership_sha256", "validation_membership_sha256",
    ]
    for library in LIBRARIES:
        fold_hashes = {
            runs[(library, layer)]["fold_hashes"][library]
            for layer in INTERMEDIATE_LAYERS
        }
        _require(
            len(fold_hashes) == 1,
            "identity-cold fold membership differs across {} layer runs".format(library),
        )
        layer33_reference = None
        for layer in INTERMEDIATE_LAYERS:
            weak = runs[(library, layer)]["weak"].copy()
            weak = weak.loc[weak["library"].astype(str).eq(library)].copy()
            target = weak.loc[weak["layer"].astype(int).eq(layer)].copy()
            control = weak.loc[weak["layer"].astype(int).eq(33)].copy()
            blocks.append(target)
            normalized = control[comparison_columns].sort_values(
                ["library", "C", "record_type", "fold"], kind="mergesort"
            ).reset_index(drop=True)
            if layer33_reference is None:
                layer33_reference = normalized
            else:
                pd.testing.assert_frame_equal(
                    normalized,
                    layer33_reference,
                    check_exact=False,
                    rtol=1e-13,
                    atol=1e-15,
                )
        blocks.append(layer33_reference)
    combined = pd.concat(blocks, ignore_index=True)
    keys = _weak_keys(combined)
    _require(len(keys) == len(set(keys)), "combined weak grid has duplicate cells")
    _require(set(keys) == _expected_weak_keys(ALL_LAYERS), "combined weak grid is incomplete")
    return combined.sort_values(
        ["library", "layer", "C", "record_type", "fold"], kind="mergesort"
    ).reset_index(drop=True)


def select_from_weak(combined):
    """Select layer and C independently per library from aggregate weak rows."""
    selected = {}
    marked = combined.copy()
    marked["selected_joint_by_aggregate"] = 0
    for library in LIBRARIES:
        candidates = marked.loc[
            marked["library"].eq(library) & marked["record_type"].eq("aggregate")
        ]
        _require(len(candidates) == len(ALL_LAYERS) * len(C_GRID), "candidate grid changed")
        index = min(
            candidates.index,
            key=lambda idx: (
                float(marked.loc[idx, "log_loss"]),
                int(marked.loc[idx, "layer"]),
                float(marked.loc[idx, "C"]),
            ),
        )
        marked.loc[index, "selected_joint_by_aggregate"] = 1
        row = marked.loc[index]
        selected[library] = {
            "library": library,
            "layer": int(row["layer"]),
            "C": float(row["C"]),
            "weak_validation_log_loss": float(row["log_loss"]),
            "weak_validation_auroc_descriptive_only": float(row["auroc"]),
            "weak_validation_ap_descriptive_only": float(row["ap"]),
            "selection_metric": "minimum pooled weak-validation log loss only",
            "direct_retention_used_for_selection": False,
        }
    _require(int(marked["selected_joint_by_aggregate"].sum()) == len(LIBRARIES), "selection is not unique")
    return marked, selected


def _origin_run(runs, library, layer):
    target = int(layer) if int(layer) != 33 else INTERMEDIATE_LAYERS[0]
    return runs[(str(library), target)]


def load_selected_retention(runs, selections):
    """Open direct-retention artifacts only after weak selection is fixed."""
    metric_blocks = []
    prediction_blocks = []
    sources = []
    for library in LIBRARIES:
        selection = selections[library]
        run = _origin_run(runs, library, selection["layer"])
        run_dir = run["run_dir"]
        metrics = pd.read_csv(run_dir / "retention_metrics.csv", float_precision="round_trip")
        predictions = pd.read_csv(run_dir / "retention_predictions.csv", float_precision="round_trip")
        chosen_metrics = metrics.loc[
            metrics["library"].eq(library)
            & metrics["model"].eq("weak_selected_frozen_mint_layer")
        ].copy()
        _require(len(chosen_metrics) == 1, "selected retention metric is not unique")
        chosen = chosen_metrics.iloc[0]
        _require(int(chosen["layer"]) == int(selection["layer"]), "selected retention layer mismatch")
        _require(_close(chosen["C"], selection["C"]), "selected retention C mismatch")
        chosen_metrics["aggregate_role"] = "weak_selected_layer"
        chosen_predictions = predictions.loc[
            predictions["library"].eq(library)
            & predictions["model"].eq("weak_selected_frozen_mint_layer")
        ].copy()
        _require(int(len(chosen_predictions)) == int(chosen["n"]), "selected prediction count mismatch")
        chosen_ranking = evaluator.cached_eval.ranking_metrics(
            chosen_predictions.reset_index(drop=True),
            chosen_predictions["binder_score"].to_numpy(dtype=float),
        )
        chosen_metrics["global_spearman"] = float(chosen_ranking["global_spearman"])
        chosen_metrics["precision_at_3"] = float(chosen_ranking["peptide_precision_at_3"])
        chosen_metrics["precision_at_3_peptide_groups"] = int(
            chosen_predictions["chain1_sha256"].astype(str).nunique()
        )
        chosen_predictions["aggregate_role"] = "weak_selected_layer"

        control_metrics = metrics.loc[
            metrics["library"].eq(library)
            & metrics["model"].eq("canonical_frozen_mint_layer33_control")
        ].copy()
        _require(len(control_metrics) == 1, "layer-33 metric is not unique")
        control = control_metrics.iloc[0]
        _require(int(control["layer"]) == 33, "canonical control is not layer 33")
        control_metrics["aggregate_role"] = "canonical_layer33_control"
        control_predictions = predictions.loc[
            predictions["library"].eq(library)
            & predictions["model"].eq("canonical_frozen_mint_layer33_control")
        ].copy()
        _require(int(len(control_predictions)) == int(control["n"]), "control prediction count mismatch")
        control_ranking = evaluator.cached_eval.ranking_metrics(
            control_predictions.reset_index(drop=True),
            control_predictions["binder_score"].to_numpy(dtype=float),
        )
        control_metrics["global_spearman"] = float(control_ranking["global_spearman"])
        control_metrics["precision_at_3"] = float(control_ranking["peptide_precision_at_3"])
        control_metrics["precision_at_3_peptide_groups"] = int(
            control_predictions["chain1_sha256"].astype(str).nunique()
        )
        control_predictions["aggregate_role"] = "canonical_layer33_control"
        left = set(chosen_predictions["pair_uid"].astype(str))
        right = set(control_predictions["pair_uid"].astype(str))
        _require(left == right, "selected/control retention membership differs")

        metric_blocks.extend([chosen_metrics, control_metrics])
        prediction_blocks.extend([chosen_predictions, control_predictions])
        sources.append(
            {
                "library": library,
                "selected_layer": int(selection["layer"]),
                "selected_C": float(selection["C"]),
                "source_run": str(run_dir),
                "retention_metrics_sha256": sha256_file(run_dir / "retention_metrics.csv"),
                "retention_predictions_sha256": sha256_file(run_dir / "retention_predictions.csv"),
            }
        )
    return (
        pd.concat(metric_blocks, ignore_index=True),
        pd.concat(prediction_blocks, ignore_index=True),
        sources,
    )


def crosscheck_monolithic(monolithic_dir, combined, selections, retention_metrics):
    path = Path(monolithic_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "reason": "monolithic run was not complete at aggregation time"}
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == EVALUATOR_SCHEMA, "monolithic schema changed")
    configuration = manifest.get("configuration", {})
    _require(tuple(int(value) for value in configuration.get("layers", [])) == ALL_LAYERS, "monolithic layer grid changed")
    _require(tuple(float(value) for value in configuration.get("c_grid", [])) == C_GRID, "monolithic C grid changed")
    mono_weak = pd.read_csv(path / "weak_validation_by_layer.csv", float_precision="round_trip")
    compare_columns = [
        "library", "layer", "record_type", "C", "fold", "n", "positive",
        "log_loss", "auroc", "ap", "n_train", "n_guard",
        "train_membership_sha256", "validation_membership_sha256",
    ]
    left = combined[compare_columns].sort_values(
        ["library", "layer", "C", "record_type", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    right = mono_weak[compare_columns].sort_values(
        ["library", "layer", "C", "record_type", "fold"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-13, atol=1e-15)
    mono_selected = _read_json(path / "selected_config.json").get("selections", {})
    for library in LIBRARIES:
        _require(int(mono_selected[library]["layer"]) == int(selections[library]["layer"]), "monolithic selected layer differs")
        _require(_close(mono_selected[library]["C"], selections[library]["C"]), "monolithic selected C differs")
    mono_metrics = pd.read_csv(path / "retention_metrics.csv", float_precision="round_trip")
    metric_columns = [
        "library", "model", "layer", "C", "n", "positive", "auroc",
        "average_precision", "within_peptide_spearman",
    ]
    left_metrics = retention_metrics[metric_columns].sort_values(
        ["library", "model"], kind="mergesort"
    ).reset_index(drop=True)
    right_metrics = mono_metrics[metric_columns].sort_values(
        ["library", "model"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_metrics, right_metrics, check_exact=False, rtol=1e-11, atol=1e-13)
    return {
        "available": True,
        "all_weak_cells_match": True,
        "selected_configurations_match": True,
        "retention_metrics_match": True,
        "manifest_sha256": sha256_file(manifest_path),
    }


def _format_metric(value):
    return "{:.3f}".format(float(value))


def render_report(selections, metrics, input_audit, monolithic, run_runtime_seconds):
    lines = [
        "# Frozen MINT intermediate-layer experiment",
        "",
        "## Question",
        "",
        "MINT has 33 transformer layers. The existing predictor uses the final layer. "
        "This experiment asks whether an earlier frozen layer gives a more useful representation "
        "for the same LibA or LibB binder classifier.",
        "",
        "For each protein pair, the mean residue representation from the pMHC-side chain and "
        "the mean residue representation from the Affibody chain are concatenated into 2,560 "
        "numbers. A balanced logistic classifier (2,561 trainable parameters including the "
        "intercept) is then trained on the unchanged selection-derived labels.",
        "",
        "## Experimental control",
        "",
        "Layers 1, 5, 9, 13, 17, 21, 25, 29, and 33 were compared with the same identity-cold "
        "three-fold split (seed 17) and C values 0.001, 0.01, 0.1, and 1. Layer and C were chosen "
        "separately for LibA and LibB by the lowest pooled validation log loss on weak labels. "
        "Direct retention measurements were opened only after this choice was fixed.",
        "",
        "## Results",
        "",
        "| Library | Model chosen without retention labels | Weak validation log loss | Retention AUROC | Retention average precision | Global Spearman | Average within-peptide Spearman | Precision@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for library in LIBRARIES:
        selection = selections[library]
        selected = metrics.loc[
            metrics["library"].eq(library)
            & metrics["aggregate_role"].eq("weak_selected_layer")
        ].iloc[0]
        control = metrics.loc[
            metrics["library"].eq(library)
            & metrics["aggregate_role"].eq("canonical_layer33_control")
        ].iloc[0]
        lines.append(
            "| {} | layer {}, C={} | {} | {} | {} | {} | {} | {} |".format(
                library,
                int(selection["layer"]),
                "{:g}".format(float(selection["C"])),
                _format_metric(selection["weak_validation_log_loss"]),
                _format_metric(selected["auroc"]),
                _format_metric(selected["average_precision"]),
                _format_metric(selected["global_spearman"]),
                _format_metric(selected["within_peptide_spearman"]),
                _format_metric(selected["precision_at_3"]),
            )
        )
        lines.append(
            "| {} | final layer 33 control, C={} | — | {} | {} | {} | {} | {} |".format(
                library,
                "{:g}".format(float(control["C"])),
                _format_metric(control["auroc"]),
                _format_metric(control["average_precision"]),
                _format_metric(control["global_spearman"]),
                _format_metric(control["within_peptide_spearman"]),
                _format_metric(control["precision_at_3"]),
            )
        )
    lines.extend(
        [
            "",
            "Retention AUROC, average precision, and Precision@3 use the 75% retention definition "
            "of a binder. Precision@3 is calculated separately for each peptide as the binder "
            "fraction among its three highest-scoring Affibodies, then averaged across peptides. "
            "Within-peptide Spearman asks whether the model ranks all tested Affibodies correctly "
            "for the same peptide. Retention numbers are evaluation results, not model-selection "
            "criteria.",
            "",
            "## Audit",
            "",
            "- Eight independently launched layer jobs were complete and internally consistent.",
            "- All jobs used identical feature, row, split, code, and canonical-control hashes.",
            "- Every layer had 3 folds × 4 C values plus the pooled validation record for each library.",
            "- The newly extracted final-layer features and readout predictions passed the recorded canonical parity bounds.",
            "- Direct-retention prediction files were read only after the weak-label layer/C choices were frozen.",
            "- Combined evaluator runtime across the eight distributed jobs: {:.1f} minutes.".format(
                float(run_runtime_seconds) / 60.0
            ),
            "- Monolithic cross-check: {}.".format(
                "complete and matching" if monolithic.get("available") else "not complete when this report was written"
            ),
            "- Verified source files: {}.".format(len(input_audit)),
            "",
        ]
    )
    return "\n".join(lines)


def discover_runs(input_root):
    """Load a complete exact grid from paired, by-library, or mixed runs."""
    input_root = Path(input_root)
    paired = {
        layer: input_root / "layer{:02d}".format(layer)
        for layer in INTERMEDIATE_LAYERS
    }
    by_library = {
        (library, layer): input_root / "{}_layer{:02d}".format(library, layer)
        for library in LIBRARIES
        for layer in INTERMEDIATE_LAYERS
    }
    runs = {}
    paired_cache = {}
    source_types = set()
    missing = []
    for library in LIBRARIES:
        for layer in INTERMEDIATE_LAYERS:
            individual = by_library[(library, layer)]
            if (individual / "manifest.json").is_file():
                runs[(library, layer)] = load_weak_run(
                    individual, layer, (library,)
                )
                source_types.add("by_library")
                continue
            shared = paired[layer]
            if (shared / "manifest.json").is_file():
                if layer not in paired_cache:
                    paired_cache[layer] = load_weak_run(shared, layer, LIBRARIES)
                runs[(library, layer)] = paired_cache[layer]
                source_types.add("paired")
                continue
            missing.append((library, layer))
    _require(not missing, "exact library/layer outputs are incomplete: {}".format(missing))
    if source_types == {"by_library"}:
        layout = "sixteen_by_library_runs"
    elif source_types == {"paired"}:
        layout = "eight_paired_library_runs"
    else:
        layout = "mixed_exact_paired_and_by_library_runs"
    return runs, layout


def run(args):
    started = time.time()
    input_root = Path(args.input_root)
    _require(input_root.is_dir(), "parallel output root does not exist")
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")

    runs, distributed_layout = discover_runs(input_root)
    combined = combine_weak_runs(runs)
    combined, selections = select_from_weak(combined)

    # All model choices are now frozen. Only this point onward reads direct
    # retention metrics or predictions.
    metrics, predictions, retention_sources = load_selected_retention(runs, selections)
    monolithic = crosscheck_monolithic(args.monolithic_dir, combined, selections, metrics)

    representative_run = runs[(LIBRARIES[0], INTERMEDIATE_LAYERS[0])]
    representative = representative_run["manifest"]
    input_audit = _verify_input_files(representative)
    unique_runs = {
        str(run["run_dir"]): run for run in runs.values()
    }
    runtime_seconds = sum(
        float(run["manifest"].get("runtime_seconds", 0.0))
        for run in unique_runs.values()
    )
    fold_membership = pd.concat(
        [
            runs[(library, INTERMEDIATE_LAYERS[0])]["folds"].loc[
                runs[(library, INTERMEDIATE_LAYERS[0])]["folds"]["library"].astype(str).eq(library)
            ].copy()
            for library in LIBRARIES
        ],
        ignore_index=True,
    )

    output_dir.mkdir(parents=True, mode=0o700)
    weak_path = output_dir / "weak_validation_by_layer.csv"
    selected_path = output_dir / "selected_config.json"
    fold_path = output_dir / "fold_membership.csv"
    metric_path = output_dir / "retention_metrics.csv"
    prediction_path = output_dir / "retention_predictions.csv"
    audit_path = output_dir / "audit.json"
    report_path = output_dir / "mint_multilayer_report.md"
    _write_csv(combined, weak_path)
    _write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "selection_rule": "minimum pooled weak-validation log loss; layer then C deterministic tie-break",
            "retention_labels_used_for_selection": False,
            "selections": selections,
        },
        selected_path,
    )
    _write_csv(fold_membership, fold_path)
    _write_csv(metrics, metric_path)
    _write_csv(predictions, prediction_path)
    audit = {
        "all_checks_passed": True,
        "distributed_layout": distributed_layout,
        "input_contract": representative_run["input_contract"],
        "verified_inputs": input_audit,
        "fold_membership_sha256_by_library": {
            library: runs[(library, INTERMEDIATE_LAYERS[0])]["fold_hashes"][library]
            for library in LIBRARIES
        },
        "parallel_runs": [
            {
                "libraries": list(run["libraries"]),
                "target_layer": int(run["target_layer"]),
                "run_dir": str(run["run_dir"]),
                "manifest_sha256": run["manifest_sha256"],
                "runtime_seconds": float(run["manifest"].get("runtime_seconds", 0.0)),
            }
            for _, run in sorted(unique_runs.items())
        ],
        "retention_sources_after_selection": retention_sources,
        "monolithic_crosscheck": monolithic,
    }
    _write_json(audit, audit_path)
    _atomic_text(
        render_report(selections, metrics, input_audit, monolithic, runtime_seconds),
        report_path,
    )

    output_files = (
        weak_path,
        selected_path,
        fold_path,
        metric_path,
        prediction_path,
        audit_path,
        report_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "runtime_seconds": float(time.time() - started),
        "configuration": {
            "libraries": list(LIBRARIES),
            "layers": list(ALL_LAYERS),
            "c_grid": list(C_GRID),
            "folds": FOLDS,
            "split_seed": SPLIT_SEED,
            "selection_metric": "minimum pooled weak-validation log loss only",
            "direct_retention_used_for_selection": False,
        },
        "selections": selections,
        "audit": audit,
        "script": _source_record(Path(__file__)),
        "outputs": {path.name: _source_record(path) for path in output_files},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _write_json(manifest, output_dir / "manifest.json")
    print("wrote {}".format(output_dir))
    print(json.dumps(selections, indent=2, sort_keys=True))
    print(metrics.to_string(index=False))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default=str(REPO_ROOT / "private_data/experiments/mint_multilayer_eval_parallel_v1"),
    )
    parser.add_argument(
        "--monolithic-dir",
        default=str(REPO_ROOT / "private_data/experiments/mint_multilayer_eval_v1"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "private_data/experiments/mint_multilayer_eval_aggregate_v1"),
    )
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
