#!/usr/bin/env python
"""Audit and aggregate the distributed frozen-MINT P/N/U experiment.

The distributed trainer writes one fixed ``(library, pi, eta)`` arm per
directory.  This program fails closed unless the complete prespecified
2-library x 4-prior x 5-eta grid is present and internally consistent.  It
then performs the only cross-arm choice allowed by the experiment:

    weak-validation AUROC (higher), AP (higher), log loss (lower),
    eta (lower), C (lower), epoch (lower)

The choice is made separately for every library and fixed class prior.  The
four priors are all retained; retention measurements are never used to choose
one.  Retention files are opened only after weak-label selection is complete.

Completed 80-epoch eta=0 reruns replace their matching 40-epoch arms only
after their trainer/cache/configuration hashes and their entire 40-epoch
history prefix have been verified.  A separate exact balanced-sklearn P/N
readout is loaded as the established frozen-MINT control.
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
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)


SCHEMA_VERSION = "affibody-frozen-mint-pnu-grid-aggregate-v1"
TRAINER_SCHEMA = "affibody-frozen-mint-pnu-readout-v1"
LIBRARIES = ("LibA", "LibB")
PRIORS = (0.02, 0.05, 0.10, 0.15)
ETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
C_GRID = (0.001, 0.01, 0.1, 1.0)
BASE_EPOCHS = tuple(range(1, 41))
EXTENDED_EPOCHS = tuple(range(1, 81))
FOLDS = 3
SPLIT_SEED = 17
TRAINING_SEED = 17
U_PER_POSITIVE = 5.0
REQUIRED_TRAINER_OUTPUTS = (
    "data_counts.csv",
    "fitted_heads.npz",
    "pnu_weak_validation.csv",
    "retention_metrics.csv",
    "retention_predictions.csv",
    "selected_hyperparameters.csv",
    "sklearn_pn_weak_validation.csv",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    output = float(value)
    _require(math.isfinite(output), "{} is not finite".format(label))
    return output


def _close(left, right, rel_tol=1e-10, abs_tol=1e-12):
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _read_json(path):
    with open(str(path), "r") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "JSON root is not an object: {}".format(path))
    return payload


def _atomic_text(text, path):
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    try:
        with open(str(temporary), "w") as handle:
            handle.write(text)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _write_json(payload, path):
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def weak_selection_key(row):
    """Return the prespecified weak-label-only ordering key."""
    return (
        -_finite(row["weak_validation_auroc"], "weak-validation AUROC"),
        -_finite(row["weak_validation_ap"], "weak-validation AP"),
        _finite(row["weak_validation_log_loss"], "weak-validation log loss"),
        _finite(row["eta"], "eta"),
        _finite(row["C"], "C"),
        int(row["epoch"]),
    )


def history_selection_key(row):
    """Equivalent ordering for an aggregate row in trainer history."""
    return (
        -_finite(row["auroc"], "weak-validation AUROC"),
        -_finite(row["ap"], "weak-validation AP"),
        _finite(row["log_loss"], "weak-validation log loss"),
        _finite(row["eta"], "eta"),
        _finite(row["C"], "C"),
        int(row["epoch"]),
    )


def select_eta_per_fixed_pi(candidates):
    """Select eta/C/epoch per fixed library and pi, retaining every pi."""
    required = {
        "library",
        "pi",
        "eta",
        "C",
        "epoch",
        "weak_validation_auroc",
        "weak_validation_ap",
        "weak_validation_log_loss",
    }
    _require(required.issubset(candidates.columns), "candidate table lacks selection columns")
    frame = candidates.copy()
    frame["selected_eta_by_aggregate"] = 0
    selected_indices = []
    expected_groups = {(library, pi) for library in LIBRARIES for pi in PRIORS}
    observed_groups = set()
    for (library, pi), indices in frame.groupby(["library", "pi"], sort=True).groups.items():
        pi = float(pi)
        group_key = (str(library), pi)
        observed_groups.add(group_key)
        block = frame.loc[list(indices)]
        observed_etas = set(float(value) for value in block["eta"])
        _require(
            observed_etas == set(ETAS) and len(block) == len(ETAS),
            "eta grid is incomplete or duplicated for {}/{}: {}".format(
                library, pi, sorted(observed_etas)
            ),
        )
        best = min(list(indices), key=lambda index: weak_selection_key(frame.loc[index]))
        frame.loc[best, "selected_eta_by_aggregate"] = 1
        selected_indices.append(best)
    _require(
        observed_groups == expected_groups,
        "library/pi grid mismatch; missing={} extra={}".format(
            sorted(expected_groups - observed_groups),
            sorted(observed_groups - expected_groups),
        ),
    )
    selected = frame.loc[selected_indices].copy()
    selected = selected.sort_values(["library", "pi"], kind="mergesort").reset_index(drop=True)
    return frame.sort_values(["library", "pi", "eta"], kind="mergesort").reset_index(drop=True), selected


def _trainer_hash(manifest):
    code = manifest.get("code", {})
    _require(Path(str(code.get("path", ""))).name == "train_pnu_mint_readout.py", "wrong trainer recorded")
    value = str(code.get("sha256", ""))
    _require(len(value) == 64, "invalid trainer hash")
    return value


def _source_hash_contract(manifest):
    inputs = manifest.get("inputs", {})
    pn = inputs.get("pn_cache_paths", [])
    _require(isinstance(pn, list) and pn, "manifest has no P/N cache")
    pn_hashes = tuple(sorted(str(item.get("sha256", "")) for item in pn))
    pnu_cache = inputs.get("pnu_cache") or {}
    pnu_rows = inputs.get("pnu_rows") or {}
    pnu_manifest = inputs.get("pnu_manifest") or {}
    contract = {
        "pn_cache_sha256": pn_hashes,
        "pnu_cache_sha256": str(pnu_cache.get("sha256", "")),
        "pnu_rows_sha256": str(pnu_rows.get("sha256", "")),
        "pnu_manifest_sha256": str(pnu_manifest.get("sha256", "")),
    }
    for name, value in contract.items():
        values = value if isinstance(value, tuple) else (value,)
        _require(all(len(item) == 64 for item in values), "invalid {}".format(name))
    return contract


def _resolve_recorded_path(value):
    path = Path(str(value))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def verify_source_files(manifest):
    """Hash each large shared source once, after cross-run uniformity checks."""
    inputs = manifest["inputs"]
    records = list(inputs["pn_cache_paths"])
    records.extend([inputs["pnu_cache"], inputs["pnu_rows"], inputs["pnu_manifest"]])
    observed = []
    seen = set()
    for record in records:
        path = _resolve_recorded_path(record["path"])
        key = (str(path), str(record["sha256"]))
        if key in seen:
            continue
        seen.add(key)
        _require(path.is_file(), "recorded source is missing: {}".format(path))
        actual = sha256_file(path)
        _require(actual == record["sha256"], "recorded source hash changed: {}".format(path))
        observed.append({"path": str(path), "sha256": actual})
    return observed


def sampled_u_audit(manifest):
    """Resolve the deterministic-U seed and full eligible-pool denominators."""
    merged = ((manifest.get("inputs", {}).get("pnu_manifest") or {}).get("payload") or {})
    prepare_record = merged.get("input", {}).get("prepare_manifest", {})
    prepare_path = _resolve_recorded_path(prepare_record.get("path", ""))
    _require(prepare_path.is_file(), "sampled-U prepare manifest is missing")
    _require(sha256_file(prepare_path) == prepare_record.get("sha256"), "sampled-U prepare manifest hash changed")
    prepare = _read_json(prepare_path)
    selection = prepare.get("selection", {})
    _require(selection.get("u_mode") == "deterministic_sample", "U sample is not deterministic")
    _require(_close(selection.get("u_per_positive"), U_PER_POSITIVE), "prepare-manifest U/P changed")
    sample_seed = int(selection.get("sample_seed"))
    pool_record = prepare.get("sources", {}).get("u_manifest", {})
    pool_path = _resolve_recorded_path(pool_record.get("path", ""))
    _require(pool_path.is_file(), "eligible-U pool manifest is missing")
    _require(sha256_file(pool_path) == pool_record.get("sha256"), "eligible-U pool manifest hash changed")
    pool = _read_json(pool_path)
    libraries = {}
    for library in LIBRARIES:
        sampled = int(prepare["rows"]["libraries"][library]["U"])
        eligible = int(pool["libraries"][library]["eligible_unlabeled_rows"])
        libraries[library] = {
            "sampled_u": sampled,
            "eligible_u": eligible,
            "coverage_fraction": float(sampled) / float(eligible),
        }
    return {
        "sample_seed": sample_seed,
        "prepare_manifest": {"path": str(prepare_path), "sha256": sha256_file(prepare_path)},
        "eligible_pool_manifest": {"path": str(pool_path), "sha256": sha256_file(pool_path)},
        "libraries": libraries,
    }


def _verify_outputs(run_dir, manifest):
    outputs = manifest.get("outputs", {})
    _require(isinstance(outputs, dict), "manifest outputs are invalid in {}".format(run_dir))
    for filename in REQUIRED_TRAINER_OUTPUTS:
        _require(filename in outputs, "manifest omits {} in {}".format(filename, run_dir))
        path = run_dir / filename
        _require(path.is_file(), "missing trainer output: {}".format(path))
        recorded = outputs[filename]
        _require(isinstance(recorded, str) and len(recorded) == 64, "invalid output hash for {}".format(path))
        _require(sha256_file(path) == recorded, "trainer output hash mismatch: {}".format(path))


def _configuration_signature(configuration, include_epoch=True):
    names = (
        "libraries",
        "pi_grid",
        "eta_grid",
        "C_grid",
        "folds",
        "split_seed",
        "training_seeds",
        "weak_selection_metric",
        "batch_size_per_role",
        "learning_rate",
        "u_contract",
        "u_per_positive",
        "risk",
        "l2",
        "standardization",
        "readout",
    )
    output = {name: configuration.get(name) for name in names}
    if include_epoch:
        output["epoch_grid"] = configuration.get("epoch_grid")
    return output


def _validate_common_manifest(manifest, run_dir, expect_control=False):
    _require(manifest.get("schema_version") == TRAINER_SCHEMA, "trainer schema mismatch in {}".format(run_dir))
    configuration = manifest.get("configuration", {})
    _require(configuration.get("C_grid") == list(C_GRID), "C grid changed in {}".format(run_dir))
    _require(int(configuration.get("folds", -1)) == FOLDS, "fold count changed in {}".format(run_dir))
    _require(int(configuration.get("split_seed", -1)) == SPLIT_SEED, "split seed changed in {}".format(run_dir))
    _require(configuration.get("training_seeds") == [TRAINING_SEED], "training seed changed in {}".format(run_dir))
    _require(configuration.get("weak_selection_metric") == "auroc", "selection metric changed in {}".format(run_dir))
    _require(configuration.get("u_contract") == "sampled_u_per_positive", "U contract changed in {}".format(run_dir))
    _require(_close(configuration.get("u_per_positive"), U_PER_POSITIVE), "U/P ratio changed in {}".format(run_dir))
    _require(
        bool(configuration.get("sklearn_pn_control_included")) == bool(expect_control),
        "sklearn-control role changed in {}".format(run_dir),
    )
    retention_usage = manifest.get("retention_usage", {})
    _require(
        retention_usage.get("used_for_training_early_stopping_or_selection") is False,
        "retention-use contract changed in {}".format(run_dir),
    )
    pnu_payload = ((manifest.get("inputs", {}).get("pnu_manifest") or {}).get("payload") or {})
    features = pnu_payload.get("features", {})
    _require(int(features.get("layer", -1)) == 33, "PNU cache is not MINT layer 33 in {}".format(run_dir))
    shape = features.get("shape", [])
    _require(len(shape) == 2 and int(shape[1]) == 2560, "MINT feature width changed in {}".format(run_dir))
    return configuration


def _single_count_contract(manifest, run_dir):
    records = manifest.get("counts", [])
    _require(isinstance(records, list) and len(records) == 1, "count contract is not singular in {}".format(run_dir))
    row = records[0]
    output = {
        "library": str(row["library"]),
        "positive": int(row["positive"]),
        "negative": int(row["negative"]),
        "unlabeled_used": int(row["unlabeled_used"]),
        "retention_evaluation": int(row["retention_evaluation"]),
        "u_contract": str(row["u_contract"]),
        "u_per_positive": float(row["u_per_positive"]),
    }
    table = pd.read_csv(run_dir / "data_counts.csv", float_precision="round_trip")
    _require(len(table) == 1, "data_counts.csv is not singular in {}".format(run_dir))
    stored = table.iloc[0]
    for name in ("library", "u_contract"):
        _require(str(stored[name]) == str(output[name]), "{} differs between manifest and data_counts.csv".format(name))
    for name in ("positive", "negative", "unlabeled_used", "retention_evaluation"):
        _require(int(stored[name]) == int(output[name]), "{} differs between manifest and data_counts.csv".format(name))
    _require(_close(stored["u_per_positive"], output["u_per_positive"]), "U/P differs between manifest and data_counts.csv")
    return output


def validate_weak_history(frame, library, pi, eta, epoch_grid):
    required = {"library", "record_type", "pi", "eta", "C", "epoch", "fold", "training_seed", "log_loss", "auroc", "ap"}
    _require(required.issubset(frame.columns), "weak history lacks required columns")
    _require(set(frame["library"].astype(str)) == {library}, "weak-history library mismatch")
    _require(np.allclose(frame["pi"].astype(float), float(pi)), "weak-history pi mismatch")
    _require(np.allclose(frame["eta"].astype(float), float(eta)), "weak-history eta mismatch")
    _require(set(frame["training_seed"].astype(int)) == {TRAINING_SEED}, "weak-history training seed mismatch")
    expected_keys = set()
    for c_value in C_GRID:
        for epoch in epoch_grid:
            expected_keys.add(("aggregate", float(c_value), int(epoch), -1))
            for fold in range(FOLDS):
                expected_keys.add(("fold", float(c_value), int(epoch), fold))
    observed_keys = [
        (str(row.record_type), float(row.C), int(row.epoch), int(row.fold))
        for row in frame.itertuples(index=False)
    ]
    _require(len(observed_keys) == len(set(observed_keys)), "duplicate weak-history cells")
    _require(set(observed_keys) == expected_keys, "weak-history grid is partial or has extra cells")
    aggregate = frame.loc[frame["record_type"].eq("aggregate")].copy()
    best_index = min(aggregate.index, key=lambda index: history_selection_key(aggregate.loc[index]))
    row = aggregate.loc[best_index]
    return {
        "library": library,
        "pi": float(pi),
        "eta": float(eta),
        "C": float(row["C"]),
        "epoch": int(row["epoch"]),
        "weak_validation_auroc": float(row["auroc"]),
        "weak_validation_ap": float(row["ap"]),
        "weak_validation_log_loss": float(row["log_loss"]),
    }


def _verify_selected_row(run_dir, selected, expected):
    fixed = selected.loc[selected["selection_scope"].eq("C_epoch_at_fixed_eta")]
    _require(len(fixed) == 1, "fixed-eta selection row is not unique in {}".format(run_dir))
    row = fixed.iloc[0]
    pairs = (
        ("library", row["library"], expected["library"], str),
        ("pi", row["pi"], expected["pi"], float),
        ("eta", row["eta"], expected["eta"], float),
        ("C", row["C"], expected["C"], float),
        ("epoch", row["epoch"], expected["epoch"], int),
        ("weak_validation_auroc", row["weak_validation_auroc"], expected["weak_validation_auroc"], float),
        ("weak_validation_ap", row["weak_validation_ap"], expected["weak_validation_ap"], float),
        ("weak_validation_log_loss", row["weak_validation_log_loss"], expected["weak_validation_log_loss"], float),
    )
    for label, observed, wanted, converter in pairs:
        if converter is str:
            match = str(observed) == str(wanted)
        elif converter is int:
            match = int(observed) == int(wanted)
        else:
            match = _close(observed, wanted)
        _require(match, "stored {} does not reproduce in {}".format(label, run_dir))
    _require(str(row["weak_selection_metric"]) == "auroc", "stored weak selection metric changed")


def load_fixed_arm_weak(run_dir, expected_epochs, expect_control=False):
    """Validate one run without opening its retention outcomes."""
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "partial run has no manifest: {}".format(run_dir))
    manifest = _read_json(manifest_path)
    _verify_outputs(run_dir, manifest)
    configuration = _validate_common_manifest(manifest, run_dir, expect_control=expect_control)
    libraries = configuration.get("libraries", [])
    pis = configuration.get("pi_grid", [])
    etas = configuration.get("eta_grid", [])
    _require(len(libraries) == len(pis) == len(etas) == 1, "run is not one fixed arm: {}".format(run_dir))
    library, pi, eta = str(libraries[0]), float(pis[0]), float(etas[0])
    _require(library in LIBRARIES and pi in PRIORS and eta in ETAS, "unexpected arm in {}".format(run_dir))
    _require(configuration.get("epoch_grid") == list(expected_epochs), "epoch grid changed in {}".format(run_dir))
    history = pd.read_csv(run_dir / "pnu_weak_validation.csv", float_precision="round_trip")
    selected = validate_weak_history(history, library, pi, eta, expected_epochs)
    stored = pd.read_csv(run_dir / "selected_hyperparameters.csv", float_precision="round_trip")
    _verify_selected_row(run_dir, stored, selected)
    counts = _single_count_contract(manifest, run_dir)
    _require(counts["library"] == library, "count/library mismatch in {}".format(run_dir))
    _require(counts["u_contract"] == "sampled_u_per_positive", "count U contract changed")
    _require(_close(counts["u_per_positive"], U_PER_POSITIVE), "count U/P ratio changed")
    selected.update(
        {
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "max_epoch": int(max(expected_epochs)),
            "trainer_sha256": _trainer_hash(manifest),
            "manifest_sha256": sha256_file(manifest_path),
            "source_hash_contract": _source_hash_contract(manifest),
            "configuration_signature": _configuration_signature(configuration, include_epoch=False),
            "manifest": manifest,
            "history": history,
            "counts": counts,
        }
    )
    return selected


def _visible_run_dirs(root):
    _require(root.is_dir(), "run root does not exist: {}".format(root))
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name != "logs")


def load_complete_base_grid(root):
    run_dirs = _visible_run_dirs(root)
    _require(len(run_dirs) == len(LIBRARIES) * len(PRIORS) * len(ETAS), "base grid must contain exactly 40 arm directories")
    arms = {}
    for run_dir in run_dirs:
        arm = load_fixed_arm_weak(run_dir, BASE_EPOCHS, expect_control=False)
        key = (arm["library"], arm["pi"], arm["eta"])
        _require(key not in arms, "duplicate base arm: {}".format(key))
        arms[key] = arm
    expected = {(library, pi, eta) for library in LIBRARIES for pi in PRIORS for eta in ETAS}
    _require(set(arms) == expected, "base arm grid is partial or mixed")
    return arms


def _verify_uniform_contract(arms):
    trainer_hashes = {arm["trainer_sha256"] for arm in arms}
    _require(len(trainer_hashes) == 1, "mixed trainer hashes across runs")
    source_contracts = {
        json.dumps(arm["source_hash_contract"], sort_keys=True) for arm in arms
    }
    _require(len(source_contracts) == 1, "mixed MINT/PNU cache hashes across runs")
    count_by_library = {}
    for arm in arms:
        library = arm["library"]
        serial = json.dumps(arm["counts"], sort_keys=True)
        if library in count_by_library:
            _require(serial == count_by_library[library], "row counts changed within {}".format(library))
        else:
            count_by_library[library] = serial
    return next(iter(trainer_hashes)), json.loads(next(iter(source_contracts)))


def _compare_history_prefix(base, extension):
    columns = ["record_type", "pi", "eta", "C", "epoch", "fold"]
    metrics = ["log_loss", "auroc", "ap"]
    extended_prefix = extension["history"].loc[extension["history"]["epoch"].astype(int) <= max(BASE_EPOCHS)]
    joined = base["history"].merge(extended_prefix, on=columns, how="outer", suffixes=("_base", "_extension"), indicator=True)
    _require(joined["_merge"].eq("both").all() and len(joined) == len(base["history"]), "extension does not reproduce complete 40-epoch prefix")
    for metric in metrics:
        _require(
            np.allclose(joined[metric + "_base"], joined[metric + "_extension"], rtol=1e-12, atol=1e-14, equal_nan=True),
            "extension 40-epoch prefix changed {}".format(metric),
        )


def integrate_boundary_extensions(base_arms, extension_root):
    """Replace exactly those eta=0 base arms whose 40-epoch optimum hit 40."""
    expected_keys = {
        key for key, arm in base_arms.items()
        if float(key[2]) == 0.0 and int(arm["epoch"]) == max(BASE_EPOCHS)
    }
    run_dirs = _visible_run_dirs(extension_root)
    _require(len(run_dirs) == len(expected_keys), "boundary extension set is partial or contains extra arms")
    extensions = {}
    for run_dir in run_dirs:
        arm = load_fixed_arm_weak(run_dir, EXTENDED_EPOCHS, expect_control=False)
        key = (arm["library"], arm["pi"], arm["eta"])
        _require(key in expected_keys, "extension is not for an eta=0 boundary arm: {}".format(key))
        _require(key not in extensions, "duplicate extension arm: {}".format(key))
        base = base_arms[key]
        _require(arm["trainer_sha256"] == base["trainer_sha256"], "extension trainer hash changed")
        _require(arm["source_hash_contract"] == base["source_hash_contract"], "extension cache hashes changed")
        _require(arm["configuration_signature"] == base["configuration_signature"], "extension configuration changed beyond epoch range")
        _require(arm["counts"] == base["counts"], "extension row counts changed")
        _compare_history_prefix(base, arm)
        extensions[key] = arm
    _require(set(extensions) == expected_keys, "not every eta=0 boundary arm has an extension")
    combined = dict(base_arms)
    audit_rows = []
    for key in sorted(base_arms):
        base = base_arms[key]
        if key in extensions:
            extension = extensions[key]
            combined[key] = extension
            audit_rows.append(
                {
                    "library": key[0], "pi": key[1], "eta": key[2],
                    "base_epoch": base["epoch"], "extended_epoch": extension["epoch"],
                    "base_run_name": base["run_name"], "selected_run_name": extension["run_name"],
                    "extension_used": 1, "prefix_reproduced": 1,
                }
            )
        else:
            audit_rows.append(
                {
                    "library": key[0], "pi": key[1], "eta": key[2],
                    "base_epoch": base["epoch"], "extended_epoch": np.nan,
                    "base_run_name": base["run_name"], "selected_run_name": base["run_name"],
                    "extension_used": 0, "prefix_reproduced": 0,
                }
            )
    return combined, pd.DataFrame(audit_rows), extensions


def _sklearn_control_key(row):
    return (
        -_finite(row["auroc"], "control weak AUROC"),
        -_finite(row["ap"], "control weak AP"),
        _finite(row["log_loss"], "control weak log loss"),
        _finite(row["C"], "control C"),
    )


def _historical_control_key(row):
    """Original frozen-MINT selector: log loss, AP, AUROC, then C."""
    return (
        _finite(row["log_loss"], "historical-control weak log loss"),
        -_finite(row["auprc"], "historical-control weak AP"),
        -_finite(row["auroc"], "historical-control weak AUROC"),
        _finite(row["C"], "historical-control C"),
    )


def load_exact_controls_weak(root):
    run_dirs = _visible_run_dirs(root)
    _require({path.name for path in run_dirs} == set(LIBRARIES), "control root must contain exactly LibA and LibB")
    controls = {}
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        _require(manifest_path.is_file(), "partial control has no manifest: {}".format(run_dir))
        manifest = _read_json(manifest_path)
        _verify_outputs(run_dir, manifest)
        configuration = _validate_common_manifest(manifest, run_dir, expect_control=True)
        _require(configuration.get("libraries") == [run_dir.name], "control library mismatch")
        library = run_dir.name
        weak = pd.read_csv(run_dir / "sklearn_pn_weak_validation.csv", float_precision="round_trip")
        required = {"library", "record_type", "C", "fold", "log_loss", "auroc", "ap"}
        _require(required.issubset(weak.columns), "control weak table lacks columns")
        _require(set(weak["library"].astype(str)) == {library}, "control weak library mismatch")
        expected_keys = set()
        for c_value in C_GRID:
            expected_keys.add(("aggregate", float(c_value), -1))
            for fold in range(FOLDS):
                expected_keys.add(("fold", float(c_value), fold))
        observed = [(str(row.record_type), float(row.C), int(row.fold)) for row in weak.itertuples(index=False)]
        _require(len(observed) == len(set(observed)) and set(observed) == expected_keys, "control weak grid is partial or mixed")
        aggregate = weak.loc[weak["record_type"].eq("aggregate")]
        best_index = min(aggregate.index, key=lambda index: _sklearn_control_key(aggregate.loc[index]))
        best = aggregate.loc[best_index]
        controls[library] = {
            "library": library,
            "C": float(best["C"]),
            "weak_validation_auroc": float(best["auroc"]),
            "weak_validation_ap": float(best["ap"]),
            "weak_validation_log_loss": float(best["log_loss"]),
            "run_name": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "trainer_sha256": _trainer_hash(manifest),
            "manifest_sha256": sha256_file(manifest_path),
            "source_hash_contract": _source_hash_contract(manifest),
            "manifest": manifest,
            "counts": _single_count_contract(manifest, run_dir),
        }
    return controls


def load_historical_controls_weak(paths, current_source_contract):
    """Load the immutable historical PN controls without reading retention yet."""
    controls = {}
    for library in LIBRARIES:
        root = Path(paths[library]).resolve()
        _require(root.is_dir(), "historical control root is missing: {}".format(root))
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        configuration = manifest.get("configuration", {})
        _require(configuration.get("libraries") == [library], "historical control library mismatch")
        _require(
            configuration.get("regularization_selection")
            == "minimum pooled weak-validation log loss; AP/AUROC/smaller-C tie-break",
            "historical control selection rule changed",
        )
        _require(configuration.get("retention_usage") == "identity holdout definition and final retrospective evaluation only", "historical retention-use contract changed")
        for filename in ("weak_validation.csv", "metrics.csv", "predictions.csv"):
            output = manifest.get("outputs", {}).get(filename, {})
            path = root / filename
            _require(path.is_file(), "historical control output is missing: {}".format(path))
            _require(output.get("sha256") == sha256_file(path), "historical control output hash mismatch: {}".format(path))

        cache_manifests = manifest.get("sources", {}).get("cache_manifests", [])
        _require(len(cache_manifests) == 1, "historical control cache provenance is ambiguous")
        historical_cache_hash = str(cache_manifests[0].get("output", {}).get("sha256", ""))
        _require(
            historical_cache_hash in set(current_source_contract["pn_cache_sha256"]),
            "historical control did not use the same frozen-MINT cache",
        )

        weak = pd.read_csv(root / "weak_validation.csv", float_precision="round_trip")
        block = weak.loc[
            weak["representation"].eq("frozen_mint_chain_mean")
            & weak["record_type"].eq("aggregate")
        ].copy()
        _require(len(block) == len(C_GRID), "historical frozen-MINT weak C grid is incomplete")
        _require(set(block["C"].astype(float)) == set(C_GRID), "historical frozen-MINT C grid changed")
        best_index = min(block.index, key=lambda index: _historical_control_key(block.loc[index]))
        best = block.loc[best_index]
        marked = block.loc[block["selected"].astype(int).eq(1)]
        _require(len(marked) == 1 and _close(marked.iloc[0]["C"], best["C"]), "historical selected C does not reproduce")
        controls[library] = {
            "library": library,
            "C": float(best["C"]),
            "weak_validation_auroc": float(best["auroc"]),
            "weak_validation_ap": float(best["auprc"]),
            "weak_validation_log_loss": float(best["log_loss"]),
            "run_name": root.name,
            "run_dir": str(root),
            "manifest_sha256": sha256_file(manifest_path),
            "cache_sha256": historical_cache_hash,
            "manifest": manifest,
        }
    return controls


def best_retrospective_f1(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "retrospective threshold requires both classes")
    candidates = []
    for threshold in np.unique(scores):
        predicted = scores >= float(threshold)
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum((~predicted) & (labels == 1)))
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidates.append((f1, precision, recall, float(threshold), tp, fp, fn))
    chosen = max(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
    return {
        "retrospective_f1_threshold": chosen[3],
        "retrospective_f1_precision": chosen[1],
        "retrospective_f1_recall": chosen[2],
        "retrospective_f1": chosen[0],
        "retrospective_f1_tp": chosen[4],
        "retrospective_f1_fp": chosen[5],
        "retrospective_f1_fn": chosen[6],
        "retrospective_f1_selected": chosen[4] + chosen[5],
    }


def retention_metrics(predictions):
    required = {"chain1_sha256", "target_retention", "target_binder", "score"}
    _require(required.issubset(predictions.columns), "retention predictions lack required columns")
    observed = pd.to_numeric(predictions["target_retention"], errors="raise").to_numpy(dtype=float)
    labels = pd.to_numeric(predictions["target_binder"], errors="raise").astype(int).to_numpy()
    scores = pd.to_numeric(predictions["score"], errors="raise").to_numpy(dtype=float)
    _require(bool(np.isfinite(observed).all() and np.isfinite(scores).all()), "non-finite retention values or scores")
    _require(np.array_equal(labels, (observed >= 75.0).astype(int)), "stored binder labels do not equal retention >=75%")
    within = []
    for _, indices in predictions.groupby("chain1_sha256", sort=True).indices.items():
        index = np.asarray(indices, dtype=int)
        if len(index) >= 2 and np.unique(observed[index]).size >= 2 and np.unique(scores[index]).size >= 2:
            within.append(float(spearmanr(observed[index], scores[index])[0]))
    output = {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "global_spearman": float(spearmanr(observed, scores)[0]),
        "average_within_peptide_spearman": float(np.mean(within)) if within else float("nan"),
        "within_peptide_groups": int(len(within)),
    }
    output.update(best_retrospective_f1(labels, scores))
    return output


def _check_recorded_metrics(recorded, recomputed, label):
    _require(len(recorded) == 1, "retention metric row is not unique for {}".format(label))
    row = recorded.iloc[0]
    for key, value in recomputed.items():
        _require(key in row.index, "retention metric {} is absent for {}".format(key, label))
        _require(_close(row[key], value, rel_tol=1e-9, abs_tol=1e-11), "retention metric {} does not reproduce for {}".format(key, label))


def _evaluation_identity(frame):
    columns = (
        "library", "pair_uid", "peptide_design_code", "affibody_design_code",
        "chain1_sha256", "chain2_sha256", "target_retention", "target_binder",
    )
    _require(set(columns).issubset(frame.columns), "prediction table lacks evaluation identity columns")
    return frame.loc[:, list(columns)].sort_values("pair_uid", kind="mergesort").reset_index(drop=True)


def load_selected_retention(selected, controls, historical_controls):
    """Open retention artifacts only after all weak-label choices are frozen."""
    metric_rows = []
    prediction_blocks = []
    evaluation_contract = {}

    for library in LIBRARIES:
        control = controls[library]
        run_dir = Path(control["run_dir"])
        stored_metrics = pd.read_csv(run_dir / "retention_metrics.csv", float_precision="round_trip")
        all_predictions = pd.read_csv(run_dir / "retention_predictions.csv", float_precision="round_trip")
        arm = "balanced_sklearn_pn_control"
        predictions = all_predictions.loc[all_predictions["arm"].eq(arm)].copy()
        recorded = stored_metrics.loc[stored_metrics["arm"].eq(arm)]
        _require(len(predictions) == control["counts"]["retention_evaluation"], "control prediction count mismatch")
        _require(predictions["pair_uid"].is_unique, "duplicate control prediction pair")
        _require(np.allclose(predictions["C"].astype(float), control["C"]), "control prediction C mismatch")
        recomputed = retention_metrics(predictions)
        _check_recorded_metrics(recorded, recomputed, "{} control".format(library))
        evaluation_contract[library] = _evaluation_identity(predictions)
        metric_rows.append(
            {
                "library": library,
                "model": "Frozen MINT PN AUROC-selected rebaseline",
                "training_method": "balanced P versus N",
                "pi": np.nan,
                "eta": np.nan,
                "C": control["C"],
                "epoch": 0,
                "weak_validation_auroc": control["weak_validation_auroc"],
                "weak_validation_ap": control["weak_validation_ap"],
                "weak_validation_log_loss": control["weak_validation_log_loss"],
                **recomputed
            }
        )
        block = predictions.copy()
        block.insert(0, "model", "Frozen MINT PN AUROC-selected rebaseline")
        block["training_method"] = "balanced P versus N"
        block["selected_pi_sensitivity"] = 0
        prediction_blocks.append(block)

    # Immutable historical frozen-MINT control.  This is deliberately kept
    # separate from the current-code AUROC-selected rebaseline above: the
    # historical pipeline selected C by weak log loss and has its own
    # manifest-tracked numeric implementation.
    for library in LIBRARIES:
        control = historical_controls[library]
        run_dir = Path(control["run_dir"])
        raw_predictions = pd.read_csv(run_dir / "predictions.csv", float_precision="round_trip")
        predictions = raw_predictions.loc[
            raw_predictions["representation"].eq("frozen_mint_chain_mean")
            & raw_predictions["matches_actual_fit_exposure"].astype(int).eq(1)
        ].copy()
        _require(len(predictions) == len(evaluation_contract[library]), "historical exact-control prediction count mismatch")
        _require(predictions["pair_uid"].is_unique, "duplicate historical exact-control prediction pair")
        _require(np.allclose(predictions["selected_C"].astype(float), control["C"]), "historical exact-control prediction C mismatch")
        design = evaluation_contract[library][
            ["pair_uid", "peptide_design_code", "affibody_design_code"]
        ]
        predictions = predictions.merge(design, on="pair_uid", how="inner", validate="one_to_one")
        predictions["score"] = predictions["binder_probability"].astype(float)
        _require(_evaluation_identity(predictions).equals(evaluation_contract[library]), "historical exact-control evaluation panel differs")
        recomputed = retention_metrics(predictions)

        stored = pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")
        recorded = stored.loc[
            stored["representation"].eq("frozen_mint_chain_mean")
            & stored["evaluation_scope"].eq("actual_fit_compatible")
        ]
        _require(len(recorded) == 1, "historical exact-control metric row is not unique")
        row = recorded.iloc[0]
        _require(_close(row["selected_C"], control["C"]), "historical exact-control metric C mismatch")
        mapping = {
            "n": "n",
            "positive": "positive",
            "auroc": "global_auroc",
            "ap": "global_auprc",
            "global_spearman": "global_spearman",
            "average_within_peptide_spearman": "within_peptide_macro_spearman",
            "within_peptide_groups": "within_peptide_evaluable_spearman_groups",
        }
        for current_name, historical_name in mapping.items():
            _require(
                _close(recomputed[current_name], row[historical_name], rel_tol=1e-9, abs_tol=1e-11),
                "historical exact-control metric {} does not reproduce".format(current_name),
            )
        metric_rows.append(
            {
                "library": library,
                "model": "Frozen MINT PN historical exact",
                "training_method": "historical balanced P versus N; weak-logloss-selected C",
                "pi": np.nan,
                "eta": np.nan,
                "C": control["C"],
                "epoch": 0,
                "weak_validation_auroc": control["weak_validation_auroc"],
                "weak_validation_ap": control["weak_validation_ap"],
                "weak_validation_log_loss": control["weak_validation_log_loss"],
                **recomputed
            }
        )
        predictions.insert(0, "model", "Frozen MINT PN historical exact")
        predictions["training_method"] = "historical balanced P versus N; weak-logloss-selected C"
        predictions["selected_pi_sensitivity"] = 0
        prediction_blocks.append(predictions)

    for _, choice in selected.sort_values(["library", "pi"]).iterrows():
        run_dir = Path(choice["run_dir"])
        stored_metrics = pd.read_csv(run_dir / "retention_metrics.csv", float_precision="round_trip")
        predictions = pd.read_csv(run_dir / "retention_predictions.csv", float_precision="round_trip")
        matching_predictions = predictions.loc[
            np.isclose(predictions["pi"].astype(float), float(choice["pi"]))
            & np.isclose(predictions["eta"].astype(float), float(choice["eta"]))
            & np.isclose(predictions["C"].astype(float), float(choice["C"]))
            & predictions["epoch"].astype(int).eq(int(choice["epoch"]))
        ].copy()
        matching_metrics = stored_metrics.loc[
            np.isclose(stored_metrics["pi"].astype(float), float(choice["pi"]))
            & np.isclose(stored_metrics["eta"].astype(float), float(choice["eta"]))
            & np.isclose(stored_metrics["C"].astype(float), float(choice["C"]))
            & stored_metrics["epoch"].astype(int).eq(int(choice["epoch"]))
        ]
        library = str(choice["library"])
        _require(len(matching_predictions) == choice["counts"]["retention_evaluation"], "selected PNU prediction count mismatch")
        _require(matching_predictions["pair_uid"].is_unique, "duplicate selected PNU prediction pair")
        _require(_evaluation_identity(matching_predictions).equals(evaluation_contract[library]), "evaluation panel differs between PNU and exact control")
        recomputed = retention_metrics(matching_predictions)
        _check_recorded_metrics(matching_metrics, recomputed, "{}/pi{}".format(library, choice["pi"]))
        eta = float(choice["eta"])
        if eta == 0.0:
            method = "prior-weighted PN (selected eta=0; U has zero loss weight)"
        elif eta == 1.0:
            method = "non-negative PU (selected eta=1)"
        else:
            method = "non-negative PNU with sampled U"
        metric_rows.append(
            {
                "library": library,
                "model": "Frozen MINT PNU sensitivity",
                "training_method": method,
                "pi": float(choice["pi"]),
                "eta": eta,
                "C": float(choice["C"]),
                "epoch": int(choice["epoch"]),
                "weak_validation_auroc": float(choice["weak_validation_auroc"]),
                "weak_validation_ap": float(choice["weak_validation_ap"]),
                "weak_validation_log_loss": float(choice["weak_validation_log_loss"]),
                **recomputed
            }
        )
        matching_predictions.insert(0, "model", "Frozen MINT PNU sensitivity")
        matching_predictions["training_method"] = method
        matching_predictions["selected_pi_sensitivity"] = 1
        prediction_blocks.append(matching_predictions)
    return pd.DataFrame(metric_rows), pd.concat(prediction_blocks, ignore_index=True, sort=False)


def _display_number(value, digits=3):
    if pd.isna(value):
        return "—"
    return ("{:." + str(digits) + "f}").format(float(value))


def render_report(metrics, candidates, extension_audit, counts, u_audit):
    lines = [
        "# Frozen MINT P/N/U pilot",
        "",
        "This experiment tests whether below-cutoff R9/R10 pairs add useful training "
        "information to the frozen layer-33 MINT representation. The MINT backbone is "
        "unchanged; each chain is mean-pooled, the two 1,280-dimensional vectors are "
        "concatenated, and one linear classification head is trained.",
        "",
        "## Data and comparison",
        "",
        "P contains the established top-count R9/R10 positives, N contains the "
        "established conservative negatives, and U contains R9/R10 pairs below the "
        "positive cutoff. This pilot embeds one deterministic sample (seed {}) of five U "
        "pairs per P pair; it is not a full-U experiment.".format(u_audit["sample_seed"]),
        "",
        "| Library | P | N | sampled U | all eligible U | sampled coverage | direct-retention pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for library in LIBRARIES:
        row = counts[library]
        lines.append(
            "| {} | {:,} | {:,} | {:,} | {:,} | {:.2%} | {:,} |".format(
                library, row["positive"], row["negative"], row["unlabeled_used"],
                u_audit["libraries"][library]["eligible_u"],
                u_audit["libraries"][library]["coverage_fraction"],
                row["retention_evaluation"],
            )
        )
    lines.extend(
        [
            "",
            "The historical exact PN row is imported directly from the manifest-tracked "
            "frozen-MINT result used in the earlier public analysis; that pipeline selected "
            "regularization by weak log loss. A second PN row refits the same architecture "
            "and data with the current code and selects regularization by weak AUROC, "
            "matching the PNU selector. It is labeled as a rebaseline rather than an exact "
            "reproduction. For PNU, the assumed hidden-positive "
            "fraction pi is treated as a sensitivity setting. All four pi values are "
            "reported rather than selecting one from retention performance.",
            "",
            "For every fixed library and pi, eta, regularization C, and epoch were chosen "
            "only by weak-validation AUROC, then AP, then log loss, then smaller eta, C, "
            "and epoch. Retention measurements were opened only after these choices.",
            "",
            "## Retrospective direct-retention evaluation",
            "",
            "A positive means measured retention of at least 75%. The threshold precision "
            "and recall use the F1-maximizing threshold on this same evaluation panel; "
            "they describe the retrospective data and are not a deployable cutoff.",
            "",
            "| Library | Training | pi | eta | AUROC | AP | Global Spearman | Within-peptide Spearman | Threshold precision | Threshold recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display = metrics.copy()
    display["order"] = display["model"].map({
        "Frozen MINT PN historical exact": 0,
        "Frozen MINT PN AUROC-selected rebaseline": 1,
        "Frozen MINT PNU sensitivity": 2,
    })
    display = display.sort_values(["library", "order", "pi"], na_position="first")
    for _, row in display.iterrows():
        if row["model"] == "Frozen MINT PN historical exact":
            label = "PN historical exact"
        elif row["model"] == "Frozen MINT PN AUROC-selected rebaseline":
            label = "PN AUROC rebaseline"
        else:
            label = "PNU sensitivity"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["library"], label,
                "—" if pd.isna(row["pi"]) else "{:.0%}".format(float(row["pi"])),
                _display_number(row["eta"], 2), _display_number(row["auroc"]),
                _display_number(row["ap"]), _display_number(row["global_spearman"]),
                _display_number(row["average_within_peptide_spearman"]),
                _display_number(row["retrospective_f1_precision"]),
                _display_number(row["retrospective_f1_recall"]),
            )
        )
    used = extension_audit.loc[extension_audit["extension_used"].eq(1)]
    selected_eta_zero = int(candidates.loc[candidates["selected_eta_by_aggregate"].eq(1), "eta"].eq(0.0).sum())
    lines.extend(
        [
            "",
            "## Result",
            "",
            "Sampled U did not improve this frozen layer-33 linear readout. LibA selected "
            "eta=0 at every assumed pi, so weak validation rejected any contribution from "
            "U. LibB selected eta=0 at pi=2% and 5%; at pi=10% and 15% it selected "
            "eta=0.25, but both direct-retention rankings were worse than both the historical "
            "exact PN control and the AUROC-selected PN rebaseline. This is a negative "
            "result for this one sampled-U, one-seed setup, "
            "not evidence that PU/PNU can never help.",
            "",
            "## Audit notes",
            "",
            "{} eta=0 boundary arms were extended from 40 to 80 epochs after their "
            "40-epoch histories reproduced exactly. Among the eight fixed-library/pi "
            "sensitivity rows, eta=0 was selected {} times; in those rows U contributes "
            "zero loss weight, so they are prior-weighted PN fits rather than evidence "
            "that unlabeled data helped.".format(len(used), selected_eta_zero),
            "",
            "The usual PU/PNU SCAR assumption is not satisfied cleanly here. P is the "
            "high-count tail, not a random labeled sample of every true binder, while N "
            "is also a deliberately conservative and potentially easier negative set. "
            "The unknown positive fraction and the sampled-U results should therefore be "
            "read as sensitivity analyses, and their scores are not calibrated retention "
            "percentages or binding probabilities. Only one training seed and one "
            "deterministic U sample were tested.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-root", type=Path,
        default=REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid40_v1",
    )
    parser.add_argument(
        "--extension-root", type=Path,
        default=REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid80_extension_v1",
    )
    parser.add_argument(
        "--control-root", type=Path,
        default=REPO_ROOT / "private_data/experiments/pnu_mint_layer33_controls_v1",
    )
    parser.add_argument(
        "--historical-control-liba", type=Path,
        default=REPO_ROOT / "private_data/experiments/mint_cached_primary_practical_liba_double_cold_v1",
    )
    parser.add_argument(
        "--historical-control-libb", type=Path,
        default=REPO_ROOT / "private_data/experiments/mint_cached_primary_practical_libb_double_cold_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    grid_root = Path(args.grid_root).resolve()
    extension_root = Path(args.extension_root).resolve()
    control_root = Path(args.control_root).resolve()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")

    base = load_complete_base_grid(grid_root)
    combined, extension_audit, extensions = integrate_boundary_extensions(base, extension_root)
    controls = load_exact_controls_weak(control_root)
    all_records = list(base.values()) + list(extensions.values()) + list(controls.values())
    trainer_hash, source_contract = _verify_uniform_contract(all_records)
    current_trainer = REPO_ROOT / "downstream/AffibodyMHC/train_pnu_mint_readout.py"
    _require(sha256_file(current_trainer) == trainer_hash, "current trainer source differs from every recorded run")
    source_files = verify_source_files(all_records[0]["manifest"])
    u_audit = sampled_u_audit(all_records[0]["manifest"])
    historical_paths = {
        "LibA": Path(args.historical_control_liba).resolve(),
        "LibB": Path(args.historical_control_libb).resolve(),
    }
    historical_controls = load_historical_controls_weak(
        historical_paths, source_contract
    )

    count_contract = {}
    for record in all_records:
        library = record["library"]
        if library in count_contract:
            _require(record["counts"] == count_contract[library], "count contract changed for {}".format(library))
        else:
            count_contract[library] = record["counts"]

    candidate_rows = []
    for key in sorted(combined):
        arm = combined[key]
        candidate_rows.append(
            {name: arm[name] for name in (
                "library", "pi", "eta", "C", "epoch", "weak_validation_auroc",
                "weak_validation_ap", "weak_validation_log_loss", "run_name",
                "run_dir", "max_epoch", "trainer_sha256", "manifest_sha256",
            )}
        )
    candidates, selected_table = select_eta_per_fixed_pi(pd.DataFrame(candidate_rows))
    selected_records = []
    for _, row in selected_table.iterrows():
        key = (str(row["library"]), float(row["pi"]), float(row["eta"]))
        selected_records.append(combined[key])
    selected = pd.DataFrame(selected_records)
    # Replace nested audit objects with the flat, independently selected table
    # while retaining count/run metadata needed for delayed retention loading.
    selected = selected.merge(
        selected_table[["library", "pi", "eta", "selected_eta_by_aggregate"]],
        on=["library", "pi", "eta"], how="inner", validate="one_to_one",
    )

    # No numeric retention value has been read before this call.
    metrics, predictions = load_selected_retention(
        selected, controls, historical_controls
    )
    report = render_report(
        metrics, candidates, extension_audit, count_contract, u_audit
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    paths = {
        "weak_arm_candidates.csv": output_dir / "weak_arm_candidates.csv",
        "selected_weak_models.csv": output_dir / "selected_weak_models.csv",
        "epoch_extension_audit.csv": output_dir / "epoch_extension_audit.csv",
        "retention_metrics.csv": output_dir / "retention_metrics.csv",
        "retention_predictions.csv": output_dir / "retention_predictions.csv",
        "frozen_mint_pnu_report.md": output_dir / "frozen_mint_pnu_report.md",
    }
    public_selected_columns = [
        "library", "pi", "eta", "C", "epoch", "weak_validation_auroc",
        "weak_validation_ap", "weak_validation_log_loss", "run_name", "run_dir",
        "max_epoch", "trainer_sha256", "manifest_sha256", "selected_eta_by_aggregate",
    ]
    _write_csv(candidates, paths["weak_arm_candidates.csv"])
    _write_csv(selected.loc[:, public_selected_columns], paths["selected_weak_models.csv"])
    _write_csv(extension_audit, paths["epoch_extension_audit.csv"])
    _write_csv(metrics, paths["retention_metrics.csv"])
    _write_csv(predictions, paths["retention_predictions.csv"])
    _atomic_text(report, paths["frozen_mint_pnu_report.md"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "created_unix": time.time(),
        "runtime_seconds": float(time.time() - started),
        "environment": {"hostname": platform.node(), "python": platform.python_version()},
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "selection_contract": {
            "scope": "one eta/C/epoch per fixed library and pi",
            "order": "weak AUROC desc, weak AP desc, weak log loss asc, eta asc, C asc, epoch asc",
            "class_prior_selection": "none; all pi=0.02/0.05/0.10/0.15 retained",
            "retention_usage": "final retrospective evaluation only",
        },
        "feature_contract": {
            "backbone": "frozen MINT",
            "layer": 33,
            "pooling": "separate whole-chain means concatenated",
            "dimension": 2560,
            "readout": "one linear logit (2561 trainable parameters)",
        },
        "u_contract": {
            "type": "deterministic sampled U",
            "u_per_positive": U_PER_POSITIVE,
            "full_u": False,
            "sample_seed": u_audit["sample_seed"],
            "coverage": u_audit["libraries"],
            "prepare_manifest": u_audit["prepare_manifest"],
            "eligible_pool_manifest": u_audit["eligible_pool_manifest"],
            "scar_warning": "top-count P is not a random labeled sample of all true positives",
        },
        "trainer_sha256": trainer_hash,
        "source_hash_contract": source_contract,
        "verified_source_files": source_files,
        "input_roots": {
            "grid40": str(grid_root), "grid80_extensions": str(extension_root),
            "auroc_selected_pn_rebaseline": str(control_root),
            "historical_exact_pn_liba": str(historical_paths["LibA"]),
            "historical_exact_pn_libb": str(historical_paths["LibB"]),
        },
        "input_manifests": [
            {"run_dir": record["run_dir"], "manifest_sha256": record["manifest_sha256"]}
            for record in sorted(all_records, key=lambda value: value["run_dir"])
        ],
        "historical_control_manifests": {
            library: {
                "run_dir": control["run_dir"],
                "manifest_sha256": control["manifest_sha256"],
                "selected_C": control["C"],
                "selection_rule": "weak log loss, AP, AUROC, smaller C",
            }
            for library, control in sorted(historical_controls.items())
        },
        "counts": count_contract,
        "rows": {
            "weak_arm_candidates": int(len(candidates)),
            "selected_pnu_sensitivity_models": int(len(selected)),
            "retention_metric_rows": int(len(metrics)),
            "retention_prediction_rows": int(len(predictions)),
        },
        "outputs": {},
    }
    for name, path in paths.items():
        manifest["outputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    _write_json(manifest, output_dir / "manifest.json")
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_pnu_sensitivity_models": int(len(selected)),
        "retention_metric_rows": int(len(metrics)),
        "all_priors_retained": list(PRIORS),
    }, indent=2, sort_keys=True))
    return metrics


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
