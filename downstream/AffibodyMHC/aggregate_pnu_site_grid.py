#!/usr/bin/env python
"""Fail-closed aggregation for the sharded designed-position P/NU grid.

Each shard is produced by ``train_pnu_site.py`` and contains one library and
one prespecified risk/prior/eta arm, with C and epoch selected using only weak
validation labels.  This aggregator:

* verifies the shard manifests and recorded output hashes;
* independently reproduces every within-configuration epoch choice;
* independently selects C, and (for nnPNU) eta, by weak AUROC, weak AP,
  weak log loss, then smaller eta/C/epoch;
* keeps all four class-prior values as separate sensitivity analyses;
* adds the established sklearn PN result as the primary PN control and keeps
  the new Adam PN fit explicitly labelled as a diagnostic;
* copies retention results only after all weak-label choices are complete.

The script deliberately has no rule for choosing a class prior from retention
performance.  It fails before creating the output directory if any expected
shard is absent or inconsistent.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
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


SCHEMA_VERSION = "affibody-pnu-site-grid-aggregate-v1"
LIBRARIES = ("LibA", "LibB")
PRIORS = (0.02, 0.05, 0.10, 0.15)
ETAS = (0.0, 0.25, 0.50, 0.75, 1.0)
REQUIRED_SHARD_OUTPUTS = (
    "config_selection.csv",
    "validation_history.csv",
    "retention_metrics.csv",
    "retention_predictions.csv",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    value = float(value)
    _require(math.isfinite(value), "{} is not finite".format(label))
    return value


def _read_json(path):
    with open(str(path), "r") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "JSON root is not an object: {}".format(path))
    return payload


def _atomic_text(text, path):
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    with open(str(temporary), "w") as handle:
        handle.write(text)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(path))


def _write_json(payload, path):
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _write_csv(frame, path):
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    frame.to_csv(str(temporary), index=False)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(path))


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _prior_key(value):
    if pd.isna(value):
        return "none"
    return "{:.8g}".format(float(value))


def weak_selection_key(row):
    """Prespecified ranking key; lower tuples are better.

    Retention columns are intentionally neither read nor accepted as key
    components.  ``eta`` is the first deterministic tie breaker after the
    three weak-validation metrics, followed by C and epoch.
    """
    eta = -1.0 if pd.isna(row.get("eta", np.nan)) else _finite(row["eta"], "eta")
    return (
        -_finite(row["validation_auroc"], "validation AUROC"),
        -_finite(row["validation_average_precision"], "validation AP"),
        _finite(row["validation_log_loss"], "validation log loss"),
        eta,
        _finite(row["C"], "C"),
        int(row["selected_epoch"]),
        str(row["config_id"]),
    )


def epoch_selection_key(row):
    """Weak-only key used to reproduce epoch selection inside one config."""
    return (
        -_finite(row["auroc"], "epoch AUROC"),
        -_finite(row["average_precision"], "epoch AP"),
        _finite(row["log_loss"], "epoch log loss"),
        int(row["epoch"]),
    )


def select_prespecified_configs(configs):
    """Select one config for PN and for every fixed (risk, library, pi).

    In particular, this returns four nnPU and four nnPNU rows per library;
    class priors are never compared with one another.
    """
    configs = configs.copy()
    required = {
        "library",
        "risk",
        "class_prior",
        "eta",
        "C",
        "selected_epoch",
        "validation_auroc",
        "validation_average_precision",
        "validation_log_loss",
        "config_id",
    }
    _require(required.issubset(configs.columns), "config table lacks required columns")
    configs["prior_key"] = configs["class_prior"].map(_prior_key)
    configs["selected_by_aggregate"] = 0
    selected_indices = []
    for group_key, indices in configs.groupby(
        ["library", "risk", "prior_key"], sort=True, dropna=False
    ).groups.items():
        risk = group_key[1]
        if risk == "pn":
            _require(group_key[2] == "none", "PN row unexpectedly has a class prior")
        else:
            _require(group_key[2] != "none", "{} row lacks class prior".format(risk))
        best = min(list(indices), key=lambda index: weak_selection_key(configs.loc[index]))
        selected_indices.append(best)
        configs.loc[best, "selected_by_aggregate"] = 1
    selected = configs.loc[selected_indices].copy()
    selected = selected.sort_values(
        ["library", "risk", "class_prior"], na_position="first"
    ).reset_index(drop=True)
    return configs.drop(columns=["prior_key"]), selected.drop(columns=["prior_key"])


def audit_epoch_choices(configs, histories, max_epochs_by_run):
    """Reproduce every selected epoch and record lower/upper-boundary hits."""
    audit_rows = []
    configs = configs.copy()
    histories = histories.copy()
    for _, config in configs.iterrows():
        run_name = str(config["run_name"])
        config_id = str(config["config_id"])
        block = histories.loc[
            histories["run_name"].eq(run_name)
            & histories["config_id"].eq(config_id)
            & histories["record_type"].eq("aggregate")
        ].copy()
        _require(len(block) > 0, "no aggregate epoch history for {}/{}".format(run_name, config_id))
        _require(block["epoch"].astype(int).is_unique, "duplicate aggregate epochs for {}/{}".format(run_name, config_id))
        recomputed = min(block.index, key=lambda index: epoch_selection_key(block.loc[index]))
        best = block.loc[recomputed]
        recorded_epoch = int(config["selected_epoch"])
        _require(
            recorded_epoch == int(best["epoch"]),
            "recorded epoch does not reproduce under weak-only rule for {}/{}: {} != {}".format(
                run_name, config_id, recorded_epoch, int(best["epoch"])
            ),
        )
        for config_column, history_column in (
            ("validation_auroc", "auroc"),
            ("validation_average_precision", "average_precision"),
            ("validation_log_loss", "log_loss"),
        ):
            _require(
                math.isclose(float(config[config_column]), float(best[history_column]), rel_tol=1e-10, abs_tol=1e-12),
                "stored weak metric does not match selected epoch for {}/{}".format(run_name, config_id),
            )
        markers = block.loc[block["epoch_selected_within_config"].astype(int).eq(1)]
        _require(len(markers) == 1, "epoch marker is not unique for {}/{}".format(run_name, config_id))
        _require(int(markers.iloc[0]["epoch"]) == recorded_epoch, "epoch marker disagrees for {}/{}".format(run_name, config_id))
        max_epochs = int(max_epochs_by_run[run_name])
        _require(int(block["epoch"].max()) == max_epochs, "epoch history is incomplete for {}".format(run_name))
        if recorded_epoch == 1:
            boundary = "first_epoch"
        elif recorded_epoch == max_epochs:
            boundary = "last_epoch"
        else:
            boundary = "interior"
        audit_rows.append(
            {
                "library": config["library"],
                "run_name": run_name,
                "config_id": config_id,
                "risk": config["risk"],
                "class_prior": config["class_prior"],
                "eta": config["eta"],
                "C": config["C"],
                "selected_epoch": recorded_epoch,
                "max_epochs": max_epochs,
                "epoch_location": boundary,
                "at_epoch_boundary": int(boundary != "interior"),
                "epoch_selection_reproduced": 1,
            }
        )
    return pd.DataFrame(audit_rows)


def _retention_metrics(predictions, score_column):
    predictions = predictions.reset_index(drop=True)
    observed = predictions["target_retention"].to_numpy(dtype=float)
    binder = predictions["target_binder"].to_numpy(dtype=int)
    score = predictions[score_column].to_numpy(dtype=float)
    within = []
    for _, indices in predictions.groupby("peptide_uid", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        if np.unique(observed[indices]).size < 2 or np.unique(score[indices]).size < 2:
            continue
        within.append(float(spearmanr(observed[indices], score[indices])[0]))
    return {
        "n": int(len(predictions)),
        "binders_ge_75": int(binder.sum()),
        "auroc": float(roc_auc_score(binder, score)),
        "average_precision": float(average_precision_score(binder, score)),
        "global_spearman": float(spearmanr(observed, score)[0]),
        "within_peptide_spearman": float(np.mean(within)) if within else float("nan"),
        "within_peptide_groups": int(len(within)),
    }


def _verify_manifest_output(run_dir, manifest, filename):
    _require(filename in manifest.get("outputs", {}), "manifest omits {} in {}".format(filename, run_dir))
    path = run_dir / filename
    _require(path.is_file(), "missing shard output: {}".format(path))
    recorded = manifest["outputs"][filename].get("sha256")
    _require(recorded == sha256_file(path), "shard output hash mismatch: {}".format(path))
    return path


def trainer_code_hash(manifest):
    """Return the uniquely recorded train_pnu_site.py hash."""
    matches = [
        value for path, value in manifest.get("code", {}).items()
        if Path(path).name == "train_pnu_site.py"
    ]
    _require(len(matches) == 1, "manifest must record exactly one train_pnu_site.py hash")
    _require(isinstance(matches[0], str) and len(matches[0]) == 64, "invalid trainer code hash")
    return matches[0]


def require_uniform_trainer_hash(manifests):
    """Reject a grid assembled across trainer revisions."""
    hashes = {name: trainer_code_hash(payload) for name, payload in manifests.items()}
    unique = set(hashes.values())
    _require(
        len(unique) == 1,
        "mixed train_pnu_site.py hashes in one grid: {}".format(
            {name: value[:12] for name, value in sorted(hashes.items())}
        ),
    )
    return next(iter(unique))


def validate_training_grid_configuration(
    configuration,
    expected_folds=None,
    expected_c_grid=None,
    expected_max_epochs=None,
    run_name="run",
):
    """Enforce the prespecified validation-fold and C-grid contract."""
    if expected_folds is not None:
        _require(
            int(configuration.get("folds", -1)) == int(expected_folds),
            "fold count differs in {}: expected {}, observed {}".format(
                run_name, int(expected_folds), configuration.get("folds")
            ),
        )
    if expected_c_grid is not None:
        expected = sorted(set(float(value) for value in expected_c_grid))
        observed = sorted(set(float(value) for value in configuration.get("c_grid", [])))
        _require(
            len(expected) == len(observed)
            and np.allclose(expected, observed, rtol=0.0, atol=1e-12),
            "C grid differs in {}: expected {}, observed {}".format(
                run_name, expected, observed
            ),
        )
    if expected_max_epochs is not None:
        _require(
            int(configuration.get("max_epochs", -1)) == int(expected_max_epochs),
            "maximum epoch count differs in {}: expected {}, observed {}".format(
                run_name, int(expected_max_epochs), configuration.get("max_epochs")
            ),
        )


def _load_shards(
    input_root,
    expected_risks=("nnpu", "nnpnu"),
    run_names=None,
    expected_folds=None,
    expected_c_grid=None,
    expected_max_epochs=None,
    expected_libraries=LIBRARIES,
):
    expected_risks = tuple(expected_risks)
    expected_libraries = tuple(expected_libraries)
    _require(
        len(expected_libraries) > 0 and set(expected_libraries).issubset(set(LIBRARIES)),
        "invalid expected libraries",
    )
    _require(set(expected_risks).issubset({"pn", "nnpu", "nnpnu"}), "invalid expected risks")
    if run_names is None:
        run_dirs = sorted(
            path for path in input_root.iterdir()
            if path.is_dir()
            and path.name not in ("logs", "aggregate")
            and (path / "manifest.json").is_file()
        )
    else:
        run_dirs = [input_root / name for name in run_names]
        _require(all((path / "manifest.json").is_file() for path in run_dirs), "a requested shard is absent")
    _require(len(run_dirs) > 0, "no completed PNU shard directories found")
    # Check code provenance before opening any result table. This makes a
    # mixed-revision grid fail for the scientifically relevant reason even if
    # one revision also changed its table schema.
    preloaded_manifests = {
        path.name: _read_json(path / "manifest.json") for path in run_dirs
    }
    trainer_hash = require_uniform_trainer_hash(preloaded_manifests)
    configs_blocks = []
    history_blocks = []
    retention_by_run = {}
    predictions_by_run = {}
    manifests = {}
    max_epochs_by_run = {}
    source_contract = None
    observed_arms = set()
    row_contract = {}
    for run_dir in run_dirs:
        run_name = run_dir.name
        manifest = preloaded_manifests[run_name]
        for filename in REQUIRED_SHARD_OUTPUTS:
            _verify_manifest_output(run_dir, manifest, filename)
        library = manifest.get("library")
        _require(library in expected_libraries, "invalid or unexpected library in {}".format(run_name))
        configuration = manifest.get("configuration", {})
        validate_training_grid_configuration(
            configuration,
            expected_folds=expected_folds,
            expected_c_grid=expected_c_grid,
            expected_max_epochs=expected_max_epochs,
            run_name=run_name,
        )
        _require(configuration.get("selection_metric", "auroc") == "auroc", "shard was not selected by weak AUROC: {}".format(run_name))
        _require(configuration.get("retention_usage") == "final retrospective evaluation only", "retention-use contract differs in {}".format(run_name))
        _require(configuration.get("risks") and len(configuration["risks"]) == 1, "shard must contain one risk arm: {}".format(run_name))
        risk = configuration["risks"][0]
        _require(risk in ("pn", "nnpu", "nnpnu"), "invalid risk in {}".format(run_name))
        _require(risk in expected_risks, "unexpected risk {} in {}".format(risk, run_name))
        priors = configuration.get("class_prior_grid", [])
        etas = configuration.get("eta_grid", [])
        if risk == "pn":
            _require(len(priors) == 0 and len(etas) == 0, "PN shard has pi/eta grid")
            arm = (library, risk, None, None)
        elif risk == "nnpu":
            _require(len(priors) == 1 and len(etas) == 0, "nnPU shard is not one fixed-pi arm")
            arm = (library, risk, float(priors[0]), None)
        else:
            _require(len(priors) == 1 and len(etas) == 1, "nnPNU shard is not one fixed-(pi,eta) arm")
            arm = (library, risk, float(priors[0]), float(etas[0]))
        _require(arm not in observed_arms, "duplicate shard arm: {}".format(arm))
        observed_arms.add(arm)
        current_source = {
            name: payload["sha256"]
            for name, payload in manifest.get("sources", {}).items()
            if isinstance(payload, dict) and "sha256" in payload
        }
        if source_contract is None:
            source_contract = current_source
        else:
            _require(current_source == source_contract, "source hashes differ in {}".format(run_name))
        rows = manifest.get("rows", {})
        current_rows = tuple(int(rows[name]) for name in ("positive", "negative", "unlabeled", "retention"))
        if library in row_contract:
            _require(current_rows == row_contract[library], "row counts differ within {}".format(library))
        else:
            row_contract[library] = current_rows

        configs = pd.read_csv(run_dir / "config_selection.csv", float_precision="round_trip")
        histories = pd.read_csv(run_dir / "validation_history.csv", float_precision="round_trip")
        metrics = pd.read_csv(run_dir / "retention_metrics.csv", float_precision="round_trip")
        predictions = pd.read_csv(run_dir / "retention_predictions.csv", float_precision="round_trip")
        # The trainer's compact configuration table is already scoped to one
        # library and therefore omits that repeated column. Restore it from
        # the manifest before cross-shard aggregation.
        if "library" not in configs.columns:
            configs.insert(0, "library", library)
        for frame in (configs, histories, metrics, predictions):
            frame.insert(0, "run_name", run_name)
        _require(set(configs["library"]) == {library}, "config library mismatch in {}".format(run_name))
        _require(set(configs["risk"]) == {risk}, "config risk mismatch in {}".format(run_name))
        _require(len(metrics) == 1, "shard retention metrics must have one row: {}".format(run_name))
        _require(metrics.iloc[0]["selection_basis"].startswith("weak_double_identity_cold_"), "retention selection basis is not weak-only in {}".format(run_name))
        configs_blocks.append(configs)
        history_blocks.append(histories)
        retention_by_run[run_name] = metrics
        predictions_by_run[run_name] = predictions
        manifests[run_name] = manifest
        max_epochs_by_run[run_name] = int(configuration["max_epochs"])

    expected = set()
    for library in expected_libraries:
        if "pn" in expected_risks:
            expected.add((library, "pn", None, None))
        if "nnpu" in expected_risks:
            for prior in PRIORS:
                expected.add((library, "nnpu", prior, None))
        if "nnpnu" in expected_risks:
            for prior in PRIORS:
                for eta in ETAS:
                    expected.add((library, "nnpnu", prior, eta))
    missing = expected - observed_arms
    extra = observed_arms - expected
    _require(not missing and not extra, "shard grid mismatch; missing={} extra={}".format(sorted(missing, key=str), sorted(extra, key=str)))
    return (
        pd.concat(configs_blocks, ignore_index=True),
        pd.concat(history_blocks, ignore_index=True),
        retention_by_run,
        predictions_by_run,
        manifests,
        max_epochs_by_run,
        row_contract,
        source_contract,
        trainer_hash,
    )


def _validate_shard_retention(configs, retention_by_run, predictions_by_run):
    """Confirm each shard evaluated exactly its weak-selected C/epoch."""
    for run_name, block in configs.groupby("run_name", sort=True):
        _, selected = select_prespecified_configs(block)
        _require(len(selected) == 1, "a shard did not reduce to one weak-selected config")
        config_id = selected.iloc[0]["config_id"]
        stored = retention_by_run[run_name]
        predictions = predictions_by_run[run_name]
        _require(stored.iloc[0]["config_id"] == config_id, "retention metric is for a non-selected config in {}".format(run_name))
        _require(set(predictions["config_id"]) == {config_id}, "retention predictions are for a non-selected config in {}".format(run_name))
        recomputed = _retention_metrics(predictions, "predicted_probability")
        for key, value in recomputed.items():
            _require(
                math.isclose(float(stored.iloc[0][key]), float(value), rel_tol=1e-9, abs_tol=1e-11),
                "retention metric {} does not reproduce in {}".format(key, run_name),
            )


def _load_primary_pn(primary_dir, expected_folds=None, expected_c_grid=None):
    manifest = _read_json(primary_dir / "manifest.json")
    configuration = manifest.get("configuration", {})
    if expected_folds is not None:
        validation_text = str(configuration.get("validation", ""))
        _require(
            validation_text.startswith("{} deterministic".format(int(expected_folds))),
            "primary PN fold contract differs: {}".format(validation_text),
        )
    if expected_c_grid is not None:
        expected = sorted(set(float(value) for value in expected_c_grid))
        observed = sorted(set(float(value) for value in configuration.get("c_grid", [])))
        _require(
            len(expected) == len(observed)
            and np.allclose(expected, observed, rtol=0.0, atol=1e-12),
            "primary PN C grid differs: expected {}, observed {}".format(expected, observed),
        )
    for filename in ("metrics.csv", "predictions.csv"):
        _require(filename in manifest.get("outputs", {}), "primary PN manifest omits {}".format(filename))
        path = primary_dir / filename
        _require(path.is_file(), "primary PN output is missing: {}".format(path))
        _require(manifest["outputs"][filename]["sha256"] == sha256_file(path), "primary PN hash mismatch: {}".format(path))
    stored = pd.read_csv(primary_dir / "metrics.csv", float_precision="round_trip")
    predictions = pd.read_csv(primary_dir / "predictions.csv", float_precision="round_trip")
    rows = []
    prediction_blocks = []
    for library in LIBRARIES:
        block = predictions.loc[predictions["library"].eq(library)].copy()
        _require(len(block) > 0, "primary PN lacks {} predictions".format(library))
        recomputed = _retention_metrics(block, "weak_site_logistic_probability")
        recorded = stored.loc[
            stored["library"].eq(library)
            & stored["evaluation_scope"].eq("within_library_primary")
            & stored["predictor"].eq("weak_site_logistic")
        ]
        _require(len(recorded) == 1, "primary PN metric row is not unique for {}".format(library))
        _require(math.isclose(recomputed["auroc"], float(recorded.iloc[0]["auroc"]), rel_tol=1e-10), "primary PN AUROC mismatch")
        _require(math.isclose(recomputed["average_precision"], float(recorded.iloc[0]["auprc"]), rel_tol=1e-10), "primary PN AP mismatch")
        rows.append(
            {
                "library": library,
                "model_role": "primary_pn_control",
                "risk": "pn",
                "class_prior": np.nan,
                "eta": np.nan,
                "config_id": "selection_weak_site_v2",
                "run_name": primary_dir.name,
                "validation_auroc": np.nan,
                "validation_average_precision": np.nan,
                "validation_log_loss": np.nan,
                "C": float(manifest["libraries"][library]["selected_C"]),
                "selected_epoch": np.nan,
                **recomputed
            }
        )
        selected_predictions = block[
            ["pair_uid", "peptide_uid", "affibody_uid", "library", "target_retention", "target_binder"]
        ].copy()
        selected_predictions.insert(0, "model_role", "primary_pn_control")
        selected_predictions.insert(1, "risk", "pn")
        selected_predictions.insert(2, "class_prior", np.nan)
        selected_predictions.insert(3, "eta", np.nan)
        selected_predictions.insert(4, "config_id", "selection_weak_site_v2")
        selected_predictions["predicted_probability"] = block["weak_site_logistic_probability"].to_numpy(dtype=float)
        prediction_blocks.append(selected_predictions)
    return pd.DataFrame(rows), pd.concat(prediction_blocks, ignore_index=True), manifest


def _selected_shard_outputs(selected, retention_by_run, predictions_by_run):
    metric_rows = []
    prediction_blocks = []
    for _, chosen in selected.iterrows():
        run_name = chosen["run_name"]
        metrics = retention_by_run[run_name].iloc[0].to_dict()
        _require(metrics["config_id"] == chosen["config_id"], "selected shard metric/config mismatch")
        role = "adam_pn_diagnostic" if chosen["risk"] == "pn" else "pnu_sensitivity"
        metric_rows.append(
            {
                "library": chosen["library"],
                "model_role": role,
                "risk": chosen["risk"],
                "class_prior": chosen["class_prior"],
                "eta": chosen["eta"],
                "config_id": chosen["config_id"],
                "run_name": run_name,
                "validation_auroc": chosen["validation_auroc"],
                "validation_average_precision": chosen["validation_average_precision"],
                "validation_log_loss": chosen["validation_log_loss"],
                "C": chosen["C"],
                "selected_epoch": chosen["selected_epoch"],
                **{key: metrics[key] for key in (
                    "n", "binders_ge_75", "auroc", "average_precision",
                    "global_spearman", "within_peptide_spearman", "within_peptide_groups"
                )}
            }
        )
        block = predictions_by_run[run_name].copy()
        block.insert(0, "model_role", role)
        # Already present in shard predictions, but explicit risk/pi/eta are
        # needed because selection_arm does not encode eta.
        block.insert(1, "risk", chosen["risk"])
        block.insert(2, "class_prior", chosen["class_prior"])
        block.insert(3, "eta", chosen["eta"])
        prediction_blocks.append(block)
    return pd.DataFrame(metric_rows), pd.concat(prediction_blocks, ignore_index=True)


def _markdown_report(
    metrics,
    selected,
    epoch_audit,
    row_contract,
    expected_folds=None,
    expected_c_grid=None,
):
    has_adam = bool(metrics["model_role"].eq("adam_pn_diagnostic").any())
    lines = [
        "# Designed-position P/NU sensitivity experiment",
        "",
        "This comparison asks whether the below-cutoff R9/R10 pairs can improve the "
        "existing mutation-position predictor without pretending that all of those pairs "
        "are non-binders.",
        "",
        "All model and epoch choices were made with the selection-derived weak labels. "
        "The retention matrix was opened only for the final retrospective evaluation. "
        "The four assumed hidden-positive fractions are shown separately; retention was "
        "not used to choose among them.",
        "",
        "## Data used",
        "",
        "| Library | Positive | Confident negative | Unlabeled | Retention evaluation |",
        "|---|---:|---:|---:|---:|",
    ]
    for library in LIBRARIES:
        p_rows, n_rows, u_rows, retention_rows = row_contract[library]
        lines.append("| {} | {:,} | {:,} | {:,} | {:,} |".format(library, p_rows, n_rows, u_rows, retention_rows))
    if expected_folds is not None and expected_c_grid is not None:
        lines.extend([
            "",
            "Validation used {} identity-blocked folds and the same C grid as the "
            "established PN baseline: {}. Both settings were enforced from every run "
            "manifest before aggregation.".format(
                int(expected_folds),
                ", ".join("{:g}".format(float(value)) for value in expected_c_grid),
            ),
            "The established PN row preserves its original weak-log-loss C selection; "
            "the PU/PNU grid selects by weak AUROC, then AP and log loss. The validation "
            "folds and candidate C values are matched, but the selection objective is not "
            "identical.",
        ])
    lines.extend(
        [
            "",
            "## Retrospective retention results",
            "",
            (
                "The established sklearn PN model is the primary control. The Adam PN row is "
                "only an optimizer/training diagnostic for the new implementation."
                if has_adam else
                "The established five-fold sklearn PN model is the primary control."
            ),
            "",
            "| Library | Training | Assumed hidden-positive fraction | eta | AUROC | AP | Global Spearman | Within-peptide Spearman |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordering = {"primary_pn_control": 0, "adam_pn_diagnostic": 1, "pnu_sensitivity": 2}
    display = metrics.copy()
    display["role_order"] = display["model_role"].map(ordering)
    display = display.sort_values(["library", "role_order", "risk", "class_prior"], na_position="first")
    for _, row in display.iterrows():
        if row["model_role"] == "primary_pn_control":
            label = "PN — established control"
        elif row["model_role"] == "adam_pn_diagnostic":
            label = "PN — Adam diagnostic"
        elif row["risk"] == "nnpu":
            label = "PU"
        else:
            label = "PNU"
        prior = "—" if pd.isna(row["class_prior"]) else "{:.0%}".format(row["class_prior"])
        eta = "—" if pd.isna(row["eta"]) else "{:.2f}".format(row["eta"])
        display_values = row.to_dict()
        display_values.update({"label": label, "prior_display": prior, "eta_display": eta})
        lines.append(
            "| {library} | {label} | {prior_display} | {eta_display} | {auroc:.3f} | {average_precision:.3f} | "
            "{global_spearman:.3f} | {within_peptide_spearman:.3f} |".format(
                **display_values
            )
        )
    selected_audit = epoch_audit.merge(
        selected[["run_name", "config_id"]].assign(selected_final=1),
        on=["run_name", "config_id"], how="inner"
    )
    first = int(selected_audit["epoch_location"].eq("first_epoch").sum())
    last = int(selected_audit["epoch_location"].eq("last_epoch").sum())
    lines.extend(
        [
            "",
            "## Selection audit",
            "",
            "For each fixed library and assumed hidden-positive fraction, eta, C, and epoch "
            "were ranked by weak-label AUROC, then weak-label AP, then lower weak-label "
            "log loss, then smaller eta, C, and epoch. All four fractions were retained.",
            "",
            "Among the {} final {} configurations, {} selected epoch 1 and {} "
            "selected the final allowed epoch. A final-epoch selection is a warning that "
            "the optimum may lie beyond the tested training window; it is not silently "
            "treated as an interior optimum.".format(
                len(selected_audit), "Adam/PU/PNU" if has_adam else "PU/PNU",
                first, last
            ),
            "",
            "The PU assumptions are imperfect here: positives are the high-count tail, not "
            "a random sample of every real binder. These rows are sensitivity analyses and "
            "the scores are not calibrated retention probabilities.",
            "An eta of 0 gives zero weight to U; that endpoint is a prior-weighted PN "
            "model and is not evidence that unlabeled pairs helped.",
            "",
        ]
    )
    if expected_c_grid is not None:
        upper_c = max(float(value) for value in expected_c_grid)
        upper_c_count = int(np.isclose(selected["C"].astype(float), upper_c).sum())
        lines.insert(
            -4,
            "{} of {} final configurations selected the largest tested C ({:g}); this is "
            "reported as a regularization-grid boundary warning.".format(
                upper_c_count, len(selected), upper_c
            ),
        )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument(
        "--primary-pn-dir",
        default=REPO_ROOT / "private_data/experiments/selection_weak_site_v2",
        type=Path,
    )
    parser.add_argument(
        "--adam-pn-root",
        default=REPO_ROOT / "private_data/experiments/pnu_site_full_u_auc_v1",
        type=Path,
        help="root containing LibA_pn and LibB_pn diagnostic shards",
    )
    parser.add_argument(
        "--omit-adam-pn",
        action="store_true",
        help="omit the historical 3-fold Adam PN diagnostic",
    )
    parser.add_argument("--expected-folds", type=int)
    parser.add_argument("--expected-c-grid", nargs="+", type=float)
    parser.add_argument("--expected-max-epochs", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    input_root = Path(args.input_root).resolve()
    _require(input_root.is_dir(), "input root does not exist: {}".format(input_root))
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    (
        configs,
        histories,
        retention_by_run,
        predictions_by_run,
        manifests,
        max_epochs_by_run,
        row_contract,
        source_contract,
        trainer_hash,
    ) = _load_shards(
        input_root,
        expected_risks=("nnpu", "nnpnu"),
        expected_folds=args.expected_folds,
        expected_c_grid=args.expected_c_grid,
        expected_max_epochs=args.expected_max_epochs,
    )
    adam_root = Path(args.adam_pn_root).resolve()
    adam_trainer_hash = None
    if not args.omit_adam_pn:
        _require(adam_root.is_dir(), "Adam PN root does not exist: {}".format(adam_root))
        (
            adam_configs,
            adam_histories,
            adam_retention,
            adam_predictions,
            adam_manifests,
            adam_max_epochs,
            adam_row_contract,
            adam_source_contract,
            adam_trainer_hash,
        ) = _load_shards(
            adam_root,
            expected_risks=("pn",),
            run_names=("LibA_pn", "LibB_pn"),
        )
        _require(adam_row_contract == row_contract, "Adam PN row-count contract differs")
        _require(adam_source_contract == source_contract, "Adam PN source-hash contract differs")
        configs = pd.concat([configs, adam_configs], ignore_index=True, sort=False)
        histories = pd.concat([histories, adam_histories], ignore_index=True, sort=False)
        retention_by_run.update(adam_retention)
        predictions_by_run.update(adam_predictions)
        manifests.update(adam_manifests)
        max_epochs_by_run.update(adam_max_epochs)
    epoch_audit = audit_epoch_choices(configs, histories, max_epochs_by_run)
    _validate_shard_retention(configs, retention_by_run, predictions_by_run)
    all_configs, selected = select_prespecified_configs(configs)

    # Contract checks after selection, still before retention is assembled.
    for library in LIBRARIES:
        for risk in ("nnpu", "nnpnu"):
            observed = set(
                selected.loc[
                    selected["library"].eq(library) & selected["risk"].eq(risk),
                    "class_prior",
                ].astype(float)
            )
            _require(observed == set(PRIORS), "not all priors retained for {}/{}".format(library, risk))

    shard_metrics, shard_predictions = _selected_shard_outputs(
        selected, retention_by_run, predictions_by_run
    )
    primary_metrics, primary_predictions, primary_manifest = _load_primary_pn(
        Path(args.primary_pn_dir).resolve(),
        expected_folds=args.expected_folds,
        expected_c_grid=args.expected_c_grid,
    )
    comparison = pd.concat([primary_metrics, shard_metrics], ignore_index=True, sort=False)
    predictions = pd.concat([primary_predictions, shard_predictions], ignore_index=True, sort=False)
    report = _markdown_report(
        comparison,
        selected,
        epoch_audit,
        row_contract,
        expected_folds=args.expected_folds,
        expected_c_grid=args.expected_c_grid,
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    output_paths = {
        "all_weak_configurations.csv": output_dir / "all_weak_configurations.csv",
        "selected_weak_configurations.csv": output_dir / "selected_weak_configurations.csv",
        "epoch_boundary_audit.csv": output_dir / "epoch_boundary_audit.csv",
        "retention_metrics_selected.csv": output_dir / "retention_metrics_selected.csv",
        "retention_predictions_selected.csv": output_dir / "retention_predictions_selected.csv",
        "pnu_site_report.md": output_dir / "pnu_site_report.md",
    }
    _write_csv(all_configs, output_paths["all_weak_configurations.csv"])
    _write_csv(selected, output_paths["selected_weak_configurations.csv"])
    _write_csv(epoch_audit, output_paths["epoch_boundary_audit.csv"])
    _write_csv(comparison, output_paths["retention_metrics_selected.csv"])
    _write_csv(predictions, output_paths["retention_predictions_selected.csv"])
    _atomic_text(report, output_paths["pnu_site_report.md"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "selection_contract": {
            "criterion": "weak AUROC, weak AP, minimum weak log loss, smaller eta, C, epoch",
            "class_prior_selection": "none; all 0.02/0.05/0.10/0.15 arms retained",
            "retention_usage": "final evaluation only; never used by aggregator selection",
            "primary_pn_control": str(Path(args.primary_pn_dir).resolve()),
            "adam_pn_role": (
                "omitted because the historical Adam run used three folds"
                if args.omit_adam_pn else
                "implementation diagnostic, not primary control"
            ),
        },
        "input_root": str(input_root),
        "trainer_sha256": trainer_hash,
        "expected_folds": args.expected_folds,
        "expected_c_grid": args.expected_c_grid,
        "expected_max_epochs": args.expected_max_epochs,
        "adam_pn_root": None if args.omit_adam_pn else str(adam_root),
        "adam_pn_trainer_sha256": adam_trainer_hash,
        "input_shards": {
            name: {
                "manifest_sha256": sha256_file(
                    (input_root / name / "manifest.json")
                    if (input_root / name / "manifest.json").is_file()
                    else (adam_root / name / "manifest.json")
                ),
                "library": payload["library"],
            }
            for name, payload in sorted(manifests.items())
        },
        "source_hash_contract": source_contract,
        "primary_pn_manifest_sha256": sha256_file(Path(args.primary_pn_dir).resolve() / "manifest.json"),
        "rows": {
            "all_weak_configurations": int(len(all_configs)),
            "selected_shard_configurations": int(len(selected)),
            "comparison_metric_rows": int(len(comparison)),
            "selected_prediction_rows": int(len(predictions)),
        },
        "epoch_boundary_counts": {
            key: int(value)
            for key, value in epoch_audit["epoch_location"].value_counts().sort_index().items()
        },
        "elapsed_seconds": float(time.time() - started),
        "outputs": {},
    }
    for name, path in output_paths.items():
        manifest["outputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    _write_json(manifest, output_dir / "manifest.json")
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_shard_configurations": int(len(selected)),
        "comparison_metric_rows": int(len(comparison)),
        "all_priors_retained": list(PRIORS),
    }, indent=2, sort_keys=True))
    return comparison


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
