#!/usr/bin/env python
"""Compare two fully aggregated P/NU epoch-window experiments.

The comparison is keyed only by library, risk, and fixed class prior. It does
not choose a class prior or configuration using retention. Its main purpose is
to determine whether extending the weak-label training window moves the
selected eta/C/epoch and whether any selected configuration remains pinned to
the new final epoch.
"""

from __future__ import print_function

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ("library", "risk", "class_prior")
RISKS = ("nnpu", "nnpnu")
PRIORS = (0.02, 0.05, 0.10, 0.15)
METRICS = (
    "auroc",
    "average_precision",
    "global_spearman",
    "within_peptide_spearman",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_json(path):
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


def _atomic_csv(frame, path):
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    frame.to_csv(str(temporary), index=False)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(path))


def _selected_tables(aggregate_dir):
    config_path = aggregate_dir / "selected_weak_configurations.csv"
    metric_path = aggregate_dir / "retention_metrics_selected.csv"
    manifest_path = aggregate_dir / "manifest.json"
    for path in (config_path, metric_path, manifest_path):
        _require(path.is_file(), "aggregate input is missing: {}".format(path))
    configs = pd.read_csv(config_path, float_precision="round_trip")
    metrics = pd.read_csv(metric_path, float_precision="round_trip")
    configs = configs.loc[configs["risk"].isin(RISKS)].copy()
    metrics = metrics.loc[metrics["risk"].isin(RISKS)].copy()
    _require(len(configs) == 16 and len(metrics) == 16, "aggregate must have 16 PU/PNU sensitivity rows")
    _require(not configs.duplicated(list(KEYS)).any(), "selected config keys are duplicated")
    _require(not metrics.duplicated(list(KEYS)).any(), "retention metric keys are duplicated")
    expected = {
        (library, risk, prior)
        for library in ("LibA", "LibB")
        for risk in RISKS
        for prior in PRIORS
    }
    observed = {
        (row.library, row.risk, float(row.class_prior))
        for row in configs.itertuples()
    }
    _require(observed == expected, "aggregate does not retain the complete fixed-prior grid")
    return configs, metrics, _load_json(manifest_path)


def build_convergence_comparison(reference_configs, reference_metrics, current_configs, current_metrics):
    """Build one matched row per fixed (library, risk, prior) arm."""
    reference = reference_configs.merge(reference_metrics[list(KEYS) + list(METRICS)], on=list(KEYS), validate="one_to_one")
    current = current_configs.merge(current_metrics[list(KEYS) + list(METRICS)], on=list(KEYS), validate="one_to_one")
    keep = list(KEYS) + [
        "eta", "C", "selected_epoch", "config_id", "validation_auroc",
        "validation_average_precision", "validation_log_loss",
    ] + list(METRICS)
    merged = reference[keep].merge(
        current[keep], on=list(KEYS), how="outer", validate="one_to_one",
        suffixes=("_reference", "_current")
    )
    _require(len(merged) == 16 and not merged.isna().all(axis=1).any(), "convergence merge is incomplete")
    merged["same_eta"] = np.isclose(merged["eta_reference"], merged["eta_current"], equal_nan=True).astype(int)
    merged["same_C"] = np.isclose(merged["C_reference"], merged["C_current"]).astype(int)
    merged["same_selected_epoch"] = merged["selected_epoch_reference"].astype(int).eq(merged["selected_epoch_current"].astype(int)).astype(int)
    merged["same_config_id"] = merged["config_id_reference"].eq(merged["config_id_current"]).astype(int)
    for metric in METRICS:
        merged["{}_change".format(metric)] = merged["{}_current".format(metric)] - merged["{}_reference".format(metric)]
    return merged.sort_values(list(KEYS)).reset_index(drop=True)


def _report(comparison, reference_max, current_max):
    current_boundary = comparison.loc[comparison["selected_epoch_current"].astype(int).eq(int(current_max))]
    reference_boundary = comparison.loc[comparison["selected_epoch_reference"].astype(int).eq(int(reference_max))]
    changed_config = comparison.loc[comparison["same_config_id"].eq(0)]
    lines = [
        "# P/NU training-window convergence check",
        "",
        "This check extends the maximum weak-label training window from {} to {} epochs. "
        "Every class-prior arm remains separate, and retention is used only to describe "
        "the already-selected configurations.".format(reference_max, current_max),
        "",
        "- Selected arms at the old final epoch: {} of 16".format(len(reference_boundary)),
        "- Selected arms at the new final epoch: {} of 16".format(len(current_boundary)),
        "- Arms whose selected eta or C changed: {} of 16".format(len(changed_config)),
        "",
    ]
    if len(current_boundary):
        lines.extend([
            "The following choices still hit epoch {}:".format(current_max),
            "",
            "| Library | Training | Assumed hidden-positive fraction | eta | C |",
            "|---|---|---:|---:|---:|",
        ])
        for row in current_boundary.itertuples():
            lines.append(
                "| {} | {} | {:.0%} | {:.2f} | {:g} |".format(
                    row.library, "PU" if row.risk == "nnpu" else "PNU",
                    row.class_prior, row.eta_current, row.C_current
                )
            )
        lines.append("")
    else:
        lines.extend([
            "No selected configuration hits epoch {}. The longer window resolves the "
            "previous upper-boundary warning for the final selected arms.".format(current_max),
            "",
        ])
    lines.extend([
        "An eta of 0 is the PNU endpoint that gives zero weight to U. Such a row is "
        "therefore a prior-weighted P-versus-N model, not evidence that the unlabeled "
        "pairs helped.",
        "",
    ])
    lines.extend([
        "## Matched retention results",
        "",
        "Changes below are the {}-epoch value minus the {}-epoch value. They are "
        "retrospective descriptions, not selection criteria.".format(current_max, reference_max),
        "",
        "| Library | Training | Hidden-positive fraction | Old epoch | New epoch | AUROC change | AP change | Within-peptide Spearman change |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison.itertuples():
        lines.append(
            "| {} | {} | {:.0%} | {} | {} | {:+.3f} | {:+.3f} | {:+.3f} |".format(
                row.library, "PU" if row.risk == "nnpu" else "PNU", row.class_prior,
                int(row.selected_epoch_reference), int(row.selected_epoch_current),
                row.auroc_change, row.average_precision_change,
                row.within_peptide_spearman_change,
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-aggregate", required=True, type=Path)
    parser.add_argument("--current-aggregate", required=True, type=Path)
    parser.add_argument("--reference-max-epochs", required=True, type=int)
    parser.add_argument("--current-max-epochs", required=True, type=int)
    return parser.parse_args(argv)


def run(args):
    reference_dir = Path(args.reference_aggregate).resolve()
    current_dir = Path(args.current_aggregate).resolve()
    _require(reference_dir != current_dir, "reference and current aggregates are identical")
    _require(int(args.current_max_epochs) > int(args.reference_max_epochs), "current window must be longer")
    ref_configs, ref_metrics, ref_manifest = _selected_tables(reference_dir)
    cur_configs, cur_metrics, cur_manifest = _selected_tables(current_dir)
    _require(ref_manifest["trainer_sha256"] == cur_manifest["trainer_sha256"], "trainer hashes differ between convergence runs")
    _require(ref_manifest["source_hash_contract"] == cur_manifest["source_hash_contract"], "source hashes differ between convergence runs")
    comparison = build_convergence_comparison(ref_configs, ref_metrics, cur_configs, cur_metrics)
    current_boundary = comparison["selected_epoch_current"].astype(int).eq(int(args.current_max_epochs))
    payload = {
        "reference_aggregate": str(reference_dir),
        "current_aggregate": str(current_dir),
        "trainer_sha256": cur_manifest["trainer_sha256"],
        "reference_max_epochs": int(args.reference_max_epochs),
        "current_max_epochs": int(args.current_max_epochs),
        "arms": int(len(comparison)),
        "selected_at_reference_boundary": int(comparison["selected_epoch_reference"].astype(int).eq(int(args.reference_max_epochs)).sum()),
        "selected_at_current_boundary": int(current_boundary.sum()),
        "selected_eta_or_C_changed": int((comparison["same_config_id"] == 0).sum()),
        "current_boundary_arms": comparison.loc[current_boundary, list(KEYS) + ["eta_current", "C_current"]].to_dict(orient="records"),
        "retention_was_not_used_for_selection": True,
    }
    outputs = {
        "convergence_vs_40epoch.csv": comparison,
        "convergence_vs_40epoch_report.md": _report(
            comparison, args.reference_max_epochs, args.current_max_epochs
        ),
        "convergence_vs_40epoch.json": json.dumps(payload, indent=2, sort_keys=True) + "\n",
    }
    for name in outputs:
        _require(not (current_dir / name).exists(), "output exists; refusing overwrite: {}".format(current_dir / name))
    _atomic_csv(outputs["convergence_vs_40epoch.csv"], current_dir / "convergence_vs_40epoch.csv")
    _atomic_text(outputs["convergence_vs_40epoch_report.md"], current_dir / "convergence_vs_40epoch_report.md")
    _atomic_text(outputs["convergence_vs_40epoch.json"], current_dir / "convergence_vs_40epoch.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return comparison


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
