#!/usr/bin/env python
"""Aggregate completed frozen-MINT weak-label evaluator runs.

The evaluator is intentionally run in small, independently restartable pieces
(typically one library and one generalization regime per directory).  This
script verifies those pieces before joining them.  It fails closed when an
output was modified after its manifest was written, when runs used different
feature/evaluator sources, or when two runs cover the same condition.

The row-level evaluator tables are retained verbatim apart from an added
``source_run`` provenance column.  A separate primary view selects the fixed
uncleaned (C0) retention panel and averages the prespecified downsampling seeds;
it never averages the three distinct balancing strategies together.
"""

from __future__ import print_function

import argparse
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


SCHEMA_VERSION = "cached-weak-mint-aggregate-v1"
EVALUATOR_SCHEMA_VERSION = "cached-weak-mint-evaluation-v1"
TABLE_FILES = {
    "metrics": "metrics.csv",
    "conditions": "conditions.csv",
    "predictions": "predictions.csv",
    "weak_validation": "weak_validation.csv",
}
REQUIRED_MANIFEST_OUTPUTS = set(TABLE_FILES.values()) | {"run_summary.md"}
CONDITION_KEY = (
    "library",
    "regime",
    "cleaning",
    "balance",
    "balance_seed",
)
TABLE_UNIQUE_KEYS = {
    "conditions": CONDITION_KEY,
    "metrics": CONDITION_KEY + ("evaluation_scope", "representation"),
    "predictions": CONDITION_KEY + ("representation", "pair_uid"),
    "weak_validation": CONDITION_KEY
    + ("representation", "record_type", "C", "fold"),
}
PRIMARY_SCOPE = "common_regime_core"
PRIMARY_CLEANING = "c0"
PRIMARY_REPRESENTATIONS = ("site", "frozen_mint_chain_mean")
PRIMARY_METRICS = (
    "global_auprc",
    "global_auroc",
    "global_spearman",
    "within_peptide_macro_auprc_lift",
    "within_peptide_p_at_1",
    "two_way_interaction_spearman",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_json(path):
    with open(str(path), "r") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "JSON root is not an object: {}".format(path))
    return payload


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _write_csv(frame, path):
    frame.to_csv(str(path), index=False)
    os.chmod(str(path), 0o600)


def _verified_path(path, expected_sha256, label):
    path = Path(path).resolve()
    _require(path.is_file(), "{} is missing: {}".format(label, path))
    _require(
        isinstance(expected_sha256, str) and len(expected_sha256) == 64,
        "{} has an invalid expected SHA256".format(label),
    )
    actual = sha256_file(path)
    _require(
        actual == expected_sha256,
        "{} hash mismatch: {}".format(label, path),
    )
    return path


def _verify_recorded_file(record, expected_path, label):
    _require(isinstance(record, dict), "{} manifest record is not an object".format(label))
    _require("path" in record and "sha256" in record, "{} manifest record is incomplete".format(label))
    recorded_path = Path(record["path"]).resolve()
    expected_path = Path(expected_path).resolve()
    _require(
        recorded_path == expected_path,
        "{} path disagrees with its run directory".format(label),
    )
    return _verified_path(expected_path, record["sha256"], label)


def _source_contract(manifest, run_name):
    """Verify source files and return a path-independent equality contract."""
    code = manifest.get("code")
    _require(isinstance(code, dict), "{} has no evaluator code record".format(run_name))
    _verified_path(code.get("path", ""), code.get("sha256"), "{} evaluator".format(run_name))

    sources = manifest.get("sources")
    _require(isinstance(sources, dict), "{} has no sources record".format(run_name))
    cache_records = sources.get("cache_npz")
    _require(
        isinstance(cache_records, list) and cache_records,
        "{} has no cache NPZ source records".format(run_name),
    )
    cache_hashes = []
    for index, record in enumerate(cache_records):
        _require(isinstance(record, dict), "{} cache source is malformed".format(run_name))
        _verified_path(
            record.get("path", ""),
            record.get("sha256"),
            "{} cache source {}".format(run_name, index),
        )
        cache_hashes.append(record["sha256"])

    retention = sources.get("retention_features")
    _require(
        isinstance(retention, dict),
        "{} has no retention-feature source record".format(run_name),
    )
    _verified_path(
        retention.get("path", ""),
        retention.get("sha256"),
        "{} retention features".format(run_name),
    )
    dependencies = manifest.get("dependencies", {})
    _require(
        isinstance(dependencies, dict),
        "{} evaluator dependencies record is malformed".format(run_name),
    )
    dependency_hashes = {}
    for dependency_name, record in sorted(dependencies.items()):
        _require(
            isinstance(record, dict),
            "{} dependency {} is malformed".format(run_name, dependency_name),
        )
        _verified_path(
            record.get("path", ""),
            record.get("sha256"),
            "{} dependency {}".format(run_name, dependency_name),
        )
        dependency_hashes[dependency_name] = record["sha256"]
    configuration = manifest.get("configuration", {})
    _require(
        isinstance(configuration, dict),
        "{} evaluator configuration is malformed".format(run_name),
    )
    return {
        "evaluator_sha256": code["sha256"],
        "evaluator_dependency_sha256": dependency_hashes,
        "solver_tolerance": configuration.get("solver_tolerance"),
        "cache_sha256": sorted(cache_hashes),
        "retention_features_sha256": retention["sha256"],
    }


def _check_manifest_row_counts(manifest, tables, run_name):
    rows = manifest.get("rows")
    _require(isinstance(rows, dict), "{} has no row-count record".format(run_name))
    comparisons = {
        "conditions": ("conditions", len(tables["conditions"])),
        "metric_records": ("metrics", len(tables["metrics"])),
        "prediction_records": ("predictions", len(tables["predictions"])),
    }
    for manifest_key, (table_name, observed) in comparisons.items():
        _require(manifest_key in rows, "{} lacks row count {}".format(run_name, manifest_key))
        _require(
            int(rows[manifest_key]) == int(observed),
            "{} {} row count disagrees with manifest".format(run_name, table_name),
        )


def load_verified_run(run_dir):
    """Load one evaluator directory only after verifying its provenance."""
    run_dir = Path(run_dir).resolve()
    _require(run_dir.is_dir(), "evaluator output directory is missing: {}".format(run_dir))
    run_name = run_dir.name
    manifest_path = run_dir / "manifest.json"
    _require(manifest_path.is_file(), "{} has no manifest.json".format(run_name))
    manifest = _read_json(manifest_path)
    _require(
        manifest.get("schema_version") == EVALUATOR_SCHEMA_VERSION,
        "{} evaluator schema mismatch".format(run_name),
    )

    outputs = manifest.get("outputs")
    _require(isinstance(outputs, dict), "{} has no output records".format(run_name))
    missing = REQUIRED_MANIFEST_OUTPUTS.difference(outputs)
    _require(not missing, "{} manifest lacks outputs {}".format(run_name, sorted(missing)))
    for filename, record in outputs.items():
        _require(
            Path(filename).name == filename,
            "{} manifest has a non-local output name {}".format(run_name, filename),
        )
        _verify_recorded_file(record, run_dir / filename, "{} {}".format(run_name, filename))

    tables = {}
    for table_name, filename in TABLE_FILES.items():
        tables[table_name] = pd.read_csv(run_dir / filename)
        _require(not tables[table_name].empty, "{} {} is empty".format(run_name, filename))
    _check_manifest_row_counts(manifest, tables, run_name)
    contract = _source_contract(manifest, run_name)
    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "contract": contract,
        "tables": tables,
    }


def _assert_columns(frame, required, label):
    missing = set(required).difference(frame.columns)
    _require(not missing, "{} lacks columns {}".format(label, sorted(missing)))


def _condition_tuples(frame):
    return set(
        frame.loc[:, CONDITION_KEY].astype(str).itertuples(index=False, name=None)
    )


def _assert_unique(frame, columns, label):
    _assert_columns(frame, columns, label)
    duplicate = frame.duplicated(list(columns), keep=False)
    if bool(duplicate.any()):
        examples = (
            frame.loc[duplicate, list(columns)]
            .head(3)
            .astype(str)
            .to_dict(orient="records")
        )
        raise ValueError("{} has duplicate keys: {}".format(label, examples))


def combine_verified_runs(verified_runs):
    """Concatenate verified runs and enforce cross-run condition integrity."""
    _require(len(verified_runs) >= 1, "at least one evaluator run is required")
    names = [run["run_name"] for run in verified_runs]
    _require(len(names) == len(set(names)), "source run directory names must be unique")
    reference_contract = verified_runs[0]["contract"]
    for run in verified_runs[1:]:
        _require(
            run["contract"] == reference_contract,
            "{} used a different evaluator/cache/retention source".format(run["run_name"]),
        )

    reference_columns = {
        table_name: tuple(verified_runs[0]["tables"][table_name].columns)
        for table_name in TABLE_FILES
    }
    blocks = {table_name: [] for table_name in TABLE_FILES}
    for run in verified_runs:
        for table_name in TABLE_FILES:
            frame = run["tables"][table_name]
            _require(
                tuple(frame.columns) == reference_columns[table_name],
                "{} {} schema differs from the other runs".format(
                    run["run_name"], table_name
                ),
            )
            block = frame.copy()
            block.insert(0, "source_run", run["run_name"])
            blocks[table_name].append(block)
    combined = {
        table_name: pd.concat(table_blocks, ignore_index=True, sort=False)
        for table_name, table_blocks in blocks.items()
    }

    for table_name, key in TABLE_UNIQUE_KEYS.items():
        _assert_unique(combined[table_name], key, "aggregate {}".format(table_name))
    condition_set = _condition_tuples(combined["conditions"])
    for table_name in ("metrics", "predictions", "weak_validation"):
        observed = _condition_tuples(combined[table_name])
        _require(
            observed == condition_set,
            "aggregate {} condition coverage differs from conditions.csv".format(table_name),
        )
    return combined, reference_contract


def _annotate_c_grid(metrics, weak_validation):
    """Attach the actually evaluated C range and a selected-max flag."""
    grid_key = CONDITION_KEY + ("representation",)
    _assert_columns(metrics, grid_key + ("selected_C",), "aggregate metrics")
    _assert_columns(weak_validation, grid_key + ("C",), "aggregate weak validation")
    grid_source = weak_validation.loc[:, list(grid_key) + ["C"]].copy()
    grid_source["C"] = pd.to_numeric(grid_source["C"], errors="raise").astype(float)
    _require(
        bool(np.isfinite(grid_source["C"]).all()) and bool(grid_source["C"].gt(0).all()),
        "weak-validation C grid contains invalid values",
    )
    grid = (
        grid_source.groupby(list(grid_key), sort=False, dropna=False)["C"]
        .agg(C_grid_min="min", C_grid_max="max", C_grid_size="nunique")
        .reset_index()
    )
    output = metrics.merge(grid, on=list(grid_key), how="left", validate="many_to_one")
    _require(not bool(output["C_grid_max"].isna().any()), "a metric row lacks its C grid")
    selected = pd.to_numeric(output["selected_C"], errors="raise").to_numpy(dtype=float)
    minimum = output["C_grid_min"].to_numpy(dtype=float)
    maximum = output["C_grid_max"].to_numpy(dtype=float)
    _require(
        bool(((selected >= minimum) & (selected <= maximum)).all()),
        "a selected C lies outside its weak-validation grid",
    )
    output["selected_C_at_grid_max"] = np.isclose(
        selected, maximum, rtol=1e-12, atol=0.0
    ).astype(int)
    return output


def build_primary_view(metrics, weak_validation=None):
    """Build the fixed-C0 common-core view, averaging only downsample seeds."""
    metrics = metrics.copy()
    if weak_validation is not None:
        metrics = _annotate_c_grid(metrics, weak_validation)
    else:
        # Unit-level callers may only be testing metric aggregation. Real
        # artifact creation always supplies weak_validation and therefore
        # always receives the explicit grid-boundary audit.
        metrics["C_grid_min"] = float("nan")
        metrics["C_grid_max"] = float("nan")
        metrics["C_grid_size"] = 0
        metrics["selected_C_at_grid_max"] = 0
    required = set(CONDITION_KEY) | {
        "evaluation_scope",
        "representation",
        "n",
        "positive",
        "global_prevalence",
        "retention_membership_sha256",
        "selected_C",
    } | set(PRIMARY_METRICS)
    _assert_columns(metrics, required, "aggregate metrics")
    primary = metrics.loc[
        metrics["cleaning"].eq(PRIMARY_CLEANING)
        & metrics["evaluation_scope"].eq(PRIMARY_SCOPE)
        & metrics["representation"].isin(PRIMARY_REPRESENTATIONS)
    ].copy()
    _require(not primary.empty, "no fixed C0 common-core metrics are available")

    # ``common_regime_core`` is useful only if its biological membership is
    # genuinely identical across the regime-specific evaluator invocations.
    # Check this explicitly instead of trusting the scope label.
    for library, library_rows in primary.groupby("library", sort=False):
        for column in (
            "n",
            "positive",
            "global_prevalence",
            "retention_membership_sha256",
        ):
            _require(
                library_rows[column].nunique(dropna=False) == 1,
                "{} common-regime core changes {} across regimes".format(
                    library, column
                ),
            )

    panel_columns = ["library", "regime", "cleaning", "balance"]
    for key, panel in primary.groupby(panel_columns, sort=False, dropna=False):
        observed_representations = set(panel["representation"].astype(str))
        _require(
            observed_representations == set(PRIMARY_REPRESENTATIONS),
            "primary panel {} does not contain both site and frozen MINT".format(key),
        )
        for column in ("n", "positive", "global_prevalence", "retention_membership_sha256"):
            _require(
                panel[column].nunique(dropna=False) == 1,
                "primary panel {} changes {} across representations/seeds".format(key, column),
            )

    group_columns = panel_columns + ["evaluation_scope", "representation"]
    rows = []
    for key, group in primary.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, key))
        for grid_column in ("C_grid_min", "C_grid_max", "C_grid_size"):
            _require(
                group[grid_column].nunique(dropna=False) == 1,
                "primary seed runs used different {} values".format(grid_column),
            )
        seeds = sorted(pd.to_numeric(group["balance_seed"], errors="raise").astype(int).unique())
        if metadata["balance"] == "balanced_downsample":
            _require(all(seed >= 0 for seed in seeds), "downsample condition has a sentinel seed")
        else:
            _require(seeds == [-1], "non-downsample condition must have exactly seed -1")
        row = dict(metadata)
        row.update(
            {
                "source_runs": ";".join(sorted(set(group["source_run"].astype(str)))),
                "balance_seeds": ",".join(str(seed) for seed in seeds),
                "n_seed_runs": int(len(seeds)),
                "n": int(group["n"].iloc[0]),
                "positive": int(group["positive"].iloc[0]),
                "global_prevalence": float(group["global_prevalence"].iloc[0]),
                "retention_membership_sha256": str(
                    group["retention_membership_sha256"].iloc[0]
                ),
                "selected_C_mean": float(pd.to_numeric(group["selected_C"]).mean()),
                "selected_C_values": ",".join(
                    "{:g}".format(value)
                    for value in sorted(
                        set(pd.to_numeric(group["selected_C"], errors="raise").astype(float))
                    )
                ),
                "C_grid_min": float(group["C_grid_min"].iloc[0]),
                "C_grid_max": float(group["C_grid_max"].iloc[0]),
                "C_grid_size": int(group["C_grid_size"].iloc[0]),
                "selected_C_at_grid_max_count": int(
                    pd.to_numeric(group["selected_C_at_grid_max"], errors="raise").sum()
                ),
                "selected_C_at_grid_max_any": int(
                    pd.to_numeric(group["selected_C_at_grid_max"], errors="raise").gt(0).any()
                ),
            }
        )
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite)) if len(finite) else float("nan")
            row[metric + "_seed_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) >= 2 else 0.0
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    representation_order = {value: index for index, value in enumerate(PRIMARY_REPRESENTATIONS)}
    output["_representation_order"] = output["representation"].map(representation_order)
    output = output.sort_values(
        ["library", "regime", "balance", "_representation_order"], kind="mergesort"
    ).drop(columns=["_representation_order"])
    return output.reset_index(drop=True)


def _metric_text(row, metric):
    value = float(row[metric])
    if not math.isfinite(value):
        return "NA"
    if int(row["n_seed_runs"]) > 1:
        spread = float(row[metric + "_seed_sd"])
        return "{:.3f}±{:.3f}".format(value, spread)
    return "{:.3f}".format(value)


def primary_markdown(primary, n_source_runs):
    """Render a compact site/MINT side-by-side primary table."""
    lines = [
        "# Cached weak-label MINT aggregate: primary C0 common-core view",
        "",
        "Fixed uncleaned common four-regime retention cores only. Each cell is site / frozen MINT. "
        "Balanced-downsample entries are mean±sample SD across their listed seeds; "
        "the three balance strategies remain separate.",
        "",
        "Source evaluator runs: {}.".format(int(n_source_runs)),
        "",
        "| Library | Regime | Balance | Seeds | N | Selected C | AP | AUROC | Spearman | Within-peptide AP lift | P@1 | Interaction rho |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    group_columns = ["library", "regime", "balance"]
    for key, group in primary.groupby(group_columns, sort=True, dropna=False):
        indexed = group.set_index("representation")
        _require(
            set(indexed.index) == set(PRIMARY_REPRESENTATIONS),
            "primary markdown panel lacks a representation",
        )
        site = indexed.loc["site"]
        mint = indexed.loc["frozen_mint_chain_mean"]
        _require(
            str(site["balance_seeds"]) == str(mint["balance_seeds"]),
            "site and MINT primary seeds differ",
        )

        def pair(metric):
            return "{} / {}".format(_metric_text(site, metric), _metric_text(mint, metric))

        def selected_c(row):
            value = str(row["selected_C_values"])
            if int(row["selected_C_at_grid_max_any"]):
                value += " (MAX)"
            return value

        lines.append(
            "| {} | {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} | {} |".format(
                key[0],
                key[1],
                key[2],
                site["balance_seeds"],
                int(site["n"]),
                selected_c(site),
                selected_c(mint),
                pair("global_auprc"),
                pair("global_auroc"),
                pair("global_spearman"),
                pair("within_peptide_macro_auprc_lift"),
                pair("within_peptide_p_at_1"),
                pair("two_way_interaction_spearman"),
            )
        )
    lines.extend(
        [
            "",
            "AP is sklearn average precision, a precision-recall summary. P@1 is precision "
            "at one within peptide. Interaction rho "
            "is Spearman correlation after removing additive peptide and Affibody effects. "
            "Selected C is site / frozen MINT; (MAX) flags weak-validation selection at the "
            "largest evaluated C and should be treated as a grid-boundary sensitivity. "
            "These retention comparisons are retrospective and exploratory.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    run_dirs = [Path(path).resolve() for path in args.run_dirs]
    _require(len(run_dirs) >= 1, "at least one --run-dirs value is required")

    verified = [load_verified_run(path) for path in run_dirs]
    combined, source_contract = combine_verified_runs(verified)
    primary = build_primary_view(
        combined["metrics"], weak_validation=combined["weak_validation"]
    )
    markdown = primary_markdown(primary, len(verified))

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    outputs = {}
    for table_name in ("metrics", "conditions", "predictions", "weak_validation"):
        path = output_dir / "aggregate_{}.csv".format(table_name)
        _write_csv(combined[table_name], path)
        outputs[path.name] = path
    primary_path = output_dir / "primary_metrics.csv"
    _write_csv(primary, primary_path)
    outputs[primary_path.name] = primary_path
    summary_path = output_dir / "primary_table.md"
    with open(str(summary_path), "w") as handle:
        handle.write(markdown)
    os.chmod(str(summary_path), 0o600)
    outputs[summary_path.name] = summary_path

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory",
        "configuration": {
            "primary_cleaning": PRIMARY_CLEANING,
            "primary_evaluation_scope": PRIMARY_SCOPE,
            "primary_representations": list(PRIMARY_REPRESENTATIONS),
            "downsample_aggregation": "arithmetic mean and sample SD within balance condition only",
            "regularization_boundary_flag": "selected C equals largest weak-validation C",
        },
        "source_contract": source_contract,
        "source_runs": [
            {
                "name": item["run_name"],
                "path": str(item["run_dir"]),
                "manifest_path": str(item["manifest_path"]),
                "manifest_sha256": item["manifest_sha256"],
            }
            for item in verified
        ],
        "rows": {
            "conditions": int(len(combined["conditions"])),
            "metric_records": int(len(combined["metrics"])),
            "prediction_records": int(len(combined["predictions"])),
            "weak_validation_records": int(len(combined["weak_validation"])),
            "primary_records": int(len(primary)),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(outputs.items())
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "permissions": {"directory": "0700", "files": "0600"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "source_runs": int(len(verified)),
                "conditions": int(len(combined["conditions"])),
                "primary_records": int(len(primary)),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="completed evaluate_cached_weak_mint.py output directories",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
