#!/usr/bin/env python
"""Evaluate independent-chain ESM-2 using the frozen-MINT split protocol."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    SPLIT_SCHEMES,
    load_retention_table,
    sha256_file,
    supervised_library_frame,
    validate_private_output_path,
)
from downstream.AffibodyMHC.evaluate_mint_features import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_K_GRID,
    crossed_bootstrap_ci,
    metrics_by_repeat,
    paired_comparisons,
    run_model,
    summarize_metrics,
)


ESM2_FEATURES = (
    "esm2_chain_mean",
    "esm2_targeted_mean",
    "esm2_designed_site_mean",
)
ESM2_MODELS = tuple(name + "_ridge" for name in ESM2_FEATURES) + (
    "esm2_targeted_mean_knn",
)
MODEL_PAIRS = (
    ("mint_chain_mean_ridge", "esm2_chain_mean_ridge", "chain_mean_ridge"),
    ("mint_targeted_mean_ridge", "esm2_targeted_mean_ridge", "targeted_mean_ridge"),
    (
        "mint_designed_site_mean_ridge",
        "esm2_designed_site_mean_ridge",
        "designed_site_mean_ridge",
    ),
    ("mint_targeted_mean_knn", "esm2_targeted_mean_knn", "targeted_mean_knn"),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def deterministic_seed(base_seed, *parts):
    token = "|".join(str(value) for value in parts)
    value = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
    return int(base_seed + value) % (2 ** 32 - 1)


def mint_vs_esm2_comparisons(mint_events, esm_events, draws, seed):
    rows = []
    keys = [
        "library",
        "split_scheme",
        "repeat",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
    ]
    for mint_model, esm_model, pooling in MODEL_PAIRS:
        mint = mint_events[mint_events["model"].eq(mint_model)][
            keys + ["y_true", "absolute_error"]
        ].rename(
            columns={
                "y_true": "mint_y_true",
                "absolute_error": "mint_absolute_error",
            }
        )
        esm = esm_events[esm_events["model"].eq(esm_model)][
            keys + ["y_true", "absolute_error"]
        ].rename(
            columns={
                "y_true": "esm_y_true",
                "absolute_error": "esm_absolute_error",
            }
        )
        merged = mint.merge(esm, on=keys, validate="one_to_one")
        _require(len(merged) == len(mint) == len(esm), "MINT/ESM event alignment mismatch")
        _require(
            bool(np.allclose(merged["mint_y_true"], merged["esm_y_true"])),
            "MINT/ESM targets differ",
        )
        per_pair = (
            merged.groupby(
                ["library", "split_scheme", "pair_uid", "peptide_uid", "affibody_uid"],
                as_index=False,
            )[["mint_absolute_error", "esm_absolute_error"]]
            .mean()
            .reset_index(drop=True)
        )
        per_pair["paired_absolute_error_improvement"] = (
            per_pair["esm_absolute_error"] - per_pair["mint_absolute_error"]
        )
        for (library, scheme), subset in per_pair.groupby(["library", "split_scheme"]):
            lower, upper = crossed_bootstrap_ci(
                subset,
                draws,
                deterministic_seed(seed, library, scheme, pooling),
            )
            rows.append(
                {
                    "library": library,
                    "split_scheme": scheme,
                    "pooling": pooling,
                    "mint_model": mint_model,
                    "esm2_model": esm_model,
                    "n": int(len(subset)),
                    "delta_mae_esm2_minus_mint": float(
                        subset["paired_absolute_error_improvement"].mean()
                    ),
                    "delta_mae_ci_2.5": lower,
                    "delta_mae_ci_97.5": upper,
                    "bootstrap_draws": int(draws),
                }
            )
    return pd.DataFrame(rows).sort_values(["library", "split_scheme", "pooling"])


def write_summary(path, metrics, site_comparisons, backbone_comparisons):
    lines = [
        "# Independent-chain ESM-2 Affibody--pMHC results",
        "",
        "Positive delta versus site code favors ESM-2. Positive ESM-2-minus-MINT",
        "delta favors MINT because MINT has the smaller paired absolute error.",
        "",
        "## ESM-2 versus site-residue code baseline",
        "",
        "| Library | Split | Model | MAE | Delta MAE vs site code | 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    joined = metrics.merge(
        site_comparisons[
            [
                "library",
                "split_scheme",
                "model",
                "delta_mae_site_minus_model",
                "delta_mae_ci_2.5",
                "delta_mae_ci_97.5",
            ]
        ],
        on=["library", "split_scheme", "model"],
        validate="one_to_one",
    )
    for _, row in joined.iterrows():
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |".format(
                row["library"],
                row["split_scheme"],
                row["model"],
                row["mae"],
                row["delta_mae_site_minus_model"],
                row["delta_mae_ci_2.5"],
                row["delta_mae_ci_97.5"],
            )
        )
    lines.extend(
        [
            "",
            "## Direct MINT versus ESM-2",
            "",
            "| Library | Split | Pooling | ESM-2 MAE minus MINT MAE | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in backbone_comparisons.iterrows():
        lines.append(
            "| {} | {} | {} | {:.4f} | [{:.4f}, {:.4f}] |".format(
                row["library"],
                row["split_scheme"],
                row["pooling"],
                row["delta_mae_esm2_minus_mint"],
                row["delta_mae_ci_2.5"],
                row["delta_mae_ci_97.5"],
            )
        )
    lines.extend(
        [
            "",
            "## Constraints",
            "",
            "- ESM-2 chains are encoded independently; a same-chain/changed-partner audit",
            "  must be zero within numerical tolerance.",
            "- MINT and ESM-2 use identical sequences, pooling definitions, outer folds,",
            "  inner tuning, and lightweight prediction heads.",
            "- The official separate-chain mean is primary; targeted/site pooling is sensitivity.",
            "- Cold MAE is primary; pooled cold correlations are not global ranking metrics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--sequence-table", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--mint-events", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--k-grid", type=int, nargs="+", default=DEFAULT_K_GRID)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    script_path = Path(__file__).resolve()
    dependency_path = Path(__file__).resolve().with_name("evaluate_mint_features.py")
    script_hash = sha256_file(script_path)
    dependency_hash = sha256_file(dependency_path)
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory already exists; refusing overwrite")
    for path in (args.retention_csv, args.sequence_table, args.features, args.mint_events):
        _require(path.is_file(), "missing input {}".format(path))
    _require(args.random_repeats >= 1, "random repeats must be positive")
    _require(args.bootstrap_draws >= 1000, "bootstrap draws must be at least 1000")

    source_hashes = {
        "retention_csv": sha256_file(args.retention_csv),
        "sequence_table": sha256_file(args.sequence_table),
        "features": sha256_file(args.features),
        "mint_events": sha256_file(args.mint_events),
    }
    source = load_retention_table(args.retention_csv)
    sequence_table = pd.read_csv(
        args.sequence_table, dtype=str, keep_default_na=False, na_filter=False
    )
    _require(set(source["pair_uid"]) == set(sequence_table["pair_uid"]), "sequence UID mismatch")
    loaded = np.load(args.features)
    feature_arrays = {name: loaded[name] for name in loaded.files}
    feature_uids = [str(value) for value in feature_arrays["pair_uid"]]
    _require(len(feature_uids) == len(set(feature_uids)), "duplicate feature UID")
    _require(set(feature_uids) == set(source["pair_uid"]), "feature UID mismatch")
    for name in ESM2_FEATURES:
        _require(name in feature_arrays, "missing feature {}".format(name))
        _require(feature_arrays[name].shape == (228, 2560), "feature shape mismatch")
        _require(bool(np.isfinite(feature_arrays[name]).all()), "non-finite feature")
    pair_uid_to_index = {value: index for index, value in enumerate(feature_uids)}

    all_events = []
    all_tuning = []
    for library in sorted(source["library"].unique()):
        frame = supervised_library_frame(source, library)
        for scheme in SPLIT_SCHEMES:
            for model_name in ESM2_MODELS:
                events, tuning = run_model(
                    frame,
                    library,
                    scheme,
                    model_name,
                    feature_arrays,
                    pair_uid_to_index,
                    args.seed,
                    args.random_repeats,
                    args.alpha_grid,
                    args.k_grid,
                )
                all_events.extend(events)
                all_tuning.extend(tuning)
                print("completed {} {} {}".format(library, scheme, model_name), flush=True)
    esm_events = pd.DataFrame(all_events)
    tuning = pd.DataFrame(all_tuning)
    by_repeat = metrics_by_repeat(esm_events)
    metrics = summarize_metrics(by_repeat)

    mint_events = pd.read_csv(args.mint_events, float_precision="round_trip")
    site_events = mint_events[mint_events["model"].eq("site_residue_additive")].copy()
    comparison_events = pd.concat([site_events, esm_events], ignore_index=True)
    site_comparisons = paired_comparisons(
        comparison_events, args.bootstrap_draws, args.seed
    )
    site_comparisons = site_comparisons[
        site_comparisons["model"].isin(ESM2_MODELS)
    ].reset_index(drop=True)
    backbone_comparisons = mint_vs_esm2_comparisons(
        mint_events, esm_events, args.bootstrap_draws, args.seed
    )

    for name, expected_hash in source_hashes.items():
        _require(expected_hash == sha256_file(getattr(args, name)), "{} changed".format(name))
    _require(script_hash == sha256_file(script_path), "runner changed during run")
    _require(dependency_hash == sha256_file(dependency_path), "evaluation dependency changed")

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    artifacts = {
        "prediction_events.csv": esm_events,
        "metrics_by_repeat.csv": by_repeat,
        "metrics.csv": metrics,
        "site_code_comparisons.csv": site_comparisons,
        "mint_vs_esm2_comparisons.csv": backbone_comparisons,
        "hyperparameters.csv": tuning,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(output_dir / filename, index=False, quoting=csv.QUOTE_MINIMAL)
        os.chmod(str(output_dir / filename), 0o600)
    summary_path = output_dir / "run_summary.md"
    write_summary(summary_path, metrics, site_comparisons, backbone_comparisons)
    os.chmod(str(summary_path), 0o600)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "sources": {
            name: {"path": str(getattr(args, name).resolve()), "sha256": value}
            for name, value in source_hashes.items()
        },
        "code": {
            "path": str(script_path),
            "sha256": script_hash,
            "evaluation_dependency": str(dependency_path),
            "evaluation_dependency_sha256": dependency_hash,
        },
        "configuration": {
            "models": list(ESM2_MODELS),
            "split_schemes": list(SPLIT_SCHEMES),
            "seed": int(args.seed),
            "random_repeats": int(args.random_repeats),
            "alpha_grid": [float(value) for value in args.alpha_grid],
            "k_grid": [int(value) for value in args.k_grid],
            "bootstrap_draws": int(args.bootstrap_draws),
            "inner_tuning_metric": "pooled inner out-of-fold MAE",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {},
    }
    for filename in sorted(list(artifacts) + ["run_summary.md"]):
        manifest["outputs"][filename] = sha256_file(output_dir / filename)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(manifest_path), 0o600)
    print(json.dumps({"output_dir": str(output_dir), "seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    run(parse_args())
