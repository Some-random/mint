#!/usr/bin/env python
"""Evaluate frozen MINT features against code-only Affibody baselines.

All fitting occurs inside held-out folds.  Cold-split conclusions use pointwise
MAE; pooled cold correlations are retained as diagnostics only because their
scores come from different fitted models.
"""

import argparse
import csv
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
import scipy
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    SPLIT_SCHEMES,
    feature_matrix,
    load_retention_table,
    make_estimator,
    make_inner_splits,
    make_outer_splits,
    regression_metrics,
    sha256_file,
    supervised_library_frame,
    validate_private_output_path,
)


CODE_MODELS = ("mean_prior", "site_residue_additive", "whole_code_additive")
MINT_FEATURES = (
    "mint_chain_mean",
    "mint_targeted_mean",
    "mint_designed_site_mean",
)
MODEL_NAMES = CODE_MODELS + tuple(name + "_ridge" for name in MINT_FEATURES) + (
    "mint_targeted_mean_knn",
)
DEFAULT_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DEFAULT_K_GRID = (1, 3, 5, 10, 20)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    return sha256_file(path)


def dense_estimator(alpha):
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")),
        ]
    )


def cosine_knn_predict(train_features, train_target, test_features, k):
    train_norm = np.linalg.norm(train_features, axis=1, keepdims=True)
    test_norm = np.linalg.norm(test_features, axis=1, keepdims=True)
    train_norm[train_norm == 0.0] = 1.0
    test_norm[test_norm == 0.0] = 1.0
    similarity = np.matmul(test_features / test_norm, (train_features / train_norm).T)
    effective_k = min(int(k), train_features.shape[0])
    neighbor_indices = np.argpartition(-similarity, effective_k - 1, axis=1)[:, :effective_k]
    return np.asarray(train_target[neighbor_indices].mean(axis=1), dtype=float)


def model_kind(model_name):
    if model_name in CODE_MODELS:
        return "code"
    if model_name.endswith("_ridge"):
        return "dense_ridge"
    if model_name.endswith("_knn"):
        return "dense_knn"
    raise ValueError("unknown model {}".format(model_name))


def model_feature_name(model_name):
    if model_name.endswith("_ridge"):
        return model_name[: -len("_ridge")]
    if model_name.endswith("_knn"):
        return model_name[: -len("_knn")]
    return model_name


def fit_predict(model_name, library, parameter, train_frame, test_frame, dense_train, dense_test, seed):
    kind = model_kind(model_name)
    target = train_frame["target_retention"].to_numpy(dtype=float)
    if kind == "code":
        estimator = make_estimator(model_name, "regression", library, parameter, seed)
        train_matrix = feature_matrix(train_frame, model_name, library)
        test_matrix = feature_matrix(test_frame, model_name, library)
        estimator.fit(train_matrix, target)
        return np.asarray(estimator.predict(test_matrix), dtype=float)
    if kind == "dense_ridge":
        estimator = dense_estimator(parameter)
        estimator.fit(dense_train, target)
        return np.asarray(estimator.predict(dense_test), dtype=float)
    return cosine_knn_predict(dense_train, target, dense_test, parameter)


def tune_parameter(model_name, library, outer_train, dense_outer_train, scheme, seed, alpha_grid, k_grid):
    if model_name == "mean_prior":
        return None, {"rule": "none", "candidates": []}
    splits, split_audit = make_inner_splits(outer_train, scheme, seed, "regression")
    grid = k_grid if model_kind(model_name) == "dense_knn" else alpha_grid
    candidates = []
    for parameter in grid:
        observed = []
        predicted = []
        for inner_train, inner_validation in splits:
            dense_train = None if dense_outer_train is None else dense_outer_train[inner_train]
            dense_validation = (
                None if dense_outer_train is None else dense_outer_train[inner_validation]
            )
            prediction = fit_predict(
                model_name,
                library,
                parameter,
                outer_train.iloc[inner_train],
                outer_train.iloc[inner_validation],
                dense_train,
                dense_validation,
                seed,
            )
            observed.extend(
                outer_train.iloc[inner_validation]["target_retention"].to_numpy(dtype=float)
            )
            predicted.extend(prediction)
        loss = float(mean_absolute_error(np.asarray(observed), np.asarray(predicted)))
        candidates.append({"parameter": float(parameter), "pooled_inner_oof_mae": loss})
    best_loss = min(record["pooled_inner_oof_mae"] for record in candidates)
    tolerance = 1e-12
    tied = [
        record
        for record in candidates
        if abs(record["pooled_inner_oof_mae"] - best_loss) <= tolerance
    ]
    if model_kind(model_name) == "dense_knn":
        chosen = min(tied, key=lambda record: record["parameter"])
        rule = "minimum_pooled_inner_oof_mae_tie_smallest_k"
    else:
        chosen = max(tied, key=lambda record: record["parameter"])
        rule = "minimum_pooled_inner_oof_mae_tie_largest_alpha"
    return chosen["parameter"], {
        "rule": rule,
        "chosen_parameter": chosen["parameter"],
        "best_pooled_inner_oof_mae": best_loss,
        "inner_split": split_audit,
        "candidates": candidates,
    }


def feature_matrix_for_rows(feature_arrays, pair_uid_to_index, pair_uids, feature_name):
    indices = np.asarray([pair_uid_to_index[value] for value in pair_uids], dtype=int)
    return np.asarray(feature_arrays[feature_name][indices], dtype=np.float64)


def run_model(frame, library, scheme, model_name, feature_arrays, pair_uid_to_index, seed, random_repeats, alpha_grid, k_grid):
    feature_name = model_feature_name(model_name)
    dense_full = None
    if model_kind(model_name).startswith("dense"):
        dense_full = feature_matrix_for_rows(
            feature_arrays,
            pair_uid_to_index,
            frame["pair_uid"].tolist(),
            feature_name,
        )
    events = []
    tuning_rows = []
    outer_splits = make_outer_splits(frame, scheme, seed, random_repeats)
    target = frame["target_retention"].to_numpy(dtype=float)
    for split_number, split in enumerate(outer_splits):
        train = split["train"]
        test = split["test"]
        outer_train = frame.iloc[train].reset_index(drop=True)
        dense_train = None if dense_full is None else dense_full[train]
        dense_test = None if dense_full is None else dense_full[test]
        fold_seed = int(seed + split_number * 1009)
        parameter, audit = tune_parameter(
            model_name,
            library,
            outer_train,
            dense_train,
            scheme,
            fold_seed,
            alpha_grid,
            k_grid,
        )
        prediction = fit_predict(
            model_name,
            library,
            parameter,
            outer_train,
            frame.iloc[test],
            dense_train,
            dense_test,
            fold_seed,
        )
        _require(bool(np.isfinite(prediction).all()), "non-finite prediction")
        for local_index, row_index in enumerate(test):
            row = frame.iloc[row_index]
            events.append(
                {
                    "library": library,
                    "split_scheme": scheme,
                    "repeat": int(split["repeat"]),
                    "outer_fold": split["outer_fold"],
                    "model": model_name,
                    "feature_set": feature_name,
                    "pair_uid": row["pair_uid"],
                    "peptide_uid": row["peptide_uid"],
                    "affibody_uid": row["affibody_uid"],
                    "y_true": float(target[row_index]),
                    "y_score": float(prediction[local_index]),
                    "absolute_error": float(abs(target[row_index] - prediction[local_index])),
                    "chosen_parameter": "" if parameter is None else float(parameter),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_guarded": int(len(split["guarded"])),
                }
            )
        tuning_rows.append(
            {
                "library": library,
                "split_scheme": scheme,
                "outer_fold": split["outer_fold"],
                "repeat": int(split["repeat"]),
                "model": model_name,
                "chosen_parameter": "" if parameter is None else float(parameter),
                "tuning_audit_json": json.dumps(audit, sort_keys=True, separators=(",", ":")),
            }
        )
    return events, tuning_rows


def metrics_by_repeat(events):
    rows = []
    group_columns = ["library", "split_scheme", "model", "feature_set", "repeat"]
    for values, subset in events.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, values))
        _require(subset["pair_uid"].nunique() == len(subset), "duplicate pair within repeat")
        row["n"] = int(len(subset))
        row.update(
            regression_metrics(
                subset["y_true"].to_numpy(dtype=float),
                subset["y_score"].to_numpy(dtype=float),
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def summarize_metrics(by_repeat):
    rows = []
    groups = ["library", "split_scheme", "model", "feature_set"]
    metric_columns = [
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "predictions_below_0",
        "predictions_above_100",
    ]
    for values, subset in by_repeat.groupby(groups, sort=True):
        row = dict(zip(groups, values))
        _require(subset["n"].nunique() == 1, "repeat size mismatch")
        row["n"] = int(subset["n"].iloc[0])
        row["repeats"] = int(len(subset))
        for column in metric_columns:
            numeric = subset[column].dropna().to_numpy(dtype=float)
            row[column] = float(np.mean(numeric)) if len(numeric) else np.nan
            row[column + "_sd"] = (
                float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def crossed_bootstrap_ci(frame, draws, seed):
    rng = np.random.RandomState(seed)
    peptide = frame["peptide_uid"].to_numpy(dtype=str)
    affibody = frame["affibody_uid"].to_numpy(dtype=str)
    values = frame["paired_absolute_error_improvement"].to_numpy(dtype=float)
    unique_peptide = np.asarray(sorted(set(peptide)), dtype=str)
    unique_affibody = np.asarray(sorted(set(affibody)), dtype=str)
    peptide_to_index = {value: index for index, value in enumerate(unique_peptide)}
    affibody_to_index = {value: index for index, value in enumerate(unique_affibody)}
    peptide_index = np.asarray([peptide_to_index[value] for value in peptide], dtype=int)
    affibody_index = np.asarray([affibody_to_index[value] for value in affibody], dtype=int)
    peptide_counts = rng.multinomial(
        len(unique_peptide),
        np.repeat(1.0 / len(unique_peptide), len(unique_peptide)),
        size=draws,
    )
    affibody_counts = rng.multinomial(
        len(unique_affibody),
        np.repeat(1.0 / len(unique_affibody), len(unique_affibody)),
        size=draws,
    )
    weights = peptide_counts[:, peptide_index] * affibody_counts[:, affibody_index]
    weight_sums = weights.sum(axis=1)
    _require(bool(np.all(weight_sums > 0)), "empty bootstrap draw")
    estimates = np.matmul(weights, values) / weight_sums
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def paired_comparisons(events, draws, seed):
    per_pair = (
        events.groupby(
            [
                "library",
                "split_scheme",
                "model",
                "feature_set",
                "pair_uid",
                "peptide_uid",
                "affibody_uid",
            ],
            as_index=False,
        )["absolute_error"]
        .mean()
        .reset_index(drop=True)
    )
    rows = []
    for (library, scheme), subset in per_pair.groupby(["library", "split_scheme"]):
        reference = subset[subset["model"].eq("site_residue_additive")][
            ["pair_uid", "absolute_error"]
        ].rename(columns={"absolute_error": "reference_absolute_error"})
        for model_name, model_rows in subset.groupby("model"):
            merged = model_rows.merge(reference, on="pair_uid", validate="one_to_one")
            merged["paired_absolute_error_improvement"] = (
                merged["reference_absolute_error"] - merged["absolute_error"]
            )
            comparison_seed = int(
                seed
                + int(
                    hashlib.sha256(
                        (library + "|" + scheme + "|" + model_name).encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                )
            ) % (2 ** 32 - 1)
            lower, upper = crossed_bootstrap_ci(
                merged,
                draws,
                comparison_seed,
            )
            rows.append(
                {
                    "library": library,
                    "split_scheme": scheme,
                    "model": model_name,
                    "reference_model": "site_residue_additive",
                    "n": int(len(merged)),
                    "delta_mae_site_minus_model": float(
                        merged["paired_absolute_error_improvement"].mean()
                    ),
                    "delta_mae_ci_2.5": lower,
                    "delta_mae_ci_97.5": upper,
                    "bootstrap_draws": int(draws),
                }
            )
    return pd.DataFrame(rows).sort_values(["library", "split_scheme", "model"])


def zero_shot_ppi_metrics(source, feature_arrays, pair_uid_to_index):
    if "bernett_generic_ppi_probability" not in feature_arrays:
        return pd.DataFrame()
    rows = []
    values = feature_arrays["bernett_generic_ppi_probability"]
    for library in sorted(source["library"].unique()):
        frame = supervised_library_frame(source, library)
        indices = [pair_uid_to_index[value] for value in frame["pair_uid"]]
        probability = np.asarray(values[indices], dtype=float)
        retention = frame["target_retention"].to_numpy(dtype=float)
        binder = frame["target_binder"].to_numpy(dtype=int)
        rows.append(
            {
                "library": library,
                "n": int(len(frame)),
                "score_min": float(probability.min()),
                "score_max": float(probability.max()),
                "score_mean": float(probability.mean()),
                "score_std": float(probability.std()),
                "retention_spearman": float(spearmanr(retention, probability)[0]),
                "retention_pearson": float(pearsonr(retention, probability)[0]),
                "binder_average_precision": float(average_precision_score(binder, probability)),
                "binder_roc_auc": float(roc_auc_score(binder, probability)),
                "interpretation": "training-free generic binary-PPI score; not calibrated retention",
            }
        )
    return pd.DataFrame(rows)


def write_summary(path, metrics, comparisons, zero_shot):
    lines = [
        "# Frozen MINT Affibody--pMHC results",
        "",
        "All supervised results are outer-held-out. Positive delta MAE means the model",
        "improves on the fixed site-residue code baseline.",
        "",
        "## Retention regression",
        "",
        "| Library | Split | Model | MAE | SD | Delta MAE vs site code | 95% crossed-bootstrap CI |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    joined = metrics.merge(
        comparisons[
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
            "| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | [{:.4f}, {:.4f}] |".format(
                row["library"],
                row["split_scheme"],
                row["model"],
                row["mae"],
                row["mae_sd"],
                row["delta_mae_site_minus_model"],
                row["delta_mae_ci_2.5"],
                row["delta_mae_ci_97.5"],
            )
        )
    lines.extend(
        [
            "",
            "## Training-free generic PPI head",
            "",
            "| Library | N | Score range | Spearman with retention | AP | AUROC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in zero_shot.iterrows():
        lines.append(
            "| {} | {} | {:.4f}--{:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["library"],
                int(row["n"]),
                row["score_min"],
                row["score_max"],
                row["retention_spearman"],
                row["binder_average_precision"],
                row["binder_roc_auc"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Random-pair CV is an interpolation diagnostic; cold splits drive conclusions.",
            "- Cold MAE is primary. Pooled cold correlations compare different fitted models",
            "  and must not be treated as one global ranking experiment.",
            "- Double-cold uses one held-out cell per fitted model; MAE is meaningful, but",
            "  a pooled rank correlation is not.",
            "- Frozen 2,560-dimensional representations are fit to only 108 or 119 labels;",
            "  ridge regularization is selected entirely inside each outer training fold.",
            "- The Bernett head predicts generic PPI, not this assay's retention endpoint.",
            "- Targeted and designed-site pooling are prespecified sensitivity analyses;",
            "  the repository-recommended separate-chain mean is the primary MINT feature.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--sequence-table", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
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
    script_hash = sha256_file(script_path)
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory already exists; refusing overwrite")
    for path in (args.retention_csv, args.sequence_table, args.features):
        _require(path.is_file(), "missing input {}".format(path))
    _require(args.random_repeats >= 1, "random repeats must be positive")
    _require(args.bootstrap_draws >= 1000, "bootstrap draws must be at least 1000")
    _require(all(value > 0 for value in args.alpha_grid), "ridge alphas must be positive")
    _require(all(value >= 1 for value in args.k_grid), "k values must be positive")

    source_hashes = {
        "retention_csv": file_sha256(args.retention_csv),
        "sequence_table": file_sha256(args.sequence_table),
        "features": file_sha256(args.features),
    }
    source = load_retention_table(args.retention_csv)
    sequence_table = pd.read_csv(
        args.sequence_table, dtype=str, keep_default_na=False, na_filter=False
    )
    _require(set(source["pair_uid"]) == set(sequence_table["pair_uid"]), "sequence-table UID mismatch")
    loaded = np.load(args.features)
    feature_arrays = {name: loaded[name] for name in loaded.files}
    feature_uids = [str(value) for value in feature_arrays["pair_uid"]]
    _require(len(feature_uids) == len(set(feature_uids)), "duplicate feature UID")
    _require(set(feature_uids) == set(source["pair_uid"]), "feature UID mismatch")
    for name in MINT_FEATURES:
        _require(name in feature_arrays, "missing feature {}".format(name))
        _require(feature_arrays[name].shape == (228, 2560), "feature shape mismatch")
        _require(bool(np.isfinite(feature_arrays[name]).all()), "non-finite feature values")
    pair_uid_to_index = {value: index for index, value in enumerate(feature_uids)}

    all_events = []
    all_tuning = []
    for library in sorted(source["library"].unique()):
        frame = supervised_library_frame(source, library)
        for scheme in SPLIT_SCHEMES:
            for model_name in MODEL_NAMES:
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

    events = pd.DataFrame(all_events)
    tuning = pd.DataFrame(all_tuning)
    by_repeat = metrics_by_repeat(events)
    metrics = summarize_metrics(by_repeat)
    comparisons = paired_comparisons(events, args.bootstrap_draws, args.seed)
    zero_shot = zero_shot_ppi_metrics(source, feature_arrays, pair_uid_to_index)

    for name, expected_hash in source_hashes.items():
        path = getattr(args, name if name != "features" else "features")
        _require(expected_hash == file_sha256(path), "{} changed during run".format(name))
    _require(script_hash == sha256_file(script_path), "runner changed during run")

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    output_dir.chmod(0o700)
    artifacts = {
        "prediction_events.csv": events,
        "metrics_by_repeat.csv": by_repeat,
        "metrics.csv": metrics,
        "paired_comparisons.csv": comparisons,
        "hyperparameters.csv": tuning,
        "zero_shot_ppi_metrics.csv": zero_shot,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(output_dir / filename, index=False, quoting=csv.QUOTE_MINIMAL)
        os.chmod(str(output_dir / filename), 0o600)
    summary_path = output_dir / "run_summary.md"
    write_summary(summary_path, metrics, comparisons, zero_shot)
    os.chmod(str(summary_path), 0o600)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "source": {
            "retention_csv": {
                "path": str(args.retention_csv.resolve()),
                "sha256": source_hashes["retention_csv"],
            },
            "sequence_table": {
                "path": str(args.sequence_table.resolve()),
                "sha256": source_hashes["sequence_table"],
            },
            "features": {
                "path": str(args.features.resolve()),
                "sha256": source_hashes["features"],
            },
        },
        "code": {"path": str(script_path), "sha256": script_hash},
        "configuration": {
            "models": list(MODEL_NAMES),
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
