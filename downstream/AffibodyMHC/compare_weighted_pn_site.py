#!/usr/bin/env python
"""Converged weighted-PN control for the additive Affibody site model.

This experiment isolates the ``eta=0`` endpoint of the PNU objective.  At
eta=0 no unlabeled row contributes: the empirical logistic risk is

    pi * mean_P(loss_positive) + (1-pi) * mean_N(loss_negative).

The script fits that convex model with scikit-learn/liblinear and compares it
with the established balanced-PN objective (pi=0.5).  It deliberately does
not load or use the U pool.  C is selected only by pooled AUROC on the same
five deterministic double-identity-cold weak-label folds used by the current
site baseline.  The four values of pi remain separate sensitivity arms; an
additional flag records which arm would be chosen by weak AUROC alone.

The one-hot encoder is fitted separately inside every weak training fold and
is refitted on all weak P/N rows for the final model.  Retention data are not
loaded until every weak-label configuration choice has been frozen.
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
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    load_retention_table,
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.evaluate_selection_weak_baseline import (
    DEFAULT_C_GRID,
    LIBRARIES,
    add_identity_blocks,
    audit_identity_cold_fold,
    identity_blocked_indices,
    load_and_validate_weak_inputs,
    make_encoder,
    primary_training_frame,
    site_code_matrix,
    site_feature_names,
)


DEFAULT_PI_GRID = (0.02, 0.05, 0.10, 0.15)
DEFAULT_FOLDS = 5
DEFAULT_MIN_NEGATIVE_R001_COUNT = 3
RETENTION_BINDER_THRESHOLD = 75.0
SOLVER = "liblinear"
SOLVER_TOLERANCE = 1e-12
SOLVER_MAX_ITERATIONS = 20000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _atomic_csv(frame, path):
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    frame.to_csv(str(temporary), index=False)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(path))


def _atomic_text(text, path):
    temporary = Path(str(path) + ".tmp-{}".format(os.getpid()))
    with open(str(temporary), "w") as handle:
        handle.write(text)
    os.chmod(str(temporary), 0o600)
    os.replace(str(temporary), str(path))


def class_mass_weights(labels, positive_mass):
    """Return weights summing to n with requested positive/negative mass.

    If n is the number of rows, every positive gets n*pi/n_P and every
    negative gets n*(1-pi)/n_N.  Therefore the positive weights sum to n*pi,
    the negative weights sum to n*(1-pi), and total weight remains n.  Keeping
    the total fixed makes a shared scikit-learn C comparable across priors.
    """
    labels = np.asarray(labels, dtype=int)
    pi = float(positive_mass)
    _require(math.isfinite(pi) and 0.0 < pi < 1.0, "positive mass must lie in (0,1)")
    _require(set(labels.tolist()) == {0, 1}, "weights require both classes")
    n = int(len(labels))
    n_positive = int(labels.sum())
    n_negative = int(n - n_positive)
    weights = np.where(
        labels == 1,
        float(n) * pi / float(n_positive),
        float(n) * (1.0 - pi) / float(n_negative),
    ).astype(np.float64)
    _require(np.isclose(weights.sum(), float(n), rtol=0.0, atol=1e-9), "weight total changed")
    _require(
        np.isclose(weights[labels == 1].sum() / float(n), pi, rtol=0.0, atol=1e-12),
        "positive class mass mismatch",
    )
    return weights


def make_converged_logistic(c_value):
    c_value = float(c_value)
    _require(math.isfinite(c_value) and c_value > 0.0, "C must be positive")
    return LogisticRegression(
        C=c_value,
        penalty="l2",
        solver=SOLVER,
        fit_intercept=True,
        class_weight=None,
        random_state=0,
        max_iter=SOLVER_MAX_ITERATIONS,
        tol=SOLVER_TOLERANCE,
    )


def fit_converged_logistic(encoded, labels, c_value, positive_mass):
    labels = np.asarray(labels, dtype=int)
    weights = class_mass_weights(labels, positive_mass)
    model = make_converged_logistic(c_value)
    model.fit(encoded, labels, sample_weight=weights)
    iterations = int(model.n_iter_[0])
    _require(iterations < int(model.max_iter), "liblinear reached max_iter without convergence")
    _require(bool(np.isfinite(model.coef_).all()), "non-finite logistic coefficients")
    _require(bool(np.isfinite(model.intercept_).all()), "non-finite logistic intercept")
    return model, weights, iterations


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    _require(set(labels.tolist()) == {0, 1}, "binary metrics require both classes")
    _require(len(labels) == len(probabilities), "metric length mismatch")
    _require(bool(np.isfinite(probabilities).all()), "non-finite probabilities")
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def select_by_weak_auroc(frame, group_columns):
    """Select exactly by maximal weak AUROC; smaller C breaks exact ties."""
    selected = frame.copy()
    selected["selected_C_within_arm"] = 0
    for _, group in selected.groupby(list(group_columns), dropna=False, sort=True):
        aggregate = group.loc[group["record_type"].eq("aggregate")]
        _require(len(aggregate) > 0, "selection group lacks aggregate rows")
        best = sorted(
            aggregate.index,
            key=lambda index: (
                -float(selected.loc[index, "validation_auroc"]),
                float(selected.loc[index, "C"]),
            ),
        )[0]
        c_value = float(selected.loc[best, "C"])
        mask = pd.Series(True, index=selected.index)
        for column in group_columns:
            value = selected.loc[best, column]
            if pd.isna(value):
                mask &= selected[column].isna()
            else:
                mask &= selected[column].eq(value)
        mask &= selected["C"].eq(c_value)
        selected.loc[mask, "selected_C_within_arm"] = 1
    return selected


def _configuration_rows(library, positive_mass, arm, primary, blocked, c_grid, folds):
    raw_sites = site_code_matrix(primary, library)
    labels = primary["weak_label"].astype(int).to_numpy()
    rows = []
    for c_value in c_grid:
        pooled_labels = []
        pooled_probabilities = []
        for fold in range(int(folds)):
            indices = identity_blocked_indices(blocked, fold)
            audit_identity_cold_fold(blocked, indices, fold)
            # Fit the categorical preprocessing inside this training fold.
            encoder = make_encoder(raw_sites.shape[1])
            encoded_train = encoder.fit_transform(raw_sites[indices["train"]])
            encoded_validation = encoder.transform(raw_sites[indices["validation"]])
            y_train = labels[indices["train"]]
            y_validation = labels[indices["validation"]]
            model, weights, iterations = fit_converged_logistic(
                encoded_train, y_train, c_value, positive_mass
            )
            probabilities = model.predict_proba(encoded_validation)[:, 1]
            metrics = binary_metrics(y_validation, probabilities)
            rows.append(
                {
                    "library": library,
                    "arm": arm,
                    "positive_mass": float(positive_mass),
                    "C": float(c_value),
                    "record_type": "fold",
                    "fold": int(fold),
                    "n_train": int(len(indices["train"])),
                    "n_guard": int(len(indices["guard"])),
                    "n_validation": int(len(indices["validation"])),
                    "train_positive": int(y_train.sum()),
                    "train_negative": int((y_train == 0).sum()),
                    "sample_weight_total": float(weights.sum()),
                    "sample_weight_positive_mass": float(weights[y_train == 1].sum() / weights.sum()),
                    "solver_iterations": iterations,
                    "solver_converged": 1,
                    "validation_log_loss": metrics["log_loss"],
                    "validation_auroc": metrics["auroc"],
                    "validation_average_precision": metrics["average_precision"],
                }
            )
            pooled_labels.extend(y_validation.tolist())
            pooled_probabilities.extend(probabilities.tolist())
        aggregate = binary_metrics(pooled_labels, pooled_probabilities)
        matching = [row for row in rows if row["C"] == float(c_value)]
        rows.append(
            {
                "library": library,
                "arm": arm,
                "positive_mass": float(positive_mass),
                "C": float(c_value),
                "record_type": "aggregate",
                "fold": -1,
                "n_train": int(sum(row["n_train"] for row in matching)),
                "n_guard": int(sum(row["n_guard"] for row in matching)),
                "n_validation": int(len(pooled_labels)),
                "train_positive": int(sum(row["train_positive"] for row in matching)),
                "train_negative": int(sum(row["train_negative"] for row in matching)),
                "sample_weight_total": float(sum(row["sample_weight_total"] for row in matching)),
                "sample_weight_positive_mass": float(positive_mass),
                "solver_iterations": int(max(row["solver_iterations"] for row in matching)),
                "solver_converged": 1,
                "validation_log_loss": aggregate["log_loss"],
                "validation_auroc": aggregate["auroc"],
                "validation_average_precision": aggregate["average_precision"],
            }
        )
    return pd.DataFrame(rows)


def retention_metrics(retention, probabilities):
    observed = retention["target_retention"].to_numpy(dtype=float)
    binders = retention["target_binder"].astype(int).to_numpy()
    probabilities = np.asarray(probabilities, dtype=float)
    _require(len(observed) == len(probabilities), "retention length mismatch")
    global_rho = float(spearmanr(observed, probabilities)[0])
    within = []
    for _, indices in retention.groupby("peptide_uid", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        if np.unique(observed[indices]).size < 2 or np.unique(probabilities[indices]).size < 2:
            continue
        within.append(float(spearmanr(observed[indices], probabilities[indices])[0]))
    return {
        "n": int(len(retention)),
        "binders_ge_75": int(binders.sum()),
        "auroc": float(roc_auc_score(binders, probabilities)),
        "average_precision": float(average_precision_score(binders, probabilities)),
        "global_spearman": global_rho,
        "within_peptide_spearman": float(np.mean(within)) if within else float("nan"),
        "within_peptide_groups": int(len(within)),
    }


def _arm_name(pi):
    if float(pi) == 0.5:
        return "balanced_pn"
    return "prior_weighted_pn_pi{:g}".format(float(pi))


def _report(selected, metrics):
    lines = [
        "# Exact weighted-PN designed-position comparison",
        "",
        "This is a convex-control check for the eta=0 PNU result. Eta=0 does not use U; "
        "it is ordinary P-versus-N logistic regression with a chosen positive class mass. "
        "All C choices below were made by weak-label AUROC before retention was loaded.",
        "",
        "| Library | Training objective | Positive class mass | Selected C | Weak AUROC | Solver iterations (final) | Retention AUROC | Retention AP | Global Spearman | Within-peptide Spearman |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    merged = selected.merge(
        metrics,
        on=["library", "arm", "positive_mass", "C"],
        validate="one_to_one",
    )
    for row in merged.sort_values(["library", "positive_mass"], ascending=[True, False]).itertuples():
        label = "balanced PN" if row.arm == "balanced_pn" else "prior-weighted PN"
        lines.append(
            "| {} | {} | {:.0%} | {:g} | {:.4f} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row.library,
                label,
                row.positive_mass,
                row.C,
                row.validation_auroc,
                int(row.final_solver_iterations),
                row.auroc,
                row.average_precision,
                row.global_spearman,
                row.within_peptide_spearman,
            )
        )
    lines.extend(
        [
            "",
            "For n training rows, positive rows receive weight n*pi/nP and negative rows "
            "receive n*(1-pi)/nN. The weights sum to n, so liblinear C has the same scale "
            "in every arm. Up to liblinear's penalized synthetic intercept, dividing its "
            "objective by C*n gives the requested empirical risk plus an L2 penalty of "
            "||w||^2/(2*C*n).",
            "",
            "The four assumed priors are sensitivity analyses. Retention did not choose a "
            "prior, C, preprocessing rule, or model.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-label-dir", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES, default=list(LIBRARIES))
    parser.add_argument("--pi-grid", nargs="+", type=float, default=list(DEFAULT_PI_GRID))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--min-negative-r001-count", type=int, default=DEFAULT_MIN_NEGATIVE_R001_COUNT)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(int(args.folds) == DEFAULT_FOLDS, "this comparison preserves the established five folds")
    _require(
        int(args.min_negative_r001_count) == DEFAULT_MIN_NEGATIVE_R001_COUNT,
        "this comparison preserves the established three-read negative rule",
    )
    libraries = tuple(dict.fromkeys(args.libraries))
    c_grid = tuple(sorted(set(float(value) for value in args.c_grid)))
    pi_grid = tuple(sorted(set(float(value) for value in args.pi_grid)))
    _require(c_grid and all(math.isfinite(value) and value > 0.0 for value in c_grid), "invalid C grid")
    _require(pi_grid and all(math.isfinite(value) and 0.0 < value < 1.0 for value in pi_grid), "invalid pi grid")

    weak_labels, _, weak_manifest, weak_paths = load_and_validate_weak_inputs(args.weak_label_dir)
    source_hashes_before = {
        "weak_labels": sha256_file(weak_paths["weak_labels"]),
        "weak_manifest": sha256_file(weak_paths["manifest"]),
        "retention_csv": sha256_file(args.retention_csv),
    }
    validation_blocks = []
    primary_by_library = {}
    blocked_by_library = {}
    for library in libraries:
        primary = primary_training_frame(
            weak_labels,
            library,
            min_negative_r001_count=args.min_negative_r001_count,
        )
        blocked = add_identity_blocks(primary, library, args.folds)
        primary_by_library[library] = primary
        blocked_by_library[library] = blocked
        for pi in (0.5,) + pi_grid:
            validation_blocks.append(
                _configuration_rows(
                    library=library,
                    positive_mass=pi,
                    arm=_arm_name(pi),
                    primary=primary,
                    blocked=blocked,
                    c_grid=c_grid,
                    folds=args.folds,
                )
            )
    validation = pd.concat(validation_blocks, ignore_index=True)
    validation = select_by_weak_auroc(
        validation, group_columns=("library", "arm", "positive_mass")
    )
    selected = validation.loc[
        validation["record_type"].eq("aggregate")
        & validation["selected_C_within_arm"].eq(1)
    ].copy()
    _require(len(selected) == len(libraries) * (1 + len(pi_grid)), "selected-arm count mismatch")
    selected["selected_prior_by_weak_auroc"] = 0
    for library in libraries:
        prior_rows = selected.loc[
            selected["library"].eq(library) & selected["arm"].ne("balanced_pn")
        ]
        best = sorted(
            prior_rows.index,
            key=lambda index: (
                -float(selected.loc[index, "validation_auroc"]),
                float(selected.loc[index, "positive_mass"]),
                float(selected.loc[index, "C"]),
            ),
        )[0]
        selected.loc[best, "selected_prior_by_weak_auroc"] = 1

    # Configurations are now frozen. Retention is loaded only for final evaluation.
    retention_all = load_retention_table(args.retention_csv)
    metric_rows = []
    prediction_blocks = []
    coefficient_blocks = []
    for row in selected.itertuples():
        library = row.library
        primary = primary_by_library[library]
        raw_sites = site_code_matrix(primary, library)
        labels = primary["weak_label"].astype(int).to_numpy()
        encoder = make_encoder(raw_sites.shape[1])
        encoded = encoder.fit_transform(raw_sites)
        model, weights, iterations = fit_converged_logistic(
            encoded, labels, row.C, row.positive_mass
        )
        held = retention_all.loc[
            retention_all["library"].eq(library)
            & retention_all["target_retention"].notna()
        ].copy()
        held = held.sort_values("pair_uid").reset_index(drop=True)
        held_sites = site_code_matrix(
            held,
            library,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
        )
        probabilities = model.predict_proba(encoder.transform(held_sites))[:, 1]
        metric = retention_metrics(held, probabilities)
        metric.update(
            {
                "library": library,
                "arm": row.arm,
                "positive_mass": float(row.positive_mass),
                "C": float(row.C),
                "final_solver_iterations": iterations,
                "final_solver_converged": 1,
                "final_sample_weight_total": float(weights.sum()),
                "final_sample_weight_positive_mass": float(weights[labels == 1].sum() / weights.sum()),
            }
        )
        metric_rows.append(metric)
        prediction = held[
            ["pair_uid", "peptide_uid", "affibody_uid", "library", "target_retention", "target_binder"]
        ].copy()
        prediction.insert(0, "arm", row.arm)
        prediction.insert(1, "positive_mass", float(row.positive_mass))
        prediction.insert(2, "C", float(row.C))
        prediction["predicted_probability"] = probabilities
        prediction_blocks.append(prediction)

        feature_names = site_feature_names(library)
        offset = 0
        for name, categories in zip(feature_names, encoder.categories_):
            for category in categories:
                coefficient_blocks.append(
                    {
                        "library": library,
                        "arm": row.arm,
                        "positive_mass": float(row.positive_mass),
                        "C": float(row.C),
                        "feature": name,
                        "residue": str(category),
                        "coefficient": float(model.coef_[0, offset]),
                        "intercept": float(model.intercept_[0]),
                    }
                )
                offset += 1
        _require(offset == model.coef_.shape[1], "coefficient width mismatch")

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_blocks)
    split_audit = pd.concat(
        [
            blocked_by_library[library][
                ["pair_uid", "peptide_uid", "affibody_uid", "weak_label", "peptide_block", "affibody_block"]
            ].assign(library=library)
            for library in libraries
        ],
        ignore_index=True,
    )
    split_audit = split_audit[
        ["library", "pair_uid", "peptide_uid", "affibody_uid", "weak_label", "peptide_block", "affibody_block"]
    ]

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    outputs = {
        "weak_validation.csv": validation,
        "selected_configurations.csv": selected,
        "retention_metrics.csv": metrics,
        "retention_predictions.csv": predictions,
        "model_coefficients.csv": coefficients,
        "split_audit.csv": split_audit,
    }
    output_paths = {}
    for name, frame in outputs.items():
        path = output_dir / name
        _atomic_csv(frame, path)
        output_paths[name] = path
    report_path = output_dir / "weighted_pn_report.md"
    _atomic_text(_report(selected, metrics), report_path)
    output_paths[report_path.name] = report_path

    source_hashes_after = {
        "weak_labels": sha256_file(weak_paths["weak_labels"]),
        "weak_manifest": sha256_file(weak_paths["manifest"]),
        "retention_csv": sha256_file(args.retention_csv),
    }
    _require(source_hashes_before == source_hashes_after, "an input changed during the run")
    script_path = Path(__file__).resolve()
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "analysis_status": "retrospective_exploratory_convex_control",
        "configuration": {
            "model": "designed-position one-hot additive logistic regression",
            "objectives": "balanced PN plus eta=0 prior-weighted PN",
            "positive_mass_grid": list(pi_grid),
            "balanced_positive_mass": 0.5,
            "C_grid": list(c_grid),
            "folds": int(args.folds),
            "split": "established deterministic double-identity-cold weak folds",
            "preprocessing": "one-hot encoder fit inside each weak training fold",
            "selection": "pooled weak-validation AUROC only; smaller C breaks exact ties",
            "retention_usage": "loaded only after all weak-label choices were frozen",
            "negative_rule": "R001 count >= 3 and absent from R002-R014",
            "solver": SOLVER,
            "solver_tolerance": SOLVER_TOLERANCE,
            "solver_max_iterations": SOLVER_MAX_ITERATIONS,
            "weight_formula": "wP=n*pi/nP; wN=n*(1-pi)/nN; sum(w)=n",
            "C_mapping": "liblinear objective divided by C*n equals weighted empirical logistic risk plus L2/(2*C*n), including liblinear's synthetic intercept coefficient",
            "retention_binder_threshold": RETENTION_BINDER_THRESHOLD,
        },
        "rows": {
            library: {
                "positive": int(primary_by_library[library]["weak_label"].astype(int).sum()),
                "negative": int(primary_by_library[library]["weak_label"].eq(0).sum()),
                "retention": int(retention_all.loc[retention_all["library"].eq(library) & retention_all["target_retention"].notna()].shape[0]),
            }
            for library in libraries
        },
        "source_weak_top_fraction": weak_manifest["configuration"]["top_fraction"],
        "sources": {
            "weak_labels": {"path": str(weak_paths["weak_labels"].resolve()), "sha256": source_hashes_before["weak_labels"]},
            "weak_manifest": {"path": str(weak_paths["manifest"].resolve()), "sha256": source_hashes_before["weak_manifest"]},
            "retention_csv": {"path": str(args.retention_csv.resolve()), "sha256": source_hashes_before["retention_csv"]},
        },
        "code": {str(script_path): sha256_file(script_path)},
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(output_paths.items())
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "privacy": {"directory": "0700", "files": "0600", "row_level_outputs_use_opaque_ids": True},
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "elapsed_seconds": manifest["elapsed_seconds"],
                "selected_arms": int(len(selected)),
                "all_solver_fits_converged": bool(validation["solver_converged"].eq(1).all() and metrics["final_solver_converged"].eq(1).all()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return selected, metrics


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
