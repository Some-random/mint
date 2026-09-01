#!/usr/bin/env python
"""CPU ablations for selection-trained, position-additive Affibody models.

This runner deliberately keeps the 227 measured retention values out of every
training decision.  It compares fold-local Affibody-promiscuity filtering,
three class-imbalance treatments, separate versus pooled library models, four
weak-label CV regimes, and four fixed transfers to the retention panel.

"Cold" is enforced with the short mutation code globally across LibA/LibB,
not with the library-scoped UIDs in the weak-label table.  The provider uses a
common chain-1 template, so this prevents a peptide code held out in LibA from
leaking through LibB.  Full-chain hashes would be preferable if the templates
ever diverge; this conservative code-level rule is recorded in the manifest.

Promiscuity is an exploratory selection-label diagnostic: inside each fit, an
Affibody is removed in full when it has positive weak labels with at least the
configured number of distinct peptides.  No retention-derived cleaning is
performed.  Reported PR metric is sklearn average precision (AP), not a
trapezoidal interpolation of the precision-recall curve.
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
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    LIBRARY_SPECS,
    load_retention_table,
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.evaluate_selection_weak_baseline import (
    load_and_validate_weak_inputs,
)


LIBRARIES = ("LibA", "LibB")
CLEANING_VARIANTS = (
    "none",
    "round_support",
    "affibody_breadth",
    "round_support_and_affibody_breadth",
)
IMBALANCE_VARIANTS = ("natural", "balanced_loss", "downsample_1to1")
MODEL_SCOPES = ("separate", "pooled_shared", "pooled_conditioned")
CV_SCHEMES = ("random_pair", "peptide_cold", "affibody_cold", "double_cold")
TRANSFER_REGIMES = ("pair_cold", "peptide_cold", "affibody_cold", "double_cold")
REQUIRED_EXTRA_WEAK_COLUMNS = (
    "shares_retention_peptide",
    "shares_retention_affibody",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stable_hash(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _write_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def base_weak_frame(labels, min_negative_r001_count=3):
    """Return the on-design label universe before any retention-cold filter."""
    missing = set(REQUIRED_EXTRA_WEAK_COLUMNS).difference(labels.columns)
    _require(not missing, "weak labels missing overlap columns {}".format(sorted(missing)))
    output = labels.copy()
    label_ok = output["weak_label"].eq(1) | (
        output["weak_label"].eq(0)
        & output["r001_count"].ge(int(min_negative_r001_count))
    )
    output = output.loc[
        output["within_declared_library_alphabet"].eq(1) & label_ok
    ].copy()
    output = output.sort_values("pair_uid").reset_index(drop=True)
    _require(set(output["weak_label"].astype(int)) == {0, 1}, "base weak data need both labels")
    _require(not bool(output["pair_uid"].duplicated().any()), "duplicate weak pair")
    return output


def global_code_key(frame, partner):
    """Global code identity; unlike provider UIDs, it is not library-scoped."""
    column = "pep" if partner == "peptide" else "aff"
    return frame[column].astype(str)


def fixed_transfer_training_frame(base, retention, regime):
    """Filter weak rows against the union of both retention libraries."""
    _require(regime in TRANSFER_REGIMES, "unknown transfer regime {}".format(regime))
    retention_peptides = set(retention["peptide_design_code"].astype(str))
    retention_affibodies = set(retention["affibody_design_code"].astype(str))
    retention_pairs = set(
        retention["peptide_design_code"].astype(str).str.cat(
            retention["affibody_design_code"].astype(str), sep="|"
        )
    )
    pair_key = base["pep"].astype(str).str.cat(base["aff"].astype(str), sep="|")
    mask = ~pair_key.isin(retention_pairs)
    if regime in ("peptide_cold", "double_cold"):
        mask &= ~base["pep"].astype(str).isin(retention_peptides)
    if regime in ("affibody_cold", "double_cold"):
        mask &= ~base["aff"].astype(str).isin(retention_affibodies)
    output = base.loc[mask].copy().reset_index(drop=True)
    _require(set(output["weak_label"].astype(int)) == {0, 1}, "transfer train lacks a class")
    return output


def site_feature_dicts(frame, model_scope):
    """Map codes to physically aligned designed-position feature keys."""
    _require(model_scope in MODEL_SCOPES, "unknown model scope {}".format(model_scope))
    records = []
    for row in frame.itertuples(index=False):
        library = str(row.library)
        spec = LIBRARY_SPECS[library]
        peptide = str(row.pep)
        affibody = str(row.aff)
        _require(len(peptide) == spec["pep_length"], "bad peptide code length")
        _require(len(affibody) == spec["aff_length"], "bad Affibody code length")
        physical = {}
        for position, residue in zip(spec["pep_positions"], peptide):
            physical["pep_p{}={}".format(position, residue)] = 1.0
        for position, residue in zip(spec["aff_positions"], affibody):
            physical["aff_p{}={}".format(position, residue)] = 1.0
        if model_scope == "separate":
            records.append(physical)
            continue
        features = dict(physical)
        features["library={}".format(library)] = 1.0
        if model_scope == "pooled_conditioned":
            for key in physical:
                features["{}|{}".format(library, key)] = 1.0
        records.append(features)
    return records


def _balanced_blocks(frame, n_folds, seed):
    """Deterministic approximately stratified random-pair assignment."""
    block = np.empty(len(frame), dtype=int)
    strata = frame["library"].astype(str).str.cat(
        frame["weak_label"].astype(str), sep="|"
    )
    for _, indices in frame.groupby(strata, sort=True).groups.items():
        ordered = sorted(
            list(indices),
            key=lambda index: (_stable_hash(seed, "pair", frame.at[index, "pair_uid"]), index),
        )
        for offset, index in enumerate(ordered):
            block[index] = offset % int(n_folds)
    return block


def make_weak_cv_splits(frame, scheme, n_folds, seed):
    """Create complete OOF splits; double-cold uses all block intersections."""
    _require(scheme in CV_SCHEMES, "unknown CV scheme {}".format(scheme))
    _require(int(n_folds) >= 2, "n_folds must be >=2")
    frame = frame.reset_index(drop=True)
    indices = np.arange(len(frame), dtype=int)
    if scheme == "random_pair":
        blocks = _balanced_blocks(frame, n_folds, seed)
        return [
            {
                "fold": "f{:02d}".format(fold),
                "train": indices[blocks != fold],
                "guard": np.asarray([], dtype=int),
                "test": indices[blocks == fold],
            }
            for fold in range(int(n_folds))
        ]

    peptide_block = np.asarray(
        [_stable_hash(seed, "peptide", code) % int(n_folds) for code in global_code_key(frame, "peptide")]
    )
    affibody_block = np.asarray(
        [_stable_hash(seed, "affibody", code) % int(n_folds) for code in global_code_key(frame, "affibody")]
    )
    splits = []
    if scheme in ("peptide_cold", "affibody_cold"):
        blocks = peptide_block if scheme == "peptide_cold" else affibody_block
        for fold in range(int(n_folds)):
            splits.append(
                {
                    "fold": "f{:02d}".format(fold),
                    "train": indices[blocks != fold],
                    "guard": np.asarray([], dtype=int),
                    "test": indices[blocks == fold],
                }
            )
        return splits

    for peptide_fold in range(int(n_folds)):
        for affibody_fold in range(int(n_folds)):
            peptide_held = peptide_block == peptide_fold
            affibody_held = affibody_block == affibody_fold
            test = peptide_held & affibody_held
            guard = peptide_held ^ affibody_held
            train = ~peptide_held & ~affibody_held
            splits.append(
                {
                    "fold": "p{:02d}_a{:02d}".format(peptide_fold, affibody_fold),
                    "train": indices[train],
                    "guard": indices[guard],
                    "test": indices[test],
                }
            )
    return splits


def audit_split(frame, split, scheme):
    train = frame.iloc[split["train"]]
    test = frame.iloc[split["test"]]
    _require(len(test) > 0, "empty test fold {}".format(split["fold"]))
    _require(set(train["pair_uid"]).isdisjoint(set(test["pair_uid"])), "pair leakage")
    if scheme in ("peptide_cold", "double_cold"):
        _require(set(train["pep"]).isdisjoint(set(test["pep"])), "peptide code leakage")
    if scheme in ("affibody_cold", "double_cold"):
        _require(set(train["aff"]).isdisjoint(set(test["aff"])), "Affibody code leakage")


def apply_promiscuity_cleaning(
    frame, variant, positive_breadth, positive_round_count_min=3
):
    """Apply only fit-local positive-support and Affibody-breadth filters."""
    _require(variant in CLEANING_VARIANTS, "unknown cleaning {}".format(variant))
    frame = frame.copy()
    original = frame.copy()
    if variant == "none":
        return frame, {
            "promiscuous_affibodies": 0,
            "rows_removed_cleaning": 0,
            "positive_rows_removed_cleaning": 0,
            "positive_rows_removed_round_support": 0,
        }
    round_removed = pd.Series(False, index=frame.index)
    if variant in ("round_support", "round_support_and_affibody_breadth"):
        _require(int(positive_round_count_min) >= 1, "round count floor must be positive")
        round_removed = frame["weak_label"].eq(1) & ~(
            frame["r009_count"].ge(int(positive_round_count_min))
            & frame["r010_count"].ge(int(positive_round_count_min))
        )
        frame = frame.loc[~round_removed].copy()

    flagged = set()
    breadth_removed = pd.Series(False, index=frame.index)
    if variant in ("affibody_breadth", "round_support_and_affibody_breadth"):
        positive = frame.loc[frame["weak_label"].eq(1)]
        breadth = positive.groupby(["library", "aff"], sort=True)["pep"].nunique()
        flagged = set(breadth.loc[breadth.ge(int(positive_breadth))].index)
        breadth_removed = pd.Series(
            [
                (library, affibody) in flagged
                for library, affibody in zip(frame["library"], frame["aff"])
            ],
            index=frame.index,
        )
        frame = frame.loc[~breadth_removed].copy()
    output = frame
    _require(set(output["weak_label"].astype(int)) == {0, 1}, "cleaning removed a class")
    return output, {
        "promiscuous_affibodies": int(len(flagged)),
        "rows_removed_cleaning": int(len(original) - len(output)),
        "positive_rows_removed_cleaning": int(
            original["weak_label"].sum() - output["weak_label"].sum()
        ),
        "positive_rows_removed_round_support": int(round_removed.sum()),
    }


def prepare_imbalance(frame, variant, seed, salt):
    """Return fit rows and optional fold-local sample weights."""
    _require(variant in IMBALANCE_VARIANTS, "unknown imbalance {}".format(variant))
    frame = frame.copy()
    if variant == "natural":
        return frame, None
    if variant == "balanced_loss":
        weights = np.zeros(len(frame), dtype=float)
        libraries = sorted(frame["library"].unique())
        for library in libraries:
            for label in (0, 1):
                mask = frame["library"].eq(library) & frame["weak_label"].eq(label)
                count = int(mask.sum())
                _require(count > 0, "library/class cell empty for balanced loss")
                weights[mask.to_numpy()] = float(len(frame)) / (
                    len(libraries) * 2.0 * count
                )
        return frame, weights

    kept = []
    for library, subset in frame.groupby("library", sort=True):
        counts = subset["weak_label"].value_counts()
        _require(set(counts.index.astype(int)) == {0, 1}, "downsample cell lacks class")
        target = int(counts.min())
        for label in (0, 1):
            candidates = subset.loc[subset["weak_label"].eq(label)].copy()
            candidates["_order"] = [
                _stable_hash(seed, salt, uid) for uid in candidates["pair_uid"]
            ]
            kept.append(candidates.sort_values(["_order", "pair_uid"]).head(target).drop(columns="_order"))
    output = pd.concat(kept, ignore_index=True).sort_values("pair_uid").reset_index(drop=True)
    return output, None


def fit_predict(train, test, model_scope, imbalance, cleaning, c_value,
                positive_breadth, positive_round_count_min, seed, salt):
    cleaned, clean_audit = apply_promiscuity_cleaning(
        train, cleaning, positive_breadth, positive_round_count_min
    )
    fit_frame, sample_weight = prepare_imbalance(cleaned, imbalance, seed, salt)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    x_train = vectorizer.fit_transform(site_feature_dicts(fit_frame, model_scope))
    x_test = vectorizer.transform(site_feature_dicts(test, model_scope))
    model = LogisticRegression(
        C=float(c_value), penalty="l2", solver="liblinear", fit_intercept=True,
        random_state=int(seed), max_iter=2000, tol=1e-8,
    )
    y = fit_frame["weak_label"].astype(int).to_numpy()
    model.fit(x_train, y, sample_weight=sample_weight)
    audit = dict(clean_audit)
    audit.update({
        "train_rows_before_cleaning": int(len(train)),
        "train_rows_after_cleaning": int(len(cleaned)),
        "fit_rows": int(len(fit_frame)),
        "fit_positive": int(y.sum()),
        "fit_negative": int((y == 0).sum()),
        "n_features": int(x_train.shape[1]),
    })
    return model.predict_proba(x_test)[:, 1], audit


def binary_metrics(y_true, score, probability_score=True):
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    result = {
        "n": int(len(y)), "positive": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "average_precision": np.nan, "ap_lift_over_prevalence": np.nan,
        "auroc": np.nan, "brier": np.nan, "log_loss": np.nan,
    }
    if len(y) and set(y) == {0, 1}:
        average_precision = float(average_precision_score(y, score))
        result.update({
            "average_precision": average_precision,
            "ap_lift_over_prevalence": average_precision - float(y.mean()),
            "auroc": float(roc_auc_score(y, score)),
        })
        if probability_score:
            _require(
                bool(((score >= 0.0) & (score <= 1.0)).all()),
                "probability score lies outside [0, 1]",
            )
            result.update({
                "brier": float(brier_score_loss(y, score)),
                "log_loss": float(log_loss(y, score, labels=[0, 1])),
            })
    return result


def _safe_correlation(x, y, method):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    value = spearmanr(x, y)[0] if method == "spearman" else pearsonr(x, y)[0]
    return float(value)


def _two_way_residual(frame, values):
    if {"peptide_design_code", "affibody_design_code"}.issubset(frame.columns):
        partner_columns = ["peptide_design_code", "affibody_design_code"]
    elif {"pep", "aff"}.issubset(frame.columns):
        partner_columns = ["pep", "aff"]
    else:
        raise ValueError("two-way residual needs peptide and Affibody identities")
    design = pd.get_dummies(
        frame[partner_columns].astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy()
    design = np.column_stack([np.ones(len(frame)), design])
    values = np.asarray(values, dtype=float)
    coefficient = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design.dot(coefficient)


def retention_metric_rows(predictions, metadata):
    """Global, conditional-ranking, and two-way interaction diagnostics."""
    rows = []
    scopes = [(library, predictions.loc[predictions["library"].eq(library)]) for library in LIBRARIES]
    scopes.append(("Combined", predictions))
    for library, subset in scopes:
        if not len(subset):
            continue
        binary = binary_metrics(subset["target_binder"], subset["score"])
        row = dict(metadata, library=library, evaluation_scope="global", n_groups=np.nan)
        row.update(binary)
        row["spearman_rho"] = _safe_correlation(subset["target_retention"], subset["score"], "spearman")
        row["pearson_r"] = _safe_correlation(subset["target_retention"], subset["score"], "pearson")
        rows.append(row)

        for group_name, group_column in (
            ("within_peptide_macro", "peptide_uid"),
            ("within_affibody_macro", "affibody_uid"),
        ):
            group_rows = []
            for _, group in subset.groupby(group_column, sort=True):
                if set(group["target_binder"].astype(int)) != {0, 1}:
                    continue
                metrics = binary_metrics(group["target_binder"], group["score"])
                metrics["spearman_rho"] = _safe_correlation(group["target_retention"], group["score"], "spearman")
                metrics["pearson_r"] = _safe_correlation(group["target_retention"], group["score"], "pearson")
                group_rows.append(metrics)
            row = dict(metadata, library=library, evaluation_scope=group_name)
            row["n"] = int(sum(item["n"] for item in group_rows))
            row["n_groups"] = int(len(group_rows))
            for column in ("positive", "prevalence", "average_precision", "ap_lift_over_prevalence", "auroc", "brier", "log_loss", "spearman_rho", "pearson_r"):
                row[column] = float(np.nanmean([item[column] for item in group_rows])) if group_rows else np.nan
            rows.append(row)

        pieces = []
        for _, library_frame in subset.groupby("library", sort=True):
            score_residual = _two_way_residual(library_frame, library_frame["score"])
            retention_residual = _two_way_residual(library_frame, library_frame["target_retention"])
            piece = library_frame.copy()
            piece["_score_residual"] = score_residual
            piece["_retention_residual"] = retention_residual
            pieces.append(piece)
        residual = pd.concat(pieces, ignore_index=True)
        binary = binary_metrics(
            residual["target_binder"], residual["_score_residual"],
            probability_score=False,
        )
        row = dict(metadata, library=library, evaluation_scope="two_way_interaction_residual", n_groups=np.nan)
        row.update(binary)
        row["spearman_rho"] = _safe_correlation(residual["_retention_residual"], residual["_score_residual"], "spearman")
        row["pearson_r"] = _safe_correlation(residual["_retention_residual"], residual["_score_residual"], "pearson")
        rows.append(row)
    return rows


def _fit_by_scope(train, test, scope, imbalance, cleaning, c_value,
                  positive_breadth, positive_round_count_min, seed, salt):
    predictions = np.empty(len(test), dtype=float)
    audits = []
    units = LIBRARIES if scope == "separate" else ("Pooled",)
    for unit in units:
        if unit == "Pooled":
            unit_train, unit_test = train, test
            positions = np.arange(len(test))
        else:
            unit_train = train.loc[train["library"].eq(unit)]
            mask = test["library"].eq(unit).to_numpy()
            unit_test = test.loc[mask]
            positions = np.flatnonzero(mask)
            if not len(unit_test):
                continue
        score, audit = fit_predict(
            unit_train, unit_test, scope, imbalance, cleaning, c_value,
            positive_breadth, positive_round_count_min, seed,
            "{}|{}".format(salt, unit),
        )
        predictions[positions] = score
        audit["fit_unit"] = unit
        audits.append(audit)
    return predictions, audits


def run_weak_cv(base, args):
    metric_rows, fold_rows, audit_rows = [], [], []
    for scope in args.model_scopes:
        for cleaning in args.cleaning_variants:
            for imbalance in args.imbalance_variants:
                for scheme in args.cv_schemes:
                    splits = make_weak_cv_splits(base, scheme, args.folds, args.seed)
                    scores = np.full(len(base), np.nan, dtype=float)
                    seen = np.zeros(len(base), dtype=int)
                    for split in splits:
                        if not len(split["test"]):
                            continue
                        audit_split(base, split, scheme)
                        train = base.iloc[split["train"]].copy()
                        test = base.iloc[split["test"]].copy()
                        predicted, fit_audits = _fit_by_scope(
                            train, test, scope, imbalance, cleaning, args.c_value,
                            args.promiscuity_positive_breadth,
                            args.positive_round_count_min, args.seed,
                            "cv|{}|{}|{}|{}".format(scope, cleaning, imbalance, split["fold"]),
                        )
                        scores[split["test"]] = predicted
                        seen[split["test"]] += 1
                        for library in list(LIBRARIES) + ["Combined"]:
                            mask = np.ones(len(test), dtype=bool) if library == "Combined" else test["library"].eq(library).to_numpy()
                            if not mask.any():
                                continue
                            metrics = binary_metrics(test.loc[mask, "weak_label"], predicted[mask])
                            metrics.update({
                                "model_scope": scope, "cleaning": cleaning,
                                "imbalance": imbalance, "cv_scheme": scheme,
                                "fold": split["fold"], "library": library,
                            })
                            fold_rows.append(metrics)
                        for fit_audit in fit_audits:
                            fit_audit.update({
                                "model_scope": scope, "cleaning": cleaning,
                                "imbalance": imbalance, "cv_scheme": scheme,
                                "fold": split["fold"], "n_test": int(len(test)),
                                "n_guard": int(len(split["guard"])),
                            })
                            audit_rows.append(fit_audit)
                    _require(bool((seen == 1).all()), "{} OOF coverage is not exactly once".format(scheme))
                    for library in list(LIBRARIES) + ["Combined"]:
                        mask = np.ones(len(base), dtype=bool) if library == "Combined" else base["library"].eq(library).to_numpy()
                        metrics = binary_metrics(base.loc[mask, "weak_label"], scores[mask])
                        metrics.update({
                            "model_scope": scope, "cleaning": cleaning,
                            "imbalance": imbalance, "cv_scheme": scheme,
                            "library": library,
                        })
                        metric_rows.append(metrics)
    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows), pd.DataFrame(audit_rows)


def _retention_exposure(test, train):
    peptides = set(train["pep"].astype(str))
    affibodies = set(train["aff"].astype(str))
    categories = []
    for peptide, affibody in zip(test["pep"], test["aff"]):
        p_seen, a_seen = peptide in peptides, affibody in affibodies
        categories.append("both_seen" if p_seen and a_seen else "peptide_only" if p_seen else "affibody_only" if a_seen else "neither_seen")
    return categories


def run_retention_transfer(base, retention, args):
    measured = retention.loc[retention["target_retention"].notna()].copy()
    measured = measured.rename(columns={
        "peptide_design_code": "pep", "affibody_design_code": "aff"
    }).reset_index(drop=True)
    prediction_rows, metric_rows, audit_rows = [], [], []
    for scope in args.model_scopes:
        for cleaning in args.cleaning_variants:
            for imbalance in args.imbalance_variants:
                for regime in args.transfer_regimes:
                    train = fixed_transfer_training_frame(base, retention, regime)
                    score, fit_audits = _fit_by_scope(
                        train, measured, scope, imbalance, cleaning, args.c_value,
                        args.promiscuity_positive_breadth,
                        args.positive_round_count_min, args.seed,
                        "transfer|{}|{}|{}|{}".format(scope, cleaning, imbalance, regime),
                    )
                    output = measured[[
                        "pair_uid", "peptide_uid", "affibody_uid", "pep", "aff", "library",
                        "target_retention", "target_binder",
                    ]].copy()
                    output["model_scope"] = scope
                    output["cleaning"] = cleaning
                    output["imbalance"] = imbalance
                    output["transfer_regime"] = regime
                    output["score"] = score
                    output["actual_training_exposure"] = _retention_exposure(measured, train)
                    prediction_rows.append(output)
                    metadata = {
                        "model_scope": scope, "cleaning": cleaning,
                        "imbalance": imbalance, "transfer_regime": regime,
                    }
                    metric_rows.extend(retention_metric_rows(output, metadata))
                    for fit_audit in fit_audits:
                        fit_audit.update(metadata)
                        fit_audit["transfer_train_rows"] = int(len(train))
                        audit_rows.append(fit_audit)
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-label-dir", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--min-negative-r001-count", type=int, default=3)
    parser.add_argument("--promiscuity-positive-breadth", type=int, default=10)
    parser.add_argument("--positive-round-count-min", type=int, default=3)
    parser.add_argument("--cleaning-variants", nargs="+", choices=CLEANING_VARIANTS, default=list(CLEANING_VARIANTS))
    parser.add_argument("--imbalance-variants", nargs="+", choices=IMBALANCE_VARIANTS, default=list(IMBALANCE_VARIANTS))
    parser.add_argument("--model-scopes", nargs="+", choices=MODEL_SCOPES, default=list(MODEL_SCOPES))
    parser.add_argument("--cv-schemes", nargs="+", choices=CV_SCHEMES, default=list(CV_SCHEMES))
    parser.add_argument("--transfer-regimes", nargs="+", choices=TRANSFER_REGIMES, default=list(TRANSFER_REGIMES))
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.weak_label_dir.is_dir(), "missing weak-label directory")
    _require(args.retention_csv.is_file(), "missing retention CSV")
    _require(math.isfinite(args.c_value) and args.c_value > 0, "C must be positive")
    _require(args.promiscuity_positive_breadth >= 2, "breadth threshold must be >=2")
    _require(args.positive_round_count_min >= 1, "positive round count floor must be >=1")

    labels, holdout, weak_manifest, weak_paths = load_and_validate_weak_inputs(args.weak_label_dir)
    for column in REQUIRED_EXTRA_WEAK_COLUMNS:
        converted = pd.to_numeric(labels[column], errors="coerce")
        _require(bool(converted.isin([0, 1]).all()), "invalid {}".format(column))
        labels[column] = converted.astype(np.int64)
    retention = load_retention_table(args.retention_csv)
    _require(
        weak_manifest["sources"]["retention_csv"]["sha256"] == sha256_file(args.retention_csv),
        "weak-label and retention source hashes differ",
    )
    _require(set(holdout["pair_uid"]) == set(retention["pair_uid"]), "holdout audit mismatch")
    base = base_weak_frame(labels, args.min_negative_r001_count)
    _require(set(base["pair_uid"]).isdisjoint(set(retention["pair_uid"])), "exact retention pair leakage")

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    weak_metrics, weak_fold_metrics, weak_audit = run_weak_cv(base, args)
    predictions, retention_metrics, transfer_audit = run_retention_transfer(base, retention, args)
    outputs = {
        "weak_cv_metrics.csv": weak_metrics,
        "weak_cv_fold_metrics.csv": weak_fold_metrics,
        "weak_cv_training_audit.csv": weak_audit,
        "retention_predictions.csv": predictions,
        "retention_metrics.csv": retention_metrics,
        "retention_training_audit.csv": transfer_audit,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    configuration = {
        "folds": int(args.folds), "seed": int(args.seed), "C": float(args.c_value),
        "min_negative_r001_count": int(args.min_negative_r001_count),
        "promiscuity_rule": "fit-local Affibody distinct positive peptide breadth >= {}".format(args.promiscuity_positive_breadth),
        "positive_round_support_rule": "fit-local positives require R009 and R010 counts >= {}".format(args.positive_round_count_min),
        "cleaning_variants": list(args.cleaning_variants),
        "imbalance_variants": list(args.imbalance_variants),
        "model_scopes": list(args.model_scopes), "cv_schemes": list(args.cv_schemes),
        "transfer_regimes": list(args.transfer_regimes),
        "cold_identity": "global raw mutation code across libraries; conservative proxy for full-chain identity",
        "retention_usage": "fixed retrospective test only; never cleaning, fitting, or selection",
        "pr_metric": "sklearn average_precision_score (AP)",
    }
    manifest = {
        "configuration": configuration,
        "sources": {
            "weak_labels.csv": sha256_file(weak_paths["weak_labels"]),
            "weak_manifest.json": sha256_file(weak_paths["manifest"]),
            "retention_csv": sha256_file(args.retention_csv),
        },
        "code": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str((REPO_ROOT / "downstream/AffibodyMHC/evaluate_selection_weak_baseline.py").resolve()): sha256_file(REPO_ROOT / "downstream/AffibodyMHC/evaluate_selection_weak_baseline.py"),
        },
        "base_weak_rows": int(len(base)),
        "outputs": {name: sha256_file(output_dir / name) for name in outputs},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json(manifest, output_dir / "manifest.json")
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
