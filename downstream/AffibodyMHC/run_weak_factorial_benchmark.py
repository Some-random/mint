#!/usr/bin/env python
"""Run a leakage-audited CPU factorial benchmark for Affibody weak labels.

This script answers four deliberately separate questions:

* do simple train-only cleaning rules change transfer to the retention panel;
* does retaining the natural class ratio, weighting the loss, or 1:1
  downsampling change the result;
* do LibA/LibB benefit from separate fits or a shared physically aligned fit;
* how does performance change when neither, one, or both test partners have
  full-sequence identities represented in the weak-label training data.

Only R001/R009/R010-derived weak labels are used for fitting.  The retention
values and binder labels are read after fitting and are used only to compute
the final metrics.  Model regularization is fixed by command line rather than
selected against the retention panel.

The output is private by construction: it must be placed under the repository
``private_data`` directory.  Row-level predictions contain opaque hashes, not
short mutation codes or full sequences.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    sha256_file,
    validate_private_output_path,
)


LIBRARIES = ("LibA", "LibB")
MODEL_SCOPES = ("LibA", "LibB", "Pooled")
REGIMES = ("pair_seen", "peptide_cold", "affibody_cold", "double_cold")
CLEANINGS = ("C0", "C1", "C3", "C4")
BALANCES = ("natural_unweighted", "natural_class_weight", "downsample_1to1")
DOWNSAMPLE_SEEDS = (0, 1, 2)
AFFIBODY_POSITIONS = (6, 10, 13, 14, 17, 27, 31)
SITE_COLUMNS = ("pep_p4", "pep_p5") + tuple(
    "aff_p{}".format(position) for position in AFFIBODY_POSITIONS
)
FEATURE_COLUMNS = SITE_COLUMNS + ("library",)
DEFAULT_C = 1.0
DEFAULT_NEGATIVE_COUNT = 3
DEFAULT_PROMISCUITY_BREADTH = 10
FAILURE_COLUMNS = (
    "configuration_id",
    "model_scope",
    "regime",
    "cleaning",
    "balance",
    "seed",
    "n_scope_weak",
    "n_after_transfer_filter",
    "n_after_cleaning",
    "n_promiscuous_affibodies_flagged",
    "n_fit",
    "n_fit_positive",
    "n_fit_negative",
    "n_fit_unique_peptide",
    "n_fit_unique_affibody",
    "failure",
)

WEAK_COLUMNS = (
    "library",
    "pep",
    "aff",
    "r001_count",
    "r009_count",
    "r010_count",
    "weak_label",
    "within_declared_library_alphabet",
)
RETENTION_COLUMNS = (
    "library",
    "target_retention",
    "target_binder",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _write_csv(frame, path):
    frame.to_csv(path, index=False)
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def _masked_template(sequence, positions_zero_based):
    output = list(str(sequence))
    for position in positions_zero_based:
        output[int(position)] = "X"
    return "".join(output)


def _fill_at_positions(template, positions_zero_based, code):
    _require(len(positions_zero_based) == len(code), "code/template position mismatch")
    output = list(str(template))
    for position, residue in zip(positions_zero_based, str(code)):
        _require(output[int(position)] in ("X", residue), "template has an unexpected designed residue")
        output[int(position)] = residue
    result = "".join(output)
    _require("X" not in result, "unfilled sequence placeholder")
    _require(set(result).issubset(set(AA_ALPHABET)), "noncanonical reconstructed sequence")
    return result


def load_retention_sequences(path):
    """Load the 228-row sequence table while preserving biological code ``NA``."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    missing = set(RETENTION_COLUMNS).difference(frame.columns)
    _require(not missing, "retention sequence table missing {}".format(sorted(missing)))
    _require(len(frame) == 228, "retention sequence table must have 228 rows")
    _require(set(frame["library"]) == set(LIBRARIES), "unexpected retention libraries")
    frame = frame.copy()
    frame["target_retention"] = pd.to_numeric(frame["target_retention"], errors="coerce")
    frame["target_binder"] = pd.to_numeric(frame["target_binder"], errors="coerce")
    _require(int(frame["target_retention"].notna().sum()) == 227, "expected 227 measured retention rows")
    measured = frame["target_retention"].notna()
    _require(bool(frame.loc[measured, "target_binder"].isin([0, 1]).all()), "invalid binder labels")
    _require(bool(frame.loc[~measured, "target_binder"].isna().all()), "unmeasured row has a binder label")
    for row in frame.itertuples(index=False):
        chain1 = str(row.chain1_smart_hla_linker_peptide_sequence)
        chain2 = str(row.chain2_affibody_sequence)
        _require(_sha256_text(chain1) == row.chain1_sha256, "chain-1 hash mismatch")
        _require(_sha256_text(chain2) == row.chain2_sha256, "chain-2 hash mismatch")
        _require(_sha256_text(chain1 + "|" + chain2) == row.sequence_pair_sha256, "pair hash mismatch")
    return frame


def infer_sequence_templates(retention):
    """Infer immutable chain templates from the already-audited sequence table."""
    chain1_templates = set()
    chain2_templates = {}
    for library in LIBRARIES:
        subset = retention.loc[retention["library"].eq(library)]
        local_chain1 = set()
        local_chain2 = set()
        aff_positions = tuple(position - 1 for position in LIBRARY_SPECS[library]["aff_positions"])
        for row in subset.itertuples(index=False):
            chain1 = str(row.chain1_smart_hla_linker_peptide_sequence)
            chain2 = str(row.chain2_affibody_sequence)
            peptide_code = str(row.peptide_design_code)
            affibody_code = str(row.affibody_design_code)
            _require(chain1[264:266] == peptide_code, "retention peptide/code mismatch")
            observed_affibody = "".join(chain2[position] for position in aff_positions)
            _require(observed_affibody == affibody_code, "retention Affibody/code mismatch")
            local_chain1.add(_masked_template(chain1, (264, 265)))
            local_chain2.add(_masked_template(chain2, aff_positions))
        _require(len(local_chain1) == 1, "inconsistent chain-1 template in {}".format(library))
        _require(len(local_chain2) == 1, "inconsistent chain-2 template in {}".format(library))
        chain1_templates.update(local_chain1)
        chain2_templates[library] = next(iter(local_chain2))
    _require(len(chain1_templates) == 1, "LibA and LibB chain-1 templates differ")
    return {
        "chain1": next(iter(chain1_templates)),
        "chain2": chain2_templates,
    }


def attach_sequence_identities_and_sites(frame, templates, peptide_column, affibody_column):
    """Reconstruct identities and aligned mutable-site residues without retaining sequences."""
    output = frame.copy()
    _require(bool(output["library"].isin(LIBRARIES).all()), "unknown library")

    peptide_codes = sorted(set(output[peptide_column].astype(str)))
    peptide_sequence = {
        code: _fill_at_positions(templates["chain1"], (264, 265), code)
        for code in peptide_codes
    }
    peptide_hash = {code: _sha256_text(sequence) for code, sequence in peptide_sequence.items()}

    affibody_sequence = {}
    affibody_hash = {}
    affibody_sites = {}
    for library in LIBRARIES:
        codes = sorted(set(output.loc[output["library"].eq(library), affibody_column].astype(str)))
        positions = tuple(position - 1 for position in LIBRARY_SPECS[library]["aff_positions"])
        for code in codes:
            sequence = _fill_at_positions(templates["chain2"][library], positions, code)
            key = (library, code)
            affibody_sequence[key] = sequence
            affibody_hash[key] = _sha256_text(sequence)
            affibody_sites[key] = tuple(sequence[position - 1] for position in AFFIBODY_POSITIONS)

    peptide_values = output[peptide_column].astype(str).tolist()
    affibody_values = output[affibody_column].astype(str).tolist()
    library_values = output["library"].astype(str).tolist()
    output["global_peptide_sha256"] = [peptide_hash[code] for code in peptide_values]
    output["global_affibody_sha256"] = [
        affibody_hash[(library, code)] for library, code in zip(library_values, affibody_values)
    ]
    output["global_pair_sha256"] = [
        _sha256_text(peptide_sequence[peptide] + "|" + affibody_sequence[(library, affibody)])
        for library, peptide, affibody in zip(library_values, peptide_values, affibody_values)
    ]
    output["pep_p4"] = [code[0] for code in peptide_values]
    output["pep_p5"] = [code[1] for code in peptide_values]
    site_rows = [
        affibody_sites[(library, code)] for library, code in zip(library_values, affibody_values)
    ]
    for index, position in enumerate(AFFIBODY_POSITIONS):
        output["aff_p{}".format(position)] = [row[index] for row in site_rows]
    return output


def load_weak_labels(path, min_negative_count):
    frame = pd.read_csv(
        path,
        usecols=list(WEAK_COLUMNS),
        dtype={"library": str, "pep": str, "aff": str},
        keep_default_na=False,
        na_filter=False,
    )
    missing = set(WEAK_COLUMNS).difference(frame.columns)
    _require(not missing, "weak labels missing {}".format(sorted(missing)))
    for column in (
        "r001_count",
        "r009_count",
        "r010_count",
        "weak_label",
        "within_declared_library_alphabet",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    _require(bool(frame["weak_label"].isin([0, 1]).all()), "invalid weak label")
    label_ok = frame["weak_label"].eq(1) | (
        frame["weak_label"].eq(0) & frame["r001_count"].ge(int(min_negative_count))
    )
    canonical = frame["within_declared_library_alphabet"].eq(1) & label_ok
    output = frame.loc[canonical].copy().reset_index(drop=True)
    _require(set(output["weak_label"].astype(int)) == {0, 1}, "canonical weak pool lacks a class")
    return output


def apply_transfer_regime(frame, retention, regime):
    """Remove retention-panel identities from the weak pool without reading labels."""
    _require(regime in REGIMES, "unknown regime {}".format(regime))
    held_pairs = set(retention["global_pair_sha256"])
    held_peptides = set(retention["global_peptide_sha256"])
    held_affibodies = set(retention["global_affibody_sha256"])
    mask = ~frame["global_pair_sha256"].isin(held_pairs)
    if regime in ("peptide_cold", "double_cold"):
        mask &= ~frame["global_peptide_sha256"].isin(held_peptides)
    if regime in ("affibody_cold", "double_cold"):
        mask &= ~frame["global_affibody_sha256"].isin(held_affibodies)
    output = frame.loc[mask].copy().reset_index(drop=True)
    _require(not bool(output["global_pair_sha256"].isin(held_pairs).any()), "exact pair leakage")
    if regime in ("peptide_cold", "double_cold"):
        _require(not bool(output["global_peptide_sha256"].isin(held_peptides).any()), "peptide leakage")
    if regime in ("affibody_cold", "double_cold"):
        _require(not bool(output["global_affibody_sha256"].isin(held_affibodies).any()), "Affibody leakage")
    return output


def apply_cleaning(frame, cleaning, promiscuity_breadth=10):
    """Apply a cleaning rule using only the current candidate training frame."""
    _require(cleaning in CLEANINGS, "unknown cleaning {}".format(cleaning))
    output = frame.copy()
    flagged_affibodies = set()
    if cleaning == "C1":
        positive = output["weak_label"].eq(1)
        confident = output["r009_count"].ge(3) & output["r010_count"].ge(3)
        output = output.loc[~positive | confident].copy()
    elif cleaning == "C3":
        positive = output.loc[output["weak_label"].eq(1)]
        breadth = positive.groupby("global_affibody_sha256")["global_peptide_sha256"].nunique()
        flagged_affibodies = set(breadth.loc[breadth.ge(int(promiscuity_breadth))].index)
        output = output.loc[~output["global_affibody_sha256"].isin(flagged_affibodies)].copy()
    elif cleaning == "C4":
        positive = output.loc[output["weak_label"].eq(1)]
        positive_peptides = set(positive["global_peptide_sha256"])
        positive_affibodies = set(positive["global_affibody_sha256"])
        hard_negative = (
            output["global_peptide_sha256"].isin(positive_peptides)
            & output["global_affibody_sha256"].isin(positive_affibodies)
        )
        output = output.loc[output["weak_label"].eq(1) | hard_negative].copy()
    output = output.reset_index(drop=True)
    return output, flagged_affibodies


def apply_balance(frame, balance, seed):
    """Return the rows actually passed to the estimator."""
    _require(balance in BALANCES, "unknown balance {}".format(balance))
    if balance != "downsample_1to1":
        return frame.copy().reset_index(drop=True)
    rng = np.random.RandomState(int(seed))
    selected = []
    # Balance each library separately.  In a pooled fit, global downsampling
    # would otherwise let LibB's prevalence determine which LibA rows survive.
    library_groups = (
        frame.groupby("library", sort=True)
        if "library" in frame.columns
        else (("single", frame),)
    )
    for _, library_frame in library_groups:
        groups = {
            label: library_frame.index[
                library_frame["weak_label"].eq(label)
            ].to_numpy(dtype=int)
            for label in (0, 1)
        }
        _require(groups[0].size and groups[1].size, "downsampling requires both classes")
        target = min(groups[0].size, groups[1].size)
        for label in (0, 1):
            if groups[label].size == target:
                values = groups[label]
            else:
                values = rng.choice(groups[label], size=target, replace=False)
            selected.extend(values.tolist())
    return frame.loc[sorted(selected)].copy().reset_index(drop=True)


def make_encoder():
    return OneHotEncoder(
        categories=[list(AA_ALPHABET) for _ in SITE_COLUMNS] + [list(LIBRARIES)],
        handle_unknown="error",
        sparse=True,
        dtype=np.float64,
    )


def fit_site_model(train, evaluation, c_value, balanced_loss):
    _require(set(train["weak_label"].astype(int)) == {0, 1}, "fit frame lacks a class")
    encoder = make_encoder()
    train_x = encoder.fit_transform(train[list(FEATURE_COLUMNS)].astype(str))
    evaluation_x = encoder.transform(evaluation[list(FEATURE_COLUMNS)].astype(str))
    model = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight=None,
        random_state=0,
        max_iter=2000,
        tol=1e-8,
    )
    y = train["weak_label"].astype(int).to_numpy()
    sample_weight = None
    if balanced_loss:
        sample_weight = np.zeros(len(train), dtype=float)
        libraries = sorted(train["library"].unique())
        for library in libraries:
            for label in (0, 1):
                mask = train["library"].eq(library) & train["weak_label"].eq(label)
                count = int(mask.sum())
                _require(count > 0, "library/class cell empty for balanced loss")
                sample_weight[mask.to_numpy()] = float(len(train)) / (
                    len(libraries) * 2.0 * count
                )
    model.fit(train_x, y, sample_weight=sample_weight)
    _require(bool(model.n_iter_[0] < model.max_iter), "logistic model did not converge")
    return model.predict_proba(evaluation_x)[:, 1], int(model.n_iter_[0])


def _safe_spearman(observed, score):
    observed = np.asarray(observed, dtype=float)
    score = np.asarray(score, dtype=float)
    if len(observed) < 2 or np.unique(observed).size < 2 or np.unique(score).size < 2:
        return np.nan
    return float(spearmanr(observed, score)[0])


def _macro_group_metrics(frame, group_column):
    ap_lifts = []
    rank_correlations = []
    for _, group in frame.groupby(group_column):
        y = group["target_binder"].astype(int).to_numpy()
        score = group["score"].to_numpy(dtype=float)
        retention = group["target_retention"].to_numpy(dtype=float)
        if len(np.unique(y)) == 2:
            ap_lifts.append(float(average_precision_score(y, score) - np.mean(y)))
        rho = _safe_spearman(retention, score)
        if np.isfinite(rho):
            rank_correlations.append(rho)
    return {
        "macro_ap_lift": float(np.mean(ap_lifts)) if ap_lifts else np.nan,
        "ap_valid_groups": int(len(ap_lifts)),
        "macro_spearman": float(np.mean(rank_correlations)) if rank_correlations else np.nan,
        "spearman_valid_groups": int(len(rank_correlations)),
    }


def _macro_precision_at_k(frame, k):
    values = []
    for _, group in frame.groupby("global_peptide_sha256"):
        chosen = group.sort_values(["score", "global_affibody_sha256"], ascending=[False, True]).head(int(k))
        if len(chosen):
            values.append(float(chosen["target_binder"].astype(int).mean()))
    return float(np.mean(values)) if values else np.nan, int(len(values))


def _two_way_center(values, peptide_ids, affibody_ids):
    """Remove peptide and Affibody fixed effects by least squares.

    Simple row-mean/column-mean subtraction is exact only on a complete,
    balanced rectangle.  Several compatible LibB panels are irregular, so use
    the projection residual from an intercept plus both categorical effects.
    The intentionally rank-deficient full indicator matrix is safe with the
    Moore-Penrose least-squares solution: fitted values and residuals are
    unique even though individual coefficients are not.
    """
    work = pd.DataFrame(
        {
            "value": np.asarray(values, dtype=float),
            "peptide": np.asarray(peptide_ids, dtype=str),
            "affibody": np.asarray(affibody_ids, dtype=str),
        }
    )
    peptide_levels = sorted(set(work["peptide"]))
    affibody_levels = sorted(set(work["affibody"]))
    columns = [np.ones(len(work), dtype=float)]
    columns.extend(work["peptide"].eq(level).to_numpy(dtype=float) for level in peptide_levels)
    columns.extend(work["affibody"].eq(level).to_numpy(dtype=float) for level in affibody_levels)
    design = np.column_stack(columns)
    observed = work["value"].to_numpy(dtype=float)
    coefficient = np.linalg.lstsq(design, observed, rcond=None)[0]
    return observed - design.dot(coefficient)


def evaluation_metrics(frame):
    """Compute global and nuisance-resistant summaries on one evaluation scope."""
    _require(len(frame) > 0, "empty evaluation frame")
    y = frame["target_binder"].astype(int).to_numpy()
    score = frame["score"].to_numpy(dtype=float)
    retention = frame["target_retention"].to_numpy(dtype=float)
    prevalence = float(np.mean(y))
    if len(np.unique(y)) == 2:
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))
    else:
        auroc = np.nan
        auprc = np.nan
    peptide_metrics = _macro_group_metrics(frame, "global_peptide_sha256")
    affibody_metrics = _macro_group_metrics(frame, "global_affibody_sha256")
    p1, p1_groups = _macro_precision_at_k(frame, 1)
    p3, p3_groups = _macro_precision_at_k(frame, 3)
    centered_retention = _two_way_center(
        retention, frame["global_peptide_sha256"], frame["global_affibody_sha256"]
    )
    centered_score = _two_way_center(
        score, frame["global_peptide_sha256"], frame["global_affibody_sha256"]
    )
    return {
        "n_test": int(len(frame)),
        "n_binder": int(y.sum()),
        "prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "global_ap_lift": float(auprc - prevalence) if np.isfinite(auprc) else np.nan,
        "spearman": _safe_spearman(retention, score),
        "peptide_macro_ap_lift": peptide_metrics["macro_ap_lift"],
        "peptide_ap_valid_groups": peptide_metrics["ap_valid_groups"],
        "peptide_macro_spearman": peptide_metrics["macro_spearman"],
        "peptide_spearman_valid_groups": peptide_metrics["spearman_valid_groups"],
        "affibody_macro_ap_lift": affibody_metrics["macro_ap_lift"],
        "affibody_ap_valid_groups": affibody_metrics["ap_valid_groups"],
        "affibody_macro_spearman": affibody_metrics["macro_spearman"],
        "affibody_spearman_valid_groups": affibody_metrics["spearman_valid_groups"],
        "peptide_macro_precision_at_1": p1,
        "peptide_precision_at_1_groups": p1_groups,
        "peptide_macro_precision_at_3": p3,
        "peptide_precision_at_3_groups": p3_groups,
        "two_way_centered_spearman": _safe_spearman(centered_retention, centered_score),
    }


def add_exposure_flags(evaluation, fitted, regime):
    output = evaluation.copy()
    seen_peptides = set(fitted["global_peptide_sha256"])
    seen_affibodies = set(fitted["global_affibody_sha256"])
    output["peptide_seen_in_fit"] = output["global_peptide_sha256"].isin(seen_peptides).astype(int)
    output["affibody_seen_in_fit"] = output["global_affibody_sha256"].isin(seen_affibodies).astype(int)
    if regime == "pair_seen":
        compatible = output["peptide_seen_in_fit"].eq(1) & output["affibody_seen_in_fit"].eq(1)
    elif regime == "peptide_cold":
        compatible = output["peptide_seen_in_fit"].eq(0) & output["affibody_seen_in_fit"].eq(1)
    elif regime == "affibody_cold":
        compatible = output["peptide_seen_in_fit"].eq(1) & output["affibody_seen_in_fit"].eq(0)
    else:
        compatible = output["peptide_seen_in_fit"].eq(0) & output["affibody_seen_in_fit"].eq(0)
    output["regime_compatible"] = compatible.astype(int)
    return output


def _scope_libraries(model_scope):
    _require(model_scope in MODEL_SCOPES, "unknown model scope {}".format(model_scope))
    return LIBRARIES if model_scope == "Pooled" else (model_scope,)


def _configuration_key(model_scope, regime, cleaning, balance, seed):
    return "{}__{}__{}__{}__s{}".format(model_scope, regime, cleaning, balance, int(seed))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-labels", required=True, type=Path)
    parser.add_argument("--retention-sequences", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--min-negative-count", type=int, default=DEFAULT_NEGATIVE_COUNT)
    parser.add_argument("--promiscuity-breadth", type=int, default=DEFAULT_PROMISCUITY_BREADTH)
    parser.add_argument("--model-scopes", nargs="+", choices=MODEL_SCOPES, default=list(MODEL_SCOPES))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--cleanings", nargs="+", choices=CLEANINGS, default=list(CLEANINGS))
    parser.add_argument("--balances", nargs="+", choices=BALANCES, default=list(BALANCES))
    parser.add_argument("--downsample-seeds", nargs="+", type=int, default=list(DOWNSAMPLE_SEEDS))
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    _require(args.weak_labels.is_file(), "weak-label CSV does not exist")
    _require(args.retention_sequences.is_file(), "retention sequence CSV does not exist")
    _require(math.isfinite(float(args.C)) and float(args.C) > 0, "C must be positive and finite")
    _require(int(args.min_negative_count) >= 1, "minimum negative count must be positive")
    _require(int(args.promiscuity_breadth) >= 2, "promiscuity breadth must be at least two")
    _require(args.downsample_seeds, "at least one downsample seed is required")
    for name, values in (
        ("model scopes", args.model_scopes),
        ("regimes", args.regimes),
        ("cleanings", args.cleanings),
        ("balances", args.balances),
        ("downsample seeds", args.downsample_seeds),
    ):
        _require(len(values) == len(set(values)), "duplicate {} are not allowed".format(name))

    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)

    source_hashes = {
        "weak_labels": sha256_file(args.weak_labels),
        "retention_sequences": sha256_file(args.retention_sequences),
    }
    retention = load_retention_sequences(args.retention_sequences)
    templates = infer_sequence_templates(retention)
    retention = attach_sequence_identities_and_sites(
        retention,
        templates,
        peptide_column="peptide_design_code",
        affibody_column="affibody_design_code",
    )
    _require(bool(retention["global_peptide_sha256"].eq(retention["chain1_sha256"]).all()), "derived retention peptide identity mismatch")
    _require(bool(retention["global_affibody_sha256"].eq(retention["chain2_sha256"]).all()), "derived retention Affibody identity mismatch")
    _require(bool(retention["global_pair_sha256"].eq(retention["sequence_pair_sha256"]).all()), "derived retention pair identity mismatch")

    weak = load_weak_labels(args.weak_labels, args.min_negative_count)
    weak = attach_sequence_identities_and_sites(weak, templates, peptide_column="pep", affibody_column="aff")

    measured_retention = retention.loc[retention["target_retention"].notna()].copy()
    metric_rows = []
    count_rows = []
    prediction_rows = []
    failure_rows = []

    for model_scope in args.model_scopes:
        libraries = _scope_libraries(model_scope)
        scope_weak = weak.loc[weak["library"].isin(libraries)].copy()
        scope_retention_design = retention.loc[retention["library"].isin(libraries)].copy()
        scope_retention = measured_retention.loc[measured_retention["library"].isin(libraries)].copy()
        # Freeze the identity-exposure panels before any cleaning or balancing.
        # This makes sensitivity comparisons use identical test rows.  The
        # four-regime core is the intersection of rows compatible with every
        # prespecified transfer regime and is especially useful for LibB,
        # whose native coverage differs by regime.
        eligible_by_regime = {
            value: apply_transfer_regime(scope_weak, scope_retention_design, value)
            for value in REGIMES
        }
        design_compatible_pairs = {}
        for value, candidate in eligible_by_regime.items():
            exposure = add_exposure_flags(scope_retention, candidate, value)
            design_compatible_pairs[value] = set(
                exposure.loc[exposure["regime_compatible"].eq(1), "global_pair_sha256"]
            )
        common_four_regime_pairs = set.intersection(
            *[design_compatible_pairs[value] for value in REGIMES]
        )
        for regime in args.regimes:
            eligible = eligible_by_regime[regime]
            for cleaning in args.cleanings:
                cleaned, flagged_affibodies = apply_cleaning(
                    eligible,
                    cleaning,
                    promiscuity_breadth=args.promiscuity_breadth,
                )
                for balance in args.balances:
                    seeds = args.downsample_seeds if balance == "downsample_1to1" else (0,)
                    for seed in seeds:
                        key = _configuration_key(model_scope, regime, cleaning, balance, seed)
                        fitted = apply_balance(cleaned, balance, seed)
                        base_count = {
                            "configuration_id": key,
                            "model_scope": model_scope,
                            "regime": regime,
                            "cleaning": cleaning,
                            "balance": balance,
                            "seed": int(seed),
                            "n_scope_weak": int(len(scope_weak)),
                            "n_after_transfer_filter": int(len(eligible)),
                            "n_after_cleaning": int(len(cleaned)),
                            "n_promiscuous_affibodies_flagged": int(len(flagged_affibodies)),
                            "n_fit": int(len(fitted)),
                            "n_fit_positive": int(fitted["weak_label"].sum()),
                            "n_fit_negative": int(fitted["weak_label"].eq(0).sum()),
                            "n_fit_unique_peptide": int(fitted["global_peptide_sha256"].nunique()),
                            "n_fit_unique_affibody": int(fitted["global_affibody_sha256"].nunique()),
                        }
                        if len(fitted) == 0 or fitted["weak_label"].nunique() < 2:
                            failure_rows.append(dict(base_count, failure="fit frame lacks both classes"))
                            count_rows.append(
                                dict(
                                    base_count,
                                    fit_iterations=np.nan,
                                    n_test_native=len(scope_retention),
                                    n_test_design_compatible=len(design_compatible_pairs[regime]),
                                    n_test_common_four_regime_core=len(common_four_regime_pairs),
                                    n_test_compatible=0,
                                )
                            )
                            continue
                        score, iterations = fit_site_model(
                            fitted,
                            scope_retention,
                            c_value=args.C,
                            balanced_loss=balance == "natural_class_weight",
                        )
                        evaluated = scope_retention.copy()
                        evaluated["score"] = score
                        evaluated = add_exposure_flags(evaluated, fitted, regime)
                        compatible = evaluated.loc[evaluated["regime_compatible"].eq(1)].copy()
                        design_compatible = evaluated.loc[
                            evaluated["global_pair_sha256"].isin(design_compatible_pairs[regime])
                        ].copy()
                        common_core = evaluated.loc[
                            evaluated["global_pair_sha256"].isin(common_four_regime_pairs)
                        ].copy()
                        count_rows.append(
                            dict(
                                base_count,
                                fit_iterations=int(iterations),
                                n_test_native=int(len(evaluated)),
                                n_test_design_compatible=int(len(design_compatible)),
                                n_test_common_four_regime_core=int(len(common_core)),
                                n_test_compatible=int(len(compatible)),
                            )
                        )

                        for scope_name, scope_frame in (
                            ("native_panel", evaluated),
                            ("design_compatible", design_compatible),
                            ("common_four_regime_core", common_core),
                            ("actual_fit_compatible", compatible),
                        ):
                            if len(scope_frame) == 0:
                                continue
                            # A pooled estimator is evaluated per library.  The
                            # pooled aggregate is intentionally descriptive
                            # because the assays use different time points.
                            metric_groups = [(library, group) for library, group in scope_frame.groupby("library")]
                            if model_scope == "Pooled":
                                metric_groups.append(("Pooled_descriptive", scope_frame))
                            for evaluation_library, group in metric_groups:
                                metrics = evaluation_metrics(group)
                                metric_rows.append(
                                    dict(
                                        {
                                            "configuration_id": key,
                                            "model_scope": model_scope,
                                            "evaluation_library": evaluation_library,
                                            "regime": regime,
                                            "cleaning": cleaning,
                                            "balance": balance,
                                            "seed": int(seed),
                                            "evaluation_scope": scope_name,
                                            "fixed_C": float(args.C),
                                        },
                                        **metrics
                                    )
                                )

                        prediction_block = evaluated[
                            [
                                "library",
                                "pair_uid",
                                "peptide_uid",
                                "affibody_uid",
                                "global_pair_sha256",
                                "global_peptide_sha256",
                                "global_affibody_sha256",
                                "target_retention",
                                "target_binder",
                                "score",
                                "peptide_seen_in_fit",
                                "affibody_seen_in_fit",
                                "regime_compatible",
                            ]
                        ].copy()
                        prediction_block["design_regime_compatible"] = prediction_block[
                            "global_pair_sha256"
                        ].isin(design_compatible_pairs[regime]).astype(int)
                        prediction_block["common_four_regime_core"] = prediction_block[
                            "global_pair_sha256"
                        ].isin(common_four_regime_pairs).astype(int)
                        prediction_block.insert(0, "configuration_id", key)
                        prediction_rows.append(prediction_block)

    metrics = pd.DataFrame(metric_rows)
    counts = pd.DataFrame(count_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    failures = pd.DataFrame(failure_rows, columns=list(FAILURE_COLUMNS))
    _require(len(metrics) > 0, "benchmark produced no metrics")

    outputs = {
        "metrics.csv": metrics,
        "training_counts.csv": counts,
        "predictions.csv": predictions,
        "failures.csv": failures,
    }
    for filename, frame in outputs.items():
        _write_csv(frame, output_dir / filename)

    configuration = {
        "fixed_C": float(args.C),
        "regularization_selection": "fixed before retention evaluation; no retention-label tuning",
        "min_negative_r001_count": int(args.min_negative_count),
        "promiscuity_positive_peptide_breadth": int(args.promiscuity_breadth),
        "model_scopes": list(args.model_scopes),
        "regimes": list(args.regimes),
        "cleanings": list(args.cleanings),
        "balances": list(args.balances),
        "downsample_seeds": [int(value) for value in args.downsample_seeds],
        "binder_threshold": 75.0,
    }
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "sources": {
            "weak_labels": {"path": str(args.weak_labels.resolve()), "sha256": source_hashes["weak_labels"]},
            "retention_sequences": {"path": str(args.retention_sequences.resolve()), "sha256": source_hashes["retention_sequences"]},
        },
        "code": {
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "code_only_dependency": {
                "path": str((REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py").resolve()),
                "sha256": sha256_file(REPO_ROOT / "downstream/AffibodyMHC/code_only_baseline.py"),
            },
        },
        "configuration": configuration,
        "analysis_status": {
            "status": "exploratory factorial sensitivity analysis",
            "selection_warning": "The same retention panel is evaluated for every configuration. Do not choose a winning cleaning, balance, scope, or regime from these retention metrics; lock choices using weak-label-only validation before any confirmatory retention evaluation.",
            "primary_reporting_scope": "common_four_regime_core for like-for-like regime comparisons; design_compatible for regime-specific coverage",
            "descriptive_only_scope": "native_panel",
            "winner_selected": False,
        },
        "methodology": {
            "fit_target": "provider-derived binary weak label only",
            "retention_use": "final evaluation only; never fitting, balancing, cleaning, or model selection",
            "identity_definition": "SHA-256 of reconstructed complete input chain; global across libraries",
            "pooled_features": list(FEATURE_COLUMNS),
            "pooled_model_scope": "naive shared additive residue effects plus a library indicator; weighted and downsampled conditions balance within library and class; no partially shared library-specific residue deviations",
            "cleaning_C0": "canonical on-design weak labels; positives plus R001>=threshold negatives",
            "cleaning_C1": "C0, retaining positives only when both R009 and R010 counts are >=3",
            "cleaning_C3": "C0, removing all rows for Affibodies with >=threshold distinct positive peptides in the current post-transfer training pool",
            "cleaning_C4": "C0, retaining negatives only when both component identities occur among positives in the current post-transfer training pool",
            "pair_seen": "short identifier for exact-pair-cold/both-partners-seen: exact retention pairs are absent; compatible evaluation additionally requires both partners observed in fitted rows",
            "peptide_cold": "all retention peptide chain identities absent; compatible evaluation requires its Affibody observed",
            "affibody_cold": "all retention Affibody chain identities absent; compatible evaluation requires its peptide observed",
            "double_cold": "both retention partner identities absent",
            "evaluation_scopes": {
                "native_panel": "every measured retention row in the evaluated library or libraries",
                "design_compatible": "identity-exposure compatibility computed before cleaning and balancing",
                "common_four_regime_core": "fixed intersection of design-compatible rows across all four regimes",
                "actual_fit_compatible": "strict identity-exposure compatibility after cleaning and balancing",
            },
            "two_way_centered_metric": "Spearman correlation between least-squares residuals after separately removing peptide and Affibody categorical fixed effects from retention and score",
        },
        "rows": {
            "canonical_weak": int(len(weak)),
            "canonical_weak_positive": int(weak["weak_label"].sum()),
            "canonical_weak_negative": int(weak["weak_label"].eq(0).sum()),
            "retention_designed": int(len(retention)),
            "retention_measured": int(len(measured_retention)),
            "configurations_attempted": int(len(counts)),
            "configurations_failed": int(len(failures)),
            "metric_rows": int(len(metrics)),
            "prediction_rows": int(len(predictions)),
        },
        "outputs": {
            filename: {"path": str((output_dir / filename).resolve()), "sha256": sha256_file(output_dir / filename)}
            for filename in outputs
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "C3 promiscuity is an assay-derived sensitivity filter, not a proven biological truth label.",
            "The retention panel is a designed grid without independent replicate metadata.",
            "Pooled LibA/LibB aggregate metrics are descriptive because retention was measured at different times.",
            "Double-cold here means new identities within these local mutation libraries, not new protein scaffolds.",
        ],
    }
    _write_json(manifest, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "configurations": int(len(counts)),
                "failures": int(len(failures)),
                "metrics": int(len(metrics)),
                "elapsed_seconds": round(time.time() - started, 3),
            },
            sort_keys=True,
        )
    )
    return manifest


if __name__ == "__main__":
    run(parse_args())
