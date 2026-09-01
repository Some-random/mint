#!/usr/bin/env python
"""Aggregate matched head-only and LoRA MINT retention pilot runs."""

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
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path


LIBRARIES = ("LibA", "LibB")
SCHEME_FOLDS = {"random_pair": (0,), "blocked3": (0, 1, 2)}
SCHEME_LABELS = {
    "random_pair": "single random holdout (fold 0)",
    "blocked3": "3 blocked folds",
}
MODEL_ORDER = (
    "mean_prior",
    "site_residue",
    "frozen_mint",
    "head_only",
    "lora_cross",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def load_json(path):
    with open(str(path)) as handle:
        return json.load(handle)


def load_model_predictions(path, model, score_name):
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {
        "library",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "y_true",
        "y_score",
        "model",
    }
    _require(required.issubset(frame.columns), "prediction schema mismatch")
    subset = frame.loc[frame["model"].eq(model)].copy()
    _require(len(subset) > 0, "missing predictions for {}".format(model))
    _require(not bool(subset["pair_uid"].duplicated().any()), "duplicate test prediction")
    return subset[
        ["library", "pair_uid", "peptide_uid", "affibody_uid", "y_true", "y_score"]
    ].rename(columns={"y_score": score_name})


def adapter_b_norm(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    blocks = [
        value.float().reshape(-1)
        for name, value in checkpoint["adapter_state"].items()
        if name.endswith("lora_b")
    ]
    return float(torch.cat(blocks).norm()) if blocks else 0.0


def joined_fold(input_dir, library, scheme, fold):
    prefix = "{}_{}_f{}".format(library.lower(), scheme, fold)
    head_dir = input_dir / (prefix + "_head")
    lora_dir = input_dir / (prefix + "_lora")
    for directory in (head_dir, lora_dir):
        _require(directory.is_dir(), "missing run directory {}".format(directory))
        _require((directory / "manifest.json").is_file(), "run lacks manifest")

    head_manifest = load_json(head_dir / "manifest.json")
    lora_manifest = load_json(lora_dir / "manifest.json")
    for key in ("source", "base_model"):
        _require(head_manifest[key] == lora_manifest[key], "head/LoRA {} mismatch".format(key))
    _require(
        head_manifest["split"]["membership_sha256"]
        == lora_manifest["split"]["membership_sha256"],
        "head/LoRA split mismatch",
    )
    _require(
        head_manifest["code"]["sha256"] == lora_manifest["code"]["sha256"],
        "head/LoRA runner mismatch",
    )

    head = load_model_predictions(
        head_dir / "test_predictions.csv", "head_only", "score_head_only"
    )
    lora = load_model_predictions(
        lora_dir / "test_predictions.csv", "lora_cross", "score_lora_cross"
    )
    frozen = load_model_predictions(
        lora_dir / "test_predictions.csv",
        "frozen_mint_ridge_live",
        "score_frozen_mint",
    )
    site = load_model_predictions(
        lora_dir / "test_predictions.csv", "site_residue_ridge", "score_site_residue"
    )
    prior = load_model_predictions(
        lora_dir / "test_predictions.csv", "mean_prior", "score_mean_prior"
    )
    keys = ["library", "pair_uid", "peptide_uid", "affibody_uid", "y_true"]
    merged = head
    for other in (lora, frozen, site, prior):
        merged = merged.merge(other, on=keys, how="inner", validate="one_to_one")
    _require(len(merged) == len(head), "paired prediction join dropped rows")
    merged["fold"] = int(fold)
    metadata = {
        "head_dir": str(head_dir.resolve()),
        "lora_dir": str(lora_dir.resolve()),
        "head_manifest_sha256": sha256_file(head_dir / "manifest.json"),
        "lora_manifest_sha256": sha256_file(lora_dir / "manifest.json"),
        "runner_sha256": head_manifest["code"]["sha256"],
        "head_best_epoch": int(head_manifest["training"]["best_epoch"]),
        "lora_best_epoch": int(lora_manifest["training"]["best_epoch"]),
        "lora_b_norm": adapter_b_norm(lora_dir / "model_delta.pt"),
    }
    return merged, metadata


def metric_values(target, prediction):
    residual = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(math.sqrt(np.mean(np.square(residual)))),
        "out_of_range": int(np.sum((prediction < 0.0) | (prediction > 100.0))),
    }


def crossed_bootstrap(frame, values, draws, seed):
    peptides = sorted(frame["peptide_uid"].unique())
    affibodies = sorted(frame["affibody_uid"].unique())
    peptide_map = {value: index for index, value in enumerate(peptides)}
    affibody_map = {value: index for index, value in enumerate(affibodies)}
    peptide_index = frame["peptide_uid"].map(peptide_map).to_numpy(dtype=int)
    affibody_index = frame["affibody_uid"].map(affibody_map).to_numpy(dtype=int)
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(int(seed))
    estimates = []
    attempts = 0
    while len(estimates) < int(draws) and attempts < int(draws) * 20:
        attempts += 1
        peptide_counts = np.bincount(
            rng.randint(len(peptides), size=len(peptides)), minlength=len(peptides)
        )
        affibody_counts = np.bincount(
            rng.randint(len(affibodies), size=len(affibodies)), minlength=len(affibodies)
        )
        weights = peptide_counts[peptide_index] * affibody_counts[affibody_index]
        if weights.sum() == 0:
            continue
        estimates.append(float(np.sum(weights * values) / weights.sum()))
    _require(len(estimates) == int(draws), "bootstrap could not produce enough draws")
    return np.percentile(np.asarray(estimates), [2.5, 97.5]).astype(float).tolist()


def summarize(args):
    started = time.time()
    input_dir = args.input_dir.resolve()
    _require(input_dir.is_dir(), "input run directory does not exist")
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)

    metric_rows = []
    comparison_rows = []
    fold_rows = []
    run_metadata = []
    all_runner_hashes = set()
    for library in LIBRARIES:
        for scheme, folds in SCHEME_FOLDS.items():
            blocks = []
            for fold in folds:
                block, metadata = joined_fold(input_dir, library, scheme, fold)
                blocks.append(block)
                all_runner_hashes.add(metadata["runner_sha256"])
                metadata.update({"library": library, "scheme": scheme, "fold": int(fold)})
                run_metadata.append(metadata)
                target = block["y_true"].to_numpy(dtype=float)
                head_error = np.abs(target - block["score_head_only"].to_numpy(dtype=float))
                lora_error = np.abs(target - block["score_lora_cross"].to_numpy(dtype=float))
                fold_rows.append(
                    {
                        "library": library,
                        "scheme": scheme,
                        "fold": int(fold),
                        "n": int(len(block)),
                        "head_only_mae": float(head_error.mean()),
                        "lora_cross_mae": float(lora_error.mean()),
                        "delta_lora_minus_head": float((lora_error - head_error).mean()),
                        "head_best_epoch": metadata["head_best_epoch"],
                        "lora_best_epoch": metadata["lora_best_epoch"],
                        "lora_b_norm": metadata["lora_b_norm"],
                    }
                )
            combined = pd.concat(blocks, ignore_index=True)
            _require(
                not bool(combined["pair_uid"].duplicated().any()),
                "test pairs repeat across {} {} folds".format(library, scheme),
            )
            target = combined["y_true"].to_numpy(dtype=float)
            model_scores = {
                "mean_prior": combined["score_mean_prior"].to_numpy(dtype=float),
                "site_residue": combined["score_site_residue"].to_numpy(dtype=float),
                "frozen_mint": combined["score_frozen_mint"].to_numpy(dtype=float),
                "head_only": combined["score_head_only"].to_numpy(dtype=float),
                "lora_cross": combined["score_lora_cross"].to_numpy(dtype=float),
            }
            absolute_errors = {}
            for model in MODEL_ORDER:
                values = metric_values(target, model_scores[model])
                values.update(
                    {
                        "library": library,
                        "scheme": scheme,
                        "n": int(len(combined)),
                        "model": model,
                    }
                )
                metric_rows.append(values)
                absolute_errors[model] = np.abs(target - model_scores[model])

            for offset, comparator in enumerate(("head_only", "frozen_mint", "site_residue")):
                difference = absolute_errors["lora_cross"] - absolute_errors[comparator]
                lower, upper = crossed_bootstrap(
                    combined,
                    difference,
                    draws=args.bootstrap_draws,
                    seed=int(args.seed) + offset,
                )
                comparison_rows.append(
                    {
                        "library": library,
                        "scheme": scheme,
                        "n": int(len(combined)),
                        "comparison": "lora_cross_minus_{}".format(comparator),
                        "delta_mae": float(difference.mean()),
                        "ci95_low": lower,
                        "ci95_high": upper,
                        "interpretation": "negative favors LoRA",
                    }
                )

    _require(len(all_runner_hashes) == 1, "runs use different fine-tuning code hashes")
    metrics = pd.DataFrame(metric_rows).sort_values(["library", "scheme", "model"])
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        ["library", "scheme", "comparison"]
    )
    folds = pd.DataFrame(fold_rows).sort_values(["library", "scheme", "fold"])
    for name, frame in (
        ("aggregate_metrics.csv", metrics),
        ("paired_comparisons.csv", comparisons),
        ("fold_metrics.csv", folds),
    ):
        path = output_dir / name
        frame.to_csv(path, index=False)
        os.chmod(str(path), 0o600)

    lines = [
        "# MINT fine-tuning pilot aggregate",
        "",
        "MAE is in retention percentage points. Delta is LoRA MAE minus comparator",
        "MAE, so a negative delta favors LoRA. Intervals are descriptive two-way",
        "peptide-by-Affibody bootstrap intervals; they are not confirmatory intervals.",
        "",
        "| Library | Split | n | Frozen MINT | Head only | LoRA | Site | LoRA − head [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for library in LIBRARIES:
        for scheme in SCHEME_FOLDS:
            subset = metrics.loc[
                metrics["library"].eq(library) & metrics["scheme"].eq(scheme)
            ].set_index("model")
            comparison = comparisons.loc[
                comparisons["library"].eq(library)
                & comparisons["scheme"].eq(scheme)
                & comparisons["comparison"].eq("lora_cross_minus_head_only")
            ].iloc[0]
            lines.append(
                "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} [{:+.3f}, {:+.3f}] |".format(
                    library,
                    SCHEME_LABELS[scheme],
                    int(comparison["n"]),
                    subset.loc["frozen_mint", "mae"],
                    subset.loc["head_only", "mae"],
                    subset.loc["lora_cross", "mae"],
                    subset.loc["site_residue", "mae"],
                    comparison["delta_mae"],
                    comparison["ci95_low"],
                    comparison["ci95_high"],
                )
            )
    head_comparisons = comparisons.loc[
        comparisons["comparison"].eq("lora_cross_minus_head_only")
    ]
    all_head_intervals_cross_zero = bool(
        (
            head_comparisons["ci95_low"].le(0.0)
            & head_comparisons["ci95_high"].ge(0.0)
        ).all()
    )
    libb_blocked_runs = [
        row
        for row in run_metadata
        if row["library"] == "LibB" and row["scheme"] == "blocked3"
    ]
    libb_blocked_is_noop = bool(libb_blocked_runs) and all(
        row["lora_best_epoch"] == 0 and row["lora_b_norm"] == 0.0
        for row in libb_blocked_runs
    )
    blocked_counts = {
        library: int(
            metrics.loc[
                metrics["library"].eq(library)
                & metrics["scheme"].eq("blocked3"),
                "n",
            ].iloc[0]
        )
        for library in LIBRARIES
    }
    lines.append("")
    if all_head_intervals_cross_zero:
        lines.extend(
            [
                "Conclusion: every descriptive LoRA-versus-head interval includes zero;",
                "this single-seed pilot provides no meaningful evidence that LoRA fine-tuning",
                "improves over the matched frozen-backbone, trainable-head control.",
            ]
        )
    else:
        lines.extend(
            [
                "Conclusion: at least one descriptive LoRA-versus-head interval excludes zero.",
                "Inspect the paired table and prespecified decision rule before interpreting it.",
            ]
        )
    if libb_blocked_is_noop:
        lines.append(
            "All LibB blocked folds selected epoch 0, so their saved adapters are exact no-ops."
        )
    lines.extend(
        [
            "",
            "The blocked result is pair-weighted and covers only three diagonal identity",
            "blocks ({} LibA and {} LibB pairs), uses interpolation validation for early".format(
                blocked_counts["LibA"], blocked_counts["LibB"]
            ),
            "stopping, and is retrospective/exploratory. Its interval excludes model-refit,",
            "seed, partition and omitted off-diagonal-block uncertainty; it is not an",
            "unbiased final estimate.",
            "",
        ]
    )
    summary_path = output_dir / "aggregate_summary.md"
    with open(str(summary_path), "w") as handle:
        handle.write("\n".join(lines))
    os.chmod(str(summary_path), 0o600)

    script_path = Path(__file__).resolve()
    output_paths = [
        output_dir / "aggregate_metrics.csv",
        output_dir / "paired_comparisons.csv",
        output_dir / "fold_metrics.csv",
        summary_path,
    ]
    manifest_path = output_dir / "aggregate_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "input_dir": str(input_dir),
            "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "fine_tune_runner_sha256": list(all_runner_hashes)[0],
            "bootstrap": {
                "draws": int(args.bootstrap_draws),
                "seed": int(args.seed),
                "method": "independent peptide and Affibody identity resampling",
                "interpretation": "descriptive; fitted models are not refit",
            },
            "runs": run_metadata,
            "outputs": {path.name: sha256_file(path) for path in output_paths},
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        },
    )
    print(str(summary_path))
    print("\n".join(lines))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args(argv)


if __name__ == "__main__":
    summarize(parse_args())
