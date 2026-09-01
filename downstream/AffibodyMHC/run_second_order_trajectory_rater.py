#!/usr/bin/env python
"""Exact two-step trajectory DataRater pilot for the Affibody weak labels.

This stage is intentionally small.  A 12-parameter linear rater reads fixed
summaries of the complete R000--R014 trajectory of each weak-label pair. Separate
within-class softmaxes preserve equal positive/negative training mass.  The
rater is optimized by differentiating a clean within-peptide retention-ranking
loss through exactly two weighted updates of a linear head on frozen MINT
features.  MINT itself is never updated.

The default crossed 3x3 partner-block evaluation covers every measured cell
once while keeping both identities of each test cell out of rater training,
weak-head training, and the clean meta-objective.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import run_meta_gradient_diagnostic as diagnostic


SCHEMA_VERSION = "affibody-second-order-trajectory-rater-v1"
COMPACT_TRAJECTORY_NAMES = (
    "r001_log_count",
    "r001_log_frequency",
    "r009_r010_mean_log_count",
    "r009_r010_mean_log_frequency",
    "r009_r010_frequency_disagreement",
    "r009_r010_both_observed",
    "r011_r014_rounds_observed",
    "r011_r014_mean_log_count",
    "r011_r014_mean_log_frequency",
    "total_rounds_observed",
    "r008_to_r009_log_frequency_change",
    "r009_to_r010_log_frequency_change",
)


def load_trajectory(path, manifest_path, cache_frame):
    manifest = diagnostic._read_json(manifest_path)
    diagnostic._require(
        manifest.get("schema_version") == "selection-trajectories-v2",
        "the corrected v2 trajectory artifact is required",
    )
    expected = manifest.get("output", {}).get("npz", {}).get("sha256")
    diagnostic._require(expected == diagnostic.sha256_file(path), "trajectory hash mismatch")
    with np.load(str(path), allow_pickle=False) as archive:
        required = {
            "cache_row_index",
            "pair_uid",
            "library",
            "weak_label",
            "trajectory_features",
            "trajectory_feature_names",
        }
        diagnostic._require(required.issubset(archive.files), "trajectory schema mismatch")
        cache_indices = np.asarray(archive["cache_row_index"], dtype=np.int64)
        pair_uids = np.asarray(archive["pair_uid"]).astype(str)
        libraries = np.asarray(archive["library"]).astype(str)
        labels = np.asarray(archive["weak_label"], dtype=np.int8)
        features = np.asarray(archive["trajectory_features"], dtype=np.float32).copy()
        names = np.asarray(archive["trajectory_feature_names"]).astype(str)
    diagnostic._require(features.shape == (diagnostic.EXPECTED_WEAK_ROWS, 63), "bad trajectory shape")
    diagnostic._require(bool(np.isfinite(features).all()), "non-finite trajectory feature")
    source = cache_frame.iloc[cache_indices]
    diagnostic._require(
        np.array_equal(source["pair_uid"].astype(str).to_numpy(), pair_uids),
        "trajectory/cache pair mismatch",
    )
    diagnostic._require(
        np.array_equal(source["library"].astype(str).to_numpy(), libraries),
        "trajectory/cache library mismatch",
    )
    diagnostic._require(
        np.array_equal(source["weak_label"].astype(str).to_numpy().astype(int), labels.astype(int)),
        "trajectory/cache label mismatch",
    )
    by_cache_index = np.full(len(cache_frame), -1, dtype=np.int64)
    by_cache_index[cache_indices] = np.arange(len(cache_indices), dtype=np.int64)
    diagnostic._require(bool(np.all(by_cache_index[cache_indices] >= 0)), "trajectory index failure")
    return features, names, by_cache_index, manifest


def compact_trajectory_features(features, feature_names):
    """Construct the fixed 12 low-capacity rater inputs."""
    features = np.asarray(features, dtype=np.float32)
    index = {str(name): position for position, name in enumerate(feature_names)}

    def column(name):
        diagnostic._require(name in index, "missing trajectory feature {}".format(name))
        return features[:, index[name]]

    output = np.column_stack(
        [
            column("log1p_count_r001"),
            column("log2_frequency_half_read_r001"),
            0.5 * (column("log1p_count_r009") + column("log1p_count_r010")),
            0.5
            * (
                column("log2_frequency_half_read_r009")
                + column("log2_frequency_half_read_r010")
            ),
            column("absolute_log2_frequency_difference_r009_r010"),
            column("observed_in_both_r009_r010"),
            column("later_rounds_observed_r011_r014"),
            np.mean(
                np.column_stack(
                    [column("log1p_count_r{:03d}".format(round_index)) for round_index in range(11, 15)]
                ),
                axis=1,
            ),
            np.mean(
                np.column_stack(
                    [
                        column("log2_frequency_half_read_r{:03d}".format(round_index))
                        for round_index in range(11, 15)
                    ]
                ),
                axis=1,
            ),
            column("rounds_observed"),
            column("censored_log2_frequency_change_r008_to_r009"),
            column("censored_log2_frequency_change_r009_to_r010"),
        ]
    ).astype(np.float32)
    diagnostic._require(
        output.shape == (len(features), len(COMPACT_TRAJECTORY_NAMES)),
        "compact trajectory shape mismatch",
    )
    diagnostic._require(bool(np.isfinite(output).all()), "non-finite compact trajectory")
    return output


def standardize_trajectory(features, labels):
    """Standardize each fixed rater input separately inside weak label class."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    transformed = np.zeros_like(features, dtype=np.float32)
    means = np.zeros((2, features.shape[1]), dtype=np.float64)
    scales = np.ones((2, features.shape[1]), dtype=np.float64)
    zero_masks = np.zeros((2, features.shape[1]), dtype=bool)
    for label in (0, 1):
        mask = labels == label
        diagnostic._require(bool(mask.any()), "trajectory standardization lacks a class")
        means[label] = features[mask].mean(axis=0, dtype=np.float64)
        scales[label] = features[mask].std(axis=0, dtype=np.float64, ddof=0)
        zero_masks[label] = scales[label] <= 0.0
        scales[label, zero_masks[label]] = 1.0
        transformed[mask] = ((features[mask] - means[label]) / scales[label]).astype(np.float32)
        # A constant within-class feature must become exact zero and therefore
        # cannot create a label-definition or numerical artifact.
        diagnostic._require(
            bool(np.all(transformed[mask][:, zero_masks[label]] == 0.0)),
            "zero-variance trajectory feature did not cancel",
        )
    diagnostic._require(bool(np.isfinite(transformed).all()), "non-finite standardized trajectory")
    return transformed, means, scales, zero_masks


def class_normalized_weights(rater_scores, labels, score_bound=2.0):
    """Return differentiable class-balanced weights and concentration diagnostics."""
    torch = diagnostic._import_torch()
    n = len(labels)
    effective_weights = torch.zeros_like(rater_scores)
    mean_one_weights = torch.zeros_like(rater_scores)
    kls = []
    esses = []
    for label in (0, 1):
        indices = torch.nonzero(labels == float(label), as_tuple=False).flatten()
        diagnostic._require(int(indices.numel()) > 0, "rater input lacks a class")
        local_scores = torch.index_select(rater_scores, 0, indices)
        bounded_scores = float(score_bound) * torch.tanh(local_scores / float(score_bound))
        relative = torch.exp(bounded_scores)
        local_mean_one = relative / torch.mean(relative)
        probability = local_mean_one / float(len(indices))
        # Learned weights have mean one within class. Multiplying by the fixed
        # base class weight assigns total effective mass n/2 to each class.
        local_effective = (float(n) / (2.0 * float(len(indices)))) * local_mean_one
        mean_one_weights = mean_one_weights.index_copy(0, indices, local_mean_one)
        effective_weights = effective_weights.index_copy(0, indices, local_effective)
        kls.append(torch.sum(probability * torch.log(probability * float(len(indices)) + 1e-12)))
        esses.append(1.0 / torch.sum(probability.square()))
    kl_by_class = torch.stack(kls)
    return (
        effective_weights,
        mean_one_weights,
        kl_by_class.mean(),
        torch.stack(esses),
        kl_by_class,
    )


def unrolled_head(x, y, example_weights, inner_steps, inner_lr, head_l2):
    """Differentiate exactly through one or two full-batch head updates."""
    torch = diagnostic._import_torch()
    diagnostic._require(int(inner_steps) in (1, 2), "inner_steps must be one or two")
    weight = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    bias = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    losses = []
    for _ in range(int(inner_steps)):
        logits = torch.mv(x, weight) + bias
        point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )
        inner_loss = torch.mean(point_loss * example_weights) + 0.5 * float(head_l2) * torch.sum(
            weight.square()
        )
        gradient_weight, gradient_bias = torch.autograd.grad(
            inner_loss, (weight, bias), create_graph=True
        )
        weight = weight - float(inner_lr) * gradient_weight
        bias = bias - float(inner_lr) * gradient_bias
        losses.append(inner_loss)
    return weight, bias, losses


def second_order_objective(
    theta,
    x,
    y,
    trajectory,
    clean_differences,
    inner_steps,
    inner_lr,
    head_l2,
    rater_l2,
    entropy_penalty,
):
    """Return the exact meta-objective and transparent intermediate values."""
    torch = diagnostic._import_torch()
    rater_scores = torch.mv(trajectory, theta)
    example_weights, mean_one_weights, kl, ess, kl_by_class = class_normalized_weights(
        rater_scores, y
    )
    head_weight, head_bias, inner_losses = unrolled_head(
        x, y, example_weights, inner_steps, inner_lr, head_l2
    )
    clean_margin = torch.mv(clean_differences, head_weight)
    clean_loss = torch.nn.functional.softplus(-clean_margin).mean()
    objective = (
        clean_loss
        + float(entropy_penalty) * kl
        + 0.5 * float(rater_l2) * torch.mean(theta.square())
    )
    return objective, {
        "rater_scores": rater_scores,
        "mean_one_weights": mean_one_weights,
        "kl": kl,
        "kl_by_class": kl_by_class,
        "ess": ess,
        "head_weight": head_weight,
        "head_bias": head_bias,
        "inner_losses": inner_losses,
        "clean_loss": clean_loss,
    }


def learn_rater(
    x,
    y,
    trajectory,
    clean_differences,
    meta_steps,
    meta_lr,
    inner_steps,
    inner_lr,
    head_l2,
    rater_l2,
    entropy_penalty,
):
    """Fit the linear trajectory rater using an exact second-order graph."""
    torch = diagnostic._import_torch()
    theta = torch.zeros(
        trajectory.shape[1], dtype=trajectory.dtype, device=trajectory.device, requires_grad=True
    )
    optimizer = torch.optim.Adam([theta], lr=float(meta_lr))
    history = []

    def history_item(meta_step, objective, state, update_gradient_norm):
        return {
            "meta_step": int(meta_step),
            "state_timing": "after_update" if int(meta_step) else "initial",
            "objective": float(objective.detach().cpu().item()),
            "clean_pairwise_loss": float(state["clean_loss"].detach().cpu().item()),
            "entropy_kl": float(state["kl"].detach().cpu().item()),
            "negative_weight_entropy": float(
                math.log(int(torch.sum(y < 0.5).item()))
                - state["kl_by_class"][0].detach().cpu().item()
            ),
            "positive_weight_entropy": float(
                math.log(int(torch.sum(y > 0.5).item()))
                - state["kl_by_class"][1].detach().cpu().item()
            ),
            "negative_effective_sample_size": float(state["ess"][0].detach().cpu().item()),
            "positive_effective_sample_size": float(state["ess"][1].detach().cpu().item()),
            "negative_max_mean_one_weight": float(
                torch.max(state["mean_one_weights"][y < 0.5]).detach().cpu().item()
            ),
            "positive_max_mean_one_weight": float(
                torch.max(state["mean_one_weights"][y > 0.5]).detach().cpu().item()
            ),
            "first_inner_loss": float(state["inner_losses"][0].detach().cpu().item()),
            "last_inner_loss": float(state["inner_losses"][-1].detach().cpu().item()),
            "theta_norm": float(torch.linalg.vector_norm(theta).detach().cpu().item()),
            "update_gradient_norm": float(update_gradient_norm),
        }

    initial_objective, initial_state = second_order_objective(
        theta,
        x,
        y,
        trajectory,
        clean_differences,
        inner_steps,
        inner_lr,
        head_l2,
        rater_l2,
        entropy_penalty,
    )
    history.append(history_item(0, initial_objective, initial_state, np.nan))
    for meta_step in range(int(meta_steps)):
        objective, state = second_order_objective(
            theta,
            x,
            y,
            trajectory,
            clean_differences,
            inner_steps,
            inner_lr,
            head_l2,
            rater_l2,
            entropy_penalty,
        )
        diagnostic._require(bool(torch.isfinite(objective).item()), "non-finite meta-objective")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = float(torch.linalg.vector_norm(theta.grad).detach().cpu().item())
        torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
        optimizer.step()
        if (meta_step + 1) % 10 == 0 or meta_step + 1 == int(meta_steps):
            evaluated_objective, evaluated_state = second_order_objective(
                theta,
                x,
                y,
                trajectory,
                clean_differences,
                inner_steps,
                inner_lr,
                head_l2,
                rater_l2,
                entropy_penalty,
            )
            history.append(
                history_item(
                    meta_step + 1,
                    evaluated_objective,
                    evaluated_state,
                    gradient_norm,
                )
            )
    with torch.no_grad():
        scores = torch.mv(trajectory, theta)
        _, weights, kl, ess, kl_by_class = class_normalized_weights(scores, y)
    return {
        "theta": theta.detach().cpu().numpy().astype(np.float64),
        "scores": scores.detach().cpu().numpy().astype(np.float64),
        "weights": weights.detach().cpu().numpy().astype(np.float64),
        "entropy_kl": float(kl.detach().cpu().item()),
        "negative_weight_entropy": float(
            math.log(int(torch.sum(y < 0.5).item())) - kl_by_class[0].detach().cpu().item()
        ),
        "positive_weight_entropy": float(
            math.log(int(torch.sum(y > 0.5).item())) - kl_by_class[1].detach().cpu().item()
        ),
        "negative_effective_sample_size": float(ess[0].detach().cpu().item()),
        "positive_effective_sample_size": float(ess[1].detach().cpu().item()),
        "negative_max_mean_one_weight": float(torch.max(weights[y < 0.5]).detach().cpu().item()),
        "positive_max_mean_one_weight": float(torch.max(weights[y > 0.5]).detach().cpu().item()),
        "history": history,
    }


def load_gradient_reference(run_dir, library):
    """Load the immutable crossed9 alignment/full/random control run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = diagnostic._read_json(manifest_path)
    diagnostic._require(
        manifest.get("schema_version") == diagnostic.SCHEMA_VERSION,
        "gradient reference schema mismatch",
    )
    protocol = manifest.get("protocol", {})
    diagnostic._require(protocol.get("fold_mode") == "crossed9", "gradient reference is not crossed9")
    diagnostic._require(protocol.get("libraries") == [library], "gradient reference library mismatch")
    diagnostic._require(int(protocol.get("outer_folds", -1)) == 9, "gradient reference fold mismatch")
    diagnostic._require(int(protocol.get("random_controls", -1)) >= 5, "too few reference controls")
    diagnostic._require(
        manifest.get("source", {}).get("sha256") == diagnostic.sha256_file(diagnostic.__file__),
        "gradient reference was produced by a different dependency source",
    )
    frames = {}
    for name in (
        "metrics.csv",
        "conditions.csv",
        "predictions.csv",
        "retention_split_membership.csv",
    ):
        path = run_dir / name
        expected = manifest.get("outputs", {}).get(name, {}).get("sha256")
        diagnostic._require(expected == diagnostic.sha256_file(path), "reference hash mismatch: {}".format(name))
        frames[name] = pd.read_csv(path)
        frames[name]["source"] = "crossed_gradient_reference"
        if "method" in frames[name]:
            frames[name]["method"] = frames[name]["method"].replace(
                {"alignment": "gradient_alignment", "full": "full_class_balanced"}
            )
        frames[name]["peptide_block"] = frames[name]["fold"].astype(int) // 3
        frames[name]["affibody_block"] = frames[name]["fold"].astype(int) % 3
    metrics = frames["metrics.csv"]
    diagnostic._require(
        len(metrics.loc[metrics["method"].eq("gradient_alignment")]) == 27,
        "gradient reference lacks alignment conditions",
    )
    diagnostic._require(
        len(metrics.loc[metrics["method"].eq("random")])
        == 9 * 3 * int(protocol["random_controls"]),
        "gradient reference lacks random controls",
    )
    return frames, manifest, manifest_path


def build_summary(metrics, gate, gate_result, histories, conditions, coefficient_stability):
    lines = [
        "# Exact second-order trajectory-rater pilot",
        "",
        "A 12-parameter linear rater used fixed summaries of each pair's R000--R014 sequencing trajectory. "
        "Its parameters were updated through two differentiable weak-head training steps. "
        "MINT remained frozen.",
        "",
        "The crossed 3x3 test covers every measured {} cell once. In each block, both held "
        "partner identities were excluded from the weak pool and the retention measurements "
        "used to train the rater.".format(metrics["library"].iloc[0]),
        "",
        "| Weak rows kept | Trajectory rater | Gradient alignment | Random mean | Rater delta | Positive blocks | Gate |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in gate.sort_values("fraction").iterrows():
        fraction = float(row["fraction"])
        rater = metrics.loc[
            metrics["method"].eq("trajectory_rater") & metrics["fraction"].eq(fraction),
            "within_peptide_pairwise_accuracy_micro",
        ]
        random = metrics.loc[
            metrics["method"].eq("random") & metrics["fraction"].eq(fraction),
            "within_peptide_pairwise_accuracy_micro",
        ]
        gradient = metrics.loc[
            metrics["method"].eq("gradient_alignment") & metrics["fraction"].eq(fraction),
            "within_peptide_pairwise_accuracy_micro",
        ]
        lines.append(
            "| {:.0f}% | {:.3f} | {:.3f} | {:.3f} | {:+.3f} | {}/{} | {} |".format(
                100.0 * fraction,
                float(rater.mean()),
                float(gradient.mean()),
                float(random.mean()),
                float(row["mean_alignment_minus_random"]),
                int(row["positive_folds"]),
                int(row["total_folds"]),
                "pass" if int(row["fraction_pass"]) else "fail",
            )
        )
    full_balanced = metrics.loc[metrics["method"].eq("full_class_balanced")]
    full_unweighted = metrics.loc[metrics["method"].eq("full_unweighted")]
    first = histories.sort_values(["fold", "meta_step"]).groupby("fold").first()
    last = histories.sort_values(["fold", "meta_step"]).groupby("fold").last()
    rater_state = conditions.loc[
        conditions["source"].eq("current_second_order_run")
        & conditions["method"].eq("full_unweighted")
    ].copy()
    negative_ess_fraction = (
        rater_state["rater_negative_effective_sample_size"]
        / rater_state["weak_selected_negative"]
    )
    positive_ess_fraction = (
        rater_state["rater_positive_effective_sample_size"]
        / rater_state["weak_selected_positive"]
    )
    lines.extend(
        [
            "",
            "Overall fixed pilot gate: **{}**. The gate requires at least two retained fractions to have "
            "a positive mean delta and a positive delta in at least six of nine crossed blocks.".format(
                "pass" if gate_result else "fail"
            ),
            "",
            "The full weak pool scored {:.3f} with class-balanced loss and {:.3f} without "
            "class weighting. The rater's clean "
            "development loss changed from {:.3f} to {:.3f} on average across blocks.".format(
                float(full_balanced["within_peptide_pairwise_accuracy_micro"].mean()),
                float(full_unweighted["within_peptide_pairwise_accuracy_micro"].mean()),
                float(first["clean_pairwise_loss"].mean()),
                float(last["clean_pairwise_loss"].mean()),
            ),
            "",
            "The median effective sample-size fraction was {:.3f} for negatives and {:.3f} "
            "for positives; the largest mean-one learned weight was {:.3f}. Coefficient signs "
            "are recorded across all nine blocks; the median majority-sign agreement was {:.0f}/9.".format(
                float(negative_ess_fraction.median()),
                float(positive_ess_fraction.median()),
                float(
                    max(
                        rater_state["rater_negative_max_mean_one_weight"].max(),
                        rater_state["rater_positive_max_mean_one_weight"].max(),
                    )
                ),
                float(coefficient_stability["majority_sign_folds"].median()),
            ),
            "",
            "This remains exploratory: the nine crossed blocks reuse overlapping development "
            "and weak-label pools, so they are correlated evaluations, not nine independent "
            "replicates. The experiment is not a substitute for a new measured panel.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz",
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/mint_weak_cache_v1/merged/manifest.json",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/selection_trajectories_v2/selection_trajectories.npz",
    )
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/selection_trajectories_v2/manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gradient-diagnostic-dir",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/meta_gradient_crossed9_liba_canonical_v1",
    )
    parser.add_argument("--library", choices=diagnostic.LIBRARIES, default="LibA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--meta-steps", type=int, default=100)
    parser.add_argument("--meta-lr", type=float, default=0.03)
    parser.add_argument("--inner-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--head-l2", type=float, default=0.005)
    parser.add_argument("--rater-l2", type=float, default=0.001)
    parser.add_argument("--entropy-penalty", type=float, default=0.01)
    parser.add_argument("--head-max-iter", type=int, default=300)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    diagnostic._require(int(args.random_controls) >= 5, "at least five random controls required")
    diagnostic._require(int(args.meta_steps) >= 1, "meta steps must be positive")
    output_dir = diagnostic._validate_private_output(args.output_dir)
    diagnostic._require(not output_dir.exists(), "output directory exists; refusing overwrite")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = diagnostic._import_torch()
    diagnostic._require(torch.cuda.is_available(), "this pilot requires CUDA")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    cache_frame, cache_features, cache_manifest = diagnostic.load_cache(
        args.cache, args.cache_manifest
    )
    trajectory_features, trajectory_names, trajectory_index, trajectory_manifest = load_trajectory(
        args.trajectory, args.trajectory_manifest, cache_frame
    )
    reference_frames, reference_manifest, reference_manifest_path = load_gradient_reference(
        args.gradient_diagnostic_dir, args.library
    )
    reference_protocol = reference_manifest["protocol"]
    diagnostic._require(
        int(reference_protocol["random_controls"]) == int(args.random_controls),
        "random-control count differs from canonical reference",
    )
    diagnostic._require(
        int(reference_protocol["lbfgs_max_iter"]) == int(args.head_max_iter),
        "head iteration cap differs from canonical reference",
    )
    diagnostic._require(
        float(reference_protocol["l2"]) == float(args.head_l2),
        "head L2 differs from canonical reference",
    )
    diagnostic._require(
        int(reference_protocol["seed"]) == int(args.seed),
        "split seed differs from canonical reference",
    )
    diagnostic._require(
        reference_manifest["input"]["cache"]["sha256"] == diagnostic.sha256_file(args.cache),
        "canonical reference used a different MINT cache",
    )
    weak_all = cache_frame.loc[
        cache_frame["source_kind"].eq("weak") & cache_frame["library"].eq(args.library)
    ].copy()
    weak_all["weak_label"] = pd.to_numeric(weak_all["weak_label"], errors="raise").astype(int)
    retention = cache_frame.loc[
        cache_frame["source_kind"].eq("retention")
        & cache_frame["library"].eq(args.library)
        & cache_frame["measurement_missing"].eq("0")
    ].copy()
    retention["target_retention"] = pd.to_numeric(retention["target_retention"], errors="raise")
    retention["target_binder"] = pd.to_numeric(retention["target_binder"], errors="raise").astype(int)
    diagnostic._require(
        len(retention) == diagnostic.EXPECTED_MEASURED[args.library], "retention count mismatch"
    )

    metric_rows = []
    condition_rows = []
    prediction_rows = []
    rater_rows = []
    history_rows = []
    coefficient_rows = []
    split_rows = []
    for fold in range(9):
        print("{} crossed fold {}: preparing".format(args.library, fold), flush=True)
        roles, held_peptides, held_affibodies, peptide_block, affibody_block = diagnostic.make_outer_roles(
            retention, args.library, fold, args.seed, "crossed9"
        )
        reference_split = reference_frames["retention_split_membership.csv"].loc[
            reference_frames["retention_split_membership.csv"]["fold"].eq(fold)
        ]
        expected_roles = dict(zip(reference_split["pair_uid"].astype(str), reference_split["role"]))
        observed_roles = dict(zip(retention["pair_uid"].astype(str), roles.astype(str)))
        diagnostic._require(observed_roles == expected_roles, "current/reference split mismatch")
        weak_roles = diagnostic.weak_outer_roles(weak_all, held_peptides, held_affibodies)
        weak = weak_all.loc[weak_roles == "eligible"].copy().reset_index(drop=True)
        development = retention.loc[roles == "development"].copy().reset_index(drop=True)
        test = retention.loc[roles == "test"].copy().reset_index(drop=True)
        for position, (_, row) in enumerate(retention.iterrows()):
            split_rows.append(
                {
                    "library": args.library,
                    "fold": fold,
                    "peptide_block": peptide_block,
                    "affibody_block": affibody_block,
                    "pair_uid": row["pair_uid"],
                    "role": roles[position],
                }
            )
        cache_indices = weak["row_index"].to_numpy(dtype=int)
        retention_indices = np.concatenate(
            [development["row_index"].to_numpy(dtype=int), test["row_index"].to_numpy(dtype=int)]
        )
        x_weak, x_retention, _, _, mint_zero = diagnostic.standardize_from_training(
            cache_features[cache_indices], cache_features[retention_indices]
        )
        x_development = x_retention[: len(development)]
        x_test = x_retention[len(development) :]
        clean_difference, clean_pairs = diagnostic.make_pair_differences(
            development, x_development
        )
        local_trajectory_indices = trajectory_index[cache_indices]
        diagnostic._require(bool(np.all(local_trajectory_indices >= 0)), "weak row lacks trajectory")
        labels = weak["weak_label"].to_numpy(dtype=int)
        compact = compact_trajectory_features(
            trajectory_features[local_trajectory_indices], trajectory_names
        )
        trajectory, _, _, trajectory_zero_masks = standardize_trajectory(compact, labels)
        trajectory_zero = int(trajectory_zero_masks.sum())
        pair_uids = weak["pair_uid"].astype(str).to_numpy()
        x_tensor = torch.as_tensor(x_weak, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
        trajectory_tensor = torch.as_tensor(trajectory, dtype=torch.float32, device=device)
        clean_tensor = torch.as_tensor(clean_difference, dtype=torch.float32, device=device)
        test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        print("{} crossed fold {}: fitting exact two-step rater".format(args.library, fold), flush=True)
        rater = learn_rater(
            x_tensor,
            y_tensor,
            trajectory_tensor,
            clean_tensor,
            args.meta_steps,
            args.meta_lr,
            args.inner_steps,
            args.inner_lr,
            args.head_l2,
            args.rater_l2,
            args.entropy_penalty,
        )
        for item in rater["history"]:
            history_rows.append(
                dict(
                    library=args.library,
                    fold=fold,
                    peptide_block=peptide_block,
                    affibody_block=affibody_block,
                    **item
                )
            )
        for name, value in zip(COMPACT_TRAJECTORY_NAMES, rater["theta"]):
            coefficient_rows.append(
                {"library": args.library, "fold": fold, "feature": name, "coefficient": value}
            )
        for index, (_, row) in enumerate(weak.iterrows()):
            rater_rows.append(
                {
                    "library": args.library,
                    "fold": fold,
                    "cache_row_index": int(row["row_index"]),
                    "pair_uid": row["pair_uid"],
                    "weak_label": int(labels[index]),
                    "rater_score": float(rater["scores"][index]),
                    "rater_weight": float(rater["weights"][index]),
                }
            )
        all_indices = np.arange(len(weak), dtype=np.int64)
        conditions = [("full_unweighted", 1.0, -1, all_indices)]
        for fraction in diagnostic.FRACTIONS:
            selected = diagnostic.deterministic_top_indices(
                rater["scores"], labels, pair_uids, fraction
            )
            conditions.append(("trajectory_rater", fraction, -1, selected))
        for method, fraction, control, selected in conditions:
            print(
                "{} crossed fold {}: fitting {} {:.0f}% control {}".format(
                    args.library, fold, method, 100.0 * fraction, control
                ),
                flush=True,
            )
            model = diagnostic.fit_linear_head(
                x_tensor,
                y_tensor,
                selected,
                args.head_l2,
                args.head_max_iter,
                class_balanced=method != "full_unweighted",
            )
            logits = (
                torch.mv(test_tensor, model["weight"]) + model["bias"]
            ).detach().cpu().numpy().astype(np.float64)
            metadata = {
                "library": args.library,
                "fold": fold,
                "peptide_block": peptide_block,
                "affibody_block": affibody_block,
                "method": method,
                "fraction": float(fraction),
                "control": int(control),
                "source": "current_second_order_run",
            }
            metric_rows.append(dict(metadata, **diagnostic.ranking_metrics(test, logits)))
            condition_rows.append(
                dict(
                    metadata,
                    weak_eligible=len(weak),
                    weak_selected=len(selected),
                    weak_selected_negative=int(np.sum(labels[selected] == 0)),
                    weak_selected_positive=int(np.sum(labels[selected] == 1)),
                    selected_membership_sha256=diagnostic._membership_digest(pair_uids[selected]),
                    development_retention=len(development),
                    development_pairwise_comparisons=len(clean_pairs),
                    test_retention=len(test),
                    mint_zero_variance_features=mint_zero,
                    trajectory_zero_variance_features=trajectory_zero,
                    rater_entropy_kl=rater["entropy_kl"],
                    rater_negative_weight_entropy=rater["negative_weight_entropy"],
                    rater_positive_weight_entropy=rater["positive_weight_entropy"],
                    rater_negative_effective_sample_size=rater[
                        "negative_effective_sample_size"
                    ],
                    rater_positive_effective_sample_size=rater[
                        "positive_effective_sample_size"
                    ],
                    rater_negative_max_mean_one_weight=rater[
                        "negative_max_mean_one_weight"
                    ],
                    rater_positive_max_mean_one_weight=rater[
                        "positive_max_mean_one_weight"
                    ],
                    head_final_objective=model["final_objective"],
                    head_final_gradient_norm=model["gradient_norm"],
                    head_lbfgs_closure_calls=model["closure_calls"],
                )
            )
            for position, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    dict(
                        metadata,
                        pair_uid=row["pair_uid"],
                        peptide_uid=row["peptide_uid"],
                        affibody_uid=row["affibody_uid"],
                        target_retention=float(row["target_retention"]),
                        target_binder=int(row["target_binder"]),
                        score=float(logits[position]),
                        binder_probability=float(expit(logits[position])),
                    )
                )
        del x_tensor, y_tensor, trajectory_tensor, clean_tensor, test_tensor
        torch.cuda.empty_cache()

    current_metrics = pd.DataFrame(metric_rows)
    current_conditions = pd.DataFrame(condition_rows)
    current_predictions = pd.DataFrame(prediction_rows)
    metrics = pd.concat(
        [current_metrics, reference_frames["metrics.csv"]], ignore_index=True, sort=False
    )
    conditions = pd.concat(
        [current_conditions, reference_frames["conditions.csv"]], ignore_index=True, sort=False
    )
    predictions = pd.concat(
        [current_predictions, reference_frames["predictions.csv"]], ignore_index=True, sort=False
    )
    histories = pd.DataFrame(history_rows)
    comparisons = diagnostic.summarize_comparisons(
        metrics, args.random_controls, selected_method="trajectory_rater"
    )
    gate, gate_results = diagnostic.diagnostic_gate(comparisons)
    gate_result = bool(gate_results[args.library])
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_stability = (
        coefficients.groupby("feature", sort=False)["coefficient"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    sign_counts = coefficients.assign(
        positive=coefficients["coefficient"].gt(0).astype(int),
        negative=coefficients["coefficient"].lt(0).astype(int),
        zero=coefficients["coefficient"].eq(0).astype(int),
    ).groupby("feature", sort=False)[["positive", "negative", "zero"]].sum().reset_index()
    coefficient_stability = coefficient_stability.merge(sign_counts, on="feature", validate="one_to_one")
    coefficient_stability["majority_sign_folds"] = coefficient_stability[["positive", "negative"]].max(axis=1)
    coefficient_stability["sign_consistency_fraction"] = (
        coefficient_stability["majority_sign_folds"] / 9.0
    )
    summary = build_summary(
        metrics, gate, gate_result, histories, conditions, coefficient_stability
    )

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    frames = {
        "metrics.csv": metrics,
        "conditions.csv": conditions,
        "predictions.csv": predictions,
        "retention_split_membership.csv": pd.DataFrame(split_rows),
        "rater_scores.csv": pd.DataFrame(rater_rows),
        "rater_training_history.csv": histories,
        "rater_coefficients.csv": coefficients,
        "rater_coefficient_stability.csv": coefficient_stability,
        "matched_random_comparisons.csv": comparisons,
        "diagnostic_gate.csv": gate,
    }
    outputs = {}
    for name, frame in frames.items():
        path = output_dir / name
        diagnostic._write_csv(frame, path)
        outputs[name] = path
    summary_path = output_dir / "run_summary.md"
    diagnostic._write_text(summary, summary_path)
    outputs[summary_path.name] = summary_path
    elapsed = float(time.time() - started)
    script_path = Path(__file__).resolve()
    source_snapshot_path = output_dir / "producing_source_snapshot.py"
    with open(str(script_path), "r") as handle:
        diagnostic._write_text(handle.read(), source_snapshot_path)
    outputs[source_snapshot_path.name] = source_snapshot_path
    diagnostic._require(
        diagnostic.sha256_file(source_snapshot_path) == diagnostic.sha256_file(script_path),
        "source snapshot hash mismatch",
    )
    dependency_path = Path(diagnostic.__file__).resolve()
    dependency_snapshot_path = output_dir / "meta_gradient_dependency_snapshot.py"
    with open(str(dependency_path), "r") as handle:
        diagnostic._write_text(handle.read(), dependency_snapshot_path)
    outputs[dependency_snapshot_path.name] = dependency_snapshot_path
    diagnostic._require(
        diagnostic.sha256_file(dependency_snapshot_path) == diagnostic.sha256_file(dependency_path),
        "dependency snapshot hash mismatch",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": int(time.time()),
        "elapsed_seconds": elapsed,
        "host": socket.gethostname(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "protocol": {
            "library": args.library,
            "fold_mode": "crossed9",
            "outer_folds": 9,
            "rater": "12-parameter shared linear scorer on fixed trajectory summaries; classwise-standardized inputs; tanh-bounded scores; mean-one weights within class",
            "rater_parameter_count": len(COMPACT_TRAJECTORY_NAMES),
            "rater_feature_names": list(COMPACT_TRAJECTORY_NAMES),
            "rater_score_bound": 2.0,
            "learned_weight_normalization": "mean one separately within weak-label class; max/min ratio bounded by exp(4)",
            "inner_model": "2561-parameter linear head on fixed MINT features",
            "inner_steps": int(args.inner_steps),
            "inner_lr": float(args.inner_lr),
            "meta_steps": int(args.meta_steps),
            "meta_lr": float(args.meta_lr),
            "head_l2": float(args.head_l2),
            "rater_l2": float(args.rater_l2),
            "entropy_penalty": float(args.entropy_penalty),
            "random_controls": int(args.random_controls),
            "head_max_iter": int(args.head_max_iter),
            "test_retention_used_for_rater_training": False,
            "fold_independence": "crossed folds cover every measured cell once but reuse overlapping development and weak pools; folds are correlated, not independent replicates",
        },
        "gate_result": gate_result,
        "input": {
            "cache": {"path": str(args.cache.resolve()), "sha256": diagnostic.sha256_file(args.cache)},
            "cache_manifest": {
                "path": str(args.cache_manifest.resolve()),
                "sha256": diagnostic.sha256_file(args.cache_manifest),
                "schema_version": cache_manifest.get("contract", {})
                .get("payload", {})
                .get("schema_version"),
            },
            "trajectory": {
                "path": str(args.trajectory.resolve()),
                "sha256": diagnostic.sha256_file(args.trajectory),
            },
            "trajectory_manifest": {
                "path": str(args.trajectory_manifest.resolve()),
                "sha256": diagnostic.sha256_file(args.trajectory_manifest),
                "schema_version": trajectory_manifest.get("schema_version"),
            },
            "canonical_gradient_reference_manifest": {
                "path": str(reference_manifest_path.resolve()),
                "sha256": diagnostic.sha256_file(reference_manifest_path),
                "source_sha256": reference_manifest.get("source", {}).get("sha256"),
            },
        },
        "source": {"path": str(script_path), "sha256": diagnostic.sha256_file(script_path)},
        "source_snapshot": {
            "path": str(source_snapshot_path.resolve()),
            "sha256": diagnostic.sha256_file(source_snapshot_path),
        },
        "dependency_snapshot": {
            "source_path": str(dependency_path),
            "path": str(dependency_snapshot_path.resolve()),
            "sha256": diagnostic.sha256_file(dependency_snapshot_path),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": diagnostic.sha256_file(path),
                "mode": diagnostic._private_mode(path),
                "rows": int(len(frames[name])) if name in frames else None,
            }
            for name, path in outputs.items()
        },
    }
    diagnostic._write_json(manifest, output_dir / "manifest.json")
    diagnostic._require(diagnostic._private_mode(output_dir) == "0700", "directory is not private")
    for path in output_dir.iterdir():
        diagnostic._require(diagnostic._private_mode(path) == "0600", "output is not private")
    print(json.dumps({"gate_result": gate_result, "elapsed_seconds": elapsed}, indent=2))
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
