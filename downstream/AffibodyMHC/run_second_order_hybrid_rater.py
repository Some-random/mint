#!/usr/bin/env python
"""Post-hoc 20-parameter trajectory-plus-MINT DataRater sensitivity.

This is one fixed sensitivity analysis after the 12-feature trajectory-only
rater failed its crossed-block gate.  It appends exactly eight unsupervised
principal-component scores from the frozen MINT pair representation.  In every
crossed block, feature scaling and PCA are fit only on eligible weak-label
rows; retention rows and held partner identities never enter PCA.

The shared linear rater therefore has exactly 20 weights and otherwise uses
the trajectory-only protocol unchanged: classwise input standardization,
tanh-bounded mean-one weights within class, two exact differentiable inner
updates, fixed optimization, and held-partner evaluation against the same five
canonical random controls.
"""

from __future__ import print_function

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import run_meta_gradient_diagnostic as diagnostic
from downstream.AffibodyMHC import run_second_order_trajectory_rater as trajectory_rater


SCHEMA_VERSION = "affibody-second-order-hybrid-rater-v1"
N_MINT_PCS = 8
HYBRID_FEATURE_NAMES = trajectory_rater.COMPACT_TRAJECTORY_NAMES + tuple(
    "mint_pc_{:02d}".format(index + 1) for index in range(N_MINT_PCS)
)


def fit_weak_mint_pcs(features, n_components=N_MINT_PCS, seed=20260807):
    """Fit deterministic randomized PCA using only the supplied weak pool."""
    features = np.asarray(features, dtype=np.float32)
    diagnostic._require(features.ndim == 2, "PCA features must be a matrix")
    diagnostic._require(
        1 <= int(n_components) <= min(features.shape), "invalid PCA component count"
    )
    diagnostic._require(bool(np.isfinite(features).all()), "non-finite PCA input")
    pca = PCA(
        n_components=int(n_components),
        svd_solver="randomized",
        iterated_power=4,
        random_state=int(seed),
    )
    # Use the public transform path after fitting so the materialized PCA mean
    # and loadings exactly define the scores used by the downstream rater.
    pca.fit(features)
    scores = pca.transform(features).astype(np.float32)
    components = np.asarray(pca.components_, dtype=np.float32).copy()
    pivots = np.argmax(np.abs(components), axis=1).astype(np.int64)
    signs = np.ones(len(components), dtype=np.float32)
    for component in range(len(components)):
        if components[component, pivots[component]] < 0.0:
            signs[component] = -1.0
            components[component] *= -1.0
            scores[:, component] *= -1.0
    diagnostic._require(
        scores.shape == (len(features), int(n_components)), "PCA score shape mismatch"
    )
    diagnostic._require(bool(np.isfinite(scores).all()), "non-finite PCA score")
    diagnostic._require(bool(np.isfinite(components).all()), "non-finite PCA loading")
    diagnostic._require(
        bool(np.all(components[np.arange(len(components)), pivots] >= 0.0)),
        "PCA sign orientation failed",
    )
    return {
        "scores": scores,
        "components": components,
        "explained_variance": np.asarray(pca.explained_variance_, dtype=np.float64),
        "explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        "singular_values": np.asarray(pca.singular_values_, dtype=np.float64),
        "pca_mean": np.asarray(pca.mean_, dtype=np.float64),
        "pivots": pivots,
        "orientation_sign": signs,
    }


def load_trajectory_only_reference(run_dir, library):
    """Load the primary trajectory-rater rows and its exact split membership."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = diagnostic._read_json(manifest_path)
    diagnostic._require(
        manifest.get("schema_version") == trajectory_rater.SCHEMA_VERSION,
        "trajectory-rater reference schema mismatch",
    )
    diagnostic._require(
        manifest.get("source", {}).get("sha256")
        == diagnostic.sha256_file(trajectory_rater.__file__),
        "trajectory-rater reference source mismatch",
    )
    frames = {}
    for name in ("metrics.csv", "conditions.csv", "predictions.csv"):
        path = run_dir / name
        expected = manifest.get("outputs", {}).get(name, {}).get("sha256")
        diagnostic._require(expected == diagnostic.sha256_file(path), "reference hash mismatch")
        frame = pd.read_csv(path)
        frame = frame.loc[frame["method"].isin(("trajectory_rater", "full_unweighted"))].copy()
        frame["method"] = frame["method"].replace(
            {"trajectory_rater": "trajectory_only_rater"}
        )
        frame["source"] = "primary_trajectory_only_reference"
        frames[name] = frame
    split_name = "retention_split_membership.csv"
    split_path = run_dir / split_name
    diagnostic._require(
        manifest.get("outputs", {}).get(split_name, {}).get("sha256")
        == diagnostic.sha256_file(split_path),
        "trajectory-rater split hash mismatch",
    )
    frames[split_name] = pd.read_csv(split_path)
    diagnostic._require(
        len(frames["metrics.csv"].loc[frames["metrics.csv"]["method"].eq("trajectory_only_rater")])
        == 27,
        "trajectory-only reference lacks conditions",
    )
    diagnostic._require(manifest.get("protocol", {}).get("library") == library, "library mismatch")
    return frames, manifest, manifest_path


def _write_npz(path, arrays):
    diagnostic._require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with open(str(temporary), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def build_summary(metrics, gate, gate_result, histories, conditions, pca_diagnostics):
    lines = [
        "# Post-hoc 20-parameter hybrid-rater sensitivity",
        "",
        "This sensitivity added eight weak-pool-only MINT principal components to the "
        "same twelve trajectory summaries used by the primary trajectory-only rater. "
        "The design was fixed once and no alternatives were tried.",
        "",
        "| Weak rows kept | Hybrid rater | Trajectory-only rater | Gradient alignment | Random mean | Hybrid delta | Positive blocks | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in gate.sort_values("fraction").iterrows():
        fraction = float(row["fraction"])

        def mean_for(method):
            values = metrics.loc[
                metrics["method"].eq(method) & metrics["fraction"].eq(fraction),
                "within_peptide_pairwise_accuracy_micro",
            ]
            diagnostic._require(not values.empty, "summary lacks {}".format(method))
            return float(values.mean())

        lines.append(
            "| {:.0f}% | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} | {}/{} | {} |".format(
                100.0 * fraction,
                mean_for("hybrid_rater"),
                mean_for("trajectory_only_rater"),
                mean_for("gradient_alignment"),
                mean_for("random"),
                float(row["mean_alignment_minus_random"]),
                int(row["positive_folds"]),
                int(row["total_folds"]),
                "pass" if int(row["fraction_pass"]) else "fail",
            )
        )
    state = conditions.loc[conditions["method"].eq("hybrid_rater")].groupby("fold").first()
    negative_ess_fraction = (
        state["rater_negative_effective_sample_size"] / state["weak_eligible_negative"]
    )
    positive_ess_fraction = (
        state["rater_positive_effective_sample_size"] / state["weak_eligible_positive"]
    )
    initial = histories.sort_values(["fold", "meta_step"]).groupby("fold").first()
    final = histories.sort_values(["fold", "meta_step"]).groupby("fold").last()
    explained = pca_diagnostics.groupby("fold")["explained_variance_ratio"].sum()
    lines.extend(
        [
            "",
            "Overall fixed pilot gate: **{}**. This is explicitly post-hoc; the trajectory-only "
            "result remains the primary DataRater result.".format("pass" if gate_result else "fail"),
            "",
            "The eight PCs explained {:.1f}%--{:.1f}% of eligible-weak MINT variance across "
            "blocks. Clean development loss changed from {:.3f} to {:.3f}. Median effective "
            "sample-size fractions were {:.3f} (negative) and {:.3f} (positive); the largest "
            "mean-one weight was {:.3f}.".format(
                100.0 * float(explained.min()),
                100.0 * float(explained.max()),
                float(initial["clean_pairwise_loss"].mean()),
                float(final["clean_pairwise_loss"].mean()),
                float(negative_ess_fraction.median()),
                float(positive_ess_fraction.median()),
                float(
                    max(
                        state["rater_negative_max_mean_one_weight"].max(),
                        state["rater_positive_max_mean_one_weight"].max(),
                    )
                ),
            ),
            "",
            "Every measured LibA cell is tested once, but the nine blocks reuse overlapping "
            "development and weak pools and are therefore correlated rather than independent replicates.",
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
    parser.add_argument(
        "--gradient-diagnostic-dir",
        type=Path,
        default=REPO_ROOT / "private_data/experiments/meta_gradient_crossed9_liba_canonical_v1",
    )
    parser.add_argument(
        "--trajectory-rater-dir",
        type=Path,
        default=REPO_ROOT
        / "private_data/experiments/second_order_trajectory_rater_liba_crossed9_canonical_v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    diagnostic._require(int(args.random_controls) == 5, "this sensitivity fixes five controls")
    diagnostic._require(args.library == "LibA", "this fixed sensitivity is LibA-only")
    output_dir = diagnostic._validate_private_output(args.output_dir)
    diagnostic._require(not output_dir.exists(), "output directory exists; refusing overwrite")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = diagnostic._import_torch()
    diagnostic._require(torch.cuda.is_available(), "CUDA is required")
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
    raw_trajectory, raw_trajectory_names, trajectory_index, trajectory_manifest = (
        trajectory_rater.load_trajectory(
            args.trajectory, args.trajectory_manifest, cache_frame
        )
    )
    gradient_frames, gradient_manifest, gradient_manifest_path = (
        trajectory_rater.load_gradient_reference(args.gradient_diagnostic_dir, args.library)
    )
    primary_frames, primary_manifest, primary_manifest_path = load_trajectory_only_reference(
        args.trajectory_rater_dir, args.library
    )
    gradient_protocol = gradient_manifest["protocol"]
    primary_protocol = primary_manifest["protocol"]
    diagnostic._require(int(gradient_protocol["seed"]) == int(args.seed), "seed mismatch")
    diagnostic._require(
        int(gradient_protocol["random_controls"]) == int(args.random_controls),
        "control mismatch",
    )
    diagnostic._require(
        int(gradient_protocol["lbfgs_max_iter"]) == int(args.head_max_iter),
        "head iteration mismatch",
    )
    diagnostic._require(
        float(gradient_protocol["l2"]) == float(args.head_l2), "head L2 mismatch"
    )
    diagnostic._require(
        primary_manifest["input"]["trajectory"]["sha256"]
        == diagnostic.sha256_file(args.trajectory),
        "primary rater used a different trajectory artifact",
    )
    diagnostic._require(
        primary_manifest["input"]["cache"]["sha256"]
        == diagnostic.sha256_file(args.cache),
        "primary rater used a different MINT cache",
    )
    diagnostic._require(
        primary_manifest["input"]["canonical_gradient_reference_manifest"]["sha256"]
        == diagnostic.sha256_file(gradient_manifest_path),
        "primary rater used a different canonical gradient reference",
    )
    fixed_primary_values = {
        "inner_steps": int(args.inner_steps),
        "inner_lr": float(args.inner_lr),
        "meta_steps": int(args.meta_steps),
        "meta_lr": float(args.meta_lr),
        "head_l2": float(args.head_l2),
        "rater_l2": float(args.rater_l2),
        "entropy_penalty": float(args.entropy_penalty),
        "head_max_iter": int(args.head_max_iter),
        "random_controls": int(args.random_controls),
    }
    for key, expected in fixed_primary_values.items():
        diagnostic._require(
            primary_protocol.get(key) == expected,
            "primary rater protocol mismatch for {}".format(key),
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

    metric_rows = []
    condition_rows = []
    prediction_rows = []
    rater_rows = []
    history_rows = []
    coefficient_rows = []
    split_rows = []
    pca_rows = []
    pca_components = []
    pca_explained = []
    pca_singular = []
    pca_means = []
    pca_input_means = []
    pca_input_scales = []
    pca_pivots = []
    pca_membership = []

    for fold in range(9):
        print("{} hybrid fold {}: preparing weak-only PCA".format(args.library, fold), flush=True)
        roles, held_peptides, held_affibodies, peptide_block, affibody_block = (
            diagnostic.make_outer_roles(retention, args.library, fold, args.seed, "crossed9")
        )
        reference_split = gradient_frames["retention_split_membership.csv"].loc[
            gradient_frames["retention_split_membership.csv"]["fold"].eq(fold)
        ]
        primary_split = primary_frames["retention_split_membership.csv"].loc[
            primary_frames["retention_split_membership.csv"]["fold"].eq(fold)
        ]
        current_split = dict(zip(retention["pair_uid"].astype(str), roles.astype(str)))
        diagnostic._require(
            current_split
            == dict(zip(reference_split["pair_uid"].astype(str), reference_split["role"])),
            "split mismatch",
        )
        diagnostic._require(
            current_split
            == dict(zip(primary_split["pair_uid"].astype(str), primary_split["role"])),
            "trajectory-rater split mismatch",
        )
        weak_roles = diagnostic.weak_outer_roles(weak_all, held_peptides, held_affibodies)
        weak = weak_all.loc[weak_roles == "eligible"].copy().reset_index(drop=True)
        development = retention.loc[roles == "development"].copy().reset_index(drop=True)
        test = retention.loc[roles == "test"].copy().reset_index(drop=True)
        diagnostic._require(
            set(weak["peptide_uid"]).isdisjoint(set(test["peptide_uid"])),
            "hybrid PCA leaks held peptide",
        )
        diagnostic._require(
            set(weak["affibody_uid"]).isdisjoint(set(test["affibody_uid"])),
            "hybrid PCA leaks held Affibody",
        )
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
        x_weak, x_retention, x_mean, x_scale, mint_zero = diagnostic.standardize_from_training(
            cache_features[cache_indices], cache_features[retention_indices]
        )
        x_development = x_retention[: len(development)]
        x_test = x_retention[len(development) :]
        clean_difference, clean_pairs = diagnostic.make_pair_differences(
            development, x_development
        )
        labels = weak["weak_label"].to_numpy(dtype=int)
        pair_uids = weak["pair_uid"].astype(str).to_numpy()
        membership_hash = diagnostic._membership_digest(pair_uids)
        pca = fit_weak_mint_pcs(x_weak, N_MINT_PCS, args.seed + fold)
        pca_components.append(pca["components"])
        pca_explained.append(pca["explained_variance_ratio"])
        pca_singular.append(pca["singular_values"])
        pca_means.append(pca["pca_mean"])
        pca_input_means.append(x_mean)
        pca_input_scales.append(x_scale)
        pca_pivots.append(pca["pivots"])
        pca_membership.append(membership_hash)
        cumulative = np.cumsum(pca["explained_variance_ratio"])
        for component in range(N_MINT_PCS):
            pca_rows.append(
                {
                    "library": args.library,
                    "fold": fold,
                    "peptide_block": peptide_block,
                    "affibody_block": affibody_block,
                    "component": component + 1,
                    "fit_rows": len(weak),
                    "fit_membership_sha256": membership_hash,
                    "random_state": int(args.seed + fold),
                    "explained_variance": float(pca["explained_variance"][component]),
                    "explained_variance_ratio": float(
                        pca["explained_variance_ratio"][component]
                    ),
                    "cumulative_explained_variance_ratio": float(cumulative[component]),
                    "singular_value": float(pca["singular_values"][component]),
                    "orientation_pivot_feature": int(pca["pivots"][component]),
                    "orientation_pivot_loading": float(
                        pca["components"][component, pca["pivots"][component]]
                    ),
                }
            )

        trajectory_indices = trajectory_index[cache_indices]
        diagnostic._require(bool(np.all(trajectory_indices >= 0)), "weak row lacks trajectory")
        compact = trajectory_rater.compact_trajectory_features(
            raw_trajectory[trajectory_indices], raw_trajectory_names
        )
        hybrid_raw = np.column_stack([compact, pca["scores"]]).astype(np.float32)
        diagnostic._require(
            hybrid_raw.shape == (len(weak), len(HYBRID_FEATURE_NAMES)),
            "hybrid feature dimension mismatch",
        )
        hybrid, _, _, hybrid_zero_masks = trajectory_rater.standardize_trajectory(
            hybrid_raw, labels
        )

        x_tensor = torch.as_tensor(x_weak, dtype=torch.float32, device=device)
        y_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
        hybrid_tensor = torch.as_tensor(hybrid, dtype=torch.float32, device=device)
        clean_tensor = torch.as_tensor(clean_difference, dtype=torch.float32, device=device)
        test_tensor = torch.as_tensor(x_test, dtype=torch.float32, device=device)
        print("{} hybrid fold {}: fitting exact 20-weight rater".format(args.library, fold), flush=True)
        rater = trajectory_rater.learn_rater(
            x_tensor,
            y_tensor,
            hybrid_tensor,
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
        for name, value in zip(HYBRID_FEATURE_NAMES, rater["theta"]):
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
        for fraction in diagnostic.FRACTIONS:
            selected = diagnostic.deterministic_top_indices(
                rater["scores"], labels, pair_uids, fraction
            )
            print(
                "{} hybrid fold {}: fitting top {:.0f}%".format(
                    args.library, fold, 100.0 * fraction
                ),
                flush=True,
            )
            model = diagnostic.fit_linear_head(
                x_tensor, y_tensor, selected, args.head_l2, args.head_max_iter
            )
            logits = (
                torch.mv(test_tensor, model["weight"]) + model["bias"]
            ).detach().cpu().numpy().astype(np.float64)
            metadata = {
                "library": args.library,
                "fold": fold,
                "peptide_block": peptide_block,
                "affibody_block": affibody_block,
                "method": "hybrid_rater",
                "fraction": float(fraction),
                "control": -1,
                "source": "current_posthoc_hybrid_run",
            }
            metric_rows.append(dict(metadata, **diagnostic.ranking_metrics(test, logits)))
            condition_rows.append(
                dict(
                    metadata,
                    weak_eligible=len(weak),
                    weak_eligible_negative=int(np.sum(labels == 0)),
                    weak_eligible_positive=int(np.sum(labels == 1)),
                    weak_selected=len(selected),
                    weak_selected_negative=int(np.sum(labels[selected] == 0)),
                    weak_selected_positive=int(np.sum(labels[selected] == 1)),
                    selected_membership_sha256=diagnostic._membership_digest(pair_uids[selected]),
                    pca_fit_membership_sha256=membership_hash,
                    development_retention=len(development),
                    development_pairwise_comparisons=len(clean_pairs),
                    test_retention=len(test),
                    mint_zero_variance_features=mint_zero,
                    hybrid_class_feature_zero_variance_cells=int(hybrid_zero_masks.sum()),
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
        del x_tensor, y_tensor, hybrid_tensor, clean_tensor, test_tensor
        torch.cuda.empty_cache()

    current_metrics = pd.DataFrame(metric_rows)
    current_conditions = pd.DataFrame(condition_rows)
    current_predictions = pd.DataFrame(prediction_rows)
    metrics = pd.concat(
        [
            current_metrics,
            gradient_frames["metrics.csv"],
            primary_frames["metrics.csv"],
        ],
        ignore_index=True,
        sort=False,
    )
    conditions = pd.concat(
        [
            current_conditions,
            gradient_frames["conditions.csv"],
            primary_frames["conditions.csv"],
        ],
        ignore_index=True,
        sort=False,
    )
    predictions = pd.concat(
        [
            current_predictions,
            gradient_frames["predictions.csv"],
            primary_frames["predictions.csv"],
        ],
        ignore_index=True,
        sort=False,
    )
    histories = pd.DataFrame(history_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    pca_diagnostics = pd.DataFrame(pca_rows)
    comparisons = diagnostic.summarize_comparisons(
        metrics, args.random_controls, selected_method="hybrid_rater"
    )
    gate, gate_results = diagnostic.diagnostic_gate(comparisons)
    gate_result = bool(gate_results[args.library])
    summary = build_summary(
        metrics, gate, gate_result, histories, current_conditions, pca_diagnostics
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
        "pca_diagnostics.csv": pca_diagnostics,
        "matched_random_comparisons.csv": comparisons,
        "diagnostic_gate.csv": gate,
    }
    outputs = {}
    for name, frame in frames.items():
        path = output_dir / name
        diagnostic._write_csv(frame, path)
        outputs[name] = path
    pca_path = output_dir / "weak_pool_mint_pca.npz"
    _write_npz(
        pca_path,
        {
            "fold": np.arange(9, dtype=np.int8),
            "components": np.asarray(pca_components, dtype=np.float32),
            "explained_variance_ratio": np.asarray(pca_explained, dtype=np.float64),
            "singular_values": np.asarray(pca_singular, dtype=np.float64),
            "pca_mean": np.asarray(pca_means, dtype=np.float64),
            "input_standardization_mean": np.asarray(pca_input_means, dtype=np.float64),
            "input_standardization_scale": np.asarray(pca_input_scales, dtype=np.float64),
            "orientation_pivot": np.asarray(pca_pivots, dtype=np.int64),
            "fit_membership_sha256": np.asarray(pca_membership, dtype="U64"),
        },
    )
    outputs[pca_path.name] = pca_path
    summary_path = output_dir / "run_summary.md"
    diagnostic._write_text(summary, summary_path)
    outputs[summary_path.name] = summary_path

    script_path = Path(__file__).resolve()
    snapshots = {
        "producing_source_snapshot.py": script_path,
        "trajectory_rater_dependency_snapshot.py": Path(trajectory_rater.__file__).resolve(),
        "meta_gradient_dependency_snapshot.py": Path(diagnostic.__file__).resolve(),
    }
    for name, source in snapshots.items():
        target = output_dir / name
        with open(str(source), "r") as handle:
            diagnostic._write_text(handle.read(), target)
        diagnostic._require(
            diagnostic.sha256_file(target) == diagnostic.sha256_file(source),
            "source snapshot mismatch",
        )
        outputs[name] = target

    elapsed = float(time.time() - started)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": int(time.time()),
        "elapsed_seconds": elapsed,
        "host": socket.gethostname(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "status": "post_hoc_sensitivity; trajectory-only rater remains primary",
        "protocol": {
            "library": args.library,
            "seed": int(args.seed),
            "fold_mode": "crossed9",
            "outer_folds": 9,
            "rater_parameter_count": len(HYBRID_FEATURE_NAMES),
            "rater_feature_names": list(HYBRID_FEATURE_NAMES),
            "pca_components": N_MINT_PCS,
            "pca_fit_scope": "only eligible weak-pool frozen MINT features within each outer block",
            "pca_solver": "sklearn randomized PCA, iterated_power=4, default oversampling=10, random_state=seed+fold",
            "inner_steps": int(args.inner_steps),
            "inner_lr": float(args.inner_lr),
            "meta_steps": int(args.meta_steps),
            "meta_lr": float(args.meta_lr),
            "head_l2": float(args.head_l2),
            "rater_l2": float(args.rater_l2),
            "entropy_penalty": float(args.entropy_penalty),
            "head_max_iter": int(args.head_max_iter),
            "random_controls": int(args.random_controls),
            "test_retention_used_for_pca_or_rater": False,
            "fold_independence": "crossed blocks are correlated, not independent replicates",
            "alternative_hybrid_designs_tried": 0,
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
            "gradient_reference_manifest": {
                "path": str(gradient_manifest_path.resolve()),
                "sha256": diagnostic.sha256_file(gradient_manifest_path),
            },
            "trajectory_rater_reference_manifest": {
                "path": str(primary_manifest_path.resolve()),
                "sha256": diagnostic.sha256_file(primary_manifest_path),
            },
        },
        "source": {"path": str(script_path), "sha256": diagnostic.sha256_file(script_path)},
        "source_snapshots": {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": diagnostic.sha256_file(output_dir / name),
            }
            for name in snapshots
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
    diagnostic._require(diagnostic._private_mode(output_dir) == "0700", "directory mode")
    for path in output_dir.iterdir():
        diagnostic._require(diagnostic._private_mode(path) == "0600", "file mode")
    print(json.dumps({"gate_result": gate_result, "elapsed_seconds": elapsed}, indent=2))
    return manifest


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
