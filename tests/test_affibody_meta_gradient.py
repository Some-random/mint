import math

import numpy as np
import pandas as pd

from downstream.AffibodyMHC import run_meta_gradient_diagnostic as diagnostic
from downstream.AffibodyMHC import run_second_order_hybrid_rater as hybrid_rater
from downstream.AffibodyMHC import run_second_order_trajectory_rater as trajectory_rater


def _retention_rectangle(n_peptides=6, n_affibodies=6):
    rows = []
    for peptide in range(n_peptides):
        for affibody in range(n_affibodies):
            rows.append(
                {
                    "pair_uid": "pair-{}-{}".format(peptide, affibody),
                    "peptide_uid": "pep-{}".format(peptide),
                    "affibody_uid": "aff-{}".format(affibody),
                    "target_retention": float(10 * peptide + affibody),
                    "target_binder": int(affibody >= n_affibodies // 2),
                }
            )
    return pd.DataFrame(rows)


def test_outer_roles_are_identity_disjoint_and_target_free():
    frame = _retention_rectangle()
    roles, held_peptides, held_affibodies, _, _ = diagnostic.make_outer_roles(
        frame, "LibA", fold=1, seed=17
    )
    development = frame.loc[roles == "development"]
    test = frame.loc[roles == "test"]
    assert set(test["peptide_uid"]) == held_peptides
    assert set(test["affibody_uid"]) == held_affibodies
    assert set(development["peptide_uid"]).isdisjoint(test["peptide_uid"])
    assert set(development["affibody_uid"]).isdisjoint(test["affibody_uid"])

    changed_targets = frame.copy()
    changed_targets["target_retention"] = changed_targets["target_retention"].iloc[::-1].to_numpy()
    changed_roles, _, _, _, _ = diagnostic.make_outer_roles(
        changed_targets, "LibA", fold=1, seed=17
    )
    assert np.array_equal(roles, changed_roles)


def test_crossed_blocks_cover_each_retention_cell_once():
    frame = _retention_rectangle()
    test_count = np.zeros(len(frame), dtype=int)
    seen_blocks = set()
    for fold in range(9):
        roles, _, _, peptide_block, affibody_block = diagnostic.make_outer_roles(
            frame, "LibA", fold=fold, seed=17, fold_mode="crossed9"
        )
        test_count += roles == "test"
        seen_blocks.add((peptide_block, affibody_block))
    assert seen_blocks == {(peptide, affibody) for peptide in range(3) for affibody in range(3)}
    assert np.array_equal(test_count, np.ones(len(frame), dtype=int))


def test_weak_roles_exclude_either_held_partner():
    weak = pd.DataFrame(
        {
            "peptide_uid": ["p0", "p0", "p1", "p1"],
            "affibody_uid": ["a0", "a1", "a0", "a1"],
        }
    )
    roles = diagnostic.weak_outer_roles(weak, {"p0"}, {"a0"})
    assert roles.tolist() == [
        "excluded_both_held",
        "excluded_held_peptide",
        "excluded_held_affibody",
        "eligible",
    ]


def test_alignment_and_random_selection_preserve_class_counts():
    labels = np.asarray([0] * 20 + [1] * 12)
    pair_uids = np.asarray(["pair-{:02d}".format(index) for index in range(len(labels))])
    score = np.linspace(-1.0, 1.0, len(labels))
    top = diagnostic.deterministic_top_indices(score, labels, pair_uids, 0.5)
    assert np.bincount(labels[top], minlength=2).tolist() == [10, 6]
    memberships = []
    for control in range(5):
        selected = diagnostic.deterministic_random_indices(
            labels, pair_uids, 0.5, seed=17, library="LibA", fold=0, control=control
        )
        assert np.bincount(labels[selected], minlength=2).tolist() == [10, 6]
        memberships.append(tuple(selected.tolist()))
    assert len(set(memberships)) == 5


def test_pairwise_metric_is_within_peptide():
    frame = pd.DataFrame(
        {
            "peptide_uid": ["p0", "p0", "p1", "p1"],
            "affibody_uid": ["a0", "a1", "a0", "a1"],
            "target_retention": [90.0, 10.0, 20.0, 80.0],
            "target_binder": [1, 0, 0, 1],
        }
    )
    perfect = diagnostic.ranking_metrics(frame, np.asarray([2.0, 0.0, 0.0, 2.0]))
    reversed_second = diagnostic.ranking_metrics(frame, np.asarray([2.0, 0.0, 2.0, 0.0]))
    assert perfect["within_peptide_pair_count"] == 2
    assert perfect["within_peptide_pairwise_accuracy_micro"] == 1.0
    assert reversed_second["within_peptide_pairwise_accuracy_micro"] == 0.5


def test_clean_pair_differences_are_oriented_high_minus_low():
    frame = pd.DataFrame(
        {
            "peptide_uid": ["p0", "p0", "p1", "p1"],
            "target_retention": [80.0, 20.0, 10.0, 70.0],
        }
    )
    features = np.asarray([[3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 5.0]], dtype=np.float32)
    differences, metadata = diagnostic.make_pair_differences(frame, features)
    assert np.array_equal(differences, np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))
    assert len(metadata) == 2


def test_classwise_trajectory_standardization_cancels_constants():
    features = np.asarray(
        [[3.0, 0.0], [3.0, 2.0], [7.0, 1.0], [7.0, 5.0]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1])
    transformed, _, _, zero = trajectory_rater.standardize_trajectory(features, labels)
    assert zero[:, 0].tolist() == [True, True]
    assert np.array_equal(transformed[:, 0], np.zeros(4, dtype=np.float32))
    assert np.allclose(transformed[labels == 0, 1], [-1.0, 1.0])
    assert np.allclose(transformed[labels == 1, 1], [-1.0, 1.0])


def test_exact_meta_gradient_matches_finite_difference():
    import torch

    dtype = torch.float64
    x = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=dtype
    )
    y = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    trajectory = torch.tensor(
        [[-1.0, 0.5], [1.0, -0.5], [-0.7, -1.0], [0.7, 1.0]], dtype=dtype
    )
    clean = torch.tensor([[1.0, -0.5], [-0.5, 1.0]], dtype=dtype)
    theta = torch.tensor([0.2, -0.1], dtype=dtype, requires_grad=True)

    def value(parameter):
        objective, _ = trajectory_rater.second_order_objective(
            parameter,
            x,
            y,
            trajectory,
            clean,
            inner_steps=2,
            inner_lr=0.01,
            head_l2=0.005,
            rater_l2=0.001,
            entropy_penalty=0.01,
        )
        return objective

    objective = value(theta)
    gradient = torch.autograd.grad(objective, theta)[0].detach().numpy()
    epsilon = 1e-5
    finite = []
    for index in range(len(theta)):
        plus = theta.detach().clone()
        minus = theta.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite.append(float((value(plus) - value(minus)) / (2.0 * epsilon)))
    assert np.allclose(gradient, finite, rtol=2e-4, atol=2e-6)


def test_rater_weights_are_mean_one_and_bounded_within_class():
    import torch

    scores = torch.tensor([-100.0, 0.0, 100.0, -100.0, 100.0])
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
    _, mean_one, _, ess, _ = trajectory_rater.class_normalized_weights(scores, labels)
    for label in (0, 1):
        local = mean_one[labels == float(label)]
        assert torch.allclose(local.mean(), torch.tensor(1.0), atol=1e-6)
        assert float(local.max() / local.min()) <= math.exp(4.0) + 1e-4
    assert bool(torch.all(ess > 0.0))


def test_hybrid_pca_excludes_held_identities_and_has_eight_dimensions():
    weak = pd.DataFrame(
        [
            {
                "pair_uid": "weak-{}-{}".format(peptide, affibody),
                "peptide_uid": "pep-{}".format(peptide),
                "affibody_uid": "aff-{}".format(affibody),
            }
            for peptide in range(6)
            for affibody in range(6)
        ]
    )
    held_peptides = {"pep-0", "pep-1"}
    held_affibodies = {"aff-0", "aff-1"}
    roles = diagnostic.weak_outer_roles(weak, held_peptides, held_affibodies)
    eligible = weak.loc[roles == "eligible"].reset_index(drop=True)
    assert set(eligible["peptide_uid"]).isdisjoint(held_peptides)
    assert set(eligible["affibody_uid"]).isdisjoint(held_affibodies)

    rng = np.random.RandomState(17)
    all_features = rng.normal(size=(len(weak), 24)).astype(np.float32)
    fit_features = all_features[roles == "eligible"]
    first = hybrid_rater.fit_weak_mint_pcs(fit_features, n_components=8, seed=23)
    second = hybrid_rater.fit_weak_mint_pcs(fit_features, n_components=8, seed=23)
    assert first["scores"].shape == (len(eligible), 8)
    assert first["components"].shape == (8, 24)
    assert first["explained_variance_ratio"].shape == (8,)
    assert np.array_equal(first["scores"], second["scores"])
    assert np.array_equal(first["components"], second["components"])
    reconstructed = (
        fit_features - first["pca_mean"].astype(np.float32)
    ) @ first["components"].T
    assert np.allclose(first["scores"], reconstructed, rtol=2e-6, atol=2e-6)
