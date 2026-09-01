import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched


def test_balanced_class_weights_match_sklearn_definition_and_primary_counts():
    weights = matched.balanced_class_weights([0, 0, 0, 1])
    assert weights == {0: 2.0 / 3.0, 1: 2.0}
    assert 3 * weights[0] == pytest.approx(weights[1])
    assert (3 * weights[0] + weights[1]) / 4 == pytest.approx(1.0)

    liba = matched.balanced_class_weights(np.r_[np.zeros(11222), np.ones(11320)])
    assert liba[0] == pytest.approx(1.0043664230974871)
    assert liba[1] == pytest.approx(0.9956713780918728)
    libb = matched.balanced_class_weights(np.r_[np.zeros(6923), np.ones(23725)])
    assert libb[0] == pytest.approx(2.213491261014011)
    assert libb[1] == pytest.approx(0.6459009483667018)


def test_weighted_bce_is_per_example_weighted_and_row_normalized():
    logits = torch.tensor([-1.0, 0.5, 1.5, -0.25], requires_grad=True)
    target = torch.tensor([0.0, 0.0, 0.0, 1.0])
    weights = matched.balanced_class_weights(target.numpy().astype(int))
    observed = matched.weighted_bce_mean(logits, target, weights)
    elementwise = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    expected_weight = torch.tensor([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0])
    expected = torch.sum(elementwise * expected_weight) / 4.0
    assert observed.item() == pytest.approx(expected.item())
    observed.backward()
    observed_gradient = logits.grad.detach().clone()
    logits.grad.zero_()
    expected.backward()
    torch.testing.assert_close(observed_gradient, logits.grad)


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_balanced_class_weights_reject_missing_class(labels):
    with pytest.raises(ValueError, match="both classes"):
        matched.balanced_class_weights(labels)


def _prediction_rows(fold, epoch, labels, probabilities, arm="lora_cross", seed=7):
    return pd.DataFrame(
        {
            "pair_uid": ["f{}-e{}-{}".format(fold, epoch, i) for i in range(len(labels))],
            "weak_label": labels,
            "epoch": epoch,
            "probability": probabilities,
            "arm": arm,
            "training_seed": seed,
            "fold": fold,
        }
    )


def test_epoch_selection_uses_row_pooled_log_loss_not_mean_fold_loss():
    # The large fold strongly favors epoch 1.  A two-row fold strongly favors
    # epoch 2; giving folds equal weight would overstate those two rows.
    labels_large = np.tile([0, 1], 50)
    epoch1_large = np.where(labels_large == 1, 0.90, 0.10)
    epoch2_large = np.where(labels_large == 1, 0.75, 0.25)
    labels_small = np.array([0, 1])
    frames = []
    for epoch in (0, 1, 2, 3):
        if epoch == 1:
            large = epoch1_large
            small = np.array([0.99, 0.01])
        elif epoch == 2:
            large = epoch2_large
            small = np.array([0.10, 0.90])
        else:
            large = np.full(100, 0.5)
            small = np.full(2, 0.5)
        frames.append(_prediction_rows(0, epoch, labels_large, large))
        frames.append(_prediction_rows(1, epoch, labels_small, small))
    selected, history = matched.select_epoch(
        pd.concat(frames, ignore_index=True), "lora_cross", 7, 3
    )
    assert selected == 1
    assert sum(row["selected"] for row in history) == 1


def test_epoch_zero_is_a_valid_deterministic_tie_winner():
    frames = []
    for epoch in range(4):
        frames.append(_prediction_rows(0, epoch, [0, 1], [0.25, 0.75]))
    selected, history = matched.select_epoch(
        pd.concat(frames, ignore_index=True), "lora_cross", 7, 3
    )
    assert selected == 0
    assert history[0]["selected"] == 1


def test_standardizer_uses_only_supplied_training_values():
    train = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    mean, scale = matched.fit_standardizer(train)
    np.testing.assert_allclose(mean, [1.0, 3.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    validation = np.array([[1000.0, -1000.0]], dtype=np.float32)
    transformed = matched.standardize(validation, mean, scale)
    np.testing.assert_allclose(transformed, [[999.0, -1003.0]])


def test_retention_metrics_delegate_to_canonical_evaluator():
    retention = pd.DataFrame(
        {
            "chain1_sha256": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "chain2_sha256": ["a1", "a2", "a3", "a1", "a2", "a3"],
            "sequence_pair_sha256": ["x{}".format(i) for i in range(6)],
            "target_retention": [10.0, 80.0, 90.0, 20.0, 70.0, 100.0],
            "target_binder": [0, 1, 1, 0, 0, 1],
        }
    )
    probability = np.array([0.1, 0.8, 0.7, 0.2, 0.5, 0.9])
    expected = cached_eval.ranking_metrics(retention, probability)
    observed = matched.retention_metric_record(
        retention, probability, "LibA", "lora_cross", 7, 1
    )
    for key in (
        "global_auroc",
        "global_auprc",
        "global_spearman",
        "within_peptide_macro_spearman",
        "within_peptide_evaluable_spearman_groups",
    ):
        assert observed[key] == pytest.approx(expected[key])


def test_actual_cache_primary_and_three_fold_goldens_if_available():
    root = Path(__file__).resolve().parents[1]
    rows_path = root / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv"
    manifest_path = root / "private_data/derived/mint_weak_cache_v1/rows/manifest.json"
    if not rows_path.is_file() or not manifest_path.is_file():
        pytest.skip("private canonical cache is unavailable")
    rows, _ = matched.read_cache_rows(rows_path, manifest_path)
    for library in matched.LIBRARIES:
        primary, retention = matched.make_primary_pool(rows, library)
        plans, membership = matched.canonical_fold_plans(primary, library)
        contract = matched.PRIMARY_CONTRACT[library]
        assert len(primary) == contract["rows"]
        assert cached_eval.membership_sha256(primary) == contract["membership_sha256"]
        assert len(plans) == 3
        assert len(membership) == 3 * len(primary)
        assert len(retention) == contract["retention_rows"]
        for plan in plans:
            train = set(primary.iloc[plan["train_keys"]]["chain1_sha256"])
            validation = set(primary.iloc[plan["validation_keys"]]["chain1_sha256"])
            assert train.isdisjoint(validation)
            train = set(primary.iloc[plan["train_keys"]]["chain2_sha256"])
            validation = set(primary.iloc[plan["validation_keys"]]["chain2_sha256"])
            assert train.isdisjoint(validation)


def test_matched_arm_seed_does_not_depend_on_arm_name():
    seed = 20260811
    assert matched.derived_seed(seed, "cross_validation", 1) == matched.derived_seed(
        seed, "cross_validation", 1
    )
    assert matched.derived_seed(seed, "cross_validation", 1) != matched.derived_seed(
        seed, "final_refit", 1
    )


def test_parameter_delta_detects_changes_and_exact_no_change():
    initial = {"adapter": torch.tensor([1.0, 2.0])}
    unchanged = {"adapter": torch.tensor([1.0, 2.0])}
    changed = {"adapter": torch.tensor([1.0, 5.0])}
    assert matched.parameter_delta(initial, unchanged) == {"l2": 0.0, "max_abs": 0.0}
    delta = matched.parameter_delta(initial, changed)
    assert delta["l2"] == pytest.approx(3.0)
    assert delta["max_abs"] == pytest.approx(3.0)
