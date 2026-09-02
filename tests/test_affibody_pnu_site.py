import math

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC.train_pnu_site import (
    AdditiveSiteLogit,
    _validate_unlabeled_uids,
    add_identity_blocks_fast,
    empirical_risk_components,
    grid_configs,
    identity_cold_u_train_indices,
    make_config,
    membership_sha256,
    pnu_objective,
    retention_metrics,
    select_configs_with_weak_validation,
    site_index_matrix,
    train_epoch,
)
from downstream.AffibodyMHC.code_only_baseline import opaque_id
from downstream.AffibodyMHC.evaluate_selection_weak_baseline import add_identity_blocks


def test_logistic_risk_components_and_balanced_pn_match_hand_calculation():
    positive = torch.tensor([0.0, math.log(3.0)])
    negative = torch.tensor([0.0, -math.log(3.0)])
    components = empirical_risk_components(positive, negative)
    objective, detail = pnu_objective(components, risk="pn")

    rp_positive = np.mean(np.logaddexp(0.0, -positive.numpy()))
    rn_negative = np.mean(np.logaddexp(0.0, negative.numpy()))
    expected = 0.5 * rp_positive + 0.5 * rn_negative
    assert float(objective) == pytest.approx(expected)
    assert float(detail["nonnegative_correction_active"]) == 0.0


def test_nnpnu_endpoints_are_prior_weighted_pn_and_nnpu():
    components = empirical_risk_components(
        torch.tensor([1.2, -0.4]),
        torch.tensor([-0.7, 0.3]),
        torch.tensor([-1.1, 0.5, 0.9]),
    )
    pi = 0.1

    eta_zero, zero_detail = pnu_objective(
        components, risk="nnpnu", class_prior=pi, eta=0.0
    )
    expected_zero = (
        pi * components["rp_positive"]
        + (1.0 - pi) * components["rn_negative"]
    )
    assert float(eta_zero) == pytest.approx(float(expected_zero))
    assert float(zero_detail["corrected_component_raw"]) == pytest.approx(
        float((1.0 - pi) * components["rn_negative"])
    )

    eta_one, _ = pnu_objective(
        components, risk="nnpnu", class_prior=pi, eta=1.0
    )
    nnpu, _ = pnu_objective(
        components, risk="nnpu", class_prior=pi
    )
    assert float(eta_one) == pytest.approx(float(nnpu))


def test_nonnegative_correction_clamps_a_negative_unbiased_component():
    # Very positive P logits make Rp- large; very negative U logits make Ru-
    # small, so Ru- - pi*Rp- is negative and must be clamped to zero.
    components = empirical_risk_components(
        torch.tensor([20.0, 20.0]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([-20.0, -20.0]),
    )
    objective, detail = pnu_objective(
        components, risk="nnpu", class_prior=0.15
    )

    assert float(detail["corrected_component_raw"]) < 0.0
    assert float(detail["corrected_component"]) == 0.0
    assert float(detail["nonnegative_correction_active"]) == 1.0
    assert float(objective) == pytest.approx(
        0.15 * float(components["rp_positive"])
    )


def test_site_indices_use_peptide_then_affibody_and_additive_logits_are_exact():
    frame = pd.DataFrame({"pep": ["AD", "CE"], "aff": ["FGHI", "KLMN"]})
    encoded = site_index_matrix(frame, "LibA")
    assert encoded.shape == (2, 6)

    model = AdditiveSiteLogit(6)
    with torch.no_grad():
        model.weight.copy_(torch.arange(120, dtype=torch.float32).reshape(6, 20))
        model.bias.fill_(2.5)
    tensor = torch.as_tensor(encoded, dtype=torch.long)
    observed = model(tensor).detach().numpy()
    expected = np.asarray(
        [
            2.5 + sum(float(model.weight[position, residue]) for position, residue in enumerate(row))
            for row in encoded
        ]
    )
    assert np.allclose(observed, expected)


def test_fast_identity_blocks_match_rowwise_reference_and_u_is_double_cold():
    rows = []
    for peptide_index in range(12):
        for affibody_index in range(14):
            rows.append(
                {
                    "peptide_uid": "pep-{}".format(peptide_index),
                    "affibody_uid": "aff-{}".format(affibody_index),
                    "pair_uid": "pair-{}-{}".format(peptide_index, affibody_index),
                }
            )
    frame = pd.DataFrame(rows)
    blocked = add_identity_blocks_fast(frame, "LibA", 3)
    reference = add_identity_blocks(frame, "LibA", 3)
    assert blocked["peptide_block"].tolist() == reference["peptide_block"].tolist()
    assert blocked["affibody_block"].tolist() == reference["affibody_block"].tolist()
    for fold in range(3):
        indices = identity_cold_u_train_indices(blocked, fold)
        selected = blocked.iloc[indices]
        assert not selected["peptide_block"].eq(fold).any()
        assert not selected["affibody_block"].eq(fold).any()


def test_unlabeled_uid_audit_rejects_forged_partner_identity():
    rows = []
    for peptide, affibody in (("AD", "FGHI"), ("CE", "KLMN")):
        rows.append(
            {
                "library": "LibA",
                "pep": peptide,
                "aff": affibody,
                "pair_uid": opaque_id("LibA", peptide, affibody),
                "peptide_uid": opaque_id("LibA", "pep", peptide),
                "affibody_uid": opaque_id("LibA", "aff", affibody),
            }
        )
    frame = pd.DataFrame(rows)
    _validate_unlabeled_uids(frame)
    forged = frame.copy()
    forged.loc[0, "peptide_uid"] = forged.loc[1, "peptide_uid"]
    with pytest.raises(ValueError, match="peptide UID"):
        _validate_unlabeled_uids(forged)


def test_membership_digest_is_order_invariant_but_membership_sensitive():
    frame = pd.DataFrame({"pair_uid": ["c", "a", "b"]})
    shuffled = frame.sample(frac=1.0, random_state=3)
    changed = pd.DataFrame({"pair_uid": ["c", "a", "x"]})
    assert membership_sha256(frame) == membership_sha256(shuffled)
    assert membership_sha256(frame) != membership_sha256(changed)


def test_grid_contains_prespecified_pi_eta_arms_without_duplicate_ids():
    configs = grid_configs(
        risks=("pn", "nnpu", "nnpnu"),
        pi_grid=(0.02, 0.05, 0.10, 0.15),
        eta_grid=(0.0, 0.25, 0.50, 0.75, 1.0),
        c_grid=(1.0,),
    )
    assert len(configs) == 1 + 4 + 4 * 5
    assert len({item["config_id"] for item in configs}) == len(configs)
    assert {item["class_prior"] for item in configs if item["risk"] == "nnpu"} == {
        0.02,
        0.05,
        0.10,
        0.15,
    }
    assert {item["eta"] for item in configs if item["risk"] == "nnpnu"} == {
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    }


def test_config_selection_never_compares_different_class_priors():
    rows = []
    for pi, eta, loss in (
        (0.02, 0.0, 0.4),
        (0.02, 0.5, 0.3),
        (0.05, 0.0, 0.2),
        (0.05, 0.5, 0.5),
    ):
        config = make_config("nnpnu", 1.0, class_prior=pi, eta=eta)
        config.update(
            {
                "selected_epoch": 2,
                "validation_log_loss": loss,
                "validation_auroc": 0.6,
                "validation_average_precision": 0.5,
            }
        )
        rows.append(config)
    selected = select_configs_with_weak_validation(rows)
    chosen = selected.loc[selected["selected_within_fixed_pi"].eq(1)]
    assert len(chosen) == 2
    assert set(chosen["class_prior"]) == {0.02, 0.05}
    assert chosen.set_index("class_prior")["eta"].to_dict() == {0.02: 0.5, 0.05: 0.0}


def test_auroc_selector_maximizes_ranking_even_when_log_loss_is_worse():
    rows = []
    for eta, loss, auroc, ap in (
        (0.0, 0.30, 0.70, 0.75),
        (0.5, 1.20, 0.91, 0.88),
    ):
        config = make_config("nnpnu", 1.0, class_prior=0.10, eta=eta)
        config.update(
            {
                "selected_epoch": 3,
                "validation_log_loss": loss,
                "validation_auroc": auroc,
                "validation_average_precision": ap,
                "selection_metric": "auroc",
            }
        )
        rows.append(config)

    by_loss = select_configs_with_weak_validation(rows, selection_metric="log_loss")
    by_auroc = select_configs_with_weak_validation(rows, selection_metric="auroc")
    assert by_loss.loc[by_loss["selected_within_fixed_pi"].eq(1), "eta"].item() == 0.0
    assert by_auroc.loc[by_auroc["selected_within_fixed_pi"].eq(1), "eta"].item() == 0.5


def test_auroc_selector_ties_by_ap_then_logloss_then_smaller_eta_c_epoch():
    rows = []
    for eta, c_value, epoch, loss, ap in (
        (0.75, 10.0, 5, 0.8, 0.80),
        (0.50, 1.0, 4, 0.9, 0.85),
        (0.25, 0.1, 3, 0.7, 0.85),
    ):
        config = make_config("nnpnu", c_value, class_prior=0.10, eta=eta)
        config.update(
            {
                "selected_epoch": epoch,
                "validation_log_loss": loss,
                "validation_auroc": 0.90,
                "validation_average_precision": ap,
                "selection_metric": "auroc",
            }
        )
        rows.append(config)
    selected = select_configs_with_weak_validation(rows, selection_metric="auroc")
    chosen = selected.loc[selected["selected_within_fixed_pi"].eq(1)].iloc[0]
    assert chosen["eta"] == 0.25
    assert chosen["C"] == 0.1


def test_train_epoch_visits_every_u_row_exactly_once():
    # Tiny integer feature matrices are sufficient to audit coverage.  The
    # labeled arms are cycled, while U must be exhausted exactly once.
    p = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int64)
    n = np.asarray([[2, 2], [2, 3], [3, 2], [3, 3]], dtype=np.int64)
    u = np.asarray([[index % 4, (index + 1) % 4] for index in range(23)], dtype=np.int64)
    model = AdditiveSiteLogit(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    audit = train_epoch(
        model=model,
        optimizer=optimizer,
        p_encoded=p,
        n_encoded=n,
        u_encoded=u,
        risk="nnpu",
        class_prior=0.1,
        eta=1.0,
        c_value=1.0,
        labeled_train_size=len(p) + len(n),
        batch_size=7,
        device=torch.device("cpu"),
        rng=np.random.RandomState(7),
    )

    assert audit["steps"] == 4
    assert audit["u_rows_seen"] == 23
    assert audit["u_rows_available"] == 23


def test_eta_zero_is_prior_weighted_pn_and_does_not_traverse_u():
    p = np.asarray([[0, 0], [0, 1]], dtype=np.int64)
    n = np.asarray([[2, 2], [2, 3]], dtype=np.int64)
    u = np.asarray([[1, 1]] * 101, dtype=np.int64)
    model = AdditiveSiteLogit(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    audit = train_epoch(
        model=model,
        optimizer=optimizer,
        p_encoded=p,
        n_encoded=n,
        u_encoded=u,
        risk="nnpnu",
        class_prior=0.1,
        eta=0.0,
        c_value=1.0,
        labeled_train_size=4,
        batch_size=7,
        device=torch.device("cpu"),
        rng=np.random.RandomState(8),
    )
    assert audit["u_rows_seen"] == 0
    assert audit["u_rows_available"] == 0


def test_retention_metrics_macro_average_spearman_within_peptide():
    frame = pd.DataFrame(
        {
            "peptide_uid": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "target_retention": [10.0, 50.0, 90.0, 20.0, 60.0, 95.0],
            "target_binder": [0, 0, 1, 0, 0, 1],
        }
    )
    probability = np.asarray([0.1, 0.5, 0.9, 0.9, 0.5, 0.1])
    metrics = retention_metrics(frame, probability)

    assert metrics["within_peptide_spearman"] == pytest.approx(0.0)
    assert metrics["within_peptide_groups"] == 2
    assert metrics["average_precision"] > 0.0
