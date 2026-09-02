import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC import train_pnu_mint_readout as pnu


def _rows(roles=("P", "N", "U")):
    rows = []
    for index, role in enumerate(roles):
        rows.append(
            {
                "library": "LibB",
                "class_source": role,
                "pair_uid": "pair{}".format(index),
                "peptide_uid": "pepuid{}".format(index),
                "affibody_uid": "affuid{}".format(index),
                "pep": "AA",
                "aff": "AAAAA",
                "chain1_sha256": "pep{}".format(index),
                "chain2_sha256": "aff{}".format(index),
                "sequence_pair_sha256": "sequence{}".format(index),
                "target_retention": "",
                "target_binder": "",
            }
        )
    return pd.DataFrame(rows)


def test_pnu_risk_matches_requested_formula_and_endpoints():
    p_logits = torch.tensor([-0.7, 0.4], dtype=torch.float64)
    n_logits = torch.tensor([-0.3, 0.9, -1.1], dtype=torch.float64)
    u_logits = torch.tensor([0.2, -0.8, 1.3], dtype=torch.float64)
    pi = 0.1

    rp_pos = torch.nn.functional.softplus(-p_logits).mean()
    rp_neg = torch.nn.functional.softplus(p_logits).mean()
    rn_neg = torch.nn.functional.softplus(n_logits).mean()
    ru_neg = torch.nn.functional.softplus(u_logits).mean()

    eta0, components0 = pnu.pnu_risk_from_logits(
        p_logits, n_logits, None, pi=pi, eta=0.0
    )
    expected0 = pi * rp_pos + (1.0 - pi) * rn_neg
    assert torch.allclose(eta0, expected0)
    assert torch.allclose(components0["raw_negative"], (1.0 - pi) * rn_neg)

    eta1, components1 = pnu.pnu_risk_from_logits(
        p_logits, n_logits, u_logits, pi=pi, eta=1.0
    )
    expected1 = pi * rp_pos + torch.clamp(ru_neg - pi * rp_neg, min=0.0)
    assert torch.allclose(eta1, expected1)
    assert torch.allclose(components1["raw_negative"], ru_neg - pi * rp_neg)

    eta = 0.25
    mixed, _ = pnu.pnu_risk_from_logits(
        p_logits, n_logits, u_logits, pi=pi, eta=eta
    )
    expected_mixed_negative = (
        (1.0 - eta) * (1.0 - pi) * rn_neg
        + eta * (ru_neg - pi * rp_neg)
    )
    assert torch.allclose(
        mixed, pi * rp_pos + torch.clamp(expected_mixed_negative, min=0.0)
    )


def test_nonnegative_correction_clamps_only_combined_negative_term():
    # Very positive P logits make Rp- large; very negative U logits make Ru-
    # tiny, hence the raw nnPU negative estimate is below zero.
    p_logits = torch.full((4,), 20.0, dtype=torch.float64)
    n_logits = torch.zeros(4, dtype=torch.float64)
    u_logits = torch.full((4,), -20.0, dtype=torch.float64)
    loss, components = pnu.pnu_risk_from_logits(
        p_logits, n_logits, u_logits, pi=0.2, eta=1.0
    )
    expected_positive = 0.2 * torch.nn.functional.softplus(-p_logits).mean()
    assert components["raw_negative"].item() < 0.0
    assert components["nonnegative_negative"].item() == 0.0
    assert torch.allclose(loss, expected_positive)


def test_eta_zero_ignores_u_and_eta_one_ignores_n():
    p_logits = torch.tensor([0.1, -0.2])
    n_a = torch.tensor([-4.0, 3.0])
    n_b = torch.tensor([10.0, 10.0])
    u_a = torch.tensor([-2.0, 2.0])
    u_b = torch.tensor([8.0, 8.0])
    eta0_a = pnu.pnu_risk_from_logits(p_logits, n_a, u_a, 0.1, 0.0)[0]
    eta0_b = pnu.pnu_risk_from_logits(p_logits, n_a, u_b, 0.1, 0.0)[0]
    assert torch.allclose(eta0_a, eta0_b)
    eta1_a = pnu.pnu_risk_from_logits(p_logits, n_a, u_a, 0.1, 1.0)[0]
    eta1_b = pnu.pnu_risk_from_logits(p_logits, n_b, u_a, 0.1, 1.0)[0]
    assert torch.allclose(eta1_a, eta1_b)


def test_training_metadata_rejects_outcome_leakage():
    safe = pnu.canonicalize_pnu_rows(_rows())
    assert list(safe["pnu_role"]) == ["P", "N", "U"]
    leaked = _rows()
    leaked.loc[0, "target_retention"] = "88.0"
    with pytest.raises(ValueError, match="outcome values"):
        pnu.canonicalize_pnu_rows(leaked)


def test_double_identity_cold_roles_have_no_partner_overlap():
    frame = pd.DataFrame(
        {
            "chain1_sha256": ["p{}".format(i // 4) for i in range(40)],
            "chain2_sha256": ["a{}".format(i % 7) for i in range(40)],
        }
    )
    for fold in range(3):
        roles = pnu.identity_cold_roles(frame, fold, 3, seed=17)
        train = frame.iloc[roles["train"]]
        validation = frame.iloc[roles["validation"]]
        assert set(train["chain1_sha256"]).isdisjoint(validation["chain1_sha256"])
        assert set(train["chain2_sha256"]).isdisjoint(validation["chain2_sha256"])
        all_indices = np.concatenate(
            (roles["train"], roles["guard"], roles["validation"])
        )
        assert sorted(all_indices.tolist()) == list(range(len(frame)))


def test_u_sampling_contract_is_explicit_and_exact():
    base = _rows(("P",))
    positive = pnu.RolePool(
        pnu.canonicalize_pnu_rows(base),
        np.zeros((1, pnu.FEATURE_DIMENSION), dtype=np.float16),
    )
    n_rows = _rows(("N",))
    n_rows["pair_uid"] = ["negative0"]
    n_rows["sequence_pair_sha256"] = ["negative-sequence0"]
    negative = pnu.RolePool(
        pnu.canonicalize_pnu_rows(n_rows),
        np.zeros((1, pnu.FEATURE_DIMENSION), dtype=np.float16),
    )
    u_rows = _rows(("U", "U", "U", "U", "U"))
    # Make identifiers unique because the fixture helper numbers from zero.
    u_rows["pair_uid"] = ["u{}".format(i) for i in range(5)]
    u_rows["sequence_pair_sha256"] = ["useq{}".format(i) for i in range(5)]
    unlabeled = pnu.RolePool(
        pnu.canonicalize_pnu_rows(u_rows),
        np.zeros((5, pnu.FEATURE_DIMENSION), dtype=np.float16),
    )
    evaluation = pd.DataFrame(
        {"chain1_sha256": ["heldp"], "chain2_sha256": ["helda"]}
    )
    pnu._audit_unlabeled_contract(
        positive,
        negative,
        unlabeled,
        evaluation,
        "sampled_u_per_positive",
        5.0,
    )
    with pytest.raises(ValueError, match="sampled-U count"):
        pnu._audit_unlabeled_contract(
            positive,
            negative,
            unlabeled,
            evaluation,
            "sampled_u_per_positive",
            4.0,
        )


def test_hyperparameter_selection_has_no_retention_argument_or_dependency():
    records = pd.DataFrame(
        [
            {
                "record_type": "aggregate",
                "pi": 0.1,
                "eta": 0.0,
                "C": 0.1,
                "epoch": 2,
                "log_loss": 0.6,
                "ap": 0.7,
                "auroc": 0.8,
            },
            {
                "record_type": "aggregate",
                "pi": 0.1,
                "eta": 0.5,
                "C": 0.01,
                "epoch": 4,
                "log_loss": 0.5,
                "ap": 0.6,
                "auroc": 0.7,
            },
        ]
    )
    signature = inspect.signature(pnu.select_weak_hyperparameters)
    assert "retention" not in " ".join(signature.parameters).lower()
    selected = pnu.select_weak_hyperparameters(records, selection_metric="log_loss")
    assert selected["eta"] == 0.5
    # Arbitrarily changing an external retention table cannot affect a
    # selector whose complete input is the weak-validation record table.
    retention = pd.DataFrame({"target_retention": [0.0, 100.0]})
    retention["target_retention"] = [100.0, 0.0]
    assert (
        pnu.select_weak_hyperparameters(records, selection_metric="log_loss")
        == selected
    )


def test_weak_auroc_is_the_primary_ranking_selector():
    records = pd.DataFrame(
        [
            {
                "record_type": "aggregate",
                "pi": 0.05,
                "eta": 0.0,
                "C": 0.1,
                "epoch": 3,
                "log_loss": 0.20,
                "ap": 0.70,
                "auroc": 0.75,
            },
            {
                "record_type": "aggregate",
                "pi": 0.05,
                "eta": 0.5,
                "C": 0.1,
                "epoch": 40,
                "log_loss": 0.80,
                "ap": 0.65,
                "auroc": 0.90,
            },
        ]
    )
    selected = pnu.select_weak_hyperparameters(records, selection_metric="auroc")
    assert selected["eta"] == 0.5
    assert selected["epoch"] == 40
    assert selected["weak_selection_metric"] == "auroc"


def test_retrospective_f1_threshold_uses_score_greater_equal_rule():
    result = pnu.best_retrospective_f1(
        np.asarray([1, 0, 1, 0]), np.asarray([0.9, 0.8, 0.8, 0.1])
    )
    threshold = result["retrospective_f1_threshold"]
    selected = np.asarray([0.9, 0.8, 0.8, 0.1]) >= threshold
    assert int(selected.sum()) == result["retrospective_f1_selected"]
    assert result["retrospective_f1_tp"] == 2
