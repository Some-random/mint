import numpy as np
import pandas as pd

from downstream.AffibodyMHC import build_libb_weak_ensemble as ensemble


def test_mean_logit_is_identity_for_one_member_and_geometric_odds_for_two():
    left = np.asarray([0.2, 0.8])
    right = np.asarray([0.5, 0.5])
    assert np.allclose(ensemble.mean_logit_score([left]), left)
    observed = ensemble.mean_logit_score([left, right])
    expected = ensemble.sigmoid(
        (ensemble.clipped_logit(left) + ensemble.clipped_logit(right)) / 2.0
    )
    assert np.allclose(observed, expected)


def test_macro_within_peptide_ap_gives_each_evaluable_peptide_one_vote():
    peptide = ["a", "a", "b", "b", "c", "c"]
    label = [1, 0, 1, 0, 1, 1]
    score = [0.9, 0.1, 0.1, 0.9, 0.2, 0.3]
    result = ensemble.macro_within_peptide_metrics(peptide, label, score)
    assert result["within_peptide_evaluable"] == 2
    # Peptide a has AP 1; peptide b has AP 0.5; peptide c is not evaluable.
    assert np.isclose(result["within_peptide_ap"], 0.75)
    assert np.isclose(result["within_peptide_auroc"], 0.5)


def test_nonnegative_stacker_cannot_assign_an_inverse_member_negative_weight():
    x = np.asarray([[-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0], [2.0, -2.0]])
    y = np.asarray([0, 0, 1, 1])
    model = ensemble.fit_nonnegative_stacker(x, y, l2=0.01)
    coefficient = np.asarray(model["coefficient"])
    assert np.all(coefficient >= 0.0)
    assert coefficient[0] > coefficient[1]


def test_cross_fitted_stacker_returns_one_score_per_row_and_never_uses_held_fold(monkeypatch):
    rows = []
    for fold in range(3):
        for peptide_index in range(2):
            peptide = f"f{fold}p{peptide_index}"
            for label in (0, 1):
                rows.append(
                    {
                        "row_id": f"{peptide}-{label}",
                        "fold": fold,
                        "peptide_id": peptide,
                        "affibody_id": f"a-{label}",
                        "weak_label": label,
                        "additive_7site": 0.2 + 0.6 * label,
                        "mint_layer5": 0.3 + 0.4 * label,
                        "stab_designed_ordered": 0.4 + 0.2 * label,
                    }
                )
    frame = pd.DataFrame(rows)
    original = ensemble._fit_stacker_partition
    calls = []

    def wrapped(train, prediction, l2):
        calls.append((set(train["fold"]), set(prediction["fold"])))
        return original(train, prediction, l2)

    monkeypatch.setattr(ensemble, "_fit_stacker_partition", wrapped)
    score, audits = ensemble.cross_fitted_stacker(frame)
    assert len(score) == len(frame)
    assert len(audits) == 3
    # Every fit used to produce a held-fold prediction excludes that fold.
    for audit in audits:
        held = audit["held_fold"]
        matching = [call for call in calls if call[1] == {held} and len(call[0]) == 2]
        assert matching
        assert all(held not in train_folds for train_folds, _ in matching)
