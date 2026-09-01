import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from downstream.AffibodyMHC.evaluate_cached_weak_mint import (
    apply_static_cleaning,
    apply_train_only_cleaning,
    balance_training_indices,
    ranking_metrics,
    regime_split_indices,
    retention_compatible_mask,
)


def _weak_frame():
    """Small two-library frame with deliberately shared global identities."""
    rows = []
    for library_index, library in enumerate(("LibA", "LibB")):
        for peptide_index in range(6):
            for affibody_index in range(6):
                # The identity strings deliberately omit the library.  In
                # particular, pep-0 in LibA and pep-0 in LibB are the same
                # full-chain sequence identity and must never straddle a cold
                # train/validation boundary.
                peptide = "pep-{}".format(peptide_index)
                affibody = "aff-{}".format(affibody_index)
                rows.append(
                    {
                        "library": library,
                        "pair_uid": "pair-{}-{}-{}".format(
                            library_index, peptide_index, affibody_index
                        ),
                        "peptide_identity": peptide,
                        "affibody_identity": affibody,
                        "weak_label": (peptide_index + affibody_index) % 2,
                        "r009_count": 4,
                        "r010_count": 4,
                    }
                )
    return pd.DataFrame(rows)


def _assert_partition(parts, n_rows):
    assert set(parts) == {"train", "guard", "validation"}
    arrays = [np.asarray(parts[name], dtype=int) for name in parts]
    combined = np.concatenate(arrays) if arrays else np.asarray([], dtype=int)
    assert sorted(combined.tolist()) == list(range(n_rows))
    assert len(np.unique(combined)) == n_rows


@pytest.mark.parametrize(
    "regime", ["pair_only", "peptide_cold", "affibody_cold", "double_cold"]
)
def test_regime_splits_partition_rows_and_enforce_identity_contracts(regime):
    frame = _weak_frame()

    for fold in range(3):
        parts = regime_split_indices(frame, regime, fold, n_folds=3, seed=17)
        _assert_partition(parts, len(frame))
        train = frame.iloc[np.asarray(parts["train"], dtype=int)]
        validation = frame.iloc[np.asarray(parts["validation"], dtype=int)]

        if regime == "pair_only":
            assert not len(parts["guard"])
            assert set(train["pair_uid"]).isdisjoint(validation["pair_uid"])
        elif regime == "peptide_cold":
            assert not len(parts["guard"])
            assert set(train["peptide_identity"]).isdisjoint(
                validation["peptide_identity"]
            )
        elif regime == "affibody_cold":
            assert not len(parts["guard"])
            assert set(train["affibody_identity"]).isdisjoint(
                validation["affibody_identity"]
            )
        else:
            assert set(train["peptide_identity"]).isdisjoint(
                validation["peptide_identity"]
            )
            assert set(train["affibody_identity"]).isdisjoint(
                validation["affibody_identity"]
            )


def test_global_identity_not_library_scoped_in_cold_splits():
    frame = _weak_frame()

    for regime, identity_column in (
        ("peptide_cold", "peptide_identity"),
        ("affibody_cold", "affibody_identity"),
    ):
        for fold in range(3):
            parts = regime_split_indices(frame, regime, fold, 3, seed=19)
            roles = {}
            for role in ("train", "validation"):
                identities = frame.iloc[parts[role]][identity_column]
                for identity in identities:
                    roles.setdefault(identity, set()).add(role)
            assert all(len(identity_roles) == 1 for identity_roles in roles.values())


def test_static_count_support_cleaning_changes_only_unsupported_positives():
    frame = pd.DataFrame(
        {
            "pair_uid": ["supported", "low-r9", "low-r10", "negative"],
            "weak_label": [1, 1, 1, 0],
            "r009_count": [3, 2, 8, 0],
            "r010_count": [3, 8, 2, 0],
            "peptide_identity": ["p0", "p1", "p2", "p3"],
            "affibody_identity": ["a0", "a1", "a2", "a3"],
        }
    )

    unchanged = apply_static_cleaning(frame, "c0")
    fold_local_placeholder = apply_static_cleaning(frame, "promiscuity_ge10")
    cleaned = apply_static_cleaning(frame, "count_both_ge3")

    assert unchanged["pair_uid"].tolist() == frame["pair_uid"].tolist()
    assert fold_local_placeholder["pair_uid"].tolist() == frame["pair_uid"].tolist()
    assert set(cleaned["pair_uid"]) == {"supported", "negative"}


def test_promiscuity_filter_is_computed_from_training_rows_only():
    # In the training fold, a-borderline has only one positive peptide and is
    # retained at threshold two.  It would be incorrectly removed if the
    # validation peptide were inspected while constructing the filter.
    train = pd.DataFrame(
        {
            "pair_uid": ["t0", "t1", "t2", "t3", "t4"],
            "peptide_identity": ["p0", "p1", "p2", "p3", "p4"],
            "affibody_identity": [
                "a-promiscuous",
                "a-promiscuous",
                "a-promiscuous",
                "a-borderline",
                "a-promiscuous",
            ],
            "weak_label": [1, 1, 0, 1, 0],
        }
    )
    validation = pd.DataFrame(
        {
            "pair_uid": ["v0"],
            "peptide_identity": ["p5"],
            "affibody_identity": ["a-borderline"],
            "weak_label": [1],
        }
    )

    cleaned_train = apply_train_only_cleaning(
        train, "promiscuity_ge10", threshold=2
    )
    incorrectly_transductive = apply_train_only_cleaning(
        pd.concat([train, validation], ignore_index=True),
        "promiscuity_ge10",
        threshold=2,
    )

    assert set(cleaned_train["pair_uid"]) == {"t3"}
    assert "t3" not in set(incorrectly_transductive["pair_uid"])


def test_downsampling_is_deterministic_balanced_and_preserves_minority():
    frame = pd.DataFrame(
        {
            "weak_label": [1] * 8 + [0] * 3,
            "pair_uid": ["pair-{}".format(index) for index in range(11)],
        },
        index=np.arange(100, 111),
    )

    natural = balance_training_indices(frame, "all_unweighted", seed=23)
    weighted = balance_training_indices(frame, "all_class_weighted", seed=23)
    first = balance_training_indices(frame, "balanced_downsample", seed=23)
    second = balance_training_indices(frame, "balanced_downsample", seed=23)

    assert set(np.asarray(natural, dtype=int)) == set(range(len(frame)))
    assert set(np.asarray(weighted, dtype=int)) == set(range(len(frame)))
    assert np.array_equal(first, second)
    selected = frame.iloc[np.asarray(first, dtype=int)]
    assert selected["weak_label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert set(np.flatnonzero(frame["weak_label"].eq(0))) <= set(first)


def test_retention_compatibility_encodes_four_generalization_questions():
    actual_train = pd.DataFrame(
        {
            "pair_uid": ["seen-pair-a", "seen-pair-b"],
            "peptide_identity": ["pep-a", "pep-b"],
            "affibody_identity": ["aff-a", "aff-b"],
        }
    )
    retention = pd.DataFrame(
        {
            "pair_uid": [
                "crossed-seen",
                "new-peptide",
                "new-affibody",
                "both-new",
                "seen-pair-a",
                "both-new-2",
            ],
            "peptide_identity": [
                "pep-a",
                "pep-new",
                "pep-a",
                "pep-new",
                "pep-a",
                "pep-new-2",
            ],
            "affibody_identity": [
                "aff-b",
                "aff-a",
                "aff-new",
                "aff-new",
                "aff-a",
                "aff-new-2",
            ],
        }
    )

    expected = {
        "pair_only": [True, False, False, False, False, False],
        "peptide_cold": [False, True, False, False, False, False],
        "affibody_cold": [False, False, True, False, False, False],
        "double_cold": [False, False, False, True, False, True],
    }
    for regime, values in expected.items():
        observed = retention_compatible_mask(retention, actual_train, regime)
        assert np.asarray(observed, dtype=bool).tolist() == values


def test_ranking_metrics_include_global_grouped_lift_and_precision_at_k():
    retention = pd.DataFrame(
        {
            "target_retention": [90.0, 40.0, 10.0, 20.0, 80.0, 30.0],
            "target_binder": [1, 0, 0, 0, 1, 0],
            "peptide_identity": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "affibody_identity": ["a1", "a2", "a3", "a1", "a2", "a3"],
        }
    )
    scores = np.asarray([0.90, 0.80, 0.10, 0.90, 0.80, 0.10])

    observed = ranking_metrics(retention, scores)
    labels = retention["target_binder"].to_numpy(dtype=int)

    assert observed["n"] == 6
    assert observed["global_prevalence"] == pytest.approx(2.0 / 6.0)
    assert observed["global_auroc"] == pytest.approx(roc_auc_score(labels, scores))
    assert observed["global_auprc"] == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert observed["global_spearman"] == pytest.approx(
        spearmanr(retention["target_retention"], scores).correlation
    )

    # Peptide APs are 1.0 and 0.5; each peptide has prevalence 1/3.
    assert observed["within_peptide_evaluable_groups"] == 2
    assert observed["within_peptide_macro_auprc"] == pytest.approx(0.75)
    assert observed["within_peptide_macro_auprc_lift"] == pytest.approx(
        0.75 - (1.0 / 3.0)
    )
    assert observed["peptide_precision_at_1"] == pytest.approx(0.5)
    assert observed["peptide_precision_at_3"] == pytest.approx(1.0 / 3.0)

    # a1 and a2 contain both labels; a3 is all-negative and must be excluded
    # from within-group AP rather than silently assigned a perfect score.
    assert observed["within_affibody_evaluable_groups"] == 2
    assert observed["within_affibody_macro_auprc"] == pytest.approx(0.5)
    assert observed["within_affibody_macro_auprc_lift"] == pytest.approx(0.0)
