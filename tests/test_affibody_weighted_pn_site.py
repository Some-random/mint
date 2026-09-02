import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from downstream.AffibodyMHC.compare_weighted_pn_site import (
    class_mass_weights,
    fit_converged_logistic,
    select_by_weak_auroc,
)


def test_class_mass_weights_have_requested_mass_and_fixed_total():
    labels = np.asarray([1, 1, 0, 0, 0, 0, 0], dtype=int)
    weights = class_mass_weights(labels, 0.10)
    np.testing.assert_allclose(weights.sum(), len(labels), atol=1e-12)
    np.testing.assert_allclose(weights[labels == 1].sum() / weights.sum(), 0.10, atol=1e-12)
    np.testing.assert_allclose(weights[labels == 0].sum() / weights.sum(), 0.90, atol=1e-12)


def test_half_mass_matches_sklearn_balanced_class_weight():
    raw = np.asarray([["A"], ["A"], ["C"], ["D"], ["D"], ["E"], ["F"]])
    labels = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=int)
    encoder = OneHotEncoder(categories=[list("ACDEFGHIKLMNPQRSTVWY")], sparse=True)
    encoded = encoder.fit_transform(raw)
    explicit, _, _ = fit_converged_logistic(encoded, labels, 0.7, 0.5)
    established = LogisticRegression(
        C=0.7,
        penalty="l2",
        solver="liblinear",
        fit_intercept=True,
        class_weight="balanced",
        random_state=0,
        max_iter=20000,
        tol=1e-12,
    ).fit(encoded, labels)
    np.testing.assert_allclose(explicit.coef_, established.coef_, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(explicit.intercept_, established.intercept_, atol=1e-11, rtol=1e-11)


def test_selection_uses_auroc_and_smaller_c_only_for_ties():
    frame = pd.DataFrame(
        [
            {"library": "LibA", "arm": "x", "positive_mass": 0.1, "C": 0.1, "record_type": "aggregate", "validation_auroc": 0.8, "validation_average_precision": 0.99, "validation_log_loss": 0.1},
            {"library": "LibA", "arm": "x", "positive_mass": 0.1, "C": 1.0, "record_type": "aggregate", "validation_auroc": 0.9, "validation_average_precision": 0.01, "validation_log_loss": 9.0},
            {"library": "LibA", "arm": "y", "positive_mass": 0.2, "C": 0.1, "record_type": "aggregate", "validation_auroc": 0.9, "validation_average_precision": 0.1, "validation_log_loss": 1.0},
            {"library": "LibA", "arm": "y", "positive_mass": 0.2, "C": 1.0, "record_type": "aggregate", "validation_auroc": 0.9, "validation_average_precision": 0.9, "validation_log_loss": 0.1},
        ]
    )
    selected = select_by_weak_auroc(frame, ("library", "arm", "positive_mass"))
    chosen = selected.loc[selected["selected_C_within_arm"].eq(1)]
    assert chosen.loc[chosen["arm"].eq("x"), "C"].iloc[0] == 1.0
    assert chosen.loc[chosen["arm"].eq("y"), "C"].iloc[0] == 0.1
