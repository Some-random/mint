import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.run_weak_factorial_benchmark import (
    AFFIBODY_POSITIONS,
    FAILURE_COLUMNS,
    _two_way_center,
    add_exposure_flags,
    apply_balance,
    apply_cleaning,
    apply_transfer_regime,
    attach_sequence_identities_and_sites,
    evaluation_metrics,
)


def templates():
    chain1 = list("A" * 270)
    chain1[264] = "X"
    chain1[265] = "X"

    chain2_a = list("A" * 58)
    for position in (13, 17, 27, 31):
        chain2_a[position - 1] = "X"

    # A different immutable scaffold proves that a library-local short code is
    # not being confused with a global full-chain identity.
    chain2_b = list("C" * 58)
    for position in (6, 10, 13, 14, 17):
        chain2_b[position - 1] = "X"
    return {
        "chain1": "".join(chain1),
        "chain2": {"LibA": "".join(chain2_a), "LibB": "".join(chain2_b)},
    }


def attach(frame):
    return attach_sequence_identities_and_sites(
        frame,
        templates(),
        peptide_column="pep",
        affibody_column="aff",
    )


def test_full_sequence_identity_is_global_and_sites_are_physically_aligned():
    frame = pd.DataFrame(
        {
            "library": ["LibA", "LibB"],
            "pep": ["AF", "AF"],
            "aff": ["DEFG", "HIKLM"],
        }
    )
    result = attach(frame)
    assert result["global_peptide_sha256"].nunique() == 1
    assert result["global_affibody_sha256"].nunique() == 2
    assert result["global_pair_sha256"].nunique() == 2
    assert tuple(result.loc[0, ["aff_p13", "aff_p17", "aff_p27", "aff_p31"]]) == tuple("DEFG")
    assert tuple(result.loc[1, ["aff_p6", "aff_p10", "aff_p13", "aff_p14", "aff_p17"]]) == tuple("HIKLM")
    assert all("aff_p{}".format(position) in result for position in AFFIBODY_POSITIONS)


def test_transfer_regimes_remove_the_intended_full_chain_identities():
    retention = attach(
        pd.DataFrame({"library": ["LibA"], "pep": ["AF"], "aff": ["DEFG"]})
    )
    weak = attach(
        pd.DataFrame(
            {
                "library": ["LibA"] * 4,
                "pep": ["AF", "AF", "GH", "GH"],
                "aff": ["DEFG", "HIKL", "DEFG", "HIKL"],
            }
        )
    )
    assert len(apply_transfer_regime(weak, retention, "pair_seen")) == 3
    peptide_cold = apply_transfer_regime(weak, retention, "peptide_cold")
    assert set(peptide_cold["pep"]) == {"GH"}
    affibody_cold = apply_transfer_regime(weak, retention, "affibody_cold")
    assert set(affibody_cold["aff"]) == {"HIKL"}
    double_cold = apply_transfer_regime(weak, retention, "double_cold")
    assert list(double_cold[["pep", "aff"]].itertuples(index=False, name=None)) == [("GH", "HIKL")]


def test_pooled_peptide_cold_blocks_same_full_chain_across_libraries():
    retention = attach(
        pd.DataFrame({"library": ["LibA"], "pep": ["AF"], "aff": ["DEFG"]})
    )
    weak = attach(
        pd.DataFrame(
            {
                "library": ["LibB", "LibB"],
                "pep": ["AF", "GH"],
                "aff": ["HIKLM", "HIKLM"],
            }
        )
    )
    result = apply_transfer_regime(weak, retention, "peptide_cold")
    assert list(result["pep"]) == ["GH"]


def test_cleaning_rules_use_only_current_training_frame():
    rows = []
    for index in range(10):
        # Twenty canonical dipeptides are enough for this synthetic breadth
        # check; their biological plausibility is irrelevant to the unit test.
        pep = "A" + "ACDEFGHIKL"[index]
        rows.append(("LibA", pep, "DEFG", 1, 5, 5, 1))
    rows.extend(
        [
            ("LibA", "MM", "DEFG", 3, 0, 0, 0),
            ("LibA", "NN", "HIKL", 1, 1, 5, 1),
            ("LibA", "PP", "HIKL", 3, 0, 0, 0),
        ]
    )
    frame = attach(
        pd.DataFrame(
            rows,
            columns=["library", "pep", "aff", "r001_count", "r009_count", "r010_count", "weak_label"],
        )
    )

    c1, _ = apply_cleaning(frame, "C1", promiscuity_breadth=10)
    assert not bool(((c1["pep"] == "NN") & (c1["weak_label"] == 1)).any())
    assert bool(((c1["pep"] == "PP") & (c1["weak_label"] == 0)).any())

    c3, flagged = apply_cleaning(frame, "C3", promiscuity_breadth=10)
    assert len(flagged) == 1
    assert set(c3["aff"]) == {"HIKL"}

    c4, _ = apply_cleaning(frame, "C4", promiscuity_breadth=10)
    # The DEFG negative shares its Affibody with positives but its peptide is
    # new; neither negative is a component-viable hard negative here.
    assert int(c4["weak_label"].eq(0).sum()) == 0


def test_balancing_preserves_natural_rows_and_downsampling_is_deterministic():
    frame = pd.DataFrame({"weak_label": [0, 0, 1, 1, 1, 1], "row": np.arange(6)})
    natural = apply_balance(frame, "natural_class_weight", seed=0)
    assert list(natural["row"]) == list(frame["row"])
    first = apply_balance(frame, "downsample_1to1", seed=7)
    second = apply_balance(frame, "downsample_1to1", seed=7)
    assert first.equals(second)
    assert first["weak_label"].value_counts().to_dict() == {0: 2, 1: 2}


def test_pooled_downsampling_balances_each_library_independently():
    frame = pd.DataFrame(
        {
            "library": ["LibA"] * 4 + ["LibB"] * 5,
            "weak_label": [0, 0, 0, 1, 0, 1, 1, 1, 1],
            "row": np.arange(9),
        }
    )
    selected = apply_balance(frame, "downsample_1to1", seed=3)
    counts = selected.groupby(["library", "weak_label"]).size()
    assert counts.to_dict() == {
        ("LibA", 0): 1,
        ("LibA", 1): 1,
        ("LibB", 0): 1,
        ("LibB", 1): 1,
    }


def test_metrics_and_compatible_scope_are_explicit():
    frame = pd.DataFrame(
        {
            "library": ["LibA"] * 4,
            "global_peptide_sha256": ["p1", "p1", "p2", "p2"],
            "global_affibody_sha256": ["a1", "a2", "a1", "a2"],
            "target_retention": [100.0, 0.0, 80.0, 20.0],
            "target_binder": [1, 0, 1, 0],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )
    metrics = evaluation_metrics(frame)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["global_ap_lift"] == pytest.approx(0.5)
    assert metrics["peptide_macro_precision_at_1"] == pytest.approx(1.0)

    fitted = pd.DataFrame(
        {
            "global_peptide_sha256": ["train-peptide"],
            "global_affibody_sha256": ["a1"],
        }
    )
    exposed = add_exposure_flags(frame, fitted, "peptide_cold")
    assert exposed["peptide_seen_in_fit"].eq(0).all()
    assert exposed["regime_compatible"].tolist() == [1, 0, 1, 0]


def test_two_way_residual_removes_fixed_effects_on_an_irregular_panel():
    # Missing p2/a2 makes one-pass row/column mean subtraction inexact.
    peptide = np.asarray(["p1", "p1", "p2", "p3", "p3"], dtype=str)
    affibody = np.asarray(["a1", "a2", "a1", "a1", "a2"], dtype=str)
    peptide_effect = {"p1": 10.0, "p2": -4.0, "p3": 2.0}
    affibody_effect = {"a1": 3.0, "a2": -5.0}
    values = np.asarray(
        [7.0 + peptide_effect[p] + affibody_effect[a] for p, a in zip(peptide, affibody)]
    )
    residual = _two_way_center(values, peptide, affibody)
    assert np.max(np.abs(residual)) < 1e-10


def test_zero_failure_csv_has_a_readable_schema(tmp_path):
    path = tmp_path / "failures.csv"
    pd.DataFrame(columns=list(FAILURE_COLUMNS)).to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert loaded.empty
    assert tuple(loaded.columns) == FAILURE_COLUMNS
