import numpy as np
import pandas as pd

from downstream.AffibodyMHC.run_selection_weak_ablation_suite import (
    _two_way_residual,
    apply_promiscuity_cleaning,
    binary_metrics,
    fixed_transfer_training_frame,
    make_weak_cv_splits,
    prepare_imbalance,
    site_feature_dicts,
)


def test_two_way_residual_accepts_internal_partner_column_names():
    frame = pd.DataFrame({
        "pep": ["A", "A", "B", "B"],
        "aff": ["X", "Y", "X", "Y"],
    })
    values = np.asarray([1.0, 2.0, 3.0, 5.0])
    residual = _two_way_residual(frame, values)
    assert residual.shape == values.shape
    assert np.isfinite(residual).all()


def test_rank_metrics_accept_non_probability_residual_scores():
    metrics = binary_metrics([0, 1, 0, 1], [-0.8, 0.4, -0.2, 0.9], probability_score=False)
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert np.isnan(metrics["brier"])
    assert np.isnan(metrics["log_loss"])


def _grid():
    rows = []
    peptides = ["AA", "AD", "AE", "AF", "AH", "AI"]
    affibodies = ["AAAA", "AAAD", "AAAE", "AAAF", "AAAH", "AAAI"]
    for p_index, peptide in enumerate(peptides):
        for a_index, affibody in enumerate(affibodies):
            rows.append({
                "library": "LibA", "pep": peptide, "aff": affibody,
                "weak_label": (p_index + a_index) % 2,
                "pair_uid": "{}-{}".format(peptide, affibody),
            })
    return pd.DataFrame(rows)


def test_all_cv_schemes_have_exactly_once_oof_and_cold_splits_do_not_leak():
    frame = _grid()
    for scheme in ("random_pair", "peptide_cold", "affibody_cold", "double_cold"):
        coverage = np.zeros(len(frame), dtype=int)
        for split in make_weak_cv_splits(frame, scheme, n_folds=3, seed=7):
            if not len(split["test"]):
                continue
            coverage[split["test"]] += 1
            train, test = frame.iloc[split["train"]], frame.iloc[split["test"]]
            assert set(train.pair_uid).isdisjoint(set(test.pair_uid))
            if scheme in ("peptide_cold", "double_cold"):
                assert set(train.pep).isdisjoint(set(test.pep))
            if scheme in ("affibody_cold", "double_cold"):
                assert set(train.aff).isdisjoint(set(test.aff))
        assert coverage.tolist() == [1] * len(frame)


def test_global_code_group_holds_same_cross_library_peptide_together():
    frame = pd.DataFrame([
        {"library": "LibA", "pep": "AA", "aff": "AAAA", "weak_label": 0, "pair_uid": "a"},
        {"library": "LibB", "pep": "AA", "aff": "AAAAA", "weak_label": 1, "pair_uid": "b"},
        {"library": "LibA", "pep": "AD", "aff": "AAAD", "weak_label": 1, "pair_uid": "c"},
        {"library": "LibB", "pep": "AE", "aff": "AAAAD", "weak_label": 0, "pair_uid": "d"},
    ])
    for split in make_weak_cv_splits(frame, "peptide_cold", 2, 9):
        test = frame.iloc[split["test"]]
        assert test.pep.eq("AA").sum() in (0, 2)


def test_promiscuity_cleaning_uses_distinct_positive_peptide_breadth():
    rows = []
    for index, peptide in enumerate(["AA", "AD", "AE", "AF"]):
        rows.append({"library": "LibA", "pep": peptide, "aff": "AAAA", "weak_label": 1, "pair_uid": "p{}".format(index)})
    rows += [
        {"library": "LibA", "pep": "AA", "aff": "AAAA", "weak_label": 1, "pair_uid": "duplicate-evidence"},
        {"library": "LibA", "pep": "AH", "aff": "AAAA", "weak_label": 0, "pair_uid": "negative-same-aff"},
        {"library": "LibA", "pep": "AA", "aff": "AAAD", "weak_label": 0, "pair_uid": "other-negative"},
        {"library": "LibA", "pep": "AD", "aff": "AAAD", "weak_label": 1, "pair_uid": "other-positive"},
    ]
    cleaned, audit = apply_promiscuity_cleaning(pd.DataFrame(rows), "affibody_breadth", 4)
    assert "AAAA" not in set(cleaned.aff)
    assert set(cleaned.weak_label) == {0, 1}
    assert audit["promiscuous_affibodies"] == 1


def test_round_support_cleaning_removes_only_unsupported_positives():
    frame = pd.DataFrame([
        {"library": "LibA", "pep": "AA", "aff": "AAAA", "weak_label": 1,
         "r009_count": 3, "r010_count": 3},
        {"library": "LibA", "pep": "AD", "aff": "AAAD", "weak_label": 1,
         "r009_count": 2, "r010_count": 9},
        {"library": "LibA", "pep": "AE", "aff": "AAAE", "weak_label": 0,
         "r009_count": 0, "r010_count": 0},
    ])
    cleaned, audit = apply_promiscuity_cleaning(
        frame, "round_support", positive_breadth=10, positive_round_count_min=3
    )
    assert cleaned[["pep", "weak_label"]].values.tolist() == [["AA", 1], ["AE", 0]]
    assert audit["positive_rows_removed_round_support"] == 1


def test_balancing_is_fold_local_deterministic_and_library_class_balanced():
    frame = pd.DataFrame([
        {"library": "LibA", "weak_label": label, "pair_uid": "a{}".format(index)}
        for index, label in enumerate([0, 0, 0, 1])
    ] + [
        {"library": "LibB", "weak_label": label, "pair_uid": "b{}".format(index)}
        for index, label in enumerate([0, 1, 1, 1, 1])
    ])
    down1, _ = prepare_imbalance(frame, "downsample_1to1", 3, "x")
    down2, _ = prepare_imbalance(frame.sample(frac=1, random_state=4), "downsample_1to1", 3, "x")
    assert set(down1.pair_uid) == set(down2.pair_uid)
    assert down1.groupby(["library", "weak_label"]).size().groupby(level=0).nunique().eq(1).all()
    kept, weights = prepare_imbalance(frame, "balanced_loss", 3, "x")
    totals = pd.DataFrame({"library": kept.library, "label": kept.weak_label, "weight": weights}).groupby(["library", "label"]).weight.sum()
    assert np.allclose(totals, totals.iloc[0])


def test_merged_features_align_physical_sites_and_add_conditioning():
    frame = pd.DataFrame([
        {"library": "LibA", "pep": "AD", "aff": "CEFG"},
        {"library": "LibB", "pep": "AH", "aff": "IKLMN"},
    ])
    shared = site_feature_dicts(frame, "pooled_shared")
    assert "aff_p13=C" in shared[0] and "aff_p17=E" in shared[0]
    assert "aff_p27=F" in shared[0] and "aff_p31=G" in shared[0]
    assert "aff_p6=I" in shared[1] and "aff_p10=K" in shared[1]
    assert "aff_p13=L" in shared[1] and "aff_p14=M" in shared[1] and "aff_p17=N" in shared[1]
    conditioned = site_feature_dicts(frame, "pooled_conditioned")
    assert "LibA|pep_p4=A" in conditioned[0]
    assert "LibB|aff_p13=L" in conditioned[1]


def test_transfer_filters_are_global_across_libraries_and_pair_cold_is_exact_code_cold():
    base = pd.DataFrame([
        {"library": "LibA", "pep": "AA", "aff": "AAAA", "weak_label": 1},
        {"library": "LibB", "pep": "AA", "aff": "AAAAD", "weak_label": 0},
        {"library": "LibA", "pep": "AD", "aff": "AAAD", "weak_label": 0},
        {"library": "LibB", "pep": "AE", "aff": "AAAAE", "weak_label": 1},
    ])
    retention = pd.DataFrame({
        "peptide_design_code": ["AA", "AF"],
        "affibody_design_code": ["AAAA", "AAAAF"],
    })
    pair = fixed_transfer_training_frame(base, retention, "pair_cold")
    assert not ((pair.pep == "AA") & (pair.aff == "AAAA")).any()
    peptide = fixed_transfer_training_frame(base, retention, "peptide_cold")
    assert "AA" not in set(peptide.pep)
    double = fixed_transfer_training_frame(base, retention, "double_cold")
    assert "AA" not in set(double.pep)
    assert "AAAA" not in set(double.aff)
