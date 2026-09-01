import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import evaluate_liba_later_round_site as later


def _tiny_retention_panel():
    return pd.DataFrame(
        {
            "pair_uid": ["p1a", "p1b", "p1c", "p2a", "p2b", "p2c"],
            "peptide_uid": ["pep1"] * 3 + ["pep2"] * 3,
            "affibody_uid": ["a", "b", "c", "a", "b", "c"],
            "target_retention": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0],
            "target_binder": [1, 0, 0, 0, 1, 0],
        }
    )


def test_metrics_average_within_peptide_and_resolve_top_ties_fractionally():
    panel = _tiny_retention_panel()
    score = np.asarray([0.9, 0.9, 0.1, 0.1, 0.9, 0.5])
    metrics, peptide = later.condition_metrics(panel, score)

    assert metrics["within_peptide_groups"] == 2
    assert np.isfinite(metrics["within_peptide_macro_spearman"])
    assert metrics["within_peptide_top_choice_success"] == pytest.approx(0.75)
    first = peptide.loc[peptide["peptide_uid"].eq("pep1")].iloc[0]
    assert first["top_tie_count"] == 2
    assert first["top_choice_binder_fraction"] == pytest.approx(0.5)


def test_peptide_cluster_bootstrap_is_paired_and_deterministic():
    panel = _tiny_retention_panel()
    score = np.asarray([0.9, 0.8, 0.1, 0.1, 0.9, 0.5])
    first = later.paired_peptide_cluster_bootstrap(panel, score, score, 200, 71)
    second = later.paired_peptide_cluster_bootstrap(panel, score, score, 200, 71)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["metric"]) == set(later.METRIC_NAMES)
    assert np.allclose(first["ci_lower_2_5"], 0.0)
    assert np.allclose(first["ci_upper_97_5"], 0.0)


def test_all_comparisons_aligns_pair_values_not_disjoint_series_indexes(monkeypatch):
    monkeypatch.setattr(later, "RETENTION_ROWS", 6)
    panel = _tiny_retention_panel()
    scores = {
        "A": np.asarray([0.9, 0.8, 0.1, 0.1, 0.9, 0.5]),
        "B": np.asarray([0.8, 0.7, 0.2, 0.2, 0.8, 0.4]),
        "C": np.asarray([0.7, 0.6, 0.3, 0.3, 0.7, 0.5]),
        "D": np.asarray([0.6, 0.5, 0.4, 0.4, 0.6, 0.5]),
    }
    blocks = []
    for arm in later.ARM_ORDER:
        block = panel.copy()
        block["arm"] = arm
        block["sampling_mode"] = "natural"
        block["sampling_seed"] = -1
        block["score"] = scores[arm]
        blocks.append(block)
    # Concatenation deliberately gives each arm a disjoint Series index range,
    # reproducing the canonical failure that elementwise Series.eq triggered.
    predictions = pd.concat(blocks, ignore_index=True)
    result = later.bootstrap_all_comparisons(predictions, draws=30, base_seed=19)
    assert len(result) == 3 * len(later.METRIC_NAMES)
    assert set(result["arm"]) == {"B", "C", "D"}


def test_arm_contract_requires_one_common_negative_set():
    blocks = []
    expected = {}
    for arm_index, arm in enumerate(later.ARM_ORDER):
        positives = 2 + arm_index
        expected[arm] = positives
        for index in range(positives):
            blocks.append(
                {
                    "arm": arm,
                    "pair_uid": "{}-positive-{}".format(arm, index),
                    "weak_label": 1,
                }
            )
        blocks.append({"arm": arm, "pair_uid": "common-negative", "weak_label": 0})
    labels = pd.DataFrame(blocks)
    observed = later.validate_arm_contract(labels, expected, expected_negatives=1)
    assert observed.set_index("arm").loc["D", "natural_positive"] == 5

    labels.loc[
        labels["arm"].eq("D") & labels["weak_label"].eq(0), "pair_uid"
    ] = "different-negative"
    with pytest.raises(ValueError, match="negative set differs"):
        later.validate_arm_contract(labels, expected, expected_negatives=1)


def _matched_fixture():
    rows = []
    for arm in later.ARM_ORDER:
        for index in range(5):
            rows.append(
                {
                    "arm": arm,
                    "pair_uid": "{}-positive-{}".format(arm, index),
                    "weak_label": 1,
                }
            )
    labels = pd.DataFrame(rows)
    membership_rows = []
    for arm in later.ARM_ORDER[1:]:
        for seed in later.MATCHED_SEEDS:
            candidates = labels.loc[labels["arm"].eq(arm), ["pair_uid"]].copy()
            candidates["rank_sha256"] = candidates["pair_uid"].map(
                lambda value: later._size_match_rank(value, seed)
            )
            candidates = candidates.sort_values(["rank_sha256", "pair_uid"]).head(2)
            for rank, row in enumerate(candidates.itertuples(index=False), start=1):
                membership_rows.append(
                    {
                        "arm": arm,
                        "sampling_seed": seed,
                        "target_positive_rows": 2,
                        "pair_uid": row.pair_uid,
                        "rank_sha256": row.rank_sha256,
                        "size_match_rank": rank,
                    }
                )
    return labels, pd.DataFrame(membership_rows)


def test_matched_contract_recomputes_sha256_subsamples():
    labels, membership = _matched_fixture()
    later.validate_matched_contract(membership, labels, matched_positive=2)
    tampered = membership.copy()
    tampered.loc[0, "rank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rank hash"):
        later.validate_matched_contract(tampered, labels, matched_positive=2)


def test_fold_audit_uses_exact_double_identity_cold_split():
    rows = []
    for peptide_index in range(30):
        for affibody_index in range(30):
            rows.append(
                {
                    "pair_uid": "pair-{}-{}".format(peptide_index, affibody_index),
                    "peptide_uid": "peptide-{}".format(peptide_index),
                    "affibody_uid": "affibody-{}".format(affibody_index),
                    "weak_label": (peptide_index + affibody_index) % 2,
                }
            )
    frame = pd.DataFrame(rows)
    primary, plans, audit = later.build_and_audit_fold_plans(
        frame, later._condition_metadata("A", "natural", -1)
    )
    assert len(primary) == 900
    assert len(plans) == 3
    audit = pd.DataFrame(audit)
    assert set(audit["role"]) == {"train", "guard", "validation"}
    assert len(audit) == 9
    assert audit["train_validation_peptide_overlap"].eq(0).all()
    assert audit["train_validation_affibody_overlap"].eq(0).all()


def test_prediction_hash_binds_order_as_well_as_values():
    first = later._prediction_sha256(["a", "b"], [0.1, 0.2])
    second = later._prediction_sha256(["b", "a"], [0.2, 0.1])
    assert first != second


def test_cli_defaults_match_prespecified_analysis():
    args = later.parse_args(
        [
            "--label-dir",
            "labels",
            "--retention-csv",
            "retention.csv",
            "--output-dir",
            "private_data/experiments/example",
        ]
    )
    assert args.bootstrap_draws == 10000
    assert args.bootstrap_seed == 20260820
    assert later.FOLDS == 3
    assert later.SPLIT_SEED == 17
    assert later.C_GRID == (0.001, 0.01, 0.1, 1.0)
    assert later.BALANCE == "all_class_weighted"
