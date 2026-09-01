import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import finetune_mint_selection_shared_epoch as shared


def _block(arm, epoch, fold, labels, probabilities, seed=20260811, library="LibA"):
    return pd.DataFrame(
        {
            "pair_uid": ["fold{}-pair{}".format(fold, index) for index in range(len(labels))],
            "weak_label": list(labels),
            "epoch": int(epoch),
            "probability": list(probabilities),
            "library": library,
            "arm": arm,
            "training_seed": int(seed),
            "fold": int(fold),
        }
    )


def _paired_predictions(per_epoch, folds=(0,), seed=20260811, library="LibA"):
    frames = []
    for fold in folds:
        labels = per_epoch[0][0]
        for epoch in shared.EPOCH_CANDIDATES:
            epoch_labels, head_probability, cross_probability = per_epoch[epoch]
            assert list(epoch_labels) == list(labels)
            frames.append(
                _block(
                    "head_only",
                    epoch,
                    fold,
                    labels,
                    head_probability,
                    seed,
                    library,
                )
            )
            frames.append(
                _block(
                    "lora_cross",
                    epoch,
                    fold,
                    labels,
                    cross_probability,
                    seed,
                    library,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _simple_predictions():
    labels = [0, 1]
    return _paired_predictions(
        {
            0: (labels, [0.01, 0.99], [0.01, 0.99]),
            1: (labels, [0.20, 0.80], [0.40, 0.60]),
            2: (labels, [0.10, 0.90], [0.10, 0.90]),
            3: (labels, [0.25, 0.75], [0.25, 0.75]),
        }
    )


def test_shared_selector_excludes_epoch_zero_and_uses_equal_arm_mean():
    validated, parity = shared.validate_paired_validation_predictions(
        _simple_predictions(), "LibA", 20260811
    )
    selected, table = shared.select_shared_positive_epoch(
        validated, "LibA", 20260811
    )
    assert selected == 2
    assert parity == pytest.approx(0.0)
    assert table.loc[table["epoch"].eq(0), "equal_arm_mean_log_loss"].iloc[0] < table.loc[
        table["epoch"].eq(2), "equal_arm_mean_log_loss"
    ].iloc[0]
    assert table["selected"].tolist() == [0, 0, 1, 0]
    assert bool(table["epoch0_better"].iloc[0])
    assert table["epoch0_minus_selected_log_loss"].iloc[0] < 0.0


def test_shared_selector_uses_pooled_rows_not_equal_fold_mean_or_class_weighting():
    labels_large = np.tile([0, 1], 50)
    labels_small = np.array([0, 1])
    frames = []
    for arm in shared.ARMS:
        for epoch in shared.EPOCH_CANDIDATES:
            if epoch == 1:
                large = np.where(labels_large == 1, 0.90, 0.10)
                small = np.array([0.99, 0.01])
            elif epoch == 2:
                large = np.where(labels_large == 1, 0.75, 0.25)
                small = np.array([0.10, 0.90])
            else:
                large = np.full(len(labels_large), 0.5)
                small = np.full(len(labels_small), 0.5)
            frames.append(_block(arm, epoch, 0, labels_large, large))
            frames.append(_block(arm, epoch, 1, labels_small, small))
    raw = pd.concat(frames, ignore_index=True)
    validated, _ = shared.validate_paired_validation_predictions(
        raw, "LibA", 20260811
    )
    selected, _ = shared.select_shared_positive_epoch(validated, "LibA", 20260811)
    assert selected == 1


def test_exact_positive_epoch_tie_uses_earlier_epoch_only():
    labels = [0, 0, 1, 1]
    same = [0.10, 0.40, 0.60, 0.90]
    raw = _paired_predictions(
        {epoch: (labels, same, same) for epoch in shared.EPOCH_CANDIDATES}
    )
    # Old per-arm selection columns, if present, are not inputs to the selector.
    raw["selected"] = np.where(raw["epoch"].eq(3), 1, 0)
    validated, _ = shared.validate_paired_validation_predictions(
        raw, "LibA", 20260811
    )
    selected, table = shared.select_shared_positive_epoch(validated, "LibA", 20260811)
    assert selected == 1
    assert table.loc[table["selected"].eq(1), "epoch"].tolist() == [1]


def test_epoch_zero_exact_tie_is_not_better_but_is_no_worse():
    labels = [0, 1]
    probability = [0.25, 0.75]
    raw = _paired_predictions(
        {
            epoch: (labels, probability, probability)
            for epoch in shared.EPOCH_CANDIDATES
        }
    )
    validated, _ = shared.validate_paired_validation_predictions(
        raw, "LibA", 20260811
    )
    selected, table = shared.select_shared_positive_epoch(validated, "LibA", 20260811)
    assert selected == 1
    assert not bool(table["epoch0_better"].iloc[0])
    assert bool(table["epoch0_no_worse"].iloc[0])
    assert table["epoch0_minus_selected_log_loss"].iloc[0] == pytest.approx(0.0)


def test_validation_and_selection_are_row_order_invariant():
    raw = _simple_predictions()
    shuffled = raw.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first, _ = shared.validate_paired_validation_predictions(raw, "LibA", 20260811)
    second, _ = shared.validate_paired_validation_predictions(
        shuffled, "LibA", 20260811
    )
    selected_first, table_first = shared.select_shared_positive_epoch(
        first, "LibA", 20260811
    )
    selected_second, table_second = shared.select_shared_positive_epoch(
        second, "LibA", 20260811
    )
    assert selected_first == selected_second
    pd.testing.assert_frame_equal(table_first, table_second)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_arm", "validation arms changed"),
        ("missing_epoch", "validation epochs changed"),
        ("duplicate", "duplicate weak-validation prediction"),
        ("changed_label", "membership or labels differ"),
        ("wrong_library", "validation library changed"),
        ("wrong_seed", "validation training seed changed"),
        ("nan", "non-finite validation probability"),
        ("infinite", "non-finite validation probability"),
        ("out_of_range", "outside"),
        ("fractional_label", "non-integer"),
        ("fractional_fold", "non-integer"),
        ("epoch0_parity", "epoch-0 validation parity failed"),
    ],
)
def test_paired_validation_fails_closed(mutation, message):
    raw = _simple_predictions()
    if mutation == "missing_arm":
        raw = raw.loc[~raw["arm"].eq("lora_cross")]
    elif mutation == "missing_epoch":
        raw = raw.loc[~raw["epoch"].eq(3)]
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    elif mutation == "changed_label":
        index = raw.index[(raw["arm"].eq("lora_cross")) & (raw["epoch"].eq(2))][0]
        raw.loc[index, "weak_label"] = 1 - int(raw.loc[index, "weak_label"])
    elif mutation == "wrong_library":
        raw.loc[0, "library"] = "LibB"
    elif mutation == "wrong_seed":
        raw.loc[0, "training_seed"] = 99
    elif mutation == "nan":
        raw.loc[0, "probability"] = np.nan
    elif mutation == "infinite":
        raw.loc[0, "probability"] = np.inf
    elif mutation == "out_of_range":
        raw.loc[0, "probability"] = 1.1
    elif mutation == "fractional_label":
        raw.loc[0, "weak_label"] = 0.49
    elif mutation == "fractional_fold":
        raw.loc[0, "fold"] = 0.5
    elif mutation == "epoch0_parity":
        index = raw.index[(raw["arm"].eq("lora_cross")) & (raw["epoch"].eq(0))][0]
        raw.loc[index, "probability"] += 1e-4
    with pytest.raises(ValueError, match=message):
        shared.validate_paired_validation_predictions(raw, "LibA", 20260811)


def test_fold_contract_rejects_missing_fold_row_even_when_arms_match():
    raw = _simple_predictions()
    contract = {0: {"validation": 3, "validation_positive": 1}}
    with pytest.raises(ValueError, match="row count changed"):
        shared.validate_paired_validation_predictions(
            raw, "LibA", 20260811, fold_contract=contract
        )


def test_validation_rejects_cross_fold_pair_reuse_and_canonical_mismatch():
    labels = [0, 1]
    raw = _paired_predictions(
        {
            epoch: (labels, [0.25, 0.75], [0.25, 0.75])
            for epoch in shared.EPOCH_CANDIDATES
        },
        folds=(0, 1),
    )
    reused = raw.copy()
    reused.loc[reused["fold"].eq(1), "pair_uid"] = reused.loc[
        reused["fold"].eq(1), "pair_uid"
    ].str.replace("fold1", "fold0")
    with pytest.raises(ValueError, match="more than one validation fold"):
        shared.validate_paired_validation_predictions(reused, "LibA", 20260811)

    expected = raw.loc[
        raw["arm"].eq("head_only") & raw["epoch"].eq(0),
        ["fold", "pair_uid", "weak_label"],
    ].copy()
    expected.loc[0, "pair_uid"] = "not-a-canonical-pair"
    with pytest.raises(ValueError, match="reconstructed canonical folds"):
        shared.validate_paired_validation_predictions(
            raw,
            "LibA",
            20260811,
            expected_membership=expected,
        )


def test_head_initialization_hash_is_deterministic_and_tensor_sensitive():
    class Classifier(object):
        pass

    first = Classifier()
    first.coef_ = np.array([[1.0, 2.0]], dtype=np.float64)
    first.intercept_ = np.array([3.0], dtype=np.float64)
    second = copy.deepcopy(first)
    assert shared.head_initialization_sha256(first) == shared.head_initialization_sha256(
        second
    )
    second.coef_[0, 1] += 0.25
    assert shared.head_initialization_sha256(first) != shared.head_initialization_sha256(
        second
    )


def test_paired_difference_is_cross_minus_head_with_primary_endpoint():
    metrics = pd.DataFrame(
        [
            {
                "arm": "head_only",
                "global_auroc": 0.50,
                "global_auprc": 0.40,
                "global_spearman": 0.30,
                "within_peptide_macro_spearman": 0.20,
            },
            {
                "arm": "lora_cross",
                "global_auroc": 0.55,
                "global_auprc": 0.47,
                "global_spearman": 0.39,
                "within_peptide_macro_spearman": 0.31,
            },
        ]
    )
    row = shared.paired_difference_record(metrics, "LibA", 20260811, 2)
    assert row["comparison"] == "lora_cross_minus_head_only"
    assert row["global_auroc_change"] == pytest.approx(0.05)
    assert row["within_peptide_macro_spearman_change"] == pytest.approx(0.11)


def test_copy_verified_source_output_rejects_tampering(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    path = source_dir / "weak_validation_predictions.csv"
    path.write_text("x\n1\n")
    source = {
        "run_dir": source_dir,
        "manifest": {
            "outputs": {
                path.name: {
                    "path": str(path),
                    "sha256": "not-the-live-hash",
                }
            }
        },
    }
    with pytest.raises(ValueError, match="changed"):
        shared._copy_verified_source_output(source, path.name, output_dir)


def _synthetic_verified_source(tmp_path):
    cache = tmp_path / "cache.npz"
    cache_manifest = tmp_path / "cache.manifest.json"
    rows = tmp_path / "rows.csv"
    rows_manifest = tmp_path / "rows.manifest.json"
    retention = tmp_path / "retention.npz"
    checkpoint = tmp_path / "mint.ckpt"
    config = tmp_path / "config.json"
    reference = tmp_path / "reference"
    reference.mkdir()
    for path in (
        cache,
        cache_manifest,
        rows,
        rows_manifest,
        retention,
        checkpoint,
        config,
    ):
        path.write_bytes(b"fixture")
    return {
        "library": "LibA",
        "training_seed": 20260811,
        "reference": {"directory": str(reference)},
        "configuration": {
            "training_seeds": [20260811],
            "epoch_candidates": [0, 1, 2, 3],
            "arms": ["frozen_logistic", "head_only", "lora_cross"],
            "split_seed": 17,
            "folds": 3,
            "c_grid": [0.001, 0.01, 0.1, 1.0],
            "batch_size": 64,
            "eval_batch_size": 64,
            "accumulation_steps": 1,
            "head_lr": 1e-4,
            "adapter_lr": 2e-4,
            "weight_decay": 0.01,
            "warmup_fraction": 0.1,
            "clip_norm": 1.0,
            "lora": {"rank": 2, "alpha": 4.0, "dropout": 0.05},
        },
        "sources": {
            "cache_npz_000": {"path": str(cache)},
            "cache_manifest_000": {"path": str(cache_manifest)},
            "cache_rows": {"path": str(rows)},
            "cache_rows_manifest": {"path": str(rows_manifest)},
            "retention_features": {"path": str(retention)},
            "checkpoint": {"path": str(checkpoint)},
            "config": {"path": str(config)},
        },
    }


def test_verified_source_arguments_preserve_three_epoch_schedule_horizon(tmp_path):
    source = _synthetic_verified_source(tmp_path)
    args = shared.training_arguments_from_verified_source(source, "cuda:0")
    assert args.max_epochs == 3
    assert args.training_seeds == [20260811]
    assert args.lora_rank == 2
    assert args.head_lr == pytest.approx(1e-4)
    assert args.adapter_lr == pytest.approx(2e-4)


def _paired_audit_rows():
    common = {
        "rows": 10,
        "positive": 5,
        "negative": 5,
        "class_weight_negative": 1.0,
        "class_weight_positive": 1.0,
        "membership_sha256": "membership",
        "feature_mean_sha256": "mean",
        "feature_scale_sha256": "scale",
        "run_seed": 123,
        "selected_epoch": 2,
        "schedule_total_steps": 30,
        "head_initialization_sha256": "head",
        "final_refit_epoch0_arm_max_abs_difference": 0.0,
        "head_parameter_delta_l2": 0.5,
    }
    return pd.DataFrame(
        [
            dict(
                common,
                arm="head_only",
                adapter_parameter_delta_l2=0.0,
                adapter_parameter_delta_max_abs=0.0,
            ),
            dict(
                common,
                arm="lora_cross",
                adapter_parameter_delta_l2=0.25,
                adapter_parameter_delta_max_abs=0.1,
            ),
        ]
    )


def test_paired_training_audit_enforces_schedule_seed_init_and_adapter_motion():
    audit = _paired_audit_rows()
    shared._assert_paired_training_audits(audit, 2, "head")
    changed_schedule = audit.copy()
    changed_schedule.loc[changed_schedule["arm"].eq("lora_cross"), "schedule_total_steps"] = 29
    with pytest.raises(ValueError, match="schedule_total_steps"):
        shared._assert_paired_training_audits(changed_schedule, 2, "head")
    no_adapter_motion = audit.copy()
    no_adapter_motion.loc[no_adapter_motion["arm"].eq("lora_cross"), "adapter_parameter_delta_l2"] = 0.0
    with pytest.raises(ValueError, match="did not move"):
        shared._assert_paired_training_audits(no_adapter_motion, 2, "head")
