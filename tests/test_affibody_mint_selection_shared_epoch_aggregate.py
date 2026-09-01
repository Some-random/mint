import copy
import concurrent.futures
import json
import math
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_mint_selection_shared_epoch as aggregate
from downstream.AffibodyMHC import evaluate_cached_weak_mint as cached_eval
from downstream.AffibodyMHC import finetune_mint_selection_matched as matched
from downstream.AffibodyMHC import finetune_mint_selection_shared_epoch as shared_trainer
from downstream.AffibodyMHC.code_only_baseline import sha256_file


def _weak_predictions(epoch_probabilities=None, library="LibA", seed=20260811):
    labels = np.array([0, 1, 0, 1], dtype=int)
    if epoch_probabilities is None:
        epoch_probabilities = {
            "head_only": {
                0: [0.4, 0.6, 0.4, 0.6],
                1: [0.3, 0.7, 0.3, 0.7],
                2: [0.2, 0.8, 0.2, 0.8],
                3: [0.25, 0.75, 0.25, 0.75],
            },
            "lora_cross": {
                0: [0.4, 0.6, 0.4, 0.6],
                1: [0.4, 0.6, 0.4, 0.6],
                2: [0.1, 0.9, 0.1, 0.9],
                3: [0.2, 0.8, 0.2, 0.8],
            },
        }
    rows = []
    for arm in aggregate.ARMS:
        for epoch in range(4):
            for index, (label, probability) in enumerate(
                zip(labels, epoch_probabilities[arm][epoch])
            ):
                rows.append(
                    {
                        "library": library,
                        "arm": arm,
                        "training_seed": seed,
                        "fold": index // 2,
                        "epoch": epoch,
                        "pair_uid": "pair-{}".format(index),
                        "weak_label": label,
                        "probability": probability,
                    }
                )
    return pd.DataFrame(rows)


def _valid_manifest_configuration(library="LibA", seed=20260811):
    configuration = {
        "library": library,
        "regime": matched.REGIME,
        "cleaning": matched.CLEANING,
        "balance": matched.BALANCE,
        "folds": matched.CANONICAL_FOLDS,
        "split_seed": matched.CANONICAL_SPLIT_SEED,
        "c_grid": list(matched.CANONICAL_C_GRID),
        "selected_C": matched.PRIMARY_CONTRACT[library]["selected_c"],
        "epoch_candidates": [0, 1, 2, 3],
        "shared_positive_epoch_candidates": [1, 2, 3],
        "epoch_selection": shared_trainer.EPOCH_SELECTION_DESCRIPTION,
        "chosen_positive_epoch": 2,
        "training_seeds": [seed],
        "arms": list(aggregate.ARMS),
        "max_epochs": 3,
        "lr_schedule_horizon_epochs": 3,
        "batch_size": 64,
        "eval_batch_size": 64,
        "accumulation_steps": 1,
        "head_lr": 1e-4,
        "adapter_lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_fraction": 0.1,
        "clip_norm": 1.0,
        "lora": {
            "rank": 2,
            "alpha": 4.0,
            "dropout": 0.05,
            "layers_zero_based": [31, 32],
            "projections": ["q_proj", "v_proj"],
            "placement": "multimer_attn",
        },
        "primary_endpoint": "within_peptide_macro_spearman",
        "secondary_endpoints": [
            "global_spearman",
            "global_auprc",
            "global_auroc",
        ],
        "retention_threshold_for_auroc_ap": 75.0,
        "retention_usage": {
            "partner_identities": "used before fitting",
            "numeric_retention_and_75_percent_labels": "final metrics only; never used for C or epoch selection",
        },
    }
    return {
        "configuration": configuration,
        "canonical_contract": matched.PRIMARY_CONTRACT[library],
    }


def test_configuration_matches_the_shared_trainer_contract_exactly():
    manifest = _valid_manifest_configuration()
    library, seed, configuration = aggregate._validate_configuration(
        manifest, "synthetic"
    )
    assert (library, seed) == ("LibA", 20260811)
    assert configuration["lr_schedule_horizon_epochs"] == 3
    compressed = copy.deepcopy(manifest)
    compressed["configuration"]["lr_schedule_horizon_epochs"] = 2
    with pytest.raises(ValueError, match="lr_schedule_horizon_epochs"):
        aggregate._validate_configuration(compressed, "compressed")


def test_shared_epoch_selection_uses_equal_arm_loss_and_positive_epochs_only():
    chosen, table = aggregate.recompute_shared_epoch_selection(
        _weak_predictions(), "LibA", 20260811
    )
    assert chosen == 2
    assert table.loc[table["selected"].eq(1), "epoch"].tolist() == [2]
    epoch2 = table.loc[table["epoch"].eq(2)].iloc[0]
    assert epoch2["equal_arm_mean_log_loss"] == pytest.approx(
        0.5
        * (epoch2["head_only_log_loss"] + epoch2["lora_cross_log_loss"])
    )
    assert not bool(epoch2["epoch0_better"])
    assert not bool(epoch2["epoch0_no_worse"])


def test_shared_epoch_selection_reports_better_epoch0_but_does_not_select_it():
    probabilities = {
        arm: {
            0: [0.01, 0.99, 0.01, 0.99],
            1: [0.4, 0.6, 0.4, 0.6],
            2: [0.4, 0.6, 0.4, 0.6],
            3: [0.3, 0.7, 0.3, 0.7],
        }
        for arm in aggregate.ARMS
    }
    chosen, table = aggregate.recompute_shared_epoch_selection(
        _weak_predictions(probabilities), "LibA", 20260811
    )
    assert chosen == 3
    assert table.loc[table["selected"].eq(1), "epoch"].tolist() == [3]
    assert table["epoch0_better"].map(bool).all()
    assert table["epoch0_no_worse"].map(bool).all()
    assert (table["epoch0_minus_selected_log_loss"] < 0.0).all()


def test_shared_epoch_selection_uses_earlier_epoch_as_only_tie_break():
    probabilities = {
        arm: {
            0: [0.45, 0.55, 0.45, 0.55],
            1: [0.2, 0.8, 0.2, 0.8],
            2: [0.2, 0.8, 0.2, 0.8],
            3: [0.3, 0.7, 0.3, 0.7],
        }
        for arm in aggregate.ARMS
    }
    chosen, _ = aggregate.recompute_shared_epoch_selection(
        _weak_predictions(probabilities), "LibA", 20260811
    )
    assert chosen == 1


def test_epoch0_exact_tie_is_no_worse_but_never_selected():
    probabilities = {
        arm: {
            0: [0.2, 0.8, 0.2, 0.8],
            1: [0.2, 0.8, 0.2, 0.8],
            2: [0.3, 0.7, 0.3, 0.7],
            3: [0.4, 0.6, 0.4, 0.6],
        }
        for arm in aggregate.ARMS
    }
    chosen, table = aggregate.recompute_shared_epoch_selection(
        _weak_predictions(probabilities), "LibA", 20260811
    )
    assert chosen == 1
    assert not table["epoch0_better"].map(bool).any()
    assert table["epoch0_no_worse"].map(bool).all()
    assert np.allclose(table["epoch0_minus_selected_log_loss"], 0.0)


def test_shared_epoch_selection_rejects_arm_membership_change():
    predictions = _weak_predictions()
    predictions.loc[
        predictions["arm"].eq("lora_cross")
        & predictions["epoch"].eq(2)
        & predictions["pair_uid"].eq("pair-0"),
        "weak_label",
    ] = 1
    with pytest.raises(ValueError, match="membership or labels differ"):
        aggregate.recompute_shared_epoch_selection(predictions, "LibA", 20260811)


def test_shared_epoch_selection_rejects_fractional_fold_or_label():
    fractional_fold = _weak_predictions()
    fractional_fold.loc[0, "fold"] = 0.5
    with pytest.raises(ValueError, match="fold must be integer-valued"):
        aggregate.recompute_shared_epoch_selection(
            fractional_fold, "LibA", 20260811
        )
    fractional_label = _weak_predictions()
    fractional_label.loc[
        fractional_label["weak_label"].eq(0), "weak_label"
    ] = 0.49
    fractional_label.loc[
        fractional_label["weak_label"].eq(1), "weak_label"
    ] = 1.49
    with pytest.raises(ValueError, match="weak_label must be integer-valued"):
        aggregate.recompute_shared_epoch_selection(
            fractional_label, "LibA", 20260811
        )


def _retention_panel(prefix="x"):
    return pd.DataFrame(
        {
            "pair_uid": ["{}-pair-{}".format(prefix, index) for index in range(6)],
            "chain1_sha256": ["p1", "p1", "p1", "p2", "p2", "p2"],
            "chain2_sha256": ["a1", "a2", "a3", "a1", "a2", "a3"],
            "sequence_pair_sha256": [
                "{}-sequence-{}".format(prefix, index) for index in range(6)
            ],
            "target_retention": [10.0, 80.0, 90.0, 20.0, 70.0, 100.0],
            "target_binder": [0, 1, 1, 0, 0, 1],
        }
    )


def _retention_predictions(library="LibA", seed=20260811, epoch=2, prefix="x"):
    panel = _retention_panel(prefix)
    probability = {
        "head_only": np.array([0.2, 0.75, 0.6, 0.25, 0.55, 0.85]),
        "lora_cross": np.array([0.15, 0.85, 0.65, 0.3, 0.45, 0.95]),
    }
    blocks = []
    for arm in aggregate.ARMS:
        block = panel.copy()
        block["library"] = library
        block["arm"] = arm
        block["training_seed"] = seed
        block["selected_epoch"] = epoch
        block["probability"] = probability[arm]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _stored_metrics(predictions, library="LibA", seed=20260811, epoch=2):
    rows = []
    for arm in aggregate.ARMS:
        block = predictions.loc[predictions["arm"].eq(arm)].reset_index(drop=True)
        rows.append(
            aggregate._canonical_retention_metrics(
                block, library, arm, seed, epoch
            )
        )
    return pd.DataFrame(rows)


def _patch_retention_contract(monkeypatch, library, panel):
    contract = copy.deepcopy(matched.PRIMARY_CONTRACT)
    contract[library]["retention_rows"] = len(panel)
    contract[library]["retention_positive"] = int(panel["target_binder"].sum())
    contract[library]["retention_membership_sha256"] = cached_eval.membership_sha256(
        panel
    )
    monkeypatch.setattr(matched, "PRIMARY_CONTRACT", contract)


def test_retention_audit_recomputes_metrics_and_rejects_tampering(monkeypatch):
    predictions = _retention_predictions()
    metrics = _stored_metrics(predictions)
    _patch_retention_contract(
        monkeypatch,
        "LibA",
        predictions.loc[predictions["arm"].eq("head_only")],
    )
    observed, panel = aggregate.audit_retention_tables(
        predictions, metrics, "LibA", 20260811, 2, "synthetic"
    )
    assert len(observed) == 2
    assert len(panel) == 6
    tampered = metrics.copy()
    tampered.loc[
        tampered["arm"].eq("lora_cross"), "within_peptide_macro_spearman"
    ] += 0.01
    with pytest.raises(ValueError, match="differs from recomputation"):
        aggregate.audit_retention_tables(
            predictions, tampered, "LibA", 20260811, 2, "tampered"
        )


def _training_audit(library="LibA", seed=20260811, epoch=2):
    contract = matched.PRIMARY_CONTRACT[library]
    weights = matched.balanced_class_weights(
        np.concatenate(
            [
                np.ones(contract["positive"], dtype=int),
                np.zeros(contract["negative"], dtype=int),
            ]
        )
    )
    run_seed = matched.derived_seed(seed, "final_refit", -1)
    rows = []
    for arm, trainable in (("head_only", 2561), ("lora_cross", 23041)):
        rows.append(
            {
                "stage": "final_refit",
                "arm": arm,
                "training_seed": seed,
                "fold": -1,
                "rows": contract["rows"],
                "positive": contract["positive"],
                "negative": contract["negative"],
                "class_weight_negative": weights[0],
                "class_weight_positive": weights[1],
                "membership_sha256": contract["membership_sha256"],
                "feature_mean_sha256": "1" * 64,
                "feature_scale_sha256": "2" * 64,
                "trainable_parameters": trainable,
                "trainable_names": json.dumps(
                    sorted(aggregate._expected_trainable_names(arm))
                ),
                "run_seed": run_seed,
                "selected_epoch": epoch,
                "schedule_total_steps": 1059,
                "lr_schedule_horizon_epochs": 3,
                "epoch0_probability_max_abs_error": 0.001,
                "head_initialization_sha256": "3" * 64,
                "source_run_manifest_sha256": "4" * 64,
                "source_validation_epoch0_arm_max_abs_difference": 0.0,
                "final_refit_epoch0_arm_max_abs_difference": 0.0,
                "source_frozen_reproduction_max_abs_difference": 0.0,
                "head_parameter_delta_l2": 1.0,
                "head_parameter_delta_max_abs": 0.1,
                "adapter_parameter_delta_l2": 0.0 if arm == "head_only" else 0.5,
                "adapter_parameter_delta_max_abs": 0.0
                if arm == "head_only"
                else 0.05,
            }
        )
    return pd.DataFrame(rows)


def _training_configuration():
    return {
        "max_epochs": 3,
        "lr_schedule_horizon_epochs": 3,
        "batch_size": 64,
        "accumulation_steps": 1,
    }


def test_training_audit_requires_paired_initialization_and_expected_deltas():
    audit = _training_audit()
    observed = aggregate._validate_training_audit(
        audit,
        "LibA",
        20260811,
        2,
        _training_configuration(),
        "synthetic",
    )
    assert len(observed) == 2
    changed_hash = audit.copy()
    changed_hash.loc[
        changed_hash["arm"].eq("lora_cross"), "head_initialization_sha256"
    ] = "4" * 64
    with pytest.raises(ValueError, match="head_initialization_sha256"):
        aggregate._validate_training_audit(
            changed_hash,
            "LibA",
            20260811,
            2,
            _training_configuration(),
            "changed-hash",
        )
    moved_adapter = audit.copy()
    moved_adapter.loc[
        moved_adapter["arm"].eq("head_only"), "adapter_parameter_delta_l2"
    ] = 0.1
    with pytest.raises(ValueError, match="head-only arm changed adapters"):
        aggregate._validate_training_audit(
            moved_adapter,
            "LibA",
            20260811,
            2,
            _training_configuration(),
            "moved-adapter",
        )
    wrong_placement = audit.copy()
    cross_index = wrong_placement.index[
        wrong_placement["arm"].eq("lora_cross")
    ][0]
    names = json.loads(wrong_placement.loc[cross_index, "trainable_names"])
    names[2] = names[2].replace("multimer_attn", "self_attn")
    wrong_placement.loc[cross_index, "trainable_names"] = json.dumps(names)
    with pytest.raises(ValueError, match="prescribed placement"):
        aggregate._validate_training_audit(
            wrong_placement,
            "LibA",
            20260811,
            2,
            _training_configuration(),
            "wrong-placement",
        )


def _model_delta_fixture():
    seed = 20260811
    epoch = 2
    output_hashes = {
        arm: ("a" if arm == "head_only" else "b") * 64
        for arm in aggregate.ARMS
    }
    verified_outputs = {
        "model_delta_seed{}_{}.pt".format(seed, arm): {
            "path": "/unused",
            "sha256": output_hashes[arm],
        }
        for arm in aggregate.ARMS
    }
    source = {"sources": {"checkpoint": {"sha256": "c" * 64}}}
    metadata = {}
    for arm in aggregate.ARMS:
        filename = "model_delta_seed{}_{}.pt".format(seed, arm)
        metadata[arm] = {
            "filename": filename,
            "sha256": output_hashes[arm],
            "library": "LibA",
            "arm": arm,
            "training_seed": seed,
            "selected_epoch": epoch,
            "base_checkpoint_sha256": "c" * 64,
            "primary_membership_sha256": matched.PRIMARY_CONTRACT["LibA"][
                "membership_sha256"
            ],
            "head_initialization_sha256": "3" * 64,
        }
    return (
        {"model_deltas": metadata},
        verified_outputs,
        _training_audit(),
        source,
    )


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("filename", "wrong.pt"),
        ("sha256", "0" * 64),
        ("library", "LibB"),
        ("arm", "head_only"),
        ("training_seed", 99),
        ("selected_epoch", 3),
        ("base_checkpoint_sha256", "0" * 64),
        ("primary_membership_sha256", "0" * 64),
        ("head_initialization_sha256", "0" * 64),
    ],
)
def test_model_delta_metadata_cross_bindings_are_enforced(key, bad_value):
    manifest, outputs, audit, source = _model_delta_fixture()
    aggregate._validate_model_delta_metadata(
        manifest,
        outputs,
        audit,
        source,
        "LibA",
        20260811,
        2,
        "synthetic",
    )
    tampered = copy.deepcopy(manifest)
    tampered["model_deltas"]["lora_cross"][key] = bad_value
    with pytest.raises(ValueError, match="model-delta"):
        aggregate._validate_model_delta_metadata(
            tampered,
            outputs,
            audit,
            source,
            "LibA",
            20260811,
            2,
            "tampered",
        )
def _fake_metrics(library, seed, cross_increment=0.05):
    seed_offset = (seed - 20260811) * 0.01
    rows = []
    for arm, value in (
        ("head_only", 0.60 + seed_offset),
        ("lora_cross", 0.60 + seed_offset + cross_increment),
    ):
        row = {
            "library": library,
            "arm": arm,
            "training_seed": seed,
            "selected_epoch": 2,
        }
        for metric in aggregate.SUMMARY_METRICS:
            row[metric] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_paired_changes_and_summary_use_paired_seeds_and_sample_sd():
    metrics = pd.concat(
        [
            _fake_metrics(library, seed)
            for library in aggregate.EXPECTED_LIBRARIES
            for seed in aggregate.EXPECTED_TRAINING_SEEDS
        ],
        ignore_index=True,
    )
    paired = aggregate.build_paired_changes(metrics)
    assert len(paired) == 6
    assert np.allclose(paired["global_auroc_change"], 0.05)
    summary = aggregate.build_summary(metrics, paired)
    head = summary.loc[
        summary["library"].eq("LibA")
        & summary["result_type"].eq("model_score")
        & summary["result"].eq("head_only")
    ].iloc[0]
    assert head["global_auroc_mean"] == pytest.approx(0.61)
    assert head["global_auroc_sample_sd"] == pytest.approx(0.01)
    change = summary.loc[
        summary["library"].eq("LibA")
        & summary["result_type"].eq("paired_change")
    ].iloc[0]
    assert change["global_auroc_mean"] == pytest.approx(0.05)
    assert change["global_auroc_sample_sd"] == pytest.approx(0.0, abs=1e-15)


def _selected_rows(epoch0_cross=0.4, selected_cross=0.3, selected_head=0.4):
    rows = []
    for library in aggregate.EXPECTED_LIBRARIES:
        for seed in aggregate.EXPECTED_TRAINING_SEEDS:
            rows.extend(
                [
                    {
                        "library": library,
                        "training_seed": seed,
                        "epoch": 0,
                        "selected": 0,
                        "head_only_log_loss": 0.5,
                        "lora_cross_log_loss": epoch0_cross,
                    },
                    {
                        "library": library,
                        "training_seed": seed,
                        "epoch": 2,
                        "selected": 1,
                        "head_only_log_loss": selected_head,
                        "lora_cross_log_loss": selected_cross,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_within_chain_gate_is_strict_and_explicitly_exploratory():
    metrics = pd.concat(
        [
            _fake_metrics(library, seed)
            for library in aggregate.EXPECTED_LIBRARIES
            for seed in aggregate.EXPECTED_TRAINING_SEEDS
        ],
        ignore_index=True,
    )
    paired = aggregate.build_paired_changes(metrics)
    passed = aggregate.build_within_chain_gate(_selected_rows(), paired)
    assert passed["placement_gate_pass"].map(bool).all()
    assert passed["interpretation"].str.contains("exploratory").all()
    epoch0_tie = aggregate.build_within_chain_gate(
        _selected_rows(epoch0_cross=0.3, selected_cross=0.3, selected_head=0.4),
        paired,
    )
    assert not epoch0_tie["placement_gate_pass"].map(bool).any()
    head_tie = aggregate.build_within_chain_gate(
        _selected_rows(epoch0_cross=0.4, selected_cross=0.3, selected_head=0.3),
        paired,
    )
    assert not head_tie["placement_gate_pass"].map(bool).any()
    retention_tie = paired.copy()
    retention_tie.loc[
        retention_tie["library"].eq("LibA"),
        "within_peptide_macro_spearman_change",
    ] = 0.0
    retention_gate = aggregate.build_within_chain_gate(
        _selected_rows(), retention_tie
    )
    assert not bool(
        retention_gate.loc[
            retention_gate["library"].eq("LibA"), "placement_gate_pass"
        ].iloc[0]
    )
    nonfinite = _selected_rows()
    nonfinite.loc[
        nonfinite["library"].eq("LibA") & nonfinite["selected"].eq(1),
        "lora_cross_log_loss",
    ] = np.nan
    nonfinite_gate = aggregate.build_within_chain_gate(nonfinite, paired)
    assert not bool(
        nonfinite_gate.loc[
            nonfinite_gate["library"].eq("LibA"), "placement_gate_pass"
        ].iloc[0]
    )


def _fake_verified_run(library, seed):
    predictions = _retention_predictions(
        library=library, seed=seed, prefix=library.lower()
    )
    metrics = _fake_metrics(library, seed)
    selection_rows = []
    for epoch in range(4):
        selection_rows.append(
            {
                "library": library,
                "training_seed": seed,
                "epoch": epoch,
                "selected": int(epoch == 2),
            }
        )
    return {
        "library": library,
        "training_seed": seed,
        "run_name": "{}_seed{}".format(library.lower(), seed),
        "configuration": {
            "library": library,
            "training_seeds": [seed],
            "selected_C": matched.PRIMARY_CONTRACT[library]["selected_c"],
            "common": "same",
        },
        "runtime_contract": {"gpu_name": "synthetic"},
        "source": {"library": library, "training_seed": seed},
        "metrics": metrics,
        "selection": pd.DataFrame(selection_rows),
        "predictions": predictions,
        "training_audit": pd.DataFrame(
            {
                "arm": list(aggregate.ARMS),
                "training_seed": [seed, seed],
            }
        ),
    }


def _six_fake_runs():
    return [
        _fake_verified_run(library, seed)
        for library in aggregate.EXPECTED_LIBRARIES
        for seed in aggregate.EXPECTED_TRAINING_SEEDS
    ]


def test_combine_enforces_exact_six_run_coverage(monkeypatch):
    monkeypatch.setattr(
        aggregate.matched_aggregate,
        "combine_verified_runs",
        lambda sources: {
            "common_source_hashes": {"checkpoint": "hash"},
            "library_reference_source_hashes": {
                "LibA": {"reference": "a"},
                "LibB": {"reference": "b"},
            },
        },
    )
    combined = aggregate.combine_verified_runs(_six_fake_runs())
    assert len(combined["metrics"]) == 12
    assert len(combined["selection"]) == 24
    runs = _six_fake_runs()
    runs[-1] = copy.deepcopy(runs[-2])
    with pytest.raises(ValueError, match="duplicate library/training-seed"):
        aggregate.combine_verified_runs(runs)


def test_source_matched_provenance_and_copied_hashes_are_bound(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "matched"
    shared_dir = tmp_path / "shared"
    source_dir.mkdir()
    shared_dir.mkdir()
    manifest_path = source_dir / "manifest.json"
    weak_path = source_dir / "weak_validation_predictions.csv"
    manifest_path.write_text("{}\n")
    weak_path.write_text("x\n1\n")
    outputs = {}
    source_outputs = {}
    for name, content in (
        ("c_validation.csv", "x\n1\n"),
        ("fold_membership.csv", "x\n2\n"),
        ("weak_validation_predictions.csv", "x\n1\n"),
    ):
        source_path = source_dir / name
        shared_path = shared_dir / name
        source_path.write_text(content)
        shared_path.write_text(content)
        source_outputs[name] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
        outputs[name] = {
            "path": str(shared_path),
            "sha256": sha256_file(shared_path),
        }
    provenance = {
        "directory": str(source_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "schema_version": matched.SCHEMA_VERSION,
        "library": "LibA",
        "training_seed": 20260811,
        "weak_validation_path": str(weak_path),
        "weak_validation_sha256": sha256_file(weak_path),
    }
    fake_source = {
        "library": "LibA",
        "training_seed": 20260811,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": {"outputs": source_outputs},
    }
    monkeypatch.setattr(
        aggregate.matched_aggregate,
        "load_verified_run",
        lambda path: fake_source,
    )
    observed, observed_provenance = aggregate._validate_source_matched_run(
        {"source_matched_run": provenance},
        shared_dir,
        outputs,
        "LibA",
        20260811,
        "synthetic",
    )
    assert observed is fake_source
    assert observed_provenance == provenance
    tampered = copy.deepcopy(outputs)
    tampered["c_validation.csv"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="copied c_validation.csv differs"):
        aggregate._validate_source_matched_run(
            {"source_matched_run": provenance},
            shared_dir,
            tampered,
            "LibA",
            20260811,
            "tampered",
        )


def test_recorded_output_rejects_symlink_even_when_hash_matches(tmp_path):
    external = tmp_path / "external.csv"
    external.write_text("x\n1\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    linked = run_dir / "table.csv"
    linked.symlink_to(external)
    record = {"path": str(linked), "sha256": sha256_file(external)}
    with pytest.raises(ValueError, match="may not be a symlink"):
        aggregate._verified_recorded_file(record, linked, "linked")


def test_output_must_be_disjoint_from_shared_and_source_inputs(tmp_path):
    shared = tmp_path / "shared"
    source = tmp_path / "source"
    shared.mkdir()
    source.mkdir()
    aggregate._assert_output_disjoint(tmp_path / "separate", [shared, source])
    with pytest.raises(ValueError, match="must be disjoint"):
        aggregate._assert_output_disjoint(shared / "aggregate", [shared, source])


def test_noreplace_link_cannot_replace_existing_output_entry(tmp_path):
    staged = tmp_path / "staged.txt"
    staged.write_text("new")
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "staged.txt"
    existing.write_text("old")
    output_fd = os.open(str(output), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(ValueError, match="refusing overwrite"):
            aggregate._link_staged_file_noreplace(
                staged, "staged.txt", output_fd
            )
    finally:
        os.close(output_fd)
    assert staged.read_text() == "new"
    assert existing.read_text() == "old"


def test_atomic_publish_is_private_hash_bound_manifest_last_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    output = tmp_path / "aggregate"
    table = pd.DataFrame({"value": [1, 2]})
    link_order = []
    real_link = aggregate.os.link

    def recording_link(source, destination, **kwargs):
        link_order.append(destination)
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(aggregate.os, "link", recording_link)

    def manifest_builder(records):
        return {"schema_version": "test", "outputs": records}

    manifest = aggregate._atomic_publish(
        output,
        {"table.csv": table},
        "summary\n",
        manifest_builder,
    )
    assert output.is_dir()
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for filename in ("table.csv", "summary.md", "manifest.json"):
        assert stat.S_IMODE((output / filename).stat().st_mode) == 0o600
    assert manifest["outputs"]["table.csv"]["path"] == str(
        (output / "table.csv").resolve()
    )
    assert manifest["outputs"]["table.csv"]["sha256"] == sha256_file(
        output / "table.csv"
    )
    assert link_order[-1] == "manifest.json"
    with pytest.raises(ValueError, match="refusing overwrite"):
        aggregate._atomic_publish(
            output,
            {"table.csv": table},
            "summary\n",
            manifest_builder,
        )


def test_atomic_publish_cleans_owned_claim_after_precommit_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "aggregate"
    real_link = aggregate._link_staged_file_noreplace
    calls = {"count": 0}

    def fail_second_link(source, filename, output_fd):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected publication failure")
        return real_link(source, filename, output_fd)

    monkeypatch.setattr(
        aggregate, "_link_staged_file_noreplace", fail_second_link
    )
    with pytest.raises(OSError, match="injected publication failure"):
        aggregate._atomic_publish(
            output,
            {"a.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert not output.exists()
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


def test_atomic_publish_cleans_payload_when_link_succeeds_then_audit_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "aggregate"
    calls = {"count": 0}

    def link_then_fail(source, filename, output_fd):
        calls["count"] += 1
        os.link(
            str(source),
            filename,
            dst_dir_fd=output_fd,
            follow_symlinks=False,
        )
        raise OSError("injected post-link audit failure")

    monkeypatch.setattr(
        aggregate, "_link_staged_file_noreplace", link_then_fail
    )
    with pytest.raises(OSError, match="post-link audit failure"):
        aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert calls["count"] == 1
    assert not os.path.lexists(str(output))
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


def test_atomic_publish_cleans_claim_when_parent_fsync_fails(tmp_path, monkeypatch):
    output = tmp_path / "aggregate"
    real_fsync = aggregate.os.fsync
    directory_syncs = {"count": 0}

    def fail_parent_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs["count"] += 1
            if directory_syncs["count"] == 2:
                raise OSError("injected parent fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(aggregate.os, "fsync", fail_parent_sync)
    with pytest.raises(OSError, match="parent fsync failure"):
        aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert directory_syncs["count"] == 2
    assert not os.path.lexists(str(output))
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


def test_atomic_publish_directory_claim_race_preserves_racer_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "aggregate"
    real_mkdir = aggregate.os.mkdir

    def raced_mkdir(path, mode=0o777, *args, **kwargs):
        if Path(path) == output:
            real_mkdir(path, mode, *args, **kwargs)
            (output / "existing.txt").write_text("existing")
            raise FileExistsError("injected claim race")
        return real_mkdir(path, mode, *args, **kwargs)

    monkeypatch.setattr(aggregate.os, "mkdir", raced_mkdir)
    with pytest.raises(ValueError, match="refusing overwrite"):
        aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert (output / "existing.txt").read_text() == "existing"
    assert set(path.name for path in output.iterdir()) == {"existing.txt"}
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


def test_atomic_publish_never_rolls_back_a_linked_manifest(tmp_path, monkeypatch):
    output = tmp_path / "aggregate"
    real_link = aggregate._link_staged_file_noreplace

    def fail_after_manifest_link(source, filename, output_fd):
        identity = real_link(source, filename, output_fd)
        if filename == "manifest.json":
            raise OSError("injected post-commit interruption")
        return identity

    monkeypatch.setattr(
        aggregate, "_link_staged_file_noreplace", fail_after_manifest_link
    )
    with pytest.raises(OSError, match="post-commit interruption"):
        aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        "summary.md",
        "table.csv",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["outputs"]["table.csv"]["sha256"] == sha256_file(
        output / "table.csv"
    )
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


def test_atomic_publish_rejects_dangling_output_symlink(tmp_path):
    output = tmp_path / "aggregate"
    target = tmp_path / "missing-target"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing overwrite"):
        aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "summary\n",
            lambda records: {"schema_version": "test", "outputs": records},
        )
    assert output.is_symlink()
    assert not target.exists()


def test_atomic_publish_concurrent_claim_has_one_winner_without_mixed_files(tmp_path):
    output = tmp_path / "aggregate"

    def publish(writer):
        return aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"writer": [writer]})},
            "writer {}\n".format(writer),
            lambda records: {
                "schema_version": "test",
                "writer": writer,
                "outputs": records,
            },
        )

    successes = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish, writer) for writer in (1, 2)]
        for future in futures:
            try:
                successes.append(future.result())
            except ValueError as error:
                failures.append(error)
    assert len(successes) == 1
    assert len(failures) == 1
    stored_manifest = json.loads((output / "manifest.json").read_text())
    stored_table = pd.read_csv(output / "table.csv")
    assert stored_manifest["writer"] == successes[0]["writer"]
    assert stored_table["writer"].tolist() == [stored_manifest["writer"]]
    assert stored_manifest["outputs"]["table.csv"]["sha256"] == sha256_file(
        output / "table.csv"
    )
    assert list(tmp_path.glob(".aggregate-staging-*")) == []


@pytest.mark.skipif(
    not os.environ.get("MINT_LUSTRE_TEST_ROOT"),
    reason="set MINT_LUSTRE_TEST_ROOT to exercise publication on Lustre",
)
def test_atomic_publish_on_configured_lustre_root():
    root = Path(os.environ["MINT_LUSTRE_TEST_ROOT"]).resolve()
    filesystem = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(root)], universal_newlines=True
    ).strip()
    assert filesystem == "lustre"
    with tempfile.TemporaryDirectory(
        prefix=".shared-epoch-publisher-test-", dir=str(root)
    ) as temporary:
        output = Path(temporary) / "aggregate"
        build = lambda records: {"schema_version": "test", "outputs": records}
        manifest = aggregate._atomic_publish(
            output,
            {"table.csv": pd.DataFrame({"value": [1, 2]})},
            "summary\n",
            build,
        )
        assert set(path.name for path in output.iterdir()) == {
            "manifest.json",
            "summary.md",
            "table.csv",
        }
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in output.iterdir()
        )
        before = {
            path.name: (path.stat().st_ino, sha256_file(path))
            for path in output.iterdir()
        }
        assert manifest["outputs"]["table.csv"]["sha256"] == sha256_file(
            output / "table.csv"
        )
        with pytest.raises(ValueError, match="refusing overwrite"):
            aggregate._atomic_publish(
                output,
                {"table.csv": pd.DataFrame({"value": [9]})},
                "replacement\n",
                build,
            )
        after = {
            path.name: (path.stat().st_ino, sha256_file(path))
            for path in output.iterdir()
        }
        assert after == before
