from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_esmfold2_liba_replicated_cv as aggregate


def _write(path: Path, value: bytes = b"fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_mean_logit_is_not_mean_probability_in_general() -> None:
    values = np.asarray([[0.1, 0.2, 0.9], [0.2, 0.8, 0.5]], dtype=float)
    logits, probabilities = aggregate._mean_logit_probabilities(values)
    expected_logits = (np.log(values) - np.log1p(-values)).mean(axis=1)
    assert np.allclose(logits, expected_logits)
    assert np.allclose(probabilities, 1.0 / (1.0 + np.exp(-expected_logits)))
    assert not np.isclose(probabilities[0], values[0].mean())
    with pytest.raises(ValueError, match="zero or one"):
        aggregate._mean_logit_probabilities(np.asarray([[0.0, 0.5]]))


def test_best_f1_threshold_uses_highest_threshold_tie_break() -> None:
    # Both selecting only the first item and selecting all four have F1=2/3.
    # The locked tie rule keeps the more conservative, higher threshold.
    result = aggregate._best_f1_threshold(
        [1, 0, 0, 1],
        [0.9, 0.8, 0.7, 0.6],
    )
    assert result["f1"] == pytest.approx(2.0 / 3.0)
    assert result["threshold"] == pytest.approx(0.9)
    assert result["predicted_positive"] == 1


def test_metric_bundle_macro_average_does_not_weight_large_peptides() -> None:
    frame = pd.DataFrame(
        {
            "peptide_id": ["small", "small", "large", "large", "large", "large"],
            "weak_label": [1, 0, 1, 0, 1, 0],
            "score": [0.9, 0.1, 0.9, 0.8, 0.7, 0.6],
        }
    )
    metrics = aggregate._metric_bundle(frame, "score")
    # small AP=1; large AP=(1 + 2/3)/2=5/6; unweighted macro=(1+5/6)/2=11/12.
    assert metrics["within_peptide_evaluable"] == 2
    assert metrics["within_peptide_ap"] == pytest.approx(11.0 / 12.0)


def test_completion_hash_validation_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    names = (
        "execution_contract.json",
        "selected_epochs.json",
        "weak_validation_summary.json",
        "weak_validation_predictions.csv.gz",
    )
    for name in names:
        _write(run_dir / name)
    contract = {"model": "distogram_only"}
    (run_dir / "execution_contract.json").write_text(json.dumps(contract))
    checkpoints = {}
    for fold in range(3):
        name = f"distogram_only__fold{fold}.pt"
        _write(run_dir / "cv_checkpoints" / name, f"fold={fold}\n".encode())
        checkpoints[name] = aggregate._sha256_file(run_dir / "cv_checkpoints" / name)
    completion = {
        "execution_contract_sha256": aggregate._sha256_file(
            run_dir / "execution_contract.json"
        ),
        "selected_epochs_sha256": aggregate._sha256_file(
            run_dir / "selected_epochs.json"
        ),
        "weak_validation_summary_sha256": aggregate._sha256_file(
            run_dir / "weak_validation_summary.json"
        ),
        "weak_validation_predictions_sha256": aggregate._sha256_file(
            run_dir / "weak_validation_predictions.csv.gz"
        ),
        "checkpoint_sha256": checkpoints,
    }
    aggregate._validate_completion_hashes(run_dir, completion)
    _write(run_dir / "weak_validation_predictions.csv.gz", b"tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        aggregate._validate_completion_hashes(run_dir, completion)


def test_cache_payload_verification_streams_declared_files(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    _write(metadata, b"row_id\nrow-1\n")
    arrays = {}
    for name in ("distogram_probabilities", "pair_states_symmetric", "single_inputs"):
        path = tmp_path / f"{name}.npy"
        _write(path, f"{name}-bytes".encode())
        arrays[name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": aggregate._sha256_file(path),
        }
    receipt = {
        "metadata": {
            "file": metadata.name,
            "bytes": metadata.stat().st_size,
            "sha256": aggregate._sha256_file(metadata),
        },
        "arrays": arrays,
    }
    result = aggregate._verify_cache_payloads(tmp_path.resolve(), receipt)
    assert result["all_declared_payloads_streamed"] is True
    assert result["stage"] == "aggregation_time_not_training_time"
    _write(tmp_path / "single_inputs.npy", b"tampered")
    with pytest.raises(ValueError, match="payload size changed|payload checksum changed"):
        aggregate._verify_cache_payloads(tmp_path.resolve(), receipt)


def test_fold_summary_metrics_are_recomputed_from_predictions() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [0, 0, 1, 1, 2, 2],
            "weak_label": [0, 1, 0, 1, 0, 1],
            "probability": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
        }
    )
    folds = []
    for fold in range(3):
        block = predictions.loc[predictions["fold"].eq(fold)]
        metrics = aggregate.training._binary_metrics(
            block["weak_label"].to_numpy(), block["probability"].to_numpy()
        )
        folds.append(
            {
                "fold": fold,
                "n_train": 10 + fold,
                "best_metrics": metrics,
            }
        )
    summary = {
        "selection_rule": "integer median of three weak-validation best epochs",
        "folds": folds,
        "mean_best_metrics": {
            name: float(np.mean([record["best_metrics"][name] for record in folds]))
            for name in ("log_loss", "brier", "auroc", "average_precision")
        },
    }
    config = {
        "validation": {
            "fold_contracts": {
                str(fold): {"train": 10 + fold} for fold in range(3)
            }
        }
    }
    aggregate._validate_summary_against_predictions(summary, predictions, config)
    summary["folds"][0]["best_metrics"]["average_precision"] = 0.0
    with pytest.raises(ValueError, match="differs from predictions"):
        aggregate._validate_summary_against_predictions(summary, predictions, config)


def _synthetic_runs() -> list[aggregate.ValidatedRun]:
    row_id = [f"row-{index}" for index in range(8)]
    peptide = ["p1"] * 4 + ["p2"] * 4
    label = [1, 0, 1, 0] * 2
    base_scores = {
        "distogram_only": [0.9, 0.2, 0.8, 0.1] * 2,
        "pair_state_only": [0.9, 0.8, 0.7, 0.6] * 2,
        "distogram_pair": [0.2, 0.9, 0.1, 0.8] * 2,
        "single_inputs_only": [0.9, 0.8, 0.6, 0.7] * 2,
        "full": [0.8, 0.9, 0.7, 0.6] * 2,
    }
    runs = []
    for family in aggregate.MODEL_NAMES:
        for seed_index, seed in enumerate(aggregate.FIXED_REPLICATE_SEEDS):
            scores = np.asarray(base_scores[family], dtype=float)
            # A seed-specific monotone logit shift changes calibration while
            # preserving the ranking used by within-peptide AP.
            logits = np.log(scores) - np.log1p(-scores) + (seed_index - 2) * 0.01
            scores = 1.0 / (1.0 + np.exp(-logits))
            predictions = pd.DataFrame(
                {
                    "row_id": row_id,
                    "fold": [0, 0, 1, 1, 2, 2, 2, 2],
                    "weak_label": label,
                    "peptide_id": peptide,
                    "probability": scores,
                }
            )
            first_epoch = seed_index * 3 + 1
            folds = [
                {
                    "model": family,
                    "fold": fold,
                    "seed": seed + 1000 * aggregate.MODEL_NAMES.index(family) + fold,
                    "best_epoch": first_epoch + fold,
                    "best_metrics": {
                        "log_loss": 0.5,
                        "auroc": 0.5,
                        "average_precision": 0.5,
                    },
                }
                for fold in range(3)
            ]
            runs.append(
                aggregate.ValidatedRun(
                    family=family,
                    replicate_seed=seed,
                    run_dir=Path(f"/{family}/{seed}"),
                    contract={},
                    summary={"folds": folds},
                    selected_epoch_in_run=first_epoch + 1,
                    predictions=predictions,
                    source_hashes={},
                )
            )
    return runs


def _dummy_inputs() -> aggregate.InputContract:
    reference = pd.DataFrame(
        {
            "row_id": [f"row-{index}" for index in range(8)],
            "fold": [0, 0, 1, 1, 2, 2, 2, 2],
            "weak_label": [1, 0, 1, 0] * 2,
            "peptide_id": ["p1"] * 4 + ["p2"] * 4,
            "frozen_mint_layer9": [0.9, 0.1, 0.8, 0.2] * 2,
        }
    )
    return aggregate.InputContract(
        config_path=Path("/config"),
        config={},
        config_sha256="a",
        labels_path=Path("/labels"),
        labels_sha256="b",
        labels_manifest_path=Path("/manifest"),
        labels_manifest_sha256="c",
        feature_cache=Path("/cache"),
        cache_manifest_path=Path("/cache/merge_complete.json"),
        cache_manifest_sha256="d",
        cache_payload_verification={},
        reference_path=Path("/reference"),
        reference_sha256="e",
        reference=reference,
        reference_metrics={"within_peptide_ap": 0.7},
    )


def test_aggregate_enforces_cartesian_product_and_structure_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP", 0.7)
    runs = _synthetic_runs()
    permuted = runs[1].predictions.iloc[::-1].reset_index(drop=True)
    for column in runs[1].predictions.columns:
        runs[1].predictions[column] = permuted[column].to_numpy()
    result = aggregate._aggregate_runs(runs, _dummy_inputs())
    metrics, family_summary, fold_epochs, aggregate_metrics, predictions, selection, thresholds = result
    assert len(metrics) == 25
    assert len(fold_epochs) == 75
    assert len(predictions) == 5 * 8
    assert set(aggregate_metrics["family"]) == set(aggregate.MODEL_NAMES)
    assert selection["selected_feature_family"] == "distogram_only"
    assert selection["models"]["distogram_only"]["selected_final_epoch"] == 8
    assert selection["models"]["distogram_only"]["supports_structure_claim"] is True
    assert selection["models"]["single_inputs_only"]["supports_structure_claim"] is False
    assert family_summary.loc[
        family_summary["family"].eq("distogram_only"), "seeds_beating_mint_layer9"
    ].iloc[0] == 5
    assert thresholds["prediction_rule"] == "score >= threshold"

    with pytest.raises(ValueError, match="Cartesian product"):
        aggregate._aggregate_runs(runs[:-1], _dummy_inputs())


def test_structure_gate_requires_four_of_five_individual_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(aggregate, "MINT_LAYER9_REFERENCE_WITHIN_PEPTIDE_AP", 0.7)
    runs = _synthetic_runs()
    affected = [
        run
        for run in runs
        if run.family == "distogram_only"
        and run.replicate_seed in aggregate.FIXED_REPLICATE_SEEDS[-2:]
    ]
    for run in affected:
        # Frozen dataclass, mutable contained frame: reverse the rank for two
        # replicates while leaving the other three favorable.
        run.predictions["probability"] = 1.0 - run.predictions["probability"]
    selection = aggregate._aggregate_runs(runs, _dummy_inputs())[5]
    record = selection["models"]["distogram_only"]
    assert record["seeds_beating_mint_layer9"] == 3
    assert record["reproducibly_beats_mint_layer9"] is False
    assert record["supports_structure_claim"] is False
    assert selection["best_structure_derived_family_by_gate_metric"] == "distogram_only"
    assert "selected_structure_derived_family" not in selection
    assert "distogram_only" not in selection["passing_structure_families"]
