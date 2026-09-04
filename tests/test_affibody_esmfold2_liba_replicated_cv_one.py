from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC import run_esmfold2_liba_replicated_cv_one as replicated
from downstream.AffibodyMHC import train_esmfold2_libb_readouts as training
from downstream.AffibodyMHC.esmfold2_libb_readout import (
    AFFIBODY_LENGTH,
    DISTOGRAM_CATEGORIES,
    PAIR_STATE_DIM,
    PEPTIDE_LENGTH,
    SINGLE_INPUT_DIM,
    FrozenFeatureDataset,
    FrozenFeatureStore,
    build_readout,
    class_weighted_binary_cross_entropy,
)


def _membership() -> pd.DataFrame:
    blocks = []
    for fold in range(3):
        for index in range(2505):
            blocks.append(
                {
                    "fold": fold,
                    "role": "validation",
                    "row_id": f"row-{fold}-{index}",
                    "weak_label": index % 2,
                }
            )
    return pd.DataFrame(blocks)


def _predictions(membership: pd.DataFrame) -> pd.DataFrame:
    frame = membership.copy()
    frame.insert(0, "model", "distogram_only")
    frame["probability"] = frame["weak_label"].map({0: 0.25, 1: 0.75})
    return frame[["model", "fold", "row_id", "weak_label", "probability"]]


def test_locked_replicate_seeds_are_exact() -> None:
    assert replicated.FIXED_REPLICATE_SEEDS == (
        20260811,
        20260812,
        20260813,
        20260814,
        20260815,
    )


def test_prediction_receipt_validates_each_fold_membership(tmp_path: Path) -> None:
    membership = _membership()
    path = tmp_path / "predictions.csv.gz"
    _predictions(membership).to_csv(path, index=False)
    records = replicated._validate_predictions(
        path, membership, "distogram_only"
    )
    assert [record["rows"] for record in records] == [2505, 2505, 2505]
    assert all(len(record["row_id_membership_sha256"]) == 64 for record in records)


def test_prediction_receipt_rejects_wrong_fold(tmp_path: Path) -> None:
    membership = _membership()
    predictions = _predictions(membership)
    predictions.loc[0, "fold"] = 1
    path = tmp_path / "predictions.csv.gz"
    predictions.to_csv(path, index=False)
    with pytest.raises(ValueError, match="membership changed"):
        replicated._validate_predictions(path, membership, "distogram_only")


def test_float16_loader_has_exact_model_input_forward_and_loss_parity(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(17)
    arrays = {
        "distogram_probabilities": rng.normal(
            size=(1, PEPTIDE_LENGTH, AFFIBODY_LENGTH, DISTOGRAM_CATEGORIES)
        ).astype(np.float16),
        "pair_states_symmetric": rng.normal(
            size=(1, PEPTIDE_LENGTH, AFFIBODY_LENGTH, PAIR_STATE_DIM)
        ).astype(np.float16),
        "single_inputs_peptide": rng.normal(
            size=(1, PEPTIDE_LENGTH, SINGLE_INPUT_DIM)
        ).astype(np.float16),
        "single_inputs_affibody": rng.normal(
            size=(1, AFFIBODY_LENGTH, SINGLE_INPUT_DIM)
        ).astype(np.float16),
    }
    store = FrozenFeatureStore(
        root=tmp_path,
        row_ids=("row-0",),
        splits=("train",),
        arrays=arrays,
    )
    dataset = FrozenFeatureDataset(
        store,
        cache_indices=[0],
        row_ids=["row-0"],
        required_features=tuple(arrays),
        labels=[1],
    )
    fast_item = dataset[0]
    assert all(fast_item[name].dtype == np.float16 for name in arrays)

    fast_batch = {
        name: torch.from_numpy(fast_item[name]).unsqueeze(0) for name in arrays
    }
    legacy_batch = {
        name: torch.from_numpy(
            np.array(arrays[name][0], dtype=np.float32, copy=True)
        ).unsqueeze(0)
        for name in arrays
    }
    fast_inputs = training._model_inputs(fast_batch, tuple(arrays), torch.device("cpu"))
    legacy_inputs = training._model_inputs(
        legacy_batch, tuple(arrays), torch.device("cpu")
    )
    for name in arrays:
        assert fast_inputs[name].dtype == torch.float32
        assert torch.equal(fast_inputs[name], legacy_inputs[name])

    torch.manual_seed(23)
    model = build_readout(
        "full",
        {
            "pair_hidden_dim": 32,
            "single_interaction_dim": 32,
            "classifier_hidden_dim": 32,
            "dropout": 0.1,
        },
    ).eval()
    with torch.inference_mode():
        fast_logit = model(**fast_inputs)
        legacy_logit = model(**legacy_inputs)
    assert torch.equal(fast_logit, legacy_logit)
    labels = torch.tensor([1.0])
    weights = torch.tensor([1.0, 1.0])
    fast_loss = class_weighted_binary_cross_entropy(fast_logit, labels, weights)
    legacy_loss = class_weighted_binary_cross_entropy(legacy_logit, labels, weights)
    assert torch.equal(fast_loss, legacy_loss)
