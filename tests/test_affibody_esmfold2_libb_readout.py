import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC.esmfold2_libb_readout import (
    MODEL_NAMES,
    EsmFold2LibBReadout,
    EsmFold2PairReadout,
    FrozenFeatureStore,
    class_balanced_weights,
    class_weighted_binary_cross_entropy,
    parameter_report,
)
from downstream.AffibodyMHC.train_esmfold2_libb_readouts import (
    REPO_ROOT,
    TRAINING_LABEL_COLUMNS,
    _read_config,
    _sha256_file,
    load_canonical_training_labels,
)


CONFIG_PATH = (
    REPO_ROOT
    / "downstream/AffibodyMHC/configs/esmfold2_libb_frozen_readouts_v1.json"
)
LIBA_CONFIG_PATH = (
    REPO_ROOT
    / "downstream/AffibodyMHC/configs/esmfold2_liba_frozen_readouts_v1.json"
)
OLD_FOLD_MEMBERSHIP = (
    REPO_ROOT
    / "private_data/experiments/"
    "mint_selection_matched_confirmatory_libb_seed20260811_v1/"
    "fold_membership.csv"
)


def _features(batch=2):
    probabilities = torch.softmax(torch.randn(batch, 9, 58, 64), dim=-1)
    return {
        "distogram_probabilities": probabilities,
        "pair_states_symmetric": torch.randn(batch, 9, 58, 256),
        "single_inputs_peptide": torch.randn(batch, 9, 451),
        "single_inputs_affibody": torch.randn(batch, 58, 451),
    }


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_all_readouts_return_one_logit_and_normalized_position_attention(model_name):
    torch.manual_seed(4)
    model = EsmFold2LibBReadout(model_name, dropout=0.0).eval()
    logits, attention = model(**_features(), return_attention=True)

    assert logits.shape == (2,)
    assert attention.shape == (2, 9, 58)
    torch.testing.assert_close(attention.sum(dim=(1, 2)), torch.ones(2))
    assert torch.isfinite(logits).all()


def test_locked_models_are_small_and_have_exact_parameter_counts():
    config = _read_config(CONFIG_PATH)
    observed = parameter_report(config["architecture"])

    assert observed == {
        "distogram_only": 7713,
        "pair_state_only": 14241,
        "distogram_pair": 16417,
        "single_inputs_only": 35553,
        "full": 46433,
    }
    assert max(observed.values()) < 50_000


def test_liba_config_and_training_sidecar_reconstruct_locked_double_cold_folds():
    config = _read_config(LIBA_CONFIG_PATH)
    labels_root = REPO_ROOT / "private_data/derived/esmfold2_liba_training_labels_v1"
    base, membership = load_canonical_training_labels(
        labels_root / "training_labels.csv",
        labels_root / "manifest.json",
        config,
    )
    assert config["dataset"]["library"] == "LibA"
    assert len(base) == 22_542
    assert len(membership) == 3 * 22_542
    assert config["dataset"]["evaluation_rows"] == 108


def test_training_cli_runs_directly_without_pythonpath(tmp_path):
    script = (
        REPO_ROOT / "downstream/AffibodyMHC/train_esmfold2_libb_readouts.py"
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--training-labels" in result.stdout


@pytest.mark.parametrize(
    "model_name,ignored_names",
    [
        (
            "distogram_only",
            ("pair_states_symmetric", "single_inputs_peptide", "single_inputs_affibody"),
        ),
        (
            "pair_state_only",
            ("distogram_probabilities", "single_inputs_peptide", "single_inputs_affibody"),
        ),
        (
            "single_inputs_only",
            ("distogram_probabilities", "pair_states_symmetric"),
        ),
    ],
)
def test_attribution_controls_are_invariant_to_excluded_modalities(
    model_name, ignored_names
):
    torch.manual_seed(8)
    model = EsmFold2LibBReadout(model_name, dropout=0.0).eval()
    features = _features(batch=1)
    perturbed = dict(features)
    for name in ignored_names:
        perturbed[name] = torch.randn_like(features[name]) * 10_000

    with torch.inference_mode():
        first = model(**features)
        second = model(**perturbed)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_class_weighted_bce_uses_inverse_frequency_weights():
    labels = torch.tensor([0, 1, 1, 1])
    weights = class_balanced_weights(labels)
    torch.testing.assert_close(weights, torch.tensor([2.0, 2.0 / 3.0]))

    logits = torch.zeros(4)
    loss = class_weighted_binary_cross_entropy(logits, labels.float(), weights)
    torch.testing.assert_close(loss, torch.tensor(np.log(2.0), dtype=torch.float32))


def _write_fake_cache(
    root: Path, schema_version: str = "esmfold2-libb-feature-merge-v1"
):
    root.mkdir()
    metadata = root / "metadata.csv"
    metadata.write_text("row_index,row_id,split\n0,train-0,train\n1,eval-0,eval\n")
    arrays = {
        "distogram_probabilities": np.full((2, 9, 58, 64), 1.0 / 64, np.float16),
        "pair_states_symmetric": np.zeros((2, 9, 58, 256), np.float16),
        "single_inputs": np.zeros((2, 67, 451), np.float16),
    }
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", value, allow_pickle=False)
    metadata_record = {
        "file": "metadata.csv",
        "bytes": metadata.stat().st_size,
        "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "columns": ["row_index", "row_id", "split"],
        "rows": 2,
    }
    outputs = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        outputs[name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": "float16",
        }
    (root / "merge_complete.json").write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "supervision_fields_read": [],
                "row_count": 2,
                "row_indices_sha256": hashlib.sha256(b"[0,1]").hexdigest(),
                "row_ids_sha256": hashlib.sha256(
                    b'["train-0","eval-0"]'
                ).hexdigest(),
                "metadata": metadata_record,
                "arrays": outputs,
            }
        )
    )


def test_store_requires_completed_label_free_manifest_and_validates_checksums(tmp_path):
    cache = tmp_path / "cache"
    _write_fake_cache(cache)
    store = FrozenFeatureStore.open(cache, verify_all_checksums=True)

    assert store.row_ids == ("train-0", "eval-0")
    assert store.splits == ("train", "eval")
    assert tuple(store.arrays["pair_states_symmetric"].shape) == (2, 9, 58, 256)

    marker = cache / "MERGE_INCOMPLETE"
    marker.write_text("incomplete")
    with pytest.raises(ValueError, match="marked incomplete"):
        FrozenFeatureStore.open(cache)
    marker.unlink()

    with (cache / "metadata.csv").open("a") as handle:
        handle.write("2,tampered,eval\n")
    with pytest.raises(ValueError, match="metadata byte size"):
        FrozenFeatureStore.open(cache)


def test_liba_merged_cache_schema_uses_the_same_pair_readout_contract(tmp_path):
    cache = tmp_path / "liba-cache"
    _write_fake_cache(cache, schema_version="esmfold2-liba-feature-merge-v1")
    store = FrozenFeatureStore.open(cache)
    assert store.row_ids == ("train-0", "eval-0")
    assert EsmFold2LibBReadout is EsmFold2PairReadout


@pytest.mark.skipif(not OLD_FOLD_MEMBERSHIP.is_file(), reason="canonical fold fixture absent")
def test_training_sidecar_reconstructs_the_exact_existing_three_fold_contract(tmp_path):
    old = pd.read_csv(OLD_FOLD_MEMBERSHIP, dtype=str)
    base = old.loc[old["fold"].eq("0")].copy()
    labels = pd.DataFrame(
        {
            "row_id": base["pair_uid"],
            "weak_label": base["weak_label"],
            "peptide_id": "peptide-" + base["chain1_sha256"],
            "affibody_id": "affibody-" + base["chain2_sha256"],
            "chain1_sha256": base["chain1_sha256"],
            "chain2_sha256": base["chain2_sha256"],
        },
        columns=TRAINING_LABEL_COLUMNS,
    )
    labels_path = tmp_path / "training_labels.csv"
    labels.to_csv(labels_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "esmfold2-libb-training-labels-v1",
                "output": {
                    "sha256": _sha256_file(labels_path),
                    "columns": list(TRAINING_LABEL_COLUMNS),
                },
            }
        )
    )

    reconstructed_base, reconstructed = load_canonical_training_labels(
        labels_path, manifest_path, _read_config(CONFIG_PATH)
    )
    assert len(reconstructed_base) == 30_648
    assert len(reconstructed) == 3 * 30_648
    observed = reconstructed[["row_id", "fold", "role"]].sort_values(
        ["fold", "row_id"]
    )
    expected = old[["pair_uid", "fold", "role"]].rename(
        columns={"pair_uid": "row_id"}
    )
    expected["fold"] = expected["fold"].astype(int)
    expected = expected.sort_values(["fold", "row_id"])
    pd.testing.assert_frame_equal(
        observed.reset_index(drop=True), expected.reset_index(drop=True)
    )
