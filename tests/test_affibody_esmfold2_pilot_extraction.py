import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from downstream.AffibodyMHC.extract_esmfold2_libb_pilot import (
    REPO_ROOT,
    SAFE_OUTPUT_ROOT,
    _array_delta,
    _derive_interface_indices,
    _feature_bundle,
    _to_numpy,
    _validate_pair,
    _validated_output_dir,
)

PILOT_INPUTS = (
    REPO_ROOT / "private_data/derived/esmfold2_libb_pilot_v1/pilot_inputs.json"
)


def _fake_forward_metadata():
    return {
        "asym_id": torch.tensor([[0] * 270 + [1] * 58]),
        "residue_index": torch.tensor(
            [list(range(270)) + list(range(58))], dtype=torch.long
        ),
        "attention_mask": torch.ones(1, 328, dtype=torch.bool),
    }


def test_manifest_sequences_and_codes_validate_against_reference():
    manifest = json.loads(PILOT_INPUTS.read_text(encoding="utf-8"))
    reference = manifest["pairs"][0]

    for pair in manifest["pairs"]:
        sequence_a, sequence_b = _validate_pair(pair, reference)
        assert len(sequence_a) == 270
        assert len(sequence_b) == 58


def test_manifest_validation_rejects_a_code_sequence_disagreement():
    manifest = json.loads(PILOT_INPUTS.read_text(encoding="utf-8"))
    reference = manifest["pairs"][0]
    corrupted = copy.deepcopy(reference)
    corrupted["peptide_design_code"] = "AW"

    with pytest.raises(ValueError, match="peptide code does not map"):
        _validate_pair(corrupted, reference)


def test_interface_indices_are_last_nine_chain_a_tokens_and_all_of_chain_b():
    peptide, affibody = _derive_interface_indices(
        _fake_forward_metadata(), "A" * 270, "B" * 58
    )

    assert peptide.tolist() == list(range(261, 270))
    assert affibody.tolist() == list(range(270, 328))


def test_interface_mapping_rejects_nonlocal_chain_numbering():
    metadata = _fake_forward_metadata()
    metadata["residue_index"][0, 270:] += 1

    with pytest.raises(ValueError, match="chain B residue numbering"):
        _derive_interface_indices(metadata, "A" * 270, "B" * 58)


def test_pair_states_are_saved_in_both_aligned_directions_and_decompose_exactly():
    length = 5
    channels = 3
    pair_states = torch.empty(1, length, length, channels)
    for left in range(length):
        for right in range(length):
            for channel in range(channels):
                pair_states[0, left, right, channel] = 100 * left + 10 * right + channel
    logits = torch.arange(length * length * 4, dtype=torch.float32).reshape(
        1, length, length, 4
    )
    single_inputs = torch.arange(length * 7, dtype=torch.float32).reshape(1, length, 7)
    output = SimpleNamespace(
        distogram_logits=logits,
        pair_states=pair_states,
        single_inputs=single_inputs,
    )
    peptide = torch.tensor([1, 3])
    affibody = torch.tensor([0, 2, 4])

    features = _feature_bundle(output, peptide, affibody)
    expected_ab = pair_states[0][peptide][:, affibody].numpy().astype(np.float16)
    expected_ba = (
        pair_states[0][affibody][:, peptide].transpose(0, 1).numpy().astype(np.float16)
    )

    np.testing.assert_array_equal(features["pair_states_ab"], expected_ab)
    np.testing.assert_array_equal(features["pair_states_ba_aligned"], expected_ba)
    np.testing.assert_array_equal(
        features["pair_states_symmetric"] + features["pair_states_antisymmetric"],
        expected_ab,
    )
    np.testing.assert_array_equal(
        features["pair_states_symmetric"] - features["pair_states_antisymmetric"],
        expected_ba,
    )
    assert features["distogram_probabilities"].shape == (2, 3, 4)
    np.testing.assert_allclose(
        features["distogram_probabilities"].sum(axis=-1), 1.0, atol=1e-3
    )
    np.testing.assert_array_equal(
        features["single_inputs_peptide"],
        single_inputs[0, peptide].numpy().astype(np.float16),
    )
    np.testing.assert_array_equal(
        features["single_inputs_affibody"],
        single_inputs[0, affibody].numpy().astype(np.float16),
    )


def test_bfloat16_is_promoted_before_numpy_and_optionally_compacted_to_float16():
    tensor = torch.tensor([0.5, -3.25, 128.0], dtype=torch.bfloat16)

    lossless_range_preserving = _to_numpy(tensor)
    compact = _to_numpy(tensor, np.float16)

    assert lossless_range_preserving.dtype == np.float32
    assert compact.dtype == np.float16
    np.testing.assert_array_equal(lossless_range_preserving, tensor.float().numpy())
    np.testing.assert_array_equal(compact, tensor.float().numpy().astype(np.float16))


def test_array_delta_reports_exact_absolute_l2_and_relative_l2_values():
    reference = np.asarray([3.0, 4.0], dtype=np.float16)
    current = np.asarray([6.0, 8.0], dtype=np.float16)

    delta = _array_delta(current, reference)

    assert delta == {
        "max_absolute": 4.0,
        "mean_absolute": 3.5,
        "l2": 5.0,
        "relative_l2": 1.0,
    }
    assert _array_delta(reference, reference) == {
        "max_absolute": 0.0,
        "mean_absolute": 0.0,
        "l2": 0.0,
        "relative_l2": 0.0,
    }


def test_only_narrowly_named_direct_children_of_structure_pilot_are_deletable():
    allowed = SAFE_OUTPUT_ROOT / "esmfold2_libb_pilot_unit_test"
    assert _validated_output_dir(allowed) == allowed.resolve()

    rejected = (
        SAFE_OUTPUT_ROOT,
        REPO_ROOT,
        REPO_ROOT / "private_data",
        SAFE_OUTPUT_ROOT / "unrelated_experiment",
        SAFE_OUTPUT_ROOT / "nested" / "esmfold2_libb_pilot_unit_test",
        Path("/tmp/esmfold2_libb_pilot_unit_test"),
    )
    for path in rejected:
        with pytest.raises(ValueError):
            _validated_output_dir(path)
