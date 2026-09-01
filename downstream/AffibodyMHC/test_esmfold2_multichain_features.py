"""CPU-only structural tests for the pinned ESMFold2 feature interface.

These tests instantiate no model and download no checkpoint.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers.models.esmfold2.modeling_esmfold2 import (
    EsmFold2AtomInputs,
    EsmFold2Model,
)
from transformers.models.esmfold2.protein_utils import prepare_protein_features

from downstream.AffibodyMHC.esmfold2_multichain_features import (
    ATOM_INPUT_FIELDS,
    bundle_trunk_inputs,
    prepare_multichain_protein_features,
    prepare_multichain_trunk_inputs,
)


def _real_atom_count(features):
    return int(features["atom_attention_mask"].sum().item())


def test_one_chain_is_bitwise_identical_to_the_official_helper():
    official = prepare_protein_features("AGWY", device="cpu")
    composed = prepare_multichain_protein_features(("AGWY",), device="cpu")

    assert composed.keys() == official.keys()
    for name in official:
        assert torch.equal(composed[name], official[name]), name


def test_two_chain_token_metadata_and_query_msa_are_explicit():
    features = prepare_multichain_protein_features(("AG", "WGG"), device="cpu")

    assert features["token_index"].tolist() == [[0, 1, 2, 3, 4]]
    assert features["residue_index"].tolist() == [[0, 1, 0, 1, 2]]
    assert features["asym_id"].tolist() == [[0, 0, 1, 1, 1]]
    assert features["entity_id"].tolist() == [[1, 1, 2, 2, 2]]
    assert features["sym_id"].tolist() == [[0, 0, 0, 0, 0]]
    assert features["attention_mask"].tolist() == [[True] * 5]
    assert features["msa"].shape == (1, 1, 5)
    assert features["msa_attention_mask"].tolist() == [[[True] * 5]]


def test_identical_chains_share_entity_but_have_distinct_symmetry_ids():
    features = prepare_multichain_protein_features(("AG", "AG", "W"))

    assert features["asym_id"].tolist() == [[0, 0, 1, 1, 2]]
    assert features["entity_id"].tolist() == [[1, 1, 1, 1, 2]]
    assert features["sym_id"].tolist() == [[0, 0, 1, 1, 0]]


def test_atoms_are_stripped_offset_and_repacked_once():
    first = prepare_protein_features("AG", device="cpu")
    second = prepare_protein_features("WY", device="cpu")
    combined = prepare_multichain_protein_features(("AG", "WY"), device="cpu")

    first_atoms = _real_atom_count(first)
    second_atoms = _real_atom_count(second)
    total_atoms = first_atoms + second_atoms
    padded_atoms = combined["atom_attention_mask"].shape[1]

    assert padded_atoms == ((total_atoms + 31) // 32) * 32
    assert combined["atom_attention_mask"][0, :total_atoms].all()
    assert not combined["atom_attention_mask"][0, total_atoms:].any()

    first_mask = first["atom_attention_mask"][0]
    second_mask = second["atom_attention_mask"][0]
    expected_atom_to_token = torch.cat(
        [
            first["atom_to_token"][0, first_mask],
            second["atom_to_token"][0, second_mask] + 2,
        ]
    )
    expected_space_uid = torch.cat(
        [
            first["ref_space_uid"][0, first_mask],
            second["ref_space_uid"][0, second_mask] + 2,
        ]
    )
    torch.testing.assert_close(
        combined["atom_to_token"][0, :total_atoms], expected_atom_to_token
    )
    torch.testing.assert_close(
        combined["ref_space_uid"][0, :total_atoms], expected_space_uid
    )
    assert not combined["atom_to_token"][0, total_atoms:].any()
    assert not combined["ref_space_uid"][0, total_atoms:].any()

    expected_distogram_indices = torch.cat(
        [
            first["distogram_atom_idx"][0],
            second["distogram_atom_idx"][0] + first_atoms,
        ]
    )
    torch.testing.assert_close(
        combined["distogram_atom_idx"][0], expected_distogram_indices
    )

    # Every representative C-beta (or glycine C-alpha) maps back to exactly
    # the residue token it represents after the two offset operations.
    representative_atoms = combined["distogram_atom_idx"][0]
    representative_tokens = combined["atom_to_token"][0, representative_atoms]
    torch.testing.assert_close(representative_tokens, torch.arange(4))


def test_cross_chain_token_bonds_are_zero():
    first_length = 3
    features = prepare_multichain_protein_features(("ACD", "WYG"))
    bonds = features["token_bonds"][0, :, :, 0]

    assert not bonds[:first_length, first_length:].any()
    assert not bonds[first_length:, :first_length].any()
    # The official protein helper currently has no explicit token-bond edges at
    # all; sequential relationships are encoded by chain/residue positions.
    assert not bonds.any()


def test_bundle_matches_forward_signature_and_keeps_distogram_mapping_separate():
    raw = prepare_multichain_protein_features(("AG", "WY"))
    prepared = bundle_trunk_inputs(raw)

    assert isinstance(prepared.forward_kwargs["atom_inputs"], EsmFold2AtomInputs)
    assert "distogram_atom_idx" not in prepared.forward_kwargs
    assert torch.equal(prepared.distogram_atom_idx, raw["distogram_atom_idx"])
    for name in ATOM_INPUT_FIELDS:
        assert name not in prepared.forward_kwargs
        assert torch.equal(
            getattr(prepared.forward_kwargs["atom_inputs"], name), raw[name]
        )

    forward_parameters = set(inspect.signature(EsmFold2Model.forward).parameters)
    assert set(prepared.forward_kwargs).issubset(forward_parameters)


def test_convenience_function_produces_cpu_trunk_inputs_without_a_model():
    prepared = prepare_multichain_trunk_inputs(("AG", "WY"), device="cpu")

    assert prepared.forward_kwargs["token_index"].device.type == "cpu"
    assert prepared.forward_kwargs["atom_inputs"].ref_pos.device.type == "cpu"
    assert prepared.distogram_atom_idx.device.type == "cpu"


@pytest.mark.parametrize("sequences", [(), ("AG", "")])
def test_empty_complex_or_chain_is_rejected(sequences):
    with pytest.raises(ValueError):
        prepare_multichain_protein_features(sequences)


def test_one_string_is_not_misread_as_many_one_residue_chains():
    with pytest.raises(TypeError, match="sequence of chain strings"):
        prepare_multichain_protein_features("AG")
