import numpy as np
import pytest
import torch

from downstream.AffibodyMHC.mint_structure_fusion import (
    LIBB_DESIGNED_SITE_MINT_ADDRESSES,
    MINT_CHAIN_MEAN_DIM,
    MINT_LAYER5_FEATURE_NAME,
    MINT_RESIDUE_DIM,
    FrozenMappedMintResidueFusion,
    FrozenMintLateFusion,
    MintResidueAddress,
    aligned_row_indices,
    gather_mint_residue_features,
)
from downstream.AffibodyMHC.libb_structural_readout import (
    CapacityMatchedNonlinearReadout,
    pool_feature_row,
)


def test_existing_libb_layer5_late_fusion_is_parameter_free_and_frozen():
    structure = torch.randn(3, 128, requires_grad=True)
    mint = torch.randn(3, MINT_CHAIN_MEAN_DIM, requires_grad=True)
    fusion = FrozenMintLateFusion(structural_dim=128)

    output = fusion(
        structural_vector=structure,
        mint_layer_05_chain_mean=mint,
    )

    assert fusion.required_features == ("structural_vector", MINT_LAYER5_FEATURE_NAME)
    assert fusion.output_dim == 128 + MINT_CHAIN_MEAN_DIM
    assert output.shape == (3, fusion.output_dim)
    assert list(fusion.parameters()) == []
    assert not output.requires_grad
    torch.testing.assert_close(output[:, :128], structure.detach())
    torch.testing.assert_close(output[:, 128:], mint.detach())


def test_late_fusion_rejects_misaligned_or_nonfinite_inputs():
    fusion = FrozenMintLateFusion(structural_dim=4)
    with pytest.raises(ValueError, match="batch sizes differ"):
        fusion(
            structural_vector=torch.zeros(2, 4),
            mint_layer_05_chain_mean=torch.zeros(3, MINT_CHAIN_MEAN_DIM),
        )
    bad = torch.zeros(2, MINT_CHAIN_MEAN_DIM)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fusion(
            structural_vector=torch.zeros(2, 4),
            mint_layer_05_chain_mean=bad,
        )


def test_late_fusion_plugs_into_the_capacity_matched_generic_readout():
    fusion = FrozenMintLateFusion(structural_dim=3, mint_dim=5)
    fused = fusion(
        structural_vector=torch.randn(4, 3),
        mint_layer_05_chain_mean=torch.randn(4, 5),
    )
    readout = CapacityMatchedNonlinearReadout(
        input_dim=fusion.output_dim,
        adapter_dim=12,
        hidden_dims=(6, 3),
        dropout=0.0,
        projection_seed=17,
    )

    logits = readout(fused)

    assert logits.shape == (4,)
    assert logits.requires_grad


@pytest.mark.parametrize("residues", [7, 128])
def test_residue_fusion_supports_stab_and_rde_layouts(residues):
    fusion = FrozenMappedMintResidueFusion(structural_dim=128)
    structural = torch.randn(2, residues, 128, requires_grad=True)
    mint = torch.randn(2, residues, MINT_RESIDUE_DIM, requires_grad=True)
    residue_mask = torch.ones(2, residues, dtype=torch.bool)
    mapped_mask = torch.ones(2, residues, dtype=torch.bool)

    output = fusion(
        structural_residue_features=structural,
        mint_residue_features=mint,
        residue_mask=residue_mask,
        mint_mapped_mask=mapped_mask,
    )

    assert output.shape == (2, residues, 128 + MINT_RESIDUE_DIM + 1)
    assert fusion.output_dim == 1409
    assert list(fusion.parameters()) == []
    assert not output.requires_grad
    torch.testing.assert_close(output[..., :128], structural.detach())
    torch.testing.assert_close(output[..., 128:-1], mint.detach())
    torch.testing.assert_close(output[..., -1], torch.ones(2, residues))


def test_unmapped_mint_and_padded_contents_cannot_change_fused_features():
    torch.manual_seed(9)
    fusion = FrozenMappedMintResidueFusion(structural_dim=3, mint_dim=5)
    structural = torch.randn(1, 4, 3)
    mint = torch.randn(1, 4, 5)
    residue_mask = torch.tensor([[True, True, True, False]])
    mapped_mask = torch.tensor([[True, False, True, False]])
    first = fusion(
        structural_residue_features=structural,
        mint_residue_features=mint,
        residue_mask=residue_mask,
        mint_mapped_mask=mapped_mask,
    )

    changed_structure = structural.clone()
    changed_mint = mint.clone()
    changed_structure[0, 3] = float("nan")
    changed_mint[0, 1] = float("nan")
    changed_mint[0, 3] = float("nan")
    second = fusion(
        structural_residue_features=changed_structure,
        mint_residue_features=changed_mint,
        residue_mask=residue_mask,
        mint_mapped_mask=mapped_mask,
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0, 3], torch.zeros(fusion.output_dim))
    torch.testing.assert_close(first[0, 1, 3:8], torch.zeros(5))
    assert first[0, 0, -1].item() == 1.0
    assert first[0, 1, -1].item() == 0.0


def test_residue_fusion_can_use_the_generic_masked_mean_max_view():
    fusion = FrozenMappedMintResidueFusion(structural_dim=2, mint_dim=3)
    values = fusion(
        structural_residue_features=torch.randn(1, 4, 2),
        mint_residue_features=torch.randn(1, 4, 3),
        residue_mask=torch.tensor([[True, True, True, False]]),
        mint_mapped_mask=torch.tensor([[True, False, True, False]]),
    )

    pooled = pool_feature_row(
        values[0].numpy(),
        "mean_max",
        np.asarray([True, True, True, False]),
    )

    assert pooled.shape == (2 * fusion.output_dim,)


def test_residue_fusion_rejects_mapping_into_padding():
    fusion = FrozenMappedMintResidueFusion(structural_dim=2, mint_dim=3)
    with pytest.raises(ValueError, match="padded structural residue"):
        fusion(
            structural_residue_features=torch.zeros(1, 2, 2),
            mint_residue_features=torch.zeros(1, 2, 3),
            residue_mask=torch.tensor([[True, False]]),
            mint_mapped_mask=torch.tensor([[True, True]]),
        )


def test_row_alignment_is_one_to_one_and_preserves_requested_order():
    observed = aligned_row_indices(
        ["pair-c", "pair-a"], ["pair-a", "pair-b", "pair-c"]
    )
    np.testing.assert_array_equal(observed, np.asarray([2, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="missing 1"):
        aligned_row_indices(["pair-x"], ["pair-a"])
    with pytest.raises(ValueError, match="not unique"):
        aligned_row_indices(["pair-a", "pair-a"], ["pair-a"])


def test_designed_site_contract_maps_to_exact_mint_sequence_positions():
    observed = [
        (address.name, address.chain_id, address.sequence_position)
        for address in LIBB_DESIGNED_SITE_MINT_ADDRESSES
    ]
    assert observed == [
        ("pep4", 0, 265),
        ("pep5", 0, 266),
        ("aff6", 1, 6),
        ("aff10", 1, 10),
        ("aff13", 1, 13),
        ("aff14", 1, 14),
        ("aff17", 1, 17),
    ]


def test_gather_uses_sequence_order_and_marks_unmapped_residues():
    chain0 = torch.zeros(2, 270, MINT_RESIDUE_DIM)
    chain1 = torch.zeros(2, 58, MINT_RESIDUE_DIM)
    chain0[:, :, 0] = torch.arange(1, 271)
    chain1[:, :, 0] = 1000 + torch.arange(1, 59)
    addresses = (
        LIBB_DESIGNED_SITE_MINT_ADDRESSES[0],
        MintResidueAddress("beta2m-unmapped", None, None),
        LIBB_DESIGNED_SITE_MINT_ADDRESSES[2],
    )

    values, mask = gather_mint_residue_features(chain0, chain1, addresses)

    assert values.shape == (2, 3, MINT_RESIDUE_DIM)
    assert mask.tolist() == [[True, False, True], [True, False, True]]
    torch.testing.assert_close(values[:, 0, 0], torch.tensor([265.0, 265.0]))
    torch.testing.assert_close(values[:, 1], torch.zeros(2, MINT_RESIDUE_DIM))
    torch.testing.assert_close(values[:, 2, 0], torch.tensor([1006.0, 1006.0]))
