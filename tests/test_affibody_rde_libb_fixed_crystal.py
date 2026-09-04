import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from downstream.AffibodyMHC import extract_rde_libb_features as extractor
from downstream.AffibodyMHC import merge_rde_libb_features as merger
from downstream.AffibodyMHC import rde_libb_fixed_crystal as crystal


AFFIBODY_TEMPLATE = (crystal.AA_ALPHABET * 3)[:58]


def test_provider_revision_uses_the_complete_30648_plus_120_roster():
    assert extractor.EXPECTED_ROWS == 30_768
    assert (
        extractor.DEFAULT_STRUCTURE_CONTRACT_DIR.name
        == "libb_fixed_crystal_contract_provider_revision_120_v1"
    )
    assert crystal.AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED == {
        6: 8,
        10: 12,
        13: 15,
        14: 16,
        17: 19,
    }


def _hash_pair(chain1, chain2):
    return hashlib.sha256((chain1 + "|" + chain2).encode("ascii")).hexdigest()


def _row(index=0, split="train", peptide="AAAAAAAAA", affibody=AFFIBODY_TEMPLATE):
    chain1 = "A" * 261 + peptide
    return crystal.CanonicalRow(
        row_index=index,
        row_id="row-{}".format(index),
        split=split,
        chain1_sequence=chain1,
        chain2_sequence=affibody,
        sequence_pair_sha256=_hash_pair(chain1, affibody),
    )


def _parsed_crystal():
    # The H chain represents full Affibody positions 3..57.  H plus P contains
    # exactly 64 residues, so a 64-residue patch centered on the designed sites
    # excludes the deliberately distant A/B residues.
    chain_id = ["A"] * 181 + ["B"] * 5 + ["H"] * 55 + ["P"] * 9
    n = len(chain_id)
    aa = torch.zeros(n, dtype=torch.long)
    aa[83] = crystal.AA_TO_INDEX["Y"]
    aa[166] = crystal.AA_TO_INDEX["W"]
    h_start = 186
    aa[h_start : h_start + 55] = torch.tensor(
        [crystal.AA_TO_INDEX[value] for value in AFFIBODY_TEMPLATE[2:57]],
        dtype=torch.long,
    )
    chain_nb = torch.tensor([0] * 181 + [1] * 5 + [2] * 55 + [3] * 9)
    resseq = torch.tensor(
        list(range(1, 182))
        + list(range(1, 6))
        + list(range(5, 60))
        + list(range(1, 10))
    )
    res_nb = torch.tensor(
        list(range(1, 182))
        + list(range(1, 6))
        + list(range(1, 56))
        + list(range(1, 10))
    )
    coordinates = torch.zeros(n, 15, 3)
    # Put A/B far away and all H/P positions near the seven design seeds.
    for index, chain in enumerate(chain_id):
        base = float(index if chain in {"H", "P"} else 10_000 + index)
        coordinates[index, :, 0] = base
        coordinates[index, :, 1] = torch.arange(15, dtype=torch.float32) * 0.01
    masks = torch.ones(n, 15, dtype=torch.bool)
    return {
        "chain_id": chain_id,
        "chain_nb": chain_nb,
        "resseq": resseq,
        "icode": [" "] * n,
        "res_nb": res_nb,
        "aa": aa,
        "pos_heavyatom": coordinates,
        "mask_heavyatom": masks,
        "bfactor_heavyatom": torch.ones(n, 15),
        "phi": torch.zeros(n),
        "phi_mask": torch.ones(n, dtype=torch.bool),
        "psi": torch.zeros(n),
        "psi_mask": torch.ones(n, dtype=torch.bool),
        "chi": torch.ones(n, 4),
        "chi_alt": torch.full((n, 4), 2.0),
        "chi_mask": torch.ones(n, 4, dtype=torch.bool),
        "chi_complete": torch.ones(n, dtype=torch.bool),
    }


def test_canonical_rows_are_hash_bound_and_label_free():
    first = _row(0, "train")
    second = _row(1, "eval", peptide="AAACDAAAA")
    payload = {
        "schema_version": "esmfold2-libb-canonical-rows-v1",
        "rows": [first.__dict__, second.__dict__],
    }
    rows = crystal.validate_canonical_payload(payload, expected_total=2)
    assert [row.row_id for row in rows] == ["row-0", "row-1"]

    poisoned = json.loads(json.dumps(payload))
    poisoned["rows"][0]["target_retention"] = 92.0
    with pytest.raises(ValueError, match="label-like"):
        crystal.validate_canonical_payload(poisoned)

    corrupted = json.loads(json.dumps(payload))
    corrupted["rows"][0]["sequence_pair_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        crystal.validate_canonical_payload(corrupted)


def test_alignment_ignores_only_designed_residue_identities():
    full = list(AFFIBODY_TEMPLATE)
    observed = list(full[2:57])
    for position, residue in zip(crystal.AFFIBODY_DESIGN_POSITIONS, "CDEFG"):
        observed[position - 3] = residue
    mapping = crystal.align_observed_subsequence_masking_design(
        "".join(full), "".join(observed), crystal.AFFIBODY_DESIGN_POSITIONS
    )
    assert mapping[0] == 3
    assert mapping[54] == 57

    observed[20] = "Y" if observed[20] != "Y" else "A"
    with pytest.raises(ValueError, match="not unique"):
        crystal.align_observed_subsequence_masking_design(
            "".join(full), "".join(observed), crystal.AFFIBODY_DESIGN_POSITIONS
        )


def test_fixed_template_hides_every_model_visible_sidechain_field():
    parsed = _parsed_crystal()
    reference = _row()
    resolved = crystal.resolve_crystal_mappings(parsed, reference)
    for displayed_position, revised_label in zip(
        crystal.AFFIBODY_DESIGN_POSITIONS,
        crystal.AFFIBODY_CRYSTAL_ALIGNED_LABELS,
    ):
        raw_index = resolved["affibody_full_to_raw"][displayed_position]
        assert parsed["chain_id"][raw_index] == "H"
        assert parsed["resseq"][raw_index].item() == revised_label
    template = crystal.build_fixed_crystal_template(parsed, reference, patch_size=64)
    assert template.patch_size == 64
    assert set(template.data["chain_id"]) == {"H", "P"}
    assert len(template.designed_patch_indices) == 7
    assert template.hla_identity_corrections == (
        {
            "chain_id": "A",
            "pdb_residue_number": 84,
            "crystal_amino_acid": "Y",
            "assay_amino_acid": "A",
            "backbone_retained": True,
            "cb_retained_as_alanine_approximation": True,
            "chi_hidden": True,
        },
        {
            "chain_id": "A",
            "pdb_residue_number": 167,
            "crystal_amino_acid": "W",
            "assay_amino_acid": "A",
            "backbone_retained": True,
            "cb_retained_as_alanine_approximation": True,
            "chi_hidden": True,
        },
    )

    designed = template.data["designed_flag"]
    # RDE receives exactly N/CA/C/O/CB.  CB and every chi-bearing field are
    # absent/zero at all seven sites; no higher side-chain atom is model-visible.
    assert template.data["pos_atoms"].shape == (64, 5, 3)
    assert not template.data["mask_atoms"][designed, crystal.CB_ATOM_INDEX].any()
    assert torch.count_nonzero(
        template.data["pos_atoms"][designed, crystal.CB_ATOM_INDEX]
    ) == 0
    assert not template.data["chi_mask"][designed].any()
    assert not template.data["chi_complete"][designed].any()
    assert torch.count_nonzero(template.data["chi"][designed]) == 0
    assert torch.count_nonzero(template.data["chi_alt"][designed]) == 0

    affibody = list(AFFIBODY_TEMPLATE)
    for position, residue in zip(crystal.AFFIBODY_DESIGN_POSITIONS, "CDEFG"):
        affibody[position - 1] = residue
    peptide = list("A" * 9)
    peptide[3:5] = list("WY")
    variant = _row(1, peptide="".join(peptide), affibody="".join(affibody))
    prepared = crystal.prepare_current_pair(template, variant)
    assert prepared["aa"][template.peptide_full_to_patch[4]].item() == crystal.AA_TO_INDEX["W"]
    assert prepared["aa"][template.peptide_full_to_patch[5]].item() == crystal.AA_TO_INDEX["Y"]
    observed_affibody = [
        prepared["aa"][template.affibody_full_to_patch[position]].item()
        for position in crystal.AFFIBODY_DESIGN_POSITIONS
    ]
    assert observed_affibody == [crystal.AA_TO_INDEX[value] for value in "CDEFG"]
    assert not prepared["mask_atoms"][prepared["designed_flag"], 4].any()
    assert "pos_heavyatom" not in crystal.model_batch(
        crystal.collate_current_pairs([prepared]), torch.device("cpu")
    )


def test_revised_affibody_labels_are_validated_as_pdb_ids_not_sequence_indices():
    parsed = _parsed_crystal()
    # Keep the displayed sequence identical but corrupt the PDB numbering.
    h_start = parsed["chain_id"].index("H")
    parsed["resseq"][h_start : h_start + 55] -= 2
    with pytest.raises(ValueError, match="revised numbering"):
        crystal.resolve_crystal_mappings(parsed, _row())


def test_hla_graft_requires_exactly_the_two_audited_assay_differences():
    parsed = _parsed_crystal()
    assert crystal.validate_assay_hla_identity_mapping(parsed, _row()) == (
        (84, "Y", "A"),
        (167, "W", "A"),
    )
    parsed["aa"][99] = crystal.AA_TO_INDEX["C"]
    with pytest.raises(ValueError, match="mismatch set changed"):
        crystal.validate_assay_hla_identity_mapping(parsed, _row())


def test_hla_graft_keeps_backbone_and_cb_but_removes_incompatible_chi():
    parsed = _parsed_crystal()
    original_coordinates = parsed["pos_heavyatom"].clone()
    original_atom_mask = parsed["mask_heavyatom"].clone()
    corrected, records = crystal.apply_assay_hla_identity_corrections(parsed)
    assert [record["pdb_residue_number"] for record in records] == [84, 167]
    for index in (83, 166):
        assert corrected["aa"][index].item() == crystal.AA_TO_INDEX["A"]
        assert torch.equal(
            corrected["pos_heavyatom"][index], original_coordinates[index]
        )
        assert torch.equal(
            corrected["mask_heavyatom"][index], original_atom_mask[index]
        )
        assert torch.count_nonzero(corrected["chi"][index]) == 0
        assert torch.count_nonzero(corrected["chi_alt"][index]) == 0
        assert not corrected["chi_mask"][index].any()
        assert not corrected["chi_complete"][index]
    # The parsed source is never mutated in place.
    assert parsed["aa"][83].item() == crystal.AA_TO_INDEX["Y"]
    assert parsed["aa"][166].item() == crystal.AA_TO_INDEX["W"]


def test_reference_and_variant_follow_identical_mask_and_geometry_paths():
    template = crystal.build_fixed_crystal_template(_parsed_crystal(), _row(), patch_size=64)
    reference = crystal.prepare_current_pair(template, _row(0))
    variant = crystal.prepare_current_pair(
        template, _row(1, peptide="AAACDAAAA", affibody=AFFIBODY_TEMPLATE)
    )
    assert torch.equal(reference["designed_flag"], variant["designed_flag"])
    assert torch.equal(reference["pos_atoms"], variant["pos_atoms"])
    assert torch.equal(reference["mask_atoms"], variant["mask_atoms"])
    assert torch.equal(reference["chi"], variant["chi"])
    assert not torch.equal(reference["aa"], variant["aa"])


class _FakeEncoder(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def encode(self, batch):
        base = batch["aa"].float().unsqueeze(-1).expand(-1, -1, 128)
        return base + self.offset


def test_rde_network_fold_contexts_are_emitted_separately_not_averaged():
    template = crystal.build_fixed_crystal_template(_parsed_crystal(), _row(), patch_size=64)
    batch = crystal.collate_current_pairs(
        [crystal.prepare_current_pair(template, _row(0))]
    )
    arrays = extractor.encode_batch(
        _FakeEncoder(0),
        [_FakeEncoder(1), _FakeEncoder(2), _FakeEncoder(3)],
        batch,
        "both",
        torch.device("cpu"),
    )
    assert arrays["rde_context"].shape == (1, 64, 128)
    for fold in range(3):
        name = "rde_network_fold{}_context".format(fold)
        assert arrays[name].shape == (1, 64, 128)
        np.testing.assert_allclose(
            arrays[name].astype(np.float32) - arrays["rde_context"].astype(np.float32),
            fold + 1,
        )
    assert "rde_network_context" not in arrays


def test_merge_specs_exclude_identity_and_integer_diagnostic_arrays(tmp_path):
    n = 2
    arrays = {
        "row_index": np.asarray([0, 1], dtype=np.int64),
        "row_id": np.asarray(["a", "b"]),
        "split": np.asarray(["train", "eval"]),
        "sequence_pair_sha256": np.asarray(["0" * 64, "1" * 64]),
        "rde_context": np.zeros((n, crystal.PATCH_SIZE, 128), dtype=np.float16),
        "residue_mask": np.ones((n, crystal.PATCH_SIZE), dtype=bool),
        "peptide_mask": np.ones((n, crystal.PATCH_SIZE), dtype=bool),
        "affibody_mask": np.ones((n, crystal.PATCH_SIZE), dtype=bool),
        "designed_mask": np.ones((n, crystal.PATCH_SIZE), dtype=bool),
        "chain_nb": np.zeros((n, crystal.PATCH_SIZE), dtype=np.int16),
        "aa": np.zeros((n, crystal.PATCH_SIZE), dtype=np.uint8),
    }
    path = tmp_path / "chunk.npz"
    np.savez(path, **arrays)
    specs = merger._array_specs(path, "rde")
    assert set(specs) == {
        "rde_context",
        "residue_mask",
        "peptide_mask",
        "affibody_mask",
        "designed_mask",
    }
    assert specs["rde_context"]["kind"] == "feature"
    assert specs["designed_mask"]["kind"] == "mask"
    expanded = merger._with_designed_ordered_specs(specs, (4, 5, 0, 1, 2, 6, 3))
    ordered = expanded["rde_context_designed_ordered"]
    assert ordered["tail_shape"] == (7, 128)
    assert ordered["source"] == "rde_context"
    assert ordered["designed_patch_indices"] == (4, 5, 0, 1, 2, 6, 3)
    assert "designed_mask_designed_ordered" not in expanded
