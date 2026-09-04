#!/usr/bin/env python3
"""Fixed-crystal LibB adapter for the official RDE-PPI encoders.

RDE-PPI was trained to model side-chain rotamers and, in its supervised form,
mutation-induced binding-energy changes.  The Affibody experiment instead
needs an *absolute current-pair representation*.  This module therefore only
prepares one current LibB peptide--Affibody pair at a time and exposes inputs
for ``CircularSplineRotamerDensityEstimator.encode`` and
``DDG_RDE_Network.encode``.  It never constructs a WT-minus-mutant feature and
does not contain a binding or retention target.

The experimentally observed A/B/P/H crystal copy supplies fixed backbone
geometry.  For every sequence variant:

* peptide positions 4/5 and displayed 58-aa Affibody positions
  6/10/13/14/17 receive the current row's amino-acid identities;
* crystallized C-beta and chi-angle information is hidden at all seven sites;
* a deterministic 128-residue patch is selected around those sites using the
  masked geometry; and
* all other residue identities and coordinates stay fixed.

The official RDE-PPI package is imported only by the production extractor.
The model-independent validation and tensor preparation here remain directly
unit-testable.
"""

from __future__ import absolute_import, division, print_function

import copy
import hashlib
import re
from dataclasses import dataclass

import torch


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {residue: index for index, residue in enumerate(AA_ALPHABET)}

HLA_CHAIN = "A"
BETA2M_CHAIN = "B"
AFFIBODY_CHAIN = "H"
PEPTIDE_CHAIN = "P"
CRYSTAL_CHAIN_IDS = (HLA_CHAIN, BETA2M_CHAIN, AFFIBODY_CHAIN, PEPTIDE_CHAIN)

PEPTIDE_LENGTH = 9
AFFIBODY_LENGTH = 58
PEPTIDE_DESIGN_POSITIONS = (4, 5)
AFFIBODY_DESIGN_POSITIONS = (6, 10, 13, 14, 17)
# Xinyu's revised provider-facing labels include two preceding residues (MA)
# that are not present in the displayed/modelled 58-aa sequence.  These are
# names for the same physical sites, not indices into ``chain2_sequence``.
AFFIBODY_CRYSTAL_ALIGNED_LABELS = (8, 12, 15, 16, 19)
AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED = dict(
    zip(AFFIBODY_DESIGN_POSITIONS, AFFIBODY_CRYSTAL_ALIGNED_LABELS)
)
PATCH_SIZE = 128
CB_ATOM_INDEX = 4
CA_ATOM_INDEX = 1
HLA_ASSAY_IDENTITY_CORRECTIONS = ((84, "Y", "A"), (167, "W", "A"))
ASSAY_HLA_CHAIN1_START_0_BASED = 70
ASSAY_HLA_RESOLVED_LENGTH = 181
FIXED_CRYSTAL_APPROXIMATION_OMISSIONS = (
    "SMART domain (canonical chain1 residues 1-70)",
    "assay intrachain linker (canonical chain1 residues 252-261)",
    "HLA segment after assay-local position 181",
    "beta2m chain B (outside the selected 128-residue patch)",
)

CANONICAL_ROW_KEYS = frozenset(
    {
        "row_index",
        "row_id",
        "split",
        "chain1_sequence",
        "chain2_sequence",
        "sequence_pair_sha256",
    }
)
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
LABEL_LIKE_PATTERN = re.compile(
    r"(?:retention|binder|weak[_-]?label|ddg|delta[_-]?g|enrichment|"
    r"(?:^|_)target(?:_|$)|(?:^|_)label(?:_|$)|r0*(?:1|9|10).*count)",
    re.IGNORECASE,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _reject_label_like_keys(value, location="$"):
    """Reject outcome/training-label material from extractor-facing input."""
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                LABEL_LIKE_PATTERN.search(str(key)) is None,
                "label-like field {!r} is forbidden at {}".format(key, location),
            )
            _reject_label_like_keys(child, "{}.{}".format(location, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_label_like_keys(child, "{}[{}]".format(location, index))


@dataclass(frozen=True)
class CanonicalRow:
    row_index: int
    row_id: str
    split: str
    chain1_sequence: str
    chain2_sequence: str
    sequence_pair_sha256: str

    @property
    def peptide_sequence(self):
        return self.chain1_sequence[-PEPTIDE_LENGTH:]


def canonical_row_from_mapping(record):
    _require(isinstance(record, dict), "canonical row must be an object")
    _require(
        set(record) == set(CANONICAL_ROW_KEYS),
        "canonical row fields changed: {}".format(sorted(record)),
    )
    try:
        row_index = int(record["row_index"])
    except (TypeError, ValueError):
        raise ValueError("row_index must be an integer")
    row = CanonicalRow(
        row_index=row_index,
        row_id=str(record["row_id"]),
        split=str(record["split"]),
        chain1_sequence=str(record["chain1_sequence"]),
        chain2_sequence=str(record["chain2_sequence"]),
        sequence_pair_sha256=str(record["sequence_pair_sha256"]),
    )
    _require(row.row_index >= 0, "negative row_index")
    _require(row.row_id, "empty row_id")
    _require(row.split in {"train", "eval"}, "unexpected split")
    _require(len(row.chain1_sequence) == 270, "chain 1 length changed")
    _require(len(row.chain2_sequence) == AFFIBODY_LENGTH, "Affibody length changed")
    _require(AA_PATTERN.fullmatch(row.chain1_sequence) is not None, "invalid chain 1")
    _require(AA_PATTERN.fullmatch(row.chain2_sequence) is not None, "invalid Affibody")
    _require(len(row.sequence_pair_sha256) == 64, "invalid sequence-pair digest")
    _require(
        sha256_text(row.chain1_sequence + "|" + row.chain2_sequence)
        == row.sequence_pair_sha256,
        "sequence-pair digest mismatch",
    )
    return row


def validate_canonical_payload(payload, expected_total=None):
    """Validate the complete label-free canonical rows payload."""
    _reject_label_like_keys(payload)
    _require(isinstance(payload, dict), "row manifest must be an object")
    _require(set(payload) == {"schema_version", "rows"}, "row manifest fields changed")
    _require(
        payload["schema_version"] == "esmfold2-libb-canonical-rows-v1",
        "unexpected canonical-row schema",
    )
    _require(isinstance(payload["rows"], list), "rows must be a list")
    rows = [canonical_row_from_mapping(record) for record in payload["rows"]]
    if expected_total is not None:
        _require(len(rows) == int(expected_total), "canonical row count changed")
    _require(
        [row.row_index for row in rows] == list(range(len(rows))),
        "canonical rows are not in contiguous row_index order",
    )
    _require(len({row.row_id for row in rows}) == len(rows), "duplicate row_id")
    _require(
        len({row.sequence_pair_sha256 for row in rows}) == len(rows),
        "duplicate sequence pair",
    )
    _require(rows, "canonical row set is empty")
    return rows


def align_observed_subsequence_masking_design(
    full_sequence, observed_sequence, designed_positions_1_based
):
    """Map a resolved crystal subsequence while ignoring designed identities.

    The LibB Affibody crystal lacks terminal residues.  Ordinary substring
    matching cannot align it to arbitrary variants because five internal
    positions change.  This routine finds the unique contiguous offset for
    which every *non-designed* resolved identity agrees.
    """
    _require(full_sequence, "empty full sequence")
    _require(observed_sequence, "empty observed sequence")
    _require(len(observed_sequence) <= len(full_sequence), "observed sequence is longer")
    designed = set(int(position) for position in designed_positions_1_based)
    matches = []
    for start in range(len(full_sequence) - len(observed_sequence) + 1):
        agrees = True
        for ordinal, residue in enumerate(observed_sequence):
            full_position = start + ordinal + 1
            if full_position in designed:
                continue
            if residue != full_sequence[start + ordinal]:
                agrees = False
                break
        if agrees:
            matches.append(start)
    _require(len(matches) == 1, "crystal subsequence alignment is not unique")
    start = matches[0]
    return {
        ordinal: start + ordinal + 1 for ordinal in range(len(observed_sequence))
    }


def _clone_data(data):
    output = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.clone()
        else:
            output[key] = copy.deepcopy(value)
    return output


def _index_data(data, index):
    """Apply one residue index to every residue-aligned field."""
    n = int(data["aa"].shape[0])
    output = {}
    index_list = index.tolist()
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == n:
            output[key] = value[index]
        elif isinstance(value, list) and len(value) == n:
            output[key] = [value[position] for position in index_list]
        else:
            output[key] = copy.deepcopy(value)
    return output


def _sequence_from_aa_indices(indices):
    output = []
    for value in indices.tolist():
        _require(0 <= int(value) < len(AA_ALPHABET), "non-canonical residue in crystal")
        output.append(AA_ALPHABET[int(value)])
    return "".join(output)


def _chain_indices(data, chain_id):
    return [index for index, value in enumerate(data["chain_id"]) if value == chain_id]


def resolve_crystal_mappings(data, reference_row):
    """Resolve chain-sequence and seven designed-site indices by sequence."""
    required = {
        "aa",
        "chain_id",
        "resseq",
        "icode",
        "chain_nb",
        "res_nb",
        "pos_heavyatom",
        "mask_heavyatom",
        "phi",
        "phi_mask",
        "psi",
        "psi_mask",
        "chi",
        "chi_alt",
        "chi_mask",
        "chi_complete",
    }
    _require(required.issubset(data), "parsed crystal lacks required RDE fields")
    n = int(data["aa"].shape[0])
    _require(len(data["chain_id"]) == n, "chain IDs are not residue-aligned")
    _require(
        set(data["chain_id"]) == set(CRYSTAL_CHAIN_IDS),
        "parsed copy must contain exactly chains A/B/H/P",
    )

    peptide_indices = _chain_indices(data, PEPTIDE_CHAIN)
    affibody_indices = _chain_indices(data, AFFIBODY_CHAIN)
    _require(len(peptide_indices) == PEPTIDE_LENGTH, "crystal peptide is not complete")
    _require(len(affibody_indices) >= 30, "too few resolved Affibody residues")

    peptide_observed = _sequence_from_aa_indices(data["aa"][peptide_indices])
    affibody_observed = _sequence_from_aa_indices(data["aa"][affibody_indices])
    peptide_reference = reference_row.peptide_sequence
    _require(len(peptide_reference) == PEPTIDE_LENGTH, "reference peptide length changed")
    for position, observed in enumerate(peptide_observed, 1):
        if position not in PEPTIDE_DESIGN_POSITIONS:
            _require(
                observed == peptide_reference[position - 1],
                "non-designed peptide identity disagrees with crystal",
            )

    affibody_ordinal_to_full = align_observed_subsequence_masking_design(
        reference_row.chain2_sequence,
        affibody_observed,
        AFFIBODY_DESIGN_POSITIONS,
    )
    peptide_full_to_raw = {
        ordinal + 1: raw_index for ordinal, raw_index in enumerate(peptide_indices)
    }
    affibody_full_to_raw = {
        affibody_ordinal_to_full[ordinal]: raw_index
        for ordinal, raw_index in enumerate(affibody_indices)
    }
    _require(
        set(PEPTIDE_DESIGN_POSITIONS).issubset(peptide_full_to_raw),
        "designed peptide site is unresolved",
    )
    _require(
        set(AFFIBODY_DESIGN_POSITIONS).issubset(affibody_full_to_raw),
        "designed Affibody site is unresolved",
    )
    # The sequence-derived alignment must independently reproduce the revised
    # provider numbering.  We deliberately do not use the revised labels as
    # Python indices: displayed position p maps to PDB H:(p + 2).
    for displayed_position, crystal_aligned_label in (
        AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED.items()
    ):
        raw_index = affibody_full_to_raw[displayed_position]
        _require(
            data["chain_id"][raw_index] == AFFIBODY_CHAIN
            and int(data["resseq"][raw_index]) == crystal_aligned_label
            and str(data["icode"][raw_index]).strip() == "",
            "sequence-derived Affibody mapping disagrees with revised numbering",
        )
    designed_raw = tuple(
        [peptide_full_to_raw[position] for position in PEPTIDE_DESIGN_POSITIONS]
        + [affibody_full_to_raw[position] for position in AFFIBODY_DESIGN_POSITIONS]
    )
    _require(len(set(designed_raw)) == 7, "designed indices are not unique")
    return {
        "peptide_full_to_raw": peptide_full_to_raw,
        "affibody_full_to_raw": affibody_full_to_raw,
        "designed_raw_indices": designed_raw,
        "peptide_observed_sequence": peptide_observed,
        "affibody_observed_sequence": affibody_observed,
    }


def select_backbone_and_cb(data):
    """Mirror the official RDE ``SelectAtom('backbone+CB')`` transform."""
    output = _clone_data(data)
    _require(output["pos_heavyatom"].shape[1] >= 5, "crystal lacks backbone+CB slots")
    output["pos_atoms"] = output["pos_heavyatom"][:, :5].clone()
    output["mask_atoms"] = output["mask_heavyatom"][:, :5].clone()
    if "bfactor_heavyatom" in output:
        output["bfactor_atoms"] = output["bfactor_heavyatom"][:, :5].clone()
    return output


def validate_assay_hla_identity_mapping(data, reference_row):
    """Prove that the assay/PDB HLA mismatch is exactly Y84A and W167A."""
    assay = reference_row.chain1_sequence[
        ASSAY_HLA_CHAIN1_START_0_BASED :
        ASSAY_HLA_CHAIN1_START_0_BASED + ASSAY_HLA_RESOLVED_LENGTH
    ]
    _require(len(assay) == ASSAY_HLA_RESOLVED_LENGTH, "assay HLA segment is incomplete")
    by_number = {}
    for index, (chain_id, resseq, icode) in enumerate(
        zip(data["chain_id"], data["resseq"], data["icode"])
    ):
        if chain_id == HLA_CHAIN and str(icode).strip() == "":
            by_number[int(resseq)] = index
    _require(
        set(range(1, ASSAY_HLA_RESOLVED_LENGTH + 1)).issubset(by_number),
        "crystal HLA does not resolve positions 1..181",
    )
    mismatches = []
    for position in range(1, ASSAY_HLA_RESOLVED_LENGTH + 1):
        crystal_aa = AA_ALPHABET[int(data["aa"][by_number[position]])]
        assay_aa = assay[position - 1]
        if crystal_aa != assay_aa:
            mismatches.append((position, crystal_aa, assay_aa))
    _require(
        tuple(mismatches) == HLA_ASSAY_IDENTITY_CORRECTIONS,
        "assay-to-crystal HLA mismatch set changed: {}".format(mismatches),
    )
    return tuple(mismatches)


def apply_assay_hla_identity_corrections(data):
    """Match two constant HLA sites to the assayed construct sequence.

    Crystal chain A has Y84 and W167, whereas the assayed SMART--HLA
    construct has alanine at both positions.  Coordinates remain the fixed
    crystal approximation.  Ala retains the backbone and C-beta coordinate,
    but all reference aromatic chi information is removed.
    """
    output = _clone_data(data)
    records = []
    for residue_number, crystal_aa, assay_aa in HLA_ASSAY_IDENTITY_CORRECTIONS:
        matches = [
            index
            for index, (chain_id, resseq, icode) in enumerate(
                zip(output["chain_id"], output["resseq"], output["icode"])
            )
            if chain_id == HLA_CHAIN
            and int(resseq) == residue_number
            and str(icode).strip() == ""
        ]
        _require(
            len(matches) == 1,
            "expected exactly one crystal HLA residue {}".format(residue_number),
        )
        index = matches[0]
        _require(
            int(output["aa"][index]) == AA_TO_INDEX[crystal_aa],
            "unexpected crystal identity at HLA {}".format(residue_number),
        )
        _require(assay_aa == "A", "only the audited alanine corrections are supported")
        output["aa"][index] = AA_TO_INDEX[assay_aa]
        output["chi"][index] = 0.0
        output["chi_alt"][index] = 0.0
        output["chi_mask"][index] = False
        output["chi_complete"][index] = False
        records.append(
            {
                "chain_id": HLA_CHAIN,
                "pdb_residue_number": residue_number,
                "crystal_amino_acid": crystal_aa,
                "assay_amino_acid": assay_aa,
                "backbone_retained": True,
                "cb_retained_as_alanine_approximation": True,
                "chi_hidden": True,
            }
        )
    return output, tuple(records)


def hide_designed_sidechains(data, designed_indices):
    """Hide reference C-beta/chi information while preserving backbone atoms."""
    output = _clone_data(data)
    index = torch.as_tensor(designed_indices, dtype=torch.long)
    _require(index.numel() == 7, "exactly seven designed residues are required")
    _require(
        output["pos_atoms"].shape[1:] == (5, 3)
        and output["mask_atoms"].shape[1:] == (5,),
        "RDE model input must expose only N/CA/C/O/CB atom slots",
    )
    _require(bool(output["mask_atoms"][index, CA_ATOM_INDEX].all()), "designed CA missing")
    output["pos_atoms"][index, CB_ATOM_INDEX] = 0.0
    output["mask_atoms"][index, CB_ATOM_INDEX] = False
    if "bfactor_atoms" in output:
        output["bfactor_atoms"][index, CB_ATOM_INDEX] = 0.0
    output["chi"][index] = 0.0
    output["chi_alt"][index] = 0.0
    output["chi_mask"][index] = False
    output["chi_complete"][index] = False
    designed_flag = torch.zeros(output["aa"].shape[0], dtype=torch.bool)
    designed_flag[index] = True
    output["mut_flag"] = designed_flag.clone()
    output["designed_flag"] = designed_flag
    _require(
        not bool(output["mask_atoms"][index, CB_ATOM_INDEX].any())
        and bool((output["pos_atoms"][index, CB_ATOM_INDEX] == 0).all()),
        "designed C-beta information was not fully hidden",
    )
    _require(
        not bool(output["chi_mask"][index].any())
        and not bool(output["chi_complete"][index].any())
        and bool((output["chi"][index] == 0).all())
        and bool((output["chi_alt"][index] == 0).all()),
        "designed chi information was not fully hidden",
    )
    return output


def fixed_patch_indices(data, designed_indices, patch_size=PATCH_SIZE):
    """Choose a stable patch using CA at every side-chain-masked design site."""
    n = int(data["aa"].shape[0])
    _require(1 <= int(patch_size) <= n, "invalid patch size")
    designed = torch.as_tensor(designed_indices, dtype=torch.long)
    _require(bool(data["mask_atoms"][:, CA_ATOM_INDEX].all()), "patch contains missing CA")
    representative = data["pos_atoms"][:, CA_ATOM_INDEX].clone()
    has_cb = data["mask_atoms"][:, CB_ATOM_INDEX]
    representative[has_cb] = data["pos_atoms"][has_cb, CB_ATOM_INDEX]
    seeds = data["pos_atoms"][designed, CA_ATOM_INDEX]
    distance = torch.cdist(representative, seeds).min(dim=1)[0]
    # Python sorting gives an explicit index tie-break independent of torch's
    # sort-stability behavior across supported versions.
    nearest = sorted(range(n), key=lambda index: (float(distance[index]), index))[
        : int(patch_size)
    ]
    # Keep nearest-first order, matching the official
    # SelectedRegionFixedSizePatch transform.  Downstream mappings are rebuilt
    # after this reordering and never assume biological sequence order.
    return torch.tensor(nearest, dtype=torch.long)


@dataclass
class FixedCrystalTemplate:
    data: dict
    peptide_full_to_patch: dict
    affibody_full_to_patch: dict
    designed_patch_indices: tuple
    reference_peptide_sequence: str
    reference_affibody_sequence: str
    patch_raw_indices: tuple
    hla_identity_corrections: tuple

    @property
    def patch_size(self):
        return int(self.data["aa"].shape[0])


def build_fixed_crystal_template(parsed_data, reference_row, patch_size=PATCH_SIZE):
    """Create the one fixed, side-chain-masked crystal patch used by all rows."""
    validate_assay_hla_identity_mapping(parsed_data, reference_row)
    corrected, hla_corrections = apply_assay_hla_identity_corrections(parsed_data)
    mapping = resolve_crystal_mappings(corrected, reference_row)
    selected = select_backbone_and_cb(corrected)
    selected = hide_designed_sidechains(selected, mapping["designed_raw_indices"])
    patch_index = fixed_patch_indices(
        selected, mapping["designed_raw_indices"], patch_size=patch_size
    )
    patched = _index_data(selected, patch_index)
    raw_to_patch = {int(raw): index for index, raw in enumerate(patch_index.tolist())}
    designed_patch = tuple(raw_to_patch[index] for index in mapping["designed_raw_indices"])
    _require(len(set(designed_patch)) == 7, "patch omitted a designed residue")
    peptide_map = {
        full: raw_to_patch[raw]
        for full, raw in mapping["peptide_full_to_raw"].items()
        if raw in raw_to_patch
    }
    affibody_map = {
        full: raw_to_patch[raw]
        for full, raw in mapping["affibody_full_to_raw"].items()
        if raw in raw_to_patch
    }
    _require(
        set(PEPTIDE_DESIGN_POSITIONS).issubset(peptide_map),
        "patch omitted designed peptide residue",
    )
    _require(
        set(AFFIBODY_DESIGN_POSITIONS).issubset(affibody_map),
        "patch omitted designed Affibody residue",
    )
    _require(
        BETA2M_CHAIN not in patched["chain_id"],
        "fixed patch unexpectedly includes beta2m rather than the assay construct",
    )
    _require(
        all(
            chain_id != HLA_CHAIN or int(resseq) <= ASSAY_HLA_RESOLVED_LENGTH
            for chain_id, resseq in zip(patched["chain_id"], patched["resseq"])
        ),
        "fixed patch includes HLA residues absent from the assayed construct",
    )
    return FixedCrystalTemplate(
        data=patched,
        peptide_full_to_patch=peptide_map,
        affibody_full_to_patch=affibody_map,
        designed_patch_indices=designed_patch,
        reference_peptide_sequence=reference_row.peptide_sequence,
        reference_affibody_sequence=reference_row.chain2_sequence,
        patch_raw_indices=tuple(int(value) for value in patch_index.tolist()),
        hla_identity_corrections=hla_corrections,
    )


def validate_variant_against_template(row, template):
    """Require that only the seven intended sequence positions differ."""
    peptide = row.peptide_sequence
    affibody = row.chain2_sequence
    for position, (reference, current) in enumerate(
        zip(template.reference_peptide_sequence, peptide), 1
    ):
        if position not in PEPTIDE_DESIGN_POSITIONS:
            _require(reference == current, "non-designed peptide position changed")
    for position, (reference, current) in enumerate(
        zip(template.reference_affibody_sequence, affibody), 1
    ):
        if position not in AFFIBODY_DESIGN_POSITIONS:
            _require(reference == current, "non-designed Affibody position changed")


def prepare_current_pair(template, row):
    """Apply one row's seven identities to the fixed masked crystal patch."""
    validate_variant_against_template(row, template)
    data = _clone_data(template.data)
    peptide = row.peptide_sequence
    affibody = row.chain2_sequence
    for position in PEPTIDE_DESIGN_POSITIONS:
        data["aa"][template.peptide_full_to_patch[position]] = AA_TO_INDEX[
            peptide[position - 1]
        ]
    for position in AFFIBODY_DESIGN_POSITIONS:
        data["aa"][template.affibody_full_to_patch[position]] = AA_TO_INDEX[
            affibody[position - 1]
        ]
    designed = torch.zeros(template.patch_size, dtype=torch.bool)
    designed[list(template.designed_patch_indices)] = True
    _require(torch.equal(designed, data["designed_flag"]), "designed mask drifted")
    _require(
        not bool(data["mask_atoms"][designed, CB_ATOM_INDEX].any()),
        "reference C-beta leaked at a designed site",
    )
    _require(
        not bool(data["chi_mask"][designed].any()),
        "reference chi angle leaked at a designed site",
    )
    _require(
        bool((data["pos_atoms"][designed, CB_ATOM_INDEX] == 0).all())
        and bool((data["chi"][designed] == 0).all())
        and bool((data["chi_alt"][designed] == 0).all())
        and not bool(data["chi_complete"][designed].any()),
        "reference side-chain values leaked at a designed site",
    )
    data["chi_corrupt"] = data["chi"].clone()
    data["chi_masked_flag"] = designed.clone()
    data["residue_flag"] = data["mask_atoms"][:, CA_ATOM_INDEX].bool()
    data["peptide_flag"] = torch.tensor(
        [chain == PEPTIDE_CHAIN for chain in data["chain_id"]], dtype=torch.bool
    )
    data["affibody_flag"] = torch.tensor(
        [chain == AFFIBODY_CHAIN for chain in data["chain_id"]], dtype=torch.bool
    )
    data["ligand_flag"] = data["affibody_flag"].clone()
    data["row_index"] = row.row_index
    data["row_id"] = row.row_id
    data["split"] = row.split
    data["sequence_pair_sha256"] = row.sequence_pair_sha256
    return data


MODEL_INPUT_KEYS = (
    "aa",
    "chain_nb",
    "res_nb",
    "pos_atoms",
    "mask_atoms",
    "phi",
    "phi_mask",
    "psi",
    "psi_mask",
    "chi",
    "chi_alt",
    "chi_mask",
    "chi_complete",
    "chi_corrupt",
    "chi_masked_flag",
    "mut_flag",
    "designed_flag",
    "residue_flag",
    "peptide_flag",
    "affibody_flag",
    "ligand_flag",
)


def collate_current_pairs(examples):
    """Stack fixed-length current-pair examples for RDE encoder inference."""
    _require(examples, "cannot collate an empty batch")
    length = int(examples[0]["aa"].shape[0])
    output = {}
    for key in MODEL_INPUT_KEYS:
        _require(key in examples[0], "example lacks {}".format(key))
        values = []
        for example in examples:
            _require(int(example["aa"].shape[0]) == length, "patch length changed")
            _require(key in example, "example lacks {}".format(key))
            values.append(example[key])
        output[key] = torch.stack(values, dim=0)
    output["row_index"] = torch.tensor(
        [int(example["row_index"]) for example in examples], dtype=torch.long
    )
    output["row_id"] = [str(example["row_id"]) for example in examples]
    output["split"] = [str(example["split"]) for example in examples]
    output["sequence_pair_sha256"] = [
        str(example["sequence_pair_sha256"]) for example in examples
    ]
    return output


def model_batch(batch, device):
    """Move only model-consumed tensors, retaining explicit boolean masks."""
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and key in MODEL_INPUT_KEYS
    }


def feature_masks(batch):
    return {
        "residue_mask": batch["residue_flag"].detach().cpu(),
        "peptide_mask": batch["peptide_flag"].detach().cpu(),
        "affibody_mask": batch["affibody_flag"].detach().cpu(),
        "designed_mask": batch["designed_flag"].detach().cpu(),
        "chain_nb": batch["chain_nb"].detach().cpu(),
        "aa": batch["aa"].detach().cpu(),
    }
