#!/usr/bin/env python3
"""Frozen StaB-ddG/ProteinMPNN features for a *current* LibB pair.

This module adapts the official StaB-ddG checkpoint as a label-free feature
extractor.  It deliberately does not compute a wild-type-to-mutant difference
and it does not call the resulting features ``ddG``.  For each current
peptide--Affibody sequence it evaluates the sequence on the resolved
assay-interface portion of the fixed LibB crystal backbone both with and
without the Affibody.  The retained coordinates are HLA chain A residues
1--181, peptide chain P, and Affibody chain H.  Crystal beta-2-microglobulin and
HLA residues 182--276 are intentionally omitted because those parts are not in
the assayed SMART construct.  The primary local feature is the difference
between the complex-conditioned and isolated-fragment ProteinMPNN decoder
states at the seven designed residues.

The implementation imports the pinned upstream package lazily.  Unit tests can
therefore exercise row validation, sequence mapping, feature assembly, and
archive writing without a GPU or a vendor checkout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPO_ROOT / "private_data"

UPSTREAM_REPOSITORY = "https://github.com/LDeng0205/StaB-ddG"
UPSTREAM_COMMIT = "b19a5d72637a6b581b16f38af39383658e5ae1a5"
FINAL_CHECKPOINT_SHA256 = (
    "51a0feb6f9e4f296dbd31db2470c2b8d917d10f64f1302ae63650f43095e348f"
)
REFERENCE_PDB_SHA256 = (
    "59d026542cc42006302117cc79ad720497e69c4298f95598413f1d172d407b0e"
)
STRUCTURE_MAPPING_SCHEMA = "libb-fixed-crystal-contract-v2"
STRUCTURE_MAPPING_SHA256 = (
    "a3d08cfd4a4800d3cb2048dab420e50896f36534548988f282dfd184779c2258"
)
ROW_MANIFEST_SCHEMA = "esmfold2-libb-canonical-rows-v1"
FEATURE_ARCHIVE_SCHEMA = "libb-opaque-frozen-features-v1"
EXTRACTOR_SCHEMA = "stab-libb-current-pair-features-v4"

EXPECTED_TRAIN_ROWS = 30_648
EXPECTED_EVAL_ROWS = 120
EXPECTED_ROW_MANIFEST_SHA256 = (
    "c179996caafc91fb17ed6e56c56953ab01c6dfdf921f138b254c8aea1951e98e"
)
EXPECTED_CHAIN1_LENGTH = 270
EXPECTED_AFFIBODY_LENGTH = 58
PEPTIDE_LENGTH = 9

# Two structure contexts are prespecified because no single fixed crystal can
# exactly represent the assay construct.  The native context preserves the
# complete crystal assembly expected by StaB/ProteinMPNN.  The resolved-fragment
# context removes beta-2-microglobulin and HLA 182--276, which are replaced by
# coordinate-less SMART in the assay.  Both retain identical A1--181/P/H
# coordinates and use the assay identities at HLA 84 and 167.
NATIVE_CRYSTAL_CONTEXT = "native_crystal_context"
ASSAY_RESOLVED_FRAGMENT = "assay_resolved_fragment"
CONTEXT_MODES = (NATIVE_CRYSTAL_CONTEXT, ASSAY_RESOLVED_FRAGMENT)
CONTEXT_SCHEMA_VERSIONS = {
    NATIVE_CRYSTAL_CONTEXT: "stab-libb-native-crystal-context-v1",
    ASSAY_RESOLVED_FRAGMENT: "stab-libb-assay-resolved-fragment-v1",
}
CONTEXT_CHAINS = {
    NATIVE_CRYSTAL_CONTEXT: {
        "complex": ("A", "B", "P", "H"),
        "partner1": ("A", "B", "P"),
    },
    ASSAY_RESOLVED_FRAGMENT: {
        "complex": ("A", "P", "H"),
        "partner1": ("A", "P"),
    },
}
AFFIBODY_CHAINS = ("H",)
PEPTIDE_CHAIN = "P"
AFFIBODY_CHAIN = "H"

PEPTIDE_DESIGNED_POSITIONS = (4, 5)
AFFIBODY_DESIGNED_POSITIONS = (6, 10, 13, 14, 17)
AFFIBODY_PYTHON_INDICES = (5, 9, 12, 13, 16)
AFFIBODY_CRYSTAL_ALIGNED_POSITIONS = (8, 12, 15, 16, 19)
SITE_LABELS = (
    "peptide_position_4",
    "peptide_position_5",
    "affibody_displayed_position_6_crystal_position_8",
    "affibody_displayed_position_10_crystal_position_12",
    "affibody_displayed_position_13_crystal_position_15",
    "affibody_displayed_position_14_crystal_position_16",
    "affibody_displayed_position_17_crystal_position_19",
)

# Chain H contains full-Affibody sequence positions 3--57.  Positions 1, 2,
# and 58 are unresolved in the selected crystallographic copy.
AFFIBODY_OBSERVED_START_1_BASED = 3
AFFIBODY_OBSERVED_END_1_BASED = 57

# The assay construct carries the first 181 residues of PDB chain A after a
# 70-residue SMART segment.  Relative to the crystal, its HLA segment contains
# the two documented scaffold substitutions Y84A and W167A.  The fixed crystal
# supplies coordinates; the sequence tensor must use these assay identities.
ASSAY_HLA_CHAIN1_START_0_BASED = 70
ASSAY_HLA_LENGTH = 181
ASSAY_HLA_SUBSTITUTIONS = ((84, "Y", "A"), (167, "W", "A"))
ASSAY_LINKER_START_0_BASED = ASSAY_HLA_CHAIN1_START_0_BASED + ASSAY_HLA_LENGTH
ASSAY_LINKER_SEQUENCE = "GSGGSGGGGS"
ASSAY_PEPTIDE_START_0_BASED = ASSAY_LINKER_START_0_BASED + len(
    ASSAY_LINKER_SEQUENCE
)

DEFAULT_STRUCTURE_MAPPING = (
    PRIVATE_ROOT
    / "derived"
    / "libb_fixed_crystal_contract_provider_revision_120_v1"
    / "residue_mapping.json"
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CANONICAL_AA = frozenset(ALPHABET[:-1])
HIDDEN_DIM = 128
VOCAB_SIZE = len(ALPHABET)

SAVED_ARRAYS = {
    "global_features": "feature",
    "residue_features": "feature",
    "residue_mask": "mask",
    "residue_hidden_delta": "feature",
    "residue_complex_hidden": "feature",
    "residue_isolated_hidden": "feature",
    "residue_log_probs_complex": "feature",
    "residue_log_probs_isolated": "feature",
    "residue_log_probs_delta": "feature",
    "residue_aa_onehot": "feature",
    "pair_compatibility_features": "feature",
}

_LABEL_LIKE = re.compile(
    r"(?:retention|binder|weak[_-]?label|outcome|target|enrichment|"
    r"(?:^|_)r0*(?:1|9|10)(?:_|$).*count)",
    re.IGNORECASE,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_contract(context_mode: str) -> dict[str, Any]:
    """Return the explicit scientific contract for one structure context."""

    _require(context_mode in CONTEXT_MODES, f"unknown context mode: {context_mode}")
    if context_mode == NATIVE_CRYSTAL_CONTEXT:
        return {
            "schema_version": CONTEXT_SCHEMA_VERSIONS[context_mode],
            "coordinates": "fixed crystal A:1-276/B:1-99/P:1-9/H:5-59",
            "complex_chains": ["A", "B", "P", "H"],
            "partner1_chains": ["A", "B", "P"],
            "complex_residue_count": 439,
            "partner1_residue_count": 384,
            "affibody_residue_count": 55,
            "assay_identity_grafts": ["A:Y84A", "A:W167A"],
            "non_assay_crystal_context": [
                "native beta-2-microglobulin B:1-99",
                "native HLA A:182-276",
            ],
            "coordinate_less_assay_components_omitted": [
                "SMART residues 1-70",
                "linker GSGGSGGGGS",
            ],
            "interpretation": (
                "upstream-faithful complete native-crystal context with the two "
                "assay HLA identities grafted; not an exact SMART assay structure"
            ),
        }
    return {
        "schema_version": CONTEXT_SCHEMA_VERSIONS[context_mode],
        "coordinates": "fixed crystal A:1-181/P:1-9/H:5-59",
        "complex_chains": ["A", "P", "H"],
        "partner1_chains": ["A", "P"],
        "complex_residue_count": 245,
        "partner1_residue_count": 190,
        "affibody_residue_count": 55,
        "assay_identity_grafts": ["A:Y84A", "A:W167A"],
        "crystal_components_omitted_to_match_assay": [
            "native beta-2-microglobulin B:1-99",
            "native HLA A:182-276",
        ],
        "coordinate_less_assay_components_omitted": [
            "SMART residues 1-70",
            "linker GSGGSGGGGS",
        ],
        "chain_topology_caveat": (
            "HLA A and peptide P remain separate ProteinMPNN chains because the "
            "assay linker has no coordinates"
        ),
        "interpretation": (
            "assay-composition-matched resolved interface fragment with an artificial "
            "A:181 coordinate terminus; not a complete isolated assay structure"
        ),
    }


def _reject_supervision_fields(value: Any, location: str = "$") -> None:
    """Reject supervision material recursively before model code sees rows."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if _LABEL_LIKE.search(key):
                raise ValueError(
                    f"supervision-like field {key!r} is forbidden at {location}"
                )
            _reject_supervision_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_supervision_fields(child, f"{location}[{index}]")


@dataclass(frozen=True)
class CanonicalRow:
    row_index: int
    row_id: str
    split: str
    chain1_sequence: str
    chain2_sequence: str
    sequence_pair_sha256: str

    @property
    def peptide_sequence(self) -> str:
        return self.chain1_sequence[-PEPTIDE_LENGTH:]

    @property
    def affibody_observed_sequence(self) -> str:
        return self.chain2_sequence[
            AFFIBODY_OBSERVED_START_1_BASED - 1 : AFFIBODY_OBSERVED_END_1_BASED
        ]

    @property
    def designed_sequence(self) -> str:
        peptide = "".join(
            self.peptide_sequence[position - 1]
            for position in PEPTIDE_DESIGNED_POSITIONS
        )
        affibody = "".join(
            self.chain2_sequence[position - 1]
            for position in AFFIBODY_DESIGNED_POSITIONS
        )
        return peptide + affibody


@dataclass(frozen=True)
class TemplateBundle:
    context_mode: str
    complex_domain: Mapping[str, Any]
    pmhc_domain: Mapping[str, Any]
    affibody_domain: Mapping[str, Any]
    complex_reference_sequence: str
    pmhc_reference_sequence: str
    affibody_reference_sequence: str
    assay_hla_segment: str
    structure_mapping_sha256: str
    complex_residue_keys: tuple[str, ...]
    pmhc_residue_keys: tuple[str, ...]
    affibody_residue_keys: tuple[str, ...]
    site_residue_keys: tuple[str, ...]
    complex_site_indices: tuple[int, ...]
    isolated_site_indices: tuple[int, ...]
    peptide_complex_slice: slice
    affibody_complex_slice: slice


@dataclass(frozen=True)
class DomainOutput:
    hidden: np.ndarray
    log_probs: np.ndarray


def archive_run_contract(
    context_mode: str,
    *,
    structure_mapping_sha256: str = STRUCTURE_MAPPING_SHA256,
    additional: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable contract written before a feature run starts.

    The contract deliberately contains the full chain-composition description,
    rather than only a short context name.  This makes it impossible to resume
    or merge a shard if the assay/native-crystal interpretation has changed.
    Runtime-only measurements (elapsed time, software versions, peak memory)
    belong in the final provenance and are not part of this pre-run contract.
    """

    _require(context_mode in CONTEXT_MODES, f"unknown context mode: {context_mode}")
    _require(
        structure_mapping_sha256 == STRUCTURE_MAPPING_SHA256,
        "archive run contract has the wrong structure mapping",
    )
    contract: dict[str, Any] = {
        "context_mode": context_mode,
        "template_context": context_contract(context_mode),
        "structure_mapping_schema": STRUCTURE_MAPPING_SCHEMA,
        "structure_mapping_sha256": structure_mapping_sha256,
    }
    if additional is not None:
        _reject_supervision_fields(additional, "$.run_contract")
        collisions = sorted(set(contract).intersection(additional))
        _require(not collisions, f"duplicate archive-contract fields: {collisions}")
        contract.update(dict(additional))
    return contract


def _validate_row(raw: Mapping[str, Any]) -> CanonicalRow:
    required = {
        "row_index",
        "row_id",
        "split",
        "chain1_sequence",
        "chain2_sequence",
        "sequence_pair_sha256",
    }
    missing = sorted(required.difference(raw))
    _require(not missing, f"row is missing fields: {missing}")
    row_index = int(raw["row_index"])
    row_id = str(raw["row_id"])
    split = str(raw["split"])
    chain1 = str(raw["chain1_sequence"])
    chain2 = str(raw["chain2_sequence"])
    pair_hash = str(raw["sequence_pair_sha256"])
    _require(row_index >= 0, "row_index must be nonnegative")
    _require(bool(row_id), "row_id is empty")
    _require(split in {"train", "eval"}, "split must be train or eval")
    _require(len(chain1) == EXPECTED_CHAIN1_LENGTH, "chain1 length changed")
    _require(len(chain2) == EXPECTED_AFFIBODY_LENGTH, "Affibody length changed")
    _require(set(chain1).issubset(CANONICAL_AA), "chain1 is noncanonical")
    _require(set(chain2).issubset(CANONICAL_AA), "Affibody is noncanonical")
    observed_hash = hashlib.sha256((chain1 + "|" + chain2).encode("ascii")).hexdigest()
    _require(pair_hash == observed_hash, "sequence-pair hash mismatch")
    return CanonicalRow(row_index, row_id, split, chain1, chain2, pair_hash)


def load_canonical_rows(
    path: str | Path,
    *,
    require_full_dataset: bool = True,
) -> tuple[dict[str, Any], list[CanonicalRow]]:
    """Load the existing label-free canonical rows and re-audit the split."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "row manifest must be a JSON object")
    _reject_supervision_fields(payload)
    _require(
        payload.get("schema_version") == ROW_MANIFEST_SCHEMA,
        "row-manifest schema changed",
    )
    raw_rows = payload.get("rows")
    _require(isinstance(raw_rows, list) and raw_rows, "row manifest has no rows")
    rows = [_validate_row(raw) for raw in raw_rows]
    _require(
        [row.row_index for row in rows] == list(range(len(rows))),
        "canonical row indices are not contiguous",
    )
    _require(len({row.row_id for row in rows}) == len(rows), "duplicate row_id")
    _require(
        len({row.sequence_pair_sha256 for row in rows}) == len(rows),
        "duplicate sequence pair",
    )
    train = [row for row in rows if row.split == "train"]
    evaluation = [row for row in rows if row.split == "eval"]
    if require_full_dataset:
        _require(
            sha256_file(path) == EXPECTED_ROW_MANIFEST_SHA256,
            "canonical row-manifest checksum changed",
        )
        _require(len(train) == EXPECTED_TRAIN_ROWS, "training-row count changed")
        _require(len(evaluation) == EXPECTED_EVAL_ROWS, "evaluation-row count changed")
    _require(
        {row.chain1_sequence for row in train}.isdisjoint(
            {row.chain1_sequence for row in evaluation}
        ),
        "evaluation peptide partner occurs in training",
    )
    _require(
        {row.chain2_sequence for row in train}.isdisjoint(
            {row.chain2_sequence for row in evaluation}
        ),
        "evaluation Affibody partner occurs in training",
    )
    return payload, rows


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def load_pinned_vendor(vendor_root: str | Path, checkpoint: str | Path):
    """Import and instantiate the exact official final StaB-ddG ProteinMPNN."""

    vendor_root = Path(vendor_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    _require(vendor_root.is_dir(), f"StaB-ddG checkout not found: {vendor_root}")
    _require(_git_commit(vendor_root) == UPSTREAM_COMMIT, "StaB-ddG commit changed")
    _require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    _require(
        sha256_file(checkpoint) == FINAL_CHECKPOINT_SHA256,
        "StaB-ddG final-checkpoint checksum changed",
    )
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    import torch
    from stabddg.mpnn_utils import ProteinMPNN

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.0,
        augment_eps=0.0,
    )
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        # PyTorch 1.12, used by the pinned upstream StaB environment, predates
        # the ``weights_only`` keyword.  The checkpoint hash and repository
        # commit have already been verified immediately above.
        state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=True)
    _require(
        sum(parameter.numel() for parameter in model.parameters()) == 1_660_485,
        "unexpected ProteinMPNN parameter count",
    )
    return model


def _truncate_chain_prefix(
    domain: dict[str, Any], chain: str, retained_length: int
) -> None:
    """Crop one parsed ProteinMPNN chain and all four backbone arrays in place."""

    sequence_key = f"seq_chain_{chain}"
    coordinates_key = f"coords_chain_{chain}"
    _require(sequence_key in domain, f"chain {chain} sequence is absent")
    _require(coordinates_key in domain, f"chain {chain} coordinates are absent")
    original_length = len(domain[sequence_key])
    _require(0 < retained_length <= original_length, "invalid retained chain length")
    domain[sequence_key] = str(domain[sequence_key])[:retained_length]
    coordinate_arrays = domain[coordinates_key]
    for atom_name in ("N", "CA", "C", "O"):
        key = f"{atom_name}_chain_{chain}"
        _require(key in coordinate_arrays, f"chain {chain} lacks {atom_name}")
        _require(
            len(coordinate_arrays[key]) == original_length,
            f"chain {chain} {atom_name} coordinate count changed",
        )
        coordinate_arrays[key] = coordinate_arrays[key][:retained_length]
    chain_order = tuple(domain["masked_list"] + domain["visible_list"])
    domain["seq"] = "".join(str(domain[f"seq_chain_{item}"]) for item in chain_order)
    _require(
        len(domain["seq"])
        == sum(len(domain[f"seq_chain_{item}"]) for item in chain_order),
        "cropped domain sequence length changed",
    )


def _validate_structure_mapping(
    path: str | Path,
    crystal_sequences: Mapping[str, str],
    assay_hla_segment: str,
) -> str:
    """Hard-gate the exact v2 sequence-to-crystal mapping used by extraction."""

    path = Path(path).resolve()
    _require(path.is_file(), f"LibB residue mapping not found: {path}")
    mapping_sha256 = sha256_file(path)
    _require(
        mapping_sha256 == STRUCTURE_MAPPING_SHA256,
        "LibB residue-mapping checksum changed",
    )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(
        payload.get("schema_version") == STRUCTURE_MAPPING_SCHEMA,
        "LibB residue-mapping schema changed",
    )
    _require(
        payload.get("pdb_source", {}).get("sha256") == REFERENCE_PDB_SHA256,
        "residue mapping points to a different crystal",
    )
    selected_copy = payload.get("chain_identification", {}).get("selected_copy", {})
    _require(
        selected_copy.get("hla_chain") == "A"
        and selected_copy.get("peptide_chain") == "P"
        and selected_copy.get("affibody_chain") == "H",
        "residue mapping selected a different crystal copy",
    )

    mappings = payload.get("residue_mappings", {})
    hla = mappings.get("assay_hla", {})
    hla_rows = hla.get("residues", [])
    _require(len(hla_rows) == ASSAY_HLA_LENGTH, "HLA mapping length changed")
    _require(
        hla.get("canonical_chain1_hla_start_index_0_based")
        == ASSAY_HLA_CHAIN1_START_0_BASED
        and hla.get("canonical_chain1_hla_end_index_exclusive_0_based")
        == ASSAY_HLA_CHAIN1_START_0_BASED + ASSAY_HLA_LENGTH,
        "canonical HLA span changed",
    )
    for offset, row in enumerate(hla_rows):
        position = offset + 1
        _require(row.get("hla_sequence_position_1_based") == position, "HLA order changed")
        _require(row.get("pdb_chain_id") == "A", "HLA chain changed")
        _require(row.get("pdb_chain_ordinal_1_based") == position, "HLA PDB order changed")
        _require(row.get("pdb_residue_id") == f"A:{position}", "HLA residue ID changed")
        _require(
            row.get("canonical_chain1_index_0_based")
            == ASSAY_HLA_CHAIN1_START_0_BASED + offset,
            "HLA canonical index changed",
        )
        _require(
            row.get("pdb_amino_acid") == crystal_sequences["A"][offset],
            "mapped crystal HLA identity changed",
        )
        _require(
            row.get("model_amino_acid") == assay_hla_segment[offset],
            "mapped assay HLA identity changed",
        )
        _require(
            row.get("coordinate_policy") == "keep PDB coordinates unchanged",
            "HLA coordinate policy changed",
        )
    expected_overrides = [
        {
            "hla_sequence_position_1_based": position,
            "pdb_amino_acid": crystal_amino_acid,
            "model_amino_acid": assay_amino_acid,
            "substitution": f"{crystal_amino_acid}{position}{assay_amino_acid}",
        }
        for position, crystal_amino_acid, assay_amino_acid in ASSAY_HLA_SUBSTITUTIONS
    ]
    observed_overrides = [
        {key: row.get(key) for key in expected_overrides[0]}
        for row in hla.get("identity_overrides", [])
    ]
    _require(observed_overrides == expected_overrides, "HLA identity overrides changed")
    _require(
        sum(bool(row.get("identity_override")) for row in hla_rows)
        == len(ASSAY_HLA_SUBSTITUTIONS),
        "unexpected HLA identity override",
    )

    peptide_rows = mappings.get("peptide", [])
    _require(len(peptide_rows) == PEPTIDE_LENGTH, "peptide mapping length changed")
    for offset, row in enumerate(peptide_rows):
        position = offset + 1
        _require(row.get("pdb_chain_id") == "P", "peptide chain changed")
        _require(row.get("pdb_residue_id") == f"P:{position}", "peptide order changed")
        _require(row.get("sequence_position_1_based") == position, "peptide index changed")
        _require(
            row.get("reference_amino_acid") == crystal_sequences["P"][offset],
            "peptide reference identity changed",
        )
    _require(
        tuple(
            row["sequence_position_1_based"]
            for row in peptide_rows
            if row.get("mutable_site")
        )
        == PEPTIDE_DESIGNED_POSITIONS,
        "peptide designed-site mapping changed",
    )

    affibody_rows = mappings.get("affibody", [])
    _require(len(affibody_rows) == 55, "Affibody mapping length changed")
    for offset, row in enumerate(affibody_rows):
        sequence_position = AFFIBODY_OBSERVED_START_1_BASED + offset
        pdb_number = sequence_position + 2
        _require(row.get("pdb_chain_id") == "H", "Affibody chain changed")
        _require(row.get("pdb_residue_id") == f"H:{pdb_number}", "Affibody order changed")
        _require(
            row.get("sequence_position_1_based") == sequence_position,
            "Affibody sequence index changed",
        )
        _require(
            row.get("reference_amino_acid") == crystal_sequences["H"][offset],
            "Affibody reference identity changed",
        )
    _require(
        tuple(
            row["sequence_position_1_based"]
            for row in affibody_rows
            if row.get("mutable_site")
        )
        == AFFIBODY_DESIGNED_POSITIONS,
        "Affibody designed-site mapping changed",
    )
    mutable_affibody_rows = [row for row in affibody_rows if row.get("mutable_site")]
    _require(
        tuple(row.get("design_code_position_1_based") for row in mutable_affibody_rows)
        == tuple(range(1, 6)),
        "Affibody mutation-code mapping changed",
    )
    _require(
        tuple(row.get("sequence_position_1_based") - 1 for row in mutable_affibody_rows)
        == AFFIBODY_PYTHON_INDICES,
        "Affibody Python-index mapping changed",
    )
    _require(
        tuple(row.get("pdb_residue_number") for row in mutable_affibody_rows)
        == AFFIBODY_CRYSTAL_ALIGNED_POSITIONS,
        "Affibody crystal-aligned numbering changed",
    )
    return mapping_sha256


def build_template_bundle(
    pdb_path: str | Path,
    vendor_root: str | Path,
    context_mode: str,
    residue_mapping_path: str | Path = DEFAULT_STRUCTURE_MAPPING,
) -> TemplateBundle:
    """Build one of two prespecified, explicitly labeled crystal contexts."""

    _require(context_mode in CONTEXT_MODES, f"unknown context mode: {context_mode}")
    context_chains = CONTEXT_CHAINS[context_mode]
    complex_chains = context_chains["complex"]
    partner1_chains = context_chains["partner1"]

    vendor_root = Path(vendor_root).resolve()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    from stabddg.mpnn_utils import parse_PDB

    pdb_path = Path(pdb_path).resolve()
    _require(pdb_path.is_file(), f"LibB crystal PDB not found: {pdb_path}")
    _require(
        sha256_file(pdb_path) == REFERENCE_PDB_SHA256,
        "LibB crystal PDB checksum changed",
    )

    def parse(chains: Sequence[str]) -> dict[str, Any]:
        parsed = parse_PDB(str(pdb_path), input_chain_list=list(chains))
        _require(len(parsed) == 1, f"failed to parse chains {chains}")
        domain = parsed[0]
        domain["masked_list"] = list(chains)
        domain["visible_list"] = []
        observed = tuple(
            key[len("seq_chain_") :]
            for key in domain
            if key.startswith("seq_chain_")
        )
        _require(observed == tuple(chains), f"crystal chain order changed: {observed}")
        return domain

    complex_domain = parse(complex_chains)
    pmhc_domain = parse(partner1_chains)
    affibody_domain = parse(AFFIBODY_CHAINS)
    chain_sequences = {
        chain: str(complex_domain[f"seq_chain_{chain}"])
        for chain in complex_chains
    }
    expected_lengths = {"A": 276, "P": 9, "H": 55}
    if context_mode == NATIVE_CRYSTAL_CONTEXT:
        expected_lengths["B"] = 99
    _require(
        {chain: len(sequence) for chain, sequence in chain_sequences.items()}
        == expected_lengths,
        "selected crystal-copy chain lengths changed",
    )
    _require(chain_sequences["P"] == "SLLMWITQV", "reference peptide changed")
    _require(
        chain_sequences["H"]
        == "NKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAP",
        "reference Affibody chain changed",
    )

    model_chain_a = list(chain_sequences["A"])
    for position, crystal_amino_acid, assay_amino_acid in ASSAY_HLA_SUBSTITUTIONS:
        _require(
            model_chain_a[position - 1] == crystal_amino_acid,
            f"crystal HLA position {position} changed",
        )
        model_chain_a[position - 1] = assay_amino_acid
    model_chain_a = "".join(model_chain_a)
    assay_hla_segment = model_chain_a[:ASSAY_HLA_LENGTH]
    structure_mapping_sha256 = _validate_structure_mapping(
        residue_mapping_path, chain_sequences, assay_hla_segment
    )

    if context_mode == ASSAY_RESOLVED_FRAGMENT:
        _truncate_chain_prefix(complex_domain, "A", ASSAY_HLA_LENGTH)
        _truncate_chain_prefix(pmhc_domain, "A", ASSAY_HLA_LENGTH)
        retained_chain_a = assay_hla_segment
    else:
        retained_chain_a = model_chain_a
    model_sequences = dict(chain_sequences)
    model_sequences["A"] = retained_chain_a
    complex_reference_sequence = "".join(
        model_sequences[chain] for chain in complex_chains
    )
    pmhc_reference_sequence = "".join(
        model_sequences[chain] for chain in partner1_chains
    )
    _require(
        len(complex_domain["seq"]) == len(complex_reference_sequence),
        "complex structure/sequence length changed",
    )
    _require(
        len(pmhc_domain["seq"]) == len(pmhc_reference_sequence),
        "partner-1 structure/sequence length changed",
    )

    residue_keys_by_chain = {
        "A": tuple(f"A:{number}" for number in range(1, len(retained_chain_a) + 1)),
        "B": tuple(f"B:{number}" for number in range(1, 100)),
        "P": tuple(f"P:{number}" for number in range(1, 10)),
        "H": tuple(f"H:{number}" for number in range(5, 60)),
    }
    complex_residue_keys = tuple(
        key for chain in complex_chains for key in residue_keys_by_chain[chain]
    )
    pmhc_residue_keys = tuple(
        key for chain in partner1_chains for key in residue_keys_by_chain[chain]
    )
    affibody_residue_keys = residue_keys_by_chain["H"]
    site_residue_keys = tuple(
        [f"P:{position}" for position in PEPTIDE_DESIGNED_POSITIONS]
        + [f"H:{position + 2}" for position in AFFIBODY_DESIGNED_POSITIONS]
    )

    peptide_start = sum(
        len(model_sequences[chain])
        for chain in partner1_chains
        if chain != PEPTIDE_CHAIN
    )
    affibody_start = peptide_start + len(chain_sequences["P"])
    peptide_indices = tuple(
        peptide_start + position - 1 for position in PEPTIDE_DESIGNED_POSITIONS
    )
    affibody_indices = tuple(
        affibody_start + position - AFFIBODY_OBSERVED_START_1_BASED
        for position in AFFIBODY_DESIGNED_POSITIONS
    )
    complex_sites = peptide_indices + affibody_indices
    isolated_sites = tuple(
        peptide_start + position - 1 for position in PEPTIDE_DESIGNED_POSITIONS
    ) + tuple(
        position - AFFIBODY_OBSERVED_START_1_BASED
        for position in AFFIBODY_DESIGNED_POSITIONS
    )
    return TemplateBundle(
        context_mode=context_mode,
        complex_domain=complex_domain,
        pmhc_domain=pmhc_domain,
        affibody_domain=affibody_domain,
        complex_reference_sequence=complex_reference_sequence,
        pmhc_reference_sequence=pmhc_reference_sequence,
        affibody_reference_sequence=str(affibody_domain["seq"]),
        assay_hla_segment=assay_hla_segment,
        structure_mapping_sha256=structure_mapping_sha256,
        complex_residue_keys=complex_residue_keys,
        pmhc_residue_keys=pmhc_residue_keys,
        affibody_residue_keys=affibody_residue_keys,
        site_residue_keys=site_residue_keys,
        complex_site_indices=complex_sites,
        isolated_site_indices=isolated_sites,
        peptide_complex_slice=slice(peptide_start, affibody_start),
        affibody_complex_slice=slice(
            affibody_start, affibody_start + len(chain_sequences["H"])
        ),
    )


def validate_row_against_template(row: CanonicalRow, template: TemplateBundle) -> None:
    assay_hla = row.chain1_sequence[
        ASSAY_HLA_CHAIN1_START_0_BASED : ASSAY_HLA_CHAIN1_START_0_BASED
        + ASSAY_HLA_LENGTH
    ]
    _require(
        assay_hla == template.assay_hla_segment,
        "canonical SMART-HLA segment does not match crystal coordinates plus Y84A/W167A",
    )
    _require(
        row.chain1_sequence[
            ASSAY_LINKER_START_0_BASED:ASSAY_PEPTIDE_START_0_BASED
        ]
        == ASSAY_LINKER_SEQUENCE,
        "canonical SMART-HLA linker changed",
    )
    _require(
        row.chain1_sequence[ASSAY_PEPTIDE_START_0_BASED:] == row.peptide_sequence,
        "canonical peptide span changed",
    )
    peptide = row.peptide_sequence
    reference_peptide = template.complex_reference_sequence[
        template.peptide_complex_slice
    ]
    mutable_peptide = {position - 1 for position in PEPTIDE_DESIGNED_POSITIONS}
    for index, (observed, expected) in enumerate(zip(peptide, reference_peptide)):
        if index not in mutable_peptide:
            _require(observed == expected, "non-designed peptide residue changed")

    affibody = row.affibody_observed_sequence
    reference_affibody = template.affibody_reference_sequence
    mutable_affibody = {
        position - AFFIBODY_OBSERVED_START_1_BASED
        for position in AFFIBODY_DESIGNED_POSITIONS
    }
    _require(len(affibody) == len(reference_affibody), "observed Affibody span changed")
    for index, (observed, expected) in enumerate(zip(affibody, reference_affibody)):
        if index not in mutable_affibody:
            _require(observed == expected, "non-designed Affibody residue changed")


def current_pair_sequences(
    row: CanonicalRow, template: TemplateBundle
) -> tuple[str, str, str]:
    """Return complex, partner-1 fragment, and Affibody resolved sequences."""

    validate_row_against_template(row, template)
    pmhc = (
        template.pmhc_reference_sequence[: template.peptide_complex_slice.start]
        + row.peptide_sequence
    )
    affibody = row.affibody_observed_sequence
    complex_sequence = pmhc + affibody
    _require(
        len(complex_sequence) == len(template.complex_reference_sequence),
        "complex sequence length changed",
    )
    return complex_sequence, pmhc, affibody


def _encode_sequences(sequences: Sequence[str]) -> np.ndarray:
    indices = {amino_acid: index for index, amino_acid in enumerate(ALPHABET)}
    _require(bool(sequences), "sequence batch is empty")
    length = len(sequences[0])
    _require(all(len(sequence) == length for sequence in sequences), "mixed lengths")
    _require(
        all(set(sequence).issubset(indices) for sequence in sequences),
        "sequence batch contains an unsupported amino acid",
    )
    return np.asarray(
        [[indices[amino_acid] for amino_acid in sequence] for sequence in sequences],
        dtype=np.int64,
    )


def _stable_target_last_orders(
    domain_residue_keys: Sequence[str], target_residue_keys: Sequence[str]
) -> np.ndarray:
    """Build shard-independent orders with each requested target decoded last."""

    domain_residue_keys = tuple(domain_residue_keys)
    _require(len(set(domain_residue_keys)) == len(domain_residue_keys), "duplicate residue key")
    index_by_key = {key: index for index, key in enumerate(domain_residue_keys)}
    missing = sorted(set(target_residue_keys).difference(index_by_key))
    _require(not missing, f"target residues are absent from domain: {missing}")
    base = sorted(
        range(len(domain_residue_keys)),
        key=lambda index: hashlib.sha256(
            ("stab-libb-target-order-v1|" + domain_residue_keys[index]).encode("ascii")
        ).digest(),
    )
    orders = []
    for target_key in target_residue_keys:
        target_index = index_by_key[target_key]
        orders.append([index for index in base if index != target_index] + [target_index])
    return np.asarray(orders, dtype=np.int64)


def run_domain_model(
    model,
    domain: Mapping[str, Any],
    sequences: Sequence[str],
    domain_residue_keys: Sequence[str],
    target_residue_keys: Sequence[str],
    device,
) -> DomainOutput:
    """Extract decoder state/log-probabilities with each target decoded last.

    ProteinMPNN is autoregressive.  Making a designed residue the final decoded
    target ensures that its representation and amino-acid distribution can use
    every other resolved residue in both partners.  The non-target priority is
    a stable hash of physical residue keys, so it is independent of batching,
    sharding, and Python hash randomization.  Isolated domains restrict the same
    physical-key priority and therefore preserve relative order.
    """

    import torch
    from stabddg.mpnn_utils import featurize

    encoded = _encode_sequences(sequences)
    batch_size, length = encoded.shape
    target_residue_keys = tuple(target_residue_keys)
    target_count = len(target_residue_keys)
    _require(target_count > 0, "no target residues requested")
    _require(len(domain_residue_keys) == length, "domain residue-key length changed")
    X, _, mask, _, chain_M, residue_idx, _, chain_encoding = featurize(
        [domain], device
    )
    _require(X.shape[1] == length, "sequence does not match structure length")
    expanded_size = batch_size * target_count
    X = X.repeat(expanded_size, 1, 1, 1)
    mask = mask.repeat(expanded_size, 1)
    chain_M = chain_M.repeat(expanded_size, 1)
    residue_idx = residue_idx.repeat(expanded_size, 1)
    chain_encoding = chain_encoding.repeat(expanded_size, 1)
    expanded_encoded = np.repeat(encoded, target_count, axis=0)
    sequence_tensor = torch.as_tensor(expanded_encoded, dtype=torch.long, device=device)
    zero_noise = torch.zeros_like(X)
    order_bank = _stable_target_last_orders(domain_residue_keys, target_residue_keys)
    orders = torch.as_tensor(
        np.tile(order_bank[None, :, :], (batch_size, 1, 1)).reshape(
            expanded_size, length
        ),
        dtype=torch.long,
        device=device,
    )
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output)

    handle = model.decoder_layers[-1].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            log_probs = model(
                X,
                sequence_tensor,
                mask,
                chain_M,
                residue_idx,
                chain_encoding,
                fix_order=orders,
                fix_backbone_noise=zero_noise,
            )
    finally:
        handle.remove()
    _require(len(captured) == 1, "failed to capture final decoder state")
    hidden_full = captured[0].detach().float().reshape(
        batch_size, target_count, length, HIDDEN_DIM
    )
    log_probs_full = log_probs.detach().float().reshape(
        batch_size, target_count, length, VOCAB_SIZE
    )
    index_by_key = {key: index for index, key in enumerate(domain_residue_keys)}
    target_indices = torch.as_tensor(
        [index_by_key[key] for key in target_residue_keys],
        dtype=torch.long,
        device=hidden_full.device,
    )
    target_axis = torch.arange(target_count, device=hidden_full.device)
    hidden = hidden_full[:, target_axis, target_indices, :].cpu().numpy()
    log_probs = log_probs_full[:, target_axis, target_indices, :].cpu().numpy()
    _require(hidden.shape == (batch_size, target_count, HIDDEN_DIM), "hidden shape changed")
    _require(
        log_probs.shape == (batch_size, target_count, VOCAB_SIZE),
        "log-probability shape changed",
    )
    _require(np.isfinite(hidden).all(), "non-finite decoder hidden state")
    _require(np.isfinite(log_probs).all(), "non-finite log probabilities")
    return DomainOutput(hidden=hidden, log_probs=log_probs)


def assemble_feature_batch(
    rows: Sequence[CanonicalRow],
    template: TemplateBundle,
    complex_output: DomainOutput,
    pmhc_output: DomainOutput,
    affibody_output: DomainOutput,
) -> dict[str, np.ndarray]:
    """Assemble aligned, current-pair features from three domain evaluations."""

    batch_size = len(rows)
    site_count = len(SITE_LABELS)
    _require(
        complex_output.hidden.shape == (batch_size, site_count, HIDDEN_DIM),
        "complex hidden shape changed",
    )
    _require(
        pmhc_output.hidden.shape == (batch_size, 2, HIDDEN_DIM),
        "pMHC hidden shape changed",
    )
    _require(
        affibody_output.hidden.shape == (batch_size, 5, HIDDEN_DIM),
        "Affibody hidden shape changed",
    )
    isolated_hidden = np.concatenate(
        (pmhc_output.hidden, affibody_output.hidden), axis=1
    )
    isolated_log_probs = np.concatenate(
        (pmhc_output.log_probs, affibody_output.log_probs), axis=1
    )
    site_complex_hidden = complex_output.hidden
    site_isolated_hidden = isolated_hidden
    site_hidden_delta = site_complex_hidden - site_isolated_hidden
    site_complex_log_probs = complex_output.log_probs
    site_isolated_log_probs = isolated_log_probs
    site_log_prob_delta = site_complex_log_probs - site_isolated_log_probs

    designed = [row.designed_sequence for row in rows]
    designed_indices = _encode_sequences(designed)
    row_index = np.arange(batch_size)[:, None]
    site_index = np.arange(len(SITE_LABELS))[None, :]
    selected_site_delta = site_log_prob_delta[
        row_index, site_index, designed_indices
    ]
    onehot = np.eye(VOCAB_SIZE, dtype=np.float32)[designed_indices]

    pair_compatibility = np.stack(
        (
            selected_site_delta.mean(axis=1),
            selected_site_delta[:, :2].mean(axis=1),
            selected_site_delta[:, 2:].mean(axis=1),
        ),
        axis=1,
    ).astype(np.float32)
    global_features = np.concatenate(
        (
            site_hidden_delta.mean(axis=1),
            selected_site_delta.astype(np.float32),
            pair_compatibility,
        ),
        axis=1,
    ).astype(np.float32)
    _require(
        global_features.shape == (batch_size, HIDDEN_DIM + len(SITE_LABELS) + 3),
        "global feature shape changed",
    )

    arrays = {
        "global_features": global_features,
        "residue_features": np.concatenate(
            (
                site_hidden_delta,
                site_log_prob_delta,
                selected_site_delta[:, :, None],
            ),
            axis=2,
        ).astype(np.float32),
        "residue_mask": np.ones((batch_size, len(SITE_LABELS)), dtype=np.bool_),
        "residue_hidden_delta": site_hidden_delta.astype(np.float32),
        "residue_complex_hidden": site_complex_hidden.astype(np.float32),
        "residue_isolated_hidden": site_isolated_hidden.astype(np.float32),
        "residue_log_probs_complex": site_complex_log_probs.astype(np.float32),
        "residue_log_probs_isolated": site_isolated_log_probs.astype(np.float32),
        "residue_log_probs_delta": site_log_prob_delta.astype(np.float32),
        "residue_aa_onehot": onehot,
        "pair_compatibility_features": pair_compatibility,
    }
    for name, array in arrays.items():
        _require(np.isfinite(array).all(), f"{name} contains non-finite values")
    return arrays


def extract_feature_batch(
    model,
    rows: Sequence[CanonicalRow],
    template: TemplateBundle,
    device,
) -> dict[str, np.ndarray]:
    sequences = [current_pair_sequences(row, template) for row in rows]
    complex_sequences = [value[0] for value in sequences]
    pmhc_sequences = [value[1] for value in sequences]
    affibody_sequences = [value[2] for value in sequences]
    complex_output = run_domain_model(
        model,
        template.complex_domain,
        complex_sequences,
        template.complex_residue_keys,
        template.site_residue_keys,
        device,
    )
    pmhc_output = run_domain_model(
        model,
        template.pmhc_domain,
        pmhc_sequences,
        template.pmhc_residue_keys,
        template.site_residue_keys[:2],
        device,
    )
    affibody_output = run_domain_model(
        model,
        template.affibody_domain,
        affibody_sequences,
        template.affibody_residue_keys,
        template.site_residue_keys[2:],
        device,
    )
    return assemble_feature_batch(
        rows, template, complex_output, pmhc_output, affibody_output
    )


def partition_rows(
    rows: Sequence[CanonicalRow], shard_index: int, num_shards: int
) -> list[CanonicalRow]:
    _require(num_shards > 0, "num_shards must be positive")
    _require(0 <= shard_index < num_shards, "invalid shard_index")
    return [row for row in rows if row.row_index % num_shards == shard_index]


def iter_batches(rows: Sequence[CanonicalRow], batch_size: int) -> Iterable[list[CanonicalRow]]:
    _require(batch_size > 0, "batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield list(rows[start : start + batch_size])


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: Sequence[CanonicalRow]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("row_index", "row_id", "split"))
            for output_index, row in enumerate(rows):
                writer.writerow((output_index, row.row_id, row.split))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_private_output_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    _require(PRIVATE_ROOT.is_dir(), "private_data directory is missing")
    try:
        path.relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output must be below private_data") from exc
    relative = path.relative_to(REPO_ROOT)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(ignored.returncode == 0, "output path is not Git-ignored")
    return path


class FeatureArchiveWriter:
    """Write a mmap-safe, label-free archive with an incomplete marker."""

    def __init__(
        self,
        root: str | Path,
        rows: Sequence[CanonicalRow],
        specs: Mapping[str, tuple[tuple[int, ...], np.dtype]],
        run_contract: Mapping[str, Any],
    ):
        self.root = validate_private_output_path(root)
        _require(not self.root.exists(), f"output already exists: {self.root}")
        self.root.mkdir(parents=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.rows = list(rows)
        self.specs = dict(specs)
        self.run_contract = dict(run_contract)
        context_mode = str(self.run_contract.get("context_mode", ""))
        expected_context = context_contract(context_mode)
        _require(
            self.run_contract.get("template_context") == expected_context,
            "archive run contract has the wrong chain composition or lengths",
        )
        _require(
            self.run_contract.get("structure_mapping_schema")
            == STRUCTURE_MAPPING_SCHEMA
            and self.run_contract.get("structure_mapping_sha256")
            == STRUCTURE_MAPPING_SHA256,
            "archive run contract has the wrong structure mapping",
        )
        self.marker = self.root / "EXTRACTION_INCOMPLETE"
        self.marker.write_text("incomplete\n", encoding="ascii")
        os.chmod(self.marker, 0o600)
        _atomic_csv(self.root / "metadata.csv", self.rows)
        _atomic_json(self.root / "extraction_contract.json", self.run_contract)
        self.maps: dict[str, np.memmap] = {}
        for name, (tail_shape, dtype) in self.specs.items():
            _require(name in SAVED_ARRAYS, f"unknown feature array {name}")
            path = self.root / f".{name}.npy.incomplete"
            self.maps[name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.dtype(dtype),
                shape=(len(self.rows),) + tuple(tail_shape),
            )
        self.offset = 0
        self.checkpoint()

    @classmethod
    def resume(
        cls,
        root: str | Path,
        rows: Sequence[CanonicalRow],
        specs: Mapping[str, tuple[tuple[int, ...], np.dtype]],
        run_contract: Mapping[str, Any],
    ) -> "FeatureArchiveWriter":
        """Reopen an interrupted archive at its last flushed batch boundary."""

        root = validate_private_output_path(root)
        _require(root.is_dir(), f"resume output does not exist: {root}")
        _require(
            (root / "EXTRACTION_INCOMPLETE").is_file(),
            "resume output is not marked incomplete",
        )
        _require(not (root / "manifest.json").exists(), "completed archive cannot resume")
        rows = list(rows)
        run_contract = dict(run_contract)
        with (root / "extraction_contract.json").open("r", encoding="utf-8") as handle:
            observed_run_contract = json.load(handle)
        _require(
            observed_run_contract == run_contract,
            "resume extraction contract changed (context, chains, mapping, or shard)",
        )
        context_mode = str(run_contract.get("context_mode", ""))
        _require(
            run_contract.get("template_context") == context_contract(context_mode),
            "resume template context changed",
        )
        with (root / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
            observed = list(csv.reader(handle))
        expected = [["row_index", "row_id", "split"]] + [
            [str(index), row.row_id, row.split] for index, row in enumerate(rows)
        ]
        _require(observed == expected, "resume row identity/order changed")
        with (root / "progress.json").open("r", encoding="utf-8") as handle:
            progress = json.load(handle)
        _require(
            progress.get("extraction_contract_sha256")
            == sha256_file(root / "extraction_contract.json"),
            "resume extraction-contract digest changed",
        )
        _require(
            progress.get("row_ids_sha256")
            == canonical_json_sha256([row.row_id for row in rows]),
            "resume row-ID digest changed",
        )
        offset = int(progress.get("completed_rows", -1))
        _require(0 <= offset <= len(rows), "resume offset is invalid")
        instance = cls.__new__(cls)
        instance.root = root
        instance.rows = rows
        instance.specs = dict(specs)
        instance.run_contract = run_contract
        instance.marker = root / "EXTRACTION_INCOMPLETE"
        instance.maps = {}
        for name, (tail_shape, dtype) in instance.specs.items():
            _require(name in SAVED_ARRAYS, f"unknown feature array {name}")
            path = root / f".{name}.npy.incomplete"
            _require(path.is_file(), f"resume feature file is missing: {path}")
            mmap = np.lib.format.open_memmap(path, mode="r+")
            _require(
                mmap.shape == (len(rows),) + tuple(tail_shape),
                f"resume shape changed for {name}",
            )
            _require(mmap.dtype == np.dtype(dtype), f"resume dtype changed for {name}")
            instance.maps[name] = mmap
        instance.offset = offset
        return instance

    def append(self, arrays: Mapping[str, np.ndarray]) -> None:
        _require(set(arrays) == set(self.maps), "feature batch keys changed")
        batch_size = next(iter(arrays.values())).shape[0]
        _require(batch_size > 0, "cannot append an empty batch")
        _require(self.offset + batch_size <= len(self.rows), "too many feature rows")
        for name, values in arrays.items():
            values = np.asarray(values)
            expected = self.maps[name].shape[1:]
            _require(values.shape == (batch_size,) + expected, f"{name} shape changed")
            _require(values.dtype == self.maps[name].dtype, f"{name} dtype changed")
            self.maps[name][self.offset : self.offset + batch_size] = values
        self.offset += batch_size

    def checkpoint(self) -> None:
        """Flush arrays before atomically advancing the resumable row offset."""

        for mmap in self.maps.values():
            mmap.flush()
        _atomic_json(
            self.root / "progress.json",
            {
                "completed_rows": self.offset,
                "row_count": len(self.rows),
                "row_ids_sha256": canonical_json_sha256(
                    [row.row_id for row in self.rows]
                ),
                "extraction_contract_sha256": sha256_file(
                    self.root / "extraction_contract.json"
                ),
            },
        )

    def finish(self, provenance: Mapping[str, Any]) -> dict[str, Any]:
        _require(self.offset == len(self.rows), "feature archive is incomplete")
        context_mode = str(provenance.get("context_mode", ""))
        context = context_contract(context_mode)
        for key, expected in self.run_contract.items():
            _require(
                provenance.get(key) == expected,
                f"final provenance changed from extraction contract: {key}",
            )
        contracts = {}
        for name, mmap in self.maps.items():
            mmap.flush()
            del mmap
            temporary = self.root / f".{name}.npy.incomplete"
            final = self.root / f"{name}.npy"
            os.chmod(temporary, 0o600)
            os.replace(temporary, final)
            contracts[name] = {
                "file": final.name,
                "bytes": final.stat().st_size,
                "sha256": sha256_file(final),
                "shape": list(np.load(final, mmap_mode="r", allow_pickle=False).shape),
                "dtype": str(np.load(final, mmap_mode="r", allow_pickle=False).dtype),
                "kind": SAVED_ARRAYS[name],
            }
        self.maps.clear()
        metadata = self.root / "metadata.csv"
        extraction_contract = self.root / "extraction_contract.json"
        manifest = {
            "schema_version": FEATURE_ARCHIVE_SCHEMA,
            "extractor_schema_version": EXTRACTOR_SCHEMA,
            "template_context_schema_version": context["schema_version"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "supervision_fields_read": [],
            "row_count": len(self.rows),
            "row_ids_sha256": canonical_json_sha256([row.row_id for row in self.rows]),
            "metadata": {
                "file": metadata.name,
                "rows": len(self.rows),
                "bytes": metadata.stat().st_size,
                "sha256": sha256_file(metadata),
            },
            "extraction_contract": {
                "file": extraction_contract.name,
                "bytes": extraction_contract.stat().st_size,
                "sha256": sha256_file(extraction_contract),
                "contract": self.run_contract,
            },
            "arrays": contracts,
            "feature_semantics": {
                "residue_order": list(SITE_LABELS),
                "residue_features": (
                    "complex-minus-isolated decoder hidden (128), 21-way log-probability "
                    "contrast, and current-amino-acid selected contrast"
                ),
                "residue_hidden_delta": "complex decoder hidden minus isolated-partner decoder hidden",
                "global_features": (
                    "mean seven-site hidden delta (128), seven current-residue "
                    "log-probability deltas, and three designed-site compatibility means"
                ),
                "pair_compatibility_order": [
                    "all_seven_designed_sites",
                    "two_designed_peptide_sites",
                    "five_designed_affibody_sites",
                ],
                "decoding_orders": "stable physical-residue-key priority with each designed target last",
                "backbone_noise_angstrom": 0.0,
                "wild_type_mutant_subtraction": False,
                "binding_energy_or_ddg": False,
                "contrast_interpretation": (
                    "current-sequence frozen ProteinMPNN representation and "
                    "sequence-backbone compatibility contrast; not a physical "
                    "binding-energy decomposition"
                ),
                "assay_scaffold_identity": {
                    "context_mode": context_mode,
                    "context_contract": context,
                    "hla_sequence_source": "canonical assay SMART-HLA segment",
                    "hla_positions_mapped_to_pdb_chain_a": [1, 181],
                    "crystal_to_assay_substitutions": ["A:Y84A", "A:W167A"],
                    "structure_mapping_schema": STRUCTURE_MAPPING_SCHEMA,
                    "structure_mapping_sha256": STRUCTURE_MAPPING_SHA256,
                },
            },
            "provenance": dict(provenance),
        }
        _atomic_json(self.root / "manifest.json", manifest)
        (self.root / "progress.json").unlink()
        self.marker.unlink()
        _fsync_parent(self.marker)
        return manifest


def feature_specs() -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    sites = len(SITE_LABELS)
    return {
        "global_features": ((HIDDEN_DIM + sites + 3,), np.dtype(np.float32)),
        "residue_features": ((sites, HIDDEN_DIM + VOCAB_SIZE + 1), np.dtype(np.float32)),
        "residue_mask": ((sites,), np.dtype(np.bool_)),
        "residue_hidden_delta": ((sites, HIDDEN_DIM), np.dtype(np.float32)),
        "residue_complex_hidden": ((sites, HIDDEN_DIM), np.dtype(np.float32)),
        "residue_isolated_hidden": ((sites, HIDDEN_DIM), np.dtype(np.float32)),
        "residue_log_probs_complex": ((sites, VOCAB_SIZE), np.dtype(np.float32)),
        "residue_log_probs_isolated": ((sites, VOCAB_SIZE), np.dtype(np.float32)),
        "residue_log_probs_delta": ((sites, VOCAB_SIZE), np.dtype(np.float32)),
        "residue_aa_onehot": ((sites, VOCAB_SIZE), np.dtype(np.float32)),
        "pair_compatibility_features": ((3,), np.dtype(np.float32)),
    }
