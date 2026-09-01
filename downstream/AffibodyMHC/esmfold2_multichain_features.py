"""Protein-complex inputs for the Transformers 5.16.1 ESMFold2 trunk.

The released ``prepare_protein_features`` helper featurizes one protein chain.
This module composes those official single-chain features into one protein-only
complex and bundles the atom tensors expected by ``EsmFold2Model.forward``.
Calling ``model(**prepared.forward_kwargs)`` runs the folding trunk and
distogram head; it does not invoke coordinate diffusion. The released trunk is
still stochastic (random pair-state initialization and forced LM dropout), so
callers must seed PyTorch immediately before ``forward`` when reproducibility
is required.

This is intentionally a narrow compatibility layer for ``transformers==5.16.1``.
It does not implement ligands, modifications, covalent links between chains, or
paired MSAs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from torch import Tensor
from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2AtomInputs
from transformers.models.esmfold2.protein_utils import prepare_protein_features

SUPPORTED_TRANSFORMERS_VERSION = "5.16.1"
ATOM_INPUT_FIELDS = (
    "ref_pos",
    "ref_charge",
    "atom_attention_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_token",
)

_TOKEN_FIELDS = (
    "mol_type",
    "res_type",
    "input_ids",
    "attention_mask",
    "deletion_mean",
)
_MSA_TOKEN_FIELDS = (
    "msa",
    "msa_attention_mask",
    "has_deletion",
    "deletion_value",
)


@dataclass(frozen=True)
class PreparedEsmFold2TrunkInputs:
    """Inputs separated according to the ESMFold2 5.16.1 ``forward`` API.

    ``forward_kwargs`` can be passed directly to ``EsmFold2Model.forward``.
    ``distogram_atom_idx`` is retained for residue-to-atom mapping and for the
    confidence head, but is deliberately absent from ``forward_kwargs`` because
    the trunk-only ``forward`` method does not accept it.
    """

    forward_kwargs: dict[str, Any]
    distogram_atom_idx: Tensor


def _check_transformers_version() -> None:
    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "ESMFold2 multichain feature preparation was validated against "
            f"transformers=={SUPPORTED_TRANSFORMERS_VERSION}, but found "
            f"{transformers.__version__}. Use the pinned ESMFold2 environment."
        )


def _validated_sequences(sequences: Sequence[str]) -> tuple[str, ...]:
    if isinstance(sequences, str):
        raise TypeError("sequences must be a sequence of chain strings, not one string")
    chains = tuple(sequences)
    if not chains:
        raise ValueError("at least one protein chain is required")
    for chain_index, sequence in enumerate(chains):
        if not isinstance(sequence, str):
            raise TypeError(f"chain {chain_index} is not a string")
        if not sequence:
            raise ValueError(f"chain {chain_index} is empty")
    return chains


def _entity_and_symmetry_ids(sequences: Sequence[str]) -> tuple[list[int], list[int]]:
    """Use standard complex semantics for repeated, sequence-identical chains.

    Entity IDs are one-based to extend the official single-chain helper's
    ``entity_id == 1`` convention. Symmetry-copy IDs are zero-based, matching
    the helper's ``sym_id == 0`` convention.
    """

    sequence_to_entity: dict[str, int] = {}
    next_copy_by_entity: dict[int, int] = {}
    entity_ids: list[int] = []
    sym_ids: list[int] = []
    for sequence in sequences:
        if sequence not in sequence_to_entity:
            sequence_to_entity[sequence] = len(sequence_to_entity) + 1
        entity_id = sequence_to_entity[sequence]
        sym_id = next_copy_by_entity.get(entity_id, 0)
        next_copy_by_entity[entity_id] = sym_id + 1
        entity_ids.append(entity_id)
        sym_ids.append(sym_id)
    return entity_ids, sym_ids


def _strip_real_atoms(
    features: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], Tensor]:
    """Remove the official helper's right padding and remap representative atoms."""

    mask = features["atom_attention_mask"][0].bool()
    if mask.ndim != 1:
        raise ValueError("atom_attention_mask must have shape [1, num_atoms]")
    real_atom_count = int(mask.sum().item())
    if real_atom_count == 0:
        raise ValueError("a protein chain unexpectedly contains no real atoms")

    stripped = {name: features[name][0, mask].clone() for name in ATOM_INPUT_FIELDS}

    # Do not rely on padding remaining strictly right-aligned when translating
    # the C-beta/C-alpha representative-atom indices into the stripped array.
    old_to_new = torch.full((mask.numel(),), -1, dtype=torch.long, device=mask.device)
    old_to_new[mask] = torch.arange(real_atom_count, device=mask.device)
    old_indices = features["distogram_atom_idx"][0].long()
    remapped_distogram_indices = old_to_new[old_indices]
    if bool((remapped_distogram_indices < 0).any()):
        raise ValueError("distogram_atom_idx points to a padded atom")
    return stripped, remapped_distogram_indices


def _pad_atoms(atoms: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Right-pad the combined complex once, to ESMFold2's 32-atom multiple."""

    number_of_real_atoms = atoms["atom_attention_mask"].shape[0]
    number_of_atoms = ((number_of_real_atoms + 31) // 32) * 32
    padded: dict[str, Tensor] = {}
    for name in ATOM_INPUT_FIELDS:
        value = atoms[name]
        output = value.new_zeros((1, number_of_atoms, *value.shape[1:]))
        output[0, :number_of_real_atoms] = value
        padded[name] = output
    return padded


def prepare_multichain_protein_features(
    sequences: Sequence[str],
    device: torch.device | str | None = None,
) -> dict[str, Tensor]:
    """Compose official single-chain features into one protein complex.

    Chain order is preserved. ``asym_id`` is the zero-based chain index,
    ``residue_index`` restarts at zero in every chain, and ``token_index`` is
    global across the complex. Sequence-identical chains share an ``entity_id``
    and receive consecutive ``sym_id`` values. Cross-chain ``token_bonds`` are
    always zero.

    The returned dictionary follows the raw schema of the official
    ``prepare_protein_features`` helper and still contains separate atom arrays.
    Use :func:`bundle_trunk_inputs` before calling ``EsmFold2Model.forward``.
    """

    _check_transformers_version()
    chains = _validated_sequences(sequences)
    chain_features = [
        prepare_protein_features(sequence, device=device) for sequence in chains
    ]

    for chain_index, features in enumerate(chain_features):
        if features["attention_mask"].shape != (1, len(chains[chain_index])):
            raise ValueError("official featurizer returned an unexpected token shape")
        if features["msa"].shape[:2] != (1, 1):
            raise ValueError("only the official depth-one sequence MSA is supported")

    entity_ids, sym_ids = _entity_and_symmetry_ids(chains)
    lengths = [len(sequence) for sequence in chains]
    total_tokens = sum(lengths)
    token_template = chain_features[0]["token_index"]
    token_device = token_template.device
    token_dtype = token_template.dtype

    combined: dict[str, Tensor] = {}
    combined["token_index"] = torch.arange(
        total_tokens, dtype=token_dtype, device=token_device
    ).unsqueeze(0)
    combined["residue_index"] = torch.cat(
        [
            torch.arange(length, dtype=token_dtype, device=token_device).unsqueeze(0)
            for length in lengths
        ],
        dim=1,
    )
    combined["asym_id"] = torch.cat(
        [
            token_template.new_full((1, length), chain_index)
            for chain_index, length in enumerate(lengths)
        ],
        dim=1,
    )
    combined["entity_id"] = torch.cat(
        [
            token_template.new_full((1, length), entity_id)
            for length, entity_id in zip(lengths, entity_ids)
        ],
        dim=1,
    )
    combined["sym_id"] = torch.cat(
        [
            token_template.new_full((1, length), sym_id)
            for length, sym_id in zip(lengths, sym_ids)
        ],
        dim=1,
    )

    for name in _TOKEN_FIELDS:
        combined[name] = torch.cat(
            [features[name] for features in chain_features], dim=1
        )
    for name in _MSA_TOKEN_FIELDS:
        combined[name] = torch.cat(
            [features[name] for features in chain_features], dim=2
        )

    # Preserve any official within-chain bond features as diagonal blocks while
    # making the cross-chain blocks structurally incapable of containing a bond.
    bond_template = chain_features[0]["token_bonds"]
    token_bonds = bond_template.new_zeros((1, total_tokens, total_tokens, 1))
    token_offset = 0
    for length, features in zip(lengths, chain_features):
        token_slice = slice(token_offset, token_offset + length)
        token_bonds[:, token_slice, token_slice, :] = features["token_bonds"]
        token_offset += length
    combined["token_bonds"] = token_bonds

    stripped_atoms: list[dict[str, Tensor]] = []
    combined_distogram_indices: list[Tensor] = []
    token_offset = 0
    atom_offset = 0
    for length, features in zip(lengths, chain_features):
        atoms, distogram_indices = _strip_real_atoms(features)
        atoms["atom_to_token"] += token_offset
        # In 5.16.1 ref_space_uid is the atom's token/group ID, so it must be
        # global for the same reason as atom_to_token.
        atoms["ref_space_uid"] += token_offset
        combined_distogram_indices.append(distogram_indices + atom_offset)
        stripped_atoms.append(atoms)
        token_offset += length
        atom_offset += atoms["atom_attention_mask"].shape[0]

    unpadded_atoms = {
        name: torch.cat([atoms[name] for atoms in stripped_atoms], dim=0)
        for name in ATOM_INPUT_FIELDS
    }
    combined.update(_pad_atoms(unpadded_atoms))
    combined["distogram_atom_idx"] = torch.cat(
        combined_distogram_indices, dim=0
    ).unsqueeze(0)
    return combined


def bundle_trunk_inputs(
    features: Mapping[str, Tensor],
) -> PreparedEsmFold2TrunkInputs:
    """Bundle raw atom arrays for ``EsmFold2Model.forward`` without diffusion."""

    missing = [
        name
        for name in (*ATOM_INPUT_FIELDS, "distogram_atom_idx")
        if name not in features
    ]
    if missing:
        raise KeyError(f"missing ESMFold2 features: {missing}")

    forward_kwargs: dict[str, Any] = dict(features)
    distogram_atom_idx = forward_kwargs.pop("distogram_atom_idx")
    atom_inputs = EsmFold2AtomInputs(
        **{name: forward_kwargs.pop(name) for name in ATOM_INPUT_FIELDS}
    )
    forward_kwargs["atom_inputs"] = atom_inputs
    return PreparedEsmFold2TrunkInputs(
        forward_kwargs=forward_kwargs,
        distogram_atom_idx=distogram_atom_idx,
    )


def prepare_multichain_trunk_inputs(
    sequences: Sequence[str],
    device: torch.device | str | None = None,
) -> PreparedEsmFold2TrunkInputs:
    """Prepare a protein complex for a direct trunk-only model call."""

    return bundle_trunk_inputs(
        prepare_multichain_protein_features(sequences=sequences, device=device)
    )


__all__ = [
    "ATOM_INPUT_FIELDS",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "PreparedEsmFold2TrunkInputs",
    "bundle_trunk_inputs",
    "prepare_multichain_protein_features",
    "prepare_multichain_trunk_inputs",
]
