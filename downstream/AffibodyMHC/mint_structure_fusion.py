"""Label-free fusion contracts for frozen MINT and structural features.

The initial structural comparison freezes both upstream encoders.  This module
therefore contains only deterministic, parameter-free fusion operations; the
shared classifier in :mod:`libb_structural_readout` remains the only trainable
component.  No function accepts labels, retention measurements, or a model-
selection metric.

Two feature layouts are supported:

``FrozenMintLateFusion``
    Concatenate a current-pair structural vector with the existing LibB MINT
    layer-5 chain-mean vector.  This is *late fusion*: MINT and the structural
    encoder process the pair independently and meet immediately before the
    matched readout.

``FrozenMappedMintResidueFusion``
    Concatenate a structural representation and an already aligned MINT
    layer-5 representation at each structural residue.  A separate mapping
    mask distinguishes real MINT values from residues for which the MINT input
    has no counterpart.  This handles both the seven-site StaB-derived layout
    and the larger RDE-derived structural patch without assuming either one's
    length or structural channel dimension.

The existing global cache contains only chain means.  Per-residue layer-5 MINT
representations still need to be extracted and aligned before using the second
contract.  The helpers at the end of this file codify the seven unambiguous
LibB designed-site addresses for that future extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[2]

MINT_LAYER = 5
MINT_RESIDUE_DIM = 1280
MINT_CHAIN_MEAN_DIM = 2 * MINT_RESIDUE_DIM
MINT_LAYER5_FEATURE_NAME = "mint_layer_05_chain_mean"
MINT_ROW_ID_NAME = "pair_uid"

MINT_CHAIN_LENGTHS = (270, 58)
MINT_PEPTIDE_CHAIN_ID = 0
MINT_AFFIBODY_CHAIN_ID = 1
MINT_PEPTIDE_START_POSITION = 262

DEFAULT_MINT_MULTILAYER_ARCHIVE = (
    REPO_ROOT
    / "private_data/derived/mint_multilayer_v1/merged/"
    "mint_multilayer_chain_mean_features.npz"
)
DEFAULT_MINT_MULTILAYER_MANIFEST = (
    REPO_ROOT / "private_data/derived/mint_multilayer_v1/merged/manifest.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _floating_tensor(value: Tensor, name: str, dimensions: int) -> None:
    _require(isinstance(value, Tensor), f"{name} must be a torch tensor")
    _require(value.ndim == dimensions, f"{name} must have {dimensions} dimensions")
    _require(torch.is_floating_point(value), f"{name} must be floating point")


def _finite_selected(value: Tensor, mask: Tensor, name: str) -> None:
    selected = value[mask]
    _require(selected.numel() > 0, f"{name} has no selected values")
    _require(bool(torch.isfinite(selected).all().item()), f"{name} contains non-finite values")


def aligned_row_indices(
    requested_row_ids: Sequence[str], available_row_ids: Sequence[str]
) -> np.ndarray:
    """Return indices that align a label-free feature archive to requested rows.

    ``pair_uid`` in the MINT cache is the same opaque identifier used as
    ``row_id`` by the structural archives.  This helper requires a one-to-one
    match and deliberately has no label or split argument.
    """

    requested = tuple(map(str, requested_row_ids))
    available = tuple(map(str, available_row_ids))
    _require(requested and all(requested), "requested row IDs are empty")
    _require(available and all(available), "available row IDs are empty")
    _require(len(set(requested)) == len(requested), "requested row IDs are not unique")
    _require(len(set(available)) == len(available), "available row IDs are not unique")
    lookup = {row_id: index for index, row_id in enumerate(available)}
    missing = sorted(set(requested).difference(lookup))
    _require(not missing, f"MINT features are missing {len(missing)} requested row IDs")
    return np.asarray([lookup[row_id] for row_id in requested], dtype=np.int64)


class FrozenMintLateFusion(nn.Module):
    """Concatenate independent frozen structure and MINT pair vectors.

    The output is intended to be passed to ``CapacityMatchedNonlinearReadout``.
    Inputs are detached inside ``forward`` so this initial experiment cannot
    accidentally fine-tune either upstream encoder.
    """

    required_features = ("structural_vector", MINT_LAYER5_FEATURE_NAME)

    def __init__(
        self,
        structural_dim: int,
        mint_dim: int = MINT_CHAIN_MEAN_DIM,
    ) -> None:
        super().__init__()
        _require(int(structural_dim) >= 1, "structural dimension must be positive")
        _require(int(mint_dim) >= 1, "MINT dimension must be positive")
        self.structural_dim = int(structural_dim)
        self.mint_dim = int(mint_dim)
        self.output_dim = self.structural_dim + self.mint_dim

    def forward(
        self,
        *,
        structural_vector: Tensor,
        mint_layer_05_chain_mean: Tensor,
    ) -> Tensor:
        _floating_tensor(structural_vector, "structural_vector", 2)
        _floating_tensor(
            mint_layer_05_chain_mean, MINT_LAYER5_FEATURE_NAME, 2
        )
        _require(
            structural_vector.shape[0] == mint_layer_05_chain_mean.shape[0],
            "structure and MINT batch sizes differ",
        )
        _require(
            structural_vector.device == mint_layer_05_chain_mean.device,
            "structure and MINT vectors are on different devices",
        )
        _require(
            structural_vector.shape[1] == self.structural_dim,
            f"structural_vector must have {self.structural_dim} channels",
        )
        _require(
            mint_layer_05_chain_mean.shape[1] == self.mint_dim,
            f"{MINT_LAYER5_FEATURE_NAME} must have {self.mint_dim} channels",
        )
        _require(
            bool(torch.isfinite(structural_vector).all().item()),
            "structural_vector contains non-finite values",
        )
        _require(
            bool(torch.isfinite(mint_layer_05_chain_mean).all().item()),
            f"{MINT_LAYER5_FEATURE_NAME} contains non-finite values",
        )
        # The feature extractors are frozen by experimental design.  Detaching
        # here makes that boundary true even if a caller forgets no_grad().
        structure = structural_vector.detach().float()
        mint = mint_layer_05_chain_mean.detach().float()
        return torch.cat((structure, mint), dim=-1)


class FrozenMappedMintResidueFusion(nn.Module):
    """Fuse frozen structural and mapped MINT values residue by residue.

    Parameters are deliberately limited to dimensions; residue count is
    dynamic.  The expected inputs are:

    ``structural_residue_features``
        ``[B, R, D_structure]`` in the extractor's documented residue order.
    ``mint_residue_features``
        ``[B, R, 1280]`` already reordered to exactly the same residues.
    ``residue_mask``
        ``[B, R]``; true for real (non-padding) structural residues.
    ``mint_mapped_mask``
        ``[B, R]``; true only where the MINT value maps to that exact residue.

    The last output channel is the mapping indicator.  Unmapped MINT values
    and padded positions are replaced with zeros, so their unused contents
    cannot affect a prediction.  The caller must preserve ``residue_mask`` for
    the generic mean/max pooling step.
    """

    required_features = (
        "structural_residue_features",
        "mint_residue_features",
        "residue_mask",
        "mint_mapped_mask",
    )

    def __init__(
        self,
        structural_dim: int,
        mint_dim: int = MINT_RESIDUE_DIM,
    ) -> None:
        super().__init__()
        _require(int(structural_dim) >= 1, "structural dimension must be positive")
        _require(int(mint_dim) >= 1, "MINT dimension must be positive")
        self.structural_dim = int(structural_dim)
        self.mint_dim = int(mint_dim)
        self.output_dim = self.structural_dim + self.mint_dim + 1

    def forward(
        self,
        *,
        structural_residue_features: Tensor,
        mint_residue_features: Tensor,
        residue_mask: Tensor,
        mint_mapped_mask: Tensor,
    ) -> Tensor:
        _floating_tensor(
            structural_residue_features, "structural_residue_features", 3
        )
        _floating_tensor(mint_residue_features, "mint_residue_features", 3)
        _require(
            tuple(structural_residue_features.shape[:2])
            == tuple(mint_residue_features.shape[:2]),
            "structure and MINT residue axes differ",
        )
        _require(
            structural_residue_features.device == mint_residue_features.device,
            "structure and MINT residues are on different devices",
        )
        _require(
            structural_residue_features.shape[-1] == self.structural_dim,
            f"structural residues must have {self.structural_dim} channels",
        )
        _require(
            mint_residue_features.shape[-1] == self.mint_dim,
            f"MINT residues must have {self.mint_dim} channels",
        )
        spatial_shape = tuple(structural_residue_features.shape[:2])
        _require(
            isinstance(residue_mask, Tensor)
            and residue_mask.dtype == torch.bool
            and tuple(residue_mask.shape) == spatial_shape,
            "residue_mask must be bool with shape [B,R]",
        )
        _require(
            isinstance(mint_mapped_mask, Tensor)
            and mint_mapped_mask.dtype == torch.bool
            and tuple(mint_mapped_mask.shape) == spatial_shape,
            "mint_mapped_mask must be bool with shape [B,R]",
        )
        _require(
            structural_residue_features.device == residue_mask.device
            == mint_mapped_mask.device,
            "residue features and masks are on different devices",
        )
        _require(
            bool((mint_mapped_mask & ~residue_mask).sum().eq(0).item()),
            "MINT mapping includes a padded structural residue",
        )
        _require(
            bool(residue_mask.any(dim=1).all().item()),
            "every example needs at least one structural residue",
        )
        _require(
            bool(mint_mapped_mask.any(dim=1).all().item()),
            "every fusion example needs at least one mapped MINT residue",
        )
        _finite_selected(
            structural_residue_features, residue_mask, "structural_residue_features"
        )
        _finite_selected(
            mint_residue_features, mint_mapped_mask, "mint_residue_features"
        )

        structure = structural_residue_features.detach().float().masked_fill(
            ~residue_mask.unsqueeze(-1), 0.0
        )
        mint = mint_residue_features.detach().float().masked_fill(
            ~mint_mapped_mask.unsqueeze(-1), 0.0
        )
        mapping_indicator = mint_mapped_mask.unsqueeze(-1).to(dtype=torch.float32)
        fused = torch.cat((structure, mint, mapping_indicator), dim=-1)
        return fused.masked_fill(~residue_mask.unsqueeze(-1), 0.0)


@dataclass(frozen=True)
class MintResidueAddress:
    """One 1-based MINT sequence address for a structural residue."""

    name: str
    chain_id: int | None
    sequence_position: int | None

    @property
    def is_mapped(self) -> bool:
        return self.chain_id is not None and self.sequence_position is not None

    def validate(self) -> None:
        _require(bool(self.name), "residue address needs a name")
        if not self.is_mapped:
            _require(
                self.chain_id is None and self.sequence_position is None,
                "an unmapped address must set both chain and position to None",
            )
            return
        _require(self.chain_id in (0, 1), "MINT chain ID must be 0 or 1")
        position = int(self.sequence_position)
        _require(
            1 <= position <= MINT_CHAIN_LENGTHS[int(self.chain_id)],
            "MINT sequence position is outside its chain",
        )


def libb_peptide_mint_address(
    peptide_position: int, name: str | None = None
) -> MintResidueAddress:
    """Map a 1-based nine-residue peptide position into MINT chain 0."""

    peptide_position = int(peptide_position)
    _require(1 <= peptide_position <= 9, "LibB peptide position must be in 1..9")
    return MintResidueAddress(
        name=name or f"pep{peptide_position}",
        chain_id=MINT_PEPTIDE_CHAIN_ID,
        sequence_position=MINT_PEPTIDE_START_POSITION + peptide_position - 1,
    )


def libb_affibody_mint_address(
    affibody_position: int, name: str | None = None
) -> MintResidueAddress:
    """Map a 1-based Affibody position into MINT chain 1."""

    affibody_position = int(affibody_position)
    _require(1 <= affibody_position <= 58, "LibB Affibody position must be in 1..58")
    return MintResidueAddress(
        name=name or f"aff{affibody_position}",
        chain_id=MINT_AFFIBODY_CHAIN_ID,
        sequence_position=affibody_position,
    )


LIBB_DESIGNED_SITE_MINT_ADDRESSES = (
    libb_peptide_mint_address(4),
    libb_peptide_mint_address(5),
    libb_affibody_mint_address(6),
    libb_affibody_mint_address(10),
    libb_affibody_mint_address(13),
    libb_affibody_mint_address(14),
    libb_affibody_mint_address(17),
)


def gather_mint_residue_features(
    chain0_residue_features: Tensor,
    chain1_residue_features: Tensor,
    addresses: Sequence[MintResidueAddress],
) -> tuple[Tensor, Tensor]:
    """Gather aligned MINT values from token-free per-chain layer-5 arrays.

    The two inputs must already exclude CLS/EOS/padding and use sequence order.
    This function does not run MINT.  Unmapped structural residues are emitted
    as zeros with a false mapping mask.
    """

    _floating_tensor(chain0_residue_features, "chain0_residue_features", 3)
    _floating_tensor(chain1_residue_features, "chain1_residue_features", 3)
    _require(
        chain0_residue_features.shape[0] == chain1_residue_features.shape[0],
        "MINT chain batch sizes differ",
    )
    _require(
        chain0_residue_features.device == chain1_residue_features.device,
        "MINT chains are on different devices",
    )
    _require(
        chain0_residue_features.dtype == chain1_residue_features.dtype,
        "MINT chains use different dtypes",
    )
    _require(
        tuple(chain0_residue_features.shape[1:])
        == (MINT_CHAIN_LENGTHS[0], MINT_RESIDUE_DIM),
        "chain 0 must have shape [B,270,1280]",
    )
    _require(
        tuple(chain1_residue_features.shape[1:])
        == (MINT_CHAIN_LENGTHS[1], MINT_RESIDUE_DIM),
        "chain 1 must have shape [B,58,1280]",
    )
    addresses = tuple(addresses)
    _require(addresses, "at least one structural residue address is required")
    for address in addresses:
        _require(isinstance(address, MintResidueAddress), "invalid MINT residue address")
        address.validate()

    batch = int(chain0_residue_features.shape[0])
    values = []
    mapped = []
    for address in addresses:
        if not address.is_mapped:
            values.append(
                torch.zeros(
                    batch,
                    MINT_RESIDUE_DIM,
                    dtype=chain0_residue_features.dtype,
                    device=chain0_residue_features.device,
                )
            )
            mapped.append(False)
            continue
        source = (
            chain0_residue_features
            if int(address.chain_id) == MINT_PEPTIDE_CHAIN_ID
            else chain1_residue_features
        )
        values.append(source[:, int(address.sequence_position) - 1, :])
        mapped.append(True)
    output = torch.stack(values, dim=1)
    mapping_mask = torch.tensor(mapped, dtype=torch.bool, device=output.device)
    return output, mapping_mask.unsqueeze(0).expand(batch, -1)
