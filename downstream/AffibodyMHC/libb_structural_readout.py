"""Readouts for label-free frozen LibB structural representations.

This module deliberately knows nothing about retention measurements.  A
structure extractor (for example an RDE-PPI- or StaB-ddG-derived extractor)
writes one or more floating-point arrays plus optional boolean residue masks.
The arrays are joined to supervision only through opaque ``row_id`` values.

Two explicitly selected readout modes are supported.  The legacy
``frozen_injective_adapter`` mode maps every input to a shared width before a
capacity-matched head.  The cleaner ``native_learned_projection`` mode learns
a direct projection from each representation's native dimension to the first
hidden width.  Keeping the modes explicit preserves old checkpoints while
preventing a new experiment from silently inheriting the legacy adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ALPHABET)}
PEPTIDE_CODE_LENGTH = 2
AFFIBODY_CODE_LENGTH = 5
SEQUENCE_CONTROL_DIM = (PEPTIDE_CODE_LENGTH + AFFIBODY_CODE_LENGTH) * len(
    AA_ALPHABET
)

FEATURE_ARCHIVE_SCHEMA_VERSION = "libb-opaque-frozen-features-v1"
SEQUENCE_SOURCE = "designed_sequence"
FEATURE_SOURCE = "frozen_features"
POOLING_MODES = ("identity", "mean", "max", "mean_max", "ordered_flatten")
FORBIDDEN_FIELD_PARTS = ("retention", "label", "outcome", "target")
FROZEN_INJECTIVE_READOUT = "frozen_injective_adapter"
NATIVE_LEARNED_READOUT = "native_learned_projection"
READOUT_MODES = (FROZEN_INJECTIVE_READOUT, NATIVE_LEARNED_READOUT)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_names(values: Sequence[str]) -> list[str]:
    return sorted(
        value
        for value in map(str, values)
        if any(part in value.lower() for part in FORBIDDEN_FIELD_PARTS)
    )


def require_manifest_subset(
    observed: Mapping[str, object], required: Mapping[str, object], prefix: str = "manifest"
) -> None:
    """Require an exact recursive subset of model-specific provenance fields."""

    for key, expected in required.items():
        path = f"{prefix}.{key}"
        _require(key in observed, f"feature archive lacks required {path}")
        actual = observed[key]
        if isinstance(expected, dict):
            _require(isinstance(actual, dict), f"feature archive {path} is not an object")
            require_manifest_subset(actual, expected, path)
        else:
            _require(actual == expected, f"feature archive {path} changed")


def encode_designed_codes(peptide_code: str, affibody_code: str) -> np.ndarray:
    """Encode the two peptide and five Affibody residues as 140 one-hot values."""

    peptide_code = str(peptide_code)
    affibody_code = str(affibody_code)
    _require(
        len(peptide_code) == PEPTIDE_CODE_LENGTH,
        "LibB peptide code must contain exactly two residues",
    )
    _require(
        len(affibody_code) == AFFIBODY_CODE_LENGTH,
        "LibB Affibody code must contain exactly five residues",
    )
    residues = peptide_code + affibody_code
    unknown = sorted(set(residues).difference(AA_TO_INDEX))
    _require(not unknown, f"designed codes contain noncanonical residues: {unknown}")
    output = np.zeros((len(residues), len(AA_ALPHABET)), dtype=np.float32)
    output[np.arange(len(residues)), [AA_TO_INDEX[value] for value in residues]] = 1.0
    return output.reshape(-1)


def _designed_codes_from_record(record: Mapping[str, object]) -> tuple[str, str]:
    """Read codes directly or derive them from label-free full sequences."""

    peptide_sequence, affibody_sequence = _partner_sequences_from_record(record)
    if "peptide_code" in record and "affibody_code" in record:
        peptide_code = str(record["peptide_code"])
        affibody_code = str(record["affibody_code"])
    else:
        peptide_code = "".join(peptide_sequence[position - 1] for position in (4, 5))
        affibody_code = "".join(
            affibody_sequence[position - 1] for position in (6, 10, 13, 14, 17)
        )

    observed = "".join(peptide_sequence[position - 1] for position in (4, 5))
    _require(observed == peptide_code, "peptide code disagrees with sequence")
    observed = "".join(
        affibody_sequence[position - 1] for position in (6, 10, 13, 14, 17)
    )
    _require(observed == affibody_code, "Affibody code disagrees with sequence")
    return peptide_code, affibody_code


def _partner_sequences_from_record(
    record: Mapping[str, object]
) -> tuple[str, str]:
    """Return full peptide/Affibody identities for strict split validation."""

    if "peptide_sequence" in record:
        peptide_sequence = str(record["peptide_sequence"])
    elif "chain1_sequence" in record:
        chain1 = str(record["chain1_sequence"])
        _require(len(chain1) >= 9, "chain1 is too short to contain the peptide")
        peptide_sequence = chain1[-9:]
    else:
        raise ValueError("sequence record lacks full peptide sequence")
    if "affibody_sequence" in record:
        affibody_sequence = str(record["affibody_sequence"])
    elif "chain2_sequence" in record:
        affibody_sequence = str(record["chain2_sequence"])
    else:
        raise ValueError("sequence record lacks full Affibody sequence")
    _require(len(peptide_sequence) == 9, "LibB peptide sequence must be 9 residues")
    _require(len(affibody_sequence) >= 17, "Affibody sequence is too short")
    unknown = sorted(
        set(peptide_sequence + affibody_sequence).difference(AA_TO_INDEX)
    )
    _require(not unknown, f"partner sequences contain noncanonical residues: {unknown}")
    return peptide_sequence, affibody_sequence


def load_sequence_control_vectors(
    path: str | Path,
    expected_row_ids: Sequence[str],
    expected_splits: Sequence[str],
) -> np.ndarray:
    """Load label-free sequence records and return row-aligned 7-site one-hot data."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), "sequence records must be a JSON object")
    rows = payload.get("rows")
    _require(isinstance(rows, list), "sequence-record JSON lacks rows")
    _require(len(rows) == len(expected_row_ids), "sequence-record row count changed")
    vectors: list[np.ndarray] = []
    partner_identities = {"train": [set(), set()], "eval": [set(), set()]}
    for expected_index, (record, row_id, split) in enumerate(
        zip(rows, expected_row_ids, expected_splits)
    ):
        _require(isinstance(record, dict), "sequence record is not an object")
        forbidden = _forbidden_names(tuple(record))
        _require(
            not forbidden,
            "sequence records contain forbidden supervision fields: "
            + ", ".join(forbidden),
        )
        _require(int(record.get("row_index", -1)) == expected_index, "sequence row index changed")
        _require(str(record.get("row_id", "")) == str(row_id), "sequence row ID changed")
        _require(str(record.get("split", "")) == str(split), "sequence split changed")
        _require(str(split) in partner_identities, "sequence record has unknown split")
        peptide_sequence, affibody_sequence = _partner_sequences_from_record(record)
        partner_identities[str(split)][0].add(peptide_sequence)
        partner_identities[str(split)][1].add(affibody_sequence)
        vectors.append(encode_designed_codes(*_designed_codes_from_record(record)))
    output = np.stack(vectors)
    _require(output.shape == (len(rows), SEQUENCE_CONTROL_DIM), "sequence encoding shape changed")
    _require(
        partner_identities["train"][0].isdisjoint(partner_identities["eval"][0]),
        "evaluation peptide identity occurs in training",
    )
    _require(
        partner_identities["train"][1].isdisjoint(partner_identities["eval"][1]),
        "evaluation Affibody identity occurs in training",
    )
    return output


@dataclass(frozen=True)
class FeatureView:
    """One parameter-free view of an array in an opaque feature archive."""

    array: str
    pooling: str
    mask: str | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, object]) -> "FeatureView":
        array = str(value.get("array", ""))
        pooling = str(value.get("pooling", ""))
        raw_mask = value.get("mask")
        mask = None if raw_mask in (None, "") else str(raw_mask)
        _require(array != "", "feature view lacks an array name")
        _require(pooling in POOLING_MODES, f"unknown pooling mode {pooling!r}")
        _require(not _forbidden_names((array,)), "feature array name resembles supervision")
        if mask is not None:
            _require(not _forbidden_names((mask,)), "feature mask name resembles supervision")
        return cls(array=array, pooling=pooling, mask=mask)


@dataclass(frozen=True)
class OpaqueFeatureStore:
    """Memory-mapped, label-free arrays in canonical opaque-row order."""

    root: Path
    row_ids: tuple[str, ...]
    splits: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    kinds: Mapping[str, str]
    manifest: Mapping[str, object]
    manifest_sha256: str

    @classmethod
    def open(
        cls,
        root: str | Path,
        required_arrays: Sequence[str],
        *,
        verify_all_checksums: bool = False,
        required_manifest: Mapping[str, object] | None = None,
    ) -> "OpaqueFeatureStore":
        root = Path(root).resolve()
        _require(root.is_dir(), f"feature archive does not exist: {root}")
        _require(not (root / "EXTRACTION_INCOMPLETE").exists(), "feature archive is incomplete")
        manifest_path = root / "manifest.json"
        _require(manifest_path.is_file(), "feature archive lacks manifest.json")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _require(
            manifest.get("schema_version") == FEATURE_ARCHIVE_SCHEMA_VERSION,
            "feature archive schema changed",
        )
        _require(
            manifest.get("supervision_fields_read") == [],
            "feature extraction was not label-free",
        )
        if required_manifest is not None:
            require_manifest_subset(manifest, required_manifest)
        contracts = manifest.get("arrays")
        _require(isinstance(contracts, dict), "feature archive lacks array contracts")
        forbidden = _forbidden_names(tuple(contracts))
        _require(
            not forbidden,
            "feature archive contains supervision-like arrays: " + ", ".join(forbidden),
        )

        metadata_path = root / "metadata.csv"
        metadata_contract = manifest.get("metadata", {})
        _require(metadata_path.is_file(), "feature archive lacks metadata.csv")
        _require(metadata_contract.get("file") == "metadata.csv", "metadata filename changed")
        _require(metadata_contract.get("bytes") == metadata_path.stat().st_size, "metadata byte size changed")
        _require(metadata_contract.get("sha256") == sha256_file(metadata_path), "metadata checksum changed")
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            _require(
                header == ["row_index", "row_id", "split"],
                "feature metadata must contain exactly row_index,row_id,split",
            )
            row_ids: list[str] = []
            splits: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                _require(len(row) == 3, f"metadata row {row_number} is malformed")
                _require(row[0] == str(row_number - 2), "metadata row order changed")
                row_ids.append(row[1])
                splits.append(row[2])
        _require(row_ids and all(row_ids), "feature metadata has no usable row IDs")
        _require(len(set(row_ids)) == len(row_ids), "feature metadata has duplicate row IDs")
        _require(set(splits) == {"train", "eval"}, "feature splits must be train and eval")
        _require(manifest.get("row_count") == len(row_ids), "feature row count changed")
        _require(metadata_contract.get("rows") == len(row_ids), "metadata row count changed")
        _require(
            manifest.get("row_ids_sha256") == canonical_json_sha256(row_ids),
            "feature row-ID digest changed",
        )

        names = tuple(dict.fromkeys(map(str, required_arrays)))
        missing = sorted(set(names).difference(contracts))
        _require(not missing, f"feature archive lacks arrays: {missing}")
        arrays: dict[str, np.ndarray] = {}
        kinds: dict[str, str] = {}
        for name in names:
            contract = contracts[name]
            _require(isinstance(contract, dict), f"invalid contract for {name}")
            kind = str(contract.get("kind", ""))
            _require(kind in {"feature", "mask"}, f"{name} has unknown kind")
            path = root / str(contract.get("file", ""))
            _require(path.name == f"{name}.npy", f"{name} filename must be {name}.npy")
            _require(path.is_file(), f"missing array {path}")
            _require(contract.get("bytes") == path.stat().st_size, f"{name} byte size changed")
            if verify_all_checksums:
                _require(contract.get("sha256") == sha256_file(path), f"{name} checksum changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            _require(list(array.shape) == contract.get("shape"), f"{name} shape changed")
            _require(array.shape[0] == len(row_ids), f"{name} row count changed")
            _require(str(array.dtype) == contract.get("dtype"), f"{name} dtype changed")
            if kind == "feature":
                _require(array.ndim >= 2, f"{name} feature array must have a final channel axis")
                _require(np.issubdtype(array.dtype, np.floating), f"{name} must be floating point")
            else:
                _require(array.dtype == np.bool_, f"{name} mask must use bool dtype")
                _require(array.ndim >= 2, f"{name} mask needs a row and spatial axis")
            arrays[name] = array
            kinds[name] = kind
        return cls(
            root,
            tuple(row_ids),
            tuple(splits),
            arrays,
            kinds,
            manifest,
            sha256_file(manifest_path),
        )

    @property
    def row_id_to_index(self) -> dict[str, int]:
        return {row_id: index for index, row_id in enumerate(self.row_ids)}


def pooled_view_dimension(store: OpaqueFeatureStore, view: FeatureView) -> int:
    values = store.arrays[view.array]
    _require(store.kinds[view.array] == "feature", f"{view.array} is not a feature array")
    if view.pooling == "identity":
        _require(values.ndim == 2, "identity pooling requires [N,D]")
        multiplier = 1
    else:
        _require(values.ndim >= 3, f"{view.pooling} pooling requires [N,...,D]")
        multiplier = 2 if view.pooling == "mean_max" else 1
    if view.mask is not None:
        _require(view.pooling != "identity", "identity pooling cannot use a mask")
        mask = store.arrays[view.mask]
        _require(store.kinds[view.mask] == "mask", f"{view.mask} is not a mask array")
        _require(
            tuple(mask.shape) == tuple(values.shape[:-1]),
            f"{view.mask} shape does not match {view.array}",
        )
    if view.pooling == "ordered_flatten":
        if view.mask is None:
            positions = int(np.prod(values.shape[1:-1]))
        else:
            counts = np.asarray(store.arrays[view.mask]).reshape(len(store.row_ids), -1).sum(axis=1)
            _require(bool((counts > 0).all()), f"{view.mask} selects no positions")
            _require(bool((counts == counts[0]).all()), f"{view.mask} has variable selected width")
            positions = int(counts[0])
        return int(values.shape[-1]) * positions
    return int(values.shape[-1]) * multiplier


def pool_feature_row(values: np.ndarray, pooling: str, mask: np.ndarray | None) -> np.ndarray:
    """Parameter-free pooling for one row; the last axis is always channels."""

    values = np.asarray(values, dtype=np.float32)
    _require(bool(np.isfinite(values).all()), "feature row contains non-finite values")
    if pooling == "identity":
        _require(values.ndim == 1 and mask is None, "identity pooling expects one vector")
        return np.array(values, dtype=np.float32, copy=True)
    _require(pooling in POOLING_MODES[1:], f"unknown pooling mode {pooling!r}")
    _require(values.ndim >= 2, "pooled feature row needs spatial and channel axes")
    flat = values.reshape(-1, values.shape[-1])
    if mask is None:
        selected = flat
    else:
        mask = np.asarray(mask)
        _require(mask.dtype == np.bool_, "feature mask must be boolean")
        _require(mask.shape == values.shape[:-1], "feature mask shape mismatch")
        selected = flat[mask.reshape(-1)]
    _require(len(selected) > 0, "feature mask selects no positions")
    if pooling == "ordered_flatten":
        output = selected.reshape(-1)
    elif pooling == "mean":
        output = selected.mean(axis=0)
    elif pooling == "max":
        output = selected.max(axis=0)
    else:
        output = np.concatenate((selected.mean(axis=0), selected.max(axis=0)))
    return np.asarray(output, dtype=np.float32)


def materialize_input_vectors(
    store: OpaqueFeatureStore,
    source: str,
    views: Sequence[FeatureView],
    sequence_vectors: np.ndarray | None,
) -> np.ndarray:
    """Pool an archive once, avoiding repeated mmap scans during training.

    A full LibB RDE context is roughly a gigabyte.  Pooling inside
    ``Dataset.__getitem__`` would reread it on every epoch.  This function
    instead builds one compact float32 matrix (normally tens of megabytes)
    before cross-validation starts.
    """

    _require(source in {SEQUENCE_SOURCE, FEATURE_SOURCE}, "unknown model source")
    if source == SEQUENCE_SOURCE:
        _require(not views, "sequence control cannot inspect frozen feature views")
        _require(sequence_vectors is not None, "sequence control lacks vectors")
        _require(
            sequence_vectors.shape == (len(store.row_ids), SEQUENCE_CONTROL_DIM),
            "sequence-control array shape changed",
        )
        return np.asarray(sequence_vectors, dtype=np.float32)
    _require(sequence_vectors is None, "frozen-feature model cannot inspect sequence control")
    _require(views, "frozen-feature model has no views")
    width = sum(pooled_view_dimension(store, view) for view in views)
    output = np.empty((len(store.row_ids), width), dtype=np.float32)
    for row_index in range(len(store.row_ids)):
        pieces = []
        for view in views:
            mask = None if view.mask is None else store.arrays[view.mask][row_index]
            pieces.append(
                pool_feature_row(store.arrays[view.array][row_index], view.pooling, mask)
            )
        output[row_index] = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    _require(bool(np.isfinite(output).all()), "materialized input contains non-finite values")
    return output


class MatchedInputDataset(Dataset):
    """Rows for either opaque frozen features or the seven-site control."""

    def __init__(
        self,
        *,
        store: OpaqueFeatureStore,
        cache_indices: Sequence[int] | np.ndarray,
        row_ids: Sequence[str],
        source: str,
        views: Sequence[FeatureView] = (),
        sequence_vectors: np.ndarray | None = None,
        materialized_vectors: np.ndarray | None = None,
        labels: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.store = store
        self.cache_indices = np.asarray(cache_indices, dtype=np.int64)
        self.row_ids = tuple(map(str, row_ids))
        self.source = str(source)
        self.views = tuple(views)
        self.sequence_vectors = sequence_vectors
        self.materialized_vectors = materialized_vectors
        _require(len(self.cache_indices) == len(self.row_ids), "dataset index length changed")
        _require(
            bool((self.cache_indices >= 0).all() and (self.cache_indices < len(store.row_ids)).all()),
            "dataset cache index is out of range",
        )
        _require(self.source in {SEQUENCE_SOURCE, FEATURE_SOURCE}, "unknown model source")
        if materialized_vectors is not None:
            _require(
                materialized_vectors.ndim == 2
                and materialized_vectors.shape[0] == len(store.row_ids),
                "materialized input matrix shape changed",
            )
            _require(
                np.issubdtype(materialized_vectors.dtype, np.floating),
                "materialized inputs must be floating point",
            )
            self.input_dim = int(materialized_vectors.shape[1])
        elif self.source == SEQUENCE_SOURCE:
            _require(not self.views, "sequence control cannot inspect frozen feature views")
            _require(sequence_vectors is not None, "sequence control lacks vectors")
            _require(
                sequence_vectors.shape == (len(store.row_ids), SEQUENCE_CONTROL_DIM),
                "sequence-control array shape changed",
            )
            self.input_dim = SEQUENCE_CONTROL_DIM
        else:
            _require(self.views, "frozen-feature model has no views")
            _require(sequence_vectors is None, "frozen-feature model cannot inspect sequence control")
            self.input_dim = sum(pooled_view_dimension(store, view) for view in self.views)
        if labels is None:
            self.labels = None
        else:
            observed = np.asarray(labels, dtype=np.int64)
            _require(observed.shape == (len(self.row_ids),), "dataset label shape changed")
            _require(bool(np.isin(observed, (0, 1)).all()), "dataset labels are not binary")
            self.labels = observed

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        cache_index = int(self.cache_indices[index])
        if self.materialized_vectors is not None:
            vector = np.array(
                self.materialized_vectors[cache_index], dtype=np.float32, copy=True
            )
        elif self.source == SEQUENCE_SOURCE:
            if self.sequence_vectors is None:  # defensive narrowing
                raise RuntimeError("sequence vectors unexpectedly absent")
            vector = np.array(self.sequence_vectors[cache_index], dtype=np.float32, copy=True)
        else:
            pieces = []
            for view in self.views:
                mask = None if view.mask is None else self.store.arrays[view.mask][cache_index]
                pieces.append(
                    pool_feature_row(
                        self.store.arrays[view.array][cache_index], view.pooling, mask
                    )
                )
            vector = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        item: dict[str, object] = {
            "input_vector": np.asarray(vector, dtype=np.float32),
            "row_id": self.row_ids[index],
        }
        if self.labels is not None:
            item["label"] = np.float32(self.labels[index])
        return item


class FrozenInjectiveAdapter(nn.Module):
    """Map an input to a shared width without adding trainable parameters."""

    def __init__(self, input_dim: int, output_dim: int, seed: int) -> None:
        super().__init__()
        _require(input_dim >= 1, "input dimension must be positive")
        _require(output_dim >= input_dim, "adapter width must not compress an input")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        # Assign every output coordinate to exactly one input coordinate.  Each
        # input occurs at least once because output_dim >= input_dim.  Normalized
        # signed repetitions give orthonormal rows (and therefore an injective
        # map) without performing a costly QR decomposition for every CV fold.
        assignment = torch.arange(output_dim, dtype=torch.long) % input_dim
        assignment = assignment[torch.randperm(output_dim, generator=generator)]
        sign = torch.where(
            torch.rand(output_dim, generator=generator) < 0.5,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        counts = torch.bincount(assignment, minlength=input_dim).to(torch.float32)
        scale = sign / torch.sqrt(counts[assignment])
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        # Keep the sparse one-source-per-output map in indexed form.  A dense
        # [input_dim, output_dim] multiplication would perform billions of
        # useless zero multiplies over one LibB epoch.
        self.register_buffer("assignment", assignment, persistent=True)
        self.register_buffer("scale", scale, persistent=True)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"input must have shape [B,{self.input_dim}], found {tuple(values.shape)}"
            )
        return values[:, self.assignment] * self.scale


class CapacityMatchedNonlinearReadout(nn.Module):
    """Shared two-hidden-layer classifier used by every comparison arm."""

    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        projection_seed: int,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(map(int, hidden_dims))
        _require(len(hidden_dims) == 2 and min(hidden_dims) >= 1, "exactly two positive hidden widths are required")
        _require(0.0 <= float(dropout) < 1.0, "dropout must be in [0,1)")
        self.adapter = FrozenInjectiveAdapter(input_dim, int(adapter_dim), projection_seed)
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(adapter_dim)),
            nn.Linear(int(adapter_dim), hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, input_vector: Tensor) -> Tensor:
        return self.classifier(self.adapter(input_vector.float())).squeeze(-1)


class NativeProjectionNonlinearReadout(nn.Module):
    """Learn a compact representation directly from a native feature vector.

    Unlike :class:`CapacityMatchedNonlinearReadout`, this readout does not copy,
    shuffle, or sign-flip input coordinates.  It normalizes the native vector
    and learns one dense projection into the first hidden width.  Consequently,
    parameter counts honestly depend on ``input_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dims = tuple(map(int, hidden_dims))
        _require(input_dim >= 1, "input dimension must be positive")
        _require(
            len(hidden_dims) == 2 and min(hidden_dims) >= 1,
            "exactly two positive hidden widths are required",
        )
        _require(0.0 <= float(dropout) < 1.0, "dropout must be in [0,1)")
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dims[0])
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(hidden_dims[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, input_vector: Tensor) -> Tensor:
        if input_vector.ndim != 2 or input_vector.shape[-1] != self.input_dim:
            raise ValueError(
                f"input must have shape [B,{self.input_dim}], "
                f"found {tuple(input_vector.shape)}"
            )
        values = self.input_norm(input_vector.float())
        return self.classifier(self.projection(values)).squeeze(-1)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def configured_readout_mode(architecture: Mapping[str, Any]) -> str:
    """Return the explicit mode, defaulting old configs to their legacy path."""

    raw_mode = architecture.get("readout_mode")
    mode = FROZEN_INJECTIVE_READOUT if raw_mode is None else str(raw_mode)
    _require(mode in READOUT_MODES, f"unknown readout mode {mode!r}")
    return mode


def build_matched_readout(
    input_dim: int, architecture: Mapping[str, Any]
) -> nn.Module:
    mode = configured_readout_mode(architecture)
    if mode == NATIVE_LEARNED_READOUT:
        return NativeProjectionNonlinearReadout(
            input_dim=input_dim,
            hidden_dims=architecture["hidden_dims"],
            dropout=float(architecture["dropout"]),
        )
    return CapacityMatchedNonlinearReadout(
        input_dim=input_dim,
        adapter_dim=int(architecture["adapter_dim"]),
        hidden_dims=architecture["hidden_dims"],
        dropout=float(architecture["dropout"]),
        projection_seed=int(architecture["projection_seed"]),
    )


def class_balanced_weights(labels: Tensor | np.ndarray | Sequence[int]) -> Tensor:
    """Return the locked N/(2*N_class) weights for class-weighted BCE."""

    values = torch.as_tensor(labels, dtype=torch.long).flatten()
    _require(values.numel() > 0, "labels are empty")
    _require(bool(torch.all((values == 0) | (values == 1))), "labels must be binary")
    counts = torch.bincount(values, minlength=2).to(torch.float64)
    _require(bool((counts > 0).all()), "both classes must be present")
    return (values.numel() / (2.0 * counts)).to(torch.float32)


def class_weighted_binary_cross_entropy(
    logits: Tensor, labels: Tensor, class_weights: Tensor
) -> Tensor:
    labels = labels.to(dtype=logits.dtype)
    _require(logits.shape == labels.shape, "logits and labels have different shapes")
    _require(tuple(class_weights.shape) == (2,), "class weights must have shape [2]")
    weights = class_weights.to(logits.device)[labels.long()]
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
