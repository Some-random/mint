"""Small frozen-feature readouts for canonical LibA or LibB experiments.

The folding model is a feature extractor in this experiment.  This module does
not run ESMFold2, predict coordinates, attach physical meanings to its 64
distogram categories, or read retention measurements.  It consumes four
precomputed arrays:

``distogram_probabilities``
    ``[N, 9, 58, 64]`` categorical probabilities from the public checkpoint's
    distogram head.  The public checkpoint metadata does not define physical
    bin edges, so the categories are learned as categories only.
``pair_states_symmetric``
    ``[N, 9, 58, 256]`` obtained by averaging the peptide-to-Affibody and the
    aligned Affibody-to-peptide trunk states.
``single_inputs_peptide`` and ``single_inputs_affibody``
    ``[N, 9, 451]`` and ``[N, 58, 451]`` residue representations.

Every pair-valued model uses the same position-aware attention pool.  That
keeps comparisons matched: only the frozen input modality changes.  The
readouts are deliberately small and output one binary logit for the current
pair's selection-derived weak label.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


PEPTIDE_LENGTH = 9
AFFIBODY_LENGTH = 58
DISTOGRAM_CATEGORIES = 64
PAIR_STATE_DIM = 256
SINGLE_INPUT_DIM = 451

MODEL_NAMES = (
    "distogram_only",
    "pair_state_only",
    "distogram_pair",
    "single_inputs_only",
    "full",
)

FEATURE_SHAPES = {
    "distogram_probabilities": (
        PEPTIDE_LENGTH,
        AFFIBODY_LENGTH,
        DISTOGRAM_CATEGORIES,
    ),
    "pair_states_symmetric": (
        PEPTIDE_LENGTH,
        AFFIBODY_LENGTH,
        PAIR_STATE_DIM,
    ),
    "single_inputs_peptide": (PEPTIDE_LENGTH, SINGLE_INPUT_DIM),
    "single_inputs_affibody": (AFFIBODY_LENGTH, SINGLE_INPUT_DIM),
}

MODEL_FEATURES = {
    "distogram_only": ("distogram_probabilities",),
    "pair_state_only": ("pair_states_symmetric",),
    "distogram_pair": (
        "distogram_probabilities",
        "pair_states_symmetric",
    ),
    "single_inputs_only": (
        "single_inputs_peptide",
        "single_inputs_affibody",
    ),
    "full": tuple(FEATURE_SHAPES),
}

# Feature metadata is an ordering/index file, not a source of outcomes.  A
# header-only check prevents accidentally materializing sealed labels even if
# somebody points the trainer at the wrong CSV.
FORBIDDEN_METADATA_COLUMN_PARTS = (
    "retention",
    "label",
    "outcome",
    "target",
)
MERGED_CACHE_SCHEMA_VERSION = "esmfold2-libb-feature-merge-v1"
LIBA_MERGED_CACHE_SCHEMA_VERSION = "esmfold2-liba-feature-merge-v1"
ACCEPTED_MERGED_CACHE_SCHEMAS = {
    MERGED_CACHE_SCHEMA_VERSION,
    LIBA_MERGED_CACHE_SCHEMA_VERSION,
}
CACHE_ARRAY_SHAPES = {
    "distogram_probabilities": FEATURE_SHAPES["distogram_probabilities"],
    "pair_states_symmetric": FEATURE_SHAPES["pair_states_symmetric"],
    "single_inputs": (PEPTIDE_LENGTH + AFFIBODY_LENGTH, SINGLE_INPUT_DIM),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_feature_names(model_name: str) -> tuple[str, ...]:
    """Return only the arrays a model is allowed to inspect."""

    _require(model_name in MODEL_FEATURES, f"unknown model {model_name!r}")
    return MODEL_FEATURES[model_name]


class PositionAwareAttentionPool(nn.Module):
    """Pool a 9-by-58 residue-pair grid without assigning distances to bins.

    A learned projection produces one value vector for each residue pair.
    Learned peptide and Affibody position embeddings affect the attention
    scores, while the pooled values remain functions of the input features.
    Attention-weighted, mean, and max summaries are concatenated so that a
    model is not forced to trust one pooling statistic.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        _require(input_dim >= 1, "input_dim must be positive")
        _require(hidden_dim >= 1, "hidden_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.value_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.peptide_position = nn.Embedding(PEPTIDE_LENGTH, self.hidden_dim)
        self.affibody_position = nn.Embedding(AFFIBODY_LENGTH, self.hidden_dim)
        self.attention_score = nn.Linear(self.hidden_dim, 1, bias=False)

    @property
    def output_dim(self) -> int:
        return 3 * self.hidden_dim

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        expected = (
            PEPTIDE_LENGTH,
            AFFIBODY_LENGTH,
            self.input_dim,
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                "pair features must have shape [B, {}, {}, {}], found {}".format(
                    *expected, tuple(features.shape)
                )
            )
        values = F.gelu(self.value_projection(self.input_norm(features)))
        peptide_position = self.peptide_position.weight[:, None, :]
        affibody_position = self.affibody_position.weight[None, :, :]
        score_features = torch.tanh(
            values + peptide_position + affibody_position
        )
        scores = self.attention_score(score_features).flatten(start_dim=1, end_dim=2)
        attention = torch.softmax(scores, dim=1).view(
            features.shape[0], PEPTIDE_LENGTH, AFFIBODY_LENGTH
        )
        weighted = torch.sum(values * attention.unsqueeze(-1), dim=(1, 2))
        mean = torch.mean(values, dim=(1, 2))
        maximum = torch.amax(values, dim=(1, 2))
        return torch.cat((weighted, mean, maximum), dim=-1), attention


class SingleInputPairInteraction(nn.Module):
    """Turn two per-residue arrays into a low-rank residue-pair grid."""

    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        _require(output_dim >= 1, "single interaction dimension must be positive")
        self.output_dim = int(output_dim)
        self.peptide_projection = nn.Linear(SINGLE_INPUT_DIM, self.output_dim)
        self.affibody_projection = nn.Linear(SINGLE_INPUT_DIM, self.output_dim)

    def forward(self, peptide: Tensor, affibody: Tensor) -> Tensor:
        if peptide.ndim != 3 or tuple(peptide.shape[1:]) != (
            PEPTIDE_LENGTH,
            SINGLE_INPUT_DIM,
        ):
            raise ValueError(
                f"peptide single inputs have the wrong shape: {tuple(peptide.shape)}"
            )
        if affibody.ndim != 3 or tuple(affibody.shape[1:]) != (
            AFFIBODY_LENGTH,
            SINGLE_INPUT_DIM,
        ):
            raise ValueError(
                f"Affibody single inputs have the wrong shape: {tuple(affibody.shape)}"
            )
        if peptide.shape[0] != affibody.shape[0]:
            raise ValueError("peptide and Affibody batch sizes differ")
        peptide_values = F.gelu(self.peptide_projection(peptide))
        affibody_values = F.gelu(self.affibody_projection(affibody))
        return peptide_values[:, :, None, :] * affibody_values[:, None, :, :]


class EsmFold2PairReadout(nn.Module):
    """One of five matched readouts over frozen ESMFold2 features."""

    def __init__(
        self,
        model_name: str,
        pair_hidden_dim: int = 32,
        single_interaction_dim: int = 32,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        _require(model_name in MODEL_NAMES, f"unknown model {model_name!r}")
        _require(classifier_hidden_dim >= 1, "classifier hidden dim must be positive")
        _require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        self.model_name = model_name
        self.required_features = model_feature_names(model_name)
        self.single_interaction = (
            SingleInputPairInteraction(single_interaction_dim)
            if model_name in ("single_inputs_only", "full")
            else None
        )

        input_dim = 0
        if model_name in ("distogram_only", "distogram_pair", "full"):
            input_dim += DISTOGRAM_CATEGORIES
        if model_name in ("pair_state_only", "distogram_pair", "full"):
            input_dim += PAIR_STATE_DIM
        if model_name in ("single_inputs_only", "full"):
            input_dim += int(single_interaction_dim)

        self.pool = PositionAwareAttentionPool(input_dim, pair_hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.pool.output_dim),
            nn.Linear(self.pool.output_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    @staticmethod
    def _required(value: Tensor | None, name: str) -> Tensor:
        if value is None:
            raise ValueError(f"model requires {name}")
        return value

    def forward(
        self,
        *,
        distogram_probabilities: Tensor | None = None,
        pair_states_symmetric: Tensor | None = None,
        single_inputs_peptide: Tensor | None = None,
        single_inputs_affibody: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        pair_features: list[Tensor] = []
        if self.model_name in ("distogram_only", "distogram_pair", "full"):
            pair_features.append(
                self._required(
                    distogram_probabilities, "distogram_probabilities"
                )
            )
        if self.model_name in ("pair_state_only", "distogram_pair", "full"):
            pair_features.append(
                self._required(pair_states_symmetric, "pair_states_symmetric")
            )
        if self.model_name in ("single_inputs_only", "full"):
            if self.single_interaction is None:  # defensive type narrowing
                raise RuntimeError("single interaction module is missing")
            pair_features.append(
                self.single_interaction(
                    self._required(single_inputs_peptide, "single_inputs_peptide"),
                    self._required(single_inputs_affibody, "single_inputs_affibody"),
                )
            )
        features = (
            pair_features[0]
            if len(pair_features) == 1
            else torch.cat(pair_features, dim=-1)
        )
        pooled, attention = self.pool(features.float())
        logits = self.classifier(pooled).squeeze(-1)
        return (logits, attention) if return_attention else logits


EsmFold2LibBReadout = EsmFold2PairReadout


def build_readout(model_name: str, architecture: Mapping[str, object]) -> EsmFold2PairReadout:
    """Construct a readout from the shared architecture section of a config."""

    return EsmFold2PairReadout(
        model_name=model_name,
        pair_hidden_dim=int(architecture["pair_hidden_dim"]),
        single_interaction_dim=int(architecture["single_interaction_dim"]),
        classifier_hidden_dim=int(architecture["classifier_hidden_dim"]),
        dropout=float(architecture["dropout"]),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parameter_report(
    architecture: Mapping[str, object],
    model_names: Sequence[str] = MODEL_NAMES,
) -> dict[str, int]:
    return {
        name: trainable_parameter_count(build_readout(name, architecture))
        for name in model_names
    }


def class_balanced_weights(labels: Tensor | np.ndarray | Sequence[int]) -> Tensor:
    """Return ``N/(2*N_class)`` weights for class-weighted BCE."""

    values = torch.as_tensor(labels, dtype=torch.long).flatten()
    if values.numel() == 0 or not bool(torch.all((values == 0) | (values == 1))):
        raise ValueError("labels must be a nonempty binary vector")
    counts = torch.bincount(values, minlength=2).to(torch.float64)
    if bool((counts == 0).any()):
        raise ValueError("both classes must be present")
    return (values.numel() / (2.0 * counts)).to(torch.float32)


def class_weighted_binary_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor,
) -> Tensor:
    labels = labels.to(dtype=logits.dtype)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    if tuple(class_weights.shape) != (2,):
        raise ValueError("class_weights must have shape [2]")
    example_weights = class_weights.to(logits.device)[labels.long()]
    return F.binary_cross_entropy_with_logits(
        logits, labels, weight=example_weights, reduction="mean"
    )


@dataclass(frozen=True)
class FrozenFeatureStore:
    """Memory-mapped full-cache arrays plus their label-free UID ordering."""

    root: Path
    row_ids: tuple[str, ...]
    splits: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]

    @classmethod
    def open(
        cls,
        root: str | Path,
        required_features: Iterable[str] = FEATURE_SHAPES,
        *,
        validate_distogram_rows: int = 32,
        verify_all_checksums: bool = False,
    ) -> "FrozenFeatureStore":
        root = Path(root).resolve()
        _require(root.is_dir(), f"feature cache directory does not exist: {root}")
        _require(
            not (root / "MERGE_INCOMPLETE").exists(),
            "feature cache merge is explicitly marked incomplete",
        )
        manifest_path = root / "merge_complete.json"
        _require(manifest_path.is_file(), f"missing {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _require(
            manifest.get("schema_version") in ACCEPTED_MERGED_CACHE_SCHEMAS,
            "merged feature-cache manifest schema is not an accepted LibA/LibB contract",
        )
        _require(
            manifest.get("supervision_fields_read") == [],
            "feature-cache merge was not label-free",
        )
        output_contracts = manifest.get("arrays")
        _require(isinstance(output_contracts, dict), "feature-cache output contract missing")
        metadata_path = root / "metadata.csv"
        _require(metadata_path.is_file(), f"missing {metadata_path}")
        metadata_contract = manifest.get("metadata", {})
        _require(metadata_contract.get("file") == "metadata.csv", "feature metadata filename changed")
        _require(
            metadata_contract.get("bytes") == metadata_path.stat().st_size,
            "feature metadata byte size changed",
        )
        _require(
            metadata_contract.get("sha256") == _sha256_file(metadata_path),
            "feature metadata checksum changed",
        )
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
        _require(
            tuple(header) == ("row_index", "row_id", "split"),
            "feature metadata must contain exactly row_index,row_id,split",
        )
        _require(metadata_contract.get("columns") == header, "metadata manifest columns changed")
        lowered = tuple(column.lower() for column in header)
        forbidden = sorted(
            column
            for column in lowered
            if any(part in column for part in FORBIDDEN_METADATA_COLUMN_PARTS)
        )
        _require(
            not forbidden,
            "feature metadata must be label-free; forbidden columns: "
            + ", ".join(forbidden),
        )
        row_id_position = header.index("row_id")
        row_index_position = header.index("row_index")
        split_position = header.index("split")
        row_ids: list[str] = []
        splits: list[str] = []
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            for row_number, row in enumerate(reader, start=2):
                _require(
                    len(row) == len(header),
                    f"metadata row {row_number} has the wrong number of columns",
                )
                expected_index = row_number - 2
                _require(
                    row[row_index_position] == str(expected_index),
                    "feature metadata is not in canonical row_index order",
                )
                row_ids.append(row[row_id_position])
                splits.append(row[split_position])
        _require(row_ids, "feature cache is empty")
        _require(all(row_ids), "feature metadata contains an empty row ID")
        _require(len(set(row_ids)) == len(row_ids), "duplicate feature row ID")
        _require(set(splits).issubset({"train", "eval"}), "unknown feature split")
        _require(set(splits) == {"train", "eval"}, "feature cache must contain train and eval rows")
        _require(manifest.get("row_count") == len(row_ids), "feature-cache row count changed")
        _require(metadata_contract.get("rows") == len(row_ids), "metadata row count changed")
        _require(
            manifest.get("row_indices_sha256")
            == _canonical_json_sha256(list(range(len(row_ids)))),
            "canonical row-index digest changed",
        )
        _require(
            manifest.get("row_ids_sha256") == _canonical_json_sha256(row_ids),
            "canonical row-ID digest changed",
        )

        feature_names = tuple(dict.fromkeys(required_features))
        unknown = sorted(set(feature_names).difference(FEATURE_SHAPES))
        _require(not unknown, f"unknown requested features: {unknown}")
        arrays: dict[str, np.ndarray] = {}
        cache_names = tuple(
            dict.fromkeys(
                "single_inputs"
                if name in ("single_inputs_peptide", "single_inputs_affibody")
                else name
                for name in feature_names
            )
        )
        cache_arrays: dict[str, np.ndarray] = {}
        for name in cache_names:
            path = root / f"{name}.npy"
            _require(path.is_file(), f"missing feature array {path}")
            contract = output_contracts.get(name, {})
            _require(contract.get("file") == path.name, f"{name} filename changed")
            _require(contract.get("bytes") == path.stat().st_size, f"{name} byte size changed")
            _require(
                contract.get("shape") == [len(row_ids), *CACHE_ARRAY_SHAPES[name]],
                f"{name} manifest shape changed",
            )
            _require(contract.get("dtype") == "float16", f"{name} manifest dtype changed")
            if verify_all_checksums:
                _require(contract.get("sha256") == _sha256_file(path), f"{name} checksum changed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected = (len(row_ids), *CACHE_ARRAY_SHAPES[name])
            _require(
                tuple(array.shape) == expected,
                f"{name} has shape {array.shape}; expected {expected}",
            )
            _require(
                np.issubdtype(array.dtype, np.floating),
                f"{name} must have a floating dtype",
            )
            cache_arrays[name] = array

        for name in feature_names:
            if name == "single_inputs_peptide":
                arrays[name] = cache_arrays["single_inputs"][:, :PEPTIDE_LENGTH, :]
            elif name == "single_inputs_affibody":
                arrays[name] = cache_arrays["single_inputs"][:, PEPTIDE_LENGTH:, :]
            else:
                arrays[name] = cache_arrays[name]

        if "distogram_probabilities" in arrays and validate_distogram_rows:
            count = min(int(validate_distogram_rows), len(row_ids))
            sample_indices = np.linspace(0, len(row_ids) - 1, count, dtype=int)
            sample = np.asarray(
                arrays["distogram_probabilities"][sample_indices], dtype=np.float32
            )
            _require(bool(np.isfinite(sample).all()), "nonfinite distogram probabilities")
            _require(bool((sample >= -1e-5).all()), "negative distogram probability")
            _require(
                bool(np.allclose(sample.sum(axis=-1), 1.0, atol=2e-3, rtol=2e-3)),
                "distogram categories do not sum to one",
            )
        return cls(
            root=root,
            row_ids=tuple(row_ids),
            splits=tuple(splits),
            arrays=arrays,
        )

    @property
    def row_id_to_index(self) -> dict[str, int]:
        return {row_id: index for index, row_id in enumerate(self.row_ids)}


class FrozenFeatureDataset(Dataset):
    """Dataset view that performs no global in-memory concatenation."""

    def __init__(
        self,
        store: FrozenFeatureStore,
        cache_indices: Sequence[int] | np.ndarray,
        row_ids: Sequence[str],
        required_features: Sequence[str],
        labels: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        self.store = store
        self.cache_indices = np.asarray(cache_indices, dtype=np.int64)
        self.row_ids = tuple(str(value) for value in row_ids)
        self.required_features = tuple(required_features)
        _require(
            len(self.cache_indices) == len(self.row_ids),
            "dataset index and UID lengths differ",
        )
        _require(
            bool(
                (self.cache_indices >= 0).all()
                and (self.cache_indices < len(store.row_ids)).all()
            ),
            "cache index is out of range",
        )
        for name in self.required_features:
            _require(name in store.arrays, f"store did not open required feature {name}")
        if labels is None:
            self.labels = None
        else:
            label_array = np.asarray(labels, dtype=np.int64)
            _require(label_array.shape == (len(self.row_ids),), "label shape mismatch")
            _require(bool(np.isin(label_array, (0, 1)).all()), "labels must be binary")
            self.labels = label_array

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        cache_index = int(self.cache_indices[index])
        item: dict[str, object] = {
            # Copy the source float16 row to writable host memory, but leave its
            # exact stored dtype unchanged.  ``_model_inputs`` performs the
            # required float32 conversion on the destination device.  Casting
            # here instead made every epoch repeat a large CPU conversion and
            # quadrupled loader time without changing a single model value.
            name: np.array(self.store.arrays[name][cache_index], copy=True)
            for name in self.required_features
        }
        item["row_id"] = self.row_ids[index]
        if self.labels is not None:
            item["label"] = np.float32(self.labels[index])
        return item
