#!/usr/bin/env python
"""Validate and stream selected frozen-MINT scores over Affibody candidates.

The supported deployment contracts are LibB/layer 5, LibA/layer 9, and the
LibA/layer-33 control.  All use the mean residue representation for each partner
(1,280 values per chain, concatenated to 2,560 values).  Ordinary MINT
inference evaluates all 33 transformer layers and the language-model head even
when an intermediate layer is requested.  This utility installs a forward hook
on the requested transformer block and deliberately stops immediately after
that block.  The ``parity`` command proves, on all known evaluation rows, that
the early-stop representation is identical to ordinary inference and that the
serialized deployment head reproduces the previously frozen scores.

Two commands are intentionally separate::

    parity  # write a model/head-specific parity receipt
    score   # stream one candidate Parquet partition, requiring that receipt

Run one ``score`` process per peptide partition to distribute files over
several GPUs.  Candidate features are never retained: each batch is pooled,
scored, and discarded.  The output contains stable IDs plus the scalar logit
and score, not a multi-terabyte feature cache.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.extract_mint_multilayer_cache import (  # noqa: E402
    pool_representations,
    residue_masks,
)
from downstream.AffibodyMHC.code_only_baseline import opaque_id  # noqa: E402
from mint.helpers.extract import CollateFn, MINTWrapper, load_config  # noqa: E402


RESIDUE_DIMENSION = 1280
FEATURE_DIMENSION = 2 * RESIDUE_DIMENSION
CHAIN_LENGTHS = (270, 58)
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
POOLING_DESCRIPTION = (
    "exclude cls/eos/padding; mean residues separately within chain_id 0 and "
    "chain_id 1; concatenate chain 0 then chain 1"
)
INPUT_COLUMNS = (
    "pair_uid",
    "library",
    "peptide_design_code",
    "peptide_full_sequence",
    "affibody_design_code",
    "peptide_uid",
    "affibody_uid",
    "chain1_sha256",
    "chain2_sha256",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
)
OUTPUT_ID_COLUMNS = INPUT_COLUMNS[:9]
LIBA_HANDOFF_FLAG_COLUMNS = (
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "mint.ckpt"
DEFAULT_CONFIG = REPO_ROOT / "data" / "esm2_t33_650M_UR50D.json"
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    (
        "retention",
        "target_retention",
        "target_binder",
        "binder_label",
        "weak_label",
        "label",
    )
)
FORBIDDEN_REFERENCE_COLUMN_FRAGMENTS = (
    "retention",
    "target",
    "binder",
    "binding_label",
    "ground_truth",
    "measurement",
    "wetlab",
    "outcome",
)


@dataclass(frozen=True)
class ScoringContract:
    """Immutable description of one selected deployment representation."""

    library: str
    layer: int
    evaluation_rows: int
    affibody_code_indices: tuple
    canonical_rows: Path
    reference_scores: Path
    head_npz: Path
    reference_model: str
    reference_score_column: str
    model_id: str
    selected_c: float
    head_schema_version: str
    training_membership_sha256: str

    @property
    def affibody_code_length(self):
        return len(self.affibody_code_indices)

    @property
    def feature_name(self):
        return "mint_layer_{:02d}_chain_mean".format(self.layer)

    @property
    def schema_version(self):
        return "mint-layer{}-streaming-candidate-scores-v1".format(self.layer)

    @property
    def parity_schema_version(self):
        return "mint-layer{}-early-stop-parity-v1".format(self.layer)

    @property
    def logit_column(self):
        return "mint_layer{}_logit".format(self.layer)

    @property
    def score_column(self):
        return "mint_layer{}_score".format(self.layer)

    @property
    def token_mapping(self):
        """Map zero-based sequence indices to concatenated MINT token indices."""

        chain1_residue_offset = 1  # leading <cls>
        chain2_residue_offset = CHAIN_LENGTHS[0] + 3  # chain1 cls/eos + chain2 cls
        return {
            "chain_lengths": list(CHAIN_LENGTHS),
            "peptide_code_chain1_python_indices": [264, 265],
            "peptide_code_mint_token_indices": [
                chain1_residue_offset + index for index in (264, 265)
            ],
            "affibody_code_python_indices": list(self.affibody_code_indices),
            "affibody_code_mint_token_indices": [
                chain2_residue_offset + index
                for index in self.affibody_code_indices
            ],
            "special_tokens_per_chain": ["<cls>", "<eos>"],
        }


LIBB_LAYER5_CONTRACT = ScoringContract(
    library="LibB",
    layer=5,
    evaluation_rows=120,
    affibody_code_indices=(5, 9, 12, 13, 16),
    canonical_rows=(
        REPO_ROOT
        / "private_data"
        / "derived"
        / "esmfold2_libb_canonical_rows_provider_revision_120_v1"
        / "rows.json"
    ),
    reference_scores=(
        REPO_ROOT
        / "private_data"
        / "experiments"
        / "libb_common_oof_base_predictions_v3_deployable"
        / "final_120_predictions.csv"
    ),
    head_npz=(
        REPO_ROOT
        / "private_data"
        / "experiments"
        / "libb_common_oof_base_predictions_v3_deployable"
        / "deployment_heads.npz"
    ),
    reference_model="frozen_mint_layer5",
    reference_score_column="score",
    model_id="frozen_mint_layer5",
    selected_c=0.1,
    head_schema_version="libb-additive-mint-deployment-heads-v1",
    training_membership_sha256=(
        "1a9a0527c5f9f0e2bbeff6a0ea3d9afaaeadace894d46e73311b049a19f8a393"
    ),
)

LIBA_LAYER9_CONTRACT = ScoringContract(
    library="LibA",
    layer=9,
    evaluation_rows=108,
    # Displayed sequence positions 13/17/27/31 are Python indices 12/16/26/30.
    affibody_code_indices=(12, 16, 26, 30),
    canonical_rows=(
        REPO_ROOT
        / "private_data"
        / "derived"
        / "esmfold2_liba_canonical_rows_v1"
        / "rows.json"
    ),
    reference_scores=(
        REPO_ROOT
        / "private_data"
        / "experiments"
        / "liba_common_oof_sequence_predictions_v1"
        / "evaluation_predictions.csv"
    ),
    head_npz=(
        REPO_ROOT
        / "private_data"
        / "experiments"
        / "liba_common_oof_sequence_predictions_v1"
        / "mint_layer9_deployment_head.npz"
    ),
    reference_model="frozen_mint_layer9",
    reference_score_column="score",
    model_id="mint_l9",
    selected_c=0.01,
    head_schema_version="liba-mint-layer9-deployment-head-v1",
    training_membership_sha256=(
        "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e"
    ),
)

LIBA_LAYER33_CONTRACT = ScoringContract(
    library="LibA",
    layer=33,
    evaluation_rows=108,
    affibody_code_indices=(12, 16, 26, 30),
    canonical_rows=LIBA_LAYER9_CONTRACT.canonical_rows,
    reference_scores=LIBA_LAYER9_CONTRACT.reference_scores,
    head_npz=(
        REPO_ROOT
        / "private_data"
        / "experiments"
        / "liba_common_oof_sequence_predictions_v1"
        / "mint_layer33_deployment_head.npz"
    ),
    reference_model="frozen_mint_layer33_control",
    reference_score_column="score",
    model_id="mint_l33",
    selected_c=0.01,
    head_schema_version="liba-mint-layer33-deployment-head-v1",
    training_membership_sha256=LIBA_LAYER9_CONTRACT.training_membership_sha256,
)

SUPPORTED_CONTRACTS = {
    (contract.library, contract.layer): contract
    for contract in (
        LIBB_LAYER5_CONTRACT,
        LIBA_LAYER9_CONTRACT,
        LIBA_LAYER33_CONTRACT,
    )
}


def resolve_scoring_contract(library="LibB", layer=None):
    library = str(library)
    if layer is None:
        layer = 5 if library == "LibB" else 9 if library == "LibA" else None
    key = (library, int(layer)) if layer is not None else None
    _require(
        key in SUPPORTED_CONTRACTS,
        (
            "unsupported library/layer contract: {}/{}; expected LibB/5, "
            "LibA/9, or LibA/33"
        ).format(library, layer),
    )
    return SUPPORTED_CONTRACTS[key]


def _contract_from_args(args):
    return resolve_scoring_contract(
        getattr(args, "library", "LibB"), getattr(args, "layer", None)
    )


def _arg_or_contract_default(args, name, contract):
    value = getattr(args, name, None)
    return value if value is not None else getattr(contract, name)


def _validate_reference_contract(contract, reference_path, reference_model):
    path = Path(reference_path).resolve()
    if contract.library == "LibA":
        _require(
            path == contract.reference_scores.resolve(),
            "LibA parity must use the locked target-free evaluation_predictions.csv",
        )
        _require(
            reference_model == contract.reference_model,
            "LibA parity reference model override is forbidden",
        )
    return path, reference_model


def _candidate_input_columns(contract):
    if contract.library == "LibA":
        return INPUT_COLUMNS + LIBA_HANDOFF_FLAG_COLUMNS
    return INPUT_COLUMNS


# Backward-compatible LibB/layer-5 aliases used by existing callers and tests.
SCHEMA_VERSION = LIBB_LAYER5_CONTRACT.schema_version
PARITY_SCHEMA_VERSION = LIBB_LAYER5_CONTRACT.parity_schema_version
LAYER = LIBB_LAYER5_CONTRACT.layer
FEATURE_NAME = LIBB_LAYER5_CONTRACT.feature_name
AFFIBODY_CODE_INDICES = LIBB_LAYER5_CONTRACT.affibody_code_indices
DEFAULT_CANONICAL_ROWS = LIBB_LAYER5_CONTRACT.canonical_rows
DEFAULT_REFERENCE_SCORES = LIBB_LAYER5_CONTRACT.reference_scores
DEFAULT_HEAD_NPZ = LIBB_LAYER5_CONTRACT.head_npz
DEFAULT_REFERENCE_MODEL = LIBB_LAYER5_CONTRACT.reference_model


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _atomic_json(path, payload):
    path = Path(path)
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(_private_mode(path) == "0600", "JSON output is not mode 0600")


def _scalar_string(array):
    values = np.asarray(array)
    _require(values.size == 1, "metadata value is not scalar")
    value = values.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_head_key(files, aliases, label, prefix=None):
    if prefix:
        prefixed = []
        for alias in aliases:
            prefixed.extend(("{}_{}".format(prefix, alias), "{}.{}".format(prefix, alias)))
        candidates = [value for value in prefixed if value in files]
    else:
        candidates = [value for value in aliases if value in files]
    _require(bool(candidates), "head NPZ lacks {} (tried {})".format(label, aliases))
    _require(len(candidates) == 1, "head NPZ has ambiguous {} keys: {}".format(label, candidates))
    return candidates[0]


def load_linear_head(path, prefix=None, contract=LIBB_LAYER5_CONTRACT):
    """Load a standardizer plus binary linear head from a deployment NPZ.

    The canonical artifact uses ``mean``, ``scale``, ``coef``, and
    ``intercept``.  ``weight``/``bias`` are accepted so that an explicitly
    supplied, equivalently serialized head can also be used.
    """

    path = Path(path).resolve()
    _require(path.is_file(), "head NPZ does not exist: {}".format(path))
    with np.load(str(path), allow_pickle=False) as archive:
        files = tuple(archive.files)
        if prefix is None and all(
            value in files
            for value in ("mint_mean", "mint_scale", "mint_coef", "mint_intercept")
        ):
            prefix = "mint"
        mean_key = _resolve_head_key(files, ("mean",), "standardizer mean", prefix)
        scale_key = _resolve_head_key(files, ("scale",), "standardizer scale", prefix)
        coefficient_key = _resolve_head_key(
            files, ("coef", "coefficient", "weight"), "linear coefficient", prefix
        )
        intercept_key = _resolve_head_key(
            files, ("intercept", "bias"), "linear intercept", prefix
        )
        mean = np.asarray(archive[mean_key], dtype=np.float64).reshape(-1)
        scale = np.asarray(archive[scale_key], dtype=np.float64).reshape(-1)
        coefficient = np.asarray(archive[coefficient_key], dtype=np.float64).reshape(-1)
        intercept_values = np.asarray(archive[intercept_key], dtype=np.float64).reshape(-1)
        metadata = {}
        for name in ("schema_version", "feature_name", "pooling", "model_name"):
            if name in files:
                metadata[name] = _scalar_string(archive[name])
        feature_keys = (
            ("{}_feature_key".format(prefix), "{}_feature_name".format(prefix))
            if prefix
            else ("feature_name",)
        )
        layer_key = "{}_layer".format(prefix) if prefix else "layer"
        pooling_key = "{}_pooling".format(prefix) if prefix else None
        c_key = "{}_C".format(prefix) if prefix else "C"
        membership_keys = ["training_membership_sha256"]
        if prefix:
            membership_keys.append("{}_training_membership_sha256".format(prefix))
        present_feature_keys = [name for name in feature_keys if name in files]
        if present_feature_keys:
            feature_values = {_scalar_string(archive[name]) for name in present_feature_keys}
            _require(len(feature_values) == 1, "head has conflicting feature-name metadata")
            metadata["feature_name"] = next(iter(feature_values))
        if layer_key in files:
            metadata["layer"] = int(np.asarray(archive[layer_key]).reshape(-1)[0])
        if pooling_key and pooling_key in files:
            metadata["pooling"] = _scalar_string(archive[pooling_key])
        if c_key in files:
            c_values = np.asarray(archive[c_key], dtype=np.float64).reshape(-1)
            _require(c_values.shape == (1,), "head C metadata must be scalar")
            metadata["C"] = float(c_values[0])
        present_membership_keys = [name for name in membership_keys if name in files]
        if present_membership_keys:
            membership_values = {
                _scalar_string(archive[name]) for name in present_membership_keys
            }
            _require(
                len(membership_values) == 1,
                "head has conflicting training-membership metadata",
            )
            metadata["training_membership_sha256"] = next(
                iter(membership_values)
            )

    _require(mean.shape == (FEATURE_DIMENSION,), "head mean must have 2,560 values")
    _require(scale.shape == (FEATURE_DIMENSION,), "head scale must have 2,560 values")
    _require(
        coefficient.shape == (FEATURE_DIMENSION,),
        "head coefficient must have 2,560 values",
    )
    _require(intercept_values.shape == (1,), "head intercept must be scalar")
    _require(bool(np.isfinite(mean).all()), "head mean contains non-finite values")
    _require(bool(np.isfinite(scale).all()), "head scale contains non-finite values")
    _require(bool(np.isfinite(coefficient).all()), "head coefficient is non-finite")
    _require(bool(np.isfinite(intercept_values).all()), "head intercept is non-finite")
    _require(bool((scale > 0.0).all()), "head scale must be positive")
    if "feature_name" in metadata:
        _require(
            metadata["feature_name"] == contract.feature_name,
            "head feature name does not match {}/layer {}".format(
                contract.library, contract.layer
            ),
        )
    if "layer" in metadata:
        _require(
            metadata["layer"] == contract.layer,
            "head is not a layer-{} head".format(contract.layer),
        )
    if "pooling" in metadata:
        _require(metadata["pooling"] == POOLING_DESCRIPTION, "head pooling contract changed")
    if "C" in metadata:
        _require(
            math.isfinite(metadata["C"])
            and metadata["C"] == contract.selected_c,
            "head regularization C does not match selected {} value".format(
                contract.selected_c
            ),
        )
    if contract.library == "LibA":
        _require(
            metadata.get("schema_version") == contract.head_schema_version,
            "LibA head schema version does not match its immutable profile",
        )
        _require(
            metadata.get("training_membership_sha256")
            == contract.training_membership_sha256,
            "LibA head training membership does not match the 22,542-row lock",
        )
    return {
        "path": path,
        "sha256": sha256_file(path),
        "mean": mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": float(intercept_values[0]),
        "keys": {
            "mean": mean_key,
            "scale": scale_key,
            "coefficient": coefficient_key,
            "intercept": intercept_key,
        },
        "metadata": metadata,
        "prefix": prefix,
    }


def score_features(features, head):
    """Reproduce the selected sklearn readout's numerical inference path."""

    values = np.asarray(features, dtype=np.float64)
    _require(
        values.ndim == 2 and values.shape[1] == FEATURE_DIMENSION,
        "features must have shape [N, 2560]",
    )
    _require(bool(np.isfinite(values).all()), "features contain non-finite values")
    # Training uses this float64 standardization followed by an explicit cast to
    # float32 before liblinear inference.  Keeping that cast is important for
    # parity with the saved evaluation predictions.
    standardized = ((values - head["mean"]) / head["scale"]).astype(np.float32)
    logits = np.matmul(standardized, head["coefficient"]) + head["intercept"]
    logits = np.asarray(logits, dtype=np.float64)
    probability = np.empty_like(logits)
    positive = logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_value = np.exp(logits[~positive])
    probability[~positive] = exp_value / (1.0 + exp_value)
    _require(bool(np.isfinite(probability).all()), "head produced non-finite scores")
    return logits, probability


class _LayerCaptured(Exception):
    """Private sentinel used to unwind MINT after the requested transformer."""


class EarlyStopLayerExtractor(object):
    """Capture one MINT layer and prevent all later blocks/LM-head execution."""

    def __init__(self, model, layer=LAYER):
        _require(hasattr(model, "layers"), "MINT model lacks transformer layers")
        _require(1 <= int(layer) <= len(model.layers), "requested layer is absent")
        self.model = model
        self.layer = int(layer)
        self._captured = None
        self._is_final_layer = self.layer == len(model.layers)
        if self._is_final_layer:
            _require(
                hasattr(model, "emb_layer_norm_after"),
                "MINT final layer lacks its output normalization",
            )
            # MINT defines its final representation *after* this normalization.
            # Raising at transformer block 33 would capture the wrong tensor.
            self._handle = model.emb_layer_norm_after.register_forward_hook(
                self._final_norm_hook
            )
        else:
            self._handle = model.layers[self.layer - 1].register_forward_hook(
                self._block_hook
            )

    def _block_hook(self, module, inputs, output):
        del module, inputs
        _require(
            isinstance(output, (tuple, list)) and len(output) >= 1,
            "transformer block returned an unexpected value",
        )
        representation = output[0]
        _require(representation.ndim == 3, "transformer output must have shape [T,B,C]")
        self._captured = representation.transpose(0, 1)
        raise _LayerCaptured()

    def _final_norm_hook(self, module, inputs, output):
        del module, inputs
        _require(
            isinstance(output, torch.Tensor) and output.ndim == 3,
            "final MINT normalization returned an unexpected value",
        )
        self._captured = output.transpose(0, 1)
        raise _LayerCaptured()

    def extract(self, chains, chain_ids):
        _require(self._handle is not None, "early-stop extractor is closed")
        self._captured = None
        try:
            self.model(chains, chain_ids, repr_layers=[])
        except _LayerCaptured:
            pass
        _require(self._captured is not None, "MINT did not reach requested layer")
        result = self._captured
        self._captured = None
        return result

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class EarlyStopMultiLayerExtractor(object):
    """Capture several layers in one forward and stop before the LM head.

    Lower requested blocks are retained by non-terminating hooks.  The highest
    requested layer terminates the forward.  For MINT's final layer, capture is
    deliberately performed after ``emb_layer_norm_after`` because that is the
    tensor returned by ordinary ``repr_layers=[33]`` inference.
    """

    def __init__(self, model, layers):
        requested = tuple(sorted(set(int(value) for value in layers)))
        _require(bool(requested), "at least one MINT layer must be requested")
        _require(
            all(1 <= value <= len(model.layers) for value in requested),
            "requested multi-layer representation is absent",
        )
        self.model = model
        self.layers = requested
        self._captured = {}
        self._handles = []
        highest = requested[-1]
        for layer in requested:
            if layer == len(model.layers):
                continue
            should_stop = layer == highest
            self._handles.append(
                model.layers[layer - 1].register_forward_hook(
                    self._block_hook(layer, should_stop)
                )
            )
        if highest == len(model.layers):
            _require(
                hasattr(model, "emb_layer_norm_after"),
                "MINT final layer lacks its output normalization",
            )
            self._handles.append(
                model.emb_layer_norm_after.register_forward_hook(
                    self._final_norm_hook(highest)
                )
            )

    def _block_hook(self, layer, should_stop):
        def hook(module, inputs, output):
            del module, inputs
            _require(
                isinstance(output, (tuple, list)) and len(output) >= 1,
                "transformer block returned an unexpected value",
            )
            representation = output[0]
            _require(
                representation.ndim == 3,
                "transformer output must have shape [T,B,C]",
            )
            # Keep an independent snapshot while later transformer blocks run.
            self._captured[layer] = representation.transpose(0, 1).clone()
            if should_stop:
                raise _LayerCaptured()

        return hook

    def _final_norm_hook(self, layer):
        def hook(module, inputs, output):
            del module, inputs
            _require(
                isinstance(output, torch.Tensor) and output.ndim == 3,
                "final MINT normalization returned an unexpected value",
            )
            self._captured[layer] = output.transpose(0, 1)
            raise _LayerCaptured()

        return hook

    def extract(self, chains, chain_ids):
        _require(bool(self._handles), "multi-layer extractor is closed")
        self._captured = {}
        try:
            self.model(chains, chain_ids, repr_layers=[])
        except _LayerCaptured:
            pass
        _require(
            set(self._captured) == set(self.layers),
            "MINT did not reach every requested layer",
        )
        result = dict(self._captured)
        self._captured = {}
        return result

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _validate_sequences(chain1, chain2):
    _require(len(chain1) == len(chain2), "chain columns have different row counts")
    for values, expected_length, label in (
        (chain1, CHAIN_LENGTHS[0], "chain 1"),
        (chain2, CHAIN_LENGTHS[1], "chain 2"),
    ):
        for value in values:
            _require(isinstance(value, str), "{} sequence is not text".format(label))
            _require(len(value) == expected_length, "{} sequence length changed".format(label))
            _require(set(value).issubset(AA_ALPHABET), "{} has noncanonical residue".format(label))


def _validate_candidate_values(values, contract=LIBB_LAYER5_CONTRACT):
    count = len(values["pair_uid"])
    required_columns = _candidate_input_columns(contract)
    _require(
        all(len(values[name]) == count for name in required_columns),
        "candidate columns differ in length",
    )
    if contract.library == "LibA":
        for name in LIBA_HANDOFF_FLAG_COLUMNS:
            _require(
                all(isinstance(value, (bool, np.bool_)) for value in values[name]),
                "{} must be a nonmissing Boolean column".format(name),
            )
    for index in range(count):
        library = str(values["library"][index])
        peptide_code = str(values["peptide_design_code"][index])
        peptide = str(values["peptide_full_sequence"][index])
        affibody_code = str(values["affibody_design_code"][index])
        chain1 = str(values["chain1_smart_hla_linker_peptide_sequence"][index])
        chain2 = str(values["chain2_affibody_sequence"][index])
        _require(
            library == contract.library,
            "candidate partition contains a row outside {}".format(contract.library),
        )
        _require(len(peptide_code) == 2, "peptide design code length changed")
        _require(
            len(affibody_code) == contract.affibody_code_length,
            "{} Affibody design code must have {} characters".format(
                contract.library, contract.affibody_code_length
            ),
        )
        _require(
            len(peptide) == 9 and chain1[-9:] == peptide,
            "full peptide/chain-1 mapping changed",
        )
        _require(peptide[3:5] == peptide_code, "peptide code does not match positions 4/5")
        reconstructed = "".join(
            chain2[position] for position in contract.affibody_code_indices
        )
        _require(
            reconstructed == affibody_code,
            "Affibody code does not match displayed positions {}".format(
                "/".join(str(index + 1) for index in contract.affibody_code_indices)
            ),
        )
        _require(
            hashlib.sha256(chain1.encode("ascii")).hexdigest()
            == str(values["chain1_sha256"][index]),
            "chain-1 checksum changed",
        )
        _require(
            hashlib.sha256(chain2.encode("ascii")).hexdigest()
            == str(values["chain2_sha256"][index]),
            "chain-2 checksum changed",
        )
        _require(
            str(values["peptide_uid"][index])
            == opaque_id(contract.library, "pep", peptide_code),
            "peptide UID changed",
        )
        _require(
            str(values["affibody_uid"][index])
            == opaque_id(contract.library, "aff", affibody_code),
            "Affibody UID changed",
        )
        _require(
            str(values["pair_uid"][index])
            == opaque_id(contract.library, peptide_code, affibody_code),
            "pair UID changed",
        )


def _collate_sequences(chain1, chain2, device, collator):
    _validate_sequences(chain1, chain2)
    chains, chain_ids = collator(list(zip(chain1, chain2)))
    return chains.to(device), chain_ids.to(device)


def _pooled_layer(model, extractor, chains, chain_ids):
    representations = extractor.extract(chains, chain_ids)
    return _pool_one_representation(model, representations, chains, chain_ids)


def _pool_one_representation(model, representations, chains, chain_ids):
    _require(
        tuple(representations.shape[:2]) == tuple(chains.shape),
        "MINT representation token shape changed",
    )
    _require(
        int(representations.shape[-1]) == RESIDUE_DIMENSION,
        "MINT residue dimension changed",
    )
    masks = residue_masks(chains, chain_ids, model, CHAIN_LENGTHS)
    return pool_representations(representations, masks, RESIDUE_DIMENSION)


def _pooled_layer5(model, extractor, chains, chain_ids):
    """Backward-compatible alias for callers testing the LibB implementation."""

    return _pooled_layer(model, extractor, chains, chain_ids)


def _ordinary_pooled_layer(model, chains, chain_ids, layer):
    layer = int(layer)
    output = model(chains, chain_ids, repr_layers=[layer])
    _require(
        isinstance(output, dict)
        and isinstance(output.get("representations"), dict)
        and layer in output["representations"],
        "ordinary MINT output lacks layer {}".format(layer),
    )
    masks = residue_masks(chains, chain_ids, model, CHAIN_LENGTHS)
    return pool_representations(
        output["representations"][layer], masks, RESIDUE_DIMENSION
    )


def _ordinary_pooled_layer5(model, chains, chain_ids):
    """Backward-compatible layer-5 ordinary-forward helper."""

    return _ordinary_pooled_layer(model, chains, chain_ids, LAYER)


def _seed_runtime():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_wrapper(checkpoint, config_path, device, contract=LIBB_LAYER5_CONTRACT):
    checkpoint = Path(checkpoint).resolve()
    config_path = Path(config_path).resolve()
    _require(checkpoint.is_file(), "MINT checkpoint does not exist")
    _require(config_path.is_file(), "MINT config does not exist")
    device = torch.device(device)
    _require(device.type == "cuda", "MINT candidate inference requires a CUDA device")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(device)
    config = load_config(str(config_path))
    _require(
        int(config.encoder_layers) >= contract.layer,
        "MINT config has fewer than {} layers".format(contract.layer),
    )
    _require(
        int(config.encoder_embed_dim) == RESIDUE_DIMENSION,
        "MINT residue dimension changed",
    )
    wrapper = MINTWrapper(
        config,
        str(checkpoint),
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()
    return wrapper, device


def _load_canonical_evaluation_rows(path, contract=LIBB_LAYER5_CONTRACT):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict) and isinstance(payload.get("rows"), list), "bad rows JSON")
    rows = [row for row in payload["rows"] if row.get("split") == "eval"]
    _require(
        len(rows) == contract.evaluation_rows,
        "{}/layer {} parity input must contain exactly {} evaluation rows".format(
            contract.library, contract.layer, contract.evaluation_rows
        ),
    )
    row_ids = [str(row["row_id"]) for row in rows]
    _require(len(set(row_ids)) == len(row_ids), "parity rows contain duplicate IDs")
    for row in rows:
        if "library" in row:
            _require(
                str(row["library"]) == contract.library,
                "canonical parity row has the wrong library",
            )
        _validate_sequences([str(row["chain1_sequence"])], [str(row["chain2_sequence"])])
    return rows


def _extract_rows(
    model,
    rows,
    device,
    batch_size,
    early_stop,
    contract=LIBB_LAYER5_CONTRACT,
):
    collator = CollateFn(512)
    blocks = []
    extractor = EarlyStopLayerExtractor(model, contract.layer) if early_stop else None
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), int(batch_size)):
                batch = rows[start : start + int(batch_size)]
                chains, chain_ids = _collate_sequences(
                    [str(row["chain1_sequence"]) for row in batch],
                    [str(row["chain2_sequence"]) for row in batch],
                    device,
                    collator,
                )
                if early_stop:
                    features = _pooled_layer(model, extractor, chains, chain_ids)
                else:
                    features = _ordinary_pooled_layer(
                        model, chains, chain_ids, contract.layer
                    )
                blocks.append(features.cpu().numpy())
    finally:
        if extractor is not None:
            extractor.close()
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False)


def _load_reference_scores(path, model_name, contract=LIBB_LAYER5_CONTRACT):
    path = Path(path).resolve()
    _require(path.is_file(), "reference score file does not exist")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if contract.library == "LibA":
        forbidden = sorted(
            name
            for name in header
            if any(
                fragment in str(name).strip().lower()
                for fragment in FORBIDDEN_REFERENCE_COLUMN_FRAGMENTS
            )
        )
        _require(
            not forbidden,
            "LibA parity reference is not target-free: {}".format(forbidden),
        )
    required = {"model", contract.reference_score_column}
    _require(required.issubset(header), "reference score schema changed")
    id_candidates = [
        name for name in ("eval_row_id", "row_id", "pair_uid") if name in header
    ]
    _require(bool(id_candidates), "reference scores lack row IDs")
    id_column = id_candidates[0]
    usecols = [id_column, "model", contract.reference_score_column]
    for optional in ("library", "layer"):
        if optional in header:
            usecols.append(optional)
    frame = pd.read_csv(
        path,
        usecols=usecols,
        keep_default_na=False,
        na_filter=False,
    )
    selected = frame.loc[frame["model"].astype(str).eq(str(model_name))].copy()
    if "library" in selected.columns:
        selected = selected.loc[
            selected["library"].astype(str).eq(contract.library)
        ].copy()
    if "layer" in selected.columns:
        selected = selected.loc[
            pd.to_numeric(selected["layer"], errors="raise").eq(contract.layer)
        ].copy()
    _require(
        len(selected) == contract.evaluation_rows,
        "reference model must have exactly {} {}/layer {} scores".format(
            contract.evaluation_rows, contract.library, contract.layer
        ),
    )
    _require(not bool(selected[id_column].duplicated().any()), "duplicate reference row ID")
    score_column = contract.reference_score_column
    selected[score_column] = pd.to_numeric(selected[score_column], errors="raise")
    _require(
        bool(np.isfinite(selected[score_column]).all()),
        "reference scores are non-finite",
    )
    return dict(
        zip(selected[id_column].astype(str), selected[score_column].astype(float))
    )


def run_parity(args):
    started = time.time()
    contract = _contract_from_args(args)
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    _require(float(args.score_atol) >= 0.0, "score tolerance must be nonnegative")
    _require(not Path(args.output_json).exists(), "parity output exists; refusing overwrite")
    _seed_runtime()
    head_path = _arg_or_contract_default(args, "head_npz", contract)
    canonical_path = _arg_or_contract_default(args, "canonical_rows", contract)
    reference_path = _arg_or_contract_default(args, "reference_scores", contract)
    reference_model = getattr(args, "reference_model", None) or contract.reference_model
    reference_path, reference_model = _validate_reference_contract(
        contract, reference_path, reference_model
    )
    head = load_linear_head(head_path, args.head_prefix, contract=contract)
    rows = _load_canonical_evaluation_rows(canonical_path, contract=contract)
    reference = _load_reference_scores(
        reference_path, reference_model, contract=contract
    )
    row_ids = [str(row["row_id"]) for row in rows]
    _require(set(row_ids) == set(reference), "parity rows/reference IDs differ")
    wrapper, device = _load_wrapper(
        args.checkpoint, args.config, args.device, contract=contract
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ordinary = _extract_rows(
        wrapper.model,
        rows,
        device,
        args.batch_size,
        early_stop=False,
        contract=contract,
    )
    early = _extract_rows(
        wrapper.model,
        rows,
        device,
        args.batch_size,
        early_stop=True,
        contract=contract,
    )
    torch.cuda.synchronize(device)
    _require(
        ordinary.shape == (contract.evaluation_rows, FEATURE_DIMENSION),
        "ordinary parity feature shape changed",
    )
    _require(early.shape == ordinary.shape, "early-stop parity feature shape changed")
    feature_exact = bool(np.array_equal(ordinary, early))
    feature_max_abs = float(np.max(np.abs(ordinary.astype(np.float64) - early)))
    _, scores = score_features(early, head)
    expected = np.asarray([reference[value] for value in row_ids], dtype=np.float64)
    score_difference = np.abs(scores - expected)
    score_max_abs = float(np.max(score_difference))
    score_mean_abs = float(np.mean(score_difference))
    passed = feature_exact and score_max_abs <= float(args.score_atol)

    report = {
        "schema_version": contract.parity_schema_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": bool(passed),
        "contract": {
            "library": contract.library,
            "layer": contract.layer,
            "feature_name": contract.feature_name,
            "feature_dimension": FEATURE_DIMENSION,
            "pooling": POOLING_DESCRIPTION,
            "token_mapping": contract.token_mapping,
            "ordinary_semantics": (
                "MINT model forward repr_layers=[{}]; all configured layers and LM head execute".format(
                    contract.layer
                )
            ),
            "early_stop_semantics": (
                (
                    "capture the normalized final-layer representation and stop before the LM head"
                    if contract.layer == 33
                    else "capture transformer block {} output and stop before block {}".format(
                        contract.layer, contract.layer + 1
                    )
                )
            ),
        },
        "rows": {
            "count": len(rows),
            "row_order_sha256": hashlib.sha256("\n".join(row_ids).encode("ascii")).hexdigest(),
        },
        "checks": {
            "ordinary_vs_early_features_bitwise_equal": feature_exact,
            "ordinary_vs_early_features_max_abs": feature_max_abs,
            "head_score_vs_reference_max_abs": score_max_abs,
            "head_score_vs_reference_mean_abs": score_mean_abs,
            "score_atol": float(args.score_atol),
            "reference_model": str(reference_model),
            "reference_score_column": contract.reference_score_column,
        },
        "inputs": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
            "checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": sha256_file(args.checkpoint)},
            "config": {"path": str(Path(args.config).resolve()), "sha256": sha256_file(args.config)},
            "head_npz": {"path": str(head["path"]), "sha256": head["sha256"]},
            "canonical_rows": {"path": str(Path(canonical_path).resolve()), "sha256": sha256_file(canonical_path)},
            "reference_scores": {"path": str(Path(reference_path).resolve()), "sha256": sha256_file(reference_path)},
        },
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    _atomic_json(args.output_json, report)
    _require(
        passed,
        "{}/layer {} parity failed; see {}".format(
            contract.library, contract.layer, args.output_json
        ),
    )
    print(json.dumps(report["checks"], sort_keys=True))


def _validate_parity_receipt(
    path,
    head,
    checkpoint,
    config_path,
    contract=LIBB_LAYER5_CONTRACT,
):
    path = Path(path).resolve()
    _require(path.is_file(), "parity receipt does not exist")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    _require(
        report.get("schema_version") == contract.parity_schema_version,
        "parity receipt schema changed",
    )
    _require(report.get("passed") is True, "parity receipt did not pass")
    receipt_contract = report.get("contract", {})
    _require(
        receipt_contract.get("library") == contract.library,
        "parity receipt library does not match scoring contract",
    )
    _require(
        int(receipt_contract.get("layer", -1)) == contract.layer,
        "parity receipt layer does not match scoring contract",
    )
    _require(
        receipt_contract.get("feature_name") == contract.feature_name,
        "parity receipt feature does not match scoring contract",
    )
    _require(
        receipt_contract.get("token_mapping") == contract.token_mapping,
        "parity receipt token mapping does not match scoring contract",
    )
    _require(
        int(report.get("rows", {}).get("count", -1)) == contract.evaluation_rows,
        "parity receipt evaluation-row count changed",
    )
    checks = report.get("checks", {})
    _require(
        checks.get("ordinary_vs_early_features_bitwise_equal") is True,
        "parity receipt lacks exact feature parity",
    )
    score_difference = checks.get("head_score_vs_reference_max_abs")
    score_tolerance = checks.get("score_atol")
    _require(
        isinstance(score_difference, (int, float))
        and math.isfinite(float(score_difference))
        and isinstance(score_tolerance, (int, float))
        and math.isfinite(float(score_tolerance))
        and float(score_tolerance) >= 0.0
        and float(score_difference) <= float(score_tolerance),
        "parity receipt lacks passing reference-score parity",
    )
    _require(
        checks.get("reference_model") == contract.reference_model,
        "parity receipt used a different reference model",
    )
    _require(
        checks.get("reference_score_column") == contract.reference_score_column,
        "parity receipt used a different reference score column",
    )
    inputs = report.get("inputs", {})
    expected = {
        "script": sha256_file(__file__),
        "checkpoint": sha256_file(checkpoint),
        "config": sha256_file(config_path),
        "head_npz": head["sha256"],
    }
    for name, digest in expected.items():
        _require(
            inputs.get(name, {}).get("sha256") == digest,
            "parity receipt does not match current {}".format(name),
        )
    return path, sha256_file(path), report, expected


def _output_schema(contract=LIBB_LAYER5_CONTRACT):
    fields = [
            pa.field("input_row_index", pa.uint64(), nullable=False),
            pa.field("pair_uid", pa.string(), nullable=False),
            pa.field("library", pa.string(), nullable=False),
            pa.field("peptide_design_code", pa.string(), nullable=False),
            pa.field("peptide_full_sequence", pa.string(), nullable=False),
            pa.field("affibody_design_code", pa.string(), nullable=False),
            pa.field("peptide_uid", pa.string(), nullable=False),
            pa.field("affibody_uid", pa.string(), nullable=False),
            pa.field("chain1_sha256", pa.string(), nullable=False),
            pa.field("chain2_sha256", pa.string(), nullable=False),
    ]
    if contract.library == "LibA":
        # The prospective handoff consumes these literal, model-agnostic names.
        # Keep the provider-displayed and model-input sequences separate even
        # though both are currently the same 58-aa string.
        fields = [
            pa.field("input_row_index", pa.uint64(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("pair_uid", pa.string(), nullable=False),
            pa.field("library", pa.string(), nullable=False),
            pa.field("peptide_design_code", pa.string(), nullable=False),
            pa.field("peptide_9mer_sequence", pa.string(), nullable=False),
            pa.field("affibody_design_code", pa.string(), nullable=False),
            pa.field("peptide_uid", pa.string(), nullable=False),
            pa.field("affibody_uid", pa.string(), nullable=False),
            pa.field("chain1_sha256", pa.string(), nullable=False),
            pa.field("chain2_sha256", pa.string(), nullable=False),
            pa.field(
                "provider_displayed_58aa_affibody_sequence",
                pa.string(),
                nullable=False,
            ),
            pa.field("model_input_affibody_sequence", pa.string(), nullable=False),
            pa.field(
                "model_input_smart_hla_linker_peptide_sequence",
                pa.string(),
                nullable=False,
            ),
            pa.field("model_score", pa.float64(), nullable=False),
            pa.field(contract.logit_column, pa.float64(), nullable=False),
        ] + [
            pa.field(name, pa.bool_(), nullable=False)
            for name in LIBA_HANDOFF_FLAG_COLUMNS
        ]
    else:
        fields += [
            pa.field(contract.logit_column, pa.float64(), nullable=False),
            pa.field(contract.score_column, pa.float64(), nullable=False),
        ]
    schema = pa.schema(
        fields,
        metadata={
            b"schema_version": contract.schema_version.encode("ascii"),
            b"library": contract.library.encode("ascii"),
            b"layer": str(contract.layer).encode("ascii"),
            b"feature_name": contract.feature_name.encode("ascii"),
            b"pooling": POOLING_DESCRIPTION.encode("ascii"),
        },
    )
    _require(
        not FORBIDDEN_OUTCOME_COLUMNS.intersection(schema.names),
        "score output schema contains an outcome column",
    )
    return schema


def _scored_output_columns(values, row_offset, logits, scores, contract):
    count = len(scores)
    columns = {
        "input_row_index": np.arange(row_offset, row_offset + count, dtype=np.uint64)
    }
    if contract.library == "LibA":
        columns.update(
            {
                "model_id": [contract.model_id] * count,
                "pair_uid": values["pair_uid"],
                "library": values["library"],
                "peptide_design_code": values["peptide_design_code"],
                "peptide_9mer_sequence": values["peptide_full_sequence"],
                "affibody_design_code": values["affibody_design_code"],
                "peptide_uid": values["peptide_uid"],
                "affibody_uid": values["affibody_uid"],
                "chain1_sha256": values["chain1_sha256"],
                "chain2_sha256": values["chain2_sha256"],
                "provider_displayed_58aa_affibody_sequence": values[
                    "chain2_affibody_sequence"
                ],
                "model_input_affibody_sequence": values[
                    "chain2_affibody_sequence"
                ],
                "model_input_smart_hla_linker_peptide_sequence": values[
                    "chain1_smart_hla_linker_peptide_sequence"
                ],
                "model_score": scores,
                contract.logit_column: logits,
            }
        )
        for name in LIBA_HANDOFF_FLAG_COLUMNS:
            columns[name] = values[name]
    else:
        for name in OUTPUT_ID_COLUMNS:
            columns[name] = values[name]
        columns[contract.logit_column] = logits
        columns[contract.score_column] = scores
    return columns


def _score_record_batch(
    batch,
    row_offset,
    model,
    extractor,
    head,
    device,
    collator,
    contract=LIBB_LAYER5_CONTRACT,
):
    names = set(batch.schema.names)
    input_columns = _candidate_input_columns(contract)
    _require(set(input_columns).issubset(names), "candidate Parquet schema changed")
    _require(
        not FORBIDDEN_OUTCOME_COLUMNS.intersection(names),
        "candidate Parquet contains forbidden retention/label columns",
    )
    values = batch.select(list(input_columns)).to_pydict()
    _validate_candidate_values(values, contract=contract)
    chains, chain_ids = _collate_sequences(
        values[INPUT_COLUMNS[-2]], values[INPUT_COLUMNS[-1]], device, collator
    )
    features = _pooled_layer(model, extractor, chains, chain_ids)
    logits, scores = score_features(features.cpu().numpy(), head)
    columns = _scored_output_columns(values, row_offset, logits, scores, contract)
    return pa.Table.from_pydict(columns, schema=_output_schema(contract))


def _score_record_batch_dual_liba(
    batch,
    row_offset,
    model,
    extractor,
    heads,
    device,
    collator,
):
    """Score LibA layer 9 and 33 from one backbone forward."""

    contracts = (LIBA_LAYER9_CONTRACT, LIBA_LAYER33_CONTRACT)
    names = set(batch.schema.names)
    input_columns = _candidate_input_columns(LIBA_LAYER9_CONTRACT)
    _require(set(input_columns).issubset(names), "candidate Parquet schema changed")
    _require(
        not FORBIDDEN_OUTCOME_COLUMNS.intersection(names),
        "candidate Parquet contains forbidden retention/label columns",
    )
    values = batch.select(list(input_columns)).to_pydict()
    _validate_candidate_values(values, contract=LIBA_LAYER9_CONTRACT)
    chains, chain_ids = _collate_sequences(
        values[INPUT_COLUMNS[-2]], values[INPUT_COLUMNS[-1]], device, collator
    )
    captured = extractor.extract(chains, chain_ids)
    tables = {}
    for contract in contracts:
        features = _pool_one_representation(
            model, captured[contract.layer], chains, chain_ids
        )
        logits, scores = score_features(
            features.cpu().numpy(), heads[contract.layer]
        )
        columns = _scored_output_columns(
            values, row_offset, logits, scores, contract
        )
        tables[contract.layer] = pa.Table.from_pydict(
            columns, schema=_output_schema(contract)
        )
    return tables


def run_score(args):
    started = time.time()
    contract = _contract_from_args(args)
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    input_path = Path(args.input_parquet).resolve()
    output_path = Path(args.output_parquet).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest is not None
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    _require(input_path.is_file(), "candidate Parquet does not exist")
    _require(not output_path.exists(), "score output exists; refusing overwrite")
    _require(not manifest_path.exists(), "score manifest exists; refusing overwrite")
    _require(output_path != input_path, "input and output paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(str(output_path.parent), 0o700)

    _seed_runtime()
    head_path = _arg_or_contract_default(args, "head_npz", contract)
    head = load_linear_head(head_path, args.head_prefix, contract=contract)
    parity_path, parity_sha256, parity, input_hashes = _validate_parity_receipt(
        args.parity_receipt,
        head,
        args.checkpoint,
        args.config,
        contract=contract,
    )
    input_sha256 = sha256_file(input_path)
    wrapper, device = _load_wrapper(
        args.checkpoint, args.config, args.device, contract=contract
    )
    parquet = pq.ParquetFile(str(input_path))
    input_columns = _candidate_input_columns(contract)
    _require(
        set(input_columns).issubset(set(parquet.schema_arrow.names)),
        "candidate Parquet schema changed",
    )
    _require(
        not FORBIDDEN_OUTCOME_COLUMNS.intersection(parquet.schema_arrow.names),
        "candidate Parquet contains forbidden retention/label columns",
    )
    expected_rows = int(parquet.metadata.num_rows)
    _require(expected_rows >= 1, "candidate partition is empty")
    collator = CollateFn(512)

    temporary = output_path.with_name(".{}.tmp-{}".format(output_path.name, os.getpid()))
    _require(not temporary.exists(), "temporary score output already exists")
    observed_rows = 0
    uid_digest = hashlib.sha256()
    seen_pair_uids = set()
    previous_affibody_code = None
    partition_peptide = None
    writer = None
    buffered_tables = []
    buffered_rows = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        writer = pq.ParquetWriter(
            str(temporary),
            _output_schema(contract),
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        with EarlyStopLayerExtractor(wrapper.model, contract.layer) as extractor:
            with torch.inference_mode():
                for step, batch in enumerate(
                    parquet.iter_batches(
                        batch_size=int(args.batch_size), columns=list(input_columns)
                    ),
                    start=1,
                ):
                    output = _score_record_batch(
                        batch,
                        observed_rows,
                        wrapper.model,
                        extractor,
                        head,
                        device,
                        collator,
                        contract=contract,
                    )
                    pair_ids = output.column("pair_uid").to_pylist()
                    affibody_codes = output.column("affibody_design_code").to_pylist()
                    peptide_codes = set(output.column("peptide_design_code").to_pylist())
                    _require(len(peptide_codes) == 1, "a score batch contains several peptides")
                    batch_peptide = next(iter(peptide_codes))
                    if partition_peptide is None:
                        partition_peptide = batch_peptide
                    _require(
                        batch_peptide == partition_peptide,
                        "partition contains several peptides",
                    )
                    _require(len(set(pair_ids)) == len(pair_ids), "duplicate pair UID within batch")
                    _require(
                        not seen_pair_uids.intersection(pair_ids),
                        "duplicate pair UID across batches",
                    )
                    seen_pair_uids.update(pair_ids)
                    _require(
                        affibody_codes == sorted(affibody_codes)
                        and len(set(affibody_codes)) == len(affibody_codes),
                        "Affibody codes are not strictly ordered within batch",
                    )
                    if previous_affibody_code is not None:
                        _require(
                            previous_affibody_code < affibody_codes[0],
                            "Affibody order changed across batches",
                        )
                    previous_affibody_code = affibody_codes[-1]
                    uid_digest.update(
                        ("\n".join(str(value) for value in pair_ids) + "\n").encode("ascii")
                    )
                    buffered_tables.append(output)
                    buffered_rows += len(output)
                    if buffered_rows >= int(args.output_row_group_rows):
                        combined = pa.concat_tables(buffered_tables)
                        writer.write_table(
                            combined, row_group_size=int(args.output_row_group_rows)
                        )
                        buffered_tables = []
                        buffered_rows = 0
                    observed_rows += len(output)
                    if step % int(args.log_every) == 0 or observed_rows == expected_rows:
                        elapsed = max(time.time() - started, 1e-9)
                        print(
                            "{}: scored {}/{} rows ({:.1f} rows/s including model load)".format(
                                input_path.name, observed_rows, expected_rows, observed_rows / elapsed
                            ),
                            flush=True,
                        )
        if buffered_tables:
            combined = pa.concat_tables(buffered_tables)
            writer.write_table(combined, row_group_size=int(args.output_row_group_rows))
            buffered_tables = []
            buffered_rows = 0
        writer.close()
        writer = None
        torch.cuda.synchronize(device)
        _require(observed_rows == expected_rows, "scored row count differs from input")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(output_path))
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()

    elapsed = time.time() - started
    output_sha256 = sha256_file(output_path)
    manifest = {
        "schema_version": contract.schema_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": observed_rows,
        "peptide_design_code": partition_peptide,
        "row_order_pair_uid_sha256": uid_digest.hexdigest(),
        "model": {
            "name": "frozen MINT layer {} plus balanced logistic readout".format(
                contract.layer
            ),
            "model_id": contract.model_id,
            "library": contract.library,
            "layer": contract.layer,
            "feature_name": contract.feature_name,
            "feature_dimension": FEATURE_DIMENSION,
            "pooling": POOLING_DESCRIPTION,
            "token_mapping": contract.token_mapping,
            "early_stop": (
                (
                    "capture normalized final representation and stop before LM head"
                    if contract.layer == 33
                    else "forward hook stops immediately after transformer block {}".format(
                        contract.layer
                    )
                )
            ),
            "head_keys": head["keys"],
            "head_metadata": head["metadata"],
        },
        "inputs": {
            "candidate_parquet": {"path": str(input_path), "sha256": input_sha256},
            "checkpoint": {"path": str(Path(args.checkpoint).resolve()), "sha256": input_hashes["checkpoint"]},
            "config": {"path": str(Path(args.config).resolve()), "sha256": input_hashes["config"]},
            "head_npz": {"path": str(head["path"]), "sha256": head["sha256"]},
            "parity_receipt": {"path": str(parity_path), "sha256": parity_sha256},
        },
        "parity": parity["checks"],
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "bytes": output_path.stat().st_size,
            "columns": _output_schema(contract).names,
        },
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "rows_per_second_including_load": float(observed_rows / elapsed),
            "batch_size": int(args.batch_size),
            "output_row_group_rows": int(args.output_row_group_rows),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"output": str(output_path), "rows": observed_rows, "seconds": elapsed}, sort_keys=True))


def run_score_dual_liba(args):
    """Score the two locked LibA MINT heads from a single full-depth forward."""

    started = time.time()
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    contracts = {
        9: LIBA_LAYER9_CONTRACT,
        33: LIBA_LAYER33_CONTRACT,
    }
    input_path = Path(args.input_parquet).resolve()
    _require(input_path.is_file(), "candidate Parquet does not exist")
    output_paths = {
        9: Path(args.layer9_output_parquet).resolve(),
        33: Path(args.layer33_output_parquet).resolve(),
    }
    manifest_paths = {
        9: (
            Path(args.layer9_manifest).resolve()
            if args.layer9_manifest is not None
            else output_paths[9].with_suffix(output_paths[9].suffix + ".manifest.json")
        ),
        33: (
            Path(args.layer33_manifest).resolve()
            if args.layer33_manifest is not None
            else output_paths[33].with_suffix(output_paths[33].suffix + ".manifest.json")
        ),
    }
    all_targets = [input_path] + list(output_paths.values()) + list(manifest_paths.values())
    _require(len(set(all_targets)) == len(all_targets), "dual score paths must be distinct")
    for layer in (9, 33):
        _require(
            not output_paths[layer].exists(),
            "layer-{} score output exists; refusing overwrite".format(layer),
        )
        _require(
            not manifest_paths[layer].exists(),
            "layer-{} score manifest exists; refusing overwrite".format(layer),
        )
        output_paths[layer].parent.mkdir(parents=True, exist_ok=True)
        os.chmod(str(output_paths[layer].parent), 0o700)

    _seed_runtime()
    head_paths = {
        9: args.layer9_head_npz or contracts[9].head_npz,
        33: args.layer33_head_npz or contracts[33].head_npz,
    }
    head_prefixes = {
        9: args.layer9_head_prefix,
        33: args.layer33_head_prefix,
    }
    heads = {
        layer: load_linear_head(
            head_paths[layer], head_prefixes[layer], contract=contracts[layer]
        )
        for layer in (9, 33)
    }
    receipt_args = {
        9: args.layer9_parity_receipt,
        33: args.layer33_parity_receipt,
    }
    receipts = {}
    for layer in (9, 33):
        receipts[layer] = _validate_parity_receipt(
            receipt_args[layer],
            heads[layer],
            args.checkpoint,
            args.config,
            contract=contracts[layer],
        )

    input_sha256 = sha256_file(input_path)
    wrapper, device = _load_wrapper(
        args.checkpoint, args.config, args.device, contract=contracts[33]
    )
    parquet = pq.ParquetFile(str(input_path))
    input_columns = _candidate_input_columns(contracts[9])
    _require(
        set(input_columns).issubset(set(parquet.schema_arrow.names)),
        "candidate Parquet schema changed",
    )
    _require(
        not FORBIDDEN_OUTCOME_COLUMNS.intersection(parquet.schema_arrow.names),
        "candidate Parquet contains forbidden retention/label columns",
    )
    expected_rows = int(parquet.metadata.num_rows)
    _require(expected_rows >= 1, "candidate partition is empty")
    collator = CollateFn(512)
    temporary_paths = {
        layer: output_paths[layer].with_name(
            ".{}.tmp-{}".format(output_paths[layer].name, os.getpid())
        )
        for layer in (9, 33)
    }
    for path in temporary_paths.values():
        _require(not path.exists(), "temporary dual-score output already exists")
    writers = {}
    buffers = {9: [], 33: []}
    buffered_rows = {9: 0, 33: 0}
    observed_rows = 0
    uid_digest = hashlib.sha256()
    seen_pair_uids = set()
    previous_affibody_code = None
    partition_peptide = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for layer in (9, 33):
            writers[layer] = pq.ParquetWriter(
                str(temporary_paths[layer]),
                _output_schema(contracts[layer]),
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        with EarlyStopMultiLayerExtractor(wrapper.model, (9, 33)) as extractor:
            with torch.inference_mode():
                for step, batch in enumerate(
                    parquet.iter_batches(
                        batch_size=int(args.batch_size), columns=list(input_columns)
                    ),
                    start=1,
                ):
                    outputs = _score_record_batch_dual_liba(
                        batch,
                        observed_rows,
                        wrapper.model,
                        extractor,
                        heads,
                        device,
                        collator,
                    )
                    base = outputs[9]
                    pair_ids = base.column("pair_uid").to_pylist()
                    affibody_codes = base.column("affibody_design_code").to_pylist()
                    peptide_codes = set(base.column("peptide_design_code").to_pylist())
                    _require(
                        len(peptide_codes) == 1,
                        "a dual score batch contains several peptides",
                    )
                    batch_peptide = next(iter(peptide_codes))
                    if partition_peptide is None:
                        partition_peptide = batch_peptide
                    _require(
                        batch_peptide == partition_peptide,
                        "partition contains several peptides",
                    )
                    _require(
                        len(set(pair_ids)) == len(pair_ids),
                        "duplicate pair UID within batch",
                    )
                    _require(
                        not seen_pair_uids.intersection(pair_ids),
                        "duplicate pair UID across batches",
                    )
                    seen_pair_uids.update(pair_ids)
                    _require(
                        affibody_codes == sorted(affibody_codes)
                        and len(set(affibody_codes)) == len(affibody_codes),
                        "Affibody codes are not strictly ordered within batch",
                    )
                    if previous_affibody_code is not None:
                        _require(
                            previous_affibody_code < affibody_codes[0],
                            "Affibody order changed across batches",
                        )
                    previous_affibody_code = affibody_codes[-1]
                    uid_digest.update(
                        ("\n".join(str(value) for value in pair_ids) + "\n").encode(
                            "ascii"
                        )
                    )
                    for layer in (9, 33):
                        _require(
                            outputs[layer].column("pair_uid").to_pylist() == pair_ids,
                            "dual outputs differ in row order",
                        )
                        buffers[layer].append(outputs[layer])
                        buffered_rows[layer] += len(outputs[layer])
                        if buffered_rows[layer] >= int(args.output_row_group_rows):
                            combined = pa.concat_tables(buffers[layer])
                            writers[layer].write_table(
                                combined,
                                row_group_size=int(args.output_row_group_rows),
                            )
                            buffers[layer] = []
                            buffered_rows[layer] = 0
                    observed_rows += len(base)
                    if step % int(args.log_every) == 0 or observed_rows == expected_rows:
                        elapsed = max(time.time() - started, 1e-9)
                        print(
                            "{}: dual-scored {}/{} rows ({:.1f} rows/s including model load)".format(
                                input_path.name,
                                observed_rows,
                                expected_rows,
                                observed_rows / elapsed,
                            ),
                            flush=True,
                        )
        for layer in (9, 33):
            if buffers[layer]:
                combined = pa.concat_tables(buffers[layer])
                writers[layer].write_table(
                    combined, row_group_size=int(args.output_row_group_rows)
                )
                buffers[layer] = []
                buffered_rows[layer] = 0
            writers[layer].close()
            del writers[layer]
        torch.cuda.synchronize(device)
        _require(observed_rows == expected_rows, "dual-scored row count differs from input")
        for layer in (9, 33):
            os.chmod(str(temporary_paths[layer]), 0o600)
        for layer in (9, 33):
            os.replace(str(temporary_paths[layer]), str(output_paths[layer]))
    finally:
        for writer in writers.values():
            writer.close()
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()

    elapsed = time.time() - started
    output_hashes = {
        layer: sha256_file(output_paths[layer]) for layer in (9, 33)
    }
    row_order_sha256 = uid_digest.hexdigest()
    for layer in (9, 33):
        contract = contracts[layer]
        parity_path, parity_sha256, parity, input_hashes = receipts[layer]
        companion_layer = 33 if layer == 9 else 9
        manifest = {
            "schema_version": contract.schema_version,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": observed_rows,
            "peptide_design_code": partition_peptide,
            "row_order_pair_uid_sha256": row_order_sha256,
            "model": {
                "name": "frozen MINT layer {} plus balanced logistic readout".format(
                    layer
                ),
                "model_id": contract.model_id,
                "library": "LibA",
                "layer": layer,
                "feature_name": contract.feature_name,
                "feature_dimension": FEATURE_DIMENSION,
                "pooling": POOLING_DESCRIPTION,
                "token_mapping": contract.token_mapping,
                "head_keys": heads[layer]["keys"],
                "head_metadata": heads[layer]["metadata"],
            },
            "joint_forward": {
                "layers": [9, 33],
                "semantics": (
                    "layer 9 captured after transformer block 9; normalized layer 33 "
                    "captured after final layer normalization; stopped before LM head"
                ),
                "backbone_forwards_per_batch": 1,
                "companion_model_id": contracts[companion_layer].model_id,
                "companion_output": {
                    "path": str(output_paths[companion_layer]),
                    "sha256": output_hashes[companion_layer],
                },
            },
            "inputs": {
                "candidate_parquet": {
                    "path": str(input_path),
                    "sha256": input_sha256,
                },
                "checkpoint": {
                    "path": str(Path(args.checkpoint).resolve()),
                    "sha256": input_hashes["checkpoint"],
                },
                "config": {
                    "path": str(Path(args.config).resolve()),
                    "sha256": input_hashes["config"],
                },
                "head_npz": {
                    "path": str(heads[layer]["path"]),
                    "sha256": heads[layer]["sha256"],
                },
                "parity_receipt": {
                    "path": str(parity_path),
                    "sha256": parity_sha256,
                },
            },
            "parity": parity["checks"],
            "output": {
                "path": str(output_paths[layer]),
                "sha256": output_hashes[layer],
                "bytes": output_paths[layer].stat().st_size,
                "columns": _output_schema(contract).names,
            },
            "runtime": {
                "elapsed_seconds": float(elapsed),
                "rows_per_second_including_load": float(observed_rows / elapsed),
                "batch_size": int(args.batch_size),
                "output_row_group_rows": int(args.output_row_group_rows),
                "hostname": socket.gethostname(),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
            },
        }
        _atomic_json(manifest_paths[layer], manifest)
    print(
        json.dumps(
            {
                "layer9_output": str(output_paths[9]),
                "layer33_output": str(output_paths[33]),
                "rows": observed_rows,
                "seconds": elapsed,
            },
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    parity = subparsers.add_parser("parity", help="validate early-stop and saved-head parity")
    parity.add_argument("--library", choices=("LibA", "LibB"), default="LibB")
    parity.add_argument(
        "--layer",
        type=int,
        help="selected MINT layer (defaults to 9 for LibA and 5 for LibB)",
    )
    parity.add_argument("--head-npz", type=Path)
    parity.add_argument("--head-prefix")
    parity.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parity.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parity.add_argument("--canonical-rows", type=Path)
    parity.add_argument("--reference-scores", type=Path)
    parity.add_argument("--reference-model")
    parity.add_argument("--score-atol", type=float, default=5e-5)
    parity.add_argument("--device", default="cuda:0")
    parity.add_argument("--batch-size", type=int, default=64)
    parity.add_argument("--output-json", required=True, type=Path)
    parity.set_defaults(func=run_parity)

    score = subparsers.add_parser("score", help="stream one candidate Parquet partition")
    score.add_argument("--library", choices=("LibA", "LibB"), default="LibB")
    score.add_argument(
        "--layer",
        type=int,
        help="selected MINT layer (defaults to 9 for LibA and 5 for LibB)",
    )
    score.add_argument("--input-parquet", required=True, type=Path)
    score.add_argument("--output-parquet", required=True, type=Path)
    score.add_argument("--manifest", type=Path)
    score.add_argument("--head-npz", type=Path)
    score.add_argument("--head-prefix")
    score.add_argument("--parity-receipt", required=True, type=Path)
    score.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    score.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--batch-size", type=int, default=64)
    score.add_argument("--output-row-group-rows", type=int, default=65536)
    score.add_argument("--log-every", type=int, default=100)
    score.set_defaults(func=run_score)

    dual = subparsers.add_parser(
        "score-dual-liba",
        help="score LibA layers 9 and 33 in one full-depth MINT forward",
    )
    dual.add_argument("--input-parquet", required=True, type=Path)
    dual.add_argument("--layer9-output-parquet", required=True, type=Path)
    dual.add_argument("--layer33-output-parquet", required=True, type=Path)
    dual.add_argument("--layer9-manifest", type=Path)
    dual.add_argument("--layer33-manifest", type=Path)
    dual.add_argument("--layer9-head-npz", type=Path)
    dual.add_argument("--layer33-head-npz", type=Path)
    dual.add_argument("--layer9-head-prefix")
    dual.add_argument("--layer33-head-prefix")
    dual.add_argument("--layer9-parity-receipt", required=True, type=Path)
    dual.add_argument("--layer33-parity-receipt", required=True, type=Path)
    dual.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    dual.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dual.add_argument("--device", default="cuda:0")
    dual.add_argument("--batch-size", type=int, default=64)
    dual.add_argument("--output-row-group-rows", type=int, default=65536)
    dual.add_argument("--log-every", type=int, default=100)
    dual.set_defaults(func=run_score_dual_liba)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    _require(int(getattr(args, "log_every", 1)) >= 1, "log interval must be positive")
    _require(
        int(getattr(args, "output_row_group_rows", 1)) >= 1,
        "output row-group size must be positive",
    )
    args.func(args)


if __name__ == "__main__":
    main()
