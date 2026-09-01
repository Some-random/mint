"""Small, model-independent helpers for OpenFold3 distogram analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


DISTOGRAM_BIN_EDGES_A = np.linspace(2.0, 22.0, 65, dtype=np.float64)
DISTOGRAM_BIN_CENTERS_A = (
    DISTOGRAM_BIN_EDGES_A[:-1] + DISTOGRAM_BIN_EDGES_A[1:]
) / 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harden_private_tree(root: Path) -> None:
    root = Path(root)
    root.chmod(0o700)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def distogram_matrices(
    logits: np.ndarray, contact_cutoff_A: float = 8.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin probabilities, P(distance < cutoff), and expected distance."""
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[-1] != 64:
        raise ValueError(f"Expected [N,N,64] logits, found {logits.shape}")
    if logits.shape[0] != logits.shape[1]:
        raise ValueError(f"Distogram must be square, found {logits.shape}")
    if not np.isfinite(logits).all():
        raise ValueError("Distogram logits contain non-finite values")
    probabilities = stable_softmax(logits)
    contact_mask = DISTOGRAM_BIN_EDGES_A[1:] <= contact_cutoff_A
    contact_probability = probabilities[..., contact_mask].sum(axis=-1)
    expected_distance = np.sum(
        probabilities * DISTOGRAM_BIN_CENTERS_A, axis=-1
    )
    return probabilities, contact_probability, expected_distance


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def squeeze_leading_singletons(array: np.ndarray, target_ndim: int) -> np.ndarray:
    array = np.asarray(array)
    while array.ndim > target_ndim and array.shape[0] == 1:
        array = array[0]
    return array


def distogram_logits_from_latent(latent: dict) -> np.ndarray:
    if "distogram_logits" not in latent:
        raise ValueError("Latent output has no distogram_logits")
    logits = squeeze_leading_singletons(
        _numpy(latent["distogram_logits"]), target_ndim=3
    )
    if logits.ndim != 3 or logits.shape[-1] != 64:
        raise ValueError(f"Unexpected distogram shape {logits.shape}")
    return logits


def token_lookup_from_batch(batch: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    residue_index = squeeze_leading_singletons(
        _numpy(batch["residue_index"]), target_ndim=1
    )
    start_atom_index = squeeze_leading_singletons(
        _numpy(batch["start_atom_index"]), target_ndim=1
    )
    atom_array = batch["atom_array"]
    if isinstance(atom_array, (list, tuple)):
        if len(atom_array) != 1:
            raise ValueError("Expected one atom array")
        atom_array = atom_array[0]
    token_chain_ids = np.asarray(
        [str(atom_array.chain_id[int(index)]) for index in start_atom_index]
    )
    if not (len(token_chain_ids) == len(residue_index)):
        raise ValueError("Token chain IDs and residue indices differ in length")
    lookup = {}
    for token, (chain, position) in enumerate(zip(token_chain_ids, residue_index)):
        key = (str(chain), int(position))
        if key in lookup:
            raise ValueError(f"Duplicate token key {key}")
        lookup[key] = token
    return lookup, token_chain_ids, residue_index.astype(np.int64)


def pair_tensor(
    matrix: np.ndarray,
    lookup: dict,
    left_chain: str,
    left_positions: tuple[int, ...] | list[int],
    right_chain: str,
    right_positions: tuple[int, ...] | list[int],
) -> np.ndarray:
    left_tokens = [lookup[(left_chain, int(position))] for position in left_positions]
    right_tokens = [
        lookup[(right_chain, int(position))] for position in right_positions
    ]
    return np.asarray(matrix)[np.ix_(left_tokens, right_tokens)]


def validate_query_document(
    document: dict,
    expected_seed: int,
    expected_chain_ids: tuple[str, ...] = ("A", "B", "P", "H"),
) -> dict:
    if document.get("seeds") != [expected_seed]:
        raise ValueError(
            f"Query seeds {document.get('seeds')} do not match [{expected_seed}]"
        )
    queries = document.get("queries", {})
    if len(queries) != 1:
        raise ValueError(f"Expected one query, found {sorted(queries)}")
    query = next(iter(queries.values()))
    if query.get("use_paired_msas", True):
        raise ValueError("Paired MSAs must be disabled")
    if not query.get("use_main_msas", False):
        raise ValueError("Main MSAs must be enabled")
    chain_ids = tuple(chain["chain_ids"][0] for chain in query["chains"])
    if chain_ids != expected_chain_ids:
        raise ValueError(f"Expected chains {expected_chain_ids}, found {chain_ids}")
    forbidden = (
        "paired_msa_file_paths",
        "template_alignment_file_path",
        "template_entry_chain_ids",
        "template_cif_paths",
        "template_cif_chain_ids",
    )
    for chain in query["chains"]:
        populated = [key for key in forbidden if chain.get(key)]
        if populated:
            raise ValueError(
                f"Chain {chain['chain_ids']} has paired/template inputs {populated}"
            )
        main_paths = chain.get("main_msa_file_paths") or []
        if len(main_paths) != 1:
            raise ValueError(
                f"Chain {chain['chain_ids']} must have exactly one main MSA path"
            )
    return query


def validate_experiment_config(config: dict, expected_seed: int) -> None:
    settings = config.get("experiment_settings", {})
    if settings.get("seeds") != [expected_seed]:
        raise ValueError(
            f"Runner seeds {settings.get('seeds')} do not match [{expected_seed}]"
        )
    if settings.get("use_msa_server") is not False:
        raise ValueError("Inference unexpectedly enabled the MSA server")
    if settings.get("use_templates") is not False:
        raise ValueError("Inference unexpectedly enabled templates")


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(values: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if values.shape != targets.shape or values.ndim != 1:
        raise ValueError("Spearman inputs must be same-shaped vectors")
    first = average_ranks(values)
    second = average_ranks(targets)
    first -= first.mean()
    second -= second.mean()
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    if denominator == 0.0:
        return float("nan")
    return float((first @ second) / denominator)


def write_json_private(path: Path, value: dict) -> None:
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.chmod(path, 0o600)
