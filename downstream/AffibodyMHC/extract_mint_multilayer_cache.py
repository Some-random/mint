#!/usr/bin/env python
"""Extract auditable, sharded chain-mean features from several frozen MINT layers.

This is a versioned extension of ``build_mint_weak_feature_cache.py``.  It reads
that program's immutable ``rows/cache_rows.csv`` table; it never rebuilds or
changes labels, sequences, or row order.  The expensive model call requests all
layers together.  For every layer, special tokens are excluded, residues are
mean-pooled separately within the smart-HLA-linker-peptide and Affibody chains,
and the two 1,280-dimensional means are concatenated.

Two restartable commands are provided::

    extract-shard  # row_index modulo shard_count; one command per GPU
    merge          # validate every shard and restore prepared row order

Outputs are refused unless they are under the repository's Git-ignored
``private_data`` tree.  Existing files and final output directories are never
overwritten.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import build_mint_weak_feature_cache as source_cache
from downstream.AffibodyMHC.code_only_baseline import (
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.extract_mint_features import PairCollator
from mint.helpers.extract import MINTWrapper, load_config


SCHEMA_VERSION = "mint-multilayer-chain-mean-cache-v1"
SOURCE_SCHEMA_VERSION = source_cache.SCHEMA_VERSION
DEFAULT_LAYERS = (1, 5, 9, 13, 17, 21, 25, 29, 33)
FINAL_LAYER = 33
RESIDUE_DIMENSION = 1280
FEATURE_DIMENSION = 2 * RESIDUE_DIMENSION
FEATURE_DTYPE = "float32"
CHAIN_LENGTHS = (270, 58)
FINAL_CACHE_FILENAME = "mint_multilayer_chain_mean_features.npz"
FINAL_MANIFEST_FILENAME = "manifest.json"
REPRESENTATION_LAYERS_KEY = "representation_layers"
POOLING_DESCRIPTION = (
    "exclude cls/eos/padding; mean residues separately within chain_id 0 and "
    "chain_id 1; concatenate chain 0 then chain 1"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def parse_layers(value):
    """Parse and strictly validate an ordered MINT representation-layer list."""
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        _require(bool(pieces), "at least one representation layer is required")
        try:
            layers = tuple(int(piece) for piece in pieces)
        except ValueError:
            raise ValueError("layers must be comma-separated integers")
    else:
        layers = tuple(int(layer) for layer in value)
    _require(bool(layers), "at least one representation layer is required")
    _require(len(set(layers)) == len(layers), "representation layers must be unique")
    _require(tuple(sorted(layers)) == layers, "representation layers must be increasing")
    _require(all(1 <= layer <= FINAL_LAYER for layer in layers), "layer is outside 1..33")
    _require(FINAL_LAYER in layers, "layer 33 is required to anchor the existing baseline")
    return layers


def feature_name(layer):
    layer = int(layer)
    _require(1 <= layer <= FINAL_LAYER, "layer is outside 1..33")
    return "mint_layer_{:02d}_chain_mean".format(layer)


def feature_names(layers):
    layers = parse_layers(layers)
    names = tuple(feature_name(layer) for layer in layers)
    _require(len(set(names)) == len(names), "feature names are not unique")
    return names


def layer_semantics(layer):
    """Describe the exact semantics implemented by ``mint.model.esm2.ESM2``."""
    layer = int(layer)
    _require(1 <= layer <= FINAL_LAYER, "layer is outside 1..33")
    if layer == FINAL_LAYER:
        return (
            "output of transformer block 33 after emb_layer_norm_after; this is "
            "the same representation requested by the existing MINTWrapper"
        )
    return "output of transformer block {} before emb_layer_norm_after".format(layer)


def _ensure_private_directory(path, must_be_new):
    path = validate_private_output_path(Path(path), REPO_ROOT)
    if must_be_new:
        _require(not path.exists(), "output directory exists; refusing overwrite")
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=not must_be_new)
    except FileExistsError:
        raise ValueError("output directory exists; refusing overwrite")
    _require(path.is_dir(), "output path is not a directory")
    os.chmod(str(path), 0o700)
    _require(private_mode(path) == "0700", "output directory is not mode 0700")
    return path


def _atomic_write_json(path, payload):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with open(str(temporary), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "JSON output is not mode 0600")


def _atomic_write_npz(path, arrays, compressed=True):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with open(str(temporary), "wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "NPZ output is not mode 0600")


def shard_filename(shard_index, shard_count):
    return "multilayer-features-shard-{:03d}-of-{:03d}.npz".format(
        int(shard_index), int(shard_count)
    )


def shard_manifest_filename(shard_index, shard_count):
    return "multilayer-features-shard-{:03d}-of-{:03d}.manifest.json".format(
        int(shard_index), int(shard_count)
    )


def load_prepared_input(input_dir):
    """Load and revalidate the immutable v1 row table."""
    frame, manifest, rows_path, manifest_path = source_cache.load_prepared_input(
        Path(input_dir)
    )
    _require(
        manifest.get("schema_version") == SOURCE_SCHEMA_VERSION,
        "source cache schema mismatch",
    )
    return frame, manifest, rows_path, manifest_path


def select_shard_rows(frame, shard_index, shard_count):
    return source_cache.select_shard_rows(frame, shard_index, shard_count)


def metadata_arrays(frame):
    return source_cache.metadata_arrays(frame)


def _model_special_indices(model):
    values = {}
    for name in ("cls_idx", "eos_idx", "padding_idx"):
        _require(hasattr(model, name), "MINT model lacks {}".format(name))
        values[name] = int(getattr(model, name))
    _require(len(set(values.values())) == 3, "MINT special-token indices are not distinct")
    return values


def residue_masks(tokens, chain_ids, model, expected_chain_lengths=CHAIN_LENGTHS):
    """Return boolean masks for real residues in each of the two MINT chains."""
    _require(tokens.ndim == 2, "tokens must have shape [batch, tokens]")
    _require(chain_ids.shape == tokens.shape, "chain_ids must match token shape")
    special = _model_special_indices(model)
    valid = (
        ~tokens.eq(special["cls_idx"])
        & ~tokens.eq(special["eos_idx"])
        & ~tokens.eq(special["padding_idx"])
    )
    masks = tuple(valid & chain_ids.eq(chain_id) for chain_id in (0, 1))
    for chain_id, (mask, expected_length) in enumerate(zip(masks, expected_chain_lengths)):
        counts = mask.sum(dim=1)
        _require(
            bool(counts.eq(int(expected_length)).all().item()),
            "unexpected chain-{} residue count".format(chain_id),
        )
    _require(
        bool((masks[0] | masks[1]).eq(valid).all().item()),
        "a residue has a chain ID other than 0 or 1",
    )
    return masks


def pool_representations(representations, masks, residue_dimension=RESIDUE_DIMENSION):
    """Pool one layer into ``[batch, chain0 || chain1]`` in float32."""
    _require(representations.ndim == 3, "representation must have shape [B,T,C]")
    _require(
        int(representations.shape[-1]) == int(residue_dimension),
        "unexpected MINT residue representation dimension",
    )
    means = []
    for mask in masks:
        _require(
            tuple(mask.shape) == tuple(representations.shape[:2]),
            "residue mask/representation shape mismatch",
        )
        counts = mask.sum(dim=1, keepdim=True).to(dtype=torch.float32)
        _require(bool(counts.gt(0).all().item()), "cannot pool an empty chain")
        values = representations.float() * mask.unsqueeze(-1).to(dtype=torch.float32)
        means.append(values.sum(dim=1) / counts)
    output = torch.cat(means, dim=-1)
    _require(
        tuple(output.shape) == (representations.shape[0], 2 * int(residue_dimension)),
        "pooled feature shape mismatch",
    )
    _require(bool(torch.isfinite(output).all().item()), "pooled features are non-finite")
    return output


def extract_multilayer_chain_means(
    model,
    chains,
    chain_ids,
    layers=DEFAULT_LAYERS,
    expected_chain_lengths=CHAIN_LENGTHS,
    residue_dimension=RESIDUE_DIMENSION,
):
    """Request all layers in one forward pass and return named pooled tensors."""
    layers = parse_layers(layers)
    output = model(chains, chain_ids, repr_layers=list(layers))
    _require(isinstance(output, dict), "MINT forward did not return a dictionary")
    representations = output.get("representations")
    _require(isinstance(representations, dict), "MINT output lacks representations")
    _require(
        set(representations) == set(layers),
        "MINT returned representation layers {} instead of {}".format(
            sorted(representations), list(layers)
        ),
    )
    masks = residue_masks(chains, chain_ids, model, expected_chain_lengths)
    features = {}
    for layer in layers:
        representation = representations[layer]
        _require(
            tuple(representation.shape[:2]) == tuple(chains.shape),
            "layer {} token shape mismatch".format(layer),
        )
        features[feature_name(layer)] = pool_representations(
            representation, masks, residue_dimension=residue_dimension
        )
    _require(tuple(features) == feature_names(layers), "feature naming/order mismatch")
    return features


def validate_layer33_single_request(
    model,
    chains,
    chain_ids,
    observed_layer33,
    expected_chain_lengths=CHAIN_LENGTHS,
    residue_dimension=RESIDUE_DIMENSION,
):
    """Prove the multi-request layer 33 equals the historical single request."""
    output = model(chains, chain_ids, repr_layers=[FINAL_LAYER])
    _require(
        isinstance(output, dict)
        and isinstance(output.get("representations"), dict)
        and set(output["representations"]) == {FINAL_LAYER},
        "single-layer MINT request did not return only layer 33",
    )
    masks = residue_masks(chains, chain_ids, model, expected_chain_lengths)
    expected = pool_representations(
        output["representations"][FINAL_LAYER],
        masks,
        residue_dimension=residue_dimension,
    )
    _require(
        observed_layer33.dtype == torch.float32 and expected.dtype == torch.float32,
        "layer-33 pooled output is not float32",
    )
    _require(
        torch.equal(observed_layer33, expected),
        "multi-layer request changed layer-33 chain-mean features",
    )
    return True


def _source_hashes(rows_path, prepare_manifest_path, checkpoint, config):
    script_path = Path(__file__).resolve()
    dependencies = {
        "source_cache_builder": Path(source_cache.__file__).resolve(),
        "pair_collator": REPO_ROOT / "downstream" / "AffibodyMHC" / "extract_mint_features.py",
        "mint_wrapper": REPO_ROOT / "mint" / "helpers" / "extract.py",
        "mint_model": REPO_ROOT / "mint" / "model" / "esm2.py",
    }
    return {
        "rows_sha256": sha256_file(rows_path),
        "prepare_manifest_sha256": sha256_file(prepare_manifest_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "script_sha256": sha256_file(script_path),
        "mint_source_tree_sha256": source_cache.sha256_source_tree(REPO_ROOT / "mint"),
        "dependency_sha256": {
            name: sha256_file(path) for name, path in sorted(dependencies.items())
        },
    }


def build_feature_contract(
    hashes,
    layers,
    shard_count,
    batch_size,
    runtime,
    model_config,
):
    layers = parse_layers(layers)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "inputs": dict(hashes),
        "model": {
            "name": "MINT",
            "use_multimer": True,
            "sep_chains": True,
            "encoder_layers": int(model_config["encoder_layers"]),
            "encoder_embed_dim": int(model_config["encoder_embed_dim"]),
        },
        "features": {
            "layers": list(layers),
            "names": list(feature_names(layers)),
            "residue_dimension": RESIDUE_DIMENSION,
            "feature_dimension": FEATURE_DIMENSION,
            "dtype": FEATURE_DTYPE,
            "pooling": POOLING_DESCRIPTION,
            "layer_semantics": {
                str(layer): layer_semantics(layer) for layer in layers
            },
        },
        "sharding": {
            "rule": "row_index modulo shard_count equals shard_index",
            "shard_count": int(shard_count),
        },
        "runtime": dict(runtime),
        "batch_size": int(batch_size),
    }
    return canonical_json_sha256(payload), payload


def _validate_config(config, layers):
    _require(hasattr(config, "encoder_layers"), "MINT config lacks encoder_layers")
    _require(hasattr(config, "encoder_embed_dim"), "MINT config lacks encoder_embed_dim")
    _require(int(config.encoder_layers) == FINAL_LAYER, "MINT checkpoint must have 33 layers")
    _require(
        int(config.encoder_embed_dim) == RESIDUE_DIMENSION,
        "MINT checkpoint must use 1,280-dimensional residues",
    )
    _require(max(parse_layers(layers)) <= int(config.encoder_layers), "requested layer absent")


def _runtime_contract(device, batch_size):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": [int(v) for v in torch.cuda.get_device_capability(device)],
        "batch_size": int(batch_size),
    }


def run_extract_shard(args):
    started = time.time()
    layers = parse_layers(args.layers)
    _require(int(args.batch_size) >= 1, "batch size must be positive")
    _require(int(args.shard_count) >= 1, "shard count must be positive")
    _require(0 <= int(args.shard_index) < int(args.shard_count), "shard index out of range")
    _require(args.checkpoint.is_file(), "MINT checkpoint does not exist")
    _require(args.config.is_file(), "MINT config does not exist")
    frame, _, rows_path, prepare_manifest_path = load_prepared_input(args.input_dir)
    shard = select_shard_rows(frame, args.shard_index, args.shard_count)

    device = torch.device(args.device)
    _require(device.type == "cuda", "extract-shard requires a CUDA device")
    _require(torch.cuda.is_available(), "CUDA is not available")
    torch.cuda.set_device(device)
    config = load_config(str(args.config))
    _validate_config(config, layers)
    hashes = _source_hashes(rows_path, prepare_manifest_path, args.checkpoint, args.config)
    runtime_contract = _runtime_contract(device, args.batch_size)
    contract_hash, contract_payload = build_feature_contract(
        hashes,
        layers,
        args.shard_count,
        args.batch_size,
        runtime_contract,
        {
            "encoder_layers": config.encoder_layers,
            "encoder_embed_dim": config.encoder_embed_dim,
        },
    )

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=False)
    output_path = output_dir / shard_filename(args.shard_index, args.shard_count)
    manifest_path = output_dir / shard_manifest_filename(args.shard_index, args.shard_count)
    _require(not output_path.exists(), "shard output exists; refusing overwrite")
    _require(not manifest_path.exists(), "shard manifest exists; refusing overwrite")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(
        source_cache.CachePairDataset(shard),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    wrapper = MINTWrapper(
        config,
        str(args.checkpoint),
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()
    observed_uids = []
    blocks = {name: [] for name in feature_names(layers)}
    layer33_validated = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, _, cache_uids) in enumerate(loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            batch_features = extract_multilayer_chain_means(
                wrapper.model, chains, chain_ids, layers=layers
            )
            if not layer33_validated:
                validate_layer33_single_request(
                    wrapper.model,
                    chains,
                    chain_ids,
                    batch_features[feature_name(FINAL_LAYER)],
                )
                layer33_validated = True
            for name in feature_names(layers):
                blocks[name].append(batch_features[name].cpu().numpy())
            observed_uids.extend(cache_uids)
            if (step + 1) % 50 == 0 or step + 1 == len(loader):
                print(
                    "shard {}/{}: embedded {}/{} batches".format(
                        args.shard_index, args.shard_count, step + 1, len(loader)
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    _require(layer33_validated, "layer-33 validation did not run")
    _require(observed_uids == shard["cache_uid"].tolist(), "embedding row order changed")
    arrays = metadata_arrays(shard)
    arrays[REPRESENTATION_LAYERS_KEY] = np.asarray(layers, dtype=np.int64)
    feature_shapes = {}
    for name in feature_names(layers):
        values = np.concatenate(blocks[name], axis=0).astype(np.float32, copy=False)
        _require(values.shape == (len(shard), FEATURE_DIMENSION), "feature shape mismatch")
        _require(bool(np.isfinite(values).all()), "features contain non-finite values")
        arrays[name] = values
        feature_shapes[name] = list(values.shape)

    current_hashes = _source_hashes(
        rows_path, prepare_manifest_path, args.checkpoint, args.config
    )
    _require(current_hashes == hashes, "an input or source file changed during extraction")
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "extract_shard",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "hostname": socket.gethostname(),
        "input": {
            "rows_csv": {"path": str(rows_path.resolve()), "sha256": hashes["rows_sha256"]},
            "prepare_manifest": {
                "path": str(prepare_manifest_path.resolve()),
                "sha256": hashes["prepare_manifest_sha256"],
            },
        },
        "contract": {"sha256": contract_hash, "payload": contract_payload},
        "model": {
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": hashes["checkpoint_sha256"]},
            "config": {"path": str(args.config.resolve()), "sha256": hashes["config_sha256"]},
            "use_multimer": True,
            "sep_chains": True,
            "encoder_layers": int(config.encoder_layers),
            "encoder_embed_dim": int(config.encoder_embed_dim),
            "layer33_single_request_exact_match": True,
            "layer33_semantics": layer_semantics(FINAL_LAYER),
        },
        "features": {
            "layers": list(layers),
            "names": list(feature_names(layers)),
            "shapes": feature_shapes,
            "dtype": FEATURE_DTYPE,
            "pooling": POOLING_DESCRIPTION,
        },
        "sharding": {
            "rule": "row_index modulo shard_count equals shard_index",
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "rows": int(len(shard)),
            "row_index_sha256": hashlib.sha256(
                arrays["row_index"].astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "cache_uid_sha256": sha256_text("\n".join(observed_uids)),
        },
        "runtime": {
            **runtime_contract,
            "device_argument": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "platform": platform.platform(),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "stage": "extract_shard",
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "rows": int(len(shard)),
        "layers": list(layers),
        "output": str(output_path),
    }, sort_keys=True))


def _validate_contract_manifest(manifest, layers, shard_count, expected_contract=None):
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "shard schema mismatch")
    _require(manifest.get("stage") == "extract_shard", "not an extraction manifest")
    contract = manifest.get("contract", {}).get("sha256")
    payload = manifest.get("contract", {}).get("payload")
    _require(isinstance(payload, dict) and bool(contract), "shard lacks feature contract")
    _require(canonical_json_sha256(payload) == contract, "contract payload/hash mismatch")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "contract schema mismatch")
    _require(payload.get("source_schema_version") == SOURCE_SCHEMA_VERSION, "source schema mismatch")
    features = payload.get("features", {})
    _require(features.get("layers") == list(layers), "contract layer mismatch")
    _require(features.get("names") == list(feature_names(layers)), "contract feature-name mismatch")
    _require(features.get("feature_dimension") == FEATURE_DIMENSION, "contract dimension mismatch")
    _require(features.get("dtype") == FEATURE_DTYPE, "contract dtype mismatch")
    _require(
        payload.get("sharding", {}).get("shard_count") == int(shard_count),
        "contract shard-count mismatch",
    )
    _require(payload.get("model", {}).get("encoder_layers") == FINAL_LAYER, "model layer-count mismatch")
    _require(
        payload.get("model", {}).get("encoder_embed_dim") == RESIDUE_DIMENSION,
        "model dimension mismatch",
    )
    _require(
        manifest.get("model", {}).get("layer33_single_request_exact_match") is True,
        "shard lacks successful layer-33 validation",
    )
    if expected_contract is not None:
        _require(contract == expected_contract, "shards use different feature contracts")
    return contract, payload


def _load_and_validate_shard(
    shard_path,
    manifest_path,
    frame,
    shard_index,
    shard_count,
    layers=DEFAULT_LAYERS,
    expected_contract=None,
):
    layers = parse_layers(layers)
    with open(str(manifest_path), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    contract, payload = _validate_contract_manifest(
        manifest, layers, shard_count, expected_contract=expected_contract
    )
    sharding = manifest.get("sharding", {})
    _require(sharding.get("shard_index") == int(shard_index), "shard-index mismatch")
    _require(sharding.get("shard_count") == int(shard_count), "shard-count mismatch")
    _require(manifest.get("output", {}).get("sha256") == sha256_file(shard_path), "shard output hash mismatch")
    expected = select_shard_rows(frame, shard_index, shard_count)
    expected_keys = (
        set(source_cache.NPZ_METADATA_COLUMNS)
        | {REPRESENTATION_LAYERS_KEY}
        | set(feature_names(layers))
    )
    with np.load(str(shard_path), allow_pickle=False) as archive:
        _require(set(archive.files) == expected_keys, "shard NPZ schema mismatch")
        observed_layers = np.asarray(archive[REPRESENTATION_LAYERS_KEY], dtype=np.int64)
        _require(np.array_equal(observed_layers, np.asarray(layers)), "stored layer list mismatch")
        row_index = np.asarray(archive["row_index"], dtype=np.int64)
        expected_index = expected["row_index"].astype(np.int64).to_numpy()
        _require(np.array_equal(row_index, expected_index), "shard row assignment mismatch")
        _require(
            hashlib.sha256(row_index.astype("<i8", copy=False).tobytes()).hexdigest()
            == sharding.get("row_index_sha256"),
            "shard row-index hash mismatch",
        )
        for column in source_cache.NPZ_METADATA_COLUMNS:
            if column == "row_index":
                continue
            _require(
                np.array_equal(
                    np.asarray(archive[column]).astype(str),
                    expected[column].to_numpy(dtype=str),
                ),
                "shard metadata mismatch in {}".format(column),
            )
        values = {}
        for name in feature_names(layers):
            raw = archive[name]
            _require(raw.dtype == np.float32, "{} is not stored as float32".format(name))
            value = np.asarray(raw, dtype=np.float32)
            _require(value.shape == (len(expected), FEATURE_DIMENSION), "{} shape mismatch".format(name))
            _require(bool(np.isfinite(value).all()), "{} has non-finite values".format(name))
            values[name] = value.copy()
    return manifest, expected_index, values, contract, payload


def run_merge(args):
    started = time.time()
    layers = parse_layers(args.layers)
    _require(int(args.shard_count) >= 1, "shard count must be positive")
    frame, prepare_manifest, rows_path, prepare_manifest_path = load_prepared_input(args.input_dir)
    _require(args.shards_dir.is_dir(), "shard directory does not exist")
    rows_hash = sha256_file(rows_path)
    prepare_hash = sha256_file(prepare_manifest_path)
    merged = {
        name: np.empty((len(frame), FEATURE_DIMENSION), dtype=np.float32)
        for name in feature_names(layers)
    }
    covered = np.zeros(len(frame), dtype=bool)
    expected_contract = None
    contract_payload = None
    first_manifest = None
    shard_records = []
    for shard_index in range(int(args.shard_count)):
        shard_path = args.shards_dir / shard_filename(shard_index, args.shard_count)
        manifest_path = args.shards_dir / shard_manifest_filename(shard_index, args.shard_count)
        _require(shard_path.is_file(), "missing shard {}".format(shard_index))
        _require(manifest_path.is_file(), "missing shard manifest {}".format(shard_index))
        manifest, indices, values, contract, payload = _load_and_validate_shard(
            shard_path,
            manifest_path,
            frame,
            shard_index,
            int(args.shard_count),
            layers=layers,
            expected_contract=expected_contract,
        )
        if expected_contract is None:
            expected_contract = contract
            contract_payload = payload
            first_manifest = manifest
        _require(
            manifest.get("input", {}).get("rows_csv", {}).get("sha256") == rows_hash,
            "shard was extracted from a different row table",
        )
        _require(
            manifest.get("input", {}).get("prepare_manifest", {}).get("sha256") == prepare_hash,
            "shard was extracted from a different prepare manifest",
        )
        _require(not bool(covered[indices].any()), "duplicate row coverage across shards")
        for name in feature_names(layers):
            merged[name][indices] = values[name]
        covered[indices] = True
        shard_records.append({
            "shard_index": shard_index,
            "rows": int(len(indices)),
            "npz_path": str(shard_path.resolve()),
            "npz_sha256": sha256_file(shard_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "hostname": manifest.get("hostname"),
            "gpu_name": manifest.get("runtime", {}).get("gpu_name"),
        })
    _require(bool(covered.all()), "shards do not cover every prepared row")
    for name, values in merged.items():
        _require(bool(np.isfinite(values).all()), "merged {} is non-finite".format(name))

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=True)
    output_path = output_dir / FINAL_CACHE_FILENAME
    manifest_path = output_dir / FINAL_MANIFEST_FILENAME
    arrays = metadata_arrays(frame)
    arrays[REPRESENTATION_LAYERS_KEY] = np.asarray(layers, dtype=np.int64)
    arrays.update(merged)
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    feature_manifest = [
        {
            "layer": layer,
            "name": feature_name(layer),
            "shape": list(merged[feature_name(layer)].shape),
            "dtype": str(merged[feature_name(layer)].dtype),
            "semantics": layer_semantics(layer),
        }
        for layer in layers
    ]
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "merge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "input": {
            "rows_csv": {"path": str(rows_path.resolve()), "sha256": rows_hash},
            "prepare_manifest": {"path": str(prepare_manifest_path.resolve()), "sha256": prepare_hash},
            "prepare_summary": prepare_manifest.get("rows"),
            "shards": shard_records,
        },
        "contract": {"sha256": expected_contract, "payload": contract_payload},
        "model": first_manifest.get("model"),
        "features": {
            "layers": list(layers),
            "by_layer": feature_manifest,
            "pooling": POOLING_DESCRIPTION,
            "row_order": "ascending row_index from prepared cache_rows.csv",
        },
        "merge_code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "directory_mode": private_mode(output_dir),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
        },
    }
    _atomic_write_json(manifest_path, output_manifest)
    print(json.dumps({
        "stage": "merge",
        "rows": int(len(frame)),
        "layers": list(layers),
        "output": str(output_path),
    }, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    default_input = REPO_ROOT / "private_data" / "derived" / "mint_weak_cache_v1" / "rows"

    extract = subparsers.add_parser("extract-shard", help="extract one deterministic GPU shard")
    extract.add_argument("--input-dir", type=Path, default=default_input)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "mint.ckpt")
    extract.add_argument("--config", type=Path, default=REPO_ROOT / "data" / "esm2_t33_650M_UR50D.json")
    extract.add_argument("--shard-index", type=int, required=True)
    extract.add_argument("--shard-count", type=int, required=True)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--layers", default=",".join(str(v) for v in DEFAULT_LAYERS))
    extract.add_argument("--uncompressed", action="store_true")

    merge = subparsers.add_parser("merge", help="validate and merge every GPU shard")
    merge.add_argument("--input-dir", type=Path, default=default_input)
    merge.add_argument("--shards-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--shard-count", type=int, required=True)
    merge.add_argument("--layers", default=",".join(str(v) for v in DEFAULT_LAYERS))
    merge.add_argument("--uncompressed", action="store_true")

    args = parser.parse_args(argv)
    _require(args.command is not None, "a subcommand is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "extract-shard":
        run_extract_shard(args)
    elif args.command == "merge":
        run_merge(args)
    else:
        raise ValueError("unknown command {}".format(args.command))


if __name__ == "__main__":
    main()
