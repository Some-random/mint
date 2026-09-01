#!/usr/bin/env python
"""Extract independent-chain ESM-2 features for the Affibody--pMHC pairs."""

import argparse
import json
import os
import platform
import re
import socket
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path
from downstream.AffibodyMHC.extract_mint_features import (
    AFFIBODY_DESIGNED_POSITIONS,
    AA_ALPHABET,
    CHAIN_LENGTHS,
    PairCollator,
    SequencePairDataset,
    extract_batch_features,
    torch_load_compat,
)
from mint.helpers.extract import load_config
from mint.model.esm2 import ESM2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class ESM2FeatureWrapper:
    def __init__(self, cfg, checkpoint_path, device):
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=False,
        )
        checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
        _require("model" in checkpoint, "unexpected ESM-2 checkpoint format")
        upgraded = OrderedDict()
        for name, value in checkpoint["model"].items():
            upgraded[re.sub(r"^(encoder\.sentence_encoder\.|encoder\.)", "", name)] = value
        self.model.load_state_dict(upgraded, strict=True)
        del checkpoint
        self.model.to(device)
        self.model.eval()


def max_within_sequence_difference(values, sequence_hashes, start, end):
    maximum = 0.0
    for sequence_hash in sorted(set(sequence_hashes)):
        indices = np.flatnonzero(sequence_hashes == sequence_hash)
        if len(indices) <= 1:
            continue
        reference = values[indices[0], start:end]
        maximum = max(maximum, float(np.max(np.abs(values[indices, start:end] - reference))))
    return maximum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    script_path = Path(__file__).resolve()
    dependency_path = Path(__file__).resolve().with_name("extract_mint_features.py")
    output_parent = validate_private_output_path(args.output.parent, REPO_ROOT)
    manifest_parent = validate_private_output_path(args.manifest.parent, REPO_ROOT)
    _require(output_parent == manifest_parent, "output and manifest must share a private directory")
    for path, label in (
        (args.input, "input"),
        (args.checkpoint, "checkpoint"),
        (args.config, "config"),
    ):
        _require(path.is_file(), "{} file does not exist".format(label))
    _require(args.batch_size >= 1, "batch size must be positive")
    _require(not args.output.exists(), "output already exists; refusing overwrite")
    _require(not args.manifest.exists(), "manifest already exists; refusing overwrite")
    _require(torch.cuda.is_available(), "CUDA is not available")

    source_hash = sha256_file(args.input)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False, na_filter=False)
    required = {
        "pair_uid",
        "library",
        "measurement_missing",
        "chain1_sha256",
        "chain2_sha256",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
    }
    _require(required.issubset(frame.columns), "input sequence-table schema mismatch")
    _require(len(frame) == 228, "expected 228 designed rows")
    _require(frame["pair_uid"].nunique() == len(frame), "duplicate pair UID")
    _require(set(frame["library"]) == set(AFFIBODY_DESIGNED_POSITIONS), "unexpected library")
    for sequence, expected_length in (
        (frame["chain1_smart_hla_linker_peptide_sequence"], CHAIN_LENGTHS[0]),
        (frame["chain2_affibody_sequence"], CHAIN_LENGTHS[1]),
    ):
        _require(bool(sequence.map(len).eq(expected_length).all()), "sequence length mismatch")
        _require(
            bool(sequence.map(lambda value: set(value).issubset(AA_ALPHABET)).all()),
            "noncanonical sequence",
        )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    wrapper = ESM2FeatureWrapper(load_config(str(args.config)), str(args.checkpoint), device)
    loader = DataLoader(
        SequencePairDataset(frame),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    feature_blocks = {
        "esm2_chain_mean": [],
        "esm2_targeted_mean": [],
        "esm2_designed_site_mean": [],
        "esm2_joint_mean": [],
    }
    observed_uids = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, libraries, pair_uids) in enumerate(loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            mint_named = extract_batch_features(wrapper, chains, chain_ids, libraries)
            for mint_name, values in mint_named.items():
                esm_name = mint_name.replace("mint_", "esm2_", 1)
                feature_blocks[esm_name].append(values.cpu())
            observed_uids.extend(pair_uids)
            if (step + 1) % 20 == 0 or step + 1 == len(loader):
                print("embedded {}/{} batches".format(step + 1, len(loader)), flush=True)
    torch.cuda.synchronize(device)

    _require(observed_uids == frame["pair_uid"].tolist(), "embedding row order changed")
    arrays = {
        "pair_uid": np.asarray(observed_uids),
        "library": frame["library"].to_numpy(dtype=str),
        "measurement_missing": frame["measurement_missing"].to_numpy(dtype=str),
    }
    shapes = {}
    for name, blocks in feature_blocks.items():
        values = torch.cat(blocks, dim=0).float().numpy()
        expected_dimension = 1280 if name == "esm2_joint_mean" else 2560
        _require(values.shape == (len(frame), expected_dimension), "{} shape mismatch".format(name))
        _require(bool(np.isfinite(values).all()), "{} has non-finite values".format(name))
        arrays[name] = values
        shapes[name] = list(values.shape)

    chain_values = arrays["esm2_chain_mean"]
    chain1_difference = max_within_sequence_difference(
        chain_values,
        frame["chain1_sha256"].to_numpy(dtype=str),
        0,
        1280,
    )
    chain2_difference = max_within_sequence_difference(
        chain_values,
        frame["chain2_sha256"].to_numpy(dtype=str),
        1280,
        2560,
    )
    _require(chain1_difference <= 1e-5, "chain-1 ESM-2 embedding depends on partner")
    _require(chain2_difference <= 1e-5, "chain-2 ESM-2 embedding depends on partner")

    np.savez_compressed(str(args.output), **arrays)
    os.chmod(str(args.output), 0o600)
    _require(source_hash == sha256_file(args.input), "input sequence table changed")
    elapsed = time.time() - started
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 6),
        "hostname": socket.gethostname(),
        "source": {"path": str(args.input.resolve()), "sha256": source_hash},
        "code": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
            "pooling_dependency": str(dependency_path),
            "pooling_dependency_sha256": sha256_file(dependency_path),
        },
        "model": {
            "name": "esm2_t33_650M_UR50D",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "use_multimer": False,
            "layer": 33,
            "chain_attention": "independent; cross-chain positions masked",
            "pooling": {
                "esm2_chain_mean": "separate mean over all residues of each chain",
                "esm2_targeted_mean": "9-aa peptide mean plus designed Affibody-site mean",
                "esm2_designed_site_mean": "peptide positions 4/5 mean plus designed Affibody-site mean",
                "esm2_joint_mean": "residue-count-weighted mean of the two chain means",
            },
        },
        "independence_audit": {
            "max_same_chain1_embedding_difference_across_partners": chain1_difference,
            "max_same_chain2_embedding_difference_across_partners": chain2_difference,
            "tolerance": 1e-5,
        },
        "rows": {
            "designed": int(len(frame)),
            "measured": int(frame["measurement_missing"].eq("0").sum()),
        },
        "features": shapes,
        "runtime": {
            "device_argument": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "batch_size": int(args.batch_size),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "mode": "0600",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(args.manifest), 0o600)
    print(json.dumps({"output": str(args.output), "rows": len(frame), "seconds": elapsed}))


if __name__ == "__main__":
    run(parse_args())
