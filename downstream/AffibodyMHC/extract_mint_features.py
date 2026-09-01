#!/usr/bin/env python
"""Extract frozen MINT features for provider-specified Affibody--pMHC chains."""

import argparse
import hashlib
import inspect
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import sha256_file, validate_private_output_path
from mint.helpers.extract import CollateFn, MINTWrapper, load_config
from mint.helpers.predict import SimpleMLP


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
CHAIN_LENGTHS = (270, 58)
PEPTIDE_POSITIONS = tuple(range(262, 271))
PEPTIDE_DESIGNED_POSITIONS = (265, 266)
AFFIBODY_DESIGNED_POSITIONS = {
    "LibA": (13, 17, 27, 31),
    "LibB": (6, 10, 13, 14, 17),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def torch_load_compat(path, map_location):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


class SequencePairDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return (
            row["chain1_smart_hla_linker_peptide_sequence"],
            row["chain2_affibody_sequence"],
            row["library"],
            row["pair_uid"],
        )


class PairCollator:
    def __init__(self):
        self.sequence_collator = CollateFn(512)

    def __call__(self, batch):
        chain1, chain2, libraries, pair_uids = zip(*batch)
        chains, chain_ids = self.sequence_collator(list(zip(chain1, chain2)))
        return chains, chain_ids, list(libraries), list(pair_uids)


def mean_pool(representations, row_indices):
    _require(len(row_indices) > 0, "cannot pool an empty residue set")
    return representations[row_indices].mean(dim=0)


def residue_indices(tokens, chain_ids, model, batch_index, chain_id):
    valid = (
        chain_ids[batch_index].eq(chain_id)
        & ~tokens[batch_index].eq(model.cls_idx)
        & ~tokens[batch_index].eq(model.eos_idx)
        & ~tokens[batch_index].eq(model.padding_idx)
    )
    return torch.nonzero(valid, as_tuple=False).flatten()


def extract_batch_features(wrapper, chains, chain_ids, libraries):
    output = wrapper.model(chains, chain_ids, repr_layers=[33])
    representations = output["representations"][33]
    chain_means = []
    targeted_means = []
    designed_site_means = []
    joint_means = []
    for batch_index, library in enumerate(libraries):
        indices0 = residue_indices(chains, chain_ids, wrapper.model, batch_index, 0)
        indices1 = residue_indices(chains, chain_ids, wrapper.model, batch_index, 1)
        _require(len(indices0) == CHAIN_LENGTHS[0], "unexpected chain-1 residue count")
        _require(len(indices1) == CHAIN_LENGTHS[1], "unexpected chain-2 residue count")

        chain0_mean = mean_pool(representations[batch_index], indices0)
        chain1_mean = mean_pool(representations[batch_index], indices1)
        chain_means.append(torch.cat((chain0_mean, chain1_mean), dim=-1))
        joint_means.append(
            (CHAIN_LENGTHS[0] * chain0_mean + CHAIN_LENGTHS[1] * chain1_mean)
            / float(sum(CHAIN_LENGTHS))
        )

        peptide_indices = indices0[
            torch.tensor([value - 1 for value in PEPTIDE_POSITIONS], device=indices0.device)
        ]
        peptide_site_indices = indices0[
            torch.tensor(
                [value - 1 for value in PEPTIDE_DESIGNED_POSITIONS],
                device=indices0.device,
            )
        ]
        affibody_positions = AFFIBODY_DESIGNED_POSITIONS[library]
        affibody_site_indices = indices1[
            torch.tensor([value - 1 for value in affibody_positions], device=indices1.device)
        ]
        targeted_means.append(
            torch.cat(
                (
                    mean_pool(representations[batch_index], peptide_indices),
                    mean_pool(representations[batch_index], affibody_site_indices),
                ),
                dim=-1,
            )
        )
        designed_site_means.append(
            torch.cat(
                (
                    mean_pool(representations[batch_index], peptide_site_indices),
                    mean_pool(representations[batch_index], affibody_site_indices),
                ),
                dim=-1,
            )
        )
    return {
        "mint_chain_mean": torch.stack(chain_means),
        "mint_targeted_mean": torch.stack(targeted_means),
        "mint_designed_site_mean": torch.stack(designed_site_means),
        "mint_joint_mean": torch.stack(joint_means),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ppi-head", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    script_path = Path(__file__).resolve()
    output_parent = validate_private_output_path(args.output.parent, REPO_ROOT)
    manifest_parent = validate_private_output_path(args.manifest.parent, REPO_ROOT)
    _require(output_parent == manifest_parent, "output and manifest must share a private directory")
    for path, label in (
        (args.input, "input"),
        (args.checkpoint, "checkpoint"),
        (args.config, "config"),
    ):
        _require(path.is_file(), "{} file does not exist".format(label))
    if args.ppi_head is not None:
        _require(args.ppi_head.is_file(), "PPI-head file does not exist")
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
    dataset = SequencePairDataset(frame)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    wrapper = MINTWrapper(
        load_config(str(args.config)),
        str(args.checkpoint),
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()

    ppi_model = None
    if args.ppi_head is not None:
        ppi_model = SimpleMLP()
        ppi_model.load_state_dict(torch_load_compat(args.ppi_head, map_location="cpu"))
        ppi_model.to(device)
        ppi_model.eval()

    feature_blocks = {
        "mint_chain_mean": [],
        "mint_targeted_mean": [],
        "mint_designed_site_mean": [],
        "mint_joint_mean": [],
    }
    probabilities = []
    observed_uids = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, libraries, pair_uids) in enumerate(loader):
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            batch_features = extract_batch_features(wrapper, chains, chain_ids, libraries)
            for name, values in batch_features.items():
                feature_blocks[name].append(values.cpu())
            if ppi_model is not None:
                probabilities.append(torch.sigmoid(ppi_model(batch_features["mint_chain_mean"])).cpu())
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
        expected_dimension = 1280 if name == "mint_joint_mean" else 2560
        _require(values.shape == (len(frame), expected_dimension), "{} shape mismatch".format(name))
        _require(bool(np.isfinite(values).all()), "{} has non-finite values".format(name))
        arrays[name] = values
        shapes[name] = list(values.shape)
    if probabilities:
        probability = torch.cat(probabilities).flatten().float().numpy()
        _require(probability.shape == (len(frame),), "PPI probability shape mismatch")
        _require(bool(np.isfinite(probability).all()), "non-finite PPI probabilities")
        arrays["bernett_generic_ppi_probability"] = probability
        shapes["bernett_generic_ppi_probability"] = list(probability.shape)

    np.savez_compressed(str(args.output), **arrays)
    os.chmod(str(args.output), 0o600)
    _require(source_hash == sha256_file(args.input), "input sequence table changed")
    elapsed = time.time() - started
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 6),
        "hostname": socket.gethostname(),
        "source": {"path": str(args.input.resolve()), "sha256": source_hash},
        "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "model": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "use_multimer": True,
            "layer": 33,
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "pooling": {
                "mint_chain_mean": "separate mean over all residues of each chain",
                "mint_targeted_mean": "9-aa peptide mean plus designed Affibody-site mean",
                "mint_designed_site_mean": "peptide positions 4/5 mean plus designed Affibody-site mean",
                "mint_joint_mean": "residue-count-weighted mean of the two chain means",
            },
        },
        "ppi_head": None
        if args.ppi_head is None
        else {
            "path": str(args.ppi_head.resolve()),
            "sha256": sha256_file(args.ppi_head),
            "interpretation": "generic Bernett binary-PPI probability; not a retention model",
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
