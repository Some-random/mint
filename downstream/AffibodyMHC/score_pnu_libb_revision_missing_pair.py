#!/usr/bin/env python
"""Score the revised LibB AH x LIFTK pair with saved frozen-MINT PNU heads.

This is an outcome-blind repair utility.  It reads the already-extracted
layer-33 MINT feature for AH x LIFTK and applies only model heads that were
saved before the provider supplied the missing retention value.  Archived
prediction files are projected to identity, arm, setting, and score columns
solely to prove parity on the original 119-pair panel.
"""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "pnu-libb-mint-missing-pair-scores-v1"
TARGET_PAIR_UID = "6e8454b1ba8587952e0d"
TARGET_SEQUENCE_PAIR_SHA256 = (
    "24b85d9e1da9d364c87774ed3bf58ab1d3b87dc09aa9c8d143a5c183a792e1c0"
)
EXPECTED_CACHE_SHA256 = (
    "309ec02050820cd1a8c0dc39805dee858b70e46f451e331ab4b6c132086b6e00"
)
EXPECTED_OLD_ROWS = 119


MODEL_SPECS = (
    {
        "model_id": "frozen_mint_pn_auroc_rebaseline",
        "kind": "sklearn_saved_float32_head",
        "head": "private_data/experiments/pnu_mint_layer33_controls_v1/LibB/fitted_heads.npz",
        "head_sha256": "c5ba0b137074a0db1c3c5b6035f71fa0c3191e150ee3163119cf57c88f12490e",
        "prefix": "LibB_sklearn",
        "predictions": "private_data/experiments/pnu_mint_layer33_controls_v1/LibB/retention_predictions.csv",
        "predictions_sha256": "81f19bc97c8721df68fd9bf67d43b02d571d1d3c895c46727869677db5bbe98a",
        "arm": "balanced_sklearn_pn_control",
        "pi": None,
        "eta": None,
        "C": 0.01,
        "epoch": 0,
        "parity_tolerance": 5e-8,
    },
    {
        "model_id": "frozen_mint_pnu_pi0p02_eta0",
        "kind": "torch_saved_float32_head",
        "head": "private_data/experiments/pnu_mint_layer33_grid80_extension_v1/LibB_pi0p02_eta0/fitted_heads.npz",
        "head_sha256": "7c9de331e84dfb877ca1a967fa0518ab20a1ff7e54a18f8bbb78a8cb53922b1a",
        "prefix": "LibB_seed17_pi0p02_eta0p0",
        "predictions": "private_data/experiments/pnu_mint_layer33_grid80_extension_v1/LibB_pi0p02_eta0/retention_predictions.csv",
        "predictions_sha256": "f5d17ce8a8f09938a650439d726fe7d372388788ec14f16d6da37b6390213bb0",
        "arm": "prior_weighted_pn_eta0",
        "pi": 0.02,
        "eta": 0.0,
        "C": 0.001,
        "epoch": 78,
        "parity_tolerance": 0.0,
    },
    {
        "model_id": "frozen_mint_pnu_pi0p05_eta0",
        "kind": "torch_saved_float32_head",
        "head": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p05_eta0/fitted_heads.npz",
        "head_sha256": "85b146de2f0df6ab79bc4d6d0278c33b06c29f676981e7195b6fba31d6b1d9c4",
        "prefix": "LibB_seed17_pi0p05_eta0p0",
        "predictions": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p05_eta0/retention_predictions.csv",
        "predictions_sha256": "9646de22b162a049cdf34c5dd76b82502f057e75c33c78035769b613d83e95e1",
        "arm": "prior_weighted_pn_eta0",
        "pi": 0.05,
        "eta": 0.0,
        "C": 0.001,
        "epoch": 39,
        "parity_tolerance": 0.0,
    },
    {
        "model_id": "frozen_mint_pnu_pi0p10_eta0p25",
        "kind": "torch_saved_float32_head",
        "head": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p1_eta0p25/fitted_heads.npz",
        "head_sha256": "c5119771701a89640e08098a9dfbc5a964c78bccfa6e29aaf04fca31728d1a6c",
        "prefix": "LibB_seed17_pi0p1_eta0p25",
        "predictions": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p1_eta0p25/retention_predictions.csv",
        "predictions_sha256": "23b8668906aab51a18489f16e4d6b6fd09b9edb13a3024d72f4426c5ab357e89",
        "arm": "nnpnu_eta0p25",
        "pi": 0.10,
        "eta": 0.25,
        "C": 0.001,
        "epoch": 39,
        "parity_tolerance": 0.0,
    },
    {
        "model_id": "frozen_mint_pnu_pi0p15_eta0p25",
        "kind": "torch_saved_float32_head",
        "head": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p15_eta0p25/fitted_heads.npz",
        "head_sha256": "57fcf89d2d59cf25b73d0419f611cd4c41e949686073dcc91bfd139eb438b557",
        "prefix": "LibB_seed17_pi0p15_eta0p25",
        "predictions": "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p15_eta0p25/retention_predictions.csv",
        "predictions_sha256": "88be583477c260b3ea26af2807f4d21b636cf4533ee35780dd2d71f795253fc9",
        "arm": "nnpnu_eta0p25",
        "pi": 0.15,
        "eta": 0.25,
        "C": 0.01,
        "epoch": 39,
        "parity_tolerance": 0.0,
    },
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty(values.shape, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def source_record(path, expected_sha256):
    path = Path(path)
    require(path.is_file(), "missing source: {}".format(path))
    observed = sha256_file(path)
    require(observed == expected_sha256, "source hash changed: {}".format(path))
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.datetime.utcfromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }


def load_archived_scores(spec):
    columns = ["pair_uid", "arm", "pi", "eta", "C", "epoch", "score"]
    frame = pd.read_csv(
        str(REPO_ROOT / spec["predictions"]),
        usecols=columns,
        dtype={"pair_uid": str, "arm": str},
        float_precision="round_trip",
    )
    frame = frame.loc[frame["arm"].eq(spec["arm"])].copy()
    if spec["pi"] is not None:
        frame = frame.loc[
            np.isclose(pd.to_numeric(frame["pi"]), float(spec["pi"]))
            & np.isclose(pd.to_numeric(frame["eta"]), float(spec["eta"]))
            & np.isclose(pd.to_numeric(frame["C"]), float(spec["C"]))
            & pd.to_numeric(frame["epoch"]).astype(int).eq(int(spec["epoch"]))
        ].copy()
    require(len(frame) == EXPECTED_OLD_ROWS, "archived score row count changed")
    require(not bool(frame["pair_uid"].duplicated().any()), "duplicate archived pair UID")
    return frame.sort_values("pair_uid").reset_index(drop=True)


def load_head(spec):
    prefix = spec["prefix"]
    names = tuple(prefix + suffix for suffix in ("_weight", "_bias", "_mean", "_scale"))
    with np.load(str(REPO_ROOT / spec["head"]), allow_pickle=False) as archive:
        require(set(names).issubset(archive.files), "saved head arrays changed")
        arrays = {name: np.asarray(archive[name]).copy() for name in names}
    weight = arrays[prefix + "_weight"]
    bias = arrays[prefix + "_bias"]
    mean = arrays[prefix + "_mean"]
    scale = arrays[prefix + "_scale"]
    require(weight.shape == (1, 2560), "saved weight shape changed")
    require(bias.shape == (1,), "saved bias shape changed")
    require(mean.shape == (2560,) and scale.shape == (2560,), "standardizer shape changed")
    require(bool(np.isfinite(weight).all()), "non-finite saved weight")
    require(bool(np.isfinite(bias).all()), "non-finite saved bias")
    require(bool(np.isfinite(mean).all() and np.isfinite(scale).all()), "non-finite standardizer")
    require(bool((scale > 0).all()), "non-positive standardization scale")
    return weight, bias, mean, scale, names


def parity_record(recomputed, archived, tolerance):
    difference = np.asarray(recomputed, dtype=np.float64) - archived["score"].to_numpy(float)
    absolute = np.abs(difference)
    maximum = float(np.max(absolute))
    return {
        "rows": int(len(absolute)),
        "max_abs_difference": maximum,
        "mean_abs_difference": float(np.mean(absolute)),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference * difference))),
        "bitwise_equal_rows": int(np.sum(absolute == 0.0)),
        "bitwise_equal_all": bool(np.all(absolute == 0.0)),
        "tolerance": float(tolerance),
        "passed": bool(maximum <= float(tolerance)),
    }


def atomic_json(path, payload):
    require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with open(str(temporary), "x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args):
    started = time.time()
    require(torch.cuda.is_available(), "CUDA is required for exact neural-head parity")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cache_path = REPO_ROOT / args.cache
    cache_source = source_record(cache_path, EXPECTED_CACHE_SHA256)
    required_cache_arrays = (
        "pair_uid",
        "library",
        "peptide_design_code",
        "affibody_design_code",
        "sequence_pair_sha256",
        "mint_chain_mean",
    )
    with np.load(str(cache_path), allow_pickle=False) as archive:
        require(set(required_cache_arrays).issubset(archive.files), "cache schema changed")
        pair_uid = np.asarray(archive["pair_uid"]).astype(str)
        library = np.asarray(archive["library"]).astype(str)
        peptide = np.asarray(archive["peptide_design_code"]).astype(str)
        affibody = np.asarray(archive["affibody_design_code"]).astype(str)
        sequence_pair = np.asarray(archive["sequence_pair_sha256"]).astype(str)
        features = np.asarray(archive["mint_chain_mean"], dtype=np.float32).copy()
    require(features.shape == (len(pair_uid), 2560), "feature cache shape changed")
    require(bool(np.isfinite(features).all()), "feature cache contains non-finite values")
    target_indices = np.flatnonzero(pair_uid == TARGET_PAIR_UID)
    require(len(target_indices) == 1, "target pair UID is not unique")
    target_index = int(target_indices[0])
    require(library[target_index] == "LibB", "target library changed")
    require(peptide[target_index] == "AH", "target peptide code changed")
    require(affibody[target_index] == "LIFTK", "target Affibody code changed")
    require(
        sequence_pair[target_index] == TARGET_SEQUENCE_PAIR_SHA256,
        "target sequence-pair identity changed",
    )

    uid_to_index = {value: index for index, value in enumerate(pair_uid)}
    require(len(uid_to_index) == len(pair_uid), "duplicate feature-cache pair UID")
    target_feature = features[target_index : target_index + 1]
    results = []
    source_records = {"feature_cache": cache_source}

    for spec in MODEL_SPECS:
        head_path = REPO_ROOT / spec["head"]
        prediction_path = REPO_ROOT / spec["predictions"]
        source_records[spec["model_id"] + "_head"] = source_record(
            head_path, spec["head_sha256"]
        )
        source_records[spec["model_id"] + "_archived_predictions"] = source_record(
            prediction_path, spec["predictions_sha256"]
        )
        archived = load_archived_scores(spec)
        missing = sorted(set(archived["pair_uid"]).difference(uid_to_index))
        require(not missing, "feature cache misses archived pair IDs")
        old_features = features[
            np.asarray([uid_to_index[value] for value in archived["pair_uid"]], dtype=np.int64)
        ]
        weight, bias, mean, scale, array_names = load_head(spec)

        if spec["kind"] == "sklearn_saved_float32_head":
            # The original sklearn coefficient was float64, but the archived
            # fitted-head artifact stores it as float32.  Promote that saved
            # coefficient back to float64 after reproducing float32 feature
            # standardization.  The resulting tiny parity error quantifies the
            # irreversible serialization rounding.
            old_x = ((old_features - mean) / scale).astype(np.float32).astype(np.float64)
            target_x = ((target_feature - mean) / scale).astype(np.float32).astype(np.float64)
            old_probability = sigmoid(
                old_x.dot(weight.astype(np.float64).reshape(-1)) + float(bias[0])
            )
            target_probability = float(
                sigmoid(
                    target_x.dot(weight.astype(np.float64).reshape(-1)) + float(bias[0])
                )[0]
            )
            scoring_backend = "numpy float32 standardization; saved head promoted to float64"
        else:
            layer = torch.nn.Linear(2560, 1, bias=True).to(device)
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight).to(device))
                layer.bias.copy_(torch.from_numpy(bias).to(device))
                mean_tensor = torch.from_numpy(mean).to(device)
                scale_tensor = torch.from_numpy(scale).to(device)
                old_tensor = torch.from_numpy(old_features).to(device)
                target_tensor = torch.from_numpy(target_feature).to(device)
                old_probability = (
                    torch.sigmoid(layer((old_tensor - mean_tensor) / scale_tensor))
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                target_probability = float(
                    torch.sigmoid(layer((target_tensor - mean_tensor) / scale_tensor))
                    .reshape(-1)[0]
                    .cpu()
                    .item()
                )
            scoring_backend = "torch float32 CUDA"

        parity = parity_record(old_probability, archived, spec["parity_tolerance"])
        require(parity["passed"], "archived parity failed for {}".format(spec["model_id"]))
        results.append(
            {
                "model_id": spec["model_id"],
                "arm": spec["arm"],
                "training_seed": -1 if spec["pi"] is None else 17,
                "pi": spec["pi"],
                "eta": spec["eta"],
                "C": spec["C"],
                "epoch": spec["epoch"],
                "score": target_probability,
                "scoring_backend": scoring_backend,
                "saved_array_names": list(array_names),
                "parity_against_original_119_scores": parity,
            }
        )

    torch.cuda.synchronize(device)
    output_dir = REPO_ROOT / args.output_dir
    require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "outcome-blind saved-head scoring of revised LibB AH x LIFTK",
        "training_performed": False,
        "retention_values_loaded": False,
        "retention_labels_loaded": False,
        "model_or_hyperparameter_selection_changed": False,
        "target": {
            "library": "LibB",
            "pair_uid": TARGET_PAIR_UID,
            "peptide_design_code": "AH",
            "affibody_design_code": "LIFTK",
            "sequence_pair_sha256": TARGET_SEQUENCE_PAIR_SHA256,
            "feature_cache_row_index": target_index,
            "feature_sha256_float32_bytes": sha256_array(target_feature),
        },
        "scores": results,
        "sources": source_records,
        "archived_prediction_columns_loaded": [
            "pair_uid",
            "arm",
            "pi",
            "eta",
            "C",
            "epoch",
            "score",
        ],
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "command": " ".join(map(str, sys.argv)),
    }
    scores_path = output_dir / "scores.json"
    atomic_json(scores_path, result)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "training_performed": False,
        "retention_values_loaded": False,
        "retention_labels_loaded": False,
        "scores": {
            "path": str(scores_path.resolve()),
            "sha256": sha256_file(scores_path),
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("private_data/experiments/pnu_libb_mint_missing_pair_scores_v1"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
