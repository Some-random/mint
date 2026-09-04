#!/usr/bin/env python3
"""Export the LibA layer-33 head from the locked combined deployment archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SOURCE_SCHEMA = "liba-additive-mint-deployment-heads-v1"
OUTPUT_SCHEMA = "liba-mint-layer33-deployment-head-v1"
FEATURE_KEY = "mint_layer_33_chain_mean"
POOLING = (
    "exclude cls/eos/padding; mean residues separately within chain_id 0 and "
    "chain_id 1; concatenate chain 0 then chain 1"
)
TRAINING_MEMBERSHIP_SHA256 = (
    "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    require(source.is_file(), "combined LibA deployment archive is missing")
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    require(bool(relative.parts), "refusing private_data root output")
    require(not output.exists(), "output exists; refusing overwrite")

    required = {
        "schema_version",
        "mint_layer33_mean",
        "mint_layer33_scale",
        "mint_layer33_coef",
        "mint_layer33_intercept",
        "mint_layer33_layer",
        "mint_layer33_feature_key",
        "mint_layer33_pooling",
        "mint_layer33_C",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        require(not missing, f"combined deployment archive lacks {sorted(missing)}")
        require(str(np.asarray(archive["schema_version"]).reshape(-1)[0]) == SOURCE_SCHEMA,
                "combined deployment schema changed")
        mean = np.asarray(archive["mint_layer33_mean"], dtype=np.float64).copy()
        scale = np.asarray(archive["mint_layer33_scale"], dtype=np.float64).copy()
        coefficient = np.asarray(archive["mint_layer33_coef"], dtype=np.float64).copy()
        intercept = np.asarray(archive["mint_layer33_intercept"], dtype=np.float64).reshape(1).copy()
        layer = np.asarray(archive["mint_layer33_layer"], dtype=np.int64).reshape(1).copy()
        feature_key = np.asarray(archive["mint_layer33_feature_key"]).reshape(1).copy()
        pooling = np.asarray(archive["mint_layer33_pooling"]).reshape(1).copy()
        c_value = np.asarray(archive["mint_layer33_C"], dtype=np.float64).reshape(1).copy()

    require(mean.shape == scale.shape == coefficient.shape == (2560,), "bad layer-33 vector shape")
    require(bool(np.isfinite(mean).all()), "non-finite layer-33 mean")
    require(bool(np.isfinite(scale).all() and (scale > 0).all()), "bad layer-33 scale")
    require(bool(np.isfinite(coefficient).all()), "non-finite layer-33 coefficient")
    require(bool(np.isfinite(intercept).all()), "non-finite layer-33 intercept")
    require(int(layer[0]) == 33, "layer-33 identifier changed")
    require(str(feature_key[0]) == FEATURE_KEY, "layer-33 feature key changed")
    require(str(pooling[0]) == POOLING, "layer-33 pooling changed")
    require(float(c_value[0]) == 0.01, "canonical LibA layer-33 C is not 0.01")

    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    np.savez(
        output,
        schema_version=np.asarray([OUTPUT_SCHEMA]),
        mint_mean=mean,
        mint_scale=scale,
        mint_coef=coefficient,
        mint_intercept=intercept,
        mint_layer=layer,
        mint_feature_key=feature_key,
        mint_pooling=pooling,
        mint_C=c_value,
        training_membership_sha256=np.asarray([TRAINING_MEMBERSHIP_SHA256]),
    )
    os.chmod(output, 0o600)
    receipt = {
        "schema_version": OUTPUT_SCHEMA,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "mint_layer": 33,
        "mint_feature_key": FEATURE_KEY,
        "mint_pooling": POOLING,
        "mint_C": 0.01,
        "training_membership_sha256": TRAINING_MEMBERSHIP_SHA256,
        "retention_labels_read": False,
        "outcome_columns_read": [],
    }
    receipt_path = output.with_suffix(".json")
    require(not receipt_path.exists(), "receipt exists; refusing overwrite")
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(receipt_path, 0o600)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base = REPO_ROOT / "private_data/experiments/liba_common_oof_sequence_predictions_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=base / "deployment_heads.npz")
    parser.add_argument("--output", type=Path, default=base / "mint_layer33_deployment_head.npz")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(export(arguments.source, arguments.output), sort_keys=True))
