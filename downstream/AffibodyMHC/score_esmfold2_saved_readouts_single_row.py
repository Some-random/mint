#!/usr/bin/env python
"""Score one label-free ESMFold2 row with already-trained LibB readouts.

This utility never loads training or retention labels.  It also recomputes the
saved predictions for the original evaluation roster and requires numerical
parity before accepting the new single-row scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.esmfold2_libb_readout import (
    AFFIBODY_LENGTH,
    PEPTIDE_LENGTH,
    MODEL_NAMES,
    build_readout,
    model_feature_names,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-row-npz", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--existing-cache", type=Path, required=True)
    parser.add_argument("--existing-predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-atol", type=float, default=2e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    private_root = (Path(__file__).resolve().parents[2] / "private_data").resolve()
    output_dir.relative_to(private_root)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if tuple(config["models"]) != MODEL_NAMES:
        raise ValueError("saved model contract changed")
    architecture = config["architecture"]

    with np.load(args.single_row_npz, allow_pickle=False) as handle:
        required = {
            "row_index",
            "row_id",
            "split",
            "distogram_probabilities",
            "pair_states_symmetric",
            "single_inputs",
        }
        if set(handle.files) != required:
            raise ValueError(f"single-row arrays changed: {handle.files}")
        row_ids = np.asarray(handle["row_id"]).astype(str)
        if row_ids.shape != (1,) or np.asarray(handle["split"]).astype(str).tolist() != ["eval"]:
            raise ValueError("expected exactly one label-free evaluation row")
        target_arrays = {
            "distogram_probabilities": np.asarray(handle["distogram_probabilities"]),
            "pair_states_symmetric": np.asarray(handle["pair_states_symmetric"]),
            "single_inputs": np.asarray(handle["single_inputs"]),
        }
    for name, array in target_arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite target array: {name}")

    metadata = pd.read_csv(
        args.existing_cache / "metadata.csv",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    eval_frame = metadata.loc[metadata["split"].eq("eval")].copy()
    if len(eval_frame) != 119:
        raise ValueError(f"expected old 119-row roster, found {len(eval_frame)}")
    old_indices = eval_frame["row_index"].astype(int).to_numpy()
    old_row_ids = eval_frame["row_id"].astype(str).tolist()
    old_arrays = {
        name: np.asarray(
            np.load(args.existing_cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)[old_indices]
        )
        for name in ("distogram_probabilities", "pair_states_symmetric", "single_inputs")
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    def tensor_inputs(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        singles = arrays["single_inputs"]
        return {
            "distogram_probabilities": torch.as_tensor(
                arrays["distogram_probabilities"], dtype=torch.float32, device=device
            ),
            "pair_states_symmetric": torch.as_tensor(
                arrays["pair_states_symmetric"], dtype=torch.float32, device=device
            ),
            "single_inputs_peptide": torch.as_tensor(
                singles[:, :PEPTIDE_LENGTH], dtype=torch.float32, device=device
            ),
            "single_inputs_affibody": torch.as_tensor(
                singles[:, PEPTIDE_LENGTH : PEPTIDE_LENGTH + AFFIBODY_LENGTH],
                dtype=torch.float32,
                device=device,
            ),
        }

    old_inputs = tensor_inputs(old_arrays)
    target_inputs = tensor_inputs(target_arrays)
    checkpoint_paths = sorted(args.checkpoints_dir.glob("*.pt"))
    if len(checkpoint_paths) != 25:
        raise ValueError(f"expected 25 checkpoints, found {len(checkpoint_paths)}")

    rows: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    with torch.inference_mode():
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model_name = str(checkpoint["model"])
            seed = int(checkpoint["seed"])
            expected_name = f"{model_name}__seed{seed}.pt"
            if checkpoint_path.name != expected_name:
                raise ValueError(f"checkpoint filename mismatch: {checkpoint_path.name}")
            model = build_readout(model_name, architecture).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            features = model_feature_names(model_name)
            old_scores = torch.sigmoid(
                model(**{name: old_inputs[name] for name in features})
            ).cpu().numpy()
            target_score = float(
                torch.sigmoid(model(**{name: target_inputs[name] for name in features}))
                .cpu()
                .item()
            )
            if not math.isfinite(target_score):
                raise ValueError(f"non-finite score from {checkpoint_path.name}")

            saved_path = args.existing_predictions_dir / f"{model_name}__seed{seed}.csv"
            saved = pd.read_csv(saved_path, dtype={"eval_row_id": str})
            if set(saved["eval_row_id"]) != set(old_row_ids):
                raise ValueError(f"old roster mismatch: {saved_path}")
            expected = saved.set_index("eval_row_id").loc[old_row_ids, "score"].to_numpy(float)
            absolute = np.abs(old_scores.astype(float) - expected)
            maximum = float(absolute.max())
            if maximum > args.parity_atol:
                raise ValueError(
                    f"old119 parity failed for {checkpoint_path.name}: {maximum}"
                )
            parity.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "old_rows": len(old_row_ids),
                    "max_abs_difference": maximum,
                    "saved_predictions_sha256": sha256_file(saved_path),
                }
            )
            rows.append(
                {
                    "eval_row_id": row_ids[0],
                    "model": model_name,
                    "seed": seed,
                    "score": target_score,
                }
            )
            del model

    scores_path = output_dir / "blinded_predictions_single_row.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("eval_row_id", "model", "seed", "score")
        )
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(scores_path, 0o600)
    manifest = {
        "schema_version": "esmfold2-libb-saved-readout-single-row-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "row_id": row_ids[0],
        "checkpoint_count": len(checkpoint_paths),
        "score_count": len(rows),
        "all_scores_finite": all(math.isfinite(float(row["score"])) for row in rows),
        "parity_atol": args.parity_atol,
        "old119_parity_max_abs_difference": max(
            float(row["max_abs_difference"]) for row in parity
        ),
        "supervision_fields_read": [],
        "inputs": {
            "single_row_npz": str(args.single_row_npz.resolve()),
            "single_row_npz_sha256": sha256_file(args.single_row_npz),
            "config_sha256": sha256_file(args.config),
            "checkpoint_hashes": {
                path.name: sha256_file(path) for path in checkpoint_paths
            },
        },
        "output": {
            "file": scores_path.name,
            "sha256": sha256_file(scores_path),
        },
        "parity": parity,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "row_id", "checkpoint_count", "all_scores_finite",
        "old119_parity_max_abs_difference", "runtime_seconds"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
