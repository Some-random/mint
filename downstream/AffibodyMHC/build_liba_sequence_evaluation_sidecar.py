#!/usr/bin/env python3
"""Build target-free LibA deployment scores from frozen sequence-model outputs.

The input and output schemas contain only opaque evaluation IDs, model names,
seed labels, and scores.  Direct-retention values are neither accepted nor
read.  The nonlinear deployment score is the sigmoid of the mean of all five
seed logits.  The two-model score follows the separately locked equal-logit
formula and is emitted only after that lock is hash-recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.code_only_baseline import sha256_file
from downstream.AffibodyMHC.evaluate_libb_readouts_sealed import (
    BLINDED_PREDICTION_COLUMNS,
    _membership_sha256,
)
from downstream.AffibodyMHC.finalize_liba_model_comparison import _mean_logit


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "liba-sequence-evaluation-sidecar-v1"
EXPECTED_ROWS = 108
EXPECTED_NONLINEAR_SEEDS = {
    "20260811",
    "20260812",
    "20260813",
    "20260814",
    "20260815",
}
EXPECTED_SOURCE_MODELS = {
    "additive_6site",
    "frozen_mint_layer9",
    "frozen_mint_layer33_control",
    "nonlinear_6site",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_lock(path: Path, expected_candidate: str) -> dict:
    _require(path.is_file(), f"weak score lock does not exist: {path}")
    payload = json.loads(path.read_text())
    _require(payload.get("schema_version") == "affibody-weak-oof-score-lock-v1", "lock schema changed")
    _require(payload.get("library") == "LibA", "lock is not LibA")
    _require(payload.get("retention_labels_read") is False, "lock read retention labels")
    _require(payload.get("candidate") == expected_candidate, "locked candidate changed")
    return payload


def _aligned_scores(frame: pd.DataFrame, model: str, seed: str) -> pd.DataFrame:
    selected = frame.loc[
        frame["model"].astype(str).eq(model) & frame["seed"].astype(str).eq(seed),
        ["eval_row_id", "score"],
    ].copy()
    _require(len(selected) == EXPECTED_ROWS and selected["eval_row_id"].is_unique, f"{model}/{seed} panel changed")
    return selected.sort_values("eval_row_id", kind="mergesort").reset_index(drop=True)


def build_sidecar(
    input_path: Path,
    selected_lock_path: Path,
    equal_lock_path: Path,
) -> tuple[pd.DataFrame, dict]:
    header = tuple(pd.read_csv(input_path, nrows=0).columns)
    _require(header == BLINDED_PREDICTION_COLUMNS, "common evaluation prediction schema changed")
    frame = pd.read_csv(
        input_path,
        dtype={"eval_row_id": str, "model": str, "seed": str},
        keep_default_na=False,
        na_filter=False,
    )
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
    _require(bool(np.isfinite(frame["score"]).all()), "common sequence score is non-finite")
    _require(set(frame["model"].astype(str)) == EXPECTED_SOURCE_MODELS, "source model set changed")
    _require(not bool(frame.duplicated(["model", "seed", "eval_row_id"]).any()), "duplicate source score")
    memberships = []
    for _, group in frame.groupby(["model", "seed"], sort=True):
        _require(len(group) == EXPECTED_ROWS, "a source group is not the exact 108-row panel")
        memberships.append(_membership_sha256(group["eval_row_id"]))
    _require(len(set(memberships)) == 1, "source groups have different opaque-ID panels")
    nonlinear = frame.loc[frame["model"].eq("nonlinear_6site")]
    _require(set(nonlinear["seed"].astype(str)) == EXPECTED_NONLINEAR_SEEDS, "nonlinear seed set changed")

    selected_lock = _load_lock(selected_lock_path, "mean_logit__frozen_mint_layer9")
    equal_lock = _load_lock(equal_lock_path, "mean_logit__additive_6site__frozen_mint_layer9")
    _require(selected_lock.get("members") == ["frozen_mint_layer9"], "selected MINT lock members changed")
    formula = equal_lock.get("formula", {})
    components = formula.get("components", [])
    _require(formula.get("equal_logit_weights") is True, "ensemble is no longer equal-logit")
    _require(
        [(item.get("name"), float(item.get("weight", -1))) for item in components]
        == [("additive_6site", 0.5), ("frozen_mint_layer9", 0.5)],
        "equal-logit component contract changed",
    )

    nonlinear_blocks = []
    for seed in sorted(EXPECTED_NONLINEAR_SEEDS):
        block = _aligned_scores(frame, "nonlinear_6site", seed).rename(columns={"score": seed})
        nonlinear_blocks.append(block)
    nonlinear_aligned = nonlinear_blocks[0]
    for block in nonlinear_blocks[1:]:
        nonlinear_aligned = nonlinear_aligned.merge(block, on="eval_row_id", validate="one_to_one")
    nonlinear_score = _mean_logit(
        nonlinear_aligned[sorted(EXPECTED_NONLINEAR_SEEDS)].to_numpy(dtype=float)
    )

    additive = _aligned_scores(frame, "additive_6site", "fixed")
    mint9 = _aligned_scores(frame, "frozen_mint_layer9", "fixed")
    pair = additive.merge(mint9, on="eval_row_id", validate="one_to_one", suffixes=("_additive", "_mint9"))
    equal_score = _mean_logit(pair[["score_additive", "score_mint9"]].to_numpy(dtype=float))

    derived = pd.concat(
        [
            pd.DataFrame(
                {
                    "eval_row_id": nonlinear_aligned["eval_row_id"],
                    "model": "nonlinear_6site_mean_logit",
                    "seed": "deployment",
                    "score": nonlinear_score,
                }
            ),
            pd.DataFrame(
                {
                    "eval_row_id": mint9["eval_row_id"],
                    "model": "mint_l9",
                    "seed": "deployment",
                    "score": mint9["score"],
                }
            ),
            pd.DataFrame(
                {
                    "eval_row_id": pair["eval_row_id"],
                    "model": "equal_logit_additive_mint_l9",
                    "seed": "deployment",
                    "score": equal_score,
                }
            ),
        ],
        ignore_index=True,
    )
    output = pd.concat([frame, derived], ignore_index=True)
    output = output[list(BLINDED_PREDICTION_COLUMNS)]
    _require(not bool(output.duplicated(["model", "seed", "eval_row_id"]).any()), "output duplicate score")
    for _, group in output.groupby(["model", "seed"], sort=True):
        _require(len(group) == EXPECTED_ROWS, "output group is not exact 108")
        _require(_membership_sha256(group["eval_row_id"]) == memberships[0], "output membership changed")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "retention_labels_read": False,
        "input": {"path": str(input_path.resolve()), "sha256": sha256_file(input_path)},
        "weak_score_locks": {
            "mint_l9": {"path": str(selected_lock_path.resolve()), "sha256": sha256_file(selected_lock_path)},
            "equal_logit_additive_mint_l9": {
                "path": str(equal_lock_path.resolve()),
                "sha256": sha256_file(equal_lock_path),
            },
        },
        "rows_per_model_seed": EXPECTED_ROWS,
        "membership_sha256": memberships[0],
        "nonlinear_deployment": {
            "aggregation": "sigmoid(mean(seed logits))",
            "seeds": sorted(EXPECTED_NONLINEAR_SEEDS),
        },
        "equal_ensemble_deployment": equal_lock["formula"],
    }
    return output, receipt


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selected-lock", type=Path, required=True)
    parser.add_argument("--equal-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    _require(not args.output.exists() and not args.receipt.exists(), "output exists; refusing overwrite")
    output, receipt = build_sidecar(
        args.input.resolve(), args.selected_lock.resolve(), args.equal_lock.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.to_csv(args.output, index=False, float_format="%.12g")
    os.chmod(args.output, 0o600)
    receipt["output"] = {
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "rows": int(len(output)),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.chmod(args.receipt, 0o600)
    print(json.dumps(receipt["output"], sort_keys=True))


if __name__ == "__main__":
    main()
