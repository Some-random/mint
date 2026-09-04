#!/usr/bin/env python
"""Complete the weak-label report's LibB predictions for the revised panel.

This is an outcome-blind prediction repair.  It never reads a retention value:

* additive designed-position models are completed from the 119 archived
  probabilities using every available 2-by-2 logit identity;
* frozen-MINT PNU models are scored with their already-saved float32 linear
  heads and the already-cached AH x LIFTK layer-33 feature.

The result is one exact four-column prediction table accepted by
``recompute_libb_metrics_120.py``.  Retention is joined only by that separate
evaluation command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PAIR_UID = "6e8454b1ba8587952e0d"
TARGET_PEPTIDE = "AH"
TARGET_AFFIBODY = "LIFTK"
EXPECTED_OLD_ROWS = 119
EXPECTED_NEW_ROWS = 120
EXPECTED_RECTANGLES = 99
MINT_PARITY_TOLERANCE = 1.0e-6

DEFAULT_METADATA = REPO_ROOT / "private_data/derived/retention_sequences_v2.csv"
DEFAULT_SITE = (
    REPO_ROOT
    / "private_data/experiments/pnu_site_full_u_auc100_folds5_v1/aggregate/"
    "retention_predictions_selected.csv"
)
DEFAULT_WEIGHTED = (
    REPO_ROOT
    / "private_data/experiments/weighted_pn_site_exact_v1/retention_predictions.csv"
)
DEFAULT_MINT = (
    REPO_ROOT
    / "private_data/experiments/pnu_mint_layer33_aggregate_v1/retention_predictions.csv"
)
DEFAULT_MINT_CACHE = (
    REPO_ROOT
    / "private_data/derived/mint_weak_cache_v1/merged/mint_chain_mean_features.npz"
)
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file(), f"missing source: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_outcome_blind_metadata(path: Path) -> pd.DataFrame:
    """Load only IDs/codes; explicitly do not read outcome columns."""

    columns = ["library", "pair_uid", "peptide_design_code", "affibody_design_code"]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[frame["library"].eq("LibB"), columns].copy()
    _require(len(frame) == EXPECTED_NEW_ROWS, "metadata must contain the 120 LibB designs")
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate pair_uid")
    _require(
        int(frame["pair_uid"].eq(TARGET_PAIR_UID).sum()) == 1,
        "metadata lacks the revised pair",
    )
    target = frame.loc[frame["pair_uid"].eq(TARGET_PAIR_UID)].iloc[0]
    _require(
        (target["peptide_design_code"], target["affibody_design_code"])
        == (TARGET_PEPTIDE, TARGET_AFFIBODY),
        "revised pair identity changed",
    )
    return frame


def _stable_float(value: Any) -> str:
    return format(float(value), ".12g")


def additive_completion(
    old: pd.DataFrame,
    metadata: pd.DataFrame,
    score_column: str,
    model: str,
    seed: str,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Complete one missing additive logit from 99 rectangular identities."""

    old = old[["pair_uid", score_column]].copy()
    old[score_column] = pd.to_numeric(old[score_column], errors="raise").astype(float)
    _require(len(old) == EXPECTED_OLD_ROWS, f"{model}: expected 119 archived scores")
    _require(not bool(old["pair_uid"].duplicated().any()), f"{model}: duplicate pair ID")
    _require(not bool(old["pair_uid"].eq(TARGET_PAIR_UID).any()), f"{model}: target already present")
    merged = old.merge(
        metadata[["pair_uid", "peptide_design_code", "affibody_design_code"]],
        on="pair_uid",
        how="left",
        validate="one_to_one",
    )
    _require(
        not bool(merged[["peptide_design_code", "affibody_design_code"]].isna().any().any()),
        f"{model}: metadata join failed",
    )
    probability = merged[score_column].to_numpy(dtype=float)
    _require(
        bool(((probability > 0.0) & (probability < 1.0)).all()),
        f"{model}: additive completion needs probabilities strictly inside (0,1)",
    )
    pivot = merged.pivot(
        index="peptide_design_code",
        columns="affibody_design_code",
        values=score_column,
    )
    _require(pivot.shape == (12, 10), f"{model}: expected a 12-by-10 matrix")
    _require(math.isnan(float(pivot.loc[TARGET_PEPTIDE, TARGET_AFFIBODY])), f"{model}: target is not missing")
    candidates: list[float] = []
    for peptide in sorted(set(pivot.index).difference({TARGET_PEPTIDE})):
        for affibody in sorted(set(pivot.columns).difference({TARGET_AFFIBODY})):
            candidate = (
                logit(float(pivot.loc[TARGET_PEPTIDE, affibody]))
                + logit(float(pivot.loc[peptide, TARGET_AFFIBODY]))
                - logit(float(pivot.loc[peptide, affibody]))
            )
            candidates.append(float(candidate))
    values = np.asarray(candidates, dtype=float)
    _require(len(values) == EXPECTED_RECTANGLES, f"{model}: rectangle count changed")
    _require(bool(np.isfinite(values).all()), f"{model}: non-finite rectangle estimate")
    # The archived torch probabilities are float32.  The median is a robust
    # recovery of the common additive logit under that final-digit rounding.
    target_logit = float(np.median(values))
    target_score = float(expit(target_logit))
    completed = pd.concat(
        [
            old.rename(columns={score_column: "score"}),
            pd.DataFrame([{"pair_uid": TARGET_PAIR_UID, "score": target_score}]),
        ],
        ignore_index=True,
    )
    completed.insert(1, "model", model)
    completed.insert(2, "seed", str(seed))
    completed = completed.rename(columns={"pair_uid": "eval_row_id"})[
        ["eval_row_id", "model", "seed", "score"]
    ]
    audit = {
        "model": model,
        "seed": str(seed),
        "family": "additive_designed_position",
        "source": source_label,
        "completion_method": "median_of_99_additive_2x2_logit_identities",
        "old_rows_copied_without_change": EXPECTED_OLD_ROWS,
        "candidate_rectangles": int(len(values)),
        "target_logit": target_logit,
        "target_score": target_score,
        "candidate_logit_min": float(np.min(values)),
        "candidate_logit_max": float(np.max(values)),
        "candidate_logit_range": float(np.ptp(values)),
        "candidate_logit_max_abs_from_median": float(
            np.max(np.abs(values - target_logit))
        ),
        "parity_rows": EXPECTED_OLD_ROWS,
        "parity_max_abs_difference": 0.0,
        "parity_tolerance": 0.0,
        "parity_passed": True,
    }
    return completed, audit


def _site_models(path: Path, metadata: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    columns = [
        "model_role",
        "risk",
        "class_prior",
        "eta",
        "config_id",
        "pair_uid",
        "library",
        "predicted_probability",
        "run_name",
    ]
    source = pd.read_csv(
        path, usecols=columns, dtype=str, keep_default_na=False, na_filter=False
    )
    source = source.loc[source["library"].eq("LibB")].copy()
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    groups = source.groupby(
        ["model_role", "risk", "class_prior", "eta", "config_id", "run_name"],
        dropna=False,
        sort=True,
    )
    for keys, group in groups:
        role, risk, pi, eta, config_id, run_name = map(str, keys)
        if role == "primary_pn_control":
            model = "site_primary_pn_control"
            report_scope = "main_table"
            seed = "fixed"
        elif risk == "nnpnu":
            model = f"site_selected_nnpnu_pi{_stable_float(pi)}_eta{_stable_float(eta)}"
            report_scope = "main_table"
            seed = "20260902"
        elif risk == "nnpu":
            model = f"site_pure_nnpu_pi{_stable_float(pi)}"
            report_scope = "prose_support"
            seed = "20260902"
        else:
            raise ValueError(f"unexpected site group: {keys}")
        completed, audit = additive_completion(
            group,
            metadata,
            "predicted_probability",
            model,
            seed,
            str(path.resolve()),
        )
        outputs.append(completed)
        audits.append(audit)
        models.append(
            {
                "model": model,
                "seed": seed,
                "report_scope": report_scope,
                "source_file": str(path.resolve()),
                "source_filter": json.dumps(
                    {
                        "model_role": role,
                        "risk": risk,
                        "class_prior": pi,
                        "eta": eta,
                        "config_id": config_id,
                        "run_name": run_name,
                    },
                    sort_keys=True,
                ),
            }
        )
    _require(len(outputs) == 9, "expected primary + four selected PNU + four pure-PU site models")
    return outputs, audits, models


def _weighted_models(path: Path, metadata: pd.DataFrame) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    columns = [
        "arm",
        "positive_mass",
        "C",
        "pair_uid",
        "library",
        "predicted_probability",
    ]
    source = pd.read_csv(
        path, usecols=columns, dtype=str, keep_default_na=False, na_filter=False
    )
    source = source.loc[source["library"].eq("LibB")].copy()
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for keys, group in source.groupby(["arm", "positive_mass", "C"], sort=True):
        arm, positive_mass, c_value = map(str, keys)
        if arm == "balanced_pn":
            model = "site_converged_balanced_pn"
            report_scope = "control_table"
        else:
            model = f"site_converged_prior_weighted_pn_pi{_stable_float(positive_mass)}"
            report_scope = (
                "control_table"
                if math.isclose(float(positive_mass), 0.15)
                else "prose_support"
                if math.isclose(float(positive_mass), 0.05)
                else "supporting_artifact"
            )
        completed, audit = additive_completion(
            group,
            metadata,
            "predicted_probability",
            model,
            "liblinear_fixed",
            str(path.resolve()),
        )
        outputs.append(completed)
        audits.append(audit)
        models.append(
            {
                "model": model,
                "seed": "liblinear_fixed",
                "report_scope": report_scope,
                "source_file": str(path.resolve()),
                "source_filter": json.dumps(
                    {"arm": arm, "positive_mass": positive_mass, "C": c_value},
                    sort_keys=True,
                ),
            }
        )
    _require(len(outputs) == 5, "expected five converged weighted-PN site models")
    return outputs, audits, models


def _sigmoid_linear(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32).reshape(-1)
    b = np.asarray(bias, dtype=np.float32).reshape(-1)
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    s = np.asarray(scale, dtype=np.float32).reshape(-1)
    _require(w.shape == m.shape == s.shape == (2560,), "MINT head width changed")
    _require(b.shape == (1,), "MINT head bias shape changed")
    logits = ((x - m) / s) @ w + float(b[0])
    return np.asarray(expit(logits), dtype=float).reshape(-1)


def _load_head(path: Path, prefix: str) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as archive:
        names = tuple(
            f"{prefix}_{suffix}" for suffix in ("weight", "bias", "mean", "scale")
        )
        missing = set(names).difference(archive.files)
        _require(not missing, f"{path}: missing arrays {sorted(missing)}")
        return tuple(np.asarray(archive[name]).copy() for name in names)


def _mint_models(
    aggregate_path: Path,
    cache_path: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    columns = [
        "model",
        "library",
        "pair_uid",
        "arm",
        "training_seed",
        "pi",
        "eta",
        "C",
        "epoch",
        "score",
    ]
    archived = pd.read_csv(
        aggregate_path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    archived = archived.loc[archived["library"].eq("LibB")].copy()

    with np.load(cache_path, allow_pickle=False) as cache:
        pair_uids = np.asarray(cache["pair_uid"]).astype(str)
        _require(len(set(pair_uids.tolist())) == len(pair_uids), "MINT cache IDs duplicate")
        lookup = {value: index for index, value in enumerate(pair_uids)}
        _require(TARGET_PAIR_UID in lookup, "MINT cache lacks revised pair feature")
        # Load once: repeated access to a compressed NPZ member would repeatedly
        # decompress this 647-MiB archive.
        features = np.asarray(cache["mint_chain_mean"], dtype=np.float32)
    target_feature = features[lookup[TARGET_PAIR_UID] : lookup[TARGET_PAIR_UID] + 1]
    _require(target_feature.shape == (1, 2560), "target MINT feature shape changed")
    _require(bool(np.isfinite(target_feature).all()), "target MINT feature is non-finite")

    cases = [
        {
            "model": "Frozen MINT PN AUROC-selected rebaseline",
            "output_model": "mint_auroc_selected_pn_rebaseline",
            "filters": {},
            "head": REPO_ROOT / "private_data/experiments/pnu_mint_layer33_controls_v1/LibB/fitted_heads.npz",
            "prefix": "LibB_sklearn",
            "report_scope": "mint_table",
            "seed": "fixed",
        },
        {
            "model": "Frozen MINT PNU sensitivity",
            "output_model": "mint_selected_pnu_pi0.02_eta0",
            "filters": {"pi": "0.02", "eta": "0.0", "epoch": "78.0"},
            "head": REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid80_extension_v1/LibB_pi0p02_eta0/fitted_heads.npz",
            "prefix": "LibB_seed17_pi0p02_eta0p0",
            "report_scope": "mint_table",
            "seed": "17",
        },
        {
            "model": "Frozen MINT PNU sensitivity",
            "output_model": "mint_selected_pnu_pi0.05_eta0",
            "filters": {"pi": "0.05", "eta": "0.0", "epoch": "39.0"},
            "head": REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p05_eta0/fitted_heads.npz",
            "prefix": "LibB_seed17_pi0p05_eta0p0",
            "report_scope": "mint_table",
            "seed": "17",
        },
        {
            "model": "Frozen MINT PNU sensitivity",
            "output_model": "mint_selected_pnu_pi0.1_eta0.25",
            "filters": {"pi": "0.1", "eta": "0.25", "epoch": "39.0"},
            "head": REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p1_eta0p25/fitted_heads.npz",
            "prefix": "LibB_seed17_pi0p1_eta0p25",
            "report_scope": "mint_table",
            "seed": "17",
        },
        {
            "model": "Frozen MINT PNU sensitivity",
            "output_model": "mint_selected_pnu_pi0.15_eta0.25",
            "filters": {"pi": "0.15", "eta": "0.25", "epoch": "39.0"},
            "head": REPO_ROOT / "private_data/experiments/pnu_mint_layer33_grid40_v1/LibB_pi0p15_eta0p25/fitted_heads.npz",
            "prefix": "LibB_seed17_pi0p15_eta0p25",
            "report_scope": "mint_table",
            "seed": "17",
        },
    ]
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    head_paths: list[Path] = []
    for case in cases:
        old = archived.loc[archived["model"].eq(case["model"])].copy()
        for column, value in case["filters"].items():
            old = old.loc[old[column].eq(value)].copy()
        _require(len(old) == EXPECTED_OLD_ROWS, f"{case['output_model']}: archived group is not 119 rows")
        _require(not bool(old["pair_uid"].duplicated().any()), "MINT archived IDs duplicate")
        old_score = pd.to_numeric(old["score"], errors="raise").to_numpy(dtype=float)
        head_path = Path(case["head"])
        head_paths.append(head_path)
        weight, bias, mean, scale = _load_head(head_path, str(case["prefix"]))
        _require(set(old["pair_uid"]).issubset(lookup), "MINT cache misses archived IDs")
        old_feature = features[[lookup[value] for value in old["pair_uid"]]]
        reproduced = _sigmoid_linear(old_feature, weight, bias, mean, scale)
        difference = np.abs(reproduced - old_score)
        maximum = float(np.max(difference))
        _require(
            maximum <= MINT_PARITY_TOLERANCE,
            f"{case['output_model']}: 119-row saved-head parity failed ({maximum})",
        )
        parity = {
            "rows": EXPECTED_OLD_ROWS,
            "max_abs_difference": maximum,
            "mean_abs_difference": float(np.mean(difference)),
            "root_mean_square_difference": float(
                np.sqrt(np.mean(difference * difference))
            ),
            "tolerance": MINT_PARITY_TOLERANCE,
            "passed": True,
        }
        target_score = float(
            _sigmoid_linear(target_feature, weight, bias, mean, scale)[0]
        )
        completion_method = "saved_float32_linear_head_on_cached_layer33_feature"
        source_file = head_path
        completed = pd.concat(
            [
                pd.DataFrame(
                    {
                        "eval_row_id": old["pair_uid"].astype(str),
                        "score": old_score,
                    }
                ),
                pd.DataFrame(
                    [{"eval_row_id": TARGET_PAIR_UID, "score": target_score}]
                ),
            ],
            ignore_index=True,
        )
        completed.insert(1, "model", str(case["output_model"]))
        completed.insert(2, "seed", str(case["seed"]))
        outputs.append(completed[["eval_row_id", "model", "seed", "score"]])
        audits.append(
            {
                "model": str(case["output_model"]),
                "seed": str(case["seed"]),
                "family": "frozen_mint_layer33_linear_readout",
                "source": str(Path(source_file).resolve()),
                "completion_method": completion_method,
                "old_rows_copied_without_change": EXPECTED_OLD_ROWS,
                "candidate_rectangles": 0,
                "target_logit": float(logit(target_score)),
                "target_score": target_score,
                "candidate_logit_min": "",
                "candidate_logit_max": "",
                "candidate_logit_range": "",
                "candidate_logit_max_abs_from_median": "",
                "parity_rows": int(parity["rows"]),
                "parity_max_abs_difference": float(parity["max_abs_difference"]),
                "parity_tolerance": float(parity["tolerance"]),
                "parity_passed": bool(parity["passed"]),
            }
        )
        models.append(
            {
                "model": str(case["output_model"]),
                "seed": str(case["seed"]),
                "report_scope": str(case["report_scope"]),
                "source_file": str(aggregate_path.resolve()),
                "source_filter": json.dumps(
                    {"model": case["model"], **case["filters"]}, sort_keys=True
                ),
            }
        )
    return outputs, audits, models, head_paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    _require(str(output_dir).startswith(str((REPO_ROOT / "private_data").resolve()) + os.sep), "output must be under private_data")
    _require(not output_dir.exists(), f"output directory exists: {output_dir}")
    for path in (
        args.metadata,
        args.site_predictions,
        args.weighted_predictions,
        args.mint_predictions,
        args.mint_cache,
    ):
        _require(Path(path).is_file(), f"missing input: {path}")

    metadata = load_outcome_blind_metadata(args.metadata)
    blocks: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []

    for loader, path in (
        (_site_models, args.site_predictions),
        (_weighted_models, args.weighted_predictions),
    ):
        model_blocks, model_audits, model_rows = loader(path, metadata)
        blocks.extend(model_blocks)
        audits.extend(model_audits)
        models.extend(model_rows)
    mint_blocks, mint_audits, mint_rows, head_paths = _mint_models(
        args.mint_predictions, args.mint_cache
    )
    blocks.extend(mint_blocks)
    audits.extend(mint_audits)
    models.extend(mint_rows)

    predictions = pd.concat(blocks, ignore_index=True)
    predictions = predictions[["eval_row_id", "model", "seed", "score"]]
    predictions["score"] = pd.to_numeric(predictions["score"], errors="raise")
    _require(bool(np.isfinite(predictions["score"]).all()), "non-finite output score")
    _require(len(blocks) == 19, "expected 19 complete LibB report/supporting models")
    _require(len(predictions) == 19 * EXPECTED_NEW_ROWS, "prediction row count changed")
    expected_ids = set(metadata["pair_uid"])
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        _require(len(group) == EXPECTED_NEW_ROWS, f"{model}/{seed}: not 120 rows")
        _require(set(group["eval_row_id"]) == expected_ids, f"{model}/{seed}: panel mismatch")

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    _atomic_csv(predictions, output_dir / "normalized_predictions.csv")
    _atomic_csv(pd.DataFrame(audits), output_dir / "reconstruction_audit.csv")
    _atomic_csv(pd.DataFrame(models), output_dir / "model_source_map.csv")

    sources: dict[str, Any] = {
        "metadata_identity_columns_only": _source(args.metadata),
        "site_predictions_score_and_identity_columns_only": _source(
            args.site_predictions
        ),
        "weighted_predictions_score_and_identity_columns_only": _source(
            args.weighted_predictions
        ),
        "mint_predictions_score_and_identity_columns_only": _source(
            args.mint_predictions
        ),
        "mint_feature_cache_pair_ids_and_features_only": _source(args.mint_cache),
        "saved_mint_heads": [_source(path) for path in sorted(set(head_paths))],
    }
    manifest = {
        "schema_version": "pnu-libb-corrected-panel-predictions-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "outcome_blind_prediction_completion_only",
        "retention_outcomes_read": False,
        "training_performed": False,
        "model_selection_performed": False,
        "target": {
            "pair_uid": TARGET_PAIR_UID,
            "peptide_design_code": TARGET_PEPTIDE,
            "affibody_design_code": TARGET_AFFIBODY,
        },
        "models": int(len(blocks)),
        "prediction_rows": int(len(predictions)),
        "additive_models": int(sum(row["family"].startswith("additive") for row in audits)),
        "frozen_mint_models": int(sum(row["family"].startswith("frozen_mint") for row in audits)),
        "all_119_row_parity_checks_passed": bool(
            all(bool(row["parity_passed"]) for row in audits)
        ),
        "sources": sources,
        "outputs": {
            name: {
                "sha256": _sha256(output_dir / name),
                "bytes": int((output_dir / name).stat().st_size),
                "rows": int(
                    len(
                        pd.read_csv(
                            output_dir / name,
                            dtype=str,
                            keep_default_na=False,
                            na_filter=False,
                        )
                    )
                ),
            }
            for name in (
                "normalized_predictions.csv",
                "reconstruction_audit.csv",
                "model_source_map.csv",
            )
        },
        "command": " ".join(map(str, __import__("sys").argv)),
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps({"output_dir": str(output_dir), "models": len(blocks), "rows": len(predictions)}, sort_keys=True))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--site-predictions", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--weighted-predictions", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--mint-predictions", type=Path, default=DEFAULT_MINT)
    parser.add_argument("--mint-cache", type=Path, default=DEFAULT_MINT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
