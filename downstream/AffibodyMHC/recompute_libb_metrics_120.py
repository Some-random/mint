#!/usr/bin/env python
"""Recompute LibB metrics on the corrected complete 12-by-10 panel.

This is an evaluation-only utility.  It accepts one corrected retention table
and one or more already-frozen, target-free prediction tables.  It does not
fit a model, choose a threshold, or otherwise use retention for training.

Prediction files have the exact schema::

    eval_row_id,model,seed,score

Every model/seed group must contain the same 120 opaque IDs as the corrected
retention panel.  Larger scores must mean "more likely to bind".  Ranking ties
are resolved by descending score and then ascending Affibody design code, as
defined in :mod:`downstream.AffibodyMHC.wetlab_metrics`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.wetlab_metrics import (  # noqa: E402
    TIE_POLICY,
    evaluate_wetlab_predictions,
)


SCHEMA_VERSION = "libb-corrected-120-metric-recompute-v1"
PREDICTION_COLUMNS = ("eval_row_id", "model", "seed", "score")
ID_COLUMN_CANDIDATES = ("eval_row_id", "pair_uid", "row_id")
RETENTION_THRESHOLD = 75.0
EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_BINDERS = 61
EXPECTED_NONBINDERS = 59
CORRECTED_PAIR = ("AH", "LIFTK")
CORRECTED_PAIR_RETENTION = 87.94
K_VALUES = (1, 3)

SUMMARY_METRICS = (
    "global_auroc",
    "global_average_precision",
    "global_spearman",
    "within_peptide_spearman_mean",
    "peptide_macro_precision_at_1",
    "peptide_macro_precision_at_3",
    "peptide_macro_hit_at_1",
    "peptide_macro_hit_at_3",
    "peptide_macro_best_retention_at_1",
    "peptide_macro_best_retention_at_3",
    "peptide_macro_regret_at_1",
    "peptide_macro_regret_at_3",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_string_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def _membership_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def file_provenance(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    _require(path.is_file(), f"source file does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_utc": _utc_timestamp(stat.st_mtime),
    }


def _normalize_panel_id(frame: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in ID_COLUMN_CANDIDATES if column in frame]
    _require(
        len(available) == 1,
        "retention panel must contain exactly one ID column from {}".format(
            ", ".join(ID_COLUMN_CANDIDATES)
        ),
    )
    return frame.rename(columns={available[0]: "eval_row_id"})


def load_corrected_retention_panel(path: Path) -> pd.DataFrame:
    """Load and fail-closed validate the corrected complete LibB matrix."""

    path = Path(path)
    _require(path.is_file(), f"retention panel does not exist: {path}")
    frame = _read_string_csv(path)
    if "library" in frame:
        frame = frame.loc[frame["library"].eq("LibB")].copy()
    frame = _normalize_panel_id(frame)
    required = {
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
    }
    missing = required.difference(frame.columns)
    _require(not missing, f"retention panel is missing columns {sorted(missing)}")
    output = frame[
        [
            "eval_row_id",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
        ]
    ].copy()

    for column in ("eval_row_id", "peptide_design_code", "affibody_design_code"):
        value = output[column].astype(str)
        _require(not bool(value.eq("").any()), f"{column} contains empty values")
        _require(
            bool(value.eq(value.str.strip()).all()),
            f"{column} contains surrounding whitespace",
        )
    _require(
        not bool(output["eval_row_id"].duplicated().any()),
        "retention panel contains duplicate evaluation IDs",
    )
    _require(
        not bool(
            output.duplicated(
                ["peptide_design_code", "affibody_design_code"], keep=False
            ).any()
        ),
        "retention panel contains duplicate peptide/Affibody pairs",
    )
    _require(
        not bool(output["target_retention"].astype(str).eq("").any()),
        "all 120 LibB cells must have a direct retention value",
    )
    output["target_retention"] = pd.to_numeric(
        output["target_retention"], errors="raise"
    ).astype(float)
    retention = output["target_retention"].to_numpy(dtype=float)
    _require(bool(np.isfinite(retention).all()), "retention contains non-finite values")
    _require(
        bool(((retention >= 0.0) & (retention <= 100.0)).all()),
        "retention must be in [0, 100]",
    )

    peptides = sorted(set(output["peptide_design_code"].astype(str)))
    affibodies = sorted(set(output["affibody_design_code"].astype(str)))
    _require(len(output) == EXPECTED_ROWS, "corrected LibB panel must contain 120 rows")
    _require(
        len(peptides) == EXPECTED_PEPTIDES,
        "corrected LibB panel must contain 12 peptides",
    )
    _require(
        len(affibodies) == EXPECTED_AFFIBODIES,
        "corrected LibB panel must contain 10 Affibodies",
    )
    observed_pairs = set(
        zip(output["peptide_design_code"], output["affibody_design_code"])
    )
    expected_pairs = {
        (peptide, affibody) for peptide in peptides for affibody in affibodies
    }
    _require(
        observed_pairs == expected_pairs,
        "LibB panel is not the complete 12-by-10 peptide/Affibody Cartesian matrix",
    )

    corrected = output.loc[
        output["peptide_design_code"].eq(CORRECTED_PAIR[0])
        & output["affibody_design_code"].eq(CORRECTED_PAIR[1])
    ]
    _require(len(corrected) == 1, "corrected AH x LIFTK cell is missing")
    _require(
        math.isclose(
            float(corrected.iloc[0]["target_retention"]),
            CORRECTED_PAIR_RETENTION,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "AH x LIFTK retention must equal 87.94",
    )

    # Always derive the binary label from the direct value; never copy a
    # provider-side summary or a possibly stale confusion-matrix column.
    output["target_binder"] = output["target_retention"].ge(
        RETENTION_THRESHOLD
    ).astype(int)
    binders = int(output["target_binder"].sum())
    _require(
        binders == EXPECTED_BINDERS
        and len(output) - binders == EXPECTED_NONBINDERS,
        "corrected threshold labels must contain 61 binders and 59 non-binders",
    )
    return output.sort_values("eval_row_id", kind="mergesort").reset_index(drop=True)


def load_normalized_predictions(paths: Sequence[Path]) -> pd.DataFrame:
    """Load target-free prediction vectors under one exact schema."""

    _require(bool(paths), "at least one prediction CSV is required")
    frames: list[pd.DataFrame] = []
    expected_columns = set(PREDICTION_COLUMNS)
    for path in map(Path, paths):
        _require(path.is_file(), f"prediction file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n\r").split(",")
        _require(
            set(header) == expected_columns and len(header) == len(PREDICTION_COLUMNS),
            "prediction schema must be exactly {}: {}".format(
                list(PREDICTION_COLUMNS), path
            ),
        )
        frame = _read_string_csv(path)
        frame = frame[list(PREDICTION_COLUMNS)].copy()
        for column in ("eval_row_id", "model", "seed"):
            _require(
                not bool(frame[column].astype(str).eq("").any()),
                f"prediction {column} contains empty values: {path}",
            )
        frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(float)
        _require(
            bool(np.isfinite(frame["score"].to_numpy(dtype=float)).all()),
            f"prediction score contains non-finite values: {path}",
        )
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    duplicate = output.duplicated(["model", "seed", "eval_row_id"], keep=False)
    _require(
        not bool(duplicate.any()),
        "a model/seed group contains duplicate evaluation IDs",
    )
    return output


def validate_prediction_membership(
    predictions: pd.DataFrame, panel: pd.DataFrame
) -> None:
    panel_ids = set(panel["eval_row_id"].astype(str))
    _require(panel_ids, "retention panel is empty")
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        observed = set(group["eval_row_id"].astype(str))
        missing = sorted(panel_ids.difference(observed))
        extra = sorted(observed.difference(panel_ids))
        _require(
            len(group) == EXPECTED_ROWS and not missing and not extra,
            (
                "model {!r} seed {!r} must score the exact 120-row panel; "
                "rows={}, missing={}, extra={}"
            ).format(model, seed, len(group), len(missing), len(extra)),
        )


def evaluate_predictions(
    predictions: pd.DataFrame, panel: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-seed, per-peptide, and ranked per-pair metric tables."""

    validate_prediction_membership(predictions, panel)
    metric_rows: list[dict[str, object]] = []
    peptide_frames: list[pd.DataFrame] = []
    ranked_frames: list[pd.DataFrame] = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        merged = group.merge(panel, on="eval_row_id", how="inner", validate="one_to_one")
        _require(len(merged) == EXPECTED_ROWS, "prediction/retention join lost rows")
        summary, peptide, ranked = evaluate_wetlab_predictions(
            merged,
            peptide_column="peptide_design_code",
            affibody_column="affibody_design_code",
            score_column="score",
            binder_column="target_binder",
            retention_column="target_retention",
            k_values=K_VALUES,
        )
        _require(
            int(summary["n_examples"]) == EXPECTED_ROWS
            and int(summary["n_binders"]) == EXPECTED_BINDERS,
            "metric evaluator changed the corrected panel",
        )
        metric_rows.append({"model": str(model), "seed": str(seed), **summary})

        top_three = (
            ranked.loc[ranked["within_peptide_rank"].le(3)]
            .sort_values(
                ["peptide_design_code", "within_peptide_rank"], kind="mergesort"
            )
            .groupby("peptide_design_code", sort=True)["affibody_design_code"]
            .apply(lambda values: json.dumps(list(map(str, values))))
        )
        peptide = peptide.merge(
            top_three.rename("top3_affibodies"),
            left_on="peptide_design_code",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        _require(
            len(peptide) == EXPECTED_PEPTIDES
            and int(peptide["peptide_design_code"].eq("AH").sum()) == 1,
            "per-peptide output must contain all 12 peptides including AH",
        )
        peptide.insert(0, "seed", str(seed))
        peptide.insert(0, "model", str(model))
        peptide_frames.append(peptide)

        # ``ranked`` retains the target-free model/seed columns from the
        # merged prediction table.  Normalize and place them first rather
        # than inserting duplicate columns.
        ranked["model"] = str(model)
        ranked["seed"] = str(seed)
        ranked = ranked[
            ["model", "seed"]
            + [column for column in ranked.columns if column not in {"model", "seed"}]
        ]
        ranked_frames.append(ranked)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["model", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    per_peptide = pd.concat(peptide_frames, ignore_index=True).sort_values(
        ["model", "seed", "peptide_design_code"], kind="mergesort"
    ).reset_index(drop=True)
    ranked = pd.concat(ranked_frames, ignore_index=True).sort_values(
        ["model", "seed", "peptide_design_code", "within_peptide_rank"],
        kind="mergesort",
    ).reset_index(drop=True)
    return metrics, per_peptide, ranked


def summarize_across_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize the requested metrics while preserving every seed value."""

    missing = set(SUMMARY_METRICS).difference(metrics.columns)
    _require(not missing, f"metric table is missing columns {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", sort=True):
        group = group.sort_values("seed", kind="mergesort")
        row: dict[str, object] = {
            "model": str(model),
            "n_seeds": int(len(group)),
            "seeds": json.dumps(group["seed"].astype(str).tolist()),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(dtype=float)
            _require(
                not bool(np.isinf(values).any()),
                f"{metric} contains an infinite value",
            )
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(finite)) if len(finite) else float("nan")
            )
            row[f"{metric}_sd"] = (
                float(np.std(finite, ddof=1)) if len(finite) >= 2 else float("nan")
            )
            row[f"{metric}_individual"] = json.dumps(
                [float(value) if math.isfinite(value) else None for value in values]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        frame.to_csv(temporary, index=False)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_outputs(
    output_dir: Path,
    tables: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    output_dir = validate_private_output_path(output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    output_records: dict[str, object] = {}
    for filename, frame in tables.items():
        _require(Path(filename).name == filename, "output filename must be local")
        path = output_dir / filename
        _atomic_write_csv(frame, path)
        output_records[filename] = {
            **file_provenance(path),
            "rows": int(len(frame)),
        }
    payload = dict(manifest)
    payload["outputs"] = output_records
    # The manifest is the completion marker and is written last.
    _atomic_write_json(payload, output_dir / "manifest.json")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-panel", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Exact four-column prediction CSV; repeat for multiple files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    panel = load_corrected_retention_panel(args.retention_panel)
    predictions = load_normalized_predictions(args.predictions)
    metrics, per_peptide, ranked = evaluate_predictions(predictions, panel)
    seed_summary = summarize_across_seeds(metrics)
    corrected = panel.loc[
        panel["peptide_design_code"].eq(CORRECTED_PAIR[0])
        & panel["affibody_design_code"].eq(CORRECTED_PAIR[1])
    ].iloc[0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "evaluation_only_no_training_or_model_selection",
        "script": file_provenance(Path(__file__)),
        "sources": {
            "corrected_retention_panel": file_provenance(args.retention_panel),
            "normalized_predictions": [
                file_provenance(path) for path in args.predictions
            ],
        },
        "panel_contract": {
            "library": "LibB",
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_AFFIBODIES,
            "complete_cartesian_matrix": True,
            "binder_threshold": RETENTION_THRESHOLD,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "membership_sha256": _membership_sha256(panel["eval_row_id"]),
            "corrected_cell": {
                "eval_row_id": str(corrected["eval_row_id"]),
                "peptide_design_code": CORRECTED_PAIR[0],
                "affibody_design_code": CORRECTED_PAIR[1],
                "target_retention": float(corrected["target_retention"]),
                "target_binder": int(corrected["target_binder"]),
            },
        },
        "prediction_contract": {
            "columns": list(PREDICTION_COLUMNS),
            "score_direction": "higher_is_more_likely_binder",
            "model_seed_groups": int(metrics.shape[0]),
            "models": sorted(set(metrics["model"].astype(str))),
            "every_group_scores_exact_panel": True,
        },
        "metrics": {
            "requested": list(SUMMARY_METRICS),
            "within_peptide_aggregation": "arithmetic mean across the 12 peptide rows",
            "precision_and_hit_aggregation": "computed within peptide then macro-averaged",
            "best_retention_at_k": "best direct retention among the model's top-k choices for each peptide",
            "regret_at_k": "best retention among all 10 choices minus best retention among the model's top-k choices",
            "tie_policy": TIE_POLICY,
            "k_values": list(K_VALUES),
        },
    }
    published = publish_outputs(
        args.output_dir,
        {
            "metrics_by_model_seed.csv": metrics,
            "metrics_seed_summary.csv": seed_summary,
            "per_peptide_metrics.csv": per_peptide,
            "ranked_per_pair.csv": ranked,
        },
        manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).resolve()),
                "models": published["prediction_contract"]["models"],
                "model_seed_groups": published["prediction_contract"][
                    "model_seed_groups"
                ],
                "rows": EXPECTED_ROWS,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
