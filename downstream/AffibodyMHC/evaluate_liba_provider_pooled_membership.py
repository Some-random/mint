#!/usr/bin/env python3
"""Evaluate LibA pooled-top-2% membership and document a candidate-menu benchmark.

This is a retrospective evaluation step, separate from the target-free score
and candidate builder.  It measures how often pooled-count membership
(``R009 + R010 >= 13``) identifies binders in the 108-pair direct-retention
panel.  It also records the purely extrapolative count obtained by multiplying
the fixed unmeasured-menu size by that measured-panel precision.

The extrapolation is explicitly not a model metric, prediction guarantee, or
prospective validation.  It has no observed outcomes for the unmeasured menu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    sha256_file,
    validate_private_output_path,
)


LIBRARY = "LibA"
SCHEMA_VERSION = "liba-provider-pooled-membership-retrospective-benchmark-v1"
EXPECTED_PANEL_ROWS = 108
EXPECTED_PEPTIDES = 9
EXPECTED_BINDERS = 38
EXPECTED_NONBINDERS = 70
EXPECTED_MENU_ROWS = 76
MEMBERSHIP_CUTOFF = 13
RETENTION_CUTOFF = 75.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _membership_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file(), f"source file does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["library", "pair_uid", "peptide_design_code", "target_retention"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[frame["library"].eq(LIBRARY)].copy()
    _require(len(frame) == EXPECTED_PANEL_ROWS, "LibA panel must have 108 rows")
    _require(frame["pair_uid"].is_unique, "duplicate LibA panel pair UID")
    frame["target_retention"] = pd.to_numeric(frame["target_retention"], errors="raise")
    _require(bool(np.isfinite(frame["target_retention"]).all()), "nonfinite retention")
    frame["target_binder"] = frame["target_retention"].ge(RETENTION_CUTOFF)
    _require(int(frame["target_binder"].sum()) == EXPECTED_BINDERS, "binder count changed")
    _require(int((~frame["target_binder"]).sum()) == EXPECTED_NONBINDERS, "nonbinder count changed")
    _require(frame["peptide_design_code"].nunique() == EXPECTED_PEPTIDES, "peptide count changed")
    return frame


def load_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "eval_row_id",
            "peptide_design_code",
            "pooled_r009_r010_count",
        ],
        dtype={"eval_row_id": str, "peptide_design_code": str},
        keep_default_na=False,
        na_filter=False,
    )
    _require(len(frame) == EXPECTED_PANEL_ROWS, "score components must have 108 rows")
    _require(frame["eval_row_id"].is_unique, "duplicate score-component row ID")
    frame["pooled_r009_r010_count"] = pd.to_numeric(
        frame["pooled_r009_r010_count"], errors="raise"
    ).astype(np.int64)
    return frame


def load_menu(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["pair_uid", "peptide_design_code", "within_peptide_rank"],
        dtype={"pair_uid": str, "peptide_design_code": str},
        keep_default_na=False,
        na_filter=False,
    )
    _require(len(frame) == EXPECTED_MENU_ROWS, "fixed unmeasured menu must have 76 rows")
    _require(frame["pair_uid"].is_unique, "duplicate unmeasured-menu pair UID")
    frame["within_peptide_rank"] = pd.to_numeric(
        frame["within_peptide_rank"], errors="raise"
    ).astype(np.int64)
    _require(bool(frame["within_peptide_rank"].between(1, 10).all()), "menu rank outside 1..10")
    return frame


def calculate(
    panel: pd.DataFrame, scores: pd.DataFrame, menu: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = panel.merge(
        scores,
        left_on="pair_uid",
        right_on="eval_row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_score"),
    )
    _require(
        len(merged) == len(panel) == len(scores),
        "panel/score join lost rows",
    )
    _require(
        merged["peptide_design_code"].eq(merged["peptide_design_code_score"]).all(),
        "peptide identity mismatch between panel and score components",
    )
    merged["pooled_positive_member"] = merged["pooled_r009_r010_count"].ge(
        MEMBERSHIP_CUTOFF
    )
    selected = merged["pooled_positive_member"]
    binder = merged["target_binder"]
    tp = int((selected & binder).sum())
    fp = int((selected & ~binder).sum())
    fn = int((~selected & binder).sum())
    tn = int((~selected & ~binder).sum())
    precision = float(tp / (tp + fp))
    recall = float(tp / (tp + fn))
    extrapolated = float(len(menu) * precision)
    metrics = pd.DataFrame(
        [
            {
                "library": LIBRARY,
                "measured_panel_rows": int(len(merged)),
                "retention_binder_cutoff": RETENTION_CUTOFF,
                "pooled_positive_membership_cutoff": MEMBERSHIP_CUTOFF,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "measured_pooled_positive_members": int(selected.sum()),
                "measured_binders": int(binder.sum()),
                "precision": precision,
                "recall": recall,
                "fixed_unmeasured_menu_rows": int(len(menu)),
                "extrapolated_expected_binders": extrapolated,
                "extrapolation_formula": "fixed_unmeasured_menu_rows * true_positives / measured_pooled_positive_members",
                "extrapolation_status": "retrospective_descriptive_benchmark_not_model_metric_or_guarantee",
            }
        ]
    )

    menu_counts = menu.groupby("peptide_design_code", sort=True).size()
    records = []
    for peptide, group in merged.groupby("peptide_design_code", sort=True):
        member = group["pooled_positive_member"]
        group_binder = group["target_binder"]
        member_count = int(member.sum())
        binders_in_members = int((member & group_binder).sum())
        menu_count = int(menu_counts.get(peptide, 0))
        records.append(
            {
                "peptide_design_code": peptide,
                "measured_pairs": int(len(group)),
                "measured_binders": int(group_binder.sum()),
                "measured_pooled_positive_members": member_count,
                "measured_binders_among_pooled_positive_members": binders_in_members,
                "measured_member_precision": (
                    float(binders_in_members / member_count) if member_count else float("nan")
                ),
                "fixed_unmeasured_menu_rows": menu_count,
                "extrapolated_expected_binders_using_global_precision": float(
                    menu_count * precision
                ),
            }
        )
    composition = pd.DataFrame(records)
    _require(int(composition["fixed_unmeasured_menu_rows"].sum()) == len(menu), "menu sum changed")
    return metrics, composition


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _require(not path.exists() and not temporary.exists(), f"output exists: {path}")
    try:
        frame.to_csv(temporary, index=False, float_format="%.12g")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-panel", type=Path, required=True)
    parser.add_argument("--score-components", type=Path, required=True)
    parser.add_argument("--unmeasured-menu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output exists; refusing overwrite")
    sources = {
        "retention_panel": file_record(args.retention_panel),
        "score_components": file_record(args.score_components),
        "fixed_unmeasured_menu": file_record(args.unmeasured_menu),
        "script": file_record(Path(__file__)),
    }
    panel = load_panel(args.retention_panel)
    scores = load_scores(args.score_components)
    menu = load_menu(args.unmeasured_menu)
    _require(not bool(menu["pair_uid"].isin(panel["pair_uid"]).any()), "menu contains measured pair")
    metrics, composition = calculate(panel, scores, menu)

    row = metrics.iloc[0]
    _require(int(row["true_positives"]) == 32, "expected 32 true positives")
    _require(int(row["false_positives"]) == 23, "expected 23 false positives")
    _require(int(row["false_negatives"]) == 6, "expected six false negatives")
    _require(int(row["true_negatives"]) == 47, "expected 47 true negatives")

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    metric_path = output_dir / "measured_membership_metrics.csv"
    composition_path = output_dir / "per_peptide_composition.csv"
    _atomic_csv(metrics, metric_path)
    _atomic_csv(composition, composition_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "retrospective_evaluation_and_descriptive_extrapolation_no_training",
        "sources": sources,
        "measured_membership": {
            "cutoff": MEMBERSHIP_CUTOFF,
            "true_positives": 32,
            "false_positives": 23,
            "false_negatives": 6,
            "true_negatives": 47,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
        },
        "unmeasured_menu_benchmark": {
            "rows": int(len(menu)),
            "extrapolated_expected_binders": float(row["extrapolated_expected_binders"]),
            "formula": "76 * 32 / 55",
            "is_model_metric": False,
            "is_prediction_guarantee": False,
            "is_prospective_validation": False,
            "warning": (
                "This applies a global measured-panel precision to a differently composed "
                "unmeasured menu. Per-peptide measured precision is highly heterogeneous, "
                "and some menu peptides have no or no-positive measured cutoff members."
            ),
        },
        "memberships": {
            "measured_panel_sha256": _membership_sha256(panel["pair_uid"]),
            "fixed_unmeasured_menu_sha256": _membership_sha256(menu["pair_uid"]),
        },
        "outputs": {
            metric_path.name: {**file_record(metric_path), "rows": int(len(metrics))},
            composition_path.name: {**file_record(composition_path), "rows": int(len(composition))},
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "extrapolated_expected_binders": float(row["extrapolated_expected_binders"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
