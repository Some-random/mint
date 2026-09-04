#!/usr/bin/env python
"""Build target-free LibB R009+R010 pooled-count predictions for all 120 designs.

The provider score is the raw R009 count plus the raw R010 count for the same
peptide--Affibody pair.  Absence from one round is treated as a zero count.
This command reads no retention values or binder labels and does not fit or
select a model.  It publishes the exact four-column prediction schema consumed
by ``recompute_libb_metrics_120.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-provider-pooled-r009-r010-predictions-v1"
MODEL_NAME = "provider_pooled_r009_r010_count"
SEED = "not_applicable"
EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
PREDICTION_COLUMNS = ("eval_row_id", "model", "seed", "score")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file(), f"source file does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _membership_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_private_output_path(path: Path) -> Path:
    private_root = (REPO_ROOT / "private_data").resolve()
    output = Path(path).resolve()
    _require(private_root.is_dir(), "private_data directory is missing")
    try:
        relative = output.relative_to(private_root)
        relative_repo = output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("output must be below private_data") from exc
    _require(bool(relative.parts), "output must be below private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    # Private derived artifacts must never be published accidentally.
    import subprocess

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative_repo)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    _require(ignored.returncode == 0, "output is not Git-ignored")
    return output


def load_panel_identities(path: Path) -> pd.DataFrame:
    """Read only target-free identity columns from the complete design roster."""

    frame = pd.read_csv(
        path,
        usecols=["library", "pep", "aff", "pair_uid"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[frame["library"].eq("LibB")].copy()
    _require(len(frame) == EXPECTED_ROWS, "LibB identity roster must contain 120 rows")
    for column in ("pep", "aff", "pair_uid"):
        _require(not bool(frame[column].eq("").any()), f"blank {column} in roster")
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate pair_uid")
    _require(
        not bool(frame.duplicated(["pep", "aff"], keep=False).any()),
        "duplicate peptide--Affibody pair",
    )
    peptides = sorted(set(frame["pep"]))
    affibodies = sorted(set(frame["aff"]))
    _require(len(peptides) == EXPECTED_PEPTIDES, "expected 12 LibB peptides")
    _require(len(affibodies) == EXPECTED_AFFIBODIES, "expected 10 LibB Affibodies")
    expected = {(pep, aff) for pep in peptides for aff in affibodies}
    observed = set(zip(frame["pep"], frame["aff"]))
    _require(observed == expected, "identity roster is not the complete 12-by-10 matrix")
    return frame.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)


def load_round(path: Path, count_name: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["pep", "aff", "count"],
        dtype={"pep": str, "aff": str, "count": np.int64},
        keep_default_na=False,
        na_filter=False,
    )
    _require(not bool(frame.duplicated(["pep", "aff"], keep=False).any()), f"duplicates in {path}")
    _require(bool(frame["count"].gt(0).all()), f"nonpositive raw count in {path}")
    return frame.rename(columns={"count": count_name})


def build_scores(
    identities: pd.DataFrame, round9: pd.DataFrame, round10: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = identities.merge(round9, on=["pep", "aff"], how="left", validate="one_to_one")
    work = work.merge(round10, on=["pep", "aff"], how="left", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        work[column] = work[column].fillna(0).astype(np.int64)
        _require(bool(work[column].ge(0).all()), f"negative {column}")
    work["pooled_r009_r010_count"] = work["r009_count"] + work["r010_count"]
    _require(
        bool(work["pooled_r009_r010_count"].ge(0).all()),
        "negative pooled count",
    )

    components = work[
        [
            "pair_uid",
            "pep",
            "aff",
            "r009_count",
            "r010_count",
            "pooled_r009_r010_count",
        ]
    ].rename(
        columns={
            "pair_uid": "eval_row_id",
            "pep": "peptide_design_code",
            "aff": "affibody_design_code",
        }
    )
    predictions = pd.DataFrame(
        {
            "eval_row_id": components["eval_row_id"],
            "model": MODEL_NAME,
            "seed": SEED,
            "score": components["pooled_r009_r010_count"].astype(np.int64),
        }
    )
    _require(tuple(predictions.columns) == PREDICTION_COLUMNS, "prediction schema changed")
    _require(len(predictions) == EXPECTED_ROWS, "prediction row count changed")
    return predictions, components


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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-roster", type=Path, required=True)
    parser.add_argument("--round9", type=Path, required=True)
    parser.add_argument("--round10", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = _validate_private_output_path(args.output_dir)
    identities = load_panel_identities(args.identity_roster)
    round9 = load_round(args.round9, "r009_count")
    round10 = load_round(args.round10, "r010_count")
    predictions, components = build_scores(identities, round9, round10)

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    prediction_path = output_dir / "normalized_predictions.csv"
    component_path = output_dir / "score_components.csv"
    _atomic_csv(predictions, prediction_path)
    _atomic_csv(components, component_path)
    corrected = components.loc[
        components["peptide_design_code"].eq("AH")
        & components["affibody_design_code"].eq("LIFTK")
    ]
    _require(len(corrected) == 1, "AH x LIFTK is missing from output")
    corrected_row = corrected.iloc[0]
    _require(
        int(corrected_row["r009_count"]) == 352
        and int(corrected_row["r010_count"]) == 903
        and int(corrected_row["pooled_r009_r010_count"]) == 1255,
        "AH x LIFTK provider counts changed",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "target_free_direct_provider_score_no_training_or_model_selection",
        "score_definition": "raw R009 count + raw R010 count for the same pair; absent round count is zero",
        "sources": {
            "identity_roster": file_record(args.identity_roster),
            "raw_r009": file_record(args.round9),
            "raw_r010": file_record(args.round10),
            "script": file_record(Path(__file__)),
        },
        "panel": {
            "library": "LibB",
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_AFFIBODIES,
            "membership_sha256": _membership_sha256(predictions["eval_row_id"]),
        },
        "corrected_pair_score": {
            "peptide_design_code": "AH",
            "affibody_design_code": "LIFTK",
            "eval_row_id": str(corrected_row["eval_row_id"]),
            "r009_count": 352,
            "r010_count": 903,
            "score": 1255,
        },
        "outputs": {
            prediction_path.name: {
                **file_record(prediction_path),
                "rows": EXPECTED_ROWS,
                "columns": list(PREDICTION_COLUMNS),
            },
            component_path.name: {
                **file_record(component_path),
                "rows": EXPECTED_ROWS,
            },
        },
    }
    _atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps({"output_dir": str(output_dir.resolve()), "rows": EXPECTED_ROWS}, sort_keys=True))


if __name__ == "__main__":
    main()
