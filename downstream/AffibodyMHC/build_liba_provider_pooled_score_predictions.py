#!/usr/bin/env python3
"""Build target-free LibA provider-count scores and a prospective candidate tier.

The score for a peptide--Affibody pair is its raw R009 count plus its raw R010
count.  A pair absent from one round receives zero for that round.  Global
pooled-positive membership is defined by first pooling the two rounds and then
taking the tie-inclusive top 2% of that pooled distribution.

This command has two outputs:

1. the exact four-column normalized prediction schema for all 108 measured
   LibA panel identities; and
2. all on-design pooled-positive pairs for the same nine peptide targets after
   removing the 108 measured pair identities, ranked within peptide by pooled
   count without using retention values.

Only identity columns are read from the panel roster.  The command reads no
retention value or binder label and fits no model.
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (  # noqa: E402
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    EXPECTED_AFFIBODY_X_POSITIONS,
    OMITTED_LINKER,
    fill_template,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    opaque_id,
    validate_private_output_path,
)


LIBRARY = "LibA"
SCHEMA_VERSION = "liba-provider-pooled-r009-r010-panel-and-candidates-v1"
MODEL_NAME = "provider_pooled_r009_r010_count"
SEED = "not_applicable"
PREDICTION_COLUMNS = ("eval_row_id", "model", "seed", "score")

EXPECTED_PANEL_ROWS = 108
EXPECTED_PEPTIDES = 9
EXPECTED_MEASURED_AFFIBODIES = 12
TOP_FRACTION = 0.02
EXPECTED_POOLED_ROWS = 881_464
EXPECTED_NOMINAL_RANK = 17_630
EXPECTED_POOLED_CUTOFF = 13
EXPECTED_GLOBAL_POSITIVES = 19_343
EXPECTED_TARGET_POSITIVES = 7_841
EXPECTED_MEASURED_POSITIVE_OVERLAP = 55
EXPECTED_UNMEASURED_TARGET_POSITIVES = 7_786
EXPECTED_TOP10_ROWS = 76

PEPTIDE_DESIGN_ALPHABET = frozenset("ADEFHIKLNPQSTVY")
AFFIBODY_DESIGN_ALPHABET = frozenset("ADEFHIKLNPQSTVY")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _membership_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_output_path(path: Path) -> Path:
    output = validate_private_output_path(path, REPO_ROOT)
    _require(not output.exists(), "output exists; refusing overwrite")
    relative = output.resolve().relative_to(REPO_ROOT)
    import subprocess

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    _require(ignored.returncode == 0, "output is not Git-ignored")
    return output


def load_panel_identities(path: Path) -> pd.DataFrame:
    """Read only nonsupervisory identity columns from the complete roster."""

    frame = pd.read_csv(
        path,
        usecols=["library", "pep", "aff", "pair_uid"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    frame = frame.loc[frame["library"].eq(LIBRARY)].copy()
    _require(len(frame) == EXPECTED_PANEL_ROWS, "LibA identity roster must contain 108 rows")
    for column in ("pep", "aff", "pair_uid"):
        _require(not bool(frame[column].eq("").any()), f"blank {column} in roster")
    _require(not bool(frame["pair_uid"].duplicated().any()), "duplicate pair_uid")
    _require(
        not bool(frame.duplicated(["pep", "aff"], keep=False).any()),
        "duplicate peptide--Affibody pair",
    )
    peptides = sorted(frame["pep"].unique())
    affibodies = sorted(frame["aff"].unique())
    _require(len(peptides) == EXPECTED_PEPTIDES, "expected nine LibA peptides")
    _require(len(affibodies) == EXPECTED_MEASURED_AFFIBODIES, "expected 12 LibA Affibodies")
    expected = {(peptide, affibody) for peptide in peptides for affibody in affibodies}
    observed = set(zip(frame["pep"], frame["aff"]))
    _require(observed == expected, "identity roster is not the complete 9-by-12 matrix")
    _require(
        frame["pep"].map(lambda value: set(value).issubset(PEPTIDE_DESIGN_ALPHABET)).all(),
        "panel peptide lies outside the LibA design alphabet",
    )
    _require(
        frame["aff"].map(lambda value: set(value).issubset(AFFIBODY_DESIGN_ALPHABET)).all(),
        "panel Affibody lies outside the LibA design alphabet",
    )
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
    _require(bool(frame["pep"].str.len().eq(2).all()), f"bad peptide code in {path}")
    _require(bool(frame["aff"].str.len().eq(4).all()), f"bad Affibody code in {path}")
    _require(bool(frame["count"].gt(0).all()), f"nonpositive raw count in {path}")
    return frame.rename(columns={"count": count_name})


def pool_rounds(round9: pd.DataFrame, round10: pd.DataFrame) -> pd.DataFrame:
    pooled = round9.merge(round10, on=["pep", "aff"], how="outer", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        pooled[column] = pooled[column].fillna(0).astype(np.int64)
        _require(bool(pooled[column].ge(0).all()), f"negative {column}")
    pooled["pooled_r009_r010_count"] = pooled["r009_count"] + pooled["r010_count"]
    _require(bool(pooled["pooled_r009_r010_count"].gt(0).all()), "nonpositive pooled count")
    return pooled


def tie_inclusive_top_fraction(
    pooled: pd.DataFrame, top_fraction: float = TOP_FRACTION
) -> tuple[pd.DataFrame, int, int]:
    _require(0.0 < float(top_fraction) < 1.0, "top fraction must lie between zero and one")
    nominal_rank = max(1, int(math.ceil(float(top_fraction) * len(pooled))))
    counts = pooled["pooled_r009_r010_count"].to_numpy(dtype=np.int64)
    cutoff = int(np.partition(counts, len(counts) - nominal_rank)[len(counts) - nominal_rank])
    selected = pooled.loc[pooled["pooled_r009_r010_count"].ge(cutoff)].copy()
    selected = selected.sort_values(
        ["pooled_r009_r010_count", "pep", "aff"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return selected, cutoff, nominal_rank


def build_panel_scores(
    identities: pd.DataFrame, pooled: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = identities.merge(pooled, on=["pep", "aff"], how="left", validate="one_to_one")
    for column in ("r009_count", "r010_count", "pooled_r009_r010_count"):
        work[column] = work[column].fillna(0).astype(np.int64)
        _require(bool(work[column].ge(0).all()), f"negative {column}")
    components = work[
        ["pair_uid", "pep", "aff", "r009_count", "r010_count", "pooled_r009_r010_count"]
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
    _require(len(predictions) == EXPECTED_PANEL_ROWS, "prediction row count changed")
    return predictions, components


def _on_design(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["pep"].map(lambda value: len(value) == 2 and set(value).issubset(PEPTIDE_DESIGN_ALPHABET))
        & frame["aff"].map(
            lambda value: len(value) == 4 and set(value).issubset(AFFIBODY_DESIGN_ALPHABET)
        )
    )


def build_candidate_membership(
    selected: pd.DataFrame,
    identities: pd.DataFrame,
    template: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_peptides = sorted(identities["pep"].unique())
    measured_keys = set(zip(identities["pep"], identities["aff"]))
    target = selected.loc[selected["pep"].isin(target_peptides)].copy()
    _require(bool(_on_design(target).all()), "a pooled-positive current-target pair is off-design")
    target["is_directly_measured_pair"] = [
        (peptide, affibody) in measured_keys for peptide, affibody in zip(target["pep"], target["aff"])
    ]
    candidates = target.loc[~target["is_directly_measured_pair"]].copy()
    candidates = candidates.sort_values(
        ["pep", "pooled_r009_r010_count", "aff"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates["within_peptide_rank"] = (
        candidates.groupby("pep", sort=False).cumcount() + 1
    ).astype(np.int64)
    candidates["pooled_count_rank_min"] = (
        candidates.groupby("pep", sort=False)["pooled_r009_r010_count"]
        .rank(method="min", ascending=False)
        .astype(np.int64)
    )
    candidates["pooled_count_tie_size"] = (
        candidates.groupby(["pep", "pooled_r009_r010_count"], sort=False)["aff"]
        .transform("size")
        .astype(np.int64)
    )
    candidates["selected_top10"] = candidates["within_peptide_rank"].le(10)

    records: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        full_construct = fill_template(template, str(row.pep) + str(row.aff))
        chain1 = full_construct[:CHAIN1_LENGTH]
        omitted_linker = full_construct[
            CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)
        ]
        chain2 = full_construct[-CHAIN2_LENGTH:]
        _require(chain1[264:266] == row.pep, "candidate peptide mapping changed")
        observed_affibody = "".join(
            chain2[position - 1] for position in EXPECTED_AFFIBODY_X_POSITIONS[LIBRARY]
        )
        _require(observed_affibody == row.aff, "candidate Affibody mapping changed")
        _require(omitted_linker == OMITTED_LINKER, "candidate omitted linker changed")
        records.append(
            {
                "candidate_tier": "existing_target_pooled_positive_unmeasured_liba_design",
                "pair_uid": opaque_id(LIBRARY, row.pep, row.aff),
                "peptide_uid": opaque_id(LIBRARY, "pep", row.pep),
                "affibody_uid": opaque_id(LIBRARY, "aff", row.aff),
                "library": LIBRARY,
                "peptide_design_code": row.pep,
                "peptide_full_sequence": chain1[-9:],
                "affibody_design_code": row.aff,
                "r009_count": int(row.r009_count),
                "r010_count": int(row.r010_count),
                "pooled_r009_r010_count": int(row.pooled_r009_r010_count),
                "within_peptide_rank": int(row.within_peptide_rank),
                "pooled_count_rank_min": int(row.pooled_count_rank_min),
                "pooled_count_tie_size": int(row.pooled_count_tie_size),
                "selected_top10": bool(row.selected_top10),
                "deterministic_tie_breaker": "affibody_design_code_ascending",
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "omitted_linker_sequence": omitted_linker,
                "full_construct_audit_sequence": full_construct,
                "chain1_sha256": _sha256_text(chain1),
                "chain2_sha256": _sha256_text(chain2),
                "sequence_pair_sha256": _sha256_text(chain1 + "|" + chain2),
            }
        )
    output = pd.DataFrame(records)
    if len(output):
        _require(not bool(output["pair_uid"].duplicated().any()), "duplicate candidate pair UID")
        _require(
            not bool(output["pair_uid"].isin(identities["pair_uid"]).any()),
            "a measured pair survived candidate exclusion",
        )

    summary_records: list[dict[str, Any]] = []
    for peptide in target_peptides:
        block = output.loc[output["peptide_design_code"].eq(peptide)].copy()
        chosen = block.loc[block["selected_top10"]]
        boundary_count: int | None = None
        boundary_tie_size = 0
        boundary_tie_extends_beyond_top10 = False
        if len(chosen):
            boundary_count = int(chosen.iloc[-1]["pooled_r009_r010_count"])
            boundary_tie_size = int(block["pooled_r009_r010_count"].eq(boundary_count).sum())
            boundary_tie_extends_beyond_top10 = bool(
                len(block) > 10
                and int(block.iloc[10]["pooled_r009_r010_count"]) == boundary_count
            )
        summary_records.append(
            {
                "peptide_design_code": peptide,
                "pooled_positive_pairs_including_measured": int(target["pep"].eq(peptide).sum()),
                "measured_pooled_positive_pairs_removed": int(
                    (target["pep"].eq(peptide) & target["is_directly_measured_pair"]).sum()
                ),
                "unmeasured_candidate_pairs": int(len(block)),
                "top10_rows": int(len(chosen)),
                "top10_boundary_pooled_count": boundary_count,
                "top10_boundary_tie_size": boundary_tie_size,
                "top10_boundary_tie_extends_beyond_top10": boundary_tie_extends_beyond_top10,
            }
        )
    summary = pd.DataFrame(summary_records)
    return output, summary


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
    parser.add_argument("--sequence-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = _validate_output_path(args.output_dir)
    source_records = {
        "identity_roster": file_record(args.identity_roster),
        "raw_r009": file_record(args.round9),
        "raw_r010": file_record(args.round10),
        "sequence_zip": file_record(args.sequence_zip),
        "script": file_record(Path(__file__)),
    }

    identities = load_panel_identities(args.identity_roster)
    round9 = load_round(args.round9, "r009_count")
    round10 = load_round(args.round10, "r010_count")
    pooled = pool_rounds(round9, round10)
    selected, cutoff, nominal_rank = tie_inclusive_top_fraction(pooled)
    _require(len(pooled) == EXPECTED_POOLED_ROWS, "pooled R009/R010 union size changed")
    _require(nominal_rank == EXPECTED_NOMINAL_RANK, "nominal top-2% rank changed")
    _require(cutoff == EXPECTED_POOLED_CUTOFF, "pooled top-2% cutoff changed")
    _require(len(selected) == EXPECTED_GLOBAL_POSITIVES, "global pooled-positive count changed")

    predictions, components = build_panel_scores(identities, pooled)
    templates, sequence_member, sequence_deck_sha256 = load_provider_templates(args.sequence_zip)
    candidates, candidate_summary = build_candidate_membership(
        selected, identities, templates[LIBRARY]
    )
    target_positive_count = int(
        candidate_summary["pooled_positive_pairs_including_measured"].sum()
    )
    measured_positive_overlap = int(
        candidate_summary["measured_pooled_positive_pairs_removed"].sum()
    )
    _require(target_positive_count == EXPECTED_TARGET_POSITIVES, "current-target positive count changed")
    _require(
        measured_positive_overlap == EXPECTED_MEASURED_POSITIVE_OVERLAP,
        "measured pooled-positive overlap changed",
    )
    _require(
        len(candidates) == EXPECTED_UNMEASURED_TARGET_POSITIVES,
        "unmeasured current-target pooled-positive count changed",
    )
    top10 = candidates.loc[candidates["selected_top10"]].copy()
    _require(len(top10) == EXPECTED_TOP10_ROWS, "candidate top-10 row count changed")

    output_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    os.chmod(output_dir, 0o700)
    output_frames = {
        "normalized_predictions.csv": predictions,
        "score_components.csv": components,
        "pooled_positive_unmeasured_candidates.csv": candidates,
        "pooled_positive_unmeasured_top10.csv": top10,
        "candidate_summary_by_peptide.csv": candidate_summary,
    }
    output_records: dict[str, Any] = {}
    for name, frame in output_frames.items():
        path = output_dir / name
        _atomic_csv(frame, path)
        output_records[name] = {
            **file_record(path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "target_free_direct_provider_score_and_candidate_ranking_no_model_fit",
        "supervision_fields_read": [],
        "retention_values_read": False,
        "binder_labels_read": False,
        "model_fitted": False,
        "score_definition": (
            "raw R009 count + raw R010 count for the same pair; an absent round count is zero"
        ),
        "positive_definition": (
            "outer-join R009/R010, zero-fill, sum counts, then retain the tie-inclusive global top 2%"
        ),
        "panel": {
            "library": LIBRARY,
            "rows": EXPECTED_PANEL_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies": EXPECTED_MEASURED_AFFIBODIES,
            "membership_sha256": _membership_sha256(predictions["eval_row_id"]),
            "prediction_schema": list(PREDICTION_COLUMNS),
        },
        "pooled_distribution": {
            "union_rows": int(len(pooled)),
            "top_fraction": TOP_FRACTION,
            "nominal_top_fraction_rank": int(nominal_rank),
            "inclusive_count_cutoff": int(cutoff),
            "global_positive_rows_including_boundary_ties": int(len(selected)),
        },
        "candidate_tier": {
            "definition": (
                "on-design pooled-positive pairs for the nine measured LibA peptide targets, "
                "after removing every exact pair in the 108-pair measured identity roster"
            ),
            "current_target_pooled_positive_rows": target_positive_count,
            "measured_positive_pairs_removed": measured_positive_overlap,
            "unmeasured_candidate_rows": int(len(candidates)),
            "top10_rows": int(len(top10)),
            "ranking": [
                "pooled_r009_r010_count descending",
                "affibody_design_code ascending deterministic tie break",
            ],
            "full_membership_sha256": _membership_sha256(candidates["pair_uid"]),
            "top10_membership_sha256": _membership_sha256(top10["pair_uid"]),
            "hidden_MA_policy": (
                "Use the provider-displayed 58-aa Affibody template. The two hidden preceding "
                "residues MA change residue labels but are not silently inserted into sequences."
            ),
        },
        "sequence_source": {
            "member": sequence_member,
            "member_sha256": sequence_deck_sha256,
            "chain1_length": CHAIN1_LENGTH,
            "chain2_length": CHAIN2_LENGTH,
            "omitted_linker": OMITTED_LINKER,
        },
        "sources": source_records,
        "outputs": output_records,
    }
    _atomic_json(manifest, output_dir / "manifest.json")

    for name, record in source_records.items():
        if name == "script":
            continue
        _require(sha256_file(Path(record["path"])) == record["sha256"], f"source changed: {name}")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "panel_rows": int(len(predictions)),
                "candidate_rows": int(len(candidates)),
                "top10_rows": int(len(top10)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
