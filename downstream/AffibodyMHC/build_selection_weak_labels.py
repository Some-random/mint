#!/usr/bin/env python
"""Build provider-defined weak binder labels from R001 and pooled R009/R010.

Positive examples are the inclusive top fraction after outer-joining R009 and
R010 and summing raw counts.  Negative examples are pairs present in R001 that
never appear in any raw positive-selection round R002--R014.  Exact pairs from
the retention matrix are materialized in a separate audit table and excluded
from the training-label table.

The output intentionally stores short mutation codes rather than duplicating
the 270-aa and 58-aa model chains for every row.  Training code must reconstruct
those chains from the hashed provider templates.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import load_provider_templates
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    load_retention_table,
    opaque_id,
    sha256_file,
    validate_private_output_path,
)


LIBRARIES = ("LibA", "LibB")
RAW_SUBDIRECTORIES = {"LibA": "LibA Raw data", "LibB": "LibB Raw data"}
EXPECTED_RAW_ROUNDS = tuple(range(15))
POSITIVE_ROUNDS = (9, 10)
NEGATIVE_REFERENCE_ROUND = 1
NEGATIVE_EXCLUSION_ROUNDS = tuple(range(2, 15))
EXPECTED_POOLED_CUTOFF = {"LibA": 13, "LibB": 12}
RAW_COLUMNS = ("pep", "aff", "count", "frequency", "pvalue")
PEPTIDE_ALPHABET = frozenset("ADEFHIKLNPQSTVY")
AFFIBODY_ALPHABET = {
    "LibA": frozenset("ADEFHIKLNPQSTVY"),
    "LibB": frozenset("ADEFIKLMNSTVY"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, payload):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def discover_raw_round_files(raw_root):
    """Return the exact raw file for every library/round despite odd names."""
    result = {}
    for library in LIBRARIES:
        directory = raw_root / RAW_SUBDIRECTORIES[library]
        _require(directory.is_dir(), "missing raw directory {}".format(directory))
        round_files = {}
        for path in sorted(directory.glob("*.tsv")):
            match = re.search(r"(\d{3})n?_count_+freq_pvalue\.tsv$", path.name)
            _require(match is not None, "cannot parse round from {}".format(path.name))
            round_index = int(match.group(1))
            _require(round_index not in round_files, "duplicate {} round {}".format(library, round_index))
            round_files[round_index] = path
        _require(
            tuple(sorted(round_files)) == EXPECTED_RAW_ROUNDS,
            "{} raw rounds are incomplete".format(library),
        )
        result[library] = round_files
    return result


def _validate_codes(frame, library, path):
    spec = LIBRARY_SPECS[library]
    _require(not bool(frame[["pep", "aff"]].duplicated().any()), "duplicate keys in {}".format(path))
    _require(bool(frame["pep"].map(len).eq(spec["pep_length"]).all()), "bad peptide length in {}".format(path))
    _require(bool(frame["aff"].map(len).eq(spec["aff_length"]).all()), "bad Affibody length in {}".format(path))
    alphabet = set(AA_ALPHABET)
    _require(bool(frame["pep"].map(lambda value: set(value).issubset(alphabet)).all()), "noncanonical peptide in {}".format(path))
    _require(bool(frame["aff"].map(lambda value: set(value).issubset(alphabet)).all()), "noncanonical Affibody in {}".format(path))


def load_presence(path, library):
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=["pep", "aff"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _validate_codes(frame, library, path)
    return frame


def load_count_round(path, library):
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype={"pep": str, "aff": str, "count": np.int64},
        keep_default_na=False,
        na_filter=False,
    )
    _require(tuple(frame.columns) == RAW_COLUMNS, "unexpected schema in {}".format(path))
    _validate_codes(frame, library, path)
    _require(bool(frame["count"].gt(0).all()), "nonpositive count in {}".format(path))
    return frame[["pep", "aff", "count", "frequency"]].copy()


def key_series(frame):
    return frame["pep"].str.cat(frame["aff"], sep="|")


def pooled_top_fraction(round9, round10, top_fraction):
    """Outer-pool two count rounds and retain the inclusive top count fraction."""
    _require(0.0 < float(top_fraction) < 1.0, "top fraction must lie in (0,1)")
    left = round9[["pep", "aff", "count"]].rename(columns={"count": "r009_count"})
    right = round10[["pep", "aff", "count"]].rename(columns={"count": "r010_count"})
    pooled = left.merge(right, on=["pep", "aff"], how="outer", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        pooled[column] = pooled[column].fillna(0).astype(np.int64)
    pooled["pooled_r009_r010_count"] = pooled["r009_count"] + pooled["r010_count"]
    _require(bool(pooled["pooled_r009_r010_count"].gt(0).all()), "pooled counts must be positive")
    rank = max(1, int(math.ceil(float(top_fraction) * len(pooled))))
    counts = pooled["pooled_r009_r010_count"].to_numpy(dtype=np.int64)
    cutoff = int(np.partition(counts, len(counts) - rank)[len(counts) - rank])
    selected = pooled.loc[pooled["pooled_r009_r010_count"].ge(cutoff)].copy()
    selected = selected.sort_values(
        ["pooled_r009_r010_count", "pep", "aff"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return selected, cutoff, rank, len(pooled)


def provider_negative_rows(reference, later_frames):
    """Apply R001 minus the union of every later raw-round key."""
    candidates = set(key_series(reference).tolist())
    for frame in later_frames:
        candidates.difference_update(key_series(frame).tolist())
    mask = key_series(reference).isin(candidates)
    result = reference.loc[mask].copy()
    return result.sort_values(["count", "pep", "aff"], ascending=[False, True, True]).reset_index(drop=True)


def within_declared_alphabet(library, peptide, affibody):
    return set(peptide).issubset(PEPTIDE_ALPHABET) and set(affibody).issubset(
        AFFIBODY_ALPHABET[library]
    )


def _label_frame(library, positives, negatives, holdout_peptides, holdout_affibodies):
    positive = positives.copy()
    positive["r001_count"] = positive.pop("r001_count").astype(np.int64)
    positive["weak_label"] = 1
    positive["weak_label_source"] = "pooled_r009_r010_top_fraction"

    negative = negatives[["pep", "aff", "count"]].rename(columns={"count": "r001_count"})
    negative["r009_count"] = 0
    negative["r010_count"] = 0
    negative["pooled_r009_r010_count"] = 0
    negative["weak_label"] = 0
    negative["weak_label_source"] = "r001_absent_r002_r014"

    columns = [
        "pep",
        "aff",
        "r001_count",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "weak_label",
        "weak_label_source",
    ]
    output = pd.concat([positive[columns], negative[columns]], ignore_index=True)
    output.insert(0, "library", library)
    output["within_declared_library_alphabet"] = [
        int(within_declared_alphabet(library, peptide, affibody))
        for peptide, affibody in zip(output["pep"], output["aff"])
    ]
    output["negative_r001_count_ge_2"] = ((output["weak_label"] == 0) & (output["r001_count"] >= 2)).astype(int)
    output["negative_r001_count_ge_3"] = ((output["weak_label"] == 0) & (output["r001_count"] >= 3)).astype(int)
    output["negative_r001_count_ge_5"] = ((output["weak_label"] == 0) & (output["r001_count"] >= 5)).astype(int)
    output["shares_retention_peptide"] = output["pep"].isin(holdout_peptides).astype(int)
    output["shares_retention_affibody"] = output["aff"].isin(holdout_affibodies).astype(int)
    output["strict_retention_identity_cold_eligible"] = (
        (output["shares_retention_peptide"] == 0)
        & (output["shares_retention_affibody"] == 0)
    ).astype(int)
    output["pair_uid"] = [opaque_id(library, peptide, affibody) for peptide, affibody in zip(output["pep"], output["aff"])]
    output["peptide_uid"] = [opaque_id(library, "pep", peptide) for peptide in output["pep"]]
    output["affibody_uid"] = [opaque_id(library, "aff", affibody) for affibody in output["aff"]]
    _require(not bool(output["pair_uid"].duplicated().any()), "positive/negative overlap in {}".format(library))
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-fraction", type=float, default=0.02)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.raw_root.is_dir(), "raw root does not exist")
    _require(args.retention_csv.is_file(), "retention source does not exist")
    _require(args.sequence_zip.is_file(), "sequence archive does not exist")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)

    script_path = Path(__file__).resolve()
    source_hashes = {
        "retention_csv": sha256_file(args.retention_csv),
        "sequence_zip": sha256_file(args.sequence_zip),
    }
    templates, template_member, template_deck_hash = load_provider_templates(args.sequence_zip)
    retention = load_retention_table(args.retention_csv)
    raw_files = discover_raw_round_files(args.raw_root)

    label_blocks = []
    overlap_blocks = []
    library_summaries = {}
    raw_hashes = {}
    for library in LIBRARIES:
        files = raw_files[library]
        for round_index, path in sorted(files.items()):
            raw_hashes["{}:R{:03d}".format(library, round_index)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }

        reference = load_count_round(files[NEGATIVE_REFERENCE_ROUND], library)
        round9 = load_count_round(files[POSITIVE_ROUNDS[0]], library)
        round10 = load_count_round(files[POSITIVE_ROUNDS[1]], library)
        positives, cutoff, requested_rank, pooled_rows = pooled_top_fraction(
            round9, round10, args.top_fraction
        )
        _require(cutoff == EXPECTED_POOLED_CUTOFF[library], "{} pooled cutoff no longer reproduces provider deck".format(library))

        reference_counts = reference[["pep", "aff", "count"]].rename(columns={"count": "r001_count"})
        positives = positives.merge(reference_counts, on=["pep", "aff"], how="left", validate="one_to_one")
        positives["r001_count"] = positives["r001_count"].fillna(0).astype(np.int64)

        later_frames = [load_presence(files[index], library) for index in NEGATIVE_EXCLUSION_ROUNDS]
        negatives = provider_negative_rows(reference, later_frames)
        del later_frames

        holdout = retention.loc[retention["library"].eq(library), [
            "peptide_design_code",
            "affibody_design_code",
            "measurement_missing",
        ]].rename(columns={"peptide_design_code": "pep", "affibody_design_code": "aff"})
        holdout_keys = set(key_series(holdout).tolist())
        positive_keys = set(key_series(positives).tolist())
        negative_keys = set(key_series(negatives).tolist())
        overlap = holdout.copy()
        overlap.insert(0, "library", library)
        overlap["is_pooled_positive_before_exclusion"] = key_series(overlap).isin(positive_keys).astype(int)
        overlap["is_provider_negative_before_exclusion"] = key_series(overlap).isin(negative_keys).astype(int)
        overlap["pair_uid"] = [opaque_id(library, peptide, affibody) for peptide, affibody in zip(overlap["pep"], overlap["aff"])]
        overlap_blocks.append(overlap)

        positive_holdout = key_series(positives).isin(holdout_keys)
        negative_holdout = key_series(negatives).isin(holdout_keys)
        positives_train = positives.loc[~positive_holdout].copy()
        negatives_train = negatives.loc[~negative_holdout].copy()
        labels = _label_frame(
            library,
            positives_train,
            negatives_train,
            set(holdout["pep"]),
            set(holdout["aff"]),
        )
        label_blocks.append(labels)

        library_summaries[library] = {
            "r001_rows": int(len(reference)),
            "r009_rows": int(len(round9)),
            "r010_rows": int(len(round10)),
            "pooled_union_rows": int(pooled_rows),
            "top_fraction_requested_rank": int(requested_rank),
            "inclusive_pooled_count_cutoff": int(cutoff),
            "positive_rows_before_holdout_exclusion": int(len(positives)),
            "positive_rows_excluded_as_holdout": int(positive_holdout.sum()),
            "negative_rows_before_holdout_exclusion": int(len(negatives)),
            "negative_rows_excluded_as_holdout": int(negative_holdout.sum()),
            "training_positive_rows": int(labels["weak_label"].sum()),
            "training_negative_rows": int((labels["weak_label"] == 0).sum()),
            "training_on_design_rows": int(labels["within_declared_library_alphabet"].sum()),
            "training_negative_r001_count_ge_2": int(labels["negative_r001_count_ge_2"].sum()),
            "training_negative_r001_count_ge_3": int(labels["negative_r001_count_ge_3"].sum()),
            "training_negative_r001_count_ge_5": int(labels["negative_r001_count_ge_5"].sum()),
            "training_strict_identity_cold_rows": int(
                labels["strict_retention_identity_cold_eligible"].sum()
            ),
            "training_strict_identity_cold_positive": int(
                (
                    (labels["strict_retention_identity_cold_eligible"] == 1)
                    & (labels["weak_label"] == 1)
                ).sum()
            ),
            "training_strict_identity_cold_negative_r001_count_ge_3": int(
                (
                    (labels["strict_retention_identity_cold_eligible"] == 1)
                    & (labels["negative_r001_count_ge_3"] == 1)
                ).sum()
            ),
        }

    labels = pd.concat(label_blocks, ignore_index=True)
    overlaps = pd.concat(overlap_blocks, ignore_index=True)
    _require(not bool(labels["pair_uid"].duplicated().any()), "duplicate weak-label pair UID")
    _require(len(overlaps) == 228, "holdout audit must contain 228 designed pairs")
    _require(not bool(overlaps["pair_uid"].duplicated().any()), "duplicate holdout audit pair")
    _require(set(labels["pair_uid"]).isdisjoint(set(overlaps["pair_uid"])), "holdout pair leaked into weak labels")

    label_path = output_dir / "weak_labels.csv"
    overlap_path = output_dir / "holdout_overlap.csv"
    summary_path = output_dir / "label_summary.csv"
    labels.to_csv(label_path, index=False)
    overlaps.to_csv(overlap_path, index=False)
    summary_rows = []
    for library in LIBRARIES:
        subset = labels.loc[labels["library"].eq(library)]
        for weak_label, label_name in ((0, "negative"), (1, "positive")):
            selected = subset.loc[subset["weak_label"].eq(weak_label)]
            summary_rows.append(
                {
                    "library": library,
                    "weak_label": weak_label,
                    "label_name": label_name,
                    "rows": int(len(selected)),
                    "on_design_rows": int(selected["within_declared_library_alphabet"].sum()),
                    "unique_peptides": int(selected["peptide_uid"].nunique()),
                    "unique_affibodies": int(selected["affibody_uid"].nunique()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    for path in (label_path, overlap_path, summary_path):
        os.chmod(str(path), 0o600)

    manifest_path = output_dir / "manifest.json"
    output_paths = (label_path, overlap_path, summary_path)
    write_json(
        manifest_path,
        {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "configuration": {
                "positive_definition": "inclusive top fraction of outer-joined zero-filled raw R009+R010 count sum",
                "top_fraction": float(args.top_fraction),
                "negative_definition": "present in raw R001 and absent from every raw round R002-R014",
                "ambiguous_pairs": "excluded",
                "holdout_policy": "exclude every exact pair in the 228-cell retention matrix (227 measured)",
                "primary_training_filter_not_applied_here": "within_declared_library_alphabet is stored for downstream filtering",
            },
            "sources": {
                "raw_root": str(args.raw_root.resolve()),
                "raw_rounds": raw_hashes,
                "retention_csv": {"path": str(args.retention_csv.resolve()), "sha256": source_hashes["retention_csv"]},
                "sequence_zip": {
                    "path": str(args.sequence_zip.resolve()),
                    "sha256": source_hashes["sequence_zip"],
                    "template_member": template_member,
                    "template_deck_sha256": template_deck_hash,
                },
            },
            "templates": {
                library: {
                    "length": len(template),
                    "sha256": hashlib.sha256(template.encode("ascii")).hexdigest(),
                }
                for library, template in sorted(templates.items())
            },
            "libraries": library_summaries,
            "rows": {
                "weak_labels": int(len(labels)),
                "positive": int(labels["weak_label"].sum()),
                "negative": int((labels["weak_label"] == 0).sum()),
                "holdout_audit": int(len(overlaps)),
            },
            "outputs": {path.name: sha256_file(path) for path in output_paths},
            "permissions": {"directory": "0700", "files": "0600"},
            "environment": {
                "python": sys.version,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(labels), "libraries": library_summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
