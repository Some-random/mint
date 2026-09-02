#!/usr/bin/env python
"""Build the audited R009/R010 unlabeled pool for Affibody PU/PNU training.

The unlabeled pool is *not* an extra negative set.  It contains distinct pairs
seen in the R009/R010 outer union whose pooled count is strictly below the
library's positive cutoff.  Construction applies the cutoff before the broad
declared-library alphabet filter and before the identity-cold retention-panel
filter.  Every pair sharing either a peptide or an Affibody identity with the
same library's retention panel is excluded.

Only mutation codes and deterministic identifiers are stored.  Full MINT
chains must be reconstructed from the same provider templates used by the
existing positive/negative pipeline.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import stat
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_selection_weak_labels import (
    EXPECTED_POOLED_CUTOFF,
    LIBRARIES,
    discover_raw_round_files,
    load_count_round,
    within_declared_alphabet,
)
from downstream.AffibodyMHC.code_only_baseline import (
    LIBRARY_SPECS,
    opaque_id,
    sha256_file,
    validate_private_output_path,
)


ARTIFACT_NAME = "selection_pu_pool"
ARTIFACT_VERSION = 1
SCHEMA_VERSION = 1
POOL_FILENAME = "unlabeled_pool.csv.gz"
EXPECTED_UNLABELED_ROWS = {"LibA": 726_532, "LibB": 1_157_625}
OUTPUT_COLUMNS = (
    "library",
    "pep",
    "aff",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "positive_cutoff",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_retention_identities(path):
    """Read retention partner identities without reading outcome columns."""
    identities = pd.read_csv(
        path,
        usecols=["library", "peptide_design_code", "affibody_design_code"],
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    _require(len(identities) == 228, "retention identity table must have 228 rows")
    _require(set(identities["library"]) == set(LIBRARIES), "unexpected retention libraries")
    keys = ["library", "peptide_design_code", "affibody_design_code"]
    _require(not bool(identities.duplicated(keys).any()), "duplicate retention identity")
    for library in LIBRARIES:
        subset = identities.loc[identities["library"].eq(library)]
        spec = LIBRARY_SPECS[library]
        _require(len(subset) == spec["grid_rows"], "{} retention row count changed".format(library))
        _require(
            bool(subset["peptide_design_code"].map(len).eq(spec["pep_length"]).all()),
            "{} retention peptide length changed".format(library),
        )
        _require(
            bool(subset["affibody_design_code"].map(len).eq(spec["aff_length"]).all()),
            "{} retention Affibody length changed".format(library),
        )
    return identities


def load_existing_labels(path):
    """Load just enough of the established P/N artifact for disjointness."""
    labels = pd.read_csv(
        path,
        usecols=["library", "weak_label", "pair_uid"],
        dtype={"library": str, "weak_label": np.int8, "pair_uid": str},
        keep_default_na=False,
        na_filter=False,
    )
    _require(not labels.empty, "existing positive/negative table is empty")
    _require(set(labels["library"]) == set(LIBRARIES), "unexpected P/N libraries")
    _require(set(labels["weak_label"].astype(int)) == {0, 1}, "P/N labels must be binary")
    _require(not bool(labels["pair_uid"].duplicated().any()), "duplicate pair UID in P/N table")
    return labels


def _add_identifiers(frame, library):
    output = frame.copy()
    output["pair_uid"] = [
        opaque_id(library, peptide, affibody)
        for peptide, affibody in zip(output["pep"], output["aff"])
    ]
    output["peptide_uid"] = [opaque_id(library, "pep", value) for value in output["pep"]]
    output["affibody_uid"] = [opaque_id(library, "aff", value) for value in output["aff"]]
    _require(not bool(output["pair_uid"].duplicated().any()), "opaque pair UID collision")
    return output


def build_library_pool(round9, round10, retention, library, cutoff):
    """Return one library's cutoff-first, alphabet-valid, identity-cold U pool."""
    _require(library in LIBRARIES, "unknown library {}".format(library))
    _require(int(cutoff) > 0, "positive cutoff must be positive")
    for name, frame in (("R009", round9), ("R010", round10)):
        _require(
            {"pep", "aff", "count"}.issubset(frame.columns),
            "{} is missing required columns".format(name),
        )
        _require(not bool(frame[["pep", "aff"]].duplicated().any()), "duplicate {} pair".format(name))
        _require(bool(pd.to_numeric(frame["count"], errors="raise").gt(0).all()), "nonpositive {} count".format(name))

    left = round9[["pep", "aff", "count"]].rename(columns={"count": "r009_count"})
    right = round10[["pep", "aff", "count"]].rename(columns={"count": "r010_count"})
    pooled = left.merge(right, on=["pep", "aff"], how="outer", validate="one_to_one")
    for column in ("r009_count", "r010_count"):
        pooled[column] = pooled[column].fillna(0).astype(np.int64)
    pooled["pooled_r009_r010_count"] = pooled["r009_count"] + pooled["r010_count"]
    _require(bool(pooled["pooled_r009_r010_count"].gt(0).all()), "pooled counts must be positive")

    # This ordering is part of the versioned contract: cutoff first, then the
    # design alphabet and retention-partner identity exclusions.
    below_cutoff = pooled.loc[pooled["pooled_r009_r010_count"].lt(int(cutoff))].copy()
    on_design = np.asarray(
        [
            within_declared_alphabet(library, peptide, affibody)
            for peptide, affibody in zip(below_cutoff["pep"], below_cutoff["aff"])
        ],
        dtype=bool,
    )
    declared = below_cutoff.loc[on_design].copy()

    held = retention.loc[retention["library"].eq(library)]
    _require(len(held) == LIBRARY_SPECS[library]["grid_rows"], "retention identity count changed")
    held_peptides = set(held["peptide_design_code"].astype(str))
    held_affibodies = set(held["affibody_design_code"].astype(str))
    shares_peptide = declared["pep"].isin(held_peptides)
    shares_affibody = declared["aff"].isin(held_affibodies)
    eligible = declared.loc[~shares_peptide & ~shares_affibody].copy()
    eligible.insert(0, "library", library)
    eligible["positive_cutoff"] = int(cutoff)
    eligible = _add_identifiers(eligible, library)
    eligible = eligible[list(OUTPUT_COLUMNS)].sort_values("pair_uid").reset_index(drop=True)

    _require(bool(eligible["pooled_r009_r010_count"].lt(int(cutoff)).all()), "positive leaked into U")
    _require(
        all(
            within_declared_alphabet(library, peptide, affibody)
            for peptide, affibody in zip(eligible["pep"], eligible["aff"])
        ),
        "out-of-alphabet pair leaked into U",
    )
    _require(not bool(eligible["pep"].isin(held_peptides).any()), "retention peptide leaked into U")
    _require(not bool(eligible["aff"].isin(held_affibodies).any()), "retention Affibody leaked into U")

    audit = {
        "r009_rows": int(len(round9)),
        "r010_rows": int(len(round10)),
        "outer_union_rows": int(len(pooled)),
        "rows_strictly_below_positive_cutoff": int(len(below_cutoff)),
        "rows_after_declared_alphabet_filter": int(len(declared)),
        "rows_sharing_retention_peptide": int(shares_peptide.sum()),
        "rows_sharing_retention_affibody": int(shares_affibody.sum()),
        "rows_sharing_either_retention_partner": int((shares_peptide | shares_affibody).sum()),
        "eligible_unlabeled_rows": int(len(eligible)),
        "positive_cutoff": int(cutoff),
    }
    return eligible, audit


def validate_pool_contract(pool, labels, retention, expected_rows=None):
    """Fail closed on U membership, P/N overlap, identities, and row counts."""
    expected_rows = EXPECTED_UNLABELED_ROWS if expected_rows is None else expected_rows
    _require(tuple(pool.columns) == OUTPUT_COLUMNS, "unexpected unlabeled-pool schema")
    _require(not bool(pool["pair_uid"].duplicated().any()), "duplicate U pair UID")
    overlap = pool["pair_uid"].isin(labels["pair_uid"])
    _require(not bool(overlap.any()), "unlabeled pool overlaps existing P/N pairs")

    for library in LIBRARIES:
        subset = pool.loc[pool["library"].eq(library)]
        _require(
            len(subset) == int(expected_rows[library]),
            "{} U count changed: expected {}, found {}".format(
                library, expected_rows[library], len(subset)
            ),
        )
        _require(
            bool(subset["positive_cutoff"].eq(EXPECTED_POOLED_CUTOFF[library]).all()),
            "{} U cutoff column changed".format(library),
        )
        _require(
            bool(subset["pooled_r009_r010_count"].lt(EXPECTED_POOLED_CUTOFF[library]).all()),
            "{} positive-count pair leaked into U".format(library),
        )
        held = retention.loc[retention["library"].eq(library)]
        _require(
            not bool(subset["pep"].isin(set(held["peptide_design_code"])).any()),
            "{} retention peptide leaked into U".format(library),
        )
        _require(
            not bool(subset["aff"].isin(set(held["affibody_design_code"])).any()),
            "{} retention Affibody leaked into U".format(library),
        )


def membership_sha256(frame):
    digest = hashlib.sha256()
    for value in sorted(frame["pair_uid"].astype(str)):
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mode_string(path):
    return "{:04o}".format(stat.S_IMODE(os.stat(str(path)).st_mode))


def _write_pool(frame, path):
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    os.chmod(str(path), 0o600)


def _write_json(payload, path):
    with open(str(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(str(path), 0o600)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--weak-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.raw_root.is_dir(), "raw root does not exist")
    _require(args.retention_csv.is_file(), "retention identity source does not exist")
    _require(args.weak_labels.is_file(), "existing P/N source does not exist")

    retention = load_retention_identities(args.retention_csv)
    labels = load_existing_labels(args.weak_labels)
    raw_files = discover_raw_round_files(args.raw_root)
    pools = []
    audits = {}
    raw_sources = {}
    for library in LIBRARIES:
        files = raw_files[library]
        round9 = load_count_round(files[9], library)
        round10 = load_count_round(files[10], library)
        pool, audit = build_library_pool(
            round9,
            round10,
            retention,
            library,
            EXPECTED_POOLED_CUTOFF[library],
        )
        pools.append(pool)
        audits[library] = audit
        raw_sources[library] = {
            "R009": {"path": str(files[9].resolve()), "sha256": sha256_file(files[9])},
            "R010": {"path": str(files[10].resolve()), "sha256": sha256_file(files[10])},
        }

    combined = pd.concat(pools, ignore_index=True)
    combined = combined.sort_values(["library", "pair_uid"]).reset_index(drop=True)
    validate_pool_contract(combined, labels, retention)
    label_counts = {
        library: {
            "positive": int(((labels["library"] == library) & (labels["weak_label"] == 1)).sum()),
            "negative": int(((labels["library"] == library) & (labels["weak_label"] == 0)).sum()),
        }
        for library in LIBRARIES
    }

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    pool_path = output_dir / POOL_FILENAME
    _write_pool(combined, pool_path)

    script_path = Path(__file__).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "artifact": {"name": ARTIFACT_NAME, "version": ARTIFACT_VERSION},
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": float(time.time() - started),
        "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "configuration": {
            "unlabeled_definition": (
                "distinct R009/R010 outer-union pairs with zero-filled pooled count "
                "strictly below the library positive cutoff"
            ),
            "operation_order": [
                "outer join R009 and R010",
                "zero-fill absent-round counts and sum",
                "keep pooled count strictly below library cutoff",
                "keep broad declared-library alphabet",
                "exclude pairs sharing either library-local retention peptide or Affibody",
                "verify disjointness from established positive and negative pairs",
            ],
            "positive_cutoffs": dict(EXPECTED_POOLED_CUTOFF),
            "expected_unlabeled_rows": dict(EXPECTED_UNLABELED_ROWS),
            "retention_outcomes_used": False,
            "full_chains_stored": False,
            "compression": "gzip level 6, mtime 0",
        },
        "sources": {
            "raw_root": str(args.raw_root.resolve()),
            "raw_rounds": raw_sources,
            "retention_identities": {
                "path": str(args.retention_csv.resolve()),
                "sha256": sha256_file(args.retention_csv),
                "columns_read": ["library", "peptide_design_code", "affibody_design_code"],
            },
            "established_positive_negative_labels": {
                "path": str(args.weak_labels.resolve()),
                "sha256": sha256_file(args.weak_labels),
                "row_counts": label_counts,
            },
        },
        "libraries": audits,
        "rows": {
            "total": int(len(combined)),
            "by_library": {
                library: int(combined["library"].eq(library).sum()) for library in LIBRARIES
            },
        },
        "membership_sha256": {
            "all": membership_sha256(combined),
            "by_library": {
                library: membership_sha256(combined.loc[combined["library"].eq(library)])
                for library in LIBRARIES
            },
        },
        "schema": {"columns": list(OUTPUT_COLUMNS)},
        "outputs": {POOL_FILENAME: sha256_file(pool_path)},
        "permissions": {
            "directory": _mode_string(output_dir),
            POOL_FILENAME: _mode_string(pool_path),
            "manifest.json": "0600",
        },
        "environment": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    _write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": manifest["rows"],
                "membership_sha256": manifest["membership_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return manifest


if __name__ == "__main__":
    run(parse_args())
