#!/usr/bin/env python
"""Build a sharded frozen-MINT cache for weak-label and retention experiments.

The cache is intentionally a superset of any particular generalization split.
It contains every on-design weak positive and every on-design provider negative
passing a configurable R001 count floor, before filtering on retention peptide
or Affibody identity, plus all retention-grid rows.  Downstream experiments can
therefore apply pair-cold, one-partner-cold, or double-partner-cold policies to
one immutable feature cache.

Three subcommands keep the expensive GPU work restartable and auditable:

``prepare``
    Validate input lineage, reconstruct the provider-confirmed two MINT chains,
    omit the experimental construct linker, and write a deterministic row table.
``extract-shard``
    Embed rows whose zero-based row index is congruent to the shard index modulo
    the shard count.  Each shard is independently runnable on one GPU.
``merge``
    Validate every shard and deterministically restore the prepared row order.

All row-level artifacts are refused unless they are below the repository's
Git-ignored ``private_data`` tree.  Biological codes are always read with
``keep_default_na=False`` so the valid peptide code ``NA`` remains literal.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    OMITTED_LINKER,
    fill_template,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    opaque_id,
    sha256_file,
    validate_private_output_path,
)
from downstream.AffibodyMHC.extract_mint_features import PairCollator
from mint.helpers.extract import MINTWrapper, load_config


SCHEMA_VERSION = "mint-weak-feature-cache-v1"
LIBRARIES = ("LibA", "LibB")
ROWS_FILENAME = "cache_rows.csv"
PREPARE_MANIFEST_FILENAME = "manifest.json"
FINAL_CACHE_FILENAME = "mint_chain_mean_features.npz"
FINAL_MANIFEST_FILENAME = "manifest.json"
FEATURE_NAME = "mint_chain_mean"
FEATURE_DIMENSION = 2560

ROW_COLUMNS = (
    "row_index",
    "cache_uid",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "weak_label",
    "weak_label_source",
    "r001_count",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "shares_retention_peptide",
    "shares_retention_affibody",
    "strict_retention_identity_cold_eligible",
    "upstream_library_local_shares_retention_peptide",
    "upstream_library_local_shares_retention_affibody",
    "upstream_library_local_strict_retention_identity_cold_eligible",
    "measurement_missing",
    "target_retention",
    "target_binder",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)
SEQUENCE_COLUMNS = (
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
)
NPZ_METADATA_COLUMNS = tuple(
    column for column in ROW_COLUMNS if column not in SEQUENCE_COLUMNS
)

WEAK_REQUIRED_COLUMNS = {
    "library",
    "pep",
    "aff",
    "r001_count",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "weak_label",
    "weak_label_source",
    "within_declared_library_alphabet",
    "shares_retention_peptide",
    "shares_retention_affibody",
    "strict_retention_identity_cold_eligible",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
}
RETENTION_REQUIRED_COLUMNS = {
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "measurement_missing",
    "target_retention",
    "target_binder",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_string_csv(path):
    """Read biological codes without interpreting the valid token ``NA``."""
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def canonical_json_sha256(payload):
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def sha256_source_tree(root):
    """Hash Python source paths and contents in stable relative-path order."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = str(path.relative_to(root)).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(str(path), "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def _ensure_private_directory(path, must_be_new):
    path = validate_private_output_path(path, REPO_ROOT)
    if must_be_new:
        _require(not path.exists(), "output directory exists; refusing overwrite")
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=not must_be_new)
    except FileExistsError:
        if must_be_new:
            raise ValueError("output directory exists; refusing overwrite")
        raise
    _require(path.is_dir(), "output path is not a directory")
    os.chmod(str(path), 0o700)
    _require(private_mode(path) == "0700", "output directory is not mode 0700")
    return path


def _atomic_write_json(path, payload):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with open(str(temporary), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "JSON output is not mode 0600")


def _atomic_write_csv(path, frame):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        frame.to_csv(str(temporary), index=False)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "CSV output is not mode 0600")


def _atomic_write_npz(path, arrays, compressed=True):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output already exists")
    try:
        with open(str(temporary), "wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "NPZ output is not mode 0600")


def validate_prepare_lineage(
    weak_labels_path,
    weak_manifest_path,
    retention_sequences_path,
    retention_manifest_path,
    sequence_zip_path,
):
    """Bind both derived tables to one provider archive and retention source."""
    weak_manifest = read_json(weak_manifest_path)
    retention_manifest = read_json(retention_manifest_path)
    hashes = {
        "weak_labels_sha256": sha256_file(weak_labels_path),
        "weak_manifest_sha256": sha256_file(weak_manifest_path),
        "retention_sequences_sha256": sha256_file(retention_sequences_path),
        "retention_manifest_sha256": sha256_file(retention_manifest_path),
        "sequence_zip_sha256": sha256_file(sequence_zip_path),
    }
    _require(
        weak_manifest.get("outputs", {}).get("weak_labels.csv")
        == hashes["weak_labels_sha256"],
        "weak-label manifest/output hash mismatch",
    )
    _require(
        retention_manifest.get("output", {}).get("sha256")
        == hashes["retention_sequences_sha256"],
        "retention manifest/output hash mismatch",
    )
    weak_sources = weak_manifest.get("sources", {})
    retention_sources = retention_manifest.get("sources", {})
    _require(
        weak_sources.get("sequence_zip", {}).get("sha256")
        == hashes["sequence_zip_sha256"],
        "weak-label manifest uses a different sequence ZIP",
    )
    _require(
        retention_sources.get("sequence_zip", {}).get("sha256")
        == hashes["sequence_zip_sha256"],
        "retention manifest uses a different sequence ZIP",
    )
    weak_retention_hash = weak_sources.get("retention_csv", {}).get("sha256")
    retention_source_hash = retention_sources.get("retention_csv", {}).get("sha256")
    _require(
        weak_retention_hash == retention_source_hash,
        "weak labels and retention sequences use different retention matrices",
    )
    weak_deck_hash = weak_sources.get("sequence_zip", {}).get(
        "template_deck_sha256"
    )
    retention_deck_hash = retention_sources.get("sequence_zip", {}).get(
        "deck_sha256"
    )
    _require(
        weak_deck_hash == retention_deck_hash,
        "weak labels and retention sequences use different template decks",
    )
    hashes["retention_matrix_sha256"] = weak_retention_hash
    hashes["template_deck_sha256"] = weak_deck_hash
    return hashes


def _validate_code(library, peptide, affibody):
    _require(library in LIBRARY_SPECS, "unexpected library {}".format(library))
    spec = LIBRARY_SPECS[library]
    _require(
        len(peptide) == int(spec["pep_length"]),
        "{} peptide code length mismatch".format(library),
    )
    _require(
        len(affibody) == int(spec["aff_length"]),
        "{} Affibody code length mismatch".format(library),
    )
    alphabet = set(AA_ALPHABET)
    _require(set(peptide).issubset(alphabet), "noncanonical peptide code")
    _require(set(affibody).issubset(alphabet), "noncanonical Affibody code")


def reconstruct_chains(template, library, peptide, affibody):
    """Fill provider placeholders and return only the two confirmed MINT chains."""
    _validate_code(library, peptide, affibody)
    full = fill_template(template, peptide + affibody)
    omitted = full[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)]
    _require(omitted == OMITTED_LINKER, "provider construct linker mismatch")
    chain1 = full[:CHAIN1_LENGTH]
    chain2 = full[-CHAIN2_LENGTH:]
    _require(len(chain1) == CHAIN1_LENGTH, "chain-1 length mismatch")
    _require(len(chain2) == CHAIN2_LENGTH, "chain-2 length mismatch")
    _require(OMITTED_LINKER not in chain1, "experimental linker leaked into chain 1")
    _require(OMITTED_LINKER not in chain2, "experimental linker leaked into chain 2")
    return chain1, chain2


def select_weak_candidates(frame, min_negative_r001_count):
    _require(WEAK_REQUIRED_COLUMNS.issubset(frame.columns), "weak-label schema mismatch")
    _require(int(min_negative_r001_count) >= 1, "negative R001 count floor must be positive")
    output = frame.copy()
    for column in (
        "r001_count",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "weak_label",
        "within_declared_library_alphabet",
        "shares_retention_peptide",
        "shares_retention_affibody",
        "strict_retention_identity_cold_eligible",
    ):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(np.int64)
    _require(set(output["library"]) == set(LIBRARIES), "weak labels lack a library")
    _require(set(output["weak_label"]) == {0, 1}, "unexpected weak-label values")
    _require(not bool(output["pair_uid"].duplicated().any()), "duplicate weak pair UID")
    eligible = output["within_declared_library_alphabet"].eq(1) & (
        output["weak_label"].eq(1)
        | (
            output["weak_label"].eq(0)
            & output["r001_count"].ge(int(min_negative_r001_count))
        )
    )
    selected = output.loc[eligible].copy()
    _require(not selected.empty, "weak candidate filter returned no rows")
    for library in LIBRARIES:
        subset = selected.loc[selected["library"].eq(library)]
        _require(set(subset["weak_label"]) == {0, 1}, "{} candidate lacks a class".format(library))
    return selected.sort_values(["library", "pair_uid"]).reset_index(drop=True)


def assemble_candidate_rows(weak, retention, templates, min_negative_r001_count=3):
    """Create the deterministic private row table used by every GPU shard."""
    selected = select_weak_candidates(weak, min_negative_r001_count)
    _require(
        RETENTION_REQUIRED_COLUMNS.issubset(retention.columns),
        "retention sequence schema mismatch",
    )
    _require(set(retention["library"]) == set(LIBRARIES), "retention lacks a library")
    _require(not bool(retention["pair_uid"].duplicated().any()), "duplicate retention pair UID")
    _require(
        set(selected["pair_uid"]).isdisjoint(set(retention["pair_uid"])),
        "an exact retention pair remains in weak candidates",
    )
    weak_keys = set(
        zip(selected["library"], selected["pep"], selected["aff"])
    )
    retention_keys = set(
        zip(
            retention["library"],
            retention["peptide_design_code"],
            retention["affibody_design_code"],
        )
    )
    _require(
        weak_keys.isdisjoint(retention_keys),
        "an exact biological retention pair remains in weak candidates",
    )
    _require(set(templates) == set(LIBRARIES), "provider templates lack a library")

    records = []
    for row in selected.itertuples(index=False):
        peptide = str(row.pep)
        affibody = str(row.aff)
        chain1, chain2 = reconstruct_chains(
            templates[row.library], row.library, peptide, affibody
        )
        records.append(
            {
                "cache_uid": opaque_id(SCHEMA_VERSION, "weak", row.pair_uid),
                "source_kind": "weak",
                "library": row.library,
                "pair_uid": row.pair_uid,
                "peptide_uid": row.peptide_uid,
                "affibody_uid": row.affibody_uid,
                "peptide_design_code": peptide,
                "affibody_design_code": affibody,
                "weak_label": str(int(row.weak_label)),
                "weak_label_source": str(row.weak_label_source),
                "r001_count": str(int(row.r001_count)),
                "r009_count": str(int(row.r009_count)),
                "r010_count": str(int(row.r010_count)),
                "pooled_r009_r010_count": str(int(row.pooled_r009_r010_count)),
                "shares_retention_peptide": "",
                "shares_retention_affibody": "",
                "strict_retention_identity_cold_eligible": "",
                "upstream_library_local_shares_retention_peptide": str(
                    int(row.shares_retention_peptide)
                ),
                "upstream_library_local_shares_retention_affibody": str(
                    int(row.shares_retention_affibody)
                ),
                "upstream_library_local_strict_retention_identity_cold_eligible": str(
                    int(row.strict_retention_identity_cold_eligible)
                ),
                "measurement_missing": "",
                "target_retention": "",
                "target_binder": "",
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "chain1_sha256": sha256_text(chain1),
                "chain2_sha256": sha256_text(chain2),
                "sequence_pair_sha256": sha256_text(chain1 + "|" + chain2),
            }
        )

    for row in retention.itertuples(index=False):
        peptide = str(row.peptide_design_code)
        affibody = str(row.affibody_design_code)
        chain1, chain2 = reconstruct_chains(
            templates[row.library], row.library, peptide, affibody
        )
        _require(
            chain1 == row.chain1_smart_hla_linker_peptide_sequence,
            "retention chain 1 does not match provider reconstruction",
        )
        _require(
            chain2 == row.chain2_affibody_sequence,
            "retention chain 2 does not match provider reconstruction",
        )
        records.append(
            {
                "cache_uid": opaque_id(SCHEMA_VERSION, "retention", row.pair_uid),
                "source_kind": "retention",
                "library": row.library,
                "pair_uid": row.pair_uid,
                "peptide_uid": row.peptide_uid,
                "affibody_uid": row.affibody_uid,
                "peptide_design_code": peptide,
                "affibody_design_code": affibody,
                "weak_label": "",
                "weak_label_source": "",
                "r001_count": "",
                "r009_count": "",
                "r010_count": "",
                "pooled_r009_r010_count": "",
                "shares_retention_peptide": "",
                "shares_retention_affibody": "",
                "strict_retention_identity_cold_eligible": "",
                "upstream_library_local_shares_retention_peptide": "",
                "upstream_library_local_shares_retention_affibody": "",
                "upstream_library_local_strict_retention_identity_cold_eligible": "",
                "measurement_missing": str(row.measurement_missing),
                "target_retention": str(row.target_retention),
                "target_binder": str(row.target_binder),
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "chain1_sha256": sha256_text(chain1),
                "chain2_sha256": sha256_text(chain2),
                "sequence_pair_sha256": sha256_text(chain1 + "|" + chain2),
            }
        )

    output = pd.DataFrame(records)
    weak_mask = output["source_kind"].eq("weak")
    retention_mask = output["source_kind"].eq("retention")
    _require(
        set(output.loc[weak_mask, "sequence_pair_sha256"]).isdisjoint(
            set(output.loc[retention_mask, "sequence_pair_sha256"])
        ),
        "a global sequence-identical retention pair remains in weak candidates",
    )
    retention_chain1 = set(output.loc[retention_mask, "chain1_sha256"])
    retention_chain2 = set(output.loc[retention_mask, "chain2_sha256"])
    global_peptide_share = output.loc[weak_mask, "chain1_sha256"].isin(
        retention_chain1
    )
    global_affibody_share = output.loc[weak_mask, "chain2_sha256"].isin(
        retention_chain2
    )
    output.loc[weak_mask, "shares_retention_peptide"] = global_peptide_share.astype(
        int
    ).astype(str)
    output.loc[weak_mask, "shares_retention_affibody"] = global_affibody_share.astype(
        int
    ).astype(str)
    output.loc[weak_mask, "strict_retention_identity_cold_eligible"] = (
        ~global_peptide_share & ~global_affibody_share
    ).astype(int).astype(str)
    output["_source_order"] = output["source_kind"].map({"weak": 0, "retention": 1})
    output = output.sort_values(
        ["_source_order", "library", "pair_uid"], kind="mergesort"
    ).drop(columns=["_source_order"])
    output.insert(0, "row_index", np.arange(len(output), dtype=np.int64))
    output = output.loc[:, list(ROW_COLUMNS)].reset_index(drop=True)
    _require(output["cache_uid"].nunique() == len(output), "duplicate cache UID")
    _require(
        bool(output.loc[output["source_kind"].eq("weak"), "weak_label"].isin(["0", "1"]).all()),
        "weak row lacks weak label",
    )
    _require(
        bool(output.loc[output["source_kind"].eq("retention"), "weak_label"].eq("").all()),
        "retention row unexpectedly has a weak label",
    )
    return output


def validate_prepared_rows(frame):
    _require(tuple(frame.columns) == ROW_COLUMNS, "prepared row-table schema mismatch")
    row_index = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(dtype=np.int64)
    _require(
        np.array_equal(row_index, np.arange(len(frame), dtype=np.int64)),
        "prepared row indices are not contiguous",
    )
    _require(frame["cache_uid"].nunique() == len(frame), "duplicate prepared cache UID")
    _require(frame["pair_uid"].map(bool).all(), "blank prepared pair UID")
    _require(set(frame["source_kind"]) == {"weak", "retention"}, "unexpected source kind")
    _require(set(frame["library"]) == set(LIBRARIES), "prepared rows lack a library")
    for sequence_column, length in zip(SEQUENCE_COLUMNS, (CHAIN1_LENGTH, CHAIN2_LENGTH)):
        _require(
            bool(frame[sequence_column].map(len).eq(length).all()),
            "prepared sequence length mismatch",
        )
        _require(
            bool(
                frame[sequence_column]
                .map(lambda value: set(value).issubset(set(AA_ALPHABET)))
                .all()
            ),
            "prepared sequence contains a noncanonical residue",
        )
        _require(
            not bool(frame[sequence_column].str.contains(OMITTED_LINKER, regex=False).any()),
            "experimental construct linker appears in a MINT chain",
        )
    expected_hash = [
        sha256_text(chain1 + "|" + chain2)
        for chain1, chain2 in zip(frame[SEQUENCE_COLUMNS[0]], frame[SEQUENCE_COLUMNS[1]])
    ]
    _require(
        expected_hash == frame["sequence_pair_sha256"].tolist(),
        "prepared sequence-pair hash mismatch",
    )
    _require(
        [sha256_text(value) for value in frame[SEQUENCE_COLUMNS[0]]]
        == frame["chain1_sha256"].tolist(),
        "prepared chain-1 hash mismatch",
    )
    _require(
        [sha256_text(value) for value in frame[SEQUENCE_COLUMNS[1]]]
        == frame["chain2_sha256"].tolist(),
        "prepared chain-2 hash mismatch",
    )
    for row in frame.itertuples(index=False):
        _validate_code(row.library, row.peptide_design_code, row.affibody_design_code)
    weak = frame.loc[frame["source_kind"].eq("weak")]
    retention = frame.loc[frame["source_kind"].eq("retention")]
    _require(bool(weak["weak_label"].isin(["0", "1"]).all()), "invalid weak label")
    _require(bool(retention["weak_label"].eq("").all()), "retention has weak label")
    return row_index


def candidate_summary(frame):
    summary = {
        "total": int(len(frame)),
        "weak": int(frame["source_kind"].eq("weak").sum()),
        "retention": int(frame["source_kind"].eq("retention").sum()),
        "unique_chain1_sequences": int(frame["chain1_sha256"].nunique()),
        "unique_chain2_sequences": int(frame["chain2_sha256"].nunique()),
        "unique_sequence_pairs": int(frame["sequence_pair_sha256"].nunique()),
    }
    libraries = {}
    for library in LIBRARIES:
        subset = frame.loc[frame["library"].eq(library)]
        weak = subset.loc[subset["source_kind"].eq("weak")]
        retention = subset.loc[subset["source_kind"].eq("retention")]
        libraries[library] = {
            "weak_rows": int(len(weak)),
            "weak_positive": int(weak["weak_label"].eq("1").sum()),
            "weak_negative": int(weak["weak_label"].eq("0").sum()),
            "weak_shares_retention_peptide": int(
                weak["shares_retention_peptide"].eq("1").sum()
            ),
            "weak_shares_retention_affibody": int(
                weak["shares_retention_affibody"].eq("1").sum()
            ),
            "weak_strict_double_cold": int(
                weak["strict_retention_identity_cold_eligible"].eq("1").sum()
            ),
            "retention_rows": int(len(retention)),
            "retention_measured": int(retention["measurement_missing"].eq("0").sum()),
            "unique_weak_peptides": int(weak["peptide_uid"].nunique()),
            "unique_weak_affibodies": int(weak["affibody_uid"].nunique()),
        }
    summary["libraries"] = libraries
    return summary


def run_prepare(args):
    started = time.time()
    _require(args.min_negative_r001_count >= 1, "negative count floor must be positive")
    weak_labels_path = args.weak_label_dir / "weak_labels.csv"
    weak_manifest_path = args.weak_label_dir / "manifest.json"
    retention_manifest_path = args.retention_manifest
    if retention_manifest_path is None:
        retention_manifest_path = args.retention_sequences.with_suffix(".manifest.json")
    for path in (
        weak_labels_path,
        weak_manifest_path,
        args.retention_sequences,
        retention_manifest_path,
        args.sequence_zip,
    ):
        _require(path.is_file(), "missing input {}".format(path))

    lineage = validate_prepare_lineage(
        weak_labels_path,
        weak_manifest_path,
        args.retention_sequences,
        retention_manifest_path,
        args.sequence_zip,
    )
    templates, template_member, template_deck_hash = load_provider_templates(
        args.sequence_zip
    )
    _require(
        template_deck_hash == lineage["template_deck_sha256"],
        "loaded template deck hash disagrees with input manifests",
    )
    weak = read_string_csv(weak_labels_path)
    retention = read_string_csv(args.retention_sequences)
    rows = assemble_candidate_rows(
        weak,
        retention,
        templates,
        min_negative_r001_count=args.min_negative_r001_count,
    )
    validate_prepared_rows(rows.astype(str))

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=True)
    rows_path = output_dir / ROWS_FILENAME
    manifest_path = output_dir / PREPARE_MANIFEST_FILENAME
    _atomic_write_csv(rows_path, rows)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "sources": {
            "weak_labels": {
                "path": str(weak_labels_path.resolve()),
                "sha256": lineage["weak_labels_sha256"],
            },
            "weak_manifest": {
                "path": str(weak_manifest_path.resolve()),
                "sha256": lineage["weak_manifest_sha256"],
            },
            "retention_sequences": {
                "path": str(args.retention_sequences.resolve()),
                "sha256": lineage["retention_sequences_sha256"],
            },
            "retention_manifest": {
                "path": str(retention_manifest_path.resolve()),
                "sha256": lineage["retention_manifest_sha256"],
            },
            "sequence_zip": {
                "path": str(args.sequence_zip.resolve()),
                "sha256": lineage["sequence_zip_sha256"],
                "template_member": template_member,
                "template_deck_sha256": template_deck_hash,
            },
            "retention_matrix_sha256": lineage["retention_matrix_sha256"],
        },
        "selection": {
            "positive": "weak_label == 1",
            "negative": "weak_label == 0 and r001_count >= {}".format(
                int(args.min_negative_r001_count)
            ),
            "on_design_required": True,
            "retention_exact_pairs": "already excluded upstream and revalidated disjoint",
            "retention_identity_filter": "not applied; global full-chain SHA256 overlap flags preserved for pair/one/double-cold experiments",
            "global_identity_policy": {
                "peptide_partner": "exact chain1_sha256 across both libraries",
                "affibody_partner": "exact chain2_sha256 across both libraries",
                "upstream_flags": "retained separately and explicitly labeled library-local",
            },
            "min_negative_r001_count": int(args.min_negative_r001_count),
        },
        "provider_model_input": {
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
            "tcr_present": False,
            "separate_b2m_supplied": False,
        },
        "templates": {
            library: {
                "sha256": sha256_text(template),
                "length": int(len(template)),
            }
            for library, template in sorted(templates.items())
        },
        "rows": candidate_summary(rows.astype(str)),
        "code": {
            "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
            "mint_source_tree_sha256": sha256_source_tree(REPO_ROOT / "mint"),
        },
        "output": {
            "rows_csv": {
                "path": str(rows_path.resolve()),
                "sha256": sha256_file(rows_path),
                "mode": private_mode(rows_path),
                "columns": list(ROW_COLUMNS),
            },
            "directory_mode": private_mode(output_dir),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "output": str(rows_path),
                "rows": int(len(rows)),
                "weak": int(rows["source_kind"].eq("weak").sum()),
                "retention": int(rows["source_kind"].eq("retention").sum()),
            },
            sort_keys=True,
        )
    )


def shard_filename(shard_index, shard_count):
    return "features-shard-{:03d}-of-{:03d}.npz".format(
        int(shard_index), int(shard_count)
    )


def shard_manifest_filename(shard_index, shard_count):
    return "features-shard-{:03d}-of-{:03d}.manifest.json".format(
        int(shard_index), int(shard_count)
    )


def select_shard_rows(frame, shard_index, shard_count):
    _require(int(shard_count) >= 1, "shard count must be positive")
    _require(0 <= int(shard_index) < int(shard_count), "shard index is out of range")
    indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(dtype=np.int64)
    selected = np.mod(indices, int(shard_count)) == int(shard_index)
    output = frame.loc[selected].copy().reset_index(drop=True)
    _require(not output.empty, "shard has no rows")
    return output


class CachePairDataset(Dataset):
    def __init__(self, frame):
        self.chain1 = frame[SEQUENCE_COLUMNS[0]].tolist()
        self.chain2 = frame[SEQUENCE_COLUMNS[1]].tolist()
        self.library = frame["library"].tolist()
        self.cache_uid = frame["cache_uid"].tolist()

    def __len__(self):
        return len(self.cache_uid)

    def __getitem__(self, index):
        return (
            self.chain1[index],
            self.chain2[index],
            self.library[index],
            self.cache_uid[index],
        )


def metadata_arrays(frame):
    arrays = {}
    for column in NPZ_METADATA_COLUMNS:
        if column == "row_index":
            arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(
                dtype=np.int64
            )
        else:
            arrays[column] = frame[column].to_numpy(dtype=str)
    return arrays


def load_prepared_input(input_dir):
    rows_path = input_dir / ROWS_FILENAME
    manifest_path = input_dir / PREPARE_MANIFEST_FILENAME
    _require(rows_path.is_file(), "prepared row table is missing")
    _require(manifest_path.is_file(), "prepared manifest is missing")
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "prepared schema mismatch")
    _require(manifest.get("stage") == "prepare", "input manifest is not prepare stage")
    _require(
        manifest.get("output", {}).get("rows_csv", {}).get("sha256")
        == sha256_file(rows_path),
        "prepared row-table hash mismatch",
    )
    frame = read_string_csv(rows_path)
    validate_prepared_rows(frame)
    return frame, manifest, rows_path, manifest_path


def feature_contract(
    rows_sha256,
    prepare_manifest_sha256,
    checkpoint_sha256,
    config_sha256,
    mint_source_tree_sha256,
    script_sha256,
    pair_collator_source_sha256,
    shard_count,
    batch_size,
    gpu_name,
    gpu_capability,
):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows_sha256": rows_sha256,
        "prepare_manifest_sha256": prepare_manifest_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "mint_source_tree_sha256": mint_source_tree_sha256,
        "script_sha256": script_sha256,
        "pair_collator_source_sha256": pair_collator_source_sha256,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "batch_size": int(batch_size),
            "gpu_name": str(gpu_name),
            "gpu_capability": [int(value) for value in gpu_capability],
        },
        "model": "MINT use_multimer=True sep_chains=True",
        "representation": FEATURE_NAME,
        "feature_dimension": FEATURE_DIMENSION,
        "shard_rule": "row_index modulo shard_count",
        "shard_count": int(shard_count),
    }
    return canonical_json_sha256(payload), payload


def run_extract_shard(args):
    started = time.time()
    _require(args.batch_size >= 1, "batch size must be positive")
    _require(args.shard_count >= 1, "shard count must be positive")
    _require(0 <= args.shard_index < args.shard_count, "shard index is out of range")
    _require(args.checkpoint.is_file(), "MINT checkpoint does not exist")
    _require(args.config.is_file(), "MINT config does not exist")
    frame, _, rows_path, prepare_manifest_path = load_prepared_input(args.input_dir)
    shard = select_shard_rows(frame, args.shard_index, args.shard_count)
    device = torch.device(args.device)
    _require(device.type == "cuda", "extract-shard requires a CUDA device")
    _require(torch.cuda.is_available(), "CUDA is not available")
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    gpu_capability = torch.cuda.get_device_capability(device)
    # Hash every input before loading the model.  The same inputs are rehashed
    # after inference so a long-running shard can never attest to files that
    # changed after the weights or code were actually consumed.
    checkpoint_hash = sha256_file(args.checkpoint)
    config_hash = sha256_file(args.config)
    rows_hash = sha256_file(rows_path)
    prepare_manifest_hash = sha256_file(prepare_manifest_path)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    mint_tree_hash = sha256_source_tree(REPO_ROOT / "mint")
    pair_collator_path = REPO_ROOT / "downstream" / "AffibodyMHC" / "extract_mint_features.py"
    pair_collator_hash = sha256_file(pair_collator_path)
    contract_hash, contract_payload = feature_contract(
        rows_hash,
        prepare_manifest_hash,
        checkpoint_hash,
        config_hash,
        mint_tree_hash,
        script_hash,
        pair_collator_hash,
        args.shard_count,
        args.batch_size,
        gpu_name,
        gpu_capability,
    )

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=False)
    output_path = output_dir / shard_filename(args.shard_index, args.shard_count)
    manifest_path = output_dir / shard_manifest_filename(
        args.shard_index, args.shard_count
    )
    _require(not output_path.exists(), "shard output exists; refusing overwrite")
    _require(not manifest_path.exists(), "shard manifest exists; refusing overwrite")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    loader = DataLoader(
        CachePairDataset(shard),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    wrapper = MINTWrapper(
        load_config(str(args.config)),
        str(args.checkpoint),
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()
    observed_uids = []
    blocks = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, _, cache_uids) in enumerate(loader):
            values = wrapper(chains.to(device), chain_ids.to(device))
            _require(
                tuple(values.shape[1:]) == (FEATURE_DIMENSION,),
                "unexpected MINT feature dimension",
            )
            blocks.append(values.float().cpu().numpy())
            observed_uids.extend(cache_uids)
            if (step + 1) % 50 == 0 or step + 1 == len(loader):
                print(
                    "shard {}/{}: embedded {}/{} batches".format(
                        args.shard_index,
                        args.shard_count,
                        step + 1,
                        len(loader),
                    ),
                    flush=True,
                )
    torch.cuda.synchronize(device)
    _require(observed_uids == shard["cache_uid"].tolist(), "embedding row order changed")
    features = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    _require(
        features.shape == (len(shard), FEATURE_DIMENSION),
        "shard feature shape mismatch",
    )
    _require(bool(np.isfinite(features).all()), "shard contains non-finite features")
    arrays = metadata_arrays(shard)
    arrays[FEATURE_NAME] = features
    _require(rows_hash == sha256_file(rows_path), "prepared rows changed during extraction")
    _require(
        prepare_manifest_hash == sha256_file(prepare_manifest_path),
        "prepare manifest changed during extraction",
    )
    _require(
        checkpoint_hash == sha256_file(args.checkpoint),
        "checkpoint changed during extraction",
    )
    _require(config_hash == sha256_file(args.config), "config changed during extraction")
    _require(script_hash == sha256_file(script_path), "extractor changed during extraction")
    _require(
        pair_collator_hash == sha256_file(pair_collator_path),
        "pair collator source changed during extraction",
    )
    _require(
        mint_tree_hash == sha256_source_tree(REPO_ROOT / "mint"),
        "MINT source tree changed during extraction",
    )
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "extract_shard",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "hostname": socket.gethostname(),
        "input": {
            "rows_csv": {"path": str(rows_path.resolve()), "sha256": rows_hash},
            "prepare_manifest": {
                "path": str(prepare_manifest_path.resolve()),
                "sha256": prepare_manifest_hash,
            },
        },
        "model": {
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": checkpoint_hash,
            },
            "config": {
                "path": str(args.config.resolve()),
                "sha256": config_hash,
            },
            "mint_source_tree_sha256": mint_tree_hash,
            "script_sha256": script_hash,
            "pair_collator_source": {
                "path": str(pair_collator_path.resolve()),
                "sha256": pair_collator_hash,
            },
            "use_multimer": True,
            "sep_chains": True,
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
            "layer": 33,
            "pooling": "separate residue mean for chain 1 and chain 2, concatenated",
            "feature_name": FEATURE_NAME,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_dtype": "float32",
        },
        "contract": {"sha256": contract_hash, "payload": contract_payload},
        "sharding": {
            "rule": "row_index modulo shard_count equals shard_index",
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "rows": int(len(shard)),
            "first_row_index": int(shard["row_index"].iloc[0]),
            "last_row_index": int(shard["row_index"].iloc[-1]),
            "row_index_sha256": hashlib.sha256(
                arrays["row_index"].astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "cache_uid_sha256": sha256_text("\n".join(observed_uids)),
        },
        "runtime": {
            "device_argument": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu_name,
            "gpu_capability": [int(value) for value in gpu_capability],
            "batch_size": int(args.batch_size),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
            "feature_shape": list(features.shape),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "stage": "extract_shard",
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
                "rows": int(len(shard)),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


def _load_and_validate_shard(
    shard_path,
    manifest_path,
    frame,
    shard_index,
    shard_count,
    expected_contract=None,
):
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "shard schema mismatch")
    _require(manifest.get("stage") == "extract_shard", "not an extraction manifest")
    sharding = manifest.get("sharding", {})
    _require(int(sharding.get("shard_index", -1)) == shard_index, "shard-index mismatch")
    _require(int(sharding.get("shard_count", -1)) == shard_count, "shard-count mismatch")
    contract = manifest.get("contract", {}).get("sha256")
    _require(bool(contract), "shard lacks a feature contract")
    contract_payload = manifest.get("contract", {}).get("payload")
    _require(isinstance(contract_payload, dict), "shard lacks a contract payload")
    _require(
        canonical_json_sha256(contract_payload) == contract,
        "shard contract payload/hash mismatch",
    )
    _require(
        int(contract_payload.get("shard_count", -1)) == int(shard_count),
        "shard contract count mismatch",
    )
    _require(
        contract_payload.get("representation") == FEATURE_NAME
        and int(contract_payload.get("feature_dimension", -1)) == FEATURE_DIMENSION,
        "shard contract feature mismatch",
    )
    model = manifest.get("model", {})
    contract_model_fields = {
        "checkpoint_sha256": model.get("checkpoint", {}).get("sha256"),
        "config_sha256": model.get("config", {}).get("sha256"),
        "mint_source_tree_sha256": model.get("mint_source_tree_sha256"),
        "script_sha256": model.get("script_sha256"),
        "pair_collator_source_sha256": model.get("pair_collator_source", {}).get(
            "sha256"
        ),
    }
    for name, value in contract_model_fields.items():
        _require(
            contract_payload.get(name) == value,
            "shard model/contract mismatch for {}".format(name),
        )
    contract_runtime = contract_payload.get("runtime", {})
    manifest_runtime = manifest.get("runtime", {})
    for name in ("torch", "torch_cuda", "numpy", "batch_size", "gpu_name"):
        _require(
            contract_runtime.get(name) == manifest_runtime.get(name),
            "shard runtime/contract mismatch for {}".format(name),
        )
    _require(
        list(contract_runtime.get("gpu_capability", []))
        == list(manifest_runtime.get("gpu_capability", [])),
        "shard runtime/contract mismatch for gpu_capability",
    )
    if expected_contract is not None:
        _require(contract == expected_contract, "shards use different feature contracts")
    _require(
        manifest.get("output", {}).get("sha256") == sha256_file(shard_path),
        "shard output hash mismatch",
    )
    expected = select_shard_rows(frame, shard_index, shard_count)
    with np.load(str(shard_path), allow_pickle=False) as archive:
        expected_keys = set(NPZ_METADATA_COLUMNS) | {FEATURE_NAME}
        _require(set(archive.files) == expected_keys, "shard NPZ schema mismatch")
        row_index = np.asarray(archive["row_index"], dtype=np.int64)
        expected_index = pd.to_numeric(expected["row_index"], errors="raise").to_numpy(
            dtype=np.int64
        )
        _require(np.array_equal(row_index, expected_index), "shard row assignment mismatch")
        _require(
            hashlib.sha256(row_index.astype("<i8", copy=False).tobytes()).hexdigest()
            == sharding.get("row_index_sha256"),
            "shard row-index hash mismatch",
        )
        for column in NPZ_METADATA_COLUMNS:
            if column == "row_index":
                continue
            observed = np.asarray(archive[column]).astype(str)
            wanted = expected[column].to_numpy(dtype=str)
            _require(
                np.array_equal(observed, wanted),
                "shard metadata mismatch in {}".format(column),
            )
        feature = np.asarray(archive[FEATURE_NAME], dtype=np.float32)
        _require(
            feature.shape == (len(expected), FEATURE_DIMENSION),
            "shard feature shape mismatch",
        )
        _require(bool(np.isfinite(feature).all()), "shard has non-finite features")
        feature = feature.copy()
    return manifest, expected_index, feature, contract


def run_merge(args):
    started = time.time()
    _require(args.shard_count >= 1, "shard count must be positive")
    frame, prepare_manifest, rows_path, prepare_manifest_path = load_prepared_input(
        args.input_dir
    )
    _require(args.shards_dir.is_dir(), "shard directory does not exist")
    rows_hash = sha256_file(rows_path)
    prepare_manifest_hash = sha256_file(prepare_manifest_path)
    features = np.empty((len(frame), FEATURE_DIMENSION), dtype=np.float32)
    covered = np.zeros(len(frame), dtype=bool)
    shard_records = []
    expected_contract = None
    first_manifest = None
    for shard_index in range(args.shard_count):
        shard_path = args.shards_dir / shard_filename(shard_index, args.shard_count)
        manifest_path = args.shards_dir / shard_manifest_filename(
            shard_index, args.shard_count
        )
        _require(shard_path.is_file(), "missing shard {}".format(shard_index))
        _require(manifest_path.is_file(), "missing shard manifest {}".format(shard_index))
        manifest, indices, values, contract = _load_and_validate_shard(
            shard_path,
            manifest_path,
            frame,
            shard_index,
            args.shard_count,
            expected_contract=expected_contract,
        )
        if expected_contract is None:
            expected_contract = contract
            first_manifest = manifest
        _require(
            manifest.get("input", {}).get("rows_csv", {}).get("sha256") == rows_hash,
            "shard was extracted from a different row table",
        )
        _require(
            manifest.get("input", {}).get("prepare_manifest", {}).get("sha256")
            == prepare_manifest_hash,
            "shard was extracted from a different prepare manifest",
        )
        _require(not bool(covered[indices].any()), "duplicate row coverage across shards")
        features[indices] = values
        covered[indices] = True
        shard_records.append(
            {
                "shard_index": int(shard_index),
                "rows": int(len(indices)),
                "npz_path": str(shard_path.resolve()),
                "npz_sha256": sha256_file(shard_path),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "hostname": manifest.get("hostname"),
                "gpu_name": manifest.get("runtime", {}).get("gpu_name"),
            }
        )
    _require(bool(covered.all()), "shards do not cover every prepared row")
    _require(bool(np.isfinite(features).all()), "merged feature array is non-finite")

    output_dir = _ensure_private_directory(args.output_dir, must_be_new=True)
    output_path = output_dir / FINAL_CACHE_FILENAME
    manifest_path = output_dir / FINAL_MANIFEST_FILENAME
    arrays = metadata_arrays(frame)
    arrays[FEATURE_NAME] = features
    _atomic_write_npz(output_path, arrays, compressed=not args.uncompressed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "merge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "input": {
            "rows_csv": {"path": str(rows_path.resolve()), "sha256": rows_hash},
            "prepare_manifest": {
                "path": str(prepare_manifest_path.resolve()),
                "sha256": prepare_manifest_hash,
            },
            "prepare_summary": prepare_manifest.get("rows"),
            "shards": shard_records,
        },
        "contract": {
            "sha256": expected_contract,
            "payload": first_manifest.get("contract", {}).get("payload"),
        },
        "model": first_manifest.get("model"),
        "rows": candidate_summary(frame),
        "features": {
            "name": FEATURE_NAME,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "row_order": "ascending row_index from prepared cache_rows.csv",
        },
        "merge_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "mode": private_mode(output_path),
            "directory_mode": private_mode(output_dir),
            "compressed": bool(not args.uncompressed),
            "keys": sorted(arrays),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "stage": "merge",
                "rows": int(len(frame)),
                "feature_shape": list(features.shape),
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="build the deterministic private row table")
    prepare.add_argument("--weak-label-dir", required=True, type=Path)
    prepare.add_argument("--retention-sequences", required=True, type=Path)
    prepare.add_argument("--retention-manifest", type=Path)
    prepare.add_argument("--sequence-zip", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--min-negative-r001-count", type=int, default=3)

    extract = subparsers.add_parser("extract-shard", help="extract one deterministic GPU shard")
    extract.add_argument("--input-dir", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--checkpoint", required=True, type=Path)
    extract.add_argument("--config", required=True, type=Path)
    extract.add_argument("--shard-index", required=True, type=int)
    extract.add_argument("--shard-count", required=True, type=int)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--uncompressed", action="store_true")

    merge = subparsers.add_parser("merge", help="validate and merge every GPU shard")
    merge.add_argument("--input-dir", required=True, type=Path)
    merge.add_argument("--shards-dir", required=True, type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.add_argument("--shard-count", required=True, type=int)
    merge.add_argument("--uncompressed", action="store_true")

    args = parser.parse_args(argv)
    _require(args.command is not None, "a subcommand is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "extract-shard":
        run_extract_shard(args)
    elif args.command == "merge":
        run_merge(args)
    else:
        raise ValueError("unknown command {}".format(args.command))


if __name__ == "__main__":
    main()
