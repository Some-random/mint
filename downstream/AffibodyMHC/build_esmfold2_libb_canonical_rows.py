#!/usr/bin/env python
"""Build the canonical, label-isolated LibB ESMFold2 row contract.

This is a CPU-only preparation step.  It reads the already audited MINT cache
row table, selects the published library-local strict LibB training set and
the measured LibB evaluation rows, and writes three physically separate
artifact groups:

* ``rows.json`` plus ``manifest.json``: label-free sequences for extraction;
* ``training_labels.csv`` plus ``manifest.json``: selection-derived labels; and
* ``evaluation_labels.csv`` plus ``manifest.json``: sealed direct outcomes.

The extractor artifact group is recursively checked for outcome and
selection-derived material.  The folding process therefore does not need
access to either label table.  This script never imports or runs a folding
model.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "esmfold2-libb-canonical-rows-v1"
AUDIT_SCHEMA_VERSION = "esmfold2-libb-canonical-rows-audit-v1"
TRAINING_LABEL_SCHEMA_VERSION = "esmfold2-libb-training-labels-v1"
EVALUATION_LABEL_SCHEMA_VERSION = "esmfold2-libb-evaluation-labels-v1"
SOURCE_SCHEMA_VERSION = "mint-weak-feature-cache-v1"

LIBRARY = "LibB"
ROWS_FILENAME = "rows.json"
MANIFEST_FILENAME = "manifest.json"
TRAINING_LABELS_FILENAME = "training_labels.csv"
EVALUATION_LABELS_FILENAME = "evaluation_labels.csv"

CHAIN1_LENGTH = 270
CHAIN2_LENGTH = 58
PEPTIDE_LENGTH = 9
PEPTIDE_CODE_POSITIONS_1_BASED = (4, 5)
PEPTIDE_CODE_CHAIN1_POSITIONS_1_BASED = (265, 266)
AFFIBODY_CODE_POSITIONS_1_BASED = (6, 10, 13, 14, 17)
FULL_CONSTRUCT_AFFIBODY_POSITIONS_1_BASED = (290, 294, 297, 298, 301)
OMITTED_LINKER = "GGSLEVLFQGPGSG"
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
BINDER_THRESHOLD = 75.0

EXPECTED = {
    "source_total": 69945,
    "source_libb_candidate": 38710,
    "source_libb_matrix": 120,
    "train": 30648,
    "train_positive": 23725,
    "train_negative": 6923,
    "eval": 119,
    "eval_positive": 60,
    "eval_negative": 59,
    "total": 30767,
    "train_unique_chain1": 214,
    "train_unique_chain2": 23081,
    "eval_unique_chain1": 12,
    "eval_unique_chain2": 10,
}

EXTRACTOR_COLUMNS = (
    "row_index",
    "row_id",
    "split",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
)
TRAINING_LABEL_COLUMNS = (
    "row_id",
    "weak_label",
    "peptide_id",
    "affibody_id",
    "chain1_sha256",
    "chain2_sha256",
)
EVALUATION_LABEL_COLUMNS = (
    "row_id",
    "peptide_design_code",
    "affibody_design_code",
    "target_retention",
    "target_binder",
)

SOURCE_REQUIRED_COLUMNS = {
    "cache_uid",
    "source_kind",
    "library",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "peptide_design_code",
    "affibody_design_code",
    "weak_label",
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
}

# These terms may occur in preparation inputs and label artifacts, but never in
# the extractor-facing rows JSON or its audit manifest. Both keys and string
# values are checked.
EXTRACTOR_FORBIDDEN_TERMS = (
    "retention",
    "binder",
    "weak_label",
    "weak-label",
    "count",
    "r001",
    "r009",
    "r010",
    "pooled",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def opaque_id(*parts):
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def read_string_csv(path):
    """Preserve the valid peptide design code ``NA`` as a literal string."""
    return pd.read_csv(
        str(path), dtype=str, keep_default_na=False, na_filter=False
    )


def read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def validate_private_output_path(path, repo_root=REPO_ROOT):
    """Fail closed unless an output directory is Git-ignored private data."""
    repo_root = Path(repo_root).resolve()
    private_root = (repo_root / "private_data").resolve()
    path = Path(path).resolve()
    _require(private_root.is_dir(), "repository private_data directory is missing")
    _require(path != private_root, "output must be a child of private_data")
    try:
        path.relative_to(private_root)
    except ValueError:
        raise ValueError("output directory must be inside private_data")
    relative = path.relative_to(repo_root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(ignored.returncode == 0, "output directory is not Git-ignored")
    return path


def _ensure_new_private_directory(path):
    path = validate_private_output_path(path)
    _require(not path.exists(), "output directory exists: {}".format(path))
    path.mkdir(parents=True, mode=0o700)
    os.chmod(str(path), 0o700)
    _require(private_mode(path) == "0700", "output directory is not mode 0700")
    return path


def _require_separate_directory_trees(paths):
    resolved = [Path(path).resolve() for path in paths]
    _require(len(set(resolved)) == len(resolved), "output directories must differ")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            for child, parent in ((left, right), (right, left)):
                try:
                    child.relative_to(parent)
                except ValueError:
                    continue
                raise ValueError("label and extractor directories must be disjoint")


def _atomic_write_csv(path, frame):
    _require(not path.exists(), "output exists: {}".format(path))
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


def _atomic_write_json(path, payload):
    _require(not path.exists(), "output exists: {}".format(path))
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


def _contains_forbidden_term(value):
    lowered = str(value).lower()
    return next((term for term in EXTRACTOR_FORBIDDEN_TERMS if term in lowered), None)


def assert_extractor_payload_is_label_free(payload, location="root"):
    """Recursively reject outcome and selection-label material."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            forbidden = _contains_forbidden_term(key)
            _require(
                forbidden is None,
                "extractor payload contains forbidden term {!r} at {} key {!r}".format(
                    forbidden, location, key
                ),
            )
            assert_extractor_payload_is_label_free(
                value, location="{}.{}".format(location, key)
            )
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            assert_extractor_payload_is_label_free(
                value, location="{}[{}]".format(location, index)
            )
    elif isinstance(payload, str):
        forbidden = _contains_forbidden_term(payload)
        _require(
            forbidden is None,
            "extractor payload contains forbidden term {!r} at {}".format(
                forbidden, location
            ),
        )


def assert_extractor_rows_are_label_free(frame):
    _require(
        len(frame.columns) == len(EXTRACTOR_COLUMNS)
        and set(frame.columns) == set(EXTRACTOR_COLUMNS),
        "extractor row schema mismatch",
    )
    for column in frame.columns:
        forbidden = _contains_forbidden_term(column)
        _require(forbidden is None, "forbidden extractor column {}".format(column))
        if frame[column].dtype == object:
            for value in frame[column].astype(str):
                forbidden = _contains_forbidden_term(value)
                _require(
                    forbidden is None,
                    "forbidden extractor value in {}".format(column),
                )


def validate_source_lineage(rows_path, source_manifest_path):
    _require(Path(rows_path).is_file(), "source row table is missing")
    _require(Path(source_manifest_path).is_file(), "source manifest is missing")
    manifest = read_json(source_manifest_path)
    _require(
        manifest.get("schema_version") == SOURCE_SCHEMA_VERSION,
        "unexpected source schema",
    )
    _require(manifest.get("stage") == "prepare", "source is not prepare stage")
    rows_sha256 = sha256_file(rows_path)
    claimed = manifest.get("output", {}).get("rows_csv", {}).get("sha256")
    _require(claimed == rows_sha256, "source row hash disagrees with source manifest")
    sequence_source = manifest.get("sources", {}).get("sequence_zip", {})
    _require(sequence_source.get("sha256"), "source manifest lacks sequence archive hash")
    _require(
        sequence_source.get("template_deck_sha256"),
        "source manifest lacks template deck hash",
    )
    return {
        "source_rows_sha256": rows_sha256,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "provider_sequence_archive_sha256": sequence_source["sha256"],
        "provider_template_deck_sha256": sequence_source[
            "template_deck_sha256"
        ],
    }


def _validate_code_and_sequence_mapping(row):
    peptide = str(row.peptide_design_code)
    affibody = str(row.affibody_design_code)
    chain1 = str(row.chain1_smart_hla_linker_peptide_sequence)
    chain2 = str(row.chain2_affibody_sequence)
    _require(len(peptide) == 2, "LibB peptide code must have length 2")
    _require(len(affibody) == 5, "LibB Affibody code must have length 5")
    _require(set(peptide).issubset(AA_ALPHABET), "noncanonical peptide code")
    _require(set(affibody).issubset(AA_ALPHABET), "noncanonical Affibody code")
    _require(len(chain1) == CHAIN1_LENGTH, "chain 1 length mismatch")
    _require(len(chain2) == CHAIN2_LENGTH, "chain 2 length mismatch")
    _require(set(chain1).issubset(AA_ALPHABET), "chain 1 is noncanonical")
    _require(set(chain2).issubset(AA_ALPHABET), "chain 2 is noncanonical")
    _require(OMITTED_LINKER not in chain1, "omitted linker appears in chain 1")
    _require(OMITTED_LINKER not in chain2, "omitted linker appears in chain 2")
    peptide_sequence = chain1[-PEPTIDE_LENGTH:]
    _require(
        peptide_sequence == "SLL{}ITQV".format(peptide),
        "peptide code-to-chain mapping mismatch",
    )
    _require(
        chain1[264:266] == peptide,
        "peptide code does not occupy chain-1 residues 265/266",
    )
    observed_affibody = "".join(
        chain2[position - 1] for position in AFFIBODY_CODE_POSITIONS_1_BASED
    )
    _require(observed_affibody == affibody, "Affibody code-to-chain mapping mismatch")
    _require(
        sha256_text(chain1) == str(row.chain1_sha256),
        "source chain-1 hash mismatch",
    )
    _require(
        sha256_text(chain2) == str(row.chain2_sha256),
        "source chain-2 hash mismatch",
    )
    pair_hash = sha256_text(chain1 + "|" + chain2)
    _require(
        pair_hash == str(row.sequence_pair_sha256),
        "source sequence-pair hash mismatch",
    )
    return pair_hash


def _expected(expected, key):
    _require(key in expected, "expected-size contract lacks {}".format(key))
    return int(expected[key])


def build_tables(source, expected=None):
    """Select, validate, and separate extractor rows and both label tables."""
    if expected is None:
        expected = EXPECTED
    missing = SOURCE_REQUIRED_COLUMNS.difference(source.columns)
    _require(not missing, "source rows lack columns {}".format(sorted(missing)))
    source = source.copy()
    _require(len(source) == _expected(expected, "source_total"), "source row total changed")
    _require(source["cache_uid"].nunique() == len(source), "duplicate source cache UID")
    _require(source["pair_uid"].nunique() == len(source), "duplicate source pair UID")

    libb_candidates = source.loc[
        source["source_kind"].eq("weak") & source["library"].eq(LIBRARY)
    ].copy()
    libb_matrix = source.loc[
        source["source_kind"].eq("retention") & source["library"].eq(LIBRARY)
    ].copy()
    _require(
        len(libb_candidates) == _expected(expected, "source_libb_candidate"),
        "LibB source candidate size changed",
    )
    _require(
        len(libb_matrix) == _expected(expected, "source_libb_matrix"),
        "LibB source matrix size changed",
    )

    train = libb_candidates.loc[
        libb_candidates[
            "upstream_library_local_strict_retention_identity_cold_eligible"
        ].eq("1")
    ].copy()
    evaluation = libb_matrix.loc[libb_matrix["measurement_missing"].eq("0")].copy()
    train = train.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    evaluation = evaluation.sort_values("pair_uid", kind="mergesort").reset_index(
        drop=True
    )

    _require(len(train) == _expected(expected, "train"), "canonical train size changed")
    _require(len(evaluation) == _expected(expected, "eval"), "canonical eval size changed")
    _require(
        bool(train["weak_label"].isin(["0", "1"]).all()),
        "training row lacks a binary training label",
    )
    _require(
        int(train["weak_label"].eq("1").sum())
        == _expected(expected, "train_positive"),
        "training positive size changed",
    )
    _require(
        int(train["weak_label"].eq("0").sum())
        == _expected(expected, "train_negative"),
        "training negative size changed",
    )
    _require(
        bool(evaluation["weak_label"].eq("").all()),
        "evaluation row unexpectedly has a training label",
    )
    _require(
        bool(train["target_retention"].eq("").all())
        and bool(train["target_binder"].eq("").all()),
        "training source row contains an evaluation outcome",
    )
    _require(
        bool(
            train["upstream_library_local_shares_retention_peptide"].eq("0").all()
        ),
        "selected training peptide identity appears in evaluation",
    )
    _require(
        bool(
            train["upstream_library_local_shares_retention_affibody"].eq("0").all()
        ),
        "selected training Affibody identity appears in evaluation",
    )

    # Validate both stable identities and the exact model-facing sequences.
    for identity_column in ("peptide_uid", "affibody_uid", "pair_uid"):
        overlap = set(train[identity_column]).intersection(set(evaluation[identity_column]))
        _require(not overlap, "partner/pair identity overlap in {}".format(identity_column))
    for hash_column in ("chain1_sha256", "chain2_sha256", "sequence_pair_sha256"):
        overlap = set(train[hash_column]).intersection(set(evaluation[hash_column]))
        _require(not overlap, "sequence identity overlap in {}".format(hash_column))

    selected = pd.concat(
        [train.assign(_split="train"), evaluation.assign(_split="eval")],
        ignore_index=True,
    )
    _require(len(selected) == _expected(expected, "total"), "combined row size changed")
    pair_hashes = []
    for row in selected.itertuples(index=False):
        pair_hashes.append(_validate_code_and_sequence_mapping(row))
    _require(len(set(pair_hashes)) == len(pair_hashes), "duplicate selected sequence pair")

    # Keep the existing opaque pair UID as the canonical join key. Existing
    # sequence-only controls and the sealed evaluator already use this value.
    row_ids = selected["pair_uid"].astype(str).tolist()
    _require(bool(pd.Series(row_ids).str.len().gt(0).all()), "empty canonical row ID")
    _require(len(set(row_ids)) == len(row_ids), "duplicate canonical row ID")
    extractor = pd.DataFrame(
        {
            "row_index": list(range(len(selected))),
            "row_id": row_ids,
            "split": selected["_split"].tolist(),
            "chain1_sequence": selected[
                "chain1_smart_hla_linker_peptide_sequence"
            ].tolist(),
            "chain2_sequence": selected["chain2_affibody_sequence"].tolist(),
            "sequence_pair_sha256": pair_hashes,
        },
        columns=list(EXTRACTOR_COLUMNS),
    )
    train_extractor = extractor.loc[extractor["split"].eq("train")]
    eval_extractor = extractor.loc[extractor["split"].eq("eval")]
    _require(
        train_extractor["chain1_sequence"].nunique()
        == _expected(expected, "train_unique_chain1"),
        "training chain-1 identity size changed",
    )
    _require(
        train_extractor["chain2_sequence"].nunique()
        == _expected(expected, "train_unique_chain2"),
        "training chain-2 identity size changed",
    )
    _require(
        eval_extractor["chain1_sequence"].nunique()
        == _expected(expected, "eval_unique_chain1"),
        "evaluation chain-1 identity size changed",
    )
    _require(
        eval_extractor["chain2_sequence"].nunique()
        == _expected(expected, "eval_unique_chain2"),
        "evaluation chain-2 identity size changed",
    )
    training_labels = pd.DataFrame(
        {
            "row_id": row_ids[: len(train)],
            "weak_label": train["weak_label"].tolist(),
            "peptide_id": train["peptide_uid"].tolist(),
            "affibody_id": train["affibody_uid"].tolist(),
            "chain1_sha256": train["chain1_sha256"].tolist(),
            "chain2_sha256": train["chain2_sha256"].tolist(),
        },
        columns=list(TRAINING_LABEL_COLUMNS),
    )

    evaluation_values = pd.to_numeric(
        evaluation["target_retention"], errors="raise"
    ).astype(float)
    _require(
        bool(evaluation_values.map(math.isfinite).all()),
        "evaluation contains a non-finite direct measurement",
    )
    evaluation_binder = evaluation["target_binder"].astype(str)
    _require(
        bool(evaluation_binder.isin(["0", "1"]).all()),
        "evaluation contains a non-binary threshold label",
    )
    calculated_binder = evaluation_values.ge(BINDER_THRESHOLD).astype(int).astype(str)
    _require(
        calculated_binder.tolist() == evaluation_binder.tolist(),
        "stored evaluation threshold labels disagree with direct measurements",
    )
    _require(
        int(evaluation_binder.eq("1").sum()) == _expected(expected, "eval_positive"),
        "evaluation positive size changed",
    )
    _require(
        int(evaluation_binder.eq("0").sum()) == _expected(expected, "eval_negative"),
        "evaluation negative size changed",
    )
    evaluation_labels = pd.DataFrame(
        {
            "row_id": row_ids[len(train) :],
            "peptide_design_code": evaluation["peptide_design_code"].tolist(),
            "affibody_design_code": evaluation["affibody_design_code"].tolist(),
            "target_retention": evaluation["target_retention"].tolist(),
            "target_binder": evaluation_binder.tolist(),
        },
        columns=list(EVALUATION_LABEL_COLUMNS),
    )

    _require(
        set(training_labels["row_id"]) == set(extractor.loc[extractor.split.eq("train"), "row_id"]),
        "training label keys do not match training rows",
    )
    _require(
        set(evaluation_labels["row_id"]) == set(extractor.loc[extractor.split.eq("eval"), "row_id"]),
        "evaluation label keys do not match evaluation rows",
    )
    _require(
        set(training_labels["row_id"]).isdisjoint(set(evaluation_labels["row_id"])),
        "training and evaluation label keys overlap",
    )
    assert_extractor_rows_are_label_free(extractor)
    return extractor, training_labels, evaluation_labels


def build_rows_payload(extractor):
    """Return the sole JSON input accepted by the folding extractor."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows": extractor.to_dict(orient="records"),
    }
    assert_extractor_payload_is_label_free(payload)
    return payload


def build_extractor_manifest(lineage, extractor, rows_sha256, script_sha256):
    """Build label-free audit metadata kept beside, but outside, rows.json."""
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": "prepare",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hashes": dict(lineage),
        "summary": {
            "total": int(len(extractor)),
            "train": int(extractor["split"].eq("train").sum()),
            "eval": int(extractor["split"].eq("eval").sum()),
            "unique_row_ids": int(extractor["row_id"].nunique()),
            "unique_sequence_pairs": int(extractor["sequence_pair_sha256"].nunique()),
            "train_unique_chain1": int(
                extractor.loc[extractor["split"].eq("train"), "chain1_sequence"].nunique()
            ),
            "train_unique_chain2": int(
                extractor.loc[extractor["split"].eq("train"), "chain2_sequence"].nunique()
            ),
            "eval_unique_chain1": int(
                extractor.loc[extractor["split"].eq("eval"), "chain1_sequence"].nunique()
            ),
            "eval_unique_chain2": int(
                extractor.loc[extractor["split"].eq("eval"), "chain2_sequence"].nunique()
            ),
        },
        "split_policy": "library_local_strict_partner_disjoint",
        "partner_isolation": {
            "peptide_identity_overlap": 0,
            "affibody_identity_overlap": 0,
            "sequence_pair_overlap": 0,
        },
        "model_input": {
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
            "tcr_present": False,
            "separate_b2m_supplied": False,
        },
        "mapping_assertions": {
            "peptide_length": PEPTIDE_LENGTH,
            "peptide_code_positions_1_based": list(PEPTIDE_CODE_POSITIONS_1_BASED),
            "peptide_code_chain1_positions_1_based": list(
                PEPTIDE_CODE_CHAIN1_POSITIONS_1_BASED
            ),
            "affibody_code_positions_1_based": list(
                AFFIBODY_CODE_POSITIONS_1_BASED
            ),
            "full_construct_affibody_positions_1_based": list(
                FULL_CONSTRUCT_AFFIBODY_POSITIONS_1_BASED
            ),
            "validated_rows": int(len(extractor)),
        },
        "code": {"script_sha256": script_sha256},
        "artifact": {
            "filename": ROWS_FILENAME,
            "sha256": rows_sha256,
            "row_fields": list(EXTRACTOR_COLUMNS),
            "mode": "0600",
        },
    }
    assert_extractor_payload_is_label_free(payload)
    return payload


def build_training_manifest(lineage, rows_sha256, labels, labels_sha256):
    return {
        "schema_version": TRAINING_LABEL_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hashes": dict(lineage),
        "canonical_rows_sha256": rows_sha256,
        "labels": {
            "rows": int(len(labels)),
            "positive": int(labels["weak_label"].eq("1").sum()),
            "negative": int(labels["weak_label"].eq("0").sum()),
            "key": "row_id",
        },
        "output": {
            "filename": TRAINING_LABELS_FILENAME,
            "sha256": labels_sha256,
            "columns": list(TRAINING_LABEL_COLUMNS),
            "mode": "0600",
        },
    }


def build_evaluation_manifest(lineage, rows_sha256, labels, labels_sha256):
    return {
        "schema_version": EVALUATION_LABEL_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hashes": dict(lineage),
        "canonical_rows_sha256": rows_sha256,
        "sealed_evaluation": {
            "rows": int(len(labels)),
            "binder_threshold": BINDER_THRESHOLD,
            "binder_definition": "target_retention >= 75.0",
            "positive": int(labels["target_binder"].eq("1").sum()),
            "negative": int(labels["target_binder"].eq("0").sum()),
            "key": "row_id",
            "use_policy": "one-time final evaluation only",
        },
        "output": {
            "filename": EVALUATION_LABELS_FILENAME,
            "sha256": labels_sha256,
            "columns": list(EVALUATION_LABEL_COLUMNS),
            "mode": "0600",
        },
    }


def run_build(args):
    started = time.time()
    output_dirs = (
        args.extractor_output_dir,
        args.training_output_dir,
        args.evaluation_output_dir,
    )
    _require_separate_directory_trees(output_dirs)
    for path in output_dirs:
        checked = validate_private_output_path(path)
        _require(not checked.exists(), "output directory exists: {}".format(checked))

    lineage = validate_source_lineage(args.cache_rows, args.cache_rows_manifest)
    source = read_string_csv(args.cache_rows)
    extractor, training_labels, evaluation_labels = build_tables(source)

    extractor_dir = _ensure_new_private_directory(args.extractor_output_dir)
    training_dir = _ensure_new_private_directory(args.training_output_dir)
    evaluation_dir = _ensure_new_private_directory(args.evaluation_output_dir)

    extractor_rows_path = extractor_dir / ROWS_FILENAME
    extractor_manifest_path = extractor_dir / MANIFEST_FILENAME
    training_labels_path = training_dir / TRAINING_LABELS_FILENAME
    training_manifest_path = training_dir / MANIFEST_FILENAME
    evaluation_labels_path = evaluation_dir / EVALUATION_LABELS_FILENAME
    evaluation_manifest_path = evaluation_dir / MANIFEST_FILENAME

    _atomic_write_json(extractor_rows_path, build_rows_payload(extractor))
    extractor_rows_sha256 = sha256_file(extractor_rows_path)
    extractor_manifest = build_extractor_manifest(
        lineage,
        extractor,
        extractor_rows_sha256,
        sha256_file(Path(__file__).resolve()),
    )
    _atomic_write_json(extractor_manifest_path, extractor_manifest)

    _atomic_write_csv(training_labels_path, training_labels)
    training_manifest = build_training_manifest(
        lineage,
        extractor_rows_sha256,
        training_labels,
        sha256_file(training_labels_path),
    )
    _atomic_write_json(training_manifest_path, training_manifest)

    _atomic_write_csv(evaluation_labels_path, evaluation_labels)
    evaluation_manifest = build_evaluation_manifest(
        lineage,
        extractor_rows_sha256,
        evaluation_labels,
        sha256_file(evaluation_labels_path),
    )
    _atomic_write_json(evaluation_manifest_path, evaluation_manifest)

    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "elapsed_seconds": round(time.time() - started, 6),
                "extractor_rows": int(len(extractor)),
                "training_rows": int(len(training_labels)),
                "evaluation_rows": int(len(evaluation_labels)),
            },
            sort_keys=True,
        )
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-rows",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv",
    )
    parser.add_argument(
        "--cache-rows-manifest",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/rows/manifest.json",
    )
    parser.add_argument("--extractor-output-dir", required=True, type=Path)
    parser.add_argument("--training-output-dir", required=True, type=Path)
    parser.add_argument("--evaluation-output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    run_build(parse_args(argv))


if __name__ == "__main__":
    main()
