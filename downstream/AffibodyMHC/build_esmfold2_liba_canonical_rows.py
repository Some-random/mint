#!/usr/bin/env python
"""Build the canonical, label-isolated LibA row contract.

This CPU-only preparation step mirrors the LibB canonical-row contract while
keeping the two library specifications independent.  It reads the audited MINT
weak-cache rows, selects the exact library-local strict LibA training set and
all 108 measured LibA evaluation pairs, and writes three disjoint artifact
groups:

* ``rows.json`` plus ``manifest.json``: label-free sequences for extraction;
* ``training_labels.csv`` plus ``manifest.json``: selection-derived labels; and
* ``evaluation_labels.csv`` plus ``manifest.json``: sealed direct outcomes.

The model-facing Affibody remains the provider-displayed 58-aa sequence.  Its
four mutable residues are indexed at displayed positions 13/17/27/31 (Python
indices 12/16/26/30) and are reported with the corrected crystal-aligned labels
15/19/29/33.  The corrected labels are metadata, not indices into the 58-aa
model input.
"""

from __future__ import print_function

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC import build_esmfold2_libb_canonical_rows as shared


REPO_ROOT = shared.REPO_ROOT

SCHEMA_VERSION = "esmfold2-liba-canonical-rows-v1"
AUDIT_SCHEMA_VERSION = "esmfold2-liba-canonical-rows-audit-v1"
TRAINING_LABEL_SCHEMA_VERSION = "esmfold2-liba-training-labels-v1"
EVALUATION_LABEL_SCHEMA_VERSION = "esmfold2-liba-evaluation-labels-v1"
SOURCE_SCHEMA_VERSION = shared.SOURCE_SCHEMA_VERSION

LIBRARY = "LibA"
ROWS_FILENAME = shared.ROWS_FILENAME
MANIFEST_FILENAME = shared.MANIFEST_FILENAME
TRAINING_LABELS_FILENAME = shared.TRAINING_LABELS_FILENAME
EVALUATION_LABELS_FILENAME = shared.EVALUATION_LABELS_FILENAME

CHAIN1_LENGTH = 270
CHAIN2_LENGTH = 58
PEPTIDE_LENGTH = 9
PEPTIDE_CODE_POSITIONS_1_BASED = (4, 5)
PEPTIDE_CODE_CHAIN1_POSITIONS_1_BASED = (265, 266)
AFFIBODY_CODE_POSITIONS_1_BASED = (13, 17, 27, 31)
AFFIBODY_CODE_PYTHON_INDICES_0_BASED = (12, 16, 26, 30)
AFFIBODY_CRYSTAL_ALIGNED_LABELS_1_BASED = (15, 19, 29, 33)
HIDDEN_AFFIBODY_PREFIX = "MA"
OMITTED_LINKER = "GGSLEVLFQGPGSG"
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
BINDER_THRESHOLD = 75.0

EXPECTED = {
    "source_total": 69945,
    "source_liba_candidate": 31007,
    "source_liba_matrix": 108,
    "train": 22542,
    "train_positive": 11320,
    "train_negative": 11222,
    "eval": 108,
    "eval_positive": 38,
    "eval_negative": 70,
    "total": 22650,
    "train_unique_chain1": 216,
    "train_unique_chain2": 13590,
    "eval_unique_chain1": 9,
    "eval_unique_chain2": 12,
}

EXTRACTOR_COLUMNS = shared.EXTRACTOR_COLUMNS
TRAINING_LABEL_COLUMNS = shared.TRAINING_LABEL_COLUMNS
EVALUATION_LABEL_COLUMNS = shared.EVALUATION_LABEL_COLUMNS
SOURCE_REQUIRED_COLUMNS = shared.SOURCE_REQUIRED_COLUMNS

# Reuse the hardened, library-neutral I/O and label-isolation primitives.  The
# LibB-specific selection and mapping logic is deliberately not reused.
_require = shared._require
sha256_file = shared.sha256_file
sha256_text = shared.sha256_text
opaque_id = shared.opaque_id
read_string_csv = shared.read_string_csv
read_json = shared.read_json
private_mode = shared.private_mode
validate_private_output_path = shared.validate_private_output_path
_ensure_new_private_directory = shared._ensure_new_private_directory
_require_separate_directory_trees = shared._require_separate_directory_trees
_atomic_write_csv = shared._atomic_write_csv
_atomic_write_json = shared._atomic_write_json
assert_extractor_payload_is_label_free = shared.assert_extractor_payload_is_label_free
assert_extractor_rows_are_label_free = shared.assert_extractor_rows_are_label_free
validate_source_lineage = shared.validate_source_lineage


def _expected(expected, key):
    _require(key in expected, "expected-size contract lacks {}".format(key))
    return int(expected[key])


def _validate_code_and_sequence_mapping(row):
    peptide = str(row.peptide_design_code)
    affibody = str(row.affibody_design_code)
    chain1 = str(row.chain1_smart_hla_linker_peptide_sequence)
    chain2 = str(row.chain2_affibody_sequence)

    _require(len(peptide) == 2, "LibA peptide code must have length 2")
    _require(len(affibody) == 4, "LibA Affibody code must have length 4")
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


def build_tables(source, expected=None):
    """Select LibA rows and physically separate extractor and label tables."""
    if expected is None:
        expected = EXPECTED
    missing = SOURCE_REQUIRED_COLUMNS.difference(source.columns)
    _require(not missing, "source rows lack columns {}".format(sorted(missing)))
    source = source.copy()
    _require(len(source) == _expected(expected, "source_total"), "source row total changed")
    _require(source["cache_uid"].nunique() == len(source), "duplicate source cache UID")
    _require(source["pair_uid"].nunique() == len(source), "duplicate source pair UID")

    candidates = source.loc[
        source["source_kind"].eq("weak") & source["library"].eq(LIBRARY)
    ].copy()
    matrix = source.loc[
        source["source_kind"].eq("retention") & source["library"].eq(LIBRARY)
    ].copy()
    _require(
        len(candidates) == _expected(expected, "source_liba_candidate"),
        "LibA source candidate size changed",
    )
    _require(
        len(matrix) == _expected(expected, "source_liba_matrix"),
        "LibA source matrix size changed",
    )
    _require(
        bool(matrix["measurement_missing"].eq("0").all()),
        "LibA evaluation matrix is not complete",
    )

    train = candidates.loc[
        candidates[
            "upstream_library_local_strict_retention_identity_cold_eligible"
        ].eq("1")
    ].copy()
    evaluation = matrix.copy()
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

    # Independently verify the upstream strict flag using both stable IDs and
    # exact sequence hashes.  This prevents a stale flag from leaking either
    # partner into evaluation.
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
    pair_hashes = [
        _validate_code_and_sequence_mapping(row)
        for row in selected.itertuples(index=False)
    ]
    _require(len(set(pair_hashes)) == len(pair_hashes), "duplicate selected sequence pair")

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
            "row_id": row_ids[len(train):],
            "peptide_design_code": evaluation["peptide_design_code"].tolist(),
            "affibody_design_code": evaluation["affibody_design_code"].tolist(),
            "target_retention": evaluation["target_retention"].tolist(),
            "target_binder": evaluation_binder.tolist(),
        },
        columns=list(EVALUATION_LABEL_COLUMNS),
    )

    _require(
        set(training_labels["row_id"])
        == set(extractor.loc[extractor.split.eq("train"), "row_id"]),
        "training label keys do not match training rows",
    )
    _require(
        set(evaluation_labels["row_id"])
        == set(extractor.loc[extractor.split.eq("eval"), "row_id"]),
        "evaluation label keys do not match evaluation rows",
    )
    _require(
        set(training_labels["row_id"]).isdisjoint(set(evaluation_labels["row_id"])),
        "training and evaluation label keys overlap",
    )
    assert_extractor_rows_are_label_free(extractor)
    return extractor, training_labels, evaluation_labels


def build_rows_payload(extractor):
    """Return the sole sequence payload intended for feature extraction."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows": extractor.to_dict(orient="records"),
    }
    assert_extractor_payload_is_label_free(payload)
    return payload


def build_extractor_manifest(lineage, extractor, rows_sha256, script_sha256):
    """Build label-free metadata and lineage for the extractor rows."""
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": "prepare",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library": LIBRARY,
        "input_hashes": dict(lineage),
        "summary": {
            "total": int(len(extractor)),
            "train": int(extractor["split"].eq("train").sum()),
            "eval": int(extractor["split"].eq("eval").sum()),
            "unique_row_ids": int(extractor["row_id"].nunique()),
            "unique_sequence_pairs": int(extractor["sequence_pair_sha256"].nunique()),
            "train_unique_chain1": int(
                extractor.loc[
                    extractor["split"].eq("train"), "chain1_sequence"
                ].nunique()
            ),
            "train_unique_chain2": int(
                extractor.loc[
                    extractor["split"].eq("train"), "chain2_sequence"
                ].nunique()
            ),
            "eval_unique_chain1": int(
                extractor.loc[
                    extractor["split"].eq("eval"), "chain1_sequence"
                ].nunique()
            ),
            "eval_unique_chain2": int(
                extractor.loc[
                    extractor["split"].eq("eval"), "chain2_sequence"
                ].nunique()
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
            "affibody_displayed_sequence_positions_1_based": list(
                AFFIBODY_CODE_POSITIONS_1_BASED
            ),
            "affibody_python_indices_0_based": list(
                AFFIBODY_CODE_PYTHON_INDICES_0_BASED
            ),
            "affibody_crystal_aligned_labels_1_based": list(
                AFFIBODY_CRYSTAL_ALIGNED_LABELS_1_BASED
            ),
            "hidden_preceding_affibody_residues": HIDDEN_AFFIBODY_PREFIX,
            "model_sequence_uses_displayed_positions": True,
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
    _atomic_write_json(
        extractor_manifest_path,
        build_extractor_manifest(
            lineage,
            extractor,
            extractor_rows_sha256,
            sha256_file(Path(__file__).resolve()),
        ),
    )

    _atomic_write_csv(training_labels_path, training_labels)
    _atomic_write_json(
        training_manifest_path,
        build_training_manifest(
            lineage,
            extractor_rows_sha256,
            training_labels,
            sha256_file(training_labels_path),
        ),
    )

    _atomic_write_csv(evaluation_labels_path, evaluation_labels)
    _atomic_write_json(
        evaluation_manifest_path,
        build_evaluation_manifest(
            lineage,
            extractor_rows_sha256,
            evaluation_labels,
            sha256_file(evaluation_labels_path),
        ),
    )

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
