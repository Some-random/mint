#!/usr/bin/env python
"""Build the corrected 228-row direct-retention panel without retraining.

This script applies one narrowly defined provider correction to the existing
retention matrix.  It refuses to run unless all of the following are true:

* the correction JSON is hash-bound to the source retention CSV;
* the source has the old 108-row LibA plus 120-cell LibB layout;
* ``LibB | AH | LIFTK`` is the only missing measurement; and
* the requested replacement is exactly 87.94.

It also consumes the existing sequence-expanded retention table and publishes
``libb_evaluation_panel.csv``.  That normalized file has the complete 120-row
LibB Cartesian matrix, one opaque identifier column (``pair_uid``), outcome
columns understood by ``recompute_libb_metrics_120.py``, and the unchanged
partner IDs, sequences, lengths, and sequence hashes.  In that sequence table,
only the missing target's ``target_retention`` and ``target_binder`` are filled.

The output is a new, Git-ignored directory below ``private_data``.  Existing
files are never overwritten.  Every non-target source row is copied as its
original bytes, including its original line ending.

The explicit correction JSON has this schema::

    {
      "schema_version": "affibody-retention-provider-correction-v1",
      "source_retention_matrix_sha256": "<sha256 of source CSV>",
      "provider_notice": {
        "provider": "Xinyu",
        "received_date": "2026-09-03",
        "statement": "LibB AH x LIFTK retention = 87.94"
      },
      "correction": {
        "library": "LibB",
        "peptide_design_code": "AH",
        "affibody_design_code": "LIFTK",
        "retention_percent": 87.94
      }
    }

``provider_notice.statement`` is retained as provenance, but the generated
``timepoint_status`` deliberately says that the value came from a provider
revision notice and was not independently verified from the deck.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

CORRECTION_SCHEMA_VERSION = "affibody-retention-provider-correction-v1"
OUTPUT_SCHEMA_VERSION = "affibody-corrected-retention-panel-v1"
TIMEPOINT_STATUS = "provider_revision_notice_not_deck_verified"

EXPECTED_COLUMNS = (
    "provisional_matrix_key",
    "library",
    "affibody_design_code",
    "peptide_design_code",
    "axis_assignment_basis",
    "retention_percent",
    "retention_time_min_as_labeled",
    "timepoint_status",
    "binder_label_ge_75",
    "measurement_missing",
    "source_slide",
)
SEQUENCE_SOURCE_COLUMNS = EXPECTED_COLUMNS + (
    "target_retention",
    "target_binder",
    "aff_p13",
    "aff_p17",
    "aff_p27",
    "aff_p31",
    "pep_p4",
    "pep_p5",
    "aff_p6",
    "aff_p10",
    "aff_p14",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_length",
    "chain2_length",
    "omitted_linker_sequence",
    "full_construct_audit_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)
EVALUATION_PANEL_COLUMNS = (
    "pair_uid",
    "provisional_matrix_key",
    "library",
    "peptide_design_code",
    "affibody_design_code",
    "target_retention",
    "target_binder",
    "peptide_uid",
    "affibody_uid",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_length",
    "chain2_length",
    "omitted_linker_sequence",
    "full_construct_audit_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
OMITTED_LINKER = "GGSLEVLFQGPGSG"
TARGET_IDENTITY = {
    "library": "LibB",
    "peptide_design_code": "AH",
    "affibody_design_code": "LIFTK",
}
TARGET_MATRIX_KEY = "LibB|LIFTK|AH"
TARGET_RETENTION = Decimal("87.94")
CHANGED_FIELDS = (
    "retention_percent",
    "timepoint_status",
    "binder_label_ge_75",
    "measurement_missing",
)
SEQUENCE_CHANGED_FIELDS = ("target_retention", "target_binder")
AUDIT_COLUMNS = (
    "output_file",
    "provisional_matrix_key",
    "library",
    "peptide_design_code",
    "affibody_design_code",
    "field",
    "old_value",
    "new_value",
    "evidence_kind",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def file_record(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_utc": _utc_from_ns(int(stat.st_mtime_ns)),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _csv_bytes(columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row[column] for column in columns})
    return buffer.getvalue().encode("utf-8")


def validate_private_output_path(path: Path, repo_root: Path | None = None) -> Path:
    """Require a not-yet-existing, Git-ignored child of ``private_data``."""
    repo_root = (REPO_ROOT if repo_root is None else repo_root).resolve()
    private_root = (repo_root / "private_data").resolve()
    output = path.resolve()
    _require(private_root.is_dir(), "repository private_data directory is missing")
    _require(output != private_root, "output directory must be below private_data")
    try:
        relative_private = output.relative_to(private_root)
        relative_repo = output.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("output directory must be below private_data") from exc
    _require(bool(relative_private.parts), "output directory must be below private_data")
    _require(not output.exists(), "output directory exists; refusing overwrite")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative_repo)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(ignored.returncode == 0, "output directory is not Git-ignored")
    return output


def _load_correction(path: Path, source_sha256: str) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError("correction JSON is invalid") from exc

    expected_top = {
        "schema_version",
        "source_retention_matrix_sha256",
        "provider_notice",
        "correction",
    }
    _require(isinstance(payload, dict), "correction JSON must be an object")
    _require(set(payload) == expected_top, "correction JSON has unexpected fields")
    _require(
        payload["schema_version"] == CORRECTION_SCHEMA_VERSION,
        "correction schema version changed",
    )
    declared_hash = payload["source_retention_matrix_sha256"]
    _require(
        isinstance(declared_hash, str) and re.fullmatch(r"[0-9a-f]{64}", declared_hash),
        "source_retention_matrix_sha256 must be a lowercase SHA256",
    )
    _require(declared_hash == source_sha256, "correction JSON is bound to another source CSV")

    notice = payload["provider_notice"]
    _require(isinstance(notice, dict), "provider_notice must be an object")
    _require(
        set(notice) == {"provider", "received_date", "statement"},
        "provider_notice has unexpected fields",
    )
    _require(notice["provider"] == "Xinyu", "unexpected correction provider")
    _require(
        isinstance(notice["received_date"], str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", notice["received_date"]),
        "provider notice received_date must be YYYY-MM-DD",
    )
    _require(
        isinstance(notice["statement"], str) and bool(notice["statement"].strip()),
        "provider notice statement is empty",
    )

    correction = payload["correction"]
    _require(isinstance(correction, dict), "correction must be an object")
    _require(
        set(correction)
        == {
            "library",
            "peptide_design_code",
            "affibody_design_code",
            "retention_percent",
        },
        "correction has unexpected fields",
    )
    for key, expected in TARGET_IDENTITY.items():
        _require(correction[key] == expected, "correction targets unexpected {}".format(key))
    try:
        retention = Decimal(str(correction["retention_percent"]))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("correction retention_percent is not numeric") from exc
    _require(retention == TARGET_RETENTION, "correction retention_percent must be exactly 87.94")
    return payload


def _detect_lines(raw: bytes) -> Tuple[List[bytes], str]:
    _require(bool(raw), "source retention CSV is empty")
    _require(not raw.startswith(b"\xef\xbb\xbf"), "source CSV must not contain a UTF-8 BOM")
    _require(b"\x00" not in raw, "source CSV contains NUL bytes")
    if b"\r\n" in raw:
        remainder = raw.replace(b"\r\n", b"")
        _require(b"\r" not in remainder and b"\n" not in remainder, "mixed CSV line endings")
        newline = "\r\n"
    else:
        _require(b"\r" not in raw, "unsupported CR-only CSV line endings")
        newline = "\n"
    _require(raw.endswith(newline.encode("ascii")), "source CSV must end with a newline")
    lines = raw.splitlines(keepends=True)
    _require(all(line.endswith(newline.encode("ascii")) for line in lines), "mixed CSV line endings")
    return lines, newline


def _parse_physical_rows(
    lines: Sequence[bytes],
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
    source_name: str = "retention CSV",
) -> Tuple[List[str], List[List[str]]]:
    parsed: List[List[str]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("source CSV is not UTF-8 at line {}".format(line_number)) from exc
        values = list(csv.reader([text], strict=True))
        _require(len(values) == 1, "unexpected embedded record at line {}".format(line_number))
        parsed.append(values[0])
    _require(bool(parsed), "source retention CSV has no header")
    header = parsed[0]
    _require(
        tuple(header) == tuple(expected_columns),
        "source {} schema changed".format(source_name),
    )
    rows = parsed[1:]
    _require(
        all(len(row) == len(header) for row in rows),
        "source retention CSV has a malformed row",
    )
    return header, rows


def _as_dicts(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[Dict[str, str]]:
    return [dict(zip(header, row)) for row in rows]


def _validate_old_panel(rows: Sequence[Mapping[str, str]]) -> int:
    _require(len(rows) == 228, "source retention matrix must contain 228 cells")
    keys = [row["provisional_matrix_key"] for row in rows]
    _require(len(set(keys)) == len(keys), "duplicate provisional_matrix_key in source")

    target_indices = [
        index
        for index, row in enumerate(rows)
        if all(row[key] == value for key, value in TARGET_IDENTITY.items())
    ]
    _require(len(target_indices) == 1, "expected exactly one LibB AH x LIFTK cell")
    target_index = target_indices[0]
    target = rows[target_index]
    _require(target["provisional_matrix_key"] == TARGET_MATRIX_KEY, "target matrix key changed")
    _require(target["retention_percent"] == "", "target old retention value is not missing")
    _require(target["binder_label_ge_75"] == "", "target old binder label is not missing")
    _require(target["measurement_missing"] == "1", "target old missing flag is not 1")
    _require(
        target["retention_time_min_as_labeled"] == "30",
        "target retention time is not the expected 30 minutes",
    )

    for row in rows:
        _require(row["library"] in {"LibA", "LibB"}, "unexpected library in source")
        _require(row["measurement_missing"] in {"0", "1"}, "invalid missing flag")
        if row["measurement_missing"] == "0":
            _require(row["retention_percent"] != "", "measured row has blank retention")
            try:
                retention = Decimal(row["retention_percent"])
            except InvalidOperation as exc:
                raise ValueError("measured retention is not numeric") from exc
            _require(row["binder_label_ge_75"] in {"0", "1"}, "measured row has invalid binder label")
            _require(
                row["binder_label_ge_75"] == str(int(retention >= Decimal("75"))),
                "stored binder label disagrees with retention >= 75",
            )
        else:
            _require(row["retention_percent"] == "", "missing row has a retention value")
            _require(row["binder_label_ge_75"] == "", "missing row has a binder label")

    liba = [row for row in rows if row["library"] == "LibA"]
    libb = [row for row in rows if row["library"] == "LibB"]
    _require(len(liba) == 108 and len(libb) == 120, "old library matrix sizes changed")
    _require(sum(row["measurement_missing"] == "0" for row in liba) == 108, "LibA measured count changed")
    _require(sum(row["measurement_missing"] == "0" for row in libb) == 119, "old LibB measured count changed")
    _require(sum(row["measurement_missing"] == "1" for row in rows) == 1, "target is not the sole missing cell")
    _require(
        len({row["peptide_design_code"] for row in liba}) == 9
        and len({row["affibody_design_code"] for row in liba}) == 12,
        "LibA is not a complete 9 x 12 matrix",
    )
    _require(
        len({row["peptide_design_code"] for row in libb}) == 12
        and len({row["affibody_design_code"] for row in libb}) == 10,
        "LibB is not a complete 12 x 10 matrix",
    )
    return target_index


def _render_target_line(values: Sequence[str], newline: str) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=newline)
    writer.writerow(list(values))
    return buffer.getvalue().encode("utf-8")


def _apply_correction(
    lines: Sequence[bytes],
    header: Sequence[str],
    source_rows: Sequence[Sequence[str]],
    target_index: int,
    newline: str,
) -> Tuple[bytes, List[Dict[str, str]], List[Dict[str, str]]]:
    corrected_rows = [list(row) for row in source_rows]
    column_index = {column: index for index, column in enumerate(header)}
    replacements = {
        "retention_percent": "87.94",
        "timepoint_status": TIMEPOINT_STATUS,
        "binder_label_ge_75": "1",
        "measurement_missing": "0",
    }
    audit: List[Dict[str, str]] = []
    target = corrected_rows[target_index]
    for field in CHANGED_FIELDS:
        old_value = target[column_index[field]]
        new_value = replacements[field]
        _require(old_value != new_value, "correction would not change {}".format(field))
        target[column_index[field]] = new_value
        audit.append(
            {
                "provisional_matrix_key": TARGET_MATRIX_KEY,
                "output_file": "retention_matrix.csv",
                "library": "LibB",
                "peptide_design_code": "AH",
                "affibody_design_code": "LIFTK",
                "field": field,
                "old_value": old_value,
                "new_value": new_value,
                "evidence_kind": "provider_revision_notice_not_independently_deck_verified",
            }
        )

    output_lines = list(lines)
    output_lines[target_index + 1] = _render_target_line(target, newline)
    output = b"".join(output_lines)

    # Strong preservation assertion: every physical row other than the target
    # remains byte-for-byte identical to the source.
    reparsed_lines, output_newline = _detect_lines(output)
    _require(output_newline == newline, "output line ending changed")
    _require(len(reparsed_lines) == len(lines), "output physical row count changed")
    for line_index, (before, after) in enumerate(zip(lines, reparsed_lines)):
        if line_index != target_index + 1:
            _require(before == after, "non-target source row bytes changed")

    corrected_dicts = _as_dicts(header, corrected_rows)
    source_dicts = _as_dicts(header, source_rows)
    for row_index, (before, after) in enumerate(zip(source_dicts, corrected_dicts)):
        if row_index != target_index:
            _require(before == after, "non-target source cell value changed")
        else:
            unchanged = set(header).difference(CHANGED_FIELDS)
            _require(
                all(before[field] == after[field] for field in unchanged),
                "an undeclared target field changed",
            )
    return output, audit, corrected_dicts


def _summarize_corrected(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    libraries: Dict[str, Dict[str, int]] = {}
    for library in ("LibA", "LibB"):
        subset = [row for row in rows if row["library"] == library]
        measured = [row for row in subset if row["measurement_missing"] == "0"]
        positive = [row for row in measured if row["binder_label_ge_75"] == "1"]
        negative = [row for row in measured if row["binder_label_ge_75"] == "0"]
        libraries[library] = {
            "matrix_cells": len(subset),
            "measured": len(measured),
            "missing": len(subset) - len(measured),
            "binder_retention_ge_75": len(positive),
            "nonbinder_retention_lt_75": len(negative),
        }
    expected = {
        "LibA": {
            "matrix_cells": 108,
            "measured": 108,
            "missing": 0,
            "binder_retention_ge_75": 38,
            "nonbinder_retention_lt_75": 70,
        },
        "LibB": {
            "matrix_cells": 120,
            "measured": 120,
            "missing": 0,
            "binder_retention_ge_75": 61,
            "nonbinder_retention_lt_75": 59,
        },
    }
    _require(libraries == expected, "corrected panel counts do not match the provider notice")
    return {
        "matrix_cells": len(rows),
        "measured": sum(values["measured"] for values in libraries.values()),
        "missing": sum(values["missing"] for values in libraries.values()),
        "libraries": libraries,
    }


def _opaque_id(*parts: str) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:20]


def _sha256_ascii(value: str, field: str) -> str:
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("{} is not ASCII".format(field)) from exc
    return sha256_bytes(payload)


def _validate_sequence_row(row: Mapping[str, str]) -> None:
    """Validate every opaque ID and sequence-bearing field in one source row."""
    library = row["library"]
    peptide_code = row["peptide_design_code"]
    affibody_code = row["affibody_design_code"]
    _require(
        row["pair_uid"] == _opaque_id(library, peptide_code, affibody_code),
        "sequence source pair_uid does not match its biological key",
    )
    _require(
        row["peptide_uid"] == _opaque_id(library, "pep", peptide_code),
        "sequence source peptide_uid does not match its peptide key",
    )
    _require(
        row["affibody_uid"] == _opaque_id(library, "aff", affibody_code),
        "sequence source affibody_uid does not match its Affibody key",
    )

    chain1 = row["chain1_smart_hla_linker_peptide_sequence"]
    chain2 = row["chain2_affibody_sequence"]
    omitted = row["omitted_linker_sequence"]
    full_construct = row["full_construct_audit_sequence"]
    _require(len(chain1) == 270 and row["chain1_length"] == "270", "sequence source chain1 length changed")
    _require(len(chain2) == 58 and row["chain2_length"] == "58", "sequence source chain2 length changed")
    _require(
        set(chain1).issubset(AA_ALPHABET) and set(chain2).issubset(AA_ALPHABET),
        "sequence source contains a noncanonical amino acid",
    )
    _require(omitted == OMITTED_LINKER, "sequence source omitted linker changed")
    _require(
        full_construct == chain1 + omitted + chain2,
        "sequence source full construct disagrees with its partner sequences",
    )
    _require(
        row["chain1_sha256"] == _sha256_ascii(chain1, "chain1 sequence"),
        "sequence source chain1 SHA256 mismatch",
    )
    _require(
        row["chain2_sha256"] == _sha256_ascii(chain2, "chain2 sequence"),
        "sequence source chain2 SHA256 mismatch",
    )
    _require(
        row["sequence_pair_sha256"]
        == _sha256_ascii(chain1 + "|" + chain2, "sequence pair"),
        "sequence source pair SHA256 mismatch",
    )

    peptide = chain1[-9:]
    _require(peptide[3:5] == peptide_code, "sequence source peptide code reconstruction failed")
    _require(
        row["pep_p4"] + row["pep_p5"] == peptide_code,
        "sequence source peptide site columns changed",
    )
    if library == "LibA":
        positions = (13, 17, 27, 31)
        site_fields = ("aff_p13", "aff_p17", "aff_p27", "aff_p31")
        empty_fields = ("aff_p6", "aff_p10", "aff_p14")
    else:
        positions = (6, 10, 13, 14, 17)
        site_fields = ("aff_p6", "aff_p10", "aff_p13", "aff_p14", "aff_p17")
        empty_fields = ("aff_p27", "aff_p31")
    reconstructed_affibody_code = "".join(chain2[position - 1] for position in positions)
    _require(
        reconstructed_affibody_code == affibody_code,
        "sequence source Affibody code reconstruction failed",
    )
    _require(
        "".join(row[field] for field in site_fields) == affibody_code,
        "sequence source Affibody site columns changed",
    )
    _require(
        all(row[field] == "" for field in empty_fields),
        "sequence source has values in library-inapplicable site columns",
    )


def _build_libb_evaluation_panel(
    sequence_rows: Sequence[Mapping[str, str]],
    matrix_source_rows: Sequence[Mapping[str, str]],
    corrected_matrix_rows: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Validate the sequence expansion and fill two outcome fields in one row."""
    _require(len(sequence_rows) == 228, "sequence source must contain 228 rows")
    sequence_keys = [row["provisional_matrix_key"] for row in sequence_rows]
    _require(len(set(sequence_keys)) == 228, "sequence source has duplicate matrix keys")
    _require(len({row["pair_uid"] for row in sequence_rows}) == 228, "sequence source has duplicate pair_uid")
    _require(
        len({row["sequence_pair_sha256"] for row in sequence_rows}) == 228,
        "sequence source has duplicate sequence pairs",
    )

    source_by_key = {row["provisional_matrix_key"]: row for row in matrix_source_rows}
    corrected_by_key = {
        row["provisional_matrix_key"]: row for row in corrected_matrix_rows
    }
    _require(set(sequence_keys) == set(source_by_key), "sequence and retention sources have different rows")
    _require(set(sequence_keys) == set(corrected_by_key), "corrected retention rows do not match sequence rows")

    output_rows: List[Dict[str, str]] = []
    audit_rows: List[Dict[str, str]] = []
    for source_sequence_row in sequence_rows:
        row = dict(source_sequence_row)
        key = row["provisional_matrix_key"]
        matrix_row = source_by_key[key]
        corrected_matrix_row = corrected_by_key[key]
        _require(
            all(row[column] == matrix_row[column] for column in EXPECTED_COLUMNS),
            "sequence expansion disagrees with source retention row {}".format(key),
        )
        _validate_sequence_row(row)

        if matrix_row["measurement_missing"] == "0":
            try:
                target_retention = Decimal(row["target_retention"])
            except InvalidOperation as exc:
                raise ValueError("sequence source has nonnumeric target_retention") from exc
            _require(
                target_retention == Decimal(matrix_row["retention_percent"]),
                "sequence target_retention disagrees with retention source",
            )
            _require(
                row["target_binder"] == matrix_row["binder_label_ge_75"],
                "sequence target_binder disagrees with retention source",
            )
        else:
            _require(key == TARGET_MATRIX_KEY, "unexpected missing sequence target")
            _require(
                row["target_retention"] == "" and row["target_binder"] == "",
                "old sequence target is not missing",
            )

        if key == TARGET_MATRIX_KEY:
            replacements = {"target_retention": "87.94", "target_binder": "1"}
            for field in SEQUENCE_CHANGED_FIELDS:
                old_value = row[field]
                new_value = replacements[field]
                _require(old_value == "", "old sequence {} is not missing".format(field))
                row[field] = new_value
                audit_rows.append(
                    {
                        "output_file": "libb_evaluation_panel.csv",
                        "provisional_matrix_key": TARGET_MATRIX_KEY,
                        "library": "LibB",
                        "peptide_design_code": "AH",
                        "affibody_design_code": "LIFTK",
                        "field": field,
                        "old_value": old_value,
                        "new_value": new_value,
                        "evidence_kind": "provider_revision_notice_not_independently_deck_verified",
                    }
                )

        if row["library"] == "LibB":
            # The normalized panel intentionally omits the old raw missingness
            # columns, which remain represented in the separately corrected
            # retention_matrix.csv artifact.
            output_rows.append({column: row[column] for column in EVALUATION_PANEL_COLUMNS})

        corrected = corrected_matrix_row
        if row["library"] == "LibB":
            _require(
                Decimal(row["target_retention"]) == Decimal(corrected["retention_percent"]),
                "normalized target_retention disagrees with corrected matrix",
            )
            _require(
                row["target_binder"] == corrected["binder_label_ge_75"],
                "normalized target_binder disagrees with corrected matrix",
            )

        # All source sequence/ID fields, including those on the corrected row,
        # must survive unchanged.  The only permitted sequence-table changes
        # are the two outcome fields audited above.
        unchanged_fields = set(SEQUENCE_SOURCE_COLUMNS).difference(SEQUENCE_CHANGED_FIELDS)
        _require(
            all(row[field] == source_sequence_row[field] for field in unchanged_fields),
            "a sequence or ID field changed while normalizing {}".format(key),
        )
        if key != TARGET_MATRIX_KEY:
            _require(row == source_sequence_row, "a non-target sequence row changed")

    _require(len(output_rows) == 120, "normalized LibB panel must contain 120 rows")
    peptides = {row["peptide_design_code"] for row in output_rows}
    affibodies = {row["affibody_design_code"] for row in output_rows}
    pairs = {(row["peptide_design_code"], row["affibody_design_code"]) for row in output_rows}
    _require(
        len(peptides) == 12
        and len(affibodies) == 10
        and pairs == {(peptide, affibody) for peptide in peptides for affibody in affibodies},
        "normalized LibB panel is not the complete 12 x 10 matrix",
    )
    _require(
        sum(row["target_binder"] == "1" for row in output_rows) == 61
        and sum(row["target_binder"] == "0" for row in output_rows) == 59,
        "normalized LibB targets must contain 61 binders and 59 non-binders",
    )
    return output_rows, audit_rows


def _write_file_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _require((path.stat().st_mode & 0o7777) == 0o600, "output file mode is not 0600")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--sequence-csv", required=True, type=Path)
    parser.add_argument("--correction-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, repo_root: Path | None = None) -> Dict[str, Any]:
    started = time.time()
    source_path = args.source_csv.resolve()
    sequence_path = args.sequence_csv.resolve()
    correction_path = args.correction_json.resolve()
    _require(source_path.is_file(), "source retention CSV does not exist")
    _require(sequence_path.is_file(), "source retention sequence CSV does not exist")
    _require(correction_path.is_file(), "correction JSON does not exist")
    output_dir = validate_private_output_path(args.output_dir, repo_root=repo_root)

    source_record = file_record(source_path)
    sequence_record = file_record(sequence_path)
    correction_record = file_record(correction_path)
    script_path = Path(__file__).resolve()
    script_record = file_record(script_path)
    correction = _load_correction(correction_path, source_record["sha256"])

    raw = source_path.read_bytes()
    _require(sha256_bytes(raw) == source_record["sha256"], "source CSV changed while reading")
    lines, newline = _detect_lines(raw)
    header, source_rows = _parse_physical_rows(lines)
    source_dicts = _as_dicts(header, source_rows)
    target_index = _validate_old_panel(source_dicts)
    corrected_csv, audit_rows, corrected_rows = _apply_correction(
        lines, header, source_rows, target_index, newline
    )
    summary = _summarize_corrected(corrected_rows)

    sequence_raw = sequence_path.read_bytes()
    _require(
        sha256_bytes(sequence_raw) == sequence_record["sha256"],
        "source retention sequence CSV changed while reading",
    )
    sequence_lines, _ = _detect_lines(sequence_raw)
    sequence_header, sequence_rows_raw = _parse_physical_rows(
        sequence_lines,
        expected_columns=SEQUENCE_SOURCE_COLUMNS,
        source_name="retention sequence CSV",
    )
    sequence_rows = _as_dicts(sequence_header, sequence_rows_raw)
    evaluation_rows, sequence_audit_rows = _build_libb_evaluation_panel(
        sequence_rows, source_dicts, corrected_rows
    )
    audit_rows.extend(sequence_audit_rows)
    evaluation_csv = _csv_bytes(EVALUATION_PANEL_COLUMNS, evaluation_rows)
    audit_csv = _csv_bytes(AUDIT_COLUMNS, audit_rows)

    # Catch source/correction replacement between validation and publication.
    _require(file_record(source_path) == source_record, "source CSV changed during the run")
    _require(
        file_record(sequence_path) == sequence_record,
        "source retention sequence CSV changed during the run",
    )
    _require(file_record(correction_path) == correction_record, "correction JSON changed during the run")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_created = False
    try:
        # mkdir is the no-clobber publication lock.  Unlike replacing a staging
        # directory with os.rename, it cannot overwrite a concurrently created
        # empty output directory.
        output_dir.mkdir(mode=0o700)
        output_created = True
        os.chmod(str(output_dir), 0o700)
        retention_output = output_dir / "retention_matrix.csv"
        evaluation_output = output_dir / "libb_evaluation_panel.csv"
        audit_output = output_dir / "change_audit.csv"
        manifest_output = output_dir / "manifest.json"
        _write_file_exclusive(retention_output, corrected_csv)
        _write_file_exclusive(evaluation_output, evaluation_csv)
        _write_file_exclusive(audit_output, audit_csv)

        outputs = {
            retention_output.name: {
                "sha256": sha256_file(retention_output),
                "bytes": int(retention_output.stat().st_size),
            },
            evaluation_output.name: {
                "sha256": sha256_file(evaluation_output),
                "bytes": int(evaluation_output.stat().st_size),
                "rows": len(evaluation_rows),
                "columns": list(EVALUATION_PANEL_COLUMNS),
            },
            audit_output.name: {
                "sha256": sha256_file(audit_output),
                "bytes": int(audit_output.stat().st_size),
                "rows": len(audit_rows),
            },
        }
        manifest = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": float(time.time() - started),
            "sources": {
                "retention_matrix": source_record,
                "retention_sequences": sequence_record,
                "provider_correction_json": correction_record,
            },
            "code": {"script": script_record},
            "correction": {
                "target": dict(TARGET_IDENTITY),
                "provisional_matrix_key": TARGET_MATRIX_KEY,
                "retention_percent": "87.94",
                "binder_definition": "retention_percent >= 75",
                "binder_label_ge_75": "1",
                "measurement_missing": "0",
                "timepoint_status": TIMEPOINT_STATUS,
                "provider_notice": correction["provider_notice"],
                "changed_fields": list(CHANGED_FIELDS),
                "sequence_evaluation_changed_fields": list(SEQUENCE_CHANGED_FIELDS),
                "non_target_physical_rows_preserved_byte_for_byte": True,
                "non_target_cell_values_preserved": True,
                "all_sequence_and_id_fields_preserved": True,
            },
            "counts": summary,
            "normalized_libb_evaluation_panel": {
                "rows": 120,
                "peptides": 12,
                "affibodies": 10,
                "binders_retention_ge_75": 61,
                "nonbinders_retention_lt_75": 59,
                "id_column": "pair_uid",
                "compatible_consumer": "recompute_libb_metrics_120.py",
            },
            "outputs": outputs,
            "permissions": {"directory": "0700", "files": "0600"},
            "training_performed": False,
        }
        _write_file_exclusive(manifest_output, _json_bytes(manifest))
    except BaseException:
        if output_created and output_dir.exists():
            shutil.rmtree(str(output_dir))
        raise

    result = {
        "output_dir": str(output_dir),
        "retention_matrix_sha256": outputs["retention_matrix.csv"]["sha256"],
        "libb_evaluation_panel_sha256": outputs["libb_evaluation_panel.csv"]["sha256"],
        "change_audit_sha256": outputs["change_audit.csv"]["sha256"],
        "counts": summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run(parse_args())
