import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from downstream.AffibodyMHC import build_corrected_retention_panel as panel


SOURCE = Path(__file__).resolve().parents[1] / "private_data" / "derived" / "retention_matrix.csv"
SEQUENCE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "private_data"
    / "derived"
    / "retention_sequences_v2.csv"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("/private_data/\n", encoding="utf-8")
    (tmp_path / "private_data" / "derived").mkdir(parents=True)
    source = tmp_path / "private_data" / "derived" / "retention_matrix.csv"
    source.write_bytes(SOURCE.read_bytes())
    sequence_source = (
        tmp_path / "private_data" / "derived" / "retention_sequences_v2.csv"
    )
    sequence_source.write_bytes(SEQUENCE_SOURCE.read_bytes())
    return source


def _write_correction(path, source, **changes):
    payload = {
        "schema_version": panel.CORRECTION_SCHEMA_VERSION,
        "source_retention_matrix_sha256": _sha256(source),
        "provider_notice": {
            "provider": "Xinyu",
            "received_date": "2026-09-03",
            "statement": "LibB AH x LIFTK retention = 87.94",
        },
        "correction": {
            "library": "LibB",
            "peptide_design_code": "AH",
            "affibody_design_code": "LIFTK",
            "retention_percent": 87.94,
        },
    }
    for dotted_key, value in changes.items():
        section, key = dotted_key.split("__", 1)
        payload[section][key] = value
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _args(source, correction, output):
    return argparse.Namespace(
        source_csv=source,
        sequence_csv=source.with_name("retention_sequences_v2.csv"),
        correction_json=correction,
        output_dir=output,
    )


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_builds_complete_panel_and_preserves_every_other_row_bytewise(tmp_path):
    source = _make_repo(tmp_path)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source
    )
    output = tmp_path / "private_data" / "derived" / "corrected-v1"

    result = panel.run(_args(source, correction, output), repo_root=tmp_path)

    before_lines = source.read_bytes().splitlines(keepends=True)
    after_lines = (output / "retention_matrix.csv").read_bytes().splitlines(keepends=True)
    assert len(before_lines) == len(after_lines) == 229
    changed_physical_rows = [
        index for index, (before, after) in enumerate(zip(before_lines, after_lines)) if before != after
    ]
    assert changed_physical_rows == [14]

    rows = _rows(output / "retention_matrix.csv")
    target = [
        row
        for row in rows
        if row["library"] == "LibB"
        and row["peptide_design_code"] == "AH"
        and row["affibody_design_code"] == "LIFTK"
    ]
    assert target == [
        {
            **_rows(source)[13],
            "retention_percent": "87.94",
            "timepoint_status": panel.TIMEPOINT_STATUS,
            "binder_label_ge_75": "1",
            "measurement_missing": "0",
        }
    ]
    assert sum(row["library"] == "LibA" and row["measurement_missing"] == "0" for row in rows) == 108
    libb = [row for row in rows if row["library"] == "LibB"]
    assert len(libb) == 120
    assert sum(row["measurement_missing"] == "0" for row in libb) == 120
    assert sum(row["binder_label_ge_75"] == "1" for row in libb) == 61
    assert sum(row["binder_label_ge_75"] == "0" for row in libb) == 59

    sequence_source_rows = _rows(source.with_name("retention_sequences_v2.csv"))
    sequence_by_uid = {row["pair_uid"]: row for row in sequence_source_rows}
    evaluation = _rows(output / "libb_evaluation_panel.csv")
    assert list(evaluation[0]) == list(panel.EVALUATION_PANEL_COLUMNS)
    assert len(evaluation) == 120
    assert len({row["pair_uid"] for row in evaluation}) == 120
    assert sum(row["target_binder"] == "1" for row in evaluation) == 61
    assert sum(row["target_binder"] == "0" for row in evaluation) == 59
    for row in evaluation:
        source_row = sequence_by_uid[row["pair_uid"]]
        for field in set(panel.EVALUATION_PANEL_COLUMNS).difference(
            panel.SEQUENCE_CHANGED_FIELDS
        ):
            assert row[field] == source_row[field]
        is_target = (
            row["peptide_design_code"] == "AH"
            and row["affibody_design_code"] == "LIFTK"
        )
        if is_target:
            assert row["target_retention"] == "87.94"
            assert row["target_binder"] == "1"
        else:
            assert row["target_retention"] == source_row["target_retention"]
            assert row["target_binder"] == source_row["target_binder"]

    audit = _rows(output / "change_audit.csv")
    assert [row["field"] for row in audit] == list(
        panel.CHANGED_FIELDS + panel.SEQUENCE_CHANGED_FIELDS
    )
    assert [row["output_file"] for row in audit] == [
        "retention_matrix.csv",
        "retention_matrix.csv",
        "retention_matrix.csv",
        "retention_matrix.csv",
        "libb_evaluation_panel.csv",
        "libb_evaluation_panel.csv",
    ]
    assert {row["evidence_kind"] for row in audit} == {
        "provider_revision_notice_not_independently_deck_verified"
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == panel.OUTPUT_SCHEMA_VERSION
    assert manifest["sources"]["retention_matrix"]["sha256"] == _sha256(source)
    assert manifest["sources"]["retention_matrix"]["mtime_ns"] == source.stat().st_mtime_ns
    sequence_path = source.with_name("retention_sequences_v2.csv")
    assert manifest["sources"]["retention_sequences"]["sha256"] == _sha256(
        sequence_path
    )
    assert manifest["sources"]["retention_sequences"]["mtime_ns"] == sequence_path.stat().st_mtime_ns
    assert manifest["sources"]["provider_correction_json"]["mtime_ns"] == correction.stat().st_mtime_ns
    assert manifest["code"]["script"]["sha256"] == _sha256(Path(panel.__file__))
    assert manifest["outputs"]["retention_matrix.csv"]["sha256"] == _sha256(
        output / "retention_matrix.csv"
    )
    assert manifest["outputs"]["change_audit.csv"]["sha256"] == _sha256(
        output / "change_audit.csv"
    )
    assert manifest["outputs"]["libb_evaluation_panel.csv"]["sha256"] == _sha256(
        output / "libb_evaluation_panel.csv"
    )
    assert manifest["normalized_libb_evaluation_panel"] == {
        "rows": 120,
        "peptides": 12,
        "affibodies": 10,
        "binders_retention_ge_75": 61,
        "nonbinders_retention_lt_75": 59,
        "id_column": "pair_uid",
        "compatible_consumer": "recompute_libb_metrics_120.py",
    }
    assert manifest["counts"]["libraries"]["LibA"]["measured"] == 108
    assert manifest["counts"]["libraries"]["LibB"] == {
        "matrix_cells": 120,
        "measured": 120,
        "missing": 0,
        "binder_retention_ge_75": 61,
        "nonbinder_retention_lt_75": 59,
    }
    assert manifest["training_performed"] is False
    assert result["retention_matrix_sha256"] == _sha256(output / "retention_matrix.csv")
    assert oct(output.stat().st_mode & 0o7777) == "0o700"
    assert all(
        oct(path.stat().st_mode & 0o7777) == "0o600"
        for path in output.iterdir()
        if path.is_file()
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"correction__retention_percent": 87.93}, "exactly 87.94"),
        ({"correction__peptide_design_code": "AF"}, "unexpected peptide_design_code"),
        ({"provider_notice__provider": "unknown"}, "unexpected correction provider"),
    ],
)
def test_rejects_any_other_provider_correction(tmp_path, change, message):
    source = _make_repo(tmp_path)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source, **change
    )
    output = tmp_path / "private_data" / "derived" / "corrected-v1"

    with pytest.raises(ValueError, match=message):
        panel.run(_args(source, correction, output), repo_root=tmp_path)
    assert not output.exists()


def test_rejects_source_when_exact_old_target_is_not_missing(tmp_path):
    source = _make_repo(tmp_path)
    raw = source.read_bytes().replace(
        b"LibB|LIFTK|AH,LibB,LIFTK,AH,inferred_from_code_length_and_number_of_mutated_positions,,30,unambiguous_in_deck,,1,3",
        b"LibB|LIFTK|AH,LibB,LIFTK,AH,inferred_from_code_length_and_number_of_mutated_positions,87.94,30,provider_revision_notice_not_deck_verified,1,0,3",
    )
    source.write_bytes(raw)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source
    )
    output = tmp_path / "private_data" / "derived" / "corrected-v1"

    with pytest.raises(ValueError, match="old retention value is not missing"):
        panel.run(_args(source, correction, output), repo_root=tmp_path)
    assert not output.exists()


def test_rejects_hash_mismatch_and_existing_output(tmp_path):
    source = _make_repo(tmp_path)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source
    )
    payload = json.loads(correction.read_text(encoding="utf-8"))
    payload["source_retention_matrix_sha256"] = "0" * 64
    correction.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "private_data" / "derived" / "corrected-v1"

    with pytest.raises(ValueError, match="bound to another source"):
        panel.run(_args(source, correction, output), repo_root=tmp_path)
    output.mkdir()
    with pytest.raises(ValueError, match="exists; refusing overwrite"):
        panel.run(_args(source, correction, output), repo_root=tmp_path)


def test_rejects_non_private_or_not_ignored_output(tmp_path):
    source = _make_repo(tmp_path)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source
    )
    outside = tmp_path / "public-output"

    with pytest.raises(ValueError, match="below private_data"):
        panel.run(_args(source, correction, outside), repo_root=tmp_path)


def test_rejects_changed_sequence_or_opaque_id(tmp_path):
    source = _make_repo(tmp_path)
    sequence = source.with_name("retention_sequences_v2.csv")
    rows = _rows(sequence)
    rows[13]["pair_uid"] = "0" * 20
    with sequence.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    correction = _write_correction(
        tmp_path / "private_data" / "provider_correction.json", source
    )
    output = tmp_path / "private_data" / "derived" / "corrected-v1"

    with pytest.raises(ValueError, match="pair_uid does not match"):
        panel.run(_args(source, correction, output), repo_root=tmp_path)
    assert not output.exists()
