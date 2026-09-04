import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from downstream.AffibodyMHC import build_esmfold2_liba_canonical_rows as rows


SMALL_EXPECTED = {
    "source_total": 8,
    "source_liba_candidate": 4,
    "source_liba_matrix": 3,
    "train": 3,
    "train_positive": 2,
    "train_negative": 1,
    "eval": 3,
    "eval_positive": 2,
    "eval_negative": 1,
    "total": 6,
    "train_unique_chain1": 3,
    "train_unique_chain2": 3,
    "eval_unique_chain1": 3,
    "eval_unique_chain2": 3,
}


def _hash(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _chains(peptide_code, affibody_code):
    chain1 = list("A" * rows.CHAIN1_LENGTH)
    chain1[-rows.PEPTIDE_LENGTH:] = "SLL{}ITQV".format(peptide_code)
    chain2 = list("A" * rows.CHAIN2_LENGTH)
    for position, residue in zip(
        rows.AFFIBODY_CODE_POSITIONS_1_BASED, affibody_code
    ):
        chain2[position - 1] = residue
    return "".join(chain1), "".join(chain2)


def _source_row(
    source_kind,
    library,
    peptide_code,
    affibody_code,
    suffix,
    weak_label="",
    strict="",
    shares_peptide="",
    shares_affibody="",
    measured=False,
    target_retention="",
    target_binder="",
):
    chain1, chain2 = _chains(peptide_code, affibody_code)
    pair_uid = rows.opaque_id("pair", library, peptide_code, affibody_code, suffix)
    return {
        "cache_uid": rows.opaque_id("cache", pair_uid),
        "source_kind": source_kind,
        "library": library,
        "pair_uid": pair_uid,
        "peptide_uid": rows.opaque_id(library, "pep", peptide_code),
        "affibody_uid": rows.opaque_id(library, "aff", affibody_code),
        "peptide_design_code": peptide_code,
        "affibody_design_code": affibody_code,
        "weak_label": weak_label,
        "upstream_library_local_shares_retention_peptide": shares_peptide,
        "upstream_library_local_shares_retention_affibody": shares_affibody,
        "upstream_library_local_strict_retention_identity_cold_eligible": strict,
        "measurement_missing": "0" if measured else "",
        "target_retention": target_retention,
        "target_binder": target_binder,
        "chain1_smart_hla_linker_peptide_sequence": chain1,
        "chain2_affibody_sequence": chain2,
        "chain1_sha256": _hash(chain1),
        "chain2_sha256": _hash(chain2),
        "sequence_pair_sha256": _hash(chain1 + "|" + chain2),
    }


def _small_source():
    return pd.DataFrame(
        [
            _source_row("weak", "LibA", "AA", "AAAA", "p1", "1", "1", "0", "0"),
            _source_row("weak", "LibA", "CD", "CDEF", "p2", "1", "1", "0", "0"),
            _source_row("weak", "LibA", "EF", "GHIK", "n1", "0", "1", "0", "0"),
            # Exact upstream strict filtering, rather than local recomputation,
            # excludes this row before the independent overlap checks run.
            _source_row("weak", "LibA", "GH", "LMNP", "excluded", "1", "0", "1", "0"),
            _source_row(
                "retention", "LibA", "GH", "QRST", "e1", measured=True,
                target_retention="88.5", target_binder="1",
            ),
            _source_row(
                "retention", "LibA", "IK", "VWYA", "e2", measured=True,
                target_retention="42.0", target_binder="0",
            ),
            _source_row(
                "retention", "LibA", "LM", "DEFG", "e3", measured=True,
                target_retention="75.0", target_binder="1",
            ),
            _source_row("weak", "LibB", "NP", "FGHI", "other", "1", "1", "0", "0"),
        ]
    )


def _write_source(tmp_path, source):
    source_path = tmp_path / "source_rows.csv"
    source.to_csv(source_path, index=False)
    manifest = {
        "schema_version": rows.SOURCE_SCHEMA_VERSION,
        "stage": "prepare",
        "sources": {
            "sequence_zip": {
                "sha256": "archive-hash",
                "template_deck_sha256": "deck-hash",
            }
        },
        "output": {"rows_csv": {"sha256": rows.sha256_file(source_path)}},
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, manifest_path


def test_mapping_uses_displayed_indices_and_records_revised_labels():
    _, chain2 = _chains("CD", "CDEF")
    assert "".join(
        chain2[position - 1] for position in rows.AFFIBODY_CODE_POSITIONS_1_BASED
    ) == "CDEF"
    assert "".join(
        chain2[position - 1]
        for position in rows.AFFIBODY_CRYSTAL_ALIGNED_LABELS_1_BASED
    ) == "AAAA"
    assert rows.AFFIBODY_CODE_PYTHON_INDICES_0_BASED == (12, 16, 26, 30)
    assert rows.AFFIBODY_CRYSTAL_ALIGNED_LABELS_1_BASED == (15, 19, 29, 33)


def test_build_tables_uses_strict_liba_rows_and_separates_labels():
    source = _small_source()
    extractor, training, evaluation = rows.build_tables(
        source, expected=SMALL_EXPECTED
    )

    assert tuple(extractor.columns) == rows.EXTRACTOR_COLUMNS
    assert extractor["row_index"].tolist() == list(range(6))
    assert extractor["split"].tolist() == ["train"] * 3 + ["eval"] * 3
    assert extractor["chain1_sequence"].map(len).eq(270).all()
    assert extractor["chain2_sequence"].map(len).eq(58).all()
    assert training["weak_label"].value_counts().to_dict() == {"1": 2, "0": 1}
    assert evaluation["target_binder"].value_counts().to_dict() == {"1": 2, "0": 1}
    assert set(training["row_id"]).isdisjoint(set(evaluation["row_id"]))

    excluded = source.loc[
        source["source_kind"].eq("weak")
        & source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("0"),
        "pair_uid",
    ].item()
    assert excluded not in set(extractor["row_id"])
    rows.assert_extractor_rows_are_label_free(extractor)


def test_mapping_tampering_and_revised_index_substitution_are_rejected():
    source = _small_source()
    selected = source.index[
        source["source_kind"].eq("weak")
        & source["library"].eq("LibA")
        & source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("1")
    ][1]
    chain2 = list(source.loc[selected, "chain2_affibody_sequence"])
    # Reproduce the exact dangerous mistake this builder must prevent: place
    # code characters at revised labels as though they indexed the 58-aa input.
    for old_position in rows.AFFIBODY_CODE_POSITIONS_1_BASED:
        chain2[old_position - 1] = "A"
    for revised_position, residue in zip(
        rows.AFFIBODY_CRYSTAL_ALIGNED_LABELS_1_BASED, "CDEF"
    ):
        chain2[revised_position - 1] = residue
    changed = "".join(chain2)
    source.loc[selected, "chain2_affibody_sequence"] = changed
    source.loc[selected, "chain2_sha256"] = _hash(changed)
    source.loc[selected, "sequence_pair_sha256"] = _hash(
        source.loc[selected, "chain1_smart_hla_linker_peptide_sequence"]
        + "|"
        + changed
    )

    with pytest.raises(ValueError, match="Affibody code-to-chain mapping mismatch"):
        rows.build_tables(source, expected=SMALL_EXPECTED)


def test_partner_overlap_is_rejected_even_when_strict_flag_claims_eligible():
    source = _small_source()
    train_index = source.index[
        source["source_kind"].eq("weak")
        & source["library"].eq("LibA")
        & source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("1")
    ][0]
    eval_index = source.index[
        source["source_kind"].eq("retention")
        & source["library"].eq("LibA")
    ][0]
    source.loc[train_index, "peptide_uid"] = source.loc[eval_index, "peptide_uid"]

    with pytest.raises(ValueError, match="identity overlap in peptide_uid"):
        rows.build_tables(source, expected=SMALL_EXPECTED)


def test_incomplete_liba_evaluation_matrix_is_rejected():
    source = _small_source()
    eval_index = source.index[
        source["source_kind"].eq("retention")
        & source["library"].eq("LibA")
    ][0]
    source.loc[eval_index, "measurement_missing"] = "1"
    with pytest.raises(ValueError, match="evaluation matrix is not complete"):
        rows.build_tables(source, expected=SMALL_EXPECTED)


def test_manifest_is_label_free_and_distinguishes_both_numbering_systems():
    extractor, _, _ = rows.build_tables(_small_source(), expected=SMALL_EXPECTED)
    payload = rows.build_rows_payload(extractor)
    manifest = rows.build_extractor_manifest(
        {"source_rows_sha256": "a" * 64}, extractor, "b" * 64, "c" * 64
    )

    assert payload["schema_version"] == rows.SCHEMA_VERSION
    assert set(payload) == {"schema_version", "rows"}
    mapping = manifest["mapping_assertions"]
    assert mapping["affibody_displayed_sequence_positions_1_based"] == [13, 17, 27, 31]
    assert mapping["affibody_python_indices_0_based"] == [12, 16, 26, 30]
    assert mapping["affibody_crystal_aligned_labels_1_based"] == [15, 19, 29, 33]
    assert mapping["hidden_preceding_affibody_residues"] == "MA"
    rows.assert_extractor_payload_is_label_free(payload)
    rows.assert_extractor_payload_is_label_free(manifest)


def test_run_writes_disjoint_private_artifacts_with_label_free_extractor(
    tmp_path, monkeypatch
):
    source_path, source_manifest_path = _write_source(tmp_path, _small_source())
    extractor_dir = tmp_path / "extractor"
    training_dir = tmp_path / "training"
    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr(rows, "EXPECTED", dict(SMALL_EXPECTED))
    monkeypatch.setattr(
        rows,
        "validate_private_output_path",
        lambda path, repo_root=rows.REPO_ROOT: Path(path).resolve(),
    )
    monkeypatch.setattr(
        rows.shared,
        "validate_private_output_path",
        lambda path, repo_root=rows.REPO_ROOT: Path(path).resolve(),
    )

    rows.run_build(
        Namespace(
            cache_rows=source_path,
            cache_rows_manifest=source_manifest_path,
            extractor_output_dir=extractor_dir,
            training_output_dir=training_dir,
            evaluation_output_dir=evaluation_dir,
        )
    )

    extractor_path = extractor_dir / rows.ROWS_FILENAME
    extractor_manifest_path = extractor_dir / rows.MANIFEST_FILENAME
    training_path = training_dir / rows.TRAINING_LABELS_FILENAME
    evaluation_path = evaluation_dir / rows.EVALUATION_LABELS_FILENAME
    assert extractor_path.is_file()
    assert extractor_manifest_path.is_file()
    assert training_path.is_file()
    assert evaluation_path.is_file()
    assert os.stat(extractor_dir).st_mode & 0o777 == 0o700
    assert os.stat(extractor_path).st_mode & 0o777 == 0o600

    serialized = extractor_path.read_text(encoding="utf-8") + extractor_manifest_path.read_text(
        encoding="utf-8"
    )
    assert "88.5" not in serialized
    assert "42.0" not in serialized
    assert "75.0" not in serialized
    assert "weak_label" not in serialized
    assert "target_binder" not in serialized
    rows.assert_extractor_payload_is_label_free(
        json.loads(extractor_path.read_text(encoding="utf-8"))
    )
    rows.assert_extractor_payload_is_label_free(
        json.loads(extractor_manifest_path.read_text(encoding="utf-8"))
    )


def test_real_private_cache_reproduces_exact_liba_contract_if_present():
    source_path = (
        rows.REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv"
    )
    if not source_path.is_file():
        pytest.skip("private canonical cache is not available")

    source = rows.read_string_csv(source_path)
    extractor, training, evaluation = rows.build_tables(source)
    assert len(extractor) == 22650
    assert extractor["row_index"].tolist() == list(range(22650))
    assert extractor["split"].value_counts().to_dict() == {
        "train": 22542,
        "eval": 108,
    }
    assert training["weak_label"].value_counts().to_dict() == {
        "1": 11320,
        "0": 11222,
    }
    assert evaluation["target_binder"].value_counts().to_dict() == {
        "0": 70,
        "1": 38,
    }
    train = extractor.loc[extractor.split.eq("train")]
    eval_rows = extractor.loc[extractor.split.eq("eval")]
    assert train["chain1_sequence"].nunique() == 216
    assert train["chain2_sequence"].nunique() == 13590
    assert eval_rows["chain1_sequence"].nunique() == 9
    assert eval_rows["chain2_sequence"].nunique() == 12
    assert set(train.chain1_sequence).isdisjoint(set(eval_rows.chain1_sequence))
    assert set(train.chain2_sequence).isdisjoint(set(eval_rows.chain2_sequence))
    rows.assert_extractor_rows_are_label_free(extractor)
