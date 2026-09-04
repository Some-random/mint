import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from downstream.AffibodyMHC import build_esmfold2_libb_canonical_rows as rows


SMALL_EXPECTED = {
    "source_total": 8,
    "source_libb_candidate": 4,
    "source_libb_matrix": 3,
    "train": 3,
    "train_positive": 2,
    "train_negative": 1,
    "eval": 2,
    "eval_positive": 1,
    "eval_negative": 1,
    "total": 5,
    "train_unique_chain1": 3,
    "train_unique_chain2": 3,
    "eval_unique_chain1": 2,
    "eval_unique_chain2": 2,
}

SMALL_REVISED_EXPECTED = {
    **SMALL_EXPECTED,
    "eval": 3,
    "eval_positive": 2,
    "total": 6,
    "eval_unique_chain1": 3,
    "eval_unique_chain2": 3,
}


def _hash(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _chains(peptide_code, affibody_code):
    chain1 = list("A" * rows.CHAIN1_LENGTH)
    chain1[-rows.PEPTIDE_LENGTH :] = "SLL{}ITQV".format(peptide_code)
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
        "measurement_missing": "0" if measured else ("1" if source_kind == "retention" else ""),
        "target_retention": target_retention,
        "target_binder": target_binder,
        "chain1_smart_hla_linker_peptide_sequence": chain1,
        "chain2_affibody_sequence": chain2,
        "chain1_sha256": _hash(chain1),
        "chain2_sha256": _hash(chain2),
        "sequence_pair_sha256": _hash(chain1 + "|" + chain2),
    }


def _small_source():
    records = [
        _source_row("weak", "LibB", "AA", "AAAAA", "p1", "1", "1", "0", "0"),
        _source_row("weak", "LibB", "CD", "CDEFG", "p2", "1", "1", "0", "0"),
        _source_row("weak", "LibB", "EF", "GHIKL", "n1", "0", "1", "0", "0"),
        # This row shares the evaluation peptide and must be rejected by the
        # exact upstream library-local strict flag rather than recomputed here.
        _source_row("weak", "LibB", "GH", "AAAAA", "excluded", "1", "0", "1", "0"),
        _source_row(
            "retention", "LibB", "GH", "MNPQR", "e1", measured=True,
            target_retention="88.5", target_binder="1"
        ),
        _source_row(
            "retention", "LibB", "IK", "STVWY", "e2", measured=True,
            target_retention="42.0", target_binder="0"
        ),
        _source_row("retention", "LibB", "LM", "ACDEF", "missing", measured=False),
        _source_row("weak", "LibA", "NP", "FGHIK", "other", "1", "1", "0", "0"),
    ]
    return pd.DataFrame(records)


def _write_source(tmp_path, source):
    source_path = tmp_path / "source_rows.csv"
    source.to_csv(source_path, index=False)
    source_sha = rows.sha256_file(source_path)
    manifest = {
        "schema_version": rows.SOURCE_SCHEMA_VERSION,
        "stage": "prepare",
        "sources": {
            "sequence_zip": {
                "sha256": "archive-hash",
                "template_deck_sha256": "deck-hash",
            }
        },
        "output": {"rows_csv": {"sha256": source_sha}},
    }
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source_path, manifest_path


def _corrected_panel(source):
    panel = source.loc[
        source["source_kind"].eq("retention") & source["library"].eq("LibB"),
        list(rows.CORRECTED_EVALUATION_COLUMNS),
    ].copy()
    # ``measurement_missing`` is intentionally not part of the provider-panel
    # schema; complete all direct outcomes through the two authoritative fields.
    index = panel["target_retention"].eq("")
    panel.loc[index, "target_retention"] = "80.0"
    panel.loc[index, "target_binder"] = "1"
    return panel


def test_build_tables_uses_exact_upstream_flag_and_separates_all_labels():
    source = _small_source()
    extractor, training, evaluation = rows.build_tables(source, expected=SMALL_EXPECTED)

    assert tuple(extractor.columns) == rows.EXTRACTOR_COLUMNS
    assert extractor["row_index"].tolist() == list(range(5))
    assert extractor["split"].tolist() == ["train", "train", "train", "eval", "eval"]
    assert extractor["row_id"].nunique() == 5
    assert extractor["chain1_sequence"].map(len).eq(270).all()
    assert extractor["chain2_sequence"].map(len).eq(58).all()
    assert (
        extractor["sequence_pair_sha256"]
        == [
            _hash(left + "|" + right)
            for left, right in zip(
                extractor["chain1_sequence"], extractor["chain2_sequence"]
            )
        ]
    ).all()
    selected_pair_ids = set(
        source.loc[
            source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("1")
            & source["library"].eq("LibB"),
            "pair_uid",
        ]
    ).union(
        set(
            source.loc[
                source["source_kind"].eq("retention")
                & source["library"].eq("LibB")
                & source["measurement_missing"].eq("0"),
                "pair_uid",
            ]
        )
    )
    assert set(extractor["row_id"]) == selected_pair_ids
    excluded_pair_id = source.loc[source["peptide_design_code"].eq("GH") & source["source_kind"].eq("weak"), "pair_uid"].iloc[0]
    assert excluded_pair_id not in set(extractor["row_id"])

    assert tuple(training.columns) == rows.TRAINING_LABEL_COLUMNS
    assert training["weak_label"].value_counts().to_dict() == {"1": 2, "0": 1}
    assert training["peptide_id"].str.len().gt(0).all()
    assert training["affibody_id"].str.len().gt(0).all()
    assert training["chain1_sha256"].str.len().eq(64).all()
    assert training["chain2_sha256"].str.len().eq(64).all()
    assert set(training["row_id"]) == set(
        extractor.loc[extractor.split.eq("train"), "row_id"]
    )
    assert tuple(evaluation.columns) == rows.EVALUATION_LABEL_COLUMNS
    assert set(evaluation["target_retention"]) == {"88.5", "42.0"}
    assert set(evaluation["peptide_design_code"]) == {"GH", "IK"}
    assert set(evaluation["affibody_design_code"]) == {"MNPQR", "STVWY"}
    assert set(evaluation["target_binder"]) == {"0", "1"}
    assert set(evaluation["row_id"]) == set(
        extractor.loc[extractor.split.eq("eval"), "row_id"]
    )
    assert set(training["row_id"]).isdisjoint(set(evaluation["row_id"]))
    rows.assert_extractor_rows_are_label_free(extractor)


def test_corrected_panel_adds_only_the_missing_outcome_and_preserves_identities():
    source = _small_source()
    corrected = _corrected_panel(source)
    updated = rows.apply_corrected_evaluation_panel(source, corrected)
    extractor, training, evaluation = rows.build_tables(
        updated, expected=SMALL_REVISED_EXPECTED
    )

    assert len(extractor) == 6
    assert len(training) == 3
    assert len(evaluation) == 3
    assert evaluation["target_binder"].value_counts().to_dict() == {"1": 2, "0": 1}
    missing_pair = source.loc[
        source["source_kind"].eq("retention")
        & source["measurement_missing"].eq("1"),
        "pair_uid",
    ].item()
    assert missing_pair in set(evaluation["row_id"])
    identity_columns = [
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "sequence_pair_sha256",
    ]
    pd.testing.assert_frame_equal(
        source[identity_columns].reset_index(drop=True),
        updated[identity_columns].reset_index(drop=True),
    )


def test_corrected_panel_rejects_any_sequence_identity_change():
    source = _small_source()
    corrected = _corrected_panel(source)
    corrected.loc[corrected.index[0], "chain2_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed stable identity/sequence"):
        rows.apply_corrected_evaluation_panel(source, corrected)


def test_extractor_recursive_guard_rejects_nested_outcomes_and_selection_fields():
    with pytest.raises(ValueError, match="forbidden term"):
        rows.assert_extractor_payload_is_label_free(
            {"safe": {"nested": {"target_binder": "1"}}}
        )
    with pytest.raises(ValueError, match="forbidden term"):
        rows.assert_extractor_payload_is_label_free(
            {"safe": ["selection count is hidden"]}
        )

    extractor, _, _ = rows.build_tables(_small_source(), expected=SMALL_EXPECTED)
    tampered = extractor.copy()
    tampered["r009_count"] = "0"
    with pytest.raises(ValueError, match="schema mismatch"):
        rows.assert_extractor_rows_are_label_free(tampered)


def test_mapping_tampering_is_rejected():
    source = _small_source()
    selected = source.index[
        source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("1")
        & source["library"].eq("LibB")
    ][0]
    chain1 = list(source.loc[selected, "chain1_smart_hla_linker_peptide_sequence"])
    chain1[264] = "Y"
    source.loc[selected, "chain1_smart_hla_linker_peptide_sequence"] = "".join(chain1)
    source.loc[selected, "chain1_sha256"] = _hash("".join(chain1))
    source.loc[selected, "sequence_pair_sha256"] = _hash(
        "".join(chain1) + "|" + source.loc[selected, "chain2_affibody_sequence"]
    )

    with pytest.raises(ValueError, match="peptide code-to-chain mapping mismatch"):
        rows.build_tables(source, expected=SMALL_EXPECTED)


def test_partner_identity_overlap_is_rejected_even_if_strict_flag_claims_eligible():
    source = _small_source()
    train_index = source.index[
        source["source_kind"].eq("weak")
        & source["library"].eq("LibB")
        & source["upstream_library_local_strict_retention_identity_cold_eligible"].eq("1")
    ][0]
    eval_index = source.index[
        source["source_kind"].eq("retention")
        & source["library"].eq("LibB")
        & source["measurement_missing"].eq("0")
    ][0]
    source.loc[train_index, "peptide_uid"] = source.loc[eval_index, "peptide_uid"]

    with pytest.raises(ValueError, match="identity overlap in peptide_uid"):
        rows.build_tables(source, expected=SMALL_EXPECTED)


def test_run_writes_three_disjoint_private_artifacts_and_label_free_manifest(
    tmp_path, monkeypatch
):
    source_path, source_manifest_path = _write_source(tmp_path, _small_source())
    corrected_panel_path = tmp_path / "corrected_panel.csv"
    corrected_manifest_path = tmp_path / "corrected_manifest.json"
    corrected_panel = _corrected_panel(_small_source())
    corrected_panel.to_csv(corrected_panel_path, index=False)
    corrected_manifest_path.write_text("{}", encoding="utf-8")
    extractor_dir = tmp_path / "extractor"
    training_dir = tmp_path / "training"
    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr(rows, "EXPECTED", dict(SMALL_REVISED_EXPECTED))
    monkeypatch.setattr(
        rows,
        "load_corrected_evaluation_panel",
        lambda *args: (corrected_panel, {"provider_correction_panel_sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        rows, "validate_private_output_path", lambda path, repo_root=rows.REPO_ROOT: Path(path).resolve()
    )

    rows.run_build(
        Namespace(
            cache_rows=source_path,
            cache_rows_manifest=source_manifest_path,
            corrected_evaluation_panel=corrected_panel_path,
            corrected_evaluation_manifest=corrected_manifest_path,
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
    assert training_path.is_file()
    assert evaluation_path.is_file()
    assert os.stat(extractor_dir).st_mode & 0o777 == 0o700
    assert os.stat(extractor_path).st_mode & 0o777 == 0o600

    extractor_payload = json.loads(extractor_path.read_text(encoding="utf-8"))
    assert set(extractor_payload) == {"schema_version", "rows"}
    assert extractor_payload["schema_version"] == rows.SCHEMA_VERSION
    assert all(
        set(record) == set(rows.EXTRACTOR_COLUMNS)
        for record in extractor_payload["rows"]
    )
    extractor = pd.DataFrame(extractor_payload["rows"])
    manifest = json.loads(extractor_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == rows.AUDIT_SCHEMA_VERSION
    assert manifest["artifact"]["filename"] == rows.ROWS_FILENAME
    assert manifest["artifact"]["sha256"] == rows.sha256_file(extractor_path)
    rows.assert_extractor_rows_are_label_free(extractor)
    rows.assert_extractor_payload_is_label_free(extractor_payload)
    rows.assert_extractor_payload_is_label_free(manifest)
    serialized = extractor_path.read_text(encoding="utf-8") + extractor_manifest_path.read_text(
        encoding="utf-8"
    )
    assert "88.5" not in serialized
    assert "42.0" not in serialized
    assert "75.0" not in serialized
    assert "weak_label" not in serialized
    assert "target_binder" not in serialized

    training = rows.read_string_csv(training_path)
    evaluation = rows.read_string_csv(evaluation_path)
    assert set(training.columns) == set(rows.TRAINING_LABEL_COLUMNS)
    assert set(evaluation.columns) == set(rows.EVALUATION_LABEL_COLUMNS)
    assert set(training.row_id).isdisjoint(set(evaluation.row_id))
    evaluation_manifest = json.loads(
        (evaluation_dir / rows.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert evaluation_manifest["sealed_evaluation"]["binder_threshold"] == 75.0


def test_source_manifest_must_bind_exact_cache_rows(tmp_path):
    source_path, source_manifest_path = _write_source(tmp_path, _small_source())
    with open(str(source_path), "a") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="source row hash disagrees"):
        rows.validate_source_lineage(source_path, source_manifest_path)


def test_real_private_cache_reproduces_published_contract_if_present():
    source_path = (
        rows.REPO_ROOT
        / "private_data/derived/mint_weak_cache_v1/rows/cache_rows.csv"
    )
    if not source_path.is_file():
        pytest.skip("private canonical cache is not available")

    corrected_path = (
        rows.REPO_ROOT
        / "private_data/derived/retention_panel_provider_revision_2026-09-03_v2/libb_evaluation_panel.csv"
    )
    corrected_manifest = corrected_path.with_name("manifest.json")
    if not (corrected_path.is_file() and corrected_manifest.is_file()):
        pytest.skip("corrected private LibB panel is not available")
    corrected, _ = rows.load_corrected_evaluation_panel(
        corrected_path, corrected_manifest
    )
    source = rows.apply_corrected_evaluation_panel(
        rows.read_string_csv(source_path), corrected
    )
    extractor, training, evaluation = rows.build_tables(source)

    assert len(extractor) == 30768
    assert extractor["row_index"].tolist() == list(range(30768))
    assert extractor["split"].value_counts().to_dict() == {
        "train": 30648,
        "eval": 120,
    }
    assert training["weak_label"].value_counts().to_dict() == {
        "1": 23725,
        "0": 6923,
    }
    assert evaluation["target_binder"].value_counts().to_dict() == {
        "1": 61,
        "0": 59,
    }
    train = extractor.loc[extractor.split.eq("train")]
    eval_rows = extractor.loc[extractor.split.eq("eval")]
    assert set(train.chain1_sequence).isdisjoint(set(eval_rows.chain1_sequence))
    assert set(train.chain2_sequence).isdisjoint(set(eval_rows.chain2_sequence))
    assert set(training.row_id) == set(train.row_id)
    assert set(evaluation.row_id) == set(eval_rows.row_id)
    assert set(extractor.row_id).issubset(
        set(rows.read_string_csv(source_path).pair_uid)
    )
    rows.assert_extractor_rows_are_label_free(extractor)
