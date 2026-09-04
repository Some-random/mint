import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from downstream.AffibodyMHC import build_libb_fixed_crystal_contract as contract


REFERENCE_AFFIBODY = (
    "VDNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"
)
SMALL_EXPECTED = {"train": 2, "eval": 2, "total": 4}


def _affibody_with_code(code):
    sequence = list(REFERENCE_AFFIBODY)
    for position, amino_acid in zip(contract.AFFIBODY_DESIGN_POSITIONS, code):
        sequence[position - 1] = amino_acid
    return "".join(sequence)


def _chain1_with_code(code):
    return "A" * 261 + "SLL{}ITQV".format(code)


def _source_row(index, split, peptide_code, affibody_code):
    chain1 = _chain1_with_code(peptide_code)
    chain2 = _affibody_with_code(affibody_code)
    return {
        "row_index": index,
        "row_id": hashlib.sha256(
            "{}|{}|{}".format(split, peptide_code, affibody_code).encode("ascii")
        ).hexdigest()[:20],
        "split": split,
        "chain1_sequence": chain1,
        "chain2_sequence": chain2,
        "sequence_pair_sha256": contract.sha256_text(chain1 + "|" + chain2),
    }


def _small_rows():
    return [
        _source_row(0, "train", "AA", "AAAAA"),
        _source_row(1, "train", "CC", "CCCCC"),
        _source_row(2, "eval", "MW", "NNYYF"),
        _source_row(3, "eval", "DD", "DDDDD"),
    ]


def _fake_mapping():
    return {
        "residue_mappings": {
            "peptide": [
                {"sequence_position_1_based": position}
                for position in range(1, 10)
            ],
            "affibody": [
                {"sequence_position_1_based": position}
                for position in range(3, 58)
            ],
        }
    }


def _write_canonical_artifact(tmp_path, rows):
    rows_path = tmp_path / "rows.json"
    manifest_path = tmp_path / "source_manifest.json"
    payload = {
        "schema_version": contract.SOURCE_ROWS_SCHEMA_VERSION,
        "rows": rows,
    }
    rows_path.write_bytes(contract._json_bytes(payload))
    manifest = {
        "schema_version": contract.SOURCE_MANIFEST_SCHEMA_VERSION,
        "split_policy": "library_local_strict_partner_disjoint",
        "partner_isolation": {
            "peptide_identity_overlap": 0,
            "affibody_identity_overlap": 0,
            "sequence_pair_overlap": 0,
        },
        "artifact": {"sha256": contract.sha256_file(rows_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return rows_path, manifest_path


def test_normalize_rows_derives_codes_and_enforces_strict_partner_split():
    records, reference, audit = contract.validate_and_normalize_rows(
        _small_rows(), expected=SMALL_EXPECTED
    )

    assert reference.peptide_sequence == "SLLMWITQV"
    assert reference.affibody_sequence == REFERENCE_AFFIBODY
    assert reference.split == "eval"
    assert records[0]["peptide_code"] == "AA"
    assert records[0]["affibody_code"] == "AAAAA"
    assert audit["partner_overlap"] == {
        "peptide_sequence_overlap": 0,
        "affibody_sequence_overlap": 0,
        "sequence_pair_overlap": 0,
        "row_id_overlap": 0,
    }

    mapped = contract.apply_residue_mapping(records, _fake_mapping())
    assert mapped[0]["peptide_resolved_sequence"] == "SLLAAITQV"
    assert mapped[0]["affibody_resolved_sequence"] == records[0][
        "affibody_sequence"
    ][2:57]
    assert all(set(record) == set(contract.CURRENT_RECORD_FIELDS) for record in mapped)


def test_normalize_rows_rejects_partner_overlap_even_with_different_pair():
    rows = _small_rows()
    rows[3] = _source_row(3, "eval", "AA", "DDDDD")

    with pytest.raises(ValueError, match="strict partner split is violated"):
        contract.validate_and_normalize_rows(rows, expected=SMALL_EXPECTED)


def test_normalize_rows_rejects_any_extra_outcome_field():
    rows = _small_rows()
    rows[0]["target_retention"] = "99.0"

    with pytest.raises(ValueError, match="non-sequence material"):
        contract.validate_and_normalize_rows(rows, expected=SMALL_EXPECTED)


def test_load_canonical_rows_requires_hash_and_zero_declared_overlap(tmp_path):
    rows_path, manifest_path = _write_canonical_artifact(tmp_path, _small_rows())
    loaded, lineage = contract.load_canonical_rows(rows_path, manifest_path)

    assert loaded == _small_rows()
    assert lineage["canonical_rows_sha256"] == contract.sha256_file(rows_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partner_isolation"]["peptide_identity_overlap"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="upstream isolation check failed"):
        contract.load_canonical_rows(rows_path, manifest_path)


def test_primary_copy_is_selected_by_completeness_not_list_order():
    chains = {
        "A": SimpleNamespace(residues=[None] * 276),
        "B": SimpleNamespace(residues=[None] * 99),
        "P": SimpleNamespace(residues=[None] * 9),
        "H": SimpleNamespace(residues=[None] * 55),
        "C": SimpleNamespace(residues=[None] * 275),
        "D": SimpleNamespace(residues=[None] * 99),
        "E": SimpleNamespace(residues=[None] * 9),
        "I": SimpleNamespace(residues=[None] * 54),
    }
    incomplete = {
        "hla_chain": "C",
        "beta2m_chain": "D",
        "peptide_chain": "E",
        "affibody_chain": "I",
    }
    complete = {
        "hla_chain": "A",
        "beta2m_chain": "B",
        "peptide_chain": "P",
        "affibody_chain": "H",
    }

    selected, score = contract.select_primary_crystal_copy(
        [incomplete, complete], chains
    )

    assert selected == complete
    assert score == 439


def test_assay_hla_difference_contract_allows_only_y84a_and_w167a():
    pdb_sequence = list("A" * contract.ASSAY_HLA_ALIGNED_LENGTH)
    pdb_sequence[83] = "Y"
    pdb_sequence[166] = "W"
    assay_sequence = "A" * contract.ASSAY_HLA_ALIGNED_LENGTH

    assert contract.validate_assay_hla_identity_differences(
        "".join(pdb_sequence), assay_sequence
    ) == ((84, "Y", "A"), (167, "W", "A"))

    tampered = list(assay_sequence)
    tampered[9] = "C"
    with pytest.raises(ValueError, match="assay/PDB HLA differences changed"):
        contract.validate_assay_hla_identity_differences(
            "".join(pdb_sequence), "".join(tampered)
        )


def test_real_private_contract_reproduces_counts_mapping_and_distances_if_present():
    rows_path = (
        contract.REPO_ROOT
        / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json"
    )
    manifest_path = rows_path.with_name("manifest.json")
    pdb_path = contract.REPO_ROOT / "nyeso_xx133_complex.pdb"
    if not (rows_path.is_file() and manifest_path.is_file() and pdb_path.is_file()):
        pytest.skip("private canonical rows or LibB crystal are unavailable")

    mapping, records_payload, _, audit = contract.build_contract(
        rows_path, manifest_path, pdb_path
    )

    assert audit["train"] == 30648
    assert audit["eval"] == 120
    assert audit["partner_overlap"] == {
        "peptide_sequence_overlap": 0,
        "affibody_sequence_overlap": 0,
        "sequence_pair_overlap": 0,
        "row_id_overlap": 0,
    }
    assert mapping["chain_identification"]["selected_copy"] == {
        "hla_chain": "A",
        "beta2m_chain": "B",
        "peptide_chain": "P",
        "affibody_chain": "H",
    }
    assert mapping["reference_pair"]["peptide_code"] == "MW"
    assert mapping["reference_pair"]["affibody_code"] == "NNYYF"
    peptide_sites = {
        row["sequence_position_1_based"]: row["pdb_residue_id"]
        for row in mapping["residue_mappings"]["peptide"]
        if row["mutable_site"]
    }
    affibody_sites = {
        row["sequence_position_1_based"]: row["pdb_residue_id"]
        for row in mapping["residue_mappings"]["affibody"]
        if row["mutable_site"]
    }
    assert peptide_sites == {4: "P:4", 5: "P:5"}
    assert affibody_sites == {
        6: "H:8",
        10: "H:12",
        13: "H:15",
        14: "H:16",
        17: "H:19",
    }
    hla_mapping = mapping["residue_mappings"]["assay_hla"]
    assert hla_mapping["canonical_chain1_hla_start_index_0_based"] == 70
    assert hla_mapping["aligned_hla_residue_count"] == 181
    assert hla_mapping["exact_identity_match_count"] == 179
    assert hla_mapping["identity_override_count"] == 2
    assert [
        row["substitution"] for row in hla_mapping["identity_overrides"]
    ] == ["Y84A", "W167A"]
    assert [
        row["pdb_residue_id"] for row in hla_mapping["identity_overrides"]
    ] == ["A:84", "A:167"]
    assert [
        row["canonical_chain1_index_0_based"]
        for row in hla_mapping["identity_overrides"]
    ] == [153, 236]
    assert all(
        row["model_amino_acid"] == "A"
        and row["coordinate_policy"] == "keep PDB coordinates unchanged"
        for row in hla_mapping["identity_overrides"]
    )
    observed = [
        check["observed_distance_angstrom"]
        for check in mapping["known_distance_checks"]
    ]
    assert observed == pytest.approx([6.9341, 7.8235, 5.9667], abs=5e-5)
    assert len(records_payload["rows"]) == 30768
    assert sum(row["split"] == "train" for row in records_payload["rows"]) == 30648
    serialized = json.dumps({"mapping": mapping, "records": records_payload})
    assert "target_retention" not in serialized
    assert "target_binder" not in serialized
    assert "weak_label" not in serialized


def test_run_build_writes_private_machine_readable_contract(tmp_path, monkeypatch):
    output_dir = tmp_path / "contract"
    mapping = {
        "pdb_source": {"sha256": "a" * 64},
        "chain_identification": {
            "selected_copy": dict(contract.EXPECTED_PRIMARY_CHAIN_IDS)
        },
        "known_distance_checks": [{}, {}, {}],
        "residue_mappings": {
            "assay_hla": {
                "canonical_chain1_hla_start_index_0_based": 70,
                "aligned_hla_residue_count": 181,
                "pdb_chain_id": "A",
                "exact_identity_match_count": 179,
                "identity_override_count": 2,
                "identity_overrides": [
                    {"substitution": "Y84A"},
                    {"substitution": "W167A"},
                ],
            }
        },
    }
    mapping_sha256 = hashlib.sha256(contract._json_bytes(mapping)).hexdigest()
    records_payload = {
        "schema_version": contract.SCHEMA_VERSION,
        "residue_mapping_file_sha256": mapping_sha256,
        "rows": [{"row_id": str(index)} for index in range(4)],
    }
    audit = {
        "train": 2,
        "eval": 2,
        "total": 4,
        "partner_overlap": {
            "peptide_sequence_overlap": 0,
            "affibody_sequence_overlap": 0,
            "sequence_pair_overlap": 0,
            "row_id_overlap": 0,
        },
    }
    monkeypatch.setattr(
        contract,
        "build_contract",
        lambda *args, **kwargs: (mapping, records_payload, {"source": "hash"}, audit),
    )
    monkeypatch.setattr(
        contract,
        "validate_private_output_path",
        lambda path, repo_root=contract.REPO_ROOT: Path(path).resolve(),
    )

    summary = contract.run_build(
        Namespace(
            canonical_rows=tmp_path / "unused-rows.json",
            canonical_rows_manifest=tmp_path / "unused-manifest.json",
            pdb=tmp_path / "unused.pdb",
            output_dir=output_dir,
            check_only=False,
        )
    )

    assert summary["train"] == 2
    assert summary["eval"] == 2
    assert (output_dir / contract.MAPPING_FILENAME).is_file()
    assert (output_dir / contract.RECORDS_FILENAME).is_file()
    assert (output_dir / contract.MANIFEST_FILENAME).is_file()
    manifest = json.loads(
        (output_dir / contract.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["data_access"] == {
        "canonical_sequence_rows_read": True,
        "training_label_table_read": False,
        "evaluation_outcome_table_read": False,
    }
    assert manifest["outputs"][contract.RECORDS_FILENAME]["rows"] == 4
    assert [
        row["substitution"]
        for row in manifest["structure_audit"]["assay_hla_alignment"][
            "identity_overrides"
        ]
    ] == ["Y84A", "W167A"]
    assert oct(output_dir.stat().st_mode & 0o777) == "0o700"
    assert oct((output_dir / contract.MAPPING_FILENAME).stat().st_mode & 0o777) == "0o600"
