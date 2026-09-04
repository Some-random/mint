import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from downstream.AffibodyMHC import stab_libb_current_pair_features as stab
from downstream.AffibodyMHC import merge_stab_libb_current_pair_features as merger


REFERENCE_AFFIBODY = "VDNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"


def _row(index=0, split="train", peptide="SLLMWITQV", affibody=REFERENCE_AFFIBODY):
    chain1 = "C" * 70 + "A" * 181 + stab.ASSAY_LINKER_SEQUENCE + peptide
    pair_hash = hashlib.sha256((chain1 + "|" + affibody).encode("ascii")).hexdigest()
    return stab.CanonicalRow(
        index, f"opaque-{index}", split, chain1, affibody, pair_hash
    )


def _template():
    a = "A" * 276
    b = "B" * 99
    p = "SLLMWITQV"
    h = REFERENCE_AFFIBODY[2:57]
    return stab.TemplateBundle(
        context_mode=stab.NATIVE_CRYSTAL_CONTEXT,
        complex_domain={},
        pmhc_domain={},
        affibody_domain={},
        complex_reference_sequence=a + b + p + h,
        pmhc_reference_sequence=a + b + p,
        affibody_reference_sequence=h,
        assay_hla_segment=a[:181],
        structure_mapping_sha256=stab.STRUCTURE_MAPPING_SHA256,
        complex_residue_keys=tuple(
            [f"A:{x}" for x in range(1, 277)]
            + [f"B:{x}" for x in range(1, 100)]
            + [f"P:{x}" for x in range(1, 10)]
            + [f"H:{x}" for x in range(5, 60)]
        ),
        pmhc_residue_keys=tuple(
            [f"A:{x}" for x in range(1, 277)]
            + [f"B:{x}" for x in range(1, 100)]
            + [f"P:{x}" for x in range(1, 10)]
        ),
        affibody_residue_keys=tuple(f"H:{x}" for x in range(5, 60)),
        site_residue_keys=("P:4", "P:5", "H:8", "H:12", "H:15", "H:16", "H:19"),
        complex_site_indices=(378, 379, 387, 391, 394, 395, 398),
        isolated_site_indices=(378, 379, 3, 7, 10, 11, 14),
        peptide_complex_slice=slice(375, 384),
        affibody_complex_slice=slice(384, 439),
    )


def test_current_pair_mapping_changes_only_seven_designed_residues():
    template = _template()
    row = _row()
    complex_sequence, pmhc, affibody = stab.current_pair_sequences(row, template)
    assert len(complex_sequence) == 439
    assert len(pmhc) == 384
    assert len(affibody) == 55
    assert complex_sequence[375:384] == "SLLMWITQV"
    assert complex_sequence[384:] == REFERENCE_AFFIBODY[2:57]
    assert template.complex_site_indices == (378, 379, 387, 391, 394, 395, 398)

    peptide = "SLLAFITQV"
    affibody_full = list(REFERENCE_AFFIBODY)
    for position, amino_acid in zip(stab.AFFIBODY_DESIGNED_POSITIONS, "LIFTK"):
        affibody_full[position - 1] = amino_acid
    variant = _row(peptide=peptide, affibody="".join(affibody_full))
    variant_complex, _, _ = stab.current_pair_sequences(variant, template)
    changed = {
        index
        for index, (reference, observed) in enumerate(
            zip(complex_sequence, variant_complex)
        )
        if reference != observed
    }
    assert changed == set(template.complex_site_indices)


def test_provider_revision_mapping_keeps_indices_and_updates_names():
    assert stab.AFFIBODY_DESIGNED_POSITIONS == (6, 10, 13, 14, 17)
    assert stab.AFFIBODY_PYTHON_INDICES == (5, 9, 12, 13, 16)
    assert stab.AFFIBODY_CRYSTAL_ALIGNED_POSITIONS == (8, 12, 15, 16, 19)
    assert stab.SITE_LABELS[2:] == tuple(
        f"affibody_displayed_position_{displayed}_crystal_position_{crystal}"
        for displayed, crystal in zip(
            stab.AFFIBODY_DESIGNED_POSITIONS,
            stab.AFFIBODY_CRYSTAL_ALIGNED_POSITIONS,
        )
    )


def test_provider_revision_rows_are_complete_and_keep_crystal_pair():
    row_manifest = (
        stab.PRIVATE_ROOT
        / "derived"
        / "esmfold2_libb_canonical_rows_provider_revision_120_v1"
        / "rows.json"
    )
    _, rows = stab.load_canonical_rows(row_manifest)
    assert len(rows) == 30_768
    assert sum(row.split == "train" for row in rows) == 30_648
    assert sum(row.split == "eval" for row in rows) == 120
    assert len(
        [row for row in rows if row.split == "eval" and row.designed_sequence == "AHLIFTK"]
    ) == 1
    assert len(
        [row for row in rows if row.split == "eval" and row.designed_sequence == "MWNNYYF"]
    ) == 1


def test_non_designed_residue_change_is_rejected():
    template = _template()
    affibody = list(REFERENCE_AFFIBODY)
    affibody[20] = "A" if affibody[20] != "A" else "C"
    with pytest.raises(ValueError, match="non-designed Affibody residue"):
        stab.current_pair_sequences(_row(affibody="".join(affibody)), template)


def test_target_last_orders_are_permutations_deterministic_and_restricted():
    all_keys = ("A:1", "P:4", "H:8", "H:9")
    targets = ("P:4", "H:8")
    first = stab._stable_target_last_orders(all_keys, targets)
    second = stab._stable_target_last_orders(all_keys, targets)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 4)
    assert first[0, -1] == all_keys.index("P:4")
    assert first[1, -1] == all_keys.index("H:8")
    assert all(sorted(row.tolist()) == list(range(4)) for row in first)

    restricted_keys = ("A:1", "P:4")
    restricted = stab._stable_target_last_orders(restricted_keys, ("P:4",))[0]
    full_physical_order = [all_keys[index] for index in first[0, :-1]]
    expected = [key for key in full_physical_order if key in restricted_keys and key != "P:4"]
    assert [restricted_keys[index] for index in restricted[:-1]] == expected


def test_assemble_features_contains_identity_aware_log_probability_contrast():
    rows = [_row(), _row(1, peptide="SLLAFITQV")]
    batch = len(rows)
    complex_hidden = np.ones((batch, 7, 128), dtype=np.float32)
    isolated_hidden = np.zeros((batch, 7, 128), dtype=np.float32)
    complex_log_probs = np.zeros((batch, 7, 21), dtype=np.float32)
    isolated_log_probs = np.zeros((batch, 7, 21), dtype=np.float32)
    designed_indices = stab._encode_sequences([row.designed_sequence for row in rows])
    for row_index in range(batch):
        for site_index in range(7):
            complex_log_probs[
                row_index, site_index, designed_indices[row_index, site_index]
            ] = float(site_index + 1)

    output = stab.assemble_feature_batch(
        rows,
        _template(),
        stab.DomainOutput(complex_hidden, complex_log_probs),
        stab.DomainOutput(isolated_hidden[:, :2], isolated_log_probs[:, :2]),
        stab.DomainOutput(isolated_hidden[:, 2:], isolated_log_probs[:, 2:]),
    )
    assert output["global_features"].shape == (2, 138)
    assert output["residue_features"].shape == (2, 7, 150)
    assert output["residue_hidden_delta"].shape == (2, 7, 128)
    assert output["residue_mask"].dtype == np.bool_
    np.testing.assert_allclose(
        output["residue_features"][:, :, -1],
        np.tile(np.arange(1, 8), (batch, 1)),
    )
    np.testing.assert_allclose(output["pair_compatibility_features"][:, 0], 4.0)
    np.testing.assert_allclose(output["pair_compatibility_features"][:, 1], 1.5)
    np.testing.assert_allclose(output["pair_compatibility_features"][:, 2], 5.0)


def test_supervision_fields_are_rejected_recursively(tmp_path):
    payload = {"schema_version": stab.ROW_MANIFEST_SCHEMA, "rows": []}
    path = tmp_path / "rows.json"
    payload["nested"] = {"target_retention": 99.0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="target_retention"):
        stab.load_canonical_rows(path, require_full_dataset=False)


def test_feature_archive_writer_matches_opaque_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(stab, "validate_private_output_path", lambda value: Path(value))
    rows = [_row(0, "train"), _row(1, "eval", peptide="SLLAFITQV")]
    specs = stab.feature_specs()
    contract = stab.archive_run_contract(stab.NATIVE_CRYSTAL_CONTEXT)
    writer = stab.FeatureArchiveWriter(tmp_path / "archive", rows, specs, contract)
    arrays = {
        name: np.zeros((len(rows),) + shape, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
    writer.append(arrays)
    manifest = writer.finish({**contract, "unit_test": True})
    assert manifest["schema_version"] == stab.FEATURE_ARCHIVE_SCHEMA
    assert manifest["supervision_fields_read"] == []
    assert not (tmp_path / "archive" / "EXTRACTION_INCOMPLETE").exists()
    for name, contract in manifest["arrays"].items():
        array = np.load(tmp_path / "archive" / contract["file"], allow_pickle=False)
        assert list(array.shape) == contract["shape"]
        assert stab.sha256_file(tmp_path / "archive" / contract["file"]) == contract["sha256"]


def test_feature_archive_resumes_only_after_flushed_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(stab, "validate_private_output_path", lambda value: Path(value))
    rows = [_row(0, "train"), _row(1, "eval", peptide="SLLAFITQV")]
    specs = stab.feature_specs()
    root = tmp_path / "resume-archive"
    contract = stab.archive_run_contract(stab.NATIVE_CRYSTAL_CONTEXT)
    writer = stab.FeatureArchiveWriter(root, rows, specs, contract)
    first = {
        name: np.ones((1,) + shape, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
    writer.append(first)
    writer.checkpoint()
    del writer

    resumed = stab.FeatureArchiveWriter.resume(root, rows, specs, contract)
    assert resumed.offset == 1
    second = {
        name: np.full((1,) + shape, 2, dtype=dtype)
        for name, (shape, dtype) in specs.items()
    }
    # Boolean masks cannot represent integer 2; this still verifies the row.
    second["residue_mask"][:] = True
    resumed.append(second)
    resumed.checkpoint()
    resumed.finish({**contract, "resumed_unit_test": True})
    global_features = np.load(root / "global_features.npy", allow_pickle=False)
    assert np.all(global_features[0] == 1)
    assert np.all(global_features[1] == 2)


def test_shard_merge_restores_canonical_row_order(tmp_path, monkeypatch):
    monkeypatch.setattr(stab, "validate_private_output_path", lambda value: Path(value))
    rows = [
        _row(0, "train"),
        _row(1, "train", peptide="SLLAFITQV"),
        _row(2, "eval", peptide="SLLTLITQV"),
        _row(3, "eval", peptide="SLLELITQV"),
    ]
    specs = stab.feature_specs()
    row_manifest = tmp_path / "rows.json"
    row_manifest.write_text("{}\n", encoding="utf-8")
    row_manifest_sha256 = stab.sha256_file(row_manifest)
    shard_roots = []
    for shard_index in range(2):
        shard_rows = stab.partition_rows(rows, shard_index, 2)
        root = tmp_path / f"shard-{shard_index}"
        contract = stab.archive_run_contract(
            stab.NATIVE_CRYSTAL_CONTEXT,
            additional={
                "template_context_schema": stab.CONTEXT_SCHEMA_VERSIONS[
                    stab.NATIVE_CRYSTAL_CONTEXT
                ],
                "upstream_repository": stab.UPSTREAM_REPOSITORY,
                "upstream_commit": stab.UPSTREAM_COMMIT,
                "checkpoint_sha256": stab.FINAL_CHECKPOINT_SHA256,
                "pdb_sha256": stab.REFERENCE_PDB_SHA256,
                "row_manifest_sha256": row_manifest_sha256,
                "source_row_schema": stab.ROW_MANIFEST_SCHEMA,
                "explicit_row_selection": False,
                "num_shards": 2,
                "shard_index": shard_index,
            },
        )
        writer = stab.FeatureArchiveWriter(root, shard_rows, specs, contract)
        values = {}
        for name, (shape, dtype) in specs.items():
            values[name] = np.stack(
                [np.full(shape, row.row_index, dtype=dtype) for row in shard_rows]
            )
        writer.append(values)
        writer.finish({**contract, "shard": shard_index})
        shard_roots.append(root)

    monkeypatch.setattr(
        stab,
        "load_canonical_rows",
        lambda unused_path: ({"schema_version": stab.ROW_MANIFEST_SCHEMA}, rows),
    )
    output = tmp_path / "merged"
    manifest = merger.merge_shards(
        row_manifest,
        shard_roots,
        output,
        stab.NATIVE_CRYSTAL_CONTEXT,
        merge_batch_size=2,
    )
    assert manifest["row_count"] == 4
    merged = np.load(output / "global_features.npy", allow_pickle=False)
    np.testing.assert_array_equal(merged[:, 0], np.arange(4))
