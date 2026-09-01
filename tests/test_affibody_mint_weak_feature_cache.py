import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_mint_weak_feature_cache as cache
from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    EXPECTED_CONSTRUCT_LENGTH,
    EXPECTED_X_POSITIONS,
    OMITTED_LINKER,
)


SYNTHETIC_MODEL_HASHES = {
    "checkpoint_sha256": "synthetic-checkpoint",
    "config_sha256": "synthetic-config",
    "mint_source_tree_sha256": "synthetic-mint-tree",
    "script_sha256": "synthetic-script",
    "pair_collator_source_sha256": "synthetic-collator",
}
SYNTHETIC_RUNTIME = {
    "python": "3.7.12",
    "torch": "1.12.1",
    "torch_cuda": "11.3",
    "numpy": "1.21.2",
    "batch_size": 8,
    "gpu_name": "synthetic-gpu",
    "gpu_capability": [8, 0],
}


def _template(library):
    template = list("A" * EXPECTED_CONSTRUCT_LENGTH)
    template[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)] = OMITTED_LINKER
    for position in EXPECTED_X_POSITIONS[library]:
        template[position - 1] = "X"
    return "".join(template)


def _uid(*parts):
    return cache.opaque_id(*parts)


def _weak_row(
    library,
    pep,
    aff,
    label,
    r001_count,
    on_design="1",
    pair_suffix="",
):
    pair_uid = _uid(library, pep, aff, pair_suffix)
    return {
        "library": library,
        "pep": pep,
        "aff": aff,
        "r001_count": str(r001_count),
        "r009_count": "4" if label else "0",
        "r010_count": "5" if label else "0",
        "pooled_r009_r010_count": "9" if label else "0",
        "weak_label": str(label),
        "weak_label_source": "positive" if label else "negative",
        "within_declared_library_alphabet": str(on_design),
        "shares_retention_peptide": "0",
        "shares_retention_affibody": "0",
        "strict_retention_identity_cold_eligible": "1",
        "pair_uid": pair_uid,
        "peptide_uid": _uid(library, "pep", pep),
        "affibody_uid": _uid(library, "aff", aff),
    }


def _retention_row(library, pep, aff, template, measured=True):
    chain1, chain2 = cache.reconstruct_chains(template, library, pep, aff)
    return {
        "library": library,
        "pair_uid": _uid("retention", library, pep, aff),
        "peptide_uid": _uid(library, "pep", pep),
        "affibody_uid": _uid(library, "aff", aff),
        "peptide_design_code": pep,
        "affibody_design_code": aff,
        "measurement_missing": "0" if measured else "1",
        "target_retention": "88.5" if measured else "",
        "target_binder": "1" if measured else "",
        "chain1_smart_hla_linker_peptide_sequence": chain1,
        "chain2_affibody_sequence": chain2,
    }


def _small_rows():
    templates = {"LibA": _template("LibA"), "LibB": _template("LibB")}
    weak = pd.DataFrame(
        [
            _weak_row("LibA", "NA", "CDEF", 1, 0),
            _weak_row("LibA", "CD", "EFGH", 0, 3),
            _weak_row("LibA", "DE", "FGHI", 0, 2),
            _weak_row("LibA", "EF", "GHIK", 1, 0, on_design="0"),
            _weak_row("LibB", "FG", "HIKLM", 1, 0),
            _weak_row("LibB", "GH", "IKLMN", 0, 4),
        ]
    )
    retention = pd.DataFrame(
        [
            _retention_row("LibA", "HI", "KLMN", templates["LibA"]),
            _retention_row(
                "LibB", "IK", "LMNPQ", templates["LibB"], measured=False
            ),
        ]
    )
    rows = cache.assemble_candidate_rows(
        weak, retention, templates, min_negative_r001_count=3
    )
    return rows


def test_candidate_rows_preserve_literal_na_filter_negatives_and_omit_linker():
    rows = _small_rows()

    assert len(rows) == 6
    weak = rows.loc[rows["source_kind"].eq("weak")]
    assert weak.groupby(["library", "weak_label"]).size().to_dict() == {
        ("LibA", "0"): 1,
        ("LibA", "1"): 1,
        ("LibB", "0"): 1,
        ("LibB", "1"): 1,
    }
    assert "NA" in weak["peptide_design_code"].tolist()
    assert "DE" not in weak["peptide_design_code"].tolist()
    assert "EF" not in weak["peptide_design_code"].tolist()
    assert rows["row_index"].tolist() == list(range(len(rows)))
    assert rows["cache_uid"].nunique() == len(rows)
    assert rows["chain1_smart_hla_linker_peptide_sequence"].map(len).eq(270).all()
    assert rows["chain2_affibody_sequence"].map(len).eq(58).all()
    assert not rows["chain1_smart_hla_linker_peptide_sequence"].str.contains(
        OMITTED_LINKER, regex=False
    ).any()
    assert not rows["chain2_affibody_sequence"].str.contains(
        OMITTED_LINKER, regex=False
    ).any()
    assert rows["chain1_sha256"].map(len).eq(64).all()
    assert rows["chain2_sha256"].map(len).eq(64).all()
    cache.validate_prepared_rows(rows.astype(str))


def test_string_csv_reader_does_not_convert_literal_na(tmp_path):
    path = tmp_path / "codes.csv"
    path.write_text("pep,aff\nNA,CDEF\n", encoding="utf-8")

    observed = cache.read_string_csv(path)

    assert observed.loc[0, "pep"] == "NA"
    assert observed.loc[0, "aff"] == "CDEF"


def test_modulo_shards_are_deterministic_disjoint_and_complete():
    rows = _small_rows().astype(str)

    first = [cache.select_shard_rows(rows, index, 3) for index in range(3)]
    shuffled = rows.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = [cache.select_shard_rows(shuffled, index, 3) for index in range(3)]

    for index in range(3):
        expected = first[index]["row_index"].astype(int).tolist()
        assert all(value % 3 == index for value in expected)
        assert set(expected) == set(second[index]["row_index"].astype(int))
    combined = sorted(
        value
        for shard in first
        for value in shard["row_index"].astype(int).tolist()
    )
    assert combined == list(range(len(rows)))


def _write_prepared_input(directory, rows):
    directory.mkdir()
    rows_path = directory / cache.ROWS_FILENAME
    rows.to_csv(rows_path, index=False)
    os.chmod(str(rows_path), 0o600)
    manifest = {
        "schema_version": cache.SCHEMA_VERSION,
        "stage": "prepare",
        "rows": cache.candidate_summary(rows.astype(str)),
        "output": {
            "rows_csv": {
                "path": str(rows_path),
                "sha256": cache.sha256_file(rows_path),
                "mode": "0600",
                "columns": list(cache.ROW_COLUMNS),
            }
        },
    }
    manifest_path = directory / cache.PREPARE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(str(manifest_path), 0o600)
    return rows_path, manifest_path


def _write_synthetic_shards(directory, input_dir, rows, shard_count):
    directory.mkdir()
    rows_path = input_dir / cache.ROWS_FILENAME
    prepare_manifest_path = input_dir / cache.PREPARE_MANIFEST_FILENAME
    rows_hash = cache.sha256_file(rows_path)
    prepare_hash = cache.sha256_file(prepare_manifest_path)
    contract_payload = dict(SYNTHETIC_MODEL_HASHES)
    contract_payload.update(
        {
            "shard_count": shard_count,
            "representation": cache.FEATURE_NAME,
            "feature_dimension": cache.FEATURE_DIMENSION,
            "runtime": dict(SYNTHETIC_RUNTIME),
        }
    )
    contract = cache.canonical_json_sha256(contract_payload)
    for shard_index in range(shard_count):
        selected = cache.select_shard_rows(rows.astype(str), shard_index, shard_count)
        indices = selected["row_index"].astype(np.int64).to_numpy()
        values = np.repeat(indices[:, None], cache.FEATURE_DIMENSION, axis=1).astype(
            np.float32
        )
        arrays = cache.metadata_arrays(selected)
        arrays[cache.FEATURE_NAME] = values
        shard_path = directory / cache.shard_filename(shard_index, shard_count)
        with open(str(shard_path), "wb") as handle:
            np.savez(handle, **arrays)
        os.chmod(str(shard_path), 0o600)
        manifest = {
            "schema_version": cache.SCHEMA_VERSION,
            "stage": "extract_shard",
            "hostname": "synthetic-host",
            "input": {
                "rows_csv": {"sha256": rows_hash},
                "prepare_manifest": {"sha256": prepare_hash},
            },
            "model": {
                "feature_name": cache.FEATURE_NAME,
                "feature_dimension": cache.FEATURE_DIMENSION,
                "checkpoint": {"sha256": SYNTHETIC_MODEL_HASHES["checkpoint_sha256"]},
                "config": {"sha256": SYNTHETIC_MODEL_HASHES["config_sha256"]},
                "mint_source_tree_sha256": SYNTHETIC_MODEL_HASHES[
                    "mint_source_tree_sha256"
                ],
                "script_sha256": SYNTHETIC_MODEL_HASHES["script_sha256"],
                "pair_collator_source": {
                    "sha256": SYNTHETIC_MODEL_HASHES[
                        "pair_collator_source_sha256"
                    ]
                },
            },
            "contract": {"sha256": contract, "payload": contract_payload},
            "sharding": {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "row_index_sha256": __import__("hashlib")
                .sha256(indices.astype("<i8").tobytes())
                .hexdigest(),
            },
            "runtime": dict(SYNTHETIC_RUNTIME),
            "output": {
                "sha256": cache.sha256_file(shard_path),
            },
        }
        manifest_path = directory / cache.shard_manifest_filename(
            shard_index, shard_count
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(str(manifest_path), 0o600)


def test_merge_restores_row_order_validates_metadata_and_sets_private_modes(
    tmp_path, monkeypatch
):
    rows = _small_rows()
    input_dir = tmp_path / "input"
    shards_dir = tmp_path / "shards"
    output_dir = tmp_path / "merged"
    _write_prepared_input(input_dir, rows)
    _write_synthetic_shards(shards_dir, input_dir, rows, shard_count=3)
    monkeypatch.setattr(
        cache,
        "validate_private_output_path",
        lambda path, repo_root: Path(path).resolve(),
    )

    cache.run_merge(
        Namespace(
            input_dir=input_dir,
            shards_dir=shards_dir,
            output_dir=output_dir,
            shard_count=3,
            uncompressed=True,
        )
    )

    cache_path = output_dir / cache.FINAL_CACHE_FILENAME
    manifest_path = output_dir / cache.FINAL_MANIFEST_FILENAME
    with np.load(str(cache_path), allow_pickle=False) as archive:
        assert np.array_equal(archive["row_index"], np.arange(len(rows)))
        assert archive[cache.FEATURE_NAME].shape == (
            len(rows),
            cache.FEATURE_DIMENSION,
        )
        assert np.array_equal(
            archive[cache.FEATURE_NAME][:, 0], np.arange(len(rows), dtype=np.float32)
        )
        assert "NA" in archive["peptide_design_code"].astype(str).tolist()
        assert np.array_equal(
            archive["chain1_sha256"].astype(str), rows["chain1_sha256"].to_numpy()
        )
    assert os.stat(str(output_dir)).st_mode & 0o777 == 0o700
    assert os.stat(str(cache_path)).st_mode & 0o777 == 0o600
    assert os.stat(str(manifest_path)).st_mode & 0o777 == 0o600
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["features"]["shape"] == [len(rows), cache.FEATURE_DIMENSION]
    expected_contract_payload = dict(SYNTHETIC_MODEL_HASHES)
    expected_contract_payload.update(
        {
            "shard_count": 3,
            "representation": cache.FEATURE_NAME,
            "feature_dimension": cache.FEATURE_DIMENSION,
            "runtime": dict(SYNTHETIC_RUNTIME),
        }
    )
    assert manifest["contract"]["sha256"] == cache.canonical_json_sha256(
        expected_contract_payload
    )


def test_shard_validation_rejects_metadata_tampering(tmp_path):
    rows = _small_rows()
    input_dir = tmp_path / "input"
    shards_dir = tmp_path / "shards"
    _write_prepared_input(input_dir, rows)
    _write_synthetic_shards(shards_dir, input_dir, rows, shard_count=2)
    shard_path = shards_dir / cache.shard_filename(0, 2)
    manifest_path = shards_dir / cache.shard_manifest_filename(0, 2)
    with np.load(str(shard_path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["peptide_design_code"] = arrays["peptide_design_code"].copy()
    arrays["peptide_design_code"][0] = "ZZ"
    with open(str(shard_path), "wb") as handle:
        np.savez(handle, **arrays)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = cache.sha256_file(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shard metadata mismatch"):
        cache._load_and_validate_shard(
            shard_path,
            manifest_path,
            rows.astype(str),
            shard_index=0,
            shard_count=2,
        )


@pytest.mark.parametrize(
    "field,value",
    [("batch_size", 4), ("gpu_name", "different-gpu"), ("gpu_capability", [9, 0])],
)
def test_shard_validation_rejects_runtime_contract_mismatch(tmp_path, field, value):
    rows = _small_rows()
    input_dir = tmp_path / "input"
    shards_dir = tmp_path / "shards"
    _write_prepared_input(input_dir, rows)
    _write_synthetic_shards(shards_dir, input_dir, rows, shard_count=2)
    shard_path = shards_dir / cache.shard_filename(0, 2)
    manifest_path = shards_dir / cache.shard_manifest_filename(0, 2)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shard runtime/contract mismatch"):
        cache._load_and_validate_shard(
            shard_path,
            manifest_path,
            rows.astype(str),
            shard_index=0,
            shard_count=2,
        )
