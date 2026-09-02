import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC import extract_mint_multilayer_cache as cache


class FakeMINTModel:
    cls_idx = 0
    padding_idx = 1
    eos_idx = 2

    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def __call__(self, tokens, chain_ids, repr_layers):
        self.calls.append(tuple(repr_layers))
        offsets = torch.arange(
            self.dimension, dtype=torch.float32, device=tokens.device
        ).view(1, 1, -1)
        representations = {}
        for layer in repr_layers:
            values = (
                tokens.float().unsqueeze(-1)
                + chain_ids.float().unsqueeze(-1) * 10.0
                + float(layer) * 100.0
                + offsets
            )
            representations[int(layer)] = values
        return {"logits": None, "representations": representations}


def _fake_batch():
    # Each concatenated sequence has: cls, two residues, eos, cls, three
    # residues, eos.  The distinct chain IDs on special tokens are deliberate.
    tokens = torch.tensor(
        [
            [0, 5, 7, 2, 0, 11, 13, 17, 2],
            [0, 19, 23, 2, 0, 29, 31, 37, 2],
        ],
        dtype=torch.int64,
    )
    chain_ids = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, 1]] * 2, dtype=torch.int32
    )
    return tokens, chain_ids


def test_layer_parser_and_feature_names_are_strict_and_stable():
    assert cache.parse_layers("1,5,33") == (1, 5, 33)
    assert cache.feature_names((1, 5, 33)) == (
        "mint_layer_01_chain_mean",
        "mint_layer_05_chain_mean",
        "mint_layer_33_chain_mean",
    )
    assert "emb_layer_norm_after" in cache.layer_semantics(33)
    assert "before emb_layer_norm_after" in cache.layer_semantics(5)
    with pytest.raises(ValueError, match="unique"):
        cache.parse_layers("1,1,33")
    with pytest.raises(ValueError, match="increasing"):
        cache.parse_layers("5,1,33")
    with pytest.raises(ValueError, match="1..33"):
        cache.parse_layers("1,34")
    with pytest.raises(ValueError, match="layer 33 is required"):
        cache.parse_layers("1,5")


def test_all_layers_are_requested_once_and_special_tokens_are_excluded():
    model = FakeMINTModel(dimension=3)
    tokens, chain_ids = _fake_batch()

    observed = cache.extract_multilayer_chain_means(
        model,
        tokens,
        chain_ids,
        layers=(1, 5, 33),
        expected_chain_lengths=(2, 3),
        residue_dimension=3,
    )

    assert model.calls == [(1, 5, 33)]
    assert tuple(observed) == cache.feature_names((1, 5, 33))
    expected_layer1_row0 = torch.tensor(
        [106.0, 107.0, 108.0, 123.6666667, 124.6666667, 125.6666667]
    )
    assert observed["mint_layer_01_chain_mean"].dtype == torch.float32
    assert torch.allclose(
        observed["mint_layer_01_chain_mean"][0], expected_layer1_row0
    )
    # If cls/eos had leaked in, neither chain mean would equal these values.
    assert observed["mint_layer_33_chain_mean"].shape == (2, 6)


def test_layer33_is_exactly_equal_to_historical_single_layer_request():
    model = FakeMINTModel(dimension=3)
    tokens, chain_ids = _fake_batch()
    observed = cache.extract_multilayer_chain_means(
        model,
        tokens,
        chain_ids,
        layers=(1, 33),
        expected_chain_lengths=(2, 3),
        residue_dimension=3,
    )

    assert cache.validate_layer33_single_request(
        model,
        tokens,
        chain_ids,
        observed["mint_layer_33_chain_mean"],
        expected_chain_lengths=(2, 3),
        residue_dimension=3,
    )
    assert model.calls == [(1, 33), (33,)]


def test_chain_contract_rejects_wrong_residue_count_or_chain_id():
    model = FakeMINTModel(dimension=3)
    tokens, chain_ids = _fake_batch()
    with pytest.raises(ValueError, match="chain-0 residue count"):
        cache.residue_masks(tokens, chain_ids, model, expected_chain_lengths=(3, 3))
    bad_chain_ids = chain_ids.clone()
    bad_chain_ids[0, 1] = 2
    with pytest.raises(ValueError, match="chain-0 residue count"):
        cache.residue_masks(tokens, bad_chain_ids, model, expected_chain_lengths=(2, 3))


def _sequence(length, index):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return "A" * (length - 1) + alphabet[index % len(alphabet)]


def _prepared_rows():
    specs = [
        ("LibA", "weak", "NA", "CDEF", "1"),
        ("LibB", "weak", "FG", "HIKLM", "0"),
        ("LibA", "weak", "CD", "EFGH", "0"),
        ("LibB", "weak", "GH", "IKLMN", "1"),
        ("LibA", "retention", "HI", "KLMN", ""),
        ("LibB", "retention", "IK", "LMNPQ", ""),
    ]
    records = []
    for index, (library, source_kind, pep, aff, weak_label) in enumerate(specs):
        chain1 = _sequence(270, index)
        chain2 = _sequence(58, index + 7)
        record = {column: "" for column in cache.source_cache.ROW_COLUMNS}
        record.update(
            {
                "row_index": index,
                "cache_uid": "cache-{}".format(index),
                "source_kind": source_kind,
                "library": library,
                "pair_uid": "pair-{}".format(index),
                "peptide_uid": "pep-{}".format(index),
                "affibody_uid": "aff-{}".format(index),
                "peptide_design_code": pep,
                "affibody_design_code": aff,
                "weak_label": weak_label,
                "weak_label_source": "synthetic" if source_kind == "weak" else "",
                "r001_count": "3" if source_kind == "weak" else "",
                "r009_count": "4" if weak_label == "1" else "0" if source_kind == "weak" else "",
                "r010_count": "5" if weak_label == "1" else "0" if source_kind == "weak" else "",
                "pooled_r009_r010_count": "9" if weak_label == "1" else "0" if source_kind == "weak" else "",
                "shares_retention_peptide": "0" if source_kind == "weak" else "",
                "shares_retention_affibody": "0" if source_kind == "weak" else "",
                "strict_retention_identity_cold_eligible": "1" if source_kind == "weak" else "",
                "upstream_library_local_shares_retention_peptide": "0" if source_kind == "weak" else "",
                "upstream_library_local_shares_retention_affibody": "0" if source_kind == "weak" else "",
                "upstream_library_local_strict_retention_identity_cold_eligible": "1" if source_kind == "weak" else "",
                "measurement_missing": "0" if source_kind == "retention" else "",
                "target_retention": "88.5" if source_kind == "retention" else "",
                "target_binder": "1" if source_kind == "retention" else "",
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "chain1_sha256": cache.source_cache.sha256_text(chain1),
                "chain2_sha256": cache.source_cache.sha256_text(chain2),
                "sequence_pair_sha256": cache.source_cache.sha256_text(chain1 + "|" + chain2),
            }
        )
        records.append(record)
    frame = pd.DataFrame(records, columns=cache.source_cache.ROW_COLUMNS)
    cache.source_cache.validate_prepared_rows(frame.astype(str))
    return frame


def _write_prepared_input(directory, rows):
    directory.mkdir()
    rows_path = directory / cache.source_cache.ROWS_FILENAME
    rows.to_csv(rows_path, index=False)
    os.chmod(str(rows_path), 0o600)
    manifest = {
        "schema_version": cache.SOURCE_SCHEMA_VERSION,
        "stage": "prepare",
        "rows": cache.source_cache.candidate_summary(rows.astype(str)),
        "output": {
            "rows_csv": {
                "path": str(rows_path),
                "sha256": cache.sha256_file(rows_path),
                "mode": "0600",
                "columns": list(cache.source_cache.ROW_COLUMNS),
            }
        },
    }
    manifest_path = directory / cache.source_cache.PREPARE_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(str(manifest_path), 0o600)
    return rows_path, manifest_path


def _write_synthetic_shards(directory, input_dir, rows, shard_count, layers=(1, 33)):
    directory.mkdir()
    rows_path = input_dir / cache.source_cache.ROWS_FILENAME
    prepare_path = input_dir / cache.source_cache.PREPARE_MANIFEST_FILENAME
    hashes = {
        "rows_sha256": cache.sha256_file(rows_path),
        "prepare_manifest_sha256": cache.sha256_file(prepare_path),
        "checkpoint_sha256": "synthetic-checkpoint",
        "config_sha256": "synthetic-config",
        "script_sha256": "synthetic-script",
        "dependency_sha256": {"fake": "synthetic-dependency"},
    }
    runtime = {
        "python": "3.11",
        "torch": "synthetic",
        "torch_cuda": "synthetic",
        "numpy": "synthetic",
        "gpu_name": "synthetic-a100",
        "gpu_capability": [8, 0],
        "batch_size": 8,
    }
    contract_hash, contract_payload = cache.build_feature_contract(
        hashes,
        layers,
        shard_count,
        8,
        runtime,
        {"encoder_layers": 33, "encoder_embed_dim": 1280},
    )
    for shard_index in range(shard_count):
        selected = cache.select_shard_rows(rows.astype(str), shard_index, shard_count)
        indices = selected["row_index"].astype(np.int64).to_numpy()
        arrays = cache.metadata_arrays(selected)
        arrays[cache.REPRESENTATION_LAYERS_KEY] = np.asarray(layers, dtype=np.int64)
        for layer in layers:
            arrays[cache.feature_name(layer)] = np.repeat(
                (indices + layer * 100)[:, None], cache.FEATURE_DIMENSION, axis=1
            ).astype(np.float32)
        shard_path = directory / cache.shard_filename(shard_index, shard_count)
        with open(str(shard_path), "wb") as handle:
            np.savez(handle, **arrays)
        os.chmod(str(shard_path), 0o600)
        manifest = {
            "schema_version": cache.SCHEMA_VERSION,
            "stage": "extract_shard",
            "hostname": "synthetic-host",
            "input": {
                "rows_csv": {"sha256": hashes["rows_sha256"]},
                "prepare_manifest": {"sha256": hashes["prepare_manifest_sha256"]},
            },
            "contract": {"sha256": contract_hash, "payload": contract_payload},
            "model": {
                "encoder_layers": 33,
                "encoder_embed_dim": 1280,
                "layer33_single_request_exact_match": True,
            },
            "features": {
                "layers": list(layers),
                "names": list(cache.feature_names(layers)),
                "dtype": "float32",
            },
            "sharding": {
                "shard_index": shard_index,
                "shard_count": shard_count,
                "row_index_sha256": hashlib.sha256(
                    indices.astype("<i8").tobytes()
                ).hexdigest(),
            },
            "runtime": runtime,
            "output": {"sha256": cache.sha256_file(shard_path)},
        }
        manifest_path = directory / cache.shard_manifest_filename(
            shard_index, shard_count
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(str(manifest_path), 0o600)


def test_merge_restores_rows_layers_float32_and_refuses_overwrite(tmp_path, monkeypatch):
    rows = _prepared_rows()
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
    args = Namespace(
        input_dir=input_dir,
        shards_dir=shards_dir,
        output_dir=output_dir,
        shard_count=3,
        layers="1,33",
        uncompressed=True,
    )

    cache.run_merge(args)

    archive_path = output_dir / cache.FINAL_CACHE_FILENAME
    manifest_path = output_dir / cache.FINAL_MANIFEST_FILENAME
    with np.load(str(archive_path), allow_pickle=False) as archive:
        assert np.array_equal(archive["row_index"], np.arange(len(rows)))
        assert np.array_equal(archive[cache.REPRESENTATION_LAYERS_KEY], [1, 33])
        assert archive[cache.feature_name(1)].dtype == np.float32
        assert archive[cache.feature_name(33)].shape == (
            len(rows),
            cache.FEATURE_DIMENSION,
        )
        assert np.array_equal(
            archive[cache.feature_name(33)][:, 0],
            np.arange(len(rows), dtype=np.float32) + 3300,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["features"]["by_layer"]] == list(
        cache.feature_names((1, 33))
    )
    assert manifest["model"]["layer33_single_request_exact_match"] is True
    assert os.stat(str(output_dir)).st_mode & 0o777 == 0o700
    assert os.stat(str(archive_path)).st_mode & 0o777 == 0o600
    assert os.stat(str(manifest_path)).st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="output directory exists; refusing overwrite"):
        cache.run_merge(args)


def test_shard_validation_rejects_metadata_tampering(tmp_path):
    rows = _prepared_rows()
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
            layers=(1, 33),
        )


def test_shard_validation_rejects_non_float32_features(tmp_path):
    rows = _prepared_rows()
    input_dir = tmp_path / "input"
    shards_dir = tmp_path / "shards"
    _write_prepared_input(input_dir, rows)
    _write_synthetic_shards(shards_dir, input_dir, rows, shard_count=2)
    shard_path = shards_dir / cache.shard_filename(0, 2)
    manifest_path = shards_dir / cache.shard_manifest_filename(0, 2)
    with np.load(str(shard_path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[cache.feature_name(1)] = arrays[cache.feature_name(1)].astype(np.float16)
    with open(str(shard_path), "wb") as handle:
        np.savez(handle, **arrays)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"]["sha256"] = cache.sha256_file(shard_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not stored as float32"):
        cache._load_and_validate_shard(
            shard_path,
            manifest_path,
            rows.astype(str),
            shard_index=0,
            shard_count=2,
            layers=(1, 33),
        )
