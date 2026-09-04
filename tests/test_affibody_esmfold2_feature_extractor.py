"""CPU/fake-output tests for the production LibB ESMFold2 extractor.

No checkpoint is loaded and no CUDA API is called by these tests.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from downstream.AffibodyMHC import build_esmfold2_libb_canonical_rows as builder
from downstream.AffibodyMHC import extract_esmfold2_libb_features as extractor


def _pair_hash(chain1: str, chain2: str) -> str:
    return hashlib.sha256((chain1 + "|" + chain2).encode("ascii")).hexdigest()


def _chain1(first: str, peptide_code: str = "MW") -> str:
    return first + "A" * 260 + "SLL" + peptide_code + "ITQV"


def _chain2(first: str) -> str:
    return first + "G" * 57


def _liba_chain2(code: str = "VVKI") -> str:
    sequence = list("A" * 58)
    for position, amino_acid in zip(
        extractor.LIBA_PROFILE.affibody_code_positions_1_based, code
    ):
        sequence[position - 1] = amino_acid
    return "".join(sequence)


def _raw_row(index: int, split: str, chain1: str, chain2: str) -> dict:
    return {
        "row_index": index,
        "row_id": f"opaque-{index}",
        "split": split,
        "library": "LibB",
        "chain1_sequence": chain1,
        "chain2_sequence": chain2,
        "sequence_pair_sha256": _pair_hash(chain1, chain2),
    }


def _small_rows():
    return [
        _raw_row(0, "train", _chain1("A"), _chain2("A")),
        _raw_row(1, "train", _chain1("C"), _chain2("C")),
        _raw_row(2, "eval", _chain1("D"), _chain2("D")),
    ]


def _small_expectations():
    return extractor.DatasetExpectations(
        train_rows=2,
        eval_rows=1,
        train_chain1=2,
        train_chain2=2,
        eval_chain1=1,
        eval_chain2=1,
    )


def _fake_output():
    length = sum(extractor.EXPECTED_CHAIN_LENGTHS)
    # Expanded views keep this full-shape contract test small in memory.
    logits = torch.zeros(1, length, length, 1).expand(1, length, length, 64)
    row = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1)
    column = torch.arange(length, dtype=torch.float32).view(1, 1, length, 1)
    pair = (row + 2.0 * column).expand(1, length, length, 256)
    single = torch.arange(length, dtype=torch.float32).view(1, length, 1)
    single = single.expand(1, length, 451)
    return SimpleNamespace(
        distogram_logits=logits,
        pair_states=pair,
        single_inputs=single,
    )


def _fake_prepared():
    chain1_length, chain2_length = extractor.EXPECTED_CHAIN_LENGTHS
    total = chain1_length + chain2_length
    return SimpleNamespace(
        forward_kwargs={
            "asym_id": torch.cat(
                (
                    torch.zeros(chain1_length, dtype=torch.long),
                    torch.ones(chain2_length, dtype=torch.long),
                )
            ).unsqueeze(0),
            "residue_index": torch.cat(
                (torch.arange(chain1_length), torch.arange(chain2_length))
            ).unsqueeze(0),
            "attention_mask": torch.ones(1, total, dtype=torch.bool),
        }
    )


class _FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def __call__(self, **kwargs):
        assert set(kwargs) == {"asym_id", "residue_index", "attention_mask"}
        self.calls += 1
        return self.output


def test_label_like_fields_are_rejected_recursively():
    clean = {
        "schema_version": extractor.ROW_MANIFEST_SCHEMA,
        "rows": _small_rows(),
        "provenance": {"source_sha256": "abc"},
    }
    extractor._reject_label_like_keys(clean)

    contaminated = dict(clean)
    contaminated["provenance"] = {"nested": {"target_retention": [99.0]}}
    with pytest.raises(ValueError, match="target_retention"):
        extractor._reject_label_like_keys(contaminated)

    contaminated = dict(clean)
    contaminated["training_labels_csv"] = "should-never-be-read.csv"
    with pytest.raises(ValueError, match="training_labels_csv"):
        extractor._reject_label_like_keys(contaminated)


def test_small_manifest_contract_and_strict_split():
    rows = extractor._validate_rows(_small_rows(), _small_expectations())
    assert [row.row_index for row in rows] == [0, 1, 2]
    assert [row.split for row in rows] == ["train", "train", "eval"]

    leaked = _small_rows()
    leaked[2] = _raw_row(2, "eval", leaked[0]["chain1_sequence"], _chain2("D"))
    with pytest.raises(ValueError, match="evaluation chain1 occurs in training"):
        extractor._validate_rows(leaked, _small_expectations())


def test_liba_profile_is_inferred_and_uses_four_displayed_sequence_positions(tmp_path):
    chain1 = _chain1("A", peptide_code="MW")
    chain2 = _liba_chain2("VVKI")
    row = {
        "row_index": 0,
        "row_id": "liba-0",
        "split": "train",
        "library": "LibA",
        "chain1_sequence": chain1,
        "chain2_sequence": chain2,
        "sequence_pair_sha256": _pair_hash(chain1, chain2),
        "peptide_design_code": "MW",
        "affibody_design_code": "VVKI",
    }
    observed = extractor._validate_row_mapping(row, profile=extractor.LIBA_PROFILE)
    assert observed.chain2_sequence == chain2

    wrong_code = dict(row, affibody_design_code="VVKV")
    with pytest.raises(ValueError, match="Affibody design code mapping mismatch"):
        extractor._validate_row_mapping(wrong_code, profile=extractor.LIBA_PROFILE)

    payload = {
        "schema_version": extractor.LIBA_ROW_MANIFEST_SCHEMA,
        "rows": [row],
    }
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, rows = extractor._load_row_manifest(
        path,
        expectations=extractor.DatasetExpectations(
            train_rows=1,
            eval_rows=0,
            train_chain1=1,
            train_chain2=1,
            eval_chain1=0,
            eval_chain2=0,
        ),
    )
    assert extractor._profile_for_manifest(loaded) == extractor.LIBA_PROFILE
    assert [value.row_id for value in rows] == ["liba-0"]


def test_builder_rows_payload_loads_directly_in_extractor(tmp_path):
    frame = pd.DataFrame(
        [
            {name: raw[name] for name in builder.EXTRACTOR_COLUMNS}
            for raw in _small_rows()
        ],
        columns=list(builder.EXTRACTOR_COLUMNS),
    )
    payload = builder.build_rows_payload(frame)
    manifest_path = tmp_path / builder.ROWS_FILENAME
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    observed_payload, observed_rows = extractor._load_row_manifest(
        manifest_path, expectations=_small_expectations()
    )
    assert observed_payload == payload
    assert [row.row_id for row in observed_rows] == ["opaque-0", "opaque-1", "opaque-2"]


def test_rows_are_partitioned_by_canonical_index_modulo_shards():
    rows = [
        extractor.CanonicalRow(i, f"id-{i}", "train", "A", "C", f"sha-{i}")
        for i in range(17)
    ]
    observed = [
        [row.row_index for row in extractor._partition_rows(rows, shard, 4)]
        for shard in range(4)
    ]
    assert observed == [
        [0, 4, 8, 12, 16],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ]


def test_compact_features_have_exact_shapes_order_and_symmetry():
    output = _fake_output()
    peptide = torch.arange(261, 270)
    affibody = torch.arange(270, 328)
    features = extractor._compact_feature_bundle(output, peptide, affibody)

    assert features["distogram_probabilities"].shape == (9, 58, 64)
    assert features["distogram_probabilities"].dtype == np.float16
    np.testing.assert_allclose(
        features["distogram_probabilities"].astype(np.float32).sum(axis=-1),
        1.0,
        atol=2e-3,
    )
    assert features["pair_states_symmetric"].shape == (9, 58, 256)
    # pair[i,j] = i + 2j; aligned reverse = j + 2i.
    expected_first = 1.5 * (261 + 270)
    assert float(features["pair_states_symmetric"][0, 0, 0]) == expected_first
    assert features["single_inputs"].shape == (67, 451)
    assert features["single_inputs"][:, 0].tolist() == list(range(261, 328))


def test_extract_one_uses_fake_model_and_returns_only_allowed_features():
    raw = _small_rows()[0]
    row = extractor._validate_row_mapping(raw)
    model = _FakeModel(_fake_output())
    prepared = _fake_prepared()

    features, timing = extractor._extract_one(
        model=model,
        row=row,
        device=torch.device("cpu"),
        seed=7,
        prepare_fn=lambda sequences, device: prepared,
    )

    assert model.calls == 1
    assert set(features) == set(extractor.SAVED_FEATURE_SPECS)
    assert timing["row_index"] == 0
    assert timing["row_id"] == "opaque-0"
    assert timing["peak_gpu_memory_bytes"] is None
    assert timing["preparation_seconds"] >= 0
    assert timing["forward_seconds"] >= 0


def test_atomic_chunk_round_trip_resume_validation_and_checksum(tmp_path):
    rows = [extractor._validate_row_mapping(row) for row in _small_rows()[:2]]
    output = _fake_output()
    peptide = torch.arange(261, 270)
    affibody = torch.arange(270, 328)
    bundle = extractor._compact_feature_bundle(output, peptide, affibody)
    bundles = [bundle, bundle]
    timings = [
        {
            "row_index": row.row_index,
            "row_id": row.row_id,
            "sequence_pair_sha256": row.sequence_pair_sha256,
            "preparation_seconds": 0.1,
            "forward_seconds": 1.2,
            "peak_gpu_memory_bytes": None,
        }
        for row in rows
    ]
    artifact = tmp_path / "chunk-00000.npz"
    metadata = tmp_path / "chunk-00000.json"
    contract_hash = "a" * 64

    written = extractor._write_chunk(
        artifact,
        metadata,
        rows,
        bundles,
        timings,
        contract_hash,
        shard_index=0,
        chunk_index=0,
    )
    validated = extractor._validate_chunk(
        artifact,
        metadata,
        rows,
        contract_hash,
        shard_index=0,
        chunk_index=0,
    )
    assert validated["artifact_sha256"] == written["artifact_sha256"]
    assert not list(tmp_path.glob(".*.tmp-*"))

    with artifact.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="metadata mismatch|checksum mismatch"):
        extractor._validate_chunk(
            artifact,
            metadata,
            rows,
            contract_hash,
            shard_index=0,
            chunk_index=0,
        )


def test_incomplete_chunk_fails_visibly(tmp_path):
    artifact = tmp_path / "chunk-00000.npz"
    artifact.touch()
    with pytest.raises(RuntimeError, match="incomplete chunk"):
        extractor._validate_chunk(
            artifact,
            tmp_path / "chunk-00000.json",
            [],
            "a" * 64,
            shard_index=0,
            chunk_index=0,
        )


def test_cublas_workspace_config_is_mandatory(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        extractor._configure_determinism()


def test_output_path_must_be_a_dedicated_direct_child(monkeypatch, tmp_path):
    private_root = tmp_path / "private_data"
    output_parent = private_root / "derived"
    output_parent.mkdir(parents=True)
    monkeypatch.setattr(extractor, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(extractor, "OUTPUT_PARENT", output_parent)

    valid = output_parent / "esmfold2_libb_features_test"
    assert extractor._validate_private_output_root(valid) == valid.resolve()

    invalid_paths = (
        private_root,
        output_parent,
        output_parent / "wrong-prefix",
        valid / "nested",
        output_parent / extractor.OUTPUT_PREFIX,
    )
    for invalid in invalid_paths:
        with pytest.raises(ValueError):
            extractor._validate_private_output_root(invalid)

    target = output_parent / "real-directory"
    target.mkdir()
    link = output_parent / "esmfold2_libb_features_link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        extractor._validate_private_output_root(link)


def test_private_directory_creation_does_not_chmod_existing_paths(tmp_path):
    created = tmp_path / "new"
    extractor._ensure_private_directory(created)
    assert created.stat().st_mode & 0o777 == 0o700

    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    with pytest.raises(PermissionError, match="0700"):
        extractor._ensure_private_directory(existing)
    assert existing.stat().st_mode & 0o777 == 0o755

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        extractor._ensure_private_directory(link)
