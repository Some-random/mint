import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.merge_libb_fixed_structure_candidate_scores import (
    EXPECTED_MODELS,
    SEEDS,
    _validate_scoring_partition_against_universe,
    _validate_worker_manifests,
    merge_peptide,
)


def _write_universe(path: Path) -> None:
    pd.DataFrame(
        {
            "pair_uid": ["pair-a", "pair-b", "pair-c"],
            "peptide_design_code": ["AF"] * 3,
            "affibody_design_code": ["AAAAA", "AAAAD", "AAAAE"],
            "chain1_smart_hla_linker_peptide_sequence": [
                "CHAIN1-A", "CHAIN1-B", "CHAIN1-C"
            ],
            "chain2_affibody_sequence": ["CHAIN2-A", "CHAIN2-B", "CHAIN2-C"],
        }
    ).to_parquet(path, index=False)


def _chunk_frame(start: int, stop: int) -> pd.DataFrame:
    pair_uids = ["pair-a", "pair-b", "pair-c"][start:stop]
    affibody_codes = ["AAAAA", "AAAAD", "AAAAE"][start:stop]
    seed_values = np.array(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50],
            [0.20, 0.30, 0.40, 0.50, 0.60],
            [0.60, 0.70, 0.80, 0.85, 0.90],
        ],
        dtype=np.float64,
    )[start:stop]
    clipped = np.clip(seed_values, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    probability_from_mean_logit = 1.0 / (1.0 + np.exp(-logits.mean(axis=1)))
    frame = pd.DataFrame(
        {
            "candidate_row_index": np.arange(start, stop),
            "pair_uid": pair_uids,
            "peptide_design_code": "AF",
            "affibody_design_code": affibody_codes,
        }
    )
    for column, seed in enumerate(SEEDS):
        frame["score_seed_%d" % seed] = seed_values[:, column]
    frame["score_mean_probability"] = seed_values.mean(axis=1)
    frame["score_mean_logit"] = probability_from_mean_logit
    frame["score_seed_sd"] = seed_values.std(axis=1, ddof=1)
    frame["model"] = EXPECTED_MODELS["rde"]
    return frame


def _write_chunk(root: Path, frame: pd.DataFrame) -> Path:
    start = int(frame["candidate_row_index"].iloc[0])
    last = int(frame["candidate_row_index"].iloc[-1])
    path = root / ("rows_%06d_%06d.csv.gz" % (start, last))
    frame.to_csv(path, index=False, compression="gzip")
    return path


def _fixture(tmp_path: Path):
    universe_path = tmp_path / "peptide_AF.parquet"
    chunk_root = tmp_path / "chunks"
    peptide_dir = chunk_root / "peptide_AF"
    peptide_dir.mkdir(parents=True)
    _write_universe(universe_path)
    _write_chunk(peptide_dir, _chunk_frame(0, 2))
    _write_chunk(peptide_dir, _chunk_frame(2, 3))
    return universe_path, chunk_root, peptide_dir


def _write_scoring_input(path: Path, universe_path: Path) -> None:
    universe = pd.read_parquet(universe_path)
    frame = pd.DataFrame({
        "candidate_row_index": np.arange(len(universe)),
        "pair_uid": universe["pair_uid"],
        "peptide_design_code": universe["peptide_design_code"],
        "affibody_design_code": universe["affibody_design_code"],
        "chain1_sequence": universe["chain1_smart_hla_linker_peptide_sequence"],
        "chain2_sequence": universe["chain2_affibody_sequence"],
    })
    frame["sequence_pair_sha256"] = [
        hashlib.sha256((left + "|" + right).encode("ascii")).hexdigest()
        for left, right in zip(frame["chain1_sequence"], frame["chain2_sequence"])
    ]
    frame.to_csv(path, index=False, compression="gzip")


def test_merge_validates_and_disambiguates_score_mean_logit(tmp_path):
    universe_path, chunk_root, peptide_dir = _fixture(tmp_path)
    merged, records = merge_peptide(
        universe_path, peptide_dir, chunk_root, "AF", "rde"
    )

    assert merged["candidate_row_index"].tolist() == [0, 1, 2]
    assert merged["pair_uid"].tolist() == ["pair-a", "pair-b", "pair-c"]
    assert len(records) == 2
    assert all(record["sha256"] and record["bytes"] > 0 for record in records)
    assert np.allclose(
        1.0 / (1.0 + np.exp(-merged["rde_logit"].to_numpy())),
        merged["rde_probability"].to_numpy(),
        atol=1e-6,
    )
    assert merged["rde_seed_mean_probability"].iloc[0] == pytest.approx(0.30)
    assert merged["rde_probability"].iloc[0] != pytest.approx(0.30)


def test_scoring_input_matches_universe_identities_sequences_and_hashes(tmp_path):
    universe_path = tmp_path / "peptide_AF.parquet"
    scoring_path = tmp_path / "peptide_AF.csv.gz"
    _write_universe(universe_path)
    _write_scoring_input(scoring_path, universe_path)

    assert _validate_scoring_partition_against_universe(
        scoring_path, universe_path, "AF"
    ) == 3


def test_scoring_input_rejects_sequence_mismatch_even_with_valid_file_manifest(tmp_path):
    universe_path = tmp_path / "peptide_AF.parquet"
    scoring_path = tmp_path / "peptide_AF.csv.gz"
    _write_universe(universe_path)
    _write_scoring_input(scoring_path, universe_path)
    frame = pd.read_csv(scoring_path)
    frame.loc[1, "chain2_sequence"] = "WRONG-SEQUENCE"
    frame.loc[1, "sequence_pair_sha256"] = hashlib.sha256(
        (str(frame.loc[1, "chain1_sequence"]) + "|WRONG-SEQUENCE").encode("ascii")
    ).hexdigest()
    frame.to_csv(scoring_path, index=False, compression="gzip")

    with pytest.raises(ValueError, match="chain2_sequence differs from universe"):
        _validate_scoring_partition_against_universe(
            scoring_path, universe_path, "AF"
        )


def test_merge_rejects_gap_in_candidate_coverage(tmp_path):
    universe_path, chunk_root, peptide_dir = _fixture(tmp_path)
    (peptide_dir / "rows_000002_000002.csv.gz").unlink()
    with pytest.raises(ValueError, match="gap, extra row, or incomplete coverage"):
        merge_peptide(universe_path, peptide_dir, chunk_root, "AF", "rde")


def test_merge_rejects_pair_mapping_mismatch(tmp_path):
    universe_path, chunk_root, peptide_dir = _fixture(tmp_path)
    path = peptide_dir / "rows_000002_000002.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "pair_uid"] = "wrong-pair"
    frame.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="pair_uid mapping mismatch"):
        merge_peptide(universe_path, peptide_dir, chunk_root, "AF", "rde")


def test_merge_rejects_nonfinite_score(tmp_path):
    universe_path, chunk_root, peptide_dir = _fixture(tmp_path)
    path = peptide_dir / "rows_000002_000002.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "score_seed_20260811"] = np.nan
    frame.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="non-finite"):
        merge_peptide(universe_path, peptide_dir, chunk_root, "AF", "rde")


def _worker_manifest_payload(
    peptide: str,
    slice_index: int,
    num_slices: int = 2,
) -> dict:
    full_rows = {"AF": 11, "AH": 13}[peptide]
    start = full_rows * slice_index // num_slices
    stop = full_rows * (slice_index + 1) // num_slices
    return {
        "schema_version": "libb-fixed-structure-candidate-scores-v1",
        "family": "rde",
        "model": EXPECTED_MODELS["rde"],
        "input_manifest_sha256": "same-input-hash",
        "readout_seeds": list(SEEDS),
        "slice_index": slice_index,
        "num_slices": num_slices,
        "peptides": [{
            "peptide": peptide,
            "full_rows": full_rows,
            "slice_start": start,
            "slice_stop": stop,
            "rows": stop - start,
        }],
    }


def _write_worker_manifest(
    root: Path,
    name: str,
    peptide: str,
    slice_index: int,
    num_slices: int = 2,
) -> Path:
    path = root / name
    path.write_text(
        json.dumps(_worker_manifest_payload(peptide, slice_index, num_slices)),
        encoding="utf-8",
    )
    return path


def _worker_partitions() -> pd.DataFrame:
    return pd.DataFrame({
        "peptide_design_code": ["AF", "AH"],
        "candidate_rows": [11, 13],
    })


def _add_native_provenance(path: Path, checkpoint_sha: str = "c" * 64) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["readout_provenance"] = {
        "config_path": "/shared/native_readout.json",
        "config_sha256": "a" * 64,
        "readout_mode": "native_learned_projection",
        "checkpoint_directory": "/shared/native_checkpoints",
        "checkpoints": [
            {"filename": "seed20260811.pt", "sha256": checkpoint_sha},
        ],
    }
    payload["producer"] = {
        "path": "/shared/score_libb_fixed_structure_candidates.py",
        "sha256": "e" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_worker_validation_accepts_more_manifests_than_slices_for_subsets(tmp_path):
    # Sixteen one-peptide receipts cover a two-peptide by eight-slice matrix.
    # This is the layout produced when accelerated jobs divide peptides across
    # GPUs, and specifically verifies that more than eight manifests is valid.
    for peptide in ("AF", "AH"):
        for slice_index in range(8):
            _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof08_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
                num_slices=8,
            )

    result = _validate_worker_manifests(tmp_path, "rde", 8, _worker_partitions())

    assert len(result["worker_manifests"]) == 16
    assert result["expected_receipt_cells"] == 16
    assert result["validated_receipt_cells"] == 16
    assert result["receipt_attestation_count"] == 16


def test_worker_validation_accepts_matching_duplicate_attestation(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )
    _write_worker_manifest(
        tmp_path,
        "worker_rde_slice00of02_AF_duplicate.json",
        "AF",
        0,
    )

    result = _validate_worker_manifests(tmp_path, "rde", 2, _worker_partitions())

    assert len(result["worker_manifests"]) == 5
    assert result["validated_receipt_cells"] == 4
    assert result["receipt_attestation_count"] == 5
    assert result["duplicate_attestation_cells"] == 1
    af_zero = next(
        row for row in result["receipt_attestations"]
        if row["peptide"] == "AF" and row["slice_index"] == 0
    )
    assert af_zero["attestation_count"] == 2
    assert len(af_zero["manifest_paths"]) == 2


def test_worker_validation_rejects_conflicting_duplicate_attestation(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )
    duplicate_path = _write_worker_manifest(
        tmp_path,
        "worker_rde_slice00of02_AF_conflict.json",
        "AF",
        0,
    )
    duplicate = json.loads(duplicate_path.read_text(encoding="utf-8"))
    duplicate["input_manifest_sha256"] = "conflicting-input-hash"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="conflicting duplicate worker receipt for peptide AF slice 0",
    ):
        _validate_worker_manifests(tmp_path, "rde", 2, _worker_partitions())


def test_worker_validation_rejects_missing_peptide_slice_receipt(tmp_path):
    for peptide, slice_index in (("AF", 0), ("AF", 1), ("AH", 0)):
        _write_worker_manifest(
            tmp_path,
            "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
            peptide,
            slice_index,
        )

    with pytest.raises(ValueError, match="receipt matrix is incomplete"):
        _validate_worker_manifests(tmp_path, "rde", 2, _worker_partitions())


def test_worker_validation_rejects_wrong_subset_row_bounds(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            path = _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )
            if peptide == "AH" and slice_index == 1:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["peptides"][0]["slice_start"] += 1
                path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="worker slice is incomplete for AH"):
        _validate_worker_manifests(tmp_path, "rde", 2, _worker_partitions())


def test_worker_validation_requires_and_propagates_matching_native_provenance(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            path = _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )
            _add_native_provenance(path)

    result = _validate_worker_manifests(
        tmp_path,
        "rde",
        2,
        _worker_partitions(),
        expected_readout_mode="native_learned_projection",
    )

    assert result["readout_provenance"]["readout_mode"] == "native_learned_projection"
    assert result["readout_provenance"]["config_sha256"] == "a" * 64
    assert result["readout_provenance"]["checkpoints"] == [
        {"filename": "seed20260811.pt", "sha256": "c" * 64}
    ]
    assert result["scorer_producer"] == {
        "path": "/shared/score_libb_fixed_structure_candidates.py",
        "sha256": "e" * 64,
    }


def test_worker_validation_rejects_missing_native_provenance(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )

    with pytest.raises(ValueError, match="native merge requires readout provenance"):
        _validate_worker_manifests(
            tmp_path,
            "rde",
            2,
            _worker_partitions(),
            expected_readout_mode="native_learned_projection",
        )


def test_worker_validation_rejects_mixed_native_checkpoint_sets(tmp_path):
    paths = []
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            path = _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )
            _add_native_provenance(path)
            paths.append(path)
    _add_native_provenance(paths[-1], checkpoint_sha="d" * 64)

    with pytest.raises(ValueError, match="do not share one readout configuration"):
        _validate_worker_manifests(
            tmp_path,
            "rde",
            2,
            _worker_partitions(),
            expected_readout_mode="native_learned_projection",
        )


def test_worker_validation_requires_supplied_scoring_input_hash(tmp_path):
    for peptide in ("AF", "AH"):
        for slice_index in range(2):
            _write_worker_manifest(
                tmp_path,
                "worker_rde_slice%02dof02_%s.json" % (slice_index, peptide),
                peptide,
                slice_index,
            )

    result = _validate_worker_manifests(
        tmp_path,
        "rde",
        2,
        _worker_partitions(),
        expected_input_manifest_sha256="same-input-hash",
    )
    assert result["scoring_input_manifest_sha256"] == "same-input-hash"

    with pytest.raises(ValueError, match="does not match the supplied"):
        _validate_worker_manifests(
            tmp_path,
            "rde",
            2,
            _worker_partitions(),
            expected_input_manifest_sha256="different-input-hash",
        )
