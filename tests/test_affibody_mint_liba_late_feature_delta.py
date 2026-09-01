import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_mint_liba_late_feature_delta as delta
from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    EXPECTED_CONSTRUCT_LENGTH,
    EXPECTED_X_POSITIONS,
    OMITTED_LINKER,
)


def _template():
    values = list("A" * EXPECTED_CONSTRUCT_LENGTH)
    values[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)] = OMITTED_LINKER
    for position in EXPECTED_X_POSITIONS["LibA"]:
        values[position - 1] = "X"
    return "".join(values)


def _round_frame(records):
    return pd.DataFrame(
        [
            {
                "pep": peptide,
                "aff": affibody,
                "count": count,
                "frequency": frequency,
            }
            for peptide, affibody, count, frequency in records
        ]
    )


def test_raw_arm_formulas_outer_join_zero_fill_and_keep_cutoff_ties():
    frames = {
        9: _round_frame([("AA", "AAAA", 10, 0.10), ("DD", "DDDD", 5, 0.05)]),
        10: _round_frame([("DD", "DDDD", 5, 0.20), ("EE", "EEEE", 8, 0.08)]),
        11: _round_frame([("FF", "FFFF", 20, 0.01), ("AA", "AAAA", 1, 0.40)]),
        12: _round_frame([("HH", "HHHH", 20, 0.30)]),
        13: _round_frame([("II", "IIII", 2, 0.50)]),
        14: _round_frame([("KK", "KKKK", 2, 0.60)]),
    }

    arms, summary = delta.derive_raw_positive_arms(frames, top_fraction=0.5)

    assert summary["A"]["universe_rows"] == 3
    assert summary["A"]["requested_rank"] == 2
    assert summary["A"]["inclusive_cutoff"] == 10
    # AA and DD tie at the cutoff and both remain.
    assert set(zip(arms["A"]["pep"], arms["A"]["aff"])) == {
        ("AA", "AAAA"),
        ("DD", "DDDD"),
    }
    # Equal-frequency arm uses count/round-depth with six equal weights,
    # including zeros for absent pairs.
    all_arms, _ = delta.derive_raw_positive_arms(frames, top_fraction=0.99)
    aa = all_arms["C"].loc[all_arms["C"]["pep"].eq("AA"), "label_score"]
    assert len(aa) == 1
    assert aa.iloc[0] == pytest.approx(
        (10.0 / (10 + 5) + 1.0 / (20 + 1)) / 6.0
    )


def _label_row(arm, peptide, affibody, label, score):
    sequence = delta._sequence_record(_template(), peptide, affibody)
    return {
        "arm": arm,
        "pep": peptide,
        "aff": affibody,
        "r001_count": 0 if label else 3,
        "weak_label": label,
        "label_score": score,
        "pair_uid": delta.opaque_id("LibA", peptide, affibody),
        "peptide_uid": delta.opaque_id("LibA", "pep", peptide),
        "affibody_uid": delta.opaque_id("LibA", "aff", affibody),
        "chain1_sha256": sequence["chain1_sha256"],
        "chain2_sha256": sequence["chain2_sha256"],
        "sequence_pair_sha256": sequence["sequence_pair_sha256"],
    }


def _small_labels():
    positives = {
        "A": [("AA", "AAAA"), ("DD", "DDDD")],
        "B": [("AA", "AAAA"), ("DD", "DDDD"), ("EE", "EEEE")],
        "C": [("AA", "AAAA"), ("DD", "DDDD"), ("FF", "FFFF")],
        "D": [("AA", "AAAA"), ("DD", "DDDD"), ("HH", "HHHH")],
    }
    rows = []
    for arm in delta.ARM_ORDER:
        for peptide, affibody in positives[arm]:
            rows.append(_label_row(arm, peptide, affibody, 1, 10))
        rows.append(_label_row(arm, "II", "IIII", 0, 0))
    return pd.DataFrame(rows)


def test_long_membership_has_natural_all_arms_and_canonical_size_matches():
    labels = _small_labels()

    observed = delta.build_long_membership(labels, seeds=(11, 12, 13))

    natural = observed.loc[observed["sampling"].eq("natural")]
    assert set(natural["arm"]) == set(delta.ARM_ORDER)
    assert not (
        observed["arm"].eq("A") & observed["sampling"].eq("size_matched")
    ).any()
    matched = observed.loc[observed["sampling"].eq("size_matched")]
    assert set(matched["arm"]) == {"B", "C", "D"}
    counts = matched.groupby(["arm", "subset_seed", "weak_label"]).size()
    for arm in ("B", "C", "D"):
        for seed in (11, 12, 13):
            assert counts[(arm, seed, 1)] == 2
            assert counts[(arm, seed, 0)] == 1
    first_uid = labels.loc[
        labels["arm"].eq("B") & labels["weak_label"].eq(1), "pair_uid"
    ].iloc[0]
    assert delta.deterministic_size_hash(11, first_uid) == hashlib.sha256(
        "size-match|11|{}".format(first_uid).encode("ascii")
    ).hexdigest()


def _retention_row(peptide="KK", affibody="KKKK"):
    sequence = delta._sequence_record(_template(), peptide, affibody)
    return dict(
        {
            "library": "LibA",
            "pair_uid": delta.opaque_id("retention", peptide, affibody),
            "peptide_uid": delta.opaque_id("LibA", "pep", peptide),
            "affibody_uid": delta.opaque_id("LibA", "aff", affibody),
            "peptide_design_code": peptide,
            "affibody_design_code": affibody,
            "measurement_missing": "0",
            "target_retention": "80",
            "target_binder": "1",
        },
        **sequence
    )


def test_delta_materializes_only_misses_plus_deterministic_old_overlap_sentinel():
    labels = _small_labels()
    pairs = labels[["pep", "aff"]].drop_duplicates()
    sequences = delta.build_pair_sequence_table(pairs, _template())
    retention = pd.DataFrame([_retention_row()])
    old_pairs = pd.concat(
        [
            sequences.loc[sequences["pep"].isin(["AA", "DD", "II"])],
            pd.DataFrame(
                [
                    {
                        "library": "LibA",
                        "pep": "KK",
                        "aff": "KKKK",
                        "pair_uid": retention.iloc[0]["pair_uid"],
                        "peptide_uid": retention.iloc[0]["peptide_uid"],
                        "affibody_uid": retention.iloc[0]["affibody_uid"],
                        "chain1_sha256": retention.iloc[0]["chain1_sha256"],
                        "chain2_sha256": retention.iloc[0]["chain2_sha256"],
                        "sequence_pair_sha256": retention.iloc[0][
                            "sequence_pair_sha256"
                        ],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    old_metadata = {
        "sequence_pair_sha256": old_pairs["sequence_pair_sha256"].to_numpy(),
    }

    rows, reuse = delta.assemble_delta_rows_and_reuse(
        labels, sequences, retention, old_metadata, sentinel_count=1
    )

    missing_hashes = set(
        rows.loc[rows["cache_role"].eq("missing"), "sequence_pair_sha256"]
    )
    assert missing_hashes == set(
        sequences.loc[sequences["pep"].isin(["EE", "FF", "HH"]), "sequence_pair_sha256"]
    )
    assert rows["cache_role"].eq("overlap_sentinel").sum() == 1
    assert reuse["feature_source"].eq("delta_cache").sum() == 3
    assert reuse.loc[reuse["source_kind"].eq("retention"), "feature_source"].eq(
        "old_cache"
    ).all()
    delta.validate_prepared_rows(rows.astype(str))
    delta.validate_reuse_index(reuse.astype(str), rows.astype(str))


def test_sentinel_comparison_requires_exact_float32_equality(tmp_path):
    sequence = delta._sequence_record(_template(), "AA", "AAAA")
    rows = pd.DataFrame(
        [
            {
                "row_index": "0",
                "cache_role": "overlap_sentinel",
                "pair_uid": "pair",
                "old_cache_row_index": "0",
                "sequence_pair_sha256": sequence["sequence_pair_sha256"],
            }
        ]
    )
    values = np.arange(delta.FEATURE_DIMENSION, dtype=np.float32)[None, :]
    old_path = tmp_path / "old.npz"
    np.savez(
        str(old_path),
        sequence_pair_sha256=np.asarray([sequence["sequence_pair_sha256"]]),
        mint_chain_mean=values,
    )

    comparison = delta.build_sentinel_comparison(rows, values.copy(), old_path)
    assert comparison.loc[0, "exact_equal"] == 1
    assert comparison.loc[0, "max_abs_difference"] == 0.0

    changed = values.copy()
    changed[0, 0] += np.float32(0.25)
    with pytest.raises(ValueError, match="do not exactly reproduce"):
        delta.build_sentinel_comparison(rows, changed, old_path)


def test_parse_cli_and_literal_na_reader(tmp_path):
    args = delta.parse_args(
        [
            "extract-shard",
            "--input-dir",
            "rows",
            "--output-dir",
            "shards",
            "--checkpoint",
            "mint.ckpt",
            "--config",
            "config.json",
            "--shard-index",
            "0",
            "--shard-count",
            "8",
        ]
    )
    assert args.command == "extract-shard"
    assert args.batch_size == 64
    path = tmp_path / "codes.csv"
    path.write_text("pep,aff\nNA,AAAA\n", encoding="utf-8")
    observed = delta.read_string_csv(path)
    assert observed.loc[0, "pep"] == "NA"


def test_private_output_path_refuses_nonprivate_directory(tmp_path):
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="private_data"):
        delta._ensure_private_directory(outside, must_be_new=True)


def _minimal_prepared_rows():
    records = []
    for index, (peptide, affibody, role, old_index) in enumerate(
        [
            ("AA", "AAAA", "missing", -1),
            ("DD", "DDDD", "overlap_sentinel", 7),
        ]
    ):
        sequence = delta._sequence_record(_template(), peptide, affibody)
        records.append(
            {
                "row_index": index,
                "delta_cache_uid": delta.opaque_id("delta", peptide, affibody),
                "cache_role": role,
                "source_kind": "weak",
                "library": "LibA",
                "pair_uid": delta.opaque_id("LibA", peptide, affibody),
                "peptide_uid": delta.opaque_id("LibA", "pep", peptide),
                "affibody_uid": delta.opaque_id("LibA", "aff", affibody),
                "peptide_design_code": peptide,
                "affibody_design_code": affibody,
                "old_cache_row_index": old_index,
                **sequence,
            }
        )
    return pd.DataFrame(records).loc[:, list(delta.ROW_COLUMNS)]


def _write_synthetic_delta_shard(tmp_path, rows, tamper=False):
    shard_path = tmp_path / delta.shard_filename(0, 1)
    manifest_path = tmp_path / delta.shard_manifest_filename(0, 1)
    arrays = delta.metadata_arrays(rows)
    arrays[delta.FEATURE_NAME] = np.zeros(
        (len(rows), delta.FEATURE_DIMENSION), dtype=np.float32
    )
    if tamper:
        arrays["peptide_design_code"] = arrays["peptide_design_code"].copy()
        arrays["peptide_design_code"][0] = "YY"
    np.savez(str(shard_path), **arrays)
    os.chmod(str(shard_path), 0o600)
    runtime = {
        "python": "3.7.12",
        "torch": "1.12.1",
        "torch_cuda": "11.3",
        "numpy": "1.21.2",
        "batch_size": 64,
        "gpu_name": "synthetic-a100",
        "gpu_capability": [8, 0],
    }
    payload = {
        "schema_version": delta.SCHEMA_VERSION,
        "shard_count": 1,
        "representation": delta.FEATURE_NAME,
        "feature_dimension": delta.FEATURE_DIMENSION,
        "feature_dtype": "float32",
        "model": "MINT use_multimer=True sep_chains=True",
        "layer": 33,
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "mint_source_tree_sha256": "tree",
        "script_sha256": "script",
        "pair_collator_source_sha256": "collator",
        "runtime": runtime,
    }
    contract = delta.canonical_json_sha256(payload)
    indices = rows["row_index"].to_numpy(dtype=np.int64)
    manifest = {
        "schema_version": delta.SCHEMA_VERSION,
        "stage": "extract_shard",
        "model": {
            "checkpoint": {"sha256": "checkpoint"},
            "config": {"sha256": "config"},
            "mint_source_tree_sha256": "tree",
            "script_sha256": "script",
            "pair_collator_source": {"sha256": "collator"},
        },
        "contract": {"sha256": contract, "payload": payload},
        "sharding": {
            "shard_index": 0,
            "shard_count": 1,
            "row_index_sha256": hashlib.sha256(
                indices.astype("<i8").tobytes()
            ).hexdigest(),
        },
        "runtime": runtime,
        "output": {"sha256": delta.sha256_file(shard_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    os.chmod(str(manifest_path), 0o600)
    return shard_path, manifest_path


def test_shard_validation_checks_metadata_and_contract(tmp_path):
    rows = _minimal_prepared_rows()
    shard_path, manifest_path = _write_synthetic_delta_shard(tmp_path, rows)

    _, indices, features, contract = delta._load_and_validate_shard(
        shard_path, manifest_path, rows.astype(str), 0, 1
    )

    assert np.array_equal(indices, np.asarray([0, 1]))
    assert features.shape == (2, delta.FEATURE_DIMENSION)
    assert len(contract) == 64


def test_shard_validation_rejects_metadata_tampering(tmp_path):
    rows = _minimal_prepared_rows()
    shard_path, manifest_path = _write_synthetic_delta_shard(
        tmp_path, rows, tamper=True
    )

    with pytest.raises(ValueError, match="shard metadata mismatch"):
        delta._load_and_validate_shard(
            shard_path, manifest_path, rows.astype(str), 0, 1
        )
