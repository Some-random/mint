import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_mint_pnu_feature_cache as cache


def _uid(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _old_row(index, library, label, pair):
    chain1 = "A" * cache.CHAIN1_LENGTH
    chain2 = "C" * cache.CHAIN2_LENGTH
    return {
        "row_index": str(index),
        "cache_uid": "old-{}".format(pair),
        "source_kind": "weak",
        "library": library,
        "weak_label": str(label),
        "pair_uid": pair,
        "peptide_uid": "pep-{}".format(pair),
        "affibody_uid": "aff-{}".format(pair),
        "peptide_design_code": "AA",
        "affibody_design_code": "AAAA" if library == "LibA" else "AAAAA",
        "r001_count": "3",
        "r009_count": "14" if label else "0",
        "r010_count": "0",
        "pooled_r009_r010_count": "14" if label else "0",
        "upstream_library_local_strict_retention_identity_cold_eligible": "1",
        "chain1_smart_hla_linker_peptide_sequence": chain1,
        "chain2_affibody_sequence": chain2,
        "chain1_sha256": cache.sha256_text(chain1),
        "chain2_sha256": cache.sha256_text(chain2),
        "sequence_pair_sha256": cache.sha256_text(chain1 + "|" + chain2),
    }


def _u_rows():
    records = []
    for library, afflen, cutoff in (("LibA", 4, 13), ("LibB", 5, 12)):
        for index, residue in enumerate("CDEFGH"):
            pep = residue + "A"
            aff = residue * afflen
            pair = "u-{}-{}".format(library, index)
            records.append(
                {
                    "library": library,
                    "pep": pep,
                    "aff": aff,
                    "r009_count": "1",
                    "r010_count": "0",
                    "pooled_r009_r010_count": "1",
                    "positive_cutoff": str(cutoff),
                    "pair_uid": pair,
                    "peptide_uid": "pep-{}".format(pair),
                    "affibody_uid": "aff-{}".format(pair),
                }
            )
    return pd.DataFrame(records, columns=cache.U_COLUMNS)


def test_stable_sample_is_order_independent_and_exact():
    frame = _u_rows()
    first = cache.stable_sample_u(frame, "LibA", 3, seed=17)
    second = cache.stable_sample_u(
        frame.sample(frac=1.0, random_state=9), "LibA", 3, seed=17
    )
    assert first["pair_uid"].tolist() == second["pair_uid"].tolist()
    assert len(first) == 3
    assert cache.stable_sample_u(frame, "LibA", 3, seed=18)[
        "pair_uid"
    ].tolist() != first["pair_uid"].tolist()


def test_assemble_reuses_pn_and_samples_u_per_positive(monkeypatch):
    old = pd.DataFrame(
        [
            _old_row(0, "LibA", 1, "pa"),
            _old_row(1, "LibA", 0, "na"),
            _old_row(2, "LibB", 1, "pb"),
            _old_row(3, "LibB", 0, "nb"),
        ]
    )

    def fake_reconstruct(template, library, pep, aff):
        # Make every synthetic pair sequence-identifiable while preserving the
        # provider chain lengths expected by validation.
        return "D" * cache.CHAIN1_LENGTH, "E" * cache.CHAIN2_LENGTH

    monkeypatch.setattr(cache.old_cache, "reconstruct_chains", fake_reconstruct)
    rows = cache.assemble_rows(
        old, _u_rows(), {"LibA": "x", "LibB": "y"}, u_per_positive=2, seed=7
    )
    assert cache.summarize_rows(rows) == {
        "total": 8,
        "libraries": {
            "LibA": {"P": 1, "N": 1, "U": 2},
            "LibB": {"P": 1, "N": 1, "U": 2},
        },
    }
    pn = rows[rows.class_source.isin(["P", "N"])]
    assert pn["feature_source"].eq("reuse_old_layer33").all()
    assert pn["source_cache_row_index"].tolist() == ["0", "1", "2", "3"]
    assert rows[rows.class_source.eq("U")]["weak_label"].eq("-1").all()


def test_sharding_extracts_only_u_and_uses_prepared_row_modulo():
    frame = pd.DataFrame(
        {
            "row_index": ["0", "1", "2", "3", "4", "5", "6"],
            "class_source": ["P", "N", "U", "U", "U", "U", "U"],
        }
    )
    observed = cache.select_shard_rows(frame, shard_index=1, shard_count=2)
    assert observed["row_index"].tolist() == ["3", "5"]
    assert observed["class_source"].eq("U").all()


def test_resume_requires_hash_bound_complete_pair(tmp_path):
    output = tmp_path / "x.npz"
    manifest = tmp_path / "x.json"
    np.savez(str(output), value=np.array([1]))
    payload = {
        "schema_version": cache.SCHEMA_VERSION,
        "stage": "extract_shard",
        "contract": {"sha256": "contract"},
        "sharding": {"shard_index": 2, "shard_count": 7},
        "output": {"sha256": cache.sha256_file(output)},
    }
    manifest.write_text(json.dumps(payload))
    assert cache._resume_valid_shard(
        output, manifest, {"contract": "contract", "index": 2, "count": 7}
    )
    payload["output"]["sha256"] = "bad"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        cache._resume_valid_shard(
            output, manifest, {"contract": "contract", "index": 2, "count": 7}
        )
