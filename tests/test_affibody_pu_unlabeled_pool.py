import os

import pandas as pd
import pytest

from downstream.AffibodyMHC.build_pu_unlabeled_pool import (
    OUTPUT_COLUMNS,
    _write_json,
    _write_pool,
    build_library_pool,
    validate_pool_contract,
)


def _retention_identities():
    rows = []
    for peptide in ("AA", "AD"):
        for affibody in ("AAAA", "AAAD"):
            rows.append(
                {
                    "library": "LibA",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                }
            )
    return pd.DataFrame(rows)


def test_pool_outer_joins_zero_fills_and_uses_strict_cutoff_before_filters(monkeypatch):
    # The small fixture has four retention identities rather than the production
    # 108-cell LibA grid.
    monkeypatch.setitem(
        __import__(
            "downstream.AffibodyMHC.build_pu_unlabeled_pool",
            fromlist=["LIBRARY_SPECS"],
        ).LIBRARY_SPECS["LibA"],
        "grid_rows",
        4,
    )
    round9 = pd.DataFrame(
        {
            "pep": ["AF", "AE", "AI", "AK", "AC"],
            "aff": ["AAAE", "AAAF", "AAAI", "AAAK", "AAAE"],
            "count": [7, 13, 3, 4, 1],
        }
    )
    round10 = pd.DataFrame(
        {
            "pep": ["AF", "AI", "AL"],
            "aff": ["AAAE", "AAAI", "AAAL"],
            "count": [5, 9, 2],
        }
    )

    pool, audit = build_library_pool(
        round9, round10, _retention_identities(), "LibA", cutoff=13
    )

    # AF sums to 12 and stays; AE equals 13 and is excluded. AI sums to 12.
    # AL is present only in R010 and is zero-filled in R009. AC is below the
    # cutoff but is removed by the declared alphabet filter.
    assert set(zip(pool["pep"], pool["aff"])) == {
        ("AF", "AAAE"),
        ("AI", "AAAI"),
        ("AK", "AAAK"),
        ("AL", "AAAL"),
    }
    al = pool.loc[pool["pep"].eq("AL")].iloc[0]
    assert al["r009_count"] == 0
    assert al["r010_count"] == 2
    assert audit["outer_union_rows"] == 6
    assert audit["rows_strictly_below_positive_cutoff"] == 5
    assert audit["rows_after_declared_alphabet_filter"] == 4


def test_pool_excludes_either_retention_partner_identity(monkeypatch):
    monkeypatch.setitem(
        __import__(
            "downstream.AffibodyMHC.build_pu_unlabeled_pool",
            fromlist=["LIBRARY_SPECS"],
        ).LIBRARY_SPECS["LibA"],
        "grid_rows",
        4,
    )
    round9 = pd.DataFrame(
        {
            "pep": ["AA", "AF", "AF", "AI"],
            "aff": ["AAAE", "AAAA", "AAAF", "AAAI"],
            "count": [1, 1, 1, 1],
        }
    )
    round10 = pd.DataFrame(columns=["pep", "aff", "count"])

    pool, audit = build_library_pool(
        round9, round10, _retention_identities(), "LibA", cutoff=13
    )

    assert set(zip(pool["pep"], pool["aff"])) == {("AF", "AAAF"), ("AI", "AAAI")}
    assert audit["rows_sharing_retention_peptide"] == 1
    assert audit["rows_sharing_retention_affibody"] == 1
    assert audit["rows_sharing_either_retention_partner"] == 2


def test_contract_rejects_overlap_with_existing_positive_or_negative(monkeypatch):
    monkeypatch.setattr(
        "downstream.AffibodyMHC.build_pu_unlabeled_pool.LIBRARIES", ("LibA",)
    )
    monkeypatch.setattr(
        "downstream.AffibodyMHC.build_pu_unlabeled_pool.EXPECTED_POOLED_CUTOFF",
        {"LibA": 13},
    )
    pool = pd.DataFrame(
        [
            {
                "library": "LibA",
                "pep": "AF",
                "aff": "AAAF",
                "r009_count": 1,
                "r010_count": 0,
                "pooled_r009_r010_count": 1,
                "positive_cutoff": 13,
                "pair_uid": "same",
                "peptide_uid": "pep",
                "affibody_uid": "aff",
            }
        ],
        columns=OUTPUT_COLUMNS,
    )
    labels = pd.DataFrame({"pair_uid": ["same"]})
    retention = _retention_identities().iloc[0:0]

    with pytest.raises(ValueError, match="overlaps existing P/N"):
        validate_pool_contract(
            pool, labels, retention, expected_rows={"LibA": 1}
        )


def test_contract_rejects_changed_expected_count(monkeypatch):
    monkeypatch.setattr(
        "downstream.AffibodyMHC.build_pu_unlabeled_pool.LIBRARIES", ("LibA",)
    )
    monkeypatch.setattr(
        "downstream.AffibodyMHC.build_pu_unlabeled_pool.EXPECTED_POOLED_CUTOFF",
        {"LibA": 13},
    )
    pool = pd.DataFrame(columns=OUTPUT_COLUMNS)
    labels = pd.DataFrame({"pair_uid": []})
    retention = _retention_identities().iloc[0:0]

    with pytest.raises(ValueError, match="U count changed"):
        validate_pool_contract(
            pool, labels, retention, expected_rows={"LibA": 1}
        )


def test_private_writers_set_file_mode_0600(tmp_path):
    pool = pd.DataFrame(columns=OUTPUT_COLUMNS)
    pool_path = tmp_path / "unlabeled_pool.csv.gz"
    manifest_path = tmp_path / "manifest.json"

    _write_pool(pool, pool_path)
    _write_json({"ok": True}, manifest_path)

    assert oct(os.stat(pool_path).st_mode & 0o777) == "0o600"
    assert oct(os.stat(manifest_path).st_mode & 0o777) == "0o600"
    reread = pd.read_csv(pool_path)
    assert tuple(reread.columns) == OUTPUT_COLUMNS
