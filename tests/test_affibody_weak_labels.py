from pathlib import Path

import pandas as pd

from downstream.AffibodyMHC.build_selection_weak_labels import (
    discover_raw_round_files,
    pooled_top_fraction,
    provider_negative_rows,
    within_declared_alphabet,
)


def test_pooled_top_fraction_outer_joins_and_keeps_cutoff_ties():
    round9 = pd.DataFrame(
        {
            "pep": ["AA", "AD", "AE"],
            "aff": ["AAAA", "AAAA", "AAAA"],
            "count": [7, 4, 2],
        }
    )
    round10 = pd.DataFrame(
        {
            "pep": ["AA", "AF", "AG"],
            "aff": ["AAAA", "AAAA", "AAAA"],
            "count": [1, 4, 4],
        }
    )

    selected, cutoff, requested_rank, pooled_rows = pooled_top_fraction(
        round9, round10, top_fraction=0.4
    )

    assert pooled_rows == 5
    assert requested_rank == 2
    assert cutoff == 4
    assert set(zip(selected["pep"], selected["aff"])) == {
        ("AA", "AAAA"),
        ("AD", "AAAA"),
        ("AF", "AAAA"),
        ("AG", "AAAA"),
    }
    aa = selected.loc[selected["pep"].eq("AA")].iloc[0]
    assert aa["r009_count"] == 7
    assert aa["r010_count"] == 1
    assert aa["pooled_r009_r010_count"] == 8


def test_provider_negative_rows_uses_r001_set_difference():
    reference = pd.DataFrame(
        {
            "pep": ["AA", "AD", "AE"],
            "aff": ["AAAA", "AAAA", "AAAA"],
            "count": [5, 4, 3],
        }
    )
    later_a = pd.DataFrame({"pep": ["AA"], "aff": ["AAAA"]})
    later_b = pd.DataFrame({"pep": ["AE"], "aff": ["AAAA"]})

    result = provider_negative_rows(reference, [later_a, later_b])

    assert list(result["pep"]) == ["AD"]
    assert list(result["count"]) == [4]


def test_declared_library_alphabets_are_position_family_specific():
    assert within_declared_alphabet("LibA", "AH", "HHPQ")
    assert not within_declared_alphabet("LibB", "AH", "HHPQS")
    assert within_declared_alphabet("LibB", "AD", "MMSVY")
    assert not within_declared_alphabet("LibA", "CM", "MMSV")


def test_round_file_discovery_handles_provider_filename_variants(tmp_path):
    names = {
        "LibA": lambda index: (
            "xxylibA_R1_R{:03d}_count_freq_pvalue.tsv".format(index)
            if index == 0 or index >= 11
            else "xxylibA_R1_{:03d}_count_freq_pvalue.tsv".format(index)
        ),
        "LibB": lambda index: (
            "xxylibB_R000_count__freq_pvalue.tsv"
            if index == 0
            else "xxylibB_R{:03d}{}_count_freq_pvalue.tsv".format(
                index, "n" if index >= 11 else ""
            )
        ),
    }
    for library, directory_name in (("LibA", "LibA Raw data"), ("LibB", "LibB Raw data")):
        directory = Path(tmp_path) / directory_name
        directory.mkdir()
        for index in range(15):
            (directory / names[library](index)).write_text("", encoding="utf-8")

    discovered = discover_raw_round_files(Path(tmp_path))

    assert set(discovered) == {"LibA", "LibB"}
    assert set(discovered["LibA"]) == set(range(15))
    assert set(discovered["LibB"]) == set(range(15))
