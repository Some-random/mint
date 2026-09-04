import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_liba_selection_missed_candidate_universe as universe
from downstream.AffibodyMHC.build_sequence_table import OMITTED_LINKER, fill_template
from downstream.AffibodyMHC.code_only_baseline import opaque_id


def _round_counts(peptide):
    result = {index: {} for index in universe.RAW_ROUNDS}
    result[0][peptide] = pd.Series({"AAAI": 1}, dtype=np.uint64)
    result[1][peptide] = pd.Series({"AAAF": 4}, dtype=np.uint64)
    result[9][peptide] = pd.Series({"AAAD": 8}, dtype=np.uint64)
    result[10][peptide] = pd.Series({"AAAD": 5}, dtype=np.uint64)
    return result


def test_candidate_frame_excludes_measured_and_positive_and_keeps_annotations():
    peptide = "AF"
    codes = np.asarray(["AAAA", "AAAD", "AAAF", "AAAI"], dtype=object)
    sequences = np.asarray([f"SEQ_{value}" for value in codes], dtype=object)
    uids = np.asarray([f"UID_{value}" for value in codes], dtype=object)
    hashes = np.asarray([f"HASH_{value}" for value in codes], dtype=object)
    annotations = {
        peptide: pd.DataFrame(
            {
                "weak_label": [1, 0],
                "weak_label_source": [
                    "pooled_r009_r010_top_fraction",
                    "r001_absent_r002_r014",
                ],
            },
            index=pd.Index(["AAAD", "AAAF"], name="aff"),
        )
    }

    candidates, excluded = universe._candidate_frame(
        peptide=peptide,
        peptide_full_sequence="SLLAFITQV",
        affibody_codes=codes,
        affibody_sequences=sequences,
        affibody_uids=uids,
        affibody_hashes=hashes,
        chain1="CHAIN1",
        measured_affibodies={"AAAA"},
        selected_positive_affibodies={"AAAD"},
        round_counts=_round_counts(peptide),
        weak_annotations=annotations,
        seen_training_affibodies={"AAAI"},
    )

    assert candidates["affibody_design_code"].tolist() == ["AAAF", "AAAI"]
    assert all(f"r{index:03d}_count" in candidates for index in universe.RAW_ROUNDS)
    negative = candidates.loc[candidates["affibody_design_code"].eq("AAAF")].iloc[0]
    assert negative["prior_weak_label"] == 0
    assert bool(negative["high_confidence_weak_negative"])
    observed = candidates.loc[candidates["affibody_design_code"].eq("AAAI")].iloc[0]
    assert pd.isna(observed["prior_weak_label"])
    assert bool(observed["observed_in_any_raw_round"])
    assert bool(observed["affibody_identity_seen_in_strict_training"])
    assert len(observed["sequence_pair_sha256"]) == 64

    measured = excluded.loc[excluded["affibody_design_code"].eq("AAAA")].iloc[0]
    selected = excluded.loc[excluded["affibody_design_code"].eq("AAAD")].iloc[0]
    assert bool(measured["excluded_directly_measured"])
    assert not bool(measured["excluded_pooled_r009_r010_top2pct"])
    assert not bool(selected["excluded_directly_measured"])
    assert bool(selected["excluded_pooled_r009_r010_top2pct"])
    assert selected["pooled_r009_r010_count"] == 13


def test_pool_first_top_fraction_is_tie_inclusive_and_input_sized():
    round9 = pd.DataFrame(
        {
            "pep": ["AF", "DL", "EA"],
            "aff": ["AAAA", "AAAD", "AADA"],
            "count": [10, 8, 1],
        }
    )
    round10 = pd.DataFrame(
        {
            "pep": ["AF", "DL", "TL"],
            "aff": ["AAAA", "AAAD", "AADD"],
            "count": [0, 2, 1],
        }
    )

    selected, summary = universe.pooled_positive_membership(
        round9, round10, top_fraction=0.25
    )

    # The four-pair union requests one row, but both count-10 pairs are kept.
    assert summary == {
        "pooled_union_rows": 4,
        "nominal_top_fraction_rank": 1,
        "inclusive_count_cutoff": 10,
        "selected_rows_including_boundary_ties": 2,
        "boundary_tie_expansion_rows": 1,
    }
    assert set(zip(selected["pep"], selected["aff"])) == {
        ("AF", "AAAA"),
        ("DL", "AAAD"),
    }
    assert selected["pooled_global_count_rank_1_based"].tolist() == [1, 1]
    assert selected["pooled_selected_order_1_based"].tolist() == [1, 2]


def _synthetic_template():
    chain1 = "A" * 261 + "SLLXXITQV"
    affibody = list("A" * 58)
    for position in (13, 17, 27, 31):
        affibody[position - 1] = "X"
    template = chain1 + OMITTED_LINKER + "".join(affibody)
    assert len(template) == 342 and template.count("X") == 6
    return template


def _write_raw_round(path, rows):
    frame = pd.DataFrame(rows, columns=universe.RAW_COLUMNS)
    frame.to_csv(path, sep="\t", index=False)


def test_small_end_to_end_build_is_outcome_blind_and_separates_exclusions(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(universe, "AFFIBODY_DESIGN_ALPHABET", tuple("AD"))
    monkeypatch.setattr(universe, "validate_private_output_path", lambda path, _: path.resolve())
    template = _synthetic_template()
    monkeypatch.setattr(
        universe,
        "load_provider_templates",
        lambda _: ({"LibA": template}, "synthetic/sequence_deck.pptx", "d" * 64),
    )

    peptide_codes = ["AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL"]
    all_affibodies = ["".join(code) for code in itertools.product("AD", repeat=4)]
    measured_affibodies = all_affibodies[:12]
    positive_affibody = all_affibodies[12]
    negative_affibody = all_affibodies[13]

    panel_rows = []
    for peptide in peptide_codes:
        for affibody in measured_affibodies:
            construct = fill_template(template, peptide + affibody)
            panel_rows.append(
                {
                    "pair_uid": opaque_id("LibA", peptide, affibody),
                    "library": "LibA",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "chain1_smart_hla_linker_peptide_sequence": construct[:270],
                    "chain2_affibody_sequence": construct[-58:],
                    # The loader must never request either sentinel outcome.
                    "target_retention": "SECRET_RETENTION",
                    "target_binder": "SECRET_BINDER",
                }
            )
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(panel_rows).to_csv(panel_path, index=False)

    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "LibA Raw data"
    raw_dir.mkdir(parents=True)
    for round_index in universe.RAW_ROUNDS:
        if round_index in (9, 10):
            rows = [
                (peptide, positive_affibody, 10, 0.0, 1.0)
                for peptide in peptide_codes
            ]
        elif round_index == 1:
            rows = [
                (peptide, negative_affibody, 3, 0.0, 1.0)
                for peptide in peptide_codes
            ] + [("YY", "AAAA", 3, 0.0, 1.0)]
        else:
            rows = [("YY", "AAAA", 1, 0.0, 1.0)]
        _write_raw_round(
            raw_dir / f"xxylibA_R1_R{round_index:03d}_count_freq_pvalue.tsv",
            rows,
        )

    weak_rows = [
        {
            "library": "LibA",
            "pep": "YY",
            "aff": "AAAA",
            "weak_label": 0,
            "weak_label_source": "r001_absent_r002_r014",
            "within_declared_library_alphabet": 1,
            "negative_r001_count_ge_3": 1,
            "strict_retention_identity_cold_eligible": 1,
        }
    ]
    for peptide in peptide_codes:
        weak_rows.extend(
            [
                {
                    "library": "LibA",
                    "pep": peptide,
                    "aff": positive_affibody,
                    "weak_label": 1,
                    "weak_label_source": "pooled_r009_r010_top_fraction",
                    "within_declared_library_alphabet": 1,
                    "negative_r001_count_ge_3": 0,
                    "strict_retention_identity_cold_eligible": 0,
                },
                {
                    "library": "LibA",
                    "pep": peptide,
                    "aff": negative_affibody,
                    "weak_label": 0,
                    "weak_label_source": "r001_absent_r002_r014",
                    "within_declared_library_alphabet": 1,
                    "negative_r001_count_ge_3": 1,
                    "strict_retention_identity_cold_eligible": 0,
                },
            ]
        )
    weak_path = tmp_path / "weak.csv"
    pd.DataFrame(weak_rows).to_csv(weak_path, index=False)
    sequence_zip = tmp_path / "sequence.zip"
    sequence_zip.write_bytes(b"synthetic zip placeholder")
    output = tmp_path / "candidate_universe"
    args = SimpleNamespace(
        raw_root=raw_root,
        measured_panel=panel_path,
        sequence_zip=sequence_zip,
        weak_labels=weak_path,
        output_dir=output,
        chunksize=4,
    )

    universe.run(args)

    manifest = universe.json.loads((output / "manifest.json").read_text())
    assert manifest["input_data_contract"]["retention_outcome_columns_read"] == []
    assert manifest["scope"]["universe_rows_before_exclusion"] == 9 * 2**4
    assert manifest["exclusions"] == {
        "directly_measured_exact_pairs": 108,
        "pooled_positive_target_pairs": 9,
        "pooled_positive_nonmeasured_pairs": 9,
        "measured_and_pooled_positive_overlap": 0,
        "excluded_union_rows": 117,
        "partition_policy": (
            "measured pairs and unmeasured pooled-positive pairs are emitted in "
            "separate files; measured-and-positive overlaps remain in the measured tier"
        ),
    }
    assert manifest["outputs"]["candidate_rows"] == 27
    measured = pd.read_parquet(output / "measured_exclusions.parquet")
    controls = pd.read_parquet(output / "pooled_positive_unmeasured_controls.parquet")
    assert len(measured) == 108
    assert len(controls) == 9
    assert not set(measured["pair_uid"]) & set(controls["pair_uid"])
    assert controls["candidate_tier"].eq(
        "existing_target_pooled_positive_unmeasured_liba_design"
    ).all()
    assert controls["pooled_global_count_rank_1_based"].eq(1).all()
    assert {
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_9mer_sequence",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "directly_measured",
        "pooled_r009_r010_top2pct",
    }.issubset(controls.columns)
    assert (~controls["directly_measured"]).all()
    assert controls["pooled_r009_r010_top2pct"].all()
    partition = pd.read_parquet(output / "candidates_by_peptide/peptide_AF.parquet")
    assert len(partition) == 3
    assert all(f"r{index:03d}_count" in partition for index in universe.RAW_ROUNDS)
    assert "target_retention" not in partition and "target_binder" not in partition
    assert "SECRET_RETENTION" not in (output / "manifest.json").read_text()

    with pytest.raises(ValueError, match="output directory exists"):
        universe.run(args)
