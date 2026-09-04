import numpy as np
import pandas as pd

from downstream.AffibodyMHC.build_libb_selection_missed_candidate_universe import (
    RAW_ROUNDS,
    _candidate_frame,
)


def _round_counts(peptide):
    result = {index: {} for index in RAW_ROUNDS}
    result[0][peptide] = pd.Series({"AAAAI": 1}, dtype=np.uint32)
    result[1][peptide] = pd.Series({"AAAAF": 4}, dtype=np.uint32)
    result[9][peptide] = pd.Series({"AAAAD": 8}, dtype=np.uint32)
    result[10][peptide] = pd.Series({"AAAAD": 5}, dtype=np.uint32)
    return result


def test_candidate_frame_excludes_measured_and_pooled_positive_and_keeps_negative():
    peptide = "AF"
    codes = np.asarray(["AAAAA", "AAAAD", "AAAAF", "AAAAI"], dtype=object)
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
            index=pd.Index(["AAAAD", "AAAAF"], name="aff"),
        )
    }

    candidates, excluded = _candidate_frame(
        peptide=peptide,
        peptide_full_sequence="SLLAFITQV",
        affibody_codes=codes,
        affibody_sequences=sequences,
        affibody_uids=uids,
        affibody_hashes=hashes,
        chain1="CHAIN1",
        measured_affibodies={"AAAAA"},
        selected_positive_affibodies={"AAAAD"},
        round_counts=_round_counts(peptide),
        weak_annotations=annotations,
        seen_training_affibodies={"AAAAI"},
    )

    assert candidates["affibody_design_code"].tolist() == ["AAAAF", "AAAAI"]
    negative = candidates.loc[candidates["affibody_design_code"].eq("AAAAF")].iloc[0]
    assert negative["prior_weak_label"] == 0
    assert bool(negative["high_confidence_weak_negative"])
    unseen = candidates.loc[candidates["affibody_design_code"].eq("AAAAI")].iloc[0]
    assert pd.isna(unseen["prior_weak_label"])
    assert bool(unseen["observed_in_any_raw_round"])
    assert bool(unseen["affibody_identity_seen_in_strict_training"])

    measured = excluded.loc[excluded["affibody_design_code"].eq("AAAAA")].iloc[0]
    selected = excluded.loc[excluded["affibody_design_code"].eq("AAAAD")].iloc[0]
    assert bool(measured["excluded_directly_measured"])
    assert not bool(measured["excluded_pooled_r009_r010_top2pct"])
    assert not bool(selected["excluded_directly_measured"])
    assert bool(selected["excluded_pooled_r009_r010_top2pct"])
    assert selected["pooled_r009_r010_count"] == 13
