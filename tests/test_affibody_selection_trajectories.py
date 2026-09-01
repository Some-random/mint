from pathlib import Path

import numpy as np

from downstream.AffibodyMHC.build_selection_trajectories import (
    _scan_round,
    make_trajectory_features,
)


def test_scan_round_preserves_literal_na_and_zero_for_unobserved(tmp_path):
    path = Path(tmp_path) / "round.tsv"
    path.write_text(
        "pep\taff\tcount\tfrequency\tpvalue\n"
        "NA\tABC\t3\t0.3\t0.1\n"
        "XY\tDEF\t7\t0.7\t0.2\n"
    )
    count = np.zeros(3, dtype=np.int64)
    frequency = np.zeros(3, dtype=np.float64)
    presence = np.zeros(3, dtype=np.uint8)
    stats = _scan_round(
        path,
        {"NA|ABC": 1, "MISSING|KEY": 2},
        count,
        frequency,
        presence,
        chunk_size=1,
    )
    assert count.tolist() == [0, 3, 0]
    assert frequency.tolist() == [0.0, 0.3, 0.0]
    assert presence.tolist() == [0, 1, 0]
    assert stats == {
        "source_rows": 2,
        "total_count": 10,
        "frequency_sum": 1.0,
        "matched_target_pairs": 1,
    }


def test_trajectory_features_are_finite_and_name_aligned():
    count = np.zeros((2, 15), dtype=np.int64)
    frequency = np.zeros((2, 15), dtype=np.float64)
    presence = np.zeros((2, 15), dtype=np.uint8)
    count[0, 9:11] = [2, 4]
    frequency[0, 9:11] = [0.02, 0.04]
    presence[0, 9:11] = 1
    round_stats = {
        "{}:R{:03d}".format(library, round_index): {
            "total_count": 100,
        }
        for library in ("LibA", "LibB")
        for round_index in range(15)
    }
    features, names = make_trajectory_features(
        count,
        frequency,
        presence,
        np.asarray(["LibA", "LibB"]),
        round_stats,
    )
    assert features.shape == (2, 63)
    assert len(names) == features.shape[1]
    assert np.isfinite(features).all()
    name_to_index = {name: index for index, name in enumerate(names.astype(str))}
    assert features[0, name_to_index["observed_in_both_r009_r010"]] == 1
    assert features[1, name_to_index["rounds_observed"]] == 0
    assert np.isclose(
        features[0, name_to_index["censored_log2_frequency_change_r009_to_r010"]],
        np.log2(0.04 + 0.005) - np.log2(0.02 + 0.005),
    )
    assert features[1, name_to_index["censored_log2_frequency_change_r009_to_r010"]] == 0


def test_scan_round_rejects_duplicate_target_within_chunk(tmp_path):
    path = Path(tmp_path) / "duplicate.tsv"
    path.write_text(
        "pep\taff\tcount\tfrequency\tpvalue\n"
        "NA\tABC\t3\t0.3\t0.1\n"
        "NA\tABC\t7\t0.7\t0.2\n"
    )
    count = np.zeros(1, dtype=np.int64)
    frequency = np.zeros(1, dtype=np.float64)
    presence = np.zeros(1, dtype=np.uint8)
    import pytest

    with pytest.raises(ValueError, match="duplicate target key within one raw chunk"):
        _scan_round(
            path,
            {"NA|ABC": 0},
            count,
            frequency,
            presence,
            chunk_size=10,
        )
