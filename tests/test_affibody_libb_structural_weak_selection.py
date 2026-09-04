from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path

import pytest

from downstream.AffibodyMHC.lock_libb_structural_weak_selection import (
    _mean_logit_ensemble_metrics,
    _prediction_metrics,
)


FIELDS = ["model", "fold", "row_id", "weak_label", "probability"]


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _rows(model: str, probabilities: tuple[float, float, float]):
    return [
        {
            "model": model,
            "fold": fold,
            "row_id": f"row-{fold}",
            "weak_label": label,
            "probability": probability,
        }
        for fold, label, probability in zip((0, 1, 2), (1, 0, 1), probabilities)
    ]


def test_grouped_prediction_artifact_is_filtered_by_named_model(tmp_path):
    path = tmp_path / "weak_validation_predictions.csv.gz"
    _write_predictions(
        path,
        _rows("first", (0.8, 0.7, 0.6)) + _rows("second", (0.1, 0.2, 0.3)),
    )

    observed = _prediction_metrics(path, "second")

    assert observed["out_of_fold_rows"] == 3
    assert observed["out_of_fold_positive"] == 2
    expected = -(math.log(0.1) + math.log(0.8) + math.log(0.3)) / 3
    assert observed["pooled_weak_validation_log_loss"] == pytest.approx(expected)


def test_prespecified_ensemble_aligns_oof_rows_and_averages_logits(tmp_path):
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    _write_predictions(first, _rows("first", (0.8, 0.2, 0.8)))
    _write_predictions(second, _rows("second", (0.2, 0.8, 0.2)))
    found = {
        "first": {"predictions": first},
        "second": {"predictions": second},
    }

    metrics, sources, derivation_sha256 = _mean_logit_ensemble_metrics(
        ensemble_name="combined",
        members=("first", "second"),
        found=found,
    )

    assert metrics["out_of_fold_rows"] == 3
    assert metrics["pooled_weak_validation_log_loss"] == pytest.approx(math.log(2.0))
    assert [item["model"] for item in sources] == ["first", "second"]
    assert len(derivation_sha256) == 64


def test_prespecified_ensemble_rejects_misaligned_rows(tmp_path):
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    rows = _rows("second", (0.2, 0.8, 0.2))
    rows[1]["row_id"] = "different-row"
    _write_predictions(first, _rows("first", (0.8, 0.2, 0.8)))
    _write_predictions(second, rows)

    with pytest.raises(ValueError, match="not aligned"):
        _mean_logit_ensemble_metrics(
            ensemble_name="combined",
            members=("first", "second"),
            found={
                "first": {"predictions": first},
                "second": {"predictions": second},
            },
        )
