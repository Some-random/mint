import numpy as np
import pandas as pd

from downstream.AffibodyMHC.compare_libb_models_corrected120 import (
    MODEL_COLUMNS,
    build_comparison,
)


def _frames():
    retention_rows = []
    for peptide_index in range(12):
        for affibody_index in range(10):
            binder = int(affibody_index < (6 if peptide_index == 0 else 5))
            retention = 80.0 + affibody_index if binder else 10.0 + affibody_index
            row = {
                "eval_row_id": f"r{peptide_index:02d}{affibody_index:02d}",
                "peptide_design_code": f"p{peptide_index:02d}",
                "affibody_design_code": f"a{affibody_index:02d}",
                "target_retention": retention,
                "target_binder": binder,
            }
            for _, _, score_column, _ in MODEL_COLUMNS:
                row[score_column] = retention / 100.0
            retention_rows.append(row)

    oof_rows = []
    for peptide_index in range(3):
        for affibody_index in range(4):
            label = int(affibody_index < 2)
            row = {
                "row_id": f"o{peptide_index}{affibody_index}",
                "peptide_id": f"wp{peptide_index}",
                "weak_label": label,
            }
            for _, _, _, score_column in MODEL_COLUMNS:
                row[score_column] = 0.9 if label else 0.1
            oof_rows.append(row)
    return pd.DataFrame(retention_rows), pd.DataFrame(oof_rows)


def test_build_comparison_is_peptide_conditioned_and_complete():
    retention, oof = _frames()
    result = build_comparison(retention, oof)
    assert result["model"].tolist() == [row[0] for row in MODEL_COLUMNS]
    assert len(result) == 7
    assert np.allclose(result["within_peptide_ap"], 1.0)
    assert np.allclose(result["within_peptide_auroc"], 1.0)
    assert np.allclose(result["within_peptide_spearman"], 1.0)
    assert (result["within_peptide_binary_evaluable_peptides"] == 12).all()
    assert (result["within_peptide_spearman_evaluable_peptides"] == 12).all()
    assert np.allclose(result["weak_oof_macro_ap"], 1.0)
    assert (result["weak_oof_evaluable_peptides"] == 3).all()
    assert not any("global" in column for column in result.columns)


def test_rejects_incomplete_retention_panel():
    retention, oof = _frames()
    try:
        build_comparison(retention.iloc[:-1], oof)
    except ValueError as exc:
        assert "expected 120" in str(exc)
    else:
        raise AssertionError("incomplete panel was accepted")
