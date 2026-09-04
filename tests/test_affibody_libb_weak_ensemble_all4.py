import numpy as np
import pandas as pd

from downstream.AffibodyMHC import build_libb_weak_ensemble as base
from downstream.AffibodyMHC import build_libb_weak_ensemble_all4 as all4


def test_rde_heads_are_aligned_then_averaged_in_logit_space(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "EXPECTED_OOF_ROWS", 2)
    rows = []
    values = {
        all4.RDE_HEADS[0]: [0.2, 0.8],
        all4.RDE_HEADS[1]: [0.5, 0.5],
        all4.RDE_HEADS[2]: [0.8, 0.2],
    }
    for model, probability in values.items():
        for index in range(2):
            rows.append(
                {
                    "model": model,
                    "fold": index,
                    "row_id": f"r{index}",
                    "weak_label": index,
                    "probability": probability[index],
                }
            )
    path = tmp_path / "rde.csv.gz"
    pd.DataFrame(rows).sample(frac=1.0, random_state=7).to_csv(path, index=False)
    observed = all4.load_rde_mean_logit_oof(path).sort_values("row_id")
    expected = base.mean_logit_score([values[head] for head in all4.RDE_HEADS])
    assert np.allclose(observed[all4.RDE_MEMBER], expected)
