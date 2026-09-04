#!/usr/bin/env python
"""Assemble label-free MINT scores into corrected-panel normalized predictions."""

from pathlib import Path
import argparse
import pandas as pd


TARGET_OPAQUE = "libb-revision-24b85d9e1da9d364c877"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-free-scores", type=Path, required=True)
    p.add_argument("--corrected-panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    scores = pd.read_csv(a.label_free_scores)
    panel = pd.read_csv(a.corrected_panel, usecols=["pair_uid", "peptide_design_code", "affibody_design_code"])
    target = panel.loc[panel.peptide_design_code.eq("AH") & panel.affibody_design_code.eq("LIFTK")]
    assert len(target) == 1
    target_uid = target.iloc[0].pair_uid
    scores.loc[scores.pair_uid.eq(TARGET_OPAQUE), "pair_uid"] = target_uid
    assert set(scores.pair_uid) == set(panel.pair_uid) and len(scores) == 120

    definitions = [
        ("frozen_mint_layer33_archived119_plus_replay_target", "primary_frozen_archived", "primary_frozen_replay"),
        ("frozen_mint_layer33_deterministic_replay120", "primary_frozen_replay", "primary_frozen_replay"),
        ("frozen_mint_layer5_archived119_plus_replay_target", "layer5_archived", "layer5_replay"),
        ("frozen_mint_layer5_deterministic_replay120", "layer5_replay", "layer5_replay"),
        ("lora_one_epoch_saved_checkpoint_exact120", "lora_archived", "lora_checkpoint_inference"),
    ]
    rows = []
    for model, old_column, target_column in definitions:
        values = scores[old_column].copy()
        is_target = scores.pair_uid.eq(target_uid)
        values.loc[is_target] = scores.loc[is_target, target_column]
        assert values.notna().all()
        rows.append(pd.DataFrame({"eval_row_id": scores.pair_uid, "model": model, "seed": "20260811", "score": values}))
    output = pd.concat(rows, ignore_index=True)
    assert len(output) == 600
    a.output.parent.mkdir(parents=True, exist_ok=False)
    output.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
