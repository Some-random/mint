import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import build_weak_oof_ensemble_lock as ensemble


def _synthetic_inputs(tmp_path, *, fit_stacker=True):
    root = tmp_path / "private_data"
    root.mkdir()
    validation_rows = []
    for fold in range(3):
        peptide = f"peptide-{fold}"
        for index, label in enumerate((0, 1)):
            validation_rows.append(
                {
                    "library": "LibA",
                    "pair_uid": f"v-{fold}-{index}",
                    "chain1_sha256": peptide,
                    "chain2_sha256": f"affibody-{fold}-{index}",
                    "weak_label": label,
                    "fold": fold,
                    "role": "validation",
                }
            )
    membership_rows = []
    for fold in range(3):
        membership_rows.extend(row for row in validation_rows if row["fold"] == fold)
        source_fold = (fold + 1) % 3
        for row in validation_rows:
            if row["fold"] == source_fold:
                copied = dict(row)
                copied["fold"] = fold
                copied["role"] = "train"
                membership_rows.append(copied)
        membership_rows.append(
            {
                "library": "LibA",
                "pair_uid": f"guard-{fold}",
                "chain1_sha256": f"guard-peptide-{fold}",
                "chain2_sha256": f"guard-affibody-{fold}",
                "weak_label": fold % 2,
                "fold": fold,
                "role": "guard",
            }
        )
    membership_path = root / "fold_membership.csv"
    pd.DataFrame(membership_rows).to_csv(membership_path, index=False)

    audit_paths = []
    oof_paths = []
    member_names = ["good_model", "mixed_model"]
    for member_index, member in enumerate(member_names):
        audit_path = root / f"{member}.audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "synthetic-oof-v1",
                    "library": "LibA",
                    "retention_labels_read": False,
                }
            ),
            encoding="utf-8",
        )
        rows = []
        for row in validation_rows:
            label = row["weak_label"]
            if member_index == 0:
                probability = 0.9 if label else 0.1
            else:
                probability = 0.7 if (label == (row["fold"] % 2)) else 0.3
            rows.append(
                {
                    "model": member,
                    "fold": row["fold"],
                    "row_id": row["pair_uid"],
                    "weak_label": label,
                    "probability": probability,
                }
            )
        oof_path = root / f"{member}.oof.csv.gz"
        pd.DataFrame(rows).to_csv(oof_path, index=False)
        audit_paths.append(audit_path)
        oof_paths.append(oof_path)

    config = {
        "schema_version": ensemble.CONFIG_SCHEMA_VERSION,
        "library": "LibA",
        "fold_membership": str(membership_path),
        "common_row_policy": "require_exact",
        "expected_folds": 3,
        "expected_common_oof_rows": 6,
        "expected_common_oof_positive": 3,
        "fit_nonnegative_stacker": fit_stacker,
        "stack_l2_grid": [0.0, 0.1],
        "max_candidates_per_peptide": 10,
        "members": [
            {
                "name": name,
                "oof_path": str(oof_path),
                "audit_path": str(audit_path),
                "model_column": "model",
                "model_value": name,
                "probability_column": "probability",
                "inference_logit_column": f"{name}_logit",
            }
            for name, oof_path, audit_path in zip(member_names, oof_paths, audit_paths)
        ],
    }
    config_path = root / "ensemble_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, root / "ensemble_output"


def test_mean_logit_uses_equal_weight_in_log_odds_space():
    left = np.asarray([0.2, 0.8])
    right = np.asarray([0.5, 0.5])
    expected = ensemble.sigmoid(
        (ensemble.clipped_logit(left) + ensemble.clipped_logit(right)) / 2.0
    )
    assert np.allclose(ensemble.mean_logit_score([left, right]), expected)


def test_macro_within_peptide_ap_gives_each_evaluable_peptide_one_vote():
    result = ensemble.macro_within_peptide_metrics(
        ["a", "a", "b", "b", "c", "c"],
        [1, 0, 1, 0, 1, 1],
        [0.9, 0.1, 0.1, 0.9, 0.2, 0.3],
    )
    assert result["within_peptide_evaluable"] == 2
    assert result["within_peptide_ap"] == pytest.approx(0.75)


def test_candidate_tie_policy_prefers_log_loss_then_parsimony_then_name():
    metrics = pd.DataFrame(
        [
            {"candidate": "z", "within_peptide_ap": 0.8, "pooled_log_loss": 0.3, "member_count": 1},
            {"candidate": "b", "within_peptide_ap": 0.8, "pooled_log_loss": 0.2, "member_count": 2},
            {"candidate": "a", "within_peptide_ap": 0.8, "pooled_log_loss": 0.2, "member_count": 2},
        ]
    )
    assert ensemble.select_one_candidate(metrics)["candidate"] == "a"


def test_f1_cutoff_tie_is_deterministic():
    result = ensemble.select_f1_threshold([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1])
    assert result["threshold"] == 4.0
    assert result["precision"] == 1.0
    assert result["f1"] == pytest.approx(2.0 / 3.0)


def test_membership_validation_rejects_partner_leakage(tmp_path):
    path = tmp_path / "membership.csv"
    pd.DataFrame(
        [
            {
                "pair_uid": "train",
                "fold": 0,
                "role": "train",
                "weak_label": 0,
                "chain1_sha256": "shared",
                "chain2_sha256": "a",
            },
            {
                "pair_uid": "validation",
                "fold": 0,
                "role": "validation",
                "weak_label": 1,
                "chain1_sha256": "shared",
                "chain2_sha256": "b",
            },
            {
                "pair_uid": "guard",
                "fold": 0,
                "role": "guard",
                "weak_label": 0,
                "chain1_sha256": "g",
                "chain2_sha256": "g",
            },
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least three|expected folds|not peptide-cold"):
        ensemble.load_and_validate_membership(path, "LibA", 1)


def test_outcome_column_is_rejected_before_full_load(tmp_path):
    path = tmp_path / "oof.csv"
    pd.DataFrame({"row_id": ["x"], "target_retention": [87.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="forbidden outcome columns"):
        ensemble._read_csv_outcome_safe(path, "OOF")


def test_member_audit_rejects_nested_retrospective_metrics(tmp_path):
    path = tmp_path / "model.audit.json"
    path.write_text(
        json.dumps(
            {
                "library": "LibA",
                "retention_labels_read": False,
                "evidence_positive_control": {
                    "retrospective_metrics": {"average_precision": 0.9}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outcome-bearing field"):
        ensemble._load_audit(path, "model", "LibA")


def test_end_to_end_synthetic_lock_has_formula_cutoff_and_no_outcome_access(tmp_path):
    config_path, output = _synthetic_inputs(tmp_path, fit_stacker=True)
    manifest = ensemble.run(config_path, output)

    assert manifest["library"] == "LibA"
    assert manifest["retention_labels_read"] is False
    assert manifest["alignment"]["common_rows"] == 6
    selected = json.loads((output / "selected_score.lock.json").read_text())
    assert selected["retention_labels_read"] is False
    assert selected["formula"]["output_transform"] == "sigmoid"
    assert selected["cutoff"]["decision_rule"] == "score >= threshold"
    assert selected["cutoff"]["max_candidates_per_peptide"] == 10

    stack = json.loads((output / "nonnegative_stacker.lock.json").read_text())
    assert all(component["weight"] >= 0.0 for component in stack["formula"]["components"])
    assert stack["formula"]["linear_predictor"].startswith("intercept + sum")
    assert (output / "best_equal_logit_ensemble.lock.json").is_file()


def test_optional_stacker_can_be_disabled(tmp_path):
    config_path, output = _synthetic_inputs(tmp_path, fit_stacker=False)
    manifest = ensemble.run(config_path, output)
    assert manifest["stacker"]["enabled"] is False
    assert not (output / "nonnegative_stacker.lock.json").exists()
    stack = json.loads((output / "stacker_locked_parameters.json").read_text())
    assert stack["enabled"] is False
