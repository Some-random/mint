import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import finalize_liba_model_comparison as finalizer


PEPTIDES = ["AF", "DL", "DP", "EA", "KF", "LA", "LL", "NF", "TL"]
AFFIBODIES = [f"A{index:02d}" for index in range(12)]
BINDER_COUNTS = {
    "AF": 11,
    "DL": 0,
    "DP": 0,
    "EA": 0,
    "KF": 10,
    "LA": 0,
    "LL": 10,
    "NF": 4,
    "TL": 3,
}


def _membership(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _panel():
    rows = []
    for peptide_index, peptide in enumerate(PEPTIDES):
        for affibody_index, affibody in enumerate(AFFIBODIES):
            binder = affibody_index < BINDER_COUNTS[peptide]
            retention = (
                99.0 - affibody_index - peptide_index / 100.0
                if binder
                else 50.0 - affibody_index - peptide_index / 100.0
            )
            rows.append(
                {
                    "pair_uid": f"{peptide}-{affibody}",
                    "peptide_design_code": peptide,
                    "affibody_design_code": affibody,
                    "target_retention": retention,
                    "target_binder": int(binder),
                }
            )
    return pd.DataFrame(rows)


def _target_free_predictions():
    panel = _panel()
    blocks = []
    specifications = [
        ("additive", "fixed", 0.0),
        ("nonlinear", "11", 0.01),
        ("nonlinear", "12", -0.01),
    ]
    for model, seed, offset in specifications:
        block = pd.DataFrame(
            {
                "eval_row_id": panel["pair_uid"],
                "model": model,
                "seed": seed,
                "score": np.clip(panel["target_retention"] / 100.0 + offset, 0.001, 0.999),
                "source_role": "candidate",
            }
        )
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _config():
    return {
        "schema_version": finalizer.CONFIG_SCHEMA_VERSION,
        "library": "LibA",
        "retention_labels_allowed": False,
        "models": [
            {
                "model_id": "additive",
                "display_name": "Additive",
                "source_model": "additive",
                "deployment": {"method": "fixed_seed", "seed": "fixed"},
                "handoff_eligible": True,
            },
            {
                "model_id": "nonlinear_mean",
                "display_name": "Nonlinear mean",
                "source_model": "nonlinear",
                "deployment": {"method": "mean_logit", "seeds": ["11", "12"]},
                "handoff_eligible": False,
            },
        ],
    }


def _write_weak_lock_inputs(tmp_path):
    source_candidate = "mean_logit__additive"
    candidate_metrics = tmp_path / "candidate_metrics.csv"
    candidate_metrics.write_text(
        "candidate,within_peptide_ap,within_peptide_evaluable\n"
        f"{source_candidate},0.71,189\n"
    )
    generic = {
        "schema_version": finalizer.GENERIC_SCORE_LOCK_SCHEMA_VERSION,
        "library": "LibA",
        "candidate": source_candidate,
        "retention_labels_read": False,
        "selection_provenance": {
            "candidate_metrics": {
                "within_peptide_ap": 0.71,
                "within_peptide_evaluable": 189,
            }
        },
    }
    generic_path = tmp_path / "additive.lock.json"
    generic_path.write_text(json.dumps(generic))
    record = {
        "display_name": "Additive",
        "scientifically_eligible": True,
        "threshold_uses_retention": False,
        "weak_oof_within_peptide_average_precision": 0.71,
        "weak_oof_evaluable_peptides_for_within_peptide_ap": 189,
        "weak_metric_provenance": {
            "generic_lock_path": str(generic_path),
            "generic_lock_sha256": finalizer.sha256_file(generic_path),
            "candidate_metrics_path": str(candidate_metrics),
            "candidate_metrics_sha256": finalizer.sha256_file(candidate_metrics),
            "candidate": source_candidate,
        },
    }
    manifest_path = tmp_path / "locks.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": finalizer.HANDOFF_LOCK_SCHEMA_VERSION,
                "library": "LibA",
                "retention_labels_read": False,
                "models": {
                    "additive": record,
                    "nonlinear_mean": {
                        "display_name": "Nonlinear mean",
                        "scientifically_eligible": False,
                    },
                },
            }
        )
    )
    return manifest_path


def test_scores_are_validated_and_deployments_are_built_before_outcomes(monkeypatch):
    predictions = _target_free_predictions()
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_MEMBERSHIP_SHA256",
        _membership(predictions["eval_row_id"].unique().tolist()),
    )
    validated = finalizer.validate_target_free_scores(predictions)
    deployed = finalizer.build_deployment_predictions(validated, _config())
    assert len(deployed) == 216
    assert set(deployed["model"]) == {"additive", "nonlinear_mean"}
    nonlinear = deployed.loc[deployed["model"].eq("nonlinear_mean")].sort_values(
        "eval_row_id"
    )
    seed_values = predictions.loc[predictions["model"].eq("nonlinear")].pivot(
        index="eval_row_id", columns="seed", values="score"
    ).sort_index()
    expected = finalizer._mean_logit(seed_values[["11", "12"]].to_numpy())
    assert np.allclose(nonlinear["score"], expected)


def test_seed_aggregation_refuses_an_unprespecified_subset(monkeypatch):
    predictions = _target_free_predictions()
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_MEMBERSHIP_SHA256",
        _membership(predictions["eval_row_id"].unique().tolist()),
    )
    finalizer.validate_target_free_scores(predictions)
    config = _config()
    config["models"][1]["deployment"]["seeds"] = ["11", "13"]
    with pytest.raises(ValueError, match="do not exactly match source seeds"):
        finalizer.build_deployment_predictions(predictions, config)


def test_weak_ap_is_hash_bound_to_both_generic_lock_and_metric_table(tmp_path):
    lock_path = _write_weak_lock_inputs(tmp_path)
    metrics, audit = finalizer.load_weak_lock_metrics(lock_path, _config())
    assert metrics.to_dict("records") == [
        {
            "model_id": "additive",
            "weak_oof_within_peptide_average_precision": 0.71,
            "weak_oof_evaluable_peptides_for_within_peptide_ap": 189,
        }
    ]
    assert audit["verified_metric_sources"]["additive"]["candidate"] == "mean_logit__additive"

    generic_path = tmp_path / "additive.lock.json"
    generic_path.write_text(generic_path.read_text() + "\n")
    with pytest.raises(ValueError, match="generic lock SHA mismatch"):
        finalizer.load_weak_lock_metrics(lock_path, _config())


def test_handoff_table_has_exact_consumer_schema(tmp_path, monkeypatch):
    predictions = _target_free_predictions()
    monkeypatch.setattr(
        finalizer,
        "EXPECTED_MEMBERSHIP_SHA256",
        _membership(predictions["eval_row_id"].unique().tolist()),
    )
    deployed = finalizer.build_deployment_predictions(
        finalizer.validate_target_free_scores(predictions), _config()
    )
    metric_input = finalizer._as_metric_predictions(
        deployed, {"additive": "Additive", "nonlinear_mean": "Nonlinear mean"}
    )
    aggregate, _, _ = finalizer.evaluate_models(metric_input, _panel())
    thresholds, _, _ = finalizer.build_retrospective_recommendations(metric_input, _panel())
    weak, _ = finalizer.load_weak_lock_metrics(
        _write_weak_lock_inputs(tmp_path), _config()
    )
    output = finalizer.build_handoff_comparison(aggregate, thresholds, weak, _config())
    assert tuple(output.columns) == finalizer.HANDOFF_COMPARISON_COLUMNS
    assert output["model_id"].tolist() == ["additive"]
    assert output.iloc[0]["evaluation_pairs"] == 108
    assert output.iloc[0]["evaluable_peptides_for_within_peptide_ap"] == 5
    assert output.iloc[0]["evaluable_peptides_for_within_peptide_spearman"] == 9
    assert output.iloc[0]["weak_oof_evaluable_peptides_for_within_peptide_ap"] == 189
    assert output.iloc[0]["retrospective_f1"] == pytest.approx(1.0)


def test_target_free_sidecar_loader_rejects_any_outcome_column(tmp_path):
    path = tmp_path / "scores.csv"
    frame = _target_free_predictions().drop(columns="source_role")
    frame["target_retention"] = 99.0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema must be exactly"):
        finalizer.load_blinded_predictions([path])
