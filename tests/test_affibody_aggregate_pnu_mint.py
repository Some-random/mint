from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_pnu_mint_grid as aggregate


def _candidate_grid():
    rows = []
    for library in aggregate.LIBRARIES:
        for pi in aggregate.PRIORS:
            for eta in aggregate.ETAS:
                rows.append(
                    {
                        "library": library,
                        "pi": pi,
                        "eta": eta,
                        "C": 0.1,
                        "epoch": 4,
                        "weak_validation_auroc": 0.9 - eta / 100.0,
                        "weak_validation_ap": 0.8,
                        "weak_validation_log_loss": 0.4,
                        "retention_auroc": 0.0 if eta == 0 else 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_weak_selection_keeps_every_pi_and_ignores_retention():
    candidates, selected = aggregate.select_eta_per_fixed_pi(_candidate_grid())
    assert len(candidates) == 40
    assert len(selected) == 8
    assert set(selected["pi"]) == set(aggregate.PRIORS)
    assert selected["eta"].eq(0.0).all()
    assert candidates["selected_eta_by_aggregate"].sum() == 8


@pytest.mark.parametrize(
    "winner_change",
    [
        {"weak_validation_auroc": 0.91},
        {"weak_validation_auroc": 0.90, "weak_validation_ap": 0.86},
        {
            "weak_validation_auroc": 0.90,
            "weak_validation_ap": 0.85,
            "weak_validation_log_loss": 0.29,
        },
    ],
)
def test_weak_selection_metric_priority_precedes_eta(winner_change):
    frame = _candidate_grid()
    target = frame["library"].eq("LibA") & np.isclose(frame["pi"], 0.02)
    frame.loc[target, ["weak_validation_auroc", "weak_validation_ap", "weak_validation_log_loss"]] = [0.90, 0.85, 0.30]
    challenger = target & np.isclose(frame["eta"], 0.25)
    for name, value in winner_change.items():
        frame.loc[challenger, name] = value
    _, selected = aggregate.select_eta_per_fixed_pi(frame)
    row = selected.loc[selected["library"].eq("LibA") & np.isclose(selected["pi"], 0.02)].iloc[0]
    assert row["eta"] == pytest.approx(0.25)


def test_full_tie_uses_smaller_eta_then_c_then_epoch():
    common = {
        "weak_validation_auroc": 0.9,
        "weak_validation_ap": 0.8,
        "weak_validation_log_loss": 0.3,
    }
    rows = [
        dict(common, eta=0.25, C=1.0, epoch=20),
        dict(common, eta=0.00, C=1.0, epoch=20),
        dict(common, eta=0.00, C=0.1, epoch=20),
        dict(common, eta=0.00, C=0.1, epoch=10),
    ]
    assert min(rows, key=aggregate.weak_selection_key) == rows[-1]


def test_historical_control_keeps_original_logloss_selector():
    rows = [
        {"C": 0.01, "log_loss": 0.20, "auprc": 0.80, "auroc": 0.99},
        {"C": 0.10, "log_loss": 0.10, "auprc": 0.75, "auroc": 0.75},
    ]
    assert min(rows, key=aggregate._historical_control_key) == rows[1]


def _weak_history(epochs=(1, 2)):
    rows = []
    for c_value in aggregate.C_GRID:
        for epoch in epochs:
            for record_type, fold in [("aggregate", -1), ("fold", 0), ("fold", 1), ("fold", 2)]:
                rows.append(
                    {
                        "library": "LibA",
                        "record_type": record_type,
                        "pi": 0.02,
                        "eta": 0.25,
                        "C": c_value,
                        "epoch": epoch,
                        "fold": fold,
                        "training_seed": 17,
                        "log_loss": 0.5,
                        "auroc": 0.8,
                        "ap": 0.8,
                    }
                )
    frame = pd.DataFrame(rows)
    aggregate_rows = frame["record_type"].eq("aggregate")
    winner = aggregate_rows & np.isclose(frame["C"], 0.01) & frame["epoch"].eq(2)
    frame.loc[winner, ["auroc", "ap", "log_loss"]] = [0.9, 0.85, 0.3]
    return frame


def test_history_grid_and_fixed_arm_selection_are_reproduced():
    selected = aggregate.validate_weak_history(
        _weak_history(), "LibA", 0.02, 0.25, (1, 2)
    )
    assert selected["C"] == pytest.approx(0.01)
    assert selected["epoch"] == 2
    assert selected["weak_validation_auroc"] == pytest.approx(0.9)


def test_history_rejects_partial_grid_and_wrong_training_seed():
    frame = _weak_history()
    with pytest.raises(ValueError, match="partial"):
        aggregate.validate_weak_history(
            frame.iloc[:-1], "LibA", 0.02, 0.25, (1, 2)
        )
    frame.loc[0, "training_seed"] = 99
    with pytest.raises(ValueError, match="training seed"):
        aggregate.validate_weak_history(frame, "LibA", 0.02, 0.25, (1, 2))


def _minimal_arm(library, pi, eta, epoch=3, suffix="base"):
    history = pd.DataFrame(
        [
            {
                "record_type": "aggregate",
                "pi": pi,
                "eta": eta,
                "C": 0.1,
                "epoch": value,
                "fold": -1,
                "log_loss": 0.5,
                "auroc": 0.8,
                "ap": 0.8,
            }
            for value in range(1, 41 if suffix == "base" else 81)
        ]
    )
    return {
        "library": library,
        "pi": pi,
        "eta": eta,
        "epoch": epoch,
        "run_name": "{}-{}-{}-{}".format(suffix, library, pi, eta),
        "trainer_sha256": "a" * 64,
        "source_hash_contract": {
            "pn_cache_sha256": ("b" * 64,),
            "pnu_cache_sha256": "c" * 64,
            "pnu_rows_sha256": "d" * 64,
            "pnu_manifest_sha256": "e" * 64,
        },
        "configuration_signature": {"same": True},
        "counts": {"library": library, "positive": 1},
        "history": history,
    }


def test_boundary_extensions_replace_only_exact_triggered_arms(monkeypatch):
    boundary = {
        ("LibA", 0.02, 0.0),
        ("LibA", 0.05, 0.0),
        ("LibA", 0.15, 0.0),
        ("LibB", 0.02, 0.0),
    }
    base = {}
    for library in aggregate.LIBRARIES:
        for pi in aggregate.PRIORS:
            for eta in aggregate.ETAS:
                key = (library, pi, eta)
                base[key] = _minimal_arm(
                    library, pi, eta, epoch=40 if key in boundary else 3
                )
    extensions = {
        key: _minimal_arm(key[0], key[1], key[2], epoch=70, suffix="extension")
        for key in boundary
    }
    paths = [Path("run-{}".format(index)) for index in range(4)]
    path_to_arm = dict(zip(paths, extensions.values()))
    monkeypatch.setattr(aggregate, "_visible_run_dirs", lambda root: paths)
    monkeypatch.setattr(
        aggregate,
        "load_fixed_arm_weak",
        lambda path, expected_epochs, expect_control=False: path_to_arm[path],
    )
    combined, audit, observed = aggregate.integrate_boundary_extensions(
        base, Path("unused")
    )
    assert len(combined) == 40
    assert set(observed) == boundary
    assert audit["extension_used"].sum() == 4
    for key in boundary:
        assert combined[key]["run_name"].startswith("extension-")


def test_boundary_extension_rejects_mixed_cache(monkeypatch):
    key = ("LibA", 0.02, 0.0)
    base = {key: _minimal_arm(*key, epoch=40)}
    extension = _minimal_arm(*key, epoch=60, suffix="extension")
    extension["source_hash_contract"] = dict(extension["source_hash_contract"])
    extension["source_hash_contract"]["pnu_cache_sha256"] = "f" * 64
    monkeypatch.setattr(aggregate, "_visible_run_dirs", lambda root: [Path("extension")])
    monkeypatch.setattr(
        aggregate,
        "load_fixed_arm_weak",
        lambda path, expected_epochs, expect_control=False: extension,
    )
    with pytest.raises(ValueError, match="cache hashes"):
        aggregate.integrate_boundary_extensions(base, Path("unused"))


def test_uniform_contract_rejects_mixed_trainer_and_cache_hashes():
    first = _minimal_arm("LibA", 0.02, 0.0)
    second = _minimal_arm("LibB", 0.02, 0.0)
    with pytest.raises(ValueError, match="trainer"):
        altered = dict(second, trainer_sha256="9" * 64)
        aggregate._verify_uniform_contract([first, altered])
    with pytest.raises(ValueError, match="cache"):
        altered = dict(second)
        altered["source_hash_contract"] = dict(second["source_hash_contract"])
        altered["source_hash_contract"]["pnu_rows_sha256"] = "9" * 64
        aggregate._verify_uniform_contract([first, altered])


def _perfect_predictions():
    return pd.DataFrame(
        {
            "chain1_sha256": ["pep-a", "pep-a", "pep-b", "pep-b"],
            "target_retention": [100.0, 0.0, 80.0, 70.0],
            "target_binder": [1, 0, 1, 0],
            "score": [0.9, 0.1, 0.8, 0.2],
        }
    )


def test_retention_metrics_and_retrospective_threshold_are_recomputed():
    metrics = aggregate.retention_metrics(_perfect_predictions())
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["global_spearman"] == pytest.approx(1.0)
    assert metrics["average_within_peptide_spearman"] == pytest.approx(1.0)
    assert metrics["within_peptide_groups"] == 2
    assert metrics["retrospective_f1_threshold"] == pytest.approx(0.8)
    assert metrics["retrospective_f1_precision"] == pytest.approx(1.0)
    assert metrics["retrospective_f1_recall"] == pytest.approx(1.0)


def test_retention_metric_tampering_and_binder_threshold_mismatch_fail():
    metrics = aggregate.retention_metrics(_perfect_predictions())
    stored = pd.DataFrame([metrics])
    stored.loc[0, "ap"] = 0.5
    with pytest.raises(ValueError, match="does not reproduce"):
        aggregate._check_recorded_metrics(stored, metrics, "test")
    predictions = _perfect_predictions()
    predictions.loc[0, "target_binder"] = 0
    with pytest.raises(ValueError, match="75%"):
        aggregate.retention_metrics(predictions)
