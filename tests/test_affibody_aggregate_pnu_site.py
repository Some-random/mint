import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC.aggregate_pnu_site_grid import (
    audit_epoch_choices,
    require_uniform_trainer_hash,
    select_prespecified_configs,
    validate_training_grid_configuration,
)


def _config(library, risk, pi, eta, c_value, epoch, auroc, ap, loss, suffix):
    return {
        "library": library,
        "risk": risk,
        "class_prior": pi,
        "eta": eta,
        "C": c_value,
        "selected_epoch": epoch,
        "validation_auroc": auroc,
        "validation_average_precision": ap,
        "validation_log_loss": loss,
        "config_id": "config-{}".format(suffix),
        "run_name": "run-{}".format(suffix),
    }


def test_selection_keeps_every_pi_and_never_uses_retention_columns():
    rows = []
    for pi in (0.02, 0.05, 0.10, 0.15):
        # The eta=0.5 row has deliberately terrible retention AUROC. It must
        # still win because only weak-validation AUROC drives selection.
        rows.append(_config("LibB", "nnpnu", pi, 0.0, 1.0, 4, 0.80, 0.81, 0.4, "{}-zero".format(pi)))
        rows[-1]["retention_auroc"] = 0.99
        rows.append(_config("LibB", "nnpnu", pi, 0.5, 1.0, 4, 0.90, 0.81, 0.8, "{}-half".format(pi)))
        rows[-1]["retention_auroc"] = 0.01
    _, selected = select_prespecified_configs(pd.DataFrame(rows))
    assert len(selected) == 4
    assert set(selected["class_prior"]) == {0.02, 0.05, 0.10, 0.15}
    assert set(selected["eta"]) == {0.5}


def test_selection_ties_by_ap_loss_then_smaller_eta_c_epoch():
    rows = [
        _config("LibA", "nnpnu", 0.1, 0.75, 10.0, 9, 0.9, 0.80, 0.20, "a"),
        _config("LibA", "nnpnu", 0.1, 0.50, 10.0, 9, 0.9, 0.85, 0.40, "b"),
        _config("LibA", "nnpnu", 0.1, 0.25, 1.0, 8, 0.9, 0.85, 0.30, "c"),
        _config("LibA", "nnpnu", 0.1, 0.00, 0.1, 7, 0.9, 0.85, 0.30, "d"),
    ]
    _, selected = select_prespecified_configs(pd.DataFrame(rows))
    assert selected.iloc[0]["config_id"] == "config-d"


def test_epoch_audit_reproduces_choice_and_flags_both_boundaries():
    configs = pd.DataFrame(
        [
            _config("LibA", "nnpu", 0.1, 1.0, 1.0, 1, 0.90, 0.80, 0.5, "first"),
            _config("LibB", "nnpu", 0.1, 1.0, 1.0, 3, 0.92, 0.81, 0.4, "last"),
        ]
    )
    rows = []
    for run_name, config_id, chosen in (
        ("run-first", "config-first", 1),
        ("run-last", "config-last", 3),
    ):
        for epoch in (1, 2, 3):
            if epoch == chosen:
                auroc, ap, loss = (0.90, 0.80, 0.5) if chosen == 1 else (0.92, 0.81, 0.4)
            else:
                auroc, ap, loss = 0.70, 0.70, 0.6
            rows.append(
                {
                    "run_name": run_name,
                    "config_id": config_id,
                    "record_type": "aggregate",
                    "epoch": epoch,
                    "auroc": auroc,
                    "average_precision": ap,
                    "log_loss": loss,
                    "epoch_selected_within_config": int(epoch == chosen),
                }
            )
    audit = audit_epoch_choices(configs, pd.DataFrame(rows), {"run-first": 3, "run-last": 3})
    assert audit.set_index("run_name")["epoch_location"].to_dict() == {
        "run-first": "first_epoch",
        "run-last": "last_epoch",
    }
    assert audit["epoch_selection_reproduced"].eq(1).all()


def test_epoch_audit_fails_if_recorded_epoch_was_not_weak_best():
    configs = pd.DataFrame([
        _config("LibA", "nnpu", 0.1, 1.0, 1.0, 2, 0.70, 0.70, 0.6, "bad")
    ])
    history = pd.DataFrame(
        [
            {
                "run_name": "run-bad",
                "config_id": "config-bad",
                "record_type": "aggregate",
                "epoch": epoch,
                "auroc": 0.95 if epoch == 1 else 0.70,
                "average_precision": 0.9 if epoch == 1 else 0.7,
                "log_loss": 0.3 if epoch == 1 else 0.6,
                "epoch_selected_within_config": int(epoch == 2),
            }
            for epoch in (1, 2)
        ]
    )
    with pytest.raises(ValueError, match="recorded epoch does not reproduce"):
        audit_epoch_choices(configs, history, {"run-bad": 2})


def test_mixed_trainer_hashes_are_rejected():
    manifests = {
        "arm-a": {"code": {"/repo/train_pnu_site.py": "a" * 64}},
        "arm-b": {"code": {"/repo/train_pnu_site.py": "b" * 64}},
    }
    with pytest.raises(ValueError, match="mixed train_pnu_site.py hashes"):
        require_uniform_trainer_hash(manifests)


def test_fold_and_c_grid_contract_is_enforced():
    configuration = {"folds": 5, "c_grid": [100, 0.001, 10, 0.01, 1, 0.1]}
    validate_training_grid_configuration(
        configuration,
        expected_folds=5,
        expected_c_grid=[0.001, 0.01, 0.1, 1, 10, 100],
    )
    with pytest.raises(ValueError, match="fold count differs"):
        validate_training_grid_configuration(configuration, expected_folds=3)
    with pytest.raises(ValueError, match="C grid differs"):
        validate_training_grid_configuration(
            configuration, expected_c_grid=[0.01, 0.1, 1, 10]
        )


def test_max_epoch_contract_is_enforced():
    configuration = {"folds": 5, "c_grid": [1], "max_epochs": 100}
    validate_training_grid_configuration(configuration, expected_max_epochs=100)
    with pytest.raises(ValueError, match="maximum epoch count differs"):
        validate_training_grid_configuration(configuration, expected_max_epochs=40)
