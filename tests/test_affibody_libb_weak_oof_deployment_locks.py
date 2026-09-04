import json

import numpy as np
import pytest

from downstream.AffibodyMHC.assemble_libb_ensemble_wetlab_shortlist import load_lock
from downstream.AffibodyMHC import build_libb_weak_oof_deployment_locks as locks


SYNTHETIC_SOURCE_RECORDS = {"oof": {"sha256": "a" * 64}}


def _assembler_score(payload, raw_logits):
    linear = float(payload["intercept"])
    for component in payload["components"]:
        raw = np.asarray(raw_logits[component["name"]], dtype=float)
        linear = linear + float(component["weight"]) * (
            (raw - float(component["center"])) / float(component["scale"])
        )
    return 1.0 / (1.0 + np.exp(-linear))


def test_f1_cutoff_tie_rule_is_explicit_and_deterministic():
    result = locks.select_f1_threshold([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1])
    assert result["threshold"] == 4.0
    assert result["recommended"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2.0 / 3.0)


def test_equal_logit_lock_matches_the_declared_formula(tmp_path):
    definition = next(
        value
        for value in locks.LOCK_DEFINITIONS
        if value["filename"] == "mint_stab_equal_logit_exploratory.lock.json"
    )
    threshold = {
        "threshold": 0.4,
        "recommended": 2,
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    payload = locks._lock_payload(definition, threshold, SYNTHETIC_SOURCE_RECORDS)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    observed_lock = load_lock(path)
    mint = np.asarray([-2.0, 1.0])
    stab = np.asarray([0.0, 3.0])
    observed = _assembler_score(
        observed_lock,
        {"mint_layer5": mint, "stab_designed_ordered": stab},
    )
    expected = 1.0 / (1.0 + np.exp(-((mint + stab) / 2.0)))
    assert np.allclose(observed, expected)
    assert observed_lock["components"][1]["score_column"] == "stab_logit"


@pytest.mark.parametrize(
    ("filename", "component_name", "score_column"),
    [
        ("stab_standalone.lock.json", "stab_designed_ordered", "stab_logit"),
        (
            "rde_standalone.lock.json",
            "rde_network_designed_3fold",
            "rde_logit",
        ),
    ],
)
def test_standalone_structure_locks_are_assembler_compatible(
    tmp_path, filename, component_name, score_column
):
    definition = next(
        value for value in locks.LOCK_DEFINITIONS if value["filename"] == filename
    )
    threshold = {
        "threshold": 0.3,
        "recommended": 2,
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 2,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    payload = locks._lock_payload(definition, threshold, SYNTHETIC_SOURCE_RECORDS)
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")

    observed_lock = load_lock(path)
    raw_logit = np.asarray([-1.25, 2.5])
    observed = _assembler_score(observed_lock, {component_name: raw_logit})
    expected = 1.0 / (1.0 + np.exp(-raw_logit))

    assert np.allclose(observed, expected)
    assert observed_lock["components"][0]["score_column"] == score_column


def test_stacker_lock_preserves_order_and_locked_standardization():
    stacker = {
        "members_in_coefficient_order": list(locks.MEMBERS),
        "coefficient": [0.2, 2.5, 1.0, 0.4],
        "mean": [2.0, 2.1, 2.2, 2.3],
        "scale": [4.0, 5.0, 3.0, 2.0],
        "intercept": 2.7,
    }
    definition = locks._stacker_lock_definition(stacker)
    assert [component["name"] for component in definition["components"]] == list(locks.MEMBERS)
    assert [component["score_column"] for component in definition["components"]] == [
        "additive_7site_logit",
        "mint_layer5_logit",
        "stab_logit",
        "rde_logit",
    ]
    raw = {member: np.asarray([index - 1.5]) for index, member in enumerate(locks.MEMBERS)}
    payload = {
        "components": definition["components"],
        "intercept": definition["intercept"],
    }
    observed = _assembler_score(payload, raw)
    expected_linear = stacker["intercept"]
    for index, member in enumerate(locks.MEMBERS):
        expected_linear += stacker["coefficient"][index] * (
            (raw[member] - stacker["mean"][index]) / stacker["scale"][index]
        )
    expected = 1.0 / (1.0 + np.exp(-expected_linear))
    assert np.allclose(observed, expected)
