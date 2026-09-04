import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import audit_libb_locked_ensemble_retention as audit


def test_structure_seeds_are_aggregated_in_logit_space():
    rows = []
    values = {
        "1": [0.1, 0.9],
        "2": [0.2, 0.8],
        "3": [0.5, 0.5],
        "4": [0.8, 0.2],
        "5": [0.9, 0.1],
    }
    for seed, probabilities in values.items():
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "eval_row_id": f"r{index}",
                    "model": "model",
                    "seed": seed,
                    "score": probability,
                }
            )
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=11)

    aggregated, wide = audit.aggregate_structure_seed_logits(frame, "structure")
    aggregated = aggregated.sort_values("eval_row_id")
    expected = audit.mean_logit_score([values[seed] for seed in sorted(values)])

    assert np.allclose(aggregated["structure"], expected)
    assert len([column for column in wide if column.startswith("structure__seed_")]) == 5


def test_locked_stacker_uses_parameter_order_scaling_and_intercept():
    scores = pd.DataFrame(
        {
            "additive_7site": [0.2, 0.8],
            "mint_layer5": [0.3, 0.7],
            "stab_designed_ordered": [0.4, 0.6],
            "rde_network_designed_3fold": [0.45, 0.55],
        }
    )
    parameters = {
        "members_in_coefficient_order": list(audit.MEMBERS),
        "mean": [0.1, 0.2, 0.3, 0.4],
        "scale": [1.0, 2.0, 3.0, 4.0],
        "coefficient": [0.5, 0.25, 0.125, 0.0625],
        "intercept": -0.3,
    }
    raw_logits = np.column_stack(
        [audit.clipped_logit(scores[member]) for member in audit.MEMBERS]
    )
    expected_logit = -0.3 + (
        (raw_logits - np.asarray(parameters["mean"]))
        / np.asarray(parameters["scale"])
    ) @ np.asarray(parameters["coefficient"])

    assert np.allclose(audit.apply_locked_stacker(scores, parameters), audit.sigmoid(expected_logit))


def test_locked_stacker_rejects_reordered_members():
    scores = pd.DataFrame({member: [0.5] for member in audit.MEMBERS})
    parameters = {
        "members_in_coefficient_order": list(reversed(audit.MEMBERS)),
        "mean": [0.0] * 4,
        "scale": [1.0] * 4,
        "coefficient": [1.0] * 4,
        "intercept": 0.0,
    }
    with pytest.raises(ValueError, match="member order"):
        audit.apply_locked_stacker(scores, parameters)


def test_f1_threshold_tie_prefers_precision_then_fewer_candidates():
    result = audit.select_f1_threshold([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 1])
    assert result["threshold"] == 4.0
    assert result["recommended"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2.0 / 3.0)


def test_weak_lock_is_hashed_before_audit_output(tmp_path):
    weak = tmp_path / "weak"
    output = tmp_path / "private_data" / "output"
    weak.mkdir()
    selection = {
        "schema_version": audit.WEAK_SCHEMA_VERSION,
        "retention_labels_read": False,
        "best_candidate": {"candidate": "mean_logit__mint_layer5"},
        "best_ensemble_candidate": {
            "candidate": "mean_logit__mint_layer5__stab_designed_ordered"
        },
    }
    parameters = {
        "schema_version": audit.WEAK_SCHEMA_VERSION,
        "retention_labels_read": False,
        "members_in_coefficient_order": list(audit.MEMBERS),
    }
    manifest = {
        "schema_version": audit.WEAK_SCHEMA_VERSION,
        "retention_labels_read": False,
    }
    for filename, payload in (
        ("locked_weak_selection.json", selection),
        ("stacker_locked_parameters.json", parameters),
        ("manifest.json", manifest),
    ):
        (weak / filename).write_text(json.dumps(payload), encoding="utf-8")
    (weak / "candidate_metrics.csv").write_text("candidate,metric\na,1\n", encoding="utf-8")
    expected_hash = hashlib.sha256((weak / "locked_weak_selection.json").read_bytes()).hexdigest()

    result = audit.validate_and_preserve_weak_lock(weak, output)
    seal = json.loads((output / "pre_retention_lock.json").read_text(encoding="utf-8"))

    assert seal["retention_sources_opened"] is False
    assert seal["weak_artifacts"]["locked_weak_selection.json"]["sha256"] == expected_hash
    assert result["selection"] == selection
