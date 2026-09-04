#!/usr/bin/env python
"""Run the locked three-arm LibB MINT/structure late-fusion comparison.

This is a fail-closed entry point around ``train_libb_structural_readouts``.
It permits exactly MINT layer 5 alone, one selected frozen structural vector
alone, and their raw concatenation.  All arms use the same 4,096-wide frozen
injective adapter and the same 272,641-parameter classifier.

The shared trainer consumes only selection-derived training labels and emits
blind scores for all 120 evaluation row IDs.  It has no retention-file input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import train_libb_structural_readouts as shared
from downstream.AffibodyMHC.libb_structural_readout import FEATURE_SOURCE, FeatureView


CONFIG_PATH = (
    REPO_ROOT
    / "downstream/AffibodyMHC/configs/"
    "libb_mint_layer5_structure_late_fusion_v1.json"
)
EXPECTED_MODEL_VIEWS = (
    (
        "mint_layer5_global",
        FEATURE_SOURCE,
        (FeatureView("mint_layer05_global", "identity"),),
    ),
    (
        "structure_selected",
        FEATURE_SOURCE,
        (FeatureView("structural_selected_vector", "identity"),),
    ),
    (
        "mint_layer5_plus_structure",
        FEATURE_SOURCE,
        (
            FeatureView("mint_layer05_global", "identity"),
            FeatureView("structural_selected_vector", "identity"),
        ),
    ),
)
EXPECTED_TRAINABLE_PARAMETERS = 272_641
STRUCTURAL_DIMENSION_PLACEHOLDER = "LOCK_SELECTED_STRUCTURAL_DIMENSION"
FUSION_DIMENSION_PLACEHOLDER = "LOCK_SELECTED_STRUCTURAL_DIMENSION_PLUS_2560"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_common_config(config: dict[str, Any]) -> dict[str, Any]:
    """Enforce every invariant shared by the template and locked config."""

    _require(
        config.get("comparison", {}).get("require_sequence_control") is False,
        "late fusion must not add an unmatched sequence-control arm",
    )
    _require(
        config.get("comparison", {}).get("arms_locked") is True,
        "late-fusion arms are not locked",
    )
    observed = tuple(
        (spec.name, spec.source, spec.views) for spec in shared.model_specs(config)
    )
    _require(observed == EXPECTED_MODEL_VIEWS, "late-fusion model arms changed")
    _require(not shared.ensemble_specs(config), "late-fusion comparison cannot add an ensemble")
    architecture = config["architecture"]
    _require(int(architecture["adapter_dim"]) == 4096, "late-fusion adapter width changed")
    _require(architecture["hidden_dims"] == [64, 32], "late-fusion hidden widths changed")
    _require(float(architecture["dropout"]) == 0.1, "late-fusion dropout changed")
    _require(
        int(config["expected_trainable_parameters"]) == EXPECTED_TRAINABLE_PARAMETERS,
        "late-fusion trainable capacity changed",
    )
    _require(config["final_training"]["backbone_frozen"] is True, "an encoder was unfrozen")
    _require(
        config["final_training"]["retention_labels_allowed"] is False,
        "retention labels were enabled",
    )
    return config


def read_template_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the tracked template, whose structural width is intentionally unset."""

    config = _validate_common_config(shared.read_config(path))
    _require(
        config.get("comparison", {}).get("template_requires_post_selection_lock")
        is True,
        "fusion template is not marked as requiring a post-selection lock",
    )
    expected = config["expected_input_dimensions"]
    _require(expected["mint_layer5_global"] == 2560, "MINT input width changed")
    _require(
        expected["structure_selected"] == STRUCTURAL_DIMENSION_PLACEHOLDER,
        "template structural-width placeholder changed",
    )
    _require(
        expected["mint_layer5_plus_structure"] == FUSION_DIMENSION_PLACEHOLDER,
        "template fusion-width placeholder changed",
    )
    return config


def read_locked_config(path: str | Path) -> dict[str, Any]:
    """Read a private post-selection config and require exact resolved widths."""

    config = _validate_common_config(shared.read_config(path))
    _require(
        config.get("comparison", {}).get("template_requires_post_selection_lock")
        is False,
        "run the post-selection config lock before training",
    )
    expected = config["expected_input_dimensions"]
    mint_dim = int(expected["mint_layer5_global"])
    structural_dim = int(expected["structure_selected"])
    fusion_dim = int(expected["mint_layer5_plus_structure"])
    _require(mint_dim == 2560, "MINT input width changed")
    _require(structural_dim >= 1, "selected structural width is invalid")
    _require(fusion_dim == mint_dim + structural_dim, "fusion width is not additive")
    _require(fusion_dim <= 4096, "selected structural vector makes fusion compressive")
    return config


def _config_argument(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    values, _ = parser.parse_known_args(argv)
    return values.config


def main() -> None:
    # Let the shared parser render its complete help without requiring a config.
    if any(value in {"-h", "--help"} for value in sys.argv[1:]):
        shared.main()
        return
    read_locked_config(_config_argument(sys.argv[1:]))
    shared.main()


if __name__ == "__main__":
    main()
