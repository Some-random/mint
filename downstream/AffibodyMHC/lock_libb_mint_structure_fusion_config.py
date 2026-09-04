#!/usr/bin/env python
"""Lock the LibB late-fusion config after weak-label structural selection.

The tracked fusion template cannot know whether weak-label validation will
select a StaB- or RDE-derived vector, nor its width.  This command consumes the
already materialized, label-free combined archive and a small weak-selection
record.  It writes a private runnable config that pins the selected arm,
source artifacts, exact dimensions, and the complete combined feature
manifest.  It never accepts or reads a retention file.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.export_libb_mint_layer5_fusion_archive import (
    EXPECTED_CANONICAL_ROW_IDS_SHA256,
    EXPECTED_CANONICAL_ROWS,
    OUTPUT_MINT_NAME,
    OUTPUT_STRUCTURE_NAME,
)
from downstream.AffibodyMHC.libb_structural_readout import OpaqueFeatureStore, sha256_file
from downstream.AffibodyMHC.train_libb_mint_structure_late_fusion import (
    CONFIG_PATH,
    read_locked_config,
    read_template_config,
)


PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()
SELECTION_SCHEMA_VERSION = "libb-structural-weak-selection-v1"
SELECTION_METRIC = "pooled_weak_validation_log_loss"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_sha256(value: object) -> bool:
    return bool(SHA256_PATTERN.fullmatch(str(value)))


def _named_hashes(value: object, prefix: str = "") -> dict[str, str]:
    output: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _valid_sha256(child):
                output[path.lower()] = str(child)
            output.update(_named_hashes(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.update(_named_hashes(child, f"{prefix}[{index}]"))
    return output


def _require_structural_provenance(source_manifest: Mapping[str, object]) -> None:
    hashes = _named_hashes(source_manifest)
    for label, terms in {
        "checkpoint": ("checkpoint",),
        "PDB": ("pdb",),
        "residue mapping": ("mapping", "structure_contract"),
        "canonical rows": ("row",),
    }.items():
        _require(
            any(any(term in path for term in terms) for path in hashes),
            f"selected structural manifest lacks a {label} SHA-256",
        )


def load_weak_selection_record(
    path: str | Path,
    *,
    structural_manifest_sha256: str,
    selected_vector_sha256: str,
) -> dict[str, Any]:
    """Validate a weak-only selection decision and its feature provenance."""

    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    _require(isinstance(record, dict), "weak-selection record is not an object")
    _require(record.get("schema_version") == SELECTION_SCHEMA_VERSION, "selection schema changed")
    _require(record.get("selection_metric") == SELECTION_METRIC, "selection metric changed")
    _require(record.get("retention_labels_read") is False, "selection used retention labels")
    _require(bool(record.get("selected_model")), "selection record lacks selected model")
    _require(
        _valid_sha256(record.get("source_weak_cv_artifact_sha256")),
        "selection record lacks the source weak-CV SHA-256",
    )
    _require(
        record.get("source_feature_manifest_sha256") == structural_manifest_sha256,
        "selection record points to a different structural feature archive",
    )
    _require(
        record.get("selected_vector_sha256") == selected_vector_sha256,
        "selection record points to a different materialized vector",
    )
    return record


def _private_new_file(path: str | Path, private_root: Path) -> Path:
    path = Path(path).resolve()
    private_root = Path(private_root).resolve()
    try:
        relative = path.relative_to(private_root)
    except ValueError as error:
        raise ValueError("locked config must stay under private_data") from error
    _require(relative.parts, "refusing to overwrite private_data")
    _require(not path.exists(), "locked config exists; refusing overwrite")
    return path


def lock_config(
    *,
    template_config: str | Path,
    fusion_archive: str | Path,
    weak_selection_record: str | Path,
    output_config: str | Path,
    private_root: Path = PRIVATE_ROOT,
    expected_canonical_rows: int = EXPECTED_CANONICAL_ROWS,
    expected_row_ids_sha256: str = EXPECTED_CANONICAL_ROW_IDS_SHA256,
) -> Path:
    """Write one immutable-by-convention, fully resolved private config."""

    output_config = _private_new_file(output_config, private_root)
    template = read_template_config(template_config)
    required_manifest = copy.deepcopy(
        template["feature_archive_contract"]["required_manifest"]
    )
    required_manifest["row_count"] = int(expected_canonical_rows)
    required_manifest["row_ids_sha256"] = str(expected_row_ids_sha256)
    store = OpaqueFeatureStore.open(
        fusion_archive,
        [OUTPUT_MINT_NAME, OUTPUT_STRUCTURE_NAME],
        verify_all_checksums=True,
        required_manifest=required_manifest,
    )
    _require(
        len(store.row_ids) == int(expected_canonical_rows),
        "fusion archive row count changed",
    )
    _require(
        store.manifest.get("row_ids_sha256") == str(expected_row_ids_sha256),
        "fusion archive canonical rows changed",
    )
    mint = store.arrays[OUTPUT_MINT_NAME]
    structure = store.arrays[OUTPUT_STRUCTURE_NAME]
    _require(mint.ndim == 2 and mint.shape[1] == 2560, "MINT vector shape changed")
    _require(structure.ndim == 2 and structure.shape[1] >= 1, "structural vector shape changed")
    structural_dim = int(structure.shape[1])
    fusion_dim = 2560 + structural_dim
    _require(fusion_dim <= 4096, "selected vector would make the adapter compressive")

    source = store.manifest.get("inputs", {}).get("structural_archive")
    _require(isinstance(source, dict), "fusion archive lacks structural provenance")
    source_manifest = source.get("source_manifest")
    _require(isinstance(source_manifest, dict), "fusion archive lacks source feature manifest")
    source_manifest_sha256 = str(source.get("manifest_sha256", ""))
    _require(_valid_sha256(source_manifest_sha256), "source feature-manifest SHA-256 is missing")
    _require_structural_provenance(source_manifest)
    selected_array_sha256 = str(
        store.manifest["arrays"][OUTPUT_STRUCTURE_NAME]["sha256"]
    )
    selection = load_weak_selection_record(
        weak_selection_record,
        structural_manifest_sha256=source_manifest_sha256,
        selected_vector_sha256=selected_array_sha256,
    )

    # ``read_template_config`` returns merged protocol sections.  Removing the
    # relative include makes the private locked config self-contained.
    locked = dict(template)
    locked.pop("protocol_config", None)
    locked["comparison"] = dict(locked["comparison"])
    locked["comparison"].update(
        {
            "template_requires_post_selection_lock": False,
            "selected_structural_model": selection["selected_model"],
            "selection_metric": selection["selection_metric"],
            "weak_selection_record_sha256": sha256_file(
                Path(weak_selection_record).resolve()
            ),
            "source_weak_cv_artifact_sha256": selection[
                "source_weak_cv_artifact_sha256"
            ],
            "source_feature_manifest_sha256": source_manifest_sha256,
            "selected_vector_sha256": selected_array_sha256,
        }
    )
    locked["expected_input_dimensions"] = {
        "mint_layer5_global": 2560,
        "structure_selected": structural_dim,
        "mint_layer5_plus_structure": fusion_dim,
    }
    # Pin the complete combined archive manifest, including the nested source
    # checkpoint/PDB/mapping/row contracts copied by the label-free exporter.
    locked["feature_archive_contract"] = {
        "required_manifest": store.manifest,
        "manifest_sha256": store.manifest_sha256,
    }
    output_config.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with output_config.open("x", encoding="utf-8") as handle:
        json.dump(locked, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(output_config, 0o600)
    read_locked_config(output_config)
    return output_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--fusion-archive", type=Path, required=True)
    parser.add_argument("--weak-selection-record", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        lock_config(
            template_config=args.template_config,
            fusion_archive=args.fusion_archive,
            weak_selection_record=args.weak_selection_record,
            output_config=args.output_config,
        )
    )


if __name__ == "__main__":
    main()
