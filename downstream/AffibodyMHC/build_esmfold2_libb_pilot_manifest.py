#!/usr/bin/env python
"""Build the label-free six-pair LibB folding-feature pilot manifest.

The model-facing JSON deliberately contains no direct-retention values.  A
separate audit JSON records why the measured diagnostic controls were chosen.
Both files are derived from the provider's updated sequence deck and the
canonical reconstructed retention sequence table; neither canonical input is
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (  # noqa: E402
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    EXPECTED_AFFIBODY_X_POSITIONS,
    OMITTED_LINKER,
    fill_template,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (  # noqa: E402
    sha256_file,
    validate_private_output_path,
)


PILOT_ROWS = (
    {
        "pilot_id": "libb_ref_mw_nnyyf",
        "peptide_design_code": "MW",
        "affibody_design_code": "NNYYF",
        "provenance": "measured_reference",
        "purpose": "reference LibB crystal pair and feature-extraction anchor",
    },
    {
        "pilot_id": "libb_single_peptide_p4_m_to_a",
        "peptide_design_code": "AW",
        "affibody_design_code": "NNYYF",
        "provenance": "synthetic_feature_qa_only",
        "purpose": "one-residue peptide perturbation at crystal-contact peptide position 4",
    },
    {
        "pilot_id": "libb_single_affibody_p14_y_to_a",
        "peptide_design_code": "MW",
        "affibody_design_code": "NNYAF",
        "provenance": "synthetic_feature_qa_only",
        "purpose": "one-residue Affibody perturbation at crystal-contact Affibody position 14",
    },
    {
        "pilot_id": "libb_measured_mw_antkv_low",
        "peptide_design_code": "MW",
        "affibody_design_code": "ANTKV",
        "provenance": "measured_diagnostic_only",
        "purpose": "same-peptide measured control from the low end of the MW retention range",
    },
    {
        "pilot_id": "libb_measured_mw_nmdkv_middle",
        "peptide_design_code": "MW",
        "affibody_design_code": "NMDKV",
        "provenance": "measured_diagnostic_only",
        "purpose": "same-peptide measured control from the middle of the MW retention range",
    },
    {
        "pilot_id": "libb_measured_mw_liftk_high",
        "peptide_design_code": "MW",
        "affibody_design_code": "LIFTK",
        "provenance": "measured_diagnostic_only",
        "purpose": "same-peptide measured control from the high end of the MW retention range",
    },
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _sequence_diff(reference: str, variant: str) -> list[dict[str, object]]:
    _require(len(reference) == len(variant), "cannot compare sequences with unequal lengths")
    return [
        {
            "sequence_position_1_based": index,
            "reference_amino_acid": reference_aa,
            "variant_amino_acid": variant_aa,
        }
        for index, (reference_aa, variant_aa) in enumerate(
            zip(reference, variant), start=1
        )
        if reference_aa != variant_aa
    ]


def _build_sequences(template: str, peptide_code: str, affibody_code: str) -> tuple[str, str]:
    full_construct = fill_template(template, peptide_code + affibody_code)
    chain1 = full_construct[:CHAIN1_LENGTH]
    omitted = full_construct[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)]
    chain2 = full_construct[-CHAIN2_LENGTH:]
    _require(omitted == OMITTED_LINKER, "provider omitted-linker sequence changed")
    _require(len(chain1) == CHAIN1_LENGTH, "chain 1 length mismatch")
    _require(len(chain2) == CHAIN2_LENGTH, "chain 2 length mismatch")
    return chain1, chain2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-zip",
        type=Path,
        default=REPO_ROOT / "Affibody coevolution dataset_updated.zip",
    )
    parser.add_argument(
        "--canonical-sequence-csv",
        type=Path,
        default=REPO_ROOT / "private_data/derived/retention_sequences_v2.csv",
    )
    parser.add_argument(
        "--canonical-sequence-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/retention_sequences_v2.manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "private_data/derived/esmfold2_libb_pilot_v1",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    started = time.time()
    script_path = Path(__file__).resolve()
    output_dir = validate_private_output_path(args.output_dir, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "pilot_inputs.json"
    audit_path = output_dir / "retention_selection_audit.json"
    checksum_path = output_dir / "checksums.sha256"
    for path in (input_path, audit_path, checksum_path):
        _require(not path.exists(), "output already exists; refusing overwrite: {}".format(path))

    for path in (
        args.sequence_zip,
        args.canonical_sequence_csv,
        args.canonical_sequence_manifest,
    ):
        _require(path.is_file(), "required source does not exist: {}".format(path))

    canonical_manifest = json.loads(args.canonical_sequence_manifest.read_text(encoding="utf-8"))
    _require(
        sha256_file(args.canonical_sequence_csv)
        == canonical_manifest["output"]["sha256"],
        "canonical sequence CSV no longer matches its manifest",
    )
    _require(
        sha256_file(args.sequence_zip)
        == canonical_manifest["sources"]["sequence_zip"]["sha256"],
        "updated provider sequence ZIP no longer matches the canonical sequence manifest",
    )

    templates, deck_member, deck_sha256 = load_provider_templates(args.sequence_zip)
    template = templates["LibB"]
    measured = pd.read_csv(args.canonical_sequence_csv)
    measured = measured[(measured["library"] == "LibB") & measured["target_retention"].notna()].copy()
    _require(len(measured) == 119, "expected 119 canonical measured LibB pairs")
    _require(measured["pair_uid"].nunique() == len(measured), "duplicate measured LibB pair UID")

    constructed: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    reference_chains: tuple[str, str] | None = None

    for spec in PILOT_ROWS:
        peptide_code = spec["peptide_design_code"]
        affibody_code = spec["affibody_design_code"]
        chain1, chain2 = _build_sequences(template, peptide_code, affibody_code)
        match = measured[
            (measured["peptide_design_code"] == peptide_code)
            & (measured["affibody_design_code"] == affibody_code)
        ]

        is_measured = spec["provenance"].startswith("measured")
        _require(len(match) == (1 if is_measured else 0), "unexpected measured-row membership")
        if is_measured:
            row = match.iloc[0]
            _require(
                chain1 == row["chain1_smart_hla_linker_peptide_sequence"],
                "deck-derived chain 1 disagrees with canonical sequence table",
            )
            _require(
                chain2 == row["chain2_affibody_sequence"],
                "deck-derived chain 2 disagrees with canonical sequence table",
            )
            pair_uid = str(row["pair_uid"])
            retention = float(row["target_retention"])
            binder = int(row["target_binder"])
        else:
            pair_uid = None
            retention = None
            binder = None

        if reference_chains is None:
            reference_chains = (chain1, chain2)
        chain1_diff = _sequence_diff(reference_chains[0], chain1)
        chain2_diff = _sequence_diff(reference_chains[1], chain2)

        constructed.append(
            {
                "pilot_id": spec["pilot_id"],
                "library": "LibB",
                "peptide_design_code": peptide_code,
                "affibody_design_code": affibody_code,
                "provenance": spec["provenance"],
                "purpose": spec["purpose"],
                "canonical_measured_pair_uid": pair_uid,
                "chains": [
                    {
                        "chain_id": "A",
                        "role": "smart-HLA-linker-peptide",
                        "sequence": chain1,
                        "length": len(chain1),
                        "sha256": _sha256_text(chain1),
                    },
                    {
                        "chain_id": "B",
                        "role": "Affibody",
                        "sequence": chain2,
                        "length": len(chain2),
                        "sha256": _sha256_text(chain2),
                    },
                ],
                "sequence_pair_sha256": _sha256_text(chain1 + "|" + chain2),
                "differences_from_reference": {
                    "chain_A": chain1_diff,
                    "chain_B": chain2_diff,
                },
            }
        )
        audit_rows.append(
            {
                "pilot_id": spec["pilot_id"],
                "peptide_design_code": peptide_code,
                "affibody_design_code": affibody_code,
                "canonical_measured_pair_uid": pair_uid,
                "direct_retention_percent_audit_only": retention,
                "binder_ge_75_percent_audit_only": binder,
            }
        )

    _require(reference_chains is not None, "reference sequence was not constructed")
    by_id = {row["pilot_id"]: row for row in constructed}
    peptide_single = by_id["libb_single_peptide_p4_m_to_a"]["differences_from_reference"]
    _require(
        peptide_single["chain_A"]
        == [
            {
                "sequence_position_1_based": 265,
                "reference_amino_acid": "M",
                "variant_amino_acid": "A",
            }
        ]
        and peptide_single["chain_B"] == [],
        "synthetic peptide single-mutant mapping changed",
    )
    affibody_single = by_id["libb_single_affibody_p14_y_to_a"]["differences_from_reference"]
    _require(
        affibody_single["chain_A"] == []
        and affibody_single["chain_B"]
        == [
            {
                "sequence_position_1_based": 14,
                "reference_amino_acid": "Y",
                "variant_amino_acid": "A",
            }
        ],
        "synthetic Affibody single-mutant mapping changed",
    )

    source_hashes = {
        "updated_provider_sequence_zip": {
            "path": str(args.sequence_zip.resolve()),
            "sha256": sha256_file(args.sequence_zip),
            "embedded_deck_member": deck_member,
            "embedded_deck_sha256": deck_sha256,
        },
        "canonical_reconstructed_sequence_csv": {
            "path": str(args.canonical_sequence_csv.resolve()),
            "sha256": sha256_file(args.canonical_sequence_csv),
        },
        "canonical_reconstructed_sequence_manifest": {
            "path": str(args.canonical_sequence_manifest.resolve()),
            "sha256": sha256_file(args.canonical_sequence_manifest),
        },
        "builder": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
    }
    inputs = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "six-pair LibB folding-model intermediate-feature extraction QA pilot",
        "label_policy": {
            "model_facing_manifest_contains_retention": False,
            "synthetic_rows_have_experimental_labels": False,
            "allowed_use": "feature mapping, mutation-sensitivity, runtime, and storage QA only",
            "forbidden_use": "training, tuning, architecture selection, or model comparison",
        },
        "provider_input_contract": {
            "chain_order": ["A:smart-HLA-linker-peptide", "B:Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "tcr_present": False,
            "separate_b2m_supplied": False,
            "interchain_construct_linker_is_model_input": False,
            "omitted_construct_linker": OMITTED_LINKER,
        },
        "residue_mapping": {
            "peptide_within_chain_A_positions_1_based": {
                str(position): 261 + position for position in range(1, 10)
            },
            "peptide_design_code": [
                {"code_index_1_based": 1, "peptide_position_1_based": 4, "chain_A_position_1_based": 265},
                {"code_index_1_based": 2, "peptide_position_1_based": 5, "chain_A_position_1_based": 266},
            ],
            "affibody_design_code": [
                {
                    "code_index_1_based": index,
                    "chain_B_affibody_position_1_based": position,
                }
                for index, position in enumerate(
                    EXPECTED_AFFIBODY_X_POSITIONS["LibB"], start=1
                )
            ],
        },
        "sources": source_hashes,
        "pairs": constructed,
    }
    audit = {
        "schema_version": 1,
        "created_utc": inputs["created_utc"],
        "warning": (
            "Direct retention was inspected only to choose three same-peptide diagnostic "
            "examples spanning the observed MW range. These values are audit-only and must "
            "not be exposed to feature extraction, training, tuning, or architecture selection."
        ),
        "choice_logic": {
            "reference": "provider/crystal reference MW + NNYYF",
            "synthetic_feature_qa": (
                "one alanine substitution on each partner at a crystal-supported contact position"
            ),
            "measured_controls": (
                "same MW peptide, low/middle/high direct-retention Affibodies to avoid changing "
                "both partners while checking whether extracted features vary"
            ),
        },
        "rows": audit_rows,
        "sources": source_hashes,
    }

    input_path.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (input_path, audit_path):
        os.chmod(path, 0o600)

    checksums = [
        "{}  {}".format(sha256_file(input_path), input_path.name),
        "{}  {}".format(sha256_file(audit_path), audit_path.name),
        "{}  {}".format(sha256_file(script_path), script_path),
    ]
    checksum_path.write_text("\n".join(checksums) + "\n", encoding="ascii")
    os.chmod(checksum_path, 0o600)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pairs": len(constructed),
                "model_facing_manifest_sha256": sha256_file(input_path),
                "retention_audit_sha256": sha256_file(audit_path),
                "elapsed_seconds": round(time.time() - started, 6),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "pandas": pd.__version__,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run(parse_args())
