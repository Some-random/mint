#!/usr/bin/env python
"""Reconstruct provider-specified MINT chains for the Affibody--pMHC assay.

The full provider sequences and row-level assay values are written only below
the repository's ignored ``private_data`` tree.  The sequence templates are
read from the updated provider PowerPoint inside its ZIP archive rather than
being copied into this reusable source file.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    LIBRARY_SPECS,
    load_retention_table,
    sha256_file,
    validate_private_output_path,
)


DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
EXPECTED_CONSTRUCT_LENGTH = 342
CHAIN1_LENGTH = 270
OMITTED_LINKER = "GGSLEVLFQGPGSG"
CHAIN2_LENGTH = 58
EXPECTED_X_POSITIONS = {
    "LibA": (265, 266, 297, 301, 311, 315),
    "LibB": (265, 266, 290, 294, 297, 298, 301),
}
EXPECTED_AFFIBODY_X_POSITIONS = {
    "LibA": (13, 17, 27, 31),
    "LibB": (6, 10, 13, 14, 17),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value):
    return _sha256_bytes(value.encode("ascii"))


def _sequence_candidates(slide_xml):
    root = ElementTree.fromstring(slide_xml)
    candidates = []
    shape_tag = "{{{}}}sp".format(PRESENTATION_NS)
    text_tag = "{{{}}}t".format(DRAWING_NS)
    for shape in root.iter(shape_tag):
        text = "".join(node.text or "" for node in shape.iter(text_tag))
        if len(text) >= 300 and set(text).issubset(set(AA_ALPHABET) | {"X"}):
            candidates.append(text)
    return candidates


def load_provider_templates(archive_path):
    with zipfile.ZipFile(str(archive_path), "r") as outer:
        members = [
            name
            for name in outer.namelist()
            if name.endswith("/pep-affibody sequences_updated.pptx")
            or name == "pep-affibody sequences_updated.pptx"
        ]
        _require(len(members) == 1, "expected exactly one updated sequence PowerPoint")
        member = members[0]
        pptx_bytes = outer.read(member)

    templates = {}
    with zipfile.ZipFile(io.BytesIO(pptx_bytes), "r") as deck:
        slide_names = sorted(
            name
            for name in deck.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for slide_name in slide_names:
            for candidate in _sequence_candidates(deck.read(slide_name)):
                positions = tuple(index + 1 for index, value in enumerate(candidate) if value == "X")
                for library, expected in EXPECTED_X_POSITIONS.items():
                    if positions == expected:
                        if library in templates:
                            _require(
                                templates[library] == candidate,
                                "conflicting duplicate {} template".format(library),
                            )
                        else:
                            templates[library] = candidate

    _require(set(templates) == set(LIBRARY_SPECS), "could not identify both library templates")
    for library, template in templates.items():
        _require(
            len(template) == EXPECTED_CONSTRUCT_LENGTH,
            "{} construct length mismatch".format(library),
        )
        _require(
            template[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)] == OMITTED_LINKER,
            "{} omitted-linker mismatch".format(library),
        )
        affibody_template = template[-CHAIN2_LENGTH:]
        affibody_positions = tuple(
            index + 1 for index, value in enumerate(affibody_template) if value == "X"
        )
        _require(
            affibody_positions == EXPECTED_AFFIBODY_X_POSITIONS[library],
            "{} Affibody placeholder positions mismatch".format(library),
        )
    _require(
        templates["LibA"][:CHAIN1_LENGTH] == templates["LibB"][:CHAIN1_LENGTH],
        "library chain-1 templates differ",
    )
    return templates, member, _sha256_bytes(pptx_bytes)


def fill_template(template, code):
    _require(template.count("X") == len(code), "placeholder/code-length mismatch")
    iterator = iter(code)
    result = "".join(next(iterator) if value == "X" else value for value in template)
    _require("X" not in result, "unfilled sequence placeholder")
    _require(set(result).issubset(set(AA_ALPHABET)), "noncanonical reconstructed sequence")
    return result


def reconstruct_row(row, templates):
    library = row["library"]
    code = row["peptide_design_code"] + row["affibody_design_code"]
    full_construct = fill_template(templates[library], code)
    chain1 = full_construct[:CHAIN1_LENGTH]
    omitted = full_construct[CHAIN1_LENGTH : CHAIN1_LENGTH + len(OMITTED_LINKER)]
    chain2 = full_construct[-CHAIN2_LENGTH:]
    _require(len(chain1) == CHAIN1_LENGTH, "chain 1 length mismatch")
    _require(omitted == OMITTED_LINKER, "row omitted-linker mismatch")
    _require(len(chain2) == CHAIN2_LENGTH, "chain 2 length mismatch")
    _require(chain1[264:266] == row["peptide_design_code"], "peptide mapping mismatch")
    positions = EXPECTED_AFFIBODY_X_POSITIONS[library]
    observed_affibody_code = "".join(chain2[position - 1] for position in positions)
    _require(observed_affibody_code == row["affibody_design_code"], "Affibody mapping mismatch")
    return chain1, chain2, omitted, full_construct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention-csv", required=True, type=Path)
    parser.add_argument("--sequence-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def run(args):
    started = time.time()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    output_parent = validate_private_output_path(args.output.parent, repo_root)
    manifest_parent = validate_private_output_path(args.manifest.parent, repo_root)
    _require(output_parent == manifest_parent, "output and manifest must share a private directory")
    _require(args.retention_csv.is_file(), "retention CSV does not exist")
    _require(args.sequence_zip.is_file(), "updated sequence ZIP does not exist")
    _require(not args.output.exists(), "output already exists; refusing overwrite")
    _require(not args.manifest.exists(), "manifest already exists; refusing overwrite")

    source_retention_hash = sha256_file(args.retention_csv)
    source_archive_hash = sha256_file(args.sequence_zip)
    source = load_retention_table(args.retention_csv)
    templates, deck_member, deck_hash = load_provider_templates(args.sequence_zip)

    records = []
    for _, row in source.iterrows():
        chain1, chain2, omitted, full_construct = reconstruct_row(row, templates)
        record = row.to_dict()
        record.update(
            {
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "chain1_length": len(chain1),
                "chain2_length": len(chain2),
                "omitted_linker_sequence": omitted,
                "full_construct_audit_sequence": full_construct,
                "chain1_sha256": _sha256_text(chain1),
                "chain2_sha256": _sha256_text(chain2),
                "sequence_pair_sha256": _sha256_text(chain1 + "|" + chain2),
            }
        )
        records.append(record)
    output = pd.DataFrame(records)

    measured = output[output["target_retention"].notna()].copy()
    _require(len(output) == 228, "expected 228 designed sequence pairs")
    _require(len(measured) == 227, "expected 227 measured sequence pairs")
    _require(output["pair_uid"].nunique() == len(output), "duplicate pair UID")
    _require(output["sequence_pair_sha256"].nunique() == len(output), "duplicate sequence pair")
    _require(measured["chain1_sha256"].nunique() == 15, "unexpected measured chain-1 count")
    _require(measured["chain2_sha256"].nunique() == 22, "unexpected measured chain-2 count")
    _require(source_retention_hash == sha256_file(args.retention_csv), "retention source changed")
    _require(source_archive_hash == sha256_file(args.sequence_zip), "sequence archive changed")

    output.to_csv(args.output, index=False)
    os.chmod(str(args.output), 0o600)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "sources": {
            "retention_csv": {
                "path": str(args.retention_csv.resolve()),
                "sha256": source_retention_hash,
            },
            "sequence_zip": {
                "path": str(args.sequence_zip.resolve()),
                "sha256": source_archive_hash,
                "deck_member": deck_member,
                "deck_sha256": deck_hash,
            },
        },
        "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "provider_model_input": {
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "omitted_construct_positions_1_based": [271, 284],
            "omitted_linker_sequence": OMITTED_LINKER,
            "tcr_present": False,
            "separate_b2m_supplied": False,
        },
        "templates": {
            library: {
                "sha256": _sha256_text(template),
                "length": len(template),
                "x_positions_1_based": list(EXPECTED_X_POSITIONS[library]),
            }
            for library, template in sorted(templates.items())
        },
        "rows": {
            "designed": int(len(output)),
            "measured": int(len(measured)),
            "missing_retention": int(output["target_retention"].isna().sum()),
            "unique_sequence_pairs": int(output["sequence_pair_sha256"].nunique()),
            "measured_unique_chain1": int(measured["chain1_sha256"].nunique()),
            "measured_unique_chain2": int(measured["chain2_sha256"].nunique()),
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "mode": "0600",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(args.manifest), 0o600)
    print(json.dumps({"output": str(args.output), "rows": len(output), "measured": len(measured)}))


if __name__ == "__main__":
    run(parse_args())
