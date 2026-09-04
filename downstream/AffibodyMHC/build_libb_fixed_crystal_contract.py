#!/usr/bin/env python
"""Build the outcome-free fixed-crystal input contract for LibB.

The StaB-ddG- and RDE-PPI-derived experiments share one structural template:
the experimentally resolved NY-ESO-1--xx133 complex.  This CPU-only builder
binds the already audited LibB sequence rows to that template.  It does not
import either model, construct labels, or open the sealed evaluation table.

The only row input is ``rows.json`` produced by
``build_esmfold2_libb_canonical_rows.py``.  That artifact contains sequences,
opaque row IDs, and train/eval membership, but no selection or direct-outcome
values.  The outputs are:

* ``residue_mapping.json``: sequence-derived crystal-chain identities,
  residue-number mappings, and three reference-distance checks;
* ``current_sequence_records.json``: the current peptide and Affibody
  identities for every canonical row in crystal-residue order; and
* ``manifest.json``: hashes, counts, and leakage checks.

All outputs must live under the Git-ignored ``private_data`` directory.  An
existing output directory is never overwritten.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.analyze_nyeso_xx133_structure import (
    classify_chains,
    build_crystal_copies,
    euclidean,
    parse_pdb,
    representative_atom,
)


SCHEMA_VERSION = "libb-fixed-crystal-contract-v2"
SOURCE_ROWS_SCHEMA_VERSION = "esmfold2-libb-canonical-rows-v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "esmfold2-libb-canonical-rows-audit-v1"

MAPPING_FILENAME = "residue_mapping.json"
RECORDS_FILENAME = "current_sequence_records.json"
MANIFEST_FILENAME = "manifest.json"

LIBRARY = "LibB"
CHAIN1_LENGTH = 270
AFFIBODY_LENGTH = 58
PEPTIDE_LENGTH = 9
SMART_PREFIX_LENGTH = 70
ASSAY_HLA_START_INDEX_0_BASED = 70
ASSAY_HLA_ALIGNED_LENGTH = 181
ASSAY_HLA_END_INDEX_EXCLUSIVE_0_BASED = (
    ASSAY_HLA_START_INDEX_0_BASED + ASSAY_HLA_ALIGNED_LENGTH
)
PEPTIDE_DESIGN_POSITIONS = (4, 5)
AFFIBODY_DESIGN_POSITIONS = (6, 10, 13, 14, 17)
AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")

REFERENCE_PEPTIDE_CODE = "MW"
REFERENCE_AFFIBODY_CODE = "NNYYF"
EXPECTED_PDB_SHA256 = (
    "59d026542cc42006302117cc79ad720497e69c4298f95598413f1d172d407b0e"
)
EXPECTED_PRIMARY_CHAIN_IDS = {
    "hla_chain": "A",
    "beta2m_chain": "B",
    "peptide_chain": "P",
    "affibody_chain": "H",
}
EXPECTED_COUNTS = {
    "train": 30648,
    "eval": 120,
    "total": 30768,
}
EXPECTED_HLA_IDENTITY_OVERRIDES = (
    (84, "Y", "A"),
    (167, "W", "A"),
)

# C-beta distances in the selected A/B/P--H crystal copy.  Glycine would use
# C-alpha, although none of these six reference residues is glycine.
KNOWN_DISTANCE_CHECKS = (
    (4, 14, 6.9341),
    (4, 17, 7.8235),
    (5, 10, 5.9667),
)
DISTANCE_TOLERANCE_ANGSTROM = 0.05

SOURCE_ROW_FIELDS = (
    "row_index",
    "row_id",
    "split",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
)
CURRENT_RECORD_FIELDS = (
    "row_index",
    "row_id",
    "split",
    "peptide_sequence",
    "affibody_sequence",
    "peptide_code",
    "affibody_code",
    "peptide_resolved_sequence",
    "affibody_resolved_sequence",
    "sequence_pair_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _json_bytes(payload):
    return (
        json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _portable_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def validate_private_output_path(path, repo_root=REPO_ROOT):
    """Fail closed unless ``path`` is a Git-ignored child of private_data."""
    repo_root = Path(repo_root).resolve()
    private_root = (repo_root / "private_data").resolve()
    path = Path(path).resolve()
    _require(private_root.is_dir(), "repository private_data directory is missing")
    _require(path != private_root, "output must be below private_data")
    try:
        relative_private = path.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("output directory must be below private_data") from exc
    _require(bool(relative_private.parts), "output must be below private_data")
    relative_repo = path.relative_to(repo_root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative_repo)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(ignored.returncode == 0, "output directory is not Git-ignored")
    return path


def _atomic_write_json(path, payload):
    _require(not path.exists(), "output exists: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    _require(not temporary.exists(), "temporary output exists")
    try:
        with open(str(temporary), "wb") as handle:
            handle.write(_json_bytes(payload))
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(private_mode(path) == "0600", "JSON output is not mode 0600")


@dataclass(frozen=True)
class ReferenceSequences:
    """The label-free canonical row matching the crystallized sequence."""

    row_id: str
    split: str
    chain1_sequence: str
    assay_hla_sequence: str
    peptide_sequence: str
    affibody_sequence: str


def _expected(expected, key):
    _require(key in expected, "expected-count contract lacks {}".format(key))
    return int(expected[key])


def load_canonical_rows(rows_path, manifest_path):
    """Load and hash-bind the label-free upstream sequence artifact."""
    rows_path = Path(rows_path)
    manifest_path = Path(manifest_path)
    _require(rows_path.is_file(), "canonical row artifact is missing")
    _require(manifest_path.is_file(), "canonical row manifest is missing")
    payload = _read_json(rows_path)
    manifest = _read_json(manifest_path)
    _require(
        set(payload) == {"schema_version", "rows"},
        "canonical row payload has unexpected top-level fields",
    )
    _require(
        payload["schema_version"] == SOURCE_ROWS_SCHEMA_VERSION,
        "canonical row schema changed",
    )
    _require(
        manifest.get("schema_version") == SOURCE_MANIFEST_SCHEMA_VERSION,
        "canonical row manifest schema changed",
    )
    rows_sha256 = sha256_file(rows_path)
    _require(
        manifest.get("artifact", {}).get("sha256") == rows_sha256,
        "canonical row hash disagrees with its manifest",
    )
    _require(
        manifest.get("split_policy") == "library_local_strict_partner_disjoint",
        "canonical rows do not declare the strict partner-disjoint split",
    )
    isolation = manifest.get("partner_isolation", {})
    for key in (
        "peptide_identity_overlap",
        "affibody_identity_overlap",
        "sequence_pair_overlap",
    ):
        _require(isolation.get(key) == 0, "upstream isolation check failed: {}".format(key))
    return payload["rows"], {
        "canonical_rows_sha256": rows_sha256,
        "canonical_manifest_sha256": sha256_file(manifest_path),
        "canonical_schema_version": payload["schema_version"],
    }


def _derive_codes(chain1, affibody):
    peptide_sequence = chain1[-PEPTIDE_LENGTH:]
    peptide_code = "".join(
        peptide_sequence[position - 1] for position in PEPTIDE_DESIGN_POSITIONS
    )
    affibody_code = "".join(
        affibody[position - 1] for position in AFFIBODY_DESIGN_POSITIONS
    )
    return peptide_sequence, peptide_code, affibody_code


def validate_and_normalize_rows(rows, expected=None):
    """Validate canonical identities and return outcome-free structure rows."""
    if expected is None:
        expected = EXPECTED_COUNTS
    _require(isinstance(rows, list), "canonical rows must be a list")
    _require(len(rows) == _expected(expected, "total"), "canonical row total changed")

    records = []
    seen_ids = set()
    seen_pair_hashes = set()
    chain1_by_row_id = {}
    shared_chain1_prefixes = set()
    assay_hla_sequences = set()
    for expected_index, row in enumerate(rows):
        _require(isinstance(row, dict), "canonical row is not an object")
        _require(
            set(row) == set(SOURCE_ROW_FIELDS),
            "canonical row schema contains non-sequence material",
        )
        _require(row["row_index"] == expected_index, "canonical row index is not contiguous")
        row_id = str(row["row_id"])
        split = str(row["split"])
        chain1 = str(row["chain1_sequence"])
        affibody = str(row["chain2_sequence"])
        pair_hash = str(row["sequence_pair_sha256"])
        _require(row_id and row_id not in seen_ids, "duplicate or empty canonical row ID")
        _require(split in {"train", "eval"}, "unknown canonical split")
        _require(len(chain1) == CHAIN1_LENGTH, "canonical chain-1 length changed")
        _require(len(affibody) == AFFIBODY_LENGTH, "canonical Affibody length changed")
        _require(set(chain1).issubset(AA_ALPHABET), "noncanonical chain-1 residue")
        _require(set(affibody).issubset(AA_ALPHABET), "noncanonical Affibody residue")
        calculated_hash = sha256_text(chain1 + "|" + affibody)
        _require(pair_hash == calculated_hash, "canonical sequence-pair hash mismatch")
        _require(pair_hash not in seen_pair_hashes, "duplicate canonical sequence pair")

        peptide_sequence, peptide_code, affibody_code = _derive_codes(chain1, affibody)
        _require(
            peptide_sequence == "SLL{}ITQV".format(peptide_code),
            "LibB peptide does not follow the expected nine-residue template",
        )
        _require(len(peptide_code) == 2, "LibB peptide code length changed")
        _require(len(affibody_code) == 5, "LibB Affibody code length changed")
        assay_hla_sequence = chain1[
            ASSAY_HLA_START_INDEX_0_BASED:ASSAY_HLA_END_INDEX_EXCLUSIVE_0_BASED
        ]
        _require(
            len(assay_hla_sequence) == ASSAY_HLA_ALIGNED_LENGTH,
            "assay HLA segment length changed",
        )
        records.append(
            {
                "row_index": expected_index,
                "row_id": row_id,
                "split": split,
                "peptide_sequence": peptide_sequence,
                "affibody_sequence": affibody,
                "peptide_code": peptide_code,
                "affibody_code": affibody_code,
                # Filled after the PDB residue mapping has been established.
                "peptide_resolved_sequence": None,
                "affibody_resolved_sequence": None,
                "sequence_pair_sha256": pair_hash,
            }
        )
        seen_ids.add(row_id)
        seen_pair_hashes.add(pair_hash)
        chain1_by_row_id[row_id] = chain1
        shared_chain1_prefixes.add(chain1[:-PEPTIDE_LENGTH])
        assay_hla_sequences.add(assay_hla_sequence)

    train = [record for record in records if record["split"] == "train"]
    evaluation = [record for record in records if record["split"] == "eval"]
    _require(len(train) == _expected(expected, "train"), "canonical train size changed")
    _require(len(evaluation) == _expected(expected, "eval"), "canonical eval size changed")
    _require(
        len(shared_chain1_prefixes) == 1,
        "canonical rows do not share one fixed SMART-HLA-linker sequence",
    )
    _require(
        len(assay_hla_sequences) == 1,
        "canonical rows do not share one fixed assay HLA sequence",
    )

    def values(block, field):
        return {record[field] for record in block}

    overlap = {
        "peptide_sequence_overlap": len(
            values(train, "peptide_sequence").intersection(
                values(evaluation, "peptide_sequence")
            )
        ),
        "affibody_sequence_overlap": len(
            values(train, "affibody_sequence").intersection(
                values(evaluation, "affibody_sequence")
            )
        ),
        "sequence_pair_overlap": len(
            values(train, "sequence_pair_sha256").intersection(
                values(evaluation, "sequence_pair_sha256")
            )
        ),
        "row_id_overlap": len(
            values(train, "row_id").intersection(values(evaluation, "row_id"))
        ),
    }
    _require(all(value == 0 for value in overlap.values()), "strict partner split is violated")

    matches = [
        record
        for record in records
        if record["peptide_code"] == REFERENCE_PEPTIDE_CODE
        and record["affibody_code"] == REFERENCE_AFFIBODY_CODE
    ]
    _require(len(matches) == 1, "expected one canonical MW/NNYYF reference row")
    reference_record = matches[0]
    _require(reference_record["split"] == "eval", "crystal reference row is not in eval")
    reference_chain1 = chain1_by_row_id[reference_record["row_id"]]
    reference_assay_hla = reference_chain1[
        ASSAY_HLA_START_INDEX_0_BASED:ASSAY_HLA_END_INDEX_EXCLUSIVE_0_BASED
    ]
    reference = ReferenceSequences(
        row_id=reference_record["row_id"],
        split=reference_record["split"],
        chain1_sequence=reference_chain1,
        assay_hla_sequence=reference_assay_hla,
        peptide_sequence=reference_record["peptide_sequence"],
        affibody_sequence=reference_record["affibody_sequence"],
    )
    audit = {
        "train": len(train),
        "eval": len(evaluation),
        "total": len(records),
        "unique_train_peptides": len(values(train, "peptide_sequence")),
        "unique_train_affibodies": len(values(train, "affibody_sequence")),
        "unique_eval_peptides": len(values(evaluation, "peptide_sequence")),
        "unique_eval_affibodies": len(values(evaluation, "affibody_sequence")),
        "unique_fixed_smart_hla_linker_sequences": len(shared_chain1_prefixes),
        "unique_assay_hla_sequences": len(assay_hla_sequences),
        "partner_overlap": overlap,
    }
    return records, reference, audit


def _copy_residue_count(copy_record, chains):
    return sum(
        len(chains[copy_record[key]].residues)
        for key in ("hla_chain", "beta2m_chain", "peptide_chain", "affibody_chain")
    )


def select_primary_crystal_copy(copies, chains):
    """Select the more completely resolved copy without using chain names."""
    _require(len(copies) >= 1, "no complete crystal copies were derived")
    scored = [(_copy_residue_count(copy, chains), copy) for copy in copies]
    best_score = max(score for score, _ in scored)
    best = [copy for score, copy in scored if score == best_score]
    _require(len(best) == 1, "primary crystal copy is ambiguous by completeness")
    return best[0], best_score


def _mapping_entry(partner, residue, sequence_position, chain_ordinal, design_positions):
    representative = representative_atom(residue)
    mutable = sequence_position in design_positions
    return {
        "partner": partner,
        "sequence_position_1_based": int(sequence_position),
        "pdb_chain_id": residue.chain_id,
        "pdb_residue_number": int(residue.number),
        "pdb_insertion_code": residue.insertion_code.strip(),
        "pdb_residue_id": "{}:{}{}".format(
            residue.chain_id, residue.number, residue.insertion_code.strip()
        ),
        "pdb_chain_ordinal_1_based": int(chain_ordinal),
        "reference_amino_acid": residue.name1,
        "mutable_site": bool(mutable),
        "design_code_position_1_based": (
            design_positions.index(sequence_position) + 1 if mutable else None
        ),
        "representative_atom": representative.name,
        "representative_xyz_angstrom": [
            float(representative.x),
            float(representative.y),
            float(representative.z),
        ],
        "resolved_atom_names": sorted(residue.atoms),
    }


def _chain_inventory_record(chain, role):
    return {
        "chain_id": chain.chain_id,
        "role": role,
        "coordinate_residue_count": len(chain.residues),
        "coordinate_heavy_atom_count": len(chain.atoms),
        "sequence": chain.sequence,
        "sequence_sha256": sha256_text(chain.sequence),
        "first_pdb_residue": chain.residues[0].pdb_label,
        "last_pdb_residue": chain.residues[-1].pdb_label,
    }


def validate_assay_hla_identity_differences(pdb_hla_sequence, assay_hla_sequence):
    """Require the two documented crystal-to-assay substitutions and no others."""
    _require(
        len(pdb_hla_sequence) == ASSAY_HLA_ALIGNED_LENGTH,
        "PDB HLA comparison length changed",
    )
    _require(
        len(assay_hla_sequence) == ASSAY_HLA_ALIGNED_LENGTH,
        "assay HLA comparison length changed",
    )
    observed = tuple(
        (position, pdb_amino_acid, assay_amino_acid)
        for position, (pdb_amino_acid, assay_amino_acid) in enumerate(
            zip(pdb_hla_sequence, assay_hla_sequence), 1
        )
        if pdb_amino_acid != assay_amino_acid
    )
    _require(
        observed == EXPECTED_HLA_IDENTITY_OVERRIDES,
        "assay/PDB HLA differences changed: expected {}; observed {}".format(
            EXPECTED_HLA_IDENTITY_OVERRIDES, observed
        ),
    )
    return observed


def build_assay_hla_mapping(reference, hla_chain):
    """Map the assay's 181 HLA residues onto chain A's fixed coordinates."""
    _require(
        len(hla_chain.residues) >= ASSAY_HLA_ALIGNED_LENGTH,
        "selected PDB HLA chain is shorter than the assay HLA segment",
    )
    _require(
        reference.chain1_sequence[
            ASSAY_HLA_START_INDEX_0_BASED:ASSAY_HLA_END_INDEX_EXCLUSIVE_0_BASED
        ]
        == reference.assay_hla_sequence,
        "reference chain-1 HLA offset is inconsistent",
    )
    pdb_residues = hla_chain.residues[:ASSAY_HLA_ALIGNED_LENGTH]
    pdb_hla_sequence = "".join(residue.name1 for residue in pdb_residues)
    differences = validate_assay_hla_identity_differences(
        pdb_hla_sequence, reference.assay_hla_sequence
    )
    difference_positions = {position for position, _, _ in differences}

    residue_rows = []
    identity_overrides = []
    for hla_position, residue in enumerate(pdb_residues, 1):
        chain1_index = ASSAY_HLA_START_INDEX_0_BASED + hla_position - 1
        assay_amino_acid = reference.assay_hla_sequence[hla_position - 1]
        overridden = hla_position in difference_positions
        row = {
            "hla_sequence_position_1_based": hla_position,
            "canonical_chain1_index_0_based": chain1_index,
            "canonical_chain1_position_1_based": chain1_index + 1,
            "pdb_chain_id": residue.chain_id,
            "pdb_chain_ordinal_1_based": hla_position,
            "pdb_residue_number": int(residue.number),
            "pdb_insertion_code": residue.insertion_code.strip(),
            "pdb_residue_id": "{}:{}{}".format(
                residue.chain_id, residue.number, residue.insertion_code.strip()
            ),
            "pdb_amino_acid": residue.name1,
            "assay_amino_acid": assay_amino_acid,
            "model_amino_acid": assay_amino_acid,
            "identity_override": overridden,
            "coordinate_policy": "keep PDB coordinates unchanged",
        }
        residue_rows.append(row)
        if overridden:
            substitution = "{}{}{}".format(
                residue.name1, hla_position, assay_amino_acid
            )
            identity_overrides.append(
                {
                    "substitution": substitution,
                    "hla_sequence_position_1_based": hla_position,
                    "canonical_chain1_index_0_based": chain1_index,
                    "canonical_chain1_position_1_based": chain1_index + 1,
                    "pdb_chain_id": residue.chain_id,
                    "pdb_chain_ordinal_1_based": hla_position,
                    "pdb_residue_number": int(residue.number),
                    "pdb_insertion_code": residue.insertion_code.strip(),
                    "pdb_residue_id": row["pdb_residue_id"],
                    "pdb_amino_acid": residue.name1,
                    "assay_amino_acid": assay_amino_acid,
                    "model_amino_acid": assay_amino_acid,
                    "coordinate_policy": "keep PDB coordinates unchanged",
                }
            )

    _require(
        [row["substitution"] for row in identity_overrides] == ["Y84A", "W167A"],
        "HLA identity-override names changed",
    )
    return {
        "canonical_chain1_hla_start_index_0_based": ASSAY_HLA_START_INDEX_0_BASED,
        "canonical_chain1_hla_end_index_exclusive_0_based": (
            ASSAY_HLA_END_INDEX_EXCLUSIVE_0_BASED
        ),
        "aligned_hla_residue_count": ASSAY_HLA_ALIGNED_LENGTH,
        "pdb_chain_id": hla_chain.chain_id,
        "pdb_chain_region": "first 181 coordinate residues",
        "exact_identity_match_count": ASSAY_HLA_ALIGNED_LENGTH
        - len(identity_overrides),
        "identity_override_count": len(identity_overrides),
        "identity_overrides": identity_overrides,
        "residues": residue_rows,
    }


def build_residue_mapping(pdb_path, reference, enforce_current_pdb=True):
    """Derive the A/B/P--H copy and validate residue-level reference mapping."""
    pdb_path = Path(pdb_path)
    _require(pdb_path.is_file(), "reference PDB is missing")
    pdb_sha256 = sha256_file(pdb_path)
    if enforce_current_pdb:
        _require(pdb_sha256 == EXPECTED_PDB_SHA256, "reference PDB content changed")

    chains = parse_pdb(pdb_path)
    roles, sequence_mappings = classify_chains(chains, reference)
    copies = build_crystal_copies(chains, roles)
    primary, completeness_score = select_primary_crystal_copy(copies, chains)
    for key, expected_chain_id in EXPECTED_PRIMARY_CHAIN_IDS.items():
        _require(
            primary[key] == expected_chain_id,
            "sequence/geometry-derived primary {} changed".format(key),
        )

    peptide_chain = chains[primary["peptide_chain"]]
    affibody_chain = chains[primary["affibody_chain"]]
    hla_chain = chains[primary["hla_chain"]]
    assay_hla_mapping = build_assay_hla_mapping(reference, hla_chain)
    peptide_mapping = sequence_mappings[peptide_chain.chain_id]
    affibody_mapping = sequence_mappings[affibody_chain.chain_id]
    peptide_entries = [
        _mapping_entry(
            "peptide",
            residue,
            peptide_mapping[id(residue)],
            ordinal,
            PEPTIDE_DESIGN_POSITIONS,
        )
        for ordinal, residue in enumerate(peptide_chain.residues, 1)
    ]
    affibody_entries = [
        _mapping_entry(
            "affibody",
            residue,
            affibody_mapping[id(residue)],
            ordinal,
            AFFIBODY_DESIGN_POSITIONS,
        )
        for ordinal, residue in enumerate(affibody_chain.residues, 1)
    ]
    peptide_entries.sort(key=lambda row: row["sequence_position_1_based"])
    affibody_entries.sort(key=lambda row: row["sequence_position_1_based"])
    _require(
        [row["sequence_position_1_based"] for row in peptide_entries]
        == list(range(1, PEPTIDE_LENGTH + 1)),
        "the complete nine-residue peptide is not resolved",
    )
    _require(len(affibody_entries) == 55, "primary Affibody resolved length changed")
    _require(
        [row["sequence_position_1_based"] for row in affibody_entries]
        == list(range(3, 58)),
        "primary Affibody must map contiguously to sequence positions 3--57",
    )
    for row in peptide_entries:
        _require(
            reference.peptide_sequence[row["sequence_position_1_based"] - 1]
            == row["reference_amino_acid"],
            "peptide reference-to-PDB residue mismatch",
        )
    for row in affibody_entries:
        _require(
            reference.affibody_sequence[row["sequence_position_1_based"] - 1]
            == row["reference_amino_acid"],
            "Affibody reference-to-PDB residue mismatch",
        )
    _require(
        [
            row["sequence_position_1_based"]
            for row in peptide_entries
            if row["mutable_site"]
        ]
        == list(PEPTIDE_DESIGN_POSITIONS),
        "not every peptide design site is resolved",
    )
    _require(
        [
            row["sequence_position_1_based"]
            for row in affibody_entries
            if row["mutable_site"]
        ]
        == list(AFFIBODY_DESIGN_POSITIONS),
        "not every Affibody design site is resolved",
    )

    peptide_lookup = {
        row["sequence_position_1_based"]: row for row in peptide_entries
    }
    affibody_lookup = {
        row["sequence_position_1_based"]: row for row in affibody_entries
    }
    distance_checks = []
    for peptide_position, affibody_position, expected_distance in KNOWN_DISTANCE_CHECKS:
        peptide_row = peptide_lookup[peptide_position]
        affibody_row = affibody_lookup[affibody_position]
        actual = euclidean(
            peptide_row["representative_xyz_angstrom"],
            affibody_row["representative_xyz_angstrom"],
        )
        difference = abs(actual - expected_distance)
        _require(
            difference <= DISTANCE_TOLERANCE_ANGSTROM,
            "known crystal distance could not be reproduced",
        )
        distance_checks.append(
            {
                "peptide_sequence_position_1_based": peptide_position,
                "peptide_pdb_residue_id": peptide_row["pdb_residue_id"],
                "peptide_reference_amino_acid": peptide_row[
                    "reference_amino_acid"
                ],
                "affibody_sequence_position_1_based": affibody_position,
                "affibody_pdb_residue_id": affibody_row["pdb_residue_id"],
                "affibody_reference_amino_acid": affibody_row[
                    "reference_amino_acid"
                ],
                "atom_rule": "C-beta; C-alpha for glycine",
                "peptide_atom": peptide_row["representative_atom"],
                "affibody_atom": affibody_row["representative_atom"],
                "expected_distance_angstrom": expected_distance,
                "observed_distance_angstrom": actual,
                "absolute_difference_angstrom": difference,
                "tolerance_angstrom": DISTANCE_TOLERANCE_ANGSTROM,
            }
        )

    selected_keys = ("hla_chain", "beta2m_chain", "peptide_chain", "affibody_chain")
    selected_roles = {
        "hla_chain": "HLA-A*02 heavy chain",
        "beta2m_chain": "beta-2-microglobulin",
        "peptide_chain": "NY-ESO-1 peptide",
        "affibody_chain": "xx133 Affibody",
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "library": LIBRARY,
        "fixed_geometry_policy": (
            "Use the selected reference coordinates for every row; graft the "
            "fixed assay HLA identities A84 and A167 onto unchanged chain-A "
            "coordinates, then replace row-specific peptide and Affibody "
            "identities from current_sequence_records.json."
        ),
        "pdb_source": {
            "repo_relative_path": _portable_path(pdb_path),
            "sha256": pdb_sha256,
        },
        "reference_pair": {
            "row_id": reference.row_id,
            "split": reference.split,
            "peptide_code": REFERENCE_PEPTIDE_CODE,
            "affibody_code": REFERENCE_AFFIBODY_CODE,
            "peptide_sequence": reference.peptide_sequence,
            "affibody_sequence": reference.affibody_sequence,
            "assay_hla_sequence_sha256": sha256_text(
                reference.assay_hla_sequence
            ),
        },
        "chain_identification": {
            "method": (
                "Sequence-role matching followed by minimum-distance copy "
                "assignment; select the uniquely most completely resolved copy."
            ),
            "all_chain_roles": {
                chain_id: roles[chain_id] for chain_id in sorted(roles)
            },
            "candidate_copy_count": len(copies),
            "primary_copy_completeness_residue_count": completeness_score,
            "selected_copy": {key: primary[key] for key in selected_keys},
            "selected_chain_inventory": [
                _chain_inventory_record(chains[primary[key]], selected_roles[key])
                for key in selected_keys
            ],
            "excluded_from_model_complex": sorted(
                chain_id
                for chain_id in chains
                if chain_id not in {primary[key] for key in selected_keys}
            ),
        },
        "residue_mappings": {
            "assay_hla": assay_hla_mapping,
            "peptide": peptide_entries,
            "affibody": affibody_entries,
        },
        "known_distance_checks": distance_checks,
    }
    return payload


def apply_residue_mapping(records, mapping):
    """Materialize each current sequence in the mapped PDB residue order."""
    peptide_positions = [
        row["sequence_position_1_based"]
        for row in mapping["residue_mappings"]["peptide"]
    ]
    affibody_positions = [
        row["sequence_position_1_based"]
        for row in mapping["residue_mappings"]["affibody"]
    ]
    output = []
    for source in records:
        record = dict(source)
        record["peptide_resolved_sequence"] = "".join(
            source["peptide_sequence"][position - 1] for position in peptide_positions
        )
        record["affibody_resolved_sequence"] = "".join(
            source["affibody_sequence"][position - 1]
            for position in affibody_positions
        )
        _require(
            set(record) == set(CURRENT_RECORD_FIELDS),
            "current structure-record schema mismatch",
        )
        _require(
            len(record["peptide_resolved_sequence"]) == len(peptide_positions),
            "current peptide mapping length mismatch",
        )
        _require(
            len(record["affibody_resolved_sequence"]) == len(affibody_positions),
            "current Affibody mapping length mismatch",
        )
        output.append(record)
    return output


def build_records_payload(records, mapping_file_sha256):
    return {
        "schema_version": SCHEMA_VERSION,
        "residue_mapping_file_sha256": mapping_file_sha256,
        "rows": records,
    }


def build_contract(rows_path, rows_manifest_path, pdb_path, expected=None):
    """Build both machine-readable payloads in memory without model calls."""
    rows, lineage = load_canonical_rows(rows_path, rows_manifest_path)
    records, reference, audit = validate_and_normalize_rows(rows, expected=expected)
    mapping = build_residue_mapping(pdb_path, reference)
    mapping_sha256 = hashlib.sha256(_json_bytes(mapping)).hexdigest()
    records = apply_residue_mapping(records, mapping)
    records_payload = build_records_payload(records, mapping_sha256)
    return mapping, records_payload, lineage, audit


def run_build(args):
    mapping, records_payload, lineage, audit = build_contract(
        args.canonical_rows,
        args.canonical_rows_manifest,
        args.pdb,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "train": audit["train"],
        "eval": audit["eval"],
        "total": audit["total"],
        "primary_chains": mapping["chain_identification"]["selected_copy"],
        "distance_checks": len(mapping["known_distance_checks"]),
        "hla_identity_overrides": [
            row["substitution"]
            for row in mapping["residue_mappings"]["assay_hla"][
                "identity_overrides"
            ]
        ],
    }
    if args.check_only:
        print(json.dumps(summary, sort_keys=True))
        return summary

    output_dir = validate_private_output_path(args.output_dir)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(str(output_dir), 0o700)
    _require(private_mode(output_dir) == "0700", "output directory is not mode 0700")

    mapping_path = output_dir / MAPPING_FILENAME
    records_path = output_dir / RECORDS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    _atomic_write_json(mapping_path, mapping)
    _require(
        sha256_file(mapping_path) == records_payload["residue_mapping_file_sha256"],
        "mapping serialization hash changed",
    )
    _atomic_write_json(records_path, records_payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare_fixed_crystal_inputs",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_hashes": dict(
            lineage,
            pdb_sha256=mapping["pdb_source"]["sha256"],
            script_sha256=sha256_file(Path(__file__).resolve()),
        ),
        "data_access": {
            "canonical_sequence_rows_read": True,
            "training_label_table_read": False,
            "evaluation_outcome_table_read": False,
        },
        "split_audit": audit,
        "structure_audit": {
            "chain_ids_derived_from_sequences_and_geometry": True,
            "selected_copy": mapping["chain_identification"]["selected_copy"],
            "known_distances_reproduced": len(mapping["known_distance_checks"]),
            "assay_hla_alignment": {
                "canonical_chain1_hla_start_index_0_based": mapping[
                    "residue_mappings"
                ]["assay_hla"]["canonical_chain1_hla_start_index_0_based"],
                "aligned_hla_residue_count": mapping["residue_mappings"][
                    "assay_hla"
                ]["aligned_hla_residue_count"],
                "pdb_chain_id": mapping["residue_mappings"]["assay_hla"][
                    "pdb_chain_id"
                ],
                "exact_identity_match_count": mapping["residue_mappings"][
                    "assay_hla"
                ]["exact_identity_match_count"],
                "identity_override_count": mapping["residue_mappings"][
                    "assay_hla"
                ]["identity_override_count"],
                "identity_overrides": mapping["residue_mappings"]["assay_hla"][
                    "identity_overrides"
                ],
                "all_other_aligned_residues_match": True,
            },
        },
        "outputs": {
            MAPPING_FILENAME: {
                "sha256": sha256_file(mapping_path),
                "mode": private_mode(mapping_path),
            },
            RECORDS_FILENAME: {
                "sha256": sha256_file(records_path),
                "mode": private_mode(records_path),
                "rows": len(records_payload["rows"]),
            },
        },
    }
    _atomic_write_json(manifest_path, manifest)
    summary["output_dir"] = _portable_path(output_dir)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-rows",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json",
    )
    parser.add_argument(
        "--canonical-rows-manifest",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/manifest.json",
    )
    parser.add_argument(
        "--pdb", type=Path, default=REPO_ROOT / "nyeso_xx133_complex.pdb"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate counts, split, mapping, and distances without writing",
    )
    return parser.parse_args(argv)


def main(argv=None):
    return run_build(parse_args(argv))


if __name__ == "__main__":
    main()
