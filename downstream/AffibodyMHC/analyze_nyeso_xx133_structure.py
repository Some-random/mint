#!/usr/bin/env python
"""Audit the NY-ESO-1--xx133 crystal interface against the LibB sequence table.

This script extracts *observed crystal-coordinate distances*.  It does not
produce an AlphaFold distogram, which is a learned probability distribution
over distance bins.  Outputs are restricted to the repository's ignored
``private_data`` tree and an existing output directory is never overwritten.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AFFIBODY_DESIGN_POSITIONS = (6, 10, 13, 14, 17)
PEPTIDE_DESIGN_POSITIONS = (4, 5)
EXPECTED_LIBRARY = "LibB"
EXPECTED_AFFIBODY_CODE = "NNYYF"
EXPECTED_PEPTIDE_CODE = "MW"
EXPECTED_PEPTIDE_LENGTH = 9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_private_output_path(output_dir, repo_root):
    repo_root = Path(repo_root).resolve()
    private_root = (repo_root / "private_data").resolve()
    output_dir = Path(output_dir).resolve()
    _require(private_root.is_dir(), "private_data directory does not exist")
    _require(output_dir != private_root, "output must be below private_data")
    try:
        output_dir.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("output directory must be below private_data") from exc
    relative = output_dir.relative_to(repo_root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=str(repo_root),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(ignored.returncode == 0, "private output path is not Git-ignored")
    return output_dir


@dataclass(frozen=True)
class Atom:
    name: str
    altloc: str
    x: float
    y: float
    z: float
    occupancy: float
    element: str

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


@dataclass
class Residue:
    chain_id: str
    number: int
    insertion_code: str
    name3: str
    atoms: dict

    @property
    def name1(self):
        return AA3_TO_1[self.name3]

    @property
    def pdb_label(self):
        suffix = self.insertion_code.strip()
        return "{}{}".format(self.number, suffix)


@dataclass
class Chain:
    chain_id: str
    residues: list

    @property
    def sequence(self):
        return "".join(residue.name1 for residue in self.residues)

    @property
    def atoms(self):
        return [atom for residue in self.residues for atom in residue.atoms.values()]


@dataclass(frozen=True)
class ReferencePair:
    pair_uid: str
    peptide_sequence: str
    affibody_sequence: str
    retention_percent: float


def _infer_element(atom_field):
    value = atom_field.strip()
    while value and value[0].isdigit():
        value = value[1:]
    return value[:1].upper()


def parse_pdb(path):
    """Parse one PDB coordinate model, selecting the highest-occupancy altloc."""
    residue_order = []
    residue_names = {}
    atom_candidates = defaultdict(list)
    explicit_models = 0
    inside_model = True
    with open(path, "r", encoding="ascii") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n").ljust(80)
            record = line[0:6].strip()
            if record == "MODEL":
                explicit_models += 1
                _require(explicit_models == 1, "PDB contains more than one coordinate model")
                inside_model = True
                continue
            if record == "ENDMDL":
                inside_model = False
                continue
            if record != "ATOM" or not inside_model:
                continue
            name3 = line[17:20].strip().upper()
            if name3 not in AA3_TO_1:
                continue
            chain_id = line[21].strip()
            _require(chain_id, "blank chain ID at PDB line {}".format(line_number))
            try:
                number = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
            except ValueError as exc:
                raise ValueError("malformed ATOM record at PDB line {}".format(line_number)) from exc
            _require(
                all(math.isfinite(value) for value in (x, y, z, occupancy)),
                "non-finite ATOM value at PDB line {}".format(line_number),
            )
            insertion_code = line[26]
            atom_name = line[12:16].strip()
            altloc = line[16]
            element = line[76:78].strip().upper() or _infer_element(line[12:16])
            if element in {"H", "D"}:
                continue
            residue_key = (chain_id, number, insertion_code)
            if residue_key not in residue_names:
                residue_order.append(residue_key)
                residue_names[residue_key] = name3
            else:
                _require(
                    residue_names[residue_key] == name3,
                    "alternate residue identities are unsupported for {} {}".format(
                        chain_id, number
                    ),
                )
            atom_candidates[(residue_key, atom_name)].append(
                Atom(atom_name, altloc, x, y, z, occupancy, element)
            )

    _require(residue_order, "no canonical protein ATOM records found")
    residues_by_chain = OrderedDict()
    for residue_key in residue_order:
        chain_id, number, insertion_code = residue_key
        atoms = {}
        for (candidate_residue, atom_name), candidates in atom_candidates.items():
            if candidate_residue != residue_key:
                continue
            # Occupancy is primary.  At equal occupancy prefer blank, then A,
            # then lexical order, so selection is deterministic.
            atom = max(
                candidates,
                key=lambda candidate: (
                    candidate.occupancy,
                    2 if candidate.altloc == " " else 1 if candidate.altloc == "A" else 0,
                    candidate.altloc,
                ),
            )
            atoms[atom_name] = atom
        residue = Residue(
            chain_id=chain_id,
            number=number,
            insertion_code=insertion_code,
            name3=residue_names[residue_key],
            atoms=atoms,
        )
        residues_by_chain.setdefault(chain_id, []).append(residue)

    chains = OrderedDict(
        (chain_id, Chain(chain_id, residues))
        for chain_id, residues in residues_by_chain.items()
    )
    return chains


def parse_resolution(path):
    pattern = re.compile(r"RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([0-9.]+)")
    with open(path, "r", encoding="ascii") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    return None


def load_reference_pair(path):
    required = {
        "library",
        "affibody_design_code",
        "peptide_design_code",
        "retention_percent",
        "measurement_missing",
        "pair_uid",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "aff_p6",
        "aff_p10",
        "aff_p13",
        "aff_p14",
        "aff_p17",
        "pep_p4",
        "pep_p5",
    }
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, "retention table has no header")
        _require(required.issubset(reader.fieldnames), "retention table is missing required columns")
        rows = [
            row
            for row in reader
            if row["library"] == EXPECTED_LIBRARY
            and row["affibody_design_code"] == EXPECTED_AFFIBODY_CODE
            and row["peptide_design_code"] == EXPECTED_PEPTIDE_CODE
        ]
    _require(len(rows) == 1, "expected one exact LibB NNYYF/MW retention row")
    row = rows[0]
    _require(row["measurement_missing"] == "0", "reference retention measurement is missing")
    affibody_sequence = row["chain2_affibody_sequence"]
    chain1_sequence = row["chain1_smart_hla_linker_peptide_sequence"]
    _require(len(affibody_sequence) == 58, "expected a 58-residue xx133 sequence")
    _require(len(chain1_sequence) >= EXPECTED_PEPTIDE_LENGTH, "chain 1 is too short")
    peptide_sequence = chain1_sequence[-EXPECTED_PEPTIDE_LENGTH:]
    _require(
        "".join(affibody_sequence[position - 1] for position in AFFIBODY_DESIGN_POSITIONS)
        == EXPECTED_AFFIBODY_CODE,
        "NNYYF does not map to Affibody positions 6/10/13/14/17",
    )
    _require(
        "".join(peptide_sequence[position - 1] for position in PEPTIDE_DESIGN_POSITIONS)
        == EXPECTED_PEPTIDE_CODE,
        "MW does not map to peptide positions 4/5",
    )
    for position, expected in zip(AFFIBODY_DESIGN_POSITIONS, EXPECTED_AFFIBODY_CODE):
        _require(row["aff_p{}".format(position)] == expected, "Affibody row mapping mismatch")
    for position, expected in zip(PEPTIDE_DESIGN_POSITIONS, EXPECTED_PEPTIDE_CODE):
        _require(row["pep_p{}".format(position)] == expected, "peptide row mapping mismatch")
    try:
        retention = float(row["retention_percent"])
    except ValueError as exc:
        raise ValueError("reference retention is not numeric") from exc
    _require(math.isfinite(retention), "reference retention is non-finite")
    return ReferencePair(
        pair_uid=row["pair_uid"],
        peptide_sequence=peptide_sequence,
        affibody_sequence=affibody_sequence,
        retention_percent=retention,
    )


def map_contiguous_subsequence(chain, expected_sequence):
    observed = chain.sequence
    start = expected_sequence.find(observed)
    _require(start >= 0, "chain {} does not map to expected sequence".format(chain.chain_id))
    _require(
        expected_sequence.rfind(observed) == start,
        "chain {} maps ambiguously to expected sequence".format(chain.chain_id),
    )
    return {
        id(residue): start + ordinal + 1
        for ordinal, residue in enumerate(chain.residues)
    }


def classify_chains(chains, reference):
    roles = {}
    mappings = {}
    for chain_id, chain in chains.items():
        sequence = chain.sequence
        if sequence == reference.peptide_sequence:
            roles[chain_id] = "NY-ESO-1 peptide"
            mappings[chain_id] = {
                id(residue): ordinal + 1 for ordinal, residue in enumerate(chain.residues)
            }
        elif len(sequence) >= 30 and sequence in reference.affibody_sequence:
            roles[chain_id] = "xx133 Affibody"
            mappings[chain_id] = map_contiguous_subsequence(chain, reference.affibody_sequence)
        elif len(sequence) >= 200 and sequence.startswith("GSHSMRY"):
            roles[chain_id] = "HLA-A*02 heavy chain"
        elif 90 <= len(sequence) <= 110 and sequence.startswith("IQRTPKIQVY"):
            roles[chain_id] = "beta-2-microglobulin"
        elif len(sequence) >= 100 and "VQPGG" in sequence[:30]:
            roles[chain_id] = "VHH-like chain"
        else:
            roles[chain_id] = "other protein"
    expected_counts = {
        "NY-ESO-1 peptide": 2,
        "xx133 Affibody": 2,
        "HLA-A*02 heavy chain": 2,
        "beta-2-microglobulin": 2,
        "VHH-like chain": 2,
    }
    for role, expected in expected_counts.items():
        observed = sum(value == role for value in roles.values())
        _require(observed == expected, "expected {} {} chains; found {}".format(expected, role, observed))
    return roles, mappings


def euclidean(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def closest_atoms(first_atoms, second_atoms):
    _require(first_atoms and second_atoms, "cannot calculate distance for an empty atom set")
    best = None
    for first in first_atoms:
        for second in second_atoms:
            distance = euclidean(first.xyz, second.xyz)
            candidate = (distance, first.name, second.name)
            if best is None or candidate < best:
                best = candidate
    return best


def chain_minimum_distance(first, second):
    return closest_atoms(first.atoms, second.atoms)[0]


def best_assignment(left_chains, right_chains):
    _require(len(left_chains) == len(right_chains), "assignment sides have different sizes")
    _require(left_chains, "cannot assign empty chain sets")
    possibilities = []
    for permutation in itertools.permutations(right_chains):
        assignments = list(zip(left_chains, permutation))
        total = sum(chain_minimum_distance(left, right) for left, right in assignments)
        possibilities.append((total, tuple(right.chain_id for right in permutation), assignments))
    possibilities.sort(key=lambda value: (value[0], value[1]))
    best = possibilities[0]
    runner_up = possibilities[1][0] if len(possibilities) > 1 else None
    return best[2], best[0], runner_up


def representative_atom(residue):
    atom_name = "CA" if residue.name1 == "G" else "CB"
    _require(
        atom_name in residue.atoms,
        "missing representative atom {} for {} {}".format(
            atom_name, residue.chain_id, residue.pdb_label
        ),
    )
    return residue.atoms[atom_name]


def _role_chains(chains, roles, role):
    return [chains[chain_id] for chain_id in sorted(chains) if roles[chain_id] == role]


def build_crystal_copies(chains, roles):
    peptides = _role_chains(chains, roles, "NY-ESO-1 peptide")
    affibodies = _role_chains(chains, roles, "xx133 Affibody")
    hlas = _role_chains(chains, roles, "HLA-A*02 heavy chain")
    beta2ms = _role_chains(chains, roles, "beta-2-microglobulin")
    vhhs = _role_chains(chains, roles, "VHH-like chain")

    peptide_affibody, target_total, target_runner_up = best_assignment(peptides, affibodies)
    peptide_hla, _, _ = best_assignment(peptides, hlas)
    hla_beta2m, _, _ = best_assignment(hlas, beta2ms)
    hla_vhh, _, _ = best_assignment(hlas, vhhs)
    affibody_by_peptide = {left.chain_id: right for left, right in peptide_affibody}
    hla_by_peptide = {left.chain_id: right for left, right in peptide_hla}
    beta2m_by_hla = {left.chain_id: right for left, right in hla_beta2m}
    vhh_by_hla = {left.chain_id: right for left, right in hla_vhh}

    _require(
        target_runner_up is not None and target_runner_up - target_total > 10.0,
        "peptide--Affibody copy assignment is not geometrically unambiguous",
    )
    copies = []
    for index, peptide in enumerate(peptides, 1):
        affibody = affibody_by_peptide[peptide.chain_id]
        hla = hla_by_peptide[peptide.chain_id]
        beta2m = beta2m_by_hla[hla.chain_id]
        vhh = vhh_by_hla[hla.chain_id]
        minimum = closest_atoms(peptide.atoms, affibody.atoms)
        _require(minimum[0] < 5.0, "assigned peptide--Affibody chains do not contact")
        copies.append(
            {
                "copy_id": "crystal_copy_{}".format(index),
                "hla_chain": hla.chain_id,
                "beta2m_chain": beta2m.chain_id,
                "peptide_chain": peptide.chain_id,
                "affibody_chain": affibody.chain_id,
                "vhh_like_chain": vhh.chain_id,
                "peptide_affibody_minimum_heavy_distance_angstrom": minimum[0],
                "peptide_closest_atom": minimum[1],
                "affibody_closest_atom": minimum[2],
                "assignment_total_distance_angstrom": target_total,
                "assignment_runner_up_total_distance_angstrom": target_runner_up,
            }
        )
    return copies


def residue_distance_record(copy, peptide_residue, affibody_residue, mappings, thresholds):
    peptide_atom = representative_atom(peptide_residue)
    affibody_atom = representative_atom(affibody_residue)
    representative_distance = euclidean(peptide_atom.xyz, affibody_atom.xyz)
    minimum_distance, peptide_closest, affibody_closest = closest_atoms(
        list(peptide_residue.atoms.values()), list(affibody_residue.atoms.values())
    )
    return {
        "copy_id": copy["copy_id"],
        "peptide_chain": peptide_residue.chain_id,
        "peptide_sequence_position": mappings[peptide_residue.chain_id][id(peptide_residue)],
        "peptide_pdb_residue": peptide_residue.pdb_label,
        "peptide_residue": peptide_residue.name1,
        "affibody_chain": affibody_residue.chain_id,
        "affibody_sequence_position": mappings[affibody_residue.chain_id][id(affibody_residue)],
        "affibody_pdb_residue": affibody_residue.pdb_label,
        "affibody_residue": affibody_residue.name1,
        "peptide_representative_atom": peptide_atom.name,
        "affibody_representative_atom": affibody_atom.name,
        "representative_distance_angstrom": representative_distance,
        "minimum_heavy_atom_distance_angstrom": minimum_distance,
        "peptide_closest_heavy_atom": peptide_closest,
        "affibody_closest_heavy_atom": affibody_closest,
        "representative_contact_lt_8a": int(
            representative_distance < thresholds["representative_contact_angstrom"]
        ),
        "minimum_heavy_contact_lt_5a": int(
            minimum_distance < thresholds["minimum_heavy_contact_angstrom"]
        ),
    }


def build_distance_records(copies, chains, mappings, thresholds):
    records = []
    for copy in copies:
        peptide = chains[copy["peptide_chain"]]
        affibody = chains[copy["affibody_chain"]]
        for peptide_residue in peptide.residues:
            for affibody_residue in affibody.residues:
                records.append(
                    residue_distance_record(
                        copy, peptide_residue, affibody_residue, mappings, thresholds
                    )
                )
    return records


def build_consensus(records, copies):
    copy_ids = [copy["copy_id"] for copy in copies]
    grouped = defaultdict(dict)
    for record in records:
        key = (
            record["peptide_sequence_position"],
            record["peptide_residue"],
            record["affibody_sequence_position"],
            record["affibody_residue"],
        )
        grouped[key][record["copy_id"]] = record
    output = []
    for key in sorted(grouped):
        peptide_position, peptide_residue, affibody_position, affibody_residue = key
        by_copy = grouped[key]
        representative = [
            by_copy[copy_id]["representative_distance_angstrom"]
            for copy_id in copy_ids
            if copy_id in by_copy
        ]
        minimum = [
            by_copy[copy_id]["minimum_heavy_atom_distance_angstrom"]
            for copy_id in copy_ids
            if copy_id in by_copy
        ]
        row = {
            "peptide_sequence_position": peptide_position,
            "peptide_residue": peptide_residue,
            "affibody_sequence_position": affibody_position,
            "affibody_residue": affibody_residue,
            "copies_observed": len(by_copy),
        }
        for copy_id in copy_ids:
            prefix = copy_id
            record = by_copy.get(copy_id)
            row[prefix + "_representative_distance_angstrom"] = (
                record["representative_distance_angstrom"] if record else None
            )
            row[prefix + "_minimum_heavy_atom_distance_angstrom"] = (
                record["minimum_heavy_atom_distance_angstrom"] if record else None
            )
            row[prefix + "_representative_contact_lt_8a"] = (
                record["representative_contact_lt_8a"] if record else None
            )
            row[prefix + "_minimum_heavy_contact_lt_5a"] = (
                record["minimum_heavy_contact_lt_5a"] if record else None
            )
        row.update(
            {
                "representative_distance_mean_angstrom": sum(representative)
                / len(representative),
                "representative_distance_range_angstrom": max(representative)
                - min(representative),
                "minimum_heavy_distance_mean_angstrom": sum(minimum) / len(minimum),
                "minimum_heavy_distance_range_angstrom": max(minimum) - min(minimum),
                "representative_contact_copies": sum(
                    by_copy[copy_id]["representative_contact_lt_8a"] for copy_id in by_copy
                ),
                "minimum_heavy_contact_copies": sum(
                    by_copy[copy_id]["minimum_heavy_contact_lt_5a"] for copy_id in by_copy
                ),
                "representative_contact_both_copies": int(
                    len(by_copy) == len(copy_ids)
                    and all(
                        by_copy[copy_id]["representative_contact_lt_8a"]
                        for copy_id in copy_ids
                    )
                ),
                "minimum_heavy_contact_both_copies": int(
                    len(by_copy) == len(copy_ids)
                    and all(
                        by_copy[copy_id]["minimum_heavy_contact_lt_5a"]
                        for copy_id in copy_ids
                    )
                ),
            }
        )
        output.append(row)
    return output


def chain_inventory(chains, roles, mappings, reference):
    rows = []
    for chain_id in sorted(chains):
        chain = chains[chain_id]
        mapped_positions = sorted(mappings.get(chain_id, {}).values())
        expected_length = None
        if roles[chain_id] == "NY-ESO-1 peptide":
            expected_length = len(reference.peptide_sequence)
        elif roles[chain_id] == "xx133 Affibody":
            expected_length = len(reference.affibody_sequence)
        missing = (
            ";".join(
                str(position)
                for position in range(1, expected_length + 1)
                if position not in set(mapped_positions)
            )
            if expected_length is not None
            else ""
        )
        rows.append(
            {
                "chain_id": chain_id,
                "role": roles[chain_id],
                "coordinate_residue_count": len(chain.residues),
                "coordinate_heavy_atom_count": len(chain.atoms),
                "first_pdb_residue": chain.residues[0].pdb_label,
                "last_pdb_residue": chain.residues[-1].pdb_label,
                "observed_sequence": chain.sequence,
                "mapped_expected_sequence_start": min(mapped_positions) if mapped_positions else None,
                "mapped_expected_sequence_end": max(mapped_positions) if mapped_positions else None,
                "missing_expected_sequence_positions": missing,
            }
        )
    return rows


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{:.4f}".format(value)
    return value


def write_csv(path, rows):
    _require(rows, "refusing to write an empty table: {}".format(path.name))
    fields = list(rows[0])
    with open(path, "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            _require(list(row) == fields, "inconsistent table schema for {}".format(path.name))
            writer.writerow({field: _csv_value(row[field]) for field in fields})
    os.chmod(path, 0o600)


def build_report(reference, resolution, chains, roles, copies, designed, thresholds):
    lines = [
        "# NY-ESO-1–xx133 observed crystal interface",
        "",
        "## Reference identity",
        "",
        "The coordinate file maps exactly to the LibB `NNYYF` Affibody and `MW` "
        "peptide design. The matching retention measurement is {:.2f}%. Designed "
        "peptide positions 4/5 are M/W, and designed Affibody positions "
        "6/10/13/14/17 are N/N/Y/Y/F.".format(reference.retention_percent),
        "",
        "The structure has two crystallographic copies. Their target interfaces are "
        "peptide `{}` with Affibody `{}`, and peptide `{}` with Affibody `{}`. The "
        "HLA/β2m chains are `{}`/`{}` and `{}`/`{}`, respectively.".format(
            copies[0]["peptide_chain"],
            copies[0]["affibody_chain"],
            copies[1]["peptide_chain"],
            copies[1]["affibody_chain"],
            copies[0]["hla_chain"],
            copies[0]["beta2m_chain"],
            copies[1]["hla_chain"],
            copies[1]["beta2m_chain"],
        ),
        "",
        "The refinement header reports {:.2f} Å resolution.".format(resolution)
        if resolution is not None
        else "No refinement resolution was found in the coordinate header.",
        "",
        "## Designed interface distances",
        "",
        "`Representative` means Cβ–Cβ (Cα for glycine), matching the usual "
        "residue-level distance convention. `Closest heavy atoms` is the shortest "
        "non-hydrogen atom distance. Contacts use <{:.1f} Å and <{:.1f} Å, "
        "respectively.".format(
            thresholds["representative_contact_angstrom"],
            thresholds["minimum_heavy_contact_angstrom"],
        ),
        "",
        "| Peptide | Affibody | Representative distance, copies 1/2 (Å) | "
        "Closest-heavy distance, copies 1/2 (Å) | Contact in both copies? |",
        "|---|---|---:|---:|---|",
    ]
    for row in designed:
        representative_both = bool(row["representative_contact_both_copies"])
        heavy_both = bool(row["minimum_heavy_contact_both_copies"])
        contact = "representative and heavy" if representative_both and heavy_both else (
            "closest-heavy only" if heavy_both else "no"
        )
        lines.append(
            "| {}{} | {}{} | {:.2f} / {:.2f} | {:.2f} / {:.2f} | {} |".format(
                row["peptide_residue"],
                row["peptide_sequence_position"],
                row["affibody_residue"],
                row["affibody_sequence_position"],
                row["crystal_copy_1_representative_distance_angstrom"],
                row["crystal_copy_2_representative_distance_angstrom"],
                row["crystal_copy_1_minimum_heavy_atom_distance_angstrom"],
                row["crystal_copy_2_minimum_heavy_atom_distance_angstrom"],
                contact,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The two copies agree closely at the designed interface. This makes the "
            "structure suitable for defining an interface mask and for checking future "
            "AlphaFold predictions against observed contacts.",
            "",
            "These values are observed distances from one reference crystal structure; "
            "they are **not an AlphaFold distogram**, a binding probability, or a "
            "variant-specific prediction. The two crystal copies are repeated views in "
            "one asymmetric unit, not two independent experiments. A static contact map "
            "cannot by itself rank the other LibB sequence variants.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, type=Path)
    parser.add_argument("--retention-sequences", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representative-contact-angstrom", type=float, default=8.0)
    parser.add_argument("--minimum-heavy-contact-angstrom", type=float, default=5.0)
    return parser.parse_args(argv)


def run(args):
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    output_dir = validate_private_output_path(args.output_dir, repo_root)
    _require(args.pdb.is_file(), "PDB input does not exist")
    _require(args.retention_sequences.is_file(), "retention sequence table does not exist")
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    thresholds = {
        "representative_contact_angstrom": float(args.representative_contact_angstrom),
        "minimum_heavy_contact_angstrom": float(args.minimum_heavy_contact_angstrom),
    }
    _require(
        all(math.isfinite(value) and value > 0 for value in thresholds.values()),
        "contact thresholds must be positive and finite",
    )

    reference = load_reference_pair(args.retention_sequences)
    chains = parse_pdb(args.pdb)
    roles, mappings = classify_chains(chains, reference)
    copies = build_crystal_copies(chains, roles)
    records = build_distance_records(copies, chains, mappings, thresholds)
    consensus = build_consensus(records, copies)
    designed = [
        row
        for row in consensus
        if row["peptide_sequence_position"] in PEPTIDE_DESIGN_POSITIONS
        and row["affibody_sequence_position"] in AFFIBODY_DESIGN_POSITIONS
    ]
    _require(len(designed) == 10, "designed 2x5 interface map is incomplete")
    _require(all(row["copies_observed"] == 2 for row in designed), "designed map lacks a copy")
    inventory = chain_inventory(chains, roles, mappings, reference)
    resolution = parse_resolution(args.pdb)

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".{}-".format(output_dir.name), dir=str(parent)))
    os.chmod(temporary, 0o700)
    try:
        table_specs = [
            ("chain_inventory.csv", inventory),
            ("biological_copies.csv", copies),
            ("residue_pair_distances.csv", records),
            ("observed_contact_map.csv", consensus),
            ("designed_interface_distances.csv", designed),
        ]
        for filename, rows in table_specs:
            write_csv(temporary / filename, rows)
        report = build_report(reference, resolution, chains, roles, copies, designed, thresholds)
        report_path = temporary / "run_summary.md"
        with open(report_path, "x", encoding="utf-8") as handle:
            handle.write(report)
        os.chmod(report_path, 0o600)

        output_entries = {}
        for filename, rows in table_specs:
            output_entries[filename] = {
                "rows": len(rows),
                "sha256": sha256_file(temporary / filename),
            }
        output_entries["run_summary.md"] = {"sha256": sha256_file(report_path)}
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "analysis_type": "observed_crystal_coordinate_interface_distances",
            "explicitly_not": [
                "AlphaFold distogram",
                "binding probability",
                "variant-specific structure prediction",
            ],
            "inputs": {
                "pdb": {"path": str(args.pdb.resolve()), "sha256": sha256_file(args.pdb)},
                "retention_sequences": {
                    "path": str(args.retention_sequences.resolve()),
                    "sha256": sha256_file(args.retention_sequences),
                },
                "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
            },
            "reference_match": {
                "library": EXPECTED_LIBRARY,
                "affibody_design_code": EXPECTED_AFFIBODY_CODE,
                "peptide_design_code": EXPECTED_PEPTIDE_CODE,
                "pair_uid": reference.pair_uid,
                "retention_percent": reference.retention_percent,
                "peptide_sequence_sha256": hashlib.sha256(
                    reference.peptide_sequence.encode("ascii")
                ).hexdigest(),
                "affibody_sequence_sha256": hashlib.sha256(
                    reference.affibody_sequence.encode("ascii")
                ).hexdigest(),
            },
            "structure": {
                "resolution_angstrom": resolution,
                "chain_count": len(chains),
                "crystal_copy_count": len(copies),
                "chain_roles": roles,
            },
            "configuration": thresholds,
            "outputs": output_entries,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
        }
        manifest_path = temporary / "manifest.json"
        with open(manifest_path, "x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(manifest_path, 0o600)
        os.replace(str(temporary), str(output_dir))
    except Exception:
        # Leave the private temporary directory intact for debugging. It is
        # unambiguously scoped and never replaces an existing artifact.
        raise
    return manifest


def main(argv=None):
    args = parse_args(argv)
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
