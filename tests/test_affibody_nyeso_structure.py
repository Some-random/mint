import csv
from pathlib import Path

import pytest

from downstream.AffibodyMHC.analyze_nyeso_xx133_structure import (
    Atom,
    Chain,
    Residue,
    best_assignment,
    closest_atoms,
    load_reference_pair,
    map_contiguous_subsequence,
    parse_pdb,
    representative_atom,
)


def _atom(name, xyz, occupancy=1.0, altloc=" ", element="C"):
    return Atom(name, altloc, xyz[0], xyz[1], xyz[2], occupancy, element)


def _residue(chain, number, name3, xyz):
    return Residue(
        chain,
        number,
        " ",
        name3,
        {"CA": _atom("CA", xyz), "CB": _atom("CB", xyz)},
    )


def _pdb_atom(serial, atom, altloc, residue, chain, number, xyz, occupancy):
    element = atom.strip()[0]
    return (
        "ATOM  {serial:5d} {atom:^4s}{altloc:1s}{residue:>3s} {chain:1s}"
        "{number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}"
        "          {element:>2s}\n"
    ).format(
        serial=serial,
        atom=atom,
        altloc=altloc,
        residue=residue,
        chain=chain,
        number=number,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        occupancy=occupancy,
        bfactor=20.0,
        element=element,
    )


def test_parse_pdb_selects_highest_occupancy_altloc(tmp_path):
    path = tmp_path / "altloc.pdb"
    path.write_text(
        _pdb_atom(1, "CA", "A", "ALA", "A", 1, (1.0, 0.0, 0.0), 0.4)
        + _pdb_atom(2, "CA", "B", "ALA", "A", 1, (2.0, 0.0, 0.0), 0.6)
        + _pdb_atom(3, "CB", " ", "ALA", "A", 1, (0.0, 0.0, 0.0), 1.0),
        encoding="ascii",
    )

    chains = parse_pdb(path)

    assert chains["A"].residues[0].atoms["CA"].altloc == "B"
    assert chains["A"].residues[0].atoms["CA"].x == 2.0


def test_contiguous_mapping_handles_unresolved_affibody_termini():
    full = "VDNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"
    observed = full[2:-1]
    residues = [
        _residue("H", pdb_number, "ASN" if value == "N" else "ALA", (index, 0, 0))
        for index, (pdb_number, value) in enumerate(zip(range(5, 60), observed))
    ]
    # Override the synthetic residue names to preserve the intended sequence.
    reverse = {value: key for key, value in {
        "ALA": "A", "ASP": "D", "GLU": "E", "PHE": "F", "HIS": "H",
        "ILE": "I", "LYS": "K", "LEU": "L", "ASN": "N", "PRO": "P",
        "GLN": "Q", "SER": "S", "VAL": "V", "TYR": "Y",
    }.items()}
    for residue, value in zip(residues, observed):
        residue.name3 = reverse[value]
    chain = Chain("H", residues)

    mapping = map_contiguous_subsequence(chain, full)

    assert mapping[id(chain.residues[0])] == 3
    assert mapping[id(chain.residues[-1])] == 57
    assert chain.residues[3].number == 8
    assert mapping[id(chain.residues[3])] == 6


def test_representative_atom_uses_ca_for_glycine_and_cb_otherwise():
    glycine = Residue("P", 1, " ", "GLY", {"CA": _atom("CA", (0, 0, 0))})
    alanine = _residue("P", 2, "ALA", (1, 0, 0))

    assert representative_atom(glycine).name == "CA"
    assert representative_atom(alanine).name == "CB"


def test_closest_atoms_and_assignment_recover_two_nearby_copies():
    peptides = [
        Chain("E", [_residue("E", 1, "ALA", (0, 0, 0))]),
        Chain("P", [_residue("P", 1, "ALA", (100, 0, 0))]),
    ]
    affibodies = [
        Chain("H", [_residue("H", 1, "ALA", (101, 0, 0))]),
        Chain("I", [_residue("I", 1, "ALA", (1, 0, 0))]),
    ]

    assignments, total, runner_up = best_assignment(peptides, affibodies)

    assert [(left.chain_id, right.chain_id) for left, right in assignments] == [
        ("E", "I"),
        ("P", "H"),
    ]
    assert total == pytest.approx(2.0)
    assert runner_up == pytest.approx(200.0)
    assert closest_atoms(peptides[0].atoms, affibodies[1].atoms)[0] == pytest.approx(1.0)


def test_reference_loader_requires_and_maps_exact_nnyyf_mw_row(tmp_path):
    path = tmp_path / "retention.csv"
    fields = [
        "library", "affibody_design_code", "peptide_design_code", "retention_percent",
        "measurement_missing", "pair_uid", "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence", "aff_p6", "aff_p10", "aff_p13", "aff_p14",
        "aff_p17", "pep_p4", "pep_p5",
    ]
    row = {
        "library": "LibB",
        "affibody_design_code": "NNYYF",
        "peptide_design_code": "MW",
        "retention_percent": "83.77",
        "measurement_missing": "0",
        "pair_uid": "test-pair",
        "chain1_smart_hla_linker_peptide_sequence": "PREFIXSLLMWITQV",
        "chain2_affibody_sequence": "VDNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK",
        "aff_p6": "N", "aff_p10": "N", "aff_p13": "Y", "aff_p14": "Y",
        "aff_p17": "F", "pep_p4": "M", "pep_p5": "W",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    reference = load_reference_pair(path)

    assert reference.peptide_sequence == "SLLMWITQV"
    assert reference.retention_percent == pytest.approx(83.77)
    assert "".join(reference.affibody_sequence[index - 1] for index in (6, 10, 13, 14, 17)) == "NNYYF"
