import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from downstream.AffibodyMHC import build_openfold3_native_panel as builder
from downstream.AffibodyMHC import build_openfold3_artifact_manifest as artifact_manifest
from downstream.AffibodyMHC.analyze_openfold3_native_panel import (
    DESIGNED_AFFIBODY_POSITIONS,
    DESIGNED_PEPTIDE_POSITIONS,
    PRIMARY_CRYSTAL_PAIRS,
    select_crystal_reference_job,
)
from downstream.AffibodyMHC.openfold3_distogram_utils import (
    distogram_matrices,
    pair_tensor,
    token_lookup_from_batch,
    validate_query_document,
)
from downstream.AffibodyMHC.run_openfold3_job_shard import build_command


AFFIBODY = "VDNKFNKEFNNAYYEIFHLPNLNEEQFDAFVQSLFDDPSQSANLLAEAKKLNDAQAPK"
CODES = ("MVKNT", "FALTA", "NNYYF", "LIFTK", "ANTKV", "EFYSV", "ATETI", "TDIDA", "VAKST", "NMDKV")
POSITIONS = (6, 10, 13, 14, 17)


def mutate_affibody(code):
    values = list(AFFIBODY)
    for position, amino_acid in zip(POSITIONS, code):
        values[position - 1] = amino_acid
    return "".join(values)


def test_distogram_math_uses_exact_64_bins_and_8A_definition():
    logits = np.zeros((2, 2, 64), dtype=np.float32)

    probabilities, contact, expected = distogram_matrices(logits)

    assert probabilities.shape == (2, 2, 64)
    assert probabilities[0, 0].sum() == pytest.approx(1.0)
    assert contact[0, 0] == pytest.approx(19.0 / 64.0)
    assert expected[0, 0] == pytest.approx(12.0)


def test_token_mapping_and_pair_extraction_are_chain_position_explicit():
    class AtomArray:
        chain_id = np.asarray(["P", "P", "H", "H"])

    batch = {
        "residue_index": np.asarray([[4, 5, 6, 10]]),
        "start_atom_index": np.asarray([[0, 1, 2, 3]]),
        "atom_array": [AtomArray()],
    }
    lookup, chains, positions = token_lookup_from_batch(batch)
    matrix = np.arange(16).reshape(4, 4)

    extracted = pair_tensor(matrix, lookup, "P", (4, 5), "H", (6, 10))

    assert lookup == {("P", 4): 0, ("P", 5): 1, ("H", 6): 2, ("H", 10): 3}
    assert chains.tolist() == ["P", "P", "H", "H"]
    assert positions.tolist() == [4, 5, 6, 10]
    assert extracted.tolist() == [[2, 3], [6, 7]]


def test_analysis_mask_is_frozen_to_three_crystal_contacts():
    assert PRIMARY_CRYSTAL_PAIRS == ((4, 14), (4, 17), (5, 10))
    assert DESIGNED_PEPTIDE_POSITIONS == (4, 5)
    assert DESIGNED_AFFIBODY_POSITIONS == (6, 10, 13, 14, 17)


def test_crystal_reference_job_is_selected_by_identity_not_manifest_order():
    jobs = [
        {
            "affibody_design_code": "MVKNT",
            "duplicate_control": "0",
            "replicate": "1",
        },
        {
            "affibody_design_code": "NNYYF",
            "duplicate_control": "0",
            "replicate": "1",
        },
        {
            "affibody_design_code": "NNYYF",
            "duplicate_control": "1",
            "replicate": "1",
        },
    ]

    assert select_crystal_reference_job(jobs) is jobs[1]


def test_query_validation_rejects_paired_msa_or_template():
    chains = [
        {
            "molecule_type": "protein",
            "chain_ids": [chain_id],
            "sequence": "A",
            "main_msa_file_paths": [f"/{chain_id}.npz"],
        }
        for chain_id in ("A", "B", "P", "H")
    ]
    document = {
        "seeds": [42],
        "queries": {
            "q": {
                "use_main_msas": True,
                "use_paired_msas": False,
                "chains": chains,
            }
        },
    }
    validate_query_document(document, 42)
    document["queries"]["q"]["chains"][0]["template_alignment_file_path"] = "/t.m8"
    with pytest.raises(ValueError, match="paired/template"):
        validate_query_document(document, 42)


def test_launcher_passes_global_ref_and_model_seeds_separately(tmp_path):
    job = {
        "global_feature_seed": "20260820",
        "ref_conformer_seed": "43",
        "model_seed": "42",
        "query_json": "/query.json",
        "runner_yaml": "/runner.yml",
        "output_dir": "/output",
    }
    command = build_command(job, "/python", tmp_path / "launcher.py")

    assert command[command.index("--global-feature-seed") + 1] == "20260820"
    assert command[command.index("--ref-conformer-seed") + 1] == "43"
    assert "42" not in command  # Model seed is carried only by query/runner files.
    assert command[-6:] == [
        "--use-msa-server",
        "false",
        "--use-templates",
        "false",
        "--output-dir",
        "/output",
    ]


def test_builder_emits_10_by_3_paired_jobs_duplicates_and_private_modes(
    tmp_path, monkeypatch
):
    msa_paths = {}
    for chain_id in ("A", "B", "P", "H"):
        path = tmp_path / f"{chain_id}.a3m"
        path.write_text(">101\nA\n")
        msa_paths[chain_id] = str(path)
    reference = {
        "queries": {
            "reference": {
                "use_msas": True,
                "use_main_msas": True,
                "use_paired_msas": False,
                "chains": [
                    {"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AAAA", "main_msa_file_paths": [msa_paths["A"]]},
                    {"molecule_type": "protein", "chain_ids": ["B"], "sequence": "BBBB", "main_msa_file_paths": [msa_paths["B"]]},
                    {"molecule_type": "protein", "chain_ids": ["P"], "sequence": "SLLMWITQV", "main_msa_file_paths": [msa_paths["P"]]},
                    {"molecule_type": "protein", "chain_ids": ["H"], "sequence": AFFIBODY, "main_msa_file_paths": [msa_paths["H"]]},
                ],
            }
        }
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    source_path = tmp_path / "retention.csv"
    fields = [
        "library",
        "peptide_design_code",
        "measurement_missing",
        "affibody_design_code",
        "chain2_affibody_sequence",
        "retention_percent",
        "pair_uid",
        "sequence_pair_sha256",
    ]
    with source_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, code in enumerate(CODES):
            writer.writerow(
                {
                    "library": "LibB",
                    "peptide_design_code": "MW",
                    "measurement_missing": "0",
                    "affibody_design_code": code,
                    "chain2_affibody_sequence": mutate_affibody(code),
                    "retention_percent": str(index),
                    "pair_uid": f"pair-{index}",
                    "sequence_pair_sha256": f"hash-{index}",
                }
            )
    source_path.chmod(0o664)
    reference_path.chmod(0o664)
    output = tmp_path / "existing_permissive_output"
    output.mkdir(mode=0o777)
    output.chmod(0o777)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--source-csv",
            str(source_path),
            "--reference-json",
            str(reference_path),
            "--output-root",
            str(output),
            "--replicate-seeds",
            "42",
            "43",
            "2746317213",
            "--global-feature-seed",
            "20260820",
        ],
    )

    builder.main()

    with (output / "job_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 33
    assert sum(row["duplicate_control"] == "0" for row in rows) == 30
    assert sum(row["duplicate_control"] == "1" for row in rows) == 3
    assert {
        (row["replicate"], row["ref_conformer_seed"], row["model_seed"])
        for row in rows
    } == {
        ("1", "42", "42"),
        ("2", "43", "43"),
        ("3", "2746317213", "2746317213"),
    }
    assert {row["global_feature_seed"] for row in rows} == {"20260820"}
    assert len({row["affibody_design_code"] for row in rows}) == 10
    for path in [output] + list(output.rglob("*")):
        expected = 0o700 if path.is_dir() else 0o600
        assert path.stat().st_mode & 0o777 == expected


def test_artifact_manifest_excludes_itself_and_refuses_overwrite(tmp_path):
    (tmp_path / "raw.pt").write_bytes(b"raw")
    manifest = tmp_path / "final_artifact_manifest.json"
    manifest.write_text("old")

    assert [path.name for path in artifact_manifest.iter_files(tmp_path)] == ["raw.pt"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        artifact_manifest.ensure_manifest_fresh(manifest)
    artifact_manifest.ensure_manifest_fresh(tmp_path / "new_manifest.json")


def test_exact_manifest_binds_transitive_builder_and_records_launcher_caveat(tmp_path):
    paths = artifact_manifest.project_code_paths(tmp_path, "exact_construct")

    assert tmp_path / "downstream/AffibodyMHC/build_openfold3_native_panel.py" in paths
    assert tmp_path / "downstream/AffibodyMHC/build_openfold3_artifact_manifest.py" in paths
    assert "execution-time source hash was not captured" in (
        artifact_manifest.source_capture_limitations("exact_construct")[0]
    )
    assert artifact_manifest.source_capture_limitations("native_panel") == []


def test_manifest_runtime_identity_validation_checks_checkpoint_and_patch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    original = tmp_path / "original.py"
    patched = tmp_path / "patched.py"
    deployed = tmp_path / "deployed.py"
    checkpoint.write_bytes(b"checkpoint")
    original.write_bytes(b"original")
    patched.write_bytes(b"patched")
    deployed.write_bytes(b"patched")
    sha = artifact_manifest.sha256_file

    artifact_manifest.validate_runtime_identities(
        checkpoint,
        original,
        patched,
        deployed,
        checkpoint_sha256=sha(checkpoint),
        checkpoint_size=checkpoint.stat().st_size,
        original_sha256=sha(original),
        patched_sha256=sha(patched),
    )
    with pytest.raises(ValueError, match="Unexpected checkpoint identity"):
        artifact_manifest.validate_runtime_identities(
            checkpoint,
            original,
            patched,
            deployed,
            checkpoint_sha256="0" * 64,
            checkpoint_size=checkpoint.stat().st_size,
            original_sha256=sha(original),
            patched_sha256=sha(patched),
        )
    deployed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Deployed conformer"):
        artifact_manifest.validate_runtime_identities(
            checkpoint,
            original,
            patched,
            deployed,
            checkpoint_sha256=sha(checkpoint),
            checkpoint_size=checkpoint.stat().st_size,
            original_sha256=sha(original),
            patched_sha256=sha(patched),
        )


def test_manifest_rejects_permissive_private_file_and_directory_modes(tmp_path):
    private_root = tmp_path / "private"
    private_root.mkdir()
    private_root.chmod(0o775)
    entries = [
        {
            "category": "experiment_tree",
            "path": "private/raw.pt",
            "mode": "0664",
        }
    ]

    file_violations, directory_violations = artifact_manifest.audit_private_modes(
        entries, {"experiment_tree"}, [private_root]
    )
    assert file_violations
    assert directory_violations
    with pytest.raises(ValueError, match="Private permission violations"):
        artifact_manifest.reject_private_mode_violations(
            file_violations, directory_violations
        )

    private_root.chmod(0o700)
    entries[0]["mode"] = "0600"
    file_violations, directory_violations = artifact_manifest.audit_private_modes(
        entries, {"experiment_tree"}, [private_root]
    )
    artifact_manifest.reject_private_mode_violations(
        file_violations, directory_violations
    )
