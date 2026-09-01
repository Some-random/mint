#!/usr/bin/env python3
"""Build a deterministic, private manifest for a canonical OpenFold3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = (
    "af09eac4f29cef856633af07558cb143226fe95ebbef2c20921769d4a5f4bee4"
)
EXPECTED_CHECKPOINT_SIZE = 2_287_928_196
EXPECTED_ORIGINAL_CONFORMER_SHA256 = (
    "63c76f07515c928d1175bc62f85ee37d080a0b8f30166c4762f7dc496062af1e"
)
EXPECTED_PATCHED_CONFORMER_SHA256 = (
    "f15352833b578f7f26f3f4a628dcac9c835e93b08519ed4fd8ed170081a7f7df"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(workspace.resolve()))
    except ValueError:
        return str(resolved)


def iter_files(root: Path, *, installed_package: bool = False):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if not path.is_file():
            continue
        if installed_package and (
            "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.name.startswith("final_artifact_manifest"):
            continue
        yield path


def digest_entries(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: (item["category"], item["path"])):
        record = (
            f"{entry['category']}\0{entry['path']}\0{entry['size_bytes']}\0"
            f"{entry['mode']}\0{entry['sha256']}\n"
        )
        digest.update(record.encode())
    return digest.hexdigest()


def ensure_manifest_fresh(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")


def project_code_paths(workspace: Path, role: str) -> list[Path]:
    common = [
        workspace / "downstream/AffibodyMHC/analyze_nyeso_xx133_structure.py",
        workspace / "downstream/AffibodyMHC/openfold3_distogram_utils.py",
        workspace / "downstream/AffibodyMHC/run_openfold3_seeded.py",
        workspace / "downstream/AffibodyMHC/run_openfold3_job_shard.py",
        workspace / "downstream/AffibodyMHC/build_openfold3_artifact_manifest.py",
        workspace / "tests/test_affibody_openfold3_distogram.py",
        workspace / "tests/test_affibody_nyeso_structure.py",
    ]
    role_specific = {
        "native_panel": [
            workspace / "downstream/AffibodyMHC/build_openfold3_native_panel.py",
            workspace / "downstream/AffibodyMHC/build_openfold3_rng_controls.py",
            workspace / "downstream/AffibodyMHC/audit_openfold3_panel_inputs.py",
            workspace / "downstream/AffibodyMHC/analyze_openfold3_native_panel.py",
            workspace / "downstream/AffibodyMHC/analyze_openfold3_reference_calibration.py",
        ],
        "exact_construct": [
            # The exact builder imports RUNNER_TEMPLATE and helpers from this file.
            workspace / "downstream/AffibodyMHC/build_openfold3_native_panel.py",
            workspace / "downstream/AffibodyMHC/build_openfold3_exact_bridge_replicates.py",
            workspace / "downstream/AffibodyMHC/analyze_openfold3_exact_bridge.py",
        ],
    }
    return common + role_specific[role]


def source_capture_limitations(role: str) -> list[str]:
    if role != "exact_construct":
        return []
    return [
        "The current run_openfold3_job_shard.py is hashed as orchestration context, but its mtime is after the exact-construct outputs. Its execution-time source hash was not captured. Frozen query/config/status records preserve the exact command, exit status, runtime package, runner, and raw outputs."
    ]


def validate_runtime_identities(
    checkpoint: Path,
    original_snapshot: Path,
    patched_snapshot: Path,
    deployed_conformer: Path,
    *,
    checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
    checkpoint_size: int = EXPECTED_CHECKPOINT_SIZE,
    original_sha256: str = EXPECTED_ORIGINAL_CONFORMER_SHA256,
    patched_sha256: str = EXPECTED_PATCHED_CONFORMER_SHA256,
    observed_checkpoint_sha256: str | None = None,
    observed_checkpoint_size: int | None = None,
) -> None:
    observed_size = (
        checkpoint.stat().st_size
        if observed_checkpoint_size is None
        else observed_checkpoint_size
    )
    observed_sha = (
        sha256_file(checkpoint)
        if observed_checkpoint_sha256 is None
        else observed_checkpoint_sha256
    )
    if observed_size != checkpoint_size or observed_sha != checkpoint_sha256:
        raise ValueError("Unexpected checkpoint identity")
    if sha256_file(original_snapshot) != original_sha256:
        raise ValueError("Original conformer snapshot hash changed")
    if sha256_file(patched_snapshot) != patched_sha256:
        raise ValueError("Patched conformer snapshot hash changed")
    if sha256_file(deployed_conformer) != patched_sha256:
        raise ValueError("Deployed conformer file does not match patched snapshot")


def audit_private_modes(
    entries: list[dict], private_categories: set[str], directory_roots: list[Path]
) -> tuple[list[dict], list[dict]]:
    file_violations = [
        {"path": entry["path"], "mode": entry["mode"]}
        for entry in entries
        if entry["category"] in private_categories
        and int(entry["mode"], 8) & 0o077
    ]
    directory_violations = []
    for root in directory_roots:
        directories = [root] + [path for path in root.rglob("*") if path.is_dir()]
        for directory in directories:
            mode = stat.S_IMODE(directory.stat().st_mode)
            if mode & 0o077:
                directory_violations.append(
                    {"path": str(directory.resolve()), "mode": f"{mode:04o}"}
                )
    return file_violations, directory_violations


def reject_private_mode_violations(
    file_violations: list[dict], directory_violations: list[dict]
) -> None:
    if file_violations or directory_violations:
        raise ValueError(
            "Private permission violations: "
            f"files={file_violations[:10]}, directories={directory_violations[:10]}"
        )


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("native_panel", "exact_construct"), required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--installed-package-root", type=Path, required=True)
    parser.add_argument("--deployed-conformer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    ensure_manifest_fresh(args.manifest_path)
    patch_root = (
        workspace
        / "private_data/structure_pilot/openfold3_pilot/of3_source_patch"
    )
    msa_roots = {
        "native_panel": [
            workspace
            / "private_data/structure_pilot/openfold3_pilot/msa_server_reference"
        ],
        "exact_construct": [
            workspace
            / "private_data/structure_pilot/openfold3_pilot/msa_server_exact_construct_mw",
            workspace
            / "private_data/structure_pilot/openfold3_pilot/msa_server_reference/query_only/libb_mw/NNYYF",
        ],
    }
    external_inputs = {
        "native_panel": [workspace / "nyeso_xx133_complex.pdb"],
        "exact_construct": [
            workspace
            / "private_data/structure_pilot/openfold3_pilot/exact_construct_mw_nnyyf_bridge.json"
        ],
    }

    sources = [
        ("experiment_tree", args.experiment_root, False),
        *(("consumed_msa", root, False) for root in msa_roots[args.role]),
        *(("project_code", path, False) for path in project_code_paths(workspace, args.role)),
        *(("runtime_patch_provenance", path, False) for path in [patch_root]),
        *(("external_input", path, False) for path in external_inputs[args.role]),
        ("installed_openfold3_tree", args.installed_package_root, True),
        ("deployed_patched_conformer", args.deployed_conformer, False),
        ("model_checkpoint", args.checkpoint, False),
    ]
    seen: dict[str, str] = {}
    entries = []
    for category, source, installed_package in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        for path in iter_files(source, installed_package=installed_package):
            shown = display_path(path, workspace)
            existing = seen.get(shown)
            if existing is not None:
                if existing == category:
                    continue
                # Keep a single content record and list every provenance role.
                for entry in entries:
                    if entry["path"] == shown:
                        roles = set(entry.get("additional_categories", []))
                        roles.add(category)
                        entry["additional_categories"] = sorted(roles)
                        break
                continue
            mode = stat.S_IMODE(path.stat().st_mode)
            entry = {
                "category": category,
                "path": shown,
                "size_bytes": path.stat().st_size,
                "mode": f"{mode:04o}",
                "sha256": sha256_file(path),
            }
            entries.append(entry)
            seen[shown] = category

    checkpoint_entry = next(
        entry for entry in entries if entry["category"] == "model_checkpoint"
    )
    original_snapshot = patch_root / "original/conformer.py"
    patched_snapshot = patch_root / "patched/conformer.py"
    validate_runtime_identities(
        args.checkpoint,
        original_snapshot,
        patched_snapshot,
        args.deployed_conformer,
        observed_checkpoint_sha256=checkpoint_entry["sha256"],
        observed_checkpoint_size=checkpoint_entry["size_bytes"],
    )

    category_entries = defaultdict(list)
    for entry in entries:
        category_entries[entry["category"]].append(entry)
    private_categories = {
        "experiment_tree",
        "consumed_msa",
        "runtime_patch_provenance",
        "external_input",
    }
    private_directory_roots = [
        args.experiment_root,
        *msa_roots[args.role],
        patch_root,
    ]
    private_mode_violations, private_directory_mode_violations = audit_private_modes(
        entries, private_categories, private_directory_roots
    )
    result = {
        "schema": "affibody-openfold3-final-artifact-manifest-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "experiment_root": display_path(args.experiment_root, workspace),
        "manifest_excludes_itself": True,
        "runtime": {
            "distribution": "openfold3",
            "installed_version": "0.4.5",
            "python_version": "3.12.3",
            "pytorch_version": "2.9.1+cu128",
            "generation_and_canonical_analysis_environment": "/scratch/dongweij_of3_distogram_pilot_20260820/venv",
            "installed_package_root": str(args.installed_package_root.resolve()),
            "installed_wheel_direct_url_metadata": None,
            "pypi_wheel_filename": "openfold3-0.4.5-py3-none-any.whl",
            "pypi_wheel_url": "https://files.pythonhosted.org/packages/f6/b0/ddceb3486b666d36520d33929edf2beae181430985b104f4e681b0c9c3ad/openfold3-0.4.5-py3-none-any.whl",
            "pypi_wheel_sha256": "ccab10f4100ccccba26b80d8c640911a0140c6040ffc80f599619703de904ec2",
            "official_git_tag": "0.4.5",
            "official_git_tag_resolution": "a4b0803223f1c5048f8cbb0b6e9464da9d7724f0",
            "git_tag_resolution_note": "Independently verified upstream correspondence; the installed wheel contains no direct_url.json and does not embed this commit identity.",
            "original_conformer_sha256": EXPECTED_ORIGINAL_CONFORMER_SHA256,
            "patched_and_deployed_conformer_sha256": EXPECTED_PATCHED_CONFORMER_SHA256,
            "patch_activation_environment_variable": "OPENFOLD_REF_CONFORMER_SEED",
            "patch_scope": "Opt-in deterministic seeding of reference-conformer random augmentation; upstream behavior is unchanged when the environment variable is absent.",
        },
        "source_capture_limitations": source_capture_limitations(args.role),
        "checkpoint": {
            "name": "of3-p2-155k.pt",
            "path": checkpoint_entry["path"],
            "size_bytes": EXPECTED_CHECKPOINT_SIZE,
            "sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "n_files": len(entries),
        "total_size_bytes": sum(entry["size_bytes"] for entry in entries),
        "all_files_tree_sha256": digest_entries(entries),
        "category_summaries": {
            category: {
                "n_files": len(values),
                "total_size_bytes": sum(item["size_bytes"] for item in values),
                "tree_sha256": digest_entries(values),
            }
            for category, values in sorted(category_entries.items())
        },
        "private_mode_audit": {
            "expected": "private directories are 0700 and private files have no group/other permission bits",
            "pass": not private_mode_violations and not private_directory_mode_violations,
            "file_violations": private_mode_violations,
            "directory_violations": private_directory_mode_violations,
        },
        "files": sorted(entries, key=lambda item: (item["category"], item["path"])),
    }
    reject_private_mode_violations(
        private_mode_violations, private_directory_mode_violations
    )
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    args.manifest_path.chmod(0o600)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest_path.resolve()),
                "role": args.role,
                "n_files": result["n_files"],
                "total_size_bytes": result["total_size_bytes"],
                "all_files_tree_sha256": result["all_files_tree_sha256"],
                "private_mode_audit_pass": result["private_mode_audit"]["pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
