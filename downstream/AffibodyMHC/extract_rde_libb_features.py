#!/usr/bin/env python3
"""Extract absolute current-pair RDE-PPI representations for canonical LibB.

This is a label-free feature-extraction stage.  It reads only the canonical
sequence rows and the fixed LibB crystal, never the weak training labels or
direct-retention measurements.  The official RDE-PPI encoders are frozen.

Two feature families can be emitted:

``rde_context``
    The 128-dimensional per-residue context from the unsupervised official
    ``RDE.pt`` rotamer-density checkpoint.

``rde_network_fold{0,1,2}_context``
    The 128-dimensional per-residue contexts from the three official
    SKEMPI-trained ``DDG_RDE_Network_30k.pt`` folds.  They are deliberately
    kept separate because independently trained latent coordinate systems
    cannot be averaged elementwise.  Only each fold's ``encode`` method is
    used.  The original WT-minus-mutant subtraction and physical-delta-
    delta-G readout are not called.

The fixed 128-residue crystal patch is built once.  Every current pair changes
only the seven designed amino-acid identities, with crystallized side-chain
information masked at those sites.  Shards are independently resumable and
safe to run on separate GPUs.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.rde_libb_fixed_crystal import (
    AA_ALPHABET,
    AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED,
    ASSAY_HLA_CHAIN1_START_0_BASED,
    ASSAY_HLA_RESOLVED_LENGTH,
    AFFIBODY_CHAIN,
    BETA2M_CHAIN,
    CRYSTAL_CHAIN_IDS,
    FIXED_CRYSTAL_APPROXIMATION_OMISSIONS,
    HLA_CHAIN,
    HLA_ASSAY_IDENTITY_CORRECTIONS,
    PATCH_SIZE,
    PEPTIDE_CHAIN,
    build_fixed_crystal_template,
    collate_current_pairs,
    feature_masks,
    model_batch,
    prepare_current_pair,
    validate_canonical_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = REPO_ROOT / "private_data"
DEFAULT_STRUCTURE_CONTRACT_DIR = (
    PRIVATE_ROOT
    / "derived"
    / "libb_fixed_crystal_contract_provider_revision_120_v1"
)
STRUCTURE_CONTRACT_SCHEMA_VERSION = "libb-fixed-crystal-contract-v2"
SCHEMA_VERSION = "rde-libb-feature-shard-v2"
CANONICAL_SCHEMA_VERSION = "esmfold2-libb-canonical-rows-v1"
EXPECTED_RDE_REPOSITORY_COMMIT = "58887deb851bd0903295671002693e180c2d51fb"
EXPECTED_RDE_SHA256 = "2dbf5413388b784df1a7e6a3408adbc4bb5bb0406cad4a6ef3bf4e473524db6f"
EXPECTED_RDE_NETWORK_SHA256 = "7504a75fe8ed153c007d7371e41f11688379a982f94679575a6014ac13a8eb7f"
EXPECTED_ROWS = 30_768
EXPECTED_CONTEXT_DIM = 128
EXPECTED_PDB_SHA256 = "59d026542cc42006302117cc79ad720497e69c4298f95598413f1d172d407b0e"
FEATURE_CHOICES = ("rde", "rde_network", "both")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _private_mode(path):
    return "{:04o}".format(os.stat(str(path)).st_mode & 0o7777)


def validate_private_output_path(path):
    root = REPO_ROOT.resolve()
    private = PRIVATE_ROOT.resolve()
    output = Path(path).resolve()
    _require(private.is_dir(), "private_data directory is missing")
    _require(output != private, "output must be below private_data")
    try:
        relative_private = output.relative_to(private)
    except ValueError:
        raise ValueError("output must be below private_data")
    _require(str(relative_private) not in {"", "."}, "output must be a private child")
    relative_repo = output.relative_to(root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative_repo)],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(ignored.returncode == 0, "output path is not Git-ignored")
    return output


def _ensure_private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(str(path), 0o700)
    _require(_private_mode(path) == "0700", "output directory is not mode 0700")


def _atomic_json(path, payload):
    temporary = path.with_name(".{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with open(str(temporary), "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path, arrays):
    temporary = path.with_name(".{}.tmp-{}-{}".format(path.name, os.getpid(), uuid.uuid4().hex))
    try:
        with open(str(temporary), "xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_commit(repository):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(result.returncode == 0, "RDE root is not a Git checkout")
    return result.stdout.strip()


def verify_upstream(rde_root, rde_checkpoint, network_checkpoint, feature_source):
    rde_root = Path(rde_root).resolve()
    rde_checkpoint = Path(rde_checkpoint).resolve()
    network_checkpoint = Path(network_checkpoint).resolve() if network_checkpoint else None
    _require((rde_root / "rde" / "models" / "rde.py").is_file(), "RDE source is missing")
    commit = _git_commit(rde_root)
    _require(commit == EXPECTED_RDE_REPOSITORY_COMMIT, "RDE repository commit changed")
    _require(rde_checkpoint.is_file(), "RDE checkpoint is missing")
    _require(sha256_file(rde_checkpoint) == EXPECTED_RDE_SHA256, "RDE checkpoint hash changed")
    if feature_source in {"rde_network", "both"}:
        _require(network_checkpoint is not None and network_checkpoint.is_file(), "RDE-Network checkpoint is missing")
        _require(
            sha256_file(network_checkpoint) == EXPECTED_RDE_NETWORK_SHA256,
            "RDE-Network checkpoint hash changed",
        )
    return commit


def verify_structure_contract(contract_dir, row_manifest, pdb_path):
    """Bind extraction to the shared, outcome-free crystal mapping audit."""
    contract_dir = Path(contract_dir).resolve()
    manifest_path = contract_dir / "manifest.json"
    mapping_path = contract_dir / "residue_mapping.json"
    records_path = contract_dir / "current_sequence_records.json"
    for path in (manifest_path, mapping_path, records_path):
        _require(path.is_file(), "structure contract is incomplete: {}".format(path.name))
    manifest = _read_json(manifest_path)
    mapping = _read_json(mapping_path)
    _require(
        manifest.get("schema_version") == STRUCTURE_CONTRACT_SCHEMA_VERSION,
        "structure-contract schema changed",
    )
    access = manifest.get("data_access", {})
    _require(access.get("canonical_sequence_rows_read") is True, "structure contract lacks rows")
    _require(access.get("training_label_table_read") is False, "structure contract read training labels")
    _require(access.get("evaluation_outcome_table_read") is False, "structure contract read outcomes")
    inputs = manifest.get("input_hashes", {})
    _require(
        inputs.get("canonical_rows_sha256") == sha256_file(row_manifest),
        "structure contract uses different canonical rows",
    )
    _require(
        inputs.get("pdb_sha256") == sha256_file(pdb_path) == EXPECTED_PDB_SHA256,
        "structure contract or PDB hash changed",
    )
    outputs = manifest.get("outputs", {})
    _require(
        outputs.get("residue_mapping.json", {}).get("sha256") == sha256_file(mapping_path),
        "structure residue mapping hash changed",
    )
    _require(
        outputs.get("current_sequence_records.json", {}).get("sha256")
        == sha256_file(records_path),
        "structure sequence-record hash changed",
    )
    _require(
        mapping.get("schema_version") == STRUCTURE_CONTRACT_SCHEMA_VERSION,
        "residue-mapping schema changed",
    )
    _require(
        mapping.get("pdb_source", {}).get("sha256") == EXPECTED_PDB_SHA256,
        "mapping PDB hash changed",
    )
    selected = mapping.get("chain_identification", {}).get("selected_copy", {})
    _require(
        selected
        == {
            "hla_chain": HLA_CHAIN,
            "beta2m_chain": BETA2M_CHAIN,
            "peptide_chain": PEPTIDE_CHAIN,
            "affibody_chain": AFFIBODY_CHAIN,
        },
        "shared structure contract selected a different crystal copy",
    )
    _require(
        len(mapping.get("known_distance_checks", [])) == 3,
        "shared structure distance checks changed",
    )
    assay_hla = mapping.get("residue_mappings", {}).get("assay_hla", {})
    _require(
        assay_hla.get("canonical_chain1_hla_start_index_0_based")
        == ASSAY_HLA_CHAIN1_START_0_BASED
        and assay_hla.get("canonical_chain1_hla_end_index_exclusive_0_based")
        == ASSAY_HLA_CHAIN1_START_0_BASED + ASSAY_HLA_RESOLVED_LENGTH
        and assay_hla.get("aligned_hla_residue_count") == ASSAY_HLA_RESOLVED_LENGTH
        and assay_hla.get("pdb_chain_id") == HLA_CHAIN,
        "shared assay-HLA segment mapping changed",
    )
    _require(
        assay_hla.get("exact_identity_match_count")
        == ASSAY_HLA_RESOLVED_LENGTH - len(HLA_ASSAY_IDENTITY_CORRECTIONS)
        and assay_hla.get("identity_override_count")
        == len(HLA_ASSAY_IDENTITY_CORRECTIONS),
        "shared assay-HLA identity counts changed",
    )
    observed_overrides = [
        (
            int(record.get("hla_sequence_position_1_based")),
            record.get("pdb_amino_acid"),
            record.get("assay_amino_acid"),
        )
        for record in assay_hla.get("identity_overrides", [])
    ]
    _require(
        observed_overrides == list(HLA_ASSAY_IDENTITY_CORRECTIONS),
        "shared assay-HLA identity overrides changed",
    )
    _require(
        len(assay_hla.get("residues", [])) == ASSAY_HLA_RESOLVED_LENGTH,
        "shared assay-HLA residue mapping changed",
    )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "residue_mapping_sha256": sha256_file(mapping_path),
        "current_sequence_records_sha256": sha256_file(records_path),
        "schema_version": manifest["schema_version"],
    }


def _trusted_torch_load(path):
    """Load a hash-pinned official checkpoint across old/new torch versions."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _import_upstream(rde_root):
    root = str(Path(rde_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from rde.models.rde import CircularSplineRotamerDensityEstimator
    from rde.models.rde_ddg import DDG_RDE_Network
    from rde.utils.protein.parsers import parse_biopython_structure

    return CircularSplineRotamerDensityEstimator, DDG_RDE_Network, parse_biopython_structure


def load_frozen_models(rde_root, rde_checkpoint, network_checkpoint, feature_source, device):
    RDE, RDENetwork, _ = _import_upstream(rde_root)
    raw_checkpoint = _trusted_torch_load(rde_checkpoint)
    _require("config" in raw_checkpoint and "model" in raw_checkpoint, "invalid RDE checkpoint")
    raw_model = RDE(raw_checkpoint["config"].model)
    raw_model.load_state_dict(raw_checkpoint["model"])
    raw_model.requires_grad_(False).eval().to(device)

    network_models = []
    if feature_source in {"rde_network", "both"}:
        checkpoint = _trusted_torch_load(network_checkpoint)
        _require("config" in checkpoint and "model" in checkpoint, "invalid RDE-Network checkpoint")
        config = checkpoint["config"]
        config.model.rde_checkpoint = str(Path(rde_checkpoint).resolve())
        states = checkpoint["model"].get("models")
        _require(isinstance(states, list) and len(states) == 3, "expected three RDE-Network folds")
        for state in states:
            model = RDENetwork(config.model)
            model.load_state_dict(state)
            model.requires_grad_(False).eval().to(device)
            network_models.append(model)
    return raw_model, network_models


def parse_crystal_copy(rde_root, pdb_path):
    """Parse only A/B/H/P, excluding the second crystal copy and VHH chain."""
    _, _, parse_biopython_structure = _import_upstream(rde_root)
    from Bio.PDB.PDBParser import PDBParser

    structure = PDBParser(QUIET=True).get_structure(None, str(pdb_path))
    models = list(structure.get_models())
    _require(len(models) == 1, "expected one coordinate model")
    model = models[0]
    available = {chain.id for chain in model.get_chains()}
    _require(set(CRYSTAL_CHAIN_IDS).issubset(available), "A/B/H/P crystal copy is incomplete")
    selected_chains = [model[chain_id] for chain_id in CRYSTAL_CHAIN_IDS]
    data, sequence_map = parse_biopython_structure(selected_chains)
    _require(data is not None, "official RDE parser returned no residues")
    _require(sequence_map is not None, "official RDE parser returned no sequence map")
    _require(set(data["chain_id"]) == set(CRYSTAL_CHAIN_IDS), "RDE parser chain set changed")
    return data, sequence_map


def _to_numpy(tensor, dtype=None):
    value = tensor.detach().cpu().numpy()
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def encode_batch(raw_model, network_models, batch, feature_source, device):
    model_inputs = model_batch(batch, device)
    arrays = {}
    with torch.no_grad():
        if feature_source in {"rde", "both"}:
            context = raw_model.encode(model_inputs)
            _require(context.ndim == 3 and context.shape[-1] == EXPECTED_CONTEXT_DIM, "RDE context shape changed")
            _require(bool(torch.isfinite(context).all()), "non-finite RDE context")
            arrays["rde_context"] = _to_numpy(context, np.float16)
        if feature_source in {"rde_network", "both"}:
            _require(len(network_models) == 3, "RDE-Network folds are missing")
            contexts = [model.encode(model_inputs) for model in network_models]
            for context in contexts:
                _require(
                    context.ndim == 3 and context.shape[-1] == EXPECTED_CONTEXT_DIM,
                    "RDE-Network context shape changed",
                )
                _require(bool(torch.isfinite(context).all()), "non-finite RDE-Network context")
            for fold, context in enumerate(contexts):
                arrays["rde_network_fold{}_context".format(fold)] = _to_numpy(
                    context, np.float16
                )
    masks = feature_masks(batch)
    arrays.update(
        {
            "residue_mask": _to_numpy(masks["residue_mask"], np.bool_),
            "peptide_mask": _to_numpy(masks["peptide_mask"], np.bool_),
            "affibody_mask": _to_numpy(masks["affibody_mask"], np.bool_),
            "designed_mask": _to_numpy(masks["designed_mask"], np.bool_),
            "chain_nb": _to_numpy(masks["chain_nb"], np.int16),
            "aa": _to_numpy(masks["aa"], np.uint8),
        }
    )
    return arrays


def chunk_arrays(batch, encoded):
    arrays = {
        "row_index": batch["row_index"].numpy().astype(np.int64, copy=False),
        "row_id": np.asarray(batch["row_id"], dtype="U"),
        "split": np.asarray(batch["split"], dtype="U5"),
        "sequence_pair_sha256": np.asarray(batch["sequence_pair_sha256"], dtype="U64"),
    }
    arrays.update(encoded)
    return arrays


def expected_feature_keys(feature_source):
    keys = {
        "row_index",
        "row_id",
        "split",
        "sequence_pair_sha256",
        "residue_mask",
        "peptide_mask",
        "affibody_mask",
        "designed_mask",
        "chain_nb",
        "aa",
    }
    if feature_source in {"rde", "both"}:
        keys.add("rde_context")
    if feature_source in {"rde_network", "both"}:
        keys.update(
            "rde_network_fold{}_context".format(fold) for fold in range(3)
        )
    return keys


def validate_chunk(path, rows, feature_source, patch_size=PATCH_SIZE):
    with np.load(str(path), allow_pickle=False) as archive:
        _require(set(archive.files) == expected_feature_keys(feature_source), "chunk fields changed")
        n = len(rows)
        _require(archive["row_index"].tolist() == [row.row_index for row in rows], "chunk row order changed")
        _require(archive["row_id"].astype(str).tolist() == [row.row_id for row in rows], "chunk row IDs changed")
        _require(archive["split"].astype(str).tolist() == [row.split for row in rows], "chunk splits changed")
        _require(
            archive["sequence_pair_sha256"].astype(str).tolist()
            == [row.sequence_pair_sha256 for row in rows],
            "chunk sequence hashes changed",
        )
        for key in ("residue_mask", "peptide_mask", "affibody_mask", "designed_mask", "chain_nb", "aa"):
            _require(archive[key].shape == (n, patch_size), "{} shape changed".format(key))
        _require(bool(archive["residue_mask"].all()), "patch contains missing residues")
        _require(bool((archive["designed_mask"].sum(axis=1) == 7).all()), "designed mask changed")
        for key in (
            "rde_context",
            "rde_network_fold0_context",
            "rde_network_fold1_context",
            "rde_network_fold2_context",
        ):
            if key in archive.files:
                _require(
                    archive[key].shape == (n, patch_size, EXPECTED_CONTEXT_DIM),
                    "{} shape changed".format(key),
                )
                _require(archive[key].dtype == np.float16, "{} dtype changed".format(key))
                _require(bool(np.isfinite(archive[key]).all()), "{} is non-finite".format(key))
    return sha256_file(path)


def run(args):
    _require(args.feature_source in FEATURE_CHOICES, "invalid feature source")
    _require(args.num_shards >= 1, "num_shards must be positive")
    _require(0 <= args.shard_index < args.num_shards, "invalid shard_index")
    _require(args.batch_size >= 1, "batch_size must be positive")
    _require(args.patch_size == PATCH_SIZE, "official experiment fixes patch_size at 128")
    _require(args.limit is None or args.limit >= 1, "limit must be positive")
    _require(
        args.row_indices is None or (args.num_shards == 1 and args.shard_index == 0),
        "explicit row indices require the default single-shard selection",
    )
    row_manifest = Path(args.row_manifest).resolve()
    pdb_path = Path(args.pdb).resolve()
    _require(row_manifest.is_file(), "canonical row manifest is missing")
    _require(pdb_path.is_file(), "crystal PDB is missing")
    output_root = validate_private_output_path(args.output_dir)
    _ensure_private_directory(output_root)
    shard_dir = output_root / "shard-{:03d}-of-{:03d}".format(args.shard_index, args.num_shards)
    _ensure_private_directory(shard_dir)

    upstream_commit = verify_upstream(
        args.rde_root,
        args.rde_checkpoint,
        args.rde_network_checkpoint,
        args.feature_source,
    )
    payload = _read_json(row_manifest)
    structure_contract = verify_structure_contract(
        args.structure_contract_dir, row_manifest, pdb_path
    )
    expected_total = None if args.allow_noncanonical_size else EXPECTED_ROWS
    rows = validate_canonical_payload(payload, expected_total=expected_total)
    if args.row_indices is not None:
        requested = [int(value) for value in args.row_indices.split(",") if value.strip()]
        _require(requested and len(set(requested)) == len(requested), "invalid explicit row indices")
        by_index = {row.row_index: row for row in rows}
        _require(set(requested).issubset(by_index), "explicit row index is out of range")
        selected = [by_index[index] for index in requested]
    else:
        selected = [row for row in rows if row.row_index % args.num_shards == args.shard_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    _require(selected, "this shard selects no rows")

    parsed, _ = parse_crystal_copy(args.rde_root, pdb_path)
    template = build_fixed_crystal_template(parsed, rows[0], patch_size=args.patch_size)
    device = torch.device(args.device)
    raw_model, network_models = load_frozen_models(
        args.rde_root,
        args.rde_checkpoint,
        args.rde_network_checkpoint,
        args.feature_source,
        device,
    )

    started = time.time()
    chunks = []
    for offset in range(0, len(selected), args.batch_size):
        rows_this = selected[offset : offset + args.batch_size]
        first = rows_this[0].row_index
        last = rows_this[-1].row_index
        chunk_path = shard_dir / "rows-{:06d}-{:06d}.npz".format(first, last)
        if chunk_path.exists():
            digest = validate_chunk(chunk_path, rows_this, args.feature_source, args.patch_size)
            chunks.append({"filename": chunk_path.name, "rows": len(rows_this), "sha256": digest})
            continue
        examples = [prepare_current_pair(template, row) for row in rows_this]
        batch = collate_current_pairs(examples)
        encoded = encode_batch(raw_model, network_models, batch, args.feature_source, device)
        arrays = chunk_arrays(batch, encoded)
        _atomic_npz(chunk_path, arrays)
        digest = validate_chunk(chunk_path, rows_this, args.feature_source, args.patch_size)
        chunks.append({"filename": chunk_path.name, "rows": len(rows_this), "sha256": digest})

    elapsed = time.time() - started
    chain_ids = list(template.data["chain_id"])
    hla_patch_numbers = [
        int(resseq)
        for chain_id, resseq in zip(
            template.data["chain_id"], template.data["resseq"]
        )
        if chain_id == HLA_CHAIN
    ]
    _require(hla_patch_numbers, "fixed patch unexpectedly lacks HLA context")
    _require(
        BETA2M_CHAIN not in chain_ids
        and max(hla_patch_numbers) <= ASSAY_HLA_RESOLVED_LENGTH,
        "fixed patch includes coordinates absent from the assay approximation",
    )
    hla_patch_lookup = {
        (int(resseq), str(icode).strip()): index
        for index, (chain_id, resseq, icode) in enumerate(
            zip(
                template.data["chain_id"],
                template.data["resseq"],
                template.data["icode"],
            )
        )
        if chain_id == HLA_CHAIN
    }
    hla_graft_patch_status = []
    for residue_number, _, assay_aa in HLA_ASSAY_IDENTITY_CORRECTIONS:
        patch_index = hla_patch_lookup.get((residue_number, ""))
        status = {
            "pdb_residue_number": residue_number,
            "retained_in_patch": patch_index is not None,
            "model_amino_acid": None,
            "chi_hidden": None,
        }
        if patch_index is not None:
            status["model_amino_acid"] = AA_ALPHABET[
                int(template.data["aa"][patch_index])
            ]
            status["chi_hidden"] = bool(
                not template.data["chi_mask"][patch_index].any()
                and not template.data["chi_complete"][patch_index]
                and (template.data["chi"][patch_index] == 0).all()
                and (template.data["chi_alt"][patch_index] == 0).all()
            )
            _require(
                status["model_amino_acid"] == assay_aa
                and status["chi_hidden"] is True,
                "assay HLA identity graft was not applied inside the patch",
            )
        hla_graft_patch_status.append(status)
    mapping_rows = []
    for patch_index, (chain_id, resseq, icode) in enumerate(
        zip(template.data["chain_id"], template.data["resseq"], template.data["icode"])
    ):
        record = {
            "patch_index_0_based": patch_index,
            "chain_id": str(chain_id),
            "pdb_residue_number": int(resseq),
            "pdb_insertion_code": str(icode).strip(),
            "role": (
                "HLA"
                if chain_id == HLA_CHAIN
                else "beta2m"
                if chain_id == BETA2M_CHAIN
                else "Affibody"
                if chain_id == AFFIBODY_CHAIN
                else "peptide"
            ),
            "designed": bool(patch_index in template.designed_patch_indices),
        }
        if chain_id == PEPTIDE_CHAIN:
            inverse = {value: key for key, value in template.peptide_full_to_patch.items()}
            record["partner_sequence_position_1_based"] = inverse.get(patch_index)
        elif chain_id == AFFIBODY_CHAIN:
            inverse = {value: key for key, value in template.affibody_full_to_patch.items()}
            displayed_position = inverse.get(patch_index)
            record["partner_sequence_position_1_based"] = displayed_position
            record["displayed_sequence_position_1_based"] = displayed_position
            record["provider_crystal_aligned_label_1_based"] = (
                AFFIBODY_DISPLAY_TO_CRYSTAL_ALIGNED.get(displayed_position)
            )
        else:
            record["partner_sequence_position_1_based"] = None
        mapping_rows.append(record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "feature_source": args.feature_source,
        "absolute_current_pair": True,
        "wt_mutant_subtraction_used": False,
        "upstream": {
            "repository": "https://github.com/luost26/RDE-PPI",
            "commit": upstream_commit,
            "license": "Apache-2.0",
            "rde_checkpoint_sha256": EXPECTED_RDE_SHA256,
            "rde_network_checkpoint_sha256": (
                EXPECTED_RDE_NETWORK_SHA256
                if args.feature_source in {"rde_network", "both"}
                else None
            ),
            "rde_network_folds_emitted_separately": 3 if network_models else 0,
        },
        "input": {
            "row_manifest_sha256": sha256_file(row_manifest),
            "pdb_sha256": sha256_file(pdb_path),
            "canonical_schema": CANONICAL_SCHEMA_VERSION,
        },
        "crystal": {
            "chains": [HLA_CHAIN, BETA2M_CHAIN, PEPTIDE_CHAIN, AFFIBODY_CHAIN],
            "patch_chain_counts": {
                chain_id: chain_ids.count(chain_id) for chain_id in CRYSTAL_CHAIN_IDS
            },
            "patch_hla_residue_numbers": hla_patch_numbers,
            "patch_hla_residue_number_min": min(hla_patch_numbers),
            "patch_hla_residue_number_max": max(hla_patch_numbers),
            "patch_size": template.patch_size,
            "designed_residue_count": len(template.designed_patch_indices),
            "designed_sidechain_cb_hidden": True,
            "designed_chi_hidden": True,
            "assay_hla_identity_graft": list(template.hla_identity_corrections),
            "assay_hla_identity_graft_patch_status": hla_graft_patch_status,
            "patch_excludes_beta2m": BETA2M_CHAIN not in chain_ids,
            "patch_excludes_hla_after_assay_local_position_181": all(
                chain_id != HLA_CHAIN or int(resseq) <= ASSAY_HLA_RESOLVED_LENGTH
                for chain_id, resseq in zip(
                    template.data["chain_id"], template.data["resseq"]
                )
            ),
            "fixed_crystal_approximation_omissions": list(
                FIXED_CRYSTAL_APPROXIMATION_OMISSIONS
            ),
            "residue_mapping": mapping_rows,
        },
        "structure_contract": structure_contract,
        "shard": {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(selected),
            "first_row_index": selected[0].row_index,
            "last_row_index": selected[-1].row_index,
            "chunks": chunks,
        },
        "features": {},
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "elapsed_seconds": elapsed,
            "seconds_per_row": elapsed / len(selected),
        },
    }
    with np.load(str(shard_dir / chunks[0]["filename"]), allow_pickle=False) as example:
        manifest["features"] = {
            key: {
                "per_row_shape": list(example[key].shape[1:]),
                "dtype": str(example[key].dtype),
            }
            for key in example.files
            if key not in {"row_index", "row_id", "split", "sequence_pair_sha256"}
        }
    manifest_path = shard_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rows": len(selected),
                "seconds": round(elapsed, 3),
                "seconds_per_row": round(elapsed / len(selected), 6),
                "feature_source": args.feature_source,
            },
            sort_keys=True,
        )
    )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rde-root", type=Path, required=True)
    parser.add_argument("--rde-checkpoint", type=Path, required=True)
    parser.add_argument("--rde-network-checkpoint", type=Path)
    parser.add_argument("--row-manifest", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument(
        "--structure-contract-dir",
        type=Path,
        default=DEFAULT_STRUCTURE_CONTRACT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-source", choices=FEATURE_CHOICES, default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--row-indices",
        help="Comma-separated canonical row indices for a small mapping pilot.",
    )
    parser.add_argument(
        "--allow-noncanonical-size",
        action="store_true",
        help="Testing/pilot escape hatch; production requires exactly 30,768 rows.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
