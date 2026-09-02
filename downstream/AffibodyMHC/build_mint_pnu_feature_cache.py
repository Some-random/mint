#!/usr/bin/env python
"""Build a restartable frozen-MINT layer-33 cache for P/N/U experiments.

The existing ``mint_weak_cache_v1`` cache is immutable.  This program reuses
its already-computed layer-33 features for the strict positive (P) and
conservative negative (N) rows and computes only the missing unlabeled (U)
features.  U is the observed R009/R010 union below the positive cutoff; it is
never expanded to a Cartesian product of unobserved peptide/Affibody pairs.

The three stages are deliberately separate so hundreds of thousands of U rows
can be distributed safely across GPU nodes:

``prepare``
    Bind the old P/N cache and the U-pool manifests, apply the exact
    library-local retention-identity-cold rule used by the current separate-
    library experiments, take a deterministic library-stratified U sample (or
    all U), reconstruct the two provider-confirmed MINT chains, and write rows.
``extract-shard``
    Embed U rows whose prepared ``row_index modulo shard_count`` equals the
    requested shard index.  Existing valid shard artifacts can be resumed.
``merge``
    Restore prepared order, reuse P/N features, insert U shard features, and
    write a unified float16 cache plus a hash-bound manifest.

All row-level output is refused outside the Git-ignored ``private_data`` tree.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.build_sequence_table import (
    CHAIN1_LENGTH,
    CHAIN2_LENGTH,
    OMITTED_LINKER,
    load_provider_templates,
)
from downstream.AffibodyMHC.code_only_baseline import (
    AA_ALPHABET,
    opaque_id,
    sha256_file,
)
from downstream.AffibodyMHC.extract_mint_features import PairCollator
from downstream.AffibodyMHC import build_mint_weak_feature_cache as old_cache
from mint.helpers.extract import MINTWrapper, load_config


SCHEMA_VERSION = "mint-pnu-layer33-cache-v1"
LIBRARIES = ("LibA", "LibB")
CLASS_ORDER = {"P": 0, "N": 1, "U": 2}
ROWS_FILENAME = "cache_rows.csv.gz"
PREPARE_MANIFEST_FILENAME = "manifest.json"
FEATURE_FILENAME = "mint_pnu_layer33_features.npz"
FEATURE_NAME = "mint_chain_mean"
FEATURE_DIMENSION = 2560
FEATURE_DTYPE = np.float16
DEFAULT_SAMPLE_SEED = 20260902

U_COLUMNS = (
    "library",
    "pep",
    "aff",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "positive_cutoff",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
)

ROW_COLUMNS = (
    "row_index",
    "cache_uid",
    "library",
    "class_source",
    "weak_label",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "pep",
    "aff",
    "r001_count",
    "r009_count",
    "r010_count",
    "pooled_r009_r010_count",
    "positive_cutoff",
    "feature_source",
    "source_cache_row_index",
    "source_cache_uid",
    "chain1_smart_hla_linker_peptide_sequence",
    "chain2_affibody_sequence",
    "chain1_sha256",
    "chain2_sha256",
    "sequence_pair_sha256",
)

NPZ_METADATA_COLUMNS = (
    "row_index",
    "cache_uid",
    "library",
    "class_source",
    "weak_label",
    "pair_uid",
    "peptide_uid",
    "affibody_uid",
    "pep",
    "aff",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_string_csv(path):
    return pd.read_csv(
        str(path), dtype=str, keep_default_na=False, na_filter=False
    )


def read_json(path):
    with open(str(path), "r") as handle:
        return json.load(handle)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def _atomic_write_csv_gz(path, frame):
    _require(not path.exists(), "output exists; refusing overwrite: {}".format(path))
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        frame.to_csv(str(temporary), index=False, compression="gzip")
        os.chmod(str(temporary), 0o600)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    _require(old_cache.private_mode(path) == "0600", "CSV output is not mode 0600")


def _source_record(path):
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def stable_sample_u(frame, library, count, seed):
    """Select U by a version-independent SHA-256 ordering."""
    subset = frame.loc[frame["library"].eq(library)].copy()
    _require(int(count) >= 1, "requested U sample must be positive")
    _require(len(subset) >= int(count), "requested U sample exceeds available pool")
    prefix = "{}|{}|".format(int(seed), library)
    subset["_sample_key"] = subset["pair_uid"].map(
        lambda value: sha256_text(prefix + str(value))
    )
    subset = subset.sort_values(["_sample_key", "pair_uid"], kind="mergesort")
    return subset.iloc[: int(count)].drop(columns=["_sample_key"]).reset_index(drop=True)


def select_strict_pn(old_rows):
    """Select current separate-library P/N rows without changing old semantics."""
    required = {
        "row_index",
        "cache_uid",
        "source_kind",
        "library",
        "weak_label",
        "pair_uid",
        "peptide_uid",
        "affibody_uid",
        "peptide_design_code",
        "affibody_design_code",
        "r001_count",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "upstream_library_local_strict_retention_identity_cold_eligible",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    }
    _require(required.issubset(set(old_rows.columns)), "old P/N row schema mismatch")
    mask = old_rows["source_kind"].eq("weak") & old_rows[
        "upstream_library_local_strict_retention_identity_cold_eligible"
    ].eq("1")
    selected = old_rows.loc[mask].copy()
    _require(bool(selected["weak_label"].isin(["0", "1"]).all()), "invalid P/N label")
    selected["class_source"] = selected["weak_label"].map({"1": "P", "0": "N"})
    _require(not bool(selected["pair_uid"].duplicated().any()), "duplicate P/N pair UID")
    return selected.reset_index(drop=True)


def _positive_cutoffs(u_frame):
    result = {}
    for library in LIBRARIES:
        values = pd.to_numeric(
            u_frame.loc[u_frame["library"].eq(library), "positive_cutoff"],
            errors="raise",
        ).unique()
        _require(len(values) == 1, "{} U pool has multiple cutoffs".format(library))
        result[library] = int(values[0])
    return result


def assemble_rows(old_rows, u_pool, templates, u_per_positive=5, all_u=False, seed=DEFAULT_SAMPLE_SEED):
    _require(tuple(u_pool.columns) == U_COLUMNS, "U-pool schema mismatch")
    _require(set(u_pool["library"]) == set(LIBRARIES), "U pool lacks a library")
    _require(not bool(u_pool["pair_uid"].duplicated().any()), "duplicate U pair UID")
    pn = select_strict_pn(old_rows)
    _require(set(pn["pair_uid"]).isdisjoint(set(u_pool["pair_uid"])), "P/N and U overlap")
    cutoffs = _positive_cutoffs(u_pool)

    sampled_blocks = []
    for library in LIBRARIES:
        positive_count = int(
            (pn["library"].eq(library) & pn["class_source"].eq("P")).sum()
        )
        wanted = int((u_pool["library"].eq(library)).sum()) if all_u else int(u_per_positive) * positive_count
        sampled_blocks.append(stable_sample_u(u_pool, library, wanted, seed))
    sampled_u = pd.concat(sampled_blocks, ignore_index=True)

    records = []
    for row in pn.itertuples(index=False):
        records.append(
            {
                "cache_uid": opaque_id(SCHEMA_VERSION, row.class_source, row.pair_uid),
                "library": row.library,
                "class_source": row.class_source,
                "weak_label": str(row.weak_label),
                "pair_uid": row.pair_uid,
                "peptide_uid": row.peptide_uid,
                "affibody_uid": row.affibody_uid,
                "pep": row.peptide_design_code,
                "aff": row.affibody_design_code,
                "r001_count": row.r001_count,
                "r009_count": row.r009_count,
                "r010_count": row.r010_count,
                "pooled_r009_r010_count": row.pooled_r009_r010_count,
                "positive_cutoff": str(cutoffs[row.library]),
                "feature_source": "reuse_old_layer33",
                "source_cache_row_index": str(row.row_index),
                "source_cache_uid": row.cache_uid,
                "chain1_smart_hla_linker_peptide_sequence": row.chain1_smart_hla_linker_peptide_sequence,
                "chain2_affibody_sequence": row.chain2_affibody_sequence,
                "chain1_sha256": row.chain1_sha256,
                "chain2_sha256": row.chain2_sha256,
                "sequence_pair_sha256": row.sequence_pair_sha256,
            }
        )

    for row in sampled_u.itertuples(index=False):
        chain1, chain2 = old_cache.reconstruct_chains(
            templates[row.library], row.library, str(row.pep), str(row.aff)
        )
        records.append(
            {
                "cache_uid": opaque_id(SCHEMA_VERSION, "U", row.pair_uid),
                "library": row.library,
                "class_source": "U",
                "weak_label": "-1",
                "pair_uid": row.pair_uid,
                "peptide_uid": row.peptide_uid,
                "affibody_uid": row.affibody_uid,
                "pep": str(row.pep),
                "aff": str(row.aff),
                "r001_count": "",
                "r009_count": str(row.r009_count),
                "r010_count": str(row.r010_count),
                "pooled_r009_r010_count": str(row.pooled_r009_r010_count),
                "positive_cutoff": str(row.positive_cutoff),
                "feature_source": "extract_layer33",
                "source_cache_row_index": "",
                "source_cache_uid": "",
                "chain1_smart_hla_linker_peptide_sequence": chain1,
                "chain2_affibody_sequence": chain2,
                "chain1_sha256": sha256_text(chain1),
                "chain2_sha256": sha256_text(chain2),
                "sequence_pair_sha256": sha256_text(chain1 + "|" + chain2),
            }
        )

    result = pd.DataFrame(records)
    result["_class_order"] = result["class_source"].map(CLASS_ORDER)
    result = result.sort_values(
        ["library", "_class_order", "pair_uid"], kind="mergesort"
    ).drop(columns=["_class_order"]).reset_index(drop=True)
    result.insert(0, "row_index", np.arange(len(result), dtype=np.int64))
    result = result.loc[:, list(ROW_COLUMNS)]
    validate_rows(result.astype(str))
    return result


def validate_rows(frame):
    _require(tuple(frame.columns) == ROW_COLUMNS, "prepared row schema mismatch")
    indices = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(dtype=np.int64)
    _require(np.array_equal(indices, np.arange(len(frame))), "row indices are not contiguous")
    _require(frame["cache_uid"].nunique() == len(frame), "duplicate cache UID")
    _require(frame["pair_uid"].nunique() == len(frame), "duplicate biological pair")
    _require(set(frame["library"]) == set(LIBRARIES), "prepared rows lack a library")
    _require(set(frame["class_source"]) == {"P", "N", "U"}, "prepared rows lack P/N/U")
    expected_labels = frame["class_source"].map({"P": "1", "N": "0", "U": "-1"})
    _require(expected_labels.tolist() == frame["weak_label"].tolist(), "class/label mismatch")
    expected_sources = frame["class_source"].map(
        {"P": "reuse_old_layer33", "N": "reuse_old_layer33", "U": "extract_layer33"}
    )
    _require(expected_sources.tolist() == frame["feature_source"].tolist(), "feature-source mismatch")
    for column, length in (
        ("chain1_smart_hla_linker_peptide_sequence", CHAIN1_LENGTH),
        ("chain2_affibody_sequence", CHAIN2_LENGTH),
    ):
        _require(bool(frame[column].map(len).eq(length).all()), "chain length mismatch")
        _require(bool(frame[column].map(lambda x: set(x).issubset(set(AA_ALPHABET))).all()), "noncanonical chain")
        _require(not bool(frame[column].str.contains(OMITTED_LINKER, regex=False).any()), "omitted linker leaked into model input")
    _require(
        [sha256_text(a + "|" + b) for a, b in zip(
            frame["chain1_smart_hla_linker_peptide_sequence"],
            frame["chain2_affibody_sequence"],
        )] == frame["sequence_pair_sha256"].tolist(),
        "sequence-pair hash mismatch",
    )
    return indices


def summarize_rows(frame):
    result = {"total": int(len(frame)), "libraries": {}}
    for library in LIBRARIES:
        subset = frame.loc[frame["library"].eq(library)]
        result["libraries"][library] = {
            source: int(subset["class_source"].eq(source).sum())
            for source in ("P", "N", "U")
        }
    return result


def _validate_bound_output(manifest, key, path):
    claimed = manifest.get("output", {}).get(key, {}).get("sha256")
    if claimed is None:
        claimed = manifest.get("outputs", {}).get(key)
    _require(claimed == sha256_file(path), "manifest/output mismatch for {}".format(key))


def run_prepare(args):
    started = time.time()
    inputs = (
        args.u_pool,
        args.u_manifest,
        args.old_rows,
        args.old_rows_manifest,
        args.old_features,
        args.old_features_manifest,
        args.sequence_zip,
    )
    for path in inputs:
        _require(path.is_file(), "missing input {}".format(path))
    _require(args.u_per_positive >= 1, "U-per-positive must be positive")

    u_manifest = read_json(args.u_manifest)
    old_rows_manifest = read_json(args.old_rows_manifest)
    old_features_manifest = read_json(args.old_features_manifest)
    _validate_bound_output(u_manifest, args.u_pool.name, args.u_pool)
    _require(
        old_rows_manifest.get("output", {}).get("rows_csv", {}).get("sha256")
        == sha256_file(args.old_rows),
        "old row manifest/output mismatch",
    )
    _require(
        old_features_manifest.get("output", {}).get("sha256")
        == sha256_file(args.old_features),
        "old feature manifest/output mismatch",
    )
    sequence_zip_hash = sha256_file(args.sequence_zip)
    _require(
        old_rows_manifest.get("sources", {}).get("sequence_zip", {}).get("sha256")
        == sequence_zip_hash,
        "old cache uses a different sequence ZIP",
    )

    old_rows = read_string_csv(args.old_rows)
    u_pool = read_string_csv(args.u_pool)
    templates, template_member, template_deck_hash = load_provider_templates(args.sequence_zip)
    _require(set(templates) == set(LIBRARIES), "provider templates lack a library")
    rows = assemble_rows(
        old_rows,
        u_pool,
        templates,
        u_per_positive=args.u_per_positive,
        all_u=args.all_u,
        seed=args.sample_seed,
    )

    output_dir = old_cache._ensure_private_directory(args.output_dir, must_be_new=True)
    rows_path = output_dir / ROWS_FILENAME
    manifest_path = output_dir / PREPARE_MANIFEST_FILENAME
    _atomic_write_csv_gz(rows_path, rows)
    u_rows = rows.loc[rows["class_source"].eq("U")]
    membership_hashes = {
        library: sha256_text("\n".join(
            u_rows.loc[u_rows["library"].eq(library), "pair_uid"].tolist()
        ))
        for library in LIBRARIES
    }
    membership_hashes["all"] = sha256_text("\n".join(u_rows["pair_uid"].tolist()))
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "sources": {
            "u_pool": _source_record(args.u_pool),
            "u_manifest": _source_record(args.u_manifest),
            "old_rows": _source_record(args.old_rows),
            "old_rows_manifest": _source_record(args.old_rows_manifest),
            "old_features": _source_record(args.old_features),
            "old_features_manifest": _source_record(args.old_features_manifest),
            "sequence_zip": dict(
                _source_record(args.sequence_zip),
                template_member=template_member,
                template_deck_sha256=template_deck_hash,
            ),
        },
        "selection": {
            "P": "old weak_label=1 and upstream library-local strict retention identity cold eligible=1",
            "N": "old weak_label=0 (R001 count >=3 and absent later) and same strict rule",
            "U": "distinct observed R009/R010-union pair below the library positive cutoff and same strict rule",
            "u_mode": "all" if args.all_u else "deterministic_sample",
            "u_per_positive": None if args.all_u else int(args.u_per_positive),
            "sample_seed": int(args.sample_seed),
            "sampling_algorithm": "ascending SHA256(seed|library|pair_uid), then pair_uid",
            "u_membership_sha256": membership_hashes,
            "no_cartesian_unobserved_pairs": True,
        },
        "rows": summarize_rows(rows),
        "provider_model_input": {
            "chain_order": ["smart-HLA-linker-peptide", "Affibody"],
            "chain_lengths": [CHAIN1_LENGTH, CHAIN2_LENGTH],
            "experimental_construct_linker_omitted": OMITTED_LINKER,
        },
        "output": {
            ROWS_FILENAME: {
                "path": str(rows_path.resolve()),
                "sha256": sha256_file(rows_path),
                "mode": old_cache.private_mode(rows_path),
                "columns": list(ROW_COLUMNS),
            },
            "directory_mode": old_cache.private_mode(output_dir),
        },
        "code": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    old_cache._atomic_write_json(manifest_path, manifest)
    print(json.dumps({"stage": "prepare", "rows": summarize_rows(rows), "output": str(rows_path)}, sort_keys=True))


def load_prepared(input_dir):
    rows_path = input_dir / ROWS_FILENAME
    manifest_path = input_dir / PREPARE_MANIFEST_FILENAME
    _require(rows_path.is_file() and manifest_path.is_file(), "prepared input is incomplete")
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "prepared schema mismatch")
    _require(manifest.get("stage") == "prepare", "input is not prepare stage")
    _require(
        manifest.get("output", {}).get(ROWS_FILENAME, {}).get("sha256") == sha256_file(rows_path),
        "prepared row hash mismatch",
    )
    frame = read_string_csv(rows_path)
    validate_rows(frame)
    return frame, manifest, rows_path, manifest_path


def shard_filename(index, count):
    return "u-layer33-shard-{:03d}-of-{:03d}.npz".format(int(index), int(count))


def shard_manifest_filename(index, count):
    return "u-layer33-shard-{:03d}-of-{:03d}.manifest.json".format(int(index), int(count))


def select_shard_rows(frame, shard_index, shard_count):
    _require(int(shard_count) >= 1, "shard count must be positive")
    _require(0 <= int(shard_index) < int(shard_count), "shard index out of range")
    index = pd.to_numeric(frame["row_index"], errors="raise").to_numpy(dtype=np.int64)
    mask = frame["class_source"].eq("U").to_numpy() & (
        np.mod(index, int(shard_count)) == int(shard_index)
    )
    result = frame.loc[mask].copy().reset_index(drop=True)
    _require(not result.empty, "U shard is empty")
    return result


class UPairDataset(Dataset):
    def __init__(self, frame):
        self.chain1 = frame["chain1_smart_hla_linker_peptide_sequence"].tolist()
        self.chain2 = frame["chain2_affibody_sequence"].tolist()
        self.library = frame["library"].tolist()
        self.cache_uid = frame["cache_uid"].tolist()

    def __len__(self):
        return len(self.cache_uid)

    def __getitem__(self, index):
        return self.chain1[index], self.chain2[index], self.library[index], self.cache_uid[index]


def _contract(rows_hash, prepare_hash, checkpoint_hash, config_hash, shard_count):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "rows_sha256": rows_hash,
        "prepare_manifest_sha256": prepare_hash,
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": config_hash,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "mint_source_tree_sha256": old_cache.sha256_source_tree(REPO_ROOT / "mint"),
        "pair_collator_sha256": sha256_file(REPO_ROOT / "downstream/AffibodyMHC/extract_mint_features.py"),
        "shard_count": int(shard_count),
        "layer": 33,
        "pooling": "separate chain residue means concatenated",
        "feature_dimension": FEATURE_DIMENSION,
        "feature_dtype": "float16",
    }
    return canonical_json_sha256(payload), payload


def _resume_valid_shard(output_path, manifest_path, expected):
    if not output_path.exists() and not manifest_path.exists():
        return False
    _require(output_path.is_file() and manifest_path.is_file(), "partial shard artifact exists")
    manifest = read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "resume shard schema mismatch")
    _require(manifest.get("stage") == "extract_shard", "resume artifact is not a shard")
    _require(manifest.get("contract", {}).get("sha256") == expected["contract"], "resume shard contract mismatch")
    _require(manifest.get("output", {}).get("sha256") == sha256_file(output_path), "resume shard hash mismatch")
    _require(int(manifest.get("sharding", {}).get("shard_index", -1)) == expected["index"], "resume shard index mismatch")
    _require(int(manifest.get("sharding", {}).get("shard_count", -1)) == expected["count"], "resume shard count mismatch")
    return True


def run_extract_shard(args):
    started = time.time()
    _require(args.batch_size >= 1, "batch size must be positive")
    _require(args.checkpoint.is_file() and args.config.is_file(), "model input missing")
    frame, _, rows_path, prepare_path = load_prepared(args.input_dir)
    shard = select_shard_rows(frame, args.shard_index, args.shard_count)
    rows_hash = sha256_file(rows_path)
    prepare_hash = sha256_file(prepare_path)
    checkpoint_hash = sha256_file(args.checkpoint)
    config_hash = sha256_file(args.config)
    contract_hash, contract_payload = _contract(
        rows_hash, prepare_hash, checkpoint_hash, config_hash, args.shard_count
    )
    output_dir = old_cache._ensure_private_directory(args.output_dir, must_be_new=False)
    output_path = output_dir / shard_filename(args.shard_index, args.shard_count)
    manifest_path = output_dir / shard_manifest_filename(args.shard_index, args.shard_count)
    if args.resume and _resume_valid_shard(
        output_path,
        manifest_path,
        {"contract": contract_hash, "index": args.shard_index, "count": args.shard_count},
    ):
        print(json.dumps({"stage": "extract_shard", "status": "resumed_existing", "shard_index": args.shard_index, "output": str(output_path)}, sort_keys=True))
        return
    _require(not output_path.exists() and not manifest_path.exists(), "shard output exists; pass --resume to validate and reuse")

    device = torch.device(args.device)
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA is required")
    torch.cuda.set_device(device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader = DataLoader(
        UPairDataset(shard),
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=PairCollator(),
        num_workers=0,
    )
    wrapper = MINTWrapper(
        load_config(str(args.config)),
        str(args.checkpoint),
        freeze_percent=1.0,
        use_multimer=True,
        sep_chains=True,
        device=str(device),
    )
    wrapper.eval()
    blocks = []
    observed_uids = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for step, (chains, chain_ids, _, cache_uids) in enumerate(loader):
            values = wrapper(chains.to(device), chain_ids.to(device))
            _require(tuple(values.shape[1:]) == (FEATURE_DIMENSION,), "unexpected MINT feature shape")
            blocks.append(values.float().cpu().numpy().astype(FEATURE_DTYPE))
            observed_uids.extend(cache_uids)
            if (step + 1) % 50 == 0 or step + 1 == len(loader):
                print("shard {}/{}: {}/{} batches".format(args.shard_index, args.shard_count, step + 1, len(loader)), flush=True)
    torch.cuda.synchronize(device)
    _require(observed_uids == shard["cache_uid"].tolist(), "embedding order changed")
    features = np.concatenate(blocks, axis=0).astype(FEATURE_DTYPE, copy=False)
    _require(features.shape == (len(shard), FEATURE_DIMENSION), "shard feature shape mismatch")
    _require(bool(np.isfinite(features).all()), "non-finite shard feature")
    arrays = {
        "row_index": pd.to_numeric(shard["row_index"], errors="raise").to_numpy(dtype=np.int64),
        "cache_uid": shard["cache_uid"].to_numpy(dtype=str),
        FEATURE_NAME: features,
    }
    old_cache._atomic_write_npz(output_path, arrays, compressed=args.compressed)
    _require(rows_hash == sha256_file(rows_path), "rows changed during extraction")
    _require(prepare_hash == sha256_file(prepare_path), "prepare manifest changed during extraction")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "extract_shard",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "hostname": socket.gethostname(),
        "input": {"rows": _source_record(rows_path), "prepare_manifest": _source_record(prepare_path)},
        "contract": {"sha256": contract_hash, "payload": contract_payload},
        "model": {
            "checkpoint": _source_record(args.checkpoint),
            "config": _source_record(args.config),
            "layer": 33,
            "pooling": "separate chain residue means concatenated",
            "feature_dimension": FEATURE_DIMENSION,
            "feature_dtype": "float16",
        },
        "sharding": {
            "rule": "class_source U and row_index modulo shard_count equals shard_index",
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "rows": int(len(shard)),
            "row_index_sha256": hashlib.sha256(arrays["row_index"].astype("<i8", copy=False).tobytes()).hexdigest(),
            "cache_uid_sha256": sha256_text("\n".join(observed_uids)),
        },
        "runtime": {
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "batch_size": int(args.batch_size),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": sys.version,
        },
        "output": dict(_source_record(output_path), compressed=bool(args.compressed), shape=list(features.shape)),
    }
    old_cache._atomic_write_json(manifest_path, manifest)
    print(json.dumps({"stage": "extract_shard", "status": "created", "shard_index": args.shard_index, "rows": len(shard), "output": str(output_path)}, sort_keys=True))


def _load_old_feature_sources(prepare_manifest):
    sources = prepare_manifest["sources"]
    old_rows_path = Path(sources["old_rows"]["path"])
    old_features_path = Path(sources["old_features"]["path"])
    _require(sha256_file(old_rows_path) == sources["old_rows"]["sha256"], "old rows changed after prepare")
    _require(sha256_file(old_features_path) == sources["old_features"]["sha256"], "old features changed after prepare")
    old_rows = read_string_csv(old_rows_path)
    archive = np.load(str(old_features_path), allow_pickle=False)
    _require(FEATURE_NAME in archive.files, "old cache lacks MINT features")
    features = np.asarray(archive[FEATURE_NAME], dtype=np.float32)
    _require(features.shape == (len(old_rows), FEATURE_DIMENSION), "old cache shape mismatch")
    _require(np.array_equal(np.asarray(archive["cache_uid"]).astype(str), old_rows["cache_uid"].to_numpy(dtype=str)), "old cache row order mismatch")
    return old_rows, features, archive


def run_merge(args):
    started = time.time()
    frame, prepare_manifest, rows_path, prepare_path = load_prepared(args.input_dir)
    _require(args.shards_dir.is_dir(), "shard directory does not exist")
    output_dir = old_cache._ensure_private_directory(args.output_dir, must_be_new=True)
    features = np.empty((len(frame), FEATURE_DIMENSION), dtype=FEATURE_DTYPE)
    covered = np.zeros(len(frame), dtype=bool)

    old_rows, old_features, old_archive = _load_old_feature_sources(prepare_manifest)
    try:
        reuse = frame["class_source"].isin(["P", "N"])
        reuse_indices = pd.to_numeric(frame.loc[reuse, "source_cache_row_index"], errors="raise").to_numpy(dtype=np.int64)
        _require(np.array_equal(
            old_rows.iloc[reuse_indices]["cache_uid"].to_numpy(dtype=str),
            frame.loc[reuse, "source_cache_uid"].to_numpy(dtype=str),
        ), "prepared P/N source-cache identity mismatch")
        output_indices = pd.to_numeric(frame.loc[reuse, "row_index"], errors="raise").to_numpy(dtype=np.int64)
        cast_values = old_features[reuse_indices].astype(FEATURE_DTYPE)
        features[output_indices] = cast_values
        covered[output_indices] = True
        max_cast_error = float(np.max(np.abs(old_features[reuse_indices] - cast_values.astype(np.float32))))
    finally:
        old_archive.close()

    expected_contract = None
    shard_records = []
    for index in range(args.shard_count):
        shard_path = args.shards_dir / shard_filename(index, args.shard_count)
        shard_manifest_path = args.shards_dir / shard_manifest_filename(index, args.shard_count)
        _require(shard_path.is_file() and shard_manifest_path.is_file(), "missing shard {}".format(index))
        manifest = read_json(shard_manifest_path)
        _require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("stage") == "extract_shard", "invalid shard manifest")
        _require(manifest.get("output", {}).get("sha256") == sha256_file(shard_path), "shard hash mismatch")
        _require(manifest.get("input", {}).get("rows", {}).get("sha256") == sha256_file(rows_path), "shard row source mismatch")
        contract = manifest.get("contract", {}).get("sha256")
        if expected_contract is None:
            expected_contract = contract
        _require(contract == expected_contract, "shards have different feature contracts")
        expected = select_shard_rows(frame, index, args.shard_count)
        with np.load(str(shard_path), allow_pickle=False) as shard:
            _require(set(shard.files) == {"row_index", "cache_uid", FEATURE_NAME}, "shard schema mismatch")
            row_index = np.asarray(shard["row_index"], dtype=np.int64)
            _require(np.array_equal(row_index, pd.to_numeric(expected["row_index"]).to_numpy(dtype=np.int64)), "shard assignment mismatch")
            _require(np.array_equal(np.asarray(shard["cache_uid"]).astype(str), expected["cache_uid"].to_numpy(dtype=str)), "shard UID mismatch")
            values = np.asarray(shard[FEATURE_NAME], dtype=FEATURE_DTYPE)
            _require(values.shape == (len(expected), FEATURE_DIMENSION), "shard shape mismatch")
            _require(not bool(covered[row_index].any()), "duplicate feature coverage")
            features[row_index] = values
            covered[row_index] = True
        shard_records.append({"index": index, "rows": int(len(expected)), "npz": _source_record(shard_path), "manifest": _source_record(shard_manifest_path), "hostname": manifest.get("hostname")})

    _require(bool(covered.all()), "merged features do not cover every row")
    _require(bool(np.isfinite(features).all()), "merged features are non-finite")
    # Prove that no P/N representation was recomputed or perturbed except for
    # the declared float32 -> float16 storage cast.
    old_rows2, old_features2, old_archive2 = _load_old_feature_sources(prepare_manifest)
    try:
        reproduced = np.array_equal(features[output_indices], old_features2[reuse_indices].astype(FEATURE_DTYPE))
    finally:
        old_archive2.close()
    _require(reproduced, "P/N layer-33 reuse failed exact float16 reproduction")

    arrays = {FEATURE_NAME: features}
    for column in NPZ_METADATA_COLUMNS:
        if column == "row_index":
            arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.int64)
        elif column == "weak_label":
            arrays[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.int8)
        else:
            arrays[column] = frame[column].to_numpy(dtype=str)
    output_path = output_dir / FEATURE_FILENAME
    old_cache._atomic_write_npz(output_path, arrays, compressed=args.compressed)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "merge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "input": {
            "rows": _source_record(rows_path),
            "prepare_manifest": _source_record(prepare_path),
            "shards": shard_records,
        },
        "rows": summarize_rows(frame),
        "features": {
            "name": FEATURE_NAME,
            "shape": list(features.shape),
            "dtype": "float16",
            "layer": 33,
            "pooling": "separate mean over all residues of chain1 and chain2, concatenated",
            "P_N_source": "immutable existing mint_weak_cache_v1 float32 layer-33 cache",
            "U_source": "new frozen-MINT inference",
        },
        "layer33_P_N_reproduction": {
            "rows": int(reuse.sum()),
            "exact_after_declared_float16_cast": bool(reproduced),
            "max_absolute_float32_to_float16_cast_error": max_cast_error,
            "independent_reinference_performed": False,
        },
        "contract_sha256": expected_contract,
        "output": dict(_source_record(output_path), compressed=bool(args.compressed), keys=sorted(arrays)),
        "environment": {"hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    old_cache._atomic_write_json(manifest_path, manifest)
    print(json.dumps({"stage": "merge", "rows": len(frame), "shape": list(features.shape), "output": str(output_path)}, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--u-pool", required=True, type=Path)
    prepare.add_argument("--u-manifest", required=True, type=Path)
    prepare.add_argument("--old-rows", required=True, type=Path)
    prepare.add_argument("--old-rows-manifest", required=True, type=Path)
    prepare.add_argument("--old-features", required=True, type=Path)
    prepare.add_argument("--old-features-manifest", required=True, type=Path)
    prepare.add_argument("--sequence-zip", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--u-per-positive", type=int, default=5)
    prepare.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    prepare.add_argument("--all-u", action="store_true")

    extract = commands.add_parser("extract-shard")
    extract.add_argument("--input-dir", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--checkpoint", required=True, type=Path)
    extract.add_argument("--config", required=True, type=Path)
    extract.add_argument("--shard-index", required=True, type=int)
    extract.add_argument("--shard-count", required=True, type=int)
    extract.add_argument("--device", default="cuda:0")
    extract.add_argument("--batch-size", type=int, default=64)
    extract.add_argument("--compressed", action="store_true")
    extract.add_argument("--resume", action="store_true")

    merge = commands.add_parser("merge")
    merge.add_argument("--input-dir", required=True, type=Path)
    merge.add_argument("--shards-dir", required=True, type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    merge.add_argument("--shard-count", required=True, type=int)
    merge.add_argument("--compressed", action="store_true")
    args = parser.parse_args(argv)
    _require(args.command is not None, "a subcommand is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "extract-shard":
        run_extract_shard(args)
    elif args.command == "merge":
        run_merge(args)
    else:
        raise ValueError("unexpected command")


if __name__ == "__main__":
    main()
