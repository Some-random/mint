#!/usr/bin/env python3
"""Stream LibB candidates through the locked RDE- or StaB-derived predictor.

This command never reads weak labels or retention measurements.  It reuses the
already trained five-seed readout checkpoints and writes scalar current-pair
scores.  Candidate partitions are checkpointed in moderately sized chunks, so
multi-hour exhaustive runs can be resumed without recomputing completed work.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC.libb_structural_readout import (  # noqa: E402
    build_matched_readout,
)


SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
INPUT_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
    "chain1_sequence",
    "chain2_sequence",
    "sequence_pair_sha256",
)
OUTPUT_PREFIX_COLUMNS = (
    "candidate_row_index",
    "pair_uid",
    "peptide_design_code",
    "affibody_design_code",
)
RDE_MEMBER_NAMES = tuple(f"rde_network_fold{fold}_designed_ordered" for fold in range(3))
RDE_MODEL_NAME = "rde_network_designed_3fold_ensemble"
STAB_MODEL_NAME = "stab_designed_ordered"
RDE_DESIGNED_PATCH_INDICES = (4, 5, 0, 1, 2, 6, 3)
EXPECTED_ROWS = 4_447_848


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _sigmoid(values: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(values)


def _clipped_logit(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(1e-7, 1.0 - 1e-7)
    return torch.log(values) - torch.log1p(-values)


def _load_readout(path: Path, expected_name: str, input_dim: int,
                  architecture: dict[str, Any], device: torch.device) -> torch.nn.Module:
    payload = _torch_load(path)
    _require(payload.get("model") == expected_name, f"wrong model in {path}")
    _require(int(payload.get("input_dim", -1)) == input_dim, f"wrong input dimension in {path}")
    model = build_matched_readout(input_dim, architecture)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.requires_grad_(False).eval().to(device)


def _architecture(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    architecture = payload.get("architecture")
    _require(isinstance(architecture, dict), "readout config lacks architecture")
    return architecture


def _candidate_rows(frame: pd.DataFrame, row_class) -> list[Any]:
    rows = []
    for record in frame.itertuples(index=False):
        rows.append(
            row_class(
                row_index=int(record.candidate_row_index),
                row_id=str(record.pair_uid),
                split="eval",
                chain1_sequence=str(record.chain1_sequence),
                chain2_sequence=str(record.chain2_sequence),
                sequence_pair_sha256=str(record.sequence_pair_sha256),
            )
        )
    return rows


class RDEScorer:
    name = RDE_MODEL_NAME

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from downstream.AffibodyMHC import extract_rde_libb_features as extractor
        from downstream.AffibodyMHC import rde_libb_fixed_crystal as features

        self.extractor = extractor
        self.features = features
        extractor.verify_upstream(
            args.rde_root, args.rde_checkpoint, args.rde_network_checkpoint, "rde_network"
        )
        with args.canonical_rows.open("r", encoding="utf-8") as handle:
            canonical = features.validate_canonical_payload(json.load(handle), expected_total=30_768)
        parsed, _ = extractor.parse_crystal_copy(args.rde_root, args.pdb)
        self.template = features.build_fixed_crystal_template(parsed, canonical[0], patch_size=128)
        _, self.network_models = extractor.load_frozen_models(
            args.rde_root,
            args.rde_checkpoint,
            args.rde_network_checkpoint,
            "rde_network",
            device,
        )
        architecture = _architecture(args.rde_readout_config)
        self.architecture = architecture
        self.readouts: dict[tuple[int, int], torch.nn.Module] = {}
        for fold, member in enumerate(RDE_MEMBER_NAMES):
            for seed in SEEDS:
                path = args.rde_checkpoint_dir / f"{member}__seed{seed}.pt"
                _require(path.is_file(), f"missing RDE readout {path}")
                self.readouts[(fold, seed)] = _load_readout(
                    path, member, 896, architecture, device
                )
        self.device = device

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        rows = _candidate_rows(frame, self.features.CanonicalRow)
        examples = [self.features.prepare_current_pair(self.template, row) for row in rows]
        batch = self.features.collate_current_pairs(examples)
        encoded = self.extractor.encode_batch(
            None, self.network_models, batch, "rde_network", self.device
        )
        fold_vectors = []
        for fold in range(3):
            context = encoded[f"rde_network_fold{fold}_context"]
            selected = context[:, RDE_DESIGNED_PATCH_INDICES, :].reshape(len(frame), 896)
            fold_vectors.append(torch.as_tensor(selected.astype(np.float32), device=self.device))
        seed_probabilities = []
        with torch.no_grad():
            for seed in SEEDS:
                logits = torch.stack(
                    [self.readouts[(fold, seed)](fold_vectors[fold]) for fold in range(3)],
                    dim=1,
                ).mean(dim=1)
                seed_probabilities.append(_sigmoid(logits))
        return torch.stack(seed_probabilities, dim=1).cpu().numpy().astype(np.float32)


class StaBScorer:
    name = STAB_MODEL_NAME

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from downstream.AffibodyMHC import stab_libb_current_pair_features as features

        self.features = features
        features.load_canonical_rows(args.canonical_rows, require_full_dataset=True)
        self.template = features.build_template_bundle(
            args.pdb, args.stab_root, features.ASSAY_RESOLVED_FRAGMENT, args.residue_mapping
        )
        self.backbone = features.load_pinned_vendor(args.stab_root, args.stab_checkpoint)
        self.backbone.requires_grad_(False).eval().to(device)
        architecture = _architecture(args.stab_readout_config)
        self.architecture = architecture
        self.readouts = {}
        for seed in SEEDS:
            path = args.stab_checkpoint_dir / f"{STAB_MODEL_NAME}__seed{seed}.pt"
            _require(path.is_file(), f"missing StaB readout {path}")
            self.readouts[seed] = _load_readout(
                path, STAB_MODEL_NAME, 1050, architecture, device
            )
        self.device = device

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        rows = _candidate_rows(frame, self.features.CanonicalRow)
        arrays = self.features.extract_feature_batch(
            self.backbone, rows, self.template, self.device
        )
        vector = torch.as_tensor(
            arrays["residue_features"].reshape(len(frame), 1050).astype(np.float32),
            device=self.device,
        )
        with torch.no_grad():
            probabilities = [_sigmoid(self.readouts[seed](vector)) for seed in SEEDS]
        return torch.stack(probabilities, dim=1).cpu().numpy().astype(np.float32)


def _validate_input(frame: pd.DataFrame, peptide: str) -> None:
    _require(tuple(frame.columns) == INPUT_COLUMNS, "scoring-input columns changed")
    _require(len(frame) > 0, "empty candidate input")
    _require(frame["peptide_design_code"].eq(peptide).all(), "mixed peptide file")
    _require(not frame["pair_uid"].duplicated().any(), "duplicate candidate pair")
    _require(
        frame["candidate_row_index"].to_numpy(dtype=int).tolist() == list(range(len(frame))),
        "candidate row indices are not contiguous",
    )


def _chunk_path(output_dir: Path, peptide: str, start: int, stop: int) -> Path:
    return output_dir / f"peptide_{peptide}" / f"rows_{start:06d}_{stop - 1:06d}.csv.gz"


def _validate_completed_chunk(path: Path, expected: pd.DataFrame, model_name: str) -> None:
    observed = pd.read_csv(path)
    required = list(OUTPUT_PREFIX_COLUMNS) + [
        *(f"score_seed_{seed}" for seed in SEEDS),
        "score_mean_probability",
        "score_mean_logit",
        "score_seed_sd",
        "model",
    ]
    _require(observed.columns.tolist() == required, f"bad completed chunk {path}")
    _require(observed["pair_uid"].astype(str).tolist() == expected["pair_uid"].astype(str).tolist(),
             f"completed chunk membership changed: {path}")
    _require(observed["model"].eq(model_name).all(), f"wrong model in {path}")


def _write_chunk(path: Path, frame: pd.DataFrame, probabilities: np.ndarray,
                 model_name: str) -> None:
    _require(probabilities.shape == (len(frame), len(SEEDS)), "score matrix shape changed")
    _require(bool(np.isfinite(probabilities).all()), "scores are non-finite")
    _require(bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()),
             "score outside [0,1]")
    output = frame[list(OUTPUT_PREFIX_COLUMNS)].copy()
    for column, seed in enumerate(SEEDS):
        output[f"score_seed_{seed}"] = probabilities[:, column]
    clipped = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    mean_logit = logits.mean(axis=1)
    output["score_mean_probability"] = probabilities.mean(axis=1)
    output["score_mean_logit"] = 1.0 / (1.0 + np.exp(-mean_logit))
    output["score_seed_sd"] = probabilities.std(axis=1, ddof=1)
    output["model"] = model_name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    output.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "compresslevel": 6},
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("rde", "stab"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peptides", required=True, help="Comma-separated peptide codes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=16_384)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--slice-index", type=int, default=0)
    parser.add_argument("--num-slices", type=int, default=1)
    parser.add_argument(
        "--canonical-rows",
        type=Path,
        default=REPO_ROOT / "private_data/derived/esmfold2_libb_canonical_rows_provider_revision_120_v1/rows.json",
    )
    parser.add_argument("--pdb", type=Path, default=REPO_ROOT / "nyeso_xx133_complex.pdb")
    parser.add_argument(
        "--residue-mapping", type=Path,
        default=REPO_ROOT / "private_data/derived/libb_fixed_crystal_contract_provider_revision_120_v1/residue_mapping.json",
    )
    parser.add_argument("--rde-root", type=Path, default=REPO_ROOT / "private_data/vendor/rde-ppi")
    parser.add_argument(
        "--rde-checkpoint", type=Path,
        default=REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/RDE.pt",
    )
    parser.add_argument(
        "--rde-network-checkpoint", type=Path,
        default=REPO_ROOT / "private_data/vendor/rde-ppi/trained_models/DDG_RDE_Network_30k.pt",
    )
    parser.add_argument(
        "--rde-checkpoint-dir", type=Path,
        default=REPO_ROOT / "private_data/experiments/rde_libb_provider_revision_120_full_readouts_v2_common_env/rde_network_designed/final_checkpoints",
    )
    parser.add_argument(
        "--rde-readout-config", type=Path,
        default=REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_frozen_readouts_v1.json",
    )
    parser.add_argument("--stab-root", type=Path, default=REPO_ROOT / "private_data/vendor/StaB-ddG")
    parser.add_argument(
        "--stab-checkpoint", type=Path,
        default=REPO_ROOT / "private_data/vendor/StaB-ddG/model_ckpts/stabddg.pt",
    )
    parser.add_argument(
        "--stab-checkpoint-dir", type=Path,
        default=REPO_ROOT / "private_data/experiments/stab_libb_provider_revision_120_readout_full_v1/stab_designed_ordered/final_checkpoints",
    )
    parser.add_argument(
        "--stab-readout-config", type=Path,
        default=REPO_ROOT / "downstream/AffibodyMHC/configs/stab_libb_frozen_readouts_v1.json",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    _require(torch.cuda.is_available(), "CUDA is required")
    _require(args.batch_size > 0 and args.chunk_rows >= args.batch_size, "invalid batching")
    _require(args.num_slices > 0 and 0 <= args.slice_index < args.num_slices,
             "invalid candidate slice")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    private_root = (REPO_ROOT / "private_data").resolve()
    _require(input_dir.is_dir(), "scoring input directory does not exist")
    _require(output_dir != private_root and private_root in output_dir.parents,
             "output must be below private_data")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    peptides = tuple(value.strip() for value in args.peptides.split(",") if value.strip())
    _require(peptides and len(peptides) == len(set(peptides)), "bad peptide selection")
    _require(all(re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{2}", value) for value in peptides),
             "invalid peptide code")
    device = torch.device(args.device)
    scorer = RDEScorer(args, device) if args.family == "rde" else StaBScorer(args, device)
    readout_config = (
        args.rde_readout_config if args.family == "rde" else args.stab_readout_config
    ).resolve()
    readout_checkpoint_dir = (
        args.rde_checkpoint_dir if args.family == "rde" else args.stab_checkpoint_dir
    ).resolve()
    checkpoint_paths = sorted(readout_checkpoint_dir.glob("*.pt"))
    expected_checkpoint_count = 15 if args.family == "rde" else 5
    _require(
        len(checkpoint_paths) == expected_checkpoint_count,
        f"expected {expected_checkpoint_count} readout checkpoints, found "
        f"{len(checkpoint_paths)}",
    )
    started = time.time()
    records = []
    for peptide in peptides:
        input_path = input_dir / f"peptide_{peptide}.csv.gz"
        _require(input_path.is_file(), f"missing scoring input {input_path}")
        frame = pd.read_csv(input_path, dtype={"pair_uid": str})
        _validate_input(frame, peptide)
        full_rows = len(frame)
        slice_start = full_rows * args.slice_index // args.num_slices
        slice_stop = full_rows * (args.slice_index + 1) // args.num_slices
        frame = frame.iloc[slice_start:slice_stop].copy()
        if args.limit is not None:
            _require(args.limit > 0, "limit must be positive")
            frame = frame.iloc[: args.limit].copy()
        peptide_started = time.time()
        for start in range(0, len(frame), args.chunk_rows):
            stop = min(start + args.chunk_rows, len(frame))
            selected = frame.iloc[start:stop].copy()
            global_start = int(selected["candidate_row_index"].iloc[0])
            global_stop = int(selected["candidate_row_index"].iloc[-1]) + 1
            path = _chunk_path(output_dir, peptide, global_start, global_stop)
            if path.exists():
                _validate_completed_chunk(path, selected, scorer.name)
                continue
            blocks = []
            for offset in range(0, len(selected), args.batch_size):
                batch = selected.iloc[offset : offset + args.batch_size]
                blocks.append(scorer.score(batch))
            _write_chunk(path, selected, np.concatenate(blocks, axis=0), scorer.name)
            print(json.dumps({"family": args.family, "peptide": peptide,
                              "slice_index": args.slice_index,
                              "slice_rows_complete": stop,
                              "slice_rows_total": len(frame),
                              "global_rows_complete_through": global_stop,
                              "global_rows_total": full_rows,
                              "elapsed_seconds": round(time.time() - peptide_started, 3)}),
                  flush=True)
        records.append({"peptide": peptide, "full_rows": int(full_rows),
                        "slice_start": int(slice_start), "slice_stop": int(slice_stop),
                        "rows": int(len(frame)),
                        "elapsed_seconds": round(time.time() - peptide_started, 6)})
    manifest = {
        "schema_version": "libb-fixed-structure-candidate-scores-v1",
        "family": args.family,
        "model": scorer.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "input_manifest_sha256": _sha256_file(input_dir / "manifest.json"),
        "readout_seeds": list(SEEDS),
        "slice_index": int(args.slice_index),
        "num_slices": int(args.num_slices),
        "peptides": records,
        "labels_read": False,
        "retention_read": False,
        "readout_provenance": {
            "config_path": str(readout_config),
            "config_sha256": _sha256_file(readout_config),
            "readout_mode": str(scorer.architecture.get("readout_mode", "legacy_default")),
            "checkpoint_directory": str(readout_checkpoint_dir),
            "checkpoints": [
                {
                    "filename": path.name,
                    "sha256": _sha256_file(path),
                }
                for path in checkpoint_paths
            ],
        },
        "producer": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    manifest_path = output_dir / (
        f"worker_{args.family}_slice{args.slice_index:02d}of{args.num_slices:02d}_"
        f"{'_'.join(peptides)}.json"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
