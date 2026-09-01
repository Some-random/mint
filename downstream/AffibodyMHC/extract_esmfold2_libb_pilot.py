"""Run the label-free LibB ESMFold2 intermediate-feature pilot.

This script calls ``EsmFold2Model.forward`` directly, so it runs the folding
trunk and distogram head but never coordinate diffusion.  It intentionally
does not convert the checkpoint's 64 distogram categories to Angstroms: the
public checkpoint metadata does not define output-bin edges.

The biological pilot contains six sequence pairs.  The reference pair is run
twice with the same RNG seed to distinguish mutation-induced feature changes
from extraction nondeterminism.  Retention values are not read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import EsmFold2Model

# Permit both ``python -m downstream...`` and direct script execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from downstream.AffibodyMHC.esmfold2_multichain_features import (
    prepare_multichain_trunk_inputs,
)

EXPECTED_CHAIN_LENGTHS = (270, 58)
PEPTIDE_LENGTH = 9
FSX_ROOT = Path("/fsx")
REPO_ROOT = Path(__file__).resolve().parents[2]
SAFE_OUTPUT_ROOT = REPO_ROOT / "private_data/structure_pilot"
SAFE_OUTPUT_PREFIX = "esmfold2_libb_pilot_"
DISTOGRAM_BIN_SEMANTICS = (
    "64 categorical distance-distribution bins; physical edges are not defined "
    "by the public biohub/ESMFold2-hf checkpoint metadata"
)


def _resolved_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"{label} must resolve under {resolved_root}, found {resolved}"
        ) from error
    return resolved


def _validated_output_dir(path: Path) -> Path:
    """Resolve one narrowly named pilot directory safe for optional deletion."""

    if path.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {path}")
    resolved = _resolved_under(path, SAFE_OUTPUT_ROOT, "output directory")
    relative = resolved.relative_to(SAFE_OUTPUT_ROOT.resolve())
    if len(relative.parts) != 1:
        raise ValueError(
            "output directory must be one direct child of "
            f"{SAFE_OUTPUT_ROOT.resolve()}, found {resolved}"
        )
    if not relative.name.startswith(SAFE_OUTPUT_PREFIX):
        raise ValueError(
            f"output directory name must start with {SAFE_OUTPUT_PREFIX!r}, "
            f"found {relative.name!r}"
        )
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Override checkpoint num_loops. Omit for the checkpoint default.",
    )
    parser.add_argument(
        "--use-kernels",
        action="store_true",
        help="Request optional fused triangle kernels during model loading.",
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action="store_true",
        help="Require PyTorch deterministic CUDA algorithms (debug/QA mode).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing pilot output directory.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _set_common_seed(seed: int) -> None:
    # ESMFold2 samples its initial pair state and forces LM dropout even under
    # eval().  Resetting one common seed immediately before every forward keeps
    # those random draws matched across sequence variants.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_pair(pair: dict[str, Any], reference: dict[str, Any]) -> tuple[str, str]:
    chains = pair.get("chains")
    if not isinstance(chains, list) or len(chains) != 2:
        raise ValueError(f"{pair.get('pilot_id')}: expected exactly two chains")
    sequence_a = str(chains[0]["sequence"])
    sequence_b = str(chains[1]["sequence"])
    if tuple(map(len, (sequence_a, sequence_b))) != EXPECTED_CHAIN_LENGTHS:
        raise ValueError(
            f"{pair.get('pilot_id')}: unexpected chain lengths "
            f"{(len(sequence_a), len(sequence_b))}"
        )
    if not sequence_a.endswith("SLL" + pair["peptide_design_code"] + "ITQV"):
        raise ValueError(
            f"{pair.get('pilot_id')}: peptide code does not map to chain A"
        )
    affibody_positions = (6, 10, 13, 14, 17)
    observed_code = "".join(sequence_b[position - 1] for position in affibody_positions)
    if observed_code != pair["affibody_design_code"]:
        raise ValueError(
            f"{pair.get('pilot_id')}: Affibody code does not map to chain B"
        )

    expected_a = []
    expected_b = []
    reference_a = reference["chains"][0]["sequence"]
    reference_b = reference["chains"][1]["sequence"]
    for index, (left, right) in enumerate(zip(reference_a, sequence_a), start=1):
        if left != right:
            expected_a.append(
                {
                    "sequence_position_1_based": index,
                    "reference_amino_acid": left,
                    "variant_amino_acid": right,
                }
            )
    for index, (left, right) in enumerate(zip(reference_b, sequence_b), start=1):
        if left != right:
            expected_b.append(
                {
                    "sequence_position_1_based": index,
                    "reference_amino_acid": left,
                    "variant_amino_acid": right,
                }
            )
    declared = pair.get("differences_from_reference", {})
    if expected_a != declared.get("chain_A", []) or expected_b != declared.get(
        "chain_B", []
    ):
        raise ValueError(
            f"{pair.get('pilot_id')}: declared mutations do not match sequences"
        )
    return sequence_a, sequence_b


def _derive_interface_indices(
    forward_kwargs: dict[str, Any], sequence_a: str, sequence_b: str
) -> tuple[torch.Tensor, torch.Tensor]:
    asym_id = forward_kwargs["asym_id"][0]
    residue_index = forward_kwargs["residue_index"][0]
    attention_mask = forward_kwargs["attention_mask"][0].bool()

    chain_a = torch.nonzero(attention_mask & asym_id.eq(0), as_tuple=False).flatten()
    chain_b = torch.nonzero(attention_mask & asym_id.eq(1), as_tuple=False).flatten()
    if chain_a.numel() != len(sequence_a) or chain_b.numel() != len(sequence_b):
        raise ValueError("prepared chain IDs do not match the manifest lengths")
    if not torch.equal(
        residue_index[chain_a], torch.arange(len(sequence_a), device=chain_a.device)
    ):
        raise ValueError("chain A residue numbering is not local and contiguous")
    if not torch.equal(
        residue_index[chain_b], torch.arange(len(sequence_b), device=chain_b.device)
    ):
        raise ValueError("chain B residue numbering is not local and contiguous")
    peptide = chain_a[residue_index[chain_a] >= len(sequence_a) - PEPTIDE_LENGTH]
    if peptide.numel() != PEPTIDE_LENGTH:
        raise ValueError(
            "could not derive the nine peptide residues from chain A metadata"
        )
    return peptide, chain_b


def _to_numpy(tensor: torch.Tensor, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    value = tensor.detach()
    # NumPy has no native bfloat16 dtype. Convert only at the serialization
    # boundary; model computation remains bfloat16/fp32 as configured.
    if value.dtype == torch.bfloat16:
        value = value.float()
    array = value.cpu().numpy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def _feature_bundle(
    output: Any, peptide_indices: torch.Tensor, affibody_indices: torch.Tensor
) -> dict[str, np.ndarray]:
    # Indexing is explicitly peptide x Affibody. Pair states are directional;
    # save both aligned directions and their symmetric/antisymmetric parts.
    logits_ab = output.distogram_logits[0][peptide_indices][:, affibody_indices, :]
    probabilities = torch.softmax(logits_ab.float(), dim=-1)
    pair_ab = output.pair_states[0][peptide_indices][:, affibody_indices, :]
    pair_ba = (
        output.pair_states[0][affibody_indices][:, peptide_indices, :]
        .transpose(0, 1)
        .contiguous()
    )
    pair_symmetric = 0.5 * (pair_ab + pair_ba)
    pair_antisymmetric = 0.5 * (pair_ab - pair_ba)
    single_peptide = output.single_inputs[0, peptide_indices, :]
    single_affibody = output.single_inputs[0, affibody_indices, :]

    return {
        "distogram_probabilities": _to_numpy(probabilities, np.float16),
        "pair_states_ab": _to_numpy(pair_ab, np.float16),
        "pair_states_ba_aligned": _to_numpy(pair_ba, np.float16),
        "pair_states_symmetric": _to_numpy(pair_symmetric, np.float16),
        "pair_states_antisymmetric": _to_numpy(pair_antisymmetric, np.float16),
        "single_inputs_peptide": _to_numpy(single_peptide, np.float16),
        "single_inputs_affibody": _to_numpy(single_affibody, np.float16),
        "distogram_mean": _to_numpy(probabilities.mean(dim=(0, 1)), np.float32),
        "pair_states_symmetric_mean": _to_numpy(
            pair_symmetric.mean(dim=(0, 1)), np.float32
        ),
        "pair_states_antisymmetric_mean": _to_numpy(
            pair_antisymmetric.mean(dim=(0, 1)), np.float32
        ),
    }


def _array_delta(current: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    left = current.astype(np.float64, copy=False)
    right = reference.astype(np.float64, copy=False)
    difference = left - right
    denominator = np.linalg.norm(right.ravel())
    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "mean_absolute": float(np.mean(np.abs(difference))),
        "l2": float(np.linalg.norm(difference.ravel())),
        "relative_l2": float(
            np.linalg.norm(difference.ravel()) / max(denominator, 1e-12)
        ),
    }


def _run_one(
    model: EsmFold2Model,
    pair: dict[str, Any],
    reference: dict[str, Any],
    device: torch.device,
    seed: int,
    num_loops: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sequence_a, sequence_b = _validate_pair(pair, reference)
    preparation_start = time.perf_counter()
    prepared = prepare_multichain_trunk_inputs((sequence_a, sequence_b), device=device)
    preparation_seconds = time.perf_counter() - preparation_start
    peptide_indices, affibody_indices = _derive_interface_indices(
        prepared.forward_kwargs, sequence_a, sequence_b
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    _set_common_seed(seed)
    forward_start = time.perf_counter()
    call_kwargs = dict(prepared.forward_kwargs)
    if num_loops is not None:
        call_kwargs["num_loops"] = num_loops
    with torch.inference_mode():
        output = model(**call_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start

    expected_shapes = {
        "distogram_logits": (1, 328, 328, 64),
        "pair_states": (1, 328, 328, 256),
        "single_inputs": (1, 328, 451),
    }
    observed_shapes = {
        name: tuple(getattr(output, name).shape) for name in expected_shapes
    }
    if observed_shapes != expected_shapes:
        raise ValueError(f"unexpected ESMFold2 output shapes: {observed_shapes}")

    features = _feature_bundle(output, peptide_indices, affibody_indices)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    metadata = {
        "pilot_id": pair["pilot_id"],
        "peptide_design_code": pair["peptide_design_code"],
        "affibody_design_code": pair["affibody_design_code"],
        "sequence_pair_sha256": pair["sequence_pair_sha256"],
        "chain_lengths": [len(sequence_a), len(sequence_b)],
        "peptide_sequence": sequence_a[-PEPTIDE_LENGTH:],
        "peptide_token_indices_0_based": peptide_indices.detach().cpu().tolist(),
        "affibody_token_indices_0_based": affibody_indices.detach().cpu().tolist(),
        "preparation_seconds": preparation_seconds,
        "forward_seconds": forward_seconds,
        "peak_gpu_memory_bytes": peak_bytes,
        "output_shapes": {key: list(value) for key, value in observed_shapes.items()},
        "output_dtypes": {
            name: str(getattr(output, name).dtype) for name in expected_shapes
        },
        "saved_feature_shapes": {
            key: list(value.shape) for key, value in features.items()
        },
        "saved_feature_dtypes": {
            key: str(value.dtype) for key, value in features.items()
        },
    }
    del output, prepared
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, metadata


def main() -> None:
    args = _parse_args()
    checkpoint = _resolved_under(args.checkpoint, FSX_ROOT, "checkpoint")
    inputs = _resolved_under(args.inputs, FSX_ROOT, "input manifest")
    output_dir = _validated_output_dir(args.output_dir)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not inputs.is_file():
        raise FileNotFoundError(inputs)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists; pass --overwrite to replace this pilot output"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with inputs.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    pairs = manifest.get("pairs", [])
    if len(pairs) != 6:
        raise ValueError(f"expected six pilot pairs, found {len(pairs)}")
    reference = pairs[0]
    if reference.get("pilot_id") != "libb_ref_mw_nnyyf":
        raise ValueError("the first pilot row must be the MW/NNYYF reference")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the official 6.6B ESMFold2 pilot requires a CUDA device")
    torch.cuda.set_device(device)
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    load_start = time.perf_counter()
    model = EsmFold2Model.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_kernels=args.use_kernels,
    )
    model.eval().to(device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start

    run_specs = [(pair, "primary") for pair in pairs]
    run_specs.insert(1, (reference, "same_seed_repeat"))
    records: list[dict[str, Any]] = []
    features_by_run: dict[str, dict[str, np.ndarray]] = {}
    for pair, run_kind in run_specs:
        run_id = pair["pilot_id"] + ("__repeat" if run_kind != "primary" else "")
        print(f"running {run_id}", flush=True)
        features, metadata = _run_one(
            model=model,
            pair=pair,
            reference=reference,
            device=device,
            seed=args.seed,
            num_loops=args.num_loops,
        )
        artifact = output_dir / f"{run_id}.npz"
        np.savez_compressed(artifact, **features)
        metadata.update(
            {
                "run_id": run_id,
                "run_kind": run_kind,
                "artifact": artifact.name,
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": _sha256(artifact),
            }
        )
        records.append(metadata)
        features_by_run[run_id] = features
        print(
            f"finished {run_id}: forward={metadata['forward_seconds']:.3f}s, "
            f"artifact={metadata['artifact_bytes'] / 1024**2:.2f} MiB",
            flush=True,
        )

    reference_id = reference["pilot_id"]
    repeat_id = reference_id + "__repeat"
    comparison_features = (
        "distogram_probabilities",
        "pair_states_symmetric",
        "pair_states_antisymmetric",
        "single_inputs_peptide",
        "single_inputs_affibody",
    )
    comparisons: dict[str, Any] = {}
    for run_id, features in features_by_run.items():
        if run_id == reference_id:
            continue
        comparisons[run_id] = {
            key: _array_delta(features[key], features_by_run[reference_id][key])
            for key in comparison_features
        }

    repeat_max = max(
        comparisons[repeat_id][feature]["max_absolute"]
        for feature in comparison_features
    )
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "label-free ESMFold2 mapping, sensitivity, runtime, memory, and storage pilot",
        "retention_labels_read": False,
        "coordinate_diffusion_run": False,
        "distogram_bin_semantics": DISTOGRAM_BIN_SEMANTICS,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _sha256(checkpoint / "config.json"),
        "input_manifest": str(inputs),
        "input_manifest_sha256": _sha256(inputs),
        "environment": {
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "settings": {
            "device": str(device),
            "dtype": "torch.bfloat16",
            "seed_reset_before_every_forward": args.seed,
            "num_loops_argument": args.num_loops,
            "checkpoint_num_loops": int(model.config.num_loops),
            "actual_trunk_steps": int(
                max(
                    1,
                    (
                        args.num_loops
                        if args.num_loops is not None
                        else model.config.num_loops
                    )
                    + 1,
                )
            ),
            "use_kernels": bool(args.use_kernels),
            "deterministic_algorithms": bool(args.deterministic_algorithms),
            "batch_size": 1,
        },
        "model_load_seconds": load_seconds,
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "same_seed_repeat_max_absolute_difference": repeat_max,
        "same_seed_repeat_is_exact_in_saved_features": bool(repeat_max == 0.0),
        "runs": records,
        "differences_from_reference": comparisons,
    }
    _json_dump(output_dir / "pilot_summary.json", summary)
    with (output_dir / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in sorted(output_dir.glob("*.npz")):
            handle.write(f"{_sha256(path)}  {path.name}\n")
        handle.write(
            f"{_sha256(output_dir / 'pilot_summary.json')}  pilot_summary.json\n"
        )
    print(f"wrote {output_dir / 'pilot_summary.json'}", flush=True)


if __name__ == "__main__":
    # Keep all potentially large caches/temp files outside the full root disk.
    if os.environ.get("TMPDIR", "").startswith("/tmp"):
        print(
            "warning: TMPDIR points to root-backed /tmp; use an /fsx path",
            file=sys.stderr,
        )
    main()
