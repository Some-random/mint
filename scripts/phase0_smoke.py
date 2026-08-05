#!/usr/bin/env python
"""Run the public MINT Phase 0 embedding and binary-head smoke tests."""

import argparse
import hashlib
import inspect
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from mint.helpers.extract import CSVDataset, CollateFn, MINTWrapper, load_config
from mint.helpers.predict import SimpleMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mlp-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("data/esm2_t33_650M_UR50D.json"))
    parser.add_argument("--csv", type=Path, default=Path("data/protein_sequences.csv"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def torch_load_compat(path, map_location):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(str(path), **kwargs)


def gibibytes(num_bytes):
    return round(num_bytes / (1024 ** 3), 3)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def repository_commit():
    repository = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()


def driver_versions():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        universal_newlines=True,
    )
    return sorted(set(line.strip() for line in output.splitlines() if line.strip()))


def validate_embeddings(embeddings, expected_shape):
    if tuple(embeddings.shape) != expected_shape:
        raise RuntimeError(
            "Expected embedding shape {}, got {}".format(expected_shape, tuple(embeddings.shape))
        )
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Embedding output contains non-finite values")


def run_embeddings(wrapper, dataset, device, batch_size, sep_chains, repeats):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=CollateFn(512),
        shuffle=False,
    )
    wrapper.sep_chains = sep_chains
    warmup_chains, warmup_chain_ids = next(iter(loader))
    with torch.inference_mode():
        wrapper(warmup_chains.to(device), warmup_chain_ids.to(device)).cpu()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    expected_dim = 2560 if sep_chains else 1280
    expected_shape = (len(dataset), expected_dim)
    timings = []
    embeddings = None
    repeat_max_abs_diff = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = []
        with torch.inference_mode():
            for chains, chain_ids in loader:
                output = wrapper(chains.to(device), chain_ids.to(device))
                outputs.append(output.cpu())
        torch.cuda.synchronize(device)
        current_embeddings = torch.cat(outputs)
        timings.append(time.perf_counter() - started)
        validate_embeddings(current_embeddings, expected_shape)
        if embeddings is None:
            embeddings = current_embeddings
        else:
            repeat_max_abs_diff = max(
                repeat_max_abs_diff,
                float(torch.max(torch.abs(embeddings - current_embeddings))),
            )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    device_free, device_total = torch.cuda.mem_get_info(device)
    result = {
        "batch_size": batch_size,
        "sep_chains": sep_chains,
        "shape": list(embeddings.shape),
        "repeats": repeats,
        "end_to_end_seconds": [round(value, 6) for value in timings],
        "mean_end_to_end_seconds": round(statistics.mean(timings), 6),
        "stdev_end_to_end_seconds": round(statistics.pstdev(timings), 6),
        "mean_seconds_per_pair": round(statistics.mean(timings) / len(dataset), 6),
        "process_baseline_allocated_gib": gibibytes(baseline_allocated),
        "process_baseline_reserved_gib": gibibytes(baseline_reserved),
        "process_peak_allocated_gib": gibibytes(peak_allocated),
        "process_peak_reserved_gib": gibibytes(peak_reserved),
        "process_incremental_peak_allocated_gib": gibibytes(
            max(0, peak_allocated - baseline_allocated)
        ),
        "process_incremental_peak_reserved_gib": gibibytes(
            max(0, peak_reserved - baseline_reserved)
        ),
        "device_free_gib_after_run": gibibytes(device_free),
        "device_total_gib": gibibytes(device_total),
        "repeat_max_abs_diff": repeat_max_abs_diff,
        "output_sha256": tensor_sha256(embeddings),
        "mean": float(embeddings.float().mean()),
        "std": float(embeddings.float().std()),
    }
    return result, embeddings


def run_binary_head(embeddings, checkpoint, device):
    model = SimpleMLP()
    model.load_state_dict(torch_load_compat(checkpoint, map_location="cpu"))
    model.eval()
    model.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(embeddings.to(device))).flatten().cpu()
    torch.cuda.synchronize(device)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Binary-head output contains non-finite values")
    return {
        "shape": list(probabilities.shape),
        "probabilities": [round(float(value), 8) for value in probabilities],
        "output_sha256": tensor_sha256(probabilities),
        "seconds": round(time.perf_counter() - started, 6),
        "process_incremental_peak_allocated_gib": gibibytes(
            max(0, torch.cuda.max_memory_allocated(device) - baseline_allocated)
        ),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least one")
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset = CSVDataset(
        str(args.csv),
        "Protein_Sequence_1",
        "Protein_Sequence_2",
    )
    if len(dataset) != 5:
        raise RuntimeError("The public smoke CSV should contain exactly five pairs")

    free_before_load, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    wrapper = MINTWrapper(
        load_config(str(args.config)),
        str(args.checkpoint),
        device=str(device),
    )
    wrapper.eval()
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - started
    model_load_peak_allocated = torch.cuda.max_memory_allocated(device)
    model_load_peak_reserved = torch.cuda.max_memory_reserved(device)
    free_after_load, _ = torch.cuda.mem_get_info(device)

    report = {
        "command": sys.argv,
        "hostname": socket.gethostname(),
        "git_commit": repository_commit(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "nvidia_driver_versions": driver_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "device_total_gib": gibibytes(total_memory),
        "device_free_gib_before_model_load": gibibytes(free_before_load),
        "device_free_gib_after_model_load": gibibytes(free_after_load),
        "dataset_rows": len(dataset),
        "artifacts": {
            "config": {"path": str(args.config), "sha256": file_sha256(args.config)},
            "csv": {"path": str(args.csv), "sha256": file_sha256(args.csv)},
            "checkpoint": {
                "path": str(args.checkpoint),
                "size_bytes": args.checkpoint.stat().st_size,
                "sha256": file_sha256(args.checkpoint),
            },
        },
        "model_load_seconds": round(model_load_seconds, 3),
        "process_model_allocated_gib": gibibytes(torch.cuda.memory_allocated(device)),
        "process_model_reserved_gib": gibibytes(torch.cuda.memory_reserved(device)),
        "process_model_load_peak_allocated_gib": gibibytes(model_load_peak_allocated),
        "process_model_load_peak_reserved_gib": gibibytes(model_load_peak_reserved),
        "embedding_runs": [],
    }

    separate_embeddings = None
    embeddings_by_configuration = {}
    for batch_size in (1, 2):
        for sep_chains in (False, True):
            result, embeddings = run_embeddings(
                wrapper,
                dataset,
                device,
                batch_size=batch_size,
                sep_chains=sep_chains,
                repeats=args.repeats,
            )
            report["embedding_runs"].append(result)
            embeddings_by_configuration[(batch_size, sep_chains)] = embeddings
            if batch_size == 2 and sep_chains:
                separate_embeddings = embeddings

    report["batch_size_parity"] = {}
    for sep_chains in (False, True):
        batch_one = embeddings_by_configuration[(1, sep_chains)]
        batch_two = embeddings_by_configuration[(2, sep_chains)]
        report["batch_size_parity"][str(sep_chains)] = {
            "max_abs_diff": float(torch.max(torch.abs(batch_one - batch_two))),
            "allclose_rtol_1e-5_atol_1e-6": bool(
                torch.allclose(batch_one, batch_two, rtol=1e-5, atol=1e-6)
            ),
        }

    if args.mlp_checkpoint:
        report["artifacts"]["mlp_checkpoint"] = {
            "path": str(args.mlp_checkpoint),
            "size_bytes": args.mlp_checkpoint.stat().st_size,
            "sha256": file_sha256(args.mlp_checkpoint),
        }
        report["binary_head"] = run_binary_head(
            separate_embeddings,
            args.mlp_checkpoint,
            device,
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
