#!/usr/bin/env python3
"""Score LibA candidates with the five locked nonlinear six-site models.

The scorer accepts only the target-free evaluation roster and the saved model
scores from the retention-blind model producer.  It must reproduce every one
of the 108 evaluation scores for every seed before any candidate output is
created.  Candidate ``model_score`` is the sigmoid of the mean seed logit,
matching ``nonlinear_6site_mean_logit`` in the producer's aligned output.

Retention values and retention-derived binder labels are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = (REPO_ROOT / "private_data").resolve()

LIBRARY = "LibA"
CHECKPOINT_SCHEMA_VERSION = "liba-nonlinear-6site-checkpoint-v1"
UNIVERSE_SCHEMA_VERSION = "liba-existing-target-selection-missed-candidates-v1"
OUTPUT_SCHEMA_VERSION = "liba-nonlinear-6site-candidate-scores-v1"
MODEL_ID = "nonlinear_6site_mean_logit"
PRODUCER_MODEL = "nonlinear_6site"
EXPECTED_KNOWN_ROWS = 108
EXPECTED_KNOWN_PEPTIDES = 9
EXPECTED_KNOWN_AFFIBODIES_PER_PEPTIDE = 12
EXPECTED_SEEDS = 5
EXPECTED_TRAINING_MEMBERSHIP_SHA256 = (
    "477d5113204d109333f74ae6051443f6a175a56b75500505418cb5efa2d99e3e"
)
PARITY_TOLERANCE = 1e-6
PROBABILITY_CLIP = 1e-7
AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {value: index for index, value in enumerate(AA_ALPHABET)}
POSITION_NAMES = (
    "peptide_position_4",
    "peptide_position_5",
    "affibody_displayed_13_crystal_15",
    "affibody_displayed_17_crystal_19",
    "affibody_displayed_27_crystal_29",
    "affibody_displayed_31_crystal_33",
)
CHECKPOINT_PATTERN = re.compile(r"^nonlinear_6site__seed([0-9]+)\.pt$")
FORBIDDEN_OUTCOME_TOKENS = ("retention", "binder", "outcome")
UNIVERSE_SCORE_INPUT_COLUMNS = (
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)
STANDARD_SCORE_COLUMNS = (
    "model_id",
    "pair_uid",
    "peptide_design_code",
    "peptide_9mer_sequence",
    "affibody_design_code",
    "provider_displayed_58aa_affibody_sequence",
    "model_input_affibody_sequence",
    "model_input_smart_hla_linker_peptide_sequence",
    "model_score",
    "observed_in_any_raw_round",
    "observed_in_r009_or_r010",
    "affibody_identity_seen_in_strict_training",
    "high_confidence_weak_negative",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _membership_sha256(values: pd.Series) -> str:
    payload = "\n".join(sorted(values.astype(str))).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict), f"expected JSON object in {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(path, 0o600)


def _validate_private_output(path: Path) -> Path:
    output = path.resolve()
    try:
        relative = output.relative_to(PRIVATE_ROOT)
    except ValueError as error:
        raise ValueError("output must stay below private_data") from error
    _require(bool(relative.parts), "refusing to write directly into private_data")
    _require(not output.exists(), "output exists; refusing overwrite")
    return output


def _validate_allow_list(columns: Sequence[str], role: str) -> None:
    forbidden = [
        column
        for column in columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_TOKENS)
    ]
    _require(not forbidden, f"{role} allow-list contains outcome fields: {forbidden}")


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    value = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def _clipped_logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return np.log(value) - np.log1p(-value)


def mean_logit_score(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean clipped logit and its sigmoid, matching the producer."""

    values = np.asarray(probabilities, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == EXPECTED_SEEDS, "expected five seed scores")
    _require(bool(np.isfinite(values).all()), "non-finite seed probability")
    _require(bool(((values >= 0.0) & (values <= 1.0)).all()), "seed probability outside [0,1]")
    mean = _clipped_logit(values).mean(axis=1)
    return mean, _sigmoid(mean)


class NonlinearSixSiteMLP(nn.Module):
    """Exact deployable architecture emitted by the LibA OOF producer."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        hidden = tuple(map(int, hidden_dims))
        _require(input_dim == 120 and hidden == (64, 32), "MLP architecture changed")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden[0]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(hidden[0]),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        _require(values.ndim == 2 and values.shape[1] == self.input_dim, "MLP input shape changed")
        return self.network(values.float()).squeeze(-1)


def _codes(frame: pd.DataFrame) -> np.ndarray:
    peptide = frame["peptide_design_code"].astype(str)
    affibody = frame["affibody_design_code"].astype(str)
    _require(peptide.str.len().eq(2).all(), "bad LibA peptide code length")
    _require(affibody.str.len().eq(4).all(), "bad LibA Affibody code length")
    joined = peptide + affibody
    _require(
        joined.map(lambda value: set(value).issubset(AA_TO_INDEX)).all(),
        "noncanonical residue in LibA designed code",
    )
    return np.asarray([list(value) for value in joined], dtype=str)


def encode_codes(frame: pd.DataFrame) -> np.ndarray:
    codes = _codes(frame)
    output = np.zeros((len(codes), len(POSITION_NAMES), len(AA_ALPHABET)), dtype=np.float32)
    rows = np.arange(len(codes))
    for position in range(len(POSITION_NAMES)):
        indices = np.asarray([AA_TO_INDEX[value] for value in codes[:, position]], dtype=int)
        output[rows, position, indices] = 1.0
    return output.reshape(len(codes), -1)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    _require(match is not None, f"unexpected checkpoint filename: {path.name}")
    _require(path.is_file(), f"missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(payload, dict), "checkpoint payload must be a mapping")
    required = {
        "schema_version",
        "model",
        "seed",
        "derived_training_seed",
        "epochs",
        "input_dimension",
        "hidden_dimensions",
        "dropout",
        "position_names",
        "amino_acid_alphabet",
        "weak_training_membership_sha256",
        "state_dict",
    }
    _require(required.issubset(payload), f"checkpoint lacks fields: {sorted(required.difference(payload))}")
    seed = int(payload["seed"])
    _require(seed == int(match.group(1)), "checkpoint seed differs from filename")
    _require(payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION, "checkpoint schema changed")
    _require(payload["model"] == PRODUCER_MODEL, "checkpoint model changed")
    _require(int(payload["input_dimension"]) == 120, "checkpoint input dimension changed")
    _require(tuple(map(int, payload["hidden_dimensions"])) == (64, 32), "checkpoint widths changed")
    dropout = float(payload["dropout"])
    _require(math.isfinite(dropout) and 0.0 <= dropout < 1.0, "invalid checkpoint dropout")
    _require(tuple(map(str, payload["position_names"])) == POSITION_NAMES, "checkpoint position order changed")
    _require(tuple(map(str, payload["amino_acid_alphabet"])) == AA_ALPHABET, "checkpoint alphabet changed")
    _require(
        str(payload["weak_training_membership_sha256"]) == EXPECTED_TRAINING_MEMBERSHIP_SHA256,
        "checkpoint weak-training membership changed",
    )
    _require(int(payload["epochs"]) > 0, "checkpoint epoch count is invalid")
    model = NonlinearSixSiteMLP(120, (64, 32), dropout)
    model.load_state_dict(payload["state_dict"], strict=True)
    trainable = int(sum(value.numel() for value in model.parameters() if value.requires_grad))
    _require(trainable == 10_225, "checkpoint trainable-parameter count changed")
    model.eval().to(device)
    return model, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "seed": seed,
        "derived_training_seed": int(payload["derived_training_seed"]),
        "epochs": int(payload["epochs"]),
        "input_dimension": 120,
        "hidden_dimensions": [64, 32],
        "dropout": dropout,
        "trainable_parameters": trainable,
        "weak_training_membership_sha256": EXPECTED_TRAINING_MEMBERSHIP_SHA256,
    }


def load_checkpoints(checkpoint_dir: Path, device: torch.device) -> tuple[list[nn.Module], list[dict[str, Any]]]:
    checkpoint_dir = checkpoint_dir.resolve()
    _require(checkpoint_dir.is_dir(), "checkpoint directory does not exist")
    paths = sorted(checkpoint_dir.glob("nonlinear_6site__seed*.pt"))
    _require(len(paths) == EXPECTED_SEEDS, "exactly five nonlinear checkpoints are required")
    loaded = [load_checkpoint(path, device) for path in paths]
    loaded.sort(key=lambda value: int(value[1]["seed"]))
    models = [value[0] for value in loaded]
    records = [value[1] for value in loaded]
    seeds = [int(value["seed"]) for value in records]
    _require(len(set(seeds)) == EXPECTED_SEEDS, "checkpoint seeds are not unique")
    epochs = {int(value["epochs"]) for value in records}
    _require(len(epochs) == 1, "final checkpoints use different selected epochs")
    return models, records


def score_features(
    models: Sequence[nn.Module], features: np.ndarray, batch_size: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require(len(models) == EXPECTED_SEEDS, "exactly five models are required")
    _require(int(batch_size) > 0, "batch size must be positive")
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=False, num_workers=0)
    seed_logits: list[list[np.ndarray]] = [[] for _ in models]
    with torch.inference_mode():
        for (values,) in loader:
            values = values.to(device)
            for index, model in enumerate(models):
                seed_logits[index].append(model(values).detach().cpu().numpy().astype(np.float64))
    logits = np.column_stack([np.concatenate(blocks) for blocks in seed_logits])
    _require(logits.shape == (len(features), EXPECTED_SEEDS), "seed-logit matrix shape changed")
    probability = _sigmoid(logits)
    mean_logit, score = mean_logit_score(probability)
    _require(bool(np.isfinite(logits).all() and np.isfinite(score).all()), "non-finite MLP score")
    return logits, probability, mean_logit, score


def _known_panel_columns(path: Path) -> tuple[list[str], str]:
    header = pd.read_csv(path, nrows=0)
    id_choices = [name for name in ("eval_row_id", "pair_uid") if name in header.columns]
    _require(len(id_choices) == 1, "known panel needs exactly one of eval_row_id or pair_uid")
    required = [id_choices[0], "peptide_design_code", "affibody_design_code"]
    if "library" in header.columns:
        required.append("library")
    _require(set(required).issubset(header.columns), "known-panel columns are incomplete")
    _validate_allow_list(required, "known-panel")
    return required, id_choices[0]


def verify_known_pair_parity(
    known_panel_path: Path,
    parity_predictions_path: Path,
    models: Sequence[nn.Module],
    checkpoint_records: Sequence[Mapping[str, Any]],
    batch_size: int,
    device: torch.device,
    tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Replay every saved seed score without loading any measured outcome."""

    _require(known_panel_path.is_file(), "known-panel file does not exist")
    _require(parity_predictions_path.is_file(), "parity-prediction file does not exist")
    _require(math.isfinite(tolerance) and tolerance >= 0.0, "invalid parity tolerance")
    panel_columns, panel_id = _known_panel_columns(known_panel_path)
    panel = pd.read_csv(
        known_panel_path,
        usecols=panel_columns,
        dtype={column: str for column in panel_columns},
        keep_default_na=False,
        na_filter=False,
    )
    if "library" in panel:
        panel = panel.loc[panel["library"].eq(LIBRARY)].copy()
    panel = panel.rename(columns={panel_id: "eval_row_id"})
    _require(len(panel) == EXPECTED_KNOWN_ROWS, "known LibA panel must contain 108 pairs")
    _require(not panel["eval_row_id"].duplicated().any(), "duplicate known LibA row ID")
    _require(panel["peptide_design_code"].nunique() == EXPECTED_KNOWN_PEPTIDES, "known peptide count changed")
    _require(
        panel.groupby("peptide_design_code")["affibody_design_code"]
        .nunique()
        .eq(EXPECTED_KNOWN_AFFIBODIES_PER_PEPTIDE)
        .all(),
        "known LibA panel is not complete 9x12 membership",
    )

    parity_columns = ("eval_row_id", "model", "seed", "score")
    _validate_allow_list(parity_columns, "parity-prediction")
    reference = pd.read_csv(
        parity_predictions_path,
        usecols=list(parity_columns),
        dtype={"eval_row_id": str, "model": str, "seed": str},
        keep_default_na=False,
        na_filter=False,
    )
    reference = reference.loc[reference["model"].eq(PRODUCER_MODEL)].copy()
    reference["score"] = pd.to_numeric(reference["score"], errors="raise").astype(np.float64)
    seeds = [str(record["seed"]) for record in checkpoint_records]
    _require(set(reference["seed"]) == set(seeds), "parity seeds differ from checkpoints")
    _require(len(reference) == EXPECTED_KNOWN_ROWS * EXPECTED_SEEDS, "parity table lacks 108 scores per seed")
    _require(not reference.duplicated(["eval_row_id", "seed"]).any(), "duplicate parity row/seed")
    _require(bool(np.isfinite(reference["score"]).all()), "non-finite parity probability")

    features = encode_codes(panel)
    _, replay_probability, _, replay_aggregate = score_features(
        models, features, batch_size, device
    )
    seed_receipts: list[dict[str, Any]] = []
    expected_matrix = np.empty_like(replay_probability)
    for index, seed in enumerate(seeds):
        # A left merge preserves the exact panel order used for the replay.
        # Recent pandas versions sort the union keys for an outer merge even
        # when ``sort=False``, which would compare scores to the wrong rows.
        expected = panel[["eval_row_id"]].merge(
            reference.loc[reference["seed"].eq(seed), ["eval_row_id", "score"]],
            on="eval_row_id",
            how="left",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
        _require(len(expected) == EXPECTED_KNOWN_ROWS and expected["_merge"].eq("both").all(),
                 f"known-panel membership differs for seed {seed}")
        expected_matrix[:, index] = expected["score"].to_numpy(dtype=np.float64)
        delta = np.abs(replay_probability[:, index] - expected_matrix[:, index])
        maximum = float(delta.max())
        _require(maximum <= tolerance, f"LibA nonlinear 108-row parity failed for seed {seed}: {maximum} > {tolerance}")
        seed_receipts.append(
            {
                "seed": int(seed),
                "rows": EXPECTED_KNOWN_ROWS,
                "maximum_absolute_probability_difference": maximum,
                "mean_absolute_probability_difference": float(delta.mean()),
            }
        )
    _, expected_aggregate = mean_logit_score(expected_matrix)
    aggregate_delta = np.abs(replay_aggregate - expected_aggregate)
    maximum_aggregate = float(aggregate_delta.max())
    _require(maximum_aggregate <= tolerance, "LibA nonlinear aggregate 108-row parity failed")
    return {
        "status": "passed",
        "rows": EXPECTED_KNOWN_ROWS,
        "seeds": seed_receipts,
        "maximum_absolute_seed_probability_difference": float(
            max(value["maximum_absolute_probability_difference"] for value in seed_receipts)
        ),
        "maximum_absolute_mean_logit_score_difference": maximum_aggregate,
        "tolerance": float(tolerance),
        "known_pair_membership_sha256": _membership_sha256(panel["eval_row_id"]),
        "known_panel": {
            "path": str(known_panel_path.resolve()),
            "sha256": sha256_file(known_panel_path),
            "columns_read": panel_columns,
        },
        "reference_predictions": {
            "path": str(parity_predictions_path.resolve()),
            "sha256": sha256_file(parity_predictions_path),
            "columns_read": list(parity_columns),
        },
        "retention_or_binder_outcome_columns_read": [],
    }


def _validate_candidate_frame(frame: pd.DataFrame, peptide: str) -> None:
    _require(frame["peptide_design_code"].astype(str).eq(peptide).all(), "mixed peptide partition")
    _require(not frame["pair_uid"].duplicated().any(), "duplicate pair UID within partition")
    _require(frame["peptide_9mer_sequence"].astype(str).str.len().eq(9).all(), "bad peptide length")
    displayed = frame["provider_displayed_58aa_affibody_sequence"].astype(str)
    _require(displayed.str.len().eq(58).all(), "bad displayed Affibody length")
    _require(displayed.eq(frame["model_input_affibody_sequence"].astype(str)).all(),
             "displayed and model-input Affibody sequences differ")
    _require(
        all(
            str(assay).endswith(str(peptide_9mer))
            for assay, peptide_9mer in zip(
                frame["model_input_smart_hla_linker_peptide_sequence"],
                frame["peptide_9mer_sequence"],
            )
        ),
        "assay-side model input does not end in its peptide 9-mer",
    )
    for column in (
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
    ):
        _require(
            pd.api.types.is_bool_dtype(frame[column]) and frame[column].notna().all(),
            f"candidate {column} must be complete Boolean data",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--known-panel", type=Path, required=True)
    parser.add_argument("--parity-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    universe = args.universe_dir.resolve()
    output = _validate_private_output(args.output_dir)
    _require(universe.is_dir(), "candidate universe does not exist")
    _require(int(args.batch_size) > 0, "batch size must be positive")
    device = torch.device(args.device)
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    universe_manifest_path = universe / "manifest.json"
    partitions_path = universe / "candidate_partitions.csv"
    _require(universe_manifest_path.is_file(), "candidate universe has no completion manifest")
    _require(partitions_path.is_file(), "candidate universe has no partition table")

    models, checkpoint_records = load_checkpoints(args.checkpoint_dir, device)
    parity = verify_known_pair_parity(
        args.known_panel,
        args.parity_predictions,
        models,
        checkpoint_records,
        int(args.batch_size),
        device,
    )
    # No file or directory is created before the complete 108 x five-seed gate.

    universe_manifest = _read_json(universe_manifest_path)
    _require(universe_manifest.get("schema_version") == UNIVERSE_SCHEMA_VERSION,
             "candidate-universe schema changed")
    _require(
        universe_manifest.get("input_data_contract", {}).get("retention_outcome_columns_read") == [],
        "candidate universe does not attest outcome-blind construction",
    )
    _require(
        universe_manifest.get("outputs", {}).get("candidate_partitions_csv", {}).get("sha256")
        == sha256_file(partitions_path),
        "candidate partition table differs from immutable manifest",
    )
    partitions = pd.read_csv(partitions_path, keep_default_na=False, na_filter=False)
    required_partition_columns = {
        "peptide_design_code", "candidate_rows", "partition", "partition_sha256"
    }
    _require(required_partition_columns.issubset(partitions), "candidate partition table is incomplete")
    _require(len(partitions) == EXPECTED_KNOWN_PEPTIDES, "expected nine candidate partitions")
    _require(not partitions["peptide_design_code"].duplicated().any(), "duplicate peptide partition")
    expected_rows = int(universe_manifest["outputs"]["candidate_rows"])
    _require(int(pd.to_numeric(partitions["candidate_rows"], errors="raise").sum()) == expected_rows,
             "candidate rows differ from immutable manifest")

    source_hashes = {
        "known_panel": sha256_file(args.known_panel),
        "parity_predictions": sha256_file(args.parity_predictions),
        "universe_manifest": sha256_file(universe_manifest_path),
        "candidate_partitions": sha256_file(partitions_path),
        **{f"checkpoint_{record['seed']}": str(record["sha256"]) for record in checkpoint_records},
    }
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    started = time.time()
    records: list[dict[str, Any]] = []
    seen_pair_uids: set[str] = set()
    combined_path = output / "candidate_scores.parquet"
    combined_writer: pq.ParquetWriter | None = None
    seeds = [int(record["seed"]) for record in checkpoint_records]
    seed_score_columns = [f"nonlinear_6site__seed{seed}_score" for seed in seeds]
    seed_logit_columns = [f"nonlinear_6site__seed{seed}_logit" for seed in seeds]
    try:
        for row in partitions.sort_values("peptide_design_code").itertuples(index=False):
            source = (universe / str(row.partition)).resolve()
            try:
                source.relative_to(universe)
            except ValueError as error:
                raise ValueError("candidate partition escapes universe directory") from error
            _require(source.is_file(), f"missing candidate partition {source}")
            source_sha256 = sha256_file(source)
            _require(source_sha256 == str(row.partition_sha256), "candidate partition hash changed")
            frame = pd.read_parquet(source, columns=list(UNIVERSE_SCORE_INPUT_COLUMNS))
            peptide = str(row.peptide_design_code)
            _require(len(frame) == int(row.candidate_rows), "candidate partition row count changed")
            _validate_candidate_frame(frame, peptide)
            _require(not frame["pair_uid"].astype(str).isin(seen_pair_uids).any(),
                     "duplicate pair UID across candidate partitions")
            seen_pair_uids.update(frame["pair_uid"].astype(str))

            features = encode_codes(frame)
            logits, probabilities, ensemble_logit, score = score_features(
                models, features, int(args.batch_size), device
            )
            result = frame.copy()
            result.insert(0, "model_id", MODEL_ID)
            result["model_score"] = score
            for index, column in enumerate(seed_score_columns):
                result[column] = probabilities[:, index]
            for index, column in enumerate(seed_logit_columns):
                result[column] = logits[:, index]
            result["nonlinear_6site_ensemble_logit"] = ensemble_logit
            result["nonlinear_6site_mean_logit"] = score
            extra_columns = seed_score_columns + seed_logit_columns + [
                "nonlinear_6site_ensemble_logit", "nonlinear_6site_mean_logit"
            ]
            result = result[list(STANDARD_SCORE_COLUMNS) + extra_columns]
            _require(list(result.columns[: len(STANDARD_SCORE_COLUMNS)]) == list(STANDARD_SCORE_COLUMNS),
                     "standard candidate-score schema changed")
            destination = output / f"peptide_{peptide}.parquet"
            result.to_parquet(destination, index=False, engine="pyarrow", compression="zstd")
            os.chmod(destination, 0o600)

            arrow_table = pa.Table.from_pandas(result, preserve_index=False)
            if combined_writer is None:
                combined_writer = pq.ParquetWriter(combined_path, arrow_table.schema, compression="zstd")
            else:
                _require(arrow_table.schema.equals(combined_writer.schema),
                         "candidate output schema differs across peptide partitions")
            combined_writer.write_table(arrow_table)
            _require(sha256_file(source) == source_sha256, "candidate partition changed during scoring")
            records.append(
                {
                    "peptide_design_code": peptide,
                    "rows": int(len(result)),
                    "source_path": str(source),
                    "source_sha256": source_sha256,
                    "output_path": destination.name,
                    "output_sha256": sha256_file(destination),
                    "output_bytes": int(destination.stat().st_size),
                }
            )
    finally:
        if combined_writer is not None:
            combined_writer.close()

    _require(combined_path.is_file(), "combined candidate-score table was not written")
    os.chmod(combined_path, 0o600)
    _require(len(seen_pair_uids) == expected_rows, "scored candidate membership changed")
    for name, expected in source_hashes.items():
        if name.startswith("checkpoint_"):
            seed = int(name.removeprefix("checkpoint_"))
            path = Path(next(record["path"] for record in checkpoint_records if int(record["seed"]) == seed))
        elif name == "known_panel":
            path = args.known_panel
        elif name == "parity_predictions":
            path = args.parity_predictions
        elif name == "universe_manifest":
            path = universe_manifest_path
        else:
            path = partitions_path
        _require(sha256_file(path) == expected, f"source {name} changed during scoring")

    manifest = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "library": LIBRARY,
        "rows": int(sum(record["rows"] for record in records)),
        "model_id": MODEL_ID,
        "score_definition": {
            "model_score_column": "model_score",
            "formula": "sigmoid(mean of the five clipped deployment-seed logits)",
            "producer_aligned_column": "nonlinear_6site_mean_logit",
            "seed_probability_columns": seed_score_columns,
            "seed_raw_logit_columns": seed_logit_columns,
            "score_direction": "higher_is_more_likely_selection_derived_binder",
            "retention_percentage_interpretation": False,
        },
        "standard_score_columns": list(STANDARD_SCORE_COLUMNS),
        "combined_candidate_scores": {
            "path": combined_path.name,
            "rows": int(len(seen_pair_uids)),
            "sha256": sha256_file(combined_path),
            "bytes": int(combined_path.stat().st_size),
        },
        "checkpoints": checkpoint_records,
        "known_108_pair_parity": parity,
        "candidate_universe": {
            "path": str(universe),
            "manifest_sha256": source_hashes["universe_manifest"],
            "candidate_partitions_sha256": source_hashes["candidate_partitions"],
            "rows": expected_rows,
        },
        "partitions": records,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "retention_or_binder_outcomes_read": False,
        "manifest_written_last": True,
    }
    _write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
