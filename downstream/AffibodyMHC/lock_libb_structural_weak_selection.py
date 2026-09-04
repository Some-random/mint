#!/usr/bin/env python
"""Select one LibB readout per structural family using weak labels only.

The command reads only the structural-readout configs and the artifacts named
``audit.json``, ``weak_validation_summary.json``, and
``weak_validation_predictions.csv.gz`` beneath each supplied pilot root.  It
has no argument for, and never opens, direct-retention outcomes.

Selection minimizes pooled out-of-fold weak-validation log loss:

    sum(fold_rows * fold_log_loss) / sum(fold_rows)

Candidate models are the ``frozen_features`` arms declared in each family
config plus prespecified mean-logit ensembles of those arms.  Ensemble scores
are derived from aligned out-of-fold weak predictions using the same 1e-7
probability clipping as the training code.  The sequence-only control is
explicitly excluded from structural-family selection.  Exact ties are broken
by model name so the result is deterministic.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-structural-weak-selection-v1"
SELECTION_METRIC = "pooled_weak_validation_log_loss"
EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_WEAK_ROWS = 30_648
EXPECTED_WEAK_POSITIVE = 23_725
EXPECTED_WEAK_NEGATIVE = 6_923
SUMMARY_MATCH_TOLERANCE = 2e-7


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: object, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{label} is not finite")
    return number


def _model_specs(config: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = config.get("models")
    _require(isinstance(raw, list) and raw, "config has no model list")
    specs: list[dict[str, Any]] = []
    for item in raw:
        _require(isinstance(item, dict), "model spec is not an object")
        name = str(item.get("name", ""))
        _require(name, "model spec lacks name")
        specs.append(dict(item))
    _require(len({item["name"] for item in specs}) == len(specs), "duplicate model name")
    return tuple(specs)


def _prediction_metrics(path: Path, expected_model: str) -> dict[str, Any]:
    """Recompute fold and pooled log loss from weak-CV predictions."""

    by_fold: dict[int, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames == ["model", "fold", "row_id", "weak_label", "probability"],
            f"unexpected weak-prediction columns in {path}",
        )
        for row in reader:
            # Parallel pilot jobs can place several explicitly requested model
            # arms in one artifact.  Read only the named arm.
            if row["model"] != expected_model:
                continue
            fold = int(row["fold"])
            _require(fold in EXPECTED_FOLDS, f"unexpected fold {fold} in {path}")
            row_id = str(row["row_id"])
            _require(row_id not in seen_ids, f"duplicate out-of-fold row {row_id} in {path}")
            seen_ids.add(row_id)
            label = int(row["weak_label"])
            _require(label in (0, 1), f"invalid weak label in {path}")
            probability = _finite_number(row["probability"], f"probability in {path}")
            _require(0.0 <= probability <= 1.0, f"probability outside [0,1] in {path}")
            clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
            loss = -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
            state = by_fold.setdefault(fold, {"rows": 0, "positive": 0, "loss_sum": 0.0})
            state["rows"] += 1
            state["positive"] += label
            state["loss_sum"] += loss

    _require(tuple(sorted(by_fold)) == EXPECTED_FOLDS, f"missing folds in {path}")
    folds = []
    for fold in EXPECTED_FOLDS:
        state = by_fold[fold]
        rows = int(state["rows"])
        _require(rows > 0, f"empty fold {fold} in {path}")
        folds.append(
            {
                "fold": fold,
                "rows": rows,
                "positive": int(state["positive"]),
                "log_loss": float(state["loss_sum"] / rows),
            }
        )
    total_rows = sum(item["rows"] for item in folds)
    pooled = sum(item["rows"] * item["log_loss"] for item in folds) / total_rows
    mean = sum(item["log_loss"] for item in folds) / len(folds)
    return {
        "folds": folds,
        "out_of_fold_rows": total_rows,
        "out_of_fold_positive": sum(item["positive"] for item in folds),
        "mean_fold_log_loss": float(mean),
        "pooled_weak_validation_log_loss": float(pooled),
    }


def _read_prediction_rows(
    path: Path, expected_model: str
) -> tuple[list[tuple[int, str, int, float]], str]:
    rows: list[tuple[int, str, int, float]] = []
    seen: set[tuple[int, str]] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames == ["model", "fold", "row_id", "weak_label", "probability"],
            f"unexpected weak-prediction columns in {path}",
        )
        for row in reader:
            if row["model"] != expected_model:
                continue
            fold = int(row["fold"])
            row_id = str(row["row_id"])
            label = int(row["weak_label"])
            probability = _finite_number(row["probability"], f"probability in {path}")
            _require(fold in EXPECTED_FOLDS, f"unexpected fold {fold} in {path}")
            _require(label in (0, 1), f"invalid weak label in {path}")
            _require(0.0 <= probability <= 1.0, f"probability outside [0,1] in {path}")
            key = (fold, row_id)
            _require(key not in seen, f"duplicate out-of-fold row {key} in {path}")
            seen.add(key)
            rows.append((fold, row_id, label, probability))
    _require(rows, f"empty weak-prediction artifact {path}")
    return rows, _sha256_file(path)


def _probability_from_logit(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    value = math.exp(logit)
    return value / (1.0 + value)


def _mean_logit_ensemble_metrics(
    *,
    ensemble_name: str,
    members: Sequence[str],
    found: Mapping[str, Mapping[str, Path]],
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    """Calculate a prespecified ensemble solely from aligned weak OOF scores."""

    _require(len(members) >= 2, f"{ensemble_name} needs at least two members")
    member_rows: list[list[tuple[int, str, int, float]]] = []
    sources: list[dict[str, str]] = []
    for member in members:
        _require(member in found, f"{ensemble_name} member {member} has no weak-CV output")
        path = found[member]["predictions"]
        rows, checksum = _read_prediction_rows(path, member)
        member_rows.append(rows)
        sources.append({"model": member, "path": _relative(path), "sha256": checksum})
    reference = [(fold, row_id, label) for fold, row_id, label, _ in member_rows[0]]
    for member, rows in zip(members[1:], member_rows[1:]):
        _require(
            [(fold, row_id, label) for fold, row_id, label, _ in rows] == reference,
            f"{ensemble_name} member {member} OOF rows/folds/labels are not aligned",
        )

    fold_state: dict[int, dict[str, Any]] = {}
    for index, (fold, _row_id, label) in enumerate(reference):
        logits = []
        for rows in member_rows:
            probability = min(max(rows[index][3], 1e-7), 1.0 - 1e-7)
            logits.append(math.log(probability) - math.log1p(-probability))
        probability = _probability_from_logit(sum(logits) / len(logits))
        clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
        loss = -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))
        state = fold_state.setdefault(fold, {"rows": 0, "positive": 0, "loss_sum": 0.0})
        state["rows"] += 1
        state["positive"] += label
        state["loss_sum"] += loss
    _require(tuple(sorted(fold_state)) == EXPECTED_FOLDS, f"{ensemble_name} is missing folds")
    folds = [
        {
            "fold": fold,
            "rows": int(fold_state[fold]["rows"]),
            "positive": int(fold_state[fold]["positive"]),
            "log_loss": float(fold_state[fold]["loss_sum"] / fold_state[fold]["rows"]),
        }
        for fold in EXPECTED_FOLDS
    ]
    total_rows = sum(item["rows"] for item in folds)
    metrics = {
        "folds": folds,
        "out_of_fold_rows": total_rows,
        "out_of_fold_positive": sum(item["positive"] for item in folds),
        "mean_fold_log_loss": float(sum(item["log_loss"] for item in folds) / len(folds)),
        "pooled_weak_validation_log_loss": float(
            sum(item["rows"] * item["log_loss"] for item in folds) / total_rows
        ),
    }
    derivation = {
        "ensemble": ensemble_name,
        "members": list(members),
        "method": "mean_logit",
        "probability_clip_before_logit": [1e-7, 1.0 - 1e-7],
        "member_weak_validation_predictions": sources,
    }
    return metrics, sources, _canonical_sha256(derivation)


def _feature_inputs(spec: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    arrays = manifest.get("arrays")
    _require(isinstance(arrays, dict), "feature manifest has no arrays object")
    output: list[dict[str, Any]] = []
    for view in spec.get("views", []):
        _require(isinstance(view, dict), "model view is not an object")
        name = str(view.get("array", ""))
        _require(name in arrays, f"feature array {name!r} is absent from manifest")
        metadata = arrays[name]
        _require(isinstance(metadata, dict), f"feature array metadata for {name} is invalid")
        checksum = str(metadata.get("sha256", ""))
        _require(len(checksum) == 64, f"feature array {name} lacks SHA-256")
        item = {
            "array": name,
            "sha256": checksum,
            "pooling": str(view.get("pooling", "")),
        }
        if "mask" in view:
            mask = str(view["mask"])
            _require(mask in arrays, f"feature mask {mask!r} is absent from manifest")
            mask_metadata = arrays[mask]
            _require(isinstance(mask_metadata, dict), f"feature mask metadata for {mask} is invalid")
            mask_checksum = str(mask_metadata.get("sha256", ""))
            _require(len(mask_checksum) == 64, f"feature mask {mask} lacks SHA-256")
            item["mask"] = mask
            item["mask_sha256"] = mask_checksum
        output.append(item)
    _require(output, f"frozen-feature model {spec.get('name')} has no feature inputs")
    return output


def _summary_by_model(pilot_root: Path) -> dict[str, dict[str, Path]]:
    output: dict[str, dict[str, Path]] = {}
    for summary_path in sorted(pilot_root.glob("*/weak_validation_summary.json")):
        payload = _read_json(summary_path)
        _require(isinstance(payload, list) and payload, f"invalid summary {summary_path}")
        for item in payload:
            _require(isinstance(item, dict), f"invalid model summary in {summary_path}")
            model = str(item.get("model", ""))
            _require(model and model not in output, f"duplicate or missing model summary: {model!r}")
            prediction_path = summary_path.parent / "weak_validation_predictions.csv.gz"
            audit_path = summary_path.parent / "audit.json"
            _require(prediction_path.is_file(), f"missing {prediction_path}")
            _require(audit_path.is_file(), f"missing {audit_path}")
            output[model] = {
                "summary": summary_path,
                "predictions": prediction_path,
                "audit": audit_path,
            }
    return output


def _audit_family(
    family: str,
    config_path: Path,
    pilot_root: Path,
    feature_manifest_path: Path,
) -> dict[str, Any]:
    config = _read_json(config_path)
    _require(isinstance(config, dict), f"{config_path} is not an object")
    _require(config.get("schema_version") == "libb-structural-frozen-readouts-v1", "config schema changed")
    specs = _model_specs(config)
    candidates = {str(spec["name"]): spec for spec in specs if spec.get("source") == "frozen_features"}
    controls = sorted(str(spec["name"]) for spec in specs if spec.get("source") != "frozen_features")
    _require(candidates, f"{family} has no frozen-feature candidates")
    raw_ensembles = config.get("prediction_ensembles", [])
    _require(isinstance(raw_ensembles, list), f"{family} prediction_ensembles is not a list")
    ensembles: list[dict[str, Any]] = []
    for item in raw_ensembles:
        _require(isinstance(item, dict), f"{family} ensemble spec is not an object")
        name = str(item.get("name", ""))
        members = tuple(map(str, item.get("members", [])))
        _require(name and name not in candidates, f"invalid {family} ensemble name {name!r}")
        _require(len(members) >= 2 and len(set(members)) == len(members), f"{name} members changed")
        _require(set(members).issubset(candidates), f"{name} has a non-structural or unknown member")
        _require(item.get("method") == "mean_logit", f"{name} is not a mean-logit ensemble")
        ensembles.append({"name": name, "members": members, "method": "mean_logit"})
    _require(len({item["name"] for item in ensembles}) == len(ensembles), f"duplicate {family} ensemble")

    found = _summary_by_model(pilot_root)
    missing = sorted(set(candidates) - set(found))
    _require(not missing, f"{family} weak-validation pilots are incomplete; missing {missing}")
    unexpected = sorted(set(found) - set(candidates) - set(controls))
    _require(not unexpected, f"{family} has unexpected weak-validation summaries: {unexpected}")

    feature_manifest = _read_json(feature_manifest_path)
    _require(isinstance(feature_manifest, dict), "feature manifest is not an object")
    feature_manifest_sha256 = _sha256_file(feature_manifest_path)
    results: list[dict[str, Any]] = []
    reference_fold_counts: tuple[tuple[int, int, int], ...] | None = None
    for model in sorted(candidates):
        paths = found[model]
        summary_payload = _read_json(paths["summary"])
        entry_matches = [item for item in summary_payload if item.get("model") == model]
        _require(len(entry_matches) == 1, f"summary entry for {model} is not unique")
        summary = entry_matches[0]
        audit = _read_json(paths["audit"])
        _require(isinstance(audit, dict), f"audit for {model} is invalid")
        _require(audit.get("retention_labels_read") is False, f"audit says {model} read retention")
        _require(audit.get("library") == "LibB", f"{model} is not LibB")
        _require(int(audit.get("weak_rows", -1)) == EXPECTED_WEAK_ROWS, f"{model} weak row count changed")
        _require(int(audit.get("weak_positive", -1)) == EXPECTED_WEAK_POSITIVE, f"{model} weak positive count changed")
        _require(int(audit.get("weak_negative", -1)) == EXPECTED_WEAK_NEGATIVE, f"{model} weak negative count changed")
        _require(audit.get("seed_mode") == "pilot", f"{model} source is not a pilot weak-CV run")
        _require(
            str(audit.get("feature_archive_manifest_sha256")) == feature_manifest_sha256,
            f"{model} points to another feature archive",
        )
        trainable = audit.get("trainable_parameters", {})
        _require(isinstance(trainable, dict) and int(trainable.get(model, -1)) == 86_785, f"{model} capacity changed")

        recomputed = _prediction_metrics(paths["predictions"], model)
        summary_folds = summary.get("folds")
        _require(isinstance(summary_folds, list) and len(summary_folds) == 3, f"{model} summary folds changed")
        by_fold = {int(item["fold"]): item for item in summary_folds}
        _require(tuple(sorted(by_fold)) == EXPECTED_FOLDS, f"{model} summary is missing folds")
        for fold in recomputed["folds"]:
            source = by_fold[fold["fold"]]
            metrics = source.get("best_metrics", {})
            _require(int(metrics.get("rows", -1)) == fold["rows"], f"{model} fold rows disagree")
            _require(int(metrics.get("positive", -1)) == fold["positive"], f"{model} fold positives disagree")
            stored_loss = _finite_number(metrics.get("log_loss"), f"{model} stored log loss")
            _require(
                abs(stored_loss - fold["log_loss"]) <= SUMMARY_MATCH_TOLERANCE,
                f"{model} recomputed and stored log loss disagree",
            )
            fold["best_epoch"] = int(source["best_epoch"])
            fold["log_loss"] = stored_loss
        rows = recomputed["folds"]
        recomputed["pooled_weak_validation_log_loss"] = float(
            sum(item["rows"] * item["log_loss"] for item in rows)
            / sum(item["rows"] for item in rows)
        )
        recomputed["mean_fold_log_loss"] = float(
            sum(item["log_loss"] for item in rows) / len(rows)
        )
        fold_counts = tuple((item["fold"], item["rows"], item["positive"]) for item in rows)
        if reference_fold_counts is None:
            reference_fold_counts = fold_counts
        _require(fold_counts == reference_fold_counts, f"{model} used different weak-validation folds")

        selected_final_epoch = int(summary.get("selected_final_epochs", -1))
        _require(selected_final_epoch >= 1, f"{model} lacks selected final epoch")
        results.append(
            {
                "model": model,
                "eligible": True,
                "selected_final_epoch": selected_final_epoch,
                **recomputed,
                "feature_inputs": _feature_inputs(candidates[model], feature_manifest),
                "source_artifacts": {
                    "audit": {"path": _relative(paths["audit"]), "sha256": _sha256_file(paths["audit"])},
                    "weak_validation_predictions": {
                        "path": _relative(paths["predictions"]),
                        "sha256": _sha256_file(paths["predictions"]),
                    },
                    "weak_validation_summary": {
                        "path": _relative(paths["summary"]),
                        "sha256": _sha256_file(paths["summary"]),
                    },
                },
            }
        )

    trained_results = {item["model"]: item for item in results}
    for ensemble in sorted(ensembles, key=lambda item: item["name"]):
        model = ensemble["name"]
        members = ensemble["members"]
        metrics, prediction_sources, derivation_sha256 = _mean_logit_ensemble_metrics(
            ensemble_name=model,
            members=members,
            found=found,
        )
        fold_counts = tuple(
            (item["fold"], item["rows"], item["positive"])
            for item in metrics["folds"]
        )
        _require(fold_counts == reference_fold_counts, f"{model} used different weak-validation folds")
        results.append(
            {
                "model": model,
                "eligible": True,
                "source": "prespecified_scalar_ensemble",
                "members": list(members),
                "method": "mean_logit",
                "member_selected_final_epochs": {
                    member: trained_results[member]["selected_final_epoch"] for member in members
                },
                **metrics,
                "feature_inputs_by_member": {
                    member: trained_results[member]["feature_inputs"] for member in members
                },
                "source_artifacts": {
                    "member_weak_validation_predictions": prediction_sources,
                    "ensemble_derivation_sha256": derivation_sha256,
                },
            }
        )

    selected = min(
        results,
        key=lambda item: (item["pooled_weak_validation_log_loss"], item["model"]),
    )
    return {
        "family": family,
        "candidate_definition": (
            "all config models with source=frozen_features, plus all prespecified "
            "mean-logit prediction ensembles whose members are frozen-feature models"
        ),
        "ineligible_sequence_controls": controls,
        "config": {"path": _relative(config_path), "sha256": _sha256_file(config_path)},
        "pilot_root": _relative(pilot_root),
        "feature_archive_manifest": {
            "path": _relative(feature_manifest_path),
            "sha256": feature_manifest_sha256,
        },
        "source_feature_manifest_sha256": feature_manifest_sha256,
        "candidates": results,
        "selected_model": selected["model"],
        "selected_pooled_weak_validation_log_loss": selected[
            "pooled_weak_validation_log_loss"
        ],
        "selected_source_weak_cv_artifact_sha256": (
            selected["source_artifacts"]["weak_validation_predictions"]["sha256"]
            if "weak_validation_predictions" in selected["source_artifacts"]
            else selected["source_artifacts"]["ensemble_derivation_sha256"]
        ),
        "source_weak_cv_artifact_sha256": (
            selected["source_artifacts"]["weak_validation_predictions"]["sha256"]
            if "weak_validation_predictions" in selected["source_artifacts"]
            else selected["source_artifacts"]["ensemble_derivation_sha256"]
        ),
        "selected_feature_inputs": (
            selected["feature_inputs"]
            if "feature_inputs" in selected
            else selected["feature_inputs_by_member"]
        ),
        "selected_feature_contract_sha256": _canonical_sha256(
            selected["feature_inputs"]
            if "feature_inputs" in selected
            else selected["feature_inputs_by_member"]
        ),
    }


def build_selection_record(
    *,
    rde_config: Path,
    rde_pilot_root: Path,
    rde_feature_manifest: Path,
    stab_config: Path,
    stab_pilot_root: Path,
    stab_feature_manifest: Path,
) -> dict[str, Any]:
    families = {
        "rde": _audit_family("RDE-PPI-derived", rde_config, rde_pilot_root, rde_feature_manifest),
        "stab": _audit_family("StaB-ddG-derived", stab_config, stab_pilot_root, stab_feature_manifest),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "lock one representative per structural family before opening direct-retention labels",
        "producer": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "readout_training_implementation": {
            "path": "downstream/AffibodyMHC/train_libb_structural_readouts.py",
            "sha256": _sha256_file(
                REPO_ROOT / "downstream/AffibodyMHC/train_libb_structural_readouts.py"
            ),
        },
        "library": "LibB",
        "selection_metric": SELECTION_METRIC,
        "selection_direction": "minimize",
        "selection_formula": "sum(fold_rows * fold_log_loss) / sum(fold_rows)",
        "ensemble_formula": (
            "align member out-of-fold rows/folds/weak labels; clip each probability to "
            "[1e-7,1-1e-7]; convert to logits; average logits; convert back to probability; "
            "then calculate pooled weak-validation log loss"
        ),
        "tie_break": "lexicographically smallest model name after exact pooled-loss tie",
        "selection_data": "weak selection-derived binder/non-binder labels only",
        "retention_labels_read": False,
        "dataset": {
            "weak_rows": EXPECTED_WEAK_ROWS,
            "weak_positive": EXPECTED_WEAK_POSITIVE,
            "weak_negative": EXPECTED_WEAK_NEGATIVE,
        },
        "selected_models_by_family": {
            key: value["selected_model"] for key, value in families.items()
        },
        "families": families,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rde-config",
        type=Path,
        default=REPO_ROOT / "downstream/AffibodyMHC/configs/rde_libb_frozen_readouts_v1.json",
    )
    parser.add_argument(
        "--rde-pilot-root",
        type=Path,
        default=REPO_ROOT
        / "private_data/experiments/rde_libb_provider_revision_120_pilot_readouts_v2_common_env",
    )
    parser.add_argument(
        "--rde-feature-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/rde_libb_provider_revision_120_opaque_v1/manifest.json",
    )
    parser.add_argument(
        "--stab-config",
        type=Path,
        default=REPO_ROOT / "downstream/AffibodyMHC/configs/stab_libb_frozen_readouts_v1.json",
    )
    parser.add_argument(
        "--stab-pilot-root",
        type=Path,
        default=REPO_ROOT
        / "private_data/experiments/stab_libb_provider_revision_120_readout_pilot_v1",
    )
    parser.add_argument(
        "--stab-feature-manifest",
        type=Path,
        default=REPO_ROOT / "private_data/derived/stab_libb_provider_revision_120_merged_v1/manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_selection_record(
        rde_config=args.rde_config.resolve(),
        rde_pilot_root=args.rde_pilot_root.resolve(),
        rde_feature_manifest=args.rde_feature_manifest.resolve(),
        stab_config=args.stab_config.resolve(),
        stab_pilot_root=args.stab_pilot_root.resolve(),
        stab_feature_manifest=args.stab_feature_manifest.resolve(),
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
