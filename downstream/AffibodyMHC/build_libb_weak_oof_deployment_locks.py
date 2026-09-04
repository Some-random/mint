#!/usr/bin/env python3
"""Build prospective LibB score locks using weak-label OOF data only.

Five locks are emitted for the shortlist assembler: frozen MINT layer 5,
standalone StaB, standalone RDE, equal-logit MINT plus StaB, and the already
locked four-component stacker.
Cutoffs maximize F1 on matched weak-label out-of-fold predictions with an
explicit deterministic tie rule. Direct-retention data are neither accepted
nor read by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-weak-oof-deployment-locks-v2"
LOCK_SCHEMA_VERSION = "libb-score-ensemble-lock-v1"
WEAK_SCHEMA_VERSION = "libb-weak-oof-ensemble-all4-v1"
EXPECTED_ROWS = 10_181
EXPECTED_POSITIVE = 7_939
EXPECTED_FOLDS = (0, 1, 2)
MEMBERS = (
    "additive_7site",
    "mint_layer5",
    "stab_designed_ordered",
    "rde_network_designed_3fold",
)

DEFAULT_WEAK_DIR = (
    REPO_ROOT / "private_data/experiments/libb_weak_ensemble_selection_all4_v2"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "private_data/prospective/libb_weak_oof_deployment_locks_v2"
)

LOCK_DEFINITIONS = (
    {
        "filename": "mint_layer5_primary.lock.json",
        "lock_id": "libb-mint-layer5-primary-weak-oof-v1",
        "display_name": "Frozen MINT layer 5 primary",
        "oof_score_column": "mint_layer5",
        "components": (
            {
                "name": "mint_layer5",
                "score_column": "mint_layer5_logit",
                "weight": 1.0,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
        ),
        "intercept": 0.0,
        "weight_provenance": (
            "Single frozen MINT layer-5 model selected as the highest macro within-peptide "
            "AP candidate on 10,181 matched three-fold weak-label OOF rows; no retention used."
        ),
    },
    {
        "filename": "stab_standalone.lock.json",
        "lock_id": "libb-stab-standalone-weak-oof-v1",
        "display_name": "Standalone StaB alternative",
        "oof_score_column": "stab_designed_ordered",
        "components": (
            {
                "name": "stab_designed_ordered",
                "score_column": "stab_logit",
                "weight": 1.0,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
        ),
        "intercept": 0.0,
        "weight_provenance": (
            "Standalone locked StaB designed-residue readout evaluated on 10,181 matched "
            "three-fold weak-label OOF rows; no retention used for the score or cutoff."
        ),
    },
    {
        "filename": "rde_standalone.lock.json",
        "lock_id": "libb-rde-standalone-weak-oof-v1",
        "display_name": "Standalone RDE alternative",
        "oof_score_column": "rde_network_designed_3fold",
        "components": (
            {
                "name": "rde_network_designed_3fold",
                "score_column": "rde_logit",
                "weight": 1.0,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
        ),
        "intercept": 0.0,
        "weight_provenance": (
            "Standalone locked RDE-Network designed-site three-head ensemble evaluated on "
            "10,181 matched three-fold weak-label OOF rows; no retention used for the score "
            "or cutoff."
        ),
    },
    {
        "filename": "mint_stab_equal_logit_exploratory.lock.json",
        "lock_id": "libb-mint-stab-equal-logit-exploratory-weak-oof-v1",
        "display_name": "Equal-logit MINT + StaB exploratory",
        "oof_score_column": "mean_logit__mint_layer5__stab_designed_ordered",
        "components": (
            {
                "name": "mint_layer5",
                "score_column": "mint_layer5_logit",
                "weight": 0.5,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
            {
                "name": "stab_designed_ordered",
                "score_column": "stab_logit",
                "weight": 0.5,
                "center": 0.0,
                "scale": 1.0,
                "higher_is_better": True,
            },
        ),
        "intercept": 0.0,
        "weight_provenance": (
            "Equal weight in logit space; this was the highest macro within-peptide AP "
            "multi-model candidate on 10,181 matched three-fold weak-label OOF rows; "
            "no retention used."
        ),
    },
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _write_csv_exclusive(path: Path, frame: pd.DataFrame) -> None:
    _require(not Path(path).exists(), f"refusing to overwrite {path}")
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    os.chmod(path, 0o600)


def select_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Choose score >= cutoff by F1, precision, fewer rows, then cutoff."""

    scores_array = np.asarray(scores, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    _require(scores_array.ndim == labels_array.ndim == 1, "scores/labels must be vectors")
    _require(len(scores_array) == len(labels_array) and len(scores_array) > 0, "bad score length")
    _require(bool(np.isfinite(scores_array).all()), "non-finite weak OOF score")
    _require(set(np.unique(labels_array).tolist()) == {0, 1}, "weak OOF labels need both classes")
    positives = int(labels_array.sum())
    candidates = []
    for threshold in np.unique(scores_array):
        selected = scores_array >= threshold
        selected_count = int(selected.sum())
        tp = int(np.logical_and(selected, labels_array == 1).sum())
        fp = selected_count - tp
        fn = positives - tp
        tn = len(labels_array) - tp - fp - fn
        precision = Fraction(tp, selected_count)
        recall = Fraction(tp, positives)
        denominator = 2 * tp + fp + fn
        f1 = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
        value = {
            "threshold": float(threshold),
            "recommended": selected_count,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        candidates.append(((f1, precision, -selected_count, float(threshold)), value))
    return max(candidates, key=lambda item: item[0])[1]


def metrics_at_threshold(
    scores: Sequence[float], labels: Sequence[int], threshold: float
) -> dict[str, Any]:
    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    selected = scores_array >= threshold
    selected_count = int(selected.sum())
    tp = int(np.logical_and(selected, labels_array == 1).sum())
    fp = selected_count - tp
    positives = int(labels_array.sum())
    fn = positives - tp
    tn = len(labels_array) - tp - fp - fn
    return {
        "rows": len(labels_array),
        "positive": positives,
        "recommended": selected_count,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / selected_count if selected_count else 0.0,
        "recall": tp / positives if positives else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def _load_weak_artifacts(weak_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    weak_dir = Path(weak_dir).resolve()
    paths = {
        "oof": weak_dir / "matched_oof_predictions.csv.gz",
        "selection": weak_dir / "locked_weak_selection.json",
        "stacker": weak_dir / "stacker_locked_parameters.json",
        "manifest": weak_dir / "manifest.json",
    }
    for path in paths.values():
        _require(path.is_file(), f"missing weak artifact: {path}")
    with paths["selection"].open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    with paths["stacker"].open("r", encoding="utf-8") as handle:
        stacker = json.load(handle)
    with paths["manifest"].open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name, payload in (("selection", selection), ("stacker", stacker), ("manifest", manifest)):
        _require(payload.get("schema_version") == WEAK_SCHEMA_VERSION, f"{name} schema changed")
        _require(payload.get("retention_labels_read") is False, f"{name} read retention")
    _require(tuple(stacker["members_in_coefficient_order"]) == MEMBERS, "stacker order changed")
    _require(selection["best_candidate"]["candidate"] == "mean_logit__mint_layer5", "primary lock changed")
    _require(
        selection["best_ensemble_candidate"]["candidate"]
        == "mean_logit__mint_layer5__stab_designed_ordered",
        "exploratory ensemble lock changed",
    )
    frame = pd.read_csv(paths["oof"], dtype={"row_id": str})
    required = {
        "row_id",
        "fold",
        "weak_label",
        *MEMBERS,
        "mean_logit__mint_layer5__stab_designed_ordered",
        "cross_fitted_nonnegative_stack__all4",
    }
    _require(required.issubset(frame.columns), "weak OOF columns changed")
    _require(len(frame) == EXPECTED_ROWS and frame["row_id"].nunique() == EXPECTED_ROWS, "OOF rows changed")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="raise").astype(int)
    _require(tuple(sorted(frame["fold"].unique())) == EXPECTED_FOLDS, "OOF folds changed")
    _require(int(frame["weak_label"].sum()) == EXPECTED_POSITIVE, "OOF positive count changed")
    return frame, selection, stacker, {name: _record(path) for name, path in paths.items()}


def _stacker_lock_definition(stacker: Mapping[str, Any]) -> dict[str, Any]:
    score_columns = {
        "additive_7site": "additive_7site_logit",
        "mint_layer5": "mint_layer5_logit",
        "stab_designed_ordered": "stab_logit",
        "rde_network_designed_3fold": "rde_logit",
    }
    lengths = [
        len(stacker["members_in_coefficient_order"]),
        len(stacker["coefficient"]),
        len(stacker["mean"]),
        len(stacker["scale"]),
    ]
    _require(len(set(lengths)) == 1, "stacker parameter lengths differ")
    components = []
    for name, coefficient, center, scale in zip(
        stacker["members_in_coefficient_order"],
        stacker["coefficient"],
        stacker["mean"],
        stacker["scale"],
    ):
        components.append(
            {
                "name": str(name),
                "score_column": score_columns[str(name)],
                "weight": float(coefficient),
                "center": float(center),
                "scale": float(scale),
                "higher_is_better": True,
            }
        )
    return {
        "filename": "all4_stacker_locked.lock.json",
        "lock_id": "libb-all4-nonnegative-stacker-weak-oof-v1",
        "display_name": "Weak-label-locked four-model stacker",
        "oof_score_column": "cross_fitted_nonnegative_stack__all4",
        "components": tuple(components),
        "intercept": float(stacker["intercept"]),
        "weight_provenance": (
            "Non-negative standardized-logit logistic stacker; L2 and weights were selected/fitted "
            "using only matched three-fold weak-label OOF data. The cutoff uses its second-level "
            "cross-fitted predictions. No retention used."
        ),
    }


def _lock_payload(
    definition: Mapping[str, Any], threshold: Mapping[str, Any], source_records: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_SCHEMA_VERSION,
        "lock_id": definition["lock_id"],
        "display_name": definition["display_name"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": list(definition["components"]),
        "intercept": float(definition["intercept"]),
        "output_transform": "sigmoid",
        "selection": {
            "score_threshold": float(threshold["threshold"]),
            "max_candidates_per_peptide": 10,
            "minimum_code_hamming_distance": 0,
            "review_pool_per_peptide": 50,
            "pad_below_threshold": False,
        },
        "weight_selection_provenance": definition["weight_provenance"],
        "threshold_selection_provenance": (
            "F1-max cutoff on 10,181 matched weak-label OOF rows; exact ties prefer higher "
            "precision, then fewer recommendations, then the higher score. Direct retention "
            "labels were not read."
        ),
        "threshold_training_summary": dict(threshold),
        "score_semantics": (
            "The output is a selection-label score, not a retention percentage or a calibrated "
            "probability of direct-retention binding."
        ),
        "component_score_contract": (
            "Every configured component score column is a true pre-sigmoid logit. In particular, "
            "StaB/RDE consolidation must convert the saved mean-logit probability back to a "
            "clipped logit before writing stab_logit/rde_logit."
        ),
        "seed_aggregation_note": (
            "The structure weak-OOF scores come from the prespecified CV pilot, while prospective "
            "structure scores average five final readout seeds in logit space. The cutoff is "
            "transferred on the common sigmoid-score scale; this stability assumption must be "
            "checked prospectively and was not tuned with retention."
        ),
        "weak_source_sha256": {name: value["sha256"] for name, value in source_records.items()},
        "retention_labels_read": False,
    }


def render_report(audit: pd.DataFrame, locks: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# LibB weak-label-only deployment score locks",
        "",
        "These five candidate rules were frozen without reading the direct-retention matrix. ",
        "Their cutoffs maximize F1 on the same 10,181 matched weak-label out-of-fold rows used ",
        "for the ensemble comparison. They therefore predict the constructed selection label, ",
        "not a retention percentage.",
        "",
        "| Prospective score | Weak OOF cutoff | Recommended | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit.loc[audit["scope"].eq("all_oof")].itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.threshold:.9g} | {row.recommended}/{row.rows} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |"
        )
    lines.extend(
        [
            "",
            "The assembler submits at most ten above-cutoff Affibodies per peptide and never pads ",
            "with below-cutoff rows. No code-diversity exclusion is applied (`minimum Hamming ",
            "distance = 0`); the model ranking determines the first ten.",
            "",
            "## Exact score formulas",
            "",
            "- MINT primary: `sigmoid(mint_layer5_logit)`.",
            "- Standalone StaB: `sigmoid(stab_logit)`.",
            "- Standalone RDE: `sigmoid(rde_logit)`.",
            "- MINT + StaB: `sigmoid((mint_layer5_logit + stab_logit) / 2)`.",
            "- Four-model stacker: `sigmoid(intercept + sum(weight_j * ",
            "  (component_logit_j - center_j) / scale_j))`, using the values in its lock JSON.",
            "",
            "The StaB and RDE candidate-score merger must expose true logit columns named ",
            "`stab_logit` and `rde_logit`. Their current raw chunk field ",
            "`score_mean_logit` is a probability produced by taking the sigmoid after averaging ",
            "seed logits; it must not be passed to the assembler as though it were a logit.",
            "",
            "The structural OOF scores were produced by the prespecified CV pilot, whereas the ",
            "prospective structure files average five final readout seeds. These locks transfer ",
            "the OOF cutoff on the common sigmoid-score scale. That assumption is explicit and ",
            "will be tested by the prospective wet-lab batch; retention was not used to adjust it.",
            "",
            "## Lock files",
            "",
        ]
    )
    for lock in locks:
        lines.append(f"- `{lock['filename']}` — {lock['display_name']}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    weak_dir = Path(args.weak_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _require(not output_dir.exists(), f"output exists; refusing overwrite: {output_dir}")
    _require("private_data" in output_dir.parts, "output must remain below private_data")
    frame, selection, stacker, source_records = _load_weak_artifacts(weak_dir)
    definitions = (*LOCK_DEFINITIONS, _stacker_lock_definition(stacker))

    output_dir.mkdir(parents=True, mode=0o700)
    audit_rows: list[dict[str, Any]] = []
    lock_payloads: list[dict[str, Any]] = []
    lock_records: dict[str, Any] = {}
    for definition in definitions:
        score_column = str(definition["oof_score_column"])
        threshold = select_f1_threshold(frame[score_column], frame["weak_label"])
        payload = _lock_payload(definition, threshold, source_records)
        lock_path = output_dir / str(definition["filename"])
        _write_json_exclusive(lock_path, payload)
        lock_records[lock_path.name] = _record(lock_path)
        lock_payloads.append({**definition, "payload": payload})
        for scope, subset in [("all_oof", frame)] + [
            (f"fold_{fold}", frame.loc[frame["fold"].eq(fold)]) for fold in EXPECTED_FOLDS
        ]:
            values = metrics_at_threshold(subset[score_column], subset["weak_label"], threshold["threshold"])
            audit_rows.append(
                {
                    "lock_id": definition["lock_id"],
                    "display_name": definition["display_name"],
                    "oof_score_column": score_column,
                    "scope": scope,
                    "threshold": threshold["threshold"],
                    **values,
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit_path = output_dir / "weak_oof_threshold_audit.csv"
    _write_csv_exclusive(audit_path, audit)
    report_path = output_dir / "deployment_locks.md"
    report_path.write_text(render_report(audit, lock_payloads), encoding="utf-8")
    os.chmod(report_path, 0o600)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": float(time.time() - started),
        "rows": EXPECTED_ROWS,
        "positive": EXPECTED_POSITIVE,
        "folds": list(EXPECTED_FOLDS),
        "retention_labels_read": False,
        "selection": selection,
        "sources": source_records,
        "locks": lock_records,
        "outputs": {
            audit_path.name: _record(audit_path),
            report_path.name: _record(report_path),
        },
        "code": _record(Path(__file__).resolve()),
    }
    _write_json_exclusive(output_dir / "manifest.json", manifest)
    print(audit.loc[audit["scope"].eq("all_oof")].to_string(index=False))
    print(f"\nwrote {output_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-dir", default=str(DEFAULT_WEAK_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
