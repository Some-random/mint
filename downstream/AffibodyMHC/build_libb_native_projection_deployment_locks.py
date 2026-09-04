#!/usr/bin/env python3
"""Build prospective LibB score locks for the native-projection ensemble.

All weights, member choices, and optional filtering cutoffs come exclusively
from the 10,181 matched weak-label OOF rows.  Native weak-artifact member names
are mapped explicitly to the canonical score-column names used by exhaustive
inference; no outcome-bearing input is accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.AffibodyMHC import build_libb_weak_oof_deployment_locks as legacy
from downstream.AffibodyMHC import build_libb_weak_ensemble_native_projection as weak


SCHEMA_VERSION = "libb-native-projection-weak-oof-deployment-locks-v1"
LOCK_SCHEMA_VERSION = legacy.LOCK_SCHEMA_VERSION
EXPECTED_ROWS = weak.base.EXPECTED_OOF_ROWS
EXPECTED_POSITIVE = weak.base.EXPECTED_OOF_POSITIVE
EXPECTED_FOLDS = (0, 1, 2)

CANONICAL_COMPONENTS = {
    "additive_7site": ("additive_7site", "additive_7site_logit"),
    "mint_layer5": ("mint_layer5", "mint_layer5_logit"),
    weak.STAB_MEMBER: ("stab_designed_ordered", "stab_logit"),
    weak.RDE_MEMBER: ("rde_network_designed_3fold", "rde_logit"),
}

DEFAULT_WEAK_DIR = weak.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "private_data/prospective/libb_native_projection_weak_oof_deployment_locks_v1"
)


def _component(weak_name: str, weight: float, center: float = 0.0, scale: float = 1.0) -> dict[str, Any]:
    canonical_name, score_column = CANONICAL_COMPONENTS[weak_name]
    return {
        "name": canonical_name,
        "score_column": score_column,
        "weight": float(weight),
        "center": float(center),
        "scale": float(scale),
        "higher_is_better": True,
        "weak_oof_member_name": weak_name,
    }


def fixed_definitions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "filename": "mint_layer5_control.lock.json",
            "lock_id": "libb-native-mint-layer5-control-weak-oof-v1",
            "display_name": "Frozen MINT layer 5 control",
            "oof_score_column": "mint_layer5",
            "components": (_component("mint_layer5", 1.0),),
            "intercept": 0.0,
            "weight_provenance": "Best overall weak-label score; no retention used.",
        },
        {
            "filename": "mint_rde_primary_ensemble.lock.json",
            "lock_id": "libb-native-mint-rde-primary-ensemble-weak-oof-v1",
            "display_name": "MINT + RDE primary ensemble",
            "oof_score_column": f"mean_logit__mint_layer5__{weak.RDE_MEMBER}",
            "components": (
                _component("mint_layer5", 0.5),
                _component(weak.RDE_MEMBER, 0.5),
            ),
            "intercept": 0.0,
            "weight_provenance": (
                "Equal mean in logit space; highest macro within-peptide AP among all "
                "multi-model candidates on matched weak-label OOF rows; no retention used."
            ),
        },
        {
            "filename": "mint_stab_comparator.lock.json",
            "lock_id": "libb-native-mint-stab-comparator-weak-oof-v1",
            "display_name": "MINT + StaB comparator",
            "oof_score_column": f"mean_logit__mint_layer5__{weak.STAB_MEMBER}",
            "components": (
                _component("mint_layer5", 0.5),
                _component(weak.STAB_MEMBER, 0.5),
            ),
            "intercept": 0.0,
            "weight_provenance": "Prespecified equal-logit comparator; no retention used.",
        },
        {
            "filename": "equal_all4_comparator.lock.json",
            "lock_id": "libb-native-equal-all4-comparator-weak-oof-v1",
            "display_name": "Equal-logit four-model comparator",
            "oof_score_column": "mean_logit__" + "__".join(weak.ALL_MEMBERS),
            "components": tuple(_component(name, 0.25) for name in weak.ALL_MEMBERS),
            "intercept": 0.0,
            "weight_provenance": "One equal logit-space vote per component; no retention used.",
        },
        {
            "filename": "rde_standalone.lock.json",
            "lock_id": "libb-native-rde-standalone-weak-oof-v1",
            "display_name": "RDE native-projection standalone",
            "oof_score_column": weak.RDE_MEMBER,
            "components": (_component(weak.RDE_MEMBER, 1.0),),
            "intercept": 0.0,
            "weight_provenance": "Native-projection RDE weak-label OOF score; no retention used.",
        },
        {
            "filename": "stab_standalone.lock.json",
            "lock_id": "libb-native-stab-standalone-weak-oof-v1",
            "display_name": "StaB native-projection standalone",
            "oof_score_column": weak.STAB_MEMBER,
            "components": (_component(weak.STAB_MEMBER, 1.0),),
            "intercept": 0.0,
            "weight_provenance": "Native-projection StaB weak-label OOF score; no retention used.",
        },
    )


def stacker_definition(stacker: Mapping[str, Any]) -> dict[str, Any]:
    order = tuple(map(str, stacker["members_in_coefficient_order"]))
    legacy._require(order == weak.ALL_MEMBERS, "native stacker member order changed")
    lengths = {len(order), len(stacker["coefficient"]), len(stacker["mean"]), len(stacker["scale"])}
    legacy._require(lengths == {len(order)}, "native stacker parameter lengths differ")
    components = tuple(
        _component(name, coefficient, center, scale)
        for name, coefficient, center, scale in zip(
            order, stacker["coefficient"], stacker["mean"], stacker["scale"]
        )
    )
    return {
        "filename": "all4_stacker_comparator.lock.json",
        "lock_id": "libb-native-all4-stacker-comparator-weak-oof-v1",
        "display_name": "Weak-label-fitted four-model stacker comparator",
        "oof_score_column": "cross_fitted_nonnegative_stack__all4_native_projection",
        "components": components,
        "intercept": float(stacker["intercept"]),
        "weight_provenance": (
            "Nonnegative standardized-logit logistic stacker. L2, centers, scales, and "
            "weights were fitted only from matched weak-label OOF predictions."
        ),
    }


def load_weak(weak_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "oof": weak_dir / "matched_oof_predictions.csv.gz",
        "selection": weak_dir / "locked_weak_selection.json",
        "stacker": weak_dir / "stacker_locked_parameters.json",
        "manifest": weak_dir / "manifest.json",
    }
    for path in paths.values():
        legacy._require(path.is_file(), f"missing weak artifact {path}")
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    stacker = json.loads(paths["stacker"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for name, payload in (("selection", selection), ("stacker", stacker), ("manifest", manifest)):
        legacy._require(payload.get("schema_version") == weak.SCHEMA_VERSION, f"{name} schema changed")
        legacy._require(payload.get("retention_labels_read") is False, f"{name} read retention")
    legacy._require(selection["best_candidate"]["candidate"] == "mean_logit__mint_layer5", "MINT lock changed")
    legacy._require(
        selection["best_ensemble_candidate"]["candidate"]
        == f"mean_logit__mint_layer5__{weak.RDE_MEMBER}",
        "best weak-only ensemble changed",
    )
    frame = pd.read_csv(paths["oof"], dtype={"row_id": str})
    required = {
        "row_id", "fold", "weak_label", *weak.ALL_MEMBERS,
        f"mean_logit__mint_layer5__{weak.RDE_MEMBER}",
        f"mean_logit__mint_layer5__{weak.STAB_MEMBER}",
        "mean_logit__" + "__".join(weak.ALL_MEMBERS),
        "cross_fitted_nonnegative_stack__all4_native_projection",
    }
    legacy._require(required.issubset(frame.columns), "native weak OOF columns changed")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="raise").astype(int)
    legacy._require(len(frame) == EXPECTED_ROWS and frame["row_id"].nunique() == EXPECTED_ROWS, "weak rows changed")
    legacy._require(int(frame["weak_label"].sum()) == EXPECTED_POSITIVE, "weak positive count changed")
    legacy._require(tuple(sorted(frame["fold"].unique())) == EXPECTED_FOLDS, "folds changed")
    return frame, selection, stacker, {name: legacy._record(path) for name, path in paths.items()}


def lock_payload(definition: Mapping[str, Any], threshold: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
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
            "F1-max cutoff on the same 10,181 weak-label OOF rows; ties prefer higher "
            "precision, fewer recommendations, then higher score. No retention used."
        ),
        "threshold_training_summary": dict(threshold),
        "score_semantics": "Selection-label score, not retention percentage or calibrated retention probability.",
        "component_score_contract": (
            "Inference uses canonical true-logit columns: additive_7site_logit, "
            "mint_layer5_logit, stab_logit, and rde_logit."
        ),
        "seed_aggregation_note": (
            "Prospective StaB/RDE scores average five final native-projection readout seeds "
            "in logit space."
        ),
        "weak_source_sha256": {name: record["sha256"] for name, record in sources.items()},
        "retention_labels_read": False,
    }


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    legacy._require(not path.exists(), f"refusing to overwrite {path}")
    frame.to_csv(path, index=False, float_format="%.12g")
    os.chmod(path, 0o600)


def render_report(audit: pd.DataFrame) -> str:
    rows = audit.loc[audit["scope"].eq("all_oof")]
    lines = [
        "# Native-projection weak-label deployment locks",
        "",
        "MINT alone remains the best overall weak-label model. MINT + RDE is the primary ",
        "multi-model ensemble because it has the highest within-peptide weak-label AP among ",
        "the multi-model candidates. This calculation did not read retention measurements. ",
        "However, the retention panel had already been examined elsewhere during model ",
        "development, so this is not an untouched prospective model-selection event.",
        "",
        "The cutoff is an optional weak-label filtering rule, not a performance metric. The ",
        "wet-lab shortlist still takes at most the ten highest-scoring passing designs per peptide.",
        "",
        "| Score | Weak-OOF cutoff | Passing rows | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.threshold:.7g} | {row.recommended}/{row.rows} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    weak_dir = Path(args.weak_dir).resolve()
    output = Path(args.output_dir).resolve()
    legacy._require(not output.exists(), f"output exists: {output}")
    legacy._require("private_data" in output.parts, "output must remain under private_data")
    frame, selection, stacker, sources = load_weak(weak_dir)
    definitions = (*fixed_definitions(), stacker_definition(stacker))
    output.mkdir(parents=True, mode=0o700)
    audit_rows = []
    lock_records = {}
    for definition in definitions:
        column = str(definition["oof_score_column"])
        threshold = legacy.select_f1_threshold(frame[column], frame["weak_label"])
        payload = lock_payload(definition, threshold, sources)
        path = output / str(definition["filename"])
        legacy._write_json_exclusive(path, payload)
        lock_records[path.name] = legacy._record(path)
        for scope, subset in [("all_oof", frame)] + [
            (f"fold_{fold}", frame.loc[frame["fold"].eq(fold)]) for fold in EXPECTED_FOLDS
        ]:
            audit_rows.append(
                {
                    "lock_id": definition["lock_id"],
                    "display_name": definition["display_name"],
                    "oof_score_column": column,
                    "scope": scope,
                    "threshold": threshold["threshold"],
                    **legacy.metrics_at_threshold(subset[column], subset["weak_label"], threshold["threshold"]),
                }
            )
    audit = pd.DataFrame(audit_rows)
    audit_path = output / "weak_oof_threshold_audit.csv"
    write_csv(audit_path, audit)
    report_path = output / "deployment_locks.md"
    report_path.write_text(render_report(audit), encoding="utf-8")
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
        "native_to_canonical_component_mapping": {
            name: {"component_name": canonical, "score_column": column}
            for name, (canonical, column) in CANONICAL_COMPONENTS.items()
        },
        "sources": sources,
        "locks": lock_records,
        "outputs": {
            audit_path.name: legacy._record(audit_path),
            report_path.name: legacy._record(report_path),
        },
        "code": legacy._record(Path(__file__).resolve()),
    }
    legacy._write_json_exclusive(output / "manifest.json", manifest)
    print(audit.loc[audit["scope"].eq("all_oof")].to_string(index=False))
    print(f"wrote {output}")
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
