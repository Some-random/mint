#!/usr/bin/env python3
"""Assemble a frozen LibB ensemble into prospective wet-lab candidate sheets.

This program does not run, fit, or tune any model.  It joins already-produced
component scores to the selection-missed candidate universe, applies weights
and a score cutoff from an immutable JSON lock, and emits a compact ranked
review pool plus a diversity-aware wet-lab shortlist.

Retention outcomes are forbidden in component-score files.  A score lock must
state how its weights and cutoff were chosen so that retrospective threshold
selection is not confused with prospective validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_SCORE_COLUMNS = {
    "target_retention",
    "retention_percent",
    "target_binder",
    "binder_label_ge_75",
    "measurement_missing",
}

# This stage is deliberately limited to the corrected 12-target LibB panel.
# Keeping the identities here makes accidental partial/extra-target scoring a
# hard error rather than a silently different prospective experiment.
EXPECTED_TARGET_SEQUENCES = {
    "AF": "SLLAFITQV",
    "AH": "SLLAHITQV",
    "DP": "SLLDPITQV",
    "EA": "SLLEAITQV",
    "EL": "SLLELITQV",
    "LL": "SLLLLITQV",
    "LV": "SLLLVITQV",
    "MW": "SLLMWITQV",
    "NF": "SLLNFITQV",
    "PH": "SLLPHITQV",
    "TL": "SLLTLITQV",
    "VV": "SLLVVITQV",
}

LOCK_COMPONENT_CONTRACTS = {
    "libb-all4-nonnegative-stacker-weak-oof-v1": (
        "additive_7site",
        "mint_layer5",
        "stab_designed_ordered",
        "rde_network_designed_3fold",
    ),
    "libb-mint-layer5-primary-weak-oof-v1": ("mint_layer5",),
    "libb-mint-stab-equal-logit-exploratory-weak-oof-v1": (
        "mint_layer5",
        "stab_designed_ordered",
    ),
    "libb-rde-standalone-weak-oof-v1": ("rde_network_designed_3fold",),
    "libb-stab-standalone-weak-oof-v1": ("stab_designed_ordered",),
    "libb-native-mint-layer5-control-weak-oof-v1": ("mint_layer5",),
    "libb-native-mint-rde-primary-ensemble-weak-oof-v1": (
        "mint_layer5",
        "rde_network_designed_3fold",
    ),
    "libb-native-mint-stab-comparator-weak-oof-v1": (
        "mint_layer5",
        "stab_designed_ordered",
    ),
    "libb-native-equal-all4-comparator-weak-oof-v1": (
        "additive_7site",
        "mint_layer5",
        "stab_designed_ordered",
        "rde_network_designed_3fold",
    ),
    "libb-native-all4-stacker-comparator-weak-oof-v1": (
        "additive_7site",
        "mint_layer5",
        "stab_designed_ordered",
        "rde_network_designed_3fold",
    ),
    "libb-native-rde-standalone-weak-oof-v1": ("rde_network_designed_3fold",),
    "libb-native-stab-standalone-weak-oof-v1": ("stab_designed_ordered",),
}

BUNDLE_SCHEMA_VERSIONS = {
    "libb-weak-oof-deployment-locks-v2",
    "libb-native-projection-weak-oof-deployment-locks-v1",
}

FORBIDDEN_OUTCOME_COLUMN_FRAGMENTS = (
    "retention",
    "binder_label",
    "binding_label",
    "ground_truth",
    "wetlab_outcome",
    "direct_measurement",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_private_output_path(output_dir: Path) -> Path:
    repo_root = REPO_ROOT.resolve()
    private_root = (repo_root / "private_data").resolve()
    output_dir = output_dir.resolve()
    _require(private_root.is_dir(), "repository private_data directory does not exist")
    _require(output_dir != private_root, "output must be below private_data")
    try:
        output_dir.relative_to(private_root)
    except ValueError as error:
        raise ValueError("output directory must be below private_data") from error
    return output_dir


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(path, 0o600)


def _forbidden_outcome_columns(columns) -> list[str]:
    leaked = set(FORBIDDEN_SCORE_COLUMNS) & set(columns)
    for column in columns:
        normalized = str(column).strip().lower()
        if any(fragment in normalized for fragment in FORBIDDEN_OUTCOME_COLUMN_FRAGMENTS):
            leaked.add(str(column))
    return sorted(leaked)


def validate_lock_bundle(bundle_path: Path, lock_path: Path) -> dict:
    """Verify that the supplied lock is one emitted by the weak-OOF bundle."""
    bundle_path = bundle_path.resolve()
    _require(bundle_path.is_file(), "deployment-lock bundle manifest does not exist")
    with bundle_path.open(encoding="utf-8") as handle:
        bundle = json.load(handle)
    _require(
        bundle.get("schema_version") in BUNDLE_SCHEMA_VERSIONS,
        "unexpected deployment-lock bundle schema",
    )
    _require(bundle.get("retention_labels_read") is False,
             "deployment-lock bundle does not explicitly forbid retention use")
    selection = bundle.get("selection")
    _require(isinstance(selection, dict), "deployment-lock bundle selection audit missing")
    _require(selection.get("retention_labels_read") is False,
             "weak-OOF selection audit does not explicitly forbid retention use")
    _require(selection.get("selection_data") == "selection-derived weak binder/non-binder labels only",
             "unexpected deployment-lock training-label provenance")
    _require(int(bundle.get("rows", 0)) > 0, "deployment-lock weak-OOF row count missing")

    lock_record = bundle.get("locks", {}).get(lock_path.name)
    _require(isinstance(lock_record, dict),
             f"lock {lock_path.name} is not registered in the deployment-lock bundle")
    _require(lock_record.get("sha256") == sha256_file(lock_path),
             "deployment lock hash differs from the registered weak-OOF artifact")
    return bundle


def parse_component_path(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        _require("=" in value, f"component path must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        _require(name, f"empty component name in {value}")
        _require(name not in result, f"duplicate component path for {name}")
        path = Path(raw_path).expanduser().resolve()
        _require(path.exists(), f"component-score path does not exist: {path}")
        result[name] = path
    return result


def load_lock(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    _require(lock.get("schema_version") == "libb-score-ensemble-lock-v1",
             "unsupported ensemble-lock schema")
    lock_id = lock.get("lock_id")
    _require(lock_id in LOCK_COMPONENT_CONTRACTS, "unknown deployment lock role")
    _require(isinstance(lock.get("display_name"), str) and lock["display_name"].strip(),
             "deployment lock display name missing")
    _require(lock.get("retention_labels_read") is False,
             "deployment lock does not explicitly forbid retention-label use")
    _require(isinstance(lock.get("score_semantics"), str) and lock["score_semantics"].strip(),
             "deployment lock score semantics missing")
    _require(isinstance(lock.get("component_score_contract"), str)
             and lock["component_score_contract"].strip(),
             "deployment lock component-score contract missing")
    weak_hashes = lock.get("weak_source_sha256")
    _require(isinstance(weak_hashes, dict) and weak_hashes,
             "deployment lock weak-source hashes missing")
    _require(
        all(isinstance(value, str) and len(value) == 64 for value in weak_hashes.values()),
        "deployment lock contains malformed weak-source hashes",
    )
    components = lock.get("components")
    _require(isinstance(components, list) and components, "lock needs a non-empty components list")
    names = [entry.get("name") for entry in components]
    _require(all(isinstance(name, str) and name for name in names), "component name missing")
    _require(len(names) == len(set(names)), "duplicate component in lock")
    _require(tuple(names) == LOCK_COMPONENT_CONTRACTS[lock_id],
             f"component order does not match registered role {lock_id}")
    for entry in components:
        _require(isinstance(entry.get("score_column"), str) and entry["score_column"],
                 f"score column missing for {entry['name']}")
        for key in ("weight", "center", "scale"):
            _require(isinstance(entry.get(key), (int, float)),
                     f"numeric {key} missing for {entry['name']}")
            _require(math.isfinite(float(entry[key])), f"non-finite {key} for {entry['name']}")
        _require(float(entry["scale"]) > 0, f"scale must be positive for {entry['name']}")
        _require(entry.get("higher_is_better") is True,
                 f"{entry['name']} must be explicitly oriented higher-is-better")
    _require(isinstance(lock.get("intercept"), (int, float)), "numeric intercept missing")
    _require(lock.get("output_transform") in {"sigmoid", "identity"},
             "output_transform must be sigmoid or identity")
    selection = lock.get("selection")
    _require(isinstance(selection, dict), "selection policy missing from lock")
    for key in ("score_threshold", "max_candidates_per_peptide",
                "minimum_code_hamming_distance", "review_pool_per_peptide"):
        _require(key in selection, f"selection.{key} missing")
    _require(math.isfinite(float(selection["score_threshold"])), "score threshold is non-finite")
    _require(0 < int(selection["max_candidates_per_peptide"]) <= 10,
             "max candidates must be between one and the wet-lab cap of ten")
    _require(0 <= int(selection["minimum_code_hamming_distance"]) <= 5,
             "minimum code Hamming distance must be 0--5")
    _require(int(selection["review_pool_per_peptide"]) >= int(selection["max_candidates_per_peptide"]),
             "review pool must be at least as large as wet-lab cap")
    _require(selection.get("pad_below_threshold") is False,
             "below-threshold padding must be explicitly false")
    for key in ("weight_selection_provenance", "threshold_selection_provenance"):
        _require(isinstance(lock.get(key), str) and lock[key].strip(), f"{key} missing")
    return lock


def resolve_score_file(root: Path, peptide: str) -> Path:
    if root.is_file():
        _require(root.suffix == ".parquet", f"score file is not Parquet: {root}")
        return root
    exact_candidates = [
        root / f"peptide_{peptide}.parquet",
        root / f"{peptide}.parquet",
        root / f"peptide={peptide}" / "scores.parquet",
    ]
    exact = [path for path in exact_candidates if path.is_file()]
    if len(exact) == 1:
        return exact[0]
    matches = sorted(path for path in root.rglob("*.parquet") if peptide in path.stem)
    _require(len(matches) == 1,
             f"could not uniquely resolve score Parquet for peptide {peptide} below {root}")
    return matches[0]


def validate_component_score_provenance(
    root: Path,
    score_path: Path,
    component_name: str,
    peptide: str,
    universe_manifest_sha256: str,
    universe_partition_sha256: str,
    expected_readout_mode: str | None = None,
) -> dict[str, str]:
    """Validate the producer receipt for one candidate-score partition."""
    score_sha256 = sha256_file(score_path)
    if component_name == "mint_layer5":
        receipt_path = score_path.with_suffix(score_path.suffix + ".manifest.json")
        _require(receipt_path.is_file(), f"MINT score receipt missing for {peptide}")
        with receipt_path.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
        _require(receipt.get("schema_version") == "mint-layer5-streaming-candidate-scores-v1",
                 f"unexpected MINT score receipt schema for {peptide}")
        _require(receipt.get("peptide_design_code") == peptide,
                 f"MINT score receipt target mismatch for {peptide}")
        _require(receipt.get("inputs", {}).get("candidate_parquet", {}).get("sha256")
                 == universe_partition_sha256,
                 f"MINT score receipt candidate hash mismatch for {peptide}")
        _require(receipt.get("output", {}).get("sha256") == score_sha256,
                 f"MINT score hash mismatch for {peptide}")
        model = receipt.get("model", {})
        _require(model.get("layer") == 5
                 and model.get("feature_name") == "mint_layer_05_chain_mean",
                 f"MINT score receipt identifies the wrong feature for {peptide}")
        return {
            "score": str(score_path),
            "score_sha256": score_sha256,
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }

    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"{component_name} score manifest is missing")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if component_name == "additive_7site":
        _require(manifest.get("schema_version") == "libb-additive-candidate-scores-v1",
                 "unexpected additive score schema")
        _require(manifest.get("retention_or_training_labels_read") is False,
                 "additive scorer does not explicitly declare outcome-free inference")
        _require(manifest.get("candidate_manifest_sha256") == universe_manifest_sha256,
                 "additive score candidate-universe hash mismatch")
        records = {str(row["peptide"]): row for row in manifest.get("partitions", [])}
        _require(peptide in records, f"additive manifest lacks {peptide}")
        _require(records[peptide].get("sha256") == score_sha256,
                 f"additive score hash mismatch for {peptide}")
    elif component_name in {"stab_designed_ordered", "rde_network_designed_3fold"}:
        family = "stab" if component_name == "stab_designed_ordered" else "rde"
        _require(manifest.get("schema_version")
                 == "libb-fixed-structure-candidate-scores-merged-v1",
                 f"unexpected {family} merged-score schema")
        _require(manifest.get("family") == family, f"wrong structure family for {component_name}")
        access = manifest.get("outcome_access", {})
        _require(access.get("selection_labels_read") is False
                 and access.get("retention_measurements_read") is False,
                 f"{family} score manifest does not explicitly declare outcome-free inference")
        _require(manifest.get("candidate_universe", {}).get("manifest_sha256")
                 == universe_manifest_sha256,
                 f"{family} score candidate-universe hash mismatch")
        if expected_readout_mode is not None:
            readout = manifest.get("source_chunks", {}).get("readout_provenance", {})
            _require(
                readout.get("readout_mode") == expected_readout_mode,
                f"{family} scores use {readout.get('readout_mode')!r}, expected "
                f"{expected_readout_mode!r}",
            )
        records = {
            str(row["peptide_design_code"]): row
            for row in manifest.get("outputs", {}).get("partitions", [])
        }
        _require(peptide in records, f"{family} manifest lacks {peptide}")
        _require(records[peptide].get("universe_partition_sha256")
                 == universe_partition_sha256,
                 f"{family} candidate partition hash mismatch for {peptide}")
        _require(records[peptide].get("output_sha256") == score_sha256,
                 f"{family} score hash mismatch for {peptide}")
    else:  # pragma: no cover - load_lock already closes this path.
        raise ValueError(f"no score-provenance contract for {component_name}")
    return {
        "score": str(score_path),
        "score_sha256": score_sha256,
        "receipt": str(manifest_path),
        "receipt_sha256": sha256_file(manifest_path),
    }


def load_component_scores(
    path: Path,
    peptide: str,
    component_name: str,
    score_column: str,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    leaked = _forbidden_outcome_columns(frame.columns)
    _require(not leaked, f"{component_name} score file contains forbidden outcomes: {sorted(leaked)}")
    _require({"pair_uid", score_column}.issubset(frame.columns),
             f"{component_name} score file lacks pair_uid/{score_column}")
    if "peptide_design_code" in frame.columns:
        frame = frame.loc[frame["peptide_design_code"].astype(str).eq(peptide)].copy()
    _require(not frame.empty, f"{component_name} has no scores for peptide {peptide}")
    _require(not frame["pair_uid"].duplicated().any(),
             f"{component_name} contains duplicate pair_uid for {peptide}")
    scores = pd.to_numeric(frame[score_column], errors="raise").astype(float)
    _require(np.isfinite(scores).all(), f"{component_name} has non-finite scores")
    output = pd.DataFrame({"pair_uid": frame["pair_uid"].astype(str), component_name: scores})
    if "affibody_design_code" in frame.columns:
        output["_score_affibody_code"] = frame["affibody_design_code"].astype(str).to_numpy()
    return output


def hamming_distance(left: str, right: str) -> int:
    _require(len(left) == len(right) == 5, "LibB Affibody codes must have length five")
    return sum(a != b for a, b in zip(left, right))


def diversity_select(
    ranked: pd.DataFrame,
    threshold: float,
    maximum: int,
    minimum_hamming: int,
) -> pd.DataFrame:
    eligible = ranked.loc[ranked["ensemble_score"].ge(threshold)].copy()
    selected_indices = []
    selected_codes: list[str] = []
    for index, row in eligible.iterrows():
        code = str(row["affibody_design_code"])
        if all(hamming_distance(code, existing) >= minimum_hamming for existing in selected_codes):
            selected_indices.append(index)
            selected_codes.append(code)
            if len(selected_indices) == maximum:
                break
    selected = eligible.loc[selected_indices].copy()
    selected["wetlab_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    selected["minimum_hamming_distance_to_other_selected"] = [
        min(
            (hamming_distance(code, other) for other in selected_codes if other != code),
            default=5,
        )
        for code in selected_codes
    ]
    return selected


def score_partition(
    universe: pd.DataFrame,
    peptide: str,
    lock: dict,
    component_paths: dict[str, Path],
    universe_manifest_sha256: str | None = None,
    universe_partition_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    leaked_universe = _forbidden_outcome_columns(universe.columns)
    _require(not leaked_universe,
             f"candidate universe contains forbidden outcomes: {leaked_universe}")
    _require(not universe["pair_uid"].duplicated().any(), f"universe duplicate pair_uid for {peptide}")
    _require(universe["peptide_design_code"].astype(str).eq(peptide).all(),
             f"wrong peptide rows in universe partition {peptide}")
    frame = universe.copy()
    source_files = {}
    normalized_columns = []
    contribution_columns = []
    component_rank_columns = []
    expected_readout_mode = (
        "native_learned_projection"
        if str(lock.get("lock_id", "")).startswith("libb-native-")
        else None
    )
    for component in lock["components"]:
        name = component["name"]
        path = resolve_score_file(component_paths[name], peptide)
        if universe_manifest_sha256 is not None and universe_partition_sha256 is not None:
            source_record = validate_component_score_provenance(
                component_paths[name],
                path,
                name,
                peptide,
                universe_manifest_sha256,
                universe_partition_sha256,
                expected_readout_mode=expected_readout_mode,
            )
        else:
            source_record = {
                "score": str(path),
                "score_sha256": sha256_file(path),
                "receipt": "not_checked_by_unit_test_helper",
                "receipt_sha256": "not_checked_by_unit_test_helper",
            }
        source_files[name] = source_record
        scores = load_component_scores(path, peptide, name, component["score_column"])
        if "_score_affibody_code" in scores.columns:
            expected = frame[["pair_uid", "affibody_design_code"]].merge(
                scores[["pair_uid", "_score_affibody_code"]],
                on="pair_uid",
                how="inner",
                validate="one_to_one",
            )
            _require(
                expected["affibody_design_code"].astype(str).eq(
                    expected["_score_affibody_code"].astype(str)
                ).all(),
                f"{name} pair_uid/Affibody mapping mismatch for {peptide}",
            )
            scores = scores.drop(columns="_score_affibody_code")
        before = len(frame)
        frame = frame.merge(scores, on="pair_uid", how="left", validate="one_to_one")
        _require(len(frame) == before, f"{name} changed universe row count for {peptide}")
        _require(frame[name].notna().all(), f"{name} lacks one or more universe scores for {peptide}")
        normalized = f"{name}_normalized"
        contribution = f"{name}_weighted_contribution"
        frame[normalized] = (frame[name] - float(component["center"])) / float(component["scale"])
        frame[contribution] = float(component["weight"]) * frame[normalized]
        # Keep a directly interpretable per-model rank in the review and wet-lab
        # sheets. Ties are broken by the stable candidate-universe order; the
        # model score itself remains unchanged.
        component_rank = f"within_peptide_{name}_rank"
        frame[component_rank] = frame[name].rank(
            method="first", ascending=False
        ).astype(np.int64)
        normalized_columns.append(normalized)
        contribution_columns.append(contribution)
        component_rank_columns.append(component_rank)

    frame["ensemble_linear_score"] = float(lock["intercept"]) + frame[
        contribution_columns
    ].sum(axis=1)
    if lock["output_transform"] == "sigmoid":
        clipped = np.clip(frame["ensemble_linear_score"].to_numpy(float), -700, 700)
        frame["ensemble_score"] = 1.0 / (1.0 + np.exp(-clipped))
    else:
        frame["ensemble_score"] = frame["ensemble_linear_score"]
    frame["normalized_component_disagreement"] = frame[normalized_columns].std(
        axis=1, ddof=0
    )
    frame["component_top10_votes"] = frame[component_rank_columns].le(10).sum(axis=1)
    frame = frame.sort_values(
        ["ensemble_score", "pair_uid"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    frame["within_peptide_ensemble_rank"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    return frame, source_files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-dir", required=True, type=Path)
    parser.add_argument("--lock-json", required=True, type=Path)
    parser.add_argument("--lock-bundle-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-lock-id",
        required=True,
        choices=tuple(LOCK_COMPONENT_CONTRACTS),
        help="Fail closed unless the supplied lock has this registered deployment role.",
    )
    parser.add_argument(
        "--component-scores",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat once per locked component; PATH may be a Parquet or peptide-partition directory.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def run(args) -> None:
    started = time.time()
    output_dir = validate_private_output_path(args.output_dir)
    _require(not output_dir.exists(), "output directory exists; refusing overwrite")
    _require(args.universe_dir.is_dir(), "candidate universe directory does not exist")
    _require(args.lock_json.is_file(), "ensemble lock does not exist")
    lock = load_lock(args.lock_json)
    _require(lock["lock_id"] == args.expected_lock_id,
             "deployment lock does not match --expected-lock-id")
    lock_bundle = validate_lock_bundle(args.lock_bundle_manifest, args.lock_json.resolve())
    bundle_source_hashes = {
        name: record.get("sha256")
        for name, record in lock_bundle.get("sources", {}).items()
    }
    _require(lock["weak_source_sha256"] == bundle_source_hashes,
             "deployment lock weak-source hashes differ from its bundle")
    for name, record in lock_bundle.get("sources", {}).items():
        source_path = Path(str(record.get("path", "")))
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        _require(source_path.is_file(), f"weak-OOF provenance source is missing: {name}")
        _require(sha256_file(source_path) == record.get("sha256"),
                 f"weak-OOF provenance source hash changed: {name}")
    component_paths = parse_component_path(args.component_scores)
    locked_names = {entry["name"] for entry in lock["components"]}
    _require(set(component_paths) == locked_names,
             f"component paths {sorted(component_paths)} do not match lock {sorted(locked_names)}")

    universe_manifest_path = args.universe_dir / "manifest.json"
    partitions_path = args.universe_dir / "candidate_partitions.csv"
    _require(universe_manifest_path.is_file(), "candidate-universe manifest missing")
    _require(partitions_path.is_file(), "candidate partition index missing")
    with universe_manifest_path.open(encoding="utf-8") as handle:
        universe_manifest = json.load(handle)
    _require(
        universe_manifest.get("schema_version")
        == "libb-existing-target-selection-missed-candidates-v1",
        "unexpected candidate-universe schema",
    )
    _require(universe_manifest.get("no_model_scores") is True,
             "candidate-universe manifest no longer declares an unscored universe")
    scope = universe_manifest.get("scope", {})
    _require(scope.get("library") == "LibB", "candidate universe is not LibB")
    _require(scope.get("target_peptide_codes") == list(EXPECTED_TARGET_SEQUENCES),
             "candidate-universe target order/identities changed")
    _require(scope.get("target_peptide_full_sequences") == EXPECTED_TARGET_SEQUENCES,
             "candidate-universe target sequence mapping changed")
    partitions = pd.read_csv(partitions_path)
    _require(len(partitions) == 12, "candidate universe must contain 12 peptide partitions")
    _require(not partitions["peptide_design_code"].astype(str).duplicated().any(),
             "candidate partition index contains duplicate targets")
    _require(set(partitions["peptide_design_code"].astype(str))
             == set(EXPECTED_TARGET_SEQUENCES),
             "candidate partition index is not the exact corrected 12-target set")
    for row in partitions.itertuples(index=False):
        _require(str(row.peptide_full_sequence)
                 == EXPECTED_TARGET_SEQUENCES[str(row.peptide_design_code)],
                 f"candidate partition sequence mismatch for {row.peptide_design_code}")

    output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    shortlist_blocks = []
    review_blocks = []
    summary_rows = []
    component_files_by_peptide = {}
    selection = lock["selection"]
    for row in partitions.sort_values("peptide_design_code").itertuples(index=False):
        peptide = str(row.peptide_design_code)
        universe_path = args.universe_dir / str(row.partition)
        _require(universe_path.is_file(), f"universe partition missing: {universe_path}")
        _require(sha256_file(universe_path) == str(row.partition_sha256),
                 f"universe partition hash mismatch for {peptide}")
        universe = pd.read_parquet(universe_path)
        _require(len(universe) == int(row.candidate_rows),
                 f"universe row count mismatch for {peptide}")
        ranked, source_files = score_partition(
            universe,
            peptide,
            lock,
            component_paths,
            universe_manifest_sha256=sha256_file(universe_manifest_path),
            universe_partition_sha256=str(row.partition_sha256),
        )
        component_files_by_peptide[peptide] = source_files
        review = ranked.head(int(selection["review_pool_per_peptide"])).copy()
        selected = diversity_select(
            ranked,
            threshold=float(selection["score_threshold"]),
            maximum=int(selection["max_candidates_per_peptide"]),
            minimum_hamming=int(selection["minimum_code_hamming_distance"]),
        )
        review_blocks.append(review)
        shortlist_blocks.append(selected)
        _require(len(selected) <= 10, f"wet-lab cap exceeded for {peptide}")
        _require(selected["wetlab_rank"].tolist() == list(range(1, len(selected) + 1)),
                 f"non-contiguous wet-lab ranks for {peptide}")
        above = ranked["ensemble_score"].ge(float(selection["score_threshold"]))
        summary_rows.append(
            {
                "peptide_design_code": peptide,
                "peptide_full_sequence": str(row.peptide_full_sequence),
                "universe_rows_scored": int(len(ranked)),
                "rows_above_locked_cutoff": int(above.sum()),
                "wetlab_candidates_selected": int(len(selected)),
                "highest_ensemble_score": float(ranked["ensemble_score"].iloc[0]),
                "lowest_selected_ensemble_score": (
                    float(selected["ensemble_score"].min()) if len(selected) else np.nan
                ),
                "selected_previously_unobserved_all_rounds": int(
                    (~selected["observed_in_any_raw_round"]).sum()
                ),
                "selected_high_confidence_weak_negatives": int(
                    selected["high_confidence_weak_negative"].sum()
                ),
                "selected_affibody_identities_unseen_in_strict_training": int(
                    (~selected["affibody_identity_seen_in_strict_training"]).sum()
                ),
            }
        )

    shortlist = pd.concat(shortlist_blocks, ignore_index=True)
    review = pd.concat(review_blocks, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    shortlist = shortlist.sort_values(
        ["peptide_design_code", "wetlab_rank"], kind="mergesort"
    ).reset_index(drop=True)
    review = review.sort_values(
        ["peptide_design_code", "within_peptide_ensemble_rank"], kind="mergesort"
    ).reset_index(drop=True)
    # A locked score cutoff may legitimately reject every candidate for a
    # target. Do not pad such a target with below-cutoff rows. The review pool
    # and summary still cover all 12 targets, while the shortlist contains only
    # rows that actually pass the frozen rule.
    _require(set(shortlist["peptide_design_code"].astype(str)).issubset(
        set(EXPECTED_TARGET_SEQUENCES)
    ), "shortlist contains an unexpected peptide target")
    _require(set(review["peptide_design_code"].astype(str))
             == set(EXPECTED_TARGET_SEQUENCES),
             "review pool is missing a corrected LibB target")
    for peptide, block in shortlist.groupby("peptide_design_code", sort=True):
        _require(len(block) <= 10, f"wet-lab cap exceeded after assembly for {peptide}")
        _require(block["wetlab_rank"].astype(int).tolist()
                 == list(range(1, len(block) + 1)),
                 f"shortlist ranks are not exactly 1..N for {peptide}")
    for peptide, block in review.groupby("peptide_design_code", sort=True):
        _require(block["within_peptide_ensemble_rank"].astype(int).tolist()
                 == list(range(1, len(block) + 1)),
                 f"review-pool ranks are not exactly 1..N for {peptide}")

    # One Affibody can be synthesized once and tested against several peptide
    # targets. Expose repeated choices instead of silently presenting them as
    # unrelated rows; repetition may be experimentally efficient, but it can
    # also reveal a broadly favoured rather than peptide-specific candidate.
    if len(shortlist):
        target_count = shortlist.groupby("affibody_design_code")[
            "peptide_design_code"
        ].transform("nunique")
        target_list = shortlist.groupby("affibody_design_code")[
            "peptide_design_code"
        ].transform(lambda values: ";".join(sorted(set(map(str, values)))))
        shortlist["selected_for_n_peptides"] = target_count.astype(np.int64)
        shortlist["selected_for_peptides"] = target_list

    # CSV is deliberate here: these are compact human-facing handoff tables,
    # while the multi-million-row universe remains in Parquet.
    shortlist_path = output_dir / "wetlab_shortlist.csv"
    review_path = output_dir / "ranked_review_pool.csv"
    summary_path = output_dir / "wetlab_shortlist_summary.csv"
    constructs_path = output_dir / "unique_affibody_constructs.csv"
    shortlist.to_csv(shortlist_path, index=False)
    review.to_csv(review_path, index=False)
    summary.to_csv(summary_path, index=False)
    if len(shortlist):
        constructs = (
            shortlist.groupby(
                ["affibody_design_code", "chain2_affibody_sequence"], as_index=False
            )
            .agg(
                selected_for_n_peptides=("peptide_design_code", "nunique"),
                selected_for_peptides=(
                    "peptide_design_code",
                    lambda values: ";".join(sorted(set(map(str, values)))),
                ),
                best_wetlab_rank=("wetlab_rank", "min"),
            )
            .sort_values(
                ["selected_for_n_peptides", "affibody_design_code"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
    else:
        constructs = pd.DataFrame(
            columns=[
                "affibody_design_code",
                "chain2_affibody_sequence",
                "selected_for_n_peptides",
                "selected_for_peptides",
                "best_wetlab_rank",
            ]
        )
    constructs.to_csv(constructs_path, index=False)
    for path in (shortlist_path, review_path, summary_path, constructs_path):
        os.chmod(path, 0o600)

    source_file_hashes = {}
    for peptide, mapping in sorted(component_files_by_peptide.items()):
        source_file_hashes[peptide] = {
            name: {
                "score_path": paths["score"],
                "score_sha256": paths["score_sha256"],
                "producer_receipt_path": paths["receipt"],
                "producer_receipt_sha256": paths["receipt_sha256"],
            }
            for name, paths in sorted(mapping.items())
        }
    manifest = {
        "schema_version": "libb-ensemble-wetlab-shortlist-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 6),
        "ensemble_lock": {
            "path": str(args.lock_json.resolve()),
            "sha256": sha256_file(args.lock_json),
            "lock_id": lock["lock_id"],
            "display_name": lock["display_name"],
            "expected_lock_id": args.expected_lock_id,
            "bundle_manifest_path": str(args.lock_bundle_manifest.resolve()),
            "bundle_manifest_sha256": sha256_file(args.lock_bundle_manifest),
            "bundle_schema_version": lock_bundle["schema_version"],
            "weight_selection_provenance": lock["weight_selection_provenance"],
            "threshold_selection_provenance": lock["threshold_selection_provenance"],
            "selection": selection,
        },
        "candidate_universe": {
            "path": str(args.universe_dir.resolve()),
            "manifest_sha256": sha256_file(universe_manifest_path),
            "rows_scored": int(summary["universe_rows_scored"].sum()),
        },
        "component_score_files": source_file_hashes,
        "outputs": {
            "wetlab_shortlist_rows": int(len(shortlist)),
            "targets_with_above_cutoff_candidates": int(
                shortlist["peptide_design_code"].nunique()
            ),
            "targets_without_above_cutoff_candidates": sorted(
                set(EXPECTED_TARGET_SEQUENCES)
                - set(shortlist["peptide_design_code"].astype(str))
            ),
            "wetlab_shortlist_sha256": sha256_file(shortlist_path),
            "ranked_review_pool_rows": int(len(review)),
            "ranked_review_pool_sha256": sha256_file(review_path),
            "summary_sha256": sha256_file(summary_path),
            "unique_affibody_constructs": int(len(constructs)),
            "unique_affibody_constructs_sha256": sha256_file(constructs_path),
        },
        "outcome_access": {
            "retention_table_read": False,
            "retention_used_for_lock_or_selection": False,
            "retention_columns_allowed_in_component_scores": False,
        },
        "interpretation": (
            "The ensemble score orders prospective candidates; it is not a retention percentage. "
            "The wet-lab shortlist contains only candidates above the locked cutoff and is never "
            "padded with below-cutoff rows."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
