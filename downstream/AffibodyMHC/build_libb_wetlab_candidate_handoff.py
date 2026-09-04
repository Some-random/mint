#!/usr/bin/env python3
"""Turn locked LibB shortlist artifacts into wet-lab-facing order sheets.

The input shortlists have already been scored and thresholded. This program
does not fit a model, change a cutoff, read retention outcomes, or rank one
peptide target against another. The locked MINT Layer-5 shortlist is the sole
primary candidate rule. The all-four stacker is retained as a separately
labelled alternative and annotated 12-target menu; shortlist overlap is audit
information only and never a composite selection rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_COLUMNS = {
    "target_retention",
    "retention_percent",
    "target_binder",
    "binder_label_ge_75",
}

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
EXPECTED_LOCK_IDS = {
    "ensemble": "libb-all4-nonnegative-stacker-weak-oof-v1",
    "MINT": "libb-mint-layer5-primary-weak-oof-v1",
    "MINT+StaB": "libb-mint-stab-equal-logit-exploratory-weak-oof-v1",
}
DEFAULT_PROPOSED_TARGETS = ("AF", "AH", "DP", "EA", "LV", "MW", "NF", "PH", "TL", "VV")
AFFIBODY_CODE_CRYSTAL_POSITIONS = (8, 12, 15, 16, 19)
AFFIBODY_CODE_DISPLAYED_POSITIONS = (6, 10, 13, 14, 17)
LIBB_CODE_ALPHABET = frozenset("ADEFIKLMNSTVY")
AFFIBODY_CODE_POSITION_MAPPING = (
    "code_char_1=crystal_8/displayed_6;"
    "code_char_2=crystal_12/displayed_10;"
    "code_char_3=crystal_15/displayed_13;"
    "code_char_4=crystal_16/displayed_14;"
    "code_char_5=crystal_19/displayed_17"
)
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


def _forbidden_outcome_columns(columns) -> list[str]:
    leaked = set(FORBIDDEN_COLUMNS) & set(columns)
    for column in columns:
        normalized = str(column).strip().lower()
        if any(fragment in normalized for fragment in FORBIDDEN_OUTCOME_COLUMN_FRAGMENTS):
            leaked.add(str(column))
    return sorted(leaked)


def _validate_target_and_rank_integrity(frame: pd.DataFrame, label: str) -> None:
    _require(set(frame["peptide_design_code"].astype(str))
             == set(EXPECTED_TARGET_SEQUENCES),
             f"{label} is not the exact corrected 12-target LibB set")
    _require(not frame["pair_uid"].astype(str).duplicated().any(),
             f"{label} duplicate pair_uid")
    for peptide, block in frame.groupby("peptide_design_code", sort=True):
        _require(len(block) <= 10, f"{label} exceeds ten candidates for {peptide}")
        observed_sequence = set(block["peptide_full_sequence"].astype(str))
        _require(observed_sequence == {EXPECTED_TARGET_SEQUENCES[str(peptide)]},
                 f"{label} peptide sequence mismatch for {peptide}")
        ranks = pd.to_numeric(block["wetlab_rank"], errors="raise").astype(int)
        _require(sorted(ranks.tolist()) == list(range(1, len(block) + 1)),
                 f"{label} ranks are not exactly 1..N for {peptide}")


def _read_shortlist(directory: Path, label: str) -> tuple[pd.DataFrame, dict]:
    path = directory / "wetlab_shortlist.csv"
    manifest_path = directory / "manifest.json"
    _require(path.is_file(), f"{label} shortlist does not exist: {path}")
    _require(manifest_path.is_file(), f"{label} shortlist manifest does not exist")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema_version") == "libb-ensemble-wetlab-shortlist-v1",
             f"{label} shortlist manifest schema changed")
    lock = manifest.get("ensemble_lock", {})
    _require(lock.get("lock_id") == EXPECTED_LOCK_IDS[label],
             f"{label} shortlist was assembled with the wrong deployment lock")
    _require(lock.get("expected_lock_id") == EXPECTED_LOCK_IDS[label],
             f"{label} manifest expected-lock role mismatch")
    _require(isinstance(lock.get("bundle_manifest_sha256"), str)
             and len(lock["bundle_manifest_sha256"]) == 64,
             f"{label} lock-bundle provenance missing")
    access = manifest.get("outcome_access", {})
    _require(access.get("retention_table_read") is False
             and access.get("retention_used_for_lock_or_selection") is False
             and access.get("retention_columns_allowed_in_component_scores") is False,
             f"{label} manifest does not fail closed on retention outcomes")
    _require(manifest.get("outputs", {}).get("wetlab_shortlist_sha256")
             == sha256_file(path), f"{label} shortlist hash differs from its manifest")
    frame = pd.read_csv(path, dtype={"pair_uid": str})
    leaked = _forbidden_outcome_columns(frame.columns)
    _require(not leaked, f"{label} shortlist leaks retention outcomes: {sorted(leaked)}")
    required = {
        "pair_uid",
        "peptide_design_code",
        "peptide_full_sequence",
        "affibody_design_code",
        "chain2_affibody_sequence",
        "wetlab_rank",
        "ensemble_score",
    }
    _require(required.issubset(frame.columns), f"{label} shortlist schema changed")
    _validate_target_and_rank_integrity(frame, label)
    return frame, manifest


def _validate_cross_shortlist_provenance(manifests: dict[str, dict]) -> None:
    """Require all shortlist roles to share one weak-OOF lock and score artifacts."""
    bundle_hashes = {
        label: manifest.get("ensemble_lock", {}).get("bundle_manifest_sha256")
        for label, manifest in manifests.items()
    }
    _require(len(set(bundle_hashes.values())) == 1,
             "shortlists were not assembled from the same weak-OOF lock bundle")
    bundle_schemas = {
        label: manifest.get("ensemble_lock", {}).get("bundle_schema_version")
        for label, manifest in manifests.items()
    }
    _require(set(bundle_schemas.values()) == {"libb-weak-oof-deployment-locks-v2"},
             "shortlists do not identify the expected weak-OOF lock-bundle schema")

    expected_components = {
        "ensemble": {
            "additive_7site",
            "mint_layer5",
            "stab_designed_ordered",
            "rde_network_designed_3fold",
        },
        "MINT": {"mint_layer5"},
        "MINT+StaB": {"mint_layer5", "stab_designed_ordered"},
    }
    for label, manifest in manifests.items():
        sources = manifest.get("component_score_files")
        _require(isinstance(sources, dict)
                 and set(sources) == set(EXPECTED_TARGET_SEQUENCES),
                 f"{label} component-score provenance is not the exact 12-target set")
        for peptide, mapping in sources.items():
            _require(set(mapping) == expected_components[label],
                     f"{label} component-score roles changed for {peptide}")
            for component, record in mapping.items():
                _require(isinstance(record.get("score_sha256"), str)
                         and len(record["score_sha256"]) == 64,
                         f"{label} {component} score hash missing for {peptide}")
                _require(isinstance(record.get("producer_receipt_sha256"), str)
                         and len(record["producer_receipt_sha256"]) == 64,
                         f"{label} {component} producer-receipt hash missing for {peptide}")

    # The same component must mean the same immutable score partition in every
    # shortlist that uses it. Otherwise shortlist overlap is not interpretable.
    for peptide in EXPECTED_TARGET_SEQUENCES:
        for component, labels in (
            ("mint_layer5", ("ensemble", "MINT", "MINT+StaB")),
            ("stab_designed_ordered", ("ensemble", "MINT+StaB")),
        ):
            records = [
                manifests[label]["component_score_files"][peptide][component]
                for label in labels
            ]
            _require(len({record["score_sha256"] for record in records}) == 1,
                     f"shared {component} score hashes differ for {peptide}")
            _require(len({record["producer_receipt_sha256"] for record in records}) == 1,
                     f"shared {component} producer-receipt hashes differ for {peptide}")


def add_wetlab_sequence_fields(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add literal sequences and corrected LibB code numbering to a shortlist."""
    required = {
        "peptide_full_sequence",
        "chain1_smart_hla_linker_peptide_sequence",
        "chain2_affibody_sequence",
        "affibody_design_code",
    }
    _require(required.issubset(frame.columns), f"{label} lacks full sequence fields")
    output = frame.copy()
    validate_affibody_code_sequences(output, "chain2_affibody_sequence", label)
    _require(
        all(
            str(assay_chain).endswith(str(peptide))
            for assay_chain, peptide in zip(
                output["chain1_smart_hla_linker_peptide_sequence"],
                output["peptide_full_sequence"],
            )
        ),
        f"{label} full assay-chain sequence does not end in the stated peptide 9-mer",
    )
    output["peptide_9mer_sequence"] = output["peptide_full_sequence"].astype(str)
    output["smart_hla_linker_peptide_full_assay_chain_sequence"] = output[
        "chain1_smart_hla_linker_peptide_sequence"
    ].astype(str)
    output["affibody_full_sequence"] = output["chain2_affibody_sequence"].astype(str)
    output["provider_displayed_58aa_affibody_sequence"] = output[
        "chain2_affibody_sequence"
    ].astype(str)
    output["model_input_affibody_sequence"] = output[
        "chain2_affibody_sequence"
    ].astype(str)
    output["model_input_smart_hla_linker_peptide_sequence"] = output[
        "chain1_smart_hla_linker_peptide_sequence"
    ].astype(str)
    output["affibody_code_character_order_crystal_aligned_positions"] = ",".join(
        map(str, AFFIBODY_CODE_CRYSTAL_POSITIONS)
    )
    output["affibody_code_character_order_displayed_58aa_positions"] = ",".join(
        map(str, AFFIBODY_CODE_DISPLAYED_POSITIONS)
    )
    output["affibody_code_position_mapping"] = AFFIBODY_CODE_POSITION_MAPPING
    for index, crystal_position in enumerate(AFFIBODY_CODE_CRYSTAL_POSITIONS):
        output[f"affibody_crystal_position_{crystal_position}_amino_acid"] = (
            output["affibody_design_code"].astype(str).str[index]
        )
    return output


def _validate_structure_score_dir(score_dir: Path, family: str) -> dict:
    manifest_path = score_dir / "manifest.json"
    _require(manifest_path.is_file(), f"missing merged {family} score manifest")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    _require(manifest.get("schema_version")
             == "libb-fixed-structure-candidate-scores-merged-v1",
             f"unexpected merged {family} score schema")
    _require(manifest.get("family") == family, f"merged {family} score role mismatch")
    access = manifest.get("outcome_access", {})
    _require(access.get("selection_labels_read") is False
             and access.get("retention_measurements_read") is False,
             f"merged {family} scores do not explicitly declare outcome-free inference")
    records = manifest.get("outputs", {}).get("partitions", [])
    _require({str(row.get("peptide_design_code")) for row in records}
             == set(EXPECTED_TARGET_SEQUENCES),
             f"merged {family} scores do not contain the exact 12-target set")
    return manifest


def _load_structure_uncertainty(
    score_dir: Path,
    peptide: str,
    family: str,
    pair_ids: pd.Series,
    manifest: dict,
) -> pd.DataFrame:
    path = score_dir / f"peptide_{peptide}.parquet"
    _require(path.is_file(), f"missing merged {family} scores: {path}")
    record = {
        str(row["peptide_design_code"]): row
        for row in manifest["outputs"]["partitions"]
    }[peptide]
    _require(record.get("output_sha256") == sha256_file(path),
             f"merged {family} score hash mismatch for {peptide}")
    leaked_schema = _forbidden_outcome_columns(pq.ParquetFile(path).schema.names)
    _require(not leaked_schema,
             f"{family} score Parquet leaks retention outcomes: {leaked_schema}")
    probability = f"{family}_probability"
    uncertainty = f"{family}_seed_probability_sd"
    values = pd.read_parquet(path, columns=["pair_uid", probability, uncertainty])
    leaked = _forbidden_outcome_columns(values.columns)
    _require(not leaked, f"{family} scores leak retention outcomes")
    values["pair_uid"] = values["pair_uid"].astype(str)
    wanted = set(pair_ids.astype(str))
    values = values.loc[values["pair_uid"].isin(wanted)].copy()
    _require(len(values) == len(wanted), f"{family} uncertainty lacks selected rows for {peptide}")
    _require(not values["pair_uid"].duplicated().any(), f"duplicate {family} pair UID")
    return values


def hamming(left: str, right: str) -> int:
    _require(len(left) == len(right) == 5, "LibB code length changed")
    return sum(a != b for a, b in zip(left, right))


def validate_affibody_code_sequences(
    frame: pd.DataFrame, sequence_column: str, label: str
) -> None:
    displayed_python_indices = [position - 1 for position in AFFIBODY_CODE_DISPLAYED_POSITIONS]
    for row in frame[["affibody_design_code", sequence_column]].itertuples(index=False):
        code = str(row[0])
        sequence = str(row[1])
        _require(len(code) == 5, f"{label} contains a non-five-character Affibody code")
        _require(set(code).issubset(LIBB_CODE_ALPHABET),
                 f"{label} contains an Affibody code outside the provider LibB alphabet")
        _require(len(sequence) == 58, f"{label} contains a non-58-aa Affibody sequence")
        reconstructed = "".join(sequence[index] for index in displayed_python_indices)
        _require(reconstructed == code,
                 f"{label} Affibody code does not match its full sequence")


def within_target_min_hamming(frame: pd.DataFrame) -> pd.Series:
    values = {}
    for _, group in frame.groupby("peptide_design_code", sort=True):
        codes = group["affibody_design_code"].astype(str).tolist()
        for index, code in zip(group.index, codes):
            values[index] = min(
                (hamming(code, other) for other in codes if other != code),
                default=5,
            )
    return pd.Series(values).reindex(frame.index).astype(np.int64)


def summarize_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize counts only; model scores are not compared across targets."""
    summary = (
        frame.groupby("peptide_design_code", as_index=False)
        .agg(
            candidates=("pair_uid", "size"),
            unique_affibody_constructs=("affibody_design_code", "nunique"),
            previously_unobserved_candidates=(
                "observed_in_any_raw_round", lambda values: int((~values.astype(bool)).sum())
            ),
            candidates_observed_in_r009_or_r010=(
                "observed_in_r009_or_r010", lambda values: int(values.astype(bool).sum())
            ),
            candidates_with_affibody_identity_seen_in_strict_training=(
                "affibody_identity_seen_in_strict_training",
                lambda values: int(values.astype(bool).sum()),
            ),
            candidates_flagged_as_prior_high_confidence_weak_negative=(
                "high_confidence_weak_negative", "sum"
            ),
        )
        .sort_values("peptide_design_code", kind="mergesort")
        .reset_index(drop=True)
    )
    summary["score_use"] = "scores_are_used_only_to_rank_Affibodies_within_each_peptide"
    return summary


def parse_selected_peptides(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_PROPOSED_TARGETS
    values = tuple(value.strip().upper() for value in raw.split(",") if value.strip())
    _require(len(values) == 10,
             "--selected-peptides must contain exactly ten comma-separated target codes")
    _require(len(values) == len(set(values)), "--selected-peptides contains duplicates")
    _require(set(values).issubset(EXPECTED_TARGET_SEQUENCES),
             "--selected-peptides contains an unknown LibB target")
    return values


def unique_construct_roster(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate displayed/model-input Affibody sequences for sequence review."""
    return (
        frame.groupby(
            [
                "affibody_design_code",
                "provider_displayed_58aa_affibody_sequence",
                "model_input_affibody_sequence",
            ],
            as_index=False,
        )
        .agg(
            assayed_for_n_peptides=("peptide_design_code", "nunique"),
            assayed_for_peptides=(
                "peptide_design_code",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
            best_assay_slot=("assay_slot", "min"),
        )
        .sort_values(
            ["assayed_for_n_peptides", "affibody_design_code"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def write_wetlab_workbook(
    path: Path,
    order_this_batch: pd.DataFrame,
    result_entry: pd.DataFrame,
    all4_menu: pd.DataFrame,
    mint_all12_menu: pd.DataFrame,
    primary_mint: pd.DataFrame,
    primary_mint_with_comparator: pd.DataFrame,
    all4_alternative: pd.DataFrame,
    comparators: pd.DataFrame,
    primary_mint_constructs: pd.DataFrame,
    comparator_alternative_constructs: pd.DataFrame,
    all4_alternative_constructs: pd.DataFrame,
    selected_peptides: tuple[str, ...],
) -> None:
    pair_assays = len(order_this_batch)
    unique_constructs = len(primary_mint_constructs)
    instructions = pd.DataFrame(
        [
            ("Purpose", "LibB proposed ten-target pair-assay batch and supporting candidate menus"),
            ("ORDER_THIS_BATCH meaning", f"Exactly {pair_assays} specified peptide-Affibody pair assays using {unique_constructs} unique displayed/model-input Affibody sequences; this is NOT a Cartesian cross."),
            ("CRITICAL Affibody sequence warning", "provider_displayed_58aa_affibody_sequence/model_input_affibody_sequence is NOT vendor- or cloning-ready until the hidden N-terminal MA and vector/tag context are confirmed with the provider."),
            ("CRITICAL assay-side sequence warning", "model_input_smart_hla_linker_peptide_sequence is a model/assay-side input, NOT a cloning-ready construct; signal peptide, tags, linker boundaries, and vector context must be verified."),
            ("Primary sheet", "primary_mint_batch: the locked weak-OOF MINT Layer-5 shortlist"),
            ("Comparator alternative", "mint_comparator_alt replaces MINT rank 10 with one pooled-R009/R010 comparator per target"),
            ("Comparator sorting warning", "Use assay_slot for execution order. The comparator has blank model_rank (CMP semantics); do NOT sort the alternative sheet by model_rank."),
            ("Prospective yield", "After results return: number of valid primary assays with retention_at_30min_percent >= 75 divided by number of valid returned primary assays. Exclude comparator and technical-control rows."),
            ("All4 alternative", "all4_alt_batch: the locked all-four stacker shortlist, kept separate from the primary rule"),
            ("Primary all-target menu", "mint_all12_menu: locked primary MINT candidates for all 12 targets"),
            ("All4 all-target menu", "all4_12target_menu: annotated all-four stacker candidates for all 12 targets"),
            ("Default target codes", ",".join(selected_peptides)),
            ("Target choice", "Prespecified sequence-code coverage; no cross-peptide model-score or retention ranking"),
            ("Candidate cap", "At most 10 Affibodies per peptide"),
            ("Score use", "Compare candidates only within the same peptide"),
            ("Score meaning", "Selection-derived model score; not retention percent and not calibrated binding probability"),
            ("Affibody code mapping", AFFIBODY_CODE_POSITION_MAPPING),
            (
                "Retention-data use",
                "The corrected panel's pair identities were used upstream only to exclude already measured pairs. Retention values were not used for model training, score fitting, ranking, cutoffs, or target choice.",
            ),
        ],
        columns=["item", "instruction"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheets = {
            "ORDER_THIS_BATCH": order_this_batch,
            "README": instructions,
            "RESULT_ENTRY": result_entry,
            "primary_mint_batch": primary_mint,
            "mint_comparator_alt": primary_mint_with_comparator,
            "all4_alt_batch": all4_alternative,
            "mint_all12_menu": mint_all12_menu,
            "all4_12target_menu": all4_menu,
            "comparators": comparators,
            "mint_sequence_roster": primary_mint_constructs,
            "cmp_alt_sequence_roster": comparator_alternative_constructs,
            "all4_alt_sequence_roster": all4_alternative_constructs,
        }
        for peptide, block in mint_all12_menu.groupby("peptide_design_code", sort=True):
            sheets[f"mint_{peptide}"] = block
        for peptide, block in all4_menu.groupby("peptide_design_code", sort=True):
            sheets[f"all4_{peptide}"] = block
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max(map(len, values), default=0) + 2, 10), 50)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    os.chmod(path, 0o600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--mint-dir", type=Path, required=True)
    parser.add_argument("--mint-stab-dir", type=Path, required=True)
    parser.add_argument("--stab-scores-dir", type=Path, required=True)
    parser.add_argument("--rde-scores-dir", type=Path, required=True)
    parser.add_argument("--candidate-universe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-peptides",
        default=None,
        help=(
            "Exactly ten comma-separated target codes for the proposed batch. If omitted, "
            "a prespecified, non-model-based ten-target code-coverage set is used."
        ),
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    final_output = args.output_dir.resolve()
    private = (REPO_ROOT / "private_data").resolve()
    _require(final_output != private and private in final_output.parents,
             "output must be below private_data")
    _require(not final_output.exists(), "output exists; refusing overwrite")

    selected_peptides = parse_selected_peptides(args.selected_peptides)
    candidate_manifest_path = args.candidate_universe_dir.resolve() / "manifest.json"
    _require(candidate_manifest_path.is_file(), "candidate-universe manifest is missing")
    with candidate_manifest_path.open(encoding="utf-8") as handle:
        candidate_manifest = json.load(handle)
    _require(candidate_manifest.get("schema_version")
             == "libb-existing-target-selection-missed-candidates-v1",
             "candidate-universe schema changed")
    _require(candidate_manifest.get("scope", {}).get("target_peptide_codes")
             == list(EXPECTED_TARGET_SEQUENCES),
             "candidate universe is not the exact corrected 12-target set")
    _require(candidate_manifest.get("scope", {}).get("target_peptide_full_sequences")
             == EXPECTED_TARGET_SEQUENCES,
             "candidate-universe target sequence mapping changed")
    candidate_manifest_sha = sha256_file(candidate_manifest_path)

    ensemble, ensemble_manifest = _read_shortlist(args.ensemble_dir.resolve(), "ensemble")
    mint, mint_manifest = _read_shortlist(args.mint_dir.resolve(), "MINT")
    mint_stab, mint_stab_manifest = _read_shortlist(
        args.mint_stab_dir.resolve(), "MINT+StaB"
    )
    shortlist_manifests = {
        "ensemble": ensemble_manifest,
        "MINT": mint_manifest,
        "MINT+StaB": mint_stab_manifest,
    }
    _validate_cross_shortlist_provenance(shortlist_manifests)
    for label, manifest in shortlist_manifests.items():
        _require(manifest.get("candidate_universe", {}).get("manifest_sha256")
                 == candidate_manifest_sha,
                 f"{label} shortlist used a different candidate universe")

    stab_manifest = _validate_structure_score_dir(args.stab_scores_dir.resolve(), "stab")
    rde_manifest = _validate_structure_score_dir(args.rde_scores_dir.resolve(), "rde")
    for family, manifest in (("stab", stab_manifest), ("rde", rde_manifest)):
        _require(manifest.get("candidate_universe", {}).get("manifest_sha256")
                 == candidate_manifest_sha,
                 f"{family} scores used a different candidate universe")
        structure_records = {
            str(row["peptide_design_code"]): row
            for row in manifest["outputs"]["partitions"]
        }
        component_name = (
            "stab_designed_ordered" if family == "stab" else "rde_network_designed_3fold"
        )
        ensemble_sources = ensemble_manifest.get("component_score_files", {})
        merged_manifest_path = (
            args.stab_scores_dir.resolve() if family == "stab"
            else args.rde_scores_dir.resolve()
        ) / "manifest.json"
        merged_manifest_sha = sha256_file(merged_manifest_path)
        for peptide in EXPECTED_TARGET_SEQUENCES:
            source_record = ensemble_sources.get(peptide, {}).get(component_name, {})
            used_hash = source_record.get("score_sha256")
            _require(used_hash == structure_records[peptide].get("output_sha256"),
                     f"all-four ensemble did not use this exact {family} score file for {peptide}")
            _require(source_record.get("producer_receipt_sha256") == merged_manifest_sha,
                     f"all-four ensemble did not use this exact {family} merged manifest")

    mint_ids = set(mint["pair_uid"].astype(str))
    mint_stab_ids = set(mint_stab["pair_uid"].astype(str))
    frame = ensemble.copy()
    uncertainty_blocks = []
    for peptide, block in frame.groupby("peptide_design_code", sort=True):
        joined = block[["pair_uid"]].copy()
        for family, directory in (
            ("stab", args.stab_scores_dir.resolve()),
            ("rde", args.rde_scores_dir.resolve()),
        ):
            structure_manifest = stab_manifest if family == "stab" else rde_manifest
            values = _load_structure_uncertainty(
                directory, str(peptide), family, block["pair_uid"], structure_manifest
            )
            joined = joined.merge(values, on="pair_uid", how="left", validate="one_to_one")
        uncertainty_blocks.append(joined)
    uncertainty = pd.concat(uncertainty_blocks, ignore_index=True)
    frame = frame.merge(uncertainty, on="pair_uid", how="left", validate="one_to_one")
    _require(frame[["stab_seed_probability_sd", "rde_seed_probability_sd"]].notna().all().all(),
             "structure uncertainty join is incomplete")
    frame["selected_by_mint"] = frame["pair_uid"].astype(str).isin(mint_ids)
    frame["selected_by_mint_stab"] = frame["pair_uid"].astype(str).isin(mint_stab_ids)
    frame["candidate_list_membership_count"] = (
        1
        + frame["selected_by_mint"].astype(np.int64)
        + frame["selected_by_mint_stab"].astype(np.int64)
    )
    frame["minimum_hamming_to_another_selected_for_target"] = within_target_min_hamming(frame)
    frame["introduced_code_contains_cysteine"] = frame["affibody_design_code"].astype(str).str.contains("C")
    frame["full_affibody_contains_n_x_s_or_t"] = frame["chain2_affibody_sequence"].astype(str).map(
        lambda value: bool(re.search(r"N[^P][ST]", value))
    )
    for source, destination in (
        ("additive_7site", "additive_selection_score"),
        ("mint_layer5", "mint_selection_score"),
        ("stab_designed_ordered", "stab_selection_score"),
        ("rde_network_designed_3fold", "rde_selection_score"),
    ):
        value = np.clip(pd.to_numeric(frame[source], errors="raise").to_numpy(float), -700, 700)
        frame[destination] = 1.0 / (1.0 + np.exp(-value))
    for family in ("stab", "rde"):
        recomputed = frame[f"{family}_selection_score"].to_numpy(float)
        merged = frame[f"{family}_probability"].to_numpy(float)
        # The merged structure probability is stored as float32, whereas the
        # handoff recomputes sigmoid from the float32 logit in float64. Their
        # unavoidable round-trip difference is below 1e-7.
        _require(np.allclose(recomputed, merged, rtol=0.0, atol=1e-7),
                 f"{family} scores used by the all-four stacker differ from merged scores")

    frame["candidate_list_selection_pattern"] = np.select(
        [
            frame["high_confidence_weak_negative"].astype(bool),
            frame["candidate_list_membership_count"].eq(3)
            & frame["component_top10_votes"].ge(3),
            frame["candidate_list_membership_count"].ge(2),
        ],
        [
            "flagged_as_a_prior_high_confidence_weak_negative",
            "in_all_three_candidate_lists_and_top10_for_at_least_three_components",
            "in_at_least_two_of_the_three_candidate_lists",
        ],
        default="in_the_all_four_stacker_candidate_list_only",
    )
    frame = frame.sort_values(
        ["peptide_design_code", "wetlab_rank"], kind="mergesort"
    ).reset_index(drop=True)
    _validate_target_and_rank_integrity(frame, "final candidate menu")
    frame = add_wetlab_sequence_fields(frame, "all-four candidate menu")

    # Recompute repetition from this exact final menu rather than trusting a
    # convenience field copied from an upstream CSV.
    frame["selected_for_n_peptides"] = frame.groupby("affibody_design_code")[
        "peptide_design_code"
    ].transform("nunique").astype(np.int64)
    frame["selected_for_peptides"] = frame.groupby("affibody_design_code")[
        "peptide_design_code"
    ].transform(lambda values: ";".join(sorted(set(map(str, values)))))

    frame["component_top10_count"] = frame["component_top10_votes"].astype(np.int64)

    target_summary = summarize_targets(frame)

    concise_columns = [
        "peptide_design_code",
        "peptide_9mer_sequence",
        "wetlab_rank",
        "affibody_design_code",
        "affibody_code_character_order_crystal_aligned_positions",
        "affibody_code_character_order_displayed_58aa_positions",
        "affibody_code_position_mapping",
        "affibody_crystal_position_8_amino_acid",
        "affibody_crystal_position_12_amino_acid",
        "affibody_crystal_position_15_amino_acid",
        "affibody_crystal_position_16_amino_acid",
        "affibody_crystal_position_19_amino_acid",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "ensemble_score",
        "candidate_list_selection_pattern",
        "candidate_list_membership_count",
        "component_top10_count",
        "selected_by_mint",
        "selected_by_mint_stab",
        "within_peptide_additive_7site_rank",
        "within_peptide_mint_layer5_rank",
        "within_peptide_stab_designed_ordered_rank",
        "within_peptide_rde_network_designed_3fold_rank",
        "additive_selection_score",
        "mint_selection_score",
        "stab_selection_score",
        "rde_selection_score",
        "stab_seed_probability_sd",
        "rde_seed_probability_sd",
        "normalized_component_disagreement",
        "minimum_hamming_to_another_selected_for_target",
        "selected_for_n_peptides",
        "selected_for_peptides",
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
        "prior_weak_label",
        "pooled_r009_r010_count",
        "r009_count",
        "r010_count",
        "introduced_code_contains_cysteine",
        "full_affibody_contains_n_x_s_or_t",
        "pair_uid",
    ]
    missing = set(concise_columns) - set(frame.columns)
    _require(not missing, f"ensemble shortlist lacks handoff fields: {sorted(missing)}")
    concise = frame[concise_columns].copy()
    concise.insert(0, "candidate_menu_source_model", "locked_all4_stacker")

    # The primary prospective rule is exactly the locked MINT Layer-5
    # shortlist. No all4/MINT overlap, structural score, or hand-built voting
    # rule can add, remove, or reorder a primary candidate.
    primary_mint_menu = add_wetlab_sequence_fields(mint, "primary MINT shortlist")
    primary_required = {
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
        "prior_weak_label",
        "pooled_r009_r010_count",
        "r009_count",
        "r010_count",
        "pair_uid",
        "ensemble_score",
    }
    _require(primary_required.issubset(primary_mint_menu.columns),
             "primary MINT shortlist lacks wet-lab audit fields")
    primary_mint_menu["mint_layer5_locked_score"] = pd.to_numeric(
        primary_mint_menu["ensemble_score"], errors="raise"
    )
    primary_mint_menu["mint_layer5_within_peptide_rank"] = pd.to_numeric(
        primary_mint_menu["wetlab_rank"], errors="raise"
    ).astype(np.int64)
    primary_mint_menu["hamming_distance_to_measured_FALTA"] = (
        primary_mint_menu["affibody_design_code"].astype(str).map(
            lambda code: hamming(code, "FALTA")
        ).astype(np.int64)
    )
    primary_mint_menu["introduced_code_contains_cysteine"] = (
        primary_mint_menu["affibody_design_code"].astype(str).str.contains("C")
    )
    primary_mint_menu["full_affibody_contains_n_x_s_or_t"] = (
        primary_mint_menu["affibody_full_sequence"].astype(str).map(
            lambda sequence: bool(re.search(r"N[^P][ST]", sequence))
        )
    )
    primary_mint_menu["minimum_hamming_to_another_mint_candidate_for_target"] = (
        within_target_min_hamming(primary_mint_menu)
    )
    primary_mint_menu["selected_for_n_peptides"] = primary_mint_menu.groupby(
        "affibody_design_code"
    )["peptide_design_code"].transform("nunique").astype(np.int64)
    primary_mint_menu["selected_for_peptides"] = primary_mint_menu.groupby(
        "affibody_design_code"
    )["peptide_design_code"].transform(
        lambda values: ";".join(sorted(set(map(str, values))))
    )
    primary_mint_columns = [
        "peptide_design_code",
        "peptide_9mer_sequence",
        "wetlab_rank",
        "affibody_design_code",
        "affibody_code_character_order_crystal_aligned_positions",
        "affibody_code_character_order_displayed_58aa_positions",
        "affibody_code_position_mapping",
        "affibody_crystal_position_8_amino_acid",
        "affibody_crystal_position_12_amino_acid",
        "affibody_crystal_position_15_amino_acid",
        "affibody_crystal_position_16_amino_acid",
        "affibody_crystal_position_19_amino_acid",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "mint_layer5_locked_score",
        "mint_layer5_within_peptide_rank",
        "hamming_distance_to_measured_FALTA",
        "observed_in_any_raw_round",
        "observed_in_r009_or_r010",
        "affibody_identity_seen_in_strict_training",
        "high_confidence_weak_negative",
        "prior_weak_label",
        "pooled_r009_r010_count",
        "r009_count",
        "r010_count",
        "introduced_code_contains_cysteine",
        "full_affibody_contains_n_x_s_or_t",
        "minimum_hamming_to_another_mint_candidate_for_target",
        "selected_for_n_peptides",
        "selected_for_peptides",
        "pair_uid",
    ]
    primary_mint_menu = primary_mint_menu[primary_mint_columns].copy()
    primary_mint_menu.insert(0, "candidate_menu_source_model", "locked_mint_layer5")
    _require(not _forbidden_outcome_columns(primary_mint_menu.columns),
             "primary MINT menu contains an outcome column")
    primary_mint_summary = summarize_targets(primary_mint_menu)

    # Keep one optional sequencing-supported comparator per peptide separate
    # from the model-rescued candidates. These rows were in pooled R009/R010's
    # top 2%, were not directly measured, and therefore do not test whether the
    # model can recover a candidate missed by selection.
    excluded_path = args.candidate_universe_dir.resolve() / "excluded_pairs.parquet"
    _require(excluded_path.is_file(), "candidate-universe exclusion table is missing")
    expected_excluded_hash = candidate_manifest.get("outputs", {}).get(
        "excluded_pairs_sha256"
    )
    expected_excluded_rows = candidate_manifest.get("outputs", {}).get(
        "excluded_pairs_rows"
    )
    _require(isinstance(expected_excluded_hash, str) and len(expected_excluded_hash) == 64,
             "candidate manifest lacks the excluded-pairs hash")
    _require(isinstance(expected_excluded_rows, int) and expected_excluded_rows > 0,
             "candidate manifest lacks the excluded-pairs row count")
    _require(sha256_file(excluded_path) == expected_excluded_hash,
             "excluded-pairs Parquet hash differs from the candidate manifest")
    comparators = pd.read_parquet(excluded_path)
    _require(len(comparators) == int(expected_excluded_rows),
             "excluded-pairs row count differs from the candidate manifest")
    excluded_required = {
        "pair_uid",
        "peptide_design_code",
        "peptide_full_sequence",
        "affibody_design_code",
        "chain2_affibody_sequence",
        "r009_count",
        "r010_count",
        "pooled_r009_r010_count",
        "excluded_directly_measured",
        "excluded_pooled_r009_r010_top2pct",
    }
    _require(set(comparators.columns) == excluded_required,
             "excluded-pairs Parquet schema changed")
    _require(not _forbidden_outcome_columns(comparators.columns),
             "excluded-pairs Parquet contains retention outcomes")
    _require(comparators["pair_uid"].notna().all(),
             "excluded-pairs Parquet contains a missing pair ID")
    _require(not comparators["pair_uid"].astype(str).duplicated().any(),
             "excluded-pairs Parquet contains duplicate pair IDs")
    _require(set(comparators["peptide_design_code"].astype(str))
             == set(EXPECTED_TARGET_SEQUENCES),
             "excluded-pairs Parquet is not the exact 12-target set")
    for peptide, block in comparators.groupby("peptide_design_code", sort=True):
        _require(set(block["peptide_full_sequence"].astype(str))
                 == {EXPECTED_TARGET_SEQUENCES[str(peptide)]},
                 f"excluded-pairs peptide sequence mismatch for {peptide}")
    validate_affibody_code_sequences(
        comparators, "chain2_affibody_sequence", "excluded-pairs Parquet"
    )
    for flag in ("excluded_directly_measured", "excluded_pooled_r009_r010_top2pct"):
        _require(pd.api.types.is_bool_dtype(comparators[flag])
                 and comparators[flag].notna().all(),
                 f"excluded-pairs {flag} must be nonmissing Boolean values")
    _require(
        (comparators["excluded_directly_measured"]
         | comparators["excluded_pooled_r009_r010_top2pct"]).all(),
        "excluded-pairs table contains a row with no exclusion reason",
    )
    for count_column in ("r009_count", "r010_count", "pooled_r009_r010_count"):
        counts = pd.to_numeric(comparators[count_column], errors="raise")
        _require(counts.notna().all() and counts.ge(0).all()
                 and np.equal(counts, np.floor(counts)).all(),
                 f"excluded-pairs {count_column} must contain nonnegative integer counts")
    _require(
        comparators["pooled_r009_r010_count"].astype(np.uint64).eq(
            comparators["r009_count"].astype(np.uint64)
            + comparators["r010_count"].astype(np.uint64)
        ).all(),
        "excluded-pairs pooled count is not R009 + R010",
    )
    excluded_ids = set(comparators["pair_uid"].astype(str))
    for label, shortlist in (
        ("all4", ensemble), ("MINT", mint), ("MINT+StaB", mint_stab)
    ):
        _require(excluded_ids.isdisjoint(shortlist["pair_uid"].astype(str)),
                 f"{label} shortlist contains an excluded pair")
    comparators = comparators.loc[
        comparators["excluded_pooled_r009_r010_top2pct"].astype(bool)
        & ~comparators["excluded_directly_measured"].astype(bool)
    ].copy()
    expected_nonmeasured_comparators = candidate_manifest.get("exclusions", {}).get(
        "pooled_positive_nonmeasured_pairs"
    )
    _require(isinstance(expected_nonmeasured_comparators, int)
             and expected_nonmeasured_comparators > 0,
             "candidate manifest lacks the nonmeasured pooled-positive count")
    _require(len(comparators) == int(expected_nonmeasured_comparators),
             "nonmeasured pooled-positive comparator count changed")
    comparators = comparators.sort_values(
        ["peptide_design_code", "pooled_r009_r010_count", "pair_uid"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    comparators["selection_comparator_rank"] = comparators.groupby(
        "peptide_design_code"
    ).cumcount() + 1
    comparators = comparators.loc[comparators["selection_comparator_rank"].eq(1)].copy()
    comparators["comparator_status"] = (
        "review_only_not_in_the_proposed_batch;selection_supported_but_not_a_confirmed_binder"
    )
    _require(len(comparators) == 12, "expected one selection comparator per peptide")
    _require(set(comparators["peptide_design_code"].astype(str))
             == set(EXPECTED_TARGET_SEQUENCES),
             "selection comparator target set changed")
    validate_affibody_code_sequences(
        comparators, "chain2_affibody_sequence", "selection comparators"
    )
    assay_chain_by_peptide = primary_mint_menu.groupby("peptide_design_code")[
        "model_input_smart_hla_linker_peptide_sequence"
    ].first().to_dict()
    comparators["peptide_9mer_sequence"] = comparators["peptide_full_sequence"].astype(str)
    comparators["model_input_smart_hla_linker_peptide_sequence"] = comparators[
        "peptide_design_code"
    ].map(assay_chain_by_peptide)
    comparators["provider_displayed_58aa_affibody_sequence"] = comparators[
        "chain2_affibody_sequence"
    ].astype(str)
    comparators["model_input_affibody_sequence"] = comparators[
        "chain2_affibody_sequence"
    ].astype(str)
    comparators["affibody_code_character_order_crystal_aligned_positions"] = ",".join(
        map(str, AFFIBODY_CODE_CRYSTAL_POSITIONS)
    )
    comparators["affibody_code_character_order_displayed_58aa_positions"] = ",".join(
        map(str, AFFIBODY_CODE_DISPLAYED_POSITIONS)
    )
    comparators["affibody_code_position_mapping"] = AFFIBODY_CODE_POSITION_MAPPING
    for index, crystal_position in enumerate(AFFIBODY_CODE_CRYSTAL_POSITIONS):
        comparators[f"affibody_crystal_position_{crystal_position}_amino_acid"] = (
            comparators["affibody_design_code"].astype(str).str[index]
        )

    proposed = primary_mint_menu.loc[
        primary_mint_menu["peptide_design_code"].astype(str).isin(selected_peptides)
    ].copy()
    proposed["proposed_target_order"] = proposed["peptide_design_code"].map(
        {peptide: index + 1 for index, peptide in enumerate(selected_peptides)}
    ).astype(np.int64)
    proposed = proposed.sort_values(
        ["proposed_target_order", "wetlab_rank"], kind="mergesort"
    ).reset_index(drop=True)
    proposed["assay_slot"] = proposed["wetlab_rank"].astype(np.int64)
    proposed["model_rank"] = proposed["wetlab_rank"].astype(np.int64)
    proposed["candidate_source"] = "primary_locked_mint_layer5_shortlist"
    proposed_prefix = [
        "proposed_target_order",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "candidate_source",
        "model_rank",
        "affibody_design_code",
    ]
    proposed = proposed[
        proposed_prefix + [column for column in proposed.columns if column not in proposed_prefix]
    ]
    _require(proposed["peptide_design_code"].nunique() == 10,
             "proposed batch does not contain exactly ten targets")
    _require(proposed.groupby("peptide_design_code").size().eq(10).all(),
             "primary MINT batch must contain exactly ten Affibodies per target")
    identity_columns = [
        "pair_uid", "wetlab_rank", "affibody_design_code", "peptide_9mer_sequence",
        "provider_displayed_58aa_affibody_sequence",
    ]
    for peptide in selected_peptides:
        expected = primary_mint_menu.loc[
            primary_mint_menu["peptide_design_code"].astype(str).eq(peptide),
            identity_columns,
        ].sort_values("wetlab_rank", kind="mergesort").reset_index(drop=True)
        observed = proposed.loc[
            proposed["peptide_design_code"].astype(str).eq(peptide), identity_columns
        ].sort_values("wetlab_rank", kind="mergesort").reset_index(drop=True)
        _require(observed.equals(expected),
                 f"primary batch differs from the exact MINT shortlist for {peptide}")
    _require(not _forbidden_outcome_columns(proposed.columns),
             "primary MINT batch contains an outcome column")

    all4_alternative = concise.loc[
        concise["peptide_design_code"].astype(str).isin(selected_peptides)
    ].copy()
    all4_alternative["proposed_target_order"] = all4_alternative[
        "peptide_design_code"
    ].map({peptide: index + 1 for index, peptide in enumerate(selected_peptides)}).astype(
        np.int64
    )
    all4_alternative = all4_alternative.sort_values(
        ["proposed_target_order", "wetlab_rank"], kind="mergesort"
    ).reset_index(drop=True)
    all4_alternative["assay_slot"] = all4_alternative["wetlab_rank"].astype(np.int64)
    all4_alternative["model_rank"] = all4_alternative["wetlab_rank"].astype(np.int64)
    all4_alternative["candidate_source"] = "alternative_locked_all4_stacker_shortlist"
    all4_prefix = [
        "proposed_target_order",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "candidate_source",
        "model_rank",
        "affibody_design_code",
    ]
    all4_alternative = all4_alternative[
        all4_prefix
        + [column for column in all4_alternative.columns if column not in all4_prefix]
    ]
    _require(all4_alternative["peptide_design_code"].nunique() == 10,
             "all4 alternative does not contain exactly ten targets")
    _require(all4_alternative.groupby("peptide_design_code").size().eq(10).all(),
             "all4 alternative must contain exactly ten Affibodies per target")
    for peptide in selected_peptides:
        expected = concise.loc[
            concise["peptide_design_code"].astype(str).eq(peptide), identity_columns
        ].sort_values("wetlab_rank", kind="mergesort").reset_index(drop=True)
        observed = all4_alternative.loc[
            all4_alternative["peptide_design_code"].astype(str).eq(peptide), identity_columns
        ].sort_values("wetlab_rank", kind="mergesort").reset_index(drop=True)
        _require(observed.equals(expected),
                 f"all4 alternative differs from the exact all4 shortlist for {peptide}")

    # A separately labelled alternative swaps the model's tenth choice for
    # one pooled-R009/R010 selection-supported comparator. It never exceeds
    # ten constructs for a target and does not alter the primary model-only
    # batch.
    alternative_blocks = []
    alternative_columns = [
        "proposed_target_order",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "candidate_source",
        "model_rank",
        "rank_semantics",
        "affibody_design_code",
        "affibody_code_character_order_crystal_aligned_positions",
        "affibody_code_character_order_displayed_58aa_positions",
        "affibody_code_position_mapping",
        "affibody_crystal_position_8_amino_acid",
        "affibody_crystal_position_12_amino_acid",
        "affibody_crystal_position_15_amino_acid",
        "affibody_crystal_position_16_amino_acid",
        "affibody_crystal_position_19_amino_acid",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "mint_layer5_locked_score",
        "pooled_r009_r010_count",
        "r009_count",
        "r010_count",
        "comparator_status",
        "pair_uid",
    ]
    for target_order, peptide in enumerate(selected_peptides, start=1):
        model_block = proposed.loc[
            proposed["peptide_design_code"].astype(str).eq(peptide)
        ].copy()
        comparator = comparators.loc[
            comparators["peptide_design_code"].astype(str).eq(peptide)
        ].iloc[0]
        _require(len(model_block) == 10,
                 f"primary MINT comparator alternative requires ten candidates for {peptide}")
        model_block = model_block.loc[model_block["wetlab_rank"].astype(int).lt(10)].copy()
        comparator_slot = 10
        model_block["comparator_status"] = "not_applicable_model_candidate"
        model_block["rank_semantics"] = "numeric_model_rank"
        comparator_record = {
            "proposed_target_order": target_order,
            "peptide_design_code": peptide,
            "peptide_9mer_sequence": comparator["peptide_9mer_sequence"],
            "assay_slot": comparator_slot,
            "candidate_source": "selection_supported_comparator_CMP",
            "model_rank": pd.NA,
            "rank_semantics": "CMP_selection_comparator_blank_model_rank",
            "affibody_design_code": comparator["affibody_design_code"],
            "affibody_code_character_order_crystal_aligned_positions": comparator[
                "affibody_code_character_order_crystal_aligned_positions"
            ],
            "affibody_code_character_order_displayed_58aa_positions": comparator[
                "affibody_code_character_order_displayed_58aa_positions"
            ],
            "affibody_code_position_mapping": comparator["affibody_code_position_mapping"],
            "provider_displayed_58aa_affibody_sequence": comparator[
                "provider_displayed_58aa_affibody_sequence"
            ],
            "model_input_affibody_sequence": comparator["model_input_affibody_sequence"],
            "model_input_smart_hla_linker_peptide_sequence": comparator[
                "model_input_smart_hla_linker_peptide_sequence"
            ],
            "mint_layer5_locked_score": np.nan,
            "pooled_r009_r010_count": comparator["pooled_r009_r010_count"],
            "r009_count": comparator["r009_count"],
            "r010_count": comparator["r010_count"],
            "comparator_status": (
                "selection_supported_comparator_replaces_model_rank10_in_this_alternative"
            ),
            "pair_uid": comparator["pair_uid"],
        }
        for crystal_position in AFFIBODY_CODE_CRYSTAL_POSITIONS:
            comparator_record[f"affibody_crystal_position_{crystal_position}_amino_acid"] = (
                comparator[f"affibody_crystal_position_{crystal_position}_amino_acid"]
            )
        alternative_blocks.append(model_block.reindex(columns=alternative_columns))
        alternative_blocks.append(
            pd.DataFrame([comparator_record]).reindex(columns=alternative_columns)
        )
    proposed_with_comparator = pd.concat(alternative_blocks, ignore_index=True).sort_values(
        ["proposed_target_order", "assay_slot"], kind="mergesort"
    ).reset_index(drop=True)
    _require(proposed_with_comparator.groupby("peptide_design_code").size().max() <= 10,
             "comparator alternative exceeds ten candidates for a target")
    for peptide in selected_peptides:
        alternative_block = proposed_with_comparator.loc[
            proposed_with_comparator["peptide_design_code"].astype(str).eq(peptide)
        ].sort_values("assay_slot", kind="mergesort")
        primary_first_nine = proposed.loc[
            proposed["peptide_design_code"].astype(str).eq(peptide)
            & proposed["wetlab_rank"].astype(int).lt(10)
        ].sort_values("wetlab_rank", kind="mergesort")
        _require(
            alternative_block.iloc[:9]["pair_uid"].astype(str).tolist()
            == primary_first_nine["pair_uid"].astype(str).tolist(),
            f"comparator alternative changed MINT ranks 1--9 for {peptide}",
        )
        _require(
            alternative_block.iloc[9]["candidate_source"]
            == "selection_supported_comparator_CMP",
            f"comparator alternative lacks exactly one comparator at slot 10 for {peptide}",
        )
    proposed_constructs = unique_construct_roster(proposed)
    if args.selected_peptides is None:
        _require(len(proposed_constructs) == 16,
                 "default primary MINT batch no longer contains the expected 16 unique sequences")
    comparator_alternative_constructs = unique_construct_roster(
        proposed_with_comparator
    )
    all4_alternative_constructs = unique_construct_roster(all4_alternative)

    order_this_batch = proposed.copy()
    order_this_batch["sample_id"] = [
        f"LIBB-{peptide}-{int(slot):02d}"
        for peptide, slot in zip(
            order_this_batch["peptide_design_code"], order_this_batch["assay_slot"]
        )
    ]
    order_this_batch["batch_design"] = (
        f"{len(order_this_batch)} specified pair assays; "
        f"{len(proposed_constructs)} unique displayed/model-input Affibody sequences; "
        "NOT a Cartesian cross"
    )
    order_this_batch["sequence_readiness_warning"] = (
        "MODEL INPUT ONLY; NOT vendor/cloning-ready until hidden N-terminal MA, vector, "
        "tag, signal-peptide, and linker context are provider-confirmed"
    )
    order_columns = [
        "sample_id",
        "peptide_design_code",
        "peptide_9mer_sequence",
        "assay_slot",
        "model_rank",
        "affibody_design_code",
        "provider_displayed_58aa_affibody_sequence",
        "model_input_affibody_sequence",
        "model_input_smart_hla_linker_peptide_sequence",
        "pair_uid",
        "batch_design",
        "sequence_readiness_warning",
    ]
    order_this_batch = order_this_batch[order_columns].copy()
    _require(len(order_this_batch) == 100
             and order_this_batch["sample_id"].nunique() == 100,
             "ORDER_THIS_BATCH must contain exactly 100 stable, unique sample IDs")
    _require(not _forbidden_outcome_columns(order_this_batch.columns),
             "ORDER_THIS_BATCH contains an outcome column")

    result_entry = order_this_batch[
        ["sample_id", "pair_uid", "peptide_design_code", "peptide_9mer_sequence",
         "assay_slot", "model_rank", "affibody_design_code"]
    ].copy()
    result_fields = [
        "retention_at_30min_percent",
        "binder_ge_75",
        "technical_failure_status",
        "replicate_id",
        "assay_date",
        "experimental_batch_id",
        "notes",
    ]
    for column in result_fields:
        result_entry[column] = ""
    _require(
        result_entry[result_fields].fillna("").astype(str).eq("").all().all(),
        "result-entry outcome fields must be blank at release",
    )
    all4_full_audit = frame.drop(
        columns=[
            "affibody_full_sequence",
            "smart_hla_linker_peptide_full_assay_chain_sequence",
        ],
        errors="raise",
    )

    # Publish only after every artifact and manifest has been written into a
    # private sibling staging directory. Readers therefore see either no
    # handoff or a complete handoff, never a partially written workbook/CSV.
    final_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = Path(tempfile.mkdtemp(
        prefix=f".{final_output.name}.staging-", dir=final_output.parent
    ))
    os.chmod(output, 0o700)
    mint_by_target = output / "primary_mint_by_target_candidate_menu"
    mint_by_target.mkdir(mode=0o700)
    all4_by_target = output / "all4_by_target_candidate_menu"
    all4_by_target.mkdir(mode=0o700)
    paths = {
        "order_this_batch": output / "ORDER_THIS_BATCH.csv",
        "primary_result_entry_template": (
            output / "primary_mint_result_entry_template_100_pair_assays.csv"
        ),
        "primary_mint_all12_menu": (
            output / "primary_mint_candidate_menu_all_12_targets.csv"
        ),
        "all4_all12_annotated_menu": (
            output / "all4_annotated_candidate_menu_all_12_targets.csv"
        ),
        "primary_mint_wetlab_batch": (
            output / "primary_mint_wetlab_batch_10_targets.csv"
        ),
        "primary_mint_wetlab_batch_with_comparator": (
            output / "primary_mint_comparator_alternative_batch_10_targets.csv"
        ),
        "all4_stacker_alternative_batch": (
            output / "all4_stacker_alternative_batch_10_targets.csv"
        ),
        "primary_mint_wetlab_batch_constructs": (
            output / "primary_mint_unique_affibody_sequence_review_roster.csv"
        ),
        "primary_mint_comparator_alternative_constructs": (
            output / "primary_mint_comparator_alternative_unique_affibody_sequence_review_roster.csv"
        ),
        "all4_stacker_alternative_constructs": (
            output / "all4_stacker_alternative_unique_affibody_sequence_review_roster.csv"
        ),
        "primary_mint_all12_qc_summary": (
            output / "primary_mint_qc_summary_all_12_targets.csv"
        ),
        "all4_all12_qc_summary": (
            output / "all4_qc_summary_all_12_targets.csv"
        ),
        "selection_comparators": (
            output / "optional_selection_supported_comparator_one_per_target.csv"
        ),
        "all4_all12_full_audit": (
            output / "all4_candidate_selection_audit_all_12_targets.csv"
        ),
    }
    order_this_batch.to_csv(paths["order_this_batch"], index=False)
    result_entry.to_csv(paths["primary_result_entry_template"], index=False)
    primary_mint_menu.to_csv(paths["primary_mint_all12_menu"], index=False)
    concise.to_csv(paths["all4_all12_annotated_menu"], index=False)
    proposed.to_csv(paths["primary_mint_wetlab_batch"], index=False)
    proposed_with_comparator.to_csv(
        paths["primary_mint_wetlab_batch_with_comparator"], index=False
    )
    all4_alternative.to_csv(paths["all4_stacker_alternative_batch"], index=False)
    proposed_constructs.to_csv(paths["primary_mint_wetlab_batch_constructs"], index=False)
    comparator_alternative_constructs.to_csv(
        paths["primary_mint_comparator_alternative_constructs"], index=False
    )
    all4_alternative_constructs.to_csv(
        paths["all4_stacker_alternative_constructs"], index=False
    )
    primary_mint_summary.to_csv(paths["primary_mint_all12_qc_summary"], index=False)
    target_summary.to_csv(paths["all4_all12_qc_summary"], index=False)
    comparators.to_csv(paths["selection_comparators"], index=False)
    all4_full_audit.to_csv(paths["all4_all12_full_audit"], index=False)
    mint_target_paths = {}
    for peptide, block in primary_mint_menu.groupby("peptide_design_code", sort=True):
        target_path = mint_by_target / f"primary_mint_peptide_{peptide}.csv"
        block.to_csv(target_path, index=False)
        os.chmod(target_path, 0o600)
        mint_target_paths[str(peptide)] = target_path
    all4_target_paths = {}
    for peptide, block in concise.groupby("peptide_design_code", sort=True):
        target_path = all4_by_target / f"all4_peptide_{peptide}.csv"
        block.to_csv(target_path, index=False)
        os.chmod(target_path, 0o600)
        all4_target_paths[str(peptide)] = target_path
    for path in paths.values():
        os.chmod(path, 0o600)
    workbook_path = output / "libb_wetlab_candidate_handoff.xlsx"
    write_wetlab_workbook(
        workbook_path,
        order_this_batch,
        result_entry,
        concise,
        primary_mint_menu,
        proposed,
        proposed_with_comparator,
        all4_alternative,
        comparators,
        proposed_constructs,
        comparator_alternative_constructs,
        all4_alternative_constructs,
        selected_peptides,
    )

    manifest = {
        "schema_version": "libb-wetlab-candidate-handoff-v4",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 6),
        "selection_model": "weak-OOF-selected frozen MINT Layer-5 readout",
        "primary_candidate_rule": (
            "Use the locked MINT Layer-5 shortlist exactly; no composite voting or overlap "
            "rule adds, removes, or reorders candidates."
        ),
        "all4_role": (
            "The all-four stacker is a separately labelled alternative batch and an annotated "
            "12-target menu; it is not the primary selection rule."
        ),
        "primary_mint_candidate_rows_all_12_targets": int(len(primary_mint_menu)),
        "all4_candidate_rows_all_12_targets": int(len(concise)),
        "target_count": int(primary_mint_menu["peptide_design_code"].nunique()),
        "proposed_batch": {
            "target_codes_in_fixed_order": list(selected_peptides),
            "target_count": 10,
            "selection_lock_id": EXPECTED_LOCK_IDS["MINT"],
            "candidate_rows_primary_mint": int(len(proposed)),
            "candidate_rows_comparator_alternative": int(len(proposed_with_comparator)),
            "candidate_rows_all4_alternative": int(len(all4_alternative)),
            "pair_assays": int(len(order_this_batch)),
            "unique_affibody_sequences_primary_mint": int(len(proposed_constructs)),
            "unique_affibody_sequences_comparator_alternative": int(
                len(comparator_alternative_constructs)
            ),
            "unique_affibody_sequences_all4_alternative": int(
                len(all4_alternative_constructs)
            ),
            "batch_geometry": (
                "The CSV lists 100 specified peptide-Affibody pair assays. It is NOT a "
                "Cartesian cross between the ten peptides and the unique Affibody sequences."
            ),
            "target_choice_source": (
                "explicit --selected-peptides argument"
                if args.selected_peptides is not None
                else "prespecified neutral code-coverage default"
            ),
            "default_rationale": (
                "AF,AH,DP,EA,LV,MW,NF,PH,TL,VV covers all nine peptide-position-4 "
                "identities and all seven peptide-position-5 identities, retains the MW "
                "crystal-reference target and the AH target, and was not chosen "
                "using model scores or retention outcomes."
            ),
            "per_target_cap": 10,
            "mint_layer5_is_primary": True,
            "all4_stacker_is_separate_alternative": True,
            "comparator_alternative": (
                "A separate alternative replaces MINT rank 10 with one nonmeasured pooled-"
                "R009/R010 top-2% comparator for each target; it does not alter the primary "
                "MINT batch."
            ),
        },
        "target_selection_rule": (
            "No target was chosen or ordered using scores from different peptides. Scores are "
            "used only to rank Affibodies within the same peptide."
        ),
        "sequence_readiness": {
            "affibody": (
                "The provider-displayed 58-aa/model-input Affibody sequence is not a final "
                "vendor or cloning construct until hidden N-terminal MA, vector, and tag "
                "context are provider-confirmed."
            ),
            "assay_side": (
                "The SMART-HLA-linker-peptide sequence is a model/assay-side input, not a "
                "cloning-ready construct until signal peptide, tags, linker boundaries, and "
                "vector context are verified."
            ),
        },
        "prospective_result_definition": {
            "yield": (
                "valid primary assays with retention_at_30min_percent >= 75 divided by all "
                "valid returned primary assays"
            ),
            "excluded": "selection comparators and technical-control/technical-failure rows",
            "result_entry_template_released_blank": True,
        },
        "outcome_access": {
            "retention_outcome_values_read_by_handoff_builder": False,
            "retention_panel_pair_identities_used_upstream_for_exclusion": True,
            "retention_outcomes_used_for_model_training_scoring_ranking_cutoff_or_target_choice": False,
            "retention_used_for_target_choice": False,
            "retention_used_for_candidate_choice": False,
            "outcome_columns_allowed": False,
            "blank_result_entry_headers_are_not_observed_outcomes": True,
            "outcome_column_scope": (
                "Outcome columns are forbidden in model/score artifacts. RESULT_ENTRY contains "
                "only deliberately blank result-entry headers at release."
            ),
            "outcome_values_present_at_release": False,
        },
        "validation": {
            "exact_corrected_12_target_menu": True,
            "ranks_contiguous_within_target": True,
            "maximum_ten_candidates_per_target": True,
            "no_cross_peptide_score_sorting": True,
            "all4_stab_score_equals_merged_stab_score": True,
            "all4_rde_score_equals_merged_rde_score": True,
            "all_shortlists_share_one_weak_oof_lock_bundle": True,
            "shared_mint_and_stab_score_hashes_match_across_shortlists": True,
            "primary_batch_exactly_matches_mint_shortlist_for_selected_targets": True,
            "primary_mint_all12_menu_exactly_matches_mint_shortlist": True,
            "all4_alternative_exactly_matches_all4_shortlist_for_selected_targets": True,
            "excluded_pairs_hash_schema_and_peptide_sequences_validated": True,
            "published_by_atomic_directory_rename": True,
            "order_this_batch_has_100_unique_stable_sample_ids": True,
            "result_entry_outcome_fields_blank_at_release": True,
            "workbook_first_sheet_is_order_this_batch": True,
        },
        "affibody_code_position_mapping": {
            "code_character_order_crystal_aligned_positions": list(
                AFFIBODY_CODE_CRYSTAL_POSITIONS
            ),
            "code_character_order_displayed_58aa_positions": list(
                AFFIBODY_CODE_DISPLAYED_POSITIONS
            ),
            "literal_mapping": AFFIBODY_CODE_POSITION_MAPPING,
        },
        "inputs": {
            "ensemble_manifest_sha256": sha256_file(args.ensemble_dir / "manifest.json"),
            "mint_manifest_sha256": sha256_file(args.mint_dir / "manifest.json"),
            "mint_stab_manifest_sha256": sha256_file(args.mint_stab_dir / "manifest.json"),
            "stab_score_manifest_sha256": sha256_file(args.stab_scores_dir / "manifest.json"),
            "rde_score_manifest_sha256": sha256_file(args.rde_scores_dir / "manifest.json"),
            "candidate_universe_manifest_sha256": sha256_file(
                args.candidate_universe_dir / "manifest.json"
            ),
            "excluded_pairs_sha256": expected_excluded_hash,
            "shared_weak_oof_lock_bundle_sha256": ensemble_manifest[
                "ensemble_lock"
            ]["bundle_manifest_sha256"],
        },
        "outputs": {
            name: {"path": path.name, "rows": int(sum(1 for _ in path.open()) - 1), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "interpretation": (
            "Scores rank similarity to the selection-derived positive class. They are neither "
            "predicted retention percentages nor calibrated probabilities of wet-lab binding."
        ),
    }
    manifest["outputs"]["primary_mint_by_target"] = {
        peptide: {
            "path": str(path.relative_to(output)),
            "rows": int(sum(1 for _ in path.open()) - 1),
            "sha256": sha256_file(path),
        }
        for peptide, path in sorted(mint_target_paths.items())
    }
    manifest["outputs"]["all4_by_target"] = {
        peptide: {
            "path": str(path.relative_to(output)),
            "rows": int(sum(1 for _ in path.open()) - 1),
            "sha256": sha256_file(path),
        }
        for peptide, path in sorted(all4_target_paths.items())
    }
    manifest["outputs"]["workbook"] = {
        "path": workbook_path.name,
        "bytes": int(workbook_path.stat().st_size),
        "sha256": sha256_file(workbook_path),
        "sheets": [
            "ORDER_THIS_BATCH",
            "README",
            "RESULT_ENTRY",
            "primary_mint_batch",
            "mint_comparator_alt",
            "all4_alt_batch",
            "mint_all12_menu",
            "all4_12target_menu",
            "comparators",
            "mint_sequence_roster",
            "cmp_alt_sequence_roster",
            "all4_alt_sequence_roster",
            *[f"mint_{peptide}" for peptide in sorted(EXPECTED_TARGET_SEQUENCES)],
            *[f"all4_{peptide}" for peptide in sorted(EXPECTED_TARGET_SEQUENCES)],
        ],
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(manifest_path, 0o600)
    _require(not final_output.exists(),
             "final output appeared while the handoff was being staged")
    os.replace(output, final_output)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
