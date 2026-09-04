#!/usr/bin/env python
"""Summarize the corrected-120 native-projection LibB readouts.

This is a deterministic post-processing command.  It does not fit a model.
The primary ranking metrics use the established project convention: calculate
the metric separately within each peptide and then give every evaluable
peptide equal weight.  The score-cutoff exercise first averages the five fit
scores for each pair and then chooses a single cutoff that maximizes F1 on the
same known 120-pair panel.  Consequently, cutoff results are retrospective and
are not an independent estimate of prospective wet-lab performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn

from downstream.AffibodyMHC.summarize_libb_operational_metrics import (
    EXPECTED_AFFIBODIES,
    EXPECTED_BINDERS,
    EXPECTED_NONBINDERS,
    EXPECTED_PEPTIDES,
    EXPECTED_ROWS,
    FALTA_CODE,
    RETENTION_BINDER_THRESHOLD,
    ModelSpec,
    _json_compact,
    _sha256_file,
    _spearman,
    build_ranking_metrics,
    build_retrospective_threshold_metrics,
    calculate_ranking_metrics,
    load_selected_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-native-projection-public-summary-v1"
INPUT_DIR = (
    REPO_ROOT
    / "private_data/experiments/libb_structure_native_projection_120_evaluation_v1"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "private_data/experiments/libb_structure_native_projection_public_summary_v1"
)
RANKED_INPUT = INPUT_DIR / "ranked_per_pair.csv"
FALTA_NONOPTIMAL_PEPTIDES = ("EL", "MW")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def model_specs(source_path: Path) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            key="nonlinear_7site_native_projection",
            display_name="Nonlinear seven-position sequence control",
            source_path=source_path,
            source_model="nonlinear_7site_control",
            expected_fits=5,
        ),
        ModelSpec(
            key="rde_native_projection",
            display_name="RDE-PPI-derived direct learned projection",
            source_path=source_path,
            source_model="rde_network_designed_3fold_ensemble_native_projection",
            expected_fits=5,
        ),
        ModelSpec(
            key="stab_native_projection",
            display_name="StaB-ddG-derived direct learned projection",
            source_path=source_path,
            source_model="stab_designed_ordered",
            expected_fits=5,
        ),
    )


def build_averaged_score_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    identifiers = ["model_order", "model_key", "display_name", "source_model"]
    metadata = [
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    ]
    for keys, group in selected.groupby(identifiers, sort=False):
        model_order, model_key, display_name, source_model = keys
        panel = group[metadata].drop_duplicates()
        _require(len(panel) == EXPECTED_ROWS, f"{model_key} panel does not have 120 rows")
        mean_scores = (
            group.groupby("eval_row_id", as_index=False, sort=True)["score"]
            .mean()
        )
        averaged = panel.merge(mean_scores, on="eval_row_id", how="inner", validate="one_to_one")
        _require(len(averaged) == EXPECTED_ROWS, f"{model_key} averaged scores are incomplete")
        rows.append(
            {
                "model_order": int(model_order),
                "model_key": str(model_key),
                "display_name": str(display_name),
                "source_model": str(source_model),
                "n_fits_averaged": int(group["seed"].nunique()),
                **calculate_ranking_metrics(averaged),
            }
        )
    return pd.DataFrame(rows).sort_values("model_order", kind="stable").reset_index(drop=True)


def build_falta_panel_context(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for peptide, group in panel.groupby("peptide_design_code", sort=True):
        falta = group.loc[group["affibody_design_code"].eq(FALTA_CODE)]
        _require(len(falta) == 1, f"{peptide} does not have exactly one FALTA row")
        best = float(group["target_retention"].max())
        best_codes = sorted(
            group.loc[
                np.isclose(group["target_retention"], best, rtol=0.0, atol=1e-12),
                "affibody_design_code",
            ].astype(str)
        )
        record = falta.iloc[0]
        retention = float(record["target_retention"])
        rows.append(
            {
                "peptide_design_code": str(peptide),
                "n_binders": int(group["target_binder"].sum()),
                "falta_retention": retention,
                "falta_is_binder": int(record["target_binder"]),
                "experimental_best_retention": best,
                "experimental_best_affibodies": _json_compact(best_codes),
                "falta_is_experimentally_optimal": int(
                    math.isclose(retention, best, rel_tol=0.0, abs_tol=1e-12)
                ),
                "falta_retention_gap_to_best": best - retention,
            }
        )
    output = pd.DataFrame(rows).sort_values("peptide_design_code", kind="stable")
    _require(len(output) == EXPECTED_PEPTIDES, "FALTA panel context is incomplete")
    _require(int(output["falta_is_binder"].sum()) == 11, "FALTA binder count changed")
    _require(
        int(output["falta_is_experimentally_optimal"].sum()) == 10,
        "FALTA optimal-peptide count changed",
    )
    observed = tuple(
        output.loc[output["falta_is_experimentally_optimal"].eq(0), "peptide_design_code"]
        .astype(str)
        .sort_values()
    )
    _require(observed == FALTA_NONOPTIMAL_PEPTIDES, f"unexpected FALTA exceptions: {observed}")
    return output.reset_index(drop=True)


def _rank_one_model_score(group: pd.DataFrame) -> pd.DataFrame:
    ordered_frames: list[pd.DataFrame] = []
    for _, peptide_rows in group.groupby("peptide_design_code", sort=True):
        ordered = peptide_rows.sort_values(
            ["score", "affibody_design_code"],
            ascending=[False, True],
            kind="stable",
        ).copy()
        ordered["calculated_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
        ordered_frames.append(ordered)
    return pd.concat(ordered_frames, ignore_index=True)


def _spearman_excluding_falta(ranked: pd.DataFrame) -> tuple[float, int, str]:
    values: list[float] = []
    excluded: list[str] = []
    for peptide, group in ranked.groupby("peptide_design_code", sort=True):
        without = group.loc[group["affibody_design_code"].ne(FALTA_CODE)]
        _require(len(without) == EXPECTED_AFFIBODIES - 1, f"{peptide} FALTA exclusion changed")
        rho = _spearman(without["score"], without["target_retention"])
        if math.isfinite(rho):
            values.append(rho)
        else:
            excluded.append(str(peptide))
    _require(values, "no finite Spearman values after excluding FALTA")
    return float(np.mean(values)), len(values), _json_compact(excluded)


def _falta_diagnostic_row(
    group: pd.DataFrame,
    *,
    model_order: int,
    model_key: str,
    display_name: str,
    source_model: str,
    score_aggregation: str,
    seed: str | None,
) -> dict[str, Any]:
    ranked = _rank_one_model_score(group)
    rho, evaluable, excluded = _spearman_excluding_falta(ranked)
    falta = ranked.loc[ranked["affibody_design_code"].eq(FALTA_CODE)]
    _require(len(falta) == EXPECTED_PEPTIDES, "FALTA row count changed")
    row: dict[str, Any] = {
        "model_order": int(model_order),
        "model_key": str(model_key),
        "display_name": str(display_name),
        "source_model": str(source_model),
        "score_aggregation": score_aggregation,
        "seed": "" if seed is None else str(seed),
        "top1_falta_peptides_of_12": int(falta["calculated_rank"].eq(1).sum()),
        "within_peptide_spearman_excluding_falta": rho,
        "spearman_excluding_falta_evaluable_peptides": int(evaluable),
        "spearman_excluding_falta_excluded_peptides": excluded,
    }
    for peptide in FALTA_NONOPTIMAL_PEPTIDES:
        peptide_rows = ranked.loc[ranked["peptide_design_code"].eq(peptide)]
        _require(len(peptide_rows) == EXPECTED_AFFIBODIES, f"missing {peptide} row")
        peptide_falta = peptide_rows.loc[peptide_rows["affibody_design_code"].eq(FALTA_CODE)].iloc[0]
        top = peptide_rows.loc[peptide_rows["calculated_rank"].eq(1)].iloc[0]
        prefix = peptide.lower()
        row[f"{prefix}_falta_rank"] = int(peptide_falta["calculated_rank"])
        row[f"{prefix}_falta_score"] = float(peptide_falta["score"])
        row[f"{prefix}_falta_retention"] = float(peptide_falta["target_retention"])
        row[f"{prefix}_predicted_top1_affibody"] = str(top["affibody_design_code"])
        row[f"{prefix}_predicted_top1_retention"] = float(top["target_retention"])
    return row


def build_falta_diagnostics(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identifiers = ["model_order", "model_key", "display_name", "source_model"]
    by_seed_rows: list[dict[str, Any]] = []
    averaged_rows: list[dict[str, Any]] = []
    metadata = [
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    ]
    for keys, model_rows in selected.groupby(identifiers, sort=False):
        model_order, model_key, display_name, source_model = keys
        for seed, group in model_rows.groupby("seed", sort=True):
            by_seed_rows.append(
                _falta_diagnostic_row(
                    group,
                    model_order=int(model_order),
                    model_key=str(model_key),
                    display_name=str(display_name),
                    source_model=str(source_model),
                    score_aggregation="individual_fit",
                    seed=str(seed),
                )
            )
        panel = model_rows[metadata].drop_duplicates()
        means = model_rows.groupby("eval_row_id", as_index=False, sort=True)["score"].mean()
        averaged = panel.merge(means, on="eval_row_id", how="inner", validate="one_to_one")
        averaged_rows.append(
            _falta_diagnostic_row(
                averaged,
                model_order=int(model_order),
                model_key=str(model_key),
                display_name=str(display_name),
                source_model=str(source_model),
                score_aggregation="arithmetic_mean_across_five_fits",
                seed=None,
            )
        )

    by_seed = pd.DataFrame(by_seed_rows).sort_values(
        ["model_order", "seed"], kind="stable"
    ).reset_index(drop=True)
    averaged = pd.DataFrame(averaged_rows).sort_values("model_order", kind="stable").reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    numeric = [
        "top1_falta_peptides_of_12",
        "within_peptide_spearman_excluding_falta",
        "el_falta_rank",
        "mw_falta_rank",
    ]
    for keys, group in by_seed.groupby(identifiers, sort=False):
        model_order, model_key, display_name, source_model = keys
        row: dict[str, Any] = {
            "model_order": int(model_order),
            "model_key": str(model_key),
            "display_name": str(display_name),
            "source_model": str(source_model),
            "n_fits": int(len(group)),
            "seeds": _json_compact(group["seed"].astype(str).tolist()),
        }
        for metric in numeric:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            row[f"{metric}_individual"] = _json_compact(
                [
                    {"seed": str(seed), "value": float(value)}
                    for seed, value in zip(group["seed"], values, strict=True)
                ]
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("model_order", kind="stable").reset_index(drop=True)
    return by_seed, summary, averaged


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g", na_rep="")


def _output_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _optional_float(value: Any) -> float | None:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def build_summary_payload(
    specs: Sequence[ModelSpec],
    ranking_summary: pd.DataFrame,
    averaged_metrics: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    falta_context: pd.DataFrame,
    falta_summary: pd.DataFrame,
    falta_averaged: pd.DataFrame,
) -> dict[str, Any]:
    rank_index = ranking_summary.set_index("model_key", verify_integrity=True)
    average_index = averaged_metrics.set_index("model_key", verify_integrity=True)
    cutoff_index = cutoff_summary.set_index("model_key", verify_integrity=True)
    falta_index = falta_summary.set_index("model_key", verify_integrity=True)
    falta_average_index = falta_averaged.set_index("model_key", verify_integrity=True)
    model_rows: list[dict[str, Any]] = []
    for spec in specs:
        rank = rank_index.loc[spec.key]
        average = average_index.loc[spec.key]
        cutoff = cutoff_index.loc[spec.key]
        falta = falta_index.loc[spec.key]
        falta_average = falta_average_index.loc[spec.key]
        metric_payload: dict[str, Any] = {}
        for metric in (
            "within_peptide_spearman",
            "within_peptide_auroc",
            "within_peptide_average_precision",
        ):
            metric_payload[metric] = {
                "mean": float(rank[f"{metric}_mean"]),
                "sample_sd": _optional_float(rank[f"{metric}_sd"]),
                "individual_fits": json.loads(rank[f"{metric}_individual"]),
            }
        model_rows.append(
            {
                "model_key": spec.key,
                "display_name": spec.display_name,
                "source_model": spec.source_model,
                "n_fits": spec.expected_fits,
                "ranking_across_fits": metric_payload,
                "ranking_from_mean_score": {
                    "within_peptide_spearman": float(average["within_peptide_spearman"]),
                    "within_peptide_auroc": float(average["within_peptide_auroc"]),
                    "within_peptide_average_precision": float(
                        average["within_peptide_average_precision"]
                    ),
                },
                "retrospective_cutoff": {
                    "threshold": float(cutoff["threshold"]),
                    "recommended": int(cutoff["selected_candidates"]),
                    "true_positives": int(cutoff["true_positives"]),
                    "false_positives": int(cutoff["false_positives"]),
                    "false_negatives": int(cutoff["false_negatives"]),
                    "precision": float(cutoff["precision"]),
                    "recall": float(cutoff["recall"]),
                    "f1": float(cutoff["f1"]),
                    "zero_recommendation_peptides": json.loads(
                        cutoff["zero_candidate_peptides"]
                    ),
                },
                "falta_sensitivity_across_fits": {
                    "top1_peptides_of_12_mean": float(
                        falta["top1_falta_peptides_of_12_mean"]
                    ),
                    "top1_peptides_of_12_sample_sd": float(
                        falta["top1_falta_peptides_of_12_sd"]
                    ),
                    "top1_peptides_of_12_individual_fits": json.loads(
                        falta["top1_falta_peptides_of_12_individual"]
                    ),
                    "within_peptide_spearman_excluding_falta_mean": float(
                        falta["within_peptide_spearman_excluding_falta_mean"]
                    ),
                    "within_peptide_spearman_excluding_falta_sample_sd": float(
                        falta["within_peptide_spearman_excluding_falta_sd"]
                    ),
                    "within_peptide_spearman_excluding_falta_individual_fits": json.loads(
                        falta["within_peptide_spearman_excluding_falta_individual"]
                    ),
                    "el_falta_rank_individual_fits": json.loads(
                        falta["el_falta_rank_individual"]
                    ),
                    "mw_falta_rank_individual_fits": json.loads(
                        falta["mw_falta_rank_individual"]
                    ),
                },
                "falta_diagnostic_from_mean_score": {
                    "top1_peptides_of_12": int(
                        falta_average["top1_falta_peptides_of_12"]
                    ),
                    "within_peptide_spearman_excluding_falta": float(
                        falta_average["within_peptide_spearman_excluding_falta"]
                    ),
                    "el_falta_rank": int(falta_average["el_falta_rank"]),
                    "el_predicted_top1_affibody": str(
                        falta_average["el_predicted_top1_affibody"]
                    ),
                    "mw_falta_rank": int(falta_average["mw_falta_rank"]),
                    "mw_predicted_top1_affibody": str(
                        falta_average["mw_predicted_top1_affibody"]
                    ),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "panel": {
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies_per_peptide": EXPECTED_AFFIBODIES,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "binder_definition": f"target_retention >= {RETENTION_BINDER_THRESHOLD:g}",
        },
        "falta_panel_facts": {
            "binder_peptides": int(falta_context["falta_is_binder"].sum()),
            "experimentally_optimal_peptides": int(
                falta_context["falta_is_experimentally_optimal"].sum()
            ),
            "nonoptimal_peptides": list(FALTA_NONOPTIMAL_PEPTIDES),
            "per_peptide": falta_context.to_dict(orient="records"),
        },
        "models": model_rows,
    }


def write_outputs(
    output_dir: Path,
    specs: Sequence[ModelSpec],
    panel: pd.DataFrame,
    ranking_by_seed: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    averaged_metrics: pd.DataFrame,
    cutoff_summary: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    decisions: pd.DataFrame,
    falta_context: pd.DataFrame,
    falta_by_seed: pd.DataFrame,
    falta_summary: pd.DataFrame,
    falta_averaged: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = {
        "ranking_metrics_by_seed.csv": ranking_by_seed.drop(columns=["model_order"]),
        "ranking_metrics_seed_summary.csv": ranking_summary.drop(columns=["model_order"]),
        "ranking_metrics_averaged_scores.csv": averaged_metrics.drop(columns=["model_order"]),
        "retrospective_cutoff_summary.csv": cutoff_summary.drop(columns=["model_order"]),
        "retrospective_candidate_counts_by_peptide.csv": candidate_counts.drop(columns=["model_order"]),
        "retrospective_mean_score_decisions.csv": decisions.drop(columns=["model_order"]),
        "falta_panel_context.csv": falta_context,
        "falta_sensitivity_by_seed.csv": falta_by_seed.drop(columns=["model_order"]),
        "falta_sensitivity_seed_summary.csv": falta_summary.drop(columns=["model_order"]),
        "falta_sensitivity_averaged_scores.csv": falta_averaged.drop(columns=["model_order"]),
    }
    output_records: dict[str, Any] = {}
    for filename, frame in frames.items():
        path = output_dir / filename
        _write_csv(frame, path)
        output_records[filename] = _output_record(path, len(frame))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            build_summary_payload(
                specs,
                ranking_summary,
                averaged_metrics,
                cutoff_summary,
                falta_context,
                falta_summary,
                falta_averaged,
            ),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_records[summary_path.name] = _output_record(summary_path)

    input_manifest = INPUT_DIR / "manifest.json"
    membership = "\n".join(sorted(panel["eval_row_id"].astype(str))) + "\n"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "deterministic_summary_of_completed_native_projection_predictions",
        "panel": {
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies_per_peptide": EXPECTED_AFFIBODIES,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "binder_definition": f"target_retention >= {RETENTION_BINDER_THRESHOLD:g}",
            "membership_sha256": hashlib.sha256(membership.encode("utf-8")).hexdigest(),
        },
        "ranking_metrics": {
            "within_peptide_spearman": "unweighted mean across 12 peptide rows",
            "within_peptide_auroc": "unweighted mean across 11 two-class peptide rows; DP excluded",
            "within_peptide_average_precision": "unweighted mean across 11 two-class peptide rows; DP excluded",
            "seed_summary": "metric calculated separately per fit, then arithmetic mean and sample SD (ddof=1)",
            "averaged_score_summary": "arithmetic mean score across five fits first, then metric calculation",
        },
        "retrospective_cutoff": {
            "status": "fit_on_same_corrected_120_labels_not_independent_evaluation",
            "score_aggregation": "arithmetic mean across five fits per pair",
            "decision_rule": "mean_score >= threshold",
            "objective": "maximize micro F1 across the 120 known binder labels",
            "tie_break_order": ["higher precision", "fewer recommendations", "higher threshold"],
        },
        "falta_sensitivity": {
            "primary_panel_unchanged": True,
            "description": "post-hoc calculation removes FALTA only while recomputing within-peptide Spearman",
            "binder_peptides": int(falta_context["falta_is_binder"].sum()),
            "experimentally_optimal_peptides": int(falta_context["falta_is_experimentally_optimal"].sum()),
            "nonoptimal_peptides": list(FALTA_NONOPTIMAL_PEPTIDES),
        },
        "models": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "source_model": spec.source_model,
                "fits": spec.expected_fits,
            }
            for spec in specs
        ],
        "sources": {
            "ranked_predictions": _output_record(RANKED_INPUT, 1800),
            "evaluation_manifest": _output_record(input_manifest),
        },
        "producer": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": output_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RANKED_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.resolve()
    specs = model_specs(input_path)
    selected, panel = load_selected_predictions(specs)
    ranking_by_seed, ranking_summary = build_ranking_metrics(selected)
    averaged_metrics = build_averaged_score_metrics(selected)
    cutoff_summary, candidate_counts, decisions = build_retrospective_threshold_metrics(selected, panel)
    falta_context = build_falta_panel_context(panel)
    falta_by_seed, falta_summary, falta_averaged = build_falta_diagnostics(selected)
    manifest = write_outputs(
        args.output_dir.resolve(),
        specs,
        panel,
        ranking_by_seed,
        ranking_summary,
        averaged_metrics,
        cutoff_summary,
        candidate_counts,
        decisions,
        falta_context,
        falta_by_seed,
        falta_summary,
        falta_averaged,
    )
    print(f"Wrote native-projection LibB public summary: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
