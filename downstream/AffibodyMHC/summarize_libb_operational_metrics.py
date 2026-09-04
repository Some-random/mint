#!/usr/bin/env python
"""Summarize LibB rankings and a retrospective score-threshold exercise.

This command compares the eight fixed, already-evaluated LibB models on the
corrected 120-pair retention panel.  It does not train any model.  It reports
peptide-conditioned ranking metrics and then performs a deliberately
retrospective exercise: for each model, predictions are averaged across its
available fits and a score threshold is chosen to maximize micro F1 on these
same 120 known binder labels.

The resulting threshold metrics are *not* an independent evaluation and must
not be presented as expected prospective wet-lab performance.  They describe
how well a threshold can be fit to the currently known panel and provide a
candidate operating rule that must be locked before new wet-lab measurements
can validate it.

Example::

    python downstream/AffibodyMHC/summarize_libb_operational_metrics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "libb-operational-metrics-v1"
CALIBRATION_STATUS = "retrospective_same_120_labels_not_independent_evaluation"

EXPECTED_ROWS = 120
EXPECTED_PEPTIDES = 12
EXPECTED_AFFIBODIES = 10
EXPECTED_BINDERS = 61
EXPECTED_NONBINDERS = 59
EXPECTED_BINARY_EVALUABLE_PEPTIDES = 11
EXPECTED_BINARY_EXCLUDED_PEPTIDES = ("DP",)
RETENTION_BINDER_THRESHOLD = 75.0
FALTA_CODE = "FALTA"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    source_path: Path
    source_model: str
    expected_fits: int


def default_model_specs() -> tuple[ModelSpec, ...]:
    """Return the fixed eight-model comparison in its public display order."""

    return (
        ModelSpec(
            key="additive_7site",
            display_name="Additive seven-position baseline",
            source_path=REPO_ROOT
            / "private_data/experiments/pnu_libb_retention_120_revision_v1/metrics/ranked_per_pair.csv",
            source_model="site_primary_pn_control",
            expected_fits=1,
        ),
        ModelSpec(
            key="nonlinear_7site",
            display_name="Nonlinear seven-position neural network",
            source_path=REPO_ROOT
            / "private_data/experiments/libb_structure_provider_revision_120_sealed_evaluation_v1/ranked_per_pair.csv",
            source_model="nonlinear_7site_control",
            expected_fits=5,
        ),
        ModelSpec(
            key="frozen_mint_layer5",
            display_name="Frozen MINT layer 5",
            source_path=REPO_ROOT
            / "private_data/experiments/libb_revision_mint_metrics_120_v1/ranked_per_pair.csv",
            source_model="frozen_mint_layer5_archived119_plus_replay_target",
            expected_fits=1,
        ),
        ModelSpec(
            key="lora_mint",
            display_name="LoRA MINT (one epoch)",
            source_path=REPO_ROOT
            / "private_data/experiments/libb_revision_mint_metrics_120_v1/ranked_per_pair.csv",
            source_model="lora_one_epoch_saved_checkpoint_exact120",
            expected_fits=1,
        ),
        ModelSpec(
            key="esmfold2_prefolding",
            display_name="ESMFold2 pre-folding residue features (sequence-like)",
            source_path=REPO_ROOT
            / "private_data/experiments/esmfold2_libb_provider_revision_120_metrics_v1/ranked_per_pair.csv",
            source_model="single_inputs_only",
            expected_fits=5,
        ),
        ModelSpec(
            key="esmfold2_pair_state",
            display_name="ESMFold2 folding-derived pair state",
            source_path=REPO_ROOT
            / "private_data/experiments/esmfold2_libb_provider_revision_120_metrics_v1/ranked_per_pair.csv",
            source_model="pair_state_only",
            expected_fits=5,
        ),
        ModelSpec(
            key="rde_locked",
            display_name="RDE-PPI-derived locked representative",
            source_path=REPO_ROOT
            / "private_data/experiments/libb_structure_provider_revision_120_sealed_evaluation_v1/ranked_per_pair.csv",
            source_model="rde_network_designed_3fold_ensemble",
            expected_fits=5,
        ),
        ModelSpec(
            key="stab_locked",
            display_name="StaB-ddG-derived locked representative",
            source_path=REPO_ROOT
            / "private_data/experiments/libb_structure_provider_revision_120_sealed_evaluation_v1/ranked_per_pair.csv",
            source_model="stab_designed_ordered",
            expected_fits=5,
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
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


def _mtime_utc(path: Path) -> str:
    timestamp = Path(path).stat().st_mtime
    return pd.Timestamp(timestamp, unit="s", tz="UTC").isoformat()


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    text = str(seed)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation using average ranks, returning NaN if undefined."""

    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if np.ptp(x_rank) == 0.0 or np.ptp(y_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _validate_panel(panel: pd.DataFrame) -> None:
    _require(len(panel) == EXPECTED_ROWS, f"panel has {len(panel)} rows, expected {EXPECTED_ROWS}")
    _require(panel["eval_row_id"].nunique() == EXPECTED_ROWS, "panel eval_row_id values are not unique")
    _require(
        panel["peptide_design_code"].nunique() == EXPECTED_PEPTIDES,
        f"panel does not have {EXPECTED_PEPTIDES} peptides",
    )
    _require(
        panel["affibody_design_code"].nunique() == EXPECTED_AFFIBODIES,
        f"panel does not have {EXPECTED_AFFIBODIES} Affibodies",
    )
    sizes = panel.groupby("peptide_design_code", sort=True).size()
    _require(bool(sizes.eq(EXPECTED_AFFIBODIES).all()), "panel is not a complete 12 x 10 matrix")
    expected_label = panel["target_retention"].ge(RETENTION_BINDER_THRESHOLD).astype(int)
    _require(
        bool(expected_label.eq(panel["target_binder"]).all()),
        "target_binder is inconsistent with retention >= 75",
    )
    binder_count = int(panel["target_binder"].sum())
    _require(binder_count == EXPECTED_BINDERS, f"panel has {binder_count} binders, expected {EXPECTED_BINDERS}")
    _require(len(panel) - binder_count == EXPECTED_NONBINDERS, "panel non-binder count changed")

    both_class = []
    excluded = []
    for peptide, group in panel.groupby("peptide_design_code", sort=True):
        if group["target_binder"].nunique() == 2:
            both_class.append(str(peptide))
        else:
            excluded.append(str(peptide))
    _require(
        len(both_class) == EXPECTED_BINARY_EVALUABLE_PEPTIDES,
        "number of peptide rows with both binder classes changed",
    )
    _require(
        tuple(excluded) == EXPECTED_BINARY_EXCLUDED_PEPTIDES,
        f"binary-metric excluded peptides changed: {excluded}",
    )


def load_selected_predictions(specs: Sequence[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fixed model rows and verify a common corrected-120 panel."""

    required = {
        "eval_row_id",
        "model",
        "seed",
        "score",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    }
    path_cache: dict[Path, pd.DataFrame] = {}
    selected_frames: list[pd.DataFrame] = []
    canonical_panel: pd.DataFrame | None = None

    for order, spec in enumerate(specs):
        path = Path(spec.source_path).resolve()
        _require(path.is_file(), f"missing ranked prediction source: {path}")
        if path not in path_cache:
            source = pd.read_csv(
                path,
                dtype={
                    "eval_row_id": str,
                    "model": str,
                    "seed": str,
                    "peptide_design_code": str,
                    "affibody_design_code": str,
                },
                keep_default_na=False,
            )
            _require(required.issubset(source.columns), f"ranked prediction schema changed: {path}")
            source["score"] = pd.to_numeric(source["score"], errors="raise").astype(float)
            source["target_retention"] = pd.to_numeric(
                source["target_retention"], errors="raise"
            ).astype(float)
            source["target_binder"] = pd.to_numeric(
                source["target_binder"], errors="raise"
            ).astype(int)
            _require(bool(np.isfinite(source["score"]).all()), f"non-finite scores in {path}")
            _require(
                bool(np.isfinite(source["target_retention"]).all()),
                f"non-finite retention values in {path}",
            )
            path_cache[path] = source

        frame = path_cache[path]
        chosen = frame.loc[frame["model"].eq(spec.source_model), list(required)].copy()
        _require(not chosen.empty, f"model {spec.source_model!r} is absent from {path}")
        duplicate = chosen.duplicated(["seed", "eval_row_id"], keep=False)
        _require(not bool(duplicate.any()), f"duplicate rows for {spec.key}")
        fit_sizes = chosen.groupby("seed", sort=True).size()
        _require(
            len(fit_sizes) == spec.expected_fits,
            f"{spec.key} has {len(fit_sizes)} fits, expected {spec.expected_fits}",
        )
        _require(
            bool(fit_sizes.eq(EXPECTED_ROWS).all()),
            f"not every {spec.key} fit contains {EXPECTED_ROWS} rows",
        )

        metadata_columns = [
            "eval_row_id",
            "peptide_design_code",
            "affibody_design_code",
            "target_retention",
            "target_binder",
        ]
        panel = (
            chosen[metadata_columns]
            .drop_duplicates()
            .sort_values("eval_row_id", kind="stable")
            .reset_index(drop=True)
        )
        _validate_panel(panel)
        if canonical_panel is None:
            canonical_panel = panel
        else:
            pd.testing.assert_frame_equal(canonical_panel, panel, check_exact=True)

        chosen["model_key"] = spec.key
        chosen["display_name"] = spec.display_name
        chosen["source_model"] = spec.source_model
        chosen["model_order"] = order
        selected_frames.append(chosen)

    _require(canonical_panel is not None, "no models were configured")
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values(
        ["model_order", "seed", "eval_row_id"], kind="stable"
    ).reset_index(drop=True)
    return selected, canonical_panel


def calculate_ranking_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """Calculate peptide-conditioned continuous and binary ranking metrics."""

    spearman_values: list[float] = []
    auroc_values: list[float] = []
    ap_values: list[float] = []
    spearman_excluded: list[str] = []
    binary_excluded: list[str] = []

    for peptide, peptide_rows in group.groupby("peptide_design_code", sort=True):
        peptide_name = str(peptide)
        scores = peptide_rows["score"].to_numpy(dtype=float)
        retention = peptide_rows["target_retention"].to_numpy(dtype=float)
        labels = peptide_rows["target_binder"].to_numpy(dtype=int)
        rho = _spearman(scores, retention)
        if math.isfinite(rho):
            spearman_values.append(rho)
        else:
            spearman_excluded.append(peptide_name)

        if np.unique(labels).size == 2:
            auroc_values.append(float(roc_auc_score(labels, scores)))
            ap_values.append(float(average_precision_score(labels, scores)))
        else:
            binary_excluded.append(peptide_name)

    _require(spearman_values, "within-peptide Spearman is undefined for every peptide")
    _require(auroc_values, "within-peptide binary metrics are undefined for every peptide")
    return {
        "within_peptide_spearman": float(np.mean(spearman_values)),
        "within_peptide_spearman_evaluable_peptides": len(spearman_values),
        "within_peptide_spearman_excluded_peptides": _json_compact(spearman_excluded),
        "within_peptide_auroc": float(np.mean(auroc_values)),
        "within_peptide_average_precision": float(np.mean(ap_values)),
        "within_peptide_binary_evaluable_peptides": len(auroc_values),
        "within_peptide_binary_excluded_peptides": _json_compact(binary_excluded),
    }


def build_ranking_metrics(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-fit and across-fit summaries in fixed model order."""

    rows: list[dict[str, Any]] = []
    group_columns = ["model_order", "model_key", "display_name", "source_model", "seed"]
    for identifiers, group in selected.groupby(group_columns, sort=False):
        order, model_key, display_name, source_model, seed = identifiers
        row = {
            "model_order": int(order),
            "model_key": str(model_key),
            "display_name": str(display_name),
            "source_model": str(source_model),
            "seed": str(seed),
            **calculate_ranking_metrics(group),
        }
        rows.append(row)

    by_seed = pd.DataFrame(rows).sort_values(["model_order", "seed"], kind="stable")
    metric_names = (
        "within_peptide_spearman",
        "within_peptide_auroc",
        "within_peptide_average_precision",
    )
    summary_rows: list[dict[str, Any]] = []
    for identifiers, group in by_seed.groupby(
        ["model_order", "model_key", "display_name", "source_model"], sort=False
    ):
        order, model_key, display_name, source_model = identifiers
        ordered = group.sort_values("seed", key=lambda s: s.map(_seed_sort_key), kind="stable")
        row: dict[str, Any] = {
            "model_order": int(order),
            "model_key": str(model_key),
            "display_name": str(display_name),
            "source_model": str(source_model),
            "n_fits": len(ordered),
            "within_peptide_binary_evaluable_peptides": int(
                ordered["within_peptide_binary_evaluable_peptides"].iloc[0]
            ),
            "within_peptide_binary_excluded_peptides": ordered[
                "within_peptide_binary_excluded_peptides"
            ].iloc[0],
        }
        for metric in metric_names:
            values = ordered[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
            row[f"{metric}_individual"] = _json_compact(
                [
                    {"seed": str(seed), "value": float(value)}
                    for seed, value in zip(ordered["seed"], values, strict=True)
                ]
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("model_order", kind="stable")
    return by_seed.reset_index(drop=True), summary.reset_index(drop=True)


def select_f1_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Select score >= threshold by exact F1/precision/count/threshold ordering."""

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    _require(score_array.ndim == 1 and label_array.ndim == 1, "scores and labels must be vectors")
    _require(len(score_array) == len(label_array) and len(score_array) > 0, "invalid score/label lengths")
    _require(bool(np.isfinite(score_array).all()), "threshold scores contain non-finite values")
    _require(set(np.unique(label_array)).issubset({0, 1}), "threshold labels must be binary")

    positives = int(label_array.sum())
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for threshold in np.unique(score_array):
        predicted = score_array >= threshold
        selected_count = int(predicted.sum())
        tp = int(np.logical_and(predicted, label_array == 1).sum())
        fp = selected_count - tp
        fn = positives - tp
        tn = len(label_array) - tp - fp - fn
        precision = Fraction(tp, selected_count) if selected_count else Fraction(0, 1)
        recall = Fraction(tp, positives) if positives else Fraction(0, 1)
        denominator = 2 * tp + fp + fn
        f1 = Fraction(2 * tp, denominator) if denominator else Fraction(0, 1)
        result = {
            "threshold": float(threshold),
            "selected_candidates": selected_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        key = (f1, precision, -selected_count, float(threshold))
        candidates.append((key, result))
    return max(candidates, key=lambda item: item[0])[1]


def build_retrospective_threshold_metrics(
    selected: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average fit scores, fit F1 thresholds, and return decisions/counts."""

    panel_columns = [
        "eval_row_id",
        "peptide_design_code",
        "affibody_design_code",
        "target_retention",
        "target_binder",
    ]
    summaries: list[dict[str, Any]] = []
    count_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []

    model_columns = ["model_order", "model_key", "display_name", "source_model"]
    for identifiers, group in selected.groupby(model_columns, sort=False):
        order, model_key, display_name, source_model = identifiers
        fits = sorted(group["seed"].unique().tolist(), key=_seed_sort_key)
        mean_scores = (
            group.groupby("eval_row_id", sort=True, as_index=False)["score"]
            .mean()
            .rename(columns={"score": "mean_score"})
        )
        decisions = panel[panel_columns].merge(
            mean_scores, on="eval_row_id", how="left", validate="one_to_one"
        )
        _require(not bool(decisions["mean_score"].isna().any()), f"missing aggregate scores for {model_key}")
        chosen = select_f1_threshold(decisions["mean_score"], decisions["target_binder"])
        decisions["recommended"] = decisions["mean_score"].ge(chosen["threshold"]).astype(int)
        decisions["correct_recommendation"] = (
            decisions["recommended"].eq(1) & decisions["target_binder"].eq(1)
        ).astype(int)
        decisions.insert(0, "model_order", int(order))
        decisions.insert(1, "model_key", str(model_key))
        decisions.insert(2, "display_name", str(display_name))
        decisions.insert(3, "source_model", str(source_model))
        decisions["threshold"] = chosen["threshold"]
        decisions["calibration_status"] = CALIBRATION_STATUS

        counts = (
            decisions.groupby("peptide_design_code", sort=True)
            .agg(
                candidate_count=("recommended", "sum"),
                true_positive_count=("correct_recommendation", "sum"),
                available_binders=("target_binder", "sum"),
            )
            .reset_index()
        )
        counts["false_positive_count"] = counts["candidate_count"] - counts["true_positive_count"]
        counts["missed_binder_count"] = counts["available_binders"] - counts["true_positive_count"]
        counts.insert(0, "model_order", int(order))
        counts.insert(1, "model_key", str(model_key))
        counts.insert(2, "display_name", str(display_name))
        counts["calibration_status"] = CALIBRATION_STATUS

        candidate_counts = {
            str(row.peptide_design_code): int(row.candidate_count)
            for row in counts.itertuples(index=False)
        }
        zero_candidate_peptides = sorted(
            peptide for peptide, count in candidate_counts.items() if count == 0
        )
        non_falta_binders = decisions[
            decisions["target_binder"].eq(1)
            & decisions["affibody_design_code"].ne(FALTA_CODE)
        ]
        non_falta_selected = int(non_falta_binders["recommended"].sum())
        non_falta_total = len(non_falta_binders)

        summaries.append(
            {
                "model_order": int(order),
                "model_key": str(model_key),
                "display_name": str(display_name),
                "source_model": str(source_model),
                "n_fits_averaged": len(fits),
                "fit_seeds": _json_compact(fits),
                "score_rule": "mean_score_across_fits >= threshold",
                "threshold": chosen["threshold"],
                "selected_candidates": chosen["selected_candidates"],
                "true_positives": chosen["true_positives"],
                "false_positives": chosen["false_positives"],
                "false_negatives": chosen["false_negatives"],
                "true_negatives": chosen["true_negatives"],
                "precision": chosen["precision"],
                "recall": chosen["recall"],
                "f1": chosen["f1"],
                "non_falta_selected_binders": non_falta_selected,
                "non_falta_total_binders": non_falta_total,
                "non_falta_binder_recall": (
                    non_falta_selected / non_falta_total if non_falta_total else float("nan")
                ),
                "candidate_counts_per_peptide": _json_compact(candidate_counts),
                "zero_candidate_peptides": _json_compact(zero_candidate_peptides),
                "calibration_status": CALIBRATION_STATUS,
            }
        )
        count_frames.append(counts)
        decision_frames.append(decisions)

    summary = pd.DataFrame(summaries).sort_values("model_order", kind="stable")
    counts = pd.concat(count_frames, ignore_index=True).sort_values(
        ["model_order", "peptide_design_code"], kind="stable"
    )
    decisions = pd.concat(decision_frames, ignore_index=True).sort_values(
        ["model_order", "peptide_design_code", "affibody_design_code"], kind="stable"
    )
    return summary.reset_index(drop=True), counts.reset_index(drop=True), decisions.reset_index(drop=True)


def _format_metric(mean: float, sd: float | None, n_fits: int) -> str:
    if n_fits == 1 or sd is None or not math.isfinite(float(sd)):
        return f"{mean:.3f} (one fit)"
    return f"{mean:.3f} ± {float(sd):.3f}"


def render_markdown(
    ranking_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    candidate_counts: pd.DataFrame,
) -> str:
    """Render a compact, outsider-readable view without Precision@k."""

    lines = [
        "# LibB operational metrics on the corrected 120-pair panel",
        "",
        "This comparison does **not** use Precision@3 or any other fixed top-k metric. "
        "It asks two separate questions: how well each model orders the ten Affibodies "
        "within each peptide, and what happens if its score is converted into a "
        "recommend/do-not-recommend rule.",
        "",
        "## Peptide-conditioned ranking",
        "",
        "Within-peptide Spearman compares the ordering of all ten Affibodies with their "
        "measured retention values. Within-peptide AUROC and average precision evaluate "
        "binder versus non-binder ordering separately inside each peptide and then average "
        "the result across the 11 peptide rows containing both classes. DP has no measured "
        "binders at the 75% retention definition, so AUROC and average precision are not "
        "defined for DP and it is excluded only from those two binary metrics.",
        "",
        "| Model | Fits | Within-peptide Spearman | Within-peptide AUROC | Within-peptide average precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in ranking_summary.itertuples(index=False):
        lines.append(
            "| {name} | {fits} | {rho} | {auc} | {ap} |".format(
                name=row.display_name,
                fits=row.n_fits,
                rho=_format_metric(
                    row.within_peptide_spearman_mean,
                    row.within_peptide_spearman_sd,
                    row.n_fits,
                ),
                auc=_format_metric(
                    row.within_peptide_auroc_mean,
                    row.within_peptide_auroc_sd,
                    row.n_fits,
                ),
                ap=_format_metric(
                    row.within_peptide_average_precision_mean,
                    row.within_peptide_average_precision_sd,
                    row.n_fits,
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Retrospective threshold exercise",
            "",
            "**These numbers are fitted to the same 120 known retention labels shown in "
            "the table. They are optimistic calibration summaries, not independent test "
            "results.** For five-fit models, the score for each pair is first averaged "
            "across the five fits. Each model then receives its own threshold, chosen to "
            "maximize F1; exact ties prefer higher precision, then fewer recommendations, "
            "then the higher threshold. A future candidate rule must be locked before new "
            "wet-lab results arrive.",
            "",
            "Threshold values should not be compared between models because their score "
            "scales are not calibrated to one another.",
            "",
            "| Model | Threshold | Recommended | Correct binders | Incorrect recommendations | Missed binders | Precision | Recall | F1 | Recall excluding FALTA binders | Peptides with no recommendation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in threshold_summary.itertuples(index=False):
        zero = json.loads(row.zero_candidate_peptides)
        lines.append(
            "| {name} | {threshold:.6g} | {selected}/120 | {tp} | {fp} | {fn} | "
            "{precision:.3f} | {recall:.3f} | {f1:.3f} | {nf:.3f} ({nf_num}/{nf_den}) | {zero} |".format(
                name=row.display_name,
                threshold=row.threshold,
                selected=row.selected_candidates,
                tp=row.true_positives,
                fp=row.false_positives,
                fn=row.false_negatives,
                precision=row.precision,
                recall=row.recall,
                f1=row.f1,
                nf=row.non_falta_binder_recall,
                nf_num=row.non_falta_selected_binders,
                nf_den=row.non_falta_total_binders,
                zero=", ".join(zero) if zero else "None",
            )
        )

    peptide_order = sorted(candidate_counts["peptide_design_code"].unique())
    lines.extend(
        [
            "",
            "## Number of recommended Affibodies for each peptide",
            "",
            "This table shows that a score threshold does not force the model to submit "
            "the same number of candidates for every peptide.",
            "",
            "| Model | " + " | ".join(peptide_order) + " |",
            "|---|" + "---:|" * len(peptide_order),
        ]
    )
    for model_key, group in candidate_counts.groupby("model_key", sort=False):
        lookup = dict(zip(group["peptide_design_code"], group["candidate_count"], strict=True))
        lines.append(
            "| "
            + str(group["display_name"].iloc[0])
            + " | "
            + " | ".join(str(int(lookup[peptide])) for peptide in peptide_order)
            + " |"
        )
    lines.extend(
        [
            "",
            "The machine-readable files retain the individual fit values, every averaged "
            "pair score, and the recommendation decision for every pair.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    )


def _output_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def write_outputs(
    output_dir: Path,
    specs: Sequence[ModelSpec],
    panel: pd.DataFrame,
    ranking_by_seed: pd.DataFrame,
    ranking_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    candidate_counts: pd.DataFrame,
    decisions: pd.DataFrame,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, pd.DataFrame]] = [
        ("ranking_metrics_by_seed.csv", ranking_by_seed.drop(columns=["model_order"])),
        ("ranking_metrics_seed_summary.csv", ranking_summary.drop(columns=["model_order"])),
        ("retrospective_threshold_summary.csv", threshold_summary.drop(columns=["model_order"])),
        (
            "retrospective_candidate_counts_by_peptide.csv",
            candidate_counts.drop(columns=["model_order"]),
        ),
        ("retrospective_mean_score_decisions.csv", decisions.drop(columns=["model_order"])),
    ]
    output_records: dict[str, Any] = {}
    for filename, frame in outputs:
        path = output_dir / filename
        _write_csv(frame, path)
        output_records[filename] = _output_record(path, len(frame))

    markdown_path = output_dir / "operational_metrics.md"
    markdown_path.write_text(
        render_markdown(ranking_summary, threshold_summary, candidate_counts),
        encoding="utf-8",
    )
    output_records[markdown_path.name] = _output_record(markdown_path)

    unique_sources: dict[Path, list[ModelSpec]] = {}
    for spec in specs:
        unique_sources.setdefault(Path(spec.source_path).resolve(), []).append(spec)
    sources = []
    for path, source_specs in unique_sources.items():
        source_manifest = path.parent / "manifest.json"
        record: dict[str, Any] = {
            "path": _relative(path),
            "bytes": path.stat().st_size,
            "mtime_utc": _mtime_utc(path),
            "sha256": _sha256_file(path),
            "selected_models": [item.source_model for item in source_specs],
        }
        if source_manifest.is_file():
            record["manifest"] = {
                "path": _relative(source_manifest),
                "bytes": source_manifest.stat().st_size,
                "mtime_utc": _mtime_utc(source_manifest),
                "sha256": _sha256_file(source_manifest),
            }
        sources.append(record)

    membership_text = "\n".join(sorted(panel["eval_row_id"].astype(str))) + "\n"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "deterministic_summary_of_fixed_corrected_120_predictions",
        "threshold_analysis_status": CALIBRATION_STATUS,
        "threshold_warning": (
            "Threshold and threshold metrics were optimized on these same 120 known labels; "
            "they are not independent estimates of prospective wet-lab performance."
        ),
        "panel": {
            "rows": EXPECTED_ROWS,
            "peptides": EXPECTED_PEPTIDES,
            "affibodies_per_peptide": EXPECTED_AFFIBODIES,
            "binders": EXPECTED_BINDERS,
            "nonbinders": EXPECTED_NONBINDERS,
            "binder_definition": "target_retention >= 75",
            "membership_sha256": hashlib.sha256(membership_text.encode("utf-8")).hexdigest(),
        },
        "ranking_metrics": {
            "within_peptide_spearman": "macro mean across all evaluable peptide rows",
            "within_peptide_auroc": (
                "macro mean across the 11 peptide rows containing both binder classes"
            ),
            "within_peptide_average_precision": (
                "macro mean across the 11 peptide rows containing both binder classes"
            ),
            "binary_metric_excluded_peptides": list(EXPECTED_BINARY_EXCLUDED_PEPTIDES),
            "precision_at_k_reported": False,
        },
        "retrospective_threshold_rule": {
            "score_aggregation": "arithmetic mean across available fits for each pair",
            "decision": "mean_score >= model_specific_threshold",
            "objective": "maximize micro F1 on the known 120 labels",
            "tie_break_order": [
                "higher precision",
                "fewer recommendations",
                "higher threshold",
            ],
            "non_falta_binder_recall_denominator": (
                "all retention>=75 pairs whose Affibody code is not FALTA"
            ),
        },
        "models": [
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "source_model": spec.source_model,
                "expected_fits": spec.expected_fits,
                "source_path": _relative(spec.source_path),
            }
            for spec in specs
        ],
        "sources": sources,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "private_data/experiments/libb_structure_provider_revision_120_operational_metrics_v1",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    specs = default_model_specs()
    selected, panel = load_selected_predictions(specs)
    ranking_by_seed, ranking_summary = build_ranking_metrics(selected)
    threshold_summary, candidate_counts, decisions = build_retrospective_threshold_metrics(
        selected, panel
    )
    manifest = write_outputs(
        args.output_dir,
        specs,
        panel,
        ranking_by_seed,
        ranking_summary,
        threshold_summary,
        candidate_counts,
        decisions,
    )
    print(f"Wrote deterministic LibB operational metrics: {manifest}")
    print(
        "Thresholds are retrospective calibrations on the same 120 labels, "
        "not independent evaluation results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
