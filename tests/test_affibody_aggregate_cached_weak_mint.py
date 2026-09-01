import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from downstream.AffibodyMHC import aggregate_cached_weak_mint as aggregate
from downstream.AffibodyMHC.code_only_baseline import sha256_file


def _source_files(tmp_path):
    paths = {
        "evaluator": tmp_path / "evaluate_cached_weak_mint.py",
        "cache": tmp_path / "mint_cache.npz",
        "retention": tmp_path / "retention_features.npz",
    }
    for name, path in paths.items():
        path.write_bytes((name + "\n").encode("ascii"))
    return paths


def _condition(library="LibA", regime="pair_only", balance="all_unweighted", seed=-1):
    return {
        "library": library,
        "regime": regime,
        "cleaning": "c0",
        "balance": balance,
        "balance_seed": seed,
    }


def _metric_rows(condition):
    rows = []
    for index, representation in enumerate(("site", "frozen_mint_chain_mean")):
        row = dict(condition)
        row.update(
            {
                "evaluation_scope": "common_regime_core",
                "representation": representation,
                "selected_C": 0.1,
                "n": 10,
                "positive": 4,
                "global_prevalence": 0.4,
                "retention_membership_sha256": "r" * 64,
                "global_auprc": 0.5 + index * 0.1,
                "global_auroc": 0.6 + index * 0.1,
                "global_spearman": 0.2 + index * 0.1,
                "within_peptide_macro_auprc_lift": 0.1 + index * 0.1,
                "within_peptide_p_at_1": 0.5 + index * 0.1,
                "two_way_interaction_spearman": -0.1 + index * 0.1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_evaluator_run(run_dir, sources, condition, dependencies=None, solver_tolerance=None):
    run_dir.mkdir()
    metrics = _metric_rows(condition)
    conditions = pd.DataFrame([dict(condition, weak_fit_rows=100)])
    predictions = []
    weak_validation = []
    for representation in ("site", "frozen_mint_chain_mean"):
        predictions.append(
            dict(condition, representation=representation, pair_uid="pair-1", binder_probability=0.5)
        )
        weak_validation.append(
            dict(
                condition,
                representation=representation,
                record_type="aggregate",
                C=0.1,
                fold=-1,
                log_loss=0.7,
            )
        )
    tables = {
        "metrics.csv": metrics,
        "conditions.csv": conditions,
        "predictions.csv": pd.DataFrame(predictions),
        "weak_validation.csv": pd.DataFrame(weak_validation),
    }
    for filename, frame in tables.items():
        frame.to_csv(run_dir / filename, index=False)
    (run_dir / "run_summary.md").write_text("# evaluator summary\n")

    output_records = {}
    for filename in tuple(tables) + ("run_summary.md",):
        path = (run_dir / filename).resolve()
        output_records[filename] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "cached-weak-mint-evaluation-v1",
        "configuration": {"solver_tolerance": solver_tolerance}
        if solver_tolerance is not None
        else {},
        "rows": {
            "conditions": len(conditions),
            "metric_records": len(metrics),
            "prediction_records": len(predictions),
        },
        "outputs": output_records,
        "code": {
            "path": str(sources["evaluator"].resolve()),
            "sha256": sha256_file(sources["evaluator"]),
        },
        "sources": {
            "cache_npz": [
                {
                    "path": str(sources["cache"].resolve()),
                    "sha256": sha256_file(sources["cache"]),
                }
            ],
            "retention_features": {
                "path": str(sources["retention"].resolve()),
                "sha256": sha256_file(sources["retention"]),
            },
        },
    }
    if dependencies is not None:
        manifest["dependencies"] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in dependencies.items()
        }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_aggregate_writes_all_records_with_provenance_and_primary_table(tmp_path, monkeypatch):
    sources = _source_files(tmp_path)
    first = _write_evaluator_run(
        tmp_path / "run-pair", sources, _condition(regime="pair_only")
    )
    second = _write_evaluator_run(
        tmp_path / "run-peptide", sources, _condition(regime="peptide_cold")
    )
    output_dir = tmp_path / "private_data" / "aggregate"
    output_dir.parent.mkdir()
    monkeypatch.setattr(
        aggregate,
        "validate_private_output_path",
        lambda path, _root: Path(path).resolve(),
    )

    aggregate.run(
        argparse.Namespace(run_dirs=[first, second], output_dir=output_dir)
    )

    combined_metrics = pd.read_csv(output_dir / "aggregate_metrics.csv")
    assert len(combined_metrics) == 4
    assert set(combined_metrics["source_run"]) == {"run-pair", "run-peptide"}
    assert len(pd.read_csv(output_dir / "aggregate_conditions.csv")) == 2
    assert len(pd.read_csv(output_dir / "aggregate_predictions.csv")) == 4
    assert len(pd.read_csv(output_dir / "aggregate_weak_validation.csv")) == 4
    primary = pd.read_csv(output_dir / "primary_metrics.csv")
    assert len(primary) == 4
    summary = (output_dir / "primary_table.md").read_text()
    assert "site / frozen MINT" in summary
    assert "pair_only" in summary and "peptide_cold" in summary
    assert "0.1 (MAX) / 0.1 (MAX)" in summary
    assert set(primary["selected_C_at_grid_max_any"]) == {1}

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == "cached-weak-mint-aggregate-v1"
    assert len(manifest["source_runs"]) == 2
    for filename, record in manifest["outputs"].items():
        assert sha256_file(output_dir / filename) == record["sha256"]


def test_modified_evaluator_output_is_rejected_before_loading(tmp_path):
    sources = _source_files(tmp_path)
    run_dir = _write_evaluator_run(
        tmp_path / "run-corrupt", sources, _condition()
    )
    with (run_dir / "metrics.csv").open("a") as handle:
        handle.write("corrupt\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        aggregate.load_verified_run(run_dir)


def test_duplicate_condition_across_runs_is_rejected(tmp_path):
    sources = _source_files(tmp_path)
    condition = _condition()
    first = aggregate.load_verified_run(
        _write_evaluator_run(tmp_path / "run-one", sources, condition)
    )
    second = aggregate.load_verified_run(
        _write_evaluator_run(tmp_path / "run-two", sources, condition)
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        aggregate.combine_verified_runs([first, second])


def test_mismatched_cache_source_is_rejected(tmp_path):
    sources = _source_files(tmp_path)
    other_sources = dict(sources)
    other_sources["cache"] = tmp_path / "different_cache.npz"
    other_sources["cache"].write_bytes(b"different cache\n")
    first = aggregate.load_verified_run(
        _write_evaluator_run(
            tmp_path / "run-one", sources, _condition(regime="pair_only")
        )
    )
    second = aggregate.load_verified_run(
        _write_evaluator_run(
            tmp_path / "run-two", other_sources, _condition(regime="double_cold")
        )
    )

    with pytest.raises(ValueError, match="different evaluator/cache/retention source"):
        aggregate.combine_verified_runs([first, second])


def test_practical_wrapper_base_dependency_is_verified_and_compared(tmp_path):
    sources = _source_files(tmp_path)
    base = tmp_path / "base_evaluator.py"
    base.write_text("# base evaluator\n")
    first = aggregate.load_verified_run(
        _write_evaluator_run(
            tmp_path / "run-one",
            sources,
            _condition(regime="pair_only"),
            dependencies={"base_evaluator": base},
            solver_tolerance=1e-4,
        )
    )
    second_dir = _write_evaluator_run(
        tmp_path / "run-two",
        sources,
        _condition(regime="double_cold"),
        dependencies={"base_evaluator": base},
        solver_tolerance=1e-4,
    )
    second = aggregate.load_verified_run(second_dir)
    aggregate.combine_verified_runs([first, second])

    # Changing the live dependency invalidates both the manifest binding and
    # any attempted aggregate before CSV rows are trusted.
    base.write_text("# changed base evaluator\n")
    with pytest.raises(ValueError, match="dependency base_evaluator hash mismatch"):
        aggregate.load_verified_run(second_dir)


def test_primary_view_averages_downsample_seeds_but_not_representations():
    blocks = []
    for seed, shift in ((0, 0.0), (1, 0.1), (2, 0.2)):
        block = _metric_rows(
            _condition(balance="balanced_downsample", seed=seed)
        )
        block["global_auprc"] += shift
        block.insert(0, "source_run", "run")
        blocks.append(block)
    primary = aggregate.build_primary_view(pd.concat(blocks, ignore_index=True))

    assert len(primary) == 2
    assert set(primary["n_seed_runs"]) == {3}
    site = primary.loc[primary["representation"].eq("site")].iloc[0]
    mint = primary.loc[primary["representation"].eq("frozen_mint_chain_mean")].iloc[0]
    assert site["global_auprc"] == pytest.approx(0.6)
    assert mint["global_auprc"] == pytest.approx(0.7)
    assert site["global_auprc_seed_sd"] == pytest.approx(0.1)
    assert site["balance_seeds"] == "0,1,2"


def test_primary_view_rejects_a_common_core_that_changes_across_regimes():
    first = _metric_rows(_condition(regime="pair_only"))
    second = _metric_rows(_condition(regime="double_cold"))
    second["retention_membership_sha256"] = "x" * 64
    first.insert(0, "source_run", "run-pair")
    second.insert(0, "source_run", "run-double")

    with pytest.raises(ValueError, match="common-regime core changes"):
        aggregate.build_primary_view(pd.concat([first, second], ignore_index=True))
