import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from downstream.AffibodyMHC import (
    aggregate_mint_selection_placement_control as aggregate,
)


def _metrics(libraries=("LibA", "LibB")):
    rows = []
    paired = []
    for library_index, library in enumerate(libraries):
        for seed_index, seed in enumerate(aggregate.matched.DEFAULT_TRAINING_SEEDS):
            epoch = 1 + (seed_index % 2)
            cross = {
                "global_auroc": 0.70 + 0.01 * seed_index,
                "global_auprc": 0.60 + 0.01 * seed_index,
                "global_spearman": 0.50 + 0.01 * seed_index,
                "within_peptide_macro_spearman": 0.20 + 0.01 * seed_index,
            }
            within = {name: value - 0.02 for name, value in cross.items()}
            for arm, values in (("lora_cross", cross), ("lora_within", within)):
                rows.append(
                    {
                        "library": library,
                        "arm": arm,
                        "training_seed": seed,
                        "selected_epoch": epoch,
                        **values
                    }
                )
            paired.append(
                {
                    "library": library,
                    "training_seed": seed,
                    "selected_epoch": epoch,
                    "comparison": "lora_within_minus_lora_cross",
                    **{
                        name + "_change": within[name] - cross[name]
                        for name in aggregate.METRICS
                    }
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(paired)


def test_summary_uses_paired_within_minus_cross_and_sample_sd():
    metrics, paired = _metrics(("LibA",))
    summary = aggregate.build_summary(metrics, paired)
    assert len(summary) == 3
    change = summary.loc[summary["result_type"].eq("paired_change")].iloc[0]
    for metric in aggregate.METRICS:
        assert change[metric + "_mean"] == pytest.approx(-0.02)
        assert change[metric + "_sample_sd"] == pytest.approx(0.0, abs=1e-15)
    markdown = aggregate.summary_markdown(summary)
    assert "Within - cross" in markdown
    assert "negative within-minus-cross value favors cross-chain" in markdown
    assert "exploratory" in markdown


def _gate_dir(tmp_path, passed):
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    table = pd.DataFrame(
        [
            {
                "library": library,
                "n_seeds": len(aggregate.matched.DEFAULT_TRAINING_SEEDS),
                "placement_gate_pass": library in set(passed),
            }
            for library in aggregate.matched.LIBRARIES
        ]
    )
    table.to_csv(gate_dir / "within_chain_placement_gate.csv", index=False)
    return {
        "run_dir": gate_dir,
        "manifest_sha256": "a" * 64,
        "gate_table": table,
    }


def _fake_runs(tmp_path, passed):
    gate = _gate_dir(tmp_path, passed)
    return [
        {
            "library": library,
            "training_seed": int(seed),
            "gate": gate,
        }
        for library in passed
        for seed in aggregate.matched.DEFAULT_TRAINING_SEEDS
    ]


def test_grid_requires_every_seed_for_every_and_only_passed_libraries(tmp_path):
    runs = _fake_runs(tmp_path, ("LibB",))
    assert aggregate.validate_run_grid(runs) == {"LibB"}
    with pytest.raises(ValueError, match="grid is incomplete"):
        aggregate.validate_run_grid(runs[:-1])
    extra = list(runs) + [
        {
            "library": "LibA",
            "training_seed": aggregate.matched.DEFAULT_TRAINING_SEEDS[0],
            "gate": runs[0]["gate"],
        }
    ]
    with pytest.raises(ValueError, match="differ from the libraries"):
        aggregate.validate_run_grid(extra)


def test_grid_rejects_duplicate_missing_and_extra_seed_metadata(tmp_path):
    runs = _fake_runs(tmp_path, ("LibA",))
    duplicate = [dict(run) for run in runs]
    duplicate[-1]["training_seed"] = duplicate[0]["training_seed"]
    with pytest.raises(ValueError, match="grid is incomplete"):
        aggregate.validate_run_grid(duplicate)

    missing = runs[:-1]
    with pytest.raises(ValueError, match="grid is incomplete"):
        aggregate.validate_run_grid(missing)

    alien = [dict(run) for run in runs]
    alien[-1]["training_seed"] = 99999999
    with pytest.raises(ValueError, match="grid is incomplete"):
        aggregate.validate_run_grid(alien)


def test_configuration_fixes_placement_and_parameter_count():
    configuration = {
        "library": "LibA",
        "training_seed": aggregate.matched.DEFAULT_TRAINING_SEEDS[0],
        "arms": list(aggregate.ARMS),
        "arm_reuse": {
            "lora_cross": "verified shared-epoch source",
            "lora_within": "new refit",
        },
        "shared_positive_epoch": 2,
        "selected_C": aggregate.matched.PRIMARY_CONTRACT["LibA"]["selected_c"],
        **aggregate.TRAINING_CONTRACT,
        "lora": {
            "rank": aggregate.trainer.RANK,
            "alpha": aggregate.trainer.ALPHA,
            "dropout": aggregate.trainer.DROPOUT,
            "layers_zero_based": list(aggregate.trainer.LAYERS),
            "projections": list(aggregate.trainer.PROJECTIONS),
            "placements": dict(aggregate.trainer.PLACEMENTS),
            "adapter_parameters_per_arm": aggregate.trainer.ADAPTER_PARAMETERS,
            "total_trainable_parameters_per_arm": aggregate.trainer.TOTAL_TRAINABLE_PARAMETERS,
        },
        "primary_endpoint": "within_peptide_macro_spearman",
    }
    manifest = {
        "configuration": configuration,
        "canonical_contract": aggregate.matched.PRIMARY_CONTRACT["LibA"],
    }
    library, seed, observed = aggregate._validate_configuration(
        manifest
    )
    assert library == "LibA"
    assert seed == aggregate.matched.DEFAULT_TRAINING_SEEDS[0]
    assert observed["lora"]["placements"] == {
        "lora_cross": "multimer_attn",
        "lora_within": "self_attn",
    }
    broken = {"configuration": dict(configuration)}
    broken["configuration"]["lora"] = dict(configuration["lora"])
    broken["configuration"]["lora"]["adapter_parameters_per_arm"] -= 1
    with pytest.raises(ValueError, match="LoRA contract changed"):
        aggregate._validate_configuration(broken)

    for field in ("training_seed", "shared_positive_epoch"):
        fractional = json.loads(json.dumps(manifest))
        fractional["configuration"][field] += 0.5
        with pytest.raises(ValueError, match="non-integral"):
            aggregate._validate_configuration(fractional)


def _paired_prediction_panel():
    panel = pd.DataFrame(
        {
            "pair_uid": ["p0", "p1", "p2", "p3", "p4", "p5"],
            "chain1_sha256": ["pep0"] * 3 + ["pep1"] * 3,
            "chain2_sha256": ["aff0", "aff1", "aff2"] * 2,
            "sequence_pair_sha256": ["s0", "s1", "s2", "s3", "s4", "s5"],
            "target_retention": [10.0, 80.0, 90.0, 20.0, 70.0, 100.0],
            "target_binder": [0, 1, 1, 0, 0, 1],
        }
    )
    blocks = []
    for arm, probability in (
        ("lora_cross", [0.1, 0.7, 0.8, 0.2, 0.6, 0.9]),
        ("lora_within", [0.2, 0.6, 0.9, 0.1, 0.5, 0.8]),
    ):
        block = panel.copy()
        block["library"] = "LibA"
        block["arm"] = arm
        block["training_seed"] = aggregate.matched.DEFAULT_TRAINING_SEEDS[0]
        block["selected_epoch"] = 2
        block["probability"] = probability
        blocks.append(block)
    return blocks


def test_cross_within_panel_accepts_only_probability_difference():
    cross, within = _paired_prediction_panel()
    observed = aggregate._audit_paired_prediction_panel(
        pd.concat([cross, within], ignore_index=True)
    )
    assert observed.loc[:, aggregate.PAIRED_PANEL_COLUMNS].equals(
        cross.loc[:, aggregate.PAIRED_PANEL_COLUMNS]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "row_order",
        "target_retention",
        "target_binder",
        "pair_uid",
        "chain1_sha256",
        "chain2_sha256",
        "sequence_pair_sha256",
    ),
)
def test_cross_within_panel_rejects_order_target_or_group_drift(mutation):
    cross, within = _paired_prediction_panel()
    if mutation == "row_order":
        within = within.iloc[::-1].reset_index(drop=True)
    elif mutation == "target_retention":
        within.loc[0, mutation] = 11.0
    elif mutation == "target_binder":
        within.loc[0, mutation] = 1
    else:
        within.loc[0, mutation] = "changed"
    with pytest.raises(ValueError, match="prediction panels differ"):
        aggregate._audit_paired_prediction_panel(
            pd.concat([cross, within], ignore_index=True)
        )


def test_configuration_requires_full_optimizer_and_training_contract():
    configuration = {
        "library": "LibA",
        "training_seed": aggregate.matched.DEFAULT_TRAINING_SEEDS[0],
        "arms": list(aggregate.ARMS),
        "arm_reuse": {
            "lora_cross": "verified shared-epoch source",
            "lora_within": "new refit",
        },
        "shared_positive_epoch": 2,
        "selected_C": aggregate.matched.PRIMARY_CONTRACT["LibA"]["selected_c"],
        **aggregate.TRAINING_CONTRACT,
        "lora": {
            "rank": aggregate.trainer.RANK,
            "alpha": aggregate.trainer.ALPHA,
            "dropout": aggregate.trainer.DROPOUT,
            "layers_zero_based": list(aggregate.trainer.LAYERS),
            "projections": list(aggregate.trainer.PROJECTIONS),
            "placements": dict(aggregate.trainer.PLACEMENTS),
            "adapter_parameters_per_arm": aggregate.trainer.ADAPTER_PARAMETERS,
            "total_trainable_parameters_per_arm": aggregate.trainer.TOTAL_TRAINABLE_PARAMETERS,
        },
    }
    manifest = {
        "configuration": configuration,
        "canonical_contract": aggregate.matched.PRIMARY_CONTRACT["LibA"],
    }
    aggregate._validate_configuration(manifest)
    for key, bad in (
        ("adapter_lr", 99.0),
        ("batch_size", 1),
        ("warmup_fraction", 0.9),
        ("retention_usage", "used during refit"),
        ("optimizer", {"name": "SGD"}),
    ):
        changed = json.loads(json.dumps(manifest))
        changed["configuration"][key] = bad
        with pytest.raises(ValueError, match="placement {} changed".format(key)):
            aggregate._validate_configuration(changed)


def test_training_configuration_is_bound_to_verified_shared_source():
    configuration = {
        "shared_positive_epoch": 2,
        "selected_C": 0.01,
        **aggregate.TRAINING_CONTRACT,
        "lora": {
            "rank": aggregate.trainer.RANK,
            "alpha": aggregate.trainer.ALPHA,
            "dropout": aggregate.trainer.DROPOUT,
            "layers_zero_based": list(aggregate.trainer.LAYERS),
            "projections": list(aggregate.trainer.PROJECTIONS),
            "placements": dict(aggregate.trainer.PLACEMENTS),
        },
    }
    source_configuration = {
        key: configuration[key]
        for key in (
            "selected_C",
            "lr_schedule_horizon_epochs",
            "batch_size",
            "eval_batch_size",
            "accumulation_steps",
            "head_lr",
            "adapter_lr",
            "weight_decay",
            "warmup_fraction",
            "clip_norm",
        )
    }
    source_configuration.update(
        {
            "chosen_positive_epoch": 2,
            "lora": {
                "rank": aggregate.trainer.RANK,
                "alpha": aggregate.trainer.ALPHA,
                "dropout": aggregate.trainer.DROPOUT,
                "layers_zero_based": list(aggregate.trainer.LAYERS),
                "projections": list(aggregate.trainer.PROJECTIONS),
                "placement": "multimer_attn",
            },
        }
    )
    source = {"configuration": source_configuration}
    aggregate._audit_configuration_against_source(configuration, source)
    source_configuration["adapter_lr"] = 9.0
    with pytest.raises(ValueError, match="adapter_lr differs"):
        aggregate._audit_configuration_against_source(configuration, source)


def _training_audit_fixture():
    library = "LibA"
    seed = aggregate.matched.DEFAULT_TRAINING_SEEDS[0]
    epoch = 2
    canonical = aggregate.matched.PRIMARY_CONTRACT[library]
    shared_hash = "1" * 64
    gate_hash = "2" * 64
    source_cross = {
        "stage": "final_refit",
        "arm": "lora_cross",
        "training_seed": seed,
        "fold": -1,
        "rows": canonical["rows"],
        "positive": canonical["positive"],
        "negative": canonical["negative"],
        "class_weight_negative": 1.01,
        "class_weight_positive": 0.99,
        "membership_sha256": canonical["membership_sha256"],
        "feature_mean_sha256": "3" * 64,
        "feature_scale_sha256": "4" * 64,
        "trainable_parameters": aggregate.trainer.TOTAL_TRAINABLE_PARAMETERS,
        "trainable_names": json.dumps(
            sorted(aggregate.trainer.expected_trainable_names("lora_cross"))
        ),
        "run_seed": aggregate.matched.derived_seed(seed, "final_refit", -1),
        "selected_epoch": epoch,
        "schedule_total_steps": 1059,
        "epoch0_probability_max_abs_error": 0.0,
        "head_parameter_delta_l2": 1.0,
        "head_parameter_delta_max_abs": 0.1,
        "adapter_parameter_delta_l2": 1.0,
        "adapter_parameter_delta_max_abs": 0.1,
        "head_initialization_sha256": "5" * 64,
    }
    rows = []
    for arm in aggregate.ARMS:
        row = dict(source_cross)
        row.update(
            {
                "arm": arm,
                "trainable_names": json.dumps(
                    sorted(aggregate.trainer.expected_trainable_names(arm))
                ),
                "arm_source": (
                    "reused_verified_shared_epoch"
                    if arm == "lora_cross"
                    else "new_parameter_matched_refit"
                ),
                "attention_placement": aggregate.trainer.PLACEMENTS[arm],
                "adapter_parameters": aggregate.trainer.ADAPTER_PARAMETERS,
                "normalized_adapter_initialization_sha256": "6" * 64,
                "training_order_sha256": "7" * 64,
                "source_shared_manifest_sha256": shared_hash,
                "gate_aggregate_manifest_sha256": gate_hash,
            }
        )
        rows.append(row)
    source_shared = {
        "manifest_sha256": shared_hash,
        "training_audit": pd.DataFrame([source_cross]),
        "manifest": {
            "refit_diagnostics": {
                "live_adapter_initialization_sha256": "8" * 64
            }
        },
    }
    gate = {"manifest_sha256": gate_hash}
    return pd.DataFrame(rows), source_shared, gate, library, seed, epoch


def test_reused_cross_training_audit_is_exactly_bound_to_verified_source():
    training, source, gate, library, seed, epoch = _training_audit_fixture()
    aggregate._audit_training(training, library, seed, epoch, source, gate)
    training.loc[:, "head_initialization_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="reused cross training audit"):
        aggregate._audit_training(training, library, seed, epoch, source, gate)

    training, source, gate, library, seed, epoch = _training_audit_fixture()
    training.loc[training["arm"].eq("lora_within"), "training_seed"] = seed + 0.5
    with pytest.raises(ValueError, match="non-integral"):
        aggregate._audit_training(training, library, seed, epoch, source, gate)


def test_pairing_diagnostics_bind_initialization_order_and_live_trainables():
    training, source, _, _, _, _ = _training_audit_fixture()
    contract = pd.DataFrame(
        [
            {
                "cross_within_epoch0_max_abs_probability_difference": 0.0,
                "normalized_adapter_initialization_sha256": "6" * 64,
            }
        ]
    )
    diagnostics = {
        "training_order_sha256": "7" * 64,
        "run_seed": int(training.iloc[0]["run_seed"]),
        "max_abs_probability_difference": 0.0,
        "head_initialization_sha256": "5" * 64,
        "cross_raw_adapter_initialization_sha256": "8" * 64,
        "within_raw_adapter_initialization_sha256": "9" * 64,
        "normalized_adapter_initialization_sha256": "6" * 64,
        "trainable": {
            arm: {
                "names": sorted(aggregate.trainer.expected_trainable_names(arm)),
                "count": aggregate.trainer.TOTAL_TRAINABLE_PARAMETERS,
                "adapter_count": aggregate.trainer.ADAPTER_PARAMETERS,
            }
            for arm in aggregate.ARMS
        },
    }
    manifest = {"pairing_diagnostics": diagnostics}
    aggregate._audit_pairing_diagnostics(manifest, contract, training, source)
    diagnostics["cross_raw_adapter_initialization_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="differs from verified source"):
        aggregate._audit_pairing_diagnostics(manifest, contract, training, source)


def _fake_publication_runs(root):
    gate = _gate_dir(root, ("LibA",))
    gate.update(
        {
            "manifest_path": gate["run_dir"] / "manifest.json",
            "manifest": {"source_runs": []},
            "verified_outputs": {},
            "verified_source_runs": [],
        }
    )
    gate["manifest_path"].write_text("{}\n")
    gate["manifest_sha256"] = aggregate.sha256_file(gate["manifest_path"])
    cache_dir = root / "verified-feature-cache"
    cache_dir.mkdir()
    cache_shard = cache_dir / "features-shard-000.npz"
    cache_shard.write_bytes(b"verified shard")
    all_metrics, all_paired = _metrics(("LibA",))
    runs = {}
    for seed in aggregate.matched.DEFAULT_TRAINING_SEEDS:
        run_dir = root / "placement-{}".format(seed)
        run_dir.mkdir()
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text("{}\n")
        metric = all_metrics.loc[all_metrics["training_seed"].eq(seed)].copy()
        paired = all_paired.loc[all_paired["training_seed"].eq(seed)].copy()
        source_dir = root / "source-{}".format(seed)
        source_dir.mkdir()
        matched_source_dir = root / "matched-source-{}".format(seed)
        matched_source_dir.mkdir()
        runs[run_dir.resolve()] = {
            "run_dir": run_dir.resolve(),
            "manifest_path": manifest_path,
            "manifest_sha256": aggregate.sha256_file(manifest_path),
            "outputs": {},
            "library": "LibA",
            "training_seed": int(seed),
            "selected_epoch": int(metric["selected_epoch"].iloc[0]),
            "gate": gate,
            "source_shared": {
                "run_dir": source_dir.resolve(),
                "source": {
                    "run_dir": matched_source_dir.resolve(),
                    "sources": {
                        "cache_npz_000": {"path": str(cache_shard.resolve())}
                    },
                },
            },
            "predictions": pd.DataFrame(
                {
                    "library": ["LibA", "LibA"],
                    "arm": list(aggregate.ARMS),
                    "training_seed": [seed, seed],
                    "probability": [0.25, 0.75],
                }
            ),
            "metrics": metric,
            "paired": paired,
            "contract": pd.DataFrame(
                {"library": ["LibA"], "training_seed": [seed]}
            ),
            "training": pd.DataFrame(
                {
                    "library": ["LibA", "LibA"],
                    "arm": list(aggregate.ARMS),
                    "training_seed": [seed, seed],
                }
            ),
        }
    return runs


def test_run_publishes_on_lustre_manifest_last_and_refuses_overwrite_and_race(
    monkeypatch,
):
    root = Path(
        os.environ.get("MINT_LUSTRE_TEST_ROOT", aggregate.REPO_ROOT / "private_data")
    ).resolve()
    filesystem = subprocess.check_output(
        ["stat", "-f", "-c", "%T", str(root)], text=True
    ).strip()
    if filesystem != "lustre":
        pytest.skip("requires a Lustre test root")
    with tempfile.TemporaryDirectory(
        prefix=".placement-aggregate-run-test-", dir=str(root)
    ) as temporary:
        temporary = Path(temporary)
        runs = _fake_publication_runs(temporary)
        monkeypatch.setattr(
            aggregate,
            "load_verified_run",
            lambda path: runs[Path(path).resolve()],
        )
        rechecks = []
        monkeypatch.setattr(
            aggregate, "_recheck", lambda runs, hashes: rechecks.append(True)
        )
        output = temporary / "published"
        args = SimpleNamespace(run_dirs=list(runs), output_dir=output)
        link_order = []
        real_link = aggregate.trainer.shared_aggregate.os.link

        def recording_link(source, destination, **kwargs):
            link_order.append(destination)
            return real_link(source, destination, **kwargs)

        monkeypatch.setattr(
            aggregate.trainer.shared_aggregate.os, "link", recording_link
        )
        manifest = aggregate.run(args)
        assert len(rechecks) == 3
        expected = {
            "retention_predictions_by_seed.csv",
            "recomputed_retention_metrics_by_seed.csv",
            "paired_within_minus_cross_by_seed.csv",
            "placement_contract_by_seed.csv",
            "training_audit_by_seed.csv",
            "mean_sample_sd_summary.csv",
            "summary.md",
            "manifest.json",
        }
        assert {path.name for path in output.iterdir()} == expected
        assert link_order[-1] == "manifest.json"
        assert stat.S_IMODE(output.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in output.iterdir())
        for name, record in manifest["outputs"].items():
            assert record["sha256"] == aggregate.sha256_file(output / name)
        before = {
            path.name: (path.stat().st_ino, aggregate.sha256_file(path))
            for path in output.iterdir()
        }
        with pytest.raises(ValueError, match="refusing overwrite"):
            aggregate.run(args)
        after = {
            path.name: (path.stat().st_ino, aggregate.sha256_file(path))
            for path in output.iterdir()
        }
        assert after == before

        nested_output = next(iter(runs)) / "nested-output"
        nested_args = SimpleNamespace(run_dirs=list(runs), output_dir=nested_output)
        with pytest.raises(ValueError, match="must be disjoint"):
            aggregate.run(nested_args)
        assert not nested_output.exists()

        cache_sibling_output = (
            temporary / "verified-feature-cache" / "poison-cache-inventory"
        )
        cache_sibling_args = SimpleNamespace(
            run_dirs=list(runs), output_dir=cache_sibling_output
        )
        with pytest.raises(ValueError, match="must be disjoint"):
            aggregate.run(cache_sibling_args)
        assert not cache_sibling_output.exists()

        raced_output = temporary / "raced"
        raced_args = SimpleNamespace(run_dirs=list(runs), output_dir=raced_output)
        real_mkdir = aggregate.trainer.shared_aggregate.os.mkdir

        def raced_mkdir(path, mode=0o777, *args, **kwargs):
            if Path(path) == raced_output:
                real_mkdir(path, mode, *args, **kwargs)
                (raced_output / "existing.txt").write_text("existing")
                raise FileExistsError("injected publication race")
            return real_mkdir(path, mode, *args, **kwargs)

        monkeypatch.setattr(
            aggregate.trainer.shared_aggregate.os, "mkdir", raced_mkdir
        )
        with pytest.raises(ValueError, match="refusing overwrite"):
            aggregate.run(raced_args)
        assert (raced_output / "existing.txt").read_text() == "existing"
        assert {path.name for path in raced_output.iterdir()} == {"existing.txt"}
        assert list(temporary.glob(".raced-staging-*")) == []
