import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from downstream.AffibodyMHC import (
    finetune_mint_selection_placement_control as placement,
)


def _selection_and_paired():
    selection_rows = []
    paired_rows = []
    for library in placement.matched.LIBRARIES:
        for seed in placement.matched.DEFAULT_TRAINING_SEEDS:
            for epoch in range(4):
                selection_rows.append(
                    {
                        "library": library,
                        "training_seed": seed,
                        "epoch": epoch,
                        "n": 10,
                        "positive": 5,
                        "head_only_log_loss": 0.50 if epoch == 1 else 0.70,
                        "lora_cross_log_loss": 0.40 if epoch == 1 else 0.80,
                        "equal_arm_mean_log_loss": 0.45 if epoch == 1 else 0.75,
                        "selected": int(epoch == 1),
                        "chosen_positive_epoch": 1,
                        "epoch0_better": False,
                        "epoch0_no_worse": False,
                        "epoch0_minus_selected_log_loss": 0.30,
                    }
                )
            paired_rows.append(
                {
                    "library": library,
                    "training_seed": seed,
                    "selected_epoch": 1,
                    "comparison": "lora_cross_minus_head_only",
                    "global_auroc_change": 0.01,
                    "global_auprc_change": 0.01,
                    "global_spearman_change": 0.01,
                    "within_peptide_macro_spearman_change": 0.01,
                }
            )
    return pd.DataFrame(selection_rows), pd.DataFrame(paired_rows)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_metrics_from_paired(paired):
    rows = []
    for paired_row in paired.to_dict("records"):
        for arm in ("head_only", "lora_cross"):
            row = {
                "library": paired_row["library"],
                "arm": arm,
                "training_seed": int(paired_row["training_seed"]),
                "selected_epoch": int(paired_row["selected_epoch"]),
            }
            for metric in placement.shared_aggregate.SUMMARY_METRICS:
                baseline = 0.50
                row[metric] = baseline + (
                    float(paired_row[metric + "_change"])
                    if arm == "lora_cross"
                    else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _make_gate_aggregate(tmp_path, monkeypatch, fail_library=None):
    gate_dir = tmp_path / "aggregate"
    gate_dir.mkdir()
    selection, paired = _selection_and_paired()
    if fail_library is not None:
        paired.loc[
            paired["library"].eq(fail_library)
            & paired["training_seed"].eq(placement.matched.DEFAULT_TRAINING_SEEDS[0]),
            "within_peptide_macro_spearman_change",
        ] = 0.0
    source_metrics = _source_metrics_from_paired(paired)
    gate = placement.shared_aggregate.build_within_chain_gate(selection, paired)
    tables = {
        "shared_epoch_selection_by_seed.csv": selection,
        "paired_cross_minus_head_by_seed.csv": paired,
        "within_chain_placement_gate.csv": gate,
        "retention_predictions_by_seed.csv": pd.DataFrame({"x": [1]}),
        "recomputed_retention_metrics_by_seed.csv": pd.DataFrame({"x": [1]}),
        "training_audit_by_seed.csv": pd.DataFrame({"x": [1]}),
        "mean_sample_sd_summary.csv": pd.DataFrame({"x": [1]}),
    }
    for name, frame in tables.items():
        frame.to_csv(gate_dir / name, index=False)
    (gate_dir / "summary.md").write_text("summary\n")
    outputs = {
        name: {
            "path": str((gate_dir / name).resolve()),
            "sha256": _sha(gate_dir / name),
        }
        for name in placement.GATE_OUTPUTS
    }
    configuration_contract = {
        "batch_size": placement.TRAINING_CONTRACT["batch_size"],
        "max_epochs": placement.TRAINING_CONTRACT["max_epochs"],
    }
    runtime_contract = {"torch": "test-torch", "torch_cuda": None}
    common_source_hashes = {"checkpoint": "c" * 64, "weak_cache": "d" * 64}
    library_reference_source_hashes = {
        library: {"reference": hashlib.sha256(library.encode("ascii")).hexdigest()}
        for library in placement.matched.LIBRARIES
    }
    source_runs = []
    verified_by_path = {}
    for library in placement.matched.LIBRARIES:
        for seed in placement.matched.DEFAULT_TRAINING_SEEDS:
            path = (tmp_path / "sources" / "{}_{}".format(library, seed)).resolve()
            path.mkdir(parents=True)
            matched_path = (
                tmp_path / "matched_sources" / "{}_{}".format(library, seed)
            ).resolve()
            matched_path.mkdir(parents=True)
            matched_manifest_path = matched_path / "manifest.json"
            matched_manifest_path.write_text(
                json.dumps(
                    {
                        "kind": "matched-source",
                        "library": library,
                        "training_seed": int(seed),
                    },
                    sort_keys=True,
                )
            )
            source_provenance = {
                "path": str(matched_path),
                "manifest_path": str(matched_manifest_path),
                "manifest_sha256": _sha(matched_manifest_path),
                "library": library,
                "training_seed": int(seed),
            }
            source_configuration = {
                "library": library,
                "training_seeds": [int(seed)],
                "selected_C": placement.matched.PRIMARY_CONTRACT[library][
                    "selected_c"
                ],
                "chosen_positive_epoch": 1,
                "lr_schedule_horizon_epochs": 3,
                **copy.deepcopy(placement.TRAINING_CONTRACT),
            }
            source_manifest = {
                "schema_version": "fake-shared-source-v1",
                "library": library,
                "training_seed": int(seed),
                "source_matched_run": source_provenance,
                "configuration": source_configuration,
            }
            manifest_path = path / "manifest.json"
            manifest_path.write_text(json.dumps(source_manifest, sort_keys=True))
            manifest_hash = _sha(manifest_path)
            run_selection = selection.loc[
                selection["library"].eq(library)
                & selection["training_seed"].eq(int(seed))
            ].copy()
            run_metrics = source_metrics.loc[
                source_metrics["library"].eq(library)
                & source_metrics["training_seed"].eq(int(seed))
            ].copy()
            contract = placement.matched.PRIMARY_CONTRACT[library]
            schedule_steps = int(
                np.ceil(
                    float(contract["rows"])
                    / float(placement.TRAINING_CONTRACT["batch_size"])
                )
            ) * 3
            run_seed = placement.matched.derived_seed(seed, "final_refit", -1)
            training_audit = pd.DataFrame(
                [
                    {
                        "arm": arm,
                        "training_seed": int(seed),
                        "fold": -1,
                        "rows": int(contract["rows"]),
                        "positive": int(contract["positive"]),
                        "negative": int(contract["negative"]),
                        "run_seed": int(run_seed),
                        "selected_epoch": 1,
                        "schedule_total_steps": schedule_steps,
                        "lr_schedule_horizon_epochs": 3,
                        "trainable_parameters": count,
                    }
                    for arm, count in (
                        ("head_only", placement.HEAD_PARAMETERS),
                        ("lora_cross", placement.TOTAL_TRAINABLE_PARAMETERS),
                    )
                ]
            )
            verified = {
                "run_dir": path,
                "run_name": path.name,
                "manifest_path": manifest_path,
                "manifest_sha256": manifest_hash,
                "manifest": source_manifest,
                "library": library,
                "training_seed": int(seed),
                "configuration": source_configuration,
                "runtime_contract": copy.deepcopy(runtime_contract),
                "verified_outputs": {},
                "verified_code": {},
                "source": {
                    "run_dir": matched_path,
                    "configuration": {"training_seeds": [int(seed)]},
                },
                "source_provenance": source_provenance,
                "selection": run_selection,
                "training_audit": training_audit,
                "metrics": run_metrics,
                "panel": pd.DataFrame(),
                "predictions": pd.DataFrame(),
                "paired": paired.loc[
                    paired["library"].eq(library)
                    & paired["training_seed"].eq(int(seed))
                ].copy(),
            }
            verified_by_path[path] = verified
            source_runs.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "library": library,
                    "training_seed": int(seed),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_hash,
                    "source_matched_run": source_provenance,
                }
            )

    selected_identity = (
        placement.matched.LIBRARIES[0],
        int(placement.matched.DEFAULT_TRAINING_SEEDS[0]),
    )
    source_shared = next(
        record
        for record in verified_by_path.values()
        if (record["library"], int(record["training_seed"])) == selected_identity
    )

    def fake_load_verified_run(path):
        return verified_by_path[Path(path).resolve()]

    def fake_combine_verified_runs(runs):
        identities = [
            (str(run["library"]), int(run["training_seed"])) for run in runs
        ]
        expected = {
            (library, int(seed))
            for library in placement.matched.LIBRARIES
            for seed in placement.matched.DEFAULT_TRAINING_SEEDS
        }
        assert len(identities) == len(set(identities)) == len(expected)
        assert set(identities) == expected
        return {
            "selection": pd.concat(
                [run["selection"].copy() for run in runs], ignore_index=True
            ),
            "metrics": pd.concat(
                [run["metrics"].copy() for run in runs], ignore_index=True
            ),
            "configuration_contract": copy.deepcopy(configuration_contract),
            "runtime_contract": copy.deepcopy(runtime_contract),
            "common_source_hashes": copy.deepcopy(common_source_hashes),
            "library_reference_source_hashes": copy.deepcopy(
                library_reference_source_hashes
            ),
        }

    def fake_build_paired_changes(metrics):
        rows = []
        for library in placement.matched.LIBRARIES:
            for seed in placement.matched.DEFAULT_TRAINING_SEEDS:
                block = metrics.loc[
                    metrics["library"].eq(library)
                    & metrics["training_seed"].eq(int(seed))
                ].set_index("arm")
                row = {
                    "library": library,
                    "training_seed": int(seed),
                    "selected_epoch": int(block.loc["lora_cross", "selected_epoch"]),
                    "comparison": "lora_cross_minus_head_only",
                }
                for metric in placement.shared_aggregate.SUMMARY_METRICS:
                    row[metric + "_change"] = float(
                        block.loc["lora_cross", metric]
                        - block.loc["head_only", metric]
                    )
                rows.append(row)
        return pd.DataFrame(rows).sort_values(
            ["library", "training_seed"], kind="mergesort"
        ).reset_index(drop=True)

    monkeypatch.setattr(
        placement.shared_aggregate, "load_verified_run", fake_load_verified_run
    )
    monkeypatch.setattr(
        placement.shared_aggregate,
        "combine_verified_runs",
        fake_combine_verified_runs,
    )
    monkeypatch.setattr(
        placement.shared_aggregate,
        "build_paired_changes",
        fake_build_paired_changes,
    )
    dependencies = {
        "shared_epoch_trainer": Path(placement.shared_trainer.__file__).resolve(),
        "matched_aggregator": Path(
            placement.shared_aggregate.matched_aggregate.__file__
        ).resolve(),
        "matched_trainer": Path(placement.matched.__file__).resolve(),
        "canonical_evaluator": Path(placement.cached_eval.__file__).resolve(),
    }
    manifest = {
        "schema_version": placement.shared_aggregate.SCHEMA_VERSION,
        "analysis_status": "retrospective_exploratory",
        "shared_training_configuration": configuration_contract,
        "shared_runtime": runtime_contract,
        "common_source_hashes": common_source_hashes,
        "library_reference_source_hashes": library_reference_source_hashes,
        "outputs": outputs,
        "source_runs": source_runs,
        "code": {
            "path": str(Path(placement.shared_aggregate.__file__).resolve()),
            "sha256": placement.sha256_file(
                Path(placement.shared_aggregate.__file__).resolve()
            ),
        },
        "dependencies": {
            name: {"path": str(path), "sha256": placement.sha256_file(path)}
            for name, path in dependencies.items()
        },
    }
    (gate_dir / "manifest.json").write_text(json.dumps(manifest))
    return gate_dir, source_shared


def _rewrite_gate_manifest(gate_dir, mutate):
    manifest_path = gate_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))


def _rewrite_gate_table_and_hash(gate_dir, filename, frame):
    path = gate_dir / filename
    frame.to_csv(path, index=False)

    def update_hash(manifest):
        manifest["outputs"][filename]["sha256"] = _sha(path)

    _rewrite_gate_manifest(gate_dir, update_hash)


def test_gate_aggregate_is_hash_bound_recomputed_and_source_bound(
    tmp_path, monkeypatch
):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    verified = placement.load_verified_gate_aggregate(gate_dir, source_shared)
    assert verified["gate_row"]["library"] == "LibA"
    assert bool(verified["gate_row"]["placement_gate_pass"])
    assert len(verified["verified_source_runs"]) == 6

    predictions = gate_dir / "retention_predictions_by_seed.csv"
    predictions.write_text("x\n2\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_gate_blocks_one_failed_seed_and_unlisted_source(tmp_path, monkeypatch):
    gate_dir, source_shared = _make_gate_aggregate(
        tmp_path, monkeypatch, fail_library="LibA"
    )
    with pytest.raises(ValueError, match="failed .*all_seeds"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)

    other = dict(source_shared)
    other["manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="not bound"):
        placement.load_verified_gate_aggregate(gate_dir, other)


@pytest.mark.parametrize(
    "filename,table_label",
    [
        ("shared_epoch_selection_by_seed.csv", "selection"),
        ("paired_cross_minus_head_by_seed.csv", "paired"),
    ],
)
@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "fractional"])
def test_gate_rejects_self_rehashed_seed_table_grid_attacks(
    tmp_path, monkeypatch, filename, table_label, mutation
):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    frame = pd.read_csv(gate_dir / filename, float_precision="round_trip")
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]].copy()], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[1:].reset_index(drop=True)
    elif mutation == "extra":
        extra = frame.iloc[[0]].copy()
        extra["training_seed"] = int(frame["training_seed"].max()) + 1
        frame = pd.concat([frame, extra], ignore_index=True)
    else:
        frame.loc[0, "training_seed"] = float(frame.loc[0, "training_seed"]) + 0.5
    _rewrite_gate_table_and_hash(gate_dir, filename, frame)

    expected = (
        "{} training_seed contains a non-integral value".format(table_label)
        if mutation == "fractional"
        else "{}.*do(?:es)? not contain exactly".format(table_label)
    )
    with pytest.raises(ValueError, match=expected):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


@pytest.mark.parametrize(
    "filename,column,expected",
    [
        (
            "shared_epoch_selection_by_seed.csv",
            "lora_cross_log_loss",
            "gate selection versus verified source runs .* differs",
        ),
        (
            "paired_cross_minus_head_by_seed.csv",
            "global_auroc_change",
            "gate paired changes versus verified source runs .* differs",
        ),
    ],
)
def test_gate_rejects_self_rehashed_tables_that_differ_from_verified_sources(
    tmp_path, monkeypatch, filename, column, expected
):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    frame = pd.read_csv(gate_dir / filename, float_precision="round_trip")
    frame.loc[0, column] = float(frame.loc[0, column]) + 0.125
    _rewrite_gate_table_and_hash(gate_dir, filename, frame)

    with pytest.raises(ValueError, match=expected):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra"])
def test_gate_rejects_source_record_grid_attacks(tmp_path, monkeypatch, mutation):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)

    def mutate(manifest):
        records = manifest["source_runs"]
        if mutation == "duplicate":
            records[-1]["library"] = records[0]["library"]
            records[-1]["training_seed"] = records[0]["training_seed"]
        elif mutation == "missing":
            records.pop()
        else:
            records.append(copy.deepcopy(records[0]))

    _rewrite_gate_manifest(gate_dir, mutate)
    expected = (
        "do not contain exactly" if mutation == "duplicate" else "exactly six"
    )
    with pytest.raises(ValueError, match=expected):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_gate_source_record_path_is_bound_to_its_library_seed_identity(
    tmp_path, monkeypatch
):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)

    def reassign_path(manifest):
        source = manifest["source_runs"][0]
        target = manifest["source_runs"][1]
        for field in ("name", "path", "manifest_path", "manifest_sha256"):
            target[field] = source[field]

    _rewrite_gate_manifest(gate_dir, reassign_path)
    with pytest.raises(ValueError, match="verified library/seed identity changed"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_gate_source_record_rejects_symlink_path_identity(tmp_path, monkeypatch):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    manifest = json.loads((gate_dir / "manifest.json").read_text())
    target = Path(manifest["source_runs"][1]["path"])
    alias = tmp_path / "source-alias"
    alias.symlink_to(target, target_is_directory=True)

    def use_alias(gate_manifest):
        record = gate_manifest["source_runs"][1]
        record["name"] = alias.name
        record["path"] = str(alias)
        record["manifest_path"] = str(alias / "manifest.json")

    _rewrite_gate_manifest(gate_dir, use_alias)
    with pytest.raises(ValueError, match="path may not traverse a symlink"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_gate_rejects_fractional_raw_source_seed_alias(tmp_path, monkeypatch):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    seed = int(source_shared["training_seed"])
    source_shared["manifest"]["configuration"]["training_seeds"] = [seed + 0.5]
    with pytest.raises(ValueError, match="training_seeds contains a non-integral"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_gate_rejects_fractional_raw_source_audit_alias(tmp_path, monkeypatch):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    source_shared["training_audit"].loc[0, "schedule_total_steps"] += 0.5
    with pytest.raises(ValueError, match="audit schedule_total_steps contains a non-integral"):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


@pytest.mark.parametrize("location", ["provenance", "nested"])
def test_gate_rejects_fractional_nested_source_seed_alias(
    tmp_path, monkeypatch, location
):
    gate_dir, source_shared = _make_gate_aggregate(tmp_path, monkeypatch)
    seed = int(source_shared["training_seed"])
    if location == "provenance":
        source_shared["source_provenance"]["training_seed"] = seed + 0.5
        expected = "source provenance training_seed contains a non-integral"
    else:
        source_shared["source"]["configuration"]["training_seeds"] = [seed + 0.5]
        expected = "nested matched training_seeds contains a non-integral"
    with pytest.raises(ValueError, match=expected):
        placement.load_verified_gate_aggregate(gate_dir, source_shared)


def test_training_order_hash_is_deterministic_and_sensitive():
    frame = pd.DataFrame({"pair_uid": ["p{}".format(i) for i in range(19)]})
    first = placement.training_order_sha256(frame, 4, 123, 2)
    assert first == placement.training_order_sha256(frame, 4, 123, 2)
    assert first != placement.training_order_sha256(frame, 5, 123, 2)
    assert first != placement.training_order_sha256(frame, 4, 124, 2)
    assert first != placement.training_order_sha256(frame, 4, 123, 3)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    assert first != placement.training_order_sha256(reversed_frame, 4, 123, 2)


def test_training_order_hash_reproduces_live_loader_uid_order():
    frame = pd.DataFrame(
        {
            "pair_uid": ["p{}".format(i) for i in range(11)],
            "weak_label": [i % 2 for i in range(11)],
            "chain1_smart_hla_linker_peptide_sequence": ["ACDE"] * 11,
            "chain2_affibody_sequence": ["FGHI"] * 11,
        }
    )
    seed = 987
    epochs = 3
    live = placement.matched.make_loader(frame, 4, True, seed)
    digest = hashlib.sha256()
    for epoch in range(1, epochs + 1):
        digest.update("epoch:{}\n".format(epoch).encode("ascii"))
        for batch_number, (_, _, _, pair_uids) in enumerate(live):
            digest.update(
                "batch:{}:{}\n".format(epoch, batch_number).encode("ascii")
            )
            for pair_uid in pair_uids:
                digest.update(str(pair_uid).encode("utf-8"))
                digest.update(b"\0")
    assert digest.hexdigest() == placement.training_order_sha256(
        frame, 4, seed, epochs
    )


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttention()
        self.multimer_attn = _TinyAttention()


class _TinyMINTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer() for _ in range(33)])

    def forward(self, chains, chain_ids):
        return torch.zeros((len(chains), 2560), dtype=torch.float32, device=chains.device)


def test_placement_constructor_changes_only_adapter_location(monkeypatch):
    monkeypatch.setattr(placement, "MINTWrapper", _TinyMINTWrapper)
    monkeypatch.setattr(placement, "load_config", lambda _: {})
    models = {}
    for arm in placement.ARMS:
        torch.manual_seed(77)
        models[arm] = placement.MINTPlacementClassifier(
            "config",
            "checkpoint",
            "cpu",
            placement.PLACEMENTS[arm],
            placement.RANK,
            placement.ALPHA,
            placement.DROPOUT,
        )
    cross_names = {
        name for name, _ in models["lora_cross"].named_parameters() if "lora_" in name
    }
    within_names = {
        name for name, _ in models["lora_within"].named_parameters() if "lora_" in name
    }
    assert len(cross_names) == len(within_names) == 8
    assert all(".multimer_attn." in name for name in cross_names)
    assert all(".self_attn." in name for name in within_names)
    assert placement.shared_trainer._hash_named_arrays(
        placement.normalized_adapter_state(models["lora_cross"])
    ) == placement.shared_trainer._hash_named_arrays(
        placement.normalized_adapter_state(models["lora_within"])
    )
    assert placement._head_hash(models["lora_cross"]) == placement._head_hash(
        models["lora_within"]
    )


def test_trainable_contract_names_and_parameter_arithmetic():
    assert placement.ADAPTER_PARAMETERS == (
        len(placement.LAYERS)
        * len(placement.PROJECTIONS)
        * placement.RANK
        * (1280 + 1280)
    )
    assert placement.TOTAL_TRAINABLE_PARAMETERS == 23041
    for arm in placement.ARMS:
        names = placement.expected_trainable_names(arm)
        assert len(names) == 10
        assert {"head.weight", "head.bias"}.issubset(names)
        placement_name = placement.PLACEMENTS[arm]
        assert all(
            name.startswith("head.") or ".{}.".format(placement_name) in name
            for name in names
        )


def test_audit_only_cli_does_not_require_output_directory():
    args = placement.parse_args(
        [
            "--source-shared-run-dir",
            "source",
            "--gate-aggregate-dir",
            "gate",
            "--audit-only",
        ]
    )
    assert args.audit_only
    assert args.output_dir is None
    with pytest.raises(SystemExit):
        placement.parse_args(
            [
                "--source-shared-run-dir",
                "source",
                "--gate-aggregate-dir",
                "gate",
            ]
        )


class _FixedClassifier(object):
    def __init__(self):
        self.coef_ = np.zeros((1, 2560), dtype=np.float64)
        self.coef_[0, 0] = 0.25
        self.intercept_ = np.asarray([-0.1], dtype=np.float64)


def _install_preflight_fixture(tmp_path, monkeypatch):
    library = "LibA"
    training_seed = int(placement.matched.DEFAULT_TRAINING_SEEDS[0])
    membership_hash = "e" * 64
    contracts = copy.deepcopy(placement.matched.PRIMARY_CONTRACT)
    contracts[library].update(
        {
            "rows": 4,
            "positive": 2,
            "negative": 2,
            "membership_sha256": membership_hash,
            "selected_c": 1.0,
        }
    )
    monkeypatch.setattr(placement.matched, "PRIMARY_CONTRACT", contracts)
    primary = pd.DataFrame(
        {
            "pair_uid": ["p0", "p1", "p2", "p3"],
            "weak_label": [0, 1, 0, 1],
        }
    )
    weak_features = np.zeros((4, 2560), dtype=np.float32)
    weak_features[:, 0] = np.asarray([-2.0, 2.0, -1.0, 1.0])
    classifier = _FixedClassifier()
    head_hash = placement.shared_trainer.head_initialization_sha256(classifier)
    run_seed = placement.matched.derived_seed(training_seed, "final_refit", -1)
    selection = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "selected": [0, 1, 0, 0],
        }
    )
    audit = pd.DataFrame(
        [
            {
                "arm": arm,
                "training_seed": training_seed,
                "fold": -1,
                "rows": 4,
                "positive": 2,
                "negative": 2,
                "run_seed": run_seed,
                "selected_epoch": 1,
                "schedule_total_steps": 3,
                "lr_schedule_horizon_epochs": 3,
                "trainable_parameters": count,
                "head_initialization_sha256": head_hash,
            }
            for arm, count in (
                ("head_only", placement.HEAD_PARAMETERS),
                ("lora_cross", placement.TOTAL_TRAINABLE_PARAMETERS),
            )
        ]
    )
    configuration = {
        "library": library,
        "training_seeds": [training_seed],
        "selected_C": 1.0,
        "chosen_positive_epoch": 1,
        "lr_schedule_horizon_epochs": 3,
        "lora": {
            "rank": placement.RANK,
            "alpha": placement.ALPHA,
            "dropout": placement.DROPOUT,
            "layers_zero_based": list(placement.LAYERS),
            "projections": list(placement.PROJECTIONS),
            "placement": "multimer_attn",
        },
        **copy.deepcopy(placement.TRAINING_CONTRACT),
    }
    runtime = {
        "python": placement.sys.version,
        "platform": placement.platform.platform(),
        "numpy": placement.np.__version__,
        "pandas": placement.pd.__version__,
        "scipy": placement.scipy.__version__,
        "sklearn": placement.sklearn.__version__,
        "torch": placement.torch.__version__,
        "torch_cuda": placement.torch.version.cuda,
        "gpu_name": "test-gpu",
    }
    source_dir = (tmp_path / "selected-shared").resolve()
    matched_dir = (tmp_path / "selected-matched").resolve()
    source_dir.mkdir()
    matched_dir.mkdir()
    source_manifest_path = source_dir / "manifest.json"
    source_manifest_path.write_text("{}\n")
    matched_source = {
        "run_dir": matched_dir,
        "configuration": {"training_seeds": [training_seed]},
        "sources": {},
    }
    source_provenance = {
        "library": library,
        "training_seed": training_seed,
    }
    source_shared = {
        "run_dir": source_dir,
        "run_name": source_dir.name,
        "manifest_path": source_manifest_path,
        "manifest_sha256": _sha(source_manifest_path),
        "manifest": {"configuration": configuration},
        "library": library,
        "training_seed": training_seed,
        "configuration": configuration,
        "runtime_contract": runtime,
        "source": matched_source,
        "source_provenance": source_provenance,
        "selection": selection,
        "training_audit": audit,
    }
    gate_dir = (tmp_path / "gate").resolve()
    gate_dir.mkdir()
    gate_manifest_path = gate_dir / "manifest.json"
    gate_manifest_path.write_text("{}\n")
    gate = {
        "run_dir": gate_dir,
        "manifest_path": gate_manifest_path,
        "manifest_sha256": _sha(gate_manifest_path),
        "verified_outputs": {},
        "verified_source_runs": [source_shared],
    }
    train_args = SimpleNamespace(
        config=tmp_path / "config.json",
        checkpoint=tmp_path / "checkpoint.pt",
        cache=tmp_path / "cache.npz",
        max_live_feature_probability_difference=0.01,
        lora_rank=placement.RANK,
        lora_alpha=placement.ALPHA,
        lora_dropout=placement.DROPOUT,
        **copy.deepcopy(placement.TRAINING_CONTRACT)
    )
    inputs = {"primary": primary, "weak_features": weak_features}
    monkeypatch.setattr(
        placement.shared_aggregate, "load_verified_run", lambda _: source_shared
    )
    monkeypatch.setattr(
        placement, "load_verified_gate_aggregate", lambda *_: gate
    )
    monkeypatch.setattr(
        placement.shared_trainer, "_load_prefit_cache", lambda _: inputs
    )
    monkeypatch.setattr(
        placement.shared_trainer,
        "training_arguments_from_verified_source",
        lambda *_: train_args,
    )
    monkeypatch.setattr(placement.matched, "fit_logistic", lambda *_: classifier)
    monkeypatch.setattr(
        placement.cached_eval, "membership_sha256", lambda _: membership_hash
    )
    monkeypatch.setattr(
        placement.shared_aggregate, "_recheck_verified_inputs", lambda _: None
    )
    args = SimpleNamespace(
        source_shared_run_dir=source_dir,
        gate_aggregate_dir=gate_dir,
        output_dir=None,
        device="cpu",
        audit_only=True,
    )
    return args, source_shared, train_args, inputs


def test_production_preflight_and_run_audit_only_are_cpu_and_write_free(
    tmp_path, monkeypatch
):
    args, _, _, _ = _install_preflight_fixture(tmp_path, monkeypatch)
    publish_calls = []
    for name in (
        "is_available",
        "set_device",
        "empty_cache",
        "reset_peak_memory_stats",
        "get_device_name",
        "max_memory_allocated",
    ):
        monkeypatch.setattr(
            placement.torch.cuda,
            name,
            lambda *unused, _name=name: pytest.fail(
                "audit-only called torch.cuda.{}".format(_name)
            ),
        )
    monkeypatch.setattr(
        placement,
        "_atomic_publish_run",
        lambda *unused: publish_calls.append(True),
    )
    prefit = placement.preflight(args)
    expected_order = placement.training_order_sha256(
        prefit["inputs"]["primary"],
        placement.TRAINING_CONTRACT["batch_size"],
        prefit["run_seed"],
        1,
    )
    assert prefit["training_order_sha256"] == expected_order
    before = {
        path.relative_to(tmp_path): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    public = placement.run(args)
    after = {
        path.relative_to(tmp_path): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert public["training_order_sha256"] == expected_order
    assert publish_calls == []
    assert after == before


@pytest.mark.parametrize(
    "failure",
    [
        "runtime",
        "configuration",
        "float_configuration",
        "raw_seed",
        "raw_chain",
        "target",
    ],
)
def test_preflight_rejects_incompatible_source_before_cuda(
    tmp_path, monkeypatch, failure
):
    args, source_shared, train_args, inputs = _install_preflight_fixture(
        tmp_path, monkeypatch
    )
    if failure == "runtime":
        source_shared["runtime_contract"]["numpy"] = "different"
        expected = "current numpy differs"
    elif failure == "configuration":
        source_shared["manifest"]["configuration"]["batch_size"] = 63
        expected = "batch_size differs"
    elif failure == "float_configuration":
        source_shared["manifest"]["configuration"]["head_lr"] += 5e-13
        expected = "head_lr differs"
    elif failure == "raw_seed":
        source_shared["manifest"]["configuration"]["training_seeds"] = [
            int(source_shared["training_seed"]) + 0.5
        ]
        expected = "training_seeds contains a non-integral"
    elif failure == "raw_chain":
        identity = {
            "row_index": 0,
            "cache_uid": "c0",
            "source_kind": "weak",
            "library": "LibA",
            "pair_uid": "p0",
            "chain1_sha256": "0" * 64,
            "chain2_sha256": "1" * 64,
            "sequence_pair_sha256": "2" * 64,
        }
        cache_frame = pd.DataFrame([identity])
        row_frame = pd.DataFrame(
            [
                {
                    **identity,
                    "chain1_smart_hla_linker_peptide_sequence": "AAAA",
                    "chain2_affibody_sequence": "CCCC",
                }
            ]
        )

        def load_tampered_cache(unused):
            placement.matched.attach_full_sequences(cache_frame, row_frame)
            pytest.fail("tampered cache unexpectedly passed identity validation")

        monkeypatch.setattr(
            placement.shared_trainer, "_load_prefit_cache", load_tampered_cache
        )
        expected = "chain-1 sequence hash mismatch"
    else:
        inputs["primary"] = inputs["primary"].assign(target_retention=75.0)
        expected = "contains retention outcome columns"
    monkeypatch.setattr(
        placement.torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA was queried after a failed preflight"),
    )
    with pytest.raises(ValueError, match=expected):
        placement.run(args)


class _ExactTinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(placement.RANK, 1280) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(1280, placement.RANK))

    def scalar(self):
        return torch.sum(self.lora_a[:, 0] * self.lora_b[0, :])


class _ExactTinyAttention(nn.Module):
    def __init__(self, active):
        super().__init__()
        self.q_proj = _ExactTinyAdapter() if active else nn.Identity()
        self.v_proj = _ExactTinyAdapter() if active else nn.Identity()


class _ExactTinyLayer(nn.Module):
    def __init__(self, active_placement=None):
        super().__init__()
        self.self_attn = _ExactTinyAttention(active_placement == "self_attn")
        self.multimer_attn = _ExactTinyAttention(
            active_placement == "multimer_attn"
        )


class _ExactTinyPlacementClassifier(nn.Module):
    def __init__(self, config, checkpoint, device, placement_name, rank, alpha, dropout):
        super().__init__()
        del config, checkpoint, device, alpha, dropout
        assert int(rank) == placement.RANK
        self.placement = placement_name
        self.wrapper = nn.Module()
        self.wrapper.model = nn.Module()
        self.wrapper.model.layers = nn.ModuleList(
            [
                _ExactTinyLayer(
                    placement_name if index in placement.LAYERS else None
                )
                for index in range(33)
            ]
        )
        self.adapters = []
        for index in placement.LAYERS:
            attention = getattr(self.wrapper.model.layers[index], placement_name)
            self.adapters.extend([attention.q_proj, attention.v_proj])
        self.head = nn.Linear(2560, 1)
        self.register_buffer("feature_mean", torch.zeros(2560))
        self.register_buffer("feature_scale", torch.ones(2560))

    def forward(self, chains, chain_ids):
        del chain_ids
        effect = sum(adapter.scalar() for adapter in self.adapters)
        first = chains[:, 0].float() + effect
        features = torch.cat(
            [
                first[:, None],
                torch.zeros(
                    (len(first), 2559), dtype=first.dtype, device=first.device
                ),
            ],
            dim=1,
        )
        features = (features - self.feature_mean) / self.feature_scale
        return self.head(features).flatten()

    def set_feature_standardization(self, mean, scale):
        with torch.no_grad():
            self.feature_mean.copy_(torch.from_numpy(np.asarray(mean, dtype=np.float32)))
            self.feature_scale.copy_(
                torch.from_numpy(np.asarray(scale, dtype=np.float32))
            )

    def set_train_mode(self):
        self.wrapper.model.eval()
        self.head.train()
        for adapter in self.adapters:
            adapter.train()

    def adapter_parameters(self):
        return [
            parameter
            for adapter in self.adapters
            for parameter in adapter.parameters()
        ]


class _FrameLoader(object):
    def __init__(self, frame, batch_size, shuffle, seed):
        self.frame = frame.reset_index(drop=True)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        self.loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.arange(len(frame))),
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=0,
            generator=generator,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for (indices,) in self.loader:
            rows = self.frame.iloc[indices.tolist()]
            chains = torch.from_numpy(
                rows["signal"].to_numpy(dtype=np.float32)[:, None]
            )
            chain_ids = torch.zeros_like(chains, dtype=torch.long)
            weak_target = torch.from_numpy(
                rows["weak_label"].to_numpy(dtype=np.float32)
            )
            yield chains, chain_ids, weak_target, rows["pair_uid"].astype(str).tolist()


def _tiny_training_args():
    return SimpleNamespace(
        config="config",
        checkpoint="checkpoint",
        lora_rank=placement.RANK,
        lora_alpha=placement.ALPHA,
        lora_dropout=placement.DROPOUT,
        batch_size=2,
        eval_batch_size=2,
        accumulation_steps=1,
        max_epochs=3,
        head_lr=0.01,
        adapter_lr=0.02,
        weight_decay=0.01,
        warmup_fraction=0.1,
        clip_norm=1.0,
        max_live_feature_probability_difference=1e-8,
    )


def test_cpu_production_pairing_training_init_optimizer_and_order(monkeypatch):
    monkeypatch.setattr(
        placement, "MINTPlacementClassifier", _ExactTinyPlacementClassifier
    )
    monkeypatch.setattr(
        placement.matched,
        "make_loader",
        lambda frame, batch_size, shuffle, seed: _FrameLoader(
            frame, batch_size, shuffle, seed
        ),
    )
    classifier = _FixedClassifier()
    args = _tiny_training_args()
    training_seed = int(placement.matched.DEFAULT_TRAINING_SEEDS[0])
    run_seed = placement.matched.derived_seed(training_seed, "final_refit", -1)
    mean = np.zeros(2560, dtype=np.float32)
    scale = np.ones(2560, dtype=np.float32)
    train = pd.DataFrame(
        {
            "pair_uid": ["t0", "t1", "t2", "t3"],
            "weak_label": [0, 1, 0, 1],
            "signal": [-2.0, 2.0, -1.0, 1.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "pair_uid": ["e0", "e1", "e2", "e3"],
            "weak_label": [0, 1, 0, 1],
            "signal": [-1.5, 1.5, -0.5, 0.5],
        }
    )
    placement.matched.set_deterministic_seed(run_seed)
    cross = _ExactTinyPlacementClassifier(
        "config",
        "checkpoint",
        "cpu",
        "multimer_attn",
        placement.RANK,
        placement.ALPHA,
        placement.DROPOUT,
    )
    placement.matched.load_head(cross, classifier)
    source_adapter_hash = placement.shared_trainer._hash_named_arrays(
        placement.raw_adapter_state(cross)
    )
    expected_head_hash = placement._head_hash(cross)
    epoch0 = placement.audit_epoch0_placement_pair(
        evaluation,
        mean,
        scale,
        classifier,
        training_seed,
        args,
        torch.device("cpu"),
        source_adapter_hash,
        expected_head_hash,
    )
    assert epoch0["max_abs_probability_difference"] == 0.0
    expected_probability = 1.0 / (
        1.0
        + np.exp(
            -(
                0.25 * evaluation["signal"].to_numpy(dtype=float)
                - 0.1
            )
        )
    )
    within = placement.train_within_arm(
        train,
        evaluation,
        mean,
        scale,
        classifier,
        training_seed,
        2,
        args,
        torch.device("cpu"),
        expected_probability,
    )
    assert within["head_initialization_sha256"] == expected_head_hash
    assert (
        within["normalized_adapter_initialization_sha256"]
        == epoch0["normalized_adapter_initialization_sha256"]
    )
    assert within["optimizer"] == placement.OPTIMIZER_CONTRACT
    assert within["training_order_sha256"] == placement.training_order_sha256(
        train, args.batch_size, run_seed, 2
    )
    assert within["head_parameter_delta"]["l2"] > 0.0
    assert within["adapter_parameter_delta"]["l2"] > 0.0


@pytest.mark.parametrize("frame_name", ["train", "evaluation"])
@pytest.mark.parametrize("column", ["target_retention", "target_binder"])
def test_train_within_rejects_outcomes_before_optimizer(
    monkeypatch, frame_name, column
):
    train = pd.DataFrame({"pair_uid": ["t"], "weak_label": [1]})
    evaluation = pd.DataFrame({"pair_uid": ["e"], "weak_label": [0]})
    if frame_name == "train":
        train[column] = 1
    else:
        evaluation[column] = 1
    optimizer_called = []
    monkeypatch.setattr(
        placement,
        "optimizer_for_placement",
        lambda *unused: optimizer_called.append(True),
    )
    with pytest.raises(ValueError, match="contains retention outcome columns"):
        placement.train_within_arm(
            train,
            evaluation,
            None,
            None,
            None,
            1,
            1,
            SimpleNamespace(),
            torch.device("cpu"),
            np.zeros(1),
        )
    assert optimizer_called == []


def test_optimizer_audit_rejects_param_group_adamw_override():
    model = _ExactTinyPlacementClassifier(
        "config",
        "checkpoint",
        "cpu",
        "self_attn",
        placement.RANK,
        placement.ALPHA,
        placement.DROPOUT,
    )
    args = _tiny_training_args()
    optimizer = placement.optimizer_for_placement(model, args)
    assert placement.audit_optimizer_for_placement(optimizer, model, args) == (
        placement.OPTIMIZER_CONTRACT
    )
    optimizer.param_groups[0]["betas"] = (0.5, 0.5)
    with pytest.raises(ValueError, match="AdamW settings changed"):
        placement.audit_optimizer_for_placement(optimizer, model, args)


@pytest.mark.parametrize(
    "name,value",
    [
        ("amsgrad", True),
        ("maximize", True),
        ("capturable", True),
        ("foreach", True),
    ],
)
def test_optimizer_audit_rejects_execution_flag_overrides(name, value):
    model = _ExactTinyPlacementClassifier(
        "config",
        "checkpoint",
        "cpu",
        "self_attn",
        placement.RANK,
        placement.ALPHA,
        placement.DROPOUT,
    )
    args = _tiny_training_args()
    optimizer = placement.optimizer_for_placement(model, args)
    optimizer.param_groups[0][name] = value
    with pytest.raises(ValueError, match="execution flags changed"):
        placement.audit_optimizer_for_placement(optimizer, model, args)


def test_atomic_run_publisher_is_private_hash_bound_and_manifest_last(
    tmp_path, monkeypatch
):
    output = tmp_path / "published"
    link_order = []
    original_link = placement.shared_aggregate._link_staged_file_noreplace

    def recording_link(source, filename, output_fd):
        link_order.append(filename)
        return original_link(source, filename, output_fd)

    monkeypatch.setattr(
        placement.shared_aggregate,
        "_link_staged_file_noreplace",
        recording_link,
    )
    artifacts = {
        "table.csv": pd.DataFrame({"value": [1, 2]}),
        "model.pt": placement._TorchSaveArtifact({"weight": torch.ones(2)}),
    }

    def build_manifest(records):
        return {"schema_version": "test-v1", "outputs": records}

    manifest = placement._atomic_publish_run(
        output, artifacts, "run_summary.md", "summary\n", build_manifest
    )
    assert link_order[-1] == "manifest.json"
    assert set(path.name for path in output.iterdir()) == {
        "table.csv",
        "model.pt",
        "run_summary.md",
        "manifest.json",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for name, record in manifest["outputs"].items():
        assert Path(record["path"]) == (output / name).resolve()
        assert record["sha256"] == _sha(output / name)
    assert torch.equal(torch.load(output / "model.pt")["weight"], torch.ones(2))
    with pytest.raises(ValueError, match="exists; refusing overwrite"):
        placement._atomic_publish_run(
            output, artifacts, "run_summary.md", "summary\n", build_manifest
        )


def test_atomic_run_publisher_cleans_precommit_failures(tmp_path, monkeypatch):
    output = tmp_path / "failed"

    def fail_builder(records):
        del records
        raise RuntimeError("manifest builder failed")

    with pytest.raises(RuntimeError, match="manifest builder failed"):
        placement._atomic_publish_run(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "run_summary.md",
            "summary\n",
            fail_builder,
        )
    assert not os.path.lexists(str(output))
    assert list(tmp_path.glob(".failed-staging-*")) == []

    original_open = placement.os.open

    def fail_claim_open(path, flags, *args, **kwargs):
        if Path(str(path)) == output:
            raise OSError("injected output open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(placement.os, "open", fail_claim_open)
    with pytest.raises(OSError, match="injected output open failure"):
        placement._atomic_publish_run(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "run_summary.md",
            "summary\n",
            lambda records: {"outputs": records},
        )
    assert not os.path.lexists(str(output))
    assert list(tmp_path.glob(".failed-staging-*")) == []


def test_atomic_run_publisher_detects_directory_swap_without_deleting_racer(
    tmp_path, monkeypatch
):
    output = tmp_path / "swapped"
    displaced = tmp_path / "displaced"
    original_open = placement.os.open
    swapped = []

    def swap_before_open(path, flags, *args, **kwargs):
        if Path(str(path)) == output and not swapped:
            output.rename(displaced)
            output.mkdir()
            (output / "racer.txt").write_text("racer\n")
            swapped.append(True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(placement.os, "open", swap_before_open)
    with pytest.raises(ValueError, match="replaced before open"):
        placement._atomic_publish_run(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "run_summary.md",
            "summary\n",
            lambda records: {"outputs": records},
        )
    assert (output / "racer.txt").read_text() == "racer\n"
    assert displaced.is_dir()


def test_atomic_run_publisher_precommit_failure_leaves_no_committed_run(tmp_path):
    output = tmp_path / "precommit-failed"
    callbacks = []

    def fail_precommit():
        callbacks.append(True)
        raise RuntimeError("input changed before commit")

    with pytest.raises(RuntimeError, match="input changed before commit"):
        placement._atomic_publish_run(
            output,
            {"table.csv": pd.DataFrame({"value": [1]})},
            "run_summary.md",
            "summary\n",
            lambda records: {"outputs": records},
            precommit=fail_precommit,
        )
    assert callbacks == [True]
    assert not os.path.lexists(str(output))
    assert list(tmp_path.glob(".precommit-failed-staging-*")) == []


def test_non_audit_run_rejects_dangling_output_before_preflight(
    tmp_path, monkeypatch
):
    output = tmp_path / "dangling"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    preflight_calls = []
    monkeypatch.setattr(
        placement, "preflight", lambda _: preflight_calls.append(True)
    )
    args = SimpleNamespace(output_dir=output, audit_only=False)
    with pytest.raises(ValueError, match="exists; refusing overwrite"):
        placement.run(args)
    assert preflight_calls == []


def test_cache_inventory_parent_is_protected_from_output_overlap(tmp_path):
    cache_dir = (tmp_path / "cache-shards").resolve()
    cache_dir.mkdir()
    cache_path = cache_dir / "features-shard-000.npz"
    cache_path.write_bytes(b"cache")
    matched_dir = (tmp_path / "matched").resolve()
    shared_dir = (tmp_path / "shared").resolve()
    gate_dir = (tmp_path / "gate").resolve()
    reference_dir = (tmp_path / "reference").resolve()
    for path in (matched_dir, shared_dir, gate_dir, reference_dir):
        path.mkdir()
    run = {
        "run_dir": shared_dir,
        "source": {
            "run_dir": matched_dir,
            "reference": {"directory": str(reference_dir)},
            "sources": {
                "cache_npz_000": {"path": str(cache_path)},
                "cache_rows": {"path": str(cache_dir / "rows.csv")},
            },
        },
    }
    prefit = {
        "gate": {"run_dir": gate_dir, "verified_source_runs": [run]},
        "source_shared": run,
        "train_args": SimpleNamespace(cache=cache_dir),
    }
    protected = placement._placement_input_artifact_paths(prefit)
    assert cache_dir in protected
    with pytest.raises(ValueError, match="must be disjoint"):
        placement.shared_aggregate._assert_output_disjoint(
            cache_dir / "poison", protected
        )


def test_atomic_run_publisher_on_real_lustre_root():
    root = Path(
        os.environ.get(
            "MINT_LUSTRE_TEST_ROOT",
            str(placement.REPO_ROOT / "private_data"),
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    filesystem = subprocess.run(
        ["stat", "-f", "-c", "%T", str(root)],
        check=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ).stdout.strip()
    if filesystem != "lustre":
        pytest.skip("configured publication root is not Lustre")
    scratch = Path(
        tempfile.mkdtemp(prefix="placement-control-publish-test-", dir=str(root))
    )
    output = scratch / "artifact"
    try:
        manifest = placement._atomic_publish_run(
            output,
            {"table.csv": pd.DataFrame({"value": [1, 2]})},
            "run_summary.md",
            "summary\n",
            lambda records: {"schema_version": "test-v1", "outputs": records},
        )
        before = {
            path.name: (path.stat().st_ino, _sha(path))
            for path in output.iterdir()
        }
        assert manifest["outputs"]["table.csv"]["sha256"] == _sha(
            output / "table.csv"
        )
        with pytest.raises(ValueError, match="exists; refusing overwrite"):
            placement._atomic_publish_run(
                output,
                {"table.csv": pd.DataFrame({"value": [9]})},
                "run_summary.md",
                "changed\n",
                lambda records: {"outputs": records},
            )
        after = {
            path.name: (path.stat().st_ino, _sha(path))
            for path in output.iterdir()
        }
        assert after == before
    finally:
        shutil.rmtree(str(scratch))
