import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from downstream.AffibodyMHC import orchestrate_mint_selection_confirmatory as orch


SCRIPT_NAMES = (
    "finetune_mint_selection_matched.py",
    "aggregate_mint_selection_matched.py",
    "finetune_mint_selection_shared_epoch.py",
    "aggregate_mint_selection_shared_epoch.py",
    "finetune_mint_selection_placement_control.py",
    "aggregate_mint_selection_placement_control.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_contract(tmp_path):
    root = tmp_path / "repo"
    code = root / "downstream" / "AffibodyMHC"
    code.mkdir(parents=True)
    for name in SCRIPT_NAMES:
        path = code / name
        path.write_text("# fake {}\n".format(name))
        path.chmod(0o700)
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())
    (root / "private_data" / "experiments").mkdir(parents=True)
    contract = orch.StudyContract(root)
    contract.validate_static()
    return contract


def write_job_manifest(job, seed_value=None):
    job.output_dir.mkdir(parents=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    if not job.log_path.exists():
        job.log_path.write_text("worker log\n")
    artifact = job.output_dir / "artifact.txt"
    artifact.write_text("complete\n")
    if job.stage in ("matched", "shared"):
        configuration = {
            "library": job.library,
            "training_seeds": [job.seed if seed_value is None else seed_value],
        }
        status = "retrospective_exploratory"
        if job.stage == "shared":
            configuration["chosen_positive_epoch"] = 1
    else:
        configuration = {
            "library": job.library,
            "training_seed": job.seed if seed_value is None else seed_value,
        }
        status = "retrospective_exploratory_retention_gated"
        configuration["shared_positive_epoch"] = 1
    (job.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": job.schema,
                "analysis_status": status,
                "configuration": configuration,
                "outputs": {
                    artifact.name: {
                        "path": str(artifact.resolve()),
                        "sha256": sha256(artifact),
                    }
                },
            }
        )
    )


def write_exit_zero(job):
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.write_text("finished\n")
    job.exit_path.write_text("0\n")


def gate_rows(passed=()):
    rows = []
    for library in orch.LIBRARIES:
        value = library in passed
        rows.append(
            {
                "library": library,
                "n_seeds": "3",
                "cross_log_loss_better_than_epoch0_all_seeds": str(value),
                "cross_log_loss_better_than_head_all_seeds": str(value),
                "within_peptide_spearman_better_all_seeds": str(value),
                "placement_gate_pass": str(value),
                "interpretation": "test",
            }
        )
    return rows


def write_aggregate(output_dir, schema, run_dirs, passed=()):
    output_dir.mkdir(parents=True)
    outputs = {}
    for name in sorted(orch.AGGREGATE_OUTPUTS[schema]):
        output = output_dir / name
        if name == "within_chain_placement_gate.csv":
            rows = gate_rows(passed)
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            output.write_text("fake audited aggregate {}\n".format(name))
        outputs[name] = {
            "path": str(output.resolve()),
            "sha256": sha256(output),
        }
    source_runs = []
    for path in run_dirs:
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            path.mkdir(parents=True)
            artifact = path / "artifact.txt"
            artifact.write_text("source\n")
            manifest_path.write_text(
                json.dumps(
                    {
                        "outputs": {
                            artifact.name: {
                                "path": str(artifact.resolve()),
                                "sha256": sha256(artifact),
                            }
                        }
                    }
                )
            )
        source_runs.append(
            {
                "path": str(path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
            }
        )
    manifest = {
        "schema_version": schema,
        "source_runs": source_runs,
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest))


def pin_aggregate(instance, output_dir):
    instance.created_aggregate_hashes[Path(output_dir).resolve()] = sha256(
        Path(output_dir) / "manifest.json"
    )


class FakeWorkers:
    def __init__(self):
        self.values = {}

    def matching_pids(self, script, output_dir):
        return tuple(self.values.get((str(script), str(output_dir)), ()))

    def set(self, job, pids):
        self.values[(str(job.trainer), str(job.output_dir))] = tuple(pids)


class FakeRunner:
    def __init__(self, aggregate_passed=()):
        self.calls = []
        self.sessions = set()
        self.aggregate_passed = set(aggregate_passed)
        self.fail_substring = None
        self.gpu_compute_pids = {}

    def run(self, argv, env=None):
        argv = [str(value) for value in argv]
        self.calls.append((argv, None if env is None else dict(env)))
        joined = " ".join(argv)
        if self.fail_substring and self.fail_substring in joined:
            return SimpleNamespace(returncode=7, stdout="", stderr="fake failure")
        if argv[:2] == ["tmux", "list-sessions"]:
            output = "".join(name + "\n" for name in sorted(self.sessions))
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        if argv and argv[0] == "nvidia-smi":
            gpu_option = next(
                (value for value in argv if value.startswith("--id=")), None
            )
            if gpu_option is None:
                return SimpleNamespace(
                    returncode=0,
                    stdout="\n".join(str(value) for value in range(8)) + "\n",
                    stderr="",
                )
            gpu = int(gpu_option.split("=", 1)[1])
            values = self.gpu_compute_pids.get(gpu, ())
            return SimpleNamespace(
                returncode=0,
                stdout="".join(str(value) + "\n" for value in values),
                stderr="",
            )
        if argv[:2] == ["tmux", "new-session"]:
            session = argv[argv.index("-s") + 1]
            if session in self.sessions:
                return SimpleNamespace(returncode=1, stdout="", stderr="duplicate")
            self.sessions.add(session)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--audit-only" in argv:
            return SimpleNamespace(returncode=0, stdout="audit ok", stderr="")
        if "--run-dirs" in argv:
            start = argv.index("--run-dirs") + 1
            stop = argv.index("--output-dir")
            run_dirs = [Path(value) for value in argv[start:stop]]
            output_dir = Path(argv[stop + 1])
            script = Path(argv[1]).name
            schemas = {
                "aggregate_mint_selection_matched.py": orch.MATCHED_AGGREGATE_SCHEMA,
                "aggregate_mint_selection_shared_epoch.py": orch.SHARED_AGGREGATE_SCHEMA,
                "aggregate_mint_selection_placement_control.py": orch.PLACEMENT_AGGREGATE_SCHEMA,
            }
            write_aggregate(
                output_dir,
                schemas[script],
                run_dirs,
                passed=self.aggregate_passed,
            )
            return SimpleNamespace(returncode=0, stdout="aggregate ok", stderr="")
        raise AssertionError("unexpected fake command: {}".format(argv))


def ready_orchestrator(contract, runner=None, workers=None):
    instance = orch.ConfirmatoryOrchestrator(
        contract,
        runner=runner or FakeRunner(),
        workers=workers or FakeWorkers(),
        journal=None,
        sleeper=lambda _: None,
    )
    instance.initial_hashes = instance._dependency_hashes()
    return instance


def test_contract_matrix_and_dry_run_are_exact_and_nonmutating(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    instance = ready_orchestrator(contract, runner=runner)
    plan = instance.dry_run_plan()

    assert runner.calls == []
    for stage in ("matched", "shared", "placement"):
        rows = plan["stages"][stage]
        assert [(row["library"], row["training_seed"]) for row in rows] == list(
            orch.EXPECTED_GRID
        )
        assert [row["gpu"] for row in rows] == list(range(6))
    assert plan["stages"]["matched"][0]["session"] == "mint_conf_liba_11"
    assert plan["stages"]["shared"][0]["session"] == "mint_conf_shared_liba_11"
    assert plan["stages"]["placement"][5]["session"] == "mint_conf_place_libb_13"
    assert "CUDA_VISIBLE_DEVICES=5" in plan["stages"]["shared"][5]["tmux_argv"][-1]
    prefixes = {
        "matched": ("mint_selection_matched_confirmatory", "mint_conf"),
        "shared": ("mint_selection_shared_epoch_confirmatory", "mint_conf_shared"),
        "placement": (
            "mint_selection_placement_control_confirmatory",
            "mint_conf_place",
        ),
    }
    for stage, jobs in (
        ("matched", contract.matched_jobs),
        ("shared", contract.shared_jobs),
        ("placement", contract.placement_jobs),
    ):
        output_prefix, session_prefix = prefixes[stage]
        for job in jobs:
            lower = job.library.lower()
            assert job.output_dir.name == "{}_{}_seed{}_v1".format(
                output_prefix, lower, job.seed
            )
            assert job.log_path.name == "{}_seed{}.log".format(lower, job.seed)
            assert job.session == "{}_{}_{}".format(
                session_prefix, lower, str(job.seed)[-2:]
            )
            assert job.gpu == orch.GPU_BY_RUN[job.key]
            if stage == "shared":
                assert job.source_dir == contract.matched_jobs[
                    list(orch.EXPECTED_GRID).index(job.key)
                ].output_dir
            elif stage == "placement":
                assert job.source_dir == contract.shared_jobs[
                    list(orch.EXPECTED_GRID).index(job.key)
                ].output_dir
    assert not contract.state_dir.exists()


def test_production_host_guard_rejects_the_login_node(monkeypatch):
    monkeypatch.setattr(
        orch.os,
        "uname",
        lambda: SimpleNamespace(nodename="not-the-prescribed-gpu-node"),
    )
    with pytest.raises(orch.OrchestrationError, match="must run on"):
        orch.ConfirmatoryOrchestrator._validate_runtime_host()


def test_manifest_is_not_complete_until_exact_worker_exits(tmp_path):
    contract = make_contract(tmp_path)
    workers = FakeWorkers()
    instance = ready_orchestrator(contract, workers=workers)
    job = contract.matched_jobs[0]
    write_job_manifest(job)
    workers.set(job, [123])
    observed = instance._observe_job(job, {job.session}, allow_not_started=False)
    assert observed["state"] == "worker_running_manifest_present"
    workers.set(job, [])
    observed = instance._observe_job(job, set(), allow_not_started=False)
    assert observed["state"] == "complete"


def test_missing_or_fractional_manifest_fails_closed(tmp_path):
    contract = make_contract(tmp_path)
    instance = ready_orchestrator(contract)
    job = contract.matched_jobs[0]
    with pytest.raises(orch.OrchestrationError, match="no worker and no manifest"):
        instance._observe_job(job, set(), allow_not_started=False)

    write_job_manifest(job, seed_value=float(job.seed))
    with pytest.raises(orch.OrchestrationError, match="seed mismatch"):
        instance._observe_job(job, set(), allow_not_started=False)


def test_new_job_requires_manifest_zero_exit_and_no_worker(tmp_path):
    contract = make_contract(tmp_path)
    instance = ready_orchestrator(contract)
    job = contract.shared_jobs[0]
    write_job_manifest(job)
    observed = instance._observe_job(job, {job.session}, allow_not_started=True)
    assert observed["state"] == "session_finishing"
    with pytest.raises(orch.OrchestrationError, match="no recorded zero exit"):
        instance._observe_job(job, set(), allow_not_started=True)
    write_exit_zero(job)
    observed = instance._observe_job(job, set(), allow_not_started=True)
    assert observed["state"] == "complete"


def test_launch_command_is_gpu_scoped_and_refuses_any_existing_path(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    instance = ready_orchestrator(contract, runner=runner)
    job = contract.shared_jobs[3]
    job.log_path.parent.mkdir(parents=True)
    job.log_path.write_text("do not overwrite")
    with pytest.raises(orch.OrchestrationError, match="existing log"):
        instance._launch_job(job, set())
    assert not any(call[0][:2] == ["tmux", "new-session"] for call in runner.calls)

    job.log_path.unlink()
    instance._launch_job(job, set())
    launch = [call[0] for call in runner.calls if call[0][:2] == ["tmux", "new-session"]]
    assert len(launch) == 1
    shell = launch[0][-1]
    assert "CUDA_VISIBLE_DEVICES=3" in shell
    assert str(job.source_dir) in shell
    assert str(job.output_dir) in shell
    assert str(job.log_path) in shell
    assert str(job.exit_path) in shell
    assert str(job.pending_exit_path) in shell
    assert "if ! ln" in shell


def test_launch_refuses_assigned_gpu_with_unrelated_compute_process(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    job = contract.shared_jobs[2]
    runner.gpu_compute_pids[job.gpu] = (99881,)
    instance = ready_orchestrator(contract, runner=runner)
    with pytest.raises(orch.OrchestrationError, match="acquired compute"):
        instance._launch_job(job, set())
    assert not any(
        argv[:2] == ["tmux", "new-session"] for argv, _ in runner.calls
    )


def test_unpublished_exit_status_is_never_read_as_complete(tmp_path):
    contract = make_contract(tmp_path)
    instance = ready_orchestrator(contract)
    job = contract.shared_jobs[0]
    write_job_manifest(job)
    job.log_path.write_text("done\n")
    job.pending_exit_path.write_text("")
    observed = instance._observe_job(job, {job.session}, allow_not_started=True)
    assert observed["state"] == "session_finishing"
    with pytest.raises(orch.OrchestrationError, match="unpublished exit"):
        instance._observe_job(job, set(), allow_not_started=True)


def test_proc_inspector_matches_exact_script_and_output_only(tmp_path):
    proc = tmp_path / "proc"
    cwd = tmp_path / "work"
    cwd.mkdir()
    script = cwd / "trainer.py"
    script.write_text("pass\n")
    output = cwd / "result"
    for pid, requested in (("101", "result"), ("102", "result_extra")):
        entry = proc / pid
        entry.mkdir(parents=True)
        (entry / "cwd").symlink_to(cwd)
        (entry / "cmdline").write_bytes(
            b"python\0trainer.py\0--output-dir\0" + requested.encode() + b"\0"
        )
    inspector = orch.ProcWorkerInspector(proc)
    assert inspector.matching_pids(script, output) == (101,)


def test_proc_job_audit_binds_source_gpu_device_and_log(tmp_path):
    contract = make_contract(tmp_path)
    job = contract.shared_jobs[0]
    proc = tmp_path / "proc"
    entry = proc / "201"
    (entry / "fd").mkdir(parents=True)
    (entry / "cwd").symlink_to(contract.repo_root)
    (entry / "fd" / "1").symlink_to(job.log_path)
    (entry / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0\0")
    tokens = [
        str(contract.python),
        str(job.trainer),
        "--source-run-dir",
        str(job.source_dir),
        "--output-dir",
        str(job.output_dir),
        "--device",
        "cuda:0",
    ]
    (entry / "cmdline").write_bytes(
        b"\0".join(value.encode() for value in tokens) + b"\0"
    )
    inspector = orch.ProcWorkerInspector(proc)
    assert inspector.matching_job_pids(job, contract.shared_aggregate) == (201,)
    (entry / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=5\0")
    with pytest.raises(orch.OrchestrationError, match="wrong CUDA"):
        inspector.matching_job_pids(job, contract.shared_aggregate)


def test_proc_job_audit_rejects_other_script_claiming_exact_output(tmp_path):
    contract = make_contract(tmp_path)
    job = contract.shared_jobs[0]
    proc = tmp_path / "proc"
    entry = proc / "301"
    (entry / "fd").mkdir(parents=True)
    (entry / "cwd").symlink_to(contract.repo_root)
    (entry / "fd" / "1").symlink_to(job.log_path)
    (entry / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0\0")
    tokens = [
        str(contract.python),
        str(contract.repo_root / "different_trainer.py"),
        "--output-dir",
        str(job.output_dir),
        "--device",
        "cuda:0",
    ]
    (entry / "cmdline").write_bytes(
        b"\0".join(value.encode() for value in tokens) + b"\0"
    )
    inspector = orch.ProcWorkerInspector(proc)
    with pytest.raises(orch.OrchestrationError, match="unexpected trainer"):
        inspector.matching_job_pids(job, contract.shared_aggregate)


def test_aggregate_command_uses_exact_grid_and_cpu_environment(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    instance = ready_orchestrator(contract, runner=runner)
    run_dirs = [job.output_dir for job in contract.matched_jobs]
    instance._ensure_aggregate(
        "matched",
        contract.matched_aggregator,
        run_dirs,
        contract.matched_aggregate,
        orch.MATCHED_AGGREGATE_SCHEMA,
    )
    argv, environment = runner.calls[-1]
    start = argv.index("--run-dirs") + 1
    stop = argv.index("--output-dir")
    assert argv[start:stop] == [str(path) for path in run_dirs]
    assert argv[stop + 1] == str(contract.matched_aggregate)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""

    calls = len(runner.calls)
    instance._ensure_aggregate(
        "matched",
        contract.matched_aggregator,
        run_dirs,
        contract.matched_aggregate,
        orch.MATCHED_AGGREGATE_SCHEMA,
    )
    assert len(runner.calls) == calls


def test_partial_aggregate_is_never_overwritten(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    instance = ready_orchestrator(contract, runner=runner)
    contract.matched_aggregate.mkdir()
    with pytest.raises(orch.OrchestrationError, match="refusing unaudited reuse"):
        instance._ensure_aggregate(
            "matched",
            contract.matched_aggregator,
            [job.output_dir for job in contract.matched_jobs],
            contract.matched_aggregate,
            orch.MATCHED_AGGREGATE_SCHEMA,
        )
    assert runner.calls == []


def test_failed_full_aggregator_blocks_all_downstream_launches(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner()
    runner.fail_substring = "aggregate_mint_selection_matched.py"
    instance = ready_orchestrator(contract, runner=runner)
    for job in contract.matched_jobs:
        write_job_manifest(job)
    with pytest.raises(orch.OrchestrationError, match="matched aggregator failed"):
        instance.step()
    assert not any(
        argv[:2] == ["tmux", "new-session"] for argv, _ in runner.calls
    )


def test_gate_requires_all_strict_fields_and_hash_binding(tmp_path):
    contract = make_contract(tmp_path)
    instance = ready_orchestrator(contract)
    write_aggregate(
        contract.shared_aggregate,
        orch.SHARED_AGGREGATE_SCHEMA,
        [job.output_dir for job in contract.shared_jobs],
        passed={"LibA"},
    )
    pin_aggregate(instance, contract.shared_aggregate)
    assert instance._read_gate() == {"LibA"}

    gate = contract.shared_aggregate / "within_chain_placement_gate.csv"
    gate.write_text(gate.read_text().replace("True,True,True,True", "True,False,True,True"))
    with pytest.raises(orch.OrchestrationError, match="hash mismatch"):
        instance._read_gate()


def test_gate_read_rechecks_source_outputs_even_when_no_library_passes(tmp_path):
    contract = make_contract(tmp_path)
    instance = ready_orchestrator(contract)
    write_aggregate(
        contract.shared_aggregate,
        orch.SHARED_AGGREGATE_SCHEMA,
        [job.output_dir for job in contract.shared_jobs],
        passed=set(),
    )
    pin_aggregate(instance, contract.shared_aggregate)
    source_artifact = contract.shared_jobs[0].output_dir / "artifact.txt"
    source_artifact.write_text("changed after gate computation\n")
    with pytest.raises(orch.OrchestrationError, match="source output changed"):
        instance._read_gate()


def test_only_passing_library_is_audited_and_launched(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner(aggregate_passed={"LibA"})
    instance = ready_orchestrator(contract, runner=runner)
    for job in contract.matched_jobs:
        write_job_manifest(job)
    for job in contract.shared_jobs:
        write_job_manifest(job)
        write_exit_zero(job)
    assert instance.step() == "waiting_placement"
    audits = [argv for argv, _ in runner.calls if "--audit-only" in argv]
    launches = [argv for argv, _ in runner.calls if argv[:2] == ["tmux", "new-session"]]
    assert len(audits) == 3
    assert all("liba" in " ".join(argv) for argv in audits)
    assert len(launches) == 3
    assert {argv[argv.index("-s") + 1] for argv in launches} == {
        "mint_conf_place_liba_11",
        "mint_conf_place_liba_12",
        "mint_conf_place_liba_13",
    }


def test_zero_pass_gate_finishes_without_any_placement_command(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner(aggregate_passed=set())
    instance = ready_orchestrator(contract, runner=runner)
    for job in contract.matched_jobs:
        write_job_manifest(job)
    for job in contract.shared_jobs:
        write_job_manifest(job)
        write_exit_zero(job)
    assert instance.step() == "complete"
    assert not any("--audit-only" in argv for argv, _ in runner.calls)
    assert not any(
        argv[:2] == ["tmux", "new-session"] for argv, _ in runner.calls
    )
    assert not contract.placement_aggregate.exists()


def test_both_pass_gate_audits_and_launches_all_six(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner(aggregate_passed=set(orch.LIBRARIES))
    instance = ready_orchestrator(contract, runner=runner)
    for job in contract.matched_jobs:
        write_job_manifest(job)
    for job in contract.shared_jobs:
        write_job_manifest(job)
        write_exit_zero(job)
    assert instance.step() == "waiting_placement"
    assert len([1 for argv, _ in runner.calls if "--audit-only" in argv]) == 6
    assert len(
        [1 for argv, _ in runner.calls if argv[:2] == ["tmux", "new-session"]]
    ) == 6


def test_completed_passing_library_aggregates_exactly_three_seed_runs(tmp_path):
    contract = make_contract(tmp_path)
    runner = FakeRunner(aggregate_passed={"LibA"})
    instance = ready_orchestrator(contract, runner=runner)
    for job in contract.matched_jobs:
        write_job_manifest(job)
    for job in contract.shared_jobs:
        write_job_manifest(job)
        write_exit_zero(job)
    assert instance.step() == "waiting_placement"
    for job in contract.placement_jobs:
        if job.library == "LibA":
            write_job_manifest(job)
            write_exit_zero(job)
    assert instance.step() == "complete"
    placement_aggregate_calls = [
        argv
        for argv, _ in runner.calls
        if "aggregate_mint_selection_placement_control.py" in " ".join(argv)
    ]
    assert len(placement_aggregate_calls) == 1
    argv = placement_aggregate_calls[0]
    start = argv.index("--run-dirs") + 1
    stop = argv.index("--output-dir")
    assert argv[start:stop] == [
        str(job.output_dir)
        for job in contract.placement_jobs
        if job.library == "LibA"
    ]


def test_lock_excludes_second_orchestrator(tmp_path):
    contract = make_contract(tmp_path)
    contract.state_dir.mkdir()
    first = ready_orchestrator(contract)
    second = ready_orchestrator(contract)
    first._acquire_lock()
    try:
        with pytest.raises(orch.OrchestrationError, match="holds the lock"):
            second._acquire_lock()
    finally:
        first.lock_handle.close()
        first.lock_handle = None


def test_fsynced_journal_commit_recovers_aggregate_pin(tmp_path):
    contract = make_contract(tmp_path)
    journal = orch.EventJournal(contract.event_log)
    journal.open()
    first = orch.ConfirmatoryOrchestrator(contract, journal=journal)
    first.initial_hashes = first._dependency_hashes()
    committed_hash = "a" * 64
    journal.write(
        "aggregate_completed",
        stage="matched",
        path=str(contract.matched_aggregate),
        manifest_sha256=committed_hash,
        dependency_hashes=first.initial_hashes,
    )
    journal.close()

    second_journal = orch.EventJournal(contract.event_log)
    second = orch.ConfirmatoryOrchestrator(contract, journal=second_journal)
    second.initial_hashes = second._dependency_hashes()
    assert second._recover_aggregate_hashes() == {
        contract.matched_aggregate.resolve(): committed_hash
    }
