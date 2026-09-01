#!/usr/bin/env python3
"""Fail-closed orchestration for the prespecified confirmatory LoRA study.

This is a CPU-only supervisor.  It does not import the model or PyTorch.  It
waits for the six already-running matched jobs, runs the existing audit
aggregators, launches the equal-duration shared-epoch refits, and launches the
within-chain placement control only for a library whose existing strict gate
passes.  Every production path, seed, GPU, and tmux session is fixed below.

The supervisor never overwrites an output, log, or tmux session.  A completed
training job requires both a valid manifest and the absence of an exact worker
process for that trainer/output pair.  Any disappeared worker without a
manifest is a hard failure.
"""

from __future__ import print_function

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CONTRACT_VERSION = "mint-selection-confirmatory-orchestrator-v1"
REQUIRED_HOSTNAME = "gpu-dy-p4d24xlarge-5"
LIBRARIES = ("LibA", "LibB")
SEEDS = (20260811, 20260812, 20260813)
EXPECTED_GRID = tuple((library, seed) for library in LIBRARIES for seed in SEEDS)
GPU_BY_RUN = {
    ("LibA", 20260811): 0,
    ("LibA", 20260812): 1,
    ("LibA", 20260813): 2,
    ("LibB", 20260811): 3,
    ("LibB", 20260812): 4,
    ("LibB", 20260813): 5,
}

MATCHED_SCHEMA = "mint-selection-matched-lora-v1"
MATCHED_AGGREGATE_SCHEMA = "mint-selection-matched-aggregate-v1"
SHARED_SCHEMA = "mint-selection-shared-epoch-v1"
SHARED_AGGREGATE_SCHEMA = "mint-selection-shared-epoch-aggregate-v1"
PLACEMENT_SCHEMA = "mint-selection-lora-placement-control-v1"
PLACEMENT_AGGREGATE_SCHEMA = (
    "mint-selection-lora-placement-control-aggregate-v1"
)
AGGREGATE_OUTPUTS = {
    MATCHED_AGGREGATE_SCHEMA: {
        "recomputed_retention_metrics.csv",
        "paired_changes_by_seed.csv",
        "combined_summary.csv",
        "summary.md",
    },
    SHARED_AGGREGATE_SCHEMA: {
        "retention_predictions_by_seed.csv",
        "recomputed_retention_metrics_by_seed.csv",
        "shared_epoch_selection_by_seed.csv",
        "training_audit_by_seed.csv",
        "paired_cross_minus_head_by_seed.csv",
        "mean_sample_sd_summary.csv",
        "within_chain_placement_gate.csv",
        "summary.md",
    },
    PLACEMENT_AGGREGATE_SCHEMA: {
        "retention_predictions_by_seed.csv",
        "recomputed_retention_metrics_by_seed.csv",
        "paired_within_minus_cross_by_seed.csv",
        "placement_contract_by_seed.csv",
        "training_audit_by_seed.csv",
        "mean_sample_sd_summary.csv",
        "summary.md",
    },
}


class OrchestrationError(RuntimeError):
    """A fail-closed contract violation or worker failure."""


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")


def _lexists(path):
    return os.path.lexists(str(path))


def _absolute(path):
    return Path(os.path.abspath(str(path)))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_command(argv):
    digest = hashlib.sha256()
    for value in argv:
        digest.update(os.fsencode(str(value)))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json_object(path, label):
    try:
        with open(str(path), "r") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise OrchestrationError("{} is unreadable: {}".format(label, exc))
    if not isinstance(value, dict):
        raise OrchestrationError("{} is not a JSON object".format(label))
    return value


def _require_regular_nonsymlink(path, label):
    path = Path(path)
    if path.is_symlink():
        raise OrchestrationError("{} may not be a symlink: {}".format(label, path))
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise OrchestrationError("{} is missing: {}".format(label, exc))
    if not stat.S_ISREG(mode):
        raise OrchestrationError("{} is not a regular file: {}".format(label, path))


@dataclass(frozen=True)
class JobSpec:
    stage: str
    library: str
    seed: int
    gpu: int
    session: str
    python: Path
    trainer: Path
    output_dir: Path
    log_path: Path
    schema: str
    source_dir: Path = None

    @property
    def exit_path(self):
        return self.log_path.with_suffix(".exit_code")

    @property
    def pending_exit_path(self):
        return self.log_path.with_suffix(".exit_code.pending")

    @property
    def key(self):
        return (self.library, self.seed)


class StudyContract:
    """All paths and mappings for one immutable production contract."""

    def __init__(self, repo_root):
        self.repo_root = _absolute(repo_root)
        self.experiments = self.repo_root / "private_data" / "experiments"
        self.python = self.repo_root / "venv" / "bin" / "python"
        code = self.repo_root / "downstream" / "AffibodyMHC"
        self.matched_trainer = code / "finetune_mint_selection_matched.py"
        self.matched_aggregator = code / "aggregate_mint_selection_matched.py"
        self.shared_trainer = code / "finetune_mint_selection_shared_epoch.py"
        self.shared_aggregator = code / "aggregate_mint_selection_shared_epoch.py"
        self.placement_trainer = code / "finetune_mint_selection_placement_control.py"
        self.placement_aggregator = code / "aggregate_mint_selection_placement_control.py"

        self.matched_aggregate = self.experiments / (
            "mint_selection_matched_confirmatory_aggregate_v1"
        )
        self.shared_aggregate = self.experiments / (
            "mint_selection_shared_epoch_confirmatory_aggregate_v1"
        )
        self.placement_aggregate = self.experiments / (
            "mint_selection_placement_control_confirmatory_aggregate_v1"
        )
        self.state_dir = self.experiments / (
            "mint_selection_confirmatory_orchestrator_v1"
        )
        self.event_log = self.state_dir / "events.jsonl"
        self.lock_path = self.state_dir / "orchestrator.lock"

        self.matched_jobs = self._build_jobs("matched")
        self.shared_jobs = self._build_jobs("shared")
        self.placement_jobs = self._build_jobs("placement")

    def _build_jobs(self, stage):
        jobs = []
        for library, seed in EXPECTED_GRID:
            lower = library.lower()
            short_seed = str(seed)[-2:]
            gpu = GPU_BY_RUN[(library, seed)]
            if stage == "matched":
                trainer = self.matched_trainer
                output_dir = self.experiments / (
                    "mint_selection_matched_confirmatory_{}_seed{}_v1".format(
                        lower, seed
                    )
                )
                log_path = self.experiments / (
                    "mint_selection_matched_confirmatory_logs_v1"
                ) / "{}_seed{}.log".format(lower, seed)
                session = "mint_conf_{}_{}".format(lower, short_seed)
                schema = MATCHED_SCHEMA
                source = None
            elif stage == "shared":
                trainer = self.shared_trainer
                output_dir = self.experiments / (
                    "mint_selection_shared_epoch_confirmatory_{}_seed{}_v1".format(
                        lower, seed
                    )
                )
                log_path = self.experiments / (
                    "mint_selection_shared_epoch_confirmatory_logs_v1"
                ) / "{}_seed{}.log".format(lower, seed)
                session = "mint_conf_shared_{}_{}".format(lower, short_seed)
                schema = SHARED_SCHEMA
                source = self.experiments / (
                    "mint_selection_matched_confirmatory_{}_seed{}_v1".format(
                        lower, seed
                    )
                )
            elif stage == "placement":
                trainer = self.placement_trainer
                output_dir = self.experiments / (
                    "mint_selection_placement_control_confirmatory_{}_seed{}_v1".format(
                        lower, seed
                    )
                )
                log_path = self.experiments / (
                    "mint_selection_placement_control_confirmatory_logs_v1"
                ) / "{}_seed{}.log".format(lower, seed)
                session = "mint_conf_place_{}_{}".format(lower, short_seed)
                schema = PLACEMENT_SCHEMA
                source = self.experiments / (
                    "mint_selection_shared_epoch_confirmatory_{}_seed{}_v1".format(
                        lower, seed
                    )
                )
            else:
                raise AssertionError(stage)
            jobs.append(
                JobSpec(
                    stage=stage,
                    library=library,
                    seed=seed,
                    gpu=gpu,
                    session=session,
                    python=self.python,
                    trainer=trainer,
                    output_dir=output_dir,
                    log_path=log_path,
                    schema=schema,
                    source_dir=source,
                )
            )
        return tuple(jobs)

    @property
    def hashed_dependencies(self):
        return {
            "orchestrator": Path(__file__).resolve(),
            "python": self.python,
            "matched_trainer": self.matched_trainer,
            "matched_aggregator": self.matched_aggregator,
            "shared_trainer": self.shared_trainer,
            "shared_aggregator": self.shared_aggregator,
            "placement_trainer": self.placement_trainer,
            "placement_aggregator": self.placement_aggregator,
        }

    def validate_static(self):
        if self.repo_root.is_symlink() or not self.repo_root.is_dir():
            raise OrchestrationError("repository root is not a real directory")
        if not self.experiments.is_dir() or self.experiments.is_symlink():
            raise OrchestrationError("private experiment root is not a real directory")
        if tuple(job.key for job in self.matched_jobs) != EXPECTED_GRID:
            raise OrchestrationError("matched grid changed")
        if tuple(job.key for job in self.shared_jobs) != EXPECTED_GRID:
            raise OrchestrationError("shared grid changed")
        if tuple(job.key for job in self.placement_jobs) != EXPECTED_GRID:
            raise OrchestrationError("placement grid changed")
        sessions = [
            job.session
            for jobs in (self.matched_jobs, self.shared_jobs, self.placement_jobs)
            for job in jobs
        ]
        if len(sessions) != len(set(sessions)):
            raise OrchestrationError("tmux session mapping is not unique")
        outputs = [
            job.output_dir
            for jobs in (self.matched_jobs, self.shared_jobs, self.placement_jobs)
            for job in jobs
        ] + [self.matched_aggregate, self.shared_aggregate, self.placement_aggregate]
        if len(outputs) != len(set(outputs)):
            raise OrchestrationError("output mapping is not unique")
        for name, path in self.hashed_dependencies.items():
            if name == "python":
                try:
                    mode = path.resolve(strict=True).stat().st_mode
                except OSError as exc:
                    raise OrchestrationError("python is unavailable: {}".format(exc))
                if not stat.S_ISREG(mode) or not os.access(str(path), os.X_OK):
                    raise OrchestrationError("python is not an executable regular file")
            else:
                _require_regular_nonsymlink(path, name)


class EventJournal:
    """Append-only, fsync'd JSON-lines journal suitable for shared FSx."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def open(self):
        parent = self.path.parent
        if _lexists(parent):
            if parent.is_symlink() or not parent.is_dir():
                raise OrchestrationError("journal parent is not a real directory")
        else:
            parent.mkdir(mode=0o700)
        os.chmod(str(parent), 0o700)
        self.fd = os.open(
            str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        os.chmod(str(self.path), 0o600)

    def write(self, event, **fields):
        if self.fd is None:
            raise OrchestrationError("journal is not open")
        payload = {"time_utc": _utc_now(), "event": str(event)}
        payload.update(fields)
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        written = os.write(self.fd, encoded)
        if written != len(encoded):
            raise OrchestrationError("short write to event journal")
        os.fsync(self.fd)

    def read_events(self):
        if not self.path.exists():
            return []
        events = []
        try:
            with open(str(self.path), "r") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise OrchestrationError(
                            "blank event-journal line {}".format(number)
                        )
                    value = json.loads(line)
                    if not isinstance(value, dict) or "event" not in value:
                        raise OrchestrationError(
                            "malformed event-journal line {}".format(number)
                        )
                    events.append(value)
        except (OSError, ValueError) as exc:
            raise OrchestrationError("event journal is unreadable: {}".format(exc))
        return events

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class CommandRunner:
    def run(self, argv, env=None):
        return subprocess.run(
            [str(value) for value in argv],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )


class ProcWorkerInspector:
    """Find exact script/--output-dir pairs without pgrep self-matches."""

    def __init__(self, proc_root=Path("/proc")):
        self.proc_root = Path(proc_root)

    @staticmethod
    def _resolved_token(token, cwd):
        path = Path(token)
        if not path.is_absolute():
            path = cwd / path
        return _absolute(path)

    @staticmethod
    def _option_value(tokens, option):
        for index, token in enumerate(tokens):
            if token == option and index + 1 < len(tokens):
                return tokens[index + 1]
            prefix = option + "="
            if token.startswith(prefix):
                return token[len(prefix):]
        return None

    def claiming_output_pids(self, output_dir):
        """Return every process, regardless of script, claiming --output-dir."""
        expected_output = _absolute(output_dir)
        found = []
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as exc:
            raise OrchestrationError("cannot inspect /proc: {}".format(exc))
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
                tokens = [
                    os.fsdecode(value)
                    for value in raw.split(b"\0")
                    if value
                ]
                if not tokens:
                    continue
                cwd = Path(os.readlink(str(entry / "cwd")))
            except (OSError, UnicodeError):
                continue
            output_value = self._option_value(tokens, "--output-dir")
            if output_value is None:
                continue
            try:
                observed_output = self._resolved_token(output_value, cwd)
            except (OSError, ValueError):
                continue
            if observed_output != expected_output:
                continue
            found.append(int(entry.name))
        return tuple(sorted(found))

    def matching_pids(self, script, output_dir):
        expected_script = _absolute(script)
        found = []
        for pid in self.claiming_output_pids(output_dir):
            try:
                entry = self.proc_root / str(pid)
                raw = (entry / "cmdline").read_bytes()
                tokens = [os.fsdecode(value) for value in raw.split(b"\0") if value]
                cwd = Path(os.readlink(str(entry / "cwd")))
            except (OSError, UnicodeError):
                continue
            scripts = []
            for token in tokens:
                if token.endswith(".py"):
                    try:
                        scripts.append(self._resolved_token(token, cwd))
                    except (OSError, ValueError):
                        pass
            if expected_script in scripts:
                found.append(pid)
        return tuple(found)

    def _read_process(self, pid):
        entry = self.proc_root / str(pid)
        try:
            raw = (entry / "cmdline").read_bytes()
            tokens = [os.fsdecode(value) for value in raw.split(b"\0") if value]
            cwd = Path(os.readlink(str(entry / "cwd")))
            raw_environment = (entry / "environ").read_bytes()
            environment = {}
            for value in raw_environment.split(b"\0"):
                if not value or b"=" not in value:
                    continue
                key, item = value.split(b"=", 1)
                environment[os.fsdecode(key)] = os.fsdecode(item)
            stdout_path = _absolute(os.readlink(str(entry / "fd" / "1")))
        except FileNotFoundError:
            # Normal exit between the candidate scan and the detailed audit.
            return None
        except (OSError, UnicodeError) as exc:
            raise OrchestrationError(
                "cannot audit exact worker {}: {}".format(pid, exc)
            )
        return tokens, cwd, environment, stdout_path

    def matching_job_pids(self, job, gate_aggregate):
        """Match and fully validate every worker targeting one exact output."""
        pids = self.claiming_output_pids(job.output_dir)
        audited_pids = []
        for pid in pids:
            process = self._read_process(pid)
            if process is None:
                continue
            tokens, cwd, environment, stdout_path = process
            output = self._option_value(tokens, "--output-dir")
            scripts = [
                self._resolved_token(token, cwd)
                for token in tokens
                if token.endswith(".py")
            ]
            # A PID may be reused between the two reads.  Such a process is not
            # this worker and must not be validated or counted.
            if (
                output is None
                or self._resolved_token(output, cwd) != _absolute(job.output_dir)
            ):
                continue
            if _absolute(job.trainer) not in scripts:
                raise OrchestrationError(
                    "process {} claims the output with an unexpected trainer".format(pid)
                )
            if not tokens or self._resolved_token(tokens[0], cwd) != _absolute(
                job.python
            ):
                raise OrchestrationError(
                    "worker {} uses the wrong Python executable".format(pid)
                )
            if environment.get("CUDA_VISIBLE_DEVICES") != str(job.gpu):
                raise OrchestrationError(
                    "worker {} has the wrong CUDA_VISIBLE_DEVICES".format(pid)
                )
            if self._option_value(tokens, "--device") != "cuda:0":
                raise OrchestrationError("worker {} has the wrong --device".format(pid))
            if stdout_path != _absolute(job.log_path):
                raise OrchestrationError("worker {} writes to the wrong log".format(pid))
            if job.stage == "matched":
                if self._option_value(tokens, "--library") != job.library:
                    raise OrchestrationError("matched worker library mismatch")
                seed = self._option_value(tokens, "--training-seeds")
                if seed != str(job.seed):
                    raise OrchestrationError("matched worker seed mismatch")
            elif job.stage == "shared":
                source = self._option_value(tokens, "--source-run-dir")
                if source is None or self._resolved_token(source, cwd) != _absolute(
                    job.source_dir
                ):
                    raise OrchestrationError("shared worker source mismatch")
            elif job.stage == "placement":
                source = self._option_value(tokens, "--source-shared-run-dir")
                gate = self._option_value(tokens, "--gate-aggregate-dir")
                if source is None or self._resolved_token(source, cwd) != _absolute(
                    job.source_dir
                ):
                    raise OrchestrationError("placement worker source mismatch")
                if gate is None or self._resolved_token(gate, cwd) != _absolute(
                    gate_aggregate
                ):
                    raise OrchestrationError("placement worker gate mismatch")
            else:
                raise OrchestrationError("unknown worker stage")
            audited_pids.append(pid)
        return tuple(audited_pids)


class ConfirmatoryOrchestrator:
    def __init__(
        self,
        contract,
        runner=None,
        workers=None,
        journal=None,
        poll_seconds=30,
        max_session_only_polls=2,
        sleeper=time.sleep,
    ):
        self.contract = contract
        self.runner = runner or CommandRunner()
        self.workers = workers or ProcWorkerInspector()
        self.journal = journal
        self.poll_seconds = int(poll_seconds)
        self.max_session_only_polls = int(max_session_only_polls)
        self.sleeper = sleeper
        self.initial_hashes = {}
        self.session_only_counts = {}
        self.placement_audited = set()
        self.created_aggregate_hashes = {}
        self.lock_handle = None

    def _log(self, event, **fields):
        if self.journal is not None:
            self.journal.write(event, **fields)

    def _run_command(self, argv, purpose, env=None):
        argv = [str(value) for value in argv]
        command_hash = _sha256_command(argv)
        self._log(
            "command_started",
            purpose=purpose,
            argv=argv,
            command_sha256=command_hash,
            environment_overrides={
                key: env[key]
                for key in ("CUDA_VISIBLE_DEVICES", "PYTHONUNBUFFERED")
                if env is not None and key in env
            },
        )
        result = self.runner.run(argv, env=env)
        self._log(
            "command_finished",
            purpose=purpose,
            argv=argv,
            command_sha256=command_hash,
            returncode=int(result.returncode),
            stdout_sha256=hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            stderr_sha256=hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
            stdout_tail=result.stdout[-4000:],
            stderr_tail=result.stderr[-4000:],
        )
        return result

    def _dependency_hashes(self):
        return {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in sorted(self.contract.hashed_dependencies.items())
        }

    @staticmethod
    def _validate_runtime_host():
        observed = os.uname().nodename
        if observed != REQUIRED_HOSTNAME:
            raise OrchestrationError(
                "production orchestration must run on {}; observed {}".format(
                    REQUIRED_HOSTNAME, observed
                )
            )

    def _validate_gpu_inventory(self):
        result = self._run_command(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            purpose="verify assigned GPU inventory",
        )
        if result.returncode != 0:
            raise OrchestrationError(
                "GPU inventory failed: {}".format(result.stderr.strip())
            )
        try:
            observed = set(int(line.strip()) for line in result.stdout.splitlines())
        except ValueError:
            raise OrchestrationError("GPU inventory returned malformed indices")
        required = set(GPU_BY_RUN.values())
        if not required.issubset(observed):
            raise OrchestrationError(
                "assigned GPUs are unavailable: required {}, observed {}".format(
                    sorted(required), sorted(observed)
                )
            )
        self._log("gpu_inventory_verified", indices=sorted(observed))

    def _assert_dependencies_unchanged(self):
        current = self._dependency_hashes()
        if current != self.initial_hashes:
            raise OrchestrationError("a hashed executable or study script changed")

    def _recover_aggregate_hashes(self):
        """Recover only fsync'd aggregate commits made by this exact code set."""
        allowed = {
            _absolute(self.contract.matched_aggregate),
            _absolute(self.contract.shared_aggregate),
            _absolute(self.contract.placement_aggregate),
        }
        recovered = {}
        for event in self.journal.read_events():
            if event.get("event") != "aggregate_completed":
                continue
            path = _absolute(event.get("path", ""))
            if path not in allowed:
                raise OrchestrationError(
                    "journal names an unexpected aggregate path"
                )
            if event.get("dependency_hashes") != self.initial_hashes:
                continue
            manifest_hash = str(event.get("manifest_sha256", ""))
            if len(manifest_hash) != 64 or any(
                value not in "0123456789abcdef" for value in manifest_hash
            ):
                raise OrchestrationError("journal aggregate hash is malformed")
            previous = recovered.setdefault(path, manifest_hash)
            if previous != manifest_hash:
                raise OrchestrationError(
                    "journal records conflicting hashes for one aggregate"
                )
        return recovered

    def _job_worker_pids(self, job):
        if hasattr(self.workers, "matching_job_pids"):
            return tuple(
                self.workers.matching_job_pids(
                    job, self.contract.shared_aggregate
                )
            )
        # Small fake inspectors used by unit tests implement the minimal exact
        # script/output interface.  Production always uses the stricter method.
        return tuple(self.workers.matching_pids(job.trainer, job.output_dir))

    def _acquire_lock(self):
        lock_path = self.contract.lock_path
        self.lock_handle = open(str(lock_path), "a+")
        os.chmod(str(lock_path), 0o600)
        try:
            fcntl.flock(
                self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except OSError:
            self.lock_handle.close()
            self.lock_handle = None
            raise OrchestrationError("another confirmatory orchestrator holds the lock")

    def _tmux_sessions(self):
        result = self._run_command(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            purpose="list tmux sessions",
        )
        if result.returncode == 0:
            return set(line.strip() for line in result.stdout.splitlines() if line.strip())
        no_server = result.returncode == 1 and (
            "no server" in result.stderr.lower()
            or "failed to connect" in result.stderr.lower()
        )
        if no_server:
            return set()
        raise OrchestrationError(
            "tmux session inventory failed: {}".format(result.stderr.strip())
        )

    def _validate_job_manifest(self, job):
        output_dir = job.output_dir
        manifest_path = output_dir / "manifest.json"
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise OrchestrationError(
                "{} output is not a real directory: {}".format(job.stage, output_dir)
            )
        _require_regular_nonsymlink(manifest_path, "{} manifest".format(job.stage))
        manifest = _read_json_object(manifest_path, "{} manifest".format(job.stage))
        if manifest.get("schema_version") != job.schema:
            raise OrchestrationError("{} manifest schema changed".format(job.stage))
        expected_status = (
            "retrospective_exploratory_retention_gated"
            if job.stage == "placement"
            else "retrospective_exploratory"
        )
        if manifest.get("analysis_status") != expected_status:
            raise OrchestrationError("{} analysis status changed".format(job.stage))
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise OrchestrationError("{} manifest lacks configuration".format(job.stage))
        if configuration.get("library") != job.library:
            raise OrchestrationError("{} manifest library mismatch".format(job.stage))
        if job.stage in ("matched", "shared"):
            observed_seeds = configuration.get("training_seeds")
            if (
                not isinstance(observed_seeds, list)
                or len(observed_seeds) != 1
                or type(observed_seeds[0]) is not int
                or observed_seeds[0] != job.seed
            ):
                raise OrchestrationError("{} manifest seed mismatch".format(job.stage))
        else:
            observed_seed = configuration.get("training_seed")
            if type(observed_seed) is not int or observed_seed != job.seed:
                raise OrchestrationError("placement manifest seed mismatch")
        if job.stage == "shared":
            epoch = configuration.get("chosen_positive_epoch")
            if type(epoch) is not int or epoch not in (1, 2, 3):
                raise OrchestrationError("shared selected epoch is malformed")
        if job.stage == "placement":
            epoch = configuration.get("shared_positive_epoch")
            if type(epoch) is not int or epoch not in (1, 2, 3):
                raise OrchestrationError("placement selected epoch is malformed")
        manifest_hash = _sha256_file(manifest_path)
        return manifest, manifest_hash

    @staticmethod
    def _read_exit_code(job):
        path = job.exit_path
        if not _lexists(path):
            return None
        _require_regular_nonsymlink(path, "job exit status")
        try:
            text = path.read_text().strip()
            value = int(text)
        except (OSError, ValueError):
            raise OrchestrationError("job exit status is malformed: {}".format(path))
        if text != str(value):
            raise OrchestrationError("job exit status is not canonical: {}".format(path))
        return value

    def _observe_job(self, job, sessions, allow_not_started):
        workers = self._job_worker_pids(job)
        manifest_path = job.output_dir / "manifest.json"
        manifest_exists = _lexists(manifest_path)
        output_exists = _lexists(job.output_dir)
        log_exists = _lexists(job.log_path)
        session_exists = job.session in sessions
        exit_code = self._read_exit_code(job)
        pending_exit = _lexists(job.pending_exit_path)

        if pending_exit and not session_exists:
            raise OrchestrationError(
                "{} {} left an unpublished exit-status file".format(
                    job.stage, job.key
                )
            )

        if len(workers) > 1:
            raise OrchestrationError(
                "{} {} has multiple exact workers: {}".format(
                    job.stage, job.key, workers
                )
            )
        if workers and exit_code is not None:
            raise OrchestrationError(
                "{} {} has both a worker and an exit status".format(job.stage, job.key)
            )
        if exit_code not in (None, 0):
            raise OrchestrationError(
                "{} {} worker exited with status {}".format(
                    job.stage, job.key, exit_code
                )
            )

        if workers:
            state = "worker_running_manifest_present" if manifest_exists else "worker_running"
            self.session_only_counts.pop((job.stage, job.key), None)
        elif manifest_exists and job.stage != "matched" and (
            exit_code is None or pending_exit
        ):
            if not session_exists:
                raise OrchestrationError(
                    "{} {} has a manifest but no recorded zero exit"
                    .format(job.stage, job.key)
                )
            state = "session_finishing"
        elif manifest_exists:
            _require_regular_nonsymlink(job.log_path, "job log")
            _, manifest_hash = self._validate_job_manifest(job)
            state = "complete"
            self.session_only_counts.pop((job.stage, job.key), None)
            return {
                "state": state,
                "pids": [],
                "session": session_exists,
                "output": output_exists,
                "log": log_exists,
                "manifest_sha256": manifest_hash,
                "exit_code": exit_code,
            }
        elif session_exists:
            count_key = (job.stage, job.key)
            count = self.session_only_counts.get(count_key, 0) + 1
            self.session_only_counts[count_key] = count
            if count > self.max_session_only_polls:
                raise OrchestrationError(
                    "{} {} has a tmux session but no exact worker or manifest"
                    .format(job.stage, job.key)
                )
            state = "session_starting"
        elif exit_code is not None:
            raise OrchestrationError(
                "{} {} exited zero without a manifest".format(job.stage, job.key)
            )
        elif output_exists or log_exists:
            raise OrchestrationError(
                "{} {} stopped without a manifest; refusing existing output/log"
                .format(job.stage, job.key)
            )
        elif allow_not_started:
            state = "not_started"
        else:
            raise OrchestrationError(
                "{} {} has no worker and no manifest".format(job.stage, job.key)
            )
        return {
            "state": state,
            "pids": list(workers),
            "session": session_exists,
            "output": output_exists,
            "log": log_exists,
            "exit_code": exit_code,
            "pending_exit": pending_exit,
        }

    def _observe_jobs(self, jobs, sessions, allow_not_started):
        observed = {
            "{}:{}".format(job.library, job.seed): self._observe_job(
                job, sessions, allow_not_started
            )
            for job in jobs
        }
        self._log(
            "state_snapshot",
            stage=jobs[0].stage,
            jobs=observed,
        )
        return observed

    def _job_command(self, job):
        if job.stage == "shared":
            return [
                str(self.contract.python),
                "-u",
                str(job.trainer),
                "--source-run-dir",
                str(job.source_dir),
                "--output-dir",
                str(job.output_dir),
                "--device",
                "cuda:0",
            ]
        if job.stage == "placement":
            return [
                str(self.contract.python),
                "-u",
                str(job.trainer),
                "--source-shared-run-dir",
                str(job.source_dir),
                "--gate-aggregate-dir",
                str(self.contract.shared_aggregate),
                "--output-dir",
                str(job.output_dir),
                "--device",
                "cuda:0",
            ]
        raise OrchestrationError("the supervisor may not launch matched jobs")

    def _tmux_launch_command(self, job):
        worker = self._job_command(job)
        shell = "set -o noclobber; umask 077; cd {}; export PYTHONUNBUFFERED=1; " \
            "export CUDA_VISIBLE_DEVICES={}; {} > {} 2>&1; " \
            "run_status=$?; printf '%s\\n' \"$run_status\" > {}; " \
            "if ! ln {} {}; then exit 125; fi; rm -f {}; " \
            "exit \"$run_status\"".format(
                shlex.quote(str(self.contract.repo_root)),
                job.gpu,
                " ".join(shlex.quote(value) for value in worker),
                shlex.quote(str(job.log_path)),
                shlex.quote(str(job.pending_exit_path)),
                shlex.quote(str(job.pending_exit_path)),
                shlex.quote(str(job.exit_path)),
                shlex.quote(str(job.pending_exit_path)),
            )
        tmux_shell_command = "/bin/bash -lc {}".format(shlex.quote(shell))
        return [
            "tmux",
            "new-session",
            "-d",
            "-s",
            job.session,
            tmux_shell_command,
        ]

    def _ensure_log_parent(self, job):
        parent = job.log_path.parent
        if _lexists(parent):
            if parent.is_symlink() or not parent.is_dir():
                raise OrchestrationError("log parent is not a real directory")
            return
        parent.mkdir(mode=0o700)
        os.chmod(str(parent), 0o700)
        self._log("directory_created", purpose="job logs", path=str(parent))

    def _assert_gpu_idle(self, job):
        """Refuse a GPU that acquired any compute process before launch."""
        result = self._run_command(
            [
                "nvidia-smi",
                "--id={}".format(job.gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            purpose="verify GPU {} is idle before {} {} seed {}".format(
                job.gpu, job.stage, job.library, job.seed
            ),
        )
        if result.returncode != 0:
            raise OrchestrationError(
                "GPU {} occupancy query failed: {}".format(
                    job.gpu, result.stderr.strip()
                )
            )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if rows:
            raise OrchestrationError(
                "GPU {} acquired compute process(es) {}; refusing launch".format(
                    job.gpu, rows
                )
            )
        self._log(
            "gpu_idle_verified",
            gpu=job.gpu,
            stage=job.stage,
            library=job.library,
            training_seed=job.seed,
        )

    def _launch_job(self, job, sessions):
        self._assert_dependencies_unchanged()
        if job.session in sessions:
            raise OrchestrationError("refusing existing tmux session {}".format(job.session))
        if _lexists(job.output_dir):
            raise OrchestrationError("refusing existing output {}".format(job.output_dir))
        if _lexists(job.log_path):
            raise OrchestrationError("refusing existing log {}".format(job.log_path))
        if _lexists(job.exit_path):
            raise OrchestrationError(
                "refusing existing exit status {}".format(job.exit_path)
            )
        if _lexists(job.pending_exit_path):
            raise OrchestrationError(
                "refusing existing pending exit status {}".format(
                    job.pending_exit_path
                )
            )
        if self._job_worker_pids(job):
            raise OrchestrationError("an exact worker appeared before launch")
        self._assert_gpu_idle(job)
        self._ensure_log_parent(job)
        argv = self._tmux_launch_command(job)
        result = self._run_command(
            argv,
            purpose="launch {} {} seed {} on GPU {}".format(
                job.stage, job.library, job.seed, job.gpu
            ),
        )
        if result.returncode != 0:
            raise OrchestrationError(
                "tmux launch failed for {} {}: {}".format(
                    job.stage, job.key, result.stderr.strip()
                )
            )
        self._log(
            "job_launch_accepted",
            stage=job.stage,
            library=job.library,
            training_seed=job.seed,
            gpu=job.gpu,
            session=job.session,
            source_dir=None if job.source_dir is None else str(job.source_dir),
            output_dir=str(job.output_dir),
            log_path=str(job.log_path),
            trainer_sha256=self.initial_hashes[
                "{}_trainer".format(job.stage)
            ]["sha256"],
        )

    def _aggregate_command(self, script, run_dirs, output_dir):
        return [
            str(self.contract.python),
            str(script),
            "--run-dirs",
        ] + [str(path) for path in run_dirs] + ["--output-dir", str(output_dir)]

    def _validate_recorded_outputs(self, aggregate_dir, manifest, schema):
        outputs = manifest.get("outputs")
        expected_names = AGGREGATE_OUTPUTS.get(schema)
        if not isinstance(outputs, dict) or set(outputs) != expected_names:
            raise OrchestrationError("aggregate recorded-output set changed")
        for name, record in outputs.items():
            if Path(name).name != name or not isinstance(record, dict):
                raise OrchestrationError("malformed aggregate output record")
            expected = aggregate_dir / name
            if _absolute(record.get("path", "")) != _absolute(expected):
                raise OrchestrationError("aggregate output path mismatch")
            _require_regular_nonsymlink(expected, "aggregate output")
            if _sha256_file(expected) != record.get("sha256"):
                raise OrchestrationError("aggregate output hash mismatch")

    def _validate_source_run_record(self, record, run_dir):
        if not isinstance(record, dict):
            raise OrchestrationError("malformed aggregate source-run record")
        run_dir = _absolute(run_dir)
        manifest_path = run_dir / "manifest.json"
        if _absolute(record.get("manifest_path", "")) != manifest_path:
            raise OrchestrationError("aggregate source manifest path mismatch")
        _require_regular_nonsymlink(manifest_path, "aggregate source manifest")
        manifest_hash = _sha256_file(manifest_path)
        if record.get("manifest_sha256") != manifest_hash:
            raise OrchestrationError("aggregate source manifest changed")
        manifest = _read_json_object(manifest_path, "aggregate source manifest")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise OrchestrationError("aggregate source manifest has no outputs")
        for name, output_record in outputs.items():
            if Path(name).name != name or not isinstance(output_record, dict):
                raise OrchestrationError("malformed source output record")
            path = run_dir / name
            if _absolute(output_record.get("path", "")) != path:
                raise OrchestrationError("source output path mismatch")
            _require_regular_nonsymlink(path, "source output")
            if _sha256_file(path) != output_record.get("sha256"):
                raise OrchestrationError("source output changed after aggregation")

    def _validate_aggregate(self, output_dir, schema, run_dirs):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise OrchestrationError("aggregate output is not a real directory")
        manifest_path = output_dir / "manifest.json"
        _require_regular_nonsymlink(manifest_path, "aggregate manifest")
        manifest = _read_json_object(manifest_path, "aggregate manifest")
        if manifest.get("schema_version") != schema:
            raise OrchestrationError("aggregate schema mismatch")
        source_runs = manifest.get("source_runs")
        if not isinstance(source_runs, list):
            raise OrchestrationError("aggregate lacks source-run records")
        expected = set(_absolute(path) for path in run_dirs)
        observed = set(
            _absolute(record.get("path", ""))
            for record in source_runs
            if isinstance(record, dict)
        )
        if observed != expected or len(source_runs) != len(expected):
            raise OrchestrationError("aggregate source grid mismatch")
        records_by_path = {
            _absolute(record["path"]): record
            for record in source_runs
            if isinstance(record, dict) and "path" in record
        }
        for run_dir in expected:
            self._validate_source_run_record(records_by_path[run_dir], run_dir)
        self._validate_recorded_outputs(output_dir, manifest, schema)
        return manifest, _sha256_file(manifest_path)

    def _ensure_aggregate(self, name, script, run_dirs, output_dir, schema):
        self._assert_dependencies_unchanged()
        if _lexists(output_dir):
            pinned_hash = self.created_aggregate_hashes.get(_absolute(output_dir))
            if pinned_hash is None:
                raise OrchestrationError(
                    "aggregate output predates this supervisor; refusing unaudited reuse: {}"
                    .format(output_dir)
                )
            manifest, manifest_hash = self._validate_aggregate(
                output_dir, schema, run_dirs
            )
            if manifest_hash != pinned_hash:
                raise OrchestrationError("a completed aggregate manifest changed")
            self._log(
                "aggregate_rechecked",
                stage=name,
                path=str(output_dir),
                manifest_sha256=manifest_hash,
            )
            return manifest
        argv = self._aggregate_command(script, run_dirs, output_dir)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        result = self._run_command(
            argv, purpose="run {} aggregator".format(name), env=environment
        )
        if result.returncode != 0:
            raise OrchestrationError(
                "{} aggregator failed: {}".format(name, result.stderr.strip())
            )
        manifest, manifest_hash = self._validate_aggregate(
            output_dir, schema, run_dirs
        )
        self.created_aggregate_hashes[_absolute(output_dir)] = manifest_hash
        self._log(
            "aggregate_completed",
            stage=name,
            path=str(output_dir),
            manifest_sha256=manifest_hash,
            dependency_hashes=self.initial_hashes,
        )
        return manifest

    @staticmethod
    def _strict_bool(value, label):
        text = str(value).strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise OrchestrationError("{} is not a strict Boolean".format(label))

    def _read_gate(self):
        aggregate_dir = self.contract.shared_aggregate
        pinned_hash = self.created_aggregate_hashes.get(_absolute(aggregate_dir))
        if pinned_hash is None:
            raise OrchestrationError(
                "shared aggregate was not created by this supervisor"
            )
        _, audited_hash = self._validate_aggregate(
            aggregate_dir,
            SHARED_AGGREGATE_SCHEMA,
            [job.output_dir for job in self.contract.shared_jobs],
        )
        if audited_hash != pinned_hash:
            raise OrchestrationError("shared aggregate manifest changed before gate use")
        manifest = _read_json_object(
            aggregate_dir / "manifest.json", "shared aggregate manifest"
        )
        outputs = manifest.get("outputs", {})
        record = outputs.get("within_chain_placement_gate.csv")
        if not isinstance(record, dict):
            raise OrchestrationError("shared aggregate lacks its placement gate")
        path = aggregate_dir / "within_chain_placement_gate.csv"
        _require_regular_nonsymlink(path, "placement gate")
        if _absolute(record.get("path", "")) != _absolute(path):
            raise OrchestrationError("placement gate path mismatch")
        gate_hash = _sha256_file(path)
        if gate_hash != record.get("sha256"):
            raise OrchestrationError("placement gate hash mismatch")
        with open(str(path), "r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 2 or set(row.get("library") for row in rows) != set(LIBRARIES):
            raise OrchestrationError("placement gate library grid changed")
        passed = set()
        required = (
            "cross_log_loss_better_than_epoch0_all_seeds",
            "cross_log_loss_better_than_head_all_seeds",
            "within_peptide_spearman_better_all_seeds",
        )
        audited = {}
        for row in rows:
            library = row["library"]
            try:
                n_seeds = int(row["n_seeds"])
            except (KeyError, ValueError):
                raise OrchestrationError("placement gate seed count is malformed")
            if n_seeds != 3:
                raise OrchestrationError("placement gate did not use three seeds")
            components = [self._strict_bool(row.get(name), name) for name in required]
            gate_pass = self._strict_bool(
                row.get("placement_gate_pass"), "placement_gate_pass"
            )
            if gate_pass != all(components):
                raise OrchestrationError("placement gate is internally inconsistent")
            if gate_pass:
                passed.add(library)
            audited[library] = {
                "n_seeds": n_seeds,
                "components": dict(zip(required, components)),
                "placement_gate_pass": gate_pass,
            }
        self._log(
            "placement_gate_audited",
            path=str(path),
            sha256=gate_hash,
            libraries=audited,
        )
        return passed

    def _reject_failed_library_artifacts(self, passed, sessions):
        for job in self.contract.placement_jobs:
            if job.library in passed:
                continue
            if (
                _lexists(job.output_dir)
                or _lexists(job.log_path)
                or _lexists(job.exit_path)
                or _lexists(job.pending_exit_path)
                or job.session in sessions
                or self._job_worker_pids(job)
            ):
                raise OrchestrationError(
                    "placement artifacts exist for gate-failed {}".format(job.library)
                )

    def _audit_placement_sources(self, passed):
        """Run the placement trainer's complete read-only gate/source audit."""
        pending = [
            job
            for job in self.contract.placement_jobs
            if job.library in passed and job.key not in self.placement_audited
        ]
        if not pending:
            return
        self._assert_dependencies_unchanged()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONUNBUFFERED"] = "1"
        # Audit every source before launching any placement worker.  A failure
        # therefore cannot leave a selectively launched library/seed subset.
        for job in pending:
            argv = [
                str(self.contract.python),
                str(job.trainer),
                "--source-shared-run-dir",
                str(job.source_dir),
                "--gate-aggregate-dir",
                str(self.contract.shared_aggregate),
                "--audit-only",
            ]
            result = self._run_command(
                argv,
                purpose="audit placement source {} seed {}".format(
                    job.library, job.seed
                ),
                env=environment,
            )
            if result.returncode != 0:
                raise OrchestrationError(
                    "placement preflight failed for {} {}: {}".format(
                        job.library, job.seed, result.stderr.strip()
                    )
                )
        for job in pending:
            self.placement_audited.add(job.key)
        self._log(
            "placement_sources_audited",
            jobs=[
                {"library": job.library, "training_seed": job.seed}
                for job in pending
            ],
        )

    @staticmethod
    def _all_complete(observed):
        return all(row["state"] == "complete" for row in observed.values())

    def _launch_not_started(self, jobs, observed, sessions):
        launched = False
        for job in jobs:
            row = observed["{}:{}".format(job.library, job.seed)]
            if row["state"] == "not_started":
                self._launch_job(job, sessions)
                sessions.add(job.session)
                launched = True
        return launched

    def step(self):
        """Advance at most one state-machine pass; return state name."""
        sessions = self._tmux_sessions()
        matched = self._observe_jobs(
            self.contract.matched_jobs, sessions, allow_not_started=False
        )
        if not self._all_complete(matched):
            return "waiting_matched"

        self._ensure_aggregate(
            "matched",
            self.contract.matched_aggregator,
            [job.output_dir for job in self.contract.matched_jobs],
            self.contract.matched_aggregate,
            MATCHED_AGGREGATE_SCHEMA,
        )

        sessions = self._tmux_sessions()
        shared = self._observe_jobs(
            self.contract.shared_jobs, sessions, allow_not_started=True
        )
        if not self._all_complete(shared):
            self._launch_not_started(self.contract.shared_jobs, shared, sessions)
            return "waiting_shared"

        self._ensure_aggregate(
            "shared",
            self.contract.shared_aggregator,
            [job.output_dir for job in self.contract.shared_jobs],
            self.contract.shared_aggregate,
            SHARED_AGGREGATE_SCHEMA,
        )
        passed = self._read_gate()

        sessions = self._tmux_sessions()
        self._reject_failed_library_artifacts(passed, sessions)
        if not passed:
            if _lexists(self.contract.placement_aggregate):
                raise OrchestrationError(
                    "placement aggregate exists although no library passed"
                )
            self._log("study_completed", placement_libraries=[])
            return "complete"

        self._audit_placement_sources(passed)

        placement_jobs = tuple(
            job for job in self.contract.placement_jobs if job.library in passed
        )
        placement = self._observe_jobs(
            placement_jobs, sessions, allow_not_started=True
        )
        if not self._all_complete(placement):
            self._launch_not_started(placement_jobs, placement, sessions)
            return "waiting_placement"

        self._ensure_aggregate(
            "placement",
            self.contract.placement_aggregator,
            [job.output_dir for job in placement_jobs],
            self.contract.placement_aggregate,
            PLACEMENT_AGGREGATE_SCHEMA,
        )
        self._log("study_completed", placement_libraries=sorted(passed))
        return "complete"

    def run(self):
        self.contract.validate_static()
        self._validate_runtime_host()
        if self.journal is None:
            raise OrchestrationError("production run requires an event journal")
        self.journal.open()
        try:
            self._acquire_lock()
            self.initial_hashes = self._dependency_hashes()
            self.created_aggregate_hashes.update(
                self._recover_aggregate_hashes()
            )
            self._log(
                "orchestrator_started",
                contract_version=CONTRACT_VERSION,
                pid=os.getpid(),
                hostname=os.uname().nodename,
                poll_seconds=self.poll_seconds,
                dependency_hashes=self.initial_hashes,
                recovered_aggregate_paths=sorted(
                    str(path) for path in self.created_aggregate_hashes
                ),
            )
            self._validate_gpu_inventory()
            while True:
                state = self.step()
                self._log("state_transition", state=state)
                if state == "complete":
                    return
                self.sleeper(self.poll_seconds)
        except Exception as exc:
            self._log(
                "orchestrator_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            if self.lock_handle is not None:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                self.lock_handle.close()
                self.lock_handle = None
            self.journal.close()

    def dry_run_plan(self):
        """Return the immutable plan without executing commands or writing."""
        self.contract.validate_static()
        hashes = self._dependency_hashes()
        stages = {}
        for name, jobs in (
            ("matched", self.contract.matched_jobs),
            ("shared", self.contract.shared_jobs),
            ("placement", self.contract.placement_jobs),
        ):
            stages[name] = [
                {
                    "library": job.library,
                    "training_seed": job.seed,
                    "gpu": job.gpu,
                    "session": job.session,
                    "source_dir": None
                    if job.source_dir is None
                    else str(job.source_dir),
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                    "exit_path": str(job.exit_path),
                    "pending_exit_path": str(job.pending_exit_path),
                    "trainer": str(job.trainer),
                    "conditional_on_gate": name == "placement",
                    "tmux_argv": None
                    if name == "matched"
                    else self._tmux_launch_command(job),
                }
                for job in jobs
            ]
        placement_by_library = {
            library: [
                job.output_dir
                for job in self.contract.placement_jobs
                if job.library == library
            ]
            for library in LIBRARIES
        }
        aggregate_commands = {
            "matched": self._aggregate_command(
                self.contract.matched_aggregator,
                [job.output_dir for job in self.contract.matched_jobs],
                self.contract.matched_aggregate,
            ),
            "shared": self._aggregate_command(
                self.contract.shared_aggregator,
                [job.output_dir for job in self.contract.shared_jobs],
                self.contract.shared_aggregate,
            ),
            "placement_if_only_liba_passes": self._aggregate_command(
                self.contract.placement_aggregator,
                placement_by_library["LibA"],
                self.contract.placement_aggregate,
            ),
            "placement_if_only_libb_passes": self._aggregate_command(
                self.contract.placement_aggregator,
                placement_by_library["LibB"],
                self.contract.placement_aggregate,
            ),
            "placement_if_both_pass": self._aggregate_command(
                self.contract.placement_aggregator,
                placement_by_library["LibA"] + placement_by_library["LibB"],
                self.contract.placement_aggregate,
            ),
        }
        return {
            "contract_version": CONTRACT_VERSION,
            "dry_run": True,
            "repo_root": str(self.contract.repo_root),
            "required_hostname": REQUIRED_HOSTNAME,
            "dependency_hashes": hashes,
            "stages": stages,
            "aggregate_commands": aggregate_commands,
            "event_log": str(self.contract.event_log),
            "rules": [
                "manifest plus no exact worker is required for completion",
                "an output, log, or tmux session is never overwritten",
                "placement is launched per library only when all strict gate fields pass",
            ],
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact immutable plan; do not write, aggregate, or launch",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-session-only-polls", type=int, default=2)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        parser.error("--poll-seconds must be between 1 and 60")
    if args.max_session_only_polls < 0:
        parser.error("--max-session-only-polls must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    contract = StudyContract(repo_root)
    if args.dry_run:
        plan = ConfirmatoryOrchestrator(contract).dry_run_plan()
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    journal = EventJournal(contract.event_log)
    orchestrator = ConfirmatoryOrchestrator(
        contract,
        journal=journal,
        poll_seconds=args.poll_seconds,
        max_session_only_polls=args.max_session_only_polls,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
