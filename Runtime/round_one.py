"""Server orchestration for round one only; imports never start an experiment."""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import signal
import sqlite3
import time
import traceback
import zipfile

from Runtime.recovery_reporting import RECOVERY_VERSION, RECOVERY_COUNT_FIELDS
from Runtime.recovery_safety import atomic_json, file_hash
from Runtime.round_one_recovery import (
    SCHEMA, atomic_pickle, open_evaluation_journal, restore_rng, restore_round_archive,
    safe_load_policy, training_callback, verified_checkpoint,
    runtime_versions,
)


ROOT = Path(__file__).resolve().parent.parent
ROUND = 1
TOTAL_ROUNDS = 5
WORKERS = 30
_EVALUATOR = {}


def _terminate_run(signum, frame):
    raise KeyboardInterrupt("Termination requested; retain the last committed recovery checkpoint.")


def configure_process():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1)


@dataclass
class TrainingEnvFactory:
    """Load one graph per worker; do not send parent graphs through closures."""
    graph_path: str
    pool_path: str
    cfg: dict
    worker_count: int = WORKERS
    checkpoint_dir: str | None = None

    def __call__(self, rank):
        configure_process()
        if self.checkpoint_dir:
            import cloudpickle
            with gzip.open(Path(self.checkpoint_dir) / f"worker_{rank:03d}.pkl.gz", "rb") as handle:
                saved = cloudpickle.load(handle)
            restore_rng(saved["rng"])
            return saved["env"]
        from Evaluations.explainable_routing import ExplainableRoutingEnv
        from Pathfinding.heuristics import lnd_cost
        from Simulation.failure_model import FailureModel, assign_failure_probabilities
        from Simulation.network_dynamics import NetworkDynamics
        with Path(self.graph_path).open("rb") as handle:
            graph = pickle.load(handle)
        with Path(self.pool_path).open("rb") as handle:
            training, _held_out = pickle.load(handle)
        cfg = copy.deepcopy(self.cfg)
        seed = int(cfg["seed"]) + rank
        assign_failure_probabilities(graph, float(cfg.get("training_failure_rate", .03)),
                                     int(cfg["seed"]))
        cfg["seed"] = seed
        cfg["rl"]["seed"] = seed
        return ExplainableRoutingEnv(
            G=graph, transactions=training, heuristic_fn=lnd_cost, config=cfg,
            mode="train", worker_rank=rank, worker_count=self.worker_count,
            failure_model=FailureModel(seed=seed),
            network_dynamics=NetworkDynamics(
                graph, seed=seed, default_capacity=cfg.get("channel", {}).get("default_capacity")),
        )


def initialize_evaluator(graph_path, cfg, model_path, reference_path):
    configure_process()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from Evaluations.explainable_routing import load_policy
    snapshot = Path(graph_path).read_bytes() if cfg.get("evaluation_graph_clone") == "pickle-snapshot-v1" else None
    graph = None
    if snapshot is None:
        with Path(graph_path).open("rb") as handle:
            graph = pickle.load(handle)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))["mean"]
    _EVALUATOR.update(graph=graph, graph_snapshot=snapshot, cfg=cfg, model=load_policy(model_path, device="cpu"),
                      reference=reference)


def evaluate_task(task):
    from Evaluations.explainable_routing import evaluate_payment
    return evaluate_payment(_EVALUATOR["graph"], _EVALUATOR["cfg"], _EVALUATOR["model"],
                            task, _EVALUATOR["reference"], graph_snapshot=_EVALUATOR.get("graph_snapshot"))


def evaluation_tasks(transactions, rates, seed, checkpoint, round_number=ROUND, *, evaluation_seed=None):
    evaluation_seed = int(seed if evaluation_seed is None else evaluation_seed)
    for rate_index, rate in enumerate(rates):
        for index, tx in enumerate(transactions, 1):
            payment_seed = evaluation_seed + round_number * 10_000_000 + rate_index * 100_000 + index
            yield {
                "phase": "final", "repetition": round_number, "training_seed": seed,
                "evaluation_seed": evaluation_seed,
                "checkpoint": checkpoint, "split": "test", "failure_rate": rate,
                "scenario_seed": payment_seed, "payment_seed": payment_seed,
                "transaction_id": int(tx.tx_id), "transaction": tx,
            }


def package_round(run_dir):
    """Package once at phase exit; omit redundant training recovery payloads."""
    run_dir = Path(run_dir).resolve()
    started = time.monotonic()
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    round_number = int(metadata.get("round", ROUND))
    try:
        complete = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
        seed = metadata["training_seed"]
        model_dir = run_dir / "models" / f"rep_{round_number:02d}_seed_{seed}"
        trained = (file_hash(model_dir / "policy.zip") == complete["model_sha256"]
                   and file_hash(model_dir / "training_reference.json") == complete["reference_sha256"])
    except (OSError, ValueError, KeyError):
        trained = False
    # Completed policies do not need simulator/RNG snapshots for evaluation.
    # Interrupted training exports only the newest intact coherent generation.
    checkpoint = None if trained else verified_checkpoint(run_dir)
    generation = checkpoint[0].name if checkpoint else None
    sources = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if (not path.is_file() or path.is_symlink() or path.name.endswith(".tmp")
                or any(part.startswith(".pending-") for part in relative.parts)
                or "checkpoints" in relative.parts
                or path.name.startswith("discarded_rollout_")
                or path.name in ("artifact_manifest.json", "evaluation.sqlite3-journal",
                                 "evaluation.sqlite3-wal", "evaluation.sqlite3-shm")):
            continue
        if relative.parts[0] == "recovery":
            if trained or not generation:
                continue
            if relative.as_posix() != "recovery/latest.json" and (
                    len(relative.parts) < 3 or relative.parts[1] != generation):
                continue
        sources[relative.as_posix()] = path
    # Do not archive a damaged latest pointer if we recovered a legacy fallback.
    if checkpoint:
        export_path = run_dir / "recovery/export.json"
        export_record = {
            "generation": generation,
            "files": {p.name: file_hash(p) for p in checkpoint[0].iterdir() if p.is_file()}}
        try:
            unchanged = json.loads(export_path.read_text(encoding="utf-8")) == export_record
        except (OSError, ValueError):
            unchanged = False
        if not unchanged:
            atomic_json(export_path, export_record)
        sources.pop("recovery/latest.json", None)
        sources.pop("recovery/export.json", None)
        sources["recovery/latest.json"] = run_dir / "recovery/export.json"
    destination = run_dir.with_suffix(".zip")
    checksum = destination.with_suffix(".zip.sha256")
    state_path = destination.with_suffix(".zip.state.json")
    state = {name: [path.stat().st_size, path.stat().st_mtime_ns]
             for name, path in sources.items()}
    if destination.is_file() and checksum.is_file() and state_path.is_file():
        try:
            cached = json.loads(state_path.read_text(encoding="utf-8"))
            if (cached.get("files") == state
                    and cached.get("archive_bytes") == destination.stat().st_size
                    and cached.get("archive_mtime_ns") == destination.stat().st_mtime_ns
                    and cached.get("checksum") == checksum.read_text(encoding="ascii")):
                print(f"[ARTIFACT] Reusing unchanged package: {destination}; "
                      f"bytes={destination.stat().st_size}", flush=True)
                return destination
        except (OSError, ValueError):
            pass
    status_path = run_dir / "run_status.json"
    ready = (status_path.is_file() and
             json.loads(status_path.read_text(encoding="utf-8")).get("status") == "COMPLETED"
             and (run_dir / "summary.json").is_file())
    db_snapshot = None
    if (run_dir / "evaluation.sqlite3").is_file():
        db_snapshot = run_dir.parent / f".{run_dir.name}.{time.time_ns()}.sqlite3.tmp"
        source = sqlite3.connect(run_dir / "evaluation.sqlite3", timeout=30)
        target = sqlite3.connect(db_snapshot)
        try:
            source.backup(target)
        finally:
            source.close()
            target.close()
    entries = {}
    for name, path in sources.items():
        value = db_snapshot if name == "evaluation.sqlite3" and db_snapshot else path
        sources[name] = value
        entries[name] = {"sha256": file_hash(value), "bytes": value.stat().st_size}
    atomic_json(run_dir / "artifact_manifest.json", {
        "round": round_number, "total_rounds": TOTAL_ROUNDS, "files": entries,
        "package_kind": "trained_policy" if trained else "training_recovery",
        "experiment_complete": bool(round_number == TOTAL_ROUNDS and ready),
        "next_round": None if round_number >= TOTAL_ROUNDS else {
            "ready": ready,
            "round": round_number + 1, "training_seed": metadata["training_seed"] + 1,
            "reuse": ["inputs/transaction_split.csv", "inputs/config.yaml", "inputs/network_snapshot.gml.geo",
                      "inputs/global_energy_mix.json", "summary.json"],
            "training": "Continue the saved PPO policy and optimizer with an additional stage training budget.",
            "implemented_rounds": list(range(1, max(3, round_number) + 1)),
        },
    })
    temporary = destination.with_suffix(".zip.tmp")
    sources["artifact_manifest.json"] = run_dir / "artifact_manifest.json"
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
            for name, path in sorted(sources.items()):
                # Already-compressed audits/images do not benefit from another
                # compression pass; keep their bytes without spending CPU.
                compression = (zipfile.ZIP_STORED if path.suffix.lower() in
                               (".gz", ".png", ".jpg", ".jpeg")
                               else zipfile.ZIP_DEFLATED)
                archive.write(path, f"{run_dir.name}/{name}", compress_type=compression)
    finally:
        if db_snapshot is not None:
            db_snapshot.unlink(missing_ok=True)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    checksum_tmp = checksum.with_name(checksum.name + ".tmp")
    checksum_tmp.write_text(f"{file_hash(destination)}  {destination.name}\n", encoding="ascii")
    checksum_tmp.replace(checksum)
    atomic_json(state_path, {"files": state, "archive_bytes": destination.stat().st_size,
                             "archive_mtime_ns": destination.stat().st_mtime_ns,
                             "checksum": checksum.read_text(encoding="ascii")})
    print(f"[ARTIFACT] {destination}; bytes={destination.stat().st_size}; "
          f"files={len(entries)}; seconds={time.monotonic() - started:.2f}", flush=True)
    return destination


class MainOneOfFive:
    """Round one's separate training and held-out evaluation phases."""
    workers = WORKERS
    round_number = 1

    @property
    def label(self):
        return f"main{self.round_number}/5"

    @property
    def prefix(self):
        return f"main{self.round_number}_5"

    @property
    def model_dir(self):
        return self.run_dir / "models" / f"rep_{self.round_number:02d}_seed_{self.cfg['seed']}"

    def _configuration(self):
        import yaml
        return yaml.safe_load(self.config.read_text(encoding="utf-8"))

    def _copy_fixed_dataset(self, inputs):
        """Later stages override this to reuse the original split."""

    def __init__(self, snapshot, config, output_dir, failure_rates=None,
                 resume_run=None, resume_archive=None, backup_seconds=300,
                 checkpoint_seconds=1800):
        self.snapshot = Path(snapshot).resolve()
        self.config = Path(config).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.failure_rates = failure_rates
        self.run_dir = None
        self._last_progress = 0.0
        self._last_stage = None
        self.resume_run = Path(resume_run).resolve() if resume_run else None
        self.resume_archive = Path(resume_archive).resolve() if resume_archive else None
        self.backup_seconds = float(backup_seconds)
        if self.backup_seconds <= 0:
            raise ValueError("Backup interval must be positive.")
        self._last_backup = float("-inf")
        self.checkpoint = None
        self.checkpoint_seconds = float(checkpoint_seconds)
        if not 0 <= self.checkpoint_seconds < float("inf"):
            raise ValueError("Checkpoint interval must be finite and nonnegative.")

    def publish(self, reason, force=False, audit=None):
        # No ZIP construction, hashing or network transfers in compute loops.
        if not force:
            return
        trace_open = audit is not None and audit.trace is not None
        if trace_open:
            audit.close()
        try:
            archive = package_round(self.run_dir)
            self._last_backup = time.monotonic()
            print(f"[RECOVERY] {reason}: {archive}", flush=True)
            if os.environ.get(f"MAIN{self.round_number}_5_REMOTE_BACKUP", "0") == "1":
                from urllib.parse import quote
                from Runtime.recovery_safety import upload_with_retry
                version = f"{os.environ['CI_PIPELINE_ID']}-{self.run_dir.name}"
                url = (f"{os.environ['CI_API_V4_URL'].rstrip('/')}/projects/{os.environ['CI_PROJECT_ID']}"
                       f"/packages/generic/rl-bucket-main{self.round_number}-5/{quote(version, safe='')}/"
                       f"recovery-{time.time_ns()}.zip")
                upload_with_retry(url, archive, os.environ["CI_JOB_TOKEN"])
                atomic_json(self.run_dir / "remote_backup.json", {
                    "download_url": url, "sha256": file_hash(archive), "reason": reason})
                print(f"[RECOVERY DOWNLOAD] {url}", flush=True)
        except Exception as error:
            # A failed remote upload cannot invalidate a previous good ZIP or
            # immutable checkpoint. Keep the error visible without leaking tokens.
            print(f"[RECOVERY WARNING] {type(error).__name__}: backup publication failed", flush=True)
            atomic_json(self.run_dir / "backup_error.json", {
                "type": type(error).__name__, "reason": reason})
        finally:
            if trace_open:
                audit.trace = gzip.open(audit.directory / "training_explanations.jsonl.gz",
                                        "at", encoding="utf-8", compresslevel=3)

    def _resume(self):
        import yaml
        self.run_dir = (restore_round_archive(self.resume_archive, self.output_dir,
                                            round_number=self.round_number)
                        if self.resume_archive else self.resume_run)
        metadata = json.loads((self.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("resume_schema") != SCHEMA or metadata.get("round") != self.round_number
                or metadata.get("recovery_version") != RECOVERY_VERSION):
            raise ValueError(f"This archive does not contain a compatible resumable {self.label} run.")
        if metadata["training_workers"] != self.workers:
            raise ValueError("Training continuation requires the original worker count.")
        for name, key in (("network_snapshot.gml.geo", "snapshot_sha256"),
                          ("config.yaml", "config_sha256"), ("global_energy_mix.json", "carbon_sha256")):
            if file_hash(self.run_dir / "inputs" / name) != metadata[key]:
                raise ValueError(f"Saved input checksum mismatch: {name}")
        self.cfg = yaml.safe_load((self.run_dir / "inputs/config.yaml").read_text(encoding="utf-8"))
        self.rates = metadata["failure_rates"]
        if self.failure_rates is not None and self.failure_rates != self.rates:
            raise ValueError("Cannot change failure-rate conditions while resuming a round.")
        self.checkpoint = (None if (self.run_dir / "training_complete.json").is_file()
                           else verified_checkpoint(self.run_dir))
        if self.checkpoint and self.checkpoint[1]["workers"] != self.workers:
            raise ValueError("Checkpoint worker count mismatch.")
        if self.checkpoint:
            from RL.state import FEATURE_NAMES
            state = self.checkpoint[1]
            if (state["features"] != list(FEATURE_NAMES)
                    or state["requested_target"] != int(self.cfg["rl"]["total_timesteps"])):
                raise ValueError("Checkpoint feature schema or training target mismatch.")
            if state["runtime_versions"] != runtime_versions():
                raise ValueError("Resume with the Python/library versions recorded in the checkpoint.")
        if (not self.checkpoint and (self.run_dir / "recovery/latest.json").exists()
                and not (self.run_dir / "training_complete.json").exists()):
            raise ValueError("Neither saved PPO checkpoint generation is intact.")
        data_manifest = self.run_dir / "inputs/data_manifest.json"
        if data_manifest.is_file():
            for name, digest in json.loads(data_manifest.read_text()).items():
                if file_hash(self.run_dir / "inputs" / name) != digest:
                    raise ValueError(f"Saved dataset/cache checksum mismatch: {name}")

    def progress(self, stage, completed):
        now = time.monotonic()
        if stage != self._last_stage or now - self._last_progress >= 60:
            print(f"[{self.label}] {stage}: {completed:,}", flush=True)
            self._last_progress, self._last_stage = now, stage

    def _prepare(self):
        import yaml
        from Runtime.paper_protocol import _validate_protocol
        cfg = self._configuration()
        rates = list(self.failure_rates if self.failure_rates is not None else cfg["failure_rates"])
        _validate_protocol(cfg, rates, 1)
        if len(set(rates)) != len(rates):
            raise ValueError("Failure-rate conditions must be unique.")
        if not self.snapshot.is_file():
            raise FileNotFoundError(self.snapshot)
        cfg = copy.deepcopy(cfg)
        from Evaluations.xai_sampling import DEFAULT_SAMPLING, validate_sampling
        cfg.setdefault("round_one_xai", dict(DEFAULT_SAMPLING))
        validate_sampling(cfg["round_one_xai"])
        cfg.setdefault("reward", {})["formula"] = "paper_base"
        cfg["rl"]["seed"] = int(cfg["seed"])
        self.cfg, self.rates = cfg, rates
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.output_dir / f"{self.prefix}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for folder in ("inputs", f"models/rep_{self.round_number:02d}_seed_{cfg['seed']}", "logs", "reports", "figures"):
            (self.run_dir / folder).mkdir(parents=True, exist_ok=True)
        inputs = self.run_dir / "inputs"
        shutil.copy2(self.snapshot, inputs / "network_snapshot.gml.geo")
        shutil.copy2(getattr(self, "carbon_source", ROOT / "global_energy_mix.json"),
                     inputs / "global_energy_mix.json")
        (inputs / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        shutil.copy2(ROOT / "requirements.txt", inputs / "requirements.txt")
        with zipfile.ZipFile(inputs / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            paths = list(ROOT.glob("*.py"))
            paths.extend(ROOT / name for name in (".gitlab-ci.yml", "README.md", "requirements.txt"))
            for folder in ("Runtime", "RL", "Simulation", "Bucket", "Pathfinding", "Network", "Evaluations"):
                paths.extend((ROOT / folder).rglob("*.py"))
            for path in sorted(paths):
                archive.write(path, path.relative_to(ROOT).as_posix())
        self._copy_fixed_dataset(inputs)
        atomic_json(self.run_dir / "run_metadata.json", {
            "name": self.label, "round": self.round_number, "total_rounds": TOTAL_ROUNDS,
            "training_seed": int(cfg["seed"]),
            "evaluation_seed": int(cfg.get("evaluation_seed", cfg["seed"])),
            "training_workers": self.workers,
            "evaluation_workers": self.workers, "failure_rates": rates,
            "training_failure_rate": float(cfg.get("training_failure_rate", .03)),
            "total_transactions": 10000, "training_transactions": 7000,
            "held_out_transactions": 3000, "expected_evaluation_trials": 3000 * len(rates),
            "requested_training_transitions": int(cfg["rl"]["total_timesteps"]),
            "recovery_version": RECOVERY_VERSION, "reward_formula": "paper_base",
            "resume_schema": SCHEMA,
            "snapshot_name": self.snapshot.name,
            "snapshot_sha256": file_hash(inputs / "network_snapshot.gml.geo"),
            "config_sha256": file_hash(inputs / "config.yaml"),
            "carbon_sha256": file_hash(inputs / "global_energy_mix.json"),
            "xai": "all physical routing audits; sampled feature occlusion; sensitivity, not causality",
            "xai_sampling": cfg["round_one_xai"],
            "next_round": "Continue this trained PPO policy and optimizer with the identical transaction split.",
        })
        (self.run_dir / "HANDOFF.md").write_text(
            f"# {self.label} handoff\n\nThis archive contains one training/evaluation stage. Verify artifact_manifest.json "
            "and the ZIP SHA256 before reuse. Use the same transaction_split.csv for all five rounds. "
            "The next stage continues this saved PPO policy and optimizer with an additional training budget. "
            "Keep stage results separate. Continual stages are correlated, not independent paper repeats. "
            "Held-out evaluation does not update the policy or optimizer.\n\n"
            f"To continue THIS interrupted round, use {self.prefix}.py --resume-archive <ZIP> "
            f"or --resume-run <directory> for training only. After training, use {self.prefix}_resume.py with the saved artifacts for testing only. Use the original seed, conditions and worker count. "
            "The last completed PPO update and committed held-out payments are restored; "
            "an interrupted rollout is replayed from its preceding safe update.\n\n"
            f"Open reports/{self.prefix}.html for heatmaps, readable explanations and routing recovery. "
            "Full training and evaluation traces are compressed JSONL. Check run_status.json: "
            "an archive may contain an interrupted run and does not by itself certify completion.\n",
            encoding="utf-8")

    def _train(self, core, graph, training, pool_path, graph_path):
        from RL.state import FEATURE_NAMES
        from RL.train import train_agent
        model_dir = self.model_dir
        marker = self.run_dir / "training_complete.json"
        if marker.is_file():
            complete = json.loads(marker.read_text())
            model_path, reference_path = model_dir / "policy.zip", model_dir / "training_reference.json"
            if (file_hash(model_path) != complete["model_sha256"]
                    or file_hash(reference_path) != complete["reference_sha256"]):
                raise ValueError("Completed training artifacts are damaged.")
            return model_path, reference_path, int(complete["step"])
        cfg = copy.deepcopy(self.cfg)
        cfg["rl"].update(model_dir=str(model_dir), log_dir=str(self.run_dir / "logs"),
                         checkpoint_dir=str(model_dir / "checkpoints"))
        # Coherent resume snapshots are handled by the audit below. The shared
        # trainer's model-only callback otherwise writes another ZIP every 128
        # transitions; keep just its final-budget save for this runner.
        cfg["rl"]["checkpoint_freq"] = int(cfg["rl"]["total_timesteps"])
        audit = training_callback(cfg, model_dir, self.progress, self.run_dir, self.publish,
                                  self.checkpoint, checkpoint_seconds=self.checkpoint_seconds)
        started = time.perf_counter()
        try:
            if self.checkpoint and self.checkpoint[1]["step"] >= int(cfg["rl"]["total_timesteps"]):
                # The final PPO update was committed, but publication of the
                # final policy/reference may have been interrupted.
                # The verified checkpoint already contains the final model and
                # optimizer. Publish its bytes directly without loading/saving
                # PPO again merely to finish an interrupted publication.
                trained_steps = int(self.checkpoint[1]["step"])
                trace_path = model_dir / "training_explanations.jsonl.gz"
                if trace_path.exists():
                    trace_path.replace(trace_path.with_name(f"discarded_rollout_{time.time_ns()}.jsonl.gz"))
                shutil.copy2(self.checkpoint[0] / "training_audit.jsonl.gz", trace_path)
                # training_complete.json is the commit marker, written only
                # after model/reference hashes below. An interrupted copy has
                # no marker and is retried from the immutable safe checkpoint.
                # This also avoids replacing a previously loaded ZIP on Windows.
                policy_path = model_dir / "policy.zip"
                shutil.copyfile(self.checkpoint[0] / "model.zip", policy_path)
                with policy_path.open("r+b") as handle:
                    os.fsync(handle.fileno())
            else:
                loader = safe_load_policy
                if self.checkpoint:
                    cfg["rl"]["resume_checkpoint"] = str(self.checkpoint[0] / "model.zip")
                elif cfg["rl"].get("initial_policy"):
                    from functools import partial
                    from Runtime.round_one_recovery import load_next_round_policy
                    initial = (self.run_dir / cfg["rl"]["initial_policy"]).resolve()
                    if (not initial.is_relative_to(self.run_dir.resolve()) or
                            file_hash(initial) != cfg["rl"]["initial_policy_sha256"]):
                        raise ValueError("Previous-stage policy checksum mismatch.")
                    cfg["rl"]["resume_checkpoint"] = str(initial)
                    loader = partial(load_next_round_policy, seed=int(cfg["seed"]))
                model = train_agent(
                    graph, training, core.lnd_cost, cfg, "policy", seed=int(cfg["seed"]),
                    fast_training=False, extra_callbacks=[audit], checkpoint_loader=loader,
                    env_factory=TrainingEnvFactory(str(graph_path), str(pool_path), cfg, self.workers,
                        str(self.checkpoint[0]) if self.checkpoint else None))
                trained_steps = int(model.num_timesteps)
        finally:
            audit.close()
        model_path = model_dir / "policy.zip"
        reference_path = model_dir / "training_reference.json"
        if audit.feature_count <= 0:
            raise RuntimeError("Training finished without observable-feature reference data.")
        atomic_json(reference_path, {
            "features": list(FEATURE_NAMES), "mean": audit.reference.tolist(),
            "observed_slots": audit.feature_count, "checkpoint": trained_steps,
            "model_sha256": file_hash(model_path), "source": "training rollouts only",
        })
        metadata_path = self.run_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(actual_training_transitions=trained_steps,
                        stage_training_transitions=trained_steps - int(metadata.get("initial_training_transitions", 0)),
                        training_seconds=time.perf_counter() - started,
                        python=platform.python_version())
        import networkx
        import numpy
        import stable_baselines3
        import torch
        metadata["versions"] = {"networkx": networkx.__version__, "numpy": numpy.__version__,
                                "stable_baselines3": stable_baselines3.__version__,
                                "torch": str(torch.__version__)}
        atomic_json(metadata_path, metadata)
        atomic_json(marker, {"step": trained_steps,
                             "model_sha256": file_hash(model_path),
                             "reference_sha256": file_hash(reference_path)})
        return model_path, reference_path, trained_steps

    def trained_artifacts(self):
        """Verify the final publication marker without ever calling training code."""
        marker = self.run_dir / "training_complete.json"
        if not marker.is_file():
            raise ValueError(f"Training is not complete. Continue training with {self.prefix}.py --resume-run first; "
                             f"{self.prefix}_resume.py only runs testing.")
        complete = json.loads(marker.read_text())
        model_dir = self.model_dir
        model_path, reference_path = model_dir / "policy.zip", model_dir / "training_reference.json"
        if (not model_path.is_file() or not reference_path.is_file() or
                file_hash(model_path) != complete["model_sha256"] or
                file_hash(reference_path) != complete["reference_sha256"]):
            raise ValueError("Completed training artifacts are damaged.")
        return model_path, reference_path, int(complete["step"])

    def _finish_training_phase(self, checkpoint):
        status_path = self.run_dir / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        # Invoking training on a fully tested archive must not downgrade it.
        if status.get("status") != "COMPLETED":
            atomic_json(status_path, {
                "status": "TRAINING_COMPLETED", "phase": "train", "round": self.round_number,
                "actual_training_transitions": checkpoint, "evaluation_pending": True,
                "expected_trials": 3000 * len(self.rates),
                "next_entry_point": f"{self.prefix}_resume.py"})
        print(f"[{self.label}] TRAINING COMPLETE. Exiting without testing; "
              f"continue with {self.prefix}_resume.py and these artifacts.", flush=True)

    def _export_evaluation(self, connection, fields):
        from Evaluations.xai_report import human_explanation
        trace_path = self.run_dir / "evaluation_explanations.jsonl.gz"
        csv_path = self.run_dir / "evaluation_trials.csv"
        trace_tmp, csv_tmp = trace_path.with_name(trace_path.name + ".tmp"), csv_path.with_name(csv_path.name + ".tmp")
        count = 0
        with gzip.open(trace_tmp, "wt", encoding="utf-8") as trace, csv_tmp.open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for (payload,) in connection.execute("SELECT payload FROM payments ORDER BY seed"):
                row = json.loads(payload)
                trace.write(payload + "\n")
                flat = {key: row.get(key) for key in fields}
                for key in ("path", "recovery_events"):
                    flat[key] = json.dumps(flat[key])
                human = human_explanation(row)
                flat["human_explanation"] = " ".join(
                    human[key] for key in ("summary", "route_reason", "recovery_summary"))
                writer.writerow(flat)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        trace_tmp.replace(trace_path)
        csv_tmp.replace(csv_path)
        return count

    def _evaluate(self, held_out, graph_path, model_path, reference_path, checkpoint):
        from Evaluations.parallel_trials import completed_trials
        from Evaluations.explainable_routing import json_default
        from Evaluations.xai_report import human_explanation
        import hashlib
        fields = ["repetition", "training_seed", "evaluation_seed", "transaction_id", "source", "destination", "amount",
                  "failure_rate", "payment_seed", "success", "eta", "reward", "fee", "delay", "carbon",
                  "path_length", "attempt_count", "reason", "successful_route_source",
                  "partial_backtrack_count", "partial_backtrack_success", "full_reroute_count",
                  *RECOVERY_COUNT_FIELDS, "path", "recovery_events", "human_explanation"]
        identity = {
            "schema": SCHEMA, "model_sha256": file_hash(model_path),
            "reference_sha256": file_hash(reference_path), "checkpoint": checkpoint,
            "cfg_sha256": hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest(),
            "rates": self.rates, "transactions": [
                [int(tx.tx_id), str(tx.source), str(tx.destination), float(tx.amount)] for tx in held_out],
        }
        connection = open_evaluation_journal(self.run_dir / "evaluation.sqlite3", identity,
                                            wal=self.cfg.get("evaluation_journal") == "wal-full-v1")
        expected = {task["payment_seed"]: (task["transaction_id"], task["failure_rate"])
                    for task in evaluation_tasks(held_out, self.rates, int(self.cfg["seed"]), checkpoint,
                                                 self.round_number, evaluation_seed=self.cfg.get("evaluation_seed"))}
        done = set()
        for seed, payload in connection.execute("SELECT seed, payload FROM payments"):
            row = json.loads(payload)
            if (seed not in expected or row["payment_seed"] != seed or
                    (row["transaction_id"], row["failure_rate"]) != expected[seed]):
                connection.close()
                raise ValueError("An evaluation commit has a different trial identity.")
            done.add(seed)
        count, results = len(done), None
        try:
            if count < len(expected):
                tasks = (task for task in evaluation_tasks(held_out, self.rates, int(self.cfg["seed"]), checkpoint,
                                                           self.round_number, evaluation_seed=self.cfg.get("evaluation_seed"))
                         if task["payment_seed"] not in done)
                results = completed_trials(
                    evaluate_task, tasks, self.workers, initializer=initialize_evaluator,
                    initargs=(str(graph_path), self.cfg, str(model_path), str(reference_path)))
                for row in results:
                    row["human"] = human_explanation(row)
                    payload = json.dumps(row, default=json_default, allow_nan=False)
                    # The XAI explanation and payment outcome are a single FULL
                    # synchronous SQLite transaction. Torn CSV/gzip tails are
                    # disposable exports, never the source of resume progress.
                    with connection:
                        connection.execute("INSERT INTO payments VALUES (?, ?)", (row["payment_seed"], payload))
                    count += 1
                    self.progress("HELD-OUT EVALUATION", count)
                    if count % 30 == 0:
                        atomic_json(self.run_dir / "run_status.json", {
                            "status": "RUNNING", "stage": "EVALUATION", "completed_trials": count,
                            "expected_trials": len(expected)})
        finally:
            if results is not None:
                results.close()
            try:
                self._export_evaluation(connection, fields)
            finally:
                connection.close()
        if count != len(expected):
            raise RuntimeError("Incomplete held-out evaluation.")
        return count

    def run(self, phase="train"):
        if phase not in ("train", "evaluate"):
            raise ValueError("Phase must be train or evaluate.")
        if phase == "evaluate" and not (self.resume_run or self.resume_archive):
            raise ValueError("Testing requires saved training artifacts; a fresh training run is not allowed.")
        configure_process()
        os.environ.update(RL_ENV_WORKERS=str(self.workers), RL_FAST_ENV="0",
                          RL_FAST_TRAINING="0", RL_FAST_VALIDATION="0", MPLBACKEND="Agg")
        if self.resume_run or self.resume_archive:
            self._resume()
        else:
            self._prepare()
        os.environ["RL_REPORT_FILE"] = str(self.run_dir / "logs/bootstrap.txt")
        previous_sigterm = signal.signal(signal.SIGTERM, _terminate_run)
        try:
            if phase == "train" and (self.run_dir / "training_complete.json").is_file():
                _, _, checkpoint = self.trained_artifacts()
                self._finish_training_phase(checkpoint)
                return self.run_dir
            if phase == "evaluate":
                model_path, reference_path, checkpoint = self.trained_artifacts()
                inputs = self.run_dir / "inputs"
                graph_path, pool_path = inputs / "network.pkl", inputs / "transactions.pkl"
                if not graph_path.is_file() or not pool_path.is_file():
                    raise ValueError("Testing requires the saved network and transaction caches.")
                with pool_path.open("rb") as handle:
                    training, held_out = pickle.load(handle)
                if len(training) != 7000 or len(held_out) != 3000:
                    raise ValueError(f"Saved training/test split does not match {self.label}.")
                del training
                atomic_json(self.run_dir / "run_status.json", {
                    "status": "RUNNING", "stage": "EVALUATION", "phase": "evaluate", "round": self.round_number})
                print(f"[{self.label}] Restored completed training. Starting testing only; "
                      "PPO training will not be called.", flush=True)
                count = self._evaluate(held_out, graph_path, model_path, reference_path, checkpoint)
                from Evaluations.round_one_report import write_round_one_reports
                write_round_one_reports(self.run_dir, self.rates)
                atomic_json(self.run_dir / "run_status.json", {
                    "status": "COMPLETED", "phase": "evaluate", "round": self.round_number,
                    "completed_trials": count, "expected_trials": 3000 * len(self.rates)})
                return self.run_dir

            # Training-only entry point: there is no test scheduling on this path.
            from Runtime import experiment_core as core
            from Runtime.paper_protocol import _write_transaction_manifest
            atomic_json(self.run_dir / "run_status.json", {"status": "RUNNING", "stage": "DATA"})
            self.progress("BUILD DATA", 0)
            core.set_seed(int(self.cfg["seed"]))
            inputs = self.run_dir / "inputs"
            graph_path, pool_path = inputs / "network.pkl", inputs / "transactions.pkl"
            if graph_path.is_file():
                with graph_path.open("rb") as handle:
                    graph = pickle.load(handle)
            else:
                graph = core.LNGraphBuilder(
                    default_capacity=self.cfg["channel"].get("default_capacity")).from_data(
                        core.geo_to_json(inputs / "network_snapshot.gml.geo"))
                atomic_pickle(graph_path, graph)
            if pool_path.is_file():
                with pool_path.open("rb") as handle:
                    training, held_out = pickle.load(handle)
            else:
                from Runtime.paper_protocol import _load_transaction_manifest
                manifest = inputs / "transaction_split.csv"
                if manifest.is_file():
                    pool = _load_transaction_manifest(manifest, graph, 7000, 10000)
                else:
                    pool = core.generate_transactions(graph, 10000, int(self.cfg["seed"]),
                        self.cfg["simulation"]["min_amount"], self.cfg["simulation"]["max_amount"])
                    _write_transaction_manifest(manifest, pool, 7000)
                training, held_out = core.split_transaction_pool(pool, 3000)
                atomic_pickle(pool_path, (training, held_out))
            if len(training) != 7000 or len(held_out) != 3000:
                raise RuntimeError(f"{self.label} requires exactly 7,000 train and 3,000 held-out transactions.")
            atomic_json(inputs / "data_manifest.json", {
                name: file_hash(inputs / name)
                for name in ("network.pkl", "transactions.pkl", "transaction_split.csv")})
            atomic_json(self.run_dir / "run_status.json", {"status": "RUNNING", "stage": "TRAINING"})
            model_path, reference_path, checkpoint = self._train(core, graph, training, pool_path, graph_path)
            self._finish_training_phase(checkpoint)
        except BaseException as exc:
            error_path = self.run_dir / "logs/error.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            status_path = self.run_dir / "run_status.json"
            status = json.loads(status_path.read_text()) if status_path.is_file() else {}
            status.update(status="FAILED", error=str(exc), round=self.round_number)
            atomic_json(status_path, status)
            raise
        finally:
            try:
                self.publish("RUN FINISHED OR INTERRUPTED", True)
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
        return self.run_dir
