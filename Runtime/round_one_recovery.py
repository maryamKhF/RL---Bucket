"""Consistent PPO update checkpoints and durable held-out payment commits."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import random
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile

from Runtime.recovery_safety import atomic_json, file_hash


SCHEMA = 1


def runtime_versions():
    from importlib.metadata import version
    import platform
    return {"python": ".".join(platform.python_version_tuple()[:2]),
            **{name: version(name) for name in ("torch", "numpy", "networkx", "stable-baselines3")}}


def capture_rng(include_cuda=True):
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if include_cuda and torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verified_checkpoint(run_dir):
    directory = Path(run_dir) / "recovery"
    for pointer in ("latest.json", "previous.json"):
        try:
            record = json.loads((directory / pointer).read_text(encoding="utf-8"))
            checkpoint = directory / record["generation"]
            if checkpoint.resolve().parent != directory.resolve() or checkpoint.is_symlink():
                continue
            if all((checkpoint / name).resolve().is_relative_to(checkpoint.resolve())
                   and (checkpoint / name).is_file()
                   and file_hash(checkpoint / name) == digest for name, digest in record["files"].items()):
                state = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
                if state.get("schema") == SCHEMA:
                    return checkpoint, state
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def safe_load_policy(path, env=None, device="cpu"):
    from Evaluations.explainable_routing import load_policy
    return load_policy(path, env=env, device=device, force_reset=False)


def load_next_round_policy(path, env=None, device="cpu", *, seed):
    """Keep learned actor/critic, optimizer and counters; start the new env safely."""
    from Evaluations.explainable_routing import load_policy
    model = load_policy(path, env=env, device=device, force_reset=True)
    model.seed = int(seed)
    model.set_random_seed(model.seed)
    return model


def training_callback(cfg, directory, progress, run_dir, publish, checkpoint=None,
                      checkpoint_seconds=1800):
    """Build lazily so package/restore commands do not import PyTorch."""
    from Evaluations.explainable_routing import LearningAuditCallback
    from RL.state import FEATURE_NAMES
    import numpy as np

    class ResumableAudit(LearningAuditCallback):
        def __init__(self):
            super().__init__(cfg, directory, tasks=[], store_rows=lambda rows: None,
                             progress=progress, diagnostic_interval=int(cfg["rl"]["total_timesteps"]) + 1)
            self.last_saved_step = None
            self.last_saved_time = time.monotonic()
            self.restored_rng = None
            if checkpoint:
                with (checkpoint[0] / "audit_rng.pkl").open("rb") as handle:
                    saved = pickle.load(handle)
                self.feature_sum = np.asarray(saved["feature_sum"], dtype=np.float64)
                self.feature_count = int(saved["feature_count"])
                self.restored_rng = saved["rng"]

        def _on_training_start(self):
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / "training_explanations.jsonl.gz"
            if checkpoint:
                # Discard only the interrupted rollout's authoritative trace;
                # preserve it separately even if its last gzip member is torn.
                if path.exists():
                    path.replace(path.with_name(f"discarded_rollout_{time.time_ns()}.jsonl.gz"))
                shutil.copy2(checkpoint[0] / "training_audit.jsonl.gz", path)
                self.last_saved_step = int(checkpoint[1]["step"])
            self.trace = gzip.open(path, "at" if checkpoint else "wt", encoding="utf-8", compresslevel=3)
            if self.restored_rng is not None:
                restore_rng(self.restored_rng)
            if not checkpoint:
                self.save_consistent_checkpoint()

        def _on_rollout_start(self):
            super()._on_rollout_start()
            if (self.last_saved_step != int(self.model.num_timesteps)
                    and time.monotonic() - self.last_saved_time >= checkpoint_seconds):
                self.save_consistent_checkpoint()

        def _on_step(self):
            return super()._on_step()

        def save_consistent_checkpoint(self):
            step = int(self.model.num_timesteps)
            if self.last_saved_step == step:
                return
            started = time.monotonic()
            directory = Path(run_dir) / "recovery"
            directory.mkdir(exist_ok=True)
            name = f"step_{step}_{uuid.uuid4().hex}"
            pending = directory / (".pending-" + name)
            pending.mkdir()
            rng = capture_rng()
            self.trace.close()
            self.trace = None
            try:
                self.model.save(str(pending / "model.zip"))
                # Workers write directly to shared disk, avoiding giant IPC
                # messages and preserving their private simulation/RNG states.
                self.training_env.env_method("save_resume_state", str(pending))
                shutil.copy2(self.directory / "training_explanations.jsonl.gz",
                             pending / "training_audit.jsonl.gz")
                atomic_pickle(pending / "audit_rng.pkl", {
                    "feature_sum": self.feature_sum, "feature_count": self.feature_count, "rng": rng})
                atomic_json(pending / "checkpoint.json", {
                    "schema": SCHEMA, "step": step, "workers": self.training_env.num_envs,
                    "n_steps": self.model.n_steps, "batch_size": self.model.batch_size,
                    "features": list(FEATURE_NAMES), "observed_slots": self.feature_count,
                    "requested_target": int(cfg["rl"]["total_timesteps"]),
                    "runtime_versions": runtime_versions(),
                    "boundary": "after completed PPO update; no unoptimized rollout is committed",
                })
                final = directory / name
                pending.replace(final)
                for path in final.iterdir():
                    if path.is_file():
                        with path.open("r+b") as handle:
                            os.fsync(handle.fileno())
                files = {path.name: file_hash(path) for path in final.iterdir() if path.is_file()}
                if len(list(final.glob("worker_*.pkl.gz"))) != self.training_env.num_envs:
                    raise RuntimeError("Incomplete worker-state checkpoint.")
                previous = verified_checkpoint(run_dir)
                if previous:
                    atomic_json(directory / "previous.json", {
                        "generation": previous[0].name,
                        "files": {path.name: file_hash(path) for path in previous[0].iterdir() if path.is_file()},
                    })
                atomic_json(directory / "latest.json", {"generation": name, "files": files})
                self.last_saved_step = step
                self.last_saved_time = time.monotonic()
                # Retain two committed generations; incomplete writes never
                # replace either verified recovery pointer.
                keep = {name, previous[0].name if previous else name}
                for old in directory.glob("step_*"):
                    if (old.is_dir() and not old.is_symlink() and old.resolve().parent == directory.resolve()
                            and old.name not in keep):
                        shutil.rmtree(old)
            finally:
                restore_rng(rng)
                self.trace = gzip.open(self.directory / "training_explanations.jsonl.gz",
                                       "at", encoding="utf-8", compresslevel=3)
            size = sum(p.stat().st_size for p in final.iterdir() if p.is_file())
            print(f"[CHECKPOINT] PPO step={step}; bytes={size}; "
                  f"seconds={time.monotonic() - started:.2f}; local coherent recovery saved", flush=True)

        def _on_training_end(self):
            try:
                self.save_consistent_checkpoint()
            finally:
                self.close()

    return ResumableAudit()


def open_evaluation_journal(path, identity, *, wal=False):
    connection = sqlite3.connect(path, timeout=30)
    if wal:
        mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            connection.close()
            raise RuntimeError("Durable evaluation WAL journal is unavailable on this filesystem.")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS payments (seed INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
    encoded = json.dumps(identity, sort_keys=True)
    previous = connection.execute("SELECT value FROM identity").fetchone()
    if previous and previous[0] != encoded:
        connection.close()
        raise ValueError("Evaluation journal belongs to a different policy, reference or trial protocol.")
    if not previous:
        connection.execute("INSERT INTO identity VALUES (?)", (encoded,))
    connection.commit()
    return connection


def restore_round_archive(archive_path, output_dir, *, round_number=1, _wrapped=False):
    """Accept our ZIP or a GitLab artifact ZIP; never overwrite a checkout."""
    output_dir = Path(output_dir).resolve()
    archive_path = Path(archive_path)
    prefix = f"main{round_number}_5_"
    checksum = archive_path.with_suffix(".zip.sha256")
    if checksum.is_file() and checksum.read_text().split()[0] != file_hash(archive_path):
        raise ValueError("Recovery ZIP checksum mismatch.")
    with zipfile.ZipFile(archive_path) as saved:
        entries = saved.infolist()
        if len({entry.filename for entry in entries}) != len(entries):
            raise ValueError("Duplicate archive paths.")
        roots = set()
        for entry in entries:
            parts = PurePosixPath(entry.filename).parts
            if (PurePosixPath(entry.filename).is_absolute() or ".." in parts
                    or "\\" in entry.filename or ":" in entry.filename
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe recovery archive path.")
            if parts and parts[-1] == "run_metadata.json":
                roots.add(parts[:-1])
        if not roots and not _wrapped:
            # Runner artifacts wrap the single compact ZIP, not a second copy
            # of every run file. Accept that wrapper as well as legacy folders.
            bundles = [entry for entry in entries if not entry.is_dir()
                       and PurePosixPath(entry.filename).name.startswith(prefix)
                       and entry.filename.endswith(".zip")]
            if len(bundles) == 1:
                with tempfile.TemporaryDirectory(prefix=f"round{round_number}_restore_") as temporary:
                    inner = Path(temporary) / "run.zip"
                    with saved.open(bundles[0]) as source, inner.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    digest_name = bundles[0].filename + ".sha256"
                    if digest_name in saved.namelist():
                        digest = saved.read(digest_name).decode("ascii").split()
                        if not digest or digest[0] != file_hash(inner):
                            raise ValueError("Recovery ZIP checksum mismatch.")
                    return restore_round_archive(inner, output_dir, round_number=round_number, _wrapped=True)
        if len(roots) != 1:
            raise ValueError(f"Select an archive containing exactly one main{round_number}/5 run.")
        root = roots.pop()
        if not root or not root[-1].startswith(prefix):
            raise ValueError(f"Not a main{round_number}/5 archive.")
        destination = output_dir / root[-1]
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing run: {destination}")
        selected = [(entry, PurePosixPath(entry.filename).parts[len(root):]) for entry in entries
                    if PurePosixPath(entry.filename).parts[:len(root)] == root]
        # Validate every target before creating files.
        if any(not destination.joinpath(*suffix).resolve().is_relative_to(destination)
               for _, suffix in selected):
            raise ValueError("Archive escapes the destination.")
        for entry, suffix in selected:
            target = destination.joinpath(*suffix)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with saved.open(entry) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
        # Direct atomic bundles have a consistent whole-run manifest. A raw
        # GitLab directory can contain newer files than its last periodic ZIP;
        # resume independently verifies immutable inputs and safe checkpoints.
        if len(root) == 1:
            manifest = json.loads((destination / "artifact_manifest.json").read_text())
            for name, details in manifest["files"].items():
                path = (destination / name).resolve()
                if not path.is_relative_to(destination) or file_hash(path) != details["sha256"]:
                    raise ValueError("Recovery archive content checksum mismatch.")
        return destination


def download_job_artifacts(job_id, output_dir):
    """Used by the explicit CI resume job; credentials never cross storage redirects."""
    import importlib
    from urllib.request import Request, build_opener
    from urllib.error import HTTPError
    if not str(job_id).isdecimal():
        raise ValueError("A numeric GitLab source job ID is required.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"job_{job_id}_{time.time_ns()}.zip"
    temporary = path.with_suffix(".zip.tmp")
    token = os.environ.get("GITLAB_ARTIFACT_TOKEN")
    headers = {"PRIVATE-TOKEN": token} if token else {"JOB-TOKEN": os.environ["CI_JOB_TOKEN"]}
    url = (f"{os.environ['CI_API_V4_URL'].rstrip('/')}/projects/"
           f"{os.environ['CI_PROJECT_ID']}/jobs/{job_id}/artifacts")
    legacy_entry = Path(__file__).resolve().parent.parent / "main-test-resume.py"
    module_name = "main-test-resume" if legacy_entry.is_file() else "main-resume"
    handler = importlib.import_module(module_name).ArtifactRedirectHandler
    try:
        with build_opener(handler()).open(Request(url, headers=headers), timeout=120) as response, \
                temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(path)
    except HTTPError as error:
        raise RuntimeError(f"GitLab artifact download failed (HTTP {error.code}); check source job access/expiry.") from None
    return path
