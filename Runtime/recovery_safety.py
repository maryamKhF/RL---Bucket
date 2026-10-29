"""Small standard-library helpers; importing this module never starts training."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import time
from urllib.parse import quote, urlsplit
import zipfile

GIB = 1024 ** 3


class MemoryStop(RuntimeError):
    """An operational stop; not a failed scientific trial."""


class RecoveryUploadError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"Recovery package upload returned HTTP {status}.")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def memory_sample(pid=None):
    """Cgroup usage includes children and charged page cache; RSS is a fallback."""
    pairs = [
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max"), "cgroup_v2"),
        (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), "cgroup_v1"),
    ]
    for usage, limit, source in pairs:
        if usage.is_file():
            raw_limit = limit.read_text().strip() if limit.is_file() else "max"
            sample = {"bytes": int(usage.read_text()), "source": source,
                      "limit_bytes": int(raw_limit) if raw_limit.isdecimal() else None}
            try:
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        sample["host_available_bytes"] = int(line.split()[1]) * 1024
                        break
                events = usage.parent / "memory.events"
                if events.is_file():
                    sample["events"] = {key: int(value) for key, value in
                                        (line.split() for line in events.read_text().splitlines())}
                pressure = usage.parent / "memory.pressure"
                if pressure.is_file():
                    for line in pressure.read_text().splitlines():
                        if line.startswith("full "):
                            sample["pressure_full_avg10"] = float(dict(
                                item.split("=") for item in line.split()[1:])["avg10"])
            except (OSError, ValueError, KeyError):
                pass
            return sample
    import psutil
    try:
        parent = psutil.Process(pid or os.getpid())
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        processes = []
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {"bytes": total, "source": "process_tree_rss_estimate", "limit_bytes": None,
            "host_available_bytes": psutil.virtual_memory().available}


def memory_stop_reason(sample, soft, host_floor, previous_events=None):
    """Request saving early under host/cgroup pressure, even below the soft cap."""
    before = previous_events or {}
    if any(sample.get("events", {}).get(name, 0) > before.get(name, 0)
           for name in ("oom", "oom_kill")):
        return "cgroup_oom_event"
    limit = sample.get("limit_bytes")
    effective_soft = min(soft, max(0, limit / GIB - 5)) if limit else soft
    if sample["bytes"] >= effective_soft * GIB:
        return "memory_headroom"
    if sample.get("host_available_bytes", float("inf")) < host_floor * GIB:
        return "host_low_available_memory"
    if sample.get("pressure_full_avg10", 0) >= 10:
        return "sustained_cgroup_memory_pressure"
    return None


def release_memory():
    """Release unreachable graphs; malloc_trim is optional and changes no state."""
    gc.collect()
    if os.name == "posix":
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass


def stop_requested(directory):
    return (Path(directory) / "stop_requested.json").is_file()


def check_stop(directory):
    if stop_requested(directory):
        raise MemoryStop("Supervisor requested a safe stop; recovery files are preserved.")


def build_bundle(run_dir, destination):
    """A consistent SQLite backup plus closed/atomic files, streamed to ZIP."""
    run_dir, destination = Path(run_dir).resolve(), Path(destination).resolve()
    if destination.is_relative_to(run_dir):
        raise ValueError("Recovery archives must be outside their source directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    db = run_dir / "results.sqlite3"
    db_copy = destination.with_suffix(".sqlite3.tmp")
    temporary = destination.with_suffix(".zip.tmp")
    try:
        if db.is_file():
            source = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=10)
            target = sqlite3.connect(db_copy)
            try:
                source.backup(target, pages=128, sleep=0.01)
            finally:
                target.close()
                source.close()
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as archive:
            for path in sorted(run_dir.rglob("*")):
                if (not path.is_file() or path.is_symlink() or path.name.endswith(".tmp")
                        or path.name == ".pending_snapshot.zip"):
                    continue
                if path.name in ("results.sqlite3", "results.sqlite3-wal", "results.sqlite3-shm"):
                    continue
                # This cache is reproducible from the snapshot; omit its large duplicate.
                if path.name == "snapshot_graph.pkl":
                    continue
                archive.write(path, f"theResult/{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
            if db_copy.is_file():
                archive.write(db_copy, f"theResult/{run_dir.name}/results.sqlite3")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
        return destination
    finally:
        db_copy.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def upload_file(url, path, token):
    """Upload bounded chunks, with TLS verification and no token in process args."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Recovery uploads require a credential-free HTTPS URL.")
    connection = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=120)
    try:
        connection.putrequest("PUT", parsed.path + ("?" + parsed.query if parsed.query else ""))
        connection.putheader("JOB-TOKEN", token)
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(Path(path).stat().st_size))
        connection.endheaders()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                connection.send(chunk)
        response = connection.getresponse()
        response.read(1024)
        if response.status not in (200, 201):
            raise RecoveryUploadError(response.status)
    finally:
        connection.close()


def upload_with_retry(url, path, token):
    """Retry transient transport/server failures; keep authorization failures visible."""
    for attempt in range(3):
        try:
            upload_file(url, path, token)
            return
        except (OSError, http.client.HTTPException, RecoveryUploadError) as error:
            transient = not isinstance(error, RecoveryUploadError) or error.status == 429 or error.status >= 500
            if not transient or attempt == 2:
                raise
            delay = 2 ** (attempt + 1)
            print(f"[RECOVERY RETRY] attempt={attempt+1}/3; {type(error).__name__}: {error}; "
                  f"retrying in {delay} seconds.", flush=True)
            time.sleep(delay)


def verified_recovery(directory):
    """Find the newest complete generation; retain a fallback if a pointer is damaged."""
    directory = Path(directory).resolve()
    for filename in ("latest_recovery.json", "previous_recovery.json"):
        try:
            record = json.loads((directory / filename).read_text(encoding="utf-8"))
            name = str(record.get("local_archive", "latest_recovery.zip")).replace("\\", "/").rsplit("/", 1)[-1]
            candidate = (directory / name).resolve()
            if (candidate.parent == directory and candidate.is_file() and not record.get("error")
                    and file_hash(candidate) == record.get("sha256")):
                return candidate, record
        except (OSError, ValueError, TypeError):
            continue
    return None


def publish_local_generation(directory, pending, record):
    """Switch verified pointers only after upload success; keep two generations."""
    directory = Path(directory).resolve()
    generation = directory / f"verified_recovery_{time.time_ns()}.zip"
    Path(pending).replace(generation)
    record["local_archive"] = str(generation)
    previous = verified_recovery(directory)
    if previous:
        atomic_json(directory / "previous_recovery.json", previous[1])
    atomic_json(directory / "latest_recovery.json", record)
    # A legacy alias helps manual downloads. Recovery uses the verified pointer,
    # so interruption while replacing this alias cannot invalidate the saved copy.
    alias_temporary = directory / ".latest_recovery.zip.tmp"
    try:
        alias_temporary.unlink(missing_ok=True)
        try:
            os.link(generation, alias_temporary)
        except OSError:
            with generation.open("rb") as source, alias_temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        alias_temporary.replace(directory / "latest_recovery.zip")
    except OSError as error:
        print(f"[RECOVERY ALIAS WARNING] {error}; verified generation remains at {generation}", flush=True)
    finally:
        try:
            alias_temporary.unlink(missing_ok=True)
        except OSError as error:
            print(f"[RECOVERY CLEANUP WARNING] {error}", flush=True)
    keep = {generation.name}
    if previous:
        keep.add(previous[0].name)
    for old in directory.glob("verified_recovery_*.zip"):
        # Only our reserved generation files in this exact output directory
        # are eligible; never follow a symlink outside the recovery directory.
        if old.resolve().parent == directory and not old.is_symlink() and old.name not in keep:
            try:
                old.unlink()
            except OSError as error:
                print(f"[RECOVERY CLEANUP WARNING] {error}", flush=True)


class RecoveryPublisher:
    """Keep local bundles and, in CI, publish them before the job finishes."""

    def __init__(self, run_dir, interval=300):
        self.run_dir = Path(run_dir).resolve()
        self.interval = interval
        self.last = 0.0
        self.remote = os.environ.get("MAIN8_REMOTE_BACKUP", "0") == "1"
        self.required = os.environ.get("MAIN8_BACKUP_REQUIRED", "0") == "1"
        self.version = os.environ.get("MAIN8_EXPERIMENT_ID") or os.environ.get("CI_PIPELINE_ID", "local")

    def publish(self, reason, force=False):
        if not force and time.monotonic() - self.last < self.interval:
            return
        if not (self.run_dir / "run_metadata.json").is_file():
            return
        directory = self.run_dir.parent / "recovery"
        path = build_bundle(self.run_dir, directory / "pending_recovery.zip")
        record = {"reason": reason, "created_unix": time.time(), "bytes": path.stat().st_size,
                  "sha256": file_hash(path), "local_archive": str(path), "remote_saved": False}
        try:
            if self.remote:
                api = os.environ["CI_API_V4_URL"].rstrip("/")
                project = os.environ["CI_PROJECT_ID"]
                token = os.environ["CI_JOB_TOKEN"]
                name = f"recovery-{time.time_ns()}.zip"
                url = (f"{api}/projects/{project}/packages/generic/rl-bucket-main8/"
                       f"{quote(self.version, safe='')}/{name}")
                upload_with_retry(url, path, token)
                record.update(remote_saved=True, download_url=url)
            elif self.required:
                raise RuntimeError("Remote recovery is required but MAIN8_REMOTE_BACKUP is disabled.")
            self.last = time.monotonic()
            publish_local_generation(directory, path, record)
            with (self.run_dir / "recovery_uploads.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"[RECOVERY SAVED] {reason}; remote={record['remote_saved']}; "
                  f"bytes={record['bytes']}; " + record.get("download_url", record["local_archive"]), flush=True)
        except Exception as error:
            record["error"] = str(error)
            # Never replace the last successful pointer after a failed upload.
            atomic_json(directory / "pending_recovery.json", record)
            if self.required:
                raise RuntimeError(f"Durable recovery failed ({type(error).__name__}: {error}); "
                                   "stopping instead of continuing unprotected.") from error
            print(f"[RECOVERY WARNING] {error}; local archive remains at {path}", flush=True)


def restore_bundle(archive_path, workspace):
    """Restore one run without overwriting sources or an existing experiment."""
    workspace = Path(workspace).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len({entry.filename for entry in members}) != len(members):
            raise ValueError("Recovery archive contains duplicate paths.")
        roots = set()
        for entry in members:
            parts = PurePosixPath(entry.filename).parts
            if (len(parts) < 3 or parts[0] != "theResult" or ".." in parts
                    or "\\" in entry.filename or ":" in entry.filename
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Invalid recovery archive member.")
            roots.add(parts[1])
        if len(roots) != 1:
            raise ValueError("Recovery archive must contain exactly one experiment.")
        run_dir = workspace / "theResult" / roots.pop()
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {run_dir}")
        for entry in members:
            target = (workspace / entry.filename).resolve()
            if not target.is_relative_to(run_dir):
                raise ValueError("Recovery archive escapes its run directory.")
        # Validate paths before extracting anything; source files are never restored.
        for entry in members:
            target = workspace / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if not entry.is_dir():
                with archive.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        return run_dir
