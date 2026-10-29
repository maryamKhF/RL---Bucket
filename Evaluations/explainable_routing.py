"""Observable-input explanations and isolated diagnostics for routing experiments.

Feature occlusion measures policy sensitivity, not causal responsibility for
payment success. Simulator probabilities and true balances never enter it.
"""

from __future__ import annotations

import copy
import gzip
import io
import json
import pickle
import random
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from Pathfinding.heuristics import adaptive_edge_cost, lnd_cost
from RL.environment import RoutingEnv
from RL.state import FEATURE_NAMES
from Simulation.failure_model import FailureModel, assign_failure_probabilities
from Simulation.network_dynamics import NetworkDynamics
from Runtime.recovery_safety import (MemoryStop, atomic_json, check_stop, file_hash,
                                  release_memory, stop_requested)
from Evaluations.xai_report import human_explanation
from Evaluations.xai_sampling import policy_sample
from Runtime.recovery_reporting import recovery_details


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_policy(path, env=None, device="cpu", force_reset=True):
    # Read trusted project checkpoints through BytesIO, avoiding main's report side effects.
    from stable_baselines3.common import save_util
    from unittest.mock import patch
    def load_zip(load_path, device="cpu", custom_objects=None, load_data=True,
                 print_system_info=False, **kwargs):
        del print_system_info, kwargs
        with zipfile.ZipFile(load_path) as archive:
            data = save_util.json_to_data(archive.read("data").decode("utf-8"),
                                         custom_objects=custom_objects) if load_data else None
            params, variables = {}, None
            for name in archive.namelist():
                if name.endswith(".pth"):
                    value = torch.load(io.BytesIO(archive.read(name)),
                                       map_location=save_util.get_device(device),
                                       weights_only=False)
                    if name in ("pytorch_variables.pth", "tensors.pth"):
                        variables = value
                    else:
                        params[name[:-4]] = value
            return data, params, variables
    with patch("stable_baselines3.common.base_class.load_from_zip_file",
               load_zip):
        return PPO.load(str(path), env=env, device=device, force_reset=force_reset)


def explain_policy(model, observation, reference):
    """Replace one feature group in observed slots by its training-only mean."""
    return explain_policy_batch(model, [observation], reference)[0]


def explain_policy_batch(model, observations, reference):
    """One inference batch for all selected workers, preserving per-decision results."""
    if len(observations) == 0:
        return []
    observations = np.asarray(observations, dtype=np.float32)
    d = len(FEATURE_NAMES)
    if observations.ndim != 2 or observations.shape[1] % (d + 1):
        raise ValueError("Explanation requires the State matrix plus channel mask.")
    k = observations.shape[1] // (d + 1)
    present = observations[:, k * d:] > 0.5
    ref = np.asarray(reference, dtype=np.float32)
    if ref.shape != (d,) or not np.isfinite(ref).all():
        raise ValueError("Invalid training reference for feature occlusion.")
    batch = np.repeat(observations[:, None, :], d + 1, axis=1)
    for feature in range(d):
        matrix = batch[:, feature + 1, :k * d].reshape(len(observations), k, d)
        matrix[:, :, feature][present] = ref[feature]
    batch = batch.reshape(-1, observations.shape[1])
    actions, _ = model.predict(batch, deterministic=True)
    eta = np.asarray(actions, dtype=float).reshape(len(observations), d + 1, -1)[:, :, 0]
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(batch)
        values = model.policy.predict_values(tensor).cpu().numpy().reshape(len(observations), d + 1)
        distribution = model.policy.get_distribution(tensor).distribution
        action_std = distribution.stddev.reshape(len(observations), d + 1, -1)[:, 0, 0].cpu().numpy()
        unclipped_mean = distribution.mean.reshape(len(observations), d + 1, -1)[:, 0, 0].cpu().numpy()
    return [{
        "method": "group_feature_occlusion",
        "reference": "mean of observed channel features from training rollouts only",
        "eta": float(eta[index, 0]),
        "critic_value": float(values[index, 0]),
        "unclipped_normal_action_mean": float(unclipped_mean[index]),
        "normal_action_std": float(action_std[index]),
        "observed_channel_slots": int(present[index].sum()),
        "eta_delta": dict(zip(FEATURE_NAMES, (eta[index, 0] - eta[index, 1:]).tolist())),
        "value_delta": dict(zip(FEATURE_NAMES, (values[index, 0] - values[index, 1:]).tolist())),
        "interpretation": "sensitivity in normalized input space; not causal or SHAP attribution",
    } for index in range(len(observations))]


class ExplainableRoutingEnv(RoutingEnv):
    """The same routing environment with an observational audit trail."""

    def __init__(self, *args, worker_rank=0, worker_count=1, **kwargs):
        self.worker_rank = worker_rank
        self.worker_count = worker_count
        self._audit = {"candidate_searches": [], "payment_attempts": []}
        self._diagnostic_model = None
        self._diagnostic_graph = None
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        started = time.perf_counter()
        observation, info = super().reset(**kwargs)
        info["reset_seconds"] = time.perf_counter() - started
        return observation, info

    def step(self, action):
        started = time.perf_counter()
        cpu_started = time.process_time()
        self._audit = {"candidate_searches": [], "payment_attempts": []}
        tx = self.current_tx
        self._audit["payment"] = {"transaction_id": getattr(tx, "tx_id", None),
            "source": str(getattr(tx, "source", "")), "destination": str(getattr(tx, "destination", "")),
            "amount": float(getattr(tx, "amount", 0))}
        result = super().step(action)
        observation, reward, terminated, truncated, info = result
        success = bool(info.get("payment_success", info.get("success", False)))
        distance = int(info.get("path_length", 0)) + 1
        carbon_mean = float(info.get("carbon", 0.0))
        self._audit["reward"] = {
            "formula": "S * D * 1000 / C",
            "S": int(success), "D_route_nodes": distance,
            "C_route_sum": carbon_mean * distance if success else None,
            "carbon_mean": carbon_mean,
            "actual_reward": float(reward),
            "note": "D cancels when C = D * mean carbon; failed payments receive zero",
        }
        self._audit["decision"] = {
            key: info.get(key) for key in (
                "eta", "success", "reason", "path", "path_length",
                "successful_bucket_rank", "successful_route_source", "attempt_count",
                "backtrack_count", "partial_backtrack_count", "partial_backtrack_success", "full_reroute_count",
                "routing_scope", "local_routing_radius", "local_routing_node_count",
                "path_search_timed_out", "path_search_timeout_stages")
        }
        self._audit["decision"].update(recovery_details(info))
        events = {event.get("attempt_id"): event for event in info.get("recovery_events", [])
                  if event.get("type") in ("partial_backtrack", "source_retry")}
        for attempt_id, attempt in enumerate(self._audit["payment_attempts"], 1):
            prefix = int(events.get(attempt_id, {}).get("branch_index", 0))
            attempt["attempt_id"] = attempt_id
            attempt["preserved_prefix_hops"] = prefix
            attempt["forwarded_channels"] = [
                item["channel"] for item in attempt["channels"][prefix:]]
        info["explanation"] = self._audit
        info["performance"] = {
            "step_seconds": time.perf_counter() - started,
            "step_cpu_seconds": time.process_time() - cpu_started,
        }
        return observation, reward, terminated, truncated, info

    def _generate_candidates(self, tx, eta):
        candidates = super()._generate_candidates(tx, eta)
        self._audit["candidate_searches"].append([
            {**{key: candidate.get(key) for key in (
                "path", "cost", "score", "total_fee", "total_delay")},
             "channels": [self._normalize_edge(edge) for edge in candidate.get("edges", [])]}
            for candidate in candidates
        ])
        return candidates

    def _search_recovery_suffixes(self, tx, eta, route, branch_index,
                                  failed_edges, failed_nodes):
        candidates = super()._search_recovery_suffixes(
            tx, eta, route, branch_index, failed_edges, failed_nodes)
        self._audit.setdefault("recovery_searches", []).append({
            "branch_node": route[branch_index], "branch_index": branch_index,
            "preserved_prefix": list(route[:branch_index + 1]),
            "excluded_channels": list(failed_edges), "excluded_nodes": list(failed_nodes),
            "candidates": [{"path": item["path"], "cost": item["cost"],
                            "channels": [self._normalize_edge(e) for e in item["edges"]]}
                           for item in candidates],
        })
        return candidates

    def _create_bucket(self, tx, candidates):
        bucket, filtered = super()._create_bucket(tx, candidates)
        self._audit.setdefault("buckets", []).append({
            "bucket_id": bucket.bucket_id if bucket else None,
            "ordering": "ascending routing cost; exact physical channel identities",
            "candidates": [{"rank": rank, "path": item["path"], "cost": item.get("cost"),
                            "channels": [self._normalize_edge(e) for e in item["edges"]]}
                           for rank, item in enumerate(bucket.candidates if bucket else [], 1)],
        })
        return bucket, filtered

    def _learn_liquidity_feedback(self, result, edges, amount):
        # Record only outcomes observable after an attempt, never latent balances.
        details = []
        eta = float(self._audit.get("eta", 0.0))
        for edge in edges:
            u, v, key = self._normalize_edge(edge)
            data = self.G[u][v][key] if self.G.is_multigraph() else self.G[u][v]
            cost = adaptive_edge_cost(self.G, u, v, data, amount, eta,
                                      heuristic_fn=self.heuristic_fn)
            details.append({"channel": [u, v, key], **cost})
        self._audit["payment_attempts"].append({
            "channels": details,
            "success": bool(result.get("success")),
            "reason": result.get("reason"),
            "failed_edge": self._normalize_edge(result.get("failed_edge")),
            "visited_edges": [self._normalize_edge(edge)
                              for edge in (result.get("visited_edges") or [])],
        })
        super()._learn_liquidity_feedback(result, edges, amount)

    def _decode_eta(self, action):
        eta = super()._decode_eta(action)
        self._audit["eta"] = eta
        return eta

    def evaluate_snapshot(self, model_path, tasks, reference, cfg):
        """Use idle training workers for diagnostics without changing training state."""
        self.begin_diagnostics(model_path, cfg)
        try:
            return self.diagnostic_batch(tasks, reference, cfg)
        finally:
            self.end_diagnostics()

    def begin_diagnostics(self, model_path, cfg, diagnostic_worker_count=None):
        with preserve_rng():
            self._diagnostic_worker_count = diagnostic_worker_count or self.worker_count
            self._diagnostic_model = load_policy(model_path)
            with Path(cfg["_main8_graph_cache"]).open("rb") as handle:
                self._diagnostic_graph = pickle.load(handle)

    def diagnostic_batch(self, tasks, reference, cfg):
        with preserve_rng():
            return [evaluate_payment(self._diagnostic_graph, cfg, self._diagnostic_model, task, reference)
                    for task in tasks[self.worker_rank::self._diagnostic_worker_count]]

    def end_diagnostics(self):
        self._diagnostic_model = self._diagnostic_graph = None
        release_memory()

    def close(self):
        self.end_diagnostics()
        return super().close()

    def save_resume_state(self, directory):
        """Save the private simulator state separately from public XAI logs."""
        import cloudpickle
        import os
        from Runtime.round_one_recovery import capture_rng, restore_rng
        path = Path(directory) / f"worker_{self.worker_rank:03d}.pkl.gz"
        temporary = path.with_name(path.name + ".tmp")
        rng = capture_rng(include_cuda=False)
        try:
            with gzip.open(temporary, "wb", compresslevel=1) as handle:
                cloudpickle.dump({"env": self, "rng": rng}, handle)
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            restore_rng(rng)
        return path.name


@contextmanager
def preserve_rng():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def evaluate_payment(base_graph, cfg, model, task, reference, *, graph_snapshot=None):
    """A deterministic case identity independent of process count and completion order."""
    setup_started = time.perf_counter()
    graph = pickle.loads(graph_snapshot) if graph_snapshot is not None else copy.deepcopy(base_graph)
    assign_failure_probabilities(graph, task["failure_rate"], task["scenario_seed"])
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg["seed"] = task["payment_seed"]
    runtime_cfg.setdefault("simulation", {})["seed"] = task["payment_seed"]
    tx = copy.deepcopy(task["transaction"])
    dynamics = NetworkDynamics(graph, seed=task["payment_seed"],
                              default_capacity=cfg.get("channel", {}).get("default_capacity"))
    env = ExplainableRoutingEnv(
        G=graph, transactions=[tx], heuristic_fn=lnd_cost, config=runtime_cfg,
        mode="eval", failure_model=FailureModel(seed=task["payment_seed"]),
        network_dynamics=dynamics,
    )
    try:
        start = time.perf_counter()
        obs, _ = env.reset(seed=task["payment_seed"])
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = env.step(action)
        policy = policy_sample(cfg, ["evaluation", task["failure_rate"], task["payment_seed"]],
                               info["explanation"])
        if policy["sampled"]:
            policy.update(explain_policy(model, obs, reference))
        row = {key: info.get(key) for key in (
            "eta", "fee", "delay", "carbon", "path_length", "attempt_count",
            "backtrack_count", "partial_backtrack_count", "partial_backtrack_success", "full_reroute_count",
            "successful_bucket_rank", "successful_route_source", "reason",
            "routing_scope", "local_routing_radius", "local_routing_node_count",
            "local_routing_channel_count", "path_search_timed_out",
            "path_search_elapsed_seconds")}
        row.update(recovery_details(info))
        row["path"] = info.get("path", [])
        row.update({key: task[key] for key in (
            "phase", "repetition", "training_seed", "checkpoint", "split",
            "failure_rate", "scenario_seed", "payment_seed")})
        row["evaluation_seed"] = task.get("evaluation_seed", task["training_seed"])
        row.update(transaction_id=int(tx.tx_id), source=str(tx.source),
                   destination=str(tx.destination), amount=float(tx.amount),
                   success=bool(info.get("success", False)), reward=float(reward),
                   pipeline_seconds=time.perf_counter() - start,
                   explanation={"policy": policy, "normalized_observation": obs.tolist(),
                                "routing": info["explanation"]})
        row["performance"] = {**info.get("performance", {}),
                              "setup_seconds": start - setup_started,
                              "total_seconds": time.perf_counter() - setup_started}
        return row
    finally:
        env.close()
        del env, graph, dynamics
        release_memory()


class LearningAuditCallback(BaseCallback):
    """Audit every training decision; diagnose frozen policies after actual updates."""

    def __init__(self, cfg, directory, tasks, store_rows, progress,
                 diagnostic_interval=10000, recovery=None, stop_directory=None,
                 completed_cases=None):
        super().__init__()
        self.cfg = cfg
        self.directory = Path(directory)
        self.tasks = tasks
        self.store_rows = store_rows
        self.progress = progress
        self.interval = diagnostic_interval
        self.next_diagnostic = diagnostic_interval
        self.last_diagnostic = -1
        self.feature_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.feature_count = 0
        self.trace = None
        self.recovery = recovery
        self.stop_directory = Path(stop_directory) if stop_directory else None
        self.completed_cases = completed_cases
        self.backup_interval = float(cfg.get("_main8_backup_interval", 300))
        if not self.backup_interval > 0:
            raise ValueError("Recovery backup interval must be positive.")
        self.next_recovery = time.monotonic() + self.backup_interval

    @property
    def reference(self):
        return (self.feature_sum / max(1, self.feature_count)).astype(np.float32)

    def _on_training_start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.stop_directory and self.model.num_timesteps:
            self._rewind_trace(int(self.model.num_timesteps))
        elif self.stop_directory and (self.directory / "training_explanations.jsonl.gz").is_file():
            self._rewind_trace(0)
        self.trace = gzip.open(self.directory / "training_explanations.jsonl.gz",
                               "at", encoding="utf-8", compresslevel=3)
        self.next_diagnostic = (self.model.num_timesteps // self.interval + 1) * self.interval
        if not self.model.num_timesteps and self.stop_directory:
            # An initial checkpoint lets us detect stops before any PPO update.
            self._save_snapshot("resume")
        if self.model.num_timesteps and (not self.stop_directory or
                (self.directory / f"policy_step_{int(self.model.num_timesteps)}.zip").is_file()):
            self.diagnose()

    def _rewind_trace(self, checkpoint):
        # Discarded partial rollouts were never optimized. Keep their audit in a
        # separate file, while authoritative charts follow the resumed checkpoint.
        path = self.directory / "training_explanations.jsonl.gz"
        if not path.is_file():
            return
        temporary = path.with_name("training_trace_rewind.gz.tmp")
        removed = 0
        try:
            with gzip.open(path, "rt", encoding="utf-8") as source, \
                 gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as target:
                for line in source:
                    if int(json.loads(line)["timestep"]) <= checkpoint:
                        target.write(line)
                    else:
                        removed += 1
            if removed:
                path.replace(path.with_name(f"discarded_training_trace_{time.time_ns()}.jsonl.gz"))
                temporary.replace(path)
                print(f"[RESUME AUDIT] Rewound to completed PPO checkpoint {checkpoint}; "
                      f"{removed} interrupted rollout records preserved separately.", flush=True)
        finally:
            temporary.unlink(missing_ok=True)

    def _on_step(self):
        observations = self.locals["obs_tensor"].detach().cpu().numpy()
        d = len(FEATURE_NAMES)
        k = observations.shape[1] // (d + 1)
        for observation in observations:
            matrix = observation[:k * d].reshape(k, d)
            valid = observation[k * d:] > 0.5
            self.feature_sum += matrix[valid].sum(axis=0)
            self.feature_count += int(valid.sum())
        policies = [policy_sample(self.cfg,
                    ["training", int(info.get("audit_timestep", self.num_timesteps)), int(info.get("worker_rank", rank)), info.get("transaction_id"),
                     float(self.cfg.get("training_failure_rate", .03))], info["explanation"])
                    for rank, info in enumerate(self.locals["infos"])]
        selected = [rank for rank, policy in enumerate(policies) if policy["sampled"]]
        explanations = explain_policy_batch(self.model, observations[selected], self.reference)
        for rank, explanation in zip(selected, explanations):
            policies[rank].update(explanation)
        for rank, (obs, info) in enumerate(zip(observations, self.locals["infos"])):
            record = {"timestep": int(info.get("audit_timestep", self.num_timesteps)), "worker": int(info.get("worker_rank", rank)),
                      "transaction_id": info.get("transaction_id"),
                      "sampled_training_action": info.get("eta"),
                      "sampled_unclipped_action": float(np.asarray(
                          self.locals["actions"][rank]).reshape(-1)[0]),
                      "sample_log_probability": float(
                          self.locals["log_probs"][rank].detach().cpu()),
                      "normalized_observation": obs.tolist(),
                      "policy": policies[rank],
                      "routing": info["explanation"]}
            # Readable prose is generated from this complete audit at report time.
            self.trace.write(json.dumps(record, default=json_default) + "\n")
        # gzip buffers normal records; safe checkpoints/periodic backups close
        # and seal it. Do not synchronously flush once per vector step.
        self.progress("TRAINING / ROLLOUT", int(self.num_timesteps))
        stopping = self.stop_directory and stop_requested(self.stop_directory)
        if not stopping and self.recovery and time.monotonic() >= self.next_recovery:
            # Preserve a durable copy during long rollouts, not only after PPO
            # updates. It contains the last completed policy/optimizer checkpoint;
            # partial rollout audit records are rewound on resume.
            self._publish("TRAINING / PERIODIC DURABLE RECOVERY")
            self.next_recovery = time.monotonic() + self.backup_interval
        return not stopping

    def _on_rollout_start(self):
        # The previous PPO update has finished here. _on_rollout_end is too early.
        metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                   if key.startswith("train/") and isinstance(value, (int, float, np.number))}
        # Explained variance can be undefined for constant returns. Preserve
        # missingness as JSON null instead of emitting the nonstandard NaN token.
        metrics = {key: value if np.isfinite(value) else None for key, value in metrics.items()}
        if metrics:
            with (self.directory / "ppo_updates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"timestep": int(self.model.num_timesteps),
                                         "metrics": metrics}) + "\n")
        if self.recovery and self.model.num_timesteps:
            self._save_snapshot("resume")
            self._publish("PPO UPDATE / SAFE CHECKPOINT")
        if self.stop_directory:
            check_stop(self.stop_directory)
        if self.num_timesteps >= self.next_diagnostic:
            self.diagnose()

    def _save_snapshot(self, prefix):
        step = int(self.model.num_timesteps)
        path = self.directory / f"{prefix}_step_{step}.zip"
        temporary = self.directory / ".pending_snapshot.zip"
        self.model.save(str(temporary))
        temporary.replace(path)
        payload = {"features": list(FEATURE_NAMES), "mean": self.reference.tolist(),
                   "checkpoint": step, "observed_slots": self.feature_count,
                   "model_sha256": file_hash(path)}
        atomic_json(path.with_name(path.stem + "_reference.json"), payload)
        atomic_json(self.directory / "training_reference.json", payload)
        return path

    def _publish(self, reason):
        if not self.recovery:
            return
        had_trace = self.trace is not None
        if had_trace:
            self.trace.close()
            self.trace = None
        try:
            self.recovery(reason)
        finally:
            if had_trace:
                self.trace = gzip.open(self.directory / "training_explanations.jsonl.gz",
                                       "at", encoding="utf-8", compresslevel=3)

    def diagnose(self):
        step = int(self.model.num_timesteps)
        if step == self.last_diagnostic:
            return
        self.progress("FROZEN POLICY / TRAIN-TEST DIAGNOSTICS", step)
        path = self._save_snapshot("policy")
        tasks = [{**task, "checkpoint": step} for task in self.tasks]
        if self.completed_cases:
            completed = self.completed_cases()
            tasks = [task for task in tasks if (
                str(task["phase"]), int(task["repetition"]), int(task["checkpoint"]),
                str(task["split"]), round(float(task["failure_rate"]), 8),
                int(task["transaction_id"])) not in completed]
        if not tasks:
            self.last_diagnostic = step
            self.next_diagnostic = (step // self.interval + 1) * self.interval
            self.progress("TRAINING / DIAGNOSTICS ALREADY SAVED", step)
            return
        env = self.training_env
        # Training environments stay resident, so load diagnostic graphs only
        # in a small subset of idle workers, not in all training processes.
        width = min(env.num_envs, int(self.cfg.get("_main8_diagnostic_workers", 4)))
        indices = list(range(width))
        count = len(tasks)
        print(f"[DIAGNOSTIC WORKERS] active={width}/{env.num_envs}; "
              "other training workers remain idle without diagnostic graph copies.", flush=True)
        try:
            env.env_method("begin_diagnostics", str(path), self.cfg, width, indices=indices)
            for offset in range(0, count, width):
                if self.stop_directory:
                    check_stop(self.stop_directory)
                batches = env.env_method("diagnostic_batch", tasks[offset:offset + width],
                                         self.reference.tolist(), self.cfg, indices=indices)
                rows = [row for batch in batches for row in batch]
                self.store_rows(rows)
                finished = min(offset + width, count)
                self.progress(f"TRAIN-TEST DIAGNOSTICS {finished}/{count}", step)
                if finished == count or offset % (width * 8) == 0:
                    print(f"[DIAGNOSTIC PROGRESS] step={step}; saved={finished}/{count}; "
                          f"split={rows[-1]['split'] if rows else 'none'}; "
                          f"failure={rows[-1]['failure_rate'] if rows else 'none'}", flush=True)
                del batches, rows
                self._publish("TRAIN-TEST DIAGNOSTICS / COMMITTED PAYMENTS")
        finally:
            env.env_method("end_diagnostics", indices=indices)
        self.last_diagnostic = step
        self.next_diagnostic = (step // self.interval + 1) * self.interval
        self.progress("TRAINING / PPO POLICY UPDATE COMPLETED", step)

    def _on_training_end(self):
        try:
            if self.stop_directory and stop_requested(self.stop_directory):
                # A partial rollout has no matching PPO update. Resume from the
                # last completed-update checkpoint, rather than counting it as learned.
                raise MemoryStop("Training stopped; the last completed PPO checkpoint is preserved.")
            self.diagnose()
        finally:
            self.close()

    def close(self):
        if self.trace is not None:
            self.trace.close()
            self.trace = None
