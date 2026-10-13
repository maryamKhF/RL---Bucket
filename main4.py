"""Full paper-protocol runner for the RL + Bucket project.

Run from the project directory with:

    python main4.py

This entry point creates one 10,000-transaction pool, trains five PPO models
with distinct seeds on the same 7,000-row training partition, and evaluates
each model on the same 3,000 held-out transactions. Every configured failure
rate is evaluated for each seed. The run is intentionally full-sized and
does not use the fast-validation model or settings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import BaseCallback


PROJECT_ROOT = Path(__file__).resolve().parent
FINAL_REPORT_PATH = PROJECT_ROOT / "result.txt"
PAPER_TRANSACTION_COUNT = 10_000
PAPER_TRAINING_COUNT = 7_000
PAPER_EVALUATION_COUNT = 3_000
PAPER_REPETITIONS = 5
T_CRITICAL_95 = {
    2: 12.706204736432095,
    3: 4.302652729696142,
    4: 3.182446305284263,
    5: 2.7764451051977987,
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_ROOT / "20190501.gml.geo",
        help="Snapshot file (default: 20190501.gml.geo in the project root).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "Configs" / "config.yaml",
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "main4_paper_protocol",
        help="Root directory for this run's reports, models, and trial CSV.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=FINAL_REPORT_PATH,
        help="Final text report path (can be placed on persistent Google Drive).",
    )
    parser.add_argument(
        "--failure-rates",
        type=float,
        nargs="+",
        help="Optional override for configured edge failure rates.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume an interrupted run from this exact run directory.",
    )
    return parser.parse_args(argv)


def _json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _mean_ci95(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    mean = statistics.fmean(clean)
    if len(clean) < 2:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": len(clean)}

    degrees_of_freedom = len(clean) - 1
    t_critical = T_CRITICAL_95.get(degrees_of_freedom, 1.96)
    half_width = t_critical * statistics.stdev(clean) / math.sqrt(len(clean))
    return {
        "mean": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "n": len(clean),
    }


def _validate_protocol(cfg, failure_rates):
    evaluation_cfg = cfg.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        raise ValueError("Configuration must define an evaluation mapping.")

    configured_total = int(cfg.get("n_transactions", -1))
    configured_test = int(evaluation_cfg.get("transaction_count", -1))
    configured_repetitions = int(evaluation_cfg.get("repetitions", -1))
    train_ratio = float(cfg.get("train_ratio", -1))
    test_ratio = float(cfg.get("test_ratio", -1))

    expected = {
        "n_transactions": (configured_total, PAPER_TRANSACTION_COUNT),
        "evaluation.transaction_count": (
            configured_test,
            PAPER_EVALUATION_COUNT,
        ),
        "evaluation.repetitions": (configured_repetitions, PAPER_REPETITIONS),
    }
    invalid = [
        f"{name}={actual} (required {required})"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if abs(train_ratio - 0.70) > 1e-9 or abs(test_ratio - 0.30) > 1e-9:
        invalid.append(
            f"train_ratio/test_ratio={train_ratio}/{test_ratio} (required 0.70/0.30)"
        )
    if invalid:
        raise ValueError(
            "main4 enforces the article's 10,000 / 7,000 / 3,000 / 5-run "
            "protocol. Correct Configs/config.yaml before starting: "
            + "; ".join(invalid)
        )

    if not failure_rates:
        raise ValueError("At least one failure rate is required.")
    if any(not 0.0 <= rate <= 1.0 for rate in failure_rates):
        raise ValueError("Failure rates must be between 0 and 1.")


def _transaction_value(transaction, field, default=""):
    value = getattr(transaction, field, default)
    return default if value is None else value


def _write_transaction_manifest(path, pool, training_count):
    fields = ["transaction_id", "partition", "source", "destination", "amount"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, transaction in enumerate(pool):
            writer.writerow(_transaction_manifest_row(transaction, index, training_count))


def _transaction_manifest_row(transaction, index, training_count):
    return {
        "transaction_id": _transaction_value(transaction, "tx_id", index + 1),
        "partition": "train" if index < training_count else "test",
        "source": _transaction_value(transaction, "source"),
        "destination": _transaction_value(transaction, "destination"),
        "amount": _transaction_value(transaction, "amount"),
    }


def _transaction_manifest_content(pool, training_count):
    fields = ["transaction_id", "partition", "source", "destination", "amount"]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for index, transaction in enumerate(pool):
        writer.writerow(_transaction_manifest_row(transaction, index, training_count))
    return handle.getvalue()


def _ensure_transaction_manifest(path, pool, training_count):
    path = Path(path)
    expected = _transaction_manifest_content(pool, training_count).encode("utf-8")
    if path.is_file():
        saved = path.read_bytes()
        if saved == expected:
            return False
        if not expected.startswith(saved):
            raise ValueError(
                "The saved transaction split differs from the regenerated pool; "
                "refusing to mix results from different transactions."
            )
        print(
            "The saved transaction manifest is incomplete; rebuilding it from "
            "the matching deterministic transaction pool.",
            flush=True,
        )
    _write_transaction_manifest(path, pool, training_count)
    return True


def _load_transaction_manifest(path, graph, training_count, total_count):
    """Rebuild the fixed transaction pool from its saved manifest."""
    from Simulation.transaction_generator import Transaction

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot resume without the saved transaction manifest: {path}"
        )

    node_by_text = {}
    for node in graph.nodes:
        node_by_text.setdefault(str(node), node)

    transactions = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "transaction_id", "partition", "source", "destination", "amount"
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Transaction manifest has an invalid header: {path}"
            )

        for expected_id, row in enumerate(reader):
            try:
                transaction_id = int(row["transaction_id"])
                source_text = row["source"]
                destination_text = row["destination"]
                source = node_by_text[source_text]
                destination = node_by_text[destination_text]
                amount = float(row["amount"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Cannot reconstruct transaction manifest row "
                    f"{expected_id + 2}: {row}"
                ) from exc

            expected_partition = "train" if expected_id < training_count else "test"
            if transaction_id != expected_id or row["partition"] != expected_partition:
                raise ValueError(
                    "Transaction manifest ordering or 70/30 partition is invalid "
                    f"at row {expected_id + 2}."
                )

            transactions.append(
                Transaction(
                    tx_id=transaction_id,
                    source=source,
                    destination=destination,
                    amount=amount,
                    timestamp=transaction_id,
                    split=expected_partition,
                    metadata={"restored_from_manifest": True},
                )
            )

    if len(transactions) != total_count:
        raise ValueError(
            f"Saved transaction manifest has {len(transactions)} rows; "
            f"expected {total_count}."
        )
    return transactions


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_checkpoint(checkpoint_dir, model_name):
    candidates = list(Path(checkpoint_dir).glob(f"{model_name}_checkpoint_*_steps.zip"))
    def timestep(path):
        try:
            return int(path.stem.rsplit("_", 2)[1])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=timestep) if candidates else None


def _load_trial_rows(path):
    rows = {}
    if not Path(path).is_file():
        return rows
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if (
                not raw
                or not raw.get("evaluation_seed")
                or None in raw
                or any(raw.get(field) is None for field in TRIAL_FIELDS)
            ):
                continue  # discard a possibly truncated final CSV record
            try:
                result = dict(raw)
                for key in ("repetition", "training_seed", "evaluation_seed",
                            "top_k", "candidate_count", "usable_candidate_count", "bucket_size",
                            "attempt_count", "hops", "backtrack_count", "partial_backtrack_count",
                            "partial_backtrack_success", "full_reroute_count", "path_length"):
                    result[key] = int(result[key]) if result.get(key) not in (None, "") else 0
                for key in ("failure_rate", "amount", "eta", "fee", "delay", "carbon", "reward",
                            "prediction_time", "pipeline_time"):
                    result[key] = float(result[key]) if result.get(key) not in (None, "") else None
                result["success"] = str(result.get("success", "")).lower() in {"true", "1", "yes"}
                result["path"] = json.loads(result.get("path") or "[]")
                result["path_length"] = result.get("hops", 0)
                for key in ("successful_bucket_rank",):
                    value = result.get(key)
                    result[key] = int(value) if value not in (None, "", "None") else None
                for key in ("successful_bucket_rank", "successful_route_source"):
                    if result.get(key) in ("", "None"):
                        result[key] = None
                result["bucket_size"] = result.get("bucket_size", 0)
                result["attempt_count"] = result.get("attempt_count", 0)
                unique_key = (
                    result["repetition"],
                    round(result["failure_rate"], 10),
                    result["evaluation_seed"],
                )
                rows[unique_key] = result
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
    return rows


def _learning_curve_summary(path, repetition, training_seed, failure_rate):
    csv_path = Path(path) / (
        f"rep_{repetition:02d}_seed_{training_seed}_learning_curve.csv"
    )
    png_path = csv_path.with_suffix(".png")
    points = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                try:
                    point = dict(raw)
                    for key in ("timestep", "repetition", "training_seed", "train_probe_size", "test_probe_size"):
                        point[key] = int(point[key])
                    for key in ("progress_percent", "failure_rate", "train_success_rate", "test_success_rate", "train_test_gap_pp"):
                        point[key] = float(point[key])
                    points.append(point)
                except (ValueError, KeyError):
                    continue
    if not points:
        return {"chart": str(png_path), "csv": str(csv_path), "points": []}
    best = max(points, key=lambda point: point["test_success_rate"])
    return {
        "repetition": repetition,
        "training_seed": training_seed,
        "failure_rate": failure_rate,
        "chart": str(png_path),
        "csv": str(csv_path),
        "train_probe_size": points[-1]["train_probe_size"],
        "test_probe_size": points[-1]["test_probe_size"],
        "best_test_success_rate": best["test_success_rate"],
        "best_test_timestep": best["timestep"],
        "last_test_success_rate": points[-1]["test_success_rate"],
        "final_test_drop_from_peak_pp": best["test_success_rate"] - points[-1]["test_success_rate"],
        "test_probe_reused_for_diagnostic_monitoring": True,
        "points": points,
    }


def _run_trial(core, graph, cfg, model, transaction, transaction_index,
               training_seed, repetition, failure_rate, failure_rate_index):
    evaluation_seed = (
        int(cfg["seed"])
        + repetition * 10_000_000
        + failure_rate_index * 100_000
        + transaction_index
    )
    evaluation_graph = copy.deepcopy(graph)
    core.assign_failure_probabilities(
        evaluation_graph,
        failure_rate,
        evaluation_seed,
    )
    dynamics = core.NetworkDynamics(evaluation_graph)
    failure_model = core.FailureModel(seed=evaluation_seed)
    runtime_env = core.RoutingEnv(
        G=evaluation_graph,
        transactions=[copy.deepcopy(transaction)],
        heuristic_fn=core.lnd_cost,
        config=copy.deepcopy(cfg),
        mode="eval",
        failure_model=failure_model,
        network_dynamics=dynamics,
    )

    try:
        model.set_env(runtime_env)
        observation, _reset_info = runtime_env.reset(seed=evaluation_seed)
        prediction_start = time.perf_counter()
        action, _state = model.predict(observation, deterministic=True)
        prediction_seconds = time.perf_counter() - prediction_start

        pipeline_start = time.perf_counter()
        _next_observation, reward, terminated, truncated, info = runtime_env.step(action)
        pipeline_seconds = time.perf_counter() - pipeline_start

        result = core.extract_episode_result(
            transaction,
            reward,
            terminated,
            truncated,
            info,
            transaction_index,
        )
        result.update(
            {
                "repetition": repetition,
                "training_seed": training_seed,
                "failure_rate": failure_rate,
                "evaluation_seed": evaluation_seed,
                "prediction_time": prediction_seconds,
                "pipeline_time": pipeline_seconds,
            }
        )
        return result
    finally:
        runtime_env.close()


class LearningCurveCallback(BaseCallback):
    """Periodically compare train/test payment success on fixed probes."""

    def __init__(
        self,
        *,
        core,
        graph,
        config,
        train_probe,
        test_probe,
        base_seed,
        repetition,
        repetition_seed,
        failure_rate,
        eval_interval=4000,
        output_dir,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.core = core
        self.graph = graph
        self.config = copy.deepcopy(config)
        self.train_probe = list(train_probe)
        self.test_probe = list(test_probe)
        self.base_seed = int(base_seed)
        self.repetition = int(repetition)
        self.repetition_seed = int(repetition_seed)
        self.failure_rate = float(failure_rate)
        self.eval_interval = max(1, int(eval_interval))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.points = []
        self.csv_path = self.output_dir / (
            f"rep_{self.repetition:02d}_seed_{self.repetition_seed}_learning_curve.csv"
        )
        self.png_path = self.output_dir / (
            f"rep_{self.repetition:02d}_seed_{self.repetition_seed}_learning_curve.png"
        )
        existing = _learning_curve_summary(
            self.output_dir, self.repetition, self.repetition_seed, self.failure_rate
        )
        self.points = existing.get("points", [])
        latest_step = max((point["timestep"] for point in self.points), default=0)
        self.next_eval_timestep = (latest_step // self.eval_interval + 1) * self.eval_interval

    def _evaluate_probe(self, transactions, split_name):
        successful = 0
        last_log_time = time.perf_counter()
        for index, transaction in enumerate(transactions, start=1):
            evaluation_seed = (
                self.base_seed * 1_000_000
                + (0 if split_name == "train" else 100_000)
                + index
            )
            eval_graph = copy.deepcopy(self.graph)
            self.core.assign_failure_probabilities(
                eval_graph, self.failure_rate, evaluation_seed
            )
            env = self.core.RoutingEnv(
                G=eval_graph,
                transactions=[copy.deepcopy(transaction)],
                heuristic_fn=self.core.lnd_cost,
                config=copy.deepcopy(self.config),
                mode="eval",
                failure_model=self.core.FailureModel(seed=evaluation_seed),
                network_dynamics=self.core.NetworkDynamics(eval_graph),
            )
            try:
                observation, _info = env.reset(seed=evaluation_seed)
                action, _state = self.model.predict(observation, deterministic=True)
                _next_observation, _reward, _terminated, _truncated, info = env.step(action)
                successful += int(bool(info.get("success", False)))
            finally:
                env.close()

            now = time.perf_counter()
            if index == len(transactions) or now - last_log_time >= 60.0:
                print(
                    f"[LEARNING-CURVE PROBE — ACTIVE] split={split_name}; "
                    f"sample={index}/{len(transactions)}; "
                    f"repetition={self.repetition}/5; "
                    "policy is routing fixed diagnostic payments",
                    flush=True,
                )
                last_log_time = now

        return 100.0 * successful / max(1, len(transactions))

    def _record_point(self, timestep):
        self.core.set_stage(
            f"MAIN4 - LEARNING-CURVE CHECKPOINT {timestep} TIMESTEPS; "
            f"REPETITION {self.repetition}/5"
        )
        print(
            f"\n[LEARNING-CURVE EVALUATION] repetition {self.repetition}/5, "
            f"seed {self.repetition_seed}, timestep {timestep:,}; "
            f"testing {len(self.train_probe)} train + {len(self.test_probe)} held-out probe transactions "
            f"at {self.failure_rate:.0%} failure rate.",
            flush=True,
        )
        train_success = self._evaluate_probe(self.train_probe, "train")
        test_success = self._evaluate_probe(self.test_probe, "test")
        point = {
            "timestep": int(timestep),
            "progress_percent": 100.0 * int(timestep) / max(1, self.model._total_timesteps),
            "repetition": self.repetition,
            "training_seed": self.repetition_seed,
            "failure_rate": self.failure_rate,
            "train_probe_size": len(self.train_probe),
            "test_probe_size": len(self.test_probe),
            "train_success_rate": train_success,
            "test_success_rate": test_success,
            "train_test_gap_pp": train_success - test_success,
        }
        self.points.append(point)
        self._write_artifacts()
        print(
            f"[LEARNING-CURVE RESULT] timestep={timestep:,}; "
            f"train success={train_success:.1f}%; "
            f"test success={test_success:.1f}%; "
            f"gap={train_success - test_success:+.1f} percentage points; "
            f"chart={self.png_path}",
            flush=True,
        )

    def _write_artifacts(self):
        fields = [
            "timestep", "progress_percent", "repetition", "training_seed",
            "failure_rate", "train_probe_size", "test_probe_size",
            "train_success_rate", "test_success_rate", "train_test_gap_pp",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.points)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [point["timestep"] for point in self.points]
        train_values = [point["train_success_rate"] for point in self.points]
        test_values = [point["test_success_rate"] for point in self.points]
        gaps = [point["train_test_gap_pp"] for point in self.points]
        figure, (axis, gap_axis) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        axis.plot(steps, train_values, marker="o", label="Train probe success")
        axis.plot(steps, test_values, marker="o", label="Held-out test probe success")
        axis.set_ylabel("Payment success rate (%)")
        axis.set_ylim(0, 100)
        axis.set_title(
            f"PPO learning curve — repetition {self.repetition}/5, "
            f"seed {self.repetition_seed}"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        gap_axis.plot(steps, gaps, marker="o", color="tab:red")
        gap_axis.axhline(0, color="black", linewidth=0.8)
        gap_axis.set_ylabel("Train − test\n(pp)")
        gap_axis.set_xlabel("PPO training timesteps")
        gap_axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.png_path, dpi=150)
        plt.close(figure)

    def _on_training_start(self):
        current = int(self.model.num_timesteps)
        if not self.points or self.points[-1]["timestep"] != current:
            self._record_point(current)
        self.next_eval_timestep = (current // self.eval_interval + 1) * self.eval_interval

    def _on_step(self):
        if self.num_timesteps >= self.next_eval_timestep:
            self._record_point(min(self.num_timesteps, self.model._total_timesteps))
            while self.num_timesteps >= self.next_eval_timestep:
                self.next_eval_timestep += self.eval_interval
        if self.num_timesteps >= self.model._total_timesteps:
            if not self.points or self.points[-1]["timestep"] != self.model._total_timesteps:
                self._record_point(self.model._total_timesteps)
        return True

    def summary(self):
        if not self.points:
            return {"chart": str(self.png_path), "csv": str(self.csv_path), "points": []}
        best_test = max(self.points, key=lambda point: point["test_success_rate"])
        return {
            "repetition": self.repetition,
            "training_seed": self.repetition_seed,
            "failure_rate": self.failure_rate,
            "chart": str(self.png_path),
            "csv": str(self.csv_path),
            "train_probe_size": len(self.train_probe),
            "test_probe_size": len(self.test_probe),
            "best_test_success_rate": best_test["test_success_rate"],
            "best_test_timestep": best_test["timestep"],
            "last_test_success_rate": self.points[-1]["test_success_rate"],
            "final_test_drop_from_peak_pp": (
                best_test["test_success_rate"]
                - self.points[-1]["test_success_rate"]
            ),
            "test_probe_reused_for_diagnostic_monitoring": True,
            "points": self.points,
        }


TRIAL_FIELDS = [
    "repetition",
    "training_seed",
    "failure_rate",
    "evaluation_seed",
    "transaction_id",
    "source",
    "destination",
    "amount",
    "success",
    "eta",
    "top_k",
    "candidate_count",
    "usable_candidate_count",
    "bucket_size",
    "attempt_count",
    "fee",
    "delay",
    "hops",
    "carbon",
    "reward",
    "successful_bucket_rank",
    "successful_route_source",
    "reason",
    "backtrack_count",
    "partial_backtrack_count",
    "partial_backtrack_success",
    "full_reroute_count",
    "path_length",
    "prediction_time",
    "pipeline_time",
    "path",
]


def _csv_row(result):
    row = {field: result.get(field) for field in TRIAL_FIELDS}
    row["path"] = json.dumps(result.get("path", []), ensure_ascii=False, default=_json_default)
    return row


def _aggregate_repetitions(batch_summaries, failure_rates):
    metrics = [
        "success_rate",
        "average_attempts",
        "average_hops",
        "average_fee",
        "average_delay",
        "p90_delay_success",
        "p95_delay_success",
        "p90_delay_routed",
        "p95_delay_routed",
        "average_carbon",
        "average_reward",
    ]
    summaries = {}
    for rate in failure_rates:
        rate_batches = [
            batch for batch in batch_summaries
            if math.isclose(batch["failure_rate"], rate, abs_tol=1e-12)
        ]
        summaries[f"{rate:.8g}"] = {
            metric: _mean_ci95(
                [batch["aggregate"].get(metric) for batch in rate_batches]
            )
            for metric in metrics
        }
        summaries[f"{rate:.8g}"]["successful_transactions"] = sum(
            batch["aggregate"]["successful_transactions"] for batch in rate_batches
        )
        summaries[f"{rate:.8g}"]["total_transactions"] = sum(
            batch["aggregate"]["total_transactions"] for batch in rate_batches
        )
    return summaries


def main4(argv=None):
    args = _parse_args(argv)
    snapshot_path = args.snapshot.resolve()
    config_path = args.config.resolve()
    output_root = args.output_dir.resolve()
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    else:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

    global FINAL_REPORT_PATH
    FINAL_REPORT_PATH = args.report_file.resolve()
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Make this entry point independent of laptop smoke-test environment flags.
    os.environ["RL_FAST_VALIDATION"] = "0"
    os.environ["RL_FAST_TRAINING"] = "0"
    os.environ["RL_REPORT_FILE"] = str(FINAL_REPORT_PATH)

    import main as core

    core.REPORT_FILE = FINAL_REPORT_PATH
    report = core.ReportWriter(core.REPORT_FILE)
    core.RUN_START_TIME = time.time()
    core.set_stage("MAIN4 - PAPER-PROTOCOL INITIALIZATION")

    csv_path = run_dir / "evaluation_trials.csv"
    transaction_manifest_path = run_dir / "transaction_split.csv"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "run_metadata.json"
    batch_summaries = []
    total_training_seconds = 0.0
    expected_trial_count = 0
    completed_trial_count = 0

    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must contain a YAML mapping.")
        # main4 is the paper-baseline runner. Formula variants are selected
        # only by main5 so edits to the ablation config cannot alter main4.
        cfg["reward"] = dict(cfg.get("reward", {}))
        cfg["reward"]["formula"] = "paper_base"

        failure_rates = (
            [float(rate) for rate in args.failure_rates]
            if args.failure_rates is not None
            else [float(rate) for rate in cfg.get("failure_rates", [])]
        )
        _validate_protocol(cfg, failure_rates)

        base_seed = int(cfg["seed"])
        repetitions = PAPER_REPETITIONS
        total_training_count = PAPER_TRANSACTION_COUNT
        evaluation_count = PAPER_EVALUATION_COUNT
        expected_trial_count = repetitions * evaluation_count * len(failure_rates)

        run_metadata = {
            "snapshot_sha256": _sha256_file(snapshot_path),
            "config_sha256": _sha256_file(config_path),
            "base_seed": base_seed,
            "failure_rates": failure_rates,
            "total_transactions": total_training_count,
            "training_transactions": PAPER_TRAINING_COUNT,
            "held_out_transactions": evaluation_count,
            "repetitions": repetitions,
            "reward_formula": "paper_base",
        }
        if args.resume_run:
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Cannot safely resume: run metadata is missing at {metadata_path}."
                )
            stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if stored_metadata != run_metadata:
                raise ValueError(
                    "Resume settings do not match the original run (snapshot, config, "
                    "seed, failure rates, or paper protocol changed). Start a new run instead."
                )
        else:
            _write_json(metadata_path, run_metadata)

        _write_json(
            run_dir / "run_status.json",
            {"status": "RUNNING", "stage": "INITIALIZING", "completed_trials": 0,
             "expected_trials": expected_trial_count, "run_directory": str(run_dir)},
        )

        report.section("MAIN4 - ARTICLE-SIZED RL EXPERIMENT")
        report.item("Run status", "RUNNING")
        report.item("Execution mode", "FULL PAPER PROTOCOL; FAST MODE DISABLED")
        report.item("Reward formula", "paper_base (locked to article baseline)")
        report.item("Snapshot", snapshot_path)
        report.item("Config", config_path)
        report.item("Base seed", base_seed)
        report.item("Generated transaction pool", total_training_count)
        report.item("Training transactions per repetition", PAPER_TRAINING_COUNT)
        report.item("Unique held-out transactions", evaluation_count)
        report.item("Training/evaluation split", "70% / 30%")
        report.item("Independent training/evaluation repetitions", repetitions)
        report.item("Failure rates", ", ".join(f"{rate:.0%}" for rate in failure_rates))
        report.item("Expected evaluation trials", expected_trial_count)
        report.item("Fast validation environment flags", "IGNORED; full training forced")
        report.item("Run output directory", run_dir)
        report.item("Final text report", FINAL_REPORT_PATH)
        report.save()

        print("MAIN4: FULL ARTICLE-SIZED RUN")
        print(f"Transactions: {PAPER_TRANSACTION_COUNT} = {PAPER_TRAINING_COUNT} train + {PAPER_EVALUATION_COUNT} held-out")
        print(f"Replications: {repetitions}; failure rates: {', '.join(f'{rate:.0%}' for rate in failure_rates)}")
        print(f"Expected evaluation trials: {expected_trial_count:,}")
        print(f"Output directory: {run_dir}")

        core.set_stage("MAIN4 - LOAD SNAPSHOT AND BUILD GRAPH")
        cfg["rl"] = dict(cfg.get("rl", {}))
        cfg["rl"]["device"] = cfg["rl"].get("device", "auto")
        graph_data = core.geo_to_json(snapshot_path)
        graph = core.LNGraphBuilder(
            default_capacity=cfg.get("channel", {}).get("default_capacity")
        ).from_data(graph_data)
        report.item("Graph nodes", graph.number_of_nodes())
        report.item("Graph directed channels", graph.number_of_edges())
        report.save()

        core.set_stage("MAIN4 - LOAD OR GENERATE FIXED TRANSACTION POOL")
        if args.resume_run:
            transaction_pool = _load_transaction_manifest(
                transaction_manifest_path,
                graph,
                PAPER_TRAINING_COUNT,
                total_training_count,
            )
            print(
                "[TransactionGenerator] Restored the original fixed transaction "
                "pool from the saved manifest.",
                flush=True,
            )
        else:
            core.set_seed(base_seed)
            transaction_pool = core.generate_transactions(
                graph,
                total_training_count,
                base_seed,
                cfg["simulation"]["min_amount"],
                cfg["simulation"]["max_amount"],
            )
            if len(transaction_pool) != total_training_count:
                raise RuntimeError(
                    f"Expected {total_training_count} generated transactions, got {len(transaction_pool)}."
                )
        train_transactions, held_out_transactions = core.split_transaction_pool(
            transaction_pool,
            evaluation_count,
        )
        if len(train_transactions) != PAPER_TRAINING_COUNT:
            raise RuntimeError(
                f"Expected {PAPER_TRAINING_COUNT} training rows, got {len(train_transactions)}."
            )
        if len(held_out_transactions) != evaluation_count:
            raise RuntimeError(
                f"Expected {evaluation_count} held-out rows, got {len(held_out_transactions)}."
            )
        if not args.resume_run:
            _ensure_transaction_manifest(
                transaction_manifest_path,
                transaction_pool,
                len(train_transactions),
            )
        report.item("Training transactions generated once", len(train_transactions))
        report.item("Held-out transactions generated once", len(held_out_transactions))
        report.item("Transaction manifest", transaction_manifest_path)
        report.add("The same held-out transaction IDs, sources, destinations, and amounts are reused for all five repetitions and every configured failure rate.")
        report.save()

        model_root = run_dir / "models"
        log_root = run_dir / "logs"
        model_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        training_failure_rate = float(cfg.get("training_failure_rate", 0.03))
        curve_probe_size = max(
            1, int(os.environ.get("RL_LEARNING_CURVE_PROBE_SIZE", "50"))
        )
        curve_interval = max(
            1, int(os.environ.get("RL_LEARNING_CURVE_INTERVAL", "4000"))
        )
        curve_rng = random.Random(base_seed)
        train_curve_probe = curve_rng.sample(
            train_transactions, min(curve_probe_size, len(train_transactions))
        )
        test_curve_probe = curve_rng.sample(
            held_out_transactions, min(curve_probe_size, len(held_out_transactions))
        )
        learning_curve_summaries = []
        learning_curve_dir = run_dir / "learning_curves"
        report.item("Learning curve probes", f"{len(train_curve_probe)} train + {len(test_curve_probe)} held-out test transactions")
        report.item("Learning curve sampling", "Fixed probe samples; measured at initialization and every 4,000 PPO timesteps")
        report.item("Learning curve outputs", learning_curve_dir)
        report.item("Learning curve purpose", "Diagnostic only; repeated test-probe measurements do not control training or select a model")
        report.save()

        existing_trials = _load_trial_rows(csv_path)
        completed_trial_count = len(existing_trials)
        if args.resume_run and csv_path.is_file():
            # Rewrite only complete, unique rows. This removes a partial final line
            # if Colab stopped while it was flushing the CSV to Drive.
            with csv_path.open("w", newline="", encoding="utf-8") as clean_file:
                clean_writer = csv.DictWriter(clean_file, fieldnames=TRIAL_FIELDS)
                clean_writer.writeheader()
                clean_writer.writerows(_csv_row(row) for row in existing_trials.values())

        with csv_path.open("a", newline="", encoding="utf-8") as trial_file:
            trial_writer = csv.DictWriter(trial_file, fieldnames=TRIAL_FIELDS)
            if trial_file.tell() == 0:
                trial_writer.writeheader()

            for repetition in range(1, repetitions + 1):
                training_seed = base_seed + repetition - 1
                core.set_stage(
                    f"MAIN4 - TRAIN MODEL {repetition}/{repetitions} (SEED {training_seed})"
                )
                print(
                    f"\nTraining repetition {repetition}/{repetitions} "
                    f"with seed {training_seed} on the same 7,000 transactions...",
                    flush=True,
                )

                repetition_cfg = copy.deepcopy(cfg)
                repetition_cfg["seed"] = training_seed
                repetition_cfg["rl"]["seed"] = training_seed
                repetition_cfg["_progress_context"] = {
                    "Reward formula": "paper_base (article baseline)",
                    "Repetition": f"{repetition}/{repetitions}",
                    "Training seed": training_seed,
                    "Training set": f"{len(train_transactions):,} transactions",
                    "Training failures": f"{training_failure_rate:.0%}",
                }
                model_dir = model_root / f"rep_{repetition:02d}_seed_{training_seed}"
                repetition_cfg["rl"]["model_dir"] = str(model_dir)
                repetition_cfg["rl"]["log_dir"] = str(
                    log_root / f"rep_{repetition:02d}_seed_{training_seed}"
                )
                repetition_cfg["rl"]["checkpoint_dir"] = str(
                    model_dir / "checkpoints"
                )
                model_name = f"end_to_end_hidden_failure_blind_seed_{training_seed}"
                final_model_path = model_dir / f"{model_name}.zip"
                checkpoint_path = _latest_checkpoint(
                    repetition_cfg["rl"]["checkpoint_dir"], model_name
                )
                curve_callback = LearningCurveCallback(
                    core=core,
                    graph=graph,
                    config=cfg,
                    train_probe=train_curve_probe,
                    test_probe=test_curve_probe,
                    base_seed=base_seed,
                    repetition=repetition,
                    repetition_seed=training_seed,
                    failure_rate=training_failure_rate,
                    eval_interval=curve_interval,
                    output_dir=learning_curve_dir,
                )

                if final_model_path.is_file():
                    print(f"Using completed model from {final_model_path}", flush=True)
                    model = core.PPO.load(
                        str(final_model_path), device=repetition_cfg["rl"].get("device", "auto")
                    )
                    curve_summary = _learning_curve_summary(
                        learning_curve_dir, repetition, training_seed, training_failure_rate
                    )
                    training_seconds = 0.0
                else:
                    training_graph = copy.deepcopy(graph)
                    core.assign_failure_probabilities(
                        training_graph,
                        training_failure_rate,
                        training_seed,
                    )
                    if checkpoint_path is not None:
                        repetition_cfg["rl"]["resume_checkpoint"] = str(checkpoint_path)
                    training_start = time.perf_counter()
                    model = core.train_agent(
                        training_graph,
                        copy.deepcopy(train_transactions),
                        core.lnd_cost,
                        repetition_cfg,
                        model_name,
                        training_seed,
                        fast_training=False,
                        extra_callbacks=[curve_callback],
                    )
                    training_seconds = time.perf_counter() - training_start
                    total_training_seconds += training_seconds
                    curve_summary = curve_callback.summary()
                    del training_graph

                report.section(f"TRAINING REPETITION {repetition}/{repetitions}")
                report.item("Training seed", training_seed)
                report.item("Training rows", len(train_transactions))
                report.item("Training failure rate", training_failure_rate)
                report.item("PPO training mode", "FULL; fast_training=False")
                report.item("PPO training time (seconds)", round(training_seconds, 4))
                report.item("Model", model_dir / f"{model_name}.zip")
                learning_curve_summaries.append(curve_summary)
                report.item("Train/test learning curve", curve_summary["chart"])
                report.item("Learning curve data", curve_summary["csv"])
                report.item("Best diagnostic test success", curve_summary.get("best_test_success_rate"))
                report.item("Timestep at best diagnostic test success", curve_summary.get("best_test_timestep"))
                report.item("Final test-probe drop from peak (percentage points)", curve_summary.get("final_test_drop_from_peak_pp"))
                report.save()
                for failure_rate_index, failure_rate in enumerate(failure_rates):
                    core.set_stage(
                        f"MAIN4 - EVALUATE REP {repetition}/{repetitions}, "
                        f"FAILURE RATE {failure_rate:.0%}"
                    )
                    print(
                        f"Evaluating repetition {repetition}/{repetitions}, "
                        f"failure rate {failure_rate:.0%}: "
                        f"{evaluation_count:,} fixed held-out transactions",
                        flush=True,
                    )
                    batch_results = []
                    batch_start = time.perf_counter()
                    last_progress_time = batch_start

                    for transaction_index, transaction in enumerate(
                        held_out_transactions,
                        start=1,
                    ):
                        evaluation_seed = (
                            base_seed + repetition * 10_000_000
                            + failure_rate_index * 100_000 + transaction_index
                        )
                        trial_key = (repetition, round(failure_rate, 10), evaluation_seed)
                        result = existing_trials.get(trial_key)
                        if result is None:
                            result = _run_trial(
                                core,
                                graph,
                                cfg,
                                model,
                                transaction,
                                transaction_index,
                                training_seed,
                                repetition,
                                failure_rate,
                                failure_rate_index,
                            )
                            trial_writer.writerow(_csv_row(result))
                            existing_trials[trial_key] = result
                            completed_trial_count += 1
                        batch_results.append(result)

                        now = time.perf_counter()
                        if (
                            transaction_index % 25 == 0
                            or now - last_progress_time >= 60.0
                            or transaction_index >= evaluation_count
                        ):
                            trial_file.flush()
                            batch_elapsed = max(now - batch_start, 1e-9)
                            rate = transaction_index / batch_elapsed
                            remaining_seconds = (
                                (evaluation_count - transaction_index) / rate
                                if rate > 0
                                else 0.0
                            )
                            batch_percent = transaction_index / evaluation_count * 100.0
                            print(
                                "\n[EVALUATION PROGRESS — ACTIVE]"
                                "\nStage       : HELD-OUT TEST / ROUTING EVALUATION"
                                f"\nRepetition  : {repetition}/{repetitions} (seed {training_seed})"
                                f"\nFailure rate: {failure_rate:.0%}"
                                f"\nTransaction : {transaction_index:,}/{evaluation_count:,}"
                                f"\nBatch       : {batch_percent:.2f}%"
                                f"\nOverall     : {completed_trial_count:,}/{expected_trial_count:,} evaluation trials"
                                f"\nElapsed     : {core._format_elapsed(batch_elapsed)}"
                                f"\nRemaining   : {core._format_elapsed(remaining_seconds)} (estimated for this batch)"
                                "\nAgent state : trained policy is routing the current held-out payment"
                                "\nProcess     : ACTIVE; an evaluation result was recorded",
                                flush=True,
                            )
                            last_progress_time = now

                    trial_file.flush()
                    batch_aggregate = core.aggregate_evaluation_results(batch_results)
                    batch_summary = {
                        "repetition": repetition,
                        "training_seed": training_seed,
                        "failure_rate": failure_rate,
                        "transaction_count": len(batch_results),
                        "elapsed_seconds": time.perf_counter() - batch_start,
                        "aggregate": batch_aggregate,
                    }
                    batch_summaries.append(batch_summary)
                    report.section(
                        f"EVALUATION SUMMARY - REP {repetition}, RATE {failure_rate:.0%}"
                    )
                    report.item("Training seed", training_seed)
                    report.item("Failure rate", failure_rate)
                    report.item("Unique held-out transactions", evaluation_count)
                    report.item("Evaluation trials in this batch", len(batch_results))
                    report.item("Success rate", f"{batch_aggregate['success_rate']:.2f}%")
                    report.item("Average fee", batch_aggregate["average_fee"])
                    report.item("Average delay", batch_aggregate["average_delay"])
                    report.item("P90 delay successful", batch_aggregate["p90_delay_success"])
                    report.item("P95 delay successful", batch_aggregate["p95_delay_success"])
                    report.item("P90 delay all routed", batch_aggregate["p90_delay_routed"])
                    report.item("P95 delay all routed", batch_aggregate["p95_delay_routed"])
                    report.item("Average carbon", batch_aggregate["average_carbon"])
                    report.item("Batch elapsed seconds", batch_summary["elapsed_seconds"])

                    current_summary = {
                        "protocol": {
                            "total_transactions": total_training_count,
                            "training_transactions": len(train_transactions),
                            "unique_held_out_transactions": evaluation_count,
                            "repetitions": repetitions,
                            "failure_rates": failure_rates,
                            "expected_trials": expected_trial_count,
                            "completed_trials": completed_trial_count,
                            "same_transaction_pool_across_repetitions": True,
                            "model_retrained_each_repetition": True,
                        },
                        "completed_batches": batch_summaries,
                        "learning_curves": learning_curve_summaries,
                        "failure_rate_summary_across_seeds": _aggregate_repetitions(
                            batch_summaries,
                            failure_rates,
                        ),
                    }
                    _write_json(summary_path, current_summary)
                    _write_json(
                        run_dir / "run_status.json",
                        {"status": "RUNNING", "stage": "EVALUATION",
                         "repetition": repetition, "failure_rate": failure_rate,
                         "completed_trials": completed_trial_count,
                         "expected_trials": expected_trial_count,
                         "run_directory": str(run_dir)},
                    )
                    report.item("Completed trial count", completed_trial_count)
                    report.item("Trial CSV", csv_path)
                    report.item("Summary JSON", summary_path)
                    report.save()
                    del batch_results

                del model

        if completed_trial_count != expected_trial_count:
            raise RuntimeError(
                f"Completed {completed_trial_count} evaluation trials; "
                f"expected {expected_trial_count}."
            )

        final_summary = {
            "protocol": {
                "total_transactions": total_training_count,
                "training_transactions": len(train_transactions),
                "unique_held_out_transactions": evaluation_count,
                "repetitions": repetitions,
                "failure_rates": failure_rates,
                "expected_trials": expected_trial_count,
                "completed_trials": completed_trial_count,
                "training_seeds": [base_seed + rep - 1 for rep in range(1, repetitions + 1)],
                "same_transaction_pool_across_repetitions": True,
                "model_retrained_each_repetition": True,
            },
            "training_time_seconds_total": total_training_seconds,
            "completed_batches": batch_summaries,
            "learning_curves": learning_curve_summaries,
            "failure_rate_summary_across_seeds": _aggregate_repetitions(
                batch_summaries,
                failure_rates,
            ),
            "artifacts": {
                "transaction_manifest": str(transaction_manifest_path),
                "evaluation_trials_csv": str(csv_path),
                "summary_json": str(summary_path),
                "report": str(core.REPORT_FILE),
                "learning_curves": [item["chart"] for item in learning_curve_summaries],
            },
        }
        _write_json(summary_path, final_summary)
        _write_json(
            run_dir / "run_status.json",
            {"status": "COMPLETED", "stage": "FINISHED",
             "completed_trials": completed_trial_count,
             "expected_trials": expected_trial_count,
             "run_directory": str(run_dir)},
        )

        report.section("FINAL RESULT")
        report.item("Run status", "COMPLETED")
        report.item("Unique held-out transactions", evaluation_count)
        report.item("Training/evaluation split", "7,000 / 3,000")
        report.item("Training seeds", ", ".join(map(str, final_summary["protocol"]["training_seeds"])))
        report.item("Completed evaluation trials", completed_trial_count)
        report.item("Total PPO training seconds", round(total_training_seconds, 3))
        report.item("Trial CSV", csv_path)
        report.item("Summary JSON", summary_path)
        report.item("Train/test learning curves", learning_curve_dir)
        report.save()

        print("\nMAIN4 FINISHED")
        print("Run status: COMPLETED")
        print(f"Unique held-out transactions: {evaluation_count:,}")
        print(f"Completed evaluation trials: {completed_trial_count:,}")
        print(f"Report: {core.REPORT_FILE}")
        print(f"CSV: {csv_path}")
        print(f"Summary: {summary_path}")
        return final_summary

    except KeyboardInterrupt as exc:
        _write_json(
            run_dir / "run_status.json",
            {"status": "INTERRUPTED", "stage": str(getattr(core, "CURRENT_STAGE", "unknown")),
             "completed_trials": completed_trial_count,
             "expected_trials": expected_trial_count,
             "run_directory": str(run_dir)},
        )
        core.write_runtime_error(exc, report, run_status="INTERRUPTED")
        print(f"MAIN4 interrupted. Partial artifacts are in {run_dir}")
        raise
    except Exception as exc:
        _write_json(
            run_dir / "run_status.json",
            {"status": "FAILED", "stage": str(getattr(core, "CURRENT_STAGE", "unknown")),
             "completed_trials": completed_trial_count,
             "expected_trials": expected_trial_count,
             "error": f"{type(exc).__name__}: {exc}",
             "run_directory": str(run_dir)},
        )
        core.write_runtime_error(exc, report, run_status="FAILED")
        print(f"MAIN4 failed: {type(exc).__name__}: {exc}")
        print(f"Detailed report: {core.REPORT_FILE}")
        raise
    finally:
        core.stop_heartbeat()


if __name__ == "__main__":
    try:
        main4()
    except Exception:
        sys.exit(1)
