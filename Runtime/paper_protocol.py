"""Internal full-snapshot protocol used by main-test.py.

This entry point creates one 10,000-transaction pool, trains five PPO models
with distinct seeds on the same 7,000-row training partition, and evaluates
each model on the same 3,000 held-out transactions. Every configured failure
rate is evaluated for each seed. The run is intentionally full-sized and
does not use the fast-validation model or settings.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import closing
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from Runtime.recovery_reporting import RECOVERY_COUNT_FIELDS, RECOVERY_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        default=PROJECT_ROOT / "theResult",
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
        "--repetitions",
        type=int,
        default=PAPER_REPETITIONS,
        help=(
            "Number of independent PPO training/evaluation repetitions "
            f"(default: {PAPER_REPETITIONS}, the paper protocol)."
        ),
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume an interrupted run from this exact run directory.",
    )
    parser.add_argument(
        "--evaluation-workers", type=int, choices=range(1, 31), default=1,
        help="Parallel held-out evaluation processes (1-30; independent of training workers).",
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


def _validate_protocol(cfg, failure_rates, repetitions):
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
            "MAIN TEST enforces the article's 10,000 / 7,000 / 3,000 / 5-run "
            "protocol. Correct Configs/config.yaml before starting: "
            + "; ".join(invalid)
        )

    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("--repetitions must be a positive integer.")

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
                for key in RECOVERY_COUNT_FIELDS:
                    result[key] = int(result.get(key) or 0)
                result["recovery_events"] = json.loads(result.get("recovery_events") or "[]")
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
        # predict() accepts a single observation even when PPO was trained with
        # multiple environments. Keep the training env attached: set_env() would
        # reject this single evaluation env when model.n_envs is greater than 1.
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
                "routing_scope": info.get("routing_scope"),
                "local_routing_radius": info.get("local_routing_radius"),
                "local_routing_node_count": info.get("local_routing_node_count"),
                "local_routing_channel_count": info.get("local_routing_channel_count"),
                "path_search_attempts": int(info.get("path_search_attempts", 0)),
                "path_search_timed_out": bool(
                    info.get("path_search_timed_out", False)
                ),
                "path_search_fallback_attempted": bool(
                    info.get("path_search_fallback_attempted", False)
                ),
                "path_search_fallback_used": bool(
                    info.get("path_search_fallback_used", False)
                ),
                "path_search_elapsed_seconds": float(
                    info.get("path_search_elapsed_seconds", 0.0)
                ),
                "path_search_timeout_stages": list(
                    info.get("path_search_timeout_stages", [])
                ),
            }
        )
        return result
    finally:
        runtime_env.close()


def _batch_trial_results(core, graph, cfg, model, model_path, transactions,
                        training_seed, repetition, failure_rate, failure_rate_index,
                        existing_trials, workers, report_dir):
    tasks = []
    for index, transaction in enumerate(transactions, start=1):
        seed = int(cfg["seed"]) + repetition * 10_000_000 + failure_rate_index * 100_000 + index
        key = (repetition, round(failure_rate, 10), seed)
        cached = existing_trials.get(key)
        if cached is not None:
            yield index, cached, False
        else:
            tasks.append((transaction, index, training_seed, repetition, failure_rate, failure_rate_index))
    if not tasks:
        return
    if workers == 1:
        for transaction, index, training_seed, repetition, rate, rate_index in tasks:
            yield index, _run_trial(
                core, graph, cfg, model, transaction, index,
                training_seed, repetition, rate, rate_index,
            ), True
    else:
        from Evaluations.parallel_trials import (
            completed_trials, initialize_trial_worker, run_trial_worker,
        )
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        with closing(completed_trials(
            run_trial_worker, tasks, workers, initializer=initialize_trial_worker,
            initargs=(graph, cfg, str(model_path), str(report_dir)),
        )) as results:
            for index, result in results:
                yield index, result, True


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
    "recovery_version",
    *RECOVERY_COUNT_FIELDS,
    "recovery_events",
    "full_reroute_count",
    "path_length",
    "prediction_time",
    "pipeline_time",
    "routing_scope",
    "local_routing_radius",
    "local_routing_node_count",
    "local_routing_channel_count",
    "path_search_attempts",
    "path_search_timed_out",
    "path_search_fallback_attempted",
    "path_search_fallback_used",
    "path_search_elapsed_seconds",
    "path_search_timeout_stages",
    "path",
]


def _csv_row(result):
    row = {field: result.get(field) for field in TRIAL_FIELDS}
    row["path"] = json.dumps(result.get("path", []), ensure_ascii=False, default=_json_default)
    row["recovery_version"] = result.get("recovery_version", RECOVERY_VERSION)
    row["recovery_events"] = json.dumps(
        result.get("recovery_events", []), ensure_ascii=False, default=_json_default,
    )
    for key in RECOVERY_COUNT_FIELDS:
        row[key] = result.get(key, 0)
    row["path_search_timeout_stages"] = json.dumps(
        result.get("path_search_timeout_stages", []), ensure_ascii=False
    )
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
        "average_path_search_seconds",
        "average_local_routing_nodes",
        "average_local_routing_channels",
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
        for field in (
            "path_search_timeout_transactions",
            "path_search_fallback_attempted_transactions",
            "path_search_fallback_used_transactions",
            "total_partial_backtracks",
            "total_partial_backtrack_success",
            "total_full_reroutes",
            *("total_" + key for key in RECOVERY_COUNT_FIELDS),
        ):
            summaries[f"{rate:.8g}"][field] = sum(
                batch["aggregate"].get(field, 0) for batch in rate_batches
            )
        summaries[f"{rate:.8g}"]["average_path_search_seconds"] = _mean_ci95(
            [batch["aggregate"].get("average_path_search_seconds") for batch in rate_batches]
        )
    return summaries


def run_protocol(argv=None):
    args = _parse_args(argv)
    snapshot_path = args.snapshot.resolve()
    config_path = args.config.resolve()
    output_root = args.output_dir.resolve()
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
        # Reject legacy results before importing reporting helpers or writing
        # any run/report files. Old evaluations must remain historical.
        saved_metadata_path = run_dir / "run_metadata.json"
        if saved_metadata_path.is_file():
            saved_recovery = json.loads(
                saved_metadata_path.read_text(encoding="utf-8")
            ).get("recovery_version")
            if saved_recovery != RECOVERY_VERSION:
                raise ValueError(
                    "Cannot resume results from an older recovery implementation. "
                    "Start a new run with local-prefix-v2; the historical results are preserved."
                )
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

    from Runtime import experiment_core as core

    core.REPORT_FILE = FINAL_REPORT_PATH
    report = core.ReportWriter(core.REPORT_FILE)
    core.RUN_START_TIME = time.time()
    core.set_stage("MAIN TEST - PAPER-PROTOCOL INITIALIZATION")

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
        # MAIN TEST always uses the paper baseline reward.
        cfg["reward"] = dict(cfg.get("reward", {}))
        cfg["reward"]["formula"] = "paper_base"

        failure_rates = (
            [float(rate) for rate in args.failure_rates]
            if args.failure_rates is not None
            else [float(rate) for rate in cfg.get("failure_rates", [])]
        )
        repetitions = args.repetitions
        _validate_protocol(cfg, failure_rates, repetitions)

        base_seed = int(cfg["seed"])
        total_training_count = PAPER_TRANSACTION_COUNT
        evaluation_count = PAPER_EVALUATION_COUNT
        expected_trial_count = repetitions * evaluation_count * len(failure_rates)

        run_metadata = {
            "snapshot_sha256": _sha256_file(snapshot_path),
            "config_sha256": _sha256_file(config_path),
            "base_seed": base_seed,
            "failure_rates": failure_rates,
            "training_env_workers": int(os.environ.get("RL_ENV_WORKERS", "1")),
            "total_transactions": total_training_count,
            "training_transactions": PAPER_TRAINING_COUNT,
            "held_out_transactions": evaluation_count,
            "repetitions": repetitions,
            "reward_formula": "paper_base",
            "recovery_version": RECOVERY_VERSION,
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
                    "seed, failure rates, recovery implementation, or paper protocol changed). "
                    "Start a new run to avoid mixing old and corrected routing results."
                )
        else:
            _write_json(metadata_path, run_metadata)

        _write_json(
            run_dir / "run_status.json",
            {"status": "RUNNING", "stage": "INITIALIZING", "completed_trials": 0,
             "expected_trials": expected_trial_count, "run_directory": str(run_dir)},
        )

        report.section("MAIN TEST - ARTICLE-SIZED RL EXPERIMENT")
        report.item("Run status", "RUNNING")
        report.item(
            "Execution mode",
            "FULL ARTICLE-SIZED RUN WITH OPERATIONAL PATH-SEARCH DEADLINES",
        )
        report.item("Reward formula", "paper_base (locked to article baseline)")
        report.item("Recovery implementation", RECOVERY_VERSION)
        report.item(
            "Recovery policy",
            "Nearest reached branch first; cached Bucket suffix then local Top-K search; "
            "reuse the forwarded prefix; retry from source only after local recovery is exhausted",
        )
        report.item(
            "Routing search scope",
            "Induced local graph around the source/destination endpoints",
        )
        report.item(
            "Local routing radius (configured m)",
            cfg.get("graph", {}).get("neighborhood_m", 5),
        )
        report.item(
            "Local routing scope note",
            "Radius follows the paper's m=5 setting; restricting candidate search to this local union is a project variant, while the paper describes local observations and complete deterministic pathfinding",
        )
        pathfinding_limits = cfg.get("pathfinding", {})
        report.item(
            "Path-search deadline settings",
            (
                f"graph build={pathfinding_limits.get('graph_build_timeout_seconds', 0)}s; "
                f"Top-K enumeration={pathfinding_limits.get('candidate_search_timeout_seconds', 0)}s; "
                f"fallback={pathfinding_limits.get('fallback_timeout_seconds', 0)}s"
            ),
        )
        report.item(
            "Path-search deadline interpretation",
            "Operational extension; set all budgets to 0 for unbounded paper comparison",
        )
        report.item("Snapshot", snapshot_path)
        report.item("Config", config_path)
        report.item("Base seed", base_seed)
        report.item("Generated transaction pool", total_training_count)
        report.item("Training transactions per repetition", PAPER_TRAINING_COUNT)
        report.item("Unique held-out transactions", evaluation_count)
        report.item("Training/evaluation split", "70% / 30%")
        report.item("Independent training/evaluation repetitions", repetitions)
        report.item("Parallel evaluation processes", args.evaluation_workers)
        report.item(
            "Parallel training environments",
            int(os.environ.get("RL_ENV_WORKERS", "1")),
        )
        report.item("Failure rates", ", ".join(f"{rate:.0%}" for rate in failure_rates))
        report.item("Expected evaluation trials", expected_trial_count)
        report.item("Fast validation environment flags", "IGNORED; full training forced")
        report.item("Run output directory", run_dir)
        report.item("Final text report", FINAL_REPORT_PATH)
        report.save()

        print("MAIN TEST: FULL ARTICLE-SIZED RUN")
        print(f"Transactions: {PAPER_TRANSACTION_COUNT} = {PAPER_TRAINING_COUNT} train + {PAPER_EVALUATION_COUNT} held-out")
        print(f"Replications: {repetitions}; failure rates: {', '.join(f'{rate:.0%}' for rate in failure_rates)}")
        print(f"Expected evaluation trials: {expected_trial_count:,}")
        print(f"Output directory: {run_dir}")

        core.set_stage("MAIN TEST - LOAD SNAPSHOT AND BUILD GRAPH")
        cfg["rl"] = dict(cfg.get("rl", {}))
        cfg["rl"]["device"] = cfg["rl"].get("device", "auto")
        graph_data = core.geo_to_json(snapshot_path)
        graph = core.LNGraphBuilder(
            default_capacity=cfg.get("channel", {}).get("default_capacity")
        ).from_data(graph_data)
        report.item("Graph nodes", graph.number_of_nodes())
        report.item("Graph directed channels", graph.number_of_edges())
        report.save()

        core.set_stage("MAIN TEST - LOAD OR GENERATE FIXED TRANSACTION POOL")
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
        report.add(
            "The same held-out transaction IDs, sources, destinations, and amounts "
            f"are reused for all {repetitions} repetition(s) and every configured failure rate."
        )
        if repetitions == 1:
            report.item(
                "Statistical interpretation",
                "Preliminary single-seed run; across-seed variation and confidence intervals cannot be estimated.",
            )
        else:
            report.item(
                "Statistical interpretation",
                f"{repetitions} independent training seeds; uncertainty summaries are calculated across repetitions.",
            )
        report.save()

        model_root = run_dir / "models"
        log_root = run_dir / "logs"
        model_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        training_failure_rate = float(cfg.get("training_failure_rate", 0.03))
        report.item(
            "Training-time 50-transaction probes",
            "Disabled; evaluation uses the complete 3,000-transaction held-out set after training.",
        )
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
                    f"MAIN TEST - TRAIN MODEL {repetition}/{repetitions} (SEED {training_seed})"
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
                if final_model_path.is_file():
                    print(f"Using completed model from {final_model_path}", flush=True)
                    if args.evaluation_workers > 1:
                        core._install_ppo_loader_compatibility()
                    model = core.PPO.load(
                        str(final_model_path), device=(
                            "cpu" if args.evaluation_workers > 1
                            else repetition_cfg["rl"].get("device", "auto")
                        )
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
                    )
                    training_seconds = time.perf_counter() - training_start
                    total_training_seconds += training_seconds
                    del training_graph

                report.section(f"TRAINING REPETITION {repetition}/{repetitions}")
                report.item("Training seed", training_seed)
                report.item("Training rows", len(train_transactions))
                report.item("Training failure rate", training_failure_rate)
                report.item("PPO training mode", "FULL; fast_training=False")
                report.item("PPO training time (seconds)", round(training_seconds, 4))
                report.item("Model", model_dir / f"{model_name}.zip")
                report.save()
                for failure_rate_index, failure_rate in enumerate(failure_rates):
                    core.set_stage(
                        f"MAIN TEST - EVALUATE REP {repetition}/{repetitions}, "
                        f"FAILURE RATE {failure_rate:.0%}"
                    )
                    print(
                        f"Evaluating repetition {repetition}/{repetitions}, "
                        f"failure rate {failure_rate:.0%}: "
                        f"{evaluation_count:,} fixed held-out transactions; "
                        f"{args.evaluation_workers} evaluation workers",
                        flush=True,
                    )
                    batch_results_by_index = {}
                    batch_start = time.perf_counter()
                    last_progress_time = batch_start

                    results = _batch_trial_results(
                        core, graph, cfg, model, final_model_path, held_out_transactions,
                        training_seed, repetition, failure_rate, failure_rate_index,
                        existing_trials, args.evaluation_workers,
                        log_root / f"evaluation_rep_{repetition:02d}",
                    )
                    try:
                        for batch_completed, (transaction_index, result, is_new) in enumerate(results, start=1):
                            if is_new:
                                trial_key = (repetition, round(failure_rate, 10), result["evaluation_seed"])
                                trial_writer.writerow(_csv_row(result))
                                trial_file.flush()
                                existing_trials[trial_key] = result
                                completed_trial_count += 1
                            batch_results_by_index[transaction_index] = result

                            now = time.perf_counter()
                            if (
                                batch_completed % 25 == 0
                                or now - last_progress_time >= 60.0
                                or batch_completed >= evaluation_count
                            ):
                                trial_file.flush()
                                batch_elapsed = max(now - batch_start, 1e-9)
                                rate = batch_completed / batch_elapsed
                                remaining_seconds = (
                                    (evaluation_count - batch_completed) / rate
                                    if rate > 0 else 0.0
                                )
                                batch_percent = batch_completed / evaluation_count * 100.0
                                print(
                                    "\n[EVALUATION PROGRESS — ACTIVE]"
                                    "\nStage       : HELD-OUT TEST / ROUTING EVALUATION"
                                    f"\nRepetition  : {repetition}/{repetitions} (seed {training_seed})"
                                    f"\nFailure rate: {failure_rate:.0%}"
                                    f"\nWorkers     : {args.evaluation_workers}"
                                    f"\nCompleted   : {batch_completed:,}/{evaluation_count:,}"
                                    f"\nBatch       : {batch_percent:.2f}%"
                                    f"\nOverall     : {completed_trial_count:,}/{expected_trial_count:,} evaluation trials"
                                    f"\nElapsed     : {core._format_elapsed(batch_elapsed)}"
                                    f"\nRemaining   : {core._format_elapsed(remaining_seconds)} (estimated for this batch)"
                                    "\nAgent state : trained policy is routing held-out payments"
                                    "\nProcess     : ACTIVE; an evaluation result was recorded",
                                    flush=True,
                                )
                                last_progress_time = now
                    finally:
                        results.close()

                    trial_file.flush()
                    batch_results = [batch_results_by_index[index] for index in sorted(batch_results_by_index)]
                    del batch_results_by_index
                    batch_aggregate = core.aggregate_evaluation_results(batch_results)
                    batch_aggregate.update(
                        {
                            "path_search_timeout_transactions": sum(
                                bool(row.get("path_search_timed_out", False))
                                for row in batch_results
                            ),
                            "path_search_fallback_attempted_transactions": sum(
                                bool(row.get("path_search_fallback_attempted", False))
                                for row in batch_results
                            ),
                            "path_search_fallback_used_transactions": sum(
                                bool(row.get("path_search_fallback_used", False))
                                for row in batch_results
                            ),
                            "average_path_search_seconds": (
                                sum(
                                    float(row.get("path_search_elapsed_seconds", 0.0))
                                    for row in batch_results
                                )
                                / max(1, len(batch_results))
                            ),
                            "average_local_routing_nodes": (
                                sum(
                                    int(row.get("local_routing_node_count", 0) or 0)
                                    for row in batch_results
                                )
                                / max(1, len(batch_results))
                            ),
                            "average_local_routing_channels": (
                                sum(
                                    int(row.get("local_routing_channel_count", 0) or 0)
                                    for row in batch_results
                                )
                                / max(1, len(batch_results))
                            ),
                        }
                    )
                    batch_summary = {
                        "repetition": repetition,
                        "training_seed": training_seed,
                        "failure_rate": failure_rate,
                        "transaction_count": len(batch_results),
                        "evaluation_workers": args.evaluation_workers,
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
                    for label, key in (
                        ("Recovery checks after failure", "total_partial_backtracks"),
                        ("Actual partial continuation attempts", "total_partial_backtrack_retry_count"),
                        ("Payments completed by partial continuation", "total_partial_backtrack_success"),
                        ("Retries starting at source", "total_source_retry_count"),
                        ("Forwarded prefix hops reused", "total_preserved_prefix_hops"),
                        ("Backtracked hops", "total_backtracked_hops"),
                        ("Full route recomputations", "total_full_reroutes"),
                    ):
                        report.item(label, batch_aggregate[key])
                    report.item("Per-payment recovery events", "evaluation_trials.csv: recovery_events (JSON)")
                    report.item(
                        "Transactions with a path-search timeout",
                        batch_aggregate["path_search_timeout_transactions"],
                    )
                    report.item(
                        "Transactions using the fallback route",
                        batch_aggregate["path_search_fallback_used_transactions"],
                    )
                    report.item(
                        "Average path-search time (seconds)",
                        batch_aggregate["average_path_search_seconds"],
                    )
                    report.item(
                        "Average local graph nodes",
                        batch_aggregate["average_local_routing_nodes"],
                    )
                    report.item(
                        "Average local graph directed channels",
                        batch_aggregate["average_local_routing_channels"],
                    )
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
            "failure_rate_summary_across_seeds": _aggregate_repetitions(
                batch_summaries,
                failure_rates,
            ),
            "artifacts": {
                "transaction_manifest": str(transaction_manifest_path),
                "evaluation_trials_csv": str(csv_path),
                "summary_json": str(summary_path),
                "report": str(core.REPORT_FILE),
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
        report.save()

        print("\nMAIN TEST FINISHED")
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
        print(f"MAIN TEST interrupted. Partial artifacts are in {run_dir}")
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
        print(f"MAIN TEST failed: {type(exc).__name__}: {exc}")
        print(f"Detailed report: {core.REPORT_FILE}")
        raise
    finally:
        core.stop_heartbeat()


