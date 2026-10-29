"""Paired four-arm RL/Bucket evaluation on a shared test set."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from Pathfinding.dijkstra import Dijkstra
from Pathfinding.heuristics import lnd_cost
from Pathfinding.top_k_paths import top_k_paths
from RL.environment import RoutingEnv
from Simulation.failure_model import FailureModel, assign_failure_probabilities
from Simulation.network_dynamics import NetworkDynamics
from Simulation.payment_simulator import PaymentSimulator
from Simulation.transaction_generator import generate_transaction_dataset


METHODS = ("baseline", "rl_only", "bucket_only", "rl_bucket")
DEFAULT_FAILURE_RATES = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
DEFAULT_DELAY_PERCENTILES = (90, 95)
STATIC_LND_ETA = 0.27876


class _FixedEtaModel:
    """Small adapter so the existing RL+Bucket pipeline can run fixed eta."""

    def __init__(self, eta):
        self.eta = float(eta)

    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return np.asarray([self.eta], dtype=np.float32), None


def generate_heldout_transactions(G, cfg, count=None, seed=None):
    """Generate the configured transaction set and return its held-out part."""
    evaluation_cfg = cfg.get("evaluation", {})
    count = int(
        evaluation_cfg.get("transaction_count", 3000) if count is None else count
    )
    total = int(cfg.get("n_transactions", 10_000))
    base_seed = int(cfg.get("seed", 42) if seed is None else seed)
    train_ratio = float(cfg.get("train_ratio", 0.7))
    simulation_cfg = cfg.get("simulation", {})

    dataset = generate_transaction_dataset(
        G=G,
        n=total,
        seed=base_seed,
        min_amount=float(simulation_cfg.get("min_amount", 1_000)),
        max_amount=float(simulation_cfg.get("max_amount", 1_000_000)),
        train_ratio=train_ratio,
    )
    if len(dataset.test) < count:
        raise ValueError(
            f"Held-out split has {len(dataset.test)} transactions; "
            f"{count} were requested. Increase n_transactions or lower "
            "evaluation.transaction_count."
        )
    return copy.deepcopy(dataset.test[:count])


def _scenario_seed(base_seed, rate_index, repetition):
    return int(base_seed + 1_000_003 * repetition + 100_000_007 * rate_index)


def _run_baseline(G, transactions, cfg, seed):
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.setdefault("simulation", {})["seed"] = int(seed)
    simulation_cfg = runtime_cfg.get("simulation", {})
    failure_model = FailureModel(
        node_failure_probability=float(
            simulation_cfg.get("node_failure_probability", 0.01)
        ),
        liquidity_failure_probability=float(
            simulation_cfg.get("liquidity_failure_probability", 0.05)
        ),
        seed=seed,
    )
    dynamics = NetworkDynamics(
        G=G,
        channel_failure_rate=float(simulation_cfg.get("channel_failure_rate", 0.01)),
        node_failure_rate=float(simulation_cfg.get("node_failure_rate", 0.005)),
        recovery_rate=float(simulation_cfg.get("recovery_rate", 0.05)),
        seed=seed,
        default_capacity=runtime_cfg.get("channel", {}).get("default_capacity"),
    )
    simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=dynamics,
    )
    router = Dijkstra(G)
    eta = STATIC_LND_ETA
    rows = []

    for tx in transactions:
        dynamics.reset(reset_balances=False)
        failure_model.reset_runtime_state(
            G,
            reset_counters=False,
            reset_rng=False,
        )
        route = router.shortest_path(
            tx.source,
            tx.destination,
            tx.amount,
            lnd_cost,
            eta,
            int(runtime_cfg.get("graph", {}).get("max_hops", 12)),
        )
        if not route.get("success") or not route.get("edges"):
            row = {
                "transaction_id": getattr(tx, "tx_id", None),
                "success": False,
                "path": [],
                "path_length": 0,
                "delay": 0.0,
                "fee": 0.0,
                "carbon": 0.0,
                "eta": eta,
                "candidate_path_count": 0,
                "usable_candidate_count": 0,
                "bucket_size": 0,
                "attempt_count": 0,
            }
        else:
            result = simulator.simulate_payment(
                path=route["path"],
                edges=route["edges"],
                amount=tx.amount,
                tx_id=getattr(tx, "tx_id", None),
            )
            row = {
                "transaction_id": getattr(tx, "tx_id", None),
                "success": bool(result.success),
                "path": list(result.path or []),
                "path_length": len(result.edges or []),
                "delay": float(result.delay),
                "fee": float(result.fee),
                "carbon": float(result.carbon),
                "eta": eta,
                "candidate_path_count": 1,
                "usable_candidate_count": 1,
                "bucket_size": 0,
                "attempt_count": 1,
            }
        row["path_length"] = int(row.get("path_length", 0))
        rows.append(row)
        # Match RoutingEnv's per-episode dynamics update while allowing the
        # next transaction to start from a reset runtime state.
        dynamics.update()
    return rows


def _run_rl_only(G, transactions, cfg, model, seed):
    """Use PPO to choose eta, then route and execute one path without Bucket."""
    env = RoutingEnv(G=G, transactions=list(transactions), config=cfg)
    rows = []
    try:
        for index, tx in enumerate(transactions):
            observation, _ = env.reset(
                seed=seed if index == 0 else None,
                options={"transaction_index": index},
            )
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_array.size != 1:
                raise ValueError("RL model must return exactly one eta value.")
            eta = float(action_array[0])
            candidates = top_k_paths(
                G=G,
                source=tx.source,
                target=tx.destination,
                amount=tx.amount,
                heuristic_fn=lnd_cost,
                eta=eta,
                k=env.top_k,
                max_hops=env.max_hops,
                lambda_h=env.lambda_h,
            )
            if not candidates:
                row = {
                    "transaction_id": getattr(tx, "tx_id", None),
                    "success": False,
                    "path": [],
                    "path_length": 0,
                    "delay": 0.0,
                    "fee": 0.0,
                    "carbon": 0.0,
                    "eta": eta,
                    "candidate_path_count": 0,
                    "usable_candidate_count": 0,
                    "bucket_size": 0,
                    "attempt_count": 0,
                }
            else:
                # The project fixes Top-K=5. RL-only suppresses all but the
                # best route at execution time, so it does not use a Bucket.
                candidate = candidates[0]
                result = env.payment_simulator.simulate_payment(
                    path=candidate["path"],
                    edges=candidate["edges"],
                    amount=tx.amount,
                    tx_id=getattr(tx, "tx_id", None),
                )
                row = {
                    "transaction_id": getattr(tx, "tx_id", None),
                    "success": bool(result.success),
                    "path": list(result.path or []),
                    "path_length": len(result.edges or []),
                    "delay": float(result.delay),
                    "fee": float(result.fee),
                    "carbon": float(result.carbon),
                    "eta": eta,
                    "candidate_path_count": 1,
                    "usable_candidate_count": 1,
                    "bucket_size": 0,
                    "attempt_count": 1,
                }
            row["path_length"] = int(row.get("path_length", 0))
            rows.append(row)
            env._update_network_dynamics()
    finally:
        env.close()
    return rows


def _run_bucketed(G, transactions, cfg, model, seed):
    """Run the existing Top-K, Bucket, and recovery pipeline per transaction."""
    env = RoutingEnv(G=G, transactions=list(transactions), config=cfg)
    rows = []
    try:
        for index, tx in enumerate(transactions):
            observation, _ = env.reset(
                seed=seed if index == 0 else None,
                options={"transaction_index": index},
            )
            action, _ = model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_array.size != 1:
                raise ValueError("Routing model must return exactly one eta value.")
            _next_obs, reward, terminated, truncated, info = env.step(action_array)
            info = dict(info or {})
            rows.append({
                "transaction_id": getattr(tx, "tx_id", index),
                "tx_id": getattr(tx, "tx_id", index),
                "success": bool(info.get("success", False)),
                "path": list(info.get("path", []) or []),
                "path_length": int(info.get("path_length", 0)),
                "delay": float(info.get("delay", 0.0)),
                "fee": float(info.get("fee", 0.0)),
                "carbon": float(info.get("carbon", 0.0)),
                "eta": float(info.get("eta", action_array[0])),
                "candidate_path_count": int(info.get("candidate_path_count", 0)),
                "usable_candidate_count": int(info.get("usable_candidate_count", 0)),
                "bucket_size": int(info.get("bucket_size", 0)),
                "attempt_count": int(info.get("attempt_count", 0)),
                "partial_backtrack_count": int(info.get("partial_backtrack_count", 0)),
                "full_reroute_count": int(info.get("full_reroute_count", 0)),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        env.close()
    return rows


def _percentile(values, percentile):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(values, percentile)) if values else None


def summarize_arm(rows):
    total = len(rows)
    successful = [row for row in rows if row.get("success")]
    routed = [row for row in rows if int(row.get("path_length", 0)) > 0]
    routed_delays = [row.get("delay") for row in routed]
    successful_delays = [row.get("delay") for row in successful]
    return {
        "transactions": total,
        "successes": len(successful),
        "success_rate": len(successful) / total if total else 0.0,
        "average_delay_success": (
            float(np.mean(successful_delays)) if successful_delays else None
        ),
        "p90_delay_success": _percentile(successful_delays, 90),
        "p95_delay_success": _percentile(successful_delays, 95),
        "average_delay_routed": float(np.mean(routed_delays)) if routed_delays else None,
        "p90_delay_routed": _percentile(routed_delays, 90),
        "p95_delay_routed": _percentile(routed_delays, 95),
        "average_fee_success": (
            float(np.mean([row.get("fee", 0.0) for row in successful]))
            if successful else None
        ),
        "average_carbon_success": (
            float(np.mean([row.get("carbon", 0.0) for row in successful]))
            if successful else None
        ),
    }


def _mean_confidence_interval(values, confidence=0.95):
    """Student-t interval over independent repetition-level measurements."""
    values = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(np.mean(values))
    if values.size == 1:
        return {"mean": mean, "ci_low": None, "ci_high": None, "n": 1}
    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=values.size - 1))
    margin = critical * sem
    return {
        "mean": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
        "n": int(values.size),
    }


def _aggregate_repetitions(repetition_summaries, confidence=0.95):
    metrics = (
        "success_rate",
        "average_delay_success",
        "p90_delay_success",
        "p95_delay_success",
        "average_delay_routed",
        "p90_delay_routed",
        "p95_delay_routed",
        "average_fee_success",
        "average_carbon_success",
    )
    return {
        metric: _mean_confidence_interval(
            [item.get(metric) for item in repetition_summaries], confidence
        )
        for metric in metrics
    }


def run_paired_ablation(
    G,
    transactions,
    cfg,
    rl_model,
    failure_rates=None,
    repetitions=None,
    seed=None,
    output_dir=None,
):
    """Compare baseline, RL only, Bucket only, and RL+Bucket on paired cases.

    Each arm receives the same held-out transaction ordering and the same
    deterministic seed for each ``(failure rate, repetition)`` scenario.
    Returns aggregate summaries and transaction-level rows.
    """
    if G is None or rl_model is None:
        raise ValueError("A graph and trained RL model are required.")
    transactions = list(transactions)
    if not transactions:
        raise ValueError("The held-out transaction list cannot be empty.")

    evaluation_cfg = cfg.get("evaluation", {})
    expected = evaluation_cfg.get("transaction_count")
    if expected is not None and len(transactions) != int(expected):
        raise ValueError(
            f"Expected {int(expected)} held-out transactions; got {len(transactions)}."
        )
    rates = tuple(float(rate) for rate in (
        failure_rates if failure_rates is not None
        else cfg.get("failure_rates", DEFAULT_FAILURE_RATES)
    ))
    if not rates or any(not 0.0 <= rate <= 1.0 for rate in rates):
        raise ValueError("Failure rates must be a non-empty sequence in [0, 1].")
    repeats = int(
        evaluation_cfg.get("repetitions", 5)
        if repetitions is None else repetitions
    )
    if repeats <= 0:
        raise ValueError("repetitions must be positive.")
    confidence = float(evaluation_cfg.get("confidence_interval", 0.95))
    if not 0.0 < confidence < 1.0:
        raise ValueError("evaluation.confidence_interval must be in (0, 1).")
    base_seed = int(cfg.get("seed", 42) if seed is None else seed)

    raw_rows = []
    summaries = {}
    static_model = _FixedEtaModel(STATIC_LND_ETA)
    for rate_index, rate in enumerate(rates):
        summaries[rate] = {}
        for repetition in range(1, repeats + 1):
            scenario_seed = _scenario_seed(base_seed, rate_index, repetition)
            for method in METHODS:
                method_cfg = copy.deepcopy(cfg)
                sim_cfg = method_cfg.setdefault("simulation", {})
                sim_cfg["channel_failure_rate"] = rate
                sim_cfg["failure_rate"] = rate
                sim_cfg["seed"] = scenario_seed
                method_graph = copy.deepcopy(G)
                assign_failure_probabilities(method_graph, rate, scenario_seed)

                if method == "baseline":
                    rows = _run_baseline(
                        method_graph, transactions, method_cfg, scenario_seed
                    )
                elif method == "rl_only":
                    rows = _run_rl_only(
                        method_graph, transactions, method_cfg, rl_model, scenario_seed
                    )
                elif method == "bucket_only":
                    rows = _run_bucketed(
                        method_graph, transactions, method_cfg, static_model, scenario_seed
                    )
                else:
                    rows = _run_bucketed(
                        method_graph, transactions, method_cfg, rl_model, scenario_seed
                    )

                for index, row in enumerate(rows):
                    row.update({
                        "method": method,
                        "failure_rate": rate,
                        "repetition": repetition,
                        "scenario_seed": scenario_seed,
                        "transaction_id": row.get(
                            "transaction_id", getattr(transactions[index], "tx_id", index)
                        ),
                    })
                if len(rows) != len(transactions):
                    raise RuntimeError(
                        f"{method} returned {len(rows)} rows for "
                        f"{len(transactions)} transactions."
                    )
                raw_rows.extend(rows)
                summaries[rate].setdefault(method, []).append({
                    "repetition": repetition,
                    "scenario_seed": scenario_seed,
                    **summarize_arm(rows),
                })

        summaries[rate] = {
            method: {
                "repetitions": method_results,
                "aggregate": _aggregate_repetitions(method_results, confidence),
            }
            for method, method_results in summaries[rate].items()
        }
        baseline_repetitions = summaries[rate]["baseline"]["repetitions"]
        summaries[rate]["paired_differences_vs_baseline"] = {}
        for method in METHODS:
            if method == "baseline":
                continue
            method_repetitions = summaries[rate][method]["repetitions"]
            differences = {}
            for metric in ("success_rate", "p95_delay_success", "p95_delay_routed"):
                paired_values = []
                for method_result, baseline_result in zip(
                    method_repetitions, baseline_repetitions
                ):
                    method_value = method_result.get(metric)
                    baseline_value = baseline_result.get(metric)
                    if method_value is not None and baseline_value is not None:
                        paired_values.append(method_value - baseline_value)
                differences[metric] = _mean_confidence_interval(
                    paired_values, confidence
                )
            summaries[rate]["paired_differences_vs_baseline"][method] = differences

    result = {
        "metadata": {
            "methods": list(METHODS),
            "failure_rates": list(rates),
            "repetitions": repeats,
            "confidence_interval": confidence,
            "unique_test_transactions": len(transactions),
            "base_seed": base_seed,
            "paired_scenario_seeds": True,
            "baseline_eta": STATIC_LND_ETA,
            "delay_metric": "sum of edge delay/CLTV values on the selected route",
            "delay_percentiles": list(DEFAULT_DELAY_PERCENTILES),
        },
        "summaries": summaries,
        "rows": raw_rows,
    }
    if output_dir is not None:
        _write_results(result, output_dir)
    return result


def run_configured_ablation(G, cfg, rl_model, output_dir="Evaluations/results/paired_ablation"):
    """Generate the shared held-out split, then run the configured study."""
    evaluation_cfg = cfg.get("evaluation", {})
    count = int(evaluation_cfg.get("transaction_count", 3000))
    transactions = generate_heldout_transactions(G, cfg, count=count)
    return run_paired_ablation(
        G=G,
        transactions=transactions,
        cfg=cfg,
        rl_model=rl_model,
        output_dir=output_dir,
    )


def _write_results(result, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "paired_ablation.json"
    csv_path = output_path / "paired_ablation_rows.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result["rows"]
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "METHODS",
    "generate_heldout_transactions",
    "run_paired_ablation",
    "run_configured_ablation",
    "summarize_arm",
]
