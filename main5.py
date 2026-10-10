"""Full-snapshot study comparing the paper reward with six one-factor variants.

Run from the project directory with ``python main5.py``. This is a full
research run: seven reward arms, five independently trained PPO repetitions
per arm, and the same fixed 7,000/3,000 transaction split in every arm.
"""

from __future__ import annotations

import copy
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import main4 as protocol


PROJECT_ROOT = Path(__file__).resolve().parent
ARMS = (
    ("paper_base", "Paper baseline"),
    ("amount", "Payment amount"),
    ("liquidity_margin", "Liquidity margin"),
    ("persistence", "Empirical persistence"),
    ("fee", "Fee penalty"),
    ("delay", "Delay penalty"),
    ("hops", "Hop penalty"),
)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=PROJECT_ROOT / "20190501.gml.geo"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "Configs" / "config.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "theResult",
    )
    parser.add_argument("--failure-rates", type=float, nargs="+")
    return parser.parse_args(argv)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=protocol._json_default)
        + "\n",
        encoding="utf-8",
    )


def _arm_summaries(batches, failure_rates):
    result = {}
    for formula, _label in ARMS:
        arm_batches = [item for item in batches if item["formula"] == formula]
        result[formula] = protocol._aggregate_repetitions(arm_batches, failure_rates)
    return result


def _paired_differences(batches, failure_rates):
    metrics = (
        "success_rate",
        "average_carbon",
        "average_delay",
        "p95_delay_success",
        "average_hops",
        "average_fee",
    )
    output = {}
    for rate in failure_rates:
        base = {
            item["repetition"]: item["aggregate"]
            for item in batches
            if item["formula"] == "paper_base"
            and abs(item["failure_rate"] - rate) <= 1e-12
        }
        rate_output = {}
        for formula, _label in ARMS:
            if formula == "paper_base":
                continue
            arm = {
                item["repetition"]: item["aggregate"]
                for item in batches
                if item["formula"] == formula
                and abs(item["failure_rate"] - rate) <= 1e-12
            }
            differences = {}
            for metric in metrics:
                paired = [
                    arm[rep][metric] - base[rep][metric]
                    for rep in sorted(set(arm) & set(base))
                    if arm[rep].get(metric) is not None
                    and base[rep].get(metric) is not None
                ]
                differences[metric] = protocol._mean_ci95(paired)
            rate_output[formula] = differences
        output[f"{rate:.8g}"] = rate_output
    return output


def main5(argv=None):
    args = _arguments(argv)
    snapshot = args.snapshot.resolve()
    config_path = args.config.resolve()
    output_root = args.output_dir.resolve()
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    os.environ["RL_FAST_VALIDATION"] = "0"
    os.environ["RL_FAST_TRAINING"] = "0"
    os.environ["RL_FAST_ENV"] = "0"
    os.environ["RL_REPORT_FILE"] = str(run_dir / "report.txt")

    import main as core

    core.REPORT_FILE = run_dir / "report.txt"
    report = core.ReportWriter(core.REPORT_FILE)
    core.RUN_START_TIME = time.time()
    core.set_stage("MAIN5 - REWARD FORMULA STUDY INITIALIZATION")
    failure_rates = []
    batches = []
    completed_trials = 0
    expected_trials = 0
    training_seconds_total = 0.0

    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not snapshot.is_file():
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must contain a YAML mapping.")
        failure_rates = (
            [float(value) for value in args.failure_rates]
            if args.failure_rates is not None
            else [float(value) for value in cfg.get("failure_rates", [])]
        )
        protocol._validate_protocol(cfg, failure_rates)

        base_seed = int(cfg["seed"])
        seeds = [base_seed + index for index in range(protocol.PAPER_REPETITIONS)]
        expected_trials = (
            len(ARMS)
            * protocol.PAPER_REPETITIONS
            * protocol.PAPER_EVALUATION_COUNT
            * len(failure_rates)
        )
        report.section("MAIN5 - REWARD FORMULA ABLATION")
        report.item("Status", "RUNNING")
        report.item("Snapshot path", snapshot)
        report.item("Snapshot scope", "FULL SNAPSHOT GRAPH; no node or channel subsampling")
        report.item("Graph protocol", "same complete graph for all seven reward arms")
        report.item("Article seed disclosure", "paper states seed is fixed but does not publish its numeric value")
        report.item("Project base seed used", base_seed)
        report.item("Five repetition seeds per formula", ", ".join(map(str, seeds)))
        report.item("Reward formula arms", ", ".join(name for name, _ in ARMS))
        report.item("Training/test transaction split", "7,000 / 3,000 from one 10,000-row pool")
        report.item("Failure rates", ", ".join(f"{rate:.0%}" for rate in failure_rates))
        report.item("PPO training sessions", len(ARMS) * protocol.PAPER_REPETITIONS)
        report.item("Expected evaluation trials", expected_trials)
        report.item("Fast modes", "FORCED OFF")
        report.item("Output directory", run_dir)
        report.add(
            "The fixed numeric base seed 42 comes from this project's configuration; "
            "the paper appendix only says the seed is fixed. Repetitions use base_seed + 0..4."
        )
        report.add(
            "A, M and P are operationalized as amount/reference, 1 + relative margin "
            "(capped at 100%) from the midpoint of agent-visible liquidity bounds, and Laplace-smoothed "
            "empirical route-channel success. P never reads simulator failure probabilities."
        )
        report.add(
            "Fee, delay, and hop additions are converted to carbon-equivalent units by "
            "the reward configuration's references and lambda coefficients."
        )
        report.save()

        core.set_stage("MAIN5 - LOAD FULL SNAPSHOT")
        cfg["rl"] = dict(cfg.get("rl", {}))
        cfg["rl"]["device"] = cfg["rl"].get("device", "auto")
        graph_data = core.geo_to_json(snapshot)
        graph = core.LNGraphBuilder(
            default_capacity=cfg.get("channel", {}).get("default_capacity")
        ).from_data(graph_data)
        report.item("Graph nodes loaded", graph.number_of_nodes())
        report.item("Directed channels loaded", graph.number_of_edges())
        report.save()

        core.set_stage("MAIN5 - GENERATE SHARED TRANSACTION POOL")
        core.set_seed(base_seed)
        transaction_pool = core.generate_transactions(
            graph,
            protocol.PAPER_TRANSACTION_COUNT,
            base_seed,
            cfg["simulation"]["min_amount"],
            cfg["simulation"]["max_amount"],
        )
        if len(transaction_pool) != protocol.PAPER_TRANSACTION_COUNT:
            raise RuntimeError(
                f"Expected {protocol.PAPER_TRANSACTION_COUNT} transactions, "
                f"got {len(transaction_pool)}."
            )
        train_transactions, held_out = core.split_transaction_pool(
            transaction_pool, protocol.PAPER_EVALUATION_COUNT
        )
        if len(train_transactions) != protocol.PAPER_TRAINING_COUNT:
            raise RuntimeError("Shared transaction pool did not yield 7,000 train rows.")
        if len(held_out) != protocol.PAPER_EVALUATION_COUNT:
            raise RuntimeError("Shared transaction pool did not yield 3,000 test rows.")
        manifest = run_dir / "transaction_split.csv"
        protocol._write_transaction_manifest(
            manifest, transaction_pool, len(train_transactions)
        )
        report.item("Training pool rows", len(train_transactions))
        report.item("Unique held-out rows", len(held_out))
        report.item("Shared transaction manifest", manifest)
        report.save()

        trial_path = run_dir / "evaluation_trials.csv"
        summary_path = run_dir / "summary.json"
        model_root = run_dir / "models"
        log_root = run_dir / "logs"
        model_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        trial_fields = ["reward_formula", *protocol.TRIAL_FIELDS]
        with trial_path.open("w", newline="", encoding="utf-8") as trial_file:
            writer = csv.DictWriter(trial_file, fieldnames=trial_fields)
            writer.writeheader()
            for formula_index, (formula, label) in enumerate(ARMS):
                formula_batches = []
                for repetition, training_seed in enumerate(seeds, start=1):
                    core.set_stage(
                        f"MAIN5 - TRAIN {formula} REP {repetition}/{len(seeds)}"
                    )
                    print(
                        f"\n[{formula_index + 1}/{len(ARMS)}] {label}: "
                        f"training repetition {repetition}/{len(seeds)}, seed {training_seed}"
                    )
                    core.set_seed(training_seed)
                    training_graph = copy.deepcopy(graph)
                    training_failure_rate = float(cfg.get("training_failure_rate", 0.03))
                    core.assign_failure_probabilities(
                        training_graph, training_failure_rate, training_seed
                    )
                    arm_cfg = copy.deepcopy(cfg)
                    arm_cfg["seed"] = training_seed
                    arm_cfg["rl"]["seed"] = training_seed
                    arm_cfg.setdefault("reward", {})["formula"] = formula
                    model_dir = model_root / formula / f"rep_{repetition:02d}_seed_{training_seed}"
                    arm_cfg["rl"]["model_dir"] = str(model_dir)
                    arm_cfg["rl"]["log_dir"] = str(log_root / formula / f"rep_{repetition:02d}")
                    arm_cfg["rl"]["checkpoint_dir"] = str(model_dir / "checkpoints")
                    model_name = f"reward_{formula}_seed_{training_seed}"
                    start = time.perf_counter()
                    model = core.train_agent(
                        training_graph,
                        copy.deepcopy(train_transactions),
                        core.lnd_cost,
                        arm_cfg,
                        model_name,
                        training_seed,
                        fast_training=False,
                    )
                    training_seconds = time.perf_counter() - start
                    training_seconds_total += training_seconds
                    report.section(f"TRAINED {formula} - REPETITION {repetition}")
                    report.item("Reward formula", formula)
                    report.item("Training seed", training_seed)
                    report.item("Full graph nodes/channels", f"{graph.number_of_nodes()} / {graph.number_of_edges()}")
                    report.item("Training rows", len(train_transactions))
                    report.item("PPO training seconds", round(training_seconds, 3))
                    report.item("Model", model_dir / f"{model_name}.zip")
                    report.save()
                    del training_graph

                    for failure_index, failure_rate in enumerate(failure_rates):
                        core.set_stage(
                            f"MAIN5 - EVALUATE {formula} REP {repetition} "
                            f"RATE {failure_rate:.0%}"
                        )
                        rows = []
                        batch_start = time.perf_counter()
                        for transaction_index, transaction in enumerate(held_out, start=1):
                            outcome = protocol._run_trial(
                                core,
                                graph,
                                arm_cfg,
                                model,
                                transaction,
                                transaction_index,
                                training_seed,
                                repetition,
                                failure_rate,
                                failure_index,
                            )
                            outcome["reward_formula"] = formula
                            rows.append(outcome)
                            writer.writerow(
                                {"reward_formula": formula, **protocol._csv_row(outcome)}
                            )
                            completed_trials += 1
                            if transaction_index % 100 == 0:
                                trial_file.flush()
                                print(
                                    f"  {formula}, rep {repetition}, {failure_rate:.0%}: "
                                    f"{transaction_index:,}/{len(held_out):,}; "
                                    f"overall {completed_trials:,}/{expected_trials:,}"
                                )
                        trial_file.flush()
                        aggregate = core.aggregate_evaluation_results(rows)
                        batch = {
                            "formula": formula,
                            "repetition": repetition,
                            "training_seed": training_seed,
                            "failure_rate": failure_rate,
                            "transaction_count": len(rows),
                            "elapsed_seconds": time.perf_counter() - batch_start,
                            "aggregate": aggregate,
                        }
                        batches.append(batch)
                        formula_batches.append(batch)
                        _write_json(
                            summary_path,
                            {
                                "protocol": {
                                    "snapshot": str(snapshot),
                                    "full_snapshot_graph": True,
                                    "base_seed": base_seed,
                                    "training_seeds_per_formula": seeds,
                                    "formulas": [name for name, _ in ARMS],
                                    "transaction_pool": 10_000,
                                    "training_transactions": 7_000,
                                    "held_out_transactions": 3_000,
                                    "repetitions_per_formula": 5,
                                    "failure_rates": failure_rates,
                                    "expected_trials": expected_trials,
                                    "completed_trials": completed_trials,
                                },
                                "formula_summaries": _arm_summaries(batches, failure_rates),
                                "paired_differences_vs_paper_base": _paired_differences(
                                    batches, failure_rates
                                ),
                                "completed_batches": batches,
                            },
                        )
                        report.section(
                            f"RESULT {formula} - REP {repetition} - {failure_rate:.0%}"
                        )
                        report.item("Success rate", aggregate.get("success_rate"))
                        report.item("Average carbon", aggregate.get("average_carbon"))
                        report.item("Average delay", aggregate.get("average_delay"))
                        report.item("P95 delay successful", aggregate.get("p95_delay_success"))
                        report.item("Average hops", aggregate.get("average_hops"))
                        report.item("Average reward (formula-specific; not cross-arm comparable)", aggregate.get("average_reward"))
                        report.item("Completed trials", completed_trials)
                        report.save()
                        del rows
                    del model

        if completed_trials != expected_trials:
            raise RuntimeError(
                f"Completed {completed_trials} trials; expected {expected_trials}."
            )
        final = {
            "protocol": {
                "snapshot": str(snapshot),
                "full_snapshot_graph": True,
                "base_seed": base_seed,
                "paper_numeric_seed_published": False,
                "training_seeds_per_formula": seeds,
                "formula_arms": [name for name, _ in ARMS],
                "transaction_pool": 10_000,
                "training_transactions": 7_000,
                "held_out_transactions": 3_000,
                "repetitions_per_formula": 5,
                "failure_rates": failure_rates,
                "expected_trials": expected_trials,
                "completed_trials": completed_trials,
            },
            "training_time_seconds_total": training_seconds_total,
            "formula_summaries": _arm_summaries(batches, failure_rates),
            "paired_differences_vs_paper_base": _paired_differences(batches, failure_rates),
            "completed_batches": batches,
            "artifacts": {
                "transaction_manifest": str(manifest),
                "evaluation_trials_csv": str(trial_path),
                "summary_json": str(summary_path),
                "report": str(core.REPORT_FILE),
            },
        }
        _write_json(summary_path, final)
        report.section("MAIN5 FINAL RESULT")
        report.item("Status", "COMPLETED")
        report.item("Reward formulas", len(ARMS))
        report.item("Repetitions per formula", protocol.PAPER_REPETITIONS)
        report.item("Full graph nodes/channels", f"{graph.number_of_nodes()} / {graph.number_of_edges()}")
        report.item("Total evaluation trials", completed_trials)
        report.item("PPO training sessions", len(ARMS) * protocol.PAPER_REPETITIONS)
        report.item("PPO training seconds total", round(training_seconds_total, 3))
        report.item("Summary", summary_path)
        report.item("Trial CSV", trial_path)
        report.save()
        print(f"\nMAIN5 completed. Report: {core.REPORT_FILE}")
        return final
    except KeyboardInterrupt as exc:
        core.write_runtime_error(exc, report, run_status="INTERRUPTED")
        raise
    except Exception as exc:
        core.write_runtime_error(exc, report, run_status="FAILED")
        print(f"MAIN5 failed: {type(exc).__name__}: {exc}")
        print(f"Detailed report: {core.REPORT_FILE}")
        raise
    finally:
        core.stop_heartbeat()


if __name__ == "__main__":
    try:
        main5()
    except Exception:
        sys.exit(1)
