"""Run the four-arm paired study on the project's held-out split.

Example smoke run:
    python -m Evaluations.run_ablation --transactions 10 --repetitions 1 --failure-rates 0.03

Full configured study:
    python -m Evaluations.run_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from Evaluations.ablation import METHODS, generate_heldout_transactions, run_paired_ablation
from Network.graph_builder import LNGraphBuilder


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="20190501.gml.geo")
    parser.add_argument(
        "--model", default="models/end_to_end_hidden_failure_blind.zip"
    )
    parser.add_argument("--config", default="Configs/config.yaml")
    parser.add_argument("--output-dir", default="Evaluations/results/paired_ablation")
    parser.add_argument("--transactions", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--failure-rates", type=float, nargs="+")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = _arguments()
    config_path = Path(args.config)
    snapshot_path = Path(args.snapshot)
    model_path = Path(args.model)
    for path in (config_path, snapshot_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path.resolve())

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    evaluation_cfg = cfg.setdefault("evaluation", {})
    if args.transactions is not None:
        evaluation_cfg["transaction_count"] = args.transactions
    if args.repetitions is not None:
        evaluation_cfg["repetitions"] = args.repetitions

    # Use the same snapshot conversion and graph builder as main.py.
    from main import geo_to_json

    from stable_baselines3 import PPO

    print(f"Loading snapshot: {snapshot_path}")
    try:
        graph = LNGraphBuilder(
            default_capacity=cfg.get("channel", {}).get("default_capacity")
        ).from_data(geo_to_json(snapshot_path))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            "The snapshot could not be parsed as a supported GML file. "
            "Check its encoding and GML syntax."
        ) from exc
    transaction_count = int(evaluation_cfg.get("transaction_count", 3000))
    transactions = generate_heldout_transactions(
        graph,
        cfg,
        count=transaction_count,
        seed=args.seed,
    )
    try:
        model = PPO.load(
            str(model_path),
            device=cfg.get("rl", {}).get("device", "auto"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load PPO model {model_path.resolve()}. "
            "Train a fresh model with RL_FAST_VALIDATION=0; the updated "
            "liquidity-belief observation has a new input dimension."
        ) from exc

    repetitions = args.repetitions or evaluation_cfg.get("repetitions", 5)
    failure_rates = args.failure_rates or cfg.get("failure_rates")
    print(
        f"Evaluating {len(transactions)} held-out transactions; "
        f"methods={', '.join(METHODS)}; "
        f"repetitions={repetitions}; "
        f"failure_rates={failure_rates}"
    )
    result = run_paired_ablation(
        G=graph,
        transactions=transactions,
        cfg=cfg,
        rl_model=model,
        failure_rates=args.failure_rates,
        repetitions=args.repetitions,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Results written to: {Path(args.output_dir).resolve()}")
    for rate, method_summaries in result["summaries"].items():
        print(f"\nFailure rate {rate:.0%}")
        for method in METHODS:
            aggregate = method_summaries[method]["aggregate"]
            success = aggregate["success_rate"]
            p95 = aggregate["p95_delay_success"]
            ci = (
                f" [{success['ci_low']:.3%}, {success['ci_high']:.3%}]"
                if success["ci_low"] is not None else ""
            )
            delay = f"{p95['mean']:.4f}" if p95["mean"] is not None else "n/a"
            print(f"  {method:12s} success={success['mean']:.3%}{ci} p95_delay={delay}")


if __name__ == "__main__":
    main()
