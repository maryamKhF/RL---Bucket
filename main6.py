"""Full-snapshot runner for the paper's original RL reward only.

Run from the project directory with ``python main6.py``. The runner uses the
article-sized 10,000-transaction pool (7,000 train / 3,000 held out), trains
five independent PPO models with the paper baseline reward, and evaluates
each model on the same held-out transactions. The six exploratory reward
variants in ``main5.py`` are not part of this run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import main4 as paper_protocol


PROJECT_ROOT = Path(__file__).resolve().parent


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
        help="Root directory for this baseline run's reports, models, and CSV.",
    )
    parser.add_argument(
        "--failure-rates",
        type=float,
        nargs="+",
        help="Optional override for configured edge failure rates.",
    )
    return parser.parse_args(argv)


def main6(argv=None):
    """Execute only the original article reward, five repetitions total."""
    args = _parse_args(argv)
    delegated_args = [
        "--snapshot",
        str(args.snapshot),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.failure_rates is not None:
        delegated_args.append("--failure-rates")
        delegated_args.extend(str(rate) for rate in args.failure_rates)

    # main4 explicitly overwrites the configured reward formula with
    # ``paper_base`` and performs exactly five full paper-protocol runs.
    return paper_protocol.main4(delegated_args)


if __name__ == "__main__":
    try:
        main6()
    except Exception:
        sys.exit(1)
