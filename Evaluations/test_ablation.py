"""Fast tests for the paired RL/Bucket ablation evaluator."""

import unittest

import networkx as nx
import numpy as np

from Evaluations.ablation import METHODS, run_paired_ablation, summarize_arm
from Simulation.transaction_generator import Transaction


def create_test_graph():
    graph = nx.MultiDiGraph()
    for node in ("A", "B"):
        graph.add_node(
            node,
            country="US",
            continent_code="NA",
            latitude=40.0,
            longitude=-74.0,
            carbon_intensity=100.0,
            online=True,
        )
    for source, target in (("A", "B"), ("B", "A")):
        graph.add_edge(
            source,
            target,
            key="channel-1",
            fee_base=1.0,
            fee_rate=0.0,
            delay=10.0,
            capacity=1_000_000,
            estimated_liquidity=1_000_000,
            available=True,
            failure_probability=0.0,
        )
    return graph


def create_test_transaction(tx_id):
    return Transaction(
        tx_id=tx_id,
        source="A",
        destination="B",
        amount=1000.0,
        timestamp=0,
        split="test",
    )


def create_test_config():
    return {
        "seed": 42,
        "graph": {
            "max_hops": 12,
            "top_k_paths": 5,
            "neighborhood_k": 15,
            "neighborhood_m": 5,
        },
        "simulation": {
            "channel_failure_rate": 0.0,
            "node_failure_probability": 0.0,
            "liquidity_failure_probability": 0.0,
            "node_failure_rate": 0.0,
            "recovery_rate": 1.0,
        },
        "bucket": {"max_candidates": 5, "enable_backtracking": True},
        "rl": {"eta_min": -1.0, "eta_max": 1.0},
    }


class FixedModel:
    def predict(self, observation, deterministic=True):
        return np.asarray([0.3], dtype=np.float32), None


class AblationEvaluationTests(unittest.TestCase):
    def test_delay_percentiles_are_reported_for_successes_and_routes(self):
        summary = summarize_arm([
            {"success": True, "path_length": 1, "delay": 10.0},
            {"success": True, "path_length": 1, "delay": 20.0},
            {"success": False, "path_length": 1, "delay": 30.0},
            {"success": False, "path_length": 0, "delay": 0.0},
        ])

        self.assertEqual(summary["p90_delay_success"], 19.0)
        self.assertEqual(summary["p95_delay_success"], 19.5)
        self.assertEqual(summary["p90_delay_routed"], 28.0)
        self.assertEqual(summary["p95_delay_routed"], 29.0)

    def test_paired_study_runs_all_four_arms_on_each_transaction(self):
        graph = create_test_graph()
        transactions = [
            create_test_transaction(tx_id=101),
            create_test_transaction(tx_id=102),
        ]
        cfg = create_test_config()
        cfg["evaluation"] = {"transaction_count": 2, "repetitions": 1}
        cfg["failure_rates"] = [0.0]

        result = run_paired_ablation(
            graph,
            transactions,
            cfg,
            FixedModel(),
        )

        self.assertEqual(result["metadata"]["methods"], list(METHODS))
        self.assertTrue(result["metadata"]["paired_scenario_seeds"])
        self.assertEqual(len(result["rows"]), 8)
        for method in METHODS:
            rows = [row for row in result["rows"] if row["method"] == method]
            self.assertEqual([row["transaction_id"] for row in rows], [101, 102])
            self.assertTrue(all(row["success"] for row in rows), method)
            self.assertEqual(
                len(result["summaries"][0.0][method]["repetitions"]), 1
            )
            self.assertEqual(
                result["summaries"][0.0][method]["aggregate"]["success_rate"]["n"],
                1,
            )
        self.assertEqual(
            result["summaries"][0.0]["paired_differences_vs_baseline"]["rl_bucket"]
            ["success_rate"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
