"""Recovery through real graph views, forwarding, local search and settlement."""

import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from RL.environment import RoutingEnv
from RL.test_environment import make_candidate, make_graph, make_transaction
from Simulation.backtrack import PartialBacktracker
from Simulation.failure_model import FailureModel
from Simulation.network_dynamics import NetworkDynamics
from Pathfinding.top_k_paths import PathSearchTimeout
from Runtime.paper_protocol import (
    _aggregate_repetitions, _csv_row, _load_trial_rows, run_protocol, TRIAL_FIELDS,
)
from Runtime.recovery_reporting import RECOVERY_VERSION


def candidate(path, cost=1):
    return make_candidate(path, [(u, v, 0) for u, v in zip(path, path[1:])], cost)


class ScriptedFailureModel(FailureModel):
    """Inject specified failures; use the real hop evaluation everywhere else."""

    def __init__(self, failures):
        super().__init__(node_failure_probability=0, liquidity_failure_probability=0)
        self.failures = dict(failures)
        self.forwarded = []

    def evaluate_edge(self, graph, u, v, amount, key=None):
        edge = (u, v, key)
        self.forwarded.append(edge)
        if edge in self.failures:
            failed_node = self.failures.pop(edge)
            return {
                "success": False,
                "reason": "node_failure" if failed_node else "channel_failure",
                "failed_edge": None if failed_node else edge,
                "failed_node": failed_node,
            }
        return super().evaluate_edge(graph, u, v, amount, key)


class RecordingDynamics(NetworkDynamics):
    def __init__(self, graph):
        super().__init__(graph, channel_failure_rate=0, node_failure_rate=0, recovery_rate=0)
        self.settlements = []

    def settle_route(self, route_edges, amount):
        self.settlements.append((list(route_edges), amount))
        return super().settle_route(route_edges, amount)


class RecoveryIntegrationTests(unittest.TestCase):
    def run_payment(self, graph, candidates, failures):
        for _, _, data in graph.edges(data=True):
            data.update(fee_base=1, fee_rate=0, delay=1, simulator_failure_probability=0)
        model = ScriptedFailureModel(failures)
        dynamics = RecordingDynamics(graph)
        env = RoutingEnv(
            G=graph, transactions=[make_transaction()], failure_model=model,
            network_dynamics=dynamics,
        )
        self.addCleanup(env.close)
        env.reset(seed=42)
        with patch.object(env, "_generate_candidates", return_value=candidates):
            _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))
        return info, env, model, dynamics

    def test_cached_suffix_on_subgraph_reuses_prefix_and_settles_once(self):
        original = candidate(["A", "B", "C", "D", "E"])
        alternative = candidate(["A", "B", "F", "D", "E"], 2)
        info, env, model, dynamics = self.run_payment(
            make_graph(), [original, alternative], {("C", "D", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertIsInstance(env.backtracker, PartialBacktracker)
        self.assertNotIsInstance(env.backtracker.network.get_edge_data("B", "F"), dict)
        self.assertEqual(info["source_retry_count"], 0)
        self.assertEqual(info["partial_backtrack_retry_count"], 1)
        self.assertEqual(info["partial_backtrack_success"], 1)
        self.assertEqual(info["successful_route_source"], "partial_backtrack")
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)
        event = info["recovery_events"][0]
        self.assertEqual((event["branch_node"], event["backtracked_hops"]), ("B", 1))
        self.assertEqual(event["preserved_prefix"], ["A", "B"])
        self.assertEqual(event["recovery_origin"], "bucket")
        self.assertEqual(len(dynamics.settlements), 1)
        self.assertEqual(env.current_bucket.status, "completed")
        self.assertEqual(env.current_bucket.attempts, 2)

    def test_searches_from_nearest_branch_before_using_upstream_bucket_suffix(self):
        graph = make_graph()
        graph.add_edge("C", "F", key=0, capacity=10000, balance_uv=10000, available=True)
        info, _, model, _ = self.run_payment(
            graph,
            [candidate(["A", "B", "C", "D", "E"]),
             candidate(["A", "B", "F", "D", "E"], 2)],
            {("C", "D", 0): None},
        )
        self.assertTrue(info["success"])
        event = info["recovery_events"][0]
        self.assertEqual(event["branch_node"], "C")
        self.assertEqual(event["backtracked_hops"], 0)
        self.assertEqual(event["recovery_origin"], "local_search")
        self.assertEqual(info["preserved_prefix_hops"], 2)
        self.assertEqual(model.forwarded.count(("B", "C", 0)), 1)

    def test_local_search_when_bucket_has_no_alternative(self):
        info, _, model, _ = self.run_payment(
            make_graph(), [candidate(["A", "B", "C", "D", "E"])],
            {("C", "D", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["recovery_events"][0]["recovery_origin"], "local_search")
        self.assertEqual(info["source_retry_count"], 0)
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)

    def test_second_failed_continuation_recovers_again_without_source(self):
        graph = make_graph()
        for u, v in [("B", "G"), ("G", "D")]:
            graph.add_edge(u, v, key=0, capacity=10000, balance_uv=10000, available=True)
        graph.nodes["G"].update(available=True, is_online=True, carbon_intensity=100)
        info, _, model, dynamics = self.run_payment(
            graph,
            [candidate(["A", "B", "C", "D", "E"]),
             candidate(["A", "B", "F", "D", "E"], 2),
             candidate(["A", "B", "G", "D", "E"], 3)],
            {("C", "D", 0): None, ("F", "D", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["attempt_count"], 3)
        self.assertEqual(info["partial_backtrack_retry_count"], 2)
        self.assertEqual(info["source_retry_count"], 0)
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)
        self.assertEqual(model.forwarded.count(("C", "D", 0)), 1)
        self.assertEqual(len(dynamics.settlements), 1)
        self.assertEqual(info["path"], ["A", "B", "G", "D", "E"])

    def test_failed_branch_node_is_not_preserved(self):
        info, _, _, _ = self.run_payment(
            make_graph(), [candidate(["A", "B", "C", "D", "E"])],
            {("C", "D", 0): "C"},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["recovery_events"][0]["branch_node"], "B")
        self.assertNotIn("C", info["path"])

    def test_parallel_channel_at_same_branch_preserves_exact_prefix(self):
        graph = make_graph()
        graph.add_edge("B", "C", key=1, capacity=10000, balance_uv=10000, available=True)
        alternative = candidate(["A", "B", "C", "D", "E"], 2)
        alternative["edges"][1] = ("B", "C", 1)
        info, _, model, _ = self.run_payment(
            graph, [candidate(["A", "B", "C", "D", "E"]), alternative],
            {("B", "C", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["recovery_events"][0]["branch_node"], "B")
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)
        self.assertIn(("B", "C", 1), model.forwarded)

    def test_first_hop_failure_is_reported_as_source_retry(self):
        graph = make_graph()
        graph.add_edge("A", "F", key=0, capacity=10000, balance_uv=10000, available=True)
        info, _, _, _ = self.run_payment(
            graph,
            [candidate(["A", "B", "C", "D", "E"]),
             candidate(["A", "F", "D", "E"], 2)],
            {("A", "B", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["source_retry_count"], 1)
        self.assertEqual(info["partial_backtrack_retry_count"], 0)
        self.assertEqual(info["partial_backtrack_success"], 0)
        self.assertEqual(info["recovery_events"][0]["branch_index"], 0)

    def test_unrecoverable_failure_is_bounded_and_reported(self):
        graph = make_graph()
        graph.remove_edge("B", "F", 0)
        info, _, _, dynamics = self.run_payment(
            graph, [candidate(["A", "B", "C", "D", "E"])],
            {("C", "D", 0): None},
        )
        self.assertFalse(info["success"])
        self.assertEqual(info["attempt_count"], 1)
        self.assertEqual(info["recovery_events"][0]["type"], "recovery_unavailable")
        self.assertEqual(len(dynamics.settlements), 0)

    def test_source_recomputation_excludes_failed_first_hop_before_search(self):
        graph = make_graph()
        graph.add_edge("A", "F", key=0, capacity=10000, balance_uv=10000, available=True)
        info, _, model, _ = self.run_payment(
            graph, [candidate(["A", "B", "C", "D", "E"])],
            {("A", "B", 0): None},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["full_reroute_count"], 1)
        self.assertEqual(info["source_retry_count"], 1)
        self.assertEqual(info["partial_backtrack_retry_count"], 0)
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)
        self.assertIn("full_reroute", [event["type"] for event in info["recovery_events"]])

    def test_repeated_failures_stop_at_payment_attempt_budget(self):
        graph = make_graph()
        failures = {("B", "C", 0): None, ("B", "F", 0): None}
        for index in range(25):
            node = "N" + str(index)
            graph.add_node(node, available=True, is_online=True, carbon_intensity=100)
            for u, v in [("B", node), (node, "E")]:
                graph.add_edge(u, v, key=0, capacity=10000, balance_uv=10000, available=True)
            failures[("B", node, 0)] = None
        info, env, model, dynamics = self.run_payment(
            graph, [candidate(["A", "B", "C", "D", "E"])], failures,
        )
        self.assertFalse(info["success"])
        self.assertEqual(info["attempt_count"], env.top_k * 3 + 5)
        self.assertEqual(info["reason"], "maximum_payment_attempts_reached")
        self.assertEqual(info["source_retry_count"], 0)
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)
        self.assertEqual(len(dynamics.settlements), 0)
        self.assertEqual(info["recovery_events"][-1]["reason"], info["reason"])

    def test_forwarding_success_without_failure_has_empty_recovery_log(self):
        info, _, _, dynamics = self.run_payment(
            make_graph(), [candidate(["A", "B", "C", "D", "E"])], {},
        )
        self.assertTrue(info["success"])
        self.assertEqual(info["attempt_count"], 1)
        self.assertEqual(info["recovery_events"], [])
        self.assertEqual(info["partial_backtrack_retry_count"], 0)
        self.assertEqual(len(dynamics.settlements), 1)

    def test_local_search_timeout_is_reported_and_upstream_cache_still_works(self):
        with patch("RL.environment.top_k_paths", side_effect=PathSearchTimeout(
            "Test deadline", stage="candidate_search",
        )):
            info, _, model, _ = self.run_payment(
                make_graph(),
                [candidate(["A", "B", "C", "D", "E"]),
                 candidate(["A", "B", "F", "D", "E"], 2)],
                {("C", "D", 0): None},
            )
        self.assertTrue(info["success"])
        self.assertTrue(info["path_search_timed_out"])
        self.assertEqual(info["recovery_events"][0]["branch_node"], "B")
        self.assertEqual(model.forwarded.count(("A", "B", 0)), 1)

    def test_insufficient_liquidity_is_not_used_as_local_alternative(self):
        graph = make_graph()
        graph["B"]["F"][0]["balance_uv"] = 500
        graph["B"]["F"][0]["estimated_liquidity"] = 500
        info, _, model, dynamics = self.run_payment(
            graph, [candidate(["A", "B", "C", "D", "E"])],
            {("C", "D", 0): None},
        )
        self.assertFalse(info["success"])
        self.assertEqual(info["partial_backtrack_retry_count"], 0)
        self.assertNotIn(("B", "F", 0), model.forwarded)
        self.assertEqual(dynamics.settlements, [])

    def test_recovery_details_reach_trial_csv_resume_and_aggregate(self):
        info, _, _, _ = self.run_payment(
            make_graph(), [candidate(["A", "B", "C", "D", "E"])],
            {("C", "D", 0): None},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"RL_REPORT_FILE": str(Path(tmp, "report.txt"))}):
                from Runtime import experiment_core as core
            result = core.extract_episode_result(make_transaction(), 1, True, False, info, 1)
            result.update(repetition=1, training_seed=42, evaluation_seed=10000043, failure_rate=0.06)
            row = _csv_row(result)
            expected_events = json.loads(json.dumps(info["recovery_events"]))
            self.assertEqual(json.loads(row["recovery_events"]), expected_events)
            self.assertEqual(row["recovery_version"], RECOVERY_VERSION)
            path = Path(tmp, "trials.csv")
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            loaded = next(iter(_load_trial_rows(path).values()))
            self.assertEqual(loaded["recovery_events"], expected_events)
            aggregate = core.aggregate_evaluation_results([loaded])
            self.assertEqual(aggregate["total_partial_backtrack_retry_count"], 1)
            self.assertEqual(aggregate["total_source_retry_count"], 0)
            self.assertEqual(aggregate["total_preserved_prefix_hops"], 1)
            across_seeds = _aggregate_repetitions(
                [{"failure_rate": 0.06, "aggregate": aggregate}], [0.06],
            )
            self.assertEqual(across_seeds["0.06"]["total_partial_backtrack_retry_count"], 1)

    def test_legacy_resume_is_rejected_before_touching_historical_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "run_metadata.json").write_text("{}", encoding="utf-8")
            report = path / "report.txt"
            report.write_text("historical evaluation", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "older recovery implementation"):
                run_protocol(["--resume-run", tmp, "--report-file", str(report)])
            self.assertEqual(report.read_text(encoding="utf-8"), "historical evaluation")
            self.assertFalse((path / "run_status.json").exists())


if __name__ == "__main__":
    unittest.main()
