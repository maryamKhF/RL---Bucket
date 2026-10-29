"""Tests for simulator-only liquidity and agent feedback intervals."""

import unittest

import networkx as nx

from RL.liquidity_belief import LiquidityBelief
from Network.graph_builder import LNGraphBuilder
from Simulation.failure_model import FailureModel
from Simulation.network_dynamics import NetworkDynamics
from Simulation.payment_simulator import PaymentSimulator
from Simulation.failure_model import assign_failure_probabilities


def make_graph():
    graph = nx.MultiDiGraph()
    graph.add_edge(
        "A", "B", key="chan", channel_id="chan", capacity=1000.0,
        balance_uv=None, available=True, fee_base=1.0, fee_rate=0.0, delay=10.0,
    )
    graph.add_edge(
        "B", "A", key="chan-rev", channel_id="chan-rev", capacity=1000.0,
        balance_uv=None, available=True, fee_base=1.0, fee_rate=0.0, delay=10.0,
    )
    return graph


class HiddenLiquidityTests(unittest.TestCase):
    def test_configured_default_capacity_does_not_become_public_balance(self):
        graph = LNGraphBuilder(default_capacity=1200.0).from_data({
            "nodes": [{"id": "A"}, {"id": "B"}],
            "channels": [{"source": "A", "target": "B", "scid": "chan"}],
        })
        forward = graph["A"]["B"]["chan"]
        self.assertEqual(forward["capacity"], 1200.0)
        self.assertIsNone(forward["balance_uv"])
        self.assertNotIn("estimated_liquidity", forward)

    def test_hidden_balances_are_seeded_private_and_capacity_conserving(self):
        graph = make_graph()
        dynamics = NetworkDynamics(
            graph,
            channel_failure_rate=0.0,
            node_failure_rate=0.0,
            recovery_rate=0.0,
            seed=31,
        )
        forward = dynamics.hidden_liquidity[("A", "B", "chan")]
        reverse = dynamics.hidden_liquidity[("B", "A", "chan-rev")]

        self.assertAlmostEqual(forward + reverse, 1000.0)
        self.assertIsNone(graph["A"]["B"]["chan"]["balance_uv"])
        self.assertNotIn("simulator_liquidity", graph["A"]["B"]["chan"])
        self.assertEqual(
            dynamics.hidden_liquidity,
            NetworkDynamics(
                make_graph(), channel_failure_rate=0.0,
                node_failure_rate=0.0, recovery_rate=0.0, seed=31,
            ).hidden_liquidity,
        )

    def test_default_capacity_can_bound_a_snapshot_without_capacity_fields(self):
        graph = make_graph()
        for _, _, _, data in graph.edges(keys=True, data=True):
            data.pop("capacity")
        dynamics = NetworkDynamics(
            graph,
            channel_failure_rate=0.0,
            node_failure_rate=0.0,
            recovery_rate=0.0,
            seed=31,
            default_capacity=1200.0,
        )
        self.assertAlmostEqual(
            dynamics.hidden_liquidity[("A", "B", "chan")]
            + dynamics.hidden_liquidity[("B", "A", "chan-rev")],
            1200.0,
        )

    def test_unknown_route_is_routed_but_simulator_checks_true_balance(self):
        graph = make_graph()
        dynamics = NetworkDynamics(
            graph,
            channel_failure_rate=0.0,
            node_failure_rate=0.0,
            recovery_rate=0.0,
            seed=31,
        )
        failure_model = FailureModel(
            node_failure_probability=0.0,
            liquidity_failure_probability=0.0,
            seed=31,
        )
        failure_model.hidden_liquidity = dynamics.hidden_liquidity
        true_balance = dynamics.hidden_liquidity[("A", "B", "chan")]

        self.assertFalse(dynamics.can_forward("A", "B", "chan", true_balance + 1.0))
        self.assertTrue(
            failure_model.check_liquidity_failure(
                graph, "A", "B", true_balance + 1.0, "chan"
            )
        )
        self.assertFalse(
            failure_model.check_liquidity_failure(
                graph, "A", "B", true_balance, "chan"
            )
        )

    def test_belief_bounds_update_from_observed_results_only(self):
        graph = make_graph()
        belief = LiquidityBelief(graph)
        edge = ("A", "B", "chan")

        self.assertEqual(belief.interval(edge), (0.0, 1000.0))
        belief.observe_failure(edge, 600.0)
        lower, upper = belief.interval(edge)
        self.assertEqual(lower, 0.0)
        self.assertLess(upper, 600.0)

        belief.observe_success([edge], 100.0)
        self.assertLessEqual(belief.interval(edge)[1], upper - 100.0)
        self.assertGreaterEqual(belief.interval(("B", "A", "chan-rev"))[0], 100.0)

    def test_payment_simulator_uses_private_ledger_and_settles_it(self):
        graph = make_graph()
        assign_failure_probabilities(graph, 0.0, 31)
        dynamics = NetworkDynamics(
            graph,
            channel_failure_rate=0.0,
            node_failure_rate=0.0,
            recovery_rate=0.0,
            seed=31,
        )
        failure_model = FailureModel(
            node_failure_probability=0.0,
            liquidity_failure_probability=0.0,
            seed=31,
        )
        simulator = PaymentSimulator(graph, failure_model, dynamics)
        edge = ("A", "B", "chan")
        hidden_before = dynamics.hidden_liquidity[edge]

        too_large = simulator.simulate_payment(
            ["A", "B"], [edge], hidden_before + 1.0, tx_id="fail"
        )
        self.assertFalse(too_large.success)
        self.assertEqual(too_large.reason, "liquidity_failure")
        self.assertIsNone(graph.edges[edge]["balance_uv"])

        amount = min(10.0, hidden_before)
        success = simulator.simulate_payment(
            ["A", "B"], [edge], amount, tx_id="success"
        )
        self.assertTrue(success.success)
        self.assertAlmostEqual(
            dynamics.hidden_liquidity[edge], hidden_before - amount
        )
        self.assertIsNone(graph.edges[edge]["balance_uv"])


if __name__ == "__main__":
    unittest.main()
