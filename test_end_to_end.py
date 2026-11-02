"""
test_end_to_end.py

End-to-End validation of the complete RL + adaptive routing
pipeline using the real repository APIs.

Validated flow
--------------

    PPO
      |
      v
    eta
      |
      v
    Adaptive Routing
      |
      v
    Top-K (k=5)
      |
      v
    CandidateManager
      |
      v
    Bucket
      |
      v
    PaymentSimulator
      |
      v
    Controlled FailureModel
      |
      v
    PartialBacktracker
      |
      v
    Existing Bucket alternative
      |
      v
    PaymentSimulator retry
      |
      v
    SUCCESS


Important invariants
--------------------

1. PPO produces eta.
2. eta is within [0, 1].
3. k is fixed at 5.
4. Dijkstra uses the adaptive routing objective.
5. Top-K uses the adaptive routing objective.
6. Top-K is channel-aware.
7. Top-K physical edges are dictionaries.
8. Physical channel identity is preserved.
9. Directional liquidity is explicitly available.
10. Channel capacity is NOT used as directional liquidity.
11. The primary route is deterministic and cheaper.
12. The primary route contains B -> C -> key=0.
13. B -> C -> key=0 is deterministically failed.
14. The alternative route A -> B -> F -> G -> E
    is already present in Top-K and therefore Bucket.
15. PartialBacktracker reuses the existing Bucket candidate.
16. No full reroute is required.
17. The retry is executed by PaymentSimulator.
18. The retry succeeds.
19. Settlement occurs exactly once.
20. The complete environment pipeline is exercised.

This test does NOT modify the main implementation.
"""


import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np

from Pathfinding.dijkstra import Dijkstra
from Pathfinding.heuristics import (
    adaptive_edge_cost,
    lnd_cost,
)
from Pathfinding.top_k_paths import top_k_paths

from RL.environment import RoutingEnv
from RL.ppo_agent import build_ppo

from Simulation.failure_model import FailureModel


# ============================================================
# Deterministic E2E graph
# ============================================================

def build_e2e_graph():
    """
    Build a deterministic MultiDiGraph.

    Primary route:

        A -> B -> C -> D -> E

    Alternative route:

        A -> B -> F -> G -> E

    Both routes share A -> B.

    The primary route has much lower forwarding fees.

    RGB values are intentionally different so that the
    adaptive heuristic responds to eta.
    """

    G = nx.MultiDiGraph()

    node_data = {
        "A": {
            "latitude": 35.0000,
            "longitude": 51.0000,
            "country": "IR",
            "rgb_color": (0, 0, 0),
        },
        "B": {
            "latitude": 35.0010,
            "longitude": 51.0010,
            "country": "IR",
            "rgb_color": (255, 0, 0),
        },
        "C": {
            "latitude": 35.0020,
            "longitude": 51.0020,
            "country": "IR",
            "rgb_color": (255, 255, 0),
        },
        "D": {
            "latitude": 35.0030,
            "longitude": 51.0030,
            "country": "IR",
            "rgb_color": (0, 255, 0),
        },
        "E": {
            "latitude": 35.0040,
            "longitude": 51.0040,
            "country": "IR",
            "rgb_color": (0, 0, 255),
        },
        "F": {
            "latitude": 35.0010,
            "longitude": 51.0020,
            "country": "IR",
            "rgb_color": (255, 0, 255),
        },
        "G": {
            "latitude": 35.0020,
            "longitude": 51.0030,
            "country": "IR",
            "rgb_color": (255, 255, 255),
        },
    }

    for node, attrs in node_data.items():
        G.add_node(
            node,
            available=True,
            is_online=True,
            online=True,
            latitude=attrs["latitude"],
            longitude=attrs["longitude"],
            country=attrs["country"],
            rgb_color=attrs["rgb_color"],
        )

    # ========================================================
    # Primary route
    # ========================================================

    primary_edges = [
        ("A", "B", 0),
        ("B", "C", 0),
        ("C", "D", 0),
        ("D", "E", 0),
    ]

    for u, v, key in primary_edges:
        G.add_edge(
            u,
            v,
            key=key,
            capacity=10000.0,
            estimated_liquidity=10000.0,
            balance_uv=10000.0,
            available=True,
            fee_base=1.0,
            fee_rate=0.0,
            delay=1.0,
            failure_probability=0.0,
            scid=f"{u}-{v}-{key}",
        )

    # ========================================================
    # Alternative route
    # ========================================================

    alternative_edges = [
        ("B", "F", 0),
        ("F", "G", 0),
        ("G", "E", 0),
    ]

    for u, v, key in alternative_edges:
        G.add_edge(
            u,
            v,
            key=key,
            capacity=10000.0,
            estimated_liquidity=10000.0,
            balance_uv=10000.0,
            available=True,
            fee_base=100.0,
            fee_rate=0.0,
            delay=1.0,
            failure_probability=0.0,
            scid=f"{u}-{v}-{key}",
        )

    return G


# ============================================================
# Transaction
# ============================================================

def make_transaction():
    """
    Create deterministic payment transaction.
    """

    return SimpleNamespace(
        tx_id="E2E-TX-001",
        source="A",
        destination="E",
        amount=1000.0,
    )


# ============================================================
# Controlled Failure Model
# ============================================================

class ControlledFailureModel(FailureModel):
    """
    Deterministic failure model used only by this test.

    Only:

        B -> C -> key=0

    fails.

    Every other physical channel succeeds.
    """

    FAILED_EDGE = (
        "B",
        "C",
        0,
    )

    def __init__(self):
        super().__init__(
            node_failure_probability=0.0,
            liquidity_failure_probability=0.0,
            seed=123,
        )

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None,
    ):
        """
        Evaluate exact physical route edges.

        PaymentSimulator passes Top-K edges as dictionaries.
        """

        if route_edges is None:
            return {
                "success": False,
                "reason": "invalid_route_edges",
                "visited_edges": [],
            }

        if not isinstance(
            route_edges,
            (list, tuple),
        ):
            return {
                "success": False,
                "reason": "invalid_route_edges",
                "visited_edges": [],
            }

        normalized_edges = []

        for edge in route_edges:

            if not isinstance(edge, dict):
                return {
                    "success": False,
                    "reason": "invalid_edge_format",
                    "visited_edges": [],
                }

            source = edge.get("source")
            target = edge.get("target")
            channel_key = edge.get("channel_key")

            if (
                source is None
                or target is None
                or channel_key is None
            ):
                return {
                    "success": False,
                    "reason": "invalid_physical_channel",
                    "visited_edges": [],
                }

            normalized_edges.append(
                (
                    source,
                    target,
                    channel_key,
                )
            )

        # ----------------------------------------------------
        # Controlled physical-channel failure
        # ----------------------------------------------------

        if self.FAILED_EDGE in normalized_edges:

            failure_index = normalized_edges.index(
                self.FAILED_EDGE
            )

            return {
                "success": False,

                "reason":
                    "controlled_channel_failure",

                "failed_edge":
                    self.FAILED_EDGE,

                "failed_index":
                    failure_index,

                "failure_index":
                    failure_index,

                "visited_edges":
                    list(
                        route_edges[
                            :failure_index
                        ]
                    ),
            }

        # ----------------------------------------------------
        # Every other route succeeds.
        # ----------------------------------------------------

        return {
            "success": True,

            "reason":
                "success",

            "visited_edges":
                list(route_edges),
        }


# ============================================================
# Deterministic Network Dynamics
# ============================================================

class DeterministicNetworkDynamics:
    """
    Minimal NetworkDynamics-compatible implementation.

    No stochastic network changes are introduced.
    """

    def __init__(self):

        self.reset_calls = []

        self.update_calls = 0

        self.settlement_calls = 0

        self.payment_records = []

    def reset(
        self,
        reset_balances=False,
    ):

        self.reset_calls.append(
            bool(reset_balances)
        )

    def update(self):

        self.update_calls += 1

    def settle_route(
        self,
        route_edges,
        amount,
    ):

        self.settlement_calls += 1

        return True

    def record_payment(
        self,
        tx_id,
        result,
    ):

        self.payment_records.append(
            {
                "tx_id": tx_id,
                "result": dict(result),
            }
        )


# ============================================================
# PPO Configuration
# ============================================================

def make_ppo_config():
    """
    Minimal PPO configuration.

    PPO is not trained.

    The untrained model produces one eta action.
    """

    return {
        "rl": {
            "hidden_layers": [
                16,
                16,
            ],
            "learning_rate": 3e-4,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "verbose": 0,
            "device": "cpu",
            "eta_min": 0.0,
            "eta_max": 1.0,
        },

        "graph": {
            "max_hops": 6,
            "lambda_h": 1.0,
            "neighborhood_k": 6,
            "neighborhood_m": 3,
        },

        "simulation": {
            "seed": 123,
            "node_failure_probability": 0.0,
            "liquidity_failure_probability": 0.0,
            "channel_failure_rate": 0.0,
            "node_failure_rate": 0.0,
            "recovery_rate": 0.0,
            "reset_balances_on_episode_reset": False,
        },
    }


# ============================================================
# Candidate helpers
# ============================================================

def candidate_channel_identity(candidate):
    """
    Return exact physical-channel identity.

    Identity includes:

        source
        target
        channel_key
        scid
    """

    edges = candidate.get("edges")

    if not isinstance(
        edges,
        (list, tuple),
    ):
        raise AssertionError(
            "Candidate edges must be a list or tuple."
        )

    identity = []

    for edge in edges:

        if not isinstance(edge, dict):
            raise AssertionError(
                "Top-K physical edges must be dictionaries."
            )

        identity.append(
            (
                edge.get("source"),
                edge.get("target"),
                edge.get("channel_key"),
                edge.get("scid"),
            )
        )

    return tuple(identity)


def candidate_has_exact_route(
    candidate,
    expected_path,
    expected_edges,
):
    """
    Validate both node path and exact physical channels.
    """

    if list(
        candidate.get("path", [])
    ) != list(expected_path):
        return False

    edges = candidate.get("edges")

    if not isinstance(
        edges,
        (list, tuple),
    ):
        return False

    if len(edges) != len(expected_edges):
        return False

    for edge, expected in zip(
        edges,
        expected_edges,
    ):

        if not isinstance(
            edge,
            dict,
        ):
            return False

        actual = (
            edge.get("source"),
            edge.get("target"),
            edge.get("channel_key"),
        )

        if actual != expected:
            return False

    return True


# ============================================================
# Test
# ============================================================

class EndToEndRoutingTest(
    unittest.TestCase
):

    def test_complete_ppo_to_backtracking_pipeline(
        self
    ):
        """
        Validate:

            PPO
             |
             v
            eta
             |
             v
        Adaptive Routing
             |
             v
           Top-K
             |
             v
        CandidateManager
             |
             v
           Bucket
             |
             v
        PaymentSimulator
             |
             v
        Controlled Failure
             |
             v
        PartialBacktracker
             |
             v
        Existing Bucket alternative
             |
             v
        PaymentSimulator retry
             |
             v
           SUCCESS
        """

        # ====================================================
        # Fixtures
        # ====================================================

        G = build_e2e_graph()

        tx = make_transaction()

        cfg = make_ppo_config()

        failure_model = ControlledFailureModel()

        network_dynamics = (
            DeterministicNetworkDynamics()
        )

        expected_primary_path = [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

        expected_primary_edges = [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0),
        ]

        expected_alternative_path = [
            "A",
            "B",
            "F",
            "G",
            "E",
        ]

        expected_alternative_edges = [
            ("A", "B", 0),
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ]

        # ====================================================
        # 1. Graph validation
        # ====================================================

        self.assertIsInstance(
            G,
            nx.MultiDiGraph,
        )

        self.assertEqual(
            G.number_of_nodes(),
            7,
        )

        self.assertEqual(
            G.number_of_edges(),
            7,
        )

        # ====================================================
        # 2. Directional liquidity validation
        # ====================================================

        for (
            u,
            v,
            key,
            data,
        ) in G.edges(
            keys=True,
            data=True,
        ):

            self.assertIn(
                "estimated_liquidity",
                data,
            )

            self.assertIn(
                "balance_uv",
                data,
            )

            self.assertEqual(
                float(
                    data["estimated_liquidity"]
                ),
                10000.0,
            )

            self.assertEqual(
                float(
                    data["balance_uv"]
                ),
                10000.0,
            )

            self.assertIn(
                "capacity",
                data,
            )

        # ====================================================
        # 3. Dijkstra baseline
        # ====================================================

        dijkstra = Dijkstra(G)

        dijkstra_result = (
            dijkstra.shortest_path(
                source=tx.source,
                target=tx.destination,
                amount=tx.amount,
                heuristic_fn=lnd_cost,
                eta=0.5,
                max_hops=6,
                lambda_h=1.0,
            )
        )

        self.assertIsInstance(
            dijkstra_result,
            dict,
        )

        self.assertTrue(
            dijkstra_result.get("success"),
            msg=(
                "Dijkstra failed: "
                f"{dijkstra_result}"
            ),
        )

        self.assertEqual(
            list(
                dijkstra_result["path"]
            ),
            expected_primary_path,
        )

        # ====================================================
        # 4. Adaptive edge cost
        # ====================================================

        test_edge = G["A"]["B"][0]

        adaptive_eta_0 = adaptive_edge_cost(
            G=G,
            u="A",
            v="B",
            data=test_edge,
            amount=tx.amount,
            eta=0.0,
            heuristic_fn=lnd_cost,
            lambda_h=1.0,
        )

        adaptive_eta_1 = adaptive_edge_cost(
            G=G,
            u="A",
            v="B",
            data=test_edge,
            amount=tx.amount,
            eta=1.0,
            heuristic_fn=lnd_cost,
            lambda_h=1.0,
        )

        for result in (
            adaptive_eta_0,
            adaptive_eta_1,
        ):

            self.assertIsInstance(
                result,
                dict,
            )

            self.assertIn(
                "native_cost",
                result,
            )

            self.assertIn(
                "raw_heuristic",
                result,
            )

            self.assertIn(
                "adaptive_penalty",
                result,
            )

            self.assertIn(
                "cost",
                result,
            )

        self.assertAlmostEqual(
            adaptive_eta_0["native_cost"],
            adaptive_eta_1["native_cost"],
            places=9,
        )

        self.assertNotAlmostEqual(
            adaptive_eta_0["raw_heuristic"],
            adaptive_eta_1["raw_heuristic"],
            places=9,
            msg=(
                "Changing eta did not change the "
                "raw adaptive heuristic."
            ),
        )

        self.assertNotAlmostEqual(
            adaptive_eta_0["adaptive_penalty"],
            adaptive_eta_1["adaptive_penalty"],
            places=9,
            msg=(
                "Changing eta did not change the "
                "adaptive penalty."
            ),
        )

        self.assertNotAlmostEqual(
            adaptive_eta_0["cost"],
            adaptive_eta_1["cost"],
            places=9,
            msg=(
                "Changing eta did not change the "
                "adaptive routing objective."
            ),
        )

        # ====================================================
        # 5. Top-K at eta=0.5
        # ====================================================

        baseline_candidates = top_k_paths(
            G=G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=lnd_cost,
            eta=0.5,
            k=5,
            max_hops=6,
            lambda_h=1.0,
        )

        self.assertIsInstance(
            baseline_candidates,
            list,
        )

        self.assertGreaterEqual(
            len(baseline_candidates),
            2,
            msg=(
                "Expected at least two deterministic "
                "candidate routes."
            ),
        )

        self.assertLessEqual(
            len(baseline_candidates),
            5,
        )

        # ====================================================
        # 6. Validate Top-K candidate structure
        # ====================================================

        identities = []

        for candidate in baseline_candidates:

            self.assertIsInstance(
                candidate,
                dict,
            )

            for field in (
                "path",
                "edges",
                "cost",
                "eta",
                "lambda_h",
                "raw_heuristic",
                "adaptive_penalty",
            ):

                self.assertIn(
                    field,
                    candidate,
                )

            self.assertIsInstance(
                candidate["edges"],
                list,
            )

            identities.append(
                candidate_channel_identity(
                    candidate
                )
            )

            for edge in candidate["edges"]:

                self.assertIsInstance(
                    edge,
                    dict,
                )

                for field in (
                    "source",
                    "target",
                    "channel_key",
                    "scid",
                    "data",
                ):

                    self.assertIn(
                        field,
                        edge,
                    )

        self.assertEqual(
            len(identities),
            len(set(identities)),
            msg=(
                "Top-K returned duplicate physical "
                "channel routes."
            ),
        )

        # ====================================================
        # 7. ETA propagation
        # ====================================================

        for candidate in baseline_candidates:

            self.assertAlmostEqual(
                float(
                    candidate["eta"]
                ),
                0.5,
                places=9,
            )

            self.assertAlmostEqual(
                float(
                    candidate["lambda_h"]
                ),
                1.0,
                places=9,
            )

        # ====================================================
        # 8. Primary route must be first
        # ====================================================

        self.assertEqual(
            list(
                baseline_candidates[0]["path"]
            ),
            expected_primary_path,
        )

        self.assertTrue(
            candidate_has_exact_route(
                baseline_candidates[0],
                expected_primary_path,
                expected_primary_edges,
            ),
            msg=(
                "Top-K primary candidate does not preserve "
                "the expected physical channel sequence."
            ),
        )

        # ====================================================
        # 9. First physical edge
        # ====================================================

        first_edge = (
            baseline_candidates[0]["edges"][0]
        )

        self.assertEqual(
            first_edge["source"],
            "A",
        )

        self.assertEqual(
            first_edge["target"],
            "B",
        )

        self.assertEqual(
            first_edge["channel_key"],
            0,
        )

        self.assertEqual(
            first_edge["scid"],
            "A-B-0",
        )

        # ====================================================
        # 10. Alternative must exist in Top-K
        # ====================================================

        alternative_candidates = [
            candidate
            for candidate in baseline_candidates
            if list(
                candidate.get("path", [])
            ) == expected_alternative_path
        ]

        self.assertGreaterEqual(
            len(alternative_candidates),
            1,
            msg=(
                "The expected Bucket fallback route was "
                "not generated by Top-K."
            ),
        )

        alternative_candidate = (
            alternative_candidates[0]
        )

        self.assertTrue(
            candidate_has_exact_route(
                alternative_candidate,
                expected_alternative_path,
                expected_alternative_edges,
            ),
            msg=(
                "The expected alternative route exists "
                "by node path but does not preserve the "
                "required physical channel identity."
            ),
        )

        # ====================================================
        # 11. ETA=0 and ETA=1 propagation
        # ====================================================

        eta0_candidates = top_k_paths(
            G=G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=lnd_cost,
            eta=0.0,
            k=5,
            max_hops=6,
            lambda_h=1.0,
        )

        eta1_candidates = top_k_paths(
            G=G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=lnd_cost,
            eta=1.0,
            k=5,
            max_hops=6,
            lambda_h=1.0,
        )

        self.assertGreater(
            len(eta0_candidates),
            0,
        )

        self.assertGreater(
            len(eta1_candidates),
            0,
        )

        for candidate in eta0_candidates:

            self.assertAlmostEqual(
                float(
                    candidate["eta"]
                ),
                0.0,
                places=9,
            )

        for candidate in eta1_candidates:

            self.assertAlmostEqual(
                float(
                    candidate["eta"]
                ),
                1.0,
                places=9,
            )

        # ====================================================
        # 12. Create real RoutingEnv
        # ====================================================

        env = RoutingEnv(
            G=G,
            transactions=[tx],
            heuristic_fn=lnd_cost,
            config=cfg,
            failure_model=failure_model,
            network_dynamics=network_dynamics,
        )

        try:

            # =================================================
            # 13. Reset
            # =================================================

            observation, reset_info = env.reset(
                seed=123
            )

            self.assertIsInstance(
                observation,
                np.ndarray,
            )

            self.assertEqual(
                observation.shape,
                env.observation_space.shape,
            )

            self.assertEqual(
                env.top_k,
                5,
            )

            self.assertIsInstance(
                reset_info,
                dict,
            )

            # =================================================
            # 14. PPO
            # =================================================

            ppo_model = build_ppo(
                env=env,
                cfg=cfg,
                seed=123,
            )

            ppo_action, _ = ppo_model.predict(
                observation,
                deterministic=True,
            )

            ppo_action = np.asarray(
                ppo_action,
                dtype=np.float32,
            )

            self.assertEqual(
                ppo_action.shape,
                (1,),
            )

            ppo_eta = float(
                ppo_action[0]
            )

            self.assertGreaterEqual(
                ppo_eta,
                env.eta_min,
            )

            self.assertLessEqual(
                ppo_eta,
                env.eta_max,
            )

            # =================================================
            # 15. Environment step
            # =================================================

            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                ppo_action
            )

            self.assertIsInstance(
                info,
                dict,
            )

            # =================================================
            # 16. PPO -> eta
            # =================================================

            self.assertIn(
                "eta",
                info,
            )

            self.assertAlmostEqual(
                float(
                    info["eta"]
                ),
                ppo_eta,
                places=6,
            )

            # =================================================
            # 17. Top-K
            # =================================================

            self.assertIn(
                "top_k",
                info,
            )

            self.assertEqual(
                info["top_k"],
                5,
            )

            # =================================================
            # 18. Candidate generation
            # =================================================

            self.assertIn(
                "candidate_path_count",
                info,
            )

            self.assertIn(
                "usable_candidate_count",
                info,
            )

            self.assertGreaterEqual(
                info["candidate_path_count"],
                2,
                msg=(
                    "Environment did not generate the "
                    "required candidate routes."
                ),
            )

            self.assertGreaterEqual(
                info["usable_candidate_count"],
                2,
                msg=(
                    "CandidateManager did not preserve "
                    "at least two usable candidates."
                ),
            )

            # =================================================
            # 19. Bucket
            # =================================================

            self.assertIsNotNone(
                env.current_bucket
            )

            self.assertIn(
                "bucket_size",
                info,
            )

            self.assertGreaterEqual(
                info["bucket_size"],
                2,
                msg=(
                    "Bucket does not contain the required "
                    "primary and alternative candidates."
                ),
            )

            # =================================================
            # 20. Payment attempts
            # =================================================

            self.assertGreaterEqual(
                len(
                    network_dynamics.payment_records
                ),
                2,
                msg=(
                    "Expected at least two payment attempts "
                    "(primary failure + successful retry). "
                    f"Payment records: "
                    f"{network_dynamics.payment_records}"
                ),
            )

            # =================================================
            # 21. First payment
            # =================================================

            first_record = (
                network_dynamics.payment_records[0]
            )

            first_result = (
                first_record["result"]
            )

            self.assertFalse(
                first_result["success"],
                msg=(
                    "The first payment attempt did not fail. "
                    f"Result: {first_result}"
                ),
            )

            self.assertEqual(
                first_result["failed_edge"],
                (
                    "B",
                    "C",
                    0,
                ),
                msg=(
                    "The first payment did not fail on "
                    "B-C-0. "
                    f"Result: {first_result}"
                ),
            )

            self.assertEqual(
                first_result["reason"],
                "controlled_channel_failure",
            )

            # =================================================
            # 22. Partial Backtracking invocation
            # =================================================

            self.assertIn(
                "attempt_count",
                info,
            )

            self.assertGreaterEqual(
                info["attempt_count"],
                2,
                msg=(
                    "The environment did not perform the "
                    "required retry attempt."
                ),
            )

            self.assertIn(
                "partial_backtrack_count",
                info,
            )

            self.assertGreaterEqual(
                info["partial_backtrack_count"],
                1,
                msg=(
                    "The controlled B-C-0 failure occurred, "
                    "but PartialBacktracker was not invoked. "
                    f"Info: {info}"
                ),
            )

            # =================================================
            # 23. Partial Backtracking success
            # =================================================

            self.assertIn(
                "partial_backtrack_success",
                info,
            )

            self.assertGreaterEqual(
                info["partial_backtrack_success"],
                1,
                msg=(
                    "PartialBacktracker was invoked but did "
                    "not produce a successful retry. "
                    f"Info: {info}"
                ),
            )

            # =================================================
            # 24. No full reroute
            # =================================================

            self.assertIn(
                "full_reroute_count",
                info,
            )

            self.assertEqual(
                info["full_reroute_count"],
                0,
                msg=(
                    "Full rerouting occurred although a "
                    "valid Bucket alternative existed. "
                    f"Info: {info}"
                ),
            )

            self.assertIn(
                "full_reroute",
                info,
            )

            self.assertFalse(
                info["full_reroute"],
            )

            # =================================================
            # 25. Final payment
            # =================================================

            self.assertIn(
                "payment_success",
                info,
            )

            self.assertTrue(
                info["payment_success"],
                msg=(
                    "The final payment was not successful. "
                    f"Info: {info}"
                ),
            )

            self.assertTrue(
                info["success"],
                msg=(
                    "Environment did not report successful "
                    "completion. "
                    f"Info: {info}"
                ),
            )

            self.assertEqual(
                info["reason"],
                "partial_backtrack_success",
                msg=(
                    "Unexpected final environment reason: "
                    f"{info.get('reason')}"
                ),
            )

            # =================================================
            # 26. Final path
            # =================================================

            self.assertIn(
                "path",
                info,
            )

            final_path = list(
                info["path"]
            )

            self.assertEqual(
                final_path,
                expected_alternative_path,
                msg=(
                    "The successful retry did not use the "
                    "expected existing Bucket alternative. "
                    f"Actual path: {final_path}"
                ),
            )

            self.assertNotEqual(
                final_path,
                expected_primary_path,
            )

            self.assertNotIn(
                "C",
                final_path,
            )

            # =================================================
            # 27. Final payment record
            # =================================================

            final_record = (
                network_dynamics.payment_records[-1]
            )

            final_result = (
                final_record["result"]
            )

            self.assertTrue(
                final_result["success"],
                msg=(
                    "The final PaymentSimulator retry "
                    "did not succeed. "
                    f"Result: {final_result}"
                ),
            )

            self.assertEqual(
                final_result["path"],
                expected_alternative_path,
            )

            self.assertEqual(
                final_result["reason"],
                "success",
            )

            # =================================================
            # 28. Exactly one settlement
            # =================================================

            self.assertEqual(
                network_dynamics.settlement_calls,
                1,
                msg=(
                    "Settlement must occur exactly once "
                    "for the successful payment."
                ),
            )

            # =================================================
            # 29. Output types
            # =================================================

            self.assertIsInstance(
                reward,
                float,
            )

            self.assertIsInstance(
                next_observation,
                np.ndarray,
            )

            self.assertEqual(
                next_observation.shape,
                env.observation_space.shape,
            )

            self.assertFalse(
                truncated,
            )

            # =================================================
            # 30. Bucket preserved
            # =================================================

            self.assertIsNotNone(
                env.current_bucket
            )

            # =================================================
            # 31. PASS summary
            # =================================================

            print()
            print("=" * 78)
            print(
                "END-TO-END PPO -> ADAPTIVE ROUTING -> "
                "BUCKET -> PARTIAL BACKTRACKING : PASS"
            )
            print("=" * 78)

            print(
                "Dijkstra path             :",
                expected_primary_path,
            )

            print(
                "PPO eta                   :",
                f"{ppo_eta:.6f}",
            )

            print(
                "Adaptive edge cost eta=0 :",
                f"{adaptive_eta_0['cost']:.6f}",
            )

            print(
                "Adaptive edge cost eta=1 :",
                f"{adaptive_eta_1['cost']:.6f}",
            )

            print(
                "Top-K                     :",
                info["top_k"],
            )

            print(
                "Candidate count           :",
                info["candidate_path_count"],
            )

            print(
                "Usable candidates         :",
                info["usable_candidate_count"],
            )

            print(
                "Bucket size               :",
                info["bucket_size"],
            )

            print(
                "Payment attempts          :",
                info["attempt_count"],
            )

            print(
                "Controlled failed edge   :",
                ("B", "C", 0),
            )

            print(
                "Partial backtracks        :",
                info["partial_backtrack_count"],
            )

            print(
                "Backtrack successes       :",
                info["partial_backtrack_success"],
            )

            print(
                "Full reroutes             :",
                info["full_reroute_count"],
            )

            print(
                "Final path                :",
                final_path,
            )

            print(
                "Settlement calls          :",
                network_dynamics.settlement_calls,
            )

            print(
                "Payment success           :",
                info["payment_success"],
            )

            print(
                "Final reason              :",
                info["reason"],
            )

            print("=" * 78)

        finally:

            env.close()


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )