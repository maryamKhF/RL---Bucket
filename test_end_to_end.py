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
    adaptive heuristic actually responds to eta.

    The fee difference is deliberately large enough that
    adaptive penalties cannot change the intended primary
    route ordering.
    """

    G = nx.MultiDiGraph()

    # --------------------------------------------------------
    # Nodes
    #
    # RGB values are deliberately different.
    #
    # This is important because adaptive_heuristic() uses
    # node RGB luminance:
    #
    #     C = 0.299R + 0.587G + 0.114B
    #
    # Therefore eta=0 and eta=1 must produce different
    # adaptive signals on A -> B.
    # --------------------------------------------------------

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
    #
    # A -> B -> C -> D -> E
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

            # Explicit directional liquidity.
            estimated_liquidity=10000.0,
            balance_uv=10000.0,

            available=True,

            # Very low native routing cost.
            fee_base=1.0,
            fee_rate=0.0,

            delay=1.0,

            # Explicit reliability.
            failure_probability=0.0,

            scid=f"{u}-{v}-{key}",
        )

    # ========================================================
    # Alternative route
    #
    # A -> B -> F -> G -> E
    #
    # It shares A -> B with the primary route.
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

            # Explicit directional liquidity.
            estimated_liquidity=10000.0,
            balance_uv=10000.0,

            available=True,

            # Intentionally much higher fee.
            #
            # This keeps the primary route first even
            # when the adaptive penalty changes with eta.
            fee_base=100.0,
            fee_rate=0.0,

            delay=1.0,

            # Explicit reliability.
            failure_probability=0.0,

            scid=f"{u}-{v}-{key}",
        )

    return G


# ============================================================
# Transaction
# ============================================================

def make_transaction():
    """
    Create the deterministic payment transaction.
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
    Deterministic FailureModel used only by this test.

    Only the exact physical channel:

        B -> C -> key=0

    is forced to fail.

    Every other route succeeds.

    The real PaymentSimulator passes Top-K physical edges
    as dictionaries, therefore this test normalizes the
    dictionary representation before checking the failed
    physical channel.
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
        Evaluate the exact physical route edges.

        Top-K edge format:

            {
                "source": ...,
                "target": ...,
                "channel_key": ...,
                "scid": ...,
                "data": ...
            }
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

            if not isinstance(
                edge,
                dict,
            ):

                return {
                    "success": False,
                    "reason": "invalid_edge_format",
                    "visited_edges": [],
                }

            normalized_edges.append(
                (
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("channel_key"),
                )
            )

        # ----------------------------------------------------
        # Controlled physical-channel failure
        # ----------------------------------------------------

        if self.FAILED_EDGE in normalized_edges:

            failure_index = (
                normalized_edges.index(
                    self.FAILED_EDGE
                )
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

    This object is used by the real PaymentSimulator.

    It intentionally introduces no stochastic network changes.
    """

    def __init__(self):

        self.reset_calls = []

        self.update_calls = 0

        self.settlement_calls = 0

        self.payment_records = []

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
        reset_balances=False,
    ):

        self.reset_calls.append(
            bool(reset_balances)
        )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(self):

        self.update_calls += 1

    # --------------------------------------------------------
    # Settlement
    # --------------------------------------------------------

    def settle_route(
        self,
        route_edges,
        amount,
    ):

        self.settlement_calls += 1

        return True

    # --------------------------------------------------------
    # Payment record
    # --------------------------------------------------------

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

    No PPO training is performed.

    The untrained PPO model only produces one eta action,
    which is then passed through the real RoutingEnv.
    """

    return {

        # ----------------------------------------------------
        # RL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Graph / routing
        # ----------------------------------------------------

        "graph": {

            "max_hops": 6,

            "lambda_h": 1.0,

            "neighborhood_k": 6,

            "neighborhood_m": 3,
        },

        # ----------------------------------------------------
        # Simulation
        # ----------------------------------------------------

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
    Return exact physical-channel identity of a Top-K
    candidate.
    """

    edges = candidate.get(
        "edges"
    )

    if not isinstance(
        edges,
        (list, tuple),
    ):

        raise AssertionError(
            "Candidate edges must be a list or tuple."
        )

    identity = []

    for edge in edges:

        if not isinstance(
            edge,
            dict,
        ):

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
        Validate the complete integration pipeline:

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
        Bucket alternative
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

        failure_model = (
            ControlledFailureModel()
        )

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

        expected_alternative_path = [
            "A",
            "B",
            "F",
            "G",
            "E",
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

        for u, v, key, data in G.edges(
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

            # Capacity exists independently but is not
            # used as directional liquidity fallback.
            self.assertIn(
                "capacity",
                data,
            )

        # ====================================================
        # 3. Dijkstra baseline
        # ====================================================

        dijkstra = Dijkstra(
            G
        )

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

        dijkstra_path = list(
            dijkstra_result["path"]
        )

        self.assertEqual(
            dijkstra_path,
            expected_primary_path,
        )

        # ====================================================
        # 4. Direct adaptive-edge-cost validation
        #
        # IMPORTANT:
        #
        # Do NOT test eta by assuming that Top-K must return
        # different candidate sets or different total path
        # costs.
        #
        # The correct invariant is that the unified adaptive
        # edge-cost function changes when eta changes.
        # ====================================================

        test_edge = G[
            "A"
        ][
            "B"
        ][
            0
        ]

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

        self.assertIsInstance(
            adaptive_eta_0,
            dict,
        )

        self.assertIsInstance(
            adaptive_eta_1,
            dict,
        )

        # ----------------------------------------------------
        # Required API fields
        # ----------------------------------------------------

        for result in (
            adaptive_eta_0,
            adaptive_eta_1,
        ):

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

        # Native LND cost does not depend on eta.
        self.assertAlmostEqual(
            adaptive_eta_0["native_cost"],
            adaptive_eta_1["native_cost"],
            places=9,
        )

        # ----------------------------------------------------
        # Raw adaptive heuristic must change with eta.
        # ----------------------------------------------------

        self.assertNotAlmostEqual(
            adaptive_eta_0["raw_heuristic"],
            adaptive_eta_1["raw_heuristic"],
            places=9,
            msg=(
                "Changing eta did not change the raw "
                "adaptive heuristic."
            ),
        )

        # ----------------------------------------------------
        # Adaptive penalty must change with eta.
        # ----------------------------------------------------

        self.assertNotAlmostEqual(
            adaptive_eta_0["adaptive_penalty"],
            adaptive_eta_1["adaptive_penalty"],
            places=9,
            msg=(
                "Changing eta did not change the adaptive "
                "penalty."
            ),
        )

        # ----------------------------------------------------
        # Final adaptive edge cost must change with eta.
        # ----------------------------------------------------

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
        # 6. Top-K candidate structure
        # ====================================================

        identities = []

        for candidate in baseline_candidates:

            self.assertIsInstance(
                candidate,
                dict,
            )

            self.assertIn(
                "path",
                candidate,
            )

            self.assertIn(
                "edges",
                candidate,
            )

            self.assertIn(
                "cost",
                candidate,
            )

            self.assertIn(
                "eta",
                candidate,
            )

            self.assertIn(
                "lambda_h",
                candidate,
            )

            self.assertIn(
                "raw_heuristic",
                candidate,
            )

            self.assertIn(
                "adaptive_penalty",
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

            # ------------------------------------------------
            # Physical edge structure
            # ------------------------------------------------

            for edge in candidate["edges"]:

                self.assertIsInstance(
                    edge,
                    dict,
                )

                self.assertIn(
                    "source",
                    edge,
                )

                self.assertIn(
                    "target",
                    edge,
                )

                self.assertIn(
                    "channel_key",
                    edge,
                )

                self.assertIn(
                    "scid",
                    edge,
                )

                self.assertIn(
                    "data",
                    edge,
                )

        # No duplicate physical route.
        self.assertEqual(
            len(identities),
            len(set(identities)),
        )

        # ====================================================
        # 7. Candidate eta propagation
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
            msg=(
                "The deterministic E2E graph must select "
                "the primary route first."
            ),
        )

        # ====================================================
        # 9. First physical edge identity
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
        # 10. Alternative route must already exist
        # ====================================================

        candidate_paths = [
            list(
                candidate["path"]
            )
            for candidate
            in baseline_candidates
        ]

        self.assertIn(
            expected_alternative_path,
            candidate_paths,
            msg=(
                "The expected Bucket fallback route was "
                "not generated by Top-K."
            ),
        )

        # ====================================================
        # 11. ETA propagation through Top-K
        #
        # The invariant here is NOT that route sets must
        # differ.
        #
        # The invariant is that Top-K receives eta and
        # evaluates the adaptive objective with that eta.
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

        self.assertIsInstance(
            eta0_candidates,
            list,
        )

        self.assertIsInstance(
            eta1_candidates,
            list,
        )

        self.assertGreater(
            len(eta0_candidates),
            0,
            msg=(
                "Top-K returned no candidates for eta=0."
            ),
        )

        self.assertGreater(
            len(eta1_candidates),
            0,
            msg=(
                "Top-K returned no candidates for eta=1."
            ),
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

            observation, reset_info = (
                env.reset(
                    seed=123
                )
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
            # 14. Build PPO
            # =================================================

            ppo_model = build_ppo(
                env=env,
                cfg=cfg,
                seed=123,
            )

            # =================================================
            # 15. PPO action
            # =================================================

            ppo_action, _ = (
                ppo_model.predict(
                    observation,
                    deterministic=True,
                )
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
            # 16. Real environment step
            #
            # This invokes the actual repository pipeline:
            #
            # PPO action
            #      ->
            # eta
            #      ->
            # Top-K
            #      ->
            # CandidateManager
            #      ->
            # Bucket
            #      ->
            # PaymentSimulator
            #      ->
            # FailureModel
            #      ->
            # PartialBacktracker
            #      ->
            # PaymentSimulator retry
            #
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
            # 17. PPO -> eta
            # =================================================

            self.assertAlmostEqual(
                float(
                    info["eta"]
                ),
                ppo_eta,
                places=6,
            )

            # =================================================
            # 18. Fixed Top-K
            # =================================================

            self.assertEqual(
                info["top_k"],
                5,
            )

            # =================================================
            # 19. Candidate generation
            # =================================================

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
            # 20. Bucket
            # =================================================

            self.assertIsNotNone(
                env.current_bucket
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
            # 21. Payment attempts
            #
            # The real environment must have:
            #
            #   attempt 1 -> primary -> B-C failure
            #   attempt 2 -> alternative -> success
            #
            # PaymentSimulator records both attempts through
            # NetworkDynamics.record_payment().
            # =================================================

            self.assertGreaterEqual(
                len(
                    network_dynamics.payment_records
                ),
                2,
                msg=(
                    "Expected at least two payment attempts."
                ),
            )

            # =================================================
            # 22. First payment must fail
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
                    "The first payment attempt did not fail."
                ),
            )

            # =================================================
            # 23. Failure must be B-C-0
            # =================================================

            self.assertEqual(
                first_result["failed_edge"],
                (
                    "B",
                    "C",
                    0,
                ),
                msg=(
                    "The first payment did not fail on "
                    "the controlled physical channel B-C-0."
                ),
            )

            self.assertEqual(
                first_result["reason"],
                "controlled_channel_failure",
            )

            # =================================================
            # 24. Partial Backtracking
            # =================================================

            self.assertGreaterEqual(
                info["attempt_count"],
                2,
            )

            self.assertGreaterEqual(
                info["partial_backtrack_count"],
                1,
                msg=(
                    "The controlled B-C-0 failure occurred, "
                    "but PartialBacktracker was not invoked."
                ),
            )

            self.assertGreaterEqual(
                info["partial_backtrack_success"],
                1,
                msg=(
                    "PartialBacktracker was invoked but "
                    "did not produce a successful retry."
                ),
            )

            # =================================================
            # 25. No full reroute
            # =================================================

            self.assertEqual(
                info["full_reroute_count"],
                0,
                msg=(
                    "Full rerouting occurred although a "
                    "valid Bucket alternative existed."
                ),
            )

            self.assertFalse(
                info["full_reroute"],
            )

            # =================================================
            # 26. Final payment
            # =================================================

            self.assertTrue(
                info["payment_success"],
                msg=(
                    "The final payment was not successful."
                ),
            )

            self.assertTrue(
                info["success"],
            )

            self.assertEqual(
                info["reason"],
                "partial_backtrack_success",
            )

            # =================================================
            # 27. Final path
            # =================================================

            final_path = list(
                info["path"]
            )

            self.assertEqual(
                final_path,
                expected_alternative_path,
                msg=(
                    "The successful retry did not use "
                    "the expected Bucket alternative."
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
            # 28. Final payment record
            # =================================================

            final_record = (
                network_dynamics.payment_records[-1]
            )

            final_result = (
                final_record["result"]
            )

            self.assertTrue(
                final_result["success"],
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
            # 29. Exactly one settlement
            #
            # The failed first attempt must NOT settle.
            #
            # The successful retry must settle exactly once.
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
            # 30. Bucket attempts
            #
            # Bucket.record_attempt() is used for the first
            # normal candidate execution.
            #
            # The environment records the retry separately
            # through its real retry path.
            #
            # The environment's reported attempt_count is
            # therefore the authoritative E2E metric.
            # =================================================

            self.assertGreaterEqual(
                info["attempt_count"],
                2,
            )

            # =================================================
            # 31. Output types
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
            # 32. Terminal state
            #
            # The environment deliberately preserves the
            # current Bucket after the terminal step.
            # =================================================

            self.assertIsNotNone(
                env.current_bucket
            )

            # =================================================
            # 33. PASS summary
            # =================================================

            print()

            print(
                "=" * 78
            )

            print(
                "END-TO-END PPO -> ADAPTIVE ROUTING -> "
                "BUCKET -> PARTIAL BACKTRACKING : PASS"
            )

            print(
                "=" * 78
            )

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

            print(
                "=" * 78
            )

        finally:

            env.close()


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )