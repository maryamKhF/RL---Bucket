"""
End-to-End validation for the complete RL + adaptive routing pipeline.

Validated flow
--------------

    PPO
      |
      v
    eta
      |
      v
    Adaptive Heuristic
      |
      +--------------------+
      |                    |
      v                    v
   Dijkstra             Top-K
                         k=5
                           |
                           v
                        Bucket
                           |
                           v
                  Payment Simulator
                           |
                           v
                     Failure Model
                           |
                           v
                  Partial Backtracking
                           |
                           v
                    Bucket Alternative
                           |
                           v
                   Payment Simulator
                           |
                           v
                       SUCCESS


Repository invariants
---------------------

1. PPO controls eta only.
2. k is fixed at 5.
3. Dijkstra and Top-K use the adaptive routing objective.
4. Top-K is channel-aware.
5. Top-K physical edges are dictionaries.
6. Bucket stores candidate routes.
7. PaymentSimulator executes one route per attempt.
8. FailureModel evaluates the actual physical edges.
9. PartialBacktracking reuses an existing Bucket candidate.
10. No full reroute is required when a valid alternative exists.
"""


import unittest
from types import SimpleNamespace

import networkx as nx
import numpy as np

from Bucket.bucket import Bucket
from Pathfinding.dijkstra import Dijkstra
from Pathfinding.heuristics import lnd_cost
from Pathfinding.top_k_paths import top_k_paths
from RL.environment import RoutingEnv
from RL.ppo_agent import build_ppo
from Simulation.failure_model import FailureModel


# ============================================================
# Deterministic Network
# ============================================================

def build_e2e_graph():
    """
    Build the deterministic graph used by the E2E test.

    Primary route:

        A -> B -> C -> D -> E

    Alternative route:

        A -> B -> F -> G -> E

    Both routes share A -> B.

    The primary route is deliberately cheaper so that it is
    selected first. The controlled FailureModel then forces
    B -> C to fail, allowing Partial Backtracking to select
    the already stored alternative route.
    """

    G = nx.MultiDiGraph()

    # --------------------------------------------------------
    # Node metadata
    #
    # These fields satisfy the current Network.topology and
    # Dijkstra requirements.
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
            "rgb_color": (0, 0, 0),
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

    # ========================================================
    # Alternative route
    # ========================================================

    alternative_edges = [
        ("B", "F", 0),
        ("F", "G", 0),
        ("G", "E", 0),
    ]

    # ========================================================
    # Primary route
    #
    # Lower fee -> lower native routing cost.
    # ========================================================

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

            scid=f"{u}-{v}-0",
        )

    # ========================================================
    # Alternative route
    #
    # Higher fee -> lower priority than primary route.
    # ========================================================

    for u, v, key in alternative_edges:

        G.add_edge(
            u,
            v,
            key=key,

            capacity=10000.0,

            estimated_liquidity=10000.0,
            balance_uv=10000.0,

            available=True,

            fee_base=20.0,
            fee_rate=0.0,

            delay=1.0,

            failure_probability=0.0,

            scid=f"{u}-{v}-0",
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
    Deterministic FailureModel for E2E validation.

    The physical channel:

        B -> C -> key=0

    is forced to fail.

    Any other route succeeds.

    Top-K represents physical edges as dictionaries:

        {
            "source": ...,
            "target": ...,
            "channel_key": ...,
            "scid": ...,
            "data": ...
        }

    Therefore this class explicitly normalizes those dictionaries
    before checking the failed physical channel.
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
        Evaluate the actual physical route edges.

        The FailureModel returns canonical tuples for failure
        identification while preserving the original edge
        dictionaries in visited_edges.
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
        # Controlled failure
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
        # All other routes succeed.
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
    # Record payment
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

    No training is performed in this E2E test.

    PPO is instantiated and produces the eta action that is
    passed into RoutingEnv.step().
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
# E2E Test
# ============================================================

class EndToEndRoutingTest(
    unittest.TestCase
):

    def test_complete_ppo_to_backtracking_pipeline(
        self,
    ):
        """
        Validate the complete pipeline:

            PPO
             |
             v
            eta
             |
             v
           Top-K
             |
             v
           Bucket
             |
             v
        Payment #1
             |
             v
          Failure
             |
             v
        Backtracking
             |
             v
        Bucket Alternative
             |
             v
        Payment #2
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

        # ====================================================
        # 1. Dijkstra
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

        expected_primary_path = [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

        self.assertEqual(
            dijkstra_path,
            expected_primary_path,
        )

        # ====================================================
        # 2. Top-K
        # ====================================================

        eta_for_consistency = 0.5

        candidates = top_k_paths(
            G=G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=lnd_cost,
            eta=eta_for_consistency,
            k=5,
            max_hops=6,
            lambda_h=1.0,
        )

        self.assertIsInstance(
            candidates,
            list,
        )

        self.assertGreaterEqual(
            len(candidates),
            2,
            msg=(
                "The deterministic E2E graph must "
                "produce at least two candidate routes."
            ),
        )

        # ====================================================
        # 3. Dijkstra / Top-K consistency
        # ====================================================

        self.assertEqual(
            list(candidates[0]["path"]),
            dijkstra_path,
        )

        # Top-K edges are dictionaries, not tuples.
        first_candidate_edge = (
            candidates[0]["edges"][0]
        )

        self.assertIsInstance(
            first_candidate_edge,
            dict,
        )

        self.assertEqual(
            first_candidate_edge["source"],
            "A",
        )

        self.assertEqual(
            first_candidate_edge["target"],
            "B",
        )

        self.assertEqual(
            first_candidate_edge["channel_key"],
            0,
        )

        self.assertEqual(
            first_candidate_edge["scid"],
            "A-B-0",
        )

        # ====================================================
        # 4. Verify alternative candidate
        # ====================================================

        expected_alternative_path = [
            "A",
            "B",
            "F",
            "G",
            "E",
        ]

        candidate_paths = [
            list(candidate["path"])
            for candidate in candidates
        ]

        self.assertIn(
            expected_alternative_path,
            candidate_paths,
            msg=(
                "The E2E graph must provide the expected "
                "Bucket fallback route."
            ),
        )

        # ====================================================
        # 5. ETA effect
        # ====================================================

        candidates_eta_0 = top_k_paths(
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

        candidates_eta_1 = top_k_paths(
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
            candidates_eta_0,
            list,
        )

        self.assertIsInstance(
            candidates_eta_1,
            list,
        )

        costs_eta_0 = {
            tuple(candidate["path"]):
                float(candidate["cost"])
            for candidate in candidates_eta_0
        }

        costs_eta_1 = {
            tuple(candidate["path"]):
                float(candidate["cost"])
            for candidate in candidates_eta_1
        }

        common_routes = (
            set(costs_eta_0)
            &
            set(costs_eta_1)
        )

        self.assertTrue(
            common_routes,
            msg=(
                "ETA sweep produced no common "
                "candidate route."
            ),
        )

        eta_effect_observed = any(
            abs(
                costs_eta_0[path]
                -
                costs_eta_1[path]
            )
            > 1e-9
            for path in common_routes
        )

        self.assertTrue(
            eta_effect_observed,
            msg=(
                "Changing eta did not change the "
                "adaptive routing objective."
            ),
        )

        # ====================================================
        # 6. Routing Environment
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
            # 7. Reset
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

            # =================================================
            # 8. PPO
            # =================================================

            ppo_model = build_ppo(
                env=env,
                cfg=cfg,
                seed=123,
            )

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

            eta = float(
                ppo_action[0]
            )

            self.assertGreaterEqual(
                eta,
                env.eta_min,
            )

            self.assertLessEqual(
                eta,
                env.eta_max,
            )

            # =================================================
            # 9. Actual Environment Pipeline
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

            # =================================================
            # 10. PPO -> eta
            # =================================================

            self.assertAlmostEqual(
                float(info["eta"]),
                eta,
                places=6,
            )

            # =================================================
            # 11. Fixed Top-K
            # =================================================

            self.assertEqual(
                info["top_k"],
                5,
            )

            self.assertGreaterEqual(
                info["candidate_path_count"],
                2,
            )

            self.assertGreaterEqual(
                info["usable_candidate_count"],
                2,
            )

            # =================================================
            # 12. Bucket
            # =================================================

            self.assertGreaterEqual(
                info["bucket_size"],
                2,
            )

            self.assertIsInstance(
                env.current_bucket,
                Bucket,
            )

            # At least two attempts are required:
            #
            #   1. primary route
            #   2. alternative route
            #
            self.assertGreaterEqual(
                env.current_bucket.attempts,
                2,
            )

            # =================================================
            # 13. Failure + Partial Backtracking
            # =================================================

            self.assertGreaterEqual(
                info["attempt_count"],
                2,
            )

            self.assertGreaterEqual(
                info["partial_backtrack_count"],
                1,
            )

            self.assertGreaterEqual(
                info["partial_backtrack_success"],
                1,
            )

            # =================================================
            # 14. No full reroute
            # =================================================

            self.assertEqual(
                info["full_reroute_count"],
                0,
            )

            self.assertFalse(
                info["full_reroute"],
            )

            # =================================================
            # 15. Final payment
            # =================================================

            self.assertTrue(
                info["payment_success"],
                msg=(
                    "E2E payment failed: "
                    f"{info}"
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
            # 16. Final path
            # =================================================

            final_path = list(
                info["path"]
            )

            self.assertEqual(
                final_path,
                expected_alternative_path,
            )

            self.assertNotEqual(
                final_path,
                dijkstra_path,
            )

            self.assertNotIn(
                "C",
                final_path,
            )

            # =================================================
            # 17. PaymentSimulator execution
            # =================================================

            self.assertGreaterEqual(
                network_dynamics.settlement_calls,
                1,
            )

            self.assertGreaterEqual(
                len(
                    network_dynamics.payment_records
                ),
                2,
            )

            # Only the successful second payment reaches
            # settlement.

            self.assertEqual(
                network_dynamics.settlement_calls,
                1,
            )

            # =================================================
            # 18. Recorded payment results
            # =================================================

            recorded_results = [
                record["result"]
                for record in
                network_dynamics.payment_records
            ]

            self.assertGreaterEqual(
                len(recorded_results),
                2,
            )

            first_result = (
                recorded_results[0]
            )

            final_result = (
                recorded_results[-1]
            )

            # -------------------------------------------------
            # First attempt must fail.
            # -------------------------------------------------

            self.assertFalse(
                first_result["success"],
            )

            self.assertEqual(
                first_result["failed_edge"],
                (
                    "B",
                    "C",
                    0,
                ),
            )

            # -------------------------------------------------
            # Final attempt must succeed.
            # -------------------------------------------------

            self.assertTrue(
                final_result["success"],
            )

            self.assertEqual(
                final_result["path"],
                expected_alternative_path,
            )

            # =================================================
            # 19. Final environment outputs
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
            # 20. PASS summary
            # =================================================

            print()

            print(
                "=" * 72
            )

            print(
                "END-TO-END ROUTING TEST : PASS"
            )

            print(
                "=" * 72
            )

            print(
                f"Dijkstra path             : "
                f"{dijkstra_path}"
            )

            print(
                f"PPO eta                   : "
                f"{eta:.6f}"
            )

            print(
                f"Top-K candidates          : "
                f"{info['candidate_path_count']}"
            )

            print(
                f"Usable candidates         : "
                f"{info['usable_candidate_count']}"
            )

            print(
                f"Bucket size               : "
                f"{info['bucket_size']}"
            )

            print(
                f"Payment attempts          : "
                f"{info['attempt_count']}"
            )

            print(
                f"Partial backtracks        : "
                f"{info['partial_backtrack_count']}"
            )

            print(
                f"Backtrack successes       : "
                f"{info['partial_backtrack_success']}"
            )

            print(
                f"Full reroutes             : "
                f"{info['full_reroute_count']}"
            )

            print(
                f"Final path                : "
                f"{final_path}"
            )

            print(
                f"Payment success           : "
                f"{info['payment_success']}"
            )

            print(
                f"Final reason              : "
                f"{info['reason']}"
            )

            print(
                "=" * 72
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