
"""
Evaluations/test_baselines.py

Tests for:
    Evaluations.baselines

Test coverage
-------------

1. Method classification
2. Failure-configuration helpers
3. MultiDiGraph baseline execution
4. All six LND-based baseline methods
5. PaymentSimulator integration
6. Exact MultiDiGraph channel-key preservation
7. Output-schema consistency
8. Routing failure handling
9. Invalid-method handling
10. Proposed RL + Bucket interface
11. Proposed-method execution with a lightweight mock model

Important
---------
These tests intentionally DO NOT train PPO.

The purpose of this file is to validate the evaluation layer
without making the test suite dependent on a long RL training
run.

The six baseline methods remain independent from RoutingEnv:

    Dijkstra
        |
        v
    PaymentSimulator
        |
        +--> FailureModel
        |
        +--> NetworkDynamics

The proposed method is tested through RoutingEnv.
"""


from __future__ import annotations


import unittest
from unittest.mock import MagicMock, patch


import networkx as nx
import numpy as np


from Evaluations.baselines import (
    BASELINE_METHODS,
    STATIC_ETA,
    HEURISTICS,
    is_baseline_method,
    is_proposed_method,
    run_method,
    _resolve_failure_probability,
    _resolve_node_failure_probability,
    _resolve_liquidity_failure_probability,
    _resolve_node_failure_rate,
    _resolve_recovery_rate,
    _zero_result_row,
)


from Simulation.transaction_generator import (
    Transaction,
)


# ============================================================
# Constants
# ============================================================

EXPECTED_BASELINES = {
    "native_lnd",
    "native_cln",
    "native_ecl",
    "static_lnd",
    "static_cln",
    "static_ecl",
}


REQUIRED_RESULT_KEYS = {
    "transaction_id",
    "tx_id",
    "eta",
    "top_k",
    "candidate_path_count",
    "usable_candidate_count",
    "bucket_size",
    "success",
    "payment_success",
    "path",
    "path_length",
    "fee",
    "delay",
    "carbon",
    "failure_probability",
    "average_delay",
    "backtrack_count",
    "partial_backtrack_count",
    "partial_backtrack_success",
    "full_reroute",
    "full_reroute_count",
    "attempt_count",
    "reason",
    "episode_reward",
    "inter_country_hops",
    "inter_continent_hops",
    "runtime",
    "recovered",
    "amount",
}


# ============================================================
# Test Configuration
# ============================================================

def create_test_config():
    """
    Minimal configuration required by baselines.py.

    Failure probabilities are set to zero so that the baseline
    integration tests are deterministic and focus on the
    evaluation pipeline itself.
    """

    return {

        "seed": 42,

        "graph": {

            "max_hops": 12,

            "top_k_paths": 5,

            "neighborhood_k": 15,

            "neighborhood_m": 5,

            "ego_radius": 2,

        },

        "simulation": {

            "channel_failure_rate": 0.0,

            "node_failure_probability": 0.0,

            "liquidity_failure_probability": 0.0,

            "node_failure_rate": 0.0,

            "recovery_rate": 1.0,

        },

        "bucket": {

            "max_candidates": 5,

            "selection_strategy": "score",

            "enable_backtracking": True,

            "max_backtracking_depth": 3,

        },

    }


# ============================================================
# Test Graph
# ============================================================

def create_test_graph():
    """
    Create a small deterministic MultiDiGraph.

    The graph deliberately contains parallel channels between
    0 and 1 so that channel identity can be tested.

    Topology:

        0 ---- key 0 ----> 1 ----> 2
         \
          ---- key 1 ----> 1

    and reverse directed channels are also present.

    Expected routing:

        0 -> 1 -> 2

    Exact channel key must be retained.
    """

    G = nx.MultiDiGraph()

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    G.add_node(
        "0",
        country="US",
        latitude=40.0,
        longitude=-74.0,
        carbon_intensity=100.0,
        online=True,
        rgb_color="3399ff",
    )

    G.add_node(
        "1",
        country="US",
        latitude=41.0,
        longitude=-73.0,
        carbon_intensity=120.0,
        online=True,
        rgb_color="3399ff",
    )

    G.add_node(
        "2",
        country="CA",
        latitude=43.0,
        longitude=-79.0,
        carbon_intensity=140.0,
        online=True,
        rgb_color="3399ff",
    )

    # --------------------------------------------------------
    # Parallel channels 0 -> 1
    # --------------------------------------------------------

    G.add_edge(
        "0",
        "1",
        key=0,
        fee_base=1.0,
        fee_rate=1.0,
        delay=10,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    G.add_edge(
        "0",
        "1",
        key=1,
        fee_base=2.0,
        fee_rate=2.0,
        delay=20,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    # --------------------------------------------------------
    # Channel 1 -> 2
    # --------------------------------------------------------

    G.add_edge(
        "1",
        "2",
        key=0,
        fee_base=1.0,
        fee_rate=1.0,
        delay=15,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    # --------------------------------------------------------
    # Reverse channels
    # --------------------------------------------------------

    G.add_edge(
        "1",
        "0",
        key=0,
        fee_base=1.0,
        fee_rate=1.0,
        delay=10,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    G.add_edge(
        "1",
        "0",
        key=1,
        fee_base=2.0,
        fee_rate=2.0,
        delay=20,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    G.add_edge(
        "2",
        "1",
        key=0,
        fee_base=1.0,
        fee_rate=1.0,
        delay=15,
        capacity=1_000_000,
        failure_probability=0.0,
        available=True,
    )

    return G


# ============================================================
# Test Transaction
# ============================================================

def create_test_transaction(
    tx_id=1,
    source="0",
    destination="2",
    amount=1000.0,
):
    """
    Create a deterministic payment transaction.
    """

    return Transaction(
        tx_id=tx_id,
        source=source,
        destination=destination,
        amount=amount,
        timestamp=0,
        split="test",
    )


# ============================================================
# Test Class
# ============================================================

class BaselinesEvaluationTest(
    unittest.TestCase
):

    # ========================================================
    # Method Classification
    # ========================================================

    def test_baseline_method_set(self):
        """
        All six LND-based baselines must exist.
        """

        self.assertEqual(
            BASELINE_METHODS,
            EXPECTED_BASELINES,
        )

    def test_baseline_classification(self):
        """
        Every LND-based method must be classified as a baseline.
        """

        for name in EXPECTED_BASELINES:

            self.assertTrue(
                is_baseline_method(name),
                msg=f"{name} was not recognized as baseline",
            )

            self.assertFalse(
                is_proposed_method(name),
                msg=f"{name} was incorrectly recognized as proposed",
            )

    def test_proposed_classification(self):
        """
        Improved/RL method names must be recognized as proposed.
        """

        proposed_names = [
            "improved_lnd",
            "improved_rl",
            "rl",
            "rl_bucket",
        ]

        for name in proposed_names:

            self.assertTrue(
                is_proposed_method(name),
                msg=f"{name} was not recognized as proposed",
            )

            self.assertFalse(
                is_baseline_method(name),
                msg=f"{name} was incorrectly recognized as baseline",
            )

    # ========================================================
    # Heuristic Configuration
    # ========================================================

    def test_heuristic_configuration(self):
        """
        Verify all native heuristic mappings exist.
        """

        self.assertIn(
            "native_lnd",
            HEURISTICS,
        )

        self.assertIn(
            "native_cln",
            HEURISTICS,
        )

        self.assertIn(
            "native_ecl",
            HEURISTICS,
        )

        self.assertEqual(
            set(HEURISTICS.keys()),
            {
                "native_lnd",
                "native_cln",
                "native_ecl",
            },
        )

    def test_static_eta_configuration(self):
        """
        Verify all static eta values are finite and inside [0, 1].
        """

        for name, eta in STATIC_ETA.items():

            self.assertGreaterEqual(
                eta,
                0.0,
            )

            self.assertLessEqual(
                eta,
                1.0,
            )

            self.assertTrue(
                np.isfinite(eta),
            )

    # ========================================================
    # Configuration Helpers
    # ========================================================

    def test_failure_configuration_helpers(self):
        """
        Verify simulation configuration values are read correctly.
        """

        cfg = create_test_config()

        self.assertEqual(
            _resolve_failure_probability(cfg),
            0.0,
        )

        self.assertEqual(
            _resolve_node_failure_probability(cfg),
            0.0,
        )

        self.assertEqual(
            _resolve_liquidity_failure_probability(cfg),
            0.0,
        )

        self.assertEqual(
            _resolve_node_failure_rate(cfg),
            0.0,
        )

        self.assertEqual(
            _resolve_recovery_rate(cfg),
            1.0,
        )

    def test_invalid_failure_probability_is_rejected(self):
        """
        Invalid probabilities must not be silently accepted.
        """

        cfg = create_test_config()

        cfg["simulation"][
            "channel_failure_rate"
        ] = 1.5

        with self.assertRaises(
            ValueError
        ):

            _resolve_failure_probability(
                cfg
            )

    # ========================================================
    # Zero Result Schema
    # ========================================================

    def test_zero_result_row_schema(self):
        """
        Failed routing results must contain the complete
        common evaluation schema.
        """

        tx = create_test_transaction()

        row = _zero_result_row(
            tx=tx,
            reason="routing_failed",
            eta=0.0,
            top_k=1,
        )

        self.assertTrue(
            REQUIRED_RESULT_KEYS.issubset(
                row.keys()
            )
        )

        self.assertEqual(
            row["tx_id"],
            tx.tx_id,
        )

        self.assertFalse(
            row["success"]
        )

        self.assertFalse(
            row["payment_success"]
        )

        self.assertEqual(
            row["reason"],
            "routing_failed",
        )

    # ========================================================
    # Single Baseline Integration
    # ========================================================

    def test_native_lnd_execution(self):
        """
        native_lnd must execute successfully on the deterministic
        MultiDiGraph.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        rows = run_method(
            G=G,
            transactions=[tx],
            name="native_lnd",
            cfg=cfg,
            seed=42,
        )

        self.assertEqual(
            len(rows),
            1,
        )

        row = rows[0]

        self.assertTrue(
            REQUIRED_RESULT_KEYS.issubset(
                row.keys()
            )
        )

        self.assertTrue(
            row["success"],
            msg=f"native_lnd failed: {row}",
        )

        self.assertEqual(
            row["tx_id"],
            tx.tx_id,
        )

        self.assertEqual(
            row["top_k"],
            1,
        )

        self.assertEqual(
            row["bucket_size"],
            0,
        )

        self.assertEqual(
            row["partial_backtrack_count"],
            0,
        )

        self.assertEqual(
            row["full_reroute_count"],
            0,
        )

        self.assertGreater(
            row["path_length"],
            0,
        )

        self.assertEqual(
            row["path"],
            ["0", "1", "2"],
        )

    # ========================================================
    # All Six Baselines
    # ========================================================

    def test_all_lnd_based_baselines_execute(self):
        """
        All six LND-based baseline methods must run successfully.

        This test verifies that the baseline layer does not
        accidentally depend on PPO, Bucket, or RoutingEnv.
        """

        cfg = create_test_config()

        for name in sorted(
            EXPECTED_BASELINES
        ):

            with self.subTest(
                method=name
            ):

                G = create_test_graph()

                tx = create_test_transaction()

                rows = run_method(
                    G=G,
                    transactions=[tx],
                    name=name,
                    cfg=cfg,
                    seed=42,
                )

                self.assertEqual(
                    len(rows),
                    1,
                )

                row = rows[0]

                self.assertTrue(
                    REQUIRED_RESULT_KEYS.issubset(
                        row.keys()
                    )
                )

                self.assertTrue(
                    row["success"],
                    msg=(
                        f"{name} failed unexpectedly: "
                        f"{row}"
                    ),
                )

                self.assertEqual(
                    row["path_length"],
                    2,
                )

    # ========================================================
    # Exact MultiDiGraph Channel Identity
    # ========================================================

    def test_multidigraph_channel_identity_is_preserved(self):
        """
        Verify that the route returned by Dijkstra and executed
        by PaymentSimulator remains valid on a MultiDiGraph.

        The graph contains parallel channels between 0 and 1.

        A valid execution must therefore use exact channel
        identity internally rather than treating (u, v) as
        sufficient identity.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        rows = run_method(
            G=G,
            transactions=[tx],
            name="native_lnd",
            cfg=cfg,
            seed=42,
        )

        self.assertEqual(
            len(rows),
            1,
        )

        row = rows[0]

        self.assertTrue(
            row["success"]
        )

        self.assertEqual(
            row["path"],
            ["0", "1", "2"],
        )

        # Confirm the graph still contains both parallel
        # channels after evaluation.
        self.assertEqual(
            len(
                G.get_edge_data(
                    "0",
                    "1",
                )
            ),
            2,
        )

        self.assertIn(
            0,
            G.get_edge_data(
                "0",
                "1",
            ),
        )

        self.assertIn(
            1,
            G.get_edge_data(
                "0",
                "1",
            ),
        )

    # ========================================================
    # No Bucket For LND Baselines
    # ========================================================

    def test_lnd_baseline_does_not_use_bucket(self):
        """
        LND-based baselines must remain independent from Bucket.

        Bucket-related values must therefore remain zero/empty.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        rows = run_method(
            G=G,
            transactions=[tx],
            name="native_lnd",
            cfg=cfg,
            seed=42,
        )

        row = rows[0]

        self.assertEqual(
            row["bucket_size"],
            0,
        )

        self.assertEqual(
            row["backtrack_count"],
            0,
        )

        self.assertEqual(
            row["partial_backtrack_count"],
            0,
        )

        self.assertEqual(
            row["partial_backtrack_success"],
            0,
        )

        self.assertFalse(
            row["full_reroute"]
        )

        self.assertEqual(
            row["full_reroute_count"],
            0,
        )

    # ========================================================
    # Routing Failure
    # ========================================================

    def test_routing_failure_returns_schema(self):
        """
        If no route exists, the evaluation layer must return
        a structured failed row rather than raising an exception.
        """

        G = nx.MultiDiGraph()

        G.add_node(
            "0",
            country="US",
            latitude=40.0,
            longitude=-74.0,
            carbon_intensity=100.0,
        )

        G.add_node(
            "1",
            country="US",
            latitude=41.0,
            longitude=-73.0,
            carbon_intensity=100.0,
        )

        tx = create_test_transaction(
            source="0",
            destination="1",
        )

        cfg = create_test_config()

        rows = run_method(
            G=G,
            transactions=[tx],
            name="native_lnd",
            cfg=cfg,
            seed=42,
        )

        self.assertEqual(
            len(rows),
            1,
        )

        row = rows[0]

        self.assertFalse(
            row["success"]
        )

        self.assertFalse(
            row["payment_success"]
        )

        self.assertEqual(
            row["path_length"],
            0,
        )

        self.assertEqual(
            row["reason"],
            "routing_failed",
        )

        self.assertTrue(
            REQUIRED_RESULT_KEYS.issubset(
                row.keys()
            )
        )

    # ========================================================
    # Multiple Transactions
    # ========================================================

    def test_multiple_transactions_produce_one_row_each(self):
        """
        One transaction must produce exactly one evaluation row.
        """

        G = create_test_graph()

        cfg = create_test_config()

        transactions = [

            create_test_transaction(
                tx_id=1,
            ),

            create_test_transaction(
                tx_id=2,
                amount=2000.0,
            ),

            create_test_transaction(
                tx_id=3,
                amount=3000.0,
            ),

        ]

        rows = run_method(
            G=G,
            transactions=transactions,
            name="native_lnd",
            cfg=cfg,
            seed=42,
        )

        self.assertEqual(
            len(rows),
            len(transactions),
        )

        self.assertEqual(
            [
                row["tx_id"]
                for row in rows
            ],
            [
                1,
                2,
                3,
            ],
        )

    # ========================================================
    # Invalid Method
    # ========================================================

    def test_invalid_method_is_rejected(self):
        """
        Unknown methods must raise ValueError rather than
        silently falling back to another routing strategy.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        with self.assertRaises(
            ValueError
        ):

            run_method(
                G=G,
                transactions=[tx],
                name="unknown_method",
                cfg=cfg,
                seed=42,
            )

    # ========================================================
    # Proposed Method Requires Model
    # ========================================================

    def test_proposed_method_requires_model(self):
        """
        RL + Bucket evaluation must not silently run without
        a trained PPO model.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        with self.assertRaises(
            ValueError
        ):

            run_method(
                G=G,
                transactions=[tx],
                name="improved_lnd",
                cfg=cfg,
                model=None,
                seed=42,
            )

    # ========================================================
    # Proposed Method: Lightweight Interface Test
    # ========================================================

    @patch(
        "Evaluations.baselines.RoutingEnv"
    )
    def test_proposed_method_interface(
        self,
        mock_env_class,
    ):
        """
        Validate the interface between baselines.py and
        RoutingEnv without training PPO or executing the
        complete routing pipeline.

        This test checks:

            model.predict()
                ->
            eta
                ->
            env.step([eta])
                ->
            evaluation row
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        # ----------------------------------------------------
        # Fake environment
        # ----------------------------------------------------

        mock_env = MagicMock()

        mock_env.transactions = [
            tx
        ]

        mock_env.reset.return_value = (
            np.zeros(
                180,
                dtype=np.float32,
            ),
            {
                "transaction_id": tx.tx_id,
                "eta": None,
                "top_k": 5,
            },
        )

        mock_env.step.return_value = (
            np.zeros(
                180,
                dtype=np.float32,
            ),
            0.5,
            True,
            False,
            {
                "transaction_id": tx.tx_id,
                "eta": 0.5,
                "top_k": 5,

                "candidate_path_count": 5,
                "usable_candidate_count": 5,
                "bucket_size": 5,

                "success": True,
                "payment_success": True,

                "path": [
                    "0",
                    "1",
                    "2",
                ],

                "path_length": 2,

                "fee": 2.0,
                "delay": 25.0,
                "carbon": 120.0,

                "failure_probability": 0.0,
                "average_delay": 25.0,

                "backtrack_count": 1,
                "partial_backtrack_count": 1,
                "partial_backtrack_success": 1,

                "full_reroute": False,
                "full_reroute_count": 0,

                "attempt_count": 2,

                "reason": "partial_backtrack_success",

                "episode_reward": 0.8,

                "inter_country_hops": 1,
                "inter_continent_hops": 0,
            },
        )

        mock_env_class.return_value = (
            mock_env
        )

        # ----------------------------------------------------
        # Fake PPO model
        # ----------------------------------------------------

        model = MagicMock()

        model.predict.return_value = (
            np.asarray(
                [0.5],
                dtype=np.float32,
            ),
            None,
        )

        # ----------------------------------------------------
        # Execute proposed method
        # ----------------------------------------------------

        rows = run_method(
            G=G,
            transactions=[tx],
            name="improved_lnd",
            cfg=cfg,
            model=model,
            seed=42,
        )

        # ----------------------------------------------------
        # Assertions
        # ----------------------------------------------------

        self.assertEqual(
            len(rows),
            1,
        )

        row = rows[0]

        self.assertTrue(
            row["success"]
        )

        self.assertTrue(
            row["payment_success"]
        )

        self.assertEqual(
            row["eta"],
            0.5,
        )

        self.assertEqual(
            row["top_k"],
            5,
        )

        self.assertEqual(
            row["candidate_path_count"],
            5,
        )

        self.assertEqual(
            row["usable_candidate_count"],
            5,
        )

        self.assertEqual(
            row["bucket_size"],
            5,
        )

        self.assertEqual(
            row["partial_backtrack_count"],
            1,
        )

        self.assertEqual(
            row["partial_backtrack_success"],
            1,
        )

        self.assertEqual(
            row["attempt_count"],
            2,
        )

        self.assertEqual(
            row["reason"],
            "partial_backtrack_success",
        )

        self.assertTrue(
            row["recovered"]
        )

        # ----------------------------------------------------
        # PPO must receive observation and produce one eta.
        # ----------------------------------------------------

        model.predict.assert_called_once()

        # ----------------------------------------------------
        # Environment must receive exactly one eta action.
        # ----------------------------------------------------

        mock_env.step.assert_called_once()

        step_action = (
            mock_env.step.call_args[0][0]
        )

        self.assertEqual(
            step_action.shape,
            (1,),
        )

        self.assertAlmostEqual(
            float(step_action[0]),
            0.5,
        )

    # ========================================================
    # Proposed Method Uses RoutingEnv
    # ========================================================

    @patch(
        "Evaluations.baselines.RoutingEnv"
    )
    def test_proposed_method_uses_routing_env(
        self,
        mock_env_class,
    ):
        """
        Ensure the proposed method goes through RoutingEnv
        rather than independently implementing Bucket logic.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        mock_env = MagicMock()

        mock_env.transactions = [
            tx
        ]

        mock_env.reset.return_value = (
            np.zeros(
                180,
                dtype=np.float32,
            ),
            {},
        )

        mock_env.step.return_value = (
            np.zeros(
                180,
                dtype=np.float32,
            ),
            0.0,
            True,
            False,
            {
                "transaction_id": tx.tx_id,
                "eta": 0.25,
                "top_k": 5,
                "candidate_path_count": 5,
                "usable_candidate_count": 5,
                "bucket_size": 5,
                "success": True,
                "payment_success": True,
                "path": [
                    "0",
                    "1",
                    "2",
                ],
                "path_length": 2,
                "fee": 2.0,
                "delay": 25.0,
                "carbon": 120.0,
                "failure_probability": 0.0,
                "average_delay": 25.0,
                "backtrack_count": 0,
                "partial_backtrack_count": 0,
                "partial_backtrack_success": 0,
                "full_reroute": False,
                "full_reroute_count": 0,
                "attempt_count": 1,
                "reason": "success",
                "episode_reward": 0.9,
            },
        )

        mock_env_class.return_value = (
            mock_env
        )

        model = MagicMock()

        model.predict.return_value = (
            np.asarray(
                [0.25],
                dtype=np.float32,
            ),
            None,
        )

        rows = run_method(
            G=G,
            transactions=[tx],
            name="improved_lnd",
            cfg=cfg,
            model=model,
            seed=42,
        )

        self.assertEqual(
            len(rows),
            1,
        )

        mock_env_class.assert_called_once()

        mock_env.reset.assert_called_once()

        mock_env.step.assert_called_once()

    # ========================================================
    # No Silent Fallback
    # ========================================================

    def test_no_silent_fallback_for_invalid_name(self):
        """
        Verify that an invalid method does not silently become
        native_lnd or another known baseline.
        """

        G = create_test_graph()

        tx = create_test_transaction()

        cfg = create_test_config()

        invalid_names = [
            "lnd",
            "native",
            "baseline",
            "improved",
            "foo",
        ]

        for name in invalid_names:

            with self.subTest(
                method=name
            ):

                with self.assertRaises(
                    ValueError
                ):

                    run_method(
                        G=G,
                        transactions=[tx],
                        name=name,
                        cfg=cfg,
                        seed=42,
                    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    unittest.main(
        verbosity=2
    )

