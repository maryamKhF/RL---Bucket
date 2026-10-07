# RL/test_environment.py

"""
Comprehensive validation suite for RL.environment.RoutingEnv.

Current design invariants
-------------------------
1. PPO controls eta only.
2. k is permanently fixed at 5.
3. reset() follows the Gymnasium API:
       observation, info
4. step() follows the Gymnasium API:
       observation, reward, terminated, truncated, info
5. current_bucket must be a real Bucket object.
6. Bucket.attempts counts actual payment attempts only.
7. _normalize_edge((u, v)) returns (u, v, None).
8. MultiDiGraph channel keys are preserved.
9. Unknown directional liquidity is not interpreted as zero.
10. A failed route may trigger partial backtracking and/or rerouting.
11. PPO actions must have exactly shape (1,).
12. ETA values outside the declared action space are rejected.
13. Unsupported candidate formats raise TypeError.
14. Bucket size is represented by len(bucket.candidates).
15. top_k_paths() returning None is an implementation error.
16. NetworkDynamics must be initialized before update.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np

from Bucket.bucket import Bucket
from RL import environment as environment_module
from RL.environment import RoutingEnv
from RL.state import FEATURE_DIM, FEATURE_NAMES, State


# ============================================================
# TEST HELPERS
# ============================================================

def make_graph():
    """
    Build a deterministic MultiDiGraph suitable for environment tests.
    """

    G = nx.MultiDiGraph()

    nodes = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]

    for node in nodes:
        G.add_node(
            node,
            available=True,
            is_online=True,
            carbon_intensity=100.0,
        )

    edges = [
        ("A", "B", 0),
        ("B", "C", 0),
        ("C", "D", 0),
        ("D", "E", 0),

        # Alternative branch
        ("B", "F", 0),
        ("F", "D", 0),
    ]

    for u, v, key in edges:
        G.add_edge(
            u,
            v,
            key=key,
            capacity=10000,
            balance_uv=10000,
            available=True,
        )

    return G


def make_transaction(
    tx_id="TX-001",
    source="A",
    destination="E",
    amount=1000.0,
):
    """
    Create the transaction object expected by RoutingEnv.
    """

    return SimpleNamespace(
        tx_id=tx_id,
        source=source,
        destination=destination,
        amount=float(amount),
    )


def make_transactions(count=1):
    """
    Create deterministic transaction sequence.
    """

    return [
        make_transaction(
            tx_id=f"TX-{index:03d}",
        )
        for index in range(1, count + 1)
    ]


def make_candidate(
    path=None,
    edges=None,
    cost=1.0,
):
    """
    Create a current-format candidate dictionary.
    """

    if path is None:
        path = [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    if edges is None:
        edges = [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0),
        ]

    return {
        "path": list(path),
        "edges": list(edges),
        "cost": float(cost),
        "total_fee": 10.0,
        "total_delay": 1.0,
        "reliability": 0.99,
        "failure_probability": 0.01,
        "hop_count": len(path) - 1,
        "eta": 0.5,
        "lambda_h": 1.0,
        "candidate": True,
        "success": None,
    }


def make_bucket(
    candidates=None,
    tx_id="TX-001",
):
    """
    Construct the actual Bucket implementation.

    RoutingEnv expects a real Bucket object.
    """

    if candidates is None:
        candidates = [
            make_candidate(),
        ]

    return Bucket(
        bucket_id=tx_id,
        transaction_id=tx_id,
        candidates=list(candidates),
    )


class FakePaymentSimulator:
    """
    Deterministic payment simulator for isolated environment tests.
    """

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def simulate_payment(
        self,
        path,
        edges,
        amount,
        tx_id,
    ):
        self.calls.append(
            {
                "path": list(path),
                "edges": list(edges),
                "amount": amount,
                "tx_id": tx_id,
            }
        )

        if self.results:
            return self.results.pop(0)

        return {
            "success": False,
            "fee": 0.0,
            "delay": 0.0,
            "carbon": 0.0,
            "reason": "no_fake_result",
        }


class FakeContinuingPaymentSimulator(FakePaymentSimulator):
    def __init__(self, results):
        super().__init__(results)
        self.continuations = []

    def continue_payment(
        self,
        path,
        edges,
        amount,
        tx_id,
        forwarded_prefix_length,
    ):
        self.continuations.append(
            {
                "path": list(path),
                "edges": list(edges),
                "amount": amount,
                "tx_id": tx_id,
                "forwarded_prefix_length": forwarded_prefix_length,
            }
        )
        if self.results:
            return self.results.pop(0)
        return {
            "success": False,
            "fee": 0.0,
            "delay": 0.0,
            "carbon": 0.0,
            "reason": "no_fake_result",
        }


class FakeNetworkDynamics:
    """
    Deterministic NetworkDynamics replacement.
    """

    def __init__(self):
        self.reset_calls = []
        self.update_calls = 0

    def reset(self, reset_balances=False):
        self.reset_calls.append(
            reset_balances
        )

    def update(self):
        self.update_calls += 1


class FakeFailureModel:
    """
    Minimal FailureModel replacement for environment lifecycle tests.
    """

    def __init__(self):
        self.reset_calls = []

    def reset_runtime_state(
        self,
        graph,
        reset_counters=False,
        reset_rng=False,
    ):
        self.reset_calls.append(
            {
                "graph": graph,
                "reset_counters": reset_counters,
                "reset_rng": reset_rng,
            }
        )


class FakeBacktracker:
    """
    Deterministic backtracker.
    """

    def __init__(
        self,
        network,
        result=None,
    ):
        self.network = network
        self.bucket = None

        self.result = result or {
            "success": False,
            "status": "no_alternative",
            "reason": "test",
            "new_route": None,
            "new_edges": None,
        }

        self.calls = []

    def backtrack(
        self,
        route,
        route_edges,
        failed_edge,
        failure_index,
        amount,
        bucket_id,
        attempt_id,
    ):
        self.calls.append(
            {
                "route": list(route),
                "route_edges": list(route_edges),
                "failed_edge": failed_edge,
                "failure_index": failure_index,
                "amount": amount,
                "bucket_id": bucket_id,
                "attempt_id": attempt_id,
            }
        )

        return dict(self.result)


# ============================================================
# TEST CASE
# ============================================================

class RoutingEnvironmentValidation(unittest.TestCase):

    # ========================================================
    # Environment Construction
    # ========================================================

    def setUp(self):
        self.G = make_graph()
        self.transactions = make_transactions(2)

        self.env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
        )

    def tearDown(self):
        try:
            self.env.close()
        except Exception:
            pass

    def test_01_environment_constructs(self):
        self.assertIsInstance(
            self.env,
            RoutingEnv,
        )

    def test_02_graph_is_stored(self):
        self.assertIs(
            self.env.G,
            self.G,
        )

    def test_03_transactions_are_stored(self):
        """
        RoutingEnv stores its own transaction collection.

        Therefore object identity is not required. The transaction
        sequence and its contents must remain equivalent.
        """

        self.assertEqual(
            self.env.transactions,
            self.transactions,
        )

    def test_04_top_k_is_fixed_to_five(self):
        self.assertEqual(
            self.env.top_k,
            5,
        )

    def test_05_eta_default_bounds(self):
        self.assertEqual(
            self.env.eta_min,
            -1.0,
        )

        self.assertEqual(
            self.env.eta_max,
            1.0,
        )

    def test_06_custom_eta_bounds(self):
        config = {
            "rl": {
                "eta_min": 0.2,
                "eta_max": 0.8,
            }
        }

        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
            config=config,
        )

        try:
            self.assertEqual(
                env.eta_min,
                0.2,
            )

            self.assertEqual(
                env.eta_max,
                0.8,
            )

            np.testing.assert_allclose(
                env.action_space.low,
                np.array(
                    [0.2],
                    dtype=np.float32,
                ),
            )

            np.testing.assert_allclose(
                env.action_space.high,
                np.array(
                    [0.8],
                    dtype=np.float32,
                ),
            )

        finally:
            env.close()

    def test_07_invalid_eta_bounds_are_rejected(self):
        config = {
            "rl": {
                "eta_min": 0.9,
                "eta_max": 0.1,
            }
        }

        with self.assertRaises(ValueError):
            RoutingEnv(
                G=self.G,
                transactions=self.transactions,
                config=config,
            )

    def test_08_none_graph_is_rejected(self):
        with self.assertRaises(ValueError):
            RoutingEnv(
                G=None,
                transactions=self.transactions,
            )

    def test_09_none_transactions_are_rejected(self):
        with self.assertRaises(ValueError):
            RoutingEnv(
                G=self.G,
                transactions=None,
            )

    def test_10_empty_transactions_are_rejected(self):
        with self.assertRaises(ValueError):
            RoutingEnv(
                G=self.G,
                transactions=[],
            )

    def test_11_empty_graph_is_not_rejected_by_current_contract(self):
        """
        Current RoutingEnv validates G=None but does not reject
        an empty NetworkX graph during construction.
        """

        empty_graph = nx.MultiDiGraph()

        env = RoutingEnv(
            G=empty_graph,
            transactions=self.transactions,
        )

        try:
            self.assertIs(
                env.G,
                empty_graph,
            )
        finally:
            env.close()

    # ========================================================
    # Spaces
    # ========================================================

    def test_12_action_space_is_box(self):
        self.assertIsInstance(
            self.env.action_space,
            environment_module.spaces.Box,
        )

    def test_13_action_space_has_one_dimension(self):
        self.assertEqual(
            self.env.action_space.shape,
            (1,),
        )

    def test_14_action_space_dtype_is_float32(self):
        self.assertEqual(
            self.env.action_space.dtype,
            np.dtype(np.float32),
        )

    def test_15_action_space_bounds_match_eta(self):
        self.assertEqual(
            float(self.env.action_space.low[0]),
            self.env.eta_min,
        )

        self.assertEqual(
            float(self.env.action_space.high[0]),
            self.env.eta_max,
        )

    def test_16_observation_space_is_box(self):
        self.assertIsInstance(
            self.env.observation_space,
            environment_module.spaces.Box,
        )

    def test_17_observation_space_is_one_dimensional(self):
        self.assertEqual(
            len(self.env.observation_space.shape),
            1,
        )

    # ========================================================
    # Reset
    # ========================================================

    def test_18_reset_returns_observation_and_info(self):
        observation, info = self.env.reset()

        self.assertIsInstance(
            observation,
            np.ndarray,
        )

        self.assertIsInstance(
            info,
            dict,
        )

    def test_19_reset_observation_matches_space(self):
        observation, _ = self.env.reset()

        self.assertEqual(
            observation.shape,
            self.env.observation_space.shape,
        )

    def test_20_reset_observation_dtype_is_float32(self):
        observation, _ = self.env.reset()

        self.assertEqual(
            observation.dtype,
            np.float32,
        )

    def test_21_reset_observation_is_finite(self):
        observation, _ = self.env.reset()

        self.assertTrue(
            np.all(
                np.isfinite(observation)
            )
        )

    def test_state_does_not_expose_simulator_failure_probability(self):
        self.assertNotIn("failure_probability", FEATURE_NAMES)
        self.assertNotIn("estimated_liquidity", FEATURE_NAMES)
        self.assertEqual(FEATURE_DIM, 11)

        tx = self.transactions[0]
        state_before = State(
            G=self.G,
            source=tx.source,
            destination=tx.destination,
            transaction=tx,
        ).vector()

        for _, _, _, data in self.G.edges(keys=True, data=True):
            data["failure_probability"] = 0.01
            data["simulator_failure_probability"] = 0.99

        state_after = State(
            G=self.G,
            source=tx.source,
            destination=tx.destination,
            transaction=tx,
        ).vector()

        self.assertEqual(state_before.shape, (180,))
        np.testing.assert_array_equal(state_after, state_before)

    def test_state_uses_interval_belief_not_graph_balance(self):
        tx = self.transactions[0]
        edge = next(iter(self.G.edges(keys=True)))
        beliefs = {edge: (200.0, 600.0)}
        before = State(
            G=self.G,
            source=tx.source,
            destination=tx.destination,
            liquidity_beliefs=beliefs,
        ).vector()
        self.G.edges[edge]["balance_uv"] = 999_999.0
        after = State(
            G=self.G,
            source=tx.source,
            destination=tx.destination,
            liquidity_beliefs=beliefs,
        ).vector()
        np.testing.assert_array_equal(before, after)

    def test_22_reset_sets_sampled_transaction(self):
        self.env.reset()

        self.assertIn(self.env.current_tx, self.transactions)

    def test_reset_can_select_requested_transaction_deterministically(self):
        _observation, info = self.env.reset(
            seed=123,
            options={"transaction_index": 1},
        )

        self.assertIs(self.env.current_tx, self.transactions[1])
        self.assertEqual(info["transaction_id"], self.transactions[1].tx_id)

    def test_reset_rejects_invalid_transaction_index(self):
        with self.assertRaises(ValueError):
            self.env.reset(options={"transaction_index": len(self.transactions)})

    def test_23_reset_clears_current_bucket(self):
        self.env.current_bucket = make_bucket()

        self.env.reset()

        self.assertIsNone(
            self.env.current_bucket
        )

        self.assertIsNone(
            self.env.bucket
        )

    def test_24_reset_clears_routing_history(self):
        self.env.current_paths = [
            make_candidate()
        ]

        self.env.last_failure_probability = 0.8
        self.env.last_average_delay = 5.0
        self.env.last_backtrack_count = 3

        self.env.reset()

        self.assertEqual(
            self.env.current_paths,
            [],
        )

        self.assertEqual(
            self.env.last_failure_probability,
            0.0,
        )

        self.assertEqual(
            self.env.last_average_delay,
            0.0,
        )

        self.assertEqual(
            self.env.last_backtrack_count,
            0,
        )

    def test_25_reset_calls_network_dynamics_reset(self):
        dynamics = FakeNetworkDynamics()

        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
            network_dynamics=dynamics,
        )

        try:
            env.reset()

            self.assertEqual(
                len(dynamics.reset_calls),
                1,
            )

        finally:
            env.close()

    def test_26_reset_calls_failure_model_reset(self):
        failure_model = FakeFailureModel()

        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
            failure_model=failure_model,
        )

        try:
            env.reset()

            self.assertEqual(
                len(failure_model.reset_calls),
                1,
            )

            self.assertIs(
                failure_model.reset_calls[0]["graph"],
                self.G,
            )

        finally:
            env.close()

    # ========================================================
    # ETA Decoder
    # ========================================================

    def test_27_decode_eta_scalar_is_rejected(self):
        """
        PPO actions must have exactly shape (1,).
        Scalar values are intentionally rejected.
        """

        with self.assertRaises(ValueError):
            self.env._decode_eta(0.5)

    def test_28_decode_eta_one_element_array(self):
        eta = self.env._decode_eta(
            np.array(
                [0.7],
                dtype=np.float32,
            )
        )

        self.assertAlmostEqual(
            eta,
            0.7,
        )

    def test_29_decode_eta_below_action_space_is_rejected(self):
        """
        Current contract rejects an eta value below eta_min.
        It does not silently clip the value.
        """

        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [-100.0],
                    dtype=np.float32,
                )
            )

    def test_30_decode_eta_above_action_space_is_rejected(self):
        """
        Current contract rejects an eta value above eta_max.
        It does not silently clip the value.
        """

        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [100.0],
                    dtype=np.float32,
                )
            )

    def test_31_decode_eta_rejects_empty_action(self):
        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [],
                    dtype=np.float32,
                )
            )

    def test_32_decode_eta_rejects_nan(self):
        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [np.nan],
                    dtype=np.float32,
                )
            )

    def test_33_decode_eta_rejects_positive_inf(self):
        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [np.inf],
                    dtype=np.float32,
                )
            )

    def test_34_decode_eta_rejects_negative_inf(self):
        with self.assertRaises(ValueError):
            self.env._decode_eta(
                np.array(
                    [-np.inf],
                    dtype=np.float32,
                )
            )

    # ========================================================
    # Edge Normalization
    # ========================================================

    def test_35_normalize_none_edge(self):
        self.assertIsNone(
            self.env._normalize_edge(None)
        )

    def test_36_normalize_two_element_edge(self):
        result = self.env._normalize_edge(
            ("A", "B")
        )

        self.assertEqual(
            result,
            ("A", "B", None),
        )

    def test_37_normalize_three_element_edge(self):
        result = self.env._normalize_edge(
            ("A", "B", 7)
        )

        self.assertEqual(
            result,
            ("A", "B", 7),
        )

    def test_38_normalize_long_edge_is_rejected(self):
        """
        The current normalization contract accepts only:
            (u, v)
            (u, v, key)

        Longer tuples are invalid.
        """

        result = self.env._normalize_edge(
            ("A", "B", 7, "extra")
        )

        self.assertIsNone(
            result
        )

    def test_39_normalize_invalid_short_edge(self):
        self.assertIsNone(
            self.env._normalize_edge(
                ("A",)
            )
        )

    def test_40_normalize_non_iterable_edge(self):
        self.assertIsNone(
            self.env._normalize_edge(
                123
            )
        )

    # ========================================================
    # Candidate Extraction
    # ========================================================

    def test_41_candidate_route_from_dict(self):
        candidate = make_candidate()

        path, edges = self.env._candidate_route(
            candidate
        )

        self.assertEqual(
            path,
            candidate["path"],
        )

        self.assertEqual(
            edges,
            candidate["edges"],
        )

    def test_42_candidate_route_from_legacy_tuple(self):
        candidate = (
            ["A", "B", "C"],
            [
                ("A", "B", 0),
                ("B", "C", 0),
            ],
            1.0,
        )

        path, edges = self.env._candidate_route(
            candidate
        )

        self.assertEqual(
            path,
            candidate[0],
        )

        self.assertEqual(
            edges,
            candidate[1],
        )

    def test_43_invalid_candidate_route_is_rejected(self):
        """
        Unsupported candidate formats are programming errors
        and therefore raise TypeError.
        """

        with self.assertRaises(TypeError):
            self.env._candidate_route(
                object()
            )

    # ========================================================
    # Path -> Exact Edges
    # ========================================================

    def test_44_path_to_edges_preserves_multigraph_key(self):
        path = [
            "A",
            "B",
            "C",
        ]

        edges = self.env._path_to_edges(
            path
        )

        self.assertEqual(
            edges,
            [
                ("A", "B", 0),
                ("B", "C", 0),
            ],
        )

    def test_45_path_to_edges_returns_none_for_missing_edge(self):
        result = self.env._path_to_edges(
            [
                "A",
                "E",
            ]
        )

        self.assertIsNone(
            result
        )

    def test_46_path_to_edges_returns_none_for_short_path(self):
        self.assertIsNone(
            self.env._path_to_edges(
                ["A"]
            )
        )

    def test_47_path_to_edges_returns_none_for_empty_path(self):
        self.assertIsNone(
            self.env._path_to_edges(
                []
            )
        )

    def test_48_path_to_edges_skips_unavailable_multichannel(self):
        G = nx.MultiDiGraph()

        G.add_node(
            "A",
            available=True,
            is_online=True,
        )

        G.add_node(
            "B",
            available=True,
            is_online=True,
        )

        G.add_edge(
            "A",
            "B",
            key=0,
            available=False,
        )

        G.add_edge(
            "A",
            "B",
            key=1,
            available=True,
        )

        env = RoutingEnv(
            G=G,
            transactions=self.transactions,
        )

        try:
            result = env._path_to_edges(
                ["A", "B"]
            )

            self.assertEqual(
                result,
                [
                    ("A", "B", 1)
                ],
            )

        finally:
            env.close()

    # ========================================================
    # Bucket
    # ========================================================

    def test_49_bucket_none_size_is_zero(self):
        self.assertEqual(
            self.env._bucket_size(None),
            0,
        )

    def test_50_real_bucket_size(self):
        bucket = make_bucket(
            candidates=[
                make_candidate(),
                make_candidate(
                    path=[
                        "A",
                        "B",
                        "F",
                        "D",
                        "E",
                    ],
                    edges=[
                        ("A", "B", 0),
                        ("B", "F", 0),
                        ("F", "D", 0),
                        ("D", "E", 0),
                    ],
                    cost=2.0,
                ),
            ]
        )

        self.assertEqual(
            self.env._bucket_size(bucket),
            2,
        )

    def test_51_bucket_info_contains_size(self):
        bucket = make_bucket(
            candidates=[
                make_candidate(),
                make_candidate(
                    cost=2.0
                ),
            ]
        )

        info = self.env._bucket_info(
            bucket
        )

        self.assertIsInstance(
            info,
            dict,
        )

        self.assertEqual(
            info["size"],
            2,
        )

    def test_52_bucket_info_none_is_empty(self):
        self.assertEqual(
            self.env._bucket_info(None),
            {},
        )

    def test_53_bucket_current_is_real_candidate(self):
        candidate = make_candidate()

        bucket = make_bucket(
            candidates=[
                candidate
            ]
        )

        self.assertIs(
            bucket.current(),
            candidate,
        )

    # ========================================================
    # Bucket Creation
    # ========================================================

    def test_54_create_bucket_returns_bucket(self):
        tx = self.transactions[0]

        candidates = [
            make_candidate(
                cost=3.0
            ),
            make_candidate(
                cost=1.0
            ),
            make_candidate(
                cost=2.0
            ),
        ]

        bucket, filtered = self.env._create_bucket(
            tx=tx,
            candidates=candidates,
        )

        self.assertIsInstance(
            bucket,
            Bucket,
        )

        self.assertEqual(
            bucket.transaction_id,
            tx.tx_id,
        )

        # Bucket does not expose size().
        # Its candidate count is the authoritative size.
        self.assertLessEqual(
            len(bucket.candidates),
            5,
        )

        self.assertEqual(
            len(bucket.candidates),
            len(filtered),
        )

    def test_55_create_bucket_with_no_candidates(self):
        bucket, candidates = self.env._create_bucket(
            tx=self.transactions[0],
            candidates=[],
        )

        self.assertIsNone(
            bucket
        )

        self.assertEqual(
            candidates,
            [],
        )

    # ========================================================
    # Top-K Generation
    # ========================================================

    def test_56_generate_candidates_passes_fixed_k_five(self):
        tx = self.transactions[0]

        captured = {}

        def fake_top_k_paths(**kwargs):
            captured.update(kwargs)

            return [
                make_candidate()
            ]

        with patch(
            "RL.environment.top_k_paths",
            side_effect=fake_top_k_paths,
        ):
            result = self.env._generate_candidates(
                tx=tx,
                eta=0.37,
            )

        self.assertEqual(
            captured["k"],
            5,
        )

        self.assertEqual(
            captured["eta"],
            0.37,
        )

        self.assertEqual(
            captured["source"],
            tx.source,
        )

        self.assertEqual(
            captured["target"],
            tx.destination,
        )

        self.assertEqual(
            captured["amount"],
            tx.amount,
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_57_generate_candidates_none_is_rejected(self):
        """
        None from top_k_paths() is different from an empty result.

        [] means no candidate route was found.

        None indicates an implementation/API failure and is therefore
        converted by RoutingEnv into RuntimeError.
        """

        tx = self.transactions[0]

        with patch(
            "RL.environment.top_k_paths",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                self.env._generate_candidates(
                    tx=tx,
                    eta=0.5,
                )

    # ========================================================
    # Result Normalization
    # ========================================================

    def test_58_result_to_dict_accepts_dict(self):
        result = {
            "success": True,
            "fee": 10.0,
        }

        normalized = self.env._result_to_dict(
            result
        )

        self.assertIs(
            normalized,
            result,
        )

    def test_59_result_to_dict_accepts_to_dict_object(self):
        class Result:
            def to_dict(self):
                return {
                    "success": True,
                    "fee": 10.0,
                }

        normalized = self.env._result_to_dict(
            Result()
        )

        self.assertEqual(
            normalized["success"],
            True,
        )

    def test_60_result_to_dict_rejects_invalid_result(self):
        with self.assertRaises(TypeError):
            self.env._result_to_dict(
                object()
            )

    # ========================================================
    # State
    # ========================================================

    def test_61_build_state_matches_observation_space(self):
        self.env.reset()

        state = self.env._build_state()

        self.assertEqual(
            state.shape,
            self.env.observation_space.shape,
        )

    def test_62_build_state_is_float32(self):
        self.env.reset()

        state = self.env._build_state()

        self.assertEqual(
            state.dtype,
            np.float32,
        )

    def test_63_build_state_is_finite(self):
        self.env.reset()

        state = self.env._build_state()

        self.assertTrue(
            np.all(
                np.isfinite(state)
            )
        )

    def test_64_empty_state_is_zero_vector(self):
        self.env.current_tx = None

        state = self.env._build_state()

        np.testing.assert_array_equal(
            state,
            np.zeros(
                self.env.observation_space.shape,
                dtype=np.float32,
            ),
        )

    # ========================================================
    # Full Reroute Filtering
    # ========================================================

    def test_65_full_reroute_rejects_failed_route(self):
        tx = self.transactions[0]

        candidate_1 = make_candidate(
            path=[
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
            edges=[
                ("A", "B", 0),
                ("B", "C", 0),
                ("C", "D", 0),
                ("D", "E", 0),
            ],
        )

        candidate_2 = make_candidate(
            path=[
                "A",
                "B",
                "F",
                "D",
                "E",
            ],
            edges=[
                ("A", "B", 0),
                ("B", "F", 0),
                ("F", "D", 0),
                ("D", "E", 0),
            ],
            cost=2.0,
        )

        with patch(
            "RL.environment.top_k_paths",
            return_value=[
                candidate_1,
                candidate_2,
            ],
        ):
            bucket, candidates = self.env._full_reroute(
                tx=tx,
                eta=0.5,
                failed_routes={
                    tuple(candidate_1["path"])
                },
                failed_edges=set(),
            )

        self.assertIsInstance(
            bucket,
            Bucket,
        )

        self.assertTrue(
            all(
                tuple(c["path"]) !=
                tuple(candidate_1["path"])
                for c in candidates
            )
        )

    def test_66_full_reroute_rejects_failed_exact_channel(self):
        tx = self.transactions[0]

        candidate_1 = make_candidate(
            path=[
                "A",
                "B",
                "C",
                "D",
                "E",
            ],
            edges=[
                ("A", "B", 0),
                ("B", "C", 0),
                ("C", "D", 0),
                ("D", "E", 0),
            ],
        )

        candidate_2 = make_candidate(
            path=[
                "A",
                "B",
                "F",
                "D",
                "E",
            ],
            edges=[
                ("A", "B", 0),
                ("B", "F", 0),
                ("F", "D", 0),
                ("D", "E", 0),
            ],
            cost=2.0,
        )

        with patch(
            "RL.environment.top_k_paths",
            return_value=[
                candidate_1,
                candidate_2,
            ],
        ):
            bucket, candidates = self.env._full_reroute(
                tx=tx,
                eta=0.5,
                failed_routes=set(),
                failed_edges={
                    ("B", "C", 0)
                },
            )

        self.assertIsInstance(
            bucket,
            Bucket,
        )

        self.assertTrue(
            all(
                ("B", "C", 0)
                not in [
                    self.env._normalize_edge(e)
                    for e in c.get("edges", [])
                ]
                for c in candidates
            )
        )

    # ========================================================
    # Reward
    # ========================================================

    def test_67_reward_is_numeric(self):
        reward = self.env._calculate_environment_reward(
            success=True,
            path_length=4,
            carbon_intensity=1.0,
            fee=10.0,
            delay=2.0,
            partial_backtrack_count=0,
            full_reroute_count=0,
            attempt_count=1,
        )

        self.assertIsInstance(
            reward,
            float,
        )

    def test_68_success_reward_is_higher_than_failure_reward(self):
        success_reward = (
            self.env._calculate_environment_reward(
                success=True,
                path_length=4,
                carbon_intensity=1.0,
                fee=10.0,
                delay=2.0,
                partial_backtrack_count=0,
                full_reroute_count=0,
                attempt_count=1,
            )
        )

        failure_reward = (
            self.env._calculate_environment_reward(
                success=False,
                path_length=0,
                carbon_intensity=0.0,
                fee=0.0,
                delay=0.0,
                partial_backtrack_count=0,
                full_reroute_count=0,
                attempt_count=1,
            )
        )

        self.assertGreater(
            success_reward,
            failure_reward,
        )

    # ========================================================
    # Step Preconditions
    # ========================================================

    def test_69_step_requires_reset(self):
        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
        )

        try:
            with self.assertRaises(RuntimeError):
                env.step(
                    np.array(
                        [0.5],
                        dtype=np.float32,
                    )
                )

        finally:
            env.close()

    # ========================================================
    # Step: No Candidate
    # ========================================================

    def test_70_step_handles_no_initial_candidate(self):
        env = RoutingEnv(
            G=self.G,
            transactions=[
                make_transaction()
            ],
        )

        dynamics = FakeNetworkDynamics()
        env.network_dynamics = dynamics

        try:
            env.reset()

            with patch(
                "RL.environment.top_k_paths",
                return_value=[],
            ):
                observation, reward, terminated, truncated, info = (
                    env.step(
                        np.array(
                            [0.5],
                            dtype=np.float32,
                        )
                    )
                )

            self.assertIsInstance(
                observation,
                np.ndarray,
            )

            self.assertIsInstance(
                reward,
                float,
            )

            self.assertIs(
                terminated,
                True,
            )

            self.assertIs(
                truncated,
                False,
            )

            self.assertFalse(
                info["success"]
            )

            self.assertEqual(
                info["candidate_path_count"],
                0,
            )

            self.assertGreaterEqual(
                dynamics.update_calls,
                1,
            )

        finally:
            env.close()

    # ========================================================
    # Step: Successful Payment
    # ========================================================

    def test_71_step_success_path(self):
        env = RoutingEnv(
            G=self.G,
            transactions=[
                make_transaction()
            ],
        )

        candidate = make_candidate()

        payment = FakePaymentSimulator(
            [
                {
                    "success": True,
                    "fee": 10.0,
                    "delay": 1.0,
                    "carbon": 0.5,
                    "reason": "success",
                }
            ]
        )

        try:
            env.payment_simulator = payment

            env.reset()

            with patch(
                "RL.environment.top_k_paths",
                return_value=[
                    candidate
                ],
            ):
                observation, reward, terminated, truncated, info = (
                    env.step(
                        np.array(
                            [0.5],
                            dtype=np.float32,
                        )
                    )
                )

            self.assertIsInstance(
                observation,
                np.ndarray,
            )

            self.assertIsInstance(
                reward,
                float,
            )

            self.assertTrue(
                terminated
            )

            self.assertFalse(
                truncated
            )

            self.assertTrue(
                info["success"]
            )

            self.assertTrue(
                info["payment_success"]
            )

            self.assertEqual(
                info["successful_bucket_rank"],
                1,
            )

            self.assertEqual(
                info["successful_route_source"],
                "bucket_candidate",
            )

            self.assertEqual(
                info["attempt_count"],
                1,
            )

            self.assertEqual(
                len(payment.calls),
                1,
            )

        finally:
            env.close()

    # ========================================================
    # Step: Bucket Must Be Real Object
    # ========================================================

    def test_72_execute_bucket_pipeline_uses_real_bucket(self):
        """
        Regression test ensuring that current_bucket remains a real
        Bucket object throughout the execution pipeline.

        RoutingEnv._execute_bucket_pipeline requires Bucket behavior,
        including:
            current()
            backtrack()
            mark_success()
        """

        env = RoutingEnv(
            G=self.G,
            transactions=[
                make_transaction()
            ],
        )

        candidate = make_candidate()

        env.current_tx = env.transactions[0]
        env.current_paths = [
            candidate
        ]

        env.current_bucket = make_bucket(
            [candidate]
        )

        env.bucket = env.current_bucket

        payment = FakePaymentSimulator(
            [
                {
                    "success": True,
                    "fee": 10.0,
                    "delay": 1.0,
                    "carbon": 0.5,
                    "reason": "success",
                }
            ]
        )

        env.payment_simulator = payment

        try:
            result = env._execute_bucket_pipeline(
                tx=env.current_tx,
                eta=0.5,
            )

            self.assertTrue(
                result["success"]
            )

            self.assertEqual(
                result["attempt_count"],
                1,
            )

            self.assertIsInstance(
                env.current_bucket,
                Bucket,
            )

        finally:
            env.close()

    # ========================================================
    # Step: Partial Backtracking
    # ========================================================

    def test_73_partial_backtracking_path_is_used(self):
        env = RoutingEnv(
            G=self.G,
            transactions=[
                make_transaction()
            ],
        )

        candidate = make_candidate()

        alternative_path = [
            "A",
            "B",
            "F",
            "D",
            "E",
        ]

        alternative_edges = [
            ("A", "B", 0),
            ("B", "F", 0),
            ("F", "D", 0),
            ("D", "E", 0),
        ]

        env.current_tx = env.transactions[0]
        env.current_paths = [
            candidate
        ]

        env.current_bucket = make_bucket(
            [candidate]
        )

        env.bucket = env.current_bucket

        env.backtracker = FakeBacktracker(
            network=self.G,
            result={
                "success": True,
                "status": "alternative_found",
                "reason": "partial_backtrack",
                "new_route": alternative_path,
                "new_edges": alternative_edges,
                "branch_index": 1,
            },
        )

        env.payment_simulator = FakeContinuingPaymentSimulator(
            [
                {
                    "success": False,
                    "fee": 0.0,
                    "delay": 0.0,
                    "carbon": 0.0,
                    "reason": "channel_failure",
                    "failed_edge": ("B", "C", 0),
                    "failure_index": 1,
                },
                {
                    "success": True,
                    "fee": 8.0,
                    "delay": 1.0,
                    "carbon": 0.4,
                    "reason": "success",
                },
            ]
        )

        try:
            result = env._execute_bucket_pipeline(
                tx=env.current_tx,
                eta=0.5,
            )

            self.assertTrue(
                result["success"]
            )

            self.assertEqual(
                result["partial_backtrack_count"],
                1,
            )

            self.assertEqual(
                result["partial_backtrack_success"],
                1,
            )

            self.assertEqual(
                result["path"],
                alternative_path,
            )

            self.assertIsNone(
                result["successful_bucket_rank"]
            )

            self.assertEqual(
                result["successful_route_source"],
                "partial_backtrack",
            )

            self.assertEqual(len(env.payment_simulator.calls), 1)
            self.assertEqual(len(env.payment_simulator.continuations), 1)
            self.assertEqual(
                env.payment_simulator.continuations[0][
                    "forwarded_prefix_length"
                ],
                1,
            )
            self.assertEqual(
                env.payment_simulator.continuations[0]["path"],
                alternative_path,
            )

            self.assertEqual(
                len(env.backtracker.calls),
                1,
            )

        finally:
            env.close()

    # ========================================================
    # Network Dynamics
    # ========================================================

    def test_74_update_network_dynamics_calls_update(self):
        dynamics = FakeNetworkDynamics()

        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
            network_dynamics=dynamics,
        )

        try:
            env._update_network_dynamics()

            self.assertEqual(
                dynamics.update_calls,
                1,
            )

        finally:
            env.close()

    def test_75_update_network_dynamics_without_initializer_is_rejected(self):
        """
        NetworkDynamics=None is not silently ignored.

        The current RoutingEnv contract explicitly raises RuntimeError
        when an update is requested without an initialized dynamics
        component.
        """

        env = RoutingEnv(
            G=self.G,
            transactions=self.transactions,
            network_dynamics=None,
        )

        try:
            env.network_dynamics = None

            with self.assertRaises(RuntimeError):
                env._update_network_dynamics()

        finally:
            env.close()

    # ========================================================
    # Transaction Lifecycle
    # ========================================================

    def test_76_get_current_transaction(self):
        self.env.tx_index = 0

        result = self.env._get_current_transaction()

        self.assertIs(
            result,
            self.transactions[0],
        )

    def test_77_get_current_transaction_after_end(self):
        self.env.tx_index = len(
            self.transactions
        )

        result = self.env._get_current_transaction()

        self.assertIsNone(
            result
        )

    def test_78_transaction_id(self):
        self.env.current_tx = self.transactions[0]

        self.assertEqual(
            self.env._transaction_id(),
            self.transactions[0].tx_id,
        )

    def test_79_transaction_id_without_transaction(self):
        self.env.current_tx = None

        self.assertIsNone(
            self.env._transaction_id()
        )

    def test_80_advance_transaction_ends_one_payment_episode(self):
        self.env.reset()

        terminated = self.env._advance_transaction()

        self.assertTrue(
            terminated
        )

        self.assertIsNone(
            self.env.current_tx,
        )

    def test_81_advance_transaction_terminates_at_end(self):
        self.env.reset()

        self.env.tx_index = 1
        self.env.current_tx = self.transactions[1]

        terminated = self.env._advance_transaction()

        self.assertTrue(
            terminated
        )

        self.assertIsNone(
            self.env.current_tx
        )

        self.assertEqual(
            self.env.current_paths,
            [],
        )

        self.assertIsNone(
            self.env.current_bucket
        )

    # ========================================================
    # PPO Invariants
    # ========================================================

    def test_82_k_never_becomes_action_dimension(self):
        self.assertEqual(
            self.env.action_space.shape,
            (1,),
        )

        self.assertEqual(
            self.env.top_k,
            5,
        )

    def test_83_eta_is_only_routing_action(self):
        captured = {}

        tx = self.transactions[0]

        def fake_top_k_paths(**kwargs):
            captured.update(kwargs)
            return []

        with patch(
            "RL.environment.top_k_paths",
            side_effect=fake_top_k_paths,
        ):
            self.env._generate_candidates(
                tx=tx,
                eta=0.73,
            )

        self.assertEqual(
            captured["eta"],
            0.73,
        )

        self.assertEqual(
            captured["k"],
            5,
        )

    # ========================================================
    # Gymnasium API
    # ========================================================

    def test_84_reset_is_gymnasium_compatible(self):
        result = self.env.reset()

        self.assertIsInstance(
            result,
            tuple,
        )

        self.assertEqual(
            len(result),
            2,
        )

    def test_85_step_is_gymnasium_compatible(self):
        env = RoutingEnv(
            G=self.G,
            transactions=[
                make_transaction()
            ],
        )

        try:
            env.reset()

            with patch(
                "RL.environment.top_k_paths",
                return_value=[],
            ):
                result = env.step(
                    np.array(
                        [0.5],
                        dtype=np.float32,
                    )
                )

            self.assertIsInstance(
                result,
                tuple,
            )

            self.assertEqual(
                len(result),
                5,
            )

            observation, reward, terminated, truncated, info = result

            self.assertIsInstance(
                observation,
                np.ndarray,
            )

            self.assertIsInstance(
                reward,
                float,
            )

            self.assertIsInstance(
                terminated,
                bool,
            )

            self.assertIsInstance(
                truncated,
                bool,
            )

            self.assertIsInstance(
                info,
                dict,
            )

        finally:
            env.close()

    # ========================================================
    # Render / Close
    # ========================================================

    def test_86_render_does_not_fail(self):
        result = self.env.render()

        self.assertIsNone(
            result
        )

    def test_87_close_does_not_fail(self):
        result = self.env.close()

        self.assertIsNone(
            result
        )


# ============================================================
# TEST RUNNER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ROUTING ENVIRONMENT VALIDATION")
    print("=" * 70)

    unittest.main(
        verbosity=2
    )
