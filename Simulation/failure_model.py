"""
failure_model.py

Lightning Network failure simulation model.

Responsibilities
----------------
1. Assign static failure probabilities to channels.
2. Evaluate node/channel/liquidity failures during payment execution.
3. Return the exact failed element and failure position.
4. Provide failure information required by Partial Backtracking.
5. Reset temporary runtime failure state between episodes.

Design
------
The NetworkX graph G is the single source of truth for network state.

The model is intentionally separated from:
    - routing
    - Bucket
    - Partial Backtracking
    - PPO
    - Onion routing
"""

from datetime import datetime
import random

import numpy as np


# ============================================================
# Static Failure Probability Assignment
# ============================================================

def assign_failure_probabilities(
    G,
    average_rate,
    seed=42
):
    """
    Assign channel-specific failure probabilities.

    Also initializes runtime channel state.

    Static attributes:
        failure_probability

    Runtime attributes:
        available
        failure_count
        last_failure
    """

    if average_rate < 0:
        raise ValueError(
            "average_rate must be non-negative."
        )

    if average_rate > 1:
        raise ValueError(
            "average_rate must not be greater than 1."
        )

    rng = np.random.default_rng(seed)

    if G.is_multigraph():

        for u, v, key, data in G.edges(
            keys=True,
            data=True
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G,
                u,
                v,
                average_rate,
                rng
            )

            data["failure_probability"] = probability

            # Runtime state
            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    else:

        for u, v, data in G.edges(
            data=True
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G,
                u,
                v,
                average_rate,
                rng
            )

            data["failure_probability"] = probability

            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None


# ============================================================
# Failure Probability Calculation
# ============================================================

def _calculate_failure_probability(
    G,
    u,
    v,
    average_rate,
    rng
):
    """
    Calculate failure probability using topology/geography.
    """

    country_u = G.nodes[u].get(
        "country",
        ""
    )

    country_v = G.nodes[v].get(
        "country",
        ""
    )

    if country_u == country_v:

        factor = 0.7

    else:

        if _intercontinental(
            G,
            u,
            v
        ):

            factor = 2.0

        else:

            factor = 1.0

    probability = (
        average_rate
        *
        factor
        *
        rng.uniform(
            0.8,
            1.2
        )
    )

    return float(
        min(
            0.95,
            probability
        )
    )


# ============================================================
# Geographic Helper
# ============================================================

def _intercontinental(
    G,
    u,
    v
):
    """
    Detect a long-distance/intercontinental connection.

    This is a simulation abstraction and does not represent
    actual geographical routing information.
    """

    node_u = G.nodes[u]
    node_v = G.nodes[v]

    longitude_u = node_u.get(
        "longitude",
        0
    )

    longitude_v = node_v.get(
        "longitude",
        0
    )

    latitude_u = node_u.get(
        "latitude",
        0
    )

    latitude_v = node_v.get(
        "latitude",
        0
    )

    try:

        longitude_distance = abs(
            float(longitude_u)
            -
            float(longitude_v)
        )

        latitude_distance = abs(
            float(latitude_u)
            -
            float(latitude_v)
        )

    except (
        TypeError,
        ValueError
    ):

        return False

    return (
        longitude_distance > 45
        and
        latitude_distance > 10
    )


# ============================================================
# Failure Model
# ============================================================

class FailureModel:
    """
    Runtime failure evaluator for Lightning payment simulation.

    Failure categories
    ------------------
    1. node_failure
    2. channel_failure
    3. liquidity_failure
    4. missing_channel

    Temporary channel failures are stored in the graph and can
    be reset between PPO episodes.
    """

    def __init__(
        self,
        node_failure_probability=0.01,
        liquidity_failure_probability=0.05,
        seed=42
    ):

        if not 0 <= node_failure_probability <= 1:

            raise ValueError(
                "node_failure_probability must be in [0, 1]."
            )

        if not 0 <= liquidity_failure_probability <= 1:

            raise ValueError(
                "liquidity_failure_probability must be in [0, 1]."
            )

        self.node_failure_probability = float(
            node_failure_probability
        )

        self.liquidity_failure_probability = float(
            liquidity_failure_probability
        )

        self.seed = int(seed)

        self.rng = random.Random(
            self.seed
        )

    # ========================================================
    # Reset Runtime State
    # ========================================================

    def reset_runtime_state(
        self,
        G,
        reset_counters=False,
        reset_rng=False
    ):
        """
        Reset temporary runtime failure state.

        Parameters
        ----------
        G : networkx graph
            Network graph.

        reset_counters : bool
            If True, failure counters are reset to zero.

        reset_rng : bool
            If True, RNG is re-seeded.

        Notes
        -----
        Static failure_probability values are preserved.

        This method is intended to be called by RoutingEnv.reset().
        """

        if G is None:

            raise ValueError(
                "G cannot be None."
            )

        # ----------------------------------------------------
        # Reset channel runtime state
        # ----------------------------------------------------

        if G.is_multigraph():

            for _, _, _, data in G.edges(
                keys=True,
                data=True
            ):

                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        else:

            for _, _, data in G.edges(
                data=True
            ):

                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        # ----------------------------------------------------
        # Reset node runtime state if present
        # ----------------------------------------------------

        for _, data in G.nodes(
            data=True
        ):

            if "available" in data:

                data["available"] = True

            if "is_online" in data:

                data["is_online"] = True

        # ----------------------------------------------------
        # Optional RNG reset
        # ----------------------------------------------------

        if reset_rng:

            self.rng.seed(
                self.seed
            )

    # ========================================================
    # Node Failure
    # ========================================================

    def check_node_failure(
        self,
        G,
        node
    ):

        if node not in G.nodes:

            return True

        data = G.nodes[node]

        if data.get(
            "available",
            True
        ) is False:

            return True

        if data.get(
            "is_online",
            True
        ) is False:

            return True

        return (
            self.rng.random()
            <
            self.node_failure_probability
        )

    # ========================================================
    # Channel Failure
    # ========================================================

    def check_channel_failure(
        self,
        G,
        u,
        v,
        key=None
    ):

        edge = self._get_edge(
            G,
            u,
            v,
            key
        )

        if edge is None:

            return True

        if edge.get(
            "available",
            True
        ) is False:

            return True

        probability = float(
            edge.get(
                "failure_probability",
                0.01
            )
        )

        failed = (
            self.rng.random()
            <
            probability
        )

        if failed:

            edge["failure_count"] = (
                edge.get(
                    "failure_count",
                    0
                )
                + 1
            )

            edge["last_failure"] = datetime.now()

            # Temporary runtime failure.
            edge["available"] = False

        return failed

    # ========================================================
    # Liquidity Failure
    # ========================================================

    def check_liquidity_failure(
        self,
        G,
        u,
        v,
        amount,
        key=None
    ):

        edge = self._get_edge(
            G,
            u,
            v,
            key
        )

        if edge is None:

            return True

        capacity = edge.get(
            "capacity",
            None
        )

        if capacity is not None:

            try:

                if amount > float(
                    capacity
                ):

                    return True

            except (
                TypeError,
                ValueError
            ):

                pass

        balance_uv = edge.get(
            "balance_uv",
            None
        )

        if balance_uv is not None:

            try:

                if amount > float(
                    balance_uv
                ):

                    return True

            except (
                TypeError,
                ValueError
            ):

                pass

        liquidity = edge.get(
            "liquidity",
            None
        )

        if liquidity is not None:

            try:

                if amount > float(
                    liquidity
                ):

                    return True

            except (
                TypeError,
                ValueError
            ):

                pass

        return (
            self.rng.random()
            <
            self.liquidity_failure_probability
        )

    # ========================================================
    # Single Edge Evaluation
    # ========================================================

    def evaluate_edge(
        self,
        G,
        u,
        v,
        amount,
        key=None
    ):

        edge = self._get_edge(
            G,
            u,
            v,
            key
        )

        if edge is None:

            return {
                "success": False,
                "reason": "missing_channel",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key
                ),
            }

        if edge.get(
            "available",
            True
        ) is False:

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key
                ),
            }

        if self.check_node_failure(
            G,
            u
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": u,
                "failed_edge": None,
            }

        if self.check_node_failure(
            G,
            v
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": v,
                "failed_edge": None,
            }

        if self.check_channel_failure(
            G,
            u,
            v,
            key
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key
                ),
            }

        if self.check_liquidity_failure(
            G,
            u,
            v,
            amount,
            key
        ):

            return {
                "success": False,
                "reason": "liquidity_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key
                ),
            }

        return {
            "success": True,
            "reason": None,
            "failed_node": None,
            "failed_edge": None,
        }

    # ========================================================
    # Full Route Evaluation
    # ========================================================

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None
    ):

        if network is None:

            raise ValueError(
                "network cannot be None."
            )

        if route is None or len(route) < 2:

            return {
                "success": False,
                "reason": "invalid_route",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": None,
                "visited_edges": [],
            }

        if amount <= 0:

            return {
                "success": False,
                "reason": "invalid_amount",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": None,
                "visited_edges": [],
            }

        visited_edges = []

        if route_edges is None:

            route_edges = self._resolve_route_edges(
                network,
                route
            )

        if route_edges is None:

            return {
                "success": False,
                "reason": "missing_channel",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": 0,
                "visited_edges": [],
            }

        if len(route_edges) != len(route) - 1:

            return {
                "success": False,
                "reason": "invalid_route_edges",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": 0,
                "visited_edges": [],
            }

        for index, edge_info in enumerate(
            route_edges
        ):

            parsed = self._parse_edge(
                edge_info
            )

            if parsed is None:

                return {
                    "success": False,
                    "reason": "invalid_edge",
                    "failed_node": None,
                    "failed_edge": edge_info,
                    "failure_index": index,
                    "visited_edges": visited_edges,
                }

            u, v, key = parsed

            result = self.evaluate_edge(
                network,
                u,
                v,
                amount,
                key
            )

            if not result["success"]:

                return {
                    "success": False,
                    "reason": result["reason"],
                    "failed_node": result["failed_node"],
                    "failed_edge": result["failed_edge"],
                    "failure_index": index,
                    "visited_edges": visited_edges,
                }

            visited_edges.append(
                (
                    u,
                    v,
                    key
                )
            )

        return {
            "success": True,
            "reason": None,
            "failed_node": None,
            "failed_edge": None,
            "failure_index": None,
            "visited_edges": visited_edges,
        }

    # ========================================================
    # Route Edge Resolution
    # ========================================================

    def _resolve_route_edges(
        self,
        G,
        route
    ):

        edges = []

        for i in range(
            len(route) - 1
        ):

            u = route[i]
            v = route[i + 1]

            if not G.has_edge(
                u,
                v
            ):

                return None

            if G.is_multigraph():

                edge_data = G.get_edge_data(
                    u,
                    v
                )

                if not edge_data:

                    return None

                selected_key = None

                for key, data in edge_data.items():

                    if data.get(
                        "available",
                        True
                    ):

                        selected_key = key
                        break

                if selected_key is None:

                    return None

                edges.append(
                    (
                        u,
                        v,
                        selected_key
                    )
                )

            else:

                edges.append(
                    (
                        u,
                        v,
                        None
                    )
                )

        return edges

    # ========================================================
    # Edge Parser
    # ========================================================

    @staticmethod
    def _parse_edge(
        edge_info
    ):

        if edge_info is None:

            return None

        try:

            length = len(
                edge_info
            )

        except TypeError:

            return None

        if length == 2:

            u, v = edge_info

            return (
                u,
                v,
                None
            )

        if length == 3:

            u, v, key = edge_info

            return (
                u,
                v,
                key
            )

        return None

    # ========================================================
    # Graph Edge Access
    # ========================================================

    @staticmethod
    def _get_edge(
        G,
        u,
        v,
        key=None
    ):

        if not G.has_edge(
            u,
            v
        ):

            return None

        if G.is_multigraph():

            if key is None:

                edge_data = G.get_edge_data(
                    u,
                    v
                )

                if not edge_data:

                    return None

                # Fallback to first available edge.
                for _, data in edge_data.items():

                    if data.get(
                        "available",
                        True
                    ):

                        return data

                return None

            return G.get_edge_data(
                u,
                v,
                key=key
            )

        return G.get_edge_data(
            u,
            v
        )