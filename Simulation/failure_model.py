"""
failure_model.py

Lightning Network failure simulation model.

Responsibilities
----------------
1. Assign static failure probabilities to channels.
2. Evaluate node/channel/liquidity failures during payment execution.
3. Return the exact failed element and failure position.
4. Provide failure information required by Partial Backtracking.

Design
------
The NetworkX graph G is the single source of truth for network state.

The model is intentionally separated from:
    - routing
    - Bucket
    - Partial Backtracking
    - PPO
    - Onion routing

Those modules consume the structured failure information returned here.
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

    Parameters
    ----------
    G : networkx.Graph / MultiGraph
        Lightning Network graph.

    average_rate : float
        Base channel failure probability.

    seed : int
        Seed used for reproducible probability assignment.

    Notes
    -----
    The function initializes channel runtime state as well.
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

    # --------------------------------------------------------
    # MultiGraph / MultiDiGraph
    # --------------------------------------------------------

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

            # Runtime channel state
            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    # --------------------------------------------------------
    # Graph / DiGraph
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Geographic factor
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Small stochastic variation
    # --------------------------------------------------------

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

    return (
        abs(
            longitude_u
            -
            longitude_v
        )
        > 45
        and
        abs(
            latitude_u
            -
            latitude_v
        )
        > 10
    )


# ============================================================
# Failure Model
# ============================================================

class FailureModel:
    """
    Runtime failure evaluator for Lightning payment simulation.

    The graph G is the single source of truth.

    Failure categories
    ------------------
    1. node_failure
    2. channel_failure
    3. liquidity_failure
    4. missing_channel

    The evaluator stops at the first failure because the exact
    failure position is required by Partial Backtracking.
    """

    def __init__(
        self,
        node_failure_probability=0.01,
        liquidity_failure_probability=0.05,
        seed=42
    ):
        """
        Parameters
        ----------
        node_failure_probability : float
            Probability of temporary node failure.

        liquidity_failure_probability : float
            Probability of a stochastic liquidity failure when
            the deterministic liquidity check passes.

        seed : int
            Random seed for reproducibility.
        """

        if not 0 <= node_failure_probability <= 1:
            raise ValueError(
                "node_failure_probability must be in [0, 1]."
            )

        if not 0 <= liquidity_failure_probability <= 1:
            raise ValueError(
                "liquidity_failure_probability must be in [0, 1]."
            )

        self.node_failure_probability = (
            float(node_failure_probability)
        )

        self.liquidity_failure_probability = (
            float(liquidity_failure_probability)
        )

        self.rng = random.Random(seed)

    # ========================================================
    # Node Failure
    # ========================================================

    def check_node_failure(
        self,
        G,
        node
    ):
        """
        Check whether a node is unavailable.

        Returns
        -------
        bool
            True if node failure occurs.
        """

        if node not in G.nodes:

            return True

        data = G.nodes[node]

        # Persistent network state
        if data.get(
            "available",
            True
        ) is False:

            return True

        # Compatibility with possible node state
        if data.get(
            "is_online",
            True
        ) is False:

            return True

        # Stochastic node failure
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
        """
        Check the exact directed channel u -> v.

        Returns
        -------
        bool
            True if the channel has failed.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key
        )

        if edge is None:

            return True

        # Persistent unavailable state
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

            edge["last_failure"] = (
                datetime.now()
            )

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
        """
        Check whether the directed channel u -> v
        can forward the requested amount.

        The check is based primarily on directional
        balance_uv rather than only total capacity.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key
        )

        if edge is None:

            return True

        # ----------------------------------------------------
        # Channel capacity
        # ----------------------------------------------------

        capacity = edge.get(
            "capacity",
            None
        )

        if capacity is not None:

            try:
                if amount > float(capacity):
                    return True

            except (
                TypeError,
                ValueError
            ):
                pass

        # ----------------------------------------------------
        # Directional liquidity
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Optional generic liquidity field
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Stochastic liquidity failure
        # ----------------------------------------------------

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
        """
        Evaluate one forwarding edge.

        Returns a structured result suitable for
        Partial Backtracking.
        """

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

        # ----------------------------------------------------
        # Channel availability
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Node state
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Channel failure
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Liquidity
        # ----------------------------------------------------

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
        """
        Evaluate a complete payment route.

        Parameters
        ----------
        route : list
            Ordered list of nodes.

        amount : float
            Payment amount.

        network : NetworkX graph
            Current Lightning graph.

        route_edges : list, optional
            Exact edge descriptors.

            Each item can be:
                (u, v)
                (u, v, key)

            If omitted, the first matching edge is selected.

        Returns
        -------
        dict

        Example successful result
        -------------------------
        {
            "success": True,
            "reason": None,
            "failed_node": None,
            "failed_edge": None,
            "failure_index": None,
            "visited_edges": [...]
        }

        Example failure result
        ----------------------
        {
            "success": False,
            "reason": "channel_failure",
            "failed_node": None,
            "failed_edge": (u, v, key),
            "failure_index": 3,
            "visited_edges": [...]
        }

        The failure_index is the position of the failed
        forwarding edge in the route.
        """

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

        # ----------------------------------------------------
        # Build edge list if not explicitly supplied
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Evaluate each forwarding step
        # ----------------------------------------------------

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
        """
        Resolve exact graph edges for a node route.

        For MultiGraph, the first available edge is selected.

        The routing module should eventually provide the exact
        selected channel key, so this fallback is mainly for
        compatibility/testing.
        """

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

                    selected_key = next(
                        iter(
                            edge_data
                        )
                    )

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
        """
        Normalize edge representation.
        """

        if edge_info is None:
            return None

        if len(edge_info) == 2:

            u, v = edge_info

            return (
                u,
                v,
                None
            )

        if len(edge_info) == 3:

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
        """
        Return the exact edge data.

        Direction is preserved.

        For MultiGraph:
            G[u][v][key]

        For Graph:
            G[u][v]
        """

        if not G.has_edge(
            u,
            v
        ):

            return None

        if G.is_multigraph():

            edge_data = G.get_edge_data(
                u,
                v,
                key=key
            )

            return edge_data

        return G.get_edge_data(
            u,
            v
        )