"""
Simulation/failure_model.py

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

Important semantic rules
------------------------
- Channel capacity is NOT directional liquidity.
- Unknown directional liquidity is NOT treated as zero.
- A known directional liquidity value may reject a payment.
- A stochastic liquidity/forwarding failure may still occur when
  liquidity is unknown or sufficient.
- Exact MultiDiGraph channel keys are preserved whenever available.
- FailureModel evaluates exactly one payment attempt.
- Bucket and Backtracking logic are handled outside this module.
- FailureModel never selects an alternative route.
- FailureModel never modifies Bucket state.
"""

from __future__ import annotations

from datetime import datetime
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Static Failure Probability Assignment
# ============================================================

def assign_failure_probabilities(
    G,
    average_rate,
    seed=42,
):
    """
    Assign static channel-specific failure probabilities.

    Static channel attributes
    -------------------------
    failure_probability

    Runtime channel attributes
    --------------------------
    available
    failure_count
    last_failure

    Parameters
    ----------
    G : networkx.Graph
        Network graph.

    average_rate : float
        Mean channel failure probability.

    seed : int
        Seed used for deterministic probability assignment.

    Returns
    -------
    networkx.Graph
        The same graph instance after attributes are assigned.
    """

    if G is None:
        raise ValueError(
            "G cannot be None."
        )

    average_rate = _validate_probability(
        average_rate,
        "average_rate",
    )

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise ValueError(
            "seed must be an integer."
        )

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # MultiGraph / MultiDiGraph
    # --------------------------------------------------------

    if G.is_multigraph():

        for u, v, key, data in G.edges(
            keys=True,
            data=True,
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G=G,
                u=u,
                v=v,
                average_rate=average_rate,
                rng=rng,
            )

            data["failure_probability"] = probability

            # Runtime failure state.
            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    # --------------------------------------------------------
    # Graph / DiGraph
    # --------------------------------------------------------

    else:

        for u, v, data in G.edges(
            data=True,
        ):

            if u == v:
                continue

            probability = _calculate_failure_probability(
                G=G,
                u=u,
                v=v,
                average_rate=average_rate,
                rng=rng,
            )

            data["failure_probability"] = probability

            # Runtime failure state.
            data["available"] = True
            data["failure_count"] = 0
            data["last_failure"] = None

    return G


# ============================================================
# Probability Validation
# ============================================================

def _validate_probability(
    value,
    name,
):
    """
    Validate a probability value.

    Returns
    -------
    float
        Valid probability in [0, 1].
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be numeric."
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]."
        )

    return value


# ============================================================
# Amount Validation
# ============================================================

def _validate_amount(
    amount,
):
    """
    Validate payment amount.

    Returns
    -------
    float or None
        Valid positive amount, otherwise None.
    """

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(amount):
        return None

    if amount <= 0.0:
        return None

    return amount


# ============================================================
# Failure Probability Calculation
# ============================================================

def _calculate_failure_probability(
    G,
    u,
    v,
    average_rate,
    rng,
):
    """
    Calculate channel failure probability using
    topology/geography factors.

    The resulting probability is always capped at 0.95.
    """

    country_u = G.nodes[u].get(
        "country",
        "",
    )

    country_v = G.nodes[v].get(
        "country",
        "",
    )

    if country_u == country_v:

        factor = 0.7

    elif _intercontinental(
        G,
        u,
        v,
    ):

        factor = 2.0

    else:

        factor = 1.0

    probability = (
        float(average_rate)
        * factor
        * float(rng.uniform(0.8, 1.2))
    )

    return float(
        min(
            0.95,
            max(
                0.0,
                probability,
            ),
        )
    )


# ============================================================
# Geographic Helper
# ============================================================

def _intercontinental(
    G,
    u,
    v,
):
    """
    Detect a long-distance/intercontinental connection.

    This is a simulation abstraction and does not represent
    actual geographical routing information.
    """

    try:
        node_u = G.nodes[u]
        node_v = G.nodes[v]
    except Exception:
        return False

    longitude_u = node_u.get(
        "longitude",
        0,
    )

    longitude_v = node_v.get(
        "longitude",
        0,
    )

    latitude_u = node_u.get(
        "latitude",
        0,
    )

    latitude_v = node_v.get(
        "latitude",
        0,
    )

    try:

        longitude_distance = abs(
            float(longitude_u)
            - float(longitude_v)
        )

        latitude_distance = abs(
            float(latitude_u)
            - float(latitude_v)
        )

    except (
        TypeError,
        ValueError,
    ):

        return False

    return (
        longitude_distance > 45.0
        and
        latitude_distance > 10.0
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
    5. invalid_route
    6. invalid_amount
    7. invalid_edge
    8. invalid_route_edges
    9. edge_path_mismatch
    10. settlement-related failures are NOT handled here.

    Important
    ---------
    This class evaluates exactly one payment attempt.

    It does NOT:
        - select routes
        - generate routes
        - manage Bucket
        - perform backtracking
        - choose PPO actions
        - perform onion routing
        - perform settlement

    Temporary channel failures are stored in the graph and may
    affect subsequent attempts until reset_runtime_state() is
    called.
    """

    def __init__(
        self,
        node_failure_probability=0.01,
        liquidity_failure_probability=0.05,
        seed=42,
    ):

        self.node_failure_probability = _validate_probability(
            node_failure_probability,
            "node_failure_probability",
        )

        self.liquidity_failure_probability = _validate_probability(
            liquidity_failure_probability,
            "liquidity_failure_probability",
        )

        try:
            self.seed = int(seed)
        except (TypeError, ValueError):
            raise ValueError(
                "seed must be an integer."
            )

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
        reset_rng=False,
    ):
        """
        Reset temporary runtime failure state.

        Static failure_probability values are preserved.

        Parameters
        ----------
        G : networkx graph

        reset_counters : bool
            Reset failure_count and last_failure.

        reset_rng : bool
            Re-seed the runtime RNG.
        """

        if G is None:
            raise ValueError(
                "G cannot be None."
            )

        # ----------------------------------------------------
        # Channel runtime state
        # ----------------------------------------------------

        if G.is_multigraph():

            for _, _, _, data in G.edges(
                keys=True,
                data=True,
            ):

                # 'available' is the runtime failure state.
                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        else:

            for _, _, data in G.edges(
                data=True,
            ):

                data["available"] = True

                if reset_counters:

                    data["failure_count"] = 0
                    data["last_failure"] = None

        # ----------------------------------------------------
        # Node runtime state
        # ----------------------------------------------------

        for _, data in G.nodes(
            data=True,
        ):

            if "available" in data:
                data["available"] = True

            if "is_online" in data:
                data["is_online"] = True

            if "online" in data:
                data["online"] = True

        # ----------------------------------------------------
        # RNG reset
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
        node,
    ):
        """
        Check whether a node is unavailable or fails
        stochastically.

        Returns
        -------
        bool
            True means node failure.
        """

        if G is None:
            return True

        if node not in G.nodes:
            return True

        data = G.nodes[node]

        if data.get(
            "available",
            True,
        ) is False:

            return True

        if data.get(
            "is_online",
            True,
        ) is False:

            return True

        if data.get(
            "online",
            True,
        ) is False:

            return True

        return (
            self.rng.random()
            < self.node_failure_probability
        )

    # ========================================================
    # Channel Failure
    # ========================================================

    def check_channel_failure(
        self,
        G,
        u,
        v,
        key=None,
    ):
        """
        Check static/runtime channel failure.

        If a stochastic channel failure occurs, the exact
        channel is marked temporarily unavailable.

        Returns
        -------
        bool
            True means channel failure.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:
            return True

        if edge.get(
            "available",
            True,
        ) is False:

            return True

        probability = edge.get(
            "failure_probability",
            0.01,
        )

        try:
            probability = float(
                probability
            )
        except (
            TypeError,
            ValueError,
        ):
            probability = 0.01

        if not math.isfinite(
            probability
        ):
            probability = 0.01

        probability = min(
            1.0,
            max(
                0.0,
                probability,
            ),
        )

        failed = (
            self.rng.random()
            < probability
        )

        if failed:

            edge["failure_count"] = (
                int(
                    edge.get(
                        "failure_count",
                        0,
                    )
                )
                + 1
            )

            edge["last_failure"] = datetime.now()

            # Temporary runtime failure.
            edge["available"] = False

        return failed

    # ========================================================
    # Directional Liquidity
    # ========================================================

    def _get_directional_liquidity(
        self,
        edge,
    ):
        """
        Return known directional liquidity.

        Priority
        --------
        1. balance_uv
        2. liquidity_uv
        3. liquidity
        4. estimated_liquidity

        IMPORTANT
        ---------
        capacity is intentionally excluded.

        Returns
        -------
        float or None
            None means directional liquidity is unknown.
        """

        if not isinstance(
            edge,
            dict,
        ):
            return None

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            value = edge.get(
                field,
                None,
            )

            if value is None:
                continue

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if not math.isfinite(
                value
            ):

                continue

            if value < 0:
                continue

            return value

        return None

    # ========================================================
    # Liquidity Failure
    # ========================================================

    def check_liquidity_failure(
        self,
        G,
        u,
        v,
        amount,
        key=None,
    ):
        """
        Evaluate directional liquidity.

        Semantic rules
        --------------
        capacity
            ignored

        balance_uv
            used when available

        liquidity_uv
            used when available

        liquidity
            used when available

        estimated_liquidity
            used when available

        Known insufficient directional liquidity
            deterministic failure

        Unknown directional liquidity
            NOT treated as zero

        Known sufficient directional liquidity
            may still experience stochastic forwarding failure

        Returns
        -------
        bool
            True means liquidity/forwarding failure.
        """

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:
            return True

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:
            return True

        directional_liquidity = (
            self._get_directional_liquidity(
                edge
            )
        )

        # ----------------------------------------------------
        # Known directional liquidity
        # ----------------------------------------------------

        if directional_liquidity is not None:

            if valid_amount > directional_liquidity:
                return True

        # ----------------------------------------------------
        # Unknown or sufficient liquidity
        # ----------------------------------------------------
        #
        # Unknown liquidity is NOT failure by itself.
        #
        # Sufficient known liquidity is NOT a guarantee of
        # forwarding success because a stochastic forwarding
        # failure can still occur.
        # ----------------------------------------------------

        return (
            self.rng.random()
            < self.liquidity_failure_probability
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
        key=None,
    ):
        """
        Evaluate exactly one directed channel traversal.

        Returns
        -------
        dict
            success
            reason
            failed_node
            failed_edge
        """

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:

            return {
                "success": False,
                "reason": "invalid_amount",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        edge = self._get_edge(
            G,
            u,
            v,
            key,
        )

        if edge is None:

            return {
                "success": False,
                "reason": "missing_channel",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        if edge.get(
            "available",
            True,
        ) is False:

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Source node
        # ----------------------------------------------------

        if self.check_node_failure(
            G,
            u,
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": u,
                "failed_edge": None,
            }

        # ----------------------------------------------------
        # Destination node
        # ----------------------------------------------------

        if self.check_node_failure(
            G,
            v,
        ):

            return {
                "success": False,
                "reason": "node_failure",
                "failed_node": v,
                "failed_edge": None,
            }

        # ----------------------------------------------------
        # Exact channel
        # ----------------------------------------------------

        if self.check_channel_failure(
            G,
            u,
            v,
            key,
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Directional liquidity / stochastic forwarding
        # ----------------------------------------------------

        if self.check_liquidity_failure(
            G,
            u,
            v,
            valid_amount,
            key,
        ):

            return {
                "success": False,
                "reason": "liquidity_failure",
                "failed_node": None,
                "failed_edge": (
                    u,
                    v,
                    key,
                ),
            }

        # ----------------------------------------------------
        # Successful hop
        # ----------------------------------------------------

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
        route_edges=None,
    ):
        """
        Evaluate exactly one complete payment attempt.

        Parameters
        ----------
        route : list
            Ordered node path.

        amount : float
            Positive payment amount.

        network : networkx.Graph
            Graph containing exact runtime state.

        route_edges : list, optional
            Exact channel edges.

            Each edge may be:
                (u, v)

            or:
                (u, v, key)

        Returns
        -------
        dict

        Example
        -------
        Route:

            A -> B -> C -> D

        Failure:

            B -> C

        Result:

            failure_index = 1
        """

        if network is None:

            raise ValueError(
                "network cannot be None."
            )

        # ----------------------------------------------------
        # Validate route
        # ----------------------------------------------------

        if not isinstance(
            route,
            (list, tuple),
        ):

            return self._failure_result(
                reason="invalid_route",
            )

        route = list(route)

        if len(route) < 2:

            return self._failure_result(
                reason="invalid_route",
            )

        # ----------------------------------------------------
        # Validate amount
        # ----------------------------------------------------

        valid_amount = _validate_amount(
            amount
        )

        if valid_amount is None:

            return self._failure_result(
                reason="invalid_amount",
            )

        visited_edges = []

        # ----------------------------------------------------
        # Resolve exact edges
        # ----------------------------------------------------

        if route_edges is None:

            route_edges = self._resolve_route_edges(
                network,
                route,
            )

        if route_edges is None:

            return self._failure_result(
                reason="missing_channel",
                failure_index=0,
                visited_edges=[],
            )

        try:
            route_edges = list(
                route_edges
            )
        except TypeError:

            return self._failure_result(
                reason="invalid_route_edges",
                failure_index=0,
                visited_edges=[],
            )

        if len(route_edges) != len(route) - 1:

            return self._failure_result(
                reason="invalid_route_edges",
                failure_index=0,
                visited_edges=[],
            )

        # ----------------------------------------------------
        # Evaluate hops sequentially
        # ----------------------------------------------------

        for index, edge_info in enumerate(
            route_edges
        ):

            parsed = self._parse_edge(
                edge_info
            )

            if parsed is None:

                return self._failure_result(
                    reason="invalid_edge",
                    failed_edge=edge_info,
                    failure_index=index,
                    visited_edges=visited_edges,
                )

            u, v, key = parsed

            # ------------------------------------------------
            # Route-edge correspondence
            # ------------------------------------------------

            expected_u = route[index]
            expected_v = route[index + 1]

            if (
                u != expected_u
                or
                v != expected_v
            ):

                return self._failure_result(
                    reason="edge_path_mismatch",
                    failed_edge=(
                        u,
                        v,
                        key,
                    ),
                    failure_index=index,
                    visited_edges=visited_edges,
                )

            # ------------------------------------------------
            # Evaluate exact channel
            # ------------------------------------------------

            result = self.evaluate_edge(
                network,
                u,
                v,
                valid_amount,
                key,
            )

            if not result["success"]:

                return self._failure_result(
                    reason=result["reason"],
                    failed_node=result.get(
                        "failed_node"
                    ),
                    failed_edge=result.get(
                        "failed_edge"
                    ),
                    failure_index=index,
                    visited_edges=visited_edges,
                )

            # ------------------------------------------------
            # Record successfully traversed edge
            # ------------------------------------------------

            visited_edges.append(
                (
                    u,
                    v,
                    key,
                )
            )

        # ----------------------------------------------------
        # Complete success
        # ----------------------------------------------------

        return {
            "success": True,
            "reason": None,
            "failed_node": None,
            "failed_edge": None,
            "failure_index": None,
            "visited_edges": list(
                visited_edges
            ),
        }

    # ========================================================
    # Standard Failure Result
    # ========================================================

    @staticmethod
    def _failure_result(
        reason,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None,
    ):
        """
        Build a normalized failure result.
        """

        if visited_edges is None:
            visited_edges = []

        return {
            "success": False,
            "reason": reason,
            "failed_node": failed_node,
            "failed_edge": failed_edge,
            "failure_index": failure_index,
            "visited_edges": list(
                visited_edges
            ),
        }

    # ========================================================
    # Route Edge Resolution
    # ========================================================

    def _resolve_route_edges(
        self,
        G,
        route,
    ):
        """
        Resolve node-path edges when exact edge information
        was not supplied.

        Important
        ---------
        This is a compatibility fallback.

        The main Top-K / PaymentSimulator pipeline should
        supply exact route_edges with channel keys.

        For MultiDiGraph:
            - only available channels are considered
            - one available channel is selected deterministically
              according to graph iteration order
        """

        if G is None:
            return None

        if route is None:
            return None

        edges = []

        for i in range(
            len(route) - 1
        ):

            u = route[i]
            v = route[i + 1]

            if not G.has_edge(
                u,
                v,
            ):

                return None

            # ------------------------------------------------
            # MultiGraph / MultiDiGraph
            # ------------------------------------------------

            if G.is_multigraph():

                edge_data = G.get_edge_data(
                    u,
                    v,
                )

                if not edge_data:
                    return None

                selected_key = None

                for key, data in edge_data.items():

                    if data.get(
                        "available",
                        True,
                    ) is False:

                        continue

                    selected_key = key
                    break

                if selected_key is None:
                    return None

                edges.append(
                    (
                        u,
                        v,
                        selected_key,
                    )
                )

            # ------------------------------------------------
            # Graph / DiGraph
            # ------------------------------------------------

            else:

                data = G.get_edge_data(
                    u,
                    v,
                )

                if data is None:
                    return None

                if data.get(
                    "available",
                    True,
                ) is False:

                    return None

                edges.append(
                    (
                        u,
                        v,
                        None,
                    )
                )

        return edges

    # ========================================================
    # Edge Parser
    # ========================================================

    @staticmethod
    def _parse_edge(
        edge_info,
    ):
        """
        Normalize a 2-tuple or 3-tuple edge description.

        Returns
        -------
        tuple or None

        (u, v, None)
        or
        (u, v, key)
        """

        if edge_info is None:
            return None

        if not isinstance(
            edge_info,
            (list, tuple),
        ):

            return None

        if len(edge_info) == 2:

            u, v = edge_info

            return (
                u,
                v,
                None,
            )

        if len(edge_info) == 3:

            u, v, key = edge_info

            return (
                u,
                v,
                key,
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
        key=None,
    ):
        """
        Return graph edge data.

        MultiGraph / MultiDiGraph
        -------------------------
        key != None:
            exact channel only.

        key == None:
            first currently available channel as compatibility
            fallback.

        The main payment pipeline should provide exact keys.
        """

        if G is None:
            return None

        try:

            if not G.has_edge(
                u,
                v,
            ):

                return None

        except Exception:

            return None

        # ----------------------------------------------------
        # MultiGraph / MultiDiGraph
        # ----------------------------------------------------

        if G.is_multigraph():

            try:

                edge_data = G.get_edge_data(
                    u,
                    v,
                )

            except Exception:

                return None

            if not edge_data:
                return None

            # ------------------------------------------------
            # Exact channel requested
            # ------------------------------------------------

            if key is not None:

                return edge_data.get(
                    key
                )

            # ------------------------------------------------
            # Compatibility fallback
            # ------------------------------------------------

            for _, data in edge_data.items():

                if data.get(
                    "available",
                    True,
                ) is False:

                    continue

                return data

            return None

        # ----------------------------------------------------
        # Graph / DiGraph
        # ----------------------------------------------------

        try:

            return G.get_edge_data(
                u,
                v,
            )

        except Exception:

            return None


# ============================================================
# Standalone Functional Test
# ============================================================

def _run_standalone_test():
    """
    Internal deterministic smoke test.

    This test verifies:

    1. capacity is NOT used as liquidity
    2. exact MultiDiGraph key is preserved
    3. known insufficient liquidity fails
    4. unknown liquidity is not rejected as zero
    5. failure_index is correct
    6. visited_edges are correct
    7. runtime channel failure is stored on exact channel
    """

    import networkx as nx

    print()
    print("=" * 72)
    print("FAILURE MODEL STANDALONE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        available=True,
    )

    G.add_node(
        "B",
        available=True,
    )

    G.add_node(
        "C",
        available=True,
    )

    # Channel A -> B, key 0.
    # capacity is large, directional liquidity is small.
    G.add_edge(
        "A",
        "B",
        key=0,
        capacity=10000,
        balance_uv=500,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    # Parallel channel A -> B, key 1.
    # This verifies exact key handling.
    G.add_edge(
        "A",
        "B",
        key=1,
        capacity=10000,
        balance_uv=10000,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    # B -> C.
    # No directional liquidity information.
    G.add_edge(
        "B",
        "C",
        key=0,
        capacity=10000,
        available=True,
        failure_probability=0.0,
        failure_count=0,
        last_failure=None,
    )

    model = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    # --------------------------------------------------------
    # Test 1: capacity must NOT be treated as liquidity
    # --------------------------------------------------------

    print()
    print("[1] CAPACITY SEMANTICS")

    result = model.evaluate_payment_failure(
        route=[
            "B",
            "C",
        ],
        amount=5000,
        network=G,
        route_edges=[
            ("B", "C", 0),
        ],
    )

    assert result["success"] is True

    print(
        "  PASS: capacity is not treated as directional liquidity."
    )

    # --------------------------------------------------------
    # Test 2: known insufficient directional liquidity
    # --------------------------------------------------------

    print()
    print("[2] KNOWN LIQUIDITY")

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=600,
        network=G,
        route_edges=[
            ("A", "B", 0),
        ],
    )

    assert result["success"] is False
    assert result["reason"] == "liquidity_failure"
    assert result["failed_edge"] == (
        "A",
        "B",
        0,
    )
    assert result["failure_index"] == 0

    print(
        "  PASS: insufficient known directional liquidity fails."
    )

    # --------------------------------------------------------
    # Test 3: exact parallel channel
    # --------------------------------------------------------

    print()
    print("[3] EXACT MULTIDIGRAPH CHANNEL")

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=600,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    assert result["success"] is True
    assert result["visited_edges"] == [
        (
            "A",
            "B",
            1,
        )
    ]

    print(
        "  PASS: exact channel key is preserved."
    )

    # --------------------------------------------------------
    # Test 4: exact failure index
    # --------------------------------------------------------

    print()
    print("[4] FAILURE INDEX")

    G["B"]["C"][0]["failure_probability"] = 1.0

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
            "C",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    assert result["success"] is False
    assert result["reason"] == "channel_failure"
    assert result["failed_edge"] == (
        "B",
        "C",
        0,
    )
    assert result["failure_index"] == 1
    assert result["visited_edges"] == [
        (
            "A",
            "B",
            1,
        )
    ]

    print(
        "  PASS: failure_index and visited_edges are correct."
    )

    # --------------------------------------------------------
    # Test 5: runtime failure state
    # --------------------------------------------------------

    print()
    print("[5] RUNTIME FAILURE STATE")

    assert G["B"]["C"][0]["available"] is False
    assert G["B"]["C"][0]["failure_count"] == 1

    print(
        "  PASS: exact failed channel is marked unavailable."
    )

    # --------------------------------------------------------
    # Test 6: reset
    # --------------------------------------------------------

    print()
    print("[6] RESET RUNTIME STATE")

    model.reset_runtime_state(
        G,
        reset_counters=True,
        reset_rng=True,
    )

    assert G["B"]["C"][0]["available"] is True
    assert G["B"]["C"][0]["failure_count"] == 0
    assert G["B"]["C"][0]["last_failure"] is None

    print(
        "  PASS: runtime state and RNG can be reset."
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FAILURE MODEL STATUS : SUCCESS")
    print("=" * 72)


# ============================================================
# Standalone Execution
# ============================================================

if __name__ == "__main__":
    _run_standalone_test()