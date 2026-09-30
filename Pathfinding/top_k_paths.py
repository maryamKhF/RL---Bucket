# Pathfinding/top_k_paths.py

import math
import networkx as nx

from .heuristics import (
    lnd_cost,
    adaptive_heuristic,
    modified_cost,
    channel_fee,
    reliability_penalty
)


# ==========================================================
# Learned / Observed Directional Liquidity
# ==========================================================

def _estimated_liquidity(data):
    """
    Return learned/observed directional liquidity.

    IMPORTANT
    ---------
    capacity is NOT interpreted as:

        balance
        liquidity
        available payment amount

    Priority:

        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    If none exists:
        liquidity is unknown -> None
    """

    for field in (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv"
    ):

        if field not in data:
            continue

        value = data.get(
            field
        )

        if value is None:
            continue

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if (
            math.isfinite(value)
            and
            value >= 0.0
        ):

            return value

    return None


# ==========================================================
# Channel Feasibility
# ==========================================================

def _channel_can_carry(
    data,
    amount
):
    """
    Check known directional liquidity.

    Cases:

        liquidity < amount
            -> reject

        liquidity >= amount
            -> accept

        liquidity unknown
            -> accept

    Unknown liquidity is NOT treated as zero.

    Actual payment feasibility is evaluated later by the
    payment simulator / failure model.
    """

    liquidity = _estimated_liquidity(
        data
    )

    if liquidity is None:
        return True

    return (
        liquidity >= float(amount)
    )


# ==========================================================
# Channel Reliability
# ==========================================================

def _channel_reliability(data):
    """
    Return estimated forwarding reliability.

    If experience statistics exist:

        reliability =
            success /
            (success + failure)

    Otherwise use failure_probability if available.

    If there is no information, use the neutral initial
    failure probability used by Dijkstra.
    """

    success = _safe_float(
        data.get(
            "success_count",
            0
        )
    )

    failure = _safe_float(
        data.get(
            "failure_count",
            0
        )
    )

    total = (
        success
        +
        failure
    )

    if total > 0:

        probability = (
            failure
            /
            total
        )

    elif "failure_probability" in data:

        probability = _safe_float(
            data.get(
                "failure_probability"
            )
        )

    else:

        probability = 0.01

    probability = min(
        max(
            probability,
            0.0
        ),
        1.0
    )

    return (
        1.0
        -
        probability
    )


# ==========================================================
# Edge Cost
# ==========================================================

def _adaptive_edge_cost(
    G,
    u,
    v,
    data,
    amount,
    eta,
    heuristic_fn=None,
    lambda_h=1.0
):
    """
    Calculate the modified routing cost.

    Native cost:

        C_LND(e)

    Adaptive heuristic:

        h(u,v)

    Modified cost:

        C'(e)
        =
        C_LND(e)
        +
        lambda_h * h(u,v)

    The adaptive heuristic is generated from the node
    carbon-intensity values.

    rgb_color is treated as the numeric carbon-intensity
    value by heuristics.py.
    """

    # ------------------------------------------------------
    # Native routing cost
    # ------------------------------------------------------

    if heuristic_fn is not None:

        try:

            native_cost = heuristic_fn(
                G,
                u,
                v,
                data,
                amount
            )

        except (
            TypeError,
            ValueError,
            KeyError
        ):

            native_cost = lnd_cost(
                G,
                u,
                v,
                data,
                amount
            )

    else:

        native_cost = lnd_cost(
            G,
            u,
            v,
            data,
            amount
        )

    if not _valid_cost(
        native_cost
    ):
        return None

    # ------------------------------------------------------
    # Adaptive heuristic
    # ------------------------------------------------------

    try:

        h = adaptive_heuristic(
            G,
            u,
            v,
            eta
        )

    except (
        TypeError,
        ValueError,
        KeyError
    ):

        h = 0.0

    if not _valid_number(
        h
    ):
        return None

    # ------------------------------------------------------
    # Modified cost
    # ------------------------------------------------------

    try:

        cost = modified_cost(
            native_cost=float(
                native_cost
            ),
            geo_penalty=float(h),
            eta=float(eta),
            lambda_h=float(lambda_h)
        )

    except (
        TypeError,
        ValueError,
        KeyError
    ):

        return None

    if not _valid_cost(
        cost
    ):
        return None

    return float(
        cost
    )


# ==========================================================
# Build Routing Graph
# ==========================================================

def _build_routing_graph(
    G,
    amount,
    eta,
    max_hops,
    heuristic_fn=None,
    lambda_h=1.0
):
    """
    Build a simplified directed graph for K-best
    node-level candidate paths.

    The original graph may be a MultiDiGraph containing
    parallel channels between two nodes.

    For each directed node pair (u,v), the currently
    cheapest usable channel is retained.

    The original channel key and data are preserved.

    IMPORTANT
    ---------
    This produces K node-level candidate paths.

    It does NOT claim that K distinct channels are returned
    when multiple parallel channels exist between the same
    two nodes.
    """

    H = nx.DiGraph()

    # ------------------------------------------------------
    # Copy node attributes
    # ------------------------------------------------------

    for node, node_data in G.nodes(
        data=True
    ):

        H.add_node(
            node,
            **dict(node_data)
        )

    # ------------------------------------------------------
    # Process channels
    # ------------------------------------------------------

    for u, v, key, raw_data in G.edges(
        keys=True,
        data=True
    ):

        # --------------------------------------------------
        # Work with a copy.
        # --------------------------------------------------

        data = dict(
            raw_data
        )

        # --------------------------------------------------
        # Channel availability
        # --------------------------------------------------

        if not bool(
            data.get(
                "available",
                True
            )
        ):
            continue

        # --------------------------------------------------
        # Known liquidity constraint
        # --------------------------------------------------

        if not _channel_can_carry(
            data,
            amount
        ):
            continue

        # --------------------------------------------------
        # Maximum hop count is a path-level constraint.
        # --------------------------------------------------

        if max_hops <= 0:
            continue

        # --------------------------------------------------
        # Adaptive edge cost
        # --------------------------------------------------

        weight = _adaptive_edge_cost(
            G=G,
            u=u,
            v=v,
            data=data,
            amount=amount,
            eta=eta,
            heuristic_fn=heuristic_fn,
            lambda_h=lambda_h
        )

        if weight is None:
            continue

        # --------------------------------------------------
        # Channel metrics
        # --------------------------------------------------

        fee = channel_fee(
            data,
            amount
        )

        if not _valid_metric(
            fee
        ):
            continue

        delay = _channel_delay(
            data
        )

        if not _valid_metric(
            delay
        ):
            continue

        reliability = _channel_reliability(
            data
        )

        if not _valid_metric(
            reliability
        ):
            continue

        # --------------------------------------------------
        # Candidate edge attributes
        # --------------------------------------------------

        edge_attributes = {

            "weight":
                float(weight),

            "channel_key":
                key,

            "channel_data":
                data,

            "fee":
                float(fee),

            "delay":
                float(delay),

            "reliability":
                float(reliability),

            "eta":
                float(eta)
        }

        # --------------------------------------------------
        # Keep cheapest channel for node-level path search
        # --------------------------------------------------

        if not H.has_edge(
            u,
            v
        ):

            H.add_edge(
                u,
                v,
                **edge_attributes
            )

        else:

            existing_weight = (
                H[u][v]["weight"]
            )

            if weight < existing_weight:

                H[u][v].update(
                    edge_attributes
                )

    return H


# ==========================================================
# Complete Path Metrics
# ==========================================================

def _path_metrics(
    H,
    path
):
    """
    Calculate aggregate candidate-path metrics.

    Returns:

        cost
        fee
        delay
        reliability
    """

    total_cost = 0.0
    total_fee = 0.0
    total_delay = 0.0
    reliability = 1.0

    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        edge = H[u][v]

        total_cost += _safe_float(
            edge.get(
                "weight",
                0.0
            )
        )

        total_fee += _safe_float(
            edge.get(
                "fee",
                0.0
            )
        )

        total_delay += _safe_float(
            edge.get(
                "delay",
                0.0
            )
        )

        reliability *= min(
            max(
                _safe_float(
                    edge.get(
                        "reliability",
                        1.0
                    )
                ),
                0.0
            ),
            1.0
        )

    return {
        "cost":
            float(total_cost),

        "total_fee":
            float(total_fee),

        "total_delay":
            float(total_delay),

        "reliability":
            float(reliability),

        "failure_probability":
            float(
                1.0
                -
                reliability
            )
    }


# ==========================================================
# Convert Node Path to Channel Path
# ==========================================================

def _path_edges(
    H,
    path
):
    """
    Convert node path into:

        (u, v, channel_key)

    tuples.

    The selected original channel key is preserved.
    """

    edges = []

    for u, v in zip(
        path[:-1],
        path[1:]
    ):

        key = H[u][v].get(
            "channel_key"
        )

        edges.append(
            (
                u,
                v,
                key
            )
        )

    return edges


# ==========================================================
# K Candidate Routes
# ==========================================================

def top_k_paths(
    G,
    source,
    target,
    amount,
    heuristic_fn=None,
    eta=0.0,
    k=10,
    max_hops=12,
    lambda_h=1.0
):
    """
    Generate K candidate routing paths.

    Routing pipeline:

        Network State
              |
              v
             PPO
              |
              v
             eta
              |
              v
      Adaptive Heuristic
              |
              v
       Modified Edge Cost
              |
              v
       K Candidate Paths
              |
              v
            Bucket
              |
              v
       Payment Simulation

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Lightning-style routing graph.

    source : node
        Payment source.

    target : node
        Payment destination.

    amount : float
        Payment amount.

    heuristic_fn : callable, optional
        Native routing-cost function.

        If omitted, lnd_cost is used.

    eta : float
        PPO-generated adaptive routing parameter.

        -1 <= eta <= 1

    k : int
        Number of candidate paths.

    max_hops : int
        Maximum number of hops.

    lambda_h : float
        Weight of adaptive heuristic.

    Returns
    -------
    list of dict
        Candidate routes sorted by modified cost.

    IMPORTANT
    ---------
    candidate=True does NOT mean payment success.

    It means only that the path has been selected by the
    routing layer and can subsequently be tested by Bucket
    and the payment simulator.
    """

    # ======================================================
    # Validation
    # ======================================================

    if source == target:
        return []

    if source not in G:
        return []

    if target not in G:
        return []

    try:

        amount = float(
            amount
        )

    except (
        TypeError,
        ValueError
    ):

        return []

    if (
        not math.isfinite(amount)
        or
        amount <= 0
    ):
        return []

    try:

        k = int(
            k
        )

    except (
        TypeError,
        ValueError
    ):

        return []

    if k <= 0:
        return []

    try:

        max_hops = int(
            max_hops
        )

    except (
        TypeError,
        ValueError
    ):

        return []

    if max_hops <= 0:
        return []

    try:

        eta = float(
            eta
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "eta must be numeric"
        )

    if not math.isfinite(
        eta
    ):
        raise ValueError(
            "eta must be finite"
        )

    if not -1.0 <= eta <= 1.0:

        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    try:

        lambda_h = float(
            lambda_h
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "lambda_h must be numeric"
        )

    if (
        not math.isfinite(lambda_h)
        or
        lambda_h < 0.0
    ):

        raise ValueError(
            "lambda_h must be finite and >= 0"
        )

    # ======================================================
    # Build simplified routing graph
    # ======================================================

    H = _build_routing_graph(
        G=G,
        amount=amount,
        eta=eta,
        max_hops=max_hops,
        heuristic_fn=heuristic_fn,
        lambda_h=lambda_h
    )

    if (
        source not in H
        or
        target not in H
    ):
        return []

    # ======================================================
    # K shortest simple node paths
    # ======================================================

    try:

        path_generator = (
            nx.shortest_simple_paths(
                H,
                source,
                target,
                weight="weight"
            )
        )

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        return []

    results = []

    # ======================================================
    # Collect candidates
    # ======================================================

    try:

        for path in path_generator:

            hops = len(path) - 1

            # ----------------------------------------------
            # Invalid / excessive path
            # ----------------------------------------------

            if hops <= 0:
                continue

            if hops > max_hops:
                continue

            # ----------------------------------------------
            # Path edges
            # ----------------------------------------------

            edges = _path_edges(
                H,
                path
            )

            # ----------------------------------------------
            # Aggregate metrics
            # ----------------------------------------------

            metrics = _path_metrics(
                H,
                path
            )

            # ----------------------------------------------
            # Candidate
            # ----------------------------------------------

            candidate = {

                "path":
                    list(path),

                "edges":
                    edges,

                "cost":
                    metrics["cost"],

                "hop_count":
                    int(hops),

                "total_fee":
                    metrics["total_fee"],

                "total_delay":
                    metrics["total_delay"],

                "reliability":
                    metrics["reliability"],

                "failure_probability":
                    metrics[
                        "failure_probability"
                    ],

                "eta":
                    float(eta),

                "lambda_h":
                    float(lambda_h),

                "candidate":
                    True,

                "success":
                    None
            }

            results.append(
                candidate
            )

            # ----------------------------------------------
            # Stop after K candidates
            # ----------------------------------------------

            if len(results) >= k:
                break

    except nx.NetworkXNoPath:

        pass

    # ======================================================
    # Explicit ranking
    # ======================================================

    results.sort(
        key=lambda item:
            item["cost"]
    )

    return results


# ==========================================================
# Channel Delay
# ==========================================================

def _channel_delay(
    data
):
    """
    Extract channel delay.

    Supported attributes:

        delay
        cltv_expiry_delta
    """

    return _safe_float(
        data.get(
            "delay",
            data.get(
                "cltv_expiry_delta",
                0.0
            )
        )
    )


# ==========================================================
# Numeric Helpers
# ==========================================================

def _safe_float(
    value,
    default=0.0
):
    try:

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):

            return float(
                default
            )

        return value

    except (
        TypeError,
        ValueError
    ):

        return float(
            default
        )


def _valid_number(
    value
):
    try:

        value = float(
            value
        )

        return math.isfinite(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return False


def _valid_metric(
    value
):
    try:

        value = float(
            value
        )

        return (
            math.isfinite(value)
            and
            value >= 0.0
        )

    except (
        TypeError,
        ValueError
    ):

        return False


def _valid_cost(
    value
):
    """
    Dijkstra / shortest-simple-path algorithms require
    non-negative edge weights.
    """

    try:

        value = float(
            value
        )

        return (
            math.isfinite(value)
            and
            value >= 0.0
        )

    except (
        TypeError,
        ValueError
    ):

        return False