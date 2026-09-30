# Pathfinding/top_k_paths.py

import math
import networkx as nx

from .heuristics import (
    lnd_cost,
    adaptive_heuristic,
    modified_cost
)


# ==========================================================
# Learned / Observed Directional Liquidity
# ==========================================================

def _estimated_liquidity(data):
    """
    Return learned/observed directional liquidity.

    Important:
        capacity is NOT interpreted as balance or liquidity.

    Priority:
        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    If none exists, liquidity is unknown.
    """

    value = data.get(
        "estimated_liquidity",
        data.get(
            "liquidity_uv",
            data.get(
                "balance_uv",
                None
            )
        )
    )

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return None


# ==========================================================
# Channel Feasibility
# ==========================================================

def _channel_can_carry(
        data,
        amount
):
    """
    Check whether a channel has enough
    known directional liquidity.

    Unknown liquidity does NOT mean failure.

    Therefore:

        known liquidity < amount
            -> reject

        known liquidity >= amount
            -> accept

        unknown liquidity
            -> accept

    Actual payment feasibility is determined later
    by the payment simulator / failure model.
    """

    liquidity = _estimated_liquidity(
        data
    )

    if liquidity is None:
        return True

    return liquidity >= float(amount)


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
        lambda_h=1.0
):
    """
    Calculate the exact adaptive routing cost.

    C'(e) =
        C_LND(e)
        +
        lambda_h * h(u,v)

    where h(u,v) is the adaptive heuristic
    defined in README.md.
    """

    native_cost = lnd_cost(
        G,
        u,
        v,
        data,
        amount
    )

    h = adaptive_heuristic(
        G,
        u,
        v,
        eta
    )

    return modified_cost(
        native_cost=native_cost,
        geo_penalty=h,
        eta=eta,
        lambda_h=lambda_h
    )


# ==========================================================
# Build Routing Graph
# ==========================================================

def _build_routing_graph(
        G,
        amount,
        eta,
        max_hops,
        lambda_h=1.0
):
    """
    Build a simplified routing graph for K-best
    node paths.

    The original G can be a MultiDiGraph.

    For every (u,v), the cheapest currently usable
    channel is selected for the node-level candidate
    search.

    The original channel key and data are preserved
    so that Bucket and PaymentSimulator can use them.
    """

    H = nx.DiGraph()

    for u, v, key, data in G.edges(
            keys=True,
            data=True
    ):

        # --------------------------------------------------
        # Channel availability
        # --------------------------------------------------

        if not data.get(
                "available",
                True
        ):
            continue

        # --------------------------------------------------
        # Maximum hop is checked later on the complete
        # path. It is not an edge property.
        # --------------------------------------------------

        # --------------------------------------------------
        # Learned/observed liquidity
        # --------------------------------------------------

        if not _channel_can_carry(
                data,
                amount
        ):
            continue

        # --------------------------------------------------
        # Adaptive routing cost
        # --------------------------------------------------

        try:

            weight = _adaptive_edge_cost(
                G,
                u,
                v,
                data,
                amount,
                eta,
                lambda_h
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue

        if not math.isfinite(
                weight
        ):
            continue

        # --------------------------------------------------
        # Parallel channels
        #
        # For node-level K-shortest paths, keep the
        # cheapest usable channel between u and v.
        #
        # The selected channel key is preserved.
        # --------------------------------------------------

        if (
            not H.has_edge(u, v)
            or
            weight < H[u][v]["weight"]
        ):

            H.add_edge(

                u,
                v,

                weight=float(weight),

                channel_key=key,

                channel_data=data

            )

    return H


# ==========================================================
# Calculate Complete Path Cost
# ==========================================================

def _path_cost(
        H,
        path
):
    """
    Calculate:

        C'(p) = sum C'(e)

    over all edges in the path.
    """

    total_cost = 0.0

    for u, v in zip(
            path[:-1],
            path[1:]
    ):

        total_cost += float(
            H[u][v]["weight"]
        )

    return total_cost


# ==========================================================
# Convert Node Path to Channel Path
# ==========================================================

def _path_edges(
        H,
        path
):
    """
    Convert a node path into
    (u, v, channel_key) tuples.
    """

    edges = []

    for u, v in zip(
            path[:-1],
            path[1:]
    ):

        key = H[u][v][
            "channel_key"
        ]

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
    Generate K candidate routes according to README.md.

    Pipeline:

        PPO
          |
          v
        eta
          |
          v
        h(u,v)
          |
          v
        C'(e)
          |
          v
        K candidate routes
          |
          v
        Route ranking
          |
          v
        Bucket

    Parameters
    ----------
    G:
        Lightning MultiDiGraph.

    source:
        Payment source.

    target:
        Payment destination.

    amount:
        Payment amount.

    heuristic_fn:
        Kept for backward compatibility.
        The README-based adaptive heuristic is used
        internally.

    eta:
        PPO-generated adaptive parameter.

    k:
        Number of candidate routes generated.

    max_hops:
        Maximum number of hops.

    lambda_h:
        Weight of the adaptive heuristic.
    """

    # ------------------------------------------------------
    # Input validation
    # ------------------------------------------------------

    if source == target:
        return []

    if k <= 0:
        return []

    if max_hops <= 0:
        return []

    eta = float(eta)

    if not -1.0 <= eta <= 1.0:
        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    amount = float(
        amount
    )

    # ------------------------------------------------------
    # Source / target existence
    # ------------------------------------------------------

    if source not in G:
        return []

    if target not in G:
        return []

    # ------------------------------------------------------
    # Build routing graph
    # ------------------------------------------------------

    H = _build_routing_graph(
        G=G,
        amount=amount,
        eta=eta,
        max_hops=max_hops,
        lambda_h=lambda_h
    )

    if (
        source not in H
        or
        target not in H
    ):
        return []

    # ------------------------------------------------------
    # Generate K shortest simple node paths
    #
    # These are candidate routes, not yet successful
    # payments.
    # ------------------------------------------------------

    try:

        path_generator = nx.shortest_simple_paths(
            H,
            source,
            target,
            weight="weight"
        )

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound
    ):

        return []

    results = []

    # ------------------------------------------------------
    # Iterate over candidate paths
    # ------------------------------------------------------

    try:

        for path in path_generator:

            hops = len(path) - 1

            # ----------------------------------------------
            # Hop constraint
            # ----------------------------------------------

            if hops <= 0:
                continue

            if hops > max_hops:
                continue

            # ----------------------------------------------
            # Convert to edge representation
            # ----------------------------------------------

            edges = _path_edges(
                H,
                path
            )

            # ----------------------------------------------
            # Complete path cost
            # ----------------------------------------------

            total_cost = _path_cost(
                H,
                path
            )

            # ----------------------------------------------
            # Store candidate
            #
            # success is NOT True here.
            # This path has only been selected as a
            # candidate by the routing algorithm.
            # ----------------------------------------------

            results.append(
                {
                    "path": list(path),

                    "edges": edges,

                    "cost":
                        float(total_cost),

                    "hop_count":
                        int(hops),

                    "eta":
                        float(eta),

                    "candidate":
                        True
                }
            )

            # ----------------------------------------------
            # Stop after K candidates
            # ----------------------------------------------

            if len(results) >= k:
                break

    except nx.NetworkXNoPath:
        pass

    # ------------------------------------------------------
    # Explicit ranking
    #
    # README:
    #
    # C(p1) <= C(p2) <= ... <= C(pK)
    # ------------------------------------------------------

    results.sort(
        key=lambda item:
            item["cost"]
    )

    return results