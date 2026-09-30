"""
Top-K candidate routing for the adaptive routing model.

The exact same adaptive edge-cost function used by Dijkstra
is used here.

Unified objective:

    C'(e) =
        C_LND(e)
        *
        (
            1 + lambda_h * penalty(e, eta)
        )

PPO controls eta only.

K is an explicit routing parameter and is never modified
inside this module.
"""

import math

import networkx as nx

from .heuristics import (
    lnd_cost,
    adaptive_edge_cost,
    channel_fee,
    channel_delay,
    validate_eta,
    validate_lambda_h,
)


# ==========================================================
# Directional Liquidity
# ==========================================================

def _estimated_liquidity(data):

    for field in (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv",
    ):

        if field not in data:
            continue

        value = data.get(field)

        if value is None:
            continue

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            math.isfinite(value)
            and value >= 0.0
        ):
            return value

    return None


# ==========================================================
# Channel Feasibility
# ==========================================================

def _channel_can_carry(
    data,
    amount,
):

    liquidity = _estimated_liquidity(
        data
    )

    if liquidity is None:
        return True

    return (
        liquidity
        >=
        float(amount)
    )


# ==========================================================
# Channel Reliability
# ==========================================================

def _channel_reliability(data):

    success = _safe_float(
        data.get(
            "success_count",
            0,
        )
    )

    failure = _safe_float(
        data.get(
            "failure_count",
            0,
        )
    )

    success = max(
        0.0,
        success,
    )

    failure = max(
        0.0,
        failure,
    )

    total = (
        success
        +
        failure
    )

    if total > 0.0:

        failure_probability = (
            failure
            /
            total
        )

    elif "failure_probability" in data:

        failure_probability = _safe_float(
            data.get(
                "failure_probability"
            ),
            default=0.01,
        )

    else:

        failure_probability = 0.01

    failure_probability = min(
        max(
            failure_probability,
            0.0,
        ),
        1.0,
    )

    return float(
        1.0
        -
        failure_probability
    )


# ==========================================================
# Adaptive Edge Cost
# ==========================================================

def _adaptive_edge_cost(
    G,
    u,
    v,
    data,
    amount,
    eta,
    heuristic_fn=None,
    lambda_h=1.0,
):
    """
    Compatibility wrapper around the unified adaptive cost.

    Returns:

        cost
        raw_heuristic
        adaptive_penalty
    """

    result = adaptive_edge_cost(
        G=G,
        u=u,
        v=v,
        data=data,
        amount=amount,
        eta=eta,
        heuristic_fn=(
            heuristic_fn
            if heuristic_fn is not None
            else lnd_cost
        ),
        lambda_h=lambda_h,
    )

    return (
        float(
            result["cost"]
        ),
        float(
            result["raw_heuristic"]
        ),
        float(
            result["adaptive_penalty"]
        ),
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
    lambda_h=1.0,
):
    """
    Build a simplified directed node-level graph.

    For parallel channels between the same node pair,
    the cheapest usable channel is retained.

    The exact channel key and channel data are preserved.
    """

    eta = validate_eta(
        eta
    )

    lambda_h = validate_lambda_h(
        lambda_h
    )

    H = nx.DiGraph()

    for node, node_data in G.nodes(
        data=True
    ):

        H.add_node(
            node,
            **dict(node_data),
        )

    if isinstance(
        G,
        (
            nx.MultiGraph,
            nx.MultiDiGraph,
        ),
    ):

        edge_iterator = G.edges(
            keys=True,
            data=True,
        )

        for (
            u,
            v,
            key,
            raw_data,
        ) in edge_iterator:

            data = dict(
                raw_data
            )

            if not bool(
                data.get(
                    "available",
                    True,
                )
            ):
                continue

            if not _node_available(
                G,
                u,
            ):
                continue

            if not _node_available(
                G,
                v,
            ):
                continue

            if not _channel_can_carry(
                data,
                amount,
            ):
                continue

            try:

                (
                    weight,
                    raw_h,
                    adaptive_penalty,
                ) = _adaptive_edge_cost(
                    G=G,
                    u=u,
                    v=v,
                    data=data,
                    amount=amount,
                    eta=eta,
                    heuristic_fn=heuristic_fn,
                    lambda_h=lambda_h,
                )

            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
            ):

                continue

            if not _valid_cost(
                weight
            ):
                continue

            try:

                fee = channel_fee(
                    data,
                    amount,
                )

            except (
                TypeError,
                ValueError,
                KeyError,
            ):

                continue

            if not _valid_metric(
                fee
            ):
                continue

            delay = channel_delay(
                data
            )

            if not _valid_metric(
                delay
            ):
                continue

            reliability = (
                _channel_reliability(
                    data
                )
            )

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
                    float(eta),

                "raw_heuristic":
                    float(raw_h),

                "adaptive_penalty":
                    float(adaptive_penalty),
            }

            if not H.has_edge(
                u,
                v,
            ):

                H.add_edge(
                    u,
                    v,
                    **edge_attributes,
                )

            else:

                existing_weight = H[u][v][
                    "weight"
                ]

                if (
                    weight
                    <
                    existing_weight
                ):

                    H[u][v].update(
                        edge_attributes
                    )

    else:

        for (
            u,
            v,
            raw_data,
        ) in G.edges(
            data=True
        ):

            data = dict(
                raw_data
            )

            if not bool(
                data.get(
                    "available",
                    True,
                )
            ):
                continue

            if not _node_available(
                G,
                u,
            ):
                continue

            if not _node_available(
                G,
                v,
            ):
                continue

            if not _channel_can_carry(
                data,
                amount,
            ):
                continue

            try:

                (
                    weight,
                    raw_h,
                    adaptive_penalty,
                ) = _adaptive_edge_cost(
                    G=G,
                    u=u,
                    v=v,
                    data=data,
                    amount=amount,
                    eta=eta,
                    heuristic_fn=heuristic_fn,
                    lambda_h=lambda_h,
                )

            except (
                TypeError,
                ValueError,
                KeyError,
                AttributeError,
            ):

                continue

            if not _valid_cost(
                weight
            ):
                continue

            try:

                fee = channel_fee(
                    data,
                    amount,
                )

            except (
                TypeError,
                ValueError,
                KeyError,
            ):

                continue

            if not _valid_metric(
                fee
            ):
                continue

            delay = channel_delay(
                data
            )

            if not _valid_metric(
                delay
            ):
                continue

            reliability = (
                _channel_reliability(
                    data
                )
            )

            edge_attributes = {

                "weight":
                    float(weight),

                "channel_key":
                    data.get(
                        "channel_key",
                        None,
                    ),

                "channel_data":
                    data,

                "fee":
                    float(fee),

                "delay":
                    float(delay),

                "reliability":
                    float(reliability),

                "eta":
                    float(eta),

                "raw_heuristic":
                    float(raw_h),

                "adaptive_penalty":
                    float(adaptive_penalty),
            }

            if (
                not H.has_edge(
                    u,
                    v,
                )
                or
                weight
                <
                H[u][v][
                    "weight"
                ]
            ):

                H.add_edge(
                    u,
                    v,
                    **edge_attributes,
                )

    return H


# ==========================================================
# Path Metrics
# ==========================================================

def _path_metrics(
    H,
    path,
):

    total_cost = 0.0
    total_fee = 0.0
    total_delay = 0.0

    reliability = 1.0

    total_raw_heuristic = 0.0
    total_adaptive_penalty = 0.0

    for (
        u,
        v,
    ) in zip(
        path[:-1],
        path[1:],
    ):

        edge = H[u][v]

        total_cost += _safe_float(
            edge.get(
                "weight",
                0.0,
            )
        )

        total_fee += _safe_float(
            edge.get(
                "fee",
                0.0,
            )
        )

        total_delay += _safe_float(
            edge.get(
                "delay",
                0.0,
            )
        )

        edge_reliability = min(
            max(
                _safe_float(
                    edge.get(
                        "reliability",
                        1.0,
                    )
                ),
                0.0,
            ),
            1.0,
        )

        reliability *= (
            edge_reliability
        )

        total_raw_heuristic += (
            _safe_float(
                edge.get(
                    "raw_heuristic",
                    0.0,
                )
            )
        )

        total_adaptive_penalty += (
            _safe_float(
                edge.get(
                    "adaptive_penalty",
                    0.0,
                )
            )
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
            ),

        "raw_heuristic":
            float(
                total_raw_heuristic
            ),

        "adaptive_penalty":
            float(
                total_adaptive_penalty
            ),
    }


# ==========================================================
# Path → Channel Edges
# ==========================================================

def _path_edges(
    H,
    path,
):

    edges = []

    for (
        u,
        v,
    ) in zip(
        path[:-1],
        path[1:],
    ):

        key = H[u][v].get(
            "channel_key"
        )

        edges.append(
            (
                u,
                v,
                key,
            )
        )

    return edges


# ==========================================================
# Candidate Validation
# ==========================================================

def _valid_candidate_path(
    path,
    source,
    target,
    max_hops,
):

    if not isinstance(
        path,
        (
            list,
            tuple,
        ),
    ):
        return False

    if len(path) < 2:
        return False

    if path[0] != source:
        return False

    if path[-1] != target:
        return False

    hops = len(path) - 1

    if hops > max_hops:
        return False

    if len(
        set(path)
    ) != len(path):
        return False

    return True


# ==========================================================
# Top-K
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
    lambda_h=1.0,
):
    """
    Generate K candidate paths.

    Important invariants:

        - eta controls the adaptive routing criterion.
        - k is explicit and is never changed internally.
        - source, target and amount are unchanged.
        - the adaptive cost is identical to Dijkstra.
    """

    if source == target:
        return []

    if source not in G:
        return []

    if target not in G:
        return []

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return []

    if (
        not math.isfinite(amount)
        or amount <= 0.0
    ):
        return []

    try:
        k = int(k)
    except (TypeError, ValueError):
        return []

    if k <= 0:
        return []

    try:
        max_hops = int(max_hops)
    except (TypeError, ValueError):
        return []

    if max_hops <= 0:
        return []

    eta = validate_eta(
        eta
    )

    lambda_h = validate_lambda_h(
        lambda_h
    )

    H = _build_routing_graph(
        G=G,
        amount=amount,
        eta=eta,
        max_hops=max_hops,
        heuristic_fn=(
            heuristic_fn
            if heuristic_fn is not None
            else lnd_cost
        ),
        lambda_h=lambda_h,
    )

    if (
        source not in H
        or target not in H
    ):
        return []

    try:

        path_generator = (
            nx.shortest_simple_paths(
                H,
                source,
                target,
                weight="weight",
            )
        )

    except (
        nx.NetworkXNoPath,
        nx.NodeNotFound,
    ):

        return []

    results = []
    seen_paths = set()

    try:

        for path in path_generator:

            if not _valid_candidate_path(
                path,
                source,
                target,
                max_hops,
            ):
                continue

            path_tuple = tuple(
                path
            )

            if path_tuple in seen_paths:
                continue

            seen_paths.add(
                path_tuple
            )

            edges = _path_edges(
                H,
                path,
            )

            if len(edges) != (
                len(path) - 1
            ):
                continue

            metrics = _path_metrics(
                H,
                path,
            )

            candidate = {

                "path":
                    list(path),

                "edges":
                    edges,

                "cost":
                    metrics["cost"],

                "hop_count":
                    int(
                        len(path) - 1
                    ),

                "total_fee":
                    metrics[
                        "total_fee"
                    ],

                "total_delay":
                    metrics[
                        "total_delay"
                    ],

                "reliability":
                    metrics[
                        "reliability"
                    ],

                "failure_probability":
                    metrics[
                        "failure_probability"
                    ],

                "eta":
                    float(eta),

                "lambda_h":
                    float(lambda_h),

                "raw_heuristic":
                    metrics[
                        "raw_heuristic"
                    ],

                "adaptive_penalty":
                    metrics[
                        "adaptive_penalty"
                    ],

                "candidate":
                    True,

                "success":
                    None,
            }

            results.append(
                candidate
            )

            if len(results) >= k:
                break

    except nx.NetworkXNoPath:
        pass

    results.sort(
        key=lambda item: (
            item["cost"],
            item["hop_count"],
            item["total_fee"],
            -item["reliability"],
            tuple(item["path"]),
        )
    )

    return results


# ==========================================================
# Node Availability
# ==========================================================

def _node_available(
    G,
    node,
):

    if node not in G:
        return False

    data = G.nodes[node]

    if not bool(
        data.get(
            "available",
            True,
        )
    ):
        return False

    if not bool(
        data.get(
            "is_online",
            True,
        )
    ):
        return False

    return True


# ==========================================================
# Numeric Helpers
# ==========================================================

def _safe_float(
    value,
    default=0.0,
):

    try:

        value = float(value)

        if not math.isfinite(value):
            return float(default)

        return value

    except (
        TypeError,
        ValueError,
    ):

        return float(default)


def _valid_metric(value):

    try:

        value = float(value)

        return (
            math.isfinite(value)
            and value >= 0.0
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def _valid_cost(value):

    try:

        value = float(value)

        return (
            math.isfinite(value)
            and value >= 0.0
        )

    except (
        TypeError,
        ValueError,
    ):

        return False