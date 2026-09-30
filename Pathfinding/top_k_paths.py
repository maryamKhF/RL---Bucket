# Pathfinding/top_k_paths.py

import math
import networkx as nx

from .heuristics import (
    lnd_cost,
    adaptive_heuristic,
    channel_fee,
)


# ==========================================================
# Learned / Observed Directional Liquidity
# ==========================================================

def _estimated_liquidity(data):
    """
    Return learned/observed directional liquidity.

    Priority:
        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    Important:
        capacity is NOT interpreted as liquidity.

    Returns:
        float
            Known directional liquidity.

        None
            Unknown liquidity.
    """

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
        except (TypeError, ValueError):
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

def _channel_can_carry(data, amount):
    """
    Check known directional liquidity.

    liquidity < amount
        -> channel is rejected

    liquidity >= amount
        -> channel is accepted

    liquidity unknown
        -> channel is accepted

    Actual payment feasibility is evaluated later by
    FailureModel / PaymentSimulator.
    """

    liquidity = _estimated_liquidity(data)

    if liquidity is None:
        return True

    return liquidity >= float(amount)


# ==========================================================
# Channel Reliability
# ==========================================================

def _channel_reliability(data):
    """
    Estimate forwarding reliability.

    Priority:
        1. success/failure history
        2. failure_probability
        3. default failure probability = 0.01

    Returns:
        reliability in [0, 1]
    """

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
# Adaptive Heuristic Penalty
# ==========================================================

def _adaptive_penalty(
    G,
    u,
    v,
    eta,
):
    """
    Convert the raw adaptive heuristic into a stable,
    non-negative routing penalty.

    The original adaptive heuristic can be negative:

        h(u,v) < 0

    Passing that value directly into the routing cost can
    create an artificial near-zero edge cost because
    modified_cost() clips negative final costs.

    Therefore:

        raw_h = adaptive_heuristic(...)

        penalty = abs(raw_h)

    This preserves the magnitude of the adaptive signal
    while guaranteeing a non-negative penalty.

    A bounded normalization is then applied:

        normalized =
            penalty / (1 + penalty)

    Therefore:

        0 <= normalized < 1

    This prevents carbon-related values from completely
    dominating the native LND cost.

    Returns:
        raw_heuristic
        normalized_penalty
    """

    raw_h = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    if not _valid_number(raw_h):
        raise ValueError(
            "Adaptive heuristic must be finite."
        )

    raw_h = float(raw_h)

    magnitude = abs(
        raw_h
    )

    normalized_penalty = (
        magnitude
        /
        (
            1.0
            +
            magnitude
        )
    )

    if not _valid_number(
        normalized_penalty
    ):
        raise ValueError(
            "Adaptive penalty is not finite."
        )

    normalized_penalty = min(
        max(
            normalized_penalty,
            0.0,
        ),
        1.0,
    )

    return (
        raw_h,
        float(
            normalized_penalty
        ),
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
    lambda_h=1.0,
):
    """
    Calculate adaptive routing cost.

    Native cost:

        C_LND(e)

    Adaptive signal:

        h(u,v)

    Stable non-negative penalty:

        p(u,v) =
            |h(u,v)| / (1 + |h(u,v)|)

    Final cost:

        C'(e) =
            C_LND(e)
            *
            (
                1
                +
                lambda_h * p(u,v)
            )

    This formulation guarantees:

        C'(e) >= C_LND(e) > 0

    and therefore avoids the previous problem where a
    negative adaptive heuristic could make the final edge
    cost collapse to approximately zero.

    PPO controls eta only.
    """

    # ------------------------------------------------------
    # Native LND cost
    # ------------------------------------------------------

    if heuristic_fn is not None:

        try:

            native_cost = heuristic_fn(
                G,
                u,
                v,
                data,
                amount,
            )

        except (
            TypeError,
            ValueError,
            KeyError,
        ):

            native_cost = lnd_cost(
                G,
                u,
                v,
                data,
                amount,
            )

    else:

        native_cost = lnd_cost(
            G,
            u,
            v,
            data,
            amount,
        )

    if not _valid_cost(
        native_cost
    ):
        return None, None, None

    native_cost = float(
        native_cost
    )

    # ------------------------------------------------------
    # Adaptive heuristic
    # ------------------------------------------------------

    try:

        raw_h, penalty = (
            _adaptive_penalty(
                G=G,
                u=u,
                v=v,
                eta=eta,
            )
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ):

        return None, None, None

    # ------------------------------------------------------
    # Lambda validation
    # ------------------------------------------------------

    try:

        lambda_h = float(
            lambda_h
        )

    except (
        TypeError,
        ValueError,
    ):

        return None, None, None

    if (
        not math.isfinite(
            lambda_h
        )
        or
        lambda_h < 0.0
    ):
        return None, None, None

    # ------------------------------------------------------
    # Adaptive cost
    # ------------------------------------------------------

    try:

        cost = (
            native_cost
            *
            (
                1.0
                +
                lambda_h
                *
                penalty
            )
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return None, None, None

    if not _valid_cost(
        cost
    ):
        return None, None, None

    return (
        float(cost),
        float(raw_h),
        float(penalty),
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
    Build a simplified directed graph for node-level
    candidate route generation.

    For MultiDiGraph:

        multiple channels between the same node pair
        are reduced to the cheapest usable channel.

    The original channel key and channel data are preserved.

    Important:
        eta affects the edge cost used for candidate ranking.
    """

    H = nx.DiGraph()

    # ------------------------------------------------------
    # Copy nodes
    # ------------------------------------------------------

    for node, node_data in G.nodes(
        data=True
    ):

        H.add_node(
            node,
            **dict(
                node_data
            ),
        )

    # ------------------------------------------------------
    # MultiGraph / MultiDiGraph
    # ------------------------------------------------------

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

            # ----------------------------------------------
            # Channel availability
            # ----------------------------------------------

            if not bool(
                data.get(
                    "available",
                    True,
                )
            ):
                continue

            # ----------------------------------------------
            # Node availability
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Directional liquidity
            # ----------------------------------------------

            if not _channel_can_carry(
                data,
                amount,
            ):
                continue

            # ----------------------------------------------
            # Adaptive cost
            # ----------------------------------------------

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

            if weight is None:
                continue

            # ----------------------------------------------
            # Channel fee
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Delay
            # ----------------------------------------------

            delay = _channel_delay(
                data
            )

            if not _valid_metric(
                delay
            ):
                continue

            # ----------------------------------------------
            # Reliability
            # ----------------------------------------------

            reliability = (
                _channel_reliability(
                    data
                )
            )

            if not _valid_metric(
                reliability
            ):
                continue

            # ----------------------------------------------
            # Edge attributes
            # ----------------------------------------------

            edge_attributes = {

                "weight":
                    float(
                        weight
                    ),

                "channel_key":
                    key,

                "channel_data":
                    data,

                "fee":
                    float(
                        fee
                    ),

                "delay":
                    float(
                        delay
                    ),

                "reliability":
                    float(
                        reliability
                    ),

                "eta":
                    float(
                        eta
                    ),

                "raw_heuristic":
                    float(
                        raw_h
                    ),

                "adaptive_penalty":
                    float(
                        adaptive_penalty
                    ),
            }

            # ----------------------------------------------
            # Keep cheapest usable channel
            # ----------------------------------------------

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

                existing_weight = (
                    H[u][v][
                        "weight"
                    ]
                )

                if (
                    weight
                    <
                    existing_weight
                ):

                    H[u][v].update(
                        edge_attributes
                    )

    # ------------------------------------------------------
    # Ordinary Graph / DiGraph
    # ------------------------------------------------------

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

            if weight is None:
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

            delay = _channel_delay(
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

            if not _valid_metric(
                reliability
            ):
                continue

            edge_attributes = {

                "weight":
                    float(
                        weight
                    ),

                "channel_key":
                    data.get(
                        "channel_key",
                        None,
                    ),

                "channel_data":
                    data,

                "fee":
                    float(
                        fee
                    ),

                "delay":
                    float(
                        delay
                    ),

                "reliability":
                    float(
                        reliability
                    ),

                "eta":
                    float(
                        eta
                    ),

                "raw_heuristic":
                    float(
                        raw_h
                    ),

                "adaptive_penalty":
                    float(
                        adaptive_penalty
                    ),
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
                    **edge_attributes
                )

    return H


# ==========================================================
# Complete Path Metrics
# ==========================================================

def _path_metrics(
    H,
    path,
):
    """
    Calculate aggregate candidate-path metrics.
    """

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
            float(
                total_cost
            ),

        "total_fee":
            float(
                total_fee
            ),

        "total_delay":
            float(
                total_delay
            ),

        "reliability":
            float(
                reliability
            ),

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
# Convert Node Path to Channel Path
# ==========================================================

def _path_edges(
    H,
    path,
):
    """
    Convert a node path into exact channel edges:

        (u, v, channel_key)

    The original channel key is preserved.
    """

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
# Validate Candidate Route
# ==========================================================

def _valid_candidate_path(
    path,
    source,
    target,
    max_hops,
):
    """
    Validate a node-level candidate path.
    """

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

    # No repeated nodes.
    if len(
        set(path)
    ) != len(path):
        return False

    return True


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
    lambda_h=1.0,
):
    """
    Generate K candidate routing paths.

    Pipeline:

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
      Adaptive Penalty
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

    PPO controls eta only.

    k remains an explicit fixed routing parameter.
    The function never changes k dynamically.

    Returns
    -------

    list of dict

        Each candidate contains:

            path
            edges
            cost
            hop_count
            total_fee
            total_delay
            reliability
            failure_probability
            eta
            lambda_h
            raw_heuristic
            adaptive_penalty
            candidate
            success
    """

    # ======================================================
    # Basic validation
    # ======================================================

    if source == target:
        return []

    if source not in G:
        return []

    if target not in G:
        return []

    # ------------------------------------------------------
    # Amount
    # ------------------------------------------------------

    try:

        amount = float(
            amount
        )

    except (
        TypeError,
        ValueError,
    ):

        return []

    if (
        not math.isfinite(
            amount
        )
        or
        amount <= 0.0
    ):
        return []

    # ------------------------------------------------------
    # K
    # ------------------------------------------------------

    try:

        k = int(
            k
        )

    except (
        TypeError,
        ValueError,
    ):

        return []

    if k <= 0:
        return []

    # ------------------------------------------------------
    # Maximum hops
    # ------------------------------------------------------

    try:

        max_hops = int(
            max_hops
        )

    except (
        TypeError,
        ValueError,
    ):

        return []

    if max_hops <= 0:
        return []

    # ------------------------------------------------------
    # Eta
    # ------------------------------------------------------

    try:

        eta = float(
            eta
        )

    except (
        TypeError,
        ValueError,
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

    if not (
        -1.0
        <=
        eta
        <=
        1.0
    ):

        raise ValueError(
            "eta must be in [-1, 1], "
            f"got {eta}"
        )

    # ------------------------------------------------------
    # Lambda
    # ------------------------------------------------------

    try:

        lambda_h = float(
            lambda_h
        )

    except (
        TypeError,
        ValueError,
    ):

        raise ValueError(
            "lambda_h must be numeric"
        )

    if (
        not math.isfinite(
            lambda_h
        )
        or
        lambda_h < 0.0
    ):

        raise ValueError(
            "lambda_h must be finite "
            "and >= 0"
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
        lambda_h=lambda_h,
    )

    if (
        source not in H
        or
        target not in H
    ):

        return []

    # ======================================================
    # Candidate generation
    # ======================================================

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

            # ------------------------------------------------
            # Validate path
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Convert node path to exact channels
            # ------------------------------------------------

            edges = _path_edges(
                H,
                path,
            )

            if len(edges) != (
                len(path)
                -
                1
            ):

                continue

            # ------------------------------------------------
            # Aggregate metrics
            # ------------------------------------------------

            metrics = _path_metrics(
                H,
                path,
            )

            # ------------------------------------------------
            # Candidate
            # ------------------------------------------------

            candidate = {

                "path":
                    list(
                        path
                    ),

                "edges":
                    edges,

                "cost":
                    metrics[
                        "cost"
                    ],

                "hop_count":
                    int(
                        len(path)
                        -
                        1
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
                    float(
                        eta
                    ),

                "lambda_h":
                    float(
                        lambda_h
                    ),

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

                # PaymentSimulator updates this.
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

    # ======================================================
    # Explicit deterministic ranking
    # ======================================================

    results.sort(
        key=lambda item: (
            item["cost"],
            item["hop_count"],
            item["total_fee"],
            -item["reliability"],
            tuple(
                item["path"]
            ),
        )
    )

    return results


# ==========================================================
# Channel Delay
# ==========================================================

def _channel_delay(
    data,
):
    """
    Extract channel delay.

    Supported attributes:

        delay
        cltv_expiry_delta
    """

    return max(
        0.0,
        _safe_float(
            data.get(
                "delay",
                data.get(
                    "cltv_expiry_delta",
                    0.0,
                ),
            )
        ),
    )


# ==========================================================
# Node Availability
# ==========================================================

def _node_available(
    G,
    node,
):
    """
    Check node availability.

    Both attributes are supported:

        available
        is_online
    """

    if node not in G:
        return False

    data = G.nodes[
        node
    ]

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
    """
    Convert value to finite float.

    Invalid or non-finite values return default.
    """

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
        ValueError,
    ):

        return float(
            default
        )


def _valid_number(
    value,
):
    """
    Check whether value is finite.
    """

    try:

        value = float(
            value
        )

        return math.isfinite(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def _valid_metric(
    value,
):
    """
    Check whether a metric is finite and non-negative.
    """

    try:

        value = float(
            value
        )

        return (
            math.isfinite(
                value
            )
            and
            value >= 0.0
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def _valid_cost(
    value,
):
    """
    Check whether routing cost is finite and non-negative.

    NetworkX weighted shortest-path algorithms require
    numerical edge weights and shortest_simple_paths does
    not support negative weights.
    """

    try:

        value = float(
            value
        )

        return (
            math.isfinite(
                value
            )
            and
            value >= 0.0
        )

    except (
        TypeError,
        ValueError,
    ):

        return False