"""
Pathfinding/top_k_paths.py

Channel-aware Top-K candidate routing for the adaptive
routing model.

Unified routing objective
-------------------------

    C'(e) =
        C_LND(e)
        *
        (
            1 + lambda_h * penalty(e, eta)
        )

The exact same adaptive_edge_cost() function used by
Dijkstra is used here.

Routing invariants
------------------

1. PPO controls eta only.
2. k is fixed at 5 in this routing model.
3. k is never modified internally.
4. Known learned/observed directional liquidity is a hard
   feasibility constraint.
5. Unknown directional liquidity is preserved as UNKNOWN
   and is resolved through payment simulation.
6. No silent fallback is used for routing metrics.
7. Exact physical channel identity is preserved.
8. MultiDiGraph parallel channels are never collapsed.
9. channel_key / SCID are preserved for every candidate edge.
10. Top-K candidates are channel-aware.
11. max_hops counts physical routing channels.
12. adaptive_edge_cost() is the single source of truth
    for adaptive routing cost.
13. Invalid Boolean state is rejected rather than coerced.
14. Missing required routing data is rejected.
15. Candidate ordering is deterministic.
16. Two paths with the same node sequence but different
    physical channels are distinct candidates.
17. Snapshot channel capacity is NEVER interpreted as
    directional liquidity.
18. Unknown directional liquidity NEVER causes a candidate
    to be rejected before payment simulation.
19. Explicit malformed directional-liquidity values are
    rejected rather than silently ignored.
20. No synthetic directional balance is created.
21. Historical snapshots without directional balances remain
    routable; actual forwarding feasibility is evaluated by
    the simulation/dynamic-state layer.
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
# Model Constants
# ==========================================================

DEFAULT_K = 5


# ==========================================================
# Directional Liquidity
# ==========================================================

def _estimated_liquidity(data):
    """
    Return a known empirical directional-liquidity estimate.

    Priority
    --------
        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    Semantics
    ---------
    A valid numeric value means that empirical directional
    transfer knowledge is currently available.

    None means that directional liquidity is UNKNOWN.

    UNKNOWN is NOT:

        - zero liquidity
        - unlimited liquidity
        - channel capacity
        - capacity / 2
        - an invalid channel

    Therefore, if all directional-liquidity fields are absent
    or explicitly set to None, this function returns None.

    Explicit malformed values are rejected.

    Channel ``capacity`` is deliberately NOT used as a
    replacement for directional liquidity.

    Actual transferable amount is learned from payment
    attempts and their observed outcomes.
    """

    liquidity_fields = (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv",
    )

    for field in liquidity_fields:

        if field not in data:
            continue

        value = data[field]

        # --------------------------------------------------
        # None means UNKNOWN.
        #
        # This is normal for historical Lightning snapshots
        # that do not expose directional balances.
        # --------------------------------------------------

        if value is None:
            continue

        # --------------------------------------------------
        # Explicitly supplied values must be valid.
        # --------------------------------------------------

        try:
            liquidity = float(value)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid liquidity value in field "
                f"{field!r}: {value!r}"
            ) from exc

        if not math.isfinite(liquidity):
            raise ValueError(
                f"Non-finite liquidity value in field "
                f"{field!r}: {value!r}"
            )

        if liquidity < 0.0:
            raise ValueError(
                f"Negative liquidity value in field "
                f"{field!r}: {value!r}"
            )

        return liquidity

    # ------------------------------------------------------
    # No directional liquidity is known.
    #
    # This is NOT an error.
    #
    # Do NOT:
    #
    #     return 0.0
    #     return capacity
    #     return capacity / 2
    #     raise ValueError
    #
    # The channel remains a candidate and its actual
    # forwarding capability is determined later.
    # ------------------------------------------------------

    return None


def _channel_can_carry(data, amount):
    """
    Determine whether a channel can remain a Top-K candidate.

    Known directional liquidity
    ----------------------------
    If empirical directional liquidity is known:

        liquidity >= amount
            -> candidate allowed

        liquidity < amount
            -> candidate rejected

    Unknown directional liquidity
    -----------------------------
    If directional liquidity is unknown:

        -> keep the channel as a candidate

    Unknown does NOT mean zero.

    The actual forwarding capability is determined later by
    Payment Simulation / Network Dynamics / FailureModel.

    Channel capacity is never used as a proxy for
    directional liquidity.
    """

    amount = _strict_positive_float(
        amount,
        "amount",
    )

    liquidity = _estimated_liquidity(data)

    # ------------------------------------------------------
    # Unknown liquidity:
    #
    # Do not eliminate the channel.
    # ------------------------------------------------------

    if liquidity is None:
        return True

    # ------------------------------------------------------
    # Known liquidity:
    #
    # Apply the observed/learned value as a hard
    # feasibility constraint.
    # ------------------------------------------------------

    return liquidity >= amount


# ==========================================================
# Channel Reliability
# ==========================================================

def _channel_reliability(data):
    """
    Calculate channel reliability.

    If empirical history exists:

        failure_probability =
            failure_count /
            (success_count + failure_count)

    If no empirical history exists, reliability is unknown and
    receives a neutral value. Simulator-only probabilities are
    deliberately not read by pathfinding.

    No probability is inferred from the simulator's latent
    distribution. No history means neutral reliability.
    """

    has_success = "success_count" in data
    has_failure = "failure_count" in data

    if has_success or has_failure:

        success = _strict_nonnegative_float(
            data.get("success_count", 0.0),
            "success_count",
        )

        failure = _strict_nonnegative_float(
            data.get("failure_count", 0.0),
            "failure_count",
        )

        total = success + failure

        if total > 0.0:
            failure_probability = failure / total
            return 1.0 - failure_probability

    return 1.0


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
    Wrapper around the unified adaptive edge cost.

    adaptive_edge_cost() remains the single source of truth.
    """

    if heuristic_fn is None:
        heuristic_fn = lnd_cost

    if not callable(heuristic_fn):
        raise TypeError(
            "heuristic_fn must be callable"
        )

    result = adaptive_edge_cost(
        G=G,
        u=u,
        v=v,
        data=data,
        amount=amount,
        eta=eta,
        heuristic_fn=heuristic_fn,
        lambda_h=lambda_h,
    )

    if not isinstance(result, dict):
        raise TypeError(
            "adaptive_edge_cost() must return a dict"
        )

    required = (
        "cost",
        "raw_heuristic",
        "adaptive_penalty",
    )

    missing = [
        field
        for field in required
        if field not in result
    ]

    if missing:
        raise KeyError(
            "adaptive_edge_cost() result is missing "
            f"fields: {missing}"
        )

    cost = result["cost"]
    raw_h = result["raw_heuristic"]
    penalty = result["adaptive_penalty"]

    if not _valid_cost(cost):
        raise ValueError(
            f"Invalid adaptive edge cost: {cost!r}"
        )

    if not _valid_number(raw_h):
        raise ValueError(
            f"Invalid adaptive heuristic: {raw_h!r}"
        )

    if not _valid_metric(penalty):
        raise ValueError(
            f"Invalid adaptive penalty: {penalty!r}"
        )

    return (
        float(cost),
        float(raw_h),
        float(penalty),
    )


# ==========================================================
# Edge Data Normalization
# ==========================================================

def _normalize_edge_data(data):
    """
    Normalize schema aliases only.

    No synthetic routing values are created.

    ``available`` is optional because an absent availability
    flag means that the snapshot does not explicitly provide
    an availability state.

    Directional liquidity is intentionally NOT synthesized
    from capacity.
    """

    data = dict(data)

    # ------------------------------------------------------
    # Fee base
    # ------------------------------------------------------

    if "fee_base" not in data:

        if "fee_base_msat" in data:
            data["fee_base"] = data["fee_base_msat"]

        else:
            raise KeyError(
                "Missing fee_base / fee_base_msat"
            )

    # ------------------------------------------------------
    # Fee rate
    # ------------------------------------------------------

    if "fee_rate" not in data:

        if "fee_proportional_millionths" in data:
            data["fee_rate"] = (
                data["fee_proportional_millionths"]
            )

        else:
            raise KeyError(
                "Missing fee_rate / "
                "fee_proportional_millionths"
            )

    # ------------------------------------------------------
    # Delay
    # ------------------------------------------------------

    if "delay" not in data:

        if "cltv_expiry_delta" in data:
            data["delay"] = data["cltv_expiry_delta"]

        else:
            raise KeyError(
                "Missing delay / cltv_expiry_delta"
            )

    return data


# ==========================================================
# Boolean Validation
# ==========================================================

def _strict_bool(
    value,
    field_name,
):
    """
    Accept actual Boolean values only.

    Examples rejected:

        "false"
        "true"
        0
        1
        None
    """

    if not isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a Boolean value, "
            f"got {value!r}"
        )

    return value


# ==========================================================
# Build Channel-Aware Routing Graph
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
    Build a channel-expanded directed graph.

    Each physical channel receives a unique virtual node:

        A
        |
        v
    [CHANNEL]
        |
        v
        B

    Parallel physical channels are therefore never collapsed.
    """

    if G is None:
        raise ValueError(
            "G must not be None"
        )

    if not isinstance(
        G,
        (
            nx.Graph,
            nx.DiGraph,
            nx.MultiGraph,
            nx.MultiDiGraph,
        ),
    ):
        raise TypeError(
            "G must be a NetworkX graph"
        )

    eta = validate_eta(eta)
    lambda_h = validate_lambda_h(lambda_h)

    amount = _strict_positive_float(
        amount,
        "amount",
    )

    max_hops = _strict_positive_int(
        max_hops,
        "max_hops",
    )

    H = nx.DiGraph()

    # ------------------------------------------------------
    # Preserve original nodes
    # ------------------------------------------------------

    for node, node_data in G.nodes(data=True):

        H.add_node(
            node,
            **dict(node_data),
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

        for (
            u,
            v,
            key,
            raw_data,
        ) in G.edges(
            keys=True,
            data=True,
        ):

            data = _normalize_edge_data(
                raw_data
            )

            _process_edge(
                H=H,
                G=G,
                u=u,
                v=v,
                key=key,
                data=data,
                amount=amount,
                eta=eta,
                heuristic_fn=heuristic_fn,
                lambda_h=lambda_h,
            )

    # ------------------------------------------------------
    # Simple Graph / DiGraph
    # ------------------------------------------------------

    else:

        for (
            u,
            v,
            raw_data,
        ) in G.edges(data=True):

            data = _normalize_edge_data(
                raw_data
            )

            key = data.get(
                "channel_key"
            )

            _process_edge(
                H=H,
                G=G,
                u=u,
                v=v,
                key=key,
                data=data,
                amount=amount,
                eta=eta,
                heuristic_fn=heuristic_fn,
                lambda_h=lambda_h,
            )

    return H


# ==========================================================
# Process One Physical Channel
# ==========================================================

def _process_edge(
    H,
    G,
    u,
    v,
    key,
    data,
    amount,
    eta,
    heuristic_fn,
    lambda_h,
):
    """
    Validate one physical channel and insert it into H.

    Unknown directional liquidity does not prevent the
    channel from becoming a routing candidate.
    """

    # ------------------------------------------------------
    # Channel availability
    # ------------------------------------------------------

    if "available" in data:

        available = _strict_bool(
            data["available"],
            "available",
        )

        if not available:
            return

    # ------------------------------------------------------
    # Node availability
    # ------------------------------------------------------

    if not _node_available(
        G,
        u,
    ):
        return

    if not _node_available(
        G,
        v,
    ):
        return

    # ------------------------------------------------------
    # Directional liquidity
    # ------------------------------------------------------
    #
    # Known liquidity:
    #     hard feasibility constraint.
    #
    # Unknown liquidity:
    #     preserve channel as candidate.
    #
    # Capacity is NOT used here.
    # ------------------------------------------------------

    if not _channel_can_carry(
        data,
        amount,
    ):
        return

    # ------------------------------------------------------
    # Adaptive cost
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Fee
    # ------------------------------------------------------

    fee = channel_fee(
        data,
        amount,
    )

    if not _valid_metric(fee):
        raise ValueError(
            f"Invalid fee on edge "
            f"{u}->{v}, key={key!r}: {fee!r}"
        )

    # ------------------------------------------------------
    # Delay
    # ------------------------------------------------------

    delay = channel_delay(
        data
    )

    if not _valid_metric(delay):
        raise ValueError(
            f"Invalid delay on edge "
            f"{u}->{v}, key={key!r}: {delay!r}"
        )

    # ------------------------------------------------------
    # Reliability
    # ------------------------------------------------------

    reliability = _channel_reliability(
        data
    )

    # ------------------------------------------------------
    # Channel identity
    # ------------------------------------------------------

    scid = data.get(
        "scid",
        data.get(
            "short_channel_id"
        ),
    )

    # ------------------------------------------------------
    # Store physical channel attributes.
    # ------------------------------------------------------

    channel_attributes = {
        "channel_key": key,
        "scid": scid,
        "channel_data": {
            name: value
            for name, value in data.items()
            if name not in {
                "failure_probability",
                "simulator_failure_probability",
            }
        },
        "fee": float(fee),
        "delay": float(delay),
        "reliability": float(reliability),
        "eta": float(eta),
        "raw_heuristic": float(raw_h),
        "adaptive_penalty": float(
            adaptive_penalty
        ),
        "weight": float(weight),
    }

    # ------------------------------------------------------
    # Unique virtual channel node
    # ------------------------------------------------------
    #
    # Every physical channel receives its own virtual node.
    # Therefore parallel channels remain distinguishable.
    # ------------------------------------------------------

    virtual_node = (
        "__channel__",
        id(H),
        len(H),
        u,
        v,
        _channel_key_sort_value(key),
        scid,
    )

    while virtual_node in H:

        virtual_node = (
            "__channel__",
            id(H),
            len(H),
            u,
            v,
            _channel_key_sort_value(key),
            scid,
            len(H.nodes),
        )

    H.add_node(
        virtual_node,
        virtual_channel=True,
        channel_key=key,
        scid=scid,
        source=u,
        target=v,
        channel_data=channel_attributes["channel_data"],
    )

    # ------------------------------------------------------
    # First half of physical channel
    # ------------------------------------------------------

    H.add_edge(
        u,
        virtual_node,
        weight=float(weight),
        physical_channel=True,
        channel_key=key,
        scid=scid,
        channel_data=dict(data),
        fee=float(fee),
        delay=float(delay),
        reliability=float(reliability),
        eta=float(eta),
        raw_heuristic=float(raw_h),
        adaptive_penalty=float(
            adaptive_penalty
        ),
        hop_increment=1,
    )

    # ------------------------------------------------------
    # Second half of physical channel
    # ------------------------------------------------------

    H.add_edge(
        virtual_node,
        v,
        weight=0.0,
        physical_channel=False,
        channel_key=key,
        scid=scid,
        channel_data=dict(data),
        fee=0.0,
        delay=0.0,
        reliability=1.0,
        eta=float(eta),
        raw_heuristic=0.0,
        adaptive_penalty=0.0,
        hop_increment=0,
    )


# ==========================================================
# Extract Physical Path
# ==========================================================

def _extract_physical_path(
    H,
    expanded_path,
    source,
    target,
):
    """
    Convert an expanded path into:

        physical node path
        physical channel edges
    """

    if not expanded_path:
        raise RuntimeError(
            "Expanded path is empty"
        )

    if expanded_path[0] != source:
        raise RuntimeError(
            "Expanded path does not start at source"
        )

    if expanded_path[-1] != target:
        raise RuntimeError(
            "Expanded path does not terminate at target"
        )

    physical_nodes = [source]
    physical_edges = []

    index = 0
    current_source = source

    while index < len(expanded_path) - 1:

        virtual_node = expanded_path[
            index + 1
        ]

        if not _is_virtual_channel_node(
            H,
            virtual_node,
        ):
            raise RuntimeError(
                "Expanded path contains an unexpected "
                "non-channel node"
            )

        if not H.has_edge(
            current_source,
            virtual_node,
        ):
            raise RuntimeError(
                "Missing source-to-channel edge"
            )

        edge_to_virtual = H[
            current_source
        ][
            virtual_node
        ]

        channel_key = edge_to_virtual.get(
            "channel_key"
        )

        scid = edge_to_virtual.get(
            "scid"
        )

        channel_data = dict(
            edge_to_virtual.get(
                "channel_data",
                {},
            )
        )

        if index + 2 >= len(
            expanded_path
        ):
            raise RuntimeError(
                "Virtual channel node has no destination"
            )

        destination = expanded_path[
            index + 2
        ]

        if not H.has_edge(
            virtual_node,
            destination,
        ):
            raise RuntimeError(
                "Missing channel-to-destination edge"
            )

        destination_edge = H[
            virtual_node
        ][
            destination
        ]

        if destination_edge.get(
            "channel_key"
        ) != channel_key:

            raise RuntimeError(
                "Channel identity mismatch between "
                "expanded edges"
            )

        if destination_edge.get(
            "scid"
        ) != scid:

            raise RuntimeError(
                "SCID mismatch between "
                "expanded channel edges"
            )

        physical_edges.append(
            {
                "source": current_source,
                "target": destination,
                "channel_key": channel_key,
                "scid": scid,
                "data": channel_data,
            }
        )

        physical_nodes.append(
            destination
        )

        current_source = destination

        index += 2

    if physical_nodes[-1] != target:
        raise RuntimeError(
            "Physical path does not terminate at target"
        )

    if len(physical_nodes) != (
        len(physical_edges) + 1
    ):
        raise RuntimeError(
            "Physical path and channel-edge counts "
            "are inconsistent"
        )

    return (
        physical_nodes,
        physical_edges,
    )


# ==========================================================
# Path Metrics
# ==========================================================

def _path_metrics(
    H,
    expanded_path,
):
    """
    Calculate metrics from physical channel edges.

    Virtual edges contribute zero additional fee, delay,
    reliability penalty, or hop count.
    """

    total_cost = 0.0
    total_fee = 0.0
    total_delay = 0.0

    reliability = 1.0

    total_raw_heuristic = 0.0
    total_adaptive_penalty = 0.0

    hop_count = 0

    for u, v in zip(
        expanded_path[:-1],
        expanded_path[1:],
    ):

        if not H.has_edge(
            u,
            v,
        ):
            raise KeyError(
                f"Missing expanded edge "
                f"{u!r}->{v!r}"
            )

        edge = H[u][v]

        weight = _strict_nonnegative_float(
            edge["weight"],
            "edge weight",
        )

        total_cost += weight

        if edge.get(
            "physical_channel",
            False,
        ):

            total_fee += _strict_nonnegative_float(
                edge["fee"],
                "edge fee",
            )

            total_delay += _strict_nonnegative_float(
                edge["delay"],
                "edge delay",
            )

            edge_reliability = _strict_probability(
                edge["reliability"],
                "edge reliability",
            )

            reliability *= edge_reliability

            total_raw_heuristic += _strict_number(
                edge["raw_heuristic"],
                "raw heuristic",
            )

            total_adaptive_penalty += (
                _strict_nonnegative_float(
                    edge["adaptive_penalty"],
                    "adaptive penalty",
                )
            )

            hop_count += 1

    return {
        "cost": float(total_cost),
        "total_fee": float(total_fee),
        "total_delay": float(total_delay),
        "reliability": float(reliability),
        "failure_probability": float(
            1.0 - reliability
        ),
        "raw_heuristic": float(
            total_raw_heuristic
        ),
        "adaptive_penalty": float(
            total_adaptive_penalty
        ),
        "hop_count": int(hop_count),
    }


# ==========================================================
# Candidate Validation
# ==========================================================

def _valid_candidate_path(
    physical_path,
    source,
    target,
    max_hops,
):
    """
    Validate a physical node path.
    """

    if not isinstance(
        physical_path,
        (list, tuple),
    ):
        return False

    if len(physical_path) < 2:
        return False

    if physical_path[0] != source:
        return False

    if physical_path[-1] != target:
        return False

    hops = len(physical_path) - 1

    if hops <= 0:
        return False

    if hops > max_hops:
        return False

    # ------------------------------------------------------
    # Physical node path must be simple.
    # ------------------------------------------------------

    try:

        if len(set(physical_path)) != len(
            physical_path
        ):
            return False

    except TypeError:
        return False

    return True


# ==========================================================
# Candidate Identity
# ==========================================================

def _candidate_channel_identity(
    physical_edges,
):
    """
    Return deterministic physical-channel identity.

    Channel identity contains:

        source
        target
        channel_key
        SCID
    """

    return tuple(
        (
            edge["source"],
            edge["target"],
            _channel_key_sort_value(
                edge["channel_key"]
            ),
            str(edge["scid"]),
        )
        for edge in physical_edges
    )


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
    k=DEFAULT_K,
    max_hops=12,
    lambda_h=1.0,
):
    """
    Generate up to K channel-aware candidate paths.

    This routing model requires:

        k = 5

    Candidate structure:

        {
            "path": [...],
            "edges": [...],
            "cost": ...,
            "hop_count": ...,
            "total_fee": ...,
            "total_delay": ...,
            "reliability": ...,
            "failure_probability": ...,
            "eta": ...,
            "lambda_h": ...,
            "raw_heuristic": ...,
            "adaptive_penalty": ...,
            "candidate": True,
            "success": None,
        }

    Liquidity behavior
    ------------------
    Known empirical liquidity may exclude a channel when the
    requested amount is above the known transferable amount.

    Unknown liquidity does NOT exclude a channel.

    Actual forwarding capability is determined by the
    simulation/dynamic-state layer.
    """

    # ------------------------------------------------------
    # Graph validation
    # ------------------------------------------------------

    if G is None:
        raise ValueError(
            "G must not be None"
        )

    if not isinstance(
        G,
        (
            nx.Graph,
            nx.DiGraph,
            nx.MultiGraph,
            nx.MultiDiGraph,
        ),
    ):
        raise TypeError(
            "G must be a NetworkX graph"
        )

    # ------------------------------------------------------
    # Node validation
    # ------------------------------------------------------

    if source not in G:
        raise ValueError(
            f"Unknown source node: {source}"
        )

    if target not in G:
        raise ValueError(
            f"Unknown target node: {target}"
        )

    if source == target:
        return []

    # ------------------------------------------------------
    # Input validation
    # ------------------------------------------------------

    amount = _strict_positive_float(
        amount,
        "amount",
    )

    k = _strict_positive_int(
        k,
        "k",
    )

    if k != DEFAULT_K:
        raise ValueError(
            f"This routing model requires "
            f"k={DEFAULT_K}; received k={k}"
        )

    max_hops = _strict_positive_int(
        max_hops,
        "max_hops",
    )

    eta = validate_eta(
        eta
    )

    lambda_h = validate_lambda_h(
        lambda_h
    )

    if heuristic_fn is None:
        heuristic_fn = lnd_cost

    if not callable(heuristic_fn):
        raise TypeError(
            "heuristic_fn must be callable"
        )

    # ------------------------------------------------------
    # Build channel-expanded graph
    # ------------------------------------------------------

    H = _build_routing_graph(
        G=G,
        amount=amount,
        eta=eta,
        max_hops=max_hops,
        heuristic_fn=heuristic_fn,
        lambda_h=lambda_h,
    )

    if source not in H or target not in H:
        return []

    # ------------------------------------------------------
    # Enumerate shortest simple expanded paths
    # ------------------------------------------------------

    path_generator = nx.shortest_simple_paths(
        H,
        source,
        target,
        weight="weight",
    )

    results = []
    seen_channel_paths = set()

    try:

        for expanded_path in path_generator:

            (
                physical_path,
                physical_edges,
            ) = _extract_physical_path(
                H=H,
                expanded_path=expanded_path,
                source=source,
                target=target,
            )

            if not _valid_candidate_path(
                physical_path,
                source,
                target,
                max_hops,
            ):
                continue

            channel_identity = (
                _candidate_channel_identity(
                    physical_edges
                )
            )

            if channel_identity in seen_channel_paths:
                continue

            seen_channel_paths.add(
                channel_identity
            )

            metrics = _path_metrics(
                H,
                expanded_path,
            )

            if metrics["hop_count"] != len(
                physical_edges
            ):
                raise RuntimeError(
                    "Physical hop count does not match "
                    "channel-edge count"
                )

            candidate = {
                "path": list(
                    physical_path
                ),
                "edges": physical_edges,
                "cost": metrics["cost"],
                "hop_count": metrics["hop_count"],
                "total_fee": metrics["total_fee"],
                "total_delay": metrics["total_delay"],
                "reliability": metrics["reliability"],
                "failure_probability": (
                    metrics["failure_probability"]
                ),
                "eta": float(eta),
                "lambda_h": float(lambda_h),
                "raw_heuristic": (
                    metrics["raw_heuristic"]
                ),
                "adaptive_penalty": (
                    metrics["adaptive_penalty"]
                ),
                "candidate": True,
                "success": None,
            }

            results.append(
                candidate
            )

            # --------------------------------------------------
            # Fixed Top-K = 5
            # --------------------------------------------------

            if len(results) >= k:
                break

    except nx.NetworkXNoPath:
        # --------------------------------------------------
        # No path is a legitimate routing result.
        # --------------------------------------------------

        pass

    except nx.NodeNotFound as exc:

        raise RuntimeError(
            "Expanded routing graph lost a required node "
            "during candidate generation"
        ) from exc

    # ------------------------------------------------------
    # Deterministic final ordering
    # ------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["cost"],
            item["hop_count"],
            item["total_fee"],
            item["total_delay"],
            -item["reliability"],
            tuple(item["path"]),
            tuple(
                (
                    edge["source"],
                    edge["target"],
                    _channel_key_sort_value(
                        edge["channel_key"]
                    ),
                    str(edge["scid"]),
                )
                for edge in item["edges"]
            ),
        )
    )

    return results


# ==========================================================
# Virtual Channel Node Detection
# ==========================================================

def _is_virtual_channel_node(
    H,
    node,
):
    """
    Determine whether a node is a channel-expansion node.
    """

    if node not in H:
        return False

    return bool(
        H.nodes[node].get(
            "virtual_channel",
            False,
        )
    )


# ==========================================================
# Node Availability
# ==========================================================

def _node_available(
    G,
    node,
):
    """
    Check explicitly provided node-state flags.

    Supported flags:

        available
        is_online
        online

    Missing flags are not interpreted as failure.

    Invalid Boolean values are rejected.
    """

    if node not in G:
        return False

    data = G.nodes[node]

    for field in (
        "available",
        "is_online",
        "online",
    ):

        if field not in data:
            continue

        value = _strict_bool(
            data[field],
            field,
        )

        if not value:
            return False

    return True


# ==========================================================
# Numeric Helpers
# ==========================================================

def _strict_number(
    value,
    field_name,
):
    """
    Convert a numeric value to float and validate finiteness.
    """

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{field_name} must be numeric: "
            f"{value!r}"
        ) from exc

    if not math.isfinite(value):

        raise ValueError(
            f"{field_name} must be finite: "
            f"{value!r}"
        )

    return value


def _strict_nonnegative_float(
    value,
    field_name,
):
    """
    Validate a finite non-negative floating-point value.
    """

    value = _strict_number(
        value,
        field_name,
    )

    if value < 0.0:

        raise ValueError(
            f"{field_name} must be non-negative: "
            f"{value!r}"
        )

    return value


def _strict_positive_float(
    value,
    field_name,
):
    """
    Validate a finite positive floating-point value.
    """

    value = _strict_number(
        value,
        field_name,
    )

    if value <= 0.0:

        raise ValueError(
            f"{field_name} must be greater than zero: "
            f"{value!r}"
        )

    return value


def _strict_positive_int(
    value,
    field_name,
):
    """
    Accept:

        5
        5.0

    Reject:

        True
        False
        5.7
        "5"
        0
        -1
        NaN
        inf
    """

    if isinstance(value, bool):

        raise ValueError(
            f"{field_name} must be an integer: "
            f"{value!r}"
        )

    if isinstance(value, float):

        if not math.isfinite(value):

            raise ValueError(
                f"{field_name} must be a finite integer: "
                f"{value!r}"
            )

        if not value.is_integer():

            raise ValueError(
                f"{field_name} must be an integer: "
                f"{value!r}"
            )

    if isinstance(value, str):

        raise ValueError(
            f"{field_name} must be an integer: "
            f"{value!r}"
        )

    try:

        integer = int(value)

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:

        raise ValueError(
            f"{field_name} must be an integer: "
            f"{value!r}"
        ) from exc

    if integer <= 0:

        raise ValueError(
            f"{field_name} must be greater than zero: "
            f"{value!r}"
        )

    return integer


def _strict_probability(
    value,
    field_name,
):
    """
    Validate a probability in [0, 1].
    """

    value = _strict_number(
        value,
        field_name,
    )

    if value < 0.0 or value > 1.0:

        raise ValueError(
            f"{field_name} must be in [0, 1]: "
            f"{value!r}"
        )

    return value


def _valid_number(
    value,
):
    """
    Return True only for finite numeric values.
    """

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return False

    return math.isfinite(value)


def _valid_metric(
    value,
):
    """
    Return True only for finite non-negative metrics.
    """

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return False

    return (
        math.isfinite(value)
        and value >= 0.0
    )


def _valid_cost(
    value,
):
    """
    Return True only for finite non-negative costs.
    """

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return False

    return (
        math.isfinite(value)
        and value >= 0.0
    )


# ==========================================================
# Deterministic Channel-Key Ordering
# ==========================================================

def _channel_key_sort_value(
    key,
):
    """
    Convert arbitrary NetworkX channel keys into a
    deterministic comparable representation.
    """

    if key is None:

        return (
            0,
            "",
        )

    return (
        1,
        str(key),
    )
