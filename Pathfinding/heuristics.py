"""
Pathfinding/heuristics.py

Routing heuristics for Lightning-style payment networks.

Core adaptive-cost model
-------------------------

    Geographic dataset
      |
      v
    Node carbon intensity
      |
      v
    h(u,v,eta)
      |
      v
    C'(u,v) = max(C_native(u,v) + h(u,v,eta), 0)


Adaptive heuristic
------------------

    h(u,v,eta)
        =
        eta * ((C_u + C_v) / 2)
        +
        (1 - eta) * (C_v - C_u)


Carbon intensity
----------------

Carbon intensity is supplied as the node attribute
``carbon_intensity`` by the geographic data preparation layer.
RGB/color metadata is not interpreted as carbon data here.


Routing objective
-----------------

    C_LND = fee + 0.5 * delay + 1

    C_adaptive = max(C_native + h, 0)


Liquidity
---------

Liquidity is NOT part of the additive routing cost.

It is a hard feasibility constraint handled by the pathfinding
layer:

    liquidity < amount
        -> edge is infeasible


Scientific-evaluation policy
----------------------------

This module does not fabricate missing routing or liquidity
values.

Carbon intensity must be present on nodes before adaptive
routing is used. Missing carbon data raises an error.

For descriptive path evaluation, optional metrics such as
liquidity and failure probability are used only when explicitly
available.

No artificial liquidity or reliability value is inserted when
those metrics are absent.
"""

import math


# ==========================================================
# Numeric Helpers
# ==========================================================

def _to_float(value, name):
    """
    Convert value to a finite float.

    Invalid values are never silently converted to zero.
    """

    try:
        result = float(value)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f"{name} must be numeric; got {value!r}."
        ) from exc

    if not math.isfinite(result):

        raise ValueError(
            f"{name} must be finite; got {value!r}."
        )

    return result


def _valid_number(value):
    """
    Return True if value is numeric and finite.
    """

    try:
        result = float(value)

    except (TypeError, ValueError):

        return False

    return math.isfinite(result)


def _valid_nonnegative(value):
    """
    Return True if value is finite and >= 0.
    """

    try:
        result = float(value)

    except (TypeError, ValueError):

        return False

    return (
        math.isfinite(result)
        and result >= 0.0
    )


# ==========================================================
# ETA Validation
# ==========================================================

def validate_eta(eta):
    """
    Validate PPO-controlled eta.

    Required range from the paper:

        -1 <= eta <= 1

    No clipping is performed.
    """

    try:
        eta = float(eta)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "eta must be numeric."
        ) from exc

    if not math.isfinite(eta):

        raise ValueError(
            "eta must be finite."
        )

    if not -1.0 <= eta <= 1.0:

        raise ValueError(
            "eta must satisfy -1 <= eta <= 1."
        )

    return eta


# ==========================================================
# Lambda Validation
# ==========================================================

def validate_lambda_h(lambda_h):
    """
    Validate adaptive-cost coefficient.

    Required:

        lambda_h >= 0
    """

    try:
        lambda_h = float(lambda_h)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "lambda_h must be numeric."
        ) from exc

    if not math.isfinite(lambda_h):

        raise ValueError(
            "lambda_h must be finite."
        )

    if lambda_h < 0.0:

        raise ValueError(
            "lambda_h must be >= 0."
        )

    return lambda_h


# ==========================================================
# Required Field Helper
# ==========================================================

def _get_required_field(data, primary, alias):
    """
    Read a required field using one primary name and one alias.

    No hidden default is allowed.
    """

    if not isinstance(data, dict):

        raise TypeError(
            "Channel data must be a dictionary."
        )

    if primary in data:

        return data[primary]

    if alias in data:

        return data[alias]

    raise KeyError(
        f"Missing required channel field "
        f"'{primary}'/'{alias}'."
    )


# ==========================================================
# Channel Fee
# ==========================================================

def channel_fee(data, amount):
    """
    Calculate forwarding fee.

    Formula:

        fee =
            base_fee
            +
            amount * fee_rate / 1,000,000

    Amount and fee fields must use the same graph unit.
    """

    if not isinstance(data, dict):

        raise TypeError(
            "Channel data must be a dictionary."
        )

    amount = _to_float(
        amount,
        "amount",
    )

    if amount < 0.0:

        raise ValueError(
            "amount must be >= 0."
        )

    base_fee = _to_float(
        _get_required_field(
            data,
            "fee_base",
            "fee_base_msat",
        ),
        "base_fee",
    )

    fee_rate = _to_float(
        _get_required_field(
            data,
            "fee_rate",
            "fee_proportional_millionths",
        ),
        "fee_rate",
    )

    if base_fee < 0.0:

        raise ValueError(
            f"base_fee must be >= 0; got {base_fee}."
        )

    if fee_rate < 0.0:

        raise ValueError(
            f"fee_rate must be >= 0; got {fee_rate}."
        )

    fee = (
        base_fee
        + amount * fee_rate / 1_000_000.0
    )

    if not _valid_nonnegative(fee):

        raise ValueError(
            "Calculated channel fee is invalid."
        )

    return float(fee)


# ==========================================================
# Channel Delay
# ==========================================================

def channel_delay(data):
    """
    Extract channel forwarding delay.

    Supported fields:

        delay
        cltv_expiry_delta
    """

    if not isinstance(data, dict):

        raise TypeError(
            "Channel data must be a dictionary."
        )

    if "delay" in data:

        value = data["delay"]

    elif "cltv_expiry_delta" in data:

        value = data["cltv_expiry_delta"]

    else:

        raise KeyError(
            "Missing required channel delay field "
            "'delay'/'cltv_expiry_delta'."
        )

    delay = _to_float(
        value,
        "delay",
    )

    if delay < 0.0:

        raise ValueError(
            f"delay must be >= 0; got {delay}."
        )

    return float(delay)


# ==========================================================
# Native LND Cost
# ==========================================================

def lnd_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    Native routing cost.

    Formula:

        C_LND = fee + 0.5 * delay + 1

    The graph and endpoint arguments are retained for API
    compatibility with the routing pipeline.
    """

    del G
    del u
    del v

    fee = channel_fee(
        data,
        amount,
    )

    delay = channel_delay(
        data,
    )

    cost = (
        fee
        + 0.5 * delay
        + 1.0
    )

    if not _valid_nonnegative(cost):

        raise ValueError(
            "Native LND cost is invalid."
        )

    return float(cost)


# ==========================================================
# RGB Parsing
# ==========================================================

def _parse_rgb(rgb, node):
    """
    Parse RGB data into:

        (R, G, B)

    Supported forms:

        [R, G, B]
        (R, G, B)

        {"r": R, "g": G, "b": B}
        {"red": R, "green": G, "blue": B}

        "#RRGGBB"
        "RRGGBB"

    All components must be in [0,255].

    Missing RGB is handled by node_carbon_intensity()
    before this function is called.

    Therefore this function deliberately rejects None
    and other unsupported representations instead of
    silently converting malformed data to zero.
    """

    if isinstance(rgb, dict):

        if all(
            key in rgb
            for key in ("r", "g", "b")
        ):

            values = [
                rgb["r"],
                rgb["g"],
                rgb["b"],
            ]

        elif all(
            key in rgb
            for key in ("red", "green", "blue")
        ):

            values = [
                rgb["red"],
                rgb["green"],
                rgb["blue"],
            ]

        else:

            raise ValueError(
                f"RGB dictionary for node {node!r} must contain "
                f"either r/g/b or red/green/blue; got {rgb!r}."
            )

        values = [
            _to_float(
                value,
                f"RGB component {index}",
            )
            for index, value in enumerate(values)
        ]

    elif isinstance(rgb, (list, tuple)):

        if len(rgb) < 3:

            raise ValueError(
                f"RGB data for node {node!r} must contain "
                f"at least three values; got {rgb!r}."
            )

        values = [
            _to_float(
                rgb[index],
                f"RGB component {index}",
            )
            for index in range(3)
        ]

    elif isinstance(rgb, str):

        text = rgb.strip()

        if text.startswith("#"):

            text = text[1:]

        if len(text) != 6:

            raise ValueError(
                f"RGB string for node {node!r} must contain "
                f"exactly 6 hexadecimal characters; got {rgb!r}."
            )

        try:

            values = [
                float(
                    int(
                        text[0:2],
                        16,
                    )
                ),
                float(
                    int(
                        text[2:4],
                        16,
                    )
                ),
                float(
                    int(
                        text[4:6],
                        16,
                    )
                ),
            ]

        except ValueError as exc:

            raise ValueError(
                f"Invalid hexadecimal RGB value for "
                f"node {node!r}: {rgb!r}."
            ) from exc

    else:

        raise TypeError(
            f"Unsupported RGB representation for node "
            f"{node!r}: {type(rgb).__name__}."
        )

    for index, value in enumerate(values):

        if not 0.0 <= value <= 255.0:

            raise ValueError(
                f"RGB component {index} for node {node!r} "
                f"must be in [0,255]; got {value}."
            )

    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )


# ==========================================================
# Carbon / RGB Luminance
# ==========================================================

def node_carbon_intensity(
    G,
    node,
):
    """
    Read the geographic carbon intensity assigned to a node.

    RGB/color attributes are deliberately ignored. The dataset
    loader is responsible for assigning ``carbon_intensity``.
    """

    if node not in G:

        raise KeyError(
            f"Node {node!r} does not exist in graph."
        )

    node_data = G.nodes[node]

    if "carbon_intensity" not in node_data:
        raise KeyError(
            f"Node {node!r} has no geographic 'carbon_intensity'. "
            "Load and map the carbon dataset before routing."
        )

    carbon = _to_float(
        node_data["carbon_intensity"],
        f"carbon_intensity for node {node!r}",
    )

    if carbon < 0.0:
        raise ValueError(
            f"carbon_intensity for node {node!r} must be >= 0; "
            f"got {carbon}."
        )

    return carbon


# ==========================================================
# Adaptive Heuristic
# ==========================================================

def adaptive_heuristic(
    G,
    u,
    v,
    eta,
):
    """
    Calculate the raw adaptive routing signal.

    Formula:

        h(u,v,eta)
            =
            eta * ((C_u + C_v) / 2)
            +
            (1 - eta) * (C_v - C_u)

    Endpoint behavior:

        eta = 0:
            h = C_v - C_u

        eta = 1:
            h = (C_u + C_v) / 2
    """

    eta = validate_eta(
        eta,
    )

    cu = node_carbon_intensity(
        G,
        u,
    )

    cv = node_carbon_intensity(
        G,
        v,
    )

    average_component = (
        cu + cv
    ) / 2.0

    transition_component = (
        cv - cu
    )

    value = (
        eta * average_component
        +
        (1.0 - eta) * transition_component
    )

    if not _valid_number(value):

        raise ValueError(
            "Adaptive heuristic is not finite."
        )

    return float(value)


# ==========================================================
# Adaptive Penalty
# ==========================================================

def adaptive_penalty(
    G,
    u,
    v,
    eta,
):
    """
    Compatibility helper returning the signed paper heuristic.

    New routing code should use ``raw_heuristic`` directly.
    """

    raw_h = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    return float(raw_h)


# ==========================================================
# Modified Cost
# ==========================================================

def modified_cost(
    native_cost,
    geo_penalty,
    eta=None,
    lambda_h=1.0,
):
    """
    Add the signed paper heuristic to native routing cost.

    The non-negative clamp keeps the edge weight valid for
    Dijkstra while retaining the raw signed heuristic in the
    diagnostics.
    """

    if not _valid_nonnegative(
        native_cost,
    ):

        raise ValueError(
            "native_cost must be finite and >= 0."
        )

    if not _valid_number(geo_penalty):
        raise ValueError(
            "geo_penalty must be finite."
        )

    geo_penalty = float(geo_penalty)

    if eta is not None:

        validate_eta(
            eta,
        )

    lambda_h = validate_lambda_h(
        lambda_h,
    )

    cost = max(
        float(native_cost) + geo_penalty,
        0.0,
    )

    if not _valid_nonnegative(cost):

        raise ValueError(
            "Modified adaptive cost is invalid."
        )

    return float(cost)


# ==========================================================
# Unified Adaptive Edge Cost
# ==========================================================

def adaptive_edge_cost(
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
    Unified adaptive edge-cost calculation.

    Returns:

        {
            "native_cost": ...,
            "raw_heuristic": ...,
            "adaptive_penalty": ...,
            "cost": ...
        }

    A failing custom heuristic is never silently replaced.
    """

    eta = validate_eta(
        eta,
    )

    # lambda_h is retained in the signature for compatibility.
    validate_lambda_h(lambda_h)

    if heuristic_fn is None:

        native_cost = lnd_cost(
            G,
            u,
            v,
            data,
            amount,
        )

    else:

        if not callable(heuristic_fn):

            raise TypeError(
                "heuristic_fn must be callable."
            )

        native_cost = heuristic_fn(
            G,
            u,
            v,
            data,
            amount,
        )

        if not _valid_nonnegative(
            native_cost,
        ):

            raise ValueError(
                "heuristic_fn returned an invalid "
                "native routing cost."
            )

        native_cost = float(
            native_cost
        )

    raw_h = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    cost = modified_cost(
        native_cost=native_cost,
        geo_penalty=raw_h,
        eta=eta,
        lambda_h=lambda_h,
    )

    return {
        "native_cost": float(native_cost),
        "raw_heuristic": float(raw_h),
        "adaptive_penalty": float(max(raw_h, 0.0)),
        "cost": float(cost),
    }


# ==========================================================
# Adaptive LND Cost
# ==========================================================

def adaptive_lnd_cost(
    G,
    u,
    v,
    data,
    amount,
    eta=0.0,
    lambda_h=1.0,
):
    """
    Convenience wrapper around adaptive_edge_cost()
    using native LND cost.
    """

    result = adaptive_edge_cost(
        G=G,
        u=u,
        v=v,
        data=data,
        amount=amount,
        eta=eta,
        heuristic_fn=lnd_cost,
        lambda_h=lambda_h,
    )

    return float(
        result["cost"]
    )


# ==========================================================
# Enhanced Diagnostic Cost
# ==========================================================

def enhanced_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    Auxiliary diagnostic cost.

    Formula:

        fee
        +
        0.5 * delay
        +
        failure_probability
        +
        1

    This is NOT the PPO-controlled objective.
    """

    del G
    del u
    del v

    if not isinstance(data, dict):

        raise TypeError(
            "Channel data must be a dictionary."
        )

    fee = channel_fee(
        data,
        amount,
    )

    delay = channel_delay(
        data,
    )

    if "failure_probability" not in data:

        raise KeyError(
            "Missing 'failure_probability' for enhanced_cost()."
        )

    failure_probability = _to_float(
        data["failure_probability"],
        "failure_probability",
    )

    if not 0.0 <= failure_probability <= 1.0:

        raise ValueError(
            "failure_probability must be in [0,1]."
        )

    cost = (
        fee
        + 0.5 * delay
        + failure_probability
        + 1.0
    )

    if not _valid_nonnegative(cost):

        raise ValueError(
            "Enhanced cost is invalid."
        )

    return float(cost)


# ==========================================================
# MultiGraph Detection
# ==========================================================

def _is_multigraph(G):
    """
    Return True if G is a NetworkX MultiGraph/MultiDiGraph.
    """

    if hasattr(G, "is_multigraph"):

        return bool(
            G.is_multigraph()
        )

    return False


# ==========================================================
# Path Edge Resolution
# ==========================================================

def _resolve_path_edge(
    G,
    u,
    v,
    edge_key=None,
):
    """
    Resolve the exact physical edge used by a path.

    MultiGraph:

        edge_key supplied
            -> exact channel

        edge_key omitted
            -> allowed only if exactly one channel exists

    No arbitrary channel is selected.
    """

    if not G.has_edge(u, v):

        raise KeyError(
            f"Path contains missing edge ({u!r}, {v!r})."
        )

    if _is_multigraph(G):

        channels = G.get_edge_data(
            u,
            v,
        )

        if channels is None:

            raise KeyError(
                f"No edge data for ({u!r}, {v!r})."
            )

        if edge_key is not None:

            if edge_key not in channels:

                raise KeyError(
                    f"Edge key {edge_key!r} does not exist "
                    f"for ({u!r}, {v!r})."
                )

            edge_data = channels[
                edge_key
            ]

            if not isinstance(
                edge_data,
                dict,
            ):

                raise TypeError(
                    f"Channel data for "
                    f"({u!r}, {v!r}, {edge_key!r}) "
                    f"must be a dictionary."
                )

            return edge_data

        keys = list(
            channels.keys()
        )

        if len(keys) != 1:

            raise ValueError(
                f"Multiple channels exist between "
                f"({u!r}, {v!r}), but no edge key "
                f"was supplied."
            )

        edge_data = channels[
            keys[0]
        ]

        if not isinstance(
            edge_data,
            dict,
        ):

            raise TypeError(
                f"Channel data for "
                f"({u!r}, {v!r}) must be a dictionary."
            )

        return edge_data

    if edge_key is not None:

        raise ValueError(
            f"edge_key={edge_key!r} supplied for "
            f"a non-multigraph."
        )

    edge_data = G.get_edge_data(
        u,
        v,
    )

    if not isinstance(
        edge_data,
        dict,
    ):

        raise TypeError(
            f"Edge data for ({u!r}, {v!r}) "
            f"must be a dictionary."
        )

    return edge_data


# ==========================================================
# Path Representation Helpers
# ==========================================================

def _is_single_edge_tuple(G, path):
    """
    Determine whether a bare 3-tuple represents one exact
    MultiGraph edge:

        (u, v, edge_key)

    This prevents ambiguity between:

        ("A", "B", "C")

    as a node path and:

        ("A", "B", "AB-1")

    as one exact channel.

    A bare edge tuple is treated as edge-aware only when the
    graph confirms that the third item is an actual edge key.
    """

    if not isinstance(
        path,
        tuple,
    ):

        return False

    if len(path) != 3:

        return False

    if not _is_multigraph(G):

        return False

    u, v, key = path

    if not G.has_edge(
        u,
        v,
    ):

        return False

    channels = G.get_edge_data(
        u,
        v,
    )

    if not isinstance(
        channels,
        dict,
    ):

        return False

    return key in channels


# ==========================================================
# Path Transition Builder
# ==========================================================

def _build_path_transitions(
    G,
    path,
):
    """
    Convert path representation into:

        (u, v, edge_key)

    Supported representations:

    1. Node path:

        ["A", "B", "C"]

    2. Edge-aware path:

        [
            ("A", "B", "AB-1"),
            ("B", "C", "BC-1")
        ]

    3. Single-edge edge-aware path:

        [
            ("A", "B", "AB-1")
        ]

    4. Bare single-edge tuple:

        ("A", "B", "AB-1")

    The fourth representation is recognized only when
    the graph confirms that "AB-1" is an actual MultiGraph
    edge key between A and B.

    No arbitrary parallel channel is selected.
    """

    if path is None:

        return []

    if not isinstance(
        path,
        (list, tuple),
    ):

        raise TypeError(
            "path must be a list or tuple."
        )

    if len(path) == 0:

        return []

    # ======================================================
    # BARE SINGLE EDGE
    # ======================================================

    if _is_single_edge_tuple(
        G,
        path,
    ):

        u, v, key = path

        return [
            (
                u,
                v,
                key,
            )
        ]

    # ======================================================
    # EDGE-AWARE SEQUENCE
    # ======================================================

    first_item = path[0]

    if (
        isinstance(first_item, tuple)
        and len(first_item) == 3
    ):

        transitions = []

        for index, item in enumerate(path):

            if not (
                isinstance(item, tuple)
                and len(item) == 3
            ):

                raise ValueError(
                    "An edge-aware path must contain "
                    "only (u,v,key) tuples."
                )

            u, v, key = item

            if index > 0:

                previous = path[
                    index - 1
                ]

                previous_v = previous[1]

                if previous_v != u:

                    raise ValueError(
                        "Edge-aware path is discontinuous: "
                        f"{previous_v!r} -> {u!r}."
                    )

            transitions.append(
                (
                    u,
                    v,
                    key,
                )
            )

        return transitions

    # ======================================================
    # NODE PATH
    # ======================================================

    if len(path) < 2:

        return []

    transitions = []

    for index in range(
        len(path) - 1
    ):

        transitions.append(
            (
                path[index],
                path[index + 1],
                None,
            )
        )

    return transitions


# ==========================================================
# Optional Failure Probability
# ==========================================================

def _optional_failure_probability(
    edge_data,
    edge_description,
):
    """
    Resolve failure probability when available.

    Supported forms:

        failure_probability

    or:

        success_count
        failure_count

    Returns:

        None
            if no reliability information exists.

        float in [0,1]
            if reliability information exists.

    No artificial fallback is created.
    """

    if (
        "success_count" in edge_data
        and "failure_count" in edge_data
    ):

        success = _to_float(
            edge_data["success_count"],
            f"success_count[{edge_description}]",
        )

        failure = _to_float(
            edge_data["failure_count"],
            f"failure_count[{edge_description}]",
        )

        if success < 0.0:

            raise ValueError(
                f"success_count must be >= 0 "
                f"for {edge_description}."
            )

        if failure < 0.0:

            raise ValueError(
                f"failure_count must be >= 0 "
                f"for {edge_description}."
            )

        total = (
            success
            + failure
        )

        if total > 0.0:

            probability = (
                failure / total
            )

        elif "failure_probability" in edge_data:

            probability = _to_float(
                edge_data["failure_probability"],
                f"failure_probability[{edge_description}]",
            )

        else:

            return None

    elif "failure_probability" in edge_data:

        probability = _to_float(
            edge_data["failure_probability"],
            f"failure_probability[{edge_description}]",
        )

    else:

        return None

    if not 0.0 <= probability <= 1.0:

        raise ValueError(
            f"failure_probability for "
            f"{edge_description} must be in [0,1]; "
            f"got {probability}."
        )

    return float(
        probability
    )


# ==========================================================
# Optional Liquidity
# ==========================================================

def _optional_edge_liquidity(
    edge_data,
    edge_description,
):
    """
    Resolve directional liquidity when available.

    Supported fields:

        estimated_liquidity
        liquidity_uv
        balance_uv

    Returns None when the graph does not provide a recognized
    liquidity field.

    No artificial liquidity value is created.
    """

    if "estimated_liquidity" in edge_data:

        liquidity = _to_float(
            edge_data["estimated_liquidity"],
            f"estimated_liquidity[{edge_description}]",
        )

    elif "liquidity_uv" in edge_data:

        liquidity = _to_float(
            edge_data["liquidity_uv"],
            f"liquidity_uv[{edge_description}]",
        )

    elif "balance_uv" in edge_data:

        liquidity = _to_float(
            edge_data["balance_uv"],
            f"balance_uv[{edge_description}]",
        )

    else:

        return None

    if liquidity < 0.0:

        raise ValueError(
            f"Liquidity must be >= 0 for "
            f"{edge_description}."
        )

    return float(
        liquidity
    )


# ==========================================================
# Backward-Compatible Liquidity Resolver
# ==========================================================

def _edge_liquidity(
    edge_data,
    edge_description,
):
    """
    Strict liquidity resolver.

    This function is retained for callers that explicitly
    require liquidity.

    evaluate_path() uses the optional resolver because
    descriptive evaluation must not fabricate or require a
    metric that is absent from the graph.
    """

    liquidity = _optional_edge_liquidity(
        edge_data,
        edge_description,
    )

    if liquidity is None:

        raise KeyError(
            f"Missing directional liquidity information "
            f"for {edge_description}."
        )

    return liquidity


# ==========================================================
# Geographic Coordinates
# ==========================================================

def _node_coordinates(
    G,
    node,
):
    """
    Return validated:

        latitude, longitude
    """

    if node not in G:

        raise KeyError(
            f"Node {node!r} does not exist in graph."
        )

    node_data = G.nodes[node]

    if "latitude" not in node_data:

        raise KeyError(
            f"Missing latitude for node {node!r}."
        )

    if "longitude" not in node_data:

        raise KeyError(
            f"Missing longitude for node {node!r}."
        )

    latitude = _to_float(
        node_data["latitude"],
        f"latitude[{node!r}]",
    )

    longitude = _to_float(
        node_data["longitude"],
        f"longitude[{node!r}]",
    )

    if not -90.0 <= latitude <= 90.0:

        raise ValueError(
            f"latitude[{node!r}] outside [-90,90]: "
            f"{latitude}"
        )

    if not -180.0 <= longitude <= 180.0:

        raise ValueError(
            f"longitude[{node!r}] outside [-180,180]: "
            f"{longitude}"
        )

    return (
        latitude,
        longitude,
    )


# ==========================================================
# Path Evaluation
# ==========================================================

def evaluate_path(
    G,
    path,
    amount,
):
    """
    Evaluate a path descriptively.

    Supported:

        Node path:
            ["A", "B", "C"]

        Edge-aware path:
            [
                ("A", "B", "AB-1"),
                ("B", "C", "BC-1")
            ]

        Single edge:
            [
                ("A", "B", "AB-1")
            ]

        Bare single edge:
            ("A", "B", "AB-1")

    Returned metrics:

        success
        total_fee
        total_delay
        min_liquidity
        reliability
        total_distance_km
        total_carbon

    Important rules:

    1. Exact channel key is respected.
    2. No arbitrary parallel channel is selected.
    3. Carbon is counted once per unique node.
    4. Missing optional liquidity does not become zero.
    5. Missing optional reliability does not become zero.
    6. Missing RGB becomes carbon proxy = 0.
    """

    empty_result = {
        "success": False,
        "total_fee": 0.0,
        "total_delay": 0.0,
        "min_liquidity": None,
        "reliability": None,
        "total_distance_km": 0.0,
        "total_carbon": 0.0,
    }

    # ======================================================
    # PATH INPUT
    # ======================================================

    if path is None:

        return empty_result

    if not isinstance(
        path,
        (list, tuple),
    ):

        raise TypeError(
            "path must be a list or tuple."
        )

    if len(path) == 0:

        return empty_result

    amount = _to_float(
        amount,
        "amount",
    )

    if amount < 0.0:

        raise ValueError(
            "amount must be >= 0."
        )

    # ======================================================
    # BUILD TRANSITIONS
    # ======================================================

    transitions = _build_path_transitions(
        G,
        path,
    )

    if not transitions:

        return empty_result

    # ======================================================
    # ACCUMULATORS
    # ======================================================

    total_fee = 0.0

    total_delay = 0.0

    total_distance = 0.0

    known_liquidity = []

    reliability_values = []

    # ======================================================
    # ORDERED NODES
    # ======================================================

    ordered_nodes = []

    for index, (
        u,
        v,
        edge_key,
    ) in enumerate(transitions):

        if index == 0:

            ordered_nodes.append(
                u
            )

        else:

            previous_v = transitions[
                index - 1
            ][1]

            if previous_v != u:

                raise ValueError(
                    "Path transitions are not continuous: "
                    f"{previous_v!r} -> {u!r}."
                )

        ordered_nodes.append(
            v
        )

    # ======================================================
    # EDGE EVALUATION
    # ======================================================

    for u, v, edge_key in transitions:

        edge_description = (
            f"({u!r},{v!r},{edge_key!r})"
        )

        # --------------------------------------------------
        # EXACT CHANNEL
        # --------------------------------------------------

        edge_data = _resolve_path_edge(
            G,
            u,
            v,
            edge_key,
        )

        # --------------------------------------------------
        # FEE
        # --------------------------------------------------

        total_fee += channel_fee(
            edge_data,
            amount,
        )

        # --------------------------------------------------
        # DELAY
        # --------------------------------------------------

        total_delay += channel_delay(
            edge_data,
        )

        # --------------------------------------------------
        # OPTIONAL RELIABILITY
        # --------------------------------------------------

        failure_probability = (
            _optional_failure_probability(
                edge_data,
                edge_description,
            )
        )

        if failure_probability is not None:

            reliability_values.append(
                1.0
                - failure_probability
            )

        # --------------------------------------------------
        # OPTIONAL LIQUIDITY
        # --------------------------------------------------

        liquidity = _optional_edge_liquidity(
            edge_data,
            edge_description,
        )

        if liquidity is not None:

            known_liquidity.append(
                liquidity
            )

        # --------------------------------------------------
        # GEOGRAPHIC DISTANCE
        # --------------------------------------------------

        lat1, lon1 = _node_coordinates(
            G,
            u,
        )

        lat2, lon2 = _node_coordinates(
            G,
            v,
        )

        total_distance += (
            _haversine_distance_km(
                lat1,
                lon1,
                lat2,
                lon2,
            )
        )

    # ======================================================
    # UNIQUE NODE CARBON
    # ======================================================

    unique_nodes = []

    seen_nodes = set()

    for node in ordered_nodes:

        if node not in seen_nodes:

            seen_nodes.add(
                node
            )

            unique_nodes.append(
                node
            )

    total_carbon = 0.0

    for node in unique_nodes:

        total_carbon += (
            node_carbon_intensity(
                G,
                node,
            )
        )

    # ======================================================
    # PATH RELIABILITY
    # ======================================================

    if reliability_values:

        path_reliability = 1.0

        for edge_reliability in reliability_values:

            path_reliability *= (
                edge_reliability
            )

    else:

        path_reliability = None

    # ======================================================
    # FINAL RESULT
    # ======================================================

    return {
        "success": True,
        "total_fee": float(
            total_fee
        ),
        "total_delay": float(
            total_delay
        ),
        "min_liquidity": (
            min(known_liquidity)
            if known_liquidity
            else None
        ),
        "reliability": (
            float(
                path_reliability
            )
            if path_reliability is not None
            else None
        ),
        "total_distance_km": float(
            total_distance
        ),
        "total_carbon": float(
            total_carbon
        ),
    }


# ==========================================================
# Haversine Distance
# ==========================================================

def _haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate geographic distance using the Haversine formula.

    Earth radius:

        R = 6371 km
    """

    lat1 = _to_float(
        lat1,
        "lat1",
    )

    lon1 = _to_float(
        lon1,
        "lon1",
    )

    lat2 = _to_float(
        lat2,
        "lat2",
    )

    lon2 = _to_float(
        lon2,
        "lon2",
    )

    if not -90.0 <= lat1 <= 90.0:

        raise ValueError(
            f"lat1 outside [-90,90]: {lat1}"
        )

    if not -90.0 <= lat2 <= 90.0:

        raise ValueError(
            f"lat2 outside [-90,90]: {lat2}"
        )

    if not -180.0 <= lon1 <= 180.0:

        raise ValueError(
            f"lon1 outside [-180,180]: {lon1}"
        )

    if not -180.0 <= lon2 <= 180.0:

        raise ValueError(
            f"lon2 outside [-180,180]: {lon2}"
        )

    radius_km = 6371.0

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2.0
        ) ** 2
        +
        math.cos(phi1)
        *
        math.cos(phi2)
        *
        math.sin(
            dlambda / 2.0
        ) ** 2
    )

    # Numerical round-off protection only.

    a = min(
        1.0,
        max(
            0.0,
            a,
        ),
    )

    c = (
        2.0
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(
                1.0 - a
            ),
        )
    )

    distance = (
        radius_km
        * c
    )

    if not _valid_nonnegative(
        distance
    ):

        raise ValueError(
            "Calculated Haversine distance is invalid."
        )

    return float(
        distance
    )
