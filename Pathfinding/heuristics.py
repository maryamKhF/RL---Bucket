"""
Routing heuristics for Lightning-style payment networks.

Core adaptive-cost model
------------------------

    raw_h(u,v,eta)
        |
        v
    normalized penalty
        |
        v
    C'(u,v) = C_LND(u,v) * (1 + lambda_h * penalty)

The same adaptive-cost definition is used by:

    - Dijkstra
    - Top-K candidate generation
    - eta sensitivity tests

Important:
    capacity is NOT interpreted as directional liquidity.
"""

import math


# ==========================================================
# Numeric Helpers
# ==========================================================

def _safe_float(value, default=0.0):
    try:
        value = float(value)

        if not math.isfinite(value):
            return float(default)

        return value

    except (TypeError, ValueError):
        return float(default)


def _valid_number(value):
    try:
        value = float(value)
        return math.isfinite(value)

    except (TypeError, ValueError):
        return False


def _valid_nonnegative(value):
    try:
        value = float(value)

        return (
            math.isfinite(value)
            and value >= 0.0
        )

    except (TypeError, ValueError):
        return False


# ==========================================================
# ETA Validation
# ==========================================================

def validate_eta(eta):
    """
    Validate and return eta.

    The PPO action space is [0, 1], therefore the entire
    routing pipeline uses the same interval.
    """

    try:
        eta = float(eta)

    except (TypeError, ValueError):
        raise ValueError(
            "eta must be numeric"
        )

    if not math.isfinite(eta):
        raise ValueError(
            "eta must be finite"
        )

    if not 0.0 <= eta <= 1.0:
        raise ValueError(
            "eta must satisfy 0 <= eta <= 1"
        )

    return eta


# ==========================================================
# Lambda Validation
# ==========================================================

def validate_lambda_h(lambda_h):
    """
    Validate adaptive-cost weight.
    """

    try:
        lambda_h = float(lambda_h)

    except (TypeError, ValueError):
        raise ValueError(
            "lambda_h must be numeric"
        )

    if (
        not math.isfinite(lambda_h)
        or lambda_h < 0.0
    ):
        raise ValueError(
            "lambda_h must be finite and >= 0"
        )

    return lambda_h


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

    fee_rate is interpreted as PPM.
    """

    base_fee = _safe_float(
        data.get(
            "fee_base",
            data.get(
                "fee_base_msat",
                0.0,
            ),
        )
    )

    fee_rate = _safe_float(
        data.get(
            "fee_rate",
            data.get(
                "fee_proportional_millionths",
                0.0,
            ),
        )
    )

    return (
        base_fee
        +
        amount
        * fee_rate
        / 1_000_000.0
    )


# ==========================================================
# Channel Delay
# ==========================================================

def channel_delay(data):
    """
    Extract forwarding delay.
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
    Native LND-style routing cost.

    The native objective combines forwarding fee and delay.

        C_LND = fee + 0.5 * delay + 1

    The constant 1 keeps the edge cost strictly positive.
    """

    fee = channel_fee(
        data,
        amount,
    )

    delay = channel_delay(
        data
    )

    cost = (
        fee
        +
        0.5 * delay
        +
        1.0
    )

    if not _valid_nonnegative(cost):
        raise ValueError(
            "Invalid native LND cost."
        )

    return float(cost)


# ==========================================================
# Carbon / Geographic Feature
# ==========================================================

def node_carbon_intensity(
    G,
    node,
):
    """
    Return node carbon intensity.

    Supported attributes:

        rgb_color
        carbon_intensity

    Missing values default to zero.
    """

    if node not in G:
        return 0.0

    data = G.nodes[node]

    value = data.get(
        "carbon_intensity",
        None,
    )

    if value is not None:
        value = _safe_float(
            value,
            default=0.0,
        )

        if value >= 0.0:
            return value

    rgb = data.get(
        "rgb_color",
        None,
    )

    if rgb is None:
        return 0.0

    # ------------------------------------------------------
    # Numeric RGB representation
    # ------------------------------------------------------

    if isinstance(
        rgb,
        (list, tuple),
    ) and len(rgb) >= 3:

        values = [
            _safe_float(x)
            for x in rgb[:3]
        ]

        values = [
            max(0.0, x)
            for x in values
        ]

        return float(
            sum(values) / len(values)
        )

    # ------------------------------------------------------
    # String RGB representation
    # ------------------------------------------------------

    if isinstance(rgb, str):

        text = rgb.strip()

        if text.startswith("#"):
            text = text[1:]

        if len(text) == 6:

            try:

                r = int(
                    text[0:2],
                    16,
                )

                g = int(
                    text[2:4],
                    16,
                )

                b = int(
                    text[4:6],
                    16,
                )

                return float(
                    (r + g + b) / 3.0
                )

            except ValueError:
                pass

    return 0.0


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
    Calculate the raw adaptive routing heuristic.

    Formula:

        h(u,v) =
            eta * ((C_u + C_v) / 2)
            +
            (1 - eta) * (C_v - C_u)

    where:

        C_u = carbon intensity of node u
        C_v = carbon intensity of node v

    eta ∈ [0,1].

    Important:
        This function returns the raw signal.

        It is NOT used directly as the final edge cost.

        The raw signal is converted into a bounded,
        non-negative penalty by adaptive_penalty().
    """

    eta = validate_eta(
        eta
    )

    cu = node_carbon_intensity(
        G,
        u,
    )

    cv = node_carbon_intensity(
        G,
        v,
    )

    value = (
        eta
        * (
            (cu + cv)
            / 2.0
        )
        +
        (1.0 - eta)
        * (
            cv - cu
        )
    )

    if not _valid_number(value):
        raise ValueError(
            "Adaptive heuristic is not finite."
        )

    return float(value)


# ==========================================================
# Normalized Adaptive Penalty
# ==========================================================

def adaptive_penalty(
    G,
    u,
    v,
    eta,
):
    """
    Convert the raw adaptive heuristic into a stable
    non-negative bounded penalty.

    raw_h = adaptive_heuristic(...)

    penalty =
        |raw_h| / (1 + |raw_h|)

    Therefore:

        0 <= penalty < 1

    This prevents the adaptive signal from producing
    negative edge costs or dominating the native routing
    objective because of raw carbon-value scale.
    """

    raw_h = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    magnitude = abs(
        raw_h
    )

    penalty = (
        magnitude
        /
        (
            1.0
            +
            magnitude
        )
    )

    if not _valid_number(
        penalty
    ):
        raise ValueError(
            "Adaptive penalty is not finite."
        )

    penalty = min(
        max(
            penalty,
            0.0,
        ),
        1.0,
    )

    return float(penalty)


# ==========================================================
# Adaptive Cost
# ==========================================================

def modified_cost(
    native_cost,
    geo_penalty,
    eta=None,
    lambda_h=1.0,
):
    """
    Convert native routing cost into the common adaptive cost.

    Final objective:

        C'(e) =
            C_LND(e)
            *
            (
                1
                +
                lambda_h * p(e)
            )

    where p(e) is the normalized adaptive penalty.

    geo_penalty is expected to already be normalized to [0,1].

    eta is retained in the signature for backward compatibility
    and for explicit reporting, but the penalty itself is already
    computed from eta before this function is called.
    """

    if not _valid_nonnegative(
        native_cost
    ):
        raise ValueError(
            "native_cost must be finite and >= 0"
        )

    if not _valid_nonnegative(
        geo_penalty
    ):
        raise ValueError(
            "geo_penalty must be finite and >= 0"
        )

    if geo_penalty > 1.0:
        raise ValueError(
            "geo_penalty must be normalized to [0,1]"
        )

    lambda_h = validate_lambda_h(
        lambda_h
    )

    cost = (
        float(native_cost)
        *
        (
            1.0
            +
            lambda_h
            * float(geo_penalty)
        )
    )

    if not _valid_nonnegative(cost):
        raise ValueError(
            "Modified cost is invalid."
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
    Unified edge-cost calculation used by both Dijkstra
    and Top-K routing.

    Returns:

        {
            "native_cost": ...,
            "raw_heuristic": ...,
            "adaptive_penalty": ...,
            "cost": ...
        }

    PPO controls eta only.
    """

    eta = validate_eta(
        eta
    )

    lambda_h = validate_lambda_h(
        lambda_h
    )

    # ------------------------------------------------------
    # Native cost
    # ------------------------------------------------------

    if heuristic_fn is None:

        native_cost = lnd_cost(
            G,
            u,
            v,
            data,
            amount,
        )

    else:

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
            AttributeError,
        ):

            native_cost = lnd_cost(
                G,
                u,
                v,
                data,
                amount,
            )

    if not _valid_nonnegative(
        native_cost
    ):
        raise ValueError(
            "Native routing cost is invalid."
        )

    # ------------------------------------------------------
    # Raw adaptive signal
    # ------------------------------------------------------

    raw_h = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    # ------------------------------------------------------
    # Normalized adaptive penalty
    # ------------------------------------------------------

    penalty = adaptive_penalty(
        G,
        u,
        v,
        eta,
    )

    # ------------------------------------------------------
    # Final cost
    # ------------------------------------------------------

    cost = modified_cost(
        native_cost=native_cost,
        geo_penalty=penalty,
        eta=eta,
        lambda_h=lambda_h,
    )

    return {
        "native_cost": float(
            native_cost
        ),
        "raw_heuristic": float(
            raw_h
        ),
        "adaptive_penalty": float(
            penalty
        ),
        "cost": float(
            cost
        ),
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
    Convenience wrapper around the unified adaptive cost.

    Use this function when an external component explicitly
    requests an adaptive LND edge cost.

    Dijkstra and Top-K should normally call adaptive_edge_cost()
    directly so the native cost is not accidentally wrapped twice.
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
# Auxiliary Enhanced Cost
# ==========================================================

def enhanced_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    Auxiliary multi-factor cost.

    This function is not the PPO-controlled objective.

    It can be used for diagnostics or future ablation studies.
    """

    fee = channel_fee(
        data,
        amount,
    )

    delay = channel_delay(
        data
    )

    failure_probability = _safe_float(
        data.get(
            "failure_probability",
            0.01,
        ),
        default=0.01,
    )

    failure_probability = min(
        max(
            failure_probability,
            0.0,
        ),
        1.0,
    )

    reliability_penalty = (
        failure_probability
    )

    cost = (
        fee
        +
        0.5 * delay
        +
        reliability_penalty
        +
        1.0
    )

    return max(
        1e-9,
        float(cost),
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
    Calculate descriptive metrics for a completed path.

    This function does not determine the route.
    """

    if not path or len(path) < 2:
        return {
            "success": False,
            "total_fee": 0.0,
            "total_delay": 0.0,
            "min_liquidity": None,
            "reliability": 0.0,
            "total_distance_km": 0.0,
            "total_carbon": 0.0,
        }

    total_fee = 0.0
    total_delay = 0.0
    total_distance = 0.0
    total_carbon = 0.0

    reliability = 1.0
    known_liquidity = []

    for u, v in zip(
        path[:-1],
        path[1:],
    ):

        if not G.has_edge(u, v):
            return {
                "success": False,
                "total_fee": 0.0,
                "total_delay": 0.0,
                "min_liquidity": None,
                "reliability": 0.0,
                "total_distance_km": 0.0,
                "total_carbon": 0.0,
            }

        data = G[u][v]

        if hasattr(
            data,
            "items",
        ) and data and all(
            isinstance(value, dict)
            for value in data.values()
        ):
            # MultiDiGraph: use first channel for descriptive
            # evaluation when no explicit key is supplied.
            key = next(
                iter(data)
            )
            edge_data = data[key]

        else:
            edge_data = data

        total_fee += channel_fee(
            edge_data,
            amount,
        )

        total_delay += channel_delay(
            edge_data
        )

        success = _safe_float(
            edge_data.get(
                "success_count",
                0.0,
            )
        )

        failure = _safe_float(
            edge_data.get(
                "failure_count",
                0.0,
            )
        )

        total = success + failure

        if total > 0:
            failure_probability = (
                failure / total
            )

        else:
            failure_probability = _safe_float(
                edge_data.get(
                    "failure_probability",
                    0.01,
                ),
                default=0.01,
            )

        failure_probability = min(
            max(
                failure_probability,
                0.0,
            ),
            1.0,
        )

        reliability *= (
            1.0
            -
            failure_probability
        )

        liquidity = None

        for field in (
            "estimated_liquidity",
            "liquidity_uv",
            "balance_uv",
        ):

            if field not in edge_data:
                continue

            value = _safe_float(
                edge_data.get(field),
                default=-1.0,
            )

            if value >= 0.0:
                liquidity = value
                break

        if liquidity is not None:
            known_liquidity.append(
                liquidity
            )

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
        "reliability": float(
            reliability
        ),
        "total_distance_km": float(
            total_distance
        ),
        "total_carbon": float(
            total_carbon
        ),
    }