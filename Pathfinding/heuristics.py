# Pathfinding/heuristics.py

import math

from Network.topology import geo_features


# ==========================================================
# Numeric Helpers
# ==========================================================

def _safe_float(value, default=0.0):
    """
    Convert a value to a finite float.

    Invalid or non-finite values are replaced with default.
    """

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


def _valid_non_negative(value):
    """
    Return True if value is finite and >= 0.
    """

    try:
        value = float(value)

        return (
            math.isfinite(value)
            and
            value >= 0.0
        )

    except (
        TypeError,
        ValueError,
    ):
        return False


def _validate_amount(amount):
    """
    Validate payment amount.
    """

    amount = _safe_float(
        amount,
        default=-1.0,
    )

    if (
        not math.isfinite(amount)
        or
        amount <= 0.0
    ):
        raise ValueError(
            "amount must be finite and greater than zero"
        )

    return amount


def _validate_eta(eta):
    """
    Validate PPO adaptive parameter.

    Required range:

        -1 <= eta <= 1
    """

    eta = _safe_float(
        eta,
        default=float("nan"),
    )

    if not math.isfinite(eta):
        raise ValueError(
            "eta must be finite"
        )

    if not -1.0 <= eta <= 1.0:
        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    return eta


def _validate_lambda_h(lambda_h):
    """
    Validate adaptive heuristic weight.
    """

    lambda_h = _safe_float(
        lambda_h,
        default=float("nan"),
    )

    if not math.isfinite(lambda_h):
        raise ValueError(
            "lambda_h must be finite"
        )

    if lambda_h < 0.0:
        raise ValueError(
            "lambda_h must be >= 0"
        )

    return lambda_h


# ==========================================================
# Channel Fee
# ==========================================================

def channel_fee(data, amount):
    """
    Calculate Lightning forwarding fee.

    Formula:

        fee =
            fee_base
            +
            amount * fee_rate / 1,000,000

    fee_rate is interpreted as proportional fee in PPM.

    Supported fields:

        fee_base
        fee_base_msat

        fee_rate
        fee_proportional_millionths

    The original numerical unit convention of the dataset
    is preserved. Unit conversion between sat and msat must
    be handled consistently by the caller and simulator.
    """

    amount = _validate_amount(
        amount
    )

    fee_base = _safe_float(
        data.get(
            "fee_base",
            data.get(
                "fee_base_msat",
                0.0,
            ),
        ),
        default=0.0,
    )

    fee_rate = _safe_float(
        data.get(
            "fee_rate",
            data.get(
                "fee_proportional_millionths",
                0.0,
            ),
        ),
        default=0.0,
    )

    # Invalid negative fee parameters are not meaningful.
    fee_base = max(
        0.0,
        fee_base,
    )

    fee_rate = max(
        0.0,
        fee_rate,
    )

    fee = (
        fee_base
        +
        (
            fee_rate
            *
            amount
            /
            1_000_000.0
        )
    )

    return float(
        max(
            0.0,
            fee,
        )
    )


# ==========================================================
# Directional Liquidity
# ==========================================================

def estimated_liquidity(data):
    """
    Return known directional liquidity.

    Priority:

        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

    Important:

        capacity is NEVER used as liquidity.

    Returns:

        float
            Known directional liquidity.

        None
            Liquidity is unknown.
    """

    for field in (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv",
    ):

        if field not in data:
            continue

        value = data.get(
            field
        )

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
            and
            value >= 0.0
        ):
            return value

    return None


# ==========================================================
# Liquidity Feasibility
# ==========================================================

def liquidity_available(data, amount):
    """
    Check whether known directional liquidity can carry
    the requested payment.

    Cases:

        known liquidity < amount
            -> False

        known liquidity >= amount
            -> True

        unknown liquidity
            -> True

    Unknown liquidity is NOT interpreted as zero.
    """

    amount = _validate_amount(
        amount
    )

    liquidity = estimated_liquidity(
        data
    )

    if liquidity is None:
        return True

    return (
        liquidity >= amount
    )


# ==========================================================
# Liquidity Penalty
# ==========================================================

def liquidity_penalty(data, amount):
    """
    Calculate optional liquidity pressure.

    Formula:

        amount / estimated_liquidity

    If liquidity is unknown:

        0.0

    If known liquidity is zero:

        inf

    Important:

        capacity is never used as liquidity.
    """

    amount = _validate_amount(
        amount
    )

    liquidity = estimated_liquidity(
        data
    )

    # Unknown liquidity is not equivalent to zero.
    if liquidity is None:
        return 0.0

    if liquidity <= 0.0:
        return math.inf

    return float(
        amount
        /
        liquidity
    )


# ==========================================================
# Reliability
# ==========================================================

def reliability_penalty(data):
    """
    Estimate channel failure probability.

    If empirical observations exist:

        failure /
        (success + failure)

    Otherwise:

        failure_probability

    Otherwise:

        0.01

    Return value is always clipped to [0,1].
    """

    success = _safe_float(
        data.get(
            "success_count",
            0,
        ),
        default=0.0,
    )

    failure = _safe_float(
        data.get(
            "failure_count",
            0,
        ),
        default=0.0,
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

        probability = (
            failure
            /
            total
        )

    else:

        probability = _safe_float(
            data.get(
                "failure_probability",
                0.01,
            ),
            default=0.01,
        )

    return float(
        min(
            max(
                probability,
                0.0,
            ),
            1.0,
        )
    )


# ==========================================================
# Channel Reliability
# ==========================================================

def channel_reliability(data):
    """
    Return estimated forwarding reliability.

    reliability =
        1 - failure_probability
    """

    probability = reliability_penalty(
        data
    )

    return float(
        1.0
        -
        probability
    )


# ==========================================================
# Channel Delay
# ==========================================================

def channel_delay(data):
    """
    Extract channel forwarding delay.

    Supported attributes:

        delay
        cltv_expiry_delta
    """

    delay = _safe_float(
        data.get(
            "delay",
            data.get(
                "cltv_expiry_delta",
                0.0,
            ),
        ),
        default=0.0,
    )

    return max(
        0.0,
        delay,
    )


# ==========================================================
# LND Base Cost
# ==========================================================

def lnd_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    Native LND-like routing cost.

    Formula:

        C_LND(e) =
            forwarding_fee
            +
            0.5 * delay
            +
            1

    PPO does not directly select the route.

    PPO produces eta, which is subsequently used by
    adaptive_heuristic() to modify this base cost.
    """

    amount = _validate_amount(
        amount
    )

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

    return float(
        max(
            1e-9,
            cost,
        )
    )


# ==========================================================
# CLN Base Cost
# ==========================================================

def cln_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    CLN-like routing cost.

    This function is retained for baseline comparison.

    It is not the main PPO adaptive routing cost.
    """

    amount = _validate_amount(
        amount
    )

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
        delay
        +
        0.7
    )

    return float(
        max(
            1e-9,
            cost,
        )
    )


# ==========================================================
# ECL Base Cost
# ==========================================================

def ecl_cost(
    G,
    u,
    v,
    data,
    amount,
):
    """
    ECL-like routing cost.

    This function is retained for baseline comparison.

    It is not the main PPO adaptive routing cost.
    """

    amount = _validate_amount(
        amount
    )

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
        0.8 * delay
        +
        1.3
    )

    return float(
        max(
            1e-9,
            cost,
        )
    )


# ==========================================================
# Node Regional Color
# ==========================================================

def node_regcolor(
    G,
    node,
):
    """
    Return the numeric regional color (regcolor) of a node.

    The real Lightning snapshot does not provide numeric
    carbon-intensity data. Therefore regcolor is the
    node-level environmental proxy used by the adaptive
    routing heuristic.

    Required node attribute:
        regcolor
    """

    if node not in G:
        raise KeyError(
            f"Node {node} does not exist in graph."
        )

    value = G.nodes[node].get(
        "regcolor",
        None,
    )

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Node {node} has invalid regcolor={value!r}."
        )

    if not math.isfinite(value):
        raise ValueError(
            f"Node {node} has non-finite regcolor={value!r}."
        )

    return float(value)


# Backward-compatible helper name.
def node_carbon_intensity(
    G,
    node,
):
    """
    Compatibility wrapper.

    Despite the historical function name, the returned
    environmental value is regcolor.
    """
    return node_regcolor(G, node)


# ==========================================================
# Adaptive Routing Heuristic
# ==========================================================

def adaptive_heuristic(
    G,
    u,
    v,
    eta,
):
    """
    Adaptive routing heuristic.

    Formula:

        h(u,v) =
            eta * ((R_u + R_v) / 2)
            +
            (1 - eta) * (R_v - R_u)

    where:

        R_u = regcolor of u
        R_v = regcolor of v

    eta is dynamically generated by PPO.

    Valid range:

        -1 <= eta <= 1
    """

    eta = _validate_eta(
        eta
    )

    regcolor_u = node_regcolor(
        G,
        u,
    )

    regcolor_v = node_regcolor(
        G,
        v,
    )

    average_regcolor = (
        regcolor_u
        +
        regcolor_v
    ) / 2.0

    regcolor_difference = (
        regcolor_v
        -
        regcolor_u
    )

    heuristic = (
        eta
        *
        average_regcolor
        +
        (
            1.0
            -
            eta
        )
        *
        regcolor_difference
    )

    if not math.isfinite(
        heuristic
    ):
        raise ValueError(
            "Adaptive heuristic produced a non-finite value."
        )

    return float(
        heuristic
    )


# ==========================================================
# Geographic Information
# ==========================================================

def geographic_penalty(
    G,
    u,
    v,
    data=None,
):
    """
    Return geographic information for an edge.

    This function is retained for compatibility.

    It is NOT the adaptive routing heuristic.

    The adaptive routing heuristic is implemented separately
    by adaptive_heuristic().
    """

    try:

        features = geo_features(
            G,
            u,
            v,
        )

    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ):

        features = {}

    return {

        "distance_km":
            max(
                0.0,
                _safe_float(
                    features.get(
                        "distance_km",
                        0.0,
                    )
                ),
            ),

        "carbon_intensity":
            max(
                0.0,
                _safe_float(
                    features.get(
                        "carbon_intensity",
                        0.0,
                    )
                ),
            ),

        "inter_country":
            float(
                bool(
                    features.get(
                        "inter_country",
                        False,
                    )
                )
            ),

        "inter_continent":
            float(
                bool(
                    features.get(
                        "inter_continent",
                        False,
                    )
                )
            ),
    }


# ==========================================================
# Modified Routing Cost
# ==========================================================

def modified_cost(
    native_cost,
    geo_penalty,
    eta,
    lambda_h=1.0,
):
    """
    Calculate adaptive routing cost.

    Formula:

        C'(e) =
            C_LND(e)
            +
            lambda_h * h(e)

    Parameters
    ----------
    native_cost:
        Native LND-like edge cost.

    geo_penalty:
        Historical parameter name.

        In the current model this value is actually the
        adaptive heuristic h(e).

    eta:
        PPO-generated adaptive coefficient.

    lambda_h:
        Weight assigned to adaptive heuristic.

    Important:
        Routing algorithms require non-negative edge costs.
        Therefore the final cost is clipped to a very small
        positive value if the adaptive term makes it <= 0.
    """

    native_cost = _safe_float(
        native_cost,
        default=float("nan"),
    )

    if not math.isfinite(
        native_cost
    ):
        raise ValueError(
            "native_cost must be finite."
        )

    if native_cost < 0.0:
        raise ValueError(
            "native_cost must be non-negative."
        )

    h = _safe_float(
        geo_penalty,
        default=float("nan"),
    )

    if not math.isfinite(
        h
    ):
        raise ValueError(
            "adaptive heuristic must be finite."
        )

    eta = _validate_eta(
        eta
    )

    lambda_h = _validate_lambda_h(
        lambda_h
    )

    cost = (
        native_cost
        +
        lambda_h * h
    )

    if not math.isfinite(
        cost
    ):
        raise ValueError(
            "modified routing cost is not finite."
        )

    return float(
        max(
            1e-9,
            cost,
        )
    )


# ==========================================================
# Complete Adaptive LND Cost
# ==========================================================

def adaptive_lnd_cost(
    G,
    u,
    v,
    data,
    amount,
    eta,
    lambda_h=1.0,
):
    """
    Complete PPO + adaptive heuristic + LND cost.

    Flow:

        PPO
         |
         v
        eta
         |
         v
        regcolor
         |
         v
    adaptive_heuristic()
         |
         v
    modified_cost()
         |
         v
       Dijkstra
    """

    base_cost = lnd_cost(
        G,
        u,
        v,
        data,
        amount,
    )

    heuristic = adaptive_heuristic(
        G,
        u,
        v,
        eta,
    )

    return modified_cost(
        native_cost=base_cost,
        geo_penalty=heuristic,
        eta=eta,
        lambda_h=lambda_h,
    )


# ==========================================================
# Enhanced Cost
# ==========================================================

def enhanced_cost(
    G,
    u,
    v,
    data,
    amount,
    fee_weight=1.0,
    liquidity_weight=0.5,
    reliability_weight=0.5,
):
    """
    Extended routing cost for auxiliary experiments.

    Important:

        This is NOT the main PPO adaptive routing cost.

    It is provided only for experiments where fee,
    liquidity pressure and reliability are jointly evaluated.
    """

    amount = _validate_amount(
        amount
    )

    fee_weight = _safe_float(
        fee_weight,
        default=1.0,
    )

    liquidity_weight = _safe_float(
        liquidity_weight,
        default=0.5,
    )

    reliability_weight = _safe_float(
        reliability_weight,
        default=0.5,
    )

    base = (
        fee_weight
        *
        lnd_cost(
            G,
            u,
            v,
            data,
            amount,
        )
    )

    liquidity = liquidity_penalty(
        data,
        amount,
    )

    reliability = reliability_penalty(
        data
    )

    if math.isinf(
        liquidity
    ):
        return math.inf

    result = (
        base
        +
        liquidity_weight
        *
        liquidity
        +
        reliability_weight
        *
        reliability
    )

    if not math.isfinite(
        result
    ):
        return math.inf

    return float(
        max(
            1e-9,
            result,
        )
    )


# ==========================================================
# Path Evaluation
# ==========================================================

def evaluate_path(
    G,
    edges,
    amount,
):
    """
    Calculate complete metrics for a selected route.

    Parameters
    ----------
    G:
        NetworkX graph.

    edges:
        Iterable of:

            (u, v, key)

        for MultiDiGraph.

    amount:
        Payment amount.

    Returns
    -------
    dict

        fee
        delay
        min_liquidity
        reliability
        distance_km
        carbon_intensity

    Important:

        Unknown liquidity is ignored when calculating
        min_liquidity.

        capacity is never treated as liquidity.
    """

    amount = _validate_amount(
        amount
    )

    if edges is None:
        raise ValueError(
            "edges cannot be None."
        )

    total_fee = 0.0
    total_delay = 0.0

    known_liquidities = []

    reliability = 1.0

    total_distance = 0.0
    total_carbon = 0.0

    # ======================================================
    # Evaluate each route edge
    # ======================================================

    for edge in edges:

        if len(edge) != 3:
            raise ValueError(
                "Each route edge must be (u, v, key)."
            )

        u, v, key = edge

        # --------------------------------------------------
        # Retrieve channel data
        # --------------------------------------------------

        try:

            if key is None:

                # Simple graph / compatibility mode.
                data = G[u][v]

            else:

                data = G[u][v][key]

        except (
            KeyError,
            TypeError,
        ):

            raise ValueError(
                f"Channel ({u}, {v}, {key}) does not exist."
            )

        # --------------------------------------------------
        # Fee
        # --------------------------------------------------

        total_fee += channel_fee(
            data,
            amount,
        )

        # --------------------------------------------------
        # Delay
        # --------------------------------------------------

        total_delay += channel_delay(
            data
        )

        # --------------------------------------------------
        # Directional liquidity
        # --------------------------------------------------

        liquidity = estimated_liquidity(
            data
        )

        if liquidity is not None:

            known_liquidities.append(
                float(
                    liquidity
                )
            )

        # --------------------------------------------------
        # Reliability
        # --------------------------------------------------

        failure_probability = (
            reliability_penalty(
                data
            )
        )

        channel_reliability = (
            1.0
            -
            failure_probability
        )

        reliability *= (
            channel_reliability
        )

        # --------------------------------------------------
        # Geographic information
        # --------------------------------------------------

        try:

            features = geo_features(
                G,
                u,
                v,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ):

            features = {}

        total_distance += max(
            0.0,
            _safe_float(
                features.get(
                    "distance_km",
                    0.0,
                )
            ),
        )

        # --------------------------------------------------
        # Carbon
        #
        # Carbon is calculated from endpoint node
        # intensities rather than treating channel capacity
        # or another channel attribute as carbon.
        # --------------------------------------------------

        try:

            carbon_u = node_carbon_intensity(
                G,
                u,
            )

            carbon_v = node_carbon_intensity(
                G,
                v,
            )

            total_carbon += (
                carbon_u
                +
                carbon_v
            ) / 2.0

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            total_carbon += max(
                0.0,
                _safe_float(
                    features.get(
                        "carbon_intensity",
                        0.0,
                    )
                ),
            )

    # ======================================================
    # Minimum Known Liquidity
    # ======================================================

    if known_liquidities:

        min_liquidity = min(
            known_liquidities
        )

    else:

        min_liquidity = None

    # ======================================================
    # Final Result
    # ======================================================

    return {

        "fee":
            float(
                total_fee
            ),

        "delay":
            float(
                total_delay
            ),

        "min_liquidity":
            (
                None
                if min_liquidity is None
                else float(
                    min_liquidity
                )
            ),

        "reliability":
            float(
                min(
                    max(
                        reliability,
                        0.0,
                    ),
                    1.0,
                )
            ),

        "distance_km":
            float(
                total_distance
            ),

        "carbon_intensity":
            float(
                max(
                    0.0,
                    total_carbon,
                )
            ),
    }