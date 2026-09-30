import math

from Network.topology import geo_features


# ==========================================================
# Channel Fee
# ==========================================================

def channel_fee(data, amount):
    """
    Lightning forwarding fee.

    fee_base:
        Base forwarding fee.

    fee_rate:
        Proportional forwarding fee in ppm.

    Important:
        The original unit convention of the input dataset is
        preserved here. Unit conversion between sat and msat
        should be handled consistently across the simulator
        and routing modules.
    """

    amount = float(amount)

    fee_base = float(
        data.get(
            "fee_base",
            data.get(
                "fee_base_msat",
                0.0
            )
        )
    )

    fee_rate = float(
        data.get(
            "fee_rate",
            data.get(
                "fee_proportional_millionths",
                0.0
            )
        )
    )

    return (
        fee_base
        +
        fee_rate * amount / 1_000_000.0
    )


# ==========================================================
# Liquidity
# ==========================================================

def liquidity_penalty(data, amount):
    """
    Optional liquidity penalty.

    Important:
        Channel capacity is NOT interpreted as directional
        liquidity.

    Only learned or observed directional liquidity can be used.

    Priority:
        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv
        4. unknown
    """

    estimated_liquidity = data.get(
        "estimated_liquidity",
        data.get(
            "liquidity_uv",
            data.get(
                "balance_uv",
                None
            )
        )
    )

    # Liquidity is unknown.
    # Unknown does not mean zero.
    if estimated_liquidity is None:
        return 0.0

    estimated_liquidity = float(
        estimated_liquidity
    )

    if estimated_liquidity <= 0:
        return math.inf

    return (
        float(amount)
        /
        estimated_liquidity
    )


# ==========================================================
# Reliability
# ==========================================================

def reliability_penalty(data):
    """
    Estimate channel failure probability.

    If empirical success/failure observations exist,
    use the observed failure ratio.

    Otherwise use the initial failure probability prior.
    """

    success = float(
        data.get(
            "success_count",
            0
        )
    )

    failure = float(
        data.get(
            "failure_count",
            0
        )
    )

    total = success + failure

    if total > 0:

        return (
            failure
            /
            total
        )

    return float(
        data.get(
            "failure_probability",
            0.01
        )
    )


# ==========================================================
# LND Base Cost
# ==========================================================

def lnd_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    Native LND-like routing cost.

    C_LND(e) =
        forwarding_fee
        + 0.5 * delay
        + 1

    PPO does not directly select a route.

    PPO produces eta, which is subsequently used by the
    adaptive heuristic to modify this native routing cost.
    """

    fee = channel_fee(
        data,
        amount
    )

    delay = float(
        data.get(
            "delay",
            data.get(
                "cltv_expiry_delta",
                0.0
            )
        )
    )

    return (
        fee
        +
        0.5 * delay
        +
        1.0
    )


# ==========================================================
# CLN Base Cost
# ==========================================================

def cln_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    CLN-like routing cost.

    Used only for baseline comparison.
    """

    fee = channel_fee(
        data,
        amount
    )

    delay = float(
        data.get(
            "delay",
            data.get(
                "cltv_expiry_delta",
                0.0
            )
        )
    )

    return (
        fee
        +
        delay
        +
        0.7
    )


# ==========================================================
# ECL Base Cost
# ==========================================================

def ecl_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    ECL-like routing cost.

    Used only for baseline comparison.
    """

    fee = channel_fee(
        data,
        amount
    )

    delay = float(
        data.get(
            "delay",
            data.get(
                "cltv_expiry_delta",
                0.0
            )
        )
    )

    return (
        fee
        +
        0.8 * delay
        +
        1.3
    )


# ==========================================================
# Node Carbon Intensity
# ==========================================================

def node_carbon_intensity(
        G,
        node
):
    """
    Return the numeric carbon-intensity value of a node.

    Modeling rule:

        carbon_intensity(node) = rgb_color(node)

    Larger rgb_color means higher carbon intensity.

    The graph_builder stores the original rgb_color attribute
    and also mirrors it into carbon_intensity.

    Priority:
        1. rgb_color
        2. carbon_intensity
        3. 0.0
    """

    if node not in G:

        raise KeyError(
            f"Node {node} does not exist in graph."
        )

    node_data = G.nodes[node]

    # Primary source:
    # rgb_color from the dataset.
    rgb_value = node_data.get(
        "rgb_color",
        None
    )

    if rgb_value is not None:

        try:

            return float(
                rgb_value
            )

        except (
            TypeError,
            ValueError
        ):

            pass

    # Backward compatibility.
    carbon_value = node_data.get(
        "carbon_intensity",
        0.0
    )

    try:

        return float(
            carbon_value
        )

    except (
        TypeError,
        ValueError
    ):

        return 0.0


# ==========================================================
# Adaptive Routing Heuristic
# ==========================================================

def adaptive_heuristic(
        G,
        u,
        v,
        eta
):
    """
    Adaptive routing heuristic defined by the README.

    h(u,v) =
        eta * ((intCO2[u] + intCO2[v]) / 2)
        +
        (1 - eta) * (intCO2[v] - intCO2[u])

    Carbon intensity is obtained directly from rgb_color.

    eta is dynamically produced by PPO.

    Valid range:

        -1 <= eta <= 1
    """

    eta = float(
        eta
    )

    if not -1.0 <= eta <= 1.0:

        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    carbon_u = node_carbon_intensity(
        G,
        u
    )

    carbon_v = node_carbon_intensity(
        G,
        v
    )

    average_carbon = (
        carbon_u
        +
        carbon_v
    ) / 2.0

    carbon_difference = (
        carbon_v
        -
        carbon_u
    )

    h = (
        eta * average_carbon
        +
        (1.0 - eta)
        *
        carbon_difference
    )

    return float(
        h
    )


# ==========================================================
# Geographic Information
# ==========================================================

def geographic_penalty(
        G,
        u,
        v,
        data=None
):
    """
    Return geographic information of an edge.

    This function is kept for compatibility with existing
    evaluation code.

    It is NOT the adaptive routing heuristic.

    The adaptive routing heuristic is implemented separately
    in adaptive_heuristic().
    """

    features = geo_features(
        G,
        u,
        v
    )

    return {

        "distance_km":
            float(
                features.get(
                    "distance_km",
                    0.0
                )
            ),

        "carbon_intensity":
            float(
                features.get(
                    "carbon_intensity",
                    0.0
                )
            ),

        "inter_country":
            float(
                features.get(
                    "inter_country",
                    0.0
                )
            ),

        "inter_continent":
            float(
                features.get(
                    "inter_continent",
                    0.0
                )
            )
    }


# ==========================================================
# Modified Routing Cost
# ==========================================================

def modified_cost(
        native_cost,
        geo_penalty,
        eta,
        lambda_h=1.0
):
    """
    Calculate the final adaptive routing cost.

    C'(e) =
        C_LND(e)
        +
        lambda_h * h(e)

    Here geo_penalty is retained as the historical parameter
    name for compatibility, but its actual meaning is:

        geo_penalty = adaptive heuristic h(e)

    eta is validated here because the modified cost belongs
    to the adaptive routing mechanism.
    """

    eta = float(
        eta
    )

    if not -1.0 <= eta <= 1.0:

        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    h = float(
        geo_penalty
    )

    lambda_h = float(
        lambda_h
    )

    cost = (
        float(native_cost)
        +
        lambda_h * h
    )

    # Routing costs must remain positive.
    return max(
        1e-9,
        cost
    )


# ==========================================================
# Complete Adaptive Edge Cost
# ==========================================================

def adaptive_lnd_cost(
        G,
        u,
        v,
        data,
        amount,
        eta,
        lambda_h=1.0
):
    """
    Complete PPO + adaptive heuristic + LND routing cost.

    Flow:

        PPO
         |
         v
        eta
         |
         v
        rgb_color
         |
         v
        adaptive heuristic h(u,v)
         |
         v
        C_LND + lambda_h * h
         |
         v
        Dijkstra
    """

    base_cost = lnd_cost(
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
        native_cost=base_cost,
        geo_penalty=h,
        eta=eta,
        lambda_h=lambda_h
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
        reliability_weight=0.5
):
    """
    Extended routing cost for auxiliary experiments.

    This is NOT the main PPO adaptive heuristic defined
    in the README.
    """

    base = (
        fee_weight
        *
        lnd_cost(
            G,
            u,
            v,
            data,
            amount
        )
    )

    liquidity = liquidity_penalty(
        data,
        amount
    )

    reliability = reliability_penalty(
        data
    )

    if math.isinf(
        liquidity
    ):

        return math.inf

    return (
        base
        +
        liquidity_weight * liquidity
        +
        reliability_weight * reliability
    )


# ==========================================================
# Path Evaluation
# ==========================================================

def evaluate_path(
        G,
        edges,
        amount
):
    """
    Calculate complete metrics for a route.

    Liquidity is based only on known/learned directional
    liquidity.

    Snapshot capacity is never interpreted as directional
    balance.

    Returned metrics:
        fee
        delay
        min_liquidity
        reliability
        distance_km
        carbon_intensity
    """

    total_fee = 0.0
    total_delay = 0.0

    known_liquidities = []

    reliability = 1.0

    total_distance = 0.0
    total_carbon = 0.0

    for u, v, k in edges:

        data = G[u][v][k]

        # --------------------------------------------------
        # Fee
        # --------------------------------------------------

        total_fee += channel_fee(
            data,
            amount
        )

        # --------------------------------------------------
        # Delay
        # --------------------------------------------------

        total_delay += float(
            data.get(
                "delay",
                data.get(
                    "cltv_expiry_delta",
                    0.0
                )
            )
        )

        # --------------------------------------------------
        # Learned / observed directional liquidity
        # --------------------------------------------------

        estimated_liquidity = data.get(
            "estimated_liquidity",
            data.get(
                "liquidity_uv",
                data.get(
                    "balance_uv",
                    None
                )
            )
        )

        if estimated_liquidity is not None:

            try:

                known_liquidities.append(
                    float(
                        estimated_liquidity
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                pass

        # --------------------------------------------------
        # Reliability
        # --------------------------------------------------

        failure_probability = reliability_penalty(
            data
        )

        failure_probability = min(
            max(
                float(
                    failure_probability
                ),
                0.0
            ),
            1.0
        )

        reliability *= (
            1.0
            -
            failure_probability
        )

        # --------------------------------------------------
        # Geographic / carbon metrics
        # --------------------------------------------------

        try:

            features = geo_features(
                G,
                u,
                v
            )

            total_distance += float(
                features.get(
                    "distance_km",
                    0.0
                )
            )

            # Use the node-based rgb_color values directly
            # rather than relying on a separately invented
            # carbon value in the topology module.

            carbon_u = node_carbon_intensity(
                G,
                u
            )

            carbon_v = node_carbon_intensity(
                G,
                v
            )

            total_carbon += (
                carbon_u
                +
                carbon_v
            ) / 2.0

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            pass

    # ------------------------------------------------------
    # Minimum Known Liquidity
    # ------------------------------------------------------

    if known_liquidities:

        min_liquidity = min(
            known_liquidities
        )

    else:

        min_liquidity = None

    return {

        "fee":
            total_fee,

        "delay":
            total_delay,

        "min_liquidity":
            min_liquidity,

        "reliability":
            reliability,

        "distance_km":
            total_distance,

        "carbon_intensity":
            total_carbon
    }