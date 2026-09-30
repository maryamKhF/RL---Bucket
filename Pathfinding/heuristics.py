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
        Proportional fee in ppm.
    """

    amount = float(amount)

    fee_base = float(
        data.get(
            "fee_base",
            data.get("fee_base_msat", 0.0)
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
        capacity is NOT interpreted as directional balance.

    Only learned/observed directional liquidity can be used.
    """

    estimated_liquidity = data.get(
        "estimated_liquidity",
        data.get(
            "liquidity_uv",
            data.get("balance_uv", None)
        )
    )

    if estimated_liquidity is None:
        return 0.0

    estimated_liquidity = float(
        estimated_liquidity
    )

    if estimated_liquidity <= 0:
        return math.inf

    return float(amount) / estimated_liquidity


# ==========================================================
# Reliability
# ==========================================================

def reliability_penalty(data):
    """
    Empirical channel failure probability.

    If previous success/failure observations exist,
    estimate the probability from those observations.

    Otherwise use the initial prior.
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
        return failure / total

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
    Base LND-like routing cost.

    This is the native routing cost C_LND(e).

    PPO does NOT directly select the route.
    PPO only produces eta, which is later used
    to modify this cost through the heuristic h.
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
    CLN-like base routing cost.

    Kept for baseline evaluation.
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
    ECL-like base routing cost.

    Kept for baseline evaluation.
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
    Return carbon intensity of a node.

    README defines the heuristic using
    intCO2 of the source and destination nodes.
    """

    node_data = G.nodes[node]

    value = node_data.get(
        "carbon_intensity",
        0.0
    )

    return float(value)


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
    Adaptive routing heuristic defined in README.

    h(u,v) =
        eta * ((intCO2[u] + intCO2[v]) / 2)
        +
        (1 - eta) * (intCO2[v] - intCO2[u])

    eta is produced dynamically by PPO.

    Valid range:
        -1 <= eta <= 1
    """

    eta = float(eta)

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
        carbon_u + carbon_v
    ) / 2.0

    carbon_difference = (
        carbon_v - carbon_u
    )

    return (
        eta * average_carbon
        +
        (1.0 - eta) * carbon_difference
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
    Return the geographic information of an edge.

    The adaptive heuristic itself is NOT based on the
    previous composite geographic penalty.

    This function is retained for compatibility with
    existing evaluation code.
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
    Final modified routing cost.

    README:
        C'(e) = C_LND(e) + lambda_h * h(e)

    Important:
        geo_penalty is actually the adaptive heuristic h(e).

    Parameters
    ----------
    native_cost:
        Base LND routing cost C_LND(e).

    geo_penalty:
        Adaptive heuristic h(e).

    eta:
        PPO-generated adaptive parameter.

    lambda_h:
        Weight of the heuristic component.
    """

    eta = float(eta)

    if not -1.0 <= eta <= 1.0:
        raise ValueError(
            f"eta must be in [-1, 1], got {eta}"
        )

    h = float(
        geo_penalty
    )

    cost = (
        float(native_cost)
        +
        float(lambda_h) * h
    )

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
    Complete routing cost used by the
    PPO + LND/Dijkstra architecture.

    Flow:

        PPO
         |
         v
        eta
         |
         v
        h(u,v)
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
    Extended cost for optional experiments.

    This is NOT the main PPO heuristic from README.

    It is retained only for auxiliary experiments/baselines.
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

    if math.isinf(liquidity):
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
    Calculate complete path metrics.

    Liquidity is based only on known/learned
    directional liquidity.

    Snapshot capacity is NOT treated as balance.
    """

    total_fee = 0.0
    total_delay = 0.0

    known_liquidities = []

    reliability = 1.0

    total_distance = 0.0
    total_carbon = 0.0

    for u, v, k in edges:

        data = G[u][v][k]

        # ----------------------------------------------
        # Fee
        # ----------------------------------------------

        total_fee += channel_fee(
            data,
            amount
        )

        # ----------------------------------------------
        # Delay
        # ----------------------------------------------

        total_delay += float(
            data.get(
                "delay",
                data.get(
                    "cltv_expiry_delta",
                    0.0
                )
            )
        )

        # ----------------------------------------------
        # Learned / observed directional liquidity
        # ----------------------------------------------

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

            known_liquidities.append(
                float(
                    estimated_liquidity
                )
            )

        # ----------------------------------------------
        # Reliability
        # ----------------------------------------------

        failure_probability = min(
            max(
                reliability_penalty(data),
                0.0
            ),
            1.0
        )

        reliability *= (
            1.0
            -
            failure_probability
        )

        # ----------------------------------------------
        # Geographic metrics
        # ----------------------------------------------

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

            total_carbon += float(
                features.get(
                    "carbon_intensity",
                    0.0
                )
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            pass

    # ------------------------------------------------------
    # Minimum known liquidity
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