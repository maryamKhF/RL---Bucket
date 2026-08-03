import math

from Network.topology import geo_features


# =====================================================
# Channel Fee
# =====================================================

def channel_fee(data, amount):
    """
    Lightning forwarding fee.

    fee_rate is ppm.
    """

    return (
        float(data.get("fee_base", 0))
        +
        float(data.get("fee_rate", 0))
        *
        amount
        /
        1_000_000.0
    )


# =====================================================
# Liquidity
# =====================================================

def liquidity_penalty(data, amount):
    """
    Penalize channels with low liquidity.
    """

    balance = float(
        data.get(
            "balance_uv",
            data.get("capacity", 0)
        )
    )

    if balance <= 0:
        return math.inf

    ratio = amount / balance

    return ratio



# =====================================================
# Reliability
# =====================================================

def reliability_penalty(data):
    """
    Channel failure probability.
    """

    success = data.get(
        "success_count",
        0
    )

    failure = data.get(
        "failure_count",
        0
    )

    total = success + failure


    if total == 0:
        return float(
            data.get(
                "failure_probability",
                0.01
            )
        )


    return failure / total



# =====================================================
# Routing Cost Models
# =====================================================

def lnd_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    LND-like routing cost.
    """

    return (
        channel_fee(
            data,
            amount
        )

        +

        0.5
        *
        float(
            data.get(
                "delay",
                0
            )
        )

        +

        1.0
    )



def cln_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    CLN-like routing cost.
    """

    return (
        channel_fee(
            data,
            amount
        )

        +

        float(
            data.get(
                "delay",
                0
            )
        )

        +

        0.7
    )



def ecl_cost(
        G,
        u,
        v,
        data,
        amount
):
    """
    Energy/efficiency aware routing cost.
    """

    return (

        channel_fee(
            data,
            amount
        )

        +

        0.8
        *
        float(
            data.get(
                "delay",
                0
            )
        )

        +

        1.3
    )



# =====================================================
# Enhanced Cost
# =====================================================

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
    Extended Lightning routing cost.

    Used for RL compatible routing.
    """

    return (

        fee_weight
        *
        lnd_cost(
            G,
            u,
            v,
            data,
            amount
        )

        +

        liquidity_weight
        *
        liquidity_penalty(
            data,
            amount
        )

        +

        reliability_weight
        *
        reliability_penalty(
            data
        )

    )



# =====================================================
# Geographic Penalty
# =====================================================

def geographic_penalty(
        G,
        u,
        v,
        data
):
    """
    Geographic and environmental penalty.
    """


    features = geo_features(
        G,
        u,
        v
    )


    distance = float(
        features["distance_km"]
    )


    carbon = float(
        features["carbon_intensity"]
    )


    inter_country = float(
        features["inter_country"]
    )


    inter_continent = float(
        features["inter_continent"]
    )


    return (

        carbon / 500.0

        +

        0.25 * inter_country

        +

        0.5 * inter_continent

        +

        distance / 10000.0

    )



# =====================================================
# Final Modified Cost
# =====================================================

def modified_cost(
        native_cost,
        geo_penalty,
        eta
):
    """
    Final edge cost used by Dijkstra.

    Cost =
        routing cost
        +
        geographic penalty
    """

    return max(
        1e-9,
        native_cost
        +
        float(eta)
        *
        geo_penalty
    )



# =====================================================
# Path Evaluation
# =====================================================

def evaluate_path(
        G,
        edges,
        amount
):
    """
    Calculate complete path metrics.

    Used by:
        - Bucket
        - Evaluation
        - RL
    """

    total_fee = 0
    total_delay = 0

    min_liquidity = math.inf

    reliability = 1.0


    for u,v,k in edges:

        data = G[u][v][k]


        total_fee += channel_fee(
            data,
            amount
        )


        total_delay += data.get(
            "delay",
            0
        )


        min_liquidity = min(
            min_liquidity,
            data.get(
                "balance_uv",
                0
            )
        )


        reliability *= (
            1 -
            reliability_penalty(data)
        )


    return {

        "fee": total_fee,

        "delay": total_delay,

        "min_liquidity": min_liquidity,

        "reliability": reliability

    }