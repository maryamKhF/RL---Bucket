import math


def haversine_km(
        lat1,
        lon1,
        lat2,
        lon2
):
    """
    Calculate geographical distance
    between two nodes.
    """

    R = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(dl / 2) ** 2
    )

    return (
        2 *
        R *
        math.asin(math.sqrt(a))
    )



def geo_features(
        G,
        u,
        v
):
    """
    Extract geographical features
    between two nodes.
    """


    node_u = G.nodes[u]
    node_v = G.nodes[v]


    distance = haversine_km(

        node_u["latitude"],
        node_u["longitude"],

        node_v["latitude"],
        node_v["longitude"]
    )


    inter_country = int(
        node_u["country"]
        !=
        node_v["country"]
    )


    inter_continent = int(
        distance > 3000
    )


    # The real snapshot does not contain numeric
    # carbon-intensity data. regcolor is the selected
    # node-level environmental proxy.
    carbon = (
        float(node_u.get("regcolor", 0.0))
        +
        float(node_v.get("regcolor", 0.0))
    ) / 2



    return {

        "distance_km": distance,

        "inter_country":
            inter_country,

        "inter_continent":
            inter_continent,

        "carbon_intensity":
            carbon
    }



def topology_features(
        G,
        node
):
    """
    Extract node-level topology features.
    """


    return {

        "degree":
            G.degree(node),

        "in_degree":
            G.in_degree(node),

        "out_degree":
            G.out_degree(node),

        "clustering":
            0 if G.degree(node)==0
            else (
                G.to_undirected()
                .degree(node)
            )
    }



def channel_features(
        G,
        u,
        v,
        key=None
):
    """
    Extract channel-level features.

    Used by RL agent.
    """


    if key:

        channel = G[u][v][key]

    else:

        channel = list(
            G[u][v].values()
        )[0]


    return {

        "capacity":
            channel["capacity"],

        "fee_base":
            channel["fee_base"],

        "fee_rate":
            channel["fee_rate"],

        "delay":
            channel["delay"],

        "failure_probability":
            channel["failure_probability"],

        "available":
            int(channel["available"]),

        "liquidity_uv":
            channel["balance_uv"] /
            channel["capacity"]
            if channel["capacity"] > 0
            else 0
    }
