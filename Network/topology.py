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




def _parse_rgb_color(value):
    """Parse a snapshot RGB color into an (R, G, B) tuple in [0, 255]."""
    if value is None:
        return None

    if isinstance(value, dict):
        # Common nested forms: {r,g,b}, {red,green,blue}
        for keys in (("r", "g", "b"), ("red", "green", "blue")):
            if all(k in value for k in keys):
                try:
                    rgb = tuple(float(value[k]) for k in keys)
                    if all(math.isfinite(x) for x in rgb):
                        if max(rgb) <= 1.0:
                            rgb = tuple(x * 255.0 for x in rgb)
                        return tuple(max(0.0, min(255.0, x)) for x in rgb)
                except (TypeError, ValueError):
                    pass
        return None

    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            rgb = tuple(float(value[i]) for i in range(3))
            if all(math.isfinite(x) for x in rgb):
                if max(rgb) <= 1.0:
                    rgb = tuple(x * 255.0 for x in rgb)
                return tuple(max(0.0, min(255.0, x)) for x in rgb)
        except (TypeError, ValueError):
            return None

    if isinstance(value, str):
        s = value.strip().strip('"').strip("'")
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 6:
                try:
                    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                except ValueError:
                    return None
            if len(h) == 8:
                try:
                    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                except ValueError:
                    return None
        if s.lower().startswith("0x") and len(s) == 8:
            try:
                n = int(s, 16)
                return ((n >> 16) & 255, (n >> 8) & 255, n & 255)
            except ValueError:
                return None
        if s.lower().startswith("rgb(") and s.endswith(")"):
            s = s[4:-1]
        parts = [p.strip() for p in s.replace(",", " ").split()]
        if len(parts) >= 3:
            try:
                rgb = tuple(float(parts[i]) for i in range(3))
                if all(math.isfinite(x) for x in rgb):
                    if max(rgb) <= 1.0:
                        rgb = tuple(x * 255.0 for x in rgb)
                    return tuple(max(0.0, min(255.0, x)) for x in rgb)
            except (TypeError, ValueError):
                pass

    return None


def node_rgb_color(G, node):
    """Return the RGB color stored by the real GML snapshot."""
    if node not in G:
        raise KeyError(f"Node {node} does not exist in graph.")

    attrs = G.nodes[node]
    candidates = [
        attrs.get("rgb_color"),
        attrs.get("color"),
        attrs.get("fill"),
    ]

    graphics = attrs.get("graphics")
    if isinstance(graphics, dict):
        candidates.extend([
            graphics.get("fill"),
            graphics.get("color"),
            graphics.get("rgb_color"),
        ])

    for value in candidates:
        rgb = _parse_rgb_color(value)
        if rgb is not None:
            return tuple(int(round(x)) for x in rgb)

    raise ValueError(
        f"Node {node} has no valid RGB color attribute in the snapshot."
    )


def node_rgb_scalar(G, node):
    """Convert snapshot RGB to a scalar luminance in [0, 255]."""
    r, g, b = node_rgb_color(G, node)
    return float(0.299 * r + 0.587 * g + 0.114 * b)

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


    # The snapshot has no numeric carbon-intensity data.
    # Use the node RGB Color attribute from GML instead.
    rgb_u = node_rgb_color(G, u)
    rgb_v = node_rgb_color(G, v)
    rgb_luminance = (
        node_rgb_scalar(G, u)
        +
        node_rgb_scalar(G, v)
    ) / 2.0



    return {

        "distance_km": distance,

        "inter_country":
            inter_country,

        "inter_continent":
            inter_continent,

        "rgb_color_u": rgb_u,
        "rgb_color_v": rgb_v,
        "rgb_luminance": rgb_luminance,

        # Backward-compatible alias for older callers.
        "carbon_intensity": rgb_luminance
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
