"""
Network topology and channel feature extraction.

This module provides:

    - geographical distance
    - dataset-backed node carbon intensity
    - node topology features
    - channel-level routing features

Carbon intensity is assigned to nodes from the geographic energy-mix
dataset by ``LNGraphBuilder``. RGB remains visual snapshot metadata and
is never used as a carbon value.
"""


import math


# ==========================================================
# Geographic Distance
# ==========================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Calculate geographical distance between two nodes.

    Haversine formula:

        a =
            sin²(dlat/2)
            +
            cos(lat1)
            cos(lat2)
            sin²(dlon/2)

        c =
            2 atan2(sqrt(a), sqrt(1-a))

        distance =
            R * c

    Earth radius:

        R = 6371 km
    """

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Latitude and longitude must be numeric."
        ) from exc

    values = {
        "lat1": lat1,
        "lon1": lon1,
        "lat2": lat2,
        "lon2": lon2,
    }

    for name, value in values.items():

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite; got {value!r}."
            )

    if not -90.0 <= lat1 <= 90.0:
        raise ValueError(
            f"lat1 must be in [-90,90]; got {lat1}."
        )

    if not -90.0 <= lat2 <= 90.0:
        raise ValueError(
            f"lat2 must be in [-90,90]; got {lat2}."
        )

    if not -180.0 <= lon1 <= 180.0:
        raise ValueError(
            f"lon1 must be in [-180,180]; got {lon1}."
        )

    if not -180.0 <= lon2 <= 180.0:
        raise ValueError(
            f"lon2 must be in [-180,180]; got {lon2}."
        )

    radius_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2.0) ** 2
        +
        math.cos(phi1)
        *
        math.cos(phi2)
        *
        math.sin(dlambda / 2.0) ** 2
    )

    # Numerical protection against tiny floating-point
    # excursions outside [0,1].
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
            math.sqrt(1.0 - a),
        )
    )

    return float(
        radius_km * c
    )


# ==========================================================
# RGB Parsing
# ==========================================================

def _parse_rgb_color(value):
    """
    Parse a snapshot RGB color.

    Supported formats:

        1. Dictionary:
            {"r": R, "g": G, "b": B}

        2. Dictionary:
            {"red": R, "green": G, "blue": B}

        3. List / tuple:
            [R, G, B]

        4. Hex:
            "#RRGGBB"

        5. Hex with alpha:
            "#RRGGBBAA"

        6. Hex integer:
            "0xRRGGBB"

        7. RGB string:
            "rgb(R,G,B)"

        8. Space/comma-separated:
            "R G B"
            "R,G,B"

    RGB values are returned as floats in [0,255].

    Important:
        This function returns None when the supplied value is
        not a recognized RGB representation.

        The caller decides whether missing/invalid RGB should
        be an error.
    """

    if value is None:
        return None

    # ------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------

    if isinstance(value, dict):

        candidates = (
            ("r", "g", "b"),
            ("red", "green", "blue"),
        )

        for keys in candidates:

            if all(
                key in value
                for key in keys
            ):

                try:

                    rgb = tuple(
                        float(value[key])
                        for key in keys
                    )

                except (TypeError, ValueError):
                    continue

                if not all(
                    math.isfinite(x)
                    for x in rgb
                ):
                    continue

                # Normalize [0,1] representation.
                if all(
                    0.0 <= x <= 1.0
                    for x in rgb
                ):
                    rgb = tuple(
                        x * 255.0
                        for x in rgb
                    )

                if not all(
                    0.0 <= x <= 255.0
                    for x in rgb
                ):
                    continue

                return rgb

        return None

    # ------------------------------------------------------
    # List / Tuple
    # ------------------------------------------------------

    if isinstance(
        value,
        (tuple, list),
    ):

        if len(value) < 3:
            return None

        try:

            rgb = tuple(
                float(value[index])
                for index in range(3)
            )

        except (TypeError, ValueError):
            return None

        if not all(
            math.isfinite(x)
            for x in rgb
        ):
            return None

        # Normalize [0,1] representation.
        if all(
            0.0 <= x <= 1.0
            for x in rgb
        ):
            rgb = tuple(
                x * 255.0
                for x in rgb
            )

        if not all(
            0.0 <= x <= 255.0
            for x in rgb
        ):
            return None

        return rgb

    # ------------------------------------------------------
    # String
    # ------------------------------------------------------

    if isinstance(value, str):

        text = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

        # ----------------------------------------------
        # #RRGGBB
        # ----------------------------------------------

        if text.startswith("#"):

            hex_value = text[1:]

            if len(hex_value) == 6:

                try:

                    rgb = tuple(
                        int(
                            hex_value[index:index + 2],
                            16,
                        )
                        for index in (0, 2, 4)
                    )

                except ValueError:
                    return None

                return tuple(
                    float(x)
                    for x in rgb
                )

            # ------------------------------------------
            # #RRGGBBAA
            # ------------------------------------------

            if len(hex_value) == 8:

                try:

                    rgb = tuple(
                        int(
                            hex_value[index:index + 2],
                            16,
                        )
                        for index in (0, 2, 4)
                    )

                except ValueError:
                    return None

                return tuple(
                    float(x)
                    for x in rgb
                )

            return None

        # ----------------------------------------------
        # 0xRRGGBB
        # ----------------------------------------------

        if (
            text.lower().startswith("0x")
            and len(text) == 8
        ):

            try:

                number = int(
                    text,
                    16,
                )

            except ValueError:
                return None

            return (
                float((number >> 16) & 255),
                float((number >> 8) & 255),
                float(number & 255),
            )

        # ----------------------------------------------
        # rgb(R,G,B)
        # ----------------------------------------------

        if (
            text.lower().startswith("rgb(")
            and text.endswith(")")
        ):

            text = text[4:-1]

        # ----------------------------------------------
        # Generic numeric string
        # ----------------------------------------------

        parts = [
            part.strip()
            for part in text.replace(
                ",",
                " ",
            ).split()
        ]

        if len(parts) >= 3:

            try:

                rgb = tuple(
                    float(parts[index])
                    for index in range(3)
                )

            except (TypeError, ValueError):
                return None

            if not all(
                math.isfinite(x)
                for x in rgb
            ):
                return None

            # Normalize [0,1] representation.
            if all(
                0.0 <= x <= 1.0
                for x in rgb
            ):
                rgb = tuple(
                    x * 255.0
                    for x in rgb
                )

            if not all(
                0.0 <= x <= 255.0
                for x in rgb
            ):
                return None

            return rgb

    return None


# ==========================================================
# Node RGB
# ==========================================================

def node_rgb_color(
    G,
    node,
):
    """
    Return the RGB color stored by the GML snapshot.

    Candidate node attributes:

        rgb_color
        color
        fill

    Nested graphics attributes are also supported.

    Returns:

        (R, G, B)

    where each value is an integer in [0,255].

    Missing or invalid RGB data raise an exception.
    """

    if node not in G:
        raise KeyError(
            f"Node {node!r} does not exist in graph."
        )

    attrs = G.nodes[node]

    candidates = [
        attrs.get("rgb_color"),
        attrs.get("color"),
        attrs.get("fill"),
    ]

    graphics = attrs.get(
        "graphics"
    )

    if isinstance(
        graphics,
        dict,
    ):

        candidates.extend([
            graphics.get("fill"),
            graphics.get("color"),
            graphics.get("rgb_color"),
        ])

    for value in candidates:

        rgb = _parse_rgb_color(
            value
        )

        if rgb is not None:

            return tuple(
                int(round(x))
                for x in rgb
            )

    raise ValueError(
        f"Node {node!r} has no valid RGB "
        "color attribute in the snapshot."
    )


# ==========================================================
# Node RGB Scalar / Carbon Proxy
# ==========================================================

def node_carbon_intensity(
    G,
    node,
):
    """
    Return validated dataset-backed carbon intensity for a node.

    The expected dataset unit is gCO2/kWh. RGB/color attributes
    are intentionally ignored.
    """

    if node not in G:
        raise KeyError(f"Node {node!r} does not exist in graph.")

    attrs = G.nodes[node]
    if "carbon_intensity" not in attrs:
        raise ValueError(
            f"Node {node!r} has no dataset-backed carbon_intensity."
        )

    try:
        intensity = float(attrs["carbon_intensity"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"carbon_intensity for node {node!r} must be numeric."
        ) from exc

    if not math.isfinite(intensity) or intensity < 0.0:
        raise ValueError(
            f"carbon_intensity for node {node!r} must be finite and >= 0; "
            f"got {intensity}."
        )

    return intensity


# ==========================================================
# Geographic Features
# ==========================================================

def geo_features(
    G,
    u,
    v,
):
    """
    Extract geographic and dataset-backed carbon features between nodes.

    Returned features:

        distance_km
        inter_country
        inter_continent
        carbon_intensity_u
        carbon_intensity_v
        carbon_intensity

    Edge intensity is the mean of its endpoint intensities:

        C_uv = (C_u + C_v) / 2
    """

    if u not in G:
        raise KeyError(
            f"Node {u!r} does not exist in graph."
        )

    if v not in G:
        raise KeyError(
            f"Node {v!r} does not exist in graph."
        )

    node_u = G.nodes[u]
    node_v = G.nodes[v]

    # ------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------

    required_geo = (
        "latitude",
        "longitude",
    )

    for field in required_geo:

        if field not in node_u:
            raise KeyError(
                f"Missing '{field}' for node {u!r}."
            )

        if field not in node_v:
            raise KeyError(
                f"Missing '{field}' for node {v!r}."
            )

    distance = haversine_km(
        node_u["latitude"],
        node_u["longitude"],
        node_v["latitude"],
        node_v["longitude"],
    )

    # ------------------------------------------------------
    # Country
    # ------------------------------------------------------

    if "country" not in node_u:
        raise KeyError(
            f"Missing 'country' for node {u!r}."
        )

    if "country" not in node_v:
        raise KeyError(
            f"Missing 'country' for node {v!r}."
        )

    inter_country = int(
        node_u["country"]
        !=
        node_v["country"]
    )

    # ------------------------------------------------------
    # Continent
    # ------------------------------------------------------
    #
    # Use the dataset-derived continent codes when available.
    # ------------------------------------------------------

    continent_u = node_u.get("continent_code")
    continent_v = node_v.get("continent_code")
    if continent_u and continent_v:
        inter_continent = int(continent_u != continent_v)
    else:
        inter_continent = int(distance > 3000.0)

    # ------------------------------------------------------
    # Dataset-backed carbon
    # ------------------------------------------------------

    carbon_u = node_carbon_intensity(
        G,
        u,
    )

    carbon_v = node_carbon_intensity(
        G,
        v,
    )

    carbon_intensity = (
        carbon_u
        +
        carbon_v
    ) / 2.0

    return {
        "distance_km": float(
            distance
        ),

        "inter_country":
            inter_country,

        "inter_continent":
            inter_continent,

        "carbon_intensity_u":
            float(carbon_u),

        "carbon_intensity_v":
            float(carbon_v),

        "carbon_intensity":
            float(carbon_intensity),
    }


# ==========================================================
# Topology Features
# ==========================================================

def topology_features(
    G,
    node,
):
    """
    Extract node-level topology features.

    Returns:

        degree
        in_degree
        out_degree
        clustering

    Note:
        The original implementation called degree(node)
        "clustering". That was semantically incorrect.

    The corrected implementation computes the actual
    clustering coefficient when NetworkX provides it.
    """

    if node not in G:
        raise KeyError(
            f"Node {node!r} does not exist in graph."
        )

    degree = G.degree(
        node
    )

    in_degree = G.in_degree(
        node
    )

    out_degree = G.out_degree(
        node
    )

    # NetworkX clustering is defined on an undirected
    # representation for this topology feature.
    undirected = G.to_undirected()

    if degree == 0:
        clustering = 0.0

    else:
        clustering = float(
            __import__(
                "networkx"
            ).clustering(
                undirected,
                node,
            )
        )

    return {
        "degree":
            int(degree),

        "in_degree":
            int(in_degree),

        "out_degree":
            int(out_degree),

        "clustering":
            clustering,
    }


# ==========================================================
# Channel Resolution
# ==========================================================

def _resolve_channel(
    G,
    u,
    v,
    key=None,
):
    """
    Resolve an exact channel.

    For MultiDiGraph:

        key provided
            -> exact channel is returned.

        key omitted and exactly one channel exists
            -> that channel is returned.

        key omitted and multiple channels exist
            -> ValueError.

    No arbitrary first-channel fallback is allowed.
    """

    if not G.has_edge(
        u,
        v,
    ):
        raise KeyError(
            f"Channel ({u!r}, {v!r}) does not exist."
        )

    channels = G[u][v]

    # ------------------------------------------------------
    # MultiDiGraph
    # ------------------------------------------------------

    if (
        hasattr(
            channels,
            "items",
        )
        and channels
        and all(
            isinstance(value, dict)
            for value in channels.values()
        )
    ):

        if key is not None:

            if key not in channels:
                raise KeyError(
                    f"Channel key {key!r} does not exist "
                    f"for ({u!r}, {v!r})."
                )

            return channels[key]

        keys = list(
            channels.keys()
        )

        if len(keys) != 1:
            raise ValueError(
                f"Multiple channels exist between "
                f"({u!r}, {v!r}); an explicit key "
                "is required."
            )

        return channels[keys[0]]

    # ------------------------------------------------------
    # Simple graph
    # ------------------------------------------------------

    if key is not None:
        raise ValueError(
            f"key={key!r} was supplied for a non-multichannel edge."
        )

    return channels


# ==========================================================
# Channel Features
# ==========================================================

def channel_features(
    G,
    u,
    v,
    key=None,
):
    """
    Extract channel-level features.

    Used by the RL agent.

    Required fields:

        capacity
        fee_base / fee_base_msat
        fee_rate / fee_proportional_millionths
        delay / cltv_expiry_delta
        available
        balance_uv

    Liquidity:

        liquidity_uv =
            balance_uv / capacity

    when capacity > 0.

    Liquidity remains a channel-feasibility feature and is not
    incorporated into the additive adaptive cost in heuristics.py.
    """

    channel = _resolve_channel(
        G,
        u,
        v,
        key,
    )

    # ------------------------------------------------------
    # Capacity
    # ------------------------------------------------------

    if "capacity" not in channel:
        raise KeyError(
            f"Missing 'capacity' for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    capacity = float(
        channel["capacity"]
    )

    if not math.isfinite(
        capacity
    ):
        raise ValueError(
            f"Invalid capacity for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    if capacity < 0.0:
        raise ValueError(
            f"Capacity must be >= 0; got {capacity}."
        )

    # ------------------------------------------------------
    # Fee base
    # ------------------------------------------------------

    if "fee_base" in channel:

        fee_base = float(
            channel["fee_base"]
        )

    elif "fee_base_msat" in channel:

        fee_base = float(
            channel["fee_base_msat"]
        )

    else:

        raise KeyError(
            f"Missing fee base for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    # ------------------------------------------------------
    # Fee rate
    # ------------------------------------------------------

    if "fee_rate" in channel:

        fee_rate = float(
            channel["fee_rate"]
        )

    elif "fee_proportional_millionths" in channel:

        fee_rate = float(
            channel[
                "fee_proportional_millionths"
            ]
        )

    else:

        raise KeyError(
            f"Missing fee rate for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    if not math.isfinite(
        fee_base
    ) or fee_base < 0.0:
        raise ValueError(
            f"Invalid fee_base: {fee_base}."
        )

    if not math.isfinite(
        fee_rate
    ) or fee_rate < 0.0:
        raise ValueError(
            f"Invalid fee_rate: {fee_rate}."
        )

    # ------------------------------------------------------
    # Delay
    # ------------------------------------------------------

    if "delay" in channel:

        delay = float(
            channel["delay"]
        )

    elif "cltv_expiry_delta" in channel:

        delay = float(
            channel[
                "cltv_expiry_delta"
            ]
        )

    else:

        raise KeyError(
            f"Missing delay for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    if not math.isfinite(
        delay
    ) or delay < 0.0:
        raise ValueError(
            f"Invalid delay: {delay}."
        )

    # ------------------------------------------------------
    # Availability
    # ------------------------------------------------------

    if "available" not in channel:
        raise KeyError(
            f"Missing 'available' for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    available = channel[
        "available"
    ]

    if isinstance(
        available,
        bool,
    ):

        available_int = int(
            available
        )

    else:

        available_int = int(
            float(available)
        )

        if available_int not in (
            0,
            1,
        ):
            raise ValueError(
                f"'available' must be 0 or 1; "
                f"got {available!r}."
            )

    # ------------------------------------------------------
    # Directional balance
    # ------------------------------------------------------

    if "balance_uv" not in channel:
        raise KeyError(
            f"Missing 'balance_uv' for channel "
            f"({u!r}, {v!r}, {key!r})."
        )

    balance_uv = float(
        channel[
            "balance_uv"
        ]
    )

    if not math.isfinite(
        balance_uv
    ) or balance_uv < 0.0:
        raise ValueError(
            f"Invalid balance_uv: {balance_uv}."
        )

    # ------------------------------------------------------
    # Liquidity ratio
    # ------------------------------------------------------

    if capacity > 0.0:

        liquidity_uv = (
            balance_uv
            /
            capacity
        )

        # A directional balance greater than channel capacity
        # indicates inconsistent snapshot data.
        if liquidity_uv > 1.0:
            raise ValueError(
                f"Directional liquidity ratio exceeds 1: "
                f"{liquidity_uv}."
            )

    else:

        if balance_uv != 0.0:
            raise ValueError(
                "A zero-capacity channel cannot have "
                "non-zero balance_uv."
            )

        liquidity_uv = 0.0

    return {
        "capacity":
            float(capacity),

        "fee_base":
            float(fee_base),

        "fee_rate":
            float(fee_rate),

        "delay":
            float(delay),

        "available":
            int(available_int),

        "liquidity_uv":
            float(liquidity_uv),
    }
