
# Pathfinding/test_pathfinding.py

import math
import random
import sys
from pathlib import Path

import networkx as nx


# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Repository Imports
# ==========================================================

from Pathfinding.top_k_paths import top_k_paths

from Pathfinding.heuristics import (
    lnd_cost,
    adaptive_edge_cost,
    channel_fee,
    validate_eta,
    validate_lambda_h,
)


# ==========================================================
# Configuration
# ==========================================================

GML_FILE = PROJECT_ROOT / "20190501.gml.geo"

K = 5
MAX_HOPS = 12

# PPO adaptive parameter.
# Current invariant:
#     0 <= ETA <= 1
ETA = 0.5

# Weight of the normalized adaptive penalty.
LAMBDA_H = 1.0

# Payment amount.
AMOUNT = 10_000

# Reproducibility.
RANDOM_SEED = 42


# ==========================================================
# Utility Functions
# ==========================================================

def print_separator(
    char="=",
    length=78,
):
    print(char * length)


def print_header(
    title,
):
    print()
    print_separator()
    print(title)
    print_separator()


def safe_float(
    value,
    default=0.0,
):
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


# ==========================================================
# Strict Boolean Validation
# ==========================================================

def strict_bool(
    value,
    field_name,
):
    """
    Accept actual Boolean values only.

    Invalid examples:

        "true"
        "false"
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
# Edge Record Helper
# ==========================================================

def get_edge_identity(
    edge,
):
    """
    Extract the canonical edge identity from a Top-K edge
    dictionary.

    Current Top-K edge contract:

        {
            "source": ...,
            "target": ...,
            "channel_key": ...,
            "scid": ...
        }

    Returns:

        source,
        target,
        channel_key

    or:

        None, None, None

    if the edge record is invalid.
    """

    if not isinstance(
        edge,
        dict,
    ):
        return (
            None,
            None,
            None,
        )

    source = edge.get(
        "source"
    )

    target = edge.get(
        "target"
    )

    channel_key = edge.get(
        "channel_key"
    )

    if (
        source is None
        or
        target is None
        or
        channel_key is None
    ):
        return (
            None,
            None,
            None,
        )

    return (
        source,
        target,
        channel_key,
    )


# ==========================================================
# Directional Liquidity Helper
# ==========================================================

def get_directional_liquidity(
    data,
):
    """
    Return explicitly available directional liquidity.

    This schema is aligned with top_k_paths.py.

    Supported explicit attributes:

        estimated_liquidity
        liquidity_uv
        balance_uv

    IMPORTANT:

        Channel capacity is NOT interpreted as directional
        liquidity.

    If no explicit directional liquidity exists,
    the result is UNKNOWN.
    """

    explicit_keys = (
        "estimated_liquidity",
        "liquidity_uv",
        "balance_uv",
    )

    for key in explicit_keys:

        if key not in data:
            continue

        value = data.get(key)

        if value is None:
            continue

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not math.isfinite(value):
            continue

        if value < 0.0:
            continue

        return value

    return None


# ==========================================================
# Channel Reliability Helper
# ==========================================================

def get_channel_reliability(
    data,
):
    """
    Return an explicitly available channel reliability.

    The current Top-K implementation supports:

        success_count + failure_count

    or:

        failure_probability

    This helper mirrors that contract.

    Additional explicit reliability aliases are accepted only
    for diagnostic reporting:

        reliability
        reliability_score
        success_rate

    Missing reliability remains UNKNOWN.
    """

    # ------------------------------------------------------
    # Empirical success/failure history
    # ------------------------------------------------------

    has_success = (
        "success_count"
        in data
    )

    has_failure = (
        "failure_count"
        in data
    )

    if (
        has_success
        or
        has_failure
    ):

        try:
            success = float(
                data.get(
                    "success_count",
                    0.0,
                )
            )

            failure = float(
                data.get(
                    "failure_count",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            success = None
            failure = None

        if (
            success is not None
            and
            failure is not None
            and
            math.isfinite(success)
            and
            math.isfinite(failure)
            and
            success >= 0.0
            and
            failure >= 0.0
        ):

            total = (
                success
                +
                failure
            )

            if total > 0.0:

                return (
                    success
                    /
                    total
                )

    # ------------------------------------------------------
    # Explicit failure probability
    # ------------------------------------------------------

    if "failure_probability" in data:

        value = data.get(
            "failure_probability"
        )

        try:
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):
            value = None

        if (
            value is not None
            and
            math.isfinite(value)
            and
            0.0 <= value <= 1.0
        ):

            return 1.0 - value

    # ------------------------------------------------------
    # Diagnostic aliases
    # ------------------------------------------------------

    for key in (
        "reliability",
        "reliability_score",
        "success_rate",
    ):

        if key not in data:
            continue

        value = data.get(key)

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
            0.0 <= value <= 1.0
        ):

            return value

    return None


# ==========================================================
# Routing Data Readiness
# ==========================================================

def validate_routing_data_readiness(
    G,
):
    """
    Verify whether the graph contains the routing data
    required by top_k_paths.py.

    Required per usable channel:

        directional liquidity
        OR
        explicit reliability information

    IMPORTANT:

        capacity is NOT accepted as directional liquidity.

    This function does not modify the graph and does not
    synthesize missing values.

    Returns:

        {
            "ready": bool,
            "total_edges": int,
            "available_edges": int,
            "liquidity_known": int,
            "liquidity_unknown": int,
            "reliability_known": int,
            "reliability_unknown": int,
            "both_known": int,
            "capacity_only": int,
        }
    """

    total_edges = 0
    available_edges = 0

    liquidity_known = 0
    liquidity_unknown = 0

    reliability_known = 0
    reliability_unknown = 0

    both_known = 0
    capacity_only = 0

    for (
        _,
        _,
        _,
        data,
    ) in G.edges(
        keys=True,
        data=True,
    ):

        total_edges += 1

        # --------------------------------------------------
        # Availability
        # --------------------------------------------------

        if "available" in data:

            available = strict_bool(
                data["available"],
                "edge.available",
            )

        else:

            available = True

        if not available:
            continue

        available_edges += 1

        # --------------------------------------------------
        # Directional liquidity
        # --------------------------------------------------

        liquidity = get_directional_liquidity(
            data
        )

        if liquidity is not None:

            liquidity_known += 1

        else:

            liquidity_unknown += 1

            if "capacity" in data:

                capacity_only += 1

        # --------------------------------------------------
        # Reliability
        # --------------------------------------------------

        reliability = get_channel_reliability(
            data
        )

        if reliability is not None:

            reliability_known += 1

        else:

            reliability_unknown += 1

        # --------------------------------------------------
        # Complete routing information
        # --------------------------------------------------

        if (
            liquidity is not None
            and
            reliability is not None
        ):

            both_known += 1

    # ------------------------------------------------------
    # Current Top-K contract
    # ------------------------------------------------------

    ready = (
        available_edges > 0
        and
        liquidity_known > 0
        and
        reliability_known > 0
    )

    return {
        "ready": ready,
        "total_edges": total_edges,
        "available_edges": available_edges,
        "liquidity_known": liquidity_known,
        "liquidity_unknown": liquidity_unknown,
        "reliability_known": reliability_known,
        "reliability_unknown": reliability_unknown,
        "both_known": both_known,
        "capacity_only": capacity_only,
    }


# ==========================================================
# Print Routing Data Readiness
# ==========================================================

def print_routing_data_readiness(
    G,
):
    """
    Print routing-data readiness before Top-K execution.

    This prevents a raw GML data problem from being reported
    as an algorithmic Top-K failure.
    """

    print_header(
        "4. ROUTING DATA READINESS"
    )

    readiness = validate_routing_data_readiness(
        G
    )

    print(
        f"Total channels           : "
        f"{readiness['total_edges']:,}"
    )

    print(
        f"Available channels       : "
        f"{readiness['available_edges']:,}"
    )

    print()

    print(
        "Directional liquidity:"
    )

    print(
        f"  Known                  : "
        f"{readiness['liquidity_known']:,}"
    )

    print(
        f"  Unknown                : "
        f"{readiness['liquidity_unknown']:,}"
    )

    print(
        f"  Capacity only          : "
        f"{readiness['capacity_only']:,}"
    )

    print()

    print(
        "Channel reliability:"
    )

    print(
        f"  Known                  : "
        f"{readiness['reliability_known']:,}"
    )

    print(
        f"  Unknown                : "
        f"{readiness['reliability_unknown']:,}"
    )

    print()

    print(
        f"Channels with both      : "
        f"{readiness['both_known']:,}"
    )

    print()

    print(
        "Data-readiness status   : "
        f"{'READY' if readiness['ready'] else 'NOT READY'}"
    )

    if not readiness["ready"]:

        print()
        print(
            "Top-K execution is skipped."
        )

        print(
            "Reason:"
        )

        if readiness["liquidity_known"] == 0:

            print(
                "  - No explicit directional liquidity "
                "is available."
            )

        if readiness["reliability_known"] == 0:

            print(
                "  - No explicit channel reliability "
                "information is available."
            )

        print()
        print(
            "Channel capacity is NOT used as a "
            "directional-liquidity fallback."
        )

        print(
            "No synthetic routing values are created."
        )

    return readiness


# ==========================================================
# Load GML
# ==========================================================

def load_graph():
    """
    Load the real Lightning snapshot.

    The Pathfinding module supports:

        Graph
        DiGraph
        MultiGraph
        MultiDiGraph

    The test converts the graph to MultiDiGraph so that
    channel keys are preserved.
    """

    print_header(
        "1. LOAD REAL GML SNAPSHOT"
    )

    print(
        "Loading:"
    )

    print(
        f"  {GML_FILE}"
    )

    if not GML_FILE.exists():

        raise FileNotFoundError(
            f"GML file not found:\n{GML_FILE}"
        )

    G = nx.read_gml(
        GML_FILE,
        label="id",
    )

    original_type = type(
        G
    ).__name__

    print()
    print(
        "Original graph type:"
    )

    print(
        f"  {original_type}"
    )

    print()
    print(
        "Original graph:"
    )

    print(
        f"  Nodes : "
        f"{G.number_of_nodes():,}"
    )

    print(
        f"  Edges : "
        f"{G.number_of_edges():,}"
    )

    # ======================================================
    # Convert to MultiDiGraph
    # ======================================================

    if isinstance(
        G,
        nx.MultiDiGraph,
    ):

        routing_graph = G

    elif isinstance(
        G,
        nx.DiGraph,
    ):

        routing_graph = nx.MultiDiGraph(
            G
        )

    elif isinstance(
        G,
        nx.MultiGraph,
    ):

        routing_graph = nx.MultiDiGraph(
            G
        )

    elif isinstance(
        G,
        nx.Graph,
    ):

        routing_graph = nx.MultiDiGraph(
            G
        )

    else:

        raise TypeError(
            "Unsupported graph type: "
            f"{type(G).__name__}"
        )

    print()
    print(
        "Routing graph:"
    )

    print(
        f"  Type  : "
        f"{type(routing_graph).__name__}"
    )

    print(
        f"  Nodes : "
        f"{routing_graph.number_of_nodes():,}"
    )

    print(
        f"  Edges : "
        f"{routing_graph.number_of_edges():,}"
    )

    # ======================================================
    # Direction warning
    # ======================================================

    if not isinstance(
        G,
        (
            nx.DiGraph,
            nx.MultiDiGraph,
        ),
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "  The source GML is not explicitly directed."
        )

        print(
            "  Reverse edges were created structurally "
            "for routing."
        )

        print(
            "  These reverse edges should not be interpreted "
            "as independent directional gossip policies."
        )

    return routing_graph


# ==========================================================
# Graph Statistics
# ==========================================================

def print_graph_statistics(
    G,
):
    """
    Print topology and routing-related statistics.
    """

    print_header(
        "2. GRAPH STATISTICS"
    )

    nodes = list(
        G.nodes
    )

    # ======================================================
    # Node Availability
    # ======================================================

    online_nodes = 0
    available_nodes = 0

    for node in nodes:

        data = G.nodes[
            node
        ]

        if "online" in data:

            online = strict_bool(
                data["online"],
                "node.online",
            )

        else:

            online = True

        if "available" in data:

            available = strict_bool(
                data["available"],
                "node.available",
            )

        else:

            available = True

        if online:
            online_nodes += 1

        if (
            online
            and
            available
        ):
            available_nodes += 1

    # ======================================================
    # Channels
    # ======================================================

    available_edges = 0
    unavailable_edges = 0

    for (
        _,
        _,
        _,
        data,
    ) in G.edges(
        keys=True,
        data=True,
    ):

        if "available" in data:

            available = strict_bool(
                data["available"],
                "edge.available",
            )

        else:

            available = True

        if available:
            available_edges += 1

        else:
            unavailable_edges += 1

    print(
        f"Total nodes              : "
        f"{G.number_of_nodes():,}"
    )

    print(
        f"Total directed channels  : "
        f"{G.number_of_edges():,}"
    )

    print(
        f"Online nodes             : "
        f"{online_nodes:,}"
    )

    print(
        f"Available nodes          : "
        f"{available_nodes:,}"
    )

    print(
        f"Available channels       : "
        f"{available_edges:,}"
    )

    print(
        f"Unavailable channels     : "
        f"{unavailable_edges:,}"
    )

    # ======================================================
    # Geographic Data
    # ======================================================

    geo_nodes = 0
    rgb_nodes = 0
    carbon_nodes = 0

    for node in nodes:

        data = G.nodes[
            node
        ]

        if (
            "latitude" in data
            and
            "longitude" in data
        ):

            geo_nodes += 1

        if "rgb_color" in data:

            rgb_nodes += 1

        if (
            "rgb_color" in data
            or
            "carbon_intensity" in data
        ):

            carbon_nodes += 1

    print()
    print(
        "Node attributes:"
    )

    print(
        f"  Nodes with coordinates  : "
        f"{geo_nodes:,}"
    )

    print(
        f"  Nodes with rgb_color    : "
        f"{rgb_nodes:,}"
    )

    print(
        f"  Nodes with carbon data  : "
        f"{carbon_nodes:,}"
    )

    # ======================================================
    # Directional Liquidity
    # ======================================================

    known_liquidity = 0
    capacity_only = 0
    unknown_liquidity = 0

    for (
        _,
        _,
        _,
        data,
    ) in G.edges(
        keys=True,
        data=True,
    ):

        liquidity = get_directional_liquidity(
            data
        )

        if liquidity is not None:

            known_liquidity += 1

        elif "capacity" in data:

            capacity_only += 1

        else:

            unknown_liquidity += 1

    print()
    print(
        "Directional liquidity:"
    )

    print(
        "  Known directional "
        "liquidity              : "
        f"{known_liquidity:,}"
    )

    print(
        "  Capacity only          : "
        f"{capacity_only:,}"
    )

    print(
        "  Completely unknown     : "
        f"{unknown_liquidity:,}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Channel capacity is NOT interpreted "
        "as directional liquidity."
    )

    # ======================================================
    # Reliability
    # ======================================================

    known_reliability = 0
    unknown_reliability = 0

    for (
        _,
        _,
        _,
        data,
    ) in G.edges(
        keys=True,
        data=True,
    ):

        reliability = get_channel_reliability(
            data
        )

        if reliability is not None:

            known_reliability += 1

        else:

            unknown_reliability += 1

    print()
    print(
        "Channel reliability:"
    )

    print(
        f"  Known                    : "
        f"{known_reliability:,}"
    )

    print(
        f"  Unknown                  : "
        f"{unknown_reliability:,}"
    )


# ==========================================================
# Select Source / Destination
# ==========================================================

def select_source_destination(
    G,
):
    """
    Select source and destination from the largest strongly
    connected component.
    """

    print_header(
        "3. SELECT SOURCE / DESTINATION"
    )

    random.seed(
        RANDOM_SEED
    )

    components = list(
        nx.strongly_connected_components(
            G
        )
    )

    if not components:

        raise RuntimeError(
            "No strongly connected component found."
        )

    largest_component = max(
        components,
        key=len,
    )

    component_nodes = list(
        largest_component
    )

    if len(component_nodes) < 2:

        raise RuntimeError(
            "Largest strongly connected component "
            "contains fewer than two nodes."
        )

    source, target = random.sample(
        component_nodes,
        2,
    )

    print(
        f"Random seed              : "
        f"{RANDOM_SEED}"
    )

    print(
        f"Selected component size  : "
        f"{len(component_nodes):,}"
    )

    print()

    print(
        f"Source                   : "
        f"{source}"
    )

    print(
        f"Destination              : "
        f"{target}"
    )

    print(
        f"Payment amount           : "
        f"{AMOUNT:,.0f}"
    )

    print(
        f"K                        : "
        f"{K}"
    )

    print(
        f"Maximum hops             : "
        f"{MAX_HOPS}"
    )

    print(
        f"ETA                      : "
        f"{ETA}"
    )

    print(
        f"Lambda_h                 : "
        f"{LAMBDA_H}"
    )

    return (
        source,
        target,
    )


# ==========================================================
# Validate Route
# ==========================================================

def validate_route(
    G,
    route,
    source,
    target,
    max_hops,
):
    """
    Validate structural consistency of one candidate route.

    Current Top-K edge representation:

        {
            "source": ...,
            "target": ...,
            "channel_key": ...,
            "scid": ...
        }

    This test does NOT execute the payment.
    """

    path = route.get(
        "path"
    )

    edges = route.get(
        "edges"
    )

    # ======================================================
    # Path
    # ======================================================

    if not path:

        return False, "empty_path"

    if path[0] != source:

        return False, "invalid_source"

    if path[-1] != target:

        return False, "invalid_target"

    # ======================================================
    # Simple Path Check
    # ======================================================

    if len(
        path
    ) != len(
        set(path)
    ):

        return False, "loop_detected"

    # ======================================================
    # Hop Count
    # ======================================================

    hop_count = route.get(
        "hop_count"
    )

    if hop_count is None:

        return False, "missing_hop_count"

    try:

        hop_count = int(
            hop_count
        )

    except (
        TypeError,
        ValueError,
    ):

        return False, "invalid_hop_count"

    if (
        len(path) - 1
        !=
        hop_count
    ):

        return False, "hop_count_mismatch"

    if hop_count > max_hops:

        return False, "max_hops_exceeded"

    # ======================================================
    # Edge Count
    # ======================================================

    if edges is None:

        return False, "missing_edges"

    if len(edges) != len(path) - 1:

        return False, "edge_count_mismatch"

    # ======================================================
    # Edge Validation
    # ======================================================

    for index, edge in enumerate(
        edges
    ):

        if not isinstance(
            edge,
            dict,
        ):

            return (
                False,
                f"invalid_edge_format:{edge}",
            )

        u, v, key = get_edge_identity(
            edge
        )

        if (
            u is None
            or
            v is None
            or
            key is None
        ):

            return (
                False,
                f"missing_edge_identity:{edge}",
            )

        expected_u = path[
            index
        ]

        expected_v = path[
            index + 1
        ]

        if u != expected_u:

            return (
                False,
                f"edge_source_mismatch:{u}",
            )

        if v != expected_v:

            return (
                False,
                f"edge_target_mismatch:{v}",
            )

        if not G.has_edge(
            u,
            v,
        ):

            return (
                False,
                f"missing_edge:{u}->{v}",
            )

        if key not in G[u][v]:

            return (
                False,
                f"missing_channel:"
                f"{u}->{v}:{key}",
            )

        data = G[u][v][key]

        # --------------------------------------------------
        # SCID consistency
        # --------------------------------------------------

        edge_scid = edge.get(
            "scid"
        )

        graph_scid = data.get(
            "scid"
        )

        if (
            edge_scid is not None
            and
            graph_scid is not None
            and
            str(edge_scid)
            !=
            str(graph_scid)
        ):

            return (
                False,
                f"scid_mismatch:"
                f"{u}->{v}:{key}:"
                f"edge={edge_scid},"
                f"graph={graph_scid}",
            )

        # --------------------------------------------------
        # Channel availability
        # --------------------------------------------------

        if "available" in data:

            try:

                available = strict_bool(
                    data["available"],
                    "edge.available",
                )

            except ValueError as exc:

                return (
                    False,
                    f"invalid_channel_availability:"
                    f"{u}->{v}:{key}:"
                    f"{exc}",
                )

        else:

            available = True

        if not available:

            return (
                False,
                f"channel_unavailable:"
                f"{u}->{v}:{key}",
            )

        # --------------------------------------------------
        # Node availability
        # --------------------------------------------------

        if "online" in G.nodes[u]:

            try:

                source_online = strict_bool(
                    G.nodes[u]["online"],
                    "node.online",
                )

            except ValueError as exc:

                return (
                    False,
                    f"invalid_source_node_state:"
                    f"{u}:{exc}",
                )

        else:

            source_online = True

        if not source_online:

            return (
                False,
                f"source_node_offline:{u}",
            )

        if "online" in G.nodes[v]:

            try:

                target_online = strict_bool(
                    G.nodes[v]["online"],
                    "node.online",
                )

            except ValueError as exc:

                return (
                    False,
                    f"invalid_target_node_state:"
                    f"{v}:{exc}",
                )

        else:

            target_online = True

        if not target_online:

            return (
                False,
                f"target_node_offline:{v}",
            )

    return True, "OK"


# ==========================================================
# Validate Candidate Metadata
# ==========================================================

def validate_candidate_metadata(
    route,
):
    """
    Validate adaptive-routing metadata returned by
    top_k_paths.

    Current invariant:

        0 <= eta <= 1

    lambda_h must satisfy:

        lambda_h >= 0
    """

    if "eta" not in route:

        return False, "missing_eta"

    if "lambda_h" not in route:

        return False, "missing_lambda_h"

    eta = safe_float(
        route.get(
            "eta"
        ),
        default=float("nan"),
    )

    lambda_h = safe_float(
        route.get(
            "lambda_h"
        ),
        default=float("nan"),
    )

    if not math.isfinite(
        eta
    ):

        return False, "invalid_eta"

    try:

        validate_eta(
            eta
        )

    except (
        TypeError,
        ValueError,
    ):

        return False, "eta_out_of_range"

    if not math.isfinite(
        lambda_h
    ):

        return False, "invalid_lambda_h"

    try:

        validate_lambda_h(
            lambda_h
        )

    except (
        TypeError,
        ValueError,
    ):

        return False, "negative_or_invalid_lambda_h"

    return True, "OK"


# ==========================================================
# Print Route
# ==========================================================

def print_route(
    G,
    route,
    rank,
    source,
    target,
):
    """
    Print complete information for one candidate route.
    """

    print()
    print_separator(
        "-",
        78,
    )

    print(
        f"ROUTE #{rank}"
    )

    print_separator(
        "-",
        78,
    )

    path = route.get(
        "path",
        [],
    )

    edges = route.get(
        "edges",
        [],
    )

    # ======================================================
    # Basic Information
    # ======================================================

    print()
    print(
        f"Rank                     : "
        f"{rank}"
    )

    print(
        f"Source                   : "
        f"{source}"
    )

    print(
        f"Destination              : "
        f"{target}"
    )

    print(
        f"Hop count                : "
        f"{route.get('hop_count')}"
    )

    print(
        f"Cost                     : "
        f"{safe_float(route.get('cost')):.6f}"
    )

    print(
        f"ETA                      : "
        f"{route.get('eta')}"
    )

    print(
        f"Lambda_h                 : "
        f"{route.get('lambda_h')}"
    )

    # ======================================================
    # Path
    # ======================================================

    print()
    print(
        "Path:"
    )

    print(
        "  "
        +
        " -> ".join(
            str(node)
            for node in path
        )
    )

    # ======================================================
    # Channels
    # ======================================================

    print()
    print(
        "Channels:"
    )

    for index, edge in enumerate(
        edges
    ):

        if not isinstance(
            edge,
            dict,
        ):

            print(
                f"  {index + 1:02d}. "
                f"INVALID EDGE RECORD: "
                f"{edge}"
            )

            continue

        u, v, key = get_edge_identity(
            edge
        )

        if (
            u is None
            or
            v is None
            or
            key is None
        ):

            print(
                f"  {index + 1:02d}. "
                f"INVALID EDGE RECORD: "
                f"{edge}"
            )

            continue

        if not G.has_edge(
            u,
            v,
        ):

            print(
                f"  {index + 1:02d}. "
                f"{u} -> {v} "
                f"(missing graph edge)"
            )

            continue

        if key not in G[u][v]:

            print(
                f"  {index + 1:02d}. "
                f"{u} -> {v} "
                f"(missing channel={key})"
            )

            continue

        data = G[u][v][key]

        fee = channel_fee(
            data,
            AMOUNT,
        )

        reliability = get_channel_reliability(
            data
        )

        liquidity = get_directional_liquidity(
            data
        )

        reliability_text = (
            "unknown"
            if reliability is None
            else f"{reliability:.6f}"
        )

        liquidity_text = (
            "unknown"
            if liquidity is None
            else f"{liquidity:.4f}"
        )

        scid = edge.get(
            "scid",
            data.get("scid"),
        )

        print(
            f"  {index + 1:02d}. "
            f"{u} -> {v} "
            f"(channel={key}, scid={scid})"
        )

        print(
            f"       fee={fee:.4f}, "
            f"reliability={reliability_text}, "
            f"liquidity={liquidity_text}"
        )

    # ======================================================
    # Route Metrics
    # ======================================================

    print()
    print(
        "Route metrics:"
    )

    metrics = [

        (
            "Total cost",
            "cost",
        ),

        (
            "Total fee",
            "total_fee",
        ),

        (
            "Total delay",
            "total_delay",
        ),

        (
            "Reliability",
            "reliability",
        ),

        (
            "Failure probability",
            "failure_probability",
        ),

        (
            "Distance",
            "distance_km",
        ),

        (
            "Carbon",
            "carbon_intensity",
        ),

        (
            "Raw heuristic",
            "raw_heuristic",
        ),

        (
            "Adaptive penalty",
            "adaptive_penalty",
        ),

        (
            "Total adaptive penalty",
            "total_adaptive_penalty",
        ),
    ]

    for label, key in metrics:

        value = route.get(
            key
        )

        if value is None:

            value_text = "Unknown"

        elif isinstance(
            value,
            (
                float,
                int,
            ),
        ):

            value_text = (
                f"{float(value):.6f}"
            )

        else:

            value_text = str(
                value
            )

        print(
            f"  {label:<28}: "
            f"{value_text}"
        )

    # ======================================================
    # Candidate Status
    # ======================================================

    print()
    print(
        "Candidate status:"
    )

    print(
        f"  Candidate                : "
        f"{route.get('candidate')}"
    )

    print(
        f"  Payment success          : "
        f"{route.get('success')}"
    )

    # ======================================================
    # Validation
    # ======================================================

    valid, reason = validate_route(
        G=G,
        route=route,
        source=source,
        target=target,
        max_hops=MAX_HOPS,
    )

    metadata_valid, metadata_reason = (
        validate_candidate_metadata(
            route
        )
    )

    print()
    print(
        "Route validation:"
    )

    print(
        f"  Structural status        : "
        f"{'PASS' if valid else 'FAIL'}"
    )

    if not valid:

        print(
            f"  Structural reason        : "
            f"{reason}"
        )

    print(
        f"  Adaptive metadata        : "
        f"{'PASS' if metadata_valid else 'FAIL'}"
    )

    if not metadata_valid:

        print(
            f"  Metadata reason          : "
            f"{metadata_reason}"
        )


# ==========================================================
# Print Summary
# ==========================================================

def print_summary(
    routes,
):
    """
    Print compact candidate-route summary.
    """

    print_header(
        "5. TOP-K ROUTES SUMMARY"
    )

    if not routes:

        print(
            "No candidate route was found."
        )

        return

    header = (
        f"{'Rank':<6}"
        f"{'Hops':<8}"
        f"{'Cost':<16}"
        f"{'Fee':<16}"
        f"{'Delay':<12}"
        f"{'Reliability':<15}"
        f"{'ETA':<8}"
    )

    print(
        header
    )

    print_separator(
        "-",
        len(header),
    )

    for rank, route in enumerate(
        routes,
        start=1,
    ):

        print(
            f"{rank:<6}"
            f"{route.get('hop_count', 0):<8}"
            f"{safe_float(route.get('cost')):<16.4f}"
            f"{safe_float(route.get('total_fee')):<16.4f}"
            f"{safe_float(route.get('total_delay')):<12.2f}"
            f"{safe_float(route.get('reliability')):<15.6f}"
            f"{safe_float(route.get('eta')):<8.3f}"
        )


# ==========================================================
# Cost Ordering Validation
# ==========================================================

def validate_cost_order(
    routes,
):
    """
    Verify:

        C(p1) <= C(p2) <= ... <= C(pk)
    """

    if len(routes) <= 1:

        return True

    costs = [

        safe_float(
            route.get(
                "cost",
                math.inf,
            ),
            default=math.inf,
        )

        for route in routes
    ]

    return all(
        costs[index]
        <=
        costs[index + 1]

        for index in range(
            len(costs) - 1
        )
    )


# ==========================================================
# Validate K
# ==========================================================

def validate_k(
    routes,
    requested_k,
):
    """
    Verify that the number of returned routes does not exceed
    K and that every returned route is unique by physical
    channel identity.

    Two candidates with the same node sequence but different
    physical channels are considered distinct.
    """

    if len(routes) > requested_k:

        return False, "more_routes_than_requested"

    channel_paths = []

    for route in routes:

        edges = route.get(
            "edges",
            [],
        )

        if not edges:

            return False, "empty_candidate_edges"

        identity = []

        for edge in edges:

            if not isinstance(
                edge,
                dict,
            ):

                return (
                    False,
                    f"invalid_edge_format:{edge}",
                )

            u, v, key = get_edge_identity(
                edge
            )

            if (
                u is None
                or
                v is None
                or
                key is None
            ):

                return (
                    False,
                    f"missing_edge_identity:{edge}",
                )

            identity.append(
                (
                    u,
                    v,
                    str(key),
                    str(
                        edge.get(
                            "scid"
                        )
                    ),
                )
            )

        identity = tuple(
            identity
        )

        if identity in channel_paths:

            return (
                False,
                "duplicate_physical_channel_candidate",
            )

        channel_paths.append(
            identity
        )

    return True, "OK"


# ==========================================================
# Validate Adaptive Cost
# ==========================================================

def validate_adaptive_cost(
    G,
    route,
):
    """
    Independently recompute the total adaptive cost of a
    candidate route using the SAME shared adaptive_edge_cost()
    helper used by the current Pathfinding implementation.

    Current edge-level model:

        native_cost *
        (1 + lambda_h * normalized_adaptive_penalty)

    where:

        normalized_adaptive_penalty =
            abs(raw_heuristic) /
            (1 + abs(raw_heuristic))

    This validation intentionally does NOT use the obsolete
    additive formulation.
    """

    path = route.get(
        "path"
    )

    edges = route.get(
        "edges"
    )

    if not path or not edges:

        return False, "empty_route"

    eta = safe_float(
        route.get(
            "eta"
        ),
        default=float("nan"),
    )

    lambda_h = safe_float(
        route.get(
            "lambda_h",
            1.0,
        ),
        default=float("nan"),
    )

    if not math.isfinite(
        eta
    ):

        return False, "invalid_eta"

    if not math.isfinite(
        lambda_h
    ):

        return False, "invalid_lambda_h"

    try:

        validate_eta(
            eta
        )

        validate_lambda_h(
            lambda_h
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        return (
            False,
            f"invalid_parameters:{exc}",
        )

    recomputed_cost = 0.0

    for edge in edges:

        if not isinstance(
            edge,
            dict,
        ):

            return (
                False,
                f"invalid_edge_format:{edge}",
            )

        u, v, key = get_edge_identity(
            edge
        )

        if (
            u is None
            or
            v is None
            or
            key is None
        ):

            return (
                False,
                f"missing_edge_identity:{edge}",
            )

        if not G.has_edge(
            u,
            v,
        ):

            return (
                False,
                f"missing_edge:{u}->{v}",
            )

        if key not in G[u][v]:

            return (
                False,
                f"missing_channel:{u}->{v}:{key}",
            )

        data = G[u][v][key]

        result = adaptive_edge_cost(
            G=G,
            u=u,
            v=v,
            data=data,
            amount=AMOUNT,
            eta=eta,
            heuristic_fn=lnd_cost,
            lambda_h=lambda_h,
        )

        if not isinstance(
            result,
            dict,
        ):

            return (
                False,
                "adaptive_edge_cost_invalid_return_type",
            )

        native_cost = safe_float(
            result.get(
                "native_cost"
            ),
            default=float("nan"),
        )

        raw_h = safe_float(
            result.get(
                "raw_heuristic"
            ),
            default=float("nan"),
        )

        penalty = safe_float(
            result.get(
                "adaptive_penalty"
            ),
            default=float("nan"),
        )

        edge_cost = safe_float(
            result.get(
                "cost"
            ),
            default=float("nan"),
        )

        if not all(
            math.isfinite(value)
            for value in (
                native_cost,
                raw_h,
                penalty,
                edge_cost,
            )
        ):

            return (
                False,
                f"non_finite_edge_cost:"
                f"{u}->{v}:{key}",
            )

        if not (
            0.0
            <=
            penalty
            <
            1.0
        ):

            return (
                False,
                f"edge_penalty_out_of_range:"
                f"{u}->{v}:{key}:"
                f"{penalty}",
            )

        recomputed_cost += edge_cost

    reported_cost = safe_float(
        route.get(
            "cost"
        ),
        default=float("nan"),
    )

    if not math.isfinite(
        reported_cost
    ):

        return False, "invalid_reported_cost"

    if not math.isclose(
        recomputed_cost,
        reported_cost,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):

        return (
            False,
            "adaptive_cost_mismatch:"
            f"reported={reported_cost},"
            f"recomputed={recomputed_cost}",
        )

    return True, "OK"


# ==========================================================
# Validate Edge-Level Adaptive Formula
# ==========================================================

def validate_edge_level_adaptive_formula(
    G,
    route,
):
    """
    Verify the mathematical invariant of the shared adaptive
    edge-cost helper.

        penalty =
            abs(raw_h) / (1 + abs(raw_h))

        cost =
            native_cost *
            (1 + lambda_h * penalty)
    """

    edges = route.get(
        "edges"
    )

    if not edges:

        return False, "empty_edges"

    eta = safe_float(
        route.get(
            "eta"
        ),
        default=float("nan"),
    )

    lambda_h = safe_float(
        route.get(
            "lambda_h",
            1.0,
        ),
        default=float("nan"),
    )

    if not math.isfinite(
        eta
    ):

        return False, "invalid_eta"

    if not math.isfinite(
        lambda_h
    ):

        return False, "invalid_lambda_h"

    try:

        validate_eta(
            eta
        )

        validate_lambda_h(
            lambda_h
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        return (
            False,
            f"invalid_parameters:{exc}",
        )

    for edge in edges:

        if not isinstance(
            edge,
            dict,
        ):

            return (
                False,
                f"invalid_edge_format:{edge}",
            )

        u, v, key = get_edge_identity(
            edge
        )

        if (
            u is None
            or
            v is None
            or
            key is None
        ):

            return (
                False,
                f"missing_edge_identity:{edge}",
            )

        if not G.has_edge(
            u,
            v,
        ):

            return (
                False,
                f"missing_edge:{u}->{v}",
            )

        if key not in G[u][v]:

            return (
                False,
                f"missing_channel:{u}->{v}:{key}",
            )

        data = G[u][v][key]

        result = adaptive_edge_cost(
            G=G,
            u=u,
            v=v,
            data=data,
            amount=AMOUNT,
            eta=eta,
            heuristic_fn=lnd_cost,
            lambda_h=lambda_h,
        )

        if not isinstance(
            result,
            dict,
        ):

            return (
                False,
                "adaptive_edge_cost_invalid_return_type",
            )

        native_cost = safe_float(
            result.get(
                "native_cost"
            ),
            default=float("nan"),
        )

        raw_h = safe_float(
            result.get(
                "raw_heuristic"
            ),
            default=float("nan"),
        )

        reported_penalty = safe_float(
            result.get(
                "adaptive_penalty"
            ),
            default=float("nan"),
        )

        reported_cost = safe_float(
            result.get(
                "cost"
            ),
            default=float("nan"),
        )

        if not all(
            math.isfinite(value)
            for value in (
                native_cost,
                raw_h,
                reported_penalty,
                reported_cost,
            )
        ):

            return (
                False,
                f"non_finite_formula_values:"
                f"edge={u}->{v}:{key}",
            )

        expected_penalty = (
            abs(raw_h)
            /
            (
                1.0
                +
                abs(raw_h)
            )
        )

        expected_cost = (
            native_cost
            *
            (
                1.0
                +
                lambda_h
                *
                expected_penalty
            )
        )

        if not math.isclose(
            reported_penalty,
            expected_penalty,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):

            return (
                False,
                "penalty_formula_mismatch:"
                f"edge={u}->{v}:{key}",
            )

        if not math.isclose(
            reported_cost,
            expected_cost,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):

            return (
                False,
                "cost_formula_mismatch:"
                f"edge={u}->{v}:{key}",
            )

    return True, "OK"


# ==========================================================
# Main Test
# ==========================================================

def main():

    print()
    print_separator()

    print(
        " REAL GML TOP-K PATHFINDING TEST "
    )

    print_separator()

    # ======================================================
    # Validate configuration
    # ======================================================

    try:

        validate_eta(
            ETA
        )

        validate_lambda_h(
            LAMBDA_H
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise RuntimeError(
            f"Invalid test configuration: {exc}"
        ) from exc

    # ======================================================
    # 1. Load graph
    # ======================================================

    G = load_graph()

    # ======================================================
    # 2. Graph statistics
    # ======================================================

    print_graph_statistics(
        G
    )

    # ======================================================
    # 3. Source / Destination
    # ======================================================

    (
        source,
        target,
    ) = select_source_destination(
        G
    )

    # ======================================================
    # 4. Routing Data Readiness
    # ======================================================

    readiness = print_routing_data_readiness(
        G
    )

    if not readiness["ready"]:

        print()
        print_separator(
            "!",
            78,
        )

        print(
            "PATHFINDING STATUS : DATA NOT READY"
        )

        print_separator(
            "!",
            78,
        )

        print()
        print(
            "The raw GML snapshot does not contain the "
            "routing data required by top_k_paths.py."
        )

        print()
        print(
            "No directional-liquidity fallback was applied."
        )

        print(
            "Channel capacity was NOT interpreted as "
            "directional liquidity."
        )

        print(
            "No synthetic reliability value was introduced."
        )

        print()
        print(
            "The Top-K algorithm was therefore not executed."
        )

        print()
        print(
            "This is a data-readiness condition, not an "
            "algorithmic Top-K exception."
        )

        print()
        print_separator()

        print(
            " END OF TOP-K PATHFINDING TEST "
        )

        print_separator()

        return

    # ======================================================
    # 5. Run Top-K Pathfinding
    # ======================================================

    print_header(
        "5. RUN TOP-K PATHFINDING"
    )

    print(
        "Routing configuration:"
    )

    print(
        f"  Source       : "
        f"{source}"
    )

    print(
        f"  Target       : "
        f"{target}"
    )

    print(
        f"  Amount       : "
        f"{AMOUNT:,.0f}"
    )

    print(
        f"  K            : "
        f"{K}"
    )

    print(
        f"  Max hops     : "
        f"{MAX_HOPS}"
    )

    print(
        f"  ETA          : "
        f"{ETA}"
    )

    print(
        f"  Lambda_h     : "
        f"{LAMBDA_H}"
    )

    print()
    print(
        "Running adaptive top-k pathfinding..."
    )

    try:

        routes = top_k_paths(
            G=G,
            source=source,
            target=target,
            amount=AMOUNT,
            heuristic_fn=lnd_cost,
            eta=ETA,
            k=K,
            max_hops=MAX_HOPS,
            lambda_h=LAMBDA_H,
        )

    except Exception as exc:

        print()
        print_separator(
            "!",
            78,
        )

        print(
            "PATHFINDING ERROR"
        )

        print_separator(
            "!",
            78,
        )

        print()
        print(
            f"Exception type : "
            f"{type(exc).__name__}"
        )

        print(
            f"Message        : "
            f"{exc}"
        )

        print()
        print(
            "The test intentionally does not hide this exception."
        )

        print(
            "The current top_k_paths() API and this test "
            "must remain consistent."
        )

        raise

    # ======================================================
    # 6. Number of routes
    # ======================================================

    print()
    print(
        f"Number of routes returned : "
        f"{len(routes)}"
    )

    # ======================================================
    # 7. Route validation
    # ======================================================

    print_header(
        "7. VALIDATE CANDIDATE ROUTES"
    )

    structural_pass = 0
    metadata_pass = 0

    for rank, route in enumerate(
        routes,
        start=1,
    ):

        valid, reason = validate_route(
            G=G,
            route=route,
            source=source,
            target=target,
            max_hops=MAX_HOPS,
        )

        metadata_valid, metadata_reason = (
            validate_candidate_metadata(
                route
            )
        )

        print(
            f"Route #{rank:<3} "
            f"structural="
            f"{'PASS' if valid else 'FAIL':<5} "
            f"adaptive="
            f"{'PASS' if metadata_valid else 'FAIL'}"
        )

        if not valid:

            print(
                f"    structural reason: "
                f"{reason}"
            )

        if not metadata_valid:

            print(
                f"    metadata reason: "
                f"{metadata_reason}"
            )

        if valid:

            structural_pass += 1

        if metadata_valid:

            metadata_pass += 1

    print()
    print(
        f"Structural validation   : "
        f"{structural_pass}/{len(routes)}"
    )

    print(
        f"Adaptive metadata       : "
        f"{metadata_pass}/{len(routes)}"
    )

    # ======================================================
    # 8. Print routes
    # ======================================================

    print_header(
        "8. CANDIDATE ROUTES"
    )

    for rank, route in enumerate(
        routes,
        start=1,
    ):

        print_route(
            G=G,
            route=route,
            rank=rank,
            source=source,
            target=target,
        )

    # ======================================================
    # 9. Summary
    # ======================================================

    print_summary(
        routes
    )

    # ======================================================
    # 10. Cost Ordering
    # ======================================================

    print_header(
        "9. ROUTE ORDER VALIDATION"
    )

    ordered = validate_cost_order(
        routes
    )

    print(
        f"Cost ordering             : "
        f"{'PASS' if ordered else 'FAIL'}"
    )

    if routes:

        print()
        print(
            "Candidate costs:"
        )

        for index, route in enumerate(
            routes,
            start=1,
        ):

            print(
                f"  Route #{index}: "
                f"{safe_float(route.get('cost')):.6f}"
            )

    # ======================================================
    # 11. K Validation
    # ======================================================

    print_header(
        "10. TOP-K VALIDATION"
    )

    k_valid, k_reason = validate_k(
        routes,
        K,
    )

    print(
        f"K validation              : "
        f"{'PASS' if k_valid else 'FAIL'}"
    )

    if not k_valid:

        print(
            f"Reason                    : "
            f"{k_reason}"
        )

    # ======================================================
    # 12. Adaptive Cost Validation
    # ======================================================

    print_header(
        "11. ADAPTIVE COST VALIDATION"
    )

    adaptive_pass = 0
    formula_pass = 0

    for rank, route in enumerate(
        routes,
        start=1,
    ):

        valid, reason = validate_adaptive_cost(
            G,
            route,
        )

        formula_valid, formula_reason = (
            validate_edge_level_adaptive_formula(
                G,
                route,
            )
        )

        print(
            f"Route #{rank:<3}: "
            f"shared_cost="
            f"{'PASS' if valid else 'FAIL':<5} "
            f"formula="
            f"{'PASS' if formula_valid else 'FAIL'}"
        )

        if not valid:

            print(
                f"    Shared-cost reason: "
                f"{reason}"
            )

        else:

            adaptive_pass += 1

        if not formula_valid:

            print(
                f"    Formula reason: "
                f"{formula_reason}"
            )

        else:

            formula_pass += 1

    print()
    print(
        f"Shared adaptive cost     : "
        f"{adaptive_pass}/{len(routes)}"
    )

    print(
        f"Adaptive formula         : "
        f"{formula_pass}/{len(routes)}"
    )

    # ======================================================
    # 13. Final Result
    # ======================================================

    print_header(
        "12. TEST RESULT"
    )

    if not routes:

        print(
            "PATHFINDING STATUS : NO ROUTE"
        )

        print()
        print(
            "No feasible candidate route was found "
            "under the current topology, payment amount, "
            "ETA and hop constraints."
        )

        print()
        print(
            "This is not necessarily a code failure."
        )

        print(
            "It may indicate that the selected source/"
            "destination pair has no feasible route."
        )

        print()
        print_separator()

        print(
            " END OF TOP-K PATHFINDING TEST "
        )

        print_separator()

        return

    all_passed = (
        structural_pass
        ==
        len(routes)

        and

        metadata_pass
        ==
        len(routes)

        and

        ordered

        and

        k_valid

        and

        adaptive_pass
        ==
        len(routes)

        and

        formula_pass
        ==
        len(routes)
    )

    print(
        f"PATHFINDING STATUS : "
        f"{'SUCCESS' if all_passed else 'PARTIAL / CHECK REQUIRED'}"
    )

    print()
    print(
        f"Requested K        : "
        f"{K}"
    )

    print(
        f"Returned routes    : "
        f"{len(routes)}"
    )

    print(
        f"Maximum hops       : "
        f"{MAX_HOPS}"
    )

    print(
        f"ETA                : "
        f"{ETA}"
    )

    print(
        f"Lambda_h           : "
        f"{LAMBDA_H}"
    )

    print()
    print(
        "Verified components:"
    )

    print(
        f"  Route structure       : "
        f"{'PASS' if structural_pass == len(routes) else 'FAIL'}"
    )

    print(
        f"  Adaptive metadata     : "
        f"{'PASS' if metadata_pass == len(routes) else 'FAIL'}"
    )

    print(
        f"  Cost ordering         : "
        f"{'PASS' if ordered else 'FAIL'}"
    )

    print(
        f"  K constraint          : "
        f"{'PASS' if k_valid else 'FAIL'}"
    )

    print(
        f"  Shared adaptive cost  : "
        f"{'PASS' if adaptive_pass == len(routes) else 'FAIL'}"
    )

    print(
        f"  Adaptive formula      : "
        f"{'PASS' if formula_pass == len(routes) else 'FAIL'}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  This test verifies Pathfinding only."
    )

    print(
        "  It does NOT execute payments."
    )

    print(
        "  It does NOT train PPO."
    )

    print(
        "  It does NOT use Bucket for route storage."
    )

    print(
        "  It does NOT perform Partial Backtracking."
    )

    print()
    print(
        "Current verified pathfinding chain:"
    )

    print(
        "  ETA"
    )

    print(
        "    -> Adaptive Heuristic"
    )

    print(
        "    -> Normalized Adaptive Penalty"
    )

    print(
        "    -> Adaptive Edge Cost"
    )

    print(
        "    -> Top-K Candidate Routes"
    )

    print()
    print(
        "Next integration stage:"
    )

    print(
        "  Top-K Candidate Routes"
    )

    print(
        "        -> Bucket"
    )

    print(
        "        -> Selected Route"
    )

    print(
        "        -> Payment Simulation"
    )

    print(
        "        -> Failure Detection"
    )

    print(
        "        -> Partial Backtracking"
    )

    print(
        "        -> Alternative Suffix"
    )

    print_separator()

    print(
        " END OF TOP-K PATHFINDING TEST "
    )

    print_separator()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()
