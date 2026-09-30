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
    adaptive_heuristic,
    modified_cost,
    channel_fee,
    estimated_liquidity,
    reliability_penalty,
)


# ==========================================================
# Configuration
# ==========================================================

GML_FILE = PROJECT_ROOT / "20190501.gml.geo"

K = 5
MAX_HOPS = 12

# ----------------------------------------------------------
# PPO adaptive parameter
# ----------------------------------------------------------

ETA = 0.5

# ----------------------------------------------------------
# Weight of adaptive heuristic
# ----------------------------------------------------------

LAMBDA_H = 1.0

# ----------------------------------------------------------
# Payment amount
# ----------------------------------------------------------

AMOUNT = 10_000

# ----------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------

RANDOM_SEED = 42


# ==========================================================
# Utility Functions
# ==========================================================

def print_separator(
    char="=",
    length=78,
):
    print(
        char * length
    )


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

        value = float(
            value
        )

        if not math.isfinite(
            value
        ):
            return float(
                default
            )

        return value

    except (
        TypeError,
        ValueError,
    ):
        return float(
            default
        )


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

    Internally, the test converts the graph to MultiDiGraph
    so that channel keys are preserved.
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

        online = bool(
            data.get(
                "online",
                True,
            )
        )

        available = bool(
            data.get(
                "available",
                True,
            )
        )

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

        available = bool(
            data.get(
                "available",
                True,
            )
        )

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

        liquidity = estimated_liquidity(
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
        "  Completely unknown    : "
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

    hop_count = int(
        hop_count
    )

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

        if len(edge) != 3:

            return (
                False,
                f"invalid_edge_format:{edge}",
            )

        u, v, key = edge

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
        # Channel availability
        # --------------------------------------------------

        if not bool(
            data.get(
                "available",
                True,
            )
        ):

            return (
                False,
                f"channel_unavailable:"
                f"{u}->{v}:{key}",
            )

        # --------------------------------------------------
        # Node availability
        # --------------------------------------------------

        if not bool(
            G.nodes[u].get(
                "online",
                True,
            )
        ):

            return (
                False,
                f"source_node_offline:{u}",
            )

        if not bool(
            G.nodes[v].get(
                "online",
                True,
            )
        ):

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
    Validate that adaptive routing metadata returned by
    top_k_paths is internally consistent.
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

    if not (
        -1.0
        <=
        eta
        <=
        1.0
    ):

        return False, "eta_out_of_range"

    if not math.isfinite(
        lambda_h
    ):

        return False, "invalid_lambda_h"

    if lambda_h < 0.0:

        return False, "negative_lambda_h"

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

        if len(edge) == 3:

            u, v, key = edge

            data = G[u][v][key]

            fee = channel_fee(
                data,
                AMOUNT,
            )

            reliability = (
                1.0
                -
                reliability_penalty(
                    data
                )
            )

            liquidity = estimated_liquidity(
                data
            )

            print(
                f"  {index + 1:02d}. "
                f"{u} -> {v} "
                f"(channel={key})"
            )

            print(
                f"       fee={fee:.4f}, "
                f"reliability={reliability:.6f}, "
                f"liquidity="
                f"{'unknown' if liquidity is None else f'{liquidity:.4f}'}"
            )

        else:

            print(
                f"  {index + 1:02d}. "
                f"{edge}"
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
    K and that every returned route is unique by node path.
    """

    if len(routes) > requested_k:

        return False, "more_routes_than_requested"

    paths = []

    for route in routes:

        path = tuple(
            route.get(
                "path",
                [],
            )
        )

        if not path:

            return False, "empty_candidate_path"

        if path in paths:

            return False, "duplicate_candidate_path"

        paths.append(
            path
        )

    return True, "OK"


# ==========================================================
# Validate Adaptive Cost on First Route
# ==========================================================

def validate_adaptive_cost(
    G,
    route,
):
    """
    Independently recompute the adaptive cost of the first
    candidate route.

    This verifies that the returned route cost is consistent
    with:

        LND cost
        +
        lambda_h * adaptive heuristic
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

    recomputed_cost = 0.0

    for u, v, key in edges:

        data = G[u][v][key]

        base_cost = lnd_cost(
            G,
            u,
            v,
            data,
            AMOUNT,
        )

        h = adaptive_heuristic(
            G,
            u,
            v,
            eta,
        )

        edge_cost = modified_cost(
            native_cost=base_cost,
            geo_penalty=h,
            eta=eta,
            lambda_h=lambda_h,
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
    # 4. Run Top-K Pathfinding
    # ======================================================

    print_header(
        "4. RUN TOP-K PATHFINDING"
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

    except TypeError as exc:

        print()
        print(
            "WARNING:"
        )

        print(
            "The current top_k_paths() signature does "
            "not accept lambda_h."
        )

        print(
            "Retrying without explicit lambda_h."
        )

        print(
            f"TypeError: {exc}"
        )

        routes = top_k_paths(
            G=G,
            source=source,
            target=target,
            amount=AMOUNT,
            heuristic_fn=lnd_cost,
            eta=ETA,
            k=K,
            max_hops=MAX_HOPS,
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

        raise

    # ======================================================
    # 5. Number of routes
    # ======================================================

    print()
    print(
        f"Number of routes returned : "
        f"{len(routes)}"
    )

    # ======================================================
    # 6. Route validation
    # ======================================================

    print_header(
        "6. VALIDATE CANDIDATE ROUTES"
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
    # 7. Print routes
    # ======================================================

    print_header(
        "7. CANDIDATE ROUTES"
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
    # 8. Summary
    # ======================================================

    print_summary(
        routes
    )

    # ======================================================
    # 9. Cost Ordering
    # ======================================================

    print_header(
        "8. ROUTE ORDER VALIDATION"
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
    # 10. K Validation
    # ======================================================

    print_header(
        "9. TOP-K VALIDATION"
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
    # 11. Adaptive Cost Validation
    # ======================================================

    print_header(
        "10. ADAPTIVE COST VALIDATION"
    )

    adaptive_pass = 0

    for rank, route in enumerate(
        routes,
        start=1,
    ):

        valid, reason = validate_adaptive_cost(
            G,
            route,
        )

        print(
            f"Route #{rank:<3}: "
            f"{'PASS' if valid else 'FAIL'}"
        )

        if not valid:

            print(
                f"    Reason: {reason}"
            )

        else:

            adaptive_pass += 1

    print()
    print(
        f"Adaptive cost validation : "
        f"{adaptive_pass}/{len(routes)}"
    )

    # ======================================================
    # 12. Final Result
    # ======================================================

    print_header(
        "11. TEST RESULT"
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
        f"  Adaptive cost         : "
        f"{'PASS' if adaptive_pass == len(routes) else 'FAIL'}"
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