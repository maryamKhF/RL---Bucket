# Pathfinding/test_pathfinding.py

import math
import random
import sys
from pathlib import Path
import math
import networkx as nx


# ==========================================================
# Project root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# Repository imports
# ==========================================================

from Pathfinding.top_k_paths import top_k_paths
from Pathfinding.heuristics import lnd_cost


# ==========================================================
# Configuration
# ==========================================================

GML_FILE = PROJECT_ROOT / "20190501.gml.geo"

K = 5
MAX_HOPS = 12

# PPO routing parameter.
# In this standalone test, eta is fixed.
ETA = 0.5

# Payment amount.
AMOUNT = 10_000

RANDOM_SEED = 42


# ==========================================================
# Utility functions
# ==========================================================

def print_separator(
    char="=",
    length=78
):
    print(
        char * length
    )


def print_header(
    title
):
    print()
    print_separator()
    print(title)
    print_separator()


# ==========================================================
# Load GML
# ==========================================================

def load_graph():
    """
    Load the real Lightning snapshot.

    The GML file may be loaded as Graph, DiGraph,
    MultiGraph or MultiDiGraph.

    The routing implementation expects a MultiDiGraph.

    IMPORTANT
    ---------
    If the source GML is undirected, converting it to
    MultiDiGraph creates reverse edges structurally.

    This does NOT mean that the original snapshot explicitly
    contained independent directional channel policies for
    those reverse edges.
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
        label="id"
    )

    original_type = type(G).__name__

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

    # ------------------------------------------------------
    # Convert to MultiDiGraph
    # ------------------------------------------------------

    if isinstance(
        G,
        nx.MultiDiGraph
    ):

        routing_graph = G

    elif isinstance(
        G,
        nx.DiGraph
    ):

        routing_graph = nx.MultiDiGraph(
            G
        )

    elif isinstance(
        G,
        nx.MultiGraph
    ):

        routing_graph = nx.MultiDiGraph(
            G
        )

    elif isinstance(
        G,
        nx.Graph
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

    if not isinstance(
        G,
        (
            nx.DiGraph,
            nx.MultiDiGraph
        )
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
    G
):

    print_header(
        "2. GRAPH STATISTICS"
    )

    nodes = list(
        G.nodes
    )

    # ------------------------------------------------------
    # Nodes
    # ------------------------------------------------------

    online_nodes = sum(
        1
        for node in nodes
        if bool(
            G.nodes[node].get(
                "online",
                True
            )
        )
    )

    # ------------------------------------------------------
    # Channels
    # ------------------------------------------------------

    available_edges = 0

    for _, _, _, data in G.edges(
        keys=True,
        data=True
    ):

        if bool(
            data.get(
                "available",
                True
            )
        ):

            available_edges += 1

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
        f"Available channels       : "
        f"{available_edges:,}"
    )

    # ======================================================
    # Geographic data
    # ======================================================

    geo_nodes = 0
    carbon_nodes = 0
    rgb_nodes = 0

    for node in nodes:

        data = G.nodes[node]

        if (
            "latitude" in data
            and
            "longitude" in data
        ):

            geo_nodes += 1

        if "rgb_color" in data:

            rgb_nodes += 1

        if (
            "carbon_intensity" in data
            or
            "rgb_color" in data
        ):

            carbon_nodes += 1

    print()
    print(
        "Node attributes:"
    )

    print(
        f"  Nodes with coordinates   : "
        f"{geo_nodes:,}"
    )

    print(
        f"  Nodes with rgb_color     : "
        f"{rgb_nodes:,}"
    )

    print(
        f"  Nodes with carbon data   : "
        f"{carbon_nodes:,}"
    )

    # ======================================================
    # Directional liquidity
    # ======================================================

    known_liquidity = 0

    capacity_only = 0

    for _, _, _, data in G.edges(
        keys=True,
        data=True
    ):

        has_liquidity = any(
            key in data
            and data.get(key) is not None
            for key in (
                "estimated_liquidity",
                "liquidity_uv",
                "balance_uv"
            )
        )

        if has_liquidity:

            known_liquidity += 1

        elif "capacity" in data:

            capacity_only += 1

    print()
    print(
        "Directional liquidity:"
    )

    print(
        "  Channels with "
        "learned/observed liquidity : "
        f"{known_liquidity:,}"
    )

    print(
        "  Channels with capacity but "
        "no directional liquidity  : "
        f"{capacity_only:,}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Snapshot capacity is NOT used "
        "as balance or liquidity."
    )


# ==========================================================
# Select Source / Destination
# ==========================================================

def select_source_destination(
    G
):

    print_header(
        "3. SELECT SOURCE / DESTINATION"
    )

    random.seed(
        RANDOM_SEED
    )

    # ------------------------------------------------------
    # Strongly connected components
    # ------------------------------------------------------

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
        key=len
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
        2
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
        f"{AMOUNT:,.0f} sat"
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

    return (
        source,
        target
    )


# ==========================================================
# Route Validation
# ==========================================================

def validate_route(
    G,
    route,
    source,
    target,
    max_hops
):
    """
    Validate structural consistency of a candidate route.

    This does NOT execute the payment.

    It only verifies that:

        - source is correct
        - destination is correct
        - hop count is correct
        - edge sequence is valid
        - selected channel keys exist
        - hop count <= max_hops
    """

    path = route.get(
        "path"
    )

    edges = route.get(
        "edges"
    )

    # ------------------------------------------------------
    # Path
    # ------------------------------------------------------

    if not path:

        return False, "empty_path"

    if path[0] != source:

        return False, "invalid_source"

    if path[-1] != target:

        return False, "invalid_target"

    # ------------------------------------------------------
    # Hop count
    # ------------------------------------------------------

    hop_count = route.get(
        "hop_count"
    )

    if hop_count is None:

        return False, "missing_hop_count"

    if (
        len(path) - 1
        !=
        int(hop_count)
    ):

        return False, "hop_count_mismatch"

    if int(hop_count) > max_hops:

        return False, "max_hops_exceeded"

    # ------------------------------------------------------
    # Edge count
    # ------------------------------------------------------

    if edges is None:

        return False, "missing_edges"

    if len(edges) != len(path) - 1:

        return False, "edge_count_mismatch"

    # ------------------------------------------------------
    # Validate every edge
    # ------------------------------------------------------

    for index, (
        u,
        v,
        key
    ) in enumerate(edges):

        expected_u = path[index]
        expected_v = path[index + 1]

        if u != expected_u:

            return (
                False,
                f"edge_source_mismatch:{u}"
            )

        if v != expected_v:

            return (
                False,
                f"edge_target_mismatch:{v}"
            )

        if not G.has_edge(
            u,
            v
        ):

            return (
                False,
                f"missing_edge:{u}->{v}"
            )

        if key not in G[u][v]:

            return (
                False,
                f"missing_channel:"
                f"{u}->{v}:{key}"
            )

    return True, "OK"


# ==========================================================
# Print Route
# ==========================================================

def print_route(
    G,
    route,
    rank,
    source,
    target
):

    print()
    print_separator(
        "-",
        78
    )

    print(
        f"ROUTE #{rank}"
    )

    print_separator(
        "-",
        78
    )

    path = route.get(
        "path",
        []
    )

    edges = route.get(
        "edges",
        []
    )

    # ------------------------------------------------------
    # Basic information
    # ------------------------------------------------------

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
        f"{route.get('cost', 0.0):.6f}"
    )

    print(
        f"ETA                      : "
        f"{route.get('eta', ETA)}"
    )

    print(
        f"Lambda_h                 : "
        f"{route.get('lambda_h', 1.0)}"
    )

    # ------------------------------------------------------
    # Path
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Nodes
    # ------------------------------------------------------

    print()
    print(
        "Path nodes:"
    )

    for index, node in enumerate(
        path
    ):

        print(
            f"  {index:02d}. {node}"
        )

    # ------------------------------------------------------
    # Channels
    # ------------------------------------------------------

    print()
    print(
        "Channels:"
    )

    for index, edge in enumerate(
        edges
    ):

        if len(edge) == 3:

            u, v, key = edge

            print(
                f"  {index + 1:02d}. "
                f"{u} -> {v} "
                f"(channel={key})"
            )

        else:

            print(
                f"  {index + 1:02d}. "
                f"{edge}"
            )

    # ------------------------------------------------------
    # Route metrics
    # ------------------------------------------------------

    print()
    print(
        "Route metrics:"
    )

    metrics = [

        (
            "Total cost",
            "cost"
        ),

        (
            "Total fee",
            "total_fee"
        ),

        (
            "Total delay",
            "total_delay"
        ),

        (
            "Reliability",
            "reliability"
        ),

        (
            "Failure probability",
            "failure_probability"
        )
    ]

    for label, key in metrics:

        value = route.get(
            key
        )

        if value is None:

            value_text = "Unknown"

        elif isinstance(
            value,
            (float, int)
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

    # ------------------------------------------------------
    # Candidate status
    # ------------------------------------------------------

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

    print()
    print(
        "Note:"
    )

    print(
        "  This test evaluates route generation only."
    )

    print(
        "  It does NOT execute the payment."
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    valid, reason = validate_route(
        G=G,
        route=route,
        source=source,
        target=target,
        max_hops=MAX_HOPS
    )

    print()
    print(
        "Route validation:"
    )

    print(
        f"  Status                   : "
        f"{'PASS' if valid else 'FAIL'}"
    )

    if not valid:

        print(
            f"  Reason                   : "
            f"{reason}"
        )


# ==========================================================
# Summary
# ==========================================================

def print_summary(
    routes
):

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
    )

    print(
        header
    )

    print_separator(
        "-",
        len(header)
    )

    for rank, route in enumerate(
        routes,
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{route.get('hop_count', 0):<8}"
            f"{route.get('cost', 0.0):<16.4f}"
            f"{route.get('total_fee', 0.0):<16.4f}"
            f"{route.get('total_delay', 0.0):<12.2f}"
            f"{route.get('reliability', 0.0):<15.6f}"
        )


# ==========================================================
# Cost Ordering Validation
# ==========================================================

def validate_cost_order(
    routes
):
    """
    Verify:

        C(p1) <= C(p2) <= ... <= C(pk)
    """

    if len(routes) <= 1:

        return True

    costs = [
        float(
            route.get(
                "cost",
                math.inf
            )
        )
        for route in routes
    ]

    return all(
        costs[i] <= costs[i + 1]
        for i in range(
            len(costs) - 1
        )
    )


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
    # 2. Statistics
    # ======================================================

    print_graph_statistics(
        G
    )

    # ======================================================
    # 3. Select source / destination
    # ======================================================

    (
        source,
        target
    ) = select_source_destination(
        G
    )

    # ======================================================
    # 4. Pathfinding
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
        f"{AMOUNT:,.0f} sat"
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

    print()
    print(
        "Running pathfinding..."
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
            max_hops=MAX_HOPS
        )

    except Exception as exc:

        print()
        print_separator(
            "!",
            78
        )

        print(
            "PATHFINDING ERROR"
        )

        print_separator(
            "!",
            78
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
    # Number of routes
    # ======================================================

    print()
    print(
        f"Number of routes returned : "
        f"{len(routes)}"
    )

    # ======================================================
    # Print routes
    # ======================================================

    for rank, route in enumerate(
        routes,
        start=1
    ):

        print_route(
            G=G,
            route=route,
            rank=rank,
            source=source,
            target=target
        )

    # ======================================================
    # Summary
    # ======================================================

    print_summary(
        routes
    )

    # ======================================================
    # Validate ordering
    # ======================================================

    print_header(
        "6. ROUTE ORDER VALIDATION"
    )

    ordered = validate_cost_order(
        routes
    )

    print(
        f"Cost ordering             : "
        f"{'PASS' if ordered else 'FAIL'}"
    )

    if routes:

        costs = [
            route.get(
                "cost",
                math.inf
            )
            for route in routes
        ]

        print()
        print(
            "Candidate costs:"
        )

        for index, cost in enumerate(
            costs,
            start=1
        ):

            print(
                f"  Route #{index}: "
                f"{float(cost):.6f}"
            )

    # ======================================================
    # Final result
    # ======================================================

    print_header(
        "7. TEST RESULT"
    )

    if routes:

        print(
            "PATHFINDING STATUS : SUCCESS"
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

        print()
        print(
            "The generated paths are candidate routes."
        )

        print(
            "They have NOT yet been executed as payments."
        )

        print()
        print(
            "Next stage:"
        )

        print(
            "  Candidate routes -> Bucket "
            "-> Payment Simulator"
        )

    else:

        print(
            "PATHFINDING STATUS : NO ROUTE"
        )

        print()
        print(
            "No feasible candidate route was found "
            "under the current topology, amount, "
            "ETA and hop constraints."
        )

    print()
    print_separator()

    print(
        " END OF TOP-K PATHFINDING TEST "
    )

    print_separator()


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()