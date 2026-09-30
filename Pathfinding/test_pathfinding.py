# Pathfinding/test_pathfinding.py

import random
import sys
from pathlib import Path

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

# eta is the adaptive parameter produced by PPO.
# For this standalone pathfinding test, we use a fixed value.
ETA = 0.5

# Payment amount in satoshi.
AMOUNT = 10_000

RANDOM_SEED = 42


# ==========================================================
# Utility functions
# ==========================================================

def print_separator(char="=", length=78):
    print(char * length)


def print_header(title):
    print()
    print_separator()
    print(title)
    print_separator()


def load_graph():
    """
    Load the real Lightning snapshot.

    The file is expected to be:
        20190501.gml.geo

    The graph is normalized to MultiDiGraph because
    Lightning channels are directional and multiple
    channels may exist between two nodes.
    """

    print_header("1. LOAD REAL GML SNAPSHOT")

    print(f"Loading:")
    print(f"  {GML_FILE}")

    if not GML_FILE.exists():
        raise FileNotFoundError(
            f"GML file not found:\n{GML_FILE}"
        )

    G = nx.read_gml(
        GML_FILE,
        label="id"
    )

    print()
    print("Original graph type:")
    print(f"  {type(G).__name__}")

    print()
    print("Original graph:")
    print(f"  Nodes : {G.number_of_nodes():,}")
    print(f"  Edges : {G.number_of_edges():,}")

    # ------------------------------------------------------
    # Convert graph type if necessary
    # ------------------------------------------------------

    if isinstance(G, nx.MultiDiGraph):
        pass

    elif isinstance(G, nx.DiGraph):
        G = nx.MultiDiGraph(G)

    elif isinstance(G, nx.MultiGraph):
        G = nx.MultiDiGraph(G)

    elif isinstance(G, nx.Graph):
        G = nx.MultiDiGraph(G)

    else:
        raise TypeError(
            f"Unsupported graph type: {type(G).__name__}"
        )

    print()
    print("Routing graph:")
    print(f"  Type  : {type(G).__name__}")
    print(f"  Nodes : {G.number_of_nodes():,}")
    print(f"  Edges : {G.number_of_edges():,}")

    return G


# ==========================================================
# Graph statistics
# ==========================================================

def print_graph_statistics(G):

    print_header("2. GRAPH STATISTICS")

    nodes = list(G.nodes)

    online_nodes = sum(
        1
        for n in nodes
        if G.nodes[n].get("online", True)
    )

    available_edges = 0

    for _, _, _, data in G.edges(
        keys=True,
        data=True
    ):
        if data.get("available", True):
            available_edges += 1

    print(f"Total nodes              : {G.number_of_nodes():,}")
    print(f"Total directed channels  : {G.number_of_edges():,}")
    print(f"Online nodes             : {online_nodes:,}")
    print(f"Available channels       : {available_edges:,}")

    # ------------------------------------------------------
    # Geographic attributes
    # ------------------------------------------------------

    geo_nodes = 0
    carbon_nodes = 0

    for node in nodes:

        data = G.nodes[node]

        if (
            "latitude" in data
            and
            "longitude" in data
        ):
            geo_nodes += 1

        if "carbon_intensity" in data:
            carbon_nodes += 1

    print(f"Nodes with coordinates   : {geo_nodes:,}")
    print(f"Nodes with carbon data   : {carbon_nodes:,}")

    # ------------------------------------------------------
    # Directional liquidity information
    # ------------------------------------------------------

    known_liquidity = 0

    for _, _, _, data in G.edges(
        keys=True,
        data=True
    ):

        if any(
            key in data
            for key in (
                "estimated_liquidity",
                "liquidity_uv",
                "balance_uv"
            )
        ):
            known_liquidity += 1

    print()
    print("Directional liquidity:")
    print(
        "  Channels with learned/observed liquidity : "
        f"{known_liquidity:,}"
    )
    print(
        "  Snapshot capacity is NOT used as balance."
    )


# ==========================================================
# Select source and destination
# ==========================================================

def select_source_destination(G):

    print_header("3. SELECT SOURCE / DESTINATION")

    random.seed(RANDOM_SEED)

    # ------------------------------------------------------
    # Use the largest strongly connected component.
    #
    # This increases the probability that a directed route
    # exists between source and destination.
    # ------------------------------------------------------

    components = list(
        nx.strongly_connected_components(G)
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

    print(f"Random seed              : {RANDOM_SEED}")
    print(f"Selected component size  : {len(component_nodes):,}")
    print()
    print(f"Source                   : {source}")
    print(f"Destination              : {target}")
    print(f"Payment amount           : {AMOUNT:,.0f} sat")
    print(f"K                        : {K}")
    print(f"Maximum hops             : {MAX_HOPS}")
    print(f"ETA                      : {ETA}")

    return source, target


# ==========================================================
# Route validation
# ==========================================================

def validate_route(G, route):

    path = route.get("path")

    if not path:
        return False, "empty_path"

    if path[0] != route["path"][0]:
        return False, "invalid_source"

    if len(path) - 1 != route["hop_count"]:
        return False, "hop_count_mismatch"

    # ------------------------------------------------------
    # Check that every consecutive pair exists.
    # ------------------------------------------------------

    for u, v in zip(
        path[:-1],
        path[1:]
    ):
        if not G.has_edge(u, v):
            return False, f"missing_edge:{u}->{v}"

    return True, "OK"


# ==========================================================
# Print individual route
# ==========================================================

def print_route(G, route, rank):

    print()
    print_separator("-", 78)

    print(f"ROUTE #{rank}")
    print_separator("-", 78)

    path = route.get("path", [])
    edges = route.get("edges", [])

    print()
    print(f"Rank                     : {rank}")
    print(f"Hop count                : {route.get('hop_count')}")
    print(f"Cost                     : {route.get('cost'):.6f}")
    print(f"ETA                      : {route.get('eta', ETA)}")

    print()
    print("Path:")
    print(
        "  "
        + " -> ".join(
            str(node)
            for node in path
        )
    )

    print()
    print("Path nodes:")
    for index, node in enumerate(path):
        print(
            f"  {index:02d}. {node}"
        )

    print()
    print("Channels:")

    for index, edge in enumerate(edges):

        if len(edge) == 3:
            u, v, key = edge

            print(
                f"  {index + 1:02d}. "
                f"{u} -> {v} "
                f"(channel={key})"
            )

        else:
            print(
                f"  {index + 1:02d}. {edge}"
            )

    # ------------------------------------------------------
    # Additional route metrics
    # ------------------------------------------------------

    print()
    print("Route metrics:")

    metrics = [
        ("Total fee", "total_fee"),
        ("Total delay", "total_delay"),
        ("Reliability", "reliability"),
        ("Failure probability", "failure_probability"),
        ("Total distance (km)", "total_distance_km"),
        ("Total carbon", "total_carbon"),
        ("Inter-country hops", "inter_country_hops"),
        ("Inter-continent hops", "inter_continent_hops"),
        ("Minimum known liquidity", "min_liquidity"),
    ]

    for label, key in metrics:

        value = route.get(key)

        if value is None:
            value_text = "Unknown"
        elif isinstance(value, float):
            value_text = f"{value:.6f}"
        else:
            value_text = str(value)

        print(
            f"  {label:<28}: {value_text}"
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    valid, reason = validate_route(
        G,
        route
    )

    print()
    print(
        f"Route validation          : "
        f"{'PASS' if valid else 'FAIL'}"
    )

    if not valid:
        print(
            f"Validation reason         : {reason}"
        )


# ==========================================================
# Summary table
# ==========================================================

def print_summary(routes):

    print_header("5. TOP-K ROUTES SUMMARY")

    if not routes:
        print("No route was found.")
        return

    header = (
        f"{'Rank':<6}"
        f"{'Hops':<8}"
        f"{'Cost':<16}"
        f"{'Fee':<16}"
        f"{'Delay':<12}"
        f"{'Reliability':<15}"
        f"{'Distance(km)':<16}"
    )

    print(header)
    print_separator("-", len(header))

    for rank, route in enumerate(
        routes,
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{route.get('hop_count', 0):<8}"
            f"{route.get('cost', 0):<16.4f}"
            f"{route.get('total_fee', 0):<16.4f}"
            f"{route.get('total_delay', 0):<12.2f}"
            f"{route.get('reliability', 0):<15.6f}"
            f"{route.get('total_distance_km', 0):<16.2f}"
        )


# ==========================================================
# Main test
# ==========================================================

def main():

    print()
    print_separator()
    print(" REAL GML PATHFINDING TEST ")
    print_separator()

    # ------------------------------------------------------
    # 1. Load graph
    # ------------------------------------------------------

    G = load_graph()

    # ------------------------------------------------------
    # 2. Graph statistics
    # ------------------------------------------------------

    print_graph_statistics(G)

    # ------------------------------------------------------
    # 3. Select source and destination
    # ------------------------------------------------------

    source, target = select_source_destination(
        G
    )

    # ------------------------------------------------------
    # 4. Run top-K pathfinding
    # ------------------------------------------------------

    print_header("4. RUN TOP-K PATHFINDING")

    print("Routing configuration:")
    print(f"  Source       : {source}")
    print(f"  Target       : {target}")
    print(f"  Amount       : {AMOUNT:,.0f} sat")
    print(f"  K            : {K}")
    print(f"  Max hops     : {MAX_HOPS}")
    print(f"  ETA          : {ETA}")

    print()
    print("Running pathfinding...")

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
        print_separator("!", 78)
        print("PATHFINDING ERROR")
        print_separator("!", 78)

        print()
        print(type(exc).__name__)
        print(str(exc))

        print()
        print(
            "This error is generated by the actual "
            "top_k_paths implementation."
        )

        raise

    print()
    print(
        f"Number of routes returned : {len(routes)}"
    )

    # ------------------------------------------------------
    # 5. Print each route
    # ------------------------------------------------------

    for rank, route in enumerate(
        routes,
        start=1
    ):
        print_route(
            G,
            route,
            rank
        )

    # ------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------

    print_summary(
        routes
    )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print_header("6. TEST RESULT")

    if routes:

        print("PATHFINDING STATUS : SUCCESS")

        print()
        print(
            f"{len(routes)} candidate route(s) "
            f"were generated."
        )

        print(
            f"Requested K        : {K}"
        )

        print(
            f"Returned routes    : {len(routes)}"
        )

        print()
        print(
            "The routes are ready to be passed "
            "to the Bucket module."
        )

    else:

        print("PATHFINDING STATUS : NO ROUTE")

        print()
        print(
            "No feasible candidate route was found "
            "for the selected source, destination, "
            "amount and routing constraints."
        )

    print()
    print_separator()
    print(" END OF PATHFINDING TEST ")
    print_separator()


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    main()