"""
Pathfinding/test_pathfinding_bucket.py

Integration test:
    Top-K Pathfinding -> Bucket

Responsibilities
----------------
1. Load the real Lightning GML snapshot.
2. Build the directed routing graph.
3. Select a source, destination and payment amount.
4. Run adaptive Top-K Pathfinding.
5. Pass the returned candidate dictionaries directly to Bucket.
6. Validate Bucket candidates.
7. Verify candidate selection and Bucket state.

Important
---------
This test does NOT:
    - train PPO
    - execute a real payment
    - use PaymentSimulator
    - use FailureModel
    - perform stochastic channel failures
    - perform Partial Backtracking after payment failure

At this stage the verified chain is:

    Real GML
        |
        v
    Routing Graph
        |
        v
    Adaptive Top-K Pathfinding
        |
        v
    Candidate Routes
        |
        v
    Bucket
        |
        v
    Candidate Validation
        |
        v
    Candidate Selection

Payment execution will be connected in the next integration stage.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from Bucket.bucket import (
    Bucket,
    validate_candidate,
    execute_bucket,
)

from Pathfinding.top_k_paths import top_k_paths
from Pathfinding.heuristics import (
    lnd_cost,
    validate_eta,
    validate_lambda_h,
)


# ============================================================
# CONFIGURATION
# ============================================================

GML_FILE = PROJECT_ROOT / "20190501.gml.geo"

RANDOM_SEED = 42

K = 5
MAX_HOPS = 12

ETA = 0.5
LAMBDA_H = 1.0

AMOUNT = 10_000


# ============================================================
# OUTPUT HELPERS
# ============================================================

def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def format_value(value: Any) -> str:
    if value is None:
        return "Unknown"

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


# ============================================================
# GRAPH LOADING
# ============================================================

def load_routing_graph(path: Path) -> tuple[nx.Graph, nx.MultiDiGraph]:
    """
    Load the real GML graph and prepare the routing graph.

    If the source graph is undirected, both structural directions
    are created explicitly.

    Important:
        Reverse structural edges do not imply that directional
        gossip or liquidity information is known.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"GML snapshot not found:\n  {path}"
        )

    print("Loading:")
    print(f"  {path}")

    original_graph = nx.read_gml(path)

    print()
    print("Original graph type:")
    print(f"  {type(original_graph).__name__}")

    print()
    print("Original graph:")
    print(f"  Nodes : {original_graph.number_of_nodes():,}")
    print(f"  Edges : {original_graph.number_of_edges():,}")

    # --------------------------------------------------------
    # Already directed
    # --------------------------------------------------------

    if original_graph.is_directed():

        routing_graph = nx.MultiDiGraph()

        for node, attrs in original_graph.nodes(data=True):
            routing_graph.add_node(
                node,
                **dict(attrs)
            )

        if original_graph.is_multigraph():

            for u, v, key, data in original_graph.edges(
                keys=True,
                data=True
            ):
                routing_graph.add_edge(
                    u,
                    v,
                    key=key,
                    **dict(data)
                )

        else:

            for u, v, data in original_graph.edges(
                data=True
            ):
                routing_graph.add_edge(
                    u,
                    v,
                    key=0,
                    **dict(data)
                )

        return original_graph, routing_graph

    # --------------------------------------------------------
    # Undirected -> explicit structural directions
    # --------------------------------------------------------

    routing_graph = nx.MultiDiGraph()

    for node, attrs in original_graph.nodes(data=True):
        routing_graph.add_node(
            node,
            **dict(attrs)
        )

    if original_graph.is_multigraph():

        for u, v, key, data in original_graph.edges(
            keys=True,
            data=True
        ):

            attrs_uv = dict(data)
            attrs_vu = dict(data)

            routing_graph.add_edge(
                u,
                v,
                key=key,
                **attrs_uv
            )

            routing_graph.add_edge(
                v,
                u,
                key=key,
                **attrs_vu
            )

    else:

        for u, v, data in original_graph.edges(
            data=True
        ):

            attrs_uv = dict(data)
            attrs_vu = dict(data)

            routing_graph.add_edge(
                u,
                v,
                key=0,
                **attrs_uv
            )

            routing_graph.add_edge(
                v,
                u,
                key=0,
                **attrs_vu
            )

    return original_graph, routing_graph


# ============================================================
# GRAPH STATISTICS
# ============================================================

def print_graph_statistics(
    G: nx.MultiDiGraph
) -> None:

    print_section("2. GRAPH STATISTICS")

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    online_nodes = 0
    available_nodes = 0

    available_channels = 0
    unavailable_channels = 0

    known_liquidity = 0
    capacity_only = 0
    completely_unknown = 0

    for _, data in G.nodes(data=True):

        online = data.get(
            "online",
            True
        )

        if online:
            online_nodes += 1

        if data.get(
            "available",
            True
        ):
            available_nodes += 1

    for _, _, _, data in G.edges(
        keys=True,
        data=True
    ):

        if data.get(
            "available",
            True
        ):
            available_channels += 1
        else:
            unavailable_channels += 1

        liquidity_keys = (
            "estimated_liquidity",
            "liquidity_uv",
            "balance_uv",
        )

        has_liquidity = any(
            key in data and data[key] is not None
            for key in liquidity_keys
        )

        has_capacity = (
            "capacity" in data
            and data["capacity"] is not None
        )

        if has_liquidity:
            known_liquidity += 1

        elif has_capacity:
            capacity_only += 1

        else:
            completely_unknown += 1

    print(f"Total nodes             : {node_count:,}")
    print(f"Total directed channels : {edge_count:,}")
    print(f"Online nodes            : {online_nodes:,}")
    print(f"Available nodes         : {available_nodes:,}")
    print(f"Available channels      : {available_channels:,}")
    print(f"Unavailable channels    : {unavailable_channels:,}")

    print()
    print("Directional liquidity:")
    print(
        f"  Known directional liquidity : "
        f"{known_liquidity:,}"
    )
    print(
        f"  Capacity only               : "
        f"{capacity_only:,}"
    )
    print(
        f"  Completely unknown          : "
        f"{completely_unknown:,}"
    )

    print()
    print("IMPORTANT:")
    print(
        "  Channel capacity is NOT interpreted "
        "as directional liquidity."
    )


# ============================================================
# SOURCE / DESTINATION SELECTION
# ============================================================

def select_source_destination(
    G: nx.MultiDiGraph,
    seed: int = RANDOM_SEED,
) -> tuple[Any, Any, int]:
    """
    Select source and destination from the largest SCC.
    """

    print_section(
        "3. SELECT SOURCE / DESTINATION"
    )

    if G.number_of_nodes() < 2:
        raise RuntimeError(
            "Routing graph contains fewer than two nodes."
        )

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

    if len(largest_component) < 2:
        raise RuntimeError(
            "Largest strongly connected component "
            "contains fewer than two nodes."
        )

    rng = random.Random(seed)

    nodes = list(largest_component)

    source, destination = rng.sample(
        nodes,
        2
    )

    print(f"Random seed              : {seed}")
    print(
        f"Selected component size : "
        f"{len(largest_component):,}"
    )

    print()
    print(f"Source                   : {source}")
    print(f"Destination              : {destination}")
    print(f"Payment amount           : {AMOUNT:,}")
    print(f"K                        : {K}")
    print(f"Maximum hops             : {MAX_HOPS}")
    print(f"ETA                      : {ETA}")
    print(f"Lambda_h                 : {LAMBDA_H}")

    return (
        source,
        destination,
        len(largest_component),
    )


# ============================================================
# PATHFINDING
# ============================================================

def run_top_k_pathfinding(
    G: nx.MultiDiGraph,
    source: Any,
    destination: Any,
) -> list[dict]:
    """
    Run adaptive Top-K Pathfinding.
    """

    print_section(
        "4. RUN TOP-K PATHFINDING"
    )

    validate_eta(ETA)
    validate_lambda_h(LAMBDA_H)

    print("Routing configuration:")
    print(f"  Source       : {source}")
    print(f"  Target       : {destination}")
    print(f"  Amount       : {AMOUNT:,}")
    print(f"  K            : {K}")
    print(f"  Max hops     : {MAX_HOPS}")
    print(f"  ETA          : {ETA}")
    print(f"  Lambda_h     : {LAMBDA_H}")

    print()
    print("Running adaptive top-k pathfinding...")

    routes = top_k_paths(
        G,
        source,
        destination,
        AMOUNT,
        heuristic_fn=lnd_cost,
        eta=ETA,
        k=K,
        max_hops=MAX_HOPS,
        lambda_h=LAMBDA_H,
    )

    if routes is None:
        routes = []

    if not isinstance(routes, list):
        routes = list(routes)

    print()
    print(
        f"Number of routes returned : "
        f"{len(routes)}"
    )

    return routes


# ============================================================
# CANDIDATE VALIDATION
# ============================================================

def validate_routes(
    G: nx.MultiDiGraph,
    routes: list[dict],
) -> tuple[int, int]:
    """
    Validate all routes before inserting them into Bucket.

    Returns
    -------
    structural_passed, metadata_passed
    """

    print_section(
        "5. VALIDATE PATHFINDING CANDIDATES"
    )

    structural_passed = 0
    metadata_passed = 0

    if not routes:
        print("No candidate routes returned.")
        return 0, 0

    for index, route in enumerate(
        routes,
        start=1
    ):

        # ----------------------------------------------------
        # Structural validation through Bucket's own validator
        # ----------------------------------------------------

        valid, failed_channel, reason = (
            validate_candidate(
                G,
                route,
                AMOUNT
            )
        )

        if valid:
            structural_passed += 1
            structural_status = "PASS"

        else:
            structural_status = "FAIL"

        # ----------------------------------------------------
        # Metadata validation
        # ----------------------------------------------------

        metadata_ok = True

        if not isinstance(
            route,
            dict
        ):
            metadata_ok = False

        else:

            required_keys = (
                "path",
                "edges",
                "cost",
            )

            for key in required_keys:

                if key not in route:
                    metadata_ok = False

            path = route.get("path")

            edges = route.get("edges")

            if not isinstance(
                path,
                (list, tuple)
            ):
                metadata_ok = False

            if not isinstance(
                edges,
                (list, tuple)
            ):
                metadata_ok = False

            route_eta = route.get(
                "eta"
            )

            if route_eta is not None:

                try:
                    route_eta = float(
                        route_eta
                    )

                    if not (
                        0.0 <= route_eta <= 1.0
                    ):
                        metadata_ok = False

                except (
                    TypeError,
                    ValueError
                ):
                    metadata_ok = False

            route_lambda = route.get(
                "lambda_h"
            )

            if route_lambda is not None:

                try:
                    route_lambda = float(
                        route_lambda
                    )

                    if route_lambda < 0:
                        metadata_ok = False

                except (
                    TypeError,
                    ValueError
                ):
                    metadata_ok = False

        if metadata_ok:
            metadata_passed += 1
            metadata_status = "PASS"

        else:
            metadata_status = "FAIL"

        print(
            f"Route #{index:<2} "
            f"structural={structural_status} "
            f"metadata={metadata_status}"
        )

        if not valid:
            print(
                f"  Validation reason : {reason}"
            )

            print(
                f"  Failed channel    : "
                f"{failed_channel}"
            )

    print()
    print(
        f"Structural validation : "
        f"{structural_passed}/{len(routes)}"
    )

    print(
        f"Metadata validation   : "
        f"{metadata_passed}/{len(routes)}"
    )

    return (
        structural_passed,
        metadata_passed,
    )


# ============================================================
# BUCKET CREATION
# ============================================================

def create_bucket(
    routes: list[dict],
    transaction_id: Any = "integration-test-tx",
) -> Bucket:
    """
    Create a Bucket directly from Top-K candidates.
    """

    print_section(
        "6. CREATE BUCKET FROM TOP-K CANDIDATES"
    )

    bucket = Bucket(
        bucket_id="pathfinding-bucket-001",
        transaction_id=transaction_id,
        candidates=list(routes),
    )

    print(
        f"Bucket ID               : "
        f"{bucket.bucket_id}"
    )

    print(
        f"Transaction ID          : "
        f"{bucket.transaction_id}"
    )

    print(
        f"Candidates inserted     : "
        f"{len(bucket.candidates)}"
    )

    print(
        f"Current index           : "
        f"{bucket.current_index}"
    )

    print(
        f"Initial status          : "
        f"{bucket.status}"
    )

    return bucket


# ============================================================
# BUCKET CONTENT VALIDATION
# ============================================================

def validate_bucket_content(
    bucket: Bucket,
) -> bool:
    """
    Verify that the Bucket received the same candidates
    generated by Pathfinding.
    """

    print_section(
        "7. VALIDATE BUCKET CONTENT"
    )

    passed = True

    if not bucket.candidates:
        print("Bucket candidates : FAIL")
        return False

    for index, candidate in enumerate(
        bucket.candidates,
        start=1
    ):

        if not isinstance(
            candidate,
            dict
        ):
            print(
                f"Candidate #{index}: FAIL "
                "(not a dictionary)"
            )
            passed = False
            continue

        path = candidate.get(
            "path"
        )

        edges = candidate.get(
            "edges"
        )

        cost = candidate.get(
            "cost"
        )

        candidate_flag = candidate.get(
            "candidate",
            True
        )

        valid = (
            isinstance(path, (list, tuple))
            and isinstance(edges, (list, tuple))
            and cost is not None
            and bool(candidate_flag)
        )

        if valid:
            print(
                f"Candidate #{index:<2}: PASS "
                f"| hops={len(path) - 1:<2} "
                f"| cost={float(cost):.6f}"
            )

        else:
            print(
                f"Candidate #{index:<2}: FAIL"
            )
            passed = False

    print()

    if passed:
        print(
            "Bucket content validation : PASS"
        )
    else:
        print(
            "Bucket content validation : FAIL"
        )

    return passed


# ============================================================
# BUCKET EXECUTION
# ============================================================

def run_bucket_execution(
    G: nx.MultiDiGraph,
    bucket: Bucket,
) -> tuple[
    bool,
    Any,
    int,
]:
    """
    Run Bucket candidate validation.

    IMPORTANT:
        This is NOT payment execution.

    execute_bucket() only verifies that a candidate is
    structurally usable by the current Bucket logic.
    """

    print_section(
        "8. EXECUTE BUCKET CANDIDATE SELECTION"
    )

    print(
        "Starting Bucket execution..."
    )

    success, selected_candidate, attempts = (
        execute_bucket(
            G,
            bucket,
            AMOUNT,
        )
    )

    print()
    print(
        f"Bucket execution result : "
        f"{'PASS' if success else 'FAIL'}"
    )

    print(
        f"Attempts                : "
        f"{attempts}"
    )

    print(
        f"Bucket status           : "
        f"{bucket.status}"
    )

    print(
        f"Current index           : "
        f"{bucket.current_index}"
    )

    print(
        f"Failed candidates      : "
        f"{bucket.failed_candidates_count}"
    )

    print(
        f"Failed channels        : "
        f"{len(bucket.failed_channels)}"
    )

    if selected_candidate is not None:

        selected_rank = (
            bucket.selected_candidate_rank
        )

        print(
            f"Selected candidate rank : "
            f"{selected_rank}"
        )

        print(
            f"Selected route cost     : "
            f"{format_value(selected_candidate.get('cost'))}"
        )

        print(
            f"Selected route hops     : "
            f"{selected_candidate.get('hop_count', 'Unknown')}"
        )

    else:

        print(
            "Selected candidate rank : None"
        )

    return (
        success,
        selected_candidate,
        attempts,
    )


# ============================================================
# SELECTED CANDIDATE VALIDATION
# ============================================================

def validate_selected_candidate(
    G: nx.MultiDiGraph,
    bucket: Bucket,
    selected_candidate: Any,
) -> bool:
    """
    Verify that Bucket's selected candidate is one of
    the original Top-K candidates and remains structurally
    valid.
    """

    print_section(
        "9. VALIDATE SELECTED CANDIDATE"
    )

    if selected_candidate is None:

        print(
            "Selected candidate : FAIL"
        )

        return False

    if selected_candidate not in bucket.candidates:

        print(
            "Selected candidate belongs to Bucket : FAIL"
        )

        return False

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            selected_candidate,
            AMOUNT,
        )
    )

    if valid:

        print(
            "Selected candidate belongs to Bucket : PASS"
        )

        print(
            "Selected candidate structural validation : PASS"
        )

        print(
            f"Selected candidate rank : "
            f"{bucket.selected_candidate_rank}"
        )

        print(
            f"Selected candidate index : "
            f"{bucket.selected_candidate_index}"
        )

        print(
            f"Selected route           : "
            f"{selected_candidate.get('path')}"
        )

        return True

    print(
        "Selected candidate structural validation : FAIL"
    )

    print(
        f"Reason         : {reason}"
    )

    print(
        f"Failed channel : {failed_channel}"
    )

    return False


# ============================================================
# BUCKET INFORMATION
# ============================================================

def print_bucket_summary(
    bucket: Bucket,
) -> None:

    print_section(
        "10. BUCKET SUMMARY"
    )

    info = bucket.info()

    print(
        f"Bucket ID                : "
        f"{info['bucket_id']}"
    )

    print(
        f"Transaction ID           : "
        f"{info['transaction_id']}"
    )

    print(
        f"Candidate count          : "
        f"{info['candidate_count']}"
    )

    print(
        f"Current index            : "
        f"{info['current_index']}"
    )

    print(
        f"Attempts                 : "
        f"{info['attempts']}"
    )

    print(
        f"Status                   : "
        f"{info['status']}"
    )

    print(
        f"Selected candidate index : "
        f"{info['selected_candidate_index']}"
    )

    print(
        f"Selected candidate rank  : "
        f"{info['selected_candidate_rank']}"
    )

    print(
        f"Failed candidates        : "
        f"{info['failed_candidates_count']}"
    )

    print(
        f"Failed channels          : "
        f"{info['failed_channels_count']}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    routes: list[dict],
    bucket: Bucket,
    content_ok: bool,
    execution_ok: bool,
    selected_ok: bool,
) -> bool:
    """
    Final Pathfinding -> Bucket integration validation.
    """

    print_section(
        "11. INTEGRATION TEST RESULT"
    )

    pathfinding_ok = (
        isinstance(routes, list)
        and len(routes) > 0
        and len(routes) <= K
    )

    bucket_count_ok = (
        len(bucket.candidates) == len(routes)
    )

    bucket_status_ok = (
        bucket.status == "completed"
        and bucket.selected_candidate is not None
    )

    all_ok = all(
        [
            pathfinding_ok,
            bucket_count_ok,
            content_ok,
            execution_ok,
            selected_ok,
            bucket_status_ok,
        ]
    )

    print(
        "Pathfinding returned candidates : "
        f"{'PASS' if pathfinding_ok else 'FAIL'}"
    )

    print(
        "All candidates transferred      : "
        f"{'PASS' if bucket_count_ok else 'FAIL'}"
    )

    print(
        "Bucket content validation       : "
        f"{'PASS' if content_ok else 'FAIL'}"
    )

    print(
        "Bucket candidate execution      : "
        f"{'PASS' if execution_ok else 'FAIL'}"
    )

    print(
        "Selected candidate validation   : "
        f"{'PASS' if selected_ok else 'FAIL'}"
    )

    print(
        "Bucket completed state          : "
        f"{'PASS' if bucket_status_ok else 'FAIL'}"
    )

    print()

    if all_ok:

        print(
            "PATHFINDING -> BUCKET : SUCCESS"
        )

    else:

        print(
            "PATHFINDING -> BUCKET : FAILED"
        )

    return all_ok


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        " PATHFINDING -> BUCKET INTEGRATION TEST "
    )
    print(
        " Real Lightning GML / Adaptive Top-K / Bucket "
    )
    print("=" * 78)

    # --------------------------------------------------------
    # 1. Load graph
    # --------------------------------------------------------

    print_section(
        "1. LOAD REAL GML SNAPSHOT"
    )

    original_graph, G = load_routing_graph(
        GML_FILE
    )

    print()
    print("Routing graph:")
    print(
        f"  Type  : {type(G).__name__}"
    )
    print(
        f"  Nodes : {G.number_of_nodes():,}"
    )
    print(
        f"  Edges : {G.number_of_edges():,}"
    )

    if not original_graph.is_directed():

        print()
        print("WARNING:")
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

    # --------------------------------------------------------
    # 2. Statistics
    # --------------------------------------------------------

    print_graph_statistics(
        G
    )

    # --------------------------------------------------------
    # 3. Source / destination
    # --------------------------------------------------------

    source, destination, _ = (
        select_source_destination(
            G,
            RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # 4. Pathfinding
    # --------------------------------------------------------

    routes = run_top_k_pathfinding(
        G,
        source,
        destination,
    )

    if not routes:

        print()
        print(
            "No routes were returned by Pathfinding."
        )

        print()
        print(
            "PATHFINDING -> BUCKET : FAILED"
        )

        return

    # --------------------------------------------------------
    # 5. Candidate validation
    # --------------------------------------------------------

    structural_passed, metadata_passed = (
        validate_routes(
            G,
            routes,
        )
    )

    pathfinding_candidates_ok = (
        structural_passed == len(routes)
        and metadata_passed == len(routes)
    )

    if not pathfinding_candidates_ok:

        print()
        print(
            "Pathfinding candidates failed validation."
        )

        print()
        print(
            "PATHFINDING -> BUCKET : FAILED"
        )

        return

    # --------------------------------------------------------
    # 6. Create Bucket
    # --------------------------------------------------------

    bucket = create_bucket(
        routes,
        transaction_id="integration-test-tx-001",
    )

    # --------------------------------------------------------
    # 7. Validate Bucket content
    # --------------------------------------------------------

    content_ok = validate_bucket_content(
        bucket
    )

    if not content_ok:

        print()
        print(
            "PATHFINDING -> BUCKET : FAILED"
        )

        return

    # --------------------------------------------------------
    # 8. Execute Bucket selection
    # --------------------------------------------------------

    execution_ok, selected_candidate, attempts = (
        run_bucket_execution(
            G,
            bucket,
        )
    )

    # --------------------------------------------------------
    # 9. Validate selected candidate
    # --------------------------------------------------------

    selected_ok = validate_selected_candidate(
        G,
        bucket,
        selected_candidate,
    )

    # --------------------------------------------------------
    # 10. Bucket summary
    # --------------------------------------------------------

    print_bucket_summary(
        bucket
    )

    # --------------------------------------------------------
    # 11. Final result
    # --------------------------------------------------------

    final_validation(
        routes,
        bucket,
        content_ok,
        execution_ok,
        selected_ok,
    )

    print()
    print("=" * 78)
    print(
        " END OF PATHFINDING -> BUCKET INTEGRATION TEST "
    )
    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()