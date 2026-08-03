import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(ROOT)
)


# -------------------------------
# Network
# -------------------------------

from Network.graph_builder import LNGraphBuilder


# -------------------------------
# Pathfinding
# -------------------------------

from Pathfinding.dijkstra import Dijkstra

from Pathfinding.top_k_paths import top_k_paths

from Pathfinding.heuristics import (
    lnd_cost,
    cln_cost,
    ecl_cost,
    evaluate_path
)



# =====================================================
# 1. Build Test Network
# =====================================================

def build_test_network():

    builder = LNGraphBuilder()


    G = builder.synthetic(

        n=30,

        extra_edges=80,

        seed=42

    )


    return G



# =====================================================
# 2. Print Path Result
# =====================================================

def print_path_result(result):

    print("\n========== PATH RESULT ==========")


    if not result["success"]:

        print(
            "No path found"
        )

        return


    print(
        "Path:"
    )

    print(
        result["path"]
    )


    print(
        "\nChannels:"
    )

    for e in result["edges"]:

        print(e)


    print(
        "\nCost:",
        result["cost"]
    )


    print(
        "Hop Count:",
        result["hop_count"]
    )


    print(
        "Total Fee:",
        result["total_fee"]
    )


    print(
        "Total Delay:",
        result["total_delay"]
    )


    print(
        "Minimum Liquidity:",
        result["min_liquidity"]
    )



# =====================================================
# 3. Test Dijkstra
# =====================================================

def test_dijkstra(G):

    print(
        "\n\n===== TEST DIJKSTRA ====="
    )


    nodes = list(
        G.nodes()
    )


    source = nodes[0]

    target = nodes[-1]


    print(
        "Source:",
        source
    )

    print(
        "Target:",
        target
    )



    router = Dijkstra(G)



    result = router.shortest_path(

        source,

        target,

        amount=1000,

        heuristic_fn=lnd_cost,

        eta=0.2

    )



    print_path_result(
        result
    )



    return result



# =====================================================
# 4. Test Heuristics
# =====================================================

def test_heuristics(
        G,
        result
):

    print(
        "\n\n===== TEST HEURISTICS ====="
    )


    if not result["success"]:

        print(
            "No path available"
        )

        return



    metrics = evaluate_path(

        G,

        result["edges"],

        amount=1000

    )


    print(
        metrics
    )



# =====================================================
# 5. Test Top-K Paths
# =====================================================

def test_top_k(G):

    print(
        "\n\n===== TEST TOP-K PATHS ====="
    )


    nodes = list(
        G.nodes()
    )


    source = nodes[0]

    target = nodes[-1]



    paths = top_k_paths(

        G,

        source,

        target,

        amount=1000,

        heuristic_fn=lnd_cost,

        eta=0.2,

        k=5,

        max_hops=12

    )



    print(
        "Number of candidate paths:",
        len(paths)
    )


    for i,p in enumerate(paths):

        print(
            "\nPath",
            i+1
        )

        print(
            "Nodes:",
            p["path"]
        )

        print(
            "Edges:",
            p["edges"]
        )

        print(
            "Cost:",
            p["cost"]
        )

        print(
            "Hops:",
            p["hop_count"]
        )



# =====================================================
# Main Test
# =====================================================

if __name__ == "__main__":


    print(
        "Creating Lightning Network..."
    )


    G = build_test_network()



    print(
        "\n===== NETWORK INFO ====="
    )

    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Channels:",
        G.number_of_edges()
    )



    # Dijkstra

    result = test_dijkstra(G)



    # Heuristic

    test_heuristics(
        G,
        result
    )



    # Top K

    test_top_k(G)



    print(
        "\n\n===== PATHFINDING TEST COMPLETED ====="
    )