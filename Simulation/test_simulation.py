# Simulation/test_simulation.py

from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from main import geo_to_json
from Network.graph_builder import LNGraphBuilder

from Simulation.transaction_generator import (
    generate_transactions
)

from Simulation.payment_simulator import (
    simulate_payment
)

from Simulation.failure_model import (
    FailureModel
)

from Simulation.network_dynamics import (
    NetworkDynamics
)

from Simulation.onion import (
    OnionRouter
)


# ==========================================================
# Load Real Lightning Network Snapshot
# ==========================================================

def build_real_graph():

    print(
        "\n===== LOAD REAL LIGHTNING SNAPSHOT ====="
    )

    geo_file = Path("20190501.gml.geo")

    if not geo_file.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {geo_file}"
        )

    print(
        f"Snapshot: {geo_file}"
    )

    # ------------------------------------------------------
    # .geo / GML -> in-memory JSON structure
    # ------------------------------------------------------

    data = geo_to_json(geo_file)

    print(
        f"JSON nodes: {len(data['nodes'])}"
    )

    print(
        f"JSON edges: {len(data['edges'])}"
    )

    # ------------------------------------------------------
    # In-memory JSON -> Lightning Network graph
    # ------------------------------------------------------

    builder = LNGraphBuilder()

    G = builder.from_data(data)

    print(
        f"Graph nodes: {G.number_of_nodes()}"
    )

    print(
        f"Graph edges: {G.number_of_edges()}"
    )

    return G


# ==========================================================
# Find a Real Connected Route
# ==========================================================

def find_real_route(G):

    print(
        "\n===== FIND REAL ROUTE ====="
    )

    nodes = list(G.nodes)

    if len(nodes) < 2:
        raise RuntimeError(
            "The graph does not contain enough nodes."
        )

    # Convert MultiGraph/MultiDiGraph to an
    # undirected simple graph for finding a test path.
    simple_graph = nx.Graph()

    for node in G.nodes:
        simple_graph.add_node(node)

    for u, v in G.edges():

        if u != v:
            simple_graph.add_edge(u, v)

    # ------------------------------------------------------
    # Try to find a reasonably short real path
    # ------------------------------------------------------

    source = None
    destination = None
    path = None

    for i in range(min(len(nodes), 200)):

        for j in range(i + 1, min(len(nodes), 200)):

            u = nodes[i]
            v = nodes[j]

            try:

                candidate = nx.shortest_path(
                    simple_graph,
                    source=u,
                    target=v
                )

                if len(candidate) >= 2:
                    source = u
                    destination = v
                    path = candidate
                    break

            except nx.NetworkXNoPath:
                continue

        if path is not None:
            break

    if path is None:
        raise RuntimeError(
            "Could not find a connected route in the snapshot."
        )

    print(
        "Source:",
        source
    )

    print(
        "Destination:",
        destination
    )

    print(
        "Path:",
        path
    )

    return path


# ==========================================================
# Build Edge List for a Real Route
# ==========================================================

def get_route_edges(G, path):

    edges = []

    for u, v in zip(path[:-1], path[1:]):

        edge_data = G.get_edge_data(u, v)

        if edge_data is None:

            edge_data = G.get_edge_data(v, u)

            if edge_data is None:
                raise RuntimeError(
                    f"No edge found between {u} and {v}"
                )

        # MultiGraph / MultiDiGraph
        if isinstance(edge_data, dict):

            key = next(iter(edge_data))

            edges.append(
                (
                    u,
                    v,
                    key
                )
            )

        else:

            edges.append(
                (
                    u,
                    v,
                    0
                )
            )

    return edges


# ==========================================================
# Failure Model Adapter
# ==========================================================

def build_failure_network(G):

    network = SimpleNamespace()

    network.nodes = {

        node:

        SimpleNamespace(

            is_online=data.get(
                "available",
                True
            )

        )

        for node, data in G.nodes(
            data=True
        )

    }

    network.channels = []

    for u, v, k, data in G.edges(
        keys=True,
        data=True
    ):

        network.channels.append(

            SimpleNamespace(

                node1=u,

                node2=v,

                available=data.get(
                    "available",
                    True
                ),

                failure_probability=data.get(
                    "failure_probability",
                    0.01
                ),

                failure_count=data.get(
                    "failure_count",
                    0
                ),

                capacity=data.get(
                    "capacity",
                    0
                )

            )

        )

    return network


# ==========================================================
# Transaction Generator
# ==========================================================

def test_transaction_generator(G):

    print(
        "\n===== TRANSACTION GENERATOR ====="
    )

    transactions = generate_transactions(
        G,
        n=3
    )

    for tx in transactions:

        print(tx)

    return transactions[0]


# ==========================================================
# Onion Router
# ==========================================================

def test_onion(path, transaction):

    print(
        "\n===== ONION ROUTER ====="
    )

    router = OnionRouter()

    packet = router.build_onion(
        path,
        bucket_id="test_bucket",
        tx_id=f"tx_{transaction.tx_id}",
        amount=transaction.amount
    )

    print(
        "Packet created"
    )

    # Peel the first layer using the actual
    # source node of the transaction.

    first_node = path[0]

    layer = router.peel_layer(
        packet,
        first_node
    )

    print(
        "Current node:",
        first_node
    )

    print(
        "Next hop:",
        layer["next_hop"]
    )

    return packet


# ==========================================================
# Failure Model
# ==========================================================

def test_failure_model(
    G,
    path,
    transaction
):

    print(
        "\n===== FAILURE MODEL ====="
    )

    model = FailureModel()

    network = build_failure_network(G)

    result = model.evaluate_payment_failure(
        route=path,
        amount=transaction.amount,
        network=network
    )

    print(
        result
    )

    return result


# ==========================================================
# Network Dynamics
# ==========================================================

def test_network_dynamics(G):

    print(
        "\n===== NETWORK DYNAMICS ====="
    )

    print(
        "Before update:",
        G
    )

    dynamics = NetworkDynamics(G)

    dynamics.update()

    print(
        "After update:",
        G
    )


# ==========================================================
# Payment Simulator
# ==========================================================

def test_payment_simulator(
    G,
    path,
    transaction
):

    print(
        "\n===== PAYMENT SIMULATOR ====="
    )

    edges = get_route_edges(
        G,
        path
    )

    print(
        "Route:",
        path
    )

    print(
        "Edges:",
        edges
    )

    result = simulate_payment(
        G,
        path,
        edges,
        amount=transaction.amount
    )

    print(
        result.to_dict()
    )

    return result


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "=============================="
    )

    print(
        " REAL LIGHTNING SNAPSHOT TEST "
    )

    print(
        "=============================="
    )

    # ------------------------------------------------------
    # 1. Load the real .geo snapshot
    # ------------------------------------------------------

    G = build_real_graph()

    # ------------------------------------------------------
    # 2. Generate transactions on the real graph
    # ------------------------------------------------------

    transaction = test_transaction_generator(G)

    # ------------------------------------------------------
    # 3. Find an actual route in the real snapshot
    # ------------------------------------------------------

    path = find_real_route(G)

    # ------------------------------------------------------
    # 4. Test Onion Router on the real route
    # ------------------------------------------------------

    test_onion(
        path,
        transaction
    )

    # ------------------------------------------------------
    # 5. Test Failure Model on the real route
    # ------------------------------------------------------

    test_failure_model(
        G,
        path,
        transaction
    )

    # ------------------------------------------------------
    # 6. Test Network Dynamics on the real graph
    # ------------------------------------------------------

    test_network_dynamics(G)

    # ------------------------------------------------------
    # 7. Test Payment Simulator on the real route
    # ------------------------------------------------------

    test_payment_simulator(
        G,
        path,
        transaction
    )

    print(
        "\n=============================="
    )

    print(
        " REAL SNAPSHOT TEST FINISHED "
    )

    print(
        "=============================="
    )


if __name__ == "__main__":

    main()