# Simulation/test_simulation.py

import networkx as nx
from types import SimpleNamespace


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
# Build Test Lightning Network
# ==========================================================

def build_test_graph():


    G = nx.MultiGraph()



    nodes = [

        "Alice",

        "Bob",

        "Carol",

        "Dave"

    ]



    for node in nodes:

        G.add_node(

            node,

            country="TEST",

            latitude=0,

            longitude=0,

            carbon_intensity=10,

            available=True

        )



    channels = [

        ("Alice","Bob"),

        ("Bob","Carol"),

        ("Carol","Dave")

    ]



    for u,v in channels:


        G.add_edge(

            u,

            v,

            capacity=1_000_000,

            balance_uv=1_000_000,

            balance_vu=1_000_000,

            available=True,

            failure_probability=0.0,

            fee_base=1,

            fee_rate=1,

            delay=1,

            success_count=0,

            failure_count=0

        )


    return G




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

        for node,data in G.nodes(
            data=True
        )

    }



    network.channels = []



    for u,v,k,data in G.edges(
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

def test_onion():


    print(
        "\n===== ONION ROUTER ====="
    )


    router = OnionRouter()



    path = [

        "Alice",

        "Bob",

        "Carol",

        "Dave"

    ]



    packet = router.build_onion(

        path,

        bucket_id="bucket_1",

        tx_id="tx_1",

        amount=5000

    )



    print(
        "Packet created"
    )


    layer = router.peel_layer(

        packet,

        "Alice"

    )



    print(

        "Next hop:",

        layer["next_hop"]

    )



    return packet





# ==========================================================
# Failure Model
# ==========================================================

def test_failure_model(G):


    print(
        "\n===== FAILURE MODEL ====="
    )


    model = FailureModel()



    network = build_failure_network(G)



    result = model.evaluate_payment_failure(

        route=[

            "Alice",

            "Bob",

            "Carol",

            "Dave"

        ],

        amount=5000,

        network=network

    )



    print(result)





# ==========================================================
# Network Dynamics
# ==========================================================

def test_network_dynamics(G):


    print(
        "\n===== NETWORK DYNAMICS ====="
    )



    dynamics = NetworkDynamics(G)



    print(

        "Before update:",

        G

    )



    dynamics.update()



    print(

        "After update:",

        G

    )





# ==========================================================
# Payment Simulator
# ==========================================================

def test_payment_simulator(G):


    print(
        "\n===== PAYMENT SIMULATOR ====="
    )


    path=[

        "Alice",

        "Bob",

        "Carol",

        "Dave"

    ]



    edges=[

        ("Alice","Bob",0),

        ("Bob","Carol",0),

        ("Carol","Dave",0)

    ]



    result = simulate_payment(

        G,

        path,

        edges,

        amount=5000

    )



    print(

        result.to_dict()

    )





# ==========================================================
# Main
# ==========================================================

def main():


    print(
        "=============================="
    )

    print(
        " SIMULATION MODULE TEST "
    )

    print(
        "=============================="
    )



    G = build_test_graph()



    test_transaction_generator(G)


    test_onion()


    test_failure_model(G)


    test_network_dynamics(G)


    test_payment_simulator(G)



    print(
        "\n=============================="
    )

    print(
        " SIMULATION TEST FINISHED "
    )

    print(
        "=============================="
    )




if __name__ == "__main__":

    main()