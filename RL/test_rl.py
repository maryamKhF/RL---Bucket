# RL/test_rl.py

import numpy as np
import networkx as nx


from .environment import RoutingEnv
from .ppo_agent import build_ppo
from .reward import calculate_reward
from .state import State



# --------------------------------------------------
# Build Dummy Network
# --------------------------------------------------

def build_test_graph():

    G = nx.MultiDiGraph()


    G.add_edge(
        "Alice",
        "Bob",
        key=0,
        fee_base=100,
        delay=5,
        capacity=1000,
        active=1
    )


    G.add_edge(
        "Bob",
        "Carol",
        key=0,
        fee_base=200,
        delay=10,
        capacity=800,
        active=1
    )


    G.add_edge(
        "Alice",
        "Dave",
        key=0,
        fee_base=300,
        delay=20,
        capacity=500,
        active=1
    )


    G.add_edge(
        "Dave",
        "Carol",
        key=0,
        fee_base=100,
        delay=5,
        capacity=900,
        active=1
    )


    return G



# --------------------------------------------------
# Dummy Transaction
# --------------------------------------------------

class Transaction:


    def __init__(
        self,
        source,
        destination,
        amount
    ):

        self.source = source
        self.destination = destination
        self.amount = amount




# --------------------------------------------------
# Dummy Heuristic
# --------------------------------------------------

def heuristic(
    edge_data,
    amount,
    eta
):

    return (

        edge_data.get(
            "fee_base",
            0
        )

        +

        eta *
        edge_data.get(
            "delay",
            0
        )

    )




# --------------------------------------------------
# Dummy Payment Simulator
# --------------------------------------------------

def test_payment_simulator(
    G,
    path,
    edges,
    amount
):


    class Result:

        success = True

        fee = 100

        carbon = 20

        delay = 5



    return Result()




# --------------------------------------------------
# Config
# --------------------------------------------------

CONFIG = {


    "graph": {

        "neighborhood_k":15,

        "neighborhood_m":5,

        "ego_radius":2,

        "max_hops":5,

        "top_k":3

    },


    "simulation": {

        "base_reward_scale":1

    },


    "rl": {


        "hidden_layers":[64,64],

        "learning_rate":3e-4,

        "n_steps":128,

        "batch_size":32,

        "n_epochs":10,

        "gamma":0.99,

        "gae_lambda":0.95,

        "clip_range":0.2,

        "ent_coef":0.01,

        "total_timesteps":1000

    }

}



# --------------------------------------------------
# TEST STATE
# --------------------------------------------------

def test_state():


    G = build_test_graph()


    state = State(

        G,

        "Alice",

        "Carol",

        transaction={

            "amount":1000,

            "source":"Alice",

            "destination":"Carol"

        }

    )


    vector = state.vector()


    print(
        "STATE DIM:",
        state.dimension
    )


    print(
        "STATE SAMPLE:",
        vector[:10]
    )


    assert isinstance(
        vector,
        np.ndarray
    )



# --------------------------------------------------
# TEST REWARD
# --------------------------------------------------

def test_reward():


    r1 = calculate_reward(

        True,

        3,

        10

    )


    r2 = calculate_reward(

        False,

        5,

        10

    )


    print(
        "SUCCESS REWARD:",
        r1
    )


    print(
        "FAILURE REWARD:",
        r2
    )


    assert r1 > r2




# --------------------------------------------------
# TEST ENVIRONMENT
# --------------------------------------------------

def test_environment():


    G = build_test_graph()


    transactions=[

        Transaction(
            "Alice",
            "Carol",
            1000
        )

    ]



    env = RoutingEnv(

        G,

        transactions,

        heuristic,

        CONFIG

    )


    obs,info = env.reset()



    print(
        "OBSERVATION SHAPE:",
        obs.shape
    )



    action = np.array(
        [0.5],
        dtype=np.float32
    )


    result = env.step(
        action
    )


    next_obs,reward,terminated,truncated,info = result



    print(
        "REWARD:",
        reward
    )


    print(
        "INFO:",
        info
    )


    assert terminated is True




# --------------------------------------------------
# TEST PPO BUILD
# --------------------------------------------------

def test_ppo():


    G = build_test_graph()


    transactions=[

        Transaction(
            "Alice",
            "Carol",
            1000
        )

    ]


    env = RoutingEnv(

        G,

        transactions,

        heuristic,

        CONFIG

    )


    model = build_ppo(

        env,

        CONFIG

    )


    print(
        "PPO CREATED:",
        type(model)
    )



# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":


    print(
        "\n===== TEST STATE ====="
    )

    test_state()



    print(
        "\n===== TEST REWARD ====="
    )

    test_reward()



    print(
        "\n===== TEST ENVIRONMENT ====="
    )

    test_environment()



    print(
        "\n===== TEST PPO ====="
    )

    test_ppo()



    print(
        "\n===== RL MODULE TEST PASSED ====="
    )