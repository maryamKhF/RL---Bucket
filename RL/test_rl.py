# RL/test_rl.py

from pathlib import Path
import time

import numpy as np
import networkx as nx

from .environment import RoutingEnv
from .ppo_agent import build_ppo
from .reward import calculate_reward
from .state import State

from Pathfinding.heuristics import lnd_cost


# ==========================================================
# CONFIG
# ==========================================================

CONFIG = {

    "graph": {
        "neighborhood_k": 15,
        "neighborhood_m": 5,
        "max_hops": 12,
        "top_k": 3
    },

    "simulation": {
        "base_reward_scale": 100.0
    },

    "rl": {
        "hidden_layers": [64, 64],
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": 256,
        "device": "cpu",

        "model_dir": "models",
        "log_dir": "logs",
        "checkpoint_dir": "checkpoints",

        "checkpoint_freq": 10000,
        "eval_freq": 5000
    }
}


# ==========================================================
# REAL GML PATH
# ==========================================================

GML_FILENAME = "20190501.gml.geo"


def get_gml_path():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    path = project_root / GML_FILENAME

    if not path.exists():

        raise FileNotFoundError(
            "\nReal Lightning snapshot was not found:\n"
            f"{path}\n\n"
            f"Expected file: {GML_FILENAME}\n"
            "Place the real GML snapshot in the repository root."
        )

    return path


# ==========================================================
# LOAD REAL GRAPH
# ==========================================================

def load_real_graph():

    gml_path = get_gml_path()

    print(
        "\n=========================================="
    )

    print(
        " LOAD REAL GML SNAPSHOT"
    )

    print(
        "=========================================="
    )

    print(
        "Snapshot:",
        GML_FILENAME
    )

    print(
        "Loading:"
    )

    print(
        gml_path
    )

    G = nx.read_gml(
        gml_path,
        label=None
    )

    print(
        "\nGraph loaded."
    )

    print(
        "Original graph type:",
        type(G).__name__
    )

    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Edges:",
        G.number_of_edges()
    )

    return G


# ==========================================================
# GRAPH PREPARATION
# ==========================================================

def prepare_graph(G):

    """
    Convert the real GML graph to the graph type expected
    by the routing modules.
    """

    if isinstance(
        G,
        nx.MultiDiGraph
    ):

        return G

    if isinstance(
        G,
        nx.DiGraph
    ):

        return nx.MultiDiGraph(G)

    if isinstance(
        G,
        nx.MultiGraph
    ):

        return nx.MultiDiGraph(G)

    if isinstance(
        G,
        nx.Graph
    ):

        return nx.MultiDiGraph(G)

    raise TypeError(
        f"Unsupported graph type: {type(G)}"
    )


# ==========================================================
# REAL TRANSACTION
# ==========================================================

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


# ==========================================================
# BUILD REAL TRANSACTIONS
# ==========================================================

def build_real_transactions(
    G,
    count=20,
    amount=1000
):

    nodes = list(
        G.nodes()
    )

    if len(nodes) < 2:

        raise RuntimeError(
            "The real GML graph does not contain enough nodes."
        )

    transactions = []

    max_hops = int(
        CONFIG["graph"]["max_hops"]
    )

    # ------------------------------------------------------
    # Select connected source/destination pairs
    # ------------------------------------------------------

    for source in nodes:

        if len(transactions) >= count:
            break

        try:

            reachable = (
                nx.single_source_shortest_path_length(
                    G,
                    source,
                    cutoff=max_hops
                )
            )

        except Exception:

            continue

        for destination, distance in reachable.items():

            if source == destination:
                continue

            if distance < 1:
                continue

            transactions.append(
                Transaction(
                    source,
                    destination,
                    amount
                )
            )

            if len(transactions) >= count:
                break

    if not transactions:

        raise RuntimeError(
            f"No valid source/destination pairs were found "
            f"in {GML_FILENAME}."
        )

    print(
        "\nTransactions generated:",
        len(transactions)
    )

    return transactions


# ==========================================================
# TEST STATE ON REAL GRAPH
# ==========================================================

def test_state(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " TEST STATE ON REAL GRAPH"
    )

    print(
        "=========================================="
    )

    if not transactions:

        raise RuntimeError(
            "No transactions available for state testing."
        )

    tx = transactions[0]

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # State.__init__ accepts:
    #
    #     k
    #     radius
    #
    # It does NOT accept:
    #
    #     m
    #
    # neighborhood_m is therefore mapped to radius.
    # ------------------------------------------------------

    state = State(

        G,

        tx.source,

        tx.destination,

        transaction={
            "source":
                tx.source,

            "destination":
                tx.destination,

            "amount":
                tx.amount
        },

        candidate_paths=[],

        simulation_info={
            "failure_probability":
                0.0,

            "average_delay":
                0.0
        },

        bucket_info={
            "bucket_size":
                0,

            "backtrack_count":
                0
        },

        k=int(
            CONFIG["graph"]["neighborhood_k"]
        ),

        radius=int(
            CONFIG["graph"]["neighborhood_m"]
        )
    )

    vector = state.vector()

    print(
        "Source:",
        tx.source
    )

    print(
        "Destination:",
        tx.destination
    )

    print(
        "State K:",
        CONFIG["graph"]["neighborhood_k"]
    )

    print(
        "State radius:",
        CONFIG["graph"]["neighborhood_m"]
    )

    print(
        "State dimension:",
        state.dimension
    )

    print(
        "Vector shape:",
        vector.shape
    )

    print(
        "Expected dimension:",
        CONFIG["graph"]["neighborhood_k"] * 11
        +
        CONFIG["graph"]["neighborhood_k"]
    )

    assert isinstance(
        vector,
        np.ndarray
    )

    assert vector.shape == (
        state.dimension,
    )

    assert vector.shape == (
        state.dimension,
    )

    assert np.all(
        np.isfinite(vector)
    )

    print(
        "STATE TEST PASSED"
    )


# ==========================================================
# TEST REWARD
# ==========================================================

def test_reward():

    print(
        "\n=========================================="
    )

    print(
        " TEST REWARD"
    )

    print(
        "=========================================="
    )

    success_reward = calculate_reward(

        success=True,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        scale=CONFIG[
            "simulation"
        ][
            "base_reward_scale"
        ]
    )

    failure_reward = calculate_reward(

        success=False,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        scale=CONFIG[
            "simulation"
        ][
            "base_reward_scale"
        ]
    )

    print(
        "Success reward:",
        success_reward
    )

    print(
        "Failure reward:",
        failure_reward
    )

    assert np.isfinite(
        success_reward
    )

    assert np.isfinite(
        failure_reward
    )

    assert -1.0 <= success_reward <= 1.0

    assert -1.0 <= failure_reward <= 1.0

    assert success_reward > failure_reward

    print(
        "REWARD TEST PASSED"
    )


# ==========================================================
# TEST ENVIRONMENT
# ==========================================================

def test_environment(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " TEST REAL ENVIRONMENT"
    )

    print(
        "=========================================="
    )

    env = RoutingEnv(

        G=G,

        transactions=transactions,

        heuristic_fn=lnd_cost,

        config=CONFIG,

        mode="test"
    )

    try:

        observation, info = env.reset(
            seed=42
        )

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "Observation space:",
            env.observation_space
        )

        print(
            "Action space:",
            env.action_space
        )

        assert (
            observation.shape
            ==
            env.observation_space.shape
        )

        assert np.all(
            np.isfinite(observation)
        )

        assert isinstance(
            info,
            dict
        )

        # --------------------------------------------------
        # Test PPO-style eta
        # --------------------------------------------------

        action = np.array(
            [0.5],
            dtype=np.float32
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action
        )

        print(
            "\nReward:",
            reward
        )

        print(
            "Terminated:",
            terminated
        )

        print(
            "Truncated:",
            truncated
        )

        print(
            "Route success:",
            info.get(
                "success"
            )
        )

        print(
            "Path length:",
            info.get(
                "path_length"
            )
        )

        print(
            "Fee:",
            info.get(
                "fee"
            )
        )

        print(
            "Delay:",
            info.get(
                "delay"
            )
        )

        assert np.isfinite(
            reward
        )

        assert next_observation.shape == (
            env.observation_space.shape
        )

        assert np.all(
            np.isfinite(next_observation)
        )

        assert isinstance(
            info,
            dict
        )

    finally:

        env.close()

    print(
        "ENVIRONMENT TEST PASSED"
    )


# ==========================================================
# TEST PPO ON REAL GRAPH
# ==========================================================

def test_ppo(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " TEST PPO ON REAL GRAPH"
    )

    print(
        "=========================================="
    )

    env = RoutingEnv(

        G=G,

        transactions=transactions,

        heuristic_fn=lnd_cost,

        config=CONFIG,

        mode="train"
    )

    try:

        model = build_ppo(

            env=env,

            cfg=CONFIG,

            seed=42
        )

        observation, info = env.reset(
            seed=42
        )

        action, _ = model.predict(

            observation,

            deterministic=True
        )

        print(
            "Observation shape:",
            observation.shape
        )

        print(
            "PPO action:",
            action
        )

        print(
            "Action shape:",
            action.shape
        )

        assert action.shape == (
            1,
        )

        assert np.all(
            np.isfinite(action)
        )

        # --------------------------------------------------
        # Check action bounds
        # --------------------------------------------------

        assert np.all(
            action >= env.action_space.low
        )

        assert np.all(
            action <= env.action_space.high
        )

    finally:

        env.close()

    print(
        "PPO TEST PASSED"
    )


# ==========================================================
# END-TO-END REAL GRAPH TEST
# ==========================================================

def test_real_graph_rl(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " REAL GRAPH END-TO-END RL TEST"
    )

    print(
        "=========================================="
    )

    env = RoutingEnv(

        G=G,

        transactions=transactions,

        heuristic_fn=lnd_cost,

        config=CONFIG,

        mode="test"
    )

    model = build_ppo(

        env=env,

        cfg=CONFIG,

        seed=42
    )

    try:

        observation, _ = env.reset(
            seed=42
        )

        total = 0

        successful = 0

        path_lengths = []
        fees = []
        delays = []
        rewards = []
        inference_times = []

        terminated = False
        truncated = False

        # --------------------------------------------------
        # Run until episode termination
        # --------------------------------------------------

        while not terminated and not truncated:

            start = time.perf_counter()

            action, _ = model.predict(

                observation,

                deterministic=True
            )

            inference_time = (
                time.perf_counter()
                -
                start
            )

            inference_times.append(
                inference_time
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )

            total += 1

            rewards.append(
                float(reward)
            )

            if info.get(
                "success",
                False
            ):

                successful += 1

            path_lengths.append(
                float(
                    info.get(
                        "path_length",
                        0
                    )
                )
            )

            fees.append(
                float(
                    info.get(
                        "fee",
                        0.0
                    )
                )
            )

            delays.append(
                float(
                    info.get(
                        "delay",
                        0.0
                    )
                )

            )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        success_rate = (
            successful
            /
            max(total, 1)
        )

        print(
            "\n=========================================="
        )

        print(
            " REAL GRAPH RESULTS"
        )

        print(
            "=========================================="
        )

        print(
            "Snapshot:",
            GML_FILENAME
        )

        print(
            "Nodes:",
            G.number_of_nodes()
        )

        print(
            "Edges:",
            G.number_of_edges()
        )

        print(
            "Transactions:",
            total
        )

        print(
            "Successful:",
            successful
        )

        print(
            "Payment Success Rate:",
            success_rate
        )

        print(
            "Average Path Length:",
            np.mean(path_lengths)
            if path_lengths
            else 0.0
        )

        print(
            "Average Fee:",
            np.mean(fees)
            if fees
            else 0.0
        )

        print(
            "Average Delay:",
            np.mean(delays)
            if delays
            else 0.0
        )

        print(
            "Average Reward:",
            np.mean(rewards)
            if rewards
            else 0.0
        )

        print(
            "Average PPO Inference Time:",
            np.mean(inference_times)
            if inference_times
            else 0.0
        )

        return {

            "snapshot":
                GML_FILENAME,

            "nodes":
                G.number_of_nodes(),

            "edges":
                G.number_of_edges(),

            "transactions":
                total,

            "successful":
                successful,

            "success_rate":
                float(
                    success_rate
                ),

            "average_path_length":
                float(
                    np.mean(path_lengths)
                    if path_lengths
                    else 0.0
                ),

            "average_fee":
                float(
                    np.mean(fees)
                    if fees
                    else 0.0
                ),

            "average_delay":
                float(
                    np.mean(delays)
                    if delays
                    else 0.0
                ),

            "average_reward":
                float(
                    np.mean(rewards)
                    if rewards
                    else 0.0
                ),

            "average_inference_time":
                float(
                    np.mean(inference_times)
                    if inference_times
                    else 0.0
                )
        }

    finally:

        env.close()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        "\n"
        "======================================================"
    )

    print(
        " REAL LIGHTNING GML RL TEST"
    )

    print(
        f" Snapshot: {GML_FILENAME}"
    )

    print(
        "======================================================"
    )

    # ------------------------------------------------------
    # 1. Load REAL GML
    # ------------------------------------------------------

    G = load_real_graph()

    # ------------------------------------------------------
    # 2. Convert to expected graph type
    # ------------------------------------------------------

    G = prepare_graph(G)

    print(
        "\nPrepared graph type:",
        type(G).__name__
    )

    print(
        "Prepared nodes:",
        G.number_of_nodes()
    )

    print(
        "Prepared edges:",
        G.number_of_edges()
    )

    # ------------------------------------------------------
    # 3. Build transactions from REAL graph
    # ------------------------------------------------------

    transactions = build_real_transactions(

        G,

        count=20,

        amount=1000
    )

    # ------------------------------------------------------
    # 4. State
    # ------------------------------------------------------

    test_state(
        G,
        transactions
    )

    # ------------------------------------------------------
    # 5. Reward
    # ------------------------------------------------------

    test_reward()

    # ------------------------------------------------------
    # 6. Environment
    # ------------------------------------------------------

    test_environment(
        G,
        transactions
    )

    # ------------------------------------------------------
    # 7. PPO
    # ------------------------------------------------------

    test_ppo(
        G,
        transactions
    )

    # ------------------------------------------------------
    # 8. End-to-end
    # ------------------------------------------------------

    results = test_real_graph_rl(

        G,

        transactions
    )

    print(
        "\n"
        "======================================================"
    )

    print(
        " REAL GML RL TEST COMPLETED"
    )

    print(
        "======================================================"
    )