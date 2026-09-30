# RL/test_rl.py

from pathlib import Path
import copy
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

        # IMPORTANT:
        # Top-K is FIXED.
        # PPO does NOT control k.
        "top_k": 5,

        "lambda_h": 1.0
    },

    "simulation": {

        "base_reward_scale": 100.0,

        "node_failure_probability": 0.01,

        "liquidity_failure_probability": 0.05,

        "channel_failure_rate": 0.01,

        "node_failure_rate": 0.005,

        "recovery_rate": 0.05,

        "seed": 42
    },

    "reward": {

        "scale": 100.0,

        "fee_reference": 1000.0,

        "delay_reference": 10.0,

        "carbon_reference": 100.0,

        "path_reference": 10.0
    },

    "rl": {

        # --------------------------------------------------
        # PPO controls ETA only.
        # --------------------------------------------------

        "eta_min": 0.0,

        "eta_max": 1.0,

        "hidden_layers": [
            64,
            64
        ],

        "learning_rate": 3e-4,

        "n_steps": 128,

        "batch_size": 32,

        "n_epochs": 10,

        "gamma": 0.99,

        "gae_lambda": 0.95,

        "clip_range": 0.2,

        "ent_coef": 0.01,

        # --------------------------------------------------
        # Training budget
        # --------------------------------------------------

        "total_timesteps": 256,

        "device": "cpu",

        # --------------------------------------------------
        # Files
        # --------------------------------------------------

        "model_dir": "models",

        "log_dir": "logs",

        "checkpoint_dir": "checkpoints",

        "checkpoint_freq": 10000,

        "eval_freq": 5000,

        # --------------------------------------------------
        # If True:
        # load models/ppo_lightning_eta.zip
        # instead of training again.
        # --------------------------------------------------

        "reuse_saved_model": False
    },

    # ======================================================
    # BASELINE CONFIGURATION
    # ======================================================

    "baseline": {

        # Fixed ETA baselines.
        #
        # IMPORTANT:
        # k remains fixed at 5.
        "eta_values": [
            0.0,
            0.5
        ],

        # Print every transaction for baselines.
        "verbose": True
    }
}


# ==========================================================
# CONSTANTS
# ==========================================================

GML_FILENAME = "20190501.gml.geo"

SEED = 42

TRANSACTION_COUNT = 20

TRANSACTION_AMOUNT = 1000

MAX_AMOUNT = 1_000_000


# ==========================================================
# REAL GML PATH
# ==========================================================

def get_gml_path():

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    candidates = [

        project_root / GML_FILENAME,

        project_root / "Data" / GML_FILENAME,

        project_root / "data" / GML_FILENAME
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(

        "\nReal Lightning snapshot was not found.\n"

        f"Expected file: {GML_FILENAME}\n\n"

        "Searched locations:\n"
        +
        "\n".join(
            f"  {path}"
            for path in candidates
        )
    )


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

    print(
        "\n=========================================="
    )

    print(
        " PREPARE GRAPH"
    )

    print(
        "=========================================="
    )

    # ------------------------------------------------------
    # Already MultiDiGraph
    # ------------------------------------------------------

    if isinstance(
        G,
        nx.MultiDiGraph
    ):

        DG = G.copy()

        print(
            "Graph already is MultiDiGraph."
        )

    # ------------------------------------------------------
    # Directed graph
    # ------------------------------------------------------

    elif isinstance(
        G,
        nx.DiGraph
    ):

        DG = nx.MultiDiGraph()

        for node, attrs in G.nodes(
            data=True
        ):

            DG.add_node(
                node,
                **dict(attrs)
            )

        for u, v, attrs in G.edges(
            data=True
        ):

            DG.add_edge(
                u,
                v,
                **dict(attrs)
            )

        print(
            "Directed graph converted to MultiDiGraph."
        )

    # ------------------------------------------------------
    # Undirected MultiGraph
    # ------------------------------------------------------

    elif isinstance(
        G,
        nx.MultiGraph
    ):

        DG = nx.MultiDiGraph()

        for node, attrs in G.nodes(
            data=True
        ):

            DG.add_node(
                node,
                **dict(attrs)
            )

        for u, v, key, attrs in G.edges(
            keys=True,
            data=True
        ):

            attrs_uv = dict(attrs)

            attrs_vu = dict(attrs)

            DG.add_edge(
                u,
                v,
                key=key,
                **attrs_uv
            )

            DG.add_edge(
                v,
                u,
                key=key,
                **attrs_vu
            )

        print(
            "Undirected MultiGraph converted "
            "to directed MultiDiGraph."
        )

    # ------------------------------------------------------
    # Undirected Graph
    # ------------------------------------------------------

    elif isinstance(
        G,
        nx.Graph
    ):

        DG = nx.MultiDiGraph()

        for node, attrs in G.nodes(
            data=True
        ):

            DG.add_node(
                node,
                **dict(attrs)
            )

        for u, v, attrs in G.edges(
            data=True
        ):

            attrs_uv = dict(attrs)

            attrs_vu = dict(attrs)

            DG.add_edge(
                u,
                v,
                **attrs_uv
            )

            DG.add_edge(
                v,
                u,
                **attrs_vu
            )

        print(
            "Undirected Graph converted "
            "to directed MultiDiGraph."
        )

    else:

        raise TypeError(
            f"Unsupported graph type: {type(G)}"
        )

    # ------------------------------------------------------
    # Prepare channel attributes
    # ------------------------------------------------------

    prepare_channel_attributes(
        DG
    )

    print()

    print(
        "Prepared graph type:",
        type(DG).__name__
    )

    print(
        "Prepared nodes:",
        DG.number_of_nodes()
    )

    print(
        "Prepared directed edges:",
        DG.number_of_edges()
    )

    return DG


# ==========================================================
# CHANNEL ATTRIBUTES
# ==========================================================

def prepare_channel_attributes(G):

    edge_count = 0

    for u, v, key, data in G.edges(
        keys=True,
        data=True
    ):

        # --------------------------------------------------
        # Capacity
        # --------------------------------------------------

        if "capacity" not in data:

            if "capacity_sat" in data:

                try:

                    data["capacity"] = float(
                        data["capacity_sat"]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    data["capacity"] = float(
                        MAX_AMOUNT
                    )

            else:

                data["capacity"] = float(
                    MAX_AMOUNT
                )

        try:

            data["capacity"] = float(
                data["capacity"]
            )

        except (
            TypeError,
            ValueError
        ):

            data["capacity"] = float(
                MAX_AMOUNT
            )

        if data["capacity"] <= 0:

            data["capacity"] = float(
                MAX_AMOUNT
            )

        # --------------------------------------------------
        # Fees
        # --------------------------------------------------

        try:

            data["fee_base_msat"] = float(
                data.get(
                    "fee_base_msat",
                    1000.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            data["fee_base_msat"] = 1000.0

        try:

            data[
                "fee_proportional_millionths"
            ] = float(
                data.get(
                    "fee_proportional_millionths",
                    1.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            data[
                "fee_proportional_millionths"
            ] = 1.0

        # --------------------------------------------------
        # Delay
        # --------------------------------------------------

        try:

            data["delay"] = float(
                data.get(
                    "delay",
                    1.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            data["delay"] = 1.0

        # --------------------------------------------------
        # Directional balances
        # --------------------------------------------------

        if "balance_uv" not in data:

            data["balance_uv"] = (
                data["capacity"] / 2.0
            )

        if "balance_vu" not in data:

            data["balance_vu"] = (
                data["capacity"] / 2.0
            )

        try:

            data["balance_uv"] = float(
                data["balance_uv"]
            )

        except (
            TypeError,
            ValueError
        ):

            data["balance_uv"] = (
                data["capacity"] / 2.0
            )

        try:

            data["balance_vu"] = float(
                data["balance_vu"]
            )

        except (
            TypeError,
            ValueError
        ):

            data["balance_vu"] = (
                data["capacity"] / 2.0
            )

        # --------------------------------------------------
        # Runtime status
        # --------------------------------------------------

        data.setdefault(
            "available",
            True
        )

        data.setdefault(
            "failure_count",
            0
        )

        data.setdefault(
            "success_count",
            0
        )

        edge_count += 1

    print(
        f"Prepared {edge_count} directed channels."
    )

    return G


# ==========================================================
# TRANSACTION
# ==========================================================

class Transaction:

    def __init__(
        self,
        tx_id,
        source,
        destination,
        amount
    ):

        self.tx_id = tx_id

        self.source = source

        self.destination = destination

        self.amount = amount

    def __repr__(self):

        return (
            f"Transaction("
            f"tx_id={self.tx_id}, "
            f"source={self.source}, "
            f"destination={self.destination}, "
            f"amount={self.amount}"
            f")"
        )


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
            "The real GML graph does not contain "
            "enough nodes."
        )

    transactions = []

    max_hops = int(
        CONFIG["graph"]["max_hops"]
    )

    tx_id = 0

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

            tx_id += 1

            transactions.append(

                Transaction(
                    tx_id=tx_id,
                    source=source,
                    destination=destination,
                    amount=amount
                )
            )

            if len(transactions) >= count:

                break

    if not transactions:

        raise RuntimeError(
            f"No valid source/destination pairs "
            f"were found in {GML_FILENAME}."
        )

    print(
        "\nTransactions generated:",
        len(transactions)
    )

    print(
        "\nFirst transactions:"
    )

    for tx in transactions[:5]:

        print(
            f"  TX {tx.tx_id}: "
            f"{tx.source} -> "
            f"{tx.destination} "
            f"| amount={tx.amount}"
        )

    return transactions


# ==========================================================
# COPY GRAPH FOR ISOLATED EXPERIMENT
# ==========================================================

def isolated_graph(G):

    """
    Return an independent graph for one experiment.

    This prevents one experiment from modifying the
    availability, balances, failure counters, or other
    runtime attributes used by another experiment.
    """

    return copy.deepcopy(G)


# ==========================================================
# BUILD ENVIRONMENT
# ==========================================================

def build_environment(
    G,
    transactions,
    mode
):

    # ------------------------------------------------------
    # IMPORTANT:
    #
    # Every environment receives an independent graph.
    # This makes baseline/PPO comparison fair.
    # ------------------------------------------------------

    env_graph = isolated_graph(
        G
    )

    env = RoutingEnv(

        G=env_graph,

        transactions=transactions,

        heuristic_fn=lnd_cost,

        config=CONFIG,

        mode=mode
    )

    # ------------------------------------------------------
    # Fixed Top-K verification
    # ------------------------------------------------------

    if env.top_k != 5:

        env.close()

        raise RuntimeError(
            "Top-K must remain fixed at 5. "
            f"Found: {env.top_k}"
        )

    return env


# ==========================================================
# TEST STATE
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
            "No transactions available "
            "for state testing."
        )

    tx = transactions[0]

    state = State(

        G,

        tx.source,

        tx.destination,

        transaction={

            "tx_id":
                tx.tx_id,

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
                0.0,

            "backtrack_count":
                0
        },

        bucket_info={

            "bucket_size":
                0,

            "backtrack_count":
                0
        },

        k=int(
            CONFIG["graph"][
                "neighborhood_k"
            ]
        ),

        radius=int(
            CONFIG["graph"][
                "neighborhood_m"
            ]
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
        CONFIG["graph"][
            "neighborhood_k"
        ]
    )

    print(
        "State radius:",
        CONFIG["graph"][
            "neighborhood_m"
        ]
    )

    print(
        "State dimension:",
        state.dimension
    )

    print(
        "Vector shape:",
        vector.shape
    )

    expected_dimension = (

        CONFIG["graph"]["neighborhood_k"]
        *
        11
        +
        CONFIG["graph"]["neighborhood_k"]
    )

    print(
        "Expected dimension:",
        expected_dimension
    )

    assert isinstance(
        vector,
        np.ndarray
    )

    assert vector.shape == (
        state.dimension,
    )

    assert vector.shape == (
        expected_dimension,
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

    reward_cfg = CONFIG["reward"]

    success_reward = calculate_reward(

        success=True,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        partial_backtrack_count=0,

        full_reroute_count=0,

        attempt_count=1,

        scale=reward_cfg["scale"],

        fee_reference=
            reward_cfg["fee_reference"],

        delay_reference=
            reward_cfg["delay_reference"],

        carbon_reference=
            reward_cfg["carbon_reference"],

        path_reference=
            reward_cfg["path_reference"]
    )

    recovered_reward = calculate_reward(

        success=True,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        partial_backtrack_count=1,

        full_reroute_count=0,

        attempt_count=2,

        scale=reward_cfg["scale"],

        fee_reference=
            reward_cfg["fee_reference"],

        delay_reference=
            reward_cfg["delay_reference"],

        carbon_reference=
            reward_cfg["carbon_reference"],

        path_reference=
            reward_cfg["path_reference"]
    )

    reroute_reward = calculate_reward(

        success=True,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        partial_backtrack_count=0,

        full_reroute_count=1,

        attempt_count=2,

        scale=reward_cfg["scale"],

        fee_reference=
            reward_cfg["fee_reference"],

        delay_reference=
            reward_cfg["delay_reference"],

        carbon_reference=
            reward_cfg["carbon_reference"],

        path_reference=
            reward_cfg["path_reference"]
    )

    failure_reward = calculate_reward(

        success=False,

        path_length=5,

        carbon_intensity=50,

        fee=100,

        delay=20,

        partial_backtrack_count=0,

        full_reroute_count=0,

        attempt_count=1,

        scale=reward_cfg["scale"],

        fee_reference=
            reward_cfg["fee_reference"],

        delay_reference=
            reward_cfg["delay_reference"],

        carbon_reference=
            reward_cfg["carbon_reference"],

        path_reference=
            reward_cfg["path_reference"]
    )

    print(
        "Normal success reward:",
        success_reward
    )

    print(
        "Backtracking success reward:",
        recovered_reward
    )

    print(
        "Full reroute success reward:",
        reroute_reward
    )

    print(
        "Failure reward:",
        failure_reward
    )

    assert np.isfinite(
        success_reward
    )

    assert np.isfinite(
        recovered_reward
    )

    assert np.isfinite(
        reroute_reward
    )

    assert np.isfinite(
        failure_reward
    )

    assert -1.0 <= success_reward <= 1.0

    assert -1.0 <= recovered_reward <= 1.0

    assert -1.0 <= reroute_reward <= 1.0

    assert -1.0 <= failure_reward <= 1.0

    assert success_reward > recovered_reward

    assert recovered_reward > failure_reward

    assert success_reward > reroute_reward

    assert reroute_reward > failure_reward

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

    env = build_environment(

        G,

        transactions,

        mode="test"
    )

    try:

        observation, info = env.reset(
            seed=SEED
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

        print(
            "Fixed Top-K:",
            env.top_k
        )

        assert (
            observation.shape
            ==
            env.observation_space.shape
        )

        assert np.all(
            np.isfinite(
                observation
            )
        )

        assert isinstance(
            info,
            dict
        )

        # --------------------------------------------------
        # Test fixed ETA action
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
            "\nTest action:",
            action
        )

        print(
            "Eta used:",
            info.get(
                "eta"
            )
        )

        print(
            "Fixed Top-K:",
            info.get(
                "top_k"
            )
        )

        print(
            "Candidate count:",
            info.get(
                "candidate_path_count"
            )
        )

        print(
            "Bucket size:",
            info.get(
                "bucket_size"
            )
        )

        print(
            "Reward:",
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

        print(
            "Backtrack count:",
            info.get(
                "backtrack_count"
            )
        )

        print(
            "Full reroute:",
            info.get(
                "full_reroute"
            )
        )

        assert np.isfinite(
            reward
        )

        assert (
            next_observation.shape
            ==
            env.observation_space.shape
        )

        assert np.all(
            np.isfinite(
                next_observation
            )
        )

        assert isinstance(
            info,
            dict
        )

        assert env.top_k == 5

        assert info.get(
            "top_k"
        ) == 5

        assert np.isclose(
            float(
                info.get(
                    "eta"
                )
            ),
            0.5,
            atol=1e-6
        )

    finally:

        env.close()

    print(
        "ENVIRONMENT TEST PASSED"
    )


# ==========================================================
# TEST PPO CONSTRUCTION
# ==========================================================

def test_ppo(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " TEST PPO CONSTRUCTION"
    )

    print(
        "=========================================="
    )

    env = build_environment(

        G,

        transactions,

        mode="train"
    )

    try:

        model = build_ppo(

            env=env,

            cfg=CONFIG,

            seed=SEED
        )

        observation, info = env.reset(
            seed=SEED
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
            np.isfinite(
                action
            )
        )

        assert np.all(
            action >= env.action_space.low
        )

        assert np.all(
            action <= env.action_space.high
        )

        print(
            "Initial PPO eta:",
            float(
                action[0]
            )
        )

    finally:

        env.close()

    print(
        "PPO CONSTRUCTION TEST PASSED"
    )


# ==========================================================
# TRAIN PPO
# ==========================================================

def train_ppo(
    G,
    transactions
):

    print(
        "\n=========================================="
    )

    print(
        " PPO TRAINING ON REAL LIGHTNING GRAPH"
    )

    print(
        "=========================================="
    )

    # ------------------------------------------------------
    # Reuse existing trained model if requested.
    # ------------------------------------------------------

    model_path = (

        Path(
            CONFIG["rl"]["model_dir"]
        )
        /
        "ppo_lightning_eta"
    )

    if CONFIG["rl"].get(
        "reuse_saved_model",
        False
    ):

        zip_path = Path(
            str(model_path) + ".zip"
        )

        if not zip_path.exists():

            raise FileNotFoundError(

                "reuse_saved_model=True, but "
                "the trained PPO model was not found:\n"
                f"{zip_path}"
            )

        print(
            "\nLoading existing PPO model:"
        )

        print(
            zip_path
        )

        env = build_environment(

            G,

            transactions,

            mode="test"
        )

        try:

            from stable_baselines3 import PPO

            model = PPO.load(

                str(model_path),

                env=env,

                device=CONFIG["rl"]["device"]
            )

            print(
                "Existing PPO model loaded."
            )

            return model, 0.0

        except Exception:

            env.close()

            raise

    # ------------------------------------------------------
    # Normal training
    # ------------------------------------------------------

    env = build_environment(

        G,

        transactions,

        mode="train"
    )

    try:

        model = build_ppo(

            env=env,

            cfg=CONFIG,

            seed=SEED
        )

        total_timesteps = int(
            CONFIG["rl"]["total_timesteps"]
        )

        if total_timesteps <= 0:

            raise ValueError(
                "total_timesteps must be greater than zero."
            )

        print(
            "\nPPO configuration:"
        )

        print(
            "  State dimension:",
            env.observation_space.shape
        )

        print(
            "  Action space:",
            env.action_space
        )

        print(
            "  Fixed Top-K:",
            env.top_k
        )

        print(
            "  PPO controls:",
            "eta only"
        )

        print(
            "  Total timesteps:",
            total_timesteps
        )

        print(
            "  Device:",
            CONFIG["rl"]["device"]
        )

        print(
            "\nStarting PPO learning..."
        )

        start_time = time.perf_counter()

        model.learn(

            total_timesteps=total_timesteps,

            reset_num_timesteps=True,

            progress_bar=False
        )

        training_time = (

            time.perf_counter()
            -
            start_time
        )

        print(
            "\nPPO training completed."
        )

        print(
            "Training time:",
            training_time,
            "seconds"
        )

        # --------------------------------------------------
        # Save trained model
        # --------------------------------------------------

        model_dir = Path(
            CONFIG["rl"]["model_dir"]
        )

        model_dir.mkdir(

            parents=True,

            exist_ok=True
        )

        model.save(
            str(model_path)
        )

        print(
            "Trained PPO model saved to:",
            model_path
        )

        return model, training_time

    finally:

        env.close()


# ==========================================================
# GENERIC EVALUATION RESULT
# ==========================================================

def _empty_metrics():

    return {

        "transactions":
            0,

        "successful":
            0,

        "success_rate":
            0.0,

        "average_eta":
            0.0,

        "average_path_length":
            0.0,

        "average_fee":
            0.0,

        "average_delay":
            0.0,

        "average_backtrack_count":
            0.0,

        "full_reroute_events":
            0,

        "average_reward":
            0.0,

        "average_inference_time":
            0.0
    }


# ==========================================================
# EVALUATE FIXED ETA BASELINE
# ==========================================================

def evaluate_fixed_eta(
    G,
    transactions,
    eta,
    label=None,
    verbose=True
):

    eta = float(eta)

    eta_min = float(
        CONFIG["rl"]["eta_min"]
    )

    eta_max = float(
        CONFIG["rl"]["eta_max"]
    )

    if not (
        eta_min
        <=
        eta
        <=
        eta_max
    ):

        raise ValueError(

            f"ETA must be in "
            f"[{eta_min}, {eta_max}], "
            f"got {eta}"
        )

    if label is None:

        label = (
            f"FIXED ETA BASELINE "
            f"(eta={eta:.3f})"
        )

    print(
        "\n=========================================="
    )

    print(
        f" {label}"
    )

    print(
        "=========================================="
    )

    env = build_environment(

        G,

        transactions,

        mode="test"
    )

    try:

        observation, _ = env.reset(
            seed=SEED
        )

        total = 0

        successful = 0

        path_lengths = []

        fees = []

        delays = []

        rewards = []

        eta_values = []

        backtrack_counts = []

        full_reroutes = 0

        # --------------------------------------------------
        # Evaluation loop
        # --------------------------------------------------

        terminated = False

        truncated = False

        while not terminated and not truncated:

            action = np.array(

                [eta],

                dtype=np.float32
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

            eta_used = info.get(
                "eta",
                eta
            )

            eta_values.append(
                float(eta_used)
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
                        0.0
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

            backtrack_counts.append(

                float(
                    info.get(
                        "backtrack_count",
                        0.0
                    )
                )
            )

            if info.get(
                "full_reroute",
                False
            ):

                full_reroutes += 1

            # ------------------------------------------------
            # Per transaction
            # ------------------------------------------------

            if verbose:

                print(
                    f"\nTX "
                    f"{info.get('transaction_id')}:"
                )

                print(
                    f"  Fixed eta     : "
                    f"{info.get('eta')}"
                )

                print(
                    f"  Top-K         : "
                    f"{info.get('top_k')}"
                )

                print(
                    f"  Candidates    : "
                    f"{info.get('candidate_path_count')}"
                )

                print(
                    f"  Bucket size   : "
                    f"{info.get('bucket_size')}"
                )

                print(
                    f"  Success       : "
                    f"{info.get('success')}"
                )

                print(
                    f"  Path length   : "
                    f"{info.get('path_length')}"
                )

                print(
                    f"  Backtracking  : "
                    f"{info.get('backtrack_count')}"
                )

                print(
                    f"  Full reroute  : "
                    f"{info.get('full_reroute')}"
                )

                print(
                    f"  Fee           : "
                    f"{info.get('fee')}"
                )

                print(
                    f"  Delay         : "
                    f"{info.get('delay')}"
                )

                print(
                    f"  Reward        : "
                    f"{reward}"
                )

        # --------------------------------------------------
        # Aggregate
        # --------------------------------------------------

        success_rate = (

            successful
            /
            max(
                total,
                1
            )
        )

        results = {

            "method":
                label,

            "transactions":
                total,

            "successful":
                successful,

            "success_rate":
                float(
                    success_rate
                ),

            "fixed_top_k":
                int(
                    env.top_k
                ),

            "average_eta":
                float(
                    np.mean(eta_values)
                    if eta_values
                    else eta
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

            "average_backtrack_count":
                float(
                    np.mean(
                        backtrack_counts
                    )
                    if backtrack_counts
                    else 0.0
                ),

            "full_reroute_events":
                int(
                    full_reroutes
                ),

            "average_reward":
                float(
                    np.mean(rewards)
                    if rewards
                    else 0.0
                ),

            "average_inference_time":
                0.0
        }

        # --------------------------------------------------
        # Final baseline results
        # --------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            f" {label} RESULTS"
        )

        print(
            "=========================================="
        )

        print(
            "Transactions:",
            results["transactions"]
        )

        print(
            "Successful:",
            results["successful"]
        )

        print(
            "Payment Success Rate:",
            results["success_rate"]
        )

        print(
            "Fixed Top-K:",
            results["fixed_top_k"]
        )

        print(
            "Average ETA:",
            results["average_eta"]
        )

        print(
            "Average Path Length:",
            results[
                "average_path_length"
            ]
        )

        print(
            "Average Fee:",
            results[
                "average_fee"
            ]
        )

        print(
            "Average Delay:",
            results[
                "average_delay"
            ]
        )

        print(
            "Average Backtrack Count:",
            results[
                "average_backtrack_count"
            ]
        )

        print(
            "Full Reroute Events:",
            results[
                "full_reroute_events"
            ]
        )

        print(
            "Average Reward:",
            results[
                "average_reward"
            ]
        )

        return results

    finally:

        env.close()


# ==========================================================
# EVALUATE TRAINED PPO
# ==========================================================

def evaluate_ppo(
    G,
    transactions,
    model,
    verbose=True
):

    print(
        "\n=========================================="
    )

    print(
        " EVALUATE TRAINED PPO"
    )

    print(
        "=========================================="
    )

    env = build_environment(

        G,

        transactions,

        mode="test"
    )

    try:

        observation, _ = env.reset(
            seed=SEED
        )

        total = 0

        successful = 0

        path_lengths = []

        fees = []

        delays = []

        rewards = []

        inference_times = []

        eta_values = []

        backtrack_counts = []

        full_reroutes = 0

        terminated = False

        truncated = False

        # --------------------------------------------------
        # Evaluation loop
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

            eta = info.get(
                "eta"
            )

            if eta is not None:

                eta_values.append(
                    float(eta)
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
                        0.0
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

            backtrack_counts.append(

                float(
                    info.get(
                        "backtrack_count",
                        0.0
                    )
                )
            )

            if info.get(
                "full_reroute",
                False
            ):

                full_reroutes += 1

            # ------------------------------------------------
            # Per transaction
            # ------------------------------------------------

            if verbose:

                print(
                    f"\nTX "
                    f"{info.get('transaction_id')}:"
                )

                print(
                    f"  PPO eta       : "
                    f"{info.get('eta')}"
                )

                print(
                    f"  Top-K         : "
                    f"{info.get('top_k')}"
                )

                print(
                    f"  Candidates    : "
                    f"{info.get('candidate_path_count')}"
                )

                print(
                    f"  Bucket size   : "
                    f"{info.get('bucket_size')}"
                )

                print(
                    f"  Success       : "
                    f"{info.get('success')}"
                )

                print(
                    f"  Path length   : "
                    f"{info.get('path_length')}"
                )

                print(
                    f"  Backtracking  : "
                    f"{info.get('backtrack_count')}"
                )

                print(
                    f"  Full reroute  : "
                    f"{info.get('full_reroute')}"
                )

                print(
                    f"  Fee           : "
                    f"{info.get('fee')}"
                )

                print(
                    f"  Delay         : "
                    f"{info.get('delay')}"
                )

                print(
                    f"  Reward        : "
                    f"{reward}"
                )

        # --------------------------------------------------
        # Aggregate
        # --------------------------------------------------

        success_rate = (

            successful
            /
            max(
                total,
                1
            )
        )

        results = {

            "method":
                "Trained PPO",

            "transactions":
                total,

            "successful":
                successful,

            "success_rate":
                float(
                    success_rate
                ),

            "fixed_top_k":
                int(
                    env.top_k
                ),

            "average_eta":
                float(
                    np.mean(eta_values)
                    if eta_values
                    else 0.0
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

            "average_backtrack_count":
                float(
                    np.mean(
                        backtrack_counts
                    )
                    if backtrack_counts
                    else 0.0
                ),

            "full_reroute_events":
                int(
                    full_reroutes
                ),

            "average_reward":
                float(
                    np.mean(rewards)
                    if rewards
                    else 0.0
                ),

            "average_inference_time":
                float(
                    np.mean(
                        inference_times
                    )
                    if inference_times
                    else 0.0
                )
        }

        # --------------------------------------------------
        # Final PPO results
        # --------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            " TRAINED PPO EVALUATION RESULTS"
        )

        print(
            "=========================================="
        )

        print(
            "Transactions:",
            results["transactions"]
        )

        print(
            "Successful:",
            results["successful"]
        )

        print(
            "Payment Success Rate:",
            results["success_rate"]
        )

        print(
            "Fixed Top-K:",
            results["fixed_top_k"]
        )

        print(
            "Average ETA:",
            results["average_eta"]
        )

        print(
            "Average Path Length:",
            results[
                "average_path_length"
            ]
        )

        print(
            "Average Fee:",
            results[
                "average_fee"
            ]
        )

        print(
            "Average Delay:",
            results[
                "average_delay"
            ]
        )

        print(
            "Average Backtrack Count:",
            results[
                "average_backtrack_count"
            ]
        )

        print(
            "Full Reroute Events:",
            results[
                "full_reroute_events"
            ]
        )

        print(
            "Average Reward:",
            results[
                "average_reward"
            ]
        )

        print(
            "Average PPO Inference Time:",
            results[
                "average_inference_time"
            ]
        )

        return results

    finally:

        env.close()


# ==========================================================
# COMPARE RESULTS
# ==========================================================

def compare_results(
    baseline_0,
    baseline_05,
    ppo_results
):

    print(
        "\n"
        "======================================================"
    )

    print(
        " BASELINE vs PPO COMPARISON"
    )

    print(
        "======================================================"
    )

    rows = [

        baseline_0,

        baseline_05,

        ppo_results
    ]

    headers = [

        "Method",

        "Success Rate",

        "Avg ETA",

        "Avg Path",

        "Avg Fee",

        "Avg Delay",

        "Avg Backtrack",

        "Full Reroute",

        "Avg Reward"
    ]

    print()

    print(
        f"{headers[0]:<24}"
        f"{headers[1]:>15}"
        f"{headers[2]:>12}"
        f"{headers[3]:>12}"
        f"{headers[4]:>15}"
        f"{headers[5]:>14}"
        f"{headers[6]:>17}"
        f"{headers[7]:>15}"
        f"{headers[8]:>14}"
    )

    print(
        "-" * 138
    )

    for result in rows:

        print(

            f"{result.get('method', ''):<24}"

            f"{result.get('success_rate', 0.0):>15.4f}"

            f"{result.get('average_eta', 0.0):>12.4f}"

            f"{result.get('average_path_length', 0.0):>12.4f}"

            f"{result.get('average_fee', 0.0):>15.4f}"

            f"{result.get('average_delay', 0.0):>14.4f}"

            f"{result.get('average_backtrack_count', 0.0):>17.4f}"

            f"{result.get('full_reroute_events', 0):>15}"

            f"{result.get('average_reward', 0.0):>14.4f}"
        )

    print(
        "-" * 138
    )

    print()

    # ------------------------------------------------------
    # Fixed Top-K assertion
    # ------------------------------------------------------

    for result in rows:

        if result.get(
            "fixed_top_k"
        ) != 5:

            raise RuntimeError(

                "Comparison detected a non-fixed "
                "Top-K value."
            )

    print(
        "Top-K verification:"
    )

    print(
        "  Baseline eta=0.0 : 5"
    )

    print(
        "  Baseline eta=0.5 : 5"
    )

    print(
        "  Trained PPO      : 5"
    )

    print()

    print(
        "PPO action interpretation:"
    )

    print(
        "  PPO controls eta only."
    )

    print(
        "  Top-K remains fixed at 5."
    )

    print(
        "  Baselines use fixed eta values."
    )

    print()

    return {

        "baseline_eta_0":
            baseline_0,

        "baseline_eta_05":
            baseline_05,

        "ppo":
            ppo_results
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print(
        "\n"
        "======================================================"
    )

    print(
        " REAL LIGHTNING GML PPO "
        "TRAINING + BASELINE COMPARISON"
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
    # 2. Prepare REAL graph
    # ------------------------------------------------------

    G = prepare_graph(
        G
    )

    print(
        "\nPrepared graph type:",
        type(G).__name__
    )

    print(
        "Prepared nodes:",
        G.number_of_nodes()
    )

    print(
        "Prepared directed edges:",
        G.number_of_edges()
    )

    # ------------------------------------------------------
    # 3. Build REAL transactions
    # ------------------------------------------------------

    transactions = build_real_transactions(

        G,

        count=TRANSACTION_COUNT,

        amount=TRANSACTION_AMOUNT
    )

    # ------------------------------------------------------
    # 4. State test
    # ------------------------------------------------------

    test_state(

        G,

        transactions
    )

    # ------------------------------------------------------
    # 5. Reward test
    # ------------------------------------------------------

    test_reward()

    # ------------------------------------------------------
    # 6. Environment test
    # ------------------------------------------------------

    test_environment(

        G,

        transactions
    )

    # ------------------------------------------------------
    # 7. PPO construction test
    # ------------------------------------------------------

    test_ppo(

        G,

        transactions
    )

    # ------------------------------------------------------
    # 8. TRAIN / LOAD PPO
    # ------------------------------------------------------

    trained_model, training_time = train_ppo(

        G,

        transactions
    )

    # ------------------------------------------------------
    # 9. BASELINE ETA = 0.0
    # ------------------------------------------------------

    baseline_eta_0 = evaluate_fixed_eta(

        G,

        transactions,

        eta=0.0,

        label="BASELINE ETA = 0.0",

        verbose=CONFIG[
            "baseline"
        ][
            "verbose"
        ]
    )

    # ------------------------------------------------------
    # 10. BASELINE ETA = 0.5
    # ------------------------------------------------------

    baseline_eta_05 = evaluate_fixed_eta(

        G,

        transactions,

        eta=0.5,

        label="BASELINE ETA = 0.5",

        verbose=CONFIG[
            "baseline"
        ][
            "verbose"
        ]
    )

    # ------------------------------------------------------
    # 11. TRAINED PPO
    # ------------------------------------------------------

    ppo_results = evaluate_ppo(

        G,

        transactions,

        trained_model,

        verbose=True
    )

    # ------------------------------------------------------
    # 12. Compare all three methods
    # ------------------------------------------------------

    comparison = compare_results(

        baseline_eta_0,

        baseline_eta_05,

        ppo_results
    )

    # ------------------------------------------------------
    # 13. Final summary
    # ------------------------------------------------------

    print(
        "\n"
        "======================================================"
    )

    print(
        " REAL GML RL EXPERIMENT COMPLETED"
    )

    print(
        "======================================================"
    )

    print()

    print(
        "Snapshot:",
        GML_FILENAME
    )

    print(
        "Nodes:",
        G.number_of_nodes()
    )

    print(
        "Directed channels:",
        G.number_of_edges()
    )

    print(
        "Transactions:",
        TRANSACTION_COUNT
    )

    print(
        "Fixed Top-K:",
        CONFIG[
            "graph"
        ][
            "top_k"
        ]
    )

    print()

    print(
        "Training time:",
        training_time,
        "seconds"
    )

    print()

    print(
        "Final comparison:"
    )

    print()

    for name, result in comparison.items():

        print(
            f"{name}:"
        )

        print(
            "  Method:",
            result.get(
                "method"
            )
        )

        print(
            "  Success Rate:",
            result.get(
                "success_rate"
            )
        )

        print(
            "  Average ETA:",
            result.get(
                "average_eta"
            )
        )

        print(
            "  Average Path Length:",
            result.get(
                "average_path_length"
            )
        )

        print(
            "  Average Fee:",
            result.get(
                "average_fee"
            )
        )

        print(
            "  Average Delay:",
            result.get(
                "average_delay"
            )
        )

        print(
            "  Average Backtrack:",
            result.get(
                "average_backtrack_count"
            )
        )

        print(
            "  Full Reroutes:",
            result.get(
                "full_reroute_events"
            )
        )

        print(
            "  Average Reward:",
            result.get(
                "average_reward"
            )
        )

        print()