# main.py

import argparse
import copy
import yaml
import random
import json
import numpy as np
import networkx as nx

from pathlib import Path


# Network
from Network.graph_builder import LNGraphBuilder


# Simulation
from Simulation.failure_model import assign_failure_probabilities
from Simulation.transaction_generator import generate_transactions
from Simulation.payment_simulator import simulate_payment
from Simulation.network_dynamics import NetworkDynamics
from Simulation.router import Router


# Pathfinding
from Pathfinding.heuristics import (
    lnd_cost,
    cln_cost,
    ecl_cost
)


# Bucket
from Bucket.candidate_manager import CandidateManager, make_bucket
from Bucket.bucket import Bucket
from Bucket.backtrack import choose_alternative


# RL
from RL.train import train_agent


# Evaluation
from Evaluations.evaluate import evaluate


# Visualization
from Visualization.plots import (
    plot_success,
    plot_fee,
    plot_delay,
    plot_path_length,
    plot_recovery,
    plot_carbon,
    plot_runtime
)


# ==================================================
# Config
# ==================================================

def load_cfg():

    return yaml.safe_load(
        Path("configs/config.yaml")
        .read_text(
            encoding="utf-8"
        )
    )


# ==================================================
# Seed
# ==================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


# ==================================================
# GML.GEO -> JSON-like Dictionary
# ==================================================

def geo_to_json(input_file):

    print(
        f"Reading GML.GEO file: {input_file}"
    )

    # ----------------------------------------------
    # Read GML.GEO
    # ----------------------------------------------

    graph = nx.read_gml(
        input_file,
        label=None
    )

    # ----------------------------------------------
    # Create JSON-like structure
    # ----------------------------------------------

    data = {

        "directed": graph.is_directed(),

        "multigraph": graph.is_multigraph(),

        "nodes": [],

        "edges": []

    }

    # ----------------------------------------------
    # Nodes
    # ----------------------------------------------

    for node_id, attributes in graph.nodes(
        data=True
    ):

        node_data = {

            "id": str(node_id)

        }

        for key, value in attributes.items():

            node_data[key] = value

        data["nodes"].append(
            node_data
        )

    # ----------------------------------------------
    # Edges
    # ----------------------------------------------

    if graph.is_multigraph():

        for source, target, key, attributes in graph.edges(
            keys=True,
            data=True
        ):

            edge_data = {

                "source": str(source),

                "target": str(target),

                "key": str(key)

            }

            for key, value in attributes.items():

                edge_data[key] = value

            data["edges"].append(
                edge_data
            )

    else:

        for source, target, attributes in graph.edges(
            data=True
        ):

            edge_data = {

                "source": str(source),

                "target": str(target)

            }

            for key, value in attributes.items():

                edge_data[key] = value

            data["edges"].append(
                edge_data
            )

    print(
        f"Converted {len(data['nodes'])} nodes "
        f"and {len(data['edges'])} edges."
    )

    return data


# ==================================================
# Main Pipeline
# ==================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--train",
        action="store_true"
    )


    parser.add_argument(
        "--evaluate",
        action="store_true"
    )


    parser.add_argument(
        "--transactions",
        type=int,
        default=None
    )


    args = parser.parse_args()


    # -------------------------------
    # Load Config
    # -------------------------------

    cfg = load_cfg()

    seed = cfg["seed"]

    set_seed(seed)


    # -------------------------------
    # GML.GEO File
    # -------------------------------

    geo_file = Path(
        "20190501.gml.geo"
    )


    if not geo_file.exists():

        raise FileNotFoundError(
            f"GML.GEO file not found: {geo_file.resolve()}"
        )


    # -------------------------------
    # Convert GML.GEO to JSON
    # -------------------------------

    data = geo_to_json(
        geo_file
    )


    # -------------------------------
    # Build Network
    # -------------------------------

    builder = LNGraphBuilder()


    G = builder.from_data(
        data
    )


    print(
        "\nReal Lightning Network graph created."
    )


    print(
        f"Nodes: {G.number_of_nodes()}"
    )


    print(
        f"Channels: {G.number_of_edges()}"
    )


    # -------------------------------
    # Transactions
    # -------------------------------

    n = (
        args.transactions
        or
        cfg["n_transactions"]
    )


    transactions = generate_transactions(
        G,
        n,
        seed,
        cfg["simulation"]["min_amount"],
        cfg["simulation"]["max_amount"]
    )


    split = int(
        n *
        cfg["train_ratio"]
    )


    train_tx = transactions[:split]

    test_tx = transactions[split:]


    # -------------------------------
    # RL Training
    # -------------------------------

    models = {}


    if args.train:

        for name, heuristic in [

            ("lnd", lnd_cost),

            ("cln", cln_cost),

            ("ecl", ecl_cost)

        ]:

            G_train = copy.deepcopy(
                G
            )


            assign_failure_probabilities(
                G_train,
                0.03,
                seed
            )


            models[name] = train_agent(
                G_train,
                train_tx,
                heuristic,
                cfg,
                name,
                seed
            )


    # -------------------------------
    # Evaluation
    # -------------------------------

    if args.evaluate or args.train:

        all_results = {}


        for rate in cfg["failure_rates"]:

            print(
                f"\nFailure rate {rate}"
            )


            G_eval = copy.deepcopy(
                G
            )


            assign_failure_probabilities(
                G_eval,
                rate,
                seed
            )


            dynamics = NetworkDynamics(
                G_eval
            )


            # ----------------------------------
            # Evaluation pipeline
            # ----------------------------------

            results = evaluate(
                G_eval,
                test_tx,
                cfg,
                models if models else None
            )


            all_results[
                f"{int(rate * 100)}%"
            ] = results


            # ----------------------------------
            # Plots
            # ----------------------------------

            plot_success(
                results,
                f"results/success_{int(rate * 100)}pct.png"
            )


            plot_fee(
                results,
                f"results/fee_{int(rate * 100)}pct.png"
            )


            plot_delay(
                results,
                f"results/delay_{int(rate * 100)}pct.png"
            )


            plot_path_length(
                results,
                f"results/path_{int(rate * 100)}pct.png"
            )


            plot_recovery(
                results,
                f"results/recovery_{int(rate * 100)}pct.png"
            )


            plot_carbon(
                results,
                f"results/carbon_{int(rate * 100)}pct.png"
            )


            plot_runtime(
                results,
                f"results/runtime_{int(rate * 100)}pct.png"
            )


            print(
                results
            )


        # ----------------------------------
        # Save Results
        # ----------------------------------

        Path(
            "results"
        ).mkdir(
            exist_ok=True
        )


        with open(
            "results/metrics.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                all_results,
                f,
                indent=4
            )


        print(
            "\nExperiment finished."
        )


# ==================================================

if __name__ == "__main__":

    main()