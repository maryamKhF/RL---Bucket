# main.py

import argparse
import copy
import yaml
import random
import json
import numpy as np

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
        .read_text()
    )



# ==================================================
# Seed
# ==================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)



# ==================================================
# Main Pipeline
# ==================================================

def main():


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--snapshot",
        default=None
    )


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
    # Build Network
    # -------------------------------

    builder = LNGraphBuilder()


    if args.snapshot:

        G = builder.from_json(
            args.snapshot
        )

        print(
            "Loaded LN snapshot."
        )


    else:

        G = builder.synthetic(
            cfg["graph"]["synthetic_nodes"],
            cfg["graph"]["synthetic_extra_edges"],
            seed
        )

        print(
            "Synthetic LN graph created."
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

            ("lnd",lnd_cost),

            ("cln",cln_cost),

            ("ecl",ecl_cost)

        ]:


            G_train = copy.deepcopy(G)


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



            G_eval = copy.deepcopy(G)


            assign_failure_probabilities(
                G_eval,
                rate,
                seed
            )



            dynamics = NetworkDynamics(
                G_eval
            )



            # Evaluation pipeline

            results = evaluate(
                G_eval,
                test_tx,
                cfg,
                models if models else None
            )



            all_results[
                f"{int(rate*100)}%"
            ] = results



            # plots

            plot_success(
                results,
                f"results/success_{int(rate*100)}pct.png"
            )



            plot_fee(
                results,
                f"results/fee_{int(rate*100)}pct.png"
            )


            plot_delay(
                results,
                f"results/delay_{int(rate*100)}pct.png"
            )


            plot_path_length(
                results,
                f"results/path_{int(rate*100)}pct.png"
            )


            plot_recovery(
                results,
                f"results/recovery_{int(rate*100)}pct.png"
            )


            plot_carbon(
                results,
                f"results/carbon_{int(rate*100)}pct.png"
            )


            plot_runtime(
                results,
                f"results/runtime_{int(rate*100)}pct.png"
            )



            print(results)



        # save results


        Path("results").mkdir(
            exist_ok=True
        )


        with open(
            "results/metrics.json",
            "w"
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