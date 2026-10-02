# main.py

import copy
import os
import random
import time
import yaml
import numpy as np
import networkx as nx
import traceback
import linecache
import sys
import threading

from pathlib import Path

from stable_baselines3 import PPO


# ============================================================
# Global Runtime State
# ============================================================

REPORT_FILE = Path("report.txt")

CURRENT_STAGE = "Program startup"

RUN_START_TIME = time.time()

CURRENT_PROGRESS = 0.0

CURRENT_STEP = 0

CURRENT_STEP_NAME = "Program startup"

PROGRESS_LOCK = threading.Lock()

HEARTBEAT_THREAD = None

HEARTBEAT_STOP_EVENT = threading.Event()


# ============================================================
# Runtime Mode
# ============================================================

# Fast validation:
#
#     GML snapshot
#          |
#          v
#     Graph construction
#          |
#          v
#     Existing PPO model
#          |
#          v
#     PPO -> eta
#          |
#          v
#     Adaptive Routing
#          |
#          v
#     Top-K (k=5)
#          |
#          v
#     Bucket
#          |
#          v
#     Payment Simulation
#          |
#          v
#     Failure Model
#          |
#          v
#     Partial Backtracking
#
# Only repeated PPO training is skipped.
#
# Default:
#     FAST_VALIDATION = True
#
# Normal training:
#     set RL_FAST_VALIDATION=0
#     py main.py
#
# Fast validation:
#     set RL_FAST_VALIDATION=1
#     py main.py

FAST_VALIDATION = (
    os.environ.get(
        "RL_FAST_VALIDATION",
        "1"
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on"
    }
)

MODEL_NAME = "end_to_end"


# ============================================================
# Progress Configuration
# ============================================================

TOTAL_STEPS = 10

STEP_PROGRESS = {
    0: 0.0,
    1: 10.0,
    2: 20.0,
    3: 30.0,
    4: 40.0,
    5: 50.0,
    6: 60.0,
    7: 70.0,
    8: 80.0,
    9: 90.0,
    10: 100.0,
}


# ============================================================
# Progress Helpers
# ============================================================

def _format_elapsed(seconds):

    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(
        seconds % 60
    )

    if hours > 0:

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return (
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def _progress_snapshot():

    with PROGRESS_LOCK:

        return (
            CURRENT_PROGRESS,
            CURRENT_STEP,
            CURRENT_STEP_NAME,
            CURRENT_STAGE
        )


def print_progress(
    progress=None,
    step=None,
    step_name=None,
    message=None
):

    global CURRENT_PROGRESS
    global CURRENT_STEP
    global CURRENT_STEP_NAME

    with PROGRESS_LOCK:

        if progress is not None:

            CURRENT_PROGRESS = float(
                max(
                    0.0,
                    min(
                        100.0,
                        progress
                    )
                )
            )

        if step is not None:

            CURRENT_STEP = int(
                step
            )

        if step_name is not None:

            CURRENT_STEP_NAME = str(
                step_name
            )

        current_progress = CURRENT_PROGRESS
        current_step = CURRENT_STEP
        current_step_name = CURRENT_STEP_NAME

    elapsed = _format_elapsed(
        time.time() - RUN_START_TIME
    )

    print(
        "\n------------------------------------------------------------"
    )

    print(
        f"OVERALL PROGRESS : "
        f"{current_progress:6.2f}%"
    )

    print(
        f"CURRENT STEP     : "
        f"STEP {current_step}"
    )

    print(
        f"CURRENT STAGE    : "
        f"{current_step_name}"
    )

    print(
        f"ELAPSED TIME     : "
        f"{elapsed}"
    )

    if message:

        print(
            f"STATUS           : "
            f"{message}"
        )

    print(
        "------------------------------------------------------------"
    )


# ============================================================
# Heartbeat Monitor
# ============================================================

def _heartbeat_worker():

    while not HEARTBEAT_STOP_EVENT.wait(
        30.0
    ):

        progress, step, step_name, stage = (
            _progress_snapshot()
        )

        elapsed = _format_elapsed(
            time.time() - RUN_START_TIME
        )

        print(
            "\n[PROGRESS MONITOR]"
        )

        print(
            f"Step      : {step}/{TOTAL_STEPS}"
        )

        print(
            f"Stage     : {step_name}"
        )

        print(
            f"Progress  : {progress:.2f}%"
        )

        print(
            f"Elapsed   : {elapsed}"
        )

        print(
            "Status    : STILL RUNNING"
        )


def start_heartbeat():

    global HEARTBEAT_THREAD

    HEARTBEAT_STOP_EVENT.clear()

    HEARTBEAT_THREAD = threading.Thread(
        target=_heartbeat_worker,
        daemon=True
    )

    HEARTBEAT_THREAD.start()


def stop_heartbeat():

    HEARTBEAT_STOP_EVENT.set()


# ============================================================
# Stage Progress
# ============================================================

def begin_step(
    step,
    name,
    report=None,
    message=None
):

    global CURRENT_STAGE

    CURRENT_STAGE = str(
        name
    )

    progress = STEP_PROGRESS.get(
        int(step),
        CURRENT_PROGRESS
    )

    print_progress(
        progress=progress,
        step=step,
        step_name=name,
        message=message
    )

    if report is not None:

        report.item(
            "Step",
            f"{step}/{TOTAL_STEPS}"
        )

        report.item(
            "Progress",
            f"{progress:.2f}%"
        )

        report.item(
            "Stage",
            name
        )


def complete_step(
    step,
    name,
    report=None,
    message="COMPLETED"
):

    global CURRENT_STAGE

    CURRENT_STAGE = str(
        name
    )

    progress = STEP_PROGRESS.get(
        int(step),
        CURRENT_PROGRESS
    )

    print_progress(
        progress=progress,
        step=step,
        step_name=name,
        message=message
    )

    if report is not None:

        report.item(
            "Step completed",
            f"{step}/{TOTAL_STEPS}"
        )

        report.item(
            "Progress",
            f"{progress:.2f}%"
        )

        report.item(
            "Status",
            message
        )


# ============================================================
# Early Error Reporting
# ============================================================

def initialize_bootstrap_report():

    try:

        REPORT_FILE.write_text(
            "RL + BUCKET END-TO-END EXECUTION REPORT\n"
            "============================================================\n"
            f"Run started: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Status: STARTED\n"
            "\n",
            encoding="utf-8"
        )

    except Exception:

        pass


def write_bootstrap_error(exc):

    try:

        exc_type = type(exc).__name__

        exc_message = str(exc)

        traceback_text = traceback.format_exc()

        report_lines = [

            "",

            "============================================================",

            "FATAL ERROR DURING MODULE IMPORT",

            "============================================================",

            f"Current stage: {CURRENT_STAGE}",

            f"Exception type: {exc_type}",

            f"Exception message: {exc_message}",

            "",

            "Traceback:",

            traceback_text,

        ]

        REPORT_FILE.write_text(

            "\n".join(report_lines),

            encoding="utf-8"

        )

    except Exception:

        pass


# ============================================================
# Initialize Early Report
# ============================================================

initialize_bootstrap_report()


# ============================================================
# Network
# ============================================================

CURRENT_STAGE = "Importing Network modules"

try:

    from Network.graph_builder import LNGraphBuilder

except Exception as exc:

    write_bootstrap_error(exc)

    raise


# ============================================================
# Simulation
# ============================================================

CURRENT_STAGE = "Importing Simulation modules"

try:

    from Simulation.failure_model import (
        assign_failure_probabilities,
        FailureModel,
    )

    from Simulation.transaction_generator import (
        generate_transactions,
    )

    from Simulation.payment_simulator import (
        PaymentSimulator,
    )

    from Simulation.network_dynamics import (
        NetworkDynamics,
    )

except Exception as exc:

    write_bootstrap_error(exc)

    raise


# ============================================================
# RL
# ============================================================

CURRENT_STAGE = "Importing RL modules"

try:

    from RL.train import train_agent

    from RL.environment import RoutingEnv

    from RL.ppo_agent import build_ppo

except Exception as exc:

    write_bootstrap_error(exc)

    raise


# ============================================================
# Pathfinding
# ============================================================

CURRENT_STAGE = "Importing Pathfinding modules"

try:

    from Pathfinding.heuristics import lnd_cost

except Exception as exc:

    write_bootstrap_error(exc)

    raise


# ============================================================
# Runtime Stage
# ============================================================

def set_stage(stage):

    global CURRENT_STAGE

    CURRENT_STAGE = str(
        stage
    )


# ============================================================
# Detailed Runtime Error Reporting
# ============================================================

def write_runtime_error(
    exc,
    report=None
):

    global CURRENT_STAGE

    try:

        exc_type = type(exc).__name__

        exc_message = str(exc)

        traceback_text = traceback.format_exc()

        report_lines = [

            "",

            "============================================================",

            "RUNTIME ERROR",

            "============================================================",

            f"Current stage : {CURRENT_STAGE}",

            f"Exception type: {exc_type}",

            f"Exception     : {exc_message}",

            "",

            "------------------------------------------------------------",

            "TRACEBACK",

            "------------------------------------------------------------",

            traceback_text,

        ]

        tb = traceback.extract_tb(
            sys.exc_info()[2]
        )

        if tb:

            last_frame = tb[-1]

            filename = last_frame.filename

            line_number = last_frame.lineno

            function_name = last_frame.name

            source_line = last_frame.line

            if not source_line:

                try:

                    source_line = linecache.getline(
                        filename,
                        line_number
                    ).strip()

                except Exception:

                    source_line = ""

            report_lines.extend(

                [

                    "",

                    "------------------------------------------------------------",

                    "ERROR LOCATION",

                    "------------------------------------------------------------",

                    f"File     : {filename}",

                    f"Line     : {line_number}",

                    f"Function : {function_name}",

                    f"Code     : {source_line}",

                ]

            )

        if report is not None:

            report.section(
                "ERROR"
            )

            report.item(
                "Current stage",
                CURRENT_STAGE
            )

            report.item(
                "Current step",
                CURRENT_STEP
            )

            report.item(
                "Current progress",
                f"{CURRENT_PROGRESS:.2f}%"
            )

            report.item(
                "Exception type",
                exc_type
            )

            report.item(
                "Exception message",
                exc_message
            )

            if tb:

                report.item(
                    "Error file",
                    filename
                )

                report.item(
                    "Error line",
                    line_number
                )

                report.item(
                    "Error function",
                    function_name
                )

                report.item(
                    "Error code",
                    source_line
                )

            report.add("")

            report.add(
                "Full traceback:"
            )

            report.add(
                traceback_text
            )

            report.add("")

            report.add(
                "Execution stopped because of the error."
            )

            report.save()

        else:

            REPORT_FILE.write_text(

                "\n".join(report_lines),

                encoding="utf-8"

            )

    except Exception:

        try:

            REPORT_FILE.write_text(

                "Runtime error occurred.\n\n"
                + traceback.format_exc(),

                encoding="utf-8"

            )

        except Exception:

            pass


# ============================================================
# Config
# ============================================================

def load_cfg():

    # Support both:
    #
    #     Configs/config.yaml
    #
    # and:
    #
    #     configs/config.yaml
    #
    # Windows is case-insensitive, but this also keeps the
    # project portable to case-sensitive systems.

    possible_paths = [

        Path("Configs/config.yaml"),

        Path("configs/config.yaml"),

    ]

    config_path = None

    for candidate in possible_paths:

        if candidate.exists():

            config_path = candidate

            break

    if config_path is None:

        expected = "\n".join(
            str(
                path.resolve()
            )
            for path in possible_paths
        )

        raise FileNotFoundError(

            "Configuration file not found.\n"
            "Expected one of:\n"
            f"{expected}"

        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        cfg = yaml.safe_load(f)

    if cfg is None:

        raise ValueError(
            "Configuration file is empty."
        )

    return cfg


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


# ============================================================
# GML.GEO -> JSON-like Dictionary
# ============================================================

def geo_to_json(input_file):

    print(
        f"Reading snapshot: {input_file}"
    )

    graph = nx.read_gml(
        input_file,
        label=None
    )

    data = {

        "directed":
            graph.is_directed(),

        "multigraph":
            graph.is_multigraph(),

        "nodes": [],

        "edges": []

    }

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    for node_id, attributes in graph.nodes(
        data=True
    ):

        node_data = {

            "id":
                str(node_id)

        }

        for key, value in attributes.items():

            node_data[key] = value

        data["nodes"].append(
            node_data
        )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    if graph.is_multigraph():

        for source, target, key, attributes in graph.edges(
            keys=True,
            data=True
        ):

            edge_data = {

                "source":
                    str(source),

                "target":
                    str(target),

                "key":
                    str(key)

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

                "source":
                    str(source),

                "target":
                    str(target)

            }

            for key, value in attributes.items():

                edge_data[key] = value

            data["edges"].append(
                edge_data
            )

    return data


# ============================================================
# Report Writer
# ============================================================

class ReportWriter:

    def __init__(self, path="report.txt"):

        self.path = Path(path)

        self.lines = []

    def add(self, text=""):

        self.lines.append(
            str(text)
        )

    def section(self, title):

        self.add("")

        self.add("=" * 64)

        self.add(title)

        self.add("=" * 64)

    def item(self, key, value):

        self.add(
            f"{key:<28}: {value}"
        )

    def save(self):

        self.path.write_text(
            "\n".join(self.lines)
            + "\n",
            encoding="utf-8"
        )

        return self.path


# ============================================================
# Utility
# ============================================================

def safe_value(value, default=None):

    if value is None:

        return default

    return value


# ============================================================
# PPO Model Loading
# ============================================================

def load_existing_ppo_model(
    model_path,
    env=None
):
    """
    Load an existing PPO model for fast validation.

    The loaded PPO model is still used for the routing
    decision:

        observation -> PPO -> eta

    The runtime environment continues with:

        eta
          |
          v
        Adaptive Routing
          |
          v
        Top-K (k=5)
          |
          v
        Bucket
          |
          v
        Payment Simulation
          |
          v
        Failure Model
          |
          v
        Partial Backtracking
    """

    model_path = Path(
        model_path
    )

    if model_path.suffix.lower() != ".zip":

        zip_path = Path(
            str(model_path) + ".zip"
        )

    else:

        zip_path = model_path

    if not zip_path.exists():

        raise FileNotFoundError(

            "Existing PPO model was not found.\n\n"

            f"Expected model:\n"
            f"{zip_path.resolve()}\n\n"

            "To create the model, run normal training once:\n\n"

            "    set RL_FAST_VALIDATION=0\n"
            "    py main.py\n\n"

            "After the model is created, use fast validation:\n\n"

            "    set RL_FAST_VALIDATION=1\n"
            "    py main.py"

        )

    print(
        "\n============================================================"
    )

    print(
        "LOADING EXISTING PPO MODEL"
    )

    print(
        "============================================================"
    )

    print(
        f"Model : {zip_path.resolve()}"
    )

    load_start = time.time()

    model = PPO.load(
        str(zip_path),
        env=env
    )

    load_time = (
        time.time()
        -
        load_start
    )

    print(
        f"PPO model loaded in {load_time:.4f} seconds."
    )

    return model, load_time


# ============================================================
# Select Random Payment Scenario
# ============================================================

def select_payment_scenario(
    G,
    cfg,
    seed
):

    transactions = generate_transactions(

        G,

        1,

        seed,

        cfg["simulation"]["min_amount"],

        cfg["simulation"]["max_amount"]

    )

    if not transactions:

        raise RuntimeError(
            "Could not generate a payment transaction."
        )

    return transactions[0]


# ============================================================
# Main
# ============================================================

def main():

    global RUN_START_TIME

    RUN_START_TIME = time.time()

    start_heartbeat()

    report = ReportWriter(
        "report.txt"
    )

    report.section(
        "RL + BUCKET END-TO-END EXECUTION REPORT"
    )

    report.item(
        "Run started",
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    report.item(
        "Execution mode",
        "FAST VALIDATION"
        if FAST_VALIDATION
        else "NORMAL TRAINING"
    )

    report.item(
        "Initial stage",
        CURRENT_STAGE
    )

    report.item(
        "Overall progress",
        "0.00%"
    )

    report.item(
        "Current step",
        "0/10"
    )

    report.save()

    runtime_env = None

    training_time = 0.0

    model_load_time = 0.0

    prediction_time = 0.0

    execution_time = 0.0

    model = None

    try:

        # ====================================================
        # STEP 0
        # ====================================================

        set_stage(
            "STEP 0 - LOAD CONFIGURATION"
        )

        report.section(
            "STEP 0 - CONFIGURATION"
        )

        begin_step(
            0,
            "STEP 0 - LOAD CONFIGURATION",
            report,
            "Loading configuration..."
        )

        cfg = load_cfg()

        seed = int(
            cfg["seed"]
        )

        set_seed(
            seed
        )

        report.item(
            "Configuration",
            "SUCCESS"
        )

        report.item(
            "Random seed",
            seed
        )

        report.item(
            "Execution mode",
            "FAST VALIDATION"
            if FAST_VALIDATION
            else "NORMAL TRAINING"
        )

        complete_step(
            0,
            "STEP 0 - LOAD CONFIGURATION",
            report,
            "COMPLETED"
        )

        # ----------------------------------------------------
        # Snapshot validation
        # ----------------------------------------------------

        snapshot_path = Path(
            "20190501.gml.geo"
        )

        if not snapshot_path.exists():

            raise FileNotFoundError(
                f"Snapshot not found: "
                f"{snapshot_path.resolve()}"
            )

        report.section(
            "RUN INFORMATION"
        )

        report.item(
            "Snapshot",
            snapshot_path.name
        )

        report.item(
            "Snapshot path",
            str(snapshot_path.resolve())
        )

        report.item(
            "Random seed",
            seed
        )

        report.item(
            "Run started",
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        # ====================================================
        # STEP 1
        # ====================================================

        set_stage(
            "STEP 1 - LOAD SNAPSHOT"
        )

        report.section(
            "STEP 1 - SNAPSHOT"
        )

        begin_step(
            1,
            "STEP 1 - LOAD SNAPSHOT",
            report,
            "Loading blockchain snapshot..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 1 - LOADING SNAPSHOT"
        )

        print(
            "============================================================"
        )

        data = geo_to_json(
            snapshot_path
        )

        builder = LNGraphBuilder()

        G = builder.from_data(
            data
        )

        node_count = (
            G.number_of_nodes()
        )

        edge_count = (
            G.number_of_edges()
        )

        report.item(
            "Snapshot name",
            snapshot_path.name
        )

        report.item(
            "Graph type",
            type(G).__name__
        )

        report.item(
            "Number of nodes",
            node_count
        )

        report.item(
            "Number of channels",
            edge_count
        )

        report.item(
            "Directed",
            G.is_directed()
        )

        report.item(
            "Multigraph",
            G.is_multigraph()
        )

        print(
            f"Snapshot : {snapshot_path.name}"
        )

        print(
            f"Nodes    : {node_count}"
        )

        print(
            f"Channels : {edge_count}"
        )

        complete_step(
            1,
            "STEP 1 - LOAD SNAPSHOT",
            report,
            "SNAPSHOT LOADED"
        )

        # ====================================================
        # STEP 2
        # ====================================================

        set_stage(
            "STEP 2 - PREPARE TRAINING GRAPH"
        )

        report.section(
            "STEP 2 - PREPARE TRAINING GRAPH"
        )

        begin_step(
            2,
            "STEP 2 - PREPARE TRAINING GRAPH",
            report,
            "Preparing training graph..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 2 - PREPARING TRAINING GRAPH"
        )

        print(
            "============================================================"
        )

        G_train = copy.deepcopy(
            G
        )

        training_failure_rate = 0.03

        assign_failure_probabilities(

            G_train,

            training_failure_rate,

            seed

        )

        report.item(
            "Training failure rate",
            training_failure_rate
        )

        report.item(
            "Training nodes",
            G_train.number_of_nodes()
        )

        report.item(
            "Training channels",
            G_train.number_of_edges()
        )

        complete_step(
            2,
            "STEP 2 - PREPARE TRAINING GRAPH",
            report,
            "TRAINING GRAPH READY"
        )

        # ====================================================
        # STEP 3
        # ====================================================

        set_stage(
            "STEP 3 - GENERATE TRAINING TRANSACTIONS"
        )

        report.section(
            "STEP 3 - TRAINING TRANSACTIONS"
        )

        begin_step(
            3,
            "STEP 3 - GENERATE TRAINING TRANSACTIONS",
            report,
            "Generating training transactions..."
        )

        configured_transaction_count = int(
            cfg["n_transactions"]
        )

        # ----------------------------------------------------
        # Fast validation optimization
        # ----------------------------------------------------
        #
        # The final end-to-end validation uses exactly one
        # payment transaction.
        #
        # Therefore generating thousands of training
        # transactions is unnecessary when PPO training is
        # skipped.

        if FAST_VALIDATION:

            training_transaction_count = 1

        else:

            training_transaction_count = (
                configured_transaction_count
            )

        training_transactions = generate_transactions(

            G_train,

            training_transaction_count,

            seed,

            cfg["simulation"]["min_amount"],

            cfg["simulation"]["max_amount"]

        )

        train_ratio = float(
            cfg["train_ratio"]
        )

        if FAST_VALIDATION:

            train_count = min(
                1,
                len(training_transactions)
            )

        else:

            train_count = max(
                1,
                int(
                    len(training_transactions)
                    *
                    train_ratio
                )
            )

        train_tx = (
            training_transactions[
                :train_count
            ]
        )

        report.item(
            "Configured transactions",
            configured_transaction_count
        )

        report.item(
            "Generated transactions",
            len(training_transactions)
        )

        report.item(
            "Train ratio",
            train_ratio
        )

        report.item(
            "Transactions used for PPO",
            len(train_tx)
        )

        if FAST_VALIDATION:

            report.item(
                "Transaction generation mode",
                "FAST VALIDATION - MINIMAL"
            )

        else:

            report.item(
                "Transaction generation mode",
                "NORMAL TRAINING"
            )

        print(
            f"Configured transactions : "
            f"{configured_transaction_count}"
        )

        print(
            f"Generated transactions  : "
            f"{len(training_transactions)}"
        )

        print(
            f"Transactions for PPO    : "
            f"{len(train_tx)}"
        )

        complete_step(
            3,
            "STEP 3 - GENERATE TRAINING TRANSACTIONS",
            report,
            "TRAINING TRANSACTIONS READY"
        )

        # ====================================================
        # STEP 4 - PPO
        # ====================================================

        set_stage(
            "STEP 4 - PPO TRAINING"
        )

        report.section(
            "STEP 4 - PPO TRAINING"
        )

        begin_step(
            4,
            "STEP 4 - PPO TRAINING",
            report,
            "Preparing PPO model..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 4 - PPO"
        )

        print(
            "============================================================"
        )

        model_dir = Path(
            cfg["rl"].get(
                "model_dir",
                "models"
            )
        )

        model_path = (
            model_dir /
            MODEL_NAME
        )

        model_zip_path = Path(
            str(model_path) + ".zip"
        )

        # ----------------------------------------------------
        # FAST VALIDATION
        # ----------------------------------------------------

        if FAST_VALIDATION:

            print(
                "\nFAST VALIDATION MODE IS ENABLED."
            )

            print(
                "PPO training will NOT be repeated."
            )

            print(
                "The existing PPO model will be loaded."
            )

            report.item(
                "Execution mode",
                "FAST VALIDATION"
            )

            report.item(
                "PPO training",
                "SKIPPED"
            )

            report.item(
                "Expected model",
                str(
                    model_zip_path.resolve()
                )
            )

            model, model_load_time = (
                load_existing_ppo_model(
                    model_path,
                    env=None
                )
            )

            report.item(
                "Model load status",
                "SUCCESS"
            )

            report.item(
                "Model load time (seconds)",
                round(
                    model_load_time,
                    4
                )
            )

            report.item(
                "Training time (seconds)",
                0.0
            )

            report.item(
                "Model",
                str(
                    model_zip_path
                )
            )

            report.add("")

            report.add(
                "Fast validation skipped PPO training and "
                "reused the existing trained PPO model."
            )

            complete_step(
                4,
                "STEP 4 - PPO TRAINING",
                report,
                "EXISTING PPO MODEL LOADED"
            )

        # ----------------------------------------------------
        # NORMAL TRAINING
        # ----------------------------------------------------

        else:

            print(
                "\nPPO TRAINING IS RUNNING..."
            )

            print(
                "Progress shown here is stage-level."
            )

            print(
                "A real PPO timestep percentage requires "
                "a callback inside RL/train.py."
            )

            report.item(
                "Execution mode",
                "NORMAL TRAINING"
            )

            report.item(
                "Algorithm",
                "PPO"
            )

            report.item(
                "Training snapshot",
                snapshot_path.name
            )

            report.item(
                "Training nodes",
                G_train.number_of_nodes()
            )

            report.item(
                "Training channels",
                G_train.number_of_edges()
            )

            training_start = time.time()

            model = train_agent(

                G_train,

                train_tx,

                lnd_cost,

                cfg,

                MODEL_NAME,

                seed

            )

            training_time = (
                time.time()
                -
                training_start
            )

            report.item(
                "Training status",
                "SUCCESS"
            )

            report.item(
                "Training time (seconds)",
                round(
                    training_time,
                    4
                )
            )

            report.item(
                "Model",
                str(
                    model_path
                )
            )

            report.add("")

            report.add(
                "The PPO agent was trained on the same "
                "snapshot that will be used for the payment scenario."
            )

            print(
                f"PPO training completed "
                f"in {training_time:.2f} seconds."
            )

            complete_step(
                4,
                "STEP 4 - PPO TRAINING",
                report,
                "PPO TRAINING COMPLETED"
            )

        # ====================================================
        # STEP 5
        # ====================================================

        set_stage(
            "STEP 5 - PAYMENT SCENARIO"
        )

        report.section(
            "STEP 5 - PAYMENT SCENARIO"
        )

        begin_step(
            5,
            "STEP 5 - PAYMENT SCENARIO",
            report,
            "Creating payment scenario..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 5 - CREATING PAYMENT SCENARIO"
        )

        print(
            "============================================================"
        )

        scenario_seed = (
            seed + 1
        )

        transaction = select_payment_scenario(

            G,

            cfg,

            scenario_seed

        )

        source = str(
            transaction.source
        )

        destination = str(
            transaction.destination
        )

        amount = float(
            transaction.amount
        )

        report.item(
            "Source",
            source
        )

        report.item(
            "Destination",
            destination
        )

        report.item(
            "Payment amount",
            amount
        )

        report.item(
            "Transaction ID",
            transaction.tx_id
        )

        print(
            f"Source      : {source}"
        )

        print(
            f"Destination : {destination}"
        )

        print(
            f"Amount      : {amount}"
        )

        complete_step(
            5,
            "STEP 5 - PAYMENT SCENARIO",
            report,
            "PAYMENT SCENARIO READY"
        )

        # ====================================================
        # STEP 6
        # ====================================================

        set_stage(
            "STEP 6 - CREATE PAYMENT GRAPH"
        )

        report.section(
            "STEP 6 - PAYMENT GRAPH"
        )

        begin_step(
            6,
            "STEP 6 - CREATE PAYMENT GRAPH",
            report,
            "Preparing evaluation graph..."
        )

        G_eval = copy.deepcopy(
            G
        )

        payment_failure_rate = float(
            cfg.get(
                "runtime_failure_rate",
                0.03
            )
        )

        assign_failure_probabilities(

            G_eval,

            payment_failure_rate,

            seed + 1

        )

        report.item(
            "Payment failure rate",
            payment_failure_rate
        )

        report.item(
            "Evaluation nodes",
            G_eval.number_of_nodes()
        )

        report.item(
            "Evaluation channels",
            G_eval.number_of_edges()
        )

        complete_step(
            6,
            "STEP 6 - CREATE PAYMENT GRAPH",
            report,
            "PAYMENT GRAPH READY"
        )

        # ====================================================
        # STEP 7
        # ====================================================

        set_stage(
            "STEP 7 - CREATE RUNTIME ENVIRONMENT"
        )

        report.section(
            "STEP 7 - END-TO-END ROUTING"
        )

        begin_step(
            7,
            "STEP 7 - CREATE RUNTIME ENVIRONMENT",
            report,
            "Creating runtime routing environment..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 7 - END-TO-END ROUTING"
        )

        print(
            "============================================================"
        )

        dynamics = NetworkDynamics(
            G_eval
        )

        failure_model = FailureModel(
            seed=seed + 1
        )

        payment_simulator = PaymentSimulator(

            G=G_eval,

            failure_model=failure_model,

            network_dynamics=dynamics

        )

        report.item(
            "Network dynamics",
            type(dynamics).__name__
        )

        report.item(
            "Failure model",
            type(failure_model).__name__
        )

        report.item(
            "Payment simulator",
            type(payment_simulator).__name__
        )

        runtime_cfg = copy.deepcopy(
            cfg
        )

        runtime_env = RoutingEnv(

            G=G_eval,

            transactions=[transaction],

            heuristic_fn=lnd_cost,

            config=runtime_cfg,

            mode="eval"

        )

        # ----------------------------------------------------
        # Attach evaluation environment to loaded/trained PPO
        # ----------------------------------------------------

        if model is None:

            raise RuntimeError(
                "PPO model is not available before evaluation."
            )

        try:

            model.set_env(
                runtime_env
            )

            report.item(
                "PPO runtime environment",
                "ATTACHED"
            )

        except Exception as env_attach_exc:

            report.item(
                "PPO runtime environment",
                f"ATTACHMENT FAILED: {env_attach_exc}"
            )

            raise

        observation, reset_info = runtime_env.reset(
            seed=seed + 1
        )

        report.item(
            "Environment reset",
            "SUCCESS"
        )

        report.item(
            "Reset info",
            reset_info
        )

        complete_step(
            7,
            "STEP 7 - CREATE RUNTIME ENVIRONMENT",
            report,
            "RUNTIME ENVIRONMENT READY"
        )

        # ====================================================
        # STEP 8
        # ====================================================

        set_stage(
            "STEP 8 - PPO ROUTING DECISION"
        )

        report.section(
            "STEP 8 - PPO ROUTING DECISION"
        )

        begin_step(
            8,
            "STEP 8 - PPO ROUTING DECISION",
            report,
            "PPO selecting routing action..."
        )

        print(
            "\nPPO is selecting the routing decision..."
        )

        prediction_start = time.time()

        action, _state = model.predict(

            observation,

            deterministic=True

        )

        prediction_time = (
            time.time()
            -
            prediction_start
        )

        report.item(
            "PPO prediction",
            "SUCCESS"
        )

        report.item(
            "Raw action",
            np.asarray(
                action
            ).tolist()
        )

        report.item(
            "Prediction time (seconds)",
            round(
                prediction_time,
                6
            )
        )

        complete_step(
            8,
            "STEP 8 - PPO ROUTING DECISION",
            report,
            "PPO ROUTING DECISION READY"
        )

        # ====================================================
        # STEP 9
        # ====================================================

        set_stage(
            "STEP 9 - EXECUTE BUCKET PAYMENT FAILURE BACKTRACKING"
        )

        report.section(
            "STEP 9 - BUCKET / PAYMENT / FAILURE / BACKTRACKING"
        )

        begin_step(
            9,
            "STEP 9 - EXECUTE BUCKET PAYMENT FAILURE BACKTRACKING",
            report,
            "Executing complete routing pipeline..."
        )

        print(
            "\nExecuting:"
        )

        print(
            "PPO -> Adaptive Routing -> Top-K -> "
            "Bucket -> Payment -> Failure -> Backtracking"
        )

        execution_start = time.time()

        (
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = runtime_env.step(
            action
        )

        execution_time = (
            time.time()
            -
            execution_start
        )

        # ====================================================
        # BASIC RESULT
        # ====================================================

        success = bool(
            info.get(
                "success",
                False
            )
        )

        eta = safe_value(
            info.get(
                "eta"
            )
        )

        top_k = safe_value(
            info.get(
                "top_k"
            ),
            5
        )

        candidate_count = safe_value(
            info.get(
                "candidate_path_count"
            ),
            0
        )

        usable_candidate_count = safe_value(
            info.get(
                "usable_candidate_count"
            ),
            0
        )

        bucket_size = safe_value(
            info.get(
                "bucket_size"
            ),
            0
        )

        final_path = safe_value(
            info.get(
                "path"
            ),
            []
        )

        path_length = safe_value(
            info.get(
                "path_length"
            ),
            0
        )

        fee = float(
            safe_value(
                info.get(
                    "fee"
                ),
                0.0
            )
        )

        delay = float(
            safe_value(
                info.get(
                    "delay"
                ),
                0.0
            )
        )

        carbon = float(
            safe_value(
                info.get(
                    "carbon"
                ),
                0.0
            )
        )

        attempt_count = int(
            safe_value(
                info.get(
                    "attempt_count"
                ),
                0
            )
        )

        backtrack_count = int(
            safe_value(
                info.get(
                    "backtrack_count"
                ),
                0
            )
        )

        partial_backtrack_count = int(
            safe_value(
                info.get(
                    "partial_backtrack_count"
                ),
                0
            )
        )

        partial_backtrack_success = int(
            safe_value(
                info.get(
                    "partial_backtrack_success"
                ),
                0
            )
        )

        full_reroute_count = int(
            safe_value(
                info.get(
                    "full_reroute_count"
                ),
                0
            )
        )

        failure_probability = float(
            safe_value(
                info.get(
                    "failure_probability"
                ),
                0.0
            )
        )

        reason = safe_value(
            info.get(
                "reason"
            ),
            ""
        )

        reward = float(
            reward
        )

        report.item(
            "Terminated",
            terminated
        )

        report.item(
            "Truncated",
            truncated
        )

        report.item(
            "Execution info keys",
            list(info.keys())
        )

        report.item(
            "Adaptive eta",
            eta
        )

        report.item(
            "Requested Top-K",
            top_k
        )

        report.item(
            "Generated candidates",
            candidate_count
        )

        report.item(
            "Usable candidates",
            usable_candidate_count
        )

        report.item(
            "Bucket size",
            bucket_size
        )

        report.add("")

        report.add(
            "The routing pipeline generated the candidate "
            "routes and placed the usable candidates into the Bucket."
        )

        report.add("")

        report.add(
            "Bucket routing policy:"
        )

        report.add(
            "Route #1 is tested first."
        )

        report.add(
            "If Route #1 fails, partial/onion backtracking "
            "moves to the next available candidate."
        )

        report.add(
            "The process continues until the payment succeeds "
            "or the Bucket is exhausted."
        )

        complete_step(
            9,
            "STEP 9 - EXECUTE BUCKET PAYMENT FAILURE BACKTRACKING",
            report,
            "ROUTING PIPELINE EXECUTED"
        )

        # ====================================================
        # STEP 10
        # ====================================================

        set_stage(
            "STEP 10 - FINAL PAYMENT RESULT"
        )

        report.section(
            "STEP 10 - FINAL PAYMENT RESULT"
        )

        begin_step(
            10,
            "STEP 10 - FINAL PAYMENT RESULT",
            report,
            "Processing final payment result..."
        )

        report.item(
            "Source",
            source
        )

        report.item(
            "Destination",
            destination
        )

        report.item(
            "Payment amount",
            amount
        )

        report.item(
            "Payment status",
            "SUCCESS" if success else "FAILED"
        )

        report.item(
            "Destination reached",
            "YES" if success else "NO"
        )

        report.item(
            "Final path",
            " -> ".join(
                str(node)
                for node in final_path
            )
            if final_path
            else "NONE"
        )

        report.item(
            "Number of hops",
            max(
                0,
                path_length - 1
            )
        )

        report.item(
            "Total attempts",
            attempt_count
        )

        report.item(
            "Partial backtracks",
            partial_backtrack_count
        )

        report.item(
            "Successful partial backtracks",
            partial_backtrack_success
        )

        report.item(
            "Full reroutes",
            full_reroute_count
        )

        report.item(
            "Failure probability",
            failure_probability
        )

        report.item(
            "Fee",
            fee
        )

        report.item(
            "Delay",
            delay
        )

        report.item(
            "Carbon",
            carbon
        )

        report.item(
            "Reward",
            reward
        )

        report.item(
            "Reason",
            reason
        )

        successful_bucket_rank = None

        if success:

            if partial_backtrack_count == 0:

                successful_bucket_rank = 1

            else:

                successful_bucket_rank = (
                    partial_backtrack_count + 1
                )

        if successful_bucket_rank is not None:

            report.item(
                "Successful Bucket route",
                f"{successful_bucket_rank} / {bucket_size}"
            )

        else:

            report.item(
                "Successful Bucket route",
                "NONE"
            )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        report.section(
            "FINAL STATUS"
        )

        if success:

            report.add(
                "PAYMENT SUCCESSFUL"
            )

            report.add(
                f"The payment reached destination "
                f"{destination} from source {source}."
            )

            if successful_bucket_rank == 1:

                report.add(
                    "The first Bucket route succeeded."
                )

            else:

                report.add(
                    f"The payment succeeded after "
                    f"{attempt_count} attempts."
                )

                report.add(
                    f"Successful route: "
                    f"Bucket candidate #{successful_bucket_rank}."
                )

        else:

            report.add(
                "PAYMENT FAILED"
            )

            report.add(
                "No usable route successfully delivered "
                "the payment to the destination."
            )

            report.add(
                "Please try again. A valid route is "
                "currently not available."
            )

        # ====================================================
        # EXECUTION TIME
        # ====================================================

        total_time = (
            time.time()
            -
            RUN_START_TIME
        )

        report.section(
            "EXECUTION TIME"
        )

        report.item(
            "Execution mode",
            "FAST VALIDATION"
            if FAST_VALIDATION
            else "NORMAL TRAINING"
        )

        report.item(
            "PPO training time (seconds)",
            round(
                training_time,
                4
            )
        )

        report.item(
            "PPO model load time (seconds)",
            round(
                model_load_time,
                4
            )
        )

        report.item(
            "PPO prediction time (seconds)",
            round(
                prediction_time,
                6
            )
        )

        report.item(
            "Payment pipeline time (seconds)",
            round(
                execution_time,
                4
            )
        )

        report.item(
            "Total execution time (seconds)",
            round(
                total_time,
                4
            )
        )

        # ====================================================
        # END
        # ====================================================

        set_stage(
            "RUN COMPLETED"
        )

        report.section(
            "END OF REPORT"
        )

        report.item(
            "Run status",
            "COMPLETED"
        )

        report.item(
            "Overall progress",
            "100.00%"
        )

        report.item(
            "Run completed",
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        report.item(
            "Report file",
            str(
                Path("report.txt").resolve()
            )
        )

        report_path = report.save()

        print_progress(
            progress=100.0,
            step=10,
            step_name="RUN COMPLETED",
            message="END-TO-END EXECUTION COMPLETED"
        )

        print(
            "\n============================================================"
        )

        print(
            "END-TO-END EXECUTION FINISHED"
        )

        print(
            "============================================================"
        )

        print(
            f"Execution mode : "
            f"{'FAST VALIDATION' if FAST_VALIDATION else 'NORMAL TRAINING'}"
        )

        print(
            f"Snapshot       : {snapshot_path.name}"
        )

        print(
            f"Source         : {source}"
        )

        print(
            f"Destination    : {destination}"
        )

        print(
            f"Amount         : {amount}"
        )

        print(
            f"Top-K          : {top_k}"
        )

        print(
            f"Candidates     : {usable_candidate_count}"
        )

        print(
            f"Attempts       : {attempt_count}"
        )

        print(
            f"Backtracks     : {partial_backtrack_count}"
        )

        print(
            f"Success        : {success}"
        )

        print(
            f"Fee            : {fee}"
        )

        print(
            f"Hops           : {max(0, path_length - 1)}"
        )

        print(
            f"PPO train time : {training_time:.4f} seconds"
        )

        print(
            f"PPO load time  : {model_load_time:.4f} seconds"
        )

        print(
            f"Pipeline time  : {execution_time:.4f} seconds"
        )

        print(
            f"Total time     : {total_time:.4f} seconds"
        )

        print(
            "\nFull report saved to:"
        )

        print(
            report_path.resolve()
        )

        return {

            "success":
                success,

            "source":
                source,

            "destination":
                destination,

            "amount":
                amount,

            "eta":
                eta,

            "top_k":
                top_k,

            "candidate_count":
                candidate_count,

            "usable_candidate_count":
                usable_candidate_count,

            "bucket_size":
                bucket_size,

            "attempt_count":
                attempt_count,

            "partial_backtrack_count":
                partial_backtrack_count,

            "successful_bucket_rank":
                successful_bucket_rank,

            "fee":
                fee,

            "delay":
                delay,

            "carbon":
                carbon,

            "path":
                final_path,

            "hops":
                max(
                    0,
                    path_length - 1
                ),

            "training_time":
                training_time,

            "model_load_time":
                model_load_time,

            "prediction_time":
                prediction_time,

            "pipeline_time":
                execution_time,

            "total_time":
                total_time,

            "execution_mode":
                (
                    "FAST VALIDATION"
                    if FAST_VALIDATION
                    else "NORMAL TRAINING"
                ),

        }

    except Exception as exc:

        write_runtime_error(
            exc,
            report
        )

        print(
            "\n============================================================"
        )

        print(
            "END-TO-END EXECUTION FAILED"
        )

        print(
            "============================================================"
        )

        print(
            f"Step     : {CURRENT_STEP}/{TOTAL_STEPS}"
        )

        print(
            f"Progress : {CURRENT_PROGRESS:.2f}%"
        )

        print(
            f"Stage    : {CURRENT_STAGE}"
        )

        print(
            f"Error    : {type(exc).__name__}: {exc}"
        )

        print(
            "\nFull traceback:"
        )

        traceback.print_exc()

        print(
            "\nDetailed error report saved to:"
        )

        print(
            REPORT_FILE.resolve()
        )

        raise

    finally:

        stop_heartbeat()

        if runtime_env is not None:

            try:

                runtime_env.close()

            except Exception as close_exc:

                print(
                    "\nWarning: runtime environment "
                    f"could not be closed cleanly: {close_exc}"
                )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        sys.exit(1)