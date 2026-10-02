# main2.py
#
# RL + Bucket - FAST 10 MINUTE TEST MODE
# ============================================================
#
#       50 candidate routes generated
#                    |
#                    v
#          diversify / select 30
#                    |
#                    v
#                 PPO -> eta
#                    |
#                    v
#             candidate ranking
#                    |
#                    v
#               best 5 routes
#                    |
#                    v
#                  Bucket
#                    |
#             +------+------+------+------+ 
#             |      |      |      |
#            #1     #2     #3     #4     #5
#             |
#             v
#          Payment
#             |
#          Success?
#
# ============================================================


# ============================================================
# STANDARD LIBRARIES
# ============================================================

import copy
import os
import random
import time
import yaml
import numpy as np
import networkx as nx
import traceback
import sys
import threading

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass


print("=" * 68, flush=True)
print("[MAIN2] Script started", flush=True)
print(f"[MAIN2] Project root: {PROJECT_ROOT}", flush=True)
print("=" * 68, flush=True)


# ============================================================
# IMPORTANT ENVIRONMENT SETTINGS
# ============================================================

# FAST MODE
#
# این مقدار باعث می‌شود train.py نیز در صورت پشتیبانی،
# حالت سریع را فعال کند.
#
os.environ["RL_FAST_TRAINING"] = "1"


# ============================================================
# TORCH
# ============================================================

try:
    import torch
except Exception:
    print("[MAIN2] ERROR: Could not import torch.", flush=True)
    traceback.print_exc()
    raise


# ============================================================
# GLOBAL RUNTIME STATE
# ============================================================

REPORT_FILE = PROJECT_ROOT / "report_main2.txt"

RUN_START_TIME = time.time()

CURRENT_STAGE = "Program startup"

CURRENT_PROGRESS = 0.0

CURRENT_STEP = 0

CURRENT_STEP_NAME = "Program startup"

PROGRESS_LOCK = threading.Lock()

HEARTBEAT_THREAD = None

HEARTBEAT_STOP_EVENT = threading.Event()


# ============================================================
# PPO TRAINING RUNTIME STATE
# ============================================================

PPO_TOTAL_TIMESTEPS = 0

PPO_CURRENT_TIMESTEP = 0

PPO_TRAINING_PROGRESS = 0.0

PPO_TRAINING_START = None

PPO_TRAINING_ETA = None

PPO_TRAINING_FPS = 0.0


# ============================================================
# FAST EXPERIMENT CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# این نسخه برای تست سریع کل pipeline ساخته شده است.
#
# مقادیر اصلی قبلی:
#
# TRAIN_TRANSACTION_COUNT = 50
# EVAL_TRANSACTION_COUNT  = 10
# ROUTE_POOL_SIZE         = 50
# ROUTE_GENERATION_POOL   = 250
# PPO                     = 20000
#
# مقادیر FAST:
#
# TRAIN_TRANSACTION_COUNT = 10
# EVAL_TRANSACTION_COUNT  = 5
# ROUTE_POOL_SIZE         = 30
# ROUTE_GENERATION_POOL   = 50
# PPO                     = 500
#
# ------------------------------------------------------------

FAST_TRAINING = True

MODEL_NAME = "fast_end_to_end_30_routes"

EVAL_TRANSACTION_COUNT = 5

TRAIN_TRANSACTION_COUNT = 10

ROUTE_POOL_SIZE = 30

ROUTE_GENERATION_POOL = 50

BUCKET_SIZE = 5

MAX_ROUTE_HOPS = 20

PREFERRED_ROUTE_LENGTHS = (
    10,
    20,
)

FAILURE_RATE = 0.06

TOTAL_STEPS = 10

# PPO fast configuration
FAST_TOTAL_TIMESTEPS = 500

FAST_N_STEPS = 64

FAST_BATCH_SIZE = 32


# ============================================================
# PROGRESS CONFIGURATION
# ============================================================

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
# PROGRESS HELPERS
# ============================================================

def _format_elapsed(seconds):

    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(seconds % 60)

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
            CURRENT_STEP = int(step)

        if step_name is not None:
            CURRENT_STEP_NAME = str(step_name)

        current_progress = CURRENT_PROGRESS

        current_step = CURRENT_STEP

        current_step_name = CURRENT_STEP_NAME

    elapsed = _format_elapsed(
        time.time() - RUN_START_TIME
    )

    print(
        "\n------------------------------------------------------------",
        flush=True
    )

    print(
        f"OVERALL PROGRESS : {current_progress:6.2f}%",
        flush=True
    )

    print(
        f"CURRENT STEP     : STEP {current_step}",
        flush=True
    )

    print(
        f"CURRENT STAGE    : {current_step_name}",
        flush=True
    )

    print(
        f"ELAPSED TIME     : {elapsed}",
        flush=True
    )

    if message:

        print(
            f"STATUS           : {message}",
            flush=True
        )

    print(
        "------------------------------------------------------------",
        flush=True
    )


# ============================================================
# HEARTBEAT
# ============================================================

def _heartbeat_worker():

    while not HEARTBEAT_STOP_EVENT.wait(30.0):

        (
            progress,
            step,
            step_name,
            stage
        ) = _progress_snapshot()

        elapsed = _format_elapsed(
            time.time() - RUN_START_TIME
        )

        print(
            "\n[PROGRESS MONITOR]",
            flush=True
        )

        print(
            f"Step             : {step}/{TOTAL_STEPS}",
            flush=True
        )

        print(
            f"Stage            : {step_name}",
            flush=True
        )

        print(
            f"Overall Progress : {progress:.2f}%",
            flush=True
        )

        print(
            f"Elapsed          : {elapsed}",
            flush=True
        )

        if (
            step == 4
            and PPO_TOTAL_TIMESTEPS > 0
        ):

            print(
                f"PPO Timestep     : "
                f"{PPO_CURRENT_TIMESTEP} / "
                f"{PPO_TOTAL_TIMESTEPS}",
                flush=True
            )

            print(
                f"PPO Progress     : "
                f"{PPO_TRAINING_PROGRESS:.2f}%",
                flush=True
            )

            print(
                f"PPO FPS          : "
                f"{PPO_TRAINING_FPS:.4f}",
                flush=True
            )

            if PPO_TRAINING_ETA is not None:

                print(
                    f"PPO ETA          : "
                    f"{_format_elapsed(PPO_TRAINING_ETA)}",
                    flush=True
                )

            print(
                f"PPO Remaining    : "
                f"{max(0, PPO_TOTAL_TIMESTEPS - PPO_CURRENT_TIMESTEP)}",
                flush=True
            )

            print(
                "Status           : PPO TRAINING",
                flush=True
            )

        else:

            print(
                "Status           : STILL RUNNING",
                flush=True
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

    global HEARTBEAT_THREAD

    HEARTBEAT_STOP_EVENT.set()

    if (
        HEARTBEAT_THREAD is not None
        and HEARTBEAT_THREAD.is_alive()
    ):

        HEARTBEAT_THREAD.join(
            timeout=1.0
        )


# ============================================================
# STAGE HELPERS
# ============================================================

def begin_step(
    step,
    name,
    report=None,
    message=None
):

    global CURRENT_STAGE

    CURRENT_STAGE = str(name)

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

    CURRENT_STAGE = str(name)

    if (
        int(step) == 4
        and PPO_TOTAL_TIMESTEPS > 0
        and PPO_TRAINING_PROGRESS >= 100.0
    ):

        progress = 50.0

    else:

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
# REPORT WRITER
# ============================================================

class ReportWriter:

    def __init__(
        self,
        path=None
    ):

        if path is None:
            path = REPORT_FILE

        self.path = Path(path)

        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path

        self.lines = []

    def add(
        self,
        *texts
    ):

        if not texts:

            self.lines.append("")

            return

        for text in texts:

            self.lines.append(
                str(text)
            )

    def section(
        self,
        title
    ):

        self.add("")

        self.add(
            "=" * 68
        )

        self.add(
            title
        )

        self.add(
            "=" * 68
        )

    def item(
        self,
        key,
        value
    ):

        self.add(
            f"{key:<38}: {value}"
        )

    def save(self):

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.path.write_text(
            "\n".join(
                self.lines
            ) + "\n",
            encoding="utf-8"
        )

        return self.path


# ============================================================
# BOOTSTRAP REPORT
# ============================================================

def initialize_bootstrap_report():

    try:

        REPORT_FILE.write_text(

            "RL + BUCKET MAIN2 REPORT\n"
            "============================================================\n"
            f"Run started: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Mode: FAST TRAINING\n"
            "Status: STARTED\n\n",

            encoding="utf-8"

        )

    except Exception as exc:

        print(
            f"[MAIN2] Warning: could not initialize report: {exc}",
            flush=True
        )


initialize_bootstrap_report()


# ============================================================
# PROJECT IMPORTS
# ============================================================

try:

    CURRENT_STAGE = "Importing Network modules"

    from Network.graph_builder import (
        LNGraphBuilder
    )

    CURRENT_STAGE = "Importing Simulation modules"

    from Simulation.failure_model import (
        assign_failure_probabilities,
        FailureModel,
    )

    from Simulation.transaction_generator import (
        generate_transactions,
    )

    from Simulation.network_dynamics import (
        NetworkDynamics,
    )

    CURRENT_STAGE = "Importing RL modules"

    from RL.train import (
        train_agent
    )

    import RL.train as train_module

    from RL.environment import (
        RoutingEnv
    )

    from RL.ppo_agent import (
        build_ppo
    )

    CURRENT_STAGE = "Importing Pathfinding modules"

    from Pathfinding.heuristics import (
        lnd_cost
    )

    import Pathfinding.top_k_paths as top_k_module

    import RL.environment as environment_module

except Exception as import_exc:

    print(
        "\n============================================================",
        flush=True
    )

    print(
        "[MAIN2] PROJECT IMPORT FAILED",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    print(
        f"Stage: {CURRENT_STAGE}",
        flush=True
    )

    print(
        f"Error: {type(import_exc).__name__}: {import_exc}",
        flush=True
    )

    traceback.print_exc()

    try:

        report = ReportWriter()

        report.section(
            "IMPORT ERROR"
        )

        report.item(
            "Project root",
            PROJECT_ROOT
        )

        report.item(
            "Current stage",
            CURRENT_STAGE
        )

        report.item(
            "Exception type",
            type(import_exc).__name__
        )

        report.item(
            "Exception",
            str(import_exc)
        )

        report.add("")
        report.add("Traceback:")

        report.add(
            traceback.format_exc()
        )

        report.save()

    except Exception:
        pass

    raise


# ============================================================
# PPO PROGRESS BRIDGE
# ============================================================

try:

    OriginalPPOProgressCallback = (
        train_module.PPOProgressCallback
    )

    class Main2PPOProgressCallback(
        OriginalPPOProgressCallback
    ):

        def _on_training_start(self):

            global PPO_TOTAL_TIMESTEPS
            global PPO_CURRENT_TIMESTEP
            global PPO_TRAINING_PROGRESS
            global PPO_TRAINING_START
            global PPO_TRAINING_ETA
            global PPO_TRAINING_FPS
            global CURRENT_STAGE
            global CURRENT_PROGRESS

            result = super()._on_training_start()

            PPO_TOTAL_TIMESTEPS = int(
                self.total_timesteps_target
            )

            PPO_CURRENT_TIMESTEP = 0

            PPO_TRAINING_PROGRESS = 0.0

            PPO_TRAINING_START = time.time()

            PPO_TRAINING_ETA = None

            PPO_TRAINING_FPS = 0.0

            CURRENT_STAGE = (
                "STEP 4 - PPO TRAINING"
            )

            CURRENT_PROGRESS = 40.0

            print(
                "\n" + "=" * 68,
                flush=True
            )

            print(
                "MAIN2 - FAST PPO TRAINING STARTED",
                flush=True
            )

            print(
                "=" * 68,
                flush=True
            )

            print(
                f"Target timesteps : "
                f"{PPO_TOTAL_TIMESTEPS}",
                flush=True
            )

            print(
                "FAST MODE        : ENABLED",
                flush=True
            )

            print(
                "=" * 68,
                flush=True
            )

            return result

        def _on_step(self):

            global PPO_TOTAL_TIMESTEPS
            global PPO_CURRENT_TIMESTEP
            global PPO_TRAINING_PROGRESS
            global PPO_TRAINING_ETA
            global PPO_TRAINING_FPS
            global CURRENT_STAGE
            global CURRENT_PROGRESS

            result = super()._on_step()

            PPO_CURRENT_TIMESTEP = int(
                self.num_timesteps
            )

            PPO_TOTAL_TIMESTEPS = max(
                1,
                int(
                    self.total_timesteps_target
                )
            )

            PPO_TRAINING_PROGRESS = min(
                100.0,
                (
                    PPO_CURRENT_TIMESTEP
                    /
                    PPO_TOTAL_TIMESTEPS
                    *
                    100.0
                )
            )

            if PPO_TRAINING_START is not None:

                elapsed = (
                    time.time()
                    -
                    PPO_TRAINING_START
                )

            else:

                elapsed = 0.0

            if elapsed > 0:

                PPO_TRAINING_FPS = (
                    PPO_CURRENT_TIMESTEP
                    /
                    elapsed
                )

            else:

                PPO_TRAINING_FPS = 0.0

            remaining = max(
                0,
                PPO_TOTAL_TIMESTEPS
                -
                PPO_CURRENT_TIMESTEP
            )

            if PPO_TRAINING_FPS > 0:

                PPO_TRAINING_ETA = (
                    remaining
                    /
                    PPO_TRAINING_FPS
                )

            else:

                PPO_TRAINING_ETA = None

            CURRENT_STAGE = (
                "STEP 4 - PPO TRAINING"
            )

            CURRENT_PROGRESS = (
                40.0
                +
                PPO_TRAINING_PROGRESS * 0.10
            )

            return result

        def _on_training_end(self):

            global PPO_CURRENT_TIMESTEP
            global PPO_TOTAL_TIMESTEPS
            global PPO_TRAINING_PROGRESS
            global CURRENT_STAGE
            global CURRENT_PROGRESS

            super()._on_training_end()

            PPO_CURRENT_TIMESTEP = int(
                self.num_timesteps
            )

            PPO_TOTAL_TIMESTEPS = max(
                1,
                int(
                    self.total_timesteps_target
                )
            )

            PPO_TRAINING_PROGRESS = min(
                100.0,
                (
                    PPO_CURRENT_TIMESTEP
                    /
                    PPO_TOTAL_TIMESTEPS
                    *
                    100.0
                )
            )

            CURRENT_STAGE = (
                "STEP 4 - PPO TRAINING"
            )

            CURRENT_PROGRESS = 50.0

            print(
                "\n" + "=" * 68,
                flush=True
            )

            print(
                "MAIN2 - FAST PPO TRAINING FINISHED",
                flush=True
            )

            print(
                "=" * 68,
                flush=True
            )

            print(
                f"Final timestep : "
                f"{PPO_CURRENT_TIMESTEP} / "
                f"{PPO_TOTAL_TIMESTEPS}",
                flush=True
            )

            print(
                f"Training       : "
                f"{PPO_TRAINING_PROGRESS:.2f}%",
                flush=True
            )

            print(
                f"Training FPS   : "
                f"{PPO_TRAINING_FPS:.4f}",
                flush=True
            )

            print(
                "=" * 68,
                flush=True
            )


    train_module.PPOProgressCallback = (
        Main2PPOProgressCallback
    )

    print(
        "[MAIN2] PPO progress bridge installed.",
        flush=True
    )

except Exception as callback_exc:

    print(
        "[MAIN2] WARNING: PPO progress bridge could not be installed.",
        flush=True
    )

    print(
        f"[MAIN2] {type(callback_exc).__name__}: "
        f"{callback_exc}",
        flush=True
    )

    OriginalPPOProgressCallback = None


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CONFIG
# ============================================================

def load_cfg():

    possible_paths = [

        PROJECT_ROOT / "Configs" / "config.yaml",

        PROJECT_ROOT / "configs" / "config.yaml",

    ]

    config_path = None

    for candidate in possible_paths:

        if candidate.exists():

            config_path = candidate

            break

    if config_path is None:

        raise FileNotFoundError(

            "Configuration file not found.\n"
            "Expected one of:\n"
            +
            "\n".join(
                str(path)
                for path in possible_paths
            )

        )

    print(
        f"[MAIN2] Config: {config_path}",
        flush=True
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
# GML -> NETWORKX
# ============================================================

def geo_to_json(input_file):

    input_file = Path(input_file)

    print(
        f"[MAIN2] Reading snapshot: {input_file}",
        flush=True
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

    if graph.is_multigraph():

        for (
            source,
            target,
            key,
            attributes
        ) in graph.edges(
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

            for name, value in attributes.items():

                edge_data[name] = value

            data["edges"].append(
                edge_data
            )

    else:

        for (
            source,
            target,
            attributes
        ) in graph.edges(
            data=True
        ):

            edge_data = {

                "source":
                    str(source),

                "target":
                    str(target)

            }

            for name, value in attributes.items():

                edge_data[name] = value

            data["edges"].append(
                edge_data
            )

    print(
        f"[MAIN2] Snapshot read: "
        f"{len(data['nodes'])} nodes / "
        f"{len(data['edges'])} edges",
        flush=True
    )

    return data


# ============================================================
# UTILITY
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:

        if value is None:
            return float(default)

        value = float(value)

        if not np.isfinite(value):
            return float(default)

        return value

    except Exception:

        return float(default)


def safe_int(
    value,
    default=0
):

    try:

        if value is None:
            return int(default)

        return int(value)

    except Exception:

        return int(default)


def format_metric(
    value,
    digits=4
):

    if value is None:
        return "N/A"

    try:

        return (
            f"{float(value):.{digits}f}"
        )

    except Exception:

        return str(value)


# ============================================================
# ROUTE DIVERSIFICATION
# ============================================================

def diversify_routes(
    candidates,
    target_count=ROUTE_POOL_SIZE
):

    if not candidates:
        return []

    candidates = list(candidates)

    unique = {}

    for candidate in candidates:

        if not isinstance(
            candidate,
            dict
        ):
            continue

        path = candidate.get(
            "path",
            []
        )

        if not path:
            continue

        try:

            identity = tuple(path)

        except Exception:

            continue

        if identity not in unique:

            unique[identity] = candidate

    candidates = list(
        unique.values()
    )

    if not candidates:
        return []

    by_hop = {}

    for candidate in candidates:

        path = candidate.get(
            "path",
            []
        )

        default_hops = max(
            0,
            len(path) - 1
        )

        hops = safe_int(
            candidate.get(
                "hop_count",
                default_hops
            ),
            default_hops
        )

        by_hop.setdefault(
            hops,
            []
        ).append(
            candidate
        )

    for hops in by_hop:

        by_hop[hops].sort(
            key=lambda item: (

                safe_float(
                    item.get(
                        "cost"
                    ),
                    float("inf")
                ),

                safe_float(
                    item.get(
                        "total_fee"
                    ),
                    float("inf")
                ),

                safe_float(
                    item.get(
                        "total_delay"
                    ),
                    float("inf")
                )

            )
        )

    selected = []

    used_ids = set()

    for hop in PREFERRED_ROUTE_LENGTHS:

        if len(selected) >= target_count:
            break

        if hop not in by_hop:
            continue

        for candidate in by_hop[hop]:

            if len(selected) >= target_count:
                break

            identity = id(candidate)

            if identity in used_ids:
                continue

            selected.append(candidate)

            used_ids.add(identity)

    remaining_hops = sorted(
        by_hop.keys()
    )

    while len(selected) < target_count:

        changed = False

        for hop in remaining_hops:

            if len(selected) >= target_count:
                break

            bucket = by_hop[hop]

            while bucket:

                candidate = bucket.pop(0)

                identity = id(candidate)

                if identity in used_ids:
                    continue

                selected.append(candidate)

                used_ids.add(identity)

                changed = True

                break

        if not changed:
            break

    if len(selected) < target_count:

        remaining = [

            candidate

            for candidate in candidates

            if id(candidate) not in used_ids

        ]

        remaining.sort(
            key=lambda item: (

                safe_float(
                    item.get(
                        "cost"
                    ),
                    float("inf")
                ),

                safe_int(
                    item.get(
                        "hop_count"
                    )
                )

            )
        )

        for candidate in remaining:

            if len(selected) >= target_count:
                break

            selected.append(candidate)

    return selected[:target_count]


# ============================================================
# FAST ROUTE POOL GENERATOR
# ============================================================

def generate_50_route_pool(
    env,
    tx,
    eta
):

    """
    FAST MODE:

    Generate up to 50 candidate routes and retain
    up to 30 diversified routes.
    """

    if tx is None:

        raise ValueError(
            "Transaction cannot be None."
        )

    eta = env._validate_eta(
        eta
    )

    original_default_k = (
        top_k_module.DEFAULT_K
    )

    try:

        top_k_module.DEFAULT_K = (
            ROUTE_GENERATION_POOL
        )

        candidates = top_k_module.top_k_paths(

            G=env.G,

            source=tx.source,

            target=tx.destination,

            amount=tx.amount,

            heuristic_fn=env.heuristic_fn,

            eta=eta,

            k=ROUTE_GENERATION_POOL,

            max_hops=MAX_ROUTE_HOPS,

            lambda_h=env.lambda_h,

        )

    finally:

        top_k_module.DEFAULT_K = (
            original_default_k
        )

    if candidates is None:

        raise RuntimeError(
            "top_k_paths returned None."
        )

    if not isinstance(
        candidates,
        (list, tuple)
    ):

        raise TypeError(
            "top_k_paths must return list/tuple."
        )

    candidates = list(
        candidates
    )

    if not candidates:
        return []

    selected = diversify_routes(
        candidates,
        target_count=ROUTE_POOL_SIZE
    )

    return selected


# ============================================================
# PATCH ROUTING ENV
# ============================================================

def install_50_route_runtime():

    def _generate_candidates_50(
        self,
        tx,
        eta
    ):

        candidates = generate_50_route_pool(
            self,
            tx,
            eta
        )

        if candidates is None:

            raise RuntimeError(
                "FAST route generator returned None."
            )

        return list(candidates)

    RoutingEnv._generate_candidates = (
        _generate_candidates_50
    )

    environment_module.RoutingEnv._generate_candidates = (
        _generate_candidates_50
    )

    print(
        "\n============================================================",
        flush=True
    )

    print(
        "MAIN2 FAST ROUTING PATCH INSTALLED",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    print(
        f"Generation pool        : "
        f"{ROUTE_GENERATION_POOL}",
        flush=True
    )

    print(
        f"Retained route pool    : "
        f"{ROUTE_POOL_SIZE}",
        flush=True
    )

    print(
        f"Bucket size            : "
        f"{BUCKET_SIZE}",
        flush=True
    )

    print(
        f"Maximum route hops     : "
        f"{MAX_ROUTE_HOPS}",
        flush=True
    )

    print(
        "Preferred route lengths: 10 / 20 hops",
        flush=True
    )

    print(
        "PPO control            : eta",
        flush=True
    )

    print(
        "Bucket selection       : best 5 after ranking",
        flush=True
    )

    print(
        "FAST MODE              : ENABLED",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )


# ============================================================
# RUNTIME CONFIGURATION
# ============================================================

def build_experiment_cfg(cfg):

    cfg = copy.deepcopy(cfg)

    graph_cfg = cfg.setdefault(
        "graph",
        {}
    )

    graph_cfg["max_hops"] = (
        MAX_ROUTE_HOPS
    )

    graph_cfg["top_k_paths"] = (
        BUCKET_SIZE
    )

    graph_cfg["top_k"] = (
        BUCKET_SIZE
    )

    bucket_cfg = cfg.setdefault(
        "bucket",
        {}
    )

    bucket_cfg["max_candidates"] = (
        BUCKET_SIZE
    )

    bucket_cfg["enable_backtracking"] = True

    simulation_cfg = cfg.setdefault(
        "simulation",
        {}
    )

    simulation_cfg["channel_failure_rate"] = (
        FAILURE_RATE
    )

    simulation_cfg["node_failure_rate"] = (
        FAILURE_RATE / 2.0
    )

    simulation_cfg["node_failure_probability"] = (
        FAILURE_RATE / 2.0
    )

    # --------------------------------------------------------
    # PPO FAST CONFIGURATION
    # --------------------------------------------------------

    rl_cfg = cfg.setdefault(
        "rl",
        {}
    )

    if FAST_TRAINING:

        # مهم:
        # این مقادیر عمداً مقدار config.yaml را override
        # می‌کنند تا اجرای FAST واقعاً سریع باشد.

        rl_cfg["total_timesteps"] = (
            FAST_TOTAL_TIMESTEPS
        )

        rl_cfg["n_steps"] = (
            FAST_N_STEPS
        )

        rl_cfg["batch_size"] = (
            FAST_BATCH_SIZE
        )

    else:

        rl_cfg["total_timesteps"] = int(
            rl_cfg.get(
                "total_timesteps",
                20000
            )
        )

        rl_cfg["n_steps"] = min(
            int(
                rl_cfg.get(
                    "n_steps",
                    256
                )
            ),
            rl_cfg["total_timesteps"]
        )

        rl_cfg["batch_size"] = min(
            int(
                rl_cfg.get(
                    "batch_size",
                    64
                )
            ),
            rl_cfg["n_steps"]
        )

    return cfg


# ============================================================
# TRAINING TRANSACTIONS
# ============================================================

def generate_training_transactions(
    G,
    cfg,
    seed
):

    print(
        "\n============================================================",
        flush=True
    )

    print(
        "GENERATING FAST TRAINING TRANSACTIONS",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    simulation_cfg = cfg["simulation"]

    transactions = generate_transactions(

        G,

        TRAIN_TRANSACTION_COUNT,

        seed,

        simulation_cfg["min_amount"],

        simulation_cfg["max_amount"]

    )

    if not transactions:

        raise RuntimeError(
            "Training transaction generation returned no data."
        )

    for index, transaction in enumerate(
        transactions,
        start=1
    ):

        try:

            transaction.tx_id = index

        except Exception:

            pass

    print(
        f"Training transactions : "
        f"{len(transactions)}",
        flush=True
    )

    return transactions


# ============================================================
# EVALUATION TRANSACTIONS
# ============================================================

def generate_evaluation_transactions(
    G,
    cfg,
    seed
):

    simulation_cfg = cfg["simulation"]

    transactions = generate_transactions(

        G,

        EVAL_TRANSACTION_COUNT,

        seed,

        simulation_cfg["min_amount"],

        simulation_cfg["max_amount"]

    )

    if len(transactions) != EVAL_TRANSACTION_COUNT:

        raise RuntimeError(

            "Expected exactly "
            f"{EVAL_TRANSACTION_COUNT} "
            "evaluation transactions, received "
            f"{len(transactions)}."

        )

    for index, transaction in enumerate(
        transactions,
        start=1
    ):

        try:

            transaction.tx_id = (
                100000 + index
            )

        except Exception:

            pass

    return transactions


# ============================================================
# EXTRACT RESULT
# ============================================================

def extract_result(
    env,
    transaction,
    reward,
    terminated,
    truncated,
    info,
    episode_index
):

    info = info or {}

    success = bool(
        info.get(
            "success",
            False
        )
    )

    bucket = getattr(
        env,
        "current_bucket",
        None
    )

    successful_bucket_rank = None

    successful_bucket_index = None

    bucket_attempts = None

    if bucket is not None:

        successful_bucket_rank = getattr(
            bucket,
            "selected_candidate_rank",
            None
        )

        successful_bucket_index = getattr(
            bucket,
            "selected_candidate_index",
            None
        )

        bucket_attempts = getattr(
            bucket,
            "attempts",
            None
        )

    if successful_bucket_rank is not None:

        try:

            successful_bucket_rank = int(
                successful_bucket_rank
            )

        except Exception:

            successful_bucket_rank = None

    if (
        successful_bucket_rank is not None
        and not (
            1 <= successful_bucket_rank <= BUCKET_SIZE
        )
    ):

        successful_bucket_rank = None

    path = info.get(
        "path",
        []
    )

    if path is None:
        path = []

    path = list(path)

    return {

        "episode_index":
            episode_index,

        "transaction_id":
            getattr(
                transaction,
                "tx_id",
                episode_index
            ),

        "source":
            str(
                getattr(
                    transaction,
                    "source",
                    ""
                )
            ),

        "destination":
            str(
                getattr(
                    transaction,
                    "destination",
                    ""
                )
            ),

        "amount":
            safe_float(
                getattr(
                    transaction,
                    "amount",
                    0.0
                )
            ),

        "success":
            success,

        "terminated":
            bool(terminated),

        "truncated":
            bool(truncated),

        "eta":
            info.get("eta"),

        "candidate_count":
            safe_int(
                info.get(
                    "candidate_path_count"
                )
            ),

        "usable_candidate_count":
            safe_int(
                info.get(
                    "usable_candidate_count"
                )
            ),

        "bucket_size":
            safe_int(
                info.get(
                    "bucket_size"
                )
            ),

        "attempt_count":
            safe_int(
                info.get(
                    "attempt_count"
                )
            ),

        "bucket_attempts":
            bucket_attempts,

        "successful_bucket_rank":
            successful_bucket_rank,

        "successful_bucket_index":
            successful_bucket_index,

        "partial_backtrack_count":
            safe_int(
                info.get(
                    "partial_backtrack_count"
                )
            ),

        "partial_backtrack_success":
            safe_int(
                info.get(
                    "partial_backtrack_success"
                )
            ),

        "full_reroute_count":
            safe_int(
                info.get(
                    "full_reroute_count"
                )
            ),

        "failure_probability":
            safe_float(
                info.get(
                    "failure_probability"
                )
            ),

        "fee":
            safe_float(
                info.get(
                    "fee"
                )
            ),

        "delay":
            safe_float(
                info.get(
                    "delay"
                )
            ),

        "carbon":
            safe_float(
                info.get(
                    "carbon"
                )
            ),

        "reward":
            safe_float(
                reward
            ),

        "path":
            path,

        "hops":
            max(
                0,
                len(path) - 1
            ),

        "reason":
            info.get(
                "reason",
                ""
            ),

    }


# ============================================================
# AGGREGATE RESULTS
# ============================================================

def aggregate_results(results):

    total = len(results)

    successes = sum(
        1
        for result in results
        if result["success"]
    )

    failures = total - successes

    def mean(
        key,
        only_success=False
    ):

        values = []

        for result in results:

            if (
                only_success
                and not result["success"]
            ):

                continue

            value = result.get(key)

            if value is None:
                continue

            try:

                value = float(value)

            except Exception:

                continue

            if np.isfinite(value):

                values.append(value)

        if not values:
            return None

        return float(
            np.mean(values)
        )

    rank_counts = {}

    for result in results:

        if not result["success"]:
            continue

        rank = result[
            "successful_bucket_rank"
        ]

        if rank is not None:

            rank_counts[rank] = (
                rank_counts.get(
                    rank,
                    0
                ) + 1
            )

    return {

        "total":
            total,

        "successes":
            successes,

        "failures":
            failures,

        "success_rate":
            (
                successes / total * 100.0
                if total
                else 0.0
            ),

        "failure_rate":
            (
                failures / total * 100.0
                if total
                else 0.0
            ),

        "average_attempts":
            mean("attempt_count"),

        "average_hops":
            mean(
                "hops",
                only_success=True
            ),

        "average_fee":
            mean(
                "fee",
                only_success=True
            ),

        "average_delay":
            mean(
                "delay",
                only_success=True
            ),

        "average_carbon":
            mean(
                "carbon",
                only_success=True
            ),

        "average_reward":
            mean("reward"),

        "average_eta":
            mean("eta"),

        "average_candidates":
            mean("candidate_count"),

        "average_bucket":
            mean("bucket_size"),

        "total_partial_backtracks":
            sum(
                result[
                    "partial_backtrack_count"
                ]
                for result in results
            ),

        "total_partial_backtrack_success":
            sum(
                result[
                    "partial_backtrack_success"
                ]
                for result in results
            ),

        "total_full_reroutes":
            sum(
                result[
                    "full_reroute_count"
                ]
                for result in results
            ),

        "successful_bucket_rank_counts":
            rank_counts,

    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(aggregate):

    print(
        "\n============================================================",
        flush=True
    )

    print(
        "MAIN2 FAST EVALUATION SUMMARY",
        flush=True
    )

    print(
        "============================================================",
        flush=True
    )

    print(
        f"Transactions           : "
        f"{aggregate['total']}",
        flush=True
    )

    print(
        f"Successful payments    : "
        f"{aggregate['successes']}",
        flush=True
    )

    print(
        f"Failed payments        : "
        f"{aggregate['failures']}",
        flush=True
    )

    print(
        f"Success rate           : "
        f"{aggregate['success_rate']:.2f}%",
        flush=True
    )

    print(
        f"Failure rate           : "
        f"{aggregate['failure_rate']:.2f}%",
        flush=True
    )

    print(
        f"Average attempts       : "
        f"{format_metric(aggregate['average_attempts'])}",
        flush=True
    )

    print(
        f"Average hops           : "
        f"{format_metric(aggregate['average_hops'])}",
        flush=True
    )

    print(
        f"Average fee            : "
        f"{format_metric(aggregate['average_fee'], 6)}",
        flush=True
    )

    print(
        f"Average delay          : "
        f"{format_metric(aggregate['average_delay'])}",
        flush=True
    )

    print(
        f"Average carbon         : "
        f"{format_metric(aggregate['average_carbon'], 6)}",
        flush=True
    )

    print(
        f"Average reward         : "
        f"{format_metric(aggregate['average_reward'], 6)}",
        flush=True
    )

    print(
        f"Average eta            : "
        f"{format_metric(aggregate['average_eta'], 6)}",
        flush=True
    )

    print(
        f"Average route pool     : "
        f"{format_metric(aggregate['average_candidates'])}",
        flush=True
    )

    print(
        f"Average Bucket size    : "
        f"{format_metric(aggregate['average_bucket'])}",
        flush=True
    )

    print(
        f"Partial backtracks     : "
        f"{aggregate['total_partial_backtracks']}",
        flush=True
    )

    print(
        f"Partial BT successes   : "
        f"{aggregate['total_partial_backtrack_success']}",
        flush=True
    )

    print(
        f"Full reroutes          : "
        f"{aggregate['total_full_reroutes']}",
        flush=True
    )

    print(
        "\nSuccessful Bucket candidate distribution:",
        flush=True
    )

    if aggregate["successful_bucket_rank_counts"]:

        for rank in sorted(
            aggregate[
                "successful_bucket_rank_counts"
            ]
        ):

            print(
                f"  Bucket candidate #{rank}: "
                f"{aggregate['successful_bucket_rank_counts'][rank]}",
                flush=True
            )

    else:

        print(
            "  No successful Bucket candidate.",
            flush=True
        )

    print(
        "============================================================",
        flush=True
    )


# ============================================================
# EVALUATE ONE TRANSACTION
# ============================================================

def evaluate_transaction(
    model,
    G,
    transaction,
    cfg,
    seed,
    episode_index
):

    if model is None:

        raise RuntimeError(
            "PPO model is None. "
            "Training must complete successfully before evaluation."
        )

    evaluation_graph = copy.deepcopy(G)

    assign_failure_probabilities(
        evaluation_graph,
        FAILURE_RATE,
        seed
    )

    dynamics = NetworkDynamics(
        evaluation_graph
    )

    failure_model = FailureModel(
        seed=seed
    )

    runtime_cfg = copy.deepcopy(
        cfg
    )

    runtime_env = RoutingEnv(

        G=evaluation_graph,

        transactions=[
            transaction
        ],

        heuristic_fn=lnd_cost,

        config=runtime_cfg,

        mode="eval",

        failure_model=failure_model,

        network_dynamics=dynamics

    )

    try:

        model.set_env(
            runtime_env
        )

        observation, reset_info = (
            runtime_env.reset(
                seed=seed
            )
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

        step_start = time.time()

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = runtime_env.step(
            action
        )

        pipeline_time = (
            time.time()
            -
            step_start
        )

        result = extract_result(

            runtime_env,

            transaction,

            reward,

            terminated,

            truncated,

            info,

            episode_index

        )

        result["prediction_time"] = (
            prediction_time
        )

        result["pipeline_time"] = (
            pipeline_time
        )

        result["reset_info"] = (
            reset_info
        )

        return result

    finally:

        try:

            runtime_env.close()

        except Exception:

            pass


# ============================================================
# MAIN
# ============================================================

def main():

    global RUN_START_TIME
    global CURRENT_STAGE
    global CURRENT_STEP
    global CURRENT_PROGRESS

    global PPO_TOTAL_TIMESTEPS
    global PPO_CURRENT_TIMESTEP
    global PPO_TRAINING_PROGRESS
    global PPO_TRAINING_START
    global PPO_TRAINING_ETA
    global PPO_TRAINING_FPS

    RUN_START_TIME = time.time()

    CURRENT_STAGE = "Program startup"

    CURRENT_STEP = 0

    CURRENT_PROGRESS = 0.0

    PPO_TOTAL_TIMESTEPS = 0

    PPO_CURRENT_TIMESTEP = 0

    PPO_TRAINING_PROGRESS = 0.0

    PPO_TRAINING_START = None

    PPO_TRAINING_ETA = None

    PPO_TRAINING_FPS = 0.0

    start_heartbeat()

    report = ReportWriter()

    report.section(
        "RL + BUCKET MAIN2 - FAST EXPERIMENT"
    )

    report.item(
        "Project root",
        PROJECT_ROOT
    )

    report.item(
        "Run started",
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    report.item(
        "FAST TRAINING",
        FAST_TRAINING
    )

    report.item(
        "Route generation pool",
        ROUTE_GENERATION_POOL
    )

    report.item(
        "Route pool retained",
        ROUTE_POOL_SIZE
    )

    report.item(
        "Bucket size",
        BUCKET_SIZE
    )

    report.item(
        "Maximum hops",
        MAX_ROUTE_HOPS
    )

    report.item(
        "Training transactions",
        TRAIN_TRANSACTION_COUNT
    )

    report.item(
        "Evaluation transactions",
        EVAL_TRANSACTION_COUNT
    )

    report.item(
        "PPO timesteps",
        FAST_TOTAL_TIMESTEPS
    )

    report.save()

    model = None

    evaluation_results = []

    training_time = 0.0

    try:

        # ====================================================
        # STEP 0
        # ====================================================

        begin_step(
            0,
            "STEP 0 - LOAD CONFIGURATION",
            report,
            "Loading configuration..."
        )

        cfg = load_cfg()

        if "seed" not in cfg:

            raise KeyError(
                "The configuration file must contain 'seed'."
            )

        seed = int(
            cfg["seed"]
        )

        set_seed(seed)

        cfg = build_experiment_cfg(
            cfg
        )

        report.section(
            "STEP 0 - FAST CONFIGURATION"
        )

        report.item(
            "Seed",
            seed
        )

        report.item(
            "FAST mode",
            FAST_TRAINING
        )

        report.item(
            "Training transactions",
            TRAIN_TRANSACTION_COUNT
        )

        report.item(
            "Evaluation transactions",
            EVAL_TRANSACTION_COUNT
        )

        report.item(
            "Route generation pool",
            ROUTE_GENERATION_POOL
        )

        report.item(
            "Route pool",
            ROUTE_POOL_SIZE
        )

        report.item(
            "Bucket",
            BUCKET_SIZE
        )

        report.item(
            "Maximum hops",
            MAX_ROUTE_HOPS
        )

        report.item(
            "Failure rate",
            FAILURE_RATE
        )

        report.item(
            "PPO total timesteps",
            cfg["rl"]["total_timesteps"]
        )

        report.item(
            "PPO n_steps",
            cfg["rl"]["n_steps"]
        )

        report.item(
            "PPO batch size",
            cfg["rl"]["batch_size"]
        )

        complete_step(
            0,
            "STEP 0 - LOAD CONFIGURATION",
            report,
            "FAST CONFIGURATION READY"
        )


        # ====================================================
        # STEP 1
        # ====================================================

        begin_step(
            1,
            "STEP 1 - LOAD SNAPSHOT",
            report,
            "Loading Lightning snapshot..."
        )

        snapshot_path = (
            PROJECT_ROOT /
            "20190501.gml.geo"
        )

        if not snapshot_path.exists():

            raise FileNotFoundError(

                "Snapshot not found:\n"
                f"{snapshot_path}"

            )

        data = geo_to_json(
            snapshot_path
        )

        builder = LNGraphBuilder()

        G = builder.from_data(
            data
        )

        report.section(
            "STEP 1 - SNAPSHOT"
        )

        report.item(
            "Snapshot",
            snapshot_path.name
        )

        report.item(
            "Nodes",
            G.number_of_nodes()
        )

        report.item(
            "Channels",
            G.number_of_edges()
        )

        print(
            f"\nNodes    : "
            f"{G.number_of_nodes()}",
            flush=True
        )

        print(
            f"Channels : "
            f"{G.number_of_edges()}",
            flush=True
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

        begin_step(
            2,
            "STEP 2 - INSTALL FAST ROUTING PIPELINE",
            report,
            "Installing fast routing modification..."
        )

        install_50_route_runtime()

        report.section(
            "STEP 2 - FAST ROUTING"
        )

        report.item(
            "Generation pool",
            ROUTE_GENERATION_POOL
        )

        report.item(
            "Initial route pool",
            ROUTE_POOL_SIZE
        )

        report.item(
            "Bucket size",
            BUCKET_SIZE
        )

        report.item(
            "Max hops",
            MAX_ROUTE_HOPS
        )

        report.add("")

        report.add(
            "FAST mode reduces candidate generation "
            "from 250 to 50."
        )

        report.add(
            "FAST mode retains up to 30 diversified routes."
        )

        report.add(
            "Only five ranked routes enter the active Bucket."
        )

        complete_step(
            2,
            "STEP 2 - INSTALL FAST ROUTING PIPELINE",
            report,
            "FAST ROUTING PIPELINE INSTALLED"
        )


        # ====================================================
        # STEP 3
        # ====================================================

        begin_step(
            3,
            "STEP 3 - PREPARE TRAINING DATA",
            report,
            f"Generating {TRAIN_TRANSACTION_COUNT} training transactions..."
        )

        G_train = copy.deepcopy(G)

        assign_failure_probabilities(
            G_train,
            FAILURE_RATE,
            seed
        )

        training_transactions = (
            generate_training_transactions(
                G_train,
                cfg,
                seed + 100
            )
        )

        report.section(
            "STEP 3 - FAST TRAINING DATA"
        )

        report.item(
            "Training transactions",
            len(training_transactions)
        )

        report.item(
            "Failure rate",
            FAILURE_RATE
        )

        report.item(
            "Training graph nodes",
            G_train.number_of_nodes()
        )

        report.item(
            "Training graph channels",
            G_train.number_of_edges()
        )

        complete_step(
            3,
            "STEP 3 - PREPARE TRAINING DATA",
            report,
            f"{len(training_transactions)} TRAINING TRANSACTIONS READY"
        )


        # ====================================================
        # STEP 4
        # ====================================================

        begin_step(
            4,
            "STEP 4 - FAST PPO TRAINING",
            report,
            "Training FAST PPO..."
        )

        print(
            "\n============================================================",
            flush=True
        )

        print(
            "STEP 4 - FAST PPO TRAINING",
            flush=True
        )

        print(
            "============================================================",
            flush=True
        )

        print(
            "Training is ENABLED.",
            flush=True
        )

        print(
            "FAST MODE is ENABLED.",
            flush=True
        )

        print(
            f"Training transactions : "
            f"{len(training_transactions)}",
            flush=True
        )

        print(
            f"Route generation pool : "
            f"{ROUTE_GENERATION_POOL}",
            flush=True
        )

        print(
            f"Retained route pool   : "
            f"{ROUTE_POOL_SIZE}",
            flush=True
        )

        print(
            f"Bucket                : "
            f"{BUCKET_SIZE}",
            flush=True
        )

        print(
            f"Failure rate          : "
            f"{FAILURE_RATE}",
            flush=True
        )

        print(
            f"Maximum hops          : "
            f"{MAX_ROUTE_HOPS}",
            flush=True
        )

        configured_timesteps = int(
            cfg["rl"]["total_timesteps"]
        )

        PPO_TOTAL_TIMESTEPS = (
            configured_timesteps
        )

        PPO_CURRENT_TIMESTEP = 0

        PPO_TRAINING_PROGRESS = 0.0

        PPO_TRAINING_START = None

        PPO_TRAINING_ETA = None

        PPO_TRAINING_FPS = 0.0

        print(
            "\n------------------------------------------------------------",
            flush=True
        )

        print(
            "FAST PPO TRAINING CONFIGURATION",
            flush=True
        )

        print(
            "------------------------------------------------------------",
            flush=True
        )

        print(
            f"Total timesteps      : "
            f"{configured_timesteps}",
            flush=True
        )

        print(
            f"n_steps             : "
            f"{cfg['rl']['n_steps']}",
            flush=True
        )

        print(
            f"batch_size          : "
            f"{cfg['rl']['batch_size']}",
            flush=True
        )

        print(
            f"Training transactions: "
            f"{len(training_transactions)}",
            flush=True
        )

        print(
            f"Route generation     : "
            f"{ROUTE_GENERATION_POOL}",
            flush=True
        )

        print(
            f"Routes retained      : "
            f"{ROUTE_POOL_SIZE}",
            flush=True
        )

        print(
            f"Bucket size          : "
            f"{BUCKET_SIZE}",
            flush=True
        )

        print(
            "------------------------------------------------------------",
            flush=True
        )

        print(
            "هدف این مرحله: اجرای سریع کل pipeline.",
            flush=True
        )

        print(
            "این تنظیمات برای آزمایش نهایی پایان‌نامه نیست.",
            flush=True
        )

        print(
            "------------------------------------------------------------",
            flush=True
        )

        training_start = time.time()

        model = train_agent(

            G_train,

            training_transactions,

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

        if model is None:

            raise RuntimeError(
                "train_agent() returned None. "
                "PPO training did not produce a model."
            )

        if (
            PPO_TOTAL_TIMESTEPS > 0
            and PPO_CURRENT_TIMESTEP
            < PPO_TOTAL_TIMESTEPS
        ):

            PPO_CURRENT_TIMESTEP = (
                PPO_TOTAL_TIMESTEPS
            )

            PPO_TRAINING_PROGRESS = 100.0

        report.section(
            "STEP 4 - FAST PPO TRAINING RESULT"
        )

        report.item(
            "Training status",
            "COMPLETED"
        )

        report.item(
            "Training mode",
            "FAST"
        )

        report.item(
            "Training time",
            f"{training_time:.4f} seconds"
        )

        report.item(
            "PPO target timesteps",
            PPO_TOTAL_TIMESTEPS
        )

        report.item(
            "PPO completed timesteps",
            PPO_CURRENT_TIMESTEP
        )

        report.item(
            "PPO training progress",
            f"{PPO_TRAINING_PROGRESS:.2f}%"
        )

        report.item(
            "PPO training FPS",
            f"{PPO_TRAINING_FPS:.4f}"
        )

        report.item(
            "Training transactions",
            len(training_transactions)
        )

        report.item(
            "Route generation pool",
            ROUTE_GENERATION_POOL
        )

        report.item(
            "Retained route pool",
            ROUTE_POOL_SIZE
        )

        report.item(
            "Bucket size during training",
            BUCKET_SIZE
        )

        report.add("")

        report.add(
            "FAST mode is intended for pipeline validation."
        )

        report.add(
            "It should not be used as the final experimental "
            "configuration for thesis results."
        )

        complete_step(
            4,
            "STEP 4 - FAST PPO TRAINING",
            report,
            "FAST PPO TRAINING COMPLETED"
        )


        # ====================================================
        # STEP 5
        # ====================================================

        begin_step(
            5,
            "STEP 5 - GENERATE EVALUATION TRANSACTIONS",
            report,
            "Generating independent evaluation transactions..."
        )

        evaluation_transactions = (
            generate_evaluation_transactions(
                G,
                cfg,
                seed + 1000
            )
        )

        report.section(
            "STEP 5 - EVALUATION TRANSACTIONS"
        )

        report.item(
            "Evaluation transactions",
            len(evaluation_transactions)
        )

        report.item(
            "Policy",
            f"{EVAL_TRANSACTION_COUNT} independent transactions"
        )

        complete_step(
            5,
            "STEP 5 - GENERATE EVALUATION TRANSACTIONS",
            report,
            "EVALUATION TRANSACTIONS READY"
        )


        # ====================================================
        # STEP 6
        # ====================================================

        begin_step(
            6,
            "STEP 6 - EVALUATION CONFIGURATION",
            report,
            "Preparing dynamic network evaluation..."
        )

        report.section(
            "STEP 6 - FAST EVALUATION CONFIGURATION"
        )

        report.item(
            "Failure rate",
            FAILURE_RATE
        )

        report.item(
            "Route generation pool",
            ROUTE_GENERATION_POOL
        )

        report.item(
            "Routes retained",
            ROUTE_POOL_SIZE
        )

        report.item(
            "Bucket candidates",
            BUCKET_SIZE
        )

        report.item(
            "Max hops",
            MAX_ROUTE_HOPS
        )

        report.add("")

        report.add(
            "Every evaluation transaction gets its own graph, "
            "FailureModel and NetworkDynamics."
        )

        complete_step(
            6,
            "STEP 6 - EVALUATION CONFIGURATION",
            report,
            "EVALUATION CONFIGURATION READY"
        )


        # ====================================================
        # STEP 7
        # ====================================================

        begin_step(
            7,
            "STEP 7 - PPO ROUTING",
            report,
            "Running trained PPO on evaluation transactions..."
        )

        report.section(
            "STEP 7 - FAST PPO ROUTING"
        )

        report.add(
            "The trained PPO model is reused for every "
            "evaluation transaction."
        )

        report.add(
            "For each transaction PPO predicts eta."
        )

        report.add(
            f"The FAST route generator produces up to "
            f"{ROUTE_GENERATION_POOL} candidates."
        )

        report.add(
            f"Up to {ROUTE_POOL_SIZE} diversified routes "
            "are retained."
        )

        report.add(
            f"Only the best {BUCKET_SIZE} candidates "
            "enter Bucket."
        )

        complete_step(
            7,
            "STEP 7 - PPO ROUTING",
            report,
            "PPO ROUTING READY"
        )


        # ====================================================
        # STEP 8
        # ====================================================

        begin_step(
            8,
            "STEP 8 - BUCKET EXECUTION",
            report,
            "Testing selected Bucket routes..."
        )

        report.section(
            "STEP 8 - BUCKET EXECUTION"
        )

        report.add(
            "The five Bucket candidates are not forced "
            "to succeed at a predetermined position."
        )

        report.add(
            "The first candidate may succeed."
        )

        report.add(
            "A later candidate may succeed after failure."
        )

        report.add(
            "All candidates may fail."
        )

        complete_step(
            8,
            "STEP 8 - BUCKET EXECUTION",
            report,
            "BUCKET EXECUTION READY"
        )


        # ====================================================
        # STEP 9
        # ====================================================

        begin_step(
            9,
            "STEP 9 - MULTI-TRANSACTION EVALUATION",
            report,
            "Evaluating all transactions..."
        )

        print(
            "\n============================================================",
            flush=True
        )

        print(
            "STEP 9 - FAST EVALUATION",
            flush=True
        )

        print(
            "============================================================",
            flush=True
        )

        for episode_index, transaction in enumerate(
            evaluation_transactions,
            start=1
        ):

            episode_start = time.time()

            evaluation_seed = (
                seed
                +
                2000
                +
                episode_index
            )

            print(
                f"\n[MAIN2] Evaluating transaction "
                f"{episode_index}/{EVAL_TRANSACTION_COUNT}...",
                flush=True
            )

            result = evaluate_transaction(

                model,

                G,

                transaction,

                cfg,

                evaluation_seed,

                episode_index

            )

            evaluation_results.append(
                result
            )

            print(
                "\n------------------------------------------------------------",
                flush=True
            )

            print(
                f"TRANSACTION "
                f"{episode_index}/{EVAL_TRANSACTION_COUNT}",
                flush=True
            )

            print(
                "------------------------------------------------------------",
                flush=True
            )

            print(
                f"Source       : "
                f"{result['source']}",
                flush=True
            )

            print(
                f"Destination  : "
                f"{result['destination']}",
                flush=True
            )

            print(
                f"Amount       : "
                f"{result['amount']}",
                flush=True
            )

            print(
                f"PPO eta      : "
                f"{result['eta']}",
                flush=True
            )

            print(
                f"Candidates   : "
                f"{result['candidate_count']}",
                flush=True
            )

            print(
                f"Usable       : "
                f"{result['usable_candidate_count']}",
                flush=True
            )

            print(
                f"Bucket       : "
                f"{result['bucket_size']}",
                flush=True
            )

            print(
                f"Attempts     : "
                f"{result['attempt_count']}",
                flush=True
            )

            print(
                f"Partial BT   : "
                f"{result['partial_backtrack_count']}",
                flush=True
            )

            print(
                f"Payment      : "
                f"{'SUCCESS' if result['success'] else 'FAILED'}",
                flush=True
            )

            if result["successful_bucket_rank"] is not None:

                print(
                    f"Successful route : Bucket candidate #"
                    f"{result['successful_bucket_rank']}",
                    flush=True
                )

            else:

                if result["success"]:

                    print(
                        "Successful route : "
                        "Partial-backtrack/retry path "
                        "without explicit Bucket rank",
                        flush=True
                    )

                else:

                    print(
                        "Successful route : NONE",
                        flush=True
                    )

            print(
                f"Hops         : "
                f"{result['hops']}",
                flush=True
            )

            print(
                f"Reward       : "
                f"{result['reward']:.6f}",
                flush=True
            )

            print(
                f"Pipeline time: "
                f"{result['pipeline_time']:.4f} s",
                flush=True
            )

            episode_elapsed = (
                time.time()
                -
                episode_start
            )

            report.section(
                f"TRANSACTION "
                f"{episode_index}/"
                f"{EVAL_TRANSACTION_COUNT}"
            )

            report.item(
                "Transaction ID",
                result["transaction_id"]
            )

            report.item(
                "Source",
                result["source"]
            )

            report.item(
                "Destination",
                result["destination"]
            )

            report.item(
                "Amount",
                result["amount"]
            )

            report.item(
                "PPO eta",
                result["eta"]
            )

            report.item(
                "Candidate pool",
                result["candidate_count"]
            )

            report.item(
                "Usable candidates",
                result["usable_candidate_count"]
            )

            report.item(
                "Bucket size",
                result["bucket_size"]
            )

            report.item(
                "Attempts",
                result["attempt_count"]
            )

            report.item(
                "Payment",
                (
                    "SUCCESS"
                    if result["success"]
                    else "FAILED"
                )
            )

            report.item(
                "Successful Bucket rank",
                (
                    result["successful_bucket_rank"]
                    if result["successful_bucket_rank"] is not None
                    else (
                        "NONE"
                        if not result["success"]
                        else "NOT EXPLICIT"
                    )
                )
            )

            report.item(
                "Hops",
                result["hops"]
            )

            report.item(
                "Partial backtracks",
                result["partial_backtrack_count"]
            )

            report.item(
                "Partial BT successes",
                result["partial_backtrack_success"]
            )

            report.item(
                "Full reroutes",
                result["full_reroute_count"]
            )

            report.item(
                "Fee",
                result["fee"]
            )

            report.item(
                "Delay",
                result["delay"]
            )

            report.item(
                "Carbon",
                result["carbon"]
            )

            report.item(
                "Reward",
                result["reward"]
            )

            report.item(
                "Reason",
                result["reason"]
            )

            report.item(
                "Prediction time",
                f"{result['prediction_time']:.4f} seconds"
            )

            report.item(
                "Pipeline time",
                f"{result['pipeline_time']:.4f} seconds"
            )

            report.item(
                "Transaction execution time",
                f"{episode_elapsed:.4f} seconds"
            )

            report.add("")

            report.add(
                "Route path:"
            )

            if result["path"]:

                report.add(
                    " -> ".join(
                        str(node)
                        for node in result["path"]
                    )
                )

            else:

                report.add(
                    "NONE"
                )

            report.save()

        complete_step(
            9,
            "STEP 9 - MULTI-TRANSACTION EVALUATION",
            report,
            "FAST EVALUATION COMPLETED"
        )


        # ====================================================
        # STEP 10
        # ====================================================

        begin_step(
            10,
            "STEP 10 - FINAL RESULT",
            report,
            "Calculating aggregate results..."
        )

        if len(evaluation_results) != EVAL_TRANSACTION_COUNT:

            raise RuntimeError(

                "Expected "
                f"{EVAL_TRANSACTION_COUNT} "
                "evaluation results but received "
                f"{len(evaluation_results)}."

            )

        aggregate = aggregate_results(
            evaluation_results
        )

        print_summary(
            aggregate
        )

        report.section(
            "FINAL AGGREGATE RESULT"
        )

        report.item(
            "Evaluation transactions",
            aggregate["total"]
        )

        report.item(
            "Successful payments",
            aggregate["successes"]
        )

        report.item(
            "Failed payments",
            aggregate["failures"]
        )

        report.item(
            "Success rate",
            f"{aggregate['success_rate']:.2f}%"
        )

        report.item(
            "Failure rate",
            f"{aggregate['failure_rate']:.2f}%"
        )

        report.item(
            "Average attempts",
            format_metric(
                aggregate["average_attempts"]
            )
        )

        report.item(
            "Average hops",
            format_metric(
                aggregate["average_hops"]
            )
        )

        report.item(
            "Average fee",
            format_metric(
                aggregate["average_fee"],
                6
            )
        )

        report.item(
            "Average delay",
            format_metric(
                aggregate["average_delay"]
            )
        )

        report.item(
            "Average carbon",
            format_metric(
                aggregate["average_carbon"],
                6
            )
        )

        report.item(
            "Average reward",
            format_metric(
                aggregate["average_reward"],
                6
            )
        )

        report.item(
            "Average eta",
            format_metric(
                aggregate["average_eta"],
                6
            )
        )

        report.item(
            "Average candidate pool",
            format_metric(
                aggregate["average_candidates"]
            )
        )

        report.item(
            "Average Bucket size",
            format_metric(
                aggregate["average_bucket"]
            )
        )

        report.item(
            "Partial backtracks",
            aggregate[
                "total_partial_backtracks"
            ]
        )

        report.item(
            "Partial BT successes",
            aggregate[
                "total_partial_backtrack_success"
            ]
        )

        report.item(
            "Full reroutes",
            aggregate[
                "total_full_reroutes"
            ]
        )

        report.add("")

        report.add(
            "Successful Bucket candidate distribution:"
        )

        if aggregate["successful_bucket_rank_counts"]:

            for rank in sorted(
                aggregate[
                    "successful_bucket_rank_counts"
                ]
            ):

                count = aggregate[
                    "successful_bucket_rank_counts"
                ][rank]

                report.item(
                    f"Bucket candidate #{rank}",
                    count
                )

        else:

            report.add(
                "No explicit successful Bucket candidate "
                "was recorded."
            )

        report.add("")

        report.add(
            "FAST Experiment interpretation:"
        )

        report.add(
            f"1. Up to {ROUTE_GENERATION_POOL} candidate "
            "routes are generated."
        )

        report.add(
            f"2. Up to {ROUTE_POOL_SIZE} diversified "
            "routes are retained."
        )

        report.add(
            f"3. {BUCKET_SIZE} routes are selected "
            "for the Bucket."
        )

        report.add(
            "4. The Bucket attempts routes sequentially."
        )

        report.add(
            "5. A transaction may succeed on any available "
            "Bucket candidate."
        )

        report.add(
            "6. PPO learns eta from routing reward."
        )

        report.add(
            "7. This configuration is intended for "
            "fast pipeline validation."
        )

        report.add(
            "8. The FAST configuration should not be "
            "used as the final thesis experiment."
        )

        total_time = (
            time.time()
            -
            RUN_START_TIME
        )

        report.section(
            "EXECUTION TIME"
        )

        report.item(
            "FAST mode",
            FAST_TRAINING
        )

        report.item(
            "PPO training time",
            f"{training_time:.4f} seconds"
        )

        report.item(
            "PPO target timesteps",
            PPO_TOTAL_TIMESTEPS
        )

        report.item(
            "PPO completed timesteps",
            PPO_CURRENT_TIMESTEP
        )

        report.item(
            "PPO progress",
            f"{PPO_TRAINING_PROGRESS:.2f}%"
        )

        report.item(
            "PPO FPS",
            f"{PPO_TRAINING_FPS:.4f}"
        )

        report.item(
            "Total execution time",
            f"{total_time:.4f} seconds"
        )

        report.item(
            "Training transactions",
            TRAIN_TRANSACTION_COUNT
        )

        report.item(
            "Evaluation transactions",
            EVAL_TRANSACTION_COUNT
        )

        report.item(
            "Routes generated per evaluation",
            ROUTE_GENERATION_POOL
        )

        report.item(
            "Routes retained before ranking",
            ROUTE_POOL_SIZE
        )

        report.item(
            "Routes retained in Bucket",
            BUCKET_SIZE
        )

        report.item(
            "Maximum hops",
            MAX_ROUTE_HOPS
        )

        report.item(
            "Failure rate",
            FAILURE_RATE
        )

        report.section(
            "FINAL STATUS"
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
            "Model",
            MODEL_NAME
        )

        report.item(
            "Report",
            str(
                REPORT_FILE.resolve()
            )
        )

        report_path = report.save()

        print_progress(
            progress=100.0,
            step=10,
            step_name="RUN COMPLETED",
            message=(
                "FAST ROUTE GENERATION -> "
                "PPO + BUCKET EXPERIMENT COMPLETED"
            )
        )

        print(
            "\n============================================================",
            flush=True
        )

        print(
            "MAIN2 FAST MODE COMPLETED",
            flush=True
        )

        print(
            "============================================================",
            flush=True
        )

        print(
            f"Training transactions : "
            f"{TRAIN_TRANSACTION_COUNT}",
            flush=True
        )

        print(
            f"Evaluation transactions: "
            f"{EVAL_TRANSACTION_COUNT}",
            flush=True
        )

        print(
            f"Generated route pool  : "
            f"{ROUTE_GENERATION_POOL}",
            flush=True
        )

        print(
            f"Retained route pool   : "
            f"{ROUTE_POOL_SIZE}",
            flush=True
        )

        print(
            f"Bucket size           : "
            f"{BUCKET_SIZE}",
            flush=True
        )

        print(
            f"PPO timesteps         : "
            f"{PPO_CURRENT_TIMESTEP} / "
            f"{PPO_TOTAL_TIMESTEPS}",
            flush=True
        )

        print(
            f"PPO training progress : "
            f"{PPO_TRAINING_PROGRESS:.2f}%",
            flush=True
        )

        print(
            f"PPO training time     : "
            f"{training_time:.2f} seconds",
            flush=True
        )

        print(
            f"Total execution time  : "
            f"{total_time:.2f} seconds",
            flush=True
        )

        print(
            f"Successful payments   : "
            f"{aggregate['successes']}",
            flush=True
        )

        print(
            f"Failed payments       : "
            f"{aggregate['failures']}",
            flush=True
        )

        print(
            f"Success rate          : "
            f"{aggregate['success_rate']:.2f}%",
            flush=True
        )

        print(
            "\nSuccessful Bucket candidate distribution:",
            flush=True
        )

        if aggregate["successful_bucket_rank_counts"]:

            for rank in sorted(
                aggregate[
                    "successful_bucket_rank_counts"
                ]
            ):

                print(
                    f"  Candidate #{rank}: "
                    f"{aggregate['successful_bucket_rank_counts'][rank]}",
                    flush=True
                )

        else:

            print(
                "  No successful candidate rank.",
                flush=True
            )

        print(
            "\nReport:",
            flush=True
        )

        print(
            report_path.resolve(),
            flush=True
        )

        return {

            "success":
                True,

            "fast_training":
                FAST_TRAINING,

            "training_transactions":
                TRAIN_TRANSACTION_COUNT,

            "evaluation_transactions":
                EVAL_TRANSACTION_COUNT,

            "route_generation_pool":
                ROUTE_GENERATION_POOL,

            "route_pool_size":
                ROUTE_POOL_SIZE,

            "bucket_size":
                BUCKET_SIZE,

            "max_hops":
                MAX_ROUTE_HOPS,

            "failure_rate":
                FAILURE_RATE,

            "ppo_total_timesteps":
                PPO_TOTAL_TIMESTEPS,

            "ppo_completed_timesteps":
                PPO_CURRENT_TIMESTEP,

            "ppo_training_progress":
                PPO_TRAINING_PROGRESS,

            "ppo_training_fps":
                PPO_TRAINING_FPS,

            "training_time":
                training_time,

            "total_time":
                total_time,

            "aggregate":
                aggregate,

            "evaluation_results":
                evaluation_results,

            "report":
                str(
                    report_path.resolve()
                ),

        }

    except Exception as exc:

        try:

            report.section(
                "RUNTIME ERROR"
            )

            report.item(
                "Project root",
                PROJECT_ROOT
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
                "Progress",
                f"{CURRENT_PROGRESS:.2f}%"
            )

            report.item(
                "FAST mode",
                FAST_TRAINING
            )

            report.item(
                "PPO target timesteps",
                PPO_TOTAL_TIMESTEPS
            )

            report.item(
                "PPO current timestep",
                PPO_CURRENT_TIMESTEP
            )

            report.item(
                "PPO progress",
                f"{PPO_TRAINING_PROGRESS:.2f}%"
            )

            report.item(
                "Exception type",
                type(exc).__name__
            )

            report.item(
                "Exception",
                str(exc)
            )

            report.add("")
            report.add("Traceback:")

            report.add(
                traceback.format_exc()
            )

            report.save()

        except Exception:
            pass

        print(
            "\n============================================================",
            flush=True
        )

        print(
            "MAIN2 EXECUTION FAILED",
            flush=True
        )

        print(
            "============================================================",
            flush=True
        )

        print(
            f"Project  : {PROJECT_ROOT}",
            flush=True
        )

        print(
            f"Stage    : {CURRENT_STAGE}",
            flush=True
        )

        print(
            f"Step     : {CURRENT_STEP}",
            flush=True
        )

        print(
            f"Progress : {CURRENT_PROGRESS:.2f}%",
            flush=True
        )

        print(
            f"PPO      : "
            f"{PPO_CURRENT_TIMESTEP} / "
            f"{PPO_TOTAL_TIMESTEPS}",
            flush=True
        )

        print(
            f"Error    : "
            f"{type(exc).__name__}: {exc}",
            flush=True
        )

        traceback.print_exc()

        print(
            "\nDetailed report:",
            flush=True
        )

        print(
            REPORT_FILE.resolve(),
            flush=True
        )

        raise

    finally:

        stop_heartbeat()


# ============================================================
# SAFE ENTRY POINT
# ============================================================

def run_main2():

    print(
        "\n[MAIN2] Starting main()...",
        flush=True
    )

    try:

        result = main()

        print(
            "\n[MAIN2] main() returned successfully.",
            flush=True
        )

        return result

    except KeyboardInterrupt:

        print(
            "\n[MAIN2] Execution interrupted by user.",
            flush=True
        )

        raise

    except Exception as exc:

        print(
            "\n[MAIN2] Fatal error reached entry point.",
            flush=True
        )

        print(
            f"[MAIN2] {type(exc).__name__}: {exc}",
            flush=True
        )

        traceback.print_exc()

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_main2()