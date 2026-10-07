# main.py

import copy
import io
import os
import random
import time
import yaml
import zipfile
import numpy as np
import networkx as nx
import traceback
import linecache
import sys
import threading

from pathlib import Path

import torch
from stable_baselines3 import PPO

# Stable-Baselines3 modules are imported explicitly because
# PPO.load() resolves load_from_zip_file through BaseAlgorithm.
import stable_baselines3.common.save_util as sb3_save_util
import stable_baselines3.common.base_class as sb3_base_class


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

MODEL_NAME = "end_to_end_hidden_failure_blind"


# ============================================================
# Evaluation Configuration
# ============================================================

# Normal runs use the held-out transaction count from config;
# fast validation can use a smaller smoke-run count.
EVAL_TRANSACTION_COUNT = 3000

EVAL_TRANSACTION_ENV = "RL_EVAL_TRANSACTIONS"


def get_evaluation_transaction_count(cfg):

    """
    Resolve the evaluation size from config and runtime mode.
    """
    evaluation_cfg = cfg.get("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        raise TypeError("evaluation configuration must be a dictionary.")
    config_key = (
        "fast_validation_transactions"
        if FAST_VALIDATION
        else "transaction_count"
    )
    count = int(
        evaluation_cfg.get(
            config_key,
            10 if FAST_VALIDATION else EVAL_TRANSACTION_COUNT,
        )
    )
    configured_env_value = os.environ.get(
        EVAL_TRANSACTION_ENV
    )
    if configured_env_value is not None:
        try:
            count = int(configured_env_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{EVAL_TRANSACTION_ENV} must be a positive integer."
            )
    if count <= 0:
        raise ValueError("Evaluation transaction count must be positive.")
    return count


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

    from Simulation.transaction_split import (
        TRAIN_RATIO,
        normalize_evaluation_count,
        split_transaction_pool,
        training_count_for_evaluation_count,
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
    report=None,
    run_status="FAILED",
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

            f"Run status    : {run_status}",

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
                "Run status",
                run_status
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

    try:

        torch.manual_seed(seed)

    except Exception:

        pass


# ============================================================
# GML.GEO -> JSON-like Dictionary
# ============================================================

def geo_to_json(input_file):

    print(
        f"Reading snapshot: {input_file}"
    )

    # NetworkX's read_gml defaults to ASCII. These snapshots are UTF-8 and
    # can contain a stray non-ASCII marker outside a quoted GML value; remove
    # only such invalid bare characters while preserving quoted text.
    with open(input_file, "r", encoding="utf-8") as snapshot_file:
        gml_lines = []
        for line in snapshot_file:
            cleaned = []
            quoted = False
            escaped = False
            for character in line:
                if escaped:
                    cleaned.append(character)
                    escaped = False
                elif character == "\\" and quoted:
                    cleaned.append(character)
                    escaped = True
                elif character == '"':
                    cleaned.append(character)
                    quoted = not quoted
                elif ord(character) > 127 and not quoted:
                    continue
                else:
                    cleaned.append(character)
            gml_lines.append("".join(cleaned))
        graph = nx.parse_gml(gml_lines, label=None)

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
            f"{key:<32}: {value}"
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


def safe_float(value, default=0.0):

    try:

        if value is None:

            return float(
                default
            )

        result = float(
            value
        )

        if not np.isfinite(
            result
        ):

            return float(
                default
            )

        return result

    except Exception:

        return float(
            default
        )


def safe_int(value, default=0):

    try:

        if value is None:

            return int(
                default
            )

        return int(
            value
        )

    except Exception:

        return int(
            default
        )


def format_metric(value, digits=6):

    if value is None:

        return "N/A"

    if isinstance(
        value,
        float
    ):

        return f"{value:.{digits}f}"

    return str(
        value
    )


# ============================================================
# SB3 MODEL ARCHIVE VALIDATION
# ============================================================

def _validate_ppo_archive(model_path):

    """
    Validate the outer Stable-Baselines3 ZIP archive.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"PPO model not found: {model_path}"
        )

    file_size = model_path.stat().st_size

    if file_size <= 0:

        raise RuntimeError(
            f"PPO model is empty: {model_path}"
        )

    try:

        with zipfile.ZipFile(
            model_path,
            mode="r"
        ) as archive:

            bad_file = archive.testzip()

            if bad_file is not None:

                raise RuntimeError(
                    "PPO model ZIP contains a corrupted "
                    f"member: {bad_file}"
                )

            names = archive.namelist()

            required_files = {
                "data",
                "policy.pth",
            }

            missing_files = [

                name

                for name in required_files

                if name not in names

            ]

            if missing_files:

                raise RuntimeError(

                    "PPO model archive is missing required "
                    f"files: {missing_files}"

                )

            pytorch_files = [

                name

                for name in names

                if name.endswith(".pth")

            ]

            if not pytorch_files:

                raise RuntimeError(
                    "PPO model archive contains no .pth files."
                )

            print(
                "\n[PPO ARCHIVE]"
            )

            print(
                "ZIP integrity       : OK"
            )

            print(
                f"Archive size        : "
                f"{file_size:,} bytes"
            )

            print(
                f"Archive members     : "
                f"{len(names)}"
            )

            print(
                "PyTorch members     : "
                + ", ".join(pytorch_files)
            )

            return {
                "size": file_size,
                "members": names,
                "pytorch_members": pytorch_files,
            }

    except zipfile.BadZipFile as exc:

        raise RuntimeError(
            f"PPO model is not a valid ZIP archive: "
            f"{model_path}"
        ) from exc


# ============================================================
# SB3 ZIP/PyTorch Compatibility Loader
# ============================================================

def _load_from_zip_file_bytesio(
    load_path,
    load_data=True,
    custom_objects=None,
    device="auto",
    verbose=0,
    print_system_info=False,
):

    if custom_objects is None:

        custom_objects = {}

    load_path = Path(
        load_path
    )

    if not load_path.exists():

        raise FileNotFoundError(
            f"PPO checkpoint not found: {load_path}"
        )

    device = sb3_save_util.get_device(
        device=device
    )

    if verbose >= 1:

        print(
            f"[SB3 COMPAT] Loading: {load_path}"
        )

        print(
            f"[SB3 COMPAT] Device: {device}"
        )

    data = None

    params = {}

    pytorch_variables = {}

    with zipfile.ZipFile(
        load_path,
        mode="r"
    ) as archive:

        names = archive.namelist()

        # ----------------------------------------------------
        # System information
        # ----------------------------------------------------

        if print_system_info:

            if "system_info.txt" in names:

                print(
                    "== SAVED MODEL SYSTEM INFO =="
                )

                try:

                    saved_system_info = (
                        archive.read(
                            "system_info.txt"
                        ).decode(
                            "utf-8",
                            errors="replace"
                        )
                    )

                    print(
                        saved_system_info
                    )

                except Exception:

                    pass

        # ----------------------------------------------------
        # JSON metadata
        # ----------------------------------------------------

        if (
            load_data
            and
            "data" in names
        ):

            json_data = archive.read(
                "data"
            ).decode(
                "utf-8"
            )

            data = sb3_save_util.json_to_data(

                json_data,

                custom_objects=custom_objects

            )

        # ----------------------------------------------------
        # PyTorch parameter files
        # ----------------------------------------------------

        pth_files = [

            name

            for name in names

            if name.endswith(".pth")

        ]

        if not pth_files:

            raise RuntimeError(
                "No PyTorch parameter files were found "
                "inside the PPO archive."
            )

        for file_path in pth_files:

            if verbose >= 2:

                print(
                    f"[SB3 COMPAT] Reading: {file_path}"
                )

            raw_bytes = archive.read(
                file_path
            )

            if not raw_bytes:

                raise RuntimeError(
                    f"Empty PyTorch parameter file: "
                    f"{file_path}"
                )

            if verbose >= 2:

                print(
                    f"[SB3 COMPAT] Size: "
                    f"{len(raw_bytes):,} bytes"
                )

            buffer = io.BytesIO(
                raw_bytes
            )

            try:

                th_object = torch.load(

                    buffer,

                    map_location=device,

                    weights_only=True

                )

            finally:

                buffer.close()

            if file_path in {
                "pytorch_variables.pth",
                "tensors.pth",
            }:

                pytorch_variables = th_object

            else:

                parameter_name = Path(
                    file_path
                ).stem

                params[
                    parameter_name
                ] = th_object

    return (
        data,
        params,
        pytorch_variables
    )


# ============================================================
# Install SB3 Compatibility Loader
# ============================================================

def _install_ppo_loader_compatibility():

    sb3_save_util.load_from_zip_file = (
        _load_from_zip_file_bytesio
    )

    sb3_base_class.load_from_zip_file = (
        _load_from_zip_file_bytesio
    )

    print(
        "[SB3 COMPAT] ZIP -> BytesIO loader installed."
    )


# ============================================================
# PPO Model Loading
# ============================================================

def load_existing_ppo_model(
    model_path,
    env=None
):

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
        f"Model path : {zip_path.resolve()}"
    )

    print(
        "Loading mode: PROJECT-LOCAL ZIP/BytesIO COMPATIBILITY"
    )

    load_start = time.time()

    _validate_ppo_archive(
        zip_path
    )

    print(
        "Archive validation : SUCCESS"
    )

    _install_ppo_loader_compatibility()

    print(
        "PPO.load()         : STARTED"
    )

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
        "PPO.load()         : SUCCESS"
    )

    print(
        "Loader             : ZIP -> BytesIO -> torch.load"
    )

    print(
        f"Model load time    : {load_time:.4f} seconds"
    )

    return model, load_time


# ============================================================
# Generate Evaluation Transactions
# ============================================================

def generate_evaluation_transactions(
    G,
    cfg,
    seed,
    count,
    source_transactions=None,
):
    if count <= 0:
        raise ValueError("Evaluation transaction count must be positive.")

    min_amount = cfg["simulation"]["min_amount"]

    max_amount = cfg["simulation"]["max_amount"]

    if source_transactions is None:
        transactions = generate_transactions(
            G,
            count,
            seed,
            min_amount,
            max_amount,
        )
    else:
        if len(source_transactions) < count:
            raise ValueError(
                f"Held-out transaction list contains {len(source_transactions)} rows; "
                f"{count} are required."
            )
        transactions = [
            copy.deepcopy(transaction)
            for transaction in source_transactions[:count]
        ]

    if not transactions:

        raise RuntimeError(
            "Could not generate evaluation transactions."
        )

    if len(transactions) != count:

        raise RuntimeError(

            "Evaluation transaction generator returned "
            f"{len(transactions)} transactions, but {count} are required."

        )

    # --------------------------------------------------------
    # Make sure evaluation transactions are independent
    # objects. This also prevents accidental object reuse.
    # --------------------------------------------------------

    transactions = [
        copy.deepcopy(
            transaction
        )
        for transaction in transactions
    ]

    # Independently generated evaluation rows get local IDs. When the
    # evaluation rows came from the shared split pool, preserve their IDs
    # so the train/test partition remains auditable in the report.
    if source_transactions is None:
        for index, transaction in enumerate(
            transactions,
            start=1
        ):
            try:
                transaction.tx_id = index
            except Exception:
                pass

    return transactions


# ============================================================
# Single Evaluation Result Extraction
# ============================================================

def extract_episode_result(
    transaction,
    reward,
    terminated,
    truncated,
    info,
    episode_index
):

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

    top_k = safe_int(
        info.get(
            "top_k"
        ),
        5
    )

    candidate_count = safe_int(
        info.get(
            "candidate_path_count"
        ),
        0
    )

    usable_candidate_count = safe_int(
        info.get(
            "usable_candidate_count"
        ),
        0
    )

    bucket_size = safe_int(
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

    path_length = safe_int(
        info.get(
            "path_length"
        ),
        0
    )

    fee = safe_float(
        info.get(
            "fee"
        ),
        0.0
    )

    delay = safe_float(
        info.get(
            "delay"
        ),
        0.0
    )

    carbon = safe_float(
        info.get(
            "carbon"
        ),
        0.0
    )

    attempt_count = safe_int(
        info.get(
            "attempt_count"
        ),
        0
    )

    backtrack_count = safe_int(
        info.get(
            "backtrack_count"
        ),
        0
    )

    partial_backtrack_count = safe_int(
        info.get(
            "partial_backtrack_count"
        ),
        0
    )

    partial_backtrack_success = safe_int(
        info.get(
            "partial_backtrack_success"
        ),
        0
    )

    full_reroute_count = safe_int(
        info.get(
            "full_reroute_count"
        ),
        0
    )

    failure_probability = safe_float(
        info.get(
            "failure_probability"
        ),
        0.0
    )

    reason = safe_value(
        info.get(
            "reason"
        ),
        ""
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT infer candidate rank from:
    #
    #     partial_backtrack_count
    #
    # or:
    #
    #     attempt_count
    #
    # The environment must explicitly report the successful
    # Bucket candidate rank.
    # --------------------------------------------------------

    successful_bucket_rank = info.get(
        "successful_bucket_rank"
    )

    successful_route_source = info.get(
        "successful_route_source"
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
        and
        (
            successful_bucket_rank < 1
            or
            (
                bucket_size > 0
                and
                successful_bucket_rank > bucket_size
            )
        )
    ):

        successful_bucket_rank = None

    return {

        "episode_index":
            episode_index,

        "transaction_id":
            safe_value(
                getattr(
                    transaction,
                    "tx_id",
                    episode_index
                ),
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
                ),
                0.0
            ),

        "success":
            success,

        "terminated":
            bool(
                terminated
            ),

        "truncated":
            bool(
                truncated
            ),

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

        "successful_bucket_rank":
            successful_bucket_rank,

        "successful_route_source":
            successful_route_source,

        "attempt_count":
            attempt_count,

        "backtrack_count":
            backtrack_count,

        "partial_backtrack_count":
            partial_backtrack_count,

        "partial_backtrack_success":
            partial_backtrack_success,

        "full_reroute_count":
            full_reroute_count,

        "failure_probability":
            failure_probability,

        "fee":
            fee,

        "delay":
            delay,

        "carbon":
            carbon,

        "reward":
            safe_float(
                reward,
                0.0
            ),

        "reason":
            reason,

        "path":
            final_path,

        "path_length":
            path_length,

        "hops":
            max(
                0,
                path_length
            ),

        "info_keys":
            list(
                info.keys()
            ),

        "raw_info":
            info,

    }


# ============================================================
# Aggregate Evaluation Results
# ============================================================

def aggregate_evaluation_results(
    results
):

    total = len(
        results
    )

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    failed = (
        total
        -
        successful
    )

    def mean(
        key,
        only_success=False
    ):

        values = []

        for result in results:

            if (
                only_success
                and
                not result["success"]
            ):

                continue

            value = result.get(
                key
            )

            if value is None:

                continue

            try:

                value = float(
                    value
                )

            except Exception:

                continue

            if np.isfinite(
                value
            ):

                values.append(
                    value
                )

        if not values:

            return None

        return float(
            np.mean(
                values
            )
        )

    def delay_percentile(percentile, only_success):
        values = []
        for result in results:
            if only_success and not result.get("success", False):
                continue
            if int(result.get("path_length", 0) or 0) <= 0:
                continue
            try:
                value = float(result.get("delay"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return float(np.percentile(values, percentile)) if values else None

    rank_counts = {}

    unknown_success_rank = 0

    for result in results:

        if not result["success"]:

            continue

        rank = result.get(
            "successful_bucket_rank"
        )

        if rank is None:

            unknown_success_rank += 1

        else:

            rank_counts[rank] = (
                rank_counts.get(
                    rank,
                    0
                )
                +
                1
            )

    bucket_exhausted = sum(

        1

        for result in results

        if (
            not result["success"]
            and
            result["bucket_size"] > 0
            and
            result["attempt_count"]
            >=
            result["bucket_size"]
        )

    )

    total_partial_backtracks = sum(
        result["partial_backtrack_count"]
        for result in results
    )

    total_partial_backtrack_success = sum(
        result["partial_backtrack_success"]
        for result in results
    )

    total_full_reroutes = sum(
        result["full_reroute_count"]
        for result in results
    )

    total_attempts = sum(
        result["attempt_count"]
        for result in results
    )

    return {

        "total_transactions":
            total,

        "successful_transactions":
            successful,

        "failed_transactions":
            failed,

        "success_rate":
            (
                successful / total * 100.0
                if total > 0
                else 0.0
            ),

        "failure_rate":
            (
                failed / total * 100.0
                if total > 0
                else 0.0
            ),

        "average_attempts":
            mean(
                "attempt_count"
            ),

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

        "p90_delay_success":
            delay_percentile(90, only_success=True),

        "p95_delay_success":
            delay_percentile(95, only_success=True),

        "p90_delay_routed":
            delay_percentile(90, only_success=False),

        "p95_delay_routed":
            delay_percentile(95, only_success=False),

        "average_carbon":
            mean(
                "carbon",
                only_success=True
            ),

        "average_reward":
            mean(
                "reward"
            ),

        "average_eta":
            mean(
                "eta"
            ),

        "average_candidates":
            mean(
                "candidate_count"
            ),

        "average_usable_candidates":
            mean(
                "usable_candidate_count"
            ),

        "average_bucket_size":
            mean(
                "bucket_size"
            ),

        "total_attempts":
            total_attempts,

        "total_partial_backtracks":
            total_partial_backtracks,

        "total_partial_backtrack_success":
            total_partial_backtrack_success,

        "total_full_reroutes":
            total_full_reroutes,

        "bucket_exhausted":
            bucket_exhausted,

        "successful_bucket_rank_counts":
            rank_counts,

        "successful_rank_unknown":
            unknown_success_rank,

    }


# ============================================================
# Print Evaluation Summary
# ============================================================

def print_evaluation_summary(
    aggregate
):

    print(
        "\n============================================================"
    )

    print(
        "MULTI-TRANSACTION EVALUATION SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Transactions evaluated : "
        f"{aggregate['total_transactions']}"
    )

    print(
        f"Successful payments    : "
        f"{aggregate['successful_transactions']}"
    )

    print(
        f"Failed payments        : "
        f"{aggregate['failed_transactions']}"
    )

    print(
        f"Success rate           : "
        f"{aggregate['success_rate']:.2f}%"
    )

    print(
        f"Failure rate           : "
        f"{aggregate['failure_rate']:.2f}%"
    )

    print(
        f"Average attempts       : "
        f"{format_metric(aggregate['average_attempts'], 4)}"
    )

    print(
        f"Average hops           : "
        f"{format_metric(aggregate['average_hops'], 4)}"
    )

    print(
        f"Average fee            : "
        f"{format_metric(aggregate['average_fee'], 6)}"
    )

    print(
        f"Average delay          : "
        f"{format_metric(aggregate['average_delay'], 4)}"
    )

    print(
        f"Average carbon         : "
        f"{format_metric(aggregate['average_carbon'], 6)}"
    )

    print(
        f"Average reward         : "
        f"{format_metric(aggregate['average_reward'], 6)}"
    )

    print(
        f"Average eta            : "
        f"{format_metric(aggregate['average_eta'], 6)}"
    )

    print(
        f"Total partial backtracks: "
        f"{aggregate['total_partial_backtracks']}"
    )

    print(
        f"Partial backtrack successes: "
        f"{aggregate['total_partial_backtrack_success']}"
    )

    print(
        f"Total full reroutes    : "
        f"{aggregate['total_full_reroutes']}"
    )

    print(
        "\nSuccessful Bucket candidate distribution:"
    )

    rank_counts = aggregate[
        "successful_bucket_rank_counts"
    ]

    if rank_counts:

        for rank in sorted(
            rank_counts
        ):

            count = rank_counts[
                rank
            ]

            print(
                f"  Candidate #{rank}: {count}"
            )

    else:

        print(
            "  No explicit successful candidate "
            "rank was reported by the environment."
        )

    if aggregate[
        "successful_rank_unknown"
    ] > 0:

        print(
            f"  Successful rank unknown: "
            f"{aggregate['successful_rank_unknown']}"
        )

    print(
        f"\nBucket exhausted          : "
        f"{aggregate['bucket_exhausted']}"
    )

    print(
        "============================================================"
    )


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
        else
        "NORMAL TRAINING"
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
        f"0/{TOTAL_STEPS}"
    )

    report.item(
        "Evaluation transaction count",
        "Resolved from configuration and RL_EVAL_TRANSACTIONS"
    )

    report.save()

    runtime_env = None

    training_time = 0.0

    model_load_time = 0.0

    prediction_time_total = 0.0

    execution_time_total = 0.0

    model = None

    evaluation_results = []

    aggregate = None

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

        # ----------------------------------------------------
        # Resolve fast-smoke or full held-out evaluation size.
        # ----------------------------------------------------

        requested_evaluation_transaction_count = (
            get_evaluation_transaction_count(
                cfg
            )
        )

        if requested_evaluation_transaction_count <= 0:
            raise RuntimeError("Evaluation transaction count must be positive.")

        evaluation_transaction_count = normalize_evaluation_count(
            requested_evaluation_transaction_count
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
            else
            "NORMAL TRAINING"
        )

        report.item(
            "Requested evaluation transactions",
            requested_evaluation_transaction_count
        )

        report.item(
            "Effective held-out transactions",
            evaluation_transaction_count
        )

        report.item(
            "Evaluation transaction policy",
            "Target is adjusted to nearest count divisible by 3 for an exact 70/30 split"
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
            str(
                snapshot_path.resolve()
            )
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

        report.item(
            "Requested evaluation transactions",
            requested_evaluation_transaction_count
        )

        report.item(
            "Effective held-out transactions",
            evaluation_transaction_count
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

        builder = LNGraphBuilder(
            default_capacity=cfg.get("channel", {}).get("default_capacity")
        )

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

        train_count = training_count_for_evaluation_count(
            evaluation_transaction_count
        )
        effective_transaction_count = (
            train_count + evaluation_transaction_count
        )

        transaction_pool = generate_transactions(

            G_train,

            effective_transaction_count,

            seed,

            cfg["simulation"]["min_amount"],

            cfg["simulation"]["max_amount"]

        )

        if len(transaction_pool) != effective_transaction_count:
            raise RuntimeError(
                f"Transaction generator returned {len(transaction_pool)} rows; "
                f"the 70/30 split requires {effective_transaction_count}."
            )

        train_tx, held_out_transactions = split_transaction_pool(
            transaction_pool,
            evaluation_transaction_count,
        )

        report.item(
            "Configured base transaction count",
            configured_transaction_count
        )

        report.item(
            "Requested evaluation transactions",
            requested_evaluation_transaction_count
        )

        report.item(
            "Effective held-out transactions",
            evaluation_transaction_count
        )

        report.item(
            "Effective transaction pool",
            len(transaction_pool)
        )

        report.item(
            "Generated transactions",
            len(transaction_pool)
        )

        report.item(
            "Required training transactions",
            train_count
        )

        report.item(
            "Training share (integer split)",
            f"{len(train_tx) / len(transaction_pool) * 100:.2f}%"
        )

        report.item(
            "Evaluation share (integer split)",
            f"{len(held_out_transactions) / len(transaction_pool) * 100:.2f}%"
        )

        report.item(
            "Configured train_ratio",
            cfg.get("train_ratio", "unspecified")
        )

        report.item(
            "Applied train/test protocol",
            f"EXACT {TRAIN_RATIO:.0%}/{1 - TRAIN_RATIO:.0%}; held-out count normalized to a multiple of 3"
        )

        report.item(
            "Transactions supplied to PPO",
            len(train_tx)
        )

        report.item(
            "Held-out transactions",
            len(held_out_transactions)
        )

        if FAST_VALIDATION:

            report.item(
                "Transaction generation mode",
                "FAST VALIDATION - DISJOINT 70/30 TRAIN/TEST POOL"
            )

        else:

            report.item(
                "Transaction generation mode",
                "NORMAL TRAINING - DISJOINT 70/30 TRAIN/TEST POOL"
            )

        print(
            f"Configured base pool    : "
            f"{configured_transaction_count}"
        )

        print(
            f"Effective pool          : "
            f"{len(transaction_pool)}"
        )

        print(
            f"Train transactions      : "
            f"{len(train_tx)}"
        )

        print(
            f"Held-out transactions   : "
            f"{len(held_out_transactions)}"
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

            report.item(
                "Execution mode",
                "FAST VALIDATION"
            )

            report.item(
                "PPO checkpoint candidate",
                str(model_zip_path.resolve())
            )

            report.item(
                "Fast validation policy",
                "Load a compatible checkpoint; if unavailable, train an isolated quick model"
            )

            report.item(
                "PPO checkpoint loader",
                "PROJECT-LOCAL ZIP/BytesIO COMPATIBILITY"
            )

            model_load_start = time.time()

            try:

                model, model_load_time = (
                    load_existing_ppo_model(
                        model_path,
                        env=None
                    )
                )

                report.item(
                    "Checkpoint status",
                    "COMPATIBLE MODEL LOADED"
                )

                report.item(
                    "Model provenance",
                    "Existing PPO checkpoint"
                )

                report.item(
                    "Model",
                    str(model_zip_path.resolve())
                )

                report.item(
                    "Training time (seconds)",
                    0.0
                )

                report.add("")

                report.add(
                    "Fast validation reused an existing checkpoint. "
                    "Its observation shape and archive were validated "
                    "before inference."
                )

            except Exception as checkpoint_error:

                model_load_time = (
                    time.time() - model_load_start
                )

                print(
                    "\nNo usable PPO checkpoint was available; "
                    "training an isolated quick-validation model."
                )

                report.item(
                    "Research checkpoint status",
                    "MISSING OR INCOMPATIBLE"
                )

                report.item(
                    "Research checkpoint load error",
                    f"{type(checkpoint_error).__name__}: {checkpoint_error}"
                )

                fast_model_dir = (
                    model_dir / "fast_validation_exact_70_30"
                )

                fast_cfg = copy.deepcopy(cfg)
                fast_cfg.setdefault("rl", {})
                fast_cfg["rl"]["model_dir"] = str(
                    fast_model_dir
                )
                fast_cfg["rl"]["log_dir"] = str(
                    Path("logs") / "fast_validation"
                )
                fast_cfg["rl"]["checkpoint_dir"] = str(
                    fast_model_dir / "checkpoints"
                )

                fast_model_path = (
                    fast_model_dir / MODEL_NAME
                )

                try:

                    model, quick_model_load_time = (
                        load_existing_ppo_model(
                            fast_model_path,
                            env=None,
                        )
                    )

                    model_load_time += quick_model_load_time

                    report.item(
                        "Checkpoint status",
                        "COMPATIBLE QUICK VALIDATION MODEL LOADED"
                    )

                    report.item(
                        "Model provenance",
                        "Previously trained quick-validation model"
                    )

                    report.item(
                        "Model",
                        str(Path(str(fast_model_path) + ".zip").resolve())
                    )

                    report.item(
                        "Training time (seconds)",
                        0.0
                    )

                    report.add("")

                    report.add(
                        "The research checkpoint was unavailable, so a "
                        "compatible model from the isolated fast-validation "
                        "directory was reused. Its results remain smoke "
                        "validation only."
                    )

                except Exception as quick_checkpoint_error:

                    report.item(
                        "Quick checkpoint status",
                        "MISSING OR INCOMPATIBLE; QUICK TRAINING STARTED"
                    )

                    report.item(
                        "Quick checkpoint load error",
                        f"{type(quick_checkpoint_error).__name__}: "
                        f"{quick_checkpoint_error}"
                    )

                    report.item(
                        "PPO training",
                        "QUICK VALIDATION TRAINING"
                    )

                    report.item(
                        "Training transactions",
                        len(train_tx)
                    )

                    report.item(
                        "Model output",
                        str(Path(str(fast_model_path) + ".zip").resolve())
                    )

                    training_start = time.time()

                    model = train_agent(
                        G_train,
                        train_tx,
                        lnd_cost,
                        fast_cfg,
                        MODEL_NAME,
                        seed,
                        fast_training=True,
                    )

                    training_time = time.time() - training_start

                    report.item(
                        "Training status",
                        "SUCCESS - QUICK VALIDATION ONLY"
                    )

                    report.item(
                        "Training time (seconds)",
                        round(training_time, 4)
                    )

                    report.item(
                        "Model provenance",
                        "Fresh quick PPO model; isolated from the research checkpoint"
                    )

                    report.item(
                        "Model",
                        str(Path(str(fast_model_path) + ".zip").resolve())
                    )

                    report.add("")

                    report.add(
                        "This quick model is trained only to validate the "
                        "end-to-end software path. Its results are not a "
                        "research evaluation and it does not overwrite the "
                        "normal-training checkpoint."
                    )

            report.item(
                "Model load time (seconds)",
                round(model_load_time, 4)
            )

            report.item(
                "PPO checkpoint loader",
                "PROJECT-LOCAL ZIP/BytesIO COMPATIBILITY"
            )

            complete_step(
                4,
                "STEP 4 - PPO TRAINING",
                report,
                "PPO MODEL READY FOR VALIDATION"
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
                "snapshot that will be used for evaluation."
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
        # STEP 5 - MULTIPLE PAYMENT SCENARIOS
        # ====================================================

        set_stage(
            "STEP 5 - GENERATE PAYMENT SCENARIOS"
        )

        report.section(
            "STEP 5 - PAYMENT SCENARIOS"
        )

        begin_step(
            5,
            "STEP 5 - GENERATE PAYMENT SCENARIOS",
            report,
            f"Generating {evaluation_transaction_count} evaluation transactions..."
        )

        print(
            "\n============================================================"
        )

        print(
            f"STEP 5 - GENERATING {evaluation_transaction_count} PAYMENT SCENARIOS"
        )

        print(
            "============================================================"
        )

        scenario_seed = (
            seed + 1000
        )

        evaluation_transactions = (
            generate_evaluation_transactions(

                G,

                cfg,

                scenario_seed,

                evaluation_transaction_count,

                source_transactions=held_out_transactions,

            )
        )

        # ----------------------------------------------------
        # Hard validation.
        # ----------------------------------------------------

        if len(evaluation_transactions) != evaluation_transaction_count:

            raise RuntimeError(

                f"Exactly {evaluation_transaction_count} evaluation transactions were "
                "required, but "
                f"{len(evaluation_transactions)} were generated."

            )

        report.item(
            "Requested evaluation transactions",
            requested_evaluation_transaction_count
        )

        report.item(
            "Effective evaluation transactions",
            evaluation_transaction_count
        )

        report.item(
            "Generated evaluation transactions",
            len(
                evaluation_transactions
            )
        )

        report.item(
            "Evaluation seed",
            scenario_seed
        )

        report.item(
            "Evaluation transaction policy",
            "Last rows of the single seeded pool; excluded from PPO training"
        )

        print(
            f"Evaluation transaction target     : "
            f"{requested_evaluation_transaction_count}"
        )

        print(
            f"Effective evaluation transactions : "
            f"{evaluation_transaction_count}"
        )

        print(
            f"Evaluation transactions generated : "
            f"{len(evaluation_transactions)}"
        )

        report.add("")

        report.add(
            "Each evaluation transaction is executed "
            "independently."
        )

        report.add(
            "Training and evaluation transactions come from one seeded pool "
            "and are split into disjoint 70/30 partitions."
        )

        report.add(
            "The same trained PPO model is reused, but PPO "
            "makes exactly one prediction for each transaction."
        )

        report.add(
            "Each transaction receives its own evaluation "
            "graph, failure model and network dynamics instance."
        )

        # ----------------------------------------------------
        # Scenario list
        # ----------------------------------------------------

        report.add("")

        report.add(
            "Evaluation transaction list:"
        )

        for index, transaction in enumerate(
            evaluation_transactions,
            start=1
        ):

            report.item(
                f"Transaction #{index}",
                (
                    f"id={getattr(transaction, 'tx_id', index)}, "
                    f"source={getattr(transaction, 'source', '')}, "
                    f"destination={getattr(transaction, 'destination', '')}, "
                    f"amount={getattr(transaction, 'amount', '')}"
                )
            )

            print(
                f"TX #{index:02d} | "
                f"source={getattr(transaction, 'source', '')} | "
                f"destination={getattr(transaction, 'destination', '')} | "
                f"amount={getattr(transaction, 'amount', '')}"
            )

        complete_step(
            5,
            "STEP 5 - GENERATE PAYMENT SCENARIOS",
            report,
            f"{evaluation_transaction_count} PAYMENT SCENARIOS READY"
        )

        # ====================================================
        # STEP 6 - EVALUATION GRAPHS
        # ====================================================

        set_stage(
            "STEP 6 - PREPARE EVALUATION GRAPHS"
        )

        report.section(
            "STEP 6 - EVALUATION GRAPHS"
        )

        begin_step(
            6,
            "STEP 6 - PREPARE EVALUATION GRAPHS",
            report,
            "Preparing independent evaluation graphs..."
        )

        payment_failure_rate = float(
            cfg.get(
                "runtime_failure_rate",
                0.03
            )
        )

        report.item(
            "Payment failure rate",
            payment_failure_rate
        )

        report.item(
            "Evaluation graph policy",
            "Independent graph per transaction"
        )

        report.item(
            "Evaluation graph source",
            snapshot_path.name
        )

        report.item(
            "Evaluation nodes",
            G.number_of_nodes()
        )

        report.item(
            "Evaluation channels",
            G.number_of_edges()
        )

        report.add("")

        report.add(
            "Evaluation graphs are copied from the original "
            "snapshot for each payment and receive their own "
            "failure probabilities."
        )

        complete_step(
            6,
            "STEP 6 - PREPARE EVALUATION GRAPHS",
            report,
            "EVALUATION GRAPHS READY"
        )

        # ====================================================
        # STEP 7 - RUNTIME ENVIRONMENT
        # ====================================================

        set_stage(
            "STEP 7 - CREATE RUNTIME ENVIRONMENTS"
        )

        report.section(
            "STEP 7 - END-TO-END ROUTING"
        )

        begin_step(
            7,
            "STEP 7 - CREATE RUNTIME ENVIRONMENTS",
            report,
            f"Preparing {evaluation_transaction_count} independent routing environments..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 7 - MULTI-TRANSACTION END-TO-END ROUTING"
        )

        print(
            "============================================================"
        )

        report.item(
            "Network dynamics",
            "One independent instance per transaction"
        )

        report.item(
            "Failure model",
            "One independent instance per transaction"
        )

        report.item(
            "Payment simulator",
            "Created internally by RoutingEnv from "
            "the transaction-specific failure model and dynamics"
        )

        report.item(
            "PPO runtime environment",
            "One independent RoutingEnv per transaction"
        )

        report.item(
            "PPO model",
            "One shared trained/loaded model"
        )

        report.item(
            "Evaluation count",
            len(
                evaluation_transactions
            )
        )

        report.item(
            "PPO predictions per transaction",
            "1"
        )

        report.item(
            "Top-K",
            "5 (fixed by RoutingEnv)"
        )

        report.item(
            "Bucket candidates",
            "Up to 5 candidates per transaction"
        )

        complete_step(
            7,
            "STEP 7 - CREATE RUNTIME ENVIRONMENTS",
            report,
            "RUNTIME ENVIRONMENT POLICY READY"
        )

        # ====================================================
        # STEP 8 - PPO DECISIONS
        # ====================================================

        set_stage(
            "STEP 8 - PPO ROUTING DECISIONS"
        )

        report.section(
            "STEP 8 - PPO ROUTING DECISIONS"
        )

        begin_step(
            8,
            "STEP 8 - PPO ROUTING DECISIONS",
            report,
            "One PPO decision will be made for every transaction..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 8 - PPO ROUTING DECISIONS"
        )

        print(
            "============================================================"
        )

        report.item(
            "Transactions",
            len(
                evaluation_transactions
            )
        )

        report.item(
            "Expected PPO predictions",
            len(
                evaluation_transactions
            )
        )

        report.item(
            "Expected PPO predictions per transaction",
            "1"
        )

        report.item(
            "Top-K",
            "5"
        )

        report.item(
            "PPO action",
            "eta only"
        )

        report.add("")

        report.add(
            "PPO does not select a fixed Bucket candidate."
        )

        report.add(
            "For every transaction, PPO predicts eta exactly once."
        )

        report.add(
            "The downstream RoutingEnv then performs the fixed "
            "Top-K=5 candidate generation and Bucket pipeline."
        )

        report.add(
            "The Bucket candidate that succeeds is not predetermined."
        )

        complete_step(
            8,
            "STEP 8 - PPO ROUTING DECISIONS",
            report,
            "PPO DECISION STAGE READY"
        )

        # ====================================================
        # STEP 9 - MULTI-TRANSACTION PIPELINE
        # ====================================================

        set_stage(
            "STEP 9 - EXECUTE MULTI-TRANSACTION ROUTING"
        )

        report.section(
            "STEP 9 - BUCKET / PAYMENT / FAILURE / BACKTRACKING"
        )

        begin_step(
            9,
            "STEP 9 - EXECUTE MULTI-TRANSACTION ROUTING",
            report,
            f"Executing all {evaluation_transaction_count} payment scenarios..."
        )

        print(
            "\n============================================================"
        )

        print(
            "STEP 9 - MULTI-TRANSACTION ROUTING"
        )

        print(
            "============================================================"
        )

        print(
            "Pipeline:"
        )

        print(
            "PPO -> Adaptive Routing -> Top-K=5 -> "
            "Bucket -> Payment -> Failure -> Backtracking"
        )

        report.add("")

        report.add(
            "Each held-out transaction is evaluated independently."
        )

        report.add(
            "Each transaction receives exactly one PPO inference."
        )

        report.add(
            "Each PPO decision produces eta only."
        )

        report.add(
            "Top-K remains fixed at 5 inside RoutingEnv."
        )

        report.add(
            "The five selected candidates are handled by the "
            "Bucket/payment/backtracking pipeline."
        )

        report.add(
            "No successful Bucket candidate is predetermined."
        )

        report.add(
            "The successful candidate may therefore be #1, #2, "
            "#3, #4, #5, or no candidate if the Bucket is exhausted."
        )

        for episode_index, transaction in enumerate(
            evaluation_transactions,
            start=1
        ):

            episode_start = time.time()

            source = str(
                transaction.source
            )

            destination = str(
                transaction.destination
            )

            amount = float(
                transaction.amount
            )

            tx_id = safe_value(
                getattr(
                    transaction,
                    "tx_id",
                    episode_index
                ),
                episode_index
            )

            print(
                "\n------------------------------------------------------------"
            )

            print(
                f"EVALUATION TRANSACTION "
                f"{episode_index}/{evaluation_transaction_count}"
            )

            print(
                "------------------------------------------------------------"
            )

            print(
                f"Transaction ID : {tx_id}"
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

            # ------------------------------------------------
            # Independent graph
            # ------------------------------------------------

            G_eval = copy.deepcopy(
                G
            )

            evaluation_seed = (
                seed
                +
                1001
                +
                episode_index
            )

            assign_failure_probabilities(

                G_eval,

                payment_failure_rate,

                evaluation_seed

            )

            # ------------------------------------------------
            # Independent runtime objects
            # ------------------------------------------------

            dynamics = NetworkDynamics(
                G_eval
            )

            failure_model = FailureModel(
                seed=evaluation_seed
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # RoutingEnv internally constructs the
            # PaymentSimulator using the exact failure_model
            # and network_dynamics supplied below.
            #
            # This avoids having an unused second simulator.
            # ------------------------------------------------

            runtime_cfg = copy.deepcopy(
                cfg
            )

            runtime_env = RoutingEnv(

                G=G_eval,

                transactions=[transaction],

                heuristic_fn=lnd_cost,

                config=runtime_cfg,

                mode="eval",

                failure_model=failure_model,

                network_dynamics=dynamics

            )

            try:

                # --------------------------------------------
                # Attach same PPO model to this environment
                # --------------------------------------------

                model.set_env(
                    runtime_env
                )

                # --------------------------------------------
                # Reset
                # --------------------------------------------

                observation, reset_info = (
                    runtime_env.reset(
                        seed=evaluation_seed
                    )
                )

                # --------------------------------------------
                # PPO prediction
                #
                # EXACTLY ONE prediction for this transaction.
                # --------------------------------------------

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

                prediction_time_total += (
                    prediction_time
                )

                # --------------------------------------------
                # Execute complete pipeline
                #
                # The same PPO action is passed once to step().
                # --------------------------------------------

                execution_start = time.time()

                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = runtime_env.step(
                    action
                )

                episode_execution_time = (
                    time.time()
                    -
                    execution_start
                )

                execution_time_total += (
                    episode_execution_time
                )

                # --------------------------------------------
                # Extract result
                # --------------------------------------------

                result = extract_episode_result(

                    transaction,

                    reward,

                    terminated,

                    truncated,

                    info,

                    episode_index

                )

                result[
                    "prediction_time"
                ] = prediction_time

                result[
                    "pipeline_time"
                ] = episode_execution_time

                result[
                    "evaluation_seed"
                ] = evaluation_seed

                result[
                    "reset_info"
                ] = reset_info

                evaluation_results.append(
                    result
                )

                # --------------------------------------------
                # Console result
                # --------------------------------------------

                print(
                    f"PPO eta        : "
                    f"{result['eta']}"
                )

                print(
                    f"Top-K          : "
                    f"{result['top_k']}"
                )

                print(
                    f"Candidates     : "
                    f"{result['candidate_count']}"
                )

                print(
                    f"Usable         : "
                    f"{result['usable_candidate_count']}"
                )

                print(
                    f"Bucket         : "
                    f"{result['bucket_size']}"
                )

                print(
                    f"Attempts       : "
                    f"{result['attempt_count']}"
                )

                print(
                    f"Partial BT     : "
                    f"{result['partial_backtrack_count']}"
                )

                print(
                    f"Full reroute   : "
                    f"{result['full_reroute_count']}"
                )

                print(
                    f"Payment        : "
                    f"{'SUCCESS' if result['success'] else 'FAILED'}"
                )

                if result[
                    "successful_bucket_rank"
                ] is not None:

                    print(
                        f"Successful Bucket candidate: "
                        f"#{result['successful_bucket_rank']}"
                    )

                else:

                    if result["success"]:

                        print(
                            "Successful route source: "
                            f"{result.get('successful_route_source') or 'UNKNOWN'}"
                        )

                    else:

                        print(
                            "Successful Bucket candidate: NONE"
                        )

                print(
                    f"Reward         : "
                    f"{result['reward']}"
                )

                print(
                    f"PPO predictions for this transaction: 1"
                )

                print(
                    f"Pipeline time  : "
                    f"{episode_execution_time:.4f} seconds"
                )

            finally:

                try:

                    runtime_env.close()

                except Exception as close_exc:

                    print(
                        "\nWarning: evaluation environment "
                        f"{episode_index} could not be closed cleanly: "
                        f"{close_exc}"
                    )

                runtime_env = None

            episode_elapsed = (
                time.time()
                -
                episode_start
            )

            # ----------------------------------------------
            # Per-transaction report
            # ----------------------------------------------

            report.section(
                f"EVALUATION TRANSACTION "
                f"{episode_index}/{evaluation_transaction_count}"
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
                "Payment amount",
                result["amount"]
            )

            report.item(
                "Evaluation seed",
                evaluation_seed
            )

            report.item(
                "PPO predictions",
                "1"
            )

            report.item(
                "PPO eta",
                result["eta"]
            )

            report.item(
                "Requested Top-K",
                result["top_k"]
            )

            report.item(
                "Generated candidates",
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
                "Payment status",
                "SUCCESS"
                if result["success"]
                else
                "FAILED"
            )

            report.item(
                "Attempts",
                result["attempt_count"]
            )

            report.item(
                "Backtracks",
                result["backtrack_count"]
            )

            report.item(
                "Partial backtracks",
                result["partial_backtrack_count"]
            )

            report.item(
                "Successful partial backtracks",
                result["partial_backtrack_success"]
            )

            report.item(
                "Full reroutes",
                result["full_reroute_count"]
            )

            report.item(
                "Successful Bucket route",
                (
                    f"{result['successful_bucket_rank']} / "
                    f"{result['bucket_size']}"
                    if result["successful_bucket_rank"]
                    is not None
                    else
                    (
                        "PARTIAL BACKTRACK"
                        if result.get("successful_route_source")
                        == "partial_backtrack"
                        else
                        "UNKNOWN"
                        if result["success"]
                        else "NONE"
                    )
                )
            )

            report.item(
                "Successful route source",
                result.get("successful_route_source") or (
                    "unknown" if result["success"] else "none"
                )
            )

            report.item(
                "Failure probability",
                result["failure_probability"]
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
                "Hops",
                result["hops"]
            )

            report.item(
                "Prediction time (seconds)",
                round(
                    result["prediction_time"],
                    6
                )
            )

            report.item(
                "Pipeline time (seconds)",
                round(
                    result["pipeline_time"],
                    4
                )
            )

            report.item(
                "Total transaction time (seconds)",
                round(
                    episode_elapsed,
                    4
                )
            )

            report.item(
                "Final path",
                (
                    " -> ".join(
                        str(node)
                        for node in result["path"]
                    )
                    if result["path"]
                    else
                    "NONE"
                )
            )

        # ====================================================
        # Aggregate evaluation
        # ====================================================

        # ----------------------------------------------------
        # The evaluation must retain exactly the configured number of rows.
        # ----------------------------------------------------

        if len(evaluation_results) != evaluation_transaction_count:

            raise RuntimeError(

                "Evaluation completed with "
                f"{len(evaluation_results)} results instead of "
                f"the configured count {evaluation_transaction_count}."

            )

        aggregate = aggregate_evaluation_results(
            evaluation_results
        )

        print_evaluation_summary(
            aggregate
        )

        # ====================================================
        # STEP 9 COMPLETION
        # ====================================================

        report.section(
            "MULTI-TRANSACTION EVALUATION SUMMARY"
        )

        report.item(
            "Transactions evaluated",
            aggregate["total_transactions"]
        )

        report.item(
            "Expected transactions",
            str(evaluation_transaction_count)
        )

        report.item(
            "Successful payments",
            aggregate["successful_transactions"]
        )

        report.item(
            "Failed payments",
            aggregate["failed_transactions"]
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
                aggregate["average_attempts"],
                4
            )
        )

        report.item(
            "Average hops",
            format_metric(
                aggregate["average_hops"],
                4
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
                aggregate["average_delay"],
                4
            )
        )

        report.item("P90 route delay (successful)", aggregate["p90_delay_success"])
        report.item("P95 route delay (successful)", aggregate["p95_delay_success"])
        report.item("P90 route delay (all routed)", aggregate["p90_delay_routed"])
        report.item("P95 route delay (all routed)", aggregate["p95_delay_routed"])

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
            "Average candidates",
            format_metric(
                aggregate["average_candidates"],
                4
            )
        )

        report.item(
            "Average usable candidates",
            format_metric(
                aggregate["average_usable_candidates"],
                4
            )
        )

        report.item(
            "Average Bucket size",
            format_metric(
                aggregate["average_bucket_size"],
                4
            )
        )

        report.item(
            "Total attempts",
            aggregate["total_attempts"]
        )

        report.item(
            "Total partial backtracks",
            aggregate["total_partial_backtracks"]
        )

        report.item(
            "Successful partial backtracks",
            aggregate["total_partial_backtrack_success"]
        )

        report.item(
            "Total full reroutes",
            aggregate["total_full_reroutes"]
        )

        report.item(
            "Bucket exhausted",
            aggregate["bucket_exhausted"]
        )

        report.add("")

        report.add(
            "Successful Bucket candidate distribution:"
        )

        if aggregate[
            "successful_bucket_rank_counts"
        ]:

            for rank in sorted(
                aggregate[
                    "successful_bucket_rank_counts"
                ]
            ):

                count = aggregate[
                    "successful_bucket_rank_counts"
                ][
                    rank
                ]

                percentage = (

                    count
                    /
                    max(
                        1,
                        aggregate[
                            "successful_transactions"
                        ]
                    )
                    *
                    100.0

                )

                report.item(
                    f"Candidate #{rank}",
                    f"{count} "
                    f"({percentage:.2f}% of successful payments)"
                )

        else:

            report.add(
                "No explicit successful Bucket candidate "
                "rank was reported by RoutingEnv."
            )

        if aggregate[
            "successful_rank_unknown"
        ] > 0:

            report.item(
                "Successful candidate rank unknown",
                aggregate[
                    "successful_rank_unknown"
                ]
            )

        report.add("")

        report.add(
            f"{evaluation_transaction_count} independent evaluation transactions "
            "were processed."
        )

        report.add(
            "Each transaction received exactly one PPO prediction."
        )

        report.add(
            "The PPO action controls eta only."
        )

        report.add(
            "Top-K is fixed at 5 inside RoutingEnv."
        )

        report.add(
            "The evaluation does not force a particular "
            "Bucket candidate to succeed."
        )

        report.add(
            "Different transactions may succeed at different "
            "Bucket positions, or all candidates may fail."
        )

        complete_step(
            9,
            "STEP 9 - EXECUTE MULTI-TRANSACTION ROUTING",
            report,
            f"{evaluation_transaction_count}-TRANSACTION ROUTING COMPLETED"
        )

        # ====================================================
        # STEP 10 - FINAL EVALUATION RESULT
        # ====================================================

        set_stage(
            "STEP 10 - FINAL EVALUATION RESULT"
        )

        report.section(
            "STEP 10 - FINAL EVALUATION RESULT"
        )

        begin_step(
            10,
            "STEP 10 - FINAL EVALUATION RESULT",
            report,
            "Processing aggregate evaluation result..."
        )

        # ----------------------------------------------------
        # Overall status
        # ----------------------------------------------------

        all_success = (
            aggregate[
                "successful_transactions"
            ]
            ==
            aggregate[
                "total_transactions"
            ]
        )

        any_success = (
            aggregate[
                "successful_transactions"
            ]
            >
            0
        )

        report.item(
            "Evaluation status",
            (
                "ALL PAYMENTS SUCCESSFUL"
                if all_success
                else
                "MIXED SUCCESS/FAILURE"
                if any_success
                else
                "ALL PAYMENTS FAILED"
            )
        )

        report.item(
            "Payment outcome",
            (
                "ALL SUCCESSFUL"
                if all_success
                else "MIXED"
                if any_success
                else "ALL FAILED"
            )
        )

        report.item(
            "Transactions evaluated",
            aggregate[
                "total_transactions"
            ]
        )

        report.item(
            "Successful payments",
            aggregate[
                "successful_transactions"
            ]
        )

        report.item(
            "Failed payments",
            aggregate[
                "failed_transactions"
            ]
        )

        report.item(
            "Overall success rate",
            f"{aggregate['success_rate']:.2f}%"
        )

        report.item(
            "Overall failure rate",
            f"{aggregate['failure_rate']:.2f}%"
        )

        report.item(
            "Average attempts",
            format_metric(
                aggregate["average_attempts"],
                4
            )
        )

        report.item(
            "Average hops",
            format_metric(
                aggregate["average_hops"],
                4
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
                aggregate["average_delay"],
                4
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
            "Total partial backtracks",
            aggregate[
                "total_partial_backtracks"
            ]
        )

        report.item(
            "Successful partial backtracks",
            aggregate[
                "total_partial_backtrack_success"
            ]
        )

        report.item(
            "Total full reroutes",
            aggregate[
                "total_full_reroutes"
            ]
        )

        report.item(
            "Bucket exhausted",
            aggregate[
                "bucket_exhausted"
            ]
        )

        report.add("")

        report.add(
            "Per-transaction result table:"
        )

        for result in evaluation_results:

            successful_rank = (
                f"#{result['successful_bucket_rank']}"
                if result[
                    "successful_bucket_rank"
                ] is not None
                else
                (
                    "PARTIAL BACKTRACK"
                    if result.get("successful_route_source")
                    == "partial_backtrack"
                    else "UNKNOWN"
                    if result["success"]
                    else "NONE"
                )
            )

            report.item(
                (
                    f"TX #{result['episode_index']} "
                    f"(id={result['transaction_id']})"
                ),
                (
                    f"{'SUCCESS' if result['success'] else 'FAILED'} | "
                    f"source={result['source']} | "
                    f"destination={result['destination']} | "
                    f"amount={result['amount']} | "
                    f"eta={result['eta']} | "
                    f"candidates={result['candidate_count']} | "
                    f"bucket={result['bucket_size']} | "
                    f"attempts={result['attempt_count']} | "
                    f"route={successful_rank} | "
                    f"route_source={result.get('successful_route_source') or 'N/A'} | "
                    f"reward={result['reward']:.6f}"
                )
            )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        report.section(
            "FINAL STATUS"
        )

        if all_success:

            report.add(
                "ALL PAYMENT SCENARIOS SUCCESSFUL"
            )

            report.add(
                f"All {aggregate['total_transactions']} "
                "independent payment scenarios reached their "
                "destinations."
            )

        elif any_success:

            report.add(
                "MULTI-TRANSACTION EVALUATION COMPLETED"
            )

            report.add(
                f"{aggregate['successful_transactions']} of "
                f"{aggregate['total_transactions']} payment "
                "scenarios succeeded."
            )

            report.add(
                f"{aggregate['failed_transactions']} payment "
                + (
                    "scenario failed."
                    if aggregate["failed_transactions"] == 1
                    else "scenarios failed."
                )
            )

        else:

            report.add(
                "ALL PAYMENT SCENARIOS FAILED"
            )

            report.add(
                "None of the evaluated payment scenarios "
                "successfully reached the destination."
            )

        report.add("")

        report.add(
            "Important evaluation property:"
        )

        report.add(
            f"Exactly {evaluation_transaction_count} independent payment transactions "
            "were evaluated."
        )

        report.add(
            "Each transaction received exactly one PPO prediction."
        )

        report.add(
            "Top-K was fixed at 5 for every transaction."
        )

        report.add(
            "The evaluation does not assume that Bucket "
            "candidate #3 must succeed."
        )

        report.add(
            "Different transactions may succeed at different "
            "Bucket positions, or all candidates may fail."
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
            else
            "NORMAL TRAINING"
        )

        report.item(
            "Evaluation transactions",
            str(evaluation_transaction_count)
        )

        report.item(
            "Expected PPO predictions",
            str(evaluation_transaction_count)
        )

        report.item(
            "Actual PPO predictions",
            len(
                evaluation_results
            )
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
            "Total PPO prediction time (seconds)",
            round(
                prediction_time_total,
                6
            )
        )

        report.item(
            "Total payment pipeline time (seconds)",
            round(
                execution_time_total,
                4
            )
        )

        report.item(
            "Average payment pipeline time (seconds)",
            round(
                execution_time_total
                /
                max(
                    1,
                    len(
                        evaluation_results
                    )
                ),
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
            "Evaluation transactions",
            evaluation_transaction_count
        )

        report.item(
            "PPO predictions",
            len(evaluation_results)
        )

        if FAST_VALIDATION:

            report.item(
                "Result interpretation",
                "Smoke validation only; not a scientific performance estimate"
            )

        report.item(
            "Top-K",
            "5"
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
                Path(
                    "report.txt"
                ).resolve()
            )
        )

        report_path = report.save()

        print_progress(
            progress=100.0,
            step=10,
            step_name="RUN COMPLETED",
            message=f"{evaluation_transaction_count}-TRANSACTION END-TO-END EVALUATION COMPLETED"
        )

        print(
            "\n============================================================"
        )

        print(
            f"{evaluation_transaction_count}-TRANSACTION END-TO-END EVALUATION FINISHED"
        )

        print(
            "Run status           : COMPLETED"
        )

        print(
            "Payment outcome      : "
            f"{'ALL SUCCESSFUL' if all_success else 'MIXED' if any_success else 'ALL FAILED'}"
        )

        print(
            "============================================================"
        )

        print(
            f"Execution mode       : "
            f"{'FAST VALIDATION' if FAST_VALIDATION else 'NORMAL TRAINING'}"
        )

        print(
            f"Snapshot             : "
            f"{snapshot_path.name}"
        )

        print(
            f"Transactions         : "
            f"{aggregate['total_transactions']} / {evaluation_transaction_count}"
        )

        print(
            f"PPO predictions      : "
            f"{len(evaluation_results)} / {evaluation_transaction_count}"
        )

        print(
            f"Top-K                : 5"
        )

        print(
            f"Successful payments  : "
            f"{aggregate['successful_transactions']}"
        )

        print(
            f"Failed payments      : "
            f"{aggregate['failed_transactions']}"
        )

        print(
            f"Success rate         : "
            f"{aggregate['success_rate']:.2f}%"
        )

        print(
            f"Average attempts     : "
            f"{format_metric(aggregate['average_attempts'], 4)}"
        )

        print(
            f"Average hops         : "
            f"{format_metric(aggregate['average_hops'], 4)}"
        )

        print(
            f"Average fee          : "
            f"{format_metric(aggregate['average_fee'], 6)}"
        )

        print(
            f"Average delay        : "
            f"{format_metric(aggregate['average_delay'], 4)}"
        )

        print(
            f"Average carbon       : "
            f"{format_metric(aggregate['average_carbon'], 6)}"
        )

        print(
            f"Average reward       : "
            f"{format_metric(aggregate['average_reward'], 6)}"
        )

        print(
            f"Partial backtracks   : "
            f"{aggregate['total_partial_backtracks']}"
        )

        print(
            f"Full reroutes        : "
            f"{aggregate['total_full_reroutes']}"
        )

        print(
            f"PPO train time       : "
            f"{training_time:.4f} seconds"
        )

        print(
            f"PPO load time        : "
            f"{model_load_time:.4f} seconds"
        )

        print(
            f"PPO prediction time  : "
            f"{prediction_time_total:.6f} seconds"
        )

        print(
            f"Pipeline time        : "
            f"{execution_time_total:.4f} seconds"
        )

        print(
            f"Total time           : "
            f"{total_time:.4f} seconds"
        )

        print(
            "\nSuccessful Bucket candidate distribution:"
        )

        if aggregate[
            "successful_bucket_rank_counts"
        ]:

            for rank in sorted(
                aggregate[
                    "successful_bucket_rank_counts"
                ]
            ):

                print(
                    f"  Candidate #{rank}: "
                    f"{aggregate['successful_bucket_rank_counts'][rank]}"
                )

        else:

            print(
                "  Candidate rank was not explicitly "
                "reported by RoutingEnv."
            )

        if aggregate[
            "successful_rank_unknown"
        ] > 0:

            print(
                f"  Unknown successful candidate rank: "
                f"{aggregate['successful_rank_unknown']}"
            )

        print(
            "\nFull report saved to:"
        )

        print(
            report_path.resolve()
        )

        return {

            "execution_completed":
                True,

            "routing_outcome":
                (
                    "all_success"
                    if all_success
                    else "mixed"
                    if any_success
                    else "all_failed"
                ),

            "success":
                all_success,

            "all_success":
                all_success,

            "any_success":
                any_success,

            "total_transactions":
                aggregate[
                    "total_transactions"
                ],

            "successful_transactions":
                aggregate[
                    "successful_transactions"
                ],

            "failed_transactions":
                aggregate[
                    "failed_transactions"
                ],

            "success_rate":
                aggregate[
                    "success_rate"
                ],

            "failure_rate":
                aggregate[
                    "failure_rate"
                ],

            "average_attempts":
                aggregate[
                    "average_attempts"
                ],

            "average_hops":
                aggregate[
                    "average_hops"
                ],

            "average_fee":
                aggregate[
                    "average_fee"
                ],

            "average_delay":
                aggregate[
                    "average_delay"
                ],

            "p90_delay_success":
                aggregate["p90_delay_success"],

            "p95_delay_success":
                aggregate["p95_delay_success"],

            "p90_delay_routed":
                aggregate["p90_delay_routed"],

            "p95_delay_routed":
                aggregate["p95_delay_routed"],

            "average_carbon":
                aggregate[
                    "average_carbon"
                ],

            "average_reward":
                aggregate[
                    "average_reward"
                ],

            "average_eta":
                aggregate[
                    "average_eta"
                ],

            "total_partial_backtracks":
                aggregate[
                    "total_partial_backtracks"
                ],

            "total_partial_backtrack_success":
                aggregate[
                    "total_partial_backtrack_success"
                ],

            "total_full_reroutes":
                aggregate[
                    "total_full_reroutes"
                ],

            "bucket_exhausted":
                aggregate[
                    "bucket_exhausted"
                ],

            "successful_bucket_rank_counts":
                aggregate[
                    "successful_bucket_rank_counts"
                ],

            "evaluation_results":
                evaluation_results,

            "training_time":
                training_time,

            "model_load_time":
                model_load_time,

            "prediction_time":
                prediction_time_total,

            "pipeline_time":
                execution_time_total,

            "total_time":
                total_time,

            "execution_mode":
                (
                    "FAST VALIDATION"
                    if FAST_VALIDATION
                    else
                    "NORMAL TRAINING"
                ),

        }

    except KeyboardInterrupt as exc:

        write_runtime_error(
            exc,
            report,
            run_status="INTERRUPTED",
        )

        print(
            "\n============================================================"
        )

        print(
            "END-TO-END EXECUTION INTERRUPTED"
        )

        print(
            "Run status: INTERRUPTED"
        )

        print(
            f"Stage: {CURRENT_STAGE}"
        )

        print(
            f"Partial report saved to: {REPORT_FILE.resolve()}"
        )

        raise

    except Exception as exc:

        write_runtime_error(
            exc,
            report,
            run_status="FAILED",
        )

        print(
            "\n============================================================"
        )

        print(
            "END-TO-END EXECUTION FAILED"
        )

        print(
            "Run status: FAILED"
        )

        print(
            "============================================================"
        )

        print(
            f"Step     : "
            f"{CURRENT_STEP}/{TOTAL_STEPS}"
        )

        print(
            f"Progress : "
            f"{CURRENT_PROGRESS:.2f}%"
        )

        print(
            f"Stage    : "
            f"{CURRENT_STAGE}"
        )

        print(
            f"Error    : "
            f"{type(exc).__name__}: {exc}"
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
