"""Shared graph, PPO loading and reporting helpers for MAIN TEST."""

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
from Runtime.recovery_reporting import RECOVERY_COUNT_FIELDS, recovery_details

import torch
from stable_baselines3 import PPO

# Stable-Baselines3 modules are imported explicitly because
# PPO.load() resolves load_from_zip_file through BaseAlgorithm.
import stable_baselines3.common.save_util as sb3_save_util
import stable_baselines3.common.base_class as sb3_base_class


# ============================================================
# Global Runtime State
# ============================================================

REPORT_FILE = Path(
    os.environ.get("RL_REPORT_FILE", "report.txt")
)

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
            "    py main-test.py\n\n"

            "After the model is created, use fast validation:\n\n"

            "    set RL_FAST_VALIDATION=1\n"
            "    py main-test.py"

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

        **recovery_details(info),

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

        **{
            "total_" + key: sum(safe_int(result.get(key), 0) for result in results)
            for key in RECOVERY_COUNT_FIELDS
        },

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
        f"Recovery checks after failure: "
        f"{aggregate['total_partial_backtracks']}"
    )

    print(
        f"Partial backtrack successes: "
        f"{aggregate['total_partial_backtrack_success']}"
    )

    for label, key in (
        ("Actual partial continuation attempts", "partial_backtrack_retry_count"),
        ("Retries starting at source", "source_retry_count"),
        ("Forwarded prefix hops reused", "preserved_prefix_hops"),
        ("Backtracked hops", "backtracked_hops"),
    ):
        print(f"{label}: {aggregate.get('total_' + key, 0)}")

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



# ============================================================
# Entry Point
# ============================================================
