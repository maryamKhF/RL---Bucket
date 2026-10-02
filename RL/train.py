
# RL/train.py

from pathlib import Path
import json
import os
import time

from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

from .ppo_agent import build_ppo
from .environment import RoutingEnv


# ============================================================
# Runtime Training Configuration
# ============================================================

# ------------------------------------------------------------
# Quick training
# ------------------------------------------------------------

FAST_TRAINING = (
    os.environ.get(
        "RL_FAST_TRAINING",
        "1"
    ).strip().lower()
    in {
        "1",
        "true",
        "yes",
        "on"
    }
)


# ============================================================
# Quick Training Defaults
# ============================================================

QUICK_TOTAL_TIMESTEPS = int(
    os.environ.get(
        "RL_QUICK_TIMESTEPS",
        "512"
    )
)

QUICK_N_STEPS = int(
    os.environ.get(
        "RL_QUICK_N_STEPS",
        "128"
    )
)

QUICK_BATCH_SIZE = int(
    os.environ.get(
        "RL_QUICK_BATCH_SIZE",
        "64"
    )
)


# ============================================================
# Checkpoint Configuration
# ============================================================

# Default:
#
#     models/checkpoints/
#
# Can be overridden through config.yaml:
#
#     rl:
#       checkpoint_dir: models/checkpoints
#
# or environment:
#
#     RL_CHECKPOINT_FREQUENCY=128
#
# The environment variable has priority.

DEFAULT_CHECKPOINT_FREQUENCY = int(
    os.environ.get(
        "RL_CHECKPOINT_FREQUENCY",
        "128"
    )
)


# ============================================================
# PPO Progress Callback
# ============================================================

class PPOProgressCallback(BaseCallback):
    """
    Lightweight PPO training progress monitor.

    Displays real PPO timestep progress instead of only
    stage-level progress.
    """

    def __init__(
        self,
        total_timesteps,
        print_interval_percent=1.0,
        verbose=0
    ):
        super().__init__(
            verbose=verbose
        )

        self.total_timesteps_target = max(
            1,
            int(total_timesteps)
        )

        self.print_interval_percent = max(
            0.1,
            float(print_interval_percent)
        )

        self.start_time = None

        self.last_printed_percent = -1.0

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def _on_training_start(self):

        self.start_time = time.time()

        self.last_printed_percent = -1.0

        print(
            "\n============================================================"
        )

        print(
            "PPO TRAINING STARTED"
        )

        print(
            "============================================================"
        )

        print(
            f"Target timesteps : "
            f"{self.total_timesteps_target}"
        )

    # --------------------------------------------------------
    # Every environment step
    # --------------------------------------------------------

    def _on_step(self):

        current_timestep = int(
            self.num_timesteps
        )

        total = max(
            1,
            self.total_timesteps_target
        )

        progress = (
            current_timestep
            /
            total
            *
            100.0
        )

        progress = min(
            100.0,
            progress
        )

        if (
            progress
            >=
            self.last_printed_percent
            +
            self.print_interval_percent
        ) or current_timestep >= total:

            elapsed = (
                time.time()
                -
                self.start_time
            )

            if elapsed > 0:

                fps = (
                    current_timestep
                    /
                    elapsed
                )

            else:

                fps = 0.0

            remaining = max(
                0,
                total - current_timestep
            )

            if fps > 0:

                eta_seconds = (
                    remaining
                    /
                    fps
                )

            else:

                eta_seconds = 0.0

            print(
                "\n[PPO PROGRESS]"
            )

            print(
                "------------------------------------------------------------"
            )

            print(
                f"Timestep : "
                f"{current_timestep} / {total}"
            )

            print(
                f"Progress : "
                f"{progress:6.2f}%"
            )

            print(
                f"Elapsed  : "
                f"{format_seconds(elapsed)}"
            )

            print(
                f"FPS      : "
                f"{fps:.3f}"
            )

            print(
                f"ETA      : "
                f"{format_seconds(eta_seconds)}"
            )

            print(
                "------------------------------------------------------------"
            )

            self.last_printed_percent = progress

        return True

    # --------------------------------------------------------
    # End of training
    # --------------------------------------------------------

    def _on_training_end(self):

        elapsed = (
            time.time()
            -
            self.start_time
            if self.start_time is not None
            else 0.0
        )

        final_timestep = int(
            self.num_timesteps
        )

        if elapsed > 0:

            fps = (
                final_timestep
                /
                elapsed
            )

        else:

            fps = 0.0

        print(
            "\n============================================================"
        )

        print(
            "PPO TRAINING FINISHED"
        )

        print(
            "============================================================"
        )

        print(
            f"Final timestep : "
            f"{final_timestep}"
        )

        print(
            f"Elapsed        : "
            f"{format_seconds(elapsed)}"
        )

        print(
            f"Average FPS    : "
            f"{fps:.3f}"
        )


# ============================================================
# Time Formatting
# ============================================================

def format_seconds(seconds):

    seconds = max(
        0.0,
        float(seconds)
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600)
        //
        60
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


# ============================================================
# Training Timestep Configuration
# ============================================================

def resolve_training_configuration(
    rl_cfg
):
    """
    Resolve PPO training parameters.

    FAST TRAINING:
        Small timestep count and smaller rollout.

    NORMAL TRAINING:
        Use config.yaml values.
    """

    configured_timesteps = int(
        rl_cfg.get(
            "total_timesteps",
            10000
        )
    )

    configured_n_steps = int(
        rl_cfg.get(
            "n_steps",
            2048
        )
    )

    configured_batch_size = int(
        rl_cfg.get(
            "batch_size",
            64
        )
    )

    if FAST_TRAINING:

        total_timesteps = max(
            1,
            QUICK_TOTAL_TIMESTEPS
        )

        n_steps = max(
            1,
            min(
                QUICK_N_STEPS,
                total_timesteps
            )
        )

        batch_size = max(
            1,
            min(
                QUICK_BATCH_SIZE,
                n_steps
            )
        )

        if n_steps % batch_size != 0:

            valid_batch_sizes = [

                value

                for value in [
                    64,
                    32,
                    16,
                    8,
                    4,
                    2,
                    1
                ]

                if value <= n_steps
                and n_steps % value == 0

            ]

            if valid_batch_sizes:

                batch_size = valid_batch_sizes[0]

            else:

                batch_size = 1

        return {

            "total_timesteps":
                total_timesteps,

            "n_steps":
                n_steps,

            "batch_size":
                batch_size,

            "mode":
                "FAST TRAINING"

        }

    # --------------------------------------------------------
    # Normal training
    # --------------------------------------------------------

    total_timesteps = max(
        1,
        configured_timesteps
    )

    n_steps = max(
        1,
        min(
            configured_n_steps,
            total_timesteps
        )
    )

    batch_size = max(
        1,
        configured_batch_size
    )

    if batch_size > n_steps:

        batch_size = n_steps

    return {

        "total_timesteps":
            total_timesteps,

        "n_steps":
            n_steps,

        "batch_size":
            batch_size,

        "mode":
            "NORMAL TRAINING"

    }


# ============================================================
# Checkpoint Frequency Resolution
# ============================================================

def resolve_checkpoint_frequency(
    rl_cfg,
    total_timesteps
):
    """
    Resolve checkpoint frequency.

    Priority:

        RL_CHECKPOINT_FREQUENCY
            >
        config.yaml rl.checkpoint_freq
            >
        default 128

    The final frequency is clipped so that at least one
    checkpoint can be produced during the training run.
    """

    configured_frequency = rl_cfg.get(
        "checkpoint_freq",
        DEFAULT_CHECKPOINT_FREQUENCY
    )

    try:

        checkpoint_freq = int(
            configured_frequency
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "'checkpoint_freq' must be an integer."
        )

    if checkpoint_freq <= 0:

        raise ValueError(
            "'checkpoint_freq' must be greater than zero."
        )

    total_timesteps = max(
        1,
        int(total_timesteps)
    )

    checkpoint_freq = min(
        checkpoint_freq,
        total_timesteps
    )

    return checkpoint_freq


# ============================================================
# Main Training Function
# ============================================================

def train_agent(
    G,
    transactions,
    heuristic,
    cfg,
    name,
    seed=42,
    eval_env=None
):
    """
    Train the PPO routing agent.

    Pipeline:

        Graph + Transactions
                |
                v
          RoutingEnv
                |
                v
             State
                |
                v
              PPO
                |
                v
              eta
                |
                v
        Routing / Simulation
                |
                v
             Reward
                |
                v
          PPO parameter update

    The PPO agent learns eta.

    Checkpoints are written during training so that
    long-running PPO training does not lose all progress
    if execution is interrupted.
    """

    # ========================================================
    # CONFIGURATION VALIDATION
    # ========================================================

    if "rl" not in cfg:

        raise KeyError(
            "Missing 'rl' section in configuration."
        )

    if "graph" not in cfg:

        raise KeyError(
            "Missing 'graph' section in configuration."
        )

    rl_cfg = cfg["rl"]

    graph_cfg = cfg["graph"]

    if not name:

        raise ValueError(
            "Model name cannot be empty."
        )

    if G is None:

        raise ValueError(
            "Training graph G cannot be None."
        )

    if transactions is None:

        raise ValueError(
            "Training transactions cannot be None."
        )

    if len(transactions) == 0:

        raise ValueError(
            "Training transactions cannot be empty."
        )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    model_dir = Path(
        rl_cfg.get(
            "model_dir",
            "models"
        )
    )

    log_dir = Path(
        rl_cfg.get(
            "log_dir",
            "logs"
        )
    )

    checkpoint_dir = Path(
        rl_cfg.get(
            "checkpoint_dir",
            "models/checkpoints"
        )
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # RESOLVE TRAINING PARAMETERS
    # ========================================================

    training_cfg = resolve_training_configuration(
        rl_cfg
    )

    total_timesteps = int(
        training_cfg["total_timesteps"]
    )

    n_steps = int(
        training_cfg["n_steps"]
    )

    batch_size = int(
        training_cfg["batch_size"]
    )

    training_mode = training_cfg[
        "mode"
    ]

    checkpoint_freq = resolve_checkpoint_frequency(
        rl_cfg,
        total_timesteps
    )

    print(
        "\n============================================================"
    )

    print(
        "PPO TRAINING CONFIGURATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Training mode       : {training_mode}"
    )

    print(
        f"Timesteps          : {total_timesteps}"
    )

    print(
        f"n_steps            : {n_steps}"
    )

    print(
        f"batch_size         : {batch_size}"
    )

    print(
        f"Transactions       : {len(transactions)}"
    )

    print(
        f"Checkpoint freq    : {checkpoint_freq}"
    )

    print(
        f"Checkpoint dir     : "
        f"{checkpoint_dir.resolve()}"
    )

    # ========================================================
    # TRAINING ENVIRONMENT
    # ========================================================

    train_env = RoutingEnv(

        G=G,

        transactions=transactions,

        heuristic_fn=heuristic,

        config=cfg,

        mode="train"

    )

    # ========================================================
    # ENVIRONMENT VALIDATION
    # ========================================================

    if not hasattr(
        train_env,
        "observation_space"
    ):

        train_env.close()

        raise AttributeError(
            "Training environment does not define "
            "observation_space."
        )

    if not hasattr(
        train_env,
        "action_space"
    ):

        train_env.close()

        raise AttributeError(
            "Training environment does not define "
            "action_space."
        )

    observation_shape = (
        train_env.observation_space.shape
    )

    action_shape = (
        train_env.action_space.shape
    )

    if observation_shape is None:

        train_env.close()

        raise ValueError(
            "Invalid observation space shape."
        )

    if action_shape is None:

        train_env.close()

        raise ValueError(
            "Invalid action space shape."
        )

    # ========================================================
    # PPO MODEL
    # ========================================================

    effective_cfg = cfg.copy()

    effective_rl_cfg = dict(
        rl_cfg
    )

    effective_rl_cfg[
        "total_timesteps"
    ] = total_timesteps

    effective_rl_cfg[
        "n_steps"
    ] = n_steps

    effective_rl_cfg[
        "batch_size"
    ] = batch_size

    effective_cfg[
        "rl"
    ] = effective_rl_cfg

    model = build_ppo(

        env=train_env,

        cfg=effective_cfg,

        seed=seed

    )

    # ========================================================
    # CALLBACKS
    # ========================================================

    callbacks = []

    # ========================================================
    # PROGRESS CALLBACK
    # ========================================================

    progress_callback = PPOProgressCallback(

        total_timesteps=total_timesteps,

        print_interval_percent=1.0,

        verbose=0

    )

    callbacks.append(
        progress_callback
    )

    # ========================================================
    # CHECKPOINT CALLBACK
    # ========================================================
    #
    # IMPORTANT:
    #
    # Checkpointing is enabled in BOTH:
    #
    #     FAST TRAINING
    #     NORMAL TRAINING
    #
    # This protects progress even during short validation
    # runs and, more importantly, long research training.
    #
    # Example:
    #
    #     total_timesteps = 512
    #     checkpoint_freq = 128
    #
    # produces approximately:
    #
    #     ppo_checkpoint_128_steps.zip
    #     ppo_checkpoint_256_steps.zip
    #     ppo_checkpoint_384_steps.zip
    #     ppo_checkpoint_512_steps.zip
    #
    # ========================================================

    checkpoint_callback = CheckpointCallback(

        save_freq=checkpoint_freq,

        save_path=str(
            checkpoint_dir
        ),

        name_prefix=f"{name}_checkpoint",

        save_replay_buffer=False,

        save_vecnormalize=False

    )

    callbacks.append(
        checkpoint_callback
    )

    print(
        "\nCheckpointing : ENABLED"
    )

    print(
        f"Checkpoint every : "
        f"{checkpoint_freq} timesteps"
    )

    print(
        f"Checkpoint path   : "
        f"{checkpoint_dir.resolve()}"
    )

    # ========================================================
    # EVALUATION CALLBACK
    # ========================================================

    if eval_env is not None:

        if FAST_TRAINING:

            print(
                "\nFast training: evaluation callback "
                "disabled."
            )

        else:

            if not isinstance(
                eval_env,
                Monitor
            ):

                eval_env = Monitor(
                    eval_env
                )

            eval_freq = int(
                rl_cfg.get(
                    "eval_freq",
                    5000
                )
            )

            if eval_freq <= 0:

                raise ValueError(
                    "'eval_freq' must be greater than zero."
                )

            best_model_dir = (
                model_dir / "best"
            )

            evaluation_log_dir = (
                log_dir / "evaluation"
            )

            best_model_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            evaluation_log_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            eval_callback = EvalCallback(

                eval_env,

                best_model_save_path=str(
                    best_model_dir
                ),

                log_path=str(
                    evaluation_log_dir
                ),

                eval_freq=eval_freq,

                deterministic=True,

                render=False

            )

            callbacks.append(
                eval_callback
            )

    # ========================================================
    # TRAINING START
    # ========================================================

    start_time = time.time()

    print(
        "\n============================================================"
    )

    print(
        "PPO TRAINING IS RUNNING"
    )

    print(
        "============================================================"
    )

    print(
        f"Mode          : {training_mode}"
    )

    print(
        f"Timesteps     : {total_timesteps}"
    )

    print(
        f"n_steps       : {n_steps}"
    )

    print(
        f"Batch size    : {batch_size}"
    )

    print(
        f"Checkpoint    : every {checkpoint_freq}"
    )

    print(
        "============================================================"
    )

    try:

        model.learn(

            total_timesteps=total_timesteps,

            callback=callbacks,

            reset_num_timesteps=True

        )

    except Exception:

        print(
            "\nWARNING: PPO training interrupted."
        )

        print(
            "Previously saved checkpoints are preserved."
        )

        train_env.close()

        if eval_env is not None:

            try:

                eval_env.close()

            except Exception:

                pass

        raise

    training_time = (
        time.time()
        -
        start_time
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    model_path = (
        model_dir / name
    )

    model.save(
        str(model_path)
    )

    print(
        "\n============================================================"
    )

    print(
        "PPO MODEL SAVED"
    )

    print(
        "============================================================"
    )

    print(
        f"Model path : "
        f"{model_path.resolve()}.zip"
    )

    print(
        f"Training   : "
        f"{format_seconds(training_time)}"
    )

    # ========================================================
    # METADATA
    # ========================================================

    observation_dimension = int(
        train_env.observation_space.shape[0]
    )

    action_dimension = int(
        train_env.action_space.shape[0]
    )

    configured_top_k = graph_cfg.get(
        "top_k",
        graph_cfg.get(
            "top_k_paths",
            5
        )
    )

    metadata = {

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        "model":
            name,

        "seed":
            seed,

        "training_mode":
            training_mode,

        "fast_training":
            bool(
                FAST_TRAINING
            ),

        "timesteps":
            total_timesteps,

        "training_time_seconds":
            float(
                training_time
            ),

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        "checkpointing_enabled":
            True,

        "checkpoint_frequency":
            checkpoint_freq,

        "checkpoint_directory":
            str(
                checkpoint_dir.resolve()
            ),

        # ----------------------------------------------------
        # Training configuration
        # ----------------------------------------------------

        "n_steps":
            n_steps,

        "batch_size":
            batch_size,

        # ----------------------------------------------------
        # Observation / Action
        # ----------------------------------------------------

        "observation_shape":
            list(
                observation_shape
            ),

        "action_shape":
            list(
                action_shape
            ),

        "observation_dimension":
            observation_dimension,

        "action_dimension":
            action_dimension,

        "action_low":
            train_env.action_space.low.tolist(),

        "action_high":
            train_env.action_space.high.tolist(),

        # ----------------------------------------------------
        # Graph / State configuration
        # ----------------------------------------------------

        "neighborhood_k":
            graph_cfg.get(
                "neighborhood_k"
            ),

        "neighborhood_m":
            graph_cfg.get(
                "neighborhood_m"
            ),

        "top_k":
            configured_top_k,

        "max_hops":
            graph_cfg.get(
                "max_hops"
            ),

        # ----------------------------------------------------
        # PPO configuration
        # ----------------------------------------------------

        "hidden_layers":
            rl_cfg.get(
                "hidden_layers",
                [128, 128]
            ),

        "learning_rate":
            rl_cfg.get(
                "learning_rate",
                3e-4
            ),

        "configured_n_steps":
            rl_cfg.get(
                "n_steps",
                2048
            ),

        "configured_batch_size":
            rl_cfg.get(
                "batch_size",
                64
            ),

        "n_epochs":
            rl_cfg.get(
                "n_epochs",
                10
            ),

        "gamma":
            rl_cfg.get(
                "gamma",
                0.99
            ),

        "gae_lambda":
            rl_cfg.get(
                "gae_lambda",
                0.95
            ),

        "clip_range":
            rl_cfg.get(
                "clip_range",
                0.2
            ),

        "ent_coef":
            rl_cfg.get(
                "ent_coef",
                0.0
            ),

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    }

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = (
        model_dir
        /
        f"{name}_metadata.json"
    )

    with open(

        metadata_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metadata,

            f,

            indent=4,

            ensure_ascii=False

        )

    # ========================================================
    # CLOSE ENVIRONMENTS
    # ========================================================

    train_env.close()

    if eval_env is not None:

        try:

            eval_env.close()

        except Exception:

            pass

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\n============================================================"
    )

    print(
        "TRAINING SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Mode            : {training_mode}"
    )

    print(
        f"Timesteps       : {total_timesteps}"
    )

    print(
        f"n_steps         : {n_steps}"
    )

    print(
        f"Batch size      : {batch_size}"
    )

    print(
        f"Checkpoint freq : {checkpoint_freq}"
    )

    print(
        f"Training time   : "
        f"{format_seconds(training_time)}"
    )

    print(
        f"Model           : "
        f"{model_path.resolve()}.zip"
    )

    print(
        f"Checkpoints     : "
        f"{checkpoint_dir.resolve()}"
    )

    print(
        f"Metadata        : "
        f"{metadata_path.resolve()}"
    )

    print(
        "============================================================"
    )

    return model
