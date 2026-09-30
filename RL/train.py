# RL/train.py

from pathlib import Path
import json
import time

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

from .ppo_agent import build_ppo
from .environment import RoutingEnv


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

    Training pipeline:

        Graph + Transactions
                ↓
          RoutingEnv
                ↓
             State
                ↓
              PPO
                ↓
              eta
                ↓
        Routing / Simulation
                ↓
             Reward
                ↓
          PPO parameter update

    PPO learns the routing heuristic parameter eta
    from the observed network state.
    """

    # =====================================================
    # CONFIGURATION
    # =====================================================

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

    # =====================================================
    # DIRECTORIES
    # =====================================================

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
            "checkpoints"
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

    # =====================================================
    # TRAINING ENVIRONMENT
    # =====================================================

    train_env = RoutingEnv(
        G=G,
        transactions=transactions,
        heuristic_fn=heuristic,
        config=cfg,
        mode="train"
    )

    # =====================================================
    # ENVIRONMENT VALIDATION
    # =====================================================

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

    # =====================================================
    # PPO MODEL
    # =====================================================

    model = build_ppo(
        env=train_env,
        cfg=cfg,
        seed=seed
    )

    # =====================================================
    # CALLBACKS
    # =====================================================

    callbacks = []

    # =====================================================
    # CHECKPOINT CALLBACK
    # =====================================================

    checkpoint_freq = int(
        rl_cfg.get(
            "checkpoint_freq",
            10000
        )
    )

    if checkpoint_freq <= 0:
        raise ValueError(
            "'checkpoint_freq' must be greater than zero."
        )

    checkpoint_callback = CheckpointCallback(

        save_freq=checkpoint_freq,

        save_path=str(
            checkpoint_dir
        ),

        name_prefix=name

    )

    callbacks.append(
        checkpoint_callback
    )

    # =====================================================
    # EVALUATION CALLBACK
    # =====================================================

    if eval_env is not None:

        # -------------------------------------------------
        # Ensure evaluation environment is monitored
        # -------------------------------------------------

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

    # =====================================================
    # TRAINING
    # =====================================================

    total_timesteps = int(
        rl_cfg.get(
            "total_timesteps",
            10000
        )
    )

    if total_timesteps <= 0:
        train_env.close()

        raise ValueError(
            "'total_timesteps' must be greater than zero."
        )

    start_time = time.time()

    try:

        model.learn(

            total_timesteps=total_timesteps,

            callback=callbacks,

            reset_num_timesteps=True

        )

    except Exception:

        train_env.close()

        if eval_env is not None:
            eval_env.close()

        raise

    training_time = (
        time.time()
        -
        start_time
    )

    # =====================================================
    # SAVE FINAL MODEL
    # =====================================================

    model_path = (
        model_dir / name
    )

    model.save(
        str(model_path)
    )

    # =====================================================
    # METADATA
    # =====================================================

    observation_dimension = int(
        train_env.observation_space.shape[0]
    )

    action_dimension = int(
        train_env.action_space.shape[0]
    )

    metadata = {

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        "model":
            name,

        "seed":
            seed,

        "timesteps":
            total_timesteps,

        "training_time_seconds":
            float(training_time),

        # -------------------------------------------------
        # Observation / Action
        # -------------------------------------------------

        "observation_shape":
            list(observation_shape),

        "action_shape":
            list(action_shape),

        "observation_dimension":
            observation_dimension,

        "action_dimension":
            action_dimension,

        "action_low":
            train_env.action_space.low.tolist(),

        "action_high":
            train_env.action_space.high.tolist(),

        # -------------------------------------------------
        # Graph / State configuration
        # -------------------------------------------------

        "neighborhood_k":
            graph_cfg.get(
                "neighborhood_k"
            ),

        "neighborhood_m":
            graph_cfg.get(
                "neighborhood_m"
            ),

        "top_k":
            graph_cfg.get(
                "top_k"
            ),

        "max_hops":
            graph_cfg.get(
                "max_hops"
            ),

        # -------------------------------------------------
        # PPO configuration
        # -------------------------------------------------

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

        "n_steps":
            rl_cfg.get(
                "n_steps",
                2048
            ),

        "batch_size":
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

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    # =====================================================
    # SAVE METADATA
    # =====================================================

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

    # =====================================================
    # CLOSE ENVIRONMENTS
    # =====================================================

    train_env.close()

    if eval_env is not None:
        eval_env.close()

    return model