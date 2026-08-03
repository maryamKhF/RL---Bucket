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


    # -------------------------------------------------
    # Directories
    # -------------------------------------------------

    model_dir = Path(
        cfg["rl"].get(
            "model_dir",
            "models"
        )
    )


    log_dir = Path(
        cfg["rl"].get(
            "log_dir",
            "logs"
        )
    )


    checkpoint_dir = Path(
        cfg["rl"].get(
            "checkpoint_dir",
            "checkpoints"
        )
    )


    model_dir.mkdir(
        exist_ok=True
    )


    log_dir.mkdir(
        exist_ok=True
    )


    checkpoint_dir.mkdir(
        exist_ok=True
    )



    # -------------------------------------------------
    # Environment
    # -------------------------------------------------

    env = RoutingEnv(

        G,

        transactions,

        heuristic,

        cfg

    )


    env = Monitor(
        env,
        str(log_dir / "train")
    )



    # -------------------------------------------------
    # PPO Model
    # -------------------------------------------------

    model = build_ppo(

        env,

        cfg,

        seed

    )



    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------

    callbacks = []



    # Save checkpoints

    checkpoint_callback = CheckpointCallback(

        save_freq=
            cfg["rl"].get(
                "checkpoint_freq",
                10000
            ),


        save_path=
            str(checkpoint_dir),


        name_prefix=name

    )


    callbacks.append(
        checkpoint_callback
    )



    # Evaluation callback

    if eval_env is not None:


        eval_env = Monitor(

            eval_env,

            str(log_dir / "eval")

        )


        eval_callback = EvalCallback(

            eval_env,


            best_model_save_path=
                str(model_dir / "best"),


            log_path=
                str(log_dir / "evaluation"),


            eval_freq=
                cfg["rl"].get(
                    "eval_freq",
                    5000
                ),


            deterministic=True

        )


        callbacks.append(
            eval_callback
        )



    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    start = time.time()


    model.learn(

        total_timesteps=
            cfg["rl"]["total_timesteps"],


        callback=
            callbacks

    )


    training_time = (
        time.time()
        -
        start
    )



    # -------------------------------------------------
    # Save Final Model
    # -------------------------------------------------

    model_path = (
        model_dir
        /
        name
    )


    model.save(
        str(model_path)
    )



    # -------------------------------------------------
    # Save Metadata
    # -------------------------------------------------

    metadata = {


        "model":
            name,


        "seed":
            seed,


        "timesteps":
            cfg["rl"]["total_timesteps"],


        "training_time":
            training_time,


        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    }



    with open(
        model_dir / f"{name}_metadata.json",
        "w"
    ) as f:


        json.dump(
            metadata,
            f,
            indent=4
        )



    return model