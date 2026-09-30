# RL/ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def build_ppo(
    env,
    cfg,
    seed=42
):
    """
    Build and configure the PPO routing agent.

    The environment is responsible for:
        - State construction
        - Candidate-path generation
        - Routing
        - Payment simulation
        - Failure handling
        - Reward calculation

    PPO is responsible for learning the routing heuristic
    parameter eta from the observed network state.
    """

    # =====================================================
    # RL CONFIGURATION
    # =====================================================

    if "rl" not in cfg:
        raise KeyError(
            "Missing 'rl' section in configuration."
        )

    rl_cfg = cfg["rl"]

    # =====================================================
    # ENVIRONMENT CHECK
    # =====================================================

    if env is None:
        raise ValueError(
            "The PPO environment cannot be None."
        )

    # -----------------------------------------------------
    # Observation / action space check
    # -----------------------------------------------------

    if not hasattr(env, "observation_space"):
        raise AttributeError(
            "Environment must define observation_space."
        )

    if not hasattr(env, "action_space"):
        raise AttributeError(
            "Environment must define action_space."
        )

    # =====================================================
    # MONITOR
    # =====================================================

    # Avoid wrapping an already monitored environment.
    if not isinstance(env, Monitor):
        env = Monitor(env)

    # =====================================================
    # POLICY NETWORK
    # =====================================================

    hidden_layers = rl_cfg.get(
        "hidden_layers",
        [128, 128]
    )

    if not isinstance(hidden_layers, (list, tuple)):
        raise TypeError(
            "'hidden_layers' must be a list or tuple."
        )

    if len(hidden_layers) == 0:
        raise ValueError(
            "'hidden_layers' cannot be empty."
        )

    hidden_layers = [
        int(layer)
        for layer in hidden_layers
    ]

    if any(layer <= 0 for layer in hidden_layers):
        raise ValueError(
            "All hidden-layer sizes must be positive."
        )

    policy_kwargs = {
        "net_arch": hidden_layers
    }

    # =====================================================
    # PPO HYPERPARAMETERS
    # =====================================================

    learning_rate = rl_cfg.get(
        "learning_rate",
        3e-4
    )

    n_steps = int(
        rl_cfg.get(
            "n_steps",
            2048
        )
    )

    batch_size = int(
        rl_cfg.get(
            "batch_size",
            64
        )
    )

    n_epochs = int(
        rl_cfg.get(
            "n_epochs",
            10
        )
    )

    gamma = float(
        rl_cfg.get(
            "gamma",
            0.99
        )
    )

    gae_lambda = float(
        rl_cfg.get(
            "gae_lambda",
            0.95
        )
    )

    clip_range = float(
        rl_cfg.get(
            "clip_range",
            0.2
        )
    )

    ent_coef = float(
        rl_cfg.get(
            "ent_coef",
            0.0
        )
    )

    # =====================================================
    # BASIC VALIDATION
    # =====================================================

    if n_steps <= 0:
        raise ValueError(
            "'n_steps' must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "'batch_size' must be greater than zero."
        )

    if n_steps < batch_size:
        raise ValueError(
            "'batch_size' cannot be larger than 'n_steps'."
        )

    if n_steps % batch_size != 0:
        raise ValueError(
            "'n_steps' must be divisible by 'batch_size' "
            "for the current PPO configuration."
        )

    if n_epochs <= 0:
        raise ValueError(
            "'n_epochs' must be greater than zero."
        )

    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            "'gamma' must be in the interval (0, 1]."
        )

    if not 0.0 < gae_lambda <= 1.0:
        raise ValueError(
            "'gae_lambda' must be in the interval (0, 1]."
        )

    if not 0.0 < clip_range < 1.0:
        raise ValueError(
            "'clip_range' must be in the interval (0, 1)."
        )

    if learning_rate <= 0:
        raise ValueError(
            "'learning_rate' must be greater than zero."
        )

    if ent_coef < 0:
        raise ValueError(
            "'ent_coef' cannot be negative."
        )

    # =====================================================
    # PPO MODEL
    # =====================================================

    model = PPO(

        policy="MlpPolicy",

        env=env,

        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------

        seed=seed,

        # -------------------------------------------------
        # Training output
        # -------------------------------------------------

        verbose=rl_cfg.get(
            "verbose",
            1
        ),

        # -------------------------------------------------
        # PPO hyperparameters
        # -------------------------------------------------

        learning_rate=learning_rate,

        n_steps=n_steps,

        batch_size=batch_size,

        n_epochs=n_epochs,

        gamma=gamma,

        gae_lambda=gae_lambda,

        clip_range=clip_range,

        ent_coef=ent_coef,

        # -------------------------------------------------
        # Policy network
        # -------------------------------------------------

        policy_kwargs=policy_kwargs,

        # -------------------------------------------------
        # TensorBoard
        # -------------------------------------------------

        tensorboard_log=rl_cfg.get(
            "tensorboard_log",
            "./logs/"
        ),

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        device=rl_cfg.get(
            "device",
            "auto"
        )
    )

    return model