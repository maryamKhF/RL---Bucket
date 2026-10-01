# RL/ppo_agent.py

from numbers import Real

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
        - Mapping PPO action to the routing parameter eta

    PPO is responsible for learning the routing heuristic
    parameter eta from the observed network state.

    Important routing constraint
    ----------------------------
    The number of candidate routes k is NOT controlled by PPO.

    k is fixed by the routing configuration and is assumed to be:

        k = 5

    Therefore the PPO action space is intended to control
    eta only.

    PPO flow
    --------

        Network State
             |
             v
        Observation
             |
             v
            PPO
             |
             v
          Action
             |
             v
            eta
             |
             v
        Modified Heuristic
             |
             v
          Top-K
          k = 5
             |
             v
          Bucket
    """

    # =====================================================
    # RL CONFIGURATION
    # =====================================================

    if not isinstance(cfg, dict):
        raise TypeError(
            "'cfg' must be a dictionary."
        )

    if "rl" not in cfg:
        raise KeyError(
            "Missing 'rl' section in configuration."
        )

    rl_cfg = cfg["rl"]

    if not isinstance(rl_cfg, dict):
        raise TypeError(
            "'rl' configuration must be a dictionary."
        )

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

    if env.observation_space is None:
        raise ValueError(
            "Environment observation_space cannot be None."
        )

    if env.action_space is None:
        raise ValueError(
            "Environment action_space cannot be None."
        )

    # =====================================================
    # FIXED TOP-K CONFIGURATION
    # =====================================================

    # k is intentionally fixed and is NOT an RL action.
    #
    # The routing architecture used by this project is:
    #
    #     PPO -> eta -> heuristic -> Top-K
    #
    # with:
    #
    #     k = 5
    #
    # Therefore PPO learns only eta.

    fixed_k = 5

    if fixed_k != 5:
        raise RuntimeError(
            "The routing configuration requires fixed k = 5."
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

    validated_hidden_layers = []

    for layer in hidden_layers:

        if isinstance(layer, bool):
            raise TypeError(
                "Hidden-layer sizes must be integers."
            )

        if not isinstance(layer, int):
            raise TypeError(
                "Hidden-layer sizes must be integers."
            )

        if layer <= 0:
            raise ValueError(
                "All hidden-layer sizes must be positive."
            )

        validated_hidden_layers.append(
            layer
        )

    hidden_layers = validated_hidden_layers

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

    n_steps = rl_cfg.get(
        "n_steps",
        2048
    )

    batch_size = rl_cfg.get(
        "batch_size",
        64
    )

    n_epochs = rl_cfg.get(
        "n_epochs",
        10
    )

    gamma = rl_cfg.get(
        "gamma",
        0.99
    )

    gae_lambda = rl_cfg.get(
        "gae_lambda",
        0.95
    )

    clip_range = rl_cfg.get(
        "clip_range",
        0.2
    )

    ent_coef = rl_cfg.get(
        "ent_coef",
        0.0
    )

    # =====================================================
    # HYPERPARAMETER TYPE VALIDATION
    # =====================================================

    # -----------------------------------------------------
    # learning_rate
    # -----------------------------------------------------

    if isinstance(learning_rate, bool):
        raise TypeError(
            "'learning_rate' must be a real numeric value."
        )

    if not isinstance(learning_rate, Real):
        raise TypeError(
            "'learning_rate' must be a real numeric value."
        )

    learning_rate = float(
        learning_rate
    )

    # -----------------------------------------------------
    # n_steps
    # -----------------------------------------------------

    if isinstance(n_steps, bool):
        raise TypeError(
            "'n_steps' must be an integer."
        )

    if not isinstance(n_steps, int):
        raise TypeError(
            "'n_steps' must be an integer."
        )

    # -----------------------------------------------------
    # batch_size
    # -----------------------------------------------------

    if isinstance(batch_size, bool):
        raise TypeError(
            "'batch_size' must be an integer."
        )

    if not isinstance(batch_size, int):
        raise TypeError(
            "'batch_size' must be an integer."
        )

    # -----------------------------------------------------
    # n_epochs
    # -----------------------------------------------------

    if isinstance(n_epochs, bool):
        raise TypeError(
            "'n_epochs' must be an integer."
        )

    if not isinstance(n_epochs, int):
        raise TypeError(
            "'n_epochs' must be an integer."
        )

    # -----------------------------------------------------
    # gamma
    # -----------------------------------------------------

    if isinstance(gamma, bool):
        raise TypeError(
            "'gamma' must be a real numeric value."
        )

    if not isinstance(gamma, Real):
        raise TypeError(
            "'gamma' must be a real numeric value."
        )

    gamma = float(
        gamma
    )

    # -----------------------------------------------------
    # gae_lambda
    # -----------------------------------------------------

    if isinstance(gae_lambda, bool):
        raise TypeError(
            "'gae_lambda' must be a real numeric value."
        )

    if not isinstance(gae_lambda, Real):
        raise TypeError(
            "'gae_lambda' must be a real numeric value."
        )

    gae_lambda = float(
        gae_lambda
    )

    # -----------------------------------------------------
    # clip_range
    # -----------------------------------------------------

    if isinstance(clip_range, bool):
        raise TypeError(
            "'clip_range' must be a real numeric value."
        )

    if not isinstance(clip_range, Real):
        raise TypeError(
            "'clip_range' must be a real numeric value."
        )

    clip_range = float(
        clip_range
    )

    # -----------------------------------------------------
    # ent_coef
    # -----------------------------------------------------

    if isinstance(ent_coef, bool):
        raise TypeError(
            "'ent_coef' must be a real numeric value."
        )

    if not isinstance(ent_coef, Real):
        raise TypeError(
            "'ent_coef' must be a real numeric value."
        )

    ent_coef = float(
        ent_coef
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

    # -----------------------------------------------------
    # Experimental configuration constraint
    # -----------------------------------------------------
    #
    # This is a project-level constraint for the current
    # PPO experiments. It is not a fundamental requirement
    # of PPO itself.
    #
    # Keeping n_steps divisible by batch_size provides a
    # clean and deterministic minibatch configuration for
    # the experiments.

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
    # SEED VALIDATION
    # =====================================================

    if isinstance(seed, bool):
        raise TypeError(
            "'seed' must be an integer or None."
        )

    if seed is not None:

        if not isinstance(seed, int):
            raise TypeError(
                "'seed' must be an integer or None."
            )

        if seed < 0:
            raise ValueError(
                "'seed' must be non-negative."
            )

    # =====================================================
    # VERBOSE VALIDATION
    # =====================================================

    verbose = rl_cfg.get(
        "verbose",
        1
    )

    if isinstance(verbose, bool):
        raise TypeError(
            "'verbose' must be an integer."
        )

    if not isinstance(verbose, int):
        raise TypeError(
            "'verbose' must be an integer."
        )

    if verbose not in (0, 1, 2):
        raise ValueError(
            "'verbose' must be 0, 1, or 2."
        )

    # =====================================================
    # TENSORBOARD CONFIGURATION
    # =====================================================

    tensorboard_log = rl_cfg.get(
        "tensorboard_log",
        "./logs/"
    )

    if tensorboard_log is not None:

        if not isinstance(
            tensorboard_log,
            str
        ):
            raise TypeError(
                "'tensorboard_log' must be a string or None."
            )

        if tensorboard_log == "":
            raise ValueError(
                "'tensorboard_log' cannot be an empty string."
            )

    # =====================================================
    # DEVICE CONFIGURATION
    # =====================================================

    device = rl_cfg.get(
        "device",
        "auto"
    )

    if not isinstance(device, str):
        raise TypeError(
            "'device' must be a string."
        )

    if device == "":
        raise ValueError(
            "'device' cannot be an empty string."
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

        verbose=verbose,

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

        tensorboard_log=tensorboard_log,

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        device=device
    )

    return model