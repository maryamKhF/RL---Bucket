# RL/test_ppo_agent.py

"""
Comprehensive validation tests for RL/ppo_agent.py.

Project decision:
    - PPO learns only eta.
    - Top-K parameter k is fixed at 5.
    - k is NOT part of the PPO action space.

The tests validate:
    1. PPO construction
    2. Environment validation
    3. Monitor wrapping
    4. Hyperparameter validation
    5. Action-space semantics
    6. Fixed-k project constraint
    7. Deterministic initialization
    8. Prediction
    9. Short learning execution
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

from RL.ppo_agent import build_ppo


# ============================================================================
# PROJECT CONSTANTS
# ============================================================================

FIXED_K = 5


# ============================================================================
# TEST ENVIRONMENT
# ============================================================================

class DummyRoutingEnv(gym.Env):
    """
    Minimal routing environment used only for validating PPO construction.

    Observation:
        4-dimensional continuous vector.

    Action:
        One-dimensional continuous action:
            [eta]

        eta is constrained to [0, 1].

    Important:
        k is intentionally absent from the action space because k is fixed
        at 5 in the project.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0
        self._max_steps = 10

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        observation = np.zeros(
            self.observation_space.shape,
            dtype=np.float32,
        )

        info = {}

        return observation, info

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (1,):
            raise ValueError(
                f"Expected action shape (1,), got {action.shape}"
            )

        eta = float(action[0])

        if not 0.0 <= eta <= 1.0:
            raise ValueError(
                f"eta must be inside [0, 1], got {eta}"
            )

        self._step_count += 1

        observation = np.zeros(
            self.observation_space.shape,
            dtype=np.float32,
        )

        reward = float(
            1.0 - abs(eta - 0.5)
        )

        terminated = False
        truncated = (
            self._step_count >= self._max_steps
        )

        info = {
            "eta": eta,
            "k": FIXED_K,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


# ============================================================================
# VALID CONFIGURATION
# ============================================================================

def valid_config():
    """
    Return a valid PPO configuration.

    The values are intentionally small enough for the test suite to run
    quickly while still exercising the PPO construction path.
    """

    return {
        "rl": {
            "hidden_layers": [32, 32],

            "learning_rate": 3e-4,

            "n_steps": 64,

            "batch_size": 32,

            "n_epochs": 2,

            "gamma": 0.99,

            "gae_lambda": 0.95,

            "clip_range": 0.2,

            "ent_coef": 0.0,

            "verbose": 0,

            "tensorboard_log": None,

            "device": "cpu",
        }
    }


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """
    Simple deterministic test runner.
    """

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0

    def run(
        self,
        number,
        description,
        test_function,
    ):
        self.total += 1

        print(
            f"[{number:02d}] "
            f"{description:<66}",
            end="",
        )

        try:
            test_function()

            self.passed += 1

            print("PASS")

        except Exception as exc:
            self.failed += 1

            print("FAIL")

            print(
                f"      {type(exc).__name__}: {exc}"
            )

    def summary(self):
        print()
        print("=" * 78)

        print(
            f"Total : {self.total}"
        )

        print(
            f"Passed: {self.passed}"
        )

        print(
            f"Failed: {self.failed}"
        )

        print("=" * 78)

        if self.failed == 0:
            print(
                "PPO AGENT VALIDATION: PASS"
            )
        else:
            print(
                "PPO AGENT VALIDATION: FAIL"
            )


# ============================================================================
# TEST 01
# ============================================================================

def test_valid_construction():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert model is not None


# ============================================================================
# TEST 02
# ============================================================================

def test_model_has_policy():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert hasattr(
        model,
        "policy",
    )

    assert model.policy is not None


# ============================================================================
# TEST 03
# ============================================================================

def test_policy_class():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert isinstance(
        model.policy,
        ActorCriticPolicy,
    )


# ============================================================================
# TEST 04
# ============================================================================

def test_observation_space_matches_environment():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert (
        model.observation_space
        == env.observation_space
    )


# ============================================================================
# TEST 05
# ============================================================================

def test_action_space_matches_environment():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert (
        model.action_space
        == env.action_space
    )


# ============================================================================
# TEST 06
# ============================================================================

def test_environment_is_monitored():
    """
    Stable-Baselines3 vectorizes environments internally.

    Therefore the Monitor wrapper is not expected to be returned directly
    by model.get_env(). Instead:

        model.get_env()
            |
            +-- DummyVecEnv
                    |
                    +-- envs[0]
                            |
                            +-- Monitor
                                    |
                                    +-- DummyRoutingEnv
    """

    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    vectorized_env = model.get_env()

    assert hasattr(
        vectorized_env,
        "envs",
    )

    assert len(
        vectorized_env.envs
    ) == 1

    wrapped_environment = (
        vectorized_env.envs[0]
    )

    assert isinstance(
        wrapped_environment,
        Monitor,
    )


# ============================================================================
# TEST 07
# ============================================================================

def test_already_monitored_environment_not_double_wrapped():
    """
    If the input environment is already a Monitor, build_ppo() must not
    create a second nested Monitor wrapper.

    Expected structure:

        DummyVecEnv
            |
            +-- Monitor
                    |
                    +-- DummyRoutingEnv

    Not:

        DummyVecEnv
            |
            +-- Monitor
                    |
                    +-- Monitor
                            |
                            +-- DummyRoutingEnv
    """

    env = DummyRoutingEnv()

    monitored_env = Monitor(
        env
    )

    cfg = valid_config()

    model = build_ppo(
        monitored_env,
        cfg,
        seed=42,
    )

    vectorized_env = model.get_env()

    assert hasattr(
        vectorized_env,
        "envs",
    )

    assert len(
        vectorized_env.envs
    ) == 1

    wrapped_environment = (
        vectorized_env.envs[0]
    )

    assert isinstance(
        wrapped_environment,
        Monitor,
    )

    assert not isinstance(
        wrapped_environment.env,
        Monitor,
    )

    assert isinstance(
        wrapped_environment.env,
        DummyRoutingEnv,
    )


# ============================================================================
# TEST 08
# ============================================================================

def test_cfg_none_rejected():
    env = DummyRoutingEnv()

    try:
        build_ppo(
            env,
            None,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "cfg=None was accepted"
    )


# ============================================================================
# TEST 09
# ============================================================================

def test_cfg_non_dict_rejected():
    env = DummyRoutingEnv()

    invalid_configs = [
        [],
        (),
        "config",
        123,
    ]

    for cfg in invalid_configs:
        try:
            build_ppo(
                env,
                cfg,
                seed=42,
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        raise AssertionError(
            f"Non-dict cfg was accepted: {cfg!r}"
        )


# ============================================================================
# TEST 10
# ============================================================================

def test_missing_rl_rejected():
    env = DummyRoutingEnv()

    cfg = {}

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Missing rl section was accepted"
    )


# ============================================================================
# TEST 11
# ============================================================================

def test_rl_non_dict_rejected():
    env = DummyRoutingEnv()

    cfg = {
        "rl": [],
    }

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Non-dict rl section was accepted"
    )


# ============================================================================
# TEST 12
# ============================================================================

def test_env_none_rejected():
    cfg = valid_config()

    try:
        build_ppo(
            None,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "env=None was accepted"
    )


# ============================================================================
# TEST 13
# ============================================================================

def test_missing_observation_space_rejected():
    cfg = valid_config()

    class InvalidEnv:
        action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    try:
        build_ppo(
            InvalidEnv(),
            cfg,
            seed=42,
        )

    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Environment without observation_space was accepted"
    )


# ============================================================================
# TEST 14
# ============================================================================

def test_missing_action_space_rejected():
    cfg = valid_config()

    class InvalidEnv:
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    try:
        build_ppo(
            InvalidEnv(),
            cfg,
            seed=42,
        )

    except (
        AttributeError,
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Environment without action_space was accepted"
    )


# ============================================================================
# TEST 15
# ============================================================================

def test_none_observation_space_rejected():
    cfg = valid_config()

    class InvalidEnv:
        observation_space = None

        action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    try:
        build_ppo(
            InvalidEnv(),
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "None observation_space was accepted"
    )


# ============================================================================
# TEST 16
# ============================================================================

def test_none_action_space_rejected():
    cfg = valid_config()

    class InvalidEnv:
        observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        action_space = None

    try:
        build_ppo(
            InvalidEnv(),
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "None action_space was accepted"
    )


# ============================================================================
# TEST 17
# ============================================================================

def test_hidden_layers_type_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = (
        "128,128"
    )

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid hidden_layers type was accepted"
    )


# ============================================================================
# TEST 18
# ============================================================================

def test_empty_hidden_layers_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = []

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Empty hidden_layers was accepted"
    )


# ============================================================================
# TEST 19
# ============================================================================

def test_bool_hidden_layer_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = [
        True
    ]

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Boolean hidden layer was accepted"
    )


# ============================================================================
# TEST 20
# ============================================================================

def test_float_hidden_layer_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = [
        32.5
    ]

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Float hidden layer was accepted"
    )


# ============================================================================
# TEST 21
# ============================================================================

def test_zero_hidden_layer_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = [
        0
    ]

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Zero hidden layer was accepted"
    )


# ============================================================================
# TEST 22
# ============================================================================

def test_negative_hidden_layer_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["hidden_layers"] = [
        -32
    ]

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Negative hidden layer was accepted"
    )


# ============================================================================
# TEST 23
# ============================================================================

def test_learning_rate_type_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["learning_rate"] = (
        "0.0003"
    )

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid learning rate type was accepted"
    )


# ============================================================================
# TEST 24
# ============================================================================

def test_zero_learning_rate_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["learning_rate"] = 0.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Zero learning rate was accepted"
    )


# ============================================================================
# TEST 25
# ============================================================================

def test_negative_learning_rate_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["learning_rate"] = (
        -0.001
    )

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Negative learning rate was accepted"
    )


# ============================================================================
# TEST 26
# ============================================================================

def test_bool_n_steps_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_steps"] = True

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Boolean n_steps was accepted"
    )


# ============================================================================
# TEST 27
# ============================================================================

def test_float_n_steps_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_steps"] = 64.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Float n_steps was accepted"
    )


# ============================================================================
# TEST 28
# ============================================================================

def test_zero_n_steps_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_steps"] = 0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Zero n_steps was accepted"
    )


# ============================================================================
# TEST 29
# ============================================================================

def test_batch_larger_than_n_steps_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_steps"] = 32
    cfg["rl"]["batch_size"] = 64

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "batch_size > n_steps was accepted"
    )


# ============================================================================
# TEST 30
# ============================================================================

def test_nondivisible_batch_size_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_steps"] = 64
    cfg["rl"]["batch_size"] = 30

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Non-divisible batch_size was accepted"
    )


# ============================================================================
# TEST 31
# ============================================================================

def test_n_epochs_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["n_epochs"] = 0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid n_epochs was accepted"
    )


# ============================================================================
# TEST 32
# ============================================================================

def test_gamma_zero_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["gamma"] = 0.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "gamma=0 was accepted"
    )


# ============================================================================
# TEST 33
# ============================================================================

def test_gamma_greater_than_one_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["gamma"] = 1.1

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "gamma>1 was accepted"
    )


# ============================================================================
# TEST 34
# ============================================================================

def test_gae_lambda_zero_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["gae_lambda"] = 0.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "gae_lambda=0 was accepted"
    )


# ============================================================================
# TEST 35
# ============================================================================

def test_gae_lambda_greater_than_one_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["gae_lambda"] = 1.1

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "gae_lambda>1 was accepted"
    )


# ============================================================================
# TEST 36
# ============================================================================

def test_clip_range_zero_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["clip_range"] = 0.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "clip_range=0 was accepted"
    )


# ============================================================================
# TEST 37
# ============================================================================

def test_clip_range_one_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["clip_range"] = 1.0

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "clip_range=1 was accepted"
    )


# ============================================================================
# TEST 38
# ============================================================================

def test_negative_ent_coef_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["ent_coef"] = -0.1

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Negative ent_coef was accepted"
    )


# ============================================================================
# TEST 39
# ============================================================================

def test_negative_seed_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    try:
        build_ppo(
            env,
            cfg,
            seed=-1,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Negative seed was accepted"
    )


# ============================================================================
# TEST 40
# ============================================================================

def test_bool_seed_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    try:
        build_ppo(
            env,
            cfg,
            seed=True,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Boolean seed was accepted"
    )


# ============================================================================
# TEST 41
# ============================================================================

def test_float_seed_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    try:
        build_ppo(
            env,
            cfg,
            seed=42.5,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Float seed was accepted"
    )


# ============================================================================
# TEST 42
# ============================================================================

def test_none_seed_accepted():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=None,
    )

    assert model is not None


# ============================================================================
# TEST 43
# ============================================================================

def test_invalid_verbose_type_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["verbose"] = "0"

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid verbose type was accepted"
    )


# ============================================================================
# TEST 44
# ============================================================================

def test_invalid_verbose_value_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["verbose"] = 3

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid verbose value was accepted"
    )


# ============================================================================
# TEST 45
# ============================================================================

def test_invalid_tensorboard_type_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["tensorboard_log"] = 123

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid tensorboard_log type was accepted"
    )


# ============================================================================
# TEST 46
# ============================================================================

def test_empty_tensorboard_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["tensorboard_log"] = ""

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Empty tensorboard_log was accepted"
    )


# ============================================================================
# TEST 47
# ============================================================================

def test_invalid_device_type_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["device"] = 123

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Invalid device type was accepted"
    )


# ============================================================================
# TEST 48
# ============================================================================

def test_empty_device_rejected():
    env = DummyRoutingEnv()
    cfg = valid_config()

    cfg["rl"]["device"] = ""

    try:
        build_ppo(
            env,
            cfg,
            seed=42,
        )

    except (
        TypeError,
        ValueError,
    ):
        return

    raise AssertionError(
        "Empty device was accepted"
    )


# ============================================================================
# TEST 49
# ============================================================================

def test_action_space_is_one_dimensional():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert len(
        model.action_space.shape
    ) == 1

    assert (
        model.action_space.shape
        == (1,)
    )


# ============================================================================
# TEST 50
# ============================================================================

def test_k_is_not_action_dimension():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    assert (
        model.action_space.shape
        != (2,)
    )

    assert (
        model.action_space.shape
        == (1,)
    )


# ============================================================================
# TEST 51
# ============================================================================

def test_fixed_k_is_five():
    """
    Project-level invariant:

        k = 5

    PPO does not learn k.
    """

    assert FIXED_K == 5


# ============================================================================
# TEST 52
# ============================================================================

def test_model_can_predict():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    observation, _ = env.reset(
        seed=42
    )

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    assert action is not None


# ============================================================================
# TEST 53
# ============================================================================

def test_prediction_action_shape():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    observation, _ = env.reset(
        seed=42
    )

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    action = np.asarray(
        action
    )

    assert (
        action.shape
        == (1,)
    )


# ============================================================================
# TEST 54
# ============================================================================

def test_predicted_eta_inside_action_space():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    observation, _ = env.reset(
        seed=42
    )

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32,
    )

    assert np.all(
        action
        >= env.action_space.low
    )

    assert np.all(
        action
        <= env.action_space.high
    )


# ============================================================================
# TEST 55
# ============================================================================

def test_same_seed_same_initial_prediction():
    """
    Same seed must produce reproducible PPO initialization and therefore
    reproducible deterministic prediction for the same observation.
    """

    env1 = DummyRoutingEnv()
    env2 = DummyRoutingEnv()

    cfg1 = valid_config()
    cfg2 = valid_config()

    model1 = build_ppo(
        env1,
        cfg1,
        seed=42,
    )

    model2 = build_ppo(
        env2,
        cfg2,
        seed=42,
    )

    obs1, _ = env1.reset(
        seed=123
    )

    obs2, _ = env2.reset(
        seed=123
    )

    action1, _ = model1.predict(
        obs1,
        deterministic=True,
    )

    action2, _ = model2.predict(
        obs2,
        deterministic=True,
    )

    np.testing.assert_allclose(
        action1,
        action2,
    )


# ============================================================================
# TEST 56
# ============================================================================

def test_different_seeds_different_initialization():
    """
    Different seeds must initialize PPO with different policy parameters.

    Important:

        Different parameter initialization does not necessarily imply that
        one deterministic prediction for one particular observation must be
        different.

    Therefore this test compares the actual policy parameters rather than
    comparing a single action output.
    """

    env1 = DummyRoutingEnv()
    env2 = DummyRoutingEnv()

    cfg1 = valid_config()
    cfg2 = valid_config()

    model1 = build_ppo(
        env1,
        cfg1,
        seed=42,
    )

    model2 = build_ppo(
        env2,
        cfg2,
        seed=43,
    )

    parameters1 = [
        parameter.detach()
        .cpu()
        .numpy()
        .copy()
        for parameter in model1.policy.parameters()
    ]

    parameters2 = [
        parameter.detach()
        .cpu()
        .numpy()
        .copy()
        for parameter in model2.policy.parameters()
    ]

    assert (
        len(parameters1)
        == len(parameters2)
    )

    any_parameter_difference = False

    for parameter1, parameter2 in zip(
        parameters1,
        parameters2,
    ):
        if (
            parameter1.shape
            != parameter2.shape
        ):
            raise AssertionError(
                "Policy parameter shapes differ "
                "between seeded models"
            )

        if not np.array_equal(
            parameter1,
            parameter2,
        ):
            any_parameter_difference = True
            break

    assert any_parameter_difference, (
        "Different seeds produced identical policy parameters"
    )


# ============================================================================
# TEST 57
# ============================================================================

def test_short_learning_run():
    env = DummyRoutingEnv()
    cfg = valid_config()

    model = build_ppo(
        env,
        cfg,
        seed=42,
    )

    model.learn(
        total_timesteps=128,
        progress_bar=False,
    )

    assert model is not None


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    print()
    print("=" * 78)
    print(
        " PPO AGENT COMPREHENSIVE VALIDATION"
    )
    print("=" * 78)

    print()

    print(
        "PPO routing parameter:"
    )

    print(
        "  eta = learned by PPO"
    )

    print(
        "Top-K:"
    )

    print(
        "  k = 5 (fixed)"
    )

    print()

    runner = TestRunner()

    runner.run(
        1,
        "valid PPO construction",
        test_valid_construction,
    )

    runner.run(
        2,
        "model has policy",
        test_model_has_policy,
    )

    runner.run(
        3,
        "policy is MLP ActorCriticPolicy",
        test_policy_class,
    )

    runner.run(
        4,
        "observation space matches environment",
        test_observation_space_matches_environment,
    )

    runner.run(
        5,
        "action space matches environment",
        test_action_space_matches_environment,
    )

    runner.run(
        6,
        "environment is wrapped with Monitor",
        test_environment_is_monitored,
    )

    runner.run(
        7,
        "already monitored environment is not double wrapped",
        test_already_monitored_environment_not_double_wrapped,
    )

    runner.run(
        8,
        "cfg=None rejected",
        test_cfg_none_rejected,
    )

    runner.run(
        9,
        "non-dict cfg rejected",
        test_cfg_non_dict_rejected,
    )

    runner.run(
        10,
        "missing rl rejected",
        test_missing_rl_rejected,
    )

    runner.run(
        11,
        "non-dict rl rejected",
        test_rl_non_dict_rejected,
    )

    runner.run(
        12,
        "env=None rejected",
        test_env_none_rejected,
    )

    runner.run(
        13,
        "missing observation_space rejected",
        test_missing_observation_space_rejected,
    )

    runner.run(
        14,
        "missing action_space rejected",
        test_missing_action_space_rejected,
    )

    runner.run(
        15,
        "None observation_space rejected",
        test_none_observation_space_rejected,
    )

    runner.run(
        16,
        "None action_space rejected",
        test_none_action_space_rejected,
    )

    runner.run(
        17,
        "invalid hidden_layers type rejected",
        test_hidden_layers_type_rejected,
    )

    runner.run(
        18,
        "empty hidden_layers rejected",
        test_empty_hidden_layers_rejected,
    )

    runner.run(
        19,
        "boolean hidden layer rejected",
        test_bool_hidden_layer_rejected,
    )

    runner.run(
        20,
        "float hidden layer rejected",
        test_float_hidden_layer_rejected,
    )

    runner.run(
        21,
        "zero hidden layer rejected",
        test_zero_hidden_layer_rejected,
    )

    runner.run(
        22,
        "negative hidden layer rejected",
        test_negative_hidden_layer_rejected,
    )

    runner.run(
        23,
        "invalid learning rate type rejected",
        test_learning_rate_type_rejected,
    )

    runner.run(
        24,
        "zero learning rate rejected",
        test_zero_learning_rate_rejected,
    )

    runner.run(
        25,
        "negative learning rate rejected",
        test_negative_learning_rate_rejected,
    )

    runner.run(
        26,
        "boolean n_steps rejected",
        test_bool_n_steps_rejected,
    )

    runner.run(
        27,
        "float n_steps rejected",
        test_float_n_steps_rejected,
    )

    runner.run(
        28,
        "zero n_steps rejected",
        test_zero_n_steps_rejected,
    )

    runner.run(
        29,
        "batch_size > n_steps rejected",
        test_batch_larger_than_n_steps_rejected,
    )

    runner.run(
        30,
        "non-divisible batch_size rejected",
        test_nondivisible_batch_size_rejected,
    )

    runner.run(
        31,
        "invalid n_epochs rejected",
        test_n_epochs_rejected,
    )

    runner.run(
        32,
        "gamma=0 rejected",
        test_gamma_zero_rejected,
    )

    runner.run(
        33,
        "gamma>1 rejected",
        test_gamma_greater_than_one_rejected,
    )

    runner.run(
        34,
        "gae_lambda=0 rejected",
        test_gae_lambda_zero_rejected,
    )

    runner.run(
        35,
        "gae_lambda>1 rejected",
        test_gae_lambda_greater_than_one_rejected,
    )

    runner.run(
        36,
        "clip_range=0 rejected",
        test_clip_range_zero_rejected,
    )

    runner.run(
        37,
        "clip_range=1 rejected",
        test_clip_range_one_rejected,
    )

    runner.run(
        38,
        "negative ent_coef rejected",
        test_negative_ent_coef_rejected,
    )

    runner.run(
        39,
        "negative seed rejected",
        test_negative_seed_rejected,
    )

    runner.run(
        40,
        "boolean seed rejected",
        test_bool_seed_rejected,
    )

    runner.run(
        41,
        "float seed rejected",
        test_float_seed_rejected,
    )

    runner.run(
        42,
        "None seed accepted",
        test_none_seed_accepted,
    )

    runner.run(
        43,
        "invalid verbose type rejected",
        test_invalid_verbose_type_rejected,
    )

    runner.run(
        44,
        "invalid verbose value rejected",
        test_invalid_verbose_value_rejected,
    )

    runner.run(
        45,
        "invalid tensorboard_log type rejected",
        test_invalid_tensorboard_type_rejected,
    )

    runner.run(
        46,
        "empty tensorboard_log rejected",
        test_empty_tensorboard_rejected,
    )

    runner.run(
        47,
        "invalid device type rejected",
        test_invalid_device_type_rejected,
    )

    runner.run(
        48,
        "empty device rejected",
        test_empty_device_rejected,
    )

    runner.run(
        49,
        "action space is one-dimensional",
        test_action_space_is_one_dimensional,
    )

    runner.run(
        50,
        "k is not action dimension",
        test_k_is_not_action_dimension,
    )

    runner.run(
        51,
        "fixed k is 5",
        test_fixed_k_is_five,
    )

    runner.run(
        52,
        "model can predict",
        test_model_can_predict,
    )

    runner.run(
        53,
        "prediction action shape",
        test_prediction_action_shape,
    )

    runner.run(
        54,
        "predicted eta is inside action space",
        test_predicted_eta_inside_action_space,
    )

    runner.run(
        55,
        "same seed produces same initial prediction",
        test_same_seed_same_initial_prediction,
    )

    runner.run(
        56,
        "different seeds produce different initialization",
        test_different_seeds_different_initialization,
    )

    runner.run(
        57,
        "short PPO learning run succeeds",
        test_short_learning_run,
    )

    runner.summary()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()