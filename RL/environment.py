# RL/environment.py

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pathlib import Path
import sys


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# PATHFINDING
# ==========================================================

from Pathfinding.top_k_paths import top_k_paths


# ==========================================================
# BUCKET
# ==========================================================

from Bucket.candidate_manager import CandidateManager


# ==========================================================
# SIMULATION
# ==========================================================

from Simulation.failure_model import FailureModel
from Simulation.network_dynamics import NetworkDynamics
from Simulation.payment_simulator import PaymentSimulator
from Simulation.backtrack import PartialBacktracker


# ==========================================================
# RL STATE
# ==========================================================

from RL.state import State


# ==========================================================
# RL REWARD
# ==========================================================

from RL.reward import calculate_reward


# ==========================================================
# ENVIRONMENT
# ==========================================================

class RoutingEnv(gym.Env):
    """
    PPO environment for adaptive Lightning routing.

    PPO controls ONLY the routing criterion eta.

    Fixed routing configuration:

        k = 5

    Main pipeline:

        Network State
             |
             v
           PPO
             |
             v
            eta
             |
             v
        Top-K Paths
          k = 5
             |
             v
       CandidateManager
             |
             v
           Bucket
             |
             v
      Payment Simulator
             |
          Failure?
          /      \
        No        Yes
        |          |
        v          v
     Success   Partial Backtrack
                   |
                   v
             Bucket Alternative
                   |
                Failed?
                /     \
              No       Yes
              |         |
              v         v
           Success   Full Reroute
                        |
                        v
                    Top-K again
                        |
                        v
                    Payment
             |
             v
          Reward
             |
             v
        PPO learning

    Important
    ---------
    k is fixed to 5.

    PPO learns eta only.

    Payment execution is performed only by PaymentSimulator.

    Failure evaluation is performed only by FailureModel.

    Partial recovery is performed only by PartialBacktracker.

    Full rerouting regenerates Top-K candidates with the same
    eta selected by PPO.
    """

    metadata = {
        "render_modes": [],
    }

    # ======================================================
    # FIXED ROUTING PARAMETER
    # ======================================================

    FIXED_TOP_K = 5

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(
        self,
        G,
        transactions,
        heuristic_fn=None,
        config=None,
        mode="hybrid",
        bucket=None,
        failure_model=None,
        network_dynamics=None,
    ):
        super().__init__()

        # --------------------------------------------------
        # Basic validation
        # --------------------------------------------------

        if G is None:
            raise ValueError(
                "RoutingEnv requires a valid graph."
            )

        if not hasattr(G, "nodes") or not hasattr(G, "edges"):
            raise TypeError(
                "G must be a NetworkX graph."
            )

        if transactions is None:
            raise ValueError(
                "RoutingEnv requires transactions."
            )

        if not isinstance(
            transactions,
            (list, tuple),
        ):
            raise TypeError(
                "transactions must be a list or tuple."
            )

        if len(transactions) == 0:
            raise ValueError(
                "RoutingEnv received an empty transaction set."
            )

        if heuristic_fn is not None and not callable(
            heuristic_fn
        ):
            raise TypeError(
                "heuristic_fn must be callable or None."
            )

        if config is not None and not isinstance(
            config,
            dict,
        ):
            raise TypeError(
                "config must be a dictionary or None."
            )

        if not isinstance(mode, str):
            raise TypeError(
                "mode must be a string."
            )

        # --------------------------------------------------
        # Store inputs
        # --------------------------------------------------

        self.G = G
        self.transactions = list(transactions)
        self.heuristic_fn = heuristic_fn
        self.mode = mode

        self.cfg = (
            dict(config)
            if config is not None
            else {}
        )

        # --------------------------------------------------
        # Graph configuration
        # --------------------------------------------------

        graph_cfg = self.cfg.get(
            "graph",
            {},
        )

        if not isinstance(
            graph_cfg,
            dict,
        ):
            raise TypeError(
                "config['graph'] must be a dictionary."
            )

        # IMPORTANT:
        # k is ALWAYS fixed to 5.
        #
        # It is deliberately not read from configuration.
        self.top_k = self.FIXED_TOP_K

        self.max_hops = self._validate_positive_int(
            graph_cfg.get(
                "max_hops",
                12,
            ),
            "graph.max_hops",
        )

        self.lambda_h = self._validate_nonnegative_float(
            graph_cfg.get(
                "lambda_h",
                1.0,
            ),
            "graph.lambda_h",
        )

        # --------------------------------------------------
        # RL configuration
        # --------------------------------------------------

        rl_cfg = self.cfg.get(
            "rl",
            {},
        )

        if not isinstance(
            rl_cfg,
            dict,
        ):
            raise TypeError(
                "config['rl'] must be a dictionary."
            )

        self.eta_min = self._validate_finite_float(
            rl_cfg.get(
                "eta_min",
                0.0,
            ),
            "rl.eta_min",
        )

        self.eta_max = self._validate_finite_float(
            rl_cfg.get(
                "eta_max",
                1.0,
            ),
            "rl.eta_max",
        )

        if self.eta_min < 0.0:
            raise ValueError(
                "rl.eta_min must be >= 0."
            )

        if self.eta_max > 1.0:
            raise ValueError(
                "rl.eta_max must be <= 1."
            )

        if self.eta_min > self.eta_max:
            raise ValueError(
                "eta_min cannot be greater than eta_max."
            )

        # --------------------------------------------------
        # State configuration
        # --------------------------------------------------

        self.state_k = self._validate_positive_int(
            graph_cfg.get(
                "neighborhood_k",
                15,
            ),
            "graph.neighborhood_k",
        )

        self.state_radius = self._validate_nonnegative_int(
            graph_cfg.get(
                "neighborhood_m",
                5,
            ),
            "graph.neighborhood_m",
        )

        # --------------------------------------------------
        # Simulation configuration
        # --------------------------------------------------

        simulation_cfg = self.cfg.get(
            "simulation",
            {},
        )

        if not isinstance(
            simulation_cfg,
            dict,
        ):
            raise TypeError(
                "config['simulation'] must be a dictionary."
            )

        self.seed = self._validate_seed(
            simulation_cfg.get(
                "seed",
                42,
            )
        )

        self.node_failure_probability = (
            self._validate_probability(
                simulation_cfg.get(
                    "node_failure_probability",
                    0.01,
                ),
                "simulation.node_failure_probability",
            )
        )

        self.liquidity_failure_probability = (
            self._validate_probability(
                simulation_cfg.get(
                    "liquidity_failure_probability",
                    0.05,
                ),
                "simulation.liquidity_failure_probability",
            )
        )

        self.channel_failure_rate = (
            self._validate_probability(
                simulation_cfg.get(
                    "channel_failure_rate",
                    0.01,
                ),
                "simulation.channel_failure_rate",
            )
        )

        self.node_failure_rate = (
            self._validate_probability(
                simulation_cfg.get(
                    "node_failure_rate",
                    0.005,
                ),
                "simulation.node_failure_rate",
            )
        )

        self.recovery_rate = (
            self._validate_probability(
                simulation_cfg.get(
                    "recovery_rate",
                    0.05,
                ),
                "simulation.recovery_rate",
            )
        )

        reset_balances = simulation_cfg.get(
            "reset_balances_on_episode_reset",
            False,
        )

        if not isinstance(
            reset_balances,
            bool,
        ):
            raise TypeError(
                "simulation.reset_balances_on_episode_reset "
                "must be bool."
            )

        self.reset_balances_on_episode_reset = (
            reset_balances
        )

        # --------------------------------------------------
        # Failure model
        # --------------------------------------------------

        if failure_model is None:

            self.failure_model = FailureModel(
                node_failure_probability=(
                    self.node_failure_probability
                ),
                liquidity_failure_probability=(
                    self.liquidity_failure_probability
                ),
                seed=self.seed,
            )

        else:

            self.failure_model = failure_model

        if self.failure_model is None:
            raise RuntimeError(
                "FailureModel initialization failed."
            )

        # --------------------------------------------------
        # Network dynamics
        # --------------------------------------------------

        if network_dynamics is None:

            self.network_dynamics = NetworkDynamics(
                self.G,
                channel_failure_rate=(
                    self.channel_failure_rate
                ),
                node_failure_rate=(
                    self.node_failure_rate
                ),
                recovery_rate=(
                    self.recovery_rate
                ),
                seed=self.seed,
            )

        else:

            self.network_dynamics = network_dynamics

        if self.network_dynamics is None:
            raise RuntimeError(
                "NetworkDynamics initialization failed."
            )

        # --------------------------------------------------
        # Payment simulator
        # --------------------------------------------------

        self.payment_simulator = PaymentSimulator(
            self.G,
            failure_model=self.failure_model,
            network_dynamics=self.network_dynamics,
        )

        if self.payment_simulator is None:
            raise RuntimeError(
                "PaymentSimulator initialization failed."
            )

        # --------------------------------------------------
        # Partial backtracker
        # --------------------------------------------------

        self.backtracker = PartialBacktracker(
            network=self.G,
        )

        if self.backtracker is None:
            raise RuntimeError(
                "PartialBacktracker initialization failed."
            )

        # --------------------------------------------------
        # External bucket
        # --------------------------------------------------

        self.bucket = bucket

        # --------------------------------------------------
        # Transaction state
        # --------------------------------------------------

        self.tx_index = 0

        self.current_tx = None

        self.current_paths = []

        self.current_bucket = None

        # --------------------------------------------------
        # Previous routing information
        # --------------------------------------------------

        self.last_failure_probability = 0.0

        self.last_average_delay = 0.0

        self.last_backtrack_count = 0

        # --------------------------------------------------
        # State builder
        # --------------------------------------------------

        self.state_builder = State(
            G=self.G,
            source=None,
            destination=None,
            transaction=None,
            candidate_paths=[],
            simulation_info={
                "failure_probability": 0.0,
                "average_delay": 0.0,
                "backtrack_count": 0,
            },
            bucket_info={},
            k=self.state_k,
            radius=self.state_radius,
        )

        # --------------------------------------------------
        # Observation space
        # --------------------------------------------------

        observation_dimension = int(
            self.state_builder.dimension
        )

        if observation_dimension <= 0:
            raise ValueError(
                "State observation dimension must be positive."
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                observation_dimension,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Action space
        #
        # PPO controls eta only.
        #
        # k is NOT an RL action.
        # --------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [self.eta_min],
                dtype=np.float32,
            ),
            high=np.array(
                [self.eta_max],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Runtime statistics
        # --------------------------------------------------

        self.episode_reward = 0.0

        self.episode_steps = 0

        self.total_successes = 0

        self.total_failures = 0

    # ======================================================
    # RESET
    # ======================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        Reset one PPO episode.

        Returns
        -------
        observation, info

        Important
        ---------
        - NetworkDynamics is reset.
        - FailureModel runtime state is reset.
        - A supplied seed resets FailureModel RNG.
        - k remains fixed at 5.
        """

        super().reset(
            seed=seed
        )

        # --------------------------------------------------
        # Seed handling
        # --------------------------------------------------

        if seed is not None:

            self.seed = self._validate_seed(
                seed
            )

        # --------------------------------------------------
        # Reset NetworkDynamics
        # --------------------------------------------------

        if self.network_dynamics is None:
            raise RuntimeError(
                "NetworkDynamics is not initialized."
            )

        reset_method = getattr(
            self.network_dynamics,
            "reset",
            None,
        )

        if not callable(reset_method):
            raise RuntimeError(
                "NetworkDynamics must provide reset()."
            )

        reset_method(
            reset_balances=(
                self.reset_balances_on_episode_reset
            )
        )

        # --------------------------------------------------
        # Reset FailureModel runtime state
        # --------------------------------------------------

        if self.failure_model is None:
            raise RuntimeError(
                "FailureModel is not initialized."
            )

        reset_failure_state = getattr(
            self.failure_model,
            "reset_runtime_state",
            None,
        )

        if not callable(reset_failure_state):

            raise RuntimeError(
                "The current FailureModel must provide "
                "reset_runtime_state()."
            )

        reset_failure_state(
            self.G,
            reset_counters=False,
            reset_rng=(
                seed is not None
            ),
        )

        # --------------------------------------------------
        # Reset transaction index
        # --------------------------------------------------

        self.tx_index = 0

        self.current_tx = None

        self.current_paths = []

        self.current_bucket = None

        self.bucket = None

        self.backtracker.bucket = None

        # --------------------------------------------------
        # Reset routing history
        # --------------------------------------------------

        self.last_failure_probability = 0.0

        self.last_average_delay = 0.0

        self.last_backtrack_count = 0

        # --------------------------------------------------
        # Reset episode statistics
        # --------------------------------------------------

        self.episode_reward = 0.0

        self.episode_steps = 0

        self.total_successes = 0

        self.total_failures = 0

        # --------------------------------------------------
        # Select first transaction
        # --------------------------------------------------

        self.current_tx = (
            self._get_current_transaction()
        )

        if self.current_tx is None:
            raise RuntimeError(
                "No valid transaction is available after reset."
            )

        # --------------------------------------------------
        # Build initial observation
        # --------------------------------------------------

        observation = self._build_state()

        info = {
            "transaction_id": self._transaction_id(),
            "eta": None,
            "top_k": self.top_k,
        }

        return observation, info

    # ======================================================
    # FAILURE MODEL COMPATIBILITY RESET
    # ======================================================

    def _reset_failure_graph_state(self):
        """
        Legacy compatibility helper.

        The current FailureModel implementation must expose
        reset_runtime_state(). This method remains only as a
        structural helper for explicit manual graph-state reset.
        """

        # Nodes
        for node, data in self.G.nodes(
            data=True
        ):

            if "available" in data:
                data["available"] = True

            if "is_online" in data:
                data["is_online"] = True

        # Edges
        if self.G.is_multigraph():

            for u, v, key, data in self.G.edges(
                keys=True,
                data=True,
            ):

                if "available" in data:
                    data["available"] = True

                if "last_failure" in data:
                    data["last_failure"] = None

        else:

            for u, v, data in self.G.edges(
                data=True
            ):

                if "available" in data:
                    data["available"] = True

                if "last_failure" in data:
                    data["last_failure"] = None

    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action,
    ):
        """
        Execute one PPO routing decision.

        One environment step corresponds to one transaction.

        PPO action:
            eta

        Routing:
            eta -> Top-K(k=5) -> Bucket -> Payment

        Failure handling:
            Partial Backtracking -> Full Reroute

        PPO learning itself is performed by Stable-Baselines3
        model.learn(), not inside this environment.
        """

        if self.current_tx is None:

            raise RuntimeError(
                "Environment has no active transaction. "
                "Call reset() before step()."
            )

        # --------------------------------------------------
        # 1. Decode PPO action
        # --------------------------------------------------

        eta = self._decode_eta(
            action
        )

        tx = self.current_tx

        # --------------------------------------------------
        # 2. Generate Top-K candidates
        # --------------------------------------------------

        candidates = self._generate_candidates(
            tx=tx,
            eta=eta,
        )

        # --------------------------------------------------
        # 3. No initial path
        # --------------------------------------------------

        if not candidates:

            reward = self._calculate_environment_reward(
                success=False,
                path_length=0,
                carbon_intensity=0.0,
                fee=0.0,
                delay=0.0,
                partial_backtrack_count=0,
                full_reroute_count=0,
                attempt_count=0,
            )

            self.last_failure_probability = 1.0

            self.last_average_delay = 0.0

            self.last_backtrack_count = 0

            self.total_failures += 1

            self.episode_reward += reward

            self.episode_steps += 1

            self._update_network_dynamics()

            info = {
                "transaction_id": self._transaction_id(),
                "success": False,
                "eta": eta,
                "top_k": self.top_k,
                "candidate_path_count": 0,
                "usable_candidate_count": 0,
                "bucket_size": 0,
                "payment_success": False,
                "failure_probability": 1.0,
                "average_delay": 0.0,
                "backtrack_count": 0,
                "partial_backtrack_count": 0,
                "partial_backtrack_success": 0,
                "full_reroute": False,
                "full_reroute_count": 0,
                "attempt_count": 0,
                "reason": "no_initial_candidate",
            }

            terminated = self._advance_transaction()

            observation = (
                self._build_state()
                if not terminated
                else np.zeros(
                    self.observation_space.shape,
                    dtype=np.float32,
                )
            )

            return (
                observation,
                float(reward),
                terminated,
                False,
                info,
            )

        # --------------------------------------------------
        # 4. CandidateManager
        # --------------------------------------------------

        bucket, filtered_candidates = (
            self._create_bucket(
                tx=tx,
                candidates=candidates,
            )
        )

        # --------------------------------------------------
        # 5. No usable Bucket
        # --------------------------------------------------

        if bucket is None:

            reward = self._calculate_environment_reward(
                success=False,
                path_length=0,
                carbon_intensity=0.0,
                fee=0.0,
                delay=0.0,
                partial_backtrack_count=0,
                full_reroute_count=0,
                attempt_count=0,
            )

            self.last_failure_probability = 1.0

            self.last_average_delay = 0.0

            self.last_backtrack_count = 0

            self.total_failures += 1

            self.episode_reward += reward

            self.episode_steps += 1

            self._update_network_dynamics()

            info = {
                "transaction_id": self._transaction_id(),
                "success": False,
                "eta": eta,
                "top_k": self.top_k,
                "candidate_path_count": len(
                    filtered_candidates
                ),
                "usable_candidate_count": 0,
                "bucket_size": 0,
                "payment_success": False,
                "failure_probability": 1.0,
                "average_delay": 0.0,
                "backtrack_count": 0,
                "partial_backtrack_count": 0,
                "partial_backtrack_success": 0,
                "full_reroute": False,
                "full_reroute_count": 0,
                "attempt_count": 0,
                "reason": "no_usable_bucket",
            }

            terminated = self._advance_transaction()

            observation = (
                self._build_state()
                if not terminated
                else np.zeros(
                    self.observation_space.shape,
                    dtype=np.float32,
                )
            )

            return (
                observation,
                float(reward),
                terminated,
                False,
                info,
            )

        # --------------------------------------------------
        # 6. Store candidates and Bucket
        # --------------------------------------------------

        self.current_paths = list(
            filtered_candidates
        )

        self.current_bucket = bucket

        self.bucket = bucket

        self.backtracker.bucket = bucket

        # --------------------------------------------------
        # 7. Execute routing pipeline
        # --------------------------------------------------

        result = self._execute_bucket_pipeline(
            tx=tx,
            eta=eta,
        )

        # --------------------------------------------------
        # 8. Extract final result
        # --------------------------------------------------

        payment_success = bool(
            result["success"]
        )

        final_path_length = int(
            result["path_length"]
        )

        final_fee = float(
            result["fee"]
        )

        final_delay = float(
            result["delay"]
        )

        final_carbon = float(
            result["carbon"]
        )

        backtrack_count = int(
            result["backtrack_count"]
        )

        partial_backtrack_count = int(
            result["partial_backtrack_count"]
        )

        partial_backtrack_success = int(
            result["partial_backtrack_success"]
        )

        full_reroute_count = int(
            result["full_reroute_count"]
        )

        attempt_count = int(
            result["attempt_count"]
        )

        # --------------------------------------------------
        # 9. Update environment state
        # --------------------------------------------------

        self.last_failure_probability = float(
            result["failure_probability"]
        )

        self.last_average_delay = (
            final_delay
        )

        self.last_backtrack_count = (
            backtrack_count
        )

        # --------------------------------------------------
        # 10. Calculate PPO reward
        # --------------------------------------------------

        reward = self._calculate_environment_reward(
            success=payment_success,
            path_length=final_path_length,
            carbon_intensity=final_carbon,
            fee=final_fee,
            delay=final_delay,
            partial_backtrack_count=(
                partial_backtrack_count
            ),
            full_reroute_count=(
                full_reroute_count
            ),
            attempt_count=(
                attempt_count
            ),
        )

        # --------------------------------------------------
        # 11. Statistics
        # --------------------------------------------------

        if payment_success:

            self.total_successes += 1

        else:

            self.total_failures += 1

        self.episode_reward += reward

        self.episode_steps += 1

        # --------------------------------------------------
        # 12. Network dynamics
        # --------------------------------------------------

        self._update_network_dynamics()

        # --------------------------------------------------
        # 13. Information returned to PPO
        # --------------------------------------------------

        info = {
            "transaction_id": self._transaction_id(),

            # RL
            "eta": eta,

            # Fixed Top-K
            "top_k": self.top_k,

            # Candidate generation
            "candidate_path_count": len(
                candidates
            ),

            "usable_candidate_count": len(
                filtered_candidates
            ),

            "bucket_size": self._bucket_size(
                self.current_bucket
            ),

            # Payment
            "success": payment_success,

            "payment_success": payment_success,

            "path": result["path"],

            "path_length": final_path_length,

            "fee": final_fee,

            "delay": final_delay,

            "carbon": final_carbon,

            # Failure / recovery
            "failure_probability": (
                self.last_failure_probability
            ),

            "average_delay": (
                self.last_average_delay
            ),

            "backtrack_count": (
                backtrack_count
            ),

            "partial_backtrack_count": (
                partial_backtrack_count
            ),

            "partial_backtrack_success": (
                partial_backtrack_success
            ),

            "full_reroute": (
                full_reroute_count > 0
            ),

            "full_reroute_count": (
                full_reroute_count
            ),

            "attempt_count": (
                attempt_count
            ),

            "reason": result["reason"],

            "episode_reward": (
                self.episode_reward
            ),
        }

        # --------------------------------------------------
        # 14. Advance transaction
        # --------------------------------------------------

        terminated = self._advance_transaction()

        if terminated:

            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )

        else:

            observation = self._build_state()

        return (
            observation,
            float(reward),
            terminated,
            False,
            info,
        )

    # ======================================================
    # GENERATE TOP-K CANDIDATES
    # ======================================================

    def _generate_candidates(
        self,
        tx,
        eta,
    ):
        """
        Generate Top-K candidates.

        k is ALWAYS 5.

        PPO controls eta only.
        """

        if tx is None:
            raise ValueError(
                "Transaction cannot be None."
            )

        eta = self._validate_eta(
            eta
        )

        candidates = top_k_paths(
            G=self.G,
            source=tx.source,
            target=tx.destination,
            amount=tx.amount,
            heuristic_fn=self.heuristic_fn,
            eta=eta,
            k=self.top_k,
            max_hops=self.max_hops,
            lambda_h=self.lambda_h,
        )

        if candidates is None:
            raise RuntimeError(
                "top_k_paths() returned None."
            )

        if not isinstance(
            candidates,
            (list, tuple),
        ):
            raise TypeError(
                "top_k_paths() must return a list or tuple."
            )

        return list(candidates)

    # ======================================================
    # CREATE BUCKET
    # ======================================================

    def _create_bucket(
        self,
        tx,
        candidates,
    ):
        """
        Filter and rank candidates using CandidateManager.
        """

        if not candidates:
            return None, []

        manager = CandidateManager(
            candidates=list(candidates),
            max_candidates=self.top_k,
        )

        filtered = manager.filter_candidates(
            self.G,
            tx.amount,
        )

        if filtered is None:
            raise RuntimeError(
                "CandidateManager.filter_candidates() "
                "returned None."
            )

        if not filtered:
            return None, []

        manager.rank_candidates()

        bucket = manager.create_bucket(
            tx_id=tx.tx_id,
            k=self.top_k,
        )

        if bucket is None:
            raise RuntimeError(
                "CandidateManager.create_bucket() "
                "returned None."
            )

        if not hasattr(
            bucket,
            "candidates",
        ):
            raise TypeError(
                "CandidateManager returned an invalid Bucket."
            )

        return (
            bucket,
            list(manager.candidates),
        )

    # ======================================================
    # EXECUTE BUCKET PIPELINE
    # ======================================================

    def _execute_bucket_pipeline(
        self,
        tx,
        eta,
    ):
        """
        Execute:

            Bucket
              |
              v
        PaymentSimulator
              |
           FailureModel
              |
              v
        Partial Backtracking
              |
              v
        Bucket Alternative
              |
              v
        Full Reroute
              |
              v
        Payment

        Important
        ---------
        attempt_count counts every actual call to
        PaymentSimulator.simulate_payment().

        This includes:

            1. initial Bucket attempts
            2. Partial Backtracking retries
            3. full-reroute attempts

        Bucket.attempts follows the same semantics.
        """

        current_bucket = (
            self.current_bucket
        )

        if current_bucket is None:
            raise RuntimeError(
                "No active Bucket exists."
            )

        self.backtracker.bucket = (
            current_bucket
        )

        attempt_count = 0

        partial_backtrack_count = 0

        partial_backtrack_success = 0

        full_reroute_count = 0

        failed_routes = set()

        failed_edges = set()

        final_success = False

        final_path = []

        final_fee = 0.0

        final_delay = 0.0

        final_carbon = 0.0

        final_failure_probability = 1.0

        final_reason = None

        # --------------------------------------------------
        # Maximum attempts
        #
        # This is a hard limit on ACTUAL payment executions.
        # --------------------------------------------------

        max_attempts = (
            (self.top_k * 3)
            + 5
        )

        # --------------------------------------------------
        # Attempt loop
        # --------------------------------------------------

        while attempt_count < max_attempts:

            candidate = (
                current_bucket.current()
            )

            # =================================================
            # BUCKET EXHAUSTED
            # =================================================

            if candidate is None:

                reroute_bucket, reroute_candidates = (
                    self._full_reroute(
                        tx=tx,
                        eta=eta,
                        failed_routes=failed_routes,
                        failed_edges=failed_edges,
                    )
                )

                if reroute_bucket is None:

                    final_reason = (
                        "bucket_exhausted_no_reroute"
                    )

                    break

                full_reroute_count += 1

                current_bucket = (
                    reroute_bucket
                )

                self.current_bucket = (
                    current_bucket
                )

                self.bucket = (
                    current_bucket
                )

                self.backtracker.bucket = (
                    current_bucket
                )

                continue

            # =================================================
            # EXTRACT CANDIDATE ROUTE
            # =================================================

            path, candidate_edges = (
                self._candidate_route(
                    candidate
                )
            )

            if not path:

                current_bucket.backtrack(
                    failed_candidate=candidate
                )

                continue

            # --------------------------------------------------
            # Exact candidate edges are authoritative.
            # --------------------------------------------------

            edges = candidate_edges

            if not edges:

                edges = self._path_to_edges(
                    path
                )

            if edges is None:

                failed_routes.add(
                    tuple(path)
                )

                current_bucket.backtrack(
                    failed_candidate=candidate,
                    reason="cannot_resolve_exact_route_edges",
                )

                continue

            # --------------------------------------------------
            # Validate candidate route against graph.
            # --------------------------------------------------

            self._validate_route_edges(
                path,
                edges,
            )

            # --------------------------------------------------
            # Bucket lifecycle:
            #
            # selection -> actual attempt
            # --------------------------------------------------

            selected = current_bucket.select_candidate(
                candidate
            )

            if selected is None:
                raise RuntimeError(
                    "Bucket could not select its current candidate."
                )

            if attempt_count >= max_attempts:
                break

            current_bucket.record_attempt()

            attempt_count += 1

            # =================================================
            # PAYMENT
            # =================================================

            payment_result = (
                self.payment_simulator.simulate_payment(
                    path=path,
                    edges=edges,
                    amount=tx.amount,
                    tx_id=tx.tx_id,
                )
            )

            result = self._result_to_dict(
                payment_result
            )

            success = self._require_bool(
                result,
                "success",
                "PaymentSimulator result",
            )

            # =================================================
            # SUCCESS
            # =================================================

            if success:

                if isinstance(
                    candidate,
                    dict,
                ):
                    candidate["success"] = True

                if not current_bucket.mark_success(
                    candidate
                ):
                    raise RuntimeError(
                        "Bucket.mark_success() failed after "
                        "a successful payment."
                    )

                final_success = True

                final_path = list(path)

                final_fee = self._result_float(
                    result,
                    "fee",
                    default=(
                        candidate.get(
                            "total_fee",
                            0.0,
                        )
                        if isinstance(
                            candidate,
                            dict,
                        )
                        else 0.0
                    ),
                )

                final_delay = self._result_float(
                    result,
                    "delay",
                    default=(
                        candidate.get(
                            "total_delay",
                            0.0,
                        )
                        if isinstance(
                            candidate,
                            dict,
                        )
                        else 0.0
                    ),
                )

                final_carbon = self._result_float(
                    result,
                    "carbon",
                    default=0.0,
                )

                final_failure_probability = (
                    self._candidate_failure_probability(
                        candidate
                    )
                )

                final_reason = result.get(
                    "reason",
                    "success",
                )

                break

            # =================================================
            # FAILURE
            # =================================================

            if isinstance(
                candidate,
                dict,
            ):
                candidate["success"] = False

            failed_routes.add(
                tuple(path)
            )

            # --------------------------------------------------
            # Record exact failed edge
            # --------------------------------------------------

            failed_edge = result.get(
                "failed_edge"
            )

            failure_index = result.get(
                "failure_index"
            )

            normalized_failed_edge = (
                self._normalize_edge(
                    failed_edge
                )
            )

            if normalized_failed_edge is not None:

                failed_edges.add(
                    normalized_failed_edge
                )

            # =================================================
            # NO FAILURE INFORMATION
            # =================================================

            if (
                failed_edge is None
                and
                failure_index is None
            ):

                current_bucket.backtrack(
                    failed_candidate=candidate,
                    reason=result.get(
                        "reason",
                        "payment_failed",
                    ),
                )

                final_reason = result.get(
                    "reason",
                    "payment_failed",
                )

                final_failure_probability = 1.0

                continue

            # =================================================
            # PARTIAL BACKTRACKING
            # =================================================

            partial_backtrack_count += 1

            backtrack_result = (
                self.backtracker.backtrack(
                    route=path,
                    failed_edge=failed_edge,
                    failure_index=failure_index,
                    amount=tx.amount,
                    bucket_id=current_bucket.bucket_id,
                    attempt_id=attempt_count,
                )
            )

            if not isinstance(
                backtrack_result,
                dict,
            ):
                raise TypeError(
                    "PartialBacktracker.backtrack() must "
                    "return a dictionary."
                )

            status = backtrack_result.get(
                "status"
            )

            # =================================================
            # PARTIAL BACKTRACK SUCCESS
            # =================================================

            if (
                status == "alternative_found"
                and
                backtrack_result.get(
                    "success",
                    False,
                )
            ):

                new_path = (
                    backtrack_result.get(
                        "new_route"
                    )
                )

                if new_path:

                    new_edges = (
                        backtrack_result.get(
                            "new_edges"
                        )
                    )

                    if not new_edges:

                        new_edges = (
                            self._path_to_edges(
                                new_path
                            )
                        )

                    if new_edges is None:

                        raise RuntimeError(
                            "PartialBacktracker returned a route "
                            "without resolvable exact edges."
                        )

                    self._validate_route_edges(
                        new_path,
                        new_edges,
                    )

                    # ------------------------------------------
                    # Retry is a REAL payment attempt.
                    # ------------------------------------------

                    if attempt_count < max_attempts:

                        partial_backtrack_success += 1

                        attempt_count += 1

                        self._record_external_bucket_attempt(
                            current_bucket
                        )

                        retry_result = (
                            self.payment_simulator.simulate_payment(
                                path=new_path,
                                edges=new_edges,
                                amount=tx.amount,
                                tx_id=tx.tx_id,
                            )
                        )

                        retry_dict = (
                            self._result_to_dict(
                                retry_result
                            )
                        )

                        retry_success = self._require_bool(
                            retry_dict,
                            "success",
                            "PaymentSimulator retry result",
                        )

                        if retry_success:

                            final_success = True

                            final_path = list(
                                new_path
                            )

                            final_fee = self._result_float(
                                retry_dict,
                                "fee",
                                default=0.0,
                            )

                            final_delay = self._result_float(
                                retry_dict,
                                "delay",
                                default=0.0,
                            )

                            final_carbon = self._result_float(
                                retry_dict,
                                "carbon",
                                default=0.0,
                            )

                            final_failure_probability = 0.0

                            final_reason = (
                                "partial_backtrack_success"
                            )

                            # ----------------------------------
                            # A partial-backtracking retry is
                            # successful payment completion.
                            #
                            # The original Bucket candidate is
                            # already failed, so it must not be
                            # marked successful.
                            # ----------------------------------

                            break

                        # --------------------------------------
                        # Retry failed.
                        # --------------------------------------

                        retry_failed_route = tuple(
                            new_path
                        )

                        failed_routes.add(
                            retry_failed_route
                        )

                        retry_failed_edge = (
                            retry_dict.get(
                                "failed_edge"
                            )
                        )

                        retry_failed_edge = (
                            self._normalize_edge(
                                retry_failed_edge
                            )
                        )

                        if retry_failed_edge is not None:

                            failed_edges.add(
                                retry_failed_edge
                            )

                        final_failure_probability = 1.0

            # =================================================
            # PARTIAL BACKTRACK FAILED
            # =================================================

            current_bucket.backtrack(
                failed_candidate=candidate,
                failed_channel=failed_edge,
                reason=result.get(
                    "reason",
                    "payment_failed",
                ),
            )

            # --------------------------------------------------
            # Bucket still contains another candidate
            # --------------------------------------------------

            if current_bucket.current() is not None:

                continue

            # =================================================
            # FULL REROUTE
            # =================================================

            if attempt_count >= max_attempts:
                final_reason = (
                    "maximum_payment_attempts_reached"
                )
                break

            reroute_bucket, reroute_candidates = (
                self._full_reroute(
                    tx=tx,
                    eta=eta,
                    failed_routes=failed_routes,
                    failed_edges=failed_edges,
                )
            )

            if reroute_bucket is None:

                final_reason = (
                    "full_reroute_failed"
                )

                break

            full_reroute_count += 1

            current_bucket = (
                reroute_bucket
            )

            self.current_bucket = (
                current_bucket
            )

            self.bucket = (
                current_bucket
            )

            self.backtracker.bucket = (
                current_bucket
            )

        # ==================================================
        # Final failure
        # ==================================================

        if not final_success:

            final_reason = (
                final_reason
                or
                "payment_failed"
            )

            final_failure_probability = 1.0

        # ==================================================
        # Final metrics
        # ==================================================

        path_length = max(
            len(final_path) - 1,
            0,
        )

        return {
            "success": final_success,

            "path": final_path,

            "path_length": path_length,

            "fee": final_fee,

            "delay": final_delay,

            "carbon": final_carbon,

            "failure_probability": (
                float(
                    final_failure_probability
                )
            ),

            "reason": final_reason,

            "attempt_count": attempt_count,

            "backtrack_count": (
                partial_backtrack_count
            ),

            "partial_backtrack_count": (
                partial_backtrack_count
            ),

            "partial_backtrack_success": (
                partial_backtrack_success
            ),

            "full_reroute": (
                full_reroute_count > 0
            ),

            "full_reroute_count": (
                full_reroute_count
            ),
        }

    # ======================================================
    # FULL REROUTE
    # ======================================================

    def _full_reroute(
        self,
        tx,
        eta,
        failed_routes=None,
        failed_edges=None,
    ):
        """
        Generate a new Top-K candidate set.

        The same eta selected by PPO is used.

        k remains fixed at 5.

        A candidate is rejected when:

            1. the complete path already failed, or
            2. it reuses an exact failed channel, or
            3. an unknown failed channel key blocks the same
               directed endpoint in a MultiDiGraph.
        """

        candidates = self._generate_candidates(
            tx=tx,
            eta=eta,
        )

        if not candidates:
            return None, []

        failed_routes = (
            failed_routes
            if failed_routes is not None
            else set()
        )

        failed_edges = (
            failed_edges
            if failed_edges is not None
            else set()
        )

        filtered = []

        for candidate in candidates:

            if not isinstance(
                candidate,
                dict,
            ):
                raise TypeError(
                    "Top-K candidate must be a dictionary."
                )

            path = candidate.get(
                "path"
            )

            if not isinstance(
                path,
                (list, tuple),
            ):
                raise ValueError(
                    "Top-K candidate has invalid path."
                )

            path_tuple = tuple(path)

            # ----------------------------------------------
            # Reject already failed complete route
            # ----------------------------------------------

            if path_tuple in failed_routes:
                continue

            # ----------------------------------------------
            # Extract exact candidate channels
            # ----------------------------------------------

            candidate_edges = (
                candidate.get(
                    "edges"
                )
            )

            if candidate_edges is None:
                raise ValueError(
                    "Top-K candidate is missing exact edges."
                )

            if not isinstance(
                candidate_edges,
                (list, tuple),
            ):
                raise TypeError(
                    "Candidate edges must be a list or tuple."
                )

            if len(candidate_edges) != len(path) - 1:
                raise ValueError(
                    "Candidate path/edge count mismatch."
                )

            # ----------------------------------------------
            # Exact failed-channel filtering
            # ----------------------------------------------

            rejected = False

            for candidate_edge in candidate_edges:

                normalized_candidate_edge = (
                    self._normalize_edge(
                        candidate_edge
                    )
                )

                if normalized_candidate_edge is None:
                    raise ValueError(
                        "Candidate contains invalid edge."
                    )

                for failed_edge in failed_edges:

                    if self._edge_matches_failure(
                        normalized_candidate_edge,
                        failed_edge,
                    ):
                        rejected = True
                        break

                if rejected:
                    break

            if rejected:
                continue

            # ----------------------------------------------
            # Candidate survives reroute filtering
            # ----------------------------------------------

            filtered.append(
                candidate
            )

        if not filtered:
            return None, []

        bucket, bucket_candidates = (
            self._create_bucket(
                tx=tx,
                candidates=filtered,
            )
        )

        return (
            bucket,
            bucket_candidates,
        )

    # ======================================================
    # CANDIDATE EXTRACTION
    # ======================================================

    def _candidate_route(
        self,
        candidate,
    ):
        """
        Extract path and exact channel list.

        Preferred format:

            {
                "path": [...],
                "edges": [(u,v,key), ...]
            }

        Exact edges are required for MultiDiGraph candidates.
        """

        if isinstance(
            candidate,
            dict,
        ):

            path = candidate.get(
                "path"
            )

            edges = candidate.get(
                "edges"
            )

            if path is None:
                raise ValueError(
                    "Candidate dictionary is missing 'path'."
                )

            if edges is None:
                raise ValueError(
                    "Candidate dictionary is missing 'edges'."
                )

            return path, edges

        # --------------------------------------------------
        # Legacy tuple format
        # --------------------------------------------------

        if isinstance(
            candidate,
            (tuple, list),
        ):

            if len(candidate) >= 2:

                return (
                    candidate[0],
                    candidate[1],
                )

        raise TypeError(
            "Unsupported candidate format."
        )

    # ======================================================
    # EDGE NORMALIZATION
    # ======================================================

    @staticmethod
    def _normalize_edge(
        edge,
    ):
        """
        Normalize an edge into:

            (u, v, key)

        For non-multigraph-style edges:

            (u, v) -> (u, v, None)

        Only exact 2-element or 3-element representations
        are accepted.
        """

        if edge is None:
            return None

        if not isinstance(
            edge,
            (tuple, list),
        ):
            return None

        if len(edge) == 3:

            return (
                edge[0],
                edge[1],
                edge[2],
            )

        if len(edge) == 2:

            return (
                edge[0],
                edge[1],
                None,
            )

        return None

    # ======================================================
    # FAILED EDGE MATCHING
    # ======================================================

    def _edge_matches_failure(
        self,
        candidate_edge,
        failed_edge,
    ):
        """
        Determine whether a candidate edge reuses a failed
        channel.

        For MultiDiGraph:

            failed key = exact key
                -> exact channel is blocked.

            failed key = None
                -> the directed endpoint (u,v) is blocked
                   conservatively because the failed parallel
                   channel is unknown.

        For ordinary graphs:

            endpoint identity is sufficient.
        """

        candidate = self._normalize_edge(
            candidate_edge
        )

        failed = self._normalize_edge(
            failed_edge
        )

        if candidate is None or failed is None:
            return False

        cu, cv, ck = candidate

        fu, fv, fk = failed

        if cu != fu or cv != fv:
            return False

        if not self.G.is_multigraph():
            return True

        if fk is None:
            return True

        return ck == fk

    # ======================================================
    # PATH -> EXACT EDGES
    # ======================================================

    def _path_to_edges(
        self,
        path,
    ):
        """
        Resolve a node path to currently available directed
        channels.

        For MultiDiGraph:

            - zero available channels -> None
            - one available channel -> exact key is retained
            - multiple available channels -> None

        The method NEVER chooses an arbitrary parallel channel.
        """

        if not isinstance(
            path,
            (list, tuple),
        ):
            raise TypeError(
                "path must be a list or tuple."
            )

        if len(path) < 2:
            return None

        edges = []

        for u, v in zip(
            path[:-1],
            path[1:],
        ):

            if not self.G.has_edge(
                u,
                v,
            ):
                return None

            edge_data = self.G.get_edge_data(
                u,
                v,
            )

            if not edge_data:
                return None

            if self.G.is_multigraph():

                available_channels = []

                for key, data in edge_data.items():

                    if not isinstance(
                        data,
                        dict,
                    ):
                        raise TypeError(
                            "MultiGraph edge data must be a dictionary."
                        )

                    available = data.get(
                        "available",
                        True,
                    )

                    if not isinstance(
                        available,
                        bool,
                    ):
                        raise TypeError(
                            "Edge 'available' must be bool."
                        )

                    if available:
                        available_channels.append(
                            key
                        )

                # Exact channel identity cannot be inferred
                # when multiple parallel channels are available.
                if len(available_channels) != 1:
                    return None

                selected_key = (
                    available_channels[0]
                )

                edges.append(
                    (
                        u,
                        v,
                        selected_key,
                    )
                )

            else:

                if not isinstance(
                    edge_data,
                    dict,
                ):
                    raise TypeError(
                        "Edge data must be a dictionary."
                    )

                available = edge_data.get(
                    "available",
                    True,
                )

                if not isinstance(
                    available,
                    bool,
                ):
                    raise TypeError(
                        "Edge 'available' must be bool."
                    )

                if not available:
                    return None

                edges.append(
                    (
                        u,
                        v,
                    )
                )

        return edges

    # ======================================================
    # ROUTE EDGE VALIDATION
    # ======================================================

    def _validate_route_edges(
        self,
        path,
        edges,
    ):
        """
        Validate exact path/edge correspondence.

        This function does not choose channels.
        """

        if not isinstance(
            path,
            (list, tuple),
        ):
            raise TypeError(
                "Route path must be a list or tuple."
            )

        if not isinstance(
            edges,
            (list, tuple),
        ):
            raise TypeError(
                "Route edges must be a list or tuple."
            )

        if len(path) < 2:
            raise ValueError(
                "Route path must contain at least two nodes."
            )

        if len(edges) != len(path) - 1:
            raise ValueError(
                "Route path and edge counts do not match."
            )

        for index, edge in enumerate(edges):

            normalized = self._normalize_edge(
                edge
            )

            if normalized is None:
                raise ValueError(
                    "Route contains an invalid edge."
                )

            u, v, key = normalized

            if u != path[index]:
                raise ValueError(
                    "Route edge source does not match path."
                )

            if v != path[index + 1]:
                raise ValueError(
                    "Route edge destination does not match path."
                )

            if self.G.is_multigraph():

                if key is None:
                    raise ValueError(
                        "MultiGraph route requires an exact channel key."
                    )

                if not self.G.has_edge(
                    u,
                    v,
                    key,
                ):
                    raise ValueError(
                        f"Exact channel ({u}, {v}, {key}) "
                        "does not exist."
                    )

            else:

                if key is not None:
                    raise ValueError(
                        "Simple graph route must not contain a channel key."
                    )

                if not self.G.has_edge(
                    u,
                    v,
                ):
                    raise ValueError(
                        f"Edge ({u}, {v}) does not exist."
                    )

    # ======================================================
    # PAYMENT RESULT NORMALIZATION
    # ======================================================

    @staticmethod
    def _result_to_dict(
        result,
    ):
        """
        Normalize PaymentResult or dictionary into dict.
        """

        if hasattr(
            result,
            "to_dict",
        ):

            result = result.to_dict()

        if not isinstance(
            result,
            dict,
        ):

            raise TypeError(
                "Payment simulator must return "
                "a dictionary or an object with to_dict()."
            )

        if "success" not in result:
            raise ValueError(
                "Payment simulator result is missing 'success'."
            )

        if not isinstance(
            result["success"],
            (bool, np.bool_),
        ):
            raise TypeError(
                "Payment simulator result 'success' must be bool."
            )

        return result

    # ======================================================
    # ETA DECODER
    # ======================================================

    def _decode_eta(
        self,
        action,
    ):
        """
        Convert PPO action into scalar eta.

        PPO action space is exactly:

            Box(
                low=[eta_min],
                high=[eta_max],
                shape=(1,)
            )

        No clipping is performed.

        An action outside the declared action space is invalid.
        """

        array = np.asarray(
            action,
            dtype=np.float32,
        )

        if array.shape != (1,):
            raise ValueError(
                "PPO action must have exactly shape (1,). "
                f"Received shape: {array.shape}"
            )

        eta = float(
            array[0]
        )

        if not np.isfinite(
            eta
        ):
            raise ValueError(
                "PPO produced a non-finite eta."
            )

        if eta < self.eta_min or eta > self.eta_max:
            raise ValueError(
                "PPO eta is outside the declared action space: "
                f"{eta} not in "
                f"[{self.eta_min}, {self.eta_max}]."
            )

        return eta

    # ======================================================
    # REWARD
    # ======================================================

    def _calculate_environment_reward(
        self,
        success,
        path_length,
        carbon_intensity,
        fee,
        delay,
        partial_backtrack_count,
        full_reroute_count,
        attempt_count,
    ):
        """
        Calculate the environment reward.

        The current reward.py interface is used directly.

        No compatibility fallback is used because silently
        switching reward semantics can invalidate the RL experiment.
        """

        reward_cfg = self.cfg.get(
            "reward",
            {},
        )

        if not isinstance(
            reward_cfg,
            dict,
        ):
            raise TypeError(
                "config['reward'] must be a dictionary."
            )

        kwargs = {
            "success": bool(success),

            "path_length": int(
                path_length
            ),

            "carbon_intensity": float(
                carbon_intensity
            ),

            "fee": float(
                fee
            ),

            "delay": float(
                delay
            ),

            "partial_backtrack_count": int(
                partial_backtrack_count
            ),

            "full_reroute_count": int(
                full_reroute_count
            ),

            "attempt_count": int(
                attempt_count
            ),

            "scale": self._reward_float(
                reward_cfg,
                "scale",
                100.0,
            ),

            "fee_reference": self._reward_float(
                reward_cfg,
                "fee_reference",
                1000.0,
            ),

            "delay_reference": self._reward_float(
                reward_cfg,
                "delay_reference",
                10.0,
            ),

            "carbon_reference": self._reward_float(
                reward_cfg,
                "carbon_reference",
                100.0,
            ),

            "path_reference": self._reward_float(
                reward_cfg,
                "path_reference",
                10.0,
            ),
        }

        reward = calculate_reward(
            **kwargs
        )

        reward = float(
            reward
        )

        if not math.isfinite(
            reward
        ):
            raise RuntimeError(
                "calculate_reward() returned a non-finite value."
            )

        return reward

    # ======================================================
    # NETWORK DYNAMICS
    # ======================================================

    def _update_network_dynamics(self):
        """
        Advance dynamic network conditions.

        NetworkDynamics is responsible for persistent
        availability/recovery evolution between transactions.
        """

        if self.network_dynamics is None:
            raise RuntimeError(
                "NetworkDynamics is not initialized."
            )

        update_method = getattr(
            self.network_dynamics,
            "update",
            None,
        )

        if not callable(
            update_method
        ):
            raise RuntimeError(
                "NetworkDynamics must provide update()."
            )

        update_method()

    # ======================================================
    # STATE
    # ======================================================

    def _build_state(self):
        """
        Build the observation for the current transaction.
        """

        if self.current_tx is None:

            return np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )

        bucket_info = {}

        if self.current_bucket is not None:

            bucket_info = self._bucket_info(
                self.current_bucket
            )

        simulation_info = {
            "failure_probability": (
                self.last_failure_probability
            ),

            "average_delay": (
                self.last_average_delay
            ),

            "backtrack_count": (
                self.last_backtrack_count
            ),
        }

        state_builder = State(
            G=self.G,
            source=self.current_tx.source,
            destination=self.current_tx.destination,
            transaction=self.current_tx,
            candidate_paths=self.current_paths,
            simulation_info=simulation_info,
            bucket_info=bucket_info,
            k=self.state_k,
            radius=self.state_radius,
        )

        vector = np.asarray(
            state_builder.vector(),
            dtype=np.float32,
        ).reshape(-1)

        expected_shape = (
            self.observation_space.shape
        )

        if vector.shape != expected_shape:

            raise RuntimeError(
                "Invalid observation shape.\n"
                f"Expected: {expected_shape}\n"
                f"Received: {vector.shape}"
            )

        if not np.all(
            np.isfinite(vector)
        ):

            raise RuntimeError(
                "Observation contains NaN or Inf."
            )

        return vector

    # ======================================================
    # BUCKET INFO
    # ======================================================

    @staticmethod
    def _bucket_size(
        bucket,
    ):
        """
        Return exact Bucket candidate count.

        The current Bucket implementation exposes candidates
        directly; no exception-based fallback is used.
        """

        if bucket is None:
            return 0

        if not hasattr(
            bucket,
            "candidates",
        ):
            raise TypeError(
                "Bucket must expose candidates."
            )

        candidates = bucket.candidates

        if not isinstance(
            candidates,
            list,
        ):
            raise TypeError(
                "Bucket.candidates must be a list."
            )

        return len(candidates)

    @classmethod
    def _bucket_info(
        cls,
        bucket,
    ):
        if bucket is None:
            return {}

        if not hasattr(
            bucket,
            "info",
        ):
            raise TypeError(
                "Bucket must provide info()."
            )

        info = bucket.info()

        if not isinstance(
            info,
            dict,
        ):
            raise TypeError(
                "Bucket.info() must return a dictionary."
            )

        result = {
            "size": cls._bucket_size(
                bucket
            ),
        }

        result.update(
            info
        )

        return result

    # ======================================================
    # REWARD CONFIGURATION
    # ======================================================

    def _reward_scale(self):
        """
        Obtain reward scale from configuration.
        """

        reward_cfg = self.cfg.get(
            "reward",
            {},
        )

        if not isinstance(
            reward_cfg,
            dict,
        ):
            raise TypeError(
                "config['reward'] must be a dictionary."
            )

        return self._reward_float(
            reward_cfg,
            "scale",
            100.0,
        )

    # ======================================================
    # TRANSACTION ACCESS
    # ======================================================

    def _get_current_transaction(
        self,
    ):

        if self.tx_index >= len(
            self.transactions
        ):

            return None

        transaction = self.transactions[
            self.tx_index
        ]

        if transaction is None:
            raise ValueError(
                f"Transaction at index {self.tx_index} is None."
            )

        for field in (
            "source",
            "destination",
            "amount",
            "tx_id",
        ):

            if not hasattr(
                transaction,
                field,
            ):
                raise TypeError(
                    f"Transaction is missing required field '{field}'."
                )

        try:

            amount = float(
                transaction.amount
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "Transaction amount must be numeric."
            )

        if not math.isfinite(
            amount
        ) or amount <= 0:

            raise ValueError(
                "Transaction amount must be finite and positive."
            )

        if transaction.source not in self.G:
            raise ValueError(
                "Transaction source is not present in graph."
            )

        if transaction.destination not in self.G:
            raise ValueError(
                "Transaction destination is not present in graph."
            )

        if transaction.source == transaction.destination:
            raise ValueError(
                "Transaction source and destination must differ."
            )

        return transaction

    # ======================================================
    # TRANSACTION ID
    # ======================================================

    def _transaction_id(
        self,
    ):

        if self.current_tx is None:
            return None

        return getattr(
            self.current_tx,
            "tx_id",
            self.tx_index,
        )

    # ======================================================
    # ADVANCE TRANSACTION
    # ======================================================

    def _advance_transaction(
        self,
    ):

        self.tx_index += 1

        if self.tx_index >= len(
            self.transactions
        ):

            self.current_tx = None

            self.current_paths = []

            self.current_bucket = None

            self.bucket = None

            self.backtracker.bucket = None

            return True

        # --------------------------------------------------
        # Prepare next transaction
        # --------------------------------------------------

        self.current_tx = (
            self._get_current_transaction()
        )

        self.current_paths = []

        self.current_bucket = None

        self.bucket = None

        self.backtracker.bucket = None

        self.last_failure_probability = 0.0

        self.last_average_delay = 0.0

        self.last_backtrack_count = 0

        return False

    # ======================================================
    # RENDER
    # ======================================================

    def render(self):
        return None

    # ======================================================
    # CLOSE
    # ======================================================

    def close(self):
        return None

    # ======================================================
    # INTERNAL VALIDATION HELPERS
    # ======================================================

    @staticmethod
    def _validate_finite_float(
        value,
        name,
    ):
        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be a real number, not bool."
            )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"{name} must be a real number."
            )

        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"{name} must be finite."
            )

        return value

    @classmethod
    def _validate_nonnegative_float(
        cls,
        value,
        name,
    ):
        value = cls._validate_finite_float(
            value,
            name,
        )

        if value < 0.0:
            raise ValueError(
                f"{name} must be >= 0."
            )

        return value

    @classmethod
    def _validate_positive_int(
        cls,
        value,
        name,
    ):
        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be an integer, not bool."
            )

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be > 0."
            )

        return value

    @classmethod
    def _validate_nonnegative_int(
        cls,
        value,
        name,
    ):
        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{name} must be an integer, not bool."
            )

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name} must be >= 0."
            )

        return value

    @classmethod
    def _validate_probability(
        cls,
        value,
        name,
    ):
        value = cls._validate_finite_float(
            value,
            name,
        )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{name} must be in [0, 1]."
            )

        return value

    @classmethod
    def _validate_seed(
        cls,
        value,
    ):
        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                "seed must be an integer, not bool."
            )

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "seed must be an integer."
            )

        if value < 0:
            raise ValueError(
                "seed must be >= 0."
            )

        return value

    @classmethod
    def _validate_eta(
        cls,
        eta,
    ):
        eta = cls._validate_finite_float(
            eta,
            "eta",
        )

        if eta < 0.0 or eta > 1.0:
            raise ValueError(
                "eta must be in [0, 1]."
            )

        return eta

    @staticmethod
    def _reward_float(
        reward_cfg,
        key,
        default,
    ):
        value = reward_cfg.get(
            key,
            default,
        )

        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"reward.{key} must be numeric, not bool."
            )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"reward.{key} must be numeric."
            )

        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"reward.{key} must be finite."
            )

        return value

    @staticmethod
    def _require_bool(
        result,
        key,
        context,
    ):
        if key not in result:
            raise ValueError(
                f"{context} is missing '{key}'."
            )

        value = result[key]

        if not isinstance(
            value,
            (bool, np.bool_),
        ):
            raise TypeError(
                f"{context} '{key}' must be bool."
            )

        return bool(value)

    @staticmethod
    def _result_float(
        result,
        key,
        default=0.0,
    ):
        value = result.get(
            key,
            default,
        )

        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"Payment result '{key}' must be numeric."
            )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"Payment result '{key}' must be numeric."
            )

        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"Payment result '{key}' must be finite."
            )

        if value < 0.0:
            raise ValueError(
                f"Payment result '{key}' must be >= 0."
            )

        return value

    @staticmethod
    def _candidate_failure_probability(
        candidate,
    ):
        if not isinstance(
            candidate,
            dict,
        ):
            return 0.0

        value = candidate.get(
            "failure_probability",
            0.0,
        )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                "Candidate failure_probability must be numeric."
            )

        if not math.isfinite(
            value
        ):
            raise ValueError(
                "Candidate failure_probability must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                "Candidate failure_probability must be in [0, 1]."
            )

        return value

    @staticmethod
    def _record_external_bucket_attempt(
        bucket,
    ):
        """
        Count a real PaymentSimulator retry that does not
        correspond to Bucket.current().

        PartialBacktracker may produce a new route directly,
        rather than inserting that route into Bucket.

        Therefore the environment records the actual payment
        attempt explicitly in Bucket.attempts.

        This does NOT mark payment success.
        """

        if bucket is None:
            raise RuntimeError(
                "Cannot record external attempt without a Bucket."
            )

        if not hasattr(
            bucket,
            "attempts",
        ):
            raise TypeError(
                "Bucket must expose attempts."
            )

        attempts = bucket.attempts

        if isinstance(
            attempts,
            bool,
        ):
            raise TypeError(
                "Bucket.attempts must be an integer."
            )

        if not isinstance(
            attempts,
            int,
        ):
            raise TypeError(
                "Bucket.attempts must be an integer."
            )

        if attempts < 0:
            raise ValueError(
                "Bucket.attempts cannot be negative."
            )

        bucket.attempts = attempts + 1