# RL/environment.py

import math
import os
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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

    PPO controls only eta.

    Fixed routing configuration:

        k = 5

    Complete routing pipeline:

        PPO
          |
          v
        eta
          |
          v
        Top-K
          |
          v
        CandidateManager
          |
          v
        Bucket
          |
          v
        PaymentSimulator
          |
       Failure?
        /    \
      No      Yes
      |        |
      v        v
    Success  PartialBacktrack
               |
               v
         Bucket Alternative
               |
               v
          PaymentSimulator
             /       \
          Success   Failure
             |         |
             v         v
          Finish    Full Reroute
                       |
                       v
                     Top-K
                       |
                       v
                    Bucket
                       |
                       v
                    Payment

    Important lifecycle rule
    ------------------------
    self.current_bucket and self.bucket reference the actual
    Bucket that participated in the completed step.

    At terminal state these references are intentionally kept
    alive so E2E tests and diagnostics can inspect the completed
    routing pipeline.

    Performance rule
    ----------------
    The environment supports an optional fast-validation mode
    through:

        RL_FAST_ENV=1

    This mode does NOT change the routing algorithm, Top-K value,
    eta semantics, Bucket semantics, channel identity, or
    backtracking logic.

    It only avoids unnecessary diagnostic/state work where possible.
    """

    metadata = {
        "render_modes": [],
    }

    # Fixed by project design.
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

        if (
            heuristic_fn is not None
            and not callable(heuristic_fn)
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
        # Fast validation mode
        #
        # IMPORTANT:
        # This does not alter routing semantics.
        # --------------------------------------------------

        self.fast_mode = (
            os.environ.get(
                "RL_FAST_ENV",
                "0",
            )
            .strip()
            .lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        # --------------------------------------------------
        # Graph configuration
        # --------------------------------------------------

        graph_cfg = self.cfg.get(
            "graph",
            {},
        )

        if not isinstance(graph_cfg, dict):
            raise TypeError(
                "config['graph'] must be a dictionary."
            )

        # --------------------------------------------------
        # Fixed Top-K
        # --------------------------------------------------

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

        if not isinstance(rl_cfg, dict):
            raise TypeError(
                "config['rl'] must be a dictionary."
            )

        self.eta_min = self._validate_finite_float(
            rl_cfg.get(
                "eta_min",
                -1.0,
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

        if self.eta_min < -1.0:
            raise ValueError(
                "rl.eta_min must be >= -1."
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

        if not isinstance(simulation_cfg, dict):
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

        if not isinstance(reset_balances, bool):
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

        # --------------------------------------------------
        # Partial backtracker
        # --------------------------------------------------

        self.backtracker = PartialBacktracker(
            network=self.G,
        )

        # --------------------------------------------------
        # Bucket references
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
        # Routing history
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
            shape=(observation_dimension,),
            dtype=np.float32,
        )

        # --------------------------------------------------
        # PPO controls only eta.
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
        super().reset(seed=seed)

        if seed is not None:
            self.seed = self._validate_seed(seed)

        # --------------------------------------------------
        # Reset network dynamics
        # --------------------------------------------------

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
        # Reset failure runtime state
        # --------------------------------------------------

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
            reset_rng=(seed is not None),
        )

        # --------------------------------------------------
        # Reset transaction state
        # --------------------------------------------------

        self.tx_index = int(
            self.np_random.integers(
                low=0,
                high=len(self.transactions),
            )
        )
        self.current_tx = None
        self.current_paths = []
        self.current_bucket = None

        self.bucket = None
        self.backtracker.bucket = None

        # --------------------------------------------------
        # Reset statistics
        # --------------------------------------------------

        self.last_failure_probability = 0.0
        self.last_average_delay = 0.0
        self.last_backtrack_count = 0

        self.episode_reward = 0.0
        self.episode_steps = 0

        self.total_successes = 0
        self.total_failures = 0

        # --------------------------------------------------
        # Load first transaction
        # --------------------------------------------------

        self.current_tx = self._get_current_transaction()

        if self.current_tx is None:
            raise RuntimeError(
                "No valid transaction is available after reset."
            )

        observation = self._build_state()

        info = {
            "transaction_id": self._transaction_id(),
            "eta": None,
            "top_k": self.top_k,
        }

        return observation, info

    # ======================================================
    # STEP
    # ======================================================

    def step(self, action):

        if self.current_tx is None:
            raise RuntimeError(
                "Environment has no active transaction. "
                "Call reset() before step()."
            )

        # --------------------------------------------------
        # New transaction step
        # --------------------------------------------------

        self.bucket = None
        self.current_bucket = None
        self.backtracker.bucket = None

        eta = self._decode_eta(action)

        tx = self.current_tx

        # --------------------------------------------------
        # Top-K
        # --------------------------------------------------

        candidates = self._generate_candidates(
            tx=tx,
            eta=eta,
        )

        if not candidates:
            return self._finish_failed_step(
                tx=tx,
                eta=eta,
                reason="no_initial_candidate",
            )

        # --------------------------------------------------
        # CandidateManager -> Bucket
        # --------------------------------------------------

        bucket, filtered_candidates = self._create_bucket(
            tx=tx,
            candidates=candidates,
        )

        if bucket is None:
            return self._finish_failed_step(
                tx=tx,
                eta=eta,
                reason="no_usable_bucket",
                candidate_path_count=len(candidates),
                usable_candidate_count=0,
            )

        # --------------------------------------------------
        # Register actual Bucket
        # --------------------------------------------------

        self.current_paths = list(
            filtered_candidates
        )

        self.current_bucket = bucket
        self.bucket = bucket
        self.backtracker.bucket = bucket

        # --------------------------------------------------
        # Execute complete routing pipeline
        # --------------------------------------------------

        result = self._execute_bucket_pipeline(
            tx=tx,
            eta=eta,
        )

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

        if payment_success:
            route_carbon = self._average_path_carbon(
                result.get("path", [])
            )
            if route_carbon is not None:
                final_carbon = route_carbon

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

        self.last_failure_probability = float(
            result["failure_probability"]
        )

        self.last_average_delay = final_delay
        self.last_backtrack_count = backtrack_count

        # --------------------------------------------------
        # Reward
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
            attempt_count=attempt_count,
        )

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        if payment_success:
            self.total_successes += 1
        else:
            self.total_failures += 1

        self.episode_reward += reward
        self.episode_steps += 1

        self._update_network_dynamics()

        # --------------------------------------------------
        # Info
        # --------------------------------------------------

        info = {
            "transaction_id": self._transaction_id(),
            "eta": eta,
            "top_k": self.top_k,

            "candidate_path_count": len(
                candidates
            ),

            "usable_candidate_count": len(
                filtered_candidates
            ),

            "bucket_size": self._bucket_size(
                self.current_bucket
            ),

            "success": payment_success,
            "payment_success": payment_success,

            "path": result["path"],
            "path_length": final_path_length,

            "fee": final_fee,
            "delay": final_delay,
            "carbon": final_carbon,

            "failure_probability": (
                self.last_failure_probability
            ),

            "average_delay": (
                self.last_average_delay
            ),

            "backtrack_count": backtrack_count,

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

            "attempt_count": attempt_count,

            "reason": result["reason"],

            "episode_reward": (
                self.episode_reward
            ),
        }

        # --------------------------------------------------
        # Advance transaction
        #
        # Terminal state preserves the actual Bucket.
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
    # FAILED STEP
    # ======================================================

    def _finish_failed_step(
        self,
        tx,
        eta,
        reason,
        candidate_path_count=0,
        usable_candidate_count=0,
    ):
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

            "candidate_path_count": (
                candidate_path_count
            ),

            "usable_candidate_count": (
                usable_candidate_count
            ),

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
            "reason": reason,

            "episode_reward": (
                self.episode_reward
            ),
        }

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
    # GENERATE TOP-K
    # ======================================================

    def _generate_candidates(
        self,
        tx,
        eta,
    ):
        if tx is None:
            raise ValueError(
                "Transaction cannot be None."
            )

        eta = self._validate_eta(eta)

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

        if not isinstance(
            bucket.candidates,
            list,
        ):
            raise TypeError(
                "Bucket.candidates must be a list."
            )

        if len(bucket.candidates) == 0:
            raise RuntimeError(
                "CandidateManager returned an empty Bucket."
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
        current_bucket = self.current_bucket

        if current_bucket is None:
            raise RuntimeError(
                "No active Bucket exists."
            )

        self.backtracker.bucket = current_bucket

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
        # Safety bound.
        #
        # This is intentionally finite to prevent a malformed
        # Bucket/reroute implementation from creating an
        # infinite simulation loop.
        # --------------------------------------------------

        max_attempts = (
            self.top_k * 3
        ) + 5

        while attempt_count < max_attempts:

            candidate = current_bucket.current()

            # ------------------------------------------------
            # Bucket exhausted
            # ------------------------------------------------

            if candidate is None:

                reroute_bucket, _ = self._full_reroute(
                    tx=tx,
                    eta=eta,
                    failed_routes=failed_routes,
                    failed_edges=failed_edges,
                )

                if reroute_bucket is None:

                    final_reason = (
                        "bucket_exhausted_no_reroute"
                    )

                    break

                full_reroute_count += 1

                current_bucket = reroute_bucket

                self.current_bucket = current_bucket
                self.bucket = current_bucket
                self.backtracker.bucket = current_bucket

                continue

            # ------------------------------------------------
            # Resolve route
            # ------------------------------------------------

            path, edges = self._candidate_route(
                candidate
            )

            if not path:

                current_bucket.backtrack(
                    failed_candidate=candidate,
                    reason="invalid_candidate_path",
                )

                continue

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
                    reason=(
                        "cannot_resolve_exact_route_edges"
                    ),
                )

                continue

            self._validate_route_edges(
                path,
                edges,
            )

            # ------------------------------------------------
            # Select candidate
            # ------------------------------------------------

            selected = current_bucket.select_candidate(
                candidate
            )

            if selected is None:
                raise RuntimeError(
                    "Bucket could not select its current candidate."
                )

            # ------------------------------------------------
            # Actual payment attempt
            # ------------------------------------------------

            current_bucket.record_attempt()

            attempt_count += 1

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

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if success:

                if isinstance(candidate, dict):
                    candidate["success"] = True

                if not current_bucket.mark_success(
                    candidate
                ):
                    raise RuntimeError(
                        "Bucket.mark_success() failed."
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
                        if isinstance(candidate, dict)
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
                        if isinstance(candidate, dict)
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

            # ------------------------------------------------
            # FAILURE
            # ------------------------------------------------

            if isinstance(candidate, dict):
                candidate["success"] = False

            failed_routes.add(
                tuple(path)
            )

            failed_edge = self._resolve_failed_edge(
                result=result,
                edges=edges,
            )

            failure_index = self._resolve_failure_index(
                result=result,
                edges=edges,
                failed_edge=failed_edge,
            )

            if (
                failure_index is not None
                and 0 <= failure_index < len(edges)
            ):
                failed_edge = edges[
                    failure_index
                ]

            normalized_failed_edge = (
                self._normalize_edge(
                    failed_edge
                )
            )

            if normalized_failed_edge is not None:
                failed_edges.add(
                    normalized_failed_edge
                )

            # ------------------------------------------------
            # One real failure => one partial backtracking
            # event.
            # ------------------------------------------------

            partial_backtrack_count += 1

            # ------------------------------------------------
            # Partial backtracking
            # ------------------------------------------------

            backtrack_result = (
                self.backtracker.backtrack(
                    route=path,
                    route_edges=edges,
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

            # ------------------------------------------------
            # Alternative found
            # ------------------------------------------------

            if status == "alternative_found":

                new_path = backtrack_result.get(
                    "new_route"
                )

                new_edges = backtrack_result.get(
                    "new_edges"
                )

                if new_path and new_edges:

                    self._validate_route_edges(
                        new_path,
                        new_edges,
                    )

                    retry_edges = (
                        self._restore_retry_edges_from_bucket(
                            route=new_path,
                            edges=new_edges,
                        )
                    )

                    self._validate_route_edges(
                        new_path,
                        retry_edges,
                    )

                    current_bucket = self.current_bucket

                    self._record_external_bucket_attempt(
                        current_bucket
                    )

                    attempt_count += 1

                    retry_result = (
                        self.payment_simulator.simulate_payment(
                            path=list(new_path),
                            edges=list(retry_edges),
                            amount=tx.amount,
                            tx_id=tx.tx_id,
                        )
                    )

                    retry_dict = self._result_to_dict(
                        retry_result
                    )

                    retry_success = (
                        self._require_bool(
                            retry_dict,
                            "success",
                            "PaymentSimulator retry result",
                        )
                    )

                    if retry_success:

                        partial_backtrack_success += 1

                        final_success = True
                        final_path = list(new_path)

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

                        # IMPORTANT:
                        # A successful Bucket alternative must
                        # terminate the current payment attempt.
                        break

                    # ------------------------------------------------
                    # Alternative failed.
                    # ------------------------------------------------

                    retry_failed_edge = (
                        self._resolve_failed_edge(
                            result=retry_dict,
                            edges=retry_edges,
                        )
                    )

                    retry_failure_index = (
                        self._resolve_failure_index(
                            result=retry_dict,
                            edges=retry_edges,
                            failed_edge=retry_failed_edge,
                        )
                    )

                    if (
                        retry_failure_index is not None
                        and
                        0 <= retry_failure_index
                        < len(retry_edges)
                    ):
                        retry_failed_edge = (
                            retry_edges[
                                retry_failure_index
                            ]
                        )

                    normalized_retry_failed_edge = (
                        self._normalize_edge(
                            retry_failed_edge
                        )
                    )

                    if normalized_retry_failed_edge is not None:
                        failed_edges.add(
                            normalized_retry_failed_edge
                        )

                    failed_routes.add(
                        tuple(new_path)
                    )

                # Continue normal Bucket failure handling.

            # ------------------------------------------------
            # Record failed current candidate
            # ------------------------------------------------

            current_bucket.backtrack(
                failed_candidate=candidate,
                failed_channel=failed_edge,
                reason=result.get(
                    "reason",
                    "payment_failed",
                ),
            )

            # ------------------------------------------------
            # Continue inside Bucket first.
            #
            # Full reroute is only used after the active
            # Bucket is exhausted.
            # ------------------------------------------------

            if current_bucket.current() is not None:
                continue

            # ------------------------------------------------
            # Full reroute
            # ------------------------------------------------

            if attempt_count >= max_attempts:

                final_reason = (
                    "maximum_payment_attempts_reached"
                )

                break

            reroute_bucket, _ = self._full_reroute(
                tx=tx,
                eta=eta,
                failed_routes=failed_routes,
                failed_edges=failed_edges,
            )

            if reroute_bucket is None:

                final_reason = (
                    "full_reroute_failed"
                )

                break

            full_reroute_count += 1

            current_bucket = reroute_bucket

            self.current_bucket = current_bucket
            self.bucket = current_bucket
            self.backtracker.bucket = current_bucket

        # --------------------------------------------------
        # Final failure
        # --------------------------------------------------

        if not final_success:

            final_reason = (
                final_reason
                or "payment_failed"
            )

            final_failure_probability = 1.0

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
            "failure_probability": float(
                final_failure_probability
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
    # RETRY EDGE REPRESENTATION
    # ======================================================

    def _restore_retry_edges_from_bucket(
        self,
        route,
        edges,
    ):
        """
        Restore the exact physical-edge representation used
        by Top-K / Bucket for a PartialBacktracker retry.

        Physical identity:

            (source, target, channel_key)

        Rich representation:

            {
                "source": ...,
                "target": ...,
                "channel_key": ...,
                "scid": ...,
                "data": ...
            }
        """

        if not isinstance(
            route,
            (list, tuple),
        ):
            raise TypeError(
                "Retry route must be a list or tuple."
            )

        if not isinstance(
            edges,
            (list, tuple),
        ):
            raise TypeError(
                "Retry edges must be a list or tuple."
            )

        route = list(route)
        edges = list(edges)

        if len(route) < 2:
            raise ValueError(
                "Retry route must contain at least two nodes."
            )

        if len(edges) != len(route) - 1:
            raise ValueError(
                "Retry route and edge counts do not match."
            )

        canonical_edges = []

        for index, edge in enumerate(edges):

            normalized = self._normalize_edge(
                edge
            )

            if normalized is None:
                raise ValueError(
                    "Retry contains an invalid physical edge."
                )

            source, target, channel_key = normalized

            if (
                source != route[index]
                or target != route[index + 1]
            ):
                raise ValueError(
                    "Retry edge does not match retry route."
                )

            canonical_edges.append(
                normalized
            )

        # --------------------------------------------------
        # Preferred source: active Bucket.
        # --------------------------------------------------

        bucket = self.current_bucket

        if bucket is not None:

            bucket_candidates = getattr(
                bucket,
                "candidates",
                None,
            )

            if isinstance(
                bucket_candidates,
                list,
            ):

                for candidate in bucket_candidates:

                    if not isinstance(
                        candidate,
                        dict,
                    ):
                        continue

                    candidate_path = candidate.get(
                        "path"
                    )

                    if not isinstance(
                        candidate_path,
                        (list, tuple),
                    ):
                        continue

                    candidate_path = list(
                        candidate_path
                    )

                    if candidate_path != route:
                        continue

                    candidate_edges = candidate.get(
                        "edges"
                    )

                    if not isinstance(
                        candidate_edges,
                        (list, tuple),
                    ):
                        continue

                    candidate_edges = list(
                        candidate_edges
                    )

                    if len(candidate_edges) != len(
                        canonical_edges
                    ):
                        continue

                    exact_match = True

                    for (
                        candidate_edge,
                        canonical_edge,
                    ) in zip(
                        candidate_edges,
                        canonical_edges,
                    ):

                        normalized_candidate = (
                            self._normalize_edge(
                                candidate_edge
                            )
                        )

                        if normalized_candidate != (
                            canonical_edge
                        ):
                            exact_match = False
                            break

                    if not exact_match:
                        continue

                    restored = []

                    for candidate_edge in candidate_edges:

                        if isinstance(
                            candidate_edge,
                            dict,
                        ):
                            restored.append(
                                dict(candidate_edge)
                            )
                        else:
                            restored.append(
                                candidate_edge
                            )

                    return restored

        # --------------------------------------------------
        # Graph reconstruction fallback.
        # --------------------------------------------------

        restored = []

        for canonical_edge in canonical_edges:

            source, target, channel_key = (
                canonical_edge
            )

            if self.G.is_multigraph():

                if channel_key is None:
                    raise ValueError(
                        "MultiGraph retry requires an exact "
                        "channel key."
                    )

                edge_data = self.G.get_edge_data(
                    source,
                    target,
                    channel_key,
                )

                if edge_data is None:
                    raise ValueError(
                        "Exact retry channel does not exist: "
                        f"({source}, {target}, "
                        f"{channel_key})."
                    )

                if not isinstance(
                    edge_data,
                    dict,
                ):
                    raise TypeError(
                        "MultiGraph edge data must be a dictionary."
                    )

                restored_edge = {
                    "source": source,
                    "target": target,
                    "channel_key": channel_key,
                    "data": dict(edge_data),
                }

                if "scid" in edge_data:
                    restored_edge["scid"] = (
                        edge_data["scid"]
                    )

                restored.append(
                    restored_edge
                )

            else:

                if channel_key is not None:
                    raise ValueError(
                        "Simple graph retry must not contain "
                        "a channel key."
                    )

                edge_data = self.G.get_edge_data(
                    source,
                    target,
                )

                if edge_data is None:
                    raise ValueError(
                        f"Retry edge ({source}, {target}) "
                        "does not exist."
                    )

                if not isinstance(
                    edge_data,
                    dict,
                ):
                    raise TypeError(
                        "Edge data must be a dictionary."
                    )

                restored.append(
                    (
                        source,
                        target,
                    )
                )

        return restored

    # ======================================================
    # FAILURE INFORMATION
    # ======================================================

    def _resolve_failed_edge(
        self,
        result,
        edges,
    ):
        """
        Resolve exact physical failed edge.

        Priority:

            failed_edge
            failed_channel
            failed_physical_edge
            edge
            channel
            failure_index
        """

        failed_edge = None

        for key in (
            "failed_edge",
            "failed_channel",
            "failed_physical_edge",
            "edge",
            "channel",
        ):

            value = result.get(key)

            if value is not None:
                failed_edge = value
                break

        if failed_edge is not None:
            return failed_edge

        failure_index = result.get(
            "failure_index"
        )

        if failure_index is None:
            failure_index = result.get(
                "failed_edge_index"
            )

        if failure_index is None:
            failure_index = result.get(
                "failure_position"
            )

        failure_index = self._normalize_failure_index(
            failure_index
        )

        if (
            failure_index is not None
            and
            0 <= failure_index < len(edges)
        ):
            return edges[
                failure_index
            ]

        return None

    # ======================================================
    # FAILURE INDEX
    # ======================================================

    def _resolve_failure_index(
        self,
        result,
        edges,
        failed_edge,
    ):
        failure_index = result.get(
            "failure_index"
        )

        if failure_index is None:
            failure_index = result.get(
                "failed_edge_index"
            )

        if failure_index is None:
            failure_index = result.get(
                "failure_position"
            )

        failure_index = self._normalize_failure_index(
            failure_index
        )

        if failure_index is not None:

            if (
                0 <= failure_index < len(edges)
            ):
                return failure_index

        if failed_edge is None:
            return None

        normalized_failed = self._normalize_edge(
            failed_edge
        )

        if normalized_failed is None:
            return None

        fu, fv, fk = normalized_failed

        for index, edge in enumerate(edges):

            normalized_edge = self._normalize_edge(
                edge
            )

            if normalized_edge is None:
                continue

            eu, ev, ek = normalized_edge

            if eu != fu or ev != fv:
                continue

            if self.G.is_multigraph():

                if fk is None or ek == fk:
                    return index

            else:

                return index

        return None

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

            if tuple(path) in failed_routes:
                continue

            candidate_edges = candidate.get(
                "edges"
            )

            if not isinstance(
                candidate_edges,
                (list, tuple),
            ):
                raise ValueError(
                    "Top-K candidate is missing exact edges."
                )

            if len(candidate_edges) != len(path) - 1:
                raise ValueError(
                    "Candidate path/edge count mismatch."
                )

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

            filtered.append(
                candidate
            )

        if not filtered:
            return None, []

        return self._create_bucket(
            tx=tx,
            candidates=filtered,
        )

    # ======================================================
    # CANDIDATE ROUTE
    # ======================================================

    def _candidate_route(
        self,
        candidate,
    ):
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

            return (
                list(path),
                list(edges),
            )

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
    def _normalize_edge(edge):

        if edge is None:
            return None

        if isinstance(
            edge,
            dict,
        ):

            source = edge.get(
                "source"
            )

            if source is None:
                source = edge.get(
                    "u"
                )

            target = edge.get(
                "target"
            )

            if target is None:
                target = edge.get(
                    "v"
                )

            if source is None or target is None:
                return None

            key = edge.get(
                "channel_key"
            )

            if key is None:
                key = edge.get(
                    "key"
                )

            if key is None:
                return (
                    source,
                    target,
                    None,
                )

            return (
                source,
                target,
                key,
            )

        if not isinstance(
            edge,
            (tuple, list),
        ):
            return None

        if len(edge) >= 3:

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
    # EDGE MATCHING
    # ======================================================

    def _edge_matches_failure(
        self,
        candidate_edge,
        failed_edge,
    ):
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
    # PATH TO EXACT EDGES
    # ======================================================

    def _path_to_edges(
        self,
        path,
    ):
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

            # --------------------------------------------------
            # MultiGraph / MultiDiGraph
            #
            # NEVER choose an arbitrary parallel channel.
            # --------------------------------------------------

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

                # If multiple channels are available,
                # physical channel identity is ambiguous.
                if len(available_channels) != 1:
                    return None

                edges.append(
                    (
                        u,
                        v,
                        available_channels[0],
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
    # ROUTE VALIDATION
    # ======================================================

    def _validate_route_edges(
        self,
        path,
        edges,
    ):
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
                        "MultiGraph route requires exact channel key."
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
                        "Simple graph route must not contain "
                        "a channel key."
                    )

                if not self.G.has_edge(
                    u,
                    v,
                ):
                    raise ValueError(
                        f"Edge ({u}, {v}) does not exist."
                    )

    # ======================================================
    # RESULT NORMALIZATION
    # ======================================================

    @staticmethod
    def _result_to_dict(
        result,
    ):

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
                "a dictionary or object with to_dict()."
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
    # ETA
    # ======================================================

    def _decode_eta(
        self,
        action,
    ):
        array = np.asarray(
            action,
            dtype=np.float32,
        )

        if array.shape != (1,):
            raise ValueError(
                "PPO action must have exactly shape (1). "
                f"Received shape: {array.shape}"
            )

        eta = float(
            array[0]
        )

        if not np.isfinite(eta):
            raise ValueError(
                "PPO produced a non-finite eta."
            )

        if (
            eta < self.eta_min
            or
            eta > self.eta_max
        ):
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
        reward = calculate_reward(
            success=bool(success),
            path_length=int(path_length),
            carbon_intensity=float(
                carbon_intensity
            ),
            fee=float(fee),
            delay=float(delay),
            partial_backtrack_count=int(
                partial_backtrack_count
            ),
            full_reroute_count=int(
                full_reroute_count
            ),
            attempt_count=int(
                attempt_count
            ),
        )

        reward = float(
            reward
        )

        if not math.isfinite(reward):
            raise RuntimeError(
                "calculate_reward() returned a non-finite value."
            )

        return reward

    def _average_path_carbon(self, path):
        """Average dataset-derived carbon intensity over route nodes."""

        if not path:
            return None

        values = []
        for node in path:
            attributes = self.G.nodes[node]
            value = attributes.get("carbon_intensity")
            if value is None:
                # Older/synthetic graphs can lack the dataset field. The
                # payment simulator's route estimate remains a fallback.
                return None
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Invalid carbon intensity for route node {node!r}: {value!r}."
                )
            values.append(value)

        if not values:
            return None
        return float(sum(values) / len(values))

    # ======================================================
    # NETWORK DYNAMICS
    # ======================================================

    def _update_network_dynamics(self):

        update_method = getattr(
            self.network_dynamics,
            "update",
            None,
        )

        if not callable(update_method):
            raise RuntimeError(
                "NetworkDynamics must provide update()."
            )

        update_method()

    # ======================================================
    # STATE
    # ======================================================

    def _build_state(self):

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

        if vector.shape != self.observation_space.shape:
            raise RuntimeError(
                "Invalid observation shape.\n"
                f"Expected: {self.observation_space.shape}\n"
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

        if bucket is None:
            return 0

        candidates = getattr(
            bucket,
            "candidates",
            None,
        )

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

        info_method = getattr(
            bucket,
            "info",
            None,
        )

        if not callable(info_method):
            raise TypeError(
                "Bucket must provide info()."
            )

        info = info_method()

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
    # TRANSACTION
    # ======================================================

    def _get_current_transaction(self):

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
                    f"Transaction is missing required field "
                    f"'{field}'."
                )

        amount = float(
            transaction.amount
        )

        if (
            not math.isfinite(amount)
            or amount <= 0
        ):
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

    def _transaction_id(self):

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

    def _advance_transaction(self):
        # The reference implementation treats one sampled payment as one
        # episode. This prevents PPO returns from coupling unrelated
        # transactions in the supplied batch.
        self.current_tx = None
        self.current_paths = []
        return True

    # ======================================================
    # RENDER / CLOSE
    # ======================================================

    def render(self):
        return None

    def close(self):
        return None

    # ======================================================
    # VALIDATION HELPERS
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
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"{name} must be a real number."
            )

        if not math.isfinite(value):
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

        if eta < -1.0 or eta > 1.0:
            raise ValueError(
                "eta must be in [-1, 1]."
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
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"reward.{key} must be numeric."
            )

        if not math.isfinite(value):
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
            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            raise TypeError(
                f"Payment result '{key}' must be numeric."
            )

        if not math.isfinite(value):
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

        if not math.isfinite(value):
            raise ValueError(
                "Candidate failure_probability must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                "Candidate failure_probability must be in [0, 1]."
            )

        return value

    # ======================================================
    # EXTERNAL BUCKET ATTEMPT
    # ======================================================

    @staticmethod
    def _record_external_bucket_attempt(
        bucket,
    ):
        """
        Count a PaymentSimulator retry created by
        PartialBacktracker.

        PartialBacktracker itself deliberately does not
        modify Bucket.attempts.
        """

        if bucket is None:
            raise RuntimeError(
                "Cannot record external attempt without Bucket."
            )

        attempts = getattr(
            bucket,
            "attempts",
            None,
        )

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

    # ======================================================
    # FAILURE INDEX NORMALIZATION
    # ======================================================

    @staticmethod
    def _normalize_failure_index(
        value,
    ):
        """
        Normalize a failure index into a valid non-negative
        integer.

        Accepted:

            int
            integer-valued float
            numeric string
            None
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
            return None

        try:

            if isinstance(
                value,
                int,
            ):
                return (
                    value
                    if value >= 0
                    else None
                )

            if isinstance(
                value,
                float,
            ):

                if not value.is_integer():
                    return None

                value = int(
                    value
                )

                return (
                    value
                    if value >= 0
                    else None
                )

            if isinstance(
                value,
                str,
            ):

                value = value.strip()

                if not value:
                    return None

                if value.isdigit():
                    return int(
                        value
                    )

                parsed = float(
                    value
                )

                if not parsed.is_integer():
                    return None

                parsed = int(
                    parsed
                )

                return (
                    parsed
                    if parsed >= 0
                    else None
                )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        return None


# ============================================================
# End of RoutingEnv
# ============================================================
