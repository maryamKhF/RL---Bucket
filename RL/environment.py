# RL/environment.py

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
    """


    metadata = {
        "render_modes": [],
    }


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

        if transactions is None:
            raise ValueError(
                "RoutingEnv requires transactions."
            )

        if len(transactions) == 0:
            raise ValueError(
                "RoutingEnv received an empty transaction set."
            )


        # --------------------------------------------------
        # Store inputs
        # --------------------------------------------------

        self.G = G
        self.transactions = transactions
        self.heuristic_fn = heuristic_fn
        self.mode = mode

        self.cfg = (
            config
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

        # IMPORTANT:
        # k is ALWAYS fixed to 5.
        self.top_k = 5

        self.max_hops = int(
            graph_cfg.get(
                "max_hops",
                12,
            )
        )

        self.lambda_h = float(
            graph_cfg.get(
                "lambda_h",
                1.0,
            )
        )


        # --------------------------------------------------
        # RL configuration
        # --------------------------------------------------

        rl_cfg = self.cfg.get(
            "rl",
            {},
        )

        self.eta_min = float(
            rl_cfg.get(
                "eta_min",
                0.0,
            )
        )

        self.eta_max = float(
            rl_cfg.get(
                "eta_max",
                1.0,
            )
        )

        if self.eta_min > self.eta_max:
            raise ValueError(
                "eta_min cannot be greater than eta_max."
            )


        # --------------------------------------------------
        # State configuration
        # --------------------------------------------------

        self.state_k = int(
            graph_cfg.get(
                "neighborhood_k",
                15,
            )
        )

        self.state_radius = int(
            graph_cfg.get(
                "neighborhood_m",
                5,
            )
        )


        # --------------------------------------------------
        # Simulation configuration
        # --------------------------------------------------

        simulation_cfg = self.cfg.get(
            "simulation",
            {},
        )

        self.seed = int(
            simulation_cfg.get(
                "seed",
                42,
            )
        )

        self.node_failure_probability = float(
            simulation_cfg.get(
                "node_failure_probability",
                0.01,
            )
        )

        self.liquidity_failure_probability = float(
            simulation_cfg.get(
                "liquidity_failure_probability",
                0.05,
            )
        )

        self.channel_failure_rate = float(
            simulation_cfg.get(
                "channel_failure_rate",
                0.01,
            )
        )

        self.node_failure_rate = float(
            simulation_cfg.get(
                "node_failure_rate",
                0.005,
            )
        )

        self.recovery_rate = float(
            simulation_cfg.get(
                "recovery_rate",
                0.05,
            )
        )

        # Network reset behavior.
        #
        # False:
        #   balances can evolve across PPO episodes.
        #
        # True:
        #   balances are restored on every reset.
        #
        # Default is False because training should expose PPO
        # to changing network conditions.
        self.reset_balances_on_episode_reset = bool(
            simulation_cfg.get(
                "reset_balances_on_episode_reset",
                False,
            )
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

        Important:
        - NetworkDynamics is reset.
        - FailureModel runtime state is reset when the
          revised FailureModel implementation is available.
        - Failure RNG is not forcibly reseeded unless a
          new seed is explicitly supplied.
        """

        super().reset(
            seed=seed
        )


        # --------------------------------------------------
        # Seed handling
        # --------------------------------------------------

        if seed is not None:
            self.seed = int(seed)


        # --------------------------------------------------
        # Reset NetworkDynamics
        # --------------------------------------------------

        if self.network_dynamics is not None:

            reset_method = getattr(
                self.network_dynamics,
                "reset",
                None,
            )

            if callable(reset_method):

                reset_method(
                    reset_balances=(
                        self.reset_balances_on_episode_reset
                    )
                )


        # --------------------------------------------------
        # Reset FailureModel runtime state
        # --------------------------------------------------

        reset_failure_state = getattr(
            self.failure_model,
            "reset_runtime_state",
            None,
        )

        if callable(reset_failure_state):

            reset_failure_state(
                self.G,
                reset_counters=False,
                reset_rng=(
                    seed is not None
                ),
            )

        else:

            # Compatibility with the older FailureModel.
            #
            # The older implementation does not expose a
            # reset_runtime_state() method, so we reset the
            # runtime graph attributes directly.

            self._reset_failure_graph_state()


        # --------------------------------------------------
        # Reset transaction index
        # --------------------------------------------------

        self.tx_index = 0

        self.current_tx = None

        self.current_paths = []

        self.current_bucket = None

        self.bucket = None


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
        Compatibility reset for the older FailureModel.

        The preferred implementation is
        FailureModel.reset_runtime_state().
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


            # Network evolves even when no route is available.
            self._update_network_dynamics()


            info = {
                "transaction_id": self._transaction_id(),
                "success": False,
                "eta": eta,
                "top_k": self.top_k,
                "candidate_path_count": 0,
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

        self.last_failure_probability = (
            0.0
            if payment_success
            else 1.0
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

        # Apply network evolution after this transaction.
        #
        # This means the next transaction observes a network
        # that may have changed because of failures/recovery.
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
            return []

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
            candidates=candidates,
            max_candidates=self.top_k,
        )


        filtered = manager.filter_candidates(
            self.G,
            tx.amount,
        )


        if not filtered:
            return None, []


        manager.rank_candidates()


        bucket = manager.create_bucket(
            tx_id=tx.tx_id,
            k=self.top_k,
        )


        if bucket is None:
            return None, []


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
           Failure
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
        """

        current_bucket = (
            self.current_bucket
        )

        self.backtracker.bucket = (
            current_bucket
        )


        attempt_count = 0

        partial_backtrack_count = 0

        partial_backtrack_success = 0

        full_reroute_count = 0


        # Full paths that have already failed.
        failed_routes = set()


        # Exact channels that have already failed.
        failed_edges = set()


        final_success = False

        final_path = []

        final_fee = 0.0

        final_delay = 0.0

        final_carbon = 0.0

        final_reason = None


        # --------------------------------------------------
        # Maximum attempts
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
            # Preserve exact Top-K edges whenever possible.
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
                    failed_candidate=candidate
                )

                continue


            # --------------------------------------------------
            # Attempt counter
            # --------------------------------------------------

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


            success = bool(
                result.get(
                    "success",
                    False,
                )
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


                current_bucket.mark_success(
                    candidate
                )


                final_success = True

                final_path = list(path)


                final_fee = float(
                    result.get(
                        "fee",
                        (
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
                )


                final_delay = float(
                    result.get(
                        "delay",
                        (
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
                )


                final_carbon = float(
                    result.get(
                        "carbon",
                        0.0,
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
                    failed_candidate=candidate
                )

                final_reason = result.get(
                    "reason",
                    "payment_failed",
                )

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

                    # If Backtracker provides exact edges,
                    # preserve them.
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


                    if new_edges is not None:

                        partial_backtrack_success += 1


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


                        retry_success = bool(
                            retry_dict.get(
                                "success",
                                False,
                            )
                        )


                        if retry_success:

                            final_success = True

                            final_path = list(
                                new_path
                            )


                            final_fee = float(
                                retry_dict.get(
                                    "fee",
                                    0.0,
                                )
                            )


                            final_delay = float(
                                retry_dict.get(
                                    "delay",
                                    0.0,
                                )
                            )


                            final_carbon = float(
                                retry_dict.get(
                                    "carbon",
                                    0.0,
                                )
                            )


                            final_reason = (
                                "partial_backtrack_success"
                            )

                            break


                        # If the retry itself exposes another
                        # failed channel, retain it for possible
                        # full rerouting.
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


            # =================================================
            # PARTIAL BACKTRACK FAILED
            # =================================================

            current_bucket.backtrack(
                failed_candidate=candidate
            )


            # --------------------------------------------------
            # Bucket still contains another candidate
            # --------------------------------------------------

            if current_bucket.current() is not None:

                continue


            # =================================================
            # FULL REROUTE
            # =================================================

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


        # IMPORTANT:
        # path length = number of forwarding edges,
        # not number of nodes.
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
            2. it reuses an exact failed channel.
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
                continue


            path = tuple(
                candidate.get(
                    "path",
                    [],
                )
            )


            # ----------------------------------------------
            # Reject already failed complete route
            # ----------------------------------------------

            if path in failed_routes:
                continue


            # ----------------------------------------------
            # Extract exact candidate channels
            # ----------------------------------------------

            candidate_edges = (
                candidate.get(
                    "edges",
                    [],
                )
            )


            normalized_candidate_edges = set()


            if candidate_edges:

                for edge in candidate_edges:

                    normalized = (
                        self._normalize_edge(
                            edge
                        )
                    )

                    if normalized is not None:

                        normalized_candidate_edges.add(
                            normalized
                        )


            # ----------------------------------------------
            # If no exact edges are supplied, resolve them.
            # ----------------------------------------------

            if not normalized_candidate_edges:

                resolved_edges = (
                    self._path_to_edges(
                        path
                    )
                )

                if resolved_edges is not None:

                    for edge in resolved_edges:

                        normalized = (
                            self._normalize_edge(
                                edge
                            )
                        )

                        if normalized is not None:

                            normalized_candidate_edges.add(
                                normalized
                            )


            # ----------------------------------------------
            # Reject candidate reusing failed channel
            # ----------------------------------------------

            if (
                normalized_candidate_edges
                &
                failed_edges
            ):
                continue


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

            return path, edges


        # --------------------------------------------------
        # Legacy tuple format
        # --------------------------------------------------

        try:

            if len(candidate) >= 2:

                return (
                    candidate[0],
                    candidate[1],
                )

        except (
            TypeError,
            IndexError,
        ):

            pass


        return None, None


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
        """

        if edge is None:
            return None


        try:

            values = tuple(edge)

        except TypeError:

            return None


        if len(values) >= 3:

            return (
                values[0],
                values[1],
                values[2],
            )


        if len(values) == 2:

            return (
                values[0],
                values[1],
                None,
            )


        return None


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

        For MultiDiGraph, an exact available channel key is
        retained.
        """

        if not path:
            return None

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

                selected_key = None


                for key, data in edge_data.items():

                    if data.get(
                        "available",
                        True,
                    ):

                        selected_key = key

                        break


                if selected_key is None:
                    return None


                edges.append(
                    (
                        u,
                        v,
                        selected_key,
                    )
                )

            else:

                if not edge_data.get(
                    "available",
                    True,
                ):

                    return None


                edges.append(
                    (
                        u,
                        v,
                    )
                )


        return edges


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

        PPO action space:

            [eta_min, eta_max]

        Current default:

            [0, 1]
        """

        array = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)


        if array.size == 0:

            raise ValueError(
                "PPO action is empty."
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


        eta = float(
            np.clip(
                eta,
                self.eta_min,
                self.eta_max,
            )
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
        Calculate reward.

        Preferred reward.py interface includes:

            partial_backtrack_count
            full_reroute_count
            attempt_count
            fee_reference
            delay_reference
            carbon_reference
            path_reference

        A compatibility fallback is retained for the older
        reward.py implementation.
        """

        reward_cfg = self.cfg.get(
            "reward",
            {},
        )


        kwargs = {
            "success": success,
            "path_length": path_length,
            "carbon_intensity": carbon_intensity,
            "fee": fee,
            "delay": delay,
            "partial_backtrack_count": (
                partial_backtrack_count
            ),
            "full_reroute_count": (
                full_reroute_count
            ),
            "attempt_count": (
                attempt_count
            ),
            "scale": float(
                reward_cfg.get(
                    "scale",
                    100.0,
                )
            ),
            "fee_reference": float(
                reward_cfg.get(
                    "fee_reference",
                    1000.0,
                )
            ),
            "delay_reference": float(
                reward_cfg.get(
                    "delay_reference",
                    10.0,
                )
            ),
            "carbon_reference": float(
                reward_cfg.get(
                    "carbon_reference",
                    100.0,
                )
            ),
            "path_reference": float(
                reward_cfg.get(
                    "path_reference",
                    10.0,
                )
            ),
        }


        # --------------------------------------------------
        # Preferred new reward API
        # --------------------------------------------------

        try:

            return float(
                calculate_reward(
                    **kwargs
                )
            )

        except TypeError as exc:

            # ------------------------------------------------
            # Compatibility with the old reward.py signature.
            #
            # This fallback can be removed after reward.py is
            # updated to the new interface.
            # ------------------------------------------------

            message = str(exc)

            extended_parameters = (
                "partial_backtrack_count",
                "full_reroute_count",
                "attempt_count",
                "fee_reference",
                "delay_reference",
                "carbon_reference",
                "path_reference",
            )


            if not any(
                parameter in message
                for parameter in extended_parameters
            ):
                raise


            return float(
                calculate_reward(
                    success=success,
                    path_length=path_length,
                    carbon_intensity=carbon_intensity,
                    fee=fee,
                    delay=delay,
                    scale=kwargs["scale"],
                )
            )


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
            return


        update_method = getattr(
            self.network_dynamics,
            "update",
            None,
        )


        if callable(update_method):

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
        if bucket is None:
            return 0


        try:

            return int(
                bucket.size()
            )

        except Exception:

            try:

                return len(
                    bucket.candidates
                )

            except Exception:

                return 0


    @classmethod
    def _bucket_info(
        cls,
        bucket,
    ):
        if bucket is None:
            return {}


        info = {
            "size": cls._bucket_size(
                bucket
            ),
        }


        try:

            info.update(
                bucket.info()
            )

        except Exception:

            pass


        return info


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


        return float(
            reward_cfg.get(
                "scale",
                100.0,
            )
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


        return self.transactions[
            self.tx_index
        ]


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