# RL/environment.py

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from Pathfinding.dijkstra import Dijkstra
from Pathfinding.top_k_paths import top_k_paths

from Simulation.payment_simulator import simulate_payment

from .state import State
from .reward import calculate_reward


class RoutingEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        G,
        transactions,
        heuristic_fn,
        config,
        mode="hybrid",
        bucket=None
    ):

        super().__init__()

        # =================================================
        # Basic configuration
        # =================================================

        self.G = G
        self.transactions = transactions
        self.heuristic = heuristic_fn
        self.cfg = config
        self.mode = mode
        self.bucket = bucket

        # =================================================
        # Router
        # =================================================

        self.router = Dijkstra(G)

        # =================================================
        # Transaction state
        # =================================================

        self.tx_index = 0
        self.current_tx = None
        self.current_paths = []

        # Information generated during previous
        # routing/payment attempts.
        self.last_failure_probability = 0.0
        self.last_average_delay = 0.0
        self.last_backtrack_count = 0

        # =================================================
        # State configuration
        #
        # State.__init__ uses:
        #
        #     k
        #     radius
        #
        # There is NO "m" argument.
        #
        # neighborhood_m is therefore interpreted as
        # the Ego-Graph observation radius.
        # =================================================

        state_k = int(
            self.cfg["graph"]["neighborhood_k"]
        )

        state_radius = int(
            self.cfg["graph"]["neighborhood_m"]
        )

        # =================================================
        # State builder
        #
        # Only used to obtain the observation dimension.
        # The real transaction-specific state is created
        # in _build_state().
        # =================================================

        self.state_builder = State(

            self.G,

            source=None,

            destination=None,

            transaction=None,

            candidate_paths=[],

            simulation_info={
                "failure_probability": 0.0,
                "average_delay": 0.0
            },

            bucket_info={
                "bucket_size": 0,
                "backtrack_count": 0
            },

            k=state_k,

            radius=state_radius
        )

        # =================================================
        # Observation space
        #
        # State dimension:
        #
        # K * 11 + K
        #
        # For K = 15:
        #
        # 15 * 11 + 15 = 180
        # =================================================

        self.observation_space = spaces.Box(

            low=-np.inf,

            high=np.inf,

            shape=(
                self.state_builder.dimension,
            ),

            dtype=np.float32
        )

        # =================================================
        # Action space
        #
        # PPO controls eta.
        # =================================================

        self.action_space = spaces.Box(

            low=np.array(
                [0.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )

    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        if len(
            self.transactions
        ) == 0:

            raise ValueError(
                "RoutingEnv received an empty transaction list."
            )

        self.tx_index = 0

        self.current_tx = (
            self.transactions[
                self.tx_index
            ]
        )

        # Reset previous routing information.

        self.current_paths = []

        self.last_failure_probability = 0.0
        self.last_average_delay = 0.0
        self.last_backtrack_count = 0

        observation = self._build_state()

        return (
            observation,
            {}
        )

    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        action
    ):

        if self.current_tx is None:

            raise RuntimeError(
                "Environment has not been reset."
            )

        tx = self.current_tx

        # =================================================
        # 1. PPO ACTION
        # =================================================

        action = np.asarray(
            action,
            dtype=np.float32
        ).reshape(-1)

        if action.size == 0:

            raise ValueError(
                "RoutingEnv received an empty action."
            )

        eta = float(
            action[0]
        )

        eta = float(
            np.clip(
                eta,
                float(
                    self.action_space.low[0]
                ),
                float(
                    self.action_space.high[0]
                )
            )
        )

        # =================================================
        # 2. ROUTE SELECTION
        #
        # PPO determines eta.
        # Dijkstra uses eta in its heuristic.
        # =================================================

        try:

            route = self.router.shortest_path(

                tx.source,

                tx.destination,

                tx.amount,

                self.heuristic,

                eta,

                self.cfg["graph"]["max_hops"]
            )

        except Exception:

            route = {

                "success": False,

                "path": None,

                "edges": [],

                "total_fee": 0.0,

                "total_delay": 0.0
            }

        # =================================================
        # 3. PAYMENT SIMULATION
        # =================================================

        if route.get(
            "success",
            False
        ):

            path = route.get(
                "path",
                []
            )

            edges = route.get(
                "edges",
                []
            )

            try:

                result = simulate_payment(

                    self.G,

                    path,

                    edges,

                    tx.amount
                )

                payment_success = bool(
                    result.success
                )

                carbon = float(
                    getattr(
                        result,
                        "carbon",
                        0.0
                    )
                )

            except Exception:

                payment_success = False
                carbon = 0.0

        else:

            path = None
            edges = []

            payment_success = False
            carbon = 0.0

        # =================================================
        # 4. ROUTING METRICS
        # =================================================

        path_length = len(
            edges
        )

        total_fee = float(
            route.get(
                "total_fee",
                0.0
            )
        )

        total_delay = float(
            route.get(
                "total_delay",
                0.0
            )
        )

        # =================================================
        # 5. FAILURE INFORMATION
        # =================================================

        self.last_failure_probability = (
            0.0
            if payment_success
            else 1.0
        )

        self.last_average_delay = (
            total_delay
        )

        # =================================================
        # 6. BUCKET INFORMATION
        # =================================================

        if self.bucket is not None:

            try:

                bucket_size = int(
                    self.bucket.size()
                )

            except Exception:

                bucket_size = 0

        else:

            bucket_size = 0

        # =================================================
        # 7. REWARD
        # =================================================

        reward = calculate_reward(

            success=payment_success,

            path_length=path_length,

            carbon_intensity=carbon,

            fee=total_fee,

            delay=total_delay,

            scale=self.cfg["simulation"][
                "base_reward_scale"
            ]
        )

        # =================================================
        # 8. TOP-K CANDIDATE PATHS
        #
        # Correct interface:
        #
        # top_k_paths(
        #     G,
        #     source,
        #     target,
        #     amount,
        #     heuristic_fn,
        #     eta,
        #     k,
        #     max_hops
        # )
        # =================================================

        try:

            self.current_paths = top_k_paths(

                G=self.G,

                source=tx.source,

                target=tx.destination,

                amount=tx.amount,

                heuristic_fn=self.heuristic,

                eta=eta,

                k=int(
                    self.cfg["graph"]["top_k"]
                ),

                max_hops=int(
                    self.cfg["graph"]["max_hops"]
                )
            )

        except Exception:

            self.current_paths = []

        # =================================================
        # 9. INFO
        # =================================================

        info = {

            "success":
                payment_success,

            "eta":
                eta,

            "path":
                path,

            "path_length":
                path_length,

            "fee":
                total_fee,

            "delay":
                total_delay,

            "carbon":
                carbon,

            "candidate_path_count":
                len(
                    self.current_paths
                ),

            "bucket_size":
                bucket_size,

            "failure_probability":
                self.last_failure_probability,

            "average_delay":
                self.last_average_delay,

            "backtrack_count":
                self.last_backtrack_count
        }

        # =================================================
        # 10. NEXT TRANSACTION
        # =================================================

        self.tx_index += 1

        terminated = (
            self.tx_index
            >=
            len(
                self.transactions
            )
        )

        truncated = False

        # =================================================
        # 11. NEXT OBSERVATION
        # =================================================

        if terminated:

            observation = np.zeros(

                self.observation_space.shape,

                dtype=np.float32
            )

        else:

            self.current_tx = (
                self.transactions[
                    self.tx_index
                ]
            )

            self.current_paths = []

            observation = (
                self._build_state()
            )

        return (

            observation,

            float(
                reward
            ),

            terminated,

            truncated,

            info
        )

    # =====================================================
    # STATE BUILDER
    # =====================================================

    def _build_state(self):

        if self.current_tx is None:

            raise RuntimeError(
                "Cannot build state without a current transaction."
            )

        tx = self.current_tx

        # =================================================
        # Bucket information
        # =================================================

        if self.bucket is not None:

            try:

                bucket_size = int(
                    self.bucket.size()
                )

            except Exception:

                bucket_size = 0

        else:

            bucket_size = 0

        # =================================================
        # Build State
        #
        # State.__init__ accepts:
        #
        #     k
        #     radius
        #
        # NOT:
        #
        #     m
        # =================================================

        state = State(

            self.G,

            tx.source,

            tx.destination,

            transaction={

                "source":
                    tx.source,

                "destination":
                    tx.destination,

                "amount":
                    tx.amount
            },

            candidate_paths=(
                self.current_paths
            ),

            simulation_info={

                "failure_probability":
                    self.last_failure_probability,

                "average_delay":
                    self.last_average_delay
            },

            bucket_info={

                "bucket_size":
                    bucket_size,

                "backtrack_count":
                    self.last_backtrack_count
            },

            k=int(
                self.cfg["graph"][
                    "neighborhood_k"
                ]
            ),

            radius=int(
                self.cfg["graph"][
                    "neighborhood_m"
                ]
            )
        )

        # =================================================
        # Observation vector
        # =================================================

        observation = np.asarray(

            state.vector(),

            dtype=np.float32
        )

        # =================================================
        # Safety checks
        # =================================================

        if observation.ndim != 1:

            raise RuntimeError(
                "State vector must be one-dimensional."
            )

        if observation.shape != (
            self.observation_space.shape
        ):

            raise RuntimeError(

                "State dimension mismatch: "

                f"State returned "
                f"{observation.shape}, "

                f"but observation_space expects "
                f"{self.observation_space.shape}."
            )

        if not np.all(
            np.isfinite(observation)
        ):

            raise RuntimeError(
                "State contains NaN or infinite values."
            )

        return observation