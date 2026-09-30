# Simulation/environment.py

"""
Gymnasium environment for RL + Bucket + Partial Backtracking.

Architecture
------------

    Lightning Snapshot
            |
            v
      Graph Construction
            |
            v
     Transaction Input
            |
            v
   Neighborhood Sampling
            |
            v
       State Matrix
            |
            v
          PPO
            |
            v
       Action: eta
            |
            v
   Modified Heuristic
            |
            v
       LND / Dijkstra
            |
            v
     Candidate Routes
            |
            v
          Bucket
            |
            v
      Selected Route
            |
            v
         Onion
            |
            v
    Payment Simulation
            |
            v
         Failure
            |
            v
   Partial Backtracking
            |
            +------> Alternative suffix
            |
            +------> Full reroute if needed
            |
            v
          Reward
            |
            v
          PPO


Responsibilities
----------------
This environment is responsible for:

1. Defining the RL state.
2. Defining the action eta in [-1, 1].
3. Applying eta to the routing heuristic.
4. Requesting route candidates from the routing layer.
5. Using Bucket-selected routes.
6. Executing payment simulation.
7. Invoking Partial Backtracking after failure.
8. Rebuilding Onion after an alternative route is selected.
9. Calculating reward.
10. Returning Gymnasium reset()/step() outputs.

This environment does NOT implement:

- PPO itself
- Dijkstra/LND itself
- Bucket storage logic
- Failure probability logic
- Payment settlement logic
- Onion cryptography
- Partial Backtracking algorithm itself

Those responsibilities belong to their respective modules.
"""


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Gymnasium is required for Simulation/environment.py. "
        "Install it with: pip install gymnasium"
    ) from exc


from Simulation.failure_model import FailureModel
from Simulation.network_dynamics import NetworkDynamics
from Simulation.payment_simulator import PaymentSimulator
from Simulation.router import Router
from Simulation.backtrack import PartialBacktracker


class LightningRoutingEnv(gym.Env):
    """
    Gymnasium environment for adaptive Lightning routing.

    MDP:

        M = (S, A, T, R, gamma)

    State:
        Local normalized neighborhood representation.

    Action:
        eta in [-1, 1]

    Transition:
        Network state + selected routing decision +
        payment outcome + network evolution.

    Reward:
        README-compatible success / distance / CO2 formulation.

    The environment is designed to work with an external
    route provider that performs:

        modified heuristic -> LND / Dijkstra ->
        candidate routes -> Bucket -> selected route
    """

    metadata = {
        "render_modes": []
    }

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(
        self,
        graph: nx.Graph,
        route_provider: Optional[Callable] = None,
        bucket=None,
        failure_model: Optional[FailureModel] = None,
        network_dynamics: Optional[NetworkDynamics] = None,
        router: Optional[Router] = None,
        payment_simulator: Optional[PaymentSimulator] = None,
        backtracker: Optional[PartialBacktracker] = None,
        transactions: Optional[List[Dict[str, Any]]] = None,
        k_neighbors: int = 15,
        feature_dim: int = 5,
        gamma: float = 0.99,
        max_steps: int = 100,
        seed: int = 42,
        reward_scale: float = 1000.0
    ):
        """
        Parameters
        ----------
        graph : NetworkX graph
            Real Lightning Network graph.

        route_provider : callable
            External routing layer.

            Expected conceptual interface:

                route_provider(
                    graph=G,
                    source=u,
                    destination=v,
                    amount=amount,
                    eta=eta,
                    k_candidates=...,
                    bucket=...
                )

            It should return either a selected route or a
            structured routing result.

        bucket : Bucket object
            Existing Bucket module.

        failure_model : FailureModel
            Payment-level failure model.

        network_dynamics : NetworkDynamics
            Network evolution and settlement model.

        router : Router
            Onion forwarding layer.

        payment_simulator : PaymentSimulator
            Payment execution layer.

        backtracker : PartialBacktracker
            Partial backtracking module.

        transactions : list of dict
            Transaction/payment dataset.

        k_neighbors : int
            Maximum local neighborhood size.
            README reference: k=15.

        feature_dim : int
            Number of state features.
            README reference: m=5.

        gamma : float
            RL discount factor.

        max_steps : int
            Maximum steps in one episode.

        seed : int
            Reproducibility seed.

        reward_scale : float
            Reward scaling factor.
            README reference: 10^3.
        """

        super().__init__()

        # ------------------------------------------------------
        # Validate graph
        # ------------------------------------------------------

        if graph is None:
            raise ValueError(
                "graph must not be None."
            )

        self.graph = graph

        # ------------------------------------------------------
        # Configuration
        # ------------------------------------------------------

        self.k_neighbors = int(k_neighbors)

        self.feature_dim = int(feature_dim)

        self.gamma = float(gamma)

        self.max_steps = int(max_steps)

        self.seed_value = int(seed)

        self.reward_scale = float(
            reward_scale
        )

        if self.k_neighbors < 1:
            raise ValueError(
                "k_neighbors must be >= 1."
            )

        if self.feature_dim < 1:
            raise ValueError(
                "feature_dim must be >= 1."
            )

        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(
                "gamma must be in (0, 1]."
            )

        # ------------------------------------------------------
        # Random generator
        # ------------------------------------------------------

        self.np_random = np.random.default_rng(
            self.seed_value
        )

        # ------------------------------------------------------
        # External routing layer
        # ------------------------------------------------------

        self.route_provider = route_provider

        self.bucket = bucket

        # ------------------------------------------------------
        # Simulation components
        # ------------------------------------------------------

        self.failure_model = (
            failure_model
            if failure_model is not None
            else FailureModel(
                seed=self.seed_value
            )
        )

        self.network_dynamics = (
            network_dynamics
            if network_dynamics is not None
            else NetworkDynamics(
                self.graph,
                seed=self.seed_value
            )
        )

        self.router = (
            router
            if router is not None
            else Router()
        )

        self.payment_simulator = (
            payment_simulator
            if payment_simulator is not None
            else PaymentSimulator(
                self.graph,
                self.failure_model,
                self.network_dynamics
            )
        )

        self.backtracker = (
            backtracker
            if backtracker is not None
            else PartialBacktracker(
                network=self.graph,
                bucket=self.bucket
            )
        )

        # ------------------------------------------------------
        # Transactions
        # ------------------------------------------------------

        self.transactions = (
            list(transactions)
            if transactions is not None
            else []
        )

        # ------------------------------------------------------
        # Gymnasium spaces
        # ------------------------------------------------------

        # Action:
        # eta in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array(
                [-1.0],
                dtype=np.float32
            ),
            high=np.array(
                [1.0],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # State:
        #
        # k x feature_dim
        #
        # Default:
        #
        # 15 x 5
        #
        # Flattened for Stable-Baselines3.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.k_neighbors *
                self.feature_dim,
            ),
            dtype=np.float32
        )

        # ------------------------------------------------------
        # Episode state
        # ------------------------------------------------------

        self.current_step = 0

        self.current_transaction = None

        self.source = None

        self.destination = None

        self.amount = 0.0

        self.current_state = None

        self.current_route = None

        self.current_bucket_id = None

        self.current_route_id = None

        self.attempt_id = 0

        self.episode_history = []

    # ==========================================================
    # Gymnasium reset
    # ==========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):
        """
        Reset the environment.

        Returns
        -------
        observation, info
        """

        super().reset(
            seed=seed
        )

        if seed is not None:
            self.seed_value = int(seed)

            self.np_random = np.random.default_rng(
                self.seed_value
            )

        self.current_step = 0

        self.attempt_id = 0

        self.current_route = None

        self.current_bucket_id = None

        self.current_route_id = None

        self.episode_history = []

        # ------------------------------------------------------
        # Reset network state
        # ------------------------------------------------------

        try:

            self.network_dynamics.reset(
                reset_balances=False
            )

        except TypeError:

            self.network_dynamics.reset()

        # ------------------------------------------------------
        # Select transaction
        # ------------------------------------------------------

        self.current_transaction = (
            self._select_transaction(
                options
            )
        )

        self.source = (
            self.current_transaction["source"]
        )

        self.destination = (
            self.current_transaction["destination"]
        )

        self.amount = float(
            self.current_transaction["amount"]
        )

        # ------------------------------------------------------
        # Build initial state
        # ------------------------------------------------------

        self.current_state = (
            self._build_state(
                self.source,
                self.destination
            )
        )

        info = {

            "source":
                self.source,

            "destination":
                self.destination,

            "amount":
                self.amount,

            "step":
                self.current_step,

            "attempt_id":
                self.attempt_id

        }

        return (
            self.current_state,
            info
        )

    # ==========================================================
    # Gymnasium step
    # ==========================================================

    def step(
        self,
        action
    ):
        """
        Execute one RL environment step.

        Process:

            state
              ↓
            eta
              ↓
            route provider
              ↓
            selected route
              ↓
            Onion
              ↓
            Payment Simulation
              ↓
            failure?
              ↓
            Partial Backtracking
              ↓
            reward
              ↓
            next state
        """

        self.current_step += 1

        # ------------------------------------------------------
        # Normalize action
        # ------------------------------------------------------

        eta = self._parse_action(
            action
        )

        # ------------------------------------------------------
        # Obtain route from routing layer
        # ------------------------------------------------------

        routing_result = (
            self._request_route(
                eta=eta
            )
        )

        route = (
            routing_result.get("route")
            if routing_result is not None
            else None
        )

        bucket_id = (
            routing_result.get(
                "bucket_id"
            )
            if routing_result is not None
            else None
        )

        route_id = (
            routing_result.get(
                "route_id"
            )
            if routing_result is not None
            else None
        )

        # ------------------------------------------------------
        # Route unavailable
        # ------------------------------------------------------

        if route is None:

            reward = 0.0

            terminated = False

            truncated = (
                self.current_step >=
                self.max_steps
            )

            observation = self._build_state(
                self.source,
                self.destination
            )

            info = {

                "success":
                    False,

                "reason":
                    "no_route",

                "eta":
                    eta,

                "step":
                    self.current_step,

                "full_reroute_required":
                    True

            }

            self.episode_history.append(
                info
            )

            return (
                observation,
                reward,
                terminated,
                truncated,
                info
            )

        # ------------------------------------------------------
        # Validate route
        # ------------------------------------------------------

        if not self._validate_route(
            route
        ):

            reward = 0.0

            observation = self._build_state(
                self.source,
                self.destination
            )

            terminated = False

            truncated = (
                self.current_step >=
                self.max_steps
            )

            info = {

                "success":
                    False,

                "reason":
                    "invalid_route",

                "eta":
                    eta,

                "route":
                    route

            }

            self.episode_history.append(
                info
            )

            return (
                observation,
                reward,
                terminated,
                truncated,
                info
            )

        # ------------------------------------------------------
        # Save selected route
        # ------------------------------------------------------

        self.current_route = list(
            route
        )

        self.current_bucket_id = (
            bucket_id
        )

        self.current_route_id = (
            route_id
        )

        # ------------------------------------------------------
        # Execute payment
        # ------------------------------------------------------

        payment_result = (
            self._execute_payment(
                route=route,
                bucket_id=bucket_id,
                route_id=route_id
            )
        )

        # ------------------------------------------------------
        # Successful payment
        # ------------------------------------------------------

        if payment_result.get(
            "success",
            False
        ):

            reward = self._calculate_reward(
                payment_result
            )

            observation = self._build_state(
                self.source,
                self.destination
            )

            terminated = True

            truncated = False

            info = {

                "success":
                    True,

                "reason":
                    "payment_success",

                "eta":
                    eta,

                "route":
                    route,

                "bucket_id":
                    bucket_id,

                "route_id":
                    route_id,

                "attempt_id":
                    self.attempt_id,

                "reward":
                    reward,

                "payment":
                    payment_result

            }

            self.episode_history.append(
                info
            )

            self.network_dynamics.update()

            return (
                observation,
                reward,
                terminated,
                truncated,
                info
            )

        # ------------------------------------------------------
        # Payment failed
        # ------------------------------------------------------

        backtrack_result = (
            self._attempt_partial_backtracking(
                route=route,
                payment_result=payment_result,
                bucket_id=bucket_id
            )
        )

        # ------------------------------------------------------
        # Alternative suffix found
        # ------------------------------------------------------

        if backtrack_result.get(
            "success",
            False
        ):

            new_route = (
                backtrack_result[
                    "new_route"
                ]
            )

            self.attempt_id = (
                backtrack_result.get(
                    "attempt_id",
                    self.attempt_id + 1
                )
            )

            # --------------------------------------------------
            # Rebuild Onion
            # --------------------------------------------------

            rebuild_result = (
                self.router.rebuild(
                    path=new_route,
                    bucket_id=bucket_id,
                    tx_id=self._transaction_id(),
                    amount=self.amount,
                    metadata=self.current_transaction,
                    attempt_id=self.attempt_id,
                    route_id=None
                )
            )

            if rebuild_result.get(
                "success",
                False
            ):

                # ----------------------------------------------
                # Execute alternative route
                # ----------------------------------------------

                retry_result = (
                    self._execute_payment(
                        route=new_route,
                        bucket_id=bucket_id,
                        route_id=None
                    )
                )

                if retry_result.get(
                    "success",
                    False
                ):

                    reward = self._calculate_reward(
                        retry_result
                    )

                    observation = (
                        self._build_state(
                            self.source,
                            self.destination
                        )
                    )

                    terminated = True

                    truncated = False

                    info = {

                        "success":
                            True,

                        "reason":
                            "payment_success_after_partial_backtrack",

                        "eta":
                            eta,

                        "original_route":
                            route,

                        "route":
                            new_route,

                        "bucket_id":
                            bucket_id,

                        "attempt_id":
                            self.attempt_id,

                        "backtrack":
                            backtrack_result,

                        "reward":
                            reward,

                        "payment":
                            retry_result

                    }

                    self.episode_history.append(
                        info
                    )

                    self.network_dynamics.update()

                    return (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info
                    )

                # ----------------------------------------------
                # Alternative route also failed
                # ----------------------------------------------

                payment_result = retry_result

        # ------------------------------------------------------
        # Full reroute required
        # ------------------------------------------------------

        reward = self._calculate_reward(
            payment_result
        )

        observation = self._build_state(
            self.source,
            self.destination
        )

        terminated = False

        truncated = (
            self.current_step >=
            self.max_steps
        )

        info = {

            "success":
                False,

            "reason":
                "payment_failed",

            "eta":
                eta,

            "route":
                route,

            "attempt_id":
                self.attempt_id,

            "payment":
                payment_result,

            "backtrack":
                backtrack_result,

            "full_reroute_required":
                backtrack_result.get(
                    "full_reroute_required",
                    True
                ),

            "reward":
                reward

        }

        self.episode_history.append(
            info
        )

        # ------------------------------------------------------
        # Evolve network
        # ------------------------------------------------------

        self.network_dynamics.update()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ==========================================================
    # Route provider
    # ==========================================================

    def _request_route(
        self,
        eta
    ):
        """
        Request a route from the external routing layer.

        The environment does not perform Dijkstra itself.

        The route provider is responsible for:

            eta
              ↓
            modified heuristic
              ↓
            candidate routes
              ↓
            Bucket
              ↓
            selected route
        """

        if self.route_provider is None:

            return {
                "route": None,
                "bucket_id": None,
                "route_id": None,
                "reason": "route_provider_not_configured"
            }

        try:

            result = self.route_provider(
                graph=self.graph,
                source=self.source,
                destination=self.destination,
                amount=self.amount,
                eta=eta,
                bucket=self.bucket,
                k_candidates=self.k_neighbors
            )

        except TypeError:

            # Backward-compatible interface.
            result = self.route_provider(
                self.graph,
                self.source,
                self.destination,
                self.amount,
                eta
            )

        # ------------------------------------------------------
        # Normalize route-provider output
        # ------------------------------------------------------

        if result is None:

            return {
                "route": None
            }

        if isinstance(
            result,
            (list, tuple)
        ):

            return {
                "route": list(result)
            }

        if isinstance(
            result,
            dict
        ):

            return result

        return {
            "route": None,
            "reason": "unsupported_route_provider_output"
        }

    # ==========================================================
    # Execute payment
    # ==========================================================

    def _execute_payment(
        self,
        route,
        bucket_id=None,
        route_id=None
    ):
        """
        Execute selected route.

        The Router is used to construct/validate Onion forwarding.

        PaymentSimulator performs:

            failure evaluation
            fee
            delay
            carbon
            settlement
        """

        # ------------------------------------------------------
        # Build Onion / forwarding representation
        # ------------------------------------------------------

        onion_result = self.router.route(

            path=route,

            bucket_id=(
                bucket_id
                if bucket_id is not None
                else "DEFAULT"
            ),

            tx_id=self._transaction_id(),

            amount=self.amount,

            metadata=self.current_transaction,

            attempt_id=self.attempt_id,

            route_id=route_id

        )

        # ------------------------------------------------------
        # Onion itself failed
        # ------------------------------------------------------

        if not onion_result.get(
            "success",
            False
        ):

            return {

                "success":
                    False,

                "reason":
                    onion_result.get(
                        "reason",
                        "onion_failure"
                    ),

                "failed_node":
                    onion_result.get(
                        "failed_node"
                    ),

                "failure_index":
                    onion_result.get(
                        "failure_index"
                    ),

                "visited_edges":
                    [],

                "fee":
                    0.0,

                "delay":
                    0.0,

                "carbon":
                    0.0,

                "onion":
                    onion_result

            }

        # ------------------------------------------------------
        # Convert node route to graph edges
        # ------------------------------------------------------

        edges = self._route_edges(
            route
        )

        # ------------------------------------------------------
        # Payment Simulation
        # ------------------------------------------------------

        result = (
            self.payment_simulator.simulate_payment(
                path=route,
                edges=edges,
                amount=self.amount,
                tx_id=self._transaction_id()
            )
        )

        # ------------------------------------------------------
        # Convert PaymentResult if necessary
        # ------------------------------------------------------

        if hasattr(
            result,
            "to_dict"
        ):

            result = result.to_dict()

        elif not isinstance(
            result,
            dict
        ):

            result = {
                "success":
                    bool(
                        getattr(
                            result,
                            "success",
                            False
                        )
                    )
            }

        # ------------------------------------------------------
        # Attach Onion result
        # ------------------------------------------------------

        result["onion"] = onion_result

        return result

    # ==========================================================
    # Partial Backtracking
    # ==========================================================

    def _attempt_partial_backtracking(
        self,
        route,
        payment_result,
        bucket_id=None
    ):
        """
        Invoke PartialBacktracker after payment failure.
        """

        failed_edge = (
            payment_result.get(
                "failed_edge"
            )
        )

        failure_index = (
            payment_result.get(
                "failure_index"
            )
        )

        return self.backtracker.backtrack(

            route=route,

            failed_edge=failed_edge,

            failure_index=failure_index,

            amount=self.amount,

            bucket_id=bucket_id,

            attempt_id=self.attempt_id

        )

    # ==========================================================
    # State Construction
    # ==========================================================

    def _build_state(
        self,
        source,
        destination
    ):
        """
        Build normalized local state matrix.

        Default:

            k = 15
            m = 5

            state = 15 x 5

        Features:

            0. normalized capacity
            1. normalized balance/liquidity
            2. normalized fee
            3. normalized delay
            4. normalized availability

        The matrix is flattened before being returned to
        Stable-Baselines3.
        """

        nodes = self._sample_neighborhood(
            source=source,
            destination=destination
        )

        matrix = np.zeros(
            (
                self.k_neighbors,
                self.feature_dim
            ),
            dtype=np.float32
        )

        for row, node in enumerate(
            nodes[:self.k_neighbors]
        ):

            features = self._node_features(
                node=node,
                source=source,
                destination=destination
            )

            length = min(
                len(features),
                self.feature_dim
            )

            matrix[
                row,
                :length
            ] = features[
                :length
            ]

        # ------------------------------------------------------
        # Clip numerical noise
        # ------------------------------------------------------

        matrix = np.clip(
            matrix,
            -1.0,
            1.0
        )

        self.current_state = (
            matrix.flatten()
        )

        return self.current_state.copy()

    # ==========================================================
    # Neighborhood sampling
    # ==========================================================

    def _sample_neighborhood(
        self,
        source,
        destination
    ):
        """
        Sample local neighborhood around source/destination.

        The README uses a local observation rather than the
        entire Lightning graph.

        The reference configuration is k=15.
        """

        selected = []

        # ------------------------------------------------------
        # Source first
        # ------------------------------------------------------

        if source in self.graph:

            selected.append(source)

        # ------------------------------------------------------
        # Destination
        # ------------------------------------------------------

        if (
            destination in self.graph
            and
            destination not in selected
        ):

            selected.append(destination)

        # ------------------------------------------------------
        # BFS around source
        # ------------------------------------------------------

        try:

            distances = nx.single_source_shortest_path_length(
                self.graph.to_undirected(),
                source,
                cutoff=3
            )

            candidates = sorted(
                distances.items(),
                key=lambda x: (
                    x[1],
                    str(x[0])
                )
            )

            for node, _ in candidates:

                if node not in selected:

                    selected.append(node)

                if len(selected) >= self.k_neighbors:

                    break

        except Exception:

            pass

        # ------------------------------------------------------
        # Fill if necessary
        # ------------------------------------------------------

        if len(selected) < self.k_neighbors:

            for node in self.graph.nodes:

                if node not in selected:

                    selected.append(node)

                if len(selected) >= self.k_neighbors:

                    break

        return selected[
            :self.k_neighbors
        ]

    # ==========================================================
    # State features
    # ==========================================================

    def _node_features(
        self,
        node,
        source,
        destination
    ):
        """
        Extract normalized node-level features.

        Five reference features:

            capacity
            liquidity
            fee
            delay
            availability
        """

        capacity_values = []

        liquidity_values = []

        fee_values = []

        delay_values = []

        available_values = []

        try:

            if self.graph.is_directed():

                edges = self.graph.out_edges(
                    node,
                    data=True
                )

            else:

                edges = self.graph.edges(
                    node,
                    data=True
                )

            for edge in edges:

                data = edge[-1]

                capacity_values.append(
                    self._safe_float(
                        data.get(
                            "capacity",
                            0.0
                        )
                    )
                )

                liquidity_values.append(
                    self._safe_float(
                        data.get(
                            "balance_uv",
                            data.get(
                                "liquidity",
                                data.get(
                                    "capacity",
                                    0.0
                                )
                            )
                        )
                    )
                )

                fee_values.append(
                    self._safe_float(
                        data.get(
                            "fee_base_msat",
                            data.get(
                                "fee_base",
                                0.0
                            )
                        )
                    )
                )

                delay_values.append(
                    self._safe_float(
                        data.get(
                            "delay",
                            data.get(
                                "cltv_expiry_delta",
                                0.0
                            )
                        )
                    )
                )

                available_values.append(
                    1.0
                    if data.get(
                        "available",
                        True
                    )
                    else 0.0
                )

        except Exception:

            pass

        capacity = self._aggregate(
            capacity_values
        )

        liquidity = self._aggregate(
            liquidity_values
        )

        fee = self._aggregate(
            fee_values
        )

        delay = self._aggregate(
            delay_values
        )

        availability = self._aggregate(
            available_values
        )

        # ------------------------------------------------------
        # Normalize individually
        # ------------------------------------------------------

        capacity = self._normalize_capacity(
            capacity
        )

        liquidity = self._normalize_liquidity(
            liquidity,
            capacity
        )

        fee = self._normalize_value(
            fee,
            100000.0
        )

        delay = self._normalize_value(
            delay,
            1000.0
        )

        # ------------------------------------------------------
        # Encode source/destination relevance
        # ------------------------------------------------------

        source_flag = (
            1.0
            if node == source
            else 0.0
        )

        destination_flag = (
            1.0
            if node == destination
            else 0.0
        )

        # ------------------------------------------------------
        # Five core features
        #
        # capacity
        # liquidity
        # fee
        # delay
        # availability
        #
        # Source/destination flags are folded into
        # capacity/liquidity signs to keep m=5.
        # ------------------------------------------------------

        if source_flag:
            capacity = min(
                1.0,
                capacity + 0.1
            )

        if destination_flag:
            liquidity = min(
                1.0,
                liquidity + 0.1
            )

        return [

            float(
                np.clip(
                    capacity,
                    -1.0,
                    1.0
                )
            ),

            float(
                np.clip(
                    liquidity,
                    -1.0,
                    1.0
                )
            ),

            float(
                np.clip(
                    fee,
                    -1.0,
                    1.0
                )
            ),

            float(
                np.clip(
                    delay,
                    -1.0,
                    1.0
                )
            ),

            float(
                np.clip(
                    availability,
                    -1.0,
                    1.0
                )
            )

        ]

    # ==========================================================
    # Reward
    # ==========================================================

    def _calculate_reward(
        self,
        payment_result
    ):
        """
        Calculate README-compatible reward.

        Reference formulation:

            Reward =
                Success × Distance × 10^3
                /
                Sum(CO2)

        For failure:

            Reward = 0

        Distance is represented by the number of forwarding
        hops in the selected route.

        CO2 is obtained from the payment simulation.
        """

        success = bool(
            payment_result.get(
                "success",
                False
            )
        )

        if not success:
            return 0.0

        path = payment_result.get(
            "path",
            self.current_route
        )

        if path is None:
            distance = 0.0
        else:
            distance = float(
                max(
                    0,
                    len(path) - 1
                )
            )

        carbon = self._safe_float(
            payment_result.get(
                "carbon",
                0.0
            )
        )

        # ------------------------------------------------------
        # Avoid division by zero.
        # ------------------------------------------------------

        if carbon <= 0.0:

            carbon = 1.0

        reward = (
            distance *
            self.reward_scale /
            carbon
        )

        return float(reward)

    # ==========================================================
    # Action
    # ==========================================================

    @staticmethod
    def _parse_action(
        action
    ):
        """
        Convert Gymnasium action into scalar eta.
        """

        value = np.asarray(
            action,
            dtype=np.float32
        ).reshape(-1)

        if len(value) == 0:
            return 0.0

        return float(
            np.clip(
                value[0],
                -1.0,
                1.0
            )
        )

    # ==========================================================
    # Transaction selection
    # ==========================================================

    def _select_transaction(
        self,
        options
    ):
        """
        Select current transaction.

        If options contains an explicit transaction, use it.

        Otherwise select from the supplied transaction dataset.

        Required transaction fields:

            source
            destination
            amount
        """

        if options is not None:

            transaction = options.get(
                "transaction"
            )

            if transaction is not None:

                self._validate_transaction(
                    transaction
                )

                return dict(
                    transaction
                )

        if not self.transactions:

            raise ValueError(
                "No transactions available. "
                "Provide transactions=[...] or "
                "options={'transaction': ...}."
            )

        index = int(
            self.np_random.integers(
                0,
                len(self.transactions)
            )
        )

        transaction = dict(
            self.transactions[index]
        )

        self._validate_transaction(
            transaction
        )

        return transaction

    # ==========================================================
    # Transaction validation
    # ==========================================================

    @staticmethod
    def _validate_transaction(
        transaction
    ):
        required = (
            "source",
            "destination",
            "amount"
        )

        for key in required:

            if key not in transaction:

                raise ValueError(
                    f"Transaction missing field: {key}"
                )

        if transaction["source"] == transaction[
            "destination"
        ]:

            raise ValueError(
                "Transaction source and destination "
                "must be different."
            )

        if float(
            transaction["amount"]
        ) <= 0:

            raise ValueError(
                "Transaction amount must be positive."
            )

    # ==========================================================
    # Transaction ID
    # ==========================================================

    def _transaction_id(self):
        """
        Return transaction identifier.
        """

        if self.current_transaction is None:

            return (
                f"TX-{self.current_step}"
            )

        return str(
            self.current_transaction.get(
                "tx_id",
                self.current_transaction.get(
                    "id",
                    f"TX-{self.current_step}"
                )
            )
        )

    # ==========================================================
    # Route edges
    # ==========================================================

    def _route_edges(
        self,
        route
    ):
        """
        Convert node path into directed graph edges.

        For MultiDiGraph the first available key is selected.

        Exact edge keys can later be supplied by the routing
        layer when the complete integration is implemented.
        """

        edges = []

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not self.graph.has_edge(
                u,
                v
            ):

                raise ValueError(
                    f"Route contains nonexistent edge: "
                    f"{u} -> {v}"
                )

            if self.graph.is_multigraph():

                edge_data = (
                    self.graph.get_edge_data(
                        u,
                        v
                    )
                )

                if not edge_data:

                    raise ValueError(
                        f"No edge data for {u} -> {v}"
                    )

                key = next(
                    iter(edge_data.keys())
                )

                edges.append(
                    (
                        u,
                        v,
                        key
                    )
                )

            else:

                edges.append(
                    (
                        u,
                        v
                    )
                )

        return edges

    # ==========================================================
    # Route validation
    # ==========================================================

    def _validate_route(
        self,
        route
    ):
        """
        Validate selected route against current graph.
        """

        if not isinstance(
            route,
            (list, tuple)
        ):
            return False

        if len(route) < 2:
            return False

        if len(route) != len(
            set(route)
        ):
            return False

        if route[0] != self.source:
            return False

        if route[-1] != self.destination:
            return False

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not self.graph.has_edge(
                u,
                v
            ):
                return False

        return True

    # ==========================================================
    # Numeric helpers
    # ==========================================================

    @staticmethod
    def _safe_float(
        value
    ):
        try:
            value = float(value)

            if not np.isfinite(value):
                return 0.0

            return value

        except (
            TypeError,
            ValueError
        ):
            return 0.0

    @staticmethod
    def _aggregate(
        values
    ):
        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    @staticmethod
    def _normalize_value(
        value,
        scale
    ):
        if scale <= 0:
            return 0.0

        return float(
            np.clip(
                value / scale,
                0.0,
                1.0
            )
        )

    @staticmethod
    def _normalize_capacity(
        value
    ):
        """
        Capacity normalization.

        Lightning capacities can vary substantially,
        therefore log normalization is used.
        """

        value = max(
            0.0,
            float(value)
        )

        return float(
            np.clip(
                np.log1p(value) / 20.0,
                0.0,
                1.0
            )
        )

    @staticmethod
    def _normalize_liquidity(
        liquidity,
        normalized_capacity
    ):
        """
        Convert liquidity to a relative representation.

        Since normalized_capacity is already bounded,
        the liquidity value is conservatively clipped.
        """

        liquidity = max(
            0.0,
            float(liquidity)
        )

        # Relative scaling for Lightning-like amounts.
        normalized = np.log1p(
            liquidity
        ) / 20.0

        return float(
            np.clip(
                normalized,
                0.0,
                1.0
            )
        )

    # ==========================================================
    # Render
    # ==========================================================

    def render(self):
        """
        Optional textual rendering.
        """

        print(
            {
                "step":
                    self.current_step,

                "source":
                    self.source,

                "destination":
                    self.destination,

                "amount":
                    self.amount,

                "route":
                    self.current_route,

                "attempt_id":
                    self.attempt_id
            }
        )

    # ==========================================================
    # Close
    # ==========================================================

    def close(self):
        """
        Release environment resources.
        """

        pass


# ==============================================================
# Backward-compatible alias
# ==============================================================

LightningEnv = LightningRoutingEnv


# ==============================================================
# Simple integration test
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Create a small directed Lightning-like graph
    # ----------------------------------------------------------

    G = nx.MultiDiGraph()

    edges = [

        ("A", "B"),
        ("B", "C"),
        ("C", "D"),

        # Alternative path
        ("B", "E"),
        ("E", "D")

    ]

    for u, v in edges:

        G.add_edge(

            u,
            v,

            capacity=10000,

            balance_uv=10000,

            balance_vu=10000,

            fee_base_msat=1000,

            fee_proportional_millionths=1,

            delay=10,

            available=True

        )

    # ----------------------------------------------------------
    # Simple route provider
    #
    # This is only for testing the environment.
    # Real implementation should be connected to:
    #
    # RL -> heuristic -> Pathfinding -> Bucket
    # ----------------------------------------------------------

    def test_route_provider(
        graph,
        source,
        destination,
        amount,
        eta,
        bucket=None,
        k_candidates=15
    ):

        try:

            route = nx.shortest_path(
                graph,
                source,
                destination
            )

        except nx.NetworkXNoPath:

            return {
                "route": None
            }

        return {

            "route":
                route,

            "bucket_id":
                "TEST_BUCKET",

            "route_id":
                "TEST_ROUTE"

        }

    # ----------------------------------------------------------
    # Transaction
    # ----------------------------------------------------------

    transactions = [

        {

            "tx_id":
                "TX001",

            "source":
                "A",

            "destination":
                "D",

            "amount":
                1000

        }

    ]

    # ----------------------------------------------------------
    # Environment
    # ----------------------------------------------------------

    env = LightningRoutingEnv(

        graph=G,

        route_provider=
            test_route_provider,

        transactions=
            transactions,

        k_neighbors=15,

        feature_dim=5,

        gamma=0.99,

        max_steps=10,

        seed=42

    )

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------

    observation, info = env.reset()

    print("=" * 70)
    print("ENVIRONMENT RESET")
    print("=" * 70)

    print(
        "Observation shape:",
        observation.shape
    )

    print(
        "Info:",
        info
    )

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------

    action = np.array(
        [0.0],
        dtype=np.float32
    )

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    print("=" * 70)
    print("ENVIRONMENT STEP")
    print("=" * 70)

    print(
        "Reward:",
        reward
    )

    print(
        "Terminated:",
        terminated
    )

    print(
        "Truncated:",
        truncated
    )

    print(
        "Info:",
        info
    )

    env.close()