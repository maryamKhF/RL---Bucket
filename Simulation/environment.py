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


Important project contracts
---------------------------

1. PPO controls eta only.
2. eta is always in [0, 1].
3. Top-K is fixed at 5.
4. MultiDiGraph channel identity must be preserved.
5. No arbitrary parallel-channel selection is allowed.
6. capacity is NOT directional liquidity.
7. Unknown directional liquidity remains unknown.
8. Bucket.attempts is not modified by this environment directly.
9. PartialBacktracker is responsible for partial backtracking.
10. PaymentSimulator is responsible for payment execution and settlement.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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

    PPO controls only eta.

    Routing itself is delegated to the routing layer:

        eta
          |
          v
    modified heuristic
          |
          v
      Top-K (5)
          |
          v
        Bucket
          |
          v
    selected route
          |
          v
    PaymentSimulator
          |
          v
    FailureModel
          |
          v
    PartialBacktracker
    """

    metadata = {
        "render_modes": []
    }

    # ==========================================================
    # Project-level constants
    # ==========================================================

    TOP_K = 5

    ETA_LOW = 0.0
    ETA_HIGH = 1.0

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
        reward_scale: float = 1000.0,
    ):
        super().__init__()

        if graph is None:
            raise ValueError("graph must not be None.")

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "graph must be a NetworkX graph."
            )

        self.graph = graph

        self.k_neighbors = int(k_neighbors)
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.seed_value = int(seed)
        self.reward_scale = float(reward_scale)

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

        if self.max_steps < 1:
            raise ValueError(
                "max_steps must be >= 1."
            )

        if self.reward_scale <= 0.0:
            raise ValueError(
                "reward_scale must be > 0."
            )

        # ------------------------------------------------------
        # Random generator
        # ------------------------------------------------------

        self.np_random = np.random.default_rng(
            self.seed_value
        )

        # ------------------------------------------------------
        # Routing layer
        # ------------------------------------------------------

        self.route_provider = route_provider
        self.bucket = bucket

        # ------------------------------------------------------
        # Failure model
        # ------------------------------------------------------

        self.failure_model = (
            failure_model
            if failure_model is not None
            else FailureModel(
                seed=self.seed_value
            )
        )

        # ------------------------------------------------------
        # Network dynamics
        # ------------------------------------------------------

        self.network_dynamics = (
            network_dynamics
            if network_dynamics is not None
            else NetworkDynamics(
                self.graph,
                seed=self.seed_value
            )
        )

        # ------------------------------------------------------
        # Onion/router
        # ------------------------------------------------------

        self.router = (
            router
            if router is not None
            else Router()
        )

        # ------------------------------------------------------
        # Payment simulator
        # ------------------------------------------------------

        self.payment_simulator = (
            payment_simulator
            if payment_simulator is not None
            else PaymentSimulator(
                self.graph,
                self.failure_model,
                self.network_dynamics,
            )
        )

        # ------------------------------------------------------
        # Partial backtracker
        # ------------------------------------------------------

        self.backtracker = (
            backtracker
            if backtracker is not None
            else PartialBacktracker(
                network=self.graph,
                bucket=self.bucket,
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

        # ======================================================
        # Gymnasium spaces
        # ======================================================

        # ------------------------------------------------------
        # Action
        #
        # PPO controls eta in [0, 1].
        # ------------------------------------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [self.ETA_LOW],
                dtype=np.float32,
            ),
            high=np.array(
                [self.ETA_HIGH],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # ------------------------------------------------------
        # Observation
        # ------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.k_neighbors *
                self.feature_dim,
            ),
            dtype=np.float32,
        )

        # ======================================================
        # Episode state
        # ======================================================

        self.current_step = 0

        self.current_transaction = None

        self.source = None
        self.destination = None
        self.amount = 0.0

        self.current_state = None

        # Node path
        self.current_route = None

        # Exact edge representation
        self.current_route_edges = None

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
        options=None,
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
        self.current_route_edges = None

        self.current_bucket_id = None
        self.current_route_id = None

        self.episode_history = []

        # ------------------------------------------------------
        # Reset network runtime state
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
        # Build state
        # ------------------------------------------------------

        self.current_state = (
            self._build_state(
                self.source,
                self.destination,
            )
        )

        info = {
            "source": self.source,
            "destination": self.destination,
            "amount": self.amount,
            "step": self.current_step,
            "attempt_id": self.attempt_id,
        }

        return (
            self.current_state,
            info,
        )

    # ==========================================================
    # Gymnasium step
    # ==========================================================

    def step(
        self,
        action,
    ):
        """
        Execute one environment step.

        Process:

            state
              |
              v
            eta
              |
              v
        route provider
              |
              v
        Bucket-selected route
              |
              v
           Onion
              |
              v
        PaymentSimulator
              |
              v
           failure?
              |
              v
        PartialBacktracking
              |
              v
           retry
              |
              v
           reward
        """

        self.current_step += 1

        # ------------------------------------------------------
        # Parse eta
        # ------------------------------------------------------

        eta = self._parse_action(
            action
        )

        # ------------------------------------------------------
        # Request route
        # ------------------------------------------------------

        routing_result = (
            self._request_route(
                eta=eta
            )
        )

        if routing_result is None:
            routing_result = {
                "route": None,
            }

        route = routing_result.get(
            "route"
        )

        route_edges = routing_result.get(
            "route_edges"
        )

        bucket_id = routing_result.get(
            "bucket_id"
        )

        route_id = routing_result.get(
            "route_id"
        )

        # ------------------------------------------------------
        # No route
        # ------------------------------------------------------

        if route is None:

            reward = 0.0

            observation = self._build_state(
                self.source,
                self.destination,
            )

            terminated = False

            truncated = (
                self.current_step >=
                self.max_steps
            )

            info = {
                "success": False,
                "reason": "no_route",
                "eta": eta,
                "step": self.current_step,
                "attempt_id": self.attempt_id,
                "full_reroute_required": True,
            }

            self.episode_history.append(
                info
            )

            return (
                observation,
                reward,
                terminated,
                truncated,
                info,
            )

        # ------------------------------------------------------
        # Normalize route
        # ------------------------------------------------------

        route = self._normalize_route(
            route
        )

        if route is None:

            return self._failure_step(
                reason="invalid_route",
                eta=eta,
                route=None,
                payment_result={
                    "success": False,
                    "reason": "invalid_route",
                },
            )

        # ------------------------------------------------------
        # Normalize supplied edge representation
        # ------------------------------------------------------

        route_edges = self._normalize_route_edges(
            route_edges
        )

        # ------------------------------------------------------
        # Validate route + exact channel identity
        # ------------------------------------------------------

        if not self._validate_route(
            route,
            route_edges=route_edges,
        ):

            return self._failure_step(
                reason="invalid_route",
                eta=eta,
                route=route,
                payment_result={
                    "success": False,
                    "reason": "invalid_route",
                },
            )

        # ------------------------------------------------------
        # Save selected route
        # ------------------------------------------------------

        self.current_route = list(
            route
        )

        self.current_route_edges = (
            list(route_edges)
            if route_edges is not None
            else None
        )

        self.current_bucket_id = bucket_id
        self.current_route_id = route_id

        # ------------------------------------------------------
        # Execute payment
        # ------------------------------------------------------

        payment_result = (
            self._execute_payment(
                route=route,
                route_edges=route_edges,
                bucket_id=bucket_id,
                route_id=route_id,
            )
        )

        # ------------------------------------------------------
        # Payment success
        # ------------------------------------------------------

        if payment_result.get(
            "success",
            False,
        ):

            reward = self._calculate_reward(
                payment_result
            )

            observation = self._build_state(
                self.source,
                self.destination,
            )

            terminated = True
            truncated = False

            info = {
                "success": True,
                "reason": "payment_success",
                "eta": eta,
                "route": route,
                "route_edges": route_edges,
                "bucket_id": bucket_id,
                "route_id": route_id,
                "attempt_id": self.attempt_id,
                "reward": reward,
                "payment": payment_result,
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
                info,
            )

        # ======================================================
        # Initial payment failed
        # ======================================================

        backtrack_result = (
            self._attempt_partial_backtracking(
                route=route,
                route_edges=route_edges,
                payment_result=payment_result,
                bucket_id=bucket_id,
            )
        )

        # ======================================================
        # Alternative suffix found
        # ======================================================

        if backtrack_result.get(
            "success",
            False,
        ):

            new_route = (
                backtrack_result.get(
                    "new_route"
                )
            )

            new_route_edges = (
                backtrack_result.get(
                    "new_route_edges"
                )
            )

            if new_route is not None:

                new_route = self._normalize_route(
                    new_route
                )

            new_route_edges = (
                self._normalize_route_edges(
                    new_route_edges
                )
            )

            # --------------------------------------------------
            # Exact channel identity is mandatory on MultiGraph
            # --------------------------------------------------

            if (
                new_route is None
                or not self._validate_route(
                    new_route,
                    route_edges=new_route_edges,
                )
            ):

                payment_result = {
                    "success": False,
                    "reason": (
                        "backtrack_route_missing_exact_channel"
                    ),
                    "failed_edge": (
                        payment_result.get(
                            "failed_edge"
                        )
                    ),
                    "failure_index": (
                        payment_result.get(
                            "failure_index"
                        )
                    ),
                }

            else:

                # --------------------------------------------------
                # Backtracking attempt ID
                # --------------------------------------------------

                self.attempt_id = int(
                    backtrack_result.get(
                        "attempt_id",
                        self.attempt_id + 1,
                    )
                )

                # --------------------------------------------------
                # Rebuild Onion
                # --------------------------------------------------

                rebuild_result = (
                    self._rebuild_onion(
                        route=new_route,
                        bucket_id=bucket_id,
                        route_id=None,
                    )
                )

                if rebuild_result.get(
                    "success",
                    False,
                ):

                    # ----------------------------------------------
                    # Execute alternative route
                    # ----------------------------------------------

                    retry_result = (
                        self._execute_payment(
                            route=new_route,
                            route_edges=new_route_edges,
                            bucket_id=bucket_id,
                            route_id=None,
                        )
                    )

                    if retry_result.get(
                        "success",
                        False,
                    ):

                        self.current_route = list(
                            new_route
                        )

                        self.current_route_edges = (
                            list(new_route_edges)
                            if new_route_edges is not None
                            else None
                        )

                        reward = self._calculate_reward(
                            retry_result
                        )

                        observation = self._build_state(
                            self.source,
                            self.destination,
                        )

                        terminated = True
                        truncated = False

                        info = {
                            "success": True,
                            "reason": (
                                "payment_success_after_partial_backtrack"
                            ),
                            "eta": eta,
                            "original_route": route,
                            "original_route_edges": route_edges,
                            "route": new_route,
                            "route_edges": new_route_edges,
                            "bucket_id": bucket_id,
                            "attempt_id": self.attempt_id,
                            "backtrack": backtrack_result,
                            "reward": reward,
                            "payment": retry_result,
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
                            info,
                        )

                    # ----------------------------------------------
                    # Alternative route also failed
                    # ----------------------------------------------

                    payment_result = retry_result

                else:

                    payment_result = {
                        "success": False,
                        "reason": (
                            "onion_rebuild_failed"
                        ),
                        "failed_edge": (
                            payment_result.get(
                                "failed_edge"
                            )
                        ),
                        "failure_index": (
                            payment_result.get(
                                "failure_index"
                            )
                        ),
                        "onion": rebuild_result,
                    }

        # ======================================================
        # Failure after backtracking
        # ======================================================

        return self._failure_step(
            reason="payment_failed",
            eta=eta,
            route=route,
            payment_result=payment_result,
            bucket_id=bucket_id,
            route_id=route_id,
            route_edges=route_edges,
            backtrack_result=backtrack_result,
        )

    # ==========================================================
    # Failure-step helper
    # ==========================================================

    def _failure_step(
        self,
        reason,
        eta,
        route,
        payment_result,
        bucket_id=None,
        route_id=None,
        route_edges=None,
        backtrack_result=None,
    ):
        """
        Build a consistent failed Gymnasium step.
        """

        if backtrack_result is None:
            backtrack_result = {
                "success": False,
                "full_reroute_required": True,
            }

        reward = self._calculate_reward(
            payment_result
        )

        observation = self._build_state(
            self.source,
            self.destination,
        )

        terminated = False

        truncated = (
            self.current_step >=
            self.max_steps
        )

        info = {
            "success": False,
            "reason": reason,
            "eta": eta,
            "route": route,
            "route_edges": route_edges,
            "bucket_id": bucket_id,
            "route_id": route_id,
            "attempt_id": self.attempt_id,
            "payment": payment_result,
            "backtrack": backtrack_result,
            "full_reroute_required": (
                backtrack_result.get(
                    "full_reroute_required",
                    True,
                )
            ),
            "reward": reward,
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
            info,
        )

    # ==========================================================
    # Route provider
    # ==========================================================

    def _request_route(
        self,
        eta,
    ):
        """
        Request routing result.

        Fixed project Top-K = 5.

        The environment never changes this to k_neighbors.
        """

        if self.route_provider is None:

            return {
                "route": None,
                "bucket_id": None,
                "route_id": None,
                "reason": (
                    "route_provider_not_configured"
                ),
            }

        # ------------------------------------------------------
        # Preferred interface
        # ------------------------------------------------------

        try:

            result = self.route_provider(
                graph=self.graph,
                source=self.source,
                destination=self.destination,
                amount=self.amount,
                eta=eta,
                bucket=self.bucket,
                k_candidates=self.TOP_K,
            )

        except TypeError as exc:

            # --------------------------------------------------
            # Only fall back for genuine signature mismatch.
            # We do not silently swallow arbitrary provider
            # exceptions.
            # --------------------------------------------------

            message = str(exc)

            signature_markers = (
                "unexpected keyword",
                "positional argument",
                "required positional",
                "got an unexpected",
                "missing",
            )

            if not any(
                marker in message
                for marker in signature_markers
            ):
                raise

            result = self.route_provider(
                self.graph,
                self.source,
                self.destination,
                self.amount,
                eta,
            )

        return self._normalize_routing_result(
            result
        )

    # ==========================================================
    # Routing result normalization
    # ==========================================================

    def _normalize_routing_result(
        self,
        result,
    ):
        """
        Normalize routing-layer output.

        Supported:

            list/tuple:
                node path

            dict:
                {
                    route,
                    route_edges,
                    bucket_id,
                    route_id,
                    ...
                }
        """

        if result is None:

            return {
                "route": None
            }

        if isinstance(
            result,
            dict,
        ):

            normalized = dict(
                result
            )

            route = normalized.get(
                "route"
            )

            if route is not None:

                route = self._normalize_route(
                    route
                )

            normalized["route"] = route

            normalized["route_edges"] = (
                self._normalize_route_edges(
                    normalized.get(
                        "route_edges"
                    )
                )
            )

            return normalized

        if isinstance(
            result,
            (list, tuple),
        ):

            return {
                "route": self._normalize_route(
                    result
                ),
                "route_edges": None,
            }

        raise TypeError(
            "Unsupported route_provider output type: "
            f"{type(result).__name__}"
        )

    # ==========================================================
    # Payment execution
    # ==========================================================

    def _execute_payment(
        self,
        route,
        route_edges=None,
        bucket_id=None,
        route_id=None,
    ):
        """
        Execute payment.

        Exact route edge identity is preserved.

        For MultiGraph/MultiDiGraph, route_edges MUST contain
        exact channel keys.
        """

        # ------------------------------------------------------
        # Validate route
        # ------------------------------------------------------

        if not self._validate_route(
            route,
            route_edges=route_edges,
        ):

            return {
                "success": False,
                "reason": "invalid_route",
                "failed_node": None,
                "failed_edge": None,
                "failure_index": None,
                "visited_edges": [],
                "fee": 0.0,
                "delay": 0.0,
                "carbon": 0.0,
            }

        # ------------------------------------------------------
        # Build Onion
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
            route_id=route_id,
        )

        if not onion_result.get(
            "success",
            False,
        ):

            return {
                "success": False,
                "reason": onion_result.get(
                    "reason",
                    "onion_failure",
                ),
                "failed_node": onion_result.get(
                    "failed_node"
                ),
                "failed_edge": onion_result.get(
                    "failed_edge"
                ),
                "failure_index": onion_result.get(
                    "failure_index"
                ),
                "visited_edges": onion_result.get(
                    "visited_edges",
                    [],
                ),
                "fee": 0.0,
                "delay": 0.0,
                "carbon": 0.0,
                "onion": onion_result,
            }

        # ------------------------------------------------------
        # PaymentSimulator
        # ------------------------------------------------------

        result = self.payment_simulator.simulate_payment(
            path=route,
            edges=list(route_edges),
            amount=self.amount,
            tx_id=self._transaction_id(),
        )

        # ------------------------------------------------------
        # Normalize PaymentResult
        # ------------------------------------------------------

        if hasattr(
            result,
            "to_dict",
        ):

            result = result.to_dict()

        elif not isinstance(
            result,
            dict,
        ):

            result = {
                "success": bool(
                    getattr(
                        result,
                        "success",
                        False,
                    )
                )
            }

        result = dict(
            result
        )

        result["onion"] = onion_result

        # ------------------------------------------------------
        # Preserve exact edges in environment-level result.
        # ------------------------------------------------------

        result.setdefault(
            "path",
            list(route),
        )

        result.setdefault(
            "route_edges",
            list(route_edges),
        )

        return result

    # ==========================================================
    # Onion rebuild
    # ==========================================================

    def _rebuild_onion(
        self,
        route,
        bucket_id,
        route_id,
    ):
        """
        Rebuild Onion after partial backtracking.
        """

        return self.router.rebuild(
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
            route_id=route_id,
        )

    # ==========================================================
    # Partial backtracking
    # ==========================================================

    def _attempt_partial_backtracking(
        self,
        route,
        route_edges,
        payment_result,
        bucket_id=None,
    ):
        """
        Invoke PartialBacktracker.

        This method does NOT modify Bucket.attempts.

        Bucket.attempts must represent actual payment attempts.
        """

        failed_edge = payment_result.get(
            "failed_edge"
        )

        failure_index = payment_result.get(
            "failure_index"
        )

        # ------------------------------------------------------
        # Prefer exact route-edge information from payment result
        # ------------------------------------------------------

        failed_edge = (
            failed_edge
            if failed_edge is not None
            else None
        )

        try:

            result = self.backtracker.backtrack(
                route=route,
                failed_edge=failed_edge,
                failure_index=failure_index,
                amount=self.amount,
                bucket_id=bucket_id,
                attempt_id=self.attempt_id,
            )

        except TypeError as exc:

            message = str(exc)

            if (
                "unexpected keyword" not in message
                and "positional argument" not in message
                and "required positional" not in message
            ):
                raise

            result = self.backtracker.backtrack(
                route,
                failed_edge,
                failure_index,
                self.amount,
                bucket_id,
                self.attempt_id,
            )

        if result is None:

            return {
                "success": False,
                "full_reroute_required": True,
            }

        if not isinstance(
            result,
            dict,
        ):

            raise TypeError(
                "PartialBacktracker.backtrack() must "
                "return a dictionary."
            )

        return result

    # ==========================================================
    # State construction
    # ==========================================================

    def _build_state(
        self,
        source,
        destination,
    ):
        """
        Build normalized local state.

        Default configuration:

            k = 15
            m = 5

        IMPORTANT:

        capacity and directional liquidity are kept
        conceptually separate.

        Unknown directional liquidity is encoded as 0 rather
        than being replaced by capacity.
        """

        nodes = self._sample_neighborhood(
            source=source,
            destination=destination,
        )

        matrix = np.zeros(
            (
                self.k_neighbors,
                self.feature_dim,
            ),
            dtype=np.float32,
        )

        for row, node in enumerate(
            nodes[:self.k_neighbors]
        ):

            features = self._node_features(
                node=node,
                source=source,
                destination=destination,
            )

            length = min(
                len(features),
                self.feature_dim,
            )

            matrix[
                row,
                :length,
            ] = features[
                :length
            ]

        matrix = np.clip(
            matrix,
            -1.0,
            1.0,
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
        destination,
    ):
        """
        Deterministic local neighborhood selection.
        """

        selected = []

        if source in self.graph:
            selected.append(source)

        if (
            destination in self.graph
            and destination not in selected
        ):
            selected.append(destination)

        try:

            undirected = (
                self.graph.to_undirected()
                if self.graph.is_directed()
                else self.graph
            )

            distances = (
                nx.single_source_shortest_path_length(
                    undirected,
                    source,
                    cutoff=3,
                )
            )

            candidates = sorted(
                distances.items(),
                key=lambda item: (
                    item[1],
                    str(item[0]),
                ),
            )

            for node, _ in candidates:

                if node not in selected:
                    selected.append(node)

                if len(selected) >= self.k_neighbors:
                    break

        except (
            nx.NetworkXError,
            TypeError,
        ):

            pass

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
    # Node features
    # ==========================================================

    def _node_features(
        self,
        node,
        source,
        destination,
    ):
        """
        Five normalized features:

            1. capacity
            2. directional liquidity estimate
            3. fee
            4. delay
            5. availability

        capacity is NEVER used as liquidity.
        """

        capacity_values = []
        liquidity_values = []
        fee_values = []
        delay_values = []
        available_values = []

        try:

            if self.graph.is_directed():

                if self.graph.is_multigraph():

                    edges = self.graph.out_edges(
                        node,
                        keys=True,
                        data=True,
                    )

                else:

                    edges = self.graph.out_edges(
                        node,
                        data=True,
                    )

            else:

                if self.graph.is_multigraph():

                    edges = self.graph.edges(
                        node,
                        keys=True,
                        data=True,
                    )

                else:

                    edges = self.graph.edges(
                        node,
                        data=True,
                    )

            for edge in edges:

                data = edge[-1]

                # --------------------------------------------------
                # Capacity
                # --------------------------------------------------

                capacity = self._read_numeric(
                    data.get(
                        "capacity"
                    )
                )

                if capacity is not None:
                    capacity_values.append(
                        capacity
                    )

                # --------------------------------------------------
                # Directional liquidity
                #
                # IMPORTANT:
                # no capacity fallback.
                # --------------------------------------------------

                liquidity = (
                    self._read_directional_liquidity(
                        data
                    )
                )

                if liquidity is not None:
                    liquidity_values.append(
                        liquidity
                    )

                # --------------------------------------------------
                # Fee
                # --------------------------------------------------

                fee = self._read_numeric(
                    data.get(
                        "fee_base_msat",
                        data.get(
                            "fee_base"
                        ),
                    )
                )

                if fee is not None:
                    fee_values.append(
                        fee
                    )

                # --------------------------------------------------
                # Delay
                # --------------------------------------------------

                delay = self._read_numeric(
                    data.get(
                        "delay",
                        data.get(
                            "cltv_expiry_delta"
                        ),
                    )
                )

                if delay is not None:
                    delay_values.append(
                        delay
                    )

                # --------------------------------------------------
                # Availability
                # --------------------------------------------------

                available_values.append(
                    1.0
                    if data.get(
                        "available",
                        True,
                    )
                    else 0.0
                )

        except (
            KeyError,
            TypeError,
            nx.NetworkXError,
        ):

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

        capacity = self._normalize_capacity(
            capacity
        )

        liquidity = self._normalize_liquidity(
            liquidity
        )

        fee = self._normalize_value(
            fee,
            100000.0,
        )

        delay = self._normalize_value(
            delay,
            1000.0,
        )

        # ------------------------------------------------------
        # Source/destination indicators remain folded into
        # existing five-feature representation.
        # ------------------------------------------------------

        if node == source:

            capacity = min(
                1.0,
                capacity + 0.1,
            )

        if node == destination:

            liquidity = min(
                1.0,
                liquidity + 0.1,
            )

        return [
            float(
                np.clip(
                    capacity,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    liquidity,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    fee,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    delay,
                    -1.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    availability,
                    -1.0,
                    1.0,
                )
            ),
        ]

    # ==========================================================
    # Directional liquidity reader
    # ==========================================================

    @classmethod
    def _read_directional_liquidity(
        cls,
        data,
    ):
        """
        Read explicitly available directional liquidity.

        Priority:

            balance_uv
            liquidity_uv
            liquidity
            estimated_liquidity

        capacity is intentionally excluded.

        None means unknown.
        """

        if not isinstance(
            data,
            dict,
        ):
            return None

        for key in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            if key not in data:
                continue

            value = cls._read_numeric(
                data.get(key)
            )

            if value is not None:
                return max(
                    0.0,
                    value,
                )

        return None

    # ==========================================================
    # Reward
    # ==========================================================

    def _calculate_reward(
        self,
        payment_result,
    ):
        """
        Success reward:

            distance * reward_scale / carbon

        Failure:

            0
        """

        if not isinstance(
            payment_result,
            dict,
        ):
            return 0.0

        success = bool(
            payment_result.get(
                "success",
                False,
            )
        )

        if not success:
            return 0.0

        path = payment_result.get(
            "path",
            self.current_route,
        )

        if path is None:

            distance = 0.0

        else:

            distance = float(
                max(
                    0,
                    len(path) - 1,
                )
            )

        carbon = self._safe_float(
            payment_result.get(
                "carbon",
                0.0,
            )
        )

        if carbon <= 0.0:
            carbon = 1.0

        reward = (
            distance *
            self.reward_scale /
            carbon
        )

        return float(
            reward
        )

    # ==========================================================
    # Action parsing
    # ==========================================================

    @staticmethod
    def _parse_action(
        action,
    ):
        """
        Convert Gymnasium action into eta in [0, 1].
        """

        value = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if value.size == 0:
            return 0.0

        eta = float(
            value[0]
        )

        if not np.isfinite(eta):
            return 0.0

        return float(
            np.clip(
                eta,
                0.0,
                1.0,
            )
        )

    # ==========================================================
    # Transaction selection
    # ==========================================================

    def _select_transaction(
        self,
        options,
    ):
        """
        Select transaction.
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
                len(self.transactions),
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
        transaction,
    ):
        if not isinstance(
            transaction,
            dict,
        ):
            raise TypeError(
                "transaction must be a dictionary."
            )

        for key in (
            "source",
            "destination",
            "amount",
        ):

            if key not in transaction:

                raise ValueError(
                    f"Transaction missing field: {key}"
                )

        if (
            transaction["source"]
            ==
            transaction["destination"]
        ):

            raise ValueError(
                "Transaction source and destination "
                "must be different."
            )

        amount = float(
            transaction["amount"]
        )

        if not np.isfinite(amount):
            raise ValueError(
                "Transaction amount must be finite."
            )

        if amount <= 0.0:
            raise ValueError(
                "Transaction amount must be positive."
            )

    # ==========================================================
    # Transaction ID
    # ==========================================================

    def _transaction_id(
        self,
    ):
        if self.current_transaction is None:

            return (
                f"TX-{self.current_step}"
            )

        return str(
            self.current_transaction.get(
                "tx_id",
                self.current_transaction.get(
                    "id",
                    f"TX-{self.current_step}",
                ),
            )
        )

    # ==========================================================
    # Route normalization
    # ==========================================================

    @staticmethod
    def _normalize_route(
        route,
    ):
        """
        Normalize node path.

        Edge dictionaries are not accepted as the node route.
        Exact channel information belongs in route_edges.
        """

        if route is None:
            return None

        if not isinstance(
            route,
            (list, tuple),
        ):
            return None

        result = list(
            route
        )

        if len(result) < 2:
            return None

        return result

    # ==========================================================
    # Route-edge normalization
    # ==========================================================

    @staticmethod
    def _normalize_route_edges(
        route_edges,
    ):
        """
        Preserve exact routing edge representation.
        """

        if route_edges is None:
            return None

        if not isinstance(
            route_edges,
            (list, tuple),
        ):
            raise TypeError(
                "route_edges must be a list or tuple."
            )

        return list(
            route_edges
        )

    # ==========================================================
    # Route validation
    # ==========================================================

    def _validate_route(
        self,
        route,
        route_edges=None,
    ):
        """
        Validate route.

        For MultiGraph/MultiDiGraph:

            route_edges is mandatory.

        The environment never chooses an arbitrary parallel
        channel.
        """

        if not isinstance(
            route,
            (list, tuple),
        ):
            return False

        if len(route) < 2:
            return False

        try:

            if len(route) != len(
                set(route)
            ):
                return False

        except TypeError:

            return False

        if route[0] != self.source:
            return False

        if route[-1] != self.destination:
            return False

        # ------------------------------------------------------
        # Simple graph
        # ------------------------------------------------------

        if not self.graph.is_multigraph():

            for u, v in zip(
                route[:-1],
                route[1:],
            ):

                if not self.graph.has_edge(
                    u,
                    v,
                ):
                    return False

            return True

        # ------------------------------------------------------
        # MultiGraph / MultiDiGraph
        # ------------------------------------------------------

        if route_edges is None:
            return False

        if len(route_edges) != len(route) - 1:
            return False

        for index, (
            u,
            v,
        ) in enumerate(
            zip(
                route[:-1],
                route[1:],
            )
        ):

            parsed = self._parse_edge(
                route_edges[index]
            )

            if parsed is None:
                return False

            edge_u, edge_v, key = parsed

            if edge_u != u or edge_v != v:
                return False

            if key is None:
                return False

            if not self.graph.has_edge(
                u,
                v,
                key,
            ):
                return False

        return True

    # ==========================================================
    # Edge parser
    # ==========================================================

    @staticmethod
    def _parse_edge(
        edge,
    ):
        """
        Parse edge representation.

        Supported dictionary forms:

            {
                source,
                target,
                channel_key
            }

            {
                source,
                target,
                key
            }

        Supported tuples:

            (u, v)
            (u, v, key)
        """

        if isinstance(
            edge,
            dict,
        ):

            u = edge.get(
                "source"
            )

            v = edge.get(
                "target"
            )

            key = edge.get(
                "channel_key"
            )

            if key is None:
                key = edge.get(
                    "key"
                )

            if u is None or v is None:
                return None

            return (
                u,
                v,
                key,
            )

        if isinstance(
            edge,
            (tuple, list),
        ):

            if len(edge) == 2:

                return (
                    edge[0],
                    edge[1],
                    None,
                )

            if len(edge) == 3:

                return (
                    edge[0],
                    edge[1],
                    edge[2],
                )

        return None

    # ==========================================================
    # Numeric helpers
    # ==========================================================

    @staticmethod
    def _read_numeric(
        value,
    ):
        try:

            number = float(
                value
            )

            if not np.isfinite(
                number
            ):
                return None

            return number

        except (
            TypeError,
            ValueError,
        ):

            return None

    @staticmethod
    def _safe_float(
        value,
    ):
        number = LightningRoutingEnv._read_numeric(
            value
        )

        if number is None:
            return 0.0

        return number

    @staticmethod
    def _aggregate(
        values,
    ):
        if not values:
            return 0.0

        return float(
            np.mean(
                values
            )
        )

    @staticmethod
    def _normalize_value(
        value,
        scale,
    ):
        if scale <= 0.0:
            return 0.0

        return float(
            np.clip(
                value / scale,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _normalize_capacity(
        value,
    ):
        value = max(
            0.0,
            float(value),
        )

        return float(
            np.clip(
                np.log1p(value) / 20.0,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _normalize_liquidity(
        liquidity,
    ):
        """
        Normalize known directional liquidity.

        Unknown liquidity has already been represented as 0
        by the caller. Capacity is never supplied here as a
        fallback.
        """

        liquidity = max(
            0.0,
            float(liquidity),
        )

        normalized = (
            np.log1p(
                liquidity
            ) / 20.0
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    # ==========================================================
    # Render
    # ==========================================================

    def render(
        self,
    ):
        print(
            {
                "step": self.current_step,
                "source": self.source,
                "destination": self.destination,
                "amount": self.amount,
                "route": self.current_route,
                "route_edges": self.current_route_edges,
                "attempt_id": self.attempt_id,
            }
        )

    # ==========================================================
    # Close
    # ==========================================================

    def close(
        self,
    ):
        pass


# ==============================================================
# Backward-compatible alias
# ==============================================================

LightningEnv = LightningRoutingEnv


# ==============================================================
# Standalone integration test
# ==============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Small MultiDiGraph
    # ----------------------------------------------------------

    G = nx.MultiDiGraph()

    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("B", "E"),
        ("E", "D"),
    ]

    for index, (u, v) in enumerate(edges):

        G.add_edge(
            u,
            v,
            key=index,
            capacity=10000,
            balance_uv=10000,
            balance_vu=10000,
            fee_base_msat=1000,
            fee_proportional_millionths=1,
            delay=10,
            available=True,
        )

    # ----------------------------------------------------------
    # Exact route provider
    # ----------------------------------------------------------

    def test_route_provider(
        graph,
        source,
        destination,
        amount,
        eta,
        bucket=None,
        k_candidates=5,
    ):
        try:

            route = nx.shortest_path(
                graph,
                source,
                destination,
            )

        except nx.NetworkXNoPath:

            return {
                "route": None
            }

        route_edges = []

        for u, v in zip(
            route[:-1],
            route[1:],
        ):

            edge_data = graph.get_edge_data(
                u,
                v,
            )

            if not edge_data:
                raise RuntimeError(
                    f"Missing edge data for {u}->{v}"
                )

            # The test graph has exactly one channel
            # for every pair, so this is deterministic.
            key = next(
                iter(edge_data)
            )

            route_edges.append(
                {
                    "source": u,
                    "target": v,
                    "channel_key": key,
                    "data": dict(
                        edge_data[key]
                    ),
                }
            )

        return {
            "route": route,
            "route_edges": route_edges,
            "bucket_id": "TEST_BUCKET",
            "route_id": "TEST_ROUTE",
        }

    # ----------------------------------------------------------
    # Transaction
    # ----------------------------------------------------------

    transactions = [
        {
            "tx_id": "TX001",
            "source": "A",
            "destination": "D",
            "amount": 1000,
        }
    ]

    # ----------------------------------------------------------
    # Environment
    # ----------------------------------------------------------

    env = LightningRoutingEnv(
        graph=G,
        route_provider=test_route_provider,
        transactions=transactions,
        k_neighbors=15,
        feature_dim=5,
        gamma=0.99,
        max_steps=10,
        seed=42,
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
        observation.shape,
    )

    print(
        "Info:",
        info,
    )

    # ----------------------------------------------------------
    # Step
    # ----------------------------------------------------------

    action = np.array(
        [0.5],
        dtype=np.float32,
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        action
    )

    print("=" * 70)
    print("ENVIRONMENT STEP")
    print("=" * 70)

    print(
        "Reward:",
        reward,
    )

    print(
        "Terminated:",
        terminated,
    )

    print(
        "Truncated:",
        truncated,
    )

    print(
        "Info:",
        info,
    )

    env.close()