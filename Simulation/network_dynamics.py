"""
network_dynamics.py

Dynamic state evolution of a Lightning Network simulation.

Responsibilities
----------------
1. Check whether a directed channel can forward a payment.
2. Update directional channel balances after successful payment.
3. Record payment/edge history.
4. Simulate network state evolution between payment steps.
5. Recover temporarily unavailable nodes/channels.
6. Provide reset and current-state functionality.

Separation of responsibilities
-------------------------------
FailureModel:
    Evaluates failures during execution of a specific payment.

NetworkDynamics:
    Maintains and evolves the persistent network state.

Router / PPO:
    Selects and adapts routes.

Bucket / PartialBacktracker:
    Stores and uses alternative routes after failure.

OnionRouter:
    Handles the simulated forwarding/privacy layer.

The NetworkX graph G is the single source of truth.
"""


from collections import defaultdict
import random


class NetworkDynamics:

    def __init__(
        self,
        G,
        channel_failure_rate=0.01,
        node_failure_rate=0.005,
        recovery_rate=0.05,
        seed=42
    ):
        """
        Parameters
        ----------
        G : networkx.Graph / MultiGraph / DiGraph / MultiDiGraph
            Lightning Network graph.

        channel_failure_rate : float
            Probability that an available channel becomes
            temporarily unavailable during a network update.

        node_failure_rate : float
            Probability that an available node becomes
            temporarily unavailable during a network update.

        recovery_rate : float
            Probability that an unavailable node/channel
            becomes available during a network update.

        seed : int
            Random seed for reproducible simulation.
        """

        if not 0 <= channel_failure_rate <= 1:
            raise ValueError(
                "channel_failure_rate must be in [0, 1]."
            )

        if not 0 <= node_failure_rate <= 1:
            raise ValueError(
                "node_failure_rate must be in [0, 1]."
            )

        if not 0 <= recovery_rate <= 1:
            raise ValueError(
                "recovery_rate must be in [0, 1]."
            )

        self.G = G

        self.history = defaultdict(list)

        self.channel_failure_rate = float(
            channel_failure_rate
        )

        self.node_failure_rate = float(
            node_failure_rate
        )

        self.recovery_rate = float(
            recovery_rate
        )

        self.rng = random.Random(
            seed
        )

    # ============================================================
    # Check Forwarding Possibility
    # ============================================================

    def can_forward(
        self,
        u,
        v,
        key,
        amount
    ):
        """
        Check whether the exact directed channel u -> v
        can currently forward the requested amount.

        This is a deterministic state check.

        It does NOT generate a stochastic failure event.
        FailureModel is responsible for stochastic payment
        failure evaluation.
        """

        if amount <= 0:
            return False

        edge_data = self._get_edge(
            u,
            v,
            key
        )

        if edge_data is None:
            return False

        # --------------------------------------------------------
        # Channel availability
        # --------------------------------------------------------

        if not edge_data.get(
            "available",
            True
        ):
            return False

        # --------------------------------------------------------
        # Source / destination node availability
        # --------------------------------------------------------

        if not self._node_available(
            u
        ):
            return False

        if not self._node_available(
            v
        ):
            return False

        # --------------------------------------------------------
        # Channel capacity
        # --------------------------------------------------------

        capacity = edge_data.get(
            "capacity",
            None
        )

        if capacity is not None:

            try:

                if float(capacity) < amount:
                    return False

            except (
                TypeError,
                ValueError
            ):

                return False

        # --------------------------------------------------------
        # Directional liquidity
        # --------------------------------------------------------

        balance_uv = edge_data.get(
            "balance_uv",
            None
        )

        if balance_uv is not None:

            try:

                if float(balance_uv) < amount:
                    return False

            except (
                TypeError,
                ValueError
            ):

                return False

        return True

    # ============================================================
    # Record Transaction / Edge Result
    # ============================================================

    def record(
        self,
        tx_id,
        edge,
        success,
        reason=None
    ):
        """
        Record the result of forwarding through one edge.

        Parameters
        ----------
        tx_id : int / str
            Transaction identifier.

        edge : tuple
            (u, v, key)

        success : bool
            Whether forwarding succeeded.

        reason : str, optional
            Failure reason if forwarding failed.
        """

        parsed = self._parse_edge(
            edge
        )

        if parsed is None:
            return

        u, v, key = parsed

        data = self._get_edge(
            u,
            v,
            key
        )

        if data is None:
            return

        data.setdefault(
            "success_count",
            0
        )

        data.setdefault(
            "failure_count",
            0
        )

        if success:

            data["success_count"] += 1

        else:

            data["failure_count"] += 1

        self.history[
            tx_id
        ].append(
            {
                "edge": (
                    u,
                    v,
                    key
                ),
                "success": bool(
                    success
                ),
                "reason": reason
            }
        )

    # ============================================================
    # Record Complete Payment
    # ============================================================

    def record_payment(
        self,
        tx_id,
        result
    ):
        """
        Record a complete payment result.

        Expected result structure:

        {
            "success": bool,
            "reason": str,
            "visited_edges": [...]
        }

        This method does not modify balances.
        Settlement is handled separately by settle().
        """

        if result is None:
            return

        success = bool(
            result.get(
                "success",
                False
            )
        )

        reason = result.get(
            "reason"
        )

        visited_edges = result.get(
            "visited_edges",
            []
        )

        for edge in visited_edges:

            self.record(
                tx_id=tx_id,
                edge=edge,
                success=success,
                reason=reason
            )

    # ============================================================
    # Settle Successful Payment
    # ============================================================

    def settle(
        self,
        edge,
        amount
    ):
        """
        Update directional channel balances after successful
        forwarding.

        For a payment u -> v:

            balance_uv -= amount
            balance_vu += amount

        Settlement is performed only after successful payment
        execution.
        """

        if amount <= 0:
            raise ValueError(
                "amount must be positive."
            )

        parsed = self._parse_edge(
            edge
        )

        if parsed is None:
            raise ValueError(
                "Invalid edge format."
            )

        u, v, key = parsed

        data = self._get_edge(
            u,
            v,
            key
        )

        if data is None:
            raise ValueError(
                f"Edge {edge} does not exist."
            )

        # --------------------------------------------------------
        # Verify forwarding state before settlement
        # --------------------------------------------------------

        if not self.can_forward(
            u,
            v,
            key,
            amount
        ):

            raise ValueError(
                "Cannot settle payment: "
                "channel is unavailable or lacks liquidity."
            )

        # --------------------------------------------------------
        # Current directional balances
        # --------------------------------------------------------

        balance_uv = float(
            data.get(
                "balance_uv",
                0
            )
        )

        balance_vu = float(
            data.get(
                "balance_vu",
                0
            )
        )

        capacity = float(
            data.get(
                "capacity",
                balance_uv + balance_vu
            )
        )

        # --------------------------------------------------------
        # Update balances
        # --------------------------------------------------------

        new_balance_uv = max(
            0.0,
            balance_uv - amount
        )

        new_balance_vu = min(
            capacity,
            balance_vu + amount
        )

        data["balance_uv"] = (
            new_balance_uv
        )

        data["balance_vu"] = (
            new_balance_vu
        )

    # ============================================================
    # Settle Complete Route
    # ============================================================

    def settle_route(
        self,
        route_edges,
        amount
    ):
        """
        Settle all edges of a successfully executed route.

        Each edge is updated independently.

        Note:
        In a real Lightning implementation, HTLC state and
        atomic settlement are more complex. This method is a
        simulation abstraction.
        """

        if not route_edges:
            return False

        # --------------------------------------------------------
        # First validate the complete route
        # --------------------------------------------------------

        for edge in route_edges:

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:
                return False

            u, v, key = parsed

            if not self.can_forward(
                u,
                v,
                key,
                amount
            ):
                return False

        # --------------------------------------------------------
        # Apply settlement
        # --------------------------------------------------------

        for edge in route_edges:

            self.settle(
                edge,
                amount
            )

        return True

    # ============================================================
    # Network Evolution Step
    # ============================================================

    def update(self):
        """
        Advance the network by one simulation step.

        The update consists of:

            1. Node availability evolution
            2. Channel availability evolution

        This method does not perform routing and does not
        evaluate a payment.
        """

        self.update_nodes()

        self.update_channels()

    # ============================================================
    # Node Availability Evolution
    # ============================================================

    def update_nodes(self):
        """
        Evolve node availability.

        Available node:
            may become unavailable.

        Unavailable node:
            may recover.
        """

        for node, data in self.G.nodes(
            data=True
        ):

            available = data.get(
                "available",
                True
            )

            # ----------------------------------------------------
            # Active node -> possible failure
            # ----------------------------------------------------

            if available:

                if (
                    self.rng.random()
                    <
                    self.node_failure_rate
                ):

                    data["available"] = False

                    data["failure_count"] = (
                        data.get(
                            "failure_count",
                            0
                        )
                        + 1
                    )

            # ----------------------------------------------------
            # Failed node -> possible recovery
            # ----------------------------------------------------

            else:

                if (
                    self.rng.random()
                    <
                    self.recovery_rate
                ):

                    data["available"] = True

                    data["last_recovery"] = (
                        self._timestamp()
                    )

    # ============================================================
    # Channel Availability Evolution
    # ============================================================

    def update_channels(self):
        """
        Evolve channel availability.

        Available channel:
            may become unavailable.

        Unavailable channel:
            may recover.

        Existing failure statistics are preserved.
        """

        if self.G.is_multigraph():

            edges = self.G.edges(
                keys=True,
                data=True
            )

            for u, v, key, data in edges:

                self._update_channel(
                    data
                )

        else:

            edges = self.G.edges(
                data=True
            )

            for u, v, data in edges:

                self._update_channel(
                    data
                )

    # ============================================================
    # Single Channel Evolution
    # ============================================================

    def _update_channel(
        self,
        data
    ):
        """
        Update availability of one channel.
        """

        available = data.get(
            "available",
            True
        )

        if available:

            if (
                self.rng.random()
                <
                self.channel_failure_rate
            ):

                data["available"] = False

                data["failure_count"] = (
                    data.get(
                        "failure_count",
                        0
                    )
                    + 1
                )

                data["last_failure"] = (
                    self._timestamp()
                )

        else:

            if (
                self.rng.random()
                <
                self.recovery_rate
            ):

                data["available"] = True

                data["last_recovery"] = (
                    self._timestamp()
                )

    # ============================================================
    # Get Current Network State
    # ============================================================

    def get_state(self):
        """
        Return the current NetworkX graph.

        The graph itself is the simulation state.
        """

        return self.G

    # ============================================================
    # Get Usable Edges
    # ============================================================

    def get_usable_edges(
        self,
        amount
    ):
        """
        Return currently usable forwarding edges.

        This is useful for routing/pathfinding modules.

        Returns
        -------
        list
            List of (u, v, key) or (u, v).
        """

        usable = []

        if self.G.is_multigraph():

            for u, v, key, data in self.G.edges(
                keys=True,
                data=True
            ):

                if self.can_forward(
                    u,
                    v,
                    key,
                    amount
                ):

                    usable.append(
                        (
                            u,
                            v,
                            key
                        )
                    )

        else:

            for u, v, data in self.G.edges(
                data=True
            ):

                if self.can_forward(
                    u,
                    v,
                    None,
                    amount
                ):

                    usable.append(
                        (
                            u,
                            v
                        )
                    )

        return usable

    # ============================================================
    # Reset Simulation State
    # ============================================================

    def reset(
        self,
        reset_balances=False
    ):
        """
        Reset dynamic simulation state.

        Parameters
        ----------
        reset_balances : bool
            If True and initial balances were stored in the graph,
            restore them.

        Availability is reset to True.

        Failure/success counters are reset.

        History is cleared.
        """

        self.history.clear()

        # --------------------------------------------------------
        # Reset nodes
        # --------------------------------------------------------

        for node, data in self.G.nodes(
            data=True
        ):

            data["available"] = True

            data["failure_count"] = 0

            data["last_failure"] = None

            data["last_recovery"] = None

        # --------------------------------------------------------
        # Reset channels
        # --------------------------------------------------------

        if self.G.is_multigraph():

            edges = self.G.edges(
                keys=True,
                data=True
            )

            for u, v, key, data in edges:

                self._reset_edge(
                    data,
                    reset_balances
                )

        else:

            edges = self.G.edges(
                data=True
            )

            for u, v, data in edges:

                self._reset_edge(
                    data,
                    reset_balances
                )

    # ============================================================
    # Reset Single Edge
    # ============================================================

    @staticmethod
    def _reset_edge(
        data,
        reset_balances
    ):
        """
        Reset dynamic state of one channel.
        """

        data["available"] = True

        data["success_count"] = 0

        data["failure_count"] = 0

        data["last_failure"] = None

        data["last_recovery"] = None

        if reset_balances:

            if "initial_balance_uv" in data:

                data["balance_uv"] = (
                    data[
                        "initial_balance_uv"
                    ]
                )

            if "initial_balance_vu" in data:

                data["balance_vu"] = (
                    data[
                        "initial_balance_vu"
                    ]
                )

    # ============================================================
    # Node State Helper
    # ============================================================

    def _node_available(
        self,
        node
    ):
        """
        Check persistent node availability.
        """

        if node not in self.G.nodes:
            return False

        data = self.G.nodes[node]

        if data.get(
            "available",
            True
        ) is False:

            return False

        # Backward compatibility
        if data.get(
            "is_online",
            True
        ) is False:

            return False

        return True

    # ============================================================
    # Edge Access
    # ============================================================

    def _get_edge(
        self,
        u,
        v,
        key=None
    ):
        """
        Return exact edge data.
        """

        if not self.G.has_edge(
            u,
            v
        ):
            return None

        if self.G.is_multigraph():

            if key is not None:

                return self.G.get_edge_data(
                    u,
                    v,
                    key=key
                )

            edge_data = self.G.get_edge_data(
                u,
                v
            )

            if not edge_data:
                return None

            # Fallback for callers that do not provide a key.
            # Prefer an available channel.
            for edge_key, data in edge_data.items():

                if data.get(
                    "available",
                    True
                ):

                    return data

            # If none is currently available, return first edge.
            first_key = next(
                iter(edge_data)
            )

            return edge_data[
                first_key
            ]

        return self.G.get_edge_data(
            u,
            v
        )

    # ============================================================
    # Edge Parser
    # ============================================================

    @staticmethod
    def _parse_edge(
        edge
    ):
        """
        Normalize edge representation.

        Accepted:
            (u, v)
            (u, v, key)
        """

        if edge is None:
            return None

        if len(edge) == 2:

            u, v = edge

            return (
                u,
                v,
                None
            )

        if len(edge) == 3:

            u, v, key = edge

            return (
                u,
                v,
                key
            )

        return None

    # ============================================================
    # Timestamp Helper
    # ============================================================

    @staticmethod
    def _timestamp():
        """
        Return a lightweight simulation timestamp.

        Uses Unix time so the value can be serialized easily.
        """

        import time

        return time.time()