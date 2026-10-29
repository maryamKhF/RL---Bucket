"""
Simulation/network_dynamics.py

Dynamic state evolution of a Lightning Network simulation.

Responsibilities
----------------
1. Check whether a directed channel can forward a payment.
2. Update known directional channel balances after successful payment.
3. Record payment/edge history.
4. Simulate network state evolution between payment steps.
5. Recover temporarily unavailable nodes/channels.
6. Provide reset and current-state functionality.

Important liquidity semantics
-----------------------------
The snapshot ``capacity`` field is NOT directional liquidity.

Therefore:

    capacity != balance_uv
    capacity != balance_vu
    capacity / 2 != directional liquidity

If directional liquidity is explicitly present, it is validated.

If directional liquidity is absent from the public graph, a seeded,
simulator-only hidden balance is sampled from the configured structural
capacity prior. The sampled value is not copied into the public graph.
The router accepts unknown-liquidity channels structurally; forwarding checks
the private ledger during payment execution.

    - the exact channel exists,
    - the channel is available,
    - both endpoint nodes are available,
    - the requested amount is valid,
    - and a valid structural capacity bound exists, if supplied.

Separation of responsibilities
-------------------------------
FailureModel:
    Evaluates failures during execution of a specific payment.

NetworkDynamics:
    Maintains and evolves persistent network state.

Router / PPO:
    Selects and adapts routes.

Bucket / PartialBacktracker:
    Stores and uses alternative routes after failure.

OnionRouter:
    Handles the simulated forwarding/privacy layer.

The public NetworkX graph contains observable network attributes. Hidden
directional balances live only in this simulator's private ledger.
"""


from collections import defaultdict
import math
import random


class NetworkDynamics:

    def __init__(
        self,
        G,
        channel_failure_rate=0.01,
        node_failure_rate=0.005,
        recovery_rate=0.05,
        seed=42,
        default_capacity=None,
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

        if G is None:
            raise ValueError(
                "G must not be None."
            )

        self.G = G
        if default_capacity is None:
            self.default_capacity = None
        else:
            self.default_capacity = float(default_capacity)
            if not math.isfinite(self.default_capacity) or self.default_capacity < 0:
                raise ValueError("default_capacity must be finite and nonnegative")

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
        self.hidden_liquidity_rng = random.Random(int(seed) + 8_021)
        self.hidden_liquidity = {}
        self.initial_hidden_liquidity = {}
        self._initialize_hidden_liquidity()

    def _initialize_hidden_liquidity(self):
        """Create simulator-only channel balances from a seeded uniform prior.

        A channel's public capacity bounds the total balance. For reciprocal
        directed edges representing the same physical channel, the sampled
        directional balances sum to capacity. These values stay in this
        simulator object and are never copied into the observation graph.
        """
        groups = {}
        edges = (
            self.G.edges(keys=True, data=True)
            if self.G.is_multigraph()
            else ((u, v, None, data) for u, v, data in self.G.edges(data=True))
        )
        for u, v, key, data in edges:
            edge = (u, v, key)
            channel_id = str(data.get("channel_id", key))
            base_id = channel_id[:-4] if channel_id.endswith("-rev") else channel_id
            group_key = (base_id, frozenset((u, v)))
            groups.setdefault(group_key, []).append((edge, data, channel_id.endswith("-rev")))

        for members in groups.values():
            if len(members) == 2:
                first, second = members
                if first[0][:2] == second[0][:2][::-1]:
                    self._initialize_reciprocal_pair(first, second)
                    continue
            for edge, data, _is_reverse in members:
                balance = self._initial_directional_balance(data)
                if balance is not None:
                    self.hidden_liquidity[edge] = balance

        self.initial_hidden_liquidity = dict(self.hidden_liquidity)

    def _initial_directional_balance(self, data):
        """Return explicit simulated balance or sample from capacity."""
        value = data.get("balance_uv")
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value) and value >= 0:
                    return value
            except (TypeError, ValueError):
                pass
        capacity = self._read_capacity(data)
        maximum = (
            capacity["value"]
            if capacity["present"] and capacity["valid"]
            else self.default_capacity
        )
        if maximum is None:
            return None
        return maximum * self.hidden_liquidity_rng.random()

    def _initialize_reciprocal_pair(self, first, second):
        (edge_a, data_a, reverse_a), (edge_b, data_b, reverse_b) = first, second
        capacity_state = self._read_capacity(data_a)
        capacity = (
            capacity_state["value"]
            if capacity_state["present"] and capacity_state["valid"]
            else self.default_capacity
        )
        if capacity is None:
            balance_a = self._explicit_balance(data_a)
            balance_b = self._explicit_balance(data_b)
            if balance_a is not None:
                self.hidden_liquidity[edge_a] = balance_a
            if balance_b is not None:
                self.hidden_liquidity[edge_b] = balance_b
            return

        balance_a = self._explicit_balance(data_a)
        balance_b = self._explicit_balance(data_b)
        if balance_a is not None and balance_b is not None:
            self.hidden_liquidity[edge_a] = balance_a
            self.hidden_liquidity[edge_b] = balance_b
        elif balance_a is not None:
            self.hidden_liquidity[edge_a] = balance_a
            self.hidden_liquidity[edge_b] = max(0.0, capacity - balance_a)
        elif balance_b is not None:
            self.hidden_liquidity[edge_b] = balance_b
            self.hidden_liquidity[edge_a] = max(0.0, capacity - balance_b)
        else:
            balance = capacity * self.hidden_liquidity_rng.random()
            self.hidden_liquidity[edge_a] = balance
            self.hidden_liquidity[edge_b] = capacity - balance

    @staticmethod
    def _explicit_balance(data):
        value = data.get("balance_uv")
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value >= 0 else None

    def _reverse_edge(self, edge):
        u, v, key = edge
        data = self._get_edge(u, v, key)
        if data is None:
            return None
        channel_id = str(data.get("channel_id", key))
        base_id = channel_id[:-4] if channel_id.endswith("-rev") else channel_id
        if self.G.is_multigraph():
            candidates = self.G.get_edge_data(v, u, default={})
            for reverse_key, reverse_data in candidates.items():
                reverse_id = str(reverse_data.get("channel_id", reverse_key))
                reverse_base = reverse_id[:-4] if reverse_id.endswith("-rev") else reverse_id
                if reverse_base == base_id:
                    return (v, u, reverse_key)
        elif self.G.has_edge(v, u):
            return (v, u, None)
        return None

    # ============================================================
    # Check Forwarding Possibility
    # ============================================================

    def can_forward(
        self,
        u,
        v,
        key,
        amount,
    ):
        """
        Check whether the exact directed channel u -> v
        can currently forward the requested amount.

        This is a deterministic state check.

        It does NOT generate a stochastic failure event.
        FailureModel is responsible for stochastic payment
        failure evaluation.

        Liquidity semantics
        -------------------
        Public capacity is checked structurally; hidden directional balance
        is checked from this simulator's private ledger.

        ``capacity`` is NEVER interpreted as directional liquidity.
        """

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return False

        edge_data = self._get_edge(
            u,
            v,
            key,
        )

        if edge_data is None:
            return False

        # --------------------------------------------------------
        # Channel availability
        # --------------------------------------------------------

        if not self._channel_is_available(
            edge_data
        ):
            return False

        # --------------------------------------------------------
        # Source / destination node availability
        # --------------------------------------------------------

        if not self._node_available(u):
            return False

        if not self._node_available(v):
            return False

        # --------------------------------------------------------
        # Structural channel capacity
        #
        # Capacity is NOT directional liquidity.
        #
        # It is only used as a structural upper bound when
        # the snapshot provides a valid capacity value.
        # --------------------------------------------------------

        capacity_state = self._read_capacity(
            edge_data
        )

        if capacity_state["present"]:

            if capacity_state["valid"] is False:
                return False

            if capacity_state["value"] < amount_value:
                return False

        # Simulator-only balances are hidden from routing and observation.
        edge_identity = (u, v, key)
        if edge_identity in self.hidden_liquidity:
            hidden_balance = self.hidden_liquidity[edge_identity]
            if hidden_balance is not None and hidden_balance < amount_value:
                return False
            return True

        # --------------------------------------------------------
        # Explicit directional liquidity
        #
        # If balance_uv exists, it is actual directional
        # liquidity and must be respected.
        #
        # If it does not exist, the liquidity state is unknown
        # and MUST NOT be reconstructed from capacity.
        # --------------------------------------------------------

        liquidity_state = self._read_directional_liquidity(
            edge_data
        )

        if liquidity_state["malformed"]:
            return False

        if liquidity_state["known"]:

            if (
                liquidity_state["value"]
                < amount_value
            ):
                return False

        # --------------------------------------------------------
        # Unknown directional liquidity is accepted structurally.
        # --------------------------------------------------------

        return True

    # ============================================================
    # Record Transaction / Edge Result
    # ============================================================

    def record(
        self,
        tx_id,
        edge,
        success,
        reason=None,
    ):
        """
        Record the result of forwarding through one edge.

        Parameters
        ----------
        tx_id : int / str
            Transaction identifier.

        edge : tuple / list / dict
            Supported representations include:

                (u, v)
                (u, v, key)
                {
                    "source": u,
                    "target": v,
                    "channel_key": key
                }

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
            key,
        )

        if data is None:
            return

        data.setdefault(
            "success_count",
            0,
        )

        data.setdefault(
            "failure_count",
            0,
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
                    key,
                ),
                "success": bool(
                    success
                ),
                "reason": reason,
            }
        )

    # ============================================================
    # Record Complete Payment
    # ============================================================

    def record_payment(
        self,
        tx_id,
        result,
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
                False,
            )
        )

        reason = result.get(
            "reason"
        )

        visited_edges = result.get(
            "visited_edges",
            [],
        )

        for edge in visited_edges:

            self.record(
                tx_id=tx_id,
                edge=edge,
                success=success,
                reason=reason,
            )

    # ============================================================
    # Settle Successful Payment
    # ============================================================

    def settle(
        self,
        edge,
        amount,
    ):
        """
        Update directional channel state after successful
        forwarding.

        For a channel with known directional balances:

            balance_uv -= amount
            balance_vu += amount

        If a private hidden balance exists, update it and the reverse
        direction in the private ledger. Never write that state to the public
        graph. Capacity bounds the configured simulator prior; it is not
        itself treated as directional liquidity.
        """

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            raise ValueError(
                "amount must be positive and finite."
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
            key,
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
            amount_value,
        ):
            raise ValueError(
                "Cannot settle payment: "
                "channel is unavailable or "
                "cannot structurally forward the amount."
            )

        edge_identity = (u, v, key)
        if edge_identity in self.hidden_liquidity:
            balance = self.hidden_liquidity[edge_identity]
            if balance is None or balance < amount_value:
                raise ValueError(
                    "Cannot settle payment: insufficient hidden directional liquidity."
                )
            self.hidden_liquidity[edge_identity] = balance - amount_value
            reverse = self._reverse_edge(edge_identity)
            if reverse in self.hidden_liquidity:
                reverse_balance = self.hidden_liquidity[reverse]
                if reverse_balance is not None:
                    capacity_state = self._read_capacity(
                        self._get_edge(*reverse)
                    )
                    capacity = (
                        capacity_state["value"]
                        if capacity_state["present"] and capacity_state["valid"]
                        else (
                            self.default_capacity
                            if self.default_capacity is not None
                            else float("inf")
                        )
                    )
                    self.hidden_liquidity[reverse] = min(
                        capacity, reverse_balance + amount_value
                    )
            return True

        # --------------------------------------------------------
        # Explicit directional balance state
        # --------------------------------------------------------

        balance_state = self._read_balance_pair(
            data
        )

        if balance_state["malformed"]:
            raise ValueError(
                "Malformed directional balance state."
            )

        # --------------------------------------------------------
        # Case 1:
        # Explicit balance_uv exists.
        #
        # This is the only case where outgoing directional
        # liquidity is actually known.
        # --------------------------------------------------------

        if balance_state["uv_known"]:

            balance_uv = balance_state[
                "balance_uv"
            ]

            if balance_uv < amount_value:
                raise ValueError(
                    "Cannot settle payment: "
                    "insufficient directional liquidity."
                )

            new_balance_uv = (
                balance_uv
                -
                amount_value
            )

            data["balance_uv"] = (
                new_balance_uv
            )

            # ----------------------------------------------------
            # If reverse directional balance is also explicitly
            # known, update it.
            #
            # If it is unknown, DO NOT invent it.
            # ----------------------------------------------------

            if balance_state["vu_known"]:

                balance_vu = balance_state[
                    "balance_vu"
                ]

                capacity_state = self._read_capacity(
                    data
                )

                if (
                    capacity_state["present"]
                    and
                    capacity_state["valid"]
                ):

                    capacity = capacity_state[
                        "value"
                    ]

                    new_balance_vu = min(
                        capacity,
                        balance_vu
                        +
                        amount_value,
                    )

                else:

                    new_balance_vu = (
                        balance_vu
                        +
                        amount_value
                    )

                data["balance_vu"] = (
                    new_balance_vu
                )

            return True

        # --------------------------------------------------------
        # Case 2:
        # Directional liquidity is unknown.
        #
        # IMPORTANT:
        # Do not create balance_uv or balance_vu.
        #
        # The simulation records that settlement occurred, while
        # preserving the unknown-liquidity semantics of the raw
        # snapshot.
        # --------------------------------------------------------

        data["settlement_count"] = (
            data.get(
                "settlement_count",
                0,
            )
            + 1
        )

        data["settled_amount"] = (
            self._safe_float(
                data.get(
                    "settled_amount",
                    0.0,
                ),
                default=0.0,
            )
            +
            amount_value
        )

        return True

    # ============================================================
    # Settle Complete Route
    # ============================================================

    def settle_route(
        self,
        route_edges,
        amount,
    ):
        """
        Settle all edges of a successfully executed route.

        The complete route is validated before any state change.

        Each exact channel is preserved.

        Note:
        In a real Lightning implementation, HTLC state and atomic
        settlement are considerably more complex. This method is
        a simulation abstraction.
        """

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return False

        if not route_edges:
            return False

        parsed_edges = []

        # --------------------------------------------------------
        # Parse and validate the complete route first.
        # --------------------------------------------------------

        for edge in route_edges:

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:
                return False

            u, v, key = parsed

            if self._is_multigraph():

                if key is None:
                    return False

            if not self.can_forward(
                u,
                v,
                key,
                amount_value,
            ):
                return False

            parsed_edges.append(
                (
                    u,
                    v,
                    key,
                )
            )

        # --------------------------------------------------------
        # Apply settlement only after every edge passed validation.
        # --------------------------------------------------------

        for edge in parsed_edges:

            self.settle(
                edge,
                amount_value,
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
                True,
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
                            0,
                        )
                        + 1
                    )

                    data["last_failure"] = (
                        self._timestamp()
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
                data=True,
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
        data,
    ):
        """
        Update availability of one channel.
        """

        if not isinstance(data, dict):
            return

        available = data.get(
            "available",
            True,
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
                        0,
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
        amount,
    ):
        """
        Return currently usable forwarding edges.

        Returns
        -------
        list
            For MultiGraph / MultiDiGraph:

                [(u, v, key), ...]

            For Graph / DiGraph:

                [(u, v), ...]
        """

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return []

        usable = []

        if self.G.is_multigraph():

            for u, v, key, data in self.G.edges(
                keys=True,
                data=True,
            ):

                if self.can_forward(
                    u,
                    v,
                    key,
                    amount_value,
                ):

                    usable.append(
                        (
                            u,
                            v,
                            key,
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
                    amount_value,
                ):

                    usable.append(
                        (
                            u,
                            v,
                        )
                    )

        return usable

    # ============================================================
    # Reset Simulation State
    # ============================================================

    def reset(
        self,
        reset_balances=False,
    ):
        """
        Reset dynamic simulation state.

        Parameters
        ----------
        reset_balances : bool
            If True, restore the simulator's initial hidden ledger and any
            explicit initial balances stored on graph edges.

        Availability is reset to True.

        Failure/success counters are reset.

        Settlement counters for unknown-liquidity channels are
        also reset.

        History is cleared.
        """

        self.history.clear()

        if reset_balances:
            self.hidden_liquidity.clear()
            self.hidden_liquidity.update(self.initial_hidden_liquidity)

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
                data=True,
            )

            for u, v, key, data in edges:

                self._reset_edge(
                    data,
                    reset_balances,
                )

        else:

            edges = self.G.edges(
                data=True
            )

            for u, v, data in edges:

                self._reset_edge(
                    data,
                    reset_balances,
                )

    # ============================================================
    # Reset Single Edge
    # ============================================================

    @staticmethod
    def _reset_edge(
        data,
        reset_balances,
    ):
        """
        Reset dynamic state of one channel.
        """

        if not isinstance(data, dict):
            return

        data["available"] = True

        data["success_count"] = 0

        data["failure_count"] = 0

        data["last_failure"] = None

        data["last_recovery"] = None

        data["settlement_count"] = 0

        data["settled_amount"] = 0.0

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
        node,
    ):
        """
        Check persistent node availability.
        """

        try:

            if node not in self.G.nodes:
                return False

            data = self.G.nodes[node]

        except Exception:

            return False

        if not isinstance(data, dict):
            return False

        if data.get(
            "available",
            True,
        ) is False:
            return False

        # Backward compatibility
        if data.get(
            "is_online",
            True,
        ) is False:
            return False

        if data.get(
            "online",
            True,
        ) is False:
            return False

        return True

    # ============================================================
    # Graph Type Helper
    # ============================================================

    def _is_multigraph(
        self,
    ):
        """
        Return whether the underlying NetworkX graph is a
        MultiGraph or MultiDiGraph.

        This helper is used where exact channel identity is
        required, especially during complete-route settlement.

        The NetworkX graph remains authoritative for public topology and
        channel identity; hidden balances are kept in the private ledger.
        """

        try:

            return bool(
                self.G.is_multigraph()
            )

        except Exception:

            return False

    # ============================================================
    # Edge Access
    # ============================================================

    def _get_edge(
        self,
        u,
        v,
        key=None,
    ):
        """
        Return exact edge data.

        For MultiDiGraph:
            key is required for exact channel access.

        If key is omitted, a backward-compatible available-channel
        lookup is retained for legacy callers.

        Routing/payment code should provide the exact channel key.
        """

        try:

            if not self.G.has_edge(
                u,
                v,
            ):
                return None

            if self.G.is_multigraph():

                # ------------------------------------------------
                # Exact channel access.
                # ------------------------------------------------

                if key is not None:

                    edge_data = self.G.get_edge_data(
                        u,
                        v,
                        key=key,
                    )

                    return edge_data

                # ------------------------------------------------
                # Legacy key-less lookup.
                #
                # This path is intentionally not used when exact
                # channel identity is available.
                # ------------------------------------------------

                edge_data = self.G.get_edge_data(
                    u,
                    v,
                )

                if not isinstance(
                    edge_data,
                    dict,
                ):
                    return None

                for edge_key, data in edge_data.items():

                    if self._channel_is_available(
                        data
                    ):

                        return data

                return None

            return self.G.get_edge_data(
                u,
                v,
            )

        except Exception:

            return None

    # ============================================================
    # Edge Parser
    # ============================================================

    @staticmethod
    def _parse_edge(
        edge,
    ):
        """
        Normalize edge representation.

        Accepted:

            (u, v)
            (u, v, key)

        Also supports Top-K / Bucket dictionaries:

            {
                "source": u,
                "target": v,
                "channel_key": key
            }

        and:

            {
                "source": u,
                "target": v,
                "key": key
            }
        """

        if edge is None:
            return None

        # --------------------------------------------------------
        # Dictionary edge representation
        # --------------------------------------------------------

        if isinstance(edge, dict):

            if (
                "source" not in edge
                or
                "target" not in edge
            ):
                return None

            u = edge.get(
                "source"
            )

            v = edge.get(
                "target"
            )

            key = edge.get(
                "channel_key",
                None,
            )

            if key is None:

                key = edge.get(
                    "key",
                    None,
                )

            return (
                u,
                v,
                key,
            )

        # --------------------------------------------------------
        # Tuple / list edge representation
        # --------------------------------------------------------

        if not isinstance(
            edge,
            (list, tuple),
        ):
            return None

        if len(edge) == 2:

            u, v = edge

            return (
                u,
                v,
                None,
            )

        if len(edge) == 3:

            u, v, key = edge

            return (
                u,
                v,
                key,
            )

        return None

    # ============================================================
    # Validate Amount
    # ============================================================

    @staticmethod
    def _validate_amount(
        amount,
    ):
        """
        Validate a payment amount.
        """

        if isinstance(
            amount,
            bool,
        ):
            return None

        try:

            value = float(
                amount
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if not math.isfinite(
            value
        ):
            return None

        if value <= 0:
            return None

        return value

    # ============================================================
    # Channel Availability Helper
    # ============================================================

    @staticmethod
    def _channel_is_available(
        data,
    ):
        """
        Return whether a channel is currently available.
        """

        if not isinstance(
            data,
            dict,
        ):
            return False

        if data.get(
            "available",
            True,
        ) is False:
            return False

        return True

    # ============================================================
    # Capacity Reader
    # ============================================================

    @staticmethod
    def _read_capacity(
        data,
    ):
        """
        Read channel capacity without interpreting it as
        directional liquidity.

        Returns
        -------
        dict
            {
                "present": bool,
                "valid": bool,
                "value": float | None
            }
        """

        if not isinstance(
            data,
            dict,
        ):

            return {
                "present": False,
                "valid": False,
                "value": None,
            }

        if "capacity" not in data:

            return {
                "present": False,
                "valid": True,
                "value": None,
            }

        value = data.get(
            "capacity"
        )

        if value is None:

            return {
                "present": False,
                "valid": True,
                "value": None,
            }

        if isinstance(
            value,
            bool,
        ):

            return {
                "present": True,
                "valid": False,
                "value": None,
            }

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return {
                "present": True,
                "valid": False,
                "value": None,
            }

        if not math.isfinite(
            value
        ):

            return {
                "present": True,
                "valid": False,
                "value": None,
            }

        if value < 0:

            return {
                "present": True,
                "valid": False,
                "value": None,
            }

        return {
            "present": True,
            "valid": True,
            "value": value,
        }

    # ============================================================
    # Directional Liquidity Reader
    # ============================================================

    @staticmethod
    def _read_directional_liquidity(
        data,
    ):
        """
        Read explicit directional liquidity.

        Priority
        --------
        1. balance_uv
        2. liquidity_uv
        3. liquidity
        4. estimated_liquidity

        If none exists, directional liquidity is UNKNOWN.

        Unknown is not the same as zero.

        Returns
        -------
        dict
            {
                "known": bool,
                "malformed": bool,
                "value": float | None,
                "field": str | None
            }
        """

        if not isinstance(
            data,
            dict,
        ):

            return {
                "known": False,
                "malformed": True,
                "value": None,
                "field": None,
            }

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            if field not in data:
                continue

            value = data.get(
                field
            )

            # Explicitly unknown.
            if value is None:
                continue

            if isinstance(
                value,
                bool,
            ):

                return {
                    "known": False,
                    "malformed": True,
                    "value": None,
                    "field": field,
                }

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                return {
                    "known": False,
                    "malformed": True,
                    "value": None,
                    "field": field,
                }

            if not math.isfinite(
                value
            ):

                return {
                    "known": False,
                    "malformed": True,
                    "value": None,
                    "field": field,
                }

            if value < 0:

                return {
                    "known": False,
                    "malformed": True,
                    "value": None,
                    "field": field,
                }

            return {
                "known": True,
                "malformed": False,
                "value": value,
                "field": field,
            }

        # --------------------------------------------------------
        # No directional liquidity information exists.
        # --------------------------------------------------------

        return {
            "known": False,
            "malformed": False,
            "value": None,
            "field": None,
        }

    # ============================================================
    # Directional Balance Pair Reader
    # ============================================================

    @staticmethod
    def _read_balance_pair(
        data,
    ):
        """
        Read explicit balance_uv / balance_vu values.

        Missing values remain UNKNOWN.

        They are never reconstructed from capacity.
        """

        if not isinstance(
            data,
            dict,
        ):

            return {
                "uv_known": False,
                "vu_known": False,
                "malformed": True,
                "balance_uv": None,
                "balance_vu": None,
            }

        uv_present = (
            "balance_uv" in data
            and
            data.get("balance_uv") is not None
        )

        vu_present = (
            "balance_vu" in data
            and
            data.get("balance_vu") is not None
        )

        balance_uv = None
        balance_vu = None

        if uv_present:

            value = data.get(
                "balance_uv"
            )

            if isinstance(
                value,
                bool,
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

            try:

                balance_uv = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

            if (
                not math.isfinite(
                    balance_uv
                )
                or
                balance_uv < 0
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

        if vu_present:

            value = data.get(
                "balance_vu"
            )

            if isinstance(
                value,
                bool,
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

            try:

                balance_vu = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

            if (
                not math.isfinite(
                    balance_vu
                )
                or
                balance_vu < 0
            ):

                return {
                    "uv_known": False,
                    "vu_known": False,
                    "malformed": True,
                    "balance_uv": None,
                    "balance_vu": None,
                }

        return {
            "uv_known": uv_present,
            "vu_known": vu_present,
            "malformed": False,
            "balance_uv": balance_uv,
            "balance_vu": balance_vu,
        }

    # ============================================================
    # Safe Float Helper
    # ============================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0,
    ):
        """
        Convert a value to a finite float.

        Invalid values return default.
        """

        if isinstance(
            value,
            bool,
        ):

            return float(
                default
            )

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return float(
                default
            )

        if not math.isfinite(
            value
        ):

            return float(
                default
            )

        return value

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
