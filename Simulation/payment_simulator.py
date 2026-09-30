"""
Simulation/payment_simulator.py

Execute a selected payment route over a simulated
Lightning Network.

Responsibilities
----------------
1. Validate the selected route.
2. Evaluate forwarding through FailureModel.
3. Calculate payment/routing metrics.
4. Commit successful settlement through NetworkDynamics.
5. Return structured PaymentResult.
6. Provide exact failure information required by
   Partial Backtracking.

Fee handling
------------
The simulator supports both normalized attributes:

    fee_base
    fee_rate

and native Lightning/GML attributes:

    fee_base_msat
    fee_proportional_millionths

This keeps PaymentSimulator compatible with both the
prepared graph and the original Lightning GML snapshot.
"""

import time


# ============================================================
# Geographic Helper
# ============================================================

def estimate_inter_continent(
    G,
    u,
    v
):
    """
    Estimate whether a hop crosses continents.

    This is a geographic abstraction used only for
    simulation metrics.
    """

    if u not in G.nodes or v not in G.nodes:
        return 0

    source = G.nodes[u]
    target = G.nodes[v]

    latitude_difference = abs(
        source.get(
            "latitude",
            0
        )
        -
        target.get(
            "latitude",
            0
        )
    )

    longitude_difference = abs(
        source.get(
            "longitude",
            0
        )
        -
        target.get(
            "longitude",
            0
        )
    )

    return int(
        latitude_difference > 10
        and
        longitude_difference > 45
    )


# ============================================================
# Payment Result
# ============================================================

class PaymentResult:
    """
    Structured result of a simulated Lightning payment.

    The result contains both successful-payment metrics and
    failure information required by Partial Backtracking.
    """

    def __init__(
        self,
        success,
        path,
        edges,
        fee=0.0,
        delay=0.0,
        carbon=0.0,
        inter_country_hops=0,
        inter_continent_hops=0,
        reason=None,
        elapsed=0.0,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None
    ):

        self.success = bool(
            success
        )

        self.path = (
            path
            if path is not None
            else []
        )

        self.edges = (
            edges
            if edges is not None
            else []
        )

        self.fee = float(
            fee
        )

        self.delay = float(
            delay
        )

        self.carbon = float(
            carbon
        )

        self.inter_country_hops = int(
            inter_country_hops
        )

        self.inter_continent_hops = int(
            inter_continent_hops
        )

        self.reason = reason

        self.elapsed = float(
            elapsed
        )

        # ----------------------------------------------------
        # Failure information
        # ----------------------------------------------------

        self.failed_node = (
            failed_node
        )

        self.failed_edge = (
            failed_edge
        )

        self.failure_index = (
            failure_index
        )

        self.visited_edges = (
            visited_edges
            if visited_edges is not None
            else []
        )

    # ========================================================
    # Dictionary Representation
    # ========================================================

    def to_dict(self):

        return {
            "success": self.success,

            "path": self.path,

            "edges": self.edges,

            "fee": self.fee,

            "delay": self.delay,

            "carbon": self.carbon,

            "inter_country_hops":
                self.inter_country_hops,

            "inter_continent_hops":
                self.inter_continent_hops,

            "reason": self.reason,

            "elapsed": self.elapsed,

            "failed_node":
                self.failed_node,

            "failed_edge":
                self.failed_edge,

            "failure_index":
                self.failure_index,

            "visited_edges":
                self.visited_edges
        }

    # ========================================================
    # Representation
    # ========================================================

    def __repr__(self):

        return (
            "PaymentResult("
            f"success={self.success}, "
            f"reason={self.reason}, "
            f"fee={self.fee:.4f}, "
            f"delay={self.delay:.4f}, "
            f"failed_edge={self.failed_edge}, "
            f"failure_index={self.failure_index}"
            ")"
        )


# ============================================================
# Payment Simulator
# ============================================================

class PaymentSimulator:
    """
    Execute payments over the current Lightning graph.

    Parameters
    ----------
    G : NetworkX graph
        Lightning Network graph.

    failure_model : FailureModel
        Runtime payment failure evaluator.

    network_dynamics : NetworkDynamics
        Network state and settlement manager.
    """

    def __init__(
        self,
        G,
        failure_model,
        network_dynamics
    ):

        if G is None:
            raise ValueError(
                "G cannot be None."
            )

        if failure_model is None:
            raise ValueError(
                "failure_model cannot be None."
            )

        if network_dynamics is None:
            raise ValueError(
                "network_dynamics cannot be None."
            )

        self.G = G

        self.failure_model = (
            failure_model
        )

        self.network_dynamics = (
            network_dynamics
        )

    # ========================================================
    # Execute Payment
    # ========================================================

    def simulate_payment(
        self,
        path,
        edges,
        amount,
        tx_id=None
    ):
        """
        Execute one selected payment route.

        Parameters
        ----------
        path : list
            Ordered node path.

        edges : list
            Exact route edges.

            Example:
                [
                    (u, v, key),
                    (v, w, key)
                ]

        amount : float
            Payment amount.

        tx_id : int / str, optional
            Transaction identifier.

        Returns
        -------
        PaymentResult
        """

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if amount <= 0:

            return self._result(
                success=False,
                path=path,
                edges=edges,
                reason="invalid_amount",
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

        if not path or not edges:

            return self._result(
                success=False,
                path=path or [],
                edges=edges or [],
                reason="no_path",
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

        # ----------------------------------------------------
        # Validate path / edge consistency
        # ----------------------------------------------------

        validation = self._validate_route(
            path,
            edges
        )

        if not validation["valid"]:

            return self._result(
                success=False,
                path=path,
                edges=edges,
                reason=validation["reason"],
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

        # ----------------------------------------------------
        # Calculate static route metrics
        # ----------------------------------------------------

        metrics = self._calculate_metrics(
            edges,
            amount
        )

        # ----------------------------------------------------
        # Evaluate payment failure
        # ----------------------------------------------------

        failure = (
            self.failure_model
            .evaluate_payment_failure(
                route=path,
                amount=amount,
                network=self.G,
                route_edges=edges
            )
        )

        # ----------------------------------------------------
        # Payment failed
        # ----------------------------------------------------

        if not failure["success"]:

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                fee=metrics["fee"],
                delay=metrics["delay"],
                carbon=metrics["carbon"],
                inter_country_hops=(
                    metrics[
                        "inter_country_hops"
                    ]
                ),
                inter_continent_hops=(
                    metrics[
                        "inter_continent_hops"
                    ]
                ),
                reason=failure["reason"],
                failed_node=failure.get(
                    "failed_node"
                ),
                failed_edge=failure.get(
                    "failed_edge"
                ),
                failure_index=failure.get(
                    "failure_index"
                ),
                visited_edges=failure.get(
                    "visited_edges",
                    []
                ),
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

            # ----------------------------------------------
            # Record failure
            # ----------------------------------------------

            if tx_id is not None:

                self.network_dynamics.record_payment(
                    tx_id,
                    result.to_dict()
                )

            return result

        # ----------------------------------------------------
        # Successful payment
        # ----------------------------------------------------

        settlement_success = (
            self.network_dynamics
            .settle_route(
                route_edges=edges,
                amount=amount
            )
        )

        if not settlement_success:

            result = self._result(
                success=False,
                path=path,
                edges=edges,
                fee=metrics["fee"],
                delay=metrics["delay"],
                carbon=metrics["carbon"],
                inter_country_hops=(
                    metrics[
                        "inter_country_hops"
                    ]
                ),
                inter_continent_hops=(
                    metrics[
                        "inter_continent_hops"
                    ]
                ),
                reason="settlement_failed",
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

            if tx_id is not None:

                self.network_dynamics.record_payment(
                    tx_id,
                    result.to_dict()
                )

            return result

        # ----------------------------------------------------
        # Successful result
        # ----------------------------------------------------

        result = self._result(
            success=True,
            path=path,
            edges=edges,
            fee=metrics["fee"],
            delay=metrics["delay"],
            carbon=metrics["carbon"],
            inter_country_hops=(
                metrics[
                    "inter_country_hops"
                ]
            ),
            inter_continent_hops=(
                metrics[
                    "inter_continent_hops"
                ]
            ),
            reason="success",
            visited_edges=list(
                edges
            ),
            elapsed=(
                time.perf_counter()
                -
                start_time
            )
        )

        # ----------------------------------------------------
        # Record successful payment
        # ----------------------------------------------------

        if tx_id is not None:

            self.network_dynamics.record_payment(
                tx_id,
                result.to_dict()
            )

        return result

    # ========================================================
    # Route Validation
    # ========================================================

    def _validate_route(
        self,
        path,
        edges
    ):
        """
        Validate consistency between node path and
        exact edge sequence.

        Example:

            path:
                A -> B -> C

            edges:
                (A, B, k1)
                (B, C, k2)
        """

        if len(path) != len(edges) + 1:

            return {
                "valid": False,
                "reason": "route_edge_mismatch"
            }

        for i, edge in enumerate(
            edges
        ):

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                return {
                    "valid": False,
                    "reason": "invalid_edge"
                }

            u, v, key = parsed

            if u != path[i]:

                return {
                    "valid": False,
                    "reason": "invalid_route_source"
                }

            if v != path[i + 1]:

                return {
                    "valid": False,
                    "reason": "invalid_route_destination"
                }

            if not self.G.has_edge(
                u,
                v
            ):

                return {
                    "valid": False,
                    "reason": "missing_channel"
                }

            if (
                self.G.is_multigraph()
                and
                key is not None
            ):

                if not self.G.has_edge(
                    u,
                    v,
                    key
                ):

                    return {
                        "valid": False,
                        "reason": "missing_channel"
                    }

        return {
            "valid": True,
            "reason": None
        }

    # ========================================================
    # Route Metrics
    # ========================================================

    def _calculate_metrics(
        self,
        edges,
        amount
    ):
        """
        Calculate payment metrics without modifying
        network state.

        Metrics:
            - fee
            - delay
            - carbon
            - inter-country hops
            - inter-continent hops
        """

        fee = 0.0

        delay = 0.0

        carbon = 0.0

        inter_country = 0

        inter_continent = 0

        for edge in edges:

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:
                continue

            u, v, key = parsed

            data = self._get_edge(
                u,
                v,
                key
            )

            if data is None:
                continue

            # ------------------------------------------------
            # Fee
            # ------------------------------------------------

            fee_base, fee_rate = (
                self._get_fee_components(
                    data
                )
            )

            fee += (
                fee_base
                +
                fee_rate
                *
                float(amount)
                /
                1_000_000.0
            )

            # ------------------------------------------------
            # Delay
            # ------------------------------------------------

            delay_value = data.get(
                "delay",
                data.get(
                    "cltv_expiry_delta",
                    0
                )
            )

            try:

                delay += float(
                    delay_value
                )

            except (
                TypeError,
                ValueError
            ):

                pass

            # ------------------------------------------------
            # Geographic metrics
            # ------------------------------------------------

            source = self.G.nodes[u]

            target = self.G.nodes[v]

            source_carbon = self._safe_float(
                source.get(
                    "carbon_intensity",
                    0
                )
            )

            target_carbon = self._safe_float(
                target.get(
                    "carbon_intensity",
                    0
                )
            )

            carbon += (
                source_carbon
                +
                target_carbon
            ) / 2.0

            inter_country += int(
                source.get(
                    "country",
                    ""
                )
                !=
                target.get(
                    "country",
                    ""
                )
            )

            inter_continent += (
                estimate_inter_continent(
                    self.G,
                    u,
                    v
                )
            )

        average_carbon = (
            carbon
            /
            max(
                1,
                len(edges)
            )
        )

        return {
            "fee": fee,
            "delay": delay,
            "carbon": average_carbon,
            "inter_country_hops":
                inter_country,
            "inter_continent_hops":
                inter_continent
        }

    # ========================================================
    # Fee Extraction
    # ========================================================

    @staticmethod
    def _get_fee_components(
        data
    ):
        """
        Extract the two Lightning fee components.

        Supported normalized representation:

            fee_base
            fee_rate

        Supported native Lightning/GML representation:

            fee_base_msat
            fee_proportional_millionths

        Returns
        -------
        tuple
            (base_fee, proportional_rate)

        Important
        ---------
        The units are kept consistent with the existing
        Pathfinding implementation.

        Therefore:

            fee =
                base
                +
                rate * amount / 1,000,000
        """

        if not data:
            return 0.0, 0.0

        # ----------------------------------------------------
        # Base fee
        # ----------------------------------------------------

        if "fee_base" in data:

            base = data.get(
                "fee_base",
                0.0
            )

        else:

            base = data.get(
                "fee_base_msat",
                0.0
            )

        # ----------------------------------------------------
        # Proportional fee
        # ----------------------------------------------------

        if "fee_rate" in data:

            rate = data.get(
                "fee_rate",
                0.0
            )

        else:

            rate = data.get(
                "fee_proportional_millionths",
                0.0
            )

        base = PaymentSimulator._safe_float(
            base
        )

        rate = PaymentSimulator._safe_float(
            rate
        )

        return base, rate

    # ========================================================
    # Safe Float
    # ========================================================

    @staticmethod
    def _safe_float(
        value,
        default=0.0
    ):
        """
        Safely convert a value to float.
        """

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return float(
                default
            )

    # ========================================================
    # Result Factory
    # ========================================================

    @staticmethod
    def _result(
        success,
        path,
        edges,
        fee=0.0,
        delay=0.0,
        carbon=0.0,
        inter_country_hops=0,
        inter_continent_hops=0,
        reason=None,
        elapsed=0.0,
        failed_node=None,
        failed_edge=None,
        failure_index=None,
        visited_edges=None
    ):
        """
        Centralized PaymentResult creation.
        """

        return PaymentResult(
            success=success,
            path=path,
            edges=edges,
            fee=fee,
            delay=delay,
            carbon=carbon,
            inter_country_hops=(
                inter_country_hops
            ),
            inter_continent_hops=(
                inter_continent_hops
            ),
            reason=reason,
            elapsed=elapsed,
            failed_node=failed_node,
            failed_edge=failed_edge,
            failure_index=failure_index,
            visited_edges=visited_edges
        )

    # ========================================================
    # Edge Parser
    # ========================================================

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

        try:

            length = len(edge)

        except TypeError:

            return None

        if length == 2:

            u, v = edge

            return (
                u,
                v,
                None
            )

        if length == 3:

            u, v, key = edge

            return (
                u,
                v,
                key
            )

        return None

    # ========================================================
    # Graph Edge Access
    # ========================================================

    def _get_edge(
        self,
        u,
        v,
        key=None
    ):
        """
        Get exact edge data from the graph.

        For MultiDiGraph, the supplied key is preferred.
        """

        if not self.G.has_edge(
            u,
            v
        ):
            return None

        if self.G.is_multigraph():

            edge_data = self.G.get_edge_data(
                u,
                v
            )

            if not edge_data:
                return None

            # ------------------------------------------------
            # Exact edge requested
            # ------------------------------------------------

            if key is not None:

                exact = edge_data.get(
                    key
                )

                if exact is not None:
                    return exact

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

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
# Backward-Compatible Functional API
# ============================================================

def simulate_payment(
    G,
    path,
    edges,
    amount,
    failure_model=None,
    network_dynamics=None,
    tx_id=None
):
    """
    Functional wrapper around PaymentSimulator.

    This preserves the old API style while using the new
    FailureModel and NetworkDynamics architecture.
    """

    if failure_model is None:

        raise ValueError(
            "failure_model is required."
        )

    if network_dynamics is None:

        raise ValueError(
            "network_dynamics is required."
        )

    simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=network_dynamics
    )

    return simulator.simulate_payment(
        path=path,
        edges=edges,
        amount=amount,
        tx_id=tx_id
    )