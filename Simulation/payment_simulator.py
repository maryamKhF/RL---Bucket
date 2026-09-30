"""
Simulation/payment_simulator.py

Execute exactly ONE selected payment route over a simulated
Lightning Network.

Responsibilities
----------------
1. Validate the selected route.
2. Preserve exact channel identity.
3. Evaluate forwarding through FailureModel.
4. Calculate payment/routing metrics.
5. Commit successful settlement through NetworkDynamics.
6. Return structured PaymentResult.
7. Expose exact failure information required by:
       - Bucket.backtrack
       - Simulation.backtrack.PartialBacktracker

This module does NOT:
    - select candidates
    - manage Bucket state
    - perform candidate-level backtracking
    - perform partial route recovery
    - perform full rerouting
    - make PPO/RL decisions

Execution flow
---------------
Bucket
    |
    v
selected candidate
    |
    v
Bucket.record_attempt()
    |
    v
PaymentSimulator
    |
    v
FailureModel
    |
    +---- failure ----> PaymentResult
    |
    +---- success ----> settlement
                         |
                         v
                    PaymentResult

Fee handling
------------
Supported attributes:

    fee_base
    fee_rate

and native Lightning/GML attributes:

    fee_base_msat
    fee_proportional_millionths

Channel identity
----------------
For MultiDiGraph / MultiDiGraph-like graphs, exact channel
keys are preserved whenever supplied:

    (u, v, key)

Capacity is NOT interpreted as directional liquidity.
"""


import math
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

    This helper is used only for routing metrics.

    Missing geographic information does NOT mean coordinates
    are zero. If either endpoint lacks valid latitude or
    longitude, the result is 0.

    Returns
    -------
    int
        1 if the hop is estimated to cross continents,
        otherwise 0.
    """

    if G is None:
        return 0

    try:
        if u not in G.nodes or v not in G.nodes:
            return 0
    except Exception:
        return 0

    source = G.nodes[u]
    target = G.nodes[v]

    source_lat = PaymentSimulator._finite_float(
        source.get("latitude")
    )

    target_lat = PaymentSimulator._finite_float(
        target.get("latitude")
    )

    source_lon = PaymentSimulator._finite_float(
        source.get("longitude")
    )

    target_lon = PaymentSimulator._finite_float(
        target.get("longitude")
    )

    if (
        source_lat is None
        or
        target_lat is None
        or
        source_lon is None
        or
        target_lon is None
    ):
        return 0

    latitude_difference = abs(
        source_lat - target_lat
    )

    longitude_difference = abs(
        source_lon - target_lon
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
    Structured result of exactly one payment attempt.
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
            list(path)
            if path is not None
            else []
        )

        self.edges = (
            list(edges)
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
            list(visited_edges)
            if visited_edges is not None
            else []
        )

    # ========================================================
    # Dictionary Representation
    # ========================================================

    def to_dict(self):
        """
        Convert result into a serializable dictionary.
        """

        return {
            "success":
                self.success,

            "path":
                list(self.path),

            "edges":
                list(self.edges),

            "fee":
                self.fee,

            "delay":
                self.delay,

            "carbon":
                self.carbon,

            "inter_country_hops":
                self.inter_country_hops,

            "inter_continent_hops":
                self.inter_continent_hops,

            "reason":
                self.reason,

            "elapsed":
                self.elapsed,

            "failed_node":
                self.failed_node,

            "failed_edge":
                self.failed_edge,

            "failure_index":
                self.failure_index,

            "visited_edges":
                list(self.visited_edges)
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
    Execute exactly one selected payment route.

    PaymentSimulator is intentionally unaware of Bucket,
    Backtracker and PPO.
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
    # Execute One Payment
    # ========================================================

    def simulate_payment(
        self,
        path,
        edges,
        amount,
        tx_id=None
    ):
        """
        Execute exactly one payment attempt.

        No retry is performed here.
        """

        start_time = time.perf_counter()

        # ----------------------------------------------------
        # Validate amount
        # ----------------------------------------------------

        amount_value = self._positive_amount(
            amount
        )

        if amount_value is None:

            return self._finalize_result(
                self._result(
                    success=False,
                    path=path,
                    edges=edges,
                    reason="invalid_amount",
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Validate basic route input
        # ----------------------------------------------------

        if not isinstance(
            path,
            (list, tuple)
        ):

            return self._finalize_result(
                self._result(
                    success=False,
                    path=[],
                    edges=edges,
                    reason="invalid_path",
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        if not isinstance(
            edges,
            (list, tuple)
        ):

            return self._finalize_result(
                self._result(
                    success=False,
                    path=path,
                    edges=[],
                    reason="invalid_edges",
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        if len(path) < 2:

            return self._finalize_result(
                self._result(
                    success=False,
                    path=path,
                    edges=edges,
                    reason="no_path",
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Validate route and exact edges
        # ----------------------------------------------------

        validation = self._validate_route(
            path,
            edges
        )

        if not validation["valid"]:

            return self._finalize_result(
                self._result(
                    success=False,
                    path=path,
                    edges=edges,
                    reason=validation["reason"],
                    failed_edge=validation.get(
                        "failed_edge"
                    ),
                    failure_index=validation.get(
                        "failure_index"
                    ),
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = self._calculate_metrics(
            edges,
            amount_value
        )

        # ----------------------------------------------------
        # FailureModel
        # ----------------------------------------------------

        try:

            failure = (
                self.failure_model
                .evaluate_payment_failure(
                    route=list(path),
                    amount=amount_value,
                    network=self.G,
                    route_edges=list(edges)
                )
            )

        except Exception as exc:

            return self._finalize_result(
                self._result(
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
                    reason=(
                        "failure_model_error:"
                        f"{type(exc).__name__}"
                    ),
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Validate FailureModel output
        # ----------------------------------------------------

        if not isinstance(
            failure,
            dict
        ):

            return self._finalize_result(
                self._result(
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
                    reason="invalid_failure_model_result",
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Payment failed
        # ----------------------------------------------------

        if not bool(
            failure.get(
                "success",
                False
            )
        ):

            visited_edges = (
                failure.get(
                    "visited_edges",
                    []
                )
            )

            if visited_edges is None:
                visited_edges = []

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
                reason=failure.get(
                    "reason",
                    "payment_failed"
                ),
                failed_node=failure.get(
                    "failed_node"
                ),
                failed_edge=failure.get(
                    "failed_edge"
                ),
                failure_index=failure.get(
                    "failure_index"
                ),
                visited_edges=visited_edges,
                elapsed=(
                    time.perf_counter()
                    -
                    start_time
                )
            )

            return self._finalize_result(
                result,
                tx_id
            )

        # ----------------------------------------------------
        # FailureModel says forwarding succeeded.
        #
        # Now settlement is attempted.
        # ----------------------------------------------------

        try:

            settlement_success = (
                self.network_dynamics
                .settle_route(
                    route_edges=list(edges),
                    amount=amount_value
                )
            )

        except Exception as exc:

            return self._finalize_result(
                self._result(
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
                    reason=(
                        "settlement_error:"
                        f"{type(exc).__name__}"
                    ),
                    elapsed=(
                        time.perf_counter()
                        -
                        start_time
                    )
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Settlement rejected
        # ----------------------------------------------------

        if not bool(
            settlement_success
        ):

            return self._finalize_result(
                self._result(
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
                ),
                tx_id
            )

        # ----------------------------------------------------
        # Successful payment
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

        return self._finalize_result(
            result,
            tx_id
        )

    # ========================================================
    # Finalize / Record
    # ========================================================

    def _finalize_result(
        self,
        result,
        tx_id
    ):
        """
        Record one completed attempt.

        Failure to record a result must not invalidate the
        PaymentResult already produced by the simulator.
        """

        if tx_id is not None:

            try:

                self.network_dynamics.record_payment(
                    tx_id,
                    result.to_dict()
                )

            except Exception:
                pass

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
        Validate node path and exact edge sequence.
        """

        if not isinstance(
            path,
            (list, tuple)
        ):

            return {
                "valid": False,
                "reason": "invalid_path",
                "failed_edge": None,
                "failure_index": None
            }

        if not isinstance(
            edges,
            (list, tuple)
        ):

            return {
                "valid": False,
                "reason": "invalid_edges",
                "failed_edge": None,
                "failure_index": None
            }

        if len(path) < 2:

            return {
                "valid": False,
                "reason": "path_too_short",
                "failed_edge": None,
                "failure_index": None
            }

        if len(path) != len(edges) + 1:

            return {
                "valid": False,
                "reason": "route_edge_mismatch",
                "failed_edge": None,
                "failure_index": None
            }

        # ----------------------------------------------------
        # Reject repeated nodes.
        # ----------------------------------------------------

        try:

            if len(path) != len(
                set(path)
            ):

                return {
                    "valid": False,
                    "reason": "route_contains_loop",
                    "failed_edge": None,
                    "failure_index": None
                }

        except TypeError:

            return {
                "valid": False,
                "reason": "unhashable_route_node",
                "failed_edge": None,
                "failure_index": None
            }

        # ----------------------------------------------------
        # Validate every exact edge.
        # ----------------------------------------------------

        for index, edge in enumerate(
            edges
        ):

            parsed = self._parse_edge(
                edge
            )

            if parsed is None:

                return {
                    "valid": False,
                    "reason": "invalid_edge",
                    "failed_edge": edge,
                    "failure_index": index
                }

            u, v, key = parsed

            # ------------------------------------------------
            # Node sequence consistency
            # ------------------------------------------------

            if u != path[index]:

                return {
                    "valid": False,
                    "reason": "invalid_route_source",
                    "failed_edge": edge,
                    "failure_index": index
                }

            if v != path[index + 1]:

                return {
                    "valid": False,
                    "reason": "invalid_route_destination",
                    "failed_edge": edge,
                    "failure_index": index
                }

            # ------------------------------------------------
            # Channel existence
            # ------------------------------------------------

            try:

                if not self.G.has_edge(
                    u,
                    v
                ):

                    return {
                        "valid": False,
                        "reason": "missing_channel",
                        "failed_edge": edge,
                        "failure_index": index
                    }

            except Exception:

                return {
                    "valid": False,
                    "reason": "channel_lookup_error",
                    "failed_edge": edge,
                    "failure_index": index
                }

            # ------------------------------------------------
            # Exact channel validation
            # ------------------------------------------------

            if self.G.is_multigraph():

                if key is None:

                    return {
                        "valid": False,
                        "reason": (
                            "missing_channel_key"
                        ),
                        "failed_edge": edge,
                        "failure_index": index
                    }

                try:

                    if not self.G.has_edge(
                        u,
                        v,
                        key
                    ):

                        return {
                            "valid": False,
                            "reason": "missing_channel",
                            "failed_edge": edge,
                            "failure_index": index
                        }

                except Exception:

                    return {
                        "valid": False,
                        "reason": "channel_lookup_error",
                        "failed_edge": edge,
                        "failure_index": index
                    }

            else:

                # Simple Graph / DiGraph should not require
                # a channel key.
                if key is not None:

                    return {
                        "valid": False,
                        "reason": (
                            "unexpected_channel_key"
                        ),
                        "failed_edge": edge,
                        "failure_index": index
                    }

        return {
            "valid": True,
            "reason": None,
            "failed_edge": None,
            "failure_index": None
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
        Calculate route-level metrics without modifying
        network state.
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

            delay += self._safe_float(
                delay_value
            )

            # ------------------------------------------------
            # Node metrics
            # ------------------------------------------------

            try:

                if (
                    u not in self.G.nodes
                    or
                    v not in self.G.nodes
                ):
                    continue

            except Exception:

                continue

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

            # ------------------------------------------------
            # Country metric
            # ------------------------------------------------

            source_country = source.get(
                "country"
            )

            target_country = target.get(
                "country"
            )

            if (
                source_country not in (
                    None,
                    ""
                )
                and
                target_country not in (
                    None,
                    ""
                )
                and
                source_country != target_country
            ):

                inter_country += 1

            # ------------------------------------------------
            # Continent metric
            # ------------------------------------------------

            inter_continent += (
                estimate_inter_continent(
                    self.G,
                    u,
                    v
                )
            )

        average_carbon = (
            carbon /
            max(
                1,
                len(edges)
            )
        )

        return {
            "fee":
                fee,

            "delay":
                delay,

            "carbon":
                average_carbon,

            "inter_country_hops":
                inter_country,

            "inter_continent_hops":
                inter_continent
        }

    # ========================================================
    # Fee Components
    # ========================================================

    @staticmethod
    def _get_fee_components(
        data
    ):
        """
        Extract base and proportional Lightning fee.

        Supported forms:

            fee_base
            fee_rate

        or:

            fee_base_msat
            fee_proportional_millionths
        """

        if not data:
            return 0.0, 0.0

        # ----------------------------------------------------
        # Base fee
        # ----------------------------------------------------

        if data.get(
            "fee_base"
        ) is not None:

            base = data.get(
                "fee_base"
            )

        else:

            base = data.get(
                "fee_base_msat",
                0.0
            )

        # ----------------------------------------------------
        # Proportional fee
        # ----------------------------------------------------

        if data.get(
            "fee_rate"
        ) is not None:

            rate = data.get(
                "fee_rate"
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

        if base < 0:
            base = 0.0

        if rate < 0:
            rate = 0.0

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
        Safe finite float conversion.
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
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

    # ========================================================
    # Finite Float
    # ========================================================

    @staticmethod
    def _finite_float(
        value
    ):
        """
        Return finite float or None.
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if not math.isfinite(
            value
        ):

            return None

        return value

    # ========================================================
    # Positive Amount
    # ========================================================

    @staticmethod
    def _positive_amount(
        amount
    ):
        """
        Return a finite positive amount or None.
        """

        try:

            value = float(
                amount
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if not math.isfinite(
            value
        ):

            return None

        if value <= 0:
            return None

        return value

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

        or:

            (u, v, key)
        """

        if edge is None:
            return None

        if not isinstance(
            edge,
            (tuple, list)
        ):
            return None

        if len(edge) == 2:

            return (
                edge[0],
                edge[1],
                None
            )

        if len(edge) == 3:

            return (
                edge[0],
                edge[1],
                edge[2]
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
        Return exact edge data.

        MultiGraph
        ----------
        If key is supplied, only that exact channel is used.

        If key is omitted, None is returned intentionally.

        PaymentSimulator requires exact channel identity for
        MultiDiGraph execution and must not silently select
        another parallel channel.

        Graph / DiGraph
        ---------------
        The ordinary edge data is returned.
        """

        try:

            if not self.G.has_edge(
                u,
                v
            ):

                return None

        except Exception:

            return None

        # ----------------------------------------------------
        # MultiGraph / MultiDiGraph
        # ----------------------------------------------------

        if self.G.is_multigraph():

            if key is None:

                return None

            try:

                edge_data = self.G.get_edge_data(
                    u,
                    v,
                    key
                )

            except Exception:

                return None

            return edge_data

        # ----------------------------------------------------
        # Graph / DiGraph
        # ----------------------------------------------------

        try:

            return self.G.get_edge_data(
                u,
                v
            )

        except Exception:

            return None


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
    Functional wrapper.

    Executes exactly ONE payment attempt.
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


# ============================================================
# Standalone Module Test
# ============================================================

if __name__ == "__main__":

    import networkx as nx

    print()
    print("=" * 72)
    print("PAYMENT SIMULATOR MODULE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Test graph
    # --------------------------------------------------------

    G = nx.MultiDiGraph()

    G.add_edge(
        "A",
        "B",
        key=10,
        fee_base_msat=1000,
        fee_proportional_millionths=100,
        cltv_expiry_delta=40,
        balance_uv=10000,
        capacity=20000,
        available=True
    )

    G.add_edge(
        "B",
        "C",
        key=20,
        fee_base_msat=1000,
        fee_proportional_millionths=200,
        cltv_expiry_delta=50,
        balance_uv=10000,
        capacity=20000,
        available=True
    )

    for node in G.nodes:

        G.nodes[node]["available"] = True
        G.nodes[node]["is_online"] = True

    # --------------------------------------------------------
    # Minimal FailureModel mock
    # --------------------------------------------------------

    class TestFailureModel:

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            return {
                "success": True,
                "reason": "success",
                "visited_edges": list(
                    route_edges
                )
            }

    # --------------------------------------------------------
    # Minimal NetworkDynamics mock
    # --------------------------------------------------------

    class TestNetworkDynamics:

        def __init__(self):

            self.records = []

        def settle_route(
            self,
            route_edges,
            amount
        ):

            return True

        def record_payment(
            self,
            tx_id,
            result
        ):

            self.records.append(
                (
                    tx_id,
                    result
                )
            )

    failure_model = TestFailureModel()

    network_dynamics = TestNetworkDynamics()

    simulator = PaymentSimulator(
        G=G,
        failure_model=failure_model,
        network_dynamics=network_dynamics
    )

    # --------------------------------------------------------
    # Exact route
    # --------------------------------------------------------

    path = [
        "A",
        "B",
        "C"
    ]

    edges = [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    result = simulator.simulate_payment(
        path=path,
        edges=edges,
        amount=1000,
        tx_id="TX-001"
    )

    print()
    print(
        f"Success             : "
        f"{result.success}"
    )

    print(
        f"Reason              : "
        f"{result.reason}"
    )

    print(
        f"Path                : "
        f"{result.path}"
    )

    print(
        f"Edges               : "
        f"{result.edges}"
    )

    print(
        f"Visited edges      : "
        f"{result.visited_edges}"
    )

    print(
        f"Fee                 : "
        f"{result.fee:.4f}"
    )

    print(
        f"Delay               : "
        f"{result.delay:.4f}"
    )

    print(
        f"Recorded results    : "
        f"{len(network_dynamics.records)}"
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert result.success is True

    assert result.reason == "success"

    assert result.path == [
        "A",
        "B",
        "C"
    ]

    assert result.edges == [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    assert result.visited_edges == [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    assert len(
        network_dynamics.records
    ) == 1

    # --------------------------------------------------------
    # Verify exact parallel-channel identity
    # --------------------------------------------------------

    G.add_edge(
        "A",
        "B",
        key=99,
        fee_base_msat=999999,
        balance_uv=10000,
        available=True
    )

    exact_data = simulator._get_edge(
        "A",
        "B",
        10
    )

    assert exact_data is not None

    assert (
        exact_data[
            "fee_base_msat"
        ]
        ==
        1000
    )

    # Missing key must NOT silently select another
    # parallel channel.

    missing_data = simulator._get_edge(
        "A",
        "B",
        777
    )

    assert missing_data is None

    no_key_data = simulator._get_edge(
        "A",
        "B"
    )

    assert no_key_data is None

    # --------------------------------------------------------
    # Failure propagation test
    # --------------------------------------------------------

    class FailedPaymentModel:

        def evaluate_payment_failure(
            self,
            route,
            amount,
            network,
            route_edges=None
        ):

            return {
                "success": False,
                "reason": "channel_failure",
                "failed_node": "B",
                "failed_edge": (
                    "B",
                    "C",
                    20
                ),
                "failure_index": 1,
                "visited_edges": [
                    ("A", "B", 10)
                ]
            }

    failed_simulator = PaymentSimulator(
        G=G,
        failure_model=FailedPaymentModel(),
        network_dynamics=network_dynamics
    )

    failed_result = (
        failed_simulator.simulate_payment(
            path=path,
            edges=edges,
            amount=1000,
            tx_id="TX-002"
        )
    )

    print()
    print(
        f"Failure test        : "
        f"{failed_result.reason}"
    )

    print(
        f"Failed edge         : "
        f"{failed_result.failed_edge}"
    )

    print(
        f"Failure index       : "
        f"{failed_result.failure_index}"
    )

    print(
        f"Visited edges      : "
        f"{failed_result.visited_edges}"
    )

    assert failed_result.success is False

    assert (
        failed_result.reason
        ==
        "channel_failure"
    )

    assert (
        failed_result.failed_edge
        ==
        ("B", "C", 20)
    )

    assert (
        failed_result.failure_index
        ==
        1
    )

    assert failed_result.visited_edges == [
        ("A", "B", 10)
    ]

    # --------------------------------------------------------
    # Invalid exact channel test
    # --------------------------------------------------------

    invalid_result = simulator.simulate_payment(
        path=path,
        edges=[
            ("A", "B", 777),
            ("B", "C", 20)
        ],
        amount=1000,
        tx_id="TX-003"
    )

    assert invalid_result.success is False

    assert (
        invalid_result.reason
        ==
        "missing_channel"
    )

    print()
    print(
        "Exact channel test  : PASS"
    )

    print(
        "Failure propagation : PASS"
    )

    print(
        "Settlement test     : PASS"
    )

    print(
        "PAYMENT SIMULATOR STATUS : SUCCESS"
    )