# Simulation/router.py

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .onion import OnionRouter


class Router:
    """
    Router
    ------

    Forward an already selected route through OnionRouter.

    Router is a forwarding layer only.

    Responsibilities
    ----------------
    1. Validate the selected route.
    2. Validate the payment amount.
    3. Build an onion packet.
    4. Preserve the authoritative route_id.
    5. Forward the onion packet hop-by-hop.
    6. Return structured forwarding information.
    7. Rebuild and forward an alternative route/suffix.

    NOT responsible for
    -------------------
    - PPO
    - Dijkstra / LND
    - Top-K route generation
    - Bucket management
    - Failure probability
    - Payment settlement
    - Network state evolution
    - Partial backtracking decisions

    Intended flow
    -------------

        Bucket
          |
          | selected candidate
          v
        Router
          |
          | Onion forwarding
          v
        PaymentSimulator
          |
          v
        FailureModel
    """

    def __init__(
        self,
        onion_router: Optional[OnionRouter] = None,
    ):
        self.onion_router = (
            onion_router
            if onion_router is not None
            else OnionRouter()
        )

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def forward(
        self,
        path: List[Any],
        bucket_id: Any,
        tx_id: Any,
        amount: float,
        attempt_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        route_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build and forward an onion packet over an already
        selected route.

        The route is NOT discovered here.

        If OnionRouter generates a route_id, the value stored
        inside the generated packet is authoritative.
        """

        self._validate_path(path)
        self._validate_amount(amount)

        packet = self.onion_router.build_onion(
            path=path,
            bucket_id=bucket_id,
            tx_id=tx_id,
            amount=float(amount),
            metadata=metadata,
            attempt_id=attempt_id,
            route_id=route_id,
        )

        self._validate_packet(packet)

        generated_route_id = packet.get(
            "route_id"
        )

        forwarding = self._forward_packet(
            packet=packet,
            path=path,
        )

        return self._build_result(
            forwarding=forwarding,
            path=path,
            attempt_id=attempt_id,
            route_id=generated_route_id,
            packet=packet,
        )

    # ==========================================================
    # REBUILD
    # ==========================================================

    def rebuild(
        self,
        path: List[Any],
        bucket_id: Any,
        tx_id: Any,
        amount: float,
        attempt_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        route_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Rebuild and forward an onion packet for an alternative
        route or alternative suffix.

        This method is intended to be called after a routing
        failure has been detected by the payment/backtracking
        layer.

        Router itself does NOT decide which alternative route
        should be selected.
        """

        self._validate_path(path)
        self._validate_amount(amount)

        packet = self.onion_router.rebuild_onion(
            path=path,
            bucket_id=bucket_id,
            tx_id=tx_id,
            amount=float(amount),
            metadata=metadata,
            attempt_id=attempt_id,
            route_id=route_id,
        )

        self._validate_packet(packet)

        generated_route_id = packet.get(
            "route_id"
        )

        forwarding = self._forward_packet(
            packet=packet,
            path=path,
        )

        return self._build_result(
            forwarding=forwarding,
            path=path,
            attempt_id=attempt_id,
            route_id=generated_route_id,
            packet=packet,
        )

    # ==========================================================
    # FORWARD EXISTING PACKET
    # ==========================================================

    def forward_packet(
        self,
        packet: Dict[str, Any],
        path: List[Any],
    ) -> Dict[str, Any]:
        """
        Forward an existing onion packet over the supplied path.

        This method does not rebuild the packet.
        """

        self._validate_path(path)
        self._validate_packet(packet)

        route_id = packet.get(
            "route_id"
        )

        forwarding = self._forward_packet(
            packet=packet,
            path=path,
        )

        return self._build_result(
            forwarding=forwarding,
            path=path,
            attempt_id=packet.get(
                "attempt_id",
                0,
            ),
            route_id=route_id,
            packet=packet,
        )

    # ==========================================================
    # INTERNAL FORWARDING
    # ==========================================================

    def _forward_packet(
        self,
        packet: Dict[str, Any],
        path: List[Any],
    ) -> Dict[str, Any]:
        """
        Forward one onion packet hop-by-hop.

        OnionRouter is authoritative for the forwarding step.

        Router only:
            - passes the current packet,
            - records forwarding results,
            - propagates the next onion layer,
            - detects destination arrival.
        """

        visited: List[Any] = []
        hops: List[Dict[str, Any]] = []

        current_packet = packet

        destination = path[-1]

        try:

            for hop_index, node in enumerate(path):

                if current_packet is None:
                    return self._failure(
                        reason=(
                            "Onion packet exhausted before "
                            "destination was reached."
                        ),
                        visited=visited,
                        hops=hops,
                    )

                step = self.onion_router.forward(
                    current_packet,
                    node,
                )

                if not isinstance(step, dict):
                    return self._failure(
                        reason=(
                            "Invalid OnionRouter forwarding "
                            "result."
                        ),
                        visited=visited,
                        hops=hops,
                    )

                # --------------------------------------------------
                # Explicit forwarding failure
                # --------------------------------------------------

                if step.get("success") is False:

                    return self._failure(
                        reason=step.get(
                            "reason",
                            "Onion forwarding failed.",
                        ),
                        visited=visited,
                        hops=hops,
                        extra={
                            "failed_node": node,
                            "failure_index": hop_index,
                        },
                    )

                # --------------------------------------------------
                # Record visited node
                # --------------------------------------------------

                visited.append(
                    step.get(
                        "node",
                        node,
                    )
                )

                hop_record = {
                    "hop_index": step.get(
                        "hop_index",
                        hop_index,
                    ),
                    "node": step.get(
                        "node",
                        node,
                    ),
                    "next_hop": step.get(
                        "next_hop"
                    ),
                }

                hops.append(
                    hop_record
                )

                # --------------------------------------------------
                # Destination reached
                # --------------------------------------------------

                if step.get(
                    "destination_reached"
                ) is True:

                    # The OnionRouter reports destination
                    # reached. Make sure this corresponds to
                    # the route destination whenever it provides
                    # a node value.

                    reached_node = step.get(
                        "node",
                        node,
                    )

                    if reached_node != destination:
                        return self._failure(
                            reason=(
                                "OnionRouter reported destination "
                                "reached at an unexpected node."
                            ),
                            visited=visited,
                            hops=hops,
                            extra={
                                "failed_node": reached_node,
                                "failure_index": hop_index,
                            },
                        )

                    return {
                        "success": True,
                        "reason": None,
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": True,
                        "failed_node": None,
                        "failure_index": None,
                    }

                # --------------------------------------------------
                # Continue with next onion packet
                # --------------------------------------------------

                next_packet = step.get(
                    "next_packet"
                )

                if next_packet is None:
                    return self._failure(
                        reason=(
                            "Missing next_packet before "
                            "destination was reached."
                        ),
                        visited=visited,
                        hops=hops,
                        extra={
                            "failed_node": node,
                            "failure_index": hop_index,
                        },
                    )

                current_packet = next_packet

            # ------------------------------------------------------
            # Loop completed without destination
            # ------------------------------------------------------

            return self._failure(
                reason=(
                    "Destination was not reached "
                    "after forwarding all route hops."
                ),
                visited=visited,
                hops=hops,
            )

        except Exception as exc:

            return self._failure(
                reason=(
                    f"onion_forwarding_error:"
                    f"{type(exc).__name__}:"
                    f"{exc}"
                ),
                visited=visited,
                hops=hops,
            )

    # ==========================================================
    # RESULT BUILDING
    # ==========================================================

    @staticmethod
    def _build_result(
        forwarding: Dict[str, Any],
        path: List[Any],
        attempt_id: int,
        route_id: Optional[str],
        packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert the internal forwarding result into the
        public Router result.
        """

        if not route_id:
            raise RuntimeError(
                "OnionRouter did not provide a valid route_id."
            )

        result = {
            "success": bool(
                forwarding.get(
                    "success",
                    False,
                )
            ),

            "reason": forwarding.get(
                "reason"
            ),

            "path": list(
                path
            ),

            "visited": list(
                forwarding.get(
                    "visited",
                    [],
                )
            ),

            "hops": list(
                forwarding.get(
                    "hops",
                    [],
                )
            ),

            "attempt_id": attempt_id,

            "route_id": route_id,

            "source": path[0],

            "destination": path[-1],

            "path_length": len(
                path
            ),

            "destination_reached": bool(
                forwarding.get(
                    "destination_reached",
                    False,
                )
            ),

            "failed_node": forwarding.get(
                "failed_node"
            ),

            "failure_index": forwarding.get(
                "failure_index"
            ),

            "packet": packet,
        }

        return result

    # ==========================================================
    # FAILURE RESULT
    # ==========================================================

    @staticmethod
    def _failure(
        reason: str,
        visited: List[Any],
        hops: List[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a consistent internal forwarding failure result.
        """

        result = {
            "success": False,
            "reason": reason,
            "visited": list(
                visited
            ),
            "hops": list(
                hops
            ),
            "destination_reached": False,
            "failed_node": None,
            "failure_index": None,
        }

        if extra:
            result.update(
                extra
            )

        return result

    # ==========================================================
    # PACKET VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_packet(
        packet: Dict[str, Any]
    ) -> None:
        """
        Validate the minimum packet contract required by Router.
        """

        if not isinstance(
            packet,
            dict,
        ):
            raise TypeError(
                "Onion packet must be a dictionary."
            )

        route_id = packet.get(
            "route_id"
        )

        if not route_id:
            raise ValueError(
                "Onion packet must contain a valid route_id."
            )

    # ==========================================================
    # PATH VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_path(
        path: List[Any]
    ) -> None:
        """
        Validate an already selected route.

        Router does not determine whether this is the optimal
        route. It only verifies that the supplied route is
        structurally valid for forwarding.
        """

        if not isinstance(
            path,
            (list, tuple),
        ):
            raise TypeError(
                "path must be a list or tuple."
            )

        if len(path) < 2:
            raise ValueError(
                "path must contain at least two nodes."
            )

        if path[0] == path[-1]:
            raise ValueError(
                "Source and destination must be different."
            )

        try:
            unique_nodes = set(
                path
            )
        except TypeError as exc:
            raise TypeError(
                "path nodes must be hashable."
            ) from exc

        if len(unique_nodes) != len(path):
            raise ValueError(
                "path contains duplicate nodes."
            )

    # ==========================================================
    # AMOUNT VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_amount(
        amount: float
    ) -> None:
        """
        Validate payment amount.
        """

        if amount is None:
            raise ValueError(
                "amount cannot be None."
            )

        try:
            numeric_amount = float(
                amount
            )
        except (
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "amount must be numeric."
            ) from exc

        if not math.isfinite(
            numeric_amount
        ):
            raise ValueError(
                "amount must be finite."
            )

        if numeric_amount <= 0:
            raise ValueError(
                "amount must be greater than zero."
            )