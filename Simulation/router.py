# Simulation/router.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .onion import OnionRouter


class Router:
    """
    Router
    ------

    This module is responsible only for forwarding an already selected
    route through the OnionRouter.

    Routing/pathfinding is NOT performed here.

    Responsibilities:
        1. Validate the selected route.
        2. Build the onion packet.
        3. Generate/propagate route_id.
        4. Forward the packet hop-by-hop.
        5. Return forwarding information.

    Not responsible for:
        - PPO
        - Dijkstra / LND
        - Bucket management
        - Candidate route generation
        - Failure probability
        - Network state evolution
    """

    def __init__(self, onion_router: Optional[OnionRouter] = None):
        self.onion_router = onion_router or OnionRouter()

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
        Build and forward an onion packet over an already selected path.

        If route_id is not supplied, OnionRouter generates it.

        The generated route_id is ALWAYS propagated to the returned
        Router result.
        """

        self._validate_path(path)
        self._validate_amount(amount)

        # ------------------------------------------------------
        # Build onion packet
        # ------------------------------------------------------
        packet = self.onion_router.build_onion(
            path=path,
            bucket_id=bucket_id,
            tx_id=tx_id,
            amount=amount,
            metadata=metadata,
            attempt_id=attempt_id,
            route_id=route_id,
        )

        # ------------------------------------------------------
        # IMPORTANT:
        # OnionRouter may generate route_id automatically.
        # Therefore the authoritative value is packet["route_id"],
        # not the original route_id argument.
        # ------------------------------------------------------
        generated_route_id = packet.get("route_id")

        if not generated_route_id:
            raise RuntimeError(
                "OnionRouter did not generate a valid route_id."
            )

        # ------------------------------------------------------
        # Forward packet hop-by-hop
        # ------------------------------------------------------
        forwarding = self._forward_packet(
            packet=packet,
            path=path,
        )

        # ------------------------------------------------------
        # Return unified result
        # ------------------------------------------------------
        result = {
            "success": forwarding["success"],
            "reason": forwarding.get("reason"),

            "path": list(path),

            "visited": forwarding.get(
                "visited",
                [],
            ),

            "hops": forwarding.get(
                "hops",
                [],
            ),

            "attempt_id": attempt_id,

            # IMPORTANT FIX
            "route_id": generated_route_id,

            "source": path[0],
            "destination": path[-1],
            "path_length": len(path),

            "destination_reached": forwarding.get(
                "destination_reached",
                False,
            ),

            "packet": packet,
        }

        return result

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
        Rebuild an onion packet for a new route or alternative suffix.

        Used after Partial Backtracking.
        """

        self._validate_path(path)
        self._validate_amount(amount)

        packet = self.onion_router.rebuild_onion(
            path=path,
            bucket_id=bucket_id,
            tx_id=tx_id,
            amount=amount,
            metadata=metadata,
            attempt_id=attempt_id,
            route_id=route_id,
        )

        generated_route_id = packet.get("route_id")

        if not generated_route_id:
            raise RuntimeError(
                "OnionRouter did not generate a valid route_id "
                "during rebuild."
            )

        forwarding = self._forward_packet(
            packet=packet,
            path=path,
        )

        return {
            "success": forwarding["success"],
            "reason": forwarding.get("reason"),

            "path": list(path),

            "visited": forwarding.get(
                "visited",
                [],
            ),

            "hops": forwarding.get(
                "hops",
                [],
            ),

            "attempt_id": attempt_id,

            "route_id": generated_route_id,

            "source": path[0],
            "destination": path[-1],
            "path_length": len(path),

            "destination_reached": forwarding.get(
                "destination_reached",
                False,
            ),

            "packet": packet,
        }

    # ==========================================================
    # FORWARD PACKET
    # ==========================================================

    def forward_packet(
        self,
        packet: Dict[str, Any],
        path: List[Any],
    ) -> Dict[str, Any]:
        """
        Forward an existing onion packet over the supplied path.
        """

        self._validate_path(path)

        if not isinstance(packet, dict):
            raise TypeError("packet must be a dictionary.")

        return self._forward_packet(
            packet=packet,
            path=path,
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
        Forward the packet using OnionRouter.forward().

        The packet itself maintains the current onion layer.
        """

        visited = []
        hops = []

        current_packet = packet

        try:

            for node in path:

                if current_packet is None:
                    return {
                        "success": False,
                        "reason": "Onion packet exhausted before destination.",
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": False,
                    }

                step = self.onion_router.forward(
                    current_packet,
                    node,
                )

                if not isinstance(step, dict):
                    return {
                        "success": False,
                        "reason": "Invalid OnionRouter forwarding result.",
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": False,
                    }

                if step.get("success") is False:
                    return {
                        "success": False,
                        "reason": step.get(
                            "reason",
                            "Onion forwarding failed.",
                        ),
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": False,
                    }

                visited.append(node)

                hops.append(
                    {
                        "hop_index": step.get(
                            "hop_index"
                        ),
                        "node": step.get(
                            "node",
                            node,
                        ),
                        "next_hop": step.get(
                            "next_hop"
                        ),
                    }
                )

                # --------------------------------------------------
                # Destination reached
                # --------------------------------------------------
                if step.get("destination_reached") is True:
                    return {
                        "success": True,
                        "reason": None,
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": True,
                    }

                # --------------------------------------------------
                # Continue with next onion packet
                # --------------------------------------------------
                current_packet = step.get(
                    "next_packet"
                )

                if current_packet is None:
                    return {
                        "success": False,
                        "reason": (
                            "Missing next_packet before "
                            "destination was reached."
                        ),
                        "visited": visited,
                        "hops": hops,
                        "destination_reached": False,
                    }

            # ------------------------------------------------------
            # If loop finishes without destination
            # ------------------------------------------------------
            return {
                "success": False,
                "reason": "Destination was not reached.",
                "visited": visited,
                "hops": hops,
                "destination_reached": False,
            }

        except Exception as exc:

            return {
                "success": False,
                "reason": str(exc),
                "visited": visited,
                "hops": hops,
                "destination_reached": False,
            }

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_path(path: List[Any]) -> None:

        if not isinstance(path, (list, tuple)):
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

        if len(set(path)) != len(path):
            raise ValueError(
                "path contains duplicate nodes."
            )

    @staticmethod
    def _validate_amount(amount: float) -> None:

        if amount is None:
            raise ValueError(
                "amount cannot be None."
            )

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValueError(
                "amount must be numeric."
            )

        if amount <= 0:
            raise ValueError(
                "amount must be greater than zero."
            )