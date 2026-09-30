# Simulation/backtrack.py

"""
Partial Backtracking for RL + Bucket routing.

Architecture defined in README:

    RL / PPO
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
    Payment Simulation
        |
        v
    Failure
        |
        v
    Partial Backtracking
        |
        +----> Alternative suffix from Bucket
        |
        +----> If no valid suffix:
                    PPO + LND / Dijkstra rerouting


Responsibilities
----------------
This module is responsible for:

1. Receiving the failed route.
2. Receiving the exact failed edge / failure index.
3. Finding the nearest valid branch point.
4. Looking for an alternative suffix in Bucket.
5. Preventing loops.
6. Checking channel availability/liquidity.
7. Preventing reuse of the failed channel in the same attempt.
8. Returning a new complete route when possible.
9. Reporting when a full reroute is required.

This module does NOT:

- perform RL decisions
- calculate PPO actions
- modify routing heuristics
- run Dijkstra/LND
- generate candidate routes
- modify Bucket contents
- simulate channel failures
- settle payments
- build Onion packets

Bucket remains the source of stored alternative routes.

The Backtracking module only consumes Bucket information.
"""


from typing import Any, Dict, List, Optional, Tuple


class PartialBacktracker:
    """
    Performs partial backtracking using alternative routes
    already stored in Bucket.

    The main idea is:

        failed route:
            A -> B -> C -> D -> E

        failure:
            C -> D

        branch point:
            B

        alternative suffix:
            B -> F -> G -> E

        new route:
            A -> B -> F -> G -> E

    The prefix A -> B is preserved.

    If no valid alternative suffix exists in Bucket,
    the result explicitly requests full rerouting.
    """

    def __init__(
        self,
        network,
        bucket=None
    ):
        """
        Parameters
        ----------
        network : NetworkX graph
            Current Lightning network graph.

        bucket : Bucket object, optional
            Bucket module containing alternative routes.

        Notes
        -----
        The implementation intentionally accepts different Bucket
        interfaces because the exact Bucket API may evolve.

        The Backtracker therefore normalizes Bucket output internally.
        """

        self.network = network
        self.bucket = bucket

    # ==========================================================
    # Main Backtracking API
    # ==========================================================

    def backtrack(
        self,
        route,
        failed_edge=None,
        failure_index=None,
        amount=0,
        bucket_id=None,
        attempt_id=0
    ):
        """
        Perform Partial Backtracking after a route failure.

        Parameters
        ----------
        route : list
            Failed complete route.

        failed_edge : tuple, optional
            Failed directed edge:

                (u, v)

            or:

                (u, v, key)

        failure_index : int, optional
            Index of the failed edge in the route.

        amount : float
            Payment amount.

        bucket_id : str, optional
            Bucket entry identifier.

        attempt_id : int
            Current payment attempt.

        Returns
        -------
        dict
            Structured backtracking result.

        Possible statuses
        -----------------
        "alternative_found"
            A valid alternative suffix was found.

        "full_reroute_required"
            Bucket has no valid alternative.

        "invalid_route"
            Input route is invalid.

        "no_branch_point"
            No usable branch point exists.
        """

        # ------------------------------------------------------
        # Validate route
        # ------------------------------------------------------

        if not self._validate_route(route):

            return {
                "success": False,
                "status": "invalid_route",
                "reason": "invalid_route",
                "route": route,
                "attempt_id": attempt_id
            }

        # ------------------------------------------------------
        # Resolve failed edge
        # ------------------------------------------------------

        resolved_failed_edge = self._resolve_failed_edge(
            route=route,
            failed_edge=failed_edge,
            failure_index=failure_index
        )

        if resolved_failed_edge is None:

            return {
                "success": False,
                "status": "invalid_failure",
                "reason": "failed_edge_not_resolved",
                "route": list(route),
                "attempt_id": attempt_id
            }

        failed_u = resolved_failed_edge[0]
        failed_v = resolved_failed_edge[1]

        # ------------------------------------------------------
        # Determine failure position
        # ------------------------------------------------------

        resolved_failure_index = self._resolve_failure_index(
            route=route,
            failed_edge=resolved_failed_edge,
            failure_index=failure_index
        )

        if resolved_failure_index is None:

            return {
                "success": False,
                "status": "invalid_failure",
                "reason": "failure_index_not_resolved",
                "failed_edge": resolved_failed_edge,
                "attempt_id": attempt_id
            }

        # ------------------------------------------------------
        # Find nearest branch point
        # ------------------------------------------------------

        branch_index = self.find_nearest_branch_point(
            route=route,
            failure_index=resolved_failure_index,
            amount=amount
        )

        if branch_index is None:

            return {
                "success": False,
                "status": "no_branch_point",
                "reason": "no_valid_branch_point",
                "failed_edge": resolved_failed_edge,
                "failure_index": resolved_failure_index,
                "attempt_id": attempt_id,
                "full_reroute_required": True
            }

        branch_node = route[branch_index]

        # ------------------------------------------------------
        # Obtain candidate suffixes from Bucket
        # ------------------------------------------------------

        candidates = self.get_alternative_suffixes(
            route=route,
            branch_index=branch_index,
            failed_edge=resolved_failed_edge,
            bucket_id=bucket_id
        )

        # ------------------------------------------------------
        # Evaluate candidate suffixes
        # ------------------------------------------------------

        for candidate in candidates:

            suffix = self._normalize_suffix(
                candidate=candidate,
                branch_node=branch_node
            )

            if suffix is None:
                continue

            # Candidate must begin at branch node.
            if suffix[0] != branch_node:
                continue

            # --------------------------------------------------
            # Failed channel must not be reused.
            # --------------------------------------------------

            if self._contains_failed_edge(
                suffix,
                resolved_failed_edge
            ):
                continue

            # --------------------------------------------------
            # No loops.
            # --------------------------------------------------

            prefix = route[:branch_index]

            if not self._is_loop_free(
                prefix=prefix,
                suffix=suffix
            ):
                continue

            # --------------------------------------------------
            # Validate channels and liquidity.
            # --------------------------------------------------

            if not self.is_suffix_valid(
                suffix=suffix,
                amount=amount
            ):
                continue

            # --------------------------------------------------
            # Construct complete route.
            # --------------------------------------------------

            new_route = self._combine_route(
                prefix=prefix,
                suffix=suffix
            )

            if not self._validate_route(new_route):
                continue

            # --------------------------------------------------
            # Successful Partial Backtracking.
            # --------------------------------------------------

            return {
                "success": True,
                "status": "alternative_found",
                "reason": "partial_backtrack_success",
                "original_route": list(route),
                "new_route": new_route,
                "preserved_prefix": list(prefix),
                "alternative_suffix": list(suffix),
                "branch_point": branch_node,
                "branch_index": branch_index,
                "failed_edge": resolved_failed_edge,
                "failure_index": resolved_failure_index,
                "amount": amount,
                "bucket_id": bucket_id,
                "attempt_id": attempt_id + 1,
                "full_reroute_required": False
            }

        # ------------------------------------------------------
        # No valid Bucket alternative.
        # ------------------------------------------------------

        return {
            "success": False,
            "status": "full_reroute_required",
            "reason": "no_valid_bucket_alternative",
            "original_route": list(route),
            "branch_point": branch_node,
            "branch_index": branch_index,
            "failed_edge": resolved_failed_edge,
            "failure_index": resolved_failure_index,
            "amount": amount,
            "bucket_id": bucket_id,
            "attempt_id": attempt_id,
            "full_reroute_required": True
        }

    # ==========================================================
    # Find nearest branch point
    # ==========================================================

    def find_nearest_branch_point(
        self,
        route,
        failure_index,
        amount=0
    ):
        """
        Find the nearest usable branch point before failure.

        For:

            A -> B -> C -> D -> E

        and failure:

            C -> D

        the search starts from C and moves backward:

            C
            B
            A

        The first node for which a valid alternative suffix can
        potentially start is selected.

        The method does not select the alternative itself.
        """

        if failure_index is None:
            return None

        if failure_index < 0:
            return None

        if failure_index >= len(route) - 1:
            return None

        # Start at the node immediately before failed edge.
        start_index = failure_index

        # Move backward toward source.
        for index in range(
            start_index,
            -1,
            -1
        ):

            node = route[index]

            if self._has_usable_outgoing_edge(
                node=node,
                amount=amount,
                excluded_edge=None
            ):
                return index

        return None

    # ==========================================================
    # Get alternative suffixes
    # ==========================================================

    def get_alternative_suffixes(
        self,
        route,
        branch_index,
        failed_edge,
        bucket_id=None
    ):
        """
        Retrieve alternative suffixes from Bucket.

        Expected conceptual Bucket content:

            {
                "routes": [
                    A-B-C-D-E,
                    A-B-F-G-E,
                    A-H-I-E
                ]
            }

        If the failed route is:

            A-B-C-D-E

        and branch point is B,

        possible suffixes include:

            B-F-G-E

        The method supports several common Bucket interfaces.
        """

        if self.bucket is None:
            return []

        branch_node = route[branch_index]

        raw_candidates = []

        # ------------------------------------------------------
        # Interface 1:
        # get_alternative_routes(...)
        # ------------------------------------------------------

        method = getattr(
            self.bucket,
            "get_alternative_routes",
            None
        )

        if callable(method):

            try:

                raw_candidates = method(
                    bucket_id=bucket_id,
                    route=route,
                    branch_node=branch_node,
                    failed_edge=failed_edge
                )

            except TypeError:

                try:

                    raw_candidates = method(
                        bucket_id
                    )

                except Exception:
                    raw_candidates = []

        # ------------------------------------------------------
        # Interface 2:
        # get_routes(...)
        # ------------------------------------------------------

        if not raw_candidates:

            method = getattr(
                self.bucket,
                "get_routes",
                None
            )

            if callable(method):

                try:

                    raw_candidates = method(
                        bucket_id=bucket_id
                    )

                except TypeError:

                    try:
                        raw_candidates = method()
                    except Exception:
                        raw_candidates = []

        # ------------------------------------------------------
        # Interface 3:
        # routes attribute
        # ------------------------------------------------------

        if not raw_candidates:

            routes = getattr(
                self.bucket,
                "routes",
                None
            )

            if routes is not None:

                raw_candidates = routes

        # ------------------------------------------------------
        # Normalize
        # ------------------------------------------------------

        if raw_candidates is None:
            return []

        if isinstance(
            raw_candidates,
            dict
        ):

            raw_candidates = (
                raw_candidates.get("routes")
                or raw_candidates.get("paths")
                or raw_candidates.get("alternatives")
                or []
            )

        return list(raw_candidates)

    # ==========================================================
    # Validate alternative suffix
    # ==========================================================

    def is_suffix_valid(
        self,
        suffix,
        amount=0
    ):
        """
        Check whether every channel in an alternative suffix
        can currently carry the payment.

        This method uses current network state.

        It does NOT generate a new route.
        """

        if not suffix:
            return False

        if len(suffix) < 2:
            return False

        for u, v in zip(
            suffix[:-1],
            suffix[1:]
        ):

            if not self._edge_exists(u, v):
                return False

            if not self._nodes_available(u, v):
                return False

            if not self._channel_available(
                u,
                v,
                amount
            ):
                return False

        return True

    # ==========================================================
    # Normalize Bucket suffix
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node
    ):
        """
        Convert different Bucket representations into a route list.

        Supported examples:

            [B, F, G, E]

        or:

            {
                "path": [B, F, G, E]
            }

        or:

            {
                "route": [B, F, G, E]
            }

        or:

            {
                "suffix": [B, F, G, E]
            }
        """

        if candidate is None:
            return None

        if isinstance(
            candidate,
            (list, tuple)
        ):

            suffix = list(candidate)

        elif isinstance(candidate, dict):

            suffix = (
                candidate.get("suffix")
                or candidate.get("path")
                or candidate.get("route")
            )

            if suffix is None:
                return None

            suffix = list(suffix)

        else:

            return None

        if len(suffix) < 2:
            return None

        if suffix[0] != branch_node:
            return None

        return suffix

    # ==========================================================
    # Resolve failed edge
    # ==========================================================

    def _resolve_failed_edge(
        self,
        route,
        failed_edge,
        failure_index
    ):
        """
        Resolve exact failed directed edge.
        """

        if failed_edge is not None:

            if len(failed_edge) < 2:
                return None

            return (
                failed_edge[0],
                failed_edge[1]
            )

        if failure_index is not None:

            if (
                0 <= failure_index < len(route) - 1
            ):

                return (
                    route[failure_index],
                    route[failure_index + 1]
                )

        return None

    # ==========================================================
    # Resolve failure index
    # ==========================================================

    def _resolve_failure_index(
        self,
        route,
        failed_edge,
        failure_index
    ):
        """
        Resolve edge index inside route.
        """

        if failure_index is not None:

            if (
                0 <= failure_index < len(route) - 1
            ):

                u = route[failure_index]
                v = route[failure_index + 1]

                if (
                    u == failed_edge[0]
                    and
                    v == failed_edge[1]
                ):
                    return failure_index

        for index, (
            u,
            v
        ) in enumerate(
            zip(
                route[:-1],
                route[1:]
            )
        ):

            if (
                u == failed_edge[0]
                and
                v == failed_edge[1]
            ):
                return index

        return None

    # ==========================================================
    # Check failed edge reuse
    # ==========================================================

    def _contains_failed_edge(
        self,
        suffix,
        failed_edge
    ):
        """
        Prevent reuse of the failed directed channel.
        """

        if failed_edge is None:
            return False

        failed_u = failed_edge[0]
        failed_v = failed_edge[1]

        for u, v in zip(
            suffix[:-1],
            suffix[1:]
        ):

            if (
                u == failed_u
                and
                v == failed_v
            ):
                return True

        return False

    # ==========================================================
    # Loop prevention
    # ==========================================================

    def _is_loop_free(
        self,
        prefix,
        suffix
    ):
        """
        Ensure the combined route does not contain repeated nodes.
        """

        combined = self._combine_route(
            prefix,
            suffix
        )

        return len(combined) == len(
            set(combined)
        )

    # ==========================================================
    # Combine prefix and suffix
    # ==========================================================

    def _combine_route(
        self,
        prefix,
        suffix
    ):
        """
        Combine:

            prefix = A -> B

        with:

            suffix = B -> F -> G -> E

        into:

            A -> B -> F -> G -> E
        """

        if not prefix:
            return list(suffix)

        if not suffix:
            return list(prefix)

        if prefix[-1] == suffix[0]:

            return list(prefix) + list(
                suffix[1:]
            )

        return list(prefix) + list(suffix)

    # ==========================================================
    # Network checks
    # ==========================================================

    def _edge_exists(
        self,
        u,
        v
    ):
        """
        Check whether a directed channel exists.
        """

        if self.network is None:
            return False

        try:

            return self.network.has_edge(
                u,
                v
            )

        except Exception:

            return False

    # ----------------------------------------------------------

    def _nodes_available(
        self,
        u,
        v
    ):
        """
        Check source and destination node availability.
        """

        try:

            u_data = self.network.nodes[u]
            v_data = self.network.nodes[v]

        except Exception:

            return False

        u_available = u_data.get(
            "available",
            u_data.get(
                "is_online",
                True
            )
        )

        v_available = v_data.get(
            "available",
            v_data.get(
                "is_online",
                True
            )
        )

        return (
            bool(u_available)
            and
            bool(v_available)
        )

    # ----------------------------------------------------------

    def _channel_available(
        self,
        u,
        v,
        amount
    ):
        """
        Check whether at least one directed channel u -> v
        can carry the payment amount.
        """

        try:

            edge_data = self.network.get_edge_data(
                u,
                v
            )

        except Exception:

            return False

        if edge_data is None:
            return False

        # ------------------------------------------------------
        # MultiDiGraph
        # ------------------------------------------------------

        if self._is_multigraph():

            for key, data in edge_data.items():

                if self._edge_data_usable(
                    data,
                    amount
                ):
                    return True

            return False

        # ------------------------------------------------------
        # DiGraph
        # ------------------------------------------------------

        return self._edge_data_usable(
            edge_data,
            amount
        )

    # ----------------------------------------------------------

    def _edge_data_usable(
        self,
        data,
        amount
    ):
        """
        Check availability and directional liquidity.
        """

        if not data:
            return False

        available = data.get(
            "available",
            True
        )

        if not available:
            return False

        capacity = data.get(
            "capacity",
            float("inf")
        )

        try:

            if float(capacity) < float(amount):
                return False

        except (
            TypeError,
            ValueError
        ):

            return False

        # ------------------------------------------------------
        # Directional balance
        # ------------------------------------------------------

        balance_uv = data.get(
            "balance_uv",
            None
        )

        if balance_uv is not None:

            try:

                if float(balance_uv) < float(amount):
                    return False

            except (
                TypeError,
                ValueError
            ):

                return False

        # ------------------------------------------------------
        # Generic liquidity field
        # ------------------------------------------------------

        liquidity = data.get(
            "liquidity",
            None
        )

        if liquidity is not None:

            try:

                if float(liquidity) < float(amount):
                    return False

            except (
                TypeError,
                ValueError
            ):

                return False

        return True

    # ==========================================================
    # Usable outgoing edge
    # ==========================================================

    def _has_usable_outgoing_edge(
        self,
        node,
        amount=0,
        excluded_edge=None
    ):
        """
        Determine whether a node can serve as a potential
        branch point.

        This is only a feasibility check.
        It does not select a route.
        """

        if self.network is None:
            return False

        try:

            neighbors = self.network.successors(
                node
            )

        except Exception:

            return False

        for neighbor in neighbors:

            if excluded_edge is not None:

                if (
                    node == excluded_edge[0]
                    and
                    neighbor == excluded_edge[1]
                ):
                    continue

            if self._channel_available(
                node,
                neighbor,
                amount
            ):

                return True

        return False

    # ==========================================================
    # Route validation
    # ==========================================================

    @staticmethod
    def _validate_route(
        route
    ):
        """
        Basic route validation.
        """

        if not isinstance(
            route,
            (list, tuple)
        ):
            return False

        if len(route) < 2:
            return False

        # No repeated nodes.
        if len(route) != len(
            set(route)
        ):
            return False

        # Source and destination must differ.
        if route[0] == route[-1]:
            return False

        return True

    # ==========================================================
    # MultiGraph detection
    # ==========================================================

    def _is_multigraph(self):
        """
        Detect NetworkX MultiGraph / MultiDiGraph.
        """

        return bool(
            getattr(
                self.network,
                "is_multigraph",
                lambda: False
            )()
        )


# ==============================================================
# Backward-compatible alias
# ==============================================================

Backtrack = PartialBacktracker


# ==============================================================
# Simple standalone test
# ==============================================================

if __name__ == "__main__":

    import networkx as nx

    # ----------------------------------------------------------
    # Build a small directed Lightning-like graph
    # ----------------------------------------------------------

    G = nx.MultiDiGraph()

    edges = [

        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),

        # Alternative branch
        ("B", "F"),
        ("F", "G"),
        ("G", "E")

    ]

    for u, v in edges:

        G.add_edge(

            u,
            v,

            capacity=10000,

            balance_uv=10000,

            available=True

        )

    # ----------------------------------------------------------
    # Simple Bucket mock
    # ----------------------------------------------------------

    class TestBucket:

        def get_alternative_routes(
            self,
            bucket_id=None,
            route=None,
            branch_node=None,
            failed_edge=None
        ):

            return [

                {
                    "route": [
                        "B",
                        "F",
                        "G",
                        "E"
                    ]
                }

            ]

    # ----------------------------------------------------------
    # Create Backtracker
    # ----------------------------------------------------------

    backtracker = PartialBacktracker(

        network=G,

        bucket=TestBucket()

    )

    # ----------------------------------------------------------
    # Original route
    # ----------------------------------------------------------

    route = [

        "A",
        "B",
        "C",
        "D",
        "E"

    ]

    # Failure: C -> D
    result = backtracker.backtrack(

        route=route,

        failed_edge=("C", "D"),

        failure_index=2,

        amount=1000,

        bucket_id="B001",

        attempt_id=0

    )

    print("=" * 70)
    print("PARTIAL BACKTRACKING TEST")
    print("=" * 70)

    print(result)