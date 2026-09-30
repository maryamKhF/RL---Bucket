
"""
Simulation/backtrack.py

Partial Backtracking for RL + Bucket routing.

Flow:

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
"""

from typing import Any, List, Optional


class PartialBacktracker:
    """
    Partial backtracking using alternative routes already
    stored in Bucket.

    Example:

        Failed route:
            A -> B -> C -> D -> E

        Failed edge:
            C -> D

        Branch point:
            B

        Alternative suffix:
            B -> F -> G -> E

        New route:
            A -> B -> F -> G -> E
    """

    def __init__(
        self,
        network,
        bucket=None
    ):
        self.network = network
        self.bucket = bucket

    # ==========================================================
    # Main API
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
        Perform partial backtracking.

        Returns a structured result dictionary.
        """

        # ------------------------------------------------------
        # 1. Validate route
        # ------------------------------------------------------

        if not self._validate_route(route):

            return self._result(
                success=False,
                status="invalid_route",
                reason="invalid_route",
                original_route=route,
                attempt_id=attempt_id
            )

        route = list(route)

        # ------------------------------------------------------
        # 2. Resolve failed edge
        # ------------------------------------------------------

        resolved_failed_edge = self._resolve_failed_edge(
            route=route,
            failed_edge=failed_edge,
            failure_index=failure_index
        )

        if resolved_failed_edge is None:

            return self._result(
                success=False,
                status="invalid_failure",
                reason="failed_edge_not_resolved",
                original_route=route,
                attempt_id=attempt_id
            )

        # ------------------------------------------------------
        # 3. Resolve failure index
        # ------------------------------------------------------

        resolved_failure_index = self._resolve_failure_index(
            route=route,
            failed_edge=resolved_failed_edge,
            failure_index=failure_index
        )

        if resolved_failure_index is None:

            return self._result(
                success=False,
                status="invalid_failure",
                reason="failure_index_not_resolved",
                original_route=route,
                failed_edge=resolved_failed_edge,
                attempt_id=attempt_id
            )

        # ------------------------------------------------------
        # 4. Find nearest branch point
        # ------------------------------------------------------

        branch_index = self.find_nearest_branch_point(
            route=route,
            failure_index=resolved_failure_index,
            amount=amount
        )

        if branch_index is None:

            return self._result(
                success=False,
                status="no_branch_point",
                reason="no_valid_branch_point",
                original_route=route,
                failed_edge=resolved_failed_edge,
                failure_index=resolved_failure_index,
                full_reroute_required=True,
                attempt_id=attempt_id
            )

        branch_node = route[branch_index]

        # ------------------------------------------------------
        # 5. Get alternatives from Bucket
        # ------------------------------------------------------

        candidates = self.get_alternative_suffixes(
            route=route,
            branch_index=branch_index,
            failed_edge=resolved_failed_edge,
            bucket_id=bucket_id
        )

        # ------------------------------------------------------
        # Debug
        # ------------------------------------------------------

        print()
        print("[Backtracker]")
        print(f"  Branch node       : {branch_node}")
        print(f"  Branch index      : {branch_index}")
        print(f"  Failed edge       : {resolved_failed_edge}")
        print(f"  Failure index     : {resolved_failure_index}")
        print(f"  Bucket candidates : {len(candidates)}")

        # ------------------------------------------------------
        # 6. Evaluate candidates
        # ------------------------------------------------------

        for candidate_index, candidate in enumerate(
            candidates
        ):

            print()
            print(
                f"  Evaluating alternative "
                f"{candidate_index + 1}: {candidate}"
            )

            suffix = self._normalize_suffix(
                candidate=candidate,
                branch_node=branch_node
            )

            if suffix is None:

                print(
                    "    REJECTED: invalid suffix format"
                )

                continue

            print(
                f"    Normalized suffix: {suffix}"
            )

            # --------------------------------------------------
            # Must start at branch node
            # --------------------------------------------------

            if suffix[0] != branch_node:

                print(
                    "    REJECTED: suffix does not start "
                    "at branch node"
                )

                continue

            # --------------------------------------------------
            # Failed edge must not be reused
            # --------------------------------------------------

            if self._contains_failed_edge(
                suffix,
                resolved_failed_edge
            ):

                print(
                    "    REJECTED: failed edge reused"
                )

                continue

            # --------------------------------------------------
            # Prevent loops
            # --------------------------------------------------

            prefix = route[:branch_index + 1]

            if not self._is_loop_free(
                prefix=prefix,
                suffix=suffix
            ):

                print(
                    "    REJECTED: route would contain a loop"
                )

                continue

            # --------------------------------------------------
            # Validate suffix channels
            # --------------------------------------------------

            if not self.is_suffix_valid(
                suffix=suffix,
                amount=amount
            ):

                print(
                    "    REJECTED: suffix channel validation failed"
                )

                continue

            # --------------------------------------------------
            # Construct complete route
            # --------------------------------------------------

            new_route = self._combine_route(
                prefix=prefix,
                suffix=suffix
            )

            print(
                f"    New route candidate: {new_route}"
            )

            if not self._validate_route(new_route):

                print(
                    "    REJECTED: resulting route invalid"
                )

                continue

            # --------------------------------------------------
            # Success
            # --------------------------------------------------

            print(
                "    ACCEPTED"
            )

            return {
                "success": True,
                "status": "alternative_found",
                "reason": "partial_backtrack_success",

                "original_route": route,
                "new_route": new_route,

                "preserved_prefix": prefix,
                "alternative_suffix": suffix,

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
        # 7. No valid alternative
        # ------------------------------------------------------

        return {
            "success": False,
            "status": "full_reroute_required",
            "reason": "no_valid_bucket_alternative",

            "original_route": route,

            "preserved_prefix": None,
            "alternative_suffix": None,
            "new_route": None,

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
    # Branch Point
    # ==========================================================

    def find_nearest_branch_point(
        self,
        route,
        failure_index,
        amount=0
    ):
        """
        Find nearest node before failure with at least one
        usable outgoing edge.

        Example:

            A -> B -> C -> D -> E
                     ^
                     |
                  failure

        Search:

            C
            B
            A
        """

        if failure_index is None:
            return None

        if failure_index < 0:
            return None

        if failure_index >= len(route) - 1:
            return None

        for index in range(
            failure_index,
            -1,
            -1
        ):

            node = route[index]

            # Exclude the failed edge only at its origin.
            excluded_edge = None

            if index == failure_index:

                excluded_edge = (
                    route[failure_index],
                    route[failure_index + 1]
                )

            if self._has_usable_outgoing_edge(
                node=node,
                amount=amount,
                excluded_edge=excluded_edge
            ):

                return index

        return None

    # ==========================================================
    # Bucket Alternatives
    # ==========================================================

    def get_alternative_suffixes(
        self,
        route,
        branch_index,
        failed_edge,
        bucket_id=None
    ):
        """
        Extract candidate routes from Bucket.

        Supports:

            bucket.get_alternative_routes(...)
            bucket.get_routes(...)
            bucket.routes
            bucket.candidates

        Candidate representations may be:

            [A, B, F, G, E]

        or:

            {
                "path": [A, B, F, G, E]
            }

        or:

            {
                "route": [A, B, F, G, E]
            }

        or:

            {
                "suffix": [B, F, G, E]
            }

        or the project Bucket candidate tuple:

            (
                path,
                edges,
                score
            )
        """

        if self.bucket is None:
            return []

        raw_candidates = []

        # ------------------------------------------------------
        # Interface 1
        # ------------------------------------------------------

        method = getattr(
            self.bucket,
            "get_alternative_routes",
            None
        )

        if callable(method):

            try:

                result = method(
                    bucket_id=bucket_id,
                    route=route,
                    branch_node=route[branch_index],
                    failed_edge=failed_edge
                )

                if result is not None:
                    raw_candidates = result

            except TypeError:

                try:

                    raw_candidates = method(
                        bucket_id
                    )

                except Exception:

                    raw_candidates = []

            except Exception:

                raw_candidates = []

        # ------------------------------------------------------
        # Interface 2
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

                except Exception:

                    raw_candidates = []

        # ------------------------------------------------------
        # Interface 3: routes
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
        # Interface 4: candidates
        # ------------------------------------------------------

        if not raw_candidates:

            candidates = getattr(
                self.bucket,
                "candidates",
                None
            )

            if candidates is not None:
                raw_candidates = candidates

        # ------------------------------------------------------
        # Normalize dictionary container
        # ------------------------------------------------------

        if isinstance(
            raw_candidates,
            dict
        ):

            raw_candidates = (
                raw_candidates.get("routes")
                or raw_candidates.get("paths")
                or raw_candidates.get("alternatives")
                or raw_candidates.get("candidates")
                or []
            )

        if raw_candidates is None:
            return []

        return list(raw_candidates)

    # ==========================================================
    # Candidate Normalization
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node
    ):
        """
        Normalize Bucket candidate into a route/suffix.

        Important:
        A complete Bucket route such as

            A -> B -> F -> G -> E

        is converted into

            B -> F -> G -> E

        when branch_node = B.
        """

        if candidate is None:
            return None

        path = None

        # ------------------------------------------------------
        # Dict representation
        # ------------------------------------------------------

        if isinstance(candidate, dict):

            path = (
                candidate.get("suffix")
                or candidate.get("path")
                or candidate.get("route")
            )

        # ------------------------------------------------------
        # Project candidate tuple:
        #
        # (
        #     path,
        #     edges,
        #     score
        # )
        # ------------------------------------------------------

        elif isinstance(candidate, tuple):

            if len(candidate) >= 3:

                first = candidate[0]

                if isinstance(
                    first,
                    (list, tuple)
                ):

                    path = first

            elif len(candidate) >= 2:

                path = candidate

        # ------------------------------------------------------
        # Plain path
        # ------------------------------------------------------

        elif isinstance(
            candidate,
            list
        ):

            path = candidate

        # ------------------------------------------------------
        # Unsupported
        # ------------------------------------------------------

        if path is None:
            return None

        path = list(path)

        if len(path) < 2:
            return None

        # ------------------------------------------------------
        # If candidate is a complete route, extract suffix.
        #
        # Example:
        #
        # route:
        #     A B F G E
        #
        # branch:
        #     B
        #
        # suffix:
        #     B F G E
        # ------------------------------------------------------

        if branch_node in path:

            branch_position = path.index(
                branch_node
            )

            suffix = path[
                branch_position:
            ]

        else:

            suffix = path

        if len(suffix) < 2:
            return None

        if suffix[0] != branch_node:
            return None

        return suffix

    # ==========================================================
    # Suffix Validation
    # ==========================================================

    def is_suffix_valid(
        self,
        suffix,
        amount=0
    ):
        """
        Validate all channels in suffix.
        """

        if not suffix:
            return False

        if len(suffix) < 2:
            return False

        for u, v in zip(
            suffix[:-1],
            suffix[1:]
        ):

            if not self._edge_exists(
                u,
                v
            ):

                print(
                    f"    Invalid edge: {u} -> {v}"
                )

                return False

            if not self._nodes_available(
                u,
                v
            ):

                print(
                    f"    Unavailable node: {u} -> {v}"
                )

                return False

            if not self._channel_available(
                u,
                v,
                amount
            ):

                print(
                    f"    Insufficient/unavailable "
                    f"channel: {u} -> {v}"
                )

                return False

        return True

    # ==========================================================
    # Failed Edge Resolution
    # ==========================================================

    def _resolve_failed_edge(
        self,
        route,
        failed_edge,
        failure_index
    ):
        """
        Normalize failed edge to:

            (u, v)

        or:

            (u, v, key)

        The key is preserved for reporting but directed
        endpoint comparison uses only u and v.
        """

        if failed_edge is not None:

            if not isinstance(
                failed_edge,
                (list, tuple)
            ):

                return None

            if len(failed_edge) < 2:
                return None

            if len(failed_edge) >= 3:

                return (
                    failed_edge[0],
                    failed_edge[1],
                    failed_edge[2]
                )

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
    # Failure Index
    # ==========================================================

    def _resolve_failure_index(
        self,
        route,
        failed_edge,
        failure_index
    ):
        """
        Resolve failed edge position.
        """

        if failed_edge is None:
            return None

        failed_u = failed_edge[0]
        failed_v = failed_edge[1]

        if failure_index is not None:

            if (
                0 <= failure_index < len(route) - 1
            ):

                if (
                    route[failure_index] == failed_u
                    and
                    route[failure_index + 1] == failed_v
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
                u == failed_u
                and
                v == failed_v
            ):

                return index

        return None

    # ==========================================================
    # Failed Edge Reuse
    # ==========================================================

    def _contains_failed_edge(
        self,
        suffix,
        failed_edge
    ):
        """
        Check whether suffix reuses failed directed edge.
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
    # Loop Prevention
    # ==========================================================

    def _is_loop_free(
        self,
        prefix,
        suffix
    ):
        """
        Check complete combined route for repeated nodes.
        """

        combined = self._combine_route(
            prefix,
            suffix
        )

        return len(combined) == len(
            set(combined)
        )

    # ==========================================================
    # Combine
    # ==========================================================

    def _combine_route(
        self,
        prefix,
        suffix
    ):
        """
        Combine:

            A -> B

        with:

            B -> F -> G -> E

        into:

            A -> B -> F -> G -> E
        """

        prefix = list(prefix)
        suffix = list(suffix)

        if not prefix:
            return suffix

        if not suffix:
            return prefix

        if prefix[-1] == suffix[0]:

            return (
                prefix
                +
                suffix[1:]
            )

        return prefix + suffix

    # ==========================================================
    # Edge Exists
    # ==========================================================

    def _edge_exists(
        self,
        u,
        v
    ):
        if self.network is None:
            return False

        try:

            return self.network.has_edge(
                u,
                v
            )

        except Exception:

            return False

    # ==========================================================
    # Node Availability
    # ==========================================================

    def _nodes_available(
        self,
        u,
        v
    ):
        try:

            u_data = self.network.nodes[u]
            v_data = self.network.nodes[v]

        except Exception:

            return False

        u_available = u_data.get(
            "available",
            u_data.get(
                "online",
                u_data.get(
                    "is_online",
                    True
                )
            )
        )

        v_available = v_data.get(
            "available",
            v_data.get(
                "online",
                v_data.get(
                    "is_online",
                    True
                )
            )
        )

        return (
            bool(u_available)
            and
            bool(v_available)
        )

    # ==========================================================
    # Channel Availability
    # ==========================================================

    def _channel_available(
        self,
        u,
        v,
        amount
    ):
        """
        Check at least one usable channel u -> v.

        Supports DiGraph and MultiDiGraph.
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
        # MultiGraph
        # ------------------------------------------------------

        if self._is_multigraph():

            for _, data in edge_data.items():

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

    # ==========================================================
    # Edge Data Validation
    # ==========================================================

    def _edge_data_usable(
        self,
        data,
        amount
    ):
        if not data:
            return False

        if not data.get(
            "available",
            True
        ):

            return False

        # ------------------------------------------------------
        # Capacity
        # ------------------------------------------------------

        capacity = data.get(
            "capacity",
            None
        )

        if capacity is not None:

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

        balance = data.get(
            "balance_uv",
            None
        )

        if balance is not None:

            try:

                if float(balance) < float(amount):
                    return False

            except (
                TypeError,
                ValueError
            ):

                return False

        # ------------------------------------------------------
        # Generic liquidity
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
    # Usable Outgoing Edge
    # ==========================================================

    def _has_usable_outgoing_edge(
        self,
        node,
        amount=0,
        excluded_edge=None
    ):
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
    # Route Validation
    # ==========================================================

    @staticmethod
    def _validate_route(
        route
    ):
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

        if route[0] == route[-1]:
            return False

        return True

    # ==========================================================
    # MultiGraph Detection
    # ==========================================================

    def _is_multigraph(self):
        if self.network is None:
            return False

        return bool(
            getattr(
                self.network,
                "is_multigraph",
                lambda: False
            )()
        )

    # ==========================================================
    # Result Helper
    # ==========================================================

    @staticmethod
    def _result(
        success,
        status,
        reason,
        original_route,
        **kwargs
    ):
        result = {
            "success": success,
            "status": status,
            "reason": reason,
            "original_route": (
                list(original_route)
                if isinstance(
                    original_route,
                    (list, tuple)
                )
                else original_route
            )
        }

        result.update(kwargs)

        return result


# ==============================================================
# Backward-compatible alias
# ==============================================================

Backtrack = PartialBacktracker


# ==============================================================
# Standalone Test
# ==============================================================

if __name__ == "__main__":

    import networkx as nx

    # ----------------------------------------------------------
    # Create Lightning-like directed network
    # ----------------------------------------------------------

    G = nx.MultiDiGraph()

    edges = [

        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),

        # Alternative route
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

        G.nodes[u]["available"] = True
        G.nodes[v]["available"] = True

    # ----------------------------------------------------------
    # Bucket mock
    #
    # IMPORTANT:
    # Store the COMPLETE alternative route.
    # The Backtracker will extract:
    #
    # A-B-F-G-E
    #
    # into:
    #
    # B-F-G-E
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
                        "A",
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

    # ----------------------------------------------------------
    # Simulate failure
    # ----------------------------------------------------------

    failed_edge = (
        "C",
        "D",
        0
    )

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    result = backtracker.backtrack(

        route=route,

        failed_edge=failed_edge,

        failure_index=2,

        amount=1000,

        bucket_id="B001",

        attempt_id=0
    )

    # ----------------------------------------------------------
    # Report
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("PARTIAL BACKTRACKING TEST")
    print("=" * 70)

    print(
        f"Status              : "
        f"{result.get('status')}"
    )

    print(
        f"Success             : "
        f"{result.get('success')}"
    )

    print(
        f"Reason              : "
        f"{result.get('reason')}"
    )

    print(
        f"Branch point        : "
        f"{result.get('branch_point')}"
    )

    print(
        f"Failed edge         : "
        f"{result.get('failed_edge')}"
    )

    print(
        f"Failure index       : "
        f"{result.get('failure_index')}"
    )

    print(
        f"Original route      : "
        f"{result.get('original_route')}"
    )

    print(
        f"Preserved prefix    : "
        f"{result.get('preserved_prefix')}"
    )

    print(
        f"Alternative suffix  : "
        f"{result.get('alternative_suffix')}"
    )

    print(
        f"New route           : "
        f"{result.get('new_route')}"
    )

    print(
        f"Full reroute needed : "
        f"{result.get('full_reroute_required')}"
    )

    print(
        f"Attempt ID          : "
        f"{result.get('attempt_id')}"
    )

    print("=" * 70)

    if (
        result.get("success")
        and
        result.get("status") == "alternative_found"
    ):

        print(
            "PARTIAL BACKTRACKING STATUS : SUCCESS"
        )

    else:

        print(
            "PARTIAL BACKTRACKING STATUS : FAILED"
        )

