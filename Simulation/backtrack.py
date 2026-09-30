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
    Top-K Candidate Routes
        |
        v
    CandidateManager
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
                    Full reroute
"""

from typing import Any


class PartialBacktracker:
    """
    Partial backtracking using alternative routes stored in Bucket.

    A previously failed Bucket candidate is never reused.

    Failed channel matching supports:

        (u, v)

    and, when candidate edge information is available:

        (u, v, key)

    This is important for NetworkX MultiDiGraph representations.
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

        The branch point is selected based on:

        1. network feasibility
        2. Bucket candidate availability
        3. candidate validity
        4. failed-candidate exclusion
        5. loop prevention
        6. failed-channel exclusion
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
        # 4. Find branch point WITH valid Bucket alternatives
        # ------------------------------------------------------

        branch_info = self._find_bucket_branch_point(
            route=route,
            failure_index=resolved_failure_index,
            failed_edge=resolved_failed_edge,
            amount=amount,
            bucket_id=bucket_id
        )

        if branch_info is None:

            return self._result(
                success=False,
                status="full_reroute_required",
                reason="no_valid_bucket_alternative",
                original_route=route,
                failed_edge=resolved_failed_edge,
                failure_index=resolved_failure_index,
                full_reroute_required=True,
                attempt_id=attempt_id
            )

        branch_index = branch_info["branch_index"]
        branch_node = branch_info["branch_node"]
        candidates = branch_info["candidates"]

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
        # 5. Evaluate candidates
        # ------------------------------------------------------

        for candidate_index, candidate in enumerate(candidates):

            print()
            print(
                f"  Evaluating alternative "
                f"{candidate_index + 1}: {candidate}"
            )

            # --------------------------------------------------
            # Previously failed candidate
            # --------------------------------------------------

            if self._candidate_is_failed(candidate):

                print(
                    "    REJECTED: candidate previously failed"
                )

                continue

            # --------------------------------------------------
            # Extract candidate path
            # --------------------------------------------------

            candidate_path = self._extract_candidate_path(
                candidate
            )

            # --------------------------------------------------
            # Do not retry the exact current route
            # --------------------------------------------------

            if (
                candidate_path is not None
                and
                self._same_route(
                    candidate_path,
                    route
                )
            ):

                print(
                    "    REJECTED: candidate is identical "
                    "to current route"
                )

                continue

            # --------------------------------------------------
            # Normalize suffix
            # --------------------------------------------------

            suffix = self._normalize_suffix(
                candidate=candidate,
                branch_node=branch_node
            )

            if suffix is None:

                print(
                    "    REJECTED: candidate does not contain "
                    "the selected branch point"
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

            if self._candidate_contains_failed_edge(
                candidate=candidate,
                suffix=suffix,
                failed_edge=resolved_failed_edge
            ):

                print(
                    "    REJECTED: failed channel reused"
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
        # 6. No valid candidate
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
    # Bucket-aware Branch Point
    # ==========================================================

    def _find_bucket_branch_point(
        self,
        route,
        failure_index,
        failed_edge,
        amount=0,
        bucket_id=None
    ):
        """
        Find the nearest branch point that has at least one
        currently valid Bucket alternative.

        Previously failed candidates are excluded.
        """

        if failure_index is None:
            return None

        if failure_index < 0:
            return None

        if failure_index >= len(route) - 1:
            return None

        start_index = failure_index

        for index in range(
            start_index,
            -1,
            -1
        ):

            node = route[index]

            # --------------------------------------------------
            # Exclude failed edge at its origin
            # --------------------------------------------------

            excluded_edge = None

            if index == failure_index:

                excluded_edge = (
                    route[failure_index],
                    route[failure_index + 1]
                )

            # --------------------------------------------------
            # Network feasibility
            # --------------------------------------------------

            if not self._has_usable_outgoing_edge(
                node=node,
                amount=amount,
                excluded_edge=excluded_edge
            ):

                continue

            # --------------------------------------------------
            # Get Bucket candidates
            # --------------------------------------------------

            candidates = self.get_alternative_suffixes(
                route=route,
                branch_index=index,
                failed_edge=failed_edge,
                bucket_id=bucket_id
            )

            valid_candidates = []

            # --------------------------------------------------
            # Validate each candidate
            # --------------------------------------------------

            for candidate in candidates:

                # ----------------------------------------------
                # Never reuse failed candidate
                # ----------------------------------------------

                if self._candidate_is_failed(candidate):

                    continue

                # ----------------------------------------------
                # Extract candidate path
                # ----------------------------------------------

                candidate_path = self._extract_candidate_path(
                    candidate
                )

                # ----------------------------------------------
                # Do not reuse current route
                # ----------------------------------------------

                if (
                    candidate_path is not None
                    and
                    self._same_route(
                        candidate_path,
                        route
                    )
                ):

                    continue

                # ----------------------------------------------
                # Normalize suffix
                # ----------------------------------------------

                suffix = self._normalize_suffix(
                    candidate=candidate,
                    branch_node=node
                )

                if suffix is None:
                    continue

                if len(suffix) < 2:
                    continue

                # ----------------------------------------------
                # Must begin at branch
                # ----------------------------------------------

                if suffix[0] != node:
                    continue

                # ----------------------------------------------
                # Failed channel
                # ----------------------------------------------

                if self._candidate_contains_failed_edge(
                    candidate=candidate,
                    suffix=suffix,
                    failed_edge=failed_edge
                ):

                    continue

                # ----------------------------------------------
                # Loop-free
                # ----------------------------------------------

                prefix = route[:index + 1]

                if not self._is_loop_free(
                    prefix=prefix,
                    suffix=suffix
                ):

                    continue

                # ----------------------------------------------
                # Channel validation
                # ----------------------------------------------

                if not self.is_suffix_valid(
                    suffix=suffix,
                    amount=amount
                ):

                    continue

                valid_candidates.append(
                    candidate
                )

            # --------------------------------------------------
            # Found valid branch
            # --------------------------------------------------

            if valid_candidates:

                return {
                    "branch_index": index,
                    "branch_node": node,
                    "candidates": valid_candidates
                }

        return None

    # ==========================================================
    # Legacy Network-only Branch Point
    # ==========================================================

    def find_nearest_branch_point(
        self,
        route,
        failure_index,
        amount=0
    ):
        """
        Legacy API.

        Finds nearest network-feasible branch point.
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

        Supported:

            [A, B, F, G, E]

            {
                "path": [...]
            }

            {
                "route": [...]
            }

            {
                "suffix": [...]
            }

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
        # Interface 3
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
        # Interface 4
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

        try:

            return list(raw_candidates)

        except Exception:

            return []

    # ==========================================================
    # Candidate Status
    # ==========================================================

    @staticmethod
    def _candidate_is_failed(
        candidate
    ):
        """
        Return True if a Bucket candidate has already failed.

        Candidate dictionaries generated by top_k_paths use:

            "success": None

        before execution,

            "success": True

        after success,

        and:

            "success": False

        after failure.
        """

        if not isinstance(
            candidate,
            dict
        ):

            return False

        return candidate.get(
            "success",
            None
        ) is False

    # ==========================================================
    # Candidate Path Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_path(
        candidate
    ):
        """
        Extract complete route/path from a candidate.
        """

        if candidate is None:
            return None

        # ------------------------------------------------------
        # Dictionary
        # ------------------------------------------------------

        if isinstance(
            candidate,
            dict
        ):

            if "path" in candidate:

                path = candidate["path"]

            elif "route" in candidate:

                path = candidate["route"]

            elif "suffix" in candidate:

                path = candidate["suffix"]

            else:

                return None

            try:

                return list(path)

            except Exception:

                return None

        # ------------------------------------------------------
        # Tuple
        # ------------------------------------------------------

        if isinstance(
            candidate,
            tuple
        ):

            if len(candidate) == 0:
                return None

            first = candidate[0]

            if isinstance(
                first,
                (list, tuple)
            ):

                return list(first)

            return None

        # ------------------------------------------------------
        # List
        # ------------------------------------------------------

        if isinstance(
            candidate,
            list
        ):

            return list(candidate)

        return None

    # ==========================================================
    # Candidate Normalization
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node
    ):
        """
        Convert a Bucket candidate into a suffix beginning
        at branch_node.

        A complete candidate route that does not contain the
        branch node is rejected.
        """

        if candidate is None:
            return None

        path = self._extract_candidate_path(
            candidate
        )

        if path is None:
            return None

        if len(path) < 2:
            return None

        if branch_node not in path:
            return None

        branch_position = path.index(
            branch_node
        )

        suffix = path[
            branch_position:
        ]

        if len(suffix) < 2:
            return None

        if suffix[0] != branch_node:
            return None

        return suffix

    # ==========================================================
    # Candidate Failed-Channel Check
    # ==========================================================

    def _candidate_contains_failed_edge(
        self,
        candidate,
        suffix,
        failed_edge
    ):
        """
        Check whether a candidate reuses the failed channel.

        If candidate edge information is available, compare:

            (u, v, key)

        exactly.

        If no edge-key information is available, fall back to:

            (u, v)

        which is conservative for MultiDiGraph.
        """

        if failed_edge is None:
            return False

        if len(failed_edge) < 2:
            return False

        failed_u = failed_edge[0]
        failed_v = failed_edge[1]

        failed_key = None

        if len(failed_edge) >= 3:
            failed_key = failed_edge[2]

        # ------------------------------------------------------
        # Prefer exact candidate edge information
        # ------------------------------------------------------

        candidate_edges = None

        if isinstance(
            candidate,
            dict
        ):

            candidate_edges = candidate.get(
                "edges"
            )

        elif isinstance(
            candidate,
            tuple
        ):

            if len(candidate) >= 2:

                possible_edges = candidate[1]

                if isinstance(
                    possible_edges,
                    (list, tuple)
                ):

                    candidate_edges = possible_edges

        # ------------------------------------------------------
        # Exact edge comparison
        # ------------------------------------------------------

        if candidate_edges is not None:

            for edge in candidate_edges:

                if not isinstance(
                    edge,
                    (list, tuple)
                ):

                    continue

                if len(edge) < 2:
                    continue

                u = edge[0]
                v = edge[1]

                if (
                    u != failed_u
                    or
                    v != failed_v
                ):

                    continue

                # If failed key is known and candidate key is
                # also known, compare exact channel.
                if (
                    failed_key is not None
                    and
                    len(edge) >= 3
                ):

                    if edge[2] == failed_key:
                        return True

                    # Same directed endpoints but a different
                    # channel key: this is a different channel.
                    continue

                # No exact key available -> conservative.
                return True

            return False

        # ------------------------------------------------------
        # Fallback to path-level comparison
        # ------------------------------------------------------

        return self._contains_failed_edge(
            suffix=suffix,
            failed_edge=failed_edge
        )

    # ==========================================================
    # Suffix Validation
    # ==========================================================

    def is_suffix_valid(
        self,
        suffix,
        amount=0
    ):
        """
        Validate every channel in suffix.
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

                return False

            if not self._nodes_available(
                u,
                v
            ):

                return False

            if not self._channel_available(
                u,
                v,
                amount
            ):

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
        Resolve failed edge position in route.
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

        for index, (u, v) in enumerate(
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

    @staticmethod
    def _contains_failed_edge(
        suffix,
        failed_edge
    ):
        """
        Path-level fallback check.

        This checks directed endpoints:

            u -> v

        and is used only when exact candidate edge-key
        information is unavailable.
        """

        if failed_edge is None:
            return False

        if len(failed_edge) < 2:
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
    # Same Route
    # ==========================================================

    @staticmethod
    def _same_route(
        route_a,
        route_b
    ):
        """
        Compare two routes.
        """

        if route_a is None or route_b is None:
            return False

        try:

            return list(route_a) == list(route_b)

        except Exception:

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
        Check complete route for repeated nodes.
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

    @staticmethod
    def _combine_route(
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
                prefix +
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

        if self._is_multigraph():

            for _, data in edge_data.items():

                if self._edge_data_usable(
                    data,
                    amount
                ):

                    return True

            return False

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

        try:

            if len(route) != len(set(route)):
                return False

        except TypeError:

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

    G = nx.MultiDiGraph()

    edges = [

        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),

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

    backtracker = PartialBacktracker(
        network=G,
        bucket=TestBucket()
    )

    route = [
        "A",
        "B",
        "C",
        "D",
        "E"
    ]

    failed_edge = (
        "C",
        "D",
        0
    )

    result = backtracker.backtrack(
        route=route,
        failed_edge=failed_edge,
        failure_index=2,
        amount=1000,
        bucket_id="B001",
        attempt_id=0
    )

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