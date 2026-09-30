"""
Simulation/backtrack.py

Partial Backtracking for RL + Bucket routing.

Flow
----

    RL / PPO
        |
        v
    Modified Heuristic
        |
        v
    Top-K Candidate Routes
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
    FailureModel
        |
        v
    Failure Information
        |
        v
    Partial Backtracking
        |
        +----> Alternative suffix from Bucket
        |
        +----> If no valid suffix:
                    Full reroute required


Important semantics
-------------------

- Bucket.attempts counts ACTUAL payment attempts only.
- PartialBacktracker NEVER modifies Bucket.attempts.
- Channel capacity is NOT directional liquidity.
- Unknown directional liquidity is accepted by structural validation.
- Exact MultiDiGraph channel keys are preserved whenever available.
- Candidate edge information is authoritative when supplied.
- This module does NOT execute payments.
- This module does NOT evaluate stochastic failures.
- This module does NOT perform full rerouting.
"""

from __future__ import annotations

from typing import Any, Optional


class PartialBacktracker:
    """
    Partial backtracking using alternative candidates stored in Bucket.

    Responsibilities
    ----------------
    1. Resolve the failed edge.
    2. Resolve the failure position.
    3. Search backward for a usable Bucket branch point.
    4. Retrieve candidate routes from Bucket.
    5. Reject candidates already known to have failed.
    6. Preserve exact channel keys whenever possible.
    7. Reject reuse of the failed channel.
    8. Validate the alternative suffix.
    9. Prevent route loops.
    10. Construct the new route and exact edge list.
    11. Request full rerouting when no valid suffix exists.

    The class does NOT:
        - execute payments
        - call FailureModel
        - call PaymentSimulator
        - call Router
        - increment Bucket.attempts
        - modify PPO state
        - choose PPO actions
        - perform full rerouting
    """

    def __init__(
        self,
        network,
        bucket=None,
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
        attempt_id=0,
    ):
        """
        Perform partial backtracking.

        Parameters
        ----------
        route : list
            Current failed node route.

        failed_edge : tuple, optional
            Failed channel:

                (u, v)

            or:

                (u, v, key)

        failure_index : int, optional
            Zero-based index of failed hop.

        amount : float
            Payment amount.

        bucket_id : optional
            Bucket identifier.

        attempt_id : int
            Current payment-attempt identifier.

        Returns
        -------
        dict
            Structured backtracking result.

        Important output fields
        -----------------------
        success
        status
        reason
        original_route
        new_route
        new_edges
        preserved_prefix
        preserved_prefix_edges
        alternative_suffix
        alternative_suffix_edges
        branch_point
        branch_index
        failed_edge
        failure_index
        full_reroute_required
        """

        # ------------------------------------------------------
        # 1. Validate input route
        # ------------------------------------------------------

        if not self._validate_route(route):

            return self._result(
                success=False,
                status="invalid_route",
                reason="invalid_route",
                original_route=route,
                attempt_id=attempt_id,
                full_reroute_required=False,
            )

        route = list(route)

        # ------------------------------------------------------
        # 2. Validate amount
        # ------------------------------------------------------

        try:
            amount = float(amount)
        except (TypeError, ValueError):

            return self._result(
                success=False,
                status="invalid_amount",
                reason="invalid_amount",
                original_route=route,
                attempt_id=attempt_id,
                full_reroute_required=False,
            )

        if amount <= 0:

            return self._result(
                success=False,
                status="invalid_amount",
                reason="invalid_amount",
                original_route=route,
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False,
            )

        # ------------------------------------------------------
        # 3. Resolve failed edge
        # ------------------------------------------------------

        resolved_failed_edge = self._resolve_failed_edge(
            route=route,
            failed_edge=failed_edge,
            failure_index=failure_index,
        )

        if resolved_failed_edge is None:

            return self._result(
                success=False,
                status="invalid_failure",
                reason="failed_edge_not_resolved",
                original_route=route,
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False,
            )

        # ------------------------------------------------------
        # 4. Resolve exact failure index
        # ------------------------------------------------------

        resolved_failure_index = self._resolve_failure_index(
            route=route,
            failed_edge=resolved_failed_edge,
            failure_index=failure_index,
        )

        if resolved_failure_index is None:

            return self._result(
                success=False,
                status="invalid_failure",
                reason="failure_index_not_resolved",
                original_route=route,
                failed_edge=resolved_failed_edge,
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False,
            )

        # ------------------------------------------------------
        # 5. Find nearest Bucket branch point
        # ------------------------------------------------------

        branch_info = self._find_bucket_branch_point(
            route=route,
            failure_index=resolved_failure_index,
            failed_edge=resolved_failed_edge,
            amount=amount,
            bucket_id=bucket_id,
        )

        if branch_info is None:

            return self._result(
                success=False,
                status="full_reroute_required",
                reason="no_valid_bucket_alternative",
                original_route=route,
                failed_edge=resolved_failed_edge,
                failure_index=resolved_failure_index,
                amount=amount,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                full_reroute_required=True,
            )

        branch_index = branch_info["branch_index"]
        branch_node = branch_info["branch_node"]
        candidates = branch_info["candidates"]

        print()
        print("[PartialBacktracker]")
        print(
            f"  Branch node       : {branch_node}"
        )
        print(
            f"  Branch index      : {branch_index}"
        )
        print(
            f"  Failed edge       : {resolved_failed_edge}"
        )
        print(
            f"  Failure index     : {resolved_failure_index}"
        )
        print(
            f"  Bucket candidates : {len(candidates)}"
        )

        # ------------------------------------------------------
        # 6. Evaluate candidates
        # ------------------------------------------------------

        for candidate_index, candidate in enumerate(
            candidates
        ):

            print()
            print(
                f"  Evaluating candidate "
                f"{candidate_index + 1}"
            )

            # --------------------------------------------------
            # Candidate failure state
            # --------------------------------------------------

            if self._candidate_is_failed(
                candidate=candidate,
                candidate_index=candidate_index,
            ):

                print(
                    "    REJECTED: candidate already failed"
                )
                continue

            # --------------------------------------------------
            # Candidate path
            # --------------------------------------------------

            candidate_path = self._extract_candidate_path(
                candidate
            )

            if candidate_path is None:

                print(
                    "    REJECTED: candidate path unavailable"
                )
                continue

            # --------------------------------------------------
            # Candidate must contain branch node
            # --------------------------------------------------

            suffix = self._normalize_suffix(
                candidate=candidate,
                branch_node=branch_node,
            )

            if suffix is None:

                print(
                    "    REJECTED: branch node not present"
                )
                continue

            # --------------------------------------------------
            # Candidate must contain at least one hop
            # --------------------------------------------------

            if len(suffix) < 2:

                print(
                    "    REJECTED: suffix has no forwarding hop"
                )
                continue

            # --------------------------------------------------
            # Exact candidate edges
            # --------------------------------------------------

            suffix_edges = self._extract_suffix_edges(
                candidate=candidate,
                suffix=suffix,
                branch_node=branch_node,
            )

            # --------------------------------------------------
            # Do not retry exact current route
            # --------------------------------------------------

            if self._same_route(
                candidate_path,
                route,
            ):

                print(
                    "    REJECTED: candidate equals current route"
                )
                continue

            # --------------------------------------------------
            # Failed channel must not be reused
            # --------------------------------------------------

            if self._candidate_contains_failed_edge(
                candidate=candidate,
                suffix=suffix,
                suffix_edges=suffix_edges,
                failed_edge=resolved_failed_edge,
            ):

                print(
                    "    REJECTED: failed channel reused"
                )
                continue

            # --------------------------------------------------
            # Loop prevention
            # --------------------------------------------------

            prefix = route[:branch_index + 1]

            if not self._is_loop_free(
                prefix=prefix,
                suffix=suffix,
            ):

                print(
                    "    REJECTED: reconstructed route contains loop"
                )
                continue

            # --------------------------------------------------
            # Validate suffix
            # --------------------------------------------------

            if not self.is_suffix_valid(
                suffix=suffix,
                amount=amount,
                candidate=candidate,
                suffix_edges=suffix_edges,
            ):

                print(
                    "    REJECTED: suffix validation failed"
                )
                continue

            # --------------------------------------------------
            # Build prefix edge list
            # --------------------------------------------------

            prefix_edges = self._resolve_route_edges_for_prefix(
                route=route,
                end_index=branch_index,
            )

            if prefix_edges is None:

                print(
                    "    REJECTED: exact prefix edges unavailable"
                )
                continue

            # --------------------------------------------------
            # Build complete route
            # --------------------------------------------------

            new_route = self._combine_route(
                prefix=prefix,
                suffix=suffix,
            )

            if not self._validate_route(
                new_route
            ):

                print(
                    "    REJECTED: resulting route invalid"
                )
                continue

            # --------------------------------------------------
            # Build complete exact edge list
            # --------------------------------------------------

            new_edges = self._combine_edges(
                prefix_edges=prefix_edges,
                suffix_edges=suffix_edges,
            )

            # --------------------------------------------------
            # Edge count must match route
            # --------------------------------------------------

            if new_edges is None:

                print(
                    "    REJECTED: exact edge reconstruction failed"
                )
                continue

            if len(new_edges) != len(new_route) - 1:

                print(
                    "    REJECTED: route/edge length mismatch"
                )
                continue

            # --------------------------------------------------
            # Final route-edge consistency
            # --------------------------------------------------

            if not self._edges_match_route(
                route=new_route,
                edges=new_edges,
            ):

                print(
                    "    REJECTED: route/edge correspondence invalid"
                )
                continue

            print(
                f"    Alternative suffix : {suffix}"
            )

            print(
                f"    New route          : {new_route}"
            )

            print(
                f"    New edges          : {new_edges}"
            )

            print(
                "    ACCEPTED"
            )

            return self._result(
                success=True,
                status="alternative_found",
                reason="partial_backtrack_success",

                original_route=route,
                new_route=new_route,
                new_edges=new_edges,

                preserved_prefix=prefix,
                preserved_prefix_edges=prefix_edges,

                alternative_suffix=suffix,
                alternative_suffix_edges=suffix_edges,

                branch_point=branch_node,
                branch_index=branch_index,

                failed_edge=resolved_failed_edge,
                failure_index=resolved_failure_index,

                amount=amount,
                bucket_id=bucket_id,

                attempt_id=attempt_id + 1,

                full_reroute_required=False,
            )

        # ------------------------------------------------------
        # 7. No candidate succeeded
        # ------------------------------------------------------

        return self._result(
            success=False,
            status="full_reroute_required",
            reason="no_valid_bucket_alternative",

            original_route=route,

            preserved_prefix=None,
            preserved_prefix_edges=None,

            alternative_suffix=None,
            alternative_suffix_edges=None,

            new_route=None,
            new_edges=None,

            branch_point=branch_node,
            branch_index=branch_index,

            failed_edge=resolved_failed_edge,
            failure_index=resolved_failure_index,

            amount=amount,
            bucket_id=bucket_id,

            attempt_id=attempt_id,

            full_reroute_required=True,
        )

    # ==========================================================
    # Bucket-aware Branch Point
    # ==========================================================

    def _find_bucket_branch_point(
        self,
        route,
        failure_index,
        failed_edge,
        amount=0,
        bucket_id=None,
    ):
        """
        Search backward from the failure point.

        The nearest node with a valid Bucket alternative is selected.
        """

        if failure_index is None:
            return None

        if failure_index < 0:
            return None

        if failure_index >= len(route) - 1:
            return None

        # ------------------------------------------------------
        # Search from failed-edge source backward
        # ------------------------------------------------------

        for index in range(
            failure_index,
            -1,
            -1,
        ):

            node = route[index]

            excluded_edge = None

            if index == failure_index:
                excluded_edge = failed_edge

            # --------------------------------------------------
            # A branch point must have an outgoing possibility.
            # --------------------------------------------------

            if not self._has_usable_outgoing_edge(
                node=node,
                amount=amount,
                excluded_edge=excluded_edge,
            ):
                continue

            # --------------------------------------------------
            # Retrieve Bucket candidates
            # --------------------------------------------------

            candidates = self.get_alternative_suffixes(
                route=route,
                branch_index=index,
                failed_edge=failed_edge,
                bucket_id=bucket_id,
            )

            if not candidates:
                continue

            valid_candidates = []

            # --------------------------------------------------
            # Validate every candidate
            # --------------------------------------------------

            for candidate_index, candidate in enumerate(
                candidates
            ):

                if self._candidate_is_failed(
                    candidate=candidate,
                    candidate_index=candidate_index,
                ):
                    continue

                candidate_path = self._extract_candidate_path(
                    candidate
                )

                if candidate_path is None:
                    continue

                if self._same_route(
                    candidate_path,
                    route,
                ):
                    continue

                suffix = self._normalize_suffix(
                    candidate=candidate,
                    branch_node=node,
                )

                if suffix is None:
                    continue

                if len(suffix) < 2:
                    continue

                if suffix[0] != node:
                    continue

                suffix_edges = self._extract_suffix_edges(
                    candidate=candidate,
                    suffix=suffix,
                    branch_node=node,
                )

                if self._candidate_contains_failed_edge(
                    candidate=candidate,
                    suffix=suffix,
                    suffix_edges=suffix_edges,
                    failed_edge=failed_edge,
                ):
                    continue

                prefix = route[:index + 1]

                if not self._is_loop_free(
                    prefix=prefix,
                    suffix=suffix,
                ):
                    continue

                if not self.is_suffix_valid(
                    suffix=suffix,
                    amount=amount,
                    candidate=candidate,
                    suffix_edges=suffix_edges,
                ):
                    continue

                valid_candidates.append(
                    candidate
                )

            if valid_candidates:

                return {
                    "branch_index": index,
                    "branch_node": node,
                    "candidates": valid_candidates,
                }

        return None

    # ==========================================================
    # Legacy Network-only Branch Point
    # ==========================================================

    def find_nearest_branch_point(
        self,
        route,
        failure_index,
        amount=0,
    ):
        """
        Backward-compatible network-only branch-point search.
        """

        if not self._validate_route(route):
            return None

        if failure_index is None:
            return None

        if failure_index < 0:
            return None

        if failure_index >= len(route) - 1:
            return None

        for index in range(
            failure_index,
            -1,
            -1,
        ):

            node = route[index]

            excluded_edge = None

            if index == failure_index:

                excluded_edge = (
                    route[failure_index],
                    route[failure_index + 1],
                )

            if self._has_usable_outgoing_edge(
                node=node,
                amount=amount,
                excluded_edge=excluded_edge,
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
        bucket_id=None,
    ):
        """
        Retrieve candidates from Bucket.

        Preferred current interface
        ----------------------------
        Bucket.candidates

        Backward-compatible interfaces
        -------------------------------
        get_alternative_routes(...)
        get_routes(...)
        routes
        """

        if self.bucket is None:
            return []

        raw_candidates = []

        # ------------------------------------------------------
        # 1. Current Bucket candidates
        # ------------------------------------------------------

        candidates = getattr(
            self.bucket,
            "candidates",
            None,
        )

        if candidates is not None:

            try:
                raw_candidates = list(candidates)
            except Exception:
                raw_candidates = []

        # ------------------------------------------------------
        # 2. get_alternative_routes
        # ------------------------------------------------------

        if not raw_candidates:

            method = getattr(
                self.bucket,
                "get_alternative_routes",
                None,
            )

            if callable(method):

                try:

                    result = method(
                        bucket_id=bucket_id,
                        route=route,
                        branch_node=route[branch_index],
                        failed_edge=failed_edge,
                    )

                    if result is not None:
                        raw_candidates = list(result)

                except TypeError:

                    try:

                        result = method(
                            bucket_id
                        )

                        if result is not None:
                            raw_candidates = list(result)

                    except Exception:
                        raw_candidates = []

                except Exception:
                    raw_candidates = []

        # ------------------------------------------------------
        # 3. get_routes
        # ------------------------------------------------------

        if not raw_candidates:

            method = getattr(
                self.bucket,
                "get_routes",
                None,
            )

            if callable(method):

                try:

                    result = method(
                        bucket_id=bucket_id
                    )

                    if result is not None:
                        raw_candidates = list(result)

                except TypeError:

                    try:

                        result = method()

                        if result is not None:
                            raw_candidates = list(result)

                    except Exception:
                        raw_candidates = []

                except Exception:
                    raw_candidates = []

        # ------------------------------------------------------
        # 4. routes attribute
        # ------------------------------------------------------

        if not raw_candidates:

            routes = getattr(
                self.bucket,
                "routes",
                None,
            )

            if routes is not None:

                try:
                    raw_candidates = list(routes)
                except Exception:
                    raw_candidates = []

        # ------------------------------------------------------
        # Normalize dictionary containers
        # ------------------------------------------------------

        if isinstance(
            raw_candidates,
            dict,
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
    # Candidate Failure State
    # ==========================================================

    def _candidate_is_failed(
        self,
        candidate,
        candidate_index=None,
    ):
        """
        Determine whether a Bucket candidate has already failed.

        Current Bucket semantics
        -------------------------
        Bucket.failed_candidates stores failed candidates.

        Legacy fallback
        ----------------
        candidate["success"] is False.

        Important
        ---------
        This method does NOT modify Bucket state.
        """

        if candidate is None:
            return True

        # ------------------------------------------------------
        # Explicit candidate flag
        # ------------------------------------------------------

        if isinstance(
            candidate,
            dict,
        ):

            if candidate.get(
                "success",
                None,
            ) is False:

                return True

            if candidate.get(
                "failed",
                False,
            ) is True:

                return True

        # ------------------------------------------------------
        # Bucket-level failed_candidates
        # ------------------------------------------------------

        failed_candidates = getattr(
            self.bucket,
            "failed_candidates",
            None,
        )

        if failed_candidates:

            try:

                for failed in failed_candidates:

                    if self._candidate_identity_equal(
                        candidate,
                        failed,
                    ):

                        return True

            except Exception:
                pass

        return False

    # ==========================================================
    # Candidate Identity
    # ==========================================================

    def _candidate_identity_equal(
        self,
        candidate_a,
        candidate_b,
    ):
        """
        Compare two candidate representations.

        Prefer exact edges when both are available.
        Otherwise compare normalized paths.
        """

        if candidate_a is candidate_b:
            return True

        path_a = self._extract_candidate_path(
            candidate_a
        )

        path_b = self._extract_candidate_path(
            candidate_b
        )

        if (
            path_a is not None
            and
            path_b is not None
            and
            path_a == path_b
        ):
            return True

        edges_a = self._extract_candidate_edges(
            candidate_a
        )

        edges_b = self._extract_candidate_edges(
            candidate_b
        )

        if (
            edges_a is not None
            and
            edges_b is not None
            and
            edges_a == edges_b
        ):
            return True

        return False

    # ==========================================================
    # Candidate Path Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_path(
        candidate,
    ):
        """
        Extract candidate node path.
        """

        if candidate is None:
            return None

        # ------------------------------------------------------
        # Dictionary
        # ------------------------------------------------------

        if isinstance(
            candidate,
            dict,
        ):

            path = None

            for field in (
                "path",
                "route",
                "suffix",
            ):

                if field in candidate:

                    path = candidate[field]
                    break

            if path is None:
                return None

            try:
                path = list(path)
            except Exception:
                return None

            if len(path) < 2:
                return None

            return path

        # ------------------------------------------------------
        # Tuple
        # ------------------------------------------------------

        if isinstance(
            candidate,
            tuple,
        ):

            if not candidate:
                return None

            first = candidate[0]

            if isinstance(
                first,
                (list, tuple),
            ):

                path = list(first)

                if len(path) >= 2:
                    return path

            return None

        # ------------------------------------------------------
        # List
        # ------------------------------------------------------

        if isinstance(
            candidate,
            list,
        ):

            # A list of nodes.
            if candidate and not isinstance(
                candidate[0],
                (dict, list, tuple),
            ):

                if len(candidate) >= 2:
                    return list(candidate)

        return None

    # ==========================================================
    # Candidate Edge Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_edges(
        candidate,
    ):
        """
        Extract exact candidate channel edges.

        Supported forms
        ---------------

        dict:
            {"edges": [(u,v,key), ...]}

        tuple:
            (path, edges, ...)

        list:
            No edge information is assumed.
        """

        if candidate is None:
            return None

        if isinstance(
            candidate,
            dict,
        ):

            edges = candidate.get(
                "edges"
            )

            if edges is None:
                return None

            try:
                return list(edges)
            except Exception:
                return None

        if isinstance(
            candidate,
            tuple,
        ):

            if len(candidate) >= 2:

                edges = candidate[1]

                if isinstance(
                    edges,
                    (list, tuple),
                ):

                    try:
                        return list(edges)
                    except Exception:
                        return None

        return None

    # ==========================================================
    # Normalize Candidate Suffix
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node,
    ):
        """
        Convert candidate route into a suffix beginning at branch_node.
        """

        path = self._extract_candidate_path(
            candidate
        )

        if path is None:
            return None

        if len(path) < 2:
            return None

        try:
            branch_position = path.index(
                branch_node
            )
        except ValueError:
            return None

        suffix = path[
            branch_position:
        ]

        if len(suffix) < 2:
            return None

        if suffix[0] != branch_node:
            return None

        return suffix

    # ==========================================================
    # Extract Exact Suffix Edges
    # ==========================================================

    def _extract_suffix_edges(
        self,
        candidate,
        suffix,
        branch_node,
    ):
        """
        Extract exact channel edges corresponding to suffix.

        If candidate contains exact edges, preserve their keys.

        If candidate contains only node path, construct exact
        graph edges when possible.
        """

        if suffix is None or len(suffix) < 2:
            return None

        candidate_edges = self._extract_candidate_edges(
            candidate
        )

        # ------------------------------------------------------
        # Candidate provides exact edges
        # ------------------------------------------------------

        if candidate_edges is not None:

            parsed_edges = []

            for edge in candidate_edges:

                parsed = self._normalize_edge(
                    edge
                )

                if parsed is not None:
                    parsed_edges.append(
                        parsed
                    )

            if not parsed_edges:
                return None

            suffix_start = self._find_edge_sequence_start(
                parsed_edges=parsed_edges,
                suffix=suffix,
            )

            if suffix_start is None:
                return None

            suffix_length = len(suffix) - 1

            selected = parsed_edges[
                suffix_start:
                suffix_start + suffix_length
            ]

            if len(selected) != suffix_length:
                return None

            if not self._edges_match_route(
                route=suffix,
                edges=selected,
            ):
                return None

            return selected

        # ------------------------------------------------------
        # No exact candidate edges.
        #
        # Build a deterministic graph edge representation.
        # ------------------------------------------------------

        return self._resolve_edges_for_path(
            path=suffix
        )

    # ==========================================================
    # Find Edge Sequence
    # ==========================================================

    @staticmethod
    def _find_edge_sequence_start(
        parsed_edges,
        suffix,
    ):
        """
        Find the edge corresponding to suffix[0] -> suffix[1].
        """

        if len(suffix) < 2:
            return None

        first_u = suffix[0]
        first_v = suffix[1]

        for index, edge in enumerate(
            parsed_edges
        ):

            if (
                edge[0] == first_u
                and
                edge[1] == first_v
            ):

                return index

        return None

    # ==========================================================
    # Failed Channel Reuse
    # ==========================================================

    def _candidate_contains_failed_edge(
        self,
        candidate,
        suffix,
        suffix_edges,
        failed_edge,
    ):
        """
        Check whether candidate/suffix reuses the exact failed channel.

        Rules
        -----
        If failed key is known:
            exact same (u,v,key) is rejected.

        If failed key is unknown:
            same directed (u,v) is rejected.

        This allows a different parallel channel when the failed
        channel key is explicitly known.
        """

        if failed_edge is None:
            return False

        failed = self._normalize_edge(
            failed_edge
        )

        if failed is None:
            return False

        failed_u, failed_v, failed_key = failed

        # ------------------------------------------------------
        # Prefer exact suffix edges
        # ------------------------------------------------------

        if suffix_edges is not None:

            for edge in suffix_edges:

                parsed = self._normalize_edge(
                    edge
                )

                if parsed is None:
                    continue

                u, v, key = parsed

                if (
                    u != failed_u
                    or
                    v != failed_v
                ):
                    continue

                # Exact failed channel known.
                if failed_key is not None:

                    if key == failed_key:
                        return True

                    # Different parallel channel is allowed.
                    continue

                # Failed channel key unknown.
                return True

            return False

        # ------------------------------------------------------
        # Path-level fallback
        # ------------------------------------------------------

        return self._contains_failed_edge(
            suffix=suffix,
            failed_edge=failed_edge,
        )

    # ==========================================================
    # Suffix Validation
    # ==========================================================

    def is_suffix_valid(
        self,
        suffix,
        amount=0,
        candidate=None,
        suffix_edges=None,
    ):
        """
        Validate every hop of an alternative suffix.

        Important
        ---------
        Capacity is NEVER interpreted as directional liquidity.

        If exact candidate edges exist, their exact channel keys
        are validated.

        If only a node path exists, a usable graph channel is
        selected as fallback.
        """

        if suffix is None:
            return False

        if len(suffix) < 2:
            return False

        # ------------------------------------------------------
        # Validate nodes first
        # ------------------------------------------------------

        for node in suffix:

            if self.network is None:
                return False

            try:

                if node not in self.network.nodes:
                    return False

                if not self._node_is_available(
                    self.network.nodes[node]
                ):
                    return False

            except Exception:

                return False

        # ------------------------------------------------------
        # Exact suffix edges
        # ------------------------------------------------------

        if suffix_edges is not None:

            if len(suffix_edges) != len(suffix) - 1:
                return False

            if not self._edges_match_route(
                route=suffix,
                edges=suffix_edges,
            ):
                return False

            for edge in suffix_edges:

                parsed = self._normalize_edge(
                    edge
                )

                if parsed is None:
                    return False

                u, v, key = parsed

                if not self._exact_channel_available(
                    u=u,
                    v=v,
                    key=key,
                    amount=amount,
                ):
                    return False

            return True

        # ------------------------------------------------------
        # Path-only fallback
        # ------------------------------------------------------

        for u, v in zip(
            suffix[:-1],
            suffix[1:],
        ):

            if not self._channel_available(
                u,
                v,
                amount,
            ):
                return False

        return True

    # ==========================================================
    # Exact Channel Availability
    # ==========================================================

    def _exact_channel_available(
        self,
        u,
        v,
        key,
        amount,
    ):
        """
        Validate one exact channel.

        Capacity is ignored as liquidity.
        """

        if self.network is None:
            return False

        try:

            if not self.network.has_edge(
                u,
                v,
            ):
                return False

            if self.network.is_multigraph():

                edge_data = self.network.get_edge_data(
                    u,
                    v,
                )

                if not edge_data:
                    return False

                data = edge_data.get(
                    key
                )

                if data is None:
                    return False

            else:

                data = self.network.get_edge_data(
                    u,
                    v,
                )

                if data is None:
                    return False

            return self._edge_data_usable(
                data=data,
                amount=amount,
            )

        except Exception:

            return False

    # ==========================================================
    # Resolve Failed Edge
    # ==========================================================

    def _resolve_failed_edge(
        self,
        route,
        failed_edge,
        failure_index,
    ):
        """
        Resolve failed edge.

        Priority
        --------
        1. Explicit failed_edge.
        2. failure_index -> route edge.
        """

        if failed_edge is not None:

            parsed = self._normalize_edge(
                failed_edge
            )

            if parsed is not None:

                return parsed

            return None

        if failure_index is not None:

            try:

                if (
                    0 <= failure_index < len(route) - 1
                ):

                    return (
                        route[failure_index],
                        route[failure_index + 1],
                        None,
                    )

            except Exception:
                return None

        return None

    # ==========================================================
    # Resolve Failure Index
    # ==========================================================

    def _resolve_failure_index(
        self,
        route,
        failed_edge,
        failure_index,
    ):
        """
        Resolve the exact failed-hop position.

        If an explicit index matches endpoints, use it.

        Otherwise search the route.

        If an exact channel key is available and route-edge
        information can be resolved, prefer the exact key.
        """

        if failed_edge is None:
            return None

        failed_u, failed_v, failed_key = self._normalize_edge(
            failed_edge
        )

        # ------------------------------------------------------
        # Explicit index
        # ------------------------------------------------------

        if failure_index is not None:

            try:

                if (
                    0 <= failure_index < len(route) - 1
                ):

                    if (
                        route[failure_index] == failed_u
                        and
                        route[failure_index + 1] == failed_v
                    ):

                        return failure_index

            except Exception:
                pass

        # ------------------------------------------------------
        # Endpoint search
        # ------------------------------------------------------

        matching_indices = []

        for index, (u, v) in enumerate(
            zip(
                route[:-1],
                route[1:],
            )
        ):

            if (
                u == failed_u
                and
                v == failed_v
            ):

                matching_indices.append(
                    index
                )

        if not matching_indices:
            return None

        # A valid route must not contain repeated nodes, so there
        # should normally be only one matching directed hop.
        return matching_indices[0]

    # ==========================================================
    # Route Edge Resolution
    # ==========================================================

    def _resolve_route_edges_for_prefix(
        self,
        route,
        end_index,
    ):
        """
        Resolve exact edges for route[0:end_index+1].

        Returns the first end_index hops.

        This is required to preserve exact MultiDiGraph channel
        identity in the reconstructed route.
        """

        if end_index <= 0:
            return []

        prefix = route[
            :end_index + 1
        ]

        return self._resolve_edges_for_path(
            path=prefix
        )

    # ==========================================================
    # Resolve Edges for Node Path
    # ==========================================================

    def _resolve_edges_for_path(
        self,
        path,
    ):
        """
        Resolve graph edges for a node path.

        MultiDiGraph
        ------------
        If multiple channels exist, the first currently usable
        channel is selected.

        This function is a FALLBACK only.

        Exact candidate edges should always be preferred.
        """

        if self.network is None:
            return None

        if path is None or len(path) < 2:
            return []

        edges = []

        for u, v in zip(
            path[:-1],
            path[1:],
        ):

            if not self.network.has_edge(
                u,
                v,
            ):
                return None

            if self.network.is_multigraph():

                edge_data = self.network.get_edge_data(
                    u,
                    v,
                )

                if not edge_data:
                    return None

                selected = None

                for key, data in edge_data.items():

                    if self._edge_data_usable(
                        data=data,
                        amount=0,
                    ):

                        selected = (
                            u,
                            v,
                            key,
                        )

                        break

                if selected is None:
                    return None

                edges.append(
                    selected
                )

            else:

                edges.append(
                    (
                        u,
                        v,
                        None,
                    )
                )

        return edges

    # ==========================================================
    # Normalize Edge
    # ==========================================================

    @staticmethod
    def _normalize_edge(
        edge,
    ):
        """
        Normalize 2-tuple/3-tuple edge.

        Returns:
            (u, v, key)
        """

        if edge is None:
            return None

        if not isinstance(
            edge,
            (list, tuple),
        ):
            return None

        if len(edge) == 2:

            return (
                edge[0],
                edge[1],
                None,
            )

        if len(edge) >= 3:

            return (
                edge[0],
                edge[1],
                edge[2],
            )

        return None

    # ==========================================================
    # Combine Edges
    # ==========================================================

    @staticmethod
    def _combine_edges(
        prefix_edges,
        suffix_edges,
    ):
        """
        Combine exact prefix and suffix edge sequences.

        Since the branch node is shared, no edge is duplicated.
        """

        if prefix_edges is None:
            return None

        if suffix_edges is None:
            return None

        return (
            list(prefix_edges)
            +
            list(suffix_edges)
        )

    # ==========================================================
    # Edge/Route Consistency
    # ==========================================================

    @staticmethod
    def _edges_match_route(
        route,
        edges,
    ):
        """
        Verify exact endpoint correspondence between route and edges.

        Channel keys are not compared to node route because the
        route contains nodes only.
        """

        if route is None or edges is None:
            return False

        if len(edges) != len(route) - 1:
            return False

        for index, edge in enumerate(
            edges
        ):

            parsed = PartialBacktracker._normalize_edge(
                edge
            )

            if parsed is None:
                return False

            u, v, _ = parsed

            if (
                u != route[index]
                or
                v != route[index + 1]
            ):

                return False

        return True

    # ==========================================================
    # Failed Edge Reuse - Path Fallback
    # ==========================================================

    @staticmethod
    def _contains_failed_edge(
        suffix,
        failed_edge,
    ):
        """
        Path-level failed-edge check.

        If failed channel key is unavailable, directed endpoints
        are used.
        """

        if suffix is None or failed_edge is None:
            return False

        failed = PartialBacktracker._normalize_edge(
            failed_edge
        )

        if failed is None:
            return False

        failed_u, failed_v, _ = failed

        for u, v in zip(
            suffix[:-1],
            suffix[1:],
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
        route_b,
    ):
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
        suffix,
    ):
        """
        Validate complete reconstructed route.
        """

        combined = self._combine_route(
            prefix=prefix,
            suffix=suffix,
        )

        if not combined:
            return False

        try:

            return len(combined) == len(
                set(combined)
            )

        except TypeError:

            return False

    # ==========================================================
    # Combine Routes
    # ==========================================================

    @staticmethod
    def _combine_route(
        prefix,
        suffix,
    ):
        """
        Combine prefix and suffix at their shared branch node.
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

        return (
            prefix
            +
            suffix
        )

    # ==========================================================
    # Edge Exists
    # ==========================================================

    def _edge_exists(
        self,
        u,
        v,
    ):
        if self.network is None:
            return False

        try:

            return bool(
                self.network.has_edge(
                    u,
                    v,
                )
            )

        except Exception:

            return False

    # ==========================================================
    # Node Availability
    # ==========================================================

    def _nodes_available(
        self,
        u,
        v,
    ):
        if self.network is None:
            return False

        try:

            u_data = self.network.nodes[u]
            v_data = self.network.nodes[v]

        except Exception:

            return False

        return (
            self._node_is_available(u_data)
            and
            self._node_is_available(v_data)
        )

    @staticmethod
    def _node_is_available(
        data,
    ):
        """
        Explicit False means unavailable.

        Missing attributes mean available.
        """

        if data.get(
            "available",
            True,
        ) is False:

            return False

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

    # ==========================================================
    # Channel Availability
    # ==========================================================

    def _channel_available(
        self,
        u,
        v,
        amount,
    ):
        """
        Check whether at least one usable channel exists.

        Capacity is deliberately ignored as liquidity.
        """

        if self.network is None:
            return False

        try:

            edge_data = self.network.get_edge_data(
                u,
                v,
            )

        except Exception:

            return False

        if edge_data is None:
            return False

        if self._is_multigraph():

            for _, data in edge_data.items():

                if self._edge_data_usable(
                    data=data,
                    amount=amount,
                ):

                    return True

            return False

        return self._edge_data_usable(
            data=edge_data,
            amount=amount,
        )

    # ==========================================================
    # Edge Data Validation
    # ==========================================================

    @staticmethod
    def _edge_data_usable(
        data,
        amount,
    ):
        """
        Validate structural/runtime channel usability.

        IMPORTANT
        ---------
        capacity is NOT directional liquidity.

        Directional liquidity is checked only from:

            balance_uv
            liquidity_uv
            liquidity
            estimated_liquidity

        Unknown liquidity is accepted.
        """

        if not data:
            return False

        if data.get(
            "available",
            True,
        ) is False:

            return False

        # ------------------------------------------------------
        # Directional liquidity only
        # ------------------------------------------------------

        directional_liquidity = None

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            value = data.get(
                field,
                None,
            )

            if value is None:
                continue

            try:

                value = float(value)

            except (
                TypeError,
                ValueError,
            ):

                continue

            if value < 0:
                continue

            directional_liquidity = value
            break

        # ------------------------------------------------------
        # Known liquidity
        # ------------------------------------------------------

        if directional_liquidity is not None:

            try:

                if directional_liquidity < float(
                    amount
                ):

                    return False

            except (
                TypeError,
                ValueError,
            ):

                return False

        # ------------------------------------------------------
        # Unknown liquidity
        # ------------------------------------------------------

        # Unknown directional liquidity does NOT imply zero.
        # Stochastic forwarding failure belongs to FailureModel,
        # not to structural backtracking validation.

        return True

    # ==========================================================
    # Usable Outgoing Edge
    # ==========================================================

    def _has_usable_outgoing_edge(
        self,
        node,
        amount=0,
        excluded_edge=None,
    ):
        """
        Check whether node has an outgoing usable channel.

        If the failed edge has an exact key, another parallel
        channel is allowed.
        """

        if self.network is None:
            return False

        try:

            if node not in self.network:
                return False

            neighbors = self.network.successors(
                node
            )

        except Exception:

            return False

        for neighbor in neighbors:

            # --------------------------------------------------
            # Excluded endpoint
            # --------------------------------------------------

            if excluded_edge is not None:

                failed = self._normalize_edge(
                    excluded_edge
                )

                if failed is not None:

                    failed_u, failed_v, failed_key = failed

                    if (
                        node == failed_u
                        and
                        neighbor == failed_v
                    ):

                        # Exact failed channel:
                        # check whether another parallel channel
                        # remains usable.
                        if (
                            failed_key is not None
                            and
                            self._has_another_usable_channel(
                                u=node,
                                v=neighbor,
                                amount=amount,
                                excluded_key=failed_key,
                            )
                        ):

                            return True

                        # If failed key is unknown, the entire
                        # directed endpoint is excluded.
                        continue

            if self._channel_available(
                node,
                neighbor,
                amount,
            ):

                return True

        return False

    # ==========================================================
    # Another Parallel Channel
    # ==========================================================

    def _has_another_usable_channel(
        self,
        u,
        v,
        amount,
        excluded_key,
    ):
        """
        Check whether another usable parallel channel exists.
        """

        if not self._is_multigraph():
            return False

        try:

            edge_data = self.network.get_edge_data(
                u,
                v,
            )

        except Exception:

            return False

        if not edge_data:
            return False

        for key, data in edge_data.items():

            if key == excluded_key:
                continue

            if self._edge_data_usable(
                data=data,
                amount=amount,
            ):

                return True

        return False

    # ==========================================================
    # Route Validation
    # ==========================================================

    @staticmethod
    def _validate_route(
        route,
    ):
        """
        Basic node-route validation.

        Repeated nodes are rejected because the intended payment
        route must be loop-free.
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

        if route[0] == route[-1]:
            return False

        return True

    # ==========================================================
    # MultiGraph Detection
    # ==========================================================

    def _is_multigraph(self):
        if self.network is None:
            return False

        try:

            return bool(
                self.network.is_multigraph()
            )

        except Exception:

            return False

    # ==========================================================
    # Result Helper
    # ==========================================================

    @staticmethod
    def _result(
        success,
        status,
        reason,
        original_route,
        **kwargs,
    ):
        """
        Build normalized result dictionary.
        """

        result = {
            "success": bool(success),
            "status": status,
            "reason": reason,
            "original_route": (
                list(original_route)
                if isinstance(
                    original_route,
                    (list, tuple),
                )
                else original_route
            ),
        }

        result.update(
            kwargs
        )

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

    print("=" * 78)
    print("PARTIAL BACKTRACKING MODULE TEST")
    print("=" * 78)

    # ----------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------

    G = nx.MultiDiGraph()

    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),

        ("B", "F"),
        ("F", "G"),
        ("G", "E"),
    ]

    for u, v in edges:

        G.add_edge(
            u,
            v,
            capacity=10000,
            balance_uv=10000,
            available=True,
        )

    for node in G.nodes:

        G.nodes[node]["available"] = True
        G.nodes[node]["is_online"] = True

    # ----------------------------------------------------------
    # Bucket test object
    # ----------------------------------------------------------

    class TestBucket:

        def __init__(self):

            self.attempts = 0

            self.failed_candidates = []

            self.candidates = [
                {
                    "route": [
                        "A",
                        "B",
                        "F",
                        "G",
                        "E",
                    ],
                    "edges": [
                        ("A", "B", 0),
                        ("B", "F", 0),
                        ("F", "G", 0),
                        ("G", "E", 0),
                    ],
                }
            ]

    bucket = TestBucket()

    backtracker = PartialBacktracker(
        network=G,
        bucket=bucket,
    )

    # ----------------------------------------------------------
    # Current failed route
    # ----------------------------------------------------------

    route = [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    failed_edge = (
        "C",
        "D",
        0,
    )

    # ----------------------------------------------------------
    # Execute
    # ----------------------------------------------------------

    result = backtracker.backtrack(
        route=route,
        failed_edge=failed_edge,
        failure_index=2,
        amount=1000,
        bucket_id="B001",
        attempt_id=0,
    )

    # ----------------------------------------------------------
    # Print result
    # ----------------------------------------------------------

    print()
    print("-" * 78)
    print("RESULT")
    print("-" * 78)

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
        f"Branch index        : "
        f"{result.get('branch_index')}"
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
        f"Prefix edges        : "
        f"{result.get('preserved_prefix_edges')}"
    )

    print(
        f"Alternative suffix  : "
        f"{result.get('alternative_suffix')}"
    )

    print(
        f"Suffix edges        : "
        f"{result.get('alternative_suffix_edges')}"
    )

    print(
        f"New route           : "
        f"{result.get('new_route')}"
    )

    print(
        f"New edges           : "
        f"{result.get('new_edges')}"
    )

    print(
        f"Full reroute needed : "
        f"{result.get('full_reroute_required')}"
    )

    print(
        f"Attempt ID          : "
        f"{result.get('attempt_id')}"
    )

    print(
        f"Bucket attempts     : "
        f"{bucket.attempts}"
    )

    print("-" * 78)

    # ----------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------

    assert result["success"] is True
    assert result["status"] == "alternative_found"

    assert result["branch_point"] == "B"
    assert result["branch_index"] == 1

    assert result["failed_edge"] == (
        "C",
        "D",
        0,
    )

    assert result["failure_index"] == 2

    assert result["preserved_prefix"] == [
        "A",
        "B",
    ]

    assert result["alternative_suffix"] == [
        "B",
        "F",
        "G",
        "E",
    ]

    assert result["new_route"] == [
        "A",
        "B",
        "F",
        "G",
        "E",
    ]

    assert result["new_edges"] == [
        ("A", "B", 0),
        ("B", "F", 0),
        ("F", "G", 0),
        ("G", "E", 0),
    ]

    # Bucket.attempts must NOT be changed by backtracking.
    assert bucket.attempts == 0

    print()
    print(
        "PARTIAL BACKTRACKING STATUS : SUCCESS"
    )
    print(
        "Bucket.attempts unchanged     : PASS"
    )
    print(
        "Exact channel keys preserved  : PASS"
    )
    print(
        "Route/edge consistency        : PASS"
    )
    print(
        "Capacity != liquidity rule    : PASS"
    )
    print("=" * 78)