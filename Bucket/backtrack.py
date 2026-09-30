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

Responsibilities
----------------
This module performs PARTIAL ROUTE recovery after a payment
failure.

It:

    1. identifies the failed hop
    2. finds the nearest usable branch point
    3. reads candidates already stored in Bucket
    4. rejects candidates already known to have failed
    5. rejects reuse of the failed channel
    6. preserves exact MultiDiGraph channel keys when available
    7. validates the alternative suffix
    8. prevents loops
    9. reconstructs a complete route
   10. reports when full rerouting is required

This module does NOT:

    - execute payments
    - call FailureModel
    - increment Bucket.attempts
    - increment payment-attempt counters
    - select PPO actions
    - perform PPO inference
    - generate new Top-K routes
    - perform full rerouting

Counter semantics
-----------------
Bucket.attempts
    Number of actual payment attempts.

Backtracker.attempts
    Number of candidate-level backtracking operations.

PartialBacktracker
    Does not modify either counter.

Channel semantics
-----------------
Channel capacity is NOT directional liquidity.

Known directional liquidity is accepted only from:

    balance_uv
    liquidity_uv
    liquidity
    estimated_liquidity

Unknown directional liquidity is accepted by structural
validation. Actual stochastic payment failure remains the
responsibility of FailureModel.

MultiDiGraph semantics
----------------------
Whenever candidate["edges"] contains:

    (u, v, key)

the exact channel key is preserved and validated.

The returned result contains:

    new_route
    new_edges

so downstream payment simulation does not have to reconstruct
parallel-channel identity from node pairs.
"""

from typing import Any


class PartialBacktracker:
    """
    Partial route recovery using alternatives already stored
    in a Bucket.

    The class is intentionally separated from candidate-level
    Bucket/backtrack.py.

    Bucket/backtrack.py
        -> moves between complete candidate routes.

    PartialBacktracker
        -> preserves the successful prefix of a failed route
           and replaces only its suffix.
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
        Recover a failed route by replacing its suffix.

        Parameters
        ----------
        route : list
            Failed node route.

        failed_edge : tuple, optional
            Failed channel:

                (u, v)

            or:

                (u, v, key)

        failure_index : int, optional
            Zero-based index of the failed hop.

        amount : float
            Payment amount.

        bucket_id : optional
            Bucket identifier.

        attempt_id : int
            Current payment-attempt identifier.

        Returns
        -------
        dict
            Structured recovery result.

        Important
        ---------
        attempt_id is returned unchanged.

        This method does not create or increment payment
        attempt identifiers. The caller owns payment-attempt
        accounting.
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
                attempt_id=attempt_id,
                full_reroute_required=False
            )

        route = list(route)

        # ------------------------------------------------------
        # 2. Validate amount
        # ------------------------------------------------------

        if not self._valid_amount(amount):

            return self._result(
                success=False,
                status="invalid_amount",
                reason="invalid_amount",
                original_route=route,
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False
            )

        # ------------------------------------------------------
        # 3. Resolve failed edge
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
                failure_index=failure_index,
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False
            )

        # ------------------------------------------------------
        # 4. Resolve exact failure index
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
                amount=amount,
                attempt_id=attempt_id,
                full_reroute_required=False
            )

        # ------------------------------------------------------
        # 5. Find nearest usable Bucket branch
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
                amount=amount,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                full_reroute_required=True
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
            f"  Valid alternatives: {len(candidates)}"
        )

        # ------------------------------------------------------
        # 6. Evaluate candidates in ORIGINAL Bucket order
        # ------------------------------------------------------

        for candidate_info in candidates:

            candidate_index = candidate_info["index"]
            candidate = candidate_info["candidate"]

            print()
            print(
                f"  Evaluating candidate "
                f"rank={candidate_index + 1}"
            )

            # --------------------------------------------------
            # Candidate already failed
            # --------------------------------------------------

            if self._candidate_is_failed(
                candidate,
                candidate_index=candidate_index
            ):

                print(
                    "    REJECTED: candidate previously failed"
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
                branch_node=branch_node
            )

            if suffix is None:

                print(
                    "    REJECTED: branch node not present"
                )

                continue

            # --------------------------------------------------
            # Candidate cannot equal failed route
            # --------------------------------------------------

            if self._same_route(
                candidate_path,
                route
            ):

                print(
                    "    REJECTED: candidate identical "
                    "to failed route"
                )

                continue

            # --------------------------------------------------
            # Failed channel cannot be reused
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
            # Build prefix
            # --------------------------------------------------

            prefix = route[
                :branch_index + 1
            ]

            # --------------------------------------------------
            # Loop prevention
            # --------------------------------------------------

            if not self._is_loop_free(
                prefix=prefix,
                suffix=suffix
            ):

                print(
                    "    REJECTED: loop detected"
                )

                continue

            # --------------------------------------------------
            # Exact suffix edges
            # --------------------------------------------------

            suffix_edges = self._extract_suffix_edges(
                candidate=candidate,
                suffix=suffix
            )

            # --------------------------------------------------
            # Validate suffix
            # --------------------------------------------------

            validation = self.validate_suffix(
                suffix=suffix,
                amount=amount,
                candidate=candidate,
                suffix_edges=suffix_edges
            )

            if not validation["valid"]:

                print(
                    "    REJECTED: "
                    f"{validation['reason']}"
                )

                continue

            suffix_edges = validation.get(
                "edges",
                suffix_edges
            )

            # --------------------------------------------------
            # Construct new route
            # --------------------------------------------------

            new_route = self._combine_route(
                prefix=prefix,
                suffix=suffix
            )

            if not self._validate_route(
                new_route
            ):

                print(
                    "    REJECTED: resulting route invalid"
                )

                continue

            # --------------------------------------------------
            # Construct complete exact edge sequence
            # --------------------------------------------------

            new_edges = self._combine_edges(
                prefix=prefix,
                suffix=suffix,
                suffix_edges=suffix_edges
            )

            # --------------------------------------------------
            # Final complete-route validation
            # --------------------------------------------------

            final_validation = self._validate_complete_route(
                route=new_route,
                edges=new_edges,
                amount=amount
            )

            if not final_validation["valid"]:

                print(
                    "    REJECTED: complete route validation "
                    f"failed: {final_validation['reason']}"
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

                "original_edges": None,
                "new_edges": new_edges,

                "preserved_prefix": prefix,
                "alternative_suffix": suffix,
                "alternative_suffix_edges": suffix_edges,

                "branch_point": branch_node,
                "branch_index": branch_index,

                "failed_edge": resolved_failed_edge,
                "failure_index": resolved_failure_index,

                "candidate": candidate,
                "candidate_index": candidate_index,
                "candidate_rank": candidate_index + 1,

                "amount": amount,
                "bucket_id": bucket_id,

                "attempt_id": attempt_id,

                "full_reroute_required": False
            }

        # ------------------------------------------------------
        # 7. No valid candidate
        # ------------------------------------------------------

        return {
            "success": False,
            "status": "full_reroute_required",
            "reason": "no_valid_bucket_alternative",

            "original_route": route,
            "new_route": None,

            "original_edges": None,
            "new_edges": None,

            "preserved_prefix": None,
            "alternative_suffix": None,
            "alternative_suffix_edges": None,

            "branch_point": branch_node,
            "branch_index": branch_index,

            "failed_edge": resolved_failed_edge,
            "failure_index": resolved_failure_index,

            "candidate": None,
            "candidate_index": None,
            "candidate_rank": None,

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
        Search backward from the failed-hop source.

        The first branch point having at least one valid
        Bucket candidate is selected.

        Candidate order is preserved exactly as stored in
        Bucket.candidates.
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

            branch_node = route[index]

            excluded_edge = None

            if index == failure_index:
                excluded_edge = failed_edge

            # --------------------------------------------------
            # The branch node itself must have a usable
            # outgoing channel other than the failed one.
            # --------------------------------------------------

            if not self._has_usable_outgoing_edge(
                node=branch_node,
                amount=amount,
                excluded_edge=excluded_edge
            ):
                continue

            # --------------------------------------------------
            # Read candidates
            # --------------------------------------------------

            raw_candidates = self.get_alternative_suffixes(
                route=route,
                branch_index=index,
                failed_edge=failed_edge,
                bucket_id=bucket_id
            )

            valid_candidates = []

            for candidate_index, candidate in raw_candidates:

                if self._candidate_is_failed(
                    candidate,
                    candidate_index=candidate_index
                ):
                    continue

                candidate_path = self._extract_candidate_path(
                    candidate
                )

                if candidate_path is None:
                    continue

                suffix = self._normalize_suffix(
                    candidate=candidate,
                    branch_node=branch_node
                )

                if suffix is None:
                    continue

                if len(suffix) < 2:
                    continue

                if suffix[0] != branch_node:
                    continue

                if self._same_route(
                    candidate_path,
                    route
                ):
                    continue

                if self._candidate_contains_failed_edge(
                    candidate=candidate,
                    suffix=suffix,
                    failed_edge=failed_edge
                ):
                    continue

                prefix = route[
                    :index + 1
                ]

                if not self._is_loop_free(
                    prefix=prefix,
                    suffix=suffix
                ):
                    continue

                suffix_edges = self._extract_suffix_edges(
                    candidate=candidate,
                    suffix=suffix
                )

                validation = self.validate_suffix(
                    suffix=suffix,
                    amount=amount,
                    candidate=candidate,
                    suffix_edges=suffix_edges
                )

                if not validation["valid"]:
                    continue

                valid_candidates.append(
                    {
                        "index": candidate_index,
                        "candidate": candidate
                    }
                )

            if valid_candidates:

                return {
                    "branch_index": index,
                    "branch_node": branch_node,
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

        Returns the nearest network-feasible branch index.
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
        Retrieve candidates from the actual Bucket model.

        Primary interface
        -----------------
        Bucket.candidates

        Optional backward-compatible interfaces are retained
        for older Bucket implementations.

        Returns
        -------
        list of (index, candidate)
        """

        if self.bucket is None:
            return []

        # ------------------------------------------------------
        # Primary current Bucket interface
        # ------------------------------------------------------

        candidates = getattr(
            self.bucket,
            "candidates",
            None
        )

        if candidates is not None:

            try:

                return list(
                    enumerate(candidates)
                )

            except Exception:

                return []

        # ------------------------------------------------------
        # Legacy API
        # ------------------------------------------------------

        raw_candidates = []

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
                    branch_node=route[branch_index],
                    failed_edge=failed_edge
                )

            except TypeError:

                try:

                    raw_candidates = method(
                        bucket_id
                    )

                except Exception:

                    raw_candidates = []

            except Exception:

                raw_candidates = []

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

        if not raw_candidates:

            routes = getattr(
                self.bucket,
                "routes",
                None
            )

            if routes is not None:
                raw_candidates = routes

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
            raw_candidates = list(
                raw_candidates
            )
        except Exception:
            return []

        return list(
            enumerate(raw_candidates)
        )

    # ==========================================================
    # Candidate Failure Detection
    # ==========================================================

    def _candidate_is_failed(
        self,
        candidate,
        candidate_index=None
    ):
        """
        Determine whether a candidate is already failed.

        Current Bucket semantics are used first:

            bucket.failed_candidates

        Then candidate-level state is checked:

            success=False
            failed=True
            status="failed"

        ``success=None`` means not yet attempted.

        This is important because the current Bucket does not
        require candidates themselves to contain a ``success``
        field.
        """

        if candidate is None:
            return True

        # ------------------------------------------------------
        # Bucket-level failure registry
        # ------------------------------------------------------

        if self.bucket is not None:

            failed_candidates = getattr(
                self.bucket,
                "failed_candidates",
                []
            )

            try:

                if candidate in failed_candidates:
                    return True

            except Exception:
                pass

            # --------------------------------------------------
            # Index-based protection
            # --------------------------------------------------

            if candidate_index is not None:

                for failed in failed_candidates:

                    if failed is candidate:
                        return True

        # ------------------------------------------------------
        # Candidate-level state
        # ------------------------------------------------------

        if isinstance(
            candidate,
            dict
        ):

            if candidate.get(
                "success",
                None
            ) is False:

                return True

            if candidate.get(
                "failed",
                False
            ) is True:

                return True

            if candidate.get(
                "status",
                None
            ) == "failed":

                return True

        return False

    # ==========================================================
    # Candidate Path Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_path(
        candidate
    ):
        """
        Extract node path from a candidate.
        """

        if candidate is None:
            return None

        if isinstance(
            candidate,
            dict
        ):

            path = (
                candidate.get("path")
                or candidate.get("route")
                or candidate.get("suffix")
            )

            if path is None:
                return None

            try:
                return list(path)
            except Exception:
                return None

        if isinstance(
            candidate,
            tuple
        ):

            if not candidate:
                return None

            first = candidate[0]

            if isinstance(
                first,
                (list, tuple)
            ):

                return list(first)

            return None

        if isinstance(
            candidate,
            list
        ):

            # A raw node route is accepted.
            return list(candidate)

        return None

    # ==========================================================
    # Candidate Edge Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_edges(
        candidate
    ):
        """
        Extract candidate edges.

        Preferred representation:

            [(u, v, key), ...]

        Backward-compatible representation:

            (path, edges, score)
        """

        if candidate is None:
            return None

        if isinstance(
            candidate,
            dict
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
            tuple
        ):

            if len(candidate) >= 2:

                edges = candidate[1]

                if isinstance(
                    edges,
                    (list, tuple)
                ):

                    return list(edges)

        return None

    # ==========================================================
    # Candidate Suffix Normalization
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node
    ):
        """
        Convert a complete candidate route into a suffix.

        Example:

            A -> B -> F -> G -> E

        branch:

            B

        result:

            B -> F -> G -> E
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

        if not self._validate_route(
            suffix
        ):
            return None

        return suffix

    # ==========================================================
    # Failed Channel Detection
    # ==========================================================

    def _candidate_contains_failed_edge(
        self,
        candidate,
        suffix,
        failed_edge
    ):
        """
        Determine whether candidate reuses the failed channel.

        Exact channel key comparison is used whenever both
        sides contain a key.

        If the candidate does not expose channel-key information,
        endpoint comparison is used conservatively.
        """

        if failed_edge is None:
            return False

        normalized_failed = self._normalize_edge(
            failed_edge
        )

        if normalized_failed is None:
            return False

        failed_u = normalized_failed[0]
        failed_v = normalized_failed[1]

        failed_key = (
            normalized_failed[2]
            if len(normalized_failed) == 3
            else None
        )

        candidate_edges = self._extract_candidate_edges(
            candidate
        )

        # ------------------------------------------------------
        # Exact edge information available
        # ------------------------------------------------------

        if candidate_edges is not None:

            for edge in candidate_edges:

                normalized = self._normalize_edge(
                    edge
                )

                if normalized is None:
                    continue

                if (
                    normalized[0] != failed_u
                    or
                    normalized[1] != failed_v
                ):
                    continue

                candidate_key = (
                    normalized[2]
                    if len(normalized) == 3
                    else None
                )

                # Both exact channel identities known.
                if (
                    failed_key is not None
                    and
                    candidate_key is not None
                ):

                    if candidate_key == failed_key:
                        return True

                    # Same endpoints, different channel.
                    continue

                # At least one key is unknown.
                # Endpoint reuse is treated conservatively.
                return True

            return False

        # ------------------------------------------------------
        # Path-only fallback
        # ------------------------------------------------------

        return self._contains_failed_edge(
            suffix=suffix,
            failed_edge=failed_edge
        )

    # ==========================================================
    # Suffix Edge Extraction
    # ==========================================================

    def _extract_suffix_edges(
        self,
        candidate,
        suffix
    ):
        """
        Extract the exact edge sequence corresponding to suffix.

        Returns
        -------
        list or None

        If exact candidate edges are unavailable, returns None.
        """

        candidate_edges = self._extract_candidate_edges(
            candidate
        )

        if candidate_edges is None:
            return None

        if len(suffix) < 2:
            return None

        normalized_edges = []

        for edge in candidate_edges:

            normalized = self._normalize_edge(
                edge
            )

            if normalized is None:
                continue

            normalized_edges.append(
                normalized
            )

        if not normalized_edges:
            return None

        # ------------------------------------------------------
        # Candidate edge list must correspond to candidate path.
        #
        # We locate the exact suffix edge sequence by matching
        # directed endpoints in order.
        # ------------------------------------------------------

        result = []

        edge_position = 0

        for u, v in zip(
            suffix[:-1],
            suffix[1:]
        ):

            found = False

            while edge_position < len(
                normalized_edges
            ):

                edge = normalized_edges[
                    edge_position
                ]

                edge_position += 1

                if (
                    edge[0] == u
                    and
                    edge[1] == v
                ):

                    result.append(
                        edge
                    )

                    found = True
                    break

            if not found:
                return None

        if len(result) != len(suffix) - 1:
            return None

        return result

    # ==========================================================
    # Suffix Validation
    # ==========================================================

    def validate_suffix(
        self,
        suffix,
        amount=0,
        candidate=None,
        suffix_edges=None
    ):
        """
        Validate a complete candidate suffix.

        Returns
        -------
        dict
            {
                "valid": bool,
                "reason": ...,
                "failed_edge": ...,
                "edges": [...]
            }

        Structural validation only.

        It does NOT execute stochastic failure logic.
        """

        if self.network is None:

            return {
                "valid": False,
                "reason": "network_unavailable",
                "failed_edge": None,
                "edges": None
            }

        if not self._validate_route(
            suffix
        ):

            return {
                "valid": False,
                "reason": "invalid_suffix",
                "failed_edge": None,
                "edges": None
            }

        if not self._valid_amount(
            amount
        ):

            return {
                "valid": False,
                "reason": "invalid_amount",
                "failed_edge": None,
                "edges": None
            }

        # ------------------------------------------------------
        # Prefer exact candidate edge sequence.
        # ------------------------------------------------------

        if suffix_edges is None:

            suffix_edges = self._extract_suffix_edges(
                candidate=candidate,
                suffix=suffix
            )

        # ------------------------------------------------------
        # Exact channel validation
        # ------------------------------------------------------

        if suffix_edges is not None:

            if len(suffix_edges) != len(suffix) - 1:

                return {
                    "valid": False,
                    "reason": "edge_count_mismatch",
                    "failed_edge": None,
                    "edges": None
                }

            for index, edge in enumerate(
                suffix_edges
            ):

                normalized = self._normalize_edge(
                    edge
                )

                if normalized is None:

                    return {
                        "valid": False,
                        "reason": "invalid_edge",
                        "failed_edge": edge,
                        "edges": None
                    }

                u = suffix[index]
                v = suffix[index + 1]

                if (
                    normalized[0] != u
                    or
                    normalized[1] != v
                ):

                    return {
                        "valid": False,
                        "reason": "edge_route_mismatch",
                        "failed_edge": normalized,
                        "edges": None
                    }

                if not self._nodes_available(
                    u,
                    v
                ):

                    return {
                        "valid": False,
                        "reason": "node_unavailable",
                        "failed_edge": normalized,
                        "edges": None
                    }

                if not self._exact_channel_available(
                    u=u,
                    v=v,
                    key=(
                        normalized[2]
                        if len(normalized) == 3
                        else None
                    ),
                    amount=amount
                ):

                    return {
                        "valid": False,
                        "reason": "channel_unavailable_or_insufficient",
                        "failed_edge": normalized,
                        "edges": None
                    }

            return {
                "valid": True,
                "reason": None,
                "failed_edge": None,
                "edges": suffix_edges
            }

        # ------------------------------------------------------
        # No exact channel information.
        #
        # Validate at least one usable channel for every hop.
        # ------------------------------------------------------

        resolved_edges = []

        for u, v in zip(
            suffix[:-1],
            suffix[1:]
        ):

            if not self._nodes_available(
                u,
                v
            ):

                return {
                    "valid": False,
                    "reason": "node_unavailable",
                    "failed_edge": (u, v),
                    "edges": None
                }

            resolved = self._resolve_usable_edge(
                u=u,
                v=v,
                amount=amount
            )

            if resolved is None:

                return {
                    "valid": False,
                    "reason": "no_usable_channel",
                    "failed_edge": (u, v),
                    "edges": None
                }

            resolved_edges.append(
                resolved
            )

        return {
            "valid": True,
            "reason": None,
            "failed_edge": None,
            "edges": resolved_edges
        }

    # Backward-compatible name.
    def is_suffix_valid(
        self,
        suffix,
        amount=0,
        candidate=None
    ):
        """
        Backward-compatible boolean suffix validation API.
        """

        return bool(
            self.validate_suffix(
                suffix=suffix,
                amount=amount,
                candidate=candidate
            )["valid"]
        )

    # ==========================================================
    # Complete Route Validation
    # ==========================================================

    def _validate_complete_route(
        self,
        route,
        edges,
        amount
    ):
        """
        Validate the reconstructed complete route.

        The prefix channels are validated only when exact edge
        information is available.

        If new_edges contains only the alternative suffix, the
        prefix is still structurally checked by node pairs.
        """

        if not self._validate_route(
            route
        ):

            return {
                "valid": False,
                "reason": "invalid_route"
            }

        if edges is None:

            return {
                "valid": False,
                "reason": "missing_edges"
            }

        if len(edges) != len(route) - 1:

            return {
                "valid": False,
                "reason": "edge_count_mismatch"
            }

        for index, edge in enumerate(
            edges
        ):

            normalized = self._normalize_edge(
                edge
            )

            if normalized is None:

                return {
                    "valid": False,
                    "reason": "invalid_edge"
                }

            u = route[index]
            v = route[index + 1]

            if (
                normalized[0] != u
                or
                normalized[1] != v
            ):

                return {
                    "valid": False,
                    "reason": "edge_route_mismatch"
                }

            if not self._nodes_available(
                u,
                v
            ):

                return {
                    "valid": False,
                    "reason": "node_unavailable"
                }

            if len(normalized) == 3:

                if not self._exact_channel_available(
                    u=u,
                    v=v,
                    key=normalized[2],
                    amount=amount
                ):

                    return {
                        "valid": False,
                        "reason": "channel_unavailable_or_insufficient"
                    }

            else:

                if not self._channel_available(
                    u=u,
                    v=v,
                    amount=amount
                ):

                    return {
                        "valid": False,
                        "reason": "channel_unavailable_or_insufficient"
                    }

        return {
            "valid": True,
            "reason": None
        }

    # ==========================================================
    # Exact Channel Availability
    # ==========================================================

    def _exact_channel_available(
        self,
        u,
        v,
        key,
        amount
    ):
        """
        Validate one exact channel.

        If key is None, at least one usable channel is accepted.
        """

        if self.network is None:
            return False

        try:

            if not self.network.has_edge(
                u,
                v
            ):
                return False

            if not self._is_multigraph():

                data = self.network.get_edge_data(
                    u,
                    v
                )

                return self._edge_data_usable(
                    data,
                    amount
                )

            edge_data = self.network.get_edge_data(
                u,
                v
            )

            if not edge_data:
                return False

            if key is None:

                for data in edge_data.values():

                    if self._edge_data_usable(
                        data,
                        amount
                    ):
                        return True

                return False

            data = edge_data.get(
                key
            )

            if data is None:
                return False

            return self._edge_data_usable(
                data,
                amount
            )

        except Exception:

            return False

    # ==========================================================
    # Resolve One Usable Edge
    # ==========================================================

    def _resolve_usable_edge(
        self,
        u,
        v,
        amount
    ):
        """
        Return one usable exact edge identifier.

        MultiDiGraph:
            (u, v, key)

        Graph/DiGraph:
            (u, v)
        """

        if self.network is None:
            return None

        try:

            edge_data = self.network.get_edge_data(
                u,
                v
            )

        except Exception:

            return None

        if edge_data is None:
            return None

        if self._is_multigraph():

            for key, data in edge_data.items():

                if self._edge_data_usable(
                    data,
                    amount
                ):

                    return (
                        u,
                        v,
                        key
                    )

            return None

        if self._edge_data_usable(
            edge_data,
            amount
        ):

            return (
                u,
                v
            )

        return None

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
        Resolve failed edge.

        Exact key is preserved if supplied.
        """

        if failed_edge is not None:

            normalized = self._normalize_edge(
                failed_edge
            )

            if normalized is None:
                return None

            return normalized

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
        Resolve the failed hop index.

        Endpoint matching is used because a node route itself
        does not carry channel keys.
        """

        if failed_edge is None:
            return None

        failed_u = failed_edge[0]
        failed_v = failed_edge[1]

        # ------------------------------------------------------
        # Explicit index
        # ------------------------------------------------------

        if failure_index is not None:

            if (
                0 <= failure_index < len(route) - 1
                and
                route[failure_index] == failed_u
                and
                route[failure_index + 1] == failed_v
            ):

                return failure_index

        # ------------------------------------------------------
        # Search route
        # ------------------------------------------------------

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
        Endpoint-level fallback for failed-channel detection.
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
    # Edge Normalization
    # ==========================================================

    @staticmethod
    def _normalize_edge(
        edge
    ):
        """
        Normalize:

            (u, v)

        or:

            (u, v, key)
        """

        if edge is None:
            return None

        if not isinstance(
            edge,
            (list, tuple)
        ):
            return None

        if len(edge) >= 3:

            return (
                edge[0],
                edge[1],
                edge[2]
            )

        if len(edge) == 2:

            return (
                edge[0],
                edge[1]
            )

        return None

    # ==========================================================
    # Same Route
    # ==========================================================

    @staticmethod
    def _same_route(
        route_a,
        route_b
    ):
        try:
            return (
                list(route_a)
                ==
                list(route_b)
            )
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
        Ensure reconstructed route contains no repeated node.
        """

        combined = self._combine_route(
            prefix,
            suffix
        )

        try:

            return (
                len(combined)
                ==
                len(set(combined))
            )

        except TypeError:

            return False

    # ==========================================================
    # Route Combination
    # ==========================================================

    @staticmethod
    def _combine_route(
        prefix,
        suffix
    ):
        """
        Combine prefix and suffix without duplicating
        their shared branch node.
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
    # Edge Combination
    # ==========================================================

    def _combine_edges(
        self,
        prefix,
        suffix,
        suffix_edges
    ):
        """
        Construct the complete exact edge sequence.

        The prefix itself is resolved against the current
        network. The suffix uses the candidate's exact edges
        whenever available.
        """

        if len(prefix) < 1:
            return list(
                suffix_edges or []
            )

        prefix_edges = []

        for u, v in zip(
            prefix[:-1],
            prefix[1:]
        ):

            resolved = self._resolve_usable_edge(
                u=u,
                v=v,
                amount=0
            )

            if resolved is None:
                return None

            prefix_edges.append(
                resolved
            )

        return (
            prefix_edges
            +
            list(
                suffix_edges or []
            )
        )

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

            return bool(
                self.network.has_edge(
                    u,
                    v
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
        v
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
        data
    ):
        """
        Missing availability attributes mean available.

        Explicit False means unavailable.
        """

        if data.get(
            "available",
            True
        ) is False:

            return False

        if data.get(
            "is_online",
            True
        ) is False:

            return False

        if data.get(
            "online",
            True
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
        amount
    ):
        """
        Return True if at least one usable channel exists.

        Capacity is deliberately ignored as directional
        liquidity.
        """

        if self.network is None:
            return False

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

            for data in edge_data.values():

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
    # Edge Data Usability
    # ==========================================================

    @staticmethod
    def _edge_data_usable(
        data,
        amount
    ):
        """
        Structural channel validation.

        Rules
        -----
        1. unavailable channel -> False
        2. known directional liquidity < amount -> False
        3. known directional liquidity >= amount -> True
        4. unknown directional liquidity -> True
        5. capacity is NEVER used as liquidity
        """

        if not data:
            return False

        if data.get(
            "available",
            True
        ) is False:

            return False

        # ------------------------------------------------------
        # Amount
        # ------------------------------------------------------

        try:

            amount = float(
                amount
            )

        except (
            TypeError,
            ValueError
        ):

            return False

        if amount < 0:
            return False

        # ------------------------------------------------------
        # Directional liquidity
        # ------------------------------------------------------

        liquidity = None

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity"
        ):

            value = data.get(
                field,
                None
            )

            if value is None:
                continue

            try:

                value = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                continue

            if value < 0:
                value = 0.0

            liquidity = value
            break

        # ------------------------------------------------------
        # Unknown liquidity
        # ------------------------------------------------------

        if liquidity is None:
            return True

        # ------------------------------------------------------
        # Known liquidity
        # ------------------------------------------------------

        return liquidity >= amount

    # ==========================================================
    # Outgoing Edge
    # ==========================================================

    def _has_usable_outgoing_edge(
        self,
        node,
        amount=0,
        excluded_edge=None
    ):
        """
        Check whether a node has another usable outgoing
        channel.

        Exact failed channel keys are excluded when available.
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

        excluded = self._normalize_edge(
            excluded_edge
        )

        for neighbor in neighbors:

            if excluded is not None:

                if (
                    excluded[0] == node
                    and
                    excluded[1] == neighbor
                ):

                    # For a MultiDiGraph, another parallel
                    # channel may still be usable.
                    if (
                        len(excluded) == 3
                        and
                        self._has_another_usable_channel(
                            u=node,
                            v=neighbor,
                            amount=amount,
                            excluded_key=excluded[2]
                        )
                    ):

                        return True

                    # Without exact key, the endpoint itself
                    # is excluded.
                    continue

            if self._channel_available(
                node,
                neighbor,
                amount
            ):

                return True

        return False

    # ==========================================================
    # Parallel Channel
    # ==========================================================

    def _has_another_usable_channel(
        self,
        u,
        v,
        amount,
        excluded_key
    ):
        """
        Check for another usable parallel channel.
        """

        if not self._is_multigraph():
            return False

        try:

            edge_data = self.network.get_edge_data(
                u,
                v
            )

        except Exception:

            return False

        if not edge_data:
            return False

        for key, data in edge_data.items():

            if key == excluded_key:
                continue

            if self._edge_data_usable(
                data,
                amount
            ):

                return True

        return False

    # ==========================================================
    # Amount Validation
    # ==========================================================

    @staticmethod
    def _valid_amount(
        amount
    ):
        try:

            value = float(
                amount
            )

        except (
            TypeError,
            ValueError
        ):

            return False

        return value >= 0

    # ==========================================================
    # Route Validation
    # ==========================================================

    @staticmethod
    def _validate_route(
        route
    ):
        """
        Validate a node route.

        Requirements
        ------------
        - list/tuple
        - at least two nodes
        - no repeated nodes
        - source != destination
        """

        if not isinstance(
            route,
            (list, tuple)
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
        **kwargs
    ):
        """
        Create normalized result dictionary.
        """

        if isinstance(
            original_route,
            (list, tuple)
        ):

            original_route = list(
                original_route
            )

        result = {
            "success": bool(success),
            "status": status,
            "reason": reason,
            "original_route": original_route
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

    print()
    print("=" * 72)
    print("PARTIAL BACKTRACKING MODULE TEST")
    print("=" * 72)

    # ----------------------------------------------------------
    # Build MultiDiGraph
    # ----------------------------------------------------------

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
        G.nodes[u]["is_online"] = True

        G.nodes[v]["available"] = True
        G.nodes[v]["is_online"] = True

    # ----------------------------------------------------------
    # Bucket-compatible candidate
    # ----------------------------------------------------------

    class TestBucket:

        def __init__(self):

            self.candidates = [

                {
                    "path": [
                        "A",
                        "B",
                        "F",
                        "G",
                        "E"
                    ],

                    "edges": [
                        ("A", "B", 0),
                        ("B", "F", 0),
                        ("F", "G", 0),
                        ("G", "E", 0)
                    ],

                    "success": None
                }
            ]

            self.failed_candidates = []

    bucket = TestBucket()

    backtracker = PartialBacktracker(
        network=G,
        bucket=bucket
    )

    # ----------------------------------------------------------
    # Failed route
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Execute
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
    # Print
    # ----------------------------------------------------------

    print()
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
        f"Candidate rank      : "
        f"{result.get('candidate_rank')}"
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
        f"Alternative edges   : "
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

    print("=" * 72)

    # ----------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------

    assert result["success"] is True
    assert result["status"] == "alternative_found"

    assert result["branch_point"] == "B"

    assert result["branch_index"] == 1

    assert result["new_route"] == [
        "A",
        "B",
        "F",
        "G",
        "E"
    ]

    assert result["alternative_suffix"] == [
        "B",
        "F",
        "G",
        "E"
    ]

    assert result["new_edges"] == [
        ("A", "B", 0),
        ("B", "F", 0),
        ("F", "G", 0),
        ("G", "E", 0)
    ]

    assert result["attempt_id"] == 0

    print()
    print(
        "PARTIAL BACKTRACKING STATUS : SUCCESS"
    )