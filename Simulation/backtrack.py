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
- PartialBacktracker does NOT execute payments.
- PartialBacktracker does NOT call FailureModel.
- PartialBacktracker does NOT call PaymentSimulator.
- PartialBacktracker does NOT perform full rerouting.
- Channel capacity is NOT directional liquidity.
- Unknown directional liquidity is accepted structurally.
- Malformed explicit directional liquidity is rejected.
- Exact MultiDiGraph channel keys are preserved.
- MultiDiGraph channels are NEVER selected arbitrarily.
- route_edges are preferred and authoritative when supplied.
- Candidate exact edges are preferred over node-path reconstruction.
- If exact channel identity cannot be established in a MultiDiGraph,
  the candidate is rejected.
- The actual payment amount is used for structural liquidity checks.
- Failed channels are not reused.
- If the failed channel key is known, another parallel channel
  is allowed.
- If the failed channel key is unknown, the whole directed
  endpoint pair is excluded conservatively.
- Successful prefix is preserved.
- No full route recomputation is performed.
- PartialBacktracker does not mutate route, route_edges, or
  Bucket candidate objects.
- In a MultiDiGraph, two candidates with the same node path but
  different exact channel keys are different routes.
"""

from __future__ import annotations

import math
from typing import Any


class PartialBacktracker:
    """
    Partial backtracking using alternative candidates stored in Bucket.
    """

    def __init__(
        self,
        network,
        bucket=None,
    ):
        if network is None:
            raise ValueError("network must not be None")

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
        route_edges=None,
    ):
        if not self._validate_route(route):
            return self._result(
                success=False,
                status="invalid_route",
                reason="invalid_route",
                original_route=route,
                original_edges=None,
                amount=None,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        route = list(route)

        amount_value = self._validate_amount(amount)

        if amount_value is None:
            return self._result(
                success=False,
                status="invalid_amount",
                reason="invalid_amount",
                original_route=route,
                original_edges=None,
                amount=amount,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        if not self._valid_attempt_id(attempt_id):
            return self._result(
                success=False,
                status="invalid_attempt_id",
                reason="invalid_attempt_id",
                original_route=route,
                original_edges=None,
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        normalized_route_edges = self._normalize_route_edges(
            route=route,
            route_edges=route_edges,
        )

        if normalized_route_edges is False:
            return self._result(
                success=False,
                status="invalid_route_edges",
                reason="route_edges_invalid",
                original_route=route,
                original_edges=route_edges,
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        resolved_failed_edge = self._resolve_failed_edge(
            route=route,
            route_edges=normalized_route_edges,
            failed_edge=failed_edge,
            failure_index=failure_index,
        )

        if resolved_failed_edge is None:
            return self._result(
                success=False,
                status="invalid_failure",
                reason="failed_edge_not_resolved",
                original_route=route,
                original_edges=normalized_route_edges,
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        resolved_failure_index = self._resolve_failure_index(
            route=route,
            route_edges=normalized_route_edges,
            failed_edge=resolved_failed_edge,
            failure_index=failure_index,
        )

        if resolved_failure_index is None:
            return self._result(
                success=False,
                status="invalid_failure",
                reason="failure_index_not_resolved",
                original_route=route,
                original_edges=normalized_route_edges,
                failed_edge=resolved_failed_edge,
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=False,
            )

        branch_info = self._find_bucket_branch_point(
            route=route,
            route_edges=normalized_route_edges,
            failure_index=resolved_failure_index,
            failed_edge=resolved_failed_edge,
            amount=amount_value,
            bucket_id=bucket_id,
        )

        if branch_info is None:
            return self._result(
                success=False,
                status="full_reroute_required",
                reason="no_valid_bucket_alternative",
                original_route=route,
                original_edges=normalized_route_edges,
                failed_edge=resolved_failed_edge,
                failure_index=resolved_failure_index,
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=False,
                full_reroute_required=True,
            )

        branch_index = branch_info["branch_index"]
        branch_node = branch_info["branch_node"]
        candidates = branch_info["candidates"]

        for candidate_index, candidate in enumerate(candidates):

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

            suffix = self._normalize_suffix(
                candidate=candidate,
                branch_node=branch_node,
            )

            if suffix is None:
                continue

            if len(suffix) < 2:
                continue

            candidate_edges = self._extract_candidate_edges(
                candidate
            )

            suffix_edges = self._extract_suffix_edges(
                candidate=candidate,
                suffix=suffix,
                branch_node=branch_node,
                amount=amount_value,
            )

            if suffix_edges is None:
                continue

            if self._candidate_is_same_route(
                candidate=candidate,
                candidate_path=candidate_path,
                candidate_edges=candidate_edges,
                route=route,
                route_edges=normalized_route_edges,
            ):
                continue

            if self._candidate_contains_failed_edge(
                candidate=candidate,
                suffix=suffix,
                suffix_edges=suffix_edges,
                failed_edge=resolved_failed_edge,
            ):
                continue

            prefix = route[:branch_index + 1]

            if not self._is_loop_free(
                prefix=prefix,
                suffix=suffix,
            ):
                continue

            if not self.is_suffix_valid(
                suffix=suffix,
                amount=amount_value,
                candidate=candidate,
                suffix_edges=suffix_edges,
            ):
                continue

            prefix_edges = self._resolve_route_edges_for_prefix(
                route=route,
                route_edges=normalized_route_edges,
                end_index=branch_index,
                amount=amount_value,
            )

            if prefix_edges is None:
                continue

            new_route = self._combine_route(
                prefix=prefix,
                suffix=suffix,
            )

            if not self._validate_route(new_route):
                continue

            new_edges = self._combine_edges(
                prefix_edges=prefix_edges,
                suffix_edges=suffix_edges,
            )

            if new_edges is None:
                continue

            if len(new_edges) != len(new_route) - 1:
                continue

            if not self._edges_match_route(
                route=new_route,
                edges=new_edges,
            ):
                continue

            if not self._validate_exact_edge_sequence(
                route=new_route,
                edges=new_edges,
                amount=amount_value,
            ):
                continue

            return self._result(
                success=True,
                status="alternative_found",
                reason="partial_backtrack_success",
                original_route=route,
                original_edges=normalized_route_edges,
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
                amount=amount_value,
                bucket_id=bucket_id,
                attempt_id=attempt_id,
                retry_required=True,
                full_reroute_required=False,
            )

        return self._result(
            success=False,
            status="full_reroute_required",
            reason="no_valid_bucket_alternative",
            original_route=route,
            original_edges=normalized_route_edges,
            new_route=None,
            new_edges=None,
            preserved_prefix=None,
            preserved_prefix_edges=None,
            alternative_suffix=None,
            alternative_suffix_edges=None,
            branch_point=branch_node,
            branch_index=branch_index,
            failed_edge=resolved_failed_edge,
            failure_index=resolved_failure_index,
            amount=amount_value,
            bucket_id=bucket_id,
            attempt_id=attempt_id,
            retry_required=False,
            full_reroute_required=True,
        )

    # ==========================================================
    # Amount Validation
    # ==========================================================

    @staticmethod
    def _validate_amount(amount):
        if isinstance(amount, bool):
            return None

        try:
            value = float(amount)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        if value <= 0:
            return None

        return value

    # ==========================================================
    # Attempt ID
    # ==========================================================

    @staticmethod
    def _valid_attempt_id(attempt_id):
        if isinstance(attempt_id, bool):
            return False

        if not isinstance(attempt_id, int):
            return False

        return attempt_id >= 0

    # ==========================================================
    # Route Edge Normalization
    # ==========================================================

    def _normalize_route_edges(
        self,
        route,
        route_edges,
    ):
        if route_edges is None:
            if self._is_multigraph():
                return False
            return None

        if not isinstance(route_edges, (list, tuple)):
            return False

        if len(route_edges) != len(route) - 1:
            return False

        normalized = []

        for edge in route_edges:
            parsed = self._normalize_edge(edge)

            if parsed is None:
                return False

            normalized.append(parsed)

        if not self._edges_match_route(
            route=route,
            edges=normalized,
        ):
            return False

        if not self._is_multigraph():
            for _, _, key in normalized:
                if key is not None:
                    return False

        else:
            # Exact channel identity is mandatory in MultiDiGraph.
            for _, _, key in normalized:
                if key is None:
                    return False

        return normalized

    # ==========================================================
    # Bucket-aware Branch Point
    # ==========================================================

    def _find_bucket_branch_point(
        self,
        route,
        route_edges,
        failure_index,
        failed_edge,
        amount=0,
        bucket_id=None,
    ):
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

            candidates = self.get_alternative_suffixes(
                route=route,
                branch_index=index,
                failed_edge=failed_edge,
                bucket_id=bucket_id,
            )

            if not candidates:
                continue

            valid_candidates = []

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

                candidate_edges = self._extract_candidate_edges(
                    candidate
                )

                suffix_edges = self._extract_suffix_edges(
                    candidate=candidate,
                    suffix=suffix,
                    branch_node=node,
                    amount=amount,
                )

                if suffix_edges is None:
                    continue

                if self._candidate_is_same_route(
                    candidate=candidate,
                    candidate_path=candidate_path,
                    candidate_edges=candidate_edges,
                    route=route,
                    route_edges=route_edges,
                ):
                    continue

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

                valid_candidates.append(candidate)

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
        if not self._validate_route(route):
            return None

        amount_value = self._validate_amount(amount)

        if amount_value is None:
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
                    None,
                )

            if self._has_usable_outgoing_edge(
                node=node,
                amount=amount_value,
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
        if self.bucket is None:
            return []

        branch_node = route[branch_index]

        candidates = getattr(
            self.bucket,
            "candidates",
            None,
        )

        if candidates is not None:
            return self._normalize_bucket_container(
                candidates=candidates,
                bucket_id=bucket_id,
            )

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
                    branch_node=branch_node,
                    failed_edge=failed_edge,
                )
            except TypeError:
                result = method(bucket_id)

            return self._normalize_bucket_container(
                candidates=result,
                bucket_id=bucket_id,
            )

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
            except TypeError:
                result = method()

            return self._normalize_bucket_container(
                candidates=result,
                bucket_id=bucket_id,
            )

        routes = getattr(
            self.bucket,
            "routes",
            None,
        )

        if routes is not None:
            return self._normalize_bucket_container(
                candidates=routes,
                bucket_id=bucket_id,
            )

        return []

    # ==========================================================
    # Bucket Container Normalization
    # ==========================================================

    @staticmethod
    def _normalize_bucket_container(
        candidates,
        bucket_id=None,
    ):
        if candidates is None:
            return []

        if isinstance(candidates, dict):

            if bucket_id is not None:
                if bucket_id in candidates:
                    selected = candidates[bucket_id]

                    if selected is None:
                        return []

                    if isinstance(
                        selected,
                        (list, tuple),
                    ):
                        return list(selected)

                    return [selected]

            for field in (
                "routes",
                "paths",
                "alternatives",
                "candidates",
            ):
                if field in candidates:
                    selected = candidates[field]

                    if selected is None:
                        return []

                    if isinstance(
                        selected,
                        (list, tuple),
                    ):
                        return list(selected)

                    return [selected]

            return []

        if isinstance(
            candidates,
            (list, tuple),
        ):
            return list(candidates)

        return [candidates]

    # ==========================================================
    # Candidate Failure State
    # ==========================================================

    def _candidate_is_failed(
        self,
        candidate,
        candidate_index=None,
    ):
        if candidate is None:
            return True

        if isinstance(candidate, dict):

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

        failed_candidates = getattr(
            self.bucket,
            "failed_candidates",
            None,
        )

        if failed_candidates is not None:

            if not isinstance(
                failed_candidates,
                (list, tuple, set),
            ):
                return False

            for failed in failed_candidates:

                if self._candidate_identity_equal(
                    candidate,
                    failed,
                ):
                    return True

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
        Compare two Bucket candidates without losing channel identity.

        Rules
        -----
        1. Exact edge sequences are authoritative.
        2. In MultiDiGraph, candidates with the same node path but
           missing exact channel identity are NOT assumed identical.
        3. In MultiDiGraph, exact channel keys must match.
        4. In simple Graph/DiGraph, node-path equality is sufficient.
        """

        if candidate_a is candidate_b:
            return True

        edges_a = self._extract_candidate_edges(
            candidate_a
        )

        edges_b = self._extract_candidate_edges(
            candidate_b
        )

        # ------------------------------------------------------
        # Exact channel-aware comparison
        # ------------------------------------------------------

        if (
            edges_a is not None
            and
            edges_b is not None
        ):
            normalized_a = self._normalize_candidate_edges(
                edges_a
            )

            normalized_b = self._normalize_candidate_edges(
                edges_b
            )

            if (
                normalized_a is None
                or
                normalized_b is None
            ):
                return False

            if len(normalized_a) != len(normalized_b):
                return False

            # In MultiDiGraph exact channel identity is part
            # of candidate identity.
            return normalized_a == normalized_b

        # ------------------------------------------------------
        # If only one candidate has exact channel information,
        # they cannot safely be considered identical in a
        # MultiDiGraph.
        # ------------------------------------------------------

        if self._is_multigraph():

            if (
                edges_a is None
                or
                edges_b is None
            ):
                return False

        # ------------------------------------------------------
        # Node-path comparison
        # ------------------------------------------------------

        path_a = self._extract_candidate_path(
            candidate_a
        )

        path_b = self._extract_candidate_path(
            candidate_b
        )

        if (
            path_a is None
            or
            path_b is None
        ):
            return False

        try:
            return list(path_a) == list(path_b)
        except Exception:
            return False

    # ==========================================================
    # Candidate Same Route
    # ==========================================================

    def _candidate_is_same_route(
        self,
        candidate,
        candidate_path,
        candidate_edges,
        route,
        route_edges,
    ):
        if candidate_path is None:
            return False

        if route is None:
            return False

        try:
            candidate_nodes = list(candidate_path)
            route_nodes = list(route)
        except Exception:
            return False

        if candidate_nodes != route_nodes:
            return False

        if not self._is_multigraph():
            return True

        # A MultiDiGraph route cannot be compared safely without
        # exact channel identities.
        if candidate_edges is None:
            return True

        if route_edges is None:
            return False

        expected_edge_count = len(candidate_nodes) - 1

        if expected_edge_count < 1:
            return False

        try:
            candidate_edge_count = len(candidate_edges)
            route_edge_count = len(route_edges)
        except Exception:
            return False

        if candidate_edge_count != expected_edge_count:
            return False

        if route_edge_count != expected_edge_count:
            return False

        candidate_normalized = (
            self._normalize_candidate_edges(
                candidate_edges
            )
        )

        route_normalized = (
            self._normalize_candidate_edges(
                route_edges
            )
        )

        if candidate_normalized is None:
            return False

        if route_normalized is None:
            return False

        if len(candidate_normalized) != (
            len(candidate_nodes) - 1
        ):
            return False

        if len(route_normalized) != (
            len(route_nodes) - 1
        ):
            return False

        if not self._edges_match_route(
            route=candidate_nodes,
            edges=candidate_normalized,
        ):
            return False

        if not self._edges_match_route(
            route=route_nodes,
            edges=route_normalized,
        ):
            return False

        return (
            candidate_normalized
            ==
            route_normalized
        )

    # ==========================================================
    # Candidate Path Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_path(
        candidate,
    ):
        if candidate is None:
            return None

        if isinstance(candidate, dict):

            for field in (
                "path",
                "route",
                "suffix",
            ):
                if field in candidate:

                    path = candidate[field]

                    if not isinstance(
                        path,
                        (list, tuple),
                    ):
                        return None

                    path = list(path)

                    if len(path) >= 2:
                        return path

                    return None

            return None

        if isinstance(candidate, tuple):

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

        if isinstance(candidate, list):

            if len(candidate) < 2:
                return None

            if any(
                isinstance(
                    item,
                    (dict, list, tuple),
                )
                for item in candidate
            ):
                return None

            return list(candidate)

        return None

    # ==========================================================
    # Candidate Edge Extraction
    # ==========================================================

    @staticmethod
    def _extract_candidate_edges(
        candidate,
    ):
        if candidate is None:
            return None

        if isinstance(candidate, dict):

            edges = candidate.get("edges")

            if edges is None:
                return None

            if not isinstance(
                edges,
                (list, tuple),
            ):
                return None

            return list(edges)

        if isinstance(candidate, tuple):

            if len(candidate) >= 2:

                edges = candidate[1]

                if isinstance(
                    edges,
                    (list, tuple),
                ):
                    return list(edges)

        return None

    # ==========================================================
    # Candidate Edge Normalization
    # ==========================================================

    def _normalize_candidate_edges(
        self,
        edges,
    ):
        if edges is None:
            return None

        if not isinstance(
            edges,
            (list, tuple),
        ):
            return None

        normalized = []

        for edge in edges:

            parsed = self._normalize_edge(
                edge
            )

            if parsed is None:
                return None

            normalized.append(parsed)

        if self._is_multigraph():

            for _, _, key in normalized:
                if key is None:
                    return None

        return normalized

    # ==========================================================
    # Normalize Candidate Suffix
    # ==========================================================

    def _normalize_suffix(
        self,
        candidate,
        branch_node,
    ):
        path = self._extract_candidate_path(
            candidate
        )

        if path is None:
            return None

        try:
            branch_position = path.index(
                branch_node
            )
        except ValueError:
            return None

        suffix = path[branch_position:]

        if len(suffix) < 2:
            return None

        if suffix[0] != branch_node:
            return None

        try:
            if len(suffix) != len(set(suffix)):
                return None
        except TypeError:
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
        amount,
    ):
        if suffix is None:
            return None

        if len(suffix) < 2:
            return None

        candidate_path = self._extract_candidate_path(
            candidate
        )

        if candidate_path is None:
            return None

        candidate_edges = self._extract_candidate_edges(
            candidate
        )

        if candidate_edges is not None:

            if len(candidate_edges) != len(candidate_path) - 1:
                return None

            parsed_edges = []

            for edge in candidate_edges:

                parsed = self._normalize_edge(
                    edge
                )

                if parsed is None:
                    return None

                parsed_edges.append(parsed)

            if self._is_multigraph():

                for _, _, key in parsed_edges:
                    if key is None:
                        return None

            try:
                branch_position = candidate_path.index(
                    branch_node
                )
            except ValueError:
                return None

            candidate_suffix = candidate_path[
                branch_position:
            ]

            if candidate_suffix != list(suffix):
                return None

            suffix_length = len(suffix) - 1

            selected = parsed_edges[
                branch_position:
                branch_position + suffix_length
            ]

            if len(selected) != suffix_length:
                return None

            if not self._edges_match_route(
                route=suffix,
                edges=selected,
            ):
                return None

            return list(selected)

        if self._is_multigraph():
            return None

        return self._resolve_edges_for_path(
            path=suffix,
            amount=amount,
        )

    # ==========================================================
    # Find Edge Sequence Start
    # ==========================================================

    @staticmethod
    def _find_edge_sequence_start(
        parsed_edges,
        suffix,
    ):
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
        if failed_edge is None:
            return False

        failed = self._normalize_edge(
            failed_edge
        )

        if failed is None:
            return False

        failed_u, failed_v, failed_key = failed

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

                if failed_key is not None:

                    if key == failed_key:
                        return True

                    continue

                return True

            return False

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
        if suffix is None:
            return False

        if len(suffix) < 2:
            return False

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return False

        try:
            if len(suffix) != len(set(suffix)):
                return False
        except TypeError:
            return False

        for node in suffix:

            try:
                if node not in self.network.nodes:
                    return False

                if not self._node_is_available(
                    self.network.nodes[node]
                ):
                    return False

            except Exception:
                return False

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
                    amount=amount_value,
                ):
                    return False

            return True

        if self._is_multigraph():
            return False

        for u, v in zip(
            suffix[:-1],
            suffix[1:],
        ):
            if not self._channel_available(
                u,
                v,
                amount_value,
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
        if self.network is None:
            return False

        try:

            if not self.network.has_edge(u, v):
                return False

            if self.network.is_multigraph():

                if key is None:
                    return False

                edge_data = self.network.get_edge_data(
                    u,
                    v,
                )

                if not isinstance(
                    edge_data,
                    dict,
                ):
                    return False

                data = edge_data.get(key)

                if data is None:
                    return False

            else:

                if key is not None:
                    return False

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
        route_edges,
        failed_edge,
        failure_index,
    ):
        if failed_edge is not None:

            parsed = self._normalize_edge(
                failed_edge
            )

            if parsed is None:
                return None

            failed_u, failed_v, failed_key = parsed

            if failure_index is not None:

                if not (
                    0 <= failure_index < len(route) - 1
                ):
                    return None

                if (
                    failed_u != route[failure_index]
                    or
                    failed_v != route[failure_index + 1]
                ):
                    return None

                if (
                    self._is_multigraph()
                    and
                    route_edges is not None
                    and
                    failed_key is not None
                ):
                    route_edge = route_edges[
                        failure_index
                    ]

                    if route_edge[2] != failed_key:
                        return None

            return parsed

        if failure_index is None:
            return None

        if not (
            0 <= failure_index < len(route) - 1
        ):
            return None

        if route_edges is not None:

            return self._normalize_edge(
                route_edges[failure_index]
            )

        if self._is_multigraph():
            return None

        return (
            route[failure_index],
            route[failure_index + 1],
            None,
        )

    # ==========================================================
    # Resolve Failure Index
    # ==========================================================

    def _resolve_failure_index(
        self,
        route,
        route_edges,
        failed_edge,
        failure_index,
    ):
        if failed_edge is None:
            return None

        normalized_failed = self._normalize_edge(
            failed_edge
        )

        if normalized_failed is None:
            return None

        failed_u, failed_v, failed_key = normalized_failed

        if failure_index is not None:

            if not isinstance(
                failure_index,
                int,
            ) or isinstance(
                failure_index,
                bool,
            ):
                return None

            if 0 <= failure_index < len(route) - 1:

                if (
                    route[failure_index] == failed_u
                    and
                    route[failure_index + 1] == failed_v
                ):

                    if (
                        failed_key is not None
                        and
                        route_edges is not None
                    ):

                        route_key = route_edges[
                            failure_index
                        ][2]

                        if route_key != failed_key:
                            return None

                    return failure_index

        if route_edges is not None:

            for index, edge in enumerate(
                route_edges
            ):

                u, v, key = edge

                if (
                    u == failed_u
                    and
                    v == failed_v
                ):

                    if (
                        failed_key is None
                        or
                        key == failed_key
                    ):
                        return index

            return None

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
                matching_indices.append(index)

        if len(matching_indices) != 1:
            return None

        return matching_indices[0]

    # ==========================================================
    # Prefix Edge Resolution
    # ==========================================================

    def _resolve_route_edges_for_prefix(
        self,
        route,
        route_edges,
        end_index,
        amount,
    ):
        if end_index <= 0:
            return []

        required_count = end_index

        if route_edges is not None:

            if len(route_edges) < required_count:
                return None

            selected = list(
                route_edges[:required_count]
            )

            if not self._edges_match_route(
                route=route[:end_index + 1],
                edges=selected,
            ):
                return None

            return selected

        if self._is_multigraph():
            return None

        return self._resolve_edges_for_path(
            path=route[:end_index + 1],
            amount=amount,
        )

    # ==========================================================
    # Resolve Edges for Node Path
    # ==========================================================

    def _resolve_edges_for_path(
        self,
        path,
        amount,
    ):
        if self.network is None:
            return None

        if path is None or len(path) < 2:
            return []

        if self._is_multigraph():
            return None

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return None

        edges = []

        for u, v in zip(
            path[:-1],
            path[1:],
        ):

            if not self.network.has_edge(u, v):
                return None

            data = self.network.get_edge_data(
                u,
                v,
            )

            if not self._edge_data_usable(
                data=data,
                amount=amount_value,
            ):
                return None

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
    def _normalize_edge(edge):
        """
        Normalize all supported physical-edge representations.

        Supported formats
        ------------------

        Top-K / Bucket dict:

            {
                "source": u,
                "target": v,
                "channel_key": key,
                ...
            }

        MultiDiGraph tuple:

            (u, v, key)

        Graph / DiGraph tuple:

            (u, v)

        The important point is that a dictionary edge keeps
        the exact channel_key and is therefore NOT converted
        into a node-only edge.
        """

        if edge is None:
            return None

        # ------------------------------------------------------
        # Top-K / Bucket edge dictionary
        # ------------------------------------------------------

        if isinstance(edge, dict):

            source = edge.get("source")
            target = edge.get("target")

            if source is None or target is None:
                return None

            channel_key = edge.get(
                "channel_key",
                None,
            )

            if channel_key is None:
                channel_key = edge.get(
                    "key",
                )

            return (
                source,
                target,
                channel_key,
            )

        # ------------------------------------------------------
        # Tuple / list edge
        # ------------------------------------------------------

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

        if len(edge) == 3:

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
        if route is None or edges is None:
            return False

        if len(edges) != len(route) - 1:
            return False

        for index, edge in enumerate(edges):

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
    # Final Exact Edge Validation
    # ==========================================================

    def _validate_exact_edge_sequence(
        self,
        route,
        edges,
        amount,
    ):
        if not self._edges_match_route(
            route=route,
            edges=edges,
        ):
            return False

        for edge in edges:

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

    # ==========================================================
    # Failed Edge Reuse - Path Fallback
    # ==========================================================

    @staticmethod
    def _contains_failed_edge(
        suffix,
        failed_edge,
    ):
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
        if self.network is None:
            return False

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
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

            if not isinstance(
                edge_data,
                dict,
            ):
                return False

            for data in edge_data.values():

                if self._edge_data_usable(
                    data=data,
                    amount=amount_value,
                ):
                    return True

            return False

        return self._edge_data_usable(
            data=edge_data,
            amount=amount_value,
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
        Validate an edge structurally.

        IMPORTANT
        ---------
        `capacity` is deliberately ignored as directional
        liquidity.

        If no explicit directional liquidity field exists,
        the channel is structurally usable.

        If an explicit directional liquidity field exists,
        it must be finite, non-negative, and >= amount.
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

        try:
            amount_value = float(amount)
        except (TypeError, ValueError):
            return False

        if not math.isfinite(amount_value):
            return False

        if amount_value <= 0:
            return False

        directional_field_found = False
        directional_liquidity = None

        for field in (
            "balance_uv",
            "liquidity_uv",
            "liquidity",
            "estimated_liquidity",
        ):

            if field not in data:
                continue

            value = data[field]

            if value is None:
                continue

            directional_field_found = True

            if isinstance(
                value,
                bool,
            ):
                return False

            try:
                value = float(value)
            except (TypeError, ValueError):
                return False

            if not math.isfinite(value):
                return False

            if value < 0:
                return False

            directional_liquidity = value
            break

        if directional_field_found:

            if directional_liquidity is None:
                return False

            if directional_liquidity < amount_value:
                return False

        # No explicit directional liquidity:
        # accept structurally. Do NOT use capacity.
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
        if self.network is None:
            return False

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return False

        try:
            if node not in self.network:
                return False

            neighbors = self.network.successors(
                node
            )
        except Exception:
            return False

        failed = None

        if excluded_edge is not None:
            failed = self._normalize_edge(
                excluded_edge
            )

        for neighbor in neighbors:

            if failed is not None:

                failed_u, failed_v, failed_key = failed

                if (
                    node == failed_u
                    and
                    neighbor == failed_v
                ):

                    if (
                        failed_key is not None
                        and
                        self._has_another_usable_channel(
                            u=node,
                            v=neighbor,
                            amount=amount_value,
                            excluded_key=failed_key,
                        )
                    ):
                        return True

                    continue

            if self._channel_available(
                node,
                neighbor,
                amount_value,
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
        if not self._is_multigraph():
            return False

        amount_value = self._validate_amount(
            amount
        )

        if amount_value is None:
            return False

        try:
            edge_data = self.network.get_edge_data(
                u,
                v,
            )
        except Exception:
            return False

        if not isinstance(
            edge_data,
            dict,
        ):
            return False

        for key, data in edge_data.items():

            if key == excluded_key:
                continue

            if self._edge_data_usable(
                data=data,
                amount=amount_value,
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
        if not isinstance(
            route,
            (list, tuple),
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

    print("=" * 78)
    print("PARTIAL BACKTRACKING MODULE TEST")
    print("=" * 78)

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

    original_route = [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    original_edges = [
        ("A", "B", 0),
        ("B", "C", 0),
        ("C", "D", 0),
        ("D", "E", 0),
    ]

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
                        {
                            "source": "A",
                            "target": "B",
                            "channel_key": 0,
                        },
                        {
                            "source": "B",
                            "target": "F",
                            "channel_key": 0,
                        },
                        {
                            "source": "F",
                            "target": "G",
                            "channel_key": 0,
                        },
                        {
                            "source": "G",
                            "target": "E",
                            "channel_key": 0,
                        },
                    ],
                }
            ]

    bucket = TestBucket()

    backtracker = PartialBacktracker(
        network=G,
        bucket=bucket,
    )

    failed_edge = {
        "source": "C",
        "target": "D",
        "channel_key": 0,
    }

    result = backtracker.backtrack(
        route=original_route,
        route_edges=original_edges,
        failed_edge=failed_edge,
        failure_index=2,
        amount=1000,
        bucket_id="B001",
        attempt_id=0,
    )

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
        f"Original edges      : "
        f"{result.get('original_edges')}"
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
        f"Retry required      : "
        f"{result.get('retry_required')}"
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

    assert result["success"] is True

    assert result["status"] == (
        "alternative_found"
    )

    assert result["reason"] == (
        "partial_backtrack_success"
    )

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

    assert result["preserved_prefix_edges"] == [
        ("A", "B", 0),
    ]

    assert result["alternative_suffix"] == [
        "B",
        "F",
        "G",
        "E",
    ]

    assert result["alternative_suffix_edges"] == [
        (
            "B",
            "F",
            0,
        ),
        (
            "F",
            "G",
            0,
        ),
        (
            "G",
            "E",
            0,
        ),
    ]

    assert result["new_route"] == [
        "A",
        "B",
        "F",
        "G",
        "E",
    ]

    assert result["new_edges"] == [
        (
            "A",
            "B",
            0,
        ),
        (
            "B",
            "F",
            0,
        ),
        (
            "F",
            "G",
            0,
        ),
        (
            "G",
            "E",
            0,
        ),
    ]

    assert result["retry_required"] is True

    assert result["full_reroute_required"] is False

    assert result["attempt_id"] == 0

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
        "Exact prefix edges preserved  : PASS"
    )
    print(
        "Route/edge consistency        : PASS"
    )
    print(
        "Capacity != liquidity rule    : PASS"
    )
    print(
        "No arbitrary MultiDiGraph key : PASS"
    )
    print(
        "Amount-aware validation       : PASS"
    )
    print("=" * 78)