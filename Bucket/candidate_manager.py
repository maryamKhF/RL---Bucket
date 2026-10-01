# Bucket/candidate_manager.py

from .bucket import Bucket


class CandidateManager:
    """
    Manage routing candidates before they are stored in Bucket.

    Candidate format
    ----------------
    The preferred candidate format is a dictionary produced by
    Pathfinding.top_k_paths(), for example:

        {
            "path": [...],
            "edges": [...],
            "cost": ...,
            "total_fee": ...,
            "total_delay": ...,
            "reliability": ...,
            ...
        }

    Physical edge format produced by Top-K:

        {
            "source": ...,
            "target": ...,
            "channel_key": ...,
            "scid": ...,
            "data": ...
        }

    Legacy tuple/list candidates are also supported.

    Legacy physical edge formats:

        (u, v, key)

    or:

        (u, v)

    Responsibilities
    ----------------
    - store candidate paths
    - validate candidate paths
    - filter invalid candidates
    - rank candidates
    - select a limited number of candidates
    - create Bucket objects

    Important
    ---------
    Missing directional liquidity is treated as UNKNOWN.

    It must NOT be interpreted as zero liquidity.

    Channel capacity is also NOT interpreted as directional
    liquidity.

    The current Top-K implementation returns physical edges
    as dictionaries. CandidateManager therefore normalizes
    those dictionaries before accessing the NetworkX graph.
    """

    def __init__(
        self,
        candidates=None,
        max_candidates=10
    ):
        """
        Initialize CandidateManager.

        Parameters
        ----------
        candidates:
            Initial candidate routes.

        max_candidates:
            Maximum number of candidates that may be retained
            when creating a Bucket.
        """

        if candidates is None:

            self.candidates = []

        else:

            self.candidates = list(
                candidates
            )

        try:

            self.max_candidates = max(
                0,
                int(
                    max_candidates
                )
            )

        except (
            TypeError,
            ValueError
        ):

            self.max_candidates = 10

    # ==================================================
    # Candidate Helpers
    # ==================================================

    @staticmethod
    def _get_path(
        candidate
    ):
        """
        Return the node path of a candidate.

        Supported candidate formats:

            {
                "path": [...]
            }

        or legacy:

            (
                path,
                edges,
                score
            )
        """

        if isinstance(
            candidate,
            dict
        ):

            path = candidate.get(
                "path",
                []
            )

            if path is None:

                return []

            return path

        if isinstance(
            candidate,
            (tuple, list)
        ):

            if len(candidate) >= 1:

                path = candidate[0]

                if path is None:

                    return []

                return path

        return []

    @staticmethod
    def _get_edges(
        candidate
    ):
        """
        Return the physical edges of a candidate.

        Current Top-K format:

            {
                "edges": [...]
            }

        Legacy format:

            (
                path,
                edges,
                score
            )
        """

        if isinstance(
            candidate,
            dict
        ):

            edges = candidate.get(
                "edges",
                []
            )

            if edges is None:

                return []

            return edges

        if isinstance(
            candidate,
            (tuple, list)
        ):

            if len(candidate) >= 2:

                edges = candidate[1]

                if edges is None:

                    return []

                return edges

        return []

    @staticmethod
    def _get_score(
        candidate
    ):
        """
        Return candidate routing score.

        Current Top-K candidates use:

            candidate["cost"]

        Some candidate implementations may use:

            candidate["score"]

        Legacy candidates may use:

            candidate[2]

        Lower values represent better candidates.
        """

        if isinstance(
            candidate,
            dict
        ):

            cost = candidate.get(
                "cost"
            )

            if cost is not None:

                try:

                    return float(
                        cost
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

            score = candidate.get(
                "score"
            )

            if score is not None:

                try:

                    return float(
                        score
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    pass

            return float(
                "inf"
            )

        if isinstance(
            candidate,
            (tuple, list)
        ):

            if len(candidate) >= 3:

                try:

                    return float(
                        candidate[2]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    return float(
                        "inf"
                    )

        return float(
            "inf"
        )

    # ==================================================
    # Edge Normalization
    # ==================================================

    @staticmethod
    def _normalize_edge(
        edge
    ):
        """
        Normalize a physical edge into:

            (source, target, channel_key)

        Supported formats
        -----------------

        1. Current Top-K dictionary format:

            {
                "source": u,
                "target": v,
                "channel_key": key,
                "scid": ...,
                "data": ...
            }

        2. NetworkX MultiDiGraph edge:

            (u, v, key)

        3. NetworkX DiGraph edge:

            (u, v)

        Returns
        -------
        tuple or None

            Canonical:

                (u, v, key)

            or None if the edge representation is invalid.

        Important
        ---------
        Top-K currently emits dictionary edges. This method is
        therefore the compatibility boundary between Pathfinding
        and Bucket.
        """

        # --------------------------------------------------
        # Current Top-K physical edge dictionary
        # --------------------------------------------------

        if isinstance(
            edge,
            dict
        ):

            source = edge.get(
                "source"
            )

            target = edge.get(
                "target"
            )

            channel_key = edge.get(
                "channel_key"
            )

            if source is None:

                return None

            if target is None:

                return None

            if channel_key is None:

                channel_key = 0

            return (
                source,
                target,
                channel_key
            )

        # --------------------------------------------------
        # Legacy tuple/list representation
        # --------------------------------------------------

        if not isinstance(
            edge,
            (tuple, list)
        ):

            return None

        # --------------------------------------------------
        # MultiDiGraph / MultiGraph
        # --------------------------------------------------

        if len(edge) >= 3:

            return (
                edge[0],
                edge[1],
                edge[2]
            )

        # --------------------------------------------------
        # DiGraph / Graph
        # --------------------------------------------------

        if len(edge) == 2:

            return (
                edge[0],
                edge[1],
                0
            )

        return None

    # ==================================================
    # Channel Access
    # ==================================================

    @staticmethod
    def _get_channel(
        G,
        edge
    ):
        """
        Safely retrieve the physical channel associated
        with an edge.

        Supports:

            MultiGraph
            MultiDiGraph
            Graph
            DiGraph

        The edge may be either:

            Top-K dictionary

        or:

            tuple/list
        """

        if G is None:

            return None

        normalized = (
            CandidateManager._normalize_edge(
                edge
            )
        )

        if normalized is None:

            return None

        source, target, channel_key = (
            normalized
        )

        try:

            if G.is_multigraph():

                return G.edges[
                    source,
                    target,
                    channel_key
                ]

            return G.edges[
                source,
                target
            ]

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            return None

    # ==================================================
    # Liquidity
    # ==================================================

    @staticmethod
    def _get_known_liquidity(
        channel
    ):
        """
        Return known directional liquidity.

        Priority
        --------

        1. estimated_liquidity
        2. liquidity_uv
        3. balance_uv

        Missing directional liquidity is UNKNOWN and therefore
        returns None.

        Channel capacity is deliberately NOT used as directional
        liquidity.
        """

        if channel is None:

            return None

        for key in (
            "estimated_liquidity",
            "liquidity_uv",
            "balance_uv"
        ):

            value = channel.get(
                key,
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

                continue

            return value

        return None

    @staticmethod
    def _is_channel_available(
        channel
    ):
        """
        Check whether a physical channel is available.

        Missing availability is interpreted as available.

        This preserves compatibility with graph snapshots that
        do not explicitly store an 'available' field.
        """

        if channel is None:

            return False

        return bool(
            channel.get(
                "available",
                True
            )
        )

    @staticmethod
    def _liquidity_is_sufficient(
        channel,
        amount
    ):
        """
        Determine whether known directional liquidity can carry
        the requested payment amount.

        Returns
        -------

        True
            Known liquidity is sufficient.

        False
            Known liquidity is insufficient.

        None
            Directional liquidity is unknown.

        Important
        ---------
        UNKNOWN is not treated as failure.

        This prevents CandidateManager from incorrectly rejecting
        valid routes merely because a snapshot does not expose
        directional liquidity.
        """

        liquidity = (
            CandidateManager
            ._get_known_liquidity(
                channel
            )
        )

        if liquidity is None:

            return None

        try:

            amount = float(
                amount
            )

        except (
            TypeError,
            ValueError
        ):

            return False

        return (
            liquidity >= amount
        )

    # ==================================================
    # Candidate Validation
    # ==================================================

    def _candidate_is_structurally_valid(
        self,
        candidate
    ):
        """
        Check the basic structure of a candidate.

        A valid candidate must contain:

            path
            edges

        and both must be non-empty.
        """

        if candidate is None:

            return False

        path = self._get_path(
            candidate
        )

        edges = self._get_edges(
            candidate
        )

        if path is None:

            return False

        if edges is None:

            return False

        if len(path) < 2:

            return False

        if len(edges) < 1:

            return False

        return True

    # ==================================================
    # Candidate Management
    # ==================================================

    def add_candidate(
        self,
        candidate
    ):
        """
        Add a candidate route.

        None is ignored.

        Structural validation is intentionally deferred to
        filter_candidates(), because graph-dependent validation
        requires access to G.
        """

        if candidate is None:

            return

        self.candidates.append(
            candidate
        )

    def remove_candidate(
        self,
        candidate
    ):
        """
        Remove a candidate from CandidateManager.
        """

        if candidate in self.candidates:

            self.candidates.remove(
                candidate
            )

    # ==================================================
    # Filtering
    # ==================================================

    def filter_candidates(
        self,
        G,
        amount
    ):
        """
        Filter candidates that cannot support the requested
        transaction amount.

        A candidate is rejected if:

        1. Its structure is invalid.
        2. Its edge list is invalid.
        3. A physical channel does not exist.
        4. A physical channel is unavailable.
        5. Known directional liquidity is insufficient.

        A candidate is retained when directional liquidity is
        unknown.

        Channel capacity is never used as directional liquidity.
        """

        valid_candidates = []

        try:

            amount = float(
                amount
            )

        except (
            TypeError,
            ValueError
        ):

            amount = 0.0

        for candidate in self.candidates:

            # ----------------------------------------------
            # Candidate structure
            # ----------------------------------------------

            if not self._candidate_is_structurally_valid(
                candidate
            ):

                continue

            edges = (
                self._get_edges(
                    candidate
                )
            )

            valid = True

            # ----------------------------------------------
            # Validate every physical edge
            # ----------------------------------------------

            for edge in edges:

                normalized = (
                    self._normalize_edge(
                        edge
                    )
                )

                # ------------------------------------------
                # Invalid edge representation
                # ------------------------------------------

                if normalized is None:

                    valid = False

                    break

                # ------------------------------------------
                # Physical channel lookup
                # ------------------------------------------

                channel = (
                    self._get_channel(
                        G,
                        edge
                    )
                )

                if channel is None:

                    valid = False

                    break

                # ------------------------------------------
                # Channel availability
                # ------------------------------------------

                if not self._is_channel_available(
                    channel
                ):

                    valid = False

                    break

                # ------------------------------------------
                # Directional liquidity
                # ------------------------------------------

                liquidity_result = (
                    self._liquidity_is_sufficient(
                        channel,
                        amount
                    )
                )

                # ------------------------------------------
                # Known insufficient liquidity
                # ------------------------------------------

                if liquidity_result is False:

                    valid = False

                    break

                # ------------------------------------------
                # UNKNOWN liquidity
                #
                # Keep candidate.
                # ------------------------------------------

                if liquidity_result is None:

                    continue

            if valid:

                valid_candidates.append(
                    candidate
                )

        self.candidates = (
            valid_candidates
        )

        return self.candidates

    # ==================================================
    # Ranking
    # ==================================================

    def rank_candidates(
        self
    ):
        """
        Sort candidates according to routing cost.

        Lower cost/score means higher priority.

        Current Top-K candidate:

            candidate["cost"]

        Legacy candidate:

            candidate[2]
        """

        self.candidates.sort(
            key=self._get_score
        )

        return self.candidates

    # ==================================================
    # Selection
    # ==================================================

    def get_best_candidates(
        self,
        k=None
    ):
        """
        Return the best k candidates.

        If k is None, max_candidates is used.
        """

        if k is None:

            k = self.max_candidates

        try:

            k = int(
                k
            )

        except (
            TypeError,
            ValueError
        ):

            k = self.max_candidates

        k = max(
            0,
            k
        )

        k = min(
            k,
            len(
                self.candidates
            )
        )

        return list(
            self.candidates[
                :k
            ]
        )

    def get_candidate(
        self,
        index
    ):
        """
        Return a candidate by zero-based index.

        Returns None for invalid indexes.
        """

        try:

            index = int(
                index
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if index < 0:

            return None

        if index >= len(
            self.candidates
        ):

            return None

        return self.candidates[
            index
        ]

    # ==================================================
    # Candidate Status
    # ==================================================

    def remove_failed_path(
        self,
        candidate
    ):
        """
        Remove a failed candidate from CandidateManager.

        This method is primarily a management helper.

        Once candidates are stored in Bucket, the Bucket /
        PartialBacktracker should control fallback selection
        so that the original candidate set is preserved.
        """

        if candidate in self.candidates:

            self.candidates.remove(
                candidate
            )

        return self.candidates

    def has_candidates(
        self
    ):
        """
        Return True when at least one candidate exists.
        """

        return (
            len(
                self.candidates
            )
            > 0
        )

    # ==================================================
    # Bucket Creation
    # ==================================================

    def create_bucket(
        self,
        tx_id,
        k=None
    ):
        """
        Create a Bucket using the best available candidates.

        Parameters
        ----------
        tx_id:
            Transaction identifier.

        k:
            Number of candidates to place in Bucket.

            If omitted, max_candidates is used.

        Returns
        -------
        Bucket
        """

        selected_candidates = (
            self.get_best_candidates(
                k
            )
        )

        return Bucket(
            tx_id,
            tx_id,
            selected_candidates
        )

    # ==================================================
    # Information
    # ==================================================

    def summary(
        self
    ):
        """
        Return a structured summary of current candidates.

        The method supports both the current dictionary-based
        candidate representation and legacy tuple/list candidates.
        """

        summary_candidates = []

        for index, candidate in enumerate(
            self.candidates
        ):

            path = (
                self._get_path(
                    candidate
                )
            )

            score = (
                self._get_score(
                    candidate
                )
            )

            item = {
                "index": index,
                "rank": index + 1,
                "path": list(path)
                    if path is not None
                    else [],
                "score": score,
            }

            # ----------------------------------------------
            # Preserve useful Top-K metadata
            # ----------------------------------------------

            if isinstance(
                candidate,
                dict
            ):

                for key in (
                    "cost",
                    "total_fee",
                    "total_delay",
                    "reliability",
                    "failure_probability",
                    "hop_count",
                ):

                    if key in candidate:

                        item[key] = (
                            candidate[key]
                        )

            summary_candidates.append(
                item
            )

        return {
            "candidate_count":
                len(
                    self.candidates
                ),

            "max_candidates":
                self.max_candidates,

            "candidates":
                summary_candidates,
        }


# ======================================================
# Helper Function
# ======================================================

def make_bucket(
    tx_id,
    candidates
):
    """
    Quick Bucket constructor.

    Parameters
    ----------
    tx_id:
        Transaction identifier.

    candidates:
        Candidate routing paths.

    Returns
    -------
    Bucket
    """

    if candidates is None:

        candidates = []

    return Bucket(
        tx_id,
        tx_id,
        list(
            candidates
        )
    )