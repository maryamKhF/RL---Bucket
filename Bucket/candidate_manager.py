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

    Legacy tuple/list candidates are also supported:

        (
            path,
            edges,
            score
        )

    Responsibilities
    ----------------
    - store candidate paths
    - filter invalid candidates
    - rank candidates
    - select a limited number of candidates
    - create Bucket object

    Important
    ---------
    Missing directional liquidity is treated as UNKNOWN.

    It must NOT be interpreted as zero liquidity and channel capacity
    must NOT be interpreted as directional liquidity.
    """

    def __init__(
        self,
        candidates=None,
        max_candidates=10
    ):

        self.candidates = (
            list(candidates)
            if candidates
            else []
        )

        self.max_candidates = max(
            0,
            int(max_candidates)
        )

    # ==================================================
    # Candidate Helpers
    # ==================================================

    @staticmethod
    def _get_path(candidate):
        """
        Return candidate path.
        """

        if isinstance(candidate, dict):
            return candidate.get(
                "path",
                []
            )

        if isinstance(candidate, (tuple, list)):
            if len(candidate) >= 1:
                return candidate[0]

        return []

    @staticmethod
    def _get_edges(candidate):
        """
        Return candidate edges.
        """

        if isinstance(candidate, dict):
            return candidate.get(
                "edges",
                []
            )

        if isinstance(candidate, (tuple, list)):
            if len(candidate) >= 2:
                return candidate[1]

        return []

    @staticmethod
    def _get_score(candidate):
        """
        Return candidate score.

        New candidates use "cost".
        Legacy candidates use element [2].
        """

        if isinstance(candidate, dict):

            if candidate.get("cost") is not None:
                return candidate["cost"]

            if candidate.get("score") is not None:
                return candidate["score"]

            return float("inf")

        if isinstance(candidate, (tuple, list)):

            if len(candidate) >= 3:
                try:
                    return float(
                        candidate[2]
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    return float("inf")

        return float("inf")

    @staticmethod
    def _normalize_edge(edge):
        """
        Normalize an edge into:

            (u, v, key)

        Supports:
            - (u, v, key)
            - (u, v)
        """

        if not isinstance(edge, (tuple, list)):
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
                edge[1],
                0
            )

        return None

    @staticmethod
    def _get_channel(G, edge):
        """
        Safely retrieve a channel from NetworkX graph.

        Supports MultiGraph / MultiDiGraph and ordinary Graph / DiGraph.
        """

        normalized = (
            CandidateManager._normalize_edge(
                edge
            )
        )

        if normalized is None:
            return None

        u, v, key = normalized

        try:

            if G.is_multigraph():

                return G.edges[
                    u,
                    v,
                    key
                ]

            return G.edges[
                u,
                v
            ]

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            return None

    @staticmethod
    def _get_known_liquidity(channel):
        """
        Return known directional liquidity.

        Priority:
            estimated_liquidity
            liquidity_uv
            balance_uv

        Missing liquidity is UNKNOWN and therefore returns None.

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

                value = float(value)

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
    def _is_channel_available(channel):
        """
        Check channel availability.

        Missing availability is interpreted as available.
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
        Check whether known directional liquidity
        can support the requested amount.

        Returns
        -------
        True
            Known liquidity is sufficient.

        False
            Known liquidity is insufficient.

        None
            Directional liquidity is unknown.
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
            amount = float(amount)
        except (
            TypeError,
            ValueError
        ):
            return False

        return liquidity >= amount

    # ==================================================
    # Candidate Management
    # ==================================================

    def add_candidate(
        self,
        candidate
    ):
        """
        Add a new candidate path.
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
        Filter candidates that cannot support
        the requested transaction amount.

        A candidate is invalid if:

        1. Its path/edges are invalid.
        2. One of its channels does not exist.
        3. A channel is unavailable.
        4. Directional liquidity is known and insufficient.

        Important:
        Missing directional liquidity is UNKNOWN, not zero.

        Channel capacity alone is NOT treated as directional
        liquidity.
        """

        valid_candidates = []

        try:
            amount = float(amount)
        except (
            TypeError,
            ValueError
        ):
            amount = 0.0

        for candidate in self.candidates:

            edges = (
                self._get_edges(
                    candidate
                )
            )

            if not edges:

                continue

            valid = True

            for edge in edges:

                channel = (
                    self._get_channel(
                        G,
                        edge
                    )
                )

                # ----------------------------------
                # Channel existence
                # ----------------------------------

                if channel is None:

                    valid = False

                    break

                # ----------------------------------
                # Channel availability
                # ----------------------------------

                if not self._is_channel_available(
                    channel
                ):

                    valid = False

                    break

                # ----------------------------------
                # Directional liquidity
                # ----------------------------------

                liquidity_result = (
                    self._liquidity_is_sufficient(
                        channel,
                        amount
                    )
                )

                # Known insufficient liquidity
                if liquidity_result is False:

                    valid = False

                    break

                # Unknown liquidity:
                # keep candidate
                #
                # We deliberately do NOT reject it.

            if valid:

                valid_candidates.append(
                    candidate
                )

        self.candidates = valid_candidates

        return self.candidates

    # ==================================================
    # Ranking
    # ==================================================

    def rank_candidates(self):
        """
        Sort candidates according to routing cost.

        Lower cost/score means a better candidate.

        For new Pathfinding candidates:
            candidate["cost"]

        For legacy candidates:
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

        If k is not provided, max_candidates
        is used.
        """

        if k is None:

            k = self.max_candidates

        try:
            k = int(k)
        except (
            TypeError,
            ValueError
        ):
            k = self.max_candidates

        k = max(
            0,
            min(
                k,
                len(self.candidates)
            )
        )

        return list(
            self.candidates[:k]
        )

    def get_candidate(
        self,
        index
    ):
        """
        Return candidate by zero-based index.
        """

        try:
            index = int(index)
        except (
            TypeError,
            ValueError
        ):
            return None

        if (
            index < 0
            or index >= len(
                self.candidates
            )
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

        Normally this should NOT be used after candidates
        have been stored in Bucket.

        Bucket/Backtracker should preserve candidate order
        during fallback.
        """

        if candidate in self.candidates:

            self.candidates.remove(
                candidate
            )

        return self.candidates

    def has_candidates(self):
        """
        Check whether candidates are available.
        """

        return len(
            self.candidates
        ) > 0

    # ==================================================
    # Bucket Creation
    # ==================================================

    def create_bucket(
        self,
        tx_id,
        k=None
    ):
        """
        Create a Bucket using selected candidates.

        Parameters
        ----------
        tx_id:
            Transaction identifier.

        k:
            Number of candidates to store in Bucket.
            If omitted, max_candidates is used.
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

    def summary(self):
        """
        Return information about candidates.

        Supports both dictionary and legacy tuple candidates.
        """

        summary_candidates = []

        for i, candidate in enumerate(
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
                "index": i,
                "rank": i + 1,
                "path": path,
                "score": score
            }

            # ----------------------------------
            # Preserve new candidate metadata
            # ----------------------------------

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
                    "hop_count"
                ):

                    if key in candidate:

                        item[key] = candidate[
                            key
                        ]

            summary_candidates.append(
                item
            )

        return {

            "candidate_count":
                len(self.candidates),

            "max_candidates":
                self.max_candidates,

            "candidates":
                summary_candidates
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
    """

    return Bucket(
        tx_id,
        tx_id,
        list(candidates)
    )