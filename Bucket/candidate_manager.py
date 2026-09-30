# Bucket/candidate_manager.py

from .bucket import Bucket


class CandidateManager:
    """
    Manage routing candidates before they are stored in Bucket.

    Responsibilities:
    - store candidate paths
    - filter invalid candidates
    - rank candidates
    - select a limited number of candidates
    - create Bucket object

    Candidate format:

        (
            path,
            edges,
            score
        )
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

        self.max_candidates = max_candidates

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

        self.candidates.append(
            candidate
        )

    def remove_candidate(
        self,
        candidate
    ):
        """
        Remove a candidate from CandidateManager.

        This operation is intended for candidate preparation
        before Bucket creation.
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

        A candidate is invalid if at least one of
        its channels is unavailable or does not
        have sufficient capacity.
        """

        valid_candidates = []

        for candidate in self.candidates:

            path, edges, score = candidate

            valid = True

            for u, v, k in edges:

                channel = G.edges[
                    u,
                    v,
                    k
                ]

                # --------------------------------------
                # Channel availability
                # --------------------------------------

                if not channel.get(
                    "available",
                    True
                ):

                    valid = False

                    break

                # --------------------------------------
                # Channel capacity
                # --------------------------------------

                if channel.get(
                    "balance_uv",
                    0
                ) < amount:

                    valid = False

                    break

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
        Sort candidates according to path score.

        Lower score means a better candidate.
        """

        self.candidates.sort(
            key=lambda x: x[2]
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

        Parameters
        ----------
        k : int, optional
            Number of candidates to select.

        Returns
        -------
        list
            Selected candidate paths.
        """

        if k is None:

            k = self.max_candidates

        # Prevent invalid values
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
        Return candidate by index.
        """

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
        Remove a failed candidate from
        CandidateManager.

        Note:
        This method should generally NOT be used
        after the candidate has already been stored
        in Bucket, because removing candidates from
        Bucket changes their original rank.

        Backtracking should be handled by Backtracker.
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
        Create a Bucket using the selected candidates.

        Parameters
        ----------
        tx_id:
            Transaction identifier.

        k : int, optional
            Number of candidates to store in Bucket.
            If omitted, max_candidates is used.
        """

        selected_candidates = (
            self.get_best_candidates(k)
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
        """

        return {

            "candidate_count":
                len(self.candidates),

            "max_candidates":
                self.max_candidates,

            "candidates":
                [
                    {
                        "index": i,
                        "rank": i + 1,
                        "path": c[0],
                        "score": c[2]
                    }

                    for i, c in enumerate(
                        self.candidates
                    )
                ]
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