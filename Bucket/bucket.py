"""
Bucket/bucket.py

Bucket container and execution logic.

Responsibilities:

- Store routing candidates
- Manage candidate attempts
- Count failed candidates
- Track failed channels
- Identify selected candidate rank
- Handle backtracking
- Execute candidate selection
"""

from dataclasses import dataclass, field


@dataclass
class Bucket:
    """
    Bucket container for transaction routing candidates.

    Stores candidate paths and manages routing attempts,
    failed candidates, failed channels, and the selected
    candidate position.
    """

    bucket_id: object

    transaction_id: object

    candidates: list = field(
        default_factory=list
    )

    current_index: int = 0

    attempts: int = 0

    status: str = "active"

    selected_candidate: object = None

    # Number of candidates that failed
    failed_candidates_count: int = 0

    # Candidate index selected successfully
    # Zero-based index
    selected_candidate_index: object = None

    # Candidate rank selected successfully
    # One-based rank
    selected_candidate_rank: object = None

    # List of failed candidates
    failed_candidates: list = field(
        default_factory=list
    )

    # List of failed channels
    failed_channels: list = field(
        default_factory=list
    )

    # ==================================================
    # Current Candidate
    # ==================================================

    def current(self):
        """
        Return current candidate.
        """

        if not self.candidates:
            return None

        if self.current_index >= len(
            self.candidates
        ):
            return None

        return self.candidates[
            self.current_index
        ]

    # ==================================================
    # Candidate Movement
    # ==================================================

    def next_candidate(self):
        """
        Move to next candidate.
        """

        self.current_index += 1

        return self.current()

    def backtrack(
        self,
        failed_candidate=None,
        failed_channel=None
    ):
        """
        Move to an alternative candidate.

        Records the failed candidate and,
        if available, the channel responsible
        for the failure.
        """

        self.attempts += 1

        self.failed_candidates_count += 1

        # Record failed candidate
        if failed_candidate is not None:

            self.failed_candidates.append(
                failed_candidate
            )

        # Record failed channel
        if failed_channel is not None:

            self.failed_channels.append(
                failed_channel
            )

        return self.next_candidate()

    # ==================================================
    # Status Management
    # ==================================================

    def mark_success(
        self,
        candidate
    ):
        """
        Mark successful routing.

        Stores both the zero-based candidate index
        and the one-based candidate rank.
        """

        self.status = "completed"

        self.selected_candidate = candidate

        self.selected_candidate_index = (
            self.current_index
        )

        self.selected_candidate_rank = (
            self.current_index + 1
        )

    def fail(self):
        """
        Mark bucket failure.
        """

        self.status = "failed"

    def finished(self):
        """
        Check bucket termination.
        """

        return self.status in [
            "completed",
            "failed"
        ]

    # ==================================================
    # Information
    # ==================================================

    def info(self):
        """
        Return complete Bucket information.
        """

        return {

            "bucket_id":
                self.bucket_id,

            "transaction_id":
                self.transaction_id,

            "candidate_count":
                len(self.candidates),

            "current_index":
                self.current_index,

            "attempts":
                self.attempts,

            "status":
                self.status,

            "selected_candidate_index":
                self.selected_candidate_index,

            "selected_candidate_rank":
                self.selected_candidate_rank,

            "failed_candidates_count":
                self.failed_candidates_count,

            "failed_candidates":
                self.failed_candidates,

            "failed_channels_count":
                len(self.failed_channels),

            "failed_channels":
                self.failed_channels,

            "selected_candidate":
                self.selected_candidate
        }


# ======================================================
# Bucket Execution Engine
# ======================================================

def execute_bucket(
    G,
    bucket,
    amount,
    rng=None
):
    """
    Execute routing candidates stored in Bucket.

    Candidate format:

        (
            path,
            edges,
            score
        )

    Parameters
    ----------
    G:
        Network graph

    bucket:
        Bucket object

    amount:
        Payment amount

    rng:
        Random generator (optional)

    Returns
    -------
    tuple:

        success,
        selected_candidate,
        attempts
    """

    attempts = 0

    while not bucket.finished():

        candidate = bucket.current()

        # -----------------------------------------
        # No candidate available
        # -----------------------------------------

        if candidate is None:

            bucket.fail()

            break

        attempts += 1

        path, edges, score = candidate

        valid = True

        failed_channel = None

        # -----------------------------------------
        # Validate candidate path
        # -----------------------------------------

        for u, v, k in edges:

            channel = G.edges[u, v, k]

            # -------------------------------------
            # Channel unavailable
            # -------------------------------------

            if not channel.get(
                "available",
                True
            ):

                valid = False

                failed_channel = (
                    u,
                    v,
                    k
                )

                break

            # -------------------------------------
            # Insufficient channel capacity
            # -------------------------------------

            if channel.get(
                "balance_uv",
                0
            ) < amount:

                valid = False

                failed_channel = (
                    u,
                    v,
                    k
                )

                break

        # -----------------------------------------
        # Candidate accepted
        # -----------------------------------------

        if valid:

            bucket.mark_success(
                candidate
            )

            return (
                True,
                candidate,
                attempts
            )

        # -----------------------------------------
        # Candidate failed
        # -----------------------------------------

        bucket.backtrack(
            failed_candidate=candidate,
            failed_channel=failed_channel
        )

    # ---------------------------------------------
    # All candidates failed
    # ---------------------------------------------

    return (
        False,
        None,
        attempts
    )