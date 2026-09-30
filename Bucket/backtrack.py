"""
Bucket/backtrack.py

Bucket candidate backtracking logic.

Responsibilities
----------------
- Select alternative routing candidates
- Manage retry attempts
- Record failed candidates
- Record failed channels
- Move Bucket to the next candidate
- Respect maximum retry attempts
- Preserve original candidate ordering/rank

Important
---------
This module performs candidate-level backtracking.

It does NOT perform:
- payment execution
- stochastic failure generation
- full network rerouting
- PPO/RL decisions

Actual payment failure is handled by:
    PaymentSimulator
    FailureModel

Partial route recovery after a failed channel is handled
by the higher-level PartialBacktracker module.
"""


# ============================================================
# Alternative Candidate Selection
# ============================================================

def choose_alternative(
    candidates,
    failed_index
):
    """
    Select the next candidate after a failed candidate.

    Parameters
    ----------
    candidates : list
        Available routing candidates.

    failed_index : int
        Zero-based index of the failed candidate.

    Returns
    -------
    candidate or None
        Next candidate if available.
    """

    if candidates is None:
        return None

    if not isinstance(
        failed_index,
        int
    ):
        return None

    next_index = failed_index + 1

    if next_index < 0:
        return None

    if next_index >= len(candidates):
        return None

    return candidates[next_index]


# ============================================================
# Backtracker
# ============================================================

class Backtracker:
    """
    Handle candidate-level Bucket recovery.

    The Backtracker moves the Bucket from a failed candidate
    to the next available candidate.

    It preserves the original candidate list, therefore
    candidate rank remains stable.

    Example
    -------
    Candidate ranks:

        1 -> failed
        2 -> failed
        3 -> selected

    The selected candidate keeps:

        selected_candidate_index = 2
        selected_candidate_rank  = 3
    """

    def __init__(
        self,
        max_attempts=10
    ):
        """
        Parameters
        ----------
        max_attempts : int
            Maximum number of backtracking operations.
        """

        try:
            max_attempts = int(
                max_attempts
            )
        except (
            TypeError,
            ValueError
        ):
            max_attempts = 10

        if max_attempts < 0:
            max_attempts = 0

        self.max_attempts = max_attempts

        # Number of performed backtracking operations.
        self.attempts = 0

    # ========================================================
    # Main Backtracking
    # ========================================================

    def backtrack(
        self,
        bucket,
        failed_candidate=None,
        failed_channel=None
    ):
        """
        Move Bucket to the next candidate after failure.

        Parameters
        ----------
        bucket : Bucket
            Bucket containing routing candidates.

        failed_candidate : object, optional
            Candidate that failed.

        failed_channel : tuple, optional
            Channel responsible for failure.

        Returns
        -------
        candidate or None
            Next candidate if available.

        Notes
        -----
        This method updates Bucket state exactly once.

        It does not call bucket.backtrack() because that
        would duplicate the attempt and failure counters.
        """

        # ----------------------------------------------------
        # Validate Bucket
        # ----------------------------------------------------

        if bucket is None:
            return None

        # ----------------------------------------------------
        # Bucket already terminated
        # ----------------------------------------------------

        if bucket.finished():
            return bucket.selected_candidate

        # ----------------------------------------------------
        # Maximum backtracking attempts
        # ----------------------------------------------------

        if self.attempts >= self.max_attempts:

            bucket.fail()

            return None

        # ----------------------------------------------------
        # Current candidate
        # ----------------------------------------------------

        current_candidate = bucket.current()

        current_index = bucket.current_index

        # If failed candidate is not explicitly supplied,
        # use the current Bucket candidate.
        if failed_candidate is None:
            failed_candidate = current_candidate

        # ----------------------------------------------------
        # Record failed candidate
        # ----------------------------------------------------

        self._record_failed_candidate(
            bucket,
            failed_candidate
        )

        # ----------------------------------------------------
        # Record failed channel
        # ----------------------------------------------------

        self._record_failed_channel(
            bucket,
            failed_channel
        )

        # ----------------------------------------------------
        # Register one backtracking operation
        # ----------------------------------------------------

        self.attempts += 1

        # Synchronize Bucket attempt counter.
        #
        # This represents a transition from one candidate
        # to another after failure.
        bucket.attempts += 1

        # ----------------------------------------------------
        # Find next candidate
        # ----------------------------------------------------

        next_candidate = choose_alternative(
            bucket.candidates,
            current_index
        )

        # ----------------------------------------------------
        # No alternative candidate
        # ----------------------------------------------------

        if next_candidate is None:

            bucket.fail()

            return None

        # ----------------------------------------------------
        # Move Bucket to next candidate
        # ----------------------------------------------------

        bucket.current_index = (
            current_index + 1
        )

        return next_candidate

    # ========================================================
    # Record Failed Candidate
    # ========================================================

    def _record_failed_candidate(
        self,
        bucket,
        failed_candidate
    ):
        """
        Record a failed candidate exactly once.
        """

        if failed_candidate is None:
            return

        if failed_candidate not in (
            bucket.failed_candidates
        ):

            bucket.failed_candidates.append(
                failed_candidate
            )

            bucket.failed_candidates_count += 1

    # ========================================================
    # Record Failed Channel
    # ========================================================

    def _record_failed_channel(
        self,
        bucket,
        failed_channel
    ):
        """
        Record a failed channel exactly once.
        """

        if failed_channel is None:
            return

        if failed_channel not in (
            bucket.failed_channels
        ):

            bucket.failed_channels.append(
                failed_channel
            )

    # ========================================================
    # Remove / Record Failed Candidate
    # ========================================================

    def remove_failed_candidate(
        self,
        bucket,
        failed_candidate
    ):
        """
        Record a failed candidate without physically removing
        it from Bucket.candidates.

        Keeping the candidate in the original list preserves
        its original rank.
        """

        if bucket is None:
            return []

        if failed_candidate is None:
            return bucket.candidates

        self._record_failed_candidate(
            bucket,
            failed_candidate
        )

        return bucket.candidates

    # ========================================================
    # Failed Candidate Check
    # ========================================================

    def is_failed(
        self,
        bucket,
        candidate
    ):
        """
        Check whether a candidate has already failed.
        """

        if bucket is None:
            return False

        if candidate is None:
            return False

        return candidate in (
            bucket.failed_candidates
        )

    # ========================================================
    # Failed Channel Check
    # ========================================================

    def is_failed_channel(
        self,
        bucket,
        channel
    ):
        """
        Check whether a channel has already been recorded
        as failed.
        """

        if bucket is None:
            return False

        if channel is None:
            return False

        return channel in (
            bucket.failed_channels
        )

    # ========================================================
    # Next Candidate
    # ========================================================

    def next_candidate(
        self,
        bucket
    ):
        """
        Return the next candidate without recording a failure.

        This method is useful when candidate movement is
        required independently of a failure event.
        """

        if bucket is None:
            return None

        if bucket.finished():
            return bucket.selected_candidate

        current_index = bucket.current_index

        next_candidate = choose_alternative(
            bucket.candidates,
            current_index
        )

        if next_candidate is None:

            bucket.fail()

            return None

        bucket.current_index = (
            current_index + 1
        )

        return next_candidate

    # ========================================================
    # Has Alternative
    # ========================================================

    def has_alternative(
        self,
        bucket
    ):
        """
        Check whether another candidate exists.
        """

        if bucket is None:
            return False

        if bucket.finished():
            return False

        next_index = (
            bucket.current_index + 1
        )

        return (
            next_index < len(
                bucket.candidates
            )
        )

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):
        """
        Reset Backtracker state.
        """

        self.attempts = 0

    # ========================================================
    # Information
    # ========================================================

    def status(self):
        """
        Return Backtracker status.
        """

        return {
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "remaining_attempts": max(
                0,
                self.max_attempts - self.attempts
            )
        }