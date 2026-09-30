"""
Bucket/backtrack.py

Bucket candidate-level backtracking logic.

Responsibilities
----------------
- Select alternative routing candidates
- Manage candidate-level backtracking operations
- Record failed candidates
- Record failed channels
- Move Bucket to the next candidate
- Respect maximum backtracking operations
- Preserve original candidate ordering/rank

Important
---------
This module performs CANDIDATE-LEVEL backtracking.

It does NOT perform:
- payment execution
- stochastic failure generation
- full network rerouting
- PPO/RL decisions
- payment-attempt counting

Actual payment failure is handled by:
    PaymentSimulator
    FailureModel

Partial route recovery after a failed channel is handled
by:
    Simulation.backtrack.PartialBacktracker

Counter semantics
-----------------
Bucket.attempts
    Number of ACTUAL PAYMENT ATTEMPTS.

Backtracker.attempts
    Number of CANDIDATE-LEVEL BACKTRACKING OPERATIONS.

Therefore this module MUST NOT modify:

    bucket.attempts

Example
-------
Payment attempt #1
    |
    +--> Candidate 1 fails
              |
              v
       Backtracker operation #1
              |
              v
       Candidate 2 selected
              |
              v
       Payment attempt #2
              |
              v
       Candidate 2 succeeds

Final state:

    bucket.attempts      = 2
    backtracker.attempts = 1

Candidate ordering is never changed. Failed candidates
remain inside Bucket.candidates so their original rank is
preserved.
"""


# ============================================================
# Alternative Candidate Selection
# ============================================================

def choose_alternative(
    candidates,
    failed_index
):
    """
    Select the next candidate after failed_index.

    This function does not modify the candidate list.

    Parameters
    ----------
    candidates : list
        Ordered candidate list.

    failed_index : int
        Zero-based index of the failed candidate.

    Returns
    -------
    candidate or None
        Immediate next candidate if it exists.
    """

    if not isinstance(
        candidates,
        (list, tuple)
    ):
        return None

    # bool is an int subclass, but is not a valid index here.
    if isinstance(
        failed_index,
        bool
    ):
        return None

    if not isinstance(
        failed_index,
        int
    ):
        return None

    if failed_index < 0:
        return None

    next_index = failed_index + 1

    if next_index >= len(candidates):
        return None

    return candidates[next_index]


# ============================================================
# Backtracker
# ============================================================

class Backtracker:
    """
    Candidate-level Bucket backtracking.

    The Backtracker moves the Bucket from a failed candidate
    to the next non-failed candidate.

    It preserves the original candidate list and therefore
    preserves candidate rank.

    Example
    -------
    Candidate ranks:

        1 -> failed
        2 -> failed
        3 -> selected

    The selected candidate keeps:

        current_index = 2
        rank          = 3

    Counter semantics
    -----------------
    self.attempts
        Number of backtracking operations.

    bucket.attempts
        Number of actual payment attempts.

    This class increments ONLY self.attempts.
    """

    def __init__(
        self,
        max_attempts=10
    ):
        """
        Parameters
        ----------
        max_attempts : int
            Maximum number of candidate-level backtracking
            operations.
        """

        if isinstance(
            max_attempts,
            bool
        ):
            max_attempts = 10

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

        # Number of candidate-level backtracking operations.
        self.attempts = 0

    # ========================================================
    # Main Backtracking
    # ========================================================

    def backtrack(
        self,
        bucket,
        failed_candidate=None,
        failed_channel=None,
        failed_index=None,
        reason=None
    ):
        """
        Move Bucket from a failed candidate to the next
        available candidate.

        Parameters
        ----------
        bucket : Bucket
            Bucket containing ordered candidates.

        failed_candidate : object, optional
            Candidate that actually failed.

        failed_channel : tuple, optional
            Channel responsible for the payment failure.

        failed_index : int, optional
            Explicit index of the failed candidate.

        reason : str, optional
            Failure reason for diagnostic bookkeeping.

        Returns
        -------
        candidate or None
            Next valid candidate.

        Important
        ---------
        This method does NOT increment bucket.attempts.

        bucket.attempts belongs exclusively to actual payment
        execution.
        """

        # ----------------------------------------------------
        # 1. Validate Bucket
        # ----------------------------------------------------

        if bucket is None:
            return None

        candidates = getattr(
            bucket,
            "candidates",
            None
        )

        if not isinstance(
            candidates,
            (list, tuple)
        ):
            return None

        if len(candidates) == 0:

            self._fail_bucket(
                bucket
            )

            return None

        # ----------------------------------------------------
        # 2. Already terminated
        # ----------------------------------------------------

        if self._bucket_finished(
            bucket
        ):

            return getattr(
                bucket,
                "selected_candidate",
                None
            )

        # ----------------------------------------------------
        # 3. Backtracking operation limit
        # ----------------------------------------------------

        if self.attempts >= self.max_attempts:

            self._fail_bucket(
                bucket
            )

            return None

        # ----------------------------------------------------
        # 4. Resolve current index
        # ----------------------------------------------------

        current_index = self._get_current_index(
            bucket
        )

        if current_index is None:

            self._fail_bucket(
                bucket
            )

            return None

        # ----------------------------------------------------
        # 5. Resolve failed candidate
        # ----------------------------------------------------

        if failed_index is not None:

            failed_index = self._normalize_index(
                failed_index
            )

            if failed_index is None:

                return None

            if failed_index >= len(candidates):

                return None

            resolved_failed_candidate = (
                candidates[failed_index]
            )

        else:

            resolved_failed_candidate = (
                failed_candidate
            )

            if resolved_failed_candidate is None:

                resolved_failed_candidate = (
                    candidates[current_index]
                )

            failed_index = self._find_candidate_index(
                candidates,
                resolved_failed_candidate,
                preferred_index=current_index
            )

            if failed_index is None:

                # If the supplied failed candidate is not in
                # the candidate list, fall back to the current
                # candidate because the Bucket state is
                # authoritative.
                failed_index = current_index

                resolved_failed_candidate = (
                    candidates[current_index]
                )

        # ----------------------------------------------------
        # 6. Record failure
        # ----------------------------------------------------

        self._record_failed_candidate(
            bucket,
            resolved_failed_candidate
        )

        self._record_failed_channel(
            bucket,
            failed_channel
        )

        self._record_failure_reason(
            bucket=bucket,
            failed_index=failed_index,
            failed_candidate=resolved_failed_candidate,
            failed_channel=failed_channel,
            reason=reason
        )

        # ----------------------------------------------------
        # 7. Register ONE candidate-level backtracking op
        # ----------------------------------------------------

        self.attempts += 1

        # IMPORTANT:
        #
        # DO NOT DO:
        #
        # bucket.attempts += 1
        #
        # bucket.attempts belongs to actual payment attempts.

        # ----------------------------------------------------
        # 8. Find next non-failed candidate
        # ----------------------------------------------------

        next_index = self._find_next_available_index(
            bucket=bucket,
            start_index=failed_index + 1
        )

        # ----------------------------------------------------
        # 9. No candidate remains
        # ----------------------------------------------------

        if next_index is None:

            self._fail_bucket(
                bucket
            )

            return None

        # ----------------------------------------------------
        # 10. Move Bucket
        # ----------------------------------------------------

        bucket.current_index = next_index

        return candidates[next_index]

    # ========================================================
    # Find Next Available Candidate
    # ========================================================

    def _find_next_available_index(
        self,
        bucket,
        start_index
    ):
        """
        Find the first candidate at or after start_index that
        has not already been recorded as failed.
        """

        candidates = getattr(
            bucket,
            "candidates",
            None
        )

        if not isinstance(
            candidates,
            (list, tuple)
        ):
            return None

        if isinstance(
            start_index,
            bool
        ):
            return None

        if not isinstance(
            start_index,
            int
        ):
            return None

        if start_index < 0:
            start_index = 0

        failed_candidates = getattr(
            bucket,
            "failed_candidates",
            []
        )

        for index in range(
            start_index,
            len(candidates)
        ):

            candidate = candidates[index]

            if self._candidate_in_failed_registry(
                candidate,
                failed_candidates
            ):
                continue

            return index

        return None

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

        The candidate is NOT removed from Bucket.candidates.
        """

        if failed_candidate is None:
            return

        failed_candidates = getattr(
            bucket,
            "failed_candidates",
            None
        )

        if failed_candidates is None:

            failed_candidates = []

            bucket.failed_candidates = (
                failed_candidates
            )

        if self._candidate_in_failed_registry(
            failed_candidate,
            failed_candidates
        ):
            return

        failed_candidates.append(
            failed_candidate
        )

        # Keep compatibility with Bucket implementations that
        # expose failed_candidates_count.
        if hasattr(
            bucket,
            "failed_candidates_count"
        ):

            bucket.failed_candidates_count = len(
                failed_candidates
            )

    # ========================================================
    # Failed Candidate Registry
    # ========================================================

    @staticmethod
    def _candidate_in_failed_registry(
        candidate,
        failed_candidates
    ):
        """
        Determine whether candidate is already registered
        as failed.

        Equality is preferred, with identity fallback.
        """

        if not failed_candidates:
            return False

        for failed in failed_candidates:

            if failed is candidate:
                return True

            try:

                if failed == candidate:
                    return True

            except Exception:
                pass

        return False

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

        failed_channels = getattr(
            bucket,
            "failed_channels",
            None
        )

        if failed_channels is None:

            failed_channels = []

            bucket.failed_channels = (
                failed_channels
            )

        if not self._channel_in_registry(
            failed_channel,
            failed_channels
        ):

            failed_channels.append(
                failed_channel
            )

    # ========================================================
    # Record Failure Reason
    # ========================================================

    def _record_failure_reason(
        self,
        bucket,
        failed_index,
        failed_candidate,
        failed_channel,
        reason
    ):
        """
        Record diagnostic information about the failure.

        This is optional and does not affect routing logic.
        """

        if not hasattr(
            bucket,
            "failure_history"
        ):
            return

        history = getattr(
            bucket,
            "failure_history"
        )

        if history is None:
            return

        record = {
            "candidate_index": failed_index,
            "candidate_rank": failed_index + 1
                if failed_index is not None
                else None,
            "failed_candidate": failed_candidate,
            "failed_channel": failed_channel,
            "reason": reason
        }

        try:
            history.append(
                record
            )
        except Exception:
            pass

    # ========================================================
    # Failed Channel Registry
    # ========================================================

    @staticmethod
    def _channel_in_registry(
        channel,
        failed_channels
    ):
        """
        Check whether a channel is already recorded.

        Exact tuple equality is preferred.
        """

        if not failed_channels:
            return False

        for failed in failed_channels:

            if failed is channel:
                return True

            try:

                if failed == channel:
                    return True

            except Exception:
                pass

        return False

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

        Candidate rank therefore remains stable.
        """

        if bucket is None:
            return []

        candidates = getattr(
            bucket,
            "candidates",
            []
        )

        if failed_candidate is None:
            return candidates

        self._record_failed_candidate(
            bucket,
            failed_candidate
        )

        return candidates

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

        failed_candidates = getattr(
            bucket,
            "failed_candidates",
            []
        )

        return self._candidate_in_failed_registry(
            candidate,
            failed_candidates
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

        failed_channels = getattr(
            bucket,
            "failed_channels",
            []
        )

        return self._channel_in_registry(
            channel,
            failed_channels
        )

    # ========================================================
    # Next Candidate
    # ========================================================

    def next_candidate(
        self,
        bucket
    ):
        """
        Move to the next non-failed candidate without recording
        a failure.

        This method does NOT increment either:

            bucket.attempts
            self.attempts

        because no payment failure/backtracking operation is
        being registered here.
        """

        if bucket is None:
            return None

        if self._bucket_finished(
            bucket
        ):

            return getattr(
                bucket,
                "selected_candidate",
                None
            )

        current_index = self._get_current_index(
            bucket
        )

        if current_index is None:

            self._fail_bucket(
                bucket
            )

            return None

        next_index = self._find_next_available_index(
            bucket=bucket,
            start_index=current_index + 1
        )

        if next_index is None:

            self._fail_bucket(
                bucket
            )

            return None

        bucket.current_index = next_index

        return bucket.candidates[next_index]

    # ========================================================
    # Has Alternative
    # ========================================================

    def has_alternative(
        self,
        bucket
    ):
        """
        Return True when at least one non-failed candidate
        exists after the current candidate.
        """

        if bucket is None:
            return False

        if self._bucket_finished(
            bucket
        ):
            return False

        current_index = self._get_current_index(
            bucket
        )

        if current_index is None:
            return False

        return (
            self._find_next_available_index(
                bucket=bucket,
                start_index=current_index + 1
            )
            is not None
        )

    # ========================================================
    # Candidate Index
    # ========================================================

    @staticmethod
    def _find_candidate_index(
        candidates,
        candidate,
        preferred_index=None
    ):
        """
        Find candidate index while preserving object identity
        when possible.
        """

        if not candidates:
            return None

        # First try the preferred position.
        if (
            preferred_index is not None
            and
            isinstance(preferred_index, int)
            and
            not isinstance(preferred_index, bool)
            and
            0 <= preferred_index < len(candidates)
        ):

            preferred = candidates[
                preferred_index
            ]

            if preferred is candidate:
                return preferred_index

            try:

                if preferred == candidate:
                    return preferred_index

            except Exception:
                pass

        # Identity search.
        for index, item in enumerate(
            candidates
        ):

            if item is candidate:
                return index

        # Equality search.
        for index, item in enumerate(
            candidates
        ):

            try:

                if item == candidate:
                    return index

            except Exception:
                pass

        return None

    # ========================================================
    # Current Index
    # ========================================================

    @staticmethod
    def _get_current_index(
        bucket
    ):
        """
        Safely obtain Bucket.current_index.
        """

        current_index = getattr(
            bucket,
            "current_index",
            None
        )

        if isinstance(
            current_index,
            bool
        ):
            return None

        if not isinstance(
            current_index,
            int
        ):
            return None

        candidates = getattr(
            bucket,
            "candidates",
            None
        )

        if not isinstance(
            candidates,
            (list, tuple)
        ):
            return None

        if not (
            0 <= current_index < len(candidates)
        ):
            return None

        return current_index

    # ========================================================
    # Normalize Index
    # ========================================================

    @staticmethod
    def _normalize_index(
        index
    ):
        """
        Validate a candidate index.
        """

        if isinstance(
            index,
            bool
        ):
            return None

        if not isinstance(
            index,
            int
        ):
            return None

        if index < 0:
            return None

        return index

    # ========================================================
    # Bucket Finished
    # ========================================================

    @staticmethod
    def _bucket_finished(
        bucket
    ):
        """
        Safely query Bucket.finished().
        """

        method = getattr(
            bucket,
            "finished",
            None
        )

        if callable(method):

            try:
                return bool(
                    method()
                )
            except Exception:
                return False

        # Fallback for compatible Bucket implementations.
        status = getattr(
            bucket,
            "status",
            None
        )

        return status in {
            "completed",
            "failed",
            "finished"
        }

    # ========================================================
    # Fail Bucket
    # ========================================================

    @staticmethod
    def _fail_bucket(
        bucket
    ):
        """
        Mark Bucket as failed without changing payment
        attempt counters.
        """

        method = getattr(
            bucket,
            "fail",
            None
        )

        if callable(method):

            try:
                method()
            except Exception:
                pass

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):
        """
        Reset Backtracker state.

        This resets ONLY candidate-level backtracking count.

        It does not modify any Bucket state.
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


# ============================================================
# Backward-compatible alias
# ============================================================

Backtrack = Backtracker


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print("BUCKET BACKTRACKER MODULE TEST")
    print("=" * 72)

    # --------------------------------------------------------
    # Minimal Bucket test double
    # --------------------------------------------------------

    class TestBucket:

        def __init__(self):

            self.candidates = [
                {
                    "path": ["A", "B", "C"],
                    "rank": 1
                },
                {
                    "path": ["A", "D", "C"],
                    "rank": 2
                },
                {
                    "path": ["A", "E", "C"],
                    "rank": 3
                }
            ]

            self.current_index = 0

            self.selected_candidate = (
                self.candidates[0]
            )

            self.failed_candidates = []

            self.failed_candidates_count = 0

            self.failed_channels = []

            self.failure_history = []

            self.attempts = 0

            self.status = "active"

        def current(self):

            return self.candidates[
                self.current_index
            ]

        def finished(self):

            return self.status in {
                "completed",
                "failed"
            }

        def fail(self):

            self.status = "failed"

    bucket = TestBucket()

    backtracker = Backtracker(
        max_attempts=5
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    assert bucket.attempts == 0
    assert backtracker.attempts == 0
    assert bucket.current_index == 0

    # --------------------------------------------------------
    # Simulate actual Payment Attempt #1
    # --------------------------------------------------------

    bucket.attempts += 1

    failed_candidate = bucket.current()

    failed_channel = (
        "B",
        "C",
        0
    )

    # --------------------------------------------------------
    # Backtrack to Candidate #2
    # --------------------------------------------------------

    next_candidate = backtracker.backtrack(
        bucket=bucket,
        failed_candidate=failed_candidate,
        failed_channel=failed_channel,
        failed_index=0,
        reason="channel_failure"
    )

    assert next_candidate is bucket.candidates[1]

    assert bucket.current_index == 1

    assert bucket.failed_candidates_count == 1

    assert bucket.failed_candidates == [
        failed_candidate
    ]

    assert bucket.failed_channels == [
        failed_channel
    ]

    assert backtracker.attempts == 1

    # IMPORTANT:
    # Backtracking must NOT increment payment attempts.
    assert bucket.attempts == 1

    # --------------------------------------------------------
    # Simulate actual Payment Attempt #2
    # --------------------------------------------------------

    bucket.attempts += 1

    # --------------------------------------------------------
    # Candidate #2 fails
    # --------------------------------------------------------

    failed_candidate_2 = bucket.current()

    next_candidate = backtracker.backtrack(
        bucket=bucket,
        failed_candidate=failed_candidate_2,
        failed_channel=("D", "C", 0),
        failed_index=1,
        reason="channel_failure"
    )

    assert next_candidate is bucket.candidates[2]

    assert bucket.current_index == 2

    assert bucket.failed_candidates_count == 2

    assert backtracker.attempts == 2

    # Still only actual payment attempts.
    assert bucket.attempts == 2

    # --------------------------------------------------------
    # Simulate actual Payment Attempt #3
    # --------------------------------------------------------

    bucket.attempts += 1

    # --------------------------------------------------------
    # Candidate #3 fails -> no alternative
    # --------------------------------------------------------

    failed_candidate_3 = bucket.current()

    next_candidate = backtracker.backtrack(
        bucket=bucket,
        failed_candidate=failed_candidate_3,
        failed_channel=("E", "C", 0),
        failed_index=2,
        reason="channel_failure"
    )

    assert next_candidate is None

    assert bucket.status == "failed"

    assert bucket.failed_candidates_count == 3

    assert backtracker.attempts == 3

    assert bucket.attempts == 3

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    print()
    print("Bucket attempts       :", bucket.attempts)
    print("Backtracker attempts  :", backtracker.attempts)
    print(
        "Failed candidates     :",
        bucket.failed_candidates_count
    )
    print(
        "Failed channels       :",
        len(bucket.failed_channels)
    )
    print(
        "Final Bucket status   :",
        bucket.status
    )
    print(
        "Current candidate idx :",
        bucket.current_index
    )

    print()
    print("=" * 72)
    print("BUCKET BACKTRACKER TEST : SUCCESS")
    print("=" * 72)