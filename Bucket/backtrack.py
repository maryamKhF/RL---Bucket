# Bucket/backtrack.py


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
        Index of the failed candidate.

    Returns
    -------
    candidate or None
        Next candidate.
    """

    next_index = failed_index + 1

    if next_index >= len(candidates):
        return None

    return candidates[next_index]


class Backtracker:
    """
    Handle Bucket route recovery.

    Responsible for:
    - selecting alternative candidates
    - managing retry attempts
    - recording failed candidates
    - recording failed channels
    - moving Bucket to the next candidate
    """

    def __init__(
        self,
        max_attempts=10
    ):

        self.max_attempts = max_attempts

        # Number of performed backtracking attempts
        self.attempts = 0

    # --------------------------------------------------
    # Main Backtracking
    # --------------------------------------------------

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

        failed_candidate : object
            Candidate that failed.

        failed_channel : tuple
            Channel responsible for failure.

        Returns
        -------
        candidate or None
            Next candidate or None.
        """

        # ----------------------------------------------
        # Check maximum attempts
        # ----------------------------------------------

        if self.attempts >= self.max_attempts:

            bucket.fail()

            return None

        # ----------------------------------------------
        # Current candidate
        # ----------------------------------------------

        current_candidate = bucket.current()

        current_index = bucket.current_index

        # If failed candidate was not supplied,
        # use current candidate.
        if failed_candidate is None:

            failed_candidate = current_candidate

        # ----------------------------------------------
        # Record failed candidate
        # ----------------------------------------------

        if failed_candidate is not None:

            if failed_candidate not in bucket.failed_candidates:

                bucket.failed_candidates.append(
                    failed_candidate
                )

                bucket.failed_candidates_count += 1

        # ----------------------------------------------
        # Record failed channel
        # ----------------------------------------------

        if failed_channel is not None:

            if failed_channel not in bucket.failed_channels:

                bucket.failed_channels.append(
                    failed_channel
                )

        # ----------------------------------------------
        # Register backtracking attempt
        # ----------------------------------------------

        self.attempts += 1

        # Keep Bucket attempt counter synchronized
        # with Backtracker.
        bucket.attempts += 1

        # ----------------------------------------------
        # Find next candidate
        # ----------------------------------------------

        next_candidate = choose_alternative(
            bucket.candidates,
            current_index
        )

        # ----------------------------------------------
        # No alternative candidate
        # ----------------------------------------------

        if next_candidate is None:

            bucket.fail()

            return None

        # ----------------------------------------------
        # Move Bucket to next candidate
        # ----------------------------------------------

        bucket.current_index += 1

        return next_candidate

    # --------------------------------------------------
    # Remove Failed Candidate
    # --------------------------------------------------

    def remove_failed_candidate(
        self,
        bucket,
        failed_candidate
    ):
        """
        Record a failed candidate.

        The candidate is not physically removed from
        Bucket.candidates so its original rank remains
        unchanged.
        """

        if failed_candidate is None:

            return bucket.candidates

        if failed_candidate not in bucket.failed_candidates:

            bucket.failed_candidates.append(
                failed_candidate
            )

            bucket.failed_candidates_count += 1

        return bucket.candidates

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self):

        self.attempts = 0

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    def status(self):

        return {
            "attempts": self.attempts,
            "max_attempts": self.max_attempts
        }