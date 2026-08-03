# Bucket/backtrack.py


def choose_alternative(
    candidates,
    failed_index
):
    """
    Select next candidate after failed path.

    Parameters
    ----------
    candidates : list
        Available routing candidates.

    failed_index : int
        Index of failed candidate.


    Returns
    -------
    candidate or None
    """


    for i in range(
        failed_index + 1,
        len(candidates)
    ):
        return candidates[i]


    return None



class Backtracker:
    """
    Handle Bucket route recovery.

    Responsible for:
    - selecting alternative paths
    - managing retry attempts
    - removing failed candidates
    """



    def __init__(
        self,
        max_attempts=10
    ):

        self.max_attempts = max_attempts

        self.attempts = 0



    # --------------------------------------------------
    # Main Backtracking
    # --------------------------------------------------

    def backtrack(
        self,
        bucket
    ):
        """
        Move Bucket to next candidate.

        Returns
        -------
        next candidate or None
        """


        if self.attempts >= self.max_attempts:

            bucket.fail()

            return None



        current_index = bucket.current_index


        next_candidate = choose_alternative(
            bucket.candidates,
            current_index
        )


        self.attempts += 1



        if next_candidate is None:

            bucket.fail()

            return None



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
        Delete failed route from candidates.
        """


        if failed_candidate in bucket.candidates:

            bucket.candidates.remove(
                failed_candidate
            )


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

            "attempts":
                self.attempts,

            "max_attempts":
                self.max_attempts

        }