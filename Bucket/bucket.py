# Bucket/bucket.py

from dataclasses import dataclass, field


@dataclass
class Bucket:
    """
    Bucket container for transaction routing candidates.

    Stores candidate paths and manages
    current routing attempt.
    """

    bucket_id: int
    transaction_id: int

    candidates: list = field(default_factory=list)

    current_index: int = 0

    attempts: int = 0

    status: str = "active"

    selected_candidate: object = None


    # ---------------------------------------------
    # Current Candidate
    # ---------------------------------------------

    def current(self):
        """
        Return current candidate path.
        """

        if not self.candidates:
            return None

        if self.current_index >= len(self.candidates):
            return None

        return self.candidates[self.current_index]


    # ---------------------------------------------
    # Candidate Movement
    # ---------------------------------------------

    def next_candidate(self):
        """
        Move to next candidate.
        """

        self.current_index += 1

        return self.current()


    def backtrack(self):
        """
        Switch to alternative candidate.
        """

        self.attempts += 1

        return self.next_candidate()


    # ---------------------------------------------
    # Status
    # ---------------------------------------------

    def success(self, candidate):
        """
        Mark successful routing.
        """

        self.status = "completed"

        self.selected_candidate = candidate


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


    # ---------------------------------------------
    # Information
    # ---------------------------------------------

    def info(self):

        return {
            "bucket_id": self.bucket_id,
            "transaction_id": self.transaction_id,
            "current_index": self.current_index,
            "attempts": self.attempts,
            "status": self.status
        }