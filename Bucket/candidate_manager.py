# Bucket/candidate_manager.py

from .bucket import Bucket


class CandidateManager:
    """
    Manage routing candidates for Bucket.

    Responsibilities:
    - store candidate paths
    - filter invalid candidates
    - rank candidates
    - create Bucket object
    """


    def __init__(
        self,
        candidates=None,
        max_candidates=10
    ):

        self.candidates = (
            candidates
            if candidates
            else []
        )

        self.max_candidates = max_candidates



    # --------------------------------------------------
    # Candidate Management
    # --------------------------------------------------

    def add_candidate(self, candidate):
        """
        Add new candidate path.

        Candidate format:

        (
            path,
            edges,
            score
        )
        """

        self.candidates.append(candidate)



    def remove_candidate(self, candidate):

        if candidate in self.candidates:
            self.candidates.remove(candidate)



    # --------------------------------------------------
    # Filtering
    # --------------------------------------------------

    def filter_candidates(
        self,
        G,
        amount
    ):
        """
        Remove candidates that cannot
        support the transaction.
        """

        valid_candidates = []


        for candidate in self.candidates:

            path, edges, score = candidate

            valid = True


            for u, v, k in edges:

                channel = G.edges[u, v, k]


                # channel availability

                if not channel.get(
                    "available",
                    True
                ):
                    valid = False
                    break


                # balance check

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



    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    def rank_candidates(self):
        """
        Sort candidates by path score.

        Lower score means better path.
        """

        self.candidates.sort(
            key=lambda x: x[2]
        )


        return self.candidates



    # --------------------------------------------------
    # Selection
    # --------------------------------------------------

    def get_best_candidates(self):

        return self.candidates[
            :self.max_candidates
        ]



    def get_candidate(self, index):

        if (
            index < 0
            or index >= len(self.candidates)
        ):
            return None


        return self.candidates[index]



    def remove_failed_path(
        self,
        candidate
    ):

        if candidate in self.candidates:

            self.candidates.remove(
                candidate
            )



    def has_candidates(self):

        return len(
            self.candidates
        ) > 0



    # --------------------------------------------------
    # Bucket Creation
    # --------------------------------------------------

    def create_bucket(
        self,
        tx_id
    ):
        """
        Create Bucket using
        prepared candidates.
        """

        return Bucket(
            tx_id,
            tx_id,
            self.get_best_candidates()
        )



    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    def summary(self):

        return {

            "candidate_count":
                len(self.candidates),

            "candidates":
                [
                    {
                        "path": c[0],
                        "score": c[2]
                    }

                    for c in self.candidates
                ]

        }



# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def make_bucket(
    tx_id,
    candidates
):
    """
    Quick Bucket constructor.

    Parameters
    ----------
    tx_id:
        Transaction identifier

    candidates:
        Candidate routing paths
    """

    return Bucket(
        tx_id,
        tx_id,
        candidates
    )