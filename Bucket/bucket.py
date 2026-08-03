"""
Bucket/bucket.py

Bucket container and execution logic.

Responsibilities:

- Store routing candidates
- Manage candidate attempts
- Handle backtracking
- Execute candidate selection
"""


from dataclasses import dataclass, field





@dataclass
class Bucket:
    """
    Bucket container for transaction routing candidates.

    Stores candidate paths and manages
    routing attempts.
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





    def backtrack(self):
        """
        Move to alternative candidate.
        """


        self.attempts += 1


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





    # ==================================================
    # Information
    # ==================================================


    def info(self):

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
                self.status

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



        if candidate is None:


            bucket.fail()


            break




        attempts += 1


        bucket.attempts += 1




        path, edges, score = candidate



        valid = True




        # -----------------------------------------
        # Validate candidate path
        # -----------------------------------------


        for u, v, k in edges:



            channel = G.edges[u, v, k]



            if not channel.get(
                "available",
                True
            ):

                valid = False

                break




            if channel.get(
                "balance_uv",
                0
            ) < amount:


                valid = False

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
        # Backtrack
        # -----------------------------------------


        bucket.backtrack()





    return (

        False,

        None,

        attempts

    )