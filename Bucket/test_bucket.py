# Bucket/test_bucket.py


from Bucket.bucket import Bucket
from Bucket.candidate_manager import (
    CandidateManager,
    make_bucket
)
from Bucket.backtrack import Backtracker



# --------------------------------------------------
# Create Fake Candidates
# --------------------------------------------------

def create_test_candidates():

    candidate_1 = (

        ["Alice", "Bob", "Carol"],

        [
            ("Alice", "Bob", 0),
            ("Bob", "Carol", 0)
        ],

        10.5
    )


    candidate_2 = (

        ["Alice", "Dave", "Carol"],

        [
            ("Alice", "Dave", 0),
            ("Dave", "Carol", 0)
        ],

        15.2
    )


    candidate_3 = (

        ["Alice", "Eve", "Carol"],

        [
            ("Alice", "Eve", 0),
            ("Eve", "Carol", 0)
        ],

        20.1
    )


    return [
        candidate_1,
        candidate_2,
        candidate_3
    ]



# --------------------------------------------------
# Test Candidate Manager
# --------------------------------------------------

def test_candidate_manager():

    print("\n===== Candidate Manager Test =====")


    candidates = create_test_candidates()


    manager = CandidateManager(
        candidates=candidates
    )


    print(
        "Initial candidates:",
        len(manager.candidates)
    )


    manager.rank_candidates()


    print(
        "Best candidate:",
        manager.get_best_candidates()[0][0]
    )



# --------------------------------------------------
# Test Bucket Creation
# --------------------------------------------------

def test_bucket_creation():

    print("\n===== Bucket Creation Test =====")


    candidates = create_test_candidates()


    bucket = make_bucket(
        tx_id=1001,
        candidates=candidates
    )


    print(
        "Bucket:",
        bucket
    )


    print(
        "Current candidate:",
        bucket.current()
    )



# --------------------------------------------------
# Test Backtracking
# --------------------------------------------------

def test_backtracking():

    print("\n===== Backtracking Test =====")


    candidates = create_test_candidates()


    bucket = Bucket(
        bucket_id=1,
        transaction_id=2001,
        candidates=candidates
    )


    print(
        "Current path:",
        bucket.current()[0]
    )


    backtracker = Backtracker()



    # simulate failure

    print(
        "\nPath failed..."
    )


    next_candidate = backtracker.backtrack(
        bucket
    )


    print(
        "New candidate:",
        next_candidate[0]
        if next_candidate
        else None
    )


    print(
        "Bucket index:",
        bucket.current_index
    )



# --------------------------------------------------
# Run Tests
# --------------------------------------------------

if __name__ == "__main__":


    test_candidate_manager()

    test_bucket_creation()

    test_backtracking()


    print(
        "\n===== ALL TESTS COMPLETED ====="
    )