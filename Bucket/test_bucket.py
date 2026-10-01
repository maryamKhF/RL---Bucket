"""
Bucket/test_bucket.py

Deterministic validation suite for Bucket.

Coverage
--------
1. Bucket initialization
2. Bucket state validation
3. Empty candidate handling
4. Current candidate
5. Candidate progression
6. Candidate exhaustion
7. Exact MultiDiGraph channel key
8. Parallel-channel protection
9. Path/edge mismatch
10. Missing channel
11. Channel availability
12. Strict boolean validation
13. Known directional liquidity
14. Insufficient liquidity
15. Unknown liquidity
16. Invalid liquidity
17. Candidate selection
18. Selection does not increment attempts
19. Actual payment attempt tracking
20. Backtracking does not increment attempts
21. Failed candidate tracking
22. Duplicate failed candidate protection
23. Failed channel tracking
24. Duplicate failed channel protection
25. Failure reason tracking
26. Alternative candidate selection
27. Successful candidate index
28. Successful candidate rank
29. Completed bucket protection
30. Failed bucket protection
31. Exhausted bucket
32. Deterministic lifecycle
33. MultiDiGraph channel identity
34. Top-K -> Bucket candidate compatibility
35. Edge-dictionary compatibility
36. Candidate ordering preservation
37. Unknown liquidity is not zero
38. Capacity is not directional liquidity
39. Invalid amount handling
40. Full failure -> backtracking lifecycle
"""

import math
import networkx as nx

from Bucket.bucket import (
    Bucket,
    execute_bucket,
    validate_candidate,
    _normalize_edge,
    _get_channel,
    _get_known_liquidity,
    _liquidity_is_sufficient,
)


# ============================================================
# Helpers
# ============================================================

def _node_data(
    available=True,
    online=True,
):
    return {
        "available": available,
        "is_online": online,
    }


def _edge_data(
    liquidity=10000,
    available=True,
    include_liquidity=True,
):
    data = {
        "available": available,
    }

    if include_liquidity:
        data["estimated_liquidity"] = liquidity

    return data


def _build_graph():
    """
    Build deterministic MultiDiGraph with:

        A -> B
          ├─ key 0
          └─ key 1

        B -> C
          └─ key 0

        C -> D
          └─ key 0

        B -> D
          └─ key 0
    """

    G = nx.MultiDiGraph()

    for node in (
        "A",
        "B",
        "C",
        "D",
        "X",
    ):
        G.add_node(
            node,
            **_node_data(),
        )

    G.add_edge(
        "A",
        "B",
        key=0,
        **_edge_data(10000),
    )

    G.add_edge(
        "A",
        "B",
        key=1,
        **_edge_data(5000),
    )

    G.add_edge(
        "B",
        "C",
        key=0,
        **_edge_data(10000),
    )

    G.add_edge(
        "C",
        "D",
        key=0,
        **_edge_data(10000),
    )

    G.add_edge(
        "B",
        "D",
        key=0,
        **_edge_data(10000),
    )

    return G


def candidate(
    path,
    edges,
    cost=10.0,
):
    """
    Standard Candidate schema.
    """

    return {
        "path": list(path),
        "edges": list(edges),
        "cost": cost,
        "hop_count": len(path) - 1,
        "total_fee": 0.0,
        "total_delay": 0.0,
        "reliability": 1.0,
        "failure_probability": 0.0,
        "eta": 0.5,
        "lambda_h": 1.0,
        "candidate": True,
        "success": None,
    }


def top_k_style_candidate(
    path,
    edges,
    cost=10.0,
):
    """
    Candidate representation matching the current Top-K
    physical edge schema.

    Edge dictionaries deliberately carry channel identity.
    """

    return {
        "path": list(path),
        "edges": [
            {
                "source": edge[0],
                "target": edge[1],
                "channel_key": edge[2],
            }
            for edge in edges
        ],
        "cost": cost,
        "hop_count": len(path) - 1,
        "total_fee": 0.0,
        "total_delay": 0.0,
        "reliability": 1.0,
        "failure_probability": 0.0,
        "eta": 0.5,
        "lambda_h": 1.0,
        "candidate": True,
        "success": None,
    }


# ============================================================
# Test Runner
# ============================================================

TESTS = []


def test(name):
    def decorator(function):
        TESTS.append((name, function))
        return function

    return decorator


# ============================================================
# 01. Initialization
# ============================================================

@test("Bucket initialization")
def test_bucket_initialization():

    bucket = Bucket(
        bucket_id="B001",
        transaction_id="TX001",
        candidates=[],
    )

    assert bucket.current_index == 0
    assert bucket.attempts == 0
    assert bucket.status == "active"
    assert bucket.selected_candidate is None
    assert bucket.selected_candidate_index is None
    assert bucket.selected_candidate_rank is None


# ============================================================
# 02. Invalid initialization
# ============================================================

@test("invalid initial status")
def test_invalid_initial_status():

    try:
        Bucket(
            bucket_id="B001",
            transaction_id="TX001",
            status="UNKNOWN",
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid status was accepted"
    )


@test("invalid initial index")
def test_invalid_initial_index():

    try:
        Bucket(
            bucket_id="B001",
            transaction_id="TX001",
            current_index=-1,
        )
    except ValueError:
        return

    raise AssertionError(
        "Negative current_index was accepted"
    )


# ============================================================
# 03. Current Candidate
# ============================================================

@test("current candidate")
def test_current_candidate():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    assert bucket.current() is c1


# ============================================================
# 04. Candidate progression
# ============================================================

@test("candidate progression")
def test_candidate_progression():

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    assert bucket.next_candidate() is c2
    assert bucket.current_index == 1
    assert bucket.status == "active"


# ============================================================
# 05. Candidate exhaustion
# ============================================================

@test("candidate exhaustion")
def test_candidate_exhaustion():

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1],
    )

    assert bucket.next_candidate() is None
    assert bucket.status == "failed"
    assert bucket.finished()


# ============================================================
# 06. Exact MultiDiGraph key
# ============================================================

@test("exact MultiDiGraph channel key")
def test_exact_channel_key():

    G = _build_graph()

    channel = _get_channel(
        G,
        ("A", "B", 1),
    )

    assert channel is not None
    assert channel["estimated_liquidity"] == 5000


# ============================================================
# 07. Parallel channel protection
# ============================================================

@test("parallel channel requires key")
def test_parallel_channel_requires_key():

    G = _build_graph()

    channel = _get_channel(
        G,
        ("A", "B"),
    )

    assert channel is None


# ============================================================
# 08. Missing channel
# ============================================================

@test("missing channel")
def test_missing_channel():

    G = _build_graph()

    c = candidate(
        ["A", "X"],
        [("A", "X", 0)],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is False
    assert reason == "missing_channel"
    assert failed_channel == (
        "A",
        "X",
        0,
    )


# ============================================================
# 09. Path / edge mismatch
# ============================================================

@test("path edge mismatch")
def test_path_edge_mismatch():

    G = _build_graph()

    c = candidate(
        ["A", "B", "C"],
        [
            ("A", "B", 0),
            ("A", "C", 0),
        ],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is False
    assert reason == "path_edge_mismatch"


# ============================================================
# 10. Path edge count mismatch
# ============================================================

@test("path edge count mismatch")
def test_path_edge_count_mismatch():

    G = _build_graph()

    c = candidate(
        ["A", "B", "C"],
        [
            ("A", "B", 0),
        ],
    )

    valid, _, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is False
    assert reason == "path_edge_count_mismatch"


# ============================================================
# 11. Channel unavailable
# ============================================================

@test("channel unavailable")
def test_channel_unavailable():

    G = _build_graph()

    G.edges["A", "B", 0]["available"] = False

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is False
    assert failed_channel == (
        "A",
        "B",
        0,
    )
    assert reason == "channel_unavailable"


# ============================================================
# 12. Invalid channel Boolean
# ============================================================

@test("invalid channel Boolean")
def test_invalid_channel_boolean():

    G = _build_graph()

    G.edges["A", "B", 0]["available"] = "false"

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    try:
        validate_candidate(
            G,
            c,
            1000,
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid channel Boolean was accepted"
    )


# ============================================================
# 13. Invalid node Boolean
# ============================================================

@test("invalid node Boolean")
def test_invalid_node_boolean():

    G = _build_graph()

    G.nodes["A"]["online"] = "false"

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    try:
        validate_candidate(
            G,
            c,
            1000,
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid node Boolean was accepted"
    )


# ============================================================
# 14. Insufficient liquidity
# ============================================================

@test("insufficient liquidity")
def test_insufficient_liquidity():

    G = _build_graph()

    G.edges["A", "B", 0][
        "estimated_liquidity"
    ] = 500

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is False
    assert failed_channel == (
        "A",
        "B",
        0,
    )
    assert reason == "insufficient_liquidity"


# ============================================================
# 15. Sufficient liquidity
# ============================================================

@test("sufficient directional liquidity")
def test_sufficient_liquidity():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is True
    assert failed_channel is None
    assert reason is None


# ============================================================
# 16. Unknown liquidity
# ============================================================

@test("unknown liquidity is accepted")
def test_unknown_liquidity():

    G = _build_graph()

    channel = G.edges[
        "A",
        "B",
        0,
    ]

    del channel[
        "estimated_liquidity"
    ]

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    valid, _, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is True
    assert reason is None


# ============================================================
# 17. Unknown liquidity is not zero
# ============================================================

@test("unknown liquidity is not zero")
def test_unknown_liquidity_not_zero():

    G = _build_graph()

    channel = G.edges[
        "A",
        "B",
        0,
    ]

    del channel[
        "estimated_liquidity"
    ]

    assert _get_known_liquidity(
        channel
    ) is None

    assert _liquidity_is_sufficient(
        channel,
        10_000_000,
    ) is True


# ============================================================
# 18. Invalid liquidity
# ============================================================

@test("invalid liquidity")
def test_invalid_liquidity():

    G = _build_graph()

    G.edges[
        "A",
        "B",
        0,
    ]["estimated_liquidity"] = "INVALID"

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    try:
        validate_candidate(
            G,
            c,
            1000,
        )
    except ValueError:
        return

    raise AssertionError(
        "Invalid liquidity was silently ignored"
    )


# ============================================================
# 19. Capacity is not directional liquidity
# ============================================================

@test("capacity is not directional liquidity")
def test_capacity_not_liquidity():

    G = _build_graph()

    channel = G.edges[
        "A",
        "B",
        0,
    ]

    del channel[
        "estimated_liquidity"
    ]

    channel["capacity"] = 1

    assert _get_known_liquidity(
        channel
    ) is None

    assert _liquidity_is_sufficient(
        channel,
        1000,
    ) is True


# ============================================================
# 20. Candidate selection
# ============================================================

@test("candidate selection")
def test_candidate_selection():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c],
    )

    selected_ok, selected, attempts = (
        execute_bucket(
            G,
            bucket,
            1000,
        )
    )

    assert selected_ok is True
    assert selected is c
    assert attempts == 0
    assert bucket.status == "active"
    assert bucket.selected_candidate is c


# ============================================================
# 21. Selection does not increment attempts
# ============================================================

@test("selection does not increment attempts")
def test_selection_no_attempt():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c],
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    assert bucket.attempts == 0


# ============================================================
# 22. Actual attempt
# ============================================================

@test("record actual payment attempt")
def test_record_attempt():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c],
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    assert bucket.record_attempt() == 1
    assert bucket.attempts == 1
    assert bucket.status == "active"


# ============================================================
# 23. Backtracking does not increment attempts
# ============================================================

@test("backtracking does not increment attempts")
def test_backtracking_no_attempt():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    before = bucket.attempts

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="channel_failure",
    )

    assert bucket.attempts == before


# ============================================================
# 24. Failed candidate tracking
# ============================================================

@test("failed candidate tracking")
def test_failed_candidate_tracking():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="failure",
    )

    assert bucket.failed_candidates_count == 1
    assert bucket.failed_candidates[0] is c1


# ============================================================
# 25. Duplicate failed candidate
# ============================================================

@test("duplicate failed candidate protection")
def test_duplicate_failed_candidate():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="failure",
    )

    # Manually move pointer back to test duplicate protection.
    bucket.current_index = 0
    bucket.status = "active"

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="failure_again",
    )

    assert bucket.failed_candidates_count == 1


# ============================================================
# 26. Failed channel tracking
# ============================================================

@test("failed channel tracking")
def test_failed_channel_tracking():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1],
    )

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="channel_failure",
    )

    assert bucket.failed_channels == [
        ("A", "B", 0)
    ]


# ============================================================
# 27. Duplicate failed channel
# ============================================================

@test("duplicate failed channel protection")
def test_duplicate_failed_channel():

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    c3 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2, c3],
    )

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="failure",
    )

    bucket.backtrack(
        failed_candidate=c2,
        failed_channel=("A", "B", 0),
        reason="failure_again",
    )

    assert bucket.failed_channels == [
        ("A", "B", 0)
    ]


# ============================================================
# 28. Failure reason
# ============================================================

@test("failure reason tracking")
def test_failure_reason_tracking():

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="channel_failure",
    )

    assert len(
        bucket.failure_reasons
    ) == 1

    record = bucket.failure_reasons[0]

    assert record[
        "candidate_index"
    ] == 0

    assert record[
        "candidate_rank"
    ] == 1

    assert record[
        "reason"
    ] == "channel_failure"

    assert record[
        "failed_channel"
    ] == ("A", "B", 0)


# ============================================================
# 29. Alternative candidate
# ============================================================

@test("alternative candidate selection")
def test_alternative_candidate_selection():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="channel_failure",
    )

    selected_ok, selected, attempts = (
        execute_bucket(
            G,
            bucket,
            1000,
        )
    )

    assert selected_ok is True
    assert selected is c2
    assert bucket.current_index == 1
    assert attempts == 1


# ============================================================
# 30. Successful candidate index
# ============================================================

@test("successful candidate index")
def test_successful_candidate_index():

    G = _build_graph()

    candidates = [
        candidate(
            ["A", "B"],
            [("A", "B", 0)],
        ),
        candidate(
            ["B", "D"],
            [("B", "D", 0)],
        ),
    ]

    bucket = Bucket(
        "B001",
        "TX001",
        candidates,
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    bucket.backtrack(
        failed_candidate=candidates[0],
        failed_channel=("A", "B", 0),
        reason="failure",
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    assert bucket.mark_success(
        candidates[1]
    ) is True

    assert bucket.selected_candidate_index == 1


# ============================================================
# 31. Successful candidate rank
# ============================================================

@test("successful candidate rank")
def test_successful_candidate_rank():

    G = _build_graph()

    candidates = [
        candidate(
            ["A", "B"],
            [("A", "B", 0)],
        ),
        candidate(
            ["B", "D"],
            [("B", "D", 0)],
        ),
    ]

    bucket = Bucket(
        "B001",
        "TX001",
        candidates,
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.backtrack(
        failed_candidate=candidates[0],
        failed_channel=("A", "B", 0),
        reason="failure",
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    assert bucket.mark_success(
        candidates[1]
    ) is True

    assert bucket.selected_candidate_rank == 2


# ============================================================
# 32. Completed Bucket protection
# ============================================================

@test("completed bucket protection")
def test_completed_bucket_protection():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c],
    )

    execute_bucket(
        G,
        bucket,
        1000,
    )

    bucket.record_attempt()

    assert bucket.mark_success(c) is True
    assert bucket.status == "completed"

    attempts_before = bucket.attempts

    result = execute_bucket(
        G,
        bucket,
        1000,
    )

    assert result[0] is True
    assert bucket.attempts == attempts_before

    assert bucket.record_attempt() == attempts_before

    assert bucket.backtrack(
        failed_candidate=c,
        failed_channel=("A", "B", 0),
        reason="late_failure",
    ) is None

    assert bucket.status == "completed"


# ============================================================
# 33. Failed Bucket protection
# ============================================================

@test("failed bucket protection")
def test_failed_bucket_protection():

    bucket = Bucket(
        "B001",
        "TX001",
        [],
    )

    bucket.fail()

    assert bucket.status == "failed"
    assert bucket.finished()

    assert bucket.current() is None

    assert bucket.next_candidate() is None

    assert bucket.record_attempt() == 0


# ============================================================
# 34. Full exhaustion
# ============================================================

@test("full candidate exhaustion")
def test_full_candidate_exhaustion():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "C"],
        [("B", "C", 0)],
    )

    # Make both structurally invalid.
    G.edges[
        "A",
        "B",
        0,
    ]["available"] = False

    G.edges[
        "B",
        "C",
        0,
    ]["available"] = False

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    selected_ok, selected, attempts = (
        execute_bucket(
            G,
            bucket,
            1000,
        )
    )

    assert selected_ok is False
    assert selected is None
    assert attempts == 0
    assert bucket.status == "failed"
    assert bucket.failed_candidates_count == 2


# ============================================================
# 35. Deterministic lifecycle
# ============================================================

@test("deterministic lifecycle")
def test_deterministic_lifecycle():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2],
    )

    # Selection.
    ok, selected, _ = execute_bucket(
        G,
        bucket,
        1000,
    )

    assert ok is True
    assert selected is c1
    assert bucket.attempts == 0

    # Attempt 1.
    bucket.record_attempt()

    assert bucket.attempts == 1

    # Failure.
    bucket.backtrack(
        failed_candidate=c1,
        failed_channel=("A", "B", 0),
        reason="channel_failure",
    )

    assert bucket.attempts == 1
    assert bucket.current() is c2

    # Selection 2.
    ok, selected, _ = execute_bucket(
        G,
        bucket,
        1000,
    )

    assert ok is True
    assert selected is c2
    assert bucket.attempts == 1

    # Attempt 2.
    bucket.record_attempt()

    assert bucket.attempts == 2

    # Success.
    assert bucket.mark_success(
        c2
    ) is True

    assert bucket.status == "completed"
    assert bucket.selected_candidate is c2
    assert bucket.selected_candidate_index == 1
    assert bucket.selected_candidate_rank == 2


# ============================================================
# 36. Top-K style edge dictionaries
# ============================================================

@test("Top-K edge dictionary compatibility")
def test_top_k_edge_dictionary():

    G = _build_graph()

    c = top_k_style_candidate(
        ["A", "B"],
        [("A", "B", 1)],
    )

    valid, failed_channel, reason = (
        validate_candidate(
            G,
            c,
            1000,
        )
    )

    assert valid is True
    assert failed_channel is None
    assert reason is None


# ============================================================
# 37. Top-K exact channel identity
# ============================================================

@test("Top-K channel identity preservation")
def test_top_k_channel_identity():

    G = _build_graph()

    c = top_k_style_candidate(
        ["A", "B"],
        [("A", "B", 1)],
    )

    edge = c["edges"][0]

    assert edge[
        "channel_key"
    ] == 1

    normalized = _normalize_edge(edge)

    assert normalized == (
        "A",
        "B",
        1,
    )

    channel = _get_channel(
        G,
        normalized,
    )

    assert channel[
        "estimated_liquidity"
    ] == 5000


# ============================================================
# 38. Candidate ordering preservation
# ============================================================

@test("candidate ordering preservation")
def test_candidate_ordering():

    G = _build_graph()

    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
        cost=1.0,
    )

    c2 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
        cost=2.0,
    )

    c3 = candidate(
        ["C", "D"],
        [("C", "D", 0)],
        cost=3.0,
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2, c3],
    )

    assert bucket.candidates[0] is c1
    assert bucket.candidates[1] is c2
    assert bucket.candidates[2] is c3


# ============================================================
# 39. Invalid amount
# ============================================================

@test("invalid amount")
def test_invalid_amount():

    G = _build_graph()

    c = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    valid, _, reason = (
        validate_candidate(
            G,
            c,
            0,
        )
    )

    assert valid is False
    assert reason == "invalid_amount"


# ============================================================
# 40. Complete failure/backtracking lifecycle
# ============================================================

@test("complete failure and backtracking lifecycle")
def test_complete_failure_backtracking():

    G = _build_graph()

    # Candidate 1: unavailable.
    G.edges[
        "A",
        "B",
        0,
    ]["available"] = False

    # Candidate 2: insufficient.
    G.edges[
        "A",
        "B",
        1,
    ]["estimated_liquidity"] = 100

    # Candidate 3: valid.
    c1 = candidate(
        ["A", "B"],
        [("A", "B", 0)],
    )

    c2 = candidate(
        ["A", "B"],
        [("A", "B", 1)],
    )

    c3 = candidate(
        ["B", "D"],
        [("B", "D", 0)],
    )

    bucket = Bucket(
        "B001",
        "TX001",
        [c1, c2, c3],
    )

    ok, selected, attempts = (
        execute_bucket(
            G,
            bucket,
            1000,
        )
    )

    assert ok is True
    assert selected is c3
    assert attempts == 0

    assert bucket.failed_candidates_count == 2

    assert bucket.current_index == 2

    assert bucket.failure_reasons[0][
        "reason"
    ] == "channel_unavailable"

    assert bucket.failure_reasons[1][
        "reason"
    ] == "insufficient_liquidity"

    # Actual payment attempt only now.
    bucket.record_attempt()

    assert bucket.attempts == 1

    # Actual payment success.
    assert bucket.mark_success(
        c3
    ) is True

    assert bucket.status == "completed"
    assert bucket.selected_candidate_index == 2
    assert bucket.selected_candidate_rank == 3


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 72)
    print(
        " BUCKET DETERMINISTIC VALIDATION"
    )
    print("=" * 72)

    passed = 0
    failed = 0

    for index, (name, function) in enumerate(
        TESTS,
        start=1,
    ):

        try:

            function()

            print(
                f"[{index:02d}/{len(TESTS)}] "
                f"{name:<45} PASS"
            )

            passed += 1

        except Exception as exc:

            print(
                f"[{index:02d}/{len(TESTS)}] "
                f"{name:<45} FAIL"
            )

            print(
                f"    {type(exc).__name__}: {exc}"
            )

            failed += 1

    print("=" * 72)
    print(
        f"Total : {len(TESTS)}"
    )
    print(
        f"Passed: {passed}"
    )
    print(
        f"Failed: {failed}"
    )
    print("=" * 72)

    if failed == 0:

        print(
            "BUCKET VALIDATION: PASS"
        )

        return 0

    print(
        "BUCKET VALIDATION: FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )