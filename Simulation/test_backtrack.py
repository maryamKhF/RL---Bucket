"""
Simulation/test_backtrack.py

Comprehensive validation suite for PartialBacktracker.

Run:
    py -m Simulation.test_backtrack

Main validation areas
---------------------

1. Constructor
2. Route validation
3. Amount validation
4. Exact failed-edge resolution
5. Exact failure-index resolution
6. Exact route_edges validation
7. MultiDiGraph channel identity
8. No arbitrary parallel-channel selection
9. Prefix preservation
10. Exact suffix preservation
11. Failed-channel rejection
12. Different parallel-channel acceptance
13. Liquidity validation
14. Capacity != liquidity
15. Malformed liquidity rejection
16. NaN / infinity rejection
17. Node availability
18. Channel availability
19. Loop prevention
20. Same-route rejection
21. Failed candidate rejection
22. Bucket alternatives
23. Bucket dictionary containers
24. Bucket ID isolation
25. Empty Bucket
26. Bucket.attempts immutability
27. Attempt ID preservation
28. No full rerouting
29. Graph / DiGraph support
30. MultiDiGraph end-to-end reconstruction
31. Input immutability
"""

from __future__ import annotations

import math
import networkx as nx

from Simulation.backtrack import PartialBacktracker


# ==============================================================
# Test Helpers
# ==============================================================

TOTAL = 0
PASSED = 0
FAILED = 0


def check(name, condition):
    global TOTAL, PASSED, FAILED

    TOTAL += 1

    if condition:
        PASSED += 1
        print(f"[{TOTAL:02d}] {name:<65} PASS")
    else:
        FAILED += 1
        print(f"[{TOTAL:02d}] {name:<65} FAIL")


def expect_exception(name, fn):
    global TOTAL, PASSED, FAILED

    TOTAL += 1

    try:
        fn()
    except Exception:
        PASSED += 1
        print(f"[{TOTAL:02d}] {name:<65} PASS")
    else:
        FAILED += 1
        print(f"[{TOTAL:02d}] {name:<65} FAIL")


# ==============================================================
# Graph Builders
# ==============================================================

def add_node_attributes(G):
    for node in G.nodes:
        G.nodes[node]["available"] = True
        G.nodes[node]["is_online"] = True


def add_channel(
    G,
    u,
    v,
    key=None,
    liquidity=10000,
    available=True,
    capacity=100000,
    **extra,
):
    data = {
        "capacity": capacity,
        "balance_uv": liquidity,
        "available": available,
    }

    data.update(extra)

    if G.is_multigraph():
        G.add_edge(
            u,
            v,
            key=key,
            **data,
        )
    else:
        G.add_edge(
            u,
            v,
            **data,
        )


def make_multigraph():
    G = nx.MultiDiGraph()

    for u, v in [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("B", "F"),
        ("F", "G"),
        ("G", "E"),
    ]:
        add_channel(
            G,
            u,
            v,
            key=0,
            liquidity=10000,
        )

    add_node_attributes(G)

    return G


def make_parallel_graph():
    G = nx.MultiDiGraph()

    add_channel(
        G,
        "A",
        "B",
        key=0,
        liquidity=10000,
    )

    add_channel(
        G,
        "A",
        "B",
        key=1,
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "C",
        key=0,
        liquidity=10000,
    )

    add_node_attributes(G)

    return G


def make_graph():
    G = nx.Graph()

    add_channel(
        G,
        "A",
        "B",
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "C",
        liquidity=10000,
    )

    add_channel(
        G,
        "C",
        "D",
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "E",
        liquidity=10000,
    )

    add_channel(
        G,
        "E",
        "D",
        liquidity=10000,
    )

    add_node_attributes(G)

    return G


def make_digraph():
    G = nx.DiGraph()

    add_channel(
        G,
        "A",
        "B",
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "C",
        liquidity=10000,
    )

    add_channel(
        G,
        "C",
        "D",
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "E",
        liquidity=10000,
    )

    add_channel(
        G,
        "E",
        "D",
        liquidity=10000,
    )

    add_node_attributes(G)

    return G


# ==============================================================
# Bucket Fixtures
# ==============================================================

class Bucket:
    def __init__(
        self,
        candidates=None,
        failed_candidates=None,
    ):
        self.candidates = (
            []
            if candidates is None
            else candidates
        )

        self.failed_candidates = (
            []
            if failed_candidates is None
            else failed_candidates
        )

        self.attempts = 0


# ==============================================================
# Common Routes
# ==============================================================

ORIGINAL_ROUTE = [
    "A",
    "B",
    "C",
    "D",
    "E",
]

ORIGINAL_EDGES = [
    ("A", "B", 0),
    ("B", "C", 0),
    ("C", "D", 0),
    ("D", "E", 0),
]

ALTERNATIVE_ROUTE = [
    "A",
    "B",
    "F",
    "G",
    "E",
]

ALTERNATIVE_EDGES = [
    ("A", "B", 0),
    ("B", "F", 0),
    ("F", "G", 0),
    ("G", "E", 0),
]


def valid_candidate():
    return {
        "route": list(ALTERNATIVE_ROUTE),
        "edges": list(ALTERNATIVE_EDGES),
    }


# ==============================================================
# 1. Constructor Tests
# ==============================================================

def test_constructor():

    G = make_multigraph()

    backtracker = PartialBacktracker(
        network=G,
        bucket=None,
    )

    check(
        "constructor accepts valid network",
        backtracker.network is G,
    )


def test_constructor_rejects_none():

    expect_exception(
        "constructor rejects None network",
        lambda: PartialBacktracker(None),
    )


# ==============================================================
# 2. Route Validation
# ==============================================================

def test_route_validation():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "valid route accepted",
        bt._validate_route(
            ORIGINAL_ROUTE
        ),
    )

    check(
        "route with one node rejected",
        not bt._validate_route(
            ["A"]
        ),
    )

    check(
        "empty route rejected",
        not bt._validate_route(
            []
        ),
    )

    check(
        "route containing loop rejected",
        not bt._validate_route(
            ["A", "B", "C", "B"]
        ),
    )

    check(
        "route ending at start rejected",
        not bt._validate_route(
            ["A", "B", "A"]
        ),
    )

    check(
        "non-list route rejected",
        not bt._validate_route(
            "A-B-C"
        ),
    )


# ==============================================================
# 3. Amount Validation
# ==============================================================

def test_amount_validation():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    valid_values = [
        1,
        1000,
        1000.5,
    ]

    for value in valid_values:

        check(
            f"valid amount {value} accepted",
            bt._validate_amount(value) is not None,
        )

    invalid_values = [
        None,
        True,
        False,
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "abc",
    ]

    for value in invalid_values:

        check(
            f"invalid amount {value!r} rejected",
            bt._validate_amount(value) is None,
        )


# ==============================================================
# 4. Exact route_edges validation
# ==============================================================

def test_route_edges_validation():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    normalized = bt._normalize_route_edges(
        ORIGINAL_ROUTE,
        ORIGINAL_EDGES,
    )

    check(
        "exact MultiDiGraph route_edges accepted",
        normalized == ORIGINAL_EDGES,
    )

    wrong_length = [
        ("A", "B", 0),
    ]

    check(
        "route_edges length mismatch rejected",
        bt._normalize_route_edges(
            ORIGINAL_ROUTE,
            wrong_length,
        )
        is False,
    )

    wrong_endpoint = list(
        ORIGINAL_EDGES
    )

    wrong_endpoint[1] = (
        "X",
        "C",
        0,
    )

    check(
        "route_edges endpoint mismatch rejected",
        bt._normalize_route_edges(
            ORIGINAL_ROUTE,
            wrong_endpoint,
        )
        is False,
    )


def test_multigraph_requires_exact_route_edges():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "MultiDiGraph requires route_edges",
        bt._normalize_route_edges(
            ORIGINAL_ROUTE,
            None,
        )
        is False,
    )


# ==============================================================
# 5. Failed Edge Resolution
# ==============================================================

def test_failed_edge_resolution():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    failed = bt._resolve_failed_edge(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
    )

    check(
        "exact failed edge resolved",
        failed == ("C", "D", 0),
    )

    failed_from_route_edges = bt._resolve_failed_edge(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=None,
        failure_index=2,
    )

    check(
        "failed edge derived from exact route_edges",
        failed_from_route_edges == (
            "C",
            "D",
            0,
        ),
    )

    missing_key = bt._resolve_failed_edge(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D"),
        failure_index=2,
    )

    check(
        "failed endpoint without key accepted for explicit failure",
        missing_key == (
            "C",
            "D",
            None,
        ),
    )


# ==============================================================
# 6. Failure Index
# ==============================================================

def test_failure_index():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    index = bt._resolve_failure_index(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
    )

    check(
        "exact failure index resolved",
        index == 2,
    )

    mismatch = bt._resolve_failure_index(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("B", "D", 0),
        failure_index=2,
    )

    check(
        "failure endpoint mismatch rejected",
        mismatch is None,
    )


# ==============================================================
# 7. MultiDiGraph Exact Channel Identity
# ==============================================================

def test_exact_parallel_channel():

    G = make_parallel_graph()

    bt = PartialBacktracker(G)

    check(
        "exact parallel channel key 0 available",
        bt._exact_channel_available(
            "A",
            "B",
            0,
            1000,
        ),
    )

    check(
        "exact parallel channel key 1 available",
        bt._exact_channel_available(
            "A",
            "B",
            1,
            1000,
        ),
    )

    check(
        "missing MultiDiGraph key rejected",
        not bt._exact_channel_available(
            "A",
            "B",
            None,
            1000,
        ),
    )


def test_no_arbitrary_parallel_selection():

    G = make_parallel_graph()

    bt = PartialBacktracker(G)

    resolved = bt._resolve_edges_for_path(
        path=["A", "B"],
        amount=1000,
    )

    check(
        "MultiDiGraph node-only path never selects arbitrary channel",
        resolved is None,
    )


# ==============================================================
# 8. Failed Channel Reuse
# ==============================================================

def test_failed_channel_reuse():

    G = make_parallel_graph()

    bt = PartialBacktracker(G)

    check(
        "same failed parallel channel rejected",
        bt._candidate_contains_failed_edge(
            candidate=None,
            suffix=["A", "B"],
            suffix_edges=[
                ("A", "B", 0),
            ],
            failed_edge=("A", "B", 0),
        ),
    )

    check(
        "different parallel channel allowed",
        not bt._candidate_contains_failed_edge(
            candidate=None,
            suffix=["A", "B"],
            suffix_edges=[
                ("A", "B", 1),
            ],
            failed_edge=("A", "B", 0),
        ),
    )

    check(
        "unknown failed key rejects same endpoint",
        bt._candidate_contains_failed_edge(
            candidate=None,
            suffix=["A", "B"],
            suffix_edges=[
                ("A", "B", 1),
            ],
            failed_edge=("A", "B", None),
        ),
    )


# ==============================================================
# 9. Liquidity
# ==============================================================

def test_liquidity():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    data = G["A"]["B"][0]

    data["balance_uv"] = 500

    check(
        "insufficient directional liquidity rejected",
        not bt._edge_data_usable(
            data,
            1000,
        ),
    )

    data["balance_uv"] = 1000

    check(
        "exact directional liquidity accepted",
        bt._edge_data_usable(
            data,
            1000,
        ),
    )

    data["balance_uv"] = 2000

    check(
        "sufficient directional liquidity accepted",
        bt._edge_data_usable(
            data,
            1000,
        ),
    )


def test_unknown_liquidity():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    data = G["A"]["B"][0]

    del data["balance_uv"]

    check(
        "unknown directional liquidity accepted",
        bt._edge_data_usable(
            data,
            1000,
        ),
    )


def test_capacity_not_liquidity():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    data = G["A"]["B"][0]

    data.pop(
        "balance_uv",
        None,
    )

    data["capacity"] = 100

    check(
        "capacity alone does not become liquidity",
        bt._edge_data_usable(
            data,
            1000,
        ),
    )


def test_malformed_liquidity():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    invalid_values = [
        "abc",
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
    ]

    for value in invalid_values:

        data = G["A"]["B"][0].copy()

        data["balance_uv"] = value

        check(
            f"malformed liquidity {value!r} rejected",
            not bt._edge_data_usable(
                data,
                1000,
            ),
        )


# ==============================================================
# 10. Node / Channel Availability
# ==============================================================

def test_node_availability():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    G.nodes["B"]["available"] = False

    check(
        "unavailable node rejected",
        not bt._node_is_available(
            G.nodes["B"]
        ),
    )


def test_channel_availability():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    G["A"]["B"][0]["available"] = False

    check(
        "unavailable exact channel rejected",
        not bt._exact_channel_available(
            "A",
            "B",
            0,
            1000,
        ),
    )


# ==============================================================
# 11. Candidate Extraction
# ==============================================================

def test_candidate_extraction():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    candidate = valid_candidate()

    path = bt._extract_candidate_path(
        candidate
    )

    edges = bt._extract_candidate_edges(
        candidate
    )

    check(
        "candidate path extracted",
        path == ALTERNATIVE_ROUTE,
    )

    check(
        "candidate exact edges extracted",
        edges == ALTERNATIVE_EDGES,
    )


def test_suffix_normalization():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    candidate = valid_candidate()

    suffix = bt._normalize_suffix(
        candidate,
        "B",
    )

    check(
        "suffix normalized from branch point",
        suffix == [
            "B",
            "F",
            "G",
            "E",
        ],
    )


# ==============================================================
# 12. Loop Prevention
# ==============================================================

def test_loop_prevention():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "loop-free route accepted",
        bt._is_loop_free(
            ["A", "B"],
            ["B", "F", "G", "E"],
        ),
    )

    check(
        "looping candidate rejected",
        not bt._is_loop_free(
            ["A", "B"],
            ["B", "A", "E"],
        ),
    )


# ==============================================================
# 13. Same Route Rejection
# ==============================================================

def test_same_route():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "same route detected",
        bt._same_route(
            ORIGINAL_ROUTE,
            ORIGINAL_ROUTE,
        ),
    )

    check(
        "different route accepted",
        not bt._same_route(
            ORIGINAL_ROUTE,
            ALTERNATIVE_ROUTE,
        ),
    )


# ==============================================================
# 14. Bucket Failed Candidate
# ==============================================================

def test_failed_candidate():

    G = make_multigraph()

    failed = valid_candidate()

    bucket = Bucket(
        candidates=[
            failed
        ],
        failed_candidates=[
            failed
        ],
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    check(
        "Bucket failed candidate detected",
        bt._candidate_is_failed(
            failed
        ),
    )


def test_candidate_success_false():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    candidate = valid_candidate()

    candidate["success"] = False

    check(
        "candidate success=False rejected",
        bt._candidate_is_failed(
            candidate
        ),
    )


def test_candidate_failed_true():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    candidate = valid_candidate()

    candidate["failed"] = True

    check(
        "candidate failed=True rejected",
        bt._candidate_is_failed(
            candidate
        ),
    )


# ==============================================================
# 15. Bucket Retrieval
# ==============================================================

def test_bucket_retrieval():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    candidates = bt.get_alternative_suffixes(
        route=ORIGINAL_ROUTE,
        branch_index=1,
        failed_edge=("C", "D", 0),
    )

    check(
        "Bucket candidates retrieved",
        len(candidates) == 1,
    )


def test_empty_bucket():

    G = make_multigraph()

    bucket = Bucket()

    bt = PartialBacktracker(
        G,
        bucket,
    )

    candidates = bt.get_alternative_suffixes(
        route=ORIGINAL_ROUTE,
        branch_index=1,
        failed_edge=("C", "D", 0),
    )

    check(
        "empty Bucket returns no candidates",
        candidates == [],
    )


def test_none_bucket():

    G = make_multigraph()

    bt = PartialBacktracker(
        G,
        bucket=None,
    )

    candidates = bt.get_alternative_suffixes(
        route=ORIGINAL_ROUTE,
        branch_index=1,
        failed_edge=("C", "D", 0),
    )

    check(
        "None Bucket returns no candidates",
        candidates == [],
    )


# ==============================================================
# 16. Bucket Dictionary Containers
# ==============================================================

def test_bucket_dictionary():

    G = make_multigraph()

    candidate_a = valid_candidate()

    candidate_b = {
        "route": [
            "B",
            "F",
            "G",
            "E",
        ],
        "edges": [
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ],
    }

    bucket = Bucket(
        candidates={
            "bucket_A": [
                candidate_a
            ],
            "bucket_B": [
                candidate_b
            ],
        }
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result_a = bt.get_alternative_suffixes(
        route=ORIGINAL_ROUTE,
        branch_index=1,
        failed_edge=("C", "D", 0),
        bucket_id="bucket_A",
    )

    result_b = bt.get_alternative_suffixes(
        route=ORIGINAL_ROUTE,
        branch_index=1,
        failed_edge=("C", "D", 0),
        bucket_id="bucket_B",
    )

    check(
        "Bucket dictionary bucket_A isolated",
        result_a == [candidate_a],
    )

    check(
        "Bucket dictionary bucket_B isolated",
        result_b == [candidate_b],
    )


# ==============================================================
# 17. Full Successful Backtrack
# ==============================================================

def test_successful_backtrack():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    original_route = list(
        ORIGINAL_ROUTE
    )

    original_edges = list(
        ORIGINAL_EDGES
    )

    result = bt.backtrack(
        route=original_route,
        route_edges=original_edges,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
        bucket_id="B001",
        attempt_id=7,
    )

    check(
        "successful partial backtrack",
        result["success"] is True,
    )

    check(
        "status alternative_found",
        result["status"]
        == "alternative_found",
    )

    check(
        "correct branch point",
        result["branch_point"] == "B",
    )

    check(
        "correct branch index",
        result["branch_index"] == 1,
    )

    check(
        "prefix preserved",
        result["preserved_prefix"]
        == ["A", "B"],
    )

    check(
        "prefix edges preserved exactly",
        result["preserved_prefix_edges"]
        == [
            ("A", "B", 0)
        ],
    )

    check(
        "alternative suffix correct",
        result["alternative_suffix"]
        == [
            "B",
            "F",
            "G",
            "E",
        ],
    )

    check(
        "alternative suffix edges exact",
        result["alternative_suffix_edges"]
        == [
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ],
    )

    check(
        "new route correct",
        result["new_route"]
        == ALTERNATIVE_ROUTE,
    )

    check(
        "new exact edges correct",
        result["new_edges"]
        == ALTERNATIVE_EDGES,
    )

    check(
        "retry required",
        result["retry_required"] is True,
    )

    check(
        "full reroute not required",
        result["full_reroute_required"] is False,
    )

    check(
        "attempt ID preserved",
        result["attempt_id"] == 7,
    )


# ==============================================================
# 18. Bucket Attempts Must Not Change
# ==============================================================

def test_bucket_attempts_unchanged():

    G = make_multigraph()

    bucket = Bucket(
        candidates=[
            valid_candidate()
        ]
    )

    bucket.attempts = 13

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
        attempt_id=4,
    )

    check(
        "backtrack succeeds",
        result["success"] is True,
    )

    check(
        "Bucket.attempts remains unchanged",
        bucket.attempts == 13,
    )


# ==============================================================
# 19. No Alternative -> Full Reroute
# ==============================================================

def test_full_reroute_required():

    G = make_multigraph()

    bucket = Bucket(
        candidates=[]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "no alternative detected",
        result["success"] is False,
    )

    check(
        "full reroute required",
        result["full_reroute_required"] is True,
    )

    check(
        "retry not requested",
        result["retry_required"] is False,
    )

    check(
        "status full_reroute_required",
        result["status"]
        == "full_reroute_required",
    )


# ==============================================================
# 20. Failed Candidate Must Not Be Used
# ==============================================================

def test_failed_candidate_rejected():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ],
        failed_candidates=[
            candidate
        ],
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "failed Bucket candidate rejected",
        result["success"] is False,
    )

    check(
        "failed candidate leads to full reroute",
        result["full_reroute_required"] is True,
    )


# ==============================================================
# 21. Failed Edge in Candidate
# ==============================================================

def test_candidate_reuses_failed_edge():

    G = make_multigraph()

    candidate = {
        "route": [
            "A",
            "B",
            "C",
            "D",
            "E",
        ],
        "edges": [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0),
        ],
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "candidate reusing failed edge rejected",
        result["success"] is False,
    )

    check(
        "failed-edge reuse triggers full reroute",
        result["full_reroute_required"] is True,
    )


# ==============================================================
# 22. Different Parallel Channel Allowed
# ==============================================================

def test_different_parallel_channel():

    G = nx.MultiDiGraph()

    add_channel(
        G,
        "A",
        "B",
        key=0,
        liquidity=10000,
    )

    add_channel(
        G,
        "A",
        "B",
        key=1,
        liquidity=10000,
    )

    add_channel(
        G,
        "B",
        "C",
        key=0,
        liquidity=10000,
    )

    add_node_attributes(G)

    route = [
        "A",
        "B",
        "C",
    ]

    route_edges = [
        ("A", "B", 0),
        ("B", "C", 0),
    ]

    candidate = {
        "route": [
            "A",
            "B",
            "C",
        ],
        "edges": [
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=route,
        route_edges=route_edges,
        failed_edge=("A", "B", 0),
        failure_index=0,
        amount=1000,
    )

    check(
        "different parallel channel can be used",
        result["success"] is True,
    )

    if result["success"]:

        check(
            "parallel key 1 preserved",
            result["new_edges"][0]
            == ("A", "B", 1),
        )


# ==============================================================
# 23. Ambiguous MultiDiGraph Candidate
# ==============================================================

def test_ambiguous_multigraph_candidate():

    G = make_parallel_graph()

    candidate = {
        "route": [
            "A",
            "B",
            "C",
        ]
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    suffix = bt._extract_suffix_edges(
        candidate=candidate,
        suffix=[
            "A",
            "B",
            "C",
        ],
        branch_node="A",
        amount=1000,
    )

    check(
        "ambiguous MultiDiGraph candidate without edges rejected",
        suffix is None,
    )


# ==============================================================
# 24. Candidate Exact Edge Preservation
# ==============================================================

def test_candidate_exact_edges():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    candidate = valid_candidate()

    suffix = [
        "B",
        "F",
        "G",
        "E",
    ]

    edges = bt._extract_suffix_edges(
        candidate=candidate,
        suffix=suffix,
        branch_node="B",
        amount=1000,
    )

    check(
        "candidate exact suffix edges preserved",
        edges == [
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ],
    )


# ==============================================================
# 25. Candidate Insufficient Liquidity
# ==============================================================

def test_candidate_insufficient_liquidity():

    G = make_multigraph()

    G["B"]["F"][0]["balance_uv"] = 500

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "insufficient candidate liquidity rejected",
        result["success"] is False,
    )

    check(
        "insufficient liquidity requires full reroute",
        result["full_reroute_required"] is True,
    )


# ==============================================================
# 26. Node Failure in Candidate
# ==============================================================

def test_candidate_unavailable_node():

    G = make_multigraph()

    G.nodes["F"]["available"] = False

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "candidate through unavailable node rejected",
        result["success"] is False,
    )


# ==============================================================
# 27. Candidate Unavailable Channel
# ==============================================================

def test_candidate_unavailable_channel():

    G = make_multigraph()

    G["B"]["F"][0]["available"] = False

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "candidate through unavailable channel rejected",
        result["success"] is False,
    )


# ==============================================================
# 28. Input Immutability
# ==============================================================

def test_input_immutability():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    route = list(
        ORIGINAL_ROUTE
    )

    edges = list(
        ORIGINAL_EDGES
    )

    candidate_copy = {
        "route": list(
            candidate["route"]
        ),
        "edges": list(
            candidate["edges"]
        ),
    }

    bt.backtrack(
        route=route,
        route_edges=edges,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "original route not mutated",
        route == ORIGINAL_ROUTE,
    )

    check(
        "original route_edges not mutated",
        edges == ORIGINAL_EDGES,
    )

    check(
        "candidate route not mutated",
        candidate["route"]
        == candidate_copy["route"],
    )

    check(
        "candidate edges not mutated",
        candidate["edges"]
        == candidate_copy["edges"],
    )


# ==============================================================
# 29. Graph Support
# ==============================================================

def test_graph_support():

    G = make_graph()

    candidate = {
        "route": [
            "B",
            "E",
            "D",
        ]
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    suffix = bt._extract_suffix_edges(
        candidate=candidate,
        suffix=[
            "B",
            "E",
            "D",
        ],
        branch_node="B",
        amount=1000,
    )

    check(
        "simple Graph node-only suffix resolved",
        suffix == [
            ("B", "E", None),
            ("E", "D", None),
        ],
    )


def test_digraph_support():

    G = make_digraph()

    candidate = {
        "route": [
            "B",
            "E",
            "D",
        ]
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    suffix = bt._extract_suffix_edges(
        candidate=candidate,
        suffix=[
            "B",
            "E",
            "D",
        ],
        branch_node="B",
        amount=1000,
    )

    check(
        "DiGraph node-only suffix resolved",
        suffix == [
            ("B", "E", None),
            ("E", "D", None),
        ],
    )


# ==============================================================
# 30. Edge/Route Consistency
# ==============================================================

def test_edge_route_consistency():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "matching route and edges accepted",
        bt._edges_match_route(
            ORIGINAL_ROUTE,
            ORIGINAL_EDGES,
        ),
    )

    bad_edges = list(
        ORIGINAL_EDGES
    )

    bad_edges[1] = (
        "X",
        "C",
        0,
    )

    check(
        "edge/route mismatch rejected",
        not bt._edges_match_route(
            ORIGINAL_ROUTE,
            bad_edges,
        ),
    )


# ==============================================================
# 31. Exact Prefix Preservation with Parallel Channels
# ==============================================================

def test_exact_prefix_parallel_channel():

    G = make_parallel_graph()

    # Add continuation.
    add_channel(
        G,
        "B",
        "C",
        key=0,
        liquidity=10000,
    )

    add_node_attributes(G)

    route = [
        "A",
        "B",
        "C",
    ]

    route_edges = [
        ("A", "B", 1),
        ("B", "C", 0),
    ]

    candidate = {
        "route": [
            "A",
            "B",
            "C",
        ],
        "edges": [
            ("A", "B", 0),
            ("B", "C", 0),
        ],
    }

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    prefix = bt._resolve_route_edges_for_prefix(
        route=route,
        route_edges=route_edges,
        end_index=1,
        amount=1000,
    )

    check(
        "exact original prefix channel preserved",
        prefix == [
            ("A", "B", 1)
        ],
    )


# ==============================================================
# 32. Attempt ID
# ==============================================================

def test_attempt_id():

    G = make_multigraph()

    bt = PartialBacktracker(G)

    check(
        "attempt_id zero valid",
        bt._valid_attempt_id(0),
    )

    check(
        "positive attempt_id valid",
        bt._valid_attempt_id(10),
    )

    check(
        "negative attempt_id invalid",
        not bt._valid_attempt_id(-1),
    )

    check(
        "boolean attempt_id invalid",
        not bt._valid_attempt_id(True),
    )

    check(
        "float attempt_id invalid",
        not bt._valid_attempt_id(1.5),
    )


# ==============================================================
# 33. Invalid Main Calls
# ==============================================================

def test_invalid_main_calls():

    G = make_multigraph()

    bt = PartialBacktracker(
        G,
        Bucket(
            candidates=[
                valid_candidate()
            ]
        ),
    )

    invalid_amount_result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=0,
    )

    check(
        "main API rejects zero amount",
        invalid_amount_result["success"] is False,
    )

    check(
        "zero amount status correct",
        invalid_amount_result["status"]
        == "invalid_amount",
    )

    invalid_attempt_result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
        attempt_id=-1,
    )

    check(
        "main API rejects negative attempt ID",
        invalid_attempt_result["success"] is False,
    )

    invalid_route_result = bt.backtrack(
        route=["A"],
        route_edges=[],
        failed_edge=None,
        failure_index=None,
        amount=1000,
    )

    check(
        "main API rejects invalid route",
        invalid_route_result["success"] is False,
    )


# ==============================================================
# 34. Failure at First Hop
# ==============================================================

def test_failure_at_first_hop():

    G = make_multigraph()

    candidate = {
        "route": [
            "A",
            "B",
            "F",
            "G",
            "E",
        ],
        "edges": [
            ("A", "B", 0),
            ("B", "F", 0),
            ("F", "G", 0),
            ("G", "E", 0),
        ],
    }

    # Candidate still contains the failed A-B edge,
    # so it must be rejected.
    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("A", "B", 0),
        failure_index=0,
        amount=1000,
    )

    check(
        "failure at first hop processed",
        result["success"] is False,
    )


# ==============================================================
# 35. Failure at Last Hop
# ==============================================================

def test_failure_at_last_hop():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("D", "E", 0),
        failure_index=3,
        amount=1000,
    )

    check(
        "failure at last hop processed",
        result["success"] is True,
    )

    if result["success"]:

        check(
            "last-hop backtrack finds B branch",
            result["branch_point"] == "B",
        )


# ==============================================================
# 36. Candidate Ordering
# ==============================================================

def test_candidate_ordering():

    G = make_multigraph()

    invalid_candidate = {
        "route": [
            "B",
            "C",
            "D",
            "E",
        ],
        "edges": [
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "E", 0),
        ],
    }

    valid = valid_candidate()

    bucket = Bucket(
        candidates=[
            invalid_candidate,
            valid,
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    check(
        "later valid candidate selected",
        result["success"] is True,
    )

    if result["success"]:

        check(
            "correct valid candidate selected",
            result["new_route"]
            == ALTERNATIVE_ROUTE,
        )


# ==============================================================
# 37. Amount Propagation
# ==============================================================

def test_amount_propagation():

    G = make_multigraph()

    G["B"]["F"][0]["balance_uv"] = 1500

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result_1000 = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
    )

    result_2000 = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=2000,
    )

    check(
        "candidate succeeds for amount within liquidity",
        result_1000["success"] is True,
    )

    check(
        "candidate rejected when amount exceeds liquidity",
        result_2000["success"] is False,
    )


# ==============================================================
# 38. Exact End-to-End MultiDiGraph
# ==============================================================

def test_end_to_end():

    G = make_multigraph()

    candidate = valid_candidate()

    bucket = Bucket(
        candidates=[
            candidate
        ]
    )

    bucket.attempts = 5

    bt = PartialBacktracker(
        G,
        bucket,
    )

    result = bt.backtrack(
        route=ORIGINAL_ROUTE,
        route_edges=ORIGINAL_EDGES,
        failed_edge=("C", "D", 0),
        failure_index=2,
        amount=1000,
        bucket_id="PAYMENT-001",
        attempt_id=3,
    )

    expected_route = [
        "A",
        "B",
        "F",
        "G",
        "E",
    ]

    expected_edges = [
        ("A", "B", 0),
        ("B", "F", 0),
        ("F", "G", 0),
        ("G", "E", 0),
    ]

    check(
        "end-to-end backtrack succeeds",
        result["success"] is True,
    )

    check(
        "end-to-end route exact",
        result["new_route"]
        == expected_route,
    )

    check(
        "end-to-end edges exact",
        result["new_edges"]
        == expected_edges,
    )

    check(
        "end-to-end branch point exact",
        result["branch_point"] == "B",
    )

    check(
        "end-to-end amount preserved",
        result["amount"] == 1000,
    )

    check(
        "end-to-end bucket ID preserved",
        result["bucket_id"]
        == "PAYMENT-001",
    )

    check(
        "end-to-end attempt ID preserved",
        result["attempt_id"] == 3,
    )

    check(
        "end-to-end Bucket.attempts unchanged",
        bucket.attempts == 5,
    )


# ==============================================================
# 39. Main Test Runner
# ==============================================================

def run_all_tests():

    print()
    print("=" * 82)
    print(" PARTIAL BACKTRACKING COMPREHENSIVE VALIDATION")
    print("=" * 82)
    print()

    # Constructor
    test_constructor()
    test_constructor_rejects_none()

    # Route
    test_route_validation()

    # Amount
    test_amount_validation()

    # Route edges
    test_route_edges_validation()
    test_multigraph_requires_exact_route_edges()

    # Failure
    test_failed_edge_resolution()
    test_failure_index()

    # MultiDiGraph
    test_exact_parallel_channel()
    test_no_arbitrary_parallel_selection()

    # Failed channel
    test_failed_channel_reuse()

    # Liquidity
    test_liquidity()
    test_unknown_liquidity()
    test_capacity_not_liquidity()
    test_malformed_liquidity()

    # Availability
    test_node_availability()
    test_channel_availability()

    # Candidate
    test_candidate_extraction()
    test_suffix_normalization()

    # Loop
    test_loop_prevention()

    # Same route
    test_same_route()

    # Failed candidates
    test_failed_candidate()
    test_candidate_success_false()
    test_candidate_failed_true()

    # Bucket
    test_bucket_retrieval()
    test_empty_bucket()
    test_none_bucket()
    test_bucket_dictionary()

    # Successful backtracking
    test_successful_backtrack()

    # Bucket attempts
    test_bucket_attempts_unchanged()

    # Full reroute
    test_full_reroute_required()

    # Failed route candidate
    test_failed_candidate_rejected()
    test_candidate_reuses_failed_edge()

    # Parallel channel
    test_different_parallel_channel()

    # Ambiguous channel
    test_ambiguous_multigraph_candidate()

    # Exact candidate edges
    test_candidate_exact_edges()

    # Liquidity / availability
    test_candidate_insufficient_liquidity()
    test_candidate_unavailable_node()
    test_candidate_unavailable_channel()

    # Immutability
    test_input_immutability()

    # Graph types
    test_graph_support()
    test_digraph_support()

    # Edge consistency
    test_edge_route_consistency()

    # Prefix identity
    test_exact_prefix_parallel_channel()

    # Attempt ID
    test_attempt_id()

    # Invalid main API
    test_invalid_main_calls()

    # Failure positions
    test_failure_at_first_hop()
    test_failure_at_last_hop()

    # Candidate ordering
    test_candidate_ordering()

    # Amount propagation
    test_amount_propagation()

    # End-to-end
    test_end_to_end()

    print()
    print("=" * 82)
    print(" VALIDATION SUMMARY")
    print("=" * 82)

    print(
        f"Total : {TOTAL}"
    )

    print(
        f"Passed: {PASSED}"
    )

    print(
        f"Failed: {FAILED}"
    )

    print()

    if FAILED == 0:

        print(
            "PARTIAL BACKTRACKING VALIDATION: PASS"
        )

    else:

        print(
            "PARTIAL BACKTRACKING VALIDATION: FAIL"
        )

    print("=" * 82)

    return FAILED == 0


# ==============================================================
# Entry Point
# ==============================================================

if __name__ == "__main__":

    success = run_all_tests()

    raise SystemExit(
        0 if success else 1
    )