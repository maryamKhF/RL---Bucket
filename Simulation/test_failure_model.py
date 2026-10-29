"""
Simulation/test_failure_model.py

Comprehensive deterministic validation suite for FailureModel.

Run:
    py -m Simulation.test_failure_model

Scope
-----
This test validates only FailureModel.

It does NOT test:
    - PPO
    - routing
    - Top-K
    - Bucket
    - PaymentSimulator
    - Partial Backtracking

Main invariants
---------------
- Exact MultiDiGraph channel identity.
- No automatic parallel-channel selection.
- capacity is not directional liquidity.
- Unknown liquidity is not treated as zero.
- Malformed explicit values are rejected.
- Node stochastic state is evaluated at most once per payment.
- Failure index is exact.
- Failed edge is exact.
- Visited edges contain only successful hops.
- Runtime failure state is reset correctly.
- RNG behavior is deterministic for the same seed.
- Geographic fallback is tested only with genuinely distant
  cross-region coordinates.
"""


from __future__ import annotations

import math

import networkx as nx

from .failure_model import (
    FailureModel,
    assign_failure_probabilities,
    _intercontinental,
)


# ============================================================
# Test Counters
# ============================================================

TOTAL = 0
PASSED = 0
FAILED = 0


# ============================================================
# Test Helpers
# ============================================================

def check(name, condition):
    """
    Register one test result.
    """

    global TOTAL
    global PASSED
    global FAILED

    TOTAL += 1

    if condition:
        PASSED += 1

        print(
            f"[{TOTAL:02d}] "
            f"{name:<58} PASS"
        )

    else:
        FAILED += 1

        print(
            f"[{TOTAL:02d}] "
            f"{name:<58} FAIL"
        )


def expect_exception(
    exception_type,
    function,
):
    """
    Return True only when the expected exception is raised.
    """

    try:
        function()

    except exception_type:
        return True

    except Exception:
        return False

    return False


# ============================================================
# Graph Builders
# ============================================================

def build_multidigraph():
    """
    Build deterministic MultiDiGraph fixture.

    A -> B has two parallel channels:

        key=0 : balance 5000
        key=1 : balance 10000

    B -> C:

        unknown directional liquidity

    C -> D:

        balance 10000

    Country metadata:

        A = US
        B = US
        C = FR
        D = DE

    Coordinates are real geographic locations and are used only
    when country-based classification is unavailable.
    """

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=40.0,
        longitude=-74.0,
    )

    G.add_node(
        "B",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=41.0,
        longitude=-73.0,
    )

    G.add_node(
        "C",
        available=True,
        is_online=True,
        online=True,
        country="FR",
        latitude=48.0,
        longitude=2.0,
    )

    G.add_node(
        "D",
        available=True,
        is_online=True,
        online=True,
        country="DE",
        latitude=52.0,
        longitude=13.0,
    )

    G.add_edge(
        "A",
        "B",
        key=0,
        capacity=10000,
        balance_uv=5000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    G.add_edge(
        "A",
        "B",
        key=1,
        capacity=10000,
        balance_uv=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    G.add_edge(
        "B",
        "C",
        key=0,
        capacity=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    G.add_edge(
        "C",
        "D",
        key=0,
        capacity=10000,
        balance_uv=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    return G


def build_digraph():
    """
    Build deterministic DiGraph fixture.
    """

    G = nx.DiGraph()

    G.add_node(
        "A",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=40.0,
        longitude=-74.0,
    )

    G.add_node(
        "B",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=41.0,
        longitude=-73.0,
    )

    G.add_edge(
        "A",
        "B",
        capacity=10000,
        balance_uv=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    return G


def build_graph():
    """
    Build deterministic undirected Graph fixture.
    """

    G = nx.Graph()

    G.add_node(
        "A",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=40.0,
        longitude=-74.0,
    )

    G.add_node(
        "B",
        available=True,
        is_online=True,
        online=True,
        country="US",
        latitude=41.0,
        longitude=-73.0,
    )

    G.add_edge(
        "A",
        "B",
        capacity=10000,
        balance_uv=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    return G


# ============================================================
# Main Test Suite
# ============================================================

def main():

    global TOTAL
    global PASSED
    global FAILED

    print()
    print("=" * 78)
    print("COMPREHENSIVE FAILURE MODEL VALIDATION")
    print("=" * 78)

    # ========================================================
    # 1. Constructor
    # ========================================================

    model = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    check(
        "constructor accepts valid configuration",
        (
            model.node_failure_probability == 0.0
            and model.liquidity_failure_probability == 0.0
            and model.seed == 42
        ),
    )

    check(
        "constructor rejects node probability > 1",
        expect_exception(
            ValueError,
            lambda: FailureModel(
                node_failure_probability=1.1,
                liquidity_failure_probability=0.0,
                seed=42,
            ),
        ),
    )

    check(
        "constructor rejects negative node probability",
        expect_exception(
            ValueError,
            lambda: FailureModel(
                node_failure_probability=-0.1,
                liquidity_failure_probability=0.0,
                seed=42,
            ),
        ),
    )

    check(
        "constructor rejects liquidity probability > 1",
        expect_exception(
            ValueError,
            lambda: FailureModel(
                node_failure_probability=0.0,
                liquidity_failure_probability=1.1,
                seed=42,
            ),
        ),
    )

    check(
        "constructor rejects negative liquidity probability",
        expect_exception(
            ValueError,
            lambda: FailureModel(
                node_failure_probability=0.0,
                liquidity_failure_probability=-0.1,
                seed=42,
            ),
        ),
    )

    check(
        "constructor rejects boolean seed",
        expect_exception(
            TypeError,
            lambda: FailureModel(
                node_failure_probability=0.0,
                liquidity_failure_probability=0.0,
                seed=True,
            ),
        ),
    )

    # ========================================================
    # 2. Amount Validation
    # ========================================================

    G = build_multidigraph()

    invalid_amounts = [
        0,
        -1,
        None,
        float("nan"),
        float("inf"),
        True,
    ]

    amount_results = []

    for amount in invalid_amounts:

        result = model.evaluate_payment_failure(
            route=[
                "A",
                "B",
            ],
            amount=amount,
            network=G,
            route_edges=[
                ("A", "B", 1),
            ],
        )

        amount_results.append(
            result["success"] is False
            and result["reason"] == "invalid_amount"
        )

    check(
        "all invalid amounts are rejected",
        all(amount_results),
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "positive finite amount is accepted",
        result["success"] is True,
    )

    # ========================================================
    # 3. MultiDiGraph Channel Identity
    # ========================================================

    G = build_multidigraph()

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B"),
        ],
    )

    check(
        "MultiDiGraph requires exact channel key",
        result["reason"] == "missing_channel_key",
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "exact parallel channel succeeds",
        result["success"] is True,
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=None,
    )

    check(
        "MultiDiGraph does not auto-select channel",
        result["reason"] == "invalid_route_edges",
    )

    # ========================================================
    # 4. Graph Type Semantics
    # ========================================================

    G = build_digraph()

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", None),
        ],
    )

    check(
        "DiGraph accepts key=None",
        result["success"] is True,
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 0),
        ],
    )

    check(
        "DiGraph rejects unexpected key",
        result["reason"] == "unexpected_channel_key",
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=None,
    )

    check(
        "DiGraph resolves unambiguous path",
        result["success"] is True,
    )

    G = build_graph()

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=None,
    )

    check(
        "Graph resolves unambiguous path",
        result["success"] is True,
    )

    # ========================================================
    # 5. Directional Liquidity
    # ========================================================

    G = build_multidigraph()

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=5001,
        network=G,
        route_edges=[
            ("A", "B", 0),
        ],
    )

    check(
        "insufficient balance causes liquidity failure",
        (
            result["success"] is False
            and result["reason"] == "liquidity_failure"
        ),
    )

    result = model.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=5001,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "sufficient balance permits forwarding",
        result["success"] is True,
    )

    result = model.evaluate_payment_failure(
        route=[
            "B",
            "C",
        ],
        amount=5000,
        network=G,
        route_edges=[
            ("B", "C", 0),
        ],
    )

    check(
        "unknown liquidity is not zero",
        result["success"] is True,
    )

    G["B"]["C"][0]["capacity"] = 100

    result = model.evaluate_payment_failure(
        route=[
            "B",
            "C",
        ],
        amount=5000,
        network=G,
        route_edges=[
            ("B", "C", 0),
        ],
    )

    check(
        "capacity is not directional liquidity",
        result["success"] is True,
    )

    # ========================================================
    # 6. Supported Liquidity Fields
    # ========================================================

    liquidity_fields = [
        "balance_uv",
        "liquidity_uv",
        "liquidity",
        "estimated_liquidity",
    ]

    field_results = []

    for field in liquidity_fields:

        G = build_multidigraph()

        edge = G["A"]["B"][1]

        for other in liquidity_fields:
            edge.pop(
                other,
                None,
            )

        edge[field] = 1000

        m = FailureModel(
            node_failure_probability=0.0,
            liquidity_failure_probability=0.0,
            seed=42,
        )

        result = m.evaluate_payment_failure(
            route=[
                "A",
                "B",
            ],
            amount=900,
            network=G,
            route_edges=[
                ("A", "B", 1),
            ],
        )

        field_results.append(
            result["success"] is True
        )

    check(
        "all supported liquidity fields work",
        all(field_results),
    )

    # ========================================================
    # 7. Malformed Liquidity
    # ========================================================

    G = build_multidigraph()

    G["A"]["B"][1]["balance_uv"] = "invalid"

    check(
        "string liquidity is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    G = build_multidigraph()

    G["A"]["B"][1]["balance_uv"] = -1

    check(
        "negative liquidity is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    G = build_multidigraph()

    G["A"]["B"][1]["balance_uv"] = float("nan")

    check(
        "non-finite liquidity is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    # ========================================================
    # 8. Channel Failure Probability
    # ========================================================

    G = build_multidigraph()

    G["A"]["B"][1]["failure_probability"] = 1.0

    m = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "probability 1 always causes channel failure",
        (
            result["success"] is False
            and result["reason"] == "channel_failure"
        ),
    )

    G = build_multidigraph()

    del G["A"]["B"][1]["failure_probability"]

    check(
        "missing failure probability is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    G = build_multidigraph()

    G["A"]["B"][1]["failure_probability"] = "invalid"

    check(
        "malformed failure probability is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    G = build_multidigraph()

    G["A"]["B"][1]["failure_probability"] = 1.5

    check(
        "out-of-range failure probability is rejected",
        expect_exception(
            ValueError,
            lambda: model.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=G,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    # ========================================================
    # 9. Node Failure
    # ========================================================

    G = build_multidigraph()

    m = FailureModel(
        node_failure_probability=1.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "node probability 1 causes node failure",
        (
            result["success"] is False
            and result["reason"] == "node_failure"
        ),
    )

    G = build_multidigraph()

    G.nodes["B"]["available"] = False

    m = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "explicit unavailable node causes failure",
        result["reason"] == "node_failure",
    )

    # ========================================================
    # 10. Node State Cache
    # ========================================================

    G = build_multidigraph()

    G.add_edge(
        "B",
        "A",
        key=0,
        capacity=10000,
        balance_uv=10000,
        failure_probability=0.0,
        available=True,
        failure_count=0,
        last_failure=None,
    )

    m = FailureModel(
        node_failure_probability=0.5,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    m._node_attempt_cache = {}

    first = m.check_node_failure(
        G,
        "B",
    )

    second = m.check_node_failure(
        G,
        "B",
    )

    check(
        "node result is cached within one payment",
        first == second,
    )

    m._node_attempt_cache = None

    # ========================================================
    # 11. Failure Propagation
    # ========================================================

    G = build_multidigraph()

    G["C"]["D"][0]["failure_probability"] = 1.0

    m = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
            "C",
            "D",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
            ("B", "C", 0),
            ("C", "D", 0),
        ],
    )

    check(
        "failed edge is exact",
        result["failed_edge"]
        == (
            "C",
            "D",
            0,
        ),
    )

    check(
        "failure index is exact",
        result["failure_index"] == 2,
    )

    check(
        "visited edges contain successful hops only",
        result["visited_edges"]
        == [
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    # ========================================================
    # 12. Route Validation
    # ========================================================

    G = build_multidigraph()

    result = m.evaluate_payment_failure(
        route=[
            "A",
        ],
        amount=100,
        network=G,
        route_edges=[],
    )

    check(
        "route with one node is rejected",
        result["reason"] == "invalid_route",
    )

    result = m.evaluate_payment_failure(
        route="A-B",
        amount=100,
        network=G,
        route_edges=[],
    )

    check(
        "non-list route is rejected",
        result["reason"] == "invalid_route",
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "C",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "edge/path mismatch is detected",
        result["reason"] == "edge_path_mismatch",
    )

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    check(
        "route-edge count mismatch is detected",
        result["reason"] == "invalid_route_edges",
    )

    # ========================================================
    # 13. Missing Channel
    # ========================================================

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "D",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "D", 0),
        ],
    )

    check(
        "missing exact channel is detected",
        result["reason"] == "missing_channel",
    )

    # ========================================================
    # 14. Channel Availability
    # ========================================================

    G = build_multidigraph()

    G["A"]["B"][1]["available"] = False

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "unavailable channel causes channel failure",
        result["reason"] == "channel_failure",
    )

    # ========================================================
    # 15. Runtime Failure State
    # ========================================================

    G = build_multidigraph()

    G["A"]["B"][1]["failure_probability"] = 1.0

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
        ],
    )

    check(
        "failed channel becomes unavailable",
        G["A"]["B"][1]["available"] is False,
    )

    check(
        "failure count increments",
        G["A"]["B"][1]["failure_count"] == 1,
    )

    check(
        "last failure timestamp is recorded",
        G["A"]["B"][1]["last_failure"] is not None,
    )

    # ========================================================
    # 16. Reset Runtime State
    # ========================================================

    m.reset_runtime_state(
        G,
        reset_counters=True,
        reset_rng=True,
    )

    check(
        "reset restores channel availability",
        G["A"]["B"][1]["available"] is True,
    )

    check(
        "reset clears failure count",
        G["A"]["B"][1]["failure_count"] == 0,
    )

    check(
        "reset clears last failure",
        G["A"]["B"][1]["last_failure"] is None,
    )

    # ========================================================
    # 17. Deterministic RNG
    # ========================================================

    def run_sequence(seed):

        G = build_multidigraph()

        G["A"]["B"][0]["failure_probability"] = 0.5

        m = FailureModel(
            node_failure_probability=0.0,
            liquidity_failure_probability=0.0,
            seed=seed,
        )

        sequence = []

        for _ in range(20):

            G["A"]["B"][0]["available"] = True

            sequence.append(
                m.check_channel_failure(
                    G,
                    "A",
                    "B",
                    0,
                )
            )

        return sequence

    seq1 = run_sequence(123)
    seq2 = run_sequence(123)

    check(
        "same seed produces identical sequence",
        seq1 == seq2,
    )

    seq3 = run_sequence(456)

    check(
        "different seed changes stochastic sequence",
        seq1 != seq3,
    )

    # ========================================================
    # 18. assign_failure_probabilities
    # ========================================================

    G1 = build_multidigraph()
    G2 = build_multidigraph()

    assign_failure_probabilities(
        G1,
        average_rate=0.2,
        seed=99,
    )

    assign_failure_probabilities(
        G2,
        average_rate=0.2,
        seed=99,
    )

    probabilities_1 = [
        G1[u][v][key]["simulator_failure_probability"]
        for u, v, key in G1.edges(
            keys=True
        )
    ]

    probabilities_2 = [
        G2[u][v][key]["simulator_failure_probability"]
        for u, v, key in G2.edges(
            keys=True
        )
    ]

    check(
        "failure probability assignment is deterministic",
        probabilities_1 == probabilities_2,
    )

    check(
        "assigned probabilities are finite and bounded",
        all(
            math.isfinite(value)
            and 0.0 <= value <= 0.95
            for value in probabilities_1
        ),
    )

    check(
        "latent probability is not exposed as routing feature",
        all(
            "failure_probability" not in G1[u][v][key]
            for u, v, key in G1.edges(keys=True)
        ),
    )

    check(
        "assignment initializes availability",
        all(
            G1[u][v][key]["available"] is True
            for u, v, key in G1.edges(
                keys=True
            )
        ),
    )

    check(
        "assignment initializes failure counters",
        all(
            G1[u][v][key]["failure_count"] == 0
            for u, v, key in G1.edges(
                keys=True
            )
        ),
    )

    check(
        "assignment initializes last_failure",
        all(
            G1[u][v][key]["last_failure"] is None
            for u, v, key in G1.edges(
                keys=True
            )
        ),
    )

    # ========================================================
    # 19. assign_failure_probabilities validation
    # ========================================================

    check(
        "assignment rejects average rate > 1",
        expect_exception(
            ValueError,
            lambda: assign_failure_probabilities(
                build_multidigraph(),
                average_rate=1.1,
                seed=42,
            ),
        ),
    )

    check(
        "assignment rejects negative average rate",
        expect_exception(
            ValueError,
            lambda: assign_failure_probabilities(
                build_multidigraph(),
                average_rate=-0.1,
                seed=42,
            ),
        ),
    )

    check(
        "assignment rejects boolean seed",
        expect_exception(
            TypeError,
            lambda: assign_failure_probabilities(
                build_multidigraph(),
                average_rate=0.1,
                seed=True,
            ),
        ),
    )

    check(
        "assignment rejects None graph",
        expect_exception(
            ValueError,
            lambda: assign_failure_probabilities(
                None,
                average_rate=0.1,
                seed=42,
            ),
        ),
    )

    # ========================================================
    # 20. Geography
    # ========================================================

    G = build_multidigraph()

    check(
        "same-country nodes are not intercontinental",
        _intercontinental(
            G,
            "A",
            "B",
        ) is False,
    )

    check(
        "US-France is intercontinental",
        _intercontinental(
            G,
            "B",
            "C",
        ) is True,
    )

    check(
        "France-Germany are not intercontinental",
        _intercontinental(
            G,
            "C",
            "D",
        ) is False,
    )

    # ========================================================
    # 21. Geographic Fallback
    # ========================================================
    #
    # IMPORTANT:
    # Country information is deliberately removed.
    #
    # C is moved to Sydney, Australia.
    # D remains in Berlin, Germany.
    #
    # Therefore the fallback has genuinely distant coordinates
    # spanning different geographic regions.
    #
    # This test must NOT use France/Germany coordinates because
    # those locations are on the same continent and would make
    # the expected result ambiguous/incorrect.
    # ========================================================

    G = build_multidigraph()

    G.nodes["C"].pop(
        "country",
        None,
    )

    G.nodes["D"].pop(
        "country",
        None,
    )

    G.nodes["C"]["latitude"] = -33.8688
    G.nodes["C"]["longitude"] = 151.2093

    G.nodes["D"]["latitude"] = 52.5200
    G.nodes["D"]["longitude"] = 13.4050

    check(
        "geographic fallback identifies distant nodes",
        _intercontinental(
            G,
            "C",
            "D",
        ) is True,
    )

    # ========================================================
    # 22. Unknown Geography
    # ========================================================

    G = build_multidigraph()

    G.nodes["C"].pop(
        "country",
        None,
    )

    G.nodes["C"].pop(
        "latitude",
        None,
    )

    G.nodes["C"].pop(
        "longitude",
        None,
    )

    G.nodes["D"].pop(
        "country",
        None,
    )

    G.nodes["D"].pop(
        "latitude",
        None,
    )

    G.nodes["D"].pop(
        "longitude",
        None,
    )

    check(
        "unknown geography is handled safely",
        _intercontinental(
            G,
            "C",
            "D",
        ) is False,
    )

    # ========================================================
    # 23. Result Integrity
    # ========================================================

    G = build_multidigraph()

    m = FailureModel(
        node_failure_probability=0.0,
        liquidity_failure_probability=0.0,
        seed=42,
    )

    route = [
        "A",
        "B",
        "C",
    ]

    route_edges = [
        ("A", "B", 1),
        ("B", "C", 0),
    ]

    original_route = list(route)
    original_edges = list(route_edges)

    result = m.evaluate_payment_failure(
        route=route,
        amount=100,
        network=G,
        route_edges=route_edges,
    )

    check(
        "successful payment has success=True",
        result["success"] is True,
    )

    check(
        "successful payment has reason='success'",
        result["reason"] == "success",
    )

    check(
        "successful payment has failed_edge=None",
        result["failed_edge"] is None,
    )

    check(
        "successful payment has failure_index=None",
        result["failure_index"] is None,
    )

    check(
        "successful payment visits all edges",
        result["visited_edges"] == original_edges,
    )

    check(
        "route input is not mutated",
        route == original_route,
    )

    check(
        "route_edges input is not mutated",
        route_edges == original_edges,
    )

    # ========================================================
    # 24. Exact End-to-End Channel Identity
    # ========================================================

    G = build_multidigraph()

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "B",
            "C",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    check(
        "exact channel identity preserved end-to-end",
        result["visited_edges"]
        == [
            ("A", "B", 1),
            ("B", "C", 0),
        ],
    )

    # ========================================================
    # 25. Missing Nodes / Missing Network
    # ========================================================

    result = m.evaluate_payment_failure(
        route=[
            "A",
            "X",
        ],
        amount=100,
        network=G,
        route_edges=[
            ("A", "X", 0),
        ],
    )

    check(
        "missing channel involving unknown node is detected",
        result["reason"] == "missing_channel",
    )

    check(
        "None network is rejected",
        expect_exception(
            ValueError,
            lambda: m.evaluate_payment_failure(
                route=["A", "B"],
                amount=100,
                network=None,
                route_edges=[
                    ("A", "B", 1),
                ],
            ),
        ),
    )

    # ========================================================
    # Final Summary
    # ========================================================

    print()
    print("=" * 78)
    print(
        f"Total: {TOTAL}, "
        f"Passed: {PASSED}, "
        f"Failed: {FAILED}"
    )
    print("=" * 78)

    if FAILED == 0:
        print(
            "FAILURE MODEL VALIDATION: PASS"
        )
    else:
        print(
            "FAILURE MODEL VALIDATION: FAIL"
        )

    return FAILED


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
