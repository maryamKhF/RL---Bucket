"""
Pathfinding/test_top_k_paths.py

Deterministic validation for channel-aware Top-K routing.

Validated invariants
--------------------

1. Graph validation
2. Unknown source / target
3. Invalid amount
4. Invalid k
5. k is fixed at 5
6. Invalid max_hops
7. Integer-valued float max_hops
8. Source == target
9. Offline source
10. Offline target
11. Invalid node Boolean
12. Invalid edge Boolean
13. Missing liquidity
14. Insufficient liquidity
15. Missing failure probability/history
16. Failure probability from history
17. Basic Top-K generation
18. Candidate result schema
19. Physical hop count
20. max_hops constraint
21. exact max_hops route
22. lambda_h=0 invariant
23. eta adaptive effect
24. custom heuristic
25. custom heuristic failure propagation
26. non-callable heuristic
27. MultiDiGraph parallel channels
28. Same node path / different channel identity
29. SCID preservation
30. Liquidity-based channel selection
31. Candidate ordering
32. No duplicate channel candidates
33. Fee preservation
34. Delay preservation
35. Reliability calculation
36. failure_probability calculation
37. No-path result
38. Deterministic repeated execution
"""

import math

import networkx as nx

from .top_k_paths import top_k_paths


# ==========================================================
# Constants
# ==========================================================

AMOUNT = 50.0
ETA_LOW = 0.0
ETA_HIGH = 1.0
LAMBDA_H = 1.0
MAX_HOPS = 4
K = 5


# ==========================================================
# Test Utilities
# ==========================================================

def _assert(
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def _expect_exception(
    fn,
    exception_type,
    name,
):
    try:
        fn()

    except exception_type:
        return

    except Exception as exc:
        raise AssertionError(
            f"{name}: expected "
            f"{exception_type.__name__}, "
            f"got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"{name}: expected "
        f"{exception_type.__name__}, "
        f"but no exception was raised"
    )


def _assert_close(
    actual,
    expected,
    name,
    tolerance=1e-9,
):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            f"{name}: expected {expected}, "
            f"got {actual}"
        )


# ==========================================================
# Node Factory
# ==========================================================

def _node(
    rgb=(50, 50, 50),
    latitude=0.0,
    longitude=0.0,
    country="US",
    continent="North America",
    available=True,
    online=True,
):
    r, g, b = rgb

    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "country": country,
        "continent": continent,

        # Provide all common RGB aliases used by the
        # routing / topology / heuristic modules.
        "rgb": [r, g, b],
        "rgb_color": [r, g, b],
        "color": f"#{r:02X}{g:02X}{b:02X}",
        "fill": f"#{r:02X}{g:02X}{b:02X}",

        "available": available,
        "online": online,
    }


# ==========================================================
# Edge Factory
# ==========================================================

def _edge(
    *,
    scid,
    liquidity=100.0,
    fee_base_msat=1000.0,
    fee_rate=10.0,
    delay=10.0,
    failure_probability=0.10,
    available=True,
):
    return {
        "scid": scid,

        "fee_base_msat": fee_base_msat,
        "fee_proportional_millionths": fee_rate,
        "cltv_expiry_delta": delay,

        "estimated_liquidity": liquidity,

        "failure_probability": failure_probability,

        "available": available,
    }


# ==========================================================
# Base MultiDiGraph
# ==========================================================

def _build_graph():
    """
    Build a deterministic directed Lightning-like graph.

    Topology:

                 AB-1
             A --------> B
              \          \
               \          \ BD-1
                \          \
                 \          > D
                  \
                   > C ----> D
                      CD-1

    Additional parallel channels:

        A -> B:
            AB-1
            AB-2

        B -> D:
            BD-1
            BD-2

    This gives several physically distinct candidates.
    """

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        **_node(
            rgb=(30, 30, 30),
            latitude=0.0,
            longitude=0.0,
            country="US",
            continent="North America",
        ),
    )

    G.add_node(
        "B",
        **_node(
            rgb=(60, 60, 60),
            latitude=1.0,
            longitude=1.0,
            country="US",
            continent="North America",
        ),
    )

    G.add_node(
        "C",
        **_node(
            rgb=(120, 120, 120),
            latitude=2.0,
            longitude=0.0,
            country="CA",
            continent="North America",
        ),
    )

    G.add_node(
        "D",
        **_node(
            rgb=(200, 200, 200),
            latitude=3.0,
            longitude=1.0,
            country="DE",
            continent="Europe",
        ),
    )

    # ------------------------------------------------------
    # A -> B
    # ------------------------------------------------------

    G.add_edge(
        "A",
        "B",
        key="AB-1",
        **_edge(
            scid="100x1x0",
            liquidity=100.0,
            fee_base_msat=1000.0,
            fee_rate=10.0,
            delay=10.0,
            failure_probability=0.10,
        ),
    )

    G.add_edge(
        "A",
        "B",
        key="AB-2",
        **_edge(
            scid="100x2x0",
            liquidity=100.0,
            fee_base_msat=2000.0,
            fee_rate=20.0,
            delay=20.0,
            failure_probability=0.20,
        ),
    )

    # ------------------------------------------------------
    # B -> D
    # ------------------------------------------------------

    G.add_edge(
        "B",
        "D",
        key="BD-1",
        **_edge(
            scid="200x1x0",
            liquidity=100.0,
            fee_base_msat=1000.0,
            fee_rate=10.0,
            delay=10.0,
            failure_probability=0.10,
        ),
    )

    G.add_edge(
        "B",
        "D",
        key="BD-2",
        **_edge(
            scid="200x2x0",
            liquidity=100.0,
            fee_base_msat=3000.0,
            fee_rate=30.0,
            delay=30.0,
            failure_probability=0.30,
        ),
    )

    # ------------------------------------------------------
    # A -> C -> D
    # ------------------------------------------------------

    G.add_edge(
        "A",
        "C",
        key="AC-1",
        **_edge(
            scid="300x1x0",
            liquidity=100.0,
            fee_base_msat=1500.0,
            fee_rate=15.0,
            delay=15.0,
            failure_probability=0.15,
        ),
    )

    G.add_edge(
        "C",
        "D",
        key="CD-1",
        **_edge(
            scid="400x1x0",
            liquidity=100.0,
            fee_base_msat=1500.0,
            fee_rate=15.0,
            delay=15.0,
            failure_probability=0.15,
        ),
    )

    return G


# ==========================================================
# 01 - Graph None
# ==========================================================

def test_graph_none():
    _expect_exception(
        lambda: top_k_paths(
            None,
            "A",
            "B",
            AMOUNT,
        ),
        ValueError,
        "graph None",
    )


# ==========================================================
# 02 - Invalid Graph Type
# ==========================================================

def test_invalid_graph():
    _expect_exception(
        lambda: top_k_paths(
            "not-a-graph",
            "A",
            "B",
            AMOUNT,
        ),
        TypeError,
        "invalid graph",
    )


# ==========================================================
# 03 - Unknown Source
# ==========================================================

def test_unknown_source():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "UNKNOWN",
            "D",
            AMOUNT,
        ),
        ValueError,
        "unknown source",
    )


# ==========================================================
# 04 - Unknown Target
# ==========================================================

def test_unknown_target():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "UNKNOWN",
            AMOUNT,
        ),
        ValueError,
        "unknown target",
    )


# ==========================================================
# 05 - Invalid Amount
# ==========================================================

def test_invalid_amount():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            0,
        ),
        ValueError,
        "invalid amount",
    )


# ==========================================================
# 06 - k Must Be 5
# ==========================================================

def test_k_fixed_at_five():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
            k=4,
        ),
        ValueError,
        "k != 5",
    )


# ==========================================================
# 07 - k=5 Accepted
# ==========================================================

def test_k_five_accepted():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        k=5,
    )

    _assert(
        isinstance(result, list),
        "k=5 must return a list",
    )


# ==========================================================
# 08 - Invalid max_hops
# ==========================================================

def test_invalid_max_hops():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
            max_hops=0,
        ),
        ValueError,
        "max_hops=0",
    )

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
            max_hops=2.5,
        ),
        ValueError,
        "non-integer max_hops",
    )


# ==========================================================
# 09 - Integer-valued Float max_hops
# ==========================================================

def test_integer_float_max_hops():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        max_hops=4.0,
    )

    _assert(
        isinstance(result, list),
        "integer-valued float max_hops must be accepted",
    )


# ==========================================================
# 10 - Source Equals Target
# ==========================================================

def test_source_equals_target():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "A",
        AMOUNT,
    )

    _assert(
        result == [],
        "source == target must return []",
    )


# ==========================================================
# 11 - Offline Source
# ==========================================================

def test_offline_source():
    G = _build_graph()

    G.nodes["A"]["online"] = False

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    _assert(
        result == [],
        "offline source must produce no candidates",
    )


# ==========================================================
# 12 - Offline Target
# ==========================================================

def test_offline_target():
    G = _build_graph()

    G.nodes["D"]["online"] = False

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    _assert(
        result == [],
        "offline target must produce no candidates",
    )


# ==========================================================
# 13 - Invalid Node Boolean
# ==========================================================

def test_invalid_node_boolean():
    G = _build_graph()

    G.nodes["A"]["online"] = "false"

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
        ),
        ValueError,
        "invalid node Boolean",
    )


# ==========================================================
# 14 - Invalid Edge Boolean
# ==========================================================

def test_invalid_edge_boolean():
    G = _build_graph()

    G["A"]["B"]["AB-1"]["available"] = "false"

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
        ),
        ValueError,
        "invalid edge Boolean",
    )


# ==========================================================
# 15 - Missing Liquidity
# ==========================================================

def test_missing_liquidity():
    G = _build_graph()

    del G["A"]["B"]["AB-1"]["estimated_liquidity"]

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
        ),
        ValueError,
        "missing liquidity",
    )


# ==========================================================
# 16 - Insufficient Liquidity
# ==========================================================

def test_insufficient_liquidity():
    G = _build_graph()

    G["A"]["B"]["AB-1"][
        "estimated_liquidity"
    ] = 10.0

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        for edge in candidate["edges"]:

            if (
                edge["source"] == "A"
                and edge["target"] == "B"
            ):
                _assert(
                    edge["channel_key"] != "AB-1",
                    "insufficient-liquidity channel "
                    "must not appear",
                )


# ==========================================================
# 17 - Missing Reliability
# ==========================================================

def test_missing_reliability():
    G = _build_graph()

    edge = G["A"]["B"]["AB-1"]

    del edge["failure_probability"]

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
        ),
        ValueError,
        "missing reliability",
    )


# ==========================================================
# 18 - Reliability From History
# ==========================================================

def test_reliability_from_history():
    G = _build_graph()

    edge = G["A"]["B"]["AB-1"]

    del edge["failure_probability"]

    edge["success_count"] = 90
    edge["failure_count"] = 10

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    candidate = result[0]

    first_edge = candidate["edges"][0]

    if first_edge["channel_key"] == "AB-1":

        _assert_close(
            candidate["reliability"],
            0.9 * 0.9,
            "path reliability",
        )


# ==========================================================
# 19 - Basic Top-K Generation
# ==========================================================

def test_basic_top_k():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    _assert(
        len(result) > 0,
        "Top-K must generate at least one candidate",
    )

    _assert(
        len(result) <= 5,
        "Top-K must never return more than 5 candidates",
    )


# ==========================================================
# 20 - Candidate Schema
# ==========================================================

def test_candidate_schema():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    required = {
        "path",
        "edges",
        "cost",
        "hop_count",
        "total_fee",
        "total_delay",
        "reliability",
        "failure_probability",
        "eta",
        "lambda_h",
        "raw_heuristic",
        "adaptive_penalty",
        "candidate",
        "success",
    }

    for candidate in result:

        missing = required.difference(
            candidate.keys()
        )

        _assert(
            not missing,
            f"candidate missing fields: {missing}",
        )


# ==========================================================
# 21 - Physical Hop Count
# ==========================================================

def test_physical_hop_count():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        _assert(
            candidate["hop_count"]
            == len(candidate["edges"]),
            "hop_count must equal physical channel count",
        )

        _assert(
            candidate["hop_count"]
            == len(candidate["path"]) - 1,
            "hop_count must equal physical node hops",
        )


# ==========================================================
# 22 - max_hops Blocks Long Routes
# ==========================================================

def test_max_hops_blocks_long_routes():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        max_hops=1,
    )

    _assert(
        result == [],
        "A->D requires at least two physical hops",
    )


# ==========================================================
# 23 - Exact max_hops Route
# ==========================================================

def test_max_hops_exact_route():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        max_hops=2,
    )

    _assert(
        len(result) > 0,
        "two-hop route must be allowed",
    )

    for candidate in result:
        _assert(
            candidate["hop_count"] <= 2,
            "candidate exceeds max_hops",
        )


# ==========================================================
# 24 - lambda_h = 0 Invariant
# ==========================================================

def test_lambda_zero_invariant():
    G = _build_graph()

    result_eta_0 = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        eta=0.0,
        lambda_h=0.0,
    )

    result_eta_1 = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        eta=1.0,
        lambda_h=0.0,
    )

    _assert(
        len(result_eta_0) == len(result_eta_1),
        "lambda_h=0 must preserve candidate count",
    )

    for a, b in zip(
        result_eta_0,
        result_eta_1,
    ):

        _assert(
            a["path"] == b["path"],
            "lambda_h=0 must preserve path",
        )

        _assert_close(
            a["cost"],
            b["cost"],
            "lambda_h=0 cost invariant",
        )


# ==========================================================
# 25 - Eta Adaptive Effect
# ==========================================================

def test_eta_adaptive_effect():
    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        **_node(
            rgb=(10, 10, 10),
            latitude=0,
            longitude=0,
        ),
    )

    G.add_node(
        "B",
        **_node(
            rgb=(200, 200, 200),
            latitude=1,
            longitude=1,
        ),
    )

    G.add_edge(
        "A",
        "B",
        key="AB",
        **_edge(
            scid="500x1x0",
            liquidity=100,
        ),
    )

    result_0 = top_k_paths(
        G,
        "A",
        "B",
        AMOUNT,
        eta=0.0,
        lambda_h=1.0,
        max_hops=1,
    )

    result_1 = top_k_paths(
        G,
        "A",
        "B",
        AMOUNT,
        eta=1.0,
        lambda_h=1.0,
        max_hops=1,
    )

    _assert(
        len(result_0) == 1,
        "eta=0 must produce one candidate",
    )

    _assert(
        len(result_1) == 1,
        "eta=1 must produce one candidate",
    )

    _assert(
        result_0[0]["path"]
        == result_1[0]["path"],
        "single-edge topology must preserve path",
    )

    _assert(
        not math.isclose(
            result_0[0]["cost"],
            result_1[0]["cost"],
        ),
        "eta must have observable effect when lambda_h > 0",
    )


# ==========================================================
# 26 - Custom Heuristic
# ==========================================================

def test_custom_heuristic():
    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        **_node(
            rgb=(50, 50, 50),
        ),
    )

    G.add_node(
        "B",
        **_node(
            rgb=(50, 50, 50),
        ),
    )

    G.add_edge(
        "A",
        "B",
        key="AB",
        **_edge(
            scid="600x1x0",
            liquidity=100,
        ),
    )

    def custom_cost(
        G,
        u,
        v,
        data,
        amount,
    ):
        return 123.0

    result = top_k_paths(
        G,
        "A",
        "B",
        AMOUNT,
        heuristic_fn=custom_cost,
        lambda_h=0.0,
        max_hops=1,
    )

    _assert(
        len(result) == 1,
        "custom heuristic test must produce one candidate",
    )

    _assert_close(
        result[0]["cost"],
        123.0,
        "custom heuristic cost",
    )


# ==========================================================
# 27 - Custom Heuristic Failure Must Propagate
# ==========================================================

def test_custom_heuristic_no_fallback():
    G = _build_graph()

    def failing_heuristic(
        G,
        u,
        v,
        data,
        amount,
    ):
        raise RuntimeError(
            "intentional heuristic failure"
        )

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
            heuristic_fn=failing_heuristic,
        ),
        RuntimeError,
        "custom heuristic no fallback",
    )


# ==========================================================
# 28 - Non-callable Heuristic
# ==========================================================

def test_non_callable_heuristic():
    G = _build_graph()

    _expect_exception(
        lambda: top_k_paths(
            G,
            "A",
            "D",
            AMOUNT,
            heuristic_fn=123,
        ),
        TypeError,
        "non-callable heuristic",
    )


# ==========================================================
# 29 - MultiDiGraph Channel Identity
# ==========================================================

def test_multidigraph_channel_identity():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    identities = []

    for candidate in result:

        identity = tuple(
            (
                edge["source"],
                edge["target"],
                edge["channel_key"],
                edge["scid"],
            )
            for edge in candidate["edges"]
        )

        identities.append(identity)

    _assert(
        len(identities)
        == len(set(identities)),
        "channel-aware candidates must be unique",
    )


# ==========================================================
# 30 - Same Node Path / Different Channels
# ==========================================================

def test_same_node_path_different_channels():
    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        **_node(),
    )

    G.add_node(
        "B",
        **_node(),
    )

    # Two physically distinct channels.
    G.add_edge(
        "A",
        "B",
        key="AB-1",
        **_edge(
            scid="700x1x0",
            liquidity=100,
            fee_base_msat=1000,
        ),
    )

    G.add_edge(
        "A",
        "B",
        key="AB-2",
        **_edge(
            scid="700x2x0",
            liquidity=100,
            fee_base_msat=2000,
        ),
    )

    result = top_k_paths(
        G,
        "A",
        "B",
        AMOUNT,
        max_hops=1,
    )

    _assert(
        len(result) == 2,
        "parallel physical channels must produce "
        "two distinct candidates",
    )

    keys = {
        candidate["edges"][0]["channel_key"]
        for candidate in result
    }

    _assert(
        keys == {"AB-1", "AB-2"},
        "both physical channel keys must be preserved",
    )


# ==========================================================
# 31 - SCID Preservation
# ==========================================================

def test_scid_preservation():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        for edge in candidate["edges"]:

            original = G[
                edge["source"]
            ][
                edge["target"]
            ][
                edge["channel_key"]
            ]

            _assert(
                edge["scid"]
                == original["scid"],
                "SCID was not preserved",
            )


# ==========================================================
# 32 - Liquidity-Based Channel Selection
# ==========================================================

def test_liquidity_channel_selection():
    G = nx.MultiDiGraph()

    G.add_node("A", **_node())
    G.add_node("B", **_node())

    # Cheap but insufficient.
    G.add_edge(
        "A",
        "B",
        key="LOW-LIQ",
        **_edge(
            scid="800x1x0",
            liquidity=10,
            fee_base_msat=1,
        ),
    )

    # More expensive but feasible.
    G.add_edge(
        "A",
        "B",
        key="HIGH-LIQ",
        **_edge(
            scid="800x2x0",
            liquidity=100,
            fee_base_msat=1000,
        ),
    )

    result = top_k_paths(
        G,
        "A",
        "B",
        AMOUNT,
        max_hops=1,
    )

    _assert(
        len(result) == 1,
        "only one channel should remain feasible",
    )

    _assert(
        result[0]["edges"][0]["channel_key"]
        == "HIGH-LIQ",
        "insufficient-liquidity channel was selected",
    )


# ==========================================================
# 33 - Candidate Ordering
# ==========================================================

def test_candidate_ordering():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for previous, current in zip(
        result,
        result[1:],
    ):

        previous_key = (
            previous["cost"],
            previous["hop_count"],
            previous["total_fee"],
            previous["total_delay"],
            -previous["reliability"],
            tuple(previous["path"]),
        )

        current_key = (
            current["cost"],
            current["hop_count"],
            current["total_fee"],
            current["total_delay"],
            -current["reliability"],
            tuple(current["path"]),
        )

        _assert(
            previous_key <= current_key,
            "candidate ordering is not deterministic",
        )


# ==========================================================
# 34 - No Duplicate Channel Candidates
# ==========================================================

def test_no_duplicate_channel_candidates():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    identities = []

    for candidate in result:

        identity = tuple(
            (
                edge["source"],
                edge["target"],
                edge["channel_key"],
                edge["scid"],
            )
            for edge in candidate["edges"]
        )

        identities.append(identity)

    _assert(
        len(identities)
        == len(set(identities)),
        "duplicate physical-channel candidates detected",
    )


# ==========================================================
# 35 - Fee Preservation
# ==========================================================

def test_fee_preservation():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        expected_fee = 0.0

        for edge in candidate["edges"]:

            data = G[
                edge["source"]
            ][
                edge["target"]
            ][
                edge["channel_key"]
            ]

            base = float(
                data["fee_base_msat"]
            )

            proportional = (
                AMOUNT
                * float(
                    data[
                        "fee_proportional_millionths"
                    ]
                )
                / 1_000_000.0
            )

            expected_fee += (
                base + proportional
            )

        _assert_close(
            candidate["total_fee"],
            expected_fee,
            "total fee",
        )


# ==========================================================
# 36 - Delay Preservation
# ==========================================================

def test_delay_preservation():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        expected_delay = 0.0

        for edge in candidate["edges"]:

            data = G[
                edge["source"]
            ][
                edge["target"]
            ][
                edge["channel_key"]
            ]

            expected_delay += float(
                data["cltv_expiry_delta"]
            )

        _assert_close(
            candidate["total_delay"],
            expected_delay,
            "total delay",
        )


# ==========================================================
# 37 - Reliability
# ==========================================================

def test_reliability():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        expected = 1.0

        for edge in candidate["edges"]:

            data = G[
                edge["source"]
            ][
                edge["target"]
            ][
                edge["channel_key"]
            ]

            reliability = (
                1.0
                - float(
                    data["failure_probability"]
                )
            )

            expected *= reliability

        _assert_close(
            candidate["reliability"],
            expected,
            "path reliability",
        )


# ==========================================================
# 38 - Failure Probability
# ==========================================================

def test_failure_probability():
    G = _build_graph()

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    for candidate in result:

        expected = (
            1.0
            - candidate["reliability"]
        )

        _assert_close(
            candidate["failure_probability"],
            expected,
            "failure probability",
        )


# ==========================================================
# 39 - No Feasible Path
# ==========================================================

def test_no_feasible_path():
    G = _build_graph()

    for u, v, key in G.edges(
        keys=True
    ):
        G[u][v][key][
            "estimated_liquidity"
        ] = 0.0

    result = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
    )

    _assert(
        result == [],
        "no feasible path must return []",
    )


# ==========================================================
# 40 - Deterministic Repeated Execution
# ==========================================================

def test_deterministic_execution():
    G = _build_graph()

    result_1 = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        eta=0.4,
        lambda_h=1.0,
    )

    result_2 = top_k_paths(
        G,
        "A",
        "D",
        AMOUNT,
        eta=0.4,
        lambda_h=1.0,
    )

    _assert(
        result_1 == result_2,
        "repeated execution must produce identical results",
    )


# ==========================================================
# Test Registry
# ==========================================================

TESTS = [
    (
        "graph None validation",
        test_graph_none,
    ),
    (
        "invalid graph type",
        test_invalid_graph,
    ),
    (
        "unknown source",
        test_unknown_source,
    ),
    (
        "unknown target",
        test_unknown_target,
    ),
    (
        "invalid amount",
        test_invalid_amount,
    ),
    (
        "k fixed at 5",
        test_k_fixed_at_five,
    ),
    (
        "k=5 accepted",
        test_k_five_accepted,
    ),
    (
        "invalid max_hops",
        test_invalid_max_hops,
    ),
    (
        "integer-valued float max_hops",
        test_integer_float_max_hops,
    ),
    (
        "source equals target",
        test_source_equals_target,
    ),
    (
        "offline source",
        test_offline_source,
    ),
    (
        "offline target",
        test_offline_target,
    ),
    (
        "invalid node Boolean",
        test_invalid_node_boolean,
    ),
    (
        "invalid edge Boolean",
        test_invalid_edge_boolean,
    ),
    (
        "missing liquidity",
        test_missing_liquidity,
    ),
    (
        "insufficient liquidity",
        test_insufficient_liquidity,
    ),
    (
        "missing reliability",
        test_missing_reliability,
    ),
    (
        "reliability from history",
        test_reliability_from_history,
    ),
    (
        "basic Top-K generation",
        test_basic_top_k,
    ),
    (
        "candidate result schema",
        test_candidate_schema,
    ),
    (
        "physical hop count",
        test_physical_hop_count,
    ),
    (
        "max_hops blocks long routes",
        test_max_hops_blocks_long_routes,
    ),
    (
        "max_hops exact route",
        test_max_hops_exact_route,
    ),
    (
        "lambda_h zero invariant",
        test_lambda_zero_invariant,
    ),
    (
        "eta adaptive effect",
        test_eta_adaptive_effect,
    ),
    (
        "custom heuristic",
        test_custom_heuristic,
    ),
    (
        "custom heuristic no fallback",
        test_custom_heuristic_no_fallback,
    ),
    (
        "non-callable heuristic",
        test_non_callable_heuristic,
    ),
    (
        "MultiDiGraph channel identity",
        test_multidigraph_channel_identity,
    ),
    (
        "same node path / different channels",
        test_same_node_path_different_channels,
    ),
    (
        "SCID preservation",
        test_scid_preservation,
    ),
    (
        "liquidity channel selection",
        test_liquidity_channel_selection,
    ),
    (
        "candidate ordering",
        test_candidate_ordering,
    ),
    (
        "no duplicate channel candidates",
        test_no_duplicate_channel_candidates,
    ),
    (
        "fee preservation",
        test_fee_preservation,
    ),
    (
        "delay preservation",
        test_delay_preservation,
    ),
    (
        "reliability",
        test_reliability,
    ),
    (
        "failure probability",
        test_failure_probability,
    ),
    (
        "no feasible path",
        test_no_feasible_path,
    ),
    (
        "deterministic repeated execution",
        test_deterministic_execution,
    ),
]


# ==========================================================
# Test Runner
# ==========================================================

def main():
    print("=" * 72)
    print(" TOP-K CHANNEL-AWARE DETERMINISTIC VALIDATION")
    print("=" * 72)

    passed = 0
    failed = 0

    for index, (
        name,
        test_fn,
    ) in enumerate(
        TESTS,
        start=1,
    ):

        try:

            test_fn()

            print(
                f"[{index:02d}/{len(TESTS):02d}] "
                f"{name:<42} PASS"
            )

            passed += 1

        except Exception as exc:

            print(
                f"[{index:02d}/{len(TESTS):02d}] "
                f"{name:<42} FAIL"
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
            "TOP-K VALIDATION: PASS"
        )
        return 0

    print(
        "TOP-K VALIDATION: FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )