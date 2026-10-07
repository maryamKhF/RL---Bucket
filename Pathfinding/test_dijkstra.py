"""
Pathfinding/test_dijkstra.py

Deterministic validation suite for the Dijkstra routing module.

The suite validates:

1. Graph validation
2. Source/target validation
3. Amount validation
4. Hop constraints
5. Node/edge availability
6. Liquidity feasibility
7. Failure-probability handling
8. Native LND cost
9. Adaptive eta behavior
10. Lambda-h invariant
11. Custom heuristic behavior
12. MultiDiGraph channel identity
13. Geographic metrics
14. Success/failure result schemas

Run:

    py -m Pathfinding.test_dijkstra
"""

import math
import networkx as nx

from Pathfinding.dijkstra import Dijkstra


# ============================================================================
# CONSTANTS
# ============================================================================

REQUIRED_RESULT_FIELDS = {
    "success",
    "path",
    "edges",
    "cost",
    "hop_count",
    "total_fee",
    "total_delay",
    "min_liquidity",
    "reliability",
    "failure_probability",
    "total_distance_km",
    "total_carbon",
    "inter_country_hops",
    "inter_continent_hops",
    "eta",
    "lambda_h",
    "candidate",
    "reason",
}


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _assert_close(actual, expected, tol=1e-9, message=None):
    if not math.isclose(
        actual,
        expected,
        rel_tol=tol,
        abs_tol=tol,
    ):
        if message is None:
            message = (
                f"Expected {expected}, got {actual}"
            )
        raise AssertionError(message)


def _expect_exception(
    exception_type,
    function,
    *args,
    **kwargs,
):
    """
    Assert that function raises the expected exception.
    """

    try:
        function(
            *args,
            **kwargs,
        )

    except exception_type:
        return

    except Exception as exc:
        raise AssertionError(
            f"Expected {exception_type.__name__}, "
            f"got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"Expected {exception_type.__name__}, "
        "but no exception was raised."
    )


# ============================================================================
# GRAPH FIXTURES
# ============================================================================

def _rgb_hex(rgb):
    """
    Convert an RGB tuple into #RRGGBB.
    """

    r, g, b = rgb

    return "#{:02X}{:02X}{:02X}".format(
        int(r),
        int(g),
        int(b),
    )


def _add_node(
    graph,
    node,
    rgb=(100, 100, 100),
    latitude=35.0,
    longitude=51.0,
    country="IR",
    continent="AS",
    available=True,
):
    """
    Add a node using the RGB representations accepted by both
    heuristics.py and the topology/geographic layer.

    The important fields are deliberately duplicated:

        rgb_color
        rgb
        color

    This avoids making the test dependent on a single parser
    representation.
    """

    rgb_tuple = (
        int(rgb[0]),
        int(rgb[1]),
        int(rgb[2]),
    )

    rgb_hex = _rgb_hex(
        rgb_tuple
    )

    graph.add_node(
        node,
        rgb_color=rgb_tuple,
        rgb=rgb_tuple,
        color=rgb_hex,
        latitude=latitude,
        longitude=longitude,
        country=country,
        continent=continent,
        carbon_intensity=100.0,
        available=available,
    )


def _add_edge(
    graph,
    u,
    v,
    *,
    channel_key=None,
    fee_base_msat=1000,
    fee_proportional_millionths=100,
    cltv_expiry_delta=40,
    estimated_liquidity=100000,
    failure_probability=0.0,
    available=True,
):
    """
    Add a deterministic Lightning-style channel.
    """

    data = {
        "fee_base_msat": fee_base_msat,
        "fee_proportional_millionths":
            fee_proportional_millionths,
        "cltv_expiry_delta":
            cltv_expiry_delta,
        "estimated_liquidity":
            estimated_liquidity,
        "failure_probability":
            failure_probability,
        "available":
            available,
    }

    if graph.is_multigraph():

        if channel_key is None:
            channel_key = (
                f"{u}-{v}-0"
            )

        graph.add_edge(
            u,
            v,
            key=channel_key,
            **data,
        )

        return channel_key

    graph.add_edge(
        u,
        v,
        **data,
    )

    return None


def _make_basic_graph():
    """
    A -> B -> D
    """

    graph = nx.DiGraph()

    _add_node(
        graph,
        "A",
        rgb=(50, 50, 50),
        latitude=35.0,
        longitude=51.0,
    )

    _add_node(
        graph,
        "B",
        rgb=(100, 100, 100),
        latitude=35.1,
        longitude=51.1,
    )

    _add_node(
        graph,
        "D",
        rgb=(60, 60, 60),
        latitude=35.2,
        longitude=51.2,
    )

    _add_edge(
        graph,
        "A",
        "B",
        fee_base_msat=1000,
        fee_proportional_millionths=10,
        cltv_expiry_delta=40,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "B",
        "D",
        fee_base_msat=1000,
        fee_proportional_millionths=10,
        cltv_expiry_delta=40,
        estimated_liquidity=100000,
    )

    return graph


def _make_two_path_graph():
    """
    Two feasible paths:

        A -> B -> D
        A -> C -> D
    """

    graph = nx.DiGraph()

    _add_node(
        graph,
        "A",
        rgb=(10, 10, 10),
        latitude=35.0,
        longitude=51.0,
    )

    _add_node(
        graph,
        "B",
        rgb=(240, 240, 240),
        latitude=35.1,
        longitude=51.1,
    )

    _add_node(
        graph,
        "C",
        rgb=(20, 20, 20),
        latitude=36.0,
        longitude=52.0,
    )

    _add_node(
        graph,
        "D",
        rgb=(20, 20, 20),
        latitude=36.1,
        longitude=52.1,
    )

    _add_edge(
        graph,
        "A",
        "B",
        fee_base_msat=100,
        fee_proportional_millionths=1,
        cltv_expiry_delta=10,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "B",
        "D",
        fee_base_msat=100,
        fee_proportional_millionths=1,
        cltv_expiry_delta=10,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "A",
        "C",
        fee_base_msat=5000,
        fee_proportional_millionths=100,
        cltv_expiry_delta=100,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "C",
        "D",
        fee_base_msat=5000,
        fee_proportional_millionths=100,
        cltv_expiry_delta=100,
        estimated_liquidity=100000,
    )

    return graph


def _make_multigraph():
    """
    MultiDiGraph containing two parallel A -> B channels.
    """

    graph = nx.MultiDiGraph()

    _add_node(
        graph,
        "A",
        rgb=(50, 50, 50),
        latitude=35.0,
        longitude=51.0,
    )

    _add_node(
        graph,
        "B",
        rgb=(80, 80, 80),
        latitude=35.1,
        longitude=51.1,
    )

    _add_node(
        graph,
        "D",
        rgb=(60, 60, 60),
        latitude=35.2,
        longitude=51.2,
    )

    _add_edge(
        graph,
        "A",
        "B",
        channel_key="AB-LOW",
        fee_base_msat=100,
        fee_proportional_millionths=1,
        cltv_expiry_delta=10,
        estimated_liquidity=1000,
    )

    _add_edge(
        graph,
        "A",
        "B",
        channel_key="AB-HIGH",
        fee_base_msat=200,
        fee_proportional_millionths=2,
        cltv_expiry_delta=20,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "B",
        "D",
        channel_key="BD-1",
        fee_base_msat=100,
        fee_proportional_millionths=1,
        cltv_expiry_delta=10,
        estimated_liquidity=100000,
    )

    return graph


# ============================================================================
# 01. GRAPH NONE
# ============================================================================

def test_graph_none_validation():

    _expect_exception(
        (ValueError, TypeError),
        Dijkstra,
        None,
    )


# ============================================================================
# 02. INVALID GRAPH TYPE
# ============================================================================

def test_invalid_graph_type():

    _expect_exception(
        (ValueError, TypeError),
        Dijkstra,
        "not-a-graph",
    )


# ============================================================================
# 03. UNKNOWN SOURCE
# ============================================================================

def test_unknown_source():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    _expect_exception(
        ValueError,
        router.shortest_path,
        "UNKNOWN",
        "D",
        amount=1000,
    )


# ============================================================================
# 04. UNKNOWN TARGET
# ============================================================================

def test_unknown_target():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    _expect_exception(
        ValueError,
        router.shortest_path,
        "A",
        "UNKNOWN",
        amount=1000,
    )


# ============================================================================
# 05. INVALID AMOUNT
# ============================================================================

def test_invalid_amount():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    invalid_amounts = [
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
    ]

    for amount in invalid_amounts:

        _expect_exception(
            ValueError,
            router.shortest_path,
            "A",
            "D",
            amount=amount,
        )


# ============================================================================
# 06. MAX HOPS VALIDATION
# ============================================================================

def test_max_hops_validation():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    invalid_values = [
        0,
        -1,
        1.5,
        "3",
        True,
        False,
        float("nan"),
        float("inf"),
    ]

    for value in invalid_values:

        _expect_exception(
            ValueError,
            router.shortest_path,
            "A",
            "D",
            amount=1000,
            max_hops=value,
        )


# ============================================================================
# 07. INTEGER-VALUED FLOAT MAX HOPS
# ============================================================================

def test_integer_valued_float_max_hops():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        max_hops=2.0,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "max_hops=2.0 must be accepted",
    )

    _assert(
        result["hop_count"] == 2,
        "Expected exactly two hops",
    )


# ============================================================================
# 08. SOURCE == TARGET
# ============================================================================

def test_source_equals_target():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "A",
        amount=1000,
    )

    _assert(
        result["success"] is False,
        "source == target must return failure",
    )

    _assert(
        result["reason"] is not None,
        "Failure must include reason",
    )


# ============================================================================
# 09. OFFLINE SOURCE
# ============================================================================

def test_offline_source():

    graph = _make_basic_graph()

    graph.nodes["A"]["available"] = False

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
    )

    _assert(
        result["success"] is False,
        "Offline source must fail",
    )


# ============================================================================
# 10. OFFLINE TARGET
# ============================================================================

def test_offline_target():

    graph = _make_basic_graph()

    graph.nodes["D"]["available"] = False

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
    )

    _assert(
        result["success"] is False,
        "Offline target must fail",
    )


# ============================================================================
# 11. OFFLINE INTERMEDIATE NODE
# ============================================================================

def test_offline_intermediate_node():

    graph = _make_basic_graph()

    graph.nodes["B"]["available"] = False

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is False,
        "Offline intermediate node must be skipped",
    )


# ============================================================================
# 12. INVALID NODE BOOLEAN
# ============================================================================

def test_invalid_node_boolean():

    graph = _make_basic_graph()

    graph.nodes["A"]["available"] = "false"

    router = Dijkstra(graph)

    _expect_exception(
        ValueError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
    )


# ============================================================================
# 13. INVALID EDGE BOOLEAN
# ============================================================================

def test_invalid_edge_boolean():

    graph = _make_basic_graph()

    graph["A"]["B"]["available"] = "false"

    router = Dijkstra(graph)

    _expect_exception(
        ValueError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
    )


# ============================================================================
# 14. BASIC SHORTEST PATH
# ============================================================================

def test_basic_shortest_path():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "Basic route must succeed",
    )

    _assert(
        result["path"] == ["A", "B", "D"],
        f"Unexpected path: {result['path']}",
    )

    _assert(
        result["hop_count"] == 2,
        "Expected two hops",
    )


# ============================================================================
# 15. LIQUIDITY HARD CONSTRAINT
# ============================================================================

def test_liquidity_hard_constraint():

    graph = _make_basic_graph()

    graph["A"]["B"]["estimated_liquidity"] = 500

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is False,
        "Insufficient liquidity must make edge infeasible",
    )


# ============================================================================
# 16. NO FEASIBLE PATH
# ============================================================================

def test_no_feasible_path():

    graph = _make_basic_graph()

    graph["A"]["B"]["estimated_liquidity"] = 0

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is False,
        "No feasible path must return failure",
    )


# ============================================================================
# 17. MISSING LIQUIDITY
# ============================================================================

def test_missing_liquidity():

    graph = _make_basic_graph()

    del graph["A"]["B"]["estimated_liquidity"]

    router = Dijkstra(graph)

    _expect_exception(
        ValueError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )


# ============================================================================
# 18. MISSING FAILURE PROBABILITY
# ============================================================================

def test_hidden_failure_probability_is_ignored():

    graph = _make_basic_graph()

    del graph["A"]["B"]["failure_probability"]

    router = Dijkstra(graph)

    graph["A"]["B"]["simulator_failure_probability"] = 0.99

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(result["success"] is True, "latent probability must not block routing")
    _assert_close(
        result["failure_probability"],
        0.0,
        message="Unknown reliability should use neutral estimate",
    )


# ============================================================================
# 19. FAILURE PROBABILITY FROM HISTORY
# ============================================================================

def test_failure_probability_from_history():

    graph = _make_basic_graph()

    edge = graph["A"]["B"]

    edge.pop(
        "failure_probability",
        None,
    )

    edge["failure_count"] = 2
    edge["success_count"] = 8

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "Valid history must allow routing",
    )

    _assert_close(
        result["failure_probability"],
        0.2,
        message=(
            "Expected failure probability 0.2"
        ),
    )


# ============================================================================
# 20. ZERO HISTORY
# ============================================================================

def test_zero_history_uses_neutral_reliability():

    graph = _make_basic_graph()

    edge = graph["A"]["B"]

    edge.pop(
        "failure_probability",
        None,
    )

    edge["failure_count"] = 0
    edge["success_count"] = 0

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(result["success"] is True, "zero history is unknown, not invalid")
    _assert_close(
        result["failure_probability"],
        0.0,
        message="Unknown reliability should use neutral estimate",
    )


# ============================================================================
# 21. MAX HOPS BLOCKS LONG ROUTE
# ============================================================================

def test_max_hops_blocks_long_route():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        max_hops=1,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is False,
        "max_hops=1 must block two-hop route",
    )


# ============================================================================
# 22. MAX HOPS EXACT ROUTE
# ============================================================================

def test_max_hops_exact_route():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        max_hops=2,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "max_hops=2 must allow two-hop route",
    )

    _assert(
        result["hop_count"] == 2,
        "Expected two hops",
    )


# ============================================================================
# 23. LAMBDA = 0 INVARIANT
# ============================================================================

def test_lambda_zero_invariant():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    native = router.shortest_path(
        "A",
        "D",
        amount=1000,
        eta=0.9,
        lambda_h=0.0,
    )

    adaptive = router.shortest_path(
        "A",
        "D",
        amount=1000,
        eta=0.9,
        lambda_h=2.0,
    )

    _assert(
        native["success"] is True,
        "Native route must succeed",
    )

    _assert(
        adaptive["success"] is True,
        "Adaptive route must succeed",
    )

    _assert(
        native["path"] == adaptive["path"],
        "legacy lambda_h must preserve route",
    )

    _assert_close(
        native["cost"],
        adaptive["cost"],
        message=(
            "legacy lambda_h must preserve cost"
        ),
    )


# ============================================================================
# 24. ETA ADAPTIVE EFFECT
# ============================================================================

def test_eta_changes_adaptive_objective():

    graph = _make_two_path_graph()
    router = Dijkstra(graph)

    result_eta_0 = router.shortest_path(
        "A",
        "D",
        amount=1000,
        eta=0.0,
        lambda_h=1.0,
    )

    result_eta_1 = router.shortest_path(
        "A",
        "D",
        amount=1000,
        eta=1.0,
        lambda_h=1.0,
    )

    _assert(
        result_eta_0["success"] is True,
        "eta=0 route must succeed",
    )

    _assert(
        result_eta_1["success"] is True,
        "eta=1 route must succeed",
    )

    _assert(
        result_eta_0["eta"] == 0.0,
        "Returned eta must be 0",
    )

    _assert(
        result_eta_1["eta"] == 1.0,
        "Returned eta must be 1",
    )

    _assert(
        not math.isclose(
            result_eta_0["cost"],
            result_eta_1["cost"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "Changing eta must change adaptive objective",
    )


# ============================================================================
# 25. CUSTOM HEURISTIC
# ============================================================================

def test_custom_heuristic_is_used():

    graph = _make_basic_graph()

    calls = []

    def custom_cost(
        G,
        u,
        v,
        data,
        amount,
    ):
        calls.append(
            (
                u,
                v,
                amount,
            )
        )

        return 0.0

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=1.0,
        heuristic_fn=custom_cost,
    )

    _assert(
        result["success"] is True,
        "Custom heuristic route must succeed",
    )

    _assert(
        len(calls) > 0,
        "Custom heuristic must be called",
    )


# ============================================================================
# 26. CUSTOM HEURISTIC FAILURE
# ============================================================================

def test_invalid_custom_heuristic_does_not_fallback():

    graph = _make_basic_graph()

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

    router = Dijkstra(graph)

    _expect_exception(
        RuntimeError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
        lambda_h=1.0,
        heuristic_fn=failing_heuristic,
    )


# ============================================================================
# 27. NON-CALLABLE HEURISTIC
# ============================================================================

def test_non_callable_heuristic():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    _expect_exception(
        TypeError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
        heuristic_fn=12345,
    )


# ============================================================================
# 28. MULTIDIGRAPH CHANNEL IDENTITY
# ============================================================================

def test_multidigraph_exact_channel_key_preserved():

    graph = _make_multigraph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=5000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "MultiDiGraph route must succeed",
    )

    _assert(
        result["edges"] is not None,
        "edges must not be None",
    )

    _assert(
        all(
            len(edge) == 3
            for edge in result["edges"]
        ),
        "MultiDiGraph edges must be (u,v,key)",
    )

    keys = [
        edge[2]
        for edge in result["edges"]
    ]

    _assert(
        "BD-1" in keys,
        "BD-1 channel key must be preserved",
    )


# ============================================================================
# 29. MULTIDIGRAPH LIQUIDITY SELECTION
# ============================================================================

def test_multidigraph_liquidity_selects_other_channel():

    graph = _make_multigraph()

    graph["A"]["B"]["AB-LOW"][
        "estimated_liquidity"
    ] = 100

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=5000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "AB-HIGH must remain feasible",
    )

    first_edge = result["edges"][0]

    _assert(
        first_edge[2] == "AB-HIGH",
        (
            "Expected AB-HIGH, "
            f"got {first_edge[2]}"
        ),
    )


# ============================================================================
# 30. MISSING FEE
# ============================================================================

def test_missing_fee():

    graph = _make_basic_graph()

    del graph["A"]["B"][
        "fee_base_msat"
    ]

    router = Dijkstra(graph)

    _expect_exception(
        KeyError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )


# ============================================================================
# 31. MISSING DELAY
# ============================================================================

def test_missing_delay():

    graph = _make_basic_graph()

    del graph["A"]["B"][
        "cltv_expiry_delta"
    ]

    router = Dijkstra(graph)

    _expect_exception(
        KeyError,
        router.shortest_path,
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )


# ============================================================================
# 32. SUCCESS RESULT SCHEMA
# ============================================================================

def test_success_result_schema():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "Expected successful route",
    )

    missing = (
        REQUIRED_RESULT_FIELDS
        - set(result.keys())
    )

    _assert(
        not missing,
        f"Missing result fields: {sorted(missing)}",
    )

    _assert(
        isinstance(result["path"], list),
        "path must be a list",
    )

    _assert(
        isinstance(result["edges"], list),
        "edges must be a list",
    )

    _assert(
        math.isfinite(result["cost"]),
        "Successful cost must be finite",
    )


# ============================================================================
# 33. FAILURE RESULT SCHEMA
# ============================================================================

def test_failure_result_schema():

    graph = _make_basic_graph()
    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is False,
        "Expected infeasible payment",
    )

    missing = (
        REQUIRED_RESULT_FIELDS
        - set(result.keys())
    )

    _assert(
        not missing,
        f"Missing failure fields: {sorted(missing)}",
    )

    _assert(
        result["reason"] is not None,
        "Failure must contain reason",
    )


# ============================================================================
# 34. GEOGRAPHIC METRICS
# ============================================================================

def test_geographic_metrics():

    graph = nx.DiGraph()

    _add_node(
        graph,
        "A",
        rgb=(10, 10, 10),
        latitude=35.0,
        longitude=51.0,
        country="IR",
        continent="AS",
    )

    _add_node(
        graph,
        "B",
        rgb=(100, 100, 100),
        latitude=40.0,
        longitude=2.0,
        country="ES",
        continent="EU",
    )

    _add_node(
        graph,
        "D",
        rgb=(200, 200, 200),
        latitude=41.0,
        longitude=3.0,
        country="ES",
        continent="EU",
    )

    _add_edge(
        graph,
        "A",
        "B",
        fee_base_msat=1000,
        fee_proportional_millionths=10,
        cltv_expiry_delta=40,
        estimated_liquidity=100000,
    )

    _add_edge(
        graph,
        "B",
        "D",
        fee_base_msat=1000,
        fee_proportional_millionths=10,
        cltv_expiry_delta=40,
        estimated_liquidity=100000,
    )

    router = Dijkstra(graph)

    result = router.shortest_path(
        "A",
        "D",
        amount=1000,
        lambda_h=0.0,
    )

    _assert(
        result["success"] is True,
        "Geographic route must succeed",
    )

    _assert(
        result["total_distance_km"] > 0,
        "Distance must be positive",
    )

    _assert(
        result["total_carbon"] > 0,
        "Carbon proxy must be positive",
    )

    _assert(
        result["inter_country_hops"] >= 1,
        "A -> B must be inter-country",
    )

    _assert(
        result["inter_continent_hops"] >= 1,
        "A -> B must be inter-continent",
    )


# ============================================================================
# TEST REGISTRY
# ============================================================================

TESTS = [
    ("graph None validation", test_graph_none_validation),
    ("invalid graph type", test_invalid_graph_type),
    ("unknown source", test_unknown_source),
    ("unknown target", test_unknown_target),
    ("invalid amount", test_invalid_amount),
    ("max_hops validation", test_max_hops_validation),
    ("integer-valued float max_hops", test_integer_valued_float_max_hops),
    ("source equals target", test_source_equals_target),
    ("offline source", test_offline_source),
    ("offline target", test_offline_target),
    ("offline intermediate node", test_offline_intermediate_node),
    ("invalid node Boolean", test_invalid_node_boolean),
    ("invalid edge Boolean", test_invalid_edge_boolean),
    ("basic shortest path", test_basic_shortest_path),
    ("liquidity hard constraint", test_liquidity_hard_constraint),
    ("no feasible path", test_no_feasible_path),
    ("missing liquidity", test_missing_liquidity),
    ("hidden failure probability ignored", test_hidden_failure_probability_is_ignored),
    ("failure probability from history", test_failure_probability_from_history),
    ("zero history uses neutral reliability", test_zero_history_uses_neutral_reliability),
    ("max_hops blocks long route", test_max_hops_blocks_long_route),
    ("max_hops exact route", test_max_hops_exact_route),
    ("lambda_h zero invariant", test_lambda_zero_invariant),
    ("eta adaptive effect", test_eta_changes_adaptive_objective),
    ("custom heuristic", test_custom_heuristic_is_used),
    ("custom heuristic no fallback", test_invalid_custom_heuristic_does_not_fallback),
    ("non-callable heuristic", test_non_callable_heuristic),
    ("MultiDiGraph channel identity", test_multidigraph_exact_channel_key_preserved),
    ("MultiDiGraph liquidity selection", test_multidigraph_liquidity_selects_other_channel),
    ("missing fee", test_missing_fee),
    ("missing delay", test_missing_delay),
    ("success result schema", test_success_result_schema),
    ("failure result schema", test_failure_result_schema),
    ("geographic metrics", test_geographic_metrics),
]


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():

    print("=" * 72)
    print(" DIJKSTRA DETERMINISTIC VALIDATION")
    print("=" * 72)

    passed = 0
    failed = 0

    for index, (
        name,
        test_function,
    ) in enumerate(
        TESTS,
        start=1,
    ):

        try:

            test_function()

            print(
                f"[{index:02d}/{len(TESTS)}] "
                f"{name:<42} PASS"
            )

            passed += 1

        except Exception as exc:

            print(
                f"[{index:02d}/{len(TESTS)}] "
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
            "DIJKSTRA VALIDATION: PASS"
        )

        return 0

    print(
        "DIJKSTRA VALIDATION: FAIL"
    )

    return 1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        run_all_tests()
    )
