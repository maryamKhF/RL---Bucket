"""
Pathfinding/test_heuristics.py

Deterministic validation suite for Pathfinding.heuristics.

Purpose
-------

This test suite verifies the mathematical and data-validation
contracts of the routing heuristic module BEFORE testing it on
the real Lightning GML snapshot.

Verified components
-------------------

1. ETA validation
2. lambda_h validation
3. RGB parsing
4. RGB carbon proxy
5. channel fee
6. channel delay
7. native LND cost
8. adaptive heuristic
9. eta = 0 behavior
10. eta = 1 behavior
11. eta = 0.5 behavior
12. adaptive penalty
13. modified adaptive cost
14. lambda_h = 0 invariant
15. adaptive cost >= native cost
16. unified adaptive_edge_cost()
17. adaptive_lnd_cost()
18. missing-data failures
19. invalid-data failures
20. path-edge resolution
21. MultiDiGraph channel-key resolution
22. path evaluation
23. Haversine distance

Important
---------

This file intentionally does NOT test:

    - Dijkstra
    - Top-K
    - PPO
    - Bucket
    - real GML parsing

Those belong to separate validation stages.

Run
---

From repository root:

    py -m Pathfinding.test_heuristics

Expected final result:

    HEURISTICS VALIDATION: PASS
"""


import math
import traceback

import networkx as nx

from Pathfinding.heuristics import (
    validate_eta,
    validate_lambda_h,
    channel_fee,
    channel_delay,
    lnd_cost,
    node_carbon_intensity,
    adaptive_heuristic,
    adaptive_penalty,
    modified_cost,
    adaptive_edge_cost,
    adaptive_lnd_cost,
    enhanced_cost,
    _parse_rgb,
    _resolve_path_edge,
    evaluate_path,
    _haversine_distance_km,
)


# ==========================================================
# Test Configuration
# ==========================================================

EPS = 1e-9


# ==========================================================
# Test Helpers
# ==========================================================

def assert_close(actual, expected, name, tolerance=EPS):
    """
    Assert two numeric values are approximately equal.
    """

    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise AssertionError(
            f"{name} failed:\n"
            f"  actual   = {actual}\n"
            f"  expected = {expected}"
        )


def assert_equal(actual, expected, name):
    """
    Assert exact equality.
    """

    if actual != expected:
        raise AssertionError(
            f"{name} failed:\n"
            f"  actual   = {actual!r}\n"
            f"  expected = {expected!r}"
        )


def assert_true(condition, name):
    """
    Assert a condition is true.
    """

    if not condition:
        raise AssertionError(
            f"{name} failed."
        )


def assert_raises(
    exception_type,
    function,
    name,
):
    """
    Assert that function raises the expected exception.
    """

    try:
        function()

    except exception_type:
        return

    except Exception as exc:
        raise AssertionError(
            f"{name} failed:\n"
            f"Expected: {exception_type.__name__}\n"
            f"Received: {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"{name} failed:\n"
        f"Expected {exception_type.__name__}, "
        f"but no exception was raised."
    )


# ==========================================================
# Deterministic Synthetic Graph
# ==========================================================

def build_test_graph():
    """
    Build a small deterministic MultiDiGraph.

    Node RGB values are chosen so that the carbon proxy values
    are easy to verify analytically.

    A:
        RGB = (100, 100, 100)
        C_A = 100

    B:
        RGB = (200, 100, 100)
        C_B = 129.9

    C:
        RGB = (100, 200, 100)
        C_C = 158.7

    D:
        RGB = (100, 100, 200)
        C_D = 111.4
    """

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        rgb_color=[100, 100, 100],
        latitude=0.0,
        longitude=0.0,
    )

    G.add_node(
        "B",
        rgb_color=[200, 100, 100],
        latitude=0.0,
        longitude=1.0,
    )

    G.add_node(
        "C",
        rgb_color=[100, 200, 100],
        latitude=1.0,
        longitude=1.0,
    )

    G.add_node(
        "D",
        rgb_color=[100, 100, 200],
        latitude=1.0,
        longitude=2.0,
    )

    # A -> B
    G.add_edge(
        "A",
        "B",
        key="AB-1",
        fee_base_msat=1000,
        fee_proportional_millionths=100,
        cltv_expiry_delta=40,
        estimated_liquidity=100_000,
        failure_probability=0.10,
        success_count=90,
        failure_count=10,
        scid="100x1x0",
    )

    # B -> C
    G.add_edge(
        "B",
        "C",
        key="BC-1",
        fee_base_msat=2000,
        fee_proportional_millionths=200,
        cltv_expiry_delta=30,
        estimated_liquidity=120_000,
        failure_probability=0.20,
        success_count=80,
        failure_count=20,
        scid="200x1x0",
    )

    # A -> C
    G.add_edge(
        "A",
        "C",
        key="AC-1",
        fee_base_msat=500,
        fee_proportional_millionths=50,
        cltv_expiry_delta=60,
        estimated_liquidity=80_000,
        failure_probability=0.05,
        success_count=95,
        failure_count=5,
        scid="300x1x0",
    )

    # C -> D
    G.add_edge(
        "C",
        "D",
        key="CD-1",
        fee_base_msat=1500,
        fee_proportional_millionths=150,
        cltv_expiry_delta=20,
        estimated_liquidity=150_000,
        failure_probability=0.10,
        success_count=90,
        failure_count=10,
        scid="400x1x0",
    )

    # ------------------------------------------------------
    # Parallel channels A -> B
    # ------------------------------------------------------

    G.add_edge(
        "A",
        "B",
        key="AB-2",
        fee_base_msat=3000,
        fee_proportional_millionths=300,
        cltv_expiry_delta=50,
        estimated_liquidity=200_000,
        failure_probability=0.15,
        success_count=85,
        failure_count=15,
        scid="100x2x0",
    )

    return G


# ==========================================================
# 1. ETA Validation
# ==========================================================

def test_eta_validation():
    assert_equal(
        validate_eta(0.0),
        0.0,
        "eta=0 validation",
    )

    assert_equal(
        validate_eta(1.0),
        1.0,
        "eta=1 validation",
    )

    assert_equal(
        validate_eta(0.5),
        0.5,
        "eta=0.5 validation",
    )

    for invalid_eta in [
        -0.000001,
        -1.0,
        1.000001,
        2.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        "abc",
        None,
    ]:
        assert_raises(
            (ValueError, TypeError),
            lambda value=invalid_eta: validate_eta(value),
            f"invalid eta={invalid_eta!r}",
        )


# ==========================================================
# 2. Lambda Validation
# ==========================================================

def test_lambda_validation():
    assert_equal(
        validate_lambda_h(0.0),
        0.0,
        "lambda_h=0",
    )

    assert_equal(
        validate_lambda_h(1.0),
        1.0,
        "lambda_h=1",
    )

    assert_equal(
        validate_lambda_h(2.5),
        2.5,
        "lambda_h=2.5",
    )

    for invalid_lambda in [
        -0.000001,
        -1.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        "abc",
        None,
    ]:
        assert_raises(
            (ValueError, TypeError),
            lambda value=invalid_lambda:
                validate_lambda_h(value),
            f"invalid lambda_h={invalid_lambda!r}",
        )


# ==========================================================
# 3. RGB Parsing
# ==========================================================

def test_rgb_parsing():
    assert_equal(
        _parse_rgb(
            [10, 20, 30],
            "A",
        ),
        (10.0, 20.0, 30.0),
        "RGB list parsing",
    )

    assert_equal(
        _parse_rgb(
            (10, 20, 30),
            "A",
        ),
        (10.0, 20.0, 30.0),
        "RGB tuple parsing",
    )

    assert_equal(
        _parse_rgb(
            "#0A141E",
            "A",
        ),
        (10.0, 20.0, 30.0),
        "RGB #hex parsing",
    )

    assert_equal(
        _parse_rgb(
            "0A141E",
            "A",
        ),
        (10.0, 20.0, 30.0),
        "RGB hex parsing",
    )

    assert_raises(
        ValueError,
        lambda: _parse_rgb(
            [1, 2],
            "A",
        ),
        "RGB too short",
    )

    assert_raises(
        ValueError,
        lambda: _parse_rgb(
            [256, 0, 0],
            "A",
        ),
        "RGB > 255",
    )

    assert_raises(
        ValueError,
        lambda: _parse_rgb(
            [-1, 0, 0],
            "A",
        ),
        "RGB < 0",
    )

    assert_raises(
        ValueError,
        lambda: _parse_rgb(
            "#GG0000",
            "A",
        ),
        "invalid hex RGB",
    )


# ==========================================================
# 4. Carbon Proxy
# ==========================================================

def test_carbon_proxy():
    G = build_test_graph()

    expected = (
        0.299 * 100
        +
        0.587 * 100
        +
        0.114 * 100
    )

    assert_close(
        node_carbon_intensity(
            G,
            "A",
        ),
        expected,
        "carbon proxy for A",
    )

    assert_close(
        node_carbon_intensity(
            G,
            "B",
        ),
        (
            0.299 * 200
            +
            0.587 * 100
            +
            0.114 * 100
        ),
        "carbon proxy for B",
    )

    assert_close(
        node_carbon_intensity(
            G,
            "C",
        ),
        (
            0.299 * 100
            +
            0.587 * 200
            +
            0.114 * 100
        ),
        "carbon proxy for C",
    )

    assert_close(
        node_carbon_intensity(
            G,
            "D",
        ),
        (
            0.299 * 100
            +
            0.587 * 100
            +
            0.114 * 200
        ),
        "carbon proxy for D",
    )


# ==========================================================
# 5. Missing Carbon Data
# ==========================================================

def test_missing_carbon_data():
    G = build_test_graph()

    G.nodes["A"].pop("rgb_color")

    assert_raises(
        KeyError,
        lambda: node_carbon_intensity(
            G,
            "A",
        ),
        "missing RGB data",
    )


# ==========================================================
# 6. Channel Fee
# ==========================================================

def test_channel_fee():
    data = {
        "fee_base_msat": 1000,
        "fee_proportional_millionths": 100,
    }

    amount = 50_000

    expected = (
        1000
        +
        50_000 * 100 / 1_000_000
    )

    assert_close(
        channel_fee(
            data,
            amount,
        ),
        expected,
        "channel fee",
    )

    # Primary field names
    data_primary = {
        "fee_base": 1000,
        "fee_rate": 100,
    }

    assert_close(
        channel_fee(
            data_primary,
            amount,
        ),
        expected,
        "channel fee using primary names",
    )


# ==========================================================
# 7. Invalid Fee Data
# ==========================================================

def test_invalid_fee_data():
    amount = 50_000

    assert_raises(
        KeyError,
        lambda: channel_fee(
            {
                "fee_base_msat": 1000,
            },
            amount,
        ),
        "missing fee rate",
    )

    assert_raises(
        KeyError,
        lambda: channel_fee(
            {
                "fee_proportional_millionths": 100,
            },
            amount,
        ),
        "missing base fee",
    )

    assert_raises(
        ValueError,
        lambda: channel_fee(
            {
                "fee_base_msat": -1,
                "fee_proportional_millionths": 100,
            },
            amount,
        ),
        "negative base fee",
    )

    assert_raises(
        ValueError,
        lambda: channel_fee(
            {
                "fee_base_msat": 1000,
                "fee_proportional_millionths": -1,
            },
            amount,
        ),
        "negative proportional fee",
    )


# ==========================================================
# 8. Channel Delay
# ==========================================================

def test_channel_delay():
    assert_close(
        channel_delay(
            {"delay": 40}
        ),
        40.0,
        "delay field",
    )

    assert_close(
        channel_delay(
            {"cltv_expiry_delta": 40}
        ),
        40.0,
        "CLTV delay field",
    )

    assert_raises(
        KeyError,
        lambda: channel_delay({}),
        "missing delay",
    )

    assert_raises(
        ValueError,
        lambda: channel_delay(
            {"delay": -1}
        ),
        "negative delay",
    )


# ==========================================================
# 9. Native LND Cost
# ==========================================================

def test_lnd_cost():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000

    expected_fee = (
        1000
        +
        50_000 * 100 / 1_000_000
    )

    expected_delay = 40.0

    expected_cost = (
        expected_fee
        +
        0.5 * expected_delay
        +
        1.0
    )

    actual = lnd_cost(
        G,
        "A",
        "B",
        data,
        amount,
    )

    assert_close(
        actual,
        expected_cost,
        "LND cost",
    )

    assert_true(
        actual > 0.0,
        "LND cost must be positive",
    )


# ==========================================================
# 10. Adaptive Heuristic eta=0
# ==========================================================

def test_adaptive_heuristic_eta_zero():
    G = build_test_graph()

    cu = node_carbon_intensity(G, "A")
    cv = node_carbon_intensity(G, "B")

    expected = cv - cu

    actual = adaptive_heuristic(
        G,
        "A",
        "B",
        0.0,
    )

    assert_close(
        actual,
        expected,
        "adaptive heuristic eta=0",
    )


# ==========================================================
# 11. Adaptive Heuristic eta=1
# ==========================================================

def test_adaptive_heuristic_eta_one():
    G = build_test_graph()

    cu = node_carbon_intensity(G, "A")
    cv = node_carbon_intensity(G, "B")

    expected = (
        cu + cv
    ) / 2.0

    actual = adaptive_heuristic(
        G,
        "A",
        "B",
        1.0,
    )

    assert_close(
        actual,
        expected,
        "adaptive heuristic eta=1",
    )


# ==========================================================
# 12. Adaptive Heuristic eta=0.5
# ==========================================================

def test_adaptive_heuristic_eta_half():
    G = build_test_graph()

    cu = node_carbon_intensity(G, "A")
    cv = node_carbon_intensity(G, "B")

    average_component = (
        cu + cv
    ) / 2.0

    transition_component = (
        cv - cu
    )

    expected = (
        0.5 * average_component
        +
        0.5 * transition_component
    )

    actual = adaptive_heuristic(
        G,
        "A",
        "B",
        0.5,
    )

    assert_close(
        actual,
        expected,
        "adaptive heuristic eta=0.5",
    )


# ==========================================================
# 13. Adaptive Heuristic Convexity
# ==========================================================

def test_adaptive_heuristic_convexity():
    """
    Verify:

        h(eta)
        =
        eta * average
        +
        (1-eta) * transition

    Therefore h(eta) must be the corresponding linear
    interpolation between h(0) and h(1).
    """

    G = build_test_graph()

    h0 = adaptive_heuristic(
        G,
        "A",
        "B",
        0.0,
    )

    h1 = adaptive_heuristic(
        G,
        "A",
        "B",
        1.0,
    )

    for eta in [
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
    ]:
        actual = adaptive_heuristic(
            G,
            "A",
            "B",
            eta,
        )

        expected = (
            (1.0 - eta) * h0
            +
            eta * h1
        )

        assert_close(
            actual,
            expected,
            f"heuristic linearity eta={eta}",
        )


# ==========================================================
# 14. Adaptive Penalty
# ==========================================================

def test_adaptive_penalty():
    G = build_test_graph()

    for eta in [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raw_h = adaptive_heuristic(
            G,
            "A",
            "B",
            eta,
        )

        normalized_h = (
            abs(raw_h) / 255.0
        )

        expected = (
            normalized_h
            /
            (1.0 + normalized_h)
        )

        actual = adaptive_penalty(
            G,
            "A",
            "B",
            eta,
        )

        assert_close(
            actual,
            expected,
            f"adaptive penalty eta={eta}",
        )

        assert_true(
            0.0 <= actual <= 0.5,
            f"penalty range eta={eta}",
        )


# ==========================================================
# 15. Zero Directional Component
# ==========================================================

def test_zero_directional_component():
    """
    Verify that the directional carbon component:

        C_v - C_u

    becomes zero when both endpoint carbon proxies are equal.

    With:

        C_u = C_v = 100

    the full adaptive heuristic is:

        h(u,v,eta)
        =
        eta * 100
        +
        (1-eta) * 0

        =
        100 * eta

    Therefore the complete heuristic is NOT zero for every eta.

    What becomes zero is specifically the directional component.
    Consequently:

        eta=0     -> h = 0
        eta=0.25  -> h = 25
        eta=0.5   -> h = 50
        eta=0.75  -> h = 75
        eta=1     -> h = 100
    """

    G = nx.MultiDiGraph()

    G.add_node(
        "A",
        rgb_color=[100, 100, 100],
    )

    G.add_node(
        "B",
        rgb_color=[100, 100, 100],
    )

    data = {
        "fee_base_msat": 1000,
        "fee_proportional_millionths": 100,
        "cltv_expiry_delta": 40,
    }

    G.add_edge(
        "A",
        "B",
        key="AB",
        **data,
    )

    carbon_a = node_carbon_intensity(
        G,
        "A",
    )

    carbon_b = node_carbon_intensity(
        G,
        "B",
    )

    assert_close(
        carbon_a,
        100.0,
        "equal-carbon node A",
    )

    assert_close(
        carbon_b,
        100.0,
        "equal-carbon node B",
    )

    directional_component = (
        carbon_b - carbon_a
    )

    assert_close(
        directional_component,
        0.0,
        "zero directional carbon component",
    )

    expected_values = {
        0.0: 0.0,
        0.25: 25.0,
        0.5: 50.0,
        0.75: 75.0,
        1.0: 100.0,
    }

    for eta, expected in expected_values.items():

        raw_h = adaptive_heuristic(
            G,
            "A",
            "B",
            eta,
        )

        penalty = adaptive_penalty(
            G,
            "A",
            "B",
            eta,
        )

        native = lnd_cost(
            G,
            "A",
            "B",
            data,
            50_000,
        )

        adaptive = adaptive_lnd_cost(
            G,
            "A",
            "B",
            data,
            50_000,
            eta=eta,
            lambda_h=1.0,
        )

        assert_close(
            raw_h,
            expected,
            f"equal-carbon heuristic eta={eta}",
        )

        normalized_h = (
            abs(expected) / 255.0
        )

        expected_penalty = (
            normalized_h
            /
            (1.0 + normalized_h)
        )

        assert_close(
            penalty,
            expected_penalty,
            f"equal-carbon penalty eta={eta}",
        )

        expected_adaptive_cost = (
            native
            *
            (
                1.0
                +
                expected_penalty
            )
        )

        assert_close(
            adaptive,
            expected_adaptive_cost,
            f"equal-carbon adaptive cost eta={eta}",
        )

    # Important boundary invariant:
    #
    # At eta=0, only the directional component is active.
    # Since C_v - C_u = 0, the adaptive penalty is zero and
    # adaptive cost must equal native cost.
    eta_zero_adaptive = adaptive_lnd_cost(
        G=G,
        u="A",
        v="B",
        data=data,
        amount=50_000,
        eta=0.0,
        lambda_h=1.0,
    )

    native = lnd_cost(
        G,
        "A",
        "B",
        data,
        50_000,
    )

    assert_close(
        eta_zero_adaptive,
        native,
        "equal-carbon eta=0 native-cost invariant",
    )


# ==========================================================
# 16. Modified Cost
# ==========================================================

def test_modified_cost():
    native = 100.0
    penalty = 0.25
    lambda_h = 2.0

    expected = (
        native
        *
        (
            1.0
            +
            lambda_h * penalty
        )
    )

    actual = modified_cost(
        native_cost=native,
        geo_penalty=penalty,
        eta=0.5,
        lambda_h=lambda_h,
    )

    assert_close(
        actual,
        expected,
        "modified adaptive cost",
    )


# ==========================================================
# 17. Lambda=0 Invariant
# ==========================================================

def test_lambda_zero_invariant():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000

    native = lnd_cost(
        G,
        "A",
        "B",
        data,
        amount,
    )

    for eta in [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        result = adaptive_edge_cost(
            G=G,
            u="A",
            v="B",
            data=data,
            amount=amount,
            eta=eta,
            lambda_h=0.0,
        )

        assert_close(
            result["cost"],
            native,
            f"lambda=0 invariant eta={eta}",
        )


# ==========================================================
# 18. Adaptive Cost >= Native Cost
# ==========================================================

def test_adaptive_cost_not_less_than_native():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000

    native = lnd_cost(
        G,
        "A",
        "B",
        data,
        amount,
    )

    for eta in [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
    ]:
        for lambda_h in [
            0.0,
            0.5,
            1.0,
            2.0,
        ]:
            result = adaptive_edge_cost(
                G=G,
                u="A",
                v="B",
                data=data,
                amount=amount,
                eta=eta,
                lambda_h=lambda_h,
            )

            assert_true(
                result["cost"] >= native,
                (
                    "adaptive cost must not be lower "
                    f"than native cost "
                    f"(eta={eta}, lambda={lambda_h})"
                ),
            )


# ==========================================================
# 19. Unified Adaptive Edge Cost
# ==========================================================

def test_adaptive_edge_cost():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000
    eta = 0.5
    lambda_h = 1.5

    native = lnd_cost(
        G,
        "A",
        "B",
        data,
        amount,
    )

    raw_h = adaptive_heuristic(
        G,
        "A",
        "B",
        eta,
    )

    penalty = adaptive_penalty(
        G,
        "A",
        "B",
        eta,
    )

    expected_cost = (
        native
        *
        (
            1.0
            +
            lambda_h * penalty
        )
    )

    result = adaptive_edge_cost(
        G=G,
        u="A",
        v="B",
        data=data,
        amount=amount,
        eta=eta,
        lambda_h=lambda_h,
    )

    assert_close(
        result["native_cost"],
        native,
        "adaptive_edge_cost native cost",
    )

    assert_close(
        result["raw_heuristic"],
        raw_h,
        "adaptive_edge_cost raw heuristic",
    )

    assert_close(
        result["adaptive_penalty"],
        penalty,
        "adaptive_edge_cost penalty",
    )

    assert_close(
        result["cost"],
        expected_cost,
        "adaptive_edge_cost final cost",
    )


# ==========================================================
# 20. adaptive_lnd_cost Consistency
# ==========================================================

def test_adaptive_lnd_cost_consistency():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000
    eta = 0.75
    lambda_h = 1.25

    result = adaptive_edge_cost(
        G=G,
        u="A",
        v="B",
        data=data,
        amount=amount,
        eta=eta,
        lambda_h=lambda_h,
    )

    wrapper_result = adaptive_lnd_cost(
        G=G,
        u="A",
        v="B",
        data=data,
        amount=amount,
        eta=eta,
        lambda_h=lambda_h,
    )

    assert_close(
        wrapper_result,
        result["cost"],
        "adaptive_lnd_cost consistency",
    )


# ==========================================================
# 21. Custom Heuristic Must Not Fallback
# ==========================================================

def test_custom_heuristic_no_fallback():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    def failing_heuristic(
        G,
        u,
        v,
        data,
        amount,
    ):
        raise RuntimeError(
            "INTENTIONAL TEST FAILURE"
        )

    assert_raises(
        RuntimeError,
        lambda: adaptive_edge_cost(
            G=G,
            u="A",
            v="B",
            data=data,
            amount=50_000,
            eta=0.5,
            heuristic_fn=failing_heuristic,
        ),
        "custom heuristic failure must propagate",
    )


# ==========================================================
# 22. Invalid Custom Heuristic Result
# ==========================================================

def test_invalid_custom_heuristic_result():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    def invalid_heuristic(
        G,
        u,
        v,
        data,
        amount,
    ):
        return -10.0

    assert_raises(
        ValueError,
        lambda: adaptive_edge_cost(
            G=G,
            u="A",
            v="B",
            data=data,
            amount=50_000,
            eta=0.5,
            heuristic_fn=invalid_heuristic,
        ),
        "invalid custom heuristic result",
    )


# ==========================================================
# 23. Missing Data Must Fail
# ==========================================================

def test_missing_required_data():
    G = build_test_graph()

    # Missing fee
    missing_fee = {
        "cltv_expiry_delta": 40,
    }

    assert_raises(
        KeyError,
        lambda: lnd_cost(
            G,
            "A",
            "B",
            missing_fee,
            50_000,
        ),
        "missing fee data",
    )

    # Missing delay
    missing_delay = {
        "fee_base_msat": 1000,
        "fee_proportional_millionths": 100,
    }

    assert_raises(
        KeyError,
        lambda: lnd_cost(
            G,
            "A",
            "B",
            missing_delay,
            50_000,
        ),
        "missing delay data",
    )

    # Missing RGB
    G_missing_rgb = build_test_graph()

    del G_missing_rgb.nodes["A"]["rgb_color"]

    assert_raises(
        KeyError,
        lambda: adaptive_heuristic(
            G_missing_rgb,
            "A",
            "B",
            0.5,
        ),
        "missing RGB data",
    )


# ==========================================================
# 24. Enhanced Cost
# ==========================================================

def test_enhanced_cost():
    G = build_test_graph()

    data = G["A"]["B"]["AB-1"]

    amount = 50_000

    fee = channel_fee(
        data,
        amount,
    )

    delay = channel_delay(
        data,
    )

    expected = (
        fee
        +
        0.5 * delay
        +
        0.10
        +
        1.0
    )

    actual = enhanced_cost(
        G,
        "A",
        "B",
        data,
        amount,
    )

    assert_close(
        actual,
        expected,
        "enhanced cost",
    )


# ==========================================================
# 25. Enhanced Cost Missing Reliability
# ==========================================================

def test_enhanced_cost_missing_reliability():
    G = build_test_graph()

    data = {
        "fee_base_msat": 1000,
        "fee_proportional_millionths": 100,
        "cltv_expiry_delta": 40,
    }

    assert_raises(
        KeyError,
        lambda: enhanced_cost(
            G,
            "A",
            "B",
            data,
            50_000,
        ),
        "enhanced cost missing reliability",
    )


# ==========================================================
# 26. MultiDiGraph Exact Channel Resolution
# ==========================================================

def test_multidigraph_exact_channel_resolution():
    G = build_test_graph()

    edge_1 = _resolve_path_edge(
        G,
        "A",
        "B",
        "AB-1",
    )

    edge_2 = _resolve_path_edge(
        G,
        "A",
        "B",
        "AB-2",
    )

    assert_equal(
        edge_1["scid"],
        "100x1x0",
        "AB-1 SCID resolution",
    )

    assert_equal(
        edge_2["scid"],
        "100x2x0",
        "AB-2 SCID resolution",
    )


# ==========================================================
# 27. Ambiguous MultiDiGraph Resolution Must Fail
# ==========================================================

def test_ambiguous_multidigraph_resolution():
    G = build_test_graph()

    assert_raises(
        ValueError,
        lambda: _resolve_path_edge(
            G,
            "A",
            "B",
        ),
        "ambiguous parallel-channel resolution",
    )


# ==========================================================
# 28. Invalid Channel Key
# ==========================================================

def test_invalid_channel_key():
    G = build_test_graph()

    assert_raises(
        KeyError,
        lambda: _resolve_path_edge(
            G,
            "A",
            "B",
            "NON_EXISTENT",
        ),
        "invalid channel key",
    )


# ==========================================================
# 29. Path Evaluation
# ==========================================================

def test_evaluate_path():
    G = build_test_graph()

    amount = 50_000

    path = [
        ("A", "B", "AB-1"),
        ("B", "C", "BC-1"),
        ("C", "D", "CD-1"),
    ]

    result = evaluate_path(
        G,
        path,
        amount,
    )

    assert_true(
        result["success"],
        "path evaluation success",
    )

    expected_fee = (
        channel_fee(
            G["A"]["B"]["AB-1"],
            amount,
        )
        +
        channel_fee(
            G["B"]["C"]["BC-1"],
            amount,
        )
        +
        channel_fee(
            G["C"]["D"]["CD-1"],
            amount,
        )
    )

    expected_delay = (
        channel_delay(
            G["A"]["B"]["AB-1"],
        )
        +
        channel_delay(
            G["B"]["C"]["BC-1"],
        )
        +
        channel_delay(
            G["C"]["D"]["CD-1"],
        )
    )

    expected_liquidity = min(
        100_000,
        120_000,
        150_000,
    )

    expected_reliability = (
        (1.0 - 0.10)
        *
        (1.0 - 0.20)
        *
        (1.0 - 0.10)
    )

    expected_carbon = (
        node_carbon_intensity(G, "A")
        +
        node_carbon_intensity(G, "B")
        +
        node_carbon_intensity(G, "C")
        +
        node_carbon_intensity(G, "D")
    )

    assert_close(
        result["total_fee"],
        expected_fee,
        "path total fee",
    )

    assert_close(
        result["total_delay"],
        expected_delay,
        "path total delay",
    )

    assert_close(
        result["min_liquidity"],
        expected_liquidity,
        "path minimum liquidity",
    )

    assert_close(
        result["reliability"],
        expected_reliability,
        "path reliability",
    )

    assert_close(
        result["total_carbon"],
        expected_carbon,
        "path total carbon",
    )

    assert_true(
        result["total_distance_km"] > 0.0,
        "path distance must be positive",
    )


# ==========================================================
# 30. Path Evaluation Must Respect Channel Key
# ==========================================================

def test_path_evaluation_exact_channel():
    G = build_test_graph()

    amount = 50_000

    path_1 = [
        ("A", "B", "AB-1"),
    ]

    path_2 = [
        ("A", "B", "AB-2"),
    ]

    result_1 = evaluate_path(
        G,
        path_1,
        amount,
    )

    result_2 = evaluate_path(
        G,
        path_2,
        amount,
    )

    expected_fee_1 = channel_fee(
        G["A"]["B"]["AB-1"],
        amount,
    )

    expected_fee_2 = channel_fee(
        G["A"]["B"]["AB-2"],
        amount,
    )

    assert_close(
        result_1["total_fee"],
        expected_fee_1,
        "AB-1 exact fee",
    )

    assert_close(
        result_2["total_fee"],
        expected_fee_2,
        "AB-2 exact fee",
    )

    assert_true(
        result_1["total_fee"]
        !=
        result_2["total_fee"],
        "parallel channels must remain distinguishable",
    )


# ==========================================================
# 31. Haversine
# ==========================================================

def test_haversine():
    distance = _haversine_distance_km(
        0.0,
        0.0,
        0.0,
        1.0,
    )

    # Approximately one degree longitude at the equator.
    expected = 111.19492664455873

    assert_close(
        distance,
        expected,
        "haversine 1-degree equator",
        tolerance=1e-6,
    )

    assert_close(
        _haversine_distance_km(
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        0.0,
        "zero haversine distance",
    )


# ==========================================================
# 32. Haversine Invalid Coordinates
# ==========================================================

def test_haversine_invalid_coordinates():
    assert_raises(
        ValueError,
        lambda: _haversine_distance_km(
            91,
            0,
            0,
            0,
        ),
        "invalid latitude",
    )

    assert_raises(
        ValueError,
        lambda: _haversine_distance_km(
            0,
            181,
            0,
            0,
        ),
        "invalid longitude",
    )


# ==========================================================
# Test Registry
# ==========================================================

TESTS = [
    (
        "ETA validation",
        test_eta_validation,
    ),
    (
        "lambda_h validation",
        test_lambda_validation,
    ),
    (
        "RGB parsing",
        test_rgb_parsing,
    ),
    (
        "carbon proxy",
        test_carbon_proxy,
    ),
    (
        "missing carbon data",
        test_missing_carbon_data,
    ),
    (
        "channel fee",
        test_channel_fee,
    ),
    (
        "invalid fee data",
        test_invalid_fee_data,
    ),
    (
        "channel delay",
        test_channel_delay,
    ),
    (
        "native LND cost",
        test_lnd_cost,
    ),
    (
        "adaptive heuristic eta=0",
        test_adaptive_heuristic_eta_zero,
    ),
    (
        "adaptive heuristic eta=1",
        test_adaptive_heuristic_eta_one,
    ),
    (
        "adaptive heuristic eta=0.5",
        test_adaptive_heuristic_eta_half,
    ),
    (
        "adaptive heuristic convexity",
        test_adaptive_heuristic_convexity,
    ),
    (
        "adaptive penalty",
        test_adaptive_penalty,
    ),
    (
        "zero directional component",
        test_zero_directional_component,
    ),
    (
        "modified cost",
        test_modified_cost,
    ),
    (
        "lambda=0 invariant",
        test_lambda_zero_invariant,
    ),
    (
        "adaptive cost >= native cost",
        test_adaptive_cost_not_less_than_native,
    ),
    (
        "adaptive edge cost",
        test_adaptive_edge_cost,
    ),
    (
        "adaptive LND consistency",
        test_adaptive_lnd_cost_consistency,
    ),
    (
        "custom heuristic no fallback",
        test_custom_heuristic_no_fallback,
    ),
    (
        "invalid custom heuristic",
        test_invalid_custom_heuristic_result,
    ),
    (
        "missing required data",
        test_missing_required_data,
    ),
    (
        "enhanced cost",
        test_enhanced_cost,
    ),
    (
        "enhanced cost missing reliability",
        test_enhanced_cost_missing_reliability,
    ),
    (
        "MultiDiGraph exact channel resolution",
        test_multidigraph_exact_channel_resolution,
    ),
    (
        "ambiguous MultiDiGraph resolution",
        test_ambiguous_multidigraph_resolution,
    ),
    (
        "invalid channel key",
        test_invalid_channel_key,
    ),
    (
        "path evaluation",
        test_evaluate_path,
    ),
    (
        "path evaluation exact channel",
        test_path_evaluation_exact_channel,
    ),
    (
        "Haversine",
        test_haversine,
    ),
    (
        "Haversine invalid coordinates",
        test_haversine_invalid_coordinates,
    ),
]


# ==========================================================
# Runner
# ==========================================================

def main():
    print()
    print("=" * 70)
    print(" ROUTING HEURISTICS DETERMINISTIC VALIDATION")
    print("=" * 70)
    print()
    print("Module: Pathfinding.heuristics")
    print("Graph:  Synthetic deterministic MultiDiGraph")
    print("Scope:  Mathematical + validation contracts")
    print()

    passed = 0
    failed = 0

    failures = []

    for index, (name, test_function) in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(TESTS):02d}] "
            f"{name:<45}",
            end="",
        )

        try:
            test_function()

        except Exception as exc:
            failed += 1

            failures.append(
                (
                    name,
                    exc,
                    traceback.format_exc(),
                )
            )

            print("FAIL")

        else:
            passed += 1
            print("PASS")

    print()
    print("=" * 70)
    print(" VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Total : {len(TESTS)}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    if failures:
        print()
        print("=" * 70)
        print(" FAILURES")
        print("=" * 70)

        for index, (
            name,
            exc,
            trace,
        ) in enumerate(
            failures,
            start=1,
        ):
            print()
            print(
                f"[FAILURE {index}] {name}"
            )
            print(
                f"Exception: {type(exc).__name__}: {exc}"
            )
            print(trace)

        print("=" * 70)
        print(" HEURISTICS VALIDATION: FAIL")
        print("=" * 70)

        raise SystemExit(1)

    print()
    print("=" * 70)
    print(" HEURISTICS VALIDATION: PASS")
    print("=" * 70)
    print()
    print(
        "The deterministic mathematical and validation "
        "contracts of Pathfinding.heuristics passed."
    )
    print(
        "Next step: validate Dijkstra against the same "
        "adaptive edge-cost contract."
    )
    print()


if __name__ == "__main__":
    main()