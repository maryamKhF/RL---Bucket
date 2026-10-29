"""
Simulation/test_payment_simulator.py

Deterministic validation suite for PaymentSimulator.

Coverage
--------
1. Constructor validation
2. PaymentResult validation
3. Amount validation
4. Route validation
5. Exact MultiDiGraph channel identity
6. Parallel-channel protection
7. Fee calculation
8. Delay calculation
9. Carbon metric
10. Country / continent metrics
11. FailureModel contract
12. Failure propagation
13. visited_edges validation
14. Settlement contract
15. Settlement failure
16. Payment recording
17. No retry
18. No settlement after forwarding failure
19. Deterministic repeated execution
20. Functional API
"""


import math
import networkx as nx

from Simulation.payment_simulator import (
    PaymentSimulator,
    PaymentResult,
    estimate_inter_continent,
    simulate_payment
)


# ============================================================
# Test Doubles
# ============================================================

class SuccessFailureModel:
    """FailureModel that always accepts forwarding."""

    def __init__(self):
        self.calls = 0

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None
    ):
        self.calls += 1

        return {
            "success": True,
            "reason": "success",
            "visited_edges": list(
                route_edges
            )
        }


class RecordingRouteFailureModel(SuccessFailureModel):
    def __init__(self):
        super().__init__()
        self.routes = []

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None,
    ):
        self.routes.append((list(route), list(route_edges)))
        return super().evaluate_payment_failure(
            route,
            amount,
            network,
            route_edges,
        )


class SuffixFailureModel:
    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None,
    ):
        return {
            "success": False,
            "reason": "channel_failure",
            "failed_node": None,
            "failed_edge": route_edges[0],
            "failure_index": 0,
            "visited_edges": [],
        }


class FailureModel:
    """FailureModel that returns a deterministic failure."""

    def __init__(
        self,
        failed_node="B",
        failed_edge=("B", "C", 20),
        failure_index=1
    ):
        self.calls = 0

        self.failed_node = failed_node
        self.failed_edge = failed_edge
        self.failure_index = failure_index

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None
    ):
        self.calls += 1

        return {
            "success": False,
            "reason": "channel_failure",
            "failed_node": self.failed_node,
            "failed_edge": self.failed_edge,
            "failure_index": self.failure_index,
            "visited_edges": [
                route_edges[0]
            ]
        }


class InvalidFailureResultModel:

    def __init__(
        self,
        result
    ):
        self.result = result

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None
    ):
        return self.result


class ExceptionFailureModel:

    def evaluate_payment_failure(
        self,
        route,
        amount,
        network,
        route_edges=None
    ):
        raise RuntimeError(
            "intentional FailureModel error"
        )


class RecordingNetworkDynamics:

    def __init__(
        self,
        settlement_result=True
    ):
        self.settlement_result = settlement_result

        self.settlement_calls = []

        self.record_calls = []

    def settle_route(
        self,
        route_edges,
        amount
    ):
        self.settlement_calls.append(
            (
                list(route_edges),
                amount
            )
        )

        return self.settlement_result

    def record_payment(
        self,
        tx_id,
        result
    ):
        self.record_calls.append(
            (
                tx_id,
                result
            )
        )


class InvalidSettlementNetworkDynamics:

    def settle_route(
        self,
        route_edges,
        amount
    ):
        return "false"

    def record_payment(
        self,
        tx_id,
        result
    ):
        pass


class ExceptionSettlementNetworkDynamics:

    def settle_route(
        self,
        route_edges,
        amount
    ):
        raise RuntimeError(
            "intentional settlement error"
        )

    def record_payment(
        self,
        tx_id,
        result
    ):
        pass


class RecordingErrorNetworkDynamics:

    def __init__(self):
        self.settlement_calls = 0

    def settle_route(
        self,
        route_edges,
        amount
    ):
        self.settlement_calls += 1
        return True

    def record_payment(
        self,
        tx_id,
        result
    ):
        raise RuntimeError(
            "intentional recording error"
        )


# ============================================================
# Graph Factory
# ============================================================

def make_graph():

    G = nx.MultiDiGraph()

    # --------------------------------------------------------
    # A -> B : key 10
    # --------------------------------------------------------

    G.add_edge(
        "A",
        "B",
        key=10,

        fee_base_msat=1000,
        fee_proportional_millionths=100,

        cltv_expiry_delta=40,

        balance_uv=10000,
        capacity=20000,

        available=True
    )

    # --------------------------------------------------------
    # A -> B : key 99
    #
    # Deliberately different fee so exact channel identity
    # can be verified.
    # --------------------------------------------------------

    G.add_edge(
        "A",
        "B",
        key=99,

        fee_base_msat=999999,
        fee_proportional_millionths=999999,

        cltv_expiry_delta=999,

        balance_uv=10000,
        capacity=20000,

        available=True
    )

    # --------------------------------------------------------
    # B -> C : key 20
    # --------------------------------------------------------

    G.add_edge(
        "B",
        "C",
        key=20,

        fee_base_msat=2000,
        fee_proportional_millionths=200,

        cltv_expiry_delta=50,

        balance_uv=10000,
        capacity=20000,

        available=True
    )

    # --------------------------------------------------------
    # C -> D : key 30
    # --------------------------------------------------------

    G.add_edge(
        "C",
        "D",
        key=30,

        fee_base_msat=3000,
        fee_proportional_millionths=300,

        cltv_expiry_delta=60,

        balance_uv=10000,
        capacity=20000,

        available=True
    )

    # --------------------------------------------------------
    # Node attributes
    # --------------------------------------------------------

    G.nodes["A"].update(
        {
            "available": True,
            "is_online": True,
            "carbon_intensity": 100.0,
            "country": "US",
            "latitude": 40.0,
            "longitude": -74.0
        }
    )

    G.nodes["B"].update(
        {
            "available": True,
            "is_online": True,
            "carbon_intensity": 200.0,
            "country": "US",
            "latitude": 41.0,
            "longitude": -73.0
        }
    )

    G.nodes["C"].update(
        {
            "available": True,
            "is_online": True,
            "carbon_intensity": 300.0,
            "country": "CA",
            "latitude": 45.0,
            "longitude": -75.0
        }
    )

    G.nodes["D"].update(
        {
            "available": True,
            "is_online": True,
            "carbon_intensity": 400.0,
            "country": "FR",
            "latitude": 48.0,
            "longitude": 2.0
        }
    )

    return G


# ============================================================
# Common Route
# ============================================================

def valid_path():

    return [
        "A",
        "B",
        "C"
    ]


def valid_edges():

    return [
        ("A", "B", 10),
        ("B", "C", 20)
    ]


# ============================================================
# Assertion Helper
# ============================================================

def expect_exception(
    exception_type,
    function
):
    try:
        function()

    except exception_type:
        return True

    except Exception as exc:
        raise AssertionError(
            f"Expected {exception_type.__name__}, "
            f"got {type(exc).__name__}: {exc}"
        ) from exc

    raise AssertionError(
        f"Expected {exception_type.__name__}, "
        "but no exception was raised."
    )


# ============================================================
# Tests
# ============================================================

def test_01_constructor_rejects_none_graph():

    expect_exception(
        ValueError,
        lambda: PaymentSimulator(
            None,
            SuccessFailureModel(),
            RecordingNetworkDynamics()
        )
    )


def test_02_constructor_rejects_none_failure_model():

    G = make_graph()

    expect_exception(
        ValueError,
        lambda: PaymentSimulator(
            G,
            None,
            RecordingNetworkDynamics()
        )
    )


def test_03_constructor_rejects_none_network_dynamics():

    G = make_graph()

    expect_exception(
        ValueError,
        lambda: PaymentSimulator(
            G,
            SuccessFailureModel(),
            None
        )
    )


def test_04_valid_payment_result():

    result = PaymentResult(
        success=True,
        path=["A", "B"],
        edges=[("A", "B", 10)],
        fee=1.0,
        delay=40.0,
        carbon=100.0,
        inter_country_hops=0,
        inter_continent_hops=0,
        reason="success",
        elapsed=0.1,
        visited_edges=[
            ("A", "B", 10)
        ]
    )

    assert result.success is True
    assert result.reason == "success"
    assert result.fee == 1.0
    assert result.delay == 40.0
    assert result.visited_edges == [
        ("A", "B", 10)
    ]


def test_05_payment_result_rejects_non_bool_success():

    expect_exception(
        TypeError,
        lambda: PaymentResult(
            success="true",
            path=["A", "B"],
            edges=[("A", "B", 10)]
        )
    )


def test_06_payment_result_rejects_negative_fee():

    expect_exception(
        ValueError,
        lambda: PaymentResult(
            success=True,
            path=["A", "B"],
            edges=[("A", "B", 10)],
            fee=-1
        )
    )


def test_07_payment_result_rejects_nan_delay():

    expect_exception(
        ValueError,
        lambda: PaymentResult(
            success=True,
            path=["A", "B"],
            edges=[("A", "B", 10)],
            delay=math.nan
        )
    )


def test_08_invalid_amount():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        0
    )

    assert result.success is False
    assert result.reason == "invalid_amount"


def test_09_negative_amount():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        -100
    )

    assert result.success is False
    assert result.reason == "invalid_amount"


def test_10_nan_amount():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        math.nan
    )

    assert result.success is False
    assert result.reason == "invalid_amount"


def test_11_infinite_amount():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        math.inf
    )

    assert result.success is False
    assert result.reason == "invalid_amount"


def test_12_bool_amount_is_invalid():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        True
    )

    assert result.success is False
    assert result.reason == "invalid_amount"


def test_13_invalid_path_type():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path="ABC",
        edges=valid_edges(),
        amount=1000
    )

    assert result.success is False
    assert result.reason == "invalid_path"


def test_14_invalid_edges_type():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=valid_path(),
        edges="ABBC",
        amount=1000
    )

    assert result.success is False
    assert result.reason == "invalid_edges"


def test_15_short_path():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A"],
        edges=[],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "no_path"


def test_16_path_edge_mismatch():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "B", "C"],
        edges=[("A", "B", 10)],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "route_edge_mismatch"


def test_17_looped_route_rejected():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "B", "A"],
        edges=[
            ("A", "B", 10),
            ("B", "A", 10)
        ],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "route_contains_loop"


def test_18_dictionary_edge_format_is_supported():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "B", "C"],
        edges=[
            {"source": "A", "target": "B", "channel_key": 10},
            {"source": "B", "target": "C", "channel_key": 20}
        ],
        amount=1000
    )

    assert result.success is True
    assert result.reason == "success"


def test_19_missing_channel():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "X"],
        edges=[
            ("A", "X", 999)
        ],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "missing_channel"


def test_20_missing_channel_key():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "B", "C"],
        edges=[
            ("A", "B"),
            ("B", "C", 20)
        ],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "missing_channel_key"


def test_21_exact_parallel_channel_identity():

    G = make_graph()

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    data = simulator._get_edge(
        "A",
        "B",
        10
    )

    assert data["fee_base_msat"] == 1000
    assert data["fee_proportional_millionths"] == 100


def test_22_wrong_parallel_channel_key_rejected():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        path=["A", "B", "C"],
        edges=[
            ("A", "B", 777),
            ("B", "C", 20)
        ],
        amount=1000
    )

    assert result.success is False
    assert result.reason == "missing_channel"


def test_23_parallel_channel_without_key_never_selected():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    assert simulator._get_edge(
        "A",
        "B"
    ) is None


def test_24_fee_calculation():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    metrics = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    expected_fee = (
        1000
        + 100 * 1000 / 1_000_000
        +
        2000
        + 200 * 1000 / 1_000_000
    )

    assert math.isclose(
        metrics["fee"],
        expected_fee,
        rel_tol=1e-12,
        abs_tol=1e-12
    )


def test_25_delay_calculation():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    metrics = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    assert metrics["delay"] == 90.0


def test_26_carbon_calculation():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    metrics = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    # A-B:
    # (100 + 200) / 2 = 150
    #
    # B-C:
    # (200 + 300) / 2 = 250
    #
    # average = 200

    assert math.isclose(
        metrics["carbon"],
        200.0
    )


def test_27_country_hops():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    metrics = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    # A-B: US -> US = 0
    # B-C: US -> CA = 1

    assert metrics[
        "inter_country_hops"
    ] == 1


def test_28_continent_proxy():

    G = make_graph()

    assert estimate_inter_continent(
        G,
        "A",
        "B"
    ) == 0

    assert estimate_inter_continent(
        G,
        "B",
        "D"
    ) == 1


def test_29_missing_coordinates_do_not_create_fake_continent_hop():

    G = make_graph()

    G.nodes["A"].pop(
        "latitude"
    )

    assert estimate_inter_continent(
        G,
        "A",
        "B"
    ) == 0


def test_30_invalid_fee_is_rejected():

    G = make_graph()

    G["A"]["B"][10][
        "fee_base_msat"
    ] = "INVALID"

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator._calculate_metrics(
            valid_edges(),
            1000
        )
    )


def test_31_negative_fee_is_rejected():

    G = make_graph()

    G["A"]["B"][10][
        "fee_base_msat"
    ] = -1

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator._calculate_metrics(
            valid_edges(),
            1000
        )
    )


def test_32_invalid_delay_is_rejected():

    G = make_graph()

    G["A"]["B"][10][
        "cltv_expiry_delta"
    ] = "INVALID"

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator._calculate_metrics(
            valid_edges(),
            1000
        )
    )


def test_33_invalid_carbon_is_rejected():

    G = make_graph()

    G.nodes["A"][
        "carbon_intensity"
    ] = "INVALID"

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator._calculate_metrics(
            valid_edges(),
            1000
        )
    )


def test_34_failure_model_success():

    failure_model = SuccessFailureModel()

    simulator = PaymentSimulator(
        make_graph(),
        failure_model,
        RecordingNetworkDynamics()
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is True
    assert failure_model.calls == 1


def test_35_failure_propagation():

    failure_model = FailureModel()

    dynamics = RecordingNetworkDynamics()

    simulator = PaymentSimulator(
        make_graph(),
        failure_model,
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is False
    assert result.reason == "channel_failure"

    assert result.failed_node == "B"

    assert result.failed_edge == (
        "B",
        "C",
        20
    )

    assert result.failure_index == 1

    assert result.visited_edges == [
        ("A", "B", 10)
    ]

    assert len(
        dynamics.settlement_calls
    ) == 0


def test_36_bucket_continuation_forwards_only_suffix():
    graph = make_graph()
    failure_model = RecordingRouteFailureModel()
    dynamics = RecordingNetworkDynamics()
    simulator = PaymentSimulator(
        graph,
        failure_model,
        dynamics,
    )

    result = simulator.continue_payment(
        path=valid_path(),
        edges=valid_edges(),
        amount=1000,
        tx_id="TX-CONTINUE",
        forwarded_prefix_length=1,
    )

    assert result.success is True
    assert result.path == valid_path()
    assert result.visited_edges == valid_edges()
    assert failure_model.routes == [
        (["B", "C"], [("B", "C", 20)])
    ]
    assert dynamics.settlement_calls == [
        (valid_edges(), 1000.0)
    ]
    assert dynamics.record_calls[0][0] == "TX-CONTINUE"


def test_37_bucket_continuation_offsets_suffix_failure():
    simulator = PaymentSimulator(
        make_graph(),
        SuffixFailureModel(),
        RecordingNetworkDynamics(),
    )

    result = simulator.continue_payment(
        path=valid_path(),
        edges=valid_edges(),
        amount=1000,
        forwarded_prefix_length=1,
    )

    assert result.success is False
    assert result.failure_index == 1
    assert result.failed_edge == ("B", "C", 20)
    assert result.visited_edges == [("A", "B", 10)]


def test_36_failure_model_invalid_result_type():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            ["invalid"]
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        TypeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_37_failure_model_missing_success():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "reason": "success"
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_38_failure_model_non_bool_success():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": "false",
                "reason": "failure"
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        TypeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_39_failure_model_missing_reason():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": True
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_40_failure_model_invalid_failure_index():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": False,
                "reason": "failure",
                "failure_index": -1
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_41_failure_model_invalid_failed_edge():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": False,
                "reason": "failure",
                "failed_edge": "INVALID"
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_42_invalid_visited_edges_type():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": False,
                "reason": "failure",
                "visited_edges": "INVALID"
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        TypeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_43_visited_edge_outside_route():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": False,
                "reason": "failure",
                "visited_edges": [
                    ("A", "B", 99)
                ]
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_44_visited_edge_order_is_validated():

    simulator = PaymentSimulator(
        make_graph(),
        InvalidFailureResultModel(
            {
                "success": False,
                "reason": "failure",
                "visited_edges": [
                    ("B", "C", 20),
                    ("A", "B", 10)
                ]
            }
        ),
        RecordingNetworkDynamics()
    )

    expect_exception(
        ValueError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_45_successful_settlement_called_once():

    dynamics = RecordingNetworkDynamics(
        settlement_result=True
    )

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is True

    assert len(
        dynamics.settlement_calls
    ) == 1

    assert dynamics.settlement_calls[0][0] == [
        ("A", "B", 10),
        ("B", "C", 20)
    ]

    assert dynamics.settlement_calls[0][1] == 1000


def test_46_settlement_failure():

    dynamics = RecordingNetworkDynamics(
        settlement_result=False
    )

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is False
    assert result.reason == "settlement_failed"

    assert len(
        dynamics.settlement_calls
    ) == 1


def test_47_invalid_settlement_result():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        InvalidSettlementNetworkDynamics()
    )

    expect_exception(
        TypeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_48_settlement_exception_is_not_swallowed():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        ExceptionSettlementNetworkDynamics()
    )

    expect_exception(
        RuntimeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_49_failure_model_exception_is_not_swallowed():

    simulator = PaymentSimulator(
        make_graph(),
        ExceptionFailureModel(),
        RecordingNetworkDynamics()
    )

    expect_exception(
        RuntimeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000
        )
    )


def test_50_payment_recording():

    dynamics = RecordingNetworkDynamics()

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000,
        tx_id="TX-001"
    )

    assert result.success is True

    assert len(
        dynamics.record_calls
    ) == 1

    tx_id, recorded_result = (
        dynamics.record_calls[0]
    )

    assert tx_id == "TX-001"

    assert recorded_result[
        "success"
    ] is True

    assert recorded_result[
        "reason"
    ] == "success"


def test_51_recording_exception_is_not_swallowed():

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        RecordingErrorNetworkDynamics()
    )

    expect_exception(
        RuntimeError,
        lambda: simulator.simulate_payment(
            valid_path(),
            valid_edges(),
            1000,
            tx_id="TX-ERROR"
        )
    )


def test_52_no_recording_without_tx_id():

    dynamics = RecordingNetworkDynamics()

    simulator = PaymentSimulator(
        make_graph(),
        SuccessFailureModel(),
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000,
        tx_id=None
    )

    assert result.success is True

    assert len(
        dynamics.record_calls
    ) == 0


def test_53_no_settlement_after_failure():

    dynamics = RecordingNetworkDynamics()

    simulator = PaymentSimulator(
        make_graph(),
        FailureModel(),
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is False

    assert len(
        dynamics.settlement_calls
    ) == 0


def test_54_exactly_one_failure_model_call():

    failure_model = SuccessFailureModel()

    simulator = PaymentSimulator(
        make_graph(),
        failure_model,
        RecordingNetworkDynamics()
    )

    simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert failure_model.calls == 1


def test_55_no_retry():

    failure_model = FailureModel()

    dynamics = RecordingNetworkDynamics()

    simulator = PaymentSimulator(
        make_graph(),
        failure_model,
        dynamics
    )

    result = simulator.simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result.success is False

    assert failure_model.calls == 1

    assert len(
        dynamics.settlement_calls
    ) == 0


def test_56_deterministic_metrics():

    G = make_graph()

    simulator = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    )

    metrics_1 = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    metrics_2 = simulator._calculate_metrics(
        valid_edges(),
        1000
    )

    assert metrics_1 == metrics_2


def test_57_deterministic_repeated_payment():

    G = make_graph()

    result_1 = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    ).simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    result_2 = PaymentSimulator(
        G,
        SuccessFailureModel(),
        RecordingNetworkDynamics()
    ).simulate_payment(
        valid_path(),
        valid_edges(),
        1000
    )

    assert result_1.success == result_2.success
    assert result_1.path == result_2.path
    assert result_1.edges == result_2.edges
    assert result_1.fee == result_2.fee
    assert result_1.delay == result_2.delay
    assert result_1.carbon == result_2.carbon
    assert result_1.reason == result_2.reason


def test_58_functional_api():

    dynamics = RecordingNetworkDynamics()

    result = simulate_payment(
        G=make_graph(),
        path=valid_path(),
        edges=valid_edges(),
        amount=1000,
        failure_model=SuccessFailureModel(),
        network_dynamics=dynamics,
        tx_id="TX-FUNCTIONAL"
    )

    assert result.success is True
    assert result.reason == "success"

    assert len(
        dynamics.settlement_calls
    ) == 1

    assert len(
        dynamics.record_calls
    ) == 1


# ============================================================
# Test Runner
# ============================================================

TESTS = [
    (
        "constructor rejects None graph",
        test_01_constructor_rejects_none_graph
    ),
    (
        "constructor rejects None FailureModel",
        test_02_constructor_rejects_none_failure_model
    ),
    (
        "constructor rejects None NetworkDynamics",
        test_03_constructor_rejects_none_network_dynamics
    ),
    (
        "valid PaymentResult",
        test_04_valid_payment_result
    ),
    (
        "PaymentResult rejects non-bool success",
        test_05_payment_result_rejects_non_bool_success
    ),
    (
        "PaymentResult rejects negative fee",
        test_06_payment_result_rejects_negative_fee
    ),
    (
        "PaymentResult rejects NaN delay",
        test_07_payment_result_rejects_nan_delay
    ),
    (
        "invalid amount",
        test_08_invalid_amount
    ),
    (
        "negative amount",
        test_09_negative_amount
    ),
    (
        "NaN amount",
        test_10_nan_amount
    ),
    (
        "infinite amount",
        test_11_infinite_amount
    ),
    (
        "bool amount rejected",
        test_12_bool_amount_is_invalid
    ),
    (
        "invalid path type",
        test_13_invalid_path_type
    ),
    (
        "invalid edges type",
        test_14_invalid_edges_type
    ),
    (
        "short path",
        test_15_short_path
    ),
    (
        "path/edge mismatch",
        test_16_path_edge_mismatch
    ),
    (
        "looped route rejected",
        test_17_looped_route_rejected
    ),
    (
        "dictionary edge format",
        test_18_dictionary_edge_format_is_supported
    ),
    (
        "missing channel",
        test_19_missing_channel
    ),
    (
        "missing channel key",
        test_20_missing_channel_key
    ),
    (
        "exact parallel channel identity",
        test_21_exact_parallel_channel_identity
    ),
    (
        "wrong parallel channel key rejected",
        test_22_wrong_parallel_channel_key_rejected
    ),
    (
        "parallel channel without key rejected",
        test_23_parallel_channel_without_key_never_selected
    ),
    (
        "fee calculation",
        test_24_fee_calculation
    ),
    (
        "delay calculation",
        test_25_delay_calculation
    ),
    (
        "carbon calculation",
        test_26_carbon_calculation
    ),
    (
        "country hops",
        test_27_country_hops
    ),
    (
        "continent proxy",
        test_28_continent_proxy
    ),
    (
        "missing coordinates",
        test_29_missing_coordinates_do_not_create_fake_continent_hop
    ),
    (
        "invalid fee rejected",
        test_30_invalid_fee_is_rejected
    ),
    (
        "negative fee rejected",
        test_31_negative_fee_is_rejected
    ),
    (
        "invalid delay rejected",
        test_32_invalid_delay_is_rejected
    ),
    (
        "invalid carbon rejected",
        test_33_invalid_carbon_is_rejected
    ),
    (
        "FailureModel success",
        test_34_failure_model_success
    ),
    (
        "failure propagation",
        test_35_failure_propagation
    ),
    (
        "Bucket continuation forwards suffix only",
        test_36_bucket_continuation_forwards_only_suffix
    ),
    (
        "Bucket suffix failure index is global",
        test_37_bucket_continuation_offsets_suffix_failure
    ),
    (
        "invalid FailureModel result type",
        test_36_failure_model_invalid_result_type
    ),
    (
        "missing FailureModel success",
        test_37_failure_model_missing_success
    ),
    (
        "non-bool FailureModel success",
        test_38_failure_model_non_bool_success
    ),
    (
        "missing FailureModel reason",
        test_39_failure_model_missing_reason
    ),
    (
        "invalid failure index",
        test_40_failure_model_invalid_failure_index
    ),
    (
        "invalid failed edge",
        test_41_failure_model_invalid_failed_edge
    ),
    (
        "invalid visited edges type",
        test_42_invalid_visited_edges_type
    ),
    (
        "visited edge outside route",
        test_43_visited_edge_outside_route
    ),
    (
        "visited edge order",
        test_44_visited_edge_order_is_validated
    ),
    (
        "successful settlement",
        test_45_successful_settlement_called_once
    ),
    (
        "settlement failure",
        test_46_settlement_failure
    ),
    (
        "invalid settlement result",
        test_47_invalid_settlement_result
    ),
    (
        "settlement exception not swallowed",
        test_48_settlement_exception_is_not_swallowed
    ),
    (
        "FailureModel exception not swallowed",
        test_49_failure_model_exception_is_not_swallowed
    ),
    (
        "payment recording",
        test_50_payment_recording
    ),
    (
        "recording exception not swallowed",
        test_51_recording_exception_is_not_swallowed
    ),
    (
        "no recording without tx_id",
        test_52_no_recording_without_tx_id
    ),
    (
        "no settlement after failure",
        test_53_no_settlement_after_failure
    ),
    (
        "exactly one FailureModel call",
        test_54_exactly_one_failure_model_call
    ),
    (
        "no retry",
        test_55_no_retry
    ),
    (
        "deterministic metrics",
        test_56_deterministic_metrics
    ),
    (
        "deterministic repeated payment",
        test_57_deterministic_repeated_payment
    ),
    (
        "functional API",
        test_58_functional_api
    )
]


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print(" PAYMENT SIMULATOR DETERMINISTIC VALIDATION")
    print("=" * 72)

    passed = 0
    failed = 0

    total = len(TESTS)

    for index, (
        name,
        test_function
    ) in enumerate(
        TESTS,
        start=1
    ):

        try:

            test_function()

            print(
                f"[{index:02d}/{total}] "
                f"{name:<48} PASS"
            )

            passed += 1

        except Exception as exc:

            print(
                f"[{index:02d}/{total}] "
                f"{name:<48} FAIL"
            )

            print(
                f"          {type(exc).__name__}: {exc}"
            )

            failed += 1

    print("=" * 72)

    print(
        f"Total : {total}"
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
            "PAYMENT SIMULATOR VALIDATION: PASS"
        )

    else:

        print(
            "PAYMENT SIMULATOR VALIDATION: FAIL"
        )

        raise SystemExit(1)
