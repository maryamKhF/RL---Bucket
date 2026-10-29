"""
Evaluations/metrics.py

Evaluation metrics for RL + Bucket routing framework.

This module processes simulation outputs and calculates
performance indicators such as:
- success rate
- fee
- delay
- path length
- recovery
- carbon intensity
- geographic routing metrics
- runtime
"""


from __future__ import annotations


from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional, Any


import numpy as np



# ============================================================
# Transaction Result Schema
# ============================================================


@dataclass
class TransactionResult:
    """
    Standard format for a completed transaction.

    Simulation outputs should be converted into this format
    before advanced evaluation.
    """


    tx_id: str

    success: bool

    fee: float

    delay: float

    path_length: int


    # Bucket / Backtracking information

    attempts: int = 1

    recovered: bool = False


    # Payment information

    amount: float = 0.0


    # Network metrics

    carbon: float = 0.0

    inter_country_hops: int = 0

    inter_continent_hops: int = 0


    # Execution information

    runtime: float = 0.0


    route: Optional[List[str]] = None


    failure_reason: Optional[str] = None





# ============================================================
# Helper Functions
# ============================================================


def _safe_mean(values):

    """
    Safe average calculation.
    """

    if not values:
        return 0.0

    return float(mean(values))



def _safe_std(values):

    """
    Safe standard deviation.
    """

    if len(values) <= 1:
        return 0.0

    return float(
        pstdev(values)
    )



def _count_true(values):

    return sum(
        1
        for v in values
        if v
    )





# ============================================================
# Metrics Class
# ============================================================


class Metrics:
    """
    Advanced evaluation engine.

    Receives transaction results and calculates
    routing performance metrics.
    """


    def __init__(
        self,
        results: List[TransactionResult]
    ):

        self.results = results



    # --------------------------------------------------------
    # Basic Statistics
    # --------------------------------------------------------


    @property
    def total_transactions(self):

        return len(
            self.results
        )



    @property
    def successful_transactions(self):

        return _count_true(
            [
                r.success
                for r in self.results
            ]
        )



    @property
    def failed_transactions(self):

        return (
            self.total_transactions
            -
            self.successful_transactions
        )

    # --------------------------------------------------------
    # Reliability Metrics
    # --------------------------------------------------------


    def success_rate(self) -> float:
        """
        Ratio of successful payments.
        """

        if self.total_transactions == 0:
            return 0.0

        return (
            self.successful_transactions
            /
            self.total_transactions
        )



    def failure_rate(self) -> float:
        """
        Ratio of failed payments.
        """

        if self.total_transactions == 0:
            return 0.0

        return (
            self.failed_transactions
            /
            self.total_transactions
        )



    def recovery_rate(self) -> float:
        """
        Percentage of failures recovered
        by Bucket backtracking.
        """

        failures = [
            r
            for r in self.results
            if not r.success
        ]


        if not failures:
            return 0.0


        recovered = _count_true(
            [
                r.recovered
                for r in failures
            ]
        )


        return (
            recovered
            /
            len(failures)
        )



    # --------------------------------------------------------
    # Cost Metrics
    # --------------------------------------------------------


    def average_fee(self) -> float:

        return _safe_mean(
            [
                r.fee
                for r in self.results
            ]
        )



    def total_fee(self) -> float:

        return float(
            sum(
                r.fee
                for r in self.results
            )
        )



    # --------------------------------------------------------
    # Delay Metrics
    # --------------------------------------------------------


    def average_delay(self) -> float:

        return _safe_mean(
            [
                r.delay
                for r in self.results
            ]
        )



    def delay_std(self) -> float:

        return _safe_std(
            [
                r.delay
                for r in self.results
            ]
        )



    # --------------------------------------------------------
    # Path Metrics
    # --------------------------------------------------------


    def average_path_length(self) -> float:

        return _safe_mean(
            [
                r.path_length
                for r in self.results
            ]
        )



    def path_length_std(self) -> float:

        return _safe_std(
            [
                r.path_length
                for r in self.results
            ]
        )



    # --------------------------------------------------------
    # Bucket / Backtracking Metrics
    # --------------------------------------------------------


    def average_attempts(self) -> float:

        return _safe_mean(
            [
                r.attempts
                for r in self.results
            ]
        )



    def maximum_attempts(self) -> int:

        if not self.results:
            return 0


        return max(
            r.attempts
            for r in self.results
        )



    # --------------------------------------------------------
    # Environmental Metrics
    # --------------------------------------------------------


    def average_carbon_intensity(self) -> float:

        return _safe_mean(
            [
                r.carbon
                for r in self.results
            ]
        )



    # --------------------------------------------------------
    # Geographic Routing Metrics
    # --------------------------------------------------------


    def average_inter_country_hops(self) -> float:

        return _safe_mean(
            [
                r.inter_country_hops
                for r in self.results
            ]
        )



    def average_inter_continent_hops(self) -> float:

        return _safe_mean(
            [
                r.inter_continent_hops
                for r in self.results
            ]
        )



    # --------------------------------------------------------
    # Runtime Metrics
    # --------------------------------------------------------


    def average_runtime(self) -> float:

        return _safe_mean(
            [
                r.runtime
                for r in self.results
            ]
        )



    # --------------------------------------------------------
    # Transaction Volume
    # --------------------------------------------------------


    def total_volume(self) -> float:

        return float(
            sum(
                r.amount
                for r in self.results
            )
        )



    # --------------------------------------------------------
    # Throughput
    # --------------------------------------------------------


    def throughput(
        self,
        simulation_time: float
    ) -> float:
        """
        Successful transactions per second.
        """

        if simulation_time <= 0:

            return 0.0


        return (
            self.successful_transactions
            /
            simulation_time
        )



    # --------------------------------------------------------
    # Failure Analysis
    # --------------------------------------------------------


    def failure_reasons(self) -> Dict[str, int]:

        reasons = {}


        for result in self.results:

            if not result.success:

                reason = (
                    result.failure_reason
                    if result.failure_reason
                    else "unknown"
                )


                reasons[reason] = (
                    reasons.get(reason, 0)
                    +
                    1
                )


        return reasons
    # --------------------------------------------------------
    # Complete Metrics Report Interface
    # --------------------------------------------------------

    def compute_all_metrics(
        self,
        simulation_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Class wrapper for complete evaluation report.

        Allows usage:

            evaluator.compute_all_metrics()

        """

        return {

            # Reliability

            "total_transactions":
                self.total_transactions,


            "successful_transactions":
                self.successful_transactions,


            "failed_transactions":
                self.failed_transactions,


            "payment_success_rate":
                self.success_rate(),


            "failure_rate":
                self.failure_rate(),


            "recovery_rate":
                self.recovery_rate(),



            # Routing Performance

            "average_path_length":
                self.average_path_length(),


            "average_fee":
                self.average_fee(),


            "average_delay":
                self.average_delay(),



            # Bucket Metrics

            "average_attempts":
                self.average_attempts(),


            "maximum_attempts":
                self.maximum_attempts(),



            # Environment

            "average_carbon_intensity":
                self.average_carbon_intensity(),



            # Geography

            "average_inter_country_hops":
                self.average_inter_country_hops(),


            "average_inter_continent_hops":
                self.average_inter_continent_hops(),



            # Runtime

            "average_runtime":
                self.average_runtime(),



            # Volume

            "total_volume":
                self.total_volume(),



            # Throughput

            "throughput":
                self.throughput(
                    simulation_time
                ),



            # Failures

            "failure_reasons":
                self.failure_reasons()
        }
    
# ============================================================
# Complete Metrics Report
# ============================================================


def compute_all_metrics(
    evaluator: Metrics,
    simulation_time: float = 0.0
) -> Dict[str, Any]:
    """
    Generate complete evaluation report.
    """


    return {


        # -------------------------
        # Reliability
        # -------------------------

        "total_transactions":
            evaluator.total_transactions,


        "successful_transactions":
            evaluator.successful_transactions,


        "failed_transactions":
            evaluator.failed_transactions,


        "payment_success_rate":
            evaluator.success_rate(),


        "failure_rate":
            evaluator.failure_rate(),


        "recovery_rate":
            evaluator.recovery_rate(),



        # -------------------------
        # Routing Performance
        # -------------------------

        "average_path_length":
            evaluator.average_path_length(),


        "average_fee":
            evaluator.average_fee(),


        "average_delay":
            evaluator.average_delay(),



        # -------------------------
        # Bucket Metrics
        # -------------------------

        "average_attempts":
            evaluator.average_attempts(),


        "maximum_attempts":
            evaluator.maximum_attempts(),



        # -------------------------
        # Network / Geography
        # -------------------------

        "average_carbon_intensity":
            evaluator.average_carbon_intensity(),


        "average_inter_country_hops":
            evaluator.average_inter_country_hops(),


        "average_inter_continent_hops":
            evaluator.average_inter_continent_hops(),



        # -------------------------
        # Runtime
        # -------------------------

        "average_runtime":
            evaluator.average_runtime(),



        # -------------------------
        # Volume
        # -------------------------

        "total_volume":
            evaluator.total_volume(),



        # -------------------------
        # Throughput
        # -------------------------

        "throughput":
            evaluator.throughput(
                simulation_time
            ),



        # -------------------------
        # Failure Details
        # -------------------------

        "failure_reasons":
            evaluator.failure_reasons()
    }





# ============================================================
# Raw Simulation Converter
# ============================================================


def build_transaction_results(
    rows: List[Dict[str, Any]]
) -> List[TransactionResult]:
    """
    Convert simulation dictionary outputs
    into TransactionResult objects.
    """


    results = []


    for row in rows:


        result = TransactionResult(

            tx_id=row.get(
                "tx_id",
                "unknown"
            ),


            success=row.get(
                "success",
                False
            ),


            fee=row.get(
                "fee",
                0.0
            ),


            delay=row.get(
                "delay",
                0.0
            ),


            path_length=row.get(
                "path_length",
                0
            ),


            attempts=row.get(
                "attempts",
                1
            ),


            recovered=row.get(
                "recovered",
                False
            ),


            amount=row.get(
                "amount",
                0.0
            ),


            carbon=row.get(
                "carbon",
                0.0
            ),


            inter_country_hops=row.get(
                "inter_country_hops",
                0
            ),


            inter_continent_hops=row.get(
                "inter_continent_hops",
                0
            ),


            runtime=row.get(
                "runtime",
                0.0
            ),


            route=row.get(
                "route",
                None
            ),


            failure_reason=row.get(
                "failure_reason",
                None
            )
        )


        results.append(result)


    return results





# ============================================================
# Main Evaluation Interface
# ============================================================


def evaluate_results(
    rows: List[Dict[str, Any]],
    simulation_time: float = 0.0
) -> Dict[str, Any]:
    """
    Main function used by evaluate.py.
    """


    transactions = build_transaction_results(
        rows
    )


    evaluator = Metrics(
        transactions
    )


    return compute_all_metrics(
        evaluator,
        simulation_time
    )





# ============================================================
# Lightweight Summary Function
# ============================================================


def summarize(
    rows: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Fast summary of simulation outputs.

    Compatible with raw Simulation results.
    """


    if not rows:

        return {}



    def avg(key):

        return float(
            np.mean(
                [
                    r.get(
                        key,
                        0.0
                    )
                    for r in rows
                ]
            )
        )



    success_rate = float(
        np.mean(
            [
                r.get(
                    "success",
                    False
                )
                for r in rows
            ]
        )
    )



    return {


        "payment_success_rate":
            success_rate,


        "failure_rate":
            1 - success_rate,


        "average_path_length":
            avg(
                "path_length"
            ),


        "average_fee":
            avg(
                "fee"
            ),


        "average_delay":
            avg(
                "delay"
            ),


        "average_carbon_intensity":
            avg(
                "carbon"
            ),


        "average_inter_country_hops":
            avg(
                "inter_country_hops"
            ),


        "average_inter_continent_hops":
            avg(
                "inter_continent_hops"
            ),


        "average_runtime":
            avg(
                "runtime"
            )
    }