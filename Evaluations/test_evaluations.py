"""
Test file for Evaluations module.

Tests:
- metrics calculation
- summary generation
- evaluation pipeline interface
"""


import numpy as np


from Evaluations import (
    Metrics,
    TransactionResult,
    summarize
)





# ============================================================
# Test Data Generator
# ============================================================


def create_test_results():

    return [

        {
            "tx_id": "tx_001",
            "success": True,
            "path_length": 4,
            "fee": 0.002,
            "delay": 25.5,
            "carbon": 40.0,
            "inter_country_hops": 1,
            "inter_continent_hops": 0,
            "runtime": 0.12
        },


        {
            "tx_id": "tx_002",
            "success": True,
            "path_length": 6,
            "fee": 0.004,
            "delay": 40.0,
            "carbon": 60.0,
            "inter_country_hops": 2,
            "inter_continent_hops": 1,
            "runtime": 0.20
        },


        {
            "tx_id": "tx_003",
            "success": False,
            "path_length": 0,
            "fee": 0,
            "delay": 0,
            "carbon": 0,
            "inter_country_hops": 0,
            "inter_continent_hops": 0,
            "runtime": 0.08
        }

    ]





# ============================================================
# Test summarize()
# ============================================================


def test_summarize():


    print("\n===== TEST summarize() =====")


    rows = create_test_results()


    result = summarize(rows)



    for key,value in result.items():

        print(
            f"{key}: {value}"
        )



    assert np.isclose(

        result["payment_success_rate"],

        2/3

    )



    assert np.isclose(

        result["failure_rate"],

        1/3

    )



    assert np.isclose(

        result["average_path_length"],

        10/3

    )



    assert np.isclose(

        result["average_fee"],

        0.002

    )



    print(
        "summarize() PASSED"
    )







# ============================================================
# Test Metrics Class
# ============================================================


def test_metrics_class():


    print(
        "\n===== TEST Metrics Class ====="
    )



    transactions = []



    for row in create_test_results():


        transactions.append(

            TransactionResult(

                tx_id=row["tx_id"],

                success=row["success"],

                fee=row["fee"],

                delay=row["delay"],

                path_length=row["path_length"],

                carbon=row["carbon"],

                inter_country_hops=row["inter_country_hops"],

                inter_continent_hops=row["inter_continent_hops"],

                runtime=row["runtime"]

            )

        )





    evaluator = Metrics(

        transactions

    )



    report = evaluator.compute_all_metrics()



    for key,value in report.items():

        print(
            f"{key}: {value}"
        )




    assert np.isclose(

        report["payment_success_rate"],

        2/3

    )



    assert np.isclose(

        report["average_path_length"],

        10/3

    )



    print(
        "Metrics Class PASSED"
    )






# ============================================================
# Main
# ============================================================


if __name__ == "__main__":


    test_summarize()


    test_metrics_class()



    print(
        "\n===== ALL EVALUATION TESTS PASSED ====="
    )