"""
Evaluations/evaluate.py

Main evaluation controller.

This module executes different routing methods,
collects simulation results and generates
comparison reports.
"""


from __future__ import annotations


import json

from pathlib import Path

from typing import Dict, Any, Optional



from .baselines import run_method

from .metrics import summarize





# ============================================================
# Evaluation Configuration
# ============================================================


DEFAULT_METHODS = [

    "native_lnd",

    "native_cln",

    "native_ecl",

    "static_lnd",

    "static_cln",

    "static_ecl"

]





# ============================================================
# Main Evaluation Function
# ============================================================


def evaluate(
    G,
    transactions,
    cfg: Dict[str, Any],
    models: Optional[Dict[str, Any]] = None
):
    """
    Run complete evaluation pipeline.

    Parameters
    ----------
    G:
        Network graph

    transactions:
        Generated payment transactions

    cfg:
        Configuration dictionary

    models:
        Optional RL models dictionary

        Example:

        {
            "lnd": model,
            "cln": model
        }


    Returns
    -------
    dict

        Evaluation summary
    """


    results = {}



    # ---------------------------------
    # Build method list
    # ---------------------------------


    methods = DEFAULT_METHODS.copy()



    if models:


        for model_name in models:

            methods.append(

                f"improved_{model_name}"

            )





    # ---------------------------------
    # Execute Methods
    # ---------------------------------


    for method in methods:


        model = None



        if (
            models
            and
            method.startswith(
                "improved_"
            )
        ):


            model_name = method.replace(

                "improved_",

                ""

            )


            model = models.get(

                model_name

            )



        print(
            f"Running evaluation method: {method}"
        )



        rows = run_method(

            G,

            transactions,

            method,

            cfg,

            model=model,

            seed=cfg.get(
                "seed",
                42
            )

        )



        metrics = summarize(

            rows

        )



        results[method] = metrics





    # ---------------------------------
    # Save Results
    # ---------------------------------


    save_results(

        results

    )


    return results





# ============================================================
# Result Storage
# ============================================================


def save_results(
    results: Dict[str, Any],
    directory="results"
):
    """
    Save evaluation results as JSON.
    """


    path = Path(
        directory
    )


    path.mkdir(

        exist_ok=True

    )



    file = path / "summary.json"



    file.write_text(

        json.dumps(

            results,

            indent=4

        ),

        encoding="utf-8"

    )





# ============================================================
# Comparison Utility
# ============================================================


def compare_methods(
    results: Dict[str, Any],
    metric="payment_success_rate"
):
    """
    Rank methods by selected metric.
    """


    ranking = sorted(

        results.items(),

        key=lambda x:

            x[1].get(

                metric,

                0

            ),

        reverse=True

    )


    return ranking