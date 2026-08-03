"""
Evaluations package

Contains:
- Baseline routing methods
- Evaluation pipeline
- Performance metrics
"""


from .evaluate import evaluate, compare_methods

from .metrics import (
    Metrics,
    TransactionResult,
    summarize,
    evaluate_results
)


from .baselines import run_method



__all__ = [

    # Evaluation
    "evaluate",
    "compare_methods",


    # Metrics
    "Metrics",
    "TransactionResult",
    "summarize",
    "evaluate_results",


    # Baselines
    "run_method"

]