"""
Evaluations package

Contains:
- Baseline routing methods
- Evaluation pipeline
- Performance metrics
"""


"""Evaluation APIs, imported lazily to keep standalone tools lightweight."""

__all__ = [
    "evaluate",
    "compare_methods",
    "Metrics",
    "TransactionResult",
    "summarize",
    "evaluate_results",
    "run_method",
]


def __getattr__(name):
    if name in {"evaluate", "compare_methods"}:
        from .evaluate import compare_methods, evaluate

        return {"evaluate": evaluate, "compare_methods": compare_methods}[name]
    if name in {"Metrics", "TransactionResult", "summarize", "evaluate_results"}:
        from .metrics import Metrics, TransactionResult, evaluate_results, summarize

        return {
            "Metrics": Metrics,
            "TransactionResult": TransactionResult,
            "summarize": summarize,
            "evaluate_results": evaluate_results,
        }[name]
    if name == "run_method":
        from .baselines import run_method

        return run_method
    raise AttributeError(name)
