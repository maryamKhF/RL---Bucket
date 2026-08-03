import copy, json
from pathlib import Path
from ..network.graph_builder import LNGraphBuilder
from ..simulation.failure_model import assign_failure_probabilities
from ..evaluation.baselines import run_method
from ..evaluation.metrics import summarize
from ..pathfinding.heuristics import lnd_cost,cln_cost,ecl_cost

def evaluate(G,transactions,cfg,models=None):
    results={}
    methods=["native_lnd","native_cln","native_ecl","static_lnd","static_cln","static_ecl"]
    if models:
        methods += [f"improved_{x}" for x in models]
    for method in methods:
        name=method.replace("improved_","")
        model=models.get(name) if models and method.startswith("improved_") else None
        rows=run_method(G,transactions,method,cfg,model=model,seed=cfg["seed"])
        results[method]=summarize(rows)
    Path("results").mkdir(exist_ok=True)
    Path("results/summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    return results
