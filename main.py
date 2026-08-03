import argparse, copy, yaml, random, numpy as np
from pathlib import Path
from network.graph_builder import LNGraphBuilder
from simulation.failure_model import assign_failure_probabilities
from simulation.transaction_generator import generate_transactions
from pathfinding.heuristics import lnd_cost,cln_cost,ecl_cost
from rl.train import train_agent
from evaluation.evaluate import evaluate
from visualization.plots import plot_success

def load_cfg():
    return yaml.safe_load(Path("configs/config.yaml").read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshot",default=None,help="Generic LN JSON snapshot; optional.")
    ap.add_argument("--train",action="store_true")
    ap.add_argument("--evaluate",action="store_true")
    ap.add_argument("--transactions",type=int,default=None)
    args=ap.parse_args()
    cfg=load_cfg(); seed=cfg["seed"]; random.seed(seed); np.random.seed(seed)
    builder=LNGraphBuilder()
    if args.snapshot:
        G=builder.from_json(args.snapshot)
    else:
        G=builder.synthetic(cfg["graph"]["synthetic_nodes"],cfg["graph"]["synthetic_extra_edges"],seed)
        print("No snapshot supplied: using synthetic LN-like graph.")
    n=args.transactions or cfg["n_transactions"]
    tx=generate_transactions(G,n,seed,cfg["simulation"]["min_amount"],cfg["simulation"]["max_amount"])
    split=int(n*cfg["train_ratio"]); train_tx,test_tx=tx[:split],tx[split:]
    models={}
    if args.train:
        for name,hf in [("lnd",lnd_cost),("cln",cln_cost),("ecl",ecl_cost)]:
            Gt=copy.deepcopy(G)
            assign_failure_probabilities(Gt,0.03,seed)
            models[name]=train_agent(Gt,train_tx,hf,cfg,name,seed)
    if args.evaluate or args.train:
        for rate in cfg["failure_rates"]:
            Ge=copy.deepcopy(G); assign_failure_probabilities(Ge,rate,seed)
            res=evaluate(Ge,test_tx,cfg,models if models else None)
            plot_success(res,f"results/success_{int(rate*100)}pct.png")
            print(f"Failure rate {rate:.0%}:"); print(res)
if __name__=="__main__":
    main()
