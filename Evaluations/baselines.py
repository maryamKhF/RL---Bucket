from ..pathfinding.dijkstra import shortest_path
from ..pathfinding.heuristics import lnd_cost,cln_cost,ecl_cost
from ..pathfinding.top_k_paths import top_k_paths
from ..bucket.bucket import Bucket,execute_bucket
from ..simulation.payment_simulator import simulate_payment
from ..simulation.onion import build_onion
import numpy as np

HEURISTICS={"native_lnd":lnd_cost,"native_cln":cln_cost,"native_ecl":ecl_cost}
STATIC_ETA={"static_lnd":0.27876,"static_cln":0.25556,"static_ecl":0.35784}

def run_method(G,transactions,name,cfg,model=None,seed=42):
    rng=np.random.default_rng(seed); rows=[]
    if name.startswith("native_"): hf=HEURISTICS[name]; eta=0
    elif name.startswith("static_"):
        hf=HEURISTICS[name.replace("static_","native_")]; eta=STATIC_ETA[name]
    else:
        hf=HEURISTICS[name.replace("improved_","native_")]; eta=None
    for tx in transactions:
        if model is not None:
            from ..rl.state import neighborhood_state
            obs=neighborhood_state(G,tx.source,tx.destination,cfg["graph"]["neighborhood_k"],cfg["graph"]["neighborhood_m"],cfg["graph"]["ego_radius"])
            action,_=model.predict(obs,deterministic=True); eta=float(action[0])
        paths=top_k_paths(G,tx.source,tx.destination,tx.amount,hf,eta or 0,cfg["graph"]["top_k_paths"],cfg["graph"]["max_hops"]) if name.startswith("improved") else []
        if name.startswith("improved") and paths:
            b=Bucket(tx.tx_id,tx.tx_id,paths)
            success,candidate,attempts=execute_bucket(G,b,tx.amount,rng)
            if success:
                path,edges,_=candidate
                # Bucket execution already settled; metrics only.
                fee=sum(G.edges[e]["fee_base"]+G.edges[e]["fee_rate"]*tx.amount/1e6 for e in edges)
                rows.append({"success":True,"path_length":len(edges),"fee":fee,"delay":sum(G.edges[e]["delay"] for e in edges),
                             "carbon":np.mean([ (G.nodes[e[0]]["carbon_intensity"]+G.nodes[e[1]]["carbon_intensity"])/2 for e in edges]),
                             "inter_country_hops":sum(G.nodes[e[0]]["country"]!=G.nodes[e[1]]["country"] for e in edges),
                             "inter_continent_hops":0,"runtime":0})
            else: rows.append({"success":False,"path_length":0,"fee":0,"delay":0,"carbon":0,"inter_country_hops":0,"inter_continent_hops":0,"runtime":0})
            continue
        path,edges,_=shortest_path(G,tx.source,tx.destination,tx.amount,hf,eta or 0,cfg["graph"]["max_hops"])
        r=simulate_payment(G,path,edges,tx.amount,rng) if path else None
        rows.append({"success":bool(r and r.success),"path_length":len(edges) if r else 0,"fee":r.fee if r else 0,
                     "delay":r.delay if r else 0,"carbon":r.carbon if r else 0,
                     "inter_country_hops":r.inter_country_hops if r else 0,
                     "inter_continent_hops":r.inter_continent_hops if r else 0,
                     "runtime":r.elapsed if r else 0})
    return rows
