from dataclasses import dataclass
from .backtrack import choose_alternative

@dataclass
class Bucket:
    bucket_id:int
    transaction_id:int
    candidates:list
    current_index:int=0

    def current(self):
        return self.candidates[self.current_index] if self.candidates else None

    def backtrack(self, failed_node=None):
        self.current_index += 1
        return self.current()

def execute_bucket(G,bucket,amount,rng):
    attempts=0
    while bucket.current() is not None:
        attempts += 1
        candidate=bucket.current()
        path,edges,_=candidate
        failed=False
        for u,v,k in edges:
            d=G.edges[u,v,k]
            if (not d.get("available",True) or d.get("balance_uv",0)<amount or
                rng.random()<d.get("failure_probability",0.01)):
                failed=True; break
        if not failed:
            for u,v,k in edges:
                d=G.edges[u,v,k]
                d["balance_uv"]=max(0,d["balance_uv"]-amount)
                d["balance_vu"]=min(d["capacity"],d["balance_vu"]+amount)
                d["success_count"]+=1
            return True, candidate, attempts
        bucket.backtrack()
    return False,None,attempts
