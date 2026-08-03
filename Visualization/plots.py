import matplotlib.pyplot as plt

def plot_success(results,out="results/success_rate.png"):
    names=list(results); vals=[results[n]["payment_success_rate"] for n in names]
    plt.figure(figsize=(11,5)); plt.bar(names,vals); plt.ylabel("Payment Success Rate")
    plt.xticks(rotation=35,ha="right"); plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()
