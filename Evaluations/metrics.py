import numpy as np

def summarize(rows):
    if not rows: return {}
    return {
        "payment_success_rate":float(np.mean([r["success"] for r in rows])),
        "failure_rate":float(1-np.mean([r["success"] for r in rows])),
        "average_path_length":float(np.mean([r["path_length"] for r in rows])),
        "average_fee":float(np.mean([r["fee"] for r in rows])),
        "average_delay":float(np.mean([r["delay"] for r in rows])),
        "average_carbon_intensity":float(np.mean([r["carbon"] for r in rows])),
        "average_inter_country_hops":float(np.mean([r["inter_country_hops"] for r in rows])),
        "average_inter_continent_hops":float(np.mean([r["inter_continent_hops"] for r in rows])),
        "average_runtime":float(np.mean([r["runtime"] for r in rows]))
    }
