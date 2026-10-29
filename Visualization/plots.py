# Visualization/plots.py

"""
Visualization plots for RL + Bucket routing framework.

This module converts evaluation metrics into
publication-ready figures.

Input:
    results dictionary from Evaluations/metrics.py

Output:
    PNG figures for analysis and paper presentation
"""


from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

def _prepare_output(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )


# ==================================================
# Success Rate Plot
# ==================================================

def plot_success(
        results,
        out="results/success_rate.png"
):

    names = list(results)

    values = [
        results[n]["payment_success_rate"]
        for n in names
    ]

    _prepare_output(out)

    plt.figure(figsize=(11,5))

    plt.bar(
        names,
        values
    )

    plt.ylabel(
        "Payment Success Rate"
    )

    plt.xlabel(
        "Routing Method"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=150
    )

    plt.close()



# ==================================================
# Fee Comparison
# ==================================================

def plot_fee(
        results,
        out="results/fee.png"
):

    names=list(results)

    values=[
        results[n]["average_fee"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))

    plt.bar(
        names,
        values
    )

    plt.ylabel(
        "Average Fee"
    )

    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()

    plt.savefig(
        out,
        dpi=150
    )

    plt.close()



# ==================================================
# Delay Comparison
# ==================================================

def plot_delay(
        results,
        out="results/delay.png"
):

    names=list(results)


    values=[
        results[n]["average_delay"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))


    plt.bar(
        names,
        values
    )


    plt.ylabel(
        "Average Delay"
    )


    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# Path Length
# ==================================================

def plot_path_length(
        results,
        out="results/path_length.png"
):

    names=list(results)


    values=[
        results[n]["average_path_length"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))


    plt.bar(
        names,
        values
    )


    plt.ylabel(
        "Average Path Length"
    )


    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# Recovery Performance
# ==================================================

def plot_recovery(
        results,
        out="results/recovery.png"
):

    names=list(results)


    values=[
        results[n]["recovery_rate"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))


    plt.bar(
        names,
        values
    )


    plt.ylabel(
        "Recovery Rate"
    )


    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# Carbon Intensity
# ==================================================

def plot_carbon(
        results,
        out="results/carbon.png"
):

    names=list(results)


    values=[
        results[n]["carbon_intensity"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))


    plt.bar(
        names,
        values
    )


    plt.ylabel(
        "Carbon Intensity"
    )


    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# Runtime Comparison
# ==================================================

def plot_runtime(
        results,
        out="results/runtime.png"
):

    names=list(results)


    values=[
        results[n]["runtime"]
        for n in names
    ]


    _prepare_output(out)


    plt.figure(figsize=(11,5))


    plt.bar(
        names,
        values
    )


    plt.ylabel(
        "Runtime (seconds)"
    )


    plt.xlabel(
        "Routing Method"
    )


    plt.xticks(
        rotation=35,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# RL Training Reward Curve
# ==================================================

def plot_reward(
        rewards,
        out="results/reward_curve.png"
):

    _prepare_output(out)


    plt.figure(figsize=(10,5))


    plt.plot(
        rewards
    )


    plt.xlabel(
        "Episode"
    )


    plt.ylabel(
        "Reward"
    )


    plt.title(
        "RL Training Reward"
    )


    plt.tight_layout()


    plt.savefig(
        out,
        dpi=150
    )


    plt.close()



# ==================================================
# Complete Visualization Pipeline
# ==================================================

def generate_all_plots(results):

    plot_success(results)

    plot_fee(results)

    plot_delay(results)

    plot_path_length(results)

    plot_recovery(results)

    plot_carbon(results)

    plot_runtime()