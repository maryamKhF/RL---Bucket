"""Scientific summaries and exportable figures for routing explanations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


RATES = tuple(i / 100 for i in range(1, 7))


def mean(values):
    values = [float(value) for value in values if value is not None
              and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def summarize(rows):
    """Keep success-conditioned metrics separate from failed or unrouted cases."""
    successful = [row for row in rows if row["success"]]
    routed = [row for row in rows if int(row.get("attempt_count") or 0) > 0]
    result = {
        "count": len(rows), "successful_count": len(successful),
        "success_rate_pct": 100 * len(successful) / len(rows) if rows else None,
        "delivered_amount_sat": sum(row["amount"] for row in successful),
        "attempted_amount_sat": sum(row["amount"] for row in rows),
        "average_successful_amount_sat": mean(row["amount"] for row in successful),
        "average_reward": mean(row["reward"] for row in rows),
        "average_path_hops_success": mean(row["path_length"] for row in successful),
        "average_path_hops_routed": mean(row["path_length"] for row in routed),
        "average_carbon_intensity_success": mean(row["carbon"] for row in successful),
        "average_route_carbon_sum_success": mean(
            row["carbon"] * (row["path_length"] + 1) for row in successful),
        "average_fee_success": mean(row["fee"] for row in successful),
        "average_simulated_delay_success": mean(row["delay"] for row in successful),
        "average_wall_seconds_all": mean(row["pipeline_seconds"] for row in rows),
        "average_attempts_all": mean(row["attempt_count"] for row in rows),
        "timeout_rate_pct": 100 * sum(bool(row.get("path_search_timed_out"))
                                      for row in rows) / len(rows) if rows else None,
        "failure_reasons": dict(Counter(str(row.get("reason"))
                                        for row in rows if not row["success"])),
    }
    result["delivered_amount_pct"] = (
        100 * result["delivered_amount_sat"] / result["attempted_amount_sat"]
        if result["attempted_amount_sat"] else None)
    for name, subset in (("success", successful), ("routed", routed)):
        for percentile in (90, 95):
            result[f"p{percentile}_simulated_delay_{name}"] = (
                float(np.percentile([row["delay"] for row in subset], percentile))
                if subset else None)
    return result


def ci95(values):
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(clean):
        return {"mean": None, "low": None, "high": None, "n": 0}
    center = float(clean.mean())
    width = (float(student_t.ppf(0.975, len(clean) - 1) * clean.std(ddof=1)
                   / np.sqrt(len(clean))) if len(clean) > 1 else None)
    return {"mean": center, "low": center - width if width is not None else None,
            "high": center + width if width is not None else None, "n": len(clean)}


def wilson(successes, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (
        1 + z * z / count)
    return 100 * (center - width), 100 * (center + width)


def assess_learning(points, tolerance_pp=2.0, patience=2):
    """Retrospective warning, never a model selection or stopping rule on test data."""
    ordered = sorted(points, key=lambda point: point["checkpoint"])
    if not ordered:
        return {"status": "insufficient_diagnostics", "suspected_overfit_step": None}
    best_index = max(range(len(ordered)), key=lambda i: ordered[i]["test_success_pct"])
    best = ordered[0]
    consecutive = 0
    onset = None
    for point in ordered[1:]:
        low_gap = point["train_ci_low"] - point["test_ci_high"]
        deterioration = best["test_success_pct"] - point["test_success_pct"]
        warning = (deterioration >= tolerance_pp and low_gap > 0
                   and point["train_success_pct"] >= best["train_success_pct"])
        consecutive = consecutive + 1 if warning else 0
        if consecutive >= patience and onset is None:
            onset = point["checkpoint"]
        if point["test_success_pct"] > best["test_success_pct"]:
            best = point
    tail = ordered[-min(3, len(ordered)):]
    plateau = (len(tail) >= 3 and max(p["test_success_pct"] for p in tail)
               - min(p["test_success_pct"] for p in tail) < tolerance_pp)
    return {
        "status": "suspected_overfitting" if onset is not None else (
            "observed_plateau" if plateau else "no_clear_overfitting_evidence"),
        "suspected_overfit_step": onset,
        "best_observed_test_step_retrospective": ordered[best_index]["checkpoint"],
        "best_observed_test_success_pct": ordered[best_index]["test_success_pct"],
        "plateau_window_steps": [p["checkpoint"] for p in tail] if plateau else [],
        "rule": f"{patience} consecutive checkpoints: test drop >= {tolerance_pp} pp, "
                "train not falling, train Wilson lower bound > test Wilson upper bound",
        "limitation": "diagnostic evidence only; sparse checkpoints cannot locate an exact onset; "
                      "multiple looks/rates are exploratory, not a corrected hypothesis test; "
                      "test data never selects checkpoints or stops training",
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})


def create_analysis(rows, output_dir):
    """All points derive from recorded transactions, never invented learning curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    diagnostics = defaultdict(dict)
    for row in rows:
        if row["phase"] == "final":
            grouped[(row["repetition"], row["failure_rate"])].append(row)
        else:
            key = (row["repetition"], row["failure_rate"], row["checkpoint"])
            diagnostics[key].setdefault(row["split"], []).append(row)
    seed_summaries = [{"repetition": rep, "failure_rate": rate, **summarize(batch)}
                      for (rep, rate), batch in sorted(grouped.items())]
    metrics = [key for key, value in summarize([]).items()
               if key not in ("count", "successful_count", "failure_reasons")]
    across_seeds = {f"{rate:.2f}": {
        metric: ci95([row[metric] for row in seed_summaries
                      if row["failure_rate"] == rate]) for metric in metrics}
        for rate in RATES}
    curve_points = []
    for (rep, rate, step), splits in sorted(diagnostics.items()):
        if "train" not in splits or "test" not in splits:
            continue
        train, test = summarize(splits["train"]), summarize(splits["test"])
        train_ci = wilson(train["successful_count"], train["count"])
        test_ci = wilson(test["successful_count"], test["count"])
        curve_points.append({
            "repetition": rep, "failure_rate": rate, "checkpoint": step,
            "train_count": train["count"], "test_count": test["count"],
            "train_success_pct": train["success_rate_pct"],
            "test_success_pct": test["success_rate_pct"],
            "train_reward": train["average_reward"], "test_reward": test["average_reward"],
            "gap_pp": train["success_rate_pct"] - test["success_rate_pct"],
            "train_ci_low": train_ci[0], "train_ci_high": train_ci[1],
            "test_ci_low": test_ci[0], "test_ci_high": test_ci[1],
        })
    warnings = {}
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=170)
        fig.savefig(figures / f"{name}.svg")
        plt.close(fig)

    comparisons = {
        "success_rate_vs_channel_failure_probability": ("success_rate_pct", "Payment success (%)"),
        "amount_vs_channel_failure_probability": ("delivered_amount_sat", "Total delivered amount (sat)"),
        "average_amount_vs_channel_failure_probability": ("average_successful_amount_sat", "Mean successful amount (sat)"),
        "channel_failure_probability_vs_average_path_length": ("average_path_hops_success", "Mean successful path length (hops)"),
        "carbon_vs_channel_failure_probability": ("average_carbon_intensity_success", "Mean route carbon intensity (successful)"),
        "fee_vs_channel_failure_probability": ("average_fee_success", "Mean successful fee (sat)"),
        "reward_vs_channel_failure_probability": ("average_reward", "Mean reward (all payments)"),
        "timeout_vs_channel_failure_probability": ("timeout_rate_pct", "Search timeout (%)"),
        "wall_time_vs_channel_failure_probability": ("average_wall_seconds_all", "Mean measured pipeline time (s)"),
    }
    for filename, (metric, label) in comparisons.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for rep in sorted({row["repetition"] for row in seed_summaries}):
            values = [row for row in seed_summaries if row["repetition"] == rep]
            ax.plot([row["failure_rate"] * 100 for row in values],
                    [np.nan if row[metric] is None else row[metric] for row in values],
                    alpha=0.4, marker=".", label=f"Repetition {rep}")
        stats = [across_seeds[f"{rate:.2f}"][metric] for rate in RATES]
        y = [s["mean"] if s["mean"] is not None else np.nan for s in stats]
        ax.plot(np.array(RATES) * 100, y, "ko-", label="Mean across seeds")
        ax.fill_between(np.array(RATES) * 100,
                        [s["low"] if s["low"] is not None else np.nan for s in stats],
                        [s["high"] if s["high"] is not None else np.nan for s in stats],
                        color="black", alpha=0.12, label="95% t CI across seeds")
        ax.set(xlabel="Mean channel failure probability (%)", ylabel=label,
               title="RL + Bucket / original paper reward")
        ax.legend(fontsize=8, ncol=2)
        save(fig, filename)

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, label in (("p90_simulated_delay_success", "P90 successful"),
                          ("p95_simulated_delay_success", "P95 successful"),
                          ("p95_simulated_delay_routed", "P95 all routed")):
        ax.plot(np.array(RATES) * 100,
                [across_seeds[f"{r:.2f}"][metric]["mean"] for r in RATES],
                marker="o", label=label)
    ax.set(xlabel="Mean channel failure probability (%)", ylabel="Simulated delay (model units)",
           title="Delay percentiles averaged across repetitions")
    ax.legend()
    save(fig, "delay_percentiles_vs_channel_failure_probability")

    for rep in sorted({point["repetition"] for point in curve_points}):
        for kind in ("success", "reward", "gap"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
            for rate, ax in zip(RATES, axes.flat):
                points = [p for p in curve_points if p["repetition"] == rep
                          and p["failure_rate"] == rate]
                if not points:
                    ax.set_title(f"Failure {rate:.0%}: no diagnostics")
                    continue
                assessment = assess_learning(points)
                warnings[f"rep_{rep}_rate_{rate:.2f}"] = assessment
                steps = [p["checkpoint"] for p in points]
                if kind == "gap":
                    ax.plot(steps, [p["gap_pp"] for p in points], "o-", color="purple")
                    ax.fill_between(steps,
                                    [p["train_ci_low"] - p["test_ci_high"] for p in points],
                                    [p["train_ci_high"] - p["test_ci_low"] for p in points],
                                    color="purple", alpha=0.1)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("Train minus test (pp)")
                else:
                    for split, color in (("train", "#1565c0"), ("test", "#d84315")):
                        key = f"{split}_success_pct" if kind == "success" else f"{split}_reward"
                        ax.plot(steps, [p[key] for p in points], "o-", label=split, color=color)
                        if kind == "success":
                            ax.fill_between(steps, [p[f"{split}_ci_low"] for p in points],
                                            [p[f"{split}_ci_high"] for p in points],
                                            color=color, alpha=0.1)
                    ax.set_ylabel("Payment success (%)" if kind == "success" else "Mean reward")
                onset = assessment["suspected_overfit_step"]
                if onset is not None:
                    ax.axvline(onset, color="red", linestyle="--", label="Suspected overfit")
                plateau_steps = assessment["plateau_window_steps"]
                if plateau_steps:
                    ax.axvspan(plateau_steps[0], plateau_steps[-1], color="gold", alpha=0.15,
                               label="Observed plateau window")
                ax.set_title(f"Rep {rep} | failure {rate:.0%}\n{assessment['status']}", fontsize=10)
                ax.set_xlabel("Completed PPO transitions")
                if kind != "gap" or onset is not None or plateau_steps:
                    ax.legend(fontsize=8)
            fig.suptitle("Fixed train/test diagnostic subsets; test never controls training", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            save(fig, f"rep_{rep:02d}_train_test_{kind}")

    importance = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["phase"] == "final":
            for feature, delta in row["explanation"]["policy"]["eta_delta"].items():
                importance[row["repetition"]][feature].append(abs(delta))
    fig, ax = plt.subplots(figsize=(11, 6))
    features = list(next(iter(importance.values()), {}))
    scores = [ci95([mean(by_feature[feature]) for by_feature in importance.values()])
              for feature in features]
    write_csv(output_dir / "policy_feature_sensitivity.csv", [
        {"feature": feature, **score} for feature, score in zip(features, scores)])
    order = sorted(range(len(features)), key=lambda i: scores[i]["mean"] or 0)
    ax.barh([features[i] for i in order], [scores[i]["mean"] for i in order], color="#1565c0")
    ax.set(xlabel="Mean absolute change in deterministic eta under feature occlusion",
           title="Policy input sensitivity / training-only reference / not a causal explanation")
    save(fig, "policy_feature_sensitivity")

    final_rows = [row for row in rows if row["phase"] == "final"]
    if final_rows:
        # Use the same amount bins for every failure rate and repetition.
        amounts = np.array([row["amount"] for row in final_rows])
        bounds = np.geomspace(max(amounts.min(), 1e-6), max(amounts.max(), amounts.min() * 1.01), 9)
        heatmap = np.full((len(RATES), len(bounds) - 1), np.nan)
        for rate_index, rate in enumerate(RATES):
            for bin_index, (low, high) in enumerate(zip(bounds[:-1], bounds[1:])):
                subset = [row for row in final_rows if row["failure_rate"] == rate
                          and low <= row["amount"] <= high
                          and (row["amount"] < high or bin_index == len(bounds) - 2)]
                if subset:
                    heatmap[rate_index, bin_index] = 100 * mean(row["success"] for row in subset)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(heatmap, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(bounds) - 1),
                      [f"{low:.0f}-{high:.0f}" for low, high in zip(bounds[:-1], bounds[1:])],
                      rotation=35, ha="right")
        ax.set_yticks(range(len(RATES)), [f"{rate:.0%}" for rate in RATES])
        ax.set(xlabel="Payment amount bin (sat)", ylabel="Channel failure probability",
               title="Payment success by amount and failure probability / pooled repetitions")
        fig.colorbar(im, ax=ax, label="Payment success (%)")
        save(fig, "success_rate_by_amount_and_failure_probability")

    write_csv(output_dir / "metrics_by_seed_and_rate.csv", seed_summaries)
    write_csv(output_dir / "learning_curves.csv", curve_points)
    from Evaluations.xai_report import write_xai_figures
    write_xai_figures(rows, output_dir)
    result = {"by_seed_and_rate": seed_summaries, "across_five_seeds": across_seeds,
              "learning_assessment": warnings,
              "figure_count": len(list(figures.glob("*.png")))}
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    images = "\n".join(f'<figure><img src="figures/{p.name}" loading="lazy">'
                       f'<figcaption>{p.stem}</figcaption></figure>'
                       for p in sorted(figures.glob("*.png")))
    (output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Main8 results</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}'
        'img{width:100%;height:auto}figure{margin:24px 0;border-top:1px solid #ddd}</style>'
        '<h1>Main8: RL + Bucket</h1><p>Five repetitions; same 7,000/3,000 transaction split; '
        'failure probabilities 1-6%. Confidence intervals use independent repetitions. '
        'Diagnostic train/test curves are descriptive and never select the final model. '
        'Overfitting warnings are conditional, not proof. Feature occlusion explains policy '
        'sensitivity, not the hidden cause of a failed payment. Missing values remain missing.</p>'
        + '<h2>Human-readable reports by repetition</h2>'
        + ''.join(f'<p><a href="reports/{path.name}">{path.stem}</a></p>'
                  for path in sorted((output_dir / "reports").glob("rep_*.html")))
        + images + '</html>', encoding="utf-8")
    return result
