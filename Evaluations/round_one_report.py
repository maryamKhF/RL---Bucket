"""Stream round-one XAI traces into portable heatmaps and readable reports."""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import html
import json
from pathlib import Path

import numpy as np

from Evaluations.explainable_analysis import summarize
from Evaluations.xai_report import FEATURE_LABELS, ROUTE_REPORT_FIELDS, human_explanation
from RL.state import FEATURE_NAMES
from Runtime.recovery_reporting import RECOVERY_COUNT_FIELDS
from Runtime.recovery_safety import atomic_json


def trace_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def refresh_trial_explanations(run_dir):
    """Refresh only prose in the flat export; preserve every numerical result."""
    from itertools import zip_longest
    run_dir = Path(run_dir)
    path = run_dir / "evaluation_trials.csv"
    if not path.is_file():
        return
    temporary = path.with_name(path.name + ".tmp")
    try:
        with path.open(encoding="utf-8", newline="") as source, \
                temporary.open("w", encoding="utf-8", newline="") as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row, record in zip_longest(reader, trace_records(run_dir / "evaluation_explanations.jsonl.gz")):
                if row is None or record is None or int(row["payment_seed"]) != record["payment_seed"]:
                    raise ValueError("Trial CSV and XAI trace identities/order differ; original CSV was retained.")
                human = human_explanation(record)
                row["human_explanation"] = " ".join(
                    human[key] for key in ("summary", "route_reason", "recovery_summary"))
                writer.writerow(row)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_heatmap(directory, filename, matrix, rows, columns, title, label, signed=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    array = np.asarray(matrix, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    maximum = float(finite.max()) if finite.size else 0
    limit = maximum if maximum > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(8, len(columns) * .9), max(4, len(rows) * .43)))
    image = ax.imshow(np.ma.masked_invalid(array), aspect="auto",
                      cmap="coolwarm" if signed else "magma",
                      vmin=-limit if signed else 0, vmax=limit)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    note = ("\nNo recorded values (missing, not zero)" if not finite.size else
            "\nAll recorded values are zero" if maximum == 0 else "")
    ax.set_title(title + note)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{filename}.{extension}", dpi=170)
    plt.close(fig)
    with (directory / f"{filename}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_or_metric", *columns])
        for name, values in zip(rows, array):
            writer.writerow([name, *[float(value) if np.isfinite(value) else "" for value in values]])


def write_round_one_reports(run_dir, rates):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    round_number = int(metadata.get("round", 1))
    round_label = f"main{round_number}/5"
    report_name = f"main{round_number}_5.html"
    continual = metadata.get("training_mode") == "continual"
    reports, figures = run_dir / "reports", run_dir / "figures"
    reports.mkdir(exist_ok=True)
    features = list(FEATURE_NAMES)
    totals = defaultdict(lambda: [0.0, 0])
    by_rate = defaultdict(list)
    identifiers = defaultdict(set)
    previews = []
    coverage = {}

    def policy_coverage(phase, rate, policy, routing):
        decision = routing.get("decision", {})
        fallback = ("partial_backtrack" if decision.get("partial_backtrack_retry_count", 0)
                    or decision.get("partial_backtrack_count", 0) else
                    "success" if decision.get("success") else "failure")
        sampling = policy.get("sampling", {})
        stratum, probability = sampling.get("stratum", fallback), float(sampling.get("probability", 1))
        sampled = bool(policy.get("eta_delta"))
        if sampled and (not 0 < probability <= 1 or
                        set(policy["eta_delta"]) != set(features)):
            raise ValueError("Invalid sampled policy explanation.")
        if policy.get("sampled", sampled) != sampled:
            raise ValueError("Policy sampling marker does not match the explanation.")
        key = (phase, rate, stratum)
        entry = coverage.setdefault(key, {"phase": phase, "failure_rate": rate, "stratum": stratum,
                                          "decisions": 0, "policy_samples": 0,
                                          "selection_probability": probability})
        if entry["selection_probability"] != probability:
            raise ValueError("A sampling probability changed inside one stratum.")
        entry["decisions"] += 1
        entry["policy_samples"] += int(sampled)
        return sampled, 1 / probability if sampled else None

    readable_fields = ["transaction", "training_seed", "evaluation_seed", "payment_seed",
                       "failure_rate", "success", "eta", "reason", "path",
                       "policy_explanation", "route_reason", "recovery_summary", "candidate_costs",
                       "policy_sampled", "sampling_stratum", "selection_probability",
                       *ROUTE_REPORT_FIELDS]
    with (reports / "evaluation_readable.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=readable_fields)
        writer.writeheader()
        for record in trace_records(run_dir / "evaluation_explanations.jsonl.gz"):
            rate = float(record["failure_rate"])
            if rate not in rates:
                raise ValueError("Unexpected failure-rate condition in XAI trace.")
            identifier = int(record["transaction_id"])
            if identifier in identifiers[rate]:
                raise ValueError("Duplicate held-out payment in a failure-rate condition.")
            identifiers[rate].add(identifier)
            by_rate[rate].append({key: value for key, value in record.items()
                                  if key not in ("explanation", "human")})
            policy = record["explanation"]["policy"]
            routing = record["explanation"]["routing"]
            sampled, weight = policy_coverage("evaluation", rate, policy, routing)
            for feature in features if sampled else []:
                for metric in ("eta_delta", "value_delta"):
                    value = float(policy[metric][feature])
                    if not np.isfinite(value):
                        raise ValueError("A nonfinite XAI sensitivity cannot be summarized.")
                    for kind, number in (("absolute", abs(value)), ("signed", value)):
                        key = (rate, feature, metric, kind)
                        totals[key][0] += number * weight
                        totals[key][1] += weight
            for bucket in routing.get("buckets", []):
                for candidate in bucket["candidates"]:
                    if candidate.get("cost") is not None:
                        key = (rate, candidate["rank"], "candidate_cost")
                        totals[key][0] += float(candidate["cost"])
                        totals[key][1] += 1
            human = human_explanation(record)
            candidate_costs = "; ".join(
                f"bucket {bucket.get('bucket_id')}: " + ", ".join(
                    f"rank {item['rank']} cost {item.get('cost')}" for item in bucket["candidates"])
                for bucket in routing.get("buckets", []))
            writer.writerow({
                "transaction": identifier, "failure_rate": rate, "success": record["success"],
                "training_seed": record["training_seed"],
                "evaluation_seed": record.get("evaluation_seed", metadata.get("evaluation_seed", metadata["training_seed"])),
                "payment_seed": record["payment_seed"],
                "eta": record["eta"], "reason": record["reason"],
                "path": json.dumps(record.get("path", [])),
                "policy_explanation": human["summary"], "route_reason": human["route_reason"],
                "recovery_summary": human["recovery_summary"], "candidate_costs": candidate_costs,
                "policy_sampled": sampled,
                "sampling_stratum": policy.get("sampling", {}).get("stratum", "exhaustive"),
                "selection_probability": policy.get("sampling", {}).get("probability", 1),
                **{key: human[key] for key in ROUTE_REPORT_FIELDS},
            })
            if len(previews) < 20:
                previews.append({"transaction": identifier, "rate": rate, **human})

    expected = int(metadata["held_out_transactions"])
    if any(len(identifiers[rate]) != expected for rate in rates):
        raise ValueError("Reports require complete held-out coverage at every requested failure rate.")
    if any(identifiers[rate] != identifiers[rates[0]] for rate in rates):
        raise ValueError("Failure-rate comparisons must use the identical held-out transaction pool.")

    def average(key):
        total, count = totals[key]
        return total / count if count else float("nan")

    labels = [FEATURE_LABELS[name] for name in features]
    columns = [f"{rate:.0%} (n={len(by_rate[rate])})" for rate in rates]
    policy_columns = [f"{rate:.0%} (sample={sum(entry['policy_samples'] for entry in coverage.values() if entry['failure_rate'] == rate)}/{len(by_rate[rate])})"
                      for rate in rates]
    for metric, kind, filename, title, label in (
        ("eta_delta", "absolute", "policy_eta_heatmap", f"{round_label}: held-out policy sensitivity",
         "Inverse-probability-weighted mean absolute eta difference (sample estimate)"),
        ("eta_delta", "signed", "policy_eta_direction_heatmap", f"{round_label}: direction of policy sensitivity",
         "Inverse-probability-weighted mean signed eta difference (sample estimate)"),
        ("value_delta", "absolute", "policy_value_heatmap", f"{round_label}: critic sensitivity",
         "Inverse-probability-weighted mean absolute critic difference (sample estimate)"),
    ):
        save_heatmap(figures, filename,
            [[average((rate, feature, metric, kind)) for rate in rates] for feature in features],
            labels, policy_columns, title, label, signed=kind == "signed")
    save_heatmap(figures, "candidate_cost_heatmap",
        [[average((rate, rank, "candidate_cost")) for rate in rates] for rank in range(1, 6)],
        [f"Bucket rank {rank}" for rank in range(1, 6)], columns,
        f"{round_label}: valid Bucket candidate costs", "Mean total routing cost (all created Buckets)")

    training = defaultdict(lambda: [0.0, 0])
    training_count = 0
    training_windows = set()
    training_samples = defaultdict(int)
    training_paths = sorted((run_dir / "models").glob("*/training_explanations.jsonl.gz"))
    if len(training_paths) != 1:
        raise ValueError("Round one must contain exactly one training audit trace.")
    with (reports / "training_readable.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "worker", "transaction", "sampled_eta", "deterministic_eta",
                         "explanation", "route_reason", "recovery_summary", "policy_sampled",
                         "sampling_stratum", "selection_probability", *ROUTE_REPORT_FIELDS])
        for record in trace_records(training_paths[0]):
            training_count += 1
            human = human_explanation(record)
            policy = record["policy"]
            sampled, weight = policy_coverage("training", float(metadata.get("training_failure_rate", .03)),
                                             policy, record["routing"])
            writer.writerow([record["timestep"], record["worker"], record["transaction_id"],
                             record["sampled_training_action"], policy.get("eta"),
                             human["summary"], human["route_reason"], human["recovery_summary"], sampled,
                             policy.get("sampling", {}).get("stratum", "exhaustive"),
                             policy.get("sampling", {}).get("probability", 1),
                             *[human[key] for key in ROUTE_REPORT_FIELDS]])
            window = (int(record["timestep"]) - 1) // 2000
            training_windows.add(window)
            training_samples[window] += int(sampled)
            for feature in features if sampled else []:
                key = (window, feature)
                value = float(policy["eta_delta"][feature])
                if not np.isfinite(value):
                    raise ValueError("A nonfinite XAI sensitivity cannot be summarized.")
                training[key][0] += abs(value) * weight
                training[key][1] += weight
    if training_count != (int(metadata["actual_training_transitions"])
                          - int(metadata.get("initial_training_transitions", 0))):
        raise ValueError("Every actual training transition must have an XAI audit record.")
    windows = sorted(training_windows)
    save_heatmap(figures, "training_eta_heatmap",
        [[training[(window, feature)][0] / training[(window, feature)][1]
          if training[(window, feature)][1] else float("nan")
          for window in windows] for feature in features],
        labels, [f"{window * 2000 + 1}-{(window + 1) * 2000} (sample={training_samples[window]})" for window in windows],
        f"{round_label}: sampled training sensitivity (online training-only reference)",
        "Inverse-probability-weighted mean absolute eta difference (sample estimate)")

    coverage_rows = sorted(coverage.values(), key=lambda row: (row["phase"], row["failure_rate"], row["stratum"]))
    with (reports / "policy_sampling_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "failure_rate", "stratum", "decisions",
                                                  "policy_samples", "selection_probability"])
        writer.writeheader()
        writer.writerows(coverage_rows)

    summaries = {}
    for rate in rates:
        rows = by_rate[rate]
        summary = summarize(rows)
        summary.update({"total_" + key: sum(int(row.get(key) or 0) for row in rows)
                        for key in (*RECOVERY_COUNT_FIELDS, "partial_backtrack_count",
                                    "partial_backtrack_success", "full_reroute_count")})
        summaries[f"{rate:.8g}"] = summary
    recovery_names = ["partial_backtrack_retry_count", "partial_backtrack_success",
                      "source_retry_count", "full_reroute_count"]
    save_heatmap(figures, "recovery_heatmap",
        [[100 * summaries[f"{rate:.8g}"]["total_" + name] / expected for rate in rates]
         for name in recovery_names],
        ["Partial continuation attempts", "Completed by partial continuation", "Source retries",
         "Full route recomputations"], columns,
        f"{round_label}: observed routing recovery", "Events per 100 held-out payments")
    payload = {
        "round": round_number, "total_rounds": 5, "independent_training_runs": 0 if continual else 1,
        "training_mode": "continual" if continual else "initial",
        "initial_training_transitions": int(metadata.get("initial_training_transitions", 0)),
        "cumulative_training_transitions": int(metadata["actual_training_transitions"]),
        "training_seed": metadata["training_seed"],
        "evaluation_seed": int(metadata.get("evaluation_seed", metadata["training_seed"])),
        "training_xai_records": training_count,
        "evaluation_xai_records": sum(len(rows) for rows in by_rate.values()),
        "training_policy_explanations": sum(row["policy_samples"] for row in coverage_rows if row["phase"] == "training"),
        "evaluation_policy_explanations": sum(row["policy_samples"] for row in coverage_rows if row["phase"] == "evaluation"),
        "policy_sampling_coverage": coverage_rows,
        "failure_rate_summaries": summaries,
        "interpretation": (
            "Continual training of the previous agent; stages are correlated and are not independent paper repeats. "
            "Held-out evaluation does not update policy weights or optimizer state." if continual else
            "One initial agent: later continual stages are not independent repetitions."),
        "xai_method": "group feature occlusion using training-only feature reference; not causal attribution or SHAP",
        "policy_heatmap_aggregation": "Self-normalized inverse-probability-weighted sample estimates; unsampled values are missing, never zero. Small or empty strata may be unrepresented; inspect coverage CSV.",
    }
    atomic_json(run_dir / "summary.json", payload)
    lines = [f"{round_label}: training/evaluation stage {round_number} of 5", "",
             f"Training seed: {metadata['training_seed']}",
             f"Evaluation base seed: {payload['evaluation_seed']}",
             f"Training workers: {metadata['training_workers']}; evaluation workers: {metadata['evaluation_workers']}",
             f"Audited training decisions: {training_count}",
             f"Audited held-out decisions: {payload['evaluation_xai_records']}", payload["interpretation"]]
    lines.extend([f"Training policy sensitivity samples: {payload['training_policy_explanations']}",
                  f"Held-out policy sensitivity samples: {payload['evaluation_policy_explanations']}",
                  payload["policy_heatmap_aggregation"], "Sampling coverage: reports/policy_sampling_coverage.csv"])
    for rate, summary in summaries.items():
        lines.extend(["", f"Failure probability: {float(rate):.0%}",
                      f"Payment success: {summary['success_rate_pct']:.2f}%"])
        for label, name in (
            ("Average reward (all payments)", "average_reward"),
            ("Average fee (successful)", "average_fee_success"),
            ("Average delay (successful)", "average_simulated_delay_success"),
            ("P95 delay (successful)", "p95_simulated_delay_success"),
            ("Average carbon intensity (successful)", "average_carbon_intensity_success"),
            ("Average hops (successful)", "average_path_hops_success"),
            ("Total delivered amount", "delivered_amount_sat"),
            ("Average attempts (all payments)", "average_attempts_all"),
            ("Path-search timeout (%)", "timeout_rate_pct"),
        ):
            lines.append(f"{label}: {summary[name]}")
        for name in recovery_names:
            lines.append(f"{name}: {summary['total_' + name]}")
    lines.extend(["", f"Open reports/{report_name} for heatmaps and the explanation guide.",
                  "Full per-payment explanations: reports/evaluation_readable.csv",
                  "All training explanations: reports/training_readable.csv"])
    (run_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    document = ["<!doctype html><html lang='fa' dir='rtl'><meta charset='utf-8'>",
        f"<title>گزارش {round_label}</title>" + "<style>body{font:17px system-ui;line-height:1.9;max-width:1100px;"
        "margin:30px auto;padding:20px}img{max-width:100%}code{direction:ltr;display:inline-block}"
        "article{border:1px solid #ccc;padding:15px;margin:15px 0}</style>",
        f"<h1>دور {round_number} از پنج دور: {round_label}</h1>",
        f"<p>seed آموزش: {metadata['training_seed']}؛ seed پایهٔ آزمون: {payload['evaluation_seed']}. "
        "بزرگ‌تر بودن عدد seed به معنی دشوارتر بودن آزمون نیست.</p>",
        "<p>عامل PPO مقدار eta را انتخاب می‌کند. مسیرهای معتبر با هزینهٔ مسیریابی مرتب می‌شوند؛ "
        "در شکست، نزدیک‌ترین گره قبلی با ادامهٔ معتبر انتخاب می‌شود و پیشوند موفق دوباره ارسال نمی‌شود.</p>",
        "<h2>خواندن heatmap</h2><p>در نمودار حساسیت، هر ویژگی با میانگین همان ویژگی در داده‌های آموزش "
        "جایگزین شده است. خانهٔ روشن‌تر یعنی تغییر بزرگ‌تر در تصمیم قطعی eta؛ این سنجشِ حساسیت است "
        "و سهم علّی ویژگی در موفقیت پرداخت را ثابت نمی‌کند. نمودار جهت، علامت تغییر را هم نشان می‌دهد. "
        "نقدینگی پنهان و احتمال واقعی خرابی وارد توضیح سیاست نمی‌شوند. نقشهٔ هزینه، رتبهٔ کاندیداها "
        "و نقشهٔ بازیابی، تلاش‌های واقعی عقبگرد را نشان می‌دهند.</p>",
        "<h2>رتبه و بازیابی مسیر</h2><p>شمارهٔ تلاش پرداخت با رتبهٔ کاندیدا متفاوت است. "
        "رتبهٔ مسیر بازیابی‌شده فقط با تطبیق مسیر کامل و شناسهٔ دقیق کانال‌ها تعیین می‌شود؛ "
        "مسیر ترکیبی ممکن است رتبهٔ اصلی نداشته باشد. صفر بودن گام برگشت یعنی ادامه از همان گره، "
        "نه نبود بازیابی. خلاصهٔ بازیابی، جست‌وجوهای ناموفق و نبود ادامهٔ معتبر را هم نشان می‌دهد.</p>",
        "<p>شرح مسیریابی برای تمام تصمیم‌ها ثبت شده است؛ تحلیل سنگین حساسیت فقط برای نمونه‌های منتخب انجام می‌شود. "
        "نمودارهای حساسیت برآورد نمونه‌ای با وزن معکوس احتمال انتخاب هستند؛ مقادیر محاسبه‌نشده صفر محسوب نمی‌شوند. "
        "ممکن است گروه‌های کم‌تعداد نمونه نداشته باشند. نمودارهای هزینه و عقبگرد از کل پرداخت‌ها استفاده می‌کنند.</p>",
        f"<p>نمونه‌های حساسیت آموزش: {payload['training_policy_explanations']} از {training_count}؛ "
        f"آزمون: {payload['evaluation_policy_explanations']} از {payload['evaluation_xai_records']}. "
        "<a href='policy_sampling_coverage.csv'>پوشش هر نرخ شکست و گروه نتیجه/عقبگرد</a></p>",
        "<p>خط مبنای آزمون فقط از rolloutهای آموزش محاسبه و سپس ثابت شده است. خط مبنای آموزش "
        "به‌صورت آنلاین به‌روزرسانی می‌شود. اقدام نمونه‌گیری‌شدهٔ آموزش و خروجی قطعی سیاست جدا ثبت شده‌اند.</p>",
        "<p>مراحل ادامهٔ آموزش یک عامل، مستقل نیستند؛ مقایسهٔ مراحل جایگزین پنج تکرار مستقل مقاله نمی‌شود. "
        "ارزیابی، وزن‌ها یا وضعیت بهینه‌ساز عامل را به‌روزرسانی نمی‌کند.</p>",
        "<p><a href='evaluation_readable.csv'>توضیح تمام پرداخت‌های آزمون</a> | "
        "<a href='training_readable.csv'>توضیح تمام تصمیم‌های آموزش</a> | "
        "<a href='../summary.json'>خلاصهٔ قابل انتقال</a></p>"]
    for path in sorted(figures.glob("*.png")):
        document.append(f"<figure><img src='../figures/{path.name}' alt='{path.stem}'>"
                        f"<figcaption>{path.stem} — <a href='../figures/{path.stem}.csv'>دادهٔ نمودار</a>"
                        f" | <a href='../figures/{path.stem}.svg'>SVG</a></figcaption></figure>")
    document.append("<h2>نمونهٔ توضیح تصمیم‌ها؛ فهرست کامل در CSV</h2>")
    for record in previews:
        text = " ".join(str(record[key]) for key in ("summary", "route_reason", "recovery_summary"))
        document.append(f"<article dir='ltr'>Transaction {record['transaction']} / "
                        f"failure {record['rate']:.0%}<p>{html.escape(text)}</p></article>")
    document.append("</html>")
    (reports / report_name).write_text("\n".join(document), encoding="utf-8")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild reports from saved XAI traces; no training or evaluation.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    write_round_one_reports(args.run_dir, metadata["failure_rates"])
    refresh_trial_explanations(args.run_dir)
