"""Readable observational explanations; no claims of causal attribution."""

from collections import defaultdict
import csv
import gzip
import html
import json
from pathlib import Path

FEATURE_LABELS = {
    "fee_base": "Base fee", "fee_rate": "Fee rate", "capacity": "Channel capacity",
    "delay": "Delay", "availability": "Observed availability",
    "liquidity_lower_bound": "Estimated liquidity lower bound",
    "liquidity_upper_bound": "Estimated liquidity upper bound",
    "geographic_distance": "Geographic distance", "inter_country": "Cross-country channel",
    "inter_continent": "Cross-continent channel", "carbon_intensity": "Carbon intensity",
}


ROUTE_REPORT_FIELDS = (
    "successful_bucket_rank", "successful_bucket_id", "bucket_rank_status",
    "payment_attempt_count", "partial_backtrack_retry_count", "source_retry_count",
    "preserved_prefix_hops", "backtracked_hops",
)


def _physical_attempt(routing, attempt_number):
    attempts = routing.get("payment_attempts", [])
    if not isinstance(attempt_number, int) or not 1 <= attempt_number <= len(attempts):
        return None
    channels = attempts[attempt_number - 1].get("channels", [])
    edges = [item.get("channel") for item in channels]
    if not edges or any(not isinstance(edge, (list, tuple)) or len(edge) != 3 for edge in edges):
        return None
    return [list(edge) for edge in edges]


def _bucket_rank(routing, path, attempt_number, recorded_rank=None):
    """Match complete routes AND physical channels; never guess from node paths."""
    edges = _physical_attempt(routing, attempt_number)
    matches = []
    if path and edges:
        for bucket in routing.get("buckets", []):
            for candidate in bucket.get("candidates", []):
                channels = candidate.get("channels")
                if (list(candidate.get("path") or []) == list(path)
                        and channels and [list(edge) for edge in channels] == edges):
                    matches.append((bucket.get("bucket_id"), candidate.get("rank")))
        if len(matches) == 1:
            bucket_id, rank = matches[0]
            return rank, bucket_id, "exact physical route match"
        if matches:
            return None, None, "ambiguous: matches multiple recorded Bucket candidates"
        return None, None, "not an exact complete Bucket candidate (combined/recomputed route)"
    if recorded_rank is not None:
        return recorded_rank, None, "recorded by routing engine"
    return None, None, "unknown: physical attempt details were not recorded"


def human_explanation(record):
    details = record.get("explanation", record)
    policy, routing = details.get("policy", {}), details.get("routing", {})
    deltas = policy.get("eta_delta", {})
    ranked = sorted(deltas, key=lambda name: abs(deltas[name]), reverse=True)
    decision, reward = routing.get("decision", {}), routing.get("reward", {})
    success = record.get("success", decision.get("success", reward.get("S")))
    outcome = "SUCCESS" if success else "FAILED" if success is not None else "UNKNOWN"
    top = ", ".join(FEATURE_LABELS.get(name, name) for name in ranked[:3]) or "no data"
    recoveries = decision.get("recovery_events", [])
    attempts = decision.get("attempt_count", record.get("attempt_count", 0))
    rank, bucket_id, rank_status = (_bucket_rank(
        routing, decision.get("path", record.get("path")), attempts,
        decision.get("successful_bucket_rank")) if success else
        (None, None, "not applicable: payment did not succeed"))
    recovery_lines = []
    for event in recoveries:
        kind = event.get("type", "unknown")
        if kind in ("partial_backtrack", "source_retry"):
            event_rank, event_bucket, status = _bucket_rank(
                routing, event.get("new_route"), event.get("attempt_id"))
            movement = ("no backward hops: continue from the already reached node"
                        if kind == "partial_backtrack" and event.get("backtracked_hops") == 0 else
                        "retry from source" if kind == "source_retry" else "backward hops were required")
            recovery_lines.append(
                f"{kind}: payment attempt #{event.get('attempt_id')} (not candidate rank); "
                f"Bucket candidate rank={event_rank if event_rank is not None else 'N/A'}, "
                f"bucket={event_bucket}; {status}; branch={event.get('branch_node')}; "
                f"preserved_prefix_hops={event.get('preserved_prefix_hops', 'unknown')}; "
                f"backtracked_hops={event.get('backtracked_hops', 'unknown')} ({movement}); "
                f"payment_success={event.get('success')}; reason={event.get('reason')}")
        elif kind == "recovery_unavailable":
            recovery_lines.append(
                f"Recovery unavailable after payment attempt #{event.get('attempt_id')}: "
                f"reason={event.get('reason')}; status={event.get('status', 'not recorded')}; "
                "no additional payment attempt was executed by this event")
        elif kind == "full_reroute":
            recovery_lines.append(
                f"Full reroute search after payment attempt #{event.get('after_attempt')}: "
                f"route_found={event.get('success')} (search outcome, not payment success); "
                f"reason={event.get('reason')}")
        else:
            recovery_lines.append(f"Recovery event {kind}: {json.dumps(event, ensure_ascii=False)}")
    recovery_summary = "; ".join(recovery_lines)
    route_reason = (
        f"PPO chooses eta={decision.get('eta')}; Bucket considers valid routes in ascending cost order. "
        f"Successful route source={decision.get('successful_route_source')}, "
        f"Bucket rank={rank if rank is not None else 'N/A'} ({rank_status}); "
        f"actual payment attempts={attempts} (payment attempts are not candidate ranks)."
    )
    sensitivity = (f"Largest deterministic policy sensitivities: {top}." if deltas else
                   "Full routing audit recorded; policy sensitivity was not sampled for this decision.")
    return {
        "summary": f"Payment outcome: {outcome}. {sensitivity}",
        "actual_eta": record.get("sampled_training_action", record.get("eta", decision.get("eta"))),
        "deterministic_eta": policy.get("eta"),
        "observed_reason": record.get("reason", decision.get("reason")),
        "reward": reward.get("actual_reward", record.get("reward")),
        "route_reason": route_reason,
        "recovery_summary": recovery_summary or "No recovery attempt recorded.",
        "successful_bucket_rank": rank,
        "successful_bucket_id": bucket_id,
        "bucket_rank_status": rank_status,
        "payment_attempt_count": attempts,
        **{key: decision.get(key) for key in ROUTE_REPORT_FIELDS[4:]},
        "reward_formula": "success * route_node_count * 1000 / summed_route_carbon_intensity; failure: zero",
        "features": [{"feature": name, "label": FEATURE_LABELS.get(name, name),
                      "eta_original_minus_replaced": deltas[name],
                      "critic_original_minus_replaced": policy.get("value_delta", {}).get(name)}
                     for name in ranked],
        "interpretation": "Sensitivity replaces each feature group with its training-only mean. "
            "This does not establish a causal contribution to payment success. "
            "The sampled training action is distinct from the deterministic policy mean.",
    }


def write_xai_figures(rows, output_dir):
    """Aggregate existing explanations; never retain training JSONL in memory."""
    import numpy as np
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    features = list(FEATURE_LABELS)
    repetitions = sorted({row["repetition"] for row in rows})
    if not repetitions:
        return
    means = defaultdict(lambda: [0.0, 0])
    for row in rows:
        for feature, delta in row["explanation"]["policy"]["eta_delta"].items():
            key = (row["repetition"], row["phase"], row["split"], row["failure_rate"], feature)
            means[key][0] += abs(delta)
            means[key][1] += 1

    def heatmap(values, columns, title, filename):
        fig, ax = plt.subplots(figsize=(max(9, len(columns) * 0.65), 7))
        array = np.asarray(values, dtype=float)
        vmax = float(np.nanmax(array)) if np.isfinite(array).any() else 1.0
        image = ax.imshow(array, aspect="auto", vmin=0, vmax=max(vmax, 1e-12), cmap="magma")
        ax.set_yticks(range(len(features)), features)
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="Mean absolute deterministic eta sensitivity")
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / "figures" / f"{filename}.{suffix}", dpi=160)
        plt.close(fig)

    for rep in repetitions:
        for phase, split in (("final", "test"), ("diagnostic", "train"), ("diagnostic", "test")):
            matrix = []
            for feature in features:
                line = []
                for rate in (.01, .02, .03, .04, .05, .06):
                    total, count = means[(rep, phase, split, rate, feature)]
                    line.append(total / count if count else np.nan)
                matrix.append(line)
            heatmap(matrix, ["1%", "2%", "3%", "4%", "5%", "6%"],
                    f"Repetition {rep} / {phase} {split} / feature sensitivity by failure rate",
                    f"rep_{rep:02d}_{phase}_{split}_feature_heatmap")
        matching = [row for row in rows if row["repetition"] == rep and row["phase"] == "final"]
        csv_path = report_dir / f"rep_{rep:02d}_evaluation_readable.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["transaction", "failure_rate", "success", "eta", "reason", "explanation"])
            for row in matching:
                writer.writerow([row["transaction_id"], row["failure_rate"], row["success"], row.get("eta"),
                                 row.get("reason"), human_explanation(row)["summary"]])
        # Only process the current repetition's training trace. Earlier reports
        # are carried by the previous job's artifacts, avoiding repeated work.
        if rep == max(repetitions):
            traces = list((output_dir / "models").glob(f"rep_{rep:02d}_seed_*/training_explanations.jsonl.gz"))
            groups = defaultdict(lambda: [0.0, 0])
            with (report_dir / f"rep_{rep:02d}_training_readable.csv").open(
                    "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestep", "worker", "transaction", "sampled_eta", "explanation"])
                for path in traces:
                    with gzip.open(path, "rt", encoding="utf-8") as source:
                        for line in source:
                            record = json.loads(line)
                            writer.writerow([record["timestep"], record["worker"], record["transaction_id"],
                                record.get("sampled_training_action"),
                                (record.get("human") or human_explanation(record))["summary"]])
                            bucket = (int(record["timestep"]) - 1) // 5000 + 1
                            for feature, delta in record["policy"]["eta_delta"].items():
                                groups[(bucket, feature)][0] += abs(delta)
                                groups[(bucket, feature)][1] += 1
            bins = sorted({key[0] for key in groups})
            if bins:
                heatmap([[groups[(bucket, feature)][0] / groups[(bucket, feature)][1]
                          if groups[(bucket, feature)][1] else np.nan for bucket in bins]
                         for feature in features], [f"{b*5000:,}" for b in bins],
                        f"Repetition {rep} / training / online training-only reference",
                        f"rep_{rep:02d}_training_feature_heatmap")
        conditions = sorted({row["failure_rate"] for row in matching})
        statistics = []
        for rate in conditions:
            cases = [row for row in matching if row["failure_rate"] == rate]
            percent = 100 * sum(row["success"] for row in cases) / len(cases)
            statistics.append(f"<li>Failure probability {rate:.0%}: {percent:.2f}% success across {len(cases)} payments.</li>")
        glossary = "".join(f"<li><code>{html.escape(key)}</code>: {value}</li>"
                           for key, value in FEATURE_LABELS.items())
        report = (f'<!doctype html><html lang="en" dir="ltr"><meta charset="utf-8">'
            f'<title>Repetition {rep} report</title><style>body{{font:17px system-ui;line-height:1.8;max-width:1000px;'
            'margin:32px auto;padding:20px}}img{max-width:100%}code{direction:ltr;display:inline-block}</style>'
            f'<h1>Human-readable report: repetition {rep}</h1><h2>Routing process</h2>'
            '<p>The agent selects eta. Local candidate paths are evaluated with routing costs and Bucket. '
            'Payment attempts and reward calculations are recorded separately. '
            'Training uses the declared training failure setting; evaluation covers all six rates.</p>'
            '<h2>Training and held-out evaluation</h2><p>The pool contains 7,000 fixed training and '
            '3,000 held-out transactions. Intermediate diagnostics are descriptive; '
            'they do not select checkpoints or stop training.</p>'
            '<h2>Interpreting XAI</h2><p>Brighter heatmap cells indicate larger sensitivity of '
            'the deterministic policy to that feature group. They do not prove a causal explanation '
            'of success or failure. Hidden directional balances and simulator failure probabilities '
            'are excluded from policy inputs.</p>'
            f'<ul>{"".join(statistics)}</ul><h2>Feature glossary</h2><ul>{glossary}</ul>'
            f'<p><a href="rep_{rep:02d}_training_readable.csv">Training decision explanations</a> | '
            f'<a href="rep_{rep:02d}_evaluation_readable.csv">Held-out payment explanations</a></p>')
        for path in sorted((output_dir / "figures").glob(f"rep_{rep:02d}_*.png")):
            report += f'<figure><img src="../figures/{path.name}"><figcaption>{path.stem}</figcaption></figure>'
        report += '<p>Overfitting warnings and observed plateau windows are diagnostic evidence, '
        report += 'not exact onset estimates or definitive conclusions.</p></html>'
        (report_dir / f"rep_{rep:02d}.html").write_text(report, encoding="utf-8")
    # Equal weight per completed repetition; a partial set is explicitly labelled.
    aggregate = []
    csv_rows = []
    for feature in features:
        line = []
        for rate in (.01, .02, .03, .04, .05, .06):
            per_rep = [total / count for rep in repetitions
                       for total, count in [means[(rep, "final", "test", rate, feature)]] if count]
            value = sum(per_rep) / len(per_rep) if per_rep else float("nan")
            line.append(value)
            csv_rows.append([feature, rate, len(per_rep), value])
        aggregate.append(line)
    heatmap(aggregate, ["1%", "2%", "3%", "4%", "5%", "6%"],
            f"Final policy sensitivity / mean across {len(repetitions)} available repetitions",
            "all_repetitions_final_feature_heatmap")
    with (report_dir / "feature_heatmap_values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "failure_rate", "repetitions", "mean_absolute_eta_delta"])
        writer.writerows(csv_rows)
