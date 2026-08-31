"""CONTINUE benchmark v0: does curated context beat full/masked context on
long-horizon workbench tasks — in success, steps, and honest dollars?

Grid: tasks × policies (full | mask | trace) × seeds. Metrics per policy:
success (mean subcheck score), completion (done called), steps, repeated actions,
parse failures, input/output tokens, cost. Every run's trace is recorded as JSONL —
the observer can render any benchmark run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tracelab.workbench.env import TASK_MAKERS
from tracelab.workbench.worker import run_task

POLICIES = ("full", "mask", "trace")


def run(client, *, kinds=("scatter", "fix"), seeds=(1, 2), policies=POLICIES,
        trace_dir: Path, max_steps: int = 45, out: Path | None = None) -> dict:
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows_path = trace_dir / "rows.jsonl"          # crash-safe incremental persistence
    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    for kind in kinds:
        for seed in seeds:
            task = TASK_MAKERS[kind](seed)
            for pol in policies:
                r = run_task(task, pol, client, trace_dir=trace_dir, seed=seed,
                             max_steps=max_steps)
                rows.append(r.__dict__)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(r.__dict__) + "\n")
                a = agg[pol]
                a["success"] += r.success
                a["done"] += 1.0 if r.done else 0.0
                a["steps"] += r.steps
                a["repeated"] += r.repeated_actions
                a["retries"] += r.retries
                a["errors_injected"] += r.errors_injected
                a["tool_steps"] += r.tool_steps
                a["view_refreshes"] += r.view_refreshes
                a["parse_failures"] += r.parse_failures
                a["input_tokens"] += r.input_tokens
                a["output_tokens"] += r.output_tokens
                a["cost_usd"] += r.cost_usd
                counts[pol] += 1

    result = {"n_runs": len(rows), "per_policy": {}, "runs": rows}
    for pol in policies:
        n = max(1, counts[pol])
        a = agg[pol]
        result["per_policy"][pol] = {
            "success": round(a["success"] / n, 4),
            "done_rate": round(a["done"] / n, 4),
            "mean_steps": round(a["steps"] / n, 2),
            "mean_repeated": round(a["repeated"] / n, 2),
            "mean_retries": round(a["retries"] / n, 2),
            "mean_errors_injected": round(a["errors_injected"] / n, 2),
            "mean_tool_steps": round(a["tool_steps"] / n, 2),
            "mean_parse_failures": round(a["parse_failures"] / n, 2),
            "input_tokens": int(a["input_tokens"]),
            "output_tokens": int(a["output_tokens"]),
            "cost_usd": round(a["cost_usd"], 4),
        }
    if out:
        board = json.loads(out.read_text()) if out.exists() else {}
        board["CONTINUE"] = {k: v for k, v in result.items() if k != "runs"}
        out.write_text(json.dumps(board, indent=2))
    return result
