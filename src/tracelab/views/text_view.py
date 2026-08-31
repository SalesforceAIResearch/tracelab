"""Compact text/markdown compilation of the trace model — the VIEW condition in
COMPREHEND, and the seed of the stage-2 curator's worker-context serializer."""

from __future__ import annotations

from tracelab.derived.nodes import NodeKind, NodeStore
from tracelab.state.runstate import RunState


def compile_text(state: RunState, store: NodeStore, *, max_episodes: int = 30) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Run model · session {state.session_id[:8]}")
    a(f"GOAL: {state.goal_text or '(unknown)'}")
    a(f"LATEST USER REQUEST: {state.latest_user_directive or '(none)'}")
    a(f"NOW: {state.frontier or '(idle)'}")
    if state.pending_tools:
        a("IN FLIGHT: " + ", ".join(f"{c.tool_name}({c.correlation_id})"
                                     for c in state.pending_tools))
    if state.active_anomalies:
        for an in state.active_anomalies:
            a(f"ANOMALY[{an.severity}] {an.kind}: {an.message}")
    a(f"COUNTERS: turns={state.n_turns} tool_calls={state.n_tool_calls} "
      f"errors={state.n_tool_errors} est_cost=${state.est_cost_usd:.2f} "
      f"tokens(out={state.tokens.output}, cache_read={state.tokens.cache_read})")
    top = sorted(state.per_tool.items(), key=lambda kv: -kv[1])[:8]
    a("TOP TOOLS: " + (", ".join(f"{k}×{v}" for k, v in top) or "(none)"))
    if state.files_touched:
        a("FILES TOUCHED: " + ", ".join(sorted(state.files_touched)))
    if state.facts:
        n_evicted = sum(agg.get("n", 0) for agg in state.fact_aggregates.values())
        a(f"KEY FACTS ({len(state.facts)} in view of {len(state.facts) + n_evicted} "
          f"extracted — grouped by key; evicted values fold into [aggregate] lines):")
        # group by base key (suffix after source prefix): all 'delta' facts contiguous —
        # scattered same-kind facts caused end-stage aggregation slips (v7 residual)
        def base(k):
            return k.rsplit(":", 1)[-1].split("#", 1)[0]
        groups: dict[str, list] = {}
        # requirement-candidate 11 (coverage stamp): the aggregate must state what it
        # covers, or the reader cannot tell whether its latest read is already folded
        # in — the exact ambiguity behind all five curated 120-link misses (v26/v23)
        last_source: dict[str, str] = {}
        for k in state.facts:            # insertion order = fold order
            last_source[base(k)] = k
        for k, v in sorted(state.facts.items(), key=lambda kv: (base(kv[0]), kv[0])):
            a(f"  {k} = {v}")
            groups.setdefault(base(k), []).append(v)
        # requirement 5 (running aggregates): the trace model does arithmetic the
        # LLM slips on — deterministic per-key sums for numeric fact groups
        for g, vals in groups.items():
            nums = []
            for v in vals:
                try:
                    nums.append(int(str(v).strip()))
                except ValueError:
                    break
            else:
                evicted = state.fact_aggregates.get(g, {"n": 0, "sum": 0})
                total_n = len(nums) + evicted["n"]
                if total_n >= 3:
                    a(f"  [aggregate] {g}: {total_n} values total "
                      f"({evicted['n']} folded out of view), sum = {sum(nums) + evicted['sum']}"
                      f" — ALREADY INCLUDES every {g} above and every folded one,"
                      f" through {last_source.get(g, '?')}")
    if state.recent_failures:
        a("RECENT FAILURES:")
        for f in state.recent_failures[-5:]:
            a(f"  - {f}")
    a("")
    a("## Episodes (chronological)")
    for ep in store.heads(NodeKind.EPISODE)[-max_episodes:]:
        s = ep.structured
        a(f"- [{s.get('status','?')}] {s.get('ask','')[:120]}")
        a(f"    {ep.content}")
    return "\n".join(lines)
