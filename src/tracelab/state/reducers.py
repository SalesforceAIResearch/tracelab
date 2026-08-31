"""Deterministic reducers: Event stream -> RunState. Pure fold, no LLM calls.

Contract (property-tested):
- fold(events) is deterministic and equals incremental folding in any chunking;
- pending_tools never negative; every closed call has result_seq >= call_seq;
- counters are exact sums over the stream.
"""

from __future__ import annotations

import json
import re

from tracelab.detect.base import Detector
from tracelab.ledger.envelope import Event, EventKind, Source
from tracelab.state.runstate import RunState, ToolCallInfo, price_for

MAX_RECENT_FAILURES = 8
MAX_FACTS = 120
FACT_RE = re.compile(r"^\s*([A-Za-z_][\w.\-]{1,40})\s*=\s*(.{1,120}?)\s*$")
FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit", "read_file", "write_file"}


class StateFolder:
    """Mutable accumulator; `.state` is always a consistent RunState."""

    def __init__(self, source_path: str = "", detectors: list[Detector] | None = None):
        self.state = RunState(source_path=source_path)
        self._open_calls: dict[str, ToolCallInfo] = {}
        self._seen_usage_ids: set[str] = set()
        self.detectors = detectors or []

    # ------------------------------------------------------------------ fold

    def apply(self, ev: Event) -> None:
        s = self.state
        s.n_events += 1
        s.observed_seq = ev.seq
        s.state_materialized_at_seq = ev.seq
        if ev.session_id != "unknown":
            s.session_id = ev.session_id
        if ev.ts:
            if s.started_at is None:
                s.started_at = ev.ts
            s.last_event_ts = ev.ts

        if ev.usage:
            usage_key = ev.api_message_id or f"rec:{ev.record_id}"
            if usage_key in self._seen_usage_ids:
                ev = ev.model_copy(update={"usage": None})  # counted already for this API message
            else:
                self._seen_usage_ids.add(usage_key)
        if ev.usage:
            t = s.tokens
            t.input += ev.usage.input_tokens
            t.output += ev.usage.output_tokens
            t.cache_read += ev.usage.cache_read_tokens
            t.cache_creation += ev.usage.cache_creation_tokens
            p = price_for(ev.usage.model)
            s.est_cost_usd += (
                ev.usage.input_tokens * p["input"]
                + ev.usage.output_tokens * p["output"]
                + ev.usage.cache_read_tokens * p["cache_read"]
                + ev.usage.cache_creation_tokens * p["cache_write"]
            ) / 1_000_000
            if ev.usage.model:
                s.models_seen[ev.usage.model] = s.models_seen.get(ev.usage.model, 0) + 1

        if ev.agent_id is not None:
            s.sidechain_events += 1
            return

        handler = {
            EventKind.MESSAGE: self._on_message,
            EventKind.TOOL_CALL: self._on_tool_call,
            EventKind.TOOL_RESULT: self._on_tool_result,
            EventKind.ERROR: self._on_error,
        }.get(ev.kind)
        if handler:
            handler(ev)

        for det in self.detectors:
            det.observe(ev, s)

    def fold(self, events) -> RunState:
        for ev in events:
            self.apply(ev)
        return self.state

    # ------------------------------------------------------------- handlers

    def _on_message(self, ev: Event) -> None:
        s = self.state
        text = (ev.payload.text or "").strip()
        if ev.source == Source.USER:
            from tracelab.derived.episodes import is_substantive_user_text
            if ev.raw.block_index > 0 or not is_substantive_user_text(text):
                return  # system-reminder / interrupt-sentinel / non-substantive
            s.n_turns += 1
            s.latest_user_directive = text[:500]
            if s.goal_text is None:
                s.goal_text = text[:1000]
        elif ev.source == Source.AGENT and text:
            s.last_agent_text = text[:500]
            s.frontier = f"said: {text[:160]}"

    def _on_tool_call(self, ev: Event) -> None:
        s = self.state
        name = ev.payload.tool_name or "unknown-tool"
        s.n_tool_calls += 1
        s.per_tool[name] = s.per_tool.get(name, 0) + 1
        info = ToolCallInfo(
            correlation_id=ev.correlation_id or f"seq:{ev.seq}",
            tool_name=name,
            input_preview=(ev.payload.text or "")[:200],
            call_seq=ev.seq,
            call_ts=ev.ts,
        )
        if ev.correlation_id:
            self._open_calls[ev.correlation_id] = info
        s.pending_tools = [c for c in self._open_calls.values() if c.open]
        s.frontier = f"calling {name}({_arg_hint(ev)})"
        self._touch_files(ev, name)

    def _on_tool_result(self, ev: Event) -> None:
        s = self.state
        info = self._open_calls.get(ev.correlation_id or "")
        self._extract_facts(ev, info)
        if info and info.open:
            info.result_seq = ev.seq
            if info.call_ts and ev.ts:
                info.duration_s = max(0.0, (ev.ts - info.call_ts).total_seconds())
        if ev.payload.is_error:
            s.n_tool_errors += 1
            name = info.tool_name if info else "?"
            s.recent_failures.append(
                f"{name}: {(ev.payload.text or '')[:160]}")
            del s.recent_failures[:-MAX_RECENT_FAILURES]
            if info:
                info.is_error = True
        s.pending_tools = [c for c in self._open_calls.values() if c.open]

    def _on_error(self, ev: Event) -> None:
        s = self.state
        s.recent_failures.append(f"api: {(ev.payload.text or '')[:160]}")
        del s.recent_failures[:-MAX_RECENT_FAILURES]

    def _extract_facts(self, ev: Event, info=None) -> None:
        """Fact identity is OCCURRENCE identity, source-scoped (CONTINUE v4-v6 lessons):
        (1) newest-wins collapses accumulators; (2) versioned keys still merge
        duplicate VALUES (birthday collisions in 30 draws of 1-99); so facts are keyed
        by their source (the correlated call's path arg) when known."""
        text = ev.payload.text or ""
        if ev.payload.is_error or not text:
            return
        s = self.state
        source = None
        if info is not None and info.input_preview:
            m = re.search(r'"(?:file_path|notebook_path|path)"\s*:\s*"([^"]+)"',
                          info.input_preview)
            if m:
                source = m.group(1)
        for line in text.splitlines()[:200]:
            m = FACT_RE.match(line)
            if m:
                key, val = m.group(1), m.group(2)
                if key in ("retries",):
                    continue
                self.note_fact(source, key, val)

    def note_fact(self, source: str | None, key: str, val: str) -> None:
        """Single entry path for facts — regex-extracted AND semantically extracted
        (LLM curator) facts share the same occurrence identity, eviction, and
        lossless-aggregate machinery."""
        s = self.state
        if source:
            key = f"{source}:{key}"
        base_k = key.rsplit(":", 1)[-1].split("#", 1)[0]
        agg = s.fact_aggregates.get(base_k)
        if agg and key in agg.get("keys", []):
            return    # already folded into the aggregate: re-reads must not
                      # double count (v11@120 lesson)
        if key in s.facts and s.facts[key] != val:
            n = 2
            while f"{key}#{n}" in s.facts and s.facts[f"{key}#{n}"] != val:
                n += 1
            key = f"{key}#{n}"
        s.facts[key] = val
        if len(s.facts) > MAX_FACTS:            # bounded: evict oldest, but
            old_key = next(iter(s.facts))       # FOLD numerics into aggregates
            old_val = s.facts.pop(old_key)      # (eviction must be lossless
            try:                                #  for accumulators — v10@120)
                num = int(str(old_val).strip())
                base = old_key.rsplit(":", 1)[-1].split("#", 1)[0]
                agg = s.fact_aggregates.setdefault(
                    base, {"n": 0, "sum": 0, "keys": []})
                agg["n"] += 1
                agg["sum"] += num
                agg.setdefault("keys", []).append(old_key)
            except ValueError:
                pass

    def _touch_files(self, ev: Event, name: str) -> None:
        if name not in FILE_TOOLS or not ev.payload.text:
            return
        fp = None
        if not ev.payload.truncated:
            try:
                inp = json.loads(ev.payload.text)
                if isinstance(inp, dict):
                    fp = inp.get("file_path") or inp.get("notebook_path")
            except (json.JSONDecodeError, TypeError):
                fp = None
        if fp is None:
            # truncated (or odd) input: file_path is a short top-level key — regex the prefix
            m = re.search(r'"(?:file_path|notebook_path|path)"\s*:\s*"([^"]+)"', ev.payload.text)
            fp = m.group(1) if m else None
        if fp:
            self.state.files_touched[fp] = self.state.files_touched.get(fp, 0) + 1


def _arg_hint(ev: Event) -> str:
    try:
        inp = json.loads(ev.payload.text or "{}")
        if isinstance(inp, dict) and inp:
            k, v = next(iter(inp.items()))
            return f"{k}={str(v)[:60]}"
    except (json.JSONDecodeError, StopIteration, TypeError):
        pass
    return ""


def fold(events, source_path: str = "", detectors: list[Detector] | None = None) -> RunState:
    return StateFolder(source_path, detectors).fold(events)
