"""Cheap streaming detectors: loop, error streak, tool flood, budget, stall.

All operate on the event stream + RunState, no LLM calls. Thresholds are
constructor args so the DETECT benchmark can sweep them.
"""

from __future__ import annotations

from collections import deque

from tracelab.detect.base import Detector
from tracelab.ledger.envelope import Event, EventKind, Source
from tracelab.state.runstate import RunState


class LoopDetector(Detector):
    """Same (tool, input-hash) signature repeating within a sliding window."""
    kind = "loop"

    def __init__(self, repeats: int = 4, window: int = 24, clear_after: int = 6):
        self.repeats = repeats
        self.clear_after = clear_after
        self.window: deque[tuple[str, str, int]] = deque(maxlen=window)
        self._raising_sig: tuple[str, str] | None = None
        self._diverged = 0

    def observe(self, ev: Event, state: RunState) -> None:
        if ev.kind != EventKind.TOOL_CALL:
            return
        sig = (ev.payload.tool_name or "?", ev.payload.sha256 or "")
        self.window.append((*sig, ev.seq))
        hits = [seq for name, h, seq in self.window if (name, h) == sig]
        if len(hits) >= self.repeats:
            self._raising_sig = sig
            self._diverged = 0
            self._raise(state, severity="warn" if len(hits) == self.repeats else "critical",
                        message=(f"{sig[0]} called {len(hits)}x with identical input "
                                 f"in last {self.window.maxlen} calls"),
                        seq=ev.seq, evidence=hits)
        elif self._raising_sig is not None and sig != self._raising_sig:
            # loop anomalies must clear once behavior genuinely diverges
            self._diverged += 1
            if self._diverged >= self.clear_after:
                self._clear(state, ev.seq)
                self._raising_sig = None


class ErrorStreakDetector(Detector):
    kind = "error_streak"

    def __init__(self, streak: int = 3):
        self.streak = streak
        self._run: list[int] = []

    def observe(self, ev: Event, state: RunState) -> None:
        if ev.kind == EventKind.TOOL_RESULT:
            if ev.payload.is_error:
                self._run.append(ev.seq)
                if len(self._run) >= self.streak:
                    self._raise(state, severity="critical",
                                message=f"{len(self._run)} consecutive tool errors",
                                seq=ev.seq, evidence=self._run)
            else:
                if len(self._run) >= self.streak:
                    self._clear(state, ev.seq)
                self._run = []


class ToolFloodDetector(Detector):
    """The '200th send_email looks like the 1st' counter — per-tool call caps."""
    kind = "tool_flood"

    def __init__(self, default_cap: int = 100, caps: dict[str, int] | None = None):
        self.default_cap = default_cap
        self.caps = caps or {}

    def observe(self, ev: Event, state: RunState) -> None:
        if ev.kind != EventKind.TOOL_CALL:
            return
        name = ev.payload.tool_name or "?"
        n = state.per_tool.get(name, 0)
        cap = self.caps.get(name, self.default_cap)
        if n >= cap:
            self._raise(state, severity="warn",
                        message=f"{name} called {n}x (cap {cap})",
                        seq=ev.seq, evidence=[ev.seq])


class BudgetDetector(Detector):
    kind = "budget"

    def __init__(self, warn_usd: float = 50.0, critical_usd: float = 200.0):
        self.warn_usd = warn_usd
        self.critical_usd = critical_usd

    def observe(self, ev: Event, state: RunState) -> None:
        if state.est_cost_usd >= self.critical_usd:
            self._raise(state, severity="critical",
                        message=f"est. cost ${state.est_cost_usd:,.0f} ≥ ${self.critical_usd:,.0f}",
                        seq=ev.seq, evidence=[ev.seq])
        elif state.est_cost_usd >= self.warn_usd:
            self._raise(state, severity="warn",
                        message=f"est. cost ${state.est_cost_usd:,.0f} ≥ ${self.warn_usd:,.0f}",
                        seq=ev.seq, evidence=[ev.seq])


class StallDetector(Detector):
    """Timestamp gap with work in flight (replay mode uses event ts deltas).

    Tracks open calls itself: reducers may close a call on the very event whose
    timestamp reveals the gap, so `state.pending_tools` is already updated by the
    time detectors run — the detector's own pre-event view is the correct one.
    """
    kind = "stall"

    def __init__(self, gap_s: float = 600.0):
        self.gap_s = gap_s
        self._last_ts = None
        self._open_ids: set[str] = set()

    def observe(self, ev: Event, state: RunState) -> None:
        if ev.ts is not None:
            if self._last_ts is not None:
                gap = (ev.ts - self._last_ts).total_seconds()
                if gap >= self.gap_s and self._open_ids:
                    self._raise(state, severity="warn",
                                message=f"{gap:.0f}s gap with {len(self._open_ids)} tool(s) in flight",
                                seq=ev.seq, evidence=[ev.seq])
            self._last_ts = ev.ts
        if ev.kind == EventKind.TOOL_CALL and ev.correlation_id:
            self._open_ids.add(ev.correlation_id)
        elif ev.kind == EventKind.TOOL_RESULT and ev.correlation_id:
            self._open_ids.discard(ev.correlation_id)
        elif ev.kind == EventKind.MESSAGE and ev.source == Source.USER:
            # a new user turn means nothing is actually in flight; dangling calls
            # (interrupts, API errors) must not poison the rest of the session
            self._open_ids.clear()
            self._clear(state, ev.seq)


def default_detectors() -> list[Detector]:
    return [LoopDetector(), ErrorStreakDetector(), ToolFloodDetector(),
            BudgetDetector(), StallDetector()]
