"""L2: the typed RunState — "what is true now" for a run.

Execution position is schema, not prose (TRACE): frontier / completed / pending /
blocked / do_not_repeat are explicit fields. Counters that no SDK provides (cumulative
per-tool, dollars with cache accounting) are computed here. The state is produced ONLY
by folding ledger events through deterministic reducers (`reducers.fold`); the
LLM-assisted semantic layer (goal refinement, subtask naming) arrives in M2 and writes
into clearly-marked `semantic_*` fields so deterministic truth and interpretation never
mix silently.
"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field

# $/MTok defaults; overridable. Keyed by substring match on model name.
PRICES = {
    "test-premium": dict(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
    "claude-opus": dict(input=15.0, output=75.0, cache_read=1.5, cache_write=18.75),
    "claude-sonnet": dict(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
    "claude-haiku": dict(input=0.8, output=4.0, cache_read=0.08, cache_write=1.0),
    "_default": dict(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
}


def price_for(model: str | None) -> dict:
    if model:
        for key, p in PRICES.items():
            if key != "_default" and key in model:
                return p
    return PRICES["_default"]


class ToolCallInfo(BaseModel):
    correlation_id: str
    tool_name: str | None
    input_preview: str | None = None
    call_seq: int
    call_ts: datetime | None = None
    result_seq: int | None = None
    duration_s: float | None = None
    is_error: bool = False

    @property
    def open(self) -> bool:
        return self.result_seq is None


class Anomaly(BaseModel):
    kind: str                 # loop | stall | error_streak | budget | tool_flood
    severity: str             # info | warn | critical
    message: str
    detected_at_seq: int
    evidence_seqs: list[int] = Field(default_factory=list)
    cleared_at_seq: int | None = None   # anomalies can resolve

    @property
    def active(self) -> bool:
        return self.cleared_at_seq is None


class Tokens(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0


class RunState(BaseModel):
    # identity
    session_id: str = "unknown"
    source_path: str = ""
    started_at: datetime | None = None
    last_event_ts: datetime | None = None

    # goal & position (deterministic layer)
    goal_text: str | None = None            # first substantive user message
    latest_user_directive: str | None = None
    frontier: str | None = None             # last meaningful action, one line
    last_agent_text: str | None = None

    # execution position (TRACE fields; deterministic approximations in M1,
    # semantically refined in M2)
    pending_tools: list[ToolCallInfo] = Field(default_factory=list)
    recent_failures: list[str] = Field(default_factory=list)   # last K error one-liners
    files_touched: dict[str, int] = Field(default_factory=dict)  # path -> edit count

    # extracted facts: key=value lines seen in tool results (deterministic pinning —
    # the M4 finding: a view without FACTS loses to full context; pins must carry
    # content, not seq numbers). Newest value wins; capped.
    facts: dict[str, str] = Field(default_factory=dict)
    # lossless numeric aggregation under bounded facts (CONTINUE v10 lesson):
    # evicting a numeric fact folds its value here instead of losing it
    fact_aggregates: dict[str, dict] = Field(default_factory=dict)  # base -> {n, sum}

    # counters (the ones no SDK provides)
    n_events: int = 0
    sidechain_events: int = 0
    n_turns: int = 0                        # user messages
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    per_tool: dict[str, int] = Field(default_factory=dict)
    tokens: Tokens = Field(default_factory=Tokens)
    est_cost_usd: float = 0.0
    models_seen: dict[str, int] = Field(default_factory=dict)

    # health
    anomalies: list[Anomaly] = Field(default_factory=list)

    # watermarks (honesty layer)
    observed_seq: int = -1
    state_materialized_at_seq: int = -1

    @property
    def active_anomalies(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.active]

    @property
    def wall_clock_s(self) -> float | None:
        if self.started_at and self.last_event_ts:
            return (self.last_event_ts - self.started_at).total_seconds()
        return None
