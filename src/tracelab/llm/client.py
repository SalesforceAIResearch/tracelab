"""LLM access with hard budget enforcement ($5k sprint cap — decision 2026-08-17).

Provider: AnthropicVertex (direct SDK; ~zero prompt overhead vs the CLI's ~20K-token
system prompt). Every call records dollars into the SpendLedger BEFORE returning;
crossing the cap raises BudgetExceeded and no further calls are possible.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from tracelab.state.runstate import price_for

DEFAULT_MODEL = "claude-sonnet-5"
SPEND_PATH = Path(__file__).resolve().parents[3] / "bench" / "spend.json"
BUDGET_CAP_USD = 5000.0



def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Set {name} to your Google Cloud project id (Vertex AI Anthropic access required).")
    return val

class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    _lock = threading.RLock()

    def __init__(self, path: Path = SPEND_PATH, cap_usd: float = BUDGET_CAP_USD):
        self.path = path
        self.cap = cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"cap_usd": cap_usd, "entries": []}))

    def _data(self) -> dict:
        # reads share the lock: a concurrent record() mid-write must never surface a
        # torn file (observed live: JSONDecodeError during a parallel grid)
        with self._lock:
            return json.loads(self.path.read_text())

    def total(self) -> float:
        return sum(e["cost_usd"] for e in self._data()["entries"])

    def record(self, purpose: str, model: str, cost_usd: float,
               in_tok: int, out_tok: int) -> None:
      with self._lock:
        # cross-PROCESS safety (parallel bench runners share spend.json): advisory
        # flock around read-modify-write, atomic replace so readers never see a torn file
        import fcntl, os
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                d = self._data()
                d["entries"].append({"ts": time.time(), "purpose": purpose,
                                     "model": model, "cost_usd": round(cost_usd, 6),
                                     "in_tok": in_tok, "out_tok": out_tok})
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=1))
                os.replace(tmp, self.path)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def check(self) -> None:
        t = self.total()
        if t >= self.cap:
            raise BudgetExceeded(f"spend ${t:.2f} >= cap ${self.cap:.2f}")


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    stop_reason: str = ""


class VertexClient:
    def __init__(self, model: str = DEFAULT_MODEL, ledger: SpendLedger | None = None,
                 purpose: str = "adhoc"):
        from anthropic import AnthropicVertex
        self.model = model
        self.purpose = purpose
        self.ledger = ledger or SpendLedger()
        self._client = AnthropicVertex(
            project_id=_require_env("ANTHROPIC_VERTEX_PROJECT_ID"),
            region=os.environ.get("CLOUD_ML_REGION", "global"),
        )

    def complete_turns(self, turns: list[dict], *, system: str | None = None,
                       max_tokens: int = 1400, retries: int = 3) -> LLMResult:
        """Append-only conversational caching: each turn is its own content block;
        the cache breakpoint rides the LAST block. Because a step adds only ~2 blocks,
        the previous request's cached prefix sits within the lookback window of the
        new breakpoint - this is the pattern that actually hits (a single rebuilt
        prefix block does NOT: 0 reads observed live, v13a)."""
        self.ledger.check()
        msgs = []
        for i, t in enumerate(turns):
            blk = {"type": "text", "text": t["content"]}
            if i == len(turns) - 1:
                blk["cache_control"] = {"type": "ephemeral"}
            msgs.append({"role": t["role"], "content": [blk]})
        sys_arg = ([{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}] if system else None)
        last_err = None
        for attempt in range(retries):
            try:
                kwargs = dict(model=self.model, max_tokens=max_tokens, messages=msgs,
                              thinking={"type": "disabled"})
                if sys_arg:
                    kwargs["system"] = sys_arg
                m = self._client.messages.create(**kwargs)
                text = "".join(b.text for b in m.content if b.type == "text")
                p = price_for(self.model)
                cr = getattr(m.usage, "cache_read_input_tokens", 0) or 0
                cc = getattr(m.usage, "cache_creation_input_tokens", 0) or 0
                cost = (m.usage.input_tokens * p["input"]
                        + m.usage.output_tokens * p["output"]
                        + cr * p["cache_read"] + cc * p["cache_write"]) / 1e6
                self.ledger.record(self.purpose, self.model, cost,
                                   m.usage.input_tokens, m.usage.output_tokens)
                return LLMResult(text=text, input_tokens=m.usage.input_tokens,
                                 output_tokens=m.usage.output_tokens,
                                 cost_usd=cost, model=self.model,
                                 cache_read_tokens=cr, cache_creation_tokens=cc,
                                 stop_reason=str(m.stop_reason or ""))
            except Exception as e:  # noqa: BLE001 — retry then surface
                last_err = e
                import time as _t
                _t.sleep(2 * (attempt + 1))
        raise last_err

    def complete(self, prompt: str, *, system: str | None = None,
                 prefix: str | None = None,
                 max_tokens: int = 4000, retries: int = 3,
                 disable_thinking: bool = True) -> LLMResult:
        """disable_thinking: this endpoint enables thinking by default; for structured
        extraction calls the thinking block starves max_tokens and truncates the JSON
        (observed live: stop=max_tokens with empty text). Falls back gracefully if the
        param is rejected."""
        self.ledger.check()
        last_err = None
        thinking_arg = {"type": "disabled"} if disable_thinking else None
        for attempt in range(retries):
            try:
                if prefix:
                    # stable prefix carries a cache breakpoint: append-only prefixes
                    # (full/mask) hit cache every step; refreshed views (trace) pay a
                    # cache WRITE per refresh — exactly the economics under test
                    content = [
                        {"type": "text", "text": prefix,
                         "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": prompt},
                    ]
                else:
                    content = prompt
                kwargs = dict(model=self.model, max_tokens=max_tokens,
                              messages=[{"role": "user", "content": content}])
                if thinking_arg is not None:
                    kwargs["thinking"] = thinking_arg
                if system:
                    kwargs["system"] = ([{"type": "text", "text": system,
                                          "cache_control": {"type": "ephemeral"}}]
                                        if prefix else system)
                m = self._client.messages.create(**kwargs)
                text = "".join(b.text for b in m.content if b.type == "text")
                p = price_for(self.model)
                cr = getattr(m.usage, "cache_read_input_tokens", 0) or 0
                cc = getattr(m.usage, "cache_creation_input_tokens", 0) or 0
                cost = (m.usage.input_tokens * p["input"]
                        + m.usage.output_tokens * p["output"]
                        + cr * p["cache_read"] + cc * p["cache_write"]) / 1e6
                self.ledger.record(self.purpose, self.model, cost,
                                   m.usage.input_tokens, m.usage.output_tokens)
                return LLMResult(text=text, input_tokens=m.usage.input_tokens,
                                 output_tokens=m.usage.output_tokens,
                                 cost_usd=cost, model=self.model,
                                 cache_read_tokens=cr, cache_creation_tokens=cc,
                                 stop_reason=str(m.stop_reason or ""))
            except Exception as e:  # noqa: BLE001 — retry then surface
                last_err = e
                if thinking_arg is not None and "thinking" in str(e).lower():
                    thinking_arg = None  # endpoint rejects the param; retry without
                    continue
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} tries: {last_err}")
