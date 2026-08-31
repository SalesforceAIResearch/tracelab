"""Worker agent loop + context policies — the curator experiment substrate.

The worker is a minimal JSON-action agent over the Workbench tools. Its context each
step is assembled by a POLICY — the experimental variable:

  full   — entire action/observation history, append-only
  mask   — actions kept verbatim; observations older than K steps elided to a stub
           (Complexity Trap baseline)
  trace  — OUR curator: tracelab watches the worker's own recorded trace, folds it,
           and the context = compiled trace-model view + last K raw steps

Cache economics: with use_cache=True, context is sent as an append-only multi-turn
message list with a cache_control breakpoint on the last block (full: whole history;
trace: view-since-refresh, so a refresh = deliberate cache flush). Costs come from
actual cache_read/cache_creation usage at real cache pricing. A rebuilt single-block
prefix does NOT hit (v13a: 0 reads) — structure, not intent, decides caching.
Framing note for reports: the trace arm is a DIFFERENT treatment, not a subset — it
retains a distilled global state incl. a failure digest that mask loses, and loses
verbatim old observations that full keeps.

Every run is RECORDED as Claude-Code-shaped JSONL so the same ledger/state/episode
pipeline (and the observer page) runs on benchmark executions — one substrate everywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tracelab.derived.episodes import EpisodeBuilder
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.state.reducers import StateFolder
from tracelab.views.text_view import compile_text
from tracelab.workbench.env import Workbench

VOLATILE_MARK = "\n<<<VOLATILE>>>\n"

SYSTEM = """You are a task agent operating tools over a small filesystem.
Available tools (call with EXACT JSON, one action per reply):
  {"tool": "list_files", "args": {}}
  {"tool": "read_file", "args": {"path": "..."}}
  {"tool": "write_file", "args": {"path": "...", "content": "..."}}
  {"tool": "search", "args": {"pattern": "..."}}
  {"tool": "done", "args": {}}
Rules: reply with ONLY the JSON action, no commentary. If a tool returns
TransientError, retry it. Call done only when the task instruction is satisfied."""


@dataclass
class StepRecord:
    step: int
    action: dict
    observation: str
    is_error: bool


@dataclass
class RunResult:
    task_id: str
    policy: str
    success: float
    subcheck_notes: list[str]
    steps: int = 0                   # total worker turns (tool + parse-failure)
    tool_steps: int = 0
    parse_failure_steps: int = 0
    done: bool = False
    repeated_actions: int = 0        # true repeats (retries excluded)
    retries: int = 0                 # protocol-compliant retries after TransientError
    errors_injected: int = 0
    view_refreshes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    curator_cost_usd: float = 0.0
    curator_calls: int = 0
    trace_path: str = ""
    parse_failures: int = 0


class Recorder:
    """Writes the run as Claude-Code-shaped JSONL (the universal substrate)."""

    def __init__(self, path: Path, task_id: str):
        self.path = path
        self.f = open(path, "w")
        self.n = 0
        self.last = None
        self.task_id = task_id

    def _emit(self, rec):
        rec["uuid"] = f"wb-{self.n:05d}"
        rec["parentUuid"] = self.last
        rec["sessionId"] = f"workbench-{self.task_id}"
        rec["timestamp"] = f"2026-08-17T12:{(self.n // 60) % 60:02d}:{self.n % 60:02d}.000Z"
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.last = rec["uuid"]
        self.n += 1

    def user(self, text):
        self._emit({"type": "user", "message": {"role": "user", "content": text}})

    def protocol_failure(self, i, raw_text, usage):
        msg = {"role": "assistant", "model": usage.pop("_model", "worker"),
               "id": f"msg_wb_{i}",
               "content": [{"type": "text", "text": f"[unparseable action] {raw_text[:300]}"}]}
        msg["usage"] = usage
        self._emit({"type": "assistant", "message": msg})
        self._emit({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"twb_{i}",
             "content": "ProtocolError: reply was not a JSON action", "is_error": True}]}})

    def action(self, i, action, usage=None, model="worker"):
        msg = {"role": "assistant", "model": model,
               "id": f"msg_wb_{i}",
               "content": [{"type": "tool_use", "id": f"twb_{i}",
                            "name": action.get("tool", "?"),
                            "input": action.get("args", {})}]}
        if usage:
            msg["usage"] = usage
        self._emit({"type": "assistant", "message": msg})

    def observation(self, i, text, is_error):
        self._emit({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"twb_{i}",
             "content": text, "is_error": is_error}]}})

    def close(self):
        self.f.close()


# ------------------------------------------------------------- policies

def ctx_full(instruction: str, history: list[StepRecord], **_) -> str:
    parts = [f"TASK: {instruction}"]
    for r in history[:-1]:
        parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
        parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                     f"{r.observation}")
    parts.append(VOLATILE_MARK.strip())
    for r in history[-1:]:
        parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
        parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                     f"{r.observation}")
    parts.append("Next action (JSON only):")
    return "\n".join(parts)


def ctx_mask(instruction: str, history: list[StepRecord], *, keep_last: int = 5, **_) -> str:
    parts = [f"TASK: {instruction}"]
    cut = max(0, len(history) - keep_last)
    for r in history[:cut]:
        parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
        parts.append(f"[step {r.step}] result: [elided {len(r.observation)} chars"
                     f"{', was ERROR' if r.is_error else ''}]")
    for r in history[cut:-1]:
        parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
        parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                     f"{r.observation}")
    parts.append(VOLATILE_MARK.strip())
    for r in history[-1:]:
        parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
        parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                     f"{r.observation}")
    parts.append("Next action (JSON only):")
    return "\n".join(parts)


class TraceViewPolicy:
    """The curator: rebuilds the compiled view from the recorded trace every
    `refresh_every` steps; context = view + last K raw steps."""

    def __init__(self, trace_path: Path, keep_last: int = 5, refresh_every: int = 5,
                 extractor=None):
        self.trace_path = trace_path
        self.keep_last = keep_last
        self.refresh_every = refresh_every
        self.extractor = extractor      # optional SemanticExtractor: the budget dial
        self._view = ""
        self._last_refresh = -1
        self.refreshes = 0

    def __call__(self, instruction: str, history: list[StepRecord], **_) -> str:
        n_recorded = sum(1 for r in history if r.action.get("tool") != "INVALID")
        if (n_recorded - self._last_refresh >= self.refresh_every
                and self.trace_path.exists()):
            events = parse_file(self.trace_path).events
            folder = StateFolder(str(self.trace_path), detectors=default_detectors())
            folder.fold(events)
            epb = EpisodeBuilder()
            for ev in events:
                epb.feed(ev)
            if events:
                epb.finalize(events[-1].seq + 1)   # open episode digest fully current
            if self.extractor is not None:
                # semantic facts flow through the SAME note_fact machinery as regex
                # facts (occurrence identity, eviction, aggregates); memoization makes
                # re-injection after each refold deterministic
                for source, k, v in self.extractor.harvest(events):
                    folder.note_fact(source, k, v)
            self._view = compile_text(folder.state, epb.store)
            self._last_refresh = n_recorded
            self.refreshes += 1
        parts = [f"TASK: {instruction}",
                 "== RUN MODEL (folded history; older steps are summarized here) ==",
                 self._view or "(no history yet)",
                 VOLATILE_MARK.strip(),
                 f"== LAST {self.keep_last} RAW STEPS =="]
        for r in history[-self.keep_last:]:
            parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
            parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                         f"{r.observation}")
        parts.append("Next action (JSON only):")
        return "\n".join(parts)


def turns_full(instruction: str, history: list[StepRecord]) -> list[dict]:
    """Append-only conversational form of ctx_full - byte-identical prior turns."""
    turns = [{"role": "user", "content":
              f"TASK: {instruction}\nReply with ONLY the JSON action."}]
    for r in history:
        turns.append({"role": "assistant", "content": json.dumps(r.action)})
        turns.append({"role": "user", "content":
                      f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                      f"{r.observation}"})
    return turns


class TraceViewTurnsPolicy(TraceViewPolicy):
    """Cached-mode curator: view rides the FIRST user turn (rewritten each refresh =
    honest cache flush); steps since the refresh are appended as byte-identical turns
    (cache hits between refreshes). The raw window resets at refresh instead of
    sliding - sliding breaks prefix identity, the one structural concession caching
    demands."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._anchor = 0

    def turns(self, instruction: str, history: list[StepRecord]) -> list[dict]:
        n_recorded = sum(1 for r in history if r.action.get("tool") != "INVALID")
        if (n_recorded - self._last_refresh >= self.refresh_every
                and self.trace_path.exists()):
            events = parse_file(self.trace_path).events
            folder = StateFolder(str(self.trace_path), detectors=default_detectors())
            folder.fold(events)
            epb = EpisodeBuilder()
            for ev in events:
                epb.feed(ev)
            if events:
                epb.finalize(events[-1].seq + 1)
            self._view = compile_text(folder.state, epb.store)
            self._last_refresh = n_recorded
            self.refreshes += 1
            self._anchor = len(history)
        turns = [{"role": "user", "content": "\n".join(
            [f"TASK: {instruction}",
             "== RUN MODEL (folded history; older steps are summarized here) ==",
             self._view or "(no history yet)",
             "Raw steps since the last fold follow. Reply with ONLY the JSON action."])}]
        for r in history[self._anchor:]:
            turns.append({"role": "assistant", "content": json.dumps(r.action)})
            turns.append({"role": "user", "content":
                          f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                          f"{r.observation}"})
        return turns


SUMMARY_SYSTEM = """You maintain a running summary of an agent's work session so the
agent can continue the task WITHOUT the full history. Rewrite the summary to fold in
the new steps. PRESERVE everything needed to finish the task: the goal, exact values
and numbers collected so far (all of them - losing one ruins the task), current
position in the work, pending items, and lessons from errors. Plain text, <= 400
words."""


class SummaryViewPolicy:
    """Literature-standard baseline: rolling LLM summarization (compaction-style).
    Same refresh cadence and raw-step window as TraceViewPolicy - the folding
    MECHANISM (prose summary vs typed state) is the only variable. The summarizer is
    steelmanned: explicitly instructed to preserve every collected value."""

    def __init__(self, summarizer, keep_last: int = 5, refresh_every: int = 5):
        self.summarizer = summarizer
        self.keep_last = keep_last
        self.refresh_every = refresh_every
        self._summary = ""
        self._anchor = 0            # first history index not yet folded in
        self._last_refresh = -1
        self.refreshes = 0
        self.cost_usd = 0.0
        self.calls = 0

    def __call__(self, instruction: str, history: list[StepRecord], **_) -> str:
        n_recorded = sum(1 for r in history if r.action.get("tool") != "INVALID")
        if n_recorded - self._last_refresh >= self.refresh_every and history:
            new_lines = []
            for r in history[self._anchor:]:
                new_lines.append(f"[step {r.step}] action: {json.dumps(r.action)}")
                new_lines.append(
                    f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                    f"{r.observation[:900]}")
            prompt = (f"TASK: {instruction}\n\nCURRENT SUMMARY:\n"
                      f"{self._summary or '(none yet)'}\n\nNEW STEPS:\n"
                      + "\n".join(new_lines) + "\n\nUpdated summary:")
            res = self.summarizer.complete(prompt, system=SUMMARY_SYSTEM,
                                           max_tokens=4000)
            self.calls += 1
            self.cost_usd += res.cost_usd
            if res.text.strip():
                self._summary = res.text.strip()
                self._anchor = len(history)
            self._last_refresh = n_recorded
            self.refreshes += 1
        parts = [f"TASK: {instruction}",
                 "== SUMMARY OF WORK SO FAR (older steps are folded here) ==",
                 self._summary or "(no history yet)",
                 f"== LAST {self.keep_last} RAW STEPS =="]
        for r in history[-self.keep_last:]:
            parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
            parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                         f"{r.observation}")
        parts.append("Next action (JSON only):")
        return "\n".join(parts)


class RetrievalPolicy:
    """RAG-over-own-trace baseline (reviewer-requested): context = task + last-K raw
    steps + top-M past steps retrieved by lexical relevance to the CURRENT step's
    observation. No external index - token-overlap scoring over the worker's own
    history, the honest minimal retrieval policy every arm could implement."""

    def __init__(self, keep_last: int = 5, top_m: int = 10):
        self.keep_last = keep_last
        self.top_m = top_m

    @staticmethod
    def _toks(text: str) -> set:
        return {w for w in text.lower().split() if len(w) > 2}

    def __call__(self, instruction: str, history: list[StepRecord], **_) -> str:
        recent = history[-self.keep_last:]
        older = history[:-self.keep_last]
        query = self._toks(" ".join(
            (json.dumps(r.action) + " " + r.observation[-300:]) for r in recent[-2:])
            or instruction)
        scored = []
        for r in older:
            doc = self._toks(json.dumps(r.action) + " " + r.observation[:400])
            inter = len(query & doc)
            if inter:
                scored.append((inter / (1 + len(doc)) ** 0.5, r))
        scored.sort(key=lambda x: -x[0])
        hits = sorted((r for _, r in scored[: self.top_m]), key=lambda r: r.step)
        parts = [f"TASK: {instruction}",
                 f"== {len(hits)} RETRIEVED PAST STEPS (relevance-ranked, then "
                 "time-ordered; the rest of history is omitted) =="]
        for r in hits:
            parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
            parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                         f"{r.observation[:400]}")
        parts.append(f"== LAST {self.keep_last} RAW STEPS ==")
        for r in recent:
            parts.append(f"[step {r.step}] action: {json.dumps(r.action)}")
            parts.append(f"[step {r.step}] result{' (ERROR)' if r.is_error else ''}: "
                         f"{r.observation}")
        parts.append("Next action (JSON only):")
        return "\n".join(parts)


# ------------------------------------------------------------- loop

def parse_action(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0), strict=False)
        return d if isinstance(d, dict) and "tool" in d else None
    except json.JSONDecodeError:
        return None


def run_task(task, policy_name: str, client, *, trace_dir: Path, seed: int = 0,
             max_steps: int = 45, keep_last: int = 5, refresh_every: int = 5,
             env_kwargs: dict | None = None, use_cache: bool = False,
             extractor=None, summarizer=None) -> RunResult:
    env = Workbench(task, seed=seed, **(env_kwargs or {}))
    trace_path = trace_dir / f"{task.task_id}-{policy_name}-{seed}.jsonl"
    rec = Recorder(trace_path, task.task_id)
    rec.user(task.instruction)

    policy = {"full": ctx_full, "mask": ctx_mask}.get(policy_name)
    if policy is None and policy_name == "retrieval":
        policy = RetrievalPolicy(keep_last=keep_last)
    elif policy is None and policy_name == "summary":
        policy = SummaryViewPolicy(summarizer, keep_last=keep_last,
                                   refresh_every=refresh_every)
    elif policy is None:
        policy = (TraceViewTurnsPolicy if use_cache else
                  TraceViewPolicy)(trace_path, keep_last=keep_last,
                                   refresh_every=refresh_every, extractor=extractor)

    history: list[StepRecord] = []
    seen_actions: dict[str, int] = {}
    repeated = retries = 0
    parse_failures = consecutive_failures = 0
    res = RunResult(task.task_id, policy_name, 0.0, [], trace_path=str(trace_path))

    for step in range(max_steps):
        if use_cache:
            turns = (policy.turns(task.instruction, history)
                     if hasattr(policy, "turns")
                     else turns_full(task.instruction, history))
            llm = client.complete_turns(turns, system=SYSTEM, max_tokens=1400)
        else:
            context = policy(task.instruction, history, keep_last=keep_last)
            llm = client.complete(context.replace(VOLATILE_MARK.strip(), ""),
                                  system=SYSTEM, max_tokens=1400)
        res.input_tokens += llm.input_tokens
        res.output_tokens += llm.output_tokens
        res.cache_read_tokens += getattr(llm, "cache_read_tokens", 0)
        res.cache_creation_tokens += getattr(llm, "cache_creation_tokens", 0)
        res.cost_usd += llm.cost_usd
        action = parse_action(llm.text)
        if action is None:
            parse_failures += 1
            consecutive_failures += 1
            rec.protocol_failure(step, llm.text,
                                 {"_model": llm.model,
                                  "input_tokens": llm.input_tokens,
                                  "output_tokens": llm.output_tokens,
                                  "cache_read_input_tokens": getattr(llm, "cache_read_tokens", 0),
                                  "cache_creation_input_tokens": getattr(llm, "cache_creation_tokens", 0)})
            history.append(StepRecord(step, {"tool": "INVALID"},
                                      "ProtocolError: reply was not a JSON action",
                                      True))
            if consecutive_failures >= 3:
                break
            continue
        consecutive_failures = 0
        sig = json.dumps(action, sort_keys=True)
        prev = history[-1] if history else None
        is_retry = (prev is not None and prev.is_error
                    and "TransientError" in prev.observation
                    and json.dumps(prev.action, sort_keys=True) == sig)
        seen_actions[sig] = seen_actions.get(sig, 0) + 1
        if is_retry:
            retries += 1
        elif seen_actions[sig] > 1 and action.get("tool") != "done":
            repeated += 1
        rec.action(step, action, model=llm.model,
                   usage={"input_tokens": llm.input_tokens,
                          "output_tokens": llm.output_tokens,
                          "cache_read_input_tokens": getattr(llm, "cache_read_tokens", 0),
                          "cache_creation_input_tokens": getattr(llm, "cache_creation_tokens", 0)})
        out = env.call(action.get("tool", "?"), action.get("args") or {})
        rec.observation(step, out.text, out.is_error)
        history.append(StepRecord(step, action, out.text, out.is_error))
        if env.done_called:
            break

    rec.close()
    score, notes = env.score()
    res.success = score
    res.subcheck_notes = notes[:6]
    res.steps = len(history)
    if extractor is not None:
        res.curator_cost_usd = extractor.cost_usd
        res.curator_calls = extractor.calls
    elif hasattr(policy, "cost_usd"):
        res.curator_cost_usd = policy.cost_usd
        res.curator_calls = policy.calls
    res.tool_steps = len(history) - parse_failures
    res.parse_failure_steps = parse_failures
    res.done = env.done_called
    res.repeated_actions = repeated
    res.retries = retries
    res.errors_injected = env.errors_injected
    res.view_refreshes = getattr(policy, "refreshes", 0)
    res.parse_failures = parse_failures
    return res
