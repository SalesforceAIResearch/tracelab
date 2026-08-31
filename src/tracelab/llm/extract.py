"""Semantic fact extraction — the curator's inference budget dial.

The deterministic curator pins facts with a regex (key = value lines). Real
observations state facts in prose; the regex is blind there by construction. This
extractor spends LLM calls INSIDE the parser to pin those facts, feeding them through
the exact same note_fact() machinery (occurrence identity, eviction, aggregates).

Design constraints that shaped it:
- one batched call per refresh (not per observation) — refresh cadence bounds cost;
- memoized per event_id — refolds re-inject identical facts, never re-extract, so a
  nondeterministic model still yields a deterministic view across refreshes;
- extraction failures degrade to "no facts from that batch", never to a crash.
"""

from __future__ import annotations

import json
import re

from tracelab.ledger.envelope import EventKind

EXTRACT_SYSTEM = (
    "You extract explicit facts from tool observations of an agent run. For each "
    "numbered observation, list the concrete facts it states: quantities, file paths, "
    "identifiers, settings. Use short snake_case keys (e.g. delta, next_file, "
    "timeout_s). Facts that play the SAME ROLE across observations MUST share one "
    "key - never vary the key with the phrasing (a running total fed by many "
    "observations is ONE key, e.g. delta). If a KNOWN KEYS list is provided, reuse "
    "those keys for facts of the same role instead of inventing synonyms. Values "
    "must appear verbatim in the observation. Reply with JSON only:"
    ' [{"i": <observation number>, "facts": [{"key": "...", "value": "..."}]}]. '
    "Observations with no concrete facts get an empty facts list.")

_PATH_RE = re.compile(r'"(?:file_path|notebook_path|path)"\s*:\s*"([^"]+)"')


class SemanticExtractor:
    def __init__(self, client, *, max_obs_chars: int = 700, batch_size: int = 12,
                 fallback_client=None):
        self.client = client
        self.fallback_client = fallback_client  # availability cascade's last rung
        self.fallbacks = 0
        self.max_obs_chars = max_obs_chars
        self.batch_size = batch_size
        self.memo: dict[str, list[tuple[str | None, str, str]]] = {}
        self.known_keys: list[str] = []   # anchored schema: canon carries forward
        self.calls = 0
        self.refusals = 0
        self.rejected = 0     # facts that failed verbatim validation
        self.cost_usd = 0.0

    def harvest(self, events) -> list[tuple[str | None, str, str]]:
        """Extract facts from all main-thread tool results, memoized by event_id.
        Returns every memoized fact in event order (callers re-inject after refolds)."""
        sources = {}          # correlation_id -> path arg of the originating call
        pending = []          # (event, source) not yet extracted
        for ev in events:
            if ev.agent_id is not None:
                continue
            if ev.kind == EventKind.TOOL_CALL and ev.correlation_id:
                m = _PATH_RE.search(ev.payload.text or "")
                if m:
                    sources[ev.correlation_id] = m.group(1)
            elif ev.kind == EventKind.TOOL_RESULT:
                if ev.event_id in self.memo:
                    continue
                if ev.payload.is_error or not (ev.payload.text or "").strip():
                    self.memo[ev.event_id] = []
                    continue
                pending.append((ev, sources.get(ev.correlation_id or "")))
        for i in range(0, len(pending), self.batch_size):
            self._extract_batch(pending[i:i + self.batch_size])
        out = []
        for ev in events:
            out.extend(self.memo.get(ev.event_id, []))
        return out

    def _extract_batch(self, batch) -> None:
        if not batch:
            return
        obs = "\n\n".join(
            f"[{j}] {(ev.payload.text or '')[: self.max_obs_chars]}"
            for j, (ev, _) in enumerate(batch))
        known = (f"KNOWN KEYS (reuse for same-role facts): "
                 f"{', '.join(self.known_keys)}\n\n" if self.known_keys else "")
        res = self.client.complete(f"{known}OBSERVATIONS:\n{obs}\n\nJSON:",
                                   system=EXTRACT_SYSTEM, max_tokens=2000)
        self.calls += 1
        self.cost_usd += res.cost_usd
        if not res.text.strip():
            # Observed live (v15): a batch of individually-benign machine-noise
            # observations can trip the extractor model's REFUSAL classifier as a
            # batch (looks like obfuscated content) while every single observation
            # passes. Cost-optimal batching must degrade by bisection, not by
            # silently dropping the batch.
            self.refusals += 1
            if len(batch) > 1:
                mid = len(batch) // 2
                self._extract_batch(batch[:mid])
                self._extract_batch(batch[mid:])
                return
            if self.fallback_client is not None:
                # a single observation the primary still refuses: last rung of the
                # cascade is a DIFFERENT model (classifiers disagree; v16 near-misses
                # were exactly these dropped leaves)
                res = self.fallback_client.complete(
                    f"{known}OBSERVATIONS:\n{obs}\n\nJSON:",
                    system=EXTRACT_SYSTEM, max_tokens=2000)
                self.calls += 1
                self.fallbacks += 1
                self.cost_usd += res.cost_usd
                if res.text.strip():
                    self._memoize_parsed(res.text, batch)
                    return
            self.memo[batch[0][0].event_id] = []
            return
        self._memoize_parsed(res.text, batch)

    def _memoize_parsed(self, text: str, batch) -> None:
        parsed: dict[int, list] = {}
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                for row in json.loads(m.group(0), strict=False):
                    if isinstance(row, dict) and isinstance(row.get("facts"), list):
                        parsed[int(row.get("i", -1))] = row["facts"]
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = {}
        for j, (ev, source) in enumerate(batch):
            facts = []
            obs_text = (ev.payload.text or "")[: self.max_obs_chars].lower()
            for f in parsed.get(j, []):
                k, v = str(f.get("key", "")).strip(), str(f.get("value", "")).strip()
                if k and v and v.lower() in obs_text:
                    # trust but VERIFY: the prompt demands verbatim values; enforcing
                    # it mechanically kills phantom facts that batch-composition
                    # nondeterminism occasionally invents (live +41 over-count, v17)
                    facts.append((source, k, v))
                    if k not in self.known_keys:
                        self.known_keys.append(k)
                elif k and v:
                    self.rejected += 1
            self.memo[ev.event_id] = facts
