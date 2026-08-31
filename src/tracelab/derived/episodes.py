"""Deterministic episode folding + the first hindsight re-parse rule.

An EPISODE = one main-thread user turn and everything the agent did until the next
user turn. Episodes are DerivedNodes built ANCHORED-ITERATIVELY: as events stream in,
the open episode's node is evolved (merge-forward, versioned), never rebuilt from
scratch. Fully LLM-free; an optional LLM polish layer can later evolve nodes with
producer="llm:<model>".

Hindsight re-parsing v0 (producer="hindsight:interrupted"):
when a NEW user turn arrives while tool calls from the open episode are still
unresolved, the just-closed episode is re-versioned with status="interrupted" —
a later event retroactively changes the interpretation of an earlier region. The
superseded version keeps the pre-hindsight reading (provenance preserved), and
staleness propagates to any node that depended on it.
"""

from __future__ import annotations

from tracelab.derived.nodes import DerivedNode, NodeKind, NodeStore
from tracelab.ledger.envelope import Event, EventKind, Source

# user records that are actions, not asks — never open an episode / count a turn
NON_TURN_SENTINELS = ("[Request interrupted by user",)


def is_substantive_user_text(text: str) -> bool:
    t = text.strip()
    return bool(t) and not t.startswith("<") and not any(
        t.startswith(sn) for sn in NON_TURN_SENTINELS)


class EpisodeBuilder:
    """Streaming consumer: feed events (main thread), maintains episode nodes."""

    def __init__(self, store: NodeStore | None = None):
        self.store = store or NodeStore()
        self._open_id: str | None = None
        self._digest: dict | None = None
        self._open_calls: dict[str, str] = {}   # correlation_id -> tool name
        self._dangling: dict[str, str] = {}     # correlation_id -> node_id of closed episode
        self._fed = 0                           # main-thread events fed to open episode

    # ---------------------------------------------------------------- feed

    def feed(self, ev: Event) -> None:
        if ev.agent_id is not None:
            return  # main-thread episodes only (v0)

        if ev.kind == EventKind.MESSAGE and ev.source == Source.USER:
            text = (ev.payload.text or "").strip()
            if ev.raw.block_index == 0 and is_substantive_user_text(text):
                self._close_open(ev.seq, next_turn_started=True)
                self._start(ev, text)
                return

        # late result for a PRIOR episode's dangling call: attribute it there,
        # never to the open episode (hindsight:late-result)
        if (ev.kind == EventKind.TOOL_RESULT and ev.correlation_id
                and ev.correlation_id in self._dangling):
            self._resolve_dangling(ev)
            return

        if self._open_id is None:
            return
        d = self._digest
        self._fed += 1
        d["hi"] = ev.seq
        if ev.ts:
            d.setdefault("t0", ev.ts)
            d["t1"] = ev.ts
        if ev.kind == EventKind.TOOL_CALL:
            name = ev.payload.tool_name or "?"
            d["tools"][name] = d["tools"].get(name, 0) + 1
            d["n_calls"] += 1
            if ev.correlation_id:
                self._open_calls[ev.correlation_id] = name
        elif ev.kind == EventKind.TOOL_RESULT:
            if ev.correlation_id:
                self._open_calls.pop(ev.correlation_id, None)
            if ev.payload.is_error:
                d["errors"] += 1
                d["pins"].append(ev.seq)
        elif ev.kind == EventKind.MESSAGE and ev.source == Source.AGENT:
            t = (ev.payload.text or "").strip()
            if t:
                d["last_text"] = t[:240]
        elif ev.kind == EventKind.ERROR:
            d["errors"] += 1
            d["pins"].append(ev.seq)

        # anchored-iterative: evolve every K MAIN-THREAD events fed (sidechain seq
        # inflation must not change version counts)
        if self._fed - d.get("last_flush_fed", 0) >= 20:
            d["last_flush_fed"] = self._fed
            self._flush(ev.seq)

    def finalize(self, at_seq: int) -> None:
        self._close_open(at_seq, next_turn_started=False)

    # ------------------------------------------------------------- internals

    def _start(self, ev: Event, text: str) -> None:
        self._open_id = self.store.new_id("ep")
        self._digest = {"lo": ev.seq, "hi": ev.seq, "ask": text[:200], "tools": {},
                        "n_calls": 0, "errors": 0, "pins": [], "last_text": None,
                        "t0": ev.ts, "t1": ev.ts, "last_flush": ev.seq}
        self._open_calls.clear()
        self._fed = 0
        self.store.add(DerivedNode(
            node_id=self._open_id, version=1, kind=NodeKind.EPISODE,
            covered_lo=ev.seq, covered_hi=ev.seq, created_at_seq=ev.seq,
            producer="deterministic", title=self._title("in-progress"),
            content=self._content("in-progress"), structured=self._structured("in-progress"),
        ))

    def _title(self, status: str) -> str:
        d = self._digest
        return f"[{status}] {d['ask'][:80]}"

    def _structured(self, status: str) -> dict:
        d = self._digest
        dur = (d["t1"] - d["t0"]).total_seconds() if d.get("t0") and d.get("t1") else None
        return {"status": status, "ask": d["ask"], "tools": dict(d["tools"]),
                "n_calls": d["n_calls"], "errors": d["errors"],
                "duration_s": dur, "conclusion": d["last_text"]}

    def _content(self, status: str) -> str:
        d = self._digest
        tools = ", ".join(f"{k}×{v}" for k, v in sorted(d["tools"].items())) or "no tools"
        tail = f' → "{d["last_text"][:120]}"' if d["last_text"] else ""
        err = f", {d['errors']} error(s)" if d["errors"] else ""
        return f"{status}: {d['n_calls']} call(s) ({tools}){err}{tail}"

    def _resolve_dangling(self, ev: Event) -> None:
        node_id = self._dangling.pop(ev.correlation_id)
        head = self.store.head(node_id)
        if head is None:
            return
        structured = dict(head.structured)
        resolved = structured.setdefault("dangling_resolved", [])
        resolved.append(ev.correlation_id)
        if ev.payload.is_error:
            structured["errors"] = structured.get("errors", 0) + 1
        still = [c for c in self._dangling.values() if c == node_id]
        if not still:
            structured["status"] = "errors" if structured.get("errors") else "ok"
        note = " [hindsight: dangling tool returned" + (" with error]" if ev.payload.is_error else "]")
        self.store.evolve(node_id, ev.seq, "hindsight:late-result",
                          structured=structured,
                          content=head.content + note,
                          evidence_pins=(head.evidence_pins + [ev.seq])[-12:])

    def _flush(self, at_seq: int, status: str = "in-progress") -> None:
        d = self._digest
        d["last_flush"] = d["hi"]
        self.store.evolve(self._open_id, at_seq, "deterministic",
                          covered_hi=d["hi"], title=self._title(status),
                          content=self._content(status),
                          structured=self._structured(status),
                          evidence_pins=d["pins"][-12:])

    def _close_open(self, at_seq: int, *, next_turn_started: bool) -> None:
        if self._open_id is None:
            return
        d = self._digest
        status = "errors" if d["errors"] else "ok"
        self._flush(at_seq, status=status)
        if not next_turn_started and self._open_calls:
            # finalize with work in flight: not green — mark unresolved, keep dangling map
            head = self.store.head(self._open_id)
            structured = dict(head.structured)
            structured["status"] = "unresolved"
            structured["dangling_tools"] = sorted(set(self._open_calls.values()))
            self.store.evolve(self._open_id, at_seq, "deterministic",
                              title=self._title("unresolved"),
                              content=self._content("unresolved"),
                              structured=structured)
            for cid in self._open_calls:
                self._dangling[cid] = self._open_id
        if next_turn_started and self._open_calls:
            # HINDSIGHT: the new turn reveals the previous episode was cut short
            head = self.store.head(self._open_id)
            structured = dict(head.structured)
            structured["status"] = "interrupted"
            structured["dangling_tools"] = sorted(set(self._open_calls.values()))
            self.store.evolve(
                self._open_id, at_seq, "hindsight:interrupted",
                title=self._title("interrupted"),
                content=self._content("interrupted")
                + f" [hindsight: {len(self._open_calls)} tool(s) never returned]",
                structured=structured)
            for cid in self._open_calls:
                self._dangling[cid] = self._open_id
            # earlier reading is superseded; only its DEPENDENTS become suspect
            self.store.propagate_from(head.ref, at_seq)
        self._open_id = None
        self._digest = None
        self._open_calls.clear()


def build_episodes(events, store: NodeStore | None = None) -> NodeStore:
    b = EpisodeBuilder(store)
    last_seq = -1
    for ev in events:
        b.feed(ev)
        last_seq = ev.seq
    b.finalize(last_seq + 1)
    return b.store
