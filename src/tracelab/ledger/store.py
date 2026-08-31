"""Ledger store + live session tailer.

The Ledger is an in-process, append-only sequence of canonical Events with:
- ingest from a source adapter, incrementally (resume by byte offset);
- deterministic serialization (JSONL of Event models) for snapshots;
- watermarks: `observed_seq` (latest event held) and `durable_offset` (byte offset in
  the source file up to which ingestion is complete — the resume point).

Replay determinism contract (property-tested): ingesting a file in any number of
chunks yields the same event sequence as ingesting it whole.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from tracelab.ledger.envelope import Event
from tracelab.ledger.adapters import claude_code


@dataclass
class Watermarks:
    observed_seq: int = -1        # seq of last event in memory
    durable_offset: int = 0       # byte offset consumed in source
    source_line: int = 1          # next line number to read
    last_ingest_ts: float = 0.0


class Ledger:
    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)
        self.events: list[Event] = []
        self.marks = Watermarks()
        self._by_id: dict[str, int] = {}
        self.malformed = 0
        self.trailing = 0
        self.unknown_types: dict[str, int] = {}

    # -- ingestion ---------------------------------------------------------

    def ingest_available(self) -> int:
        """Consume whatever complete lines exist beyond the durable offset."""
        res = claude_code.parse_file(
            self.source_path,
            start_offset=self.marks.durable_offset,
            start_seq=len(self.events),
            start_line=self.marks.source_line,
        )
        for ev in res.events:
            self._by_id[ev.event_id] = len(self.events)
            self.events.append(ev)
        self.marks.durable_offset = res.resume_offset
        self.marks.source_line += res.stats.lines
        self.marks.observed_seq = len(self.events) - 1
        self.marks.last_ingest_ts = time.time()
        self.trailing = res.stats.trailing_partial_bytes
        self.malformed += res.stats.malformed
        for k, v in res.stats.unknown_types.items():
            self.unknown_types[k] = self.unknown_types.get(k, 0) + v
        return len(res.events)

    def tail(self, *, poll_s: float = 1.0, stop: Callable[[], bool] | None = None,
             on_batch: Callable[[list[Event]], None] | None = None) -> Iterator[Event]:
        """Follow the source file live. Yields events as they land."""
        while True:
            before = len(self.events)
            self.ingest_available()
            new = self.events[before:]
            if new and on_batch:
                on_batch(new)
            yield from new
            if stop and stop():
                return
            time.sleep(poll_s)

    # -- lookup ------------------------------------------------------------

    def get(self, event_id: str) -> Event | None:
        i = self._by_id.get(event_id)
        return self.events[i] if i is not None else None

    def children_of(self, record_id: str) -> list[Event]:
        return [e for e in self.events if e.parent_event_id == record_id]

    # -- snapshot / replay ---------------------------------------------------

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"marks": {
                "observed_seq": self.marks.observed_seq,
                "durable_offset": self.marks.durable_offset,
                "source_line": self.marks.source_line,
            }, "source": str(self.source_path),
               "malformed": self.malformed,
               "unknown_types": self.unknown_types}) + "\n")
            for ev in self.events:
                f.write(ev.model_dump_json() + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        with open(path) as f:
            head = json.loads(f.readline())
            led = cls(head["source"])
            for line in f:
                if line.strip():
                    ev = Event.model_validate_json(line)
                    led._by_id[ev.event_id] = len(led.events)
                    led.events.append(ev)
        led.marks.observed_seq = head["marks"]["observed_seq"]
        led.marks.durable_offset = head["marks"]["durable_offset"]
        led.marks.source_line = head["marks"]["source_line"]
        led.malformed = head.get("malformed", 0)
        led.unknown_types = dict(head.get("unknown_types", {}))
        return led
