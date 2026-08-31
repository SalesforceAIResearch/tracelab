"""Synthetic trace generator with LABELED pathologies — ground truth for DETECT.

Generates Claude-Code-shaped JSONL so the same adapter/reducers/detectors run on
synthetic and real traces identically. Deterministic under a seed.

Pathology taxonomy (v0, drawn from MAST + meltdown lore):
- loop:          same tool+input repeated k times
- error_streak:  k consecutive failing tool results
- tool_flood:    one tool called far beyond its cap
- stall:         large timestamp gap with a call in flight
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOOLS = ["Read", "Grep", "Bash", "Edit", "WebSearch"]


@dataclass
class PathologyLabel:
    kind: str
    start_line: int   # 1-based, inclusive
    end_line: int


@dataclass
class SyntheticTrace:
    path: Path
    labels: list[PathologyLabel]
    n_lines: int


class _Writer:
    def __init__(self, path: Path, seed: int):
        self.f = open(path, "w")
        self.rng = random.Random(seed)
        self.line = 0
        self.uuid_n = 0
        self.last_uuid = None
        self.ts = datetime(2026, 8, 17, 9, 0, 0, tzinfo=timezone.utc)

    def _uuid(self):
        self.uuid_n += 1
        return f"syn-{self.uuid_n:05d}"

    def _emit(self, rec: dict) -> int:
        self.f.write(json.dumps(rec) + "\n")
        self.line += 1
        return self.line

    def _stamp(self, seconds: float = None):
        self.ts += timedelta(seconds=self.rng.uniform(1, 8) if seconds is None else seconds)
        return self.ts.isoformat().replace("+00:00", "Z")

    def base(self, rtype, **extra):
        u = self._uuid()
        rec = {"type": rtype, "uuid": u, "parentUuid": self.last_uuid,
               "sessionId": "synthetic", "timestamp": self._stamp(extra.pop("gap_s", None))}
        rec.update(extra)
        self.last_uuid = u
        return rec

    def user_msg(self, text):
        return self._emit(self.base("user", message={"role": "user", "content": text}))

    def agent_text(self, text):
        return self._emit(self.base("assistant", message={
            "role": "assistant", "model": "claude-sonnet-5", "id": f"msg_syn_{self.uuid_n}",
            "usage": {"input_tokens": self.rng.randint(50, 500),
                      "output_tokens": self.rng.randint(50, 800),
                      "cache_read_input_tokens": self.rng.randint(1000, 80000),
                      "cache_creation_input_tokens": self.rng.randint(0, 5000)},
            "content": [{"type": "text", "text": text}]}))

    def tool_call(self, name, inp, gap_s=None):
        tid = f"toolu_syn_{self.uuid_n}"
        self._emit(self.base("assistant", message={
            "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]},
            gap_s=gap_s))
        return tid, self.line

    def tool_result(self, tid, text, is_error=False, gap_s=None):
        return self._emit(self.base("user", message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": text,
             "is_error": is_error}]}, gap_s=gap_s))


def generate(path: str | Path, *, seed: int = 0,
             pathologies: list[str] | None = None,
             n_phases: int = 4) -> SyntheticTrace:
    path = Path(path)
    w = _Writer(path, seed)
    labels: list[PathologyLabel] = []
    todo = list(pathologies or [])
    w.rng.shuffle(todo)

    if len(todo) > n_phases:
        raise ValueError(f"{len(todo)} pathologies > {n_phases} phases")
    w.user_msg("Synthetic long-horizon task: refactor the widget pipeline end to end.")
    # non-colliding insertion points: every requested pathology IS injected
    phases_drawn = w.rng.sample(range(n_phases), k=len(todo)) if todo else []
    insert_at = dict(zip(phases_drawn, todo))

    for phase in range(n_phases):
        w.agent_text(f"Working on phase {phase}: inspecting inputs.")
        for _ in range(w.rng.randint(2, 5)):
            tool = w.rng.choice(TOOLS)
            tid, _ = w.tool_call(tool, {"arg": f"phase{phase}-{w.rng.randint(0, 999)}"})
            w.tool_result(tid, f"ok result {w.rng.randint(0, 9999)}")
        for p in [p for at, p in insert_at.items() if at == phase]:
            labels.append(_inject(w, p))
        w.agent_text(f"Phase {phase} complete.")

    w.agent_text("All phases complete. Task done.")
    w.f.close()
    return SyntheticTrace(path=path, labels=labels, n_lines=w.line)


def _inject(w: _Writer, kind: str) -> PathologyLabel:
    start = w.line + 1
    if kind == "loop":
        frozen_input = {"cmd": "npm test -- --grep widget"}
        for _ in range(6):
            tid, _ = w.tool_call("Bash", frozen_input)
            w.tool_result(tid, "FAIL: widget.spec.ts — timeout", is_error=False)
    elif kind == "error_streak":
        for i in range(5):
            tid, _ = w.tool_call("Edit", {"file_path": f"/src/w{i}.ts"})
            w.tool_result(tid, "Error: old_string not found", is_error=True)
    elif kind == "tool_flood":
        # must exceed ToolFloodDetector's default cap (100) under default config
        for i in range(120):
            tid, _ = w.tool_call("WebSearch", {"query": f"widget docs page {i}"})
            w.tool_result(tid, f"results {i}")
    elif kind == "stall":
        tid, _ = w.tool_call("Bash", {"cmd": "long build"})
        w.tool_result(tid, "done after long wait", gap_s=1800.0)
    else:
        raise ValueError(f"unknown pathology {kind}")
    return PathologyLabel(kind=kind, start_line=start, end_line=w.line)
