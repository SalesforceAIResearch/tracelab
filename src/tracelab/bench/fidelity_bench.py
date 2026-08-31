"""FIDELITY benchmark v0: does the RunState tell the truth about the ledger?

Method: an INDEPENDENT oracle recomputes checkable facts straight from the raw JSONL
(different code path from the adapter/reducers — deliberately naive), then samples
assertions from the folded RunState and verifies each against the oracle:

  per-tool call counts · total tool calls · tool errors · user turns ·
  token sums (deduped by message id) · files touched · pending (dangling) calls

Scored per field across synthetic + (if present) real-corpus traces. LLM-judged
semantic fidelity (goal/frontier/conclusions) arrives with M3; this v0 catches
reducer/adapter drift against ground truth with zero inference cost.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.state.reducers import StateFolder

REAL_DIR = Path.home() / ".claude" / "projects"


def oracle(path: Path) -> dict:
    """Naive, independent recount straight from raw JSON records (main thread only)."""
    per_tool: Counter = Counter()
    calls = errors = turns = 0
    out_tok = cache_read = 0
    seen_msg = set()
    open_ids: set[str] = set()
    files: set[str] = set()
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict) or rec.get("isSidechain") or rec.get("isMeta"):
                continue
            t = rec.get("type")
            msg = rec.get("message") or {}
            content = msg.get("content")
            if t == "assistant":
                u = msg.get("usage")
                mid = msg.get("id") or rec.get("requestId")
                if isinstance(u, dict) and (mid or f"rec:{rec.get('uuid')}") not in seen_msg:
                    seen_msg.add(mid or f"rec:{rec.get('uuid')}")
                    out_tok += u.get("output_tokens", 0) or 0
                    cache_read += u.get("cache_read_input_tokens", 0) or 0
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            calls += 1
                            per_tool[b.get("name") or "unknown-tool"] += 1
                            if b.get("id"):
                                open_ids.add(b["id"])
                            inp = b.get("input")
                            if isinstance(inp, dict):
                                fp = inp.get("file_path") or inp.get("notebook_path")
                                if fp and (b.get("name") in
                                           ("Read", "Write", "Edit", "NotebookEdit")):
                                    files.add(fp)
            elif t == "user":
                handled = False
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result":
                            handled = True
                            open_ids.discard(b.get("tool_use_id"))
                            if b.get("is_error"):
                                errors += 1
                if not handled and rec.get("toolUseResult") is not None:
                    handled = True
                    open_ids.discard(rec.get("sourceToolUseID"))
                if not handled:
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "\n".join(b.get("text", "") for b in content
                                          if isinstance(b, dict) and b.get("type") == "text")
                    else:
                        text = ""
                    t = text.strip()
                    non_turn = ("[Request interrupted by user",)
                    if t and not t.startswith("<") and not any(
                            t.startswith(x) for x in non_turn):
                        turns += 1
    return {"per_tool": dict(per_tool), "n_tool_calls": calls, "n_tool_errors": errors,
            "n_turns": turns, "output_tokens": out_tok, "cache_read": cache_read,
            "n_files": len(files), "n_dangling": len(open_ids)}


FIELDS = ["per_tool", "n_tool_calls", "n_tool_errors", "n_turns",
          "output_tokens", "cache_read", "n_files", "n_dangling"]


def check_one(path: Path) -> dict:
    truth = oracle(path)
    s = StateFolder(str(path), detectors=default_detectors()).fold(parse_file(path).events)
    got = {"per_tool": s.per_tool, "n_tool_calls": s.n_tool_calls,
           "n_tool_errors": s.n_tool_errors, "n_turns": s.n_turns,
           "output_tokens": s.tokens.output, "cache_read": s.tokens.cache_read,
           "n_files": len(s.files_touched), "n_dangling": len(s.pending_tools)}
    return {f: (got[f] == truth[f], got[f], truth[f]) for f in FIELDS}


def run(n_synth: int = 12, n_real: int = 5, out: Path | None = None) -> dict:
    per_field: dict[str, list[bool]] = {f: [] for f in FIELDS}
    mismatches: list[str] = []

    def score(path: Path, tag: str):
        for f, (ok, got, want) in check_one(path).items():
            per_field[f].append(ok)
            if not ok:
                mismatches.append(f"{tag}:{path.name}:{f} got={got} want={want}")

    with tempfile.TemporaryDirectory() as td:
        kinds_cycle = [[], ["loop"], ["error_streak", "stall"], ["tool_flood"],
                       ["loop", "error_streak", "tool_flood", "stall"]]
        for i in range(n_synth):
            t = generate(Path(td) / f"f{i}.jsonl", seed=900 + i,
                         pathologies=kinds_cycle[i % len(kinds_cycle)])
            score(t.path, "synth")

    if REAL_DIR.exists() and n_real:
        real = sorted(REAL_DIR.glob("*/*.jsonl"), key=lambda p: p.stat().st_size,
                      reverse=True)[:n_real]
        for f in real:
            score(f, "real")

    field_scores = {f: (sum(v) / len(v) if v else None) for f, v in per_field.items()}
    overall = sum(sum(v) for v in per_field.values()) / max(
        1, sum(len(v) for v in per_field.values()))
    result = {"overall": round(overall, 4), "per_field": field_scores,
              "mismatches": mismatches[:20]}
    if out:
        board = json.loads(out.read_text()) if out.exists() else {}
        board["FIDELITY"] = result
        out.write_text(json.dumps(board, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(out=Path(__file__).resolve().parents[3] / "bench" / "scoreboard.json"),
                     indent=2))
