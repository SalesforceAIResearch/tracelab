#!/usr/bin/env python3
"""From-scratch recount oracle for Appendix B (fold fidelity on CONTINUE traces).

Two independent sides, compared key by key:

  Side A — oracle recount. Parses the raw workbench JSONL with REGEXES ONLY:
    no json module, no tracelab event pipeline. It reconstructs, straight from
    the bytes on disk, every `key = value` fact line that appears in a
    non-error tool_result, scoped by the source file of the correlated tool
    call (the fold's occurrence identity: key = "<path arg>:<field>", with a
    "#n" suffix for the n-th DISTINCT value seen under the same scoped key).

  Side B — the fold under test. Drives the public tracelab pipeline exactly
    the way the workbench curator does (see TraceViewPolicy in
    src/tracelab/workbench/worker.py):

        events = parse_file(path).events
        folder = StateFolder(str(path), detectors=default_detectors())
        folder.fold(events)

    and reads state.facts (per-key facts) and state.fact_aggregates
    (base -> {n, sum, keys}: numeric facts folded losslessly on eviction).

Checks (each failure is one mismatch):
  A. every per-key fact the fold reports exists in the raw recount with the
     exact same value;
  B. every aggregate's bookkeeping is right: n == len(keys), sum equals the
     raw recount's sum over exactly those keys, no key is simultaneously in
     facts and in an aggregate (double count), no aggregate key is absent
     from the raw recount;
  C. per-base totals: sum of numeric fold facts + aggregate sum == raw sum of
     ALL numeric occurrences of that base (e.g. base "delta" must total the
     sum of every delta in the trace), and occurrence counts agree;
  D. completeness: every NUMERIC raw occurrence is accounted for by the fold,
     either as a live fact or inside an aggregate's key list. (Non-numeric
     facts evicted from the bounded store are dropped by design; they are
     reported informationally, not as mismatches.)

Also printed per trace (informational): the raw recount's delta total vs the
answer the worker itself wrote to total.txt ("total = N" in the write_file
input) — an environment-level cross-check of the oracle.

Usage:
    uv run python tools/recount_oracle.py TRACE.jsonl [TRACE.jsonl ...]
    uv run python tools/recount_oracle.py            # defaults to the first
                                                     # five chain-120 traces

Exit status: 0 iff total mismatches == 0.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Side A: regex-only recount (never imports the event pipeline)
# --------------------------------------------------------------------------

# Structural patterns over the raw JSONL line. Field order matches the
# workbench Recorder's dict literals. These cannot false-positive inside
# string content: there, every quote is escaped (\"), so the bare-quote
# patterns below do not match.
TOOL_USE_RE = re.compile(
    r'"type"\s*:\s*"tool_use"\s*,\s*"id"\s*:\s*"([^"]+)"\s*,'
    r'\s*"name"\s*:\s*"([^"]+)"\s*,\s*"input"\s*:\s*'
)
TOOL_RESULT_RE = re.compile(
    r'"type"\s*:\s*"tool_result"\s*,\s*"tool_use_id"\s*:\s*"([^"]+)"\s*,'
    r'\s*"content"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"is_error"\s*:\s*(true|false)'
)
# Same source-attribution rule as the fold (reducers._extract_facts), applied
# to the first ~200 chars of the tool_use input — mirroring input_preview.
SOURCE_RE = re.compile(r'"(?:file_path|notebook_path|path)"\s*:\s*"((?:[^"\\]|\\.)*)"')
# Byte-identical to reducers.FACT_RE.
FACT_RE = re.compile(r"^\s*([A-Za-z_][\w.\-]{1,40})\s*=\s*(.{1,120}?)\s*$")
SKIP_KEYS = {"retries"}          # reducers._extract_facts skips these
MAX_FACT_LINES = 200             # reducers scan at most 200 lines per result

WRITTEN_TOTAL_RE = re.compile(r"total\s*=\s*(-?\d+)")

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
            "n": "\n", "r": "\r", "t": "\t"}


def _unescape(s: str) -> str:
    """Undo JSON string escapes with regex substitution (no json module)."""
    return re.sub(
        r"\\u([0-9a-fA-F]{4})|\\(.)",
        lambda m: chr(int(m.group(1), 16)) if m.group(1)
        else _ESCAPES.get(m.group(2), m.group(2)),
        s,
    )


class Recount:
    def __init__(self):
        # scoped key -> distinct values in first-appearance order
        self.values: dict[str, list[str]] = {}
        self.written_total: int | None = None
        self.n_records = 0
        self.n_tool_results = 0

    @property
    def occurrences(self) -> dict[str, str]:
        """Flatten to the fold's key naming: 1st distinct value -> K,
        n-th distinct value -> K#n."""
        occ: dict[str, str] = {}
        for k, vals in self.values.items():
            for i, v in enumerate(vals):
                occ[k if i == 0 else f"{k}#{i + 1}"] = v
        return occ


def recount(path: Path) -> Recount:
    rc = Recount()
    calls: dict[str, str | None] = {}   # tool_use_id -> source path arg
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            rc.n_records += 1

            for m in TOOL_USE_RE.finditer(line):
                tool_id, tool_name = m.group(1), m.group(2)
                region = line[m.end():m.end() + 200]   # ~ input_preview[:200]
                sm = SOURCE_RE.search(region)
                calls[tool_id] = _unescape(sm.group(1)) if sm else None
                if tool_name == "write_file":
                    cm = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"',
                                   line[m.end():])
                    if cm:
                        tm = WRITTEN_TOTAL_RE.search(_unescape(cm.group(1)))
                        if tm:
                            rc.written_total = int(tm.group(1))

            for m in TOOL_RESULT_RE.finditer(line):
                tool_id, raw_content, is_error = m.group(1), m.group(2), m.group(3)
                rc.n_tool_results += 1
                if is_error == "true":
                    continue                      # fold skips error payloads
                source = calls.get(tool_id)
                text = _unescape(raw_content)
                for fact_line in text.splitlines()[:MAX_FACT_LINES]:
                    fm = FACT_RE.match(fact_line)
                    if not fm:
                        continue
                    key, val = fm.group(1), fm.group(2)
                    if key in SKIP_KEYS:
                        continue
                    scoped = f"{source}:{key}" if source else key
                    vals = rc.values.setdefault(scoped, [])
                    if val not in vals:           # occurrence identity:
                        vals.append(val)          # distinct values per key
    return rc


# --------------------------------------------------------------------------
# Side B: drive the fold via the public API (as the workbench curator does)
# --------------------------------------------------------------------------

def drive_fold(path: Path):
    from tracelab.detect.detectors import default_detectors
    from tracelab.ledger.adapters.claude_code import parse_file
    from tracelab.state.reducers import StateFolder

    events = parse_file(str(path)).events
    folder = StateFolder(str(path), detectors=default_detectors())
    folder.fold(events)
    return folder.state


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _base(key: str) -> str:
    return key.rsplit(":", 1)[-1].split("#", 1)[0]


def _as_int(val) -> int | None:
    try:
        return int(str(val).strip())
    except ValueError:
        return None


def compare(path: Path) -> tuple[int, int]:
    rc = recount(path)
    occ = rc.occurrences
    state = drive_fold(path)
    facts, aggs = state.facts, state.fact_aggregates
    mismatches: list[str] = []

    # A. every fold per-key fact reproduced by the raw recount
    for k, v in facts.items():
        if k not in occ:
            mismatches.append(f"[A] fold fact {k!r}={v!r} not found in raw recount")
        elif occ[k] != v:
            mismatches.append(f"[A] fold fact {k!r}: fold={v!r} raw={occ[k]!r}")

    # B. aggregate bookkeeping vs raw values of exactly the evicted keys
    agg_keys_all: set[str] = set()
    for base, agg in aggs.items():
        keys = agg.get("keys", [])
        agg_keys_all.update(keys)
        raw_sum = 0
        for k in keys:
            if k in facts:
                mismatches.append(
                    f"[B] key {k!r} double-counted: live fact AND in aggregate {base!r}")
            if k not in occ:
                mismatches.append(
                    f"[B] aggregate {base!r} key {k!r} not found in raw recount")
                continue
            n = _as_int(occ[k])
            if n is None:
                mismatches.append(
                    f"[B] aggregate {base!r} holds non-numeric key {k!r}={occ[k]!r}")
            else:
                raw_sum += n
        if agg.get("n") != len(keys):
            mismatches.append(
                f"[B] aggregate {base!r}: n={agg.get('n')} != len(keys)={len(keys)}")
        if agg.get("sum") != raw_sum:
            mismatches.append(
                f"[B] aggregate sum {base!r}: fold={agg.get('sum')} raw={raw_sum}")

    # C. per-base totals: live numeric facts + aggregate == all raw numerics
    fold_sum: dict[str, int] = {}
    fold_n: dict[str, int] = {}
    for k, v in facts.items():
        n = _as_int(v)
        if n is None:
            continue
        b = _base(k)
        fold_sum[b] = fold_sum.get(b, 0) + n
        fold_n[b] = fold_n.get(b, 0) + 1
    for base, agg in aggs.items():
        fold_sum[base] = fold_sum.get(base, 0) + agg.get("sum", 0)
        fold_n[base] = fold_n.get(base, 0) + agg.get("n", 0)
    raw_sum_by_base: dict[str, int] = {}
    raw_n_by_base: dict[str, int] = {}
    for k, v in occ.items():
        n = _as_int(v)
        if n is None:
            continue
        b = _base(k)
        raw_sum_by_base[b] = raw_sum_by_base.get(b, 0) + n
        raw_n_by_base[b] = raw_n_by_base.get(b, 0) + 1
    for b in sorted(set(fold_sum) | set(raw_sum_by_base)):
        if (fold_sum.get(b, 0) != raw_sum_by_base.get(b, 0)
                or fold_n.get(b, 0) != raw_n_by_base.get(b, 0)):
            mismatches.append(
                f"[C] total[{b}]: fold sum={fold_sum.get(b, 0)} "
                f"(n={fold_n.get(b, 0)}) vs raw sum={raw_sum_by_base.get(b, 0)} "
                f"(n={raw_n_by_base.get(b, 0)})")

    # D. every raw NUMERIC occurrence accounted for somewhere in the fold
    dropped_nonnumeric = 0
    for k, v in occ.items():
        if k in facts or k in agg_keys_all:
            continue
        if _as_int(v) is None:
            dropped_nonnumeric += 1    # bounded-store eviction: by design
        else:
            mismatches.append(
                f"[D] raw numeric occurrence {k!r}={v!r} absent from fold "
                f"facts and aggregates")

    # ---------------- report ----------------
    agg_desc = ", ".join(
        f"{b}(n={a.get('n')}, sum={a.get('sum')})" for b, a in sorted(aggs.items())
    ) or "none"
    delta_raw = raw_sum_by_base.get("delta")
    print(f"== {path.name}")
    print(f"   raw recount : {rc.n_records} records, {rc.n_tool_results} tool_results, "
          f"{len(occ)} fact occurrences ({sum(1 for v in occ.values() if _as_int(v) is not None)} numeric)")
    print(f"   fold        : {len(facts)} live facts, aggregates: {agg_desc}")
    print(f"   delta total : raw={delta_raw}  "
          f"fold(live+agg)={fold_sum.get('delta')}  "
          f"worker-written total.txt={rc.written_total}"
          + ("" if rc.written_total == delta_raw else "  [worker answer differs]"))
    print(f"   non-numeric facts evicted by the bounded store (by design, "
          f"not mismatches): {dropped_nonnumeric}")
    for msg in mismatches:
        print(f"   MISMATCH {msg}")
    print(f"   mismatches  : {len(mismatches)}")
    return len(mismatches), dropped_nonnumeric


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(a) for a in argv]
    else:
        trace_dir = Path(__file__).resolve().parent.parent / \
            "bench" / "continue_v23_traces" / "trace_120"
        paths = sorted(trace_dir.glob("*.jsonl"))[:5]
    if not paths:
        print("no trace files given/found", file=sys.stderr)
        return 2
    total = 0
    for p in paths:
        n, _ = compare(p)
        total += n
    print(f"\nTOTAL: {len(paths)} traces, {total} mismatches")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
