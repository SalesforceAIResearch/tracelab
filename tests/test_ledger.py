"""M0 tests: adapter correctness, ingest-resume equivalence, robustness, snapshot replay."""

import json
import os
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.envelope import Event, EventKind, Source
from tracelab.ledger.store import Ledger


# ---------------------------------------------------------------- fixtures

def rec_user(uuid, text, parent=None, session="s1", **kw):
    return {"type": "user", "uuid": uuid, "parentUuid": parent, "sessionId": session,
            "timestamp": "2026-08-17T10:00:00.000Z",
            "message": {"role": "user", "content": text}, **kw}


def rec_assistant(uuid, blocks, parent=None, session="s1", usage=None, **kw):
    msg = {"role": "assistant", "model": "test-premium-1", "content": blocks}
    if usage:
        msg["usage"] = usage
    return {"type": "assistant", "uuid": uuid, "parentUuid": parent, "sessionId": session,
            "timestamp": "2026-08-17T10:00:01.000Z", "message": msg, **kw}


def rec_tool_result(uuid, tool_use_id, text, parent=None, session="s1"):
    return {"type": "user", "uuid": uuid, "parentUuid": parent, "sessionId": session,
            "timestamp": "2026-08-17T10:00:02.000Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]}}


def write_jsonl(path: Path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "session.jsonl"
    write_jsonl(p, [
        {"type": "ai-title", "sessionId": "s1", "aiTitle": "Test run"},
        rec_user("u1", "fix the bug in foo.py"),
        rec_assistant("a1", [
            {"type": "thinking", "thinking": "let me look at foo.py", "signature": "x"},
            {"type": "text", "text": "I'll inspect foo.py."},
            {"type": "tool_use", "id": "toolu_01", "name": "Read",
             "input": {"file_path": "/tmp/foo.py"}},
        ], parent="u1", usage={"input_tokens": 10, "output_tokens": 50,
                               "cache_read_input_tokens": 900,
                               "cache_creation_input_tokens": 100}),
        rec_tool_result("u2", "toolu_01", "def foo():\n    return 1\n", parent="a1"),
        rec_assistant("a2", [{"type": "text", "text": "Found it."}], parent="u2"),
    ])
    return p


# ---------------------------------------------------------------- adapter

def test_block_explosion_and_kinds(sample):
    res = parse_file(sample)
    kinds = [(e.event_id, e.kind) for e in res.events]
    assert (("a1#0", EventKind.THINKING) in kinds)
    assert (("a1#1", EventKind.MESSAGE) in kinds)
    assert (("a1#2", EventKind.TOOL_CALL) in kinds)
    assert (("u2#0", EventKind.TOOL_RESULT) in kinds)
    assert res.stats.malformed == 0


def test_tool_correlation(sample):
    res = parse_file(sample)
    call = next(e for e in res.events if e.kind == EventKind.TOOL_CALL)
    result = next(e for e in res.events if e.kind == EventKind.TOOL_RESULT)
    assert call.correlation_id == result.correlation_id == "toolu_01"
    assert call.payload.tool_name == "Read"


def test_usage_attached_once(sample):
    res = parse_file(sample)
    with_usage = [e for e in res.events if e.usage is not None]
    assert len(with_usage) == 1
    u = with_usage[0].usage
    assert u.cache_read_tokens == 900 and u.cache_creation_tokens == 100


def test_causal_parent_preserved(sample):
    res = parse_file(sample)
    a1_blocks = [e for e in res.events if e.record_id == "a1"]
    assert all(e.parent_event_id == "u1" for e in a1_blocks)


def test_seq_monotone_and_dense(sample):
    res = parse_file(sample)
    assert [e.seq for e in res.events] == list(range(len(res.events)))


# ------------------------------------------------------- robustness contract

def test_malformed_lines_counted_not_fatal(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps(rec_user("u1", "hello")) + "\n")
        f.write("{this is not json\n")
        f.write("[1,2,3]\n")  # json but not an object
        f.write(json.dumps(rec_user("u2", "world")) + "\n")
    res = parse_file(p)
    assert res.stats.malformed == 2
    assert [e.record_id for e in res.events] == ["u1", "u2"]


def test_partial_final_line_not_consumed(tmp_path):
    p = tmp_path / "partial.jsonl"
    full = json.dumps(rec_user("u1", "hello")) + "\n"
    partial = json.dumps(rec_user("u2", "in-flight"))[:20]  # writer mid-line
    p.write_bytes((full + partial).encode())
    res = parse_file(p)
    assert [e.record_id for e in res.events] == ["u1"]
    assert res.resume_offset == len(full.encode())
    # writer finishes the line -> resume picks up exactly u2
    p.write_bytes((full + json.dumps(rec_user("u2", "in-flight")) + "\n").encode())
    res2 = parse_file(p, start_offset=res.resume_offset, start_seq=len(res.events),
                      start_line=2)
    assert [e.record_id for e in res2.events] == ["u2"]


def test_unknown_types_preserved(tmp_path):
    p = tmp_path / "unk.jsonl"
    write_jsonl(p, [{"type": "future-thing", "uuid": "x1", "sessionId": "s1", "data": 42}])
    res = parse_file(p)
    assert len(res.events) == 1
    assert res.events[0].kind == EventKind.UNKNOWN
    assert res.stats.unknown_types == {"future-thing": 1}


# ------------------------------------------------- ingest-resume equivalence

@st.composite
def record_streams(draw):
    n = draw(st.integers(min_value=1, max_value=12))
    recs = []
    for i in range(n):
        which = draw(st.integers(min_value=0, max_value=2))
        text = draw(st.text(min_size=0, max_size=80))
        if which == 0:
            recs.append(rec_user(f"u{i}", text))
        elif which == 1:
            recs.append(rec_assistant(f"a{i}", [{"type": "text", "text": text}]))
        else:
            recs.append(rec_tool_result(f"t{i}", f"toolu_{i}", text))
    return recs


@settings(max_examples=40, deadline=None)
@given(recs=record_streams(), cut_frac=st.floats(min_value=0.0, max_value=1.0))
def test_chunked_ingest_equals_whole(tmp_path_factory, recs, cut_frac):
    tmp = tmp_path_factory.mktemp("prop")
    p = tmp / "s.jsonl"
    write_jsonl(p, recs)
    whole = parse_file(p)

    raw = p.read_bytes()
    # cut at an arbitrary byte position: first pass sees a possibly-partial file
    cut = int(len(raw) * cut_frac)
    p2 = tmp / "s2.jsonl"
    p2.write_bytes(raw[:cut])
    r1 = parse_file(p2)
    p2.write_bytes(raw)  # writer completes
    r2 = parse_file(p2, start_offset=r1.resume_offset, start_seq=len(r1.events),
                    start_line=1 + r1.stats.lines)
    combined = r1.events + r2.events
    assert [e.event_id for e in combined] == [e.event_id for e in whole.events]
    assert [e.kind for e in combined] == [e.kind for e in whole.events]
    assert [e.payload.sha256 for e in combined] == [e.payload.sha256 for e in whole.events]


# --------------------------------------------------------- snapshot / replay

def test_ledger_snapshot_roundtrip(sample, tmp_path):
    led = Ledger(sample)
    led.ingest_available()
    snap = tmp_path / "snap.jsonl"
    led.dump(snap)
    led2 = Ledger.load(snap)
    assert [e.event_id for e in led2.events] == [e.event_id for e in led.events]
    assert led2.marks.durable_offset == led.marks.durable_offset
    # resumed ledger continues ingesting with no duplication
    assert led2.ingest_available() == 0


# --------------------------------------------------------- real corpus smoke

REAL_DIR = Path.home() / ".claude" / "projects"


@pytest.mark.skipif(not REAL_DIR.exists(), reason="no local Claude Code corpus")
def test_real_corpus_smoke():
    files = sorted(REAL_DIR.glob("*/*.jsonl"),
                   key=lambda p: p.stat().st_size, reverse=True)[:3]
    assert files, "corpus present but empty"
    for f in files:
        res = parse_file(f)
        assert res.stats.events > 0
        # malformed lines must be a tiny fraction of a real transcript
        assert res.stats.malformed <= max(2, res.stats.lines // 100)
        # every tool_call correlation eventually resolves or dangles at the tail
        calls = {e.correlation_id for e in res.events
                 if e.kind == EventKind.TOOL_CALL and e.correlation_id}
        results = {e.correlation_id for e in res.events
                   if e.kind == EventKind.TOOL_RESULT and e.correlation_id}
        matched = len(calls & results)
        if calls:
            assert matched / len(calls) > 0.8, f"{f}: only {matched}/{len(calls)} matched"
