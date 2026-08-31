"""Regression pins for every M0/M1 gate finding (adversarial review 2026-08-17).

Each test names the finding it pins. If one of these fails, a previously-fixed
bug is back.
"""

import json
import threading
import time
from pathlib import Path

from tracelab.detect.detectors import (
    BudgetDetector, LoopDetector, StallDetector, ToolFloodDetector, default_detectors,
)
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.ledger.envelope import INLINE_CAP, EventKind, fingerprint, make_payload
from tracelab.ledger.store import Ledger
from tracelab.state.reducers import StateFolder, fold

from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


# ---- CRITICAL: usage double counting across per-block records (same message.id)

def test_usage_counted_once_per_api_message(tmp_path):
    usage = {"input_tokens": 10, "output_tokens": 100,
             "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}
    r1 = rec_assistant("a1", [{"type": "text", "text": "part one"}], usage=usage)
    r2 = rec_assistant("a2", [{"type": "tool_use", "id": "t1", "name": "Bash",
                               "input": {"cmd": "ls"}}], usage=usage)
    # Claude Code style: same API message split across two records
    r1["message"]["id"] = r2["message"]["id"] = "msg_SAME"
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [rec_user("u1", "go"), r1, r2])
    s = fold(parse_file(p).events)
    assert s.tokens.output == 100          # not 200
    assert s.tokens.cache_read == 1000     # not 2000
    assert sum(s.models_seen.values()) == 1


def test_usage_distinct_messages_both_counted(tmp_path):
    usage = {"input_tokens": 0, "output_tokens": 100,
             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    r1 = rec_assistant("a1", [{"type": "text", "text": "x"}], usage=usage)
    r2 = rec_assistant("a2", [{"type": "text", "text": "y"}], usage=usage)
    r1["message"]["id"], r2["message"]["id"] = "msg_A", "msg_B"
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [r1, r2])
    assert fold(parse_file(p).events).tokens.output == 200


# ---- MAJOR: naive datetime must not crash the fold

def test_naive_timestamp_does_not_crash(tmp_path):
    r1 = rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                               "input": {}}])
    r2 = rec_tool_result("u2", "t1", "ok")
    r2["timestamp"] = "2026-08-17T10:00:05"  # offset-less
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [r1, r2])
    s = fold(parse_file(p).events, detectors=default_detectors())
    assert s.n_tool_calls == 1  # got here without TypeError


# ---- MAJOR: synthetic generator injects EVERY requested pathology

def test_synthetic_no_silent_pathology_drop(tmp_path):
    for seed in range(20):
        t = generate(tmp_path / f"t{seed}.jsonl", seed=seed,
                     pathologies=["loop", "error_streak", "tool_flood"])
        assert sorted(l.kind for l in t.labels) == ["error_streak", "loop", "tool_flood"]


# ---- MAJOR: out-of-order tool_result; dangling calls; stall clears on user turn

def test_result_before_call_and_user_turn_clears_stall(tmp_path):
    recs = [
        rec_tool_result("u0", "tX", "orphan result"),          # result before its call
        rec_assistant("a1", [{"type": "tool_use", "id": "tX", "name": "Bash",
                              "input": {}}]),                   # late call, never resolved
        rec_user("u1", "never mind, do something else"),        # new user turn
    ]
    # then quiet traffic with big gaps but nothing in flight
    r = rec_assistant("a2", [{"type": "text", "text": "ok"}])
    r["timestamp"] = "2026-08-17T11:00:00.000Z"                 # >> gap_s after u1
    recs.append(r)
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    s = fold(parse_file(p).events, detectors=[StallDetector(gap_s=60)])
    stalls = [a for a in s.anomalies if a.kind == "stall" and a.active]
    assert stalls == [], "dangling call must not poison post-user-turn gaps"


# ---- MAJOR: real invariant for pending tools (replaces tautology)

def test_pending_equals_calls_minus_matched_results(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=11, pathologies=["error_streak"])
    events = parse_file(p).events
    f = StateFolder()
    calls, matched = set(), set()
    for ev in events:
        f.apply(ev)
        if ev.agent_id is None:
            if ev.kind == EventKind.TOOL_CALL and ev.correlation_id:
                calls.add(ev.correlation_id)
            elif ev.kind == EventKind.TOOL_RESULT and ev.correlation_id in calls:
                matched.add(ev.correlation_id)
        assert len(f.state.pending_tools) == len(calls - matched)


# ---- MAJOR: INLINE_CAP / sha256 semantics

def test_payload_cap_boundary_and_full_hash():
    exactly = "х" * INLINE_CAP           # multibyte unicode
    over = "у" * (INLINE_CAP + 1)
    p1, p2 = make_payload(exactly), make_payload(over)
    assert p1.truncated is False and p1.char_len == INLINE_CAP
    assert p2.truncated is True and len(p2.text) == INLINE_CAP
    assert p2.char_len == INLINE_CAP + 1
    assert p2.sha256 == fingerprint(over)  # hash of FULL text, not the prefix


def test_loop_detected_on_huge_identical_inputs(tmp_path):
    big = {"cmd": "x" * (INLINE_CAP * 2)}
    recs = []
    for i in range(5):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "Bash", "input": big}]))
        recs.append(rec_tool_result(f"u{i}", f"t{i}", "same failure"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    s = fold(parse_file(p).events, detectors=[LoopDetector(repeats=4)])
    assert any(a.kind == "loop" for a in s.anomalies)


def test_files_touched_survives_truncated_write(tmp_path):
    huge_content = "line\n" * 2000  # pushes input json over INLINE_CAP
    recs = [rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Write",
                                  "input": {"file_path": "/big/file.txt",
                                            "content": huge_content}}])]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    s = fold(parse_file(p).events)
    assert s.files_touched.get("/big/file.txt") == 1


# ---- MAJOR: sidechain events do not mutate main-thread position state

def test_sidechain_excluded_from_position_but_costed(tmp_path):
    side_user = rec_user("su1", "subagent internal prompt", isSidechain=True)
    side_call = rec_assistant("sa1", [{"type": "tool_use", "id": "st1", "name": "Grep",
                                       "input": {}}],
                              usage={"input_tokens": 0, "output_tokens": 500,
                                     "cache_read_input_tokens": 0,
                                     "cache_creation_input_tokens": 0})
    side_call["isSidechain"] = True
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [rec_user("u1", "main goal"), side_user, side_call])
    events = parse_file(p).events
    assert any(e.agent_id == "sidechain" for e in events)
    s = fold(events)
    assert s.n_turns == 1
    assert s.goal_text == "main goal"
    assert s.latest_user_directive == "main goal"
    assert s.pending_tools == [] and "Grep" not in s.per_tool
    assert s.tokens.output == 500          # cost still counted
    assert s.sidechain_events == 2


# ---- detector threshold boundaries

def _mk_call(i, name="Bash", inp=None):
    return rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}", "name": name,
                                    "input": inp if inp is not None else {"n": i}}])


def test_loop_boundary_exact(tmp_path):
    frozen = {"cmd": "same"}
    recs = [_mk_call(i, inp=frozen) for i in range(3)]
    p = tmp_path / "s3.jsonl"
    write_jsonl(p, recs)
    s = fold(parse_file(p).events, detectors=[LoopDetector(repeats=4)])
    assert not any(a.kind == "loop" for a in s.anomalies), "3 repeats must not fire"

    recs.append(_mk_call(3, inp=frozen))
    write_jsonl(p, recs)
    s = fold(parse_file(p).events, detectors=[LoopDetector(repeats=4)])
    loop = [a for a in s.anomalies if a.kind == "loop"]
    assert loop and loop[0].severity == "warn", "4th repeat fires warn"

    recs.append(_mk_call(4, inp=frozen))
    write_jsonl(p, recs)
    s = fold(parse_file(p).events, detectors=[LoopDetector(repeats=4)])
    assert [a for a in s.anomalies if a.kind == "loop"][0].severity == "critical"


def test_flood_boundary_exact(tmp_path):
    p = tmp_path / "f.jsonl"
    write_jsonl(p, [_mk_call(i) for i in range(4)])
    s = fold(parse_file(p).events, detectors=[ToolFloodDetector(default_cap=5)])
    assert not any(a.kind == "tool_flood" for a in s.anomalies)
    write_jsonl(p, [_mk_call(i) for i in range(5)])
    s = fold(parse_file(p).events, detectors=[ToolFloodDetector(default_cap=5)])
    assert any(a.kind == "tool_flood" for a in s.anomalies), "fires at exactly cap"


def test_budget_boundary_and_escalation(tmp_path):
    # fixture model is test-premium-1 (synthetic premium tier): output $75/MTok -> 200K tokens = $15 per spend
    def spend(i, out_tokens=200_000):
        r = rec_assistant(f"a{i}", [{"type": "text", "text": "x"}],
                          usage={"input_tokens": 0, "output_tokens": out_tokens,
                                 "cache_read_input_tokens": 0,
                                 "cache_creation_input_tokens": 0})
        r["message"]["id"] = f"msg_{i}"
        return r
    p = tmp_path / "b.jsonl"
    write_jsonl(p, [spend(0)])                    # $15 < warn(20)
    s = fold(parse_file(p).events, detectors=[BudgetDetector(warn_usd=20, critical_usd=40)])
    assert not any(a.kind == "budget" for a in s.anomalies)
    write_jsonl(p, [spend(i) for i in range(2)])  # $30 >= warn
    s = fold(parse_file(p).events, detectors=[BudgetDetector(warn_usd=20, critical_usd=40)])
    assert [a for a in s.anomalies if a.kind == "budget"][0].severity == "warn"
    write_jsonl(p, [spend(i) for i in range(3)])  # $45 >= critical, same anomaly escalates
    s = fold(parse_file(p).events, detectors=[BudgetDetector(warn_usd=20, critical_usd=40)])
    budget = [a for a in s.anomalies if a.kind == "budget"]
    assert len(budget) == 1 and budget[0].severity == "critical"


# ---- store: tail() live contract, resume-then-append, quality counters survive

def test_tail_yields_each_event_exactly_once(tmp_path):
    p = tmp_path / "live.jsonl"
    write_jsonl(p, [rec_user("u1", "start")])
    appended = threading.Event()

    def writer():
        time.sleep(0.05)
        with open(p, "a") as f:
            f.write(json.dumps(rec_assistant("a1", [{"type": "text", "text": "hi"}])) + "\n")
            f.write(json.dumps(rec_user("u2", "more")) + "\n")
        appended.set()

    threading.Thread(target=writer, daemon=True).start()
    led = Ledger(p)
    got = []
    batches = []
    for ev in led.tail(poll_s=0.02,
                       stop=lambda: appended.is_set() and len(got) >= 3,
                       on_batch=lambda b: batches.append([e.event_id for e in b])):
        got.append(ev.event_id)
        if len(got) >= 3 and appended.is_set():
            break
    whole = [e.event_id for e in parse_file(p).events]
    assert got == whole
    assert [i for b in batches for i in b] == whole  # batches partition the stream


def test_load_then_append_then_ingest(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [rec_user("u1", "one"), rec_user("u2", "two")])
    led = Ledger(p)
    led.ingest_available()
    snap = tmp_path / "snap.jsonl"
    led.dump(snap)
    with open(p, "a") as f:
        f.write(json.dumps(rec_user("u3", "three")) + "\n")
    led2 = Ledger.load(snap)
    assert led2.ingest_available() == 1
    fresh = Ledger(p)
    fresh.ingest_available()
    assert [e.event_id for e in led2.events] == [e.event_id for e in fresh.events]
    assert [e.seq for e in led2.events] == [e.seq for e in fresh.events]


def test_quality_counters_survive_snapshot(tmp_path):
    p = tmp_path / "s.jsonl"
    with open(p, "w") as f:
        f.write("{broken\n")
        f.write(json.dumps({"type": "weird-new", "uuid": "w1", "sessionId": "s"}) + "\n")
    led = Ledger(p)
    led.ingest_available()
    assert led.malformed == 1 and led.unknown_types == {"weird-new": 1}
    snap = tmp_path / "snap.jsonl"
    led.dump(snap)
    led2 = Ledger.load(snap)
    assert led2.malformed == 1 and led2.unknown_types == {"weird-new": 1}


# ---- refold-after-load reproduces state incl. anomaly lifecycle

def test_refold_after_load_reproduces_state(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=21, pathologies=["error_streak", "loop"])
    led = Ledger(p)
    led.ingest_available()
    s1 = fold(led.events, detectors=default_detectors())
    snap = tmp_path / "snap.jsonl"
    led.dump(snap)
    led2 = Ledger.load(snap)
    s2 = fold(led2.events, detectors=default_detectors())
    assert s1.model_dump() == s2.model_dump()
