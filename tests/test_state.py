"""M1 tests: reducer correctness + fold determinism + detector behavior + DETECT bench."""

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tracelab.bench.detect_bench import run as run_detect
from tracelab.detect.detectors import (
    ErrorStreakDetector, LoopDetector, StallDetector, ToolFloodDetector, default_detectors,
)
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.state.reducers import StateFolder, fold

from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


@pytest.fixture
def events(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "refactor the parser"),
        rec_assistant("a1", [
            {"type": "text", "text": "Starting."},
            {"type": "tool_use", "id": "t1", "name": "Read",
             "input": {"file_path": "/x/parser.py"}},
        ], parent="u1", usage={"input_tokens": 100, "output_tokens": 200,
                               "cache_read_input_tokens": 5000,
                               "cache_creation_input_tokens": 1000}),
        rec_tool_result("u2", "t1", "contents", parent="a1"),
        rec_assistant("a2", [
            {"type": "tool_use", "id": "t2", "name": "Edit",
             "input": {"file_path": "/x/parser.py", "old_string": "a", "new_string": "b"}},
        ], parent="u2"),
    ])
    return parse_file(p).events


def test_reducers_basics(events):
    s = fold(events)
    assert s.goal_text.startswith("refactor the parser")
    assert s.n_turns == 1
    assert s.n_tool_calls == 2
    assert s.per_tool == {"Read": 1, "Edit": 1}
    assert s.files_touched == {"/x/parser.py": 2}
    assert s.tokens.cache_read == 5000 and s.tokens.output == 200
    assert s.est_cost_usd > 0
    # t1 closed, t2 open
    assert [c.correlation_id for c in s.pending_tools] == ["t2"]
    assert s.frontier.startswith("calling Edit")


def test_incremental_fold_equals_whole(events):
    whole = fold(events)
    f = StateFolder()
    for ev in events:
        f.apply(ev)
    inc = f.state
    assert inc.model_dump() == whole.model_dump()


@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=0, max_value=10))
def test_fold_deterministic_under_chunking(tmp_path_factory, cut):
    tmp = tmp_path_factory.mktemp("fold")
    p = tmp / "s.jsonl"
    generate(p, seed=7, pathologies=["loop", "error_streak"])
    events = parse_file(p).events
    cut = min(cut * len(events) // 10, len(events))
    f = StateFolder()
    f.fold(events[:cut])
    f.fold(events[cut:])
    assert f.state.model_dump() == fold(events).model_dump()


def test_pending_never_negative_and_closed_have_duration(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=3)
    events = parse_file(p).events
    f = StateFolder()
    calls, matched = set(), set()
    for ev in events:
        f.apply(ev)
        if ev.kind.value == "tool_call" and ev.correlation_id:
            calls.add(ev.correlation_id)
        elif ev.kind.value == "tool_result" and ev.correlation_id in calls:
            matched.add(ev.correlation_id)
        assert len(f.state.pending_tools) == len(calls - matched)
    closed = [c for c in f._open_calls.values() if not c.open]
    assert closed and all(c.result_seq >= c.call_seq for c in closed)
    assert all(c.duration_s is not None and c.duration_s >= 0 for c in closed)


# ------------------------------------------------------------- detectors

def _run_detected(tmp_path, pathologies, detectors):
    p = tmp_path / "d.jsonl"
    generate(p, seed=42, pathologies=pathologies)
    events = parse_file(p).events
    folder = StateFolder(detectors=detectors)
    folder.fold(events)
    return folder.state


def test_loop_detector_fires(tmp_path):
    s = _run_detected(tmp_path, ["loop"], [LoopDetector()])
    assert any(a.kind == "loop" for a in s.anomalies)


def test_error_streak_fires_and_clears(tmp_path):
    s = _run_detected(tmp_path, ["error_streak"], [ErrorStreakDetector()])
    streaks = [a for a in s.anomalies if a.kind == "error_streak"]
    assert streaks
    # generator resumes healthy tool results after the streak -> anomaly clears
    assert all(not a.active for a in streaks)


def test_flood_and_stall_fire(tmp_path):
    s = _run_detected(tmp_path, ["tool_flood", "stall"],
                      [ToolFloodDetector(default_cap=20), StallDetector(gap_s=600)])
    kinds = {a.kind for a in s.anomalies}
    assert "tool_flood" in kinds and "stall" in kinds


def test_clean_trace_no_anomalies(tmp_path):
    s = _run_detected(tmp_path, [], default_detectors())
    assert s.active_anomalies == []


# ------------------------------------------------------------- DETECT bench

def test_detect_benchmark_floor():
    res = run_detect(n_traces=16, seed0=500)
    for kind in ("loop", "error_streak", "tool_flood", "stall"):
        r = res["per_kind"][kind]
        assert r["tp"] + r["fn"] > 0, f"{kind}: no labels scored — generator regression"
        assert r["recall"] >= 0.9, (kind, r)
    assert res["clean_trace_false_positives"] == 0
