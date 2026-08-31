"""Regression pins for the M2 gate findings (adversarial review #2, 2026-08-17)."""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tracelab.bench import fidelity_bench
from tracelab.cli import main as cli_main
from tracelab.derived.episodes import build_episodes
from tracelab.derived.nodes import DerivedNode, NodeKind, NodeStore, Validity
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.state.reducers import fold
from tracelab.state.runstate import RunState
from tracelab.views.human_html import render

from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


def _node(nid, deps=None):
    return DerivedNode(node_id=nid, version=1, kind=NodeKind.FACT, covered_lo=0,
                       covered_hi=1, created_at_seq=1, producer="deterministic",
                       dependencies=deps or [])


# ---- lifecycle guard + version-sensitive staleness

def test_terminal_states_never_downgrade():
    st_ = NodeStore()
    a1 = st_.add(_node("a"))
    st_.evolve("a", at_seq=2, producer="deterministic")     # a@1 SUPERSEDED
    st_.mark_stale(a1.ref, at_seq=5)                        # must be a no-op on a@1
    assert st_.validity(a1.ref) == Validity.SUPERSEDED
    st_.invalidate(a1.ref, at_seq=6)                        # escalation allowed
    assert st_.validity(a1.ref) == Validity.INVALIDATED
    st_.mark_stale(a1.ref, at_seq=7)                        # INVALIDATED is sticky
    assert st_.validity(a1.ref) == Validity.INVALIDATED


def test_version_pinned_dependency_isolated_from_other_versions():
    st_ = NodeStore()
    st_.add(_node("a"))
    a2 = st_.evolve("a", at_seq=2, producer="deterministic")
    f = st_.add(_node("f", deps=[a2.ref]))                  # pinned to a@2 only
    st_.invalidate("a-0000@1" if False else st_.history("a")[0].ref, at_seq=9)
    assert st_.validity(f.ref) == Validity.CURRENT          # a@1 is not f's dep
    st_.invalidate(a2.ref, at_seq=10)
    assert st_.validity(f.ref) == Validity.SUSPECTED_STALE  # exact pin affected


def test_unpinned_dependency_matches_any_version():
    st_ = NodeStore()
    a1 = st_.add(_node("a"))
    g = st_.add(_node("g", deps=[a1.node_id]))              # bare id = any version
    st_.evolve("a", at_seq=2, producer="deterministic")
    st_.invalidate(st_.head("a").ref, at_seq=9)
    assert st_.validity(g.ref) == Validity.SUSPECTED_STALE


def test_propagation_passes_through_already_stale_nodes_and_diamonds():
    st_ = NodeStore()
    a = st_.add(_node("a"))
    b = st_.add(_node("b", deps=[a.ref]))
    c = st_.add(_node("c", deps=[a.ref]))
    st_.mark_stale(b.ref, at_seq=4)                          # b pre-staled (no dependents yet)
    d = st_.add(_node("d", deps=[b.ref, c.ref]))             # added AFTER b went stale
    assert st_.validity(d.ref) == Validity.CURRENT
    touched = st_.invalidate(a.ref, at_seq=9)
    assert st_.validity(d.ref) == Validity.SUSPECTED_STALE   # reached THROUGH stale b
    assert touched.count(d.ref) == 1                         # diamond: exactly once
    stale_audits = [t for t in st_.audit if t[0] == d.ref and t[1] == "suspected_stale"]
    assert len(stale_audits) == 1


@settings(max_examples=15, deadline=None)
@given(n=st.integers(min_value=1, max_value=30))
def test_version_chain_integrity_under_many_evolutions(n):
    st_ = NodeStore()
    st_.add(_node("x"))
    for i in range(n):
        st_.evolve("x", at_seq=10 + i, producer="deterministic", content=f"v{i+2}")
    hist = st_.history("x")
    assert len(hist) == n + 1
    assert [v.version for v in hist] == list(range(1, n + 2))
    assert all(st_.validity(v.ref) == Validity.SUPERSEDED for v in hist[:-1])
    assert st_.validity(hist[-1].ref) == Validity.CURRENT
    assert all(hist[k].supersedes == hist[k - 1].ref for k in range(1, len(hist)))
    assert st_.heads() == [hist[-1]]
    with pytest.raises(KeyError):
        st_.evolve("nope", at_seq=1, producer="deterministic")


# ---- episodes: late-result attribution + unresolved finalize + sentinels

def _late_result_events(tmp_path, *, is_error):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "run the long job"),
        rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                              "input": {"cmd": "long"}}], parent="u1"),
        rec_user("u2", "next thing please"),
        rec_tool_result("r1", "t1", "late output", parent="u2"),
    ])
    events = parse_file(p).events
    if is_error:
        # rebuild with error flag
        write_jsonl(p, [
            rec_user("u1", "run the long job"),
            rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                                  "input": {"cmd": "long"}}], parent="u1"),
            rec_user("u2", "next thing please"),
            {"type": "user", "uuid": "r1", "parentUuid": "u2", "sessionId": "s1",
             "timestamp": "2026-08-17T10:00:03.000Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "t1",
                  "content": "boom", "is_error": True}]}},
        ])
        events = parse_file(p).events
    return events


def test_late_result_attributed_to_prior_episode_not_next(tmp_path):
    events = _late_result_events(tmp_path, is_error=True)
    store = build_episodes(events)
    e1, e2 = store.heads(NodeKind.EPISODE)
    # prior episode: hindsight-resolved, error attributed THERE
    assert e1.producer == "hindsight:late-result"
    assert e1.structured["dangling_resolved"] == ["t1"]
    assert e1.structured["errors"] == 1
    assert e1.structured["status"] == "errors"
    # next episode untouched by the stray result
    assert e2.structured["errors"] == 0
    assert e2.structured["status"] in ("ok", "in-progress")
    assert all(pin < e2.covered_lo or pin >= e2.covered_lo for pin in e2.evidence_pins)
    assert e2.evidence_pins == []


def test_late_result_success_clears_interrupted(tmp_path):
    events = _late_result_events(tmp_path, is_error=False)
    store = build_episodes(events)
    e1 = store.heads(NodeKind.EPISODE)[0]
    assert e1.structured["dangling_resolved"] == ["t1"]
    assert e1.structured["status"] == "ok"                  # un-hindsighted


def test_finalize_with_dangling_is_unresolved_not_ok(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "start"),
        rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                              "input": {}}], parent="u1"),
    ])
    store = build_episodes(parse_file(p).events)
    ep = store.heads(NodeKind.EPISODE)[0]
    assert ep.structured["status"] == "unresolved"
    assert ep.structured["dangling_tools"] == ["Bash"]


def test_interrupt_sentinel_is_not_a_turn_or_episode(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "real ask"),
        rec_user("u2", "[Request interrupted by user for tool use]"),
        rec_assistant("a1", [{"type": "text", "text": "resuming"}], parent="u2"),
    ])
    events = parse_file(p).events
    s = fold(events)
    assert s.n_turns == 1
    assert s.latest_user_directive == "real ask"
    store = build_episodes(events)
    assert len(store.heads(NodeKind.EPISODE)) == 1


# ---- flush cadence

def test_flush_cadence_and_final_digest(tmp_path):
    p = tmp_path / "s.jsonl"
    recs = [rec_user("u1", "big task")]
    for i in range(30):  # 60 main-thread events
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "Grep", "input": {"q": i}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"res {i}"))
    write_jsonl(p, recs)
    events = parse_file(p).events
    store = build_episodes(events)
    ep = store.heads(NodeKind.EPISODE)[0]
    hist = store.history(ep.node_id)
    assert len(hist) >= 3                       # periodic evolves happened
    assert ep.structured["tools"] == {"Grep": 30}
    assert ep.structured["n_calls"] == 30       # final digest complete despite flushes
    # determinism: same input -> same version count
    store2 = build_episodes(parse_file(p).events)
    assert len(store2.history(store2.heads(NodeKind.EPISODE)[0].node_id)) == len(hist)


# ---- CLI

def test_cli_render_end_to_end(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=13, pathologies=["loop"])
    out = tmp_path / "page.html"
    assert cli_main(["render", str(p), "--out", str(out)]) == 0
    html = out.read_text()
    assert "state materialized at seq" in html and "Episodes" in html
    assert 'http-equiv="refresh"' not in html   # one-shot render: no auto-refresh


def test_cli_watch_single_cycle(tmp_path, monkeypatch):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [rec_user("u1", "watch me")])
    out = tmp_path / "w.html"

    import tracelab.cli as cli_mod
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        if calls["n"] == 1:
            with open(p, "a") as f:
                f.write(json.dumps(rec_assistant("a1", [{"type": "text",
                                                         "text": "hello"}])) + "\n")
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)
    assert cli_main(["watch", str(p), "--out", str(out), "--poll", "0.01"]) == 0
    html = out.read_text()
    assert 'http-equiv="refresh"' in html       # live page auto-refreshes
    assert "watch me" in html and "hello" in html  # second cycle saw appended event


# ---- FIDELITY mutation sensitivity (the shared-assumption symmetry breaker)

def _mutated_fidelity(monkeypatch, mutate):
    from tracelab.state import reducers
    orig = reducers.StateFolder.apply

    def patched(self, ev):
        mutate(self, ev, orig)
    monkeypatch.setattr(reducers.StateFolder, "apply", patched)
    return fidelity_bench.run(n_synth=4, n_real=0)


def test_fidelity_catches_dropped_tool_calls(monkeypatch):
    def mutate(self, ev, orig):
        if ev.kind.value == "tool_call" and ev.seq % 3 == 0:
            return  # silently drop every 3rd call
        orig(self, ev)
    res = _mutated_fidelity(monkeypatch, mutate)
    assert res["overall"] < 1.0
    assert any("n_tool_calls" in m or "per_tool" in m for m in res["mismatches"])


def test_fidelity_catches_missed_errors(monkeypatch):
    def mutate(self, ev, orig):
        if ev.kind.value == "tool_result" and ev.payload.is_error:
            ev = ev.model_copy(update={"payload": ev.payload.model_copy(
                update={"is_error": False})})
        orig(self, ev)
    res = _mutated_fidelity(monkeypatch, mutate)
    assert any("n_tool_errors" in m for m in res["mismatches"])


def test_fidelity_catches_usage_double_count(monkeypatch):
    def mutate(self, ev, orig):
        orig(self, ev)
        if ev.usage:  # count usage twice (regression of the M1 critical)
            t = self.state.tokens
            t.output += ev.usage.output_tokens
    res = _mutated_fidelity(monkeypatch, mutate)
    assert any("output_tokens" in m for m in res["mismatches"])


# ---- HTML edges

def test_render_empty_state_and_flags():
    html = render(RunState(session_id="s", source_path="x"), NodeStore())
    assert "(no goal captured yet)" in html and "no active anomalies" in html
    html_r = render(RunState(session_id="s", source_path="x"), NodeStore(), refresh=5)
    assert 'http-equiv="refresh"' in html_r
    html_t = render(RunState(session_id="s", source_path="x"), NodeStore(), trailing=17)
    assert "partial line pending" in html_t


def test_render_escapes_hostile_episode_content(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [rec_user("u1", "<script>alert(1)</script> task"),
                    rec_assistant("a1", [{"type": "text",
                                          "text": "<img onerror=x> done"}])])
    events = parse_file(p).events
    from tracelab.state.reducers import StateFolder
    f = StateFolder()
    f.fold(events)
    html = render(f.state, build_episodes(events))
    assert "<script>alert(1)</script>" not in html
    assert "<img onerror" not in html
