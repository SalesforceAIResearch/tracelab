"""M2 tests: node store lifecycle, staleness propagation, episodes (incl. hindsight),
HTML render, FIDELITY bench."""

import pytest

from tracelab.bench.fidelity_bench import run as run_fidelity
from tracelab.derived.episodes import EpisodeBuilder, build_episodes
from tracelab.derived.nodes import DerivedNode, NodeKind, NodeStore, Validity
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.state.reducers import StateFolder
from tracelab.views.human_html import render

from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


# ------------------------------------------------------------- node store

def _node(nid, version=1, deps=None, supersedes=None):
    return DerivedNode(node_id=nid, version=version, kind=NodeKind.FACT,
                       covered_lo=0, covered_hi=1, created_at_seq=1,
                       producer="deterministic", dependencies=deps or [],
                       supersedes=supersedes)


def test_versioning_and_supersede():
    st = NodeStore()
    a1 = st.add(_node("a"))
    a2 = st.evolve("a", at_seq=5, producer="deterministic", content="v2")
    assert st.validity(a1.ref) == Validity.SUPERSEDED
    assert st.validity(a2.ref) == Validity.CURRENT
    assert st.head("a").version == 2
    assert st.history("a")[0].content == ""          # old version preserved verbatim
    with pytest.raises(ValueError):
        st.add(_node("a", version=5))                 # version gap rejected


def test_staleness_propagates_through_dependency_closure():
    st = NodeStore()
    a = st.add(_node("a"))
    b = st.add(_node("b", deps=[a.ref]))
    c = st.add(_node("c", deps=[b.ref]))
    d = st.add(_node("d"))                            # unrelated
    touched = st.invalidate(a.ref, at_seq=9)
    assert st.validity(a.ref) == Validity.INVALIDATED
    assert st.validity(b.ref) == Validity.SUSPECTED_STALE
    assert st.validity(c.ref) == Validity.SUSPECTED_STALE   # transitive
    assert st.validity(d.ref) == Validity.CURRENT
    assert set(touched) == {b.ref, c.ref}
    # audit trail records every transition with a seq
    assert ("a-0001@1" if False else a.ref, "invalidated", 9) in st.audit


def test_superseded_nodes_do_not_go_stale():
    st = NodeStore()
    a1 = st.add(_node("a"))
    st.evolve("a", at_seq=3, producer="deterministic")
    b = st.add(_node("b", deps=[a1.ref]))
    st.invalidate(a1.ref, at_seq=9)
    assert st.validity(a1.ref) == Validity.INVALIDATED
    assert st.validity(b.ref) == Validity.SUSPECTED_STALE
    # head of a chain stays current (it did not depend on itself)
    assert st.validity(st.head("a").ref) == Validity.CURRENT


# ------------------------------------------------------------- episodes

@pytest.fixture
def two_turn_events(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "first task: check the tests"),
        rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                              "input": {"cmd": "pytest"}}], parent="u1"),
        rec_tool_result("r1", "t1", "3 passed", parent="a1"),
        rec_assistant("a2", [{"type": "text", "text": "All tests pass."}], parent="r1"),
        rec_user("u2", "second task: update the docs"),
        rec_assistant("a3", [{"type": "tool_use", "id": "t2", "name": "Edit",
                              "input": {"file_path": "/d.md"}}], parent="u2"),
        rec_tool_result("r2", "t2", "ok", parent="a3"),
    ])
    return parse_file(p).events


def test_episode_segmentation_and_digest(two_turn_events):
    store = build_episodes(two_turn_events)
    eps = store.heads(NodeKind.EPISODE)
    assert len(eps) == 2
    e1, e2 = eps
    assert e1.structured["ask"].startswith("first task")
    assert e1.structured["status"] == "ok"
    assert e1.structured["tools"] == {"Bash": 1}
    assert "All tests pass." in (e1.structured["conclusion"] or "")
    assert e2.structured["ask"].startswith("second task")
    assert e2.covered_lo > e1.covered_hi


def test_hindsight_interrupted_reparse(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "run the long job"),
        rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Bash",
                              "input": {"cmd": "sleep 999"}}], parent="u1"),
        # user interrupts: t1 never returns
        rec_user("u2", "forget it, do something else"),
        rec_assistant("a2", [{"type": "text", "text": "ok"}], parent="u2"),
    ])
    events = parse_file(p).events
    store = build_episodes(events)
    eps = store.heads(NodeKind.EPISODE)
    first = eps[0]
    assert first.structured["status"] == "interrupted"          # hindsight applied
    assert first.producer == "hindsight:interrupted"
    assert first.structured["dangling_tools"] == ["Bash"]
    hist = store.history(first.node_id)
    assert len(hist) >= 2                                        # pre-hindsight preserved
    pre = hist[-2]
    assert pre.structured["status"] in ("ok", "errors", "in-progress")
    # exact lifecycle: pre-hindsight version is SUPERSEDED, never re-marked stale
    assert store.validity(pre.ref) == Validity.SUPERSEDED
    terminal_transitions = [t for t in store.audit if t[0] == pre.ref and t[1] != "created"]
    assert terminal_transitions == [(pre.ref, "superseded", first.created_at_seq)]


def test_incremental_feed_equals_batch(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=5, pathologies=["error_streak"])
    events = parse_file(p).events
    batch = build_episodes(events)
    inc = EpisodeBuilder()
    for ev in events:
        inc.feed(ev)
    inc.finalize(events[-1].seq + 1)
    b_eps = [(n.covered_lo, n.covered_hi, n.structured) for n in batch.heads(NodeKind.EPISODE)]
    i_eps = [(n.covered_lo, n.covered_hi, n.structured) for n in inc.store.heads(NodeKind.EPISODE)]
    assert b_eps == i_eps


def test_sidechain_events_do_not_open_episodes(tmp_path):
    p = tmp_path / "s.jsonl"
    side = rec_user("su", "sidechain prompt", isSidechain=True)
    write_jsonl(p, [rec_user("u1", "main"), side,
                    rec_assistant("a1", [{"type": "text", "text": "done"}])])
    store = build_episodes(parse_file(p).events)
    eps = store.heads(NodeKind.EPISODE)
    assert len(eps) == 1 and eps[0].structured["ask"] == "main"


# ------------------------------------------------------------- HTML view

def test_render_contains_the_load_bearing_pieces(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=9, pathologies=["error_streak"])
    events = parse_file(p).events
    folder = StateFolder(str(p), detectors=default_detectors())
    folder.fold(events)
    store = build_episodes(events)
    html = render(folder.state, store, malformed=0)
    assert "Synthetic long-horizon task" in html          # goal
    assert "Episodes" in html and "drill down" in html
    assert "state materialized at seq" in html            # watermark honesty
    assert "error_streak" in html or "no active anomalies" in html
    assert "provenance" in html                           # version visibility
    # escaping: no raw < from payloads leaking into markup unescaped is hard to
    # assert generally; at minimum the page parses as utf-8 and is non-trivial
    assert len(html) > 3000


# ------------------------------------------------------------- FIDELITY

def test_fidelity_synthetic_perfect_and_real_floor():
    res = run_fidelity(n_synth=8, n_real=3)
    # synthetic traces are fully regular: any mismatch is a reducer/adapter bug
    synth_mismatches = [m for m in res["mismatches"] if m.startswith("synth")]
    assert synth_mismatches == [], synth_mismatches
    assert res["overall"] >= 0.9, res
