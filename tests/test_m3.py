"""M3 tests: budget ledger, question builder, grading, COMPREHEND end-to-end with a
fake provider (no network), text view compilation."""

import json
from pathlib import Path

import pytest

from tracelab.bench.comprehend_bench import (
    Question, build_conditions, build_questions, grade, run as run_comprehend,
)
from tracelab.derived.episodes import build_episodes
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.llm.client import BudgetExceeded, SpendLedger
from tracelab.state.reducers import StateFolder
from tracelab.views.text_view import compile_text

from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


# ------------------------------------------------------------- budget ledger

def test_spend_ledger_records_and_caps(tmp_path):
    led = SpendLedger(tmp_path / "spend.json", cap_usd=1.0)
    led.check()
    led.record("test", "m", 0.4, 100, 50)
    led.record("test", "m", 0.4, 100, 50)
    assert abs(led.total() - 0.8) < 1e-9
    led.check()                       # still under
    led.record("test", "m", 0.4, 100, 50)
    with pytest.raises(BudgetExceeded):
        led.check()                   # over cap: hard stop
    data = json.loads((tmp_path / "spend.json").read_text())
    assert len(data["entries"]) == 3


# ------------------------------------------------------------- questions + grading

@pytest.fixture
def rich_session(tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, [
        rec_user("u1", "please refactor the auth module"),
        rec_assistant("a1", [{"type": "tool_use", "id": "t1", "name": "Read",
                              "input": {"file_path": "/src/auth.py"}}], parent="u1"),
        rec_tool_result("r1", "t1", "contents", parent="a1"),
        rec_assistant("a2", [{"type": "tool_use", "id": "t2", "name": "Edit",
                              "input": {"file_path": "/src/auth.py"}}], parent="r1"),
        {"type": "user", "uuid": "r2", "parentUuid": "a2", "sessionId": "s1",
         "timestamp": "2026-08-17T10:00:04.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "t2",
              "content": "Error: old_string not found in file", "is_error": True}]}},
        rec_user("u2", "ok just add a docstring instead"),
        rec_assistant("a3", [{"type": "tool_use", "id": "t3", "name": "Bash",
                              "input": {"cmd": "sleep"}}], parent="u2"),
    ])
    events = parse_file(p).events
    folder = StateFolder(str(p), detectors=default_detectors())
    folder.fold(events)
    store = build_episodes(events)
    return p, events, folder.state, store


def test_question_builder_ground_truth(rich_session):
    _, events, state, store = rich_session
    qs = {q.qid: q for q in build_questions(events, state, store)}
    assert qs["latest_ask"].answer == "ok just add a docstring instead"
    assert qs["n_turns"].answer == 2
    assert qs["files"].answer == {"/src/auth.py"}
    assert "old_string not found" in qs["last_error"].answer
    assert qs["dangling"].answer == {"Bash"}


def test_grading_semantics():
    assert grade(Question("q", "", "exact_int", 7), "there were 7 turns") == 1.0
    assert grade(Question("q", "", "exact_int", 7), "8") == 0.0
    assert grade(Question("q", "", "exact_str", "Bash"), "the Bash tool") == 1.0
    assert grade(Question("q", "", "substring_of_truth",
                          "Error: old_string not found in file"),
                 "Error: old_string not found") == 1.0
    assert grade(Question("q", "", "set_f1", {"/a.py", "/b.py"}),
                 "/a.py\n/b.py") == 1.0
    assert 0 < grade(Question("q", "", "set_f1", {"/a.py", "/b.py"}), "/a.py") < 1.0
    assert grade(Question("q", "", "set_f1_or_none", set()), "none") == 1.0
    assert grade(Question("q", "", "set_f1_or_none", {"Bash"}), "none") == 0.0
    assert grade(Question("q", "", "exact_int", 7), None) == 0.0


# ------------------------------------------------------------- end-to-end (fake LLM)

class FakeClient:
    """Answers by cheating differently per condition: perfect from 'view', partial
    from 'flatlog', empty from 'raw' — verifies the bench discriminates conditions."""
    model = "fake"

    def __init__(self, truth_by_qid):
        self.truth = truth_by_qid

    def complete(self, prompt, *, system=None, max_tokens=0):
        from tracelab.llm.client import LLMResult
        if "CONTEXT (view)" in prompt:
            answers = {qid: self._fmt(v) for qid, v in self.truth.items()}
        elif "CONTEXT (flatlog)" in prompt:
            keep = list(self.truth)[: max(1, len(self.truth) // 2)]
            answers = {qid: self._fmt(self.truth[qid]) for qid in keep}
        else:
            answers = {}
        return LLMResult(text=json.dumps(answers), input_tokens=len(prompt) // 4,
                         output_tokens=50, cost_usd=0.0, model="fake")

    @staticmethod
    def _fmt(v):
        if isinstance(v, set):
            return "\n".join(sorted(v)) if v else "none"
        return str(v)


def test_comprehend_end_to_end_discriminates(rich_session, tmp_path):
    p, events, state, store = rich_session
    qs = build_questions(events, state, store)
    client = FakeClient({q.qid: q.answer for q in qs})
    out = tmp_path / "scoreboard.json"
    res = run_comprehend([p], client, out=out)
    acc = {c: v["accuracy"] for c, v in res["conditions"].items()}
    assert acc["view"] == 1.0
    assert 0.0 < acc["flatlog"] < 1.0
    assert acc["raw"] == 0.0
    board = json.loads(out.read_text())
    assert board["COMPREHEND"]["n_questions"] == len(qs)
    # view context must be dramatically smaller than raw
    ctx = build_conditions(p, events, state, store)
    assert len(ctx["view"]) < len(ctx["raw"])


# ------------------------------------------------------------- text view

def test_text_view_contains_model_essentials(tmp_path):
    p = tmp_path / "s.jsonl"
    generate(p, seed=17, pathologies=["error_streak"])
    events = parse_file(p).events
    folder = StateFolder(str(p), detectors=default_detectors())
    folder.fold(events)
    store = build_episodes(events)
    txt = compile_text(folder.state, store)
    assert "GOAL:" in txt and "## Episodes" in txt
    assert "COUNTERS:" in txt and "TOP TOOLS:" in txt
    assert "error" in txt.lower()
    assert len(txt) < 20_000            # stays compact by construction
