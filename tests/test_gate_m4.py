"""Regression pins for the M4 gate findings (experiment-validity review)."""

import json
from pathlib import Path

from tracelab.derived.episodes import build_episodes
from tracelab.derived.nodes import NodeKind
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.llm.client import LLMResult
from tracelab.state.reducers import StateFolder, fold
from tracelab.views.text_view import compile_text
from tracelab.workbench.env import TASK_MAKERS, Workbench, make_fix_task
from tracelab.workbench.worker import TraceViewPolicy, run_task

from test_m4 import ScriptedSolver  # noqa: F401
from test_ledger import rec_assistant, rec_tool_result, rec_user, write_jsonl  # noqa: F401


# ---- CRITICAL: identical adversity across arms at identical call depth

def test_error_schedule_identical_across_arms():
    t = make_fix_task(11)
    schedules = []
    for _ in range(3):  # three "arms" instantiating the same env seed
        env = Workbench(t, seed=11)
        outs = [env.call("list_files", {}).is_error for _ in range(40)]
        schedules.append(outs)
    assert schedules[0] == schedules[1] == schedules[2]


def test_error_schedule_independent_of_noise_consumption():
    t = make_fix_task(11)
    env_a = Workbench(t, seed=11)
    env_b = Workbench(t, seed=11)
    # arm B consumes extra noise via different (valid) call mix; error pattern by
    # CALL INDEX must stay identical to arm A's
    pattern_a = [env_a.call("list_files", {}).text.startswith("TransientError")
                 for _ in range(20)]
    pattern_b = []
    for i in range(20):
        out = env_b.call("search", {"pattern": "timeout"})  # heavier rng consumer
        pattern_b.append(out.text.startswith("TransientError"))
    assert pattern_a == pattern_b


def test_unknown_tool_never_masked_as_transient():
    t = make_fix_task(2)
    env = Workbench(t, seed=2, error_rate=1.0)   # every valid call would error
    out = env.call("hallucinated_tool", {})
    assert out.text.startswith("UnknownTool")     # validation precedes injection


# ---- CRITICAL: retries are not repeats

class RetryScriptedSolver(ScriptedSolver):
    pass


def test_retry_counted_separately_from_repeat(tmp_path):
    task = make_fix_task(21)
    # error_rate high enough that retries certainly occur
    r = run_task(task, "full", ScriptedSolver(task), trace_dir=tmp_path, seed=21,
                 max_steps=80)
    assert r.done
    # scripted solver never truly repeats — any same-sig re-issue follows an error
    assert r.repeated_actions == 0
    assert r.retries == r.errors_injected  # every injected error retried exactly once


# ---- CRITICAL: parse failures are recorded, costed, and counted separately

class FlakyProtocolSolver(ScriptedSolver):
    def __init__(self, task):
        super().__init__(task)
        self._sent_garbage = False

    def complete(self, prompt, *, system=None, max_tokens=0):
        if not self._sent_garbage:
            self._sent_garbage = True
            return LLMResult(text="I think I should look at the files first.",
                             input_tokens=100, output_tokens=10, cost_usd=0.001,
                             model="flaky")
        return super().complete(prompt, system=system, max_tokens=max_tokens)


def test_parse_failure_recorded_and_split(tmp_path):
    task = make_fix_task(8)
    r = run_task(task, "full", FlakyProtocolSolver(task), trace_dir=tmp_path, seed=8,
                 max_steps=80)
    assert r.parse_failure_steps == 1
    assert r.tool_steps == r.steps - 1
    assert r.done and r.success == 1.0
    events = parse_file(Path(r.trace_path)).events
    protocol_errors = [e for e in events
                       if e.payload.is_error and "ProtocolError" in (e.payload.text or "")]
    assert protocol_errors, "parse-failure turn must appear in the recorded trace"
    # its LLM cost must be in the trace too
    s = fold(events)
    assert s.tokens.output >= 10


def test_consecutive_abort_resets_on_success(tmp_path):
    class AlternatingGarbage(ScriptedSolver):
        def __init__(self, task):
            super().__init__(task)
            self.n = 0

        def complete(self, prompt, *, system=None, max_tokens=0):
            self.n += 1
            if self.n % 2 == 1 and self.n < 10:   # garbage, valid, garbage, valid…
                return LLMResult(text="hmm", input_tokens=10, output_tokens=1,
                                 cost_usd=0.0, model="alt")
            return super().complete(prompt, system=system, max_tokens=max_tokens)

    task = make_fix_task(6)
    r = run_task(task, "full", AlternatingGarbage(task), trace_dir=tmp_path, seed=6,
                 max_steps=80)
    # 5 scattered failures but never 3 consecutive: run must NOT abort
    assert r.parse_failure_steps >= 3 and r.done


# ---- MAJOR: the trace view actually carries workbench file activity now

def test_view_carries_workbench_files_and_detectors(tmp_path):
    task = make_fix_task(31)
    r = run_task(task, "trace", ScriptedSolver(task), trace_dir=tmp_path, seed=31,
                 max_steps=80)
    events = parse_file(Path(r.trace_path)).events
    folder = StateFolder(detectors=default_detectors())
    folder.fold(events)
    assert folder.state.files_touched, "workbench read_file/write_file must register"
    assert any(p.startswith("conf/") for p in folder.state.files_touched)
    view = compile_text(folder.state, build_episodes(events))
    assert "FILES TOUCHED: " in view and "conf/" in view


def test_recorder_uses_actual_model_name(tmp_path):
    task = make_fix_task(4)
    r = run_task(task, "full", ScriptedSolver(task), trace_dir=tmp_path, seed=4)
    events = parse_file(Path(r.trace_path)).events
    models = {e.usage.model for e in events if e.usage}
    assert models == {"scripted"}


# ---- fact pinning (M4 v1 lesson: views must carry content, not references)

def test_facts_extracted_and_in_view(tmp_path):
    task = TASK_MAKERS["scatter"](5)
    r = run_task(task, "full", ScriptedSolver(task), trace_dir=tmp_path, seed=5,
                 max_steps=80)
    events = parse_file(Path(r.trace_path)).events
    folder = StateFolder()
    folder.fold(events)
    metric_facts = {k: v for k, v in folder.state.facts.items()
                    if "metric_" in k}
    assert len(metric_facts) >= 5, "scatter values must be pinned as facts"
    view = compile_text(folder.state, build_episodes(events))
    assert "KEY FACTS" in view
    k, v = next(iter(metric_facts.items()))
    assert f"{k} = {v}" in view and "metric_" in view


def test_repeated_key_facts_all_retained(tmp_path):
    # CONTINUE v4 lesson: accumulator sequences must not collapse newest-wins
    recs = [rec_user("u1", "chain")]
    for i in range(6):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"n{i}"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"pad\ndelta = {10 + i}\npad"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    folder = StateFolder()
    folder.fold(parse_file(p).events)
    delta_vals = {v for k, v in folder.state.facts.items() if "delta" in k}
    assert delta_vals == {str(10 + i) for i in range(6)}, folder.state.facts


def test_duplicate_valued_facts_from_different_files_both_retained(tmp_path):
    # CONTINUE v6 lesson: occurrence identity, not (key,value) identity
    recs = [rec_user("u1", "chain")]
    for i, val in enumerate([41, 41]):   # SAME value from two different files
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"delta = {val}"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    folder = StateFolder()
    folder.fold(parse_file(p).events)
    delta_keys = [k for k in folder.state.facts if k.endswith(":delta")]
    assert len(delta_keys) == 2, folder.state.facts


def test_numeric_fact_groups_get_deterministic_aggregates(tmp_path):
    recs = [rec_user("u1", "chain")]
    for i, val in enumerate([10, 20, 30, 40]):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"delta = {val}"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    folder = StateFolder()
    folder.fold(parse_file(p).events)
    view = compile_text(folder.state, build_episodes(parse_file(p).events))
    assert "[aggregate] delta: 4 values total (0 folded out of view), sum = 100" in view


def test_eviction_folds_numerics_into_aggregate(tmp_path):
    # CONTINUE v10@120 lesson: bounded facts must not lose accumulator values
    recs = [rec_user("u1", "chain")]
    n = 150  # > MAX_FACTS
    for i in range(n):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"delta = {i + 1}"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    folder = StateFolder()
    folder.fold(parse_file(p).events)
    view = compile_text(folder.state, build_episodes(parse_file(p).events))
    want = n * (n + 1) // 2
    assert f"sum = {want}" in view, "aggregate must be lossless across eviction"
    assert "folded out of view" in view


def test_reread_after_eviction_does_not_double_count(tmp_path):
    recs = [rec_user("u1", "chain")]
    n = 150
    for i in range(n):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"delta = {i + 1}"))
    # re-read file 0 (its fact was evicted+folded long ago)
    recs.append(rec_assistant("aR", [{"type": "tool_use", "id": "tR",
                                      "name": "read_file",
                                      "input": {"path": "node/f0.txt"}}]))
    recs.append(rec_tool_result("rR", "tR", "delta = 1"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    folder = StateFolder()
    folder.fold(parse_file(p).events)
    view = compile_text(folder.state, build_episodes(parse_file(p).events))
    want = n * (n + 1) // 2
    assert f"sum = {want}" in view, "re-read must not inflate the aggregate"


# ---- cache-mode turn builders (v13 lesson: only append-only structure hits)

def test_turns_full_is_append_only_prefix():
    from tracelab.workbench.worker import StepRecord, turns_full
    h = [StepRecord(i, {"tool": "read_file", "args": {"path": f"f{i}"}}, f"obs{i}", False)
         for i in range(4)]
    t3, t4 = turns_full("T", h[:3]), turns_full("T", h)
    assert t4[:len(t3)] == t3, "prior turns must be byte-identical (cache prefix)"
    assert t4[0]["role"] == "user" and t4[-1]["role"] == "user"
    roles = [t["role"] for t in t4]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


def test_trace_turns_reset_anchor_on_refresh(tmp_path):
    from tracelab.workbench.worker import StepRecord, TraceViewTurnsPolicy
    task = make_fix_task(12)
    r = run_task(task, "full", ScriptedSolver(task), trace_dir=tmp_path, seed=12)
    pol = TraceViewTurnsPolicy(Path(r.trace_path), refresh_every=3)
    h = [StepRecord(i, {"tool": "read_file", "args": {"path": f"f{i}"}}, f"obs{i}", False)
         for i in range(8)]
    t_a = pol.turns("T", h[:4])          # triggers a refresh, anchor=4
    assert pol.refreshes == 1 and len(t_a) == 1
    t_b = pol.turns("T", h[:5])          # within window: append-only growth
    assert t_b[0] == t_a[0] and len(t_b) == 3


# ---- v14: semantic extraction (the budget dial in the parser)

def test_prose_chain_invisible_to_regex_extractor(tmp_path):
    from tracelab.state.reducers import FACT_RE
    from tracelab.workbench.env import make_prose_chain_task
    task = make_prose_chain_task(3, n_links=10)
    for body in task.files.values():
        assert not any(FACT_RE.match(l) for l in body.splitlines()), body


def test_prose_chain_oracle_solves_all_policies(tmp_path):
    from tracelab.workbench.env import make_prose_chain_task
    task = make_prose_chain_task(5, n_links=12)
    for pol in ("full", "trace"):
        r = run_task(task, pol, ScriptedSolver(task), trace_dir=tmp_path,
                     seed=5, max_steps=40)
        assert r.success == 1.0, (pol, r.subcheck_notes)


class FakeExtractClient:
    """Answers extraction prompts deterministically: finds prose deltas by regex."""
    model = "fake-extract"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system=None, max_tokens=0):
        import re as _re
        self.calls += 1
        rows = []
        for m in _re.finditer(r"\[(\d+)\] (.*?)(?=\n\n\[|\Z)", prompt, _re.DOTALL):
            facts = []
            d = _re.search(r"(?:advanced by|adjustment of|contributes) (\d+)",
                           m.group(2))
            if d:
                facts.append({"key": "delta", "value": d.group(1)})
            rows.append({"i": int(m.group(1)), "facts": facts})
        return LLMResult(text=json.dumps(rows), input_tokens=10, output_tokens=10,
                         cost_usd=0.001, model="fake-extract")


def test_semantic_extractor_memoizes_and_feeds_aggregates(tmp_path):
    from tracelab.llm.extract import SemanticExtractor
    from tracelab.views.text_view import compile_text
    from tracelab.derived.episodes import build_episodes
    recs = [rec_user("u1", "trail")]
    vals = [10, 20, 30]
    for i, v in enumerate(vals):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(
            f"r{i}", f"t{i}", f"pad text. An adjustment of {v} was applied. pad"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    events = parse_file(p).events
    fake = FakeExtractClient()
    ex = SemanticExtractor(fake)
    # two refresh cycles over the same events: one extraction call, identical facts
    for _ in range(2):
        folder = StateFolder()
        folder.fold(events)
        for source, k, v in ex.harvest(events):
            folder.note_fact(source, k, v)
    assert fake.calls == 1, "memoization must prevent re-extraction on refolds"
    delta_facts = {k: v for k, v in folder.state.facts.items() if "delta" in k}
    assert len(delta_facts) == 3, folder.state.facts
    view = compile_text(folder.state, build_episodes(events))
    assert "[aggregate] delta: 3 values total (0 folded out of view), sum = 60" in view


def test_note_fact_eviction_lossless_via_injection():
    folder = StateFolder()
    n = 200
    for i in range(n):
        folder.note_fact(f"node/f{i}.txt", "delta", str(i + 1))
    total = sum(int(v) for k, v in folder.state.facts.items() if "delta" in k)
    agg = folder.state.fact_aggregates.get("delta", {"sum": 0})
    assert total + agg["sum"] == n * (n + 1) // 2


def test_extractor_carries_known_keys_across_batches(tmp_path):
    # v14 lesson (requirement 8): same-role facts fragmented across synonym keys
    # ("contribution"/"adjustment"/"advancement") -> aggregates split. The extractor
    # must anchor its schema: keys established early are offered to later batches.
    from tracelab.llm.extract import SemanticExtractor

    class KeyEchoClient:
        model = "fake"

        def __init__(self):
            self.prompts = []

        def complete(self, prompt, *, system=None, max_tokens=0):
            self.prompts.append(prompt)
            import re as _re
            n = len(_re.findall(r"^\[\d+\]", prompt, _re.MULTILINE))
            rows = [{"i": i, "facts": [{"key": "delta", "value": "1"}]}
                    for i in range(n)]
            return LLMResult(text=json.dumps(rows), input_tokens=1, output_tokens=1,
                             cost_usd=0.0, model="fake")

    recs = [rec_user("u1", "trail")]
    for i in range(4):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"advanced by {i + 1} units"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    events = parse_file(p).events
    fake = KeyEchoClient()
    ex = SemanticExtractor(fake, batch_size=2)   # forces two batches
    ex.harvest(events)
    assert len(fake.prompts) == 2
    assert "KNOWN KEYS" not in fake.prompts[0]
    assert "KNOWN KEYS" in fake.prompts[1] and "delta" in fake.prompts[1]


def test_extractor_bisects_refused_batches(tmp_path):
    # v15 lesson: batches of individually-benign machine noise can trip the
    # extractor model's refusal classifier AS A BATCH; bisection recovers all facts.
    from tracelab.llm.extract import SemanticExtractor

    class BatchRefusingClient:
        model = "fake"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, max_tokens=0):
            import re as _re
            self.calls += 1
            ids = _re.findall(r"^\[(\d+)\]", prompt, _re.MULTILINE)
            if len(ids) > 2:   # "refuse" any batch larger than 2
                return LLMResult(text="", input_tokens=1, output_tokens=1,
                                 cost_usd=0.0, model="fake", stop_reason="refusal")
            rows = [{"i": int(i), "facts": [{"key": "delta", "value": "7"}]}
                    for i in ids]
            return LLMResult(text=json.dumps(rows), input_tokens=1, output_tokens=1,
                             cost_usd=0.0, model="fake")

    recs = [rec_user("u1", "trail")]
    for i in range(8):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"advanced by 7 units ({i})"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    events = parse_file(p).events
    fake = BatchRefusingClient()
    ex = SemanticExtractor(fake, batch_size=8)
    facts = ex.harvest(events)
    assert len([f for f in facts if f[1] == "delta"]) == 8, "bisection must recover all"
    assert ex.refusals >= 1 and fake.calls > 2


def test_extractor_falls_back_on_leaf_refusal(tmp_path):
    # v16 lesson: bisection alone leaves holes when SINGLE observations refuse;
    # the cascade's last rung is a different model.
    from tracelab.llm.extract import SemanticExtractor

    class AlwaysRefusing:
        model = "refuser"

        def complete(self, prompt, *, system=None, max_tokens=0):
            return LLMResult(text="", input_tokens=1, output_tokens=1,
                             cost_usd=0.0, model="refuser", stop_reason="refusal")

    class Answering:
        model = "fallback"

        def complete(self, prompt, *, system=None, max_tokens=0):
            import re as _re
            ids = _re.findall(r"^\[(\d+)\]", prompt, _re.MULTILINE)
            rows = [{"i": int(i), "facts": [{"key": "delta", "value": "3"}]}
                    for i in ids]
            return LLMResult(text=json.dumps(rows), input_tokens=1, output_tokens=1,
                             cost_usd=0.0, model="fallback")

    recs = [rec_user("u1", "trail")]
    for i in range(4):
        recs.append(rec_assistant(f"a{i}", [{"type": "tool_use", "id": f"t{i}",
                                             "name": "read_file",
                                             "input": {"path": f"node/f{i}.txt"}}]))
        recs.append(rec_tool_result(f"r{i}", f"t{i}", f"advanced by 3 units ({i})"))
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    events = parse_file(p).events
    ex = SemanticExtractor(AlwaysRefusing(), batch_size=4,
                           fallback_client=Answering())
    facts = ex.harvest(events)
    assert len([f for f in facts if f[1] == "delta"]) == 4
    assert ex.fallbacks == 4   # every leaf went to the fallback rung


def test_extracted_facts_validated_verbatim_against_source(tmp_path):
    # v17 lesson (requirement 10): batch-composition nondeterminism can invent a
    # phantom fact; deterministic verbatim validation at injection kills it.
    from tracelab.llm.extract import SemanticExtractor

    class PhantomClient:
        model = "fake"

        def complete(self, prompt, *, system=None, max_tokens=0):
            rows = [{"i": 0, "facts": [{"key": "delta", "value": "17"},
                                       {"key": "delta", "value": "41"}]}]
            return LLMResult(text=json.dumps(rows), input_tokens=1, output_tokens=1,
                             cost_usd=0.0, model="fake")

    recs = [rec_user("u1", "trail"),
            rec_assistant("a0", [{"type": "tool_use", "id": "t0",
                                  "name": "read_file",
                                  "input": {"path": "node/f0.txt"}}]),
            rec_tool_result("r0", "t0", "the counter advanced by 17 units")]
    p = tmp_path / "s.jsonl"
    write_jsonl(p, recs)
    ex = SemanticExtractor(PhantomClient())
    facts = ex.harvest(parse_file(p).events)
    assert [v for _, k, v in facts if k == "delta"] == ["17"]
    assert ex.rejected == 1


def test_summary_policy_baseline_shape(tmp_path):
    # the compaction-style baseline arm: rolling LLM summary at the trace arm's
    # cadence; summarizer cost tracked as curation cost
    from tracelab.workbench.worker import StepRecord, SummaryViewPolicy

    class FakeSummarizer:
        model = "fake-sum"

        def __init__(self):
            self.calls = 0

        def complete(self, prompt, *, system=None, max_tokens=0):
            self.calls += 1
            return LLMResult(text=f"SUMMARY v{self.calls}: totals so far preserved.",
                             input_tokens=50, output_tokens=20, cost_usd=0.01,
                             model="fake-sum")

    fake = FakeSummarizer()
    pol = SummaryViewPolicy(fake, keep_last=3, refresh_every=5)
    hist = [StepRecord(i, {"tool": "read_file", "args": {"path": f"f{i}"}},
                       f"delta = {i}", False) for i in range(12)]
    ctx = pol(  # noqa: F841 - trigger refresh path
        "T", hist)
    assert fake.calls >= 1 and "SUMMARY v" in ctx
    assert ctx.count("[step") >= 6                      # last-3 raw = 3 steps x 2 lines
    assert pol.cost_usd > 0
