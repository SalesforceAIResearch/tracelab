"""COMPREHEND benchmark v0: can a reader answer live-monitoring questions better
from our compiled view than from the raw trace?

The field has no benchmark for human comprehension of live long runs (confirmed
independently by all three SOTA surveys) — this is v0 of that instrument, with an
LLM panel standing in as proxy reader (VCC-style). Questions are generated
DETERMINISTICALLY from ledger ground truth, so grading is objective (exact / substring
/ set-F1) — no LLM judge, no circularity through our own semantic layer.

Conditions per transcript (identical questions, identical answerer):
  raw     — tail of the raw JSONL (capped)
  flatlog — chronological plain-text event rendering (capped)
  view    — our compiled trace-model text (uncapped; it is small)

Metrics: accuracy per condition + prompt size per condition (the claim under test is
accuracy AND economy, not accuracy alone).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tracelab.derived.episodes import build_episodes
from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.envelope import Event, EventKind, Source
from tracelab.state.reducers import StateFolder
from tracelab.views.text_view import compile_text

RAW_CAP = 100_000       # chars of raw JSONL tail
FLAT_CAP = 100_000      # chars of flat log tail


# ------------------------------------------------------------ question builder

@dataclass
class Question:
    qid: str
    text: str
    kind: str           # exact | substring | set_f1
    answer: object      # ground truth


def build_questions(events: list[Event], state, store) -> list[Question]:
    qs: list[Question] = []
    if state.latest_user_directive:
        qs.append(Question("latest_ask",
                           "What was the user's most recent request? Quote or closely "
                           "paraphrase it.", "substring_of_truth",
                           state.latest_user_directive))
    qs.append(Question("n_turns", "How many user turns (real user messages, not tool "
                       "results) have there been? Answer with just an integer.",
                       "exact_int", state.n_turns))
    if state.per_tool:
        top = max(state.per_tool.items(), key=lambda kv: kv[1])
        qs.append(Question("top_tool", "Which tool has been called the most? Answer "
                           "with just the tool name.", "exact_str", top[0]))
    write_files = sorted({fp for fp in state.files_touched})
    if write_files:
        qs.append(Question("files", "List the file paths that were read or modified "
                           "with file tools (one per line, paths only).", "set_f1",
                           set(write_files)))
    last_err = next((e for e in reversed(events)
                     if e.kind == EventKind.TOOL_RESULT and e.payload.is_error
                     and e.agent_id is None), None)
    if last_err:
        qs.append(Question("last_error", "What was the most recent tool error? Quote "
                           "part of the error message.", "substring_of_truth",
                           (last_err.payload.text or "")[:300]))
    dangling = sorted({c.tool_name for c in state.pending_tools if c.tool_name})
    qs.append(Question("dangling", "Are any tool calls still awaiting their result? "
                       "Answer 'none' or list the tool names.", "set_f1_or_none",
                       set(dangling)))
    return qs


# ------------------------------------------------------------ conditions

def render_flatlog(events: list[Event]) -> str:
    lines = []
    for e in events:
        if e.agent_id is not None:
            continue
        t = (e.payload.text or "").replace("\n", " ")[:200]
        lines.append(f"[{e.seq}] {e.source.value}/{e.kind.value} "
                     f"{e.payload.tool_name or ''} {t}")
    return "\n".join(lines)


def build_conditions(path: Path, events, state, store) -> dict[str, str]:
    raw_txt = path.read_text(errors="replace")
    return {
        "raw": raw_txt[-RAW_CAP:],
        "flatlog": render_flatlog(events)[-FLAT_CAP:],
        "view": compile_text(state, store),
    }


# ------------------------------------------------------------ answering

ANSWER_SYSTEM = (
    "You are monitoring an AI coding agent's work session. Using ONLY the provided "
    "context, answer each numbered question. Reply with a JSON object mapping question "
    "ids to answers (strings; for list questions use a single string with items "
    "separated by '; '). No commentary, JSON only.")


def ask_panel(client, condition_name: str, context: str, qs: list[Question]) -> dict:
    qtext = "\n".join(f'{q.qid}: {q.text}' for q in qs)
    prompt = (f"CONTEXT ({condition_name}):\n```\n{context}\n```\n\n"
              f"QUESTIONS:\n{qtext}\n\nJSON answers:")
    res = client.complete(prompt, system=ANSWER_SYSTEM, max_tokens=4000)
    m = re.search(r"\{.*\}", res.text, re.DOTALL)
    answers = {}
    if m:
        try:
            # strict=False: models emit literal newlines inside JSON strings when
            # listing items — accept control chars rather than dropping the answer set
            answers = json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            answers = {}
    return {"answers": answers, "input_tokens": res.input_tokens,
            "cost_usd": res.cost_usd}


# ------------------------------------------------------------ grading

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def grade(q: Question, answer) -> float:
    if answer is None:
        return 0.0
    a = _norm(answer if isinstance(answer, str) else json.dumps(answer))
    if q.kind == "exact_int":
        m = re.search(r"-?\d+", a)
        return 1.0 if (m and int(m.group()) == q.answer) else 0.0
    if q.kind == "exact_str":
        return 1.0 if _norm(q.answer) in a else 0.0
    if q.kind == "substring_of_truth":
        truth = _norm(q.answer)
        # credit if a >=15-char chunk of the model's answer appears in the truth,
        # or the truth's head appears in the answer
        if truth[:40] and truth[:40] in a:
            return 1.0
        words = [w for w in a.split() if len(w) > 3]
        hits = sum(1 for w in words if w in truth)
        return 1.0 if (words and hits / len(words) >= 0.6) else 0.0
    if q.kind in ("set_f1", "set_f1_or_none"):
        truth: set = {str(x) for x in q.answer}
        if not truth:
            return 1.0 if ("none" in a or a == "") else 0.0
        got = {_norm(x) for x in re.split(r"[\n,;]+", a) if _norm(x)}
        got = {g for g in got if g != "none"}
        if not got:
            return 0.0
        matched_truth = sum(1 for t in truth
                            if any(_norm(t) in g or g in _norm(t) for g in got))
        matched_got = sum(1 for g in got
                          if any(_norm(t) in g or g in _norm(t) for t in truth))
        prec = matched_got / len(got)
        rec = matched_truth / len(truth)
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    raise ValueError(q.kind)


# ------------------------------------------------------------ runner

@dataclass
class CondScore:
    scores: list[float] = field(default_factory=list)
    input_tokens: int = 0
    cost_usd: float = 0.0
    per_q: dict = field(default_factory=dict)


def run(paths: list[Path], client, out: Path | None = None) -> dict:
    conds: dict[str, CondScore] = {c: CondScore() for c in ("raw", "flatlog", "view")}
    n_questions = 0
    for path in paths:
        events = parse_file(path).events
        folder = StateFolder(str(path), detectors=default_detectors())
        folder.fold(events)
        store = build_episodes(events)
        qs = build_questions(events, folder.state, store)
        n_questions += len(qs)
        contexts = build_conditions(path, events, folder.state, store)
        for cname, ctx in contexts.items():
            resp = ask_panel(client, cname, ctx, qs)
            cs = conds[cname]
            cs.input_tokens += resp["input_tokens"]
            cs.cost_usd += resp["cost_usd"]
            tr_scores = []
            for q in qs:
                sc = grade(q, resp["answers"].get(q.qid))
                cs.scores.append(sc)
                cs.per_q.setdefault(q.qid, []).append(sc)
                tr_scores.append(sc)
            cs.per_q.setdefault("_per_transcript", []).append(
                sum(tr_scores) / len(tr_scores) if tr_scores else 0.0)

    result = {"n_transcripts": len(paths), "n_questions": n_questions,
              "model": client.model, "conditions": {}}
    for cname, cs in conds.items():
        result["conditions"][cname] = {
            "accuracy": round(sum(cs.scores) / len(cs.scores), 4) if cs.scores else None,
            "input_tokens": cs.input_tokens,
            "cost_usd": round(cs.cost_usd, 4),
            "per_question": {k: (v if k == "_per_transcript" else round(sum(v) / len(v), 3)) for k, v in cs.per_q.items()},
        }
    if out:
        board = json.loads(out.read_text()) if out.exists() else {}
        board["COMPREHEND"] = result
        out.write_text(json.dumps(board, indent=2))
    return result
