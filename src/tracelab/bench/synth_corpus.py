"""COMPREHEND v2 corpus: realistic synthetic Claude Code sessions, seeded.

The corpus is CODE — `build_corpus(root)` regenerates every transcript byte-for-byte
from fixed seeds, so releasing this module releases the corpus with zero privacy
surface, and any reader can audit every question against every line.

Each session also returns its ground truth computed BY THE GENERATOR, independently
of the ledger/adapter: this lets the benchmark cross-check the adapter-derived truth
against generator truth (answering the "the system defines its own ground truth"
circularity for this corpus).

Realism knobs per session (all seeded): 6-24 user turns; 40-400 tool calls across
Read/Grep/Bash/Edit/Write/WebSearch; project-specific file pools; transient and
not-found errors; interrupted turns; a sometimes-dangling final call; meta records;
multi-block assistant messages sharing one usage object (exercising usage dedup);
result payloads padded to megabyte-scale sessions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from tracelab.ledger.adapters.synthetic import _Writer

PROJECTS = ["billing-api", "search-svc", "etl-jobs", "web-app", "infra-tf",
            "ml-pipeline", "auth-gw", "docs-site"]
DIRS = ["src", "lib", "tests", "config", "scripts", "internal"]
STEMS = ["handler", "router", "models", "utils", "client", "worker", "schema",
         "parser", "cache", "queue", "metrics", "auth"]
EXTS = [".py", ".ts", ".go", ".yaml"]
ASKS = ["fix the failing test in {f}", "add retry logic to {f}",
        "refactor {f} to remove the global state", "why is {f} slow? profile it",
        "add input validation to {f}", "write unit tests for {f}",
        "trace the null pointer coming from {f}", "bump deps and fix breakage in {f}",
        "document the public functions of {f}", "make {f} idempotent"]
WORDS = ("config legacy vendor handler widget module service adapter registry "
         "pipeline buffer schema latency retry commit branch deploy").split()


@dataclass
class SessionTruth:
    """Ground truth computed by the generator itself — adapter-independent."""
    n_turns: int = 0
    per_tool: dict = field(default_factory=dict)
    files_touched: set = field(default_factory=set)
    latest_ask: str = ""
    last_error_text: str = ""
    dangling_tools: set = field(default_factory=set)


def _pad(rng: random.Random, lo: int, hi: int) -> str:
    n = rng.randint(lo, hi)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def build_session(path: Path, seed: int) -> SessionTruth:
    rng = random.Random(seed)
    w = _Writer(path, seed)
    t = SessionTruth()
    project = rng.choice(PROJECTS)
    files = [f"{project}/{rng.choice(DIRS)}/{rng.choice(STEMS)}{rng.choice(EXTS)}"
             for _ in range(rng.randint(8, 24))]

    n_turns = rng.randint(6, 24)
    for turn in range(n_turns):
        ask = rng.choice(ASKS).format(f=rng.choice(files))
        w.user_msg(ask)
        t.n_turns += 1
        t.latest_ask = ask
        interrupted = rng.random() < 0.08 and turn < n_turns - 1

        for _ in range(rng.randint(4, 18)):
            tool = rng.choices(["Read", "Grep", "Bash", "Edit", "Write", "WebSearch"],
                               weights=[38, 18, 16, 14, 8, 6])[0]
            f = rng.choice(files)
            if tool in ("Read", "Edit", "Write"):
                inp = {"file_path": f}
                t.files_touched.add(f)
            elif tool == "Grep":
                inp = {"pattern": rng.choice(STEMS), "path": project}
            elif tool == "Bash":
                inp = {"command": f"pytest {rng.choice(files)} -q"}
            else:
                inp = {"query": f"{rng.choice(STEMS)} best practice"}
            tid, _ = w.tool_call(tool, inp)
            t.per_tool[tool] = t.per_tool.get(tool, 0) + 1

            r = rng.random()
            if r < 0.05:
                err = f"TransientError: upstream timeout while {tool.lower()}ing {f}"
                w.tool_result(tid, err, is_error=True)
                t.last_error_text = err
            elif r < 0.08:
                err = f"NotFound: {f} does not exist"
                w.tool_result(tid, err, is_error=True)
                t.last_error_text = err
            else:
                body = _pad(rng, 120, 1400)
                w.tool_result(tid, f"{body}\n[{tool} ok: {f}]")

        if interrupted:
            # dangling call: issued, never answered (until maybe next turn's flow)
            tool = "Bash"
            tid, _ = w.tool_call(tool, {"command": "sleep 600 && make build"})
            t.per_tool[tool] = t.per_tool.get(tool, 0) + 1
            w.user_msg("[Request interrupted by user]")
            t.dangling_tools.add(tool)
        else:
            w.agent_text(f"Done with: {ask}. " + _pad(rng, 20, 60))

    # occasionally end with an in-flight call (live-session shape)
    if rng.random() < 0.4:
        tid, _ = w.tool_call("Bash", {"command": "npm run test:integration"})
        t.per_tool["Bash"] = t.per_tool.get("Bash", 0) + 1
        t.dangling_tools.add("Bash")
    w.f.close()
    return t


CORPUS_SEEDS = list(range(201, 213))    # 12 sessions, fixed forever


def build_corpus(root: Path) -> dict[str, SessionTruth]:
    root.mkdir(parents=True, exist_ok=True)
    truths = {}
    for seed in CORPUS_SEEDS:
        p = root / f"synth-session-{seed}.jsonl"
        truths[p.name] = build_session(p, seed)
    return truths


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate the COMPREHEND synthetic corpus byte-identically.")
    ap.add_argument("--seeds", default="201-212", help="seed range, e.g. 201-212 (must match the released corpus)")
    ap.add_argument("--out", default="bench/synth_corpus", help="output directory")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (lo, hi) != (201, 212):
        raise SystemExit("released corpus is seeds 201-212; other ranges are not supported")
    build_corpus(out)
    print(f"wrote {len(list(out.glob('*.jsonl')))} sessions to {out}")


if __name__ == "__main__":
    main()
