"""Mini-workbench: deterministic long-horizon tasks with verifiable outcomes.

In-process simulated filesystem + tools. Tasks need 15–40 tool calls, produce verbose
observations (long-horizon context pressure by construction), and inject transient
tool errors (recovery pressure). Outcome checking is exact — no judge.

Tools: list_files · read_file · write_file · search · done
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


PAD_WORDS = ("lorem", "ipsum", "config", "legacy", "vendor", "handler", "widget",
             "module", "service", "adapter", "registry", "pipeline")


def _pad(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(PAD_WORDS) for _ in range(n))


@dataclass
class ToolOutcome:
    text: str
    is_error: bool = False


@dataclass
class Task:
    task_id: str
    instruction: str
    files: dict[str, str]
    check: callable                    # (files) -> (score 0..1, list[str] subcheck notes)
    optimal_calls: int
    disabled_tools: tuple = ()         # shortcut-proofing (e.g. no search on chain tasks)


def make_scatter_task(seed: int, *, n_override: int | None = None) -> Task:
    """Values scattered across many files; agent must collect and write a summary."""
    rng = random.Random(seed)
    n = n_override or rng.randint(6, 9)
    files, wanted = {}, {}
    for i in range(n):
        key = f"metric_{chr(97 + i)}"
        val = rng.randint(100, 999)
        wanted[key] = val
        body = "\n".join(_pad(rng, 12) for _ in range(rng.randint(6, 14)))
        pos = rng.randint(0, 3)
        lines = body.split("\n")
        lines.insert(pos, f"{key} = {val}")
        files[f"src/part_{i}.txt"] = "\n".join(lines)
    for j in range(rng.randint(4, 7)):   # decoys
        files[f"docs/note_{j}.txt"] = "\n".join(_pad(rng, 15) for _ in range(8))

    def check(fs: dict[str, str]):
        out = fs.get("summary.txt", "")
        pairs = {}
        for line in out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                pairs[k.strip()] = v.strip()
        notes, hits = [], 0
        for k, v in wanted.items():
            if pairs.get(k) == str(v):           # key AND value paired on one line
                hits += 1
            else:
                notes.append(f"missing/mispaired {k}={v}")
        return hits / len(wanted), notes

    instr = (f"Collect the values of all metric_* keys defined in files under src/ "
             f"(there are {n}) and write them to a new file 'summary.txt', one "
             f"'key = value' per line. Then call done.")
    return Task(f"scatter-{seed}", instr, files, check, optimal_calls=n + 3)


def make_fix_task(seed: int) -> Task:
    """Config values violate a stated rule across files; agent must fix every one."""
    rng = random.Random(seed)
    n = rng.randint(5, 8)
    files = {}
    bad_paths = []
    for i in range(n):
        path = f"conf/service_{i}.cfg"
        good = rng.randint(10, 49)
        bad = rng.randint(90, 250)
        use_bad = rng.random() < 0.7
        if use_bad:
            bad_paths.append(path)
        body = [f"# service {i} config", _pad(rng, 8),
                f"timeout_s = {bad if use_bad else good}", _pad(rng, 8),
                "retries = 3"]
        files[path] = "\n".join(body)

    original_paths = sorted(files)               # captured at creation: no dilution
    violations = list(bad_paths)

    def check(fs: dict[str, str]):
        notes, fixed = [], 0
        for p in violations:                      # baseline-0: score = violations fixed
            body = fs.get(p, "")
            try:
                line = next(l for l in body.splitlines() if l.startswith("timeout_s"))
                val = int(line.split("=")[1].strip())
                if val <= 60:
                    fixed += 1
                else:
                    notes.append(f"{p} timeout still {val}")
            except (StopIteration, ValueError):
                notes.append(f"{p} timeout line broken")
        # regression guard: originally-good files must stay valid
        for p in original_paths:
            if p not in violations and p in fs:
                if "timeout_s" not in fs[p]:
                    notes.append(f"{p} timeout line removed")
        denom = max(1, len(violations))
        return fixed / denom, notes

    instr = ("Policy: every conf/*.cfg must have timeout_s <= 60. Find violations and "
             "fix each by setting timeout_s = 30 (keep everything else unchanged). "
             "Then call done.")
    return Task(f"fix-{seed}", instr, files, check, optimal_calls=2 * len(bad_paths) + 3)


def make_chain_task(seed: int, *, n_links: int = 30) -> Task:
    """Sequential-dependency chain: each file holds a delta and the NEXT file's path.
    No global shortcut exists (search disabled); the running knowledge (deltas seen,
    next hop) must survive in context. The genuine long-horizon memory regime."""
    rng = random.Random(seed)
    names = [f"node/{rng.randrange(16**8):08x}.txt" for _ in range(n_links)]
    deltas = [rng.randint(1, 99) for _ in range(n_links)]
    files = {}
    for i, name in enumerate(names):
        nxt = names[i + 1] if i + 1 < n_links else "END"
        body = [_pad(rng, 10), f"delta = {deltas[i]}", _pad(rng, 10),
                f"next = {nxt}", _pad(rng, 10)]
        files[name] = "\n".join(body)
    files["start.txt"] = f"first = {names[0]}\n{_pad(rng, 10)}"
    total = sum(deltas)

    def check(fs: dict[str, str]):
        out = fs.get("total.txt", "")
        for line in out.splitlines():
            if line.replace(" ", "").startswith("total="):
                try:
                    got = int(line.split("=")[1].strip())
                except ValueError:
                    return 0.0, ["total not an int"]
                if got == total:
                    return 1.0, []
                # partial credit: fraction of chain summed correctly from the start
                run = 0
                for k in range(n_links, 0, -1):
                    if got == sum(deltas[:k]):
                        run = k
                        break
                return run / n_links, [f"total {got} != {total} (prefix match {run}/{n_links})"]
        return 0.0, ["no total= line in total.txt"]

    instr = ("Follow the chain: read start.txt for the first file; every file gives a "
             "'delta' and the 'next' file path (chain ends at END). Sum ALL deltas and "
             "write 'total = <sum>' to total.txt. search is disabled. Then call done.")
    return Task(f"chain-{seed}", instr, files, check,
                optimal_calls=n_links + 3, disabled_tools=("search",))


DELTA_PROSE = ("During this hop the counter advanced by {d} units.",
               "An adjustment of {d} was applied to the running tally.",
               "This node contributes {d} to the overall total.")
NEXT_PROSE = ("When done here, continue with the file {n}.",
              "The trail proceeds onward at {n}.",
              "Consult {n} for the following hop.")


def make_prose_chain_task(seed: int, *, n_links: int = 30) -> Task:
    """Chain task with facts stated in PROSE — no 'key = value' lines, so the
    deterministic regex extractor is blind by construction. Same checker as chain.
    The budget-dial substrate: only semantic (LLM) extraction can pin these facts."""
    rng = random.Random(seed)
    names = [f"node/{rng.randrange(16**8):08x}.txt" for _ in range(n_links)]
    deltas = [rng.randint(1, 99) for _ in range(n_links)]
    files = {}
    for i, name in enumerate(names):
        d_line = rng.choice(DELTA_PROSE).format(d=deltas[i])
        n_line = ("This is the final node; the trail ends here."
                  if i + 1 >= n_links else
                  rng.choice(NEXT_PROSE).format(n=names[i + 1]))
        files[name] = "\n".join([_pad(rng, 10), d_line, _pad(rng, 10), n_line,
                                 _pad(rng, 10)])
    files["start.txt"] = (f"The trail begins at the file {names[0]}.\n"
                          f"{_pad(rng, 10)}")
    total = sum(deltas)

    def check(fs: dict[str, str]):
        out = fs.get("total.txt", "")
        for line in out.splitlines():
            if line.replace(" ", "").startswith("total="):
                try:
                    got = int(line.split("=")[1].strip())
                except ValueError:
                    return 0.0, ["total not an int"]
                if got == total:
                    return 1.0, []
                run = 0
                for k in range(n_links, 0, -1):
                    if got == sum(deltas[:k]):
                        run = k
                        break
                return run / n_links, [f"total {got} != {total} "
                                       f"(prefix match {run}/{n_links})"]
        return 0.0, ["no total= line in total.txt"]

    instr = ("Follow the trail: read start.txt for the first file. Every node file "
             "states, in prose, an integer contribution to a running total and the "
             "next file on the trail (the last node says the trail ends). Sum ALL "
             "contributions and write 'total = <sum>' to total.txt. search is "
             "disabled. Then call done.")
    return Task(f"prosechain-{seed}", instr, files, check,
                optimal_calls=n_links + 3, disabled_tools=("search",))


def make_altchain_task(seed: int, *, n_links: int = 60) -> Task:
    """Order-sensitive chain: total = d1 - d2 + d3 - d4 ... (sign alternates with
    TRAVERSAL position). The fold's per-key sum aggregate does NOT precompute this
    answer - the arm must preserve order, not just values. Built to bound, not
    flatter, the requirements story."""
    rng = random.Random(seed)
    names = [f"node/{rng.randrange(16**8):08x}.txt" for _ in range(n_links)]
    deltas = [rng.randint(1, 99) for _ in range(n_links)]
    files = {}
    for i, name in enumerate(names):
        nxt = names[i + 1] if i + 1 < n_links else "END"
        body = [_pad(rng, 10), f"delta = {deltas[i]}", _pad(rng, 10),
                f"next = {nxt}", _pad(rng, 10)]
        files[name] = "\n".join(body)
    files["start.txt"] = f"first = {names[0]}\n{_pad(rng, 10)}"
    total = sum(d if i % 2 == 0 else -d for i, d in enumerate(deltas))

    def check(fs: dict[str, str]):
        out = fs.get("total.txt", "")
        for line in out.splitlines():
            if line.replace(" ", "").startswith("total="):
                try:
                    got = int(line.split("=")[1].strip())
                except ValueError:
                    return 0.0, ["total not an int"]
                return (1.0, []) if got == total else (
                    0.0, [f"total {got} != {total}"])
        return 0.0, ["no total= line in total.txt"]

    instr = ("Follow the chain: read start.txt for the first file; every file gives "
             "a 'delta' and the 'next' file path (chain ends at END). Compute the "
             "ALTERNATING total: ADD the 1st delta, SUBTRACT the 2nd, ADD the 3rd, "
             "and so on in traversal order. Write 'total = <sum>' to total.txt. "
             "search is disabled. Then call done.")
    return Task(f"altchain-{seed}", instr, files, check,
                optimal_calls=n_links + 3, disabled_tools=("search",))


TASK_MAKERS = {"scatter": make_scatter_task, "fix": make_fix_task,
               "chain": make_chain_task, "prosechain": make_prose_chain_task,
               "altchain": make_altchain_task}


class Workbench:
    def __init__(self, task: Task, *, seed: int = 0, error_rate: float = 0.08,
                 verbosity_pad: int = 40):
        self.task = task
        self.files = dict(task.files)
        self.rng = random.Random(seed ^ 0xBEEF)          # noise only
        err_rng = random.Random(seed ^ 0xE44)             # dedicated: same schedule
        self._err_schedule = [err_rng.random() < error_rate for _ in range(2000)]
        self.error_rate = error_rate
        self.verbosity_pad = verbosity_pad
        self.calls = 0
        self.errors_injected = 0
        self.done_called = False

    # ------------------------------------------------------------- tools

    def call(self, tool: str, args: dict) -> ToolOutcome:
        """Validation precedes injection: hallucinated tools get UnknownTool, never a
        misleading retryable TransientError. Error schedule is pre-committed per
        (seed, call-index) so every policy arm faces identical adversity at identical
        call depth. `done` is exempt (documented)."""
        self.calls += 1
        if tool in self.task.disabled_tools:
            return ToolOutcome(f"ToolDisabled: {tool} is unavailable for this task",
                               is_error=True)
        fn = getattr(self, f"_t_{tool}", None)
        if fn is None:
            return ToolOutcome(f"UnknownTool: {tool}", is_error=True)
        if tool != "done" and self._err_schedule[min(self.calls - 1,
                                                     len(self._err_schedule) - 1)]:
            self.errors_injected += 1
            return ToolOutcome("TransientError: backend unavailable, retry the call",
                               is_error=True)
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return ToolOutcome(f"BadArgs: {e}", is_error=True)

    def _noise(self) -> str:
        return _pad(self.rng, self.verbosity_pad)

    def _t_list_files(self) -> ToolOutcome:
        listing = "\n".join(sorted(self.files))
        return ToolOutcome(f"{listing}\n\n[fs-meta] {self._noise()}")

    def _t_read_file(self, path: str) -> ToolOutcome:
        if path not in self.files:
            return ToolOutcome(f"NotFound: {path}", is_error=True)
        return ToolOutcome(f"{self.files[path]}\n\n[io-meta] {self._noise()}")

    def _t_write_file(self, path: str, content: str) -> ToolOutcome:
        self.files[path] = content
        return ToolOutcome(f"wrote {len(content)} chars to {path}\n[io-meta] {self._noise()}")

    def _t_search(self, pattern: str) -> ToolOutcome:
        hits = []
        for p, body in sorted(self.files.items()):
            for i, line in enumerate(body.splitlines(), 1):
                if pattern in line:
                    hits.append(f"{p}:{i}: {line.strip()[:120]}")
        return ToolOutcome(("\n".join(hits[:40]) or "no matches")
                           + f"\n[search-meta] {self._noise()}")

    def _t_tally(self, add: int = 0) -> ToolOutcome:
        """Env-side accumulator tool (reviewer-requested calculator control):
        available only when the runner advertises it in the system prompt."""
        if not hasattr(self, "_tally"):
            self._tally = 0
        self._tally += int(add)
        return ToolOutcome(f"tally = {self._tally}")

    def _t_done(self) -> ToolOutcome:
        self.done_called = True
        return ToolOutcome("done acknowledged")

    # ------------------------------------------------------------- outcome

    def score(self) -> tuple[float, list[str]]:
        return self.task.check(self.files)
