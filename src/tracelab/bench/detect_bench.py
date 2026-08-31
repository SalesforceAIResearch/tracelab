"""DETECT benchmark: detector precision/recall/lead-time on labeled synthetic traces.

Fully automatic, LLM-free, CI-friendly. Writes results into bench/scoreboard.json
under key "DETECT". Ground truth = PathologyLabels from the synthetic generator;
a detection counts as a hit if an anomaly of the matching kind fires with evidence
inside (or within `slack` events after) the labeled region. Lead-time = events between
the pathology's onset and first detection (lower is better).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from tracelab.detect.detectors import default_detectors
from tracelab.ledger.adapters.claude_code import parse_file
from tracelab.ledger.adapters.synthetic import generate
from tracelab.state.reducers import StateFolder

PATHOLOGIES = ["loop", "error_streak", "tool_flood", "stall"]
SLACK_EVENTS = 6  # detection may fire slightly after the region ends


@dataclass
class KindScore:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    lead_times: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None
        mean_lead = (sum(self.lead_times) / len(self.lead_times)) if self.lead_times else None
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": prec, "recall": rec, "mean_lead_events": mean_lead}


def _line_to_seq_ranges(events):
    """Map source line numbers -> seq span, for label matching."""
    by_line: dict[int, list[int]] = {}
    for e in events:
        by_line.setdefault(e.raw.line_no, []).append(e.seq)
    return by_line


def run(n_traces: int = 30, seed0: int = 100, out: Path | None = None) -> dict:
    scores = {k: KindScore() for k in PATHOLOGIES}
    clean_fp = 0

    with tempfile.TemporaryDirectory() as td:
        for i in range(n_traces):
            # mix: 1/5 clean traces (false-positive pressure), rest carry 1-3 pathologies
            rng_kinds = PATHOLOGIES[i % len(PATHOLOGIES):] + PATHOLOGIES[:i % len(PATHOLOGIES)]
            kinds = [] if i % 5 == 0 else rng_kinds[: (i % 3) + 1]
            trace = generate(Path(td) / f"t{i}.jsonl", seed=seed0 + i, pathologies=kinds)
            res = parse_file(trace.path)
            folder = StateFolder(str(trace.path), detectors=default_detectors())
            folder.fold(res.events)
            anomalies = folder.state.anomalies
            by_line = _line_to_seq_ranges(res.events)

            fired = {}
            matched_anoms: set[int] = set()
            for a in anomalies:
                fired.setdefault(a.kind, []).append(a)

            for lbl in trace.labels:
                seqs = [s for ln in range(lbl.start_line, lbl.end_line + 1)
                        for s in by_line.get(ln, [])]
                if not seqs:
                    continue
                lo, hi = min(seqs), max(seqs) + SLACK_EVENTS
                hits = [a for a in fired.get(lbl.kind, [])
                        if any(lo <= s <= hi for s in a.evidence_seqs + [a.detected_at_seq])]
                if hits:
                    scores[lbl.kind].tp += 1
                    # lead time = distance from onset to first DETECTION (not evidence,
                    # which contains the pathology's own earliest events)
                    first_detection = min(a.detected_at_seq for a in hits)
                    scores[lbl.kind].lead_times.append(max(0, first_detection - lo))
                    for a in hits:
                        matched_anoms.add(id(a))
                else:
                    scores[lbl.kind].fn += 1

            # false positives: any anomaly that matched NO label window of its kind —
            # including anomalies of labeled kinds firing entirely outside their windows
            for kind, alist in fired.items():
                if kind not in scores:
                    continue
                for a in alist:
                    if id(a) not in matched_anoms:
                        scores[kind].fp += 1
                        if not trace.labels:
                            clean_fp += 1

    result = {"n_traces": n_traces,
              "per_kind": {k: v.as_dict() for k, v in scores.items()},
              "clean_trace_false_positives": clean_fp}
    if out:
        board = {}
        if out.exists():
            board = json.loads(out.read_text())
        board["DETECT"] = result
        out.write_text(json.dumps(board, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(out=Path(__file__).resolve().parents[3] / "bench" / "scoreboard.json"),
                     indent=2))
