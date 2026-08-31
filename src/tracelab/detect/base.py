"""Detector protocol + registry. Detectors are cheap, LLM-free, streaming."""

from __future__ import annotations

from tracelab.ledger.envelope import Event
from tracelab.state.runstate import Anomaly, RunState


class Detector:
    kind: str = "base"

    def observe(self, ev: Event, state: RunState) -> None:  # pragma: no cover
        raise NotImplementedError

    def _raise(self, state: RunState, *, severity: str, message: str,
               seq: int, evidence: list[int]) -> None:
        # de-dup: refresh existing active anomaly of same kind instead of stacking.
        # Keep detected_at_seq (first detection) and EXTEND evidence — replacing it
        # would lose the original firing point (and with it, lead-time semantics).
        for a in state.anomalies:
            if a.kind == self.kind and a.active:
                a.message = message
                a.severity = severity
                merged = a.evidence_seqs + [s for s in evidence if s not in a.evidence_seqs]
                a.evidence_seqs = a.evidence_seqs[:1] + merged[1:][-11:] if merged else []
                return
        state.anomalies.append(Anomaly(
            kind=self.kind, severity=severity, message=message,
            detected_at_seq=seq, evidence_seqs=evidence[-12:]))

    def _clear(self, state: RunState, seq: int) -> None:
        for a in state.anomalies:
            if a.kind == self.kind and a.active:
                a.cleared_at_seq = seq
