"""L3: DerivedNodes — versioned, reversible artifacts over the ledger.

Design rules (from the three-survey consensus + our innovation mandate):
- nodes are immutable once created; changes create a NEW version that `supersedes`
  the old one (anchored-iterative: merge forward, never rewrite history);
- every node carries provenance: covered seq range, evidence pins, creation frontier;
- validity lifecycle: current -> suspected_stale -> invalidated / superseded;
- automatic STALENESS PROPAGATION: invalidating/staling a node marks its dependent
  closure suspected_stale (identified as unimplemented across the surveyed literature —
  the first less-crowded-space feature of the parsing mechanism).
"""

from __future__ import annotations

import itertools
from enum import StrEnum
from pydantic import BaseModel, Field


class Validity(StrEnum):
    CURRENT = "current"
    SUSPECTED_STALE = "suspected_stale"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


class NodeKind(StrEnum):
    EPISODE = "episode"       # one user-turn's worth of work, folded
    SUMMARY = "summary"
    FACT = "fact"
    HYPOTHESIS = "hypothesis"


class DerivedNode(BaseModel, frozen=True):
    node_id: str
    version: int
    kind: NodeKind
    covered_lo: int                      # seq range in the ledger
    covered_hi: int
    created_at_seq: int                  # ledger frontier when this version was built
    producer: str                        # "deterministic" | "llm:<model>" | "hindsight:<rule>"
    title: str = ""
    content: str = ""                    # human-readable digest
    structured: dict = Field(default_factory=dict)
    evidence_pins: list[int] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)   # node_ids this one relies on
    supersedes: str | None = None        # "<node_id>@<version>"

    @property
    def ref(self) -> str:
        return f"{self.node_id}@{self.version}"


class NodeStore:
    """Versioned store with validity tracking + dependency-driven staleness."""

    def __init__(self):
        self._versions: dict[str, list[DerivedNode]] = {}
        self._validity: dict[str, Validity] = {}     # keyed by ref
        self._audit: list[tuple[str, str, int]] = [] # (ref, transition, at_seq)
        self._ids = itertools.count(1)

    # ------------------------------------------------------------- create

    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids):04d}"

    def add(self, node: DerivedNode) -> DerivedNode:
        chain = self._versions.setdefault(node.node_id, [])
        if chain and node.version != chain[-1].version + 1:
            raise ValueError(f"{node.node_id}: version {node.version} after {chain[-1].version}")
        if not chain and node.version != 1:
            raise ValueError(f"{node.node_id}: first version must be 1")
        if node.supersedes:
            if node.supersedes not in self._validity:
                raise ValueError(f"supersedes unknown ref {node.supersedes}")
            self._set(node.supersedes, Validity.SUPERSEDED, node.created_at_seq)
        chain.append(node)
        self._validity[node.ref] = Validity.CURRENT
        self._audit.append((node.ref, "created", node.created_at_seq))
        return node

    def evolve(self, node_id: str, at_seq: int, producer: str, **updates) -> DerivedNode:
        """Anchored-iterative merge: new version superseding the head."""
        head = self.head(node_id)
        if head is None:
            raise KeyError(node_id)
        data = head.model_dump()
        data.update(updates)
        data.update(version=head.version + 1, created_at_seq=at_seq,
                    producer=producer, supersedes=head.ref)
        return self.add(DerivedNode(**data))

    # ------------------------------------------------------------- lookup

    def head(self, node_id: str) -> DerivedNode | None:
        chain = self._versions.get(node_id)
        return chain[-1] if chain else None

    def heads(self, kind: NodeKind | None = None) -> list[DerivedNode]:
        out = [c[-1] for c in self._versions.values()]
        if kind:
            out = [n for n in out if n.kind == kind]
        return sorted(out, key=lambda n: n.covered_lo)

    def validity(self, ref: str) -> Validity:
        return self._validity[ref]

    def history(self, node_id: str) -> list[DerivedNode]:
        return list(self._versions.get(node_id, []))

    @property
    def audit(self) -> list[tuple[str, str, int]]:
        return list(self._audit)

    # ---------------------------------------------------- validity + staleness

    def _set(self, ref: str, v: Validity, at_seq: int) -> None:
        cur = self._validity.get(ref)
        # lifecycle guard: INVALIDATED is fully terminal; SUPERSEDED may only be
        # escalated to INVALIDATED (a replaced reading later found WRONG) — never
        # downgraded to current/stale
        if cur == Validity.INVALIDATED and v != cur:
            return
        if cur == Validity.SUPERSEDED and v not in (Validity.SUPERSEDED, Validity.INVALIDATED):
            return
        if cur != v:
            self._validity[ref] = v
            self._audit.append((ref, v.value, at_seq))

    def invalidate(self, ref: str, at_seq: int) -> list[str]:
        """Invalidate a node version; dependents become suspected_stale (closure)."""
        self._set(ref, Validity.INVALIDATED, at_seq)
        return self._propagate_stale(ref, at_seq)

    def mark_stale(self, ref: str, at_seq: int) -> list[str]:
        self._set(ref, Validity.SUSPECTED_STALE, at_seq)
        return self._propagate_stale(ref, at_seq)

    def propagate_from(self, ref: str, at_seq: int) -> list[str]:
        """Staleness for DEPENDENTS of ref without touching ref's own validity."""
        return self._propagate_stale(ref, at_seq)

    def _dependents_of(self, ref: str) -> list[DerivedNode]:
        """Version-sensitive: a dependent is affected only if it pinned EXACTLY this
        version, or depends on the bare node_id (unpinned = any version)."""
        node_id = ref.split("@", 1)[0]
        out = []
        for chain in self._versions.values():
            head = chain[-1]
            for dep in head.dependencies:
                if dep == ref or dep == node_id:
                    out.append(head)
                    break
        return out

    def _propagate_stale(self, ref: str, at_seq: int) -> list[str]:
        """BFS over the dependent closure; only CURRENT heads transition."""
        touched: list[str] = []
        frontier = [ref]
        seen = {ref}
        while frontier:
            nxt: list[str] = []
            for r in frontier:
                for dep_head in self._dependents_of(r):
                    dref = dep_head.ref
                    if dref in seen:
                        continue
                    seen.add(dref)
                    if self._validity.get(dref) == Validity.CURRENT:
                        self._set(dref, Validity.SUSPECTED_STALE, at_seq)
                        touched.append(dref)
                    nxt.append(dref)
            frontier = nxt
        return touched
