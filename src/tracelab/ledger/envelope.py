"""Canonical event envelope — L1 of the trace model.

Every source format (Claude Code JSONL first, OTel later) normalizes into `Event`.
Payloads live BY REFERENCE: `raw` points back into the immutable source transcript
(path, line, byte offset) and `payload.sha256` fingerprints the content; `payload.text`
holds an inline copy only up to `INLINE_CAP` characters so the ledger stays small and
remains useful in metadata-only (redacted) mode.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
INLINE_CAP = 4_000  # chars of payload text kept inline; full content stays in the source file


class EventKind(StrEnum):
    MESSAGE = "message"            # user or assistant text
    THINKING = "thinking"          # assistant reasoning block
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ATTACHMENT = "attachment"
    COMPACTION = "compaction"      # source-recorded context compaction/summary
    SESSION_META = "session_meta"  # mode changes, titles, queue ops, prompts
    ERROR = "error"
    UNKNOWN = "unknown"            # preserved, never dropped


class Source(StrEnum):
    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"
    META = "meta"


class RawRef(BaseModel, frozen=True):
    """Coordinates into the immutable source transcript (the blob store)."""
    path: str
    line_no: int          # 1-based line in the source file
    byte_offset: int      # offset of the line start
    block_index: int = 0  # index within a multi-block record (assistant content[])


class Payload(BaseModel, frozen=True):
    text: str | None = None       # inline copy, truncated to INLINE_CAP
    truncated: bool = False
    sha256: str | None = None     # fingerprint of the FULL original text
    char_len: int = 0
    tool_name: str | None = None  # for tool_call/tool_result
    tool_input_preview: str | None = None
    is_error: bool = False


class Usage(BaseModel, frozen=True):
    """Token/cost accounting — cache fields are first-class (KV-cache economics)."""
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class Event(BaseModel, frozen=True):
    event_id: str                       # source-native uuid, suffixed '#<block>' for sub-blocks
    seq: int                            # monotone per ledger, assigned at ingest
    ts: datetime | None = None
    source: Source
    kind: EventKind
    session_id: str
    agent_id: str | None = None         # sidechain/subagent identity, None = main thread
    parent_event_id: str | None = None  # source-native causal parent
    correlation_id: str | None = None   # tool_use_id linking call <-> result
    api_message_id: str | None = None   # provider message id; usage dedup key (Claude Code
                                        # writes one record PER BLOCK, repeating usage)
    payload: Payload = Field(default_factory=Payload)
    usage: Usage | None = None
    raw: RawRef
    schema_version: int = SCHEMA_VERSION

    @property
    def record_id(self) -> str:
        """Source-record id without the block suffix."""
        return self.event_id.split("#", 1)[0]


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def make_payload(text: str | None, **kw) -> Payload:
    if text is None:
        return Payload(**kw)
    return Payload(
        text=text[:INLINE_CAP],
        truncated=len(text) > INLINE_CAP,
        sha256=fingerprint(text),
        char_len=len(text),
        **kw,
    )
