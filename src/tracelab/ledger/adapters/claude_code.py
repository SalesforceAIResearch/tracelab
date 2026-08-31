"""Claude Code JSONL adapter: source records -> canonical Events.

Schema notes (verified against real transcripts, Aug 2026):
- Record types seen: user, assistant, attachment, queue-operation, last-prompt,
  ai-title, mode, summary, system. Unknown types map to SESSION_META/UNKNOWN but are
  never dropped.
- `uuid`/`parentUuid` form the native causal chain; `isSidechain: true` marks
  subagent branches (agent_id = first sidechain ancestor's record id is out of scope
  for v0 — we mark agent_id = "sidechain").
- assistant.message.content is a list of blocks (thinking | text | tool_use ...);
  each block becomes its own Event with event_id "<uuid>#<i>".
- user records may carry tool results either as message.content blocks of type
  tool_result or via the `toolUseResult` field; correlation_id = tool_use_id.
- assistant.message.usage carries cache_read/cache_creation tokens.

Robustness contract (tested):
- Malformed lines are counted, skipped, never fatal.
- A partial (mid-write) final line is NOT consumed; `resume_offset` stops before it so
  a tailer can continue exactly there once the writer finishes the line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tracelab.ledger.envelope import (
    Event, EventKind, Payload, RawRef, Source, Usage, make_payload,
)

META_TYPES = {"queue-operation", "last-prompt", "ai-title", "mode",
              "pr-link", "permission-mode", "file-history-snapshot"}


@dataclass
class ParseStats:
    lines: int = 0
    events: int = 0
    malformed: int = 0
    trailing_partial_bytes: int = 0
    unknown_types: dict[str, int] = field(default_factory=dict)


@dataclass
class ParseResult:
    events: list[Event]
    stats: ParseStats
    resume_offset: int  # byte offset AFTER the last fully-consumed line


def _ts(rec: dict) -> datetime | None:
    raw = rec.get("timestamp")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # naive stamps must not poison arithmetic
    return dt


def _content_text(content) -> str:
    """Flatten user-message content (str or block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    parts.append(_content_text(inner) if inner is not None else "")
                else:
                    parts.append(json.dumps(block, ensure_ascii=False)[:500])
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _usage(msg: dict) -> Usage | None:
    u = msg.get("usage")
    if not isinstance(u, dict):
        return None
    return Usage(
        model=msg.get("model"),
        input_tokens=u.get("input_tokens", 0) or 0,
        output_tokens=u.get("output_tokens", 0) or 0,
        cache_read_tokens=u.get("cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=u.get("cache_creation_input_tokens", 0) or 0,
    )


def _base_kwargs(rec: dict, path: str, line_no: int, offset: int, block: int = 0) -> dict:
    return dict(
        ts=_ts(rec),
        session_id=rec.get("sessionId", "unknown"),
        agent_id="sidechain" if rec.get("isSidechain") else None,
        parent_event_id=rec.get("parentUuid"),
        raw=RawRef(path=path, line_no=line_no, byte_offset=offset, block_index=block),
    )


def _events_from_record(rec: dict, path: str, line_no: int, offset: int,
                        stats: ParseStats) -> Iterator[dict]:
    """Yield kwargs-dicts (seq/event assembly happens in the caller)."""
    rtype = rec.get("type")
    rid = rec.get("uuid") or f"{Path(path).stem}:{line_no}"

    if rtype == "assistant":
        msg = rec.get("message") or {}
        usage = _usage(msg)
        msg_id = msg.get("id") or rec.get("requestId")
        blocks = msg.get("content")
        if not isinstance(blocks, list):
            blocks = [{"type": "text", "text": _content_text(blocks)}]
        is_err = bool(rec.get("isApiErrorMessage"))
        for i, b in enumerate(blocks):
            if not isinstance(b, dict):
                b = {"type": "text", "text": str(b)}
            btype = b.get("type")
            common = _base_kwargs(rec, path, line_no, offset, block=i)
            # usage attaches once per record (first block) to avoid double counting
            u = usage if i == 0 else None
            if btype == "thinking":
                yield dict(event_id=f"{rid}#{i}", source=Source.AGENT, kind=EventKind.THINKING,
                           api_message_id=msg_id,
                           payload=make_payload(b.get("thinking", "")), usage=u, **common)
            elif btype == "tool_use":
                yield dict(event_id=f"{rid}#{i}", source=Source.AGENT, kind=EventKind.TOOL_CALL,
                           api_message_id=msg_id, correlation_id=b.get("id"),
                           payload=make_payload(
                               json.dumps(b.get("input", {}), ensure_ascii=False),
                               tool_name=b.get("name")),
                           usage=u, **common)
            else:  # text or anything else in an assistant message
                yield dict(event_id=f"{rid}#{i}", source=Source.AGENT, api_message_id=msg_id,
                           kind=EventKind.ERROR if is_err else EventKind.MESSAGE,
                           payload=make_payload(b.get("text", _content_text(b))),
                           usage=u, **common)
        if not blocks:
            yield dict(event_id=f"{rid}#0", source=Source.AGENT, kind=EventKind.MESSAGE,
                       api_message_id=msg_id, payload=make_payload(""), usage=usage,
                       **_base_kwargs(rec, path, line_no, offset))

    elif rtype == "user" and rec.get("isMeta"):
        # meta user records (caveats, command echoes, local stdout) are plumbing,
        # not user turns — kind them as SESSION_META so reducers never miscount
        yield dict(event_id=f"{rid}#0", source=Source.META, kind=EventKind.SESSION_META,
                   payload=make_payload(_content_text((rec.get("message") or {}).get("content"))[:500],
                                        tool_name="isMeta"),
                   **_base_kwargs(rec, path, line_no, offset))

    elif rtype == "user":
        msg = rec.get("message") or {}
        content = msg.get("content")
        tur = rec.get("toolUseResult")
        blocks = content if isinstance(content, list) else None
        tool_result_blocks = [b for b in (blocks or [])
                              if isinstance(b, dict) and b.get("type") == "tool_result"]
        if tool_result_blocks:
            for i, b in enumerate(tool_result_blocks):
                yield dict(event_id=f"{rid}#{i}", source=Source.TOOL, kind=EventKind.TOOL_RESULT,
                           correlation_id=b.get("tool_use_id"),
                           payload=make_payload(_content_text(b.get("content")),
                                                is_error=bool(b.get("is_error"))),
                           **_base_kwargs(rec, path, line_no, offset, block=i))
        elif tur is not None:
            text = tur if isinstance(tur, str) else json.dumps(tur, ensure_ascii=False)
            yield dict(event_id=f"{rid}#0", source=Source.TOOL, kind=EventKind.TOOL_RESULT,
                       correlation_id=rec.get("sourceToolUseID"),
                       payload=make_payload(text),
                       **_base_kwargs(rec, path, line_no, offset))
        else:
            yield dict(event_id=f"{rid}#0", source=Source.USER, kind=EventKind.MESSAGE,
                       payload=make_payload(_content_text(content)),
                       **_base_kwargs(rec, path, line_no, offset))

    elif rtype == "attachment":
        yield dict(event_id=f"{rid}#0", source=Source.SYSTEM, kind=EventKind.ATTACHMENT,
                   payload=make_payload(json.dumps(rec.get("attachment", {}),
                                                   ensure_ascii=False)[:2000]),
                   **_base_kwargs(rec, path, line_no, offset))

    elif rtype == "summary":
        # Continuation/compaction marker written by Claude Code
        yield dict(event_id=f"{rid}#0", source=Source.SYSTEM, kind=EventKind.COMPACTION,
                   payload=make_payload(str(rec.get("summary", ""))),
                   **_base_kwargs(rec, path, line_no, offset))

    elif rtype in META_TYPES or rtype == "system":
        yield dict(event_id=f"{rid}#0", source=Source.META, kind=EventKind.SESSION_META,
                   payload=make_payload(json.dumps(
                       {k: v for k, v in rec.items() if k not in ("type",)},
                       ensure_ascii=False, default=str)[:1000],
                       tool_name=rtype),
                   **_base_kwargs(rec, path, line_no, offset))

    else:
        stats.unknown_types[str(rtype)] = stats.unknown_types.get(str(rtype), 0) + 1
        yield dict(event_id=f"{rid}#0", source=Source.META, kind=EventKind.UNKNOWN,
                   payload=make_payload(json.dumps(rec, ensure_ascii=False, default=str)[:1000],
                                        tool_name=str(rtype)),
                   **_base_kwargs(rec, path, line_no, offset))


def parse_file(path: str | Path, *, start_offset: int = 0, start_seq: int = 0,
               start_line: int = 1) -> ParseResult:
    """Parse a Claude Code JSONL transcript from a byte offset.

    Ingest-resume contract: parse_file(path, start_offset=r1.resume_offset, ...)
    continues exactly where the previous call stopped; concatenating the event lists
    equals a single whole-file parse (property-tested).
    """
    path = Path(path)
    stats = ParseStats()
    events: list[Event] = []
    seq = start_seq
    line_no = start_line
    offset = start_offset
    resume = start_offset

    with open(path, "rb") as f:
        f.seek(start_offset)
        data = f.read()

    pos = 0
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl == -1:
            stats.trailing_partial_bytes = len(data) - pos  # visible, never silent
            break  # partial final line: leave for next round; resume stays before it
        line = data[pos:nl]
        line_offset = offset + pos
        pos = nl + 1
        resume = offset + pos
        stats.lines += 1
        stripped = line.strip()
        if not stripped:
            line_no += 1
            continue
        try:
            rec = json.loads(stripped)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except Exception:
            stats.malformed += 1
            line_no += 1
            continue
        for kw in _events_from_record(rec, str(path), line_no, line_offset, stats):
            events.append(Event(seq=seq, **kw))
            seq += 1
        line_no += 1

    stats.events = len(events)
    return ParseResult(events=events, stats=stats, resume_offset=resume)
