"""How the review logs print chat items and task rows: one short, greppable line each."""

from collections.abc import Iterable

from livekit.agents._generated.storage_run import TaskRecord  # the row type, not re-exported
from livekit.agents.llm import ChatItem

SHOWN_METADATA = (
    "livekit/interrupted",
    "livekit/mergedInto",
    "livekit/reason",
    "livekit/background",
    "livekit/queued",
    "livekit/cancelRequested",
)


def describe(item: ChatItem) -> str:
    """One chat item as `role:'text'`, `call:name(args)` or `out:name='output'`."""
    if item.type == "message":
        return f"{item.role}:{(item.text_content or '')[:120]!r}"
    if item.type == "function_call":
        return f"call:{item.name}({item.arguments})"
    if item.type == "function_call_output":
        return f"out:{item.name}={item.output[:120]!r}"
    return item.type


def rows(records: Iterable[TaskRecord], limit: int = 8) -> str:
    """Task rows newest first as `id:status`, with the open ask and the stop metadata."""
    parts = []
    for record in sorted(records, key=lambda r: r.seq, reverse=True)[:limit]:
        part = f"{record.id[-6:]}:{record.status}"
        ask = record.internal.ask if record.internal is not None else None
        if isinstance(ask, dict):
            call_id = ask.get("call_id")
            part += f"+ask({call_id})" if call_id else "+ask(no call_id)"
        for key in SHOWN_METADATA:
            if key in record.data.metadata:
                part += f"[{key.removeprefix('livekit/')}={record.data.metadata[key]}]"
        parts.append(part)
    return " ".join(parts) or "none"
