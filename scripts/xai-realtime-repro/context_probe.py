#!/usr/bin/env python3
"""Multi-turn context probe: seed a fact, filler turns, ask for the fact back."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_MODEL,
    drain_until,
    event_error_message,
    open_realtime_ws,
    pass_fail,
    require_xai_api_key,
    run_async,
    send_event,
)

SECRET = "BLUE-ORBIT-7"


def _missing_id_fields(event: dict[str, Any]) -> list[str]:
    etype = str(event.get("type") or "")
    if not etype.startswith("conversation.item"):
        return []
    missing: list[str] = []
    if "item_id" in event and not event.get("item_id"):
        missing.append("item_id")
    if "previous_item_id" in event and event.get("previous_item_id") in (None, ""):
        missing.append("previous_item_id")
    item = event.get("item")
    if isinstance(item, dict) and "id" in item and not item.get("id"):
        missing.append("item.id")
    return missing


def _collect_text(events: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for event in events:
        etype = event.get("type")
        if etype in {"response.text.delta", "response.output_text.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                chunks.append(delta)
        elif etype in {"response.audio_transcript.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                chunks.append(delta)
        elif etype == "response.done":
            resp = event.get("response") or {}
            for output in resp.get("output") or []:
                if not isinstance(output, dict):
                    continue
                for content in output.get("content") or []:
                    if not isinstance(content, dict):
                        continue
                    for key in ("text", "transcript"):
                        val = content.get(key)
                        if isinstance(val, str):
                            chunks.append(val)
    return "".join(chunks)


async def _user_turn(
    ws: Any,
    text: str,
    *,
    missing_counts: dict[str, int],
) -> str:
    errors: list[str] = []

    def on_event(event: dict[str, Any]) -> None:
        msg = event_error_message(event)
        if msg:
            errors.append(msg)
            print(f"server error: {msg}", file=sys.stderr)
        for field in _missing_id_fields(event):
            missing_counts[field] = missing_counts.get(field, 0) + 1
            print(f"WARNING: missing {field} on {event.get('type')}")

    await send_event(
        ws,
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
    )
    await send_event(
        ws,
        {
            "type": "response.create",
            "response": {"modalities": ["text"]},
        },
    )
    events = await drain_until(
        ws,
        lambda e: e.get("type") in {"response.done", "response.completed"}
        or event_error_message(e) is not None,
        timeout=60.0,
        on_event=on_event,
    )
    if errors:
        raise RuntimeError(errors[0])
    return _collect_text(events)


async def main() -> int:
    api_key = require_xai_api_key()
    missing_counts: dict[str, int] = {}

    livekit_ready = all(
        os.environ.get(name)
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    )
    if livekit_ready:
        print(
            "note: LIVEKIT_* is set; this default path still uses the raw websocket "
            "so you can repro without a room."
        )
    else:
        print("note: default path needs only XAI_API_KEY (no LiveKit room).")

    http, ws = await open_realtime_ws(api_key)
    try:
        await drain_until(
            ws,
            lambda e: e.get("type") in {"session.created", "session.updated"},
            timeout=30.0,
        )
        await send_event(
            ws,
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": DEFAULT_MODEL,
                    "instructions": (
                        "You are a concise text assistant. Remember facts the user states. "
                        "Answer in one short sentence."
                    ),
                },
            },
        )
        await drain_until(ws, lambda e: e.get("type") == "session.updated", timeout=30.0)

        await _user_turn(
            ws,
            f"Remember that the secret code is {SECRET}. Confirm with the word remembered.",
            missing_counts=missing_counts,
        )
        for i, filler in enumerate(
            (
                "What is 2+2? Reply with just the number.",
                "Name a primary color in one word.",
                "Reply with the word ready.",
            ),
            start=1,
        ):
            reply = await _user_turn(ws, filler, missing_counts=missing_counts)
            print(f"filler turn {i} reply: {reply!r}")

        final = await _user_turn(
            ws,
            "What is the secret code? Reply with only the code.",
            missing_counts=missing_counts,
        )
        print(f"recall reply: {final!r}")
        print(f"missing id field counts: {missing_counts or '{}'}")

        ok = SECRET in final
        return pass_fail(
            ok,
            f"context held ({SECRET} in reply)" if ok else f"context lost; reply={final!r}",
        )
    except Exception as exc:
        return pass_fail(False, f"context-probe error: {exc}")
    finally:
        await ws.close()
        await http.close()


if __name__ == "__main__":
    run_async(main)
