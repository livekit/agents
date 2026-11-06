#!/usr/bin/env python3
"""Mid-session nested $ref tools update + response.create probe against xAI realtime."""

from __future__ import annotations

import asyncio
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

NESTED_REF_RAW_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "NESTED_REF_RAW_SCHEMA",
    "description": "Lookup a catalog item by nested reference fields.",
    "parameters": {
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/catalog_item"},
        },
        "required": ["item"],
        "$defs": {
            "catalog_item": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "location": {"$ref": "#/$defs/warehouse_bin"},
                },
                "required": ["sku", "location"],
            },
            "warehouse_bin": {
                "type": "object",
                "properties": {
                    "aisle": {"type": "string"},
                    "shelf": {"type": "string"},
                },
                "required": ["aisle", "shelf"],
            },
        },
    },
}


async def main() -> int:
    api_key = require_xai_api_key()
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
                        "You are a concise text assistant. "
                        "If tools are available, acknowledge them briefly."
                    ),
                    "tools": [NESTED_REF_RAW_SCHEMA],
                    "tool_choice": "auto",
                },
            },
        )

        tool_errors: list[str] = []

        def note_errors(event: dict[str, Any]) -> None:
            msg = event_error_message(event)
            if msg:
                tool_errors.append(msg)
                print(f"server error: {msg}", file=sys.stderr)

        await drain_until(
            ws,
            lambda e: e.get("type") == "session.updated" or event_error_message(e) is not None,
            timeout=30.0,
            on_event=note_errors,
        )
        if tool_errors:
            return pass_fail(False, f"session.update rejected tools: {tool_errors[0]}")

        await send_event(
            ws,
            {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": "Reply with exactly: tools_ok",
                },
            },
        )

        events = await drain_until(
            ws,
            lambda e: e.get("type") in {"response.done", "response.completed"}
            or event_error_message(e) is not None,
            timeout=45.0,
            on_event=note_errors,
        )
        if tool_errors:
            return pass_fail(False, f"response path failed after $ref tools: {tool_errors[0]}")

        types = {e.get("type") for e in events}
        if "response.created" not in types and not any(
            t and str(t).startswith("response.") for t in types
        ):
            return pass_fail(False, "no response.* events after response.create")

        return pass_fail(
            True,
            "mid-session nested $ref tools update accepted; response completed",
        )
    except Exception as exc:
        return pass_fail(False, f"ref-probe error: {exc}")
    finally:
        await ws.close()
        await http.close()


if __name__ == "__main__":
    run_async(main)
