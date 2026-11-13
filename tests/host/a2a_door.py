"""The A2A text door served in process on the worker's loopback listener."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from livekit.agents import AgentServer

PROJECT_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_AGENT_NAME",
    "LIVEKIT_AGENT_NAME_OVERRIDE",
    "LIVEKIT_AGENT_DEPLOYMENT",
    "LIVEKIT_AGENT_VERSION",
)


@contextlib.asynccontextmanager
async def serving(server: AgentServer) -> AsyncIterator[str]:
    """``http://127.0.0.1:<port>/agents/<agent>/<deployment>`` while ``server`` runs.

    `lk agent dev` is the only public way to serve; this runs the same server loop in process,
    unregistered, with its listener on a free port.
    """
    server._update_options(listeners={"health": {"port": 0}})
    started = asyncio.Event()
    server._on_started.append(started.set)
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    listening = asyncio.create_task(started.wait())
    try:
        await asyncio.wait({run, listening}, return_when=asyncio.FIRST_COMPLETED)
        if run.done():
            run.result()
        yield server._config.delegation_url("").rstrip("/")
    finally:
        listening.cancel()
        await server.aclose()
        await run


def message_send(
    text: str, message_id: str = "m1", *, context_id: str | None = None
) -> dict[str, object]:
    """The JSON-RPC body of an A2A ``message/send`` of ``text``."""
    message: dict[str, object] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id is not None:
        message["contextId"] = context_id
    return {"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": message}}
