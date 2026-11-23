"""The agent under test as a websocket service, for benchmarks that cannot load it in-process.

    uv run python -m lk_voice_bench.ws_agent --port 8765

EVA pins dependencies that livekit-agents cannot share a venv with, so its ``livekit`` framework
bridge (``benchmarks/voice/eva/livekit_server.py``) talks to this process instead, the way EVA's
Cartesia server talks to a hosted agent. Each websocket connection is one call with its own
``BenchAgent``; the speaker plays on its own real-time clock and every frame is sent as it plays.

Protocol (JSON text frames):

client -> agent
    {"event": "config", "model": "<preset>", "instructions": str, "tools": [function schema],
     "greeting": str | null}                       first message; greeting is spoken
    {"event": "media", "payload": b64}             caller audio, 8 kHz mu-law
    {"event": "tool_result", "call_id": str, "result": str, "is_error": bool}
    {"event": "stop"}

agent -> client
    {"event": "ready"}
    {"event": "audio", "payload": b64, "segment_id": str, "played_at": float}
                                                   agent audio as played, 24 kHz PCM16
    {"event": "tool_call", "call_id": str, "name": str, "arguments": dict}
    {"event": "message", "role": "user" | "assistant", "text": str, "interrupted": bool,
     "started_speaking_at": float | null}           a committed turn, as far as it was spoken
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from livekit.agents.llm import ToolError
from livekit.agents.utils import http_context

from .audio_io import PlayedChunk
from .codec import ulaw_to_pcm16
from .config import get_preset
from .session import BenchAgent

logger = logging.getLogger("lk_voice_bench.ws_agent")

OUTPUT_RATE = 24000
CALLER_RATE = 8000
TOOL_RESULT_TIMEOUT = 120.0


async def _handle_call(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    log_dir: Path | None = request.app["log_dir"]
    pending: dict[str, asyncio.Future[tuple[str, bool]]] = {}
    loop = asyncio.get_running_loop()
    # one queue and one sender keep the events in the order they happened
    outbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def send(payload: dict[str, Any]) -> None:
        outbox.put_nowait(payload)

    async def _sender() -> None:
        while (payload := await outbox.get()) is not None:
            if not ws.closed:
                await ws.send_str(json.dumps(payload))

    def on_played(chunk: PlayedChunk) -> None:
        send(
            {
                "event": "audio",
                "payload": base64.b64encode(chunk.pcm).decode(),
                "segment_id": chunk.segment_id,
                "played_at": chunk.played_at,
            }
        )

    async def execute(name: str, arguments: dict[str, Any]) -> str:
        call_id = f"call_{uuid.uuid4().hex[:16]}"
        fut: asyncio.Future[tuple[str, bool]] = loop.create_future()
        pending[call_id] = fut
        send({"event": "tool_call", "call_id": call_id, "name": name, "arguments": arguments})
        try:
            result, is_error = await asyncio.wait_for(fut, timeout=TOOL_RESULT_TIMEOUT)
        finally:
            pending.pop(call_id, None)
        if is_error:
            raise ToolError(result)
        return result

    first = await ws.receive()
    if first.type != WSMsgType.TEXT or (config := json.loads(first.data)).get("event") != "config":
        await ws.close(message=b"expected a config message first")
        return ws

    agent = BenchAgent(
        get_preset(config.get("model") or "baseline-gemma4"),
        instructions=config["instructions"],
        tools=config.get("tools") or [],
        tool_executor=execute,
        output_sample_rate=OUTPUT_RATE,
        on_played=on_played,
    )

    @agent.session.on("conversation_item_added")
    def _on_item(ev: Any) -> None:
        item = ev.item
        if getattr(item, "type", None) != "message" or item.role not in ("user", "assistant"):
            return
        send(
            {
                "event": "message",
                "role": item.role,
                "text": item.text_content or "",
                "interrupted": bool(item.interrupted),
                "started_speaking_at": (item.metrics or {}).get("started_speaking_at"),
            }
        )

    sender = asyncio.create_task(_sender(), name="ws-agent-sender")
    await agent.start()
    send({"event": "ready"})
    if greeting := config.get("greeting"):
        agent.session.say(greeting)

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            event = data.get("event")
            if event == "media":
                agent.push_caller_audio(
                    ulaw_to_pcm16(base64.b64decode(data["payload"])), CALLER_RATE
                )
            elif event == "tool_result":
                fut = pending.get(data["call_id"])
                if fut is not None and not fut.done():
                    fut.set_result((data.get("result", ""), bool(data.get("is_error"))))
            elif event == "stop":
                break
    finally:
        for fut in pending.values():
            if not fut.done():
                fut.cancel()
        await agent.aclose()
        outbox.put_nowait(None)
        await sender
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            agent.dump_log(str(log_dir / f"{config.get('call_id') or uuid.uuid4().hex}.jsonl"))
        if not ws.closed:
            await ws.close()
    return ws


async def _serve(host: str, port: int, log_dir: Path | None) -> None:
    async with http_context.open():
        app = web.Application()
        app["log_dir"] = log_dir
        app.router.add_get("/", _handle_call)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        logger.info("lk_voice_bench agent listening on ws://%s:%d/", host, port)
        await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the agent under test over a websocket.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", type=Path, default=None, help="per-call agent session logs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_serve(args.host, args.port, args.log_dir))


if __name__ == "__main__":
    main()
