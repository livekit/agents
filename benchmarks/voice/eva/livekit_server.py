"""EVA-Bench assistant server for LiveKit Agents (``framework: livekit``).

EVA cannot share a venv with livekit-agents (it pins litellm, which pins openai<3), so, like EVA's
Cartesia server bridging to a hosted agent, this server bridges EVA's simulated caller to the agent
under test running as a separate process (``lk_voice_bench.ws_agent``, from the livekit/agents
repo's ``benchmarks/voice``). It is symlinked into EVA as ``eva/assistant/livekit_server.py`` by
``benchmarks/voice/eva/setup.sh``.

Run config::

    framework: livekit
    model: {s2s: livekit, s2s_params: {model: baseline-gemma4, url: ws://127.0.0.1:8765/}}

``model`` names the agent preset; ``url`` defaults to ``$LK_BENCH_AGENT_URL`` or
``ws://127.0.0.1:8765/``.

The bridge does EVA's bookkeeping: it records both tracks at 24 kHz on a wall-clock timeline, runs
tool calls through ``execute_tool`` (the scenario database and audit log), writes each turn the
agent reports (as far as it was spoken) to the audit and framework logs, and measures
``model_response`` latency from the caller's ``user_speech_stop`` to the agent's first audio.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import contextlib
import json
import os
import time
from typing import Any

import numpy as np
import soxr
import uvicorn
import websockets
from eva.assistant.base_server import AbstractAssistantServer
from eva.assistant.pipeline.observers import FrameworkLogWriter, MetricsLogWriter
from eva.utils.audio_utils import (
    create_twilio_media_message,
    mulaw_8k_to_pcm16_24k,
    parse_twilio_media_message,
)
from eva.utils.logging import get_logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = get_logger(__name__)

RECORD_RATE = 24000  # the agent's output rate, and the rate both tracks are recorded at
DEFAULT_AGENT_URL = "ws://127.0.0.1:8765/"


class LiveKitAssistantServer(AbstractAssistantServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        params = self.pipeline_config.s2s_params or {}
        self._model = params.get("model") or "baseline-gemma4"
        self._agent_url = (
            params.get("url") or os.environ.get("LK_BENCH_AGENT_URL") or DEFAULT_AGENT_URL
        )
        self._audio_sample_rate = RECORD_RATE
        self._record_t0: float | None = None

    # -- server lifecycle ---------------------------------------------------------------------

    async def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fw_log = FrameworkLogWriter(self.output_dir)
        self._metrics_log = MetricsLogWriter(self.output_dir)

        app = self._app = FastAPI()

        @app.websocket("/ws")
        async def ws(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        @app.websocket("/")
        async def ws_root(websocket: WebSocket) -> None:
            await websocket.accept()
            await self._handle_session(websocket)

        config = uvicorn.Config(app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.05)
        self._running = True

    async def _shutdown(self) -> None:
        # the base class's stop() calls this, then extracts audio and saves outputs
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._server_task:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task

    # -- recording ----------------------------------------------------------------------------

    def _record(self, buffer: bytearray, pcm: bytes) -> None:
        """Append to a track, first padding it with silence up to the wall-clock position."""
        if self._record_t0 is None:
            self._record_t0 = time.monotonic()
        elapsed = time.monotonic() - self._record_t0 - len(pcm) / 2 / RECORD_RATE
        target = max(0, int(elapsed * RECORD_RATE)) * 2
        if len(buffer) < target:
            buffer.extend(b"\x00" * (target - len(buffer)))
        buffer.extend(pcm)

    # -- one call -----------------------------------------------------------------------------

    async def _handle_session(self, websocket: WebSocket) -> None:  # noqa: C901
        stream_sid = self.conversation_id
        user_speech_stop_ms: int | None = None
        started_segments: set[str] = set()
        down = soxr.ResampleStream(RECORD_RATE, 8000, 1, dtype="int16", quality="HQ")

        async with websockets.connect(self._agent_url, max_size=None) as agent_ws:
            send_lock = asyncio.Lock()

            async def to_agent(payload: dict[str, Any]) -> None:
                async with send_lock:
                    await agent_ws.send(json.dumps(payload))

            await to_agent(
                {
                    "event": "config",
                    "model": self._model,
                    "instructions": self._build_system_prompt(),
                    "tools": [t["function"] for t in self.agent.build_tools_for_agent() or []],
                    # EVA's agents open the call with its canned greeting
                    "greeting": self.initial_message,
                    "call_id": self.conversation_id,
                }
            )
            ready = json.loads(await agent_ws.recv())
            if ready.get("event") != "ready":
                raise RuntimeError(f"agent did not get ready: {ready}")

            async def run_tool(call: dict[str, Any]) -> None:
                try:
                    result = await self.execute_tool(call["name"], call.get("arguments") or {})
                    text = result if isinstance(result, str) else json.dumps(result, default=str)
                    is_error = False
                except Exception as e:  # the agent should hear about it, not hang
                    logger.warning(f"tool {call['name']} failed: {e!r}")
                    text, is_error = f"Error: {e}", True
                await to_agent(
                    {
                        "event": "tool_result",
                        "call_id": call["call_id"],
                        "result": text,
                        "is_error": is_error,
                    }
                )

            def on_message(msg: dict[str, Any]) -> None:
                text = (msg.get("text") or "").strip()
                started = msg.get("started_speaking_at")
                ts_ms = int(started * 1000) if started else int(time.time() * 1000)
                if msg["role"] == "user":
                    if text:
                        self.audit_log.append_user_input(text, timestamp_ms=str(ts_ms))
                    return
                if text:
                    logged = f"{text} [interrupted]" if msg.get("interrupted") else text
                    self.audit_log.append_assistant_output(logged, timestamp_ms=str(ts_ms))
                    self._fw_log.llm_response(text, timestamp_ms=ts_ms)
                    self._fw_log.tts_text(text, timestamp_ms=ts_ms)
                self._fw_log.turn_end(was_interrupted=bool(msg.get("interrupted")))

            async def on_audio(msg: dict[str, Any]) -> None:
                nonlocal user_speech_stop_ms
                pcm = base64.b64decode(msg["payload"])
                segment = msg.get("segment_id")
                if segment not in started_segments:
                    started_segments.add(segment)
                    start_ms = int(msg.get("played_at", time.time()) * 1000)
                    self._fw_log.turn_start(timestamp_ms=start_ms)
                    if user_speech_stop_ms is not None:
                        latency_ms = start_ms - user_speech_stop_ms
                        if 0 < latency_ms < 30_000:
                            self._metrics_log.write_latency(
                                "model_response", latency_ms / 1000, self._model
                            )
                        user_speech_stop_ms = None
                self._record(self.assistant_audio_buffer, pcm)
                pcm8k = down.resample_chunk(np.frombuffer(pcm, dtype=np.int16))
                if len(pcm8k):
                    await websocket.send_text(
                        create_twilio_media_message(
                            stream_sid, audioop.lin2ulaw(pcm8k.tobytes(), 2)
                        )
                    )

            tool_tasks: set[asyncio.Task[None]] = set()

            async def from_agent() -> None:
                async for raw in agent_ws:
                    msg = json.loads(raw)
                    event = msg.get("event")
                    if event == "audio":
                        await on_audio(msg)
                    elif event == "tool_call":
                        task = asyncio.create_task(run_tool(msg))
                        tool_tasks.add(task)
                        task.add_done_callback(tool_tasks.discard)
                    elif event == "message":
                        on_message(msg)

            async def from_caller() -> None:
                nonlocal stream_sid, user_speech_stop_ms
                try:
                    while self._running:
                        raw = await websocket.receive_text()
                        msg = json.loads(raw)
                        event = msg.get("event")
                        if event == "start":
                            stream_sid = msg.get("start", {}).get("streamSid", stream_sid)
                        elif event == "media":
                            ulaw = parse_twilio_media_message(raw)
                            if not ulaw:
                                continue
                            await to_agent(
                                {"event": "media", "payload": base64.b64encode(ulaw).decode()}
                            )
                            self._record(self.user_audio_buffer, mulaw_8k_to_pcm16_24k(ulaw))
                        elif event == "user_speech_stop":
                            if ts := msg.get("timestamp_ms"):
                                user_speech_stop_ms = int(ts)
                        elif event == "stop":
                            break
                except WebSocketDisconnect:
                    pass
                with contextlib.suppress(Exception):
                    await to_agent({"event": "stop"})

            agent_task = asyncio.create_task(from_agent())
            try:
                await from_caller()
                # let the agent flush its last turn events before the socket closes
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(agent_task), timeout=5.0)
            finally:
                agent_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await agent_task
