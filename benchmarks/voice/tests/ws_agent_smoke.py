"""Drive lk_voice_bench.ws_agent like EVA's bridge does, with a synthesized caller and a fake tool.

cd benchmarks/voice && ../../.venv/bin/python -m lk_voice_bench.ws_agent --port 8765 &
../../.venv/bin/python tests/ws_agent_smoke.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import time

import aiohttp
from lk_voice_bench.codec import pcm16_to_ulaw

from livekit import rtc
from livekit.agents import inference, utils

TOOLS = [
    {
        "name": "get_flight_status",
        "description": "Get the status of a flight by flight number.",
        "parameters": {
            "type": "object",
            "properties": {"flight_number": {"type": "string"}},
            "required": ["flight_number"],
        },
    }
]


async def synth(text: str) -> bytes:
    tts = inference.TTS("cartesia/sonic-3", voice="a167e0f3-df7e-4d52-a9c3-f949145efdab")
    resampler, out = None, bytearray()
    async for ev in tts.synthesize(text):
        if resampler is None:
            resampler = rtc.AudioResampler(ev.frame.sample_rate, 8000, num_channels=1)
        for r in resampler.push(ev.frame):
            out += r.data.cast("B")
    for r in resampler.flush():
        out += r.data.cast("B")
    await tts.aclose()
    return pcm16_to_ulaw(bytes(out))


async def main() -> None:
    async with utils.http_context.open():
        caller = await synth("Hi, is flight UA 2 4 1 on time today?")
    t0 = time.time()
    audio_s: dict[str, float] = {}
    events: list[str] = []
    async with aiohttp.ClientSession() as http, http.ws_connect("ws://127.0.0.1:8765/") as ws:
        await ws.send_json(
            {
                "event": "config",
                "model": "baseline-gemma4",
                "instructions": "You are an airline agent on a call. Use tools. Be brief.",
                "tools": TOOLS,
                "greeting": "Thanks for calling. How can I help?",
                "call_id": "ws-smoke",
            }
        )

        async def send_caller() -> None:
            await asyncio.sleep(3.0)  # let the greeting play
            for i in range(0, len(caller), 160):
                await ws.send_json(
                    {"event": "media", "payload": base64.b64encode(caller[i : i + 160]).decode()}
                )
                await asyncio.sleep(0.02)
            silence = base64.b64encode(b"\xff" * 160).decode()
            for _ in range(500):
                await ws.send_json({"event": "media", "payload": silence})
                await asyncio.sleep(0.02)
            await ws.send_json({"event": "stop"})

        sender = asyncio.create_task(send_caller())
        async for msg in ws:
            data = json.loads(msg.data)
            ev = data["event"]
            if ev == "audio":
                seg = data["segment_id"]
                audio_s[seg] = audio_s.get(seg, 0) + len(base64.b64decode(data["payload"])) / 48000
            elif ev == "tool_call":
                events.append(
                    f"{time.time() - t0:5.1f}s tool_call {data['name']} {data['arguments']}"
                )
                await ws.send_json(
                    {
                        "event": "tool_result",
                        "call_id": data["call_id"],
                        "result": json.dumps({"status": "delayed", "new_departure": "6:40 PM"}),
                        "is_error": False,
                    }
                )
            elif ev != "audio":
                events.append(f"{time.time() - t0:5.1f}s {ev} {json.dumps(data)[:160]}")
        await sender
    print("\n".join(events))
    print("audio seconds per segment:", {k: round(v, 2) for k, v in audio_s.items()})


asyncio.run(main())
