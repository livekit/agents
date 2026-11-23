"""Live smoke test of BenchAgent: synthetic caller speech in, agent speech + tool calls out.

Uses the LiveKit gateway (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET). Not a unit test.

    uv run python benchmarks/voice/tests/live_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lk_voice_bench import BenchAgent, get_preset  # noqa: E402

from livekit import rtc  # noqa: E402
from livekit.agents import inference, utils  # noqa: E402

RATE = 8000
FRAME = RATE // 50 * 2  # 20 ms of int16


async def synth(text: str) -> bytes:
    tts = inference.TTS("cartesia/sonic-3", voice="a167e0f3-df7e-4d52-a9c3-f949145efdab")
    resampler = None
    out = bytearray()
    async for ev in tts.synthesize(text):
        f = ev.frame
        if resampler is None:
            resampler = rtc.AudioResampler(f.sample_rate, RATE, num_channels=1)
        for r in resampler.push(f):
            out += r.data.cast("B")
    for r in resampler.flush():
        out += r.data.cast("B")
    await tts.aclose()
    return bytes(out)


async def main() -> None:
    calls: list[tuple[str, dict]] = []

    async def executor(name: str, args: dict) -> str:
        calls.append((name, args))
        await asyncio.sleep(0.3)
        return json.dumps(
            {
                "reservation_id": args.get("reservation_id"),
                "origin": "SFO",
                "destination": "JFK",
                "date": "2026-12-03",
                "passengers": [{"first_name": "Ana", "last_name": "Lopez"}],
                "cabin": "economy",
                "status": "confirmed",
            }
        )

    tools = [
        {
            "name": "get_reservation_details",
            "description": "Get the details of a reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "The reservation ID, such as '8JX2WO'.",
                    }
                },
                "required": ["reservation_id"],
            },
        }
    ]
    u1 = await synth(
        "Hi, can you tell me where my flight goes? My reservation code is A B C 1 2 3."
    )
    u2 = await synth("Sorry to cut in, what date is that flight?")

    agent = BenchAgent(
        get_preset("baseline-gemma4"),
        instructions=(
            "You are an airline customer service agent on a phone call. Be concise. Use tools to "
            "look up reservations."
        ),
        tools=tools,
        tool_executor=executor,
        greeting="Hi! How can I help you today?",
    )
    await agent.start()
    t0 = time.time()
    silence = b"\x00" * FRAME
    loop = asyncio.get_running_loop()

    # script: 0.5 s silence, u1, then silence until agent has spoken ~1.5 s, then barge in with u2
    script = [silence] * 25 + [u1[i : i + FRAME] for i in range(0, len(u1), FRAME)]
    barged = False
    next_at = loop.time()
    played_s = {}
    texts = []
    deadline = t0 + 40
    while time.time() < deadline:
        frame = script.pop(0) if script else silence
        agent.push_caller_audio(frame, RATE)
        for c in agent.speaker.drain_played():
            played_s[c.segment_id] = played_s.get(c.segment_id, 0) + len(c.pcm) / (RATE * 2)
            if not barged and len(played_s) >= 2 and played_s[c.segment_id] > 1.5:
                barged = True
                script = [u2[i : i + FRAME] for i in range(0, len(u2), FRAME)]
                print(f"[{time.time() - t0:5.1f}s] >>> barge-in on {c.segment_id}")
        for st in agent.spoken_text.drain():
            texts.append((round(st.at - t0, 2), st.segment_id, st.text))
        for it in agent.speaker.drain_interruptions():
            print(
                f"[{it.at - t0:5.1f}s] interrupted {it.segment_id} after {it.played_s:.2f}s, "
                f"dropped {it.dropped_bytes} bytes"
            )
        next_at += 0.02
        await asyncio.sleep(max(0, next_at - loop.time()))

    await agent.aclose()
    print("tool calls:", calls)
    print("played per segment:", {k: round(v, 2) for k, v in played_s.items()})
    by_seg: dict = {}
    for _at, seg, txt in texts:
        by_seg.setdefault(seg, []).append(txt)
    for seg, parts in by_seg.items():
        print(f"spoken[{seg}]:", "".join(parts))
    for ev in agent.log.events:
        if ev["kind"] in ("user_transcript", "message", "tool_call", "error"):
            print(
                f"  {ev['t'] - t0:5.1f}s {ev['kind']}: "
                + json.dumps({k: v for k, v in ev.items() if k not in ("t", "kind", "metrics")})[
                    :220
                ]
            )
    agent.dump_log(
        "/tmp/claude-0/-home-user-agents/b93dc044-ddf0-541c-bb23-223c203f2907/scratchpad/smoke/session_log.jsonl"
    )


async def _run() -> None:
    async with utils.http_context.open():
        await main()


asyncio.run(_run())
