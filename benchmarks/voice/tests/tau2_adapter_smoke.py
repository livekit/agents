"""Drive LiveKitAgentsAdapter the way tau2's DiscreteTimeAudioNativeAgent does, without tau2's
(paid) caller simulator: synthesized caller turns, tool calls answered locally.

    benchmarks/voice/.venv-tau2/bin/python benchmarks/voice/tests/tau2_adapter_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import time

from lk_voice_bench.codec import pcm16_to_ulaw
from lk_voice_bench.tau2_adapter import LiveKitAgentsAdapter

from livekit import rtc
from livekit.agents import inference, utils

RATE = 8000
TICK_MS = 200
TICK_BYTES = RATE * TICK_MS // 1000  # mu-law: 1 byte per sample
SILENCE = b"\xff" * TICK_BYTES


class FakeTool:
    def __init__(self, schema: dict) -> None:
        self.openai_schema = {"type": "function", "function": schema}


TOOLS = [
    FakeTool(
        {
            "name": "get_user_details",
            "description": "Get the details of a user, including their reservations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID, such as 'sara_doe_496'.",
                    }
                },
                "required": ["user_id"],
            },
        }
    ),
    FakeTool(
        {
            "name": "get_reservation_details",
            "description": "Get the details of a reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "string", "description": "The reservation ID."}
                },
                "required": ["reservation_id"],
            },
        }
    ),
]


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    if name == "get_user_details":
        if "mia" not in args.get("user_id", "").lower():
            return "Error: user not found", True
        return json.dumps(
            {"user_id": "mia_li_3668", "name": "Mia Li", "reservations": ["NO6JO3"]}
        ), False
    if name == "get_reservation_details":
        return json.dumps(
            {
                "reservation_id": args.get("reservation_id"),
                "origin": "JFK",
                "destination": "SEA",
                "flight_date": "2026-12-14",
                "cabin": "business",
                "status": "active",
            }
        ), False
    return "Error: unknown tool", True


async def synth(text: str) -> bytes:
    async with utils.http_context.open():
        tts = inference.TTS("cartesia/sonic-3", voice="a167e0f3-df7e-4d52-a9c3-f949145efdab")
        resampler, out = None, bytearray()
        async for ev in tts.synthesize(text):
            if resampler is None:
                resampler = rtc.AudioResampler(ev.frame.sample_rate, RATE, num_channels=1)
            for r in resampler.push(ev.frame):
                out += r.data.cast("B")
        for r in resampler.flush():
            out += r.data.cast("B")
        await tts.aclose()
    return pcm16_to_ulaw(bytes(out))


def ticks(audio: bytes) -> list[bytes]:
    chunks = [audio[i : i + TICK_BYTES] for i in range(0, len(audio), TICK_BYTES)]
    chunks[-1] = chunks[-1].ljust(TICK_BYTES, b"\xff")
    return chunks


def main() -> None:
    async def synth_all() -> list[bytes]:
        return list(
            await asyncio.gather(
                synth(
                    "Hi, I want to check on my upcoming flight. My user ID is mia underscore li "
                    "underscore three six six eight."
                ),
                synth("Yes, that one please. Where does it go?"),
            )
        )

    turns = asyncio.run(synth_all())
    # tau2 passes the tick as a float (tick_duration_seconds * 1000)
    adapter = LiveKitAgentsAdapter(tick_duration_ms=TICK_MS / 1000 * 1000.0)
    adapter.set_opening_message("Hi! How can I help you today?")
    adapter.connect(
        "You are an airline agent on a phone call. Look up the user's details and reservations "
        "with your tools before answering. Be brief.",
        TOOLS,
        None,
    )

    script = [SILENCE] * 3 + ticks(turns[0])
    t0 = time.time()
    agent_silent_ticks, agent_spoke, second_sent = 0, False, False
    transcript: list[str] = []
    for n in range(1, 400):
        user = script.pop(0) if script else SILENCE
        res = adapter.run_tick(user, n)
        for tc in res.tool_calls:
            out, err = run_tool(tc.name, tc.arguments)
            print(
                f"[{time.time() - t0:5.1f}s tick {n}] tool {tc.name}({tc.arguments}) -> err={err}"
            )
            adapter.send_tool_result(tc.id, out, is_error=err)
        if res.proportional_transcript:
            transcript.append(res.proportional_transcript)
        if res.vad_events:
            print(f"[{time.time() - t0:5.1f}s tick {n}] vad {res.vad_events}")
        played = len(res.get_played_agent_audio())
        assert played == TICK_BYTES, played
        if res.agent_audio_bytes:
            agent_spoke, agent_silent_ticks = True, 0
        elif agent_spoke:
            agent_silent_ticks += 1
        # after the agent has finished a turn (1.5 s of quiet), say the next line
        if agent_spoke and agent_silent_ticks == 8 and not script:
            if second_sent:
                break
            print(f"[{time.time() - t0:5.1f}s] agent said: {' '.join(transcript)!r}")
            transcript = []
            script = ticks(turns[1])
            second_sent, agent_spoke = True, False
    print(f"[{time.time() - t0:5.1f}s] agent said: {' '.join(transcript)!r}")
    elapsed = time.time() - t0
    print(f"{n} ticks in {elapsed:.1f}s ({elapsed / n * 1000:.0f} ms/tick)")
    adapter.disconnect()


if __name__ == "__main__":
    main()
