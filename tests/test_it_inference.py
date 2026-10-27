"""The LiveKit Inference clients (the core's gateway STT, TTS and LLM) against the real gateway:
what its protocol, auth and audio formats accept, which no fake of the gateway can tell.

Needs LIVEKIT_INFERENCE_API_KEY and LIVEKIT_INFERENCE_API_SECRET (or LIVEKIT_API_KEY and
LIVEKIT_API_SECRET), and LIVEKIT_INFERENCE_URL when not the cloud gateway's.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.plugin("inference")

_core = pytest.importorskip("livekit.agents._core", reason="needs livekit-agents-next")

from livekit import rtc  # noqa: E402
from livekit.agents import (  # noqa: E402
    APIConnectOptions,
    APIError,
    function_tool,
    inference,
    llm,
    stt,
)

if not (
    (os.environ.get("LIVEKIT_INFERENCE_API_KEY") and os.environ.get("LIVEKIT_INFERENCE_API_SECRET"))
    or (os.environ.get("LIVEKIT_API_KEY") and os.environ.get("LIVEKIT_API_SECRET"))
):
    pytest.skip("needs LiveKit Inference credentials", allow_module_level=True)

SENTENCE = "The quick brown fox jumps over the lazy dog."
TTS_MODEL = "cartesia/sonic-3"
STT_MODEL = "deepgram/nova-3"
LLM_MODEL = "openai/gpt-4.1-mini"


async def synthesize(tts: inference.TTS, text: str) -> list[rtc.AudioFrame]:
    stream = tts.stream()
    stream.push_text(text)
    stream.end_input()
    frames = [ev.frame async for ev in stream]
    await stream.aclose()
    return frames


async def transcribe(recognizer: inference.STT, frames: list[rtc.AudioFrame]) -> str:
    stream = recognizer.stream()
    for frame in frames:
        stream.push_frame(frame)
    # trailing silence so the provider finalizes
    rate = frames[0].sample_rate
    silence = rtc.AudioFrame(b"\x00\x00" * (rate // 10), rate, 1, rate // 10)
    for _ in range(15):
        stream.push_frame(silence)
    stream.end_input()
    finals = [
        ev.alternatives[0].text
        async for ev in stream
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives
    ]
    await stream.aclose()
    return " ".join(finals).lower()


async def test_tts_to_stt_round_trip() -> None:
    """Guards the gateway STT and TTS streams (auth, the session handshake, the audio encodings
    and sample rates each side declares) and a second synthesis on the same client: speech
    synthesized by one must be transcribed back by the other."""
    tts = inference.TTS(TTS_MODEL)
    recognizer = inference.STT(STT_MODEL, language="en")
    try:
        frames = await synthesize(tts, SENTENCE)
        assert frames, "the gateway returned no audio"
        assert {f.sample_rate for f in frames} == {tts.sample_rate}
        assert sum(f.duration for f in frames) > 1.0

        # a second synthesis on the same client: the connection it reuses must still work
        again = await synthesize(tts, "Hello there.")
        assert sum(f.duration for f in again) > 0.3

        text = await transcribe(recognizer, frames)
        assert "fox" in text and "lazy dog" in text, text
    finally:
        await tts.aclose()
        await recognizer.aclose()


async def test_llm_tool_call_round_trip() -> None:
    """Guards the gateway LLM's request and stream formats for tools: a declared tool must come
    back as a call with parsed arguments, and its output must be accepted on the next turn."""

    @function_tool
    async def get_weather(city: str) -> str:
        """Look up the current weather in a city."""
        return "sunny and 21 degrees"

    model = inference.LLM(LLM_MODEL)
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="Always use the tools you have to answer.")
    ctx.add_message(role="user", content="What's the weather in Paris right now?")
    try:
        calls: list[llm.FunctionToolCall] = []
        async with model.chat(chat_ctx=ctx, tools=[get_weather]) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.tool_calls:
                    calls.extend(chunk.delta.tool_calls)
        assert [c.name for c in calls] == ["get_weather"], calls
        assert "paris" in calls[0].arguments.lower(), calls[0].arguments

        call = calls[0]
        ctx.items.append(
            llm.FunctionCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
        )
        ctx.items.append(
            llm.FunctionCallOutput(
                call_id=call.call_id, name=call.name, output="sunny and 21 degrees", is_error=False
            )
        )
        answer = ""
        async with model.chat(chat_ctx=ctx, tools=[get_weather]) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    answer += chunk.delta.content
        assert "sunny" in answer.lower() or "21" in answer, answer
    finally:
        await model.aclose()


@pytest.mark.parametrize("leg", ["stt", "tts", "llm"])
async def test_bad_credentials_fail_fast(leg: str) -> None:
    """Guards a rejected API key hanging a call or retrying forever: the client must raise an
    APIError within its connect timeout."""
    creds = {"api_key": "it-not-a-key", "api_secret": "it-not-a-secret"}
    conn = APIConnectOptions(max_retry=0, timeout=10)
    started = time.monotonic()
    with pytest.raises(APIError):
        async with asyncio.timeout(30):
            if leg == "stt":
                recognizer = inference.STT(STT_MODEL, conn_options=conn, **creds)
                silence = rtc.AudioFrame(b"\x00\x00" * 4800, 48000, 1, 4800)
                await transcribe(recognizer, [silence])
            elif leg == "tts":
                await synthesize(inference.TTS(TTS_MODEL, conn_options=conn, **creds), SENTENCE)
            else:
                ctx = llm.ChatContext.empty()
                ctx.add_message(role="user", content="hi")
                async with inference.LLM(LLM_MODEL, **creds).chat(
                    chat_ctx=ctx, conn_options=conn
                ) as stream:
                    async for _ in stream:
                        pass
    assert time.monotonic() - started < 20
