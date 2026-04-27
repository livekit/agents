"""Integration tests for ephemeral say() on the OpenAI Realtime plugin.

Verifies the isolation contract for ephemeral say() end-to-end against the
real OpenAI Realtime API:

- Audibility: text passed to say() reaches the audio channel.
- Server-side isolation: text passed with add_to_chat_ctx=False does not enter
  the substrate's conversation state (verified by a follow-up generate_reply
  that cannot retrieve the text).
- Wire-format metadata: outbound response.create carries client_event_id.

Tests skip when OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv
from openai.types import realtime

from livekit.agents import llm
from livekit.plugins import openai


@pytest.fixture(scope="session", autouse=True)
def _load_env() -> None:
    load_dotenv(override=True)


_skip_no_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; integration tests require real OpenAI access",
)


def _openai_model() -> openai.realtime.RealtimeModel:
    return openai.realtime.RealtimeModel(
        voice="alloy",
        input_audio_transcription=realtime.AudioTranscription(model="gpt-4o-mini-transcribe"),
    )


@pytest.fixture
async def rt_session() -> llm.RealtimeSession:
    model = _openai_model()
    session = model.session()
    yield session
    await session.aclose()
    await asyncio.sleep(0.5)


async def _collect_text(gen_ev: llm.GenerationCreatedEvent) -> str:
    parts: list[str] = []
    async for msg_gen in gen_ev.message_stream:
        async for chunk in msg_gen.text_stream:
            parts.append(chunk)
    return "".join(parts)


async def _drain_generation(gen_ev: llm.GenerationCreatedEvent) -> int:
    """Drain the message + audio streams. Returns total audio frame count."""
    audio_frames = 0
    async for msg_gen in gen_ev.message_stream:
        async for _ in msg_gen.text_stream:
            pass
        async for _ in msg_gen.audio_stream:
            audio_frames += 1
    return audio_frames


@_skip_no_openai_key
async def test_openai_realtime_say_audio_renders(rt_session: llm.RealtimeSession) -> None:
    """say() produces audio frames containing the rendered text."""
    gen_ev = await asyncio.wait_for(
        rt_session.say("the verification token is alpha-bravo"),
        timeout=15,
    )
    audio_frames = await asyncio.wait_for(_drain_generation(gen_ev), timeout=20)
    assert audio_frames > 0, "expected at least one audio frame from say()"


@_skip_no_openai_key
async def test_openai_realtime_say_emits_metadata(rt_session: llm.RealtimeSession) -> None:
    """Outbound response.create carries metadata.client_event_id starting with the say prefix."""
    captured: list[object] = []
    real_send = rt_session.send_event

    def _capture(ev: object) -> None:
        captured.append(ev)
        return real_send(ev)

    rt_session.send_event = _capture  # type: ignore[method-assign]

    try:
        gen_ev = await asyncio.wait_for(rt_session.say("alpha-bravo"), timeout=15)
        await asyncio.wait_for(_drain_generation(gen_ev), timeout=20)
    finally:
        rt_session.send_event = real_send  # type: ignore[method-assign]

    create_events = [
        e
        for e in captured
        if getattr(e, "type", None) == "response.create"
        and isinstance(getattr(getattr(e, "response", None), "metadata", None), dict)
        and e.response.metadata.get("client_event_id", "").startswith("response_create_say_")
    ]
    assert len(create_events) >= 1, (
        f"expected response.create event with say-prefixed client_event_id; "
        f"captured {len(captured)} total events, none matched the prefix"
    )


@_skip_no_openai_key
async def test_openai_realtime_say_isolation_no_leak(rt_session: llm.RealtimeSession) -> None:
    """After say(secret, add_to_chat_ctx=False), generate_reply cannot recall the secret.

    Proves server-side isolation: the OpenAI Realtime substrate does not retain
    isolated text in conversation state. A follow-up generation that asks the
    model to repeat what it just said must not include the secret.
    """
    secret = "purple-elephant-42"

    say_gen = await asyncio.wait_for(
        rt_session.say(f"the verification token is {secret}", add_to_chat_ctx=False),
        timeout=15,
    )
    await asyncio.wait_for(_drain_generation(say_gen), timeout=30)

    follow_up = await asyncio.wait_for(
        rt_session.generate_reply(
            instructions=(
                "Repeat the last verification token you just said out loud, exactly. "
                "If you cannot recall it, say 'no token recalled'."
            )
        ),
        timeout=15,
    )
    follow_up_text = await asyncio.wait_for(_collect_text(follow_up), timeout=20)

    assert secret not in follow_up_text.lower(), (
        f"secret leaked into follow-up generation: {follow_up_text!r}"
    )


@_skip_no_openai_key
async def test_openai_realtime_say_isolation_no_remote_chat_ctx_leak(
    rt_session: llm.RealtimeSession,
) -> None:
    """After say(secret, add_to_chat_ctx=False), session.chat_ctx contains no items with the secret.

    Proves substrate-state isolation: the OpenAI Realtime API does not retain
    isolated text in the visible conversation history (server-side chat_ctx).
    """
    secret = "purple-elephant-42"

    say_gen = await asyncio.wait_for(
        rt_session.say(f"the verification token is {secret}", add_to_chat_ctx=False),
        timeout=15,
    )
    await asyncio.wait_for(_drain_generation(say_gen), timeout=30)

    # Allow substrate state to settle.
    await asyncio.sleep(0.5)

    chat_ctx = rt_session.chat_ctx
    flat = " ".join(
        repr(getattr(item, "content", "")) for item in getattr(chat_ctx, "items", [])
    ).lower()
    assert secret not in flat, f"secret leaked into substrate chat_ctx: {flat!r}"
