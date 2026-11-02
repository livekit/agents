from __future__ import annotations

import base64
import io

import pytest

from livekit.agents import llm
from livekit.plugins import vlmrun

pytestmark = pytest.mark.plugin("vlmrun")


def _stop_sign_jpeg() -> str:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 40, 260, 200), fill="red")
    draw.text((140, 110), "STOP", fill="white")
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def test_streams_answer_about_image() -> None:
    # qwen/qwen3.5-0.8b is served on the anonymous tier, so this runs without VLMRUN_API_KEY.
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", max_completion_tokens=60)
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(
        role="user",
        content=[
            "Describe this image in one sentence.",
            llm.ImageContent(image=_stop_sign_jpeg()),
        ],
    )

    text = ""
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
    finally:
        await model.aclose()

    lowered = text.lower()
    assert "stop" in lowered or "red" in lowered, text


async def test_greeting_without_user_message() -> None:
    # generate_reply(instructions=...) sends only instructions; the gateway rejects that
    # without a user message.
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", max_completion_tokens=30)
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="Greet the user in one short sentence.")

    text = ""
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
    finally:
        await model.aclose()

    assert text.strip(), "the gateway answered an instructions-only request"


async def test_reply_instructions_after_history() -> None:
    # generate_reply(instructions=...) mid-conversation adds a system message after the
    # history; the gateway only accepts a system message at the start.
    model = vlmrun.LLM(model="qwen/qwen3.5-0.8b", max_completion_tokens=30)
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="You are a helpful assistant.")
    chat_ctx.add_message(role="user", content="Hi there.")
    chat_ctx.add_message(role="assistant", content="Hello!")
    chat_ctx.add_message(role="system", content="Say goodbye in one short sentence.")

    text = ""
    try:
        async with model.chat(chat_ctx=chat_ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
    finally:
        await model.aclose()

    assert text.strip(), "the gateway answered with instructions after the history"
