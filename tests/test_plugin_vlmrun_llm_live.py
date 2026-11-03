from __future__ import annotations

import base64
import io
import os

import pytest

from livekit.agents import llm
from livekit.plugins import vlmrun

pytestmark = [
    pytest.mark.plugin("vlmrun"),
    pytest.mark.skipif(not os.environ.get("VLMRUN_API_KEY"), reason="needs VLMRUN_API_KEY"),
]


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


async def test_decisions_on_the_default_model() -> None:
    from livekit.agents import utils

    async with utils.http_context.open():
        model = vlmrun.DecisionModel()
        chat_ctx = llm.ChatContext.empty()
        chat_ctx.add_message(role="user", content="Please let me talk to a real person.")
        response = await model.evaluate(
            input=chat_ctx,
            decisions={
                "handoff": llm.Probability("The caller requests a human."),
                "intent": llm.Choice(
                    "What does the caller want?",
                    options={"booking": "Make a reservation", "other": "Anything else"},
                ),
                "frustration": llm.Score(
                    "How frustrated is the caller?", levels=["Calm", "Annoyed", "Angry"]
                ),
            },
        )

    assert response.errors == {}, response.errors
    assert response.model == "qwen/qwen3.8-27b"
    assert response.results["handoff"].value > 0.5
    assert response.results["intent"].value == "other"
