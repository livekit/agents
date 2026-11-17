"""Shared agent definition; every execution has a deterministic inference provider."""

import os

from livekit.agents import Agent
from livekit.agents.llm import ChatContext
from tests.fakes import FakeLLM, last_input


def answer(history: ChatContext) -> str:
    text = last_input(history) or ""
    if text == "recall":
        return next(
            item.text_content or ""
            for item in history.items
            if item.type == "message" and item.role == "user"
        )
    return {
        "hello there agent": os.environ.get("GREETING", "hello, how can I help you"),
        "what time is it": "it is noon",
    }.get(text, "noted")


def agent() -> Agent:
    return Agent(name="desk", instructions="Remember the conversation.", llm=FakeLLM(answer))
