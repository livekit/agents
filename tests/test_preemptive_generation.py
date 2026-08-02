from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, EndpointingOptions, InterruptionOptions
from livekit.agents.llm import ChatContext, ChatMessage, LLMStream, Tool, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeUserSpeech
from .fake_tts import FakeTTS
from .fake_turn_stt import ScriptedTurn, TurnScriptedSTT
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class RecordingFakeLLM(FakeLLM):
    def __init__(self, *, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.inputs: list[str] = []

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        self.inputs.append(chat_ctx.items[-1].text_content or "")
        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


class FinalizingAgent(Agent):
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        new_message.content = ["book a room!"]
        new_message.transcript_confidence = 0.42


async def _run_soniox_event_order(
    *, preflight_text: str, final_text: str, agent: Agent | None = None
) -> tuple[list[str], list[ChatMessage]]:
    speech = FakeUserSpeech(
        start_time=0.1,
        end_time=0.6,
        transcript=final_text,
        stt_delay=0.0,
    )
    stt = TurnScriptedSTT(
        turns=[
            ScriptedTurn(
                speech_start=0.15,
                eager_at=0.35,
                eager_text=preflight_text,
                final_at=0.65,
                final_text=final_text,
            )
        ]
    )
    llm = RecordingFakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=preflight_text,
                content="speculative answer",
                ttft=0.02,
                duration=0.1,
            ),
            FakeLLMResponse(
                input=final_text,
                content="committed answer",
                ttft=0.02,
                duration=0.1,
            ),
        ]
    )
    session = AgentSession[None](
        vad=FakeVAD(fake_user_speeches=[speech]),
        stt=stt,
        llm=llm,
        tts=FakeTTS(fake_audio_duration=0.1),
        turn_handling={
            "turn_detection": "stt",
            "endpointing": EndpointingOptions(min_delay=0.2, max_delay=1.0),
            "interruption": InterruptionOptions(
                mode="vad",
                resume_false_interruption=False,
            ),
            "preemptive_generation": {
                "enabled": True,
                "preemptive_tts": False,
            },
        },
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    agent = agent or Agent(instructions="You are a helpful assistant.")
    user_messages: list[ChatMessage] = []

    try:
        await session.start(agent)
        audio_input.push(0.1)
        await asyncio.wait_for(stt.turns_done, timeout=5.0)
        await asyncio.sleep(2.0)
        await session.drain()
        user_messages = [
            item
            for item in agent.chat_ctx.items
            if isinstance(item, ChatMessage) and item.role == "user"
        ]
    finally:
        await session.aclose()

    return llm.inputs, user_messages


async def test_formatting_only_final_reuses_preemptive_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")

    inputs, user_messages = await _run_soniox_event_order(
        preflight_text="BOOK   A ROOM",
        final_text="book a room.",
    )

    assert inputs == ["BOOK   A ROOM"]
    assert len(user_messages) == 1
    assert user_messages[0].raw_text_content == "book a room."
    assert user_messages[0].transcript_confidence == 0.9
    assert "using preemptive generation" in caplog.text


async def test_reused_preemptive_generation_keeps_user_turn_callback_edits() -> None:
    inputs, user_messages = await _run_soniox_event_order(
        preflight_text="BOOK   A ROOM",
        final_text="book a room.",
        agent=FinalizingAgent(instructions="You are a helpful assistant."),
    )

    assert inputs == ["BOOK   A ROOM"]
    assert len(user_messages) == 1
    assert user_messages[0].raw_text_content == "book a room!"
    assert user_messages[0].transcript_confidence == 0.42


async def test_changed_words_invalidate_preemptive_generation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")

    inputs, user_messages = await _run_soniox_event_order(
        preflight_text="book a",
        final_text="book a room",
    )

    assert inputs == ["book a", "book a room"]
    assert len(user_messages) == 1
    assert user_messages[0].raw_text_content == "book a room"
    assert "transcript, chat context, tools, or tool choice changed" in caplog.text
    assert "using preemptive generation" not in caplog.text
