from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    Optional,
    Union,
)

from livekit import rtc

from .. import llm, stt, tokenize, tts, utils, vad
from ..llm import (
    AIFunction,
    ChatContext,
    FunctionContext,
    find_ai_functions,
)
from ..types import NOT_GIVEN, NotGivenOr
from .audio_recognition import _TurnDetector

if TYPE_CHECKING:
    from .pipeline_agent import PipelineAgent
    from .task_activity import TaskActivity


class AgentTask:
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        ai_functions: list[llm.AIFunction] = [],
        turn_detector: NotGivenOr[_TurnDetector | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
    ) -> None:
        if tts and not tts.capabilities.streaming:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        if stt and not stt.capabilities.streaming:
            from .. import stt as speech_to_text

            if not vad:
                raise ValueError(
                    "VAD is required when streaming is not supported by the STT"
                )

            stt = speech_to_text.StreamAdapter(
                stt=stt,
                vad=vad,
            )

        self._instructions = instructions
        self._chat_ctx = chat_ctx or ChatContext.empty()
        self._fnc_ctx = FunctionContext(ai_functions + find_ai_functions(self))
        self._eou = turn_detector
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._vad = vad
        self._activity: TaskActivity | None = None

    @property
    def instructions(self) -> str:
        return self._instructions

    @property
    def ai_functions(self) -> list[llm.AIFunction]:
        return list(self._fnc_ctx.ai_functions.values())

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def turn_detector(self) -> NotGivenOr[_TurnDetector | None]:
        return self._eou

    @property
    def stt(self) -> NotGivenOr[stt.STT | None]:
        return self._stt

    @property
    def llm(self) -> NotGivenOr[llm.LLM | llm.RealtimeModel | None]:
        return self._llm

    @property
    def tts(self) -> NotGivenOr[tts.TTS | None]:
        return self._tts

    @property
    def vad(self) -> NotGivenOr[vad.VAD | None]:
        return self._vad

    @property
    def agent(self) -> PipelineAgent:
        """
        Retrieve the PipelineAgent associated with the current task;.

        Raises:
            RuntimeError: If the task is not running
        """
        return self.__get_activity_or_raise().agent

    async def on_enter(self) -> None:
        pass

    async def on_exit(self) -> None:
        pass

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame]
    ) -> Optional[AsyncIterable[stt.SpeechEvent]]:
        activity = self.__get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node is available"

        async with activity.stt.stream() as stream:

            async def _forward_input():
                async for frame in audio:
                    stream.push_frame(frame)

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for event in stream:
                    yield event
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, fnc_ctx: list[AIFunction]
    ) -> Union[
        Optional[AsyncIterable[llm.ChatChunk]],
        Optional[AsyncIterable[str]],
        Optional[str],
    ]:
        activity = self.__get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM), (
            "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
        )

        async with activity.llm.chat(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx) as stream:
            async for chunk in stream:
                yield chunk

    async def tts_node(
        self, text: AsyncIterable[str]
    ) -> Optional[AsyncIterable[rtc.AudioFrame]]:
        activity = self.__get_activity_or_raise()
        assert activity.tts is not None, "tts_node called but no TTS node is available"

        async with activity.tts.stream() as stream:

            async def _forward_input():
                async for chunk in text:
                    stream.push_text(chunk)

                stream.end_input()

            forward_task = asyncio.create_task(_forward_input())
            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await utils.aio.cancel_and_wait(forward_task)

    def __get_activity_or_raise(self) -> TaskActivity:
        """Get the current activity context for this task (internal)"""
        if self._activity is None:
            raise RuntimeError("no activity context found, this task is not running")

        return self._activity
