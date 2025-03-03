from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterable, Generic, Optional, TypeVar, Union

from livekit import rtc

from .. import llm, stt, tokenize, tts, utils, vad
from ..llm import (
    AIError,
    AIFunction,
    ChatContext,
    FunctionContext,
    find_ai_functions,
)
from ..log import logger
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
                raise ValueError("VAD is required when streaming is not supported by the STT")

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

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the PipelineAgent

    async def on_enter(self) -> None:
        """Called when the task is entered"""
        pass

    async def on_exit(self) -> None:
        """Called when the task is exited"""
        pass

    async def on_end_of_turn(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        """Called when the user has finished speaking, and the LLM is about to respond

        This is a good opportunity to update the chat context or edit the new message before it is
        sent to the LLM.
        """
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

    async def tts_node(self, text: AsyncIterable[str]) -> Optional[AsyncIterable[rtc.AudioFrame]]:
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


TaskResult_T = TypeVar("TaskResult_T")


class InlineTask(AgentTask, Generic[TaskResult_T]):
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
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            ai_functions=ai_functions,
            turn_detector=turn_detector,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
        )

        self.__started = False
        self.__fut = asyncio.Future[TaskResult_T]()

    def complete(self, result: TaskResult_T | AIError) -> None:
        if self.__fut.done():
            raise RuntimeError(f"{self.__class__.__name__} is already done")

        if isinstance(result, AIError):
            self.__fut.set_exception(result)
        else:
            self.__fut.set_result(result)

    async def __await_impl(self):
        if self.__started:
            raise RuntimeError(f"{self.__class__.__name__} is not re-entrant, await only once")

        self.__started = True

        task = asyncio.current_task()
        if task is None or not _is_inline_task_authorized(task):
            raise RuntimeError(
                f"{self.__class__.__name__} should only be awaited inside an async ai_function or the on_enter/on_exit methods of an AgentTask"
            )

        def _handle_task_done(_) -> None:
            if self.__fut.done():
                return

            # if the asyncio.Task running the InlineTask completes before the InlineTask itself, log
            # an error and attempt to recover by terminating the InlineTask.
            self.__fut.set_exception(
                RuntimeError(
                    f"{self.__class__.__name__} was not completed by the time the asyncio.Task running it was done"
                )
            )
            logger.error(
                f"{self.__class__.__name__} was not completed by the time the asyncio.Task running it was done"
            )

            # TODO(theomonnom): recover somehow

        task.add_done_callback(_handle_task_done)

        # enter task
        await asyncio.shield(self.__fut)
        # exit task

    def __await__(self):
        return self.__await_impl().__await__()


def _authorize_inline_task(task: asyncio.Task) -> None:
    setattr(task, "__livekit_agents_inline_task", True)


def _is_inline_task_authorized(task: asyncio.Task) -> bool:
    return getattr(task, "__livekit_agents_inline_task", False)
