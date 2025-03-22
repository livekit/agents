from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from livekit import rtc

from .. import llm, stt, tokenize, tts, utils, vad
from ..llm import ChatContext, FunctionTool, ToolError, find_function_tools
from ..llm.chat_context import _ReadOnlyChatContext
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr

if TYPE_CHECKING:
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession
    from .audio_recognition import _TurnDetector


@dataclass
class ModelSettings:
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN
    """The tool choice to use when calling the LLM."""


class Agent:
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: list[llm.FunctionTool] | None = None,
        turn_detector: NotGivenOr[_TurnDetector | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        self._instructions = instructions
        self._chat_ctx = chat_ctx or ChatContext.empty()
        self._tools = tools + find_function_tools(self)
        self._eou = turn_detector
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._vad = vad
        self._allow_interruptions = allow_interruptions
        self._activity: AgentActivity | None = None

    @property
    def instructions(self) -> str:
        """
        Returns:
            str: The core instructions that guide the agent's behavior.
        """
        return self._instructions

    @property
    def tools(self) -> list[llm.FunctionTool]:
        """
        Returns:
            list[llm.FunctionTool]: A list of function tools available to the agent.
        """
        return self._tools.copy()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """
        Provides a read-only view of the agent's current chat context.

        Returns:
            llm.ChatContext: A read-only version of the agent's conversation history.

        See Also:
            update_chat_ctx: Method to update the internal chat context.
        """
        return _ReadOnlyChatContext(self._chat_ctx.items)

    async def update_instructions(self, instructions: str) -> None:
        """
        Updates the agent's instructions.

        If the agent is running in realtime mode, this method also updates
        the instructions for the ongoing realtime session.

        Args:
            instructions (str):
                The new instructions to set for the agent.

        Raises:
            llm.RealtimeError: If updating the realtime session instructions fails.
        """
        if self._activity is None:
            self._instructions = instructions
            return

        await self._activity.update_instructions(instructions)

    async def update_tools(self, tools: list[llm.FunctionTool]) -> None:
        """
        Updates the agent's available function tools.

        If the agent is running in realtime mode, this method also updates
        the tools for the ongoing realtime session.

        Args:
            tools (list[llm.FunctionTool]):
                The new list of function tools available to the agent.

        Raises:
            llm.RealtimeError: If updating the realtime session tools fails.
        """
        if self._activity is None:
            self._tools = list(set(tools))
            return

        await self._activity.update_tools(tools)

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """
        Updates the agent's chat context.

        If the agent is running in realtime mode, this method also updates
        the chat context for the ongoing realtime session.

        Args:
            chat_ctx (llm.ChatContext):
                The new or updated chat context for the agent.

        Raises:
            llm.RealtimeError: If updating the realtime session chat context fails.
        """
        if self._activity is None:
            self._chat_ctx = chat_ctx.copy()
            return

        await self._activity.update_chat_ctx(chat_ctx)

    @property
    def turn_detector(self) -> NotGivenOr[_TurnDetector | None]:
        """
        Retrieves the turn detector for identifying conversational turns.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a turn detector,
        the session's turn detector will be used at runtime instead.

        Returns:
            NotGivenOr[_TurnDetector | None]: An optional turn detector for managing conversation flow.
        """  # noqa: E501
        return self._eou

    @property
    def stt(self) -> NotGivenOr[stt.STT | None]:
        """
        Retrieves the Speech-To-Text component for the agent.

        If this property was not set at Agent creation, but an ``AgentSession`` provides an STT component,
        the session's STT will be used at runtime instead.

        Returns:
            NotGivenOr[stt.STT | None]: An optional STT component.
        """  # noqa: E501
        return self._stt

    @property
    def llm(self) -> NotGivenOr[llm.LLM | llm.RealtimeModel | None]:
        """
        Retrieves the Language Model or RealtimeModel used for text generation.

        If this property was not set at Agent creation, but an ``AgentSession`` provides an LLM or RealtimeModel,
        the session's model will be used at runtime instead.

        Returns:
            NotGivenOr[llm.LLM | llm.RealtimeModel | None]: The language model for text generation.
        """  # noqa: E501
        return self._llm

    @property
    def tts(self) -> NotGivenOr[tts.TTS | None]:
        """
        Retrieves the Text-To-Speech component for the agent.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a TTS component,
        the session's TTS will be used at runtime instead.

        Returns:
            NotGivenOr[tts.TTS | None]: An optional TTS component for generating audio output.
        """  # noqa: E501
        return self._tts

    @property
    def vad(self) -> NotGivenOr[vad.VAD | None]:
        """
        Retrieves the Voice Activity Detection component for the agent.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a VAD component,
        the session's VAD will be used at runtime instead.

        Returns:
            NotGivenOr[vad.VAD | None]: An optional VAD component for detecting voice activity.
        """  # noqa: E501
        return self._vad

    @property
    def allow_interruptions(self) -> NotGivenOr[bool]:
        """
        Indicates whether interruptions (e.g., stopping TTS playback) are allowed.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a value for
        allowing interruptions, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether interruptions are permitted.
        """
        return self._allow_interruptions

    @property
    def realtime_llm_session(self) -> llm.RealtimeSession:
        """
        Retrieve the realtime LLM session associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running or the realtime LLM session is not available
        """
        if (rt_session := self.__get_activity_or_raise().realtime_llm_session) is None:
            raise RuntimeError("no realtime LLM session")

        return rt_session

    @property
    def session(self) -> AgentSession:
        """
        Retrieve the VoiceAgent associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running
        """
        return self.__get_activity_or_raise().agent

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the VoiceAgent

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
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent] | None:
        """
        A node in the processing pipeline that transcribes audio frames into speech events.

        By default, this node uses a Speech-To-Text (STT) capability from the current agent. If the STT
        implementation does not support streaming natively, a VAD (Voice Activity Detection)
        mechanism is required to wrap the STT.

        You can override this node with your own implementation for more flexibility (e.g.,
        custom pre-processing of audio, additional buffering, or alternative STT strategies).

        Args:
            audio (AsyncIterable[rtc.AudioFrame]): An asynchronous stream of audio frames.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            stt.SpeechEvent: An event containing transcribed text or other STT-related data.
        """  # noqa: E501
        activity = self.__get_activity_or_raise()
        assert activity.stt is not None, "stt_node called but no STT node is available"

        wrapped_stt = activity.stt

        if not activity.stt.capabilities.streaming:
            if not activity.vad:
                raise RuntimeError(
                    f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"  # noqa: E501
                    "Or manually wrap your STT in a stt.StreamAdapter"
                )

            wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

        async with wrapped_stt.stream() as stream:

            @utils.log_exceptions(logger=logger)
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
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk] | None | AsyncIterable[str] | None | str | None:
        """
        A node in the processing pipeline that processes text generation with an LLM.

        By default, this node uses the agent's LLM to process the provided context. It may yield
        plain text (as `str`) for straightforward text generation, or `llm.ChatChunk` objects that
        can include text and optional tool calls. `ChatChunk` is helpful for capturing more complex
        outputs such as function calls, usage statistics, or other metadata.

        You can override this node to customize how the LLM is used or how tool invocations
        and responses are handled.

        Args:
            chat_ctx (llm.ChatContext): The context for the LLM (including conversation history, etc.).
            tools (list[FunctionTool]): A list of callable tools that the LLM may invoke.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            str: Plain text output from the LLM.
            llm.ChatChunk: An object that can contain both text and optional tool calls.
        """  # noqa: E501
        activity = self.__get_activity_or_raise()
        assert activity.llm is not None, "llm_node called but no LLM node is available"
        assert isinstance(activity.llm, llm.LLM), (
            "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
        )

        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN

        async with activity.llm.chat(
            chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[str]:
        """
        A node in the processing pipeline that finalizes transcriptions from text segments.

        This node can be used to adjust or post-process text coming from an LLM (or any other source)
        into a final transcribed form. For instance, you might clean up formatting, fix punctuation,
        or perform any other text transformations here.

        You can override this node to customize post-processing logic according to your needs.

        Args:
            text (AsyncIterable[str]): An asynchronous stream of text segments.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            str: Finalized or post-processed text segments.
        """  # noqa: E501
        self.__get_activity_or_raise()
        async for delta in text:
            yield delta

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame] | None:
        """
        A node in the processing pipeline that synthesizes audio from text segments.

        By default, this node converts incoming text into audio frames using the Text-To-Speech (TTS)
        from the agent.
        If the TTS implementation does not support streaming natively, it uses a sentence tokenizer
        to split text for incremental synthesis.

        You can override this node to provide different text chunking behavior, a custom TTS engine,
        or any other specialized processing.

        Args:
            text (AsyncIterable[str]): An asynchronous stream of text segments to be synthesized.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            rtc.AudioFrame: Audio frames synthesized from the provided text.
        """  # noqa: E501
        activity = self.__get_activity_or_raise()
        assert activity.tts is not None, "tts_node called but no TTS node is available"

        wrapped_tts = activity.tts

        if not activity.tts.capabilities.streaming:
            wrapped_tts = tts.StreamAdapter(
                tts=wrapped_tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        async with wrapped_tts.stream() as stream:

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

    def __get_activity_or_raise(self) -> AgentActivity:
        """Get the current activity context for this task (internal)"""
        if self._activity is None:
            raise RuntimeError("no activity context found, this task is not running")

        return self._activity


TaskResult_T = TypeVar("TaskResult_T")


# TODO: rename to InlineAgent?
class InlineTask(Agent, Generic[TaskResult_T]):
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: list[llm.FunctionTool] | None = None,
        turn_detector: NotGivenOr[_TurnDetector | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=tools,
            turn_detector=turn_detector,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
        )

        self.__started = False
        self.__fut = asyncio.Future[TaskResult_T]()

    def complete(self, result: TaskResult_T | ToolError) -> None:
        if self.__fut.done():
            raise RuntimeError(f"{self.__class__.__name__} is already done")

        if isinstance(result, ToolError):
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
                f"{self.__class__.__name__} should only be awaited inside an async ai_function or the on_enter/on_exit methods of an AgentTask"  # noqa: E501
            )

        def _handle_task_done(_) -> None:
            if self.__fut.done():
                return

            # if the asyncio.Task running the InlineTask completes before the InlineTask itself, log
            # an error and attempt to recover by terminating the InlineTask.
            self.__fut.set_exception(
                RuntimeError(
                    f"{self.__class__.__name__} was not completed by the time the asyncio.Task running it was done"  # noqa: E501
                )
            )
            logger.error(
                f"{self.__class__.__name__} was not completed by the time the asyncio.Task running it was done"  # noqa: E501
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
