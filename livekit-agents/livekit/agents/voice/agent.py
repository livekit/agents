from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from livekit import rtc

from .. import llm, stt, tokenize, tts, utils, vad
from ..llm import (
    ChatContext,
    FunctionTool,
    RawFunctionTool,
    find_function_tools,
)
from ..llm.chat_context import _ReadOnlyChatContext
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ..llm import mcp
    from .agent_activity import AgentActivity
    from .agent_session import AgentSession, TurnDetectionMode
    from .io import TimedString


@dataclass
class ModelSettings:
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN
    """The tool choice to use when calling the LLM."""


class Agent:
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext | None] = NOT_GIVEN,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        self._instructions = instructions
        self._tools = tools.copy() + find_function_tools(self)
        self._chat_ctx = chat_ctx.copy(tools=self._tools) if chat_ctx else ChatContext.empty()
        self._turn_detection = turn_detection
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._vad = vad
        self._allow_interruptions = allow_interruptions
        self._min_consecutive_speech_delay = min_consecutive_speech_delay
        self._use_tts_aligned_transcript = use_tts_aligned_transcript

        if isinstance(mcp_servers, list) and len(mcp_servers) == 0:
            mcp_servers = None  # treat empty list as None (but keep NOT_GIVEN)

        self._mcp_servers = mcp_servers
        self._activity: AgentActivity | None = None

    @property
    def label(self) -> str:
        """
        Returns:
            str: The label of the agent.
        """
        return f"{type(self).__module__}.{type(self).__name__}"

    @property
    def instructions(self) -> str:
        """
        Returns:
            str: The core instructions that guide the agent's behavior.
        """
        return self._instructions

    @property
    def tools(self) -> list[llm.FunctionTool | llm.RawFunctionTool]:
        """
        Returns:
            list[llm.FunctionTool | llm.RawFunctionTool]:
                A list of function tools available to the agent.
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

    async def update_tools(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
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
            self._chat_ctx = self._chat_ctx.copy(tools=self._tools)
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
            self._chat_ctx = chat_ctx.copy(tools=self._tools)
            return

        await self._activity.update_chat_ctx(chat_ctx)

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the VoiceAgent

    async def on_enter(self) -> None:
        """Called when the task is entered"""
        pass

    async def on_exit(self) -> None:
        """Called when the task is exited"""
        pass

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Called when the user has finished speaking, and the LLM is about to respond

        This is a good opportunity to update the chat context or edit the new message before it is
        sent to the LLM.
        """
        pass

    def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        """
        A node in the processing pipeline that transcribes audio frames into speech events.

        By default, this node uses a Speech-To-Text (STT) capability from the current agent.
        If the STT implementation does not support streaming natively, a VAD (Voice Activity
        Detection) mechanism is required to wrap the STT.

        You can override this node with your own implementation for more flexibility (e.g.,
        custom pre-processing of audio, additional buffering, or alternative STT strategies).

        Args:
            audio (AsyncIterable[rtc.AudioFrame]): An asynchronous stream of audio frames.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            stt.SpeechEvent: An event containing transcribed text or other STT-related data.
        """
        return Agent.default.stt_node(self, audio, model_settings)

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        model_settings: ModelSettings,
    ) -> (
        AsyncIterable[llm.ChatChunk | str]
        | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str]]
        | Coroutine[Any, Any, str]
        | Coroutine[Any, Any, llm.ChatChunk]
        | Coroutine[Any, Any, None]
    ):
        """
        A node in the processing pipeline that processes text generation with an LLM.

        By default, this node uses the agent's LLM to process the provided context. It may yield
        plain text (as `str`) for straightforward text generation, or `llm.ChatChunk` objects that
        can include text and optional tool calls. `ChatChunk` is helpful for capturing more complex
        outputs such as function calls, usage statistics, or other metadata.

        You can override this node to customize how the LLM is used or how tool invocations
        and responses are handled.

        Args:
            chat_ctx (llm.ChatContext): The context for the LLM (the conversation history).
            tools (list[FunctionTool]): A list of callable tools that the LLM may invoke.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields/Returns:
            str: Plain text output from the LLM.
            llm.ChatChunk: An object that can contain both text and optional tool calls.
        """
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> (
        AsyncIterable[str | TimedString]
        | Coroutine[Any, Any, AsyncIterable[str | TimedString]]
        | Coroutine[Any, Any, None]
    ):
        """
        A node in the processing pipeline that finalizes transcriptions from text segments.

        This node can be used to adjust or post-process text coming from an LLM (or any other
        source) into a final transcribed form. For instance, you might clean up formatting, fix
        punctuation, or perform any other text transformations here.

        You can override this node to customize post-processing logic according to your needs.

        Args:
            text (AsyncIterable[str | TimedString]): An asynchronous stream of text segments.
            model_settings (ModelSettings): Configuration and parameters for model execution.

        Yields:
            str: Finalized or post-processed text segments.
        """
        return Agent.default.transcription_node(self, text, model_settings)

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> (
        AsyncIterable[rtc.AudioFrame]
        | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]
        | Coroutine[Any, Any, None]
    ):
        """
        A node in the processing pipeline that synthesizes audio from text segments.

        By default, this node converts incoming text into audio frames using the Text-To-Speech
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
        """
        return Agent.default.tts_node(self, text, model_settings)

    def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[rtc.AudioFrame]
        | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]
        | Coroutine[Any, Any, None]
    ):
        """A node processing the audio from the realtime LLM session before it is played out."""
        return Agent.default.realtime_audio_output_node(self, audio, model_settings)

    def _get_activity_or_raise(self) -> AgentActivity:
        """Get the current activity context for this task (internal)"""
        if self._activity is None:
            raise RuntimeError("no activity context found, the agent is not running")

        return self._activity

    class default:
        @staticmethod
        async def stt_node(
            agent: Agent, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncGenerator[stt.SpeechEvent, None]:
            """Default implementation for `Agent.stt_node`"""
            activity = agent._get_activity_or_raise()
            assert activity.stt is not None, "stt_node called but no STT node is available"

            wrapped_stt = activity.stt

            if not activity.stt.capabilities.streaming:
                if not activity.vad:
                    raise RuntimeError(
                        f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"  # noqa: E501
                        "Or manually wrap your STT in a stt.StreamAdapter"
                    )

                wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

            conn_options = activity.session.conn_options.stt_conn_options
            async with wrapped_stt.stream(conn_options=conn_options) as stream:

                @utils.log_exceptions(logger=logger)
                async def _forward_input() -> None:
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(_forward_input())
                try:
                    async for event in stream:
                        yield event
                finally:
                    await utils.aio.cancel_and_wait(forward_task)

        @staticmethod
        async def llm_node(
            agent: Agent,
            chat_ctx: llm.ChatContext,
            tools: list[FunctionTool | RawFunctionTool],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[llm.ChatChunk | str, None]:
            """Default implementation for `Agent.llm_node`"""
            activity = agent._get_activity_or_raise()
            assert activity.llm is not None, "llm_node called but no LLM node is available"
            assert isinstance(activity.llm, llm.LLM), (
                "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
            )

            tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
            activity_llm = activity.llm

            conn_options = activity.session.conn_options.llm_conn_options
            async with activity_llm.chat(
                chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice, conn_options=conn_options
            ) as stream:
                async for chunk in stream:
                    yield chunk

        @staticmethod
        async def tts_node(
            agent: Agent, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncGenerator[rtc.AudioFrame, None]:
            """Default implementation for `Agent.tts_node`"""
            activity = agent._get_activity_or_raise()
            assert activity.tts is not None, "tts_node called but no TTS node is available"

            wrapped_tts = activity.tts

            if not activity.tts.capabilities.streaming:
                wrapped_tts = tts.StreamAdapter(
                    tts=wrapped_tts,
                    sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True),
                )

            conn_options = activity.session.conn_options.tts_conn_options
            async with wrapped_tts.stream(conn_options=conn_options) as stream:

                async def _forward_input() -> None:
                    async for chunk in text:
                        stream.push_text(chunk)

                    stream.end_input()

                forward_task = asyncio.create_task(_forward_input())
                try:
                    async for ev in stream:
                        yield ev.frame
                finally:
                    await utils.aio.cancel_and_wait(forward_task)

        @staticmethod
        async def transcription_node(
            agent: Agent, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
        ) -> AsyncGenerator[str | TimedString, None]:
            """Default implementation for `Agent.transcription_node`"""
            async for delta in text:
                yield delta

        @staticmethod
        async def realtime_audio_output_node(
            agent: Agent, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncGenerator[rtc.AudioFrame, None]:
            """Default implementation for `Agent.realtime_audio_output_node`"""
            activity = agent._get_activity_or_raise()
            assert activity.realtime_llm_session is not None, (
                "realtime_audio_output_node called but no realtime LLM session is available"
            )

            async for frame in audio:
                yield frame

    @property
    def realtime_llm_session(self) -> llm.RealtimeSession:
        """
        Retrieve the realtime LLM session associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running or the realtime LLM session is not available
        """
        if (rt_session := self._get_activity_or_raise().realtime_llm_session) is None:
            raise RuntimeError("no realtime LLM session")

        return rt_session

    @property
    def turn_detection(self) -> NotGivenOr[TurnDetectionMode | None]:
        """
        Retrieves the turn detection mode for identifying conversational turns.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a turn detection,
        the session's turn detection mode will be used at runtime instead.

        Returns:
            NotGivenOr[TurnDetectionMode | None]: An optional turn detection mode for managing conversation flow.
        """  # noqa: E501
        return self._turn_detection

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
    def mcp_servers(self) -> NotGivenOr[list[mcp.MCPServer] | None]:
        """
        Retrieves the list of Model Context Protocol (MCP) servers providing external tools.

        If this property was not set at Agent creation, but an ``AgentSession`` provides MCP servers,
        the session's MCP servers will be used at runtime instead.

        Returns:
            NotGivenOr[list[mcp.MCPServer]]: An optional list of MCP servers.
        """  # noqa: E501
        return self._mcp_servers

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
    def min_consecutive_speech_delay(self) -> NotGivenOr[float]:
        """
        Retrieves the minimum consecutive speech delay for the agent.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a value for
        the minimum consecutive speech delay, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[float]: The minimum consecutive speech delay.
        """
        return self._min_consecutive_speech_delay

    @property
    def use_tts_aligned_transcript(self) -> NotGivenOr[bool]:
        """
        Indicates whether to use TTS-aligned transcript as the input of
        the ``transcription_node``.

        If this property was not set at Agent creation, but an ``AgentSession`` provides a value for
        the use of TTS-aligned transcript, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether to use TTS-aligned transcript.
        """
        return self._use_tts_aligned_transcript

    @property
    def session(self) -> AgentSession:
        """
        Retrieve the VoiceAgent associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running
        """
        return self._get_activity_or_raise().session


TaskResult_T = TypeVar("TaskResult_T")


class AgentTask(Agent, Generic[TaskResult_T]):
    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: list[llm.FunctionTool | llm.RawFunctionTool] | None = None,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=tools,
            turn_detection=turn_detection,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            mcp_servers=mcp_servers,
            allow_interruptions=allow_interruptions,
        )

        self.__started = False
        self.__fut = asyncio.Future[TaskResult_T]()

    def done(self) -> bool:
        return self.__fut.done()

    def complete(self, result: TaskResult_T | Exception) -> None:
        if self.__fut.done():
            raise RuntimeError(f"{self.__class__.__name__} is already done")

        if isinstance(result, Exception):
            self.__fut.set_exception(result)
        else:
            self.__fut.set_result(result)

        self.__fut.exception()  # silence exc not retrieved warnings

        from .agent_activity import _SpeechHandleContextVar

        speech_handle = _SpeechHandleContextVar.get(None)

        if speech_handle:
            speech_handle._maybe_run_final_output = result

        # if not self.__inline_mode:
        #    session._close_soon(reason=CloseReason.TASK_COMPLETED, drain=True)

    async def __await_impl(self) -> TaskResult_T:
        if self.__started:
            raise RuntimeError(f"{self.__class__.__name__} is not re-entrant, await only once")

        self.__started = True

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be executed inside an async context"
            )

        task_info = _get_activity_task_info(current_task)
        if not task_info or not task_info.inline_task:
            raise RuntimeError(
                f"{self.__class__.__name__} should only be awaited inside tool_functions or the on_enter/on_exit methods of an Agent"  # noqa: E501
            )

        def _handle_task_done(_: asyncio.Task[Any]) -> None:
            if self.__fut.done():
                return

            # if the asyncio.Task running the InlineTask completes before the InlineTask itself, log
            # an error and attempt to recover by terminating the InlineTask.
            logger.error(
                f"The asyncio.Task finished before {self.__class__.__name__} was completed."
            )

            self.complete(
                RuntimeError(
                    f"The asyncio.Task finished before {self.__class__.__name__} was completed."
                )
            )

        current_task.add_done_callback(_handle_task_done)

        from .agent_activity import _AgentActivityContextVar, _SpeechHandleContextVar

        # TODO(theomonnom): add a global lock for inline tasks
        # This may currently break in the case we use parallel tool calls.

        speech_handle = _SpeechHandleContextVar.get(None)
        old_activity = _AgentActivityContextVar.get()
        old_agent = old_activity.agent
        session = old_activity.session

        # TODO(theomonnom): could the RunResult watcher & the blocked_tasks share the same logic?
        await session._update_activity(
            self, previous_activity="pause", blocked_tasks=[current_task]
        )

        # NOTE: _update_activity is calling the on_enter method, so the RunResult can capture all speeches
        run_state = session._global_run_state
        if speech_handle and run_state and not run_state.done():
            # make sure to not deadlock on the current speech handle
            run_state._unwatch_handle(speech_handle)
            # it is OK to call _mark_done_if_needed here, the above _update_activity will call on_enter
            # so handles added inside the on_enter will make sure we're not completing the run_state too early.
            run_state._mark_done_if_needed(None)

        try:
            return await asyncio.shield(self.__fut)

        finally:
            # run_state could have changed after self.__fut
            run_state = session._global_run_state

            if session.current_agent != self:
                logger.warning(
                    f"{self.__class__.__name__} completed, but the agent has changed in the meantime. "
                    "Ignoring handoff to the previous agent, likely due to `AgentSession.update_agent` being invoked."
                )
                await old_activity.aclose()
            else:
                if speech_handle and run_state and not run_state.done():
                    run_state._watch_handle(speech_handle)

                merged_chat_ctx = old_agent.chat_ctx.merge(
                    self.chat_ctx, exclude_function_call=True, exclude_instructions=True
                )
                # set the chat_ctx directly, `session._update_activity` will sync it to the rt_session if needed
                old_agent._chat_ctx.items[:] = merged_chat_ctx.items
                # await old_agent.update_chat_ctx(merged_chat_ctx)

                await session._update_activity(
                    old_agent, new_activity="resume", wait_on_enter=False
                )

    def __await__(self) -> Generator[None, None, TaskResult_T]:
        return self.__await_impl().__await__()


@dataclass
class _ActivityTaskInfo:
    function_call: llm.FunctionCall | None = None
    speech_handle: SpeechHandle | None = None
    inline_task: bool = False


def _set_activity_task_info(
    task: asyncio.Task[Any],
    *,
    function_call: NotGivenOr[llm.FunctionCall | None] = NOT_GIVEN,
    speech_handle: NotGivenOr[SpeechHandle | None] = NOT_GIVEN,
    inline_task: NotGivenOr[bool] = NOT_GIVEN,
) -> None:
    info = _get_activity_task_info(task) or _ActivityTaskInfo()

    if is_given(function_call):
        info.function_call = function_call

    if is_given(speech_handle):
        info.speech_handle = speech_handle

    if is_given(inline_task):
        info.inline_task = inline_task

    setattr(task, "__livekit_agents_activity_task", info)


def _get_activity_task_info(task: asyncio.Task[Any]) -> _ActivityTaskInfo | None:
    return getattr(task, "__livekit_agents_activity_task", None)
