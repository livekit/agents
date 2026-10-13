from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Sequence,
)
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import TypeAdapter
from typing_extensions import ParamSpec, TypeVar

from livekit import rtc

from .. import inference, llm, stt, tokenize, tts, utils, vad
from ..llm import (
    LLM,
    DuplexModel,
    DuplexRealtimeAdapter,
    RealtimeModel,
    ToolError,
    find_function_tools,
)
from ..llm.chat_context import Instructions
from ..llm.duplex_adapter import _DuplexRealtimeSession
from ..llm.tool_context import (
    DuplicateMode,
    DuplicateScope,
    FunctionTool,
    RawFunctionDescription,
    RawFunctionTool,
    ReadBack,
    ToolFlag,
    function_tool,
)
from ..log import logger
from ..types import NOT_GIVEN, FlushSentinel, NotGivenOr
from ..utils import is_given, misc
from .events import UserTurnExceededEvent
from .speech_handle import SpeechHandle
from .tool_executor import ToolHandlingOptions
from .turn import TurnHandlingOptions, _migrate_turn_handling

if TYPE_CHECKING:
    from ..inference import LLMModels, RealtimeModels, STTModels, TTSModels
    from ..llm import mcp
    from .agent_activity import AgentActivity
    from .agent_run import AgentRun
    from .agent_session import ExpressiveOptions, LiveSession
    from .audio_recognition import AudioRecognition
    from .events import RunContext
    from .io import TimedString
    from .run_result import RunResult, RunUsage
    from .turn import TurnDetectionMode


_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])
_InstructionsFn = Callable[["RunContext"], str | Awaitable[str]]
_InstructionsFnT = TypeVar("_InstructionsFnT", bound=Callable[..., str | Awaitable[str]])
_OnEnterFn = Callable[["LiveSession"], Any]
_OutputValidatorFn = Callable[["RunContext", Any], Any]

DEFAULT_AGENT_ID = "default_agent"


class _OnEnterHook:
    """``Agent.on_enter``: the hook to await, or the decorator registering it.

    A descriptor so a subclass still overrides ``async def on_enter(self) -> None``; the
    bound form is untyped on purpose, both uses go through it.
    """

    def __get__(self, agent: Agent | None, owner: type[Agent]) -> Any:
        if agent is None:
            return self
        return _BoundOnEnter(agent)


class _BoundOnEnter:
    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def __call__(self, f: _OnEnterFn | None = None, /) -> Any:
        if f is not None:
            return self._agent._register_on_enter(f)
        return self._agent._run_on_enter()


def _assignment_namespace(frame: Any) -> dict[str, Any] | None:
    """The namespace a module-level ``Agent(...)`` is assigned into, or ``None``.

    Only a module (or an ``exec``) frame has a live namespace to scan later; a function frame's
    locals are a snapshot, and holding its proxy would keep the frame alive.
    """
    if frame is None or frame.f_locals is not frame.f_globals:
        return None
    namespace: dict[str, Any] = frame.f_locals
    return namespace


@dataclass
class ModelSettings:
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN
    """The tool choice to use when calling the LLM."""


class Agent:
    def __init__(
        self,
        *,
        instructions: str | Instructions,
        name: str | None = None,
        id: str | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        deps_type: type[Any] | None = None,
        state_type: type[Any] | None = None,
        output_type: type[Any] | None = None,
        stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        tool_handling: NotGivenOr[ToolHandlingOptions] = NOT_GIVEN,
        llm: NotGivenOr[
            llm.LLM | llm.RealtimeModel | llm.DuplexModel | LLMModels | RealtimeModels | str | None
        ] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | ExpressiveOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
        # deprecated
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        # a plain Agent is named after the variable it is assigned to: the assigning frame's
        # namespace is kept and scanned the first time the name is read, since the assignment completes only after the constructor returns
        self._id: str | None = name or id
        self._defining_namespace: dict[str, Any] | None = None
        if self._id is None:
            if type(self) is Agent:
                self._defining_namespace = _assignment_namespace(sys._getframe(1))
            else:
                self._id = misc.camel_to_snake_case(type(self).__name__)

        turn_handling = (
            _migrate_turn_handling(
                min_endpointing_delay=min_endpointing_delay,
                max_endpointing_delay=max_endpointing_delay,
                turn_detection=turn_detection,
                allow_interruptions=allow_interruptions,
            )
            if not is_given(turn_handling)
            else turn_handling
        )

        self._instructions = instructions
        self._tools = [*tools, *find_function_tools(self)]
        # registered at import by the decorators below; frozen once the agent is first bound
        self._instruction_fns: list[_InstructionsFn] = []
        self._instructions_view: _InstructionsView | None = None
        self._on_enter_fn: _OnEnterFn | None = None
        self._output_validators: list[_OutputValidatorFn] = []
        self._frozen = False
        self._turn_detection = turn_handling.get("turn_detection", NOT_GIVEN)

        # the three type parameters of Agent[Deps, State, Output]; declared for the type
        # checker and the output parser, the values travel with the run
        self._deps_type = deps_type
        self._state_type = state_type
        self._output_type: type[Any] = output_type if output_type is not None else str

        if isinstance(stt, str):
            stt = inference.STT.from_model_string(stt)

        if isinstance(llm, str):
            llm = inference.llm_from_model_string(llm)

        if isinstance(tts, str):
            tts = inference.TTS.from_model_string(tts)

        self._stt = stt
        # a duplex model is wrapped on the way in, so nothing downstream sees one
        self._llm: NotGivenOr[LLM | RealtimeModel | None] = (
            DuplexRealtimeAdapter(llm) if isinstance(llm, DuplexModel) else llm
        )
        self._tts = tts
        self._vad = vad
        self._expressive: NotGivenOr[bool | ExpressiveOptions] = expressive

        self._allow_interruptions: NotGivenOr[bool] = NOT_GIVEN
        self._interruption_detection: NotGivenOr[Literal["adaptive", "vad"]] = NOT_GIVEN
        if is_given(raw_interruption := turn_handling.get("interruption", NOT_GIVEN)):
            if "enabled" in raw_interruption:
                self._allow_interruptions = raw_interruption["enabled"]
            if "mode" in raw_interruption:
                self._interruption_detection = raw_interruption["mode"]
        endpointing = turn_handling.get("endpointing", {})
        self._min_consecutive_speech_delay = min_consecutive_speech_delay
        self._use_tts_aligned_transcript = use_tts_aligned_transcript
        self._min_endpointing_delay = endpointing.get("min_delay", NOT_GIVEN)
        self._max_endpointing_delay = endpointing.get("max_delay", NOT_GIVEN)
        self._turn_handling = turn_handling
        # stored unresolved so the resolution chain can tell "set on agent" from "fall
        # back to session"; async_options absent on a given tool_handling means NOT_GIVEN
        self._async_tool_options = (
            tool_handling.get("async_options", NOT_GIVEN) if is_given(tool_handling) else NOT_GIVEN
        )

        if isinstance(mcp_servers, list) and len(mcp_servers) == 0:
            mcp_servers = None  # treat empty list as None (but keep NOT_GIVEN)

        self._mcp_servers = mcp_servers
        if self._mcp_servers:
            logger.warning(
                "passing MCP servers to LiveSession or Agent is deprecated "
                "and will be removed in a future version. Use `MCPToolset` instead."
            )

    @property
    def id(self) -> str:
        if self._id is None and self._defining_namespace is not None:
            for var_name, value in self._defining_namespace.items():
                if value is self:
                    self._id = var_name
                    self._defining_namespace = None
                    break
        return self._id or DEFAULT_AGENT_ID

    @property
    def label(self) -> str:
        return self.id

    @property
    def instructions(self) -> _InstructionsView:
        """The static instructions, also the decorator registering dynamic ones.

        Reads as the ``str`` or ``Instructions`` the agent was built with. Used as
        ``@agent.instructions`` it registers a function rendered into the instructions on every
        turn, which receives the ``RunContext`` and returns a string, sync or async::

            @scheduler.instructions
            def known_appointment(ctx: RunContext) -> str: ...
        """
        if self._instructions_view is None or self._instructions_view._source is not (
            self._instructions
        ):
            self._instructions_view = _wrap_instructions(self._instructions, self)
        return self._instructions_view

    @property
    def tools(self) -> list[llm.Tool | llm.Toolset]:
        """
        Returns:
            list[llm.Tool | llm.ToolSet]:
                A list of function tools available to the agent.
        """
        return self._tools.copy()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """A read-only view of the history the running activity works on.

        The history belongs to the session; this resolves through the activity bound in the
        current context, so it is only available inside the agent's hooks and tools.

        Raises:
            RuntimeError: If the agent is not running in this context.
        """
        return self._get_binding_or_raise().chat_ctx

    @property
    def interruption_detection(self) -> NotGivenOr[Literal["adaptive", "vad"]]:
        return self._interruption_detection

    @property
    def audio_recognition(self) -> AudioRecognition:
        """Access the audio recognition system for this agent.

        The only public member is ``stt_context`` — live speaker metadata from the
        STT stream.

        Raises:
            RuntimeError: If the agent is not running.
        """
        activity = self._get_activity_or_raise()
        assert activity._audio_recognition is not None
        return activity._audio_recognition

    def update_options(
        self,
        *,
        stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[
            llm.LLM | llm.RealtimeModel | llm.DuplexModel | LLMModels | RealtimeModels | str | None
        ] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | ExpressiveOptions] = NOT_GIVEN,
    ) -> None:
        """Swap the STT, VAD, LLM, or TTS on this agent, or change its expressive setting.
        Only the options passed are changed.

        Useful for switching a component mid-call (e.g. a different STT language or TTS voice).
        Strings resolve to inference models like the constructor. Pass ``None`` to disable a
        model (overriding the session), matching ``Agent(stt=None)``. If the agent is running,
        the swap applies to the live pipeline. ``expressive`` overrides the session value and
        takes effect on the next reply; pass ``False`` to force it off for this agent.

        The live swap applies to the activity bound in the current context, so it must be
        called from the agent's hooks or tools; elsewhere only the stored configuration changes.

        Raises:
            RuntimeError: When swapping to or from a ``RealtimeModel`` while the agent is
                running; use ``LiveSession.update_agent`` instead.
        """
        if isinstance(stt, str):
            stt = inference.STT.from_model_string(stt)
        if isinstance(llm, str):
            llm = inference.llm_from_model_string(llm)
        if isinstance(tts, str):
            tts = inference.TTS.from_model_string(tts)

        if isinstance(llm, DuplexModel):
            llm = DuplexRealtimeAdapter(llm)

        activity = self._bound_activity()
        if activity is None:
            # not running in this context: replace stored config, applied on the next start
            if is_given(stt):
                self._stt = stt
            if is_given(vad):
                self._vad = vad
            if is_given(llm):
                self._llm = llm
            if is_given(tts):
                self._tts = tts
            if is_given(expressive):
                self._expressive = expressive
            return

        activity._update_models(new_stt=stt, new_vad=vad, new_llm=llm, new_tts=tts)
        if is_given(expressive):
            # after _update_models so a rejected model swap leaves expressive untouched;
            # resolved per turn (agent value over session), no live plumbing needed
            self._expressive = expressive

    @property
    def deps_type(self) -> type[Any] | None:
        return self._deps_type

    @property
    def state_type(self) -> type[Any] | None:
        return self._state_type

    @property
    def output_type(self) -> type[Any]:
        """What ``run()`` returns as ``result.output``: ``str`` unless declared."""
        return self._output_type

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        state: Any = None,
        deps: Any = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> RunResult[Any]:
        """One turn on the given history: the model and tool loop until the answer.

        Pure: a message in, a typed ``RunResult`` out, nothing kept after. The run stops at
        an ``InputRequired`` from a tool, with ``result.input_required`` set and
        ``result.output`` ``None``. ``stream()`` is the same turn with the reply as it is
        written.

        Args:
            message: The user's message.
            message_history: The conversation so far; omit for a fresh one.
            state: The session's typed state, mutated in place by the tools.
            deps: What the tools need for this run, read as ``ctx.deps``.
            usage: A ``RunUsage`` to add this run's usage to, for a delegated run.
            max_steps: The most model requests one turn may make.
        """
        run = self.stream(
            message,
            message_history=message_history,
            state=state,
            deps=deps,
            usage=usage,
            max_steps=max_steps,
        )
        return await run._run()

    def stream(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        state: Any = None,
        deps: Any = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> AgentRun[Any]:
        """The turn of ``run()`` as an async iterator of the reply's text, one piece as the
        model writes it; ``run.result`` is the ``RunResult`` once the loop ends::

            run = agent.stream("book me in", message_history=history)
            async for delta in run:
                print(delta, end="")
            answer = run.result.output
        """
        from .agent_run import AgentRun

        return AgentRun[Any](
            self,
            message,
            message_history=message_history,
            state=state,
            deps=deps,
            usage=usage,
            max_steps=max_steps,
            output_type=self._output_type,
        )

    def output_validator(self, f: _OutputValidatorFn) -> _OutputValidatorFn:
        """Register a validator of the run's output, as a decorator.

        It receives the ``RunContext`` and the output, returns the value or raises
        ``ModelRetry`` to send the model back with the reason::

            @fnol.output_validator
            async def cites_real_jobs(ctx: RunContext, output: Claim) -> Claim: ...
        """
        self._check_not_frozen("an output validator")
        self._output_validators.append(f)
        return f

    async def _validate_output(self, ctx: RunContext, output: Any) -> Any:
        """The output through every validator; raises ``ModelRetry`` from the first that does."""
        for fn in self._output_validators:
            validated = fn(ctx, output)
            if inspect.isawaitable(validated):
                validated = await validated
            output = validated
        return output

    # -- Registration at import --
    # `@agent.tool`, `@agent.instructions` and `@agent.on_enter` register on a module-level
    # instance; the lists freeze when the agent is first bound to a session

    def _check_not_frozen(self, what: str) -> None:
        if self._frozen:
            raise RuntimeError(
                f"cannot register {what} on agent `{self.id}`: it is already bound to a session"
            )

    @overload
    def tool(self, f: Callable[_P, _R], /) -> FunctionTool[_P, _R]: ...

    @overload
    def tool(
        self,
        f: None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
        flags: ToolFlag = ToolFlag.NONE,
        on_duplicate: DuplicateMode = "allow",
        duplicate_scope: DuplicateScope = "name",
        requires_approval: bool | ReadBack = False,
    ) -> Callable[[Callable[_P, _R]], FunctionTool[_P, _R] | RawFunctionTool[_P, _R]]: ...

    def tool(
        self,
        f: Callable[_P, _R] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
        flags: ToolFlag = ToolFlag.NONE,
        on_duplicate: DuplicateMode = "allow",
        duplicate_scope: DuplicateScope = "name",
        requires_approval: bool | ReadBack = False,
    ) -> Any:
        """Register a function tool on this agent, as a decorator.

        The first parameter of the function is the ``RunContext``; the rest are the tool's
        arguments and its docstring feeds the schema, as with ``@function_tool``::

            @scheduler.tool
            async def find_appointment(ctx: RunContext, last_name: str) -> str: ...
        """

        def register(func: Callable[_P, _R]) -> FunctionTool[_P, _R] | RawFunctionTool[_P, _R]:
            self._check_not_frozen("a tool")
            tool: FunctionTool[_P, _R] | RawFunctionTool[_P, _R]
            if isinstance(func, (FunctionTool, RawFunctionTool)):
                # an already built tool, registered as it is
                tool = func
            elif raw_schema is not None:
                tool = function_tool(
                    func,
                    raw_schema=raw_schema,
                    flags=flags,
                    on_duplicate=on_duplicate,
                    duplicate_scope=duplicate_scope,
                )
            else:
                tool = function_tool(
                    func,
                    name=name,
                    description=description,
                    flags=flags,
                    on_duplicate=on_duplicate,
                    duplicate_scope=duplicate_scope,
                    requires_approval=requires_approval,
                )
            self._tools.append(tool)
            return tool

        return register(f) if f is not None else register

    def _register_instructions(self, f: _InstructionsFn) -> _InstructionsFn:
        self._check_not_frozen("an instructions function")
        self._instruction_fns.append(f)
        return f

    async def _render_dynamic_instructions(self, ctx: RunContext) -> str:
        """The registered instruction functions, rendered for one turn and joined."""
        parts: list[str] = []
        for fn in self._instruction_fns:
            rendered = fn(ctx)
            if inspect.isawaitable(rendered):
                rendered = await rendered
            if rendered:
                parts.append(str(rendered))
        return "\n\n".join(parts)

    async def _render_instructions(
        self, ctx: RunContext, *, modality: Literal["audio", "text"] | None = None
    ) -> str | Instructions:
        """The static instructions with the dynamic ones appended, for one turn."""
        dynamic = await self._render_dynamic_instructions(ctx)
        if not dynamic:
            return self._instructions

        static = self._instructions
        if isinstance(static, Instructions):
            static = static.render(modality=modality)
        return f"{static}\n\n{dynamic}" if static else dynamic

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the VoiceAgent

    on_enter = _OnEnterHook()
    """Called when the agent takes the conversation.

    Subclasses override it as ``async def on_enter(self) -> None``. On a plain instance it is
    a decorator registering the function to run then, which receives the session::

        @scheduler.on_enter
        async def greet(session: LiveSession) -> None:
            session.generate_reply(instructions="Greet the caller.")
    """

    def _register_on_enter(self, f: _OnEnterFn) -> _OnEnterFn:
        self._check_not_frozen("an on_enter function")
        self._on_enter_fn = f
        return f

    async def _run_on_enter(self) -> Any:
        if self._on_enter_fn is None:
            return None
        result = self._on_enter_fn(self.session)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _enter(self, ctx: RunContext[Any, Any]) -> Any:
        """``on_enter`` as it was written: an override takes the context when it declares one,
        the decorator form takes the call's session and is nothing in text."""
        hook = type(self).on_enter
        if isinstance(hook, _OnEnterHook):
            return await self._run_on_enter() if ctx._session is not None else None
        if len(inspect.signature(hook).parameters) > 1:
            return await hook(self, ctx)
        return await hook(self)

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

    async def on_user_turn_exceeded(self, ev: UserTurnExceededEvent) -> None:
        """Called when the user turn has exceeded the configured limit.

        The user has been speaking for too long without the agent successfully
        responding. By default, generates a reply using the current turn's
        transcript (previous turns are already in the chat context).

        Override to customize (e.g., use session.say() with a canned message,
        or skip the interruption entirely).
        """
        await self.session.generate_reply(
            user_input=ev.transcript,
            instructions=(
                "The user has been speaking too long without giving a chance to reply. "
                "Politely cut in with a short reply or notice. Keep it short since the user cannot interrupt it."
            ),
            allow_interruptions=False,
            tool_choice="none",
        )

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
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> (
        AsyncIterable[llm.ChatChunk | str | FlushSentinel]
        | Coroutine[Any, Any, AsyncIterable[llm.ChatChunk | str | FlushSentinel]]
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

    def _bound_binding(self) -> AgentActivity | AgentRun[Any] | None:
        """What runs this agent in the current context: a voice activity or a run (internal).

        Resolved through the contextvar both set around every hook, tool and pipeline node,
        so one shared instance serving several sessions reads the right one.
        """
        from .agent_activity import _AgentActivityContextVar

        binding = _AgentActivityContextVar.get(None)
        if binding is None or binding.agent is not self:
            return None
        return binding

    def _bound_activity(self) -> AgentActivity | None:
        """The voice activity running this agent in the current context, if any (internal)."""
        from .agent_activity import AgentActivity

        binding = self._bound_binding()
        return binding if isinstance(binding, AgentActivity) else None

    def _get_binding_or_raise(self) -> AgentActivity | AgentRun[Any]:
        binding = self._bound_binding()
        if binding is None:
            raise RuntimeError(
                f"agent `{self.id}` is not running in this context; its session and history "
                "are only available inside its hooks and tools"
            )
        return binding

    def _get_activity_or_raise(self) -> AgentActivity:
        """Get the current voice activity for this agent (internal)"""
        from .agent_activity import AgentActivity

        binding = self._get_binding_or_raise()
        if not isinstance(binding, AgentActivity):
            raise RuntimeError(
                f"agent `{self.id}` is running in agent.run(), which has no live session"
            )
        return binding

    class default:
        @staticmethod
        async def stt_node(
            agent: Agent, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncGenerator[stt.SpeechEvent, None]:
            """Default implementation for `Agent.stt_node`"""
            activity = agent._get_activity_or_raise()
            assert activity.stt is not None, "stt_node called but no STT node is available"

            wrapped_stt = activity.stt
            temporary_adapter: stt.StreamAdapter | None = None

            if not activity.stt.capabilities.streaming:
                if not activity.vad:
                    raise RuntimeError(
                        f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"  # noqa: E501
                        "Or manually wrap your STT in a stt.StreamAdapter"
                    )

                temporary_adapter = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)
                wrapped_stt = temporary_adapter

            try:
                conn_options = activity.session.conn_options.stt_conn_options
                async with wrapped_stt.stream(conn_options=conn_options) as stream:
                    _audio_input_started_at: float = (
                        activity._audio_recognition._input_started_at
                        if activity._audio_recognition is not None
                        and activity._audio_recognition._input_started_at is not None
                        else (
                            activity.session._recorder_io.recording_started_at
                            if activity.session._recorder_io
                            and activity.session._recorder_io.recording_started_at
                            else activity.session._started_at
                            if activity.session._started_at
                            else time.time()
                        )
                    )
                    stream.start_time_offset = time.time() - _audio_input_started_at

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
            finally:
                if temporary_adapter is not None:
                    await temporary_adapter.aclose()

        @staticmethod
        async def llm_node(
            agent: Agent,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[llm.ChatChunk | str | FlushSentinel, None]:
            """Default implementation for `Agent.llm_node`"""
            binding = agent._get_binding_or_raise()
            assert binding.llm is not None, "llm_node called but no LLM node is available"
            assert isinstance(binding.llm, llm.LLM), (
                "llm_node should only be used with LLM (non-multimodal/realtime APIs) nodes"
            )

            tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
            activity_llm = binding.llm

            conn_options = binding.llm_conn_options
            async with activity_llm.chat(
                chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice, conn_options=conn_options
            ) as stream:
                async for chunk in stream:
                    yield chunk

        @staticmethod
        async def tts_node(
            agent: Agent,
            text: AsyncIterable[str],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[rtc.AudioFrame, None]:
            """Default implementation for `Agent.tts_node`"""
            activity = agent._get_activity_or_raise()
            if activity.tts is None:
                raise RuntimeError(
                    "`tts_node` called but no TTS node is available. If audio output is not needed, disable it using "
                    "`session.output.set_audio_enabled(False)`."
                )

            expressive_active = activity._resolve_expressive_options() is not None
            wrapped_tts = activity.tts
            temporary_adapter: tts.StreamAdapter | None = None

            if not activity.tts.capabilities.streaming:
                temporary_adapter = tts.StreamAdapter(
                    tts=wrapped_tts,
                    sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
                        retain_format=True,
                        # markup only exists in the stream when expressive is active
                        xml_aware=expressive_active,
                    ),
                )
                wrapped_tts = temporary_adapter

            try:
                # Mark whether expressive is active for this synthesis, synchronously
                # just before stream() snapshots it. Doing it here (the single synthesis
                # choke point for both generate_reply and say()) scopes it to this turn
                # rather than leaving stale state on the instance. The provider's chunk
                # defaults then drive the TTS's input tokenizer.
                activity.tts._set_expressive(expressive_active)

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
            finally:
                if temporary_adapter is not None:
                    await temporary_adapter.aclose()

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
    def duplex_session(self) -> llm.DuplexSession:
        """
        Retrieve the duplex session of the current agent, for provider-specific APIs.

        A duplex model is driven through an adapter that presents it as a realtime session; this
        returns the plugin's own session, where a provider puts what the abstraction does not carry.

        Raises:
            RuntimeError: If the agent is not running, or is not running on a duplex model
        """
        rt_session = self._get_activity_or_raise().realtime_llm_session
        if not isinstance(rt_session, _DuplexRealtimeSession):
            raise RuntimeError("no duplex session, this agent is not running a DuplexModel")

        return rt_session.duplex_session

    @property
    def turn_detection(self) -> NotGivenOr[TurnDetectionMode | None]:
        """
        Retrieves the turn detection mode for identifying conversational turns.

        If this property was not set at Agent creation, but an ``LiveSession`` provides a turn detection,
        the session's turn detection mode will be used at runtime instead.

        Returns:
            NotGivenOr[TurnDetectionMode | None]: An optional turn detection mode for managing conversation flow.
        """  # noqa: E501
        return self._turn_detection

    @turn_detection.setter
    def turn_detection(self, value: TurnDetectionMode | None) -> None:
        self._turn_detection = value

        if (activity := self._bound_activity()) is not None:
            activity.update_options(turn_detection=value)

    @property
    def stt(self) -> NotGivenOr[stt.STT | None]:
        """
        Retrieves the Speech-To-Text component for the agent.

        If this property was not set at Agent creation, but an ``LiveSession`` provides an STT component,
        the session's STT will be used at runtime instead.

        Returns:
            NotGivenOr[stt.STT | None]: An optional STT component.
        """  # noqa: E501
        return self._stt

    @property
    def llm(self) -> NotGivenOr[llm.LLM | llm.RealtimeModel | None]:
        """
        Retrieves the Language Model or RealtimeModel used for text generation.

        If this property was not set at Agent creation, but an ``LiveSession`` provides an LLM or RealtimeModel,
        the session's model will be used at runtime instead.

        Returns:
            NotGivenOr[llm.LLM | llm.RealtimeModel | None]: The language model for text generation.
        """  # noqa: E501
        return self._llm

    @property
    def tts(self) -> NotGivenOr[tts.TTS | None]:
        """
        Retrieves the Text-To-Speech component for the agent.

        If this property was not set at Agent creation, but an ``LiveSession`` provides a TTS component,
        the session's TTS will be used at runtime instead.

        Returns:
            NotGivenOr[tts.TTS | None]: An optional TTS component for generating audio output.
        """  # noqa: E501
        return self._tts

    @property
    def expressive(self) -> NotGivenOr[bool | ExpressiveOptions]:
        """
        Retrieves the expressive TTS delivery setting for the agent.

        If this property was not set at Agent creation, the ``LiveSession``'s ``expressive``
        value will be used at runtime instead. When set, it overrides the session value for
        this agent's turns, matching how ``llm`` and ``tts`` overrides behave.

        Returns:
            NotGivenOr[bool | ExpressiveOptions]: Whether expressive delivery is enabled,
                or its configuration.
        """
        return self._expressive

    @property
    def mcp_servers(self) -> NotGivenOr[list[mcp.MCPServer] | None]:
        """
        Retrieves the list of Model Context Protocol (MCP) servers providing external tools.

        If this property was not set at Agent creation, but an ``LiveSession`` provides MCP servers,
        the session's MCP servers will be used at runtime instead.

        Returns:
            NotGivenOr[list[mcp.MCPServer]]: An optional list of MCP servers.
        """  # noqa: E501
        return self._mcp_servers

    @property
    def vad(self) -> NotGivenOr[vad.VAD | None]:
        """
        Retrieves the Voice Activity Detection component for the agent.

        If this property was not set at Agent creation, but an ``LiveSession`` provides a VAD component,
        the session's VAD will be used at runtime instead.

        Returns:
            NotGivenOr[vad.VAD | None]: An optional VAD component for detecting voice activity.
        """  # noqa: E501
        return self._vad

    @property
    def allow_interruptions(self) -> NotGivenOr[bool]:
        """
        Indicates whether interruptions (e.g., stopping TTS playback) are allowed.

        If this property was not set at Agent creation, but an ``LiveSession`` provides a value for
        allowing interruptions, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether interruptions are permitted.
        """
        return self._allow_interruptions

    @property
    def min_endpointing_delay(self) -> NotGivenOr[float]:
        """
        Minimum time-in-seconds since the last detected speech before the agent
        declares the user’s turn complete. In VAD mode this effectively behaves
        like max(VAD silence, min_endpointing_delay); in STT mode it is applied
        after the STT end-of-speech signal, so it can be additive with the STT
        provider’s endpointing delay.

        If this property was set at Agent creation, it will be used at runtime instead of the session's value.
        """
        return self._min_endpointing_delay

    @property
    def max_endpointing_delay(self) -> NotGivenOr[float]:
        """
        Maximum time-in-seconds the agent will wait before terminating the turn.

        If this property was set at Agent creation, it will be used at runtime instead of the session's value.
        """
        return self._max_endpointing_delay

    @property
    def min_consecutive_speech_delay(self) -> NotGivenOr[float]:
        """
        Retrieves the minimum consecutive speech delay for the agent.

        If this property was not set at Agent creation, but an ``LiveSession`` provides a value for
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

        If this property was not set at Agent creation, but an ``LiveSession`` provides a value for
        the use of TTS-aligned transcript, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether to use TTS-aligned transcript.
        """
        return self._use_tts_aligned_transcript

    @property
    def session(self) -> LiveSession:
        """
        Retrieve the VoiceAgent associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running
        """
        return self._get_activity_or_raise().session


TaskResult_T = TypeVar("TaskResult_T")
TaskState_T = TypeVar("TaskState_T", default=None)


def _task_state_type(cls: type[Any]) -> type[Any] | None:
    """The ``State`` of ``AgentTask[Output, State]`` as the class declares it, else ``None``."""
    for klass in cls.__mro__:
        for base in getattr(klass, "__orig_bases__", ()):
            if get_origin(base) is not AgentTask:
                continue
            args = get_args(base)
            if len(args) < 2 or isinstance(args[1], TypeVar) or args[1] is type(None):
                return None
            return args[1]  # type: ignore[no-any-return]
    return None


class TaskAwaited(Exception):
    """Raised by ``await task`` inside ``agent.run()``: the run pushes the task on the lane's
    agent stack and the call stays pending until the task completes (docs/agent-tasks.md)."""

    def __init__(self, task: AgentTask[Any, Any]) -> None:
        super().__init__(f"{type(task).__name__} takes the conversation")
        self.task = task


class AgentTask(Agent, Generic[TaskResult_T, TaskState_T]):
    """An agent with a job that ends with a value: ``AgentTask[Output, State]``.

    Its constructor arguments are its input, ``self.state`` its own typed state, and it is
    awaited once. On a call the task takes the conversation in memory; in text the await is a
    handoff plus a deferred call: the task rides the lane's agent stack with its arguments and
    state, each message is a turn of the task, and ``Completed`` becomes the pending call's
    output. A dataclass subclass makes the arguments JSON for the stack.
    """

    def __init__(
        self,
        *,
        instructions: str | Instructions = "",
        tools: list[llm.Tool | llm.Toolset] | None = None,
        stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | LLMModels | RealtimeModels | str | None] = (
            NOT_GIVEN
        ),
        tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
        preserve_function_call_history: bool = False,
        # deprecated
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
        mcp_servers: NotGivenOr[list[mcp.MCPServer] | None] = NOT_GIVEN,
    ) -> None:
        tools = tools or []
        turn_handling = (
            _migrate_turn_handling(
                turn_detection=turn_detection,
                allow_interruptions=allow_interruptions,
                min_endpointing_delay=min_endpointing_delay,
                max_endpointing_delay=max_endpointing_delay,
            )
            if not is_given(turn_handling)
            else turn_handling
        )
        super().__init__(
            instructions=instructions,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            mcp_servers=mcp_servers,
            turn_handling=turn_handling,
        )

        self.__started = False
        self.__fut = asyncio.Future[TaskResult_T]()
        self.__inactive_ev = asyncio.Event()
        self.__inactive_ev.set()  # set when the agent is not awaited or activity is closed
        self._preserve_function_call_history = preserve_function_call_history

        # the activity this task paused, and its own; both bound in __await_impl
        self._old_activity: AgentActivity | None = None
        self._task_activity: AgentActivity | None = None

        # the task's own state, and the call of the agent below that awaits it in text
        state_type = _task_state_type(type(self))
        self.state: TaskState_T = state_type() if state_type is not None else None  # type: ignore[assignment]
        self._pending_call_id: str | None = None

    def done(self) -> bool:
        return self.__fut.done()

    # -- the stack entry, in text --

    def _arguments(self) -> dict[str, Any]:
        """The constructor arguments as JSON: a dataclass subclass's fields."""
        if not dataclasses.is_dataclass(self):
            return {}
        hints = get_type_hints(type(self))
        return {
            f.name: TypeAdapter(hints[f.name]).dump_python(getattr(self, f.name), mode="json")
            for f in dataclasses.fields(self)
        }

    @classmethod
    def _rebuild(cls, args: Any) -> AgentTask[Any, Any]:
        """The task from its stack entry's arguments: a dataclass's fields, else keywords."""
        if dataclasses.is_dataclass(cls):
            # the arguments are JSON; the fields' types make them objects again
            task: AgentTask[Any, Any] = TypeAdapter(cls).validate_python(args or {})
            return task
        return cls(**(args or {}))

    async def _take(self, message: llm.ChatMessage) -> llm.Transition | None:
        """The person's message, answered by the task itself instead of its model (a graph's
        waiting node); ``None`` leaves it to the model."""
        return None

    def _dump_state(self) -> Any:
        state_type = _task_state_type(type(self))
        if state_type is None:
            return None
        return TypeAdapter(state_type).dump_python(self.state, mode="json")

    def _restore(self, state: Any, *, call_id: str | None) -> None:
        """The task as it waited: its state and the call it answers, from the stack entry."""
        state_type = _task_state_type(type(self))
        if state_type is not None and state is not None:
            self.state = TypeAdapter(state_type).validate_python(state)
        self._pending_call_id = call_id

    async def _update_tools(self, tools: list[llm.Tool | llm.Toolset]) -> None:
        """Replace this instance's tools (internal): live when bound, stored otherwise.

        A task is instantiated per use, so its tool list is its own; a shared ``Agent`` is
        configuration and does not change after import.
        """
        activity = self._bound_activity()
        if activity is None:
            self._tools = list({tool.id: tool for tool in tools}.values())
            return
        await activity.update_tools(tools)

    def cancel(self) -> None:
        if self._task_activity:
            self._task_activity.interrupt(force=True)
        if self.__fut.done():
            return
        self._complete(ToolError(f"AgentTask {self.id} is cancelled"))

    def _complete(self, result: TaskResult_T | Exception) -> None:
        """End the task with its result (internal): a tool returns ``Completed(value)``."""
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

        # imported at call time: agent_activity imports Agent at module scope, so the reverse can't
        from .agent_activity import _AgentActivityContextVar, _SpeechHandleContextVar

        task_info = _get_activity_task_info(current_task)

        speech_handle = _SpeechHandleContextVar.get(None)
        old_activity = _AgentActivityContextVar.get(None)
        from .agent_activity import AgentActivity

        if old_activity is not None and not isinstance(old_activity, AgentActivity):
            # in text the run takes it from here: the call stays pending, the task is pushed on
            # the lane's agent stack, and the value comes back on a later turn
            raise TaskAwaited(self)
        if old_activity is None or not task_info or not task_info.inline_task:
            raise RuntimeError(
                f"{self.__class__.__name__} should only be awaited inside tool_functions or the on_enter/on_exit methods of an Agent"  # noqa: E501
            )
        old_agent = old_activity.agent
        session = old_activity.session
        self._old_activity = old_activity

        def _handle_task_done(_: asyncio.Task[Any]) -> None:
            if self.__fut.done():
                return

            # if the asyncio.Task running the InlineTask completes before the InlineTask itself, log
            # an error and attempt to recover by terminating the InlineTask.
            logger.error(
                f"The asyncio.Task finished before {self.__class__.__name__} was completed."
            )

            self._complete(
                RuntimeError(
                    f"The asyncio.Task finished before {self.__class__.__name__} was completed."
                )
            )

        blocked_tasks = [current_task]
        if (
            old_activity._on_enter_task
            and not old_activity._on_enter_task.done()
            and current_task is not old_activity._on_enter_task
        ):
            blocked_tasks.append(old_activity._on_enter_task)

        async with old_activity._inline_task_slot(
            speech_handle=speech_handle, blocked_tasks=blocked_tasks
        ):
            current_task.add_done_callback(_handle_task_done)

            if (
                task_info.function_call
                and isinstance(old_activity.llm, RealtimeModel)
                and not old_activity.llm.capabilities.manual_function_calls
            ):
                logger.error(
                    f"Realtime model '{old_activity.llm.label}' does not support resuming function calls from chat context, "
                    "using AgentTask inside a function tool may have unexpected behavior."
                )

            # TODO(theomonnom): could the RunResult watcher & the blocked_tasks share the same logic?
            self.__inactive_ev.clear()
            suspended_handles: list[SpeechHandle | asyncio.Future[Any]] = []
            pending_on_enter_task: asyncio.Task[None] | None = None
            try:
                # use wait_on_enter=False to avoid deadlock: on_enter may spawn nested
                # AgentTasks that require user input, but session.run() can't return until
                # all watched handles complete — creating a circular wait.
                await session._update_activity(
                    self,
                    previous_activity="pause",
                    blocked_tasks=blocked_tasks,
                    wait_on_enter=False,
                )
                self._task_activity = session._activity_of(self)

                if not self._task_activity and not self.done():
                    self._complete(
                        ToolError(
                            f"activity doesn't start for {self.id}, likely due to session closing"
                        )
                    )

                run_state = session._global_run_state

                if self._task_activity and (on_enter_task := self._task_activity._on_enter_task):
                    if run_state and not run_state.done():
                        # watch the on_enter task as a guard so RunResult won't complete
                        # before on_enter has registered its own speech handles
                        run_state._watch_handle(on_enter_task)
                        pending_on_enter_task = on_enter_task
                    else:
                        # no active run to guard — just wait for on_enter directly
                        await asyncio.shield(on_enter_task)

                # now unwatch the parent speech handle and blocked tasks that belong to the
                # old activity — they can't complete while this AgentTask is running, and
                # keeping them watched would block RunResult from completing. A foreground
                # hold waiting on this task is in the same position, so its guard suspends too.
                if run_state and not run_state.done():
                    if speech_handle and run_state._unwatch_handle(speech_handle):
                        suspended_handles.append(speech_handle)
                    for blocked in [*blocked_tasks, *session._foreground_guards]:
                        if run_state._unwatch_handle(blocked):
                            suspended_handles.append(blocked)
                    if suspended_handles:
                        run_state._mark_done_if_needed(None)
            # asyncio.CancelledError derives from BaseException, not Exception
            except BaseException:
                self.__inactive_ev.set()
                raise

            try:
                return await asyncio.shield(self.__fut)

            finally:
                # run_state could have changed after self.__fut
                run_state = session._global_run_state

                # re-watch the suspended handles so the resumed parent activity
                # is tracked by the current RunResult again
                if run_state and not run_state.done():
                    for handle in [*suspended_handles, current_task]:
                        run_state._watch_handle(handle)

                if pending_on_enter_task:
                    try:
                        await asyncio.shield(pending_on_enter_task)
                    except BaseException:
                        logger.exception("error in on_enter task of agent %s", self.id)

                task_activity = self._task_activity
                if session._closing and task_activity is None:
                    # the activity never started (session closing), skip the handoff;
                    # the close path owns the previous activity
                    pass
                elif session.current_agent != self:
                    logger.warning(
                        f"{self.__class__.__name__} completed, but the agent has changed in the meantime. "
                        "Ignoring handoff to the previous agent, likely due to `LiveSession.update_agent` being invoked."
                    )
                    await old_activity.aclose()
                else:
                    # the task worked on the session's history: what it added since the pause
                    # stays, filtered of its function calls unless they are to be kept
                    if task_activity is not None:
                        merged_chat_ctx = old_activity._chat_ctx.copy().merge(
                            task_activity._chat_ctx,
                            exclude_function_call=not self._preserve_function_call_history,
                            exclude_instructions=True,
                        )
                        # set directly, `session._update_activity` syncs it to the rt_session if needed
                        old_activity._chat_ctx.items[:] = merged_chat_ctx.items

                    await session._update_activity(
                        old_agent, new_activity="resume", wait_on_enter=False
                    )
                self.__inactive_ev.set()

    def __await__(self) -> Generator[None, None, TaskResult_T]:
        return self.__await_impl().__await__()

    async def _wait_for_inactive(self) -> None:
        await self.__inactive_ev.wait()


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


def _pass_through_activity_task_info(task: asyncio.Task[Any]) -> None:
    current_task = asyncio.current_task()
    if current_task and (info := _get_activity_task_info(current_task)):
        setattr(task, "__livekit_agents_activity_task", info)


class _InstructionsText(str):
    """``Agent.instructions`` for a ``str``: the value, callable as the decorator."""

    __slots__ = ("_source", "_agent")
    _source: str | Instructions
    _agent: Agent

    def __call__(self, f: _InstructionsFnT) -> _InstructionsFnT:
        self._agent._register_instructions(f)
        return f


class _InstructionsObject(Instructions):
    """``Agent.instructions`` for an ``Instructions``: the value, callable as the decorator."""

    _source: str | Instructions
    _agent: Agent

    def __call__(self, f: _InstructionsFnT) -> _InstructionsFnT:
        self._agent._register_instructions(f)
        return f


if TYPE_CHECKING:

    class _InstructionsView(str, Instructions):
        """What a type checker sees: the value, read as a ``str`` or an ``Instructions``, and
        the decorator. One class, since a union of callables is an untyped decorator to mypy."""

        _source: str | Instructions
        _agent: Agent

        def __call__(self, f: _InstructionsFnT) -> _InstructionsFnT: ...

else:
    _InstructionsView = _InstructionsText | _InstructionsObject


def _wrap_instructions(source: str | Instructions, agent: Agent) -> _InstructionsView:
    view: _InstructionsText | _InstructionsObject
    if isinstance(source, Instructions):
        view = _InstructionsObject(source.common, audio=source.audio, text=source.text)
    else:
        view = _InstructionsText(source)
    view._source = source
    view._agent = agent
    return cast("_InstructionsView", view)
