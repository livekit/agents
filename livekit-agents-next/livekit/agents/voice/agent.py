from __future__ import annotations

import contextvars
import inspect
import sys
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Sequence,
)
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    cast,
    overload,
)

from typing_extensions import ParamSpec, TypeVar

from livekit import rtc

from .. import _core, inference, llm, stt, tts
from .._generated.tool_executor import ToolHandlingOptions
from .._generated.vad import VAD
from ..llm import (
    LLM,
    DuplexModel,
    DuplexRealtimeAdapter,
    RealtimeModel,
    find_function_tools,
)
from ..llm.chat_context import Instructions
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
from ..types import NOT_GIVEN, APIConnectOptions, FlushSentinel, NotGivenOr, TimedString
from ..utils import is_given, misc
from .events import Context_T, UserTurnExceededEvent
from .turn import TurnHandlingOptions

if TYPE_CHECKING:
    from ..inference import LLMModels, RealtimeModels, STTModels, TTSModels
    from .agent_run import AgentRun
    from .agent_session import ExpressiveOptions, VoiceSession
    from .events import RunContext
    from .run_result import RunResult, RunUsage
    from .turn import TurnDetectionMode


_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])
_InstructionsFn = Callable[["RunContext"], str | Awaitable[str]]
_InstructionsFnT = TypeVar("_InstructionsFnT", bound=Callable[..., str | Awaitable[str]])
_OnEnterFn = Callable[["VoiceSession"], Any]
_OutputValidatorFn = Callable[["RunContext", Any], Any]
Output_T = TypeVar("Output_T", default=Any)

DEFAULT_AGENT_ID = "default_agent"

# what runs the agents of this context: a voice session, or a text run
_BindingContextVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_agent_binding")


class _VoiceBinding(NamedTuple):
    """The voice session running ``agent``'s hook or tool, and what its commands are sent as."""

    session: VoiceSession
    agent: Agent
    origin: str | None = None

    # the providers the agent's default nodes call: its own, else the session's
    @property
    def stt(self) -> stt.STT | None:
        return self.agent._stt if is_given(self.agent._stt) else self.session.stt

    @property
    def vad(self) -> VAD | None:
        return self.agent._vad if is_given(self.agent._vad) else self.session.vad

    @property
    def llm(self) -> LLM | RealtimeModel | None:
        return self.agent._llm if is_given(self.agent._llm) else self.session.llm

    @property
    def tts(self) -> tts.TTS | None:
        return self.agent._tts if is_given(self.agent._tts) else self.session.tts

    @property
    def llm_conn_options(self) -> APIConnectOptions:
        return self.session.conn_options.llm_conn_options


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


class Agent(Generic[Context_T, Output_T]):
    """Shared agent configuration, parameterized by application context and final output."""

    def __init__(
        self,
        *,
        instructions: str | Instructions,
        name: str | None = None,
        id: str | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: type[Output_T] | None = None,
        stt: NotGivenOr[stt.STT | STTModels | str | None] = NOT_GIVEN,
        vad: NotGivenOr[VAD | None] = NOT_GIVEN,
        turn_handling: NotGivenOr[TurnHandlingOptions] = NOT_GIVEN,
        tool_handling: NotGivenOr[ToolHandlingOptions] = NOT_GIVEN,
        llm: NotGivenOr[
            llm.LLM | llm.RealtimeModel | llm.DuplexModel | LLMModels | RealtimeModels | str | None
        ] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | TTSModels | str | None] = NOT_GIVEN,
        expressive: NotGivenOr[bool | ExpressiveOptions] = NOT_GIVEN,
        min_consecutive_speech_delay: NotGivenOr[float] = NOT_GIVEN,
        use_tts_aligned_transcript: NotGivenOr[bool] = NOT_GIVEN,
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

        self._instructions = instructions
        self._tools = [*tools, *find_function_tools(self)]
        # registered at import by the decorators below; frozen once the agent is first bound
        self._instruction_fns: list[_InstructionsFn] = []
        self._instructions_view: _InstructionsView[Any] | None = None
        self._on_enter_fn: _OnEnterFn | None = None
        self._output_validators: list[_OutputValidatorFn] = []
        self._frozen = False

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

        self._min_consecutive_speech_delay = min_consecutive_speech_delay
        self._use_tts_aligned_transcript = use_tts_aligned_transcript
        # the agent layer of the turn options, as written: the core resolves it over the session's
        self._turn_handling: TurnHandlingOptions = turn_handling if is_given(turn_handling) else {}

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
    def instructions(self) -> _InstructionsView[Context_T]:
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
        """A copy of the context the running agent works on: its own view of the
        conversation, not the session's history of every agent.

        This resolves through the activity bound in the current context, so it is only available
        inside the agent's hooks and tools.

        Raises:
            RuntimeError: If the agent is not running in this context.
        """
        binding = self._get_binding_or_raise()
        if isinstance(binding, _VoiceBinding):
            return binding.session._agent_chat_ctx()
        return binding.chat_ctx

    @property
    def interruption_detection(self) -> NotGivenOr[Literal["adaptive", "vad"]]:
        return (self._turn_handling.get("interruption") or {}).get("mode", NOT_GIVEN)

    @property
    def output_type(self) -> type[Output_T]:
        """What ``run()`` returns as ``result.output``: ``str`` unless declared."""
        return self._output_type

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        context: Context_T | None = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> RunResult[Output_T]:
        """One turn on the given history: the model and tool loop until the answer.

        The run executes this agent using the supplied history and application context.
        Requests for human input are returned in ``result.output``. ``stream()`` exposes
        the same invocation with reply text delivered incrementally.

        Args:
            message: The user's message.
            message_history: The conversation so far; omit for a fresh one.
            context: Application data and dependencies, read as ``ctx.context``.
            usage: A ``RunUsage`` to add this run's usage to, for a delegated run.
            max_steps: The most model requests one turn may make.
        """
        run = self.stream(
            message,
            message_history=message_history,
            context=context,
            usage=usage,
            max_steps=max_steps,
        )
        return await run._run()

    def stream(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        context: Context_T | None = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> AgentRun[Context_T, Output_T]:
        """The turn of ``run()`` as an async iterator of the reply's text, one piece as the
        model writes it; ``run.result`` is the ``RunResult`` once the loop ends::

            run = agent.stream("book me in", message_history=history)
            async for delta in run:
                print(delta, end="")
            answer = run.result.output
        """
        from .agent_run import AgentRun

        return AgentRun[Context_T, Output_T](
            self,
            message,
            message_history=message_history,
            context=context,
            usage=usage,
            max_steps=max_steps,
            output_type=self._output_type,
        )

    def output_validator(
        self,
        f: Callable[[RunContext[Context_T], Output_T], Output_T | Awaitable[Output_T]],
    ) -> Callable[[RunContext[Context_T], Output_T], Output_T | Awaitable[Output_T]]:
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
        needs_approval: bool | ReadBack = False,
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
        needs_approval: bool | ReadBack = False,
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
                    needs_approval=needs_approval,
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
        async def greet(session: VoiceSession) -> None:
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

    async def _enter(self, ctx: RunContext[Any]) -> Any:
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
        """One model request: plain text, or ``llm.ChatChunk``s with text and tool calls, in
        ``agent.run()`` and on a call. Override it to change how the model is called; a
        ``FlushSentinel`` ends the spoken segment there.
        """
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent]]
        | Coroutine[Any, Any, None]
    ):
        """The user's audio on a call, transcribed into speech events by the agent's STT.

        Override it to pre-process the audio or filter the events. A non-streaming STT is
        wrapped with the VAD.
        """
        return Agent.default.stt_node(self, audio, model_settings)

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame] | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]:
        """One spoken segment on a call: its text, synthesized into audio frames by the
        agent's TTS.

        Override it to change the text, the voice or the audio. A non-streaming TTS is fed
        sentence by sentence.
        """
        return Agent.default.tts_node(self, text, model_settings)

    def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> (
        AsyncIterable[str | TimedString]
        | Coroutine[Any, Any, AsyncIterable[str | TimedString]]
        | Coroutine[Any, Any, None]
    ):
        """One spoken segment's transcript on its way to the transcription output.

        Override it to clean up formatting, fix punctuation or rewrite the text the user reads.
        """
        return Agent.default.transcription_node(self, text, model_settings)

    def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[rtc.AudioFrame]
        | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]
        | Coroutine[Any, Any, None]
    ):
        """One generation's audio from the realtime model, before it is played out."""
        return Agent.default.realtime_audio_output_node(self, audio, model_settings)

    class default:
        @staticmethod
        def stt_node(
            agent: Agent, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncIterable[stt.SpeechEvent]:
            """Default implementation for `Agent.stt_node`"""
            binding = agent._get_voice_binding_or_raise()
            if binding.stt is None:
                raise RuntimeError("stt_node called but no STT is available")
            conn_options = binding.session.conn_options.stt_conn_options
            return _core.stt_node(binding.stt, audio, conn_options=conn_options, vad=binding.vad)

        @staticmethod
        def tts_node(
            agent: Agent, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncIterable[rtc.AudioFrame]:
            """Default implementation for `Agent.tts_node`: a TTS that does not stream is fed
            sentence by sentence."""
            binding = agent._get_voice_binding_or_raise()
            if binding.tts is None:
                raise RuntimeError("tts_node called but no TTS is available")
            conn_options = binding.session.conn_options.tts_conn_options
            return _core.tts_node(binding.tts, text, conn_options=conn_options)

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
            async for frame in audio:
                yield frame

        @staticmethod
        async def llm_node(
            agent: Agent,
            chat_ctx: llm.ChatContext,
            tools: list[llm.Tool],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[llm.ChatChunk | str | FlushSentinel, None]:
            """Default implementation for `Agent.llm_node`"""
            binding = agent._get_binding_or_raise()
            if isinstance(binding, _VoiceBinding):
                model = binding.session.llm
                conn_options = binding.session.conn_options.llm_conn_options
            else:
                model = binding.llm
                conn_options = binding.llm_conn_options
            assert isinstance(model, llm.LLM), "llm_node needs an LLM (not a realtime model)"
            async with model.chat(
                chat_ctx=chat_ctx,
                tools=tools,
                tool_choice=model_settings.tool_choice,
                conn_options=conn_options,
            ) as stream:
                async for chunk in stream:
                    yield chunk

    def _bound_binding(self) -> _VoiceBinding | AgentRun[Any] | None:
        """What runs this agent in the current context: a voice session or a run (internal).

        Resolved through the contextvar both set around every hook and tool, so one shared
        instance serving several sessions reads the right one.
        """
        binding = _BindingContextVar.get(None)
        if binding is None or binding.agent is not self:
            return None
        return binding

    def _get_binding_or_raise(self) -> _VoiceBinding | AgentRun[Any]:
        binding = self._bound_binding()
        if binding is None:
            raise RuntimeError(
                f"agent `{self.id}` is not running in this context; its session and history "
                "are only available inside its hooks and tools"
            )
        return binding

    def _get_session_or_raise(self) -> VoiceSession:
        return self._get_voice_binding_or_raise().session

    def _get_voice_binding_or_raise(self) -> _VoiceBinding:
        binding = self._get_binding_or_raise()
        if not isinstance(binding, _VoiceBinding):
            raise RuntimeError(
                f"agent `{self.id}` is running in agent.run(), which has no live session"
            )
        return binding

    @property
    def turn_detection(self) -> NotGivenOr[TurnDetectionMode | None]:
        """
        Retrieves the turn detection mode for identifying conversational turns.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a turn detection,
        the session's turn detection mode will be used at runtime instead.

        Returns:
            NotGivenOr[TurnDetectionMode | None]: An optional turn detection mode for managing conversation flow.
        """  # noqa: E501
        return self._turn_handling.get("turn_detection", NOT_GIVEN)

    @property
    def stt(self) -> NotGivenOr[stt.STT | None]:
        """
        Retrieves the Speech-To-Text component for the agent.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides an STT component,
        the session's STT will be used at runtime instead.

        Returns:
            NotGivenOr[stt.STT | None]: An optional STT component.
        """  # noqa: E501
        return self._stt

    @property
    def llm(self) -> NotGivenOr[llm.LLM | llm.RealtimeModel | None]:
        """
        Retrieves the Language Model or RealtimeModel used for text generation.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides an LLM or RealtimeModel,
        the session's model will be used at runtime instead.

        Returns:
            NotGivenOr[llm.LLM | llm.RealtimeModel | None]: The language model for text generation.
        """  # noqa: E501
        return self._llm

    @property
    def tts(self) -> NotGivenOr[tts.TTS | None]:
        """
        Retrieves the Text-To-Speech component for the agent.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a TTS component,
        the session's TTS will be used at runtime instead.

        Returns:
            NotGivenOr[tts.TTS | None]: An optional TTS component for generating audio output.
        """  # noqa: E501
        return self._tts

    @property
    def expressive(self) -> NotGivenOr[bool | ExpressiveOptions]:
        """
        Retrieves the expressive TTS delivery setting for the agent.

        If this property was not set at Agent creation, the ``VoiceSession``'s ``expressive``
        value will be used at runtime instead. When set, it overrides the session value for
        this agent's turns, matching how ``llm`` and ``tts`` overrides behave.

        Returns:
            NotGivenOr[bool | ExpressiveOptions]: Whether expressive delivery is enabled,
                or its configuration.
        """
        return self._expressive

    @property
    def vad(self) -> NotGivenOr[VAD | None]:
        """
        Retrieves the Voice Activity Detection component for the agent.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a VAD component,
        the session's VAD will be used at runtime instead.

        Returns:
            NotGivenOr[vad.VAD | None]: An optional VAD component for detecting voice activity.
        """  # noqa: E501
        return self._vad

    @property
    def allow_interruptions(self) -> NotGivenOr[bool]:
        """
        Indicates whether interruptions (e.g., stopping TTS playback) are allowed.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a value for
        allowing interruptions, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether interruptions are permitted.
        """
        return (self._turn_handling.get("interruption") or {}).get("enabled", NOT_GIVEN)

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
        return self._turn_handling.get("endpointing", {}).get("min_delay", NOT_GIVEN)

    @property
    def max_endpointing_delay(self) -> NotGivenOr[float]:
        """
        Maximum time-in-seconds the agent will wait before terminating the turn.

        If this property was set at Agent creation, it will be used at runtime instead of the session's value.
        """
        return self._turn_handling.get("endpointing", {}).get("max_delay", NOT_GIVEN)

    @property
    def min_consecutive_speech_delay(self) -> NotGivenOr[float]:
        """
        Retrieves the minimum consecutive speech delay for the agent.

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a value for
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

        If this property was not set at Agent creation, but an ``VoiceSession`` provides a value for
        the use of TTS-aligned transcript, the session's value will be used at runtime instead.

        Returns:
            NotGivenOr[bool]: Whether to use TTS-aligned transcript.
        """
        return self._use_tts_aligned_transcript

    @property
    def session(self) -> VoiceSession:
        """
        Retrieve the VoiceAgent associated with the current agent.

        Raises:
            RuntimeError: If the agent is not running
        """
        return self._get_session_or_raise()


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
    _RenderedT = TypeVar("_RenderedT", bound=str | Awaitable[str])

    class _InstructionsView(str, Instructions, Generic[Context_T]):
        """What a type checker sees: the value, read as a ``str`` or an ``Instructions``, and
        the decorator, which takes the agent's ``RunContext``. One class, since a union of
        callables is an untyped decorator to mypy."""

        _source: str | Instructions
        _agent: Agent

        def __call__(
            self, f: Callable[[RunContext[Context_T]], _RenderedT]
        ) -> Callable[[RunContext[Context_T]], _RenderedT]: ...

else:
    _InstructionsView = _InstructionsText | _InstructionsObject


def _wrap_instructions(source: str | Instructions, agent: Agent) -> _InstructionsView[Any]:
    view: _InstructionsText | _InstructionsObject
    if isinstance(source, Instructions):
        view = _InstructionsObject(source.common, audio=source.audio, text=source.text)
    else:
        view = _InstructionsText(source)
    view._source = source
    view._agent = agent
    return cast("_InstructionsView[Any]", view)
