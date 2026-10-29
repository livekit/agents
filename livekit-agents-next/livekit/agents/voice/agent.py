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
    TypedDict,
    overload,
)

from typing_extensions import ParamSpec, TypeVar

from livekit import rtc

from .. import _core, inference, llm, stt, tts
from .._generated.agent_run import AgentDefinition, Retries
from .._generated.vad import VAD
from ..llm import (
    LLM,
    DuplexModel,
    DuplexRealtimeAdapter,
    RealtimeModel,
)
from ..llm.chat_context import _ReadOnlyChatContext
from ..llm.tool_context import (
    DuplicateMode,
    DuplicateScope,
    FunctionTool,
    RawFunctionDescription,
    RawFunctionTool,
    ReadBack,
    ToolFlag,
    find_function_tools,
    function_tool,
)
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    FlushSentinel,
    NotGivenOr,
    TimedString,
)
from ..utils import is_given
from .events import Context_T, UserTurnExceededEvent

if TYPE_CHECKING:
    from .._generated.agent_run import RunSettled
    from ..inference import LLMModels, RealtimeModels
    from .agent_host import AgentHost
    from .agent_run import _TextRun
    from .agent_session import VoiceSession
    from .events import RunContext
    from .run_result import RunResult, RunUsage
    from .run_stream import RunStream
    from .tool_executor import ToolHandlingOptions


_P = ParamSpec("_P")
_R = TypeVar("_R")
_InstructionsFn = Callable[["RunContext"], str | Awaitable[str]]
_HookFn = Callable[["VoiceSession"], Any]
_OutputValidatorFn = Callable[["RunContext", Any], Any]
Output_T = TypeVar("Output_T", default=Any)

DEFAULT_AGENT_NAME = "default_agent"

# what runs the agents of this context: a voice session, or a text run
_BindingContextVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_agent_binding")


class _VoiceBinding(NamedTuple):
    """The voice session running ``agent``'s hook or tool, and what its commands are sent as."""

    session: VoiceSession
    agent: Agent
    origin: str | None = None
    # the agent runs a `VoiceTask`: a `Completed` from its tools, validated, ends the task
    task: bool = False

    # the providers the agent's default nodes call: the session's, and the agent's own LLM
    @property
    def stt(self) -> stt.STT | None:
        return self.session.stt

    @property
    def vad(self) -> VAD | None:
        return self.session.vad

    @property
    def llm(self) -> LLM | RealtimeModel | None:
        return self.agent._llm if is_given(self.agent._llm) else self.session.llm

    @property
    def tts(self) -> tts.TTS | None:
        return self.session.tts

    @property
    def llm_conn_options(self) -> APIConnectOptions:
        return self.session.conn_options.llm_conn_options


class _LiveHook:
    """``Agent.on_enter`` and ``Agent.on_exit``: the decorator registering the function the
    hook runs, which receives the ``VoiceSession``, or, called bare, the hook to await.

    A descriptor so a subclass still overrides ``async def on_enter(self) -> None``; the
    bound form is untyped on purpose, both uses go through it.
    """

    def __set_name__(self, owner: type[Agent], name: str) -> None:
        self._name = name

    @overload
    def __get__(self, agent: None, owner: type[Agent]) -> _LiveHook: ...

    @overload
    def __get__(self, agent: Agent, owner: type[Agent]) -> _BoundLiveHook: ...

    def __get__(self, agent: Agent | None, owner: type[Agent]) -> _LiveHook | _BoundLiveHook:
        if agent is None:
            return self
        return _BoundLiveHook(agent, self._name)


_HookFnT = TypeVar("_HookFnT", bound=_HookFn)


class _BoundLiveHook:
    def __init__(self, agent: Agent, name: str) -> None:
        self._agent = agent
        self._name = name

    @overload
    def __call__(self, f: _HookFnT, /) -> _HookFnT: ...

    @overload
    def __call__(self) -> Awaitable[Any]: ...

    def __call__(self, *f: _HookFn) -> Any:
        if f:
            self._agent._register_hook(self._name, f[0])
            return f[0]
        return self._agent._run_hook(self._name)


def _assignment_namespace(frame: Any) -> dict[str, Any] | None:
    """The namespace a module-level ``Agent(...)`` is assigned into, or ``None``.

    Only a module (or an ``exec``) frame has a live namespace to scan later; a function frame's
    locals are a snapshot, and holding its proxy would keep the frame alive. ``Agent[Ctx](...)``
    constructs through ``typing``'s generic alias, whose frame is skipped.
    """
    while frame is not None and frame.f_globals.get("__name__") in ("typing", "typing_extensions"):
        frame = frame.f_back
    if frame is None or frame.f_locals is not frame.f_globals:
        return None
    namespace: dict[str, Any] = frame.f_locals
    return namespace


class AgentRetries(TypedDict, total=False):
    """Retry budgets, as pydantic-ai's: how many consecutive retries the model gets before the
    run fails. ``Agent(retries=)`` defaults each key to ``1``; ``run(retries=)`` keeps the
    agent's value for an omitted key.
    """

    tools: int
    """Per tool: invalid arguments and ``ModelRetry`` from the tool. A tool's own ``retries=``
    replaces it for that tool."""
    output: int
    """The final output: a candidate failing validation and ``ModelRetry`` from an output
    validator."""


def _retries(value: int | AgentRetries | None, base: Retries) -> Retries:
    """``retries=`` over ``base``: an int sets both budgets, a dict the keys it has."""
    if value is None:
        return base
    if isinstance(value, int):
        value = {"tools": value, "output": value}
    resolved = Retries(
        tools=value.get("tools", base.tools), output=value.get("output", base.output)
    )
    for budget in (resolved.tools, resolved.output):
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(f"retries must be non-negative integers, got {value!r}")
    return resolved


@dataclass
class ModelSettings:
    tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN
    """The tool choice to use when calling the LLM."""


class Agent(Generic[Context_T, Output_T]):
    """Shared agent configuration, parameterized by application context and final output.

    ``retries`` sets the retry budgets of every run (``AgentRetries``); an int sets both.
    ``tool_handling`` sets the templates rendered around its tools' background calls on a voice
    session (``ToolHandlingOptions``); an ``AsyncToolset`` overrides them for its own tools.
    """

    def __init__(
        self,
        *,
        instructions: str,
        name: str | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: type[Output_T]
        | Callable[..., Awaitable[Output_T]]
        | Callable[..., Output_T]
        | None = None,
        llm: NotGivenOr[
            llm.LLM | llm.RealtimeModel | llm.DuplexModel | LLMModels | RealtimeModels | str | None
        ] = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None:
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        tools = tools or []
        # an unnamed agent takes the name of the module-level variable it is assigned to: the
        # assigning frame's namespace is kept and scanned the first time the name is read, since
        # the assignment completes only after the constructor returns
        self._name = name
        self._defining_namespace = _assignment_namespace(sys._getframe(1)) if name is None else None

        self._instructions = instructions
        self._tools = [*tools, *find_function_tools(self)]
        # registered at import by the decorators below; frozen once the agent is first bound
        self._instruction_fns: list[_InstructionsFn] = []
        self._instructions_view: _InstructionsView[Any] | None = None
        self._hook_fns: dict[str, _HookFn] = {}
        self._output_validators: list[_OutputValidatorFn] = []
        self._frozen = False
        # the retry budgets every invocation of this agent reads, in text and on a call
        self._retries = _retries(retries, Retries())
        self._tool_handling = tool_handling

        # a type, or an output function whose return value is the output
        self._output_type_explicit = output_type is not None
        self._output_spec: Any = output_type if output_type is not None else str
        self._output_type: type[Any] = _output_type_of(self._output_spec)
        self._host_instance: AgentHost | None = None

        if isinstance(llm, str):
            llm = inference.llm_from_model_string(llm)
        # a duplex model is wrapped on the way in, so nothing downstream sees one
        self._llm: NotGivenOr[LLM | RealtimeModel | None] = (
            DuplexRealtimeAdapter(llm) if isinstance(llm, DuplexModel) else llm
        )

    @property
    def name(self) -> str:
        """The agent's identity (agents.md "Definition"): ``name=``, else the module-level
        variable it is assigned to, else ``default_agent``."""
        if self._name is None and self._defining_namespace is not None:
            for var_name, value in self._defining_namespace.items():
                if value is self:
                    self._name = var_name
                    self._defining_namespace = None
                    break
        return self._name or DEFAULT_AGENT_NAME

    @property
    def instructions(self) -> _InstructionsView[Context_T]:
        """The static instructions, also the decorator registering dynamic ones.

        Reads as the static string the agent was built with. Used as
        ``@agent.instructions`` it registers a function rendered into the instructions on every
        turn, which receives the ``RunContext`` and returns a string, sync or async::

            @scheduler.instructions
            def known_appointment(ctx: RunContext) -> str: ...
        """
        if self._instructions_view is None or self._instructions_view._source is not (
            self._instructions
        ):
            self._instructions_view = _InstructionsView(self._instructions, self)
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
        """A read-only view of the history the current run works on (agents.md "History"):
        the agent's own view of the conversation, not the session's history of every agent.

        This resolves through the activity bound in the current context, so it is only available
        inside the agent's hooks and tools.

        Raises:
            RuntimeError: If the agent is not running in this context.
        """
        binding = self._get_binding_or_raise()
        if isinstance(binding, _VoiceBinding):
            return _ReadOnlyChatContext(binding.session._agent_chat_ctx().items)
        return binding.chat_ctx

    @property
    def output_type(self) -> type[Output_T]:
        """What ``run()`` returns as ``result.output``: ``str`` unless declared."""
        return self._output_type

    def as_tool(
        self,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        parameters: type[object] | None = None,
        needs_approval: bool | llm.tool_context.ReadBack = False,
        chat_history: Literal["fresh", "fork"] = "fresh",
        merge_chat_history: bool = False,
    ) -> llm.FunctionTool[..., Awaitable[Output_T]]:
        """A typed child conversation managed by TextSession (text-session.md)."""
        from .agent_tool import _LocalAgentTool

        return _LocalAgentTool(
            self,
            tool_name=tool_name,
            tool_description=tool_description,
            parameters=parameters,
            needs_approval=needs_approval,
            chat_history=chat_history,
            merge_chat_history=merge_chat_history,
        )  # type: ignore[return-value]

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        context: Context_T | None = None,
        usage: RunUsage | None = None,
        retries: int | AgentRetries | None = None,
    ) -> RunResult[Output_T]:
        """One invocation on the given history: the model and tool loop until the accepted
        output or a request for human input (agents.md "Running a turn").

        Args:
            message: The user's message.
            message_history: The conversation so far; omit for a fresh one.
            context: Application data and dependencies, read as ``ctx.context``.
            usage: A ``RunUsage`` this run's usage is added to, for a delegated run.
            retries: The agent's retry budgets for this run (``AgentRetries``); an int sets both.
        """
        from .agent_run import door_hooks

        hooks = door_hooks()
        async with self.run_stream(
            message,
            message_history=message_history,
            context=context,
            usage=usage,
            retries=retries,
        ) as stream:
            if hooks is not None and hooks.on_token is not None:
                # inside a managed turn the reply reaches the client as it is written
                async for delta in stream.stream_text(delta=True):
                    await hooks.on_token(delta)
            return await stream.result()

    def run_stream(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        context: Context_T | None = None,
        usage: RunUsage | None = None,
        retries: int | AgentRetries | None = None,
    ) -> RunStream[RunResult[Output_T]]:
        """``run()`` as an async context manager streaming its text and events; execution
        starts on entry (agents.md "Streaming runs")::

            async with agent.run_stream(message, message_history=history) as stream:
                async for delta in stream.stream_text(delta=True):
                    print(delta, end="")
                result = await stream.result()
        """
        from .agent_run import run_error, settled_result, start_run
        from .run_stream import RunStream, _Started

        async def start() -> _Started[RunResult[Output_T]]:
            run, updates = await start_run(
                self,
                message,
                message_history=message_history,
                context=context,
                usage=usage,
                retries=retries,
                conn_options=DEFAULT_API_CONNECT_OPTIONS,
            )

            def settle(finished: RunSettled | None) -> RunResult[Output_T]:
                assert finished is not None
                result = settled_result(run, finished)
                if run.hooks is not None and run.hooks.on_result is not None:
                    run.hooks.on_result(result)
                return result

            return _Started(
                id=run.id,
                updates=updates,
                settle=settle,
                fail=run_error,
                cleanup=run.release_toolsets,
            )

        return RunStream(start, agent_name=self.name)

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

    @property
    def _host(self) -> AgentHost:
        """The host side of this agent's invocations, built once its registrations froze."""
        if self._host_instance is None:
            from .agent_host import AgentHost

            self._frozen = True
            self._host_instance = AgentHost(self)
        return self._host_instance

    def _definition(
        self, tools: llm.ToolContext, retries: int | AgentRetries | None = None
    ) -> AgentDefinition:
        """The core's ``AgentDefinition`` of this agent with ``tools``, under a run's
        ``retries=``."""
        from .agent_run import tool_infos

        return AgentDefinition(
            name=self.name,
            instructions=self._instructions,
            dynamic_instructions=bool(self._instruction_fns),
            tools=tool_infos(tools),
            output=self._host._output.spec(validated=bool(self._output_validators)),
            retries=_retries(retries, self._retries),
        )

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
                f"cannot register {what} on agent `{self.name}`: it is already bound to a session"
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
        retries: int | None = None,
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
        retries: int | None = None,
    ) -> Any:
        """Register a function tool on this agent, as a decorator.

        The first parameter of the function is the ``RunContext``; the rest are the tool's
        arguments and its docstring feeds the schema, as with ``@function_tool``::

            @scheduler.tool
            async def find_appointment(ctx: RunContext, last_name: str) -> str: ...
        """

        options: dict[str, Any] = {
            "name": name,
            "description": description,
            "raw_schema": raw_schema,
            "flags": flags,
            "on_duplicate": on_duplicate,
            "duplicate_scope": duplicate_scope,
            "needs_approval": needs_approval,
            "retries": retries,
        }

        def register(func: Callable[_P, _R]) -> FunctionTool[_P, _R] | RawFunctionTool[_P, _R]:
            self._check_not_frozen("a tool")
            tool: FunctionTool[_P, _R] | RawFunctionTool[_P, _R]
            if isinstance(func, (FunctionTool, RawFunctionTool)):
                # an already built tool, registered as it is: its options were given when built
                default = inspect.signature(Agent.tool).parameters
                given = [k for k, v in options.items() if v != default[k].default]
                if given:
                    raise ValueError(f"{given} apply to a function, not to a built tool")
                tool = func
            else:
                # `function_tool` refuses what a raw schema cannot take (`needs_approval`)
                tool = function_tool(func, **options)
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

    # -- Pipeline nodes --
    # They can all be overriden by subclasses, by default they use the STT/LLM/TTS specified in the
    # constructor of the VoiceAgent

    on_enter = _LiveHook()
    """Called when the agent takes the active live role (agents.md "Hooks").

    A decorator registering the function to run then, which receives the session::

        @scheduler.on_enter
        async def greet(session: VoiceSession) -> None:
            session.generate_reply(instructions="Greet the caller.")
    """

    on_exit = _LiveHook()
    """Called when the agent relinquishes the active live role, on an orderly replacement or
    session close (agents.md "Hooks").

    A decorator registering the function to run then, which receives the session::

        @front_desk.on_exit
        async def leave(session: VoiceSession) -> None: ...
    """

    def _register_hook(self, name: str, f: _HookFn) -> _HookFn:
        self._check_not_frozen(f"an {name} function")
        self._hook_fns[name] = f
        return f

    async def _run_hook(self, name: str) -> Any:
        fn = self._hook_fns.get(name)
        if fn is None:
            return None
        result = fn(self._get_session_or_raise())
        if inspect.isawaitable(result):
            result = await result
        return result

    def _has_on_enter(self) -> bool:
        """An ``on_enter`` was registered or overridden."""
        return "on_enter" in self._hook_fns or not isinstance(
            inspect.getattr_static(type(self), "on_enter"), _LiveHook
        )

    async def _enter(self, ctx: RunContext[Any]) -> Any:
        """``on_enter`` as it was written: an override takes the context when it declares one,
        the decorator form takes the call's session and is nothing in text."""
        hook = type(self).on_enter
        if isinstance(hook, _LiveHook):
            return await self._run_hook("on_enter") if ctx._session is not None else None
        if len(inspect.signature(hook).parameters) > 1:
            return await hook(self, ctx)
        return await hook(self)

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
        await self._get_session_or_raise().generate_reply(
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
        return Agent._default.llm_node(self, chat_ctx, tools, model_settings)

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
        return Agent._default.stt_node(self, audio, model_settings)

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame] | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]:
        """One spoken segment on a call: its text, synthesized into audio frames by the
        agent's TTS.

        Override it to change the text, the voice or the audio. A non-streaming TTS is fed
        sentence by sentence.
        """
        return Agent._default.tts_node(self, text, model_settings)

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
        return Agent._default.transcription_node(self, text, model_settings)

    def realtime_audio_output_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[rtc.AudioFrame]
        | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]
        | Coroutine[Any, Any, None]
    ):
        """One generation's audio from the realtime model, before it is played out."""
        return Agent._default.realtime_audio_output_node(self, audio, model_settings)

    class _default:
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

    def _bound_binding(self) -> _VoiceBinding | _TextRun[Any] | None:
        """What runs this agent in the current context: a voice session or a run (internal).

        Resolved through the contextvar both set around every hook and tool, so one shared
        instance serving several sessions reads the right one.
        """
        binding = _BindingContextVar.get(None)
        if binding is None or binding.agent is not self:
            return None
        return binding

    def _get_binding_or_raise(self) -> _VoiceBinding | _TextRun[Any]:
        binding = self._bound_binding()
        if binding is None:
            raise RuntimeError(
                f"agent `{self.name}` is not running in this context; its session and history "
                "are only available inside its hooks and tools"
            )
        return binding

    def _get_session_or_raise(self) -> VoiceSession:
        return self._get_voice_binding_or_raise().session

    def _get_voice_binding_or_raise(self) -> _VoiceBinding:
        binding = self._get_binding_or_raise()
        if not isinstance(binding, _VoiceBinding):
            raise RuntimeError(
                f"agent `{self.name}` is running in agent.run(), which has no live session"
            )
        return binding

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


def _output_type_of(output: Any) -> type[Any]:
    """The type of ``result.output``: the type itself, or an output function's return type."""
    if isinstance(output, type) or not callable(output):
        return output  # type: ignore[no-any-return]
    from typing import get_type_hints

    return get_type_hints(output).get("return", Any)  # type: ignore[no-any-return]


_RenderedT = TypeVar("_RenderedT", bound=str | Awaitable[str])


class _InstructionsView(str, Generic[Context_T]):
    """The static instructions, callable to register a dynamic instruction function."""

    _source: str
    _agent: Agent

    def __new__(cls, source: str, agent: Agent) -> _InstructionsView[Context_T]:
        view = super().__new__(cls, source)
        view._source = source
        view._agent = agent
        return view

    def __call__(
        self, f: Callable[[RunContext[Context_T]], _RenderedT]
    ) -> Callable[[RunContext[Context_T]], _RenderedT]:
        self._agent._register_instructions(f)
        return f
