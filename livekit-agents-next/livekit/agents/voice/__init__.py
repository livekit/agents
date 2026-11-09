from __future__ import annotations

import contextvars
import inspect
import sys
from collections.abc import (
    Awaitable,
    Callable,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeAlias,
    cast,
    overload,
)

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python
from typing_extensions import ParamSpec, TypeVar, Unpack

from .. import _core, llm, stt, tts
from .._generated import public_voice as _public
from .._generated.agent_authoring import (
    DEFAULT_AGENT_NAME as DEFAULT_AGENT_NAME,
    CallbackKind,
    ToolOptions,
    ToolOptionsKwargs,
)
from .._generated.agent_run import AgentDefinition
from .._generated.public_voice import *  # noqa: F403
from .._generated.text_session import LocalAgentTool
from .._generated.vad import VAD
from ..inference.llm import llm_from_model_string
from ..llm import LLM, DuplexModel, DuplexRealtimeAdapter, RealtimeModel
from ..llm.chat_context import _ReadOnlyChatContext
from ..llm.tool_context import (
    FunctionTool,
    FunctionToolInfo,
    RawFunctionTool,
    ReadBack,
    Tool,
    ToolFlag,
    Toolset,
    function_tool,
)
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from . import io, room_io
from .events import (
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    Context_T,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    RunContext,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from .io import SpeechHandle

if TYPE_CHECKING:
    from .._generated.agent_run import RunFinished
    from .._generated.voice_session import VoiceCommandOrigin
    from ..inference import LLMModels, RealtimeModels
    from ..llm.tool_context import ToolHandlingOptions
    from .agent_run import AgentCallbacks, _TextRun
    from .events import RunContext
    from .run_result import RunResult, RunStream

    _LLMArg: TypeAlias = NotGivenOr[
        llm.LLM | llm.RealtimeModel | llm.DuplexModel | LLMModels | RealtimeModels | str | None
    ]


_P = ParamSpec("_P")
_R = TypeVar("_R")
_InstructionsFn = Callable[["RunContext"], str | Awaitable[str]]
_HookFn = Callable[["VoiceSession"], Any]
_OutputValidatorFn = Callable[["RunContext", Any], Any]
Output_T = TypeVar("Output_T", default=Any)
_Context = TypeVar("_Context")

# the voice session or text run executing the current agent callback
_BindingContextVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_agent_binding")


class _VoiceBinding(NamedTuple):
    session: VoiceSession
    agent: Agent
    origin: VoiceCommandOrigin | None = None
    # the agent runs a `VoiceTask`: a validated `Completed` from its tools ends the task
    task: bool = False

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
        return self.session.conn_options.llm_conn_options or DEFAULT_API_CONNECT_OPTIONS


_HookFnT = TypeVar("_HookFnT", bound=_HookFn)


def _assignment_namespace(frame: Any) -> dict[str, Any] | None:
    # Only a module (or exec) frame has a live namespace; a function frame's locals are a
    # snapshot and holding them would keep the frame alive. `Agent[Ctx](...)` adds a typing frame.
    while frame is not None and frame.f_globals.get("__name__") in ("typing", "typing_extensions"):
        frame = frame.f_back
    if frame is None or frame.f_locals is not frame.f_globals:
        return None
    namespace: dict[str, Any] = frame.f_locals
    return namespace


class Agent(Generic[Context_T, Output_T]):
    """Shared agent configuration, parameterized by application context and final output.

    ``Agent(context_type=Ctx, output_type=X)`` is an ``Agent[Ctx, X]``; the output is ``str``
    when ``output_type`` is omitted. ``context_type`` exists only for the type checker.

    ``tools`` are offered on every run, alongside a call's session tools; two different tools
    sharing a name raise ``ValueError`` when the agent runs.

    ``retries`` sets the retry budgets of every run (``AgentRetries``); an int sets both.
    ``tool_handling`` sets the templates rendered around background tool calls on a voice
    session (``ToolHandlingOptions``); an ``AsyncToolset`` overrides them for its own tools.
    """

    # The async output-function overload precedes the union one, which mypy cannot solve for it.
    @overload
    def __init__(
        self: Agent[Any, str],
        *,
        instructions: str,
        name: str | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: None = None,
        llm: _LLMArg = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: Agent[_Context, str],
        *,
        instructions: str,
        name: str | None = None,
        context_type: type[_Context],
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: None = None,
        llm: _LLMArg = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        instructions: str,
        name: str | None = None,
        context_type: type[Context_T] | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: Callable[..., Awaitable[Output_T]],
        llm: _LLMArg = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        instructions: str,
        name: str | None = None,
        context_type: type[Context_T] | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: type[Output_T] | Callable[..., Output_T],
        llm: _LLMArg = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        instructions: str,
        name: str | None = None,
        context_type: type[object] | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        output_type: type[Output_T]
        | Callable[..., Awaitable[Output_T]]
        | Callable[..., Output_T]
        | None = None,
        llm: _LLMArg = NOT_GIVEN,
        retries: int | AgentRetries | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None:
        if not isinstance(instructions, str):
            raise TypeError("instructions must be a string")
        tools = tools or []
        # an unnamed agent takes its module-level variable name, which exists only after the
        # constructor returns: keep the namespace and scan it when the name is first read
        self._config = _core._core_AgentConfig(instructions, name, retries)
        self._defining_namespace = _assignment_namespace(sys._getframe(1)) if name is None else None

        self._tools: dict[int, Tool | Toolset] = {}
        for tool in tools:
            token = self._config.register(CallbackKind.TOOL).token
            self._tools[token] = tool
        # filled by the decorators at import; frozen once the agent is first bound
        self._instruction_fns: dict[int, _InstructionsFn] = {}
        self._instructions_view: _InstructionsView[Any] | None = None
        self._hook_fns: dict[int, _HookFn] = {}
        self._output_validators: dict[int, _OutputValidatorFn] = {}
        self._tool_handling = tool_handling

        # a type, or an output function whose return value is the output
        self._output_type_explicit = output_type is not None
        self._output_spec: Any = output_type if output_type is not None else str
        self._output_type: type[Any] = _output_type_of(self._output_spec)
        self._host_instance: AgentCallbacks | None = None

        if isinstance(llm, str):
            llm = llm_from_model_string(llm)
        # a duplex model is wrapped on the way in, so nothing downstream sees one
        self._llm: NotGivenOr[LLM | RealtimeModel | None] = (
            DuplexRealtimeAdapter(llm) if isinstance(llm, DuplexModel) else llm
        )

    @property
    def name(self) -> str:
        """``name=``, else the module-level variable it is assigned to, else ``default_agent``."""
        discovered = None
        if self._defining_namespace is not None:
            for var_name, value in self._defining_namespace.items():
                if value is self:
                    discovered = var_name
                    self._defining_namespace = None
                    break
        return self._config.resolve_name(discovered)

    @property
    def instructions(self) -> _InstructionsView[Context_T]:
        """The static instructions, also the decorator registering dynamic ones.

        Reads as the static string the agent was built with. Used as
        ``@agent.instructions`` it registers a function rendered into the instructions on every
        turn, which receives the ``RunContext`` and returns a string, sync or async::

            @scheduler.instructions
            def known_appointment(ctx: RunContext) -> str: ...
        """
        if self._instructions_view is None:
            self._instructions_view = _InstructionsView(self._config.instructions, self)
        return self._instructions_view

    @property
    def tools(self) -> list[llm.Tool | llm.Toolset]:
        """The tools and toolsets available to the agent."""
        return [self._tools[token] for token in self._config.callbacks(CallbackKind.TOOL)]

    @property
    def chat_ctx(self) -> llm.ChatContext:
        """A read-only view of the agent's own history in the current run.

        Only available inside the agent's hooks and tools.

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
        """This agent as a tool running a typed child conversation inside a TextSession."""
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
        retries: int | AgentRetries | None = None,
    ) -> RunResult[Output_T]:
        """Run the model and tool loop until an accepted output or a request for human input.

        Args:
            message: The user's message.
            message_history: The conversation so far; omit for a fresh one.
            context: Application data and dependencies, read as ``ctx.context``.
            retries: The agent's retry budgets for this run (``AgentRetries``); an int sets both.
        """
        from .agent_run import door_hooks

        hooks = door_hooks()
        async with self.run_stream(
            message,
            message_history=message_history,
            context=context,
            retries=retries,
        ) as stream:
            if hooks is not None and hooks.on_token is not None:
                async for delta in stream.stream_text(delta=True):
                    await hooks.on_token(delta)
            return await stream.result()

    def run_stream(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        context: Context_T | None = None,
        retries: int | AgentRetries | None = None,
    ) -> RunStream[RunResult[Output_T]]:
        """``run()`` as an async context manager streaming text and events; it starts on entry::

        async with agent.run_stream(message, message_history=history) as stream:
            async for delta in stream.stream_text(delta=True):
                print(delta, end="")
            result = await stream.result()
        """
        from .agent_run import settled_result, start_run
        from .run_result import RunStream, _Started

        async def start(stream: _core._core_RunStream) -> _Started[RunResult[Output_T]]:
            run = await start_run(
                self,
                stream,
                message,
                message_history=message_history,
                context=context,
                retries=retries,
                conn_options=DEFAULT_API_CONNECT_OPTIONS,
            )

            def settle(finished: RunFinished | None) -> RunResult[Output_T]:
                assert finished is not None
                result = settled_result(finished, run.hooks)
                if run.hooks is not None and run.hooks.on_result is not None:
                    run.hooks.on_result(result)
                return result

            return _Started(settle=settle)

        return RunStream(start)

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
        token = self._config.register(CallbackKind.VALIDATOR).token
        self._output_validators[token] = f
        return f

    @property
    def _host(self) -> AgentCallbacks:
        if self._host_instance is None:
            from .agent_run import AgentCallbacks

            self._config.freeze()
            self._host_instance = AgentCallbacks(self)
        return self._host_instance

    def _definition(
        self, tools: llm.ToolContext, retries: int | AgentRetries | None = None
    ) -> AgentDefinition:
        from .agent_run import tool_infos

        _ = self.name  # resolve the discovered name before the configuration is read
        return AgentDefinition(
            **self._config.configuration(retries),
            tools=tool_infos(tools),
            output=self._host._output.spec(validated=self._config.validated_output),
        )

    async def _validate_output(self, ctx: RunContext, output: Any) -> Any:
        for token in self._config.callbacks(CallbackKind.VALIDATOR):
            validated = self._output_validators[token](ctx, output)
            if inspect.isawaitable(validated):
                validated = await validated
            output = validated
        return output

    def _check_not_frozen(self, what: str) -> None:
        _ = self.name
        self._config.check_mutable(what)

    @overload
    def tool(self, f: Callable[_P, _R], /) -> FunctionTool[_P, _R]: ...

    @overload
    def tool(
        self, f: None = None, /, **options: Unpack[ToolOptionsKwargs]
    ) -> Callable[[Callable[_P, _R]], FunctionTool[_P, _R] | RawFunctionTool[_P, _R]]: ...

    def tool(
        self, f: Callable[_P, _R] | None = None, /, **options: Unpack[ToolOptionsKwargs]
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
                # an already built tool keeps the options it was built with
                self._config.check_tool_options(ToolOptions(**options))
                tool = func
            else:
                # `function_tool` refuses what a raw schema cannot take (`needs_approval`)
                tool = function_tool(func, **options)
            token = self._config.register(CallbackKind.TOOL).token
            self._tools[token] = tool
            return tool

        return register(f) if f is not None else register

    def _register_instructions(self, f: _InstructionsFn) -> _InstructionsFn:
        self._check_not_frozen("an instructions function")
        token = self._config.register(CallbackKind.INSTRUCTIONS).token
        self._instruction_fns[token] = f
        return f

    async def _render_dynamic_instructions(self, ctx: RunContext) -> str:
        parts: list[str] = []
        for token in self._config.callbacks(CallbackKind.INSTRUCTIONS):
            rendered = self._instruction_fns[token](ctx)
            if inspect.isawaitable(rendered):
                rendered = await rendered
            if rendered:
                parts.append(str(rendered))
        return self._config.join_instructions(parts)

    def on_enter(self, f: _HookFnT) -> _HookFnT:
        """Register the function called when the agent takes the active live role, as a
        decorator. It receives the session::

            @scheduler.on_enter
            async def greet(session: VoiceSession) -> None:
                session.generate_reply(instructions="Greet the caller.")
        """
        self._register_hook("on_enter", f)
        return f

    def on_exit(self, f: _HookFnT) -> _HookFnT:
        """Register the function called when the agent relinquishes the active live role, on an
        orderly replacement or session close, as a decorator. It receives the session::

            @front_desk.on_exit
            async def leave(session: VoiceSession) -> None: ...
        """
        self._register_hook("on_exit", f)
        return f

    def _register_hook(self, name: str, f: _HookFn) -> _HookFn:
        self._check_not_frozen(f"an {name} function")
        registered = self._config.register(cast(CallbackKind, name))
        if registered.retired is not None:
            self._hook_fns.pop(registered.retired)
        self._hook_fns[registered.token] = f
        return f

    async def _run_hook(self, name: str) -> Any:
        token = self._config.hook(name)
        if token is None:
            return None
        result = self._hook_fns[token](self._get_session_or_raise())
        if inspect.isawaitable(result):
            result = await result
        return result

    def _has_on_enter(self) -> bool:
        return self._config.has_enter()

    def _bound_binding(self) -> _VoiceBinding | _TextRun[Any] | None:
        # per-context, so one shared instance serving several sessions reads the right one
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
        binding = self._get_binding_or_raise()
        if not isinstance(binding, _VoiceBinding):
            raise RuntimeError(
                f"agent `{self.name}` is running in agent.run(), which has no live session"
            )
        return binding.session

    @property
    def llm(self) -> NotGivenOr[llm.LLM | llm.RealtimeModel | None]:
        """The agent's LLM or RealtimeModel; when not given, the session's model is used."""
        return self._llm


def _output_type_of(output: Any) -> type[Any]:
    if isinstance(output, type) or not callable(output):
        return output  # type: ignore[no-any-return]
    from typing import get_type_hints

    return get_type_hints(output).get("return", Any)  # type: ignore[no-any-return]


_RenderedT = TypeVar("_RenderedT", bound=str | Awaitable[str])


class _InstructionsView(str, Generic[Context_T]):
    _agent: Agent

    def __new__(cls, source: str, agent: Agent) -> _InstructionsView[Context_T]:
        view = super().__new__(cls, source)
        view._agent = agent
        return view

    def __call__(
        self, f: Callable[[RunContext[Context_T]], _RenderedT]
    ) -> Callable[[RunContext[Context_T]], _RenderedT]:
        self._agent._register_instructions(f)
        return f


class _LocalAgentTool(FunctionTool[..., Awaitable[object]]):
    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        tool_name: str | None,
        tool_description: str | None,
        parameters: type[object] | None,
        needs_approval: bool | ReadBack,
        chat_history: Literal["fresh", "fork"],
        merge_chat_history: bool,
    ) -> None:
        if not agent._output_type_explicit:
            raise ValueError("an agent tool requires an explicitly declared output_type")
        if chat_history not in ("fresh", "fork"):
            raise ValueError('chat_history must be "fresh" or "fork"')
        self.agent = agent
        self.parameters = TypeAdapter(parameters) if parameters is not None else None
        self.schema = (
            self.parameters.json_schema()
            if self.parameters is not None
            else {"type": "object", "properties": {}, "additionalProperties": False}
        )
        if self.schema.get("type") != "object":
            raise ValueError("agent-tool parameters must describe an object")
        self.routing = LocalAgentTool(
            agent_name=agent.name,
            chat_history=chat_history,
            merge_chat_history=merge_chat_history,
        )

        async def invoke(**arguments: object) -> object:
            raise RuntimeError("a local agent tool must run inside a TextSession")

        super().__init__(
            invoke,
            FunctionToolInfo(
                name=tool_name or agent.name,
                flags=ToolFlag.NONE,
                description=tool_description or f"Run {agent.name}.",
                needs_approval=bool(needs_approval),
                read_back=needs_approval if callable(needs_approval) else None,
            ),
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.parameters is None:
            if arguments:
                raise ValueError("this agent tool accepts no parameters")
            return {}
        value = self.parameters.validate_python(arguments)
        encoded = to_jsonable_python(value)
        if not isinstance(encoded, dict):
            raise ValueError("validated agent-tool parameters must be an object")
        return encoded


# imported after Agent is defined: these modules import it
from .agent_session import (  # noqa: E402
    VoiceSession,
    VoiceTask,
    VoiceTaskFailed,
)
from .amd import AMD, AMDCategory, AMDPredictionEvent, DetectionOptions  # noqa: E402
from .background_audio import (  # noqa: E402
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    PlayHandle,
)

__all__ = [
    "VoiceSession",
    "VoiceTask",
    "VoiceTaskFailed",
    "Agent",
    "SpeechHandle",
    "RunContext",
    "UserInputTranscribedEvent",
    "ConversationItemAddedEvent",
    "SpeechCreatedEvent",
    "ErrorEvent",
    "CloseEvent",
    "CloseReason",
    "UserStateChangedEvent",
    "AgentStateChangedEvent",
    "FunctionToolsExecutedEvent",
    "AMD",
    "AMDCategory",
    "AMDPredictionEvent",
    "DetectionOptions",
    "BackgroundAudioPlayer",
    "AudioConfig",
    "BuiltinAudioClip",
    "PlayHandle",
    "io",
    "room_io",
]
__all__ += _public.__all__
