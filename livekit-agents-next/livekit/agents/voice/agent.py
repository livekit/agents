from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import inspect
import sys
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
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
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import TypeAdapter
from typing_extensions import ParamSpec, TypeVar

from .. import _core, inference, llm, stt, tts, vad
from .._generated.tool_executor import ToolHandlingOptions
from ..llm import (
    LLM,
    DuplexModel,
    DuplexRealtimeAdapter,
    RealtimeModel,
    ToolError,
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
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given, misc
from .events import Deps_T, State_T, UserTurnExceededEvent
from .turn import TurnHandlingOptions

if TYPE_CHECKING:
    from ..inference import LLMModels, RealtimeModels, STTModels, TTSModels
    from ..llm import mcp
    from .agent_run import AgentRun
    from .agent_session import ExpressiveOptions, LiveSession
    from .events import RunContext
    from .run_result import RunResult, RunUsage
    from .turn import TurnDetectionMode


_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])
_InstructionsFn = Callable[["RunContext"], str | Awaitable[str]]
_InstructionsFnT = TypeVar("_InstructionsFnT", bound=Callable[..., str | Awaitable[str]])
_OnEnterFn = Callable[["LiveSession"], Any]
_OutputValidatorFn = Callable[["RunContext", Any], Any]
Output_T = TypeVar("Output_T", default=Any)

DEFAULT_AGENT_ID = "default_agent"

# what runs the agents of this context: a voice session, or a text run
_BindingContextVar: contextvars.ContextVar[Any] = contextvars.ContextVar("lk_agent_binding")


class _VoiceBinding(NamedTuple):
    """The voice session running ``agent``'s hook or tool, and what its commands are sent as."""

    session: LiveSession
    agent: Agent
    origin: str | None = None
    # the core's id of the speech whose tool is running
    speech_id: int | None = None


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


class Agent(Generic[Deps_T, State_T, Output_T]):
    """``Agent[Deps, State, Output]``, read from ``deps_type``, ``state_type`` and
    ``output_type``; a parameter left out is ``Any``."""

    def __init__(
        self,
        *,
        instructions: str | Instructions,
        name: str | None = None,
        id: str | None = None,
        tools: list[llm.Tool | llm.Toolset] | None = None,
        deps_type: type[Deps_T] | None = None,
        state_type: type[State_T] | None = None,
        output_type: type[Output_T] | None = None,
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
            _core.migrate_turn_handling(
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
        self._instructions_view: _InstructionsView[Any, Any] | None = None
        self._on_enter_fn: _OnEnterFn | None = None
        self._output_validators: list[_OutputValidatorFn] = []
        self._frozen = False
        self._turn_detection = turn_handling.get("turn_detection", NOT_GIVEN)

        # the three type parameters of Agent[Deps, State, Output]; declared for the type
        # checker and the output parser, the values travel with the run
        self._deps_type: type[Deps_T] | None = deps_type
        self._state_type: type[State_T] | None = state_type
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
    def instructions(self) -> _InstructionsView[Deps_T, State_T]:
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
        binding = self._get_binding_or_raise()
        return binding.session.history if isinstance(binding, _VoiceBinding) else binding.chat_ctx

    @property
    def interruption_detection(self) -> NotGivenOr[Literal["adaptive", "vad"]]:
        return self._interruption_detection

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
        session = self._bound_session()
        if session is not None:
            given = {"stt": stt, "vad": vad, "llm": llm, "tts": tts}
            session._running.update_models(**{k: v for k, v in given.items() if is_given(v)})

    @property
    def deps_type(self) -> type[Deps_T] | None:
        return self._deps_type

    @property
    def state_type(self) -> type[State_T] | None:
        return self._state_type

    @property
    def output_type(self) -> type[Output_T]:
        """What ``run()`` returns as ``result.output``: ``str`` unless declared."""
        return self._output_type

    async def run(
        self,
        message: str | llm.ChatMessage,
        *,
        message_history: llm.ChatContext | Sequence[llm.ChatItem] | None = None,
        state: State_T | None = None,
        deps: Deps_T | None = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> RunResult[Output_T]:
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
        state: State_T | None = None,
        deps: Deps_T | None = None,
        usage: RunUsage | None = None,
        max_steps: int | None = None,
    ) -> AgentRun[Output_T]:
        """The turn of ``run()`` as an async iterator of the reply's text, one piece as the
        model writes it; ``run.result`` is the ``RunResult`` once the loop ends::

            run = agent.stream("book me in", message_history=history)
            async for delta in run:
                print(delta, end="")
            answer = run.result.output
        """
        from .agent_run import AgentRun

        return AgentRun[Output_T](
            self,
            message,
            message_history=message_history,
            state=state,
            deps=deps,
            usage=usage,
            max_steps=max_steps,
            output_type=self._output_type,
        )

    def output_validator(
        self,
        f: Callable[[RunContext[Deps_T, State_T], Output_T], Output_T | Awaitable[Output_T]],
    ) -> Callable[[RunContext[Deps_T, State_T], Output_T], Output_T | Awaitable[Output_T]]:
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

    def _bound_binding(self) -> _VoiceBinding | AgentRun[Any] | None:
        """What runs this agent in the current context: a voice session or a run (internal).

        Resolved through the contextvar both set around every hook and tool, so one shared
        instance serving several sessions reads the right one.
        """
        binding = _BindingContextVar.get(None)
        if binding is None or binding.agent is not self:
            return None
        return binding  # type: ignore[no-any-return]

    def _bound_session(self) -> LiveSession | None:
        """The voice session running this agent in the current context, if any (internal)."""
        binding = self._bound_binding()
        return binding.session if isinstance(binding, _VoiceBinding) else None

    def _get_binding_or_raise(self) -> _VoiceBinding | AgentRun[Any]:
        binding = self._bound_binding()
        if binding is None:
            raise RuntimeError(
                f"agent `{self.id}` is not running in this context; its session and history "
                "are only available inside its hooks and tools"
            )
        return binding

    def _get_session_or_raise(self) -> LiveSession:
        binding = self._get_binding_or_raise()
        if not isinstance(binding, _VoiceBinding):
            raise RuntimeError(
                f"agent `{self.id}` is running in agent.run(), which has no live session"
            )
        return binding.session

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

        if (session := self._bound_session()) is not None:
            session.update_options(turn_detection=value)

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
        return self._get_session_or_raise()


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
            _core.migrate_turn_handling(
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
        self._preserve_function_call_history = preserve_function_call_history
        # the core's handle while the task runs on a call
        self._handle: _core.TaskHandle | None = None

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
        from .._core_view import tool_specs
        from .tool_executor import ToolRunner

        self._tools = list({tool.id: tool for tool in tools}.values())
        if (session := self._bound_session()) is not None:
            pool = session._pool(self)
            session._running.update_tools(
                tool_specs(pool), tools_host=pool, tool_runner=ToolRunner(session, self, pool)
            )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel(self.id)
        if not self.__fut.done():
            self._complete(ToolError(f"AgentTask {self.id} is cancelled"))

    def _complete(self, result: TaskResult_T | Exception) -> None:
        """End the task with its result (internal): a tool returns ``Completed(value)``."""
        if self.__fut.done():
            raise RuntimeError(f"{self.__class__.__name__} is already done")
        if isinstance(result, Exception):
            self.__fut.set_exception(result)
            self.__fut.exception()  # silence exc not retrieved warnings
        else:
            self.__fut.set_result(result)
        if self._handle is not None and not self._handle.done:
            if isinstance(result, Exception):
                self._handle.fail(str(result), class_name=type(result).__name__)
            else:
                self._handle.complete(result)

    async def __await_impl(self) -> TaskResult_T:
        if self.__started:
            raise RuntimeError(f"{self.__class__.__name__} is not re-entrant, await only once")
        self.__started = True

        binding = _BindingContextVar.get(None)
        if binding is not None and not isinstance(binding, _VoiceBinding):
            # in text the run takes it from here: the call stays pending, the task is pushed on
            # the lane's agent stack, and the value comes back on a later turn
            raise TaskAwaited(self)
        if binding is None:
            raise RuntimeError(
                f"{self.__class__.__name__} should only be awaited inside tool_functions or the "
                "on_enter/on_exit methods of an Agent"
            )

        session = binding.session
        self._handle = handle = _core.TaskHandle()
        session._running.enter_task(
            handle,
            **await session._frame(self),
            speech_id=binding.speech_id,
            preserve_function_call_history=self._preserve_function_call_history,
            parent=binding.agent.id,
            origin=binding.origin,
        )
        session._agent = self
        try:
            await handle.joined()
        except Exception:
            # a failure is the one `_complete` set; a close leaves the task unsettled
            pass
        finally:
            handle.abandon()
            session._agent = binding.agent
        if not self.__fut.done():
            self._complete(ToolError(f"AgentTask {self.id} ended before it completed"))
        return await self.__fut

    def __await__(self) -> Generator[None, None, TaskResult_T]:
        return self.__await_impl().__await__()


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

    class _InstructionsView(str, Instructions, Generic[Deps_T, State_T]):
        """What a type checker sees: the value, read as a ``str`` or an ``Instructions``, and
        the decorator, which takes the agent's ``RunContext``. One class, since a union of
        callables is an untyped decorator to mypy."""

        _source: str | Instructions
        _agent: Agent

        def __call__(
            self, f: Callable[[RunContext[Deps_T, State_T]], _RenderedT]
        ) -> Callable[[RunContext[Deps_T, State_T]], _RenderedT]: ...

else:
    _InstructionsView = _InstructionsText | _InstructionsObject


def _wrap_instructions(source: str | Instructions, agent: Agent) -> _InstructionsView[Any, Any]:
    view: _InstructionsText | _InstructionsObject
    if isinstance(source, Instructions):
        view = _InstructionsObject(source.common, audio=source.audio, text=source.text)
    else:
        view = _InstructionsText(source)
    view._source = source
    view._agent = agent
    return cast("_InstructionsView[Any, Any]", view)
