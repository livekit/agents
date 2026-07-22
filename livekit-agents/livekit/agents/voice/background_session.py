from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from docstring_parser import parse_from_object
from typing_extensions import TypedDict

from .. import llm, utils
from ..llm.chat_context import ChatMessage
from ..llm.utils import _is_valid_function_output
from ..log import logger
from .events import (
    BackgroundMessageReceived,
    BackgroundMessageUpdatedEvent,
    BackgroundReplyUpdated,
)
from .reply_scheduler import ReplyOptions, ReplyPromptArgs, ReplyStatus, _ReplyScheduler

if TYPE_CHECKING:
    from .agent_session import AgentSession


class BackgroundUpdatePromptArgs(TypedDict):
    background_session_id: str
    background_session_description: str
    message: str


class BackgroundReplyPromptArgs(TypedDict):
    background_session_id: str
    message_ids: list[str]


class BackgroundHandlingOptions(TypedDict, total=False):
    update_template: str | Callable[[BackgroundUpdatePromptArgs], str]
    reply_at_tail_template: str | Callable[[BackgroundReplyPromptArgs], str]
    reply_maybe_covered_template: str | Callable[[BackgroundReplyPromptArgs], str]


UPDATE_TEMPLATE = """Background session `{background_session_id}` sent an update:
{message}"""
REPLY_AT_TAIL_TEMPLATE = """New updates arrived from background session `{background_session_id}`.
Summarize the updates naturally. Do not repeat information you already told the user."""
REPLY_MAYBE_COVERED_TEMPLATE = """New updates arrived from background session `{background_session_id}`.
You may have already mentioned them in a recent reply. Respond only if there is new information
the user has not heard."""

_BACKGROUND_HANDLING_DEFAULTS: BackgroundHandlingOptions = {
    "update_template": UPDATE_TEMPLATE,
    "reply_at_tail_template": REPLY_AT_TAIL_TEMPLATE,
    "reply_maybe_covered_template": REPLY_MAYBE_COVERED_TEMPLATE,
}
_BACKGROUND_SEND_TOOL_NAME = "lk_background_send"
_BACKGROUND_STATE_TOOL_NAME = "lk_background_state"
_RESERVED_BACKGROUND_TOOL_NAMES = (_BACKGROUND_SEND_TOOL_NAME, _BACKGROUND_STATE_TOOL_NAME)

_PromptArgs = TypeVar("_PromptArgs")


def _copy_background_handling(
    options: Mapping[str, Any],
) -> BackgroundHandlingOptions:
    copied: BackgroundHandlingOptions = {}
    if "update_template" in options:
        copied["update_template"] = cast(
            str | Callable[[BackgroundUpdatePromptArgs], str],
            options["update_template"],
        )
    if "reply_at_tail_template" in options:
        copied["reply_at_tail_template"] = cast(
            str | Callable[[BackgroundReplyPromptArgs], str],
            options["reply_at_tail_template"],
        )
    if "reply_maybe_covered_template" in options:
        copied["reply_maybe_covered_template"] = cast(
            str | Callable[[BackgroundReplyPromptArgs], str],
            options["reply_maybe_covered_template"],
        )
    return copied


def _render(template: str | Callable[[_PromptArgs], str], args: _PromptArgs) -> str:
    if callable(template):
        return template(args)
    return template.format(**cast(dict[str, Any], args))


def _resolve_background_handling_options(
    definition_options: Mapping[str, Any] | None,
    session_options: BackgroundHandlingOptions | None,
) -> BackgroundHandlingOptions:
    selected = definition_options if definition_options is not None else session_options
    resolved = _copy_background_handling(_BACKGROUND_HANDLING_DEFAULTS)
    if selected is not None:
        resolved.update(_copy_background_handling(selected))
    return resolved


class BackgroundEntrypoint(Protocol):
    async def __call__(self, ctx: BackgroundContext) -> None: ...


def _validate_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("background name must be non-empty")
    return name


def _validate_description(description: object) -> str:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("background session description must be non-empty")
    return description


def _validate_metadata(id: object, description: object) -> None:
    _validate_name(id)
    _validate_description(description)


def _validate_entrypoint(entrypoint: object) -> None:
    is_async = inspect.iscoroutinefunction(entrypoint) or (
        callable(entrypoint) and inspect.iscoroutinefunction(type(entrypoint).__call__)
    )
    if not is_async:
        raise TypeError("background session entrypoint must be async")
    callable_entrypoint = cast(Callable[..., Any], entrypoint)

    try:
        parameters = list(inspect.signature(callable_entrypoint).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "background session entrypoint must accept exactly one context argument"
        ) from exc
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("background session entrypoint must accept exactly one context argument")


class BackgroundContext:
    def __init__(self, runtime: _BackgroundRuntime) -> None:
        self._runtime = runtime

    @property
    def id(self) -> str:
        return self._runtime.definition.id

    @property
    def description(self) -> str:
        return self._runtime.definition.description

    @property
    def session(self) -> AgentSession:
        return self._runtime.session

    def message_stream(self) -> AsyncIterable[str]:
        return self._runtime.message_stream()

    async def send(self, content: str, *, silent: bool = False) -> None:
        """Publish an update to the voice conversation.

        With ``silent=True`` the rendered message is inserted into the current
        Agent's chat context and the session history, but no reply is scheduled —
        context-only insertion the voice agent can draw on later.
        """
        await self._runtime.send(content, silent=silent)

    def set_state(self, state: Any) -> None:
        """Report the real-time state of this background session.

        The state is returned verbatim when the voice LLM calls the generated
        ``lk_background_state`` tool, so it must be a valid function-tool return
        value: a string, number, bool, None, or a JSON-serializable
        list/dict/tuple/set of those. Setting state never inserts context and
        never triggers a reply.
        """
        self._runtime.set_state(state)


@dataclass(frozen=True, init=False)
class BackgroundDefinition:
    id: str
    description: str
    entrypoint: BackgroundEntrypoint
    _background_handling: Mapping[str, Any] | None = field(repr=False)

    def __init__(
        self,
        *,
        id: str,
        description: str,
        background_handling: BackgroundHandlingOptions | None,
        entrypoint: BackgroundEntrypoint,
    ) -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "entrypoint", entrypoint)
        snapshot = (
            None if background_handling is None else MappingProxyType(dict(background_handling))
        )
        object.__setattr__(self, "_background_handling", snapshot)

    @property
    def background_handling(self) -> BackgroundHandlingOptions | None:
        if self._background_handling is None:
            return None
        return _copy_background_handling(self._background_handling)

    @property
    def _background_handling_snapshot(self) -> Mapping[str, Any] | None:
        return self._background_handling


class _BackgroundRuntime:
    def __init__(
        self,
        definition: BackgroundDefinition,
        *,
        session: AgentSession,
        background_handling: BackgroundHandlingOptions | None = None,
    ) -> None:
        self.definition = definition
        self._session = session
        self._handling = _resolve_background_handling_options(
            definition._background_handling_snapshot, background_handling
        )
        self._incoming: asyncio.Queue[str] = asyncio.Queue()
        self._state: Any = None
        self._closed = False
        self._task: asyncio.Task[None] | None = None
        self._started = asyncio.Event()
        self.context = BackgroundContext(self)
        self._reply_scheduler = _ReplyScheduler(
            reply_options=self._reply_options(),
            on_reply_scheduled=self._on_reply_scheduled,
            on_reply_done=self._on_reply_done,
        )

    @property
    def session(self) -> AgentSession:
        return self._session

    def start(self) -> None:
        if self._closed:
            raise RuntimeError(f"background session {self.definition.id!r} is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"background_{self.definition.id}")

    def enqueue(self, content: str) -> None:
        self._incoming.put_nowait(content)

    async def message_stream(self) -> AsyncIterable[str]:
        while True:
            yield await self._incoming.get()

    async def send(self, content: str, *, silent: bool = False) -> None:
        if self._closed:
            raise RuntimeError(f"background session {self.definition.id!r} is closed")

        message_id = utils.shortuuid("background_message_")
        update_args: BackgroundUpdatePromptArgs = {
            "background_session_id": self.definition.id,
            "background_session_description": self.definition.description,
            "message": content,
        }
        rendered = _render(self._handling["update_template"], update_args)
        message = ChatMessage(
            role="user",
            content=[rendered],
            extra={
                "background_session_id": self.definition.id,
                "background_message_id": message_id,
            },
        )
        if silent:
            accepted = await self._reply_scheduler.insert_only(
                session=self._session, items=[message]
            )
        else:
            accepted = await self._reply_scheduler.enqueue(
                session=self._session,
                items=[message],
                item_ids=[message_id],
                source={"background_session_id": self.definition.id},
            )
        if not accepted:
            raise RuntimeError(f"background session {self.definition.id!r} is closed")
        self._session.emit(
            "background_message_updated",
            BackgroundMessageUpdatedEvent(
                update=BackgroundMessageReceived(
                    background_id=self.definition.id,
                    message_id=message_id,
                    content=content,
                    silent=silent,
                )
            ),
        )

    def set_state(self, state: Any) -> None:
        if self._closed:
            raise RuntimeError(f"background session {self.definition.id!r} is closed")
        if not _is_valid_function_output(state):
            raise ValueError(
                "background state must be a valid function-tool return value: a string, "
                "number, bool, None, or a JSON-serializable list/dict/tuple/set of those"
            )
        self._state = state

    @property
    def state(self) -> Any:
        return self._state

    async def aclose(self) -> None:
        self._mark_closed()
        self._reply_scheduler._mark_closed()
        await self._cancel_entrypoint()
        await self._reply_scheduler.aclose()

    def _mark_closed(self) -> None:
        self._closed = True

    async def _cancel_entrypoint(self) -> None:
        if self._task is not None and not self._task.done():
            await utils.aio.cancel_and_wait(self._task)

    def _reply_options(self) -> ReplyOptions:
        def adapt(
            template: str | Callable[[BackgroundReplyPromptArgs], str],
        ) -> Callable[[ReplyPromptArgs], str]:
            def render(args: ReplyPromptArgs) -> str:
                reply_args: BackgroundReplyPromptArgs = {
                    "background_session_id": self.definition.id,
                    "message_ids": args["call_ids"],
                }
                return _render(template, reply_args)

            return render

        return {
            "reply_at_tail_template": adapt(self._handling["reply_at_tail_template"]),
            "reply_maybe_covered_template": adapt(self._handling["reply_maybe_covered_template"]),
        }

    def _on_reply_scheduled(
        self, session: AgentSession, message_ids: list[str], speech_id: str
    ) -> None:
        session.emit(
            "background_message_updated",
            BackgroundMessageUpdatedEvent(
                update=BackgroundReplyUpdated(
                    background_id=self.definition.id,
                    message_ids=message_ids,
                    status="scheduled",
                    speech_id=speech_id,
                )
            ),
        )

    def _on_reply_done(
        self,
        session: AgentSession,
        message_ids: list[str],
        speech_id: str,
        status: ReplyStatus,
    ) -> None:
        session.emit(
            "background_message_updated",
            BackgroundMessageUpdatedEvent(
                update=BackgroundReplyUpdated(
                    background_id=self.definition.id,
                    message_ids=message_ids,
                    status=status,
                    speech_id=speech_id,
                )
            ),
        )

    async def _run(self) -> None:
        self._started.set()
        try:
            await self.definition.entrypoint(self.context)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "background session entrypoint failed",
                extra={"background_session_id": self.definition.id},
            )


class _BackgroundRuntimeManager:
    def __init__(
        self,
        definitions: Sequence[BackgroundDefinition],
        *,
        session: AgentSession,
        background_handling: BackgroundHandlingOptions | None = None,
    ) -> None:
        for definition in definitions:
            if not isinstance(definition, BackgroundDefinition):
                raise TypeError("background sessions must be BackgroundDefinition instances")
            _validate_metadata(definition.id, definition.description)
            _validate_entrypoint(definition.entrypoint)

        ids = [definition.id for definition in definitions]
        duplicates = sorted({id for id in ids if ids.count(id) > 1})
        if duplicates:
            raise ValueError(f"duplicate background session ids: {', '.join(duplicates)}")

        self._definitions = tuple(definitions)
        self._session = session
        self._background_handling = (
            None if background_handling is None else _copy_background_handling(background_handling)
        )
        self._runtimes = self._create_runtimes()
        self._closed = False

    def _create_runtimes(self) -> dict[str, _BackgroundRuntime]:
        return {
            definition.id: _BackgroundRuntime(
                definition,
                session=self._session,
                background_handling=self._background_handling,
            )
            for definition in self._definitions
        }

    def start(self) -> None:
        if self._closed:
            self._runtimes = self._create_runtimes()
            self._closed = False
        for runtime in self._runtimes.values():
            runtime.start()

    async def wait_started(self) -> None:
        await asyncio.gather(*(runtime._started.wait() for runtime in self._runtimes.values()))

    def enqueue(self, background_session_id: str, content: str) -> None:
        if self._closed:
            raise RuntimeError("background runtime manager is closed")
        self._runtimes[background_session_id].enqueue(content)

    def get_state(self, background_session_id: str) -> Any:
        return self._runtimes[background_session_id].state

    async def aclose(self) -> None:
        self._closed = True
        runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            runtime._mark_closed()
            runtime._reply_scheduler._mark_closed()
        await asyncio.gather(
            *(runtime._cancel_entrypoint() for runtime in runtimes),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(runtime._reply_scheduler.aclose() for runtime in runtimes),
            return_exceptions=True,
        )


def _create_background_send_tool(
    manager: _BackgroundRuntimeManager,
    definitions: Sequence[BackgroundDefinition],
) -> llm.Tool:
    ordered = sorted(definitions, key=lambda definition: definition.id)
    valid_ids = [definition.id for definition in ordered]
    description = (
        "Send a message to a background session. Delivery is asynchronous and does not wait for "
        "a response.\n\nAvailable background sessions:\n"
        + "\n".join(f"- {definition.id}: {definition.description}" for definition in ordered)
    )

    async def lk_background_send(background_session_id: str, content: str) -> str:
        if background_session_id not in valid_ids:
            raise llm.ToolError(
                f"Unknown background session ID {background_session_id!r}. "
                f"Valid IDs: {', '.join(valid_ids)}"
            )
        manager.enqueue(background_session_id, content)
        return "Message has been delivered."

    return llm.function_tool(name=_BACKGROUND_SEND_TOOL_NAME, description=description)(
        lk_background_send
    )


def _create_background_state_tool(
    manager: _BackgroundRuntimeManager,
    definitions: Sequence[BackgroundDefinition],
) -> llm.Tool:
    ordered = sorted(definitions, key=lambda definition: definition.id)
    valid_ids = [definition.id for definition in ordered]
    description = (
        "Get the real-time state reported by a background session — what it is working on "
        "right now. Use this when the user asks about the progress or status of background "
        "work.\n\nAvailable background sessions:\n"
        + "\n".join(f"- {definition.id}: {definition.description}" for definition in ordered)
    )

    async def lk_background_state(background_session_id: str) -> Any:
        if background_session_id not in valid_ids:
            raise llm.ToolError(
                f"Unknown background session ID {background_session_id!r}. "
                f"Valid IDs: {', '.join(valid_ids)}"
            )
        state = manager.get_state(background_session_id)
        if state is None:
            return "The background session has not reported any state yet."
        return state

    return llm.function_tool(name=_BACKGROUND_STATE_TOOL_NAME, description=description)(
        lk_background_state
    )


def background(
    *,
    name: str,
    description: str | None = None,
    background_handling: BackgroundHandlingOptions | None = None,
) -> Callable[[BackgroundEntrypoint], BackgroundDefinition]:
    internal_id = _validate_name(name)

    def decorator(entrypoint: BackgroundEntrypoint) -> BackgroundDefinition:
        _validate_entrypoint(entrypoint)
        resolved_description = _validate_description(
            description or parse_from_object(entrypoint).description
        )

        return BackgroundDefinition(
            id=internal_id,
            description=resolved_description,
            background_handling=background_handling,
            entrypoint=entrypoint,
        )

    return decorator
